"""
Composite team rankings — blending profile sub-scores with ELO.

Produces tier labels, projected wins, and overall rankings for all 30
MLB teams.  Pythagorean W% with exponent 1.83 derives projected wins
from run differential.

Also provides projection-integrated *power rankings* that blend
roster-adjusted ELO, 2026 player projections, and team profile scores.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path("C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Composite weights (sum to 1.0)
# ---------------------------------------------------------------------------
# Scouting grade component weights (sum to 0.55 — the talent portion)
_TALENT_WEIGHTS = {
    # Walk-forward validated 2022-2025 (team_ranking_validation.py).
    # Rotation and defense were massively underweighted; offense overweighted.
    # Depth signal is real but noisy — keeps a modest weight.
    "offense": 0.25,     # Lineup diamond (starters + bench)
    "rotation": 0.30,    # Top 5 SP diamond — strongest validated signal
    "bullpen": 0.12,     # Role-weighted RP diamond
    "defense": 0.18,     # OAA + framing
    "depth": 0.15,       # Health/depth + roster concentration/gaps
}
# Performance + projection signal (sum to 0.45)
# Projections (BaseRuns wins) are the stronger guardrail (r=0.814 residual
# after talent). ELO adds moderate momentum signal (r=0.399) but offense
# ELO is nearly useless (r=0.066). Projections get more weight.
_ELO_WEIGHT = 0.15       # Recent game performance — light momentum/culture signal
_PROJ_WEIGHT = 0.30       # Pythagorean projected wins (BaseRuns) — primary guardrail

# Tier thresholds on projected wins (wins-based, not composite score)
_TIERS = [
    (90, "Elite"),
    (85, "Contender"),
    (78, "Average"),
    (70, "Fringe"),
    (0, "Rebuilding"),
]

PYTH_EXP = 1.83  # Fallback static exponent (used if RPG unavailable)


def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1)."""
    return series.rank(pct=True, method="average")


def _assign_tier(projected_wins: float) -> str:
    """Map projected wins to tier label."""
    for threshold, label in _TIERS:
        if projected_wins >= threshold:
            return label
    return "Rebuilding"


def _pythagorean_wins(
    rpg: float,
    ra_per_game: float,
    games: int = 162,
    exp: float | None = None,
) -> float:
    """Projected wins via PythagenPat (dynamic exponent).

    PythagenPat sets the exponent based on the run environment:
    ``exp = ((RS + RA) / G) ^ 0.287``.  In higher-scoring environments
    the exponent is larger (more regression to .500); in lower-scoring
    it is smaller (extreme teams separate more).  Falls back to the
    static 1.83 exponent if RPG data is unavailable.

    Reference: Davenport & Woolner (Baseball Prospectus), Smyth/Patriot.
    """
    if rpg <= 0 or ra_per_game <= 0:
        return games / 2
    if exp is None:
        # PythagenPat: dynamic exponent from run environment
        total_rpg = rpg + ra_per_game
        exp = total_rpg ** 0.287
    wpct = rpg ** exp / (rpg ** exp + ra_per_game ** exp)
    return round(wpct * games, 1)


def compute_roster_waa_delta(
    prior_season: int,
    current_season: int,
    current_roster: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute WAA delta from roster turnover between seasons.

    For each team, calculates the difference in WAA between arriving
    and departing players.  Uses simplified WAA:
    - Hitters: (wRC+ - 100) / 100 * PA * 0.12 / 10
    - Pitchers: (lgRA9 - playerRA9) * IP / 9 / 10

    WAA values come from prior-season stats.  Team assignments for
    the current season come from ``current_roster`` when provided
    (essential early in a new season when game logs are sparse),
    falling back to game-log appearances otherwise.

    Parameters
    ----------
    prior_season : int
        Season whose rosters/stats form the baseline.
    current_season : int
        Season whose rosters define the new team composition.
    current_roster : pd.DataFrame, optional
        Active roster with ``player_id`` and ``org_id`` (team_id) columns.
        Typically ``roster.parquet``.  If None, falls back to game logs.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, waa_delta.
    """
    from src.data.db import read_sql

    # ── Prior season: player WAA + team from game logs ──
    hitters = read_sql("""
        WITH player_teams AS (
            SELECT player_id, season, team_id, COUNT(*) as games,
                   ROW_NUMBER() OVER (
                       PARTITION BY player_id, season
                       ORDER BY COUNT(*) DESC
                   ) as rn
            FROM production.fact_player_game_mlb
            WHERE season = :s1
            GROUP BY player_id, season, team_id
        )
        SELECT b.batter_id as player_id, b.pa, b.wrc_plus, pt.team_id
        FROM production.fact_batting_advanced b
        JOIN player_teams pt
            ON b.batter_id = pt.player_id AND b.season = pt.season AND pt.rn = 1
        WHERE b.season = :s1 AND b.pa >= 50
    """, {"s1": prior_season})
    hitters["waa"] = (hitters["wrc_plus"] - 100) / 100 * hitters["pa"] * 0.12 / 10

    pitchers = read_sql("""
        WITH player_teams AS (
            SELECT player_id, season, team_id, COUNT(*) as games,
                   ROW_NUMBER() OVER (
                       PARTITION BY player_id, season
                       ORDER BY COUNT(*) DESC
                   ) as rn
            FROM production.fact_player_game_mlb
            WHERE season = :s1 AND pit_ip > 0
            GROUP BY player_id, season, team_id
        ),
        pitcher_stats AS (
            SELECT player_id, season, SUM(pit_ip) as ip, SUM(pit_r) as runs
            FROM production.fact_player_game_mlb
            WHERE season = :s1 AND pit_ip > 0
            GROUP BY player_id, season
            HAVING SUM(pit_ip) >= 20
        )
        SELECT ps.player_id, ps.ip, ps.runs, pt.team_id
        FROM pitcher_stats ps
        JOIN player_teams pt
            ON ps.player_id = pt.player_id AND ps.season = pt.season AND pt.rn = 1
    """, {"s1": prior_season})

    if not pitchers.empty:
        lg_ra9 = pitchers["runs"].sum() / pitchers["ip"].sum() * 9
        pitchers["ra9"] = pitchers["runs"] / pitchers["ip"] * 9
        pitchers["waa"] = (lg_ra9 - pitchers["ra9"]) * pitchers["ip"] / 9 / 10

    prior_players = pd.concat([
        hitters[["player_id", "team_id", "waa"]],
        pitchers[["player_id", "team_id", "waa"]]
        if not pitchers.empty else pd.DataFrame(),
    ], ignore_index=True)

    # Build WAA lookup: player_id -> prior season WAA (regardless of team)
    waa_lookup = prior_players.groupby("player_id")["waa"].sum().to_dict()

    # ── Current season: team assignments ──
    if current_roster is not None and not current_roster.empty:
        # Use roster parquet (org_id = team_id)
        cur_teams = current_roster[["player_id", "org_id"]].rename(
            columns={"org_id": "team_id"}
        ).drop_duplicates()
    else:
        # Fall back to game logs (works mid-season)
        cur_teams = read_sql("""
            SELECT player_id, team_id
            FROM (
                SELECT player_id, team_id, COUNT(*) as games,
                       ROW_NUMBER() OVER (
                           PARTITION BY player_id
                           ORDER BY COUNT(*) DESC
                       ) as rn
                FROM production.fact_player_game_mlb
                WHERE season = :s2
                GROUP BY player_id, team_id
            ) sub WHERE rn = 1
        """, {"s2": current_season})

    # ── Compare rosters ──
    all_teams = set(prior_players["team_id"].unique()) | set(cur_teams["team_id"].unique())
    rows = []
    for tid in all_teams:
        prior_pids = set(prior_players[prior_players["team_id"] == tid]["player_id"])
        cur_pids = set(cur_teams[cur_teams["team_id"] == tid]["player_id"])

        departed = prior_pids - cur_pids
        arrived = cur_pids - prior_pids

        # Departed WAA: what did the team lose (from prior season stats)
        dep_waa = sum(waa_lookup.get(pid, 0) for pid in departed)
        # Arrived WAA: what did the team gain (from prior season stats)
        arr_waa = sum(waa_lookup.get(pid, 0) for pid in arrived)

        rows.append({"team_id": tid, "waa_delta": arr_waa - dep_waa})

    return pd.DataFrame(rows)


def rank_teams(
    profiles: pd.DataFrame,
    elo_ratings: pd.DataFrame | None = None,
    waa_deltas: pd.DataFrame | None = None,
    observed_rs_ra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Produce final team rankings from profiles and ELO.

    Parameters
    ----------
    profiles : pd.DataFrame
        Output of ``build_all_team_profiles()``.
    elo_ratings : pd.DataFrame, optional
        Output of ``get_current_ratings()``.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, abbreviation, team_name, division, league,
        composite_score, tier, projected_wins, offense_score, pitching_score,
        defense_score, health_depth_score, elo_composite, elo_rank, rank.
    """
    df = profiles.copy()

    # Ensure score columns exist with fallback
    for col, default in [
        ("offense_score", 0.5),
        ("rotation_score", 0.5),
        ("bullpen_score", 0.5),
        ("pitching_score", 0.5),
        ("defense_score", 0.5),
        ("health_depth_score", 0.5),
    ]:
        if col not in df.columns:
            df[col] = default

    # Fill NaN scores with median
    for col in ["offense_score", "rotation_score", "bullpen_score",
                "pitching_score", "defense_score", "health_depth_score"]:
        df[col] = df[col].fillna(df[col].median())

    # ── Projected wins (Pythagorean from blended RS/RA) ──
    # Blend projected (Bayesian posteriors, roster-aware) with observed (prior
    # season actual, 2H-weighted) to get realistic spread.  Projected alone
    # is too compressed; observed alone ignores roster changes.
    # 35/65 produces ~42-win spread (matches PECOTA/ZiPS).
    # Observed RS/RA uses 40/60 1H/2H weighting when available to capture
    # trade-deadline roster transformations.
    _PROJ_WT = 0.35  # projected (forward-looking, roster-aware)
    _OBS_WT = 0.65   # observed (backward-looking, more spread)
    has_proj = "proj_rs_per_game" in df.columns and "proj_ra_per_game" in df.columns
    has_obs = "rpg" in df.columns and "ra_per_game" in df.columns

    # Override observed RS/RA with 2H-weighted values when provided
    if observed_rs_ra is not None and not observed_rs_ra.empty:
        obs_map_rs = dict(zip(observed_rs_ra["team_id"], observed_rs_ra["rpg"]))
        obs_map_ra = dict(zip(observed_rs_ra["team_id"], observed_rs_ra["ra_per_game"]))
        df["_obs_rpg"] = df["team_id"].map(obs_map_rs).fillna(df.get("rpg", 4.5))
        df["_obs_ra"] = df["team_id"].map(obs_map_ra).fillna(df.get("ra_per_game", 4.5))
    else:
        df["_obs_rpg"] = df["rpg"].fillna(4.5) if "rpg" in df.columns else 4.5
        df["_obs_ra"] = df["ra_per_game"].fillna(4.5) if "ra_per_game" in df.columns else 4.5

    if has_proj:
        df["_rs_blend"] = (
            _PROJ_WT * df["proj_rs_per_game"].fillna(4.5)
            + _OBS_WT * df["_obs_rpg"]
        )
        df["_ra_blend"] = (
            _PROJ_WT * df["proj_ra_per_game"].fillna(4.5)
            + _OBS_WT * df["_obs_ra"]
        )
    elif has_obs:
        df["_rs_blend"] = df["_obs_rpg"]
        df["_ra_blend"] = df["_obs_ra"]
    else:
        df["_rs_blend"] = 4.5
        df["_ra_blend"] = 4.5

    df["projected_wins"] = df.apply(
        lambda r: _pythagorean_wins(r["_rs_blend"], r["_ra_blend"]),
        axis=1,
    )
    df.drop(columns=["_rs_blend", "_ra_blend", "_obs_rpg", "_obs_ra"],
            inplace=True, errors="ignore")

    # ── Roster WAA adjustment ──
    # Adjust projected wins by dampened WAA delta from roster turnover.
    # Prior-year WAA doesn't fully repeat, so apply 55% dampening.
    # Cap at ±8 wins to prevent extreme outliers (e.g. COL) from
    # distorting the entire league distribution after normalization.
    # Backtested 2022-2025: drops MAE from 9.0 to ~7.5 wins.
    _WAA_DAMPEN = 0.55
    _WAA_CAP = 8.0
    if waa_deltas is not None and not waa_deltas.empty:
        waa_map = dict(zip(waa_deltas["team_id"], waa_deltas["waa_delta"]))
        df["projected_wins"] = df.apply(
            lambda r: r["projected_wins"] + max(-_WAA_CAP, min(
                _WAA_CAP, waa_map.get(r["team_id"], 0) * _WAA_DAMPEN,
            )),
            axis=1,
        )

    # Normalize projected wins to sum to 2430
    _raw_total = df["projected_wins"].sum()
    if _raw_total > 0 and abs(_raw_total - 2430) > 1:
        df["projected_wins"] = (df["projected_wins"] * 2430 / _raw_total).round(1)

    # Schedule-adjusted wins
    if "avg_opp_elo" in df.columns:
        elo_diff = df["avg_opp_elo"].fillna(1500) - 1500
        schedule_adjustment = -elo_diff * 0.10
        df["schedule_adjusted_wins"] = (df["projected_wins"] + schedule_adjustment).round(1)
    else:
        df["schedule_adjusted_wins"] = df["projected_wins"]

    # ── Merge ELO ──
    if elo_ratings is not None and not elo_ratings.empty:
        elo_cols = ["team_id", "composite_elo", "composite_rank",
                    "offense_elo", "pitching_elo"]
        elo_available = [c for c in elo_cols if c in elo_ratings.columns]
        df = df.merge(elo_ratings[elo_available], on="team_id", how="left")
        if "composite_rank" in df.columns:
            df.rename(columns={"composite_rank": "elo_rank"}, inplace=True)
    else:
        df["composite_elo"] = None
        df["elo_rank"] = None

    # ── Composite score: talent (55%) + ELO (20%) + projections (25%) ──
    # Talent component: scouting diamond ratings when available, else old sub-scores
    has_diamonds = all(
        c in df.columns and df[c].notna().any()
        for c in ["lineup_diamond", "rotation_diamond", "bullpen_diamond"]
    )
    if has_diamonds:
        talent = (
            _TALENT_WEIGHTS["offense"] * (df["lineup_diamond"].fillna(5.0) / 10.0)
            + _TALENT_WEIGHTS["rotation"] * (df["rotation_diamond"].fillna(5.0) / 10.0)
            + _TALENT_WEIGHTS["bullpen"] * (df["bullpen_diamond"].fillna(5.0) / 10.0)
            + _TALENT_WEIGHTS["defense"] * df["defense_score"]
            + _TALENT_WEIGHTS["depth"] * df["health_depth_score"]
        )
    else:
        talent = (
            _TALENT_WEIGHTS["offense"] * df["offense_score"]
            + _TALENT_WEIGHTS["rotation"] * df["rotation_score"]
            + _TALENT_WEIGHTS["bullpen"] * df["bullpen_score"]
            + _TALENT_WEIGHTS["defense"] * df["defense_score"]
            + _TALENT_WEIGHTS["depth"] * df["health_depth_score"]
        )

    # ELO component: normalize composite_elo to 0-1 (1400-1600 range)
    if "composite_elo" in df.columns and df["composite_elo"].notna().any():
        elo_norm = ((df["composite_elo"].fillna(1500) - 1400) / 200).clip(0, 1)
        elo_component = _ELO_WEIGHT * elo_norm
    else:
        # No ELO: redistribute weight to talent
        elo_component = 0.0
        talent = talent * (1.0 + _ELO_WEIGHT / sum(_TALENT_WEIGHTS.values()))

    # Projection component: normalize projected wins to 0-1 (70-110 range)
    proj_norm = ((df["projected_wins"].fillna(81) - 70) / 40).clip(0, 1)
    proj_component = _PROJ_WEIGHT * proj_norm

    raw_composite = talent + elo_component + proj_component
    df["composite_score"] = raw_composite  # keep raw for backward compat

    # Map composite to TDD Score (1-10). Rescale so the actual worst
    # team = 1.0 and best team = 10.0 — relative to the league this year.
    raw_min = raw_composite.min()
    raw_max = raw_composite.max()
    if raw_max - raw_min > 1e-9:
        df["tdd_score"] = (1.0 + (raw_composite - raw_min) / (raw_max - raw_min) * 9.0).round(1)
    else:
        df["tdd_score"] = 5.5

    # Tier assignment (on 0-10 TDD score)
    df["tier"] = df["projected_wins"].apply(_assign_tier)

    # Final rank by tdd_score
    # Use composite_score (continuous) to break TDD score ties
    df["rank"] = df["composite_score"].rank(ascending=False, method="first").astype(int)

    # Sort and select output columns
    output_cols = [
        "team_id", "abbreviation", "team_name", "division", "league",
        "rank", "tdd_score", "composite_score", "tier",
        "projected_wins", "schedule_adjusted_wins",
        "offense_score", "pitching_score", "rotation_score", "bullpen_score",
        "defense_score", "health_depth_score",
    ]
    # Add optional enrichment columns
    optional = [
        "offense_ceiling", "pitching_ceiling",
        "offense_style", "pitching_style", "age_trajectory",
        "rpg", "ra_per_game", "hr_per_game", "k_per_game",
        "rotation_strength", "rotation_strength_ceiling",
        "bullpen_depth", "lineup_depth",
        "team_oaa", "catcher_framing_runs",
        "pct_homegrown", "farm_avg_score", "n_ranked_prospects", "top_100_count",
        "total_il_days", "avg_age",
        "composite_elo", "elo_rank", "offense_elo", "pitching_elo",
        "projected_wins",
        # Scouting grade team diamonds
        "lineup_diamond", "rotation_diamond", "bullpen_diamond",
        "lineup_grade_hit", "lineup_grade_power", "lineup_grade_speed",
        "lineup_grade_fielding", "lineup_grade_discipline",
        "rotation_grade_stuff", "rotation_grade_command", "rotation_grade_durability",
        "bullpen_grade_stuff", "bullpen_grade_command",
        "team_fielding_grade",
    ]
    for col in optional:
        if col in df.columns and col not in output_cols:
            output_cols.append(col)

    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].sort_values("rank").reset_index(drop=True)

    logger.info(
        "Team rankings: %d teams | Tiers: %s",
        len(result),
        result["tier"].value_counts().to_dict(),
    )
    return result


# ---------------------------------------------------------------------------
# Config helpers for power rankings
# ---------------------------------------------------------------------------
def _load_power_config() -> dict[str, Any]:
    """Load power_rankings config block from model.yaml.

    Also pulls ``game_elo_blend`` / ``series_elo_blend`` from the
    ``series_elo`` section so the power rankings can blend game and
    series ELO components.
    """
    cfg_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    result = cfg.get("power_rankings", {})
    # Pull blend weights from series_elo section
    series_cfg = cfg.get("series_elo", {})
    if "game_elo_blend" in series_cfg:
        result.setdefault("game_elo_blend", series_cfg["game_elo_blend"])
    if "series_elo_blend" in series_cfg:
        result.setdefault("series_elo_blend", series_cfg["series_elo_blend"])
    return result


_POWER_DEFAULTS: dict[str, Any] = {
    "elo_weight": 0.40,
    "projection_weight": 0.20,
    "profile_weight": 0.25,
    "glicko_weight": 0.15,
    "breakout_delta_threshold": 0.03,
    "regression_delta_threshold": -0.03,
}


def _get_power_config() -> dict[str, Any]:
    """Merge YAML overrides with defaults."""
    cfg = {**_POWER_DEFAULTS}
    cfg.update(_load_power_config())
    return cfg


# ---------------------------------------------------------------------------
# Team Glicko aggregation
# ---------------------------------------------------------------------------
_RP_ROLE_WEIGHTS: dict[str, float] = {"CL": 1.5, "SU": 1.0, "MR": 0.5}


def _build_team_glicko_component(
    roster: pd.DataFrame,
    batter_glicko_path: Path,
    pitcher_glicko_path: Path,
    pitcher_rankings_path: Path,
) -> pd.DataFrame:
    """Aggregate individual Glicko-2 ratings per team.

    For batters, compute a games_rated-weighted average of mu per team.
    For pitchers, weight SPs by games_rated and RPs by role multiplier
    (CL 1.5x, SU 1.0x, MR 0.5x) times games_rated.

    Both batting and pitching averages are percentile-ranked across teams
    and combined 50/50.

    Parameters
    ----------
    roster : pd.DataFrame
        Current roster with ``player_id``, ``team_abbr``, ``primary_position``.
    batter_glicko_path : Path
        Path to ``batter_glicko.parquet`` (batter_id, mu, games_rated).
    pitcher_glicko_path : Path
        Path to ``pitcher_glicko.parquet`` (pitcher_id, mu, games_rated).
    pitcher_rankings_path : Path
        Path to ``pitchers_rankings.parquet`` (pitcher_id, role_detail).

    Returns
    -------
    pd.DataFrame
        Columns: team_abbr, team_batting_glicko, team_pitching_glicko,
        team_glicko_combined.
    """
    _pitcher_pos = {"SP", "RP", "P"}

    # --- Load parquets ------------------------------------------------
    try:
        bg = pd.read_parquet(batter_glicko_path)
    except FileNotFoundError:
        logger.warning("batter_glicko.parquet not found at %s", batter_glicko_path)
        bg = pd.DataFrame(columns=["batter_id", "mu", "games_rated"])

    try:
        pg = pd.read_parquet(pitcher_glicko_path)
    except FileNotFoundError:
        logger.warning("pitcher_glicko.parquet not found at %s", pitcher_glicko_path)
        pg = pd.DataFrame(columns=["pitcher_id", "mu", "games_rated"])

    try:
        pr = pd.read_parquet(pitcher_rankings_path)
    except FileNotFoundError:
        logger.warning("pitchers_rankings.parquet not found at %s", pitcher_rankings_path)
        pr = pd.DataFrame(columns=["pitcher_id", "role_detail"])

    # --- Batter Glicko per team (games_rated-weighted avg mu) ---------
    bat_roster = roster[~roster["primary_position"].isin(_pitcher_pos)].copy()
    bat_merged = bat_roster.merge(
        bg[["batter_id", "mu", "games_rated"]],
        left_on="player_id",
        right_on="batter_id",
        how="inner",
    )

    team_bat: dict[str, float] = {}
    for abbr, grp in bat_merged.groupby("team_abbr"):
        w = grp["games_rated"].values.astype(float)
        total_w = w.sum()
        if total_w > 0:
            team_bat[abbr] = float(np.average(grp["mu"].values, weights=w))
        else:
            team_bat[abbr] = float(grp["mu"].mean())

    # --- Pitcher Glicko per team (role-weighted avg mu) ---------------
    pit_roster = roster[roster["primary_position"].isin(_pitcher_pos)].copy()
    pit_merged = pit_roster.merge(
        pg[["pitcher_id", "mu", "games_rated"]],
        left_on="player_id",
        right_on="pitcher_id",
        how="inner",
    )

    # Get role_detail from pitcher rankings
    role_lookup: dict[int, str] = {}
    if "pitcher_id" in pr.columns and "role_detail" in pr.columns:
        role_lookup = dict(zip(
            pr["pitcher_id"].astype(int),
            pr["role_detail"].astype(str),
        ))

    # Assign weight: SP uses games_rated directly; RP uses role multiplier
    def _pitcher_weight(row: pd.Series) -> float:
        pid = int(row["player_id"])
        role_det = role_lookup.get(pid, "")
        pos = str(row["primary_position"]).upper()
        gr = float(row["games_rated"])
        if pos == "SP" or role_det == "SP":
            return gr
        # RP: role-based multiplier on games_rated
        mult = _RP_ROLE_WEIGHTS.get(role_det, 0.5)
        return gr * mult

    if not pit_merged.empty:
        pit_merged["_weight"] = pit_merged.apply(_pitcher_weight, axis=1)
    else:
        pit_merged["_weight"] = 0.0

    team_pit: dict[str, float] = {}
    for abbr, grp in pit_merged.groupby("team_abbr"):
        w = grp["_weight"].values.astype(float)
        total_w = w.sum()
        if total_w > 0:
            team_pit[abbr] = float(np.average(grp["mu"].values, weights=w))
        else:
            team_pit[abbr] = float(grp["mu"].mean())

    # --- Assemble per-team frame and percentile-rank ------------------
    all_abbrs = sorted(set(roster["team_abbr"].unique()))
    rows: list[dict[str, Any]] = []
    for abbr in all_abbrs:
        rows.append({
            "team_abbr": abbr,
            "team_batting_glicko": team_bat.get(abbr, np.nan),
            "team_pitching_glicko": team_pit.get(abbr, np.nan),
        })

    glicko_df = pd.DataFrame(rows)

    # Fill missing with median so they land in the middle
    for col in ["team_batting_glicko", "team_pitching_glicko"]:
        glicko_df[col] = glicko_df[col].fillna(glicko_df[col].median())

    glicko_df["batting_glicko_pctl"] = _pctl(glicko_df["team_batting_glicko"])
    glicko_df["pitching_glicko_pctl"] = _pctl(glicko_df["team_pitching_glicko"])
    glicko_df["team_glicko_combined"] = (
        0.50 * glicko_df["batting_glicko_pctl"]
        + 0.50 * glicko_df["pitching_glicko_pctl"]
    )

    logger.info(
        "Team Glicko: %d teams | batting range [%.0f, %.0f] | pitching range [%.0f, %.0f]",
        len(glicko_df),
        glicko_df["team_batting_glicko"].min(),
        glicko_df["team_batting_glicko"].max(),
        glicko_df["team_pitching_glicko"].min(),
        glicko_df["team_pitching_glicko"].max(),
    )
    return glicko_df


# ---------------------------------------------------------------------------
# Projection-integrated power rankings
# ---------------------------------------------------------------------------
def build_power_rankings(
    elo_ratings: pd.DataFrame,
    profiles: pd.DataFrame,
    current_roster: pd.DataFrame,
    hitter_rankings: pd.DataFrame,
    pitcher_rankings: pd.DataFrame,
    hitter_projections: pd.DataFrame,
    pitcher_projections: pd.DataFrame,
    batter_glicko_path: Path | None = None,
    pitcher_glicko_path: Path | None = None,
    team_sim: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build projection-integrated power rankings for all 30 teams.

    Blends four components:

    * **ELO component** (default 40%) -- roster-adjusted pre-season ELO,
      percentile-ranked across teams.
    * **Projection component** (default 20%) -- average projected quality
      of the team's 2026 roster, plus breakout/regression narrative counts.
    * **Profile component** (default 25%) -- existing team-profile sub-scores
      (offense, pitching, defense, depth).
    * **Glicko component** (default 15%) -- aggregated individual Glicko-2
      ratings per team, games_rated/role-weighted.

    Parameters
    ----------
    elo_ratings : pd.DataFrame
        Roster-adjusted pre-season ELO (``get_current_ratings()`` output
        after ``apply_roster_talent_adjustment()``).
    profiles : pd.DataFrame
        Output of ``build_all_team_profiles()``.
    current_roster : pd.DataFrame
        2026 MLB roster with ``player_id``, ``team_abbr``, ``org_id``,
        ``primary_position``.
    hitter_rankings : pd.DataFrame
        Dashboard ``hitters_rankings.parquet`` with ``batter_id``,
        ``tdd_value_score``.
    pitcher_rankings : pd.DataFrame
        Dashboard ``pitchers_rankings.parquet`` with ``pitcher_id``,
        ``tdd_value_score``, ``role_detail``.
    hitter_projections : pd.DataFrame
        Dashboard ``hitter_projections.parquet`` with ``batter_id``,
        ``composite_score``, ``delta_k_rate``, ``delta_bb_rate``.
    pitcher_projections : pd.DataFrame
        Dashboard ``pitcher_projections.parquet`` with ``pitcher_id``,
        ``composite_score``, ``delta_k_rate``, ``delta_bb_rate``.
    batter_glicko_path : Path, optional
        Path to ``batter_glicko.parquet``. When None the Glicko component
        weight is redistributed to the other three components.
    pitcher_glicko_path : Path, optional
        Path to ``pitcher_glicko.parquet``.
    config : dict, optional
        Override weight / threshold configuration.

    Returns
    -------
    pd.DataFrame
        Power rankings with ``power_rank``, ``power_score``, ``power_tier``,
        component scores (including ``glicko_component``),
        breakout/regression counts, and projected wins.
    """
    cfg = _get_power_config()
    if config:
        cfg.update(config)

    w_elo: float = cfg["elo_weight"]
    w_proj: float = cfg["projection_weight"]
    w_prof: float = cfg["profile_weight"]
    w_glicko: float = cfg.get("glicko_weight", 0.10)
    w_wins: float = cfg.get("wins_weight", 0.30)
    breakout_thresh: float = cfg["breakout_delta_threshold"]
    regression_thresh: float = cfg["regression_delta_threshold"]

    # ---------------------------------------------------------------
    # Abbreviation <-> team_id mapping (from ELO table)
    # ---------------------------------------------------------------
    if "abbreviation" not in elo_ratings.columns or "team_id" not in elo_ratings.columns:
        logger.error("elo_ratings must contain 'abbreviation' and 'team_id'")
        return pd.DataFrame()

    abbr_tid = dict(zip(elo_ratings["abbreviation"], elo_ratings["team_id"].astype(int)))
    tid_abbr = {v: k for k, v in abbr_tid.items()}

    # Also pull team_name / division / league from ELO or profiles
    team_meta: dict[int, dict[str, str]] = {}
    meta_src = elo_ratings if "team_name" in elo_ratings.columns else profiles
    if meta_src is not None and "team_id" in meta_src.columns:
        for _, row in meta_src.iterrows():
            tid = int(row["team_id"])
            team_meta[tid] = {
                "team_name": str(row.get("team_name", "")),
                "division": str(row.get("division", "")),
                "league": str(row.get("league", "")),
            }

    # ---------------------------------------------------------------
    # 1. ELO component  (blend game ELO + series ELO percentiles)
    # ---------------------------------------------------------------
    elo_df = elo_ratings[["team_id", "composite_elo"]].copy()
    elo_df["game_elo_pctl"] = _pctl(elo_df["composite_elo"])

    # Load series ELO if available and blend with game ELO
    _series_elo_path = PROJECT_ROOT.parent / "tdd-dashboard" / "data" / "dashboard" / "team_series_elo_preseason.parquet"
    _game_blend = cfg.get("game_elo_blend", 0.60)
    _series_blend = cfg.get("series_elo_blend", 0.40)

    try:
        if _series_elo_path.exists():
            _series_df = pd.read_parquet(_series_elo_path)
            _series_df = _series_df[["team_id", "series_mu"]].copy()
            _series_df["series_elo_pctl"] = _pctl(_series_df["series_mu"])
            elo_df = elo_df.merge(
                _series_df[["team_id", "series_elo_pctl"]],
                on="team_id",
                how="left",
            )
            elo_df["series_elo_pctl"] = elo_df["series_elo_pctl"].fillna(0.5)
            elo_df["elo_pctl"] = (
                _game_blend * elo_df["game_elo_pctl"]
                + _series_blend * elo_df["series_elo_pctl"]
            )
            logger.info(
                "ELO blended: %.0f%% game + %.0f%% series",
                _game_blend * 100, _series_blend * 100,
            )
        else:
            elo_df["elo_pctl"] = elo_df["game_elo_pctl"]
            logger.info("Series ELO not found -- using game ELO only")
    except Exception:
        elo_df["elo_pctl"] = elo_df["game_elo_pctl"]
        logger.warning("Failed to load series ELO -- using game ELO only")

    elo_lookup: dict[int, float] = dict(
        zip(elo_df["team_id"].astype(int), elo_df["elo_pctl"])
    )

    # ---------------------------------------------------------------
    # 2. Projection component  (avg quality of team's current roster)
    # ---------------------------------------------------------------
    _pitcher_positions = {"SP", "RP", "P"}

    # Build player -> quality lookups (prefer current_value_score for team usage)
    h_score_col = next(
        (c for c in ("current_value_score", "tdd_value_score")
         if c in hitter_rankings.columns), None
    )
    h_quality: dict[int, float] = {}
    if "batter_id" in hitter_rankings.columns and h_score_col:
        valid = hitter_rankings.dropna(subset=[h_score_col])
        h_quality = dict(zip(valid["batter_id"].astype(int), valid[h_score_col]))

    p_score_col = next(
        (c for c in ("current_value_score", "tdd_value_score")
         if c in pitcher_rankings.columns), None
    )
    p_quality: dict[int, float] = {}
    if "pitcher_id" in pitcher_rankings.columns and p_score_col:
        valid = pitcher_rankings.dropna(subset=[p_score_col])
        p_quality = dict(zip(valid["pitcher_id"].astype(int), valid[p_score_col]))

    # PA/IP weight lookups from counting sim (projected playing time)
    h_pa_weights: dict[int, float] = {}
    p_ip_weights: dict[int, float] = {}
    if "total_pa_mean" in hitter_rankings.columns:
        h_pa_weights = dict(zip(
            hitter_rankings["batter_id"].astype(int),
            hitter_rankings["total_pa_mean"].fillna(300).clip(50),
        ))
    else:
        try:
            hcs = pd.read_parquet(DASHBOARD_DIR / "hitter_counting_sim.parquet")
            h_pa_weights = dict(zip(hcs["batter_id"].astype(int), hcs["total_pa_mean"].clip(50)))
        except Exception:
            pass
    if "projected_ip_mean" in pitcher_rankings.columns:
        p_ip_weights = dict(zip(
            pitcher_rankings["pitcher_id"].astype(int),
            pitcher_rankings.get("projected_ip_mean", pd.Series(100)).fillna(100).clip(10),
        ))
    else:
        # Load from dashboard if available
        try:
            pcs = pd.read_parquet(DASHBOARD_DIR / "pitcher_counting_sim.parquet")
            p_ip_weights = dict(zip(pcs["pitcher_id"].astype(int), pcs["projected_ip_mean"].clip(10)))
        except Exception:
            pass  # fallback: equal weight (empty dict -> 1.0 default)

    # Build player -> breakout/regression flag from projections
    # A breakout: big positive delta in K% (hitter: negative delta = better)
    # or BB% (hitter: positive delta = better, pitcher: negative = better)
    h_proj_delta: dict[int, float] = {}
    if "batter_id" in hitter_projections.columns and "composite_score" in hitter_projections.columns:
        h_proj_delta = dict(zip(
            hitter_projections["batter_id"].astype(int),
            hitter_projections["composite_score"],
        ))

    p_proj_delta: dict[int, float] = {}
    if "pitcher_id" in pitcher_projections.columns and "composite_score" in pitcher_projections.columns:
        p_proj_delta = dict(zip(
            pitcher_projections["pitcher_id"].astype(int),
            pitcher_projections["composite_score"],
        ))

    # Breakout/regression from K% delta direction
    # Hitter: negative delta_k_rate = improving (fewer Ks)
    # Pitcher: positive delta_k_rate = improving (more Ks)
    h_k_delta: dict[int, float] = {}
    if "batter_id" in hitter_projections.columns and "delta_k_rate" in hitter_projections.columns:
        h_k_delta = dict(zip(
            hitter_projections["batter_id"].astype(int),
            hitter_projections["delta_k_rate"],
        ))

    p_k_delta: dict[int, float] = {}
    if "pitcher_id" in pitcher_projections.columns and "delta_k_rate" in pitcher_projections.columns:
        p_k_delta = dict(zip(
            pitcher_projections["pitcher_id"].astype(int),
            pitcher_projections["delta_k_rate"],
        ))

    proj_rows: list[dict[str, Any]] = []
    for abbr, tid in abbr_tid.items():
        team_roster = current_roster[current_roster["team_abbr"] == abbr]
        if team_roster.empty:
            proj_rows.append({
                "team_id": tid,
                "proj_score": 0.5,
                "breakout_count": 0,
                "regression_count": 0,
            })
            continue

        # Split into hitters and pitchers
        hitter_pids: list[int] = []
        pitcher_pids: list[int] = []
        for _, row in team_roster.iterrows():
            pid = int(row["player_id"])
            pos = str(row.get("primary_position", "")).upper()
            if pos in _pitcher_positions:
                pitcher_pids.append(pid)
            else:
                hitter_pids.append(pid)

        # PA/IP-weighted quality (projected playing time as weights)
        h_scores = [(h_quality[p], h_pa_weights.get(p, 1.0)) for p in hitter_pids if p in h_quality]
        p_scores = [(p_quality[p], p_ip_weights.get(p, 1.0)) for p in pitcher_pids if p in p_quality]

        if h_scores:
            h_vals, h_wts = zip(*h_scores)
            avg_h = float(np.average(h_vals, weights=h_wts))
        else:
            avg_h = 0.5
        if p_scores:
            p_vals, p_wts = zip(*p_scores)
            avg_p = float(np.average(p_vals, weights=p_wts))
        else:
            avg_p = 0.5
        # Blend offense (55%) and pitching (45%) — offense has slightly more
        # within-season variance in MLB
        proj_raw = 0.55 * avg_h + 0.45 * avg_p

        # Breakout / regression counts
        breakout_n = 0
        regression_n = 0
        # Hitters: breakout if K rate projected to DROP (delta < threshold)
        for pid in hitter_pids:
            dk = h_k_delta.get(pid)
            if dk is not None:
                if dk < regression_thresh:  # negative = fewer Ks = better
                    breakout_n += 1
                elif dk > breakout_thresh:  # positive = more Ks = worse
                    regression_n += 1
        # Pitchers: breakout if K rate projected to RISE (delta > threshold)
        for pid in pitcher_pids:
            dk = p_k_delta.get(pid)
            if dk is not None:
                if dk > breakout_thresh:  # positive = more Ks = better
                    breakout_n += 1
                elif dk < regression_thresh:  # negative = fewer Ks = worse
                    regression_n += 1

        proj_rows.append({
            "team_id": tid,
            "proj_score_raw": proj_raw,
            "breakout_count": breakout_n,
            "regression_count": regression_n,
        })

    proj_df = pd.DataFrame(proj_rows)
    # Percentile-rank the projection score across teams
    if "proj_score_raw" in proj_df.columns:
        proj_df["projection_component"] = _pctl(proj_df["proj_score_raw"])
    else:
        proj_df["projection_component"] = 0.5

    proj_lookup: dict[int, float] = dict(
        zip(proj_df["team_id"].astype(int), proj_df["projection_component"])
    )
    breakout_lookup: dict[int, int] = dict(
        zip(proj_df["team_id"].astype(int), proj_df["breakout_count"])
    )
    regression_lookup: dict[int, int] = dict(
        zip(proj_df["team_id"].astype(int), proj_df["regression_count"])
    )

    # ---------------------------------------------------------------
    # 3. Profile component  (weighted sub-score from team_profiles)
    # ---------------------------------------------------------------
    prof_df = profiles.copy()
    for col, default in [
        ("offense_score", 0.5),
        ("rotation_score", 0.5),
        ("bullpen_score", 0.5),
        ("pitching_score", 0.5),
        ("defense_score", 0.5),
        ("health_depth_score", 0.5),
    ]:
        if col not in prof_df.columns:
            prof_df[col] = default
        prof_df[col] = prof_df[col].fillna(prof_df[col].median())

    prof_df["profile_raw"] = (
        _TALENT_WEIGHTS["offense"] * prof_df["offense_score"]
        + _TALENT_WEIGHTS["rotation"] * prof_df["rotation_score"]
        + _TALENT_WEIGHTS["bullpen"] * prof_df["bullpen_score"]
        + _TALENT_WEIGHTS["defense"] * prof_df["defense_score"]
        + _TALENT_WEIGHTS["depth"] * prof_df["health_depth_score"]
    )
    prof_df["profile_component"] = _pctl(prof_df["profile_raw"])
    prof_lookup: dict[int, float] = dict(
        zip(prof_df["team_id"].astype(int), prof_df["profile_component"])
    )

    # ---------------------------------------------------------------
    # 4. Glicko component  (aggregated individual Glicko per team)
    # ---------------------------------------------------------------
    glicko_lookup: dict[str, float] = {}
    glicko_bat_lookup: dict[str, float] = {}
    glicko_pit_lookup: dict[str, float] = {}
    _has_glicko = False

    if (
        batter_glicko_path is not None
        and pitcher_glicko_path is not None
        and batter_glicko_path.exists()
        and pitcher_glicko_path.exists()
    ):
        _pr_path = (
            pitcher_glicko_path.parent / "pitchers_rankings.parquet"
        )
        glicko_df = _build_team_glicko_component(
            roster=current_roster,
            batter_glicko_path=batter_glicko_path,
            pitcher_glicko_path=pitcher_glicko_path,
            pitcher_rankings_path=_pr_path,
        )
        if not glicko_df.empty:
            _has_glicko = True
            glicko_lookup = dict(zip(
                glicko_df["team_abbr"],
                glicko_df["team_glicko_combined"],
            ))
            glicko_bat_lookup = dict(zip(
                glicko_df["team_abbr"],
                glicko_df["team_batting_glicko"],
            ))
            glicko_pit_lookup = dict(zip(
                glicko_df["team_abbr"],
                glicko_df["team_pitching_glicko"],
            ))
    else:
        logger.info(
            "Glicko parquets not available -- redistributing %.0f%% weight",
            w_glicko * 100,
        )

    # If no Glicko data, redistribute its weight proportionally
    if not _has_glicko:
        _base = w_elo + w_proj + w_prof
        if _base > 0:
            w_elo = w_elo / _base
            w_proj = w_proj / _base
            w_prof = w_prof / _base
        w_glicko = 0.0

    # ---------------------------------------------------------------
    # Phase 4+6: De-correlated orthogonal assembly
    #
    # Four genuinely independent components:
    #   1. Projected Wins (60%) -- blended RS/RA Pythagorean from rank_teams()
    #   2. Recent Form (15%) -- ELO momentum (what talent alone can't see)
    #   3. Depth/Fragility (10%) -- downside risk (reduced until metric enriched)
    #   4. Trajectory (15%) -- net breakouts minus regressions
    #
    # Old correlated components (projection, profile, glicko) are merged
    # into Projected Wins via the run-based framework.
    # ---------------------------------------------------------------
    w_wins = cfg.get("projected_wins_weight", 0.60)
    w_form = cfg.get("recent_form_weight", 0.15)
    w_depth = cfg.get("depth_fragility_weight", 0.10)
    w_traj = cfg.get("trajectory_weight", 0.15)

    # Normalize sim wins to sum to 2430 (independent team sims don't
    # enforce zero-sum, so every team's RS can exceed RA simultaneously)
    if team_sim is not None and not team_sim.empty and "sim_wins_mean" in team_sim.columns:
        raw_sum = team_sim["sim_wins_mean"].sum()
        if raw_sum > 0:
            sim_scale = (81.0 * len(team_sim)) / raw_sum
            team_sim = team_sim.copy()
            team_sim["sim_wins_mean"] = team_sim["sim_wins_mean"] * sim_scale
            logger.info("Sim wins normalized: %.0f → %.0f (scale=%.3f)",
                        raw_sum, team_sim["sim_wins_mean"].sum(), sim_scale)

    # Pre-compute PA/IP-weighted breakout scores per team from the
    # XGBoost breakout model (Power Surge, Diamond in the Rough, etc.)
    _team_breakout_scores: dict[int, float] = {}
    try:
        _project_root = Path(__file__).resolve().parents[2]
        _dash = _project_root.parent / "tdd-dashboard" / "data" / "dashboard"
        _hr_brk = pd.read_parquet(_dash / "hitters_rankings.parquet")
        _pr_brk = pd.read_parquet(_dash / "pitchers_rankings.parquet")
        _hcs_brk = pd.read_parquet(_dash / "hitter_counting_sim.parquet")
        _pcs_brk = pd.read_parquet(_dash / "pitcher_counting_sim.parquet")

        _hb = _hr_brk[["batter_id", "breakout_score"]].merge(
            _hcs_brk[["batter_id", "total_pa_mean"]], on="batter_id", how="left",
        )
        _hb["weighted"] = _hb["breakout_score"] * _hb["total_pa_mean"].fillna(200) / 600
        # Map batter to team via roster
        _hb_team = _hb.merge(
            current_roster[["player_id", "org_id"]],
            left_on="batter_id", right_on="player_id", how="left",
        )
        for _tid, _grp in _hb_team.groupby("org_id"):
            _team_breakout_scores[int(_tid)] = _grp["weighted"].sum()

        _pb = _pr_brk[["pitcher_id", "breakout_score"]].merge(
            _pcs_brk[["pitcher_id", "projected_ip_mean"]], on="pitcher_id", how="left",
        )
        _pb["weighted"] = _pb["breakout_score"] * _pb["projected_ip_mean"].fillna(50) / 180
        _pb_team = _pb.merge(
            current_roster[["player_id", "org_id"]],
            left_on="pitcher_id", right_on="player_id", how="left",
        )
        for _tid, _grp in _pb_team.groupby("org_id"):
            _team_breakout_scores[int(_tid)] = _team_breakout_scores.get(int(_tid), 0) + _grp["weighted"].sum()

        logger.info("XGBoost breakout scores loaded for %d teams", len(_team_breakout_scores))
    except Exception as e:
        logger.warning("Could not load XGBoost breakout scores: %s", e)

    pre_rows: list[dict[str, Any]] = []
    for tid in sorted(abbr_tid.values()):
        abbr = tid_abbr.get(tid, "???")
        meta = team_meta.get(tid, {})

        # --- Component 1: Projected Wins ---
        # Use projected_wins from rank_teams() (blended RS/RA Pythagorean)
        # to avoid duplicating the win computation logic.
        avg_opp_elo_val = None
        blended_wins = 81.0
        baseruns_wins = 81.0
        sim_wins_val = None
        sim_std_val = None
        if not prof_df.empty and tid in prof_df["team_id"].values:
            team_prof = prof_df[prof_df["team_id"] == tid].iloc[0]
            avg_opp_elo_val = team_prof.get("avg_opp_elo")
            # Read pre-computed projected_wins from rank_teams() output
            pw = team_prof.get("projected_wins")
            if pd.notna(pw):
                blended_wins = float(pw)
                baseruns_wins = float(pw)

        # Blend with sim wins if available
        if team_sim is not None and not team_sim.empty:
            sim_row = team_sim[team_sim["team_abbr"] == abbr]
            if not sim_row.empty:
                sim_wins_val = float(sim_row.iloc[0]["sim_wins_mean"])
                sim_std_val = float(sim_row.iloc[0]["sim_wins_std"])
        if sim_wins_val is not None and sim_std_val is not None:
            confidence = np.clip(1.0 - sim_std_val / 2.0, 0.15, 0.85)
            blended_wins = confidence * blended_wins + (1 - confidence) * sim_wins_val

        # --- Component 2: Recent Form (ELO only) ---
        elo_comp = elo_lookup.get(tid, 0.5)

        # --- Component 3: Depth/Fragility ---
        # Use CURRENT roster quality (lineup_depth, bench_quality, roster
        # breadth) rather than backwards-looking IL history.  Teams with
        # deep rosters are more resilient regardless of prior-year injuries.
        depth_comp = 0.5
        if not prof_df.empty and tid in prof_df["team_id"].values:
            tp = prof_df[prof_df["team_id"] == tid].iloc[0]
            lineup_d = float(tp.get("lineup_depth", 0.5))
            bench_q = float(tp.get("bench_quality", 0.4))
            roster_d = float(tp.get("roster_depth_score", 0.5))
            concentration = float(tp.get("concentration", 0.3))
            # Low concentration = good (not dependent on a few stars)
            depth_comp = (
                0.35 * lineup_d
                + 0.25 * roster_d
                + 0.20 * bench_q
                + 0.20 * (1.0 - concentration)
            )

        # --- Component 4: Trajectory (XGBoost breakout model) ---
        # Use PA/IP-weighted breakout_score from the full breakout model
        # (Power Surge, Diamond in the Rough, Stuff Dominant, etc.)
        # instead of raw K% delta head count.
        breakout_n = breakout_lookup.get(tid, 0)
        regression_n = regression_lookup.get(tid, 0)
        net_trajectory = breakout_n - regression_n

        # Override net_trajectory with XGBoost breakout upside if data available
        _brk_upside = _team_breakout_scores.get(tid, 0.0) if _team_breakout_scores else 0.0

        pre_rows.append({
            "tid": tid, "abbr": abbr, "meta": meta,
            "elo_comp": elo_comp,
            "depth_comp": depth_comp,
            "proj_wins": blended_wins,
            "baseruns_wins": baseruns_wins,
            "sim_wins": sim_wins_val,
            "sim_std": sim_std_val,
            "net_trajectory": net_trajectory,
            "brk_upside": _brk_upside,
            "breakout_count": breakout_n,
            "regression_count": regression_n,
            "avg_opp_elo_val": avg_opp_elo_val,
            # Legacy: keep for display
            "proj_comp": proj_lookup.get(tid, 0.5),
            "prof_comp": prof_lookup.get(tid, 0.5),
            "glicko_comp": glicko_lookup.get(abbr, 0.5),
        })

    # ---------------------------------------------------------------
    # Normalize projected wins to sum to exactly 2430
    # (30 teams x 162 games / 2 = 2430 total wins in MLB).
    # Without this constraint, optimistic player projections and the
    # inflated season sim produce impossible win totals (e.g. avg 88+).
    # Every serious projection system (FanGraphs, PECOTA) does this.
    # ---------------------------------------------------------------
    REQUIRED_TOTAL_WINS = 30 * 162 / 2  # 2430
    raw_total = sum(r["proj_wins"] for r in pre_rows)
    if raw_total > 0 and abs(raw_total - REQUIRED_TOTAL_WINS) > 1:
        scale = REQUIRED_TOTAL_WINS / raw_total
        for r in pre_rows:
            r["proj_wins"] = r["proj_wins"] * scale
        logger.info(
            "Win normalization: raw total %.0f -> %.0f (scale %.3f)",
            raw_total, REQUIRED_TOTAL_WINS, scale,
        )

    # Scale each component to 0-1.
    # Wins: use raw scale (65-100 range) so actual win gaps matter.
    # Percentile rank compressed 10-win differences to ~0.17 — too flat.
    # Trajectory/depth: keep percentile (ordinal signal, magnitude meaningless).
    from scipy.stats import rankdata

    all_wins = np.array([r["proj_wins"] for r in pre_rows])
    # Use XGBoost breakout upside for trajectory instead of K%-only count
    all_traj = np.array([r.get("brk_upside", r["net_trajectory"]) for r in pre_rows], dtype=float)
    all_depth = np.array([r["depth_comp"] for r in pre_rows])

    # Use league sim wins if available (overwrites Pythagorean wins from
    # rank_teams).  The league sim is zero-sum and schedule-aware.
    # Look for league_sim.parquet in common dashboard locations
    _project_root = Path(__file__).resolve().parents[2]
    _ls_candidates = [
        _project_root.parent / "tdd-dashboard" / "data" / "dashboard" / "league_sim.parquet",
        _project_root / "data" / "dashboard" / "league_sim.parquet",
    ]
    _ls_path = None
    for _c in _ls_candidates:
        if _c.exists():
            _ls_path = _c
            break

    if _ls_path is not None:
        _ls = pd.read_parquet(_ls_path)
        _ls_map = dict(zip(_ls["team_id"].astype(int), _ls["sim_wins_mean"]))
        for r in pre_rows:
            if r["tid"] in _ls_map:
                r["proj_wins"] = _ls_map[r["tid"]]
        all_wins = np.array([r["proj_wins"] for r in pre_rows])
        logger.info("Power rankings using league sim wins (league_sim.parquet)")

    # Projected wins is the primary signal (70%).  Depth and trajectory
    # get meaningful weight to reflect roster quality beyond raw RS/RA.
    w_wins = 0.70
    w_form = 0.08
    w_depth = 0.10
    w_traj = 0.12

    wins_pctl = np.clip((all_wins - 70) / 30, 0, 1)  # 70W=0, 100W=1 (sharper top separation)
    traj_pctl = (rankdata(all_traj) - 1) / max(len(all_traj) - 1, 1)
    depth_pctl = (rankdata(all_depth) - 1) / max(len(all_depth) - 1, 1)

    rows: list[dict[str, Any]] = []
    for i, pr in enumerate(pre_rows):
        tid = pr["tid"]
        abbr = pr["abbr"]
        meta = pr["meta"]

        wins_comp = float(wins_pctl[i])
        form_comp = pr["elo_comp"]
        frag_comp = float(depth_pctl[i])
        traj_comp = float(traj_pctl[i])

        power_score = (
            w_wins * wins_comp
            + w_form * form_comp
            + w_depth * frag_comp
            + w_traj * traj_comp
        )

        rows.append({
            "team_id": tid,
            "abbreviation": abbr,
            "team_name": meta.get("team_name", ""),
            "division": meta.get("division", ""),
            "league": meta.get("league", ""),
            "power_score": round(power_score, 4),
            # New orthogonal components
            "wins_component": round(wins_comp, 4),
            "form_component": round(form_comp, 4),
            "depth_component": round(frag_comp, 4),
            "trajectory_component": round(traj_comp, 4),
            # Legacy components (for display / backward compat)
            "elo_component": round(pr["elo_comp"], 4),
            "projection_component": round(pr["proj_comp"], 4),
            "profile_component": round(pr["prof_comp"], 4),
            "glicko_component": round(pr["glicko_comp"], 4),
            # Projected wins detail
            "projected_wins": pr["proj_wins"],
            "baseruns_wins": pr.get("baseruns_wins"),
            "sim_wins": pr.get("sim_wins"),
            "depth_confidence": np.clip(1.0 - (pr.get("sim_std") or 1.0) / 2.0, 0.15, 0.85),
            "avg_opp_elo": pr["avg_opp_elo_val"],
            "breakout_count": pr["breakout_count"],
            "regression_count": pr["regression_count"],
            "net_trajectory": pr["net_trajectory"],
            "team_batting_glicko": glicko_bat_lookup.get(abbr),
            "team_pitching_glicko": glicko_pit_lookup.get(abbr),
        })

    result = pd.DataFrame(rows)

    # Schedule-adjusted wins: teams with easier schedules project more wins
    # avg_opp_elo centered on ~1500; each point above means harder schedule
    if "avg_opp_elo" in result.columns:
        # Convert opponent ELO to a wins adjustment
        # avg_opp_elo of 1500 = neutral (0 adjustment)
        # avg_opp_elo of 1520 = harder schedule (-2 wins)
        # avg_opp_elo of 1480 = easier schedule (+2 wins)
        elo_diff = result["avg_opp_elo"].fillna(1500) - 1500
        schedule_adjustment = -elo_diff * 0.10  # ~1 win per 10 ELO points
        result["schedule_adjusted_wins"] = (result["projected_wins"] + schedule_adjustment).round(1)
    else:
        result["schedule_adjusted_wins"] = result["projected_wins"]

    # Scale power_score to 1-10 TDD display score (worst=1, best=10)
    ps = result["power_score"]
    ps_min, ps_max = ps.min(), ps.max()
    if ps_max - ps_min > 1e-9:
        result["tdd_score"] = (1.0 + (ps - ps_min) / (ps_max - ps_min) * 9.0).round(1)
    else:
        result["tdd_score"] = 5.5

    result["power_rank"] = (
        result["power_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    def _power_tier(rank: int) -> str:
        if rank <= 6:
            return "Elite"
        if rank <= 14:
            return "Contender"
        if rank <= 22:
            return "Competitive"
        return "Rebuilding"

    result["power_tier"] = result["power_rank"].apply(_power_tier)

    # Sort by rank
    result = result.sort_values("power_rank").reset_index(drop=True)

    logger.info(
        "Power rankings: %d teams | Tiers: %s",
        len(result),
        result["power_tier"].value_counts().to_dict(),
    )
    return result
