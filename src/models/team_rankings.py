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

from src.data.paths import dashboard_dir

logger = logging.getLogger(__name__)

DASHBOARD_DIR = dashboard_dir()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Composite weights (sum to 1.0)
# ---------------------------------------------------------------------------
# Scouting grade component weights (sum to 0.55 — the talent portion)
_TALENT_WEIGHTS = {
    # Walk-forward validated 2022-2025.
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
    # Backtested 2022-2025: improves projected-win accuracy from 9.0 to ~7.5 win error.
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
# Power rankings weights
# ---------------------------------------------------------------------------
# Final composition of power_score.  wins is the primary signal; form,
# depth, and trajectory add orthogonal context.
_POWER_WEIGHTS = {
    "wins": 0.70,        # Blended projected wins (preseason sim + in-season)
    "form": 0.08,        # In-season overperformance vs preseason expectation
    "depth": 0.10,       # Roster depth / fragility
    "trajectory": 0.12,  # Breakout upside from XGBoost breakout model
}
# Thresholds that flag individual breakout / regression candidates from
# hitter+pitcher K% projections.  Used only for display counts.
_BREAKOUT_DELTA_THRESHOLD = 0.03
_REGRESSION_DELTA_THRESHOLD = -0.03


# ---------------------------------------------------------------------------
# Projection-integrated power rankings
# ---------------------------------------------------------------------------
def build_power_rankings(
    elo_ratings: pd.DataFrame,
    profiles: pd.DataFrame,
    current_roster: pd.DataFrame,
    hitter_projections: pd.DataFrame,
    pitcher_projections: pd.DataFrame,
    league_sim_df: pd.DataFrame | None = None,
    team_records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build projection-integrated power rankings for all 30 teams.

    Composes four orthogonal components:

    * **Wins (70%)** — preseason league-sim wins blended with the
      current-season record and Pythagorean run differential via a
      Beta-Binomial update.  When ``team_records`` is supplied the
      preseason sim is updated with live results before scoring.
    * **Form (8%)** — current-season overperformance: percentile of
      ``posterior_win_pct - preseason_win_pct``.  Falls back to live
      game/series ELO percentile when no records are available.
    * **Depth (10%)** — roster depth / concentration from team profiles.
    * **Trajectory (12%)** — PA/IP-weighted XGBoost breakout upside,
      with a fallback on breakout - regression K% head count.

    Parameters
    ----------
    elo_ratings : pd.DataFrame
        Live team ELO (``get_current_ratings()`` output).
    profiles : pd.DataFrame
        Output of ``build_all_team_profiles()``.  Used for depth fields
        and ``avg_opp_elo``.
    current_roster : pd.DataFrame
        Current MLB roster with ``player_id``, ``team_abbr``, ``org_id``,
        ``primary_position``.  Used by the trajectory breakout lookup.
    hitter_projections, pitcher_projections : pd.DataFrame
        Dashboard projection parquets.  Only ``delta_k_rate`` is read,
        to compute breakout/regression head-count fallback.
    league_sim_df : pd.DataFrame, optional
        League season sim wins (``run_league_sim`` output).  Taken as
        the preseason prior for the in-season blend.  Disk fallback:
        ``tdd-dashboard/data/dashboard/league_sim.parquet``.
    team_records : pd.DataFrame, optional
        Current-season team records from
        ``in_season_wins.load_current_team_records()``.  When provided,
        the preseason sim is updated with actual W/L and run
        differential via a Beta-Binomial posterior.

    Returns
    -------
    pd.DataFrame
        Columns include ``power_rank``, ``power_score``, ``power_tier``,
        ``tdd_score``, ``projected_wins``, ``schedule_adjusted_wins``,
        ``sim_wins``, ``games_played``, ``posterior_win_pct``,
        ``pythag_win_pct``, ``wins_component``, ``form_component``,
        ``depth_component``, ``trajectory_component``,
        ``elo_component``, ``breakout_count``, ``regression_count``,
        ``net_trajectory``, and ``avg_opp_elo``.
    """
    from scipy.stats import rankdata

    from src.models.in_season_wins import apply_in_season_blend

    if "abbreviation" not in elo_ratings.columns or "team_id" not in elo_ratings.columns:
        logger.error("elo_ratings must contain 'abbreviation' and 'team_id'")
        return pd.DataFrame()

    abbr_tid = dict(zip(elo_ratings["abbreviation"], elo_ratings["team_id"].astype(int)))
    tid_abbr = {v: k for k, v in abbr_tid.items()}

    # Pull team meta (name/division/league) from whichever source has it
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
    # ELO percentile (live game + series) — fallback signal for form
    # ---------------------------------------------------------------
    elo_df = elo_ratings[["team_id", "composite_elo"]].copy()
    elo_df["game_elo_pctl"] = _pctl(elo_df["composite_elo"])

    _dash = PROJECT_ROOT.parent / "tdd-dashboard" / "data" / "dashboard"
    _series_elo_path = _dash / "team_series_elo.parquet"
    try:
        if _series_elo_path.exists():
            _series_df = pd.read_parquet(_series_elo_path)[["team_id", "series_mu"]]
            _series_df["series_elo_pctl"] = _pctl(_series_df["series_mu"])
            elo_df = elo_df.merge(
                _series_df[["team_id", "series_elo_pctl"]], on="team_id", how="left",
            )
            elo_df["series_elo_pctl"] = elo_df["series_elo_pctl"].fillna(0.5)
            elo_df["elo_pctl"] = (
                0.60 * elo_df["game_elo_pctl"] + 0.40 * elo_df["series_elo_pctl"]
            )
        else:
            elo_df["elo_pctl"] = elo_df["game_elo_pctl"]
    except Exception:
        elo_df["elo_pctl"] = elo_df["game_elo_pctl"]
        logger.warning("Failed to read series ELO; using game ELO only")

    elo_lookup: dict[int, float] = dict(
        zip(elo_df["team_id"].astype(int), elo_df["elo_pctl"])
    )

    # ---------------------------------------------------------------
    # Breakout / regression head count (display + trajectory fallback)
    # ---------------------------------------------------------------
    _pitcher_positions = {"SP", "RP", "P"}
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

    breakout_lookup: dict[int, int] = {}
    regression_lookup: dict[int, int] = {}
    for abbr, tid in abbr_tid.items():
        team_roster = current_roster[current_roster["team_abbr"] == abbr]
        b_n, r_n = 0, 0
        for _, row in team_roster.iterrows():
            pid = int(row["player_id"])
            pos = str(row.get("primary_position", "")).upper()
            if pos in _pitcher_positions:
                dk = p_k_delta.get(pid)
                if dk is None:
                    continue
                if dk > _BREAKOUT_DELTA_THRESHOLD:
                    b_n += 1
                elif dk < _REGRESSION_DELTA_THRESHOLD:
                    r_n += 1
            else:
                dk = h_k_delta.get(pid)
                if dk is None:
                    continue
                if dk < _REGRESSION_DELTA_THRESHOLD:
                    b_n += 1
                elif dk > _BREAKOUT_DELTA_THRESHOLD:
                    r_n += 1
        breakout_lookup[tid] = b_n
        regression_lookup[tid] = r_n

    # ---------------------------------------------------------------
    # Profile fields (depth + avg_opp_elo + projected_wins baseline)
    # ---------------------------------------------------------------
    prof_df = profiles.copy()
    for col, default in [
        ("lineup_depth", 0.5), ("roster_depth_score", 0.5),
        ("bench_quality", 0.4), ("concentration", 0.3),
    ]:
        if col not in prof_df.columns:
            prof_df[col] = default
        prof_df[col] = prof_df[col].fillna(prof_df[col].median())

    # ---------------------------------------------------------------
    # XGBoost breakout upside per team (primary trajectory signal)
    # ---------------------------------------------------------------
    _team_breakout_scores: dict[int, float] = {}
    try:
        _hr_brk = pd.read_parquet(_dash / "hitters_rankings.parquet")
        _pr_brk = pd.read_parquet(_dash / "pitchers_rankings.parquet")
        _hcs_brk = pd.read_parquet(_dash / "hitter_counting_sim.parquet")
        _pcs_brk = pd.read_parquet(_dash / "pitcher_counting_sim.parquet")

        _hb = _hr_brk[["batter_id", "breakout_score"]].merge(
            _hcs_brk[["batter_id", "total_pa_mean"]], on="batter_id", how="left",
        )
        _hb["weighted"] = _hb["breakout_score"] * _hb["total_pa_mean"].fillna(200) / 600
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
            _team_breakout_scores[int(_tid)] = (
                _team_breakout_scores.get(int(_tid), 0) + _grp["weighted"].sum()
            )
    except Exception as e:
        logger.warning("XGBoost breakout scores unavailable: %s", e)

    # ---------------------------------------------------------------
    # Load league sim (preseason prior for in-season wins blend)
    # ---------------------------------------------------------------
    _ls: pd.DataFrame | None = league_sim_df
    if _ls is None:
        for _c in (
            _dash / "league_sim.parquet",
            PROJECT_ROOT / "data" / "dashboard" / "league_sim.parquet",
        ):
            if _c.exists():
                _ls = pd.read_parquet(_c)
                break

    # Apply in-season Beta-Binomial blend to preseason sim wins.
    if _ls is not None:
        _ls_blended = apply_in_season_blend(_ls, team_records)
        _sim_wins_map = dict(zip(_ls_blended["team_id"].astype(int), _ls_blended["sim_wins_mean"]))
        _blended_wins_map = dict(zip(_ls_blended["team_id"].astype(int), _ls_blended["blended_wins"]))
        _post_pct_map = dict(zip(_ls_blended["team_id"].astype(int), _ls_blended["posterior_win_pct"]))
        _pre_pct_map = dict(zip(_ls_blended["team_id"].astype(int), _ls_blended["preseason_win_pct"]))
        _pyth_map = dict(zip(_ls_blended["team_id"].astype(int), _ls_blended["pythag_win_pct"]))
        _gp_map = dict(zip(_ls_blended["team_id"].astype(int), _ls_blended["games_played"]))
        if team_records is not None and not team_records.empty:
            logger.info(
                "In-season wins blend applied: %d teams have current records",
                int((_ls_blended["games_played"] > 0).sum()),
            )
    else:
        _sim_wins_map, _blended_wins_map, _post_pct_map = {}, {}, {}
        _pre_pct_map, _pyth_map, _gp_map = {}, {}, {}
        logger.warning("No league sim available; falling back to 81-win default")

    # ---------------------------------------------------------------
    # Per-team assembly
    # ---------------------------------------------------------------
    pre_rows: list[dict[str, Any]] = []
    for tid in sorted(abbr_tid.values()):
        abbr = tid_abbr.get(tid, "???")
        meta = team_meta.get(tid, {})

        # Projected wins: blended preseason sim + current-season update.
        # League sim is zero-sum and schedule-aware; team_sim is per-team
        # independent and routinely over-optimistic, so we do not blend
        # team_sim into the mean.
        proj_wins_val = float(_blended_wins_map.get(tid, 81.0))
        sim_wins_val = float(_sim_wins_map.get(tid, 81.0))
        preseason_pct = float(_pre_pct_map.get(tid, 0.5))
        posterior_pct = float(_post_pct_map.get(tid, 0.5))
        pythag_pct = _pyth_map.get(tid)
        games_played = int(_gp_map.get(tid, 0))

        # Depth: weighted mix of lineup/roster/bench/concentration
        depth_comp = 0.5
        avg_opp_elo_val = None
        if tid in prof_df["team_id"].values:
            tp = prof_df[prof_df["team_id"] == tid].iloc[0]
            depth_comp = (
                0.35 * float(tp["lineup_depth"])
                + 0.25 * float(tp["roster_depth_score"])
                + 0.20 * float(tp["bench_quality"])
                + 0.20 * (1.0 - float(tp["concentration"]))
            )
            avg_opp_elo_val = tp.get("avg_opp_elo")

        # Trajectory: prefer XGBoost breakout upside, fall back to head count
        breakout_n = breakout_lookup.get(tid, 0)
        regression_n = regression_lookup.get(tid, 0)
        net_trajectory = breakout_n - regression_n
        traj_raw = (
            _team_breakout_scores.get(tid, 0.0)
            if _team_breakout_scores
            else float(net_trajectory)
        )

        # Form: in-season overperformance delta (signed).  Shrinks at
        # very small samples so one hot week doesn't dominate.
        shrink = games_played / (games_played + 20.0)
        form_raw = shrink * (posterior_pct - preseason_pct)

        pre_rows.append({
            "tid": tid, "abbr": abbr, "meta": meta,
            "proj_wins": proj_wins_val,
            "sim_wins": sim_wins_val,
            "preseason_win_pct": preseason_pct,
            "posterior_win_pct": posterior_pct,
            "pythag_win_pct": pythag_pct,
            "games_played": games_played,
            "form_raw": form_raw,
            "depth_comp": depth_comp,
            "traj_raw": traj_raw,
            "avg_opp_elo_val": avg_opp_elo_val,
            "elo_pctl": elo_lookup.get(tid, 0.5),
            "breakout_count": breakout_n,
            "regression_count": regression_n,
            "net_trajectory": net_trajectory,
        })

    # Normalize blended wins to sum to 2430 so the final distribution
    # stays honest after per-team shrinkage.
    REQUIRED_TOTAL_WINS = 30 * 162 / 2  # 2430
    raw_total = sum(r["proj_wins"] for r in pre_rows)
    if raw_total > 0 and abs(raw_total - REQUIRED_TOTAL_WINS) > 1:
        scale = REQUIRED_TOTAL_WINS / raw_total
        for r in pre_rows:
            r["proj_wins"] = r["proj_wins"] * scale

    # Component scaling to [0, 1]
    all_wins = np.array([r["proj_wins"] for r in pre_rows])
    all_traj = np.array([r["traj_raw"] for r in pre_rows], dtype=float)
    all_depth = np.array([r["depth_comp"] for r in pre_rows])
    all_form = np.array([r["form_raw"] for r in pre_rows], dtype=float)

    # Wins: raw linear scale — 70W=0, 100W=1.  Keeps actual win gaps
    # visible instead of percentile-flattening a 10-win spread.
    wins_pctl = np.clip((all_wins - 70) / 30, 0, 1)
    # Form, depth, trajectory: percentile — ordinal signals where
    # absolute magnitude is either noisy or arbitrary.
    form_pctl = (rankdata(all_form) - 1) / max(len(all_form) - 1, 1)
    depth_pctl = (rankdata(all_depth) - 1) / max(len(all_depth) - 1, 1)
    traj_pctl = (rankdata(all_traj) - 1) / max(len(all_traj) - 1, 1)

    # If no team has any games played, form_raw is all zeros — fall
    # back to live ELO percentile to keep the component meaningful.
    _all_preseason = all(r["games_played"] == 0 for r in pre_rows)
    if _all_preseason:
        form_pctl = np.array([r["elo_pctl"] for r in pre_rows])

    w_wins = _POWER_WEIGHTS["wins"]
    w_form = _POWER_WEIGHTS["form"]
    w_depth = _POWER_WEIGHTS["depth"]
    w_traj = _POWER_WEIGHTS["trajectory"]

    rows: list[dict[str, Any]] = []
    for i, pr in enumerate(pre_rows):
        wins_comp = float(wins_pctl[i])
        form_comp = float(form_pctl[i])
        frag_comp = float(depth_pctl[i])
        traj_comp = float(traj_pctl[i])

        power_score = (
            w_wins * wins_comp
            + w_form * form_comp
            + w_depth * frag_comp
            + w_traj * traj_comp
        )

        rows.append({
            "team_id": pr["tid"],
            "abbreviation": pr["abbr"],
            "team_name": pr["meta"].get("team_name", ""),
            "division": pr["meta"].get("division", ""),
            "league": pr["meta"].get("league", ""),
            "power_score": round(power_score, 4),
            "wins_component": round(wins_comp, 4),
            "form_component": round(form_comp, 4),
            "depth_component": round(frag_comp, 4),
            "trajectory_component": round(traj_comp, 4),
            "elo_component": round(pr["elo_pctl"], 4),
            "projected_wins": round(pr["proj_wins"], 1),
            "sim_wins": round(pr["sim_wins"], 1),
            "games_played": pr["games_played"],
            "preseason_win_pct": round(pr["preseason_win_pct"], 4),
            "posterior_win_pct": round(pr["posterior_win_pct"], 4),
            "pythag_win_pct": (
                round(pr["pythag_win_pct"], 4)
                if pr["pythag_win_pct"] is not None else None
            ),
            "avg_opp_elo": pr["avg_opp_elo_val"],
            "breakout_count": pr["breakout_count"],
            "regression_count": pr["regression_count"],
            "net_trajectory": pr["net_trajectory"],
        })

    result = pd.DataFrame(rows)

    # Schedule-adjusted wins: teams with easier SOS project more wins.
    # ~1 win per 10 ELO points of opponent strength differential.
    if "avg_opp_elo" in result.columns:
        elo_diff = result["avg_opp_elo"].fillna(1500) - 1500
        result["schedule_adjusted_wins"] = (
            result["projected_wins"] + (-elo_diff * 0.10)
        ).round(1)
    else:
        result["schedule_adjusted_wins"] = result["projected_wins"]

    # Scale power_score to 1-10 TDD display score
    ps = result["power_score"]
    ps_min, ps_max = ps.min(), ps.max()
    if ps_max - ps_min > 1e-9:
        result["tdd_score"] = (1.0 + (ps - ps_min) / (ps_max - ps_min) * 9.0).round(1)
    else:
        result["tdd_score"] = 5.5

    result["power_rank"] = (
        result["power_score"].rank(ascending=False, method="first").astype(int)
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
    result = result.sort_values("power_rank").reset_index(drop=True)

    logger.info(
        "Power rankings: %d teams | Tiers: %s",
        len(result),
        result["power_tier"].value_counts().to_dict(),
    )
    return result
