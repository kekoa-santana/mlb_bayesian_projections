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
# Research-aligned weights: rotation + offense are most predictive of
# one-year-ahead wins; bullpen is volatile (small-sample relievers);
# health/depth is inherently uncertain. Defense is moderately predictive
# but requires 3+ years of data for reliability.
_WEIGHTS = {
    "offense": 0.35,
    "rotation": 0.25,
    "bullpen": 0.10,
    "defense": 0.15,
    "depth": 0.15,
}

# Tier thresholds (on 0-1 composite_score)
_TIERS = [
    (0.85, "Elite"),
    (0.65, "Contender"),
    (0.40, "Competitive"),
    (0.00, "Rebuilding"),
]

PYTH_EXP = 1.83  # Baseball Pythagorean exponent


def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1)."""
    return series.rank(pct=True, method="average")


def _assign_tier(score: float) -> str:
    """Map composite score to tier label."""
    for threshold, label in _TIERS:
        if score >= threshold:
            return label
    return "Rebuilding"


def _pythagorean_wins(
    rpg: float,
    ra_per_game: float,
    games: int = 162,
    exp: float = PYTH_EXP,
) -> float:
    """Projected wins from Pythagorean formula."""
    if rpg <= 0 or ra_per_game <= 0:
        return games / 2
    wpct = rpg ** exp / (rpg ** exp + ra_per_game ** exp)
    return round(wpct * games, 1)


def rank_teams(
    profiles: pd.DataFrame,
    elo_ratings: pd.DataFrame | None = None,
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

    # Composite score (weighted average of sub-scores)
    df["composite_score"] = (
        _WEIGHTS["offense"] * df["offense_score"]
        + _WEIGHTS["rotation"] * df["rotation_score"]
        + _WEIGHTS["bullpen"] * df["bullpen_score"]
        + _WEIGHTS["defense"] * df["defense_score"]
        + _WEIGHTS["depth"] * df["health_depth_score"]
    )

    # Tier assignment
    df["tier"] = df["composite_score"].apply(_assign_tier)

    # Projected wins from Pythagorean — prefer BaseRuns-projected RS/RA,
    # fall back to observed 2025 rpg/ra_per_game
    rs_col = "proj_rs_per_game" if "proj_rs_per_game" in df.columns else "rpg"
    ra_col = "proj_ra_per_game" if "proj_ra_per_game" in df.columns else "ra_per_game"
    if rs_col in df.columns and ra_col in df.columns:
        df["projected_wins"] = df.apply(
            lambda r: _pythagorean_wins(
                r[rs_col] if pd.notna(r.get(rs_col)) else r.get("rpg", 4.5),
                r[ra_col] if pd.notna(r.get(ra_col)) else r.get("ra_per_game", 4.5),
            ),
            axis=1,
        )
    else:
        df["projected_wins"] = 81.0

    # Schedule-adjusted wins: teams with easier schedules project more wins
    # avg_opp_elo centered on ~1500; each point above means harder schedule
    if "avg_opp_elo" in df.columns:
        # Convert opponent ELO to a wins adjustment
        # avg_opp_elo of 1500 = neutral (0 adjustment)
        # avg_opp_elo of 1520 = harder schedule (-2 wins)
        # avg_opp_elo of 1480 = easier schedule (+2 wins)
        elo_diff = df["avg_opp_elo"].fillna(1500) - 1500
        schedule_adjustment = -elo_diff * 0.10  # ~1 win per 10 ELO points
        df["schedule_adjusted_wins"] = (df["projected_wins"] + schedule_adjustment).round(1)
    else:
        df["schedule_adjusted_wins"] = df["projected_wins"]

    # Merge ELO if available
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

    # Final rank by composite_score
    df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)

    # Sort and select output columns
    output_cols = [
        "team_id", "abbreviation", "team_name", "division", "league",
        "rank", "composite_score", "tier", "projected_wins", "schedule_adjusted_wins",
        "offense_score", "pitching_score", "rotation_score", "bullpen_score",
        "defense_score", "health_depth_score",
    ]
    # Add optional enrichment columns
    optional = [
        "offense_style", "pitching_style", "age_trajectory",
        "rpg", "ra_per_game", "hr_per_game", "k_per_game",
        "rotation_strength", "bullpen_depth", "lineup_depth",
        "team_oaa", "catcher_framing_runs",
        "pct_homegrown", "farm_avg_score", "n_ranked_prospects", "top_100_count",
        "total_il_days", "avg_age",
        "composite_elo", "elo_rank", "offense_elo", "pitching_elo",
        "projected_wins",
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

    # Build player -> quality lookups
    h_quality: dict[int, float] = {}
    if "batter_id" in hitter_rankings.columns and "tdd_value_score" in hitter_rankings.columns:
        valid = hitter_rankings.dropna(subset=["tdd_value_score"])
        h_quality = dict(zip(valid["batter_id"].astype(int), valid["tdd_value_score"]))

    p_quality: dict[int, float] = {}
    if "pitcher_id" in pitcher_rankings.columns and "tdd_value_score" in pitcher_rankings.columns:
        valid = pitcher_rankings.dropna(subset=["tdd_value_score"])
        p_quality = dict(zip(valid["pitcher_id"].astype(int), valid["tdd_value_score"]))

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
        _WEIGHTS["offense"] * prof_df["offense_score"]
        + _WEIGHTS["rotation"] * prof_df["rotation_score"]
        + _WEIGHTS["bullpen"] * prof_df["bullpen_score"]
        + _WEIGHTS["defense"] * prof_df["defense_score"]
        + _WEIGHTS["depth"] * prof_df["health_depth_score"]
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
    # Assemble final power rankings
    # ---------------------------------------------------------------
    pre_rows: list[dict[str, Any]] = []
    for tid in sorted(abbr_tid.values()):
        abbr = tid_abbr.get(tid, "???")
        meta = team_meta.get(tid, {})

        elo_comp = elo_lookup.get(tid, 0.5)
        proj_comp = proj_lookup.get(tid, 0.5)
        prof_comp = prof_lookup.get(tid, 0.5)
        glicko_comp = glicko_lookup.get(abbr, 0.5)

        # Projected wins: use Pythagorean if available from profiles
        rpg_val = None
        ra_val = None
        avg_opp_elo_val = None
        if not prof_df.empty and tid in prof_df["team_id"].values:
            team_prof = prof_df[prof_df["team_id"] == tid].iloc[0]
            rpg_val = team_prof.get("proj_rs_per_game", team_prof.get("rpg"))
            ra_val = team_prof.get("proj_ra_per_game", team_prof.get("ra_per_game"))
            avg_opp_elo_val = team_prof.get("avg_opp_elo")
        proj_wins = _pythagorean_wins(
            rpg_val if pd.notna(rpg_val) else 4.5,
            ra_val if pd.notna(ra_val) else 4.5,
        )

        # Blend BaseRuns wins with sim wins using confidence
        # Confidence = inverse of sim std (deep rosters → trust BaseRuns ceiling)
        sim_wins_val = None
        sim_std_val = None
        blended_wins = proj_wins  # default: BaseRuns only

        if team_sim is not None and not team_sim.empty:
            sim_row = team_sim[team_sim["team_abbr"] == abbr]
            if not sim_row.empty:
                sim_wins_val = float(sim_row.iloc[0]["sim_wins_mean"])
                sim_std_val = float(sim_row.iloc[0]["sim_wins_std"])

        if sim_wins_val is not None and sim_std_val is not None:
            # Normalize std to 0-1 confidence: low std = high confidence
            # Typical range: std 0.5–2.0 → confidence 0.75–0.0
            confidence = np.clip(1.0 - sim_std_val / 2.0, 0.15, 0.85)
            blended_wins = confidence * proj_wins + (1 - confidence) * sim_wins_val

        pre_rows.append({
            "tid": tid, "abbr": abbr, "meta": meta,
            "elo_comp": elo_comp, "proj_comp": proj_comp,
            "prof_comp": prof_comp, "glicko_comp": glicko_comp,
            "proj_wins": blended_wins,
            "baseruns_wins": proj_wins,
            "sim_wins": sim_wins_val,
            "sim_std": sim_std_val,
            "depth_confidence": np.clip(1.0 - (sim_std_val or 1.0) / 2.0, 0.15, 0.85),
            "avg_opp_elo_val": avg_opp_elo_val,
        })

    # Percentile-rank projected wins across all teams for wins_component
    all_wins = np.array([r["proj_wins"] for r in pre_rows])
    if len(all_wins) > 1 and all_wins.std() > 0:
        from scipy.stats import rankdata
        wins_pctl = (rankdata(all_wins) - 1) / (len(all_wins) - 1)
    else:
        wins_pctl = np.full(len(pre_rows), 0.5)

    rows: list[dict[str, Any]] = []
    for i, pr in enumerate(pre_rows):
        tid = pr["tid"]
        abbr = pr["abbr"]
        meta = pr["meta"]
        elo_comp = pr["elo_comp"]
        proj_comp = pr["proj_comp"]
        prof_comp = pr["prof_comp"]
        glicko_comp = pr["glicko_comp"]
        proj_wins = pr["proj_wins"]
        avg_opp_elo_val = pr["avg_opp_elo_val"]
        wins_comp = float(wins_pctl[i])

        power_score = (
            w_elo * elo_comp
            + w_proj * proj_comp
            + w_prof * prof_comp
            + w_glicko * glicko_comp
            + w_wins * wins_comp
        )

        rows.append({
            "team_id": tid,
            "abbreviation": abbr,
            "team_name": meta.get("team_name", ""),
            "division": meta.get("division", ""),
            "league": meta.get("league", ""),
            "power_score": round(power_score, 4),
            "elo_component": round(elo_comp, 4),
            "projection_component": round(proj_comp, 4),
            "profile_component": round(prof_comp, 4),
            "glicko_component": round(glicko_comp, 4),
            "wins_component": round(wins_comp, 4),
            "team_batting_glicko": glicko_bat_lookup.get(abbr),
            "team_pitching_glicko": glicko_pit_lookup.get(abbr),
            "projected_wins": pr["proj_wins"],
            "baseruns_wins": pr.get("baseruns_wins"),
            "sim_wins": pr.get("sim_wins"),
            "depth_confidence": pr.get("depth_confidence"),
            "avg_opp_elo": avg_opp_elo_val,
            "breakout_count": breakout_lookup.get(tid, 0),
            "regression_count": regression_lookup.get(tid, 0),
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

    result["power_rank"] = (
        result["power_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    result["power_tier"] = result["power_score"].apply(_assign_tier)

    # Sort by rank
    result = result.sort_values("power_rank").reset_index(drop=True)

    logger.info(
        "Power rankings: %d teams | Tiers: %s",
        len(result),
        result["power_tier"].value_counts().to_dict(),
    )
    return result
