"""
TDD Positional Player Rankings — 2026 value composite.

Ranks MLB players within each position by blending observed production
with Bayesian projections, fielding value (OAA), catcher framing,
and projected playing time.

Positions: C, 1B, 2B, 3B, SS, LF, CF, RF, DH, SP, RP.

Two-score architecture:
  - ``current_value_score``: production-dominant, feeds team rankings and
    projected wins.  Scouting grades are a minor input that shrinks as
    sample size grows (configurable via config/model.yaml ``rankings:``).
  - ``talent_upside_score``: scouting-dominant, feeds dynasty rankings and
    prospect evaluation.  Uses the same regressed diamond rating blend that
    was previously the sole ``tdd_value_score``.

``tdd_value_score`` is retained as a backward-compatible alias for
``current_value_score`` so that team profiles / team rankings / power
rankings continue to work without modification.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"
DASHBOARD_DIR = Path("C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard")

# ---------------------------------------------------------------------------
# Load ranking blend config (defaults if file/section missing)
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model.yaml"

def _load_ranking_config() -> dict:
    """Load rankings section from model.yaml with safe defaults."""
    defaults = {
        "hitter": {
            "scout_weight_floor": 0.10,
            "scout_weight_ceil": 0.30,
            "upside_scout_weight": 0.55,
            "pa_ramp_min": 150,
            "pa_ramp_max": 600,
        },
        "pitcher": {
            "scout_weight_floor": 0.10,
            "scout_weight_ceil": 0.30,
            "upside_scout_weight": 0.55,
            "bf_ramp_min": 100,
            "bf_ramp_max": 500,
        },
    }
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f).get("rankings", {})
        for role in ("hitter", "pitcher"):
            for k, v in defaults[role].items():
                defaults[role][k] = cfg.get(role, {}).get(k, v)
    except Exception:
        logger.debug("Could not load rankings config; using defaults")
    return defaults

_RANK_CFG = _load_ranking_config()


def _exposure_conditioned_scouting_weight(
    exposure: pd.Series | float,
    min_exp: float,
    max_exp: float,
    weight_ceil: float,
    weight_floor: float,
) -> pd.Series | float:
    """Compute scouting blend weight that decreases as sample grows.

    At *min_exp*: weight = *weight_ceil*  (lean on tools for small samples).
    At *max_exp*: weight = *weight_floor* (lean on production for large samples).

    Parameters
    ----------
    exposure : pd.Series | float
        PA (hitters) or BF (pitchers).
    min_exp, max_exp : float
        Ramp endpoints.
    weight_ceil, weight_floor : float
        Scouting weight at min/max exposure.

    Returns
    -------
    pd.Series | float
        Per-player scouting weight (higher = more scouting influence).
    """
    ramp = ((exposure - min_exp) / (max_exp - min_exp))
    if isinstance(ramp, pd.Series):
        ramp = ramp.clip(0, 1)
    else:
        ramp = max(0.0, min(1.0, ramp))
    return weight_ceil - ramp * (weight_ceil - weight_floor)

# ---------------------------------------------------------------------------
# Hitter sub-component weights (sum to 1.0)
# ---------------------------------------------------------------------------
_HITTER_WEIGHTS = {
    "offense": 0.52,
    "baserunning": 0.07,
    "fielding": 0.16,
    "health": 0.07,
    "role": 0.04,
    "trajectory": 0.10,
    "versatility": 0.03,
    "roster_value": 0.01,
}

# Offense sub-weights: dynamic blend based on PA (see _dynamic_blend_weights)
_PROJ_WEIGHT_BASE = 0.40  # at ~400 PA
_OBS_WEIGHT_BASE = 0.60

# ---------------------------------------------------------------------------
# Pitcher sub-component weights by role (sum to 1.0)
# ---------------------------------------------------------------------------
_SP_WEIGHTS = {
    # Walk-forward validated 2022-2025, then calibrated 2026-03-28.
    # Workload raised from 0.05 → 0.12: for SPs, innings availability IS value.
    # A 3.00 ERA over 160 IP is worth more than 2.50 ERA over 100 IP.
    # Trajectory 0.25 → 0.20, glicko 0.10 → 0.08 to fund the increase.
    # Workload + health = 23% (availability is ~quarter of SP value).
    "stuff": 0.27,
    "command": 0.22,
    "workload": 0.12,
    "health": 0.11,
    "role": 0.00,
    "trajectory": 0.20,
    "glicko": 0.08,
}
_RP_WEIGHTS = {
    # RP: stuff stays dominant; trajectory/command boosted same direction
    # as SP validation.  Workload minimal for relievers.
    "stuff": 0.32,
    "command": 0.22,
    "workload": 0.03,
    "health": 0.10,
    "role": 0.00,
    "trajectory": 0.23,
    "glicko": 0.10,
}


# ---------------------------------------------------------------------------
# Glicko-2 score loader
# ---------------------------------------------------------------------------

def _load_glicko_scores(role: str = "batter") -> pd.DataFrame:
    """Load Glicko-2 ratings and convert to a 0-1 score.

    Uses ``mu_percentile`` directly from the precomputed parquet (already
    within-group).  Only dampens toward 0.5 for very uncertain ratings
    (phi > 150, i.e. fewer than ~20 games rated) to avoid compressing
    batter scores whose mu range is narrower than pitchers.

    Parameters
    ----------
    role : str
        ``"batter"`` or ``"pitcher"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``{role}_id``, ``glicko_score``, ``glicko_mu``, ``glicko_phi``.
    """
    id_col = "batter_id" if role == "batter" else "pitcher_id"
    path = DASHBOARD_DIR / f"{role}_glicko.parquet"
    if not path.exists():
        return pd.DataFrame(columns=[id_col, "glicko_score"])

    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=[id_col, "glicko_score"])

    # Rename player_id if needed
    if "player_id" in df.columns and id_col not in df.columns:
        df = df.rename(columns={"player_id": id_col})
    if f"{role}_id" in df.columns:
        id_col = f"{role}_id"

    # Use mu_percentile directly (already within-group from precompute)
    if "mu_percentile" in df.columns:
        base_score = df["mu_percentile"].copy()
    else:
        base_score = df["mu"].rank(pct=True)

    # Only dampen for very uncertain ratings (phi > 150 = few games)
    high_uncertainty = df["phi"] > 150
    df["glicko_score"] = base_score
    df.loc[high_uncertainty, "glicko_score"] = (
        0.5 * base_score[high_uncertainty] + 0.5 * 0.5
    )

    df["glicko_mu"] = df["mu"]
    df["glicko_phi"] = df["phi"]

    return df[[id_col, "glicko_score", "glicko_mu", "glicko_phi"]]


# ---------------------------------------------------------------------------
# fWAR-style positional adjustments (runs per 162 games, normalized to 0-1)
# Used for overall_rank only — pos_rank stays within-position.
# ---------------------------------------------------------------------------
_POS_ADJUSTMENT = {
    "C": 1.00,   # +12.5 runs
    "SS": 0.88,  # +7.5 runs
    "CF": 0.72,  # +2.5 runs
    "2B": 0.72,  # +2.5 runs (aligned with CF per fWAR)
    "3B": 0.65,  # +2.5 runs
    "RF": 0.42,  # -7.5 runs (LF/RF equalized per fWAR)
    "LF": 0.42,  # -7.5 runs
    "1B": 0.22,  # -12.5 runs
    "DH": 0.08,  # -17.5 runs
}

# Position value tiers for versatility scoring
_POS_TIER = {
    "C": 3, "SS": 3, "2B": 2, "CF": 2, "3B": 2,
    "RF": 1, "LF": 1, "1B": 0, "DH": 0,
}

# All hitter positions
HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
PITCHER_ROLES = ["SP", "RP"]


def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0–1, higher = better)."""
    return series.rank(pct=True, method="average")


def _inv_pctl(series: pd.Series) -> pd.Series:
    """Inverse percentile rank (lower raw value = higher score)."""
    return 1.0 - series.rank(pct=True, method="average")


def _zscore_pctl(series: pd.Series) -> pd.Series:
    """Z-score normalized to 0-1, preserving magnitude at tails.

    Unlike ``_pctl`` (rank-based, compresses extremes), this maps the
    actual standard-deviation distance from the mean to a 0-1 scale
    via a sigmoid.  A player 3 SD above the mean scores ~0.95; a player
    at the mean scores 0.50.  This properly separates a 210 wRC+ from
    a 131 wRC+ in a way that rank-based percentiles cannot.
    """
    mu = series.mean()
    sd = series.std()
    if sd < 1e-6:
        return pd.Series(0.5, index=series.index)
    z = (series - mu) / sd
    # Sigmoid maps z to (0, 1) — steepness 1.0 gives good spread
    return 1.0 / (1.0 + np.exp(-z))


def _inv_zscore_pctl(series: pd.Series) -> pd.Series:
    """Inverse z-score percentile (lower raw value = higher score)."""
    return _zscore_pctl(-series)


def _dynamic_blend_weights(pa: pd.Series, min_pa: int = 150, full_pa: int = 600) -> tuple[pd.Series, pd.Series]:
    """Compute per-player observed/projected blend weights based on PA.

    More PA → trust observed more. Fewer PA → lean on projections.

    Returns (proj_weight, obs_weight) Series that sum to 1.0 per row.
    """
    # Linearly scale observed weight from 0.35 (at min_pa) to 0.75 (at full_pa)
    frac = ((pa - min_pa) / (full_pa - min_pa)).clip(0, 1)
    obs_w = 0.35 + 0.40 * frac
    proj_w = 1.0 - obs_w
    return proj_w, obs_w


def _hitter_age_factor(age: pd.Series) -> pd.Series:
    """Research-aligned hitter aging curve (piecewise linear).

    Based on industry consensus (FanGraphs, ZiPS, Marcel, OOPSY):
    - Plateau 26-29: wRC+ peaks 26-27, ISO holds to ~28-29, BB%
      improves through 29.  No penalty during prime window.
    - Gradual decline 30-34: ~3-4% of peak per year (~0.5 WAR/yr)
    - Steep decline 35-40: accelerating loss

    Uses explicit biology-based curve, not population-relative _inv_pctl,
    because aging is biological — a 33-year-old declines regardless of
    how young or old the league population is.
    """
    # Phase 1: Development climb (20 → 26)
    climb = 0.60 + 0.40 * ((age - 20) / 6.0).clip(0, 1)
    # Phase 2: Prime plateau (26 → 29) — no decline
    plateau = 1.0
    # Phase 3: Gradual post-prime decline (30 → 34, ~4%/yr → 0.80 at 34)
    slow_decline = 1.0 - 0.20 * ((age - 29) / 5.0).clip(0, 1)
    # Phase 4: Steep late-career decline (35 → 40)
    steep_decline = 0.80 - 0.60 * ((age - 34) / 6.0).clip(0, 1)

    raw = np.where(
        age < 26, climb,
        np.where(age <= 29, plateau,
                 np.where(age <= 34, slow_decline, steep_decline))
    )
    return pd.Series(raw, index=age.index).clip(0, 1)


def _pitcher_age_factor(age: pd.Series) -> pd.Series:
    """Research-aligned pitcher aging curve (piecewise linear).

    Pitchers peak slightly later than hitters (27-30).  K% and SwStr%
    hold through ~30; decline driven by velocity loss which is captured
    separately in the velo_trend component of trajectory scoring.
    """
    # Phase 1: Development climb (20 → 27)
    climb = 0.55 + 0.45 * ((age - 20) / 7.0).clip(0, 1)
    # Phase 2: Prime plateau (27 → 30) — no decline
    plateau = 1.0
    # Phase 3: Gradual post-prime decline (31 → 35, ~4%/yr → 0.80 at 35)
    slow_decline = 1.0 - 0.20 * ((age - 30) / 5.0).clip(0, 1)
    # Phase 4: Steep late-career decline (36 → 41)
    steep_decline = 0.80 - 0.60 * ((age - 35) / 6.0).clip(0, 1)

    raw = np.where(
        age < 27, climb,
        np.where(age <= 30, plateau,
                 np.where(age <= 35, slow_decline, steep_decline))
    )
    return pd.Series(raw, index=age.index).clip(0, 1)


# ===================================================================
# Position assignment
# ===================================================================

def _get_career_weighted_positions(season: int = 2025) -> pd.DataFrame:
    """Query career position starts with recency weighting.

    Weighting scheme (by game recency within each player):
      - Last 50 games:  3x  (catches recent position switches)
      - Games 51-162:   2x  (recent full season)
      - Games 163+:     1x  (career history)

    If 80%+ of the last 50 games are at a single position that differs
    from the career-weighted primary, the recent position overrides
    (e.g., Bichette: career SS but playing 3B for the Mets in 2026).

    Parameters
    ----------
    season : int
        Most recent completed season (lineup data through this season
        plus any spring training / early games from season + 1).

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, weighted_games, is_primary,
        recent_position (position from last 50 games if override applies).
    """
    from src.data.db import read_sql

    # Career starts with row number for recency weighting
    raw = read_sql(f"""
        WITH numbered AS (
            SELECT player_id, position, season,
                   ROW_NUMBER() OVER (
                       PARTITION BY player_id
                       ORDER BY season DESC, game_pk DESC
                   ) AS rn
            FROM production.fact_lineup
            WHERE is_starter = true
              AND position NOT IN ('P', 'PR', 'PH', 'EH')
              AND season <= {season + 1}
        )
        SELECT player_id, position, rn
        FROM numbered
    """, {})

    if raw.empty:
        return pd.DataFrame(columns=[
            "player_id", "position", "weighted_games", "is_primary",
        ])

    # Recency weights
    raw["weight"] = np.where(
        raw["rn"] <= 50, 3.0,
        np.where(raw["rn"] <= 162, 2.0, 1.0),
    )

    # Weighted games per player-position
    weighted = (
        raw.groupby(["player_id", "position"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "weighted_games"})
    )

    # --- Override 1: Current season (season+1) position ---
    # If a player has 10+ games in ST / early season and 80%+ are at a
    # NEW position, override.  Catches mid-career switches that show up
    # in spring training before the career data has enough weight
    # (e.g., Bichette SS→3B when traded to the Mets).
    next_season = raw.merge(
        raw.groupby("player_id")["rn"].min().reset_index().rename(columns={"rn": "min_rn"}),
        on="player_id",
    )
    # Games in the most recent season are those with smallest rn values
    # Use the fact that rn is ordered by season DESC, game_pk DESC
    # Get games from season+1 by checking if rn is within the count of season+1 games
    from src.data.db import read_sql as _read_next
    next_season_counts = _read_next(f"""
        SELECT player_id, position, COUNT(*) as games
        FROM production.fact_lineup
        WHERE season = {season + 1} AND is_starter = true
          AND position NOT IN ('P', 'PR', 'PH', 'EH')
        GROUP BY player_id, position
    """, {})

    current_override = pd.DataFrame(columns=["player_id", "recent_position"])
    if not next_season_counts.empty:
        ns_totals = next_season_counts.groupby("player_id")["games"].sum().reset_index(name="total")
        ns_top = next_season_counts.sort_values("games", ascending=False).groupby("player_id").first().reset_index()
        ns_top = ns_top.merge(ns_totals, on="player_id")
        ns_top["pct"] = ns_top["games"] / ns_top["total"]
        current_override = ns_top[
            (ns_top["games"] >= 10) & (ns_top["pct"] >= 0.80)
        ][["player_id", "position"]].rename(columns={"position": "recent_position"})

    # --- Override 2: Last 50 games ---
    # Broader check across seasons: if 80%+ of last 50 games are at one
    # position, override.  Catches switches that span a season boundary.
    recent = raw[raw["rn"] <= 50].copy()
    recent_counts = (
        recent.groupby(["player_id", "position"])
        .size()
        .reset_index(name="recent_games")
    )
    recent_totals = recent.groupby("player_id").size().reset_index(name="total_recent")
    recent_counts = recent_counts.merge(recent_totals, on="player_id")
    recent_counts["recent_pct"] = recent_counts["recent_games"] / recent_counts["total_recent"]

    recent_primary = (
        recent_counts
        .sort_values("recent_games", ascending=False)
        .groupby("player_id")
        .first()
        .reset_index()
    )
    last50_override = recent_primary[recent_primary["recent_pct"] >= 0.80][
        ["player_id", "position"]
    ].rename(columns={"position": "recent_position"})

    # Merge overrides: current season takes priority over last-50
    recent_override = pd.concat([last50_override, current_override], ignore_index=True)
    recent_override = recent_override.drop_duplicates(subset="player_id", keep="last")

    # Career-weighted primary = highest weighted_games
    weighted["is_primary"] = (
        weighted["weighted_games"]
        == weighted.groupby("player_id")["weighted_games"].transform("max")
    )

    # Apply last-50 override: if recent position differs from career primary
    primary = weighted[weighted["is_primary"]].merge(
        recent_override, on="player_id", how="left",
    )
    overrides = primary[
        primary["recent_position"].notna()
        & (primary["position"] != primary["recent_position"])
    ][["player_id", "recent_position"]]

    if not overrides.empty:
        override_map = dict(zip(overrides["player_id"], overrides["recent_position"]))
        # Flip primary flag: old primary → False, new recent position → True
        for pid, new_pos in override_map.items():
            mask = weighted["player_id"] == pid
            weighted.loc[mask, "is_primary"] = (
                weighted.loc[mask, "position"] == new_pos
            )
            # If the new position doesn't exist in career data, add it
            if not (mask & (weighted["position"] == new_pos)).any():
                weighted = pd.concat([weighted, pd.DataFrame([{
                    "player_id": pid,
                    "position": new_pos,
                    "weighted_games": 50 * 3,  # assume 50 recent games
                    "is_primary": True,
                }])], ignore_index=True)
                # Unflag old primary
                old_primary = mask & weighted["is_primary"] & (weighted["position"] != new_pos)
                weighted.loc[old_primary, "is_primary"] = False
        logger.info(
            "Position override (last 50 games): %d players — %s",
            len(override_map),
            ", ".join(
                f"{pid}→{pos}" for pid, pos in list(override_map.items())[:5]
            ) + ("..." if len(override_map) > 5 else ""),
        )

    return weighted.sort_values(
        ["player_id", "weighted_games"], ascending=[True, False],
    ).reset_index(drop=True)


def _verify_positions_mlb_api(
    positions: pd.DataFrame,
    season: int = 2026,
    lineup_override_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Cross-check position assignments against the MLB Stats API.

    Fetches current rosters from the API and uses the API's
    ``primaryPosition`` as a final sanity check.  Only overrides when
    the API position is a fielding position and our assignment differs.

    Players in ``lineup_override_ids`` are SKIPPED — their position was
    already set from actual game lineup data (more current than the
    API's static ``primaryPosition`` field which MLB updates slowly).

    Parameters
    ----------
    positions : pd.DataFrame
        Must have ``player_id`` and ``position`` (primary only).
    season : int
        Season for roster lookup.
    lineup_override_ids : set[int] or None
        Player IDs whose position was set by lineup override.
        API will not override these.

    Returns
    -------
    pd.DataFrame
        Same as input with positions corrected where API disagrees.
    """
    import requests

    MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
    FIELDING_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}

    # Fetch all 30 teams
    try:
        teams_resp = requests.get(
            f"{MLB_API_BASE}/teams",
            params={"season": season, "sportId": 1},
            timeout=10,
        )
        teams_resp.raise_for_status()
        team_ids = [t["id"] for t in teams_resp.json().get("teams", [])]
    except Exception as e:
        logger.warning("MLB API teams fetch failed: %s", e)
        return positions

    api_positions: dict[int, str] = {}
    api_teams: dict[int, str] = {}
    for tid in team_ids:
        try:
            resp = requests.get(
                f"{MLB_API_BASE}/teams/{tid}/roster",
                params={"season": season, "rosterType": "fullSeason"},
                timeout=10,
            )
            resp.raise_for_status()
            roster = resp.json().get("roster", [])
            # Get team abbreviation
            team_resp = requests.get(
                f"{MLB_API_BASE}/teams/{tid}",
                timeout=10,
            )
            team_data = team_resp.json().get("teams", [{}])[0]
            team_abbr = team_data.get("abbreviation", "")

            for player in roster:
                pid = player.get("person", {}).get("id")
                pos_abbr = player.get("position", {}).get("abbreviation", "")
                if pid and pos_abbr in FIELDING_POSITIONS:
                    api_positions[pid] = pos_abbr
                    api_teams[pid] = team_abbr
        except Exception:
            continue  # skip team on error, don't block

    if not api_positions:
        logger.warning("MLB API returned no roster data — skipping verification")
        return positions

    # Compare and override where API disagrees
    positions = positions.copy()
    skip_ids = lineup_override_ids or set()
    n_fixed = 0
    n_skipped = 0
    for idx, row in positions.iterrows():
        pid = row["player_id"]
        if pid in skip_ids:
            n_skipped += 1
            continue
        if pid in api_positions:
            api_pos = api_positions[pid]
            if api_pos != row["position"] and api_pos in FIELDING_POSITIONS:
                logger.debug(
                    "API override: %d %s → %s (%s)",
                    pid, row["position"], api_pos,
                    api_teams.get(pid, "?"),
                )
                positions.at[idx, "position"] = api_pos
                n_fixed += 1

    logger.info(
        "MLB API verification: %d players checked, %d corrected, %d skipped (lineup override)",
        len(api_positions), n_fixed, n_skipped,
    )
    return positions


def _assign_hitter_positions(season: int = 2025, min_starts: int = 20) -> pd.DataFrame:
    """Assign primary position to each hitter using career-weighted data.

    Uses career lineup starts with recency weighting (last 50 games 3x,
    recent season 2x, career 1x).  Overrides with last-50-games position
    if 80%+ of recent games are at a new position (catches mid-career
    switches like Bichette SS→3B).  Verifies against the MLB Stats API
    as a final sanity check.

    Falls back to ``dim_player.primary_position`` for players with no
    lineup history.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_starts : int
        Minimum weighted games to qualify.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position.
    """
    from src.data.db import read_sql

    # Career-weighted positions with recency bias
    career = _get_career_weighted_positions(season)

    if not career.empty:
        primary = career[career["is_primary"]][["player_id", "position"]].copy()
        # Filter by minimum weighted games
        top_weighted = career.groupby("player_id")["weighted_games"].max().reset_index()
        qualified = top_weighted[top_weighted["weighted_games"] >= min_starts]["player_id"]
        primary = primary[primary["player_id"].isin(qualified)]
    else:
        primary = pd.DataFrame(columns=["player_id", "position"])

    # Fallback: dim_player for players not in lineup data
    fallback = read_sql("""
        SELECT player_id, primary_position as position
        FROM production.dim_player
        WHERE primary_position NOT IN ('P', 'TWP')
          AND active = true
    """, {})

    if not primary.empty:
        missing = fallback[~fallback["player_id"].isin(primary["player_id"])]
        assigned = pd.concat(
            [primary, missing[["player_id", "position"]]],
            ignore_index=True,
        )
    else:
        assigned = fallback[["player_id", "position"]].copy()

    # Collect IDs that were overridden by lineup data (current season
    # or last-50 games).  These should NOT be re-overridden by the API's
    # static primaryPosition field which MLB updates slowly.
    lineup_override_ids: set[int] = set()
    if not career.empty:
        # Players whose is_primary position differs from their highest
        # career weighted_games position before override
        raw_primary = (
            career.sort_values("weighted_games", ascending=False)
            .drop_duplicates("player_id", keep="first")
        )
        for _, row in primary.iterrows():
            pid = row["player_id"]
            raw_row = raw_primary[raw_primary["player_id"] == pid]
            if not raw_row.empty and raw_row.iloc[0]["position"] != row["position"]:
                lineup_override_ids.add(int(pid))
        if lineup_override_ids:
            logger.info(
                "Lineup-overridden positions (protected from API): %d players",
                len(lineup_override_ids),
            )

    # MLB API verification (final pass — skips lineup-overridden players)
    try:
        assigned = _verify_positions_mlb_api(
            assigned, season=season + 1,
            lineup_override_ids=lineup_override_ids,
        )
    except Exception:
        logger.warning("MLB API verification failed — using lineup-based positions",
                       exc_info=True)

    return assigned


def get_hitter_position_eligibility(
    season: int = 2025, min_starts: int = 10,
) -> pd.DataFrame:
    """Return all positions each hitter is eligible for using career data.

    Uses career-weighted lineup starts (recency-biased) rather than a
    single season.  A player qualifies at a position with ``min_starts``
    or more weighted games there.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_starts : int
        Minimum weighted games at a position to be eligible.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, starts, pct, is_primary.
    """
    from src.data.db import read_sql

    career = _get_career_weighted_positions(season)

    if career.empty:
        fallback = read_sql("""
            SELECT player_id, primary_position as position
            FROM production.dim_player
            WHERE primary_position NOT IN ('P', 'TWP')
              AND active = true
        """, {})
        fallback["starts"] = 0
        fallback["pct"] = 1.0
        fallback["is_primary"] = True
        return fallback

    # Filter by minimum weighted games
    starts = career[career["weighted_games"] >= min_starts].copy()
    starts = starts.rename(columns={"weighted_games": "starts"})

    # Compute percentage of each player's total
    totals = starts.groupby("player_id")["starts"].transform("sum")
    starts["pct"] = starts["starts"] / totals

    # Add fallback for players not in career data
    fallback = read_sql("""
        SELECT player_id, primary_position as position
        FROM production.dim_player
        WHERE primary_position NOT IN ('P', 'TWP')
          AND active = true
    """, {})
    missing = fallback[~fallback["player_id"].isin(starts["player_id"])]
    if not missing.empty:
        missing = missing.copy()
        missing["starts"] = 0
        missing["pct"] = 1.0
        missing["is_primary"] = True
        starts = pd.concat(
            [starts, missing[["player_id", "position", "starts", "pct", "is_primary"]]],
            ignore_index=True,
        )

    return starts.sort_values(
        ["player_id", "starts"], ascending=[True, False],
    ).reset_index(drop=True)


def _assign_pitcher_roles() -> pd.DataFrame:
    """Assign SP/RP role from pitcher projections.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, role ('SP' or 'RP').
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")
    proj["role"] = np.where(proj["is_starter"] == 1, "SP", "RP")
    return proj[["pitcher_id", "role"]].copy()


# ===================================================================
# Hitter ranking
# ===================================================================

def _stat_family_trust(
    pa: pd.Series,
    min_pa: float,
    full_pa: float,
) -> pd.Series:
    """Stat-family reliability ramp: 0 at *min_pa*, 1 at *full_pa*.

    Different stat families stabilize at different rates.  Contact skill
    (K%) stabilizes faster (~150 PA) than damage metrics (~300 PA).
    This ramp controls how much to trust observed data vs projections
    within each skill bucket.
    """
    return ((pa - min_pa) / (full_pa - min_pa)).clip(0, 1)


def _build_hitter_offense_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
    aggressiveness: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Decomposed hitter offense: contact + decisions + damage.

    Three skill buckets, each with its own reliability curve:

    1. **Contact skill** (stabilizes ~150 PA): K% projected, K% observed.
       Fast-stabilizing — the most projectable hitter trait.
    2. **Swing decisions** (stabilizes ~200 PA): BB% projected, chase rate,
       two-strike whiff rate.  Medium stability — discipline metrics.
    3. **Damage on contact** (stabilizes ~300 PA): xwOBA, barrel%, hard_hit%,
       sweet_spot%.  Slow-stabilizing — requires more BIP for signal.

    Each bucket blends projected + observed using its own trust ramp, then
    buckets are combined.  Replaces the old monolithic offense blend that
    mixed all signals at one global PA reliability.

    Parameters
    ----------
    proj : pd.DataFrame
        Hitter projections (from dashboard parquet).
    observed : pd.DataFrame
        Observed season stats from ``fact_batting_advanced``.
    aggressiveness : pd.DataFrame, optional
        From ``get_hitter_aggressiveness()`` — chase_rate, two_strike_whiff_rate.

    Returns
    -------
    pd.DataFrame
        batter_id + offense_score + sub-bucket columns.
    """
    # Merge projection + observed on batter_id
    obs_cols = ["batter_id", "pa", "k_pct", "bb_pct", "woba", "wrc_plus",
                "barrel_pct", "hard_hit_pct", "xwoba", "sweet_spot_pct",
                # Multi-year recency-weighted columns (when available)
                "wrc_plus_multiyear", "xwoba_multiyear",
                "barrel_pct_multiyear", "hard_hit_pct_multiyear"]
    obs_cols = [c for c in obs_cols if c in observed.columns]
    proj_cols = ["batter_id", "projected_k_rate", "projected_bb_rate",
                  "projected_hr_per_fb", "composite_score"]
    if "age" in proj.columns:
        proj_cols.append("age")
    merged = proj[proj_cols].merge(
        observed[obs_cols],
        on="batter_id", how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=["batter_id", "offense_score",
                                      "contact_skill", "decision_skill",
                                      "damage_skill", "production_skill"])

    pa = merged["pa"]

    # Merge aggressiveness (chase, 2-strike whiff) if provided
    if aggressiveness is not None and not aggressiveness.empty:
        agg_cols = ["batter_id"]
        for c in ("chase_rate", "two_strike_whiff_rate"):
            if c in aggressiveness.columns:
                agg_cols.append(c)
        if len(agg_cols) > 1:
            merged = merged.merge(aggressiveness[agg_cols], on="batter_id", how="left")

    # Optionally load sim wRC+ for damage bucket
    sim_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
    has_sim = False
    if sim_path.exists():
        sim_df = pd.read_parquet(sim_path)
        if "projected_wrc_plus_mean" in sim_df.columns:
            merged = merged.merge(
                sim_df[["batter_id", "projected_wrc_plus_mean"]],
                on="batter_id", how="left",
            )
            has_sim = merged["projected_wrc_plus_mean"].notna().any()
            if has_sim:
                logger.info("Offense using sim wRC+ for damage bucket (%d hitters)",
                            merged["projected_wrc_plus_mean"].notna().sum())

    # =================================================================
    # Bucket 1: CONTACT SKILL (stabilizes ~150 PA)
    # Projected K% is the most stable hitter stat (r=0.80 YoY).
    # Observed K% confirms or challenges the projection quickly.
    # =================================================================
    proj_contact = _inv_zscore_pctl(merged["projected_k_rate"])
    obs_k = _inv_zscore_pctl(merged["k_pct"]) if "k_pct" in merged.columns else proj_contact

    contact_trust = _stat_family_trust(pa, min_pa=100, full_pa=350)
    merged["contact_skill"] = (1 - contact_trust) * proj_contact + contact_trust * obs_k

    # =================================================================
    # Bucket 2: SWING DECISIONS (stabilizes ~200 PA)
    # BB% projection + observed chase/discipline metrics.
    # Chase rate is extremely stable (r=0.84 YoY).
    # =================================================================
    proj_decisions = _zscore_pctl(merged["projected_bb_rate"])

    obs_decision_parts = [proj_decisions]  # fallback: just projection
    obs_decision_weights = [1.0]

    if "chase_rate" in merged.columns:
        obs_chase = _inv_zscore_pctl(merged["chase_rate"].fillna(
            merged["chase_rate"].median()))
        obs_decision_parts = [obs_chase]
        obs_decision_weights = [0.55]
        if "two_strike_whiff_rate" in merged.columns:
            obs_2s = _inv_zscore_pctl(merged["two_strike_whiff_rate"].fillna(
                merged["two_strike_whiff_rate"].median()))
            obs_decision_parts.append(obs_2s)
            obs_decision_weights.append(0.45)

    obs_decisions = sum(w * p for w, p in zip(obs_decision_weights, obs_decision_parts))

    decision_trust = _stat_family_trust(pa, min_pa=150, full_pa=450)
    merged["decision_skill"] = (1 - decision_trust) * proj_decisions + decision_trust * obs_decisions

    # =================================================================
    # Bucket 3: DAMAGE ON CONTACT (stabilizes ~300 PA)
    # xwOBA + barrel% + hard_hit% + sweet_spot%.  Noisier metrics
    # that need more BIP to stabilize.  Sim wRC+ serves as projected
    # anchor when available; falls back to HR/FB projection.
    # Removed xBA and xSLG (redundant with xwOBA, ~80% shared variance).
    # =================================================================
    if has_sim:
        proj_damage = _zscore_pctl(merged["projected_wrc_plus_mean"].fillna(100))
    else:
        proj_damage = _zscore_pctl(merged["projected_hr_per_fb"])

    # Use multi-year recency-weighted Statcast when available (smooths
    # outlier single seasons like Springer's 2025 career-best at 35).
    # Falls back to single-season when multi-year data is absent.
    _xwoba_col = "xwoba_multiyear" if "xwoba_multiyear" in merged.columns else "xwoba"
    _barrel_col = "barrel_pct_multiyear" if "barrel_pct_multiyear" in merged.columns else "barrel_pct"
    _hardhit_col = "hard_hit_pct_multiyear" if "hard_hit_pct_multiyear" in merged.columns else "hard_hit_pct"

    obs_damage_parts = []
    obs_damage_weights = []
    if _xwoba_col in merged.columns:
        obs_damage_parts.append(_zscore_pctl(merged[_xwoba_col].fillna(merged[_xwoba_col].median())))
        obs_damage_weights.append(0.35)
    if _barrel_col in merged.columns:
        obs_damage_parts.append(_zscore_pctl(merged[_barrel_col].fillna(0)))
        obs_damage_weights.append(0.30)
    if _hardhit_col in merged.columns:
        obs_damage_parts.append(_zscore_pctl(merged[_hardhit_col].fillna(0)))
        obs_damage_weights.append(0.20)
    if "sweet_spot_pct" in merged.columns:
        obs_damage_parts.append(_zscore_pctl(merged["sweet_spot_pct"].fillna(0)))
        obs_damage_weights.append(0.15)

    if obs_damage_parts:
        # Renormalize weights in case some columns are missing
        total_w = sum(obs_damage_weights)
        obs_damage = sum(
            (w / total_w) * p for w, p in zip(obs_damage_weights, obs_damage_parts)
        )
    else:
        obs_damage = proj_damage

    # Age discount on observed stats: older players' outlier seasons get
    # less trust, regressing more toward projections.  A 36-year-old's
    # career-best is far more likely regression-bound than a 26-year-old's.
    # Applies to damage + production (the two observation-heavy buckets).
    # Contact + decisions are already projection-anchored via K%/BB%.
    age = merged["age"].fillna(28)
    # Two-phase age trust: gentle 30-33, steeper 33+ (biological decline
    # accelerates, and career-best seasons at 33+ are far more likely to
    # regress than at 28).
    phase1 = (1.0 - ((age - 30).clip(0) * 0.03)).clip(0.91, 1.0)  # 30-33: gentle
    phase2 = (0.91 - ((age - 33).clip(0) * 0.06)).clip(0.50, 0.91)  # 33+: steeper
    age_trust = np.where(age <= 33, phase1, phase2)
    # age 30: 1.00, age 32: 0.94, age 33: 0.91, age 35: 0.79, age 37: 0.67, age 39: 0.55

    # Projection-deviation dampening: when observed stats diverge sharply
    # from the Bayesian projection, reduce trust in observed.  The model
    # has multi-year context — a huge deviation signals an outlier season
    # (positive or negative) that is unlikely to repeat.
    # Works both ways: a career-worst year for a good player AND a
    # career-best year for an aging player both get dampened.
    #
    # Computed on RAW stat scale (not percentiles) to avoid tail
    # compression — a 44-point wRC+ gap is meaningful even if both
    # values are above the 90th percentile.
    damage_deviation = (obs_damage - proj_damage).abs()
    damage_dev_trust = (1.0 - damage_deviation).clip(0.50, 1.0)

    # Aging spike penalty: if a 33+ year-old's most recent season sharply
    # outperforms their multi-year baseline, it's almost certainly an
    # outlier that will regress.  A 26-year-old breakout is believable;
    # a 36-year-old career-best after two bad years is not.
    # The penalty scales with age: harsher at 35+ than at 33.
    aging_spike_penalty = pd.Series(1.0, index=merged.index)
    if "xwoba" in merged.columns and "xwoba_multiyear" in merged.columns:
        single = merged["xwoba"].fillna(0)
        multi = merged["xwoba_multiyear"].fillna(single)
        spike = (single - multi).clip(0)  # only penalize positive spikes
        # Age-scaled severity: 33 = 15% per unit, 35 = 25%, 37 = 35%
        age_severity = (0.15 + ((age - 33).clip(0) * 0.05)).clip(0.15, 0.40)
        is_aging_spike = (age >= 33) & (spike > 0.015)
        spike_factor = (1.0 - (spike - 0.015) / 0.030 * age_severity).clip(0.35, 1.0)
        aging_spike_penalty = np.where(is_aging_spike, spike_factor, 1.0)

    damage_trust = (
        _stat_family_trust(pa, min_pa=200, full_pa=550)
        * age_trust * damage_dev_trust * aging_spike_penalty
    )
    merged["damage_skill"] = (1 - damage_trust) * proj_damage + damage_trust * obs_damage

    # =================================================================
    # Bucket 4: PRODUCTION (stabilizes ~250 PA)
    # Observed wRC+ — the bottom line.  Anchors the score to actual
    # run production so that high-process / low-results players
    # (e.g. low-K% / low-chase but .280 wOBA) cannot score elite.
    # Projected component uses sim wRC+ when available.
    # =================================================================
    if has_sim:
        proj_production = _zscore_pctl(merged["projected_wrc_plus_mean"].fillna(100))
    else:
        proj_production = pd.Series(0.5, index=merged.index)

    # Use multi-year recency-weighted wRC+ when available (same smoothing
    # as damage bucket — prevents single-season outliers from dominating).
    _wrc_col = "wrc_plus_multiyear" if "wrc_plus_multiyear" in merged.columns else "wrc_plus"
    if _wrc_col in merged.columns:
        obs_production = _zscore_pctl(merged[_wrc_col].fillna(100))
    else:
        obs_production = proj_production

    # Production deviation: use raw wRC+ gap normalized by population SD.
    # Percentile deviation compresses the tails (165 vs 121 wRC+ both map
    # to >93rd pctl, hiding the 44-point gap).  Raw z-score preserves it.
    if has_sim and _wrc_col in merged.columns:
        raw_wrc_gap = (merged[_wrc_col].fillna(100) - merged["projected_wrc_plus_mean"].fillna(100)).abs()
        wrc_std = merged["wrc_plus"].std()
        if wrc_std > 0:
            production_deviation_z = raw_wrc_gap / wrc_std
        else:
            production_deviation_z = pd.Series(0.0, index=merged.index)
        # z=0: trust 1.0, z=1 (~30 wRC+ gap): 0.70, z=1.67+: 0.50 floor
        production_dev_trust = (1.0 - production_deviation_z * 0.30).clip(0.50, 1.0)
    else:
        production_dev_trust = pd.Series(1.0, index=merged.index)

    # Aging spike penalty for production (same logic as damage bucket)
    aging_spike_prod = pd.Series(1.0, index=merged.index)
    if "wrc_plus" in merged.columns and "wrc_plus_multiyear" in merged.columns:
        wrc_single = merged["wrc_plus"].fillna(100)
        wrc_multi = merged["wrc_plus_multiyear"].fillna(wrc_single)
        wrc_spike = (wrc_single - wrc_multi).clip(0)
        # Age-scaled wRC+ spike penalty (same pattern as damage)
        age_severity_wrc = (0.15 + ((age - 33).clip(0) * 0.05)).clip(0.15, 0.40)
        is_wrc_spike = (age >= 33) & (wrc_spike > 10)
        spike_factor_wrc = (1.0 - (wrc_spike - 10) / 15 * age_severity_wrc).clip(0.35, 1.0)
        aging_spike_prod = np.where(is_wrc_spike, spike_factor_wrc, 1.0)

    production_trust = (
        _stat_family_trust(pa, min_pa=150, full_pa=500)
        * age_trust * production_dev_trust * aging_spike_prod
    )

    merged["production_skill"] = (
        (1 - production_trust) * proj_production
        + production_trust * obs_production
    )

    # =================================================================
    # Combine buckets into offense_score
    # Production anchor (35%) ensures actual run output dominates.
    # Contact + decisions (25%) reward projectable skills but can't
    # carry a player whose results don't follow.
    # =================================================================
    merged["offense_score"] = (
        0.15 * merged["contact_skill"]
        + 0.10 * merged["decision_skill"]
        + 0.40 * merged["damage_skill"]
        + 0.35 * merged["production_skill"]
    )

    # Small-sample dampening toward league average (same as before)
    pa_confidence = ((pa - 150) / (400 - 150)).clip(0, 1)
    dampening = 0.50 * (1.0 - pa_confidence)
    merged["offense_score"] = merged["offense_score"] * (1 - dampening) + 0.50 * dampening

    return merged[["batter_id", "offense_score", "contact_skill",
                    "decision_skill", "damage_skill", "production_skill"]]


def _build_hitter_baserunning_score(season: int = 2025) -> pd.DataFrame:
    """Score baserunning from speed, hustle, SB volume, efficiency, and utilization.

    Components
    ----------
    - **Sprint speed** (25%): raw foot speed from Statcast
    - **HP to 1B hustle** (15%): home-to-first time (faster = better).
      Captures effort on routine plays beyond raw speed.
    - **SB volume** (25%): projected SB per PA — aggressive baserunning.
    - **SB efficiency** (15%): success rate regressed toward league avg (~78%).
      Smart baserunning — not just fast, but picks good spots.
    - **Speed utilization** (20%): SB rate relative to sprint-speed expectation.
      A player who steals more than their speed predicts gets credit for
      baserunning IQ and aggressiveness beyond raw ability.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, baserunning_score.
    """
    from src.data.db import read_sql

    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # Sprint speed + HP-to-1B from Statcast
    speed_df = read_sql(f"""
        SELECT player_id AS batter_id, sprint_speed, hp_to_1b
        FROM staging.statcast_sprint_speed
        WHERE season = {season}
    """, {})

    # Observed SB/CS (2-year window for stability)
    sb_obs = read_sql(f"""
        SELECT pg.player_id AS batter_id,
               SUM(pg.bat_sb) AS obs_sb, SUM(pg.bat_cs) AS obs_cs
        FROM production.fact_player_game_mlb pg
        JOIN production.dim_game dg ON pg.game_pk = dg.game_pk
        WHERE dg.season BETWEEN {season - 1} AND {season}
          AND dg.game_type = 'R'
        GROUP BY pg.player_id
    """, {})

    # Projected SB from sim
    sim_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
    old_path = DASHBOARD_DIR / "hitter_counting.parquet"
    if sim_path.exists():
        counting = pd.read_parquet(sim_path)
        sb_col = "total_sb_mean"
        pa_col = "total_pa_mean" if "total_pa_mean" in counting.columns else "projected_pa_mean"
    elif old_path.exists():
        counting = pd.read_parquet(old_path)
        sb_col = "total_sb_mean"
        pa_col = "projected_pa_mean"
    else:
        counting = pd.DataFrame()

    # Assemble
    merged = proj[["batter_id", "sprint_speed"]].copy()
    merged = merged.merge(speed_df, on="batter_id", how="left", suffixes=("_proj", ""))
    # Prefer Statcast sprint_speed over projection parquet
    if "sprint_speed_proj" in merged.columns:
        merged["sprint_speed"] = merged["sprint_speed"].fillna(merged["sprint_speed_proj"])
        merged.drop(columns=["sprint_speed_proj"], inplace=True)
    merged = merged.merge(sb_obs, on="batter_id", how="left")
    if not counting.empty:
        merged = merged.merge(
            counting[["batter_id", sb_col, pa_col]], on="batter_id", how="left",
        )

    if merged.empty:
        return pd.DataFrame(columns=["batter_id", "baserunning_score"])

    # --- Component 1: Sprint speed (25%) ---
    speed_med = merged["sprint_speed"].median()
    speed_score = _pctl(merged["sprint_speed"].fillna(speed_med))

    # --- Component 2: HP-to-1B hustle (15%) ---
    # Lower = faster = better.  Captures effort beyond raw speed.
    hp1b_med = merged["hp_to_1b"].median() if "hp_to_1b" in merged.columns else 4.45
    hp1b_score = _inv_pctl(merged["hp_to_1b"].fillna(hp1b_med))

    # --- Component 3: SB volume rate (25%) ---
    if sb_col in merged.columns and pa_col in merged.columns:
        sb_rate = merged[sb_col] / merged[pa_col].clip(lower=1)
    else:
        sb_rate = merged["obs_sb"].fillna(0) / 500  # rough fallback
    sb_volume_score = _pctl(sb_rate)

    # --- Component 4: SB efficiency (15%) ---
    # Regress toward league average (~78%) using Bayesian shrinkage
    _LEAGUE_SB_SUCCESS = 0.78
    _SB_REGRESS_N = 10  # regress with 10 pseudo-attempts at league avg
    obs_sb = merged["obs_sb"].fillna(0)
    obs_cs = merged["obs_cs"].fillna(0)
    obs_att = obs_sb + obs_cs
    regressed_success = (
        (obs_sb + _SB_REGRESS_N * _LEAGUE_SB_SUCCESS)
        / (obs_att + _SB_REGRESS_N)
    )
    # Only score efficiency for players with enough attempts
    has_attempts = obs_att >= 3
    sb_eff_score = _pctl(regressed_success)
    # Neutral (0.50) for players with < 3 attempts
    sb_eff_score = np.where(has_attempts, sb_eff_score, 0.50)

    # --- Component 5: Speed utilization (20%) ---
    # How much does this player steal relative to their sprint speed?
    # Fit linear model: expected SB rate = f(sprint_speed)
    valid = merged[merged["sprint_speed"].notna() & (sb_rate > 0)].copy()
    if len(valid) >= 20:
        from numpy.polynomial import polynomial as P
        coef = P.polyfit(valid["sprint_speed"].values, sb_rate[valid.index].values, 1)
        expected_sb_rate = P.polyval(merged["sprint_speed"].fillna(speed_med).values, coef)
        sb_over_expected = sb_rate - pd.Series(expected_sb_rate, index=merged.index)
        utilization_score = _pctl(sb_over_expected)
    else:
        utilization_score = pd.Series(0.50, index=merged.index)

    merged["baserunning_score"] = (
        0.25 * speed_score
        + 0.15 * hp1b_score
        + 0.25 * sb_volume_score
        + 0.15 * pd.Series(sb_eff_score, index=merged.index)
        + 0.20 * utilization_score
    )

    return merged[["batter_id", "baserunning_score"]]


def _build_hitter_platoon_modifier(season: int = 2025, min_pa_side: int = 30) -> pd.DataFrame:
    """Score platoon balance with regressed splits and exposure weighting.

    Uses 3-year PA-weighted platoon data with handedness-specific regression:
    - LHH: regress toward league avg OPS with 1000 PA shrinkage
    - RHH: regress toward league avg OPS with 2200 PA shrinkage
    Hitters do NOT learn to close platoon splits over time (research consensus).

    Exposure-weighted: weak vs LHP (28% of games) penalized less than
    weak vs RHP (72% of games).

    Parameters
    ----------
    season : int
        Most recent season for platoon data.
    min_pa_side : int
        Minimum total PA vs each side across all years to be scored.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, platoon_score (0–1, 1 = balanced).
    """
    from src.data.db import read_sql

    # 3-year window for stability
    splits = read_sql(f"""
        SELECT player_id, platoon_side, season, pa, ops
        FROM production.fact_platoon_splits
        WHERE season BETWEEN {season - 2} AND {season}
          AND player_role = 'batter'
    """, {})

    if splits.empty:
        return pd.DataFrame(columns=["player_id", "platoon_score"])

    # Get batter handedness for regression calibration
    handedness = read_sql("""
        SELECT player_id, COALESCE(bat_side, 'R') AS bat_side
        FROM production.dim_player
    """, {})

    # PA-weighted average OPS per player per side (across years)
    def _pa_weighted_ops(g: pd.DataFrame) -> pd.Series:
        total_pa = g["pa"].sum()
        if total_pa == 0:
            return pd.Series({"ops": np.nan, "pa": 0})
        return pd.Series({
            "ops": np.average(g["ops"], weights=g["pa"]),
            "pa": total_pa,
        })

    vlh = (
        splits[splits["platoon_side"] == "vLH"]
        .groupby("player_id")
        .apply(_pa_weighted_ops, include_groups=False)
        .rename(columns={"ops": "ops_vlh", "pa": "pa_vlh"})
    )
    vrh = (
        splits[splits["platoon_side"] == "vRH"]
        .groupby("player_id")
        .apply(_pa_weighted_ops, include_groups=False)
        .rename(columns={"ops": "ops_vrh", "pa": "pa_vrh"})
    )

    merged = vlh.join(vrh, how="inner").reset_index()
    merged = merged.merge(handedness, on="player_id", how="left")
    merged["bat_side"] = merged["bat_side"].fillna("R")

    # Only score players with enough PA on both sides
    has_both = (merged["pa_vlh"] >= min_pa_side) & (merged["pa_vrh"] >= min_pa_side)
    scored = merged[has_both].copy()

    if scored.empty:
        return pd.DataFrame(columns=["player_id", "platoon_score"])

    # League average OPS (~.710-.730 range)
    lg_avg_ops = 0.720

    # Handedness-specific regression (Tango research):
    # LHH splits more variable → less PA needed → shrinkage = 1000
    # RHH splits more stable → need more data to trust extreme → shrinkage = 2200
    shrinkage = np.where(scored["bat_side"] == "L", 1000, 2200)

    # Regress each side toward league average
    scored["reg_ops_vlh"] = (
        scored["ops_vlh"] * scored["pa_vlh"] + lg_avg_ops * shrinkage
    ) / (scored["pa_vlh"] + shrinkage)
    scored["reg_ops_vrh"] = (
        scored["ops_vrh"] * scored["pa_vrh"] + lg_avg_ops * shrinkage
    ) / (scored["pa_vrh"] + shrinkage)

    # Regressed gap
    scored["ops_gap"] = (scored["reg_ops_vlh"] - scored["reg_ops_vrh"]).abs()

    # Exposure weighting: penalty is proportional to how often you face your weak side
    # LHP ~28% of games, RHP ~72%
    weak_vs_lhp = scored["reg_ops_vlh"] < scored["reg_ops_vrh"]
    # If weak vs LHP (28% exposure), gap impact is muted
    # If weak vs RHP (72% exposure), gap impact is amplified
    exposure = np.where(weak_vs_lhp, 0.28, 0.72)
    scored["adjusted_gap"] = scored["ops_gap"] * exposure * 2  # rescale to preserve range

    # Convert to 0-1 score: no gap = 1.0, 250+ adjusted gap = 0.0
    scored["platoon_score"] = (1.0 - scored["adjusted_gap"] / 0.250).clip(0, 1)

    # Players without enough PA get neutral
    neutral = merged[~has_both][["player_id"]].copy()
    neutral["platoon_score"] = 0.50

    result = pd.concat([scored[["player_id", "platoon_score"]], neutral], ignore_index=True)
    return result


def _build_hitter_fielding_score(season: int = 2025) -> pd.DataFrame:
    """Build fielding score from multi-year OAA with position-aware regression.

    Uses 3-year rolling OAA with recency weights (3/2/1), then regresses
    toward 0 (position average) based on sample size and position group.
    OF metrics are more stable (k=2), IF noisier (k=3).

    Parameters
    ----------
    season : int
        Most recent season for OAA data.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, fielding_score.
    """
    from src.data.db import read_sql

    oaa = read_sql(f"""
        SELECT player_id, season, position, outs_above_average
        FROM production.fact_fielding_oaa
        WHERE season BETWEEN {season - 2} AND {season}
    """, {})

    if oaa.empty:
        return pd.DataFrame(columns=["player_id", "fielding_score"])

    # Recency weights: most recent season = 3, year-1 = 2, year-2 = 1
    oaa["weight"] = oaa["season"].map(
        {season: 3, season - 1: 2, season - 2: 1}
    ).fillna(1)

    # Position-group regression constants (higher k = more regression)
    # OF OAA is more stable (larger sample of chances); IF is noisier
    _IF_POSITIONS = {"SS", "2B", "3B", "1B"}
    _OF_POSITIONS = {"LF", "CF", "RF"}

    # Determine primary position group per player (mode of positions played)
    pos_mode = (
        oaa.groupby("player_id")["position"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "IF")
    )

    # Weighted average OAA per player
    weighted = oaa.groupby("player_id").apply(
        lambda g: np.average(g["outs_above_average"], weights=g["weight"]),
        include_groups=False,
    ).rename("weighted_oaa")

    # Count seasons with data per player (for reliability)
    n_seasons = oaa.groupby("player_id")["season"].nunique().rename("n_seasons")

    result = pd.DataFrame({"weighted_oaa": weighted, "n_seasons": n_seasons})
    result["position_group"] = pos_mode

    # Regression constant: k=2 for OF (more stable), k=3 for IF (noisier)
    result["k"] = result["position_group"].apply(
        lambda p: 2 if p in _OF_POSITIONS else 3
    )

    # Position-specific OAA priors (empirical means, 2023-2025).
    # Unknown fielders regress toward their POSITION average, not 0.
    # This prevents unproven 1B from getting neutral (50th pctl) scores
    # when the typical 1B is below average defensively.
    _POSITION_OAA_PRIOR = {
        "CF": 2.8, "SS": 1.1, "2B": 0.2, "3B": -0.5,
        "1B": -0.9, "RF": -1.1, "LF": -1.4,
    }
    result["pos_prior"] = result["position_group"].map(_POSITION_OAA_PRIOR).fillna(0.0)

    # Reliability: n_seasons / (n_seasons + k)
    # 1 year IF: 1/4=0.25, 3 years IF: 3/6=0.50
    # 1 year OF: 1/3=0.33, 3 years OF: 3/5=0.60
    result["reliability"] = result["n_seasons"] / (result["n_seasons"] + result["k"])
    result["regressed_oaa"] = (
        result["reliability"] * result["weighted_oaa"]
        + (1 - result["reliability"]) * result["pos_prior"]
    )

    # Percentile rank the regressed OAA, with floor so worst defenders
    # aren't zeroed out (even -15 OAA provides some value vs empty position)
    result["fielding_score"] = _pctl(result["regressed_oaa"]).clip(lower=0.10)

    return result.reset_index()[["player_id", "fielding_score"]]


def _build_catcher_framing_score(season: int = 2025) -> pd.DataFrame:
    """Build catcher framing score from multi-year framing data.

    Uses 3-year rolling with recency weights (3/2/1) and regression
    toward league average, matching the pattern in fielding.  Single-season
    framing is noisy (Will Smith: +9.1 in 2021, -7.6 in 2025).

    Parameters
    ----------
    season : int
        Most recent season for framing data.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, framing_score.
    """
    from src.data.db import read_sql

    framing = read_sql(f"""
        SELECT player_id, season, runs_extra_strikes
        FROM production.fact_catcher_framing
        WHERE season BETWEEN {season - 2} AND {season}
    """, {})

    if framing.empty:
        return pd.DataFrame(columns=["player_id", "framing_score"])

    # Recency weights
    framing["weight"] = framing["season"].map(
        {season: 3, season - 1: 2, season - 2: 1}
    ).fillna(1)

    # Weighted average per catcher
    weighted = framing.groupby("player_id").apply(
        lambda g: np.average(g["runs_extra_strikes"], weights=g["weight"]),
        include_groups=False,
    ).rename("weighted_framing")

    n_seasons = framing.groupby("player_id")["season"].nunique().rename("n_seasons")
    result = pd.DataFrame({"weighted_framing": weighted, "n_seasons": n_seasons})

    # Regress toward 0 (league-average framing) based on sample size
    # k=2: 1 year → 33% trust, 2 years → 50%, 3 years → 60%
    k = 2
    result["reliability"] = result["n_seasons"] / (result["n_seasons"] + k)
    result["regressed_framing"] = result["weighted_framing"] * result["reliability"]

    result["framing_score"] = _pctl(result["regressed_framing"])
    return result.reset_index()[["player_id", "framing_score"]]


def _build_hitter_playing_time_score() -> pd.DataFrame:
    """Score projected playing time from counting projections + health.

    Prefers sim-based ``hitter_counting_sim.parquet`` (PA + games from
    game sim). Falls back to old ``hitter_counting.parquet``.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, pt_score, health_score, health_label.
    """
    sim_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
    old_path = DASHBOARD_DIR / "hitter_counting.parquet"

    if sim_path.exists():
        counting = pd.read_parquet(sim_path)
        pa_col = "total_pa_mean" if "total_pa_mean" in counting.columns else "projected_pa_mean"
        games_col = "total_games_mean"
        if pa_col in counting.columns and games_col in counting.columns:
            pa_pctl = _pctl(counting[pa_col].fillna(0))
            games_pctl = _pctl(counting[games_col].fillna(0))
            base_pt = 0.60 * pa_pctl + 0.40 * games_pctl
        else:
            base_pt = _pctl(counting[pa_col].fillna(0))
        logger.info("Playing time score from sim parquet: %d hitters", len(counting))
    elif old_path.exists():
        counting = pd.read_parquet(old_path)
        base_pt = _pctl(counting["projected_pa_mean"])
    else:
        return pd.DataFrame(columns=["batter_id", "pt_score"])

    # Rename for downstream compatibility
    if "projected_pa_mean" not in counting.columns and "total_pa_mean" in counting.columns:
        counting["projected_pa_mean"] = counting["total_pa_mean"]

    # Health scores: prefer columns already on counting parquet,
    # fall back to standalone health_scores.parquet
    has_health = "health_score" in counting.columns and counting["health_score"].notna().any()
    if not has_health:
        health_path = DASHBOARD_DIR / "health_scores.parquet"
        if health_path.exists():
            health_df = pd.read_parquet(health_path)
            counting = counting.merge(
                health_df[["player_id", "health_score", "health_label"]].rename(
                    columns={"player_id": "batter_id"}
                ),
                on="batter_id",
                how="left",
            )
            has_health = True

    if has_health:
        counting["health_score"] = counting["health_score"].fillna(0.85)
        counting["health_label"] = counting["health_label"].fillna("Unknown")
        health_pctl = _pctl(counting["health_score"])
        counting["pt_score"] = 0.70 * base_pt + 0.30 * health_pctl
    else:
        counting["pt_score"] = base_pt
        counting["health_score"] = np.nan
        counting["health_label"] = ""

    cols = ["batter_id", "pt_score", "health_score", "health_label"]
    return counting[[c for c in cols if c in counting.columns]]


def _build_hitter_trajectory_score() -> pd.DataFrame:
    """Score trajectory using posterior certainty, age, and proven status.

    Designed so that elite proven players (like a 33-year-old with 8
    years of elite data) score at least neutral on trajectory.  Uses
    coefficient of variation (SD/mean) instead of raw SD for certainty,
    so high-K% hitters aren't mechanically penalized by wider rate-scale
    SDs.  Does NOT include rate quality — that's offense_score's job.

    Components
    ----------
    - **Projection certainty** (55%): lower CV = more proven.  Uses
      SD/mean on rate scale, which normalizes for the logit-scale
      stretching that makes high-K% hitters look artificially uncertain.
    - **Age factor** (25%): younger players get upside credit.  Decline
      penalty starts at 33 (not 31) and is softer — proven veterans
      shouldn't be destroyed for being in their early 30s.
    - **Season trend** (20%): if the Bayesian projection improves on the
      prior season's observed rate, that's a positive signal.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, trajectory_score.
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # Posterior certainty via coefficient of variation (SD / mean)
    # CV normalizes for rate-dependent SD scaling — a 26% K hitter with
    # 3.6pp SD has CV=0.14, comparable to a 15% K hitter with 2.1pp SD
    k_mean = proj["projected_k_rate"].clip(0.01)
    bb_mean = proj["projected_bb_rate"].clip(0.01)
    k_cv = proj["projected_k_rate_sd"].fillna(k_mean.median() * 0.15) / k_mean
    bb_cv = proj["projected_bb_rate_sd"].fillna(bb_mean.median() * 0.15) / bb_mean
    k_certainty = _inv_pctl(k_cv)
    bb_certainty = _inv_pctl(bb_cv)
    certainty_score = 0.50 * k_certainty + 0.50 * bb_certainty

    # Age factor: research-aligned curve (peak 26-29, decline from 29, accel 35+)
    age = proj["age"].fillna(30)
    age_factor = _hitter_age_factor(age)

    # Season trend: did projected rate improve on observed?
    # Uses K% delta (negative = improving) and BB% delta (positive = improving)
    delta_k = proj.get("delta_k_rate", pd.Series(0.0, index=proj.index))
    delta_bb = proj.get("delta_bb_rate", pd.Series(0.0, index=proj.index))
    k_trend = _inv_pctl(delta_k.fillna(0))   # lower delta = better (K% dropping)
    bb_trend = _pctl(delta_bb.fillna(0))      # higher delta = better (BB% rising)
    trend_score = 0.50 * k_trend + 0.50 * bb_trend

    proj["trajectory_score"] = (
        0.55 * certainty_score
        + 0.25 * age_factor
        + 0.20 * trend_score
    )
    return proj[["batter_id", "trajectory_score"]]


def _build_versatility_score(base: pd.DataFrame) -> pd.Series:
    """Positional versatility bonus (0-1 scale).

    Players eligible at multiple premium positions (SS, C, CF) are more
    valuable than single-position players at low-value spots (1B/DH).

    Parameters
    ----------
    base : pd.DataFrame
        Hitter base DataFrame (must contain ``batter_id``).

    Returns
    -------
    pd.Series
        Versatility score aligned with ``base.index``.
    """
    try:
        elig = pd.read_parquet(DASHBOARD_DIR / "hitter_position_eligibility.parquet")
        # Count positions and sum tier values per player
        player_elig = elig.groupby("player_id").agg(
            n_positions=("position", "nunique"),
            tier_sum=("position", lambda x: sum(_POS_TIER.get(p, 0) for p in x)),
        )
        # Normalize: 1 position = 0.0, 4+ positions with premium = 1.0
        player_elig["versatility"] = (
            (player_elig["n_positions"] - 1) * 0.15
            + player_elig["tier_sum"] * 0.05
        ).clip(0, 1)
        return base["batter_id"].map(player_elig["versatility"]).fillna(0.0)
    except FileNotFoundError:
        logger.warning("Position eligibility parquet not found — versatility scores set to 0")
        return pd.Series(0.0, index=base.index)


def _build_roster_construction_score(base: pd.DataFrame) -> pd.Series:
    """Roster scarcity bonus -- players at thin positions get a boost.

    For each team, counts how many ranked players are eligible at each
    position.  A player at a position with thin depth (1-2 eligible)
    gets a small value boost; deep positions (3+) get no bonus.

    Parameters
    ----------
    base : pd.DataFrame
        Hitter base DataFrame (must contain ``batter_id``, ``position``).

    Returns
    -------
    pd.Series
        Roster construction score aligned with ``base.index``.
    """
    try:
        elig = pd.read_parquet(DASHBOARD_DIR / "hitter_position_eligibility.parquet")
        player_teams_df = pd.read_parquet(DASHBOARD_DIR / "player_teams.parquet")

        # Map player -> team
        pid_team = dict(zip(player_teams_df["player_id"], player_teams_df["team_abbr"]))

        # For each team-position combo, count eligible ranked players
        ranked_pids = set(base["batter_id"])
        elig_ranked = elig[elig["player_id"].isin(ranked_pids)].copy()
        elig_ranked["team"] = elig_ranked["player_id"].map(pid_team)

        team_pos_depth = elig_ranked.groupby(["team", "position"])["player_id"].nunique()

        # For each player, find their primary position's depth on their team
        scores: dict[int, float] = {}
        for _, row in base.iterrows():
            pid = row["batter_id"]
            pos = row["position"]
            team = pid_team.get(pid, "")
            depth = team_pos_depth.get((team, pos), 1)
            # Thin depth (1-2 players) = higher value; deep (4+) = no bonus
            scores[pid] = max(0, (3 - depth) / 3)  # 1 player = 0.67, 2 = 0.33, 3+ = 0

        return base["batter_id"].map(scores).fillna(0.0)
    except Exception:
        logger.warning("Could not compute roster construction scores — set to 0")
        return pd.Series(0.0, index=base.index)


def rank_hitters(
    season: int = 2025,
    projection_season: int = 2026,
    min_pa: int = 100,
) -> pd.DataFrame:
    """Rank all hitters by position for 2026 value.

    Parameters
    ----------
    season : int
        Most recent completed season (for observed stats).
    projection_season : int
        Target projection season.
    min_pa : int
        Minimum PA in observed season to qualify.

    Returns
    -------
    pd.DataFrame
        Positional rankings with composite score, sub-scores,
        and key stats. Sorted by position then rank.
    """
    from src.data.db import read_sql

    # Load projections
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")

    # Load observed stats
    observed = read_sql(f"""
        SELECT batter_id, pa, k_pct, bb_pct, woba, wrc_plus,
               xwoba, xba, xslg, barrel_pct, hard_hit_pct
        FROM production.fact_batting_advanced
        WHERE season = {season} AND pa >= {min_pa}
    """, {})

    # Park-adjust wOBA (xwOBA/barrel%/hard_hit% are already park-neutral)
    try:
        from src.data.park_factors import compute_multi_stat_park_factors, get_player_park_adjustments
        from src.data.queries import get_hitter_team_venue
        park_factors = compute_multi_stat_park_factors(
            seasons=[season - 2, season - 1, season],
        )
        player_venues = get_hitter_team_venue(season)
        if not park_factors.empty and not player_venues.empty:
            pf_map = get_player_park_adjustments(
                park_factors,
                player_venues.rename(columns={"batter_id": "player_id"}),
            )
            pf_df = pd.DataFrame([
                {"batter_id": pid, "pf_r": v.get("pf_r", 1.0)}
                for pid, v in pf_map.items()
            ])
            observed = observed.merge(pf_df, on="batter_id", how="left")
            observed["pf_r"] = observed["pf_r"].fillna(1.0)
            observed["woba_raw"] = observed["woba"]
            observed["woba"] = observed["woba"] / observed["pf_r"]
            logger.info("Park-adjusted wOBA for %d hitters", (observed["pf_r"] != 1.0).sum())
    except Exception:
        logger.warning("Could not load park factors — using raw wOBA", exc_info=True)

    # Multi-year observed stats (recency-weighted) for production and damage
    # buckets.  Single-season observed overweights outlier years (Springer's
    # career-best at 35, Grisham's breakout at 28).  Multi-year smoothing
    # uses the same 3/2/1 pattern as fielding.
    try:
        obs_multi = read_sql(f"""
            SELECT batter_id, season, pa, wrc_plus,
                   xwoba, barrel_pct, hard_hit_pct
            FROM production.fact_batting_advanced
            WHERE season BETWEEN {season - 2} AND {season} AND pa >= 50
        """, {})
        if not obs_multi.empty:
            # Flatten recency weights for age 33+: a 35-year-old's
            # career-best most-recent season should not dominate the
            # blend the way a 26-year-old's breakout should.
            # Standard: 3/2/1.  Age 33+: 2/2/1 (recent year loses edge).
            obs_multi = obs_multi.merge(
                proj[["batter_id", "age"]].drop_duplicates("batter_id"),
                on="batter_id", how="left",
            )
            is_aging = obs_multi["age"].fillna(28) >= 33
            base_wt = obs_multi["season"].map(
                {season: 3, season - 1: 2, season - 2: 1}
            ).fillna(1)
            aging_wt = obs_multi["season"].map(
                {season: 2, season - 1: 2, season - 2: 1}
            ).fillna(1)
            obs_multi["recency_wt"] = np.where(is_aging, aging_wt, base_wt)
            obs_multi.drop(columns=["age"], inplace=True)
            obs_multi["total_wt"] = obs_multi["recency_wt"] * obs_multi["pa"]

            def _wtd_avg(g, col):
                valid = g[g[col].notna()]
                if valid.empty or valid["total_wt"].sum() == 0:
                    return np.nan
                return np.average(valid[col], weights=valid["total_wt"])

            multiyear = obs_multi.groupby("batter_id").agg(
                wrc_plus_multiyear=("wrc_plus", lambda g: _wtd_avg(obs_multi.loc[g.index], "wrc_plus")),
                xwoba_multiyear=("xwoba", lambda g: _wtd_avg(obs_multi.loc[g.index], "xwoba")),
                barrel_pct_multiyear=("barrel_pct", lambda g: _wtd_avg(obs_multi.loc[g.index], "barrel_pct")),
                hard_hit_pct_multiyear=("hard_hit_pct", lambda g: _wtd_avg(obs_multi.loc[g.index], "hard_hit_pct")),
                obs_years=("season", "nunique"),
            ).reset_index()
            observed = observed.merge(multiyear, on="batter_id", how="left")
            logger.info("Multi-year observed stats: %d hitters with 2+ years",
                        (observed["obs_years"].fillna(1) >= 2).sum())
    except Exception:
        logger.warning("Could not load multi-year observed stats", exc_info=True)

    # Position assignments
    positions = _assign_hitter_positions(season=season)

    # Load aggressiveness data for discipline modifier
    from src.data.queries import get_hitter_aggressiveness
    aggressiveness = get_hitter_aggressiveness(season)

    # Build sub-scores
    offense = _build_hitter_offense_score(proj, observed, aggressiveness=aggressiveness)
    baserunning = _build_hitter_baserunning_score(season=season)
    platoon = _build_hitter_platoon_modifier(season=season)
    fielding = _build_hitter_fielding_score(season=season)
    framing = _build_catcher_framing_score(season=season)
    playing_time = _build_hitter_playing_time_score()
    trajectory = _build_hitter_trajectory_score()

    # Start from projections as base (has batter_id, name, age, etc.)
    base = proj[["batter_id", "batter_name", "age", "batter_stand",
                 "projected_k_rate", "projected_bb_rate", "projected_hr_per_fb",
                 "projected_k_rate_sd", "projected_bb_rate_sd",
                 "composite_score"]].copy()

    # Merge observed stats for display (include park-adjusted columns if available)
    obs_cols = ["batter_id", "pa", "woba", "wrc_plus", "xwoba",
                "xba", "xslg", "barrel_pct", "hard_hit_pct"]
    for extra in ["woba_raw", "pf_r"]:
        if extra in observed.columns:
            obs_cols.append(extra)
    base = base.merge(observed[obs_cols], on="batter_id", how="inner")

    # Merge aggressiveness data for discipline modifier and display
    if not aggressiveness.empty:
        base = base.merge(
            aggressiveness[["batter_id", "chase_rate", "two_strike_whiff_rate"]],
            on="batter_id", how="left",
        )

    # Merge position
    base = base.merge(
        positions.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="inner",
    )

    # Merge sub-scores
    base = base.merge(offense, on="batter_id", how="left")
    base = base.merge(baserunning, on="batter_id", how="left")
    base = base.merge(
        platoon.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="left",
    )
    base = base.merge(
        fielding.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="left",
    )
    base = base.merge(
        framing.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="left",
    )
    base = base.merge(playing_time, on="batter_id", how="left")
    base = base.merge(trajectory, on="batter_id", how="left")

    # Merge sim-based counting projections for display
    sim_path = DASHBOARD_DIR / "hitter_counting_sim.parquet"
    if sim_path.exists():
        sim_df = pd.read_parquet(sim_path)
        sim_cols = ["batter_id", "total_k_mean", "total_bb_mean", "total_hr_mean",
                     "total_h_mean", "total_r_mean", "total_rbi_mean", "total_sb_mean",
                     "projected_woba_mean", "projected_wrc_plus_mean",
                     "projected_wraa_mean", "projected_ops_mean", "projected_avg_mean",
                     "dk_season_mean", "espn_season_mean", "total_games_mean", "total_pa_mean"]
        sim_cols = [c for c in sim_cols if c in sim_df.columns]
        base = base.merge(sim_df[sim_cols], on="batter_id", how="left")
        logger.info("Merged hitter sim projections for %d batters",
                     base["projected_wrc_plus_mean"].notna().sum() if "projected_wrc_plus_mean" in base.columns else 0)

    # Breakout archetype data (GMM-derived)
    breakout_path = DASHBOARD_DIR / "hitter_breakout_candidates.parquet"
    if breakout_path.exists():
        breakout_df = pd.read_parquet(breakout_path)
        breakout_cols = [
            "batter_id", "breakout_type", "breakout_score",
            "breakout_tier", "breakout_hole", "gmm_fit",
            "prob_power_surge", "prob_diamond_in_the_rough",
        ]
        available_bc = [c for c in breakout_cols if c in breakout_df.columns]
        base = base.merge(breakout_df[available_bc], on="batter_id", how="left")
    else:
        for col in ["breakout_type", "breakout_tier", "breakout_hole"]:
            base[col] = ""
        for col in ["breakout_score", "gmm_fit"]:
            base[col] = np.nan

    # Fill missing with neutral
    base["fielding_score"] = base["fielding_score"].fillna(0.50)
    base["framing_score"] = base["framing_score"].fillna(0.50)
    base["offense_score"] = base["offense_score"].fillna(0.50)
    base["baserunning_score"] = base["baserunning_score"].fillna(0.50)
    base["platoon_score"] = base["platoon_score"].fillna(0.50)

    # Discipline modifier: low chase + low 2-strike whiff = more valuable
    if "chase_rate" in base.columns:
        chase_pctl = _inv_pctl(base["chase_rate"].fillna(base["chase_rate"].median()))
        whiff_2s_pctl = _inv_pctl(base["two_strike_whiff_rate"].fillna(base["two_strike_whiff_rate"].median()))
        discipline_score = 0.60 * chase_pctl + 0.40 * whiff_2s_pctl
        discipline_modifier = 0.95 + 0.10 * discipline_score  # scales ±5%
        base["offense_score"] = (base["offense_score"] * discipline_modifier).clip(0, 1)

    # Apply platoon modifier to offense: balanced hitters get a boost,
    # extreme-split hitters get penalized. Scales offense by ±10%
    # (widened from ±5% — industry research shows extreme splits = 1+ WAR penalty).
    platoon_modifier = 0.90 + 0.20 * base["platoon_score"]  # 0.90 to 1.10
    base["offense_score"] = (base["offense_score"] * platoon_modifier).clip(0, 1)
    base["pt_score"] = base["pt_score"].fillna(0.50)
    base["trajectory_score"] = base["trajectory_score"].fillna(0.50)

    # Sustain/upside blend into trajectory.
    # Replaces pure breakout blend which penalized elite established players
    # (Soto, Judge) who have no "room to break out" by definition.
    # sustain_upside = max(breakout_pctl, offense_score):
    #   - Developing players (Elly, J-Rod): breakout_pctl drives the score
    #   - Elite players (Soto, Judge): offense_score provides a high floor
    # Net effect: trajectory rewards both paths to value — improving OR
    # sustaining elite production.  Minor offense double-counting (~3.5%
    # of total composite) is acceptable.
    base["breakout_score"] = base["breakout_score"].fillna(0.0)
    breakout_pctl = _pctl(base["breakout_score"].clip(lower=0))
    raw_trajectory = base["trajectory_score"].copy()  # save for upside calc
    sustain_upside = np.maximum(breakout_pctl, base["offense_score"])
    base["trajectory_score"] = (
        0.65 * base["trajectory_score"] + 0.35 * sustain_upside
    )

    if "health_score" not in base.columns:
        base["health_score"] = np.nan
        base["health_label"] = ""

    # --- Health & role scores (replace old playing_time composite) ---
    base["health_adj"] = base["health_score"].fillna(0.50)
    # Role score: everyday (140+ games) = 1.0, platoon (70) = 0.5, bench (30) = 0.21
    if "total_games_mean" in base.columns:
        base["role_score"] = (base["total_games_mean"].fillna(0) / 140).clip(0, 1)
    else:
        # Fallback to pt_score which already captures playing time
        base["role_score"] = base["pt_score"]

    # --- Versatility & roster construction ---
    base["versatility_score"] = _build_versatility_score(base)
    base["roster_value_score"] = _build_roster_construction_score(base)

    # --- Glicko-2 opponent-adjusted performance ---
    glicko = _load_glicko_scores("batter")
    if not glicko.empty:
        base = base.merge(glicko, on="batter_id", how="left")
    if "glicko_score" not in base.columns:
        base["glicko_score"] = 0.5  # neutral default
    base["glicko_score"] = base["glicko_score"].fillna(0.5)

    # --- Composite score ---
    # For catchers: blend framing into fielding component
    is_catcher = base["position"] == "C"
    base["fielding_combined"] = base["fielding_score"]
    base.loc[is_catcher, "fielding_combined"] = (
        0.50 * base.loc[is_catcher, "fielding_score"]
        + 0.50 * base.loc[is_catcher, "framing_score"]
    )

    # Percentile-rank fielding and baserunning so their effective spread
    # matches offense.  Raw fielding has IQR ~0.66 vs offense ~0.20 — at
    # 16% weight the raw scale gives fielding equal influence to 52% offense.
    # Percentile-ranking compresses both to a uniform 0-1 distribution so
    # the weights mean what they say.
    base["fielding_combined"] = _pctl(base["fielding_combined"])
    base["baserunning_score"] = _pctl(base["baserunning_score"])

    # For DH: redistribute fielding weight to offense + baserunning.
    # DHs are evaluated purely on production — no artificial cap.
    is_dh = base["position"] == "DH"
    dh_offense_wt = _HITTER_WEIGHTS["offense"] + _HITTER_WEIGHTS["fielding"] * 0.70
    dh_baserunning_wt = _HITTER_WEIGHTS["baserunning"] + _HITTER_WEIGHTS["fielding"] * 0.30

    # Dynamic fielding dampening for elite hitters.
    # If you're so good with the bat, teams will hide you at DH/1B/corner OF.
    # Worst fielder costs ~-15 runs; best hitter adds ~+50 runs — fielding
    # range is ~30% of offense range, so elite bats should have reduced
    # fielding penalty.  Kicks in above 0.70 offense, halves fielding
    # weight at 1.00 offense.  Redistributed weight goes to offense.
    offense_excess = (base["offense_score"] - 0.70).clip(0) / 0.30
    fielding_dampening = 0.50 * offense_excess
    eff_fielding_wt = _HITTER_WEIGHTS["fielding"] * (1 - fielding_dampening)
    eff_offense_boost = _HITTER_WEIGHTS["fielding"] * fielding_dampening

    # Standard composite (non-DH)
    base["tdd_value_score"] = (
        (_HITTER_WEIGHTS["offense"] + eff_offense_boost) * base["offense_score"]
        + _HITTER_WEIGHTS["baserunning"] * base["baserunning_score"]
        + eff_fielding_wt * base["fielding_combined"]
        + _HITTER_WEIGHTS["health"] * base["health_adj"]
        + _HITTER_WEIGHTS["role"] * base["role_score"]
        + _HITTER_WEIGHTS["trajectory"] * base["trajectory_score"]
        + _HITTER_WEIGHTS["versatility"] * base["versatility_score"]
        + _HITTER_WEIGHTS["roster_value"] * base["roster_value_score"]
    )
    # DH composite: fielding weight fully redistributed (no dampening needed)
    base.loc[is_dh, "tdd_value_score"] = (
        dh_offense_wt * base.loc[is_dh, "offense_score"]
        + dh_baserunning_wt * base.loc[is_dh, "baserunning_score"]
        + _HITTER_WEIGHTS["health"] * base.loc[is_dh, "health_adj"]
        + _HITTER_WEIGHTS["role"] * base.loc[is_dh, "role_score"]
        + _HITTER_WEIGHTS["trajectory"] * base.loc[is_dh, "trajectory_score"]
        + _HITTER_WEIGHTS["versatility"] * base.loc[is_dh, "versatility_score"]
        + _HITTER_WEIGHTS["roster_value"] * base.loc[is_dh, "roster_value_score"]
    )

    # --- Two-way player bonus ---
    # Players who also pitch (e.g. Ohtani) get credit for pitching value.
    # Check both Bayesian pitcher projections and raw pitching stats,
    # since a two-way player may not appear in projections (e.g. missed
    # a full season pitching due to injury).
    base["two_way_bonus"] = 0.0
    base["is_two_way"] = False
    try:
        from src.data.db import read_sql as _read_sql

        pitcher_proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")

        # Also check raw pitching advanced stats for recent seasons
        recent_pitching = _read_sql(f"""
            SELECT pitcher_id, k_pct, bb_pct, swstr_pct, batters_faced
            FROM production.fact_pitching_advanced
            WHERE season >= {season - 2} AND batters_faced >= 100
        """, {})
        # Aggregate across recent seasons (PA-weighted)
        if not recent_pitching.empty:
            rp_agg = (
                recent_pitching.groupby("pitcher_id")
                .apply(
                    lambda g: pd.Series({
                        "k_pct": np.average(g["k_pct"], weights=g["batters_faced"]),
                        "bb_pct": np.average(g["bb_pct"], weights=g["batters_faced"]),
                        "total_bf": g["batters_faced"].sum(),
                    }),
                    include_groups=False,
                )
                .reset_index()
            )
        else:
            rp_agg = pd.DataFrame(columns=["pitcher_id", "k_pct", "bb_pct", "total_bf"])

        # Find two-way players: hitters who appear in pitcher data
        hitter_ids = set(base["batter_id"])
        proj_pitcher_ids = set(pitcher_proj["pitcher_id"]) & hitter_ids
        raw_pitcher_ids = set(rp_agg["pitcher_id"]) & hitter_ids
        two_way_ids = proj_pitcher_ids | raw_pitcher_ids

        if two_way_ids:
            # Use projections if available, raw stats as fallback
            for pid in two_way_ids:
                if pid in proj_pitcher_ids:
                    pp = pitcher_proj[pitcher_proj["pitcher_id"] == pid].iloc[0]
                    k_val = pp["projected_k_rate"]
                    bb_val = pp["projected_bb_rate"]
                    ref_k = pitcher_proj["projected_k_rate"]
                    ref_bb = pitcher_proj["projected_bb_rate"]
                else:
                    pp_raw = rp_agg[rp_agg["pitcher_id"] == pid].iloc[0]
                    k_val = pp_raw["k_pct"]
                    bb_val = pp_raw["bb_pct"]
                    ref_k = rp_agg["k_pct"]
                    ref_bb = rp_agg["bb_pct"]

                k_pctl = (ref_k < k_val).mean()
                bb_pctl = (ref_bb > bb_val).mean()
                pitcher_value = 0.60 * k_pctl + 0.40 * bb_pctl
                # Scale bonus: elite pitcher value (0.8+) adds up to ~0.15
                bonus = pitcher_value * 0.18
                mask = base["batter_id"] == pid
                base.loc[mask, "two_way_bonus"] = bonus
                base.loc[mask, "is_two_way"] = True
                pname = base.loc[mask, "batter_name"].iloc[0]
                logger.info(
                    "Two-way bonus: %s — pitcher value=%.3f, bonus=+%.3f",
                    pname, pitcher_value, bonus,
                )
    except Exception:
        logger.exception("Could not compute two-way player bonus")

    base["tdd_value_score"] = base["tdd_value_score"] + base["two_way_bonus"]

    # --- Two-score architecture: current_value + talent_upside ---
    # Save the production composite before scouting blending.
    # This is the weighted sum of offense, fielding, trajectory, health,
    # role, glicko, etc. — the "what has he done / what will he do" signal.
    production_composite = base["tdd_value_score"].copy()

    # Initialize both scores to production composite (scouting may fail)
    base["current_value_score"] = production_composite
    base["talent_upside_score"] = production_composite

    try:
        from src.models.scouting_grades import grade_hitter_tools
        scouting = grade_hitter_tools(base, season=season)
        if not scouting.empty:
            base = base.merge(scouting, on="batter_id", how="left")
            logger.info("Scouting grades computed for %d hitters", scouting["tools_rating"].notna().sum())

            # Regress diamond rating by PA reliability (same for both scores)
            dr_norm = (base["tools_rating"] / 10.0).clip(0, 1)

            pa_col = base["pa"] if "pa" in base.columns else base.get("total_pa", 400)
            cfg = _RANK_CFG["hitter"]
            reliability = ((pa_col - cfg["pa_ramp_min"]) / (cfg["pa_ramp_max"] - cfg["pa_ramp_min"])).clip(0, 1)

            # Low-PA players: regress diamond rating toward 0.50 (league avg)
            dr_regressed = reliability * dr_norm + (1 - reliability) * 0.50

            # Health penalty on DR: injury-prone players get dampened for
            # current_value (availability matters for near-term), but NOT
            # for talent_upside (tools and health are separate — an injured
            # player's tools don't change, only their playing time does).
            health = base["health_adj"] if "health_adj" in base.columns else 0.50
            health_penalty = np.where(health < 0.40, 0.70 + 0.75 * health, 1.0)
            dr_talent = dr_regressed.copy()       # no health penalty for talent
            dr_regressed = dr_regressed * health_penalty  # health penalty for current value

            # ----- talent_upside_score: scouting-dominant -----
            # Tool grades carry heavy weight for forward-looking assessment.
            # Uses dr_talent (no health penalty — tools and health are separate).
            # No positional multiplier — raw talent regardless of position.
            upside_w = cfg["upside_scout_weight"]
            base["talent_upside_score"] = (
                upside_w * dr_talent + (1 - upside_w) * production_composite
            )

            # ----- current_value_score: production-dominant -----
            # Scouting grades are a feature, not the override.  The weight
            # shrinks with sample size: 30% at 150 PA → 10% at 600+ PA.
            # These defaults are configurable in config/model.yaml and will
            # be learned by Phase 2 walk-forward validation.
            scout_w = _exposure_conditioned_scouting_weight(
                pa_col,
                min_exp=cfg["pa_ramp_min"],
                max_exp=cfg["pa_ramp_max"],
                weight_ceil=cfg["scout_weight_ceil"],
                weight_floor=cfg["scout_weight_floor"],
            )
            base["current_value_score"] = (
                scout_w * dr_regressed + (1 - scout_w) * production_composite
            )

            n_scouted = scouting["tools_rating"].notna().sum()
            logger.info(
                "Two-score split: %d hitters — scout weight range [%.0f%%, %.0f%%]",
                n_scouted, cfg["scout_weight_floor"] * 100, cfg["scout_weight_ceil"] * 100,
            )
    except Exception:
        logger.warning("Could not compute hitter scouting grades", exc_info=True)

    # --- Upside adjustments: breakout trajectory + harsher age curve ---
    # talent_upside_score is the 2-3yr forward view (dynasty rankings).
    # Swap sustain trajectory for breakout: developing players should be
    # rewarded for room to improve, not penalized for not being elite yet.
    # Apply harsher age curve: 2-3yr horizon amplifies decline risk.
    upside_trajectory = 0.65 * raw_trajectory + 0.35 * breakout_pctl
    traj_swap = _HITTER_WEIGHTS["trajectory"] * (upside_trajectory - base["trajectory_score"])
    base["talent_upside_score"] = base["talent_upside_score"] + traj_swap

    # Harsher age: peak 25-27, young players boosted, 30+ penalized on forward horizon
    age = base["age"].fillna(28)
    upside_age_mult = (1.0 + (26 - age) * 0.02).clip(0.76, 1.10)
    # age 21: 1.10, age 24: 1.04, age 26: 1.00, age 30: 0.92, age 33: 0.86, age 38: 0.76
    base["talent_upside_score"] = base["talent_upside_score"] * upside_age_mult

    # DH penalty: DHs don't field, which is a real value gap (~1.75 WAR/season).
    # Applied after all other scoring so it's the final adjustment, not a
    # compounding factor.  Two-way players (Ohtani) exempt — they DH because
    # they pitch, not because they can't field.
    is_pure_dh = (base["position"] == "DH") & (~base["is_two_way"])
    if is_pure_dh.any():
        base.loc[is_pure_dh, "current_value_score"] *= 0.90
        logger.info("DH penalty (10%%): %d pure DHs, %d two-way exempt",
                     is_pure_dh.sum(), (base["is_two_way"] & (base["position"] == "DH")).sum())

    # tdd_value_score = current_value_score (backward compat for team consumption)
    base["tdd_value_score"] = base["current_value_score"]

    # --- Rank within each position ---
    base = base.sort_values("current_value_score", ascending=False)
    base["pos_rank"] = base.groupby("position").cumcount() + 1

    # --- Overall rank: directly from current_value_score ---
    # No positional adjustment — the displayed score IS the ranking order.
    # Positional value is captured by pos_rank within each position.
    base["overall_rank"] = base["current_value_score"].rank(ascending=False, method="min").astype(int)

    # --- Talent-based ranking (scouting-dominant, no positional adj) ---
    base["talent_rank"] = base["talent_upside_score"].rank(ascending=False, method="min").astype(int)

    # Select output columns
    output_cols = [
        "pos_rank", "overall_rank", "talent_rank",
        "batter_id", "batter_name", "position",
        "age", "batter_stand", "is_two_way",
        # Two-score architecture
        "current_value_score", "talent_upside_score",
        "tdd_value_score",
        # Sub-scores
        "offense_score", "contact_skill", "decision_skill", "damage_skill", "production_skill",
        "baserunning_score", "platoon_score",
        "fielding_combined", "framing_score", "pt_score",
        "health_adj", "role_score", "versatility_score", "roster_value_score",
        "trajectory_score", "two_way_bonus",
        # Health
        "health_score", "health_label",
        # Observed
        "pa", "woba", "woba_raw", "pf_r", "wrc_plus", "xwoba", "xba", "xslg",
        "barrel_pct", "hard_hit_pct",
        "chase_rate", "two_strike_whiff_rate",
        # Projected
        "projected_k_rate", "projected_bb_rate", "projected_hr_per_fb",
        "projected_k_rate_sd", "projected_bb_rate_sd",
        # Breakout archetype (GMM-derived)
        "breakout_type", "breakout_score", "breakout_tier",
        "breakout_hole", "gmm_fit",
        "prob_power_surge", "prob_diamond_in_the_rough",
        # Sim-based season projections
        "total_k_mean", "total_bb_mean", "total_hr_mean", "total_h_mean",
        "total_r_mean", "total_rbi_mean", "total_sb_mean",
        "projected_woba_mean", "projected_wrc_plus_mean", "projected_wraa_mean",
        "projected_ops_mean", "projected_avg_mean",
        "dk_season_mean", "espn_season_mean", "total_games_mean", "total_pa_mean",
        # Glicko-2 opponent-adjusted rating
        "glicko_score", "glicko_mu", "glicko_phi",
        # Scouting grades (20-80) + diamond rating (0-10)
        "grade_hit", "grade_power", "grade_speed", "grade_fielding",
        "grade_discipline", "tools_rating",
    ]
    available = [c for c in output_cols if c in base.columns]
    result = base[available].sort_values(["position", "pos_rank"])

    for pos in HITTER_POSITIONS:
        pos_df = result[result["position"] == pos]
        if not pos_df.empty:
            logger.info(
                "%s: %d ranked — #1 %s (%.3f)",
                pos, len(pos_df),
                pos_df.iloc[0]["batter_name"],
                pos_df.iloc[0]["tdd_value_score"],
            )

    return result


# ===================================================================
# Pitcher ranking
# ===================================================================

def _compute_stuff_plus(run_values: pd.DataFrame) -> pd.DataFrame:
    """Compute arsenal-level Stuff+ from per-pitch-type run values.

    Stuff+ = 100 + (league_mean - pitcher_rv) / league_std * 10
    Inverted because negative run value = better for pitcher.
    Arsenal-level = usage-weighted average across pitch types.

    Parameters
    ----------
    run_values : pd.DataFrame
        From ``get_pitcher_run_values``. Must have pitcher_id, pitch_type,
        run_value_per_100, usage_pct columns.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, arsenal_stuff_plus.
    """
    if run_values is None or run_values.empty:
        return pd.DataFrame(columns=["pitcher_id", "arsenal_stuff_plus"])

    rv = run_values.copy()

    # League mean and std per pitch type
    lg_stats = rv.groupby("pitch_type")["run_value_per_100"].agg(["mean", "std"])
    lg_stats["std"] = lg_stats["std"].clip(lower=0.5)  # floor to avoid division by ~0

    # Merge league stats and compute per-pitch Stuff+
    rv = rv.merge(lg_stats, on="pitch_type", how="left", suffixes=("", "_lg"))
    rv["stuff_plus_pitch"] = 100 + (rv["mean"] - rv["run_value_per_100"]) / rv["std"] * 10

    # Arsenal-level: usage-weighted average
    arsenal = (
        rv.groupby("pitcher_id")
        .apply(
            lambda g: np.average(g["stuff_plus_pitch"], weights=g["usage_pct"])
            if g["usage_pct"].sum() > 0 else 100.0,
            include_groups=False,
        )
        .rename("arsenal_stuff_plus")
        .reset_index()
    )

    return arsenal


def _build_pitcher_stuff_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
    run_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Blend projected and observed stuff/quality metrics.

    Parameters
    ----------
    proj : pd.DataFrame
        Pitcher projections.
    observed : pd.DataFrame
        Observed pitching advanced stats.
    run_values : pd.DataFrame, optional
        Pitcher run values from ``get_pitcher_run_values``.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, stuff_score.
    """
    merged = proj[["pitcher_id", "projected_k_rate", "projected_hr_per_bf",
                    "whiff_rate", "gb_pct"]].merge(
        observed[["pitcher_id", "swstr_pct", "csw_pct", "xwoba_against",
                   "barrel_pct_against", "hard_hit_pct_against"]],
        on="pitcher_id", how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=["pitcher_id", "stuff_score"])

    # Merge run values and compute Stuff+ if available
    has_stuff_plus = False
    if run_values is not None and not run_values.empty:
        rv_dedup = run_values[["pitcher_id", "weighted_rv_per_100"]].drop_duplicates()
        merged = merged.merge(rv_dedup, on="pitcher_id", how="left")

        # Compute arsenal Stuff+ (league-normalized, 100 = average)
        stuff_plus = _compute_stuff_plus(run_values)
        if not stuff_plus.empty:
            merged = merged.merge(stuff_plus, on="pitcher_id", how="left")
            has_stuff_plus = "arsenal_stuff_plus" in merged.columns and merged["arsenal_stuff_plus"].notna().any()

    # Projected: K rate (higher = better), HR/BF (lower = better),
    # GB% (higher = better — values groundball pitchers)
    proj_k = _pctl(merged["projected_k_rate"])
    proj_hr = _inv_pctl(merged["projected_hr_per_bf"])
    proj_gb = _pctl(merged["gb_pct"].fillna(merged["gb_pct"].median()))
    projected = 0.55 * proj_k + 0.25 * proj_hr + 0.20 * proj_gb

    # Observed: SwStr%, CSW%, xwOBA-against, barrel%-against,
    # plus Stuff+ when available (league-normalized pitch quality)
    obs_swstr = _pctl(merged["swstr_pct"].fillna(0))
    obs_csw = _pctl(merged["csw_pct"].fillna(0))
    obs_xwoba = _inv_pctl(merged["xwoba_against"].fillna(merged["xwoba_against"].median()))
    obs_barrel = _inv_pctl(merged["barrel_pct_against"].fillna(0))

    if has_stuff_plus:
        # Stuff+ captures underlying pitch quality better than raw SwStr%/CSW%;
        # stabilizes in fewer pitches (~80) and predicts ROS better
        stuff_plus_score = _zscore_pctl(merged["arsenal_stuff_plus"].fillna(100))
        observed_score = (
            0.30 * stuff_plus_score + 0.20 * obs_swstr + 0.20 * obs_csw
            + 0.20 * obs_xwoba + 0.10 * obs_barrel
        )
    elif "weighted_rv_per_100" in merged.columns:
        obs_run_value = _inv_pctl(merged["weighted_rv_per_100"].fillna(merged["weighted_rv_per_100"].median()))
        observed_score = (
            0.25 * obs_swstr + 0.25 * obs_csw + 0.20 * obs_run_value
            + 0.20 * obs_xwoba + 0.10 * obs_barrel
        )
    else:
        observed_score = 0.30 * obs_swstr + 0.30 * obs_csw + 0.25 * obs_xwoba + 0.15 * obs_barrel

    merged["stuff_score"] = _PROJ_WEIGHT_BASE * projected + _OBS_WEIGHT_BASE * observed_score

    # Preserve Stuff+ for output display
    cols = ["pitcher_id", "stuff_score"]
    if has_stuff_plus:
        cols.append("arsenal_stuff_plus")
    return merged[cols]


def _build_pitcher_command_score(
    proj: pd.DataFrame,
    observed: pd.DataFrame,
    efficiency: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score command/control from projected BB% and observed metrics.

    Parameters
    ----------
    proj : pd.DataFrame
        Pitcher projections.
    observed : pd.DataFrame
        Observed pitching advanced stats.
    efficiency : pd.DataFrame, optional
        Pitcher efficiency data from ``get_pitcher_efficiency``.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, command_score.
    """
    merged = proj[["pitcher_id", "projected_bb_rate"]].merge(
        observed[["pitcher_id", "bb_pct", "zone_pct", "chase_pct"]],
        on="pitcher_id", how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=["pitcher_id", "command_score"])

    # Merge efficiency data if available
    if efficiency is not None and not efficiency.empty:
        eff_cols = ["pitcher_id", "first_strike_pct", "putaway_rate"]
        eff_subset = efficiency[[c for c in eff_cols if c in efficiency.columns]].copy()
        merged = merged.merge(eff_subset, on="pitcher_id", how="left")

    # Lower BB% = better command
    proj_bb = _inv_pctl(merged["projected_bb_rate"])
    obs_bb = _inv_pctl(merged["bb_pct"].fillna(merged["bb_pct"].median()))
    obs_zone = _pctl(merged["zone_pct"].fillna(0))
    obs_chase = _pctl(merged["chase_pct"].fillna(0))

    projected_cmd = proj_bb

    if "first_strike_pct" in merged.columns and "putaway_rate" in merged.columns:
        obs_first_strike = _pctl(merged["first_strike_pct"].fillna(merged["first_strike_pct"].median()))
        obs_putaway = _pctl(merged["putaway_rate"].fillna(merged["putaway_rate"].median()))
        observed_cmd = (
            0.30 * obs_bb + 0.20 * obs_zone + 0.20 * obs_chase
            + 0.15 * obs_first_strike + 0.15 * obs_putaway
        )
    else:
        observed_cmd = 0.40 * obs_bb + 0.30 * obs_zone + 0.30 * obs_chase

    merged["command_score"] = _PROJ_WEIGHT_BASE * projected_cmd + _OBS_WEIGHT_BASE * observed_cmd
    return merged[["pitcher_id", "command_score"]]


def _build_pitcher_workload_score() -> pd.DataFrame:
    """Score projected workload from sim-based counting projections + health.

    Prefers ``pitcher_counting_sim.parquet`` (sim-based IP/games) over
    the old ``pitcher_counting.parquet`` (rate x BF). Falls back to old
    file if sim parquet doesn't exist.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, workload_score, health_score, health_label,
        plus sim projection columns for display.
    """
    # Prefer sim-based projections (IP, games, ERA, fantasy)
    sim_path = DASHBOARD_DIR / "pitcher_counting_sim.parquet"
    old_path = DASHBOARD_DIR / "pitcher_counting.parquet"

    if sim_path.exists():
        counting = pd.read_parquet(sim_path)
        # Sim has projected_ip_mean and total_games_mean
        ip_pctl = _pctl(counting["projected_ip_mean"].fillna(0))
        games_pctl = _pctl(counting["total_games_mean"].fillna(0))
        base_workload = 0.60 * ip_pctl + 0.40 * games_pctl
        logger.info("Workload score from sim parquet: %d pitchers", len(counting))
    elif old_path.exists():
        counting = pd.read_parquet(old_path)
        base_workload = _pctl(counting["projected_bf_mean"])
        logger.info("Workload score from old counting parquet (sim not found)")
    else:
        logger.warning("No counting parquet found for workload score")
        return pd.DataFrame(columns=["pitcher_id", "workload_score"])

    # Health scores
    has_health = "health_score" in counting.columns and counting["health_score"].notna().any()
    if not has_health:
        health_path = DASHBOARD_DIR / "health_scores.parquet"
        if health_path.exists():
            health_df = pd.read_parquet(health_path)
            counting = counting.merge(
                health_df[["player_id", "health_score", "health_label"]].rename(
                    columns={"player_id": "pitcher_id"}
                ),
                on="pitcher_id",
                how="left",
            )
            has_health = True

    if has_health:
        counting["health_score"] = counting["health_score"].fillna(0.85)
        counting["health_label"] = counting["health_label"].fillna("Unknown")
        health_pctl = _pctl(counting["health_score"])
        counting["workload_score"] = 0.70 * base_workload + 0.30 * health_pctl
    else:
        counting["workload_score"] = base_workload
        counting["health_score"] = np.nan
        counting["health_label"] = ""

    cols = ["pitcher_id", "workload_score", "health_score", "health_label"]
    return counting[[c for c in cols if c in counting.columns]]


def _build_pitcher_velo_trend(season: int = 2025) -> pd.DataFrame:
    """Compute YoY velocity trend from pitch-level data.

    Positive delta = velo gain (good), negative = velo loss (risk).

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, velo_delta (mph change from prior year).
    """
    from src.data.db import read_sql

    velo = read_sql(f"""
        SELECT fp.pitcher_id, dg.season,
               AVG(fp.release_speed) as avg_velo,
               COUNT(*) as pitches
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season >= {season - 1} AND dg.game_type = 'R'
              AND fp.release_speed != 'NaN'
        GROUP BY fp.pitcher_id, dg.season
        HAVING COUNT(*) >= 300
    """, {})

    if velo.empty:
        return pd.DataFrame(columns=["pitcher_id", "velo_delta"])

    # Pivot to get prior and current season
    prior = velo[velo["season"] == season - 1][["pitcher_id", "avg_velo"]].rename(
        columns={"avg_velo": "velo_prior"},
    )
    current = velo[velo["season"] == season][["pitcher_id", "avg_velo"]].rename(
        columns={"avg_velo": "velo_current"},
    )
    merged = current.merge(prior, on="pitcher_id", how="inner")
    merged["velo_delta"] = merged["velo_current"] - merged["velo_prior"]

    return merged[["pitcher_id", "velo_delta"]]


def _build_pitcher_innings_durability(min_seasons: int = 2) -> pd.DataFrame:
    """Score SP innings durability from historical IP track record.

    Rewards starters who consistently log high innings. Separate from
    health score (which measures IL stints).

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, durability_score (0–1).
    """
    from src.data.db import read_sql

    ip_hist = read_sql("""
        SELECT player_id as pitcher_id, season,
               SUM(pit_ip) as total_ip,
               COUNT(*) as games,
               SUM(CASE WHEN pit_is_starter THEN 1 ELSE 0 END) as starts
        FROM production.fact_player_game_mlb
        WHERE player_role = 'pitcher' AND season >= 2020
        GROUP BY player_id, season
        HAVING SUM(pit_bf) >= 50
    """, {})

    if ip_hist.empty:
        return pd.DataFrame(columns=["pitcher_id", "durability_score"])

    # Only consider pitchers with enough seasons
    season_counts = ip_hist.groupby("pitcher_id")["season"].nunique()
    eligible = season_counts[season_counts >= min_seasons].index
    ip_hist = ip_hist[ip_hist["pitcher_id"].isin(eligible)].copy()

    if ip_hist.empty:
        return pd.DataFrame(columns=["pitcher_id", "durability_score"])

    # Recency-weighted average IP per season
    ip_hist = ip_hist.sort_values(["pitcher_id", "season"])
    ip_hist["weight"] = ip_hist.groupby("pitcher_id").cumcount() + 1  # older=1, newer=higher

    agg = (
        ip_hist.groupby("pitcher_id")
        .apply(
            lambda g: pd.Series({
                "wtd_ip": np.average(g["total_ip"], weights=g["weight"]),
                "max_ip": g["total_ip"].max(),
                "seasons": g["season"].nunique(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # Score: 180+ IP average = elite durability, 100 IP = average, <60 = poor
    agg["durability_score"] = ((agg["wtd_ip"] - 40) / 160.0).clip(0, 1)

    return agg[["pitcher_id", "durability_score"]]


def _build_pitcher_trajectory_score(season: int = 2025) -> pd.DataFrame:
    """Score trajectory using posterior certainty, projected quality,
    age/decline risk, and velocity trend.

    Components
    ----------
    - **Projection certainty** (35%): tighter posterior SD = more proven
    - **Projected rate quality** (25%): high K%, low BB%
    - **Age factor** (20%): upside for youth, decline penalty for 32+
    - **Velocity trend** (20%): velo gain = upside, velo loss = risk

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, trajectory_score, velo_delta.
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")

    # Posterior certainty via CV (SD/mean) — same fix as hitters
    k_mean = proj["projected_k_rate"].clip(0.01)
    bb_mean = proj["projected_bb_rate"].clip(0.01)
    k_cv = proj["projected_k_rate_sd"].fillna(k_mean.median() * 0.15) / k_mean
    bb_cv = proj["projected_bb_rate_sd"].fillna(bb_mean.median() * 0.15) / bb_mean
    k_certainty = _inv_pctl(k_cv)
    bb_certainty = _inv_pctl(bb_cv)
    certainty_score = 0.50 * k_certainty + 0.50 * bb_certainty

    # Projected rate quality: elite rates signal upside regardless of track
    # record length.  Without this, short-track-record aces (Crochet, Yamamoto)
    # get penalized purely for having wide posteriors.
    k_quality = _pctl(proj["projected_k_rate"])
    bb_quality = _inv_pctl(proj["projected_bb_rate"])
    rate_quality = 0.60 * k_quality + 0.40 * bb_quality

    # Age factor: research-aligned curve (peak 27-29, decline from 30, accel 35+)
    age = proj["age"].fillna(28)
    age_factor = _pitcher_age_factor(age)

    # Velocity trend
    velo_trend = _build_pitcher_velo_trend(season=season)
    proj = proj.merge(velo_trend, on="pitcher_id", how="left")
    proj["velo_delta"] = proj["velo_delta"].fillna(0)

    # Convert velo delta to 0-1 score: +2 mph gain = 1.0, -2 mph loss = 0.0
    velo_score = ((proj["velo_delta"] + 2.0) / 4.0).clip(0, 1)

    proj["trajectory_score"] = (
        0.35 * certainty_score
        + 0.25 * rate_quality
        + 0.20 * age_factor
        + 0.20 * velo_score
    )
    return proj[["pitcher_id", "trajectory_score", "velo_delta"]]


def rank_pitchers(
    season: int = 2025,
    projection_season: int = 2026,
    min_bf: int = 50,
) -> pd.DataFrame:
    """Rank all pitchers by role (SP/RP) for 2026 value.

    Parameters
    ----------
    season : int
        Most recent completed season.
    projection_season : int
        Target projection season.
    min_bf : int
        Minimum batters faced to qualify.

    Returns
    -------
    pd.DataFrame
        Pitchers ranked by role with composite score, sub-scores.
    """
    from src.data.db import read_sql

    # Load projections
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")

    # Load observed stats
    observed = read_sql(f"""
        SELECT pitcher_id, k_pct, bb_pct, swstr_pct, csw_pct,
               zone_pct, chase_pct, contact_pct, xwoba_against,
               barrel_pct_against, hard_hit_pct_against,
               batters_faced, woba_against
        FROM production.fact_pitching_advanced
        WHERE season = {season} AND batters_faced >= {min_bf}
    """, {})

    # Role assignment — prefer sim-derived CL/SU/MR roles, fall back to SP/RP
    roles_path = DASHBOARD_DIR / "pitcher_roles.parquet"
    if roles_path.exists():
        rp_roles = pd.read_parquet(roles_path)
        # Map CL/SU/MR -> RP for composite weights, keep detail for display
        roles = rp_roles[["pitcher_id", "role"]].copy()
        roles["role_detail"] = roles["role"]
        roles["role"] = roles["role"].map(
            {"CL": "RP", "SU": "RP", "MR": "RP"}
        ).fillna(roles["role"])
        # Add SP pitchers not in reliever roles
        sp_roles = _assign_pitcher_roles()
        sp_only = sp_roles[
            (sp_roles["role"] == "SP")
            & (~sp_roles["pitcher_id"].isin(roles["pitcher_id"]))
        ].copy()
        sp_only["role_detail"] = "SP"
        roles = pd.concat([roles, sp_only], ignore_index=True)
        logger.info("Roles from sim: %s", roles["role_detail"].value_counts().to_dict())
    else:
        roles = _assign_pitcher_roles()
        roles["role_detail"] = roles["role"]

    # Load run values and efficiency for enhanced scoring
    from src.data.queries import get_pitcher_run_values, get_pitcher_efficiency
    run_values = get_pitcher_run_values(season)
    efficiency = get_pitcher_efficiency(season)

    # Build sub-scores
    stuff = _build_pitcher_stuff_score(proj, observed, run_values=run_values)
    command = _build_pitcher_command_score(proj, observed, efficiency=efficiency)
    workload = _build_pitcher_workload_score()
    durability = _build_pitcher_innings_durability()
    trajectory = _build_pitcher_trajectory_score(season=season)

    # Base from projections (include ERA/FIP if available)
    base_cols = ["pitcher_id", "pitcher_name", "age", "pitch_hand",
                 "is_starter", "projected_k_rate", "projected_bb_rate",
                 "projected_hr_per_bf", "projected_k_rate_sd",
                 "projected_bb_rate_sd", "composite_score",
                 "projected_era", "projected_era_sd",
                 "projected_era_2_5", "projected_era_97_5",
                 "projected_fip", "projected_fip_sd",
                 "observed_era", "observed_fip"]
    base_cols = [c for c in base_cols if c in proj.columns]
    base = proj[base_cols].copy()

    # Merge observed for display
    base = base.merge(
        observed[["pitcher_id", "batters_faced", "k_pct", "bb_pct",
                   "swstr_pct", "csw_pct", "xwoba_against", "woba_against"]],
        on="pitcher_id", how="inner",
    )

    # Merge run values for display
    if not run_values.empty:
        rv_display = run_values[["pitcher_id", "weighted_rv_per_100"]].drop_duplicates()
        base = base.merge(rv_display, on="pitcher_id", how="left")

    # Merge efficiency for display
    if not efficiency.empty:
        base = base.merge(
            efficiency[["pitcher_id", "first_strike_pct", "putaway_rate"]],
            on="pitcher_id", how="left",
        )

    # Merge role
    base = base.merge(roles, on="pitcher_id", how="left")
    missing_role = base["role"].isna()
    base.loc[missing_role, "role"] = np.where(
        base.loc[missing_role, "is_starter"] == 1, "SP", "RP"
    )
    if "role_detail" not in base.columns:
        base["role_detail"] = base["role"]
    base["role_detail"] = base["role_detail"].fillna(base["role"])

    # Merge sim-based counting projections for display
    sim_path = DASHBOARD_DIR / "pitcher_counting_sim.parquet"
    if sim_path.exists():
        sim_df = pd.read_parquet(sim_path)
        sim_cols = ["pitcher_id", "total_k_mean", "total_bb_mean", "total_sv_mean",
                     "total_hld_mean", "projected_ip_mean", "projected_era_mean",
                     "projected_fip_era_mean", "projected_whip_mean",
                     "dk_season_mean", "espn_season_mean", "total_games_mean"]
        sim_cols = [c for c in sim_cols if c in sim_df.columns]
        base = base.merge(sim_df[sim_cols], on="pitcher_id", how="left")
        logger.info("Merged sim projections for %d pitchers", base["total_k_mean"].notna().sum())

    # Merge sub-scores
    base = base.merge(stuff, on="pitcher_id", how="left")
    base = base.merge(command, on="pitcher_id", how="left")
    base = base.merge(workload, on="pitcher_id", how="left")
    base = base.merge(durability, on="pitcher_id", how="left")
    base = base.merge(trajectory, on="pitcher_id", how="left")

    # Pitcher breakout archetype data (GMM-derived)
    p_breakout_path = DASHBOARD_DIR / "pitcher_breakout_candidates.parquet"
    if p_breakout_path.exists():
        p_bo = pd.read_parquet(p_breakout_path)
        p_bo_cols = [
            "pitcher_id", "breakout_type", "breakout_score",
            "breakout_tier", "breakout_hole", "gmm_fit",
            "prob_stuff_dominant", "prob_command_leap", "prob_era_correction",
        ]
        available_pbc = [c for c in p_bo_cols if c in p_bo.columns]
        base = base.merge(p_bo[available_pbc], on="pitcher_id", how="left")
    else:
        for col in ["breakout_type", "breakout_tier", "breakout_hole"]:
            base[col] = ""
        for col in ["breakout_score", "gmm_fit"]:
            base[col] = np.nan

    # Fill missing with neutral
    for col in ["stuff_score", "command_score", "workload_score", "trajectory_score"]:
        base[col] = base[col].fillna(0.50)
    base["durability_score"] = base["durability_score"].fillna(0.50)
    base["velo_delta"] = base["velo_delta"].fillna(0)
    if "health_score" not in base.columns:
        base["health_score"] = np.nan
        base["health_label"] = ""

    # --- Health & role scores ---
    base["health_adj"] = base["health_score"].fillna(0.50)
    # Role score: SP uses starts (min(starts/30, 1)), RP uses appearances (min(apps/60, 1))
    if "total_games_mean" in base.columns:
        # SP: projected games ~= starts for starters, use /30 target
        # RP: projected games ~= appearances for relievers, use /60 target
        sp_mask = base["role"] == "SP"
        rp_mask = base["role"] == "RP"
        base["role_score"] = 0.5  # default
        base.loc[sp_mask, "role_score"] = (
            base.loc[sp_mask, "total_games_mean"].fillna(0) / 30
        ).clip(0, 1)
        base.loc[rp_mask, "role_score"] = (
            base.loc[rp_mask, "total_games_mean"].fillna(0) / 60
        ).clip(0, 1)
    elif "batters_faced" in base.columns:
        # Fallback: use BF as proxy (SP ~700 BF/season, RP ~250)
        base["role_score"] = _pctl(base["batters_faced"].fillna(0))
    else:
        base["role_score"] = 0.5

    # Sustain/upside blend into trajectory (mirrors hitter pattern).
    # Without this, elite established pitchers (Crochet, Yamamoto) get
    # penalized for having "no room to break out" — same problem hitters
    # had with Soto/Judge before the sustain fix.
    #   - Elite pitchers: stuff_score provides a high trajectory floor
    #   - Developing pitchers: breakout_pctl drives the score
    base["breakout_score"] = base["breakout_score"].fillna(0.0)
    p_breakout_pctl = _pctl(base["breakout_score"].clip(lower=0))
    sustain_upside = np.maximum(p_breakout_pctl, base["stuff_score"])
    base["trajectory_score"] = (
        0.65 * base["trajectory_score"] + 0.35 * sustain_upside
    )

    # Blend innings durability into SP workload (30% durability, 70% base workload)
    is_sp = base["role"] == "SP"
    base.loc[is_sp, "workload_score"] = (
        0.70 * base.loc[is_sp, "workload_score"]
        + 0.30 * _pctl(base.loc[is_sp, "durability_score"])
    )

    # --- Glicko-2 opponent-adjusted performance ---
    glicko = _load_glicko_scores("pitcher")
    if not glicko.empty:
        base = base.merge(glicko, on="pitcher_id", how="left")
    if "glicko_score" not in base.columns:
        base["glicko_score"] = 0.5
    base["glicko_score"] = base["glicko_score"].fillna(0.5)

    # --- Composite (role-specific weights) ---
    is_sp = base["role"] == "SP"
    is_rp = base["role"] == "RP"

    # SP composite
    base.loc[is_sp, "tdd_value_score"] = (
        _SP_WEIGHTS["stuff"] * base.loc[is_sp, "stuff_score"]
        + _SP_WEIGHTS["command"] * base.loc[is_sp, "command_score"]
        + _SP_WEIGHTS["workload"] * base.loc[is_sp, "workload_score"]
        + _SP_WEIGHTS["health"] * base.loc[is_sp, "health_adj"]
        + _SP_WEIGHTS["role"] * base.loc[is_sp, "role_score"]
        + _SP_WEIGHTS["trajectory"] * base.loc[is_sp, "trajectory_score"]
        + _SP_WEIGHTS["glicko"] * base.loc[is_sp, "glicko_score"]
    )
    # RP composite
    base.loc[is_rp, "tdd_value_score"] = (
        _RP_WEIGHTS["stuff"] * base.loc[is_rp, "stuff_score"]
        + _RP_WEIGHTS["command"] * base.loc[is_rp, "command_score"]
        + _RP_WEIGHTS["workload"] * base.loc[is_rp, "workload_score"]
        + _RP_WEIGHTS["health"] * base.loc[is_rp, "health_adj"]
        + _RP_WEIGHTS["role"] * base.loc[is_rp, "role_score"]
        + _RP_WEIGHTS["trajectory"] * base.loc[is_rp, "trajectory_score"]
        + _RP_WEIGHTS["glicko"] * base.loc[is_rp, "glicko_score"]
    )
    # Fallback for any unassigned role
    neither = ~is_sp & ~is_rp
    if neither.any():
        base.loc[neither, "tdd_value_score"] = (
            _SP_WEIGHTS["stuff"] * base.loc[neither, "stuff_score"]
            + _SP_WEIGHTS["command"] * base.loc[neither, "command_score"]
            + _SP_WEIGHTS["workload"] * base.loc[neither, "workload_score"]
            + _SP_WEIGHTS["health"] * base.loc[neither, "health_adj"]
            + _SP_WEIGHTS["role"] * base.loc[neither, "role_score"]
            + _SP_WEIGHTS["trajectory"] * base.loc[neither, "trajectory_score"]
            + _SP_WEIGHTS["glicko"] * base.loc[neither, "glicko_score"]
        )

    # --- Two-score architecture: current_value + talent_upside ---
    # Save production composite before scouting blending.
    production_composite = base["tdd_value_score"].copy()
    base["current_value_score"] = production_composite
    base["talent_upside_score"] = production_composite

    try:
        from src.models.scouting_grades import grade_pitcher_tools
        scouting = grade_pitcher_tools(base, season=season)
        if not scouting.empty:
            base = base.merge(scouting, on="pitcher_id", how="left")
            logger.info("Scouting grades computed for %d pitchers", scouting["tools_rating"].notna().sum())

            # Regress diamond rating by BF reliability (same for both scores)
            dr_norm = (base["tools_rating"] / 10.0).clip(0, 1)

            cfg = _RANK_CFG["pitcher"]
            bf_col = base["batters_faced"] if "batters_faced" in base.columns else 400
            reliability = ((bf_col - cfg["bf_ramp_min"]) / (cfg["bf_ramp_max"] - cfg["bf_ramp_min"])).clip(0, 1)

            # Low-IP pitchers: regress diamond rating toward 0.50
            dr_regressed = reliability * dr_norm + (1 - reliability) * 0.50

            # Health penalty: injury-prone pitchers get DR dampened.
            health = base["health_adj"] if "health_adj" in base.columns else 0.50
            health_penalty = np.where(health < 0.40, 0.70 + 0.75 * health, 1.0)
            dr_regressed = dr_regressed * health_penalty

            # ----- talent_upside_score: scouting-dominant -----
            upside_w = cfg["upside_scout_weight"]
            base["talent_upside_score"] = (
                upside_w * dr_regressed + (1 - upside_w) * production_composite
            )

            # ----- current_value_score: production-dominant -----
            scout_w = _exposure_conditioned_scouting_weight(
                bf_col,
                min_exp=cfg["bf_ramp_min"],
                max_exp=cfg["bf_ramp_max"],
                weight_ceil=cfg["scout_weight_ceil"],
                weight_floor=cfg["scout_weight_floor"],
            )
            base["current_value_score"] = (
                scout_w * dr_regressed + (1 - scout_w) * production_composite
            )

            n_scouted = scouting["tools_rating"].notna().sum()
            logger.info(
                "Two-score split: %d pitchers — scout weight range [%.0f%%, %.0f%%]",
                n_scouted, cfg["scout_weight_floor"] * 100, cfg["scout_weight_ceil"] * 100,
            )
    except Exception:
        logger.warning("Could not compute pitcher scouting grades", exc_info=True)

    # tdd_value_score = current_value_score (backward compat for team consumption)
    base["tdd_value_score"] = base["current_value_score"]

    # Rank within role
    base = base.sort_values("current_value_score", ascending=False)
    base["role_rank"] = base.groupby("role").cumcount() + 1

    # Overall pitcher rank
    base["overall_rank"] = base["current_value_score"].rank(ascending=False, method="min").astype(int)

    # Talent-based ranking (scouting-dominant)
    base["talent_rank"] = base["talent_upside_score"].rank(ascending=False, method="min").astype(int)

    # Select output
    output_cols = [
        "role_rank", "overall_rank", "talent_rank",
        "pitcher_id", "pitcher_name", "role",
        "role_detail", "age", "pitch_hand",
        # Two-score architecture
        "current_value_score", "talent_upside_score",
        "tdd_value_score",
        # Sub-scores
        "stuff_score", "arsenal_stuff_plus", "command_score", "workload_score",
        "health_adj", "role_score",
        "trajectory_score",
        "durability_score", "velo_delta",
        # Health
        "health_score", "health_label",
        # Observed
        "batters_faced", "k_pct", "bb_pct", "swstr_pct", "csw_pct",
        "xwoba_against", "woba_against",
        "weighted_rv_per_100", "first_strike_pct", "putaway_rate",
        "observed_era", "observed_fip",
        # Projected
        "projected_k_rate", "projected_bb_rate", "projected_hr_per_bf",
        "projected_k_rate_sd", "projected_bb_rate_sd",
        "projected_era", "projected_era_sd",
        "projected_era_2_5", "projected_era_97_5",
        "projected_fip", "projected_fip_sd",
        # Breakout archetype (GMM-derived)
        "breakout_type", "breakout_score", "breakout_tier",
        "breakout_hole", "gmm_fit",
        "prob_stuff_dominant", "prob_command_leap", "prob_era_correction",
        # Sim-based season projections
        "total_k_mean", "total_bb_mean", "total_sv_mean", "total_hld_mean",
        "projected_ip_mean", "projected_era_mean", "projected_fip_era_mean",
        "projected_whip_mean", "dk_season_mean", "espn_season_mean",
        "total_games_mean",
        # Glicko-2 opponent-adjusted rating
        "glicko_score", "glicko_mu", "glicko_phi",
        # Scouting grades (20-80) + diamond rating (0-10)
        "grade_stuff", "grade_command", "grade_durability", "tools_rating",
    ]
    available = [c for c in output_cols if c in base.columns]
    result = base[available].sort_values(["role", "role_rank"])

    for role in PITCHER_ROLES:
        role_df = result[result["role"] == role]
        if not role_df.empty:
            logger.info(
                "%s: %d ranked — #1 %s (%.3f)",
                role, len(role_df),
                role_df.iloc[0]["pitcher_name"],
                role_df.iloc[0]["tdd_value_score"],
            )

    return result


# ===================================================================
# Combined entry point
# ===================================================================

def rank_all(
    season: int = 2025,
    projection_season: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Run all positional rankings.

    Parameters
    ----------
    season : int
        Most recent completed season for observed data.
    projection_season : int
        Target projection season.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'hitters', 'pitchers'. Each sorted by position/role then rank.
    """
    logger.info("Building 2026 positional rankings...")
    hitters = rank_hitters(season=season, projection_season=projection_season)
    pitchers = rank_pitchers(season=season, projection_season=projection_season)
    logger.info(
        "Rankings complete: %d hitters, %d pitchers",
        len(hitters), len(pitchers),
    )
    return {"hitters": hitters, "pitchers": pitchers}
