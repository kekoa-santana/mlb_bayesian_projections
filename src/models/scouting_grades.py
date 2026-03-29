"""
TDD 20-80 Scouting Grade System.

Grades individual tools on the traditional 20-80 scale (5-point increments)
for MLB players and MiLB prospects. Produces a composite Diamond Rating (0-10).

Population reference: always MLB qualifiers (200+ PA hitters, 100+ BF pitchers).
Prospects are graded on the same MLB scale using translated rates, so a prospect
with a 60-grade Power tool has ISO equivalent to a top-16% MLB player.

The scale follows the standard normal distribution:
  grade = clip(round_to_5(50 + 10 * z_score), 20, 80)
  80 = elite (+3 SD), 70 = plus-plus (+2 SD), 60 = plus (+1 SD),
  50 = average, 40 = below average (-1 SD), 30 = well below (-2 SD)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRADE_MIN = 20
GRADE_MAX = 80

# Spread factor: multiply population std by this to widen grade distribution.
# Without this, MLB qualifier pools self-select competent players and compress
# grades into the 40-65 range. 0.70 gives proper 20-80 spread where elite
# players reach 70-80 and poor performers hit 20-30.
_SPREAD_FACTOR = 0.70

# ===================================================================
# Diamond Rating weights — tool → overall (0-10)
# ===================================================================

# Position-specific tool weights for diamond rating.
# The bat outweighs defense — a 212 wRC+ hitter shouldn't be rank #24
# because he runs slow. But defense matters MORE at premium positions
# (C, SS, CF) where it's genuinely scarce.
_POS_DIAMOND_WEIGHTS = {
    "C":  {"hit": 0.20, "power": 0.20, "speed": 0.05, "fielding": 0.30, "discipline": 0.25},
    "SS": {"hit": 0.20, "power": 0.15, "speed": 0.15, "fielding": 0.25, "discipline": 0.25},
    "CF": {"hit": 0.20, "power": 0.15, "speed": 0.20, "fielding": 0.20, "discipline": 0.25},
    "2B": {"hit": 0.25, "power": 0.15, "speed": 0.15, "fielding": 0.20, "discipline": 0.25},
    "3B": {"hit": 0.20, "power": 0.25, "speed": 0.10, "fielding": 0.20, "discipline": 0.25},
    "RF": {"hit": 0.25, "power": 0.25, "speed": 0.10, "fielding": 0.15, "discipline": 0.25},
    "LF": {"hit": 0.25, "power": 0.25, "speed": 0.10, "fielding": 0.15, "discipline": 0.25},
    "1B": {"hit": 0.25, "power": 0.30, "speed": 0.05, "fielding": 0.10, "discipline": 0.30},
    "DH": {"hit": 0.30, "power": 0.30, "speed": 0.05, "fielding": 0.00, "discipline": 0.35},
}

# Fallback for unknown positions
_HITTER_DIAMOND_WEIGHTS = {"hit": 0.25, "power": 0.25, "speed": 0.10, "fielding": 0.15, "discipline": 0.25}

# SP: command + durability matter more (grind 200 IP, keep pitch count down)
# RP: stuff IS the value (max effort, 1 inning, blow guys away)
_SP_DIAMOND_WEIGHTS = {"stuff": 0.35, "command": 0.35, "durability": 0.30}
_RP_DIAMOND_WEIGHTS = {"stuff": 0.60, "command": 0.30, "durability": 0.10}


# ===================================================================
# Core grading utilities
# ===================================================================

def _round_to_5(x: float) -> int:
    """Round to nearest 5 for standard scouting increments."""
    return int(5 * round(x / 5))


def _z_grade_series(
    series: pd.Series,
    pop_mean: float | None = None,
    pop_std: float | None = None,
    invert: bool = False,
) -> pd.Series:
    """Convert a metric Series to 20-80 grades via z-score.

    Parameters
    ----------
    series : pd.Series
        Raw metric values.
    pop_mean, pop_std : float, optional
        Population parameters. If None, computed from the series itself.
    invert : bool
        If True, lower values are better (e.g., K% for hitters).

    Returns
    -------
    pd.Series
        Integer grades in 5-point increments, 20-80.
    """
    if pop_mean is None:
        pop_mean = series.mean()
    if pop_std is None:
        pop_std = series.std()
    if pop_std < 1e-9:
        return pd.Series(50, index=series.index)

    z = (series - pop_mean) / (pop_std * _SPREAD_FACTOR)
    if invert:
        z = -z
    raw = 50 + 10 * z

    # Rescale so the actual min → 20 and max → 80. This guarantees the
    # full 20-80 range is used for every tool — the best player IS 80
    # and the worst IS 20, with everyone else scaled relative to them.
    raw_min, raw_max = raw.min(), raw.max()
    if raw_max - raw_min > 1e-9:
        rescaled = GRADE_MIN + (raw - raw_min) / (raw_max - raw_min) * (GRADE_MAX - GRADE_MIN)
    else:
        rescaled = raw
    grades = rescaled.apply(lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX)))
    return grades.astype(int)


def _composite_z(
    df: pd.DataFrame,
    specs: list[tuple[str, float, bool, float, float]],
    pa: pd.Series | None = None,
    min_pa: int = 150,
    full_pa: int = 500,
) -> pd.Series:
    """Weighted composite z-score from multiple metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the metric columns referenced in specs.
    specs : list of (col, weight, invert, pop_mean, pop_std)
        Each tuple: column name, weight (sums to 1), whether to invert,
        population mean, population std.
    pa : pd.Series, optional
        Plate appearances for sample-size dampening.
    min_pa, full_pa : int
        Dampening range.

    Returns
    -------
    pd.Series
        Composite z-score (not yet graded).
    """
    total_w = 0.0
    composite = pd.Series(0.0, index=df.index)

    for col, weight, invert, pmean, pstd in specs:
        if col not in df.columns:
            continue
        vals = df[col].astype(float)
        if pstd < 1e-9:
            continue
        z = (vals.fillna(pmean) - pmean) / (pstd * _SPREAD_FACTOR)
        if invert:
            z = -z
        composite += weight * z
        total_w += weight

    if total_w > 0:
        composite /= total_w

    # Sample-size dampening: pull toward 0 (grade 50) for small samples.
    # Uses sqrt curve so dampening is aggressive at low PA:
    # 200 PA → 50% confidence, 350 PA → 71%, 500+ PA → 100%
    if pa is not None:
        frac = ((pa - min_pa) / (full_pa - min_pa)).clip(0, 1)
        confidence = np.sqrt(frac)  # sqrt makes it steeper at low end
        composite = composite * confidence

    return composite


def _compute_hitter_pop_stats(season: int) -> dict[str, tuple[float, float]]:
    """Load MLB hitter population mean/std for each tool metric.

    Uses 2-3 year pool (recency-weighted) for stability.
    """
    from src.data.db import read_sql

    # Pool 3 seasons for stable population stats
    seasons = [season - 2, season - 1, season]
    weights = {season - 2: 1, season - 1: 2, season: 3}

    # Batting advanced stats
    batting = read_sql(f"""
        SELECT season, k_pct, bb_pct, xba, barrel_pct, hard_hit_pct, pa
        FROM production.fact_batting_advanced
        WHERE season IN ({','.join(str(s) for s in seasons)})
          AND pa >= 200
    """, {})

    # Traditional stats for ISO
    from src.data.queries import get_hitter_traditional_stats
    trad_frames = []
    for s in seasons:
        t = get_hitter_traditional_stats(s)
        if not t.empty:
            t = t[t["pa"] >= 200]
            trad_frames.append(t[["iso", "pa"]])
    trad = pd.concat(trad_frames, ignore_index=True) if trad_frames else pd.DataFrame()

    # Observed profile (whiff, contact, exit velo)
    from src.data.feature_eng import get_cached_hitter_observed_profile
    obs_frames = []
    for s in seasons:
        try:
            o = get_cached_hitter_observed_profile(s)
            if not o.empty:
                obs_frames.append(o)
        except Exception:
            pass
    obs = pd.concat(obs_frames, ignore_index=True) if obs_frames else pd.DataFrame()

    # Sprint speed
    from src.data.feature_eng import get_cached_sprint_speed
    spd_frames = []
    for s in seasons:
        try:
            sp = get_cached_sprint_speed(s)
            if not sp.empty:
                spd_frames.append(sp)
        except Exception:
            pass
    spd = pd.concat(spd_frames, ignore_index=True) if spd_frames else pd.DataFrame()

    stats: dict[str, tuple[float, float]] = {}

    # Compute mean/std for each metric
    for col in ["k_pct", "bb_pct", "xba", "barrel_pct", "hard_hit_pct"]:
        if col in batting.columns:
            vals = batting[col].dropna()
            if len(vals) > 30:
                stats[col] = (vals.mean(), vals.std())

    if not trad.empty and "iso" in trad.columns:
        vals = trad["iso"].dropna()
        if len(vals) > 30:
            stats["iso"] = (vals.mean(), vals.std())

    for col in ["whiff_rate", "z_contact_pct", "avg_exit_velo", "hard_hit_pct"]:
        if not obs.empty and col in obs.columns:
            vals = obs[col].dropna()
            if len(vals) > 30:
                stats[col] = (vals.mean(), vals.std())

    if not spd.empty and "sprint_speed" in spd.columns:
        vals = spd["sprint_speed"].dropna()
        if len(vals) > 30:
            stats["sprint_speed"] = (vals.mean(), vals.std())

    # O-contact% (outside-zone contact rate) — ability to fight off tough pitches
    try:
        o_contact = read_sql(f"""
            SELECT
                fp.batter_id,
                COUNT(*) FILTER (
                    WHERE (plate_x < -0.83 OR plate_x > 0.83
                        OR plate_z < 1.5 OR plate_z > 3.5)
                    AND is_swing AND NOT is_whiff
                )::float
                / NULLIF(COUNT(*) FILTER (
                    WHERE (plate_x < -0.83 OR plate_x > 0.83
                        OR plate_z < 1.5 OR plate_z > 3.5)
                    AND is_swing
                ), 0) AS o_contact_pct
            FROM production.fact_pitch fp
            JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
            WHERE dg.season IN ({','.join(str(s) for s in seasons)})
              AND dg.game_type = 'R'
            GROUP BY fp.batter_id
            HAVING COUNT(*) > 500
        """, {})
        if not o_contact.empty:
            vals = o_contact["o_contact_pct"].dropna()
            if len(vals) > 30:
                stats["o_contact_pct"] = (vals.mean(), vals.std())
    except Exception:
        logger.warning("Could not compute O-contact% population stats", exc_info=True)

    logger.info("Hitter population stats computed for %d metrics", len(stats))
    return stats


def _compute_pitcher_pop_stats(
    season: int,
    role: str = "all",
) -> dict[str, tuple[float, float]]:
    """Load MLB pitcher population mean/std, optionally split by role.

    Parameters
    ----------
    season : int
        Most recent completed season.
    role : str
        "SP", "RP", or "all". When split by role, population stats
        are computed against only that role — a 60-grade SP Stuff means
        top-16% among starters, not all pitchers.
    """
    from src.data.db import read_sql

    seasons = [season - 2, season - 1, season]

    # Use BF threshold to distinguish SP vs RP in the population:
    # SP typically face 400+ BF/season, RP < 300.
    if role == "SP":
        bf_filter = "AND batters_faced >= 300"
    elif role == "RP":
        bf_filter = "AND batters_faced < 300 AND batters_faced >= 50"
    else:
        bf_filter = "AND batters_faced >= 100"

    pitching = read_sql(f"""
        SELECT season, k_pct, bb_pct, swstr_pct, csw_pct,
               zone_pct, chase_pct, xwoba_against,
               barrel_pct_against, batters_faced
        FROM production.fact_pitching_advanced
        WHERE season IN ({','.join(str(s) for s in seasons)})
          {bf_filter}
    """, {})

    stats: dict[str, tuple[float, float]] = {}
    for col in ["k_pct", "bb_pct", "swstr_pct", "csw_pct",
                "zone_pct", "chase_pct", "xwoba_against", "barrel_pct_against"]:
        if col in pitching.columns:
            vals = pitching[col].dropna()
            if len(vals) > 30:
                stats[col] = (vals.mean(), vals.std())

    logger.info("Pitcher %s population stats: %d metrics (%d player-seasons)",
                role, len(stats), len(pitching))
    return stats


# ===================================================================
# Hitter tool grades
# ===================================================================

def _grade_hit(df: pd.DataFrame, pop: dict) -> pd.Series:
    """Hit/Contact tool: ability to produce hits and avoid outs.

    xBA (40%) is the anchor — it captures exit velocity, launch angle, and
    sprint speed into an expected batting average. A .300 xBA hitter should
    grade 70+ regardless of K rate. z_contact_pct (25%) measures genuine
    bat-to-ball skill in the zone. K% (20%) and O-contact% (15%) contribute
    but don't dominate — a high-K hitter with elite xBA (Judge) is still
    an elite hit-tool player.
    """
    specs = []
    if "xba" in pop:
        specs.append(("xba", 0.40, False, *pop["xba"]))
    if "z_contact_pct" in pop:
        specs.append(("z_contact_pct", 0.25, False, *pop["z_contact_pct"]))
    if "k_pct" in pop:
        specs.append(("k_pct", 0.20, True, *pop["k_pct"]))
    if "o_contact_pct" in pop:
        specs.append(("o_contact_pct", 0.15, False, *pop["o_contact_pct"]))

    if not specs:
        return pd.Series(50, index=df.index, dtype=int)

    z = _composite_z(df, specs, pa=df.get("pa"))
    raw = 50 + 10 * z
    return raw.apply(lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))).astype(int)


def _grade_power(df: pd.DataFrame, pop: dict) -> pd.Series:
    """Power tool: game power (ISO) + raw power indicators.

    ISO (park-adj) 35%, barrel% 25%, hard_hit% 20%, avg_exit_velo 20%.
    """
    # Use park-adjusted ISO if available, else raw
    iso_col = "iso_park_adj" if "iso_park_adj" in df.columns else "iso"

    specs = []
    if "iso" in pop and iso_col in df.columns:
        specs.append((iso_col, 0.35, False, *pop["iso"]))
    if "barrel_pct" in pop:
        specs.append(("barrel_pct", 0.25, False, *pop["barrel_pct"]))
    if "hard_hit_pct" in pop:
        specs.append(("hard_hit_pct", 0.20, False, *pop["hard_hit_pct"]))
    if "avg_exit_velo" in pop:
        specs.append(("avg_exit_velo", 0.20, False, *pop["avg_exit_velo"]))

    if not specs:
        return pd.Series(50, index=df.index, dtype=int)

    z = _composite_z(df, specs, pa=df.get("pa"))
    raw = 50 + 10 * z
    return raw.apply(lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))).astype(int)


def _grade_speed(df: pd.DataFrame, pop: dict) -> pd.Series:
    """Speed/Run tool: sprint speed 60%, SB rate 40%."""
    specs = []
    if "sprint_speed" in pop:
        specs.append(("sprint_speed", 0.60, False, *pop["sprint_speed"]))
    if "sb_rate" in df.columns:
        # SB rate population: compute from data if not in pop
        vals = df["sb_rate"].dropna()
        if len(vals) > 30:
            specs.append(("sb_rate", 0.40, False, vals.mean(), vals.std()))

    if not specs:
        return pd.Series(50, index=df.index, dtype=int)

    z = _composite_z(df, specs)  # no PA dampening for speed (physical tool)
    raw = 50 + 10 * z
    return raw.apply(lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))).astype(int)


def _grade_fielding(df: pd.DataFrame) -> pd.Series:
    """Fielding tool from existing regressed OAA percentile.

    The fielding_score (0-1) is already a percentile. Convert to z-score,
    then to 20-80 grade. For catchers, fielding_combined blends OAA + framing.
    """
    from scipy.stats import norm

    score_col = "fielding_combined" if "fielding_combined" in df.columns else "fielding_score"
    if score_col not in df.columns:
        return pd.Series(45, index=df.index, dtype=int)

    pctl = df[score_col].fillna(0.50).clip(0.01, 0.99)
    z = pd.Series(norm.ppf(pctl.values), index=df.index)
    raw = 50 + 10 * z
    return raw.apply(lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))).astype(int)


def _grade_discipline(df: pd.DataFrame, pop: dict) -> pd.Series:
    """Discipline/Eye tool (6th tool — analytics innovation).

    projected_bb_rate 35%, chase_rate (inv) 35%, two_strike_whiff_rate (inv) 30%.
    Uses Bayesian projected BB% for same reason as pitcher Command.
    """
    specs = []
    bb_col = "projected_bb_rate" if "projected_bb_rate" in df.columns else "bb_pct"
    if "bb_pct" in pop:
        specs.append((bb_col, 0.35, False, *pop["bb_pct"]))
    if "chase_rate" in df.columns:
        vals = df["chase_rate"].dropna()
        if len(vals) > 30:
            specs.append(("chase_rate", 0.35, True, vals.mean(), vals.std()))
    if "two_strike_whiff_rate" in df.columns:
        vals = df["two_strike_whiff_rate"].dropna()
        if len(vals) > 30:
            specs.append(("two_strike_whiff_rate", 0.30, True, vals.mean(), vals.std()))

    if not specs:
        return pd.Series(50, index=df.index, dtype=int)

    z = _composite_z(df, specs, pa=df.get("pa"))
    raw = 50 + 10 * z
    return raw.apply(lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))).astype(int)


def _tools_rating(grades: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute 0-10 Diamond Rating from tool grades.

    Formula: (weighted_avg - 20) / 60 * 10, rounded to 1 decimal.
    """
    weighted = pd.Series(0.0, index=grades.index)
    total_w = 0.0
    for tool, w in weights.items():
        col = f"grade_{tool}"
        if col in grades.columns:
            weighted += w * grades[col].fillna(50)
            total_w += w
    if total_w > 0:
        weighted /= total_w
    return ((weighted - 20) / 60 * 10).round(1).clip(0, 10)


# ===================================================================
# Main entry: MLB hitter grading
# ===================================================================

def grade_hitter_tools(
    base: pd.DataFrame,
    season: int = 2025,
) -> pd.DataFrame:
    """Compute 20-80 tool grades for MLB hitters.

    Parameters
    ----------
    base : pd.DataFrame
        Hitter rankings base DataFrame. Must have batter_id, pa, k_pct,
        bb_pct, barrel_pct, hard_hit_pct, xba, fielding_score/fielding_combined.
        chase_rate, two_strike_whiff_rate optional.
    season : int
        Most recent completed season.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, grade_hit, grade_power, grade_speed,
        grade_fielding, grade_discipline, tools_rating.
    """
    pop = _compute_hitter_pop_stats(season)

    # Enrich with additional metrics not in base
    from src.data.feature_eng import (
        get_cached_hitter_observed_profile,
        get_cached_sprint_speed,
    )
    from src.data.queries import get_hitter_traditional_stats

    enriched = base.copy()

    # Merge observed profile (z_contact_pct, avg_exit_velo)
    obs = get_cached_hitter_observed_profile(season)
    if not obs.empty:
        obs_cols = ["batter_id"]
        for c in ["z_contact_pct", "avg_exit_velo"]:
            if c in obs.columns:
                obs_cols.append(c)
        enriched = enriched.merge(obs[obs_cols], on="batter_id", how="left")

    # Merge sprint speed
    spd = get_cached_sprint_speed(season)
    if not spd.empty and "sprint_speed" in spd.columns:
        enriched = enriched.merge(
            spd[["player_id", "sprint_speed"]].rename(columns={"player_id": "batter_id"}),
            on="batter_id", how="left",
        )

    # Load traditional ISO and park-adjust
    trad = get_hitter_traditional_stats(season)
    if not trad.empty and "iso" in trad.columns:
        enriched = enriched.merge(
            trad[["batter_id", "iso"]],
            on="batter_id", how="left",
        )

        # Park-adjust ISO using HR park factor
        try:
            from src.data.park_factors import compute_multi_stat_park_factors, get_player_park_adjustments
            from src.data.queries import get_hitter_team_venue
            pf = compute_multi_stat_park_factors(seasons=[season - 2, season - 1, season])
            venues = get_hitter_team_venue(season)
            if not pf.empty and not venues.empty:
                pf_map = get_player_park_adjustments(
                    pf, venues.rename(columns={"batter_id": "player_id"}),
                )
                pf_df = pd.DataFrame([
                    {"batter_id": pid, "pf_hr": v.get("pf_hr", 1.0)}
                    for pid, v in pf_map.items()
                ])
                enriched = enriched.merge(pf_df, on="batter_id", how="left")
                enriched["pf_hr"] = enriched["pf_hr"].fillna(1.0)
                enriched["iso_park_adj"] = enriched["iso"] / enriched["pf_hr"]
        except Exception:
            logger.warning("Could not park-adjust ISO", exc_info=True)

    # Compute SB rate from sim data if available
    if "total_sb_mean" in enriched.columns and "total_pa_mean" in enriched.columns:
        enriched["sb_rate"] = (
            enriched["total_sb_mean"] / enriched["total_pa_mean"].clip(lower=1)
        )
    elif "sb" in enriched.columns and "pa" in enriched.columns:
        enriched["sb_rate"] = enriched["sb"] / enriched["pa"].clip(lower=1)

    # Compute O-contact% per player (outside-zone contact / outside-zone swings)
    try:
        from src.data.db import read_sql as _read_sql
        o_contact = _read_sql(f"""
            SELECT
                fp.batter_id,
                COUNT(*) FILTER (
                    WHERE (plate_x < -0.83 OR plate_x > 0.83
                        OR plate_z < 1.5 OR plate_z > 3.5)
                    AND is_swing AND NOT is_whiff
                )::float
                / NULLIF(COUNT(*) FILTER (
                    WHERE (plate_x < -0.83 OR plate_x > 0.83
                        OR plate_z < 1.5 OR plate_z > 3.5)
                    AND is_swing
                ), 0) AS o_contact_pct
            FROM production.fact_pitch fp
            JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
            WHERE dg.season = {season} AND dg.game_type = 'R'
            GROUP BY fp.batter_id
            HAVING COUNT(*) FILTER (
                WHERE (plate_x < -0.83 OR plate_x > 0.83
                    OR plate_z < 1.5 OR plate_z > 3.5)
                AND is_swing
            ) >= 50
        """, {})
        if not o_contact.empty:
            enriched = enriched.merge(o_contact, on="batter_id", how="left")
            logger.info("O-contact%% computed for %d hitters", o_contact["o_contact_pct"].notna().sum())
    except Exception:
        logger.warning("Could not compute O-contact%%", exc_info=True)

    # Grade each tool
    result = pd.DataFrame({"batter_id": enriched["batter_id"]})
    result["grade_hit"] = _grade_hit(enriched, pop)
    result["grade_power"] = _grade_power(enriched, pop)
    result["grade_speed"] = _grade_speed(enriched, pop)
    result["grade_fielding"] = _grade_fielding(enriched)
    result["grade_discipline"] = _grade_discipline(enriched, pop)

    # Diamond rating: UNIFORM tool weights for cross-position comparison.
    # All players graded on the same scale so a 7.0 at CF means the same
    # raw talent as a 7.0 at 1B.  Position-specific weights were causing
    # 1B to be inflated (power=30% + discipline=30% hid their weak speed/
    # fielding) while CF players were penalized for needing 5 tools.
    #
    # DH discount applied separately (no fielding contribution).
    positions = enriched.get("position", pd.Series("", index=enriched.index))
    is_two_way = enriched.get("is_two_way", pd.Series(False, index=enriched.index)).fillna(False)

    # Uniform weights for all fielding positions
    _UNIFORM_WEIGHTS = {"hit": 0.25, "power": 0.20, "speed": 0.10, "fielding": 0.20, "discipline": 0.25}
    # DH: no fielding, redistribute to bat tools
    _DH_WEIGHTS = {"hit": 0.30, "power": 0.25, "speed": 0.05, "fielding": 0.00, "discipline": 0.40}

    dr = pd.Series(5.0, index=result.index)
    is_dh = positions == "DH"
    if (~is_dh).any():
        dr.loc[~is_dh] = _tools_rating(result.loc[~is_dh], _UNIFORM_WEIGHTS)
    if is_dh.any():
        dr.loc[is_dh] = _tools_rating(result.loc[is_dh], _DH_WEIGHTS)

    # Blend tools (85%) with projected wRC+ (15%) as a production guardrail.
    # Prevents speed-only players with terrible bats from ranking top-10.
    # Projected wRC+ comes from the Bayesian sim and already regresses
    # small samples toward population priors.
    wrc_grade = pd.Series(5.0, index=result.index)  # default: average
    wrc_col = "projected_wrc_plus_mean"
    if wrc_col in enriched.columns and enriched[wrc_col].notna().any():
        wrc_grade = ((enriched[wrc_col].fillna(100) - 60) / 80 * 10).clip(0, 10)
        raw_dr = 0.85 * dr + 0.15 * wrc_grade
    else:
        raw_dr = dr

    # Two-way bonus: pitcher value adds up to +1.0 diamond point
    if is_two_way.any() and "two_way_bonus" in enriched.columns:
        two_way_boost = enriched["two_way_bonus"].fillna(0).clip(0, 0.05) * 20
        raw_dr = raw_dr + np.where(is_two_way, two_way_boost, 0)

    # DH discount: DHs don't provide fielding value, so their overall
    # contribution is lower than a fielding player with equal bat tools.
    # Exempt two-way players (e.g. Ohtani) who DH only because they pitch.
    dh_discount = is_dh & ~is_two_way
    if dh_discount.any():
        raw_dr = raw_dr.copy()
        raw_dr.loc[dh_discount] *= 0.92  # 8% discount for pure DHs
        logger.info("DH discount applied to %d pure DHs (exempting %d two-way)",
                    dh_discount.sum(), (is_dh & is_two_way).sum())

    # Rescale so worst hitter = 1.0, best = 10.0 (league-relative)
    dr_min, dr_max = raw_dr.min(), raw_dr.max()
    if dr_max - dr_min > 1e-9:
        result["tools_rating"] = (1.0 + (raw_dr - dr_min) / (dr_max - dr_min) * 9.0).round(1)
    else:
        result["tools_rating"] = 5.5

    logger.info(
        "Hitter scouting grades: %d players, avg diamond=%.1f",
        len(result), result["tools_rating"].mean(),
    )
    return result


# ===================================================================
# Pitcher tool grades
# ===================================================================

def _grade_stuff(df: pd.DataFrame, pop: dict) -> pd.Series:
    """Stuff tool: pitch quality + swing-and-miss + results.

    arsenal_stuff_plus 25%, SwStr% 20%, CSW% 20%, xwOBA-against (inv) 15%,
    projected_fip (inv) 20%.

    Projected FIP is included directly in the Stuff tool (not just as a
    guardrail) because it captures the aggregate K/BB/HR outcome that
    pitch-level metrics can miss. A 7-pitch arsenal with average individual
    stuff but elite results (Skenes: .253 xwOBA, 3.07 FIP) should grade
    as elite stuff — the deception and sequencing ARE the stuff.
    """
    specs = []

    if "arsenal_stuff_plus" in df.columns:
        specs.append(("arsenal_stuff_plus", 0.25, False, 100.0, 10.0))
    if "swstr_pct" in pop:
        specs.append(("swstr_pct", 0.20, False, *pop["swstr_pct"]))
    if "csw_pct" in pop:
        specs.append(("csw_pct", 0.20, False, *pop["csw_pct"]))
    if "projected_fip" in df.columns:
        # FIP directly in stuff: lower FIP = better stuff (inverted)
        fip_vals = df["projected_fip"].dropna()
        if len(fip_vals) > 30:
            specs.append(("projected_fip", 0.20, True, fip_vals.mean(), fip_vals.std()))
    if "xwoba_against" in pop:
        specs.append(("xwoba_against", 0.15, True, *pop["xwoba_against"]))

    if not specs:
        return pd.Series(50, index=df.index, dtype=int)

    bf = df.get("batters_faced")
    z = _composite_z(df, specs, pa=bf, min_pa=50, full_pa=300)
    raw = 50 + 10 * z
    return raw.apply(lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))).astype(int)


def _grade_command(df: pd.DataFrame, pop: dict) -> pd.Series:
    """Command tool: location + control.

    BB% blend (inv) 35%, zone% 25%, chase% 20%, first_strike% 20%.
    Uses a blend of projected BB% (Bayesian, multi-year) and observed BB%
    weighted by sample size. For large samples (500+ BF), trust observed
    more — the data IS the signal. For small samples, lean on projections.
    This prevents over-regression for sustained-excellence pitchers
    (Skenes: 5.9% career BB over 5000+ pitches → projected 7.1% is too regressed).
    """
    specs = []
    # Blend projected and observed BB% based on sample size
    if "projected_bb_rate" in df.columns and "bb_pct" in df.columns:
        bf = df.get("batters_faced", pd.Series(200, index=df.index))
        # Confidence in observed: sqrt ramp from 0 at 200 BF to 1.0 at 800 BF
        obs_confidence = np.sqrt(((bf.fillna(200) - 200) / 600).clip(0, 1))
        # Blend: more BF → trust observed more, less BF → trust projected
        blended_bb = (
            obs_confidence * df["bb_pct"].fillna(df["projected_bb_rate"])
            + (1 - obs_confidence) * df["projected_bb_rate"].fillna(df["bb_pct"])
        )
        df = df.copy()
        df["blended_bb_rate"] = blended_bb
        bb_col = "blended_bb_rate"
    elif "projected_bb_rate" in df.columns:
        bb_col = "projected_bb_rate"
    else:
        bb_col = "bb_pct"
    if "bb_pct" in pop:
        specs.append((bb_col, 0.35, True, *pop["bb_pct"]))
    if "zone_pct" in pop:
        specs.append(("zone_pct", 0.25, False, *pop["zone_pct"]))
    if "chase_pct" in pop:
        specs.append(("chase_pct", 0.20, False, *pop["chase_pct"]))
    if "first_strike_pct" in df.columns:
        vals = df["first_strike_pct"].dropna()
        if len(vals) > 30:
            specs.append(("first_strike_pct", 0.20, False, vals.mean(), vals.std()))

    if not specs:
        return pd.Series(50, index=df.index, dtype=int)

    bf = df.get("batters_faced")
    z = _composite_z(df, specs, pa=bf, min_pa=50, full_pa=300)
    raw = 50 + 10 * z
    return raw.apply(lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))).astype(int)


def _grade_durability(df: pd.DataFrame) -> pd.Series:
    """Durability tool: workload capacity + health.

    SP: projected_ip 50%, durability_score 30%, health_score 20%.
    RP: games/60 50%, health_score 30%, neutral fill 20%.
    """
    result = pd.Series(50, index=df.index, dtype=int)

    is_sp = df.get("role", pd.Series("SP", index=df.index)) == "SP"

    # SP durability
    if "projected_ip_mean" in df.columns:
        ip = df["projected_ip_mean"].fillna(0)
        # SP: ~170 IP mean, ~40 SD among qualifiers
        sp_z = (ip - 170) / 40
    elif "workload_score" in df.columns:
        from scipy.stats import norm
        sp_z = pd.Series(norm.ppf(df["workload_score"].fillna(0.5).clip(0.01, 0.99).values), index=df.index)
    else:
        sp_z = pd.Series(0.0, index=df.index)

    # RP durability via games
    if "total_games_mean" in df.columns:
        games = df["total_games_mean"].fillna(0)
        rp_z = (games - 55) / 15  # ~55 games mean, ~15 SD for RPs
    else:
        rp_z = pd.Series(0.0, index=df.index)

    # Health component
    health_z = pd.Series(0.0, index=df.index)
    if "health_score" in df.columns:
        from scipy.stats import norm
        h = df["health_score"].fillna(0.5).clip(0.01, 0.99)
        health_z = pd.Series(norm.ppf(h.values), index=df.index)

    # Durability track record
    dur_z = pd.Series(0.0, index=df.index)
    if "durability_score" in df.columns:
        from scipy.stats import norm
        d = df["durability_score"].fillna(0.5).clip(0.01, 0.99)
        dur_z = pd.Series(norm.ppf(d.values), index=df.index)

    # Combine: SP vs RP weights
    sp_composite = 0.50 * sp_z + 0.30 * dur_z + 0.20 * health_z
    rp_composite = 0.50 * rp_z + 0.30 * health_z + 0.20 * pd.Series(0.0, index=df.index)

    composite = np.where(is_sp, sp_composite, rp_composite)
    raw = 50 + 10 * composite
    return pd.Series(raw, index=df.index).apply(
        lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))
    ).astype(int)


def grade_pitcher_tools(
    base: pd.DataFrame,
    season: int = 2025,
) -> pd.DataFrame:
    """Compute 20-80 tool grades for MLB pitchers.

    Parameters
    ----------
    base : pd.DataFrame
        Pitcher rankings base DataFrame. Must have pitcher_id, role,
        and observed pitching stats.
    season : int
        Most recent completed season.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, grade_stuff, grade_command, grade_durability,
        tools_rating.
    """
    # Grade SP and RP against their OWN populations.
    # A 60 Stuff SP = top-16% among starters (harder to achieve).
    # A 60 Stuff RP = top-16% among relievers (different baseline).
    is_sp = base.get("role", pd.Series("SP", index=base.index)) == "SP"
    sp_pop = _compute_pitcher_pop_stats(season, role="SP")
    rp_pop = _compute_pitcher_pop_stats(season, role="RP")

    result = pd.DataFrame({"pitcher_id": base["pitcher_id"]})

    # Grade each role against its own population
    sp_mask = is_sp.values if hasattr(is_sp, "values") else is_sp
    result["grade_stuff"] = 50
    result["grade_command"] = 50
    result["grade_durability"] = _grade_durability(base)

    if sp_mask.any():
        sp_idx = base.index[sp_mask]
        result.loc[sp_idx, "grade_stuff"] = _grade_stuff(base.loc[sp_idx], sp_pop).values
        result.loc[sp_idx, "grade_command"] = _grade_command(base.loc[sp_idx], sp_pop).values

    rp_mask = ~sp_mask
    if rp_mask.any():
        rp_idx = base.index[rp_mask]
        result.loc[rp_idx, "grade_stuff"] = _grade_stuff(base.loc[rp_idx], rp_pop).values
        result.loc[rp_idx, "grade_command"] = _grade_command(base.loc[rp_idx], rp_pop).values

    # Diamond rating: role-specific weights
    sp_dr = _tools_rating(result, _SP_DIAMOND_WEIGHTS)
    rp_dr = _tools_rating(result, _RP_DIAMOND_WEIGHTS)
    tools_dr = np.where(is_sp, sp_dr, rp_dr)

    # Blend tools (85%) with projected FIP (15%) as production guardrail.
    # FIP strips BABIP luck and fielding — measures what the pitcher controls
    # (K, BB, HR). Better guardrail than ERA which includes luck.
    fip_col = "projected_fip"
    if fip_col in base.columns and base[fip_col].notna().any():
        fip_grade = ((6.5 - base[fip_col].fillna(4.5)) / 4.5 * 10).clip(0, 10)
        raw_dr = 0.85 * tools_dr + 0.15 * fip_grade
    else:
        raw_dr = pd.Series(tools_dr, index=result.index)

    # Rescale SP and RP SEPARATELY so both use the full 1-10 range.
    # "9.0 SP" = elite starter, "9.0 RP" = elite reliever — comparable
    # because both represent the same within-role percentile.
    result["tools_rating"] = 5.5
    for mask, label in [(is_sp, "SP"), (~is_sp, "RP")]:
        role_dr = raw_dr[mask]
        if len(role_dr) < 2:
            continue
        dr_min, dr_max = role_dr.min(), role_dr.max()
        if dr_max - dr_min > 1e-9:
            result.loc[mask, "tools_rating"] = (
                1.0 + (role_dr - dr_min) / (dr_max - dr_min) * 9.0
            ).round(1)

    logger.info(
        "Pitcher scouting grades: %d players, avg diamond=%.1f",
        len(result), result["tools_rating"].mean(),
    )
    return result


# ===================================================================
# Per-position fielding grades
# ===================================================================

def grade_position_fielding(
    eligibility: pd.DataFrame,
    season: int = 2025,
) -> pd.DataFrame:
    """Compute 20-80 fielding grades per player per position.

    Uses the per-position OAA from hitter_position_eligibility.parquet.
    Each position's OAA is graded against the MLB population at THAT position
    (a 5 OAA at SS is more impressive than 5 OAA at 1B because the
    population spread differs). No positional boost — that happens at
    team level only.

    Parameters
    ----------
    eligibility : pd.DataFrame
        From hitter_position_eligibility.parquet. Must have player_id,
        position, oaa columns.
    season : int
        For population stats reference.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, starts, pct, is_primary, oaa,
        grade_fielding_at_pos (20-80 grade at this specific position).
    """
    if eligibility.empty or "oaa" not in eligibility.columns:
        return eligibility

    result = eligibility.copy()

    # Grade OAA per position group — each position has its own population
    # distribution (SS OAA spread is wider than 1B)
    result["grade_fielding_at_pos"] = 50  # default

    for pos in result["position"].unique():
        mask = result["position"] == pos
        pos_oaa = result.loc[mask, "oaa"].dropna()
        if len(pos_oaa) < 5:
            continue
        result.loc[mask, "grade_fielding_at_pos"] = _z_grade_series(
            result.loc[mask, "oaa"].fillna(0),
            pop_mean=pos_oaa.mean(),
            pop_std=pos_oaa.std(),
        )

    return result


# ===================================================================
# Prospect grading (MLB population scale, translated rates)
# ===================================================================

def grade_prospect_hitter(
    df: pd.DataFrame,
    mlb_pop: dict[str, tuple[float, float]] | None = None,
    season: int = 2025,
) -> pd.DataFrame:
    """Grade MiLB batting prospects on MLB 20-80 scale.

    Uses translated rates graded against MLB population.
    For metrics unavailable in MiLB, widens effective SD by 1.15x.

    Parameters
    ----------
    df : pd.DataFrame
        Prospect data with translated stats: wtd_k_pct, wtd_bb_pct,
        wtd_iso, sb_rate. Optionally trajectory/age scores.
    mlb_pop : dict, optional
        MLB population stats. Loaded if not provided.
    season : int
        Reference season for population.

    Returns
    -------
    pd.DataFrame
        Present + future tool grades and tools_rating.
    """
    if mlb_pop is None:
        mlb_pop = _compute_hitter_pop_stats(season)

    result = pd.DataFrame({"player_id": df["player_id"]})

    # Uncertainty multiplier for MiLB (fewer inputs than MLB grading)
    UNC = 1.15

    # Hit tool: K% only (no z_contact_pct or xBA in MiLB)
    if "k_pct" in mlb_pop and "wtd_k_pct" in df.columns:
        m, s = mlb_pop["k_pct"]
        result["grade_hit"] = _z_grade_series(df["wtd_k_pct"], m, s * UNC, invert=True)
    else:
        result["grade_hit"] = 50

    # Power: ISO only (no barrel%/EV in most MiLB levels)
    if "iso" in mlb_pop and "wtd_iso" in df.columns:
        m, s = mlb_pop["iso"]
        result["grade_power"] = _z_grade_series(df["wtd_iso"], m, s * UNC, invert=False)
    else:
        result["grade_power"] = 50

    # Speed: SB rate (no sprint speed in MiLB)
    if "sb_rate" in df.columns:
        vals = df["sb_rate"].dropna()
        if len(vals) > 10:
            result["grade_speed"] = _z_grade_series(
                df["sb_rate"], vals.mean(), vals.std() * UNC, invert=False,
            )
        else:
            result["grade_speed"] = 50
    else:
        result["grade_speed"] = 50

    # Fielding: positional default (no OAA in MiLB)
    # Premium positions (C, SS, CF) get 50, others get 45
    if "position" in df.columns:
        premium = df["position"].isin(["C", "SS", "CF"])
        result["grade_fielding"] = np.where(premium, 50, 45)
    else:
        result["grade_fielding"] = 45

    # Discipline: BB% (no chase/whiff in most MiLB)
    if "bb_pct" in mlb_pop and "wtd_bb_pct" in df.columns:
        m, s = mlb_pop["bb_pct"]
        result["grade_discipline"] = _z_grade_series(df["wtd_bb_pct"], m, s * UNC, invert=False)
    else:
        result["grade_discipline"] = 50

    # Present diamond rating
    result["tools_rating"] = _tools_rating(result, _HITTER_DIAMOND_WEIGHTS)

    # Future grades: adjust present by trajectory + age
    for tool in ["hit", "power", "speed", "fielding", "discipline"]:
        present_col = f"grade_{tool}"
        future_col = f"future_{tool}"
        result[future_col] = _compute_future_grade(
            result[present_col], df, tool,
        )

    return result


def grade_prospect_pitcher(
    df: pd.DataFrame,
    mlb_pop: dict[str, tuple[float, float]] | None = None,
    season: int = 2025,
) -> pd.DataFrame:
    """Grade MiLB pitching prospects on MLB 20-80 scale."""
    if mlb_pop is None:
        mlb_pop = _compute_pitcher_pop_stats(season)

    result = pd.DataFrame({"player_id": df["player_id"]})
    UNC = 1.15

    # Stuff: K% (no Stuff+ in MiLB)
    if "k_pct" in mlb_pop and "wtd_k_pct" in df.columns:
        m, s = mlb_pop["k_pct"]
        result["grade_stuff"] = _z_grade_series(df["wtd_k_pct"], m, s * UNC, invert=False)
    else:
        result["grade_stuff"] = 50

    # Command: BB%
    if "bb_pct" in mlb_pop and "wtd_bb_pct" in df.columns:
        m, s = mlb_pop["bb_pct"]
        result["grade_command"] = _z_grade_series(df["wtd_bb_pct"], m, s * UNC, invert=True)
    else:
        result["grade_command"] = 50

    # Durability: default 45 for SP, 40 for RP (unknown workload capacity)
    if "position" in df.columns:
        is_sp = df["position"].isin(["SP", "P"])
        result["grade_durability"] = np.where(is_sp, 45, 40)
    else:
        result["grade_durability"] = 45

    result["tools_rating"] = _tools_rating(result, _SP_DIAMOND_WEIGHTS)

    # Future grades
    for tool in ["stuff", "command", "durability"]:
        result[f"future_{tool}"] = _compute_future_grade(
            result[f"grade_{tool}"], df, tool,
        )

    return result


def _compute_future_grade(
    present: pd.Series,
    df: pd.DataFrame,
    tool: str,
) -> pd.Series:
    """Adjust present grade for future projection.

    Uses trajectory_score (0-1) and age_score (younger = higher) from
    prospect ranking data. Max boost: +15 grade points.
    """
    trajectory = df.get("comp_trajectory", pd.Series(0.5, index=df.index)).fillna(0.5)
    age_score = df.get("comp_age", pd.Series(0.5, index=df.index)).fillna(0.5)

    # Upside multiplier: 0-1
    upside = 0.5 * trajectory + 0.5 * age_score
    # Center at 0.35 so average prospect gets slight positive adjustment
    adjustment = (upside - 0.35) * 23  # range: ~-8 to +15

    future_raw = present + adjustment
    return future_raw.apply(
        lambda x: _round_to_5(np.clip(x, GRADE_MIN, GRADE_MAX))
    ).astype(int)
