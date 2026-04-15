"""Shared utility functions and constants for the player rankings system.

Extracted from ``player_rankings.py`` — zero behavioral changes.
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

# ---------------------------------------------------------------------------
# Hitter sub-component weights (sum to 1.0)
# ---------------------------------------------------------------------------
_HITTER_WEIGHTS = {
    "offense": 0.59,
    "fielding": 0.15,
    "baserunning": 0.08,
    "trajectory": 0.09,
    "positional_adj": 0.06,
    "role": 0.02,
    "versatility": 0.01,
}

# Position-dependent fielding scale factors.
# Fielding weight is scaled by position; the excess/deficit is traded 1:1
# with offense weight.  This mirrors fWAR decomposition where defense is
# a larger fraction of total value at premium positions (CF, SS) and a
# smaller fraction at bat-first positions (1B, DH).
#
# Mechanism (extends existing DH redistribution pattern):
#   eff_fielding = _HITTER_WEIGHTS["fielding"] × scale
#   eff_offense  = _HITTER_WEIGHTS["offense"] + _HITTER_WEIGHTS["fielding"] × (1 - scale)
#
# Effective weights per position:
#   CF:    off=53.7%, fld=20.3%  → defense share ≈ 24%
#   SS:    off=56.0%, fld=18.0%  → defense share ≈ 23%
#   C:     off=58.3%, fld=15.8%  → defense share ≈ 22%  (+framing)
#   2B/3B: off=59.0%, fld=15.0%  → defense share ≈ 19%
#   LF/RF: off=61.3%, fld=12.8%  → defense share ≈ 15%
#   1B:    off=66.5%, fld= 7.5%  → defense share ≈  9%
#   DH:    off=74.0%, fld= 0.0%  → defense share =  0%
_POS_FIELDING_SCALE: dict[str, float] = {
    "CF": 1.35,
    "SS": 1.20,
    "C": 1.05,
    "2B": 1.00,
    "3B": 1.00,
    "RF": 0.85,
    "LF": 0.85,
    "1B": 0.50,
    "DH": 0.00,
}

# FanGraphs positional adjustment per 162 games (runs):
# C: +12.5, SS: +7.5, 2B/3B/CF: +2.5, LF/RF: -7.5, 1B: -12.5, DH: -17.5
# Normalized to 0-1 scale (DH=0, C=1)
_POSITIONAL_OFFENSE_ADJ: dict[str, float] = {
    "C": 1.00, "SS": 0.833, "CF": 0.667, "2B": 0.667, "3B": 0.667,
    "LF": 0.333, "RF": 0.333, "1B": 0.167, "DH": 0.0,
}

# Offense sub-weights: dynamic blend based on PA (see _dynamic_blend_weights)
_PROJ_WEIGHT_BASE = 0.40  # at ~400 PA
_OBS_WEIGHT_BASE = 0.60

# ---------------------------------------------------------------------------
# Pitcher sub-component weights by role (sum to 1.0)
# ---------------------------------------------------------------------------
_SP_WEIGHTS = {
    # Reweighted 2026-04-09 to apply the same talent-first philosophy
    # used for hitters: elite stuff/command shouldn't be penalized by
    # historical injury volatility that's already reflected in the
    # sim's BF projection downstream.
    #
    # Workload 12 -> 7, health 11 -> 7 (combined 23 -> 14, -9 points).
    # Redistributed: stuff +5, cmd +2, trajectory +2.  Stuff becomes
    # the dominant signal (32%) matching RP weighting, which better
    # rewards K-dominant aces (Crochet, Skenes, Ragans) whose injury
    # histories were penalizing them against lower-talent but durable
    # control pitchers (Gilbert, Sánchez).
    "stuff": 0.32,
    "command": 0.24,
    "workload": 0.07,
    "health": 0.07,
    "role": 0.00,
    "trajectory": 0.22,
    "glicko": 0.08,
}
_RP_WEIGHTS = {
    # Rebalanced 2026-04-10 to be K/stuff-centric for 1-inning relievers.
    # For short-burst high-leverage work, strikeouts dominate the value
    # equation: walks matter less than for SPs because closers only
    # face 3-4 batters.  Previous weights (stuff 32, command 22,
    # trajectory 26, glicko 10) gave pitch-to-contact setups (Speier,
    # Morejon) inflated ranks over elite-K closers (Díaz, Hader, Williams,
    # Mason Miller) whose command scores were merely average.
    #
    # Shifts: stuff +8 (32->40), command -7 (22->15), trajectory -6
    # (26->20), glicko +5 (10->15).  Glicko is outcome-based (did the
    # batters get out?) which is exactly what matters for closers.
    "stuff": 0.40,
    "command": 0.15,
    "workload": 0.03,
    "health": 0.07,
    "role": 0.00,
    "trajectory": 0.20,
    "glicko": 0.15,
}

# fWAR-style positional adjustments (runs per 162 games, normalized to 0-1)
# Used for overall_rank only -- pos_rank stays within-position.
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1, higher = better)."""
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
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std()
    if sd < 1e-6:
        return pd.Series(0.5, index=series.index)
    z = (s - mu) / sd
    # Sigmoid maps z to (0, 1) -- steepness 1.0 gives good spread
    return pd.Series(1.0 / (1.0 + np.exp(-z.values)), index=series.index)


def _inv_zscore_pctl(series: pd.Series) -> pd.Series:
    """Inverse z-score percentile (lower raw value = higher score)."""
    return _zscore_pctl(-series)


def _dynamic_blend_weights(pa: pd.Series, min_pa: int = 150, full_pa: int = 600) -> tuple[pd.Series, pd.Series]:
    """Compute per-player observed/projected blend weights based on PA.

    More PA -> trust observed more. Fewer PA -> lean on projections.

    Returns (proj_weight, obs_weight) Series that sum to 1.0 per row.
    """
    # Linearly scale observed weight from 0.35 (at min_pa) to 0.75 (at full_pa)
    frac = ((pa - min_pa) / (full_pa - min_pa)).clip(0, 1)
    obs_w = 0.35 + 0.40 * frac
    proj_w = 1.0 - obs_w
    return proj_w, obs_w


def _hitter_age_factor(age: pd.Series) -> pd.Series:
    """Forward-looking hitter trajectory curve (piecewise linear).

    Models expected *future* value trajectory, not current aging level.
    Young players approaching their prime get upside credit; aging
    players past peak get penalized proportionally to expected decline.

    Based on industry consensus (FanGraphs, ZiPS, Marcel, OOPSY):
    - Youth upside (20-24): climbing toward peak, years of prime ahead.
    - Prime plateau (25-29): at peak, stable production expected.
    - Early decline (30-33): ~5.5% per year as skills erode.
    - Late decline (34+): accelerating loss, steep drop-off.

    Calibrated against FanGraphs Depth Charts fWAR ranking comparison.
    Research-aligned: rate-stat aging studies (FG, OOPSY) show 2-3%/yr
    decline from 30-33, with talent-level moderation (elite producers
    decline slower).  5.5%/yr balances talent-averaged decline with
    forward-looking value discount.
    """
    # Phase 1: Youth upside (20 -> 24) — approaching prime
    youth = 0.85 + 0.15 * ((age - 20) / 5.0).clip(0, 1)
    # Phase 2: Prime (25 -> 29) — mild arc peaking at 27.
    # Old: flat 1.0 for all 185 players aged 25-29 (zero differentiation).
    # New: peaks at 27 (1.0), tapers to 0.97 at edges (25, 29).
    # Spread is small (0.03) but restores rank ordering within the
    # largest population segment.
    prime = 1.0 - 0.03 * ((age - 27).abs() / 2.0).clip(0, 1)
    # Phase 3: Early decline (30 -> 33, ~5.5%/yr -> 0.75 at 33)
    # Was 0.97 -> 0.60 (~10%/yr) which penalized aging elite producers
    # like Judge too heavily.  Research shows 2-3%/yr for rate stats;
    # 5.5%/yr accounts for declining playing time and injury risk.
    early_decline = 0.97 - 0.22 * ((age - 29) / 4.0).clip(0, 1)
    # Phase 4: Late career decline (34 -> 39)
    late_decline = 0.75 - 0.50 * ((age - 33) / 6.0).clip(0, 1)

    raw = np.where(
        age < 25, youth,
        np.where(age <= 29, prime,
                 np.where(age <= 33, early_decline, late_decline))
    )
    return pd.Series(raw, index=age.index).clip(0.10, 1)


def _pitcher_age_factor(age: pd.Series) -> pd.Series:
    """Research-aligned pitcher aging curve (piecewise linear).

    Revised 2026-04-10 (v2): previous curve peaked at 28 mirroring the
    hitter shape, but pitcher peak is demonstrably earlier — elite
    velocity (the dominant stuff input) typically peaks 24-27 and erodes
    after 28.  Old curve gave Whitlock (29) 0.985 but Mason Miller (26)
    only 0.97, effectively treating young elite arms as "not yet peak"
    when they're actually maximizing the upside window.

    New shape shifts the peak earlier and plateaus 24-27:
    - Youth climb (20-23): 0.88 → 0.97 (rookies already delivering value)
    - Peak plateau (24-27): 0.97 → 1.00 → 1.00 (young arm prime)
    - Early decline (28-31): 1.00 → 0.91 (velo slipping, ~3%/yr)
    - Mid decline (32-34): 0.91 → 0.75 (~5%/yr)
    - Late decline (35-40): 0.75 → 0.15 (accelerating velo loss)
    """
    # Phase 1: Youth climb (20 -> 22): 0.88 -> 1.00
    youth = 0.88 + 0.12 * ((age - 20) / 3.0).clip(0, 1)
    # Phase 2: Peak plateau (23 -> 27): flat at 1.00
    peak = pd.Series(1.00, index=age.index)
    # Phase 3: Early decline (28 -> 31): 1.00 -> 0.91 (~3%/yr)
    early_decline = 1.00 - 0.09 * ((age - 27) / 4.0).clip(0, 1)
    # Phase 4: Mid decline (32 -> 34): 0.91 -> 0.75 (~5%/yr)
    mid_decline = 0.91 - 0.16 * ((age - 31) / 3.0).clip(0, 1)
    # Phase 5: Late decline (35 -> 40): steep velo-driven drop
    late_decline = 0.75 - 0.60 * ((age - 34) / 6.0).clip(0, 1)

    raw = np.where(
        age < 23, youth,
        np.where(age <= 27, peak,
                 np.where(age <= 31, early_decline,
                          np.where(age <= 34, mid_decline, late_decline)))
    )
    return pd.Series(raw, index=age.index).clip(0.10, 1)


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
