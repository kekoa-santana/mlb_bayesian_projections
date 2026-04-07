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
    "offense": 0.63,
    "fielding": 0.13,
    "baserunning": 0.08,
    "trajectory": 0.08,
    "positional_adj": 0.05,
    "role": 0.02,
    "versatility": 0.01,
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
    # Walk-forward validated 2022-2025, then calibrated 2026-03-28.
    # Workload raised from 0.05 -> 0.12: for SPs, innings availability IS value.
    # A 3.00 ERA over 160 IP is worth more than 2.50 ERA over 100 IP.
    # Trajectory 0.25 -> 0.20, glicko 0.10 -> 0.08 to fund the increase.
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
    """Research-aligned hitter aging curve (piecewise linear).

    Based on industry consensus (FanGraphs, ZiPS, Marcel, OOPSY):
    - Plateau 26-29: wRC+ peaks 26-27, ISO holds to ~28-29, BB%
      improves through 29.  No penalty during prime window.
    - Gradual decline 30-34: ~3-4% of peak per year (~0.5 WAR/yr)
    - Steep decline 35-40: accelerating loss

    Uses explicit biology-based curve, not population-relative _inv_pctl,
    because aging is biological -- a 33-year-old declines regardless of
    how young or old the league population is.
    """
    # Phase 1: Development climb (20 -> 26)
    climb = 0.60 + 0.40 * ((age - 20) / 6.0).clip(0, 1)
    # Phase 2: Prime plateau (26 -> 29) -- no decline
    plateau = 1.0
    # Phase 3: Gradual post-prime decline (30 -> 34, ~4%/yr -> 0.80 at 34)
    slow_decline = 1.0 - 0.20 * ((age - 29) / 5.0).clip(0, 1)
    # Phase 4: Steep late-career decline (35 -> 40)
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
    # Phase 1: Development climb (20 -> 27)
    climb = 0.55 + 0.45 * ((age - 20) / 7.0).clip(0, 1)
    # Phase 2: Prime plateau (27 -> 30) -- no decline
    plateau = 1.0
    # Phase 3: Gradual post-prime decline (31 -> 35, ~4%/yr -> 0.80 at 35)
    slow_decline = 1.0 - 0.20 * ((age - 30) / 5.0).clip(0, 1)
    # Phase 4: Steep late-career decline (36 -> 41)
    steep_decline = 0.80 - 0.60 * ((age - 35) / 6.0).clip(0, 1)

    raw = np.where(
        age < 27, climb,
        np.where(age <= 30, plateau,
                 np.where(age <= 35, slow_decline, steep_decline))
    )
    return pd.Series(raw, index=age.index).clip(0, 1)


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
