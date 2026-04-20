"""
Shared simulation utilities for game_sim package.

Extracted from duplicated code across simulator.py, batter_simulator.py,
lineup_simulator.py, and pa_outcome_model.py.  Zero behavioral changes —
pure deduplication.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.math_helpers import safe_logit

logger = logging.getLogger(__name__)


__all__ = ["safe_logit", "resample_posterior", "rates_to_multinomial_logits"]


# ---------------------------------------------------------------------------
# Posterior resampling
# ---------------------------------------------------------------------------

def resample_posterior(
    arr: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample posterior array to exactly n_sims samples."""
    if len(arr) == n_sims:
        return arr.copy()
    return rng.choice(arr, size=n_sims, replace=True)


# ---------------------------------------------------------------------------
# Multinomial logit conversion
# ---------------------------------------------------------------------------

def rates_to_multinomial_logits(
    k_rate: float | np.ndarray,
    bb_rate: float | np.ndarray,
    hr_rate: float | np.ndarray,
    hbp_rate: float | np.ndarray,
) -> tuple[np.ndarray | float, ...]:
    """Convert outcome rates to multinomial logits with BIP as reference.

    Standard binary logit — log(p/(1-p)) — gives the log-odds of an
    outcome vs *all other outcomes combined*.  A multinomial softmax with
    BIP as the reference category needs log(p/p_BIP) instead.  Using
    binary logits in a softmax systematically deflates K/BB/HR
    probabilities and inflates BIP.

    Parameters
    ----------
    k_rate, bb_rate, hr_rate, hbp_rate : float or np.ndarray
        Per-PA outcome rates (must sum to < 1.0; remainder is BIP).

    Returns
    -------
    tuple of (eta_k, eta_bb, eta_hr, eta_hbp)
        Multinomial logits: log(rate / bip_rate) for each outcome.
    """
    bip_rate = 1.0 - k_rate - bb_rate - hr_rate - hbp_rate
    bip_rate = np.maximum(bip_rate, 0.01)  # safety floor

    # Clip rates to avoid log(0)
    k_safe = np.maximum(k_rate, 1e-6)
    bb_safe = np.maximum(bb_rate, 1e-6)
    hr_safe = np.maximum(hr_rate, 1e-6)
    hbp_safe = np.maximum(hbp_rate, 1e-6)

    return (
        np.log(k_safe / bip_rate),
        np.log(bb_safe / bip_rate),
        np.log(hr_safe / bip_rate),
        np.log(hbp_safe / bip_rate),
    )


# ---------------------------------------------------------------------------
# Matchup shrinkage slopes (Path A calibration, 2026-04-10)
# ---------------------------------------------------------------------------
#
# Slopes fitted via logistic regression of actual PA outcomes (K, BB) against
# the raw matchup logit lift produced by score_matchup / score_matchup_bb.
# The simulator multiplies per-slot raw lifts by these slopes before adding
# them to the PA outcome logit — equivalent to the old scalar dampeners but
# derived from walk-forward train (2020-2023) -> holdout (2024) data rather
# than hand-picked from a backtest.
#
# Fitting script: scripts/fit_matchup_shrinkage.py
# Coefficient cache: data/cached/matchup_shrinkage_coefs.parquet
# Diagnostic: memory/layer2_bvp_diagnostic_2026_04_10.md
#
# HR lifts are disabled at the source (see matchup.py:score_matchup_for_stat
# HR branch). The slope remains as a pass-through scalar for array-shape
# compatibility, but the underlying lift is always zero.

_FALLBACK_MATCHUP_DAMPEN: dict[str, float] = {"k": 0.55, "bb": 0.40, "hr": 0.0}

# BB slope is pinned to the hand-picked value regardless of what the PA-level
# fit produces. The 2026-04-11 A/B on 422 real 2024 starter games showed that
# the fitted BB slope (0.766) amplifies a pre-existing +0.1 BB/game structural
# bias and loses to the hand-picked 0.40 on Poisson log-likelihood, Brier at
# BB > 2.5, and bias. The root problem is that BB calibration slope is ~0.5
# at game level (model overreacts to its own BB signal), which no matchup
# slope can fix. Tracked for follow-up — see memory/bb_game_level_bias.md.
_BB_SLOPE_PIN: float = 0.40

_SIM_UTILS_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SHRINKAGE_COEFS_PATH = (
    _SIM_UTILS_PROJECT_ROOT / "data" / "cached" / "matchup_shrinkage_coefs.parquet"
)


def _load_matchup_dampen() -> dict[str, float]:
    """Load fitted matchup slopes from cache, or fall back to hand-picked."""
    if not _SHRINKAGE_COEFS_PATH.exists():
        logger.warning(
            "Matchup shrinkage coefs not found at %s; using fallback %s. "
            "Run scripts/fit_matchup_shrinkage.py to fit.",
            _SHRINKAGE_COEFS_PATH,
            _FALLBACK_MATCHUP_DAMPEN,
        )
        return dict(_FALLBACK_MATCHUP_DAMPEN)
    df = pd.read_parquet(_SHRINKAGE_COEFS_PATH)
    result = dict(_FALLBACK_MATCHUP_DAMPEN)
    for _, row in df.iterrows():
        stat = str(row["stat"])
        result[stat] = float(row["slope"])
    # HR always zero — Path B short-circuits the HR lift in
    # score_matchup_for_stat and we do not want a fitted slope to
    # accidentally reintroduce it.
    result["hr"] = 0.0
    # BB pinned — game-level A/B preferred the hand-picked value.
    result["bb"] = _BB_SLOPE_PIN
    logger.info("Loaded matchup shrinkage slopes: %s", result)
    return result


MATCHUP_DAMPEN: dict[str, float] = _load_matchup_dampen()


# ---------------------------------------------------------------------------
# Pitcher quality lift computation
# ---------------------------------------------------------------------------

def compute_pitcher_quality_lifts(
    k_rate: float | np.ndarray,
    bb_rate: float | np.ndarray,
    hr_rate: float | np.ndarray,
    league_k: float,
    league_bb: float,
    league_hr: float,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Compute logit-scale pitcher quality lifts vs league average.

    Accepts scalar rates (backward-compatible) or arrays of posterior
    samples.  When arrays are passed the returned lifts have the same
    shape — one lift per sample.
    """
    return (
        safe_logit(k_rate) - safe_logit(league_k),
        safe_logit(bb_rate) - safe_logit(league_bb),
        safe_logit(hr_rate) - safe_logit(league_hr),
    )


def pitcher_rate_to_lift_array(
    rate: float | np.ndarray,
    league_rate: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert a pitcher rate (scalar or posterior array) to (n_sims,) logit lifts.

    When *rate* is a scalar float the lift is a constant array (backward-
    compatible with the old float-only path).  When it is a posterior
    sample array, the samples are resampled to *n_sims* draws and each
    draw gets its own logit lift — propagating pitcher posterior
    uncertainty through every sim independently.
    """
    if np.ndim(rate) == 0:
        lift = float(safe_logit(float(rate)) - safe_logit(league_rate))
        return np.full(n_sims, lift)
    arr = resample_posterior(np.asarray(rate, dtype=np.float64), n_sims, rng)
    return safe_logit(arr) - safe_logit(league_rate)


# ---------------------------------------------------------------------------
# None-to-zero array defaulting
# ---------------------------------------------------------------------------

def default_lift_array(
    arr: np.ndarray | None,
    size: int,
) -> np.ndarray:
    """Return arr if not None, else zeros of given size."""
    return arr if arr is not None else np.zeros(size)
