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


__all__ = ["safe_logit", "resample_posterior"]


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
    k_rate: float,
    bb_rate: float,
    hr_rate: float,
    league_k: float,
    league_bb: float,
    league_hr: float,
) -> tuple[float, float, float]:
    """Compute logit-scale pitcher quality lifts vs league average."""
    return (
        safe_logit(k_rate) - safe_logit(league_k),
        safe_logit(bb_rate) - safe_logit(league_bb),
        safe_logit(hr_rate) - safe_logit(league_hr),
    )


# ---------------------------------------------------------------------------
# None-to-zero array defaulting
# ---------------------------------------------------------------------------

def default_lift_array(
    arr: np.ndarray | None,
    size: int,
) -> np.ndarray:
    """Return arr if not None, else zeros of given size."""
    return arr if arr is not None else np.zeros(size)
