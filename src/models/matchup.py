"""
Pitch-type matchup scoring module (Layer 2).

Quantifies how a pitcher's arsenal maps onto a specific hitter's
vulnerabilities using an odds-ratio (log-odds additive) method.

The matchup lift is purely bottom-up — computed from pitch-type profiles
with no parameters fit to game outcomes — so same-season profiles are
valid for in-sample evaluation.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils.constants import LEAGUE_AVG_BY_PITCH_TYPE, PITCH_TO_FAMILY

logger = logging.getLogger(__name__)

# Clipping bounds for logit to avoid infinities
_CLIP_LO = 1e-4
_CLIP_HI = 1.0 - 1e-4

# Minimum pitches for a pitch type to be included in a pitcher's arsenal
_MIN_PITCHES = 20

# Sample-size threshold for full reliability on hitter whiff data
_RELIABILITY_N = 50


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _logit(p: float | np.ndarray) -> float | np.ndarray:
    """Log-odds transform with clipping to avoid infinities.

    Parameters
    ----------
    p : float or array-like
        Probability in (0, 1).

    Returns
    -------
    float or ndarray
        logit(p) = log(p / (1 - p)).
    """
    p = np.clip(p, _CLIP_LO, _CLIP_HI)
    return np.log(p / (1.0 - p))


def _inv_logit(x: float | np.ndarray) -> float | np.ndarray:
    """Inverse logit (sigmoid) transform.

    Parameters
    ----------
    x : float or array-like
        Log-odds value.

    Returns
    -------
    float or ndarray
        1 / (1 + exp(-x)).
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ---------------------------------------------------------------------------
# Hitter whiff fallback chain
# ---------------------------------------------------------------------------
def _get_hitter_whiff_with_fallback(
    hitter_vuln: pd.DataFrame,
    batter_id: int,
    pitch_type: str,
    league_whiff: float,
) -> tuple[float, float]:
    """Look up a hitter's whiff rate for a pitch type with fallback.

    Fallback chain
    --------------
    1. Direct match on (batter_id, pitch_type) — reliability =
       min(swings, 50) / 50, blended toward league baseline.
    2. Pitch-family fallback — weighted average of same-family pitch types,
       reliability scaled by 0.5.
    3. League baseline — hitter_delta = 0 (average vulnerability).

    Parameters
    ----------
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles with columns: batter_id, pitch_type,
        swings, whiff_rate, pitch_family.
    batter_id : int
        Batter MLB ID.
    pitch_type : str
        Pitch type abbreviation (e.g. "FF", "SL").
    league_whiff : float
        League-average whiff rate for this pitch type.

    Returns
    -------
    tuple[float, float]
        (whiff_rate, reliability) where reliability is in [0, 1].
    """
    batter_rows = hitter_vuln[hitter_vuln["batter_id"] == batter_id]

    # Level 1: direct match
    direct = batter_rows[batter_rows["pitch_type"] == pitch_type]
    if len(direct) > 0:
        row = direct.iloc[0]
        swings = row.get("swings", 0)
        if pd.notna(swings) and swings > 0:
            raw_whiff = row["whiff_rate"]
            if pd.notna(raw_whiff):
                reliability = min(float(swings), _RELIABILITY_N) / _RELIABILITY_N
                blended = reliability * raw_whiff + (1.0 - reliability) * league_whiff
                return float(blended), reliability

    # Level 2: family fallback
    target_family = PITCH_TO_FAMILY.get(pitch_type)
    if target_family is not None and len(batter_rows) > 0:
        family_rows = batter_rows[
            batter_rows["pitch_family"] == target_family
        ]
        if len(family_rows) > 0:
            valid = family_rows.dropna(subset=["whiff_rate"])
            valid = valid[valid["swings"] > 0]
            if len(valid) > 0:
                total_swings = valid["swings"].sum()
                weighted_whiff = (
                    (valid["whiff_rate"] * valid["swings"]).sum() / total_swings
                )
                raw_reliability = min(float(total_swings), _RELIABILITY_N) / _RELIABILITY_N
                reliability = raw_reliability * 0.5  # discount for indirect match
                blended = reliability * weighted_whiff + (1.0 - reliability) * league_whiff
                return float(blended), reliability

    # Level 3: league baseline
    return league_whiff, 0.0


# ---------------------------------------------------------------------------
# Single matchup scoring
# ---------------------------------------------------------------------------
def score_matchup(
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Score a single pitcher-batter matchup using log-odds additive method.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    batter_id : int
        Batter MLB ID.
    pitcher_arsenal : pd.DataFrame
        Pitcher arsenal profiles (pitcher_id, pitch_type, pitches, usage_pct,
        whiff_rate, swings).
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles (batter_id, pitch_type, swings,
        whiff_rate, pitch_family).
    baselines_pt : dict[str, dict[str, float]]
        League-average baselines keyed by pitch_type with at least
        "whiff_rate" key.

    Returns
    -------
    dict
        Keys: pitcher_id, batter_id, matchup_whiff_rate, baseline_whiff_rate,
        matchup_k_logit_lift, n_pitch_types, avg_reliability.
    """
    p_arsenal = pitcher_arsenal[pitcher_arsenal["pitcher_id"] == pitcher_id].copy()

    # Filter to pitch types with sufficient volume
    p_arsenal = p_arsenal[p_arsenal["pitches"] >= _MIN_PITCHES].copy()
    if len(p_arsenal) == 0:
        return {
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "matchup_whiff_rate": np.nan,
            "baseline_whiff_rate": np.nan,
            "matchup_k_logit_lift": 0.0,
            "n_pitch_types": 0,
            "avg_reliability": 0.0,
        }

    # Renormalize usage
    total_usage = p_arsenal["usage_pct"].sum()
    if total_usage > 0:
        p_arsenal["usage_norm"] = p_arsenal["usage_pct"] / total_usage
    else:
        p_arsenal["usage_norm"] = 1.0 / len(p_arsenal)

    matchup_whiff = 0.0
    baseline_whiff = 0.0
    reliabilities = []

    for _, row in p_arsenal.iterrows():
        pt = row["pitch_type"]
        usage = row["usage_norm"]
        pitcher_whiff = row.get("whiff_rate", np.nan)

        # League baseline for this pitch type
        league_whiff = baselines_pt.get(pt, {}).get(
            "whiff_rate", LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get("whiff_rate", 0.25)
        )

        if pd.isna(pitcher_whiff):
            pitcher_whiff = league_whiff

        # Hitter whiff with fallback
        hitter_whiff, reliability = _get_hitter_whiff_with_fallback(
            hitter_vuln, batter_id, pt, league_whiff
        )
        reliabilities.append(reliability)

        # Log-odds additive method
        league_logit = _logit(league_whiff)
        pitcher_delta = _logit(pitcher_whiff) - league_logit
        hitter_delta = _logit(hitter_whiff) - league_logit
        matchup_logit = league_logit + pitcher_delta + hitter_delta
        matchup_whiff_pt = float(_inv_logit(matchup_logit))

        matchup_whiff += usage * matchup_whiff_pt
        baseline_whiff += usage * pitcher_whiff

    # Compute lift in logit space
    matchup_k_logit_lift = float(_logit(matchup_whiff) - _logit(baseline_whiff))

    return {
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "matchup_whiff_rate": matchup_whiff,
        "baseline_whiff_rate": baseline_whiff,
        "matchup_k_logit_lift": matchup_k_logit_lift,
        "n_pitch_types": len(p_arsenal),
        "avg_reliability": float(np.mean(reliabilities)) if reliabilities else 0.0,
    }


# ---------------------------------------------------------------------------
# Batch matchup scoring
# ---------------------------------------------------------------------------
def score_matchups_batch(
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
    matchup_pairs: list[tuple[int, int]],
) -> pd.DataFrame:
    """Score multiple pitcher-batter matchups.

    Parameters
    ----------
    pitcher_arsenal : pd.DataFrame
        Full pitcher arsenal profiles for the season.
    hitter_vuln : pd.DataFrame
        Full hitter vulnerability profiles for the season.
    baselines_pt : dict[str, dict[str, float]]
        League baselines keyed by pitch_type.
    matchup_pairs : list[tuple[int, int]]
        List of (pitcher_id, batter_id) pairs to score.

    Returns
    -------
    pd.DataFrame
        One row per matchup pair with score columns.
    """
    results = []
    for pitcher_id, batter_id in matchup_pairs:
        result = score_matchup(
            pitcher_id, batter_id,
            pitcher_arsenal, hitter_vuln, baselines_pt,
        )
        results.append(result)

    df = pd.DataFrame(results)
    logger.info("Scored %d matchups", len(df))
    return df


# ---------------------------------------------------------------------------
# Game-level matchup-adjusted K rate
# ---------------------------------------------------------------------------
def compute_game_matchup_k_rate(
    pitcher_baseline_k_rate: float,
    game_batter_pa: pd.DataFrame,
    matchup_scores: pd.DataFrame,
) -> dict[str, Any]:
    """Compute a matchup-adjusted K prediction for a single game.

    For each batter in the game lineup, adjusts the pitcher's baseline K rate
    by the matchup lift. Batters without matchup data use the raw baseline.

    Parameters
    ----------
    pitcher_baseline_k_rate : float
        Pitcher's season-level K rate (K / BF).
    game_batter_pa : pd.DataFrame
        Per-batter PA counts for this game. Columns: batter_id, pa.
    matchup_scores : pd.DataFrame
        Matchup scores with columns: batter_id, matchup_k_logit_lift.

    Returns
    -------
    dict
        Keys: game_adjusted_k_rate, predicted_k_matchup, predicted_k_baseline,
        total_bf, n_matched, n_total.
    """
    total_pa = game_batter_pa["pa"].sum()
    if total_pa == 0:
        return {
            "game_adjusted_k_rate": pitcher_baseline_k_rate,
            "predicted_k_matchup": 0.0,
            "predicted_k_baseline": 0.0,
            "total_bf": 0,
            "n_matched": 0,
            "n_total": 0,
        }

    baseline_logit = _logit(pitcher_baseline_k_rate)

    # Merge matchup lifts onto game batters
    merged = game_batter_pa.merge(
        matchup_scores[["batter_id", "matchup_k_logit_lift"]],
        on="batter_id",
        how="left",
    )
    merged["matchup_k_logit_lift"] = merged["matchup_k_logit_lift"].fillna(0.0)

    # Per-batter adjusted K probability
    merged["adjusted_k_prob"] = _inv_logit(
        baseline_logit + merged["matchup_k_logit_lift"]
    )

    # PA-weighted game-level K rate
    game_adjusted_k_rate = float(
        (merged["adjusted_k_prob"] * merged["pa"]).sum() / total_pa
    )
    predicted_k_matchup = float(
        (merged["adjusted_k_prob"] * merged["pa"]).sum()
    )
    predicted_k_baseline = float(pitcher_baseline_k_rate * total_pa)
    n_matched = int((merged["matchup_k_logit_lift"] != 0.0).sum())
    n_total = len(merged)

    return {
        "game_adjusted_k_rate": game_adjusted_k_rate,
        "predicted_k_matchup": predicted_k_matchup,
        "predicted_k_baseline": predicted_k_baseline,
        "total_bf": int(total_pa),
        "n_matched": n_matched,
        "n_total": n_total,
    }
