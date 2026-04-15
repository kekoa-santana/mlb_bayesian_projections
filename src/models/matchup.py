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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.constants import CLIP_LO, CLIP_HI, LEAGUE_AVG_BY_PITCH_TYPE, LEAGUE_AVG_OVERALL, PITCH_TO_FAMILY

logger = logging.getLogger(__name__)

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
    p = np.clip(p, CLIP_LO, CLIP_HI)
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


# ===================================================================
# Archetype-based matchup scoring
# ===================================================================

def _get_hitter_whiff_with_fallback_archetype(
    hitter_vuln_arch: pd.DataFrame,
    batter_id: int,
    archetype: int,
    league_whiff: float,
    cluster_metadata: pd.DataFrame | None = None,
    hitter_vuln_pt: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
) -> tuple[float, float]:
    """Look up a hitter's whiff rate for a pitch archetype with fallback.

    Fallback chain
    --------------
    1. Direct match on (batter_id, pitch_archetype) — reliability =
       min(swings, 50) / 50, blended toward league archetype baseline.
    2. Primary pitch_type of that archetype (from cluster metadata) →
       look up in pitch_type hitter vuln.
    3. League archetype baseline — hitter_delta = 0.

    Parameters
    ----------
    hitter_vuln_arch : pd.DataFrame
        Hitter vulnerability by archetype with columns: batter_id,
        pitch_archetype, swings, whiff_rate.
    batter_id : int
        Batter MLB ID.
    archetype : int
        Pitch archetype cluster ID.
    league_whiff : float
        League-average whiff rate for this archetype.
    cluster_metadata : pd.DataFrame | None
        Cluster summaries with pitch_archetype and primary_pitch_type.
    hitter_vuln_pt : pd.DataFrame | None
        Fallback hitter vulnerability by pitch_type.
    baselines_pt : dict | None
        Pitch-type-level league baselines for the fallback.

    Returns
    -------
    tuple[float, float]
        (whiff_rate, reliability).
    """
    batter_rows = hitter_vuln_arch[hitter_vuln_arch["batter_id"] == batter_id]

    # Level 1: direct archetype match
    direct = batter_rows[batter_rows["pitch_archetype"] == archetype]
    if len(direct) > 0:
        row = direct.iloc[0]
        swings = row.get("swings", 0)
        if pd.notna(swings) and swings > 0:
            raw_whiff = row["whiff_rate"]
            if pd.notna(raw_whiff):
                reliability = min(float(swings), _RELIABILITY_N) / _RELIABILITY_N
                blended = reliability * raw_whiff + (1.0 - reliability) * league_whiff
                return float(blended), reliability

    # Level 2: primary pitch_type fallback via cluster metadata
    if cluster_metadata is not None and hitter_vuln_pt is not None:
        meta = cluster_metadata[cluster_metadata["pitch_archetype"] == archetype]
        if len(meta) > 0:
            primary_pt = meta.iloc[0].get("primary_pitch_type")
            if primary_pt is not None and pd.notna(primary_pt):
                pt_league = (baselines_pt or {}).get(primary_pt, {}).get(
                    "whiff_rate", league_whiff
                )
                whiff, rel = _get_hitter_whiff_with_fallback(
                    hitter_vuln_pt, batter_id, primary_pt, pt_league
                )
                if rel > 0:
                    return whiff, rel * 0.7  # discount for cross-reference

    # Level 3: league baseline
    return league_whiff, 0.0


def score_matchup_by_archetype(
    pitcher_id: int,
    batter_id: int,
    pitcher_offerings: pd.DataFrame,
    hitter_vuln_arch: pd.DataFrame,
    baselines_arch: dict[int, dict[str, float]],
    cluster_metadata: pd.DataFrame | None = None,
    hitter_vuln_pt: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Score a single pitcher-batter matchup using pitch archetypes.

    Same log-odds additive math as ``score_matchup``, but grouped by
    pitch archetype instead of raw pitch_type.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    batter_id : int
        Batter MLB ID.
    pitcher_offerings : pd.DataFrame
        Pitcher offerings with pitch_archetype, pitches, usage_pct,
        whiff_rate columns.
    hitter_vuln_arch : pd.DataFrame
        Hitter vulnerability by archetype (batter_id, pitch_archetype,
        swings, whiff_rate).
    baselines_arch : dict[int, dict[str, float]]
        League baselines keyed by archetype int with at least
        "whiff_rate" key.
    cluster_metadata : pd.DataFrame | None
        Cluster summaries for fallback chain.
    hitter_vuln_pt : pd.DataFrame | None
        Pitch-type hitter vuln for fallback.
    baselines_pt : dict | None
        Pitch-type baselines for fallback.

    Returns
    -------
    dict
        Same schema as ``score_matchup``.
    """
    p_off = pitcher_offerings[
        pitcher_offerings["pitcher_id"] == pitcher_id
    ].copy()

    # Filter to offerings with sufficient volume
    p_off = p_off[p_off["pitches"] >= _MIN_PITCHES].copy()
    if len(p_off) == 0:
        return {
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "matchup_whiff_rate": np.nan,
            "baseline_whiff_rate": np.nan,
            "matchup_k_logit_lift": 0.0,
            "n_pitch_types": 0,
            "avg_reliability": 0.0,
        }

    # Aggregate to pitcher-level per archetype
    arch_agg = p_off.groupby("pitch_archetype", as_index=False).agg(
        pitches=("pitches", "sum"),
        swings=("swings", "sum"),
        whiffs=("whiffs", "sum"),
    )
    arch_agg["whiff_rate"] = arch_agg["whiffs"] / arch_agg["swings"].replace(0, np.nan)
    total_pitches = arch_agg["pitches"].sum()
    arch_agg["usage_norm"] = arch_agg["pitches"] / total_pitches if total_pitches > 0 else 1.0 / len(arch_agg)

    matchup_whiff = 0.0
    baseline_whiff = 0.0
    reliabilities = []

    for _, row in arch_agg.iterrows():
        arch = int(row["pitch_archetype"])
        usage = row["usage_norm"]
        pitcher_whiff = row.get("whiff_rate", np.nan)

        # League baseline for this archetype
        league_whiff = baselines_arch.get(arch, {}).get(
            "whiff_rate", LEAGUE_AVG_OVERALL.get("whiff_rate", 0.25)
        )

        if pd.isna(pitcher_whiff):
            pitcher_whiff = league_whiff

        # Hitter whiff with fallback
        hitter_whiff, reliability = _get_hitter_whiff_with_fallback_archetype(
            hitter_vuln_arch, batter_id, arch, league_whiff,
            cluster_metadata=cluster_metadata,
            hitter_vuln_pt=hitter_vuln_pt,
            baselines_pt=baselines_pt,
        )
        reliabilities.append(reliability)

        # Log-odds additive method
        league_logit = _logit(league_whiff)
        pitcher_delta = _logit(pitcher_whiff) - league_logit
        hitter_delta = _logit(hitter_whiff) - league_logit
        matchup_logit = league_logit + pitcher_delta + hitter_delta
        matchup_whiff_arch = float(_inv_logit(matchup_logit))

        matchup_whiff += usage * matchup_whiff_arch
        baseline_whiff += usage * pitcher_whiff

    matchup_k_logit_lift = float(_logit(matchup_whiff) - _logit(baseline_whiff))

    return {
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "matchup_whiff_rate": matchup_whiff,
        "baseline_whiff_rate": baseline_whiff,
        "matchup_k_logit_lift": matchup_k_logit_lift,
        "n_pitch_types": len(arch_agg),
        "avg_reliability": float(np.mean(reliabilities)) if reliabilities else 0.0,
    }


def score_matchups_batch_by_archetype(
    pitcher_offerings: pd.DataFrame,
    hitter_vuln_arch: pd.DataFrame,
    baselines_arch: dict[int, dict[str, float]],
    matchup_pairs: list[tuple[int, int]],
    cluster_metadata: pd.DataFrame | None = None,
    hitter_vuln_pt: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Score multiple pitcher-batter matchups using pitch archetypes.

    Parameters
    ----------
    pitcher_offerings : pd.DataFrame
        Full pitcher offerings with pitch_archetype column.
    hitter_vuln_arch : pd.DataFrame
        Full hitter vulnerability by archetype.
    baselines_arch : dict[int, dict[str, float]]
        League baselines keyed by archetype int.
    matchup_pairs : list[tuple[int, int]]
        List of (pitcher_id, batter_id) pairs to score.
    cluster_metadata : pd.DataFrame | None
        Cluster summaries for fallback chain.
    hitter_vuln_pt : pd.DataFrame | None
        Pitch-type hitter vuln for fallback.
    baselines_pt : dict | None
        Pitch-type baselines for fallback.

    Returns
    -------
    pd.DataFrame
        One row per matchup pair with score columns.
    """
    results = []
    for pitcher_id, batter_id in matchup_pairs:
        result = score_matchup_by_archetype(
            pitcher_id, batter_id,
            pitcher_offerings, hitter_vuln_arch, baselines_arch,
            cluster_metadata=cluster_metadata,
            hitter_vuln_pt=hitter_vuln_pt,
            baselines_pt=baselines_pt,
        )
        results.append(result)

    df = pd.DataFrame(results)
    logger.info("Scored %d archetype matchups", len(df))
    return df


# ===================================================================
# BB and HR matchup scoring
# ===================================================================

# ---------------------------------------------------------------------------
# Hitter chase-rate fallback chain (for BB matchup)
# ---------------------------------------------------------------------------
def _get_hitter_chase_with_fallback(
    hitter_vuln: pd.DataFrame,
    batter_id: int,
    pitch_type: str,
    league_chase: float,
) -> tuple[float, float]:
    """Look up a hitter's chase rate for a pitch type with fallback.

    Same fallback chain as ``_get_hitter_whiff_with_fallback`` but uses
    ``chase_rate`` (chase_swings / out_of_zone_pitches).

    Fallback chain
    --------------
    1. Direct match on (batter_id, pitch_type) — reliability =
       min(out_of_zone_pitches, 50) / 50, blended toward league baseline.
    2. Pitch-family fallback — weighted average of same-family pitch types,
       reliability scaled by 0.5.
    3. League baseline — hitter_delta = 0 (average chase tendency).

    Parameters
    ----------
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles with columns: batter_id, pitch_type,
        out_of_zone_pitches, chase_swings, chase_rate, pitch_family.
    batter_id : int
        Batter MLB ID.
    pitch_type : str
        Pitch type abbreviation (e.g. "FF", "SL").
    league_chase : float
        League-average chase rate for this pitch type.

    Returns
    -------
    tuple[float, float]
        (chase_rate, reliability) where reliability is in [0, 1].
    """
    batter_rows = hitter_vuln[hitter_vuln["batter_id"] == batter_id]

    # Level 1: direct match
    direct = batter_rows[batter_rows["pitch_type"] == pitch_type]
    if len(direct) > 0:
        row = direct.iloc[0]
        ooz = row.get("out_of_zone_pitches", 0)
        if pd.notna(ooz) and ooz > 0:
            raw_chase = row.get("chase_rate", np.nan)
            if pd.notna(raw_chase):
                reliability = min(float(ooz), _RELIABILITY_N) / _RELIABILITY_N
                blended = reliability * raw_chase + (1.0 - reliability) * league_chase
                return float(blended), reliability

    # Level 2: family fallback
    target_family = PITCH_TO_FAMILY.get(pitch_type)
    if target_family is not None and len(batter_rows) > 0:
        family_rows = batter_rows[
            batter_rows["pitch_family"] == target_family
        ]
        if len(family_rows) > 0:
            valid = family_rows.dropna(subset=["chase_rate"])
            valid = valid[valid.get("out_of_zone_pitches", valid.get("swings", pd.Series(dtype=float))) > 0]
            if len(valid) > 0:
                total_ooz = valid["out_of_zone_pitches"].sum()
                if total_ooz > 0:
                    weighted_chase = (
                        (valid["chase_rate"] * valid["out_of_zone_pitches"]).sum()
                        / total_ooz
                    )
                    raw_reliability = min(float(total_ooz), _RELIABILITY_N) / _RELIABILITY_N
                    reliability = raw_reliability * 0.5  # discount for indirect match
                    blended = reliability * weighted_chase + (1.0 - reliability) * league_chase
                    return float(blended), reliability

    # Level 3: league baseline
    return league_chase, 0.0


# ---------------------------------------------------------------------------
# BB matchup scoring
# ---------------------------------------------------------------------------
def score_matchup_bb(
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Score a pitcher-batter matchup for walk (BB) tendency.

    Uses the same log-odds additive framework as ``score_matchup`` but
    with chase rate instead of whiff rate.  The lift is **inverted**:
    a hitter who chases *less* than league average will draw *more*
    walks, producing a positive ``matchup_bb_logit_lift``.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    batter_id : int
        Batter MLB ID.
    pitcher_arsenal : pd.DataFrame
        Pitcher arsenal profiles (pitcher_id, pitch_type, pitches,
        usage_pct, chase_rate or derivable fields).
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles (batter_id, pitch_type,
        out_of_zone_pitches, chase_swings, chase_rate, pitch_family).
    baselines_pt : dict[str, dict[str, float]]
        League-average baselines keyed by pitch_type with at least
        "chase_rate" key.

    Returns
    -------
    dict
        Keys: pitcher_id, batter_id, matchup_chase_rate,
        baseline_chase_rate, matchup_bb_logit_lift, n_pitch_types,
        avg_reliability.
    """
    p_arsenal = pitcher_arsenal[pitcher_arsenal["pitcher_id"] == pitcher_id].copy()
    p_arsenal = p_arsenal[p_arsenal["pitches"] >= _MIN_PITCHES].copy()

    if len(p_arsenal) == 0:
        return {
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "matchup_chase_rate": np.nan,
            "baseline_chase_rate": np.nan,
            "matchup_bb_logit_lift": 0.0,
            "n_pitch_types": 0,
            "avg_reliability": 0.0,
        }

    # Renormalize usage
    total_usage = p_arsenal["usage_pct"].sum()
    if total_usage > 0:
        p_arsenal["usage_norm"] = p_arsenal["usage_pct"] / total_usage
    else:
        p_arsenal["usage_norm"] = 1.0 / len(p_arsenal)

    matchup_chase = 0.0
    baseline_chase = 0.0
    reliabilities = []

    for _, row in p_arsenal.iterrows():
        pt = row["pitch_type"]
        usage = row["usage_norm"]

        # Pitcher's chase-inducing rate (if available)
        pitcher_chase = row.get("chase_rate", np.nan)

        # League baseline for this pitch type
        league_chase = baselines_pt.get(pt, {}).get(
            "chase_rate", LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get("chase_rate", 0.30)
        )

        if pd.isna(pitcher_chase):
            pitcher_chase = league_chase

        # Hitter chase with fallback
        hitter_chase, reliability = _get_hitter_chase_with_fallback(
            hitter_vuln, batter_id, pt, league_chase
        )
        reliabilities.append(reliability)

        # Log-odds additive method (same math as whiff)
        league_logit = _logit(league_chase)
        pitcher_delta = _logit(pitcher_chase) - league_logit
        hitter_delta = _logit(hitter_chase) - league_logit
        matchup_logit = league_logit + pitcher_delta + hitter_delta
        matchup_chase_pt = float(_inv_logit(matchup_logit))

        matchup_chase += usage * matchup_chase_pt
        baseline_chase += usage * pitcher_chase

    # Lift: INVERTED — lower chase rate → more walks
    # Negative chase lift means hitter chases less → positive BB lift
    chase_logit_lift = float(_logit(matchup_chase) - _logit(baseline_chase))
    matchup_bb_logit_lift = -chase_logit_lift

    return {
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "matchup_chase_rate": matchup_chase,
        "baseline_chase_rate": baseline_chase,
        "matchup_bb_logit_lift": matchup_bb_logit_lift,
        "n_pitch_types": len(p_arsenal),
        "avg_reliability": float(np.mean(reliabilities)) if reliabilities else 0.0,
    }


# ---------------------------------------------------------------------------
# Hitter barrel-rate fallback chain (for HR matchup)
# ---------------------------------------------------------------------------
def _get_hitter_barrel_with_fallback(
    hitter_vuln: pd.DataFrame,
    batter_id: int,
    pitch_type: str,
    league_barrel: float,
) -> tuple[float, float]:
    """Look up a hitter's barrel rate for a pitch type with fallback.

    Barrel rate is computed as barrels_proxy / bip.

    Fallback chain
    --------------
    1. Direct match on (batter_id, pitch_type) — reliability =
       min(bip, 50) / 50, blended toward league baseline.
    2. Pitch-family fallback — weighted average of same-family pitch types,
       reliability scaled by 0.5.
    3. League baseline — hitter_delta = 0 (average barrel tendency).

    Parameters
    ----------
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles with columns: batter_id, pitch_type,
        bip, barrels_proxy, pitch_family.
    batter_id : int
        Batter MLB ID.
    pitch_type : str
        Pitch type abbreviation (e.g. "FF", "SL").
    league_barrel : float
        League-average barrel rate for this pitch type.

    Returns
    -------
    tuple[float, float]
        (barrel_rate, reliability) where reliability is in [0, 1].
    """
    batter_rows = hitter_vuln[hitter_vuln["batter_id"] == batter_id]

    # Level 1: direct match
    direct = batter_rows[batter_rows["pitch_type"] == pitch_type]
    if len(direct) > 0:
        row = direct.iloc[0]
        bip = row.get("bip", 0)
        if pd.notna(bip) and bip > 0:
            barrels = row.get("barrels_proxy", 0)
            if pd.notna(barrels):
                raw_barrel = float(barrels) / float(bip)
                reliability = min(float(bip), _RELIABILITY_N) / _RELIABILITY_N
                blended = reliability * raw_barrel + (1.0 - reliability) * league_barrel
                return float(blended), reliability

    # Level 2: family fallback
    target_family = PITCH_TO_FAMILY.get(pitch_type)
    if target_family is not None and len(batter_rows) > 0:
        family_rows = batter_rows[
            batter_rows["pitch_family"] == target_family
        ]
        if len(family_rows) > 0:
            valid = family_rows.dropna(subset=["bip", "barrels_proxy"])
            valid = valid[valid["bip"] > 0]
            if len(valid) > 0:
                total_bip = valid["bip"].sum()
                total_barrels = valid["barrels_proxy"].sum()
                weighted_barrel = float(total_barrels) / float(total_bip)
                raw_reliability = min(float(total_bip), _RELIABILITY_N) / _RELIABILITY_N
                reliability = raw_reliability * 0.5  # discount for indirect match
                blended = reliability * weighted_barrel + (1.0 - reliability) * league_barrel
                return float(blended), reliability

    # Level 3: league baseline
    return league_barrel, 0.0


# ---------------------------------------------------------------------------
# HR matchup scoring
# ---------------------------------------------------------------------------
def score_matchup_hr(
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Score a pitcher-batter matchup for HR tendency.

    Uses the same log-odds additive framework as ``score_matchup`` but
    with barrel rate instead of whiff rate.  High barrel rate = more HR.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    batter_id : int
        Batter MLB ID.
    pitcher_arsenal : pd.DataFrame
        Pitcher arsenal profiles (pitcher_id, pitch_type, pitches,
        usage_pct, bip, barrels_proxy).
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles (batter_id, pitch_type, bip,
        barrels_proxy, pitch_family).
    baselines_pt : dict[str, dict[str, float]]
        League-average baselines keyed by pitch_type with at least
        "barrel_rate" key.

    Returns
    -------
    dict
        Keys: pitcher_id, batter_id, matchup_barrel_rate,
        baseline_barrel_rate, matchup_hr_logit_lift, n_pitch_types,
        avg_reliability.
    """
    p_arsenal = pitcher_arsenal[pitcher_arsenal["pitcher_id"] == pitcher_id].copy()
    p_arsenal = p_arsenal[p_arsenal["pitches"] >= _MIN_PITCHES].copy()

    if len(p_arsenal) == 0:
        return {
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "matchup_barrel_rate": np.nan,
            "baseline_barrel_rate": np.nan,
            "matchup_hr_logit_lift": 0.0,
            "n_pitch_types": 0,
            "avg_reliability": 0.0,
        }

    # Renormalize usage
    total_usage = p_arsenal["usage_pct"].sum()
    if total_usage > 0:
        p_arsenal["usage_norm"] = p_arsenal["usage_pct"] / total_usage
    else:
        p_arsenal["usage_norm"] = 1.0 / len(p_arsenal)

    matchup_barrel = 0.0
    baseline_barrel = 0.0
    reliabilities = []

    for _, row in p_arsenal.iterrows():
        pt = row["pitch_type"]
        usage = row["usage_norm"]

        # Pitcher's barrel rate
        p_bip = row.get("bip", 0)
        p_barrels = row.get("barrels_proxy", 0)
        if pd.notna(p_bip) and p_bip > 0 and pd.notna(p_barrels):
            pitcher_barrel = float(p_barrels) / float(p_bip)
        else:
            pitcher_barrel = np.nan

        # League baseline for this pitch type
        league_barrel = baselines_pt.get(pt, {}).get(
            "barrel_rate", LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get("barrel_rate", 0.06)
        )

        if pd.isna(pitcher_barrel):
            pitcher_barrel = league_barrel

        # Hitter barrel with fallback
        hitter_barrel, reliability = _get_hitter_barrel_with_fallback(
            hitter_vuln, batter_id, pt, league_barrel
        )
        reliabilities.append(reliability)

        # Log-odds additive method
        league_logit = _logit(league_barrel)
        pitcher_delta = _logit(pitcher_barrel) - league_logit
        hitter_delta = _logit(hitter_barrel) - league_logit
        matchup_logit = league_logit + pitcher_delta + hitter_delta
        matchup_barrel_pt = float(_inv_logit(matchup_logit))

        matchup_barrel += usage * matchup_barrel_pt
        baseline_barrel += usage * pitcher_barrel

    # Lift: positive = more barrels = more HR
    matchup_hr_logit_lift = float(_logit(matchup_barrel) - _logit(baseline_barrel))

    return {
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "matchup_barrel_rate": matchup_barrel,
        "baseline_barrel_rate": baseline_barrel,
        "matchup_hr_logit_lift": matchup_hr_logit_lift,
        "n_pitch_types": len(p_arsenal),
        "avg_reliability": float(np.mean(reliabilities)) if reliabilities else 0.0,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def score_matchup_for_stat(
    stat_name: str,
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
    shrinkage: float = 0.0,
) -> dict[str, Any]:
    """Dispatch matchup scoring based on stat type.

    Parameters
    ----------
    stat_name : str
        One of 'k' (whiff-based), 'bb' (chase-based), 'hr' (barrel-based).
        Any other value returns lift = 0 (no matchup adjustment).
    pitcher_id : int
        Pitcher MLB ID.
    batter_id : int
        Batter MLB ID.
    pitcher_arsenal : pd.DataFrame
        Pitcher arsenal profiles.
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles.
    baselines_pt : dict[str, dict[str, float]]
        League-average baselines keyed by pitch_type.
    shrinkage : float
        Reliability exponent for lift shrinkage.  ``0.0`` = no shrinkage
        (backward compatible).  ``0.5`` = moderate: a matchup with
        reliability 0.25 has its lift halved.  The lift is multiplied by
        ``reliability ** shrinkage``.

    Returns
    -------
    dict
        Matchup result dict.  The lift key varies by stat:
        'matchup_k_logit_lift', 'matchup_bb_logit_lift',
        'matchup_hr_logit_lift', or a generic 'matchup_logit_lift' = 0.
    """
    stat = stat_name.lower().strip()

    _LIFT_KEYS = {"k": "matchup_k_logit_lift", "bb": "matchup_bb_logit_lift", "hr": "matchup_hr_logit_lift"}

    if stat == "k":
        result = score_matchup(
            pitcher_id, batter_id, pitcher_arsenal, hitter_vuln, baselines_pt
        )
    elif stat == "bb":
        result = score_matchup_bb(
            pitcher_id, batter_id, pitcher_arsenal, hitter_vuln, baselines_pt
        )
    elif stat == "hr":
        # Path B (2026-04-10): HR matchup lifts disabled at the source. Raw
        # per-pair profile math has reliability ~0.62 and a systematic logit
        # mean bias of -0.43 from low-denominator pitch types; dampening to
        # 0.20 left noise plus a pitcher-favoring drift. See
        # memory/layer2_bvp_diagnostic_2026_04_10.md. Reliability=1.0 so the
        # simulator's (1-reliability) noise term also collapses to zero.
        return {
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "matchup_hr_logit_lift": 0.0,
            "n_pitch_types": 0,
            "avg_reliability": 1.0,
        }
    else:
        # No matchup adjustment for other stats (hits, outs, etc.)
        return {
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "matchup_logit_lift": 0.0,
            "n_pitch_types": 0,
            "avg_reliability": 0.0,
        }

    # Apply reliability-weighted shrinkage
    if shrinkage > 0.0:
        lift_key = _LIFT_KEYS.get(stat)
        reliability = result.get("avg_reliability", 0.0)
        if lift_key and reliability < 1.0:
            scale = max(reliability, 0.01) ** shrinkage
            result[lift_key] = result.get(lift_key, 0.0) * scale

    return result


# ---------------------------------------------------------------------------
# Unified matchup advantage
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_matchup_config() -> dict[str, Any]:
    """Load matchup_advantage config block from model.yaml."""
    cfg_path = _PROJECT_ROOT / "config" / "model.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)
    return cfg.get("matchup_advantage", {})


_MATCHUP_CFG_DEFAULTS: dict[str, Any] = {
    "weights": {
        "k_lift": 1.0,
        "bb_lift": -0.4,
        "hr_lift": -0.5,
        "platoon_lift": 0.3,
        "primary_pitch_edge": 0.2,
        "damage_score": -0.15,
        "trajectory_edge": -0.1,
        "glicko_edge": 0.1,
    },
    "tiers": {"strong": 0.30, "moderate": 0.12},
    "platoon_reliability_pa": 200,
}


def _get_matchup_config() -> dict[str, Any]:
    """Merge YAML config with defaults."""
    cfg = {**_MATCHUP_CFG_DEFAULTS}
    try:
        loaded = _load_matchup_config()
        if "weights" in loaded:
            cfg["weights"] = {**cfg["weights"], **loaded["weights"]}
        if "tiers" in loaded:
            cfg["tiers"] = {**cfg["tiers"], **loaded["tiers"]}
        if "platoon_reliability_pa" in loaded:
            cfg["platoon_reliability_pa"] = loaded["platoon_reliability_pa"]
    except Exception:
        logger.warning("Could not load matchup_advantage config; using defaults")
    return cfg


def _compute_platoon_lift(
    pitcher_hand: str,
    batter_platoon_splits: dict,
    league_platoon: dict[str, dict[str, float]] | None,
    reliability_pa: int,
) -> float:
    """Compute platoon logit lift (positive = pitcher advantage).

    Parameters
    ----------
    pitcher_hand : str
        "L" or "R".
    batter_platoon_splits : dict
        ``{"L": {"k_rate": float, "bb_rate": float, "pa": int},
           "R": {...}, "overall_k_rate": float, "overall_bb_rate": float}``.
    league_platoon : dict | None
        Precomputed league platoon baselines with ``platoon_k_logit`` and
        ``platoon_bb_logit`` sub-dicts.
    reliability_pa : int
        PA threshold for full individual weight.

    Returns
    -------
    float
        Platoon lift on logit scale (positive = pitcher advantage).
    """
    hand_data = batter_platoon_splits.get(pitcher_hand)
    if hand_data is None:
        return 0.0

    overall_k = batter_platoon_splits.get("overall_k_rate", 0.224)
    overall_bb = batter_platoon_splits.get("overall_bb_rate", 0.083)
    k_vs_hand = hand_data.get("k_rate", overall_k)
    bb_vs_hand = hand_data.get("bb_rate", overall_bb)
    pa_vs_hand = hand_data.get("pa", 0)

    reliability = min(pa_vs_hand, reliability_pa) / max(reliability_pa, 1)

    # Individual platoon delta (positive K delta = pitcher advantage)
    indiv_k = float(_logit(k_vs_hand) - _logit(overall_k))
    indiv_bb = float(_logit(bb_vs_hand) - _logit(overall_bb))

    # Determine same/opposite side for league baseline
    # We need batter_stand — infer from which hands have data
    # If batter has data vs both hands, check which side of pitcher_hand
    other_hand = "L" if pitcher_hand == "R" else "R"
    other_data = batter_platoon_splits.get(other_hand)

    # Heuristic: if batter K% is higher vs this hand → likely same-side
    # But more reliable: compare to overall
    if k_vs_hand > overall_k:
        side = "same"
    else:
        side = "opposite"

    # Fallback: use league baseline
    league_k = 0.0
    league_bb = 0.0
    if league_platoon:
        league_k = league_platoon.get("platoon_k_logit", {}).get(side, 0.0)
        league_bb = league_platoon.get("platoon_bb_logit", {}).get(side, 0.0)

    # Shrink toward league
    platoon_k = reliability * indiv_k + (1 - reliability) * league_k
    platoon_bb = reliability * indiv_bb + (1 - reliability) * league_bb

    # Combine: K component (positive = pitcher advantage) minus BB component
    # BB: positive indiv_bb means more walks vs this hand → hitter advantage
    # So subtract bb contribution (scaled down since BB less impactful than K)
    return platoon_k - platoon_bb * 0.5


def _compute_primary_pitch_edge(
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> float:
    """Compute edge from top-2 pitch reweighting vs full arsenal.

    Positive = hitter struggles more with pitcher's primary weapons than
    the usage-weighted blend suggests (pitcher advantage).

    Parameters
    ----------
    pitcher_id, batter_id : int
        Player IDs.
    pitcher_arsenal : pd.DataFrame
        Full arsenal data.
    hitter_vuln : pd.DataFrame
        Hitter vulnerability data.
    baselines_pt : dict
        League baselines by pitch type.

    Returns
    -------
    float
        Delta on logit scale (positive = pitcher advantage from primary pitches).
    """
    p_ars = pitcher_arsenal[pitcher_arsenal["pitcher_id"] == pitcher_id].copy()
    p_ars = p_ars[p_ars["pitches"] >= _MIN_PITCHES]
    if len(p_ars) < 2:
        return 0.0

    p_ars = p_ars.sort_values("usage_pct", ascending=False)
    top2 = p_ars.head(2)

    # Compute hitter's whiff delta vs league for top-2 vs all pitches
    def _hitter_delta_for_pitches(rows: pd.DataFrame) -> float:
        total_usage = rows["usage_pct"].sum()
        if total_usage <= 0:
            return 0.0
        weighted_hitter_logit = 0.0
        weighted_league_logit = 0.0
        for _, row in rows.iterrows():
            pt = row["pitch_type"]
            usage = row["usage_pct"] / total_usage
            league_whiff = baselines_pt.get(pt, {}).get(
                "whiff_rate",
                LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get("whiff_rate", 0.25),
            )
            hitter_whiff, _ = _get_hitter_whiff_with_fallback(
                hitter_vuln, batter_id, pt, league_whiff,
            )
            weighted_hitter_logit += usage * float(_logit(hitter_whiff))
            weighted_league_logit += usage * float(_logit(league_whiff))
        return weighted_hitter_logit - weighted_league_logit

    top2_delta = _hitter_delta_for_pitches(top2)
    all_delta = _hitter_delta_for_pitches(p_ars)

    # Positive top2_delta means hitter whiffs more than league on these pitches
    # If top2_delta > all_delta, hitter is *more* vulnerable to the primary
    # pitches than the full arsenal suggests → pitcher advantage
    return top2_delta - all_delta


def _compute_damage_score(
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_str: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> float:
    """Compute contact quality advantage (positive = hitter does damage).

    Parameters
    ----------
    pitcher_id, batter_id : int
        Player IDs.
    pitcher_arsenal : pd.DataFrame
        Pitcher arsenal profiles.
    hitter_str : pd.DataFrame
        Hitter strength profiles with xwoba_contact, hard_hit_rate columns.
    baselines_pt : dict
        League baselines by pitch type.

    Returns
    -------
    float
        Damage score (positive = hitter advantage on contact quality).
    """
    p_ars = pitcher_arsenal[pitcher_arsenal["pitcher_id"] == pitcher_id].copy()
    p_ars = p_ars[p_ars["pitches"] >= _MIN_PITCHES]
    if len(p_ars) == 0:
        return 0.0

    h_str = hitter_str[hitter_str["batter_id"] == batter_id]
    if h_str.empty:
        return 0.0

    total_usage = p_ars["usage_pct"].sum()
    if total_usage <= 0:
        return 0.0

    damage = 0.0
    for _, row in p_ars.iterrows():
        pt = row["pitch_type"]
        usage_w = row["usage_pct"] / total_usage
        lg_xwoba = baselines_pt.get(pt, {}).get(
            "xwoba_contact",
            LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get("xwoba_contact", 0.320),
        )
        lg_hh = baselines_pt.get(pt, {}).get(
            "hard_hit_rate",
            LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get("hard_hit_rate", 0.33),
        )
        s_row = h_str[h_str["pitch_type"] == pt]
        if s_row.empty:
            continue
        h_xwoba = s_row["xwoba_contact"].iloc[0] if "xwoba_contact" in s_row.columns else np.nan
        h_hh = s_row["hard_hit_rate"].iloc[0] if "hard_hit_rate" in s_row.columns else np.nan
        if pd.notna(h_xwoba):
            damage += usage_w * (h_xwoba - lg_xwoba)
        if pd.notna(h_hh):
            damage += usage_w * (h_hh - lg_hh) * 0.5

    return damage


def _find_dominant_reason(
    breakdown: dict[str, float],
    advantage: str,
    weights: dict[str, float],
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    hitter_str: pd.DataFrame | None,
    baselines_pt: dict[str, dict[str, float]],
) -> str:
    """Identify the dominant component and return a human-readable reason.

    Parameters
    ----------
    breakdown : dict
        Component values from score_matchup_advantage.
    advantage : str
        "pitcher", "hitter", or "neutral".
    weights : dict
        Composite weights from config.
    pitcher_id, batter_id : int
        Player IDs for pitch-specific detail.
    pitcher_arsenal, hitter_vuln, hitter_str : DataFrames
        Data for pitch-specific reason strings.
    baselines_pt : dict
        League baselines.

    Returns
    -------
    str
        Human-readable reason string.
    """
    if advantage == "neutral":
        return "no strong edges"

    # Find component with largest weighted magnitude
    weighted_contributions = {
        k: abs(weights.get(k, 0) * v)
        for k, v in breakdown.items()
    }
    dominant = max(weighted_contributions, key=weighted_contributions.get)

    # Build pitch-specific detail for pitch-type signals
    p_ars = pitcher_arsenal[pitcher_arsenal["pitcher_id"] == pitcher_id].copy()
    p_ars = p_ars[p_ars["pitches"] >= _MIN_PITCHES]
    if not p_ars.empty:
        p_ars = p_ars.sort_values("usage_pct", ascending=False)

    h_vuln = hitter_vuln[hitter_vuln["batter_id"] == batter_id]

    pt_map = {
        "FF": "fastball", "SI": "sinker", "SL": "slider", "CH": "changeup",
        "CU": "curve", "FC": "cutter", "ST": "sweeper", "KC": "knuckle curve",
        "FS": "splitter", "SV": "slurve",
    }

    if dominant == "k_lift" and not p_ars.empty:
        # Find best/worst pitch by whiff matchup
        best_pt, best_whiff = None, -1.0
        for _, row in p_ars.iterrows():
            pt = row["pitch_type"]
            if row["usage_pct"] / p_ars["usage_pct"].sum() < 0.10:
                continue
            league_whiff = baselines_pt.get(pt, {}).get(
                "whiff_rate",
                LEAGUE_AVG_BY_PITCH_TYPE.get(pt, {}).get("whiff_rate", 0.25),
            )
            h_whiff, _ = _get_hitter_whiff_with_fallback(
                hitter_vuln, batter_id, pt, league_whiff
            )
            combined = 0.6 * row.get("whiff_rate", league_whiff) + 0.4 * h_whiff
            if combined > best_whiff:
                best_whiff = combined
                best_pt = pt

        if best_pt:
            pt_name = pt_map.get(best_pt, best_pt)
            if advantage == "pitcher":
                return f"whiffs on {pt_name} ({best_whiff:.0%})"
            else:
                return f"handles {pt_name} ({best_whiff:.0%} whiff)"

    if dominant == "bb_lift" and not p_ars.empty:
        # Find pitch with highest/lowest chase
        for _, row in p_ars.iterrows():
            pt = row["pitch_type"]
            if row["usage_pct"] / p_ars["usage_pct"].sum() < 0.10:
                continue
            h_rows = h_vuln[h_vuln["pitch_type"] == pt]
            if h_rows.empty:
                continue
            chase = h_rows.iloc[0].get("chase_rate", np.nan)
            if pd.notna(chase):
                pt_name = pt_map.get(pt, pt)
                if advantage == "pitcher":
                    return f"chases {pt_name} ({chase:.0%})"
                else:
                    return f"lays off {pt_name} ({chase:.0%} chase)"

    if dominant == "hr_lift":
        if advantage == "hitter" and hitter_str is not None and not hitter_str.empty:
            h_str = hitter_str[hitter_str["batter_id"] == batter_id]
            if not h_str.empty and not p_ars.empty:
                for _, row in p_ars.iterrows():
                    pt = row["pitch_type"]
                    s_row = h_str[h_str["pitch_type"] == pt]
                    if not s_row.empty and "xwoba_contact" in s_row.columns:
                        xwoba = s_row["xwoba_contact"].iloc[0]
                        if pd.notna(xwoba) and xwoba > 0.350:
                            pt_name = pt_map.get(pt, pt)
                            return f"barrels {pt_name} (.{int(xwoba * 1000):03d})"
        return "weak contact expected" if advantage == "pitcher" else "hard contact"

    if dominant == "platoon_lift":
        return "platoon advantage" if advantage == "pitcher" else "platoon mismatch"

    if dominant == "primary_pitch_edge":
        if not p_ars.empty:
            top_pt = p_ars.iloc[0]["pitch_type"]
            pt_name = pt_map.get(top_pt, top_pt)
            if advantage == "pitcher":
                return f"vulnerable to {pt_name}"
            else:
                return f"handles {pt_name}"
        return "primary pitch edge"

    if dominant == "damage_score":
        return "hard contact on arsenal" if advantage == "hitter" else "weak contact"

    if dominant == "trajectory_edge":
        return "trajectory mismatch" if advantage == "hitter" else "trajectory favors pitcher"

    if dominant == "glicko_edge":
        return "skill gap" if advantage == "pitcher" else "batter outclasses"

    return "mixed signals"


def score_matchup_advantage(
    pitcher_id: int,
    batter_id: int,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
    *,
    hitter_str: pd.DataFrame | None = None,
    pitcher_hand: str | None = None,
    batter_platoon_splits: dict | None = None,
    pitcher_gb_pct: float | None = None,
    batter_gb_rate: float | None = None,
    batter_fb_rate: float | None = None,
    pitcher_glicko_mu: float | None = None,
    batter_glicko_mu: float | None = None,
    league_platoon: dict[str, dict[str, float]] | None = None,
    matchup_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Unified matchup advantage score (positive edge = pitcher advantage).

    Computes an 8-component composite on logit scale using config-driven
    weights and data-derived baselines.  All optional signals degrade
    gracefully to 0 when data is unavailable.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    batter_id : int
        Batter MLB ID.
    pitcher_arsenal : pd.DataFrame
        Pitcher arsenal profiles.
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles.
    baselines_pt : dict[str, dict[str, float]]
        League baselines keyed by pitch_type.
    hitter_str : pd.DataFrame | None
        Hitter strength profiles (xwoba_contact, hard_hit_rate per pitch type).
    pitcher_hand : str | None
        "L" or "R".
    batter_platoon_splits : dict | None
        ``{"L": {"k_rate", "bb_rate", "pa"}, "R": {...},
           "overall_k_rate", "overall_bb_rate"}``.
    pitcher_gb_pct : float | None
        Pitcher ground-ball percentage.
    batter_gb_rate : float | None
        Batter ground-ball rate.
    batter_fb_rate : float | None
        Batter fly-ball rate.
    pitcher_glicko_mu : float | None
        Pitcher Glicko-2 mu rating.
    batter_glicko_mu : float | None
        Batter Glicko-2 mu rating.
    league_platoon : dict | None
        Precomputed league platoon baselines from
        ``get_league_platoon_baselines()``.
    matchup_scales : dict | None
        ``{"trajectory_scale": float, "glicko_scale": float}``
        derived from signal distributions during precompute.

    Returns
    -------
    dict[str, Any]
        Keys: pitcher_id, batter_id, advantage, edge_score, confidence,
        reason, breakdown, k_result, bb_result, hr_result.
    """
    cfg = _get_matchup_config()
    weights = cfg["weights"]
    tiers = cfg["tiers"]
    platoon_rel_pa = cfg["platoon_reliability_pa"]

    # Default scaling factors if not precomputed
    t_scale = 5.0
    g_scale = 1.0 / 1000.0
    if matchup_scales:
        t_scale = matchup_scales.get("trajectory_scale", t_scale)
        g_scale = matchup_scales.get("glicko_scale", g_scale)

    # ------------------------------------------------------------------
    # 1. K / BB / HR lifts (existing functions)
    # ------------------------------------------------------------------
    k_result = score_matchup(
        pitcher_id, batter_id, pitcher_arsenal, hitter_vuln, baselines_pt,
    )
    bb_result = score_matchup_bb(
        pitcher_id, batter_id, pitcher_arsenal, hitter_vuln, baselines_pt,
    )
    hr_result = score_matchup_hr(
        pitcher_id, batter_id, pitcher_arsenal, hitter_vuln, baselines_pt,
    )

    k_lift = float(k_result.get("matchup_k_logit_lift", 0.0) or 0.0)
    bb_lift = float(bb_result.get("matchup_bb_logit_lift", 0.0) or 0.0)
    hr_lift = float(hr_result.get("matchup_hr_logit_lift", 0.0) or 0.0)
    avg_reliability = float(k_result.get("avg_reliability", 0.0))

    # ------------------------------------------------------------------
    # 2. Platoon lift
    # ------------------------------------------------------------------
    platoon_lift = 0.0
    if pitcher_hand and batter_platoon_splits:
        platoon_lift = _compute_platoon_lift(
            pitcher_hand, batter_platoon_splits, league_platoon, platoon_rel_pa,
        )

    # ------------------------------------------------------------------
    # 3. Primary pitch edge
    # ------------------------------------------------------------------
    primary_pitch_edge = _compute_primary_pitch_edge(
        pitcher_id, batter_id, pitcher_arsenal, hitter_vuln, baselines_pt,
    )

    # ------------------------------------------------------------------
    # 4. Damage score (contact quality)
    # ------------------------------------------------------------------
    damage_score = 0.0
    if hitter_str is not None and not hitter_str.empty:
        damage_score = _compute_damage_score(
            pitcher_id, batter_id, pitcher_arsenal, hitter_str, baselines_pt,
        )

    # ------------------------------------------------------------------
    # 5. Trajectory edge (GB/FB mismatch)
    # ------------------------------------------------------------------
    trajectory_edge = 0.0
    if pitcher_gb_pct is not None and batter_fb_rate is not None:
        lg_gb = 0.446
        lg_fb = 0.321
        if league_platoon:
            lg_gb = league_platoon.get("lg_gb_rate", lg_gb)
            lg_fb = league_platoon.get("lg_fb_rate", lg_fb)
        trajectory_edge = (
            (pitcher_gb_pct - lg_gb) * (batter_fb_rate - lg_fb) * t_scale
        )

    # ------------------------------------------------------------------
    # 6. Glicko edge
    # ------------------------------------------------------------------
    glicko_edge = 0.0
    if pitcher_glicko_mu is not None and batter_glicko_mu is not None:
        glicko_edge = (pitcher_glicko_mu - batter_glicko_mu) * g_scale

    # ------------------------------------------------------------------
    # 7. Composite
    # ------------------------------------------------------------------
    breakdown = {
        "k_lift": k_lift,
        "bb_lift": bb_lift,
        "hr_lift": hr_lift,
        "platoon_lift": platoon_lift,
        "primary_pitch_edge": primary_pitch_edge,
        "damage_score": damage_score,
        "trajectory_edge": trajectory_edge,
        "glicko_edge": glicko_edge,
    }
    edge_score = sum(weights.get(k, 0.0) * v for k, v in breakdown.items())

    # ------------------------------------------------------------------
    # 8. Tier / advantage / reason
    # ------------------------------------------------------------------
    strong_t = tiers.get("strong", 0.30)
    moderate_t = tiers.get("moderate", 0.12)

    if edge_score > moderate_t:
        advantage = "pitcher"
    elif edge_score < -moderate_t:
        advantage = "hitter"
    else:
        advantage = "neutral"

    # Confidence based on data completeness and reliability
    optional_signals = sum([
        pitcher_hand is not None and batter_platoon_splits is not None,
        hitter_str is not None and not (
            isinstance(hitter_str, pd.DataFrame) and hitter_str.empty
        ),
        pitcher_glicko_mu is not None and batter_glicko_mu is not None,
    ])
    if avg_reliability > 0.6 and optional_signals >= 2:
        confidence = "high"
    elif avg_reliability > 0.3 or optional_signals >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    reason = _find_dominant_reason(
        breakdown, advantage, weights,
        pitcher_id, batter_id, pitcher_arsenal, hitter_vuln, hitter_str,
        baselines_pt,
    )

    return {
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "advantage": advantage,
        "edge_score": edge_score,
        "confidence": confidence,
        "reason": reason,
        "breakdown": breakdown,
        "k_result": k_result,
        "bb_result": bb_result,
        "hr_result": hr_result,
    }


# ---------------------------------------------------------------------------
# Bullpen matchup lifts
# ---------------------------------------------------------------------------

def compute_bullpen_matchup_lifts(
    batter_id: int,
    reliever_roster: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
    *,
    _cache: dict[tuple[int, int, str], float] | None = None,
) -> dict[str, float]:
    """Compute BF-share-weighted bullpen matchup lifts for one batter.

    Parameters
    ----------
    batter_id : int
        Batter MLB ID.
    reliever_roster : pd.DataFrame
        Relievers for *one team-season*, with columns ``pitcher_id``
        and ``bf_share``.
    pitcher_arsenal, hitter_vuln, baselines_pt
        Standard matchup data (same as ``score_matchup_for_stat``).
    _cache : dict or None
        Optional ``(pitcher_id, batter_id, stat) -> lift`` cache to
        avoid redundant computation across batter-games.

    Returns
    -------
    dict
        Keys: ``bullpen_matchup_k_lift``, ``bullpen_matchup_bb_lift``,
        ``bullpen_matchup_hr_lift``.
    """
    totals = {"k": 0.0, "bb": 0.0, "hr": 0.0}
    weight_sum = 0.0

    for _, rrow in reliever_roster.iterrows():
        pid = int(rrow["pitcher_id"])
        share = float(rrow["bf_share"])

        for stat in ("k", "bb", "hr"):
            cache_key = (pid, batter_id, stat)
            if _cache is not None and cache_key in _cache:
                lift = _cache[cache_key]
            else:
                try:
                    res = score_matchup_for_stat(
                        stat, pid, batter_id,
                        pitcher_arsenal, hitter_vuln, baselines_pt,
                    )
                    lift_key = f"matchup_{stat}_logit_lift"
                    val = res.get(lift_key, 0.0)
                    lift = val if isinstance(val, float) and not np.isnan(val) else 0.0
                except Exception:
                    lift = 0.0
                if _cache is not None:
                    _cache[cache_key] = lift

            totals[stat] += lift * share

        weight_sum += share

    # Normalise in case bf_shares don't sum to 1 after filtering
    if weight_sum > 0 and abs(weight_sum - 1.0) > 0.01:
        for stat in totals:
            totals[stat] /= weight_sum

    return {
        "bullpen_matchup_k_lift": totals["k"],
        "bullpen_matchup_bb_lift": totals["bb"],
        "bullpen_matchup_hr_lift": totals["hr"],
    }
