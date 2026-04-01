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
        result = score_matchup_hr(
            pitcher_id, batter_id, pitcher_arsenal, hitter_vuln, baselines_pt
        )
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
