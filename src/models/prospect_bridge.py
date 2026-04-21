"""
Prospect-to-MLB projection bridge.

Seeds Bayesian priors for rookie MLB players from their translated MiLB
rates, so the conjugate updating pipeline can give them MiLB-informed
posteriors instead of falling back to league-average population priors.

The bridge:
1. Identifies players in 2026 MLB data who lack preseason projections.
2. Looks up their translated MiLB rates (recency + confidence weighted).
3. Converts translated rates to Beta(alpha, beta) priors scaled by
   confidence and sample size.
4. Returns DataFrames matching the preseason projection format for
   seamless integration with ``update_in_season.py``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.feature_eng import load_milb_translated
from src.data.paths import CACHE_DIR
from src.utils.constants import SIM_LEAGUE_K_RATE, SIM_LEAGUE_BB_RATE, SIM_LEAGUE_HR_RATE

logger = logging.getLogger(__name__)


# Recency decay half-life (years) — matches prospect_ranking / mlb_readiness
_RECENCY_HALF_LIFE = 2.0

# Level numeric mapping for weighting
_LEVEL_NUM = {"ROK": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4}

# Maximum effective PA/BF for the prior (cap to avoid overconfident priors)
_MAX_EFFECTIVE_TRIALS = 200

# Minimum effective PA/BF floor (even low-confidence gets some prior mass)
_MIN_EFFECTIVE_TRIALS = 20

# League-average rates (fallback when translated rates are unavailable)
_LEAGUE_AVG = {
    "k_rate": SIM_LEAGUE_K_RATE,
    "bb_rate": SIM_LEAGUE_BB_RATE,
    "hr_per_bf": SIM_LEAGUE_HR_RATE,
}


def _aggregate_batter_milb(
    milb_df: pd.DataFrame,
    player_ids: set[int],
) -> pd.DataFrame:
    """Aggregate multi-season MiLB batter data with recency + confidence weighting.

    Mirrors the aggregation logic in ``mlb_readiness.py`` and
    ``prospect_ranking.py`` for consistency.

    Parameters
    ----------
    milb_df : pd.DataFrame
        Raw translated MiLB batter data (one row per player-season-level).
    player_ids : set[int]
        Player IDs to aggregate for.

    Returns
    -------
    pd.DataFrame
        One row per player with: player_id, player_name, career_milb_pa,
        translated_k_pct, translated_bb_pct, translation_confidence,
        max_level, latest_season.
    """
    sub = milb_df[milb_df["player_id"].isin(player_ids)].copy()
    if sub.empty:
        return pd.DataFrame()

    records: list[dict] = []
    for pid, grp in sub.groupby("player_id"):
        grp = grp.sort_values(["season", "level"])
        total_pa = grp["pa"].sum()
        if total_pa < 1:
            continue

        max_season = grp["season"].max()
        conf = grp["translation_confidence"].fillna(0.3)

        # Recency decay: recent seasons matter more
        recency = np.exp(
            -np.log(2) * (max_season - grp["season"]) / _RECENCY_HALF_LIFE
        )
        weights = conf * grp["pa"] * recency
        w_sum = weights.sum()

        if w_sum > 0:
            wtd_k = (grp["translated_k_pct"] * weights).sum() / w_sum
            wtd_bb = (grp["translated_bb_pct"] * weights).sum() / w_sum
        else:
            wtd_k = (grp["translated_k_pct"] * grp["pa"]).sum() / total_pa
            wtd_bb = (grp["translated_bb_pct"] * grp["pa"]).sum() / total_pa

        # Average confidence (PA-weighted)
        avg_conf = (conf * grp["pa"]).sum() / total_pa

        # Best name and level
        best = grp.sort_values(["season"], ascending=False).iloc[0]
        grp["_lvl_num"] = grp["level"].map(_LEVEL_NUM).fillna(0)
        max_level = grp.sort_values("_lvl_num", ascending=False).iloc[0]["level"]

        records.append({
            "player_id": pid,
            "player_name": best["player_name"],
            "career_milb_pa": total_pa,
            "translated_k_pct": wtd_k,
            "translated_bb_pct": wtd_bb,
            "translation_confidence": avg_conf,
            "max_level": max_level,
            "latest_season": int(max_season),
        })

    return pd.DataFrame(records)


def _aggregate_pitcher_milb(
    milb_df: pd.DataFrame,
    player_ids: set[int],
) -> pd.DataFrame:
    """Aggregate multi-season MiLB pitcher data with recency + confidence weighting.

    Parameters
    ----------
    milb_df : pd.DataFrame
        Raw translated MiLB pitcher data (one row per player-season-level).
    player_ids : set[int]
        Player IDs to aggregate for.

    Returns
    -------
    pd.DataFrame
        One row per player with: player_id, player_name, career_milb_bf,
        translated_k_pct, translated_bb_pct, translation_confidence,
        max_level, latest_season.
    """
    sub = milb_df[milb_df["player_id"].isin(player_ids)].copy()
    if sub.empty:
        return pd.DataFrame()

    records: list[dict] = []
    for pid, grp in sub.groupby("player_id"):
        grp = grp.sort_values(["season", "level"])
        total_bf = grp["bf"].sum()
        if total_bf < 1:
            continue

        max_season = grp["season"].max()
        conf = grp["translation_confidence"].fillna(0.3)

        recency = np.exp(
            -np.log(2) * (max_season - grp["season"]) / _RECENCY_HALF_LIFE
        )
        weights = conf * grp["bf"] * recency
        w_sum = weights.sum()

        if w_sum > 0:
            wtd_k = (grp["translated_k_pct"] * weights).sum() / w_sum
            wtd_bb = (grp["translated_bb_pct"] * weights).sum() / w_sum
        else:
            wtd_k = (grp["translated_k_pct"] * grp["bf"]).sum() / total_bf
            wtd_bb = (grp["translated_bb_pct"] * grp["bf"]).sum() / total_bf

        avg_conf = (conf * grp["bf"]).sum() / total_bf

        best = grp.sort_values(["season"], ascending=False).iloc[0]
        grp["_lvl_num"] = grp["level"].map(_LEVEL_NUM).fillna(0)
        max_level = grp.sort_values("_lvl_num", ascending=False).iloc[0]["level"]

        records.append({
            "player_id": pid,
            "player_name": best["player_name"],
            "career_milb_bf": total_bf,
            "translated_k_pct": wtd_k,
            "translated_bb_pct": wtd_bb,
            "translation_confidence": avg_conf,
            "max_level": max_level,
            "latest_season": int(max_season),
        })

    return pd.DataFrame(records)


def _compute_effective_trials(
    career_trials: float,
    confidence: float,
) -> float:
    """Compute effective prior strength from MiLB sample + confidence.

    The prior strength scales with both the MiLB sample size and the
    translation confidence.  AAA players with 500+ PA and high confidence
    get a stronger prior (~150 PA equivalent); A-ball players with 200 PA
    and low confidence get a weaker prior (~50 PA equivalent).

    Parameters
    ----------
    career_trials : float
        Total career MiLB PA (batters) or BF (pitchers).
    confidence : float
        Translation confidence (0-1), combining level reliability,
        PA reliability, and factor reliability.

    Returns
    -------
    float
        Effective prior strength (clamped to [_MIN, _MAX]).
    """
    raw = career_trials * confidence * 0.5
    return float(np.clip(raw, _MIN_EFFECTIVE_TRIALS, _MAX_EFFECTIVE_TRIALS))


def _rate_to_beta(
    rate: float,
    effective_trials: float,
) -> tuple[float, float]:
    """Convert a translated rate to Beta(alpha, beta) parameters.

    Parameters
    ----------
    rate : float
        Translated rate (0-1 scale, e.g., K%).
    effective_trials : float
        Effective prior sample size.

    Returns
    -------
    tuple[float, float]
        (alpha, beta) parameters, both >= 0.5.
    """
    rate = float(np.clip(rate, 0.005, 0.995))
    alpha = rate * effective_trials
    beta = (1 - rate) * effective_trials
    return max(alpha, 0.5), max(beta, 0.5)


def _build_hitter_prior_row(
    row: pd.Series,
) -> dict:
    """Build a single hitter projection row from aggregated MiLB data.

    Produces columns matching the preseason hitter projections format
    so it can be concatenated directly.

    Parameters
    ----------
    row : pd.Series
        One row from ``_aggregate_batter_milb`` output.

    Returns
    -------
    dict
        Column values for the hitter projections DataFrame.
    """
    eff_pa = _compute_effective_trials(
        row["career_milb_pa"], row["translation_confidence"],
    )

    k_rate = row["translated_k_pct"]
    bb_rate = row["translated_bb_pct"]

    k_alpha, k_beta = _rate_to_beta(k_rate, eff_pa)
    bb_alpha, bb_beta = _rate_to_beta(bb_rate, eff_pa)

    # Generate sd and CI from the Beta distribution
    from scipy import stats as sp_stats

    k_dist = sp_stats.beta(k_alpha, k_beta)
    bb_dist = sp_stats.beta(bb_alpha, bb_beta)

    return {
        "batter_id": int(row["player_id"]),
        "batter_name": row["player_name"],
        "batter_stand": "R",  # unknown, placeholder
        "season": row.get("latest_season", 2025),
        "age": 0,  # unknown at this stage
        "age_bucket": 0,
        "pa": int(row["career_milb_pa"]),
        "skill_tier": 2,  # default mid-tier
        "observed_k_rate": k_rate,
        "career_k_rate": k_rate,
        "projected_k_rate": float(k_dist.mean()),
        "projected_k_rate_sd": float(k_dist.std()),
        "projected_k_rate_2_5": float(k_dist.ppf(0.025)),
        "projected_k_rate_97_5": float(k_dist.ppf(0.975)),
        "delta_k_rate": 0.0,
        "observed_bb_rate": bb_rate,
        "career_bb_rate": bb_rate,
        "projected_bb_rate": float(bb_dist.mean()),
        "projected_bb_rate_sd": float(bb_dist.std()),
        "projected_bb_rate_2_5": float(bb_dist.ppf(0.025)),
        "projected_bb_rate_97_5": float(bb_dist.ppf(0.975)),
        "delta_bb_rate": 0.0,
        # Bridge metadata
        "_bridge_source": "milb_translation",
        "_effective_pa": eff_pa,
        "_milb_max_level": row["max_level"],
        "_translation_confidence": row["translation_confidence"],
    }


def _build_pitcher_prior_row(
    row: pd.Series,
) -> dict:
    """Build a single pitcher projection row from aggregated MiLB data.

    Parameters
    ----------
    row : pd.Series
        One row from ``_aggregate_pitcher_milb`` output.

    Returns
    -------
    dict
        Column values for the pitcher projections DataFrame.
    """
    eff_bf = _compute_effective_trials(
        row["career_milb_bf"], row["translation_confidence"],
    )

    k_rate = row["translated_k_pct"]
    bb_rate = row["translated_bb_pct"]

    k_alpha, k_beta = _rate_to_beta(k_rate, eff_bf)
    bb_alpha, bb_beta = _rate_to_beta(bb_rate, eff_bf)

    from scipy import stats as sp_stats

    k_dist = sp_stats.beta(k_alpha, k_beta)
    bb_dist = sp_stats.beta(bb_alpha, bb_beta)

    return {
        "pitcher_id": int(row["player_id"]),
        "pitcher_name": row["player_name"],
        "pitch_hand": "R",  # unknown, placeholder
        "season": row.get("latest_season", 2025),
        "age": 0,
        "age_bucket": 0,
        "batters_faced": int(row["career_milb_bf"]),
        "is_starter": True,  # unknown, assume starter
        "skill_tier": 2,
        "observed_k_rate": k_rate,
        "career_k_rate": k_rate,
        "projected_k_rate": float(k_dist.mean()),
        "projected_k_rate_sd": float(k_dist.std()),
        "projected_k_rate_2_5": float(k_dist.ppf(0.025)),
        "projected_k_rate_97_5": float(k_dist.ppf(0.975)),
        "delta_k_rate": 0.0,
        "observed_bb_rate": bb_rate,
        "career_bb_rate": bb_rate,
        "projected_bb_rate": float(bb_dist.mean()),
        "projected_bb_rate_sd": float(bb_dist.std()),
        "projected_bb_rate_2_5": float(bb_dist.ppf(0.025)),
        "projected_bb_rate_97_5": float(bb_dist.ppf(0.975)),
        "delta_bb_rate": 0.0,
        # Bridge metadata
        "_bridge_source": "milb_translation",
        "_effective_bf": eff_bf,
        "_milb_max_level": row["max_level"],
        "_translation_confidence": row["translation_confidence"],
    }


def _get_rookie_batter_ids(season: int) -> set[int]:
    """Find batter IDs in observed data that lack preseason projections.

    Parameters
    ----------
    season : int
        Target season (e.g., 2026).

    Returns
    -------
    set[int]
        Batter IDs appearing in MLB data without preseason projections.
    """
    from src.data.db import read_sql

    # All batters with at least 1 PA in the target season
    observed = read_sql("""
        SELECT DISTINCT fp.batter_id
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
    """, {"season": season})

    if observed.empty:
        return set()

    observed_ids = set(observed["batter_id"].astype(int))

    # Load preseason snapshot to find who already has projections
    from src.data.paths import dashboard_dir as _dashboard_dir
    dashboard_dir = _dashboard_dir()
    snap_dir = dashboard_dir / "snapshots"

    snap_path = snap_dir / f"hitter_projections_{season}_preseason.parquet"
    if not snap_path.exists():
        snap_path = dashboard_dir / "hitter_projections.parquet"

    if snap_path.exists():
        preseason = pd.read_parquet(snap_path, columns=["batter_id"])
        preseason_ids = set(preseason["batter_id"].astype(int))
    else:
        preseason_ids = set()

    rookie_ids = observed_ids - preseason_ids
    return rookie_ids


def _get_rookie_pitcher_ids(season: int) -> set[int]:
    """Find pitcher IDs in observed data that lack preseason projections.

    Parameters
    ----------
    season : int
        Target season (e.g., 2026).

    Returns
    -------
    set[int]
        Pitcher IDs appearing in MLB data without preseason projections.
    """
    from src.data.db import read_sql

    observed = read_sql("""
        SELECT DISTINCT pb.pitcher_id
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
    """, {"season": season})

    if observed.empty:
        return set()

    observed_ids = set(observed["pitcher_id"].astype(int))

    from src.data.paths import dashboard_dir as _dashboard_dir
    dashboard_dir = _dashboard_dir()
    snap_dir = dashboard_dir / "snapshots"

    snap_path = snap_dir / f"pitcher_projections_{season}_preseason.parquet"
    if not snap_path.exists():
        snap_path = dashboard_dir / "pitcher_projections.parquet"

    if snap_path.exists():
        preseason = pd.read_parquet(snap_path, columns=["pitcher_id"])
        preseason_ids = set(preseason["pitcher_id"].astype(int))
    else:
        preseason_ids = set()

    rookie_ids = observed_ids - preseason_ids
    return rookie_ids


def build_rookie_priors(
    season: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build MiLB-informed priors for rookies lacking preseason projections.

    Identifies players who appear in MLB observed data but have no
    preseason projection snapshot, looks up their translated MiLB
    rates, and converts those rates to Beta distribution priors with
    strength scaled by confidence and sample size.

    Parameters
    ----------
    season : int
        Target MLB season.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (hitter_priors, pitcher_priors) — DataFrames matching the
        preseason projection format.  Empty DataFrame if no rookies
        found or no MiLB data available.
    """
    logger.info("Building rookie priors for season %d...", season)

    # --- Hitters ---
    h_rookie_ids = _get_rookie_batter_ids(season)
    logger.info("Found %d rookie batters without preseason projections", len(h_rookie_ids))

    hitter_priors = pd.DataFrame()
    if h_rookie_ids:
        milb_bat = load_milb_translated("batters")
        if not milb_bat.empty:
            agg = _aggregate_batter_milb(milb_bat, h_rookie_ids)

            if not agg.empty:
                rows = [_build_hitter_prior_row(r) for _, r in agg.iterrows()]
                hitter_priors = pd.DataFrame(rows)

                # Log details
                matched = set(agg["player_id"])
                unmatched = h_rookie_ids - matched
                logger.info(
                    "Built hitter bridge priors: %d matched MiLB data, "
                    "%d unmatched (no MiLB translations)",
                    len(matched), len(unmatched),
                )
                for _, r in agg.iterrows():
                    eff = _compute_effective_trials(
                        r["career_milb_pa"], r["translation_confidence"],
                    )
                    logger.info(
                        "  %s (ID %d): K%%=%.1f%%, BB%%=%.1f%%, "
                        "MiLB PA=%d, conf=%.2f, effective PA=%.0f, "
                        "level=%s",
                        r["player_name"], r["player_id"],
                        r["translated_k_pct"] * 100,
                        r["translated_bb_pct"] * 100,
                        r["career_milb_pa"],
                        r["translation_confidence"],
                        eff, r["max_level"],
                    )
            else:
                logger.info("No MiLB batter translations found for rookie batters")
        else:
            logger.warning(
                "MiLB batter translations not found at %s", milb_bat_path,
            )

    # --- Pitchers ---
    p_rookie_ids = _get_rookie_pitcher_ids(season)
    logger.info("Found %d rookie pitchers without preseason projections", len(p_rookie_ids))

    pitcher_priors = pd.DataFrame()
    if p_rookie_ids:
        milb_pit = load_milb_translated("pitchers")
        if not milb_pit.empty:
            agg = _aggregate_pitcher_milb(milb_pit, p_rookie_ids)

            if not agg.empty:
                rows = [_build_pitcher_prior_row(r) for _, r in agg.iterrows()]
                pitcher_priors = pd.DataFrame(rows)

                matched = set(agg["player_id"])
                unmatched = p_rookie_ids - matched
                logger.info(
                    "Built pitcher bridge priors: %d matched MiLB data, "
                    "%d unmatched (no MiLB translations)",
                    len(matched), len(unmatched),
                )
                for _, r in agg.iterrows():
                    eff = _compute_effective_trials(
                        r["career_milb_bf"], r["translation_confidence"],
                    )
                    logger.info(
                        "  %s (ID %d): K%%=%.1f%%, BB%%=%.1f%%, "
                        "MiLB BF=%d, conf=%.2f, effective BF=%.0f, "
                        "level=%s",
                        r["player_name"], r["player_id"],
                        r["translated_k_pct"] * 100,
                        r["translated_bb_pct"] * 100,
                        r["career_milb_bf"],
                        r["translation_confidence"],
                        eff, r["max_level"],
                    )
            else:
                logger.info("No MiLB pitcher translations found for rookie pitchers")
        else:
            logger.warning(
                "MiLB pitcher translations not found at %s", milb_pit_path,
            )

    logger.info(
        "Rookie bridge complete: %d hitter priors, %d pitcher priors",
        len(hitter_priors), len(pitcher_priors),
    )
    return hitter_priors, pitcher_priors


def merge_rookie_priors(
    preseason: pd.DataFrame,
    rookie_priors: pd.DataFrame,
    id_col: str,
) -> pd.DataFrame:
    """Merge rookie priors into preseason projections (fill gaps only).

    Rookies are appended to the preseason DataFrame. Existing players
    are never overwritten.

    Parameters
    ----------
    preseason : pd.DataFrame
        Frozen preseason projections.
    rookie_priors : pd.DataFrame
        Output of ``build_rookie_priors`` (hitter or pitcher half).
    id_col : str
        Player ID column ('batter_id' or 'pitcher_id').

    Returns
    -------
    pd.DataFrame
        Combined projections with rookies appended.
    """
    if rookie_priors.empty:
        return preseason

    if preseason.empty:
        return rookie_priors

    # Only keep rookies NOT already in preseason
    existing_ids = set(preseason[id_col].astype(int))
    new_rookies = rookie_priors[
        ~rookie_priors[id_col].astype(int).isin(existing_ids)
    ].copy()

    if new_rookies.empty:
        logger.info("No new rookies to add (all already in preseason)")
        return preseason

    # Align columns: add missing columns as NaN to rookies
    for col in preseason.columns:
        if col not in new_rookies.columns:
            new_rookies[col] = np.nan

    # Only keep columns that exist in preseason (drop bridge metadata)
    combined = pd.concat(
        [preseason, new_rookies[preseason.columns]],
        ignore_index=True,
    )

    logger.info(
        "Merged %d rookie priors into %d preseason projections (total: %d)",
        len(new_rookies), len(preseason), len(combined),
    )
    return combined
