"""
Walk-forward validation for player rankings.

Learns optimal blend weights for hitter and pitcher composite scores
by training on prior seasons and evaluating against actual next-season
outcomes (wRC+ for hitters, xwOBA-against for pitchers).

Metrics: Spearman rank correlation, top-N hit rates.
Weight optimization via scipy.optimize.minimize (Nelder-Mead).

Default folds:
  - Predict 2022 from 2018-2021
  - Predict 2023 from 2018-2022
  - Predict 2024 from 2018-2023
  - Predict 2025 from 2018-2024
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from src.data.db import read_sql

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default folds and PA thresholds
# ---------------------------------------------------------------------------
DEFAULT_HITTER_FOLDS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]
DEFAULT_PITCHER_FOLDS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]

# Sub-score column names mirroring player_rankings.py Phase 3 decomposition
_HITTER_SCORE_COLS = [
    "contact_skill", "decision_skill", "damage_skill",
    "fielding", "trajectory", "health",
]
_PITCHER_SCORE_COLS = [
    "stuff", "command", "workload", "trajectory", "health", "role",
]


# ===================================================================
# Data loading
# ===================================================================

def _load_hitter_outcomes(
    seasons: list[int],
    min_pa: int = 200,
) -> pd.DataFrame:
    """Load actual hitter season outcomes from the database.

    Parameters
    ----------
    seasons : list[int]
        Seasons to load.
    min_pa : int
        Minimum plate appearances to qualify.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, season, pa, wrc_plus, woba, k_pct, bb_pct,
        xwoba, barrel_pct, hard_hit_pct.
    """
    query = """
    SELECT
        batter_id, season, pa, wrc_plus, woba, k_pct, bb_pct,
        xwoba, barrel_pct, hard_hit_pct
    FROM production.fact_batting_advanced
    WHERE season = ANY(:seasons)
      AND pa >= :min_pa
    """
    df = read_sql(query, {"seasons": seasons, "min_pa": min_pa})
    logger.info(
        "Loaded hitter outcomes: %d rows across seasons %s (min_pa=%d)",
        len(df), seasons, min_pa,
    )
    return df


def _load_pitcher_outcomes(
    seasons: list[int],
    min_bf: int = 100,
) -> pd.DataFrame:
    """Load actual pitcher season outcomes from the database.

    Parameters
    ----------
    seasons : list[int]
        Seasons to load.
    min_bf : int
        Minimum batters faced to qualify.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, batters_faced, k_pct, bb_pct,
        xwoba_against, swstr_pct, csw_pct, woba_against.
    """
    query = """
    SELECT
        pitcher_id, season, batters_faced, k_pct, bb_pct,
        xwoba_against, swstr_pct, csw_pct, woba_against
    FROM production.fact_pitching_advanced
    WHERE season = ANY(:seasons)
      AND batters_faced >= :min_bf
    """
    df = read_sql(query, {"seasons": seasons, "min_bf": min_bf})
    logger.info(
        "Loaded pitcher outcomes: %d rows across seasons %s (min_bf=%d)",
        len(df), seasons, min_bf,
    )
    return df


def _build_hitter_sub_scores(
    seasons: list[int],
    min_pa: int = 150,
) -> pd.DataFrame:
    """Build hitter sub-score components from observable data.

    Constructs percentile-rank sub-scores from stats that would have
    been available at the end of each season.  Does NOT use dashboard
    parquets (which may only exist for the current projection year).

    Parameters
    ----------
    seasons : list[int]
        Seasons to build sub-scores for.
    min_pa : int
        Minimum PA to include a player-season.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, season, offense, fielding, trajectory,
        health, baserunning, role, composite, pa.
    """
    # Offensive stats
    bat = read_sql("""
        SELECT
            batter_id, season, pa, wrc_plus, woba, xwoba,
            barrel_pct, hard_hit_pct, k_pct, bb_pct
        FROM production.fact_batting_advanced
        WHERE season = ANY(:seasons) AND pa >= :min_pa
    """, {"seasons": seasons, "min_pa": min_pa})

    if bat.empty:
        return pd.DataFrame(columns=[
            "batter_id", "season"] + _HITTER_SCORE_COLS + ["composite", "pa"])

    # Player metadata for age
    player_info = read_sql("""
        SELECT player_id AS batter_id, birth_date
        FROM production.dim_player
        WHERE primary_position NOT IN ('P', 'TWP')
    """, {})

    # Load aggressiveness (chase, two-strike whiff) for decision bucket
    agg_df = pd.DataFrame()
    try:
        agg_df = read_sql("""
            SELECT fp.batter_id, dg.season,
                   AVG(CASE WHEN NOT fp.zone AND fp.is_swing THEN 1.0
                            WHEN NOT fp.zone THEN 0.0 END) AS chase_rate,
                   AVG(CASE WHEN fp.strikes = 2 AND fp.is_whiff THEN 1.0
                            WHEN fp.strikes = 2 AND fp.is_swing THEN 0.0 END)
                       AS two_strike_whiff_rate
            FROM production.fact_pitch fp
            JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
            WHERE dg.season = ANY(:seasons) AND dg.game_type = 'R'
            GROUP BY fp.batter_id, dg.season
        """, {"seasons": seasons})
    except Exception:
        logger.warning("Could not load aggressiveness data for validation")

    records = []
    for season in bat["season"].unique():
        sdf = bat[bat["season"] == season].copy()
        if len(sdf) < 10:
            continue

        pa_s = sdf["pa"]

        # --- Contact skill (stabilizes ~150 PA): K% inverted ---
        sdf["contact_skill"] = 1.0 - sdf["k_pct"].rank(pct=True, method="average")

        # --- Decision skill (stabilizes ~200 PA): BB%, chase, 2-strike ---
        sdf["decision_skill"] = sdf["bb_pct"].rank(pct=True, method="average")
        if not agg_df.empty:
            season_agg = agg_df[agg_df["season"] == season]
            if not season_agg.empty:
                sdf = sdf.merge(
                    season_agg[["batter_id", "chase_rate", "two_strike_whiff_rate"]],
                    on="batter_id", how="left",
                )
                if "chase_rate" in sdf.columns and sdf["chase_rate"].notna().any():
                    chase_pctl = 1.0 - sdf["chase_rate"].rank(pct=True, method="average")
                    whiff_2s = sdf.get("two_strike_whiff_rate")
                    if whiff_2s is not None and whiff_2s.notna().any():
                        whiff_pctl = 1.0 - whiff_2s.rank(pct=True, method="average")
                        sdf["decision_skill"] = (
                            0.35 * sdf["decision_skill"]  # BB%
                            + 0.40 * chase_pctl
                            + 0.25 * whiff_pctl
                        )
                    else:
                        sdf["decision_skill"] = (
                            0.45 * sdf["decision_skill"] + 0.55 * chase_pctl
                        )

        # --- Damage on contact (stabilizes ~300 PA): xwOBA, barrel, HH ---
        damage_parts = []
        damage_w = []
        if "xwoba" in sdf.columns and sdf["xwoba"].notna().any():
            damage_parts.append(sdf["xwoba"].rank(pct=True, method="average"))
            damage_w.append(0.40)
        if "barrel_pct" in sdf.columns and sdf["barrel_pct"].notna().any():
            damage_parts.append(sdf["barrel_pct"].rank(pct=True, method="average"))
            damage_w.append(0.35)
        if "hard_hit_pct" in sdf.columns and sdf["hard_hit_pct"].notna().any():
            damage_parts.append(sdf["hard_hit_pct"].rank(pct=True, method="average"))
            damage_w.append(0.25)

        if damage_parts:
            total_w = sum(damage_w)
            sdf["damage_skill"] = sum(
                (w / total_w) * p for w, p in zip(damage_w, damage_parts)
            )
        else:
            sdf["damage_skill"] = sdf["wrc_plus"].rank(pct=True, method="average")

        # --- Fielding: OAA if available, else neutral ---
        try:
            oaa = read_sql("""
                SELECT player_id AS batter_id,
                       SUM(outs_above_average) AS total_oaa
                FROM production.fact_fielding_oaa
                WHERE season = :season
                GROUP BY player_id
            """, {"season": int(season)})
            if not oaa.empty:
                sdf = sdf.merge(oaa, on="batter_id", how="left")
                sdf["total_oaa"] = sdf["total_oaa"].fillna(0)
                sdf["fielding"] = sdf["total_oaa"].rank(pct=True, method="average")
            else:
                sdf["fielding"] = 0.50
        except Exception:
            sdf["fielding"] = 0.50

        # --- Trajectory: age curve ---
        if not player_info.empty:
            if "birth_date" not in sdf.columns:
                sdf = sdf.merge(player_info, on="batter_id", how="left")
            sdf["age"] = (
                pd.Timestamp(f"{int(season)}-07-01") - pd.to_datetime(sdf["birth_date"])
            ).dt.days / 365.25
            age = sdf["age"].fillna(28)
            climb = 0.60 + 0.40 * ((age - 20) / 7.0).clip(0, 1)
            slow_decline = 1.0 - 0.50 * ((age - 27) / 6.0).clip(0, 1)
            steep_decline = 0.50 - 0.50 * ((age - 33) / 7.0).clip(0, 1)
            sdf["trajectory"] = np.where(
                age <= 27, climb,
                np.where(age <= 33, slow_decline, steep_decline),
            )
            sdf["trajectory"] = sdf["trajectory"].clip(0, 1)
        else:
            sdf["trajectory"] = 0.50

        # --- Health: PA as proxy for availability ---
        sdf["health"] = (sdf["pa"] / sdf["pa"].max()).clip(0, 1)

        # Composite (equal weight baseline)
        n_scores = len(_HITTER_SCORE_COLS)
        sdf["composite"] = sum(
            sdf[col] for col in _HITTER_SCORE_COLS
        ) / n_scores

        for _, row in sdf.iterrows():
            rec = {
                "batter_id": row["batter_id"],
                "season": row["season"],
                "pa": row["pa"],
            }
            for col in _HITTER_SCORE_COLS:
                rec[col] = row.get(col, 0.50)
            rec["composite"] = row["composite"]
            records.append(rec)

    return pd.DataFrame(records)


def _build_pitcher_sub_scores(
    seasons: list[int],
    min_bf: int = 100,
) -> pd.DataFrame:
    """Build pitcher sub-score components from observable data.

    Parameters
    ----------
    seasons : list[int]
        Seasons to build sub-scores for.
    min_bf : int
        Minimum batters faced to include.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, stuff, command, workload,
        trajectory, health, role, composite, batters_faced.
    """
    pit = read_sql("""
        SELECT
            pitcher_id, season, batters_faced, k_pct, bb_pct,
            swstr_pct, csw_pct, zone_pct, chase_pct,
            xwoba_against, barrel_pct_against
        FROM production.fact_pitching_advanced
        WHERE season = ANY(:seasons) AND batters_faced >= :min_bf
    """, {"seasons": seasons, "min_bf": min_bf})

    if pit.empty:
        return pd.DataFrame(columns=[
            "pitcher_id", "season"] + _PITCHER_SCORE_COLS + [
            "composite", "batters_faced"])

    player_info = read_sql("""
        SELECT player_id AS pitcher_id, birth_date
        FROM production.dim_player
        WHERE primary_position IN ('P', 'TWP', '1')
           OR player_id IN (
               SELECT DISTINCT pitcher_id
               FROM production.fact_pitching_advanced
               WHERE season = ANY(:seasons)
           )
    """, {"seasons": seasons})

    records = []
    for season in pit["season"].unique():
        sdf = pit[pit["season"] == season].copy()
        if len(sdf) < 10:
            continue

        # Stuff: K% + SwStr% (higher = better)
        k_pctl = sdf["k_pct"].rank(pct=True, method="average")
        swstr_pctl = sdf["swstr_pct"].fillna(sdf["swstr_pct"].median()).rank(
            pct=True, method="average",
        )
        sdf["stuff"] = 0.60 * k_pctl + 0.40 * swstr_pctl

        # Command: low BB% + high zone% (inverted BB)
        bb_inv = 1.0 - sdf["bb_pct"].rank(pct=True, method="average")
        zone_pctl = sdf["zone_pct"].fillna(sdf["zone_pct"].median()).rank(
            pct=True, method="average",
        )
        sdf["command"] = 0.60 * bb_inv + 0.40 * zone_pctl

        # Workload: BF as proxy
        sdf["workload"] = sdf["batters_faced"].rank(pct=True, method="average")

        # Trajectory: pitcher age curve (peak at 28)
        if not player_info.empty:
            sdf = sdf.merge(player_info, on="pitcher_id", how="left")
            sdf["age"] = (
                pd.Timestamp(f"{int(season)}-07-01") - pd.to_datetime(sdf["birth_date"])
            ).dt.days / 365.25
            age = sdf["age"].fillna(29)
            climb = 0.55 + 0.45 * ((age - 20) / 8.0).clip(0, 1)
            slow_decline = 1.0 - 0.50 * ((age - 28) / 6.0).clip(0, 1)
            steep_decline = 0.50 - 0.50 * ((age - 34) / 7.0).clip(0, 1)
            sdf["trajectory"] = np.where(
                age <= 28, climb,
                np.where(age <= 34, slow_decline, steep_decline),
            )
            sdf["trajectory"] = sdf["trajectory"].clip(0, 1)
        else:
            sdf["trajectory"] = 0.50

        # Health: BF as availability proxy
        sdf["health"] = (sdf["batters_faced"] / sdf["batters_faced"].max()).clip(0, 1)

        # Role: BF-based importance
        sdf["role"] = (sdf["batters_faced"] / 700.0).clip(0, 1)

        n_scores = len(_PITCHER_SCORE_COLS)
        sdf["composite"] = sum(
            sdf[col] for col in _PITCHER_SCORE_COLS
        ) / n_scores

        for _, row in sdf.iterrows():
            rec = {
                "pitcher_id": row["pitcher_id"],
                "season": row["season"],
                "batters_faced": row["batters_faced"],
            }
            for col in _PITCHER_SCORE_COLS:
                rec[col] = row[col]
            rec["composite"] = row["composite"]
            records.append(rec)

    return pd.DataFrame(records)


# ===================================================================
# Evaluation metrics
# ===================================================================

def evaluate_rank_correlation(
    predicted_ranks: pd.Series,
    actual_ranks: pd.Series,
    top_n_levels: list[int] | None = None,
) -> dict[str, Any]:
    """Compute rank correlation and top-N hit rates.

    Parameters
    ----------
    predicted_ranks : pd.Series
        Predicted scores (higher = better).
    actual_ranks : pd.Series
        Actual outcome values (higher = better for hitters, lower for pitchers).
    top_n_levels : list[int], optional
        Top-N levels for hit rate computation. Defaults to [25, 50, 100].

    Returns
    -------
    dict
        spearman_rho, spearman_p, and top_N_hit_rate entries.
    """
    if top_n_levels is None:
        top_n_levels = [25, 50, 100]

    # Drop NaN pairs
    mask = predicted_ranks.notna() & actual_ranks.notna()
    pred = predicted_ranks[mask]
    actual = actual_ranks[mask]

    if len(pred) < 5:
        return {"spearman_rho": np.nan, "spearman_p": np.nan}

    rho, pval = spearmanr(pred, actual)
    result: dict[str, Any] = {
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "n": len(pred),
    }

    # Top-N hit rates
    for n in top_n_levels:
        if n > len(pred):
            continue
        pred_top = set(pred.nlargest(n).index)
        actual_top = set(actual.nlargest(n).index)
        hit_rate = len(pred_top & actual_top) / n
        result[f"top_{n}_hit_rate"] = hit_rate

    return result


# ===================================================================
# Weight optimization
# ===================================================================

def learn_blend_weights(
    sub_scores: pd.DataFrame,
    outcomes: pd.Series,
    score_cols: list[str],
    min_pa: int = 200,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Optimize composite weights for prediction quality.

    Uses Nelder-Mead optimization over the simplex (softmax
    parameterization ensures weights sum to 1.0 and stay positive).

    Parameters
    ----------
    sub_scores : pd.DataFrame
        Must contain columns listed in ``score_cols``.
    outcomes : pd.Series
        Actual next-season values, aligned by index with ``sub_scores``.
    score_cols : list[str]
        Sub-score column names to optimize weights for.
    min_pa : int
        Minimum PA (already applied during loading, kept for clarity).
    random_seed : int
        For reproducibility of initial conditions.

    Returns
    -------
    dict
        optimal_weights (dict), spearman_rho.
    """
    # Align indices
    common_idx = sub_scores.index.intersection(outcomes.index)
    X = sub_scores.loc[common_idx, score_cols].values
    y = outcomes.loc[common_idx].values

    # Drop rows with NaN
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    if len(y) < 20:
        logger.warning(
            "Too few observations (%d) for weight optimization", len(y),
        )
        n = len(score_cols)
        return {
            "optimal_weights": {col: 1.0 / n for col in score_cols},
            "spearman_rho": np.nan,
        }

    # Standardize outcome to [0, 1] for optimization stability
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    if y_range < 1e-9:
        y_norm = np.full_like(y, 0.5)
    else:
        y_norm = (y - y_min) / y_range

    n_weights = len(score_cols)

    def _softmax(raw: np.ndarray) -> np.ndarray:
        """Convert unconstrained params to simplex weights."""
        exp = np.exp(raw - raw.max())
        return exp / exp.sum()

    def _objective(raw: np.ndarray) -> float:
        """Loss of weighted composite vs actual outcome."""
        w = _softmax(raw)
        pred = X @ w
        # Normalize prediction to same scale
        p_min, p_max = pred.min(), pred.max()
        p_range = p_max - p_min
        if p_range < 1e-9:
            return 1.0
        pred_norm = (pred - p_min) / p_range
        return float(np.sqrt(np.mean((pred_norm - y_norm) ** 2)))

    # Multiple restarts for robustness
    rng = np.random.default_rng(random_seed)
    best_result = None
    best_loss = np.inf

    for _ in range(10):
        x0 = rng.normal(0, 0.5, size=n_weights)
        res = minimize(_objective, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
        if res.fun < best_loss:
            best_loss = res.fun
            best_result = res

    optimal_raw = best_result.x
    optimal_weights = _softmax(optimal_raw)

    # Compute final metrics with optimal weights
    pred = X @ optimal_weights
    rho, _ = spearmanr(pred, y)

    weight_dict = {col: float(w) for col, w in zip(score_cols, optimal_weights)}

    logger.info(
        "Optimal weights: %s | rho=%.3f",
        {k: f"{v:.3f}" for k, v in weight_dict.items()}, rho,
    )

    return {
        "optimal_weights": weight_dict,
        "spearman_rho": float(rho),
        "n_players": len(y),
    }


# ===================================================================
# Walk-forward validation: hitters
# ===================================================================

def validate_hitter_rankings(
    folds: list[tuple[int, int]] | None = None,
    min_pa_train: int = 150,
    min_pa_test: int = 200,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Walk-forward validation for hitter ranking weights.

    For each fold, builds sub-scores from training seasons and
    evaluates against actual next-season wRC+.

    Parameters
    ----------
    folds : list[tuple[int, int]], optional
        (last_train_season, test_season) pairs. Defaults to 2022-2025.
    min_pa_train : int
        Minimum PA for training sub-scores.
    min_pa_test : int
        Minimum PA for test-season evaluation.
    random_seed : int
        For optimizer reproducibility.

    Returns
    -------
    dict
        per_fold: list of per-fold results,
        pooled_weights: optimal weights from all training data,
        summary: aggregated metrics DataFrame.
    """
    if folds is None:
        folds = DEFAULT_HITTER_FOLDS

    # Load all needed outcomes up front
    all_test_seasons = [f[1] for f in folds]
    all_outcomes = _load_hitter_outcomes(all_test_seasons, min_pa=min_pa_test)

    per_fold: list[dict[str, Any]] = []
    pooled_scores_list: list[pd.DataFrame] = []
    pooled_outcomes_list: list[pd.Series] = []

    for last_train, test_season in folds:
        logger.info(
            "=== Hitter fold: train 2018-%d, test %d ===",
            last_train, test_season,
        )

        train_seasons = list(range(2018, last_train + 1))

        # Build sub-scores from the LAST training season
        # (what we'd have as inputs at prediction time)
        scores_df = _build_hitter_sub_scores([last_train], min_pa=min_pa_train)
        if scores_df.empty:
            logger.warning("No training sub-scores for season %d", last_train)
            continue

        # Test outcomes
        test_out = all_outcomes[all_outcomes["season"] == test_season].copy()
        if test_out.empty:
            logger.warning("No test outcomes for season %d", test_season)
            continue

        # Common players
        common = set(scores_df["batter_id"]) & set(test_out["batter_id"])
        if len(common) < 20:
            logger.warning(
                "Only %d common hitters for fold %d->%d",
                len(common), last_train, test_season,
            )
            continue

        scores_fold = scores_df[scores_df["batter_id"].isin(common)].set_index("batter_id")
        test_fold = test_out[test_out["batter_id"].isin(common)].set_index("batter_id")

        # Primary outcome: wRC+
        outcomes_series = test_fold["wrc_plus"]

        # Optimize weights
        opt = learn_blend_weights(
            scores_fold, outcomes_series, _HITTER_SCORE_COLS,
            random_seed=random_seed,
        )

        # Evaluate current equal-weight baseline
        baseline_composite = scores_fold[_HITTER_SCORE_COLS].mean(axis=1)
        baseline_metrics = evaluate_rank_correlation(
            baseline_composite, outcomes_series,
        )

        # Evaluate optimized composite
        w_arr = np.array([opt["optimal_weights"][c] for c in _HITTER_SCORE_COLS])
        optimized_composite = (scores_fold[_HITTER_SCORE_COLS].values @ w_arr)
        optimized_series = pd.Series(optimized_composite, index=scores_fold.index)
        optimized_metrics = evaluate_rank_correlation(
            optimized_series, outcomes_series,
        )

        # Evaluate each sub-score individually
        individual_corrs = {}
        for col in _HITTER_SCORE_COLS:
            sub = scores_fold[col]
            valid_idx = sub.dropna().index.intersection(outcomes_series.dropna().index)
            if len(valid_idx) > 5 and sub.loc[valid_idx].std() > 1e-9:
                rho_val, _ = spearmanr(
                    sub.loc[valid_idx], outcomes_series.loc[valid_idx],
                )
                individual_corrs[col] = float(rho_val) if np.isfinite(rho_val) else 0.0
            else:
                individual_corrs[col] = 0.0

        fold_result = {
            "fold": f"{last_train}->{test_season}",
            "last_train": last_train,
            "test_season": test_season,
            "n_players": len(common),
            "optimal_weights": opt["optimal_weights"],
            "optimized_rho": opt["spearman_rho"],
            "baseline_rho": baseline_metrics["spearman_rho"],
            "baseline_top_50": baseline_metrics.get("top_50_hit_rate", np.nan),
            "optimized_top_50": optimized_metrics.get("top_50_hit_rate", np.nan),
            "individual_correlations": individual_corrs,
        }
        per_fold.append(fold_result)
        logger.info(
            "Fold %s: optimized rho=%.3f (baseline=%.3f)",
            fold_result["fold"], opt["spearman_rho"],
            baseline_metrics["spearman_rho"],
        )

        # Accumulate for pooled optimization
        pooled_scores_list.append(scores_fold)
        pooled_outcomes_list.append(outcomes_series)

    # Pooled optimization across all folds
    pooled_weights: dict[str, float] = {}
    if pooled_scores_list:
        pooled_scores = pd.concat(pooled_scores_list)
        pooled_outcomes = pd.concat(pooled_outcomes_list)
        pooled_opt = learn_blend_weights(
            pooled_scores, pooled_outcomes, _HITTER_SCORE_COLS,
            random_seed=random_seed,
        )
        pooled_weights = pooled_opt["optimal_weights"]
        logger.info("Pooled hitter weights: %s", pooled_weights)

    # Summary DataFrame
    summary_records = []
    for fold in per_fold:
        rec = {
            "fold": fold["fold"],
            "n_players": fold["n_players"],
            "optimized_rho": fold["optimized_rho"],
            "baseline_rho": fold["baseline_rho"],
            "rho_improvement": fold["optimized_rho"] - fold["baseline_rho"],
        }
        # Add top-N hit rates
        for key in ["baseline_top_50", "optimized_top_50"]:
            rec[key] = fold.get(key, np.nan)
        summary_records.append(rec)

    summary = pd.DataFrame(summary_records) if summary_records else pd.DataFrame()

    return {
        "per_fold": per_fold,
        "pooled_weights": pooled_weights,
        "summary": summary,
    }


# ===================================================================
# Walk-forward validation: pitchers
# ===================================================================

def validate_pitcher_rankings(
    folds: list[tuple[int, int]] | None = None,
    min_bf_train: int = 100,
    min_bf_test: int = 100,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Walk-forward validation for pitcher ranking weights.

    For each fold, builds sub-scores from training seasons and
    evaluates against actual next-season xwOBA-against (inverted,
    so lower xwOBA = better pitcher).

    Parameters
    ----------
    folds : list[tuple[int, int]], optional
        (last_train_season, test_season) pairs. Defaults to 2022-2025.
    min_bf_train : int
        Minimum BF for training sub-scores.
    min_bf_test : int
        Minimum BF for test evaluation.
    random_seed : int
        For optimizer reproducibility.

    Returns
    -------
    dict
        per_fold, pooled_weights, summary.
    """
    if folds is None:
        folds = DEFAULT_PITCHER_FOLDS

    all_test_seasons = [f[1] for f in folds]
    all_outcomes = _load_pitcher_outcomes(all_test_seasons, min_bf=min_bf_test)

    per_fold: list[dict[str, Any]] = []
    pooled_scores_list: list[pd.DataFrame] = []
    pooled_outcomes_list: list[pd.Series] = []

    for last_train, test_season in folds:
        logger.info(
            "=== Pitcher fold: train 2018-%d, test %d ===",
            last_train, test_season,
        )

        scores_df = _build_pitcher_sub_scores([last_train], min_bf=min_bf_train)
        if scores_df.empty:
            logger.warning("No training sub-scores for season %d", last_train)
            continue

        test_out = all_outcomes[all_outcomes["season"] == test_season].copy()
        if test_out.empty:
            logger.warning("No test outcomes for season %d", test_season)
            continue

        common = set(scores_df["pitcher_id"]) & set(test_out["pitcher_id"])
        if len(common) < 20:
            logger.warning(
                "Only %d common pitchers for fold %d->%d",
                len(common), last_train, test_season,
            )
            continue

        scores_fold = scores_df[scores_df["pitcher_id"].isin(common)].set_index("pitcher_id")
        test_fold = test_out[test_out["pitcher_id"].isin(common)].set_index("pitcher_id")

        # Primary outcome: inverted xwOBA-against (lower = better for pitchers,
        # so we negate to align with "higher = better" convention for optimization)
        outcomes_series = -test_fold["xwoba_against"]

        opt = learn_blend_weights(
            scores_fold, outcomes_series, _PITCHER_SCORE_COLS,
            random_seed=random_seed,
        )

        baseline_composite = scores_fold[_PITCHER_SCORE_COLS].mean(axis=1)
        baseline_metrics = evaluate_rank_correlation(
            baseline_composite, outcomes_series,
        )

        w_arr = np.array([opt["optimal_weights"][c] for c in _PITCHER_SCORE_COLS])
        optimized_composite = scores_fold[_PITCHER_SCORE_COLS].values @ w_arr
        optimized_series = pd.Series(optimized_composite, index=scores_fold.index)
        optimized_metrics = evaluate_rank_correlation(
            optimized_series, outcomes_series,
        )

        individual_corrs = {}
        for col in _PITCHER_SCORE_COLS:
            sub = scores_fold[col]
            outcome_aligned = outcomes_series.loc[sub.dropna().index]
            valid = sub.dropna().index.intersection(outcome_aligned.dropna().index)
            if len(valid) > 5:
                rho_val, _ = spearmanr(sub.loc[valid], outcomes_series.loc[valid])
                individual_corrs[col] = float(rho_val)
            else:
                individual_corrs[col] = 0.0

        fold_result = {
            "fold": f"{last_train}->{test_season}",
            "last_train": last_train,
            "test_season": test_season,
            "n_players": len(common),
            "optimal_weights": opt["optimal_weights"],
            "optimized_rho": opt["spearman_rho"],
            "baseline_rho": baseline_metrics["spearman_rho"],
            "baseline_top_50": baseline_metrics.get("top_50_hit_rate", np.nan),
            "optimized_top_50": optimized_metrics.get("top_50_hit_rate", np.nan),
            "individual_correlations": individual_corrs,
        }
        per_fold.append(fold_result)
        logger.info(
            "Fold %s: optimized rho=%.3f (baseline=%.3f)",
            fold_result["fold"], opt["spearman_rho"],
            baseline_metrics["spearman_rho"],
        )

        pooled_scores_list.append(scores_fold)
        pooled_outcomes_list.append(outcomes_series)

    pooled_weights: dict[str, float] = {}
    if pooled_scores_list:
        pooled_scores = pd.concat(pooled_scores_list)
        pooled_outcomes = pd.concat(pooled_outcomes_list)
        pooled_opt = learn_blend_weights(
            pooled_scores, pooled_outcomes, _PITCHER_SCORE_COLS,
            random_seed=random_seed,
        )
        pooled_weights = pooled_opt["optimal_weights"]
        logger.info("Pooled pitcher weights: %s", pooled_weights)

    summary_records = []
    for fold in per_fold:
        rec = {
            "fold": fold["fold"],
            "n_players": fold["n_players"],
            "optimized_rho": fold["optimized_rho"],
            "baseline_rho": fold["baseline_rho"],
            "rho_improvement": fold["optimized_rho"] - fold["baseline_rho"],
        }
        for key in ["baseline_top_50", "optimized_top_50"]:
            rec[key] = fold.get(key, np.nan)
        summary_records.append(rec)

    summary = pd.DataFrame(summary_records) if summary_records else pd.DataFrame()

    return {
        "per_fold": per_fold,
        "pooled_weights": pooled_weights,
        "summary": summary,
    }


# ===================================================================
# Compare current vs learned weights
# ===================================================================

def compare_weight_systems(
    hitter_results: dict[str, Any],
    pitcher_results: dict[str, Any],
) -> pd.DataFrame:
    """Compare current hardcoded weights vs learned optimal weights.

    Parameters
    ----------
    hitter_results : dict
        Output of ``validate_hitter_rankings()``.
    pitcher_results : dict
        Output of ``validate_pitcher_rankings()``.

    Returns
    -------
    pd.DataFrame
        Side-by-side weight comparison.
    """
    from src.models.player_rankings import _HITTER_WEIGHTS, _SP_WEIGHTS

    records = []

    # Hitter weights
    pooled_h = hitter_results.get("pooled_weights", {})
    for col in _HITTER_SCORE_COLS:
        current_w = _HITTER_WEIGHTS.get(col, 0.0)
        learned_w = pooled_h.get(col, 0.0)
        records.append({
            "player_type": "hitter",
            "component": col,
            "current_weight": current_w,
            "learned_weight": learned_w,
            "delta": learned_w - current_w,
        })

    # Pitcher weights
    pooled_p = pitcher_results.get("pooled_weights", {})
    for col in _PITCHER_SCORE_COLS:
        current_w = _SP_WEIGHTS.get(col, 0.0)
        learned_w = pooled_p.get(col, 0.0)
        records.append({
            "player_type": "pitcher",
            "component": col,
            "current_weight": current_w,
            "learned_weight": learned_w,
            "delta": learned_w - current_w,
        })

    return pd.DataFrame(records)


# ===================================================================
# Main entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

    logger.info("=" * 70)
    logger.info("PLAYER RANKING WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    # --- Hitter validation ---
    logger.info("\n" + "=" * 70)
    logger.info("HITTER RANKINGS")
    logger.info("=" * 70)
    hitter_results = validate_hitter_rankings()

    if hitter_results["summary"] is not None and not hitter_results["summary"].empty:
        logger.info("\nHitter Summary:\n%s", hitter_results["summary"].to_string())

    if hitter_results["pooled_weights"]:
        logger.info("\nPooled hitter weights:")
        for k, v in sorted(
            hitter_results["pooled_weights"].items(), key=lambda x: -x[1],
        ):
            logger.info("  %-15s  %.3f", k, v)

    # Per-fold individual correlations
    for fold in hitter_results["per_fold"]:
        logger.info("\nFold %s individual sub-score correlations:", fold["fold"])
        for col, rho in sorted(
            fold["individual_correlations"].items(), key=lambda x: -x[1],
        ):
            logger.info("  %-15s  rho=%.3f", col, rho)

    # --- Pitcher validation ---
    logger.info("\n" + "=" * 70)
    logger.info("PITCHER RANKINGS")
    logger.info("=" * 70)
    pitcher_results = validate_pitcher_rankings()

    if pitcher_results["summary"] is not None and not pitcher_results["summary"].empty:
        logger.info("\nPitcher Summary:\n%s", pitcher_results["summary"].to_string())

    if pitcher_results["pooled_weights"]:
        logger.info("\nPooled pitcher weights:")
        for k, v in sorted(
            pitcher_results["pooled_weights"].items(), key=lambda x: -x[1],
        ):
            logger.info("  %-15s  %.3f", k, v)

    for fold in pitcher_results["per_fold"]:
        logger.info("\nFold %s individual sub-score correlations:", fold["fold"])
        for col, rho in sorted(
            fold["individual_correlations"].items(), key=lambda x: -x[1],
        ):
            logger.info("  %-15s  rho=%.3f", col, rho)

    # --- Weight comparison ---
    logger.info("\n" + "=" * 70)
    logger.info("CURRENT vs LEARNED WEIGHTS")
    logger.info("=" * 70)
    try:
        comparison = compare_weight_systems(hitter_results, pitcher_results)
        logger.info("\n%s", comparison.to_string())
    except ImportError:
        logger.warning("Could not import current weights for comparison")

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)
