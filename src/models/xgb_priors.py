"""
XGBoost-based prior models for stats too noisy for Bayesian projection.

Provides informed priors for:
- ERA-FIP gap: replaces simple GB%-linear prior with multi-feature prediction
- Pitcher BABIP: replaces fixed league average (0.292) with player-specific prediction

These predictions serve as prior MEANS in the existing conjugate shrinkage
framework.  The shrinkage structure is preserved — XGBoost just provides
a better starting point than the league average or a single-covariate model.

Architecture:
    Current features (season N) → XGBoost → predicted prior (season N+1)
    ↓
    Conjugate shrinkage with observed data → final posterior

Training: walk-forward on year-pairs (2018→2019 through 2024→2025).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features used by each model (must exist in _build_pitcher_season_features output)
ERA_FIP_FEATURES = [
    "k_rate", "bb_rate", "gb_pct", "avg_velo", "whiff_rate",
    "zone_pct", "go_ao", "is_starter", "ip", "era_fip_gap",
]

BABIP_FEATURES = [
    "k_rate", "bb_rate", "gb_pct", "avg_velo", "whiff_rate",
    "zone_pct", "go_ao", "is_starter", "babip",
]


def _build_pitcher_season_features(season: int) -> pd.DataFrame:
    """Build feature matrix for pitchers in a single season.

    Merges pitcher observed profile, season totals, and traditional stats
    into a single row-per-pitcher DataFrame.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        One row per pitcher with all feature columns.
    """
    from src.data.feature_eng import (
        get_cached_pitcher_observed_profile,
        get_cached_pitcher_season_totals_with_age,
    )
    from src.data.queries import get_pitcher_traditional_stats

    # Season totals: k_rate, bb_rate, ip, bf, etc.
    totals = get_cached_pitcher_season_totals_with_age(season)

    # Observed profile: whiff_rate, avg_velo, zone_pct, gb_pct
    try:
        obs = get_cached_pitcher_observed_profile(season)
        totals = totals.merge(
            obs[["pitcher_id", "whiff_rate", "avg_velo", "zone_pct", "gb_pct"]],
            on="pitcher_id", how="left",
        )
    except Exception:
        logger.warning("No pitcher observed profile for %d", season)
        for col in ["whiff_rate", "avg_velo", "zone_pct", "gb_pct"]:
            if col not in totals.columns:
                totals[col] = np.nan

    # Traditional stats: era, fip, go_ao, hbp, hits
    try:
        trad = get_pitcher_traditional_stats(season)
        trad_cols = ["pitcher_id", "era", "fip", "go_ao", "hbp",
                     "hits_allowed", "hr_allowed"]
        trad_avail = [c for c in trad_cols if c in trad.columns]
        totals = totals.merge(trad[trad_avail], on="pitcher_id", how="left")
    except Exception:
        logger.warning("No pitcher traditional stats for %d", season)
        for col in ["era", "fip", "go_ao", "hbp", "hits_allowed", "hr_allowed"]:
            if col not in totals.columns:
                totals[col] = np.nan

    # Derived features
    totals["is_starter"] = (
        totals["ip"] / totals["games"].clip(1) >= 3.0
    ).astype(int)

    # ERA-FIP gap
    totals["era_fip_gap"] = totals["era"] - totals["fip"]

    # BABIP = (H - HR) / (BF - K - BB - HR - HBP)
    hbp = totals["hbp"].fillna(0)
    hr_col = totals.get("hr_allowed", totals.get("hr", 0))
    if isinstance(hr_col, (int, float)):
        hr_col = totals["hr"]
    h_col = totals.get("hits_allowed", pd.Series(0, index=totals.index))
    bip_denom = (
        totals["batters_faced"] - totals["k"] - totals["bb"] - hr_col - hbp
    ).clip(1)
    totals["babip"] = ((h_col - hr_col) / bip_denom).clip(0, 0.500)

    totals["season"] = season
    return totals


def _build_training_pairs(
    seasons: list[int],
    target_col: str,
    feature_cols: list[str],
    min_ip: float = 40.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (features, target) pairs with 1-year lag.

    For each consecutive year-pair (N, N+1), uses season N features
    to predict season N+1 target.  Only includes pitchers present
    in both seasons with sufficient IP.

    Parameters
    ----------
    seasons : list[int]
        All available seasons (e.g., [2018, ..., 2025]).
    target_col : str
        Column name for the target variable (e.g., "era_fip_gap", "babip").
    feature_cols : list[str]
        Feature column names.
    min_ip : float
        Minimum IP to include a pitcher-season.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X has feature columns and y has the target.
    """
    pairs: list[pd.DataFrame] = []

    for i in range(len(seasons) - 1):
        s_curr, s_next = seasons[i], seasons[i + 1]
        try:
            curr = _build_pitcher_season_features(s_curr)
            nxt = _build_pitcher_season_features(s_next)
        except Exception as e:
            logger.warning("Skipping %d→%d pair: %s", s_curr, s_next, e)
            continue

        # Filter by minimum IP
        curr = curr[curr["ip"] >= min_ip].copy()
        nxt = nxt[nxt["ip"] >= min_ip].copy()

        # Common pitchers
        common = set(curr["pitcher_id"]) & set(nxt["pitcher_id"])
        if not common:
            continue

        curr_sub = curr[curr["pitcher_id"].isin(common)].set_index("pitcher_id")
        nxt_sub = nxt[nxt["pitcher_id"].isin(common)].set_index("pitcher_id")

        avail_features = [c for c in feature_cols if c in curr_sub.columns]
        feature_df = curr_sub[avail_features].copy()

        if target_col not in nxt_sub.columns:
            logger.warning("Target %s not in next-season data", target_col)
            continue

        feature_df["_target"] = nxt_sub[target_col]
        feature_df["_pair"] = f"{s_curr}→{s_next}"
        pairs.append(feature_df)

    if not pairs:
        return pd.DataFrame(), pd.Series(dtype=float)

    combined = pd.concat(pairs)

    # Drop rows with NaN target
    valid = combined["_target"].notna()
    combined = combined[valid]

    X = combined[avail_features]
    y = combined["_target"]

    logger.info(
        "Training pairs for %s: %d rows across %d year-pairs",
        target_col, len(X), len(pairs),
    )
    return X, y


def train_pitcher_prior_model(
    seasons: list[int],
    target: str,
    min_ip: float = 40.0,
) -> dict[str, Any]:
    """Train XGBoost model for pitcher prior prediction.

    Parameters
    ----------
    seasons : list[int]
        All available seasons (training uses consecutive pairs).
    target : str
        "era_fip_gap" or "babip".
    min_ip : float
        Minimum IP to include a pitcher-season.

    Returns
    -------
    dict
        Bundle with 'model', 'feature_cols', 'n_train', 'target',
        'train_rmse'.
    """
    import xgboost as xgb

    feature_cols = ERA_FIP_FEATURES if target == "era_fip_gap" else BABIP_FEATURES
    X, y = _build_training_pairs(seasons, target, feature_cols, min_ip)

    if X.empty:
        logger.warning("No training data for %s model", target)
        return {"model": None, "feature_cols": feature_cols, "n_train": 0,
                "target": target, "train_rmse": float("nan")}

    # Conservative XGBoost: small trees, heavy regularization
    # ~3000 rows, 9-10 features — overfitting is the main risk
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.08,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        random_state=42,
    )
    model.fit(X, y)

    train_pred = model.predict(X)
    train_rmse = float(np.sqrt(np.mean((y.values - train_pred) ** 2)))

    # Feature importances
    importances = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
    logger.info(
        "Trained %s XGBoost: n=%d, train RMSE=%.4f, top features: %s",
        target, len(X), train_rmse,
        ", ".join(f"{f}={v:.3f}" for f, v in top_features),
    )

    return {
        "model": model,
        "feature_cols": list(X.columns),
        "n_train": len(X),
        "target": target,
        "train_rmse": train_rmse,
        "feature_importances": importances,
    }


def predict_pitcher_priors(
    model_bundle: dict[str, Any],
    season: int,
    min_ip: float = 20.0,
) -> dict[int, float]:
    """Predict prior means for each pitcher in a season.

    Parameters
    ----------
    model_bundle : dict
        Output of ``train_pitcher_prior_model``.
    season : int
        Season whose features to use for prediction.
    min_ip : float
        Minimum IP to include.

    Returns
    -------
    dict[int, float]
        {pitcher_id: predicted_prior_value}.
    """
    if model_bundle.get("model") is None:
        return {}

    features = _build_pitcher_season_features(season)
    features = features[features["ip"] >= min_ip].copy()

    if features.empty:
        return {}

    feature_cols = model_bundle["feature_cols"]
    avail = [c for c in feature_cols if c in features.columns]

    # Add missing columns as NaN (XGBoost handles NaN natively)
    for c in feature_cols:
        if c not in features.columns:
            features[c] = np.nan

    X = features[feature_cols].copy()
    pred = model_bundle["model"].predict(X)

    predictions = {
        int(pid): float(val)
        for pid, val in zip(features["pitcher_id"], pred)
    }

    target = model_bundle["target"]
    logger.info(
        "Predicted %s priors for %d pitchers (season %d features): "
        "mean=%.4f, range=[%.4f, %.4f]",
        target, len(predictions), season,
        np.mean(list(predictions.values())),
        min(predictions.values()),
        max(predictions.values()),
    )
    return predictions


def train_all_pitcher_priors(
    seasons: list[int] | None = None,
    min_ip: float = 40.0,
) -> dict[str, dict[str, Any]]:
    """Train both ERA-FIP gap and BABIP prior models.

    Parameters
    ----------
    seasons : list[int], optional
        All available seasons. Defaults to 2018-2025.
    min_ip : float
        Minimum IP to include.

    Returns
    -------
    dict[str, dict]
        {"era_fip_gap": model_bundle, "babip": model_bundle}.
    """
    if seasons is None:
        seasons = list(range(2018, 2026))

    return {
        "era_fip_gap": train_pitcher_prior_model(seasons, "era_fip_gap", min_ip),
        "babip": train_pitcher_prior_model(seasons, "babip", min_ip),
    }
