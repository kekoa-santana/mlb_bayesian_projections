"""Shared XGBoost breakout model engine.

Provides training, SHAP-based archetype clustering, scoring, and walk-forward
validation infrastructure for both hitter and pitcher breakout models.

Architecture:
    Tiered features (T1 boxscore 2000+ / T2 Statcast 2018+ / T3 MiLB) →
    XGBoost binary classifier → P(breakout) →
    SHAP values → KMeans archetype clustering →
    breakout_score = P(breakout) × room_to_grow
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BreakoutConfig:
    """Configuration for a breakout model (hitter or pitcher)."""

    player_type: str              # "hitter" or "pitcher"
    id_col: str                   # "batter_id" or "pitcher_id"
    name_col: str                 # "batter_name" or "pitcher_name"
    feature_cols: list[str]       # all feature columns (T1 + T2 + T3)
    breakout_metric: str          # column to measure breakout ("est_woba" or "era")
    breakout_direction: str       # "gain" or "drop"
    breakout_threshold: float     # minimum delta for breakout
    outcome_bound: float          # outcome floor (wOBA >= 0.300) or ceiling (ERA <= 4.00)
    outcome_is_floor: bool        # True = outcome must be >= bound, False = <=
    n_archetypes: int             # number of SHAP clusters
    age_window: tuple[int, int]   # (peak_start, peak_end) for piecewise age curve
    xgb_params: dict = field(default_factory=dict)


# Feature name → human-readable label for hole identification
FEATURE_LABELS: dict[str, str] = {
    # Hitter T1
    "k_pct": "Strikeout rate",
    "bb_pct": "Walk rate",
    "iso": "Isolated power",
    "hr_ab": "Home run rate",
    "babip": "BABIP",
    "slg": "Slugging",
    "obp": "On-base",
    "ops": "OPS",
    "go_ao": "Ground ball tendency",
    "age": "Age",
    "delta_k_pct": "K% trend",
    "delta_bb_pct": "BB% trend",
    "delta_iso": "ISO trend",
    "mlb_seasons": "Experience",
    "est_woba": "Estimated wOBA",
    # Hitter T2
    "barrel_pct": "Barrel rate",
    "hard_hit_pct": "Hard-hit rate",
    "xwoba": "Expected wOBA",
    "whiff_rate": "Whiff rate",
    "chase_rate": "Chase rate",
    "z_contact_pct": "Zone contact",
    "avg_exit_velo": "Exit velocity",
    "sprint_speed": "Sprint speed",
    "oaa": "Fielding (OAA)",
    "woba": "wOBA",
    "xwoba_minus_woba": "xwOBA-wOBA gap",
    # Hitter T3
    "milb_translated_k_pct": "MiLB strikeout rate",
    "milb_translated_bb_pct": "MiLB walk rate",
    "milb_translated_iso": "MiLB power",
    "milb_age_relative": "Age-for-level",
    "fg_future_value": "Prospect pedigree",
    # Pitcher T1
    "hr_bf": "HR rate",
    "era": "ERA",
    "fip": "FIP",
    "whip": "WHIP",
    "k_per_9": "K/9",
    "bb_per_9": "BB/9",
    "delta_era": "ERA trend",
    "is_starter": "Role",
    # Pitcher T2
    "swstr_pct": "Swinging strike rate",
    "csw_pct": "CSW rate",
    "avg_velo": "Velocity",
    "xwoba_against": "xwOBA against",
    "barrel_pct_against": "Barrel rate against",
    "zone_pct": "Zone rate",
    "chase_pct": "Chase rate (induced)",
    "hr_per_fb": "HR/FB",
    "era_minus_xfip": "ERA-xFIP gap",
    # Pitcher T3
    "milb_translated_k_pct": "MiLB K rate",
    "milb_translated_bb_pct": "MiLB BB rate",
    "milb_translated_hr_bf": "MiLB HR rate",
}


def _default_xgb_params() -> dict:
    """Conservative XGBoost hyperparameters."""
    return {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 15,
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
    }


# ---------------------------------------------------------------------------
# Piecewise age curve
# ---------------------------------------------------------------------------

def piecewise_age_mult(
    age: float | np.ndarray,
    peak_start: int = 24,
    peak_end: int = 27,
) -> float | np.ndarray:
    """Piecewise developmental age multiplier.

    Flat 1.0 in prime window, ramp up before, decay after.
    """
    age = np.asarray(age, dtype=float)
    mult = np.where(
        age < peak_start,
        np.clip(0.50 + (age - 20) * (0.50 / (peak_start - 20)), 0.50, 1.0),
        np.where(
            age <= peak_end,
            1.0,
            np.clip(1.0 - (age - peak_end) * 0.10, 0.15, 1.0),
        ),
    )
    return float(mult) if mult.ndim == 0 else mult


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def build_walk_forward_data(
    config: BreakoutConfig,
    feature_loader: Callable[[int, int], pd.DataFrame],
    outcome_loader: Callable[[int, int], pd.DataFrame],
    folds: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build pooled (X, y, meta) across all walk-forward folds.

    Parameters
    ----------
    config : BreakoutConfig
    feature_loader : callable(season, min_pa/bf) -> DataFrame
    outcome_loader : callable(season, min_pa/bf) -> DataFrame
        Must return id_col + breakout_metric column.
    folds : list of (pred_season, outcome_season) pairs

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all tiers, NaN where unavailable).
    y : pd.Series
        Binary breakout labels.
    meta : pd.DataFrame
        Metadata (id, season, metric values) for diagnostics.
    """
    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    all_meta: list[pd.DataFrame] = []
    id_col = config.id_col
    metric = config.breakout_metric

    for pred_season, out_season in folds:
        try:
            features = feature_loader(pred_season, 100)  # lower bar for training
            outcomes = outcome_loader(out_season, 80)
        except Exception as e:
            logger.warning("Skipping fold %d→%d: %s", pred_season, out_season, e)
            continue

        if features.empty or outcomes.empty:
            continue

        merged = features.merge(
            outcomes[[id_col, metric]].rename(columns={metric: "_outcome"}),
            on=id_col, how="inner",
        )
        if merged.empty:
            continue

        # Label breakouts
        delta = merged["_outcome"] - merged.get(metric, merged["_outcome"])
        if config.breakout_direction == "gain":
            is_breakout = (delta >= config.breakout_threshold)
            if config.outcome_is_floor:
                is_breakout &= (merged["_outcome"] >= config.outcome_bound)
            else:
                is_breakout &= (merged["_outcome"] <= config.outcome_bound)
        else:  # "drop"
            is_breakout = (delta <= -config.breakout_threshold)
            if config.outcome_is_floor:
                is_breakout &= (merged["_outcome"] >= config.outcome_bound)
            else:
                is_breakout &= (merged["_outcome"] <= config.outcome_bound)

        # Extract features
        avail = [c for c in config.feature_cols if c in merged.columns]
        X_fold = merged[avail].copy()
        # Add missing feature cols as NaN
        for c in config.feature_cols:
            if c not in X_fold.columns:
                X_fold[c] = np.nan
        X_fold = X_fold[config.feature_cols]

        y_fold = is_breakout.astype(int)
        meta_fold = merged[[id_col]].copy()
        meta_fold["pred_season"] = pred_season
        meta_fold["out_season"] = out_season
        meta_fold["_outcome"] = merged["_outcome"]
        meta_fold["breakout"] = y_fold.values

        all_X.append(X_fold)
        all_y.append(y_fold)
        all_meta.append(meta_fold)

        n_bo = int(y_fold.sum())
        logger.info(
            "  %d→%d: %d players, %d breakouts (%.1f%%)",
            pred_season, out_season, len(merged), n_bo,
            100 * n_bo / max(len(merged), 1),
        )

    if not all_X:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    meta = pd.concat(all_meta, ignore_index=True)

    n_pos = int(y.sum())
    logger.info(
        "Training data: %d total, %d breakouts (%.1f%%)",
        len(X), n_pos, 100 * n_pos / max(len(X), 1),
    )
    return X, y, meta


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_breakout_model(
    config: BreakoutConfig,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """Train XGBoost classifier with SHAP archetype discovery.

    Parameters
    ----------
    config : BreakoutConfig
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary labels.

    Returns
    -------
    dict
        Model bundle: model, feature_cols, shap_values, archetype_model,
        archetype_names, metrics.
    """
    import xgboost as xgb

    if X.empty or y.sum() < 10:
        logger.warning("Insufficient data for breakout model training")
        return {"model": None}

    # Class imbalance weight
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    params = _default_xgb_params()
    params.update(config.xgb_params)
    params["scale_pos_weight"] = scale_pos

    model = xgb.XGBClassifier(**params)
    model.fit(X, y)

    # Training metrics
    train_proba = model.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score, log_loss
    train_auc = roc_auc_score(y, train_proba)
    train_loss = log_loss(y, train_proba)

    # Feature importances
    importances = dict(zip(X.columns, model.feature_importances_))
    top_feats = sorted(importances.items(), key=lambda x: -x[1])[:8]
    logger.info(
        "XGBoost trained: n=%d, pos=%d (%.1f%%), train AUC=%.3f, "
        "top features: %s",
        len(X), n_pos, 100 * n_pos / len(X), train_auc,
        ", ".join(f"{f}={v:.3f}" for f, v in top_feats),
    )

    # SHAP values for archetype discovery
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
    except ImportError:
        logger.warning("shap not installed — archetype clustering disabled")
        shap_values = None

    # Archetype clustering on SHAP values
    archetype_model = None
    archetype_names: dict[int, str] = {}
    if shap_values is not None and config.n_archetypes > 1:
        archetype_model, archetype_names = _cluster_archetypes(
            shap_values, X.columns.tolist(), config,
        )

    return {
        "model": model,
        "feature_cols": list(X.columns),
        "shap_values_train": shap_values,
        "archetype_model": archetype_model,
        "archetype_names": archetype_names,
        "metrics": {
            "train_auc": train_auc,
            "train_loss": train_loss,
            "n_train": len(X),
            "n_positive": n_pos,
        },
        "feature_importances": importances,
    }


def _cluster_archetypes(
    shap_values: np.ndarray,
    feature_names: list[str],
    config: BreakoutConfig,
) -> tuple[Any, dict[int, str]]:
    """Cluster SHAP values to discover breakout archetypes.

    Returns (kmeans_model, {cluster_id: archetype_name}).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Only cluster rows where SHAP values are meaningful
    # (positive class — focus on what drives breakout prediction)
    scaler = StandardScaler()
    X_shap = scaler.fit_transform(shap_values)

    km = KMeans(
        n_clusters=config.n_archetypes,
        random_state=42,
        n_init=10,
    )
    km.fit(X_shap)

    # Auto-name clusters by examining top SHAP dimensions per centroid
    centroids = scaler.inverse_transform(km.cluster_centers_)
    names: dict[int, str] = {}

    if config.player_type == "hitter":
        names = _name_hitter_archetypes(centroids, feature_names)
    else:
        names = _name_pitcher_archetypes(centroids, feature_names)

    for i, name in names.items():
        n_members = int((km.labels_ == i).sum())
        logger.info("  Archetype %d: %s (%d members)", i, name, n_members)

    # Store scaler with model for inference
    km._shap_scaler = scaler  # type: ignore[attr-defined]
    return km, names


def _name_hitter_archetypes(
    centroids: np.ndarray,
    feature_names: list[str],
) -> dict[int, str]:
    """Auto-name hitter archetypes from SHAP centroids."""
    names: dict[int, str] = {}
    n = centroids.shape[0]

    # Find which centroid has the highest power-related SHAP features
    power_features = {"iso", "barrel_pct", "hard_hit_pct", "avg_exit_velo",
                      "hr_ab", "slg"}
    approach_features = {"k_pct", "bb_pct", "chase_rate", "whiff_rate",
                         "z_contact_pct", "delta_k_pct", "delta_bb_pct"}

    power_scores = np.zeros(n)
    approach_scores = np.zeros(n)
    for j, fname in enumerate(feature_names):
        if fname in power_features:
            power_scores += np.abs(centroids[:, j])
        if fname in approach_features:
            approach_scores += np.abs(centroids[:, j])

    power_cluster = int(power_scores.argmax())
    names[power_cluster] = "Power Surge"

    for i in range(n):
        if i not in names:
            names[i] = "Diamond in the Rough"

    return names


def _name_pitcher_archetypes(
    centroids: np.ndarray,
    feature_names: list[str],
) -> dict[int, str]:
    """Auto-name pitcher archetypes from SHAP centroids."""
    names: dict[int, str] = {}
    n = centroids.shape[0]

    stuff_features = {"whiff_rate", "swstr_pct", "csw_pct", "avg_velo",
                      "k_pct", "k_per_9"}
    command_features = {"bb_pct", "bb_per_9", "zone_pct", "chase_pct",
                        "delta_bb_pct"}
    era_features = {"era", "fip", "era_minus_xfip", "hr_per_fb", "hr_bf"}

    stuff_scores = np.zeros(n)
    command_scores = np.zeros(n)
    era_scores = np.zeros(n)
    for j, fname in enumerate(feature_names):
        if fname in stuff_features:
            stuff_scores += np.abs(centroids[:, j])
        if fname in command_features:
            command_scores += np.abs(centroids[:, j])
        if fname in era_features:
            era_scores += np.abs(centroids[:, j])

    # Assign in priority order
    stuff_cluster = int(stuff_scores.argmax())
    names[stuff_cluster] = "Stuff Dominant"

    remaining = [i for i in range(n) if i not in names]
    if remaining:
        cmd_cluster = max(remaining, key=lambda i: command_scores[i])
        names[cmd_cluster] = "Command Leap"

    for i in range(n):
        if i not in names:
            names[i] = "ERA Correction"

    return names


# ---------------------------------------------------------------------------
# Archetype validation (raw-stats override)
# ---------------------------------------------------------------------------

def _validate_hitter_archetypes(
    df: pd.DataFrame,
    archetype_names: dict[int, str],
) -> pd.Series:
    """Override SHAP archetype when raw power stats contradict 'Power Surge'.

    A player with below-median power metrics (barrel%, exit velo, ISO all
    below the cohort's 35th percentile) cannot be Power Surge — force to
    Diamond in the Rough.
    """
    result = df["breakout_type"].copy()

    # Compute power percentiles within this cohort
    iso_pctl = df["iso"].rank(pct=True) if "iso" in df.columns else pd.Series(0.5, index=df.index)
    ev_pctl = df["avg_exit_velo"].rank(pct=True) if "avg_exit_velo" in df.columns else pd.Series(0.5, index=df.index)
    barrel_pctl = df["barrel_pct"].rank(pct=True) if "barrel_pct" in df.columns else pd.Series(0.5, index=df.index)

    # If 2+ of 3 power metrics are below 35th percentile, can't be Power Surge
    low_power = (
        (iso_pctl < 0.35).astype(int)
        + (ev_pctl < 0.35).astype(int)
        + (barrel_pctl < 0.35).astype(int)
    ) >= 2
    overrides = (result == "Power Surge") & low_power
    n_overrides = overrides.sum()
    if n_overrides > 0:
        result.loc[overrides] = "Diamond in the Rough"
        logger.info(
            "Archetype override: %d players moved Power Surge -> Diamond in the Rough "
            "(low power metrics)", n_overrides,
        )

    # Reverse: Diamond in the Rough with elite power should be Power Surge
    high_power = (iso_pctl > 0.70) & (ev_pctl > 0.70)
    reverse = (result == "Diamond in the Rough") & high_power
    n_reverse = reverse.sum()
    if n_reverse > 0:
        result.loc[reverse] = "Power Surge"
        logger.info(
            "Archetype override: %d players moved Diamond in the Rough -> Power Surge "
            "(high power metrics)", n_reverse,
        )

    return result


def _validate_pitcher_archetypes(
    df: pd.DataFrame,
    archetype_names: dict[int, str],
) -> pd.Series:
    """Override SHAP archetype when raw stats clearly contradict the label.

    - Stuff Dominant requires above-median K metrics (k_pct or whiff_rate)
    - Command Leap requires below-median walk rate
    """
    result = df["breakout_type"].copy()

    k_pctl = df["k_pct"].rank(pct=True) if "k_pct" in df.columns else pd.Series(0.5, index=df.index)
    bb_pctl = df["bb_pct"].rank(pct=True) if "bb_pct" in df.columns else pd.Series(0.5, index=df.index)

    # Stuff Dominant with bottom-quartile K% -> ERA Correction
    low_k = k_pctl < 0.25
    stuff_low_k = (result == "Stuff Dominant") & low_k
    if stuff_low_k.sum() > 0:
        result.loc[stuff_low_k] = "ERA Correction"
        logger.info("Archetype override: %d pitchers Stuff Dominant -> ERA Correction (low K%%)",
                     stuff_low_k.sum())

    # Command Leap with top-quartile BB% -> ERA Correction
    high_bb = bb_pctl > 0.75
    cmd_high_bb = (result == "Command Leap") & high_bb
    if cmd_high_bb.sum() > 0:
        result.loc[cmd_high_bb] = "ERA Correction"
        logger.info("Archetype override: %d pitchers Command Leap -> ERA Correction (high BB%%)",
                     cmd_high_bb.sum())

    return result


# ---------------------------------------------------------------------------
# Scoring & inference
# ---------------------------------------------------------------------------

def score_players(
    model_bundle: dict[str, Any],
    df: pd.DataFrame,
    config: BreakoutConfig,
) -> pd.DataFrame:
    """Score a cohort of players for breakout potential.

    Parameters
    ----------
    model_bundle : dict
        Output of train_breakout_model().
    df : pd.DataFrame
        Player features (must contain config.feature_cols + id/age/name).
    config : BreakoutConfig

    Returns
    -------
    pd.DataFrame
        Input df augmented with breakout_prob, breakout_score,
        breakout_type, breakout_tier, breakout_hole, breakout_narrative.
    """
    model = model_bundle.get("model")
    if model is None:
        logger.warning("No trained model — returning empty scores")
        for col in ["breakout_prob", "breakout_score", "breakout_type",
                     "breakout_tier", "breakout_hole", "breakout_narrative"]:
            df[col] = np.nan if "score" in col or "prob" in col else ""
        return df

    feature_cols = model_bundle["feature_cols"]

    # Prepare feature matrix
    X = df[[]].copy()
    for c in feature_cols:
        X[c] = df[c] if c in df.columns else np.nan

    # P(breakout) from XGBoost
    proba = model.predict_proba(X)[:, 1]
    df["breakout_prob"] = proba

    # Room to grow: piecewise age × trajectory
    age = df["age"].fillna(28).values
    peak_start, peak_end = config.age_window
    age_mult = piecewise_age_mult(age, peak_start, peak_end)

    # Trajectory from K%/BB% deltas
    delta_k = df.get("delta_k_pct", pd.Series(0, index=df.index)).fillna(0)
    delta_bb = df.get("delta_bb_pct", pd.Series(0, index=df.index)).fillna(0)

    if config.player_type == "hitter":
        k_improving = 1.0 - delta_k.rank(pct=True, method="average")
        bb_improving = delta_bb.rank(pct=True, method="average")
    else:  # pitcher: higher K = better, lower BB = better
        k_improving = delta_k.rank(pct=True, method="average")
        bb_improving = 1.0 - delta_bb.rank(pct=True, method="average")

    trajectory = 0.50 * k_improving + 0.50 * bb_improving
    room_to_grow = trajectory.values * age_mult
    df["room_to_grow"] = room_to_grow

    # Final score
    df["breakout_score"] = proba * room_to_grow

    # SHAP-based archetypes
    archetype_model = model_bundle.get("archetype_model")
    archetype_names = model_bundle.get("archetype_names", {})
    if archetype_model is not None:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            scaler = archetype_model._shap_scaler  # type: ignore[attr-defined]
            X_shap = scaler.transform(shap_vals)
            clusters = archetype_model.predict(X_shap)
            df["breakout_type"] = pd.Series(clusters, index=df.index).map(
                archetype_names)

            # Raw-stats sanity check: override SHAP cluster when stats
            # clearly contradict the archetype label
            if config.player_type == "hitter":
                df["breakout_type"] = _validate_hitter_archetypes(
                    df, archetype_names)
            elif config.player_type == "pitcher":
                df["breakout_type"] = _validate_pitcher_archetypes(
                    df, archetype_names)

            # SHAP-based hole identification
            df["breakout_hole"] = _identify_holes_shap(
                shap_vals, feature_cols, config)

            # Store archetype probabilities (soft: 1.0 for assigned, 0.0 otherwise)
            for i, name in archetype_names.items():
                col = f"prob_{name.lower().replace(' ', '_').replace('-', '_')}"
                df[col] = (df["breakout_type"] == name).astype(float)

        except Exception as e:
            logger.warning("SHAP archetype assignment failed: %s", e)
            df["breakout_type"] = "Unknown"
            df["breakout_hole"] = "General development"
    else:
        df["breakout_type"] = "Unknown"
        df["breakout_hole"] = "General development"

    # Tiers (within-archetype)
    if "breakout_type" in df.columns and df["breakout_type"].nunique() > 1:
        df["archetype_rank"] = (
            df.groupby("breakout_type")["breakout_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
    else:
        df["archetype_rank"] = (
            df["breakout_score"].rank(ascending=False, method="min").astype(int)
        )

    df["breakout_tier"] = np.where(
        df["archetype_rank"] <= 10, "Breakout Candidate",
        np.where(df["archetype_rank"] <= 25, "On the Radar", ""),
    )
    df["breakout_rank"] = (
        df["breakout_score"].rank(ascending=False, method="min").astype(int)
    )

    # Narratives
    df["breakout_narrative"] = df.apply(
        lambda row: _generate_narrative(row, config), axis=1)

    return df


def _identify_holes_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    config: BreakoutConfig,
) -> pd.Series:
    """Identify the primary weakness per player from SHAP values.

    The feature with the most negative SHAP contribution (dragging P(breakout)
    down the most) is the "hole."
    """
    holes: list[str] = []
    # Features that represent weaknesses when SHAP is negative
    for i in range(shap_values.shape[0]):
        row_shap = shap_values[i]
        # Find the most negative SHAP value
        min_idx = int(np.argmin(row_shap))
        min_feat = feature_names[min_idx]
        label = FEATURE_LABELS.get(min_feat, min_feat.replace("_", " ").title())
        holes.append(label)

    return pd.Series(holes)


def _generate_narrative(row: pd.Series, config: BreakoutConfig) -> str:
    """Generate a stat-driven explanation for why this player was flagged."""
    archetype = row.get("breakout_type", "")
    age = int(row.get("age", 0))
    hole = row.get("breakout_hole", "General development")
    prob = row.get("breakout_prob", 0)

    if config.player_type == "hitter":
        return _hitter_narrative(row, archetype, age, hole, prob)
    else:
        return _pitcher_narrative(row, archetype, age, hole, prob)


def _hitter_narrative(
    row: pd.Series, archetype: str, age: int, hole: str, prob: float,
) -> str:
    """Hitter breakout narrative."""
    woba = row.get("woba") or row.get("est_woba") or 0
    xwoba = row.get("xwoba") or 0

    if archetype == "Power Surge":
        strengths: list[str] = []
        ev = row.get("avg_exit_velo") or 0
        if ev and ev > 89:
            strengths.append(f"{ev:.1f} mph exit velo")
        hh = row.get("hard_hit_pct") or 0
        if hh and hh > 0.40:
            strengths.append(f"{hh:.0%} hard-hit rate")
        barrel = row.get("barrel_pct") or 0
        if barrel and barrel > 0.05:
            strengths.append(f"{barrel:.1%} barrel rate")
        iso_val = row.get("iso") or 0
        if iso_val and iso_val > 0.150 and not strengths:
            strengths.append(f".{int(iso_val*1000):03d} ISO")

        tools = " + ".join(strengths[:2]) if strengths else "strong batted ball profile"
        gap = (xwoba - woba) if xwoba and woba else 0
        if gap > 0.015:
            return (
                f"{tools} with .{int(xwoba*1000):03d} xwOBA vs "
                f".{int(woba*1000):03d} wOBA. "
                f"Production due for correction at age {age}."
            )
        return (
            f"{tools} but .{int(woba*1000):03d} wOBA. "
            f"{hole} is the unlock at age {age}."
        )

    else:  # Diamond in the Rough
        strengths = []
        sprint = row.get("sprint_speed") or 0
        if sprint and sprint > 28.0:
            strengths.append(f"{sprint:.1f} ft/s sprint speed")
        oaa = row.get("oaa") or 0
        if oaa and oaa >= 5:
            strengths.append(f"{int(oaa)} OAA")
        zc = row.get("z_contact_pct") or 0
        if zc and zc > 0.82:
            strengths.append(f"{zc:.0%} zone contact")
        chase = row.get("chase_rate") or 0
        if chase and chase < 0.25:
            strengths.append(f"elite {chase:.0%} chase rate")
        bb_pct = row.get("bb_pct") or 0
        if bb_pct and bb_pct > 0.10 and not strengths:
            strengths.append(f"{bb_pct:.1%} walk rate")

        if not strengths:
            strengths.append(f"age-{age} upside")

        tools = " + ".join(strengths[:2])
        return (
            f"{tools}, but just .{int(woba*1000):03d} wOBA. "
            f"{hole} is the development key at age {age}."
        )


def _pitcher_narrative(
    row: pd.Series, archetype: str, age: int, hole: str, prob: float,
) -> str:
    """Pitcher breakout narrative."""
    era = row.get("era") or 0

    if archetype == "Stuff Dominant":
        strengths: list[str] = []
        k_pct = row.get("k_pct") or 0
        if k_pct and k_pct > 0.22:
            strengths.append(f"{k_pct:.1%} K rate")
        swstr = row.get("swstr_pct") or 0
        if swstr and swstr > 0.11:
            strengths.append(f"{swstr:.1%} SwStr%")
        velo = row.get("avg_velo") or 0
        if velo and velo > 94:
            strengths.append(f"{velo:.1f} mph")

        tools = " + ".join(strengths[:2]) if strengths else "elite stuff metrics"
        bb = row.get("bb_pct") or 0
        return (
            f"{tools}, but {bb:.1%} walk rate keeps ERA at {era:.2f}. "
            f"Command development is the breakout trigger at age {age}."
        )

    elif archetype == "Command Leap":
        strengths = []
        bb = row.get("bb_pct") or 0
        if bb and bb < 0.08:
            strengths.append(f"low {bb:.1%} walk rate")
        zone = row.get("zone_pct") or 0
        if zone and zone > 0.45:
            strengths.append(f"{zone:.0%} zone rate")

        tools = " + ".join(strengths[:2]) if strengths else "solid command profile"
        fip = row.get("fip") or 0
        if fip and abs(era - fip) > 0.3:
            return (
                f"{tools} with {fip:.2f} FIP vs {era:.2f} ERA. "
                f"Command is there — results should converge."
            )
        return f"{tools} but {era:.2f} ERA. {hole} at age {age}."

    else:  # ERA Correction
        fip = row.get("fip") or 0
        gap = era - fip if fip else 0

        if gap > 0.5:
            target = min(era, fip + 0.50)
            return (
                f"{era:.2f} ERA vs {fip:.2f} FIP ({gap:+.2f} gap). "
                f"Peripherals say sub-{target:.1f} ERA is the true talent."
            )
        return (
            f"{era:.2f} ERA with peripherals suggesting improvement. "
            f"{hole} is the primary regression driver."
        )


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    config: BreakoutConfig,
    feature_loader: Callable[[int, int], pd.DataFrame],
    outcome_loader: Callable[[int, int], pd.DataFrame],
    folds: list[tuple[int, int]],
    min_train_folds: int = 5,
) -> pd.DataFrame:
    """Walk-forward validation: train on folds[:i], predict fold[i].

    Parameters
    ----------
    config : BreakoutConfig
    feature_loader, outcome_loader : callables
    folds : list of (pred_season, out_season)
    min_train_folds : int
        Minimum training folds before first prediction.

    Returns
    -------
    pd.DataFrame
        Per-fold metrics: fold, n_test, n_breakouts, auc, precision_at_25,
        base_rate, top_quartile_rate, lift.
    """
    from sklearn.metrics import roc_auc_score

    results: list[dict] = []

    for test_idx in range(min_train_folds, len(folds)):
        train_folds = folds[:test_idx]
        test_fold = folds[test_idx]

        # Build training data
        X_train, y_train, _ = build_walk_forward_data(
            config, feature_loader, outcome_loader, train_folds)

        if X_train.empty or y_train.sum() < 10:
            continue

        # Train model
        bundle = train_breakout_model(config, X_train, y_train)
        if bundle.get("model") is None:
            continue

        # Load test data
        pred_s, out_s = test_fold
        try:
            test_features = feature_loader(pred_s, 100)
            test_outcomes = outcome_loader(out_s, 80)
        except Exception:
            continue

        if test_features.empty or test_outcomes.empty:
            continue

        id_col = config.id_col
        metric = config.breakout_metric
        test_merged = test_features.merge(
            test_outcomes[[id_col, metric]].rename(
                columns={metric: "_outcome"}),
            on=id_col, how="inner",
        )
        if len(test_merged) < 20:
            continue

        # Compute labels
        delta = test_merged["_outcome"] - test_merged.get(
            metric, test_merged["_outcome"])
        if config.breakout_direction == "gain":
            y_test = (
                (delta >= config.breakout_threshold)
                & (test_merged["_outcome"] >= config.outcome_bound
                   if config.outcome_is_floor
                   else test_merged["_outcome"] <= config.outcome_bound)
            ).astype(int)
        else:
            y_test = (
                (delta <= -config.breakout_threshold)
                & (test_merged["_outcome"] <= config.outcome_bound
                   if not config.outcome_is_floor
                   else test_merged["_outcome"] >= config.outcome_bound)
            ).astype(int)

        # Predict
        X_test = test_merged[[]].copy()
        for c in bundle["feature_cols"]:
            X_test[c] = test_merged[c] if c in test_merged.columns else np.nan

        proba = bundle["model"].predict_proba(X_test)[:, 1]

        # Metrics
        base_rate = float(y_test.mean())
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")

        # Precision@25
        top25_idx = np.argsort(-proba)[:25]
        p_at_25 = float(y_test.iloc[top25_idx].mean()) if len(top25_idx) > 0 else 0

        # Top-quartile lift
        q75 = np.percentile(proba, 75)
        top_q = y_test[proba >= q75]
        top_q_rate = float(top_q.mean()) if len(top_q) > 0 else 0
        lift = top_q_rate / max(base_rate, 0.01)

        results.append({
            "fold": f"{pred_s}->{out_s}",
            "n_test": len(y_test),
            "n_breakouts": int(y_test.sum()),
            "base_rate": round(base_rate, 3),
            "auc": round(auc, 3),
            "precision_at_25": round(p_at_25, 3),
            "top_quartile_rate": round(top_q_rate, 3),
            "lift": round(lift, 2),
        })

        logger.info(
            "  Fold %s: n=%d, breakouts=%d (%.1f%%), AUC=%.3f, P@25=%.3f, lift=%.1fx",
            f"{pred_s}->{out_s}", len(y_test), int(y_test.sum()),
            100 * base_rate, auc, p_at_25, lift,
        )

    return pd.DataFrame(results)
