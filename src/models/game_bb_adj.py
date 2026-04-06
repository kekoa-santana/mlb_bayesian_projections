"""
Game-level BB% XGBoost adjustment model.

Learns game-context-dependent logit adjustments for pitcher BB rate
from features the additive Bayesian model cannot capture: umpire
tendencies (not applied in production sim), opposing lineup walk
proneness, talent × context interactions, and rest effects.

Architecture:
    Game context features → XGBoost → logit δ
    ↓
    Applied as GameContext.xgb_bb_lift in PA outcome model
    ↓
    Shifts BB rate per PA while preserving posterior CI width

Training: pitcher starts 2018-2025, walk-forward validated.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

from src.data.db import read_sql

logger = logging.getLogger(__name__)

# Feature columns used by the model.
# Interactions and non-linear terms that the additive logit model misses.
GAME_BB_FEATURES = [
    "pitcher_bb_rate",          # base talent (reference)
    "pitcher_zone_pct",         # command metric
    "umpire_bb_lift",           # umpire tendency (main effect, not in prod sim)
    "lineup_avg_bb_rate",       # opposing lineup walk proneness
    "is_home",                  # home/away
    "days_rest",                # rest since last start
    "pitcher_bb_x_umpire",      # interaction: wild pitcher × generous ump
    "pitcher_bb_x_lineup_bb",   # interaction: wild pitcher × patient lineup
    "pitcher_zone_x_umpire",    # interaction: good command × ump
]

_CLIP_LO = 0.01
_CLIP_HI = 0.99


# ---------------------------------------------------------------------------
# 1. Data assembly
# ---------------------------------------------------------------------------

def _get_pitcher_starts(seasons: list[int], min_bf: int = 15) -> pd.DataFrame:
    """Get pitcher starts with actuals, home/away, umpire, days rest.

    Single efficient query against fact_player_game_mlb + dim_game +
    dim_umpire with a LAG window for days rest.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_bf : int
        Minimum batters faced to include a start.

    Returns
    -------
    pd.DataFrame
        One row per pitcher start.
    """
    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH starter_games AS (
        SELECT
            fpg.game_pk,
            fpg.player_id   AS pitcher_id,
            fpg.season,
            fpg.game_date,
            fpg.pit_bb       AS actual_bb,
            fpg.pit_bf       AS actual_bf,
            fpg.team_id      AS pitcher_team_id,
            CASE WHEN fpg.team_id = dg.home_team_id THEN 1 ELSE 0 END AS is_home,
            LAG(fpg.game_date) OVER (
                PARTITION BY fpg.player_id ORDER BY fpg.game_date
            ) AS prev_start_date
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true
          AND fpg.pit_bf >= {int(min_bf)}
          AND fpg.pit_ip > 0
          AND dg.game_type = 'R'
          AND dg.season IN ({season_list})
    )
    SELECT
        sg.game_pk,
        sg.pitcher_id,
        sg.season,
        sg.game_date,
        sg.actual_bb,
        sg.actual_bf,
        sg.pitcher_team_id,
        sg.is_home,
        CASE
            WHEN sg.prev_start_date IS NOT NULL
                 AND (sg.game_date - sg.prev_start_date) BETWEEN 3 AND 14
            THEN (sg.game_date - sg.prev_start_date)
            ELSE 5
        END AS days_rest,
        du.hp_umpire_name
    FROM starter_games sg
    LEFT JOIN production.dim_umpire du ON sg.game_pk = du.game_pk
    ORDER BY sg.season, sg.game_date, sg.pitcher_id
    """
    logger.info("Fetching pitcher starts for seasons %s", seasons)
    df = read_sql(query, {})
    logger.info("Retrieved %d pitcher starts", len(df))
    return df


def _get_pitcher_season_bb_rates(seasons: list[int]) -> pd.DataFrame:
    """Pitcher seasonal BB rate from aggregated game stats.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, pitcher_bb_rate.
    """
    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    SELECT
        player_id AS pitcher_id,
        season,
        SUM(pit_bb)::float / NULLIF(SUM(pit_bf), 0) AS pitcher_bb_rate
    FROM production.fact_player_game_mlb
    WHERE pit_bf > 0
      AND season IN ({season_list})
    GROUP BY player_id, season
    HAVING SUM(pit_bf) >= 50
    """
    return read_sql(query, {})


def _get_lineup_avg_bb_rates(seasons: list[int]) -> pd.DataFrame:
    """Opposing lineup average BB rate per pitcher start.

    Joins pitcher starts → opposing lineup → batter season BB rates,
    aggregating to one lineup_avg_bb_rate per (game_pk, pitcher_id).

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, lineup_avg_bb_rate.
    """
    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH pitcher_starts AS (
        SELECT
            fpg.game_pk,
            fpg.player_id AS pitcher_id,
            fpg.team_id   AS pitcher_team_id,
            fpg.season
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true
          AND fpg.pit_bf >= 15
          AND dg.game_type = 'R'
          AND dg.season IN ({season_list})
    ),
    batter_season_bb AS (
        SELECT
            player_id AS batter_id,
            season,
            SUM(bat_bb)::float / NULLIF(SUM(bat_pa), 0) AS batter_bb_rate
        FROM production.fact_player_game_mlb
        WHERE bat_pa > 0
          AND season IN ({season_list})
        GROUP BY player_id, season
        HAVING SUM(bat_pa) >= 50
    )
    SELECT
        ps.game_pk,
        ps.pitcher_id,
        AVG(COALESCE(bbb.batter_bb_rate, 0.085)) AS lineup_avg_bb_rate
    FROM pitcher_starts ps
    JOIN production.fact_lineup fl
        ON ps.game_pk = fl.game_pk
        AND fl.team_id != ps.pitcher_team_id
        AND fl.batting_order BETWEEN 1 AND 9
        AND fl.is_starter = true
    LEFT JOIN batter_season_bb bbb
        ON fl.player_id = bbb.batter_id
        AND ps.season = bbb.season
    GROUP BY ps.game_pk, ps.pitcher_id
    """
    logger.info("Computing lineup avg BB rates for seasons %s", seasons)
    return read_sql(query, {})


def _compute_umpire_bb_lifts(seasons: list[int]) -> dict[str, float]:
    """Compute umpire BB logit lifts from multi-season data.

    Parameters
    ----------
    seasons : list[int]
        Seasons to use for umpire tendency estimation.

    Returns
    -------
    dict[str, float]
        {hp_umpire_name: bb_logit_lift}.
    """
    from src.data.queries import get_umpire_tendencies

    ump = get_umpire_tendencies(seasons=seasons, min_games=30)
    if ump.empty:
        return {}
    return dict(zip(ump["hp_umpire_name"], ump["bb_logit_lift"]))


# ---------------------------------------------------------------------------
# 2. Feature matrix builder
# ---------------------------------------------------------------------------

def build_game_bb_features(
    seasons: list[int],
    min_bf: int = 15,
) -> pd.DataFrame:
    """Build complete game-level feature matrix for BB adjustment model.

    Assembles pitcher starts, seasonal profiles, umpire tendencies,
    and opposing lineup composition into a single DataFrame with
    interaction features and the training target.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_bf : int
        Minimum BF per start.

    Returns
    -------
    pd.DataFrame
        One row per pitcher start with features and target column.
    """
    from src.data.feature_eng import get_cached_pitcher_observed_profile

    # 1. Pitcher starts with actuals + home/away + umpire + rest
    starts = _get_pitcher_starts(seasons, min_bf)
    if starts.empty:
        return pd.DataFrame()

    # 2. Pitcher seasonal BB rate
    season_bb = _get_pitcher_season_bb_rates(seasons)
    starts = starts.merge(
        season_bb, on=["pitcher_id", "season"], how="left",
    )
    starts = starts.dropna(subset=["pitcher_bb_rate"])

    # 3. Pitcher zone_pct from observed profiles
    zone_frames = []
    for s in seasons:
        try:
            obs = get_cached_pitcher_observed_profile(s)
            zone = obs[["pitcher_id", "zone_pct"]].drop_duplicates("pitcher_id")
            zone["season"] = s
            zone_frames.append(zone)
        except Exception:
            logger.debug("No pitcher profile for season %d", s)
    if zone_frames:
        zone_df = pd.concat(zone_frames, ignore_index=True)
        starts = starts.merge(
            zone_df, on=["pitcher_id", "season"], how="left",
        )
    else:
        starts["zone_pct"] = np.nan
    starts.rename(columns={"zone_pct": "pitcher_zone_pct"}, inplace=True)
    starts["pitcher_zone_pct"] = starts["pitcher_zone_pct"].fillna(0.45)

    # 4. Umpire BB lifts
    ump_lift_map = _compute_umpire_bb_lifts(seasons)
    starts["umpire_bb_lift"] = (
        starts["hp_umpire_name"].map(ump_lift_map).fillna(0.0)
    )

    # 5. Opposing lineup avg BB rate
    lineup_bb = _get_lineup_avg_bb_rates(seasons)
    starts = starts.merge(
        lineup_bb, on=["game_pk", "pitcher_id"], how="left",
    )
    starts["lineup_avg_bb_rate"] = starts["lineup_avg_bb_rate"].fillna(0.085)

    # 6. Compute target: total game-level BB deviation on logit scale
    actual_bb_rate = (starts["actual_bb"] / starts["actual_bf"]).clip(
        _CLIP_LO, _CLIP_HI,
    )
    pitcher_bb_rate_clipped = starts["pitcher_bb_rate"].clip(_CLIP_LO, _CLIP_HI)
    starts["target"] = logit(actual_bb_rate) - logit(pitcher_bb_rate_clipped)

    # 7. Interaction features
    starts["pitcher_bb_x_umpire"] = (
        starts["pitcher_bb_rate"] * starts["umpire_bb_lift"]
    )
    starts["pitcher_bb_x_lineup_bb"] = (
        starts["pitcher_bb_rate"] * starts["lineup_avg_bb_rate"]
    )
    starts["pitcher_zone_x_umpire"] = (
        starts["pitcher_zone_pct"] * starts["umpire_bb_lift"]
    )

    # Ensure days_rest is numeric
    starts["days_rest"] = pd.to_numeric(starts["days_rest"], errors="coerce").fillna(5)

    n_valid = starts[GAME_BB_FEATURES + ["target"]].dropna().shape[0]
    logger.info(
        "Built game BB feature matrix: %d rows (%d with all features)",
        len(starts), n_valid,
    )
    return starts


# ---------------------------------------------------------------------------
# 3. Model training
# ---------------------------------------------------------------------------

def train_game_bb_model(
    seasons: list[int] | None = None,
    min_bf: int = 15,
) -> dict[str, Any]:
    """Train XGBoost model for game-level BB adjustment.

    Predicts the logit-scale deviation of a pitcher's game BB rate
    from their seasonal BB rate, based on game context features.

    Parameters
    ----------
    seasons : list[int], optional
        Training seasons. Defaults to 2018-2025.
    min_bf : int
        Minimum BF per start.

    Returns
    -------
    dict[str, Any]
        Bundle with 'model', 'feature_cols', 'n_train', 'rmse_logit',
        'feature_importances'.
    """
    import xgboost as xgb

    if seasons is None:
        seasons = list(range(2018, 2026))

    features = build_game_bb_features(seasons, min_bf)
    if features.empty:
        logger.warning("No training data for game BB model")
        return {"model": None, "feature_cols": GAME_BB_FEATURES, "n_train": 0}

    # Drop rows with missing features or target
    mask = features[GAME_BB_FEATURES + ["target"]].notna().all(axis=1)
    train = features[mask].copy()

    if len(train) < 100:
        logger.warning("Too few training rows (%d) for game BB model", len(train))
        return {"model": None, "feature_cols": GAME_BB_FEATURES, "n_train": len(train)}

    X = train[GAME_BB_FEATURES]
    y = train["target"]

    # Conservative XGBoost: prevent overfitting on ~20K rows with 9 features.
    # Game-level BB is noisy (single game counts), so heavy regularization
    # and shallow trees are critical.
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        reg_alpha=2.0,
        reg_lambda=8.0,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=15,
        random_state=42,
    )
    model.fit(X, y)

    # Training diagnostics
    predictions = model.predict(X)
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))

    importances = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
    logger.info(
        "Trained game BB XGBoost: n=%d, RMSE(logit)=%.4f, "
        "top features: %s",
        len(X), rmse,
        ", ".join(f"{f}={v:.3f}" for f, v in top_features),
    )

    return {
        "model": model,
        "feature_cols": list(X.columns),
        "n_train": len(X),
        "rmse_logit": rmse,
        "feature_importances": importances,
    }


# ---------------------------------------------------------------------------
# 4. Walk-forward cross-validation
# ---------------------------------------------------------------------------

def cross_validate_game_bb(
    seasons: list[int] | None = None,
    min_bf: int = 15,
    min_train_seasons: int = 3,
) -> pd.DataFrame:
    """Walk-forward cross-validation for the game BB model.

    For each fold, trains on all prior seasons and evaluates on the
    next season.  Reports RMSE improvement vs baseline (zero adjustment).

    Parameters
    ----------
    seasons : list[int], optional
        All available seasons. Defaults to 2018-2025.
    min_bf : int
        Minimum BF per start.
    min_train_seasons : int
        Minimum training seasons before first fold.

    Returns
    -------
    pd.DataFrame
        Per-fold metrics: test_season, n_games, rmse_baseline,
        rmse_xgb, improvement_pct.
    """
    import xgboost as xgb

    if seasons is None:
        seasons = list(range(2018, 2026))

    # Build all features once (efficient)
    all_features = build_game_bb_features(seasons, min_bf)
    if all_features.empty:
        return pd.DataFrame()

    mask = all_features[GAME_BB_FEATURES + ["target"]].notna().all(axis=1)
    all_features = all_features[mask].copy()

    results = []
    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_mask = all_features["season"].isin(train_seasons)
        test_mask = all_features["season"] == test_season

        X_train = all_features.loc[train_mask, GAME_BB_FEATURES]
        y_train = all_features.loc[train_mask, "target"]
        X_test = all_features.loc[test_mask, GAME_BB_FEATURES]
        y_test = all_features.loc[test_mask, "target"]

        if len(X_train) < 100 or len(X_test) < 50:
            continue

        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            reg_alpha=2.0,
            reg_lambda=8.0,
            subsample=0.7,
            colsample_bytree=0.8,
            min_child_weight=15,
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse_xgb = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        rmse_baseline = float(np.sqrt(np.mean(y_test ** 2)))  # baseline: δ=0

        improvement = (1 - rmse_xgb / rmse_baseline) * 100 if rmse_baseline > 0 else 0.0

        results.append({
            "test_season": test_season,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "rmse_baseline": round(rmse_baseline, 4),
            "rmse_xgb": round(rmse_xgb, 4),
            "improvement_pct": round(improvement, 2),
        })
        logger.info(
            "CV fold %d→%d: n_train=%d, n_test=%d, "
            "RMSE baseline=%.4f, XGB=%.4f, improvement=%.1f%%",
            train_seasons[-1], test_season,
            len(X_train), len(X_test),
            rmse_baseline, rmse_xgb, improvement,
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Inference
# ---------------------------------------------------------------------------

def predict_game_bb_adjustment(
    model_bundle: dict[str, Any],
    pitcher_bb_rate: float,
    pitcher_zone_pct: float = 0.45,
    umpire_bb_lift: float = 0.0,
    lineup_avg_bb_rate: float = 0.085,
    is_home: int = 0,
    days_rest: int = 5,
) -> float:
    """Predict logit-scale BB adjustment for a single game.

    Parameters
    ----------
    model_bundle : dict
        Output of ``train_game_bb_model``.
    pitcher_bb_rate : float
        Pitcher's seasonal BB rate.
    pitcher_zone_pct : float
        Pitcher's zone percentage.
    umpire_bb_lift : float
        HP umpire's BB logit lift (from umpire tendencies).
    lineup_avg_bb_rate : float
        Opposing lineup's average BB rate.
    is_home : int
        1 if pitcher is at home, 0 if away.
    days_rest : int
        Days since pitcher's last start.

    Returns
    -------
    float
        Logit-scale adjustment δ. Positive = model expects more BB
        than the seasonal rate alone would predict.
    """
    if model_bundle.get("model") is None:
        return 0.0

    features = {
        "pitcher_bb_rate": pitcher_bb_rate,
        "pitcher_zone_pct": pitcher_zone_pct,
        "umpire_bb_lift": umpire_bb_lift,
        "lineup_avg_bb_rate": lineup_avg_bb_rate,
        "is_home": is_home,
        "days_rest": days_rest,
        "pitcher_bb_x_umpire": pitcher_bb_rate * umpire_bb_lift,
        "pitcher_bb_x_lineup_bb": pitcher_bb_rate * lineup_avg_bb_rate,
        "pitcher_zone_x_umpire": pitcher_zone_pct * umpire_bb_lift,
    }

    X = pd.DataFrame([features])[model_bundle["feature_cols"]]
    delta = float(model_bundle["model"].predict(X)[0])

    return delta


def predict_game_bb_batch(
    model_bundle: dict[str, Any],
    game_features: pd.DataFrame,
) -> np.ndarray:
    """Predict logit-scale BB adjustments for a batch of games.

    Parameters
    ----------
    model_bundle : dict
        Output of ``train_game_bb_model``.
    game_features : pd.DataFrame
        Must contain all columns in GAME_BB_FEATURES.

    Returns
    -------
    np.ndarray
        Array of logit δ values, shape (n_games,).
    """
    if model_bundle.get("model") is None:
        return np.zeros(len(game_features))

    # Ensure interaction features exist
    df = game_features.copy()
    if "pitcher_bb_x_umpire" not in df.columns:
        df["pitcher_bb_x_umpire"] = df["pitcher_bb_rate"] * df["umpire_bb_lift"]
    if "pitcher_bb_x_lineup_bb" not in df.columns:
        df["pitcher_bb_x_lineup_bb"] = df["pitcher_bb_rate"] * df["lineup_avg_bb_rate"]
    if "pitcher_zone_x_umpire" not in df.columns:
        df["pitcher_zone_x_umpire"] = df["pitcher_zone_pct"] * df["umpire_bb_lift"]

    X = df[model_bundle["feature_cols"]]
    return model_bundle["model"].predict(X)
