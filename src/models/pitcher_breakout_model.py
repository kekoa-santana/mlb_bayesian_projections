"""Pitcher breakout model — XGBoost with SHAP archetypes.

Replaces the GMM-based approach with XGBoost trained on 22+ walk-forward
folds (2001-2025), leveraging tiered features:

- **T1 (2000+):** boxscore rates (K%, BB%, ERA, FIP, WHIP, etc.)
- **T2 (2018+):** Statcast advanced (whiff%, SwStr%, CSW%, velo, etc.)
- **T3 (young pitchers):** translated MiLB rates

Archetypes (k=3) discovered via SHAP-value clustering:

1. **Stuff Dominant**: Elite whiff/K metrics — results follow the stuff.
2. **Command Leap**: Good command metrics — ERA catches up to peripherals.
3. **ERA Correction**: Biggest ERA drop — luck/BABIP/HR regression.

breakout_score = P(breakout) × room_to_grow
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.queries import get_pitcher_breakout_features

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(
    "C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard"
)

# ---------------------------------------------------------------------------
# Feature columns (T1 + T2 + T3)
# ---------------------------------------------------------------------------

PITCHER_FEATURES = [
    # T1 — boxscore (2000+)
    "k_pct", "bb_pct", "hr_bf", "era", "fip", "whip",
    "k_per_9", "bb_per_9", "go_ao", "age",
    "delta_k_pct", "delta_bb_pct", "delta_era",
    "mlb_seasons", "is_starter",
    # T2 — Statcast (2018+)
    "whiff_rate", "swstr_pct", "csw_pct", "avg_velo",
    "xwoba_against", "barrel_pct_against", "zone_pct",
    "chase_pct", "hr_per_fb", "era_minus_xfip",
    # T3 — MiLB (young pitchers)
    "milb_translated_k_pct", "milb_translated_bb_pct",
    "milb_translated_hr_bf", "milb_age_relative",
]


def _load_config() -> dict:
    """Load breakout config from model.yaml."""
    cfg_path = Path("config/model.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("breakout", {})
    return {}


def _build_folds(cfg: dict) -> list[tuple[int, int]]:
    """Build training fold list from config, skipping COVID seasons."""
    start = cfg.get("folds_start", 2001)
    end = cfg.get("folds_end", 2025)
    skip_covid = cfg.get("skip_covid_deltas", True)

    folds = []
    for y in range(start, end):
        if skip_covid and y in (2019, 2020):
            continue
        folds.append((y, y + 1))
    return folds


def _pitcher_outcome_loader(season: int, min_bf: int) -> pd.DataFrame:
    """Load pitcher outcome data for a season (ERA for breakout labeling)."""
    return get_pitcher_breakout_features(season, min_bf)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_pitcher_breakout_candidates(
    season: int = 2025,
    min_bf: int = 200,
) -> pd.DataFrame:
    """Score all qualified pitchers for breakout potential.

    Uses XGBoost trained on 22+ walk-forward folds (2001-2025) with
    tiered features and SHAP-based archetype assignment.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_bf : int
        Minimum batters faced to qualify.

    Returns
    -------
    pd.DataFrame
        One row per pitcher with breakout scores, archetype assignment,
        hole identification, and narratives.
    """
    from src.models.breakout_engine import (
        BreakoutConfig,
        build_walk_forward_data,
        score_players,
        train_breakout_model,
    )

    cfg = _load_config()
    pitcher_cfg = cfg.get("pitcher", {})
    xgb_cfg = cfg.get("xgb", {})

    config = BreakoutConfig(
        player_type="pitcher",
        id_col="pitcher_id",
        name_col="pitcher_name",
        feature_cols=PITCHER_FEATURES,
        breakout_metric="era",
        breakout_direction="drop",
        breakout_threshold=pitcher_cfg.get("era_drop_threshold", 0.75),
        outcome_bound=pitcher_cfg.get("era_outcome_ceiling", 4.00),
        outcome_is_floor=False,
        n_archetypes=pitcher_cfg.get("n_archetypes", 3),
        age_window=tuple(pitcher_cfg.get("age_window", [23, 27])),
        xgb_params=xgb_cfg,
    )

    folds = _build_folds(cfg)

    # --- 1. Build training data ---
    logger.info("Building pitcher breakout training data (%d folds)...", len(folds))
    X_train, y_train, meta = build_walk_forward_data(
        config,
        feature_loader=get_pitcher_breakout_features,
        outcome_loader=_pitcher_outcome_loader,
        folds=folds,
    )

    if X_train.empty:
        logger.warning("No training data for pitcher breakout model")
        return pd.DataFrame()

    # --- 2. Train XGBoost ---
    logger.info("Training pitcher breakout XGBoost model...")
    model_bundle = train_breakout_model(config, X_train, y_train)

    if model_bundle.get("model") is None:
        logger.warning("Model training failed")
        return pd.DataFrame()

    # --- 3. Load and score current season ---
    logger.info("Loading pitcher features for season %d...", season)
    df = get_pitcher_breakout_features(season, min_bf)
    if df.empty:
        logger.warning("No qualified pitchers for breakout scoring")
        return pd.DataFrame()

    # is_starter from projections parquet if not already present
    if "is_starter" not in df.columns:
        proj_path = DASHBOARD_DIR / "pitcher_projections.parquet"
        if proj_path.exists():
            try:
                proj = pd.read_parquet(proj_path)[["pitcher_id", "is_starter"]]
                df = df.merge(proj, on="pitcher_id", how="left")
            except Exception:
                pass
        if "is_starter" not in df.columns:
            df["is_starter"] = np.nan

    # --- 4. Score ---
    df = score_players(model_bundle, df, config)

    # --- 5. Output ---
    df = df.sort_values("breakout_rank")

    # Backward compat column names
    if "delta_k_pct" in df.columns and "delta_k_rate" not in df.columns:
        df["delta_k_rate"] = df["delta_k_pct"]
    if "delta_bb_pct" in df.columns and "delta_bb_rate" not in df.columns:
        df["delta_bb_rate"] = df["delta_bb_pct"]

    output_cols = [
        "pitcher_id", "pitcher_name", "age", "is_starter",
        # Scores
        "breakout_prob", "breakout_type",
        "prob_stuff_dominant", "prob_command_leap", "prob_era_correction",
        "room_to_grow", "breakout_score",
        "breakout_rank", "archetype_rank", "breakout_tier",
        "breakout_hole", "breakout_narrative",
        # Key stats (dashboard expects these for archetype cards)
        "bf", "k_pct", "bb_pct",
        "whiff_rate", "swstr_pct", "csw_pct", "avg_velo",
        "contact_pct", "zone_pct", "chase_pct",
        "xwoba_against", "barrel_pct_against",
        "era", "fip", "hr_per_fb", "era_minus_xfip",
        "first_strike_pct", "putaway_rate",
        # Trajectory
        "delta_k_rate", "delta_bb_rate",
    ]
    available = [c for c in output_cols if c in df.columns]
    result = df[available].reset_index(drop=True)

    n_candidate = (result["breakout_tier"] == "Breakout Candidate").sum()
    n_radar = (result["breakout_tier"] == "On the Radar").sum()
    logger.info(
        "Pitcher breakout: %d Candidates, %d On the Radar (of %d total)",
        n_candidate, n_radar, len(result),
    )
    return result
