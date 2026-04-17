"""Hitter breakout model — XGBoost with SHAP archetypes.

Replaces the GMM-based approach with XGBoost trained on 22+ walk-forward
folds (2001-2025), leveraging tiered features:

- **T1 (2000+):** boxscore rates (K%, BB%, ISO, BABIP, OPS, etc.)
- **T2 (2018+):** Statcast advanced (barrel%, xwOBA, chase, EV, etc.)
- **T3 (young players):** translated MiLB rates, prospect pedigree

Archetypes (k=2) discovered via SHAP-value clustering:

1. **Power Surge**: Tools are there — results catch up to underlying quality.
2. **Diamond in the Rough**: Low production, but fixable weaknesses + upside.

breakout_score = P(breakout) × room_to_grow
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.paths import dashboard_dir
from src.data.queries import get_hitter_breakout_features

logger = logging.getLogger(__name__)

DASHBOARD_DIR = dashboard_dir()

# ---------------------------------------------------------------------------
# Feature columns (T1 + T2 + T3)
# ---------------------------------------------------------------------------

HITTER_FEATURES = [
    # T1 — boxscore (2000+)
    "k_pct", "bb_pct", "iso", "hr_ab", "babip", "slg", "obp", "ops",
    "go_ao", "age", "delta_k_pct", "delta_bb_pct", "delta_iso", "mlb_seasons",
    # T2 — Statcast (2018+)
    "barrel_pct", "hard_hit_pct", "xwoba", "whiff_rate", "chase_rate",
    "z_contact_pct", "avg_exit_velo", "sprint_speed", "oaa", "woba",
    "xwoba_minus_woba",
    # T3 — MiLB (young players)
    "milb_translated_k_pct", "milb_translated_bb_pct", "milb_translated_iso",
    "milb_age_relative", "fg_future_value",
]


def _load_config() -> dict:
    """Load breakout config from model.yaml."""
    from src.data.paths import CONFIG_DIR

    cfg_path = CONFIG_DIR / "model.yaml"
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


def _hitter_outcome_loader(season: int, min_pa: int) -> pd.DataFrame:
    """Load hitter outcome data for a season (est_wOBA for breakout labeling)."""
    return get_hitter_breakout_features(season, min_pa)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_breakout_candidates(
    season: int = 2025,
    min_pa: int = 200,
) -> pd.DataFrame:
    """Score all qualified hitters for breakout potential.

    Uses XGBoost trained on 22+ walk-forward folds (2001-2025) with
    tiered features and SHAP-based archetype assignment.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_pa : int
        Minimum PA to qualify.

    Returns
    -------
    pd.DataFrame
        One row per qualified hitter with breakout scores, archetype
        assignment, hole identification, and narratives.
    """
    from src.models.breakout_engine import (
        BreakoutConfig,
        build_walk_forward_data,
        score_players,
        train_breakout_model,
    )

    cfg = _load_config()
    hitter_cfg = cfg.get("hitter", {})
    xgb_cfg = cfg.get("xgb", {})

    config = BreakoutConfig(
        player_type="hitter",
        id_col="batter_id",
        name_col="batter_name",
        feature_cols=HITTER_FEATURES,
        breakout_metric="est_woba",
        breakout_direction="gain",
        breakout_threshold=hitter_cfg.get("woba_gain_threshold", 0.020),
        outcome_bound=hitter_cfg.get("woba_outcome_floor", 0.300),
        outcome_is_floor=True,
        n_archetypes=hitter_cfg.get("n_archetypes", 2),
        age_window=tuple(hitter_cfg.get("age_window", [24, 27])),
        xgb_params=xgb_cfg,
    )

    folds = _build_folds(cfg)

    # --- 1. Build training data ---
    logger.info("Building hitter breakout training data (%d folds)...", len(folds))
    X_train, y_train, meta = build_walk_forward_data(
        config,
        feature_loader=get_hitter_breakout_features,
        outcome_loader=_hitter_outcome_loader,
        folds=folds,
    )

    if X_train.empty:
        logger.warning("No training data for hitter breakout model")
        return pd.DataFrame()

    # --- 2. Train XGBoost ---
    logger.info("Training hitter breakout XGBoost model...")
    model_bundle = train_breakout_model(config, X_train, y_train)

    if model_bundle.get("model") is None:
        logger.warning("Model training failed")
        return pd.DataFrame()

    # --- 3. Load and score current season ---
    logger.info("Loading hitter features for season %d...", season)
    df = get_hitter_breakout_features(season, min_pa)
    if df.empty:
        logger.warning("No qualified hitters for breakout scoring")
        return pd.DataFrame()

    # Position assignment
    try:
        from src.models.player_rankings import _assign_hitter_positions
        positions = _assign_hitter_positions(season=season)
        df = df.merge(
            positions.rename(columns={"player_id": "batter_id"}),
            on="batter_id", how="left",
        )
    except Exception:
        pass
    if "position" not in df.columns:
        df["position"] = ""

    # --- 4. Score ---
    df = score_players(model_bundle, df, config)

    # --- 5. Output ---
    df = df.sort_values("breakout_rank")

    # Backward compat column names
    if "batter_stand" not in df.columns:
        df["batter_stand"] = ""

    # Map delta column names for compat
    if "delta_k_pct" in df.columns and "delta_k_rate" not in df.columns:
        df["delta_k_rate"] = df["delta_k_pct"]
    if "delta_bb_pct" in df.columns and "delta_bb_rate" not in df.columns:
        df["delta_bb_rate"] = df["delta_bb_pct"]

    output_cols = [
        "batter_id", "batter_name", "age", "position", "batter_stand",
        # Scores
        "breakout_prob", "breakout_type",
        "prob_power_surge", "prob_diamond_in_the_rough",
        "room_to_grow", "breakout_score",
        "breakout_rank", "archetype_rank", "breakout_tier",
        "breakout_hole", "breakout_narrative",
        # Key stats (dashboard expects these for archetype cards)
        "pa", "woba", "xwoba", "wrc_plus", "est_woba",
        "barrel_pct", "z_contact_pct", "pull_pct",
        "hard_hit_pct", "avg_exit_velo", "sprint_speed", "oaa",
        "k_pct", "bb_pct", "chase_rate", "iso", "ops", "babip",
        "whiff_rate",
        # Trajectory
        "delta_k_rate", "delta_bb_rate",
    ]
    available = [c for c in output_cols if c in df.columns]
    result = df[available].reset_index(drop=True)

    n_candidate = (result["breakout_tier"] == "Breakout Candidate").sum()
    n_radar = (result["breakout_tier"] == "On the Radar").sum()
    logger.info(
        "Hitter breakout: %d Candidates, %d On the Radar (of %d total)",
        n_candidate, n_radar, len(result),
    )
    return result
