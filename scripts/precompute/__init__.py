"""Precompute package -- shared configuration for all precompute modules."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_REPO = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard")
DASHBOARD_DIR = DASHBOARD_REPO / "data" / "dashboard"

SEASONS = list(range(2018, 2026))
FROM_SEASON = 2025

logger = logging.getLogger("precompute")


def precache_profiles(from_season: int = FROM_SEASON) -> None:
    """Pre-cache observed profiles needed by projection enrichment."""
    from src.data.feature_eng import (
        get_cached_hitter_observed_profile,
        get_cached_pitcher_observed_profile,
        get_cached_sprint_speed,
    )

    logger.info("=" * 60)
    logger.info("Pre-caching observed profiles for %d...", from_season)
    try:
        obs_h = get_cached_hitter_observed_profile(from_season)
        logger.info("Hitter observed profile: %d rows", len(obs_h))
    except Exception as e:
        logger.warning("Could not cache hitter observed profile: %s", e)

    try:
        sprint = get_cached_sprint_speed(from_season)
        logger.info("Sprint speed: %d rows", len(sprint))
    except Exception as e:
        logger.warning("Could not cache sprint speed: %s", e)

    try:
        obs_p = get_cached_pitcher_observed_profile(from_season)
        logger.info("Pitcher observed profile: %d rows", len(obs_p))
    except Exception as e:
        logger.warning("Could not cache pitcher observed profile: %s", e)


def load_calibration_t() -> tuple[dict, dict]:
    """Load posterior calibration temperatures from config/model.yaml."""
    import yaml

    with open(PROJECT_ROOT / "config" / "model.yaml") as f:
        _cfg = yaml.safe_load(f)
    _cal = _cfg.get("calibration", {})
    hitter_cal_t = _cal.get("hitter", {})
    pitcher_cal_t = _cal.get("pitcher", {})
    return hitter_cal_t, pitcher_cal_t
