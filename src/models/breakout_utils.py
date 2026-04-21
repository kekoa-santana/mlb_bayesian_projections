"""Shared utilities for hitter and pitcher breakout models."""
from __future__ import annotations

import yaml


def load_breakout_config() -> dict:
    """Load breakout config from model.yaml."""
    from src.data.paths import CONFIG_DIR

    cfg_path = CONFIG_DIR / "model.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("breakout", {})
    return {}


def build_breakout_folds(cfg: dict) -> list[tuple[int, int]]:
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
