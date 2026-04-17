"""
Shared path helpers.

Resolves external paths (sibling tdd-dashboard repo) via environment variables
with sensible defaults relative to this repo's location. Prefer these helpers
over hardcoded absolute paths so the code is portable across machines.

Environment variables:
    TDD_DASHBOARD_REPO  -- absolute path to the sibling tdd-dashboard repo.
                           Defaults to ``<repo_parent>/tdd-dashboard``.
    TDD_DASHBOARD_DIR   -- absolute path to tdd-dashboard/data/dashboard.
                           Defaults to ``<TDD_DASHBOARD_REPO>/data/dashboard``.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cached"
CONFIG_DIR = PROJECT_ROOT / "config"


def get_clustering_seasons() -> list[int]:
    """Read ``seasons.clustering`` from ``config/model.yaml``."""
    path = CONFIG_DIR / "model.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["seasons"]["clustering"]


def dashboard_repo() -> Path:
    """Root of the sibling tdd-dashboard repo."""
    env_val = os.getenv("TDD_DASHBOARD_REPO")
    if env_val:
        return Path(env_val)
    return PROJECT_ROOT.parent / "tdd-dashboard"


def dashboard_dir() -> Path:
    """tdd-dashboard/data/dashboard precomputed artifact directory."""
    env_val = os.getenv("TDD_DASHBOARD_DIR")
    if env_val:
        return Path(env_val)
    return dashboard_repo() / "data" / "dashboard"
