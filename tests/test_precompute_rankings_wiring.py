"""Tests for the daily-standouts / weekly-form precompute wiring.

These verify that the precompute entrypoints
(``scripts/precompute/rankings.py``) call the right routines, register the
right groups in ``scripts/precompute_dashboard_data.py``, and produce
schema-safe output parquets even when the data source is empty.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _load_precompute_script_module():
    """Load scripts/precompute_dashboard_data.py as a module."""
    path = _SCRIPTS_DIR / "precompute_dashboard_data.py"
    spec = importlib.util.spec_from_file_location("precompute_dashboard_data", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_precompute_script_exposes_daily_and_weekly_groups():
    mod = _load_precompute_script_module()
    assert "daily_standouts" in mod._SECTION_GROUPS
    assert "weekly_form" in mod._SECTION_GROUPS
    assert "daily_standouts" in mod._SECTION_GROUPS["daily_standouts"]
    assert "weekly_form" in mod._SECTION_GROUPS["weekly_form"]


def test_run_daily_standouts_writes_schema_safe_empty_parquets(tmp_path, monkeypatch):
    """If no rows exist, both hitter and pitcher parquets are still written."""
    from precompute import rankings as rk
    from src.models import daily_standouts as ds

    import precompute as pkg
    monkeypatch.setattr(pkg, "DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(rk, "DASHBOARD_DIR", tmp_path)
    # Force the underlying SQL-hitting helpers to return empty frames so the
    # test doesn't need a live database.
    monkeypatch.setattr(ds, "get_hitter_daily_standouts", lambda game_date=None: pd.DataFrame())
    monkeypatch.setattr(ds, "get_pitcher_daily_standouts", lambda game_date=None: pd.DataFrame())

    rk.run_daily_standouts()

    h_path = tmp_path / "hitters_daily_standouts.parquet"
    p_path = tmp_path / "pitchers_daily_standouts.parquet"
    assert h_path.exists()
    assert p_path.exists()

    h = pd.read_parquet(h_path)
    p = pd.read_parquet(p_path)
    assert h.empty and p.empty

    hitter_required = {
        "batter_id", "batter_name", "daily_standout_score",
        "hits", "hr", "rbi", "ops", "woba",
    }
    pitcher_required = {
        "pitcher_id", "pitcher_name", "role", "daily_standout_score",
        "k", "era", "whip",
    }
    assert hitter_required.issubset(set(h.columns))
    assert pitcher_required.issubset(set(p.columns))


def test_run_weekly_form_writes_schema_safe_empty_parquets(tmp_path, monkeypatch):
    """If no rows exist, both hitter and pitcher parquets are still written."""
    from precompute import rankings as rk
    from src.models import weekly_form as wf

    import precompute as pkg
    monkeypatch.setattr(pkg, "DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(rk, "DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(wf, "get_hitter_recent_form",
                        lambda days=14, as_of_date=None: pd.DataFrame())
    monkeypatch.setattr(wf, "get_pitcher_recent_form",
                        lambda days=14, as_of_date=None: pd.DataFrame())

    rk.run_weekly_form()

    h_path = tmp_path / "hitters_weekly_form.parquet"
    p_path = tmp_path / "pitchers_weekly_form.parquet"
    assert h_path.exists()
    assert p_path.exists()

    h = pd.read_parquet(h_path)
    p = pd.read_parquet(p_path)
    assert h.empty and p.empty

    hitter_required = {
        "batter_id", "batter_name", "weekly_form_score",
        "hits_14d", "hr_14d", "woba_14d", "xwoba_14d", "k_rate_14d",
    }
    pitcher_required = {
        "pitcher_id", "pitcher_name", "role_14d", "weekly_form_score",
        "k_minus_bb_14d", "era_14d", "whip_14d",
    }
    assert hitter_required.issubset(set(h.columns))
    assert pitcher_required.issubset(set(p.columns))
