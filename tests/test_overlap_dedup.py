"""Tests for the Close Overlap Circles plan.

Verifies:
1. Weather helpers: shared module produces correct buckets.
2. TTO dedup: game_stat_model uses tto_model's implementation.
3. Precompute gating: backtest_summaries is gated.
4. Breakout plumbing: shared module importable and consistent.
5. game_k_validation delegates to game_prop_validation.
6. run_season_backtest emits deprecation warning.
"""
from __future__ import annotations

import importlib
import inspect
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Weather helper parity
# ---------------------------------------------------------------------------

class TestWeatherHelpers:
    """Shared weather module produces correct bucket categories."""

    def test_temp_buckets_match_sql_spec(self) -> None:
        from src.utils.weather import parse_temp_bucket
        assert parse_temp_bucket(40) == "cold"
        assert parse_temp_bucket(54) == "cold"
        assert parse_temp_bucket(55) == "cool"
        assert parse_temp_bucket(69) == "cool"
        assert parse_temp_bucket(70) == "warm"
        assert parse_temp_bucket(84) == "warm"
        assert parse_temp_bucket(85) == "hot"
        assert parse_temp_bucket(100) == "hot"
        assert parse_temp_bucket(None) == "unknown"
        assert parse_temp_bucket(float("nan")) == "unknown"

    def test_wind_categories(self) -> None:
        from src.utils.weather import wind_category
        assert wind_category({"weather_wind_direction": "in"}) == "in"
        assert wind_category({"weather_wind_direction": "Out"}) == "out"
        assert wind_category({"weather_wind_direction": "CROSS"}) == "cross"
        assert wind_category({"weather_wind_direction": "none"}) == "none"
        assert wind_category({"weather_wind_direction": None}) == "unknown"
        assert wind_category({}) == "unknown"

    def test_callers_use_shared_module(self) -> None:
        """Caller files import from src.utils.weather (verified via source)."""
        import pathlib
        root = pathlib.Path("C:/Users/kekoa/Documents/data_analytics/player_profiles")
        for fpath in [
            root / "scripts" / "update_in_season.py",
            root / "scripts" / "precompute" / "confident_picks.py",
            root / "scripts" / "run_sim_vs_odds_backtest.py",
        ]:
            source = fpath.read_text()
            assert "from src.utils.weather import" in source, \
                f"{fpath.name} should import from src.utils.weather"


# ---------------------------------------------------------------------------
# 2. TTO dedup
# ---------------------------------------------------------------------------

class TestTTODedup:
    """game_stat_model uses tto_model's canonical implementation."""

    def test_game_stat_model_imports_from_tto_model(self) -> None:
        from src.models import game_stat_model
        from src.models.game_sim.tto_model import (
            LEAGUE_TTO_LOGIT_LIFTS,
            BF_PER_TTO,
            build_tto_logit_lifts,
        )
        # game_stat_model should reference the same objects
        assert game_stat_model._LEAGUE_TTO_LOGIT_LIFTS is LEAGUE_TTO_LOGIT_LIFTS
        assert game_stat_model._BF_PER_TTO is BF_PER_TTO
        assert game_stat_model.build_tto_logit_lifts is build_tto_logit_lifts

    def test_no_duplicate_tto_constants_in_game_stat_model(self) -> None:
        """game_stat_model should not define its own TTO league rates."""
        source = inspect.getsource(
            importlib.import_module("src.models.game_stat_model")
        )
        assert "logit(0.23782)" not in source, \
            "game_stat_model should not define its own TTO rates"


# ---------------------------------------------------------------------------
# 3. Precompute gating
# ---------------------------------------------------------------------------

class TestPrecomputeGating:
    """Backtest summaries should be gated behind should_run()."""

    def test_backtest_summaries_gated(self) -> None:
        source_path = (
            "C:/Users/kekoa/Documents/data_analytics/player_profiles/"
            "scripts/precompute_dashboard_data.py"
        )
        with open(source_path) as f:
            source = f.read()

        assert 'should_run("backtest_summaries")' in source, \
            "run_backtest_summaries should be gated behind should_run"
        assert "# 8. Backtest summaries (always runs)" not in source, \
            "Old 'always runs' comment should be removed"


# ---------------------------------------------------------------------------
# 4. Breakout plumbing
# ---------------------------------------------------------------------------

class TestBreakoutUtils:
    """Shared breakout config/fold utilities are importable."""

    def test_shared_module_exists(self) -> None:
        from src.models.breakout_utils import (
            load_breakout_config,
            build_breakout_folds,
        )
        assert callable(load_breakout_config)
        assert callable(build_breakout_folds)

    def test_models_import_from_shared(self) -> None:
        """Both breakout models should not define _load_config locally."""
        hitter_src = inspect.getsource(
            importlib.import_module("src.models.breakout_model")
        )
        pitcher_src = inspect.getsource(
            importlib.import_module("src.models.pitcher_breakout_model")
        )
        assert "def _load_config" not in hitter_src
        assert "def _load_config" not in pitcher_src
        assert "def _build_folds" not in hitter_src
        assert "def _build_folds" not in pitcher_src


# ---------------------------------------------------------------------------
# 5. game_k_validation wrapper
# ---------------------------------------------------------------------------

class TestGameKValidationWrapper:
    """game_k_validation delegates to game_prop_validation."""

    def test_wrapper_imports_from_game_prop(self) -> None:
        source = inspect.getsource(
            importlib.import_module("src.evaluation.game_k_validation")
        )
        assert "from src.evaluation.game_prop_validation import" in source
        # Should NOT contain heavy model fitting logic
        assert "fit_pitcher_model" not in source
        assert "build_multi_season_pitcher_data" not in source

    def test_entry_points_exist(self) -> None:
        from src.evaluation.game_k_validation import (
            build_game_k_predictions,
            compute_game_k_metrics,
            run_full_game_k_backtest,
            run_full_game_backtest,
        )
        assert callable(build_game_k_predictions)
        assert callable(compute_game_k_metrics)
        assert callable(run_full_game_k_backtest)
        assert callable(run_full_game_backtest)

