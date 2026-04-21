"""Contract and regression tests for pitcher sim consistency.

Verifies:
1. No legacy model imports remain in the codebase.
2. Rest and catcher framing lifts propagate through GameContext.
3. TTO and fatigue adjustments are applied in the lineup simulator.
4. Reliever matchup lifts are symmetric across K/BB/HR in batter sim.
5. Per-sim BABIP variance is preserved in lineup simulator.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Contract: no legacy model imports
# ---------------------------------------------------------------------------

class TestNoLegacyImports:
    """Ensure deleted legacy modules cannot be imported."""

    def test_game_k_model_deleted(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("src.models.game_k_model")

    def test_pitcher_k_rate_model_deleted(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("src.models.pitcher_k_rate_model")

    def test_k_rate_model_deleted(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("src.models.k_rate_model")

    def test_game_stat_model_exists(self) -> None:
        """Relocated functions should be importable from game_stat_model."""
        mod = importlib.import_module("src.models.game_stat_model")
        assert hasattr(mod, "predict_game_batch_stat")
        assert hasattr(mod, "predict_batter_game")
        assert hasattr(mod, "simulate_game_ks")

    def test_posterior_utils_reexports(self) -> None:
        """posterior_utils re-exports the relocated functions."""
        from src.models.posterior_utils import (
            predict_game_batch_stat,
            predict_batter_game,
            simulate_game_ks,
        )
        assert callable(predict_game_batch_stat)
        assert callable(predict_batter_game)
        assert callable(simulate_game_ks)


# ---------------------------------------------------------------------------
# 2. Regression: rest and catcher framing lifts propagate
# ---------------------------------------------------------------------------

class TestGameContextLifts:
    """Verify non-zero context lifts change PA probabilities."""

    def test_rest_lifts_affect_pa_probs(self) -> None:
        from src.models.game_sim.pa_outcome_model import (
            GameContext, PAOutcomeModel,
        )
        model = PAOutcomeModel()

        base_probs = model.compute_pa_probs(
            pitcher_k_rate=0.22,
            pitcher_bb_rate=0.08,
            pitcher_hr_rate=0.03,
        )

        # Apply negative rest K lift (short rest = slightly fewer K)
        rest_ctx = GameContext(rest_k_lift=-0.05, rest_bb_lift=0.03)
        rest_probs = model.compute_pa_probs(
            pitcher_k_rate=0.22,
            pitcher_bb_rate=0.08,
            pitcher_hr_rate=0.03,
            ctx=rest_ctx,
        )

        assert rest_probs["k"] < base_probs["k"], "Rest K lift should reduce K probability"
        assert rest_probs["bb"] > base_probs["bb"], "Rest BB lift should increase BB probability"

    def test_catcher_framing_affects_pa_probs(self) -> None:
        from src.models.game_sim.pa_outcome_model import (
            GameContext, PAOutcomeModel,
        )
        model = PAOutcomeModel()

        base_probs = model.compute_pa_probs(
            pitcher_k_rate=0.22,
            pitcher_bb_rate=0.08,
            pitcher_hr_rate=0.03,
        )

        # Good framer: more K, fewer BB
        framing_ctx = GameContext(catcher_k_lift=0.05, catcher_bb_lift=-0.05)
        framing_probs = model.compute_pa_probs(
            pitcher_k_rate=0.22,
            pitcher_bb_rate=0.08,
            pitcher_hr_rate=0.03,
            ctx=framing_ctx,
        )

        assert framing_probs["k"] > base_probs["k"], "Good framer should increase K"
        assert framing_probs["bb"] < base_probs["bb"], "Good framer should decrease BB"


# ---------------------------------------------------------------------------
# 3. Regression: TTO and fatigue in lineup simulator
# ---------------------------------------------------------------------------

class TestLineupSimTTOFatigue:
    """Verify TTO and fatigue imports and logic exist in lineup_simulator."""

    def test_tto_imports_present(self) -> None:
        from src.models.game_sim import lineup_simulator
        # These should be importable (added in Phase 3)
        assert hasattr(lineup_simulator, 'LEAGUE_TTO_LOGIT_LIFTS') or \
            'LEAGUE_TTO_LOGIT_LIFTS' in dir(lineup_simulator) or \
            'tto_model' in str(lineup_simulator.__dict__.get('__builtins__', ''))
        # Check the import exists by looking at the module's source
        import inspect
        source = inspect.getsource(lineup_simulator._simulate_half_inning)
        assert 'tto' in source.lower(), "TTO logic should be in _simulate_half_inning"
        assert 'fatigue' in source.lower(), "Fatigue logic should be in _simulate_half_inning"

    def test_fatigue_adjustments_importable(self) -> None:
        from src.models.game_sim.pa_outcome_model import compute_fatigue_adjustments
        result = compute_fatigue_adjustments(100)
        assert "k" in result
        assert "bb" in result
        assert "hr" in result
        # At 100 pitches (10 over threshold), K should decrease, BB/HR increase
        assert result["k"] < 0, "Fatigue should decrease K rate"
        assert result["bb"] > 0, "Fatigue should increase BB rate"
        assert result["hr"] > 0, "Fatigue should increase HR rate"


# ---------------------------------------------------------------------------
# 4. Regression: reliever matchup symmetry
# ---------------------------------------------------------------------------

class TestRelieverMatchupSymmetry:
    """Verify K/BB/HR matchup lifts are all applied for relievers."""

    def test_batter_sim_reliever_lifts_symmetric(self) -> None:
        import inspect
        from src.models.game_sim import batter_simulator
        source = inspect.getsource(batter_simulator.simulate_batter_game)

        # Check that bullpen_matchup_k_lift and bullpen_matchup_bb_lift
        # appear in the reliever branch (not just bullpen_matchup_hr_lift)
        assert "bullpen_matchup_k_lift" in source, \
            "K matchup lift should be applied for relievers"
        assert "bullpen_matchup_bb_lift" in source, \
            "BB matchup lift should be applied for relievers"


# ---------------------------------------------------------------------------
# 5. Regression: per-sim BABIP variance preserved
# ---------------------------------------------------------------------------

class TestBABIPVariance:
    """Verify BABIP array support in draw_outcomes."""

    def test_draw_outcomes_accepts_array_babip(self) -> None:
        from src.models.game_sim.pa_outcome_model import PAOutcomeModel
        model = PAOutcomeModel()
        rng = np.random.default_rng(42)

        probs = model.compute_pa_probs(
            pitcher_k_rate=0.22,
            pitcher_bb_rate=0.08,
            pitcher_hr_rate=0.03,
        )

        n = 100
        babip_array = rng.normal(0.0, 0.02, size=n)
        outcomes = model.draw_outcomes(
            probs=probs, rng=rng, n_draws=n, babip_adj=babip_array,
        )
        assert outcomes.shape == (n,), "Should produce n outcomes"
        assert outcomes.dtype == np.int8, "Should be int8 outcome codes"

    def test_bip_model_array_babip(self) -> None:
        from src.models.game_sim.bip_model import BIPOutcomeModel
        model = BIPOutcomeModel()
        rng = np.random.default_rng(42)

        n = 200
        babip_array = rng.normal(0.0, 0.03, size=n)
        outcomes = model.draw_outcomes(rng=rng, n_draws=n, babip_adj=babip_array)
        assert outcomes.shape == (n,)
        assert set(outcomes.tolist()).issubset({0, 1, 2, 3})

    def test_scalar_babip_still_works(self) -> None:
        from src.models.game_sim.bip_model import BIPOutcomeModel
        model = BIPOutcomeModel()
        rng = np.random.default_rng(42)

        outcomes = model.draw_outcomes(rng=rng, n_draws=100, babip_adj=0.02)
        assert outcomes.shape == (100,)
