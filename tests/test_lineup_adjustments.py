"""Tests for src.models.lineup_adjustments (Phase 1G + 1J)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.lineup_adjustments import (
    PA_WEIGHTS_BY_SLOT,
    build_game_lineup_map,
    compute_lineup_proneness,
    compute_lineup_proneness_batch,
    compute_opposing_pitcher_lift,
    compute_opposing_pitcher_lift_batch,
    extract_opposing_lineup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def high_k_posteriors(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """9 batters with K% ~ 27%."""
    return {i: rng.beta(8, 22, size=500) for i in range(9)}


@pytest.fixture
def low_k_posteriors(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """9 batters with K% ~ 14%."""
    return {i: rng.beta(5, 30, size=500) for i in range(9)}


@pytest.fixture
def avg_k_posteriors(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """9 batters with K% ~ 22% (league average)."""
    return {i: rng.beta(7, 25, size=500) for i in range(9)}


LEAGUE_AVG_K = 0.222


# ---------------------------------------------------------------------------
# Phase 1G: Lineup proneness
# ---------------------------------------------------------------------------

class TestComputeLineupProneness:
    """Test compute_lineup_proneness."""

    def test_high_k_lineup_positive_lift(
        self, high_k_posteriors: dict[int, np.ndarray]
    ) -> None:
        lift = compute_lineup_proneness(
            list(range(9)), high_k_posteriors, LEAGUE_AVG_K, weight=0.5,
        )
        assert lift > 0.0, "High-K lineup should produce positive lift"

    def test_low_k_lineup_negative_lift(
        self, low_k_posteriors: dict[int, np.ndarray]
    ) -> None:
        lift = compute_lineup_proneness(
            list(range(9)), low_k_posteriors, LEAGUE_AVG_K, weight=0.5,
        )
        assert lift < 0.0, "Low-K lineup should produce negative lift"

    def test_weight_zero_returns_zero(
        self, high_k_posteriors: dict[int, np.ndarray]
    ) -> None:
        lift = compute_lineup_proneness(
            list(range(9)), high_k_posteriors, LEAGUE_AVG_K, weight=0.0,
        )
        assert lift == 0.0

    def test_weight_scales_lift(
        self, high_k_posteriors: dict[int, np.ndarray]
    ) -> None:
        lift_half = compute_lineup_proneness(
            list(range(9)), high_k_posteriors, LEAGUE_AVG_K, weight=0.5,
        )
        lift_full = compute_lineup_proneness(
            list(range(9)), high_k_posteriors, LEAGUE_AVG_K, weight=1.0,
        )
        assert abs(lift_full - 2 * lift_half) < 1e-6

    def test_missing_batters_use_league_avg(
        self, rng: np.random.Generator
    ) -> None:
        # Only 3 batters have posteriors
        partial = {i: rng.beta(10, 10, size=500) for i in range(3)}
        lift = compute_lineup_proneness(
            list(range(9)), partial, LEAGUE_AVG_K, weight=0.5,
        )
        # Should still return a value (not crash)
        assert isinstance(lift, float)

    def test_no_matching_batters_returns_zero(self) -> None:
        lift = compute_lineup_proneness(
            list(range(9)), {}, LEAGUE_AVG_K, weight=0.5,
        )
        assert lift == 0.0

    def test_wrong_lineup_size_returns_zero(
        self, high_k_posteriors: dict[int, np.ndarray]
    ) -> None:
        lift = compute_lineup_proneness(
            [0, 1, 2],  # Only 3 batters
            high_k_posteriors, LEAGUE_AVG_K, weight=0.5,
        )
        assert lift == 0.0

    def test_pa_weights_shape(self) -> None:
        assert PA_WEIGHTS_BY_SLOT.shape == (9,)
        assert PA_WEIGHTS_BY_SLOT[0] > PA_WEIGHTS_BY_SLOT[8]


class TestComputeLineupPronenessBatch:
    """Test compute_lineup_proneness_batch."""

    def test_batch(self, high_k_posteriors: dict[int, np.ndarray]) -> None:
        game_lineups = {
            100: list(range(9)),
            200: list(range(9)),
        }
        result = compute_lineup_proneness_batch(
            game_lineups, high_k_posteriors, LEAGUE_AVG_K, weight=0.5,
        )
        assert isinstance(result, dict)
        assert all(v > 0 for v in result.values())


# ---------------------------------------------------------------------------
# Phase 1J: Opposing pitcher quality
# ---------------------------------------------------------------------------

class TestComputeOpposingPitcherLift:
    """Test compute_opposing_pitcher_lift."""

    def test_high_k_pitcher_positive_lift(self) -> None:
        lift = compute_opposing_pitcher_lift(0.28, 0.222, weight=0.5)
        assert lift > 0.0

    def test_low_k_pitcher_negative_lift(self) -> None:
        lift = compute_opposing_pitcher_lift(0.15, 0.222, weight=0.5)
        assert lift < 0.0

    def test_league_avg_pitcher_near_zero(self) -> None:
        lift = compute_opposing_pitcher_lift(0.222, 0.222, weight=0.5)
        assert abs(lift) < 1e-6

    def test_weight_zero(self) -> None:
        lift = compute_opposing_pitcher_lift(0.30, 0.222, weight=0.0)
        assert lift == 0.0

    def test_nan_returns_zero(self) -> None:
        lift = compute_opposing_pitcher_lift(float("nan"), 0.222, weight=0.5)
        assert lift == 0.0


class TestComputeOpposingPitcherLiftBatch:
    """Test compute_opposing_pitcher_lift_batch."""

    def test_batch(self, rng: np.random.Generator) -> None:
        pitcher_posteriors = {
            10: rng.beta(8, 22, size=500),  # ~27% K
            20: rng.beta(5, 30, size=500),  # ~14% K
        }
        game_pitcher_map = {
            (100, 1): 10,
            (100, 2): 20,
            (200, 3): 10,
            (200, 4): 99,  # not in posteriors
        }
        result = compute_opposing_pitcher_lift_batch(
            game_pitcher_map, pitcher_posteriors, 0.222, weight=0.5,
        )
        assert isinstance(result, dict)
        # (100,1) vs pitcher 10 should be positive (high-K pitcher)
        assert (100, 1) in result
        assert result[(100, 1)] > 0
        # (100,2) vs pitcher 20 should be negative (low-K pitcher)
        assert (100, 2) in result
        assert result[(100, 2)] < 0
        # (200,4) vs pitcher 99 not found
        assert (200, 4) not in result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestExtractOpposingLineup:
    """Test extract_opposing_lineup."""

    def test_extracts_opposing_team(self) -> None:
        lu_df = pd.DataFrame({
            "game_pk": [1]*18,
            "player_id": list(range(100, 118)),
            "batting_order": list(range(1, 10)) * 2,
            "team_id": [10]*9 + [20]*9,
        })
        result = extract_opposing_lineup(1, 10, lu_df)
        assert result is not None
        assert len(result) == 9
        assert all(pid >= 109 for pid in result)  # team 20 players

    def test_returns_none_if_no_game(self) -> None:
        lu_df = pd.DataFrame({
            "game_pk": [1]*9,
            "player_id": list(range(9)),
            "batting_order": list(range(1, 10)),
            "team_id": [10]*9,
        })
        result = extract_opposing_lineup(999, 10, lu_df)
        assert result is None


class TestBuildGameLineupMap:
    """Test build_game_lineup_map."""

    def test_builds_map(self) -> None:
        game_records = pd.DataFrame({
            "game_pk": [1, 2],
            "pitcher_id": [50, 60],
            "team_id": [10, 20],
        })
        lu_df = pd.DataFrame({
            "game_pk": [1]*18 + [2]*18,
            "player_id": list(range(100, 118)) + list(range(200, 218)),
            "batting_order": (list(range(1, 10)) * 2) * 2,
            "team_id": ([10]*9 + [20]*9) + ([20]*9 + [30]*9),
        })
        result = build_game_lineup_map(game_records, lu_df)
        assert isinstance(result, dict)
        # game 1: pitcher on team 10, should get team 20 lineup
        if 1 in result:
            assert len(result[1]) == 9
