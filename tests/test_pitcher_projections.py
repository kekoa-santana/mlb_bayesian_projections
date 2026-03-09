"""Tests for the composite pitcher projection system."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.pitcher_projections import (
    ALL_STATS,
    COMPOSITE_DIMENSIONS,
    find_breakouts_and_regressions,
    project_forward,
)


def _make_synthetic_pitcher_data(
    n_players: int = 20,
    n_seasons: int = 3,
    bf_per_season: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic pitcher data matching the expected schema."""
    rng = np.random.default_rng(seed)
    records = []
    for p in range(n_players):
        true_k = rng.beta(6, 20)
        true_bb = rng.beta(3, 33)
        true_hr = rng.beta(1, 35)
        age_base = rng.integers(23, 36)
        is_starter = rng.random() > 0.3
        for s in range(n_seasons):
            bf = bf_per_season + rng.integers(-80, 80)
            if not is_starter:
                bf = bf // 3
            k = rng.binomial(bf, true_k)
            bb = rng.binomial(bf, true_bb)
            hr = rng.binomial(bf, true_hr)
            age = age_base + s
            if age <= 25:
                ab = 0
            elif age <= 30:
                ab = 1
            else:
                ab = 2
            records.append({
                "pitcher_id": 200000 + p,
                "pitcher_name": f"Pitcher, Test{p}",
                "pitch_hand": rng.choice(["L", "R"]),
                "season": 2022 + s,
                "age": age,
                "age_bucket": ab,
                "batters_faced": bf,
                "is_starter": is_starter,
                "k": k,
                "bb": bb,
                "hr": hr,
                "k_rate": round(k / bf, 4),
                "bb_rate": round(bb / bf, 4),
                "hr_per_bf": round(hr / bf, 4),
            })
    return pd.DataFrame(records)


class TestCompositeDimensions:
    """Test composite dimension configuration."""

    def test_weights_sum_to_one(self):
        total = sum(w for w, _ in COMPOSITE_DIMENSIONS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_projected_stats_have_models(self):
        """ALL_STATS should only contain stable, projectable stats."""
        assert "k_rate" in ALL_STATS
        assert "bb_rate" in ALL_STATS
        # HR/BF removed — not stable enough for Bayesian projection
        assert "hr_per_bf" not in ALL_STATS

    def test_four_dimensions(self):
        expected = {"stuff", "command", "ground_ball", "trajectory"}
        assert set(COMPOSITE_DIMENSIONS.keys()) == expected


class TestProjectForward:
    """Test projection logic with mock model results."""

    @pytest.fixture
    def mock_model_results(self):
        """Build minimal mock model results to test project_forward."""
        from unittest.mock import MagicMock
        from src.models.pitcher_model import PITCHER_STAT_CONFIGS

        df = _make_synthetic_pitcher_data(n_players=10, n_seasons=2)
        results = {}

        for stat in ALL_STATS:
            cfg = PITCHER_STAT_CONFIGS[stat]
            data = {"df": df, "player_ids": df["pitcher_id"].unique()}

            trace = MagicMock()
            results[stat] = {
                "data": data,
                "trace": trace,
                "convergence": {"converged": True, "max_rhat": 1.01},
                "posteriors": df,
            }

        return results

    def test_empty_season_returns_empty(self, mock_model_results):
        """Projecting from a season with no data returns empty DataFrame."""
        proj = project_forward(mock_model_results, from_season=1999, min_bf=200)
        assert len(proj) == 0


class TestFindBreakoutsAndRegressions:
    """Test breakout/regression splitting."""

    def test_split_sizes(self):
        n = 30
        df = pd.DataFrame({
            "pitcher_id": range(n),
            "composite_score": np.linspace(-2, 2, n),
        })
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        breakouts, regressions = find_breakouts_and_regressions(df, n_top=5)
        assert len(breakouts) == 5
        assert len(regressions) == 5

    def test_breakouts_have_higher_scores(self):
        n = 20
        df = pd.DataFrame({
            "pitcher_id": range(n),
            "composite_score": np.linspace(-2, 2, n),
        })
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        breakouts, regressions = find_breakouts_and_regressions(df, n_top=5)
        assert breakouts["composite_score"].min() > regressions["composite_score"].max()

    def test_n_top_larger_than_df(self):
        df = pd.DataFrame({
            "pitcher_id": range(5),
            "composite_score": [1, 0.5, 0, -0.5, -1],
        })
        breakouts, regressions = find_breakouts_and_regressions(df, n_top=10)
        assert len(breakouts) == 5
        assert len(regressions) == 5
