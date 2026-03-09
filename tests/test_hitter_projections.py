"""Tests for the composite hitter projection system."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.hitter_projections import (
    ALL_STATS,
    COMPOSITE_DIMENSIONS,
    find_breakouts_and_regressions,
    project_forward,
)


def _make_synthetic_data(
    n_players: int = 20,
    n_seasons: int = 3,
    pa_per_season: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic hitter data matching the expected schema."""
    rng = np.random.default_rng(seed)
    records = []
    for p in range(n_players):
        true_k = rng.beta(5, 18)
        true_bb = rng.beta(3, 33)
        true_hr = rng.beta(1, 29)
        true_xwoba = rng.normal(0.315, 0.035)
        age_base = rng.integers(22, 36)
        for s in range(n_seasons):
            pa = pa_per_season + rng.integers(-50, 50)
            k = rng.binomial(pa, true_k)
            bb = rng.binomial(pa, true_bb)
            hr = rng.binomial(pa, true_hr)
            age = age_base + s
            if age <= 25:
                ab = 0
            elif age <= 30:
                ab = 1
            else:
                ab = 2
            records.append({
                "batter_id": 100000 + p,
                "batter_name": f"Player, Test{p}",
                "batter_stand": rng.choice(["L", "R"]),
                "season": 2022 + s,
                "age": age,
                "age_bucket": ab,
                "pa": pa,
                "k": k,
                "bb": bb,
                "hits": int(pa * 0.25),
                "hr": hr,
                "xwoba_avg": float(np.clip(true_xwoba + rng.normal(0, 0.02), 0.1, 0.5)),
                "barrel_pct": float(rng.beta(2, 30)),
                "hard_hit_pct": float(rng.beta(8, 16)),
                "k_rate": round(k / pa, 4),
                "bb_rate": round(bb / pa, 4),
                "hr_rate": round(hr / pa, 4),
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
        # xwOBA and HR/PA removed — not stable enough for Bayesian projection
        assert "hr_rate" not in ALL_STATS
        assert "xwoba" not in ALL_STATS

    def test_six_dimensions(self):
        expected = {"contact", "discipline", "power", "speed",
                    "batted_ball", "trajectory"}
        assert set(COMPOSITE_DIMENSIONS.keys()) == expected


class TestProjectForward:
    """Test projection logic with mock model results."""

    @pytest.fixture
    def mock_model_results(self):
        """Build minimal mock model results to test project_forward."""
        from unittest.mock import MagicMock
        from src.models.hitter_model import STAT_CONFIGS

        df = _make_synthetic_data(n_players=10, n_seasons=2)
        results = {}

        for stat in ALL_STATS:
            cfg = STAT_CONFIGS[stat]
            data = {"df": df, "player_ids": df["batter_id"].unique()}

            # Mock trace that returns plausible samples
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
        proj = project_forward(mock_model_results, from_season=1999, min_pa=200)
        assert len(proj) == 0


class TestFindBreakoutsAndRegressions:
    """Test breakout/regression splitting."""

    def test_split_sizes(self):
        n = 30
        df = pd.DataFrame({
            "batter_id": range(n),
            "composite_score": np.linspace(-2, 2, n),
        })
        # Sort descending like project_forward does
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        breakouts, regressions = find_breakouts_and_regressions(df, n_top=5)
        assert len(breakouts) == 5
        assert len(regressions) == 5

    def test_breakouts_have_higher_scores(self):
        n = 20
        df = pd.DataFrame({
            "batter_id": range(n),
            "composite_score": np.linspace(-2, 2, n),
        })
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        breakouts, regressions = find_breakouts_and_regressions(df, n_top=5)
        assert breakouts["composite_score"].min() > regressions["composite_score"].max()

    def test_n_top_larger_than_df(self):
        df = pd.DataFrame({
            "batter_id": range(5),
            "composite_score": [1, 0.5, 0, -0.5, -1],
        })
        breakouts, regressions = find_breakouts_and_regressions(df, n_top=10)
        assert len(breakouts) == 5
        assert len(regressions) == 5
