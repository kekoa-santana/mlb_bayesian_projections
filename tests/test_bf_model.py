"""Tests for the BF distribution model (Step 13)."""
import numpy as np
import pandas as pd
import pytest

from src.models.bf_model import (
    compute_pitcher_bf_priors,
    draw_bf_samples,
    get_bf_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def game_logs() -> pd.DataFrame:
    """Synthetic pitcher game logs."""
    rng = np.random.default_rng(42)
    records = []

    # Pitcher 1: 30 starts, mean BF ~ 23
    for i in range(30):
        records.append({
            "pitcher_id": 1,
            "season": 2024,
            "batters_faced": int(rng.normal(23, 3)),
            "is_starter": True,
            "innings_pitched": 6.0,
            "strike_outs": 6,
        })

    # Pitcher 2: 3 starts (below min_starts=5)
    for i in range(3):
        records.append({
            "pitcher_id": 2,
            "season": 2024,
            "batters_faced": int(rng.normal(20, 4)),
            "is_starter": True,
            "innings_pitched": 5.0,
            "strike_outs": 4,
        })

    # Pitcher 3: 10 starts, mean BF ~ 18
    for i in range(10):
        records.append({
            "pitcher_id": 3,
            "season": 2024,
            "batters_faced": int(rng.normal(18, 2)),
            "is_starter": True,
            "innings_pitched": 4.5,
            "strike_outs": 3,
        })

    # Reliever (should be excluded)
    for i in range(40):
        records.append({
            "pitcher_id": 4,
            "season": 2024,
            "batters_faced": int(rng.normal(4, 1)),
            "is_starter": False,
            "innings_pitched": 1.0,
            "strike_outs": 1,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestComputeBfPriorsShapes:
    def test_output_columns(self, game_logs: pd.DataFrame) -> None:
        """Output has expected columns, one row per pitcher-season."""
        priors = compute_pitcher_bf_priors(game_logs)
        expected_cols = {
            "pitcher_id", "season", "n_starts", "raw_mean_bf",
            "raw_std_bf", "mu_bf", "sigma_bf", "reliability",
        }
        assert expected_cols.issubset(set(priors.columns))
        # 3 starters, reliever excluded
        assert len(priors) == 3

    def test_no_relievers(self, game_logs: pd.DataFrame) -> None:
        """Relievers should not appear in BF priors."""
        priors = compute_pitcher_bf_priors(game_logs)
        assert 4 not in priors["pitcher_id"].values


class TestShrinkageHighStarts:
    def test_high_starts_reliability(self, game_logs: pd.DataFrame) -> None:
        """30-start pitcher: reliability > 0.85, mu_bf ≈ raw mean."""
        priors = compute_pitcher_bf_priors(game_logs)
        p1 = priors[priors["pitcher_id"] == 1].iloc[0]
        assert p1["reliability"] > 0.85
        # mu_bf should be close to raw_mean_bf (mostly own data)
        assert abs(p1["mu_bf"] - p1["raw_mean_bf"]) < 1.0


class TestShrinkageLowStarts:
    def test_low_starts_pure_population(self, game_logs: pd.DataFrame) -> None:
        """3-start pitcher (< min_starts=5): reliability = 0, population prior."""
        priors = compute_pitcher_bf_priors(game_logs, min_starts=5)
        p2 = priors[priors["pitcher_id"] == 2].iloc[0]
        assert p2["reliability"] == 0.0
        assert p2["mu_bf"] == 23.0  # pop_mu default (DEFAULT_POP_BF_MU)


class TestDrawBfSamplesBounded:
    def test_samples_in_range(self) -> None:
        """All samples in [bf_min, bf_max], integer-valued."""
        samples = draw_bf_samples(
            mu_bf=22.0, sigma_bf=4.0, n_draws=10000,
            bf_min=3, bf_max=35, rng=np.random.default_rng(42),
        )
        assert samples.dtype == int
        assert samples.min() >= 3
        assert samples.max() <= 35
        assert len(samples) == 10000


class TestDrawBfSamplesMean:
    def test_mean_near_mu(self) -> None:
        """Mean of draws ≈ mu_bf (within 0.5)."""
        samples = draw_bf_samples(
            mu_bf=22.0, sigma_bf=4.0, n_draws=50000,
            bf_min=3, bf_max=35, rng=np.random.default_rng(42),
        )
        assert abs(samples.mean() - 22.0) < 0.5


class TestGetBfDistribution:
    def test_found_pitcher(self, game_logs: pd.DataFrame) -> None:
        """Known pitcher returns shrinkage distribution."""
        priors = compute_pitcher_bf_priors(game_logs)
        result = get_bf_distribution(1, 2024, priors)
        assert result["dist_type"] == "shrinkage"
        assert result["reliability"] > 0.5

    def test_missing_pitcher(self, game_logs: pd.DataFrame) -> None:
        """Unknown pitcher falls back to population."""
        priors = compute_pitcher_bf_priors(game_logs)
        result = get_bf_distribution(999, 2024, priors)
        assert result["dist_type"] == "population_fallback"
        assert result["mu_bf"] == 23.0
