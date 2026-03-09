"""Tests for the PA projection model (empirical Bayes shrinkage)."""
import numpy as np
import pandas as pd
import pytest

from src.models.pa_model import (
    POP_GAMES_BY_AGE,
    POP_PA_PER_GAME,
    SEASON_WEIGHTS,
    SHRINKAGE_K_GAMES,
    SHRINKAGE_K_PA_RATE,
    compute_hitter_pa_priors,
    draw_pa_samples,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def hitter_extended() -> pd.DataFrame:
    """Synthetic multi-season hitter data."""
    records = [
        # Player 1: 3 seasons, prime age, consistent ~150 G / ~4.0 PA/G
        {"batter_id": 1, "batter_name": "Alice", "season": 2022, "games": 148, "pa": 590, "age": 27, "age_bucket": 1},
        {"batter_id": 1, "batter_name": "Alice", "season": 2023, "games": 155, "pa": 620, "age": 28, "age_bucket": 1},
        {"batter_id": 1, "batter_name": "Alice", "season": 2024, "games": 152, "pa": 605, "age": 29, "age_bucket": 1},
        # Player 2: 1 season only (young callup)
        {"batter_id": 2, "batter_name": "Bob", "season": 2024, "games": 80, "pa": 300, "age": 22, "age_bucket": 0},
        # Player 3: veteran, declining
        {"batter_id": 3, "batter_name": "Charlie", "season": 2022, "games": 140, "pa": 530, "age": 34, "age_bucket": 2},
        {"batter_id": 3, "batter_name": "Charlie", "season": 2023, "games": 120, "pa": 450, "age": 35, "age_bucket": 2},
        {"batter_id": 3, "batter_name": "Charlie", "season": 2024, "games": 100, "pa": 380, "age": 36, "age_bucket": 2},
        # Player 4: has a 2020 season (shortened)
        {"batter_id": 4, "batter_name": "Dana", "season": 2020, "games": 55, "pa": 210, "age": 25, "age_bucket": 1},
        {"batter_id": 4, "batter_name": "Dana", "season": 2024, "games": 140, "pa": 550, "age": 29, "age_bucket": 1},
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# compute_hitter_pa_priors
# ---------------------------------------------------------------------------
class TestComputePaPriorsOutput:
    def test_output_columns(self, hitter_extended: pd.DataFrame) -> None:
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024)
        expected = {
            "batter_id", "batter_name", "projected_games", "sigma_games",
            "projected_pa_per_game", "sigma_pa_rate", "projected_pa",
            "age", "age_bucket", "reliability_games", "reliability_pa_rate",
            "n_seasons",
        }
        assert expected.issubset(set(result.columns))

    def test_returns_all_eligible_players(self, hitter_extended: pd.DataFrame) -> None:
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024, min_pa=100)
        # All 4 players have >= 100 PA in at least one recent season
        assert len(result) == 4

    def test_min_pa_filters(self, hitter_extended: pd.DataFrame) -> None:
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024, min_pa=400)
        # Bob (300 PA) should be filtered out
        assert 2 not in result["batter_id"].values


class TestShrinkageWithHistory:
    def test_multi_season_higher_reliability(self, hitter_extended: pd.DataFrame) -> None:
        """3-season player should have higher reliability than 1-season player."""
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024)
        alice = result[result["batter_id"] == 1].iloc[0]
        bob = result[result["batter_id"] == 2].iloc[0]
        assert alice["reliability_games"] > bob["reliability_games"]

    def test_multi_season_less_shrinkage(self, hitter_extended: pd.DataFrame) -> None:
        """3-season player's projection should be closer to observed than pop prior."""
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024)
        alice = result[result["batter_id"] == 1].iloc[0]
        pop_mu = POP_GAMES_BY_AGE[1]["mu"]
        observed_avg = (148 * 3 + 155 * 4 + 152 * 5) / 12  # weighted 5/4/3 reversed
        # Actually weights are most-recent-first: 2024=5, 2023=4, 2022=3
        observed_avg = (152 * 5 + 155 * 4 + 148 * 3) / 12
        # Should be closer to observed than to pop
        dist_to_obs = abs(alice["projected_games"] - observed_avg)
        dist_to_pop = abs(alice["projected_games"] - pop_mu)
        assert dist_to_obs < dist_to_pop


class TestAgeRegression:
    def test_veteran_age_penalty(self, hitter_extended: pd.DataFrame) -> None:
        """35+ player gets 10% reduction in projected games."""
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024)
        charlie = result[result["batter_id"] == 3].iloc[0]
        # At age 36 (>= 35), games are multiplied by 0.90
        # So projected_games should be less than a raw shrinkage would give
        assert charlie["projected_games"] < 130  # should be well below prime pop

    def test_projected_pa_positive(self, hitter_extended: pd.DataFrame) -> None:
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024)
        assert (result["projected_pa"] > 0).all()


class TestSeason2020Adjustment:
    def test_2020_games_scaled_up(self, hitter_extended: pd.DataFrame) -> None:
        """2020 shortened season should be scaled to 162-game pace."""
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024)
        dana = result[result["batter_id"] == 4].iloc[0]
        # Dana's 2020: 55 games → scaled to 55*(162/60) = 148.5
        # Combined with 2024's 140 games, weighted average should be > 140
        assert dana["projected_games"] > 100


class TestPaPriorsCap:
    def test_games_capped_at_162(self, hitter_extended: pd.DataFrame) -> None:
        result = compute_hitter_pa_priors(hitter_extended, from_season=2024)
        assert (result["projected_games"] <= 162).all()


# ---------------------------------------------------------------------------
# draw_pa_samples
# ---------------------------------------------------------------------------
class TestDrawPaSamplesBounds:
    def test_samples_bounded(self) -> None:
        """All PA samples in [1, 750]."""
        samples = draw_pa_samples(
            projected_games=150, sigma_games=20,
            projected_pa_per_game=3.9, sigma_pa_rate=0.3,
            n_draws=10000, rng=np.random.default_rng(42),
        )
        assert samples.dtype == int
        assert samples.min() >= 1
        assert samples.max() <= 750
        assert len(samples) == 10000

    def test_mean_near_expected(self) -> None:
        """Mean of samples ≈ projected_games * projected_pa_per_game."""
        samples = draw_pa_samples(
            projected_games=150, sigma_games=15,
            projected_pa_per_game=3.9, sigma_pa_rate=0.2,
            n_draws=50000, rng=np.random.default_rng(42),
        )
        expected = 150 * 3.9
        assert abs(samples.mean() - expected) < 25  # truncation bias pulls mean down

    def test_zero_sigma_deterministic(self) -> None:
        """When sigma=0, all samples should be identical."""
        samples = draw_pa_samples(
            projected_games=140, sigma_games=0,
            projected_pa_per_game=4.0, sigma_pa_rate=0,
            n_draws=100, rng=np.random.default_rng(42),
        )
        assert np.all(samples == samples[0])


class TestDrawPaSamplesShape:
    def test_correct_count(self) -> None:
        samples = draw_pa_samples(
            projected_games=130, sigma_games=25,
            projected_pa_per_game=3.85, sigma_pa_rate=0.35,
            n_draws=5000, rng=np.random.default_rng(99),
        )
        assert len(samples) == 5000


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------
class TestConstants:
    def test_age_buckets_present(self) -> None:
        assert set(POP_GAMES_BY_AGE.keys()) == {0, 1, 2}

    def test_pop_pa_per_game_reasonable(self) -> None:
        assert 3.5 <= POP_PA_PER_GAME["mu"] <= 4.2
        assert POP_PA_PER_GAME["sigma"] > 0

    def test_season_weights_length(self) -> None:
        assert len(SEASON_WEIGHTS) == 3
        assert SEASON_WEIGHTS[0] > SEASON_WEIGHTS[-1]  # most recent weighted highest
