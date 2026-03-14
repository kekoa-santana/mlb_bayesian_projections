"""Tests for the game K posterior Monte Carlo engine (Step 14)."""
import numpy as np
import pandas as pd
import pytest

from src.models.game_k_model import (
    _LEAGUE_TTO_LOGIT_LIFTS,
    build_tto_logit_lifts,
    compute_k_over_probs,
    simulate_game_ks,
    simulate_game_stat,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSimulateGameKsOutputShape:
    def test_returns_correct_shape(self) -> None:
        """Returns (n_draws,) array of non-negative ints."""
        k_rate_samples = np.full(4000, 0.25)
        result = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0,
            bf_sigma=3.5,
            n_draws=4000,
            random_seed=42,
        )
        assert result.shape == (4000,)
        assert result.dtype in (np.int32, np.int64, int)
        assert result.min() >= 0


class TestSimulateGameKsExpectedValue:
    def test_expected_k_reasonable(self) -> None:
        """K%=0.25, BF~23 → expected K ≈ 5.75 ± 0.5."""
        k_rate_samples = np.full(10000, 0.25)
        result = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0,
            bf_sigma=3.5,
            n_draws=10000,
            random_seed=42,
        )
        expected = result.mean()
        # 0.25 * 23 = 5.75, allow ±0.5 for MC noise
        assert abs(expected - 5.75) < 0.5


class TestSimulateGameKsNoMatchup:
    def test_no_matchup_works(self) -> None:
        """lineup_matchup_lifts=None works and returns baseline K."""
        k_rate_samples = np.full(2000, 0.22)
        result = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0,
            bf_sigma=4.0,
            lineup_matchup_lifts=None,
            n_draws=2000,
            random_seed=42,
        )
        assert result.shape == (2000,)
        assert result.mean() > 0


class TestSimulateGameKsPositiveLiftIncreasesKs:
    def test_positive_lifts(self) -> None:
        """All positive lifts → higher expected K than baseline."""
        k_rate_samples = np.full(5000, 0.22)

        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            lineup_matchup_lifts=None,
            n_draws=5000, random_seed=42,
        )
        boosted = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            lineup_matchup_lifts=np.full(9, 0.5),  # +0.5 logit lift
            n_draws=5000, random_seed=42,
        )
        assert boosted.mean() > baseline.mean()


class TestComputeKOverProbsMonotonic:
    def test_p_over_decreases(self) -> None:
        """P(over X.5) decreases as X increases."""
        rng = np.random.default_rng(42)
        k_samples = rng.poisson(6.0, size=10000)
        result = compute_k_over_probs(k_samples)

        p_overs = result["p_over"].values
        # Each successive line should have lower or equal P(over)
        for i in range(len(p_overs) - 1):
            assert p_overs[i] >= p_overs[i + 1]

    def test_columns(self) -> None:
        """Output has expected columns."""
        k_samples = np.array([3, 4, 5, 6, 7, 5, 6, 4, 8, 5])
        result = compute_k_over_probs(k_samples)
        expected_cols = {"line", "p_over", "p_under", "expected_k", "std_k"}
        assert expected_cols.issubset(set(result.columns))
        # Default 13 lines (0.5 through 12.5)
        assert len(result) == 13

    def test_custom_lines(self) -> None:
        """Custom lines produce correct number of rows."""
        k_samples = np.array([3, 4, 5, 6, 7])
        result = compute_k_over_probs(k_samples, lines=[4.5, 5.5, 6.5])
        assert len(result) == 3


class TestUmpireKLogitLift:
    def test_positive_umpire_lift_increases_ks(self) -> None:
        """Umpire with high K-rate tendency should increase expected Ks."""
        k_rate_samples = np.full(5000, 0.22)
        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            umpire_k_logit_lift=0.0,
            n_draws=5000, random_seed=42,
        )
        high_k_ump = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            umpire_k_logit_lift=0.15,  # ~2pp above avg
            n_draws=5000, random_seed=42,
        )
        assert high_k_ump.mean() > baseline.mean()

    def test_negative_umpire_lift_decreases_ks(self) -> None:
        """Umpire with low K-rate tendency should decrease expected Ks."""
        k_rate_samples = np.full(5000, 0.22)
        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            umpire_k_logit_lift=0.0,
            n_draws=5000, random_seed=42,
        )
        low_k_ump = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            umpire_k_logit_lift=-0.15,  # ~2pp below avg
            n_draws=5000, random_seed=42,
        )
        assert low_k_ump.mean() < baseline.mean()

    def test_zero_lift_matches_baseline(self) -> None:
        """umpire_k_logit_lift=0 should match no-umpire baseline."""
        k_rate_samples = np.full(2000, 0.25)
        with_zero = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0, bf_sigma=3.5,
            umpire_k_logit_lift=0.0,
            n_draws=2000, random_seed=42,
        )
        without = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0, bf_sigma=3.5,
            n_draws=2000, random_seed=42,
        )
        assert np.array_equal(with_zero, without)

    def test_umpire_combines_with_matchup(self) -> None:
        """Umpire lift stacks with matchup lifts."""
        k_rate_samples = np.full(5000, 0.22)
        matchup_only = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            lineup_matchup_lifts=np.full(9, 0.3),
            umpire_k_logit_lift=0.0,
            n_draws=5000, random_seed=42,
        )
        matchup_plus_ump = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            lineup_matchup_lifts=np.full(9, 0.3),
            umpire_k_logit_lift=0.15,
            n_draws=5000, random_seed=42,
        )
        assert matchup_plus_ump.mean() > matchup_only.mean()


class TestWeatherKLogitLift:
    def test_cold_weather_increases_ks(self) -> None:
        """Cold weather (positive K lift) should increase expected Ks."""
        k_rate_samples = np.full(5000, 0.22)
        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            weather_k_logit_lift=0.0,
            n_draws=5000, random_seed=42,
        )
        cold = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            weather_k_logit_lift=0.05,  # cold weather boost
            n_draws=5000, random_seed=42,
        )
        assert cold.mean() > baseline.mean()

    def test_hot_weather_decreases_ks(self) -> None:
        """Hot weather (negative K lift) should decrease expected Ks."""
        k_rate_samples = np.full(5000, 0.22)
        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            weather_k_logit_lift=0.0,
            n_draws=5000, random_seed=42,
        )
        hot = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            weather_k_logit_lift=-0.05,  # hot weather suppression
            n_draws=5000, random_seed=42,
        )
        assert hot.mean() < baseline.mean()

    def test_all_adjustments_stack(self) -> None:
        """Umpire + weather + matchup all stack additively on logit scale."""
        k_rate_samples = np.full(5000, 0.22)
        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            n_draws=5000, random_seed=42,
        )
        all_positive = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            lineup_matchup_lifts=np.full(9, 0.2),
            umpire_k_logit_lift=0.10,
            weather_k_logit_lift=0.05,
            n_draws=5000, random_seed=42,
        )
        assert all_positive.mean() > baseline.mean()

    def test_dome_neutral(self) -> None:
        """weather_k_logit_lift=0 (dome) should match baseline exactly."""
        k_rate_samples = np.full(2000, 0.25)
        with_zero = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0, bf_sigma=3.5,
            weather_k_logit_lift=0.0,
            n_draws=2000, random_seed=42,
        )
        without = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0, bf_sigma=3.5,
            n_draws=2000, random_seed=42,
        )
        assert np.array_equal(with_zero, without)


# ---------------------------------------------------------------------------
# TTO (Times-Through-Order) adjustment tests
# ---------------------------------------------------------------------------
class TestTTOLogitLiftsConstants:
    def test_league_tto_k_lifts_direction(self) -> None:
        """TTO1 should be positive (higher K), TTO3 negative (lower K)."""
        k_lifts = _LEAGUE_TTO_LOGIT_LIFTS["k"]
        assert k_lifts[0] > 0, "TTO1 K lift should be positive"
        assert k_lifts[1] < 0, "TTO2 K lift should be negative"
        assert k_lifts[2] < 0, "TTO3 K lift should be negative"
        assert k_lifts[2] < k_lifts[1], "TTO3 K lift should be more negative than TTO2"

    def test_league_tto_hr_lifts_direction(self) -> None:
        """HR rate increases through the order (familiarity)."""
        hr_lifts = _LEAGUE_TTO_LOGIT_LIFTS["hr"]
        assert hr_lifts[0] < 0, "TTO1 HR lift should be negative"
        assert hr_lifts[2] > 0, "TTO3 HR lift should be positive"


class TestBuildTTOLogitLifts:
    def test_none_profiles_returns_league_avg(self) -> None:
        """With no profile data, fall back to league-average lifts."""
        lifts = build_tto_logit_lifts(None, pitcher_id=12345, season=2024)
        np.testing.assert_array_almost_equal(
            lifts, _LEAGUE_TTO_LOGIT_LIFTS["k"], decimal=5,
        )

    def test_empty_df_returns_league_avg(self) -> None:
        """Empty DataFrame also falls back to league average."""
        lifts = build_tto_logit_lifts(
            pd.DataFrame(), pitcher_id=12345, season=2024,
        )
        np.testing.assert_array_almost_equal(
            lifts, _LEAGUE_TTO_LOGIT_LIFTS["k"], decimal=5,
        )

    def test_pitcher_specific_profile(self) -> None:
        """When pitcher data exists, lifts blend toward pitcher-specific values."""
        tto_df = pd.DataFrame({
            "pitcher_id": [999] * 3,
            "season": [2024] * 3,
            "tto": [1, 2, 3],
            "k_rate": [0.30, 0.20, 0.15],
            "bb_rate": [0.08, 0.08, 0.08],
            "hr_rate": [0.03, 0.03, 0.03],
            "overall_k_rate": [0.25, 0.25, 0.25],
            "overall_bb_rate": [0.08, 0.08, 0.08],
            "overall_hr_rate": [0.03, 0.03, 0.03],
            "pa_count": [200, 150, 100],
        })
        lifts = build_tto_logit_lifts(tto_df, pitcher_id=999, season=2024)
        assert lifts[0] > 0
        assert lifts[2] < 0
        assert lifts[2] < lifts[1]

    def test_unknown_stat_returns_zeros(self) -> None:
        """Unknown stat name returns zero lifts."""
        lifts = build_tto_logit_lifts(None, pitcher_id=12345, season=2024,
                                       stat_name="unknown")
        np.testing.assert_array_equal(lifts, np.zeros(3))


class TestSimulateGameKsWithTTO:
    def test_tto_none_backward_compatible(self) -> None:
        """tto_logit_lifts=None should match the no-TTO baseline exactly."""
        k_rate_samples = np.full(2000, 0.25)
        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0, bf_sigma=3.5,
            tto_logit_lifts=None,
            n_draws=2000, random_seed=42,
        )
        no_tto = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0, bf_sigma=3.5,
            n_draws=2000, random_seed=42,
        )
        assert np.array_equal(baseline, no_tto)

    def test_tto_changes_expected_ks(self) -> None:
        """TTO adjustment should change expected Ks vs flat rate."""
        k_rate_samples = np.full(10000, 0.22)
        flat = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=25.0, bf_sigma=3.0,
            tto_logit_lifts=None,
            n_draws=10000, random_seed=42,
        )
        with_tto = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=25.0, bf_sigma=3.0,
            tto_logit_lifts=_LEAGUE_TTO_LOGIT_LIFTS["k"],
            n_draws=10000, random_seed=42,
        )
        assert flat.mean() != with_tto.mean()

    def test_tto_stacks_with_other_lifts(self) -> None:
        """TTO lifts stack with umpire and matchup lifts."""
        k_rate_samples = np.full(5000, 0.22)
        tto_only = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            tto_logit_lifts=_LEAGUE_TTO_LOGIT_LIFTS["k"],
            n_draws=5000, random_seed=42,
        )
        tto_plus_ump = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            tto_logit_lifts=_LEAGUE_TTO_LOGIT_LIFTS["k"],
            umpire_k_logit_lift=0.15,
            n_draws=5000, random_seed=42,
        )
        assert tto_plus_ump.mean() > tto_only.mean()

    def test_extreme_tto_penalty_lowers_ks(self) -> None:
        """Extreme TTO penalty across all TTOs should dramatically lower Ks."""
        k_rate_samples = np.full(5000, 0.25)
        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0, bf_sigma=3.5,
            tto_logit_lifts=None,
            n_draws=5000, random_seed=42,
        )
        penalized = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0, bf_sigma=3.5,
            tto_logit_lifts=np.array([-0.5, -0.5, -0.5]),
            n_draws=5000, random_seed=42,
        )
        assert penalized.mean() < baseline.mean()


class TestSimulateGameStatWithTTO:
    def test_tto_none_backward_compatible(self) -> None:
        """simulate_game_stat with tto=None matches old behavior."""
        rate_samples = np.full(2000, 0.22)
        baseline = simulate_game_stat(
            rate_samples=rate_samples,
            opp_mu=23.0, opp_sigma=3.5,
            tto_logit_lifts=None,
            n_draws=2000, random_seed=42,
        )
        no_tto = simulate_game_stat(
            rate_samples=rate_samples,
            opp_mu=23.0, opp_sigma=3.5,
            n_draws=2000, random_seed=42,
        )
        assert np.array_equal(baseline, no_tto)

    def test_batter_mode_ignores_tto(self) -> None:
        """In batter mode (n_slots=1), TTO lifts are ignored."""
        rate_samples = np.full(2000, 0.22)
        with_tto = simulate_game_stat(
            rate_samples=rate_samples,
            opp_mu=4.0, opp_sigma=0.8,
            tto_logit_lifts=_LEAGUE_TTO_LOGIT_LIFTS["k"],
            n_slots=1,
            n_draws=2000, random_seed=42,
        )
        without_tto = simulate_game_stat(
            rate_samples=rate_samples,
            opp_mu=4.0, opp_sigma=0.8,
            tto_logit_lifts=None,
            n_slots=1,
            n_draws=2000, random_seed=42,
        )
        assert np.array_equal(with_tto, without_tto)
