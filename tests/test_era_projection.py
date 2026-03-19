"""Tests for ERA/FIP derivation functions."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.derived_stats import (
    LEAGUE_FIP_CONSTANT,
    LEAGUE_HBP_RATE,
    LEAGUE_GB_PCT,
    GB_ERA_COEFFICIENT,
    ERA_FIP_GAP_EFFECTIVE_N,
    derive_pitcher_fip,
    _shrink_era_fip_gap,
    derive_pitcher_era,
    derive_pitcher_fip_batch,
    derive_pitcher_era_batch,
)


class TestDerivePitcherFip:
    """Tests for derive_pitcher_fip."""

    def test_known_inputs(self):
        """FIP with known rates should match manual computation."""
        n = 1000
        k_rate = np.full(n, 0.25)
        bb_rate = np.full(n, 0.08)
        hr_rate = np.full(n, 0.03)
        outs_rate = np.full(n, 0.65)

        fip = derive_pitcher_fip(k_rate, bb_rate, hr_rate, outs_rate)

        # Manual FIP calculation
        ip_per_bf = 0.65 / 3.0
        numer = 13 * 0.03 + 3 * (0.08 + LEAGUE_HBP_RATE) - 2 * 0.25
        expected = numer / ip_per_bf + LEAGUE_FIP_CONSTANT

        np.testing.assert_allclose(fip, expected, atol=0.001)

    def test_high_k_lowers_fip(self):
        """High strikeout pitcher should have lower FIP."""
        n = 1000
        outs_rate = np.full(n, 0.65)
        bb_rate = np.full(n, 0.07)
        hr_rate = np.full(n, 0.025)

        fip_high_k = derive_pitcher_fip(
            np.full(n, 0.30), bb_rate, hr_rate, outs_rate
        )
        fip_low_k = derive_pitcher_fip(
            np.full(n, 0.15), bb_rate, hr_rate, outs_rate
        )
        assert np.mean(fip_high_k) < np.mean(fip_low_k)

    def test_high_hr_raises_fip(self):
        """High HR rate should raise FIP."""
        n = 1000
        k_rate = np.full(n, 0.22)
        bb_rate = np.full(n, 0.08)
        outs_rate = np.full(n, 0.65)

        fip_high_hr = derive_pitcher_fip(
            k_rate, bb_rate, np.full(n, 0.05), outs_rate
        )
        fip_low_hr = derive_pitcher_fip(
            k_rate, bb_rate, np.full(n, 0.02), outs_rate
        )
        assert np.mean(fip_high_hr) > np.mean(fip_low_hr)

    def test_clipped_to_range(self):
        """FIP should be clipped to [0, 15]."""
        n = 100
        # Extreme inputs
        fip = derive_pitcher_fip(
            np.full(n, 0.50),   # extreme K
            np.full(n, 0.01),
            np.full(n, 0.001),
            np.full(n, 0.80),
        )
        assert np.all(fip >= 0.0)
        assert np.all(fip <= 15.0)

    def test_output_shape(self):
        """Output shape should match input."""
        n = 500
        rng = np.random.default_rng(42)
        fip = derive_pitcher_fip(
            rng.random(n) * 0.3,
            rng.random(n) * 0.1,
            rng.random(n) * 0.05,
            rng.random(n) * 0.3 + 0.5,
        )
        assert fip.shape == (n,)

    def test_no_nan_output(self):
        """FIP should not contain NaN."""
        n = 500
        fip = derive_pitcher_fip(
            np.full(n, 0.22),
            np.full(n, 0.08),
            np.full(n, 0.03),
            np.full(n, 0.68),
        )
        assert not np.any(np.isnan(fip))


class TestShrinkEraFipGap:
    """Tests for _shrink_era_fip_gap."""

    def test_no_data_returns_near_zero(self):
        """With no observed data and league-average GB%, gap ≈ 0."""
        mean, sd = _shrink_era_fip_gap(None, 0.0, None, LEAGUE_GB_PCT)
        assert abs(mean) < 0.01
        assert sd > 0

    def test_nan_gap_returns_prior(self):
        """NaN gap should return prior."""
        mean, sd = _shrink_era_fip_gap(float("nan"), 180.0, 0.45, LEAGUE_GB_PCT)
        assert abs(mean) < 0.05  # near-league GB%

    def test_low_ip_more_shrinkage(self):
        """Low IP should shrink more toward prior than high IP."""
        gap_observed = 1.0
        mean_low, _ = _shrink_era_fip_gap(gap_observed, 30.0, None, LEAGUE_GB_PCT)
        mean_high, _ = _shrink_era_fip_gap(gap_observed, 200.0, None, LEAGUE_GB_PCT)

        # Both between 0 (prior) and 1.0 (observed)
        assert 0 < mean_low < 1.0
        assert 0 < mean_high < 1.0
        # High IP closer to observed
        assert abs(mean_high - gap_observed) < abs(mean_low - gap_observed)

    def test_gb_pitcher_negative_prior(self):
        """High GB% pitcher should have negative prior mean."""
        mean, _ = _shrink_era_fip_gap(None, 0.0, 0.60, LEAGUE_GB_PCT)
        assert mean < 0  # negative gap → ERA below FIP

    def test_fb_pitcher_positive_prior(self):
        """Low GB% pitcher should have positive prior mean."""
        mean, _ = _shrink_era_fip_gap(None, 0.0, 0.30, LEAGUE_GB_PCT)
        assert mean > 0  # positive gap → ERA above FIP

    def test_sd_decreases_with_ip(self):
        """Posterior SD should decrease with more IP."""
        _, sd_30 = _shrink_era_fip_gap(0.5, 30.0, None, LEAGUE_GB_PCT)
        _, sd_200 = _shrink_era_fip_gap(0.5, 200.0, None, LEAGUE_GB_PCT)
        assert sd_200 < sd_30

    def test_gb_coefficient_direction(self):
        """GB coefficient should produce expected direction."""
        # 10pp above league → gap = -0.035 * 10 = -0.35
        mean, _ = _shrink_era_fip_gap(
            None, 0.0, LEAGUE_GB_PCT + 0.10, LEAGUE_GB_PCT
        )
        expected = GB_ERA_COEFFICIENT * 10
        assert abs(mean - expected) < 0.01


class TestDerivePitcherEra:
    """Tests for derive_pitcher_era."""

    def test_equals_fip_when_no_gap_data(self):
        """With no gap data, ERA should be approximately FIP."""
        n = 2000
        fip = np.full(n, 4.00)
        era = derive_pitcher_era(
            fip, None, 0.0, None,
            rng=np.random.default_rng(42),
        )
        # Gap prior ≈ 0 with no GB data, so ERA ≈ FIP
        assert abs(np.mean(era) - 4.00) < 0.6

    def test_positive_gap_raises_era(self):
        """Positive observed gap should raise ERA above FIP."""
        n = 2000
        fip = np.full(n, 3.50)
        era_pos = derive_pitcher_era(
            fip, 1.5, 180.0, None,
            rng=np.random.default_rng(42),
        )
        era_zero = derive_pitcher_era(
            fip, 0.0, 180.0, None,
            rng=np.random.default_rng(43),
        )
        assert np.mean(era_pos) > np.mean(era_zero)

    def test_negative_gap_lowers_era(self):
        """Negative observed gap should lower ERA below FIP."""
        n = 2000
        fip = np.full(n, 4.00)
        era_neg = derive_pitcher_era(
            fip, -1.0, 180.0, None,
            rng=np.random.default_rng(42),
        )
        assert np.mean(era_neg) < 4.00

    def test_bounded(self):
        """ERA should be clipped to [0, 15]."""
        n = 1000
        rng = np.random.default_rng(42)
        fip = rng.normal(4.0, 1.0, n)
        era = derive_pitcher_era(
            fip, 2.0, 180.0, 0.45,
            rng=rng,
        )
        assert np.all(era >= 0.0)
        assert np.all(era <= 15.0)

    def test_no_nan(self):
        """ERA should not contain NaN."""
        n = 500
        fip = np.random.default_rng(42).normal(4.0, 0.5, n)
        era = derive_pitcher_era(
            fip, 0.5, 150.0, 0.42,
            rng=np.random.default_rng(42),
        )
        assert not np.any(np.isnan(era))

    def test_reasonable_range_for_starter(self):
        """ERA for a typical starter should be in [2.0, 6.0]."""
        n = 2000
        fip = np.full(n, 3.80)
        era = derive_pitcher_era(
            fip, 0.2, 180.0, 0.45,
            rng=np.random.default_rng(42),
        )
        mean_era = np.mean(era)
        assert 2.0 < mean_era < 6.0


class TestFipBatch:
    """Tests for derive_pitcher_fip_batch."""

    def test_basic_batch(self):
        """Should produce FIP for pitchers with all posteriors."""
        n = 500
        rng = np.random.default_rng(42)
        posteriors = {
            "k_rate": {1: rng.beta(5, 15, n), 2: rng.beta(6, 14, n)},
            "bb_rate": {1: rng.beta(2, 23, n), 2: rng.beta(3, 22, n)},
            "hr_per_bf": {1: rng.beta(1, 30, n), 2: rng.beta(1, 28, n)},
        }
        babip_data = {1: (0.290, 300), 2: (0.310, 250)}

        result = derive_pitcher_fip_batch(posteriors, babip_data, rng=rng)

        assert 1 in result
        assert 2 in result
        assert len(result[1]) == n
        assert np.all(result[1] >= 0)
        assert np.all(result[1] <= 15)

    def test_missing_posteriors_returns_empty(self):
        """Missing required posteriors should return empty dict."""
        posteriors = {"k_rate": {1: np.ones(100)}}
        result = derive_pitcher_fip_batch(posteriors, {})
        assert len(result) == 0

    def test_partial_overlap(self):
        """Only pitchers with all three rate posteriors get FIP."""
        n = 100
        posteriors = {
            "k_rate": {
                1: np.full(n, 0.22),
                2: np.full(n, 0.20),
                3: np.full(n, 0.18),
            },
            "bb_rate": {
                1: np.full(n, 0.08),
                2: np.full(n, 0.08),
            },
            "hr_per_bf": {
                1: np.full(n, 0.03),
            },
        }
        babip_data = {1: (None, 0), 2: (None, 0), 3: (None, 0)}

        result = derive_pitcher_fip_batch(posteriors, babip_data)
        # Only pitcher 1 has all three
        assert 1 in result
        assert 2 not in result
        assert 3 not in result

    def test_no_nan_leakage(self):
        """Batch FIP should not contain NaN."""
        n = 200
        rng = np.random.default_rng(42)
        posteriors = {
            "k_rate": {1: rng.beta(5, 15, n)},
            "bb_rate": {1: rng.beta(2, 23, n)},
            "hr_per_bf": {1: rng.beta(1, 30, n)},
        }
        result = derive_pitcher_fip_batch(posteriors, {1: (None, 0)}, rng=rng)
        assert 1 in result
        assert not np.any(np.isnan(result[1]))


class TestEraBatch:
    """Tests for derive_pitcher_era_batch."""

    def test_basic_batch(self):
        """Should produce ERA for all pitchers with FIP posteriors."""
        n = 500
        rng = np.random.default_rng(42)
        fip_posteriors = {
            1: rng.normal(3.5, 0.3, n),
            2: rng.normal(4.5, 0.4, n),
        }
        era_fip_data = {
            1: (0.3, 180.0, 0.50),
            2: (-0.2, 160.0, 0.40),
        }

        result = derive_pitcher_era_batch(fip_posteriors, era_fip_data, rng=rng)

        assert 1 in result
        assert 2 in result
        assert len(result[1]) == n
        assert np.all(result[1] >= 0)
        assert np.all(result[1] <= 15)

    def test_missing_era_data_uses_prior(self):
        """Pitcher without ERA-FIP data should get ERA ≈ FIP."""
        n = 2000
        fip_posteriors = {1: np.full(n, 4.00)}
        era_fip_data = {}  # no data for pitcher 1

        result = derive_pitcher_era_batch(
            fip_posteriors, era_fip_data,
            rng=np.random.default_rng(42),
        )
        assert 1 in result
        # ERA should be close to FIP (gap prior ≈ 0)
        assert abs(np.mean(result[1]) - 4.00) < 0.6

    def test_no_nan_leakage(self):
        """Batch ERA should not contain NaN."""
        n = 500
        rng = np.random.default_rng(42)
        fip_posteriors = {1: rng.normal(4.0, 0.3, n)}
        era_fip_data = {1: (0.5, 180.0, 0.45)}

        result = derive_pitcher_era_batch(fip_posteriors, era_fip_data, rng=rng)
        assert not np.any(np.isnan(result[1]))

    def test_all_bounded(self):
        """All ERA values should be in [0, 15]."""
        n = 1000
        rng = np.random.default_rng(42)
        fip_posteriors = {
            pid: rng.normal(4.0, 0.5, n)
            for pid in range(1, 11)
        }
        era_fip_data = {
            pid: (rng.normal(0, 0.5), 150.0, rng.uniform(0.3, 0.6))
            for pid in range(1, 11)
        }

        result = derive_pitcher_era_batch(fip_posteriors, era_fip_data, rng=rng)
        for pid, samples in result.items():
            assert np.all(samples >= 0), f"Pitcher {pid} has ERA < 0"
            assert np.all(samples <= 15), f"Pitcher {pid} has ERA > 15"
