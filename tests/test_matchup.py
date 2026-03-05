"""Tests for the pitch-type matchup scoring module."""
import numpy as np
import pandas as pd
import pytest

from src.models.matchup import (
    _get_hitter_whiff_with_fallback,
    _inv_logit,
    _logit,
    score_matchup,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
LEAGUE_BASELINES = {
    "FF": {"whiff_rate": 0.22},
    "SL": {"whiff_rate": 0.33},
    "CH": {"whiff_rate": 0.32},
    "ST": {"whiff_rate": 0.36},
}


@pytest.fixture()
def pitcher_arsenal() -> pd.DataFrame:
    """Synthetic pitcher 100: FF 60%, SL 25%, CH 15%."""
    return pd.DataFrame([
        {
            "pitcher_id": 100, "pitch_type": "FF", "pitches": 600,
            "total_pitches": 1000, "usage_pct": 0.60, "swings": 300,
            "whiffs": 72, "whiff_rate": 0.24, "bip": 180,
        },
        {
            "pitcher_id": 100, "pitch_type": "SL", "pitches": 250,
            "total_pitches": 1000, "usage_pct": 0.25, "swings": 125,
            "whiffs": 50, "whiff_rate": 0.40, "bip": 60,
        },
        {
            "pitcher_id": 100, "pitch_type": "CH", "pitches": 150,
            "total_pitches": 1000, "usage_pct": 0.15, "swings": 75,
            "whiffs": 24, "whiff_rate": 0.32, "bip": 40,
        },
    ])


@pytest.fixture()
def hitter_vuln() -> pd.DataFrame:
    """Synthetic hitters: 200 has FF+SL data, 300 is league-avg, 999 has no data."""
    return pd.DataFrame([
        # Batter 200: vulnerable to FF (high whiff), average on SL
        {
            "batter_id": 200, "pitch_type": "FF", "swings": 80,
            "whiff_rate": 0.35, "pitch_family": "fastball",
        },
        {
            "batter_id": 200, "pitch_type": "SL", "swings": 40,
            "whiff_rate": 0.33, "pitch_family": "breaking",
        },
        # Batter 300: exactly league-average rates
        {
            "batter_id": 300, "pitch_type": "FF", "swings": 100,
            "whiff_rate": 0.22, "pitch_family": "fastball",
        },
        {
            "batter_id": 300, "pitch_type": "SL", "swings": 60,
            "whiff_rate": 0.33, "pitch_family": "breaking",
        },
        {
            "batter_id": 300, "pitch_type": "CH", "swings": 50,
            "whiff_rate": 0.32, "pitch_family": "offspeed",
        },
    ])


# ---------------------------------------------------------------------------
# Test math helpers
# ---------------------------------------------------------------------------
class TestLogitInvLogit:
    def test_roundtrip(self) -> None:
        """logit then inv_logit should return the original value."""
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = _inv_logit(_logit(p))
            assert np.isclose(result, p, atol=1e-10)

    def test_clipping(self) -> None:
        """logit(0) and logit(1) should not produce inf."""
        assert np.isfinite(_logit(0.0))
        assert np.isfinite(_logit(1.0))
        # inv_logit of moderate values should stay in (0, 1)
        assert 0 < _inv_logit(-10) < 1
        assert 0 < _inv_logit(10) < 1


# ---------------------------------------------------------------------------
# Test hitter whiff fallback chain
# ---------------------------------------------------------------------------
class TestHitterWhiffFallback:
    def test_direct_match(self, hitter_vuln: pd.DataFrame) -> None:
        """Full reliability with >= 50 swings."""
        whiff, rel = _get_hitter_whiff_with_fallback(
            hitter_vuln, batter_id=200, pitch_type="FF",
            league_whiff=0.22,
        )
        # 80 swings → reliability = 80/50 capped at 1.0
        assert rel == 1.0
        # With full reliability, should match raw whiff rate
        assert np.isclose(whiff, 0.35)

    def test_low_sample_blends(self) -> None:
        """10 swings → 0.2 reliability, blends toward league baseline."""
        small_sample = pd.DataFrame([{
            "batter_id": 400, "pitch_type": "FF", "swings": 10,
            "whiff_rate": 0.50, "pitch_family": "fastball",
        }])
        whiff, rel = _get_hitter_whiff_with_fallback(
            small_sample, batter_id=400, pitch_type="FF",
            league_whiff=0.22,
        )
        assert np.isclose(rel, 10 / 50)
        expected = 0.2 * 0.50 + 0.8 * 0.22
        assert np.isclose(whiff, expected)

    def test_family_fallback(self, hitter_vuln: pd.DataFrame) -> None:
        """ST (sweeper) should use SL data from breaking family."""
        whiff, rel = _get_hitter_whiff_with_fallback(
            hitter_vuln, batter_id=200, pitch_type="ST",
            league_whiff=0.36,
        )
        # Should have non-zero reliability (family match), discounted by 0.5
        assert rel > 0
        assert rel < 1.0
        # Should not be exactly league baseline
        assert whiff != 0.36

    def test_league_fallback(self, hitter_vuln: pd.DataFrame) -> None:
        """Unknown batter → league baseline, reliability = 0."""
        whiff, rel = _get_hitter_whiff_with_fallback(
            hitter_vuln, batter_id=999, pitch_type="FF",
            league_whiff=0.22,
        )
        assert rel == 0.0
        assert whiff == 0.22


# ---------------------------------------------------------------------------
# Test matchup scoring
# ---------------------------------------------------------------------------
class TestScoreMatchup:
    def test_average_hitter_near_zero_lift(
        self, pitcher_arsenal: pd.DataFrame, hitter_vuln: pd.DataFrame,
    ) -> None:
        """League-average hitter should produce lift near zero."""
        result = score_matchup(
            pitcher_id=100, batter_id=300,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=LEAGUE_BASELINES,
        )
        assert abs(result["matchup_k_logit_lift"]) < 0.15
        assert result["n_pitch_types"] == 3

    def test_vulnerable_hitter_positive_lift(
        self, pitcher_arsenal: pd.DataFrame, hitter_vuln: pd.DataFrame,
    ) -> None:
        """High-whiff hitter should produce positive lift."""
        result = score_matchup(
            pitcher_id=100, batter_id=200,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=LEAGUE_BASELINES,
        )
        assert result["matchup_k_logit_lift"] > 0
        assert result["matchup_whiff_rate"] > result["baseline_whiff_rate"]
