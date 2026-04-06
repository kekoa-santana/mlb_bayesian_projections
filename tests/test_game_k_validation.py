"""Tests for game K validation module (Step 15)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.game_k_validation import (
    compare_to_baselines,
    compute_calibration_by_line,
    compute_game_k_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def predictions() -> pd.DataFrame:
    """Synthetic predictions DataFrame mimicking predict_game_batch output."""
    rng = np.random.default_rng(42)
    n = 200

    actual_k = rng.poisson(5.5, size=n)
    expected_k = actual_k + rng.normal(0, 1.5, size=n)
    expected_k = np.clip(expected_k, 0.5, 15)

    df = pd.DataFrame({
        "game_pk": range(1000, 1000 + n),
        "pitcher_id": rng.choice([100, 200, 300, 400], size=n),
        "season": 2024,
        "actual_k": actual_k,
        "actual_bf": rng.integers(18, 30, size=n),
        "expected_k": expected_k,
        "std_k": rng.uniform(1.5, 3.0, size=n),
        "pitcher_k_rate_mean": rng.uniform(0.18, 0.30, size=n),
        "bf_mu": rng.uniform(20, 25, size=n),
        "n_matched_batters": rng.integers(0, 9, size=n),
    })

    # Add P(over) columns for standard lines
    for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
        col = f"p_over_{line:.1f}".replace(".", "_")
        # Simulate rough probabilities (higher expected → higher P(over))
        logit = (expected_k - line) / 2.0
        p = 1.0 / (1.0 + np.exp(-logit))
        p = np.clip(p + rng.normal(0, 0.05, size=n), 0.01, 0.99)
        df[col] = p

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestComputeCalibrationShape:
    def test_output_columns(self, predictions: pd.DataFrame) -> None:
        """Output DataFrame has expected columns and bins."""
        result = compute_calibration_by_line(predictions)
        expected_cols = {
            "line", "bin_center", "predicted_prob",
            "actual_freq", "n_games", "calibration_error",
        }
        assert expected_cols.issubset(set(result.columns))
        # Should have rows for each line
        assert len(result) > 0

    def test_calibration_error_nonneg(self, predictions: pd.DataFrame) -> None:
        """Calibration error should be non-negative."""
        result = compute_calibration_by_line(predictions)
        assert (result["calibration_error"] >= 0).all()


class TestComputeGameKMetricsKeys:
    def test_all_keys_present(self, predictions: pd.DataFrame) -> None:
        """Returns all expected metric keys."""
        result = compute_game_k_metrics(predictions)
        expected_keys = {
            "brier_scores", "avg_brier", "calibration_df",
            "coverage_50", "coverage_80", "coverage_90",
            "log_score", "n_games",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert result["n_games"] == 200
        assert result["avg_brier"] > 0

    def test_empty_predictions(self) -> None:
        """Empty predictions return NaN metrics."""
        empty = pd.DataFrame()
        result = compute_game_k_metrics(empty)
        assert result["n_games"] == 0
        assert np.isnan(result["avg_brier"])


class TestCompareToBaselinesShape:
    def test_four_baselines(self, predictions: pd.DataFrame) -> None:
        """Returns 4 rows (one per baseline) with metric columns."""
        result = compare_to_baselines(predictions)
        assert len(result) == 4
        assert set(result["baseline"]) == {
            "naive", "poisson", "model_no_matchup", "full_model",
        }
        assert "avg_brier" in result.columns

    def test_all_metrics_finite(self, predictions: pd.DataFrame) -> None:
        """All baseline metrics should be finite."""
        result = compare_to_baselines(predictions)
        assert result["avg_brier"].notna().all()
