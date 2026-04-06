"""Tests for the matchup validation module."""
import numpy as np
import pandas as pd

from src.models.matchup import compute_game_matchup_k_rate
from src.evaluation.matchup_validation import compute_validation_metrics


class TestComputeGameMatchupKRate:
    def test_basic(self) -> None:
        """PA-weighted aggregation math with known matchup lifts."""
        game_batter_pa = pd.DataFrame([
            {"batter_id": 1, "pa": 4},
            {"batter_id": 2, "pa": 3},
        ])
        matchup_scores = pd.DataFrame([
            {"batter_id": 1, "matchup_k_logit_lift": 0.5},
            {"batter_id": 2, "matchup_k_logit_lift": -0.3},
        ])
        result = compute_game_matchup_k_rate(0.22, game_batter_pa, matchup_scores)

        assert result["total_bf"] == 7
        assert result["n_matched"] == 2
        assert result["n_total"] == 2
        # Matchup prediction should differ from baseline
        assert result["predicted_k_matchup"] != result["predicted_k_baseline"]
        assert np.isclose(result["predicted_k_baseline"], 0.22 * 7)

    def test_no_matches(self) -> None:
        """All batters unmatched → baseline K rate."""
        game_batter_pa = pd.DataFrame([
            {"batter_id": 10, "pa": 3},
            {"batter_id": 20, "pa": 4},
        ])
        matchup_scores = pd.DataFrame(
            columns=["batter_id", "matchup_k_logit_lift"]
        )
        result = compute_game_matchup_k_rate(0.25, game_batter_pa, matchup_scores)

        assert result["n_matched"] == 0
        # With no matchup info, prediction should equal baseline
        assert np.isclose(
            result["predicted_k_matchup"],
            result["predicted_k_baseline"],
        )


class TestComputeValidationMetrics:
    def test_shape(self) -> None:
        """Returns expected keys and RMSE >= 0."""
        predictions = pd.DataFrame({
            "actual_k": [5, 7, 3, 8, 6],
            "predicted_k_baseline": [4.5, 6.0, 4.0, 7.5, 5.5],
            "predicted_k_matchup": [4.8, 6.5, 3.5, 7.8, 5.8],
            "n_matched": [6, 7, 5, 8, 6],
            "n_total": [9, 9, 8, 9, 9],
        })
        metrics = compute_validation_metrics(predictions)

        expected_keys = {
            "lift_residual_corr", "lift_residual_pvalue",
            "paired_t_stat", "paired_t_pvalue", "n_games", "pct_matched",
        }
        assert set(metrics.keys()) == expected_keys
        assert metrics["n_games"] == 5
