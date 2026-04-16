"""Pin numerical behavior of per-line binary + calibration helpers.

These helpers replace 6 inline loops across game_k_validation,
game_sim_validation, and game_prop_validation. The tests below build
synthetic prediction frames that match each call-site shape and assert
the helper reproduces the original inline math byte-for-byte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.evaluation.metrics import (
    compute_ece,
    compute_log_loss,
    compute_mce,
    compute_per_line_binary_metrics,
    compute_per_line_calibration,
    compute_temperature,
)


def _synthetic_predictions(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    expected_k = rng.normal(6.0, 1.5, n).clip(0, None)
    std_k = np.full(n, 1.8)
    actual_k = rng.poisson(expected_k).astype(int)

    df = pd.DataFrame({
        "actual_k": actual_k,
        "expected_k": expected_k,
        "std_k": std_k,
    })
    for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
        col = f"p_over_{line:.1f}".replace(".", "_")
        # Rough p(over) from normal approximation
        from scipy.stats import norm
        df[col] = norm.sf(line, loc=expected_k, scale=std_k)
    return df


def test_per_line_binary_matches_inline_game_k():
    """game_k_validation-shaped loop produces identical output."""
    df = _synthetic_predictions(seed=1)
    lines = [3.5, 4.5, 5.5, 6.5, 7.5]

    inline_brier: dict[float, float] = {}
    inline_log: dict[float, float] = {}
    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        y_true = (df["actual_k"] > line).astype(float).values
        y_prob = df[col].values
        inline_brier[line] = float(brier_score_loss(y_true, y_prob))
        inline_log[line] = compute_log_loss(y_prob, y_true)

    got = compute_per_line_binary_metrics(df, "actual_k", lines)

    assert got["brier_scores"] == inline_brier
    assert got["log_losses"] == inline_log
    assert got["avg_brier"] == float(np.mean(list(inline_brier.values())))
    assert got["avg_log_loss"] == float(np.mean(list(inline_log.values())))


def test_per_line_binary_clip_and_skip_degenerate():
    """clip + skip_degenerate flags preserve game_prop_validation behavior."""
    df = pd.DataFrame({
        "actual_k": [0, 0, 0, 0],
        "p_over_0_5": [0.3, 1.2, -0.1, 0.5],
        "p_over_5_5": [0.05, 0.02, 0.03, 0.01],
    })
    # All actual_k = 0 < 0.5 → y_true is all zeros → degenerate; should skip.
    got = compute_per_line_binary_metrics(
        df, "actual_k", [0.5, 5.5], clip=True, skip_degenerate=True,
    )
    assert got["brier_scores"] == {}  # both lines degenerate
    assert np.isnan(got["avg_brier"])


def test_per_line_binary_missing_prob_col_skipped():
    df = _synthetic_predictions(seed=2)
    # Request an extra line whose prob column does NOT exist
    got = compute_per_line_binary_metrics(df, "actual_k", [3.5, 4.5, 99.5])
    assert 99.5 not in got["brier_scores"]
    assert len(got["brier_scores"]) == 2


def test_per_line_calibration_matches_inline_game_k():
    df = _synthetic_predictions(seed=3)
    lines = [3.5, 4.5, 5.5, 6.5, 7.5]

    inline_ece: dict[float, float] = {}
    inline_temp: dict[float, float] = {}
    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        y_true = (df["actual_k"] > line).astype(float).values
        y_prob = df[col].values
        inline_ece[line] = compute_ece(y_prob, y_true)
        inline_temp[line] = compute_temperature(y_prob, y_true)

    got = compute_per_line_calibration(df, "actual_k", lines)

    assert got["ece_per_line"] == inline_ece
    assert got["temperature_per_line"] == inline_temp


def test_per_line_calibration_include_mce():
    df = _synthetic_predictions(seed=4)
    lines = [3.5, 4.5, 5.5]

    got = compute_per_line_calibration(
        df, "actual_k", lines, include_mce=True,
    )

    assert "mce_per_line" in got
    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        y_true = (df["actual_k"] > line).astype(float).values
        y_prob = df[col].values
        assert got["mce_per_line"][line] == compute_mce(y_prob, y_true)


def test_custom_prob_col_fn():
    """Supports alternate templates like 'p_outs_over_<line>'."""
    df = pd.DataFrame({
        "actual_outs": [15, 17, 16, 18, 14],
        "p_outs_over_15_5": [0.4, 0.9, 0.5, 0.95, 0.2],
        "p_outs_over_16_5": [0.3, 0.7, 0.4, 0.8, 0.1],
    })
    got = compute_per_line_binary_metrics(
        df, "actual_outs", [15.5, 16.5],
        prob_col_fn=lambda line: f"p_outs_over_{line:.1f}".replace(".", "_"),
    )
    assert 15.5 in got["brier_scores"]
    assert 16.5 in got["brier_scores"]
