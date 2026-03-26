"""
AutoResearch evaluation module — IMMUTABLE.

Loads cached game data and provides the scoring function.
Do NOT modify this file during experiments.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"


def load_cache(cache_file: str = "fold_2025.pkl") -> dict[str, Any]:
    """Load pre-computed game data from cache.

    Returns
    -------
    dict
        Keys: 'games' (list[dict]), 'exit_model' (ExitModel),
              'train_seasons', 'test_season'.
    """
    path = CACHE_DIR / cache_file
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            "Run: python autoresearch/cache_data.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute composite score and per-metric breakdown.

    Parameters
    ----------
    predictions : pd.DataFrame
        Columns: expected_k, actual_k, std_k, expected_bb, actual_bb,
                 expected_h, actual_h, expected_hr, actual_hr,
                 expected_ip, actual_ip, expected_bf, actual_bf,
                 p_over_3_5, p_over_4_5, p_over_5_5, p_over_6_5, p_over_7_5.

    Returns
    -------
    dict[str, float]
        'composite' (higher = better, main optimization target),
        plus per-metric breakdowns.
    """
    n = len(predictions)
    if n < 100:
        return {"composite": 0.0, "n_games": n, "error": "too_few_games"}

    metrics: dict[str, float] = {"n_games": float(n)}

    # --- Per-stat RMSE ---
    for stat in ("k", "bb", "h", "hr", "outs"):
        exp_col = f"expected_{stat}"
        act_col = f"actual_{stat}"
        if exp_col not in predictions.columns or act_col not in predictions.columns:
            continue
        exp = predictions[exp_col].values.astype(float)
        act = predictions[act_col].values.astype(float)
        errors = exp - act
        metrics[f"{stat}_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
        metrics[f"{stat}_mae"] = float(np.mean(np.abs(errors)))
        metrics[f"{stat}_bias"] = float(np.mean(errors))
        if np.std(act) > 0 and np.std(exp) > 0:
            metrics[f"{stat}_corr"] = float(np.corrcoef(act, exp)[0, 1])

    # IP metrics
    if "expected_ip" in predictions.columns and "actual_ip" in predictions.columns:
        ip_exp = predictions["expected_ip"].values.astype(float)
        ip_act = predictions["actual_ip"].values.astype(float)
        metrics["ip_rmse"] = float(np.sqrt(np.mean((ip_exp - ip_act) ** 2)))

    # --- K Brier scores ---
    k_lines = [3.5, 4.5, 5.5, 6.5, 7.5]
    brier_scores = []
    for line in k_lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            continue
        y_true = (predictions["actual_k"] > line).astype(float).values
        y_prob = predictions[col].values.astype(float)
        brier = float(brier_score_loss(y_true, y_prob))
        metrics[f"k_brier_{line}"] = brier
        brier_scores.append(brier)
    metrics["k_avg_brier"] = float(np.mean(brier_scores)) if brier_scores else 0.25

    # --- Outs Brier scores ---
    outs_lines = [14.5, 15.5, 16.5, 17.5, 18.5]
    outs_brier_scores = []
    for line in outs_lines:
        col = f"p_outs_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns or "actual_outs" not in predictions.columns:
            continue
        y_true = (predictions["actual_outs"] > line).astype(float).values
        y_prob = predictions[col].values.astype(float)
        brier = float(brier_score_loss(y_true, y_prob))
        metrics[f"outs_brier_{line}"] = brier
        outs_brier_scores.append(brier)
    if outs_brier_scores:
        metrics["outs_avg_brier"] = float(np.mean(outs_brier_scores))

    # --- K calibration (coverage accuracy) ---
    if "std_k" in predictions.columns:
        exp_k = predictions["expected_k"].values.astype(float)
        std_k = predictions["std_k"].values.astype(float)
        act_k = predictions["actual_k"].values.astype(float)

        cal_errors = []
        for ci_name, z, nominal in [
            ("50", 0.6745, 0.50),
            ("80", 1.2816, 0.80),
            ("90", 1.6449, 0.90),
        ]:
            lo = exp_k - z * std_k
            hi = exp_k + z * std_k
            actual_cov = float(np.mean((act_k >= lo) & (act_k <= hi)))
            metrics[f"k_cov_{ci_name}"] = actual_cov
            cal_errors.append(abs(actual_cov - nominal))
        metrics["k_cal_error"] = float(np.mean(cal_errors))

    # --- Composite score (higher = better) ---
    # Weights chosen to balance prop prediction quality, multi-stat accuracy,
    # and calibration. Normalization baselines from current sim performance.
    k_brier_score = max(0.0, 1.0 - metrics["k_avg_brier"] / 0.25)
    k_rmse_score = max(0.0, 1.0 - metrics.get("k_rmse", 3.0) / 3.5)
    bb_rmse_score = max(0.0, 1.0 - metrics.get("bb_rmse", 2.0) / 2.5)
    h_rmse_score = max(0.0, 1.0 - metrics.get("h_rmse", 3.0) / 3.5)
    ip_rmse_score = max(0.0, 1.0 - metrics.get("ip_rmse", 2.0) / 2.5)
    outs_rmse_score = max(0.0, 1.0 - metrics.get("outs_rmse", 4.5) / 5.5)
    outs_brier_score = max(0.0, 1.0 - metrics.get("outs_avg_brier", 0.25) / 0.30)
    cal_score = max(0.0, 1.0 - metrics.get("k_cal_error", 0.10) * 5.0)

    composite = (
        0.20 * k_brier_score      # K Brier (prop prediction quality)
        + 0.15 * k_rmse_score     # K RMSE
        + 0.10 * bb_rmse_score    # BB prediction quality
        + 0.10 * h_rmse_score     # H prediction quality
        + 0.05 * ip_rmse_score    # IP prediction quality
        + 0.15 * outs_rmse_score  # Outs RMSE
        + 0.10 * outs_brier_score # Outs Brier (prop quality)
        + 0.15 * cal_score        # Calibration accuracy
    )
    metrics["composite"] = composite

    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics as a human-readable string."""
    lines = [
        f"SCORE: {metrics.get('composite', 0.0):.6f}",
        f"N_GAMES: {int(metrics.get('n_games', 0))}",
        f"K_RMSE: {metrics.get('k_rmse', 0.0):.4f}",
        f"K_BRIER: {metrics.get('k_avg_brier', 0.0):.4f}",
        f"BB_RMSE: {metrics.get('bb_rmse', 0.0):.4f}",
        f"H_RMSE: {metrics.get('h_rmse', 0.0):.4f}",
        f"OUTS_RMSE: {metrics.get('outs_rmse', 0.0):.4f}",
        f"OUTS_BRIER: {metrics.get('outs_avg_brier', 0.0):.4f}",
        f"IP_RMSE: {metrics.get('ip_rmse', 0.0):.4f}",
        f"K_COV_50: {metrics.get('k_cov_50', 0.0):.4f}",
        f"K_COV_80: {metrics.get('k_cov_80', 0.0):.4f}",
        f"K_COV_90: {metrics.get('k_cov_90', 0.0):.4f}",
        f"K_BIAS: {metrics.get('k_bias', 0.0):.4f}",
        f"BB_BIAS: {metrics.get('bb_bias', 0.0):.4f}",
        f"H_BIAS: {metrics.get('h_bias', 0.0):.4f}",
        f"OUTS_BIAS: {metrics.get('outs_bias', 0.0):.4f}",
    ]
    return "\n".join(lines)
