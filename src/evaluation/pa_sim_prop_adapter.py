"""
Adapter: convert PA-sim backtest output into the legacy game-prop schema.

``build_game_sim_predictions`` in ``game_sim_validation.py`` returns one
row per (game, pitcher) with columns for every pitcher stat it tracks
(``expected_k``, ``std_k``, ``p_k_over_{line}``, and the same for
``bb``/``h``/``hr``/``outs``).  The legacy prop validator emits one
DataFrame *per stat* with the schema expected by
``compute_game_prop_metrics`` and ``compute_stratified_metrics``:

    game_pk, pitcher_id, season, actual_{sn}, expected_{sn}, std_{sn},
    p_over_{line}   (dots replaced with underscores, no stat prefix)

plus context columns (``umpire_lift``, ``weather_lift``, ``park_factor``,
``pitcher_rate_mean``, ``matchup_logit_lift``, ``rest_bucket``,
``bf_mu``) when available.

This adapter splits the unified game-sim frame into per-stat frames with
that exact schema so the same downstream metrics/stratification code can
score both engines side-by-side.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.game_prop_validation import (
    PITCHER_PROP_CONFIGS,
    compute_game_prop_metrics,
    run_full_game_prop_backtest,
)
from src.evaluation.game_sim_validation import build_game_sim_predictions

logger = logging.getLogger(__name__)


# Stats the PA sim covers and the actual-column names used by each engine.
_PA_SIM_STATS: tuple[str, ...] = ("k", "bb", "h", "hr", "outs")


def _line_col_legacy(line: float) -> str:
    """Legacy engine column name, e.g. 3.5 -> 'p_over_3_5'."""
    return f"p_over_{line:.1f}".replace(".", "_")


def _line_col_pa_sim(stat: str, line: float) -> str:
    """PA-sim engine column name, e.g. ('k', 3.5) -> 'p_k_over_3.5'."""
    return f"p_{stat}_over_{line:.1f}"


def pa_sim_frame_to_legacy_schema(
    sim_df: pd.DataFrame,
    stat: str,
    lines: list[float],
) -> pd.DataFrame:
    """Project a unified PA-sim frame onto the legacy per-stat schema.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Output of ``build_game_sim_predictions`` — one row per
        (game_pk, pitcher_id).
    stat : str
        One of ``k``, ``bb``, ``h``, ``hr``, ``outs``.
    lines : list[float]
        Prop lines to carry over (e.g. ``[3.5, 4.5, 5.5, 6.5, 7.5]``).

    Returns
    -------
    pd.DataFrame
        Per-stat frame with legacy column names. Missing PA-sim columns
        are filled with NaN so downstream scoring degrades gracefully.
    """
    if sim_df.empty:
        return sim_df.copy()

    sn = stat.lower()
    exp_col = f"expected_{sn}"
    std_col = f"std_{sn}"
    act_col = f"actual_{sn}"

    carry_cols = [
        c for c in (
            "game_pk", "pitcher_id", "test_season", "fold_test_season",
            "expected_bf", "std_bf", "actual_bf",
        ) if c in sim_df.columns
    ]
    out = sim_df[carry_cols].copy()

    # Rename season for compatibility with legacy schema.
    if "test_season" in out.columns and "season" not in out.columns:
        out["season"] = out["test_season"]
    elif "fold_test_season" in out.columns and "season" not in out.columns:
        out["season"] = out["fold_test_season"]

    # Per-stat stat columns
    if exp_col in sim_df.columns:
        out[f"expected_{sn}"] = sim_df[exp_col].values
    if std_col in sim_df.columns:
        out[f"std_{sn}"] = sim_df[std_col].values
    if act_col in sim_df.columns:
        out[f"actual_{sn}"] = sim_df[act_col].values

    # bf_mu for stratification
    if "expected_bf" in sim_df.columns:
        out["bf_mu"] = sim_df["expected_bf"].values

    # Rename per-line p(over) columns: p_{stat}_over_X.Y -> p_over_X_Y
    for line in lines:
        src_col = _line_col_pa_sim(sn, line)
        dst_col = _line_col_legacy(line)
        if src_col in sim_df.columns:
            out[dst_col] = sim_df[src_col].values
        else:
            out[dst_col] = np.nan

    # Stratification columns used by compute_stratified_metrics — populate
    # what we can, leave the rest as NaN (compute_stratified_metrics skips
    # missing strata).
    for col in ("umpire_lift", "weather_lift", "park_factor",
                "pitcher_rate_mean", "matchup_logit_lift", "rest_bucket"):
        if col in sim_df.columns and col not in out.columns:
            out[col] = sim_df[col].values

    return out


def build_pa_sim_prop_predictions(
    train_seasons: list[int],
    test_season: int,
    stats: tuple[str, ...] = _PA_SIM_STATS,
    *,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_sims: int = 2000,
    min_bf_game: int = 15,
    random_seed: int = 42,
    sim_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build per-stat prop predictions from the PA simulator for one fold.

    Parameters
    ----------
    train_seasons, test_season, draws, tune, chains, n_sims,
    min_bf_game, random_seed
        Passed through to ``build_game_sim_predictions``. Ignored when
        ``sim_df`` is supplied.
    stats : tuple[str, ...]
        Subset of ``k, bb, h, hr, outs`` to emit.
    sim_df : pd.DataFrame, optional
        Precomputed unified game-sim frame. When supplied, the simulator
        is not re-run (useful for feeding both engines off the same
        simulations).

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{stat: legacy-schema DataFrame}`` for each requested stat.
    """
    if sim_df is None:
        sim_df = build_game_sim_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws,
            tune=tune,
            chains=chains,
            n_sims=n_sims,
            starters_only=True,
            min_bf_game=min_bf_game,
            random_seed=random_seed,
        )

    out: dict[str, pd.DataFrame] = {}
    for stat in stats:
        cfg = PITCHER_PROP_CONFIGS.get(stat)
        if cfg is None:
            continue
        out[stat] = pa_sim_frame_to_legacy_schema(
            sim_df, stat, cfg.default_lines,
        )
    return out


def run_pa_sim_prop_backtest(
    folds: list[tuple[list[int], int]],
    stats: tuple[str, ...] = _PA_SIM_STATS,
    *,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_sims: int = 2000,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Walk-forward PA-sim prop backtest across folds.

    For each fold we run the PA simulator once and project the unified
    frame onto per-stat legacy schemas, computing the same metrics
    ``run_full_game_prop_backtest`` produces for the legacy engine.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per (stat, fold) with the fold-level metrics dict from
        ``compute_game_prop_metrics`` flattened into columns.
    all_predictions : dict[str, pd.DataFrame]
        Concatenated per-stat frames across folds (keyed by stat).
    """
    fold_rows: list[dict[str, Any]] = []
    per_stat_frames: dict[str, list[pd.DataFrame]] = {s: [] for s in stats}

    for train_seasons, test_season in folds:
        logger.info(
            "[pa_sim] Fold train=%s -> test=%d", train_seasons, test_season,
        )
        sim_df = build_game_sim_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws,
            tune=tune,
            chains=chains,
            n_sims=n_sims,
            starters_only=True,
            min_bf_game=min_bf_game,
            random_seed=random_seed,
        )
        if sim_df.empty:
            logger.warning(
                "[pa_sim] no predictions for fold test=%d", test_season,
            )
            continue

        by_stat = build_pa_sim_prop_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            stats=stats,
            sim_df=sim_df,
        )

        for stat, df in by_stat.items():
            if df.empty:
                continue
            df = df.copy()
            df["fold_test_season"] = test_season
            per_stat_frames[stat].append(df)

            cfg = PITCHER_PROP_CONFIGS[stat]
            metrics = compute_game_prop_metrics(cfg, df)
            rec: dict[str, Any] = {
                "stat_name": stat,
                "side": "pitcher",
                "test_season": test_season,
                "engine": "pa_sim",
                "n_games": metrics["n_games"],
                "avg_brier": metrics["avg_brier"],
                "avg_log_loss": metrics["avg_log_loss"],
                "crps": metrics["crps"],
                "ece": metrics["ece"],
                "mce": metrics["mce"],
                "temperature": metrics["temperature"],
                "sharpness_mean_confidence":
                    metrics["sharpness_mean_confidence"],
                "sharpness_pct_actionable_60":
                    metrics["sharpness_pct_actionable_60"],
                "sharpness_pct_actionable_65":
                    metrics["sharpness_pct_actionable_65"],
                "sharpness_pct_actionable_70":
                    metrics["sharpness_pct_actionable_70"],
                "sharpness_entropy": metrics["sharpness_entropy"],
                "coverage_50": metrics["coverage_50"],
                "coverage_80": metrics["coverage_80"],
                "coverage_90": metrics["coverage_90"],
                "coverage_95": metrics["coverage_95"],
            }
            for line, brier in metrics["brier_scores"].items():
                rec[f"brier_{line:.1f}".replace(".", "_")] = brier
            for line, ll in metrics["log_losses"].items():
                rec[f"log_loss_{line:.1f}".replace(".", "_")] = ll
            fold_rows.append(rec)

    summary_df = pd.DataFrame(fold_rows)
    all_preds: dict[str, pd.DataFrame] = {}
    for stat, frames in per_stat_frames.items():
        if frames:
            all_preds[stat] = pd.concat(frames, ignore_index=True)

    return summary_df, all_preds


# ---------------------------------------------------------------------------
# Legacy + PA-sim diff helpers
# ---------------------------------------------------------------------------


def _per_line_brier_slope_ece(
    df: pd.DataFrame, stat: str, lines: list[float],
) -> dict[float, dict[str, float]]:
    """Per-line Brier/slope/ECE on a single-stat predictions DataFrame."""
    from sklearn.metrics import brier_score_loss

    from src.evaluation.metrics import compute_ece

    sn = stat.lower()
    act_col = f"actual_{sn}"
    if act_col not in df.columns:
        return {}

    results: dict[float, dict[str, float]] = {}
    for line in lines:
        col = _line_col_legacy(line)
        if col not in df.columns:
            continue
        y_true = (df[act_col] > line).astype(float).values
        y_prob = np.clip(df[col].values, 0, 1)
        mask = ~np.isnan(y_prob)
        if mask.sum() < 5 or y_true[mask].std() == 0:
            continue
        y_prob = y_prob[mask]
        y_true = y_true[mask]

        brier = float(brier_score_loss(y_true, y_prob))
        ece = float(compute_ece(y_prob, y_true))

        # Slope = correlation-style scaling of prediction vs outcome
        # (logistic regression slope on logit(p) -> outcome).
        p_clipped = np.clip(y_prob, 1e-4, 1 - 1e-4)
        logit_p = np.log(p_clipped / (1 - p_clipped))
        if logit_p.std() > 0:
            slope = float(np.cov(logit_p, y_true, ddof=0)[0, 1]
                          / np.var(logit_p))
        else:
            slope = np.nan

        results[line] = {"brier": brier, "slope": slope, "ece": ece}
    return results


def compute_engine_diff(
    legacy_preds: pd.DataFrame,
    pa_sim_preds: pd.DataFrame,
    stat: str,
    lines: list[float],
) -> pd.DataFrame:
    """Per-line diff: PA-sim minus legacy on Brier/slope/ECE.

    Joined on (game_pk, pitcher_id) so only games both engines scored
    are compared. Positive diff for Brier/ECE = PA-sim worse; positive
    slope diff = PA-sim more responsive.
    """
    if legacy_preds.empty or pa_sim_preds.empty:
        return pd.DataFrame()

    keys = ["game_pk", "pitcher_id"]
    # Ensure both have the same subset of games
    shared = pd.merge(
        legacy_preds[keys].drop_duplicates(),
        pa_sim_preds[keys].drop_duplicates(),
        on=keys,
        how="inner",
    )
    legacy = legacy_preds.merge(shared, on=keys, how="inner")
    pa_sim = pa_sim_preds.merge(shared, on=keys, how="inner")

    legacy_m = _per_line_brier_slope_ece(legacy, stat, lines)
    pa_sim_m = _per_line_brier_slope_ece(pa_sim, stat, lines)

    rows: list[dict[str, Any]] = []
    for line in lines:
        if line not in legacy_m or line not in pa_sim_m:
            continue
        lm = legacy_m[line]
        pm = pa_sim_m[line]
        rows.append({
            "stat": stat,
            "line": line,
            "n_games": len(shared),
            "legacy_brier": lm["brier"],
            "pa_sim_brier": pm["brier"],
            "brier_delta": pm["brier"] - lm["brier"],
            "legacy_slope": lm["slope"],
            "pa_sim_slope": pm["slope"],
            "slope_delta": pm["slope"] - lm["slope"],
            "legacy_ece": lm["ece"],
            "pa_sim_ece": pm["ece"],
            "ece_delta": pm["ece"] - lm["ece"],
        })
    return pd.DataFrame(rows)


__all__ = [
    "build_pa_sim_prop_predictions",
    "compute_engine_diff",
    "pa_sim_frame_to_legacy_schema",
    "run_pa_sim_prop_backtest",
]
