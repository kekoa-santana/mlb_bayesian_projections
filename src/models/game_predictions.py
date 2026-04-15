"""Game-level predictions: moneyline, spread, over/under.

Combines per-pitcher run distributions from the PA-by-PA game simulator
to produce game-level betting market predictions and CRPS evaluation.

Terminology
-----------
- **Moneyline**: pure win probability (does not matter by how much).
- **Spread (run line)**: P(favored team wins by more than the spread).
- **Over/Under (total)**: P(combined runs exceed the posted total).

CRPS (Continuous Ranked Probability Score) is used as the primary
evaluation metric until we have enough historical odds to evaluate
against actual market lines.  CRPS behaves like MAE but respects
the full predictive distribution.

Canonical: ``player_profiles/src/models/game_predictions.py``. Dashboard copy:
``tdd-dashboard/lib/game_predictions.py`` — keep identical (``sync_lib.py``).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def crps_sample(samples: np.ndarray, observed: float) -> float:
    """Compute CRPS for a set of forecast samples against an observation.

    Uses the representation:
        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the forecast distribution.

    Parameters
    ----------
    samples : np.ndarray
        1-D array of Monte Carlo draws from the predictive distribution.
    observed : float
        The actual observed value.

    Returns
    -------
    float
        CRPS value (lower is better, 0 = perfect).
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n == 0:
        return np.nan

    # E|X - y|
    term1 = np.mean(np.abs(samples - observed))

    # E|X - X'| via the CDF-based formula:
    # CRPS = E|X-y| - 0.5 * E|X-X'|
    # For the spread term, use: E|X-X'| = (2/(n^2)) * sum_{i<j} (x_(j) - x_(i))
    # which simplifies to: (2/(n^2)) * sum_i (2i - n - 1) * x_(i)  [1-indexed]
    # This is always non-negative for sorted samples.
    sorted_s = np.sort(samples)
    idx = np.arange(1, n + 1, dtype=np.float64)
    spread = (2.0 / (n * n)) * np.sum((2 * idx - n - 1) * sorted_s)

    return float(term1 - 0.5 * spread)


# ---------------------------------------------------------------------------
# Game-level prediction from two pitcher sim runs
# ---------------------------------------------------------------------------

def compute_game_predictions(
    away_runs_samples: np.ndarray,
    home_runs_samples: np.ndarray,
    spread_lines: list[float] | None = None,
    total_lines: list[float] | None = None,
) -> dict[str, Any]:
    """Derive moneyline, spread, and over/under from run distributions.

    The simulator gives us ``runs_samples`` for each pitcher, which
    represent *runs scored by the opposing lineup* against that pitcher.
    Therefore:
        - ``away_pitcher.runs_samples`` → runs scored by **home** team
        - ``home_pitcher.runs_samples`` → runs scored by **away** team

    The caller is responsible for passing the correct arrays (see
    ``build_game_predictions_from_sims``).

    Parameters
    ----------
    away_runs_samples : np.ndarray
        Runs scored by the away team (shape ``(n_sims,)``).
    home_runs_samples : np.ndarray
        Runs scored by the home team (shape ``(n_sims,)``).
    spread_lines : list[float], optional
        Spread lines to evaluate (away perspective, e.g. ``[-1.5, 1.5]``).
        Default ``[-1.5, 1.5]``.
    total_lines : list[float], optional
        Over/under total lines.  Default ``[6.5, 7.5, 8.5, 9.5]``.

    Returns
    -------
    dict[str, Any]
        Keys: ``moneyline``, ``spread``, ``over_under``, ``summary``.
    """
    if spread_lines is None:
        spread_lines = [-1.5, 1.5]
    if total_lines is None:
        total_lines = [6.5, 7.5, 8.5, 9.5]

    n = min(len(away_runs_samples), len(home_runs_samples))
    away = away_runs_samples[:n].astype(np.float64)
    home = home_runs_samples[:n].astype(np.float64)
    margin = away - home  # positive = away wins
    total = away + home

    # --- Moneyline ---
    away_wins = np.sum(margin > 0)
    home_wins = np.sum(margin < 0)
    draws = np.sum(margin == 0)
    # Assign half of draws to each side (discrete sim artifact)
    p_away = (away_wins + 0.5 * draws) / n
    p_home = (home_wins + 0.5 * draws) / n

    moneyline = {
        "p_away_win": float(p_away),
        "p_home_win": float(p_home),
        "away_wins": int(away_wins),
        "home_wins": int(home_wins),
        "draws": int(draws),
    }

    # --- Spread ---
    spread_results = []
    for line in spread_lines:
        # "Away -1.5" means away must win by more than 1.5
        p_cover = float(np.mean(margin > line))
        spread_results.append({
            "line": line,
            "p_away_cover": p_cover,
            "p_home_cover": 1.0 - p_cover,
        })

    # --- Over/Under ---
    ou_results = []
    for line in total_lines:
        p_over = float(np.mean(total > line))
        ou_results.append({
            "line": line,
            "p_over": p_over,
            "p_under": 1.0 - p_over,
        })

    # --- Summary stats ---
    summary = {
        "away_runs_mean": float(np.mean(away)),
        "away_runs_std": float(np.std(away)),
        "away_runs_median": float(np.median(away)),
        "home_runs_mean": float(np.mean(home)),
        "home_runs_std": float(np.std(home)),
        "home_runs_median": float(np.median(home)),
        "total_mean": float(np.mean(total)),
        "total_std": float(np.std(total)),
        "margin_mean": float(np.mean(margin)),
        "margin_std": float(np.std(margin)),
    }

    return {
        "moneyline": moneyline,
        "spread": spread_results,
        "over_under": ou_results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Build game predictions from today's sims + sample arrays
# ---------------------------------------------------------------------------

def build_game_predictions_from_sims(
    sims_df: pd.DataFrame,
    sample_arrays: dict[str, np.ndarray],
    spread_lines: list[float] | None = None,
    total_lines: list[float] | None = None,
) -> pd.DataFrame:
    """Build game-level predictions from pitcher simulation results.

    Parameters
    ----------
    sims_df : pd.DataFrame
        ``todays_sims.parquet`` with columns: game_pk, side, pitcher_id,
        team_abbr, opp_abbr, expected_k, ... .
    sample_arrays : dict[str, np.ndarray]
        From ``pitcher_game_sim_samples.npz``.  Keys like
        ``{game_pk}_{pitcher_id}_runs``.
    spread_lines, total_lines : list[float], optional
        Lines to evaluate.

    Returns
    -------
    pd.DataFrame
        One row per game with moneyline, spread, and O/U predictions.
    """
    if sims_df.empty:
        return pd.DataFrame()

    games = sims_df["game_pk"].unique()
    rows: list[dict[str, Any]] = []

    for gpk in games:
        game_sims = sims_df[sims_df["game_pk"] == gpk]
        away_row = game_sims[game_sims["side"] == "away"]
        home_row = game_sims[game_sims["side"] == "home"]

        if away_row.empty or home_row.empty:
            logger.debug("Game %s missing a side, skipping", gpk)
            continue

        away_pid = int(away_row.iloc[0]["pitcher_id"])
        home_pid = int(home_row.iloc[0]["pitcher_id"])

        # runs_samples for a pitcher = runs scored BY opposing lineup
        # away_pitcher's runs = home team's runs
        # home_pitcher's runs = away team's runs
        home_runs_key = f"{gpk}_{away_pid}_runs"
        away_runs_key = f"{gpk}_{home_pid}_runs"

        if home_runs_key not in sample_arrays or away_runs_key not in sample_arrays:
            logger.debug("Game %s missing runs samples, skipping", gpk)
            continue

        home_runs = sample_arrays[home_runs_key]
        away_runs = sample_arrays[away_runs_key]

        preds = compute_game_predictions(
            away_runs_samples=away_runs,
            home_runs_samples=home_runs,
            spread_lines=spread_lines,
            total_lines=total_lines,
        )

        row: dict[str, Any] = {
            "game_pk": gpk,
            "away_abbr": str(away_row.iloc[0].get("team_abbr", "")),
            "home_abbr": str(home_row.iloc[0].get("team_abbr", "")),
            "away_pitcher_id": away_pid,
            "home_pitcher_id": home_pid,
            "away_pitcher_name": str(away_row.iloc[0].get("pitcher_name", "")),
            "home_pitcher_name": str(home_row.iloc[0].get("pitcher_name", "")),
            # Moneyline
            "p_away_win": preds["moneyline"]["p_away_win"],
            "p_home_win": preds["moneyline"]["p_home_win"],
            # Summary
            "away_runs_mean": preds["summary"]["away_runs_mean"],
            "away_runs_std": preds["summary"]["away_runs_std"],
            "home_runs_mean": preds["summary"]["home_runs_mean"],
            "home_runs_std": preds["summary"]["home_runs_std"],
            "total_mean": preds["summary"]["total_mean"],
            "total_std": preds["summary"]["total_std"],
            "margin_mean": preds["summary"]["margin_mean"],
        }

        # Flatten spread columns
        for sp in preds["spread"]:
            label = f"{sp['line']:+.1f}".replace(".", "_").replace("+", "p").replace("-", "m")
            row[f"p_away_cover_{label}"] = sp["p_away_cover"]

        # Flatten O/U columns
        for ou in preds["over_under"]:
            label = f"{ou['line']:.1f}".replace(".", "_")
            row[f"p_over_{label}"] = ou["p_over"]
            row[f"p_under_{label}"] = ou["p_under"]

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info("Built game predictions for %d games", len(df))
    return df


# ---------------------------------------------------------------------------
# Evaluate predictions against actuals
# ---------------------------------------------------------------------------

def evaluate_game_predictions(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    sample_arrays: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Join predictions to actuals and compute accuracy + CRPS.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of ``build_game_predictions_from_sims``.
    actuals : pd.DataFrame
        Must have columns: game_pk, away_score, home_score.
    sample_arrays : dict[str, np.ndarray], optional
        If provided, computes CRPS for total runs and margin.

    Returns
    -------
    pd.DataFrame
        Merged predictions + actuals + accuracy flags + CRPS.
    """
    if predictions.empty or actuals.empty:
        return pd.DataFrame()

    merged = predictions.merge(actuals, on="game_pk", how="inner")
    if merged.empty:
        return merged

    merged["actual_total"] = merged["away_score"] + merged["home_score"]
    merged["actual_margin"] = merged["away_score"] - merged["home_score"]

    # Moneyline accuracy
    merged["away_won"] = merged["actual_margin"] > 0
    merged["ml_predicted_away"] = merged["p_away_win"] > 0.5
    merged["ml_correct"] = merged["ml_predicted_away"] == merged["away_won"]

    # Spread accuracy (standard -1.5 run line)
    if "p_away_cover_m1_5" in merged.columns:
        merged["away_covered_m1_5"] = merged["actual_margin"] > -1.5
        merged["spread_m1_5_predicted_cover"] = merged["p_away_cover_m1_5"] > 0.5
        merged["spread_m1_5_correct"] = (
            merged["spread_m1_5_predicted_cover"] == merged["away_covered_m1_5"]
        )

    if "p_away_cover_p1_5" in merged.columns:
        merged["away_covered_p1_5"] = merged["actual_margin"] > 1.5
        merged["spread_p1_5_predicted_cover"] = merged["p_away_cover_p1_5"] > 0.5
        merged["spread_p1_5_correct"] = (
            merged["spread_p1_5_predicted_cover"] == merged["away_covered_p1_5"]
        )

    # O/U accuracy for each total line
    for col in merged.columns:
        if col.startswith("p_over_") and not col.startswith("p_over_bb") and not col.startswith("p_over_h_"):
            suffix = col.replace("p_over_", "")
            try:
                line_val = float(suffix.replace("_", "."))
            except ValueError:
                continue
            merged[f"actual_over_{suffix}"] = merged["actual_total"] > line_val
            merged[f"ou_{suffix}_predicted_over"] = merged[col] > 0.5
            merged[f"ou_{suffix}_correct"] = (
                merged[f"ou_{suffix}_predicted_over"] == merged[f"actual_over_{suffix}"]
            )

    # CRPS (if sample arrays provided)
    if sample_arrays is not None:
        total_crps_vals = []
        margin_crps_vals = []
        for _, row in merged.iterrows():
            gpk = row["game_pk"]
            away_pid = row.get("away_pitcher_id")
            home_pid = row.get("home_pitcher_id")

            away_runs_key = f"{gpk}_{home_pid}_runs"
            home_runs_key = f"{gpk}_{away_pid}_runs"

            if away_runs_key in sample_arrays and home_runs_key in sample_arrays:
                away_r = sample_arrays[away_runs_key].astype(np.float64)
                home_r = sample_arrays[home_runs_key].astype(np.float64)
                n = min(len(away_r), len(home_r))
                total_samples = away_r[:n] + home_r[:n]
                margin_samples = away_r[:n] - home_r[:n]
                total_crps_vals.append(crps_sample(total_samples, row["actual_total"]))
                margin_crps_vals.append(crps_sample(margin_samples, row["actual_margin"]))
            else:
                total_crps_vals.append(np.nan)
                margin_crps_vals.append(np.nan)

        merged["crps_total"] = total_crps_vals
        merged["crps_margin"] = margin_crps_vals

    return merged
