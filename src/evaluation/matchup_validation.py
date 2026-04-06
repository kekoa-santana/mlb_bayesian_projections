"""
Validation of the pitch-type matchup model against actual game-level K outcomes.

Compares a baseline prediction (pitcher K rate x BF) against a matchup-adjusted
prediction to measure whether pitch-type vulnerability profiles add signal.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.data.feature_eng import (
    get_cached_game_batter_ks,
    get_cached_pitcher_game_logs,
    get_cached_pitcher_season_totals,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.models.matchup import (
    compute_game_matchup_k_rate,
    score_matchup,
)
logger = logging.getLogger(__name__)



def build_game_predictions(
    season: int,
    starters_only: bool = True,
) -> pd.DataFrame:
    """Build matchup-adjusted and baseline K predictions for every starter game.

    Parameters
    ----------
    season : int
        MLB season year.
    starters_only : bool
        If True, only evaluate starting pitchers.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, season, actual_k, batters_faced,
        pitcher_k_rate, predicted_k_baseline, predicted_k_matchup,
        n_matched, n_total.
    """
    # Load data
    game_logs = get_cached_pitcher_game_logs(season)
    batter_ks = get_cached_game_batter_ks(season)
    pitcher_totals = get_cached_pitcher_season_totals(season)
    pitcher_arsenal = get_pitcher_arsenal(season)
    hitter_vuln = get_hitter_vulnerability(season)

    # Build league baselines from this season's data
    from src.data.league_baselines import get_baselines_dict
    baselines_pt = get_baselines_dict(seasons=[season], recency_weights="equal")

    # Filter to starters if requested (BF >= 9 excludes openers/bullpen games)
    if starters_only:
        game_logs = game_logs[
            (game_logs["is_starter"] == True)  # noqa: E712
            & (game_logs["batters_faced"] >= 9)
        ].copy()

    # Build pitcher season K rates lookup
    pitcher_k_rates = dict(
        zip(pitcher_totals["pitcher_id"], pitcher_totals["k_rate"])
    )

    results = []
    n_games = len(game_logs)

    for i, (_, game_row) in enumerate(game_logs.iterrows()):
        game_pk = game_row["game_pk"]
        pitcher_id = game_row["pitcher_id"]
        actual_k = game_row["strike_outs"]
        bf = game_row["batters_faced"]

        if pd.isna(bf) or bf == 0:
            continue

        # Pitcher's season K rate
        pitcher_k_rate = pitcher_k_rates.get(pitcher_id)
        if pitcher_k_rate is None or pd.isna(pitcher_k_rate) or pitcher_k_rate == 0:
            continue

        # Get per-batter PA for this game and pitcher
        game_batters = batter_ks[
            (batter_ks["game_pk"] == game_pk)
            & (batter_ks["pitcher_id"] == pitcher_id)
        ][["batter_id", "pa"]].copy()

        if len(game_batters) == 0:
            continue

        # Score matchups for each batter in the game
        matchup_pairs = [
            (pitcher_id, int(bid))
            for bid in game_batters["batter_id"].unique()
        ]
        matchup_scores_list = []
        for pid, bid in matchup_pairs:
            score = score_matchup(
                pid, bid, pitcher_arsenal, hitter_vuln, baselines_pt,
            )
            matchup_scores_list.append(score)
        matchup_scores = pd.DataFrame(matchup_scores_list)

        # Compute game-level prediction
        game_result = compute_game_matchup_k_rate(
            float(pitcher_k_rate), game_batters, matchup_scores,
        )

        results.append({
            "game_pk": game_pk,
            "pitcher_id": pitcher_id,
            "season": season,
            "actual_k": int(actual_k),
            "batters_faced": int(bf),
            "pitcher_k_rate": float(pitcher_k_rate),
            "predicted_k_baseline": game_result["predicted_k_baseline"],
            "predicted_k_matchup": game_result["predicted_k_matchup"],
            "n_matched": game_result["n_matched"],
            "n_total": game_result["n_total"],
        })

        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d games", i + 1, n_games)

    df = pd.DataFrame(results)
    logger.info(
        "Built predictions for %d games in %d (starters_only=%s)",
        len(df), season, starters_only,
    )
    return df


def compute_validation_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute validation metrics comparing baseline vs matchup predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of ``build_game_predictions`` with columns: actual_k,
        predicted_k_baseline, predicted_k_matchup.

    Returns
    -------
    dict[str, float]
        Keys: lift_residual_corr, lift_residual_pvalue, paired_t_stat,
        paired_t_pvalue, n_games, pct_matched.
    """
    actual = predictions["actual_k"].values
    baseline = predictions["predicted_k_baseline"].values
    matchup = predictions["predicted_k_matchup"].values

    resid_baseline = actual - baseline
    resid_matchup = actual - matchup

    # Lift-residual correlation: does the matchup adjustment correlate with
    # where the baseline under/over-predicts?
    lift = matchup - baseline  # matchup adjustment (in K units)
    corr_result = stats.pearsonr(lift, resid_baseline)
    lift_residual_corr = float(corr_result[0])
    lift_residual_pvalue = float(corr_result[1])

    # Paired t-test on squared errors
    se_baseline = resid_baseline ** 2
    se_matchup = resid_matchup ** 2
    t_result = stats.ttest_rel(se_baseline, se_matchup)
    paired_t_stat = float(t_result.statistic)
    paired_t_pvalue = float(t_result.pvalue)

    # Match coverage
    pct_matched = float(
        predictions["n_matched"].sum()
        / predictions["n_total"].sum()
    ) if predictions["n_total"].sum() > 0 else 0.0

    return {
        "lift_residual_corr": lift_residual_corr,
        "lift_residual_pvalue": lift_residual_pvalue,
        "paired_t_stat": paired_t_stat,
        "paired_t_pvalue": paired_t_pvalue,
        "n_games": len(predictions),
        "pct_matched": pct_matched,
    }


# NOTE: Not currently wired into any pipeline or backtest runner.
# Kept for future use — standalone walk-forward matchup model validation.
# Run manually when revisiting the matchup scoring system.
def run_matchup_validation(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Run full matchup validation across multiple seasons.

    Parameters
    ----------
    seasons : list[int] | None
        Seasons to validate. Defaults to [2022, 2023, 2024].

    Returns
    -------
    pd.DataFrame
        One row per season with all validation metrics.
    """
    if seasons is None:
        seasons = [2022, 2023, 2024]

    rows = []
    for season in seasons:
        logger.info("Running matchup validation for %d", season)
        predictions = build_game_predictions(season, starters_only=True)

        if len(predictions) < 10:
            logger.warning(
                "Only %d games for %d — skipping metrics", len(predictions), season
            )
            continue

        metrics = compute_validation_metrics(predictions)
        metrics["season"] = season
        rows.append(metrics)

        logger.info(
            "Season %d: lift-residual r=%.4f (p=%.4f), "
            "paired_t=%.4f (p=%.4f), n=%d",
            season,
            metrics["lift_residual_corr"],
            metrics["lift_residual_pvalue"],
            metrics["paired_t_stat"],
            metrics["paired_t_pvalue"],
            metrics["n_games"],
        )

    df = pd.DataFrame(rows)
    col_order = [
        "season", "n_games", "pct_matched",
        "lift_residual_corr", "lift_residual_pvalue",
        "paired_t_stat", "paired_t_pvalue",
    ]
    return df[[c for c in col_order if c in df.columns]]
