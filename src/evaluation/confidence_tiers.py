"""
Confidence tier system for game-level prop predictions.

Assigns HIGH / MEDIUM / LOW confidence to each prop type based on
backtest performance (CRPS, ECE, Brier improvement over baseline).
Each tier includes a human-readable explanation of WHY.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceTier:
    """Confidence tier assignment for a prop type."""
    prop: str              # e.g. "pitcher_k"
    side: str              # "pitcher" or "batter"
    stat: str              # "k", "bb", "hr", "h", "outs"
    tier: str              # "HIGH", "MEDIUM", "LOW"
    tier_int: int          # 3=HIGH, 2=MEDIUM, 1=LOW
    explanation: str       # Human-readable explanation
    avg_brier: float       # Average Brier score
    ece: float             # Expected Calibration Error
    brier_improvement: float  # % improvement over baseline
    crps: float            # CRPS score (NaN if unavailable)
    best_line: str         # Best-performing line description
    best_win_rate: float   # Win rate at 70%+ confidence on best line


# Tier thresholds
HIGH_ECE_MAX = 0.05
HIGH_BRIER_IMPROVEMENT_MIN = 0.05  # 5% improvement over baseline
MEDIUM_METRICS_NEEDED = 2  # Must beat baseline on at least 2 of 3 metrics
LOW_ECE_MAX = 0.10


# Human-readable explanations per stat
STAT_EXPLANATIONS = {
    ("pitcher", "k"): {
        "HIGH": "Pitcher strikeouts are our strongest prop. K% is the most stable pitcher rate stat (r=0.795 YoY), matchup lifts from whiff rates add real signal, and BF distribution is well-calibrated for starters.",
        "MEDIUM": "Pitcher K predictions are solid but calibration has room to improve. The model beats baseline on most metrics but not all.",
        "LOW": "Pitcher K predictions are underperforming baseline. Investigate model fit or data issues.",
    },
    ("pitcher", "bb"): {
        "HIGH": "Pitcher walks benefit from strong chase rate signal (r=0.84 YoY). BB% is stable enough for Bayesian projection with meaningful matchup adjustments.",
        "MEDIUM": "Pitcher walk predictions beat some baselines but walks are inherently noisier than Ks. Game-to-game variance is high even with a good rate estimate.",
        "LOW": "Pitcher walk predictions don't reliably beat baseline. BB counts per game are low (1-3 typical), making binary over/under calls difficult.",
    },
    ("pitcher", "hr"): {
        "HIGH": "Pitcher HR predictions beat baseline despite being a rare event. Park factors provide strong additional signal.",
        "MEDIUM": "Pitcher HR predictions show some edge but home runs are rare (~3% of BF). With only 0-2 HR per game, even a well-calibrated model has wide uncertainty. Park factors help but game-to-game noise is high.",
        "LOW": "Pitcher HR predictions are lower confidence because home runs are rare events (~3% of batters faced). With only 0-2 HR per game, the signal-to-noise ratio is poor. HR/BF has the lowest YoY correlation (r=0.267) of any stat we track.",
    },
    ("pitcher", "h"): {
        "HIGH": "Pitcher hits allowed predictions surprisingly beat baseline. Shrinkage estimator captures enough signal.",
        "MEDIUM": "Pitcher hits allowed are moderately predictable. We use population shrinkage (no Bayesian posterior) because BABIP is too noisy (r~0.35 YoY) for full modeling.",
        "LOW": "Pitcher hits allowed are difficult to predict. BABIP noise (r~0.35 YoY) means observed hit rates don't reliably predict future rates. Our shrinkage estimator doesn't add enough over simple population average.",
    },
    ("pitcher", "outs"): {
        "HIGH": "Pitcher outs (innings) are well-predicted — workload patterns are stable for starters.",
        "MEDIUM": "Pitcher outs are moderately predictable. Driven primarily by BF distribution quality rather than rate modeling.",
        "LOW": "Pitcher outs predictions underperform. Outs are driven by workload management decisions that don't follow statistical patterns well.",
    },
    ("batter", "k"): {
        "HIGH": "Batter strikeout predictions are strong. Hitter K% is very stable (r=0.795 YoY) and whiff-rate matchup scoring adds meaningful signal even at the single-game level.",
        "MEDIUM": "Batter K predictions show edge but single-game batter props have limited PA (3-5 per game), creating inherent noise even with good rate estimates.",
        "LOW": "Batter K predictions don't reliably beat baseline. With only 3-5 PA per game, the variance in binary K/no-K outcomes is too high for reliable single-game prediction.",
    },
    ("batter", "bb"): {
        "HIGH": "Batter walk predictions beat baseline. Chase rate (r=0.84 YoY) provides strong matchup signal.",
        "MEDIUM": "Batter walk predictions are moderate. Walks are rare per game (0-1 typical) and the small PA count makes single-game prediction noisy.",
        "LOW": "Batter walks are very difficult to predict at the game level. With 0-1 walks typical and only 3-5 PA, there's insufficient signal for reliable over/under calls.",
    },
    ("batter", "hr"): {
        "HIGH": "Batter HR predictions beat baseline — park factors and barrel rate matchups provide real edge.",
        "MEDIUM": "Batter HR predictions show some signal from park factors and barrel rates, but HRs are binary (usually 0 or 1) with high randomness per game.",
        "LOW": "Batter HR predictions are our lowest-confidence prop. A typical batter hits 0 HR in ~85% of games. The 0/1 binary nature plus only 3-5 PA makes this essentially unpredictable at the single-game level.",
    },
    ("batter", "h"): {
        "HIGH": "Batter hits predictions beat baseline despite BABIP noise.",
        "MEDIUM": "Batter hits are moderately predictable. We use shrinkage estimation (no full Bayesian model) because BABIP randomness dominates at the game level.",
        "LOW": "Batter hits predictions don't add much over baseline. BABIP dominates single-game hit outcomes, and 3-5 AB per game provides minimal signal.",
    },
}


def assign_confidence_tiers(
    backtest_summary: pd.DataFrame,
) -> list[ConfidenceTier]:
    """Assign confidence tiers to each prop based on backtest results.

    Parameters
    ----------
    backtest_summary : pd.DataFrame
        Output of run_game_prop_backtest.py with columns:
        prop, side, stat, test_season, n_games, rmse, mae,
        avg_brier, ece, crps, coverage_80, coverage_90,
        baseline_avg_brier (from comparison).

    Returns
    -------
    list[ConfidenceTier]
        One per unique prop.
    """
    tiers = []

    for prop_name in backtest_summary["prop"].unique():
        prop_data = backtest_summary[backtest_summary["prop"] == prop_name]
        side = prop_data["side"].iloc[0]
        stat = prop_data["stat"].iloc[0]

        # Average metrics across folds
        avg_brier = float(prop_data["avg_brier"].mean())
        avg_ece = float(prop_data["ece"].mean()) if "ece" in prop_data.columns else np.nan
        avg_crps = float(prop_data["crps"].mean()) if "crps" in prop_data.columns else np.nan

        # Brier improvement over baseline
        if "baseline_avg_brier" in prop_data.columns:
            baseline_brier = float(prop_data["baseline_avg_brier"].mean())
            if baseline_brier > 0:
                brier_improvement = (baseline_brier - avg_brier) / baseline_brier
            else:
                brier_improvement = 0.0
        else:
            brier_improvement = 0.0

        # Best line performance (find highest win rate at 70%+ confidence)
        best_line = "N/A"
        best_win_rate = 0.0
        # This would need prediction-level data; use summary approximation
        # We'll populate this from prediction files in D2

        # Tier assignment logic
        beats_brier = brier_improvement >= HIGH_BRIER_IMPROVEMENT_MIN
        good_ece = (not np.isnan(avg_ece)) and avg_ece < HIGH_ECE_MAX
        good_crps = not np.isnan(avg_crps)  # If CRPS exists and is finite

        n_good = sum([beats_brier, good_ece, good_crps])

        if beats_brier and good_ece:
            tier = "HIGH"
            tier_int = 3
        elif n_good >= MEDIUM_METRICS_NEEDED:
            tier = "MEDIUM"
            tier_int = 2
        else:
            tier = "LOW"
            tier_int = 1

        # Override: if ECE > LOW_ECE_MAX, cap at LOW
        if not np.isnan(avg_ece) and avg_ece > LOW_ECE_MAX:
            tier = "LOW"
            tier_int = 1

        explanation = STAT_EXPLANATIONS.get((side, stat), {}).get(
            tier, f"{side.title()} {stat} prop: {tier} confidence."
        )

        tiers.append(ConfidenceTier(
            prop=prop_name,
            side=side,
            stat=stat,
            tier=tier,
            tier_int=tier_int,
            explanation=explanation,
            avg_brier=avg_brier,
            ece=avg_ece if not np.isnan(avg_ece) else -1.0,
            brier_improvement=brier_improvement,
            crps=avg_crps if not np.isnan(avg_crps) else -1.0,
            best_line=best_line,
            best_win_rate=best_win_rate,
        ))

    # Sort: HIGH first, then MEDIUM, then LOW
    tiers.sort(key=lambda t: (-t.tier_int, t.prop))

    logger.info(
        "Confidence tiers: %d HIGH, %d MEDIUM, %d LOW",
        sum(1 for t in tiers if t.tier == "HIGH"),
        sum(1 for t in tiers if t.tier == "MEDIUM"),
        sum(1 for t in tiers if t.tier == "LOW"),
    )

    return tiers


def tiers_to_dataframe(tiers: list[ConfidenceTier]) -> pd.DataFrame:
    """Convert tier list to DataFrame for dashboard consumption.

    Returns
    -------
    pd.DataFrame
        Columns: prop, side, stat, tier, tier_int, explanation,
        avg_brier, ece, brier_improvement, crps, best_line, best_win_rate.
    """
    records = []
    for t in tiers:
        records.append({
            "prop": t.prop,
            "side": t.side,
            "stat": t.stat,
            "tier": t.tier,
            "tier_int": t.tier_int,
            "explanation": t.explanation,
            "avg_brier": t.avg_brier,
            "ece": t.ece,
            "brier_improvement": t.brier_improvement,
            "crps": t.crps,
            "best_line": t.best_line,
            "best_win_rate": t.best_win_rate,
        })
    return pd.DataFrame(records)
