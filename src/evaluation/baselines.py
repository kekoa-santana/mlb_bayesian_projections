"""
Shared baseline projection implementations.

Contains the generic Marcel projection used by all backtest modules.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default regression PA constants per stat (how much to regress toward mean).
# More stable stats need less regression.
DEFAULT_REGRESSION_CONSTANTS = {
    # Hitter stats
    "k_rate": 1200,
    "bb_rate": 1200,
    "gb_rate": 800,
    "fb_rate": 800,
    "hr_per_fb": 1500,
    # Pitcher stats (less stable -> smaller constant)
    "hr_per_bf": 1200,
}

# Default fallback for stats not in the table
DEFAULT_REGRESSION_PA = 1200


def marcel_rate_projection(
    df: pd.DataFrame,
    id_col: str,
    trials_col: str,
    stat_configs: dict[str, Any],
    regression_constants: dict[str, int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Marcel projection: 5/4/3 weighted seasons, regressed to mean.

    Generic implementation that handles both hitter and pitcher rate stats.
    Each stat config must provide at minimum:
    - count_col: str (numerator column name)
    - trials_col: str (denominator column name)
    - rate_col: str (pre-computed rate column name)
    - league_avg: float (league-average rate)
    - likelihood: str ("binomial" or "normal")

    Parameters
    ----------
    df : pd.DataFrame
        Multi-season data. Must contain ``id_col``, ``"season"``,
        ``trials_col``, and each stat's count/rate columns.
    id_col : str
        Player identifier column (e.g. ``"batter_id"`` or ``"pitcher_id"``).
    trials_col : str
        Primary trials column for the weight column output name
        (e.g. ``"pa"`` or ``"batters_faced"``).
    stat_configs : dict[str, Any]
        Mapping of stat key -> config object. Each config must have
        ``count_col``, ``trials_col``, ``rate_col``, ``league_avg``,
        and ``likelihood`` attributes.
    regression_constants : dict[str, int] | None
        Optional per-stat regression PA constants. Falls back to
        ``DEFAULT_REGRESSION_CONSTANTS`` then ``DEFAULT_REGRESSION_PA``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of stat key -> DataFrame with columns:
        ``[id_col, f"marcel_{stat}", "reliability", f"weighted_{trials_col}"]``.
    """
    if regression_constants is None:
        regression_constants = {}

    weights = {0: 5, 1: 4, 2: 3}
    available_seasons = sorted(df["season"].unique(), reverse=True)

    results: dict[str, pd.DataFrame] = {}

    for stat_key, cfg in stat_configs.items():
        # Determine league average
        if cfg.likelihood == "binomial":
            total_count = df[cfg.count_col].sum()
            total_trials = df[cfg.trials_col].sum()
            league_avg = (
                total_count / total_trials if total_trials > 0 else cfg.league_avg
            )
        else:
            league_avg = df[cfg.rate_col].mean()

        # Determine regression constant
        reg = regression_constants.get(
            stat_key,
            DEFAULT_REGRESSION_CONSTANTS.get(stat_key, DEFAULT_REGRESSION_PA),
        )

        # Weight column name for output
        weight_col = f"weighted_{trials_col}"

        records: list[dict[str, Any]] = []
        for player_id, group in df.groupby(id_col):
            weighted_val = 0.0
            weighted_trials = 0.0

            for offset, season in enumerate(available_seasons):
                if offset > 2:
                    break
                w = weights.get(offset, 0)
                row = group[group["season"] == season]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                pa = float(r[cfg.trials_col])

                if cfg.likelihood == "binomial":
                    weighted_val += w * float(r[cfg.count_col])
                else:
                    weighted_val += w * float(r[cfg.rate_col]) * pa
                weighted_trials += w * pa

            if weighted_trials == 0:
                continue

            raw_rate = weighted_val / weighted_trials
            reliability = weighted_trials / (weighted_trials + reg)
            marcel_rate = reliability * raw_rate + (1 - reliability) * league_avg

            records.append({
                id_col: player_id,
                f"marcel_{stat_key}": marcel_rate,
                "reliability": reliability,
                weight_col: weighted_trials,
            })

        results[stat_key] = pd.DataFrame(records)

    return results
