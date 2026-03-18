"""
Health/durability score for playing time projections.

Replaces blunt age penalties with data-driven health adjustments
using IL stint history from fact_player_status_timeline.

Score = weighted composite of 4 sub-components:
  - Total IL days (40%): recency-weighted average days per season
  - Stint frequency (25%): recency-weighted average stints per season
  - Seasons breadth (20%): fraction of lookback seasons with any IL
  - Recency (15%): IL days in the most recent completed season

Scores range from 0.0 (worst) to 1.0 (best/healthy).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"

# Recency weights: most recent season gets highest weight
RECENCY_WEIGHTS = [5, 4, 3, 2, 1]
LOOKBACK_YEARS = 5

# Sub-component weights
_DAYS_WEIGHT = 0.40
_FREQ_WEIGHT = 0.25
_BREADTH_WEIGHT = 0.20
_RECENCY_WEIGHT = 0.15

# Normalization caps (per-season weighted-average basis)
_DAYS_CAP = 100.0       # weighted avg IL days per season → floor at 100+
_STINTS_CAP = 2.0       # weighted avg stints per season → floor at 2+
_RECENCY_CAP = 100.0    # IL days in most recent season → floor at 100+

# 2020 shortened season scaling
_2020_SCALE = 162.0 / 60.0

# Defaults for players without sufficient IL data
DEFAULT_NO_IL_HISTORY = 0.85   # player exists but no IL stints found
DEFAULT_TRUE_ROOKIE = 0.80     # no MLB history at all

# Label thresholds
HEALTH_LABELS = [
    (0.85, "Iron Man"),
    (0.70, "Durable"),
    (0.50, "Average"),
    (0.30, "Questionable"),
    (0.00, "Injury Prone"),
]

# IL status types in the database
IL_STATUS_TYPES = ("IL-7", "IL-10", "IL-15", "IL-60")


def _get_health_label(score: float) -> str:
    """Map health score to descriptive label."""
    for threshold, label in HEALTH_LABELS:
        if score >= threshold:
            return label
    return "Injury Prone"


def _query_il_history(
    from_season: int,
    player_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Query IL stint history from fact_player_status_timeline.

    Parameters
    ----------
    from_season : int
        Most recent completed season (lookback window ends here).
    player_ids : list[int], optional
        Restrict to these player IDs. If None, query all.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season, days_on_status, status_type.
    """
    start_season = from_season - LOOKBACK_YEARS + 1
    status_list = ", ".join(f"'{s}'" for s in IL_STATUS_TYPES)

    pid_filter = ""
    if player_ids is not None and len(player_ids) > 0:
        pid_str = ", ".join(str(int(p)) for p in player_ids)
        pid_filter = f"AND player_id IN ({pid_str})"

    query = f"""
        SELECT player_id,
               season,
               COALESCE(days_on_status, 0) AS days_on_status,
               status_type
        FROM production.fact_player_status_timeline
        WHERE status_type IN ({status_list})
          AND season BETWEEN {start_season} AND {from_season}
          {pid_filter}
        ORDER BY player_id, season, status_start_date
    """
    return read_sql(query, {})


def compute_health_scores(
    from_season: int,
    all_player_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Compute health/durability scores for all players.

    Parameters
    ----------
    from_season : int
        Most recent completed season. Lookback window is
        [from_season - 4, from_season] (5 seasons).
    all_player_ids : list[int], optional
        Player IDs to score. If None, scores all players with IL data
        plus returns defaults for those without.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, health_score, health_label,
        days_score, freq_score, breadth_score, recency_score,
        total_il_days, total_stints, seasons_with_il.
    """
    cache_path = CACHE_DIR / f"health_scores_{from_season}.parquet"
    if cache_path.exists():
        logger.info("Loading cached health scores from %s", cache_path)
        return pd.read_parquet(cache_path)

    start_season = from_season - LOOKBACK_YEARS + 1
    lookback_seasons = list(range(start_season, from_season + 1))

    # Query IL history
    il_df = _query_il_history(from_season, all_player_ids)
    logger.info(
        "IL history: %d stints for %d players (seasons %d-%d)",
        len(il_df),
        il_df["player_id"].nunique() if not il_df.empty else 0,
        start_season,
        from_season,
    )

    # Build recency weight lookup: most recent season = highest weight
    season_weights = {}
    for i, s in enumerate(sorted(lookback_seasons)):
        season_weights[s] = RECENCY_WEIGHTS[LOOKBACK_YEARS - 1 - i]
    # Result: oldest season = weight 1, most recent = weight 5

    # Aggregate per player-season
    if not il_df.empty:
        # Scale 2020 days
        il_df["adj_days"] = il_df["days_on_status"].copy()
        mask_2020 = il_df["season"] == 2020
        il_df.loc[mask_2020, "adj_days"] = (
            il_df.loc[mask_2020, "days_on_status"] * _2020_SCALE
        ).clip(upper=182)  # cap at full season + some

        per_season = (
            il_df.groupby(["player_id", "season"])
            .agg(
                season_days=("adj_days", "sum"),
                season_stints=("player_id", "count"),
            )
            .reset_index()
        )
    else:
        per_season = pd.DataFrame(
            columns=["player_id", "season", "season_days", "season_stints"]
        )

    # Compute scores per player
    total_weight = sum(RECENCY_WEIGHTS[:LOOKBACK_YEARS])
    players_with_il = set(per_season["player_id"].unique()) if not per_season.empty else set()

    # Determine all player IDs to score
    if all_player_ids is not None:
        score_ids = set(int(p) for p in all_player_ids)
    else:
        score_ids = players_with_il

    results = []
    for pid in score_ids:
        player_seasons = per_season[per_season["player_id"] == pid]

        if player_seasons.empty and pid not in players_with_il:
            # No IL data at all — use default
            results.append({
                "player_id": int(pid),
                "health_score": DEFAULT_NO_IL_HISTORY,
                "health_label": _get_health_label(DEFAULT_NO_IL_HISTORY),
                "days_score": DEFAULT_NO_IL_HISTORY,
                "freq_score": DEFAULT_NO_IL_HISTORY,
                "breadth_score": DEFAULT_NO_IL_HISTORY,
                "recency_score": DEFAULT_NO_IL_HISTORY,
                "total_il_days": 0,
                "total_stints": 0,
                "seasons_with_il": 0,
            })
            continue

        # Build per-season data (fill missing seasons with 0)
        season_data = {}
        for s in lookback_seasons:
            row = player_seasons[player_seasons["season"] == s]
            if not row.empty:
                season_data[s] = {
                    "days": float(row["season_days"].sum()),
                    "stints": int(row["season_stints"].sum()),
                }
            else:
                season_data[s] = {"days": 0.0, "stints": 0}

        # Sub-component 1: Total IL days (recency-weighted average)
        weighted_days = sum(
            season_data[s]["days"] * season_weights[s] for s in lookback_seasons
        )
        weighted_avg_days = weighted_days / total_weight
        days_score = max(1.0 - weighted_avg_days / _DAYS_CAP, 0.0)

        # Sub-component 2: Stint frequency (recency-weighted average)
        weighted_stints = sum(
            season_data[s]["stints"] * season_weights[s] for s in lookback_seasons
        )
        weighted_avg_stints = weighted_stints / total_weight
        freq_score = max(1.0 - weighted_avg_stints / _STINTS_CAP, 0.0)

        # Sub-component 3: Seasons breadth (fraction with any IL)
        n_seasons_with_il = sum(
            1 for s in lookback_seasons if season_data[s]["days"] > 0
        )
        breadth_score = 1.0 - n_seasons_with_il / LOOKBACK_YEARS

        # Sub-component 4: Recency (most recent season severity)
        recent_days = season_data[from_season]["days"]
        recency_score = max(1.0 - recent_days / _RECENCY_CAP, 0.0)

        # Combined score
        health = (
            _DAYS_WEIGHT * days_score
            + _FREQ_WEIGHT * freq_score
            + _BREADTH_WEIGHT * breadth_score
            + _RECENCY_WEIGHT * recency_score
        )

        total_il = sum(season_data[s]["days"] for s in lookback_seasons)
        total_st = sum(season_data[s]["stints"] for s in lookback_seasons)

        results.append({
            "player_id": int(pid),
            "health_score": round(health, 4),
            "health_label": _get_health_label(health),
            "days_score": round(days_score, 4),
            "freq_score": round(freq_score, 4),
            "breadth_score": round(breadth_score, 4),
            "recency_score": round(recency_score, 4),
            "total_il_days": int(total_il),
            "total_stints": int(total_st),
            "seasons_with_il": int(n_seasons_with_il),
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        logger.info(
            "Health scores: %d players, mean=%.3f, "
            "Iron Man=%d, Durable=%d, Average=%d, Questionable=%d, Injury Prone=%d",
            len(result_df),
            result_df["health_score"].mean(),
            (result_df["health_label"] == "Iron Man").sum(),
            (result_df["health_label"] == "Durable").sum(),
            (result_df["health_label"] == "Average").sum(),
            (result_df["health_label"] == "Questionable").sum(),
            (result_df["health_label"] == "Injury Prone").sum(),
        )

        # Cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(cache_path, index=False)
        logger.info("Cached health scores to %s", cache_path)

    return result_df
