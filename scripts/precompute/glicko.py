"""Precompute: Glicko-2 player ratings from PA-level data."""
from __future__ import annotations

import logging

import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.glicko")


def run_glicko(*, from_season: int = FROM_SEASON) -> None:
    """Compute Glicko-2 player ratings and save to dashboard."""
    from src.data.queries import get_pa_outcomes
    from src.models.player_glicko import compute_ratings, get_current_ratings

    logger.info("=" * 60)
    logger.info("Computing Glicko-2 player ratings...")

    try:
        # 1. Load PA data (all seasons for full history)
        pa_data = get_pa_outcomes()
        if pa_data.empty:
            logger.warning("No PA data found — skipping Glicko computation")
            return

        # 2. Compute ratings
        ratings, history = compute_ratings(pa_data)

        # 3. Extract current ratings with player names
        try:
            from src.data.db import read_sql
            player_info = read_sql(
                "SELECT player_id, player_name FROM production.dim_player"
            )
        except Exception:
            player_info = None

        current = get_current_ratings(ratings, player_info=player_info)
        logger.info("Glicko-2: %d players rated", len(current))

        # 4. Split into batters and pitchers by PA role
        # A player's primary role is determined by whether they appear more
        # often as batter_id or pitcher_id in the PA data. This correctly
        # handles two-way players and historical pitchers who batted.
        batter_counts = pa_data["batter_id"].value_counts()
        pitcher_counts = pa_data["pitcher_id"].value_counts()

        all_pids = set(current["player_id"])
        pitcher_role_ids: set[int] = set()
        for pid in all_pids:
            n_batting = batter_counts.get(pid, 0)
            n_pitching = pitcher_counts.get(pid, 0)
            if n_pitching > n_batting:
                pitcher_role_ids.add(pid)

        batter_df = current[~current["player_id"].isin(pitcher_role_ids)].copy()
        pitcher_df = current[current["player_id"].isin(pitcher_role_ids)].copy()

        # 5. Filter to recently active players and add percentile ranks
        # Only include players with games in the last 2 seasons to avoid
        # retired/historical players compressing the percentile distribution.
        recent_seasons = pa_data[pa_data["season"] >= from_season - 1]
        recent_batter_ids = set(recent_seasons["batter_id"].unique())
        recent_pitcher_ids = set(recent_seasons["pitcher_id"].unique())

        batter_df = batter_df[batter_df["player_id"].isin(recent_batter_ids)].copy()
        pitcher_df = pitcher_df[pitcher_df["player_id"].isin(recent_pitcher_ids)].copy()

        for df, label in [(batter_df, "batter"), (pitcher_df, "pitcher")]:
            if not df.empty:
                df["mu_percentile"] = df["mu"].rank(pct=True)
                id_col = f"{label}_id"
                df = df.rename(columns={"player_id": id_col,
                                         "player_name": f"{label}_name"})

                df.to_parquet(
                    DASHBOARD_DIR / f"{label}_glicko.parquet", index=False,
                )
                top5 = df.nlargest(5, "mu")
                names = ", ".join(
                    f'{r.get(f"{label}_name", "?")} ({r["mu"]:.0f})'
                    for _, r in top5.iterrows()
                )
                logger.info(
                    "Saved %s_glicko.parquet: %d players (top 5: %s)",
                    label, len(df), names,
                )

        # 6. Save history (recent seasons for dashboard charts)
        if not history.empty:
            recent = history[history["season"] >= from_season - 1]
            recent.to_parquet(
                DASHBOARD_DIR / "glicko_history.parquet", index=False,
            )
            logger.info("Saved glicko_history.parquet: %d rows", len(recent))

    except Exception:
        logger.exception("Failed to compute Glicko-2 ratings")
