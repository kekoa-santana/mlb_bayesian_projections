"""Precompute: Player rankings, breakout candidates."""
from __future__ import annotations

import logging

from precompute import DASHBOARD_DIR, FROM_SEASON

logger = logging.getLogger("precompute.rankings")


def run_player_rankings(
    *,
    health_df: "pd.DataFrame | None" = None,
    pitcher_roles_df: "pd.DataFrame | None" = None,
    from_season: int = FROM_SEASON,
) -> None:
    """Build MLB positional rankings and position eligibility.

    Parameters
    ----------
    health_df : pd.DataFrame or None
        Pre-loaded health scores.  Passed through to ranking functions
        to avoid a disk read when the pipeline already has it in memory.
    pitcher_roles_df : pd.DataFrame or None
        Pre-loaded reliever roles.  Passed through to pitcher rankings
        to avoid a disk read.
    """
    logger.info("=" * 60)
    logger.info("Building MLB positional rankings...")

    try:
        from src.models.player_rankings import rank_all

        rankings = rank_all(
            season=from_season, projection_season=from_season + 1,
            health_df=health_df, pitcher_roles_df=pitcher_roles_df,
        )
        for key, rdf in rankings.items():
            if not rdf.empty:
                fname = f"{key}_rankings.parquet"
                rdf.to_parquet(DASHBOARD_DIR / fname, index=False)
                logger.info("Saved %s: %d rows", fname, len(rdf))

    except Exception:
        logger.exception("Failed to build MLB positional rankings")

    # Hitter position eligibility (multi-position depth chart)
    logger.info("=" * 60)
    logger.info("Building hitter position eligibility...")

    try:
        from src.models.player_rankings import get_hitter_position_eligibility

        elig = get_hitter_position_eligibility(season=from_season, min_starts=10)
        if not elig.empty:
            elig.to_parquet(
                DASHBOARD_DIR / "hitter_position_eligibility.parquet",
                index=False,
            )
            n_players = elig["player_id"].nunique()
            n_rows = len(elig)
            logger.info(
                "Saved hitter_position_eligibility.parquet: "
                "%d rows (%d unique players)",
                n_rows, n_players,
            )
        else:
            logger.warning("No position eligibility data generated")

    except Exception:
        logger.exception("Failed to build position eligibility")

    # Enrich position eligibility with per-position OAA
    try:
        import pandas as pd
        from src.data.db import read_sql as _read_oaa

        elig = pd.read_parquet(DASHBOARD_DIR / "hitter_position_eligibility.parquet")
        oaa_by_pos = _read_oaa("""
            SELECT player_id, position,
                   SUM(outs_above_average) as oaa,
                   SUM(fielding_runs_prevented) as frp
            FROM production.fact_fielding_oaa
            WHERE season >= :min_season
            GROUP BY player_id, position
        """, {"min_season": from_season - 1})

        if not oaa_by_pos.empty and not elig.empty:
            elig = elig.merge(oaa_by_pos, on=["player_id", "position"], how="left")
            elig["oaa"] = elig["oaa"].fillna(0).astype(int)
            elig["frp"] = elig["frp"].fillna(0).astype(int)
            elig.to_parquet(
                DASHBOARD_DIR / "hitter_position_eligibility.parquet", index=False,
            )
            logger.info("Enriched position eligibility with per-position OAA: %d rows", len(elig))

            # Convert per-position OAA to 20-80 fielding grades
            from src.models.scouting_grades import grade_position_fielding
            elig = grade_position_fielding(elig, season=from_season)
            elig.to_parquet(
                DASHBOARD_DIR / "hitter_position_eligibility.parquet", index=False,
            )
            logger.info("Added per-position fielding grades (20-80)")
    except Exception as e:
        logger.warning("Could not enrich position OAA: %s", e)


def run_breakouts(
    *,
    from_season: int = FROM_SEASON,
) -> None:
    """Score hitter and pitcher breakout candidates."""
    # Hitter breakout archetypes
    logger.info("=" * 60)
    logger.info("Scoring hitter breakout archetypes...")

    try:
        from src.models.breakout_model import score_breakout_candidates

        breakouts = score_breakout_candidates(season=from_season, min_pa=150)
        if not breakouts.empty:
            breakouts.to_parquet(
                DASHBOARD_DIR / "hitter_breakout_candidates.parquet",
                index=False,
            )
            logger.info(
                "Saved hitter_breakout_candidates.parquet: %d rows",
                len(breakouts),
            )
        else:
            logger.warning("No breakout candidates generated")

    except Exception:
        logger.exception("Failed to compute hitter breakout archetypes")

    # Pitcher breakout archetypes
    logger.info("=" * 60)
    logger.info("Scoring pitcher breakout archetypes...")

    try:
        from src.models.pitcher_breakout_model import score_pitcher_breakout_candidates

        p_breakouts = score_pitcher_breakout_candidates(season=from_season, min_bf=150)
        if not p_breakouts.empty:
            p_breakouts.to_parquet(
                DASHBOARD_DIR / "pitcher_breakout_candidates.parquet",
                index=False,
            )
            logger.info(
                "Saved pitcher_breakout_candidates.parquet: %d rows",
                len(p_breakouts),
            )
        else:
            logger.warning("No pitcher breakout candidates generated")

    except Exception:
        logger.exception("Failed to compute pitcher breakout archetypes")
