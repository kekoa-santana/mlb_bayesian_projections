"""Precompute: Prospect data, MiLB translations, readiness, rankings, comps."""
from __future__ import annotations

import logging
import shutil

import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, PROJECT_ROOT, save_dashboard_parquet

logger = logging.getLogger("precompute.prospects")


def run(
    *,
    from_season: int = FROM_SEASON,
) -> None:
    """Copy MiLB translated prospect data and build readiness/rankings/comps."""
    logger.info("=" * 60)
    logger.info("Copying MiLB translated prospect data...")

    milb_files = {
        "milb_translated_batters.parquet": "MiLB translated batters",
        "milb_translated_pitchers.parquet": "MiLB translated pitchers",
        "milb_batter_factors.parquet": "MiLB batter translation factors",
        "milb_pitcher_factors.parquet": "MiLB pitcher translation factors",
    }
    cached_dir = PROJECT_ROOT / "data" / "cached"
    for fname, label in milb_files.items():
        src_path = cached_dir / fname
        if src_path.exists():
            shutil.copy2(src_path, DASHBOARD_DIR / fname)
            _df = pd.read_parquet(src_path)
            logger.info("Copied %s: %d rows", label, len(_df))
        else:
            logger.warning("Missing %s -- run build_milb_translations.py first", fname)

    # Prospect readiness scores + FanGraphs rankings
    logger.info("=" * 60)
    logger.info("Building prospect readiness scores...")

    try:
        from src.models.mlb_readiness import train_readiness_model, score_prospects
        from src.data.db import read_sql

        target_season = from_season + 1  # e.g. 2026

        # Train model and score prospects
        bundle = train_readiness_model()
        prospects_df = score_prospects(projection_season=target_season)
        logger.info(
            "Readiness model: AUC=%.3f, scored %d prospects",
            bundle["train_auc"], len(prospects_df),
        )

        # Load FanGraphs rankings for the target season
        rankings = read_sql(
            "SELECT player_id, player_name, org, position, overall_rank, "
            "org_rank, future_value, risk, eta, source "
            "FROM production.dim_prospect_ranking "
            f"WHERE season = {target_season}",
            {},
        )
        if not rankings.empty:
            # Prefer fg_report, deduplicate
            rankings = rankings.sort_values(
                "source", ascending=True  # fg_report before fg_updated
            ).drop_duplicates("player_id", keep="first")
            logger.info("FanGraphs rankings: %d prospects for %d", len(rankings), target_season)
        else:
            logger.warning("No FanGraphs rankings for season %d", target_season)

        # Merge readiness scores with rankings
        if not prospects_df.empty:
            readiness_cols = [
                "player_id", "name", "pos_group", "primary_position",
                "max_level", "max_level_num", "readiness_score", "readiness_tier",
                "wtd_k_pct", "wtd_bb_pct", "wtd_iso", "k_bb_diff", "sb_rate",
                "youngest_age_rel", "min_age", "career_milb_pa",
                "n_above", "total_at_pos_in_org",
            ]
            available_cols = [c for c in readiness_cols if c in prospects_df.columns]
            prospect_out = prospects_df[available_cols].copy()

            if not rankings.empty:
                fg_cols = ["player_id", "org", "overall_rank", "org_rank", "risk", "eta"]
                fg_available = [c for c in fg_cols if c in rankings.columns]
                prospect_out = prospect_out.merge(
                    rankings[fg_available], on="player_id", how="left",
                )

            save_dashboard_parquet(prospect_out, "prospect_readiness.parquet")
            logger.info("Saved prospect_readiness.parquet: %d rows", len(prospect_out))
        else:
            logger.warning("No prospect readiness scores generated")

    except Exception:
        logger.exception("Failed to build prospect readiness scores")

    # TDD Prospect Rankings (composite score)
    logger.info("=" * 60)
    logger.info("Building TDD prospect rankings...")

    try:
        from src.models.prospect_ranking import rank_prospects, rank_pitching_prospects

        prospect_rankings = rank_prospects(projection_season=from_season + 1)
        if not prospect_rankings.empty:
            save_dashboard_parquet(prospect_rankings, "prospect_rankings.parquet")
            logger.info(
                "Saved prospect_rankings.parquet: %d rows", len(prospect_rankings),
            )
        else:
            logger.warning("No batting prospect rankings generated")

        pitching_prospect_rankings = rank_pitching_prospects(
            projection_season=from_season + 1,
        )
        if not pitching_prospect_rankings.empty:
            save_dashboard_parquet(pitching_prospect_rankings, "pitching_prospect_rankings.parquet")
            logger.info(
                "Saved pitching_prospect_rankings.parquet: %d rows",
                len(pitching_prospect_rankings),
            )
        else:
            logger.warning("No pitching prospect rankings generated")

    except Exception:
        logger.exception("Failed to build prospect rankings")

    # Prospect-to-MLB player comps
    logger.info("=" * 60)
    logger.info("Building prospect-to-MLB player comps...")

    try:
        from src.models.prospect_comps import find_all_comps

        comps = find_all_comps(projection_season=from_season + 1)
        for key, cdf in comps.items():
            if not cdf.empty:
                fname = f"prospect_comps_{key}.parquet"
                cdf.to_parquet(DASHBOARD_DIR / fname, index=False)
                logger.info("Saved %s: %d rows", fname, len(cdf))

    except Exception:
        logger.exception("Failed to build prospect comps")
