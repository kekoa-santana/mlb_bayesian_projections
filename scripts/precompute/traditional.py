"""Precompute: Traditional stats, aggressiveness, efficiency, historical multi-season."""
from __future__ import annotations

import logging
import shutil

import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, PROJECT_ROOT, SEASONS

logger = logging.getLogger("precompute.traditional")


def run_trad_stats(
    *,
    from_season: int = FROM_SEASON,
) -> None:
    """Compute traditional / actual stats from boxscores + Statcast."""
    from src.data.queries import (
        get_hitter_traditional_stats,
        get_pitcher_traditional_stats,
    )

    logger.info("=" * 60)
    logger.info("Computing traditional stats (%d)...", from_season)
    hitter_trad = get_hitter_traditional_stats(from_season)
    hitter_trad.to_parquet(DASHBOARD_DIR / "hitter_traditional.parquet", index=False)
    logger.info("Saved hitter traditional stats: %d players", len(hitter_trad))

    pitcher_trad = get_pitcher_traditional_stats(from_season)
    pitcher_trad.to_parquet(DASHBOARD_DIR / "pitcher_traditional.parquet", index=False)
    logger.info("Saved pitcher traditional stats: %d players", len(pitcher_trad))


def run_agg_eff(
    *,
    from_season: int = FROM_SEASON,
) -> None:
    """Compute hitter aggressiveness and pitcher efficiency profiles."""
    from src.data.queries import (
        get_hitter_aggressiveness,
        get_pitcher_efficiency,
    )

    logger.info("=" * 60)
    logger.info("Computing hitter aggressiveness profiles...")
    hitter_agg = get_hitter_aggressiveness(from_season)
    hitter_agg.to_parquet(DASHBOARD_DIR / "hitter_aggressiveness.parquet", index=False)
    logger.info("Saved hitter aggressiveness: %d players", len(hitter_agg))

    logger.info("Computing pitcher efficiency profiles...")
    pitcher_eff = get_pitcher_efficiency(from_season)
    pitcher_eff.to_parquet(DASHBOARD_DIR / "pitcher_efficiency.parquet", index=False)
    logger.info("Saved pitcher efficiency: %d players", len(pitcher_eff))


def run_historical_all(
    *,
    seasons: list[int] = SEASONS,
) -> None:
    """Build multi-season datasets (all seasons concatenated for season selector)."""
    from src.data.feature_eng import (
        get_hitter_strength,
        get_hitter_vulnerability,
        get_pitcher_arsenal,
    )
    from src.data.queries import (
        get_hitter_aggressiveness,
        get_hitter_traditional_stats,
        get_hitter_zone_grid,
        get_pitcher_efficiency,
        get_pitcher_location_grid,
        get_pitcher_traditional_stats,
    )

    logger.info("=" * 60)
    logger.info("Building multi-season datasets (2018-2025)...")

    # Copy full_stats parquets to dashboard dir (already have all seasons)
    for fname in ["hitter_full_stats.parquet", "pitcher_full_stats.parquet"]:
        src = PROJECT_ROOT / "data" / "cached" / fname
        if src.exists():
            shutil.copy2(src, DASHBOARD_DIR / fname)
            _fs = pd.read_parquet(DASHBOARD_DIR / fname)
            logger.info("Copied %s: %d rows", fname, len(_fs))
        else:
            logger.warning("Missing %s -- run feature_eng first", fname)

    # Multi-season traditional stats
    trad_frames_h, trad_frames_p = [], []
    for s in seasons:
        try:
            trad_frames_h.append(get_hitter_traditional_stats(s))
        except Exception as e:
            logger.warning("Hitter trad stats %d failed: %s", s, e)
        try:
            trad_frames_p.append(get_pitcher_traditional_stats(s))
        except Exception as e:
            logger.warning("Pitcher trad stats %d failed: %s", s, e)
    if trad_frames_h:
        all_hitter_trad = pd.concat(trad_frames_h, ignore_index=True)
        all_hitter_trad.to_parquet(DASHBOARD_DIR / "hitter_traditional_all.parquet", index=False)
        logger.info("Saved hitter_traditional_all: %d rows", len(all_hitter_trad))
    if trad_frames_p:
        all_pitcher_trad = pd.concat(trad_frames_p, ignore_index=True)
        all_pitcher_trad.to_parquet(DASHBOARD_DIR / "pitcher_traditional_all.parquet", index=False)
        logger.info("Saved pitcher_traditional_all: %d rows", len(all_pitcher_trad))

    # Multi-season aggressiveness + efficiency
    agg_frames, eff_frames = [], []
    for s in seasons:
        try:
            agg_frames.append(get_hitter_aggressiveness(s))
        except Exception as e:
            logger.warning("Hitter aggressiveness %d failed: %s", s, e)
        try:
            eff_frames.append(get_pitcher_efficiency(s))
        except Exception as e:
            logger.warning("Pitcher efficiency %d failed: %s", s, e)
    if agg_frames:
        all_agg = pd.concat(agg_frames, ignore_index=True)
        all_agg.to_parquet(DASHBOARD_DIR / "hitter_aggressiveness_all.parquet", index=False)
        logger.info("Saved hitter_aggressiveness_all: %d rows", len(all_agg))
    if eff_frames:
        all_eff = pd.concat(eff_frames, ignore_index=True)
        all_eff.to_parquet(DASHBOARD_DIR / "pitcher_efficiency_all.parquet", index=False)
        logger.info("Saved pitcher_efficiency_all: %d rows", len(all_eff))

    # Multi-season arsenal
    arsenal_frames = []
    for s in seasons:
        try:
            arsenal_frames.append(get_pitcher_arsenal(s))
        except Exception as e:
            logger.warning("Pitcher arsenal %d failed: %s", s, e)
    if arsenal_frames:
        all_arsenal = pd.concat(arsenal_frames, ignore_index=True)
        all_arsenal.to_parquet(DASHBOARD_DIR / "pitcher_arsenal_all.parquet", index=False)
        logger.info("Saved pitcher_arsenal_all: %d rows", len(all_arsenal))

    # Multi-season vuln + strength
    vuln_frames = []
    for s in seasons:
        try:
            vuln_frames.append(get_hitter_vulnerability(s))
        except Exception as e:
            logger.warning("Hitter vuln %d failed: %s", s, e)
    if vuln_frames:
        all_vuln_seasons = pd.concat(vuln_frames, ignore_index=True)
        all_vuln_seasons.to_parquet(DASHBOARD_DIR / "hitter_vuln_all.parquet", index=False)
        logger.info("Saved hitter_vuln_all: %d rows", len(all_vuln_seasons))

    str_frames = []
    for s in seasons:
        try:
            str_frames.append(get_hitter_strength(s))
        except Exception as e:
            logger.warning("Hitter str %d failed: %s", s, e)
    if str_frames:
        all_str_seasons = pd.concat(str_frames, ignore_index=True)
        all_str_seasons.to_parquet(DASHBOARD_DIR / "hitter_str_all.parquet", index=False)
        logger.info("Saved hitter_str_all: %d rows", len(all_str_seasons))

    # Multi-season location grids (add season column since queries don't include it)
    pitcher_loc_frames, hitter_zone_frames = [], []
    for s in seasons:
        try:
            _pl = get_pitcher_location_grid(s)
            _pl["season"] = s
            pitcher_loc_frames.append(_pl)
        except Exception as e:
            logger.warning("Pitcher location grid %d failed: %s", s, e)
        try:
            _hz = get_hitter_zone_grid(s)
            _hz["season"] = s
            hitter_zone_frames.append(_hz)
        except Exception as e:
            logger.warning("Hitter zone grid %d failed: %s", s, e)
    if pitcher_loc_frames:
        all_pitcher_loc = pd.concat(pitcher_loc_frames, ignore_index=True)
        all_pitcher_loc.to_parquet(DASHBOARD_DIR / "pitcher_location_grid_all.parquet", index=False)
        logger.info("Saved pitcher_location_grid_all: %d rows", len(all_pitcher_loc))
    if hitter_zone_frames:
        all_hitter_zone = pd.concat(hitter_zone_frames, ignore_index=True)
        all_hitter_zone.to_parquet(DASHBOARD_DIR / "hitter_zone_grid_all.parquet", index=False)
        logger.info("Saved hitter_zone_grid_all: %d rows", len(all_hitter_zone))

    logger.info("Multi-season datasets complete.")
