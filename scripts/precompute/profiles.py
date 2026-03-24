"""Precompute: Arsenal, vulnerability, archetypes, zones."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.profiles")


def run_arsenal_vuln(
    *,
    from_season: int = FROM_SEASON,
    seasons: list[int] = SEASONS,
) -> None:
    """Load and save matchup profiles (arsenal, vulnerability, strength)."""
    from src.data.feature_eng import (
        get_hitter_strength,
        get_hitter_vulnerability,
        get_pitcher_arsenal,
    )

    logger.info("=" * 60)
    logger.info("Loading matchup profiles for %d...", from_season)

    pitcher_arsenal = get_pitcher_arsenal(from_season)
    pitcher_arsenal.to_parquet(DASHBOARD_DIR / "pitcher_arsenal.parquet", index=False)
    logger.info("Saved pitcher arsenal: %d rows", len(pitcher_arsenal))

    hitter_vuln = get_hitter_vulnerability(from_season)
    hitter_vuln.to_parquet(DASHBOARD_DIR / "hitter_vuln.parquet", index=False)
    logger.info("Saved hitter vulnerability: %d rows", len(hitter_vuln))

    hitter_str = get_hitter_strength(from_season)
    hitter_str.to_parquet(DASHBOARD_DIR / "hitter_str.parquet", index=False)
    logger.info("Saved hitter strength: %d rows", len(hitter_str))

    # Career-aggregated vulnerability (weighted across all seasons)
    vuln_frames = []
    for s in seasons:
        v = get_hitter_vulnerability(s)
        vuln_frames.append(v)
    all_vuln = pd.concat(vuln_frames, ignore_index=True)

    # Aggregate: sum raw counts, recompute rates
    career_vuln = all_vuln.groupby(["batter_id", "batter_stand", "pitch_type"]).agg(
        pitches=("pitches", "sum"),
        swings=("swings", "sum"),
        whiffs=("whiffs", "sum"),
        out_of_zone_pitches=("out_of_zone_pitches", "sum"),
        chase_swings=("chase_swings", "sum"),
        called_strikes=("called_strikes", "sum"),
        csw=("csw", "sum"),
        bip=("bip", "sum"),
        hard_hits=("hard_hits", "sum"),
        barrels_proxy=("barrels_proxy", "sum"),
    ).reset_index()
    career_vuln["whiff_rate"] = career_vuln["whiffs"] / career_vuln["swings"].replace(0, np.nan)
    career_vuln["chase_rate"] = career_vuln["chase_swings"] / career_vuln["out_of_zone_pitches"].replace(0, np.nan)
    career_vuln["csw_pct"] = career_vuln["csw"] / career_vuln["pitches"].replace(0, np.nan)
    career_vuln["pitch_family"] = career_vuln["pitch_type"].map(
        hitter_vuln.set_index("pitch_type")["pitch_family"].to_dict()
        if "pitch_family" in hitter_vuln.columns else {}
    )
    # xwoba_contact: weighted average by BIP across seasons
    xwoba_rows = all_vuln[all_vuln["xwoba_contact"].notna() & (all_vuln["bip"] > 0)]
    if not xwoba_rows.empty:
        xwoba_career = xwoba_rows.groupby(["batter_id", "pitch_type"]).apply(
            lambda g: (g["xwoba_contact"] * g["bip"]).sum() / g["bip"].sum()
            if g["bip"].sum() > 0 else np.nan,
            include_groups=False,
        ).reset_index(name="xwoba_contact")
        career_vuln = career_vuln.merge(xwoba_career, on=["batter_id", "pitch_type"], how="left")
    else:
        career_vuln["xwoba_contact"] = np.nan

    career_vuln.to_parquet(DASHBOARD_DIR / "hitter_vuln_career.parquet", index=False)
    logger.info("Saved career hitter vulnerability: %d rows", len(career_vuln))

    # Career-aggregated strength
    str_frames = []
    for s in seasons:
        st_df = get_hitter_strength(s)
        str_frames.append(st_df)
    all_str = pd.concat(str_frames, ignore_index=True)

    career_str = all_str.groupby(["batter_id", "batter_stand", "pitch_type"]).agg(
        bip=("bip", "sum"),
        barrels_proxy=("barrels_proxy", "sum"),
        hard_hits=("hard_hits", "sum"),
    ).reset_index()
    career_str["barrel_rate_contact"] = career_str["barrels_proxy"] / career_str["bip"].replace(0, np.nan)
    career_str["hard_hit_rate"] = career_str["hard_hits"] / career_str["bip"].replace(0, np.nan)
    career_str["pitch_family"] = career_str["pitch_type"].map(
        hitter_str.set_index("pitch_type")["pitch_family"].to_dict()
        if "pitch_family" in hitter_str.columns else {}
    )
    # xwoba_contact: weighted average by BIP
    xwoba_str_rows = all_str[all_str["xwoba_contact"].notna() & (all_str["bip"] > 0)]
    if not xwoba_str_rows.empty:
        xwoba_str_career = xwoba_str_rows.groupby(["batter_id", "pitch_type"]).apply(
            lambda g: (g["xwoba_contact"] * g["bip"]).sum() / g["bip"].sum()
            if g["bip"].sum() > 0 else np.nan,
            include_groups=False,
        ).reset_index(name="xwoba_contact")
        career_str = career_str.merge(xwoba_str_career, on=["batter_id", "pitch_type"], how="left")
    else:
        career_str["xwoba_contact"] = np.nan

    career_str.to_parquet(DASHBOARD_DIR / "hitter_str_career.parquet", index=False)
    logger.info("Saved career hitter strength: %d rows", len(career_str))


def run_zones(
    *,
    from_season: int = FROM_SEASON,
    seasons: list[int] = SEASONS,
) -> None:
    """Compute location grids (pitcher heatmaps + hitter zone profiles)."""
    from src.data.queries import (
        get_pitcher_location_grid, get_hitter_zone_grid,
        get_pitcher_pitch_locations,
    )

    logger.info("=" * 60)
    logger.info("Computing location grids for %d...", from_season)

    pitcher_loc = get_pitcher_location_grid(from_season)
    pitcher_loc.to_parquet(DASHBOARD_DIR / "pitcher_location_grid.parquet", index=False)
    logger.info("Saved pitcher location grid: %d rows (%d pitchers)",
                len(pitcher_loc), pitcher_loc["pitcher_id"].nunique())

    # Raw pitch coordinates for KDE density charts
    pitch_locs = get_pitcher_pitch_locations(from_season)
    # Store plate_x/plate_z as float32 to keep file size reasonable
    for col in ("plate_x", "plate_z"):
        pitch_locs[col] = pitch_locs[col].astype("float32")
    pitch_locs.to_parquet(DASHBOARD_DIR / "pitcher_pitch_locations.parquet", index=False)
    logger.info("Saved raw pitch locations: %d pitches (%d pitchers)",
                len(pitch_locs), pitch_locs["pitcher_id"].nunique())

    hitter_zone = get_hitter_zone_grid(from_season)
    hitter_zone.to_parquet(DASHBOARD_DIR / "hitter_zone_grid.parquet", index=False)
    logger.info("Saved hitter zone grid: %d rows (%d batters)",
                len(hitter_zone), hitter_zone["batter_id"].nunique())

    # Career-aggregated hitter zone grid (sum counts across all seasons)
    zone_frames = []
    for s in seasons:
        try:
            zf = get_hitter_zone_grid(s)
            zone_frames.append(zf)
        except Exception as e:
            logger.warning("Hitter zone grid for %d failed: %s", s, e)
    if zone_frames:
        all_zones = pd.concat(zone_frames, ignore_index=True)
        sum_cols = ["pitches", "swings", "whiffs", "called_strikes", "bip",
                    "xwoba_sum", "xwoba_count", "hard_hits", "barrels"]
        sum_cols = [c for c in sum_cols if c in all_zones.columns]
        career_zones = all_zones.groupby(
            ["batter_id", "batter_name", "batter_stand", "pitch_type", "grid_row", "grid_col"]
        )[sum_cols].sum().reset_index()
        # Keep most recent name per batter
        latest_names = all_zones.sort_values("pitches", ascending=False).drop_duplicates(
            "batter_id"
        )[["batter_id", "batter_name"]]
        career_zones = career_zones.drop(columns=["batter_name"]).merge(
            latest_names, on="batter_id", how="left"
        )
        career_zones.to_parquet(DASHBOARD_DIR / "hitter_zone_grid_career.parquet", index=False)
        logger.info("Saved career hitter zone grid: %d rows (%d batters)",
                    len(career_zones), career_zones["batter_id"].nunique())


def run_archetypes(
    *,
    from_season: int = FROM_SEASON,
    seasons: list[int] = SEASONS,
) -> None:
    """Compute archetype-based matchup data."""
    logger.info("=" * 60)
    logger.info("Computing archetype-based matchup data for %d...", from_season)

    try:
        from src.data.pitch_archetypes import (
            get_pitch_archetype_offerings,
            get_pitch_archetype_clusters,
        )
        from src.data.league_baselines import get_baselines_by_archetype_stand

        pitcher_offerings = get_pitch_archetype_offerings(from_season)
        pitcher_offerings.to_parquet(DASHBOARD_DIR / "pitcher_offerings.parquet", index=False)
        logger.info("Saved pitcher offerings with archetypes: %d rows", len(pitcher_offerings))

        cluster_metadata = get_pitch_archetype_clusters()
        cluster_metadata.to_parquet(DASHBOARD_DIR / "pitcher_cluster_metadata.parquet", index=False)
        logger.info("Saved pitcher cluster metadata: %d archetypes", len(cluster_metadata))

        baselines_arch = get_baselines_by_archetype_stand(from_season)
        baselines_arch.to_parquet(DASHBOARD_DIR / "baselines_arch.parquet", index=False)
        logger.info("Saved league baselines by archetype: %d rows", len(baselines_arch))
    except Exception as e:
        logger.warning("Archetype pitcher data failed: %s", e)

    try:
        from src.data.feature_eng import get_hitter_vulnerability_by_archetype

        hitter_vuln_arch = get_hitter_vulnerability_by_archetype(from_season)
        hitter_vuln_arch.to_parquet(DASHBOARD_DIR / "hitter_vuln_arch.parquet", index=False)
        logger.info("Saved hitter vulnerability by archetype: %d rows", len(hitter_vuln_arch))

        # Career-aggregated archetype vulnerability
        arch_frames = []
        for s in seasons:
            try:
                df = get_hitter_vulnerability_by_archetype(s)
                arch_frames.append(df)
            except Exception as e:
                logger.warning("Archetype vuln for %d failed: %s", s, e)
        if arch_frames:
            all_arch = pd.concat(arch_frames, ignore_index=True)
            sum_cols = ["swings", "whiffs", "out_of_zone_pitches", "chase_swings", "csw"]
            sum_cols = [c for c in sum_cols if c in all_arch.columns]
            career_arch = all_arch.groupby(
                ["batter_id", "pitch_archetype"]
            )[sum_cols].sum().reset_index()
            if "swings" in career_arch.columns:
                career_arch["whiff_rate"] = (
                    career_arch["whiffs"] / career_arch["swings"].clip(lower=1)
                )
            if "out_of_zone_pitches" in career_arch.columns:
                career_arch["chase_rate"] = (
                    career_arch["chase_swings"] / career_arch["out_of_zone_pitches"].clip(lower=1)
                )
            career_arch.to_parquet(DASHBOARD_DIR / "hitter_vuln_arch_career.parquet", index=False)
            logger.info("Saved career hitter archetype vulnerability: %d rows", len(career_arch))
    except Exception as e:
        logger.warning("Archetype hitter data failed: %s", e)

    # Player-level archetype clustering (hitter + pitcher archetypes)
    logger.info("=" * 60)
    logger.info("Exporting player archetype assignments...")
    try:
        from src.data.player_clustering import export_for_dashboard as export_archetypes

        arch_paths = export_archetypes(
            export_season=from_season,
            output_dir=DASHBOARD_DIR,
            force_rebuild=True,
        )
        logger.info("Exported player archetypes: %s", list(arch_paths.keys()))
    except Exception as e:
        logger.warning("Player archetype export failed: %s", e)

    # Archetype matchup matrix (pitcher type vs hitter type outcomes)
    logger.info("=" * 60)
    logger.info("Building archetype matchup matrix...")
    try:
        from src.data.archetype_matchups import (
            build_archetype_matchup_matrix,
            export_archetype_matchups_for_dashboard,
        )

        arch_matrix = build_archetype_matchup_matrix(force_rebuild=True)
        logger.info("Built archetype matchup matrix: %d pairs", len(arch_matrix))

        arch_path = export_archetype_matchups_for_dashboard(output_dir=DASHBOARD_DIR)
        logger.info("Exported archetype matchup matrix to %s", arch_path)
    except Exception as e:
        logger.warning("Archetype matchup matrix failed: %s", e)
