"""Precompute: Arsenal, vulnerability, archetypes, zones, matchup baselines."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS, save_dashboard_parquet

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
    save_dashboard_parquet(pitcher_arsenal, "pitcher_arsenal.parquet")
    logger.info("Saved pitcher arsenal: %d rows", len(pitcher_arsenal))

    hitter_vuln = get_hitter_vulnerability(from_season)
    save_dashboard_parquet(hitter_vuln, "hitter_vuln.parquet")
    logger.info("Saved hitter vulnerability: %d rows", len(hitter_vuln))

    hitter_str = get_hitter_strength(from_season)
    save_dashboard_parquet(hitter_str, "hitter_str.parquet")
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

    save_dashboard_parquet(career_vuln, "hitter_vuln_career.parquet")
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

    save_dashboard_parquet(career_str, "hitter_str_career.parquet")
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
    save_dashboard_parquet(pitcher_loc, "pitcher_location_grid.parquet")
    logger.info("Saved pitcher location grid: %d rows (%d pitchers)",
                len(pitcher_loc), pitcher_loc["pitcher_id"].nunique())

    # Raw pitch coordinates for KDE density charts
    pitch_locs = get_pitcher_pitch_locations(from_season)
    # Store plate_x/plate_z as float32 to keep file size reasonable
    for col in ("plate_x", "plate_z"):
        pitch_locs[col] = pitch_locs[col].astype("float32")
    save_dashboard_parquet(pitch_locs, "pitcher_pitch_locations.parquet")
    logger.info("Saved raw pitch locations: %d pitches (%d pitchers)",
                len(pitch_locs), pitch_locs["pitcher_id"].nunique())

    hitter_zone = get_hitter_zone_grid(from_season)
    save_dashboard_parquet(hitter_zone, "hitter_zone_grid.parquet")
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
        save_dashboard_parquet(career_zones, "hitter_zone_grid_career.parquet")
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
        save_dashboard_parquet(pitcher_offerings, "pitcher_offerings.parquet")
        logger.info("Saved pitcher offerings with archetypes: %d rows", len(pitcher_offerings))

        cluster_metadata = get_pitch_archetype_clusters()
        save_dashboard_parquet(cluster_metadata, "pitcher_cluster_metadata.parquet")
        logger.info("Saved pitcher cluster metadata: %d archetypes", len(cluster_metadata))

        baselines_arch = get_baselines_by_archetype_stand(from_season)
        save_dashboard_parquet(baselines_arch, "baselines_arch.parquet")
        logger.info("Saved league baselines by archetype: %d rows", len(baselines_arch))
    except Exception as e:
        logger.warning("Archetype pitcher data failed: %s", e)

    try:
        from src.data.feature_eng import get_hitter_vulnerability_by_archetype

        hitter_vuln_arch = get_hitter_vulnerability_by_archetype(from_season)
        save_dashboard_parquet(hitter_vuln_arch, "hitter_vuln_arch.parquet")
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
            save_dashboard_parquet(career_arch, "hitter_vuln_arch_career.parquet")
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


def run_matchup_advantage_data(
    *,
    from_season: int = FROM_SEASON,
    seasons: list[int] = SEASONS,
) -> None:
    """Precompute platoon splits, league baselines, and signal scaling factors."""
    from src.data.feature_eng import get_cached_season_totals_by_pitcher_hand
    from src.data.queries import (
        get_league_platoon_baselines,
        get_season_totals_with_age,
        get_pitcher_observed_profile,
    )

    logger.info("=" * 60)
    logger.info("Building matchup advantage data...")

    # ------------------------------------------------------------------
    # 1. Batter platoon splits parquet
    # ------------------------------------------------------------------
    platoon_df = get_cached_season_totals_by_pitcher_hand(from_season)
    if platoon_df.empty:
        logger.warning("No platoon data for season %d", from_season)
        return

    # Also get overall rates per batter (for shrinkage baseline)
    overall = (
        platoon_df.groupby("batter_id")
        .agg(overall_pa=("pa", "sum"), overall_k=("k", "sum"), overall_bb=("bb", "sum"))
        .reset_index()
    )
    overall["overall_k_rate"] = overall["overall_k"] / overall["overall_pa"].clip(lower=1)
    overall["overall_bb_rate"] = overall["overall_bb"] / overall["overall_pa"].clip(lower=1)

    # Keep per (batter_id, pitch_hand) rows with overall rates merged
    platoon_out = platoon_df[["batter_id", "pitch_hand", "k_rate", "bb_rate", "pa"]].copy()
    platoon_out = platoon_out.merge(
        overall[["batter_id", "overall_k_rate", "overall_bb_rate", "overall_pa"]],
        on="batter_id",
        how="left",
    )

    platoon_path = DASHBOARD_DIR / "batter_platoon_splits.parquet"
    platoon_out.to_parquet(platoon_path, index=False)
    logger.info("Saved batter platoon splits: %d rows → %s", len(platoon_out), platoon_path)

    # ------------------------------------------------------------------
    # 2. League platoon baselines + GB/FB rates
    # ------------------------------------------------------------------
    baseline_seasons = [s for s in seasons if s >= 2022]
    baselines = get_league_platoon_baselines(baseline_seasons)
    logger.info(
        "League platoon baselines: K same=%.3f opp=%.3f, BB same=%.3f opp=%.3f",
        baselines["platoon_k_logit"].get("same", 0),
        baselines["platoon_k_logit"].get("opposite", 0),
        baselines["platoon_bb_logit"].get("same", 0),
        baselines["platoon_bb_logit"].get("opposite", 0),
    )

    # ------------------------------------------------------------------
    # 3. Signal scaling factors (trajectory + glicko)
    # ------------------------------------------------------------------
    # Load pitcher GB% and batter GB/FB rates for trajectory scaling
    try:
        pitcher_obs = get_pitcher_observed_profile(from_season)
        batter_totals = get_season_totals_with_age(from_season)
    except Exception as e:
        logger.warning("Could not load pitcher/batter profiles for scaling: %s", e)
        pitcher_obs = pd.DataFrame()
        batter_totals = pd.DataFrame()

    # Also save pitcher gb_pct for dashboard use
    if not pitcher_obs.empty and "gb_pct" in pitcher_obs.columns:
        pitcher_gb = pitcher_obs[["pitcher_id", "gb_pct"]].dropna(subset=["gb_pct"])
        pitcher_gb_path = DASHBOARD_DIR / "pitcher_gb_pct.parquet"
        pitcher_gb.to_parquet(pitcher_gb_path, index=False)
        logger.info("Saved pitcher GB%%: %d rows", len(pitcher_gb))

    # Compute trajectory scale from observed distributions
    trajectory_scale = 5.0  # fallback
    if (
        not pitcher_obs.empty
        and not batter_totals.empty
        and "gb_pct" in pitcher_obs.columns
        and "gb_rate" in batter_totals.columns
        and "fb_rate" in batter_totals.columns
    ):
        lg_gb = baselines.get("lg_gb_rate", 0.446)
        lg_fb = baselines.get("lg_fb_rate", 0.321)

        p_gb_excess = pitcher_obs["gb_pct"].dropna() - lg_gb
        b_fb_excess = batter_totals["fb_rate"].dropna() - lg_fb

        # Cross-product std: sample ~500 random matchups
        rng = np.random.default_rng(42)
        n_sample = min(500, len(p_gb_excess) * len(b_fb_excess))
        p_idx = rng.choice(len(p_gb_excess), size=n_sample, replace=True)
        b_idx = rng.choice(len(b_fb_excess), size=n_sample, replace=True)
        raw_traj = p_gb_excess.values[p_idx] * b_fb_excess.values[b_idx]
        traj_std = float(np.std(raw_traj))
        if traj_std > 1e-6:
            # Target: K lift typically has std ~0.15-0.25 on logit scale
            target_std = 0.15
            trajectory_scale = target_std / traj_std
        logger.info("Trajectory scale: %.2f (raw std=%.4f)", trajectory_scale, traj_std)

    # Compute glicko scale from observed distributions
    glicko_scale = 1.0 / 1000.0  # fallback
    try:
        pitcher_glicko_path = DASHBOARD_DIR / "pitcher_glicko.parquet"
        batter_glicko_path = DASHBOARD_DIR / "batter_glicko.parquet"
        if pitcher_glicko_path.exists() and batter_glicko_path.exists():
            p_glicko = pd.read_parquet(pitcher_glicko_path)
            b_glicko = pd.read_parquet(batter_glicko_path)
            if "mu" in p_glicko.columns and "mu" in b_glicko.columns:
                rng = np.random.default_rng(42)
                n_sample = min(500, len(p_glicko) * len(b_glicko))
                p_idx = rng.choice(len(p_glicko), size=n_sample, replace=True)
                b_idx = rng.choice(len(b_glicko), size=n_sample, replace=True)
                raw_gap = p_glicko["mu"].values[p_idx] - b_glicko["mu"].values[b_idx]
                gap_std = float(np.std(raw_gap))
                if gap_std > 1e-6:
                    target_std = 0.15
                    glicko_scale = target_std / gap_std
                logger.info("Glicko scale: %.6f (raw std=%.1f)", glicko_scale, gap_std)
    except Exception as e:
        logger.warning("Could not compute Glicko scale: %s", e)

    # ------------------------------------------------------------------
    # Write baselines + scales JSON
    # ------------------------------------------------------------------
    baselines["trajectory_scale"] = trajectory_scale
    baselines["glicko_scale"] = glicko_scale

    baselines_path = DASHBOARD_DIR / "matchup_baselines.json"
    with open(baselines_path, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)
    logger.info("Saved matchup baselines → %s", baselines_path)
