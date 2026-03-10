#!/usr/bin/env python
"""
Pre-compute all data needed by the Streamlit dashboard.

Fits composite hitter + pitcher models (K% and BB% only), enriches with
observed profiles, extracts posterior K% samples, computes BF priors,
and saves everything to data/dashboard/.

Usage
-----
    python scripts/precompute_dashboard_data.py          # full quality
    python scripts/precompute_dashboard_data.py --quick   # fast iteration
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_eng import (
    build_multi_season_hitter_extended,
    build_multi_season_pitcher_extended,
    build_multi_season_pitcher_k_data,
    get_cached_hitter_observed_profile,
    get_cached_pitcher_observed_profile,
    get_cached_sprint_speed,
    get_hitter_strength,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.data.queries import (
    get_game_batter_ks,
    get_game_lineups,
    get_park_factors,
    get_hitter_team_venue,
    get_pitcher_game_logs,
    get_player_teams,
    get_umpire_k_tendencies,
    get_weather_effects,
)
from src.models.bf_model import compute_pitcher_bf_priors
from src.models.counting_projections import (
    project_hitter_counting,
    project_pitcher_counting,
)
from src.models.game_k_model import extract_pitcher_k_rate_samples
from src.models.hitter_projections import (
    fit_all_models as fit_hitter_models,
    project_forward as project_hitter_forward,
)
from src.models.pa_model import compute_hitter_pa_priors
from src.models.pitcher_k_rate_model import (
    check_pitcher_convergence,
    fit_pitcher_k_rate_model,
    prepare_pitcher_model_data,
)
from src.models.pitcher_projections import (
    fit_all_models as fit_pitcher_models,
    project_forward as project_pitcher_forward,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")

SEASONS = list(range(2018, 2026))
FROM_SEASON = 2025
DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute dashboard data")
    parser.add_argument(
        "--quick", action="store_true",
        help="Fewer MCMC draws for fast iteration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        draws, tune, chains = 500, 250, 2
        logger.info("QUICK mode: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    else:
        draws, tune, chains = 2000, 1000, 4
        logger.info("FULL mode: draws=%d, tune=%d, chains=%d", draws, tune, chains)

    # =================================================================
    # 0. Pre-cache observed profiles (needed by projection enrichment)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Pre-caching observed profiles for %d...", FROM_SEASON)
    try:
        obs_h = get_cached_hitter_observed_profile(FROM_SEASON)
        logger.info("Hitter observed profile: %d rows", len(obs_h))
    except Exception as e:
        logger.warning("Could not cache hitter observed profile: %s", e)

    try:
        sprint = get_cached_sprint_speed(FROM_SEASON)
        logger.info("Sprint speed: %d rows", len(sprint))
    except Exception as e:
        logger.warning("Could not cache sprint speed: %s", e)

    try:
        obs_p = get_cached_pitcher_observed_profile(FROM_SEASON)
        logger.info("Pitcher observed profile: %d rows", len(obs_p))
    except Exception as e:
        logger.warning("Could not cache pitcher observed profile: %s", e)

    # =================================================================
    # 1. Hitter composite projections (K% + BB% projected, observed enriched)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Fitting hitter composite models (K%%, BB%%)...")
    hitter_results = fit_hitter_models(
        seasons=SEASONS, min_pa=100,
        draws=draws, tune=tune, chains=chains, random_seed=42,
    )
    hitter_proj = project_hitter_forward(
        hitter_results, from_season=FROM_SEASON, min_pa=200,
    )
    hitter_proj.to_parquet(DASHBOARD_DIR / "hitter_projections.parquet", index=False)
    logger.info("Saved hitter projections: %d players", len(hitter_proj))

    # =================================================================
    # 2. Pitcher composite projections (K% + BB% projected, observed enriched)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Fitting pitcher composite models (K%%, BB%%)...")
    pitcher_results = fit_pitcher_models(
        seasons=SEASONS, min_bf=100,
        draws=draws, tune=tune, chains=chains, random_seed=42,
    )
    pitcher_proj = project_pitcher_forward(
        pitcher_results, from_season=FROM_SEASON, min_bf=200,
    )
    pitcher_proj.to_parquet(DASHBOARD_DIR / "pitcher_projections.parquet", index=False)
    logger.info("Saved pitcher projections: %d players", len(pitcher_proj))

    # =================================================================
    # 2b. Counting stat projections (K, BB, HR, SB for hitters; K, BB, Outs for pitchers)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing counting stat projections...")

    # Hitter counting stats (with park factor adjustment for HR)
    hitter_ext = build_multi_season_hitter_extended(SEASONS, min_pa=1)
    pa_priors = compute_hitter_pa_priors(
        hitter_ext, from_season=FROM_SEASON, min_pa=100,
    )

    # Load park factors for HR adjustment
    logger.info("Loading park factors for HR adjustment...")
    park_factors = get_park_factors(FROM_SEASON)
    hitter_venues = get_hitter_team_venue(FROM_SEASON)
    logger.info("Park factors: %d venue-hand combos, %d hitter-venue mappings",
                len(park_factors), len(hitter_venues))

    # Save park factor data for dashboard display
    hitter_venues.to_parquet(DASHBOARD_DIR / "hitter_venues.parquet", index=False)
    park_factors.to_parquet(DASHBOARD_DIR / "park_factors.parquet", index=False)

    hitter_counting = project_hitter_counting(
        rate_model_results=hitter_results,
        pa_priors=pa_priors,
        hitter_extended=hitter_ext,
        from_season=FROM_SEASON,
        n_draws=4000,
        min_pa=200,
        random_seed=42,
        park_factors=park_factors,
        hitter_venues=hitter_venues,
    )
    hitter_counting.to_parquet(DASHBOARD_DIR / "hitter_counting.parquet", index=False)
    logger.info("Saved hitter counting projections: %d players", len(hitter_counting))

    # Pitcher counting stats
    pitcher_ext = build_multi_season_pitcher_extended(SEASONS, min_bf=1)
    pitcher_counting = project_pitcher_counting(
        rate_model_results=pitcher_results,
        pitcher_extended=pitcher_ext,
        from_season=FROM_SEASON,
        n_draws=4000,
        min_bf=200,
        random_seed=42,
    )
    pitcher_counting.to_parquet(DASHBOARD_DIR / "pitcher_counting.parquet", index=False)
    logger.info("Saved pitcher counting projections: %d players", len(pitcher_counting))

    # =================================================================
    # 3. Pitcher K% model (for posterior samples → Game K sim)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Fitting pitcher K%% model for posterior samples...")
    df_pitcher = build_multi_season_pitcher_k_data(SEASONS, min_bf=1)
    pitcher_data = prepare_pitcher_model_data(df_pitcher)
    _model, pitcher_trace = fit_pitcher_k_rate_model(
        pitcher_data, draws=draws, tune=tune, chains=chains,
    )
    conv = check_pitcher_convergence(pitcher_trace)
    logger.info("Pitcher K%% convergence: %s (r_hat=%.4f)",
                "OK" if conv["converged"] else "ISSUES", conv["max_rhat"])

    # Extract forward-projected K% samples for each pitcher active in FROM_SEASON
    active = df_pitcher[
        (df_pitcher["season"] == FROM_SEASON) & (df_pitcher["batters_faced"] >= 50)
    ]["pitcher_id"].unique()

    k_samples_dict: dict[str, np.ndarray] = {}
    for pid in active:
        try:
            samples = extract_pitcher_k_rate_samples(
                pitcher_trace, pitcher_data,
                pitcher_id=int(pid),
                season=FROM_SEASON,
                project_forward=True,
            )
            k_samples_dict[str(int(pid))] = samples
        except ValueError:
            continue

    np.savez_compressed(
        DASHBOARD_DIR / "pitcher_k_samples.npz",
        **k_samples_dict,
    )
    logger.info("Saved K%% posterior samples for %d pitchers", len(k_samples_dict))

    # =================================================================
    # 4. BF priors
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing BF priors...")
    game_logs_list = []
    for s in SEASONS:
        gl = get_pitcher_game_logs(s)
        game_logs_list.append(gl)
    game_logs = pd.concat(game_logs_list, ignore_index=True)

    bf_priors = compute_pitcher_bf_priors(game_logs)
    bf_priors.to_parquet(DASHBOARD_DIR / "bf_priors.parquet", index=False)
    logger.info("Saved BF priors: %d pitcher-seasons", len(bf_priors))

    # =================================================================
    # 4b. Umpire K-rate tendencies
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing umpire K-rate tendencies...")
    umpire_tendencies = get_umpire_k_tendencies(
        seasons=list(range(2021, 2026)), min_games=30,
    )
    umpire_tendencies.to_parquet(DASHBOARD_DIR / "umpire_tendencies.parquet", index=False)
    logger.info("Saved umpire tendencies: %d umpires", len(umpire_tendencies))

    # =================================================================
    # 4c. Weather effects on K and HR rates
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing weather effects...")
    weather_effects = get_weather_effects(seasons=SEASONS)
    weather_effects.to_parquet(DASHBOARD_DIR / "weather_effects.parquet", index=False)
    logger.info("Saved weather effects: %d combinations", len(weather_effects))

    # =================================================================
    # 4d. Player team mapping
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing player-team mapping...")
    player_teams = get_player_teams(FROM_SEASON)
    player_teams.to_parquet(DASHBOARD_DIR / "player_teams.parquet", index=False)
    logger.info("Saved player teams: %d players", len(player_teams))

    # =================================================================
    # 4e. Game browser data (lineups + batter Ks for historical games)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing game browser data for %d...", FROM_SEASON)
    game_lineups = get_game_lineups(FROM_SEASON)
    game_lineups.to_parquet(DASHBOARD_DIR / "game_lineups.parquet", index=False)
    logger.info("Saved game lineups: %d rows (%d games)",
                len(game_lineups), game_lineups["game_pk"].nunique())

    game_batter_ks = get_game_batter_ks(FROM_SEASON)
    game_batter_ks.to_parquet(DASHBOARD_DIR / "game_batter_ks.parquet", index=False)
    logger.info("Saved game batter Ks: %d rows", len(game_batter_ks))

    # Pitcher game logs (used by Game Browser for game selection)
    pitcher_game_logs = get_pitcher_game_logs(FROM_SEASON)
    pitcher_game_logs.to_parquet(DASHBOARD_DIR / "pitcher_game_logs.parquet", index=False)
    logger.info("Saved pitcher game logs: %d rows", len(pitcher_game_logs))

    # =================================================================
    # 5. Matchup profiles (arsenal, vulnerability, strength)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Loading matchup profiles for %d...", FROM_SEASON)

    pitcher_arsenal = get_pitcher_arsenal(FROM_SEASON)
    pitcher_arsenal.to_parquet(DASHBOARD_DIR / "pitcher_arsenal.parquet", index=False)
    logger.info("Saved pitcher arsenal: %d rows", len(pitcher_arsenal))

    hitter_vuln = get_hitter_vulnerability(FROM_SEASON)
    hitter_vuln.to_parquet(DASHBOARD_DIR / "hitter_vuln.parquet", index=False)
    logger.info("Saved hitter vulnerability: %d rows", len(hitter_vuln))

    hitter_str = get_hitter_strength(FROM_SEASON)
    hitter_str.to_parquet(DASHBOARD_DIR / "hitter_str.parquet", index=False)
    logger.info("Saved hitter strength: %d rows", len(hitter_str))

    # Career-aggregated vulnerability (weighted across all seasons)
    vuln_frames = []
    for s in SEASONS:
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
    for s in SEASONS:
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

    # =================================================================
    # 6. Save preseason snapshot (frozen projections for end-of-season comparison)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Saving preseason snapshot...")
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    from datetime import date
    snapshot_date = date.today().isoformat()
    target_season = FROM_SEASON + 1  # projecting INTO this season

    hitter_proj["snapshot_date"] = snapshot_date
    hitter_proj["target_season"] = target_season
    pitcher_proj["snapshot_date"] = snapshot_date
    pitcher_proj["target_season"] = target_season

    h_path = snapshot_dir / f"hitter_projections_{target_season}_preseason.parquet"
    p_path = snapshot_dir / f"pitcher_projections_{target_season}_preseason.parquet"
    hitter_proj.to_parquet(h_path, index=False)
    pitcher_proj.to_parquet(p_path, index=False)
    logger.info("Saved preseason snapshot: %s, %s", h_path.name, p_path.name)

    # Remove snapshot columns from live projections (they stay in the snapshot files)
    hitter_proj.drop(columns=["snapshot_date", "target_season"], inplace=True)
    pitcher_proj.drop(columns=["snapshot_date", "target_season"], inplace=True)

    # =================================================================
    # Summary
    # =================================================================
    logger.info("=" * 60)
    logger.info("Dashboard pre-computation complete!")
    logger.info("  Hitter projections:  %d players", len(hitter_proj))
    logger.info("  Pitcher projections: %d players", len(pitcher_proj))
    logger.info("  Hitter counting:     %d players", len(hitter_counting))
    logger.info("  Pitcher counting:    %d players", len(pitcher_counting))
    logger.info("  K%% samples:          %d pitchers", len(k_samples_dict))
    logger.info("  BF priors:           %d pitcher-seasons", len(bf_priors))
    logger.info("  Pitcher arsenal:     %d rows", len(pitcher_arsenal))
    logger.info("  Hitter vulnerability:%d rows", len(hitter_vuln))
    logger.info("  Hitter strength:     %d rows", len(hitter_str))
    logger.info("  Output dir:          %s", DASHBOARD_DIR)


if __name__ == "__main__":
    main()
