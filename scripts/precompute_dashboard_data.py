#!/usr/bin/env python
"""
Pre-compute all data needed by the Streamlit dashboard.

Fits composite hitter + pitcher models (K% and BB% only), enriches with
observed profiles, extracts posterior K% samples, computes BF priors,
and saves everything to tdd-dashboard/data/dashboard/.

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

# Dashboard repo location — precompute writes directly to tdd-dashboard
DASHBOARD_REPO = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard")

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
    get_hitter_aggressiveness,
    get_hitter_traditional_stats,
    get_park_factors,
    get_hitter_team_venue,
    get_pitcher_efficiency,
    get_pitcher_game_logs,
    get_pitcher_traditional_stats,
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
DASHBOARD_DIR = DASHBOARD_REPO / "data" / "dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute dashboard data")
    parser.add_argument(
        "--quick", action="store_true",
        help="Fewer MCMC draws for fast iteration",
    )
    return parser.parse_args()


def precompute_backtest_summaries() -> None:
    """Copy backtest results to dashboard directory and compute confidence tiers."""
    import shutil
    from src.evaluation.confidence_tiers import assign_confidence_tiers, tiers_to_dataframe

    outputs_dir = PROJECT_ROOT / "outputs"
    dashboard_dir = DASHBOARD_DIR

    # Copy game prop backtest summary if it exists
    summary_path = outputs_dir / "game_prop_backtest_summary.parquet"
    if summary_path.exists():
        dest = dashboard_dir / "backtest_game_prop_summary.parquet"
        shutil.copy2(summary_path, dest)
        logger.info("Copied game prop backtest summary to %s", dest)

        # Compute confidence tiers
        summary = pd.read_parquet(summary_path)
        tiers = assign_confidence_tiers(summary)
        tiers_df = tiers_to_dataframe(tiers)
        tier_path = dashboard_dir / "backtest_confidence_tiers.parquet"
        tiers_df.to_parquet(tier_path, index=False)
        logger.info("Saved confidence tiers to %s (%d props)", tier_path, len(tiers_df))
    else:
        logger.info("No game prop backtest summary found at %s — skipping tier computation", summary_path)

    # Copy calibration data if available
    for pred_file in outputs_dir.glob("game_prop_predictions_*.parquet"):
        dest = dashboard_dir / pred_file.name
        shutil.copy2(pred_file, dest)
        logger.info("Copied %s to dashboard", pred_file.name)

    # Copy existing season backtest CSVs as standardized parquets
    for csv_file in outputs_dir.glob("*backtest*.csv"):
        try:
            df = pd.read_csv(csv_file)
            dest = dashboard_dir / f"backtest_{csv_file.stem}.parquet"
            df.to_parquet(dest, index=False)
            logger.info("Converted %s to parquet", csv_file.name)
        except Exception as e:
            logger.warning("Failed to convert %s: %s", csv_file.name, e)


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

    # ── Injury adjustment: scale counting stats by games-available fraction ──
    inj_path = DASHBOARD_DIR / "preseason_injuries.parquet"
    if inj_path.exists():
        inj_df = pd.read_parquet(inj_path)
        inj_df = inj_df[inj_df["player_id"].notna() & (inj_df["est_missed_games"] > 0)]
        inj_frac = dict(zip(
            inj_df["player_id"].astype(int),
            ((162 - inj_df["est_missed_games"].clip(upper=162)) / 162).values,
        ))
        logger.info("Applying injury adjustments to %d players", len(inj_frac))

        # Columns to scale (all counting distribution columns)
        hitter_scale_cols = [
            c for c in hitter_counting.columns
            if c.startswith("total_") or c in ("projected_pa_mean", "projected_games_mean")
        ]
        for pid, frac in inj_frac.items():
            mask = hitter_counting["batter_id"] == pid
            if mask.any():
                hitter_counting.loc[mask, hitter_scale_cols] *= frac
                logger.info("  Hitter %d: %.0f%% season", pid, frac * 100)

        pitcher_scale_cols = [
            c for c in pitcher_counting.columns
            if c.startswith("total_") or c in ("projected_bf_mean", "projected_games_mean")
        ]
        for pid, frac in inj_frac.items():
            mask = pitcher_counting["pitcher_id"] == pid
            if mask.any():
                pitcher_counting.loc[mask, pitcher_scale_cols] *= frac
                logger.info("  Pitcher %d: %.0f%% season", pid, frac * 100)
    else:
        logger.info("No preseason injuries file found — skipping injury adjustments")

    hitter_counting.to_parquet(DASHBOARD_DIR / "hitter_counting.parquet", index=False)
    logger.info("Saved hitter counting projections: %d players", len(hitter_counting))
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

    # Save preseason K samples snapshot for in-season conjugate updating
    snapshot_dir = DASHBOARD_DIR / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snapshot_dir / "pitcher_k_samples_preseason.npz",
        **k_samples_dict,
    )
    logger.info("Saved preseason K%% samples snapshot")

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

    # Compute pitcher P/PA from game logs for the projection season
    # (starters only, need BF > 0 to avoid div-by-zero)
    _starter_logs = game_logs[
        (game_logs["is_starter"] == True)  # noqa: E712
        & (game_logs["batters_faced"] >= 3)
        & (game_logs["number_of_pitches"].notna())
        & (game_logs["number_of_pitches"] > 0)
    ]
    if not _starter_logs.empty:
        _ppa = (
            _starter_logs.groupby("pitcher_id")
            .apply(
                lambda g: g["number_of_pitches"].sum() / g["batters_faced"].sum(),
                include_groups=False,
            )
            .reset_index(name="pitches_per_pa")
        )
        logger.info("Computed P/PA for %d pitchers (mean=%.2f)", len(_ppa), _ppa["pitches_per_pa"].mean())
    else:
        _ppa = None

    bf_priors = compute_pitcher_bf_priors(game_logs, pitcher_ppa=_ppa)
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
    # 4d. Player team mapping (2025 regular season + 2026 spring training override)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing player-team mapping...")
    player_teams = get_player_teams(FROM_SEASON)

    # Override with 2026 spring training teams for players who changed teams
    try:
        from src.data.db import read_sql as _read_sql
        st_teams = _read_sql("""
            WITH st_appearances AS (
                SELECT bb.batter_id AS player_id, bb.team_id, COUNT(*) AS games
                FROM staging.batting_boxscores bb
                JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
                WHERE dg.season = :season AND dg.game_type = 'S'
                GROUP BY bb.batter_id, bb.team_id
                UNION ALL
                SELECT pb.pitcher_id AS player_id, pb.team_id, COUNT(*) AS games
                FROM staging.pitching_boxscores pb
                JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
                WHERE dg.season = :season AND dg.game_type = 'S'
                GROUP BY pb.pitcher_id, pb.team_id
            ),
            primary_st AS (
                SELECT player_id, team_id,
                       SUM(games) AS total_games,
                       ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY SUM(games) DESC) AS rn
                FROM st_appearances GROUP BY player_id, team_id
            )
            SELECT p.player_id,
                   COALESCE(dt.abbreviation, '') AS team_abbr,
                   COALESCE(dt.team_name, '') AS team_name
            FROM primary_st p
            LEFT JOIN production.dim_team dt ON p.team_id = dt.team_id
            WHERE p.rn = 1
        """, {"season": FROM_SEASON + 1})
        if not st_teams.empty:
            # Only override for players already in our projections
            st_lookup = dict(zip(st_teams["player_id"], st_teams["team_abbr"]))
            st_name_lookup = dict(zip(st_teams["player_id"], st_teams["team_name"]))
            n_updated = 0
            for idx, row in player_teams.iterrows():
                pid = row["player_id"]
                if pid in st_lookup and st_lookup[pid] and st_lookup[pid] != row["team_abbr"]:
                    player_teams.at[idx, "team_abbr"] = st_lookup[pid]
                    player_teams.at[idx, "team_name"] = st_name_lookup.get(pid, "")
                    n_updated += 1
            # Also add players who are in spring training but not in 2025 regular season
            existing_pids = set(player_teams["player_id"])
            new_rows = st_teams[~st_teams["player_id"].isin(existing_pids)]
            if not new_rows.empty:
                player_teams = pd.concat([player_teams, new_rows], ignore_index=True)
            logger.info("Spring training override: %d team changes, %d new players",
                        n_updated, len(new_rows))
    except Exception as e:
        logger.warning("Could not load spring training teams: %s", e)

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

    # Game info (date, teams) — used by Game Browser enrichment
    from src.data.db import read_sql as _read_sql_gi
    game_info = _read_sql_gi("""
        SELECT game_pk, game_date, season,
               home_team_id, away_team_id,
               home_team_name, away_team_name
        FROM production.dim_game
        WHERE game_type = 'R' AND season = :season
    """, {"season": FROM_SEASON})
    game_info.to_parquet(DASHBOARD_DIR / "game_info.parquet", index=False)
    logger.info("Saved game info: %d games", len(game_info))

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
    # 5b. Location grid data (pitcher heatmaps + hitter zone profiles)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing location grids for %d...", FROM_SEASON)

    from src.data.queries import get_pitcher_location_grid, get_hitter_zone_grid

    pitcher_loc = get_pitcher_location_grid(FROM_SEASON)
    pitcher_loc.to_parquet(DASHBOARD_DIR / "pitcher_location_grid.parquet", index=False)
    logger.info("Saved pitcher location grid: %d rows (%d pitchers)",
                len(pitcher_loc), pitcher_loc["pitcher_id"].nunique())

    hitter_zone = get_hitter_zone_grid(FROM_SEASON)
    hitter_zone.to_parquet(DASHBOARD_DIR / "hitter_zone_grid.parquet", index=False)
    logger.info("Saved hitter zone grid: %d rows (%d batters)",
                len(hitter_zone), hitter_zone["batter_id"].nunique())

    # Career-aggregated hitter zone grid (sum counts across all seasons)
    zone_frames = []
    for s in SEASONS:
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

    # =================================================================
    # 5c. Archetype-based matchup profiles
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing archetype-based matchup data for %d...", FROM_SEASON)

    try:
        from src.data.pitch_archetypes import (
            get_pitch_archetype_offerings,
            get_pitch_archetype_clusters,
        )
        from src.data.league_baselines import get_baselines_by_archetype_stand

        pitcher_offerings = get_pitch_archetype_offerings(FROM_SEASON)
        pitcher_offerings.to_parquet(DASHBOARD_DIR / "pitcher_offerings.parquet", index=False)
        logger.info("Saved pitcher offerings with archetypes: %d rows", len(pitcher_offerings))

        cluster_metadata = get_pitch_archetype_clusters()
        cluster_metadata.to_parquet(DASHBOARD_DIR / "pitcher_cluster_metadata.parquet", index=False)
        logger.info("Saved pitcher cluster metadata: %d archetypes", len(cluster_metadata))

        baselines_arch = get_baselines_by_archetype_stand(FROM_SEASON)
        baselines_arch.to_parquet(DASHBOARD_DIR / "baselines_arch.parquet", index=False)
        logger.info("Saved league baselines by archetype: %d rows", len(baselines_arch))
    except Exception as e:
        logger.warning("Archetype pitcher data failed: %s", e)

    try:
        from src.data.feature_eng import get_hitter_vulnerability_by_archetype

        hitter_vuln_arch = get_hitter_vulnerability_by_archetype(FROM_SEASON)
        hitter_vuln_arch.to_parquet(DASHBOARD_DIR / "hitter_vuln_arch.parquet", index=False)
        logger.info("Saved hitter vulnerability by archetype: %d rows", len(hitter_vuln_arch))

        # Career-aggregated archetype vulnerability
        arch_frames = []
        for s in SEASONS:
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

    # =================================================================
    # 6. Traditional / actual stats (2025 season from boxscores + Statcast)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing traditional stats (2025)...")
    hitter_trad = get_hitter_traditional_stats(FROM_SEASON)
    hitter_trad.to_parquet(DASHBOARD_DIR / "hitter_traditional.parquet", index=False)
    logger.info("Saved hitter traditional stats: %d players", len(hitter_trad))

    pitcher_trad = get_pitcher_traditional_stats(FROM_SEASON)
    pitcher_trad.to_parquet(DASHBOARD_DIR / "pitcher_traditional.parquet", index=False)
    logger.info("Saved pitcher traditional stats: %d players", len(pitcher_trad))

    # =================================================================
    # 6b. Hitter aggressiveness & pitcher efficiency profiles
    # =================================================================
    logger.info("=" * 60)
    logger.info("Computing hitter aggressiveness profiles...")
    hitter_agg = get_hitter_aggressiveness(FROM_SEASON)
    hitter_agg.to_parquet(DASHBOARD_DIR / "hitter_aggressiveness.parquet", index=False)
    logger.info("Saved hitter aggressiveness: %d players", len(hitter_agg))

    logger.info("Computing pitcher efficiency profiles...")
    pitcher_eff = get_pitcher_efficiency(FROM_SEASON)
    pitcher_eff.to_parquet(DASHBOARD_DIR / "pitcher_efficiency.parquet", index=False)
    logger.info("Saved pitcher efficiency: %d players", len(pitcher_eff))

    # =================================================================
    # 6c. Multi-season data (concat all seasons for season selector)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Building multi-season datasets (2018-2025)...")

    # Copy full_stats parquets to dashboard dir (already have all seasons)
    import shutil
    for fname in ["hitter_full_stats.parquet", "pitcher_full_stats.parquet"]:
        src = PROJECT_ROOT / "data" / "cached" / fname
        if src.exists():
            shutil.copy2(src, DASHBOARD_DIR / fname)
            _fs = pd.read_parquet(DASHBOARD_DIR / fname)
            logger.info("Copied %s: %d rows", fname, len(_fs))
        else:
            logger.warning("Missing %s — run feature_eng first", fname)

    # Multi-season traditional stats
    trad_frames_h, trad_frames_p = [], []
    for s in SEASONS:
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
    for s in SEASONS:
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

    # Multi-season arsenal (already loaded per-season during career vuln)
    arsenal_frames = []
    for s in SEASONS:
        try:
            arsenal_frames.append(get_pitcher_arsenal(s))
        except Exception as e:
            logger.warning("Pitcher arsenal %d failed: %s", s, e)
    if arsenal_frames:
        all_arsenal = pd.concat(arsenal_frames, ignore_index=True)
        all_arsenal.to_parquet(DASHBOARD_DIR / "pitcher_arsenal_all.parquet", index=False)
        logger.info("Saved pitcher_arsenal_all: %d rows", len(all_arsenal))

    # Multi-season vuln + strength (concat the frames already loaded for career)
    if vuln_frames:
        all_vuln_seasons = pd.concat(vuln_frames, ignore_index=True)
        all_vuln_seasons.to_parquet(DASHBOARD_DIR / "hitter_vuln_all.parquet", index=False)
        logger.info("Saved hitter_vuln_all: %d rows", len(all_vuln_seasons))
    if str_frames:
        all_str_seasons = pd.concat(str_frames, ignore_index=True)
        all_str_seasons.to_parquet(DASHBOARD_DIR / "hitter_str_all.parquet", index=False)
        logger.info("Saved hitter_str_all: %d rows", len(all_str_seasons))

    # Multi-season location grids (add season column since queries don't include it)
    pitcher_loc_frames, hitter_zone_frames = [], []
    for s in SEASONS:
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

    # =================================================================
    # 7. Save preseason snapshot (frozen projections for end-of-season comparison)
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
    # 8. Backtest summaries (copy outputs → dashboard, compute confidence tiers)
    # =================================================================
    logger.info("=" * 60)
    logger.info("Processing backtest summaries...")
    precompute_backtest_summaries()

    # =================================================================
    # Summary
    # =================================================================
    logger.info("=" * 60)
    logger.info("Dashboard pre-computation complete!")
    logger.info("  Hitter projections:  %d players", len(hitter_proj))
    logger.info("  Pitcher projections: %d players", len(pitcher_proj))
    logger.info("  Hitter counting:     %d players", len(hitter_counting))
    logger.info("  Pitcher counting:    %d players", len(pitcher_counting))
    logger.info("  Hitter traditional:  %d players", len(hitter_trad))
    logger.info("  Pitcher traditional: %d players", len(pitcher_trad))
    logger.info("  Hitter aggressiveness:%d players", len(hitter_agg))
    logger.info("  Pitcher efficiency:  %d players", len(pitcher_eff))
    logger.info("  K%% samples:          %d pitchers", len(k_samples_dict))
    logger.info("  BF priors:           %d pitcher-seasons", len(bf_priors))
    logger.info("  Pitcher arsenal:     %d rows", len(pitcher_arsenal))
    logger.info("  Hitter vulnerability:%d rows", len(hitter_vuln))
    logger.info("  Hitter strength:     %d rows", len(hitter_str))
    logger.info("  Multi-season files:  hitter/pitcher trad_all, agg_all, eff_all, arsenal_all, vuln_all, str_all, loc_all, zone_all, full_stats")
    logger.info("  Output dir:          %s", DASHBOARD_DIR)


if __name__ == "__main__":
    main()
