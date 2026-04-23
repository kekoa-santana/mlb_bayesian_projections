#!/usr/bin/env python
"""
In-season daily update script.

Performs Beta-Binomial conjugate updating of preseason projections
with observed 2026 data, regenerates dashboard parquets, and
fetches today's schedule with matchup analysis.

Game-level **model** probabilities (ML / spread / O/U) are built by
``confident_picks`` and saved as ``game_predictions.parquet``.
**Sportsbook** lines are still fetched by the dashboard
``scripts/collect_game_odds.py`` (``game_odds_history.parquet``,
``game_odds_daily.parquet``). ``confident_picks`` merges DraftKings **player**
props into ``game_props.parquet``.

Usage
-----
    python scripts/update_in_season.py                    # today's date
    python scripts/update_in_season.py --date 2026-04-15  # specific date
    python scripts/update_in_season.py --skip-schedule    # skip API calls
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit as _scipy_logit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import dashboard_dir, dashboard_repo
from src.utils.weather import parse_temp_bucket, wind_category

DASHBOARD_REPO = dashboard_repo()
DASHBOARD_DIR = dashboard_dir()
SNAPSHOT_DIR = DASHBOARD_DIR / "snapshots"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEASON = 2026


# Weather helpers imported from src.utils.weather
_parse_temp_bucket = parse_temp_bucket
_wind_category = wind_category


# Rate stats to update via conjugate updating
HITTER_RATE_STATS = [
    {"name": "k_rate", "trials": "pa", "successes": "strikeouts"},
    {"name": "bb_rate", "trials": "pa", "successes": "walks"},
]
PITCHER_RATE_STATS = [
    {"name": "k_rate", "trials": "batters_faced", "successes": "strike_outs"},
    {"name": "bb_rate", "trials": "batters_faced", "successes": "walks"},
]


def load_preseason_snapshot(player_type: str) -> pd.DataFrame:
    """Load frozen preseason projections."""
    fname = f"{player_type}_projections_{SEASON}_preseason.parquet"
    path = SNAPSHOT_DIR / fname
    if not path.exists():
        logger.warning("No preseason snapshot at %s — using live projections", path)
        path = DASHBOARD_DIR / f"{player_type}_projections.parquet"
    if not path.exists():
        logger.error("No projections found for %s", player_type)
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_preseason_k_samples() -> dict[str, np.ndarray]:
    """Load frozen preseason K% samples."""
    path = SNAPSHOT_DIR / "pitcher_k_samples_preseason.npz"
    if not path.exists():
        path = DASHBOARD_DIR / "pitcher_k_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_preseason_rate_samples(stat_name: str) -> dict[str, np.ndarray]:
    """Load frozen preseason BB% or HR/BF samples."""
    npz_name = f"pitcher_{stat_name}_samples"
    path = SNAPSHOT_DIR / f"{npz_name}_preseason.npz"
    if not path.exists():
        path = DASHBOARD_DIR / f"{npz_name}.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_hitter_preseason_samples(stat_name: str) -> dict[str, np.ndarray]:
    """Load frozen preseason hitter rate samples (k, bb, or hr)."""
    npz_name = f"hitter_{stat_name}_samples"
    path = SNAPSHOT_DIR / f"{npz_name}_preseason.npz"
    if not path.exists():
        path = DASHBOARD_DIR / f"{npz_name}.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_observed_hitter_totals() -> pd.DataFrame:
    """Query 2026 hitter season totals from the database."""
    from src.data.db import read_sql
    df = read_sql("""
        SELECT
            fp.batter_id,
            COUNT(DISTINCT fp.pa_id) AS pa,
            SUM(CASE WHEN fp.events = 'strikeout' THEN 1 ELSE 0 END) AS strikeouts,
            SUM(CASE WHEN fp.events = 'walk' THEN 1 ELSE 0 END) AS walks,
            SUM(CASE WHEN fp.events = 'home_run' THEN 1 ELSE 0 END) AS hr
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY fp.batter_id
        HAVING COUNT(DISTINCT fp.pa_id) >= 1
    """, {"season": SEASON})
    logger.info("Observed 2026 hitter totals: %d batters", len(df))
    return df


def get_observed_pitcher_totals() -> pd.DataFrame:
    """Query 2026 pitcher season totals from the database."""
    from src.data.db import read_sql
    df = read_sql("""
        SELECT
            pb.pitcher_id,
            SUM(pb.batters_faced) AS batters_faced,
            SUM(pb.strike_outs) AS strike_outs,
            SUM(pb.walks) AS walks,
            SUM(pb.home_runs) AS hr,
            COUNT(*) AS games_pitched,
            SUM(CASE WHEN pb.is_starter THEN 1 ELSE 0 END) AS games_started
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY pb.pitcher_id
        HAVING SUM(pb.batters_faced) >= 1
    """, {"season": SEASON})
    logger.info("Observed 2026 pitcher totals: %d pitchers", len(df))
    return df


def update_projections_step() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: Conjugate-update rate projections.

    After loading preseason snapshots, merges MiLB-informed priors for
    rookies who lack preseason projections via the prospect bridge.
    """
    from src.models.in_season_updater import update_projections
    from src.models.prospect_bridge import build_rookie_priors, merge_rookie_priors

    # Build MiLB-informed priors for rookies
    h_rookie_priors, p_rookie_priors = build_rookie_priors(SEASON)

    # Hitters
    h_preseason = load_preseason_snapshot("hitter")
    h_obs = get_observed_hitter_totals()

    # Merge rookie priors into preseason (fill gaps, don't overwrite)
    if not h_preseason.empty and not h_rookie_priors.empty:
        h_preseason = merge_rookie_priors(h_preseason, h_rookie_priors, "batter_id")

    if h_preseason.empty:
        logger.error("No hitter preseason projections — skipping hitter update")
        h_updated = pd.DataFrame()
    elif h_obs.empty:
        logger.info("No 2026 hitter data yet — using preseason projections")
        h_updated = h_preseason.copy()
    else:
        h_updated = update_projections(
            h_preseason, h_obs,
            id_col="batter_id",
            rate_stats=HITTER_RATE_STATS,
            min_trials=10,
        )
        n_updated = h_updated.get("obs_2026_pa", pd.Series(dtype=float)).gt(0).sum()
        logger.info("Updated %d hitters with 2026 data", n_updated)

    # Pitchers
    p_preseason = load_preseason_snapshot("pitcher")
    p_obs = get_observed_pitcher_totals()

    # Merge rookie priors into preseason (fill gaps, don't overwrite)
    if not p_preseason.empty and not p_rookie_priors.empty:
        p_preseason = merge_rookie_priors(p_preseason, p_rookie_priors, "pitcher_id")

    if p_preseason.empty:
        logger.error("No pitcher preseason projections — skipping pitcher update")
        p_updated = pd.DataFrame()
    elif p_obs.empty:
        logger.info("No 2026 pitcher data yet — using preseason projections")
        p_updated = p_preseason.copy()
    else:
        p_updated = update_projections(
            p_preseason, p_obs,
            id_col="pitcher_id",
            rate_stats=PITCHER_RATE_STATS,
            min_trials=10,
        )
        n_updated = p_updated.get("obs_2026_batters_faced", pd.Series(dtype=float)).gt(0).sum()
        logger.info("Updated %d pitchers with 2026 data", n_updated)

    return h_updated, p_updated


def update_k_samples_step() -> dict[str, np.ndarray]:
    """Step 2: Regenerate pitcher K% samples via conjugate updating."""
    from src.models.in_season_updater import update_pitcher_k_samples

    preseason_samples = load_preseason_k_samples()
    p_obs = get_observed_pitcher_totals()

    if not preseason_samples:
        logger.warning("No preseason K%% samples — skipping K sample update")
        return {}

    if p_obs.empty:
        logger.info("No 2026 pitcher data — using preseason K%% samples")
        return preseason_samples

    updated = update_pitcher_k_samples(
        preseason_samples, p_obs,
        min_bf=10, n_samples=1000,
    )
    logger.info("Updated K%% samples: %d pitchers", len(updated))
    return updated


def update_player_rate_samples_step(
    player_type: str,
    stat_name: str,
    trials_col: str,
    successes_col: str,
    league_avg: float,
) -> dict[str, np.ndarray]:
    """Conjugate-update per-player rate samples for a given stat.

    Parameters
    ----------
    player_type : str
        ``"pitcher"`` or ``"hitter"`` — controls which preseason loader,
        observed-totals query, and id column are used.
    stat_name : str
        Rate stat key (e.g. ``"bb"``, ``"hr"``, ``"k"``).
    trials_col, successes_col : str
        Column names in the observed totals frame.
    league_avg : float
        Prior mean for the Beta-Binomial conjugate update.

    Returns
    -------
    dict[str, np.ndarray]
        ``{player_id_str: samples}``. Empty dict when the preseason
        samples are missing; the raw preseason samples when no 2026
        observed data has been seen yet.
    """
    from src.models.in_season_updater import update_rate_samples

    if player_type == "pitcher":
        preseason = load_preseason_rate_samples(stat_name)
        id_col = "pitcher_id"
        label_noun = "pitchers"
        missing_label = "%s"
        fallback_label = "pitcher"
    elif player_type == "hitter":
        preseason = load_hitter_preseason_samples(stat_name)
        id_col = "batter_id"
        label_noun = "batters"
        missing_label = "hitter %s"
        fallback_label = "hitter"
    else:
        raise ValueError(f"Unknown player_type: {player_type!r}")

    if not preseason:
        logger.warning("No preseason %s samples — skipping", missing_label % stat_name)
        return {}

    obs = (
        get_observed_pitcher_totals() if player_type == "pitcher"
        else get_observed_hitter_totals()
    )
    if obs.empty:
        logger.info(
            "No 2026 %s data — using preseason %s samples",
            fallback_label, stat_name,
        )
        return preseason

    updated = update_rate_samples(
        preseason, obs,
        player_id_col=id_col,
        trials_col=trials_col,
        successes_col=successes_col,
        league_avg=league_avg,
        min_trials=10, n_samples=1000,
    )
    if player_type == "hitter":
        logger.info("Updated hitter %s samples: %d batters", stat_name, len(updated))
    else:
        logger.info("Updated %s samples: %d pitchers", stat_name, len(updated))
    return updated


def fetch_schedule_step(
    game_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 3: Fetch today's schedule and lineups."""
    from src.data.schedule import fetch_todays_schedule, fetch_all_lineups

    schedule = fetch_todays_schedule(game_date)
    if schedule.empty:
        logger.info("No games scheduled for %s", game_date)
        return schedule, pd.DataFrame()

    lineups = fetch_all_lineups(schedule)
    logger.info("Fetched lineups: %d batters across %d games",
                len(lineups), lineups["game_pk"].nunique() if not lineups.empty else 0)

    return schedule, lineups


def update_season_stats_step() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 5: Query 2026 traditional stats for season leaders."""
    from src.data.queries import (
        get_hitter_traditional_stats,
        get_pitcher_traditional_stats,
    )

    h_trad = pd.DataFrame()
    p_trad = pd.DataFrame()

    try:
        h_trad = get_hitter_traditional_stats(SEASON)
        if not h_trad.empty:
            h_trad.to_parquet(
                DASHBOARD_DIR / "hitter_traditional.parquet", index=False,
            )
            logger.info(
                "Saved 2026 hitter traditional stats: %d players", len(h_trad),
            )
        else:
            logger.info("No 2026 hitter traditional stats yet")
    except Exception as e:
        logger.warning("Hitter traditional stats failed: %s", e)

    try:
        p_trad = get_pitcher_traditional_stats(SEASON)
        if not p_trad.empty:
            p_trad.to_parquet(
                DASHBOARD_DIR / "pitcher_traditional.parquet", index=False,
            )
            logger.info(
                "Saved 2026 pitcher traditional stats: %d players", len(p_trad),
            )
        else:
            logger.info("No 2026 pitcher traditional stats yet")
    except Exception as e:
        logger.warning("Pitcher traditional stats failed: %s", e)

    return h_trad, p_trad


def update_advanced_stats_step() -> None:
    """Save observed advanced/Statcast stats for hitters and pitchers.

    Queries fact_batting_advanced, fact_pitching_advanced, and pitch-level
    observed profiles to build population-level parquets suitable for
    percentile ranking on the dashboard.
    """
    from src.data.db import read_sql
    from src.data.queries.hitter import get_hitter_observed_profile
    from src.data.queries.pitcher import get_pitcher_observed_profile

    # ---- Hitter advanced ----
    try:
        h_adv = read_sql("""
            SELECT batter_id, season, pa, xba, xslg, xwoba, wrc_plus,
                   barrel_pct, hard_hit_pct, sweet_spot_pct, k_pct, bb_pct
            FROM production.fact_batting_advanced
            WHERE season = :season AND pa >= 10
        """, {"season": SEASON})

        h_obs = get_hitter_observed_profile(SEASON)
        if not h_adv.empty and not h_obs.empty:
            obs_cols = ["batter_id", "whiff_rate", "chase_rate",
                        "z_contact_pct", "avg_exit_velo", "fb_pct",
                        "hard_hit_pct"]
            obs_avail = [c for c in obs_cols if c in h_obs.columns]
            h_merged = h_adv.merge(h_obs[obs_avail], on="batter_id", how="left",
                                   suffixes=("", "_obs"))
            # Prefer pitch-level hard_hit if available
            if "hard_hit_pct_obs" in h_merged.columns:
                h_merged["hard_hit_pct"] = h_merged["hard_hit_pct"].fillna(
                    h_merged.pop("hard_hit_pct_obs"))
        elif not h_adv.empty:
            h_merged = h_adv
        else:
            h_merged = pd.DataFrame()

        if not h_merged.empty:
            h_merged.to_parquet(
                DASHBOARD_DIR / "hitter_advanced.parquet", index=False)
            logger.info("Saved hitter_advanced.parquet: %d rows", len(h_merged))
        else:
            logger.info("No 2026 hitter advanced stats yet")
    except Exception as e:
        logger.warning("Hitter advanced stats failed: %s", e)

    # ---- Pitcher advanced ----
    try:
        p_adv = read_sql("""
            SELECT pitcher_id, season, batters_faced, k_pct, bb_pct,
                   swstr_pct, csw_pct, zone_pct, chase_pct, contact_pct,
                   xwoba_against, barrel_pct_against, hard_hit_pct_against
            FROM production.fact_pitching_advanced
            WHERE season = :season AND batters_faced >= 10
        """, {"season": SEASON})

        p_obs = get_pitcher_observed_profile(SEASON)
        if not p_adv.empty and not p_obs.empty:
            obs_cols = ["pitcher_id", "whiff_rate", "avg_velo",
                        "release_extension", "zone_pct", "gb_pct"]
            obs_avail = [c for c in obs_cols if c in p_obs.columns]
            p_merged = p_adv.merge(p_obs[obs_avail], on="pitcher_id", how="left",
                                   suffixes=("", "_obs"))
            if "zone_pct_obs" in p_merged.columns:
                p_merged["zone_pct"] = p_merged["zone_pct"].fillna(
                    p_merged.pop("zone_pct_obs"))
        elif not p_adv.empty:
            p_merged = p_adv
        else:
            p_merged = pd.DataFrame()

        if not p_merged.empty:
            p_merged.to_parquet(
                DASHBOARD_DIR / "pitcher_advanced.parquet", index=False)
            logger.info("Saved pitcher_advanced.parquet: %d rows", len(p_merged))
        else:
            logger.info("No 2026 pitcher advanced stats yet")
    except Exception as e:
        logger.warning("Pitcher advanced stats failed: %s", e)


def refresh_prospect_rankings() -> None:
    """Weekly: Re-score readiness, re-rank prospects, update comps.

    Loads current MiLB translated data from cache, re-trains the
    readiness model, re-runs TDD prospect rankings for both batters
    and pitchers, and refreshes prospect-to-MLB comps.  All outputs
    are written to the dashboard directory.
    """
    import shutil

    logger.info("=" * 60)
    logger.info("Weekly prospect rankings refresh (season %d)...", SEASON)

    cached_dir = PROJECT_ROOT / "data" / "cached"

    # Rebuild MiLB translations with current season data so new AAA/AA
    # performance (e.g. a prospect demolishing AAA) flows into rankings.
    try:
        from src.data.feature_eng import build_milb_translated_data

        seasons = [y for y in range(2005, SEASON + 1) if y != 2020]
        logger.info("Rebuilding MiLB translations for %d-%d...", seasons[0], seasons[-1])
        build_milb_translated_data(seasons=seasons, force_rebuild=True)
        logger.info("MiLB translations rebuilt (includes %d data)", SEASON)
    except Exception:
        logger.exception("Failed to rebuild MiLB translations — using cached data")

    # Guard: check for MiLB translated data
    batter_path = cached_dir / "milb_translated_batters.parquet"
    pitcher_path = cached_dir / "milb_translated_pitchers.parquet"
    if not batter_path.exists() or not pitcher_path.exists():
        logger.warning(
            "MiLB translated data not found in %s — skipping prospect refresh. "
            "Run build_milb_translations.py first.",
            cached_dir,
        )
        return

    # Copy MiLB translated data to dashboard
    milb_files = {
        "milb_translated_batters.parquet": "MiLB translated batters",
        "milb_translated_pitchers.parquet": "MiLB translated pitchers",
    }
    for fname, label in milb_files.items():
        src_path = cached_dir / fname
        if src_path.exists():
            shutil.copy2(src_path, DASHBOARD_DIR / fname)
            logger.info("Copied %s to dashboard", label)

    # 1. Re-score readiness
    n_batters_ranked = 0
    n_pitchers_ranked = 0

    try:
        from src.models.mlb_readiness import train_readiness_model, score_prospects
        from src.data.db import read_sql

        bundle = train_readiness_model()
        prospects_df = score_prospects(projection_season=SEASON)
        logger.info(
            "Readiness model: AUC=%.3f, scored %d prospects",
            bundle["train_auc"], len(prospects_df),
        )

        # Merge with FanGraphs rankings
        rankings = read_sql(
            "SELECT player_id, player_name, org, position, overall_rank, "
            "org_rank, future_value, risk, eta, source "
            "FROM production.dim_prospect_ranking "
            f"WHERE season = {SEASON}",
            {},
        )
        if not rankings.empty:
            rankings = rankings.sort_values(
                "source", ascending=True,
            ).drop_duplicates("player_id", keep="first")
            logger.info("FanGraphs rankings: %d prospects for %d", len(rankings), SEASON)

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

            prospect_out.to_parquet(
                DASHBOARD_DIR / "prospect_readiness.parquet", index=False,
            )
            logger.info("Saved prospect_readiness.parquet: %d rows", len(prospect_out))

    except Exception:
        logger.exception("Failed to update prospect readiness scores")

    # 2. Re-run TDD prospect rankings
    try:
        from src.models.prospect_ranking import rank_prospects, rank_pitching_prospects

        prospect_rankings = rank_prospects(projection_season=SEASON)
        if not prospect_rankings.empty:
            prospect_rankings.to_parquet(
                DASHBOARD_DIR / "prospect_rankings.parquet", index=False,
            )
            n_batters_ranked = len(prospect_rankings)
            logger.info(
                "Saved prospect_rankings.parquet: %d batters", n_batters_ranked,
            )
        else:
            logger.warning("No batting prospect rankings generated")

        pitching_rankings = rank_pitching_prospects(projection_season=SEASON)
        if not pitching_rankings.empty:
            pitching_rankings.to_parquet(
                DASHBOARD_DIR / "pitching_prospect_rankings.parquet", index=False,
            )
            n_pitchers_ranked = len(pitching_rankings)
            logger.info(
                "Saved pitching_prospect_rankings.parquet: %d pitchers",
                n_pitchers_ranked,
            )
        else:
            logger.warning("No pitching prospect rankings generated")

    except Exception:
        logger.exception("Failed to update prospect rankings")

    # 3. Re-run prospect comps
    try:
        from src.models.prospect_comps import find_all_comps

        comps = find_all_comps(projection_season=SEASON)
        for key, cdf in comps.items():
            if not cdf.empty:
                fname = f"prospect_comps_{key}.parquet"
                cdf.to_parquet(DASHBOARD_DIR / fname, index=False)
                logger.info("Saved %s: %d rows", fname, len(cdf))

    except Exception:
        logger.exception("Failed to update prospect comps")

    logger.info(
        "Prospect refresh complete: %d batters, %d pitchers ranked",
        n_batters_ranked, n_pitchers_ranked,
    )


def update_weekly_rankings_step() -> None:
    """Weekly: Re-rank players with 2026 observed data.

    Uses the same ``rank_all`` engine as preseason, but pointed at
    2026 observed stats + conjugate-updated projections.  The
    exposure-conditioned scouting weight naturally shifts from
    projection-dominant (early season) to production-dominant
    (mid-season onward).
    """
    from src.models.player_rankings import rank_all

    logger.info("=" * 60)
    logger.info("Weekly player rankings refresh (season %d)...", SEASON)

    try:
        rankings = rank_all(
            season=SEASON,
            projection_season=SEASON,
            min_pa=40,
            min_bf=35,
        )
        for key, rdf in rankings.items():
            if not rdf.empty:
                fname = f"{key}_rankings.parquet"
                rdf.to_parquet(DASHBOARD_DIR / fname, index=False)
                logger.info("Saved %s: %d rows", fname, len(rdf))
            else:
                logger.warning("No %s rankings produced", key)
    except Exception:
        logger.exception("Failed to update player rankings")


def update_weekly_team_step() -> None:
    """Weekly: Update team ELO with 2026 games + rebuild power rankings."""
    logger.info("=" * 60)
    logger.info("Weekly team rankings refresh...")

    # 1. Recompute ELO including 2026 games
    try:
        from src.data.team_queries import (
            get_game_results,
            get_team_info,
            get_venue_run_factors,
        )
        from src.models.team_elo import (
            compute_elo_history,
            get_current_ratings,
        )

        elo_games = get_game_results()
        elo_venue = get_venue_run_factors()
        elo_team_info = get_team_info()
        logger.info(
            "ELO input: %d games (including 2026)", len(elo_games),
        )

        elo_ratings, elo_history = compute_elo_history(elo_games, elo_venue)
        elo_current = get_current_ratings(elo_ratings, elo_team_info)
        elo_current.to_parquet(
            DASHBOARD_DIR / "team_elo.parquet", index=False,
        )
        logger.info("Saved updated team ELO: %d teams", len(elo_current))

    except Exception:
        logger.exception("Failed to update team ELO")
        return

    # 2. Rebuild team profiles + rankings
    profiles = pd.DataFrame()
    try:
        from src.models.team_profiles import build_all_team_profiles
        from src.models.team_rankings import rank_teams

        profiles = build_all_team_profiles(
            season=SEASON,
            projection_season=SEASON,
            elo_history=elo_history,
        )
        if not profiles.empty:
            profiles.to_parquet(
                DASHBOARD_DIR / "team_profiles.parquet", index=False,
            )
            logger.info("Saved team profiles: %d teams", len(profiles))

            # Observed RS/RA for projected wins
            obs_rs_ra = None
            try:
                from src.data.team_queries import get_game_results as _get_gr
                from src.models.team_sim.league_season_sim import (
                    compute_2h_weighted_rs_ra,
                )
                obs_rs_ra = compute_2h_weighted_rs_ra(
                    _get_gr(), SEASON,
                )
            except Exception:
                logger.warning("Could not compute observed RS/RA")

            team_rankings = rank_teams(
                profiles,
                elo_ratings=elo_current,
                observed_rs_ra=obs_rs_ra,
            )
            if not team_rankings.empty:
                team_rankings.to_parquet(
                    DASHBOARD_DIR / "team_rankings.parquet", index=False,
                )
                logger.info(
                    "Saved team rankings: %d teams", len(team_rankings),
                )
    except Exception:
        logger.exception("Failed to update team profiles/rankings")

    # 3. Rebuild power rankings (with in-season Beta-Binomial blend)
    try:
        from scripts.precompute.team import _merge_power_into_team_rankings
        from src.models.in_season_wins import load_current_team_records
        from src.models.team_rankings import build_power_rankings

        h_proj_path = DASHBOARD_DIR / "hitter_projections.parquet"
        p_proj_path = DASHBOARD_DIR / "pitcher_projections.parquet"
        roster_path = DASHBOARD_DIR / "roster.parquet"

        if roster_path.exists():
            try:
                team_records = load_current_team_records(SEASON)
            except Exception:
                logger.warning("Could not load team records for blend", exc_info=True)
                team_records = None

            power = build_power_rankings(
                elo_ratings=elo_current,
                profiles=profiles,
                current_roster=pd.read_parquet(roster_path),
                hitter_projections=(
                    pd.read_parquet(h_proj_path) if h_proj_path.exists()
                    else pd.DataFrame()
                ),
                pitcher_projections=(
                    pd.read_parquet(p_proj_path) if p_proj_path.exists()
                    else pd.DataFrame()
                ),
                team_records=team_records,
            )
            if not power.empty:
                power.to_parquet(
                    DASHBOARD_DIR / "team_power_rankings.parquet",
                    index=False,
                )
                _merge_power_into_team_rankings(power)
                logger.info("Saved power rankings: %d teams", len(power))
        else:
            logger.warning(
                "Skipping power rankings — missing roster.parquet",
            )
    except Exception:
        logger.exception("Failed to update power rankings")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-season daily update")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--skip-schedule", action="store_true",
                        help="Skip fetching schedule/lineups from MLB API.")
    parser.add_argument("--weekly", action="store_true",
                        help="Run weekly refresh: player, team, and prospect rankings.")
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    logger.info("=" * 60)
    logger.info("In-season update for %s (season %d)", game_date, SEASON)
    logger.info("=" * 60)

    # Step 1: Update rate projections
    logger.info("Step 1: Updating rate projections...")
    h_updated, p_updated = update_projections_step()

    if not h_updated.empty:
        # Drop snapshot columns if present
        for col in ["snapshot_date", "target_season"]:
            if col in h_updated.columns:
                h_updated = h_updated.drop(columns=[col])
        h_updated.to_parquet(DASHBOARD_DIR / "hitter_projections.parquet", index=False)
        logger.info("Saved updated hitter projections: %d rows", len(h_updated))

    if not p_updated.empty:
        for col in ["snapshot_date", "target_season"]:
            if col in p_updated.columns:
                p_updated = p_updated.drop(columns=[col])
        p_updated.to_parquet(DASHBOARD_DIR / "pitcher_projections.parquet", index=False)
        logger.info("Saved updated pitcher projections: %d rows", len(p_updated))

    # Step 2: Update K% samples
    logger.info("Step 2: Updating pitcher K%% samples...")
    k_samples = update_k_samples_step()
    if k_samples:
        np.savez_compressed(
            DASHBOARD_DIR / "pitcher_k_samples.npz",
            **k_samples,
        )
        logger.info("Saved updated K%% samples for %d pitchers", len(k_samples))

    # Step 2b: Update BB% and HR/BF samples
    logger.info("Step 2b: Updating pitcher BB%% and HR/BF samples...")
    bb_samples = update_player_rate_samples_step(
        "pitcher", "bb", trials_col="batters_faced", successes_col="walks",
        league_avg=0.08,
    )
    if bb_samples:
        np.savez_compressed(DASHBOARD_DIR / "pitcher_bb_samples.npz", **bb_samples)
        logger.info("Saved updated BB%% samples for %d pitchers", len(bb_samples))

    hr_samples = update_player_rate_samples_step(
        "pitcher", "hr", trials_col="batters_faced", successes_col="hr",
        league_avg=0.03,
    )
    if hr_samples:
        np.savez_compressed(DASHBOARD_DIR / "pitcher_hr_samples.npz", **hr_samples)
        logger.info("Saved updated HR/BF samples for %d pitchers", len(hr_samples))

    # Step 2c: Update hitter K%, BB%, HR samples
    logger.info("Step 2c: Updating hitter K%%, BB%%, HR samples...")
    h_k_samples = update_player_rate_samples_step(
        "hitter", "k", trials_col="pa", successes_col="strikeouts",
        league_avg=0.226,
    )
    if h_k_samples:
        np.savez_compressed(DASHBOARD_DIR / "hitter_k_samples.npz", **h_k_samples)
        logger.info("Saved updated hitter K%% samples for %d batters", len(h_k_samples))

    h_bb_samples = update_player_rate_samples_step(
        "hitter", "bb", trials_col="pa", successes_col="walks",
        league_avg=0.082,
    )
    if h_bb_samples:
        np.savez_compressed(DASHBOARD_DIR / "hitter_bb_samples.npz", **h_bb_samples)
        logger.info("Saved updated hitter BB%% samples for %d batters", len(h_bb_samples))

    h_hr_samples = update_player_rate_samples_step(
        "hitter", "hr", trials_col="pa", successes_col="hr",
        league_avg=0.031,
    )
    if h_hr_samples:
        np.savez_compressed(DASHBOARD_DIR / "hitter_hr_samples.npz", **h_hr_samples)
        logger.info("Saved updated hitter HR samples for %d batters", len(h_hr_samples))

    # Step 3: Update team assignments
    logger.info("Step 3: Updating team assignments...")
    try:
        from src.data.queries import get_player_teams
        teams = get_player_teams(SEASON - 1)  # will pick up 2026 data when available
        teams.to_parquet(DASHBOARD_DIR / "player_teams.parquet", index=False)
        logger.info("Updated player teams: %d players", len(teams))
    except Exception as e:
        logger.warning("Team update failed: %s", e)

    # Step 4: Fetch schedule and lineups
    # Game simulations are handled by confident_picks.py (precompute),
    # which writes game_props.parquet — the single source of truth for
    # all prop predictions. This step only fetches schedule/lineup data
    # consumed by the dashboard and precompute pipeline.
    if not args.skip_schedule:
        logger.info("Step 4: Fetching schedule for %s...", game_date)
        schedule, lineups = fetch_schedule_step(game_date)

        if not schedule.empty:
            schedule.to_parquet(DASHBOARD_DIR / "todays_games.parquet", index=False)
            logger.info("Saved schedule: %d games", len(schedule))

            if not lineups.empty:
                lineups.to_parquet(DASHBOARD_DIR / "todays_lineups.parquet", index=False)
                logger.info("Saved lineups: %d entries", len(lineups))
        else:
            logger.info("No games today")
    else:
        logger.info("Step 4: Skipped (--skip-schedule)")

    # Step 5: Update 2026 season stats (traditional stats for leaders page)
    logger.info("Step 5: Updating 2026 season stats...")
    h_trad, p_trad = update_season_stats_step()

    # Step 5b: Update advanced/Statcast stats for percentile populations
    logger.info("Step 5b: Updating advanced stats (Statcast + observed profiles)...")
    update_advanced_stats_step()

    # Step 6 (weekly only): Refresh player + team + prospect rankings
    if args.weekly:
        logger.info("Step 6: Weekly rankings refresh...")
        update_weekly_rankings_step()
        update_weekly_team_step()
        refresh_prospect_rankings()
    else:
        logger.info("Step 6: Skipped (use --weekly to refresh rankings)")

    # Save update metadata
    metadata = {
        "last_updated": datetime.now().isoformat(),
        "game_date": game_date,
        "season": SEASON,
        "hitters_updated": len(h_updated) if not h_updated.empty else 0,
        "pitchers_updated": len(p_updated) if not p_updated.empty else 0,
        "pitcher_k_samples_count": len(k_samples),
        "pitcher_bb_samples_count": len(bb_samples),
        "pitcher_hr_samples_count": len(hr_samples),
        "hitter_k_samples_count": len(h_k_samples),
        "hitter_bb_samples_count": len(h_bb_samples),
        "hitter_hr_samples_count": len(h_hr_samples),
        "hitter_trad_count": len(h_trad),
        "pitcher_trad_count": len(p_trad),
        "weekly_refresh": args.weekly,
    }
    meta_path = DASHBOARD_DIR / "update_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved update metadata to %s", meta_path)

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
