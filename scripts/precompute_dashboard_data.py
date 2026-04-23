#!/usr/bin/env python
"""
Pre-compute all data needed by the Streamlit dashboard.

Runs 15+ stages: Bayesian model fitting (K%, BB%, GB%, FB%, HR/FB, HR/BF),
posterior samples, BF priors, projections, rankings, scouting grades, prospects,
team ELO/profiles/power rankings, game simulations, and prop confidence picks.
Saves 70+ parquets to tdd-dashboard/data/dashboard/.

Usage
-----
    python scripts/precompute_dashboard_data.py          # full quality
    python scripts/precompute_dashboard_data.py --quick   # fast iteration
    python scripts/precompute_dashboard_data.py --include team,rankings
    python scripts/precompute_dashboard_data.py --list-groups
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the scripts/ directory is on sys.path so `import precompute` works
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from precompute import (
    DASHBOARD_DIR,
    FROM_SEASON,
    SEASONS,
    precache_profiles,
)
from precompute import models, projections, samples, team, profiles
from precompute import rankings, game_data, traditional, game_sim
from precompute import prospects, snapshots, glicko, confident_picks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")


_SECTION_GROUPS: dict[str, list[str]] = {
    "models":      ["hitter_models", "pitcher_models"],
    "projections": ["hitter_proj", "pitcher_proj", "counting", "sim_pitcher", "sim_hitter"],
    "samples":     ["pitcher_samples", "hitter_samples"],
    "team":        ["team_elo", "series_elo", "team_profiles", "league_sim", "team_power", "team_sim", "depth_chart", "roster"],
    "profiles":    ["arsenal", "vuln", "archetypes", "zones", "matchup_advantage"],
    "rankings":    ["player_rankings", "breakouts"],
    "weekly_form": ["weekly_form"],
    "daily_standouts": ["daily_standouts"],
    "game_data":   ["player_teams", "game_logs", "bf_priors", "outs_priors", "umpire", "weather", "catcher_framing"],
    "traditional": ["trad_stats", "agg_eff"],
    "glicko":      ["player_glicko"],
    "historical":  ["historical_all"],
    "game_sim":    ["game_sim_data"],
    "picks":       ["confident_picks"],
    "prospects":   ["prospect_data"],
    "health":      ["health_parks"],
    "snapshots":   ["projection_snapshots"],
}

# Flatten: section_name -> group_name for help text
_ALL_SECTIONS = set()
for _secs in _SECTION_GROUPS.values():
    _ALL_SECTIONS.update(_secs)


def parse_args() -> argparse.Namespace:
    group_names = sorted(_SECTION_GROUPS.keys())
    parser = argparse.ArgumentParser(
        description="Pre-compute dashboard data",
        epilog=f"Available groups: {', '.join(group_names)}",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fewer MCMC draws for fast iteration",
    )
    parser.add_argument(
        "--include", type=str, default="",
        help="Comma-separated groups to run (e.g. --include team,rankings). "
             "Omit to run everything.",
    )
    parser.add_argument(
        "--list-groups", action="store_true",
        help="Print available groups and their sections, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_groups:
        print("Available groups for --include:\n")
        for group, sections in sorted(_SECTION_GROUPS.items()):
            print(f"  {group:14s}  {', '.join(sections)}")
        print(f"\nUsage: python {__file__} --include team,rankings")
        return

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # Build the set of sections to run
    _include_sections: set[str] | None = None  # None = run everything
    if args.include:
        _include_sections = set()
        for token in args.include.split(","):
            token = token.strip()
            if token in _SECTION_GROUPS:
                _include_sections.update(_SECTION_GROUPS[token])
            elif token in _ALL_SECTIONS:
                _include_sections.add(token)
            else:
                logger.warning("Unknown group/section: %s (skipping)", token)
        logger.info("Selective run -- sections: %s", ", ".join(sorted(_include_sections)))

    def should_run(section: str) -> bool:
        """Return True if this section should execute."""
        if _include_sections is None:
            return True
        return section in _include_sections

    if args.quick:
        draws, tune, chains = 500, 250, 2
        sim_seasons = 1_000
        logger.info("QUICK mode: draws=%d, tune=%d, chains=%d, sim_seasons=%d",
                     draws, tune, chains, sim_seasons)
    else:
        draws, tune, chains = 2000, 1000, 4
        sim_seasons = 10_000
        logger.info("FULL mode: draws=%d, tune=%d, chains=%d, sim_seasons=%d",
                     draws, tune, chains, sim_seasons)

    # =================================================================
    # 0. Pre-cache observed profiles (needed by projection enrichment)
    # =================================================================
    precache_profiles()

    # -- Load existing outputs as fallbacks when sections are skipped --
    hitter_proj = pd.DataFrame()
    pitcher_proj = pd.DataFrame()
    hitter_results: dict = {}
    pitcher_results: dict = {}
    try:
        hitter_proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")
    except FileNotFoundError:
        pass
    try:
        pitcher_proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")
    except FileNotFoundError:
        pass

    # =================================================================
    # 1-2. Hitter + Pitcher composite models and projections
    # =================================================================
    if should_run("hitter_models") or should_run("hitter_proj") or \
       should_run("pitcher_models") or should_run("pitcher_proj"):
        hitter_results, pitcher_results, hitter_proj, pitcher_proj = models.run(
            seasons=SEASONS, from_season=FROM_SEASON,
            draws=draws, tune=tune, chains=chains,
            include_hitter=should_run("hitter_models") or should_run("hitter_proj"),
            include_pitcher=should_run("pitcher_models") or should_run("pitcher_proj"),
        )

    # =================================================================
    # 2a. Health scores + park factors
    # =================================================================
    health_df = pd.DataFrame()
    park_factors_df: pd.DataFrame | None = None
    hitter_venues_df: pd.DataFrame | None = None
    if should_run("health_parks"):
        health_df, park_factors_df, hitter_venues_df = projections.run_health_parks(
            from_season=FROM_SEASON,
        )

    # =================================================================
    # 2b. Counting stat projections
    # =================================================================
    if should_run("counting"):
        projections.run_counting(
            hitter_results=hitter_results,
            pitcher_results=pitcher_results,
            health_df=health_df,
            park_factors=park_factors_df,
            hitter_venues=hitter_venues_df,
            seasons=SEASONS,
            from_season=FROM_SEASON,
        )

    # =================================================================
    # 2c. Sim-based pitcher counting stats + reliever roles & rankings
    # =================================================================
    pitcher_roles_df: pd.DataFrame | None = None
    if should_run("sim_pitcher"):
        pitcher_roles_df = projections.run_sim_pitcher(
            pitcher_results=pitcher_results,
            health_df=health_df,
            seasons=SEASONS,
            from_season=FROM_SEASON,
            n_seasons=sim_seasons,
        )

    # =================================================================
    # 2d. Sim-based hitter counting stats
    # =================================================================
    if should_run("sim_hitter"):
        projections.run_sim_hitter(
            hitter_results=hitter_results,
            health_df=health_df,
            seasons=SEASONS,
            from_season=FROM_SEASON,
            n_seasons=sim_seasons,
        )

    # =================================================================
    # 3. Pitcher posterior samples (BB%, HR/BF, K%)
    # =================================================================
    if should_run("pitcher_samples"):
        samples.run_pitcher_samples(
            pitcher_results=pitcher_results,
            seasons=SEASONS,
            from_season=FROM_SEASON,
            draws=draws, tune=tune, chains=chains,
        )

    # =================================================================
    # 4. BF priors
    # =================================================================
    if should_run("bf_priors"):
        game_data.run_bf_priors(seasons=SEASONS)

    # =================================================================
    # 4a-ii. Outs priors (outs-anchored exit model)
    # =================================================================
    if should_run("outs_priors"):
        game_data.run_outs_priors(seasons=SEASONS)

    # =================================================================
    # 4b. Umpire K-rate tendencies
    # =================================================================
    if should_run("umpire"):
        game_data.run_umpire(from_season=FROM_SEASON)

    # =================================================================
    # 4c. Weather effects
    # =================================================================
    if should_run("weather"):
        game_data.run_weather(seasons=SEASONS)

    # =================================================================
    # 4c-ii. Catcher framing effects
    # =================================================================
    if should_run("catcher_framing"):
        game_data.run_catcher_framing(seasons=SEASONS)

    # =================================================================
    # 4d. Player team mapping
    # =================================================================
    if should_run("player_teams"):
        game_data.run_player_teams(from_season=FROM_SEASON)

    # =================================================================
    # 4e. Game browser data
    # =================================================================
    if should_run("game_logs"):
        game_data.run_game_logs(from_season=FROM_SEASON)

    # =================================================================
    # 5. Matchup profiles (arsenal, vulnerability, strength)
    # =================================================================
    if should_run("arsenal") or should_run("vuln"):
        profiles.run_arsenal_vuln(from_season=FROM_SEASON, seasons=SEASONS)

    # =================================================================
    # 5b. Location grid data
    # =================================================================
    if should_run("zones"):
        profiles.run_zones(from_season=FROM_SEASON, seasons=SEASONS)

    # =================================================================
    # 5c-5e. Archetype-based matchup profiles
    # =================================================================
    if should_run("archetypes"):
        profiles.run_archetypes(from_season=FROM_SEASON, seasons=SEASONS)

    # =================================================================
    # 5d. Matchup advantage data (platoon splits, baselines, scales)
    # =================================================================
    if should_run("matchup_advantage"):
        profiles.run_matchup_advantage_data(from_season=FROM_SEASON, seasons=SEASONS)

    # =================================================================
    # 6. Traditional / actual stats
    # =================================================================
    if should_run("trad_stats"):
        traditional.run_trad_stats(from_season=FROM_SEASON)

    # =================================================================
    # 6b. Aggressiveness & efficiency
    # =================================================================
    if should_run("agg_eff"):
        traditional.run_agg_eff(from_season=FROM_SEASON)

    # =================================================================
    # 6c. Multi-season historical data
    # =================================================================
    if should_run("historical_all"):
        traditional.run_historical_all(seasons=SEASONS)

    # =================================================================
    # 6d-6f. Prospect data
    # =================================================================
    if should_run("prospect_data"):
        prospects.run(from_season=FROM_SEASON)

    # =================================================================
    # 6f2. Glicko-2 player ratings (before rankings so they can consume)
    # =================================================================
    if should_run("player_glicko"):
        glicko.run_glicko(from_season=FROM_SEASON)

    # =================================================================
    # 6g-6h. MLB positional rankings
    # =================================================================
    if should_run("player_rankings"):
        rankings.run_player_rankings(
            health_df=health_df,
            pitcher_roles_df=pitcher_roles_df,
            from_season=FROM_SEASON,
        )

    # =================================================================
    # 6i-6j. Breakout candidates
    # =================================================================
    if should_run("breakouts"):
        rankings.run_breakouts(from_season=FROM_SEASON)

    # =================================================================
    # 6j2. Weekly form leaderboards (separate from frozen core rank)
    # =================================================================
    if should_run("weekly_form"):
        rankings.run_weekly_form(days=14)

    # =================================================================
    # 6j3. Daily standouts (single-game performances)
    # =================================================================
    if should_run("daily_standouts"):
        rankings.run_daily_standouts()

    # =================================================================
    # 6k. Team ELO (game-level)
    # =================================================================
    elo_current = None
    elo_history = None
    if should_run("team_elo"):
        elo_current, _elo_preseason, elo_history = team.run_team_elo()

    # =================================================================
    # 6k2. Series ELO (series-level)
    # =================================================================
    if should_run("series_elo"):
        team.run_series_elo()

    # =================================================================
    # 6l. Team profiles + rankings
    # =================================================================
    if should_run("team_profiles"):
        team.run_team_profiles(
            elo_history=elo_history,
            elo_current=elo_current,
            pitcher_roles_df=pitcher_roles_df,
            from_season=FROM_SEASON,
        )

    # =================================================================
    # 6m. League season simulator (zero-sum Bernoulli on actual schedule)
    #     Must run BEFORE power rankings so league_sim results are available.
    # =================================================================
    league_sim_df: pd.DataFrame | None = None
    if should_run("league_sim"):
        league_sim_df = team.run_league_sim(n_sims=1000)

    # =================================================================
    # 6n. Team season simulator (injury cascading, per-team MC)
    #     Must run BEFORE power rankings so team_sim results are available.
    # =================================================================
    team_sim_df: pd.DataFrame | None = None
    if should_run("team_sim"):
        team_sim_df = team.run_team_sim(health_df=health_df, n_sims=1000)

    # =================================================================
    # 6o. Power rankings (uses league sim + team sim wins)
    # =================================================================
    if should_run("team_power"):
        team.run_team_power(league_sim_df=league_sim_df)

    # =================================================================
    # 7. Preseason snapshot
    # =================================================================
    if should_run("projection_snapshots"):
        snapshots.run_projection_snapshots(
            hitter_proj=hitter_proj,
            pitcher_proj=pitcher_proj,
            from_season=FROM_SEASON,
        )

    # =================================================================
    # 8. Backtest summaries
    # =================================================================
    if should_run("backtest_summaries"):
        snapshots.run_backtest_summaries()
    else:
        logger.info("Skipping backtest_summaries (not in --include list)")

    # =================================================================
    # 9a. Hitter posterior samples
    # =================================================================
    if should_run("hitter_samples"):
        samples.run_hitter_samples(hitter_results=hitter_results)

    # =================================================================
    # 9b-9f. Game simulator data
    # =================================================================
    if should_run("game_sim_data"):
        game_sim.run(seasons=SEASONS, from_season=FROM_SEASON)

    # =================================================================
    # 9g. Confident picks (daily game-level prop picks)
    # =================================================================
    if should_run("confident_picks"):
        confident_picks.run(n_sims=10_000)

    # =================================================================
    # 8 (reordered). Depth chart
    # =================================================================
    if should_run("depth_chart"):
        team.run_depth_chart(
            hitter_proj=hitter_proj,
            from_season=FROM_SEASON,
        )

    # =================================================================
    # 10. Full MLB roster
    # =================================================================
    if should_run("roster"):
        team.run_roster()

    # =================================================================
    # Summary
    # =================================================================
    logger.info("=" * 60)
    logger.info("Dashboard pre-computation complete!")
    logger.info("  Hitter projections:  %d players", len(hitter_proj))
    logger.info("  Pitcher projections: %d players", len(pitcher_proj))
    logger.info("  Output dir:          %s", DASHBOARD_DIR)


if __name__ == "__main__":
    main()

# =========================================================================
# PRECOMPUTE GROUP REFERENCE
# =========================================================================
#
# Usage: python scripts/precompute_dashboard_data.py --include <groups>
#        Add --quick for faster MCMC (500 draws vs 2000, good for iteration)
#
# -------------------------------------------------------------------------
# GROUP            SECTIONS                    WHAT IT DOES
# -------------------------------------------------------------------------
# models           hitter_models,              Bayesian MCMC for K%, BB%,
#                  pitcher_models              GB%, FB%, HR/FB, wOBA, chase.
#                                              ~3 min (quick), ~15 min (full).
#                                              REQUIRED before projections.
#
# projections      hitter_proj, pitcher_proj,  Composite projections, counting
#                  counting, sim_pitcher,      stats (rate x PA), sim-based
#                  sim_hitter                  season projections with BIP
#                                              profiles + Marcel HR + BABIP.
#                                              ~2 min. Needs models.
#
# samples          pitcher_samples,            Save posterior rate samples as
#                  hitter_samples              NPZ for game sim / dashboard.
#
# team             team_elo, series_elo,       Full team pipeline: ELO ratings,
#                  team_profiles, team_power,  profiles (BaseRuns, pitcher-
#                  team_sim, depth_chart,      defense synergy), power rankings
#                  roster                      (confidence blend), MC season
#                                              sim (injury cascade), depth
#                                              chart, roster export.
#                                              ~6 min total.
#
# rankings         player_rankings, breakouts  Positional player rankings
#                                              (PA-weighted), breakout scoring.
#                                              Needs projections + sim.
#
# profiles         arsenal, vuln,              Pitch-level player profiles
#                  archetypes, zones           (arsenal, vulnerability, zone
#                                              grids). DB-heavy, ~5 min.
#
# game_data        player_teams, game_logs,    Game-level data: team mappings,
#                  bf_priors, umpire, weather  pitcher BF priors, umpire/weather
#                                              tendencies.
#
# traditional      trad_stats, agg_eff         Traditional stats + multi-season
#                                              historical datasets.
#
# glicko           player_glicko              Glicko-2 player ratings from PA
#                                              outcomes. ~1 min.
#
# game_sim         game_sim_data              Exit model, pitch count features,
#                                              TTO profiles, bullpen rates.
#                                              DB-heavy, ~2 min.
#
# picks            confident_picks            Daily confident prop picks from
#                                              pitcher + batter game sims.
#                                              Requires schedule from MLB API.
#                                              ~5 min (10K sims per game).
#
# prospects        prospect_data              MiLB translations, readiness
#                                              model, TDD prospect rankings,
#                                              prospect-to-MLB comps.
#
# health           health_parks               Health/durability scores + park
#                                              factors. Quick.
#
# historical       historical_all             Multi-season historical profiles
#                                              (all 2018-2025 stacked).
#
# snapshots        projection_snapshots       Preseason snapshot + backtest
#                                              CSV-to-parquet conversion.
#
# -------------------------------------------------------------------------
# COMMON WORKFLOWS
# -------------------------------------------------------------------------
#
# Full rebuild (everything):
#   python scripts/precompute_dashboard_data.py --quick
#   ~20-25 min. Run after major model changes or start of new season.
#
# Player projections only (after model tweaks):
#   python scripts/precompute_dashboard_data.py --include models,projections --quick
#   ~5 min. Regenerates all player-level projections.
#
# Team rankings only (after projection changes):
#   python scripts/precompute_dashboard_data.py --include team
#   ~6 min. ELO + profiles + power rankings + sim + depth chart.
#
# Quick team sim check:
#   python scripts/precompute_dashboard_data.py --include team_sim,team_power
#   ~5 min. Just the MC sim + power rankings (needs existing parquets).
#
# Player rankings + team (most common after any change):
#   python scripts/precompute_dashboard_data.py --include rankings,team
#   ~7 min. Updates player rankings then full team pipeline.
#
# Full dashboard update (models + everything downstream):
#   python scripts/precompute_dashboard_data.py --include models,projections,rankings,team --quick
#   ~15 min. The "safe" full update.
#
# List all groups:
#   python scripts/precompute_dashboard_data.py --list-groups
