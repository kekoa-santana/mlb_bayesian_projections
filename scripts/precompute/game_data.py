"""Precompute: Player teams, game logs, BF priors, umpire, weather."""
from __future__ import annotations

import logging

import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.game_data")


def run_player_teams(
    *,
    from_season: int = FROM_SEASON,
) -> None:
    """Compute player-team mapping (regular season + spring training override)."""
    from src.data.queries import get_player_teams

    logger.info("=" * 60)
    logger.info("Computing player-team mapping...")
    player_teams = get_player_teams(from_season)

    # Override with spring training teams for players who changed teams
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
        """, {"season": from_season + 1})
        if not st_teams.empty:
            st_lookup = dict(zip(st_teams["player_id"], st_teams["team_abbr"]))
            st_name_lookup = dict(zip(st_teams["player_id"], st_teams["team_name"]))
            n_updated = 0
            for idx, row in player_teams.iterrows():
                pid = row["player_id"]
                if pid in st_lookup and st_lookup[pid] and st_lookup[pid] != row["team_abbr"]:
                    player_teams.at[idx, "team_abbr"] = st_lookup[pid]
                    player_teams.at[idx, "team_name"] = st_name_lookup.get(pid, "")
                    n_updated += 1
            # Also add players who are in spring training but not in regular season
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


def run_game_logs(
    *,
    from_season: int = FROM_SEASON,
) -> None:
    """Compute game browser data (lineups + batter Ks for historical games)."""
    from src.data.queries import (
        get_game_batter_ks,
        get_game_lineups,
        get_pitcher_game_logs,
    )

    logger.info("=" * 60)
    logger.info("Computing game browser data for %d...", from_season)
    game_lineups = get_game_lineups(from_season)
    game_lineups.to_parquet(DASHBOARD_DIR / "game_lineups.parquet", index=False)
    logger.info("Saved game lineups: %d rows (%d games)",
                len(game_lineups), game_lineups["game_pk"].nunique())

    game_batter_ks = get_game_batter_ks(from_season)
    game_batter_ks.to_parquet(DASHBOARD_DIR / "game_batter_ks.parquet", index=False)
    logger.info("Saved game batter Ks: %d rows", len(game_batter_ks))

    # Pitcher game logs (used by Game Browser for game selection)
    pitcher_game_logs = get_pitcher_game_logs(from_season)
    pitcher_game_logs.to_parquet(DASHBOARD_DIR / "pitcher_game_logs.parquet", index=False)
    logger.info("Saved pitcher game logs: %d rows", len(pitcher_game_logs))

    # Game info (date, teams)
    from src.data.db import read_sql as _read_sql_gi
    game_info = _read_sql_gi("""
        SELECT game_pk, game_date, season,
               home_team_id, away_team_id,
               home_team_name, away_team_name
        FROM production.dim_game
        WHERE game_type = 'R' AND season = :season
    """, {"season": from_season})
    game_info.to_parquet(DASHBOARD_DIR / "game_info.parquet", index=False)
    logger.info("Saved game info: %d games", len(game_info))


def run_bf_priors(
    *,
    seasons: list[int] = SEASONS,
) -> None:
    """Compute BF priors."""
    from src.data.queries import get_pitcher_game_logs
    from src.models.bf_model import compute_pitcher_bf_priors

    logger.info("=" * 60)
    logger.info("Computing BF priors...")
    game_logs_list = []
    for s in seasons:
        gl = get_pitcher_game_logs(s)
        game_logs_list.append(gl)
    game_logs = pd.concat(game_logs_list, ignore_index=True)

    # Compute pitcher P/PA from game logs for the projection season
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


def run_umpire(
    *,
    from_season: int = FROM_SEASON,
) -> None:
    """Compute umpire K-rate tendencies."""
    from src.data.queries import get_umpire_k_tendencies

    logger.info("=" * 60)
    logger.info("Computing umpire K-rate tendencies...")
    umpire_tendencies = get_umpire_k_tendencies(
        seasons=list(range(2021, from_season + 1)), min_games=30,
    )
    umpire_tendencies.to_parquet(DASHBOARD_DIR / "umpire_tendencies.parquet", index=False)
    logger.info("Saved umpire tendencies: %d umpires", len(umpire_tendencies))


def run_weather(
    *,
    seasons: list[int] = SEASONS,
) -> None:
    """Compute weather effects on K and HR rates."""
    from src.data.queries import get_weather_effects

    logger.info("=" * 60)
    logger.info("Computing weather effects...")
    weather_effects = get_weather_effects(seasons=seasons)
    weather_effects.to_parquet(DASHBOARD_DIR / "weather_effects.parquet", index=False)
    logger.info("Saved weather effects: %d combinations", len(weather_effects))
