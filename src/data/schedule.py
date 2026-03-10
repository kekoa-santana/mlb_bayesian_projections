"""
Schedule and lineup data for in-season dashboard.

Fetches today's games, probable pitchers, and lineups from the MLB Stats API.
Falls back gracefully when data isn't available.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def fetch_todays_schedule(
    game_date: str | None = None,
) -> pd.DataFrame:
    """Fetch today's MLB schedule from the Stats API.

    Parameters
    ----------
    game_date : str | None
        Date as 'YYYY-MM-DD'. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, game_date, game_time, status,
        away_team_id, away_team_name, away_abbr,
        home_team_id, home_team_name, home_abbr,
        away_pitcher_id, away_pitcher_name,
        home_pitcher_id, home_pitcher_name.
    """
    import urllib.request

    if game_date is None:
        game_date = date.today().isoformat()

    url = (
        f"{MLB_API_BASE}/schedule"
        f"?date={game_date}&sportId=1"
        f"&hydrate=probablePitcher,team"
    )

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error("Failed to fetch schedule from MLB API: %s", e)
        return pd.DataFrame()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            gpk = game.get("gamePk")
            status = game.get("status", {}).get("detailedState", "")
            game_dt = game.get("gameDate", "")

            # Parse game time
            game_time = ""
            if game_dt:
                try:
                    dt = datetime.fromisoformat(game_dt.replace("Z", "+00:00"))
                    # Convert UTC to ET (UTC-4 during DST, UTC-5 otherwise)
                    from datetime import timedelta
                    et_dt = dt - timedelta(hours=4)
                    hour = et_dt.hour
                    minute = et_dt.minute
                    ampm = "AM" if hour < 12 else "PM"
                    display_hour = hour % 12 or 12
                    game_time = f"{display_hour}:{minute:02d} {ampm} ET"
                except Exception:
                    game_time = game_dt

            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})

            away_team = away.get("team", {})
            home_team = home.get("team", {})

            # Probable pitchers
            away_pp = away.get("probablePitcher", {})
            home_pp = home.get("probablePitcher", {})

            rows.append({
                "game_pk": gpk,
                "game_date": game_date,
                "game_time": game_time,
                "status": status,
                "away_team_id": away_team.get("id"),
                "away_team_name": away_team.get("name", ""),
                "away_abbr": away_team.get("abbreviation", ""),
                "home_team_id": home_team.get("id"),
                "home_team_name": home_team.get("name", ""),
                "home_abbr": home_team.get("abbreviation", ""),
                "away_pitcher_id": away_pp.get("id"),
                "away_pitcher_name": away_pp.get("fullName", ""),
                "home_pitcher_id": home_pp.get("id"),
                "home_pitcher_name": home_pp.get("fullName", ""),
            })

    df = pd.DataFrame(rows)
    logger.info("Fetched %d games for %s", len(df), game_date)
    return df


def fetch_game_lineups(
    game_pk: int,
) -> pd.DataFrame:
    """Fetch lineup for a specific game from the Stats API.

    Parameters
    ----------
    game_pk : int
        MLB game primary key.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, team_id, team_abbr, batting_order,
        batter_id, batter_name.
    """
    import urllib.request

    url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Failed to fetch lineup for game %d: %s", game_pk, e)
        return pd.DataFrame()

    rows = []
    for side in ("away", "home"):
        team_data = data.get("teams", {}).get(side, {})
        team_info = team_data.get("team", {})
        team_id = team_info.get("id")
        team_abbr = team_info.get("abbreviation", "")

        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        for order, pid in enumerate(batting_order[:9], 1):
            pid_key = f"ID{pid}"
            player_info = players.get(pid_key, {}).get("person", {})
            rows.append({
                "game_pk": game_pk,
                "team_id": team_id,
                "team_abbr": team_abbr,
                "batting_order": order,
                "batter_id": pid,
                "batter_name": player_info.get("fullName", "Unknown"),
            })

    return pd.DataFrame(rows)


def fetch_all_lineups(
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch lineups for all games in a schedule.

    Only fetches for games that have started or have lineups posted.

    Parameters
    ----------
    schedule_df : pd.DataFrame
        Schedule with game_pk column.

    Returns
    -------
    pd.DataFrame
        Combined lineup data for all games.
    """
    if schedule_df.empty:
        return pd.DataFrame()

    frames = []
    for gpk in schedule_df["game_pk"].unique():
        lu = fetch_game_lineups(int(gpk))
        if not lu.empty:
            frames.append(lu)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
