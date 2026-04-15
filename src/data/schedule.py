"""
Schedule and lineup data for in-season dashboard.

Fetches today's games, probable pitchers, and lineups from the MLB Stats API.
Falls back gracefully when data isn't available.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
MLB_API_BASE_V11 = "https://statsapi.mlb.com/api/v1.1"


# MLB wind strings look like "14 mph, In From LF", "0 mph, None",
# "7 mph, L To R", "5 mph, Varies". Direction substrings are what
# drive the (in/out/cross/none) bucketing used downstream.
_WIND_SPEED_RE = re.compile(r"(\d+)\s*mph", re.IGNORECASE)


def parse_wind_string(wind_str: str | None) -> tuple[int | None, str]:
    """Parse an MLB wind string into (speed_mph, direction_category).

    Parameters
    ----------
    wind_str : str | None
        Raw wind string from gameData.weather.wind, e.g.
        "14 mph, In From LF" or "0 mph, None" or None.

    Returns
    -------
    (speed, category)
        speed: int mph or None when unparseable.
        category: one of "in", "out", "cross", "none", "unknown".
        Categories match production.dim_weather.wind_category and the
        weather_effects.parquet bucketing, so downstream lookups
        resolve.
    """
    if not wind_str:
        return (None, "unknown")

    s = str(wind_str)

    # Speed extraction — tolerant of whitespace and missing 'mph'
    speed_match = _WIND_SPEED_RE.search(s)
    speed = int(speed_match.group(1)) if speed_match else None

    # Direction classification via substring match (same patterns the
    # ETL's load_weather.sql uses when populating dim_weather)
    upper = s.upper()
    if "OUT TO" in upper:
        direction = "out"
    elif "IN FROM" in upper:
        direction = "in"
    elif "L TO R" in upper or "R TO L" in upper:
        direction = "cross"
    elif "NONE" in upper or "CALM" in upper:
        direction = "none"
    elif "VARIES" in upper:
        direction = "unknown"
    else:
        direction = "unknown"

    # 0 mph always collapses to "none" regardless of the text
    if speed == 0:
        direction = "none"

    return (speed, direction)


def fetch_game_weather(game_pk: int) -> dict[str, Any]:
    """Fetch weather for a single game from MLB's /feed/live endpoint.

    The /schedule endpoint does NOT return weather, but /feed/live
    exposes ``gameData.weather`` as soon as MLB publishes the venue's
    game-day report (typically 2-4 hours before first pitch on game
    day). Future-day games return an empty block.

    Parameters
    ----------
    game_pk : int
        MLB game primary key.

    Returns
    -------
    dict
        Keys: weather_temp (int or None), weather_wind_speed (int or
        None), weather_wind_direction (str), weather_condition (str).
        Missing fields left as None / "" so the caller can left-join
        cleanly.
    """
    import urllib.request

    url = f"{MLB_API_BASE_V11}/game/{game_pk}/feed/live"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.debug("feed/live weather fetch failed for %d: %s", game_pk, e)
        return {
            "weather_temp": None,
            "weather_wind_speed": None,
            "weather_wind_direction": "",
            "weather_condition": "",
        }

    wx = (data.get("gameData") or {}).get("weather") or {}

    temp_raw = wx.get("temp")
    try:
        temp = int(temp_raw) if temp_raw not in (None, "") else None
    except (TypeError, ValueError):
        temp = None

    wind_speed, wind_direction = parse_wind_string(wx.get("wind"))

    return {
        "weather_temp": temp,
        "weather_wind_speed": wind_speed,
        "weather_wind_direction": wind_direction,
        "weather_condition": str(wx.get("condition") or ""),
    }


def fetch_todays_schedule(
    game_date: str | None = None,
    include_weather: bool = True,
) -> pd.DataFrame:
    """Fetch today's MLB schedule from the Stats API.

    Parameters
    ----------
    game_date : str | None
        Date as 'YYYY-MM-DD'. Defaults to today.
    include_weather : bool
        When True (default), follow-up to /feed/live per game to
        attach weather columns. Set False to skip the extra HTTP calls
        (useful for lightweight schedule probes that don't feed the
        sim). Missing weather (future days, endpoint failure) leaves
        the fields as None / empty string.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, game_date, game_time, status,
        away_team_id, away_team_name, away_abbr,
        home_team_id, home_team_name, home_abbr,
        away_pitcher_id, away_pitcher_name,
        home_pitcher_id, home_pitcher_name,
        hp_umpire_name, venue_id,
        weather_temp, weather_wind_speed, weather_wind_direction,
        weather_condition.
    """
    import urllib.request

    if game_date is None:
        game_date = date.today().isoformat()

    url = (
        f"{MLB_API_BASE}/schedule"
        f"?date={game_date}&sportId=1"
        f"&hydrate=probablePitcher,team,officials"
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

            # Home plate umpire (available once crew is assigned, usually day-of)
            hp_umpire_name = ""
            for official in game.get("officials", []):
                if official.get("officialType") == "Home Plate":
                    hp_umpire_name = official.get("official", {}).get("fullName", "")
                    break

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
                "hp_umpire_name": hp_umpire_name,
                "venue_id": game.get("venue", {}).get("id"),
            })

    df = pd.DataFrame(rows)

    if include_weather and not df.empty:
        df = _attach_weather(df)

    logger.info("Fetched %d games for %s", len(df), game_date)
    return df


def _attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Hit /feed/live per game_pk in parallel and attach weather columns.

    Empty weather blocks (future-day games, endpoint failures) leave
    fields as None / empty string so downstream temp/wind bucket
    helpers fall through to zero lift.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    game_pks = [int(x) for x in df["game_pk"].dropna().unique().tolist()]
    weather_by_gpk: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(fetch_game_weather, gpk): gpk for gpk in game_pks
        }
        for fut in as_completed(futures):
            gpk = futures[fut]
            try:
                weather_by_gpk[gpk] = fut.result()
            except Exception as e:
                logger.debug("weather fetch failed for %d: %s", gpk, e)

    for col in (
        "weather_temp", "weather_wind_speed",
        "weather_wind_direction", "weather_condition",
    ):
        df[col] = df["game_pk"].map(
            lambda g, _c=col: weather_by_gpk.get(int(g), {}).get(
                _c, None if "wind_speed" in _c or "temp" in _c else ""
            )
        )

    n_with_temp = df["weather_temp"].notna().sum()
    n_with_dir = (df["weather_wind_direction"] != "").sum()
    logger.info(
        "Weather attached: %d/%d with temp, %d/%d with direction",
        n_with_temp, len(df), n_with_dir, len(df),
    )
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
