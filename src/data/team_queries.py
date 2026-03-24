"""
Team-level SQL queries for team profiles, ELO, and rankings.

Every function returns a pandas DataFrame. All SQL lives here —
no raw query strings scattered through the codebase.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Game results (self-join home/away + SP identification)
# ---------------------------------------------------------------------------
def get_game_results(
    seasons: list[int] | None = None,
    include_postseason: bool = True,
) -> pd.DataFrame:
    """Game-level results with starting pitchers and venue.

    Self-joins fact_game_totals (home vs away), joins dim_game for
    venue/date, and fact_player_game_mlb for SP identification.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.  ``None`` returns all available (2000-2025).
    include_postseason : bool
        If True, include postseason games (Wild Card, Division Series,
        League Championship Series, World Series) in addition to regular
        season.  Default False for backward compatibility.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, game_date, season, game_type, venue_id,
        home_team_id, away_team_id, home_runs, away_runs, home_sp_id,
        away_sp_id.
    """
    season_filter = ""
    params: dict = {}
    if seasons:
        season_filter = "AND dg.season = ANY(:seasons)"
        params["seasons"] = seasons

    if include_postseason:
        game_type_filter = "AND dg.game_type IN ('R', 'F', 'D', 'L', 'W')"
    else:
        game_type_filter = "AND dg.game_type = 'R'"

    query = f"""
    SELECT
        dg.game_pk,
        dg.game_date,
        dg.season,
        dg.game_type,
        dg.venue_id,
        h.team_id  AS home_team_id,
        a.team_id  AS away_team_id,
        h.runs     AS home_runs,
        a.runs     AS away_runs,
        hsp.player_id AS home_sp_id,
        asp.player_id AS away_sp_id
    FROM production.fact_game_totals h
    JOIN production.fact_game_totals a
        ON h.game_pk = a.game_pk AND a.home_away = 'away'
    JOIN production.dim_game dg
        ON h.game_pk = dg.game_pk
    LEFT JOIN production.fact_player_game_mlb hsp
        ON hsp.game_pk = dg.game_pk
        AND hsp.team_id = h.team_id
        AND hsp.pit_is_starter = true
    LEFT JOIN production.fact_player_game_mlb asp
        ON asp.game_pk = dg.game_pk
        AND asp.team_id = a.team_id
        AND asp.pit_is_starter = true
    WHERE h.home_away = 'home'
      {game_type_filter}
      {season_filter}
    ORDER BY dg.game_date, dg.game_pk
    """
    label = "all+postseason" if include_postseason else "regular season"
    logger.info("Fetching game results (%s, %s)", seasons or "all", label)
    df = read_sql(query, params)
    logger.info("Game results: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Venue run factors (park factors based on scoring environment)
# ---------------------------------------------------------------------------
def get_venue_run_factors(
    seasons: list[int] | None = None,
    min_games: int = 100,
) -> pd.DataFrame:
    """Compute runs-based park factors per venue, regressed toward 1.0.

    Uses home vs away RPG ratio with Bayesian shrinkage.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.
    min_games : int
        Minimum home games at a venue to include.

    Returns
    -------
    pd.DataFrame
        Columns: venue_id, run_factor, home_games.
    """
    season_filter = ""
    params: dict = {"min_games": min_games}
    if seasons:
        season_filter = "AND dg.season = ANY(:seasons)"
        params["seasons"] = seasons

    query = f"""
    WITH venue_splits AS (
        SELECT
            dg.venue_id,
            fgt.home_away,
            COUNT(DISTINCT dg.game_pk) AS games,
            SUM(fgt.runs)::float / NULLIF(COUNT(DISTINCT dg.game_pk), 0) AS rpg
        FROM production.fact_game_totals fgt
        JOIN production.dim_game dg ON fgt.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.venue_id IS NOT NULL
          {season_filter}
        GROUP BY dg.venue_id, fgt.home_away
    )
    SELECT
        h.venue_id,
        h.rpg / NULLIF(a.rpg, 0) AS raw_factor,
        h.games AS home_games
    FROM venue_splits h
    JOIN venue_splits a ON h.venue_id = a.venue_id AND a.home_away = 'away'
    WHERE h.home_away = 'home'
      AND h.games >= :min_games
    """
    logger.info("Fetching venue run factors")
    df = read_sql(query, params)
    # Regress toward 1.0: shrinkage = games / (games + k)
    k = 200  # prior strength
    df["shrinkage"] = df["home_games"] / (df["home_games"] + k)
    df["run_factor"] = df["shrinkage"] * df["raw_factor"] + (1 - df["shrinkage"]) * 1.0
    df = df[["venue_id", "run_factor", "home_games"]].copy()
    logger.info("Venue run factors: %d venues", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Team season offense aggregates
# ---------------------------------------------------------------------------
def get_team_season_offense(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate offense stats per team-season from fact_game_totals.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, season, runs, hits, home_runs, walks,
        strikeouts, sb, plate_appearances, games.
    """
    season_filter = ""
    params: dict = {}
    if seasons:
        season_filter = "AND dg.season = ANY(:seasons)"
        params["seasons"] = seasons

    query = f"""
    SELECT
        fgt.team_id,
        dg.season,
        SUM(fgt.runs)           AS runs,
        SUM(fgt.hits)           AS hits,
        SUM(fgt.home_runs)      AS home_runs,
        SUM(fgt.walks)          AS walks,
        SUM(fgt.strikeouts)     AS strikeouts,
        SUM(fgt.sb)             AS sb,
        SUM(fgt.plate_appearances) AS plate_appearances,
        COUNT(DISTINCT fgt.game_pk) AS games
    FROM production.fact_game_totals fgt
    JOIN production.dim_game dg ON fgt.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
      {season_filter}
    GROUP BY fgt.team_id, dg.season
    ORDER BY dg.season, fgt.team_id
    """
    logger.info("Fetching team season offense")
    df = read_sql(query, params)
    logger.info("Team season offense: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 4. Team roster composition (acquisition types)
# ---------------------------------------------------------------------------
def get_team_roster_composition(season: int) -> pd.DataFrame:
    """Trace how each player joined their current organization.

    Looks at the *first* transaction bringing a player to their current
    team. Type codes mapped to human-readable categories.

    Parameters
    ----------
    season : int
        Season to analyze.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, player_id, acquisition_type, acquisition_date.
    """
    query = """
    WITH active_players AS (
        -- Players who appeared for each team in the given season
        SELECT DISTINCT team_id, player_id
        FROM production.fact_player_game_mlb
        WHERE season = :season
    ),
    first_arrival AS (
        -- Earliest transaction bringing each player to their team
        SELECT
            ap.team_id,
            ap.player_id,
            dt.type_code,
            dt.transaction_date,
            ROW_NUMBER() OVER (
                PARTITION BY ap.team_id, ap.player_id
                ORDER BY dt.transaction_date ASC
            ) AS rn
        FROM active_players ap
        JOIN production.dim_transaction dt
            ON dt.player_id = ap.player_id
            AND dt.to_team_id = ap.team_id::int
        WHERE dt.type_code IN ('DR', 'SGN', 'SFA', 'TR', 'CLW', 'R5', 'R5M', 'PUR')
    )
    SELECT
        team_id,
        player_id,
        type_code,
        transaction_date AS acquisition_date
    FROM first_arrival
    WHERE rn = 1
    """
    logger.info("Fetching roster composition for %d", season)
    df = read_sql(query, {"season": season})

    # Map type codes to categories
    acq_map = {
        "DR": "Draft",
        "SGN": "Signed",
        "SFA": "Free Agent",
        "TR": "Trade",
        "CLW": "Waiver",
        "R5": "Rule 5",
        "R5M": "Rule 5",
        "PUR": "Trade",
    }
    df["acquisition_type"] = df["type_code"].map(acq_map).fillna("Other")
    df = df[["team_id", "player_id", "acquisition_type", "acquisition_date"]]
    logger.info("Roster composition: %d player-team pairs", len(df))
    return df


# ---------------------------------------------------------------------------
# 5. Team IL history
# ---------------------------------------------------------------------------
def get_team_il_history(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate IL usage per team-season.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, season, total_il_days, il_stints, unique_players.
    """
    season_filter = ""
    params: dict = {}
    if seasons:
        season_filter = "AND pst.season = ANY(:seasons)"
        params["seasons"] = seasons

    query = f"""
    SELECT
        pst.team_id,
        pst.season,
        SUM(pst.days_on_status)           AS total_il_days,
        COUNT(*)                           AS il_stints,
        COUNT(DISTINCT pst.player_id)      AS unique_players
    FROM production.fact_player_status_timeline pst
    WHERE pst.status_type LIKE '%%IL%%'
      {season_filter}
    GROUP BY pst.team_id, pst.season
    ORDER BY pst.season, pst.team_id
    """
    logger.info("Fetching team IL history")
    df = read_sql(query, params)
    logger.info("Team IL history: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 6. Team age profile
# ---------------------------------------------------------------------------
def get_team_age_profile(season: int) -> pd.DataFrame:
    """Age distribution for each team's active roster.

    Parameters
    ----------
    season : int
        Season to analyze (uses July 1 of that year as reference).

    Returns
    -------
    pd.DataFrame
        Columns: team_id, avg_age, pct_under_27, pct_27_31, pct_over_31.
    """
    query = """
    WITH active AS (
        SELECT DISTINCT team_id, player_id
        FROM production.fact_player_game_mlb
        WHERE season = :season
    ),
    ages AS (
        SELECT
            a.team_id,
            EXTRACT(YEAR FROM AGE(
                MAKE_DATE(:season, 7, 1), dp.birth_date
            )) AS age
        FROM active a
        JOIN production.dim_player dp ON a.player_id = dp.player_id
        WHERE dp.birth_date IS NOT NULL
    )
    SELECT
        team_id,
        AVG(age)::float AS avg_age,
        AVG(CASE WHEN age < 27 THEN 1.0 ELSE 0.0 END)::float AS pct_under_27,
        AVG(CASE WHEN age >= 27 AND age <= 31 THEN 1.0 ELSE 0.0 END)::float AS pct_27_31,
        AVG(CASE WHEN age > 31 THEN 1.0 ELSE 0.0 END)::float AS pct_over_31
    FROM ages
    GROUP BY team_id
    """
    logger.info("Fetching team age profile for %d", season)
    df = read_sql(query, {"season": season})
    logger.info("Team age profile: %d teams", len(df))
    return df


# ---------------------------------------------------------------------------
# 7. Team info (abbreviation, division, league)
# ---------------------------------------------------------------------------
def get_team_info() -> pd.DataFrame:
    """Lookup table: team_id -> abbreviation, team_name, division, league.

    Returns
    -------
    pd.DataFrame
    """
    query = """
    SELECT team_id, abbreviation, team_name, full_name, division, league
    FROM production.dim_team
    """
    return read_sql(query)


# ---------------------------------------------------------------------------
# 8. Team season pitching aggregates
# ---------------------------------------------------------------------------
def get_team_season_pitching(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate pitching stats per team-season from fact_game_totals.

    Uses the *opponent* line to derive runs allowed.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, season, runs_allowed, hits_allowed,
        hr_allowed, walks_allowed, strikeouts_by, games.
    """
    season_filter = ""
    params: dict = {}
    if seasons:
        season_filter = "AND dg.season = ANY(:seasons)"
        params["seasons"] = seasons

    # For each team-game, the *other* team's runs = this team's runs allowed
    query = f"""
    WITH game_pairs AS (
        SELECT
            us.team_id,
            dg.season,
            them.runs       AS runs_allowed,
            them.hits       AS hits_allowed,
            them.home_runs  AS hr_allowed,
            them.walks      AS walks_allowed,
            us.strikeouts   AS strikeouts_by
        FROM production.fact_game_totals us
        JOIN production.fact_game_totals them
            ON us.game_pk = them.game_pk
            AND us.home_away != them.home_away
        JOIN production.dim_game dg ON us.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          {season_filter}
    )
    SELECT
        team_id,
        season,
        SUM(runs_allowed)    AS runs_allowed,
        SUM(hits_allowed)    AS hits_allowed,
        SUM(hr_allowed)      AS hr_allowed,
        SUM(walks_allowed)   AS walks_allowed,
        SUM(strikeouts_by)   AS strikeouts_by,
        COUNT(*)             AS games
    FROM game_pairs
    GROUP BY team_id, season
    ORDER BY season, team_id
    """
    logger.info("Fetching team season pitching")
    df = read_sql(query, params)
    logger.info("Team season pitching: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 9. Team park factors (run + HR, mapped from venue to team)
# ---------------------------------------------------------------------------
def get_team_park_factors(
    seasons: list[int] | None = None,
    min_home_games: int = 100,
) -> pd.DataFrame:
    """Park factors per team, combining venue run factors and HR park factors.

    Maps each team to its primary home venue (most home games), then joins:
    - Run factor from home/away RPG ratio (regressed toward 1.0)
    - HR park factor from dim_park_factor (3yr smoothed, averaged across
      batter stands)

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to compute over.  Defaults to last 3 full seasons for
        stability.
    min_home_games : int
        Minimum home games at a venue to include in run factor calc.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, venue_id, run_pf, hr_pf.
        Both factors are centered on 1.0 (>1 = hitter-friendly).
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    # --- 1. Team -> primary home venue ---
    team_venue_q = """
    SELECT fgt.team_id, dg.venue_id, COUNT(*) AS home_games
    FROM production.fact_game_totals fgt
    JOIN production.dim_game dg ON fgt.game_pk = dg.game_pk
    WHERE fgt.home_away = 'home'
      AND dg.game_type = 'R'
      AND dg.season = ANY(:seasons)
    GROUP BY fgt.team_id, dg.venue_id
    """
    tv = read_sql(team_venue_q, {"seasons": seasons})
    # Keep the venue with the most home games per team
    tv["home_games"] = tv["home_games"].astype(int)
    tv = tv.sort_values("home_games", ascending=False).drop_duplicates("team_id")
    tv = tv[["team_id", "venue_id", "home_games"]].copy()

    # --- 2. Venue run factors (home/away RPG ratio, regressed) ---
    run_factors = get_venue_run_factors(seasons=seasons, min_games=min_home_games)

    # --- 3. HR park factors from dim_park_factor (avg across batter stands) ---
    max_season = max(seasons)
    hr_q = """
    SELECT
        venue_id,
        AVG(COALESCE(hr_pf_3yr, hr_pf_season, 1.0)) AS hr_pf
    FROM production.dim_park_factor
    WHERE season = :season
    GROUP BY venue_id
    """
    hr_pf = read_sql(hr_q, {"season": max_season})

    # --- 4. Join everything on venue_id ---
    result = tv.merge(run_factors[["venue_id", "run_factor"]], on="venue_id", how="left")
    result = result.merge(hr_pf[["venue_id", "hr_pf"]], on="venue_id", how="left")

    # Fill missing with neutral (1.0)
    result["run_pf"] = result["run_factor"].fillna(1.0)
    result["hr_pf"] = result["hr_pf"].fillna(1.0)
    result = result[["team_id", "venue_id", "run_pf", "hr_pf"]].copy()

    logger.info("Team park factors: %d teams (run_pf range %.3f–%.3f)",
                len(result), result["run_pf"].min(), result["run_pf"].max())
    return result


# ---------------------------------------------------------------------------
# 10. Game results for series detection
# ---------------------------------------------------------------------------
def get_series_results(include_postseason: bool = True) -> pd.DataFrame:
    """Get game results for series detection.

    Thin wrapper around ``get_game_results()`` — the series detection
    logic lives in ``src.models.series_elo.detect_series()``.

    Parameters
    ----------
    include_postseason : bool
        If True, include postseason games so postseason series can
        be detected and rated.

    Returns
    -------
    pd.DataFrame
        Same schema as ``get_game_results()``.
    """
    return get_game_results(include_postseason=include_postseason)
