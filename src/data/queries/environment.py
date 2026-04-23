"""Park factors, umpire tendencies, weather effects, catcher framing, days rest,
rolling league baseline."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import logit as _logit

from src.data.db import read_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Park factors (HR by batter handedness)
# ---------------------------------------------------------------------------
def get_park_factors(season: int) -> pd.DataFrame:
    """HR park factors by venue and batter handedness.

    Uses 3-year smoothed park factor (more stable than single-season).
    Falls back to single-season if 3yr is missing.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: venue_id, venue_name, batter_stand, hr_pf.
    """
    query = """
    SELECT
        venue_id,
        venue_name,
        batter_stand,
        COALESCE(hr_pf_3yr, hr_pf_season, 1.0) AS hr_pf
    FROM production.dim_park_factor
    WHERE season = :season
    ORDER BY venue_id, batter_stand
    """
    logger.info("Fetching park factors for %d", season)
    return read_sql(query, {"season": season})


def get_hitter_team_venue(season: int) -> pd.DataFrame:
    """Map each hitter to their primary team and home venue for a season.

    Uses batting boxscores to find the team where each hitter played the
    most games, then maps that team to its home venue via dim_game.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, team_id, team_name, venue_id, bat_side.
    """
    query = """
    WITH batter_teams AS (
        SELECT
            bb.batter_id,
            bb.team_id,
            bb.team_name,
            COUNT(*) AS games,
            ROW_NUMBER() OVER (
                PARTITION BY bb.batter_id ORDER BY COUNT(*) DESC
            ) AS rn
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
        GROUP BY bb.batter_id, bb.team_id, bb.team_name
    ),
    team_venue AS (
        SELECT
            home_team_id AS team_id,
            venue_id,
            ROW_NUMBER() OVER (
                PARTITION BY home_team_id ORDER BY COUNT(*) DESC
            ) AS rn
        FROM production.dim_game
        WHERE season = :season
          AND game_type = 'R'
        GROUP BY home_team_id, venue_id
    )
    SELECT
        bt.batter_id,
        bt.team_id,
        bt.team_name,
        tv.venue_id,
        COALESCE(dp.bat_side, 'R') AS bat_side
    FROM batter_teams bt
    JOIN team_venue tv ON bt.team_id = tv.team_id AND tv.rn = 1
    LEFT JOIN production.dim_player dp ON bt.batter_id = dp.player_id
    WHERE bt.rn = 1
    ORDER BY bt.batter_id
    """
    logger.info("Fetching hitter team-venue mapping for %d", season)
    return read_sql(query, {"season": season})


def get_pitcher_team_venue(season: int) -> pd.DataFrame:
    """Map each pitcher to their primary team and home venue for a season.

    Uses pitching boxscores to find the team where each pitcher played the
    most games, then maps that team to its home venue via dim_game.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, team_id, venue_id.
    """
    query = """
    WITH pitcher_teams AS (
        SELECT
            pb.pitcher_id,
            pb.team_id,
            COUNT(*) AS games,
            ROW_NUMBER() OVER (
                PARTITION BY pb.pitcher_id ORDER BY COUNT(*) DESC
            ) AS rn
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.season = :season
          AND dg.game_type = 'R'
        GROUP BY pb.pitcher_id, pb.team_id
    ),
    team_venue AS (
        SELECT
            home_team_id AS team_id,
            venue_id,
            ROW_NUMBER() OVER (
                PARTITION BY home_team_id ORDER BY COUNT(*) DESC
            ) AS rn
        FROM production.dim_game
        WHERE season = :season
          AND game_type = 'R'
        GROUP BY home_team_id, venue_id
    )
    SELECT
        pt.pitcher_id,
        pt.team_id,
        tv.venue_id
    FROM pitcher_teams pt
    JOIN team_venue tv ON pt.team_id = tv.team_id AND tv.rn = 1
    WHERE pt.rn = 1
    ORDER BY pt.pitcher_id
    """
    logger.info("Fetching pitcher team-venue mapping for %d", season)
    return read_sql(query, {"season": season})


# ---------------------------------------------------------------------------
# Umpire tendencies (K, BB, HR rate)
# ---------------------------------------------------------------------------
def get_umpire_tendencies(
    seasons: list[int] | None = None,
    min_games: int = 30,
) -> pd.DataFrame:
    """Compute per-umpire K, BB, and HR rate tendencies with shrinkage.

    Uses multi-season PA-level data joined to dim_umpire to compute each
    HP umpire's K-rate, BB-rate, and HR-rate, then shrinks toward the
    league mean based on games umpired (more games = more trust).

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to 2021-2025 (5 recent seasons).
    min_games : int
        Minimum games umpired to be included.

    Returns
    -------
    pd.DataFrame
        Columns: hp_umpire_name, games, total_pa, k_rate, bb_rate, hr_rate,
        league_k_rate, league_bb_rate, league_hr_rate,
        k_rate_shrunk, bb_rate_shrunk, hr_rate_shrunk,
        k_logit_lift, bb_logit_lift, hr_logit_lift.
    """
    if seasons is None:
        seasons = list(range(2021, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH ump_stats AS (
        SELECT
            du.hp_umpire_name,
            COUNT(DISTINCT du.game_pk) AS games,
            COUNT(fpa.pa_id) AS total_pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END) AS total_k,
            SUM(CASE WHEN fpa.events = 'walk'
                     THEN 1 ELSE 0 END) AS total_bb,
            SUM(CASE WHEN fpa.events = 'home_run'
                     THEN 1 ELSE 0 END) AS total_hr
        FROM production.dim_umpire du
        JOIN production.dim_game dg ON du.game_pk = dg.game_pk
        JOIN production.fact_pa fpa ON du.game_pk = fpa.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fpa.events IS NOT NULL
        GROUP BY du.hp_umpire_name
        HAVING COUNT(DISTINCT du.game_pk) >= {int(min_games)}
    )
    SELECT
        hp_umpire_name,
        games,
        total_pa,
        total_k,
        total_bb,
        total_hr,
        ROUND((total_k::numeric / NULLIF(total_pa, 0)), 5) AS k_rate,
        ROUND((total_bb::numeric / NULLIF(total_pa, 0)), 5) AS bb_rate,
        ROUND((total_hr::numeric / NULLIF(total_pa, 0)), 5) AS hr_rate
    FROM ump_stats
    ORDER BY games DESC
    """
    logger.info("Fetching umpire tendencies for seasons %s (min_games=%d)",
                seasons, min_games)
    df = read_sql(query, {})

    if df.empty:
        return df

    from scipy.special import logit as _logit
    _clip = lambda x: np.clip(x, 1e-6, 1 - 1e-6)

    # Shrinkage: trust umpires with more games
    # k = 80 games (~1 full season) as the shrinkage constant
    shrinkage_k = 80.0
    df["reliability"] = df["games"] / (df["games"] + shrinkage_k)

    # --- K-rate ---
    league_k_rate = float(df["total_k"].sum() / df["total_pa"].sum())
    df["league_k_rate"] = league_k_rate
    df["k_rate_shrunk"] = (
        df["reliability"] * df["k_rate"]
        + (1 - df["reliability"]) * league_k_rate
    )
    df["k_logit_lift"] = (
        _logit(_clip(df["k_rate_shrunk"].values))
        - _logit(_clip(league_k_rate))
    ).round(4)

    # --- BB-rate ---
    league_bb_rate = float(df["total_bb"].sum() / df["total_pa"].sum())
    df["league_bb_rate"] = league_bb_rate
    df["bb_rate_shrunk"] = (
        df["reliability"] * df["bb_rate"]
        + (1 - df["reliability"]) * league_bb_rate
    )
    df["bb_logit_lift"] = (
        _logit(_clip(df["bb_rate_shrunk"].values))
        - _logit(_clip(league_bb_rate))
    ).round(4)

    # --- HR-rate ---
    league_hr_rate = float(df["total_hr"].sum() / df["total_pa"].sum())
    df["league_hr_rate"] = league_hr_rate
    df["hr_rate_shrunk"] = (
        df["reliability"] * df["hr_rate"]
        + (1 - df["reliability"]) * league_hr_rate
    )
    df["hr_logit_lift"] = (
        _logit(_clip(df["hr_rate_shrunk"].values))
        - _logit(_clip(league_hr_rate))
    ).round(4)

    df = df.drop(columns=["reliability"])
    logger.info(
        "Umpire tendencies: %d umpires, league K%%=%.4f BB%%=%.4f HR%%=%.5f, "
        "K lift range=[%.4f, %.4f], BB lift range=[%.4f, %.4f], "
        "HR lift range=[%.4f, %.4f]",
        len(df), league_k_rate, league_bb_rate, league_hr_rate,
        df["k_logit_lift"].min(), df["k_logit_lift"].max(),
        df["bb_logit_lift"].min(), df["bb_logit_lift"].max(),
        df["hr_logit_lift"].min(), df["hr_logit_lift"].max(),
    )
    return df


def get_umpire_k_tendencies(
    seasons: list[int] | None = None,
    min_games: int = 30,
) -> pd.DataFrame:
    """Backward-compatible alias for ``get_umpire_tendencies()``.

    Returns the same DataFrame (which now includes bb/hr columns too).
    """
    return get_umpire_tendencies(seasons=seasons, min_games=min_games)


# ---------------------------------------------------------------------------
# Weather effects on K and HR rates
# ---------------------------------------------------------------------------
def get_weather_effects(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute weather-adjusted K and HR rate multipliers.

    Groups outdoor games by temperature bucket, wind category, and wind
    speed bucket, computes K-rate and HR-rate for each combination, and
    expresses as a multiplier relative to the overall outdoor average.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to 2018-2025.

    Returns
    -------
    pd.DataFrame
        Columns: temp_bucket, wind_category, wind_speed_bucket, games,
        k_rate, hr_rate, k_multiplier, hr_multiplier.
    """
    if seasons is None:
        seasons = list(range(2018, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH game_stats AS (
        SELECT
            fpa.game_pk,
            COUNT(*) AS pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END) AS k,
            SUM(CASE WHEN fpa.events = 'home_run' THEN 1 ELSE 0 END) AS hr
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fpa.events IS NOT NULL
        GROUP BY fpa.game_pk
    )
    SELECT
        CASE
            WHEN dw.temperature < 55 THEN 'cold'
            WHEN dw.temperature BETWEEN 55 AND 69 THEN 'cool'
            WHEN dw.temperature BETWEEN 70 AND 84 THEN 'warm'
            ELSE 'hot'
        END AS temp_bucket,
        dw.wind_category,
        CASE
            WHEN dw.wind_speed <= 5 THEN 'calm'
            WHEN dw.wind_speed <= 10 THEN 'light'
            WHEN dw.wind_speed <= 15 THEN 'moderate'
            ELSE 'strong'
        END AS wind_speed_bucket,
        COUNT(*) AS games,
        SUM(gs.pa) AS total_pa,
        SUM(gs.k) AS total_k,
        SUM(gs.hr) AS total_hr
    FROM game_stats gs
    JOIN production.dim_weather dw ON gs.game_pk = dw.game_pk
    WHERE NOT dw.is_dome
    GROUP BY 1, dw.wind_category, 3
    HAVING COUNT(*) >= 30
    ORDER BY 1, dw.wind_category, 3
    """
    logger.info("Fetching weather effects for seasons %s", seasons)
    df = read_sql(query, {})

    if df.empty:
        return df

    df["k_rate"] = (df["total_k"] / df["total_pa"]).round(5)
    df["hr_rate"] = (df["total_hr"] / df["total_pa"]).round(5)

    # Overall outdoor averages
    overall_k = float(df["total_k"].sum() / df["total_pa"].sum())
    overall_hr = float(df["total_hr"].sum() / df["total_pa"].sum())

    df["k_multiplier"] = (df["k_rate"] / overall_k).round(4)
    df["hr_multiplier"] = (df["hr_rate"] / overall_hr).round(4)
    df["overall_k_rate"] = overall_k
    df["overall_hr_rate"] = overall_hr

    df = df.drop(columns=["total_pa", "total_k", "total_hr"])

    logger.info("Weather effects: %d combinations, K mult range=[%.3f, %.3f], HR mult range=[%.3f, %.3f]",
                len(df),
                df["k_multiplier"].min(), df["k_multiplier"].max(),
                df["hr_multiplier"].min(), df["hr_multiplier"].max())
    return df


# ---------------------------------------------------------------------------
# Rolling league baseline (14-day trailing K%, BB%, HR%)
# ---------------------------------------------------------------------------
def get_rolling_league_baseline(
    season: int | None = None,
    window: int = 14,
    game_date: str | None = None,
) -> pd.DataFrame:
    """Compute trailing league K%, BB%, HR% and logit offsets.

    Uses the last ``window`` game-dates (not strict calendar days --
    off-days are skipped) to compute league rates, then converts to
    logit-scale offsets vs the static baselines in constants.py.

    Parameters
    ----------
    season : int, optional
        Season to query. Derived from ``game_date`` if not provided.
    window : int
        Number of trailing game-dates to include (default 14).
    game_date : str, optional
        Reference date (ISO format). Used to derive season if needed.

    Returns
    -------
    pd.DataFrame
        Columns: game_date, rolling_pa, rolling_k_rate, rolling_bb_rate,
        rolling_hr_rate, k_offset, bb_offset, hr_offset.
    """
    if season is None:
        if game_date is not None:
            season = int(game_date[:4])
        else:
            from datetime import date as _date
            season = _date.today().year
    from src.utils.constants import (
        SIM_LEAGUE_K_RATE,
        SIM_LEAGUE_BB_RATE,
        SIM_LEAGUE_HR_RATE,
    )

    query = f"""
    WITH daily AS (
        SELECT
            dg.game_date,
            COUNT(*) AS pa,
            SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play')
                     THEN 1 ELSE 0 END) AS k,
            SUM(CASE WHEN fpa.events IN ('walk','intent_walk')
                     THEN 1 ELSE 0 END) AS bb,
            SUM(CASE WHEN fpa.events = 'home_run'
                     THEN 1 ELSE 0 END) AS hr
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season = {int(season)}
          AND fpa.events IS NOT NULL
        GROUP BY dg.game_date
    )
    SELECT
        game_date,
        pa,
        SUM(pa) OVER w AS rolling_pa,
        SUM(k) OVER w AS rolling_k,
        SUM(bb) OVER w AS rolling_bb,
        SUM(hr) OVER w AS rolling_hr
    FROM daily
    WINDOW w AS (ORDER BY game_date
                 ROWS BETWEEN {int(window) - 1} PRECEDING AND CURRENT ROW)
    ORDER BY game_date
    """
    logger.info("Fetching rolling league baseline for %d (window=%d)", season, window)
    df = read_sql(query, {})
    if df.empty:
        return df

    df["rolling_k_rate"] = (df["rolling_k"] / df["rolling_pa"]).round(5)
    df["rolling_bb_rate"] = (df["rolling_bb"] / df["rolling_pa"]).round(5)
    df["rolling_hr_rate"] = (df["rolling_hr"] / df["rolling_pa"]).round(5)

    base_k = _logit(SIM_LEAGUE_K_RATE)
    base_bb = _logit(SIM_LEAGUE_BB_RATE)
    base_hr = _logit(SIM_LEAGUE_HR_RATE)

    df["k_offset"] = (_logit(np.clip(df["rolling_k_rate"], 0.01, 0.99)) - base_k).round(4)
    df["bb_offset"] = (_logit(np.clip(df["rolling_bb_rate"], 0.01, 0.99)) - base_bb).round(4)
    df["hr_offset"] = (_logit(np.clip(df["rolling_hr_rate"], 0.01, 0.99)) - base_hr).round(4)

    df = df.drop(columns=["pa", "rolling_k", "rolling_bb", "rolling_hr"])
    df["game_date"] = df["game_date"].astype(str)

    logger.info(
        "Rolling baseline: %d days, latest K_off=%.3f BB_off=%.3f HR_off=%.3f",
        len(df),
        df["k_offset"].iloc[-1], df["bb_offset"].iloc[-1], df["hr_offset"].iloc[-1],
    )
    return df


# ---------------------------------------------------------------------------
# Days rest for starting pitchers
# ---------------------------------------------------------------------------
def get_days_rest(seasons: list[int] | None = None) -> pd.DataFrame:
    """Compute days since previous start for each pitcher start.

    Uses ``fact_player_game_mlb`` with ``pit_is_starter = true`` and
    ``pit_bf >= 9`` to filter out openers / bullpen games.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include.  Defaults to 2018-2025.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, game_pk, game_date, season, days_rest,
        rest_bucket, pit_k, pit_bb, pit_bf, pit_ip.
        ``rest_bucket`` is one of ``'short'``, ``'normal'``, ``'extended'``.
    """
    if seasons is None:
        seasons = list(range(2018, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH starter_games AS (
        SELECT
            fpg.player_id  AS pitcher_id,
            fpg.game_pk,
            fpg.game_date,
            fpg.season,
            fpg.pit_k,
            fpg.pit_bb,
            fpg.pit_bf,
            fpg.pit_ip,
            LAG(fpg.game_date) OVER (
                PARTITION BY fpg.player_id ORDER BY fpg.game_date
            ) AS prev_start_date
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true
          AND fpg.pit_bf >= 9
          AND fpg.pit_ip > 0
          AND dg.game_type = 'R'
          AND dg.season IN ({season_list})
    )
    SELECT
        pitcher_id,
        game_pk,
        game_date,
        season,
        (game_date - prev_start_date) AS days_rest,
        CASE
            WHEN (game_date - prev_start_date) <= 4 THEN 'short'
            WHEN (game_date - prev_start_date) = 5  THEN 'normal'
            ELSE 'extended'
        END AS rest_bucket,
        pit_k,
        pit_bb,
        pit_bf,
        pit_ip
    FROM starter_games
    WHERE prev_start_date IS NOT NULL
      AND (game_date - prev_start_date) BETWEEN 3 AND 14
    ORDER BY pitcher_id, game_date
    """
    logger.info("Fetching days rest for seasons %s", seasons)
    return read_sql(query)


# ---------------------------------------------------------------------------
# Catcher framing effects
# ---------------------------------------------------------------------------
def get_team_defense_lifts(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate Statcast Outs Above Average to per-team-season defense
    lifts that the simulator can apply as a BABIP shift.

    Sums ``fielding_runs_prevented`` across all players on each team in
    the season, divides by 162 games to get a runs/game rate, then
    converts that to a probability shift on BIP hit rate using:

        babip_adj = -frp_per_game / (run_value_of_hit * BIP_per_game)

    With league-average run value of an avg hit ~0.55 and BIP/team-game
    ~25, the conversion factor is ~-0.0727 per R/G of frp. Positive frp
    means good defense, so the BABIP adj is negative (fewer hits).

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to 2018-2026.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, abbreviation, season, frp, frp_per_game,
        defense_babip_adj.
    """
    if seasons is None:
        seasons = list(range(2018, 2027))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
        SELECT t.team_id,
               t.abbreviation,
               agg.season,
               agg.frp
        FROM (
            SELECT f.team_name,
                   f.season,
                   SUM(f.fielding_runs_prevented) AS frp
            FROM production.fact_fielding_oaa f
            WHERE f.season IN ({season_list})
              AND f.team_name <> '---'
            GROUP BY f.team_name, f.season
        ) agg
        JOIN production.dim_team t ON t.team_name = agg.team_name
    """
    logger.info("Fetching team defense lifts for seasons %s", seasons)
    df = read_sql(query, {})
    if df.empty:
        return df

    # Run-value-per-hit weighted across single/double/triple ~0.55
    # BIP per team-game (after K/BB/HBP/HR removed) ~25
    _CONV = -1.0 / (0.55 * 25.0)  # ≈ -0.0727 per R/G frp

    df["frp_per_game"] = df["frp"] / 162.0
    df["defense_babip_adj"] = (df["frp_per_game"] * _CONV).round(5)
    df = df.sort_values(["season", "frp"], ascending=[True, False]).reset_index(drop=True)
    logger.info(
        "Team defense lifts: %d team-seasons, babip_adj range=[%.4f, %.4f]",
        len(df), df["defense_babip_adj"].min(), df["defense_babip_adj"].max(),
    )
    return df


def get_catcher_framing_effects(
    seasons: list[int] | None = None,
    shrinkage_k: int = 500,
) -> pd.DataFrame:
    """Compute per-catcher called-strike-rate lift on taken pitches.

    For each catcher-season, computes the called-strike rate on non-swing
    pitches, applies empirical Bayes shrinkage, and returns a logit-scale
    lift relative to the season league average.

    Join path: fact_lineup (position='C', starter) -> fact_player_game_mlb
    (pitcher team_id) -> fact_pitch (is_swing=false).

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to 2018-2025.
    shrinkage_k : int
        Shrinkage constant in taken-pitch units.  ``shrunk_rate =
        (n * observed + k * league_avg) / (n + k)``.  Default 500
        provides heavy shrinkage for low-sample catchers.

    Returns
    -------
    pd.DataFrame
        Columns: catcher_id, season, pitches_received, called_strikes,
        called_strike_rate_raw, league_cs_rate, called_strike_rate_shrunk,
        logit_lift.
    """
    if seasons is None:
        seasons = list(range(2018, 2026))

    season_list = ",".join(str(int(s)) for s in seasons)
    query = f"""
    WITH catcher_game AS (
        SELECT fl.game_pk, fl.player_id AS catcher_id,
               fl.team_id, fl.season
        FROM production.fact_lineup fl
        WHERE fl.position = 'C'
          AND fl.is_starter = true
          AND fl.season IN ({season_list})
    ),
    catcher_pitch AS (
        SELECT
            cg.catcher_id,
            cg.season,
            COUNT(*)                       AS pitches_received,
            SUM(fp.is_called_strike::int)  AS called_strikes
        FROM catcher_game cg
        JOIN production.fact_player_game_mlb pg
            ON pg.game_pk = cg.game_pk
           AND pg.team_id = cg.team_id
        JOIN production.fact_pitch fp
            ON fp.game_pk = cg.game_pk
           AND fp.pitcher_id = pg.player_id
        JOIN production.dim_game dg
            ON fp.game_pk = dg.game_pk
        WHERE fp.is_swing = false
          AND dg.game_type = 'R'
          AND pg.player_role = 'pitcher'
        GROUP BY cg.catcher_id, cg.season
    )
    SELECT
        catcher_id,
        season,
        pitches_received,
        called_strikes,
        ROUND(called_strikes::numeric
              / NULLIF(pitches_received, 0), 5) AS called_strike_rate_raw
    FROM catcher_pitch
    ORDER BY season, pitches_received DESC
    """
    logger.info(
        "Fetching catcher framing effects for seasons %s (shrinkage_k=%d)",
        seasons, shrinkage_k,
    )
    df = read_sql(query, {})

    if df.empty:
        return df

    from scipy.special import logit as _logit

    _clip = lambda x: np.clip(x, 1e-6, 1 - 1e-6)  # noqa: E731

    # League-average called-strike rate per season (weighted by pitches)
    season_league = (
        df.groupby("season")
        .apply(
            lambda g: g["called_strikes"].sum() / g["pitches_received"].sum(),
            include_groups=False,
        )
        .rename("league_cs_rate")
    )
    df = df.merge(season_league, on="season", how="left")

    # Shrinkage toward season league average
    n = df["pitches_received"].values.astype(float)
    obs = df["called_strike_rate_raw"].values.astype(float)
    lg = df["league_cs_rate"].values.astype(float)
    df["called_strike_rate_shrunk"] = (
        (n * obs + shrinkage_k * lg) / (n + shrinkage_k)
    ).round(5)

    # Logit lift relative to league average
    df["logit_lift"] = (
        _logit(_clip(df["called_strike_rate_shrunk"].values))
        - _logit(_clip(lg))
    ).round(5)

    logger.info(
        "Catcher framing: %d catcher-seasons, lift range=[%.4f, %.4f]",
        len(df), df["logit_lift"].min(), df["logit_lift"].max(),
    )
    return df
