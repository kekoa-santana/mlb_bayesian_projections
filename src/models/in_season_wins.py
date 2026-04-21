"""In-season blending of preseason win projections with actual results.

The preseason league simulation produces a strong prior on each team's
true-talent win percentage.  As games are played, that prior should be
updated with actual evidence using a Beta-Binomial conjugate update.

We use Pythagorean win% from actual runs scored/allowed alongside raw
W-L, because run differential is a cleaner talent signal at small
sample sizes.  A ``prior_strength_games`` parameter controls how fast
the posterior reacts to new data: at the default of 55, 12 actual
games get roughly 18% weight.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

PYTHAG_EXPONENT = 1.83
DEFAULT_PRIOR_STRENGTH = 55.0
DEFAULT_RECORD_WEIGHT = 0.40
FULL_SEASON_GAMES = 162
MIN_TALENT_PCT = 0.25
MAX_TALENT_PCT = 0.75


def pythag_win_pct(
    rs: float, ra: float, exponent: float = PYTHAG_EXPONENT,
) -> float:
    """Pythagorean win percentage from runs scored and allowed."""
    rs_e = max(float(rs), 0.01) ** exponent
    ra_e = max(float(ra), 0.01) ** exponent
    return rs_e / (rs_e + ra_e)


def blend_projected_wins(
    preseason_wins: float,
    games_played: int,
    current_wins: int,
    runs_scored: float,
    runs_allowed: float,
    prior_strength_games: float = DEFAULT_PRIOR_STRENGTH,
    record_weight: float = DEFAULT_RECORD_WEIGHT,
    xrs_per_game: float | None = None,
    xra_per_game: float | None = None,
) -> dict[str, float | None]:
    """Blend preseason projection with current-season record and run diff.

    Treats the preseason projection as a Beta prior with effective sample
    size ``prior_strength_games``.  The posterior talent estimate is
    projected over the remaining schedule and added to games already won.

    Parameters
    ----------
    preseason_wins : float
        Preseason projected wins for the full 162-game season.
    games_played : int
        Games already completed in the current season.
    current_wins : int
        Wins in completed games.
    runs_scored, runs_allowed : float
        Cumulative runs for and against.
    prior_strength_games : float, default 55
        Effective sample size of the preseason prior.
    record_weight : float, default 0.40
        Blend weight on raw W-L evidence vs Pythagorean evidence.  The
        default leans on Pythag because it is less noisy at small n.
    xrs_per_game, xra_per_game : float, optional
        Statcast-based expected runs scored/allowed per game.  When
        provided, used for the Pythagorean component instead of actual
        runs (strips BABIP and sequencing noise).

    Returns
    -------
    dict
        expected_final_wins, posterior_win_pct, pythag_win_pct,
        observed_win_pct, preseason_win_pct.
    """
    preseason_pct = float(
        np.clip(preseason_wins / FULL_SEASON_GAMES, MIN_TALENT_PCT, MAX_TALENT_PCT)
    )

    if games_played <= 0:
        return {
            "expected_final_wins": float(preseason_wins),
            "posterior_win_pct": preseason_pct,
            "pythag_win_pct": None,
            "observed_win_pct": None,
            "preseason_win_pct": preseason_pct,
        }

    record_pct = current_wins / games_played

    # Prefer xRuns (Statcast contact quality) for Pythagorean when available
    if xrs_per_game is not None and xra_per_game is not None:
        pyt_pct = pythag_win_pct(xrs_per_game, xra_per_game)
    else:
        pyt_pct = pythag_win_pct(runs_scored, runs_allowed)

    observed_pct = record_weight * record_pct + (1.0 - record_weight) * pyt_pct

    posterior_pct = (
        prior_strength_games * preseason_pct + games_played * observed_pct
    ) / (prior_strength_games + games_played)
    posterior_pct = float(np.clip(posterior_pct, MIN_TALENT_PCT, MAX_TALENT_PCT))

    remaining = max(0, FULL_SEASON_GAMES - games_played)
    expected_final = current_wins + remaining * posterior_pct

    return {
        "expected_final_wins": float(expected_final),
        "posterior_win_pct": posterior_pct,
        "pythag_win_pct": float(pyt_pct),
        "observed_win_pct": float(observed_pct),
        "preseason_win_pct": preseason_pct,
    }


def apply_in_season_blend(
    preseason_wins: pd.DataFrame,
    team_records: pd.DataFrame | None,
    wins_col: str = "sim_wins_mean",
    prior_strength_games: float = DEFAULT_PRIOR_STRENGTH,
    record_weight: float = DEFAULT_RECORD_WEIGHT,
) -> pd.DataFrame:
    """Attach blended wins columns to a preseason projection DataFrame.

    Parameters
    ----------
    preseason_wins : pd.DataFrame
        Must contain ``team_id`` and ``wins_col`` (default
        ``sim_wins_mean``).  Typically the output of the league sim.
    team_records : pd.DataFrame or None
        Must contain ``team_id``, ``games_played``, ``wins``, ``losses``,
        ``runs_scored``, ``runs_allowed``.  If None or empty, returns a
        copy of ``preseason_wins`` with ``blended_wins`` copied from the
        preseason column and other blend columns filled with None.
    wins_col : str
        Column in ``preseason_wins`` holding preseason projection wins.

    Returns
    -------
    pd.DataFrame
        Copy of ``preseason_wins`` plus columns ``blended_wins``,
        ``posterior_win_pct``, ``pythag_win_pct``, ``observed_win_pct``,
        ``preseason_win_pct``, and ``games_played``.
    """
    out = preseason_wins.copy()

    if team_records is None or team_records.empty:
        out["blended_wins"] = out[wins_col]
        out["posterior_win_pct"] = out[wins_col] / FULL_SEASON_GAMES
        out["pythag_win_pct"] = np.nan
        out["observed_win_pct"] = np.nan
        out["preseason_win_pct"] = out[wins_col] / FULL_SEASON_GAMES
        out["games_played"] = 0
        return out

    rec = team_records.set_index("team_id")

    results: list[dict[str, float | None]] = []
    for _, row in out.iterrows():
        tid = int(row["team_id"])
        preseason_w = float(row[wins_col])
        if tid in rec.index:
            rr = rec.loc[tid]
            gp = int(rr["games_played"])
            w = int(rr["wins"])
            rs = float(rr["runs_scored"])
            ra = float(rr["runs_allowed"])
            xrs = float(rr["xrs_per_game"]) if "xrs_per_game" in rec.columns and pd.notna(rr.get("xrs_per_game")) else None
            xra = float(rr["xra_per_game"]) if "xra_per_game" in rec.columns and pd.notna(rr.get("xra_per_game")) else None
        else:
            gp, w, rs, ra = 0, 0, 0.0, 0.0
            xrs, xra = None, None

        blend = blend_projected_wins(
            preseason_wins=preseason_w,
            games_played=gp,
            current_wins=w,
            runs_scored=rs,
            runs_allowed=ra,
            prior_strength_games=prior_strength_games,
            record_weight=record_weight,
            xrs_per_game=xrs,
            xra_per_game=xra,
        )
        results.append({"games_played": gp, **blend})

    blend_df = pd.DataFrame(results)
    out["blended_wins"] = blend_df["expected_final_wins"].values
    out["posterior_win_pct"] = blend_df["posterior_win_pct"].values
    out["pythag_win_pct"] = blend_df["pythag_win_pct"].values
    out["observed_win_pct"] = blend_df["observed_win_pct"].values
    out["preseason_win_pct"] = blend_df["preseason_win_pct"].values
    out["games_played"] = blend_df["games_played"].values

    return out


def _baseruns(
    pa: float, k: float, bb: float, hbp: float, hr: float,
    xba_on_bip: float, xslg_on_bip: float, games: int,
) -> float:
    """Hybrid BaseRuns: K/BB/HR at face value, xBA/xSLG on BIP.

    Returns expected runs per game.
    """
    # BIP = all batted balls including HR (Statcast xBA/xSLG covers HR events)
    bip = pa - k - bb - hbp
    if bip <= 0 or games <= 0:
        return 0.0

    # xBA and xSLG from Statcast already include HR outcomes on batted balls,
    # so xH and xTB capture HR contribution without separate addition.
    xH = xba_on_bip * bip
    xTB = xslg_on_bip * bip

    A = xH + bb + hbp - hr
    B = (1.4 * xTB - 0.6 * xH - 3 * hr + 0.1 * (bb + hbp)) * 1.02
    sf_est = pa * 0.008
    C = (pa - bb - hbp - sf_est) - xH
    D = hr

    denom = max(B + C, 1.0)
    xruns = A * B / denom + D
    return float(xruns / games)


def compute_team_xruns(season: int) -> pd.DataFrame:
    """Compute team-level expected runs from Statcast contact quality.

    Uses hybrid BaseRuns: K/BB/HR at face value (no contact component),
    xBA/xSLG on balls in play from Statcast (strips BABIP/sequencing luck).

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, xrs_per_game, xra_per_game.
        Empty DataFrame if Statcast data is not available.
    """
    from src.data.db import read_sql

    # Offense: team batting stats + BIP contact quality
    offense = read_sql(
        "SELECT fg.team_id, "
        "  COUNT(DISTINCT fg.game_pk) AS games, "
        "  SUM(fg.bat_pa) AS pa, "
        "  SUM(fg.bat_k) AS k, "
        "  SUM(fg.bat_bb) AS bb, "
        "  SUM(COALESCE(fg.bat_hbp, 0)) AS hbp, "
        "  SUM(fg.bat_hr) AS hr "
        "FROM production.fact_player_game_mlb fg "
        "JOIN production.dim_game dg ON fg.game_pk = dg.game_pk "
        "WHERE dg.season = :season AND dg.game_type = 'R' "
        "  AND fg.player_role = 'batter' "
        "GROUP BY fg.team_id",
        {"season": season},
    )
    if offense.empty:
        return pd.DataFrame(columns=["team_id", "xrs_per_game", "xra_per_game"])

    bip_offense = read_sql(
        "SELECT fg.team_id, "
        "  AVG(CASE WHEN sbb.xba::text != 'NaN' THEN sbb.xba END) AS xba_on_bip, "
        "  AVG(CASE WHEN sbb.xslg::text != 'NaN' THEN sbb.xslg END) AS xslg_on_bip "
        "FROM production.fact_pa fpa "
        "JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk "
        "JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id "
        "JOIN production.fact_player_game_mlb fg "
        "  ON fpa.batter_id = fg.player_id AND fpa.game_pk = fg.game_pk "
        "  AND fg.player_role = 'batter' "
        "WHERE dg.season = :season AND dg.game_type = 'R' "
        "GROUP BY fg.team_id",
        {"season": season},
    )

    off = offense.merge(bip_offense, on="team_id", how="left")
    # Fallback: if no Statcast BIP data, use league-average xBA/xSLG
    lg_xba = bip_offense["xba_on_bip"].mean() if not bip_offense.empty else 0.328
    lg_xslg = bip_offense["xslg_on_bip"].mean() if not bip_offense.empty else 0.540
    off["xba_on_bip"] = off["xba_on_bip"].fillna(lg_xba)
    off["xslg_on_bip"] = off["xslg_on_bip"].fillna(lg_xslg)

    off["xrs_per_game"] = off.apply(
        lambda r: _baseruns(
            r["pa"], r["k"], r["bb"], r["hbp"], r["hr"],
            r["xba_on_bip"], r["xslg_on_bip"], int(r["games"]),
        ),
        axis=1,
    )

    # Pitching: team pitching stats + BIP contact quality allowed
    # pit_bf = batters faced (no pit_pa column), estimate HBP at 0.8% of BF
    pitching = read_sql(
        "SELECT fg.team_id, "
        "  COUNT(DISTINCT fg.game_pk) AS games, "
        "  SUM(fg.pit_bf) AS pa, "
        "  SUM(fg.pit_k) AS k, "
        "  SUM(fg.pit_bb) AS bb, "
        "  SUM(fg.pit_bf) * 0.008 AS hbp, "
        "  SUM(fg.pit_hr) AS hr "
        "FROM production.fact_player_game_mlb fg "
        "JOIN production.dim_game dg ON fg.game_pk = dg.game_pk "
        "WHERE dg.season = :season AND dg.game_type = 'R' "
        "  AND fg.player_role = 'pitcher' "
        "GROUP BY fg.team_id",
        {"season": season},
    )

    bip_pitching = read_sql(
        "SELECT fg.team_id, "
        "  AVG(CASE WHEN sbb.xba::text != 'NaN' THEN sbb.xba END) AS xba_on_bip, "
        "  AVG(CASE WHEN sbb.xslg::text != 'NaN' THEN sbb.xslg END) AS xslg_on_bip "
        "FROM production.fact_pa fpa "
        "JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk "
        "JOIN production.sat_batted_balls sbb ON fpa.pa_id = sbb.pa_id "
        "JOIN production.fact_player_game_mlb fg "
        "  ON fpa.pitcher_id = fg.player_id AND fpa.game_pk = fg.game_pk "
        "  AND fg.player_role = 'pitcher' "
        "WHERE dg.season = :season AND dg.game_type = 'R' "
        "GROUP BY fg.team_id",
        {"season": season},
    )

    pit = pitching.merge(bip_pitching, on="team_id", how="left")
    lg_xba_p = bip_pitching["xba_on_bip"].mean() if not bip_pitching.empty else 0.328
    lg_xslg_p = bip_pitching["xslg_on_bip"].mean() if not bip_pitching.empty else 0.540
    pit["xba_on_bip"] = pit["xba_on_bip"].fillna(lg_xba_p)
    pit["xslg_on_bip"] = pit["xslg_on_bip"].fillna(lg_xslg_p)

    pit["xra_per_game"] = pit.apply(
        lambda r: _baseruns(
            r["pa"], r["k"], r["bb"], r["hbp"], r["hr"],
            r["xba_on_bip"], r["xslg_on_bip"], int(r["games"]),
        ),
        axis=1,
    )

    result = off[["team_id", "xrs_per_game"]].merge(
        pit[["team_id", "xra_per_game"]], on="team_id", how="outer",
    )

    logger.info(
        "Team xRuns computed for %d teams: mean xRS=%.2f, mean xRA=%.2f",
        len(result), result["xrs_per_game"].mean(), result["xra_per_game"].mean(),
    )
    return result


def load_current_team_records(
    season: int,
    engine: "Engine | None" = None,
) -> pd.DataFrame:
    """Load current-season team records from fact_game_totals.

    Parameters
    ----------
    season : int
        Season year (e.g. 2026).
    engine : sqlalchemy Engine, optional
        Database engine.  If None, uses ``src.data.db.get_engine()``.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, games_played, wins, losses, runs_scored,
        runs_allowed.  Empty DataFrame if the season has no games yet.
    """
    from src.data.db import read_sql

    query = """
        WITH g AS (
            SELECT
                a.team_id,
                a.game_pk,
                a.runs AS rs,
                b.runs AS ra
            FROM production.fact_game_totals a
            JOIN production.fact_game_totals b
                ON a.game_pk = b.game_pk AND a.team_id <> b.team_id
            JOIN production.dim_game d
                ON d.game_pk = a.game_pk
            WHERE d.season = :season
              AND d.game_type = 'R'
              AND a.runs IS NOT NULL
              AND b.runs IS NOT NULL
        )
        SELECT
            team_id,
            COUNT(*) AS games_played,
            SUM(CASE WHEN rs > ra THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN rs < ra THEN 1 ELSE 0 END) AS losses,
            SUM(rs) AS runs_scored,
            SUM(ra) AS runs_allowed
        FROM g
        GROUP BY team_id
        ORDER BY team_id
    """
    try:
        df = read_sql(query, params={"season": season})
    except Exception:
        logger.exception("Failed to load current team records for %d", season)
        return pd.DataFrame(
            columns=[
                "team_id", "games_played", "wins", "losses",
                "runs_scored", "runs_allowed",
            ]
        )

    for col in ("team_id", "games_played", "wins", "losses"):
        df[col] = df[col].astype(int)
    for col in ("runs_scored", "runs_allowed"):
        df[col] = df[col].astype(float)

    # Merge Statcast-based xRuns for more stable Pythagorean estimates
    try:
        xruns = compute_team_xruns(season)
        if not xruns.empty:
            df = df.merge(xruns, on="team_id", how="left")
            logger.info("Merged xRuns for %d teams", xruns["xrs_per_game"].notna().sum())
    except Exception:
        logger.warning("Failed to compute team xRuns for %d", season, exc_info=True)

    return df
