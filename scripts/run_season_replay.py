#!/usr/bin/env python
"""
Walk-forward replay of 2026 season game predictions.

Replays the entire 2026 season day-by-day using preseason posteriors
+ daily conjugate updates, running the full-game sim for each day's
games. Produces a graded prediction file for ML/O-U/spread analysis.

Two PyMC epochs:
  - Mar 26 epoch: original preseason posteriors (2018-2025 only)
  - Apr 21 epoch: retrained with early 2026 data (current snapshots)

Usage
-----
    python scripts/run_season_replay.py              # full replay
    python scripts/run_season_replay.py --skip-pymc  # skip retrain, use Apr 21 for all
    python scripts/run_season_replay.py --start 2026-04-20 --end 2026-04-26
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql
from src.models.in_season_updater import update_rate_samples
from src.models.game_sim.lineup_simulator import simulate_full_game_both_teams
from src.models.matchup import score_matchup_for_stat
from src.utils.weather import parse_temp_bucket, wind_category, wind_speed_bucket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("season_replay")

DASHBOARD_DIR = PROJECT_ROOT.parent / "tdd-dashboard" / "data" / "dashboard"
SNAPSHOT_DIR = DASHBOARD_DIR / "snapshots"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Epoch switch date: posteriors were retrained on this date
EPOCH_SWITCH_DATE = "2026-04-21"

# League average rates for conjugate update priors
LEAGUE_AVGS = {
    ("pitcher", "k"): 0.22,
    ("pitcher", "bb"): 0.08,
    ("pitcher", "hr"): 0.03,
    ("hitter", "k"): 0.226,
    ("hitter", "bb"): 0.082,
    ("hitter", "hr"): 0.031,
}

# Rate stat configs: (player_type, stat_name, trials_col, successes_col)
RATE_STATS = [
    ("pitcher", "k", "batters_faced", "strike_outs"),
    ("pitcher", "bb", "batters_faced", "walks"),
    ("pitcher", "hr", "batters_faced", "hr"),
    ("hitter", "k", "pa", "strikeouts"),
    ("hitter", "bb", "pa", "walks"),
    ("hitter", "hr", "pa", "hr"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_apr21_snapshots() -> dict[str, dict[str, np.ndarray]]:
    """Load the Apr 21 retrained preseason snapshots."""
    samples = {}
    for ptype in ("pitcher", "hitter"):
        for stat in ("k", "bb", "hr"):
            key = f"{ptype}_{stat}"
            path = SNAPSHOT_DIR / f"{ptype}_{stat}_samples_preseason.npz"
            samples[key] = _load_npz(path)
            logger.info("Loaded %s: %d players", key, len(samples[key]))
    return samples


def load_static_priors() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Load BF priors, outs priors, and exit model from snapshots."""
    bf_path = SNAPSHOT_DIR / "bf_priors_preseason.parquet"
    bf = pd.read_parquet(bf_path) if bf_path.exists() else pd.DataFrame()

    outs_path = SNAPSHOT_DIR / "outs_priors_preseason.parquet"
    outs = pd.read_parquet(outs_path) if outs_path.exists() else pd.DataFrame()

    import pickle
    exit_path = SNAPSHOT_DIR / "exit_model_preseason.pkl"
    exit_model = None
    if exit_path.exists():
        with open(exit_path, "rb") as f:
            exit_model = pickle.load(f)

    return bf, outs, exit_model


def load_game_context_data() -> dict:
    """Load all static context data for game sims (park/ump/weather/matchup/etc)."""
    ctx: dict = {}

    # Park factor lifts
    p = DASHBOARD_DIR / "park_factor_lifts.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        ctx["park_lifts"] = {
            int(r["venue_id"]): {
                "k_lift": float(r.get("k_lift", 0)),
                "bb_lift": float(r.get("bb_lift", 0)),
                "hr_lift": float(r.get("hr_lift", 0)),
                "h_babip_adj": float(r.get("h_babip_adj", 0)),
            }
            for _, r in df.iterrows()
        }
    else:
        ctx["park_lifts"] = {}

    # Umpire tendencies
    p = DASHBOARD_DIR / "umpire_tendencies.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        ctx["ump_k"] = {}
        ctx["ump_bb"] = {}
        for _, r in df.iterrows():
            name = r.get("hp_umpire_name", r.get("umpire_name", ""))
            ctx["ump_k"][name] = float(r.get("k_logit_lift", r.get("k_lift", 0)))
            ctx["ump_bb"][name] = float(r.get("bb_logit_lift", r.get("bb_lift", 0)))
    else:
        ctx["ump_k"] = {}
        ctx["ump_bb"] = {}

    # Weather effects
    ctx["wx_3key"] = {}
    ctx["wx_2key"] = {}
    p = DASHBOARD_DIR / "weather_effects.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            entry = {
                "k_multiplier": float(r.get("k_multiplier", 1.0)),
                "overall_k_rate": float(r.get("overall_k_rate", 0.22)),
                "hr_multiplier": float(r.get("hr_multiplier", 1.0)),
                "overall_hr_rate": float(r.get("overall_hr_rate", 0.03)),
            }
            tb = r.get("temp_bucket", "")
            wc = r.get("wind_category", "")
            ws = r.get("wind_speed_bucket", "")
            if ws:
                ctx["wx_3key"][(tb, wc, ws)] = entry
            ctx["wx_2key"][(tb, wc)] = entry

    # Matchup data
    for name in ("pitcher_arsenal", "hitter_vuln"):
        p = DASHBOARD_DIR / f"{name}.parquet"
        ctx[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()

    # Baselines for matchup scoring
    try:
        from src.data.league_baselines import get_baselines_dict
        ctx["baselines_pt"] = get_baselines_dict(
            grouping="pitch_type", recency_weights="marcel",
        )
    except Exception:
        ctx["baselines_pt"] = {}

    # BIP profiles
    _LEAGUE_BIP = np.array([0.700, 0.222, 0.065, 0.005])
    ctx["bip_lookup"] = {}
    p = DASHBOARD_DIR / "batter_bip_profiles.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            latest = df.sort_values("season").groupby("batter_id").last().reset_index()
            for _, r in latest.iterrows():
                ctx["bip_lookup"][int(r["batter_id"])] = np.array([
                    float(r.get("p_out", _LEAGUE_BIP[0])),
                    float(r.get("p_single", _LEAGUE_BIP[1])),
                    float(r.get("p_double", _LEAGUE_BIP[2])),
                    float(r.get("p_triple", _LEAGUE_BIP[3])),
                ])
    ctx["league_bip"] = _LEAGUE_BIP

    # LD-rate BABIP adjustment
    ctx["ld_babip"] = {}
    p = DASHBOARD_DIR / "batter_ld_rate.parquet"
    if p.exists():
        from src.utils.constants import BABIP_LD_COEFF, BABIP_LEAGUE_LD_RATE
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            ld = float(r.get("ld_rate_regressed", BABIP_LEAGUE_LD_RATE))
            pid = int(r.get("batter_id", r.get("player_id", 0)))
            if pid:
                ctx["ld_babip"][pid] = (ld - BABIP_LEAGUE_LD_RATE) * BABIP_LD_COEFF

    # Sprint speed BABIP
    ctx["speed_babip"] = {}
    p = DASHBOARD_DIR / "batter_sprint_speed.parquet"
    if p.exists():
        from src.utils.constants import BABIP_LEAGUE_SPEED, BABIP_SPEED_COEFF
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            spd = float(r.get("sprint_speed_regressed", BABIP_LEAGUE_SPEED))
            pid = int(r.get("batter_id", r.get("player_id", 0)))
            if pid:
                ctx["speed_babip"][pid] = (spd - BABIP_LEAGUE_SPEED) * BABIP_SPEED_COEFF

    # Form lifts
    ctx["batter_form"] = {}
    ctx["pitcher_form"] = {}
    try:
        from src.models.game_sim.form_model import (
            BatterFormLifts, PitcherFormLifts,
            build_batter_form_lifts_batch, build_pitcher_form_lifts_batch,
        )
        from src.data.queries.environment import get_rolling_form, get_rolling_hard_hit
        bat_roll = get_rolling_form("batter")
        pit_roll = get_rolling_form("pitcher")
        hh_df = get_rolling_hard_hit()
        ctx["batter_form"] = build_batter_form_lifts_batch(bat_roll, hard_hit_df=hh_df)
        ctx["pitcher_form"] = build_pitcher_form_lifts_batch(pit_roll)
        ctx["BatterFormLifts"] = BatterFormLifts
        ctx["PitcherFormLifts"] = PitcherFormLifts
    except Exception as e:
        logger.warning("Form lifts unavailable: %s", e)
        from dataclasses import dataclass, field as dc_field
        @dataclass
        class _BFL:
            k_lift: float = 0.0
            bb_lift: float = 0.0
            hr_lift: float = 0.0
            hh_lift: float = 0.0
        @dataclass
        class _PFL:
            bb_lift: float = 0.0
        ctx["BatterFormLifts"] = _BFL
        ctx["PitcherFormLifts"] = _PFL

    # Bullpen profiles
    ctx["bullpen_profiles"] = {}
    p = DASHBOARD_DIR / "team_bullpen_profiles.parquet"
    if p.exists():
        try:
            from src.models.game_sim.bullpen_model import TeamBullpenProfile
            df = pd.read_parquet(p)
            for _, r in df.iterrows():
                ctx["bullpen_profiles"][int(r["team_id"])] = TeamBullpenProfile(
                    team_id=int(r["team_id"]),
                    high_lev_k_rate=float(r.get("high_lev_k_rate", 0.25)),
                    high_lev_bb_rate=float(r.get("high_lev_bb_rate", 0.08)),
                    high_lev_hr_rate=float(r.get("high_lev_hr_rate", 0.025)),
                    high_lev_bf=int(r.get("high_lev_bf", 5)),
                    low_lev_k_rate=float(r.get("low_lev_k_rate", 0.22)),
                    low_lev_bb_rate=float(r.get("low_lev_bb_rate", 0.09)),
                    low_lev_hr_rate=float(r.get("low_lev_hr_rate", 0.03)),
                    low_lev_bf=int(r.get("low_lev_bf", 10)),
                )
        except Exception as e:
            logger.warning("Bullpen profiles unavailable: %s", e)

    # Bullpen scalar rates
    ctx["bullpen_rates"] = {}
    p = DASHBOARD_DIR / "team_bullpen_rates.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            latest = df.sort_values("season").groupby("team_id").last().reset_index()
            for _, r in latest.iterrows():
                ctx["bullpen_rates"][int(r["team_id"])] = (
                    float(r.get("k_rate", 0.253)),
                    float(r.get("bb_rate", 0.084)),
                    float(r.get("hr_rate", 0.024)),
                )

    logger.info(
        "Context loaded: %d park, %d ump, %d wx_3key, %d bip, %d ld_babip, "
        "%d speed_babip, %d batter_form, %d pitcher_form, %d bp_profiles",
        len(ctx["park_lifts"]), len(ctx["ump_k"]),
        len(ctx["wx_3key"]), len(ctx["bip_lookup"]),
        len(ctx["ld_babip"]), len(ctx["speed_babip"]),
        len(ctx["batter_form"]), len(ctx["pitcher_form"]),
        len(ctx["bullpen_profiles"]),
    )
    return ctx


def _compute_matchup_lifts(
    pitcher_id: int,
    lineup_ids: list[int],
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict,
) -> dict[str, np.ndarray]:
    """Compute per-batter matchup lifts for a 9-man lineup."""
    result = {"k": np.zeros(9), "bb": np.zeros(9), "hr": np.zeros(9)}
    if pitcher_arsenal.empty or hitter_vuln.empty or not baselines_pt:
        return result
    for stat in ("k", "bb", "hr"):
        for i, bid in enumerate(lineup_ids[:9]):
            try:
                lift = score_matchup_for_stat(
                    stat, pitcher_id, bid,
                    pitcher_arsenal, hitter_vuln, baselines_pt,
                )
                result[stat][i] = float(lift) if lift is not None else 0.0
            except Exception:
                pass
    return result


def _weather_lifts(game: pd.Series, ctx: dict) -> tuple[float, float]:
    """Extract weather K and HR logit lifts for a game."""
    from scipy.special import logit as _logit
    temp = game.get("weather_temp")
    if pd.isna(temp):
        return 0.0, 0.0
    tb = parse_temp_bucket(temp)
    wc = wind_category(game)
    ws = wind_speed_bucket(game)
    entry = ctx["wx_3key"].get((tb, wc, ws)) or ctx["wx_2key"].get((tb, wc))
    if entry is None:
        return 0.0, 0.0
    clip = lambda x: max(min(x, 1 - 1e-6), 1e-6)
    k_adj = entry["overall_k_rate"] * entry["k_multiplier"]
    wx_k = _logit(clip(k_adj)) - _logit(clip(entry["overall_k_rate"]))
    hr_adj = entry["overall_hr_rate"] * entry["hr_multiplier"]
    wx_hr = _logit(clip(hr_adj)) - _logit(clip(entry["overall_hr_rate"]))
    return float(wx_k), float(wx_hr)


def get_game_dates() -> list[str]:
    """Get all 2026 regular-season game dates from the DB."""
    df = read_sql("""
        SELECT DISTINCT game_date::date::text AS game_date
        FROM production.dim_game
        WHERE season = 2026 AND game_type = 'R'
        ORDER BY game_date
    """)
    return df["game_date"].tolist()


def get_schedule_for_date(game_date: str) -> pd.DataFrame:
    """Get games + starters + umpire/weather for a specific date."""
    df = read_sql("""
        SELECT
            dg.game_pk,
            dg.game_date::date::text AS game_date,
            dg.home_team_id,
            dg.away_team_id,
            ht.abbreviation AS home_abbr,
            at.abbreviation AS away_abbr,
            dg.venue_id,
            du.hp_umpire_name,
            dw.temperature AS weather_temp,
            dw.wind_speed AS weather_wind_speed,
            dw.wind_direction AS weather_wind_direction,
            dw.condition AS weather_condition
        FROM production.dim_game dg
        JOIN production.dim_team ht ON dg.home_team_id = ht.team_id
        JOIN production.dim_team at ON dg.away_team_id = at.team_id
        LEFT JOIN production.dim_umpire du
            ON dg.game_pk = du.game_pk
        LEFT JOIN production.dim_weather dw
            ON dg.game_pk = dw.game_pk
        WHERE dg.game_date::date = :gd AND dg.game_type = 'R'
        ORDER BY dg.game_pk
    """, {"gd": game_date})

    # Get starters: one per team per game
    if df.empty:
        return df

    starters = read_sql("""
        SELECT pb.game_pk, pb.pitcher_id, pb.team_id
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.game_date::date = :gd AND dg.game_type = 'R'
          AND pb.is_starter = true
    """, {"gd": game_date})

    # Pivot to home_pitcher_id / away_pitcher_id
    for _, row in starters.iterrows():
        mask = df["game_pk"] == row["game_pk"]
        if (df.loc[mask, "home_team_id"] == row["team_id"]).any():
            df.loc[mask, "home_pitcher_id"] = int(row["pitcher_id"])
        else:
            df.loc[mask, "away_pitcher_id"] = int(row["pitcher_id"])

    return df


def get_lineups_for_date(game_date: str) -> dict[tuple[int, int], list[int]]:
    """Get confirmed batting orders: {(game_pk, team_id): [batter_ids]}."""
    df = read_sql("""
        SELECT fl.game_pk, fl.team_id, fl.player_id, fl.batting_order
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        WHERE dg.game_date::date = :gd AND fl.is_starter = true
        ORDER BY fl.game_pk, fl.team_id, fl.batting_order
    """, {"gd": game_date})

    lineups: dict[tuple[int, int], list[int]] = {}
    for (gp, tid), grp in df.groupby(["game_pk", "team_id"]):
        lineups[(int(gp), int(tid))] = grp.sort_values("batting_order")["player_id"].tolist()
    return lineups


def get_actual_scores(game_date: str) -> dict[int, tuple[int, int]]:
    """Get actual scores: {game_pk: (away_score, home_score)}."""
    df = read_sql("""
        SELECT
            dg.game_pk,
            SUM(CASE WHEN bb.team_id = dg.away_team_id THEN bb.runs ELSE 0 END) AS away_runs,
            SUM(CASE WHEN bb.team_id = dg.home_team_id THEN bb.runs ELSE 0 END) AS home_runs
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        WHERE dg.game_date::date = :gd AND dg.game_type = 'R'
        GROUP BY dg.game_pk
    """, {"gd": game_date})

    return {
        int(row["game_pk"]): (int(row["away_runs"]), int(row["home_runs"]))
        for _, row in df.iterrows()
    }


def get_observed_totals(
    player_type: str, up_to_date: str,
) -> pd.DataFrame:
    """Query cumulative 2026 observed totals through a given date."""
    if player_type == "pitcher":
        return read_sql("""
            SELECT
                pb.pitcher_id,
                SUM(pb.batters_faced) AS batters_faced,
                SUM(pb.strike_outs) AS strike_outs,
                SUM(pb.walks) AS walks,
                SUM(pb.home_runs) AS hr
            FROM staging.pitching_boxscores pb
            JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
            WHERE dg.season = 2026 AND dg.game_type = 'R'
              AND dg.game_date::date <= :up_to
            GROUP BY pb.pitcher_id
            HAVING SUM(pb.batters_faced) >= 1
        """, {"up_to": up_to_date})
    else:
        return read_sql("""
            SELECT
                fp.batter_id,
                COUNT(DISTINCT fp.pa_id) AS pa,
                SUM(CASE WHEN fp.events = 'strikeout' THEN 1 ELSE 0 END) AS strikeouts,
                SUM(CASE WHEN fp.events = 'walk' THEN 1 ELSE 0 END) AS walks,
                SUM(CASE WHEN fp.events = 'home_run' THEN 1 ELSE 0 END) AS hr
            FROM production.fact_pa fp
            JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
            WHERE dg.season = 2026 AND dg.game_type = 'R'
              AND dg.game_date::date <= :up_to
            GROUP BY fp.batter_id
            HAVING COUNT(DISTINCT fp.pa_id) >= 1
        """, {"up_to": up_to_date})


def get_bullpen_rates() -> dict[int, tuple[float, float, float]]:
    """Get team bullpen aggregate rates: {team_id: (k_rate, bb_rate, hr_rate)}."""
    df = read_sql("""
        SELECT
            pb.team_id,
            SUM(pb.strike_outs)::float / NULLIF(SUM(pb.batters_faced), 0) AS k_rate,
            SUM(pb.walks)::float / NULLIF(SUM(pb.batters_faced), 0) AS bb_rate,
            SUM(pb.home_runs)::float / NULLIF(SUM(pb.batters_faced), 0) AS hr_rate
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.season = 2026 AND dg.game_type = 'R' AND pb.is_starter = false
        GROUP BY pb.team_id
        HAVING SUM(pb.batters_faced) >= 50
    """)
    default = (0.253, 0.084, 0.024)
    return {
        int(row["team_id"]): (
            float(row["k_rate"] or default[0]),
            float(row["bb_rate"] or default[1]),
            float(row["hr_rate"] or default[2]),
        )
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Conjugate update
# ---------------------------------------------------------------------------
def conjugate_update_all(
    base_samples: dict[str, dict[str, np.ndarray]],
    up_to_date: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Run conjugate updates for all 6 rate stats through a given date."""
    updated = {}
    for ptype, stat, trials_col, succ_col in RATE_STATS:
        key = f"{ptype}_{stat}"
        preseason = base_samples.get(key, {})
        if not preseason:
            updated[key] = {}
            continue

        obs = get_observed_totals(ptype, up_to_date)
        id_col = "pitcher_id" if ptype == "pitcher" else "batter_id"

        if obs.empty:
            updated[key] = preseason
        else:
            updated[key] = update_rate_samples(
                preseason, obs,
                player_id_col=id_col,
                trials_col=trials_col,
                successes_col=succ_col,
                league_avg=LEAGUE_AVGS[(ptype, stat)],
                min_trials=10,
                n_samples=1000,
                random_seed=42,
            )
    return updated


# ---------------------------------------------------------------------------
# Game simulation
# ---------------------------------------------------------------------------
def _posterior_samples(
    npz: dict[str, np.ndarray], pid: str, fallback: float,
    n: int = 1000,
) -> np.ndarray:
    s = npz.get(pid)
    if s is not None:
        return s
    return np.full(n, fallback)


def simulate_game_day(
    schedule: pd.DataFrame,
    lineups: dict[tuple[int, int], list[int]],
    samples: dict[str, dict[str, np.ndarray]],
    bf_priors: pd.DataFrame,
    ctx: dict,
    n_sims: int = 5000,
) -> list[dict]:
    """Run full-game sims for all games on a given date."""
    predictions = []
    _FN = n_sims
    default_bp = (0.253, 0.084, 0.024)
    bullpen_rates = ctx.get("bullpen_rates", {})
    BFL = ctx.get("BatterFormLifts")
    PFL = ctx.get("PitcherFormLifts")

    for _, game in schedule.iterrows():
        game_pk = int(game["game_pk"])
        home_team = int(game["home_team_id"])
        away_team = int(game["away_team_id"])
        home_pid = game.get("home_pitcher_id")
        away_pid = game.get("away_pitcher_id")

        if pd.isna(home_pid) or pd.isna(away_pid):
            continue
        home_pid = int(home_pid)
        away_pid = int(away_pid)

        # Lineups
        home_lineup = lineups.get((game_pk, home_team), [])
        away_lineup = lineups.get((game_pk, away_team), [])
        if len(home_lineup) < 9 or len(away_lineup) < 9:
            continue

        # Posterior samples
        pk, pbb, phr = samples["pitcher_k"], samples["pitcher_bb"], samples["pitcher_hr"]
        hk, hbb, hhr = samples["hitter_k"], samples["hitter_bb"], samples["hitter_hr"]

        home_starter_k = _posterior_samples(pk, str(away_pid), 0.22, _FN)
        home_starter_bb = _posterior_samples(pbb, str(away_pid), 0.08, _FN)
        home_starter_hr = _posterior_samples(phr, str(away_pid), 0.03, _FN)
        away_starter_k = _posterior_samples(pk, str(home_pid), 0.22, _FN)
        away_starter_bb = _posterior_samples(pbb, str(home_pid), 0.08, _FN)
        away_starter_hr = _posterior_samples(phr, str(home_pid), 0.03, _FN)

        def _build_lineup_samples(lineup_ids, k_npz, bb_npz, hr_npz):
            ks, bbs, hrs = [], [], []
            for bid in lineup_ids[:9]:
                bid_str = str(bid)
                ks.append(k_npz.get(bid_str, np.full(_FN, 0.226)))
                bbs.append(bb_npz.get(bid_str, np.full(_FN, 0.082)))
                hrs.append(hr_npz.get(bid_str, np.full(_FN, 0.031)))
            return ks, bbs, hrs

        home_k_s, home_bb_s, home_hr_s = _build_lineup_samples(home_lineup, hk, hbb, hhr)
        away_k_s, away_bb_s, away_hr_s = _build_lineup_samples(away_lineup, hk, hbb, hhr)

        # BF priors
        def _get_bf(pid):
            row = bf_priors[bf_priors["pitcher_id"] == pid]
            if len(row) > 0:
                return float(row.iloc[0]["mu_bf"]), float(row.iloc[0]["sigma_bf"])
            return 22.0, 4.5

        home_bf_mu, home_bf_sigma = _get_bf(away_pid)
        away_bf_mu, away_bf_sigma = _get_bf(home_pid)

        # Bullpen rates
        home_bp = bullpen_rates.get(away_team, default_bp)
        away_bp = bullpen_rates.get(home_team, default_bp)

        # Bullpen profiles
        home_bp_profile = ctx["bullpen_profiles"].get(away_team)
        away_bp_profile = ctx["bullpen_profiles"].get(home_team)

        # --- Context lifts ---
        venue_id = game.get("venue_id")
        park = ctx["park_lifts"].get(int(venue_id), {}) if pd.notna(venue_id) else {}
        park_k = park.get("k_lift", 0.0)
        park_bb = park.get("bb_lift", 0.0)
        park_hr = park.get("hr_lift", 0.0)
        park_babip = park.get("h_babip_adj", 0.0)

        hp_ump = game.get("hp_umpire_name", "")
        ump_k = ctx["ump_k"].get(hp_ump, 0.0)
        ump_bb = ctx["ump_bb"].get(hp_ump, 0.0)

        wx_k, wx_hr = _weather_lifts(game, ctx)

        # Matchup lifts
        away_matchup = _compute_matchup_lifts(
            away_pid, home_lineup, ctx["pitcher_arsenal"],
            ctx["hitter_vuln"], ctx["baselines_pt"],
        )
        home_matchup = _compute_matchup_lifts(
            home_pid, away_lineup, ctx["pitcher_arsenal"],
            ctx["hitter_vuln"], ctx["baselines_pt"],
        )

        # BABIP adjustments per batter
        def _babip_adjs(lineup_ids):
            return np.array([
                ctx["ld_babip"].get(int(bid), 0.0) + ctx["speed_babip"].get(int(bid), 0.0)
                for bid in lineup_ids[:9]
            ])

        away_babip = _babip_adjs(away_lineup)
        home_babip = _babip_adjs(home_lineup)

        # BIP profiles per batter
        league_bip = ctx["league_bip"]
        def _bip_probs(lineup_ids):
            return np.stack([
                ctx["bip_lookup"].get(int(bid), league_bip)
                for bid in lineup_ids[:9]
            ], axis=0).astype(np.float64)

        away_bip = _bip_probs(away_lineup)
        home_bip = _bip_probs(home_lineup)

        # Form lifts per batter
        batter_form = ctx.get("batter_form", {})
        def _form_lifts(lineup_ids):
            fk, fbb, fhr = np.zeros(9), np.zeros(9), np.zeros(9)
            for i, bid in enumerate(lineup_ids[:9]):
                bf = batter_form.get(int(bid), BFL())
                fk[i] = bf.k_lift
                fbb[i] = bf.bb_lift
                fhr[i] = bf.hr_lift + bf.hh_lift
            return fk, fbb, fhr

        away_fk, away_fbb, away_fhr = _form_lifts(away_lineup)
        home_fk, home_fbb, home_fhr = _form_lifts(home_lineup)

        try:
            fg = simulate_full_game_both_teams(
                away_batter_k_rate_samples=away_k_s,
                away_batter_bb_rate_samples=away_bb_s,
                away_batter_hr_rate_samples=away_hr_s,
                home_starter_k_rate=home_starter_k,
                home_starter_bb_rate=home_starter_bb,
                home_starter_hr_rate=home_starter_hr,
                home_starter_bf_mu=home_bf_mu,
                home_starter_bf_sigma=home_bf_sigma,
                home_batter_k_rate_samples=home_k_s,
                home_batter_bb_rate_samples=home_bb_s,
                home_batter_hr_rate_samples=home_hr_s,
                away_starter_k_rate=away_starter_k,
                away_starter_bb_rate=away_starter_bb,
                away_starter_hr_rate=away_starter_hr,
                away_starter_bf_mu=away_bf_mu,
                away_starter_bf_sigma=away_bf_sigma,
                # Matchup lifts
                away_matchup_k_lifts=home_matchup["k"],
                away_matchup_bb_lifts=home_matchup["bb"],
                away_matchup_hr_lifts=home_matchup["hr"],
                home_matchup_k_lifts=away_matchup["k"],
                home_matchup_bb_lifts=away_matchup["bb"],
                home_matchup_hr_lifts=away_matchup["hr"],
                # Bullpen
                home_bullpen_k_rate=home_bp[0],
                home_bullpen_bb_rate=home_bp[1],
                home_bullpen_hr_rate=home_bp[2],
                away_bullpen_k_rate=away_bp[0],
                away_bullpen_bb_rate=away_bp[1],
                away_bullpen_hr_rate=away_bp[2],
                home_bullpen_profile=home_bp_profile,
                away_bullpen_profile=away_bp_profile,
                # BABIP + BIP
                away_batter_babip_adjs=away_babip,
                home_batter_babip_adjs=home_babip,
                away_batter_bip_probs=away_bip,
                home_batter_bip_probs=home_bip,
                # Shared context
                umpire_k_lift=ump_k,
                umpire_bb_lift=ump_bb,
                park_k_lift=park_k,
                park_bb_lift=park_bb,
                park_hr_lift=park_hr,
                park_h_babip_adj=park_babip,
                weather_k_lift=wx_k,
                weather_hr_lift=wx_hr,
                # Form lifts
                away_form_k_lifts=away_fk,
                away_form_bb_lifts=away_fbb,
                away_form_hr_lifts=away_fhr,
                home_form_k_lifts=home_fk,
                home_form_bb_lifts=home_fbb,
                home_form_hr_lifts=home_fhr,
                n_sims=n_sims,
                random_seed=42 + game_pk % 10000 + 7777,
            )
        except Exception as e:
            logger.warning("Sim failed game %d: %s", game_pk, e)
            continue

        home_runs = fg.home_runs.astype(float)
        away_runs = fg.away_runs.astype(float)
        total = home_runs + away_runs
        margin = home_runs - away_runs

        n = len(margin)
        home_wp = (np.sum(margin > 0) + 0.5 * np.sum(margin == 0)) / n

        rec = {
            "game_pk": game_pk,
            "game_date": game["game_date"],
            "home_abbr": game.get("home_abbr", ""),
            "away_abbr": game.get("away_abbr", ""),
            "home_win_prob": round(float(home_wp), 4),
            "away_win_prob": round(1.0 - float(home_wp), 4),
            "home_runs_mean": round(float(np.mean(home_runs)), 2),
            "away_runs_mean": round(float(np.mean(away_runs)), 2),
            "total_runs_mean": round(float(np.mean(total)), 2),
            "total_runs_std": round(float(np.std(total)), 2),
            "home_margin_mean": round(float(np.mean(margin)), 2),
        }
        for line in [5.5, 6.5, 7.5, 8.5, 9.5, 10.5]:
            rec[f"p_over_{line:.1f}"] = round(float(np.mean(total > line)), 4)
        for line in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            rec[f"p_home_cover_{line:+.1f}"] = round(float(np.mean(margin > line)), 4)

        predictions.append(rec)

    return predictions


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def grade_predictions(df: pd.DataFrame) -> None:
    """Print graded results for ML, O/U, spread."""
    n = len(df)
    if n == 0:
        print("No predictions to grade.")
        return

    print(f"\n{'=' * 80}")
    print(f"SEASON REPLAY RESULTS — {n} games")
    print(f"{'=' * 80}")

    # Moneyline
    df["fav_is_home"] = df["home_win_prob"] > 0.5
    df["fav_prob"] = df[["home_win_prob", "away_win_prob"]].max(axis=1)
    df["fav_correct"] = (
        (df["fav_is_home"] & df["actual_home_win"]) |
        (~df["fav_is_home"] & ~df["actual_home_win"])
    )
    ml_acc = df["fav_correct"].mean()
    print(f"\nMONEYLINE: {df['fav_correct'].sum()}/{n} = {ml_acc:.1%}")

    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-80%", "80%+"]
    df["conf_bin"] = pd.cut(df["fav_prob"], bins=bins, labels=labels)
    for bucket, grp in df.groupby("conf_bin", observed=False):
        if len(grp) < 3:
            continue
        acc = grp["fav_correct"].mean()
        print(f"  {bucket}: {grp['fav_correct'].sum()}/{len(grp)} = {acc:.1%}")

    # O/U at 8.5
    df["model_over_8_5"] = df["p_over_8.5"] > 0.5
    df["actual_over_8_5"] = df["actual_total"] > 8.5
    ou_correct = (df["model_over_8_5"] == df["actual_over_8_5"]).sum()
    print(f"\nOVER/UNDER (8.5): {ou_correct}/{n} = {ou_correct/n:.1%}")
    print(f"  Pred total: {df['total_runs_mean'].mean():.1f} | Actual: {df['actual_total'].mean():.1f} | Bias: {df['total_runs_mean'].mean() - df['actual_total'].mean():+.1f}")

    # Spread -1.5
    df["model_home_cover"] = df["p_home_cover_-1.5"] > 0.5
    df["actual_home_cover"] = df["actual_home_margin"] > 1.5
    sp_correct = (df["model_home_cover"] == df["actual_home_cover"]).sum()
    print(f"\nSPREAD (-1.5): {sp_correct}/{n} = {sp_correct/n:.1%}")

    # By epoch
    if "epoch" in df.columns:
        print(f"\n{'=' * 80}")
        print("BY EPOCH")
        for epoch, grp in df.groupby("epoch"):
            ml = grp["fav_correct"].mean()
            ou = (grp["model_over_8_5"] == grp["actual_over_8_5"]).mean()
            sp = (grp["model_home_cover"] == grp["actual_home_cover"]).mean()
            bias = grp["total_runs_mean"].mean() - grp["actual_total"].mean()
            print(f"  {epoch}: ML={ml:.1%} O/U={ou:.1%} Spread={sp:.1%} RunBias={bias:+.1f} (n={len(grp)})")

    # Brier score
    df["brier"] = (df["home_win_prob"] - df["actual_home_win"].astype(float)) ** 2
    print(f"\nBrier score: {df['brier'].mean():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward 2026 season replay")
    parser.add_argument("--skip-pymc", action="store_true",
                        help="Skip PyMC retrain; use Apr 21 snapshots for all dates")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD). Default: first game date.")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD). Default: latest game date.")
    parser.add_argument("--n-sims", type=int, default=5000,
                        help="MC sims per game (default: 5000)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    # --- Load static resources ---
    logger.info("Loading static priors and context data...")
    bf_priors, outs_priors, exit_model = load_static_priors()
    game_ctx = load_game_context_data()
    logger.info("Loaded BF priors (%d)", len(bf_priors))

    # --- Determine epochs ---
    if args.skip_pymc:
        logger.info("Skipping PyMC retrain — using Apr 21 snapshots for all dates")
        mar26_samples = load_apr21_snapshots()
        apr21_samples = mar26_samples
    else:
        # Phase 1: Retrain Mar 26 epoch
        logger.info("=" * 70)
        logger.info("PHASE 1: Retraining PyMC for Mar 26 epoch...")
        logger.info("=" * 70)
        try:
            from scripts.precompute.models import run as run_models
            hitter_res, pitcher_res, h_proj, p_proj = run_models(
                draws=1000, tune=500, chains=2,
                include_hitter=True, include_pitcher=True,
            )
            # Extract samples from model results
            mar26_samples = {}
            for ptype, results in [("hitter", hitter_res), ("pitcher", pitcher_res)]:
                for stat in ("k", "bb", "hr"):
                    key = f"{ptype}_{stat}"
                    rate_key = f"{stat}_rate" if stat != "hr" else "hr_per_fb" if ptype == "hitter" else "hr_rate"
                    if rate_key in results and "rate_samples" in results[rate_key]:
                        mar26_samples[key] = results[rate_key]["rate_samples"]
                    else:
                        mar26_samples[key] = {}
            logger.info("Mar 26 epoch samples extracted")
        except Exception as e:
            logger.warning("PyMC retrain failed: %s — falling back to Apr 21 snapshots", e)
            mar26_samples = load_apr21_snapshots()

        apr21_samples = load_apr21_snapshots()

    # --- Get game dates ---
    all_dates = get_game_dates()
    if args.start:
        all_dates = [d for d in all_dates if d >= args.start]
    if args.end:
        all_dates = [d for d in all_dates if d <= args.end]
    logger.info("Replaying %d game dates: %s to %s", len(all_dates), all_dates[0], all_dates[-1])

    # --- Day-by-day replay ---
    all_predictions = []
    for i, game_date in enumerate(all_dates):
        day_t0 = time.time()

        # Epoch selection
        if game_date >= EPOCH_SWITCH_DATE:
            base_samples = apr21_samples
            epoch = "apr21"
        else:
            base_samples = mar26_samples
            epoch = "mar26"

        # Conjugate update with data through yesterday
        yesterday = (datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        updated = conjugate_update_all(base_samples, yesterday)

        # Schedule + lineups
        schedule = get_schedule_for_date(game_date)
        if schedule.empty:
            logger.info("[%d/%d] %s: no games", i + 1, len(all_dates), game_date)
            continue

        lineups = get_lineups_for_date(game_date)
        scores = get_actual_scores(game_date)

        # Run sims
        day_preds = simulate_game_day(
            schedule, lineups, updated, bf_priors, game_ctx,
            n_sims=args.n_sims,
        )

        # Attach actuals + epoch
        for pred in day_preds:
            gp = pred["game_pk"]
            if gp in scores:
                away_r, home_r = scores[gp]
                pred["actual_away_score"] = away_r
                pred["actual_home_score"] = home_r
                pred["actual_total"] = away_r + home_r
                pred["actual_home_margin"] = home_r - away_r
                pred["actual_home_win"] = home_r > away_r
            pred["epoch"] = epoch

        all_predictions.extend(day_preds)

        day_elapsed = time.time() - day_t0
        logger.info(
            "[%d/%d] %s (%s): %d games simmed in %.1fs",
            i + 1, len(all_dates), game_date, epoch,
            len(day_preds), day_elapsed,
        )

    # --- Save + grade ---
    results = pd.DataFrame(all_predictions)
    out_path = OUTPUT_DIR / "replay_game_predictions_2026.parquet"
    results.to_parquet(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(results), out_path)

    # Filter to games with actual scores
    if not results.empty and "actual_total" in results.columns:
        graded = results[results["actual_total"].notna()].copy()
        grade_predictions(graded)
    else:
        logger.warning("No predictions with actuals to grade")

    elapsed = time.time() - t0
    logger.info("Total replay time: %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
