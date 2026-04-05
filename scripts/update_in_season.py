#!/usr/bin/env python
"""
In-season daily update script.

Performs Beta-Binomial conjugate updating of preseason projections
with observed 2026 data, regenerates dashboard parquets, and
fetches today's schedule with matchup analysis.

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_REPO = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard")
DASHBOARD_DIR = DASHBOARD_REPO / "data" / "dashboard"
SNAPSHOT_DIR = DASHBOARD_DIR / "snapshots"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEASON = 2026

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


def update_rate_samples_step(
    stat_name: str,
    trials_col: str,
    successes_col: str,
    league_avg: float,
) -> dict[str, np.ndarray]:
    """Conjugate-update pitcher BB% or HR/BF samples."""
    from src.models.in_season_updater import update_rate_samples

    preseason = load_preseason_rate_samples(stat_name)
    if not preseason:
        logger.warning("No preseason %s samples — skipping", stat_name)
        return {}

    p_obs = get_observed_pitcher_totals()
    if p_obs.empty:
        logger.info("No 2026 pitcher data — using preseason %s samples", stat_name)
        return preseason

    updated = update_rate_samples(
        preseason, p_obs,
        player_id_col="pitcher_id",
        trials_col=trials_col,
        successes_col=successes_col,
        league_avg=league_avg,
        min_trials=10, n_samples=1000,
    )
    logger.info("Updated %s samples: %d pitchers", stat_name, len(updated))
    return updated


def update_hitter_rate_samples_step(
    stat_name: str,
    trials_col: str,
    successes_col: str,
    league_avg: float,
) -> dict[str, np.ndarray]:
    """Conjugate-update hitter K%, BB%, or HR rate samples."""
    from src.models.in_season_updater import update_rate_samples

    preseason = load_hitter_preseason_samples(stat_name)
    if not preseason:
        logger.warning("No preseason hitter %s samples — skipping", stat_name)
        return {}

    h_obs = get_observed_hitter_totals()
    if h_obs.empty:
        logger.info("No 2026 hitter data — using preseason %s samples", stat_name)
        return preseason

    updated = update_rate_samples(
        preseason, h_obs,
        player_id_col="batter_id",
        trials_col=trials_col,
        successes_col=successes_col,
        league_avg=league_avg,
        min_trials=10, n_samples=1000,
    )
    logger.info("Updated hitter %s samples: %d batters", stat_name, len(updated))
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


def simulate_todays_games(
    schedule: pd.DataFrame,
    lineups: pd.DataFrame,
    k_samples: dict[str, np.ndarray],
    pitcher_proj: pd.DataFrame,
    bf_priors: pd.DataFrame,
) -> pd.DataFrame:
    """Step 4: Run PA-by-PA simulations for each starting pitcher.

    Uses the full game simulator (simulate_game) with exit model,
    matchup lifts, TTO adjustments, and pitch count features for
    comprehensive stat distributions (K, BB, H, HR, IP, pitches).
    """
    from src.models.game_sim.exit_model import ExitModel
    from src.models.game_sim.pa_outcome_model import GameContext
    from src.models.game_sim.simulator import (
        simulate_game,
        compute_stamina_offset,
        compute_exit_offset,
    )
    from src.models.game_sim.pitch_count_model import build_pitch_count_features
    from src.models.game_sim.tto_model import build_all_tto_lifts
    from src.models.game_sim.form_model import (
        build_pitcher_form_lifts_batch,
        build_batter_form_lifts_batch,
        BatterFormLifts,
        PitcherFormLifts,
    )
    from src.models.matchup import score_matchup_for_stat
    from src.data.league_baselines import get_baselines_dict
    from src.data.queries import get_rolling_form, get_rolling_hard_hit

    # Load matchup data
    arsenal_path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    vuln_path = DASHBOARD_DIR / "hitter_vuln_career.parquet"
    arsenal_df = pd.read_parquet(arsenal_path) if arsenal_path.exists() else pd.DataFrame()
    vuln_df = pd.read_parquet(vuln_path) if vuln_path.exists() else pd.DataFrame()
    baselines_pt = get_baselines_dict(seasons=list(range(2020, SEASON)), recency_weights="equal")

    # Load exit model
    exit_model = ExitModel()
    exit_model_path = DASHBOARD_DIR / "exit_model.pkl"
    if exit_model_path.exists():
        exit_model.load(exit_model_path)
    else:
        logger.warning("No exit model found at %s — using fallback", exit_model_path)

    # Load BB% and HR/BF samples
    bb_npz_path = DASHBOARD_DIR / "pitcher_bb_samples.npz"
    hr_npz_path = DASHBOARD_DIR / "pitcher_hr_samples.npz"
    bb_npz = dict(np.load(bb_npz_path)) if bb_npz_path.exists() else {}
    hr_npz = dict(np.load(hr_npz_path)) if hr_npz_path.exists() else {}

    # Load game-level BB adjustment model
    import pickle
    from src.models.game_bb_adj import predict_game_bb_adjustment
    bb_adj_bundle = None
    bb_adj_path = DASHBOARD_DIR / "game_bb_adj_model.pkl"
    if bb_adj_path.exists():
        try:
            with open(bb_adj_path, "rb") as f:
                bb_adj_bundle = pickle.load(f)
            logger.info(
                "Loaded game BB adjustment model (n_train=%d)",
                bb_adj_bundle.get("n_train", 0),
            )
        except Exception as e:
            logger.warning("Failed to load BB adjustment model: %s", e)

    # Pitcher zone% for BB adjustment features
    obs_path = DASHBOARD_DIR / "pitcher_observed.parquet"
    pitcher_obs = pd.read_parquet(obs_path) if obs_path.exists() else pd.DataFrame()
    pitcher_zone_lookup: dict[int, float] = {}
    if not pitcher_obs.empty and "zone_pct" in pitcher_obs.columns:
        for _, r in pitcher_obs[["pitcher_id", "zone_pct"]].drop_duplicates("pitcher_id").iterrows():
            pitcher_zone_lookup[int(r["pitcher_id"])] = float(r["zone_pct"])

    # Umpire BB lifts for BB adjustment
    ump_bb_lift_map: dict[str, float] = {}
    try:
        from src.data.queries import get_umpire_tendencies
        ump_df = get_umpire_tendencies(seasons=list(range(2021, SEASON)), min_games=30)
        if not ump_df.empty:
            ump_bb_lift_map = dict(zip(ump_df["hp_umpire_name"], ump_df["bb_logit_lift"]))
    except Exception:
        pass

    # Hitter BB samples for lineup BB rate computation
    hitter_bb_npz: dict[str, np.ndarray] = {}
    _hbb_path = DASHBOARD_DIR / "hitter_bb_samples.npz"
    if _hbb_path.exists():
        hitter_bb_npz = dict(np.load(_hbb_path))

    # Load exit tendencies, pitch count features, TTO profiles
    tend_path = DASHBOARD_DIR / "pitcher_exit_tendencies.parquet"
    exit_tend = pd.read_parquet(tend_path) if tend_path.exists() else pd.DataFrame()

    pc_pitcher_path = DASHBOARD_DIR / "pitcher_pitch_count_features.parquet"
    pc_batter_path = DASHBOARD_DIR / "batter_pitch_count_features.parquet"
    pitcher_pc = pd.read_parquet(pc_pitcher_path) if pc_pitcher_path.exists() else pd.DataFrame()
    batter_pc = pd.read_parquet(pc_batter_path) if pc_batter_path.exists() else pd.DataFrame()

    tto_path = DASHBOARD_DIR / "tto_profiles.parquet"
    tto_profiles = pd.read_parquet(tto_path) if tto_path.exists() else pd.DataFrame()

    last_train = SEASON - 1

    # --- Load rolling form data for momentum lifts ---
    # Collect all pitcher IDs from today's schedule
    pitcher_ids = set()
    for side in ("away", "home"):
        col = f"{side}_pitcher_id"
        if col in schedule.columns:
            pitcher_ids.update(
                int(x) for x in schedule[col].dropna().unique()
            )

    # Collect all batter IDs from lineups
    batter_ids = set()
    if not lineups.empty and "batter_id" in lineups.columns:
        batter_ids.update(
            int(x) for x in lineups["batter_id"].dropna().unique()
        )

    # Batch-query rolling data and compute form lifts
    pitcher_form_lifts: dict[int, PitcherFormLifts] = {}
    batter_form_lifts: dict[int, BatterFormLifts] = {}

    if pitcher_ids:
        try:
            pit_rolling = get_rolling_form(
                list(pitcher_ids), player_role="pitcher", season=SEASON,
            )
            pitcher_form_lifts = build_pitcher_form_lifts_batch(pit_rolling)
            logger.info(
                "Computed pitcher form lifts for %d/%d pitchers",
                len(pitcher_form_lifts), len(pitcher_ids),
            )
        except Exception as e:
            logger.warning("Failed to load pitcher rolling form: %s", e)

    if batter_ids:
        try:
            bat_rolling = get_rolling_form(
                list(batter_ids), player_role="batter", season=SEASON,
            )
            hh_df = get_rolling_hard_hit(
                list(batter_ids), season=SEASON, window=15,
            )
            batter_form_lifts = build_batter_form_lifts_batch(
                bat_rolling, hard_hit_df=hh_df,
            )
            logger.info(
                "Computed batter form lifts for %d/%d batters",
                len(batter_form_lifts), len(batter_ids),
            )
        except Exception as e:
            logger.warning("Failed to load batter rolling form: %s", e)

    results = []
    for _, game in schedule.iterrows():
        gpk = game["game_pk"]

        for side in ("away", "home"):
            pid = game.get(f"{side}_pitcher_id")
            pname = game.get(f"{side}_pitcher_name", "")

            if pd.isna(pid):
                continue
            pid = int(pid)
            pid_str = str(pid)

            if pid_str not in k_samples:
                continue

            k_samp = k_samples[pid_str]
            bb_samp = bb_npz.get(pid_str)
            hr_samp = hr_npz.get(pid_str)

            # Fall back to league-average samples if missing
            rng_fb = np.random.default_rng(42 + pid)
            if bb_samp is None:
                bb_samp = rng_fb.beta(8, 90, size=len(k_samp))
            if hr_samp is None:
                hr_samp = rng_fb.beta(3, 95, size=len(k_samp))

            # Opposing lineup and matchup lifts
            opp_side = "home" if side == "away" else "away"
            opp_team_id = game.get(f"{opp_side}_team_id")

            lineup_lifts: dict[str, np.ndarray] = {
                "k": np.zeros(9), "bb": np.zeros(9), "hr": np.zeros(9),
            }
            lineup_ids: list[int] = []
            has_lineup = False

            if not lineups.empty and not arsenal_df.empty and not vuln_df.empty:
                game_lu = lineups[
                    (lineups["game_pk"] == gpk)
                    & (lineups["team_id"] == opp_team_id)
                ].sort_values("batting_order")

                if len(game_lu) >= 9:
                    lineup_ids = [int(b) for b in game_lu.head(9)["batter_id"]]
                    has_lineup = True
                    for i, bid in enumerate(lineup_ids):
                        for stat in ("k", "bb", "hr"):
                            try:
                                res = score_matchup_for_stat(
                                    stat, pid, bid,
                                    arsenal_df, vuln_df, baselines_pt,
                                )
                                v = res.get(f"matchup_{stat}_logit_lift", 0.0)
                                if isinstance(v, float) and np.isnan(v):
                                    v = 0.0
                            except Exception:
                                v = 0.0
                            lineup_lifts[stat][i] = v

            # TTO lifts
            tto_lifts: dict[str, np.ndarray] = {
                "k": np.zeros(3), "bb": np.zeros(3), "hr": np.zeros(3),
            }
            if not tto_profiles.empty:
                try:
                    tto_lifts = build_all_tto_lifts(tto_profiles, pid, last_train)
                except Exception:
                    pass

            # Pitch count features
            pitcher_adj = 0.0
            batter_adjs = np.zeros(9)
            if not pitcher_pc.empty and not batter_pc.empty and lineup_ids:
                try:
                    pitcher_adj, batter_adjs = build_pitch_count_features(
                        pitcher_features=pitcher_pc,
                        batter_features=batter_pc,
                        pitcher_id=pid,
                        batter_ids=lineup_ids,
                        season=last_train,
                    )
                except Exception:
                    pass

            # Pitcher exit tendencies + stamina offset + manager tendency
            avg_pitches = 88.0
            avg_ip = 5.28
            team_avg_p = 88.0
            if not exit_tend.empty:
                tend_row = exit_tend[exit_tend["pitcher_id"] == pid]
                if len(tend_row) > 0:
                    tr = tend_row.iloc[-1]
                    avg_pitches = float(tr.get("avg_pitches", 88.0))
                    avg_ip = float(tr.get("avg_ip", 5.28)) if pd.notna(tr.get("avg_ip")) else 5.28
                    team_avg_p = float(tr.get("team_avg_pitches", 88.0)) if pd.notna(tr.get("team_avg_pitches")) else 88.0
            # BF prior for unified exit offset
            mu_bf_val = None
            if bf_priors is not None and not bf_priors.empty:
                bf_row = bf_priors[bf_priors["pitcher_id"] == pid]
                if len(bf_row) > 0:
                    mu_bf_val = float(
                        bf_row.sort_values("season").iloc[-1]["mu_bf"]
                    )

            # Lineup patience aggregate
            lineup_ppa_agg = float(np.mean(batter_adjs))

            exit_offset = compute_exit_offset(
                mu_bf=mu_bf_val,
                pitcher_avg_ip=avg_ip,
                lineup_ppa_aggregate=lineup_ppa_agg,
            )

            # Pitcher form lift (BB% only — K% form near-zero for pitchers)
            pit_form = pitcher_form_lifts.get(pid, PitcherFormLifts())

            # XGBoost game-level BB adjustment
            xgb_bb_delta = 0.0
            if bb_adj_bundle is not None and bb_adj_bundle.get("model") is not None:
                pitcher_bb_mean = float(np.mean(bb_samp))
                zone_pct = pitcher_zone_lookup.get(pid, 0.45)
                # Umpire: look up from schedule if available
                ump_name = game.get("hp_umpire_name", "")
                ump_bb = ump_bb_lift_map.get(ump_name, 0.0) if ump_name else 0.0
                # Lineup avg BB rate from hitter BB posteriors
                lineup_avg_bb = 0.085
                if has_lineup and hitter_bb_npz:
                    lu_bbs = [
                        float(np.mean(hitter_bb_npz[str(bid)]))
                        for bid in lineup_ids if str(bid) in hitter_bb_npz
                    ]
                    if lu_bbs:
                        lineup_avg_bb = float(np.mean(lu_bbs))
                xgb_bb_delta = predict_game_bb_adjustment(
                    bb_adj_bundle,
                    pitcher_bb_rate=pitcher_bb_mean,
                    pitcher_zone_pct=zone_pct,
                    umpire_bb_lift=ump_bb,
                    lineup_avg_bb_rate=lineup_avg_bb,
                    is_home=int(side == "home"),
                    days_rest=5,
                )

            # Run PA-by-PA simulation
            try:
                sim = simulate_game(
                    pitcher_k_rate_samples=k_samp,
                    pitcher_bb_rate_samples=bb_samp,
                    pitcher_hr_rate_samples=hr_samp,
                    lineup_matchup_lifts=lineup_lifts,
                    tto_lifts=tto_lifts,
                    pitcher_ppa_adj=pitcher_adj,
                    batter_ppa_adjs=batter_adjs,
                    exit_model=exit_model,
                    pitcher_avg_pitches=avg_pitches,
                    game_context=GameContext(
                        form_bb_lift=pit_form.bb_lift,
                        xgb_bb_lift=xgb_bb_delta,
                    ),
                    exit_calibration_offset=exit_offset,
                    manager_pull_tendency=team_avg_p,
                    n_sims=10000,
                    random_seed=42 + gpk,
                )
            except Exception as e:
                logger.warning("Sim failed for pitcher %d game %d: %s", pid, gpk, e)
                continue

            summary = sim.summary()
            ip_samples = sim.ip_samples()

            # P(over) for common K lines
            k_over = sim.over_probs("k")
            p_over_dict = {}
            for _, kr in k_over.iterrows():
                line = kr["line"]
                if line in (4.5, 5.5, 6.5, 7.5):
                    p_over_dict[f"p_over_{line:.1f}".replace(".", "_")] = kr["p_over"]

            # P(over) for outs lines
            outs_over = sim.over_probs("outs")
            for _, or_ in outs_over.iterrows():
                line = or_["line"]
                if line in (14.5, 15.5, 16.5, 17.5, 18.5):
                    p_over_dict[f"p_outs_over_{line:.1f}".replace(".", "_")] = or_["p_over"]

            # Pitcher projection info
            p_row = pitcher_proj[pitcher_proj["pitcher_id"] == pid]
            proj_k_rate = float(p_row.iloc[0]["projected_k_rate"]) if not p_row.empty else float(np.mean(k_samp))
            composite = float(p_row.iloc[0].get("composite_score", 0)) if not p_row.empty else 0.0

            results.append({
                "game_pk": gpk,
                "side": side,
                "pitcher_id": pid,
                "pitcher_name": pname,
                "team_abbr": game.get(f"{side}_abbr", ""),
                "opp_abbr": game.get(f"{opp_side}_abbr", ""),
                "projected_k_rate": proj_k_rate,
                "composite_score": composite,
                "expected_k": summary["k"]["mean"],
                "k_std": summary["k"]["std"],
                "median_k": summary["k"]["median"],
                "expected_bb": summary["bb"]["mean"],
                "expected_h": summary["h"]["mean"],
                "expected_hr": summary["hr"]["mean"],
                "expected_ip": float(np.mean(ip_samples)),
                "expected_pitches": summary["pitch_count"]["mean"],
                "expected_outs": summary["outs"]["mean"],
                "expected_runs": summary["runs"]["mean"],
                "has_lineup": has_lineup,
                "avg_matchup_lift": float(np.mean(lineup_lifts["k"])),
                **p_over_dict,
            })

    return pd.DataFrame(results)


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

    # 3. Rebuild power rankings
    try:
        from src.models.team_rankings import build_power_rankings

        h_rank_path = DASHBOARD_DIR / "hitters_rankings.parquet"
        p_rank_path = DASHBOARD_DIR / "pitchers_rankings.parquet"
        h_proj_path = DASHBOARD_DIR / "hitter_projections.parquet"
        p_proj_path = DASHBOARD_DIR / "pitcher_projections.parquet"
        roster_path = DASHBOARD_DIR / "roster.parquet"

        if all(p.exists() for p in [h_rank_path, p_rank_path, roster_path]):
            power = build_power_rankings(
                elo_ratings=elo_current,
                profiles=profiles,
                current_roster=pd.read_parquet(roster_path),
                hitter_rankings=pd.read_parquet(h_rank_path),
                pitcher_rankings=pd.read_parquet(p_rank_path),
                hitter_projections=(
                    pd.read_parquet(h_proj_path) if h_proj_path.exists()
                    else pd.DataFrame()
                ),
                pitcher_projections=(
                    pd.read_parquet(p_proj_path) if p_proj_path.exists()
                    else pd.DataFrame()
                ),
            )
            if not power.empty:
                power.to_parquet(
                    DASHBOARD_DIR / "team_power_rankings.parquet",
                    index=False,
                )
                logger.info(
                    "Saved power rankings: %d teams", len(power),
                )
        else:
            logger.warning(
                "Skipping power rankings — missing required parquets",
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
    bb_samples = update_rate_samples_step(
        "bb", trials_col="batters_faced", successes_col="walks",
        league_avg=0.08,
    )
    if bb_samples:
        np.savez_compressed(DASHBOARD_DIR / "pitcher_bb_samples.npz", **bb_samples)
        logger.info("Saved updated BB%% samples for %d pitchers", len(bb_samples))

    hr_samples = update_rate_samples_step(
        "hr", trials_col="batters_faced", successes_col="hr",
        league_avg=0.03,
    )
    if hr_samples:
        np.savez_compressed(DASHBOARD_DIR / "pitcher_hr_samples.npz", **hr_samples)
        logger.info("Saved updated HR/BF samples for %d pitchers", len(hr_samples))

    # Step 2c: Update hitter K%, BB%, HR samples
    logger.info("Step 2c: Updating hitter K%%, BB%%, HR samples...")
    h_k_samples = update_hitter_rate_samples_step(
        "k", trials_col="pa", successes_col="strikeouts",
        league_avg=0.226,
    )
    if h_k_samples:
        np.savez_compressed(DASHBOARD_DIR / "hitter_k_samples.npz", **h_k_samples)
        logger.info("Saved updated hitter K%% samples for %d batters", len(h_k_samples))

    h_bb_samples = update_hitter_rate_samples_step(
        "bb", trials_col="pa", successes_col="walks",
        league_avg=0.082,
    )
    if h_bb_samples:
        np.savez_compressed(DASHBOARD_DIR / "hitter_bb_samples.npz", **h_bb_samples)
        logger.info("Saved updated hitter BB%% samples for %d batters", len(h_bb_samples))

    h_hr_samples = update_hitter_rate_samples_step(
        "hr", trials_col="pa", successes_col="hr",
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

    # Step 4: Fetch schedule and simulate today's games
    if not args.skip_schedule:
        logger.info("Step 4: Fetching schedule for %s...", game_date)
        schedule, lineups = fetch_schedule_step(game_date)

        if not schedule.empty:
            schedule.to_parquet(DASHBOARD_DIR / "todays_games.parquet", index=False)

            if not lineups.empty:
                lineups.to_parquet(DASHBOARD_DIR / "todays_lineups.parquet", index=False)

            # Simulate K props for each starter
            bf_priors_path = DASHBOARD_DIR / "bf_priors.parquet"
            bf_priors = pd.read_parquet(bf_priors_path) if bf_priors_path.exists() else pd.DataFrame()

            logger.info("Simulating K props for today's starters...")
            sim_results = simulate_todays_games(
                schedule, lineups, k_samples, p_updated, bf_priors,
            )
            if not sim_results.empty:
                sim_results.to_parquet(DASHBOARD_DIR / "todays_sims.parquet", index=False)
                logger.info("Saved K simulations for %d pitcher appearances", len(sim_results))
        else:
            logger.info("No games today — skipping simulation")
    else:
        logger.info("Step 4: Skipped (--skip-schedule)")

    # Step 5: Update 2026 season stats (traditional stats for leaders page)
    logger.info("Step 5: Updating 2026 season stats...")
    h_trad, p_trad = update_season_stats_step()

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
