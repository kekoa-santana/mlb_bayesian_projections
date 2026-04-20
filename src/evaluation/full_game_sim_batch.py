"""
Fit hitter/pitcher models once, then simulate many games (full-game lineup sim).

Shared by ``scripts/run_full_game_backtest.py`` and
``scripts/simulate_season_games_to_date.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.feature_eng import (
    build_multi_season_pitcher_data,
    get_cached_pitcher_game_logs,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.data.league_baselines import get_baselines_dict
from src.data.queries import get_team_bullpen_rates
from src.models.bf_model import compute_pitcher_bf_priors
from src.models.game_sim.lineup_simulator import simulate_full_game_both_teams
from src.models.matchup import score_matchup_for_stat
from src.models.pitcher_model import (
    extract_rate_samples as extract_pitcher_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)
from src.utils.constants import (
    BULLPEN_BB_RATE,
    BULLPEN_HR_RATE,
    BULLPEN_K_RATE,
)

logger = logging.getLogger(__name__)


@dataclass
class FullGameSimBundle:
    """Posteriors + lookup tables for :func:`simulate_full_games`."""

    last_train_season: int
    batter_k_post: dict[int, np.ndarray]
    batter_bb_post: dict[int, np.ndarray]
    batter_hr_post: dict[int, np.ndarray]
    valid_batters: set[int]
    pitcher_k_post: dict[int, np.ndarray]
    pitcher_bb_post: dict[int, np.ndarray]
    pitcher_hr_post: dict[int, np.ndarray]
    valid_pitchers: set[int]
    bf_priors: pd.DataFrame
    bullpen_latest: pd.DataFrame
    pitcher_arsenal: pd.DataFrame
    hitter_vuln: pd.DataFrame
    baselines_pt: dict[str, dict[str, float]]


def fit_full_game_sim_bundle(
    train_seasons: list[int],
    *,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
) -> FullGameSimBundle:
    """Fit MCMC hitter/pitcher models and load supporting tables."""
    last_train = max(train_seasons)
    logger.info("Fitting full-game bundle: train=%s, last_train=%d", train_seasons, last_train)

    from src.evaluation.runner import fit_hitter_posteriors
    hitter_posts = fit_hitter_posteriors(
        train_seasons, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )
    batter_k_post = hitter_posts["k"]
    batter_bb_post = hitter_posts["bb"]
    batter_hr_post = hitter_posts["hr"]

    valid_batters = set(batter_k_post) & set(batter_bb_post) & set(batter_hr_post)
    logger.info(
        "Batter posteriors: K=%d BB=%d HR=%d valid=%d",
        len(batter_k_post), len(batter_bb_post), len(batter_hr_post), len(valid_batters),
    )

    logger.info("Fitting pitcher models...")
    df_pitcher = build_multi_season_pitcher_data(train_seasons, min_bf=10)

    data_pk = prepare_pitcher_data(df_pitcher, "k_rate")
    _, trace_pk = fit_pitcher_model(
        data_pk, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 10,
    )
    pitcher_k_post: dict[int, np.ndarray] = {}
    for pid in data_pk["df"][data_pk["df"]["season"] == last_train]["pitcher_id"].unique():
        try:
            pitcher_k_post[int(pid)] = extract_pitcher_rate_samples(
                trace_pk, data_pk, pid, last_train,
                project_forward=True, random_seed=random_seed + 10,
            )
        except ValueError:
            continue

    data_pbb = prepare_pitcher_data(df_pitcher, "bb_rate")
    _, trace_pbb = fit_pitcher_model(
        data_pbb, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 11,
    )
    pitcher_bb_post: dict[int, np.ndarray] = {}
    for pid in data_pbb["df"][data_pbb["df"]["season"] == last_train]["pitcher_id"].unique():
        try:
            pitcher_bb_post[int(pid)] = extract_pitcher_rate_samples(
                trace_pbb, data_pbb, pid, last_train,
                project_forward=True, random_seed=random_seed + 11,
            )
        except ValueError:
            continue

    data_phr = prepare_pitcher_data(df_pitcher, "hr_per_bf")
    _, trace_phr = fit_pitcher_model(
        data_phr, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 12,
    )
    pitcher_hr_post: dict[int, np.ndarray] = {}
    for pid in data_phr["df"][data_phr["df"]["season"] == last_train]["pitcher_id"].unique():
        try:
            pitcher_hr_post[int(pid)] = extract_pitcher_rate_samples(
                trace_phr, data_phr, pid, last_train,
                project_forward=True, random_seed=random_seed + 12,
            )
        except ValueError:
            continue

    valid_pitchers = set(pitcher_k_post) & set(pitcher_bb_post) & set(pitcher_hr_post)
    logger.info(
        "Pitcher posteriors: %d (K=%d, BB=%d, HR=%d)",
        len(valid_pitchers), len(pitcher_k_post), len(pitcher_bb_post), len(pitcher_hr_post),
    )

    logger.info("Loading supporting data...")
    game_logs_frames = [get_cached_pitcher_game_logs(s) for s in train_seasons]
    all_game_logs = pd.concat(game_logs_frames, ignore_index=True)
    bf_priors = compute_pitcher_bf_priors(all_game_logs)

    bullpen_rates = get_team_bullpen_rates(train_seasons)
    bullpen_latest = (
        bullpen_rates.sort_values("season")
        .groupby("team_id").last().reset_index()
    )

    pitcher_arsenal = get_pitcher_arsenal(last_train)
    hitter_vuln = get_hitter_vulnerability(last_train)
    baselines_pt = get_baselines_dict(seasons=train_seasons, recency_weights="equal")

    return FullGameSimBundle(
        last_train_season=last_train,
        batter_k_post=batter_k_post,
        batter_bb_post=batter_bb_post,
        batter_hr_post=batter_hr_post,
        valid_batters=valid_batters,
        pitcher_k_post=pitcher_k_post,
        pitcher_bb_post=pitcher_bb_post,
        pitcher_hr_post=pitcher_hr_post,
        valid_pitchers=valid_pitchers,
        bf_priors=bf_priors,
        bullpen_latest=bullpen_latest,
        pitcher_arsenal=pitcher_arsenal,
        hitter_vuln=hitter_vuln,
        baselines_pt=baselines_pt,
    )


def simulate_full_games(
    bundle: FullGameSimBundle,
    games: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    n_sims: int = 5000,
    random_seed: int = 42,
    include_run_quantiles: bool = True,
) -> pd.DataFrame:
    """Simulate each row of *games* and compare to actual box-score runs.

    Parameters
    ----------
    games
        Columns: ``game_pk``, ``home_team_id``, ``away_team_id``,
        ``home_runs``, ``away_runs``, ``home_sp``, ``away_sp`` (starter
        pitcher IDs). Optional: ``game_date``.
    lineups
        From :func:`~src.data.feature_eng.get_cached_game_lineups`.
    include_run_quantiles
        If True, add 10th/50th/90th percentiles of simulated runs per side.
    """
    last_train = bundle.last_train_season
    results: list[dict] = []
    n_skipped = 0

    batter_k_post = bundle.batter_k_post
    batter_bb_post = bundle.batter_bb_post
    batter_hr_post = bundle.batter_hr_post
    valid_batters = bundle.valid_batters
    pitcher_k_post = bundle.pitcher_k_post
    pitcher_bb_post = bundle.pitcher_bb_post
    pitcher_hr_post = bundle.pitcher_hr_post
    valid_pitchers = bundle.valid_pitchers
    bf_priors = bundle.bf_priors
    bullpen_latest = bundle.bullpen_latest
    pitcher_arsenal = bundle.pitcher_arsenal
    hitter_vuln = bundle.hitter_vuln
    baselines_pt = bundle.baselines_pt

    def _get_bf(pid: int) -> tuple[float, float]:
        row = bf_priors[
            (bf_priors["pitcher_id"] == pid)
            & (bf_priors["season"] == last_train)
        ]
        if len(row) > 0:
            return float(row.iloc[0]["mu_bf"]), float(row.iloc[0]["sigma_bf"])
        return 22.0, 4.5

    def _get_bp(tid: int) -> tuple[float, float, float]:
        row = bullpen_latest[bullpen_latest["team_id"] == tid]
        if len(row) > 0:
            return (
                float(row.iloc[0].get("k_rate", BULLPEN_K_RATE)),
                float(row.iloc[0].get("bb_rate", BULLPEN_BB_RATE)),
                float(row.iloc[0].get("hr_rate", BULLPEN_HR_RATE)),
            )
        return BULLPEN_K_RATE, BULLPEN_BB_RATE, BULLPEN_HR_RATE

    def _lineup_matchup(pitcher_id: int, batter_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_lifts, bb_lifts, hr_lifts = [], [], []
        for bid in batter_ids:
            for stat, lst in [("k", k_lifts), ("bb", bb_lifts), ("hr", hr_lifts)]:
                try:
                    res = score_matchup_for_stat(
                        stat, pitcher_id, bid,
                        pitcher_arsenal, hitter_vuln, baselines_pt,
                    )
                    v = res.get(f"matchup_{stat}_logit_lift", 0.0)
                    if isinstance(v, float) and np.isnan(v):
                        v = 0.0
                except Exception:
                    v = 0.0
                lst.append(v)
        return np.array(k_lifts), np.array(bb_lifts), np.array(hr_lifts)

    for _, game in games.iterrows():
        gpk = int(game["game_pk"])
        actual_home = int(game["home_runs"])
        actual_away = int(game["away_runs"])
        home_tid = int(game["home_team_id"])
        away_tid = int(game["away_team_id"])
        home_sp = int(game["home_sp"])
        away_sp = int(game["away_sp"])

        if home_sp not in valid_pitchers or away_sp not in valid_pitchers:
            n_skipped += 1
            continue

        glu = lineups[lineups["game_pk"] == gpk]
        home_lu = glu[glu["team_id"] == home_tid].sort_values("batting_order")
        away_lu = glu[glu["team_id"] == away_tid].sort_values("batting_order")
        home_bids = [int(x) for x in home_lu["player_id"].tolist()[:9]]
        away_bids = [int(x) for x in away_lu["player_id"].tolist()[:9]]

        if len(home_bids) < 9 or len(away_bids) < 9:
            n_skipped += 1
            continue

        if not all(b in valid_batters for b in home_bids + away_bids):
            n_skipped += 1
            continue

        home_k_samples = [batter_k_post[b] for b in home_bids]
        home_bb_samples = [batter_bb_post[b] for b in home_bids]
        home_hr_samples = [batter_hr_post[b] for b in home_bids]
        away_k_samples = [batter_k_post[b] for b in away_bids]
        away_bb_samples = [batter_bb_post[b] for b in away_bids]
        away_hr_samples = [batter_hr_post[b] for b in away_bids]

        home_sp_k = np.asarray(pitcher_k_post[home_sp], dtype=np.float64)
        home_sp_bb = np.asarray(pitcher_bb_post[home_sp], dtype=np.float64)
        home_sp_hr = np.asarray(pitcher_hr_post[home_sp], dtype=np.float64)
        away_sp_k = np.asarray(pitcher_k_post[away_sp], dtype=np.float64)
        away_sp_bb = np.asarray(pitcher_bb_post[away_sp], dtype=np.float64)
        away_sp_hr = np.asarray(pitcher_hr_post[away_sp], dtype=np.float64)

        home_bf_mu, home_bf_sig = _get_bf(home_sp)
        away_bf_mu, away_bf_sig = _get_bf(away_sp)

        home_bp_k, home_bp_bb, home_bp_hr = _get_bp(home_tid)
        away_bp_k, away_bp_bb, away_bp_hr = _get_bp(away_tid)

        away_mk, away_mbb, away_mhr = _lineup_matchup(home_sp, away_bids)
        home_mk, home_mbb, home_mhr = _lineup_matchup(away_sp, home_bids)

        try:
            result = simulate_full_game_both_teams(
                away_batter_k_rate_samples=away_k_samples,
                away_batter_bb_rate_samples=away_bb_samples,
                away_batter_hr_rate_samples=away_hr_samples,
                home_starter_k_rate=home_sp_k,
                home_starter_bb_rate=home_sp_bb,
                home_starter_hr_rate=home_sp_hr,
                home_starter_bf_mu=home_bf_mu,
                home_starter_bf_sigma=home_bf_sig,
                home_batter_k_rate_samples=home_k_samples,
                home_batter_bb_rate_samples=home_bb_samples,
                home_batter_hr_rate_samples=home_hr_samples,
                away_starter_k_rate=away_sp_k,
                away_starter_bb_rate=away_sp_bb,
                away_starter_hr_rate=away_sp_hr,
                away_starter_bf_mu=away_bf_mu,
                away_starter_bf_sigma=away_bf_sig,
                away_matchup_k_lifts=away_mk,
                away_matchup_bb_lifts=away_mbb,
                away_matchup_hr_lifts=away_mhr,
                home_matchup_k_lifts=home_mk,
                home_matchup_bb_lifts=home_mbb,
                home_matchup_hr_lifts=home_mhr,
                home_bullpen_k_rate=home_bp_k,
                home_bullpen_bb_rate=home_bp_bb,
                home_bullpen_hr_rate=home_bp_hr,
                away_bullpen_k_rate=away_bp_k,
                away_bullpen_bb_rate=away_bp_bb,
                away_bullpen_hr_rate=away_bp_hr,
                n_sims=n_sims,
                random_seed=random_seed + gpk % 10000,
            )
        except Exception as e:
            logger.warning("Sim failed game %d: %s", gpk, e)
            n_skipped += 1
            continue

        row: dict = {
            "game_pk": gpk,
            "home_team_id": home_tid,
            "away_team_id": away_tid,
            "home_sp": home_sp,
            "away_sp": away_sp,
            "pred_home_win_prob": result.home_win_prob,
            "pred_home_runs_mean": float(np.mean(result.home_runs)),
            "pred_away_runs_mean": float(np.mean(result.away_runs)),
            "pred_total_runs_mean": float(np.mean(result.away_runs + result.home_runs)),
            "pred_home_margin_mean": float(
                np.mean(result.home_runs.astype(np.float64) - result.away_runs.astype(np.float64))
            ),
            "p_over_7_5": float(np.mean(result.away_runs + result.home_runs > 7.5)),
            "p_over_8_5": float(np.mean(result.away_runs + result.home_runs > 8.5)),
            "p_over_9_5": float(np.mean(result.away_runs + result.home_runs > 9.5)),
            "actual_home_runs": actual_home,
            "actual_away_runs": actual_away,
            "actual_total_runs": actual_home + actual_away,
            "actual_home_win": int(actual_home > actual_away),
        }
        if "game_date" in game.index:
            gd = game["game_date"]
            if pd.notna(gd):
                row["game_date"] = gd

        if include_run_quantiles:
            row["pred_home_runs_p10"] = float(np.percentile(result.home_runs, 10))
            row["pred_home_runs_p50"] = float(np.percentile(result.home_runs, 50))
            row["pred_home_runs_p90"] = float(np.percentile(result.home_runs, 90))
            row["pred_away_runs_p10"] = float(np.percentile(result.away_runs, 10))
            row["pred_away_runs_p50"] = float(np.percentile(result.away_runs, 50))
            row["pred_away_runs_p90"] = float(np.percentile(result.away_runs, 90))
            row["pred_total_runs_p50"] = float(
                np.percentile(result.away_runs + result.home_runs, 50)
            )

        results.append(row)

        if len(results) % 100 == 0:
            logger.info("  %d games simulated...", len(results))

    logger.info("Simulate complete: %d games, %d skipped", len(results), n_skipped)
    return pd.DataFrame(results)
