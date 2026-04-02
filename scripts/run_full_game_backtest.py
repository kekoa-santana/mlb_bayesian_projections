#!/usr/bin/env python
"""
Walk-forward backtest for the full-game (both-teams) simulator.

Evaluates win probabilities, run totals, and run lines against
actual game outcomes using simulate_full_game_both_teams().

Usage
-----
    python scripts/run_full_game_backtest.py          # full quality
    python scripts/run_full_game_backtest.py --quick   # fast iteration
    python scripts/run_full_game_backtest.py --single-fold 2025
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql
from src.data.feature_eng import (
    build_multi_season_hitter_data,
    build_multi_season_pitcher_data,
    get_cached_game_lineups,
    get_cached_pitcher_game_logs,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.data.league_baselines import get_baselines_dict
from src.data.queries import get_team_bullpen_rates
from src.models.bf_model import compute_pitcher_bf_priors
from src.models.game_sim.lineup_simulator import simulate_full_game_both_teams
from src.models.hitter_model import (
    extract_rate_samples as extract_hitter_rate_samples,
    fit_hitter_model,
    prepare_hitter_data,
)
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
    SIM_LEAGUE_BB_RATE,
    SIM_LEAGUE_HR_RATE,
    SIM_LEAGUE_K_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _get_game_scores(season: int) -> pd.DataFrame:
    """Get home/away scores for all regular-season games."""
    return read_sql(f"""
        WITH team_scores AS (
            SELECT fpg.game_pk, fpg.team_id,
                   SUM(COALESCE(fpg.bat_r, 0)) AS team_runs
            FROM production.fact_player_game_mlb fpg
            JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
            WHERE fpg.season = {int(season)} AND dg.game_type = 'R'
            GROUP BY fpg.game_pk, fpg.team_id
        )
        SELECT dg.game_pk, dg.game_date,
               dg.home_team_id, dg.away_team_id,
               COALESCE(h.team_runs, 0) AS home_runs,
               COALESCE(a.team_runs, 0) AS away_runs
        FROM production.dim_game dg
        LEFT JOIN team_scores h
            ON dg.game_pk = h.game_pk AND dg.home_team_id = h.team_id
        LEFT JOIN team_scores a
            ON dg.game_pk = a.game_pk AND dg.away_team_id = a.team_id
        WHERE dg.season = {int(season)} AND dg.game_type = 'R'
    """)


def _get_game_starters(season: int) -> pd.DataFrame:
    """Get starting pitchers per game (home + away)."""
    return read_sql(f"""
        SELECT fpg.game_pk, fpg.player_id AS pitcher_id,
               fpg.team_id,
               dg.home_team_id, dg.away_team_id
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = TRUE
          AND fpg.season = {int(season)}
          AND dg.game_type = 'R'
          AND fpg.pit_bf >= 3
    """)


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def run_one_fold(
    train_seasons: list[int],
    test_season: int,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_sims: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run full-game backtest for one fold.

    Returns
    -------
    pd.DataFrame
        One row per game with predicted and actual outcomes.
    """
    last_train = max(train_seasons)
    logger.info("Full-game backtest: train=%s, test=%d", train_seasons, test_season)

    # ---------------------------------------------------------------
    # 1. Fit hitter models (K%, BB%)
    # ---------------------------------------------------------------
    logger.info("Fitting hitter models...")
    df_hitter = build_multi_season_hitter_data(train_seasons, min_pa=50)

    data_hk = prepare_hitter_data(df_hitter, "k_rate")
    _, trace_hk = fit_hitter_model(
        data_hk, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )
    batter_k_post: dict[int, np.ndarray] = {}
    for bid in data_hk["df"][data_hk["df"]["season"] == last_train]["batter_id"].unique():
        try:
            batter_k_post[int(bid)] = extract_hitter_rate_samples(
                trace_hk, data_hk, bid, last_train,
                project_forward=True, random_seed=random_seed,
            )
        except ValueError:
            continue
    logger.info("Batter K posteriors: %d", len(batter_k_post))

    data_hbb = prepare_hitter_data(df_hitter, "bb_rate")
    _, trace_hbb = fit_hitter_model(
        data_hbb, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 1,
    )
    batter_bb_post: dict[int, np.ndarray] = {}
    for bid in data_hbb["df"][data_hbb["df"]["season"] == last_train]["batter_id"].unique():
        try:
            batter_bb_post[int(bid)] = extract_hitter_rate_samples(
                trace_hbb, data_hbb, bid, last_train,
                project_forward=True, random_seed=random_seed + 1,
            )
        except ValueError:
            continue
    logger.info("Batter BB posteriors: %d", len(batter_bb_post))

    # Batter HR: pseudo-posteriors from observed rates
    batter_hr_post: dict[int, np.ndarray] = {}
    hr_data = df_hitter[df_hitter["season"].isin(train_seasons)].copy()
    for bid, grp in hr_data.groupby("batter_id"):
        total_hr = grp["hr"].sum() if "hr" in grp.columns else 0
        total_pa = grp["pa"].sum()
        if total_pa >= 50:
            rate = total_hr / total_pa
            rng_hr = np.random.default_rng(random_seed + int(bid) % 10000)
            std = max(0.005, rate * 0.15)
            samples = rng_hr.normal(rate, std, size=2000)
            batter_hr_post[int(bid)] = np.clip(samples, 0.001, 0.10)
    logger.info("Batter HR posteriors: %d", len(batter_hr_post))

    valid_batters = set(batter_k_post) & set(batter_bb_post) & set(batter_hr_post)

    # ---------------------------------------------------------------
    # 2. Fit pitcher models (K%, BB%, HR/BF)
    # ---------------------------------------------------------------
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
    logger.info("Pitcher posteriors: %d (K=%d, BB=%d, HR=%d)",
                len(valid_pitchers), len(pitcher_k_post),
                len(pitcher_bb_post), len(pitcher_hr_post))

    # ---------------------------------------------------------------
    # 3. BF priors, bullpen rates, matchup data
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # 4. Load test season data
    # ---------------------------------------------------------------
    logger.info("Loading test season %d...", test_season)
    game_scores = _get_game_scores(test_season)
    game_starters = _get_game_starters(test_season)
    game_lineups = get_cached_game_lineups(test_season)

    # Build per-game starter lookup: {game_pk: {home_pid, away_pid}}
    starter_map: dict[int, dict[str, int]] = {}
    for _, row in game_starters.iterrows():
        gpk = int(row["game_pk"])
        pid = int(row["pitcher_id"])
        tid = int(row["team_id"])
        if gpk not in starter_map:
            starter_map[gpk] = {}
        if tid == int(row["home_team_id"]):
            starter_map[gpk]["home"] = pid
        elif tid == int(row["away_team_id"]):
            starter_map[gpk]["away"] = pid

    logger.info("Games with scores: %d, with starters: %d",
                len(game_scores), len(starter_map))

    # ---------------------------------------------------------------
    # 5. Simulate each game
    # ---------------------------------------------------------------
    results = []
    n_skipped = 0

    for _, game in game_scores.iterrows():
        gpk = int(game["game_pk"])
        actual_home = int(game["home_runs"])
        actual_away = int(game["away_runs"])
        home_tid = int(game["home_team_id"])
        away_tid = int(game["away_team_id"])

        # Get starters
        starters = starter_map.get(gpk)
        if not starters or "home" not in starters or "away" not in starters:
            n_skipped += 1
            continue

        home_sp = starters["home"]
        away_sp = starters["away"]

        if home_sp not in valid_pitchers or away_sp not in valid_pitchers:
            n_skipped += 1
            continue

        # Get lineups
        glu = game_lineups[game_lineups["game_pk"] == gpk]
        home_lu = glu[glu["team_id"] == home_tid].sort_values("batting_order")
        away_lu = glu[glu["team_id"] == away_tid].sort_values("batting_order")
        home_bids = home_lu["player_id"].tolist()[:9]
        away_bids = away_lu["player_id"].tolist()[:9]

        if len(home_bids) < 9 or len(away_bids) < 9:
            n_skipped += 1
            continue

        # Check all batters have posteriors
        if not all(b in valid_batters for b in home_bids + away_bids):
            n_skipped += 1
            continue

        # Batter posterior samples
        home_k_samples = [batter_k_post[b] for b in home_bids]
        home_bb_samples = [batter_bb_post[b] for b in home_bids]
        home_hr_samples = [batter_hr_post[b] for b in home_bids]
        away_k_samples = [batter_k_post[b] for b in away_bids]
        away_bb_samples = [batter_bb_post[b] for b in away_bids]
        away_hr_samples = [batter_hr_post[b] for b in away_bids]

        # Pitcher rates (point estimates for lineup sim)
        home_sp_k = float(np.mean(pitcher_k_post[home_sp]))
        home_sp_bb = float(np.mean(pitcher_bb_post[home_sp]))
        home_sp_hr = float(np.mean(pitcher_hr_post[home_sp]))
        away_sp_k = float(np.mean(pitcher_k_post[away_sp]))
        away_sp_bb = float(np.mean(pitcher_bb_post[away_sp]))
        away_sp_hr = float(np.mean(pitcher_hr_post[away_sp]))

        # BF priors
        def _get_bf(pid):
            row = bf_priors[
                (bf_priors["pitcher_id"] == pid)
                & (bf_priors["season"] == last_train)
            ]
            if len(row) > 0:
                return float(row.iloc[0]["mu_bf"]), float(row.iloc[0]["sigma_bf"])
            return 22.0, 4.5

        home_bf_mu, home_bf_sig = _get_bf(home_sp)
        away_bf_mu, away_bf_sig = _get_bf(away_sp)

        # Bullpen rates per team
        def _get_bp(tid):
            row = bullpen_latest[bullpen_latest["team_id"] == tid]
            if len(row) > 0:
                return (
                    float(row.iloc[0].get("k_rate", BULLPEN_K_RATE)),
                    float(row.iloc[0].get("bb_rate", BULLPEN_BB_RATE)),
                    float(row.iloc[0].get("hr_rate", BULLPEN_HR_RATE)),
                )
            return BULLPEN_K_RATE, BULLPEN_BB_RATE, BULLPEN_HR_RATE

        home_bp_k, home_bp_bb, home_bp_hr = _get_bp(home_tid)
        away_bp_k, away_bp_bb, away_bp_hr = _get_bp(away_tid)

        # Matchup lifts: away batters vs home starter, home batters vs away starter
        def _lineup_matchup(pitcher_id, batter_ids):
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

        away_mk, away_mbb, away_mhr = _lineup_matchup(home_sp, away_bids)
        home_mk, home_mbb, home_mhr = _lineup_matchup(away_sp, home_bids)

        # Run full-game simulation
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

        pred_home_runs = float(np.mean(result.home_runs))
        pred_away_runs = float(np.mean(result.away_runs))
        pred_total = float(np.mean(result.away_runs + result.home_runs))

        results.append({
            "game_pk": gpk,
            "test_season": test_season,
            "home_team_id": home_tid,
            "away_team_id": away_tid,
            "home_sp": home_sp,
            "away_sp": away_sp,
            # Predictions
            "pred_home_win_prob": result.home_win_prob,
            "pred_home_runs": pred_home_runs,
            "pred_away_runs": pred_away_runs,
            "pred_total_runs": pred_total,
            "pred_home_margin": pred_home_runs - pred_away_runs,
            # Over/under probs
            "p_over_7_5": float(np.mean(result.away_runs + result.home_runs > 7.5)),
            "p_over_8_5": float(np.mean(result.away_runs + result.home_runs > 8.5)),
            "p_over_9_5": float(np.mean(result.away_runs + result.home_runs > 9.5)),
            # Actuals
            "actual_home_runs": actual_home,
            "actual_away_runs": actual_away,
            "actual_total_runs": actual_home + actual_away,
            "actual_home_win": int(actual_home > actual_away),
        })

        if len(results) % 100 == 0:
            logger.info("  %d games simulated...", len(results))

    logger.info("Fold complete: %d games, %d skipped", len(results), n_skipped)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute full-game validation metrics."""
    n = len(df)
    if n == 0:
        return {}

    metrics: dict = {"n_games": n}

    # --- Win probability ---
    home_win = df["actual_home_win"].values
    home_prob = df["pred_home_win_prob"].values

    # Brier score
    metrics["win_brier"] = float(brier_score_loss(home_win, home_prob))

    # Log loss
    eps = 1e-6
    clipped = np.clip(home_prob, eps, 1 - eps)
    ll = -(home_win * np.log(clipped) + (1 - home_win) * np.log(1 - clipped))
    metrics["win_log_loss"] = float(np.mean(ll))

    # Accuracy at 50% threshold
    pred_home = (home_prob > 0.5).astype(int)
    metrics["win_accuracy"] = float(np.mean(pred_home == home_win))

    # Calibration by bin
    bins = [0.0, 0.35, 0.45, 0.50, 0.55, 0.65, 1.0]
    for i in range(len(bins) - 1):
        mask = (home_prob >= bins[i]) & (home_prob < bins[i + 1])
        if mask.sum() >= 20:
            metrics[f"win_cal_{bins[i]:.2f}_{bins[i+1]:.2f}"] = float(
                home_win[mask].mean()
            )

    # --- Run totals ---
    total_err = df["pred_total_runs"] - df["actual_total_runs"]
    metrics["total_runs_rmse"] = float(np.sqrt((total_err ** 2).mean()))
    metrics["total_runs_mae"] = float(total_err.abs().mean())
    metrics["total_runs_bias"] = float(total_err.mean())
    metrics["total_runs_corr"] = float(
        df["pred_total_runs"].corr(df["actual_total_runs"])
    )

    # Over/under calibration
    for line, col in [(7.5, "p_over_7_5"), (8.5, "p_over_8_5"), (9.5, "p_over_9_5")]:
        actual_over = (df["actual_total_runs"] > line).astype(int)
        pred_over = df[col]
        metrics[f"ou_{line}_pred_rate"] = float(pred_over.mean())
        metrics[f"ou_{line}_actual_rate"] = float(actual_over.mean())
        metrics[f"ou_{line}_brier"] = float(brier_score_loss(actual_over, pred_over))

    # --- Margin / run line ---
    margin_err = df["pred_home_margin"] - (df["actual_home_runs"] - df["actual_away_runs"])
    metrics["margin_rmse"] = float(np.sqrt((margin_err ** 2).mean()))
    metrics["margin_mae"] = float(margin_err.abs().mean())
    metrics["margin_corr"] = float(
        df["pred_home_margin"].corr(df["actual_home_runs"] - df["actual_away_runs"])
    )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full-game walk-forward backtest")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer MCMC draws and sims")
    parser.add_argument("--single-fold", type=int, default=None,
                        help="Run one test season only")
    args = parser.parse_args()

    if args.quick:
        draws, tune, chains, n_sims = 500, 250, 2, 2000
    else:
        draws, tune, chains, n_sims = 1000, 500, 2, 5000

    if args.single_fold:
        test_season = args.single_fold
        folds = [(list(range(2020, test_season)), test_season)]
    else:
        folds = [
            (list(range(2020, 2023)), 2023),
            (list(range(2020, 2024)), 2024),
            (list(range(2020, 2025)), 2025),
        ]

    logger.info("=" * 70)
    logger.info("FULL-GAME BACKTEST")
    logger.info("  MCMC: draws=%d, tune=%d, chains=%d", draws, tune, chains)
    logger.info("  MC sims per game: %d", n_sims)
    logger.info("  Folds: %d", len(folds))
    logger.info("=" * 70)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    all_preds = []
    all_metrics = []

    for train_seasons, test_season in folds:
        preds = run_one_fold(
            train_seasons, test_season,
            draws=draws, tune=tune, chains=chains,
            n_sims=n_sims, random_seed=42,
        )
        all_preds.append(preds)

        m = compute_metrics(preds)
        m["test_season"] = test_season
        all_metrics.append(m)

    # Save
    predictions = pd.concat(all_preds, ignore_index=True)
    predictions.to_csv(out_dir / "full_game_backtest_predictions.csv", index=False)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "full_game_backtest_summary.csv", index=False)

    # Print results
    _print_results(metrics_df)


def _print_results(metrics: pd.DataFrame) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("FULL-GAME BACKTEST RESULTS")
    print("=" * 70)

    for _, row in metrics.iterrows():
        s = int(row["test_season"])
        n = int(row["n_games"])
        print(f"\nTest Season {s} ({n} games):")
        print(f"  Win Prob:  Brier={row['win_brier']:.4f}  "
              f"LogLoss={row['win_log_loss']:.4f}  "
              f"Accuracy={row['win_accuracy']:.3f}")
        print(f"  Runs:      RMSE={row['total_runs_rmse']:.3f}  "
              f"MAE={row['total_runs_mae']:.3f}  "
              f"Bias={row['total_runs_bias']:+.3f}  "
              f"Corr={row['total_runs_corr']:.3f}")
        print(f"  Margin:    RMSE={row['margin_rmse']:.3f}  "
              f"MAE={row['margin_mae']:.3f}  "
              f"Corr={row['margin_corr']:.3f}")

        for line in [7.5, 8.5, 9.5]:
            pred = row.get(f"ou_{line}_pred_rate", float("nan"))
            actual = row.get(f"ou_{line}_actual_rate", float("nan"))
            brier = row.get(f"ou_{line}_brier", float("nan"))
            print(f"  O/U {line}:   pred={pred:.3f}  actual={actual:.3f}  "
                  f"Brier={brier:.4f}")

        # Calibration bins
        cal_cols = [c for c in row.index if c.startswith("win_cal_")]
        if cal_cols:
            print("  Win calibration:")
            for c in sorted(cal_cols):
                lo, hi = c.replace("win_cal_", "").split("_")
                print(f"    P({lo}-{hi}): actual_win_rate={row[c]:.3f}")

    # Averages
    print("\n--- Averages ---")
    print(f"  Win Brier:    {metrics['win_brier'].mean():.4f}")
    print(f"  Win Accuracy: {metrics['win_accuracy'].mean():.3f}")
    print(f"  Runs RMSE:    {metrics['total_runs_rmse'].mean():.3f}")
    print(f"  Margin Corr:  {metrics['margin_corr'].mean():.3f}")

    # Baselines
    print("\n--- Baselines ---")
    print("  Coin flip Brier: 0.2500")
    print("  Home-field-only accuracy: ~0.540")
    print()


if __name__ == "__main__":
    main()
