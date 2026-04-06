"""
Walk-forward backtest for the sequential PA game simulator.

Evaluates the full pipeline: Layer 1 posteriors (K%, BB%, HR%) →
Layer 2 matchup lifts → PA-by-PA simulator → game-level stat distributions.

Compares against the current Layer 3 (game_k_model.py) on K prediction
and provides new metrics for BB, H, HR, pitches, IP.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.evaluation.metrics import compute_log_loss, compute_sharpness

from src.data.db import read_sql
from src.data.feature_eng import (
    build_multi_season_pitcher_data,
    build_multi_season_pitcher_k_data,
    get_cached_game_lineups,
    get_cached_pitcher_game_logs,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.data.queries import (
    get_batter_pitch_count_features,
    get_exit_model_training_data,
    get_pitcher_exit_tendencies,
    get_pitcher_pitch_count_features,
    get_tto_adjustment_profiles,
)
from src.evaluation.context_lifts import (
    build_umpire_logit_lifts,
    build_weather_logit_lifts,
)
from src.models.game_sim.exit_model import ExitModel
from src.models.game_sim.pa_outcome_model import GameContext
from src.models.game_sim.simulator import (
    simulate_game,
    compute_stamina_offset,
    compute_exit_offset,
)
from src.models.game_sim.tto_model import build_all_tto_lifts
from src.models.matchup import score_matchup_for_stat
from src.models.pitcher_k_rate_model import (
    fit_pitcher_k_rate_model,
    prepare_pitcher_model_data,
)
from src.models.pitcher_model import (
    extract_rate_samples as extract_generalized_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)
from src.models.posterior_utils import extract_pitcher_k_rate_samples

from src.data.league_baselines import get_baselines_dict
from src.models.bf_model import compute_pitcher_bf_priors
from src.models.game_sim.pitch_count_model import build_pitch_count_features
from src.data.queries import get_bullpen_trailing_workload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers — extracted from build_game_sim_predictions for readability
# ---------------------------------------------------------------------------


def _fit_pitcher_posteriors(
    train_seasons: list[int],
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> dict[str, dict[int, np.ndarray]]:
    """Fit K%, BB%, HR% models and extract forward-projected posteriors.

    Returns
    -------
    dict[str, dict[int, np.ndarray]]
        ``{"k": {pid: samples}, "bb": ..., "hr": ...}``
    """
    last_train = max(train_seasons)

    # -- K% dedicated model --
    logger.info("Fitting pitcher K%% model...")
    df_k = build_multi_season_pitcher_k_data(train_seasons, min_bf=10)
    data_k = prepare_pitcher_model_data(df_k)
    _, trace_k = fit_pitcher_k_rate_model(
        data_k, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    df_k_df = data_k["df"]
    pids_k = df_k_df[df_k_df["season"] == last_train]["pitcher_id"].unique()
    k_posteriors: dict[int, np.ndarray] = {}
    for pid in pids_k:
        try:
            samples = extract_pitcher_k_rate_samples(
                trace_k, data_k, pid, last_train,
                project_forward=True, random_seed=random_seed,
            )
            k_posteriors[pid] = samples
        except ValueError:
            continue
    logger.info("K posteriors: %d pitchers", len(k_posteriors))

    # -- BB model --
    logger.info("Fitting pitcher BB model...")
    df_bb = build_multi_season_pitcher_data(train_seasons, min_bf=10)
    data_bb = prepare_pitcher_data(df_bb, "bb_rate")
    _, trace_bb = fit_pitcher_model(
        data_bb, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 1,
    )

    df_bb_df = data_bb["df"]
    pids_bb = df_bb_df[df_bb_df["season"] == last_train]["pitcher_id"].unique()
    bb_posteriors: dict[int, np.ndarray] = {}
    for pid in pids_bb:
        try:
            samples = extract_generalized_rate_samples(
                trace_bb, data_bb, pid, last_train,
                project_forward=True, random_seed=random_seed + 1,
            )
            bb_posteriors[pid] = samples
        except ValueError:
            continue
    logger.info("BB posteriors: %d pitchers", len(bb_posteriors))

    # -- HR model (reuses df_bb data) --
    logger.info("Fitting pitcher HR model...")
    data_hr = prepare_pitcher_data(df_bb, "hr_per_bf")
    _, trace_hr = fit_pitcher_model(
        data_hr, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 2,
    )

    df_hr_df = data_hr["df"]
    pids_hr = df_hr_df[df_hr_df["season"] == last_train]["pitcher_id"].unique()
    hr_posteriors: dict[int, np.ndarray] = {}
    for pid in pids_hr:
        try:
            samples = extract_generalized_rate_samples(
                trace_hr, data_hr, pid, last_train,
                project_forward=True, random_seed=random_seed + 2,
            )
            hr_posteriors[pid] = samples
        except ValueError:
            continue
    logger.info("HR posteriors: %d pitchers", len(hr_posteriors))

    return {"k": k_posteriors, "bb": bb_posteriors, "hr": hr_posteriors}


def _load_game_context(
    train_seasons: list[int],
    test_season: int,
) -> dict[str, Any]:
    """Load all per-game context data needed by the simulation loop.

    Returns a dict with keys:
        exit_model, pitcher_pc_latest, batter_pc_latest, tto_profiles,
        pitcher_arsenal, hitter_vuln, baselines_pt, umpire_lifts,
        weather_lifts, pitcher_tend_latest, bf_priors_latest,
        bullpen_wl_lookup
    """
    last_train = max(train_seasons)

    # Exit model
    logger.info("Training exit model...")
    exit_train_data = get_exit_model_training_data(train_seasons)
    exit_tendencies = get_pitcher_exit_tendencies(train_seasons)
    exit_model = ExitModel()
    exit_metrics = exit_model.train(exit_train_data, exit_tendencies)
    logger.info("Exit model AUC: %.4f", exit_metrics["auc"])

    # Pitch count features — use most recent season per player
    logger.info("Loading pitch count features...")
    pitcher_pc_features = get_pitcher_pitch_count_features(train_seasons)
    batter_pc_features = get_batter_pitch_count_features(train_seasons)

    pitcher_pc_latest = (
        pitcher_pc_features
        .sort_values("season")
        .groupby("pitcher_id")
        .last()
        .reset_index()
    )
    pitcher_pc_latest["season"] = last_train

    batter_pc_latest = (
        batter_pc_features
        .sort_values("season")
        .groupby("batter_id")
        .last()
        .reset_index()
    )
    batter_pc_latest["season"] = last_train

    # TTO, matchup, and context lifts
    logger.info("Loading matchup and context data...")
    tto_profiles = get_tto_adjustment_profiles(train_seasons)

    pitcher_arsenal = get_pitcher_arsenal(last_train)
    hitter_vuln = get_hitter_vulnerability(last_train)
    baselines_pt = get_baselines_dict(seasons=train_seasons, recency_weights="equal")

    umpire_lifts = build_umpire_logit_lifts(train_seasons, test_season)
    weather_lifts = build_weather_logit_lifts(train_seasons, test_season)

    # Pitcher exit tendencies (latest per pitcher)
    pitcher_tend_latest = (
        exit_tendencies.copy()
        .sort_values("season")
        .groupby("pitcher_id")
        .last()
        .reset_index()
    )

    # BF priors (empirical Bayes)
    bf_game_logs = get_cached_pitcher_game_logs(last_train)
    for s in train_seasons[:-1]:
        bf_game_logs = pd.concat(
            [bf_game_logs, get_cached_pitcher_game_logs(s)],
            ignore_index=True,
        )
    bf_priors = compute_pitcher_bf_priors(bf_game_logs)
    bf_priors_latest = (
        bf_priors.sort_values("season")
        .groupby("pitcher_id").last().reset_index()
    )
    logger.info("BF priors: %d pitchers", len(bf_priors_latest))

    # Bullpen trailing workload for test season
    bullpen_workload = get_bullpen_trailing_workload([test_season])
    bullpen_wl_lookup: dict[tuple[int, int], float] = {}
    if not bullpen_workload.empty:
        for _, bw_row in bullpen_workload.iterrows():
            key = (int(bw_row["team_id"]), int(bw_row["game_pk"]))
            bullpen_wl_lookup[key] = float(bw_row["bullpen_trailing_ip"])
        logger.info("Bullpen workload: %d team-games", len(bullpen_wl_lookup))

    return {
        "exit_model": exit_model,
        "pitcher_pc_latest": pitcher_pc_latest,
        "batter_pc_latest": batter_pc_latest,
        "tto_profiles": tto_profiles,
        "pitcher_arsenal": pitcher_arsenal,
        "hitter_vuln": hitter_vuln,
        "baselines_pt": baselines_pt,
        "umpire_lifts": umpire_lifts,
        "weather_lifts": weather_lifts,
        "pitcher_tend_latest": pitcher_tend_latest,
        "bf_priors_latest": bf_priors_latest,
        "bullpen_wl_lookup": bullpen_wl_lookup,
    }


def _resolve_opposing_lineup(
    game_pk: int,
    pitcher_id: int,
    game_lineups: pd.DataFrame,
    actuals: pd.DataFrame,
) -> list[int] | None:
    """Resolve the opposing 9-batter lineup for a pitcher in a game.

    Returns
    -------
    list[int] | None
        Nine batter IDs in batting-order, or ``None`` if unresolvable.
    """
    game_lu = game_lineups[game_lineups["game_pk"] == game_pk]

    if len(game_lu) == 0:
        return None

    # Find pitcher's team, then get the OTHER team's batters
    pitcher_team = game_lu[game_lu["player_id"] == pitcher_id]["team_id"]
    if len(pitcher_team) == 0:
        # Pitcher not in lineup data — fallback: pick team pitcher is NOT on
        team_counts = game_lu["team_id"].value_counts()
        if len(team_counts) < 2:
            return None
        teams = game_lu["team_id"].unique()
        opposing_team = None
        for t in teams:
            team_players = game_lu[game_lu["team_id"] == t]["player_id"].values
            if pitcher_id not in team_players:
                opposing_team = t
                break
        if opposing_team is None:
            return None
    else:
        pitcher_team_id = int(pitcher_team.iloc[0])
        opp_rows = game_lu[game_lu["team_id"] != pitcher_team_id]
        if len(opp_rows) == 0:
            return None
        opposing_team = opp_rows["team_id"].iloc[0]

    opposing_lu = game_lu[game_lu["team_id"] == opposing_team].sort_values(
        "batting_order"
    )
    batter_ids = opposing_lu["player_id"].tolist()[:9]

    if len(batter_ids) < 9:
        return None

    return batter_ids


def _extract_game_actuals(
    pitcher_id: int,
    game_pk: int,
    game_row: pd.Series,
    actuals: pd.DataFrame,
) -> dict[str, int | float]:
    """Extract actual stat line for one pitcher game.

    Returns
    -------
    dict
        Keys: actual_k, actual_bb, actual_h, actual_hr, actual_bf,
        actual_pitches, actual_ip, actual_outs
    """
    actual_row = actuals[
        (actuals["pitcher_id"] == pitcher_id)
        & (actuals["game_pk"] == game_pk)
    ]
    if len(actual_row) == 0:
        actual_k = int(game_row.get("strike_outs", 0))
        actual_bb = int(game_row.get("walks", 0))
        actual_h = int(game_row.get("hits", 0))
        actual_hr = int(game_row.get("home_runs", 0))
        actual_bf = int(game_row.get("batters_faced", 0))
        actual_pitches = 0
        actual_ip = float(game_row.get("innings_pitched", 0))
    else:
        ar = actual_row.iloc[0]
        actual_k = int(ar.get("pit_k", 0))
        actual_bb = int(ar.get("pit_bb", 0))
        actual_h = int(ar.get("pit_h", 0))
        actual_hr = int(ar.get("pit_hr", 0))
        actual_bf = int(ar.get("pit_bf", 0))
        actual_pitches = int(ar.get("pit_pitches", 0))
        actual_ip = float(ar.get("pit_ip", 0))

    actual_outs = int(actual_ip) * 3 + round((actual_ip % 1) * 10)

    return {
        "actual_k": actual_k,
        "actual_bb": actual_bb,
        "actual_h": actual_h,
        "actual_hr": actual_hr,
        "actual_bf": actual_bf,
        "actual_pitches": actual_pitches,
        "actual_ip": actual_ip,
        "actual_outs": actual_outs,
    }


def _compute_lineup_matchup_lifts(
    pitcher_id: int,
    batter_ids: list[int],
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Score matchups for a 9-batter lineup across K, BB, HR.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        (lifts, reliabilities). Each dict has keys 'k', 'bb', 'hr'
        with shape (9,) arrays. Reliability ranges [0, 1].
    """
    lifts: dict[str, np.ndarray] = {
        "k": np.zeros(9),
        "bb": np.zeros(9),
        "hr": np.zeros(9),
    }
    reliabilities: dict[str, np.ndarray] = {
        "k": np.zeros(9),
        "bb": np.zeros(9),
        "hr": np.zeros(9),
    }

    for i, batter_id in enumerate(batter_ids):
        for stat in ("k", "bb", "hr"):
            result = score_matchup_for_stat(
                stat_name=stat,
                pitcher_id=pitcher_id,
                batter_id=batter_id,
                pitcher_arsenal=pitcher_arsenal,
                hitter_vuln=hitter_vuln,
                baselines_pt=baselines_pt,
            )
            lift_key = f"matchup_{stat}_logit_lift"
            lift = result.get(lift_key, 0.0)
            if np.isnan(lift) if isinstance(lift, float) else False:
                lift = 0.0
            lifts[stat][i] = lift
            reliabilities[stat][i] = result.get("avg_reliability", 0.0)

    return lifts, reliabilities


# ---------------------------------------------------------------------------
# Core backtest: build predictions for one fold
# ---------------------------------------------------------------------------

def build_game_sim_predictions(
    train_seasons: list[int],
    test_season: int,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_sims: int = 5000,
    starters_only: bool = True,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build game-level predictions using the PA simulator for one fold.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons for training models.
    test_season : int
        Season to predict.
    draws, tune, chains : int
        MCMC parameters for pitcher models.
    n_sims : int
        Monte Carlo simulations per game.
    starters_only : bool
        Only predict starter games.
    min_bf_game : int
        Minimum BF in a game to include.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per game with predicted and actual stats.
    """
    last_train = max(train_seasons)
    logger.info(
        "Building game sim predictions: train=%s, test=%d",
        train_seasons, test_season,
    )

    # ---------------------------------------------------------------
    # 1-3. Fit pitcher K%, BB%, HR% models and extract posteriors
    # ---------------------------------------------------------------
    posteriors = _fit_pitcher_posteriors(
        train_seasons, draws, tune, chains, random_seed,
    )
    k_posteriors = posteriors["k"]
    bb_posteriors = posteriors["bb"]
    hr_posteriors = posteriors["hr"]

    # ---------------------------------------------------------------
    # 4-6. Load exit model, pitch count, matchup, and context data
    # ---------------------------------------------------------------
    ctx = _load_game_context(train_seasons, test_season)
    exit_model = ctx["exit_model"]
    pitcher_pc_latest = ctx["pitcher_pc_latest"]
    batter_pc_latest = ctx["batter_pc_latest"]
    tto_profiles = ctx["tto_profiles"]
    pitcher_arsenal = ctx["pitcher_arsenal"]
    hitter_vuln = ctx["hitter_vuln"]
    baselines_pt = ctx["baselines_pt"]
    umpire_lifts = ctx["umpire_lifts"]
    weather_lifts = ctx["weather_lifts"]
    pitcher_tend_latest = ctx["pitcher_tend_latest"]
    bf_priors_latest = ctx["bf_priors_latest"]
    bullpen_wl_lookup = ctx["bullpen_wl_lookup"]

    # ---------------------------------------------------------------
    # 7. Load test season data — actuals + lineups
    # ---------------------------------------------------------------
    logger.info("Loading test season %d data...", test_season)
    test_game_logs = get_cached_pitcher_game_logs(test_season)
    if starters_only:
        test_game_logs = test_game_logs[
            test_game_logs["is_starter"] == True  # noqa: E712
        ].copy()
    test_game_logs = test_game_logs[
        test_game_logs["batters_faced"] >= min_bf_game
    ].copy()
    test_game_logs["season"] = test_season

    game_lineups = get_cached_game_lineups(test_season)

    # Load actuals from fact_player_game_mlb
    actuals = read_sql(f"""
        SELECT player_id AS pitcher_id, game_pk, team_id,
               pit_k, pit_bb, pit_h, pit_hr, pit_bf,
               pit_pitches, pit_ip, pit_er
        FROM production.fact_player_game_mlb
        WHERE pit_is_starter = TRUE
          AND season = {int(test_season)}
          AND pit_bf >= {min_bf_game}
    """)

    logger.info("Test games: %d starter appearances", len(test_game_logs))

    # ---------------------------------------------------------------
    # 8. Run simulator for each game
    # ---------------------------------------------------------------
    results = []
    n_skipped = 0

    # Set of pitchers that have all three posteriors
    valid_pids = (
        set(k_posteriors.keys())
        & set(bb_posteriors.keys())
        & set(hr_posteriors.keys())
    )
    logger.info("Pitchers with all 3 posteriors: %d", len(valid_pids))

    for _, game_row in test_game_logs.iterrows():
        pitcher_id = int(game_row["pitcher_id"])
        game_pk = int(game_row["game_pk"])

        # Skip if pitcher not in model
        if pitcher_id not in valid_pids:
            n_skipped += 1
            continue

        # Resolve opposing lineup
        batter_ids = _resolve_opposing_lineup(
            game_pk, pitcher_id, game_lineups, actuals,
        )
        if batter_ids is None:
            n_skipped += 1
            continue

        # Compute matchup lifts + reliability
        matchup_lifts, matchup_reliabilities = _compute_lineup_matchup_lifts(
            pitcher_id, batter_ids,
            pitcher_arsenal, hitter_vuln, baselines_pt,
        )

        # TTO lifts
        tto_lifts = build_all_tto_lifts(tto_profiles, pitcher_id, last_train)

        # Pitcher avg pitches + manager tendency
        tend_row = pitcher_tend_latest[
            pitcher_tend_latest["pitcher_id"] == pitcher_id
        ]
        if len(tend_row) > 0:
            avg_pitches = float(tend_row.iloc[0]["avg_pitches"])
            avg_ip = (
                float(tend_row.iloc[0]["avg_ip"])
                if "avg_ip" in tend_row.columns
                and pd.notna(tend_row.iloc[0].get("avg_ip"))
                else 5.28
            )
            team_avg_p = (
                float(tend_row.iloc[0]["team_avg_pitches"])
                if "team_avg_pitches" in tend_row.columns
                and pd.notna(tend_row.iloc[0].get("team_avg_pitches"))
                else 88.0
            )
        else:
            avg_pitches = 88.0
            avg_ip = 5.2
            team_avg_p = 88.0

        # BF prior for this pitcher (bf_model empirical Bayes)
        bf_row = bf_priors_latest[
            bf_priors_latest["pitcher_id"] == pitcher_id
        ]
        mu_bf = (
            float(bf_row.iloc[0]["mu_bf"])
            if len(bf_row) > 0 else None
        )

        # Bullpen workload: look up pitcher's team for this game
        actual_row_team = actuals[
            (actuals["pitcher_id"] == pitcher_id)
            & (actuals["game_pk"] == game_pk)
        ]
        pitcher_team_id_for_bp = (
            int(actual_row_team.iloc[0]["team_id"])
            if len(actual_row_team) > 0 else None
        )
        bullpen_ip = (
            bullpen_wl_lookup.get((pitcher_team_id_for_bp, game_pk))
            if pitcher_team_id_for_bp is not None else None
        )

        # Context lifts
        ump_k = umpire_lifts["k"].get(game_pk, 0.0)
        ump_bb = umpire_lifts["bb"].get(game_pk, 0.0)
        wx_k = weather_lifts.get("k", {}).get(game_pk, 0.0)
        park_hr = weather_lifts.get("hr", {}).get(game_pk, 0.0)

        # Build pitch count features
        pitcher_adj, batter_adjs = build_pitch_count_features(
            pitcher_features=pitcher_pc_latest,
            batter_features=batter_pc_latest,
            pitcher_id=pitcher_id,
            batter_ids=batter_ids,
            season=last_train,
        )

        # Lineup patience: mean of batter P/PA adjustments
        lineup_ppa_agg = float(np.mean(batter_adjs))

        # Unified exit offset with all three signals
        exit_offset = compute_exit_offset(
            mu_bf=mu_bf,
            pitcher_avg_ip=avg_ip,
            bullpen_trailing_ip=bullpen_ip,
            lineup_ppa_aggregate=lineup_ppa_agg,
        )

        # Run simulation
        try:
            sim_result = simulate_game(
                pitcher_k_rate_samples=k_posteriors[pitcher_id],
                pitcher_bb_rate_samples=bb_posteriors[pitcher_id],
                pitcher_hr_rate_samples=hr_posteriors[pitcher_id],
                lineup_matchup_lifts=matchup_lifts,
                tto_lifts=tto_lifts,
                pitcher_ppa_adj=pitcher_adj,
                batter_ppa_adjs=batter_adjs,
                exit_model=exit_model,
                pitcher_avg_pitches=avg_pitches,
                exit_calibration_offset=exit_offset,
                game_context=GameContext(
                    umpire_k_lift=ump_k,
                    umpire_bb_lift=ump_bb,
                    park_hr_lift=park_hr,
                    weather_k_lift=wx_k,
                ),
                lineup_matchup_reliabilities=matchup_reliabilities,
                manager_pull_tendency=team_avg_p,
                n_sims=n_sims,
                random_seed=random_seed + game_pk % 10000,
            )
        except Exception as e:
            logger.warning("Simulation failed for game %d pitcher %d: %s",
                           game_pk, pitcher_id, e)
            n_skipped += 1
            continue

        # Get actuals
        game_actuals = _extract_game_actuals(
            pitcher_id, game_pk, game_row, actuals,
        )

        summary = sim_result.summary()

        # Build K over-prob columns
        k_over = sim_result.over_probs("k", [3.5, 4.5, 5.5, 6.5, 7.5])
        # Outs prop lines (14.5 ~ 5.0 IP, 17.5 ~ 6.0 IP, etc.)
        outs_over = sim_result.over_probs(
            "outs", [14.5, 15.5, 16.5, 17.5, 18.5],
        )

        rec = {
            "game_pk": game_pk,
            "pitcher_id": pitcher_id,
            "test_season": test_season,
            # Predicted
            "expected_k": summary["k"]["mean"],
            "std_k": summary["k"]["std"],
            "expected_bb": summary["bb"]["mean"],
            "std_bb": summary["bb"]["std"],
            "expected_h": summary["h"]["mean"],
            "std_h": summary["h"]["std"],
            "expected_hr": summary["hr"]["mean"],
            "std_hr": summary["hr"]["std"],
            "expected_bf": summary["bf"]["mean"],
            "expected_pitches": summary["pitch_count"]["mean"],
            "expected_outs": summary["outs"]["mean"],
            "std_outs": summary["outs"]["std"],
            "expected_ip": float(np.mean(sim_result.outs_samples)) / 3.0,
            "expected_runs": summary["runs"]["mean"],
            **game_actuals,
        }

        # K prop line probabilities
        for _, prow in k_over.iterrows():
            col = f"p_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        # Outs prop line probabilities
        for _, prow in outs_over.iterrows():
            col = f"p_outs_over_{prow['line']:.1f}".replace(".", "_")
            rec[col] = prow["p_over"]

        results.append(rec)

    logger.info(
        "Completed: %d games predicted, %d skipped", len(results), n_skipped
    )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_game_sim_metrics(
    predictions: pd.DataFrame,
    k_lines: list[float] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive metrics for game simulator predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of build_game_sim_predictions().
    k_lines : list[float], optional
        K lines for Brier score. Default: [3.5, 4.5, 5.5, 6.5, 7.5].

    Returns
    -------
    dict
        Metrics per stat (bias, correlation) plus K-specific
        (brier, coverage, ECE).
    """
    if k_lines is None:
        k_lines = [3.5, 4.5, 5.5, 6.5, 7.5]

    n = len(predictions)
    if n == 0:
        return {"n_games": 0}

    metrics: dict[str, Any] = {"n_games": n}

    # Per-stat bias, correlation
    for stat in ("k", "bb", "h", "hr", "bf", "pitches", "outs"):
        exp_col = f"expected_{stat}"
        act_col = f"actual_{stat}"
        if exp_col not in predictions.columns or act_col not in predictions.columns:
            continue

        expected = predictions[exp_col].values.astype(float)
        actual = predictions[act_col].values.astype(float)

        # Skip stats with all zeros in actuals
        if np.all(actual == 0):
            continue

        errors = expected - actual
        metrics[f"{stat}_bias"] = float(np.mean(errors))

        if np.std(actual) > 0 and np.std(expected) > 0:
            metrics[f"{stat}_corr"] = float(np.corrcoef(actual, expected)[0, 1])

    # K-specific: Brier scores and log loss
    brier_scores: dict[float, float] = {}
    k_log_losses: dict[float, float] = {}
    for line in k_lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            continue
        y_true = (predictions["actual_k"] > line).astype(float).values
        y_prob = predictions[col].values
        brier_scores[line] = float(brier_score_loss(y_true, y_prob))
        k_log_losses[line] = compute_log_loss(y_prob, y_true)
    metrics["k_brier_scores"] = brier_scores
    if brier_scores:
        metrics["k_avg_brier"] = float(np.mean(list(brier_scores.values())))
    metrics["k_log_losses"] = k_log_losses
    if k_log_losses:
        metrics["k_avg_log_loss"] = float(np.mean(list(k_log_losses.values())))

    # Outs Brier scores and log loss
    outs_lines = [14.5, 15.5, 16.5, 17.5, 18.5]
    outs_brier: dict[float, float] = {}
    outs_log_losses: dict[float, float] = {}
    for line in outs_lines:
        col = f"p_outs_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            continue
        if "actual_outs" not in predictions.columns:
            break
        y_true = (predictions["actual_outs"] > line).astype(float).values
        y_prob = predictions[col].values
        outs_brier[line] = float(brier_score_loss(y_true, y_prob))
        outs_log_losses[line] = compute_log_loss(y_prob, y_true)
    metrics["outs_brier_scores"] = outs_brier
    if outs_brier:
        metrics["outs_avg_brier"] = float(np.mean(list(outs_brier.values())))
    metrics["outs_log_losses"] = outs_log_losses
    if outs_log_losses:
        metrics["outs_avg_log_loss"] = float(np.mean(list(outs_log_losses.values())))

    # K sharpness
    k_lines_used = [3.5, 4.5, 5.5, 6.5, 7.5]
    k_probs_all: list[np.ndarray] = []
    for line in k_lines_used:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col in predictions.columns:
            k_probs_all.append(predictions[col].values.astype(float))
    if k_probs_all:
        k_sharp = compute_sharpness(np.concatenate(k_probs_all))
    else:
        k_sharp = compute_sharpness(np.array([]))
    for key, val in k_sharp.items():
        metrics[f"k_sharpness_{key}"] = val

    # Outs sharpness
    outs_probs_all: list[np.ndarray] = []
    for line in outs_lines:
        col = f"p_outs_over_{line:.1f}".replace(".", "_")
        if col in predictions.columns:
            outs_probs_all.append(predictions[col].values.astype(float))
    if outs_probs_all:
        outs_sharp = compute_sharpness(np.concatenate(outs_probs_all))
    else:
        outs_sharp = compute_sharpness(np.array([]))
    for key, val in outs_sharp.items():
        metrics[f"outs_sharpness_{key}"] = val

    # K coverage
    if "std_k" in predictions.columns:
        expected_k = predictions["expected_k"].values
        std_k = predictions["std_k"].values
        actual_k = predictions["actual_k"].values

        for ci_name, z in [("50", 0.6745), ("80", 1.2816), ("90", 1.6449)]:
            lo = expected_k - z * std_k
            hi = expected_k + z * std_k
            metrics[f"k_coverage_{ci_name}"] = float(
                np.mean((actual_k >= lo) & (actual_k <= hi))
            )

    # Outs coverage
    if "std_outs" in predictions.columns and "actual_outs" in predictions.columns:
        expected_outs = predictions["expected_outs"].values
        std_outs = predictions["std_outs"].values
        actual_outs_arr = predictions["actual_outs"].values

        for ci_name, z in [("50", 0.6745), ("80", 1.2816), ("90", 1.6449)]:
            lo = expected_outs - z * std_outs
            hi = expected_outs + z * std_outs
            metrics[f"outs_coverage_{ci_name}"] = float(
                np.mean((actual_outs_arr >= lo) & (actual_outs_arr <= hi))
            )

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------

def run_full_game_sim_backtest(
    folds: list[tuple[list[int], int]] | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_sims: int = 5000,
    min_bf_game: int = 15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward backtest across multiple folds.

    Parameters
    ----------
    folds : list of (train_seasons, test_season) or None
        Default: train on expanding window, test 2023-2025.
    draws, tune, chains : int
        MCMC parameters.
    n_sims : int
        MC simulations per game.
    min_bf_game : int
        Minimum BF per game.
    random_seed : int
        For reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (fold_summary, all_predictions)
    """
    if folds is None:
        folds = [
            ([2020, 2021, 2022], 2023),
            ([2020, 2021, 2022, 2023], 2024),
            ([2020, 2021, 2022, 2023, 2024], 2025),
        ]

    fold_results = []
    all_predictions = []

    for train_seasons, test_season in folds:
        logger.info("=" * 60)
        logger.info("FOLD: train=%s → test=%d", train_seasons, test_season)
        logger.info("=" * 60)

        predictions = build_game_sim_predictions(
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws,
            tune=tune,
            chains=chains,
            n_sims=n_sims,
            min_bf_game=min_bf_game,
            random_seed=random_seed,
        )

        if len(predictions) == 0:
            logger.warning("No predictions for fold")
            continue

        metrics = compute_game_sim_metrics(predictions)

        fold_rec = {
            "test_season": test_season,
            "n_games": metrics["n_games"],
        }

        # Add per-stat metrics
        for stat in ("k", "bb", "h", "hr", "bf", "pitches", "outs"):
            for metric in ("bias", "corr"):
                key = f"{stat}_{metric}"
                if key in metrics:
                    fold_rec[key] = metrics[key]

        # K-specific
        if "k_avg_brier" in metrics:
            fold_rec["k_avg_brier"] = metrics["k_avg_brier"]
        if "k_avg_log_loss" in metrics:
            fold_rec["k_avg_log_loss"] = metrics["k_avg_log_loss"]
        for ci in ("50", "80", "90"):
            key = f"k_coverage_{ci}"
            if key in metrics:
                fold_rec[key] = metrics[key]

        # Outs-specific
        if "outs_avg_brier" in metrics:
            fold_rec["outs_avg_brier"] = metrics["outs_avg_brier"]
        if "outs_avg_log_loss" in metrics:
            fold_rec["outs_avg_log_loss"] = metrics["outs_avg_log_loss"]
        for ci in ("50", "80", "90"):
            key = f"outs_coverage_{ci}"
            if key in metrics:
                fold_rec[key] = metrics[key]

        # Sharpness
        for prefix in ("k", "outs"):
            for skey in ("mean_confidence", "pct_actionable_60",
                         "pct_actionable_65", "pct_actionable_70", "entropy"):
                mkey = f"{prefix}_sharpness_{skey}"
                if mkey in metrics:
                    fold_rec[mkey] = metrics[mkey]

        fold_results.append(fold_rec)
        predictions["fold_test_season"] = test_season
        all_predictions.append(predictions)

        # Log summary
        logger.info(
            "Fold results: K Brier=%.4f, K LogLoss=%.4f, "
            "Outs Brier=%.4f, Outs LogLoss=%.4f, n=%d",
            fold_rec.get("k_avg_brier", np.nan),
            fold_rec.get("k_avg_log_loss", np.nan),
            fold_rec.get("outs_avg_brier", np.nan),
            fold_rec.get("outs_avg_log_loss", np.nan),
            metrics["n_games"],
        )
        logger.info(
            "  Sharpness: K(conf=%.3f, act60=%.1f%%, entropy=%.3f) "
            "Outs(conf=%.3f, act60=%.1f%%, entropy=%.3f)",
            fold_rec.get("k_sharpness_mean_confidence", np.nan),
            fold_rec.get("k_sharpness_pct_actionable_60", np.nan),
            fold_rec.get("k_sharpness_entropy", np.nan),
            fold_rec.get("outs_sharpness_mean_confidence", np.nan),
            fold_rec.get("outs_sharpness_pct_actionable_60", np.nan),
            fold_rec.get("outs_sharpness_entropy", np.nan),
        )

    summary_df = pd.DataFrame(fold_results)
    pred_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions else pd.DataFrame()
    )

    # Overall metrics
    if len(pred_df) > 0:
        overall = compute_game_sim_metrics(pred_df)
        logger.info("=" * 60)
        logger.info("OVERALL RESULTS (%d games)", overall["n_games"])
        for stat in ("k", "bb", "h", "hr", "outs", "bf", "pitches"):
            corr_key = f"{stat}_corr"
            bias_key = f"{stat}_bias"
            if corr_key in overall or bias_key in overall:
                logger.info(
                    "  %s: bias=%.3f, corr=%.3f",
                    stat.upper(),
                    overall.get(bias_key, np.nan),
                    overall.get(corr_key, np.nan),
                )
        if "k_avg_brier" in overall:
            logger.info("  K Avg Brier: %.4f", overall["k_avg_brier"])
        if "k_avg_log_loss" in overall:
            logger.info("  K Avg Log Loss: %.4f", overall["k_avg_log_loss"])
        if "outs_avg_brier" in overall:
            logger.info("  Outs Avg Brier: %.4f", overall["outs_avg_brier"])
        if "outs_avg_log_loss" in overall:
            logger.info("  Outs Avg Log Loss: %.4f", overall["outs_avg_log_loss"])
        for prefix, label in [("k", "K"), ("outs", "Outs")]:
            mc_key = f"{prefix}_sharpness_mean_confidence"
            if mc_key in overall:
                logger.info(
                    "  %s Sharpness: conf=%.3f, act60=%.1f%%, "
                    "act65=%.1f%%, act70=%.1f%%, entropy=%.3f",
                    label,
                    overall[mc_key],
                    overall.get(f"{prefix}_sharpness_pct_actionable_60", np.nan),
                    overall.get(f"{prefix}_sharpness_pct_actionable_65", np.nan),
                    overall.get(f"{prefix}_sharpness_pct_actionable_70", np.nan),
                    overall.get(f"{prefix}_sharpness_entropy", np.nan),
                )

    return summary_df, pred_df
