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

from src.evaluation.metrics import (
    compute_coverage_levels,
    compute_log_loss,
    compute_per_line_binary_metrics,
    compute_sharpness,
)

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
from src.data.queries import get_bullpen_trailing_workload, get_team_bullpen_rates
from src.models.game_predictions import crps_sample
from src.utils.constants import BULLPEN_K_RATE, BULLPEN_BB_RATE, BULLPEN_HR_RATE

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


def _load_team_bullpen_rates_lookup(
    train_seasons: list[int],
) -> dict[int, tuple[float, float, float]]:
    """Build a team_id → (k_rate, bb_rate, hr_rate) lookup using ONLY
    training seasons, to avoid look-ahead leakage into the test fold.

    Aggregates across all provided training seasons (weighted by BF) so a
    team that existed for all seasons gets a stable multi-year rate.
    """
    bp_df = get_team_bullpen_rates(train_seasons)
    if bp_df.empty:
        return {}

    # Weighted aggregate across seasons by total_bf
    lookup: dict[int, tuple[float, float, float]] = {}
    for team_id, g in bp_df.groupby("team_id"):
        w = g["total_bf"].astype(float)
        if w.sum() <= 0:
            continue
        lookup[int(team_id)] = (
            float(np.average(g["k_rate"], weights=w)),
            float(np.average(g["bb_rate"], weights=w)),
            float(np.average(g["hr_rate"], weights=w)),
        )
    return lookup


def _load_batter_bip_profiles(
    train_seasons: list[int],
) -> dict[int, np.ndarray]:
    """Build a batter_id → (4,) BIP probability lookup using only training
    seasons. Rates returned as [p_out, p_single, p_double, p_triple].

    Uses the most-recent training season for each batter (so breakout
    profiles don't get averaged with older weaker years). Batters not
    present in any training season fall back to league average at call
    time by returning nothing for their ID.
    """
    from src.data.queries.hitter import get_batter_bip_profile
    from src.data.queries.traditional import get_sprint_speed
    from src.models.game_sim.bip_model import compute_player_bip_probs

    raw = get_batter_bip_profile(train_seasons)
    if raw.empty:
        logger.warning("No batter BIP profile data for train seasons %s",
                       train_seasons)
        return {}

    # Sprint speed for the latest train season (most recent proxy available
    # at test time).
    latest = max(train_seasons)
    try:
        sprint = get_sprint_speed(latest)[["player_id", "sprint_speed"]]
        sprint = sprint.rename(columns={"player_id": "batter_id"})
    except Exception:
        logger.exception("Sprint speed fetch failed; defaulting to 27.0")
        sprint = pd.DataFrame(columns=["batter_id", "sprint_speed"])

    # Most-recent-training-season row per batter (breakouts dominate)
    raw = raw.sort_values(["batter_id", "season"])
    latest_rows = raw.groupby("batter_id").tail(1)
    latest_rows = latest_rows.merge(sprint, on="batter_id", how="left")
    latest_rows["sprint_speed"] = latest_rows["sprint_speed"].fillna(27.0)
    latest_rows["avg_ev"] = latest_rows["avg_ev"].fillna(88.0).astype(float)
    latest_rows["avg_la"] = latest_rows["avg_la"].fillna(12.0).astype(float)
    latest_rows["gb_pct"] = latest_rows["gb_pct"].fillna(0.44).astype(float)

    lookup: dict[int, np.ndarray] = {}
    for _, r in latest_rows.iterrows():
        resolved = int(
            r["bip_outs"] + r["bip_singles"] + r["bip_doubles"] + r["bip_triples"]
        )
        if resolved <= 0:
            continue
        observed = {
            "out": r["bip_outs"] / resolved,
            "single": r["bip_singles"] / resolved,
            "double": r["bip_doubles"] / resolved,
            "triple": r["bip_triples"] / resolved,
        }
        probs = compute_player_bip_probs(
            avg_ev=float(r["avg_ev"]),
            avg_la=float(r["avg_la"]),
            gb_pct=float(r["gb_pct"]),
            sprint_speed=float(r["sprint_speed"]),
            observed_bip_splits=observed,
            bip_count=int(r["bip"]),
            shrinkage_k=900,
        )
        lookup[int(r["batter_id"])] = np.asarray(probs, dtype=np.float64)

    logger.info(
        "Loaded batter BIP profiles: %d batters from train seasons %s",
        len(lookup), train_seasons,
    )
    return lookup


# League-average BIP fallback for batters without a training profile
_LEAGUE_BIP_PROBS = np.array([0.700, 0.222, 0.065, 0.005])


def _build_lineup_bip_probs(
    batter_ids: list[int],
    bip_profiles: dict[int, np.ndarray],
) -> np.ndarray:
    """Assemble a (9, 4) BIP probability matrix for a lineup.

    Missing batters fall back to league average.
    """
    rows = []
    for bid in batter_ids[:9]:
        rows.append(bip_profiles.get(int(bid), _LEAGUE_BIP_PROBS))
    while len(rows) < 9:
        rows.append(_LEAGUE_BIP_PROBS)
    return np.asarray(rows, dtype=np.float64)


def _load_team_runs_allowed(
    test_season: int,
) -> dict[tuple[int, int], int]:
    """Build a (game_pk, allowing_team_id) → runs_allowed lookup for the
    test season, joining ``fact_game_totals`` against ``dim_game`` so we
    know which team was on defense when those runs were scored.
    """
    df = read_sql(f"""
        WITH game_teams AS (
            SELECT game_pk, home_team_id, away_team_id
            FROM production.dim_game
            WHERE season = {int(test_season)}
        )
        SELECT
            fgt.game_pk,
            fgt.team_id AS scoring_team_id,
            fgt.runs AS runs_scored,
            CASE WHEN fgt.team_id = gt.home_team_id
                 THEN gt.away_team_id
                 ELSE gt.home_team_id END AS allowing_team_id
        FROM production.fact_game_totals fgt
        JOIN game_teams gt ON fgt.game_pk = gt.game_pk
        WHERE fgt.season = {int(test_season)}
    """)

    lookup: dict[tuple[int, int], int] = {}
    for _, r in df.iterrows():
        lookup[(int(r["game_pk"]), int(r["allowing_team_id"]))] = int(r["runs_scored"])
    logger.info(
        "Loaded team runs-allowed lookup: %d (game_pk, team) entries", len(lookup),
    )
    return lookup


def _load_team_defense_lifts(
    train_seasons: list[int],
) -> dict[int, float]:
    """Build a team_id → defense BABIP shift lookup from training-season
    fielding_runs_prevented totals. Strict train-only to avoid leakage.

    Aggregates across the requested train seasons (most-recent-wins so
    later seasons override earlier ones with more representative rosters).
    """
    from src.data.queries import get_team_defense_lifts

    lookup: dict[int, float] = {}
    df = get_team_defense_lifts(seasons=train_seasons)
    if df.empty:
        return lookup
    df = df.sort_values("season")
    for _, r in df.iterrows():
        lookup[int(r["team_id"])] = float(r["defense_babip_adj"])
    logger.info(
        "Team defense lifts: %d teams from train seasons %s",
        len(lookup), train_seasons,
    )
    return lookup


def _load_park_lifts(
    test_season: int,
) -> tuple[dict[int, tuple[float, float, float, float]], dict[int, int]]:
    """Load park factor logit lifts and per-game venue mapping for the
    backtest.

    Reuses the precomputed ``park_factor_lifts.parquet`` from the dashboard
    data directory. Since park factors are derived from multi-year rolling
    windows and change very slowly, using precomputed values introduces
    negligible leakage for walk-forward evaluation.

    Returns
    -------
    (park_lift_lookup, game_venue_lookup)
        ``park_lift_lookup[venue_id] = (k, bb, hr, babip)`` logit lifts.
        ``game_venue_lookup[game_pk] = venue_id``.
    """
    from pathlib import Path

    park_lift_lookup: dict[int, tuple[float, float, float, float]] = {}
    park_path = (
        Path(__file__).resolve().parents[2].parent
        / "tdd-dashboard" / "data" / "dashboard" / "park_factor_lifts.parquet"
    )
    if park_path.exists():
        pl = pd.read_parquet(park_path)
        for _, r in pl.iterrows():
            park_lift_lookup[int(r["venue_id"])] = (
                float(r.get("k_lift", 0.0) or 0.0),
                float(r.get("bb_lift", 0.0) or 0.0),
                float(r.get("hr_lift", 0.0) or 0.0),
                float(r.get("h_babip_adj", 0.0) or 0.0),
            )
        logger.info("Loaded park factor lifts for %d venues", len(park_lift_lookup))
    else:
        logger.warning("Park factor lifts parquet not found at %s", park_path)

    # game_pk → venue_id for the test season
    game_venue_lookup: dict[int, int] = {}
    df = read_sql(f"""
        SELECT DISTINCT game_pk, venue_id
        FROM production.dim_game
        WHERE season = {int(test_season)}
          AND game_type = 'R'
          AND venue_id IS NOT NULL
    """)
    for _, r in df.iterrows():
        game_venue_lookup[int(r["game_pk"])] = int(r["venue_id"])
    logger.info(
        "Loaded game_pk → venue_id map: %d games", len(game_venue_lookup),
    )
    return park_lift_lookup, game_venue_lookup


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
    park_lift_lookup, game_venue_lookup = _load_park_lifts(test_season)
    team_defense_lookup = _load_team_defense_lifts(train_seasons)

    # Catcher framing lifts — keyed by (game_pk, pitcher_id). Legacy engine
    # applies both K and BB lifts here; PA-sim now reads both via GameContext.
    from src.data.catcher_framing import build_catcher_framing_lookup
    framing_lookup = build_catcher_framing_lookup(train_seasons, test_season)
    catcher_k_lifts = framing_lookup.get("k", {}) or {}
    catcher_bb_lifts = framing_lookup.get("bb", {}) or {}

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

    # Team bullpen rate aggregate — training seasons only (no leakage)
    team_bullpen_rates_lookup = _load_team_bullpen_rates_lookup(train_seasons)
    logger.info(
        "Team bullpen rates: %d teams loaded from train seasons",
        len(team_bullpen_rates_lookup),
    )

    # Actual team runs allowed per (game_pk, team) for test season
    team_runs_allowed_lookup = _load_team_runs_allowed(test_season)

    # Batter BIP profiles — train-only (no leakage)
    batter_bip_lookup = _load_batter_bip_profiles(train_seasons)

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
        "park_lift_lookup": park_lift_lookup,
        "game_venue_lookup": game_venue_lookup,
        "team_defense_lookup": team_defense_lookup,
        "pitcher_tend_latest": pitcher_tend_latest,
        "bf_priors_latest": bf_priors_latest,
        "bullpen_wl_lookup": bullpen_wl_lookup,
        "team_bullpen_rates_lookup": team_bullpen_rates_lookup,
        "team_runs_allowed_lookup": team_runs_allowed_lookup,
        "batter_bip_lookup": batter_bip_lookup,
        "catcher_k_lifts": catcher_k_lifts,
        "catcher_bb_lifts": catcher_bb_lifts,
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
    # 3b. Fit hitter K%, BB%, HR% posteriors for batter quality lifts
    # ---------------------------------------------------------------
    from src.evaluation.runner import fit_hitter_posteriors
    hitter_posts = fit_hitter_posteriors(
        train_seasons, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed + 100,
    )
    batter_k_posteriors = hitter_posts["k"]
    batter_bb_posteriors = hitter_posts["bb"]
    batter_hr_posteriors = hitter_posts["hr"]
    logger.info(
        "Batter posteriors for quality lifts: K=%d BB=%d HR=%d",
        len(batter_k_posteriors), len(batter_bb_posteriors),
        len(batter_hr_posteriors),
    )

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
    park_lift_lookup = ctx["park_lift_lookup"]
    game_venue_lookup = ctx["game_venue_lookup"]
    team_defense_lookup = ctx["team_defense_lookup"]
    pitcher_tend_latest = ctx["pitcher_tend_latest"]
    bf_priors_latest = ctx["bf_priors_latest"]
    bullpen_wl_lookup = ctx["bullpen_wl_lookup"]
    team_bullpen_rates_lookup = ctx["team_bullpen_rates_lookup"]
    team_runs_allowed_lookup = ctx["team_runs_allowed_lookup"]
    batter_bip_lookup = ctx["batter_bip_lookup"]
    catcher_k_lifts = ctx["catcher_k_lifts"]
    catcher_bb_lifts = ctx["catcher_bb_lifts"]

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

        # BF prior for BF-anchored exit
        bf_row = bf_priors_latest[
            bf_priors_latest["pitcher_id"] == pitcher_id
        ]
        if len(bf_row) > 0:
            mu_bf = float(bf_row.iloc[0]["mu_bf"])
            sigma_bf = float(bf_row.iloc[0]["sigma_bf"])
        else:
            mu_bf = None
            sigma_bf = None

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
        wx_hr = weather_lifts.get("hr", {}).get(game_pk, 0.0)
        venue_id = game_venue_lookup.get(game_pk)
        pk_k, pk_bb, pk_hr, pk_babip = park_lift_lookup.get(
            venue_id, (0.0, 0.0, 0.0, 0.0),
        )
        # Defense BABIP shift for the team currently fielding
        defense_babip = (
            team_defense_lookup.get(int(pitcher_team_id_for_bp), 0.0)
            if pitcher_team_id_for_bp is not None else 0.0
        )
        combined_babip = pk_babip + defense_babip

        # Catcher framing (per (game_pk, pitcher_id))
        fram_k = catcher_k_lifts.get((game_pk, pitcher_id), 0.0)
        fram_bb = catcher_bb_lifts.get((game_pk, pitcher_id), 0.0)

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


        # Pitching team's bullpen rates — train-only aggregate
        bp_rates = (
            team_bullpen_rates_lookup.get(pitcher_team_id_for_bp)
            if pitcher_team_id_for_bp is not None else None
        )
        if bp_rates is not None:
            bp_k, bp_bb, bp_hr = bp_rates
        else:
            bp_k, bp_bb, bp_hr = (
                BULLPEN_K_RATE, BULLPEN_BB_RATE, BULLPEN_HR_RATE,
            )

        # Assemble per-batter BIP probabilities for this lineup (9, 4)
        lineup_bip_probs = _build_lineup_bip_probs(
            batter_ids, batter_bip_lookup,
        )

        # Per-batter K/BB/HR posteriors for quality lifts
        _FB_K = np.full(200, 0.226)
        _FB_BB = np.full(200, 0.082)
        _FB_HR = np.full(200, 0.031)
        opp_bat_k = [batter_k_posteriors.get(int(b), _FB_K) for b in batter_ids]
        opp_bat_bb = [batter_bb_posteriors.get(int(b), _FB_BB) for b in batter_ids]
        opp_bat_hr = [batter_hr_posteriors.get(int(b), _FB_HR) for b in batter_ids]

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
                mu_bf=mu_bf,
                sigma_bf=sigma_bf,
                game_context=GameContext(
                    umpire_k_lift=ump_k,
                    umpire_bb_lift=ump_bb,
                    park_k_lift=pk_k,
                    park_bb_lift=pk_bb,
                    park_hr_lift=pk_hr,
                    park_h_babip_adj=combined_babip,
                    weather_k_lift=wx_k,
                    weather_hr_lift=wx_hr,
                    catcher_k_lift=fram_k,
                    catcher_bb_lift=fram_bb,
                ),
                lineup_matchup_reliabilities=matchup_reliabilities,
                manager_pull_tendency=team_avg_p,
                babip_adj=combined_babip,
                bullpen_k_rate=bp_k,
                bullpen_bb_rate=bp_bb,
                bullpen_hr_rate=bp_hr,
                lineup_bip_probs=lineup_bip_probs,
                lineup_batter_k_samples=opp_bat_k,
                lineup_batter_bb_samples=opp_bat_bb,
                lineup_batter_hr_samples=opp_bat_hr,
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

        # Team-level actual runs allowed: uses the pitcher's team + game_pk
        # key in the lookup (runs scored by the opposing offense).
        actual_total_runs = (
            team_runs_allowed_lookup.get((game_pk, pitcher_team_id_for_bp))
            if pitcher_team_id_for_bp is not None else None
        )

        summary = sim_result.summary()

        # Full 0.5-24.5 p_over grids for every pitcher stat — matches the
        # format that confident_picks.py writes to game_props.parquet so the
        # dashboard validator can score these directly.
        _ALL_LINES = [x + 0.5 for x in range(25)]
        k_over = sim_result.over_probs("k", _ALL_LINES)
        bb_over = sim_result.over_probs("bb", _ALL_LINES)
        h_over = sim_result.over_probs("h", _ALL_LINES)
        hr_over = sim_result.over_probs("hr", _ALL_LINES)
        outs_over = sim_result.over_probs("outs", _ALL_LINES)

        # Run-level quantities — runs_samples is full-game (starter+pen)
        runs_samples = sim_result.runs_samples.astype(np.float64)
        runs_q10, runs_q50, runs_q90 = np.percentile(runs_samples, [10, 50, 90])
        runs_crps = (
            crps_sample(runs_samples, float(actual_total_runs))
            if actual_total_runs is not None else np.nan
        )

        rec = {
            "game_pk": game_pk,
            "pitcher_id": pitcher_id,
            "pitcher_team_id": pitcher_team_id_for_bp,
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
            # Run-level predictions (full game, starter + bullpen tail)
            "expected_runs": summary["runs"]["mean"],
            "std_runs": summary["runs"]["std"],
            "runs_q10": float(runs_q10),
            "runs_q50": float(runs_q50),
            "runs_q90": float(runs_q90),
            "expected_starter_runs": (
                float(np.mean(sim_result.starter_runs_samples))
                if sim_result.starter_runs_samples is not None else np.nan
            ),
            "expected_bullpen_runs": (
                float(np.mean(sim_result.bullpen_runs_samples))
                if sim_result.bullpen_runs_samples is not None else np.nan
            ),
            "actual_total_runs": actual_total_runs,
            "runs_crps": runs_crps,
            **game_actuals,
        }

        # Per-stat p_over columns. We tag with the stat so this dataframe
        # has p_k_over_X.Y, p_bb_over_X.Y, etc. at every half-line. The
        # dashboard's game_props uses one row per (player, stat) — this
        # single-row-per-pitcher format is exploded when validation joins
        # by stat.
        for stat_key, over_df in (
            ("k", k_over), ("bb", bb_over), ("h", h_over),
            ("hr", hr_over), ("outs", outs_over),
        ):
            for _, prow in over_df.iterrows():
                col = f"p_{stat_key}_over_{prow['line']:.1f}"
                rec[col] = round(float(prow["p_over"]), 4)

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
    k_bin = compute_per_line_binary_metrics(predictions, "actual_k", k_lines)
    metrics["k_brier_scores"] = k_bin["brier_scores"]
    if k_bin["brier_scores"]:
        metrics["k_avg_brier"] = k_bin["avg_brier"]
    metrics["k_log_losses"] = k_bin["log_losses"]
    if k_bin["log_losses"]:
        metrics["k_avg_log_loss"] = k_bin["avg_log_loss"]

    # Outs Brier scores and log loss
    outs_lines = [14.5, 15.5, 16.5, 17.5, 18.5]
    if "actual_outs" in predictions.columns:
        outs_bin = compute_per_line_binary_metrics(
            predictions, "actual_outs", outs_lines,
            prob_col_fn=lambda line: f"p_outs_over_{line:.1f}".replace(".", "_"),
        )
        metrics["outs_brier_scores"] = outs_bin["brier_scores"]
        if outs_bin["brier_scores"]:
            metrics["outs_avg_brier"] = outs_bin["avg_brier"]
        metrics["outs_log_losses"] = outs_bin["log_losses"]
        if outs_bin["log_losses"]:
            metrics["outs_avg_log_loss"] = outs_bin["avg_log_loss"]
    else:
        metrics["outs_brier_scores"] = {}
        metrics["outs_log_losses"] = {}

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
        for pct, val in compute_coverage_levels(
            predictions["actual_k"].values,
            predictions["expected_k"].values,
            predictions["std_k"].values,
        ).items():
            metrics[f"k_coverage_{pct}"] = val

    # Outs coverage
    if "std_outs" in predictions.columns and "actual_outs" in predictions.columns:
        for pct, val in compute_coverage_levels(
            predictions["actual_outs"].values,
            predictions["expected_outs"].values,
            predictions["std_outs"].values,
        ).items():
            metrics[f"outs_coverage_{pct}"] = val

    # Run-level metrics (full game, starter + bullpen tail). Requires
    # actual_total_runs joined from fact_game_totals.
    if (
        "expected_runs" in predictions.columns
        and "actual_total_runs" in predictions.columns
    ):
        run_mask = predictions["actual_total_runs"].notna()
        if run_mask.sum() > 0:
            rdf = predictions[run_mask]
            expected = rdf["expected_runs"].values.astype(float)
            actual = rdf["actual_total_runs"].values.astype(float)
            errors = expected - actual

            metrics["runs_n"] = int(len(rdf))
            metrics["runs_bias"] = float(np.mean(errors))
            metrics["runs_mae"] = float(np.mean(np.abs(errors)))
            metrics["runs_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
            if np.std(actual) > 0 and np.std(expected) > 0:
                metrics["runs_corr"] = float(np.corrcoef(actual, expected)[0, 1])

            if "runs_crps" in rdf.columns:
                crps_vals = rdf["runs_crps"].dropna().values
                if len(crps_vals) > 0:
                    metrics["runs_crps_mean"] = float(np.mean(crps_vals))
                    metrics["runs_crps_median"] = float(np.median(crps_vals))

            # Coverage via std-based CI (Normal approximation)
            if "std_runs" in rdf.columns:
                std_runs = rdf["std_runs"].values.astype(float)
                for pct, val in compute_coverage_levels(actual, expected, std_runs).items():
                    metrics[f"runs_coverage_{pct}"] = val

            # Coverage via empirical quantile bands (q10/q90 = 80%)
            if "runs_q10" in rdf.columns and "runs_q90" in rdf.columns:
                q10 = rdf["runs_q10"].values.astype(float)
                q90 = rdf["runs_q90"].values.astype(float)
                metrics["runs_coverage_q80_empirical"] = float(
                    np.mean((actual >= q10) & (actual <= q90))
                )

            # Decomposition: how much of the prediction comes from each phase
            if "expected_starter_runs" in rdf.columns:
                sr = rdf["expected_starter_runs"].dropna()
                if len(sr) > 0:
                    metrics["expected_starter_runs_mean"] = float(sr.mean())
            if "expected_bullpen_runs" in rdf.columns:
                br = rdf["expected_bullpen_runs"].dropna()
                if len(br) > 0:
                    metrics["expected_bullpen_runs_mean"] = float(br.mean())

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

        # Run-level metrics (new — game total runs vs actual)
        for key in (
            "runs_n", "runs_bias", "runs_mae", "runs_rmse", "runs_corr",
            "runs_crps_mean", "runs_crps_median",
            "runs_coverage_50", "runs_coverage_80", "runs_coverage_90",
            "runs_coverage_q80_empirical",
            "expected_starter_runs_mean", "expected_bullpen_runs_mean",
        ):
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
