"""
Unified game prop backtest framework.

Generalizes the K-only game_k_validation.py to support any pitcher or batter
prop: K, BB, HR, H, Outs.  Each stat is configured via a GamePropConfig
dataclass that captures the rate column, model type, matchup metric, etc.

Walk-forward backtesting, comprehensive metrics (CRPS, ECE, MCE, temperature),
and isotonic recalibration are all supported.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.data.feature_eng import (
    build_multi_season_hitter_data,
    build_multi_season_pitcher_data,
    get_cached_batter_game_logs,
    get_cached_game_batter_stats,
    get_cached_game_lineups,
    get_cached_pitcher_game_logs,
    get_hitter_vulnerability,
    get_pitcher_arsenal,
)
from src.evaluation.metrics import (
    compute_crps_single,
    compute_ece,
    compute_mce,
    compute_temperature,
)
from src.models.bf_model import (
    compute_batter_pa_priors,
    compute_pitcher_bf_priors,
    get_bf_distribution,
    get_pa_distribution,
)
from src.models.game_k_model import (
    compute_over_probs,
    predict_batter_game,
    predict_game_batch_stat,
)
from src.models.hitter_model import (
    STAT_CONFIGS as HITTER_STAT_CONFIGS,
    extract_rate_samples as extract_hitter_rate_samples,
    fit_hitter_model,
    prepare_hitter_data,
)
from src.models.matchup import score_matchup_for_stat
from src.models.pitcher_model import (
    PITCHER_STAT_CONFIGS,
    extract_rate_samples as extract_pitcher_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. GamePropConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class GamePropConfig:
    """Configuration for a single game-level prop backtest."""

    stat_name: str           # "k", "bb", "hr", "h", "outs"
    side: str                # "pitcher" or "batter"
    rate_col: str            # posterior or shrinkage rate column
    actual_col: str          # column in boxscores / game logs
    bayesian: bool           # posterior samples vs shrinkage
    model_type: str          # "binomial" or "poisson"
    matchup_metric: str | None  # "whiff_rate", "chase_rate", "barrel_rate", None
    park_adjusted: bool      # apply park factor?
    default_lines: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Config dictionaries
# ---------------------------------------------------------------------------

PITCHER_PROP_CONFIGS: dict[str, GamePropConfig] = {
    "k": GamePropConfig(
        "k", "pitcher", "k_rate", "strike_outs", True, "binomial",
        "whiff_rate", False, [3.5, 4.5, 5.5, 6.5, 7.5],
    ),
    "bb": GamePropConfig(
        "bb", "pitcher", "bb_rate", "walks", True, "binomial",
        "chase_rate", False, [0.5, 1.5, 2.5, 3.5],
    ),
    "hr": GamePropConfig(
        "hr", "pitcher", "hr_per_bf", "home_runs", True, "poisson",
        "barrel_rate", True, [0.5, 1.5],
    ),
    "h": GamePropConfig(
        "h", "pitcher", "h_per_bf", "hits", False, "binomial",
        None, False, [3.5, 4.5, 5.5, 6.5, 7.5],
    ),
    "outs": GamePropConfig(
        "outs", "pitcher", "outs_per_bf", "outs", False, "binomial",
        None, False, [14.5, 15.5, 16.5, 17.5, 18.5],
    ),
}

BATTER_PROP_CONFIGS: dict[str, GamePropConfig] = {
    "k": GamePropConfig(
        "k", "batter", "k_rate", "strikeouts", True, "binomial",
        "whiff_rate", False, [0.5, 1.5, 2.5],
    ),
    "bb": GamePropConfig(
        "bb", "batter", "bb_rate", "walks", True, "binomial",
        "chase_rate", False, [0.5, 1.5],
    ),
    "hr": GamePropConfig(
        "hr", "batter", "hr_rate", "home_runs", True, "poisson",
        "barrel_rate", True, [0.5, 1.5],
    ),
    "h": GamePropConfig(
        "h", "batter", "h_rate", "hits", False, "binomial",
        None, False, [0.5, 1.5, 2.5],
    ),
}


# ---------------------------------------------------------------------------
# 3. _get_pitcher_rate_samples
# ---------------------------------------------------------------------------

# League-average priors for shrinkage stats (natural scale).
# Used for pitcher H/BF and Outs/BF when no Bayesian model is trained.
_PITCHER_SHRINKAGE_PRIORS: dict[str, tuple[float, float]] = {
    # (league_avg_rate, effective_sample_size for prior)
    "h_per_bf": (0.230, 80),   # ~23% H/BF league avg
    "outs_per_bf": (0.640, 80),  # ~64% Outs/BF league avg
}

# Batter shrinkage priors
_BATTER_SHRINKAGE_PRIORS: dict[str, tuple[float, float]] = {
    "h_rate": (0.250, 60),   # ~25% H/PA (hits per PA, not batting avg)
}


def _get_pitcher_rate_samples(
    config: GamePropConfig,
    pitcher_id: int,
    season_totals: pd.DataFrame,
    posteriors: dict[str, dict[int, np.ndarray]] | None = None,
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Get rate samples for a pitcher stat.

    For Bayesian stats: look up in posteriors dict.
    For shrinkage stats: compute Beta posterior from observed rate + shrinkage.

    Parameters
    ----------
    config : GamePropConfig
        Prop configuration.
    pitcher_id : int
        MLB pitcher ID.
    season_totals : pd.DataFrame
        Pitcher season totals with rate columns and batters_faced.
    posteriors : dict or None
        Mapping {stat_name: {pitcher_id: samples}}.
    n_samples : int
        Number of samples to draw for shrinkage stats.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    np.ndarray or None
        Rate samples in [0, 1], or None if pitcher not found.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if config.bayesian:
        # Look up in posteriors dict
        if posteriors is None:
            return None
        stat_posteriors = posteriors.get(config.rate_col)
        if stat_posteriors is None:
            return None
        return stat_posteriors.get(pitcher_id)

    # Shrinkage: compute Beta posterior from observed rate
    row = season_totals[season_totals["pitcher_id"] == pitcher_id]
    if row.empty:
        return None

    row = row.iloc[0]
    bf = float(row.get("batters_faced", 0))
    if bf < 1:
        return None

    # Compute observed counts
    prior_info = _PITCHER_SHRINKAGE_PRIORS.get(config.rate_col)
    if prior_info is None:
        return None
    league_avg, prior_n = prior_info

    # Map rate_col to count column
    _rate_to_count: dict[str, str] = {
        "h_per_bf": "hits",
        "outs_per_bf": "outs",
    }
    count_col = _rate_to_count.get(config.rate_col)
    if count_col is None or count_col not in row.index:
        # Try computing from rate
        if config.rate_col in row.index:
            observed_rate = float(row[config.rate_col])
            successes = observed_rate * bf
            failures = bf - successes
        else:
            return None
    else:
        successes = float(row[count_col])
        failures = bf - successes

    # Beta prior parameters from league average and effective sample size
    alpha_pop = league_avg * prior_n
    beta_pop = (1 - league_avg) * prior_n

    # Posterior: Beta(alpha_pop + successes, beta_pop + failures)
    alpha_post = alpha_pop + successes
    beta_post = beta_pop + failures

    # Ensure valid parameters
    alpha_post = max(alpha_post, 0.1)
    beta_post = max(beta_post, 0.1)

    return rng.beta(alpha_post, beta_post, size=n_samples)


# ---------------------------------------------------------------------------
# 4. _get_batter_rate_samples
# ---------------------------------------------------------------------------

def _get_batter_rate_samples(
    config: GamePropConfig,
    batter_id: int,
    season_totals: pd.DataFrame,
    posteriors: dict[str, dict[int, np.ndarray]] | None = None,
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Get rate samples for a batter stat.

    For Bayesian stats: look up in posteriors dict.
    For shrinkage stats: compute Beta posterior from observed rate + shrinkage.

    Parameters
    ----------
    config : GamePropConfig
        Prop configuration.
    batter_id : int
        MLB batter ID.
    season_totals : pd.DataFrame
        Batter season totals with rate columns and pa.
    posteriors : dict or None
        Mapping {stat_name: {batter_id: samples}}.
    n_samples : int
        Number of samples to draw for shrinkage stats.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    np.ndarray or None
        Rate samples in [0, 1], or None if batter not found.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if config.bayesian:
        if posteriors is None:
            return None
        stat_posteriors = posteriors.get(config.rate_col)
        if stat_posteriors is None:
            return None
        return stat_posteriors.get(batter_id)

    # Shrinkage: compute Beta posterior from observed rate
    row = season_totals[season_totals["batter_id"] == batter_id]
    if row.empty:
        return None

    row = row.iloc[0]
    pa = float(row.get("pa", 0))
    if pa < 1:
        return None

    prior_info = _BATTER_SHRINKAGE_PRIORS.get(config.rate_col)
    if prior_info is None:
        return None
    league_avg, prior_n = prior_info

    # Map rate_col to count column
    _rate_to_count: dict[str, str] = {
        "h_rate": "hits",
    }
    count_col = _rate_to_count.get(config.rate_col)
    if count_col is None or count_col not in row.index:
        if config.rate_col in row.index:
            observed_rate = float(row[config.rate_col])
            successes = observed_rate * pa
            failures = pa - successes
        else:
            return None
    else:
        successes = float(row[count_col])
        failures = pa - successes

    alpha_pop = league_avg * prior_n
    beta_pop = (1 - league_avg) * prior_n

    alpha_post = max(alpha_pop + successes, 0.1)
    beta_post = max(beta_pop + failures, 0.1)

    return rng.beta(alpha_post, beta_post, size=n_samples)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_baselines_pt(
    pitcher_arsenal: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Build league-average baselines per pitch type from arsenal data.

    Computes whiff_rate, chase_rate, and barrel_rate per pitch type.

    Parameters
    ----------
    pitcher_arsenal : pd.DataFrame
        Full pitcher arsenal profiles for a season.

    Returns
    -------
    dict[str, dict[str, float]]
        {pitch_type: {"whiff_rate": float, "chase_rate": float, ...}}.
    """
    baselines: dict[str, dict[str, float]] = {}

    # Whiff rate
    agg = pitcher_arsenal.groupby("pitch_type").agg(
        total_whiffs=("whiffs", "sum"),
        total_swings=("swings", "sum"),
    ).reset_index()
    agg["whiff_rate"] = agg["total_whiffs"] / agg["total_swings"].replace(0, np.nan)

    for _, row in agg.iterrows():
        pt = row["pitch_type"]
        wr = row["whiff_rate"]
        if pd.notna(wr):
            baselines.setdefault(pt, {})["whiff_rate"] = float(wr)

    # Chase rate (if columns exist)
    if "chase_swings" in pitcher_arsenal.columns and "pitches_out_zone" in pitcher_arsenal.columns:
        agg_chase = pitcher_arsenal.groupby("pitch_type").agg(
            total_chase=("chase_swings", "sum"),
            total_out_zone=("pitches_out_zone", "sum"),
        ).reset_index()
        agg_chase["chase_rate"] = (
            agg_chase["total_chase"] / agg_chase["total_out_zone"].replace(0, np.nan)
        )
        for _, row in agg_chase.iterrows():
            pt = row["pitch_type"]
            cr = row["chase_rate"]
            if pd.notna(cr):
                baselines.setdefault(pt, {})["chase_rate"] = float(cr)

    # Barrel rate (if columns exist)
    if "barrels" in pitcher_arsenal.columns and "bip" in pitcher_arsenal.columns:
        agg_barrel = pitcher_arsenal.groupby("pitch_type").agg(
            total_barrels=("barrels", "sum"),
            total_bip=("bip", "sum"),
        ).reset_index()
        agg_barrel["barrel_rate"] = (
            agg_barrel["total_barrels"] / agg_barrel["total_bip"].replace(0, np.nan)
        )
        for _, row in agg_barrel.iterrows():
            pt = row["pitch_type"]
            br = row["barrel_rate"]
            if pd.notna(br):
                baselines.setdefault(pt, {})["barrel_rate"] = float(br)

    return baselines


def _extract_all_pitcher_posteriors(
    config: GamePropConfig,
    train_seasons: list[int],
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> dict[str, dict[int, np.ndarray]]:
    """Train model and extract posteriors for all Bayesian pitcher stats.

    Parameters
    ----------
    config : GamePropConfig
        Prop configuration (must have bayesian=True, side="pitcher").
    train_seasons : list[int]
        Training seasons.
    draws, tune, chains, random_seed : int
        MCMC parameters.

    Returns
    -------
    dict
        {rate_col: {pitcher_id: samples_array}}.
    """
    last_train = max(train_seasons)
    df_model = build_multi_season_pitcher_data(train_seasons, min_bf=10)

    # Map config rate_col to the PITCHER_STAT_CONFIGS key
    stat_key = config.rate_col
    if stat_key not in PITCHER_STAT_CONFIGS:
        logger.error("No pitcher stat config for %s", stat_key)
        return {}

    data = prepare_pitcher_data(df_model, stat=stat_key)
    _, trace = fit_pitcher_model(
        data, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    # Extract posteriors for each pitcher in the last training season
    df = data["df"]
    pitcher_ids = df[df["season"] == last_train]["pitcher_id"].unique()

    posteriors: dict[int, np.ndarray] = {}
    for pid in pitcher_ids:
        try:
            samples = extract_pitcher_rate_samples(
                trace, data, pid, last_train,
                project_forward=True, random_seed=random_seed,
            )
            posteriors[pid] = samples
        except (ValueError, KeyError):
            continue

    logger.info(
        "Extracted pitcher %s posteriors for %d pitchers",
        stat_key, len(posteriors),
    )
    return {stat_key: posteriors}


def _extract_all_batter_posteriors(
    config: GamePropConfig,
    train_seasons: list[int],
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> dict[str, dict[int, np.ndarray]]:
    """Train model and extract posteriors for a Bayesian batter stat.

    Parameters
    ----------
    config : GamePropConfig
        Prop configuration (must have bayesian=True, side="batter").
    train_seasons : list[int]
        Training seasons.
    draws, tune, chains, random_seed : int
        MCMC parameters.

    Returns
    -------
    dict
        {rate_col: {batter_id: samples_array}}.
    """
    last_train = max(train_seasons)
    df_model = build_multi_season_hitter_data(train_seasons, min_pa=10)

    stat_key = config.rate_col
    if stat_key not in HITTER_STAT_CONFIGS:
        logger.error("No hitter stat config for %s", stat_key)
        return {}

    data = prepare_hitter_data(df_model, stat=stat_key)
    _, trace = fit_hitter_model(
        data, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    df = data["df"]
    batter_ids = df[df["season"] == last_train]["batter_id"].unique()

    posteriors: dict[int, np.ndarray] = {}
    for bid in batter_ids:
        try:
            samples = extract_hitter_rate_samples(
                trace, data, bid, last_train,
                project_forward=True, random_seed=random_seed,
            )
            posteriors[bid] = samples
        except (ValueError, KeyError):
            continue

    logger.info(
        "Extracted batter %s posteriors for %d batters",
        stat_key, len(posteriors),
    )
    return {stat_key: posteriors}


def _load_pitcher_season_totals(seasons: list[int]) -> pd.DataFrame:
    """Load and combine pitcher season totals across seasons.

    Computes derived rate columns (h_per_bf, outs_per_bf) for shrinkage
    stats that don't have Bayesian models.

    Parameters
    ----------
    seasons : list[int]
        Seasons to load.

    Returns
    -------
    pd.DataFrame
        Combined pitcher season totals.
    """
    df = build_multi_season_pitcher_data(seasons, min_bf=1)

    # Add derived rate columns for shrinkage stats
    if "hits" in df.columns and "batters_faced" in df.columns:
        bf = df["batters_faced"].replace(0, np.nan)
        df["h_per_bf"] = df["hits"] / bf
    elif "h_per_bf" not in df.columns:
        # Try from boxscore-level data
        pass

    if "outs" not in df.columns:
        # Compute from IP: outs = IP * 3
        if "ip" in df.columns:
            df["outs"] = df["ip"] * 3
    if "outs" in df.columns and "batters_faced" in df.columns:
        bf = df["batters_faced"].replace(0, np.nan)
        df["outs_per_bf"] = df["outs"] / bf

    return df


def _load_batter_season_totals(seasons: list[int]) -> pd.DataFrame:
    """Load and combine batter season totals across seasons.

    Computes derived rate columns (h_rate) for shrinkage stats.

    Parameters
    ----------
    seasons : list[int]
        Seasons to load.

    Returns
    -------
    pd.DataFrame
        Combined batter season totals.
    """
    df = build_multi_season_hitter_data(seasons, min_pa=1)

    # Add derived rate columns for shrinkage stats
    if "hits" in df.columns and "pa" in df.columns:
        pa = df["pa"].replace(0, np.nan)
        df["h_rate"] = df["hits"] / pa

    return df


# ---------------------------------------------------------------------------
# 5. build_game_prop_predictions — generic prediction builder
# ---------------------------------------------------------------------------

def build_game_prop_predictions(
    config: GamePropConfig,
    train_seasons: list[int],
    test_season: int,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    min_bf_game: int = 15,
    min_pa_game: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build game-level predictions for a single train -> test fold.

    For pitcher props: trains model, extracts posteriors, uses
    predict_game_batch_stat.
    For batter props: trains model, extracts posteriors, predicts
    per-batter per-game.

    Parameters
    ----------
    config : GamePropConfig
        Prop configuration.
    train_seasons : list[int]
        Seasons for training.
    test_season : int
        Season to predict.
    draws : int
        MCMC posterior draws.
    tune : int
        MCMC tuning steps.
    chains : int
        MCMC chains.
    n_mc_draws : int
        Monte Carlo draws per game.
    min_bf_game : int
        Minimum BF in a game to include (pitcher side).
    min_pa_game : int
        Minimum PA in a game to include (batter side).
    random_seed : int
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per game/appearance with predictions and actuals.
    """
    logger.info(
        "Building %s %s predictions: train=%s, test=%d",
        config.side, config.stat_name, train_seasons, test_season,
    )

    if config.side == "pitcher":
        return _build_pitcher_prop_predictions(
            config, train_seasons, test_season,
            draws, tune, chains, n_mc_draws, min_bf_game, random_seed,
        )
    else:
        return _build_batter_prop_predictions(
            config, train_seasons, test_season,
            draws, tune, chains, n_mc_draws, min_pa_game, random_seed,
        )


def _build_pitcher_prop_predictions(
    config: GamePropConfig,
    train_seasons: list[int],
    test_season: int,
    draws: int,
    tune: int,
    chains: int,
    n_mc_draws: int,
    min_bf_game: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build predictions for a pitcher prop.

    Mirrors build_game_k_predictions but generalized for any stat.
    """
    last_train = max(train_seasons)
    rng = np.random.default_rng(random_seed)

    # 1. Get rate posteriors/samples
    posteriors: dict[str, dict[int, np.ndarray]] | None = None
    season_totals: pd.DataFrame | None = None

    if config.bayesian:
        posteriors = _extract_all_pitcher_posteriors(
            config, train_seasons, draws, tune, chains, random_seed,
        )
        if not posteriors or config.rate_col not in posteriors:
            logger.warning("No posteriors extracted for %s", config.rate_col)
            return pd.DataFrame()
        pitcher_posteriors = posteriors[config.rate_col]
    else:
        # Shrinkage: load season totals for the last training season
        season_totals = _load_pitcher_season_totals(train_seasons)
        # Filter to last training season for rate priors
        last_season_totals = season_totals[
            season_totals["season"] == last_train
        ].copy()

        # Build shrinkage posteriors for each pitcher
        pitcher_posteriors = {}
        for pid in last_season_totals["pitcher_id"].unique():
            samples = _get_pitcher_rate_samples(
                config, pid, last_season_totals,
                n_samples=n_mc_draws, rng=rng,
            )
            if samples is not None:
                pitcher_posteriors[pid] = samples

    logger.info(
        "Rate samples for %d pitchers (%s, bayesian=%s)",
        len(pitcher_posteriors), config.rate_col, config.bayesian,
    )

    # 2. Build BF priors from training seasons
    game_logs_frames = []
    for s in train_seasons:
        game_logs_frames.append(get_cached_pitcher_game_logs(s))
    all_game_logs = pd.concat(game_logs_frames, ignore_index=True)
    bf_priors = compute_pitcher_bf_priors(all_game_logs)

    # Remap to test season
    bf_last = bf_priors[bf_priors["season"] == last_train].copy()
    bf_last["season"] = test_season
    bf_priors = pd.concat([bf_priors, bf_last], ignore_index=True)

    # 3. Load test season game logs
    test_game_logs = get_cached_pitcher_game_logs(test_season)
    test_game_logs = test_game_logs[
        test_game_logs["is_starter"] == True  # noqa: E712
    ].copy()
    test_game_logs = test_game_logs[
        test_game_logs["batters_faced"] >= min_bf_game
    ].copy()
    test_game_logs["season"] = test_season

    # 4. Load matchup data from last training season
    pitcher_arsenal = get_pitcher_arsenal(last_train)
    hitter_vuln = get_hitter_vulnerability(last_train)
    baselines_pt = _build_baselines_pt(pitcher_arsenal)

    # Lineup data from test season (public pre-game)
    game_lineups = get_cached_game_lineups(test_season)
    game_batter_stats = get_cached_game_batter_stats(test_season)

    # 5. Context lifts (umpire + weather, from game_k_validation helpers)
    from src.evaluation.game_k_validation import (
        _build_umpire_lift_lookup,
        _build_weather_lift_lookup,
    )
    umpire_lifts = _build_umpire_lift_lookup(train_seasons, test_season)
    weather_lifts = _build_weather_lift_lookup(train_seasons, test_season)

    # Combine context lifts
    context_lifts: dict[int, float] = {}
    all_gpks = set(umpire_lifts.keys()) | set(weather_lifts.keys())
    for gpk in all_gpks:
        context_lifts[gpk] = umpire_lifts.get(gpk, 0.0) + weather_lifts.get(gpk, 0.0)

    # 6. Park factors (for HR)
    park_factors: dict[int, float] | None = None
    if config.park_adjusted:
        park_factors = _build_park_factor_lookup(train_seasons, test_season)

    # 7. Batch predict
    predictions = predict_game_batch_stat(
        stat_name=config.stat_name,
        game_records=test_game_logs,
        pitcher_posteriors=pitcher_posteriors,
        bf_priors=bf_priors,
        pitcher_arsenal=pitcher_arsenal,
        hitter_vuln=hitter_vuln,
        baselines_pt=baselines_pt,
        game_batter_ks=game_batter_stats,
        game_lineups=game_lineups,
        context_lifts=context_lifts,
        park_factors=park_factors,
        model_type=config.model_type,
        default_lines=config.default_lines if config.default_lines else None,
        actual_col=config.actual_col,
        n_draws=n_mc_draws,
    )

    sn = config.stat_name.lower()
    logger.info(
        "Pitcher %s predictions: %d games, mean expected=%.2f, mean actual=%.2f",
        sn, len(predictions),
        predictions[f"expected_{sn}"].mean() if len(predictions) else 0,
        predictions[f"actual_{sn}"].mean() if len(predictions) else 0,
    )

    return predictions


def _build_park_factor_lookup(
    train_seasons: list[int],
    test_season: int,
) -> dict[int, float]:
    """Build game_pk -> park factor for the test season.

    Uses park factors computed from training seasons if available.
    Falls back to 1.0 (no adjustment) if park factor data is unavailable.

    Parameters
    ----------
    train_seasons : list[int]
        Training seasons.
    test_season : int
        Test season.

    Returns
    -------
    dict[int, float]
        {game_pk: park_factor}.
    """
    try:
        from src.data.db import read_sql

        # Try loading park factors from database
        pf_data = read_sql(f"""
            SELECT dg.game_pk, dg.venue_id
            FROM production.dim_game dg
            WHERE dg.season = {int(test_season)}
              AND dg.game_type = 'R'
        """, {})

        if pf_data.empty:
            return {}

        # Compute park factors from training data
        train_str = ", ".join(str(s) for s in train_seasons)
        venue_hr = read_sql(f"""
            SELECT dg.venue_id,
                   SUM(CASE WHEN fpa.events = 'home_run' THEN 1 ELSE 0 END)::float
                       / NULLIF(COUNT(*), 0) AS hr_rate
            FROM production.fact_pa fpa
            JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
            WHERE dg.season IN ({train_str})
              AND dg.game_type = 'R'
              AND fpa.events IS NOT NULL
            GROUP BY dg.venue_id
            HAVING COUNT(*) >= 1000
        """, {})

        if venue_hr.empty:
            return {}

        league_hr = float(venue_hr["hr_rate"].mean())
        if league_hr <= 0:
            return {}

        venue_pf = dict(zip(
            venue_hr["venue_id"],
            venue_hr["hr_rate"] / league_hr,
        ))

        result: dict[int, float] = {}
        for _, row in pf_data.iterrows():
            gpk = int(row["game_pk"])
            vid = row.get("venue_id")
            if vid is not None and vid in venue_pf:
                result[gpk] = float(venue_pf[vid])

        logger.info(
            "Park factors: %d games mapped, range [%.2f, %.2f]",
            len(result),
            min(result.values()) if result else 1.0,
            max(result.values()) if result else 1.0,
        )
        return result

    except Exception as e:
        logger.warning("Could not build park factors: %s", e)
        return {}


def _build_batter_prop_predictions(
    config: GamePropConfig,
    train_seasons: list[int],
    test_season: int,
    draws: int,
    tune: int,
    chains: int,
    n_mc_draws: int,
    min_pa_game: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build predictions for a batter prop.

    For each batter-game in the test season, predict the stat total
    using rate posteriors, PA priors, and matchup scoring.
    """
    last_train = max(train_seasons)
    rng = np.random.default_rng(random_seed)

    # 1. Get rate posteriors/samples
    posteriors: dict[str, dict[int, np.ndarray]] | None = None
    season_totals: pd.DataFrame | None = None

    if config.bayesian:
        posteriors = _extract_all_batter_posteriors(
            config, train_seasons, draws, tune, chains, random_seed,
        )
        if not posteriors or config.rate_col not in posteriors:
            logger.warning("No posteriors extracted for %s", config.rate_col)
            return pd.DataFrame()
        batter_posteriors = posteriors[config.rate_col]
    else:
        season_totals = _load_batter_season_totals(train_seasons)
        last_season_totals = season_totals[
            season_totals["season"] == last_train
        ].copy()

        batter_posteriors = {}
        for bid in last_season_totals["batter_id"].unique():
            samples = _get_batter_rate_samples(
                config, bid, last_season_totals,
                n_samples=n_mc_draws, rng=rng,
            )
            if samples is not None:
                batter_posteriors[bid] = samples

    logger.info(
        "Rate samples for %d batters (%s, bayesian=%s)",
        len(batter_posteriors), config.rate_col, config.bayesian,
    )

    # 2. Build PA priors from training batter game logs
    batter_gl_frames = []
    for s in train_seasons:
        batter_gl_frames.append(get_cached_batter_game_logs(s))
    all_batter_gl = pd.concat(batter_gl_frames, ignore_index=True)
    pa_priors = compute_batter_pa_priors(all_batter_gl)

    # Remap to test season
    pa_last = pa_priors[pa_priors["season"] == last_train].copy()
    pa_last["season"] = test_season
    pa_priors = pd.concat([pa_priors, pa_last], ignore_index=True)

    # 3. Load test season batter game logs
    test_batter_gl = get_cached_batter_game_logs(test_season)
    if min_pa_game > 1:
        test_batter_gl = test_batter_gl[
            test_batter_gl["plate_appearances"] >= min_pa_game
        ].copy()

    # 4. Load matchup data from last training season
    pitcher_arsenal = get_pitcher_arsenal(last_train)
    hitter_vuln = get_hitter_vulnerability(last_train)
    baselines_pt = _build_baselines_pt(pitcher_arsenal)

    # 5. Load game lineups for pitcher identification
    game_lineups = get_cached_game_lineups(test_season)

    # Build batter->pitcher mapping per game from game_batter_stats
    game_batter_stats = get_cached_game_batter_stats(test_season)

    # Build (game_pk, batter_id) -> pitcher_id lookup from game_batter_stats
    batter_pitcher_lookup: dict[tuple[int, int], int] = {}
    if game_batter_stats is not None and not game_batter_stats.empty:
        for _, row in game_batter_stats.iterrows():
            key = (int(row["game_pk"]), int(row["batter_id"]))
            # Use the pitcher who faced this batter (take first/primary)
            if key not in batter_pitcher_lookup:
                batter_pitcher_lookup[key] = int(row["pitcher_id"])

    # 6. Context lifts
    from src.evaluation.game_k_validation import (
        _build_umpire_lift_lookup,
        _build_weather_lift_lookup,
    )
    umpire_lifts = _build_umpire_lift_lookup(train_seasons, test_season)
    weather_lifts = _build_weather_lift_lookup(train_seasons, test_season)

    # Park factors
    park_factors: dict[int, float] | None = None
    if config.park_adjusted:
        park_factors = _build_park_factor_lookup(train_seasons, test_season)

    # 7. Predict per batter-game
    sn = config.stat_name.lower()
    records: list[dict[str, Any]] = []
    n_total = len(test_batter_gl)
    n_predicted = 0
    n_skipped_no_rate = 0
    n_skipped_no_pitcher = 0

    # Map actual column from batter game log columns
    actual_col = config.actual_col

    for i, (_, game_row) in enumerate(test_batter_gl.iterrows()):
        batter_id = int(game_row["batter_id"])
        game_pk = int(game_row["game_pk"])

        # Get rate samples
        if batter_id not in batter_posteriors:
            n_skipped_no_rate += 1
            continue

        rate_samples = batter_posteriors[batter_id]

        # Get opposing pitcher
        pitcher_id = batter_pitcher_lookup.get((game_pk, batter_id))
        if pitcher_id is None:
            n_skipped_no_pitcher += 1
            continue

        # PA distribution
        pa_info = get_pa_distribution(batter_id, test_season, pa_priors)
        pa_mu = pa_info["mu_pa"]
        pa_sigma = pa_info["sigma_pa"]

        # Context
        context_lift = (
            umpire_lifts.get(game_pk, 0.0) + weather_lifts.get(game_pk, 0.0)
        )
        pf = 1.0
        if park_factors is not None:
            pf = park_factors.get(game_pk, 1.0)

        # Predict
        pred = predict_batter_game(
            stat_name=sn,
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            rate_samples=rate_samples,
            pa_mu=pa_mu,
            pa_sigma=pa_sigma,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
            context_logit_lift=context_lift,
            park_factor=pf,
            model_type=config.model_type,
            default_lines=config.default_lines if config.default_lines else None,
            n_draws=n_mc_draws,
            random_seed=random_seed + i,  # Vary seed per game
        )

        # Actual stat
        actual_val = game_row.get(actual_col, np.nan)
        if pd.isna(actual_val):
            continue

        # Build record
        rec: dict[str, Any] = {
            "game_pk": game_pk,
            "batter_id": batter_id,
            "pitcher_id": pitcher_id,
            "season": test_season,
            f"actual_{sn}": int(actual_val),
            "actual_pa": int(game_row.get("plate_appearances", 0)),
            f"expected_{sn}": pred[f"expected_{sn}"],
            f"std_{sn}": pred[f"std_{sn}"],
            "pa_mu": pred["pa_mu"],
            "pa_sigma": pred["pa_sigma"],
            "matchup_logit_lift": pred["matchup_logit_lift"],
            f"batter_{config.rate_col}_mean": float(np.mean(rate_samples)),
        }

        # Add p_over columns from over_probs
        over_probs = pred.get("over_probs")
        if over_probs is not None and isinstance(over_probs, pd.DataFrame):
            for _, op_row in over_probs.iterrows():
                line = op_row["line"]
                col_name = f"p_over_{line:.1f}".replace(".", "_")
                rec[col_name] = op_row["p_over"]

        records.append(rec)
        n_predicted += 1

        if (i + 1) % 5000 == 0:
            logger.info(
                "Batter %s progress: %d/%d processed, %d predicted",
                sn, i + 1, n_total, n_predicted,
            )

    logger.info(
        "Batter %s predictions: %d predicted, %d skipped (no rate), "
        "%d skipped (no pitcher), %d total",
        sn, n_predicted, n_skipped_no_rate, n_skipped_no_pitcher, n_total,
    )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. compute_game_prop_metrics — comprehensive metrics
# ---------------------------------------------------------------------------

def compute_game_prop_metrics(
    config: GamePropConfig,
    predictions: pd.DataFrame,
    stat_samples: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Parameters
    ----------
    config : GamePropConfig
        Prop configuration.
    predictions : pd.DataFrame
        Predictions from build_game_prop_predictions.
    stat_samples : dict or None
        Optional mapping of row index -> MC samples for CRPS computation.

    Returns
    -------
    dict
        Keys: rmse, mae, brier_scores (per line), avg_brier, crps,
        ece, mce, temperature, coverage_50/80/90/95, n_games.
    """
    sn = config.stat_name.lower()
    expected_col = f"expected_{sn}"
    actual_col = f"actual_{sn}"
    std_col = f"std_{sn}"
    lines = config.default_lines if config.default_lines else [0.5, 1.5, 2.5]

    n_games = len(predictions)
    if n_games == 0:
        return _empty_metrics()

    if expected_col not in predictions.columns or actual_col not in predictions.columns:
        logger.warning(
            "Missing columns: expected=%s actual=%s in predictions",
            expected_col, actual_col,
        )
        return _empty_metrics()

    # RMSE and MAE
    errors = predictions[expected_col].values - predictions[actual_col].values
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # Brier scores per line
    brier_scores: dict[float, float] = {}
    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            continue
        y_true = (predictions[actual_col] > line).astype(float).values
        y_prob = np.clip(predictions[col].values, 0, 1)
        # Skip if all same class
        if y_true.std() == 0:
            continue
        brier_scores[line] = float(brier_score_loss(y_true, y_prob))

    avg_brier = float(np.mean(list(brier_scores.values()))) if brier_scores else np.nan

    # CRPS (if samples available)
    crps = np.nan
    if stat_samples is not None and len(stat_samples) > 0:
        crps_values = []
        for idx, row in predictions.iterrows():
            if idx in stat_samples:
                samples = stat_samples[idx]
                actual = row[actual_col]
                crps_values.append(compute_crps_single(float(actual), samples))
        if crps_values:
            crps = float(np.mean(crps_values))

    # ECE, MCE, Temperature — compute for each line and average
    ece_per_line: dict[float, float] = {}
    mce_per_line: dict[float, float] = {}
    temp_per_line: dict[float, float] = {}

    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            continue
        y_true = (predictions[actual_col] > line).astype(float).values
        y_prob = np.clip(predictions[col].values, 0, 1)
        if y_true.std() == 0:
            continue

        ece_per_line[line] = compute_ece(y_prob, y_true)
        mce_per_line[line] = compute_mce(y_prob, y_true)
        temp_per_line[line] = compute_temperature(y_prob, y_true)

    avg_ece = float(np.mean(list(ece_per_line.values()))) if ece_per_line else np.nan
    avg_mce = float(np.mean(list(mce_per_line.values()))) if mce_per_line else np.nan
    avg_temp = float(np.mean(list(temp_per_line.values()))) if temp_per_line else np.nan

    # Coverage intervals
    coverage_50 = np.nan
    coverage_80 = np.nan
    coverage_90 = np.nan
    coverage_95 = np.nan

    if std_col in predictions.columns:
        expected = predictions[expected_col].values
        std = predictions[std_col].values
        actual = predictions[actual_col].values

        def _coverage(z: float) -> float:
            lo = expected - z * std
            hi = expected + z * std
            return float(np.mean((actual >= lo) & (actual <= hi)))

        coverage_50 = _coverage(0.6745)
        coverage_80 = _coverage(1.2816)
        coverage_90 = _coverage(1.6449)
        coverage_95 = _coverage(1.9600)

    return {
        "stat_name": config.stat_name,
        "side": config.side,
        "rmse": rmse,
        "mae": mae,
        "brier_scores": brier_scores,
        "avg_brier": avg_brier,
        "crps": crps,
        "ece": avg_ece,
        "ece_per_line": ece_per_line,
        "mce": avg_mce,
        "mce_per_line": mce_per_line,
        "temperature": avg_temp,
        "temperature_per_line": temp_per_line,
        "coverage_50": coverage_50,
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "coverage_95": coverage_95,
        "n_games": n_games,
    }


def _empty_metrics() -> dict[str, Any]:
    """Return an empty metrics dict."""
    return {
        "stat_name": "",
        "side": "",
        "rmse": np.nan,
        "mae": np.nan,
        "brier_scores": {},
        "avg_brier": np.nan,
        "crps": np.nan,
        "ece": np.nan,
        "ece_per_line": {},
        "mce": np.nan,
        "mce_per_line": {},
        "temperature": np.nan,
        "temperature_per_line": {},
        "coverage_50": np.nan,
        "coverage_80": np.nan,
        "coverage_90": np.nan,
        "coverage_95": np.nan,
        "n_games": 0,
    }


# ---------------------------------------------------------------------------
# 7. run_full_game_prop_backtest — multi-fold runner
# ---------------------------------------------------------------------------

def run_full_game_prop_backtest(
    config: GamePropConfig,
    folds: list[tuple[list[int], int]] | None = None,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    n_mc_draws: int = 2000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward backtest across multiple folds.

    Parameters
    ----------
    config : GamePropConfig
        Prop configuration.
    folds : list or None
        List of (train_seasons, test_season) tuples.
        Default: 3 walk-forward folds.
    draws, tune, chains : int
        MCMC parameters.
    n_mc_draws : int
        Monte Carlo draws per game.
    random_seed : int
        For reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (summary_df, all_predictions_df).
        summary_df has one row per fold with all metrics.
        all_predictions_df has all individual predictions.
    """
    if folds is None:
        folds = [
            ([2020, 2021, 2022], 2023),
            ([2020, 2021, 2022, 2023], 2024),
            ([2020, 2021, 2022, 2023, 2024], 2025),
        ]

    fold_results: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []

    for train_seasons, test_season in folds:
        logger.info("=" * 60)
        logger.info(
            "Fold [%s %s]: train=%s -> test=%d",
            config.side, config.stat_name, train_seasons, test_season,
        )

        predictions = build_game_prop_predictions(
            config=config,
            train_seasons=train_seasons,
            test_season=test_season,
            draws=draws,
            tune=tune,
            chains=chains,
            n_mc_draws=n_mc_draws,
            random_seed=random_seed,
        )

        if len(predictions) == 0:
            logger.warning(
                "No predictions for fold train=%s test=%d",
                train_seasons, test_season,
            )
            continue

        metrics = compute_game_prop_metrics(config, predictions)

        fold_rec: dict[str, Any] = {
            "stat_name": config.stat_name,
            "side": config.side,
            "test_season": test_season,
            "n_games": metrics["n_games"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "avg_brier": metrics["avg_brier"],
            "crps": metrics["crps"],
            "ece": metrics["ece"],
            "mce": metrics["mce"],
            "temperature": metrics["temperature"],
            "coverage_50": metrics["coverage_50"],
            "coverage_80": metrics["coverage_80"],
            "coverage_90": metrics["coverage_90"],
            "coverage_95": metrics["coverage_95"],
        }

        # Add per-line Brier scores
        for line, brier in metrics["brier_scores"].items():
            fold_rec[f"brier_{line:.1f}".replace(".", "_")] = brier

        fold_results.append(fold_rec)
        predictions["fold_test_season"] = test_season
        all_predictions.append(predictions)

        logger.info(
            "Fold results: RMSE=%.3f, MAE=%.3f, Brier=%.4f, "
            "ECE=%.4f, Temp=%.3f, Coverage(50/80/90)=%.2f/%.2f/%.2f",
            metrics["rmse"], metrics["mae"], metrics["avg_brier"],
            metrics.get("ece", np.nan), metrics.get("temperature", np.nan),
            metrics["coverage_50"], metrics["coverage_80"],
            metrics["coverage_90"],
        )

    summary_df = pd.DataFrame(fold_results)

    if all_predictions:
        all_pred_df = pd.concat(all_predictions, ignore_index=True)

        # Overall metrics across all folds
        overall = compute_game_prop_metrics(config, all_pred_df)
        logger.info(
            "Overall [%s %s]: RMSE=%.3f, MAE=%.3f, Brier=%.4f, n=%d",
            config.side, config.stat_name,
            overall["rmse"], overall["mae"],
            overall["avg_brier"], overall["n_games"],
        )
    else:
        all_pred_df = pd.DataFrame()

    return summary_df, all_pred_df


# ---------------------------------------------------------------------------
# 8. Isotonic recalibration
# ---------------------------------------------------------------------------

def fit_isotonic_recalibration(
    predictions: pd.DataFrame,
    lines: list[float],
    actual_col: str,
) -> dict[float, IsotonicRegression]:
    """Fit isotonic regression for recalibration per line.

    Isotonic regression maps raw predicted probabilities to calibrated
    probabilities using a monotonic step function fitted on historical
    data. This is a post-hoc correction for systematic miscalibration.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions with p_over columns.
    lines : list[float]
        Lines to calibrate (e.g., [3.5, 4.5, 5.5]).
    actual_col : str
        Column name for actual outcomes (e.g., "actual_k").

    Returns
    -------
    dict[float, IsotonicRegression]
        Mapping of line -> fitted IsotonicRegression model.
    """
    calibrators: dict[float, IsotonicRegression] = {}

    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        if col not in predictions.columns:
            logger.warning("Column %s not found in predictions", col)
            continue

        y_prob = predictions[col].values
        y_true = (predictions[actual_col] > line).astype(float).values

        # Need at least some variation in predictions
        if len(np.unique(y_prob)) < 3:
            logger.warning(
                "Insufficient variation in %s for isotonic fit (line=%.1f)",
                col, line,
            )
            continue

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(y_prob, y_true)
        calibrators[line] = ir

        logger.info(
            "Fitted isotonic recalibration for line=%.1f "
            "(%d samples, %.1f%% positive)",
            line, len(y_true), 100 * y_true.mean(),
        )

    return calibrators


def apply_isotonic_recalibration(
    predictions: pd.DataFrame,
    calibrators: dict[float, IsotonicRegression],
    lines: list[float],
) -> pd.DataFrame:
    """Apply isotonic recalibration to predictions.

    Creates new columns ``p_over_X_5_cal`` with recalibrated probabilities.
    Original columns are preserved.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions with p_over columns.
    calibrators : dict[float, IsotonicRegression]
        Mapping of line -> fitted IsotonicRegression (from
        ``fit_isotonic_recalibration``).
    lines : list[float]
        Lines to recalibrate.

    Returns
    -------
    pd.DataFrame
        Copy of predictions with added ``*_cal`` columns.
    """
    result = predictions.copy()

    for line in lines:
        col = f"p_over_{line:.1f}".replace(".", "_")
        cal_col = f"{col}_cal"

        if col not in result.columns:
            continue
        if line not in calibrators:
            # No calibrator — copy raw
            result[cal_col] = result[col]
            continue

        ir = calibrators[line]
        raw_probs = result[col].values
        result[cal_col] = ir.predict(raw_probs)

    return result
