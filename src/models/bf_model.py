"""
Step 13: Batters-Faced distribution model with empirical Bayes partial pooling.

BF for starters is very stable (mean~22, std~4.5). A shrinkage estimator
is appropriate — no MCMC needed.

Shrinkage formula:
    reliability = n / (n + k)       where k = sigma^2 / tau^2
    mu_pitcher  = rel * obs_mean + (1 - rel) * pop_mean
    sigma_pitcher = rel * obs_std + (1 - rel) * pop_within_std

Pitches-per-PA adjustment (v1.7):
    Efficient pitchers (low P/PA) face more batters for the same pitch budget.
    implied_bf = avg_pitch_count / pitches_per_pa
    mu_bf is adjusted by blending with implied_bf (weight = PPA_ADJ_WEIGHT).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Population defaults (validated against 2022-2025 starter data, BF >= 9
# to exclude openers/bullpen games that drag the mean down)
DEFAULT_POP_BF_MU = 22.4
DEFAULT_POP_WITHIN_STD = 3.3
DEFAULT_SHRINKAGE_K = 3.4  # sigma^2 / tau^2 ≈ 10.97 / 3.26

# Pitches-per-PA adjustment
POP_MEAN_PPA = 3.91        # League avg P/PA for starters (2023-2025)
PPA_ADJ_WEIGHT = 0.15      # Blend weight for P/PA implied BF adjustment


def compute_pitcher_bf_priors(
    game_logs: pd.DataFrame,
    pitcher_ppa: pd.DataFrame | None = None,
    pop_mu: float = DEFAULT_POP_BF_MU,
    pop_within_std: float = DEFAULT_POP_WITHIN_STD,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    min_starts: int = 5,
    ppa_adj_weight: float = PPA_ADJ_WEIGHT,
) -> pd.DataFrame:
    """Compute shrinkage-estimated BF priors per pitcher-season.

    Parameters
    ----------
    game_logs : pd.DataFrame
        Stacked game logs across seasons. Must have columns:
        pitcher_id, season, batters_faced, is_starter.
    pitcher_ppa : pd.DataFrame, optional
        Pitches-per-PA data. Must have columns: pitcher_id, pitches_per_pa.
        If provided, adjusts mu_bf based on pitcher efficiency.
    pop_mu : float
        Population mean BF for starters.
    pop_within_std : float
        Population within-pitcher game-to-game std.
    shrinkage_k : float
        Shrinkage constant k = sigma^2 / tau^2.
    min_starts : int
        Pitchers below this get pure population prior.
    ppa_adj_weight : float
        Blend weight for the P/PA implied BF adjustment (0-1).

    Returns
    -------
    pd.DataFrame
        One row per (pitcher_id, season) with columns:
        pitcher_id, season, n_starts, raw_mean_bf, raw_std_bf,
        mu_bf, sigma_bf, reliability, pitches_per_pa, ppa_adj.
    """
    # Filter to starters with BF >= 9 (excludes openers/bullpen games
    # that drag down population BF mean and add artificial variance)
    starters = game_logs[
        (game_logs["is_starter"] == True)  # noqa: E712
        & (game_logs["batters_faced"] >= 9)
    ].copy()

    if starters.empty:
        logger.warning("No starter game logs found")
        return pd.DataFrame(columns=[
            "pitcher_id", "season", "n_starts", "raw_mean_bf", "raw_std_bf",
            "mu_bf", "sigma_bf", "reliability", "pitches_per_pa", "ppa_adj",
        ])

    # Per-pitcher-season aggregation
    agg = starters.groupby(["pitcher_id", "season"]).agg(
        n_starts=("batters_faced", "count"),
        raw_mean_bf=("batters_faced", "mean"),
        raw_std_bf=("batters_faced", "std"),
    ).reset_index()

    # Also compute avg pitches per start for implied BF
    if "number_of_pitches" in starters.columns:
        pitch_agg = starters.groupby(["pitcher_id", "season"]).agg(
            avg_pitches=("number_of_pitches", "mean"),
        ).reset_index()
        agg = agg.merge(pitch_agg, on=["pitcher_id", "season"], how="left")
    else:
        agg["avg_pitches"] = np.nan

    # Fill NaN std (single-start pitchers) with pop within-pitcher std
    agg["raw_std_bf"] = agg["raw_std_bf"].fillna(pop_within_std)

    # Shrinkage
    agg["reliability"] = agg["n_starts"] / (agg["n_starts"] + shrinkage_k)

    # Below min_starts: reliability = 0 (pure population prior)
    below_min = agg["n_starts"] < min_starts
    agg.loc[below_min, "reliability"] = 0.0

    agg["mu_bf"] = (
        agg["reliability"] * agg["raw_mean_bf"]
        + (1 - agg["reliability"]) * pop_mu
    )
    agg["sigma_bf"] = (
        agg["reliability"] * agg["raw_std_bf"]
        + (1 - agg["reliability"]) * pop_within_std
    )

    # --- Pitches-per-PA adjustment ---
    # Efficient pitchers (low P/PA) face more BF for the same pitch budget.
    # Compute implied BF = avg_pitches / pitches_per_pa, then adjust mu_bf.
    agg["pitches_per_pa"] = np.nan
    agg["ppa_adj"] = 0.0

    if pitcher_ppa is not None and not pitcher_ppa.empty:
        ppa_map = pitcher_ppa.set_index("pitcher_id")["pitches_per_pa"].to_dict()
        agg["pitches_per_pa"] = agg["pitcher_id"].map(ppa_map)

        has_ppa = agg["pitches_per_pa"].notna() & agg["avg_pitches"].notna()
        if has_ppa.any():
            # Implied BF from this pitcher's pitch budget / their P/PA
            implied_bf = agg.loc[has_ppa, "avg_pitches"] / agg.loc[has_ppa, "pitches_per_pa"]
            # Baseline implied BF using population P/PA
            baseline_implied_bf = agg.loc[has_ppa, "avg_pitches"] / POP_MEAN_PPA
            # Adjustment = difference from what population P/PA would predict
            ppa_delta = implied_bf - baseline_implied_bf
            agg.loc[has_ppa, "ppa_adj"] = ppa_delta * ppa_adj_weight
            agg.loc[has_ppa, "mu_bf"] = agg.loc[has_ppa, "mu_bf"] + agg.loc[has_ppa, "ppa_adj"]

            logger.info(
                "P/PA adjustment applied to %d pitchers (mean adj=%.2f BF, range=%.2f to %.2f)",
                has_ppa.sum(),
                agg.loc[has_ppa, "ppa_adj"].mean(),
                agg.loc[has_ppa, "ppa_adj"].min(),
                agg.loc[has_ppa, "ppa_adj"].max(),
            )

    # Drop intermediate column
    agg.drop(columns=["avg_pitches"], inplace=True)

    logger.info(
        "BF priors: %d pitcher-seasons, mean reliability=%.3f",
        len(agg), agg["reliability"].mean(),
    )
    return agg


def get_bf_distribution(
    pitcher_id: int,
    season: int,
    bf_priors: pd.DataFrame,
    pop_mu: float = DEFAULT_POP_BF_MU,
    pop_within_std: float = DEFAULT_POP_WITHIN_STD,
) -> dict[str, Any]:
    """Look up BF distribution parameters for a pitcher.

    Parameters
    ----------
    pitcher_id : int
        MLB pitcher ID.
    season : int
        Season to look up.
    bf_priors : pd.DataFrame
        Output of ``compute_pitcher_bf_priors``.
    pop_mu : float
        Fallback population mean.
    pop_within_std : float
        Fallback population std.

    Returns
    -------
    dict
        Keys: mu_bf, sigma_bf, reliability, dist_type.
    """
    mask = (bf_priors["pitcher_id"] == pitcher_id) & (bf_priors["season"] == season)
    rows = bf_priors[mask]

    if rows.empty:
        return {
            "mu_bf": pop_mu,
            "sigma_bf": pop_within_std,
            "reliability": 0.0,
            "dist_type": "population_fallback",
        }

    row = rows.iloc[0]
    return {
        "mu_bf": float(row["mu_bf"]),
        "sigma_bf": float(row["sigma_bf"]),
        "reliability": float(row["reliability"]),
        "dist_type": "shrinkage",
    }


def draw_bf_samples(
    mu_bf: float,
    sigma_bf: float,
    n_draws: int,
    bf_min: int = 3,
    bf_max: int = 35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw integer BF samples from a truncated normal.

    Parameters
    ----------
    mu_bf : float
        Mean BF.
    sigma_bf : float
        Std BF.
    n_draws : int
        Number of samples to draw.
    bf_min : int
        Minimum BF (inclusive).
    bf_max : int
        Maximum BF (inclusive).
    rng : numpy Generator, optional
        For reproducibility. If None, uses default.

    Returns
    -------
    np.ndarray
        Integer BF values, shape (n_draws,).
    """
    if sigma_bf <= 0:
        return np.full(n_draws, int(np.clip(np.round(mu_bf), bf_min, bf_max)), dtype=int)

    a = (bf_min - mu_bf) / sigma_bf
    b = (bf_max - mu_bf) / sigma_bf

    if rng is None:
        rng = np.random.default_rng()

    # Use scipy truncnorm with numpy rng
    samples = stats.truncnorm.rvs(
        a, b, loc=mu_bf, scale=sigma_bf, size=n_draws,
        random_state=rng.integers(0, 2**31),
    )

    return np.clip(np.round(samples), bf_min, bf_max).astype(int)


# ---------------------------------------------------------------------------
# Batter PA distribution model
# ---------------------------------------------------------------------------

# Batter PA population defaults (validated against 2022-2025 batting boxscores)
DEFAULT_POP_PA_MU = 3.9          # Population mean PA per game
DEFAULT_POP_PA_WITHIN_STD = 0.9  # Within-batter game-to-game std
DEFAULT_PA_SHRINKAGE_K = 3.0     # Shrinkage constant (more conservative than BF)


def compute_batter_pa_priors(
    game_logs: pd.DataFrame,
    pop_mu: float = DEFAULT_POP_PA_MU,
    pop_within_std: float = DEFAULT_POP_PA_WITHIN_STD,
    shrinkage_k: float = DEFAULT_PA_SHRINKAGE_K,
    min_games: int = 10,
) -> pd.DataFrame:
    """Compute shrinkage-estimated PA priors per batter-season.

    Parameters
    ----------
    game_logs : pd.DataFrame
        Batter game logs. Must have columns:
        batter_id, season, plate_appearances.
        (From get_batter_game_logs or get_cached_batter_game_logs)
    pop_mu : float
        Population mean PA per game.
    pop_within_std : float
        Population within-batter game-to-game std.
    shrinkage_k : float
        Shrinkage constant.
    min_games : int
        Batters below this get pure population prior.

    Returns
    -------
    pd.DataFrame
        One row per (batter_id, season) with columns:
        batter_id, season, n_games, raw_mean_pa, raw_std_pa,
        mu_pa, sigma_pa, reliability.
    """
    # Filter to games with at least 1 PA
    valid = game_logs[game_logs["plate_appearances"] >= 1].copy()

    if valid.empty:
        logger.warning("No batter game logs found")
        return pd.DataFrame(columns=[
            "batter_id", "season", "n_games", "raw_mean_pa", "raw_std_pa",
            "mu_pa", "sigma_pa", "reliability",
        ])

    # Per-batter-season aggregation
    agg = valid.groupby(["batter_id", "season"]).agg(
        n_games=("plate_appearances", "count"),
        raw_mean_pa=("plate_appearances", "mean"),
        raw_std_pa=("plate_appearances", "std"),
    ).reset_index()

    # Fill NaN std (single-game batters) with population within-batter std
    agg["raw_std_pa"] = agg["raw_std_pa"].fillna(pop_within_std)

    # Shrinkage
    agg["reliability"] = agg["n_games"] / (agg["n_games"] + shrinkage_k)

    # Below min_games: reliability = 0 (pure population prior)
    below_min = agg["n_games"] < min_games
    agg.loc[below_min, "reliability"] = 0.0

    agg["mu_pa"] = (
        agg["reliability"] * agg["raw_mean_pa"]
        + (1 - agg["reliability"]) * pop_mu
    )
    agg["sigma_pa"] = (
        agg["reliability"] * agg["raw_std_pa"]
        + (1 - agg["reliability"]) * pop_within_std
    )

    logger.info(
        "PA priors: %d batter-seasons, mean reliability=%.3f",
        len(agg), agg["reliability"].mean(),
    )
    return agg


def get_pa_distribution(
    batter_id: int,
    season: int,
    pa_priors: pd.DataFrame,
    pop_mu: float = DEFAULT_POP_PA_MU,
    pop_within_std: float = DEFAULT_POP_PA_WITHIN_STD,
) -> dict[str, Any]:
    """Look up PA distribution parameters for a batter.

    Parameters
    ----------
    batter_id : int
        MLB batter ID.
    season : int
        Season to look up.
    pa_priors : pd.DataFrame
        Output of ``compute_batter_pa_priors``.
    pop_mu : float
        Fallback population mean.
    pop_within_std : float
        Fallback population std.

    Returns
    -------
    dict
        Keys: mu_pa, sigma_pa, reliability, dist_type.
    """
    mask = (pa_priors["batter_id"] == batter_id) & (pa_priors["season"] == season)
    rows = pa_priors[mask]

    if rows.empty:
        return {
            "mu_pa": pop_mu,
            "sigma_pa": pop_within_std,
            "reliability": 0.0,
            "dist_type": "population_fallback",
        }

    row = rows.iloc[0]
    return {
        "mu_pa": float(row["mu_pa"]),
        "sigma_pa": float(row["sigma_pa"]),
        "reliability": float(row["reliability"]),
        "dist_type": "shrinkage",
    }


def draw_pa_samples(
    mu_pa: float,
    sigma_pa: float,
    n_draws: int,
    pa_min: int = 1,
    pa_max: int = 7,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw integer PA samples from a truncated normal.

    Parameters
    ----------
    mu_pa : float
        Mean PA per game.
    sigma_pa : float
        Std PA per game.
    n_draws : int
        Number of samples to draw.
    pa_min : int
        Minimum PA (inclusive).
    pa_max : int
        Maximum PA (inclusive).
    rng : numpy Generator, optional
        For reproducibility. If None, uses default.

    Returns
    -------
    np.ndarray
        Integer PA values, shape (n_draws,).
    """
    return draw_bf_samples(
        mu_bf=mu_pa,
        sigma_bf=sigma_pa,
        n_draws=n_draws,
        bf_min=pa_min,
        bf_max=pa_max,
        rng=rng,
    )
