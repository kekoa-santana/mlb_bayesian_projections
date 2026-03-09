"""
Hitter PA (playing time) projection model with empirical Bayes partial pooling.

PA is NOT a talent stat — it depends on health, roster decisions, and
opportunity. A shrinkage estimator with age-based population priors is
more appropriate than MCMC.

Approach:
1. Project PA/game rate (very stable, shrink toward ~3.85)
2. Project games played (age-bucket priors, Marcel-style weighting)
3. Projected PA = projected_games * projected_pa_per_game
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Population priors by age bucket (from empirical 2018-2025 data, regulars >= 100 PA)
POP_GAMES_BY_AGE = {
    0: {"mu": 130, "sigma": 35},   # young: callups, injury risk
    1: {"mu": 145, "sigma": 26},   # prime: most durable
    2: {"mu": 130, "sigma": 37},   # veteran: rest days, DL stints
}

POP_PA_PER_GAME = {"mu": 3.85, "sigma": 0.40}

# Marcel-style weights for recent seasons (most recent first)
SEASON_WEIGHTS = [5, 4, 3]

SHRINKAGE_K_GAMES = 1.5   # low k = trust observed quickly
SHRINKAGE_K_PA_RATE = 3.0  # PA/game is very stable, shrink more


def compute_hitter_pa_priors(
    hitter_extended: pd.DataFrame,
    from_season: int,
    min_pa: int = 100,
    pop_games: dict | None = None,
    pop_pa_rate: dict | None = None,
) -> pd.DataFrame:
    """Compute shrinkage-estimated PA priors per hitter.

    Uses Marcel-style 5/4/3 weighting of recent seasons, with
    age-bucket population priors for games and PA/game rate.

    Parameters
    ----------
    hitter_extended : pd.DataFrame
        Multi-season stacked output of build_multi_season_hitter_extended.
        Must have: batter_id, season, games, pa, age, age_bucket.
    from_season : int
        Most recent completed season (project forward from here).
    min_pa : int
        Minimum PA in any included season.
    pop_games : dict, optional
        Override population game priors by age bucket.
    pop_pa_rate : dict, optional
        Override population PA/game rate.

    Returns
    -------
    pd.DataFrame
        One row per batter with columns: batter_id, batter_name,
        projected_games, sigma_games, projected_pa_per_game, sigma_pa_rate,
        projected_pa, age, age_bucket, reliability_games, reliability_pa_rate,
        n_seasons.
    """
    if pop_games is None:
        pop_games = POP_GAMES_BY_AGE
    if pop_pa_rate is None:
        pop_pa_rate = POP_PA_PER_GAME

    # Filter to relevant recent seasons (up to 3 most recent)
    recent_seasons = sorted(hitter_extended["season"].unique())
    recent_seasons = [s for s in recent_seasons if s <= from_season][-len(SEASON_WEIGHTS):]

    df = hitter_extended[
        (hitter_extended["season"].isin(recent_seasons))
        & (hitter_extended["pa"] >= min_pa)
    ].copy()

    # Get latest info per player
    latest = df[df["season"] == from_season].copy()
    if latest.empty:
        latest = df.sort_values("season").groupby("batter_id").last().reset_index()

    results = []
    for _, player_latest in latest.iterrows():
        bid = player_latest["batter_id"]
        age_bucket = int(player_latest.get("age_bucket", 1))
        age = player_latest.get("age", 28)

        # Population priors for this age bucket
        pop_g = pop_games.get(age_bucket, pop_games[1])
        pop_mu_g = pop_g["mu"]
        pop_sigma_g = pop_g["sigma"]

        # Get this player's history
        player_hist = df[df["batter_id"] == bid].sort_values("season", ascending=False)
        n_seasons = len(player_hist)

        if n_seasons == 0:
            results.append({
                "batter_id": bid,
                "batter_name": player_latest.get("batter_name", ""),
                "projected_games": pop_mu_g,
                "sigma_games": pop_sigma_g,
                "projected_pa_per_game": pop_pa_rate["mu"],
                "sigma_pa_rate": pop_pa_rate["sigma"],
                "projected_pa": int(pop_mu_g * pop_pa_rate["mu"]),
                "age": age,
                "age_bucket": age_bucket,
                "reliability_games": 0.0,
                "reliability_pa_rate": 0.0,
                "n_seasons": 0,
            })
            continue

        # Marcel-style weighted average of games and PA/game
        weights = SEASON_WEIGHTS[:n_seasons]
        total_weight = sum(weights)

        # Handle 2020 shortened season — scale up to 162-game pace
        games_list = []
        pa_rate_list = []
        for i, (_, row) in enumerate(player_hist.iterrows()):
            g = row["games"]
            pa_pg = row["pa"] / max(row["games"], 1)

            # 2020 adjustment: scale 60-game season to 162
            if row["season"] == 2020:
                g = min(g * (162 / 60), 162)

            games_list.append(g)
            pa_rate_list.append(pa_pg)

        weighted_games = sum(g * w for g, w in zip(games_list, weights)) / total_weight
        weighted_pa_rate = sum(r * w for r, w in zip(pa_rate_list, weights)) / total_weight

        # Observed std
        if n_seasons >= 2:
            obs_std_g = float(np.std(games_list[:n_seasons], ddof=1))
        else:
            obs_std_g = pop_sigma_g

        # Shrinkage on games
        rel_g = n_seasons / (n_seasons + SHRINKAGE_K_GAMES)
        mu_g = rel_g * weighted_games + (1 - rel_g) * pop_mu_g
        sigma_g = rel_g * obs_std_g + (1 - rel_g) * pop_sigma_g

        # Shrinkage on PA/game rate
        rel_pa = n_seasons / (n_seasons + SHRINKAGE_K_PA_RATE)
        mu_pa_rate = rel_pa * weighted_pa_rate + (1 - rel_pa) * pop_pa_rate["mu"]
        sigma_pa_rate = rel_pa * 0.2 + (1 - rel_pa) * pop_pa_rate["sigma"]

        # Age regression: slight decline in games for veterans
        if age >= 35:
            mu_g *= 0.90
        elif age >= 33:
            mu_g *= 0.95

        # Cap at 162 games
        mu_g = min(mu_g, 162)

        projected_pa = int(mu_g * mu_pa_rate)

        results.append({
            "batter_id": bid,
            "batter_name": player_latest.get("batter_name", ""),
            "projected_games": round(mu_g, 1),
            "sigma_games": round(sigma_g, 1),
            "projected_pa_per_game": round(mu_pa_rate, 3),
            "sigma_pa_rate": round(sigma_pa_rate, 3),
            "projected_pa": projected_pa,
            "age": age,
            "age_bucket": age_bucket,
            "reliability_games": round(rel_g, 3),
            "reliability_pa_rate": round(rel_pa, 3),
            "n_seasons": n_seasons,
        })

    result_df = pd.DataFrame(results)
    logger.info(
        "PA priors: %d hitters, mean projected PA=%.0f, mean reliability=%.3f",
        len(result_df),
        result_df["projected_pa"].mean() if len(result_df) > 0 else 0,
        result_df["reliability_games"].mean() if len(result_df) > 0 else 0,
    )
    return result_df


def draw_pa_samples(
    projected_games: float,
    sigma_games: float,
    projected_pa_per_game: float,
    sigma_pa_rate: float,
    n_draws: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw integer PA samples by sampling games * pa_per_game.

    Parameters
    ----------
    projected_games, sigma_games : float
        Games played distribution parameters.
    projected_pa_per_game, sigma_pa_rate : float
        PA/game distribution parameters.
    n_draws : int
        Number of Monte Carlo samples.
    rng : numpy Generator, optional

    Returns
    -------
    np.ndarray
        Integer PA values, shape (n_draws,).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw games from truncated normal [1, 162]
    if sigma_games <= 0:
        games_samples = np.full(n_draws, projected_games)
    else:
        a_g = (1 - projected_games) / sigma_games
        b_g = (162 - projected_games) / sigma_games
        games_samples = stats.truncnorm.rvs(
            a_g, b_g, loc=projected_games, scale=sigma_games,
            size=n_draws, random_state=rng.integers(0, 2**31),
        )

    # Draw PA/game from truncated normal [2.5, 5.0]
    if sigma_pa_rate <= 0:
        pa_rate_samples = np.full(n_draws, projected_pa_per_game)
    else:
        a_r = (2.5 - projected_pa_per_game) / sigma_pa_rate
        b_r = (5.0 - projected_pa_per_game) / sigma_pa_rate
        pa_rate_samples = stats.truncnorm.rvs(
            a_r, b_r, loc=projected_pa_per_game, scale=sigma_pa_rate,
            size=n_draws, random_state=rng.integers(0, 2**31),
        )

    pa_samples = np.clip(np.round(games_samples * pa_rate_samples), 1, 750).astype(int)
    return pa_samples
