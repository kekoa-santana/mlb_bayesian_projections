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

# Per-age population priors from 2000-2025 empirical data (200+ PA, excl 2020).
# Computed from fact_player_game_mlb: 4,402 player-seasons across 25 years.
# Ages with <15 observations use nearest well-estimated age.
_EMPIRICAL_GAMES_BY_AGE = {
    20: {"mu": 95, "sigma": 28},
    21: {"mu": 99, "sigma": 37},
    22: {"mu": 106, "sigma": 34},
    23: {"mu": 101, "sigma": 34},
    24: {"mu": 105, "sigma": 33},
    25: {"mu": 106, "sigma": 33},
    26: {"mu": 106, "sigma": 33},
    27: {"mu": 105, "sigma": 32},
    28: {"mu": 106, "sigma": 33},
    29: {"mu": 108, "sigma": 34},
    30: {"mu": 109, "sigma": 34},
    31: {"mu": 106, "sigma": 31},
    32: {"mu": 107, "sigma": 31},
    33: {"mu": 106, "sigma": 31},
    34: {"mu": 105, "sigma": 32},
    35: {"mu": 102, "sigma": 30},
    36: {"mu": 100, "sigma": 32},
    37: {"mu": 106, "sigma": 34},
    38: {"mu": 107, "sigma": 34},
    39: {"mu": 100, "sigma": 32},
    40: {"mu": 95, "sigma": 32},
}

# Legacy 3-bucket mapping (for backward compatibility with age_bucket column)
POP_GAMES_BY_AGE = {
    0: {"mu": 104, "sigma": 33},   # young <=25: empirical 2000-2025, 200+ PA
    1: {"mu": 107, "sigma": 33},   # prime 26-30
    2: {"mu": 105, "sigma": 31},   # veteran 31+
}

POP_PA_PER_GAME = {"mu": 3.85, "sigma": 0.40}


def _get_age_prior(age: float, pop_games: dict | None = None) -> dict:
    """Get population games prior for a specific age.

    Uses per-age empirical priors from 2000-2025 data when available,
    falls back to age-bucket priors.
    """
    if pop_games is not None:
        # Caller provided custom priors (age-bucket keyed)
        return pop_games
    age_int = int(round(age))
    if age_int in _EMPIRICAL_GAMES_BY_AGE:
        return _EMPIRICAL_GAMES_BY_AGE[age_int]
    # Clamp to edges
    if age_int < 20:
        return _EMPIRICAL_GAMES_BY_AGE[20]
    return _EMPIRICAL_GAMES_BY_AGE[40]

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
    health_scores: pd.DataFrame | None = None,
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
    health_scores : pd.DataFrame, optional
        Output of health_score.compute_health_scores.
        Must have: player_id, health_score, health_label.
        When provided, replaces blunt age penalties with data-driven
        health adjustments. Falls back to age penalties when None.

    Returns
    -------
    pd.DataFrame
        One row per batter with columns: batter_id, batter_name,
        projected_games, sigma_games, projected_pa_per_game, sigma_pa_rate,
        projected_pa, age, age_bucket, reliability_games, reliability_pa_rate,
        n_seasons, health_score, health_label.
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

        # Population priors — per-age if available, else age-bucket
        if pop_games is not None:
            pop_g = pop_games.get(age_bucket, pop_games.get(1, {"mu": 88, "sigma": 40}))
        else:
            pop_g = _get_age_prior(age)
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

        # Health/durability adjustment (replaces blunt age penalties)
        h_score = None
        h_label = ""
        if health_scores is not None and not health_scores.empty:
            h_row = health_scores[health_scores["player_id"] == bid]
            if not h_row.empty:
                h_score = float(h_row.iloc[0]["health_score"])
                h_label = str(h_row.iloc[0]["health_label"])

        if h_score is not None:
            # Data-driven: games_mult 0.75x (worst) to 1.02x (best)
            games_mult = 0.75 + 0.27 * h_score
            # Wider intervals for injury-prone: 1.50x (worst) to 0.85x (best)
            sigma_mult = 1.50 - 0.65 * h_score
            mu_g *= games_mult
            sigma_g *= sigma_mult
        else:
            # Fallback: blunt age penalties (backward compat)
            if age >= 35:
                mu_g *= 0.90
            elif age >= 33:
                mu_g *= 0.95

        # Cap at 162 games
        mu_g = min(mu_g, 162)

        projected_pa = int(mu_g * mu_pa_rate)

        row_dict = {
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
        }
        if h_score is not None:
            row_dict["health_score"] = round(h_score, 4)
            row_dict["health_label"] = h_label
        results.append(row_dict)

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
