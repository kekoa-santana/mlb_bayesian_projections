"""
In-season updater for sim-based projections.

Updates preseason rate posteriors with in-season observed data using
Beta-Binomial conjugate updating, then re-runs the season simulator
for rest-of-season projections.

The key insight: the preseason posterior becomes the prior, and
in-season PAs provide the likelihood update.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# How much to weight in-season data vs preseason posterior
# Higher = trust in-season more quickly (less "sticky" priors)
_PRIOR_WEIGHT_DECAY = 0.85  # preseason prior worth this fraction of its original weight


def update_rate_posterior(
    preseason_samples: np.ndarray,
    in_season_successes: int,
    in_season_trials: int,
    prior_strength: float = 50.0,
) -> np.ndarray:
    """Update a rate posterior with in-season observations.

    Uses Beta-Binomial conjugate updating. The preseason posterior
    is summarized as a Beta distribution, then updated with the
    in-season binomial observation.

    Parameters
    ----------
    preseason_samples : np.ndarray
        Posterior rate samples from preseason model.
    in_season_successes : int
        Observed successes in-season (e.g., strikeouts).
    in_season_trials : int
        Observed trials in-season (e.g., plate appearances).
    prior_strength : float
        Effective sample size of the preseason prior.
        Higher = more resistant to in-season updating.

    Returns
    -------
    np.ndarray
        Updated posterior samples.
    """
    if in_season_trials <= 0:
        return preseason_samples

    # Summarize preseason posterior as Beta
    mean = float(np.mean(preseason_samples))
    mean = np.clip(mean, 0.001, 0.999)

    # Effective prior: Beta(alpha, beta) where alpha + beta = prior_strength
    alpha_prior = mean * prior_strength * _PRIOR_WEIGHT_DECAY
    beta_prior = (1 - mean) * prior_strength * _PRIOR_WEIGHT_DECAY

    # Conjugate update: Beta(alpha + successes, beta + failures)
    alpha_post = alpha_prior + in_season_successes
    beta_post = beta_prior + (in_season_trials - in_season_successes)

    # Draw new samples
    rng = np.random.default_rng(42)
    return rng.beta(alpha_post, beta_post, size=len(preseason_samples))


def compute_rest_of_season_games(
    preseason_games_mu: float,
    preseason_games_sigma: float,
    games_played: int,
    season_games: int = 162,
) -> tuple[float, float]:
    """Compute rest-of-season games projection.

    Parameters
    ----------
    preseason_games_mu, preseason_games_sigma : float
        Preseason games projection.
    games_played : int
        Games already played.
    season_games : int
        Total games in the season.

    Returns
    -------
    tuple[float, float]
        (ros_games_mu, ros_games_sigma).
    """
    if games_played >= season_games:
        return (0.0, 0.0)

    # Pace-based: extrapolate from games_played
    team_games_elapsed = max(games_played * 1.05, 1)  # rough team game count
    pace_fraction = games_played / team_games_elapsed
    pace_games = pace_fraction * season_games

    # Blend preseason projection with pace
    # Early season: mostly preseason. Late season: mostly pace.
    season_fraction = team_games_elapsed / season_games
    blend = min(season_fraction * 2, 1.0)  # full pace weight by midseason

    blended_total = (1 - blend) * preseason_games_mu + blend * pace_games
    ros_mu = max(blended_total - games_played, 0)

    # Sigma shrinks as season progresses (less uncertainty)
    ros_sigma = preseason_games_sigma * (1 - season_fraction) * 0.8

    return (ros_mu, max(ros_sigma, 2.0))


def build_in_season_posteriors(
    preseason_samples: dict[int, dict[str, np.ndarray]],
    in_season_stats: pd.DataFrame,
    prior_strength: float = 50.0,
) -> dict[int, dict[str, np.ndarray]]:
    """Update all player posteriors with in-season data.

    Parameters
    ----------
    preseason_samples : dict[int, dict[str, np.ndarray]]
        player_id -> {k_rate, bb_rate, hr_rate} preseason samples.
    in_season_stats : pd.DataFrame
        Columns: player_id, pa (or bf), k, bb, hr.
    prior_strength : float
        Effective sample size of preseason prior.

    Returns
    -------
    dict[int, dict[str, np.ndarray]]
        Updated posteriors.
    """
    updated = {}

    for pid, preseason in preseason_samples.items():
        player_row = in_season_stats[in_season_stats["player_id"] == pid]

        if player_row.empty:
            updated[pid] = preseason
            continue

        row = player_row.iloc[0]
        pa = int(row.get("pa", row.get("bf", 0)))
        k = int(row.get("k", 0))
        bb = int(row.get("bb", 0))
        hr = int(row.get("hr", 0))

        rates = {}
        rates["k_rate"] = update_rate_posterior(
            preseason["k_rate"], k, pa, prior_strength,
        )
        rates["bb_rate"] = update_rate_posterior(
            preseason["bb_rate"], bb, pa, prior_strength,
        )
        rates["hr_rate"] = update_rate_posterior(
            preseason["hr_rate"], hr, pa, prior_strength,
        )

        updated[pid] = rates

    n_updated = sum(
        1 for pid in updated
        if pid in dict(in_season_stats.set_index("player_id").iterrows())
    )
    logger.info("Updated %d / %d posteriors with in-season data", n_updated, len(updated))
    return updated
