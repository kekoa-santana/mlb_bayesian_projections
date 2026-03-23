"""Player injury probability model.

Estimates P(games missed) for each player based on age, position,
and IL history. Fitted from 2019-2025 IL data (6,757 player-seasons).

Two components:
1. P(any IL stint) — logistic regression on age + position + prior IL
   AUC=0.603. Prior IL is strongest predictor.
2. Games missed | IL stint — empirical lognormal (median=27d, mean=42d)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import expit

logger = logging.getLogger(__name__)

# Logistic regression coefficients fitted on 2019-2025 IL data (n=6757, AUC=0.603)
# Features: age, is_pitcher, prev_il, prev_days, prev2_il, games
_LOGISTIC_COEFS = np.array([0.0219, -0.2918, 0.3891, 0.0009, 0.4670, -0.0064])
_LOGISTIC_INTERCEPT = -0.5240

# Days-missed distribution (conditional on going on IL)
# Fitted from 2019-2025: median=27, mean=42, std=40, p90=100
# LogNormal fits well: ln(days) ~ N(3.15, 0.85)
_DAYS_LOG_MU = 3.15    # exp(3.15) ≈ 23, close to median after clipping
_DAYS_LOG_SIGMA = 0.85


def estimate_player_injury_params(
    player_id: int,
    age: float,
    is_pitcher: bool,
    il_history: pd.DataFrame | None = None,
    health_score: float | None = None,
    prev_games: int = 130,
) -> dict:
    """Estimate injury parameters for a single player.

    Uses logistic regression fitted on 2019-2025 IL data.

    Parameters
    ----------
    player_id : int
    age : float
    is_pitcher : bool
    il_history : pd.DataFrame, optional
        Player's IL stints. Columns: season, days_on_status.
    health_score : float, optional
        Pre-computed health score (0=fragile, 1=durable).
    prev_games : int
        Games played last season.

    Returns
    -------
    dict with keys:
        il_prob: P(at least one IL stint this season)
        games_missed_mu: expected games missed (unconditional)
        games_missed_sigma: std of games missed
    """
    # Extract prior IL features
    prev_il, prev_days, prev2_il = 0.0, 0.0, 0.0
    if il_history is not None and len(il_history) > 0:
        seasons = sorted(il_history["season"].unique())
        if len(seasons) >= 1:
            last = il_history[il_history["season"] == seasons[-1]]
            prev_il = 1.0
            prev_days = float(last["days_on_status"].sum())
        if len(seasons) >= 2:
            prev2_il = 1.0

    # Logistic regression prediction
    features = np.array([age, float(is_pitcher), prev_il, prev_days, prev2_il, prev_games])
    logit = float(features @ _LOGISTIC_COEFS + _LOGISTIC_INTERCEPT)
    il_prob = float(expit(logit))

    # Override with health score if available (more data-driven)
    if health_score is not None:
        # Blend: 60% logistic model + 40% health-score-derived
        h_prob = 0.70 - 0.40 * health_score  # 0.70 (fragile) -> 0.30 (durable)
        il_prob = 0.60 * il_prob + 0.40 * h_prob

    il_prob = np.clip(il_prob, 0.05, 0.85)

    # Expected games missed (unconditional)
    e_days_given_il = np.exp(_DAYS_LOG_MU + _DAYS_LOG_SIGMA ** 2 / 2)  # lognormal mean
    e_games = il_prob * e_days_given_il
    sigma_games = il_prob * 25

    return {
        "il_prob": round(il_prob, 3),
        "games_missed_mu": round(e_games, 1),
        "games_missed_sigma": round(sigma_games, 1),
    }


def draw_games_missed(
    il_prob: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw games-missed samples for one player across N seasons.

    Returns array of shape (n_sims,) with integer games missed.
    """
    # Step 1: Does the player hit the IL this season?
    goes_on_il = rng.random(n_sims) < il_prob

    # Step 2: If yes, how many days?
    games_missed = np.zeros(n_sims, dtype=np.int32)
    n_il = goes_on_il.sum()
    if n_il > 0:
        # LogNormal stint duration (fitted: median=27, mean=42, p90=100)
        days = rng.lognormal(_DAYS_LOG_MU, _DAYS_LOG_SIGMA, n_il)
        days = days.clip(7, 140)
        games_missed[goes_on_il] = np.round(days).astype(np.int32)

    return games_missed


def build_team_injury_params(
    roster: pd.DataFrame,
    health_scores: pd.DataFrame | None = None,
    il_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build injury parameters for all players on a team roster.

    Parameters
    ----------
    roster : pd.DataFrame
        Columns: player_id, age, position (or primary_position).
    health_scores : pd.DataFrame, optional
        Columns: player_id, health_score.
    il_history : pd.DataFrame, optional
        Columns: player_id, season, days_on_status.

    Returns
    -------
    pd.DataFrame
        One row per player with il_prob, games_missed_mu, games_missed_sigma.
    """
    pitcher_positions = {"P", "SP", "RP", "TWP"}
    h_lookup = {}
    if health_scores is not None and not health_scores.empty:
        h_lookup = dict(zip(
            health_scores["player_id"].astype(int),
            health_scores["health_score"],
        ))

    rows = []
    for _, player in roster.iterrows():
        pid = int(player["player_id"])
        age = float(player.get("age", 28))
        pos = str(player.get("primary_position", player.get("position", ""))).upper()
        is_pitcher = pos in pitcher_positions

        player_il = None
        if il_history is not None:
            player_il = il_history[il_history["player_id"] == pid]

        prev_games = int(player.get("games", player.get("prev_games", 130)))

        params = estimate_player_injury_params(
            player_id=pid,
            age=age,
            is_pitcher=is_pitcher,
            il_history=player_il,
            health_score=h_lookup.get(pid),
            prev_games=prev_games,
        )
        params["player_id"] = pid
        params["is_pitcher"] = is_pitcher
        rows.append(params)

    return pd.DataFrame(rows)
