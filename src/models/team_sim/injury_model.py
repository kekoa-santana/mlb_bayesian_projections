"""Player injury probability model.

Estimates P(games missed) for each player based on age, position,
and IL history. Uses empirical distributions from 2019-2025 IL data.

Two components:
1. P(any IL stint) — logistic based on age + position + prior IL
2. Games missed | IL stint — empirical distribution by IL type
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Empirical IL rates by age bucket (calibrated to 2019-2025 data).
# Avg team: 577 IL days/season, 13 stints, ~30 rostered players.
# Target: ~40% of players hit IL in a season, avg stint ~44 days.
_BASE_IL_RATE_BY_AGE = {
    "young": 0.35,    # <=25: less wear, but callup-level conditioning
    "prime": 0.38,    # 26-30: peak workload
    "veteran": 0.42,  # 31-35: increasing breakdown
    "old": 0.50,      # 36+: high fragility
}

# Pitchers have higher IL rates than hitters
_PITCHER_IL_MULTIPLIER = 1.20

# Empirical games-missed distribution (conditional on going on IL)
# From 2019-2025 IL stints: median=27d, p75=61d, p90=110d
# Approximate with a mixture: 60% short (10-30 days), 40% long (30-120+ days)
_SHORT_STINT_PARAMS = {"mu": 18, "sigma": 8}    # IL-7/IL-10 stints
_LONG_STINT_PARAMS = {"mu": 70, "sigma": 35}    # IL-15/IL-60 stints
_SHORT_STINT_PROB = 0.60


def estimate_player_injury_params(
    player_id: int,
    age: float,
    is_pitcher: bool,
    il_history: pd.DataFrame | None = None,
    health_score: float | None = None,
) -> dict:
    """Estimate injury parameters for a single player.

    Parameters
    ----------
    player_id : int
    age : float
    is_pitcher : bool
    il_history : pd.DataFrame, optional
        Player's IL stints. Columns: season, days_on_status.
    health_score : float, optional
        Pre-computed health score (0=fragile, 1=durable).

    Returns
    -------
    dict with keys:
        il_prob: P(at least one IL stint this season)
        games_missed_mu: expected games missed (unconditional)
        games_missed_sigma: std of games missed
    """
    # Base IL rate by age
    if age <= 25:
        base_rate = _BASE_IL_RATE_BY_AGE["young"]
    elif age <= 30:
        base_rate = _BASE_IL_RATE_BY_AGE["prime"]
    elif age <= 35:
        base_rate = _BASE_IL_RATE_BY_AGE["veteran"]
    else:
        base_rate = _BASE_IL_RATE_BY_AGE["old"]

    if is_pitcher:
        base_rate *= _PITCHER_IL_MULTIPLIER
    base_rate = min(base_rate, 0.90)

    # Adjust by IL history (players with prior IL stints are more likely)
    if il_history is not None and len(il_history) > 0:
        recent = il_history[il_history["season"] >= il_history["season"].max() - 2]
        n_recent_stints = len(recent)
        if n_recent_stints >= 3:
            base_rate = min(base_rate * 1.3, 0.90)
        elif n_recent_stints >= 1:
            base_rate = min(base_rate * 1.1, 0.85)

    # Adjust by health score if available (overrides age-based)
    if health_score is not None:
        # health_score: 0=fragile, 1=durable
        # Map to IL rate: 0.75 (fragile) -> 0.30 (durable)
        base_rate = 0.75 - 0.45 * health_score

    # Expected games missed = P(IL) * E[days|IL]
    e_days_given_il = (
        _SHORT_STINT_PROB * _SHORT_STINT_PARAMS["mu"]
        + (1 - _SHORT_STINT_PROB) * _LONG_STINT_PARAMS["mu"]
    )
    e_games = base_rate * e_days_given_il
    sigma_games = base_rate * 30  # rough approximation

    return {
        "il_prob": round(base_rate, 3),
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
        # Mixture: short vs long stint
        is_short = rng.random(n_il) < _SHORT_STINT_PROB
        days = np.where(
            is_short,
            rng.normal(_SHORT_STINT_PARAMS["mu"], _SHORT_STINT_PARAMS["sigma"], n_il),
            rng.normal(_LONG_STINT_PARAMS["mu"], _LONG_STINT_PARAMS["sigma"], n_il),
        )
        # ~15% chance of a second stint (reduced from earlier estimate)
        second_stint = rng.random(n_il) < 0.15
        second_days = np.where(
            second_stint,
            rng.normal(20, 10, n_il).clip(7),
            0,
        )
        days = (days + second_days).clip(7, 140)
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

        params = estimate_player_injury_params(
            player_id=pid,
            age=age,
            is_pitcher=is_pitcher,
            il_history=player_il,
            health_score=h_lookup.get(pid),
        )
        params["player_id"] = pid
        params["is_pitcher"] = is_pitcher
        rows.append(params)

    return pd.DataFrame(rows)
