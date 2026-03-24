"""
Glicko-2 player rating engine for MLB batters and pitchers.

Rates individual players from plate-appearance-level data using the
Glicko-2 algorithm (Glickman 2001).  Each PA is scored on [0, 1] using
wOBA values when available, with categorical fallbacks for events that
lack batted-ball data (strikeouts, walks, HBP).

Batters receive the PA score directly; pitchers receive 1 - PA score.
Within a single game-date, all updates use a *snapshot* of pre-date
ratings so that game ordering within the day does not matter.

Architecture mirrors ``src/models/team_elo.py``: YAML config with
defaults dict, dataclass for mutable state, pure-function math, and
a single ``compute_ratings()`` entry point that returns final ratings
plus a game-by-game history DataFrame.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_config() -> dict[str, Any]:
    """Load player_glicko config block from model.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("player_glicko", {})


_DEFAULTS: dict[str, Any] = {
    "initial_mu": 1500,
    "initial_phi": 350,
    "initial_sigma": 0.06,
    "phi_floor": 50,
    "tau": 0.5,
    "season_regression_mu": 0.30,
    "season_phi_reset": 200,
    "max_inactivity_days": 180,
}


def get_config() -> dict[str, Any]:
    """Merge YAML config with defaults."""
    cfg = {**_DEFAULTS}
    cfg.update(_load_config())
    return cfg


# ---------------------------------------------------------------------------
# Glicko-2 ratings state
# ---------------------------------------------------------------------------
@dataclass
class GlickoRatings:
    """Mutable container for all player Glicko-2 ratings."""

    mu: dict[int, float] = field(default_factory=dict)
    phi: dict[int, float] = field(default_factory=dict)
    sigma: dict[int, float] = field(default_factory=dict)
    games_rated: dict[int, int] = field(default_factory=dict)
    last_seen: dict[int, str] = field(default_factory=dict)

    def get_mu(self, pid: int, initial: float = 1500.0) -> float:
        """Return current rating for *pid*, initializing if absent."""
        return self.mu.setdefault(pid, initial)

    def get_phi(self, pid: int, initial: float = 350.0) -> float:
        """Return current RD for *pid*, initializing if absent."""
        return self.phi.setdefault(pid, initial)

    def get_sigma(self, pid: int, initial: float = 0.06) -> float:
        """Return current volatility for *pid*, initializing if absent."""
        return self.sigma.setdefault(pid, initial)


# ---------------------------------------------------------------------------
# PA outcome scoring
# ---------------------------------------------------------------------------
_SKIP_EVENTS: frozenset[str] = frozenset({
    "intent_walk",
    "sac_bunt",
    "sac_bunt_double_play",
    "truncated_pa",
    "catcher_interf",
})

_OUT_EVENTS: frozenset[str] = frozenset({
    "field_out",
    "force_out",
    "double_play",
    "grounded_into_double_play",
    "fielders_choice",
    "fielders_choice_out",
    "triple_play",
    "sac_fly",
    "sac_fly_double_play",
    "field_error",
    "other_out",
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
    "pickoff_1b",
    "pickoff_2b",
    "pickoff_3b",
})

_HIT_FALLBACK: dict[str, float] = {
    "home_run": 1.00,
    "triple": 0.80,
    "double": 0.63,
    "single": 0.45,
}


def _pa_outcome_score(events: str, woba_value: float | None) -> float | None:
    """Convert PA outcome to batter score in [0, 1].

    Parameters
    ----------
    events : str
        The ``events`` column value from the PA record.
    woba_value : float or None
        Linear-weight wOBA value for the PA, if available.

    Returns
    -------
    float or None
        Batter score in [0, 1], or None if the PA should be skipped
        (intentional walks, sac bunts, catcher interference, etc.).
    """
    if events is None or (isinstance(events, float) and math.isnan(events)):
        return None
    ev = str(events).strip().lower()
    if ev in _SKIP_EVENTS or ev == "":
        return None

    # Strikeouts
    if ev in ("strikeout", "strikeout_double_play"):
        return 0.0

    # Walks / HBP
    if ev == "walk":
        return 0.35
    if ev == "hit_by_pitch":
        return 0.36

    # BIP / hits — prefer woba_value when present
    if woba_value is not None and not (
        isinstance(woba_value, float) and math.isnan(woba_value)
    ):
        return min(float(woba_value) / 2.0, 1.0)

    # Fallback: categorical hit types
    if ev in _HIT_FALLBACK:
        return _HIT_FALLBACK[ev]

    # All other outs
    return 0.10


# ---------------------------------------------------------------------------
# Glicko-2 math (Glickman 2001)
# ---------------------------------------------------------------------------
_GLICKO2_SCALE: float = 173.7178  # 400 / ln(10)
_EPSILON: float = 0.000001        # convergence threshold for volatility


def _to_glicko2(mu: float, phi: float) -> tuple[float, float]:
    """Convert from Glicko-1 scale to Glicko-2 internal scale.

    Parameters
    ----------
    mu : float
        Rating on Glicko-1 scale (centered at 1500).
    phi : float
        Rating deviation on Glicko-1 scale.

    Returns
    -------
    tuple[float, float]
        (mu', phi') on Glicko-2 scale (centered at 0).
    """
    return (mu - 1500) / _GLICKO2_SCALE, phi / _GLICKO2_SCALE


def _from_glicko2(mu_prime: float, phi_prime: float) -> tuple[float, float]:
    """Convert from Glicko-2 internal scale back to Glicko-1 scale.

    Parameters
    ----------
    mu_prime : float
        Rating on Glicko-2 scale.
    phi_prime : float
        Rating deviation on Glicko-2 scale.

    Returns
    -------
    tuple[float, float]
        (mu, phi) on Glicko-1 scale.
    """
    return mu_prime * _GLICKO2_SCALE + 1500, phi_prime * _GLICKO2_SCALE


def _g(phi_prime: float) -> float:
    """Glicko-2 g function: reduces impact of uncertain opponents.

    Parameters
    ----------
    phi_prime : float
        Opponent RD on Glicko-2 scale.

    Returns
    -------
    float
        Scaling factor in (0, 1].
    """
    return 1.0 / math.sqrt(1.0 + 3.0 * phi_prime ** 2 / math.pi ** 2)


def _E(mu_prime: float, mu_j_prime: float, phi_j_prime: float) -> float:
    """Expected score against opponent j.

    Parameters
    ----------
    mu_prime : float
        Player rating on Glicko-2 scale.
    mu_j_prime : float
        Opponent rating on Glicko-2 scale.
    phi_j_prime : float
        Opponent RD on Glicko-2 scale.

    Returns
    -------
    float
        Expected score in (0, 1).
    """
    return 1.0 / (1.0 + math.exp(-_g(phi_j_prime) * (mu_prime - mu_j_prime)))


def _update_volatility(
    sigma: float,
    delta: float,
    phi_prime: float,
    v: float,
    tau: float,
) -> float:
    """Compute new volatility via the Illinois algorithm.

    Implements Step 5 of the Glicko-2 algorithm from Glickman's paper:
    http://www.glicko.net/glicko/glicko2.pdf

    Parameters
    ----------
    sigma : float
        Current volatility.
    delta : float
        Estimated improvement.
    phi_prime : float
        Current RD on Glicko-2 scale.
    v : float
        Variance of expected scores.
    tau : float
        System constant controlling volatility change rate.

    Returns
    -------
    float
        Updated volatility.
    """
    a = math.log(sigma ** 2)

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta ** 2 - phi_prime ** 2 - v - ex)
        den = 2.0 * (phi_prime ** 2 + v + ex) ** 2
        return num / den - (x - a) / tau ** 2

    # Initial bounds
    A = a
    if delta ** 2 > phi_prime ** 2 + v:
        B = math.log(delta ** 2 - phi_prime ** 2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    fA = f(A)
    fB = f(B)

    # Illinois algorithm iteration
    for _ in range(50):
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A = B
            fA = fB
        else:
            fA /= 2.0
        B = C
        fB = fC
        if abs(B - A) < _EPSILON:
            break

    return math.exp(A / 2.0)


def _update_player(
    mu: float,
    phi: float,
    sigma: float,
    opponents: list[tuple[float, float, float]],
    tau: float,
    phi_floor: float,
) -> tuple[float, float, float]:
    """Full Glicko-2 update for one player given their opponents.

    Parameters
    ----------
    mu : float
        Current rating on Glicko-1 scale.
    phi : float
        Current RD on Glicko-1 scale.
    sigma : float
        Current volatility.
    opponents : list[tuple[float, float, float]]
        List of (opponent_mu, opponent_phi, score) in Glicko-1 scale.
        ``score`` is the actual outcome in [0, 1] for this player.
    tau : float
        System constant.
    phi_floor : float
        Minimum RD (Glicko-1 scale).

    Returns
    -------
    tuple[float, float, float]
        Updated (mu, phi, sigma) in Glicko-1 scale.
    """
    if not opponents:
        return mu, phi, sigma

    mu_p, phi_p = _to_glicko2(mu, phi)

    # Step 3: Compute variance v and delta
    v_inv = 0.0
    delta_sum = 0.0
    for opp_mu, opp_phi, score in opponents:
        opp_mu_p, opp_phi_p = _to_glicko2(opp_mu, opp_phi)
        g_val = _g(opp_phi_p)
        e_val = _E(mu_p, opp_mu_p, opp_phi_p)
        v_inv += g_val ** 2 * e_val * (1 - e_val)
        delta_sum += g_val * (score - e_val)

    v = 1.0 / v_inv if v_inv > 0 else 1e6
    delta = v * delta_sum

    # Step 4: Update volatility
    new_sigma = _update_volatility(sigma, delta, phi_p, v, tau)

    # Step 5: Update phi
    phi_star = math.sqrt(phi_p ** 2 + new_sigma ** 2)
    new_phi_p = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)

    # Step 6: Update mu
    new_mu_p = mu_p + new_phi_p ** 2 * delta_sum

    # Convert back to Glicko-1 scale
    new_mu, new_phi = _from_glicko2(new_mu_p, new_phi_p)
    new_phi = max(new_phi, phi_floor)

    return new_mu, new_phi, new_sigma


# ---------------------------------------------------------------------------
# Game aggregation
# ---------------------------------------------------------------------------
def _aggregate_game_opponents(
    game_pas: pd.DataFrame,
    snapshot_mu: dict[int, float],
    snapshot_phi: dict[int, float],
    cfg: dict[str, Any],
) -> dict[int, list[tuple[float, float, float]]]:
    """Group PAs by (player, opponent) within games, return opponent lists.

    For each game_pk in *game_pas*, PAs are grouped by (batter_id,
    pitcher_id).  The average PA score within each group becomes a single
    opponent observation.  Batters receive the raw PA score; pitchers
    receive ``1 - score``.

    Parameters
    ----------
    game_pas : pd.DataFrame
        PAs for one game date.  Must have columns: game_pk, batter_id,
        pitcher_id, events, woba_value.
    snapshot_mu : dict[int, float]
        Snapshot of mu ratings at the start of this date.
    snapshot_phi : dict[int, float]
        Snapshot of phi ratings at the start of this date.
    cfg : dict
        Config dict (for initial values).

    Returns
    -------
    dict[int, list[tuple[float, float, float]]]
        Mapping of player_id to list of (opp_mu, opp_phi, avg_score).
    """
    initial_mu = cfg["initial_mu"]
    initial_phi = cfg["initial_phi"]

    # Accumulate per-player opponent lists
    player_opponents: dict[int, list[tuple[float, float, float]]] = {}

    for game_pk, game_df in game_pas.groupby("game_pk"):
        # Batter side: group by (batter_id, pitcher_id)
        batter_groups: dict[tuple[int, int], list[float]] = {}
        for _, pa in game_df.iterrows():
            bid = int(pa["batter_id"])
            pid = int(pa["pitcher_id"])
            score = _pa_outcome_score(pa["events"], pa.get("woba_value"))
            if score is None:
                continue
            batter_groups.setdefault((bid, pid), []).append(score)

        # Build batter opponent entries
        for (bid, pid), scores in batter_groups.items():
            avg_score = sum(scores) / len(scores)
            opp_mu = snapshot_mu.get(pid, initial_mu)
            opp_phi = snapshot_phi.get(pid, initial_phi)
            player_opponents.setdefault(bid, []).append(
                (opp_mu, opp_phi, avg_score)
            )
            # Pitcher gets inverse
            bat_mu = snapshot_mu.get(bid, initial_mu)
            bat_phi = snapshot_phi.get(bid, initial_phi)
            player_opponents.setdefault(pid, []).append(
                (bat_mu, bat_phi, 1.0 - avg_score)
            )

    return player_opponents


# ---------------------------------------------------------------------------
# Season regression
# ---------------------------------------------------------------------------
def _regress_between_seasons(
    ratings: GlickoRatings,
    cfg: dict[str, Any],
) -> None:
    """Pull all ratings toward initial between seasons.

    Parameters
    ----------
    ratings : GlickoRatings
        Mutable ratings state.
    cfg : dict
        Config with regression parameters.
    """
    reg = cfg["season_regression_mu"]
    initial_mu = cfg["initial_mu"]
    phi_reset = cfg["season_phi_reset"]
    initial_sigma = cfg["initial_sigma"]

    for pid in list(ratings.mu.keys()):
        ratings.mu[pid] = ratings.mu[pid] * (1 - reg) + initial_mu * reg
        ratings.phi[pid] = min(ratings.phi[pid] * 1.5, phi_reset)
        ratings.sigma[pid] = initial_sigma

    logger.debug(
        "Regressed player ratings between seasons "
        "(mu=%.0f%%, phi_reset=%d)",
        reg * 100, phi_reset,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_ratings(
    pa_data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[GlickoRatings, pd.DataFrame]:
    """Process all PAs chronologically to compute Glicko-2 ratings.

    Parameters
    ----------
    pa_data : pd.DataFrame
        PA-level data.  Must have columns: ``game_pk``, ``game_date``,
        ``season``, ``batter_id``, ``pitcher_id``, ``events``,
        ``woba_value``.
    config : dict, optional
        Override config values.

    Returns
    -------
    tuple[GlickoRatings, pd.DataFrame]
        Final ratings and game-by-game history with columns:
        player_id, game_date, season, mu, phi, sigma, n_opponents,
        avg_opponent_mu, avg_score.
    """
    cfg = get_config()
    if config:
        cfg.update(config)

    initial_mu = cfg["initial_mu"]
    initial_phi = cfg["initial_phi"]
    initial_sigma = cfg["initial_sigma"]
    tau = cfg["tau"]
    phi_floor = cfg["phi_floor"]

    ratings = GlickoRatings()
    history_rows: list[dict[str, Any]] = []

    # Sort chronologically
    pa_sorted = pa_data.sort_values(["game_date", "game_pk"]).reset_index(
        drop=True
    )

    # Group by date for snapshot-based updating
    prev_season: int | None = None
    dates = pa_sorted.groupby("game_date", sort=True)

    n_dates = 0
    n_players_updated = 0

    for game_date, date_pas in dates:
        season = int(date_pas["season"].iloc[0])

        # Season transition: regress ratings
        if prev_season is not None and season != prev_season:
            _regress_between_seasons(ratings, cfg)
            logger.debug(
                "Season transition %d -> %d at %s",
                prev_season, season, game_date,
            )
        prev_season = season

        # Snapshot current ratings so within-date updates are simultaneous
        snapshot_mu = dict(ratings.mu)
        snapshot_phi = dict(ratings.phi)
        snapshot_sigma = dict(ratings.sigma)

        # Build opponent lists from all games on this date
        player_opponents = _aggregate_game_opponents(
            date_pas, snapshot_mu, snapshot_phi, cfg,
        )

        # Update each player
        date_str = str(game_date)
        for pid, opponents in player_opponents.items():
            if not opponents:
                continue

            old_mu = snapshot_mu.get(pid, initial_mu)
            old_phi = snapshot_phi.get(pid, initial_phi)
            old_sigma = snapshot_sigma.get(pid, initial_sigma)

            new_mu, new_phi, new_sigma = _update_player(
                old_mu, old_phi, old_sigma, opponents, tau, phi_floor,
            )

            ratings.mu[pid] = new_mu
            ratings.phi[pid] = new_phi
            ratings.sigma[pid] = new_sigma
            ratings.games_rated[pid] = ratings.games_rated.get(pid, 0) + 1
            ratings.last_seen[pid] = date_str

            # History row
            opp_mus = [o[0] for o in opponents]
            opp_scores = [o[2] for o in opponents]
            history_rows.append({
                "player_id": pid,
                "game_date": game_date,
                "season": season,
                "mu": new_mu,
                "phi": new_phi,
                "sigma": new_sigma,
                "n_opponents": len(opponents),
                "avg_opponent_mu": sum(opp_mus) / len(opp_mus),
                "avg_score": sum(opp_scores) / len(opp_scores),
            })
            n_players_updated += 1

        n_dates += 1

    history = pd.DataFrame(history_rows)
    n_players = len(ratings.mu)
    logger.info(
        "Glicko-2 computed: %d dates, %d unique players, %d total updates",
        n_dates, n_players, n_players_updated,
    )
    return ratings, history


# ---------------------------------------------------------------------------
# Extract current ratings
# ---------------------------------------------------------------------------
def get_current_ratings(
    ratings: GlickoRatings,
    player_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract final ratings as a sorted DataFrame.

    Parameters
    ----------
    ratings : GlickoRatings
        Final ratings from ``compute_ratings()``.
    player_info : pd.DataFrame, optional
        Player info lookup (must have ``player_id`` and ``player_name``
        columns) for merging names.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, mu, phi, sigma, games_rated.
        Optionally includes player_name.  Sorted by mu descending.
    """
    rows = []
    for pid in ratings.mu:
        rows.append({
            "player_id": pid,
            "mu": ratings.mu[pid],
            "phi": ratings.phi[pid],
            "sigma": ratings.sigma.get(pid, 0.06),
            "games_rated": ratings.games_rated.get(pid, 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if player_info is not None and "player_name" in player_info.columns:
        df = df.merge(
            player_info[["player_id", "player_name"]],
            on="player_id",
            how="left",
        )

    return df.sort_values("mu", ascending=False).reset_index(drop=True)
