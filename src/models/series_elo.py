"""
Series-based ELO rating system for MLB teams.

Instead of rating teams per-game, this module rates them per-SERIES.
A series is a consecutive set of games between the same two teams at
the same venue (regular season 2-4 game sets, postseason 3-7 game
series).  Winning a series is the unit of competition.

Simpler than ``team_elo.py`` — one rating per team rather than
component ratings (offense/pitching/SP).
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
def _load_series_elo_config() -> dict[str, Any]:
    """Load series_elo config block from model.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("series_elo", {})


_DEFAULTS: dict[str, Any] = {
    "initial_rating": 1500,
    "k_factor_regular": 8,
    "k_factor_wc": 12,       # Wild Card (best of 3)
    "k_factor_ds": 16,       # Division Series (best of 5)
    "k_factor_cs": 20,       # LCS (best of 7)
    "k_factor_ws": 24,       # World Series (best of 7)
    "season_regression": 0.35,
    "margin_bonus": 0.15,    # extra K multiplier for sweeps
    "game_elo_blend": 0.60,
    "series_elo_blend": 0.40,
}

# Map game_type codes to postseason K-factor config keys
_POSTSEASON_K_MAP: dict[str, str] = {
    "F": "k_factor_wc",   # Wild Card
    "D": "k_factor_ds",   # Division Series
    "L": "k_factor_cs",   # League Championship Series
    "W": "k_factor_ws",   # World Series
}


def get_config() -> dict[str, Any]:
    """Merge YAML config with defaults."""
    cfg = {**_DEFAULTS}
    cfg.update(_load_series_elo_config())
    return cfg


# ---------------------------------------------------------------------------
# Series detection
# ---------------------------------------------------------------------------
def detect_series(games: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive games between same two teams into series.

    A series is defined as consecutive games (by date) between the same
    two teams at the same venue. A gap of >2 days or venue change starts
    a new series.

    Parameters
    ----------
    games : pd.DataFrame
        Output of ``get_game_results()``.  Expected columns: game_pk,
        game_date, season, home_team_id, away_team_id, home_runs,
        away_runs, venue_id, game_type.

    Returns
    -------
    pd.DataFrame
        One row per series with: series_id, season, team_a, team_b,
        venue_id, game_type, start_date, end_date, games_played,
        team_a_wins, team_b_wins, series_winner (team_id or None if
        split).
    """
    df = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Canonical matchup key: always (min_id, max_id) so home/away order
    # doesn't matter for series continuity.
    df["team_a"] = df[["home_team_id", "away_team_id"]].min(axis=1)
    df["team_b"] = df[["home_team_id", "away_team_id"]].max(axis=1)

    # Group by (home_team_id, away_team_id) to detect series within each
    # directional matchup.  Using home_team_id (not venue_id) because
    # venue_id data has quality issues (catch-all IDs covering multiple
    # stadiums).  The home team reliably identifies where a series is
    # being played.  Games from different matchups are interleaved on
    # the same date, so series detection must happen within each
    # matchup group separately.
    matchup_key = (
        df["home_team_id"].astype(str) + "_"
        + df["away_team_id"].astype(str)
    )
    df["_matchup_key"] = matchup_key

    # Within each matchup group (sorted by date), a gap > 2 days
    # starts a new series.  Cumsum within each group to get per-group
    # series numbers, then combine with matchup_key for globally unique
    # series IDs.
    df["_prev_date"] = df.groupby("_matchup_key")["game_date"].shift(1)
    df["_gap_days"] = (df["game_date"] - df["_prev_date"]).dt.days
    df["_new_series"] = df["_prev_date"].isna() | (df["_gap_days"] > 2)
    df["_group_series_num"] = df.groupby("_matchup_key")["_new_series"].cumsum()

    # Create globally unique series_id: combine matchup key + group
    # series number, then factorize to get clean integer IDs.
    df["_series_key"] = df["_matchup_key"] + "_" + df["_group_series_num"].astype(str)
    df["series_id"] = df["_series_key"].factorize()[0] + 1

    # Clean up temp columns
    df.drop(
        columns=["_matchup_key", "_prev_date", "_gap_days",
                 "_new_series", "_group_series_num", "_series_key"],
        inplace=True,
    )

    # Determine winner per game (using original home/away ids)
    df["game_winner"] = np.where(
        df["home_runs"] > df["away_runs"],
        df["home_team_id"],
        np.where(
            df["away_runs"] > df["home_runs"],
            df["away_team_id"],
            0,  # tie (exceedingly rare in MLB)
        ),
    )

    # Aggregate per series
    series_rows: list[dict[str, Any]] = []
    for sid, grp in df.groupby("series_id"):
        team_a = int(grp["team_a"].iloc[0])
        team_b = int(grp["team_b"].iloc[0])
        venue_id = grp["venue_id"].iloc[0]
        season = int(grp["season"].iloc[0])
        game_type = str(grp["game_type"].iloc[0])
        start_date = grp["game_date"].min()
        end_date = grp["game_date"].max()
        games_played = len(grp)

        team_a_wins = int((grp["game_winner"] == team_a).sum())
        team_b_wins = int((grp["game_winner"] == team_b).sum())

        # Series winner: team with majority of wins
        if team_a_wins > team_b_wins:
            series_winner = team_a
        elif team_b_wins > team_a_wins:
            series_winner = team_b
        else:
            series_winner = None  # tied (e.g. 2-2 in a 4-game set)

        is_sweep = (
            series_winner is not None
            and games_played >= 2
            and (team_a_wins == 0 or team_b_wins == 0)
        )

        series_rows.append({
            "series_id": int(sid),
            "season": season,
            "team_a": team_a,
            "team_b": team_b,
            "venue_id": venue_id,
            "game_type": game_type,
            "start_date": start_date,
            "end_date": end_date,
            "games_played": games_played,
            "team_a_wins": team_a_wins,
            "team_b_wins": team_b_wins,
            "series_winner": series_winner,
            "is_sweep": is_sweep,
        })

    result = pd.DataFrame(series_rows)
    logger.info(
        "Detected %d series from %d games (%d tied, excluded from updates)",
        len(result), len(df),
        int((result["series_winner"].isna()).sum()),
    )
    return result


# ---------------------------------------------------------------------------
# Series ELO state
# ---------------------------------------------------------------------------
@dataclass
class SeriesRatings:
    """Mutable container for series-based ELO ratings."""

    mu: dict[int, float] = field(default_factory=dict)
    series_count: dict[int, int] = field(default_factory=dict)

    def get_mu(self, team_id: int, initial: float = 1500.0) -> float:
        """Get team's current series ELO rating."""
        return self.mu.setdefault(team_id, initial)

    def increment_count(self, team_id: int) -> None:
        """Increment the series count for a team."""
        self.series_count[team_id] = self.series_count.get(team_id, 0) + 1


# ---------------------------------------------------------------------------
# Core ELO math
# ---------------------------------------------------------------------------
def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard ELO expected score for A vs B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _get_k_factor(game_type: str, cfg: dict[str, Any]) -> float:
    """K-factor for a series based on game type.

    Regular season series use a flat K.  Postseason series use
    escalating K-factors reflecting the stronger signal in playoff
    outcomes.
    """
    cfg_key = _POSTSEASON_K_MAP.get(game_type)
    if cfg_key is not None:
        return float(cfg[cfg_key])
    return float(cfg["k_factor_regular"])


# ---------------------------------------------------------------------------
# Process a single series
# ---------------------------------------------------------------------------
def _process_series(
    row: pd.Series,
    ratings: SeriesRatings,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Update ratings for one series, return history row.

    Returns None for tied series (no rating update).
    """
    initial = cfg["initial_rating"]
    margin_bonus = cfg["margin_bonus"]

    team_a = int(row["team_a"])
    team_b = int(row["team_b"])
    series_winner = row["series_winner"]
    is_sweep = bool(row["is_sweep"])
    game_type = str(row.get("game_type", "R"))

    # Skip tied series (no signal)
    if pd.isna(series_winner) or series_winner is None:
        return None

    series_winner = int(series_winner)

    # Current ratings
    mu_a = ratings.get_mu(team_a, initial)
    mu_b = ratings.get_mu(team_b, initial)

    # Expected scores
    exp_a = _expected_score(mu_a, mu_b)
    exp_b = 1.0 - exp_a

    # Actual: 1.0 for winner, 0.0 for loser
    actual_a = 1.0 if series_winner == team_a else 0.0
    actual_b = 1.0 - actual_a

    # K-factor with sweep bonus
    k = _get_k_factor(game_type, cfg)
    if is_sweep:
        k *= (1.0 + margin_bonus)

    # Update ratings
    ratings.mu[team_a] = mu_a + k * (actual_a - exp_a)
    ratings.mu[team_b] = mu_b + k * (actual_b - exp_b)

    # Track series played
    ratings.increment_count(team_a)
    ratings.increment_count(team_b)

    return {
        "series_id": int(row["series_id"]),
        "season": int(row["season"]),
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "team_a": team_a,
        "team_b": team_b,
        "game_type": game_type,
        "games_played": int(row["games_played"]),
        "team_a_wins": int(row["team_a_wins"]),
        "team_b_wins": int(row["team_b_wins"]),
        "series_winner": series_winner,
        "is_sweep": is_sweep,
        "mu_a_before": mu_a,
        "mu_b_before": mu_b,
        "mu_a_after": ratings.mu[team_a],
        "mu_b_after": ratings.mu[team_b],
        "expected_a": exp_a,
        "k_used": k,
    }


# ---------------------------------------------------------------------------
# Season regression
# ---------------------------------------------------------------------------
def _regress_between_seasons(
    ratings: SeriesRatings,
    prev_season: int,
    cfg: dict[str, Any],
) -> None:
    """Pull all ratings toward initial between seasons."""
    initial = cfg["initial_rating"]
    reg = cfg["season_regression"]

    for team_id in list(ratings.mu.keys()):
        ratings.mu[team_id] = initial * reg + ratings.mu[team_id] * (1 - reg)

    logger.debug(
        "Regressed series ratings after season %d (%.0f%%)",
        prev_season, reg * 100,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_series_elo(
    series_data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[SeriesRatings, pd.DataFrame]:
    """Compute series-based ELO ratings.

    Parameters
    ----------
    series_data : pd.DataFrame
        Output of ``detect_series()``.  Must be sorted chronologically
        (by start_date, series_id).
    config : dict, optional
        Override config values.

    Returns
    -------
    tuple[SeriesRatings, pd.DataFrame]
        Final ratings state and full series-by-series history.
    """
    cfg = get_config()
    if config:
        cfg.update(config)

    ratings = SeriesRatings()

    # Sort chronologically
    series = series_data.sort_values(["start_date", "series_id"]).reset_index(drop=True)

    history_rows: list[dict[str, Any]] = []
    prev_season: int | None = None

    for _, row in series.iterrows():
        season = int(row["season"])

        # Season transition: regress
        if prev_season is not None and season != prev_season:
            _regress_between_seasons(ratings, prev_season, cfg)
        prev_season = season

        result = _process_series(row, ratings, cfg)
        if result is not None:
            history_rows.append(result)

    history = pd.DataFrame(history_rows)
    n_series = len(series)
    n_decided = len(history)
    n_teams = len(ratings.mu)

    logger.info(
        "Series ELO computed: %d total series, %d decided, %d teams",
        n_series, n_decided, n_teams,
    )
    return ratings, history


# ---------------------------------------------------------------------------
# Extract current ratings
# ---------------------------------------------------------------------------
def get_current_ratings(
    ratings: SeriesRatings,
    team_info: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract current series ELO ratings as a DataFrame.

    Parameters
    ----------
    ratings : SeriesRatings
        Final ratings from ``compute_series_elo()``.
    team_info : pd.DataFrame, optional
        dim_team lookup for names/abbreviations.
    config : dict, optional
        Config for initial_rating.

    Returns
    -------
    pd.DataFrame
        Sorted by series_mu descending.
    """
    cfg = get_config()
    if config:
        cfg.update(config)

    rows = []
    for tid, mu in ratings.mu.items():
        rows.append({
            "team_id": tid,
            "series_mu": mu,
            "series_count": ratings.series_count.get(tid, 0),
        })

    df = pd.DataFrame(rows)
    df["series_rank"] = (
        df["series_mu"].rank(ascending=False, method="min").astype(int)
    )

    if team_info is not None:
        df = df.merge(
            team_info[["team_id", "abbreviation", "team_name", "division", "league"]],
            on="team_id",
            how="left",
        )

    return df.sort_values("series_mu", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pre-season projection
# ---------------------------------------------------------------------------
def project_preseason_elo(
    ratings: SeriesRatings,
    config: dict[str, Any] | None = None,
) -> SeriesRatings:
    """Apply between-season regression to get pre-season ratings.

    Returns a *new* SeriesRatings (does not mutate input).
    """
    cfg = get_config()
    if config:
        cfg.update(config)
    initial = cfg["initial_rating"]
    reg = cfg["season_regression"]

    projected = SeriesRatings()
    for tid, val in ratings.mu.items():
        projected.mu[tid] = initial * reg + val * (1 - reg)
    projected.series_count = dict(ratings.series_count)

    return projected
