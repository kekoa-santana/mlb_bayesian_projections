"""
Component ELO system for MLB teams.

Baseball's unique problem: ~50% of game outcome variance comes from the
starting pitcher, so a single team rating is insufficient.  This system
maintains three separate ratings per team:

1. **Offense ELO** — park-adjusted runs scored vs expected
2. **Pitching ELO** — park-adjusted runs allowed vs expected
3. **SP ELO** — individual starting pitcher ratings

Game-day effective rating blends all three, elevating teams when they
start their ace and penalizing fifth-starter days.
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
def _load_elo_config() -> dict[str, Any]:
    """Load team_elo config block from model.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "model.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("team_elo", {})


_DEFAULTS = {
    "initial_rating": 1500,
    "home_advantage": 25,
    "k_factor_base": 8,
    "k_factor_early_season": 12,
    "k_factor_early_threshold": 30,
    # Postseason K-factors (elevated: playoff games are high-signal)
    "k_factor_wc": 10,   # Wild Card
    "k_factor_ds": 14,   # Division Series
    "k_factor_cs": 18,   # League Championship Series
    "k_factor_ws": 22,   # World Series
    "season_regression": 0.15,
    "margin_log_base": 1.5,
    "sp_weight": 0.40,
    "sp_k_factor": 6,
    "sp_regression": 0.25,
}


def get_config() -> dict[str, Any]:
    """Merge YAML config with defaults."""
    cfg = {**_DEFAULTS}
    cfg.update(_load_elo_config())
    return cfg


# ---------------------------------------------------------------------------
# ELO ratings state
# ---------------------------------------------------------------------------
@dataclass
class ELORatings:
    """Mutable container for all ELO ratings."""

    offense: dict[int, float] = field(default_factory=dict)
    pitching: dict[int, float] = field(default_factory=dict)
    sp: dict[int, float] = field(default_factory=dict)
    # Track games played per team per season (for K-factor schedule)
    team_season_games: dict[tuple[int, int], int] = field(default_factory=dict)

    def get_offense(self, team_id: int, initial: float = 1500.0) -> float:
        return self.offense.setdefault(team_id, initial)

    def get_pitching(self, team_id: int, initial: float = 1500.0) -> float:
        return self.pitching.setdefault(team_id, initial)

    def get_sp(self, sp_id: int, initial: float = 1500.0) -> float:
        return self.sp.setdefault(sp_id, initial)

    def composite(self, team_id: int, initial: float = 1500.0) -> float:
        """Simple composite = (offense + pitching) / 2."""
        return (
            self.get_offense(team_id, initial)
            + self.get_pitching(team_id, initial)
        ) / 2

    def effective(
        self,
        team_id: int,
        sp_id: int | None,
        sp_weight: float,
        initial: float = 1500.0,
    ) -> float:
        """Game-day effective rating incorporating specific SP."""
        off = self.get_offense(team_id, initial)
        pit_base = self.get_pitching(team_id, initial)

        if sp_id is None or sp_id not in self.sp:
            return off + pit_base - initial  # center around initial

        # Compute team avg SP ELO
        team_sps = self._team_sp_ids.get(team_id, set())
        if team_sps:
            avg_sp = np.mean([self.sp.get(sid, initial) for sid in team_sps])
        else:
            avg_sp = initial

        sp_elo = self.get_sp(sp_id, initial)
        pit_adj = pit_base + sp_weight * (sp_elo - avg_sp)

        return off + pit_adj - initial

    # Track which SPs belong to which team (updated during computation)
    _team_sp_ids: dict[int, set[int]] = field(default_factory=dict)

    def register_sp(self, team_id: int, sp_id: int) -> None:
        """Associate a SP with a team (for avg SP calc)."""
        self._team_sp_ids.setdefault(team_id, set()).add(sp_id)


# ---------------------------------------------------------------------------
# Core ELO math
# ---------------------------------------------------------------------------
def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard ELO expected score for A vs B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _margin_multiplier(margin: int, log_base: float = 1.5) -> float:
    """Margin-of-victory multiplier: ln(1 + |margin| / base).

    1-run: 0.81, 3-run: 1.39, 10-run: 1.95.
    """
    return math.log(1.0 + abs(margin) / log_base)


_POSTSEASON_K_MAP: dict[str, str] = {
    "F": "k_factor_wc",  # Wild Card
    "D": "k_factor_ds",  # Division Series
    "L": "k_factor_cs",  # League Championship Series
    "W": "k_factor_ws",  # World Series
}


def _get_k_factor(
    team_id: int,
    season: int,
    ratings: ELORatings,
    cfg: dict[str, Any],
    game_type: str = "R",
) -> float:
    """K-factor with early-season boost and postseason escalation.

    Parameters
    ----------
    team_id : int
        Team identifier.
    season : int
        Season year.
    ratings : ELORatings
        Current ratings state (for game counter lookup).
    cfg : dict
        Config dict with K-factor values.
    game_type : str
        Game type code: 'R' (regular), 'F' (Wild Card), 'D' (Division
        Series), 'L' (League Championship Series), 'W' (World Series).
    """
    # Postseason games use fixed elevated K-factors
    cfg_key = _POSTSEASON_K_MAP.get(game_type)
    if cfg_key is not None:
        return cfg[cfg_key]

    # Regular season: early-season boost then settle to base
    games = ratings.team_season_games.get((team_id, season), 0)
    if games < cfg["k_factor_early_threshold"]:
        return cfg["k_factor_early_season"]
    return cfg["k_factor_base"]


# ---------------------------------------------------------------------------
# Process a single game
# ---------------------------------------------------------------------------
def _process_game(
    row: pd.Series,
    ratings: ELORatings,
    venue_factors: dict[int, float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Update ratings for one game, return history row."""
    initial = cfg["initial_rating"]
    ha = cfg["home_advantage"]
    sp_weight = cfg["sp_weight"]
    log_base = cfg["margin_log_base"]

    home_id = int(row["home_team_id"])
    away_id = int(row["away_team_id"])
    home_runs = int(row["home_runs"])
    away_runs = int(row["away_runs"])
    season = int(row["season"])
    game_type = str(row.get("game_type", "R"))
    venue_id = row.get("venue_id")

    home_sp = row.get("home_sp_id")
    away_sp = row.get("away_sp_id")
    if pd.notna(home_sp):
        home_sp = int(home_sp)
        ratings.register_sp(home_id, home_sp)
    else:
        home_sp = None
    if pd.notna(away_sp):
        away_sp = int(away_sp)
        ratings.register_sp(away_id, away_sp)
    else:
        away_sp = None

    # Park factor
    pf = venue_factors.get(int(venue_id), 1.0) if pd.notna(venue_id) else 1.0
    pf_sqrt = math.sqrt(pf)

    # Current ratings
    home_off = ratings.get_offense(home_id, initial)
    away_off = ratings.get_offense(away_id, initial)
    home_pit = ratings.get_pitching(home_id, initial)
    away_pit = ratings.get_pitching(away_id, initial)

    # Effective ratings (with SP)
    home_eff = ratings.effective(home_id, home_sp, sp_weight, initial) + ha
    away_eff = ratings.effective(away_id, away_sp, sp_weight, initial)

    # Expected scores
    exp_home = _expected_score(home_eff, away_eff)
    exp_away = 1.0 - exp_home

    # Actual score (binary W/L for ELO; margin used as multiplier)
    if home_runs > away_runs:
        actual_home, actual_away = 1.0, 0.0
    elif away_runs > home_runs:
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home, actual_away = 0.5, 0.5  # ties (rare in MLB)

    margin = abs(home_runs - away_runs)
    # Cap margin multiplier in postseason — the signal is W/L, not blowout size.
    # Postseason game margins are noisy (bullpen management, series context).
    if game_type in _POSTSEASON_K_MAP:
        mov = _margin_multiplier(min(margin, 3), log_base)
    else:
        mov = _margin_multiplier(margin, log_base)

    # K-factors (pass game_type for postseason escalation)
    k_home = _get_k_factor(home_id, season, ratings, cfg, game_type)
    k_away = _get_k_factor(away_id, season, ratings, cfg, game_type)
    # Scale SP K-factor proportionally for postseason games
    k_sp_base = cfg["sp_k_factor"]
    if game_type in _POSTSEASON_K_MAP:
        ps_team_k = cfg[_POSTSEASON_K_MAP[game_type]]
        k_sp = k_sp_base * (ps_team_k / cfg["k_factor_base"])
    else:
        k_sp = k_sp_base

    # --- Split credit between offense and pitching ---
    # Use sigmoid of park-adjusted runs to get component "actuals" on [0, 1].
    # Offense actual: higher when team scored more.
    # Pitching actual: higher when team allowed fewer.
    # This lets offense and pitching ratings diverge.
    league_rpg = 4.5
    scale = 3.0

    def _runs_to_score(runs: float) -> float:
        return 1.0 / (1.0 + math.exp(-(runs - league_rpg) / scale))

    # Park-adjust: divide runs by sqrt(pf)
    home_scored_adj = home_runs / pf_sqrt
    away_scored_adj = away_runs / pf_sqrt

    off_actual_home = _runs_to_score(home_scored_adj)
    off_actual_away = _runs_to_score(away_scored_adj)
    pit_actual_home = 1 - _runs_to_score(away_scored_adj)  # good if opponent scored little
    pit_actual_away = 1 - _runs_to_score(home_scored_adj)

    # --- Component-specific expectations (cross-component) ---
    # Offense expectation depends on the OPPONENT'S pitching, not their
    # offense.  A good offense facing bad pitching should expect to score a
    # lot — and only gets credit for exceeding that.  Pitching expectation
    # depends on the OPPONENT'S offense for the same reason.
    # Without cross-component expectations, good pitching teams have
    # inflated composite expectations that suppress their offense ELO.
    off_exp_home = _expected_score(home_off + ha, away_pit)
    off_exp_away = _expected_score(away_off, home_pit + ha)
    pit_exp_home = _expected_score(home_pit + ha, away_off)
    pit_exp_away = _expected_score(away_pit, home_off + ha)

    # --- Update OFFENSE ratings ---
    off_surprise_home = off_actual_home - off_exp_home
    off_surprise_away = off_actual_away - off_exp_away

    ratings.offense[home_id] = home_off + k_home * mov * off_surprise_home
    ratings.offense[away_id] = away_off + k_away * mov * off_surprise_away

    # --- Update PITCHING ratings ---
    pit_surprise_home = pit_actual_home - pit_exp_home
    pit_surprise_away = pit_actual_away - pit_exp_away

    ratings.pitching[home_id] = home_pit + k_home * mov * pit_surprise_home
    ratings.pitching[away_id] = away_pit + k_away * mov * pit_surprise_away

    # --- Update SP ratings ---
    # SP uses the team pitching expectation (my pitching vs their offense)
    # so SPs aren't penalised for facing a strong lineup — they're expected
    # to allow more runs in that spot.
    if home_sp is not None:
        sp_elo_h = ratings.get_sp(home_sp, initial)
        sp_exp_h = _expected_score(sp_elo_h + ha, away_off)
        ratings.sp[home_sp] = sp_elo_h + k_sp * mov * (pit_actual_home - sp_exp_h)
    if away_sp is not None:
        sp_elo_a = ratings.get_sp(away_sp, initial)
        sp_exp_a = _expected_score(sp_elo_a, home_off + ha)
        ratings.sp[away_sp] = sp_elo_a + k_sp * mov * (pit_actual_away - sp_exp_a)

    # Track games — only regular season counts toward early-season threshold
    if game_type == "R":
        ratings.team_season_games[(home_id, season)] = (
            ratings.team_season_games.get((home_id, season), 0) + 1
        )
        ratings.team_season_games[(away_id, season)] = (
            ratings.team_season_games.get((away_id, season), 0) + 1
        )

    # Build history rows (one per team)
    base = {
        "game_pk": row["game_pk"],
        "game_date": row["game_date"],
        "season": season,
        "game_type": game_type,
    }
    home_row = {
        **base,
        "team_id": home_id,
        "offense_elo": ratings.offense[home_id],
        "pitching_elo": ratings.pitching[home_id],
        "composite_elo": ratings.composite(home_id, initial),
        "sp_id": home_sp,
        "sp_elo": ratings.sp.get(home_sp, initial) if home_sp else None,
        "effective_elo": ratings.effective(home_id, home_sp, sp_weight, initial),
        "opponent_id": away_id,
        "runs_scored": home_runs,
        "runs_allowed": away_runs,
        "is_home": True,
        "win": actual_home == 1.0,
        "expected_win": exp_home,
    }
    away_row = {
        **base,
        "team_id": away_id,
        "offense_elo": ratings.offense[away_id],
        "pitching_elo": ratings.pitching[away_id],
        "composite_elo": ratings.composite(away_id, initial),
        "sp_id": away_sp,
        "sp_elo": ratings.sp.get(away_sp, initial) if away_sp else None,
        "effective_elo": ratings.effective(away_id, away_sp, sp_weight, initial),
        "opponent_id": home_id,
        "runs_scored": away_runs,
        "runs_allowed": home_runs,
        "is_home": False,
        "win": actual_away == 1.0,
        "expected_win": exp_away,
    }
    return {"home": home_row, "away": away_row}


# ---------------------------------------------------------------------------
# Season regression
# ---------------------------------------------------------------------------
def _regress_between_seasons(
    ratings: ELORatings,
    prev_season: int,
    cfg: dict[str, Any],
) -> None:
    """Pull all ratings toward initial between seasons."""
    initial = cfg["initial_rating"]
    team_reg = cfg["season_regression"]
    sp_reg = cfg["sp_regression"]

    for team_id in list(ratings.offense.keys()):
        ratings.offense[team_id] = (
            initial * team_reg + ratings.offense[team_id] * (1 - team_reg)
        )
    for team_id in list(ratings.pitching.keys()):
        ratings.pitching[team_id] = (
            initial * team_reg + ratings.pitching[team_id] * (1 - team_reg)
        )
    for sp_id in list(ratings.sp.keys()):
        ratings.sp[sp_id] = (
            initial * sp_reg + ratings.sp[sp_id] * (1 - sp_reg)
        )
    # Clear team-SP associations for new season
    ratings._team_sp_ids.clear()

    logger.debug(
        "Regressed ratings after season %d (team=%.0f%%, SP=%.0f%%)",
        prev_season, team_reg * 100, sp_reg * 100,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_elo_history(
    game_results: pd.DataFrame,
    venue_factors: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[ELORatings, pd.DataFrame]:
    """Compute full ELO history from game results.

    Parameters
    ----------
    game_results : pd.DataFrame
        Output of ``get_game_results()``.  If the DataFrame contains a
        ``game_type`` column, postseason games ('F', 'D', 'L', 'W') use
        escalated K-factors.  If the column is absent, all games are
        treated as regular season ('R').
    venue_factors : pd.DataFrame, optional
        Output of ``get_venue_run_factors()``.  If None, all parks = 1.0.
    config : dict, optional
        Override config values.

    Returns
    -------
    tuple[ELORatings, pd.DataFrame]
        Final ratings state and full game-by-game history.
    """
    cfg = get_config()
    if config:
        cfg.update(config)

    ratings = ELORatings()

    # Build venue factor lookup
    vf_dict: dict[int, float] = {}
    if venue_factors is not None:
        vf_dict = dict(zip(
            venue_factors["venue_id"].astype(int),
            venue_factors["run_factor"],
        ))

    # Sort chronologically
    games = game_results.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    history_rows: list[dict[str, Any]] = []
    prev_season: int | None = None

    for _, row in games.iterrows():
        season = int(row["season"])
        # Season transition: regress
        if prev_season is not None and season != prev_season:
            _regress_between_seasons(ratings, prev_season, cfg)
        prev_season = season

        result = _process_game(row, ratings, vf_dict, cfg)
        history_rows.append(result["home"])
        history_rows.append(result["away"])

    history = pd.DataFrame(history_rows)
    n_games = len(games)
    n_teams = len(set(ratings.offense.keys()))
    n_sps = len(ratings.sp)
    logger.info(
        "ELO computed: %d games, %d teams, %d SPs", n_games, n_teams, n_sps,
    )
    return ratings, history


# ---------------------------------------------------------------------------
# Extract current ratings
# ---------------------------------------------------------------------------
def get_current_ratings(
    ratings: ELORatings,
    team_info: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract current ELO ratings as a DataFrame.

    Parameters
    ----------
    ratings : ELORatings
        Final ratings from ``compute_elo_history()``.
    team_info : pd.DataFrame, optional
        dim_team lookup for names/abbreviations.
    config : dict, optional
        Config for initial_rating.

    Returns
    -------
    pd.DataFrame
        Sorted by composite_elo descending.
    """
    cfg = get_config()
    if config:
        cfg.update(config)
    initial = cfg["initial_rating"]

    rows = []
    all_teams = set(ratings.offense.keys()) | set(ratings.pitching.keys())
    for tid in all_teams:
        rows.append({
            "team_id": tid,
            "offense_elo": ratings.get_offense(tid, initial),
            "pitching_elo": ratings.get_pitching(tid, initial),
            "composite_elo": ratings.composite(tid, initial),
        })

    df = pd.DataFrame(rows)
    df["offense_rank"] = df["offense_elo"].rank(ascending=False, method="min").astype(int)
    df["pitching_rank"] = df["pitching_elo"].rank(ascending=False, method="min").astype(int)
    df["composite_rank"] = df["composite_elo"].rank(ascending=False, method="min").astype(int)

    if team_info is not None:
        df = df.merge(
            team_info[["team_id", "abbreviation", "team_name", "division", "league"]],
            on="team_id",
            how="left",
        )

    return df.sort_values("composite_elo", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pre-season projection
# ---------------------------------------------------------------------------
def project_preseason_elo(
    ratings: ELORatings,
    config: dict[str, Any] | None = None,
) -> ELORatings:
    """Apply between-season regression to get pre-season ratings.

    Returns a *new* ELORatings (does not mutate input).
    """
    cfg = get_config()
    if config:
        cfg.update(config)
    initial = cfg["initial_rating"]
    team_reg = cfg["season_regression"]
    sp_reg = cfg["sp_regression"]

    projected = ELORatings()
    for tid, val in ratings.offense.items():
        projected.offense[tid] = initial * team_reg + val * (1 - team_reg)
    for tid, val in ratings.pitching.items():
        projected.pitching[tid] = initial * team_reg + val * (1 - team_reg)
    for sid, val in ratings.sp.items():
        projected.sp[sid] = initial * sp_reg + val * (1 - sp_reg)
    # Copy team-SP associations
    for tid, sps in ratings._team_sp_ids.items():
        projected._team_sp_ids[tid] = sps.copy()

    return projected


# ---------------------------------------------------------------------------
# Roster talent adjustment
# ---------------------------------------------------------------------------
def apply_roster_talent_adjustment(
    ratings: ELORatings,
    current_roster: pd.DataFrame,
    previous_roster: pd.DataFrame,
    hitter_quality: pd.DataFrame,
    pitcher_quality: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> ELORatings:
    """Shift pre-season ELO based on offseason roster talent changes.

    For each team, computes the average quality (tdd_value_score) of hitters
    and pitchers on the 2026 roster vs the 2025 roster.  The delta is scaled
    into ELO points and applied additively to offense/pitching ratings.

    Parameters
    ----------
    ratings : ELORatings
        Pre-season ratings (output of ``project_preseason_elo()``).
        **Mutated in-place** and returned.
    current_roster : pd.DataFrame
        2026 roster with columns ``player_id``, ``team_abbr``.
    previous_roster : pd.DataFrame
        2025 roster with columns ``player_id``, ``team_abbr``.
    hitter_quality : pd.DataFrame
        Hitter rankings with ``batter_id`` and ``tdd_value_score``.
    pitcher_quality : pd.DataFrame
        Pitcher rankings with ``pitcher_id`` and ``tdd_value_score``.
    config : dict, optional
        Override config values.  Relevant keys:
        ``roster_adjustment_weight`` (max ELO per talent-unit delta,
        default 15) and ``roster_adjustment_cap`` (max total shift per
        team per component, default 40).

    Returns
    -------
    ELORatings
        The same ``ratings`` object, adjusted in-place.
    """
    cfg = get_config()
    if config:
        cfg.update(config)

    adjustment_weight: float = cfg.get("roster_adjustment_weight", 15.0)
    adjustment_cap: float = cfg.get("roster_adjustment_cap", 40.0)

    # Build abbr -> team_id mapping from existing ratings + any info
    # We rely on the caller's roster having team_abbr and the ELO ratings
    # using integer team_ids.  Build lookup from current_roster's org_id
    # or from a supplied mapping.
    abbr_to_tid: dict[str, int] = {}
    if "org_id" in current_roster.columns:
        pairs = current_roster[["team_abbr", "org_id"]].drop_duplicates()
        abbr_to_tid = dict(zip(pairs["team_abbr"], pairs["org_id"].astype(int)))
    elif "team_id" in current_roster.columns:
        pairs = current_roster[["team_abbr", "team_id"]].drop_duplicates()
        abbr_to_tid = dict(zip(pairs["team_abbr"], pairs["team_id"].astype(int)))

    if not abbr_to_tid:
        logger.warning("Cannot map team_abbr to team_id — skipping roster adjustment")
        return ratings

    # Build quality lookups  {player_id: tdd_value_score}
    hitter_q: dict[int, float] = {}
    if "batter_id" in hitter_quality.columns and "tdd_value_score" in hitter_quality.columns:
        valid = hitter_quality.dropna(subset=["tdd_value_score"])
        hitter_q = dict(zip(valid["batter_id"].astype(int), valid["tdd_value_score"]))

    pitcher_q: dict[int, float] = {}
    if "pitcher_id" in pitcher_quality.columns and "tdd_value_score" in pitcher_quality.columns:
        valid = pitcher_quality.dropna(subset=["tdd_value_score"])
        pitcher_q = dict(zip(valid["pitcher_id"].astype(int), valid["tdd_value_score"]))

    # Determine which players are hitters vs pitchers based on position
    _pitcher_positions = {"SP", "RP", "P"}

    def _is_pitcher_pos(pos: str | None) -> bool:
        return str(pos).upper() in _pitcher_positions if pos else False

    # Build team -> set of player_ids for current and previous rosters
    def _team_players(roster_df: pd.DataFrame) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for _, row in roster_df.iterrows():
            abbr = row["team_abbr"]
            pid = int(row["player_id"])
            out.setdefault(abbr, set()).add(pid)
        return out

    cur_by_team = _team_players(current_roster)
    prev_by_team = _team_players(previous_roster)

    # Determine pitcher/hitter split from current roster positions
    pid_is_pitcher: dict[int, bool] = {}
    if "primary_position" in current_roster.columns:
        for _, row in current_roster.iterrows():
            pid_is_pitcher[int(row["player_id"])] = _is_pitcher_pos(
                row.get("primary_position")
            )
    # Also mark from previous roster (for players who left)
    if "primary_position" in previous_roster.columns:
        for _, row in previous_roster.iterrows():
            pid = int(row["player_id"])
            if pid not in pid_is_pitcher:
                pid_is_pitcher[pid] = _is_pitcher_pos(
                    row.get("primary_position")
                )
    # Fallback: if a player is in pitcher_q, they are a pitcher
    for pid in pitcher_q:
        if pid not in pid_is_pitcher:
            pid_is_pitcher[pid] = True
    for pid in hitter_q:
        if pid not in pid_is_pitcher:
            pid_is_pitcher[pid] = False

    def _avg_quality(
        player_ids: set[int],
        quality_lookup: dict[int, float],
    ) -> float | None:
        scores = [quality_lookup[pid] for pid in player_ids if pid in quality_lookup]
        return np.mean(scores) if scores else None

    all_teams = set(cur_by_team.keys()) | set(prev_by_team.keys())
    adjustments: list[dict[str, Any]] = []

    for abbr in sorted(all_teams):
        tid = abbr_to_tid.get(abbr)
        if tid is None or tid not in ratings.offense:
            continue

        cur_pids = cur_by_team.get(abbr, set())
        prev_pids = prev_by_team.get(abbr, set())

        # Split into hitters / pitchers
        cur_hitters = {p for p in cur_pids if not pid_is_pitcher.get(p, False)}
        prev_hitters = {p for p in prev_pids if not pid_is_pitcher.get(p, False)}
        cur_pitchers = {p for p in cur_pids if pid_is_pitcher.get(p, False)}
        prev_pitchers = {p for p in prev_pids if pid_is_pitcher.get(p, False)}

        # Average quality for each group
        cur_h_avg = _avg_quality(cur_hitters, hitter_q)
        prev_h_avg = _avg_quality(prev_hitters, hitter_q)
        cur_p_avg = _avg_quality(cur_pitchers, pitcher_q)
        prev_p_avg = _avg_quality(prev_pitchers, pitcher_q)

        # Offense delta (hitter quality change)
        off_delta = 0.0
        if cur_h_avg is not None and prev_h_avg is not None:
            off_delta = (cur_h_avg - prev_h_avg) * adjustment_weight
            off_delta = np.clip(off_delta, -adjustment_cap, adjustment_cap)

        # Pitching delta
        pit_delta = 0.0
        if cur_p_avg is not None and prev_p_avg is not None:
            pit_delta = (cur_p_avg - prev_p_avg) * adjustment_weight
            pit_delta = np.clip(pit_delta, -adjustment_cap, adjustment_cap)

        ratings.offense[tid] = ratings.offense[tid] + off_delta
        ratings.pitching[tid] = ratings.pitching[tid] + pit_delta

        adjustments.append({
            "team_abbr": abbr,
            "team_id": tid,
            "offense_delta": round(off_delta, 2),
            "pitching_delta": round(pit_delta, 2),
            "cur_hitter_avg": round(cur_h_avg, 4) if cur_h_avg else None,
            "prev_hitter_avg": round(prev_h_avg, 4) if prev_h_avg else None,
            "cur_pitcher_avg": round(cur_p_avg, 4) if cur_p_avg else None,
            "prev_pitcher_avg": round(prev_p_avg, 4) if prev_p_avg else None,
        })

    if adjustments:
        adj_df = pd.DataFrame(adjustments)
        nonzero = adj_df[
            (adj_df["offense_delta"].abs() > 0.5) | (adj_df["pitching_delta"].abs() > 0.5)
        ]
        if not nonzero.empty:
            logger.info(
                "Roster talent adjustments applied to %d teams (of %d):",
                len(nonzero), len(adj_df),
            )
            for _, row in nonzero.iterrows():
                logger.info(
                    "  %-4s  offense %+.1f  pitching %+.1f",
                    row["team_abbr"], row["offense_delta"], row["pitching_delta"],
                )
        else:
            logger.info("Roster talent adjustments: no team shifted > 0.5 ELO")

    return ratings
