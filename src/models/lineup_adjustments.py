"""
Lineup-aware adjustments for game-level prop predictions.

Phase 1G: Aggregate lineup K/BB/HR-proneness for pitcher props.
Phase 1J: Opposing pitcher quality lift for batter props.
Platoon: Multiplicative adjustment using observed vs-hand splits.

All adjustments operate on the logit scale, consistent with the
existing umpire/weather/matchup lift pattern in game_k_model.py.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.utils.constants import CLIP_LO, CLIP_HI

logger = logging.getLogger(__name__)

# Expected PA per batting order slot in a 9-inning game.
# Slot 1 (leadoff) gets ~4.5 PA, slot 9 gets ~3.5 PA.
# Source: historical averages across MLB seasons.
PA_WEIGHTS_BY_SLOT: np.ndarray = np.array([
    4.50,  # slot 1 (leadoff)
    4.35,  # slot 2
    4.25,  # slot 3
    4.15,  # slot 4 (cleanup)
    4.05,  # slot 5
    3.95,  # slot 6
    3.85,  # slot 7
    3.75,  # slot 8
    3.55,  # slot 9
])


def _safe_logit(p: float | np.ndarray) -> float | np.ndarray:
    """Logit with clipping to avoid infinities."""
    return logit(np.clip(p, CLIP_LO, CLIP_HI))


# ---------------------------------------------------------------------------
# Phase 1G: Lineup proneness for pitcher props
# ---------------------------------------------------------------------------

def compute_lineup_proneness(
    lineup_batter_ids: list[int],
    batter_posteriors: dict[int, np.ndarray],
    league_avg_rate: float,
    stat_name: str = "k",
    weight: float = 0.5,
) -> float:
    """Compute aggregate lineup proneness lift for a pitcher prop.

    For each batter in the lineup with available posteriors, compute their
    posterior mean rate.  Weight by expected PA per batting order slot,
    compute the PA-weighted average rate, and return the logit-scale lift
    relative to the league average, scaled by ``weight``.

    Parameters
    ----------
    lineup_batter_ids : list[int]
        Exactly 9 batter IDs in batting order (slot 0 = leadoff).
    batter_posteriors : dict[int, np.ndarray]
        Mapping of batter_id -> rate posterior samples (values in [0, 1]).
    league_avg_rate : float
        League average rate for this stat (e.g. 0.22 for K%).
    stat_name : str
        Stat name for logging ('k', 'bb', 'hr').
    weight : float
        Scaling factor for the lift. 0.5 = half-weight (conservative).
        0.0 = disabled, 1.0 = full weight.

    Returns
    -------
    float
        Logit-scale lift. Positive = lineup more prone to this outcome
        than average (good for pitcher props).
        Returns 0.0 if weight is 0 or insufficient data.
    """
    if weight == 0.0:
        return 0.0

    if len(lineup_batter_ids) != 9:
        logger.debug(
            "Lineup proneness (%s): expected 9 batters, got %d",
            stat_name, len(lineup_batter_ids),
        )
        return 0.0

    pa_weights = PA_WEIGHTS_BY_SLOT.copy()
    batter_means = np.full(9, np.nan)
    n_found = 0

    for i, bid in enumerate(lineup_batter_ids):
        if bid in batter_posteriors:
            samples = batter_posteriors[bid]
            batter_means[i] = float(np.mean(samples))
            n_found += 1

    if n_found == 0:
        return 0.0

    # For batters without posteriors, use league average (neutral)
    missing = np.isnan(batter_means)
    batter_means[missing] = league_avg_rate

    # PA-weighted average lineup rate
    lineup_avg_rate = float(np.average(batter_means, weights=pa_weights))

    # Logit-scale lift, scaled by weight
    lift = weight * (_safe_logit(lineup_avg_rate) - _safe_logit(league_avg_rate))

    logger.debug(
        "Lineup %s proneness: %d/%d batters matched, "
        "lineup_avg=%.3f, league_avg=%.3f, lift=%.4f (weight=%.2f)",
        stat_name, n_found, 9, lineup_avg_rate, league_avg_rate,
        lift, weight,
    )

    return float(lift)


def compute_lineup_proneness_batch(
    game_lineups: dict[int, list[int]],
    batter_posteriors: dict[int, np.ndarray],
    league_avg_rate: float,
    stat_name: str = "k",
    weight: float = 0.5,
) -> dict[int, float]:
    """Compute lineup proneness lifts for a batch of games.

    Parameters
    ----------
    game_lineups : dict[int, list[int]]
        Mapping of game_pk -> list of 9 batter_ids in batting order.
    batter_posteriors : dict[int, np.ndarray]
        Mapping of batter_id -> rate posterior samples.
    league_avg_rate : float
        League average rate for this stat.
    stat_name : str
        Stat name for logging.
    weight : float
        Scaling factor for the lift.

    Returns
    -------
    dict[int, float]
        {game_pk: logit_lift}. Games without lineups are omitted.
    """
    if weight == 0.0:
        return {}

    result: dict[int, float] = {}
    for game_pk, lineup in game_lineups.items():
        lift = compute_lineup_proneness(
            lineup_batter_ids=lineup,
            batter_posteriors=batter_posteriors,
            league_avg_rate=league_avg_rate,
            stat_name=stat_name,
            weight=weight,
        )
        if lift != 0.0:
            result[game_pk] = lift

    logger.info(
        "Lineup %s proneness batch: %d/%d games with non-zero lift",
        stat_name, len(result), len(game_lineups),
    )
    return result


# ---------------------------------------------------------------------------
# Platoon-adjusted lineup proneness
# ---------------------------------------------------------------------------

# Mapping from stat name to the rate column in platoon splits DataFrame
_PLATOON_RATE_COL: dict[str, str] = {
    "k": "k_rate",
    "bb": "bb_rate",
}


def compute_platoon_adjusted_proneness(
    lineup_batter_ids: list[int],
    batter_posteriors: dict[int, np.ndarray],
    platoon_splits: pd.DataFrame,
    pitcher_hand: str,
    league_avg_rate: float,
    stat_name: str = "k",
    weight: float = 0.5,
    platoon_cap: tuple[float, float] = (0.7, 1.4),
    min_pa_for_platoon: int = 50,
) -> float:
    """Lineup proneness with platoon adjustment for pitcher handedness.

    Uses observed vs-hand splits as a multiplicative ratio on each
    batter's posterior mean, capped to avoid blowups on thin data.

    Parameters
    ----------
    lineup_batter_ids : list[int]
        Exactly 9 batter IDs in batting order.
    batter_posteriors : dict[int, np.ndarray]
        Mapping of batter_id -> rate posterior samples.
    platoon_splits : pd.DataFrame
        Output of ``get_season_totals_by_pitcher_hand()``.  Must contain
        columns: batter_id, pitch_hand, pa, and the relevant rate column
        (e.g. k_rate, bb_rate).
    pitcher_hand : str
        ``'R'`` or ``'L'`` — the starting pitcher's throwing hand.
    league_avg_rate : float
        League average rate for this stat.
    stat_name : str
        ``'k'`` or ``'bb'``.  ``'hr'`` and others fall back to
        non-platoon ``compute_lineup_proneness`` (platoon HR splits
        are too noisy to be useful).
    weight : float
        Scaling factor for the lift (0.0-1.0).
    platoon_cap : tuple[float, float]
        (min_ratio, max_ratio) to cap the platoon multiplier.
    min_pa_for_platoon : int
        Minimum PA vs. hand to apply platoon adjustment.

    Returns
    -------
    float
        Logit-scale lift. Falls back to non-platoon computation if
        platoon data is unavailable.
    """
    if weight == 0.0:
        return 0.0

    if len(lineup_batter_ids) != 9:
        return 0.0

    rate_col = _PLATOON_RATE_COL.get(stat_name)
    if rate_col is None or platoon_splits.empty:
        # No platoon data for this stat — fall back to generic
        return compute_lineup_proneness(
            lineup_batter_ids, batter_posteriors,
            league_avg_rate, stat_name, weight,
        )

    # Pre-index platoon splits for fast lookup
    # Key: (batter_id, pitch_hand) -> (rate, pa)
    plat_idx: dict[tuple[int, str], tuple[float, int]] = {}
    for _, row in platoon_splits.iterrows():
        bid = int(row["batter_id"])
        ph = str(row["pitch_hand"])
        pa = int(row.get("pa", 0))
        rate = row.get(rate_col)
        if pd.notna(rate) and pa > 0:
            plat_idx[(bid, ph)] = (float(rate), pa)

    # Compute per-batter overall rate from both sides
    batter_overall: dict[int, float] = {}
    for bid in set(r[0] for r in plat_idx.keys()):
        rates_pa = []
        for hand in ("R", "L"):
            entry = plat_idx.get((bid, hand))
            if entry:
                rates_pa.append(entry)
        if rates_pa:
            total_pa = sum(e[1] for e in rates_pa)
            if total_pa > 0:
                batter_overall[bid] = sum(
                    e[0] * e[1] for e in rates_pa
                ) / total_pa

    pa_weights = PA_WEIGHTS_BY_SLOT.copy()
    batter_means = np.full(9, np.nan)
    n_found = 0
    n_platoon = 0

    for i, bid in enumerate(lineup_batter_ids):
        if bid not in batter_posteriors:
            continue

        posterior_mean = float(np.mean(batter_posteriors[bid]))
        n_found += 1

        # Try platoon adjustment
        entry = plat_idx.get((bid, pitcher_hand))
        overall = batter_overall.get(bid)
        if (
            entry is not None
            and overall is not None
            and entry[1] >= min_pa_for_platoon
            and overall > 0
        ):
            ratio = entry[0] / overall
            ratio = float(np.clip(ratio, platoon_cap[0], platoon_cap[1]))
            batter_means[i] = posterior_mean * ratio
            n_platoon += 1
        else:
            batter_means[i] = posterior_mean

    if n_found == 0:
        return 0.0

    # Fill missing with league average
    missing = np.isnan(batter_means)
    batter_means[missing] = league_avg_rate

    lineup_avg = float(np.average(batter_means, weights=pa_weights))
    lift = weight * (_safe_logit(lineup_avg) - _safe_logit(league_avg_rate))

    logger.debug(
        "Platoon %s proneness (vs %sHP): %d/%d batters, %d platoon-adjusted, "
        "lineup_avg=%.3f, lift=%.4f",
        stat_name, pitcher_hand, n_found, 9, n_platoon, lineup_avg, lift,
    )

    return float(lift)


# ---------------------------------------------------------------------------
# Phase 1J: Opposing pitcher quality for batter props
# ---------------------------------------------------------------------------

def compute_opposing_pitcher_lift(
    pitcher_rate_posterior_mean: float,
    league_avg_pitcher_rate: float,
    stat_name: str = "k",
    weight: float = 0.5,
) -> float:
    """Compute opposing pitcher quality lift for a batter prop.

    If the opposing pitcher has a higher K% than league average, a batter
    facing them should have a higher probability of striking out, etc.

    Parameters
    ----------
    pitcher_rate_posterior_mean : float
        Opposing pitcher's posterior mean rate for this stat (e.g. K%).
    league_avg_pitcher_rate : float
        League average pitcher rate for this stat.
    stat_name : str
        Stat name for logging.
    weight : float
        Scaling factor for the lift. 0.5 = half-weight (conservative).
        0.0 = disabled, 1.0 = full weight.

    Returns
    -------
    float
        Logit-scale lift. Positive = opposing pitcher is better than
        average at generating this outcome (bad for batter, good for K props).
        Returns 0.0 if weight is 0 or inputs are invalid.
    """
    if weight == 0.0:
        return 0.0

    if (
        np.isnan(pitcher_rate_posterior_mean)
        or np.isnan(league_avg_pitcher_rate)
        or league_avg_pitcher_rate <= 0
    ):
        return 0.0

    lift = weight * (
        _safe_logit(pitcher_rate_posterior_mean)
        - _safe_logit(league_avg_pitcher_rate)
    )

    logger.debug(
        "Opposing pitcher %s lift: pitcher_rate=%.3f, league_avg=%.3f, "
        "lift=%.4f (weight=%.2f)",
        stat_name, pitcher_rate_posterior_mean, league_avg_pitcher_rate,
        lift, weight,
    )

    return float(lift)


def compute_opposing_pitcher_lift_batch(
    game_pitcher_map: dict[tuple[int, int], int],
    pitcher_posteriors: dict[int, np.ndarray],
    league_avg_pitcher_rate: float,
    stat_name: str = "k",
    weight: float = 0.5,
) -> dict[tuple[int, int], float]:
    """Compute opposing pitcher lifts for a batch of batter-games.

    Parameters
    ----------
    game_pitcher_map : dict[tuple[int, int], int]
        Mapping of (game_pk, batter_id) -> opposing pitcher_id.
    pitcher_posteriors : dict[int, np.ndarray]
        Mapping of pitcher_id -> rate posterior samples.
    league_avg_pitcher_rate : float
        League average pitcher rate for this stat.
    stat_name : str
        Stat name for logging.
    weight : float
        Scaling factor for the lift.

    Returns
    -------
    dict[tuple[int, int], float]
        {(game_pk, batter_id): logit_lift}.
    """
    if weight == 0.0:
        return {}

    result: dict[tuple[int, int], float] = {}
    n_found = 0

    for (game_pk, batter_id), pitcher_id in game_pitcher_map.items():
        if pitcher_id not in pitcher_posteriors:
            continue

        pitcher_mean = float(np.mean(pitcher_posteriors[pitcher_id]))
        n_found += 1

        lift = compute_opposing_pitcher_lift(
            pitcher_rate_posterior_mean=pitcher_mean,
            league_avg_pitcher_rate=league_avg_pitcher_rate,
            stat_name=stat_name,
            weight=weight,
        )
        if lift != 0.0:
            result[(game_pk, batter_id)] = lift

    logger.info(
        "Opposing pitcher %s lift batch: %d/%d lookups matched, "
        "%d non-zero lifts",
        stat_name, n_found, len(game_pitcher_map), len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Helpers for extracting lineups per game from cached lineup data
# ---------------------------------------------------------------------------

def extract_opposing_lineup(
    game_pk: int,
    pitcher_team_id: int,
    game_lineups_df: "pd.DataFrame",
) -> list[int] | None:
    """Extract the opposing team's batting lineup for a given game.

    Parameters
    ----------
    game_pk : int
        Game identifier.
    pitcher_team_id : int
        The pitcher's team_id. We want the OTHER team's lineup.
    game_lineups_df : pd.DataFrame
        Lineup data with columns: game_pk, player_id, batting_order, team_id.

    Returns
    -------
    list[int] or None
        9 batter IDs in batting order, or None if not available.
    """
    import pandas as pd  # noqa: local import to avoid circular

    game_lu = game_lineups_df[game_lineups_df["game_pk"] == game_pk]
    if game_lu.empty:
        return None

    # Find the opposing team (not the pitcher's team)
    opposing = game_lu[game_lu["team_id"] != pitcher_team_id]
    if len(opposing) < 9:
        return None

    opposing = opposing.sort_values("batting_order")
    return opposing["player_id"].tolist()[:9]


def build_game_lineup_map(
    game_records: "pd.DataFrame",
    game_lineups_df: "pd.DataFrame",
    pitcher_team_lookup: dict[int, int] | None = None,
) -> dict[int, list[int]]:
    """Build game_pk -> opposing lineup mapping for pitcher prop games.

    For each game in game_records, identifies the pitcher's team and
    extracts the opposing team's batting lineup.

    Parameters
    ----------
    game_records : pd.DataFrame
        Game records with columns: game_pk, pitcher_id, and optionally team_id.
    game_lineups_df : pd.DataFrame
        Lineup data with columns: game_pk, player_id, batting_order, team_id.
    pitcher_team_lookup : dict[int, int] or None
        Optional {pitcher_id: team_id} mapping. If not provided, infers
        the pitcher's team from the game_lineups_df by finding which team
        the pitcher does NOT bat for.

    Returns
    -------
    dict[int, list[int]]
        {game_pk: [9 batter IDs in order]}.
    """
    import pandas as pd  # noqa: local import

    result: dict[int, list[int]] = {}

    # Pre-index lineups
    lu_index: dict[int, pd.DataFrame] = {}
    for gpk, grp in game_lineups_df.groupby("game_pk"):
        lu_index[int(gpk)] = grp

    for _, row in game_records.iterrows():
        game_pk = int(row["game_pk"])
        pitcher_id = int(row["pitcher_id"])

        if game_pk not in lu_index:
            continue

        lu = lu_index[game_pk]
        teams = lu["team_id"].unique()

        if len(teams) < 2:
            continue

        # Determine pitcher's team
        pitcher_team = None
        if pitcher_team_lookup and pitcher_id in pitcher_team_lookup:
            pitcher_team = pitcher_team_lookup[pitcher_id]
        elif "team_id" in row.index and not pd.isna(row.get("team_id")):
            pitcher_team = int(row["team_id"])
        else:
            # Infer: the pitcher is NOT in the batting lineup (NL DH era),
            # so try both teams and pick the one whose batters don't include
            # the pitcher
            for tid in teams:
                team_lu = lu[lu["team_id"] == tid]
                if pitcher_id not in team_lu["player_id"].values:
                    # This is the opposing team
                    opposing = team_lu.sort_values("batting_order")
                    if len(opposing) >= 9:
                        result[game_pk] = opposing["player_id"].tolist()[:9]
                    break
            continue

        # Extract opposing team lineup
        opposing = lu[lu["team_id"] != pitcher_team].sort_values("batting_order")
        if len(opposing) >= 9:
            result[game_pk] = opposing["player_id"].tolist()[:9]

    logger.info(
        "Game lineup map: %d/%d games with opposing lineup",
        len(result), len(game_records),
    )
    return result
