"""
Layer 4: Portfolio / Market Edge Module.

Translates Layer 3 posterior distributions into betting edge estimates
and parlay recommendations for PrizePicks and DraftKings.

Both platforms require 2+ picks combined into parlays. The hidden vig
is embedded in the payout multiplier, not in per-leg odds.

PrizePicks: Fixed payout multipliers (3x for 2-pick, 6x for 3-pick, etc.)
  - Power Play: all picks must hit
  - Flex Play: partial payouts for missing 1-2 legs
  - Break-even per leg: ~55-58% depending on format
  - Best formats: 3-pick Power (55.03%), 5-pick Flex (54.25%)

DraftKings Pick6: Parimutuel with variable payouts, generally worse value.
DraftKings SGP: Traditional sportsbook parlay with per-leg odds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform payout tables
# ---------------------------------------------------------------------------

# PrizePicks Power Play: all picks must be correct
PP_POWER_PAYOUTS: dict[int, float] = {
    2: 3.0,
    3: 6.0,
    4: 10.0,
    5: 20.0,
    6: 25.0,
}

# PrizePicks Flex Play: partial credit for missing legs
# {n_picks: {n_correct: multiplier}}
PP_FLEX_PAYOUTS: dict[int, dict[int, float]] = {
    3: {3: 3.0, 2: 1.0},
    4: {4: 6.0, 3: 1.5},
    5: {5: 10.0, 4: 2.0, 3: 0.4},
    6: {6: 12.5, 5: 2.0, 4: 0.4},
}

# Per-leg break-even probability for each format
# Computed as (1/multiplier)^(1/n) for Power Play
PP_BREAKEVEN: dict[str, dict[int, float]] = {
    "power": {
        2: 0.5774,  # (1/3)^(1/2)
        3: 0.5503,  # (1/6)^(1/3)
        4: 0.5623,  # (1/10)^(1/4)
        5: 0.5493,  # (1/20)^(1/5)
        6: 0.5848,  # (1/25)^(1/6)
    },
    # Flex break-evens are approximate (computed via simulation)
    "flex": {
        3: 0.5774,
        4: 0.5503,
        5: 0.5425,
        6: 0.5898,
    },
}

# DraftKings Pick6 estimated minimum guaranteed multipliers
DK_PICK6_PAYOUTS: dict[int, float] = {
    2: 3.0,
    3: 6.0,
    4: 10.0,
    5: 15.0,
    6: 20.0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropLeg:
    """A single prop leg for a parlay.

    Parameters
    ----------
    player_id : int
        Player MLB ID.
    game_pk : int
        Game identifier.
    stat : str
        Stat type ('k', 'bb', 'hr', 'h', 'outs', 'r', 'rbi', 'tb').
    line : float
        Prop line (e.g. 5.5).
    side : str
        'over' or 'under'.
    model_prob : float
        Model's probability of this leg hitting.
    source : str
        Line source ('draftkings', 'prizepicks').
    player_type : str
        'pitcher' or 'batter'.
    player_name : str
        For display.
    """

    player_id: int
    game_pk: int
    stat: str
    line: float
    side: str
    model_prob: float
    source: str = ""
    player_type: str = "pitcher"
    player_name: str = ""


@dataclass
class ParlayResult:
    """Evaluation of a specific parlay combination.

    Parameters
    ----------
    legs : list[PropLeg]
        The props in this parlay.
    platform : str
        'prizepicks_power', 'prizepicks_flex', 'dk_pick6'.
    n_legs : int
        Number of legs.
    multiplier : float
        Payout multiplier if all legs hit (Power) or max payout (Flex).
    joint_prob : float
        Model's joint probability of all legs hitting (from MC sim).
    ev_per_dollar : float
        Expected value per $1 wagered (>1.0 is profitable).
    kelly_fraction : float
        Kelly criterion bet size as fraction of bankroll.
    edge_pct : float
        (EV - 1) as a percentage.
    breakeven_per_leg : float
        Required per-leg probability to break even on this format.
    avg_leg_prob : float
        Geometric mean of individual leg probabilities.
    leg_surplus : float
        avg_leg_prob - breakeven_per_leg (positive = profitable).
    """

    legs: list[PropLeg]
    platform: str
    n_legs: int
    multiplier: float
    joint_prob: float
    ev_per_dollar: float
    kelly_fraction: float
    edge_pct: float
    breakeven_per_leg: float
    avg_leg_prob: float
    leg_surplus: float


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------


def american_to_decimal(american: str | float) -> float:
    """Convert American odds (e.g. '-110', '+150') to decimal odds.

    Parameters
    ----------
    american : str or float
        American odds. Strings may use unicode minus (\\u2212).

    Returns
    -------
    float
        Decimal odds (e.g. 1.909 for -110, 2.50 for +150).
    """
    s = str(american).replace("\u2212", "-").strip()
    try:
        val = float(s)
    except (ValueError, TypeError):
        return 2.0

    if val < 0:
        return 1.0 + 100.0 / abs(val)
    elif val > 0:
        return 1.0 + val / 100.0
    else:
        return 2.0


def remove_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    """Remove vig from decimal odds to get fair implied probabilities."""
    raw_over = 1.0 / over_odds
    raw_under = 1.0 / under_odds
    total = raw_over + raw_under
    return raw_over / total, raw_under / total


# ---------------------------------------------------------------------------
# Per-leg edge scoring
# ---------------------------------------------------------------------------


def score_leg(
    stat_samples: np.ndarray,
    line: float,
    side: str,
    breakeven: float = 0.55,
) -> dict[str, float]:
    """Score a single prop leg against the model posterior.

    Parameters
    ----------
    stat_samples : np.ndarray
        MC posterior draws for this stat.
    line : float
        Prop line (e.g. 5.5).
    side : str
        'over' or 'under'.
    breakeven : float
        Platform break-even probability per leg.

    Returns
    -------
    dict
        model_prob, edge_vs_breakeven, edge_vs_even.
    """
    p_over = float(np.mean(stat_samples > line))
    model_prob = p_over if side == "over" else 1.0 - p_over

    return {
        "model_prob": model_prob,
        "edge_vs_breakeven": model_prob - breakeven,
        "edge_vs_even": model_prob - 0.50,
    }


# ---------------------------------------------------------------------------
# Joint probability from MC simulation (handles correlations)
# ---------------------------------------------------------------------------


def compute_joint_hit_prob(
    player_posteriors: dict[tuple[int, int], dict[str, np.ndarray]],
    legs: list[PropLeg],
) -> float:
    """Compute joint probability of all legs hitting using MC draws.

    Because pitcher stats within a game are correlated (more K often
    means more outs, etc.), we compute the joint probability by checking
    which simulation draws have ALL legs hitting simultaneously.

    For cross-game or cross-player legs, we assume independence
    (multiply individual probabilities).

    Parameters
    ----------
    player_posteriors : dict
        {(game_pk, player_id): {stat: np.ndarray}}.
    legs : list[PropLeg]
        Parlay legs to evaluate jointly.

    Returns
    -------
    float
        Joint probability of all legs hitting.
    """
    # Group legs by (game_pk, player_id) for correlated computation
    groups: dict[tuple[int, int], list[PropLeg]] = {}
    for leg in legs:
        key = (leg.game_pk, leg.player_id)
        groups.setdefault(key, []).append(leg)

    group_probs = []
    for key, group_legs in groups.items():
        samples = player_posteriors.get(key)
        if samples is None:
            # Fall back to independent product
            for leg in group_legs:
                group_probs.append(leg.model_prob)
            continue

        # Find minimum common length across all stats
        arrays = []
        for leg in group_legs:
            s = samples.get(leg.stat)
            if s is None:
                group_probs.append(leg.model_prob)
                continue
            arrays.append((s, leg))

        if not arrays:
            continue

        min_len = min(len(s) for s, _ in arrays)

        # Build boolean mask: True where ALL legs in this group hit
        all_hit = np.ones(min_len, dtype=bool)
        for s, leg in arrays:
            if leg.side == "over":
                all_hit &= s[:min_len] > leg.line
            else:
                all_hit &= s[:min_len] <= leg.line

        group_probs.append(float(np.mean(all_hit)))

    if not group_probs:
        return 0.0

    # Multiply across independent groups
    joint = 1.0
    for p in group_probs:
        joint *= max(p, 1e-6)
    return joint


# ---------------------------------------------------------------------------
# Parlay EV computation
# ---------------------------------------------------------------------------


def _pp_flex_ev(
    n_legs: int,
    leg_probs: list[float],
    n_mc: int = 100_000,
    rng_seed: int = 42,
) -> float:
    """Compute expected value of a PrizePicks Flex play via MC simulation.

    Flex pays partial credit for missing 1-2 legs, so EV requires
    simulating the distribution of correct picks.
    """
    payouts = PP_FLEX_PAYOUTS.get(n_legs)
    if payouts is None:
        return 0.0

    rng = np.random.default_rng(rng_seed)
    # Simulate each leg independently (conservative; ignores correlation)
    leg_hits = np.column_stack([
        rng.random(n_mc) < p for p in leg_probs
    ])  # (n_mc, n_legs) bool

    n_correct = leg_hits.sum(axis=1)  # (n_mc,)

    total_payout = 0.0
    for correct, mult in payouts.items():
        total_payout += mult * np.mean(n_correct == correct)

    return total_payout  # EV per $1 wagered


def evaluate_parlay(
    legs: list[PropLeg],
    player_posteriors: dict[tuple[int, int], dict[str, np.ndarray]],
    platform: str = "prizepicks_power",
) -> ParlayResult | None:
    """Evaluate a specific parlay combination on a given platform.

    Parameters
    ----------
    legs : list[PropLeg]
        The 2-6 legs to combine.
    player_posteriors : dict
        Posterior draws for joint probability computation.
    platform : str
        One of 'prizepicks_power', 'prizepicks_flex', 'dk_pick6'.

    Returns
    -------
    ParlayResult or None
        None if the format is invalid for this leg count.
    """
    n = len(legs)
    if n < 2:
        return None

    # Get payout multiplier and break-even
    if platform == "prizepicks_power":
        multiplier = PP_POWER_PAYOUTS.get(n)
        breakeven = PP_BREAKEVEN["power"].get(n, 0.55)
    elif platform == "prizepicks_flex":
        if n < 3:
            return None  # Flex requires 3+
        multiplier = PP_FLEX_PAYOUTS.get(n, {}).get(n, 0.0)  # max payout
        breakeven = PP_BREAKEVEN["flex"].get(n, 0.55)
    elif platform == "dk_pick6":
        multiplier = DK_PICK6_PAYOUTS.get(n)
        breakeven = (1.0 / multiplier) ** (1.0 / n) if multiplier else 0.60
    else:
        return None

    if multiplier is None or multiplier <= 0:
        return None

    # Joint probability (handles within-player correlations)
    joint_prob = compute_joint_hit_prob(player_posteriors, legs)

    # EV computation
    if platform == "prizepicks_flex" and n >= 3:
        leg_probs = [leg.model_prob for leg in legs]
        ev = _pp_flex_ev(n, leg_probs)
    else:
        # Power play / Pick6: all-or-nothing
        ev = joint_prob * multiplier

    # Kelly criterion for the parlay as a single bet
    # f = (b*p - q) / b where b = multiplier - 1, p = joint_prob
    b = multiplier - 1.0
    if b > 0 and joint_prob > 0:
        kelly = (b * joint_prob - (1.0 - joint_prob)) / b
    else:
        kelly = 0.0
    kelly = max(kelly, 0.0)

    # Geometric mean of individual leg probs
    geo_mean = np.exp(np.mean(np.log([max(l.model_prob, 1e-6) for l in legs])))

    return ParlayResult(
        legs=legs,
        platform=platform,
        n_legs=n,
        multiplier=multiplier,
        joint_prob=joint_prob,
        ev_per_dollar=ev,
        kelly_fraction=kelly,
        edge_pct=(ev - 1.0) * 100,
        breakeven_per_leg=breakeven,
        avg_leg_prob=geo_mean,
        leg_surplus=geo_mean - breakeven,
    )


# ---------------------------------------------------------------------------
# Build all prop legs from available lines
# ---------------------------------------------------------------------------


def build_prop_legs(
    all_props: pd.DataFrame,
    player_posteriors: dict[tuple[int, int], dict[str, np.ndarray]],
    stat_map: dict[str, str],
    min_edge_vs_even: float = 0.03,
) -> list[PropLeg]:
    """Build scored PropLeg objects from the merged props DataFrame.

    Only includes legs where the model has at least min_edge_vs_even
    over a coin flip on the chosen side.

    Parameters
    ----------
    all_props : pd.DataFrame
        Merged game props with vegas_line column.
    player_posteriors : dict
        Posterior draws per (game_pk, player_id).
    stat_map : dict
        Maps stat labels (e.g. 'K') to posterior keys (e.g. 'k').
    min_edge_vs_even : float
        Minimum model_prob - 0.50 to include a leg.

    Returns
    -------
    list[PropLeg]
        All viable legs, sorted by model_prob descending.
    """
    has_line = all_props["vegas_line"].notna()
    eligible = all_props[has_line]

    legs = []
    for _, row in eligible.iterrows():
        stat_lower = stat_map.get(row["stat"])
        if stat_lower is None:
            continue

        key = (int(row["game_pk"]), int(row["player_id"]))
        samples_dict = player_posteriors.get(key)
        if samples_dict is None:
            continue
        samples = samples_dict.get(stat_lower)
        if samples is None:
            continue

        line = float(row["vegas_line"])
        p_over = float(np.mean(samples > line))
        p_under = 1.0 - p_over

        # Pick the stronger side
        if p_over >= p_under:
            side, model_prob = "over", p_over
        else:
            side, model_prob = "under", p_under

        # Filter: must have meaningful edge over a coin flip
        if model_prob - 0.50 < min_edge_vs_even:
            continue

        legs.append(PropLeg(
            player_id=int(row["player_id"]),
            game_pk=int(row["game_pk"]),
            stat=stat_lower,
            line=line,
            side=side,
            model_prob=model_prob,
            source=str(row.get("line_source", "")),
            player_type=str(row.get("player_type", "")),
            player_name=str(row.get("player_name", "")),
        ))

    legs.sort(key=lambda l: l.model_prob, reverse=True)
    return legs


# ---------------------------------------------------------------------------
# Find best parlay combinations
# ---------------------------------------------------------------------------


def find_best_parlays(
    legs: list[PropLeg],
    player_posteriors: dict[tuple[int, int], dict[str, np.ndarray]],
    platforms: list[str] | None = None,
    n_picks_range: tuple[int, int] = (2, 5),
    max_combos_per_size: int = 10,
    min_ev: float = 1.0,
    kelly_fraction: float = 0.25,
    max_stake: float = 0.03,
) -> list[ParlayResult]:
    """Find the highest-EV parlay combinations across formats.

    Evaluates combinations of the top-ranked legs across multiple
    platform formats and returns the best ones.

    Parameters
    ----------
    legs : list[PropLeg]
        Available legs sorted by model_prob descending.
    player_posteriors : dict
        For joint probability computation.
    platforms : list[str]
        Platforms to evaluate. Defaults to PP power + flex.
    n_picks_range : tuple
        (min_picks, max_picks) to consider.
    max_combos_per_size : int
        Max combinations to return per (platform, n_picks).
    min_ev : float
        Minimum EV per dollar to include (1.0 = break-even).
    kelly_fraction : float
        Fractional Kelly multiplier for stake sizing.
    max_stake : float
        Maximum stake per parlay as fraction of bankroll.

    Returns
    -------
    list[ParlayResult]
        Best parlays sorted by edge_pct descending.
    """
    if platforms is None:
        platforms = ["prizepicks_power", "prizepicks_flex"]

    # Limit search space: top N legs by model_prob
    top_legs = legs[:20]
    if len(top_legs) < 2:
        return []

    results = []
    min_n, max_n = n_picks_range

    for platform in platforms:
        for n in range(min_n, min(max_n + 1, len(top_legs) + 1)):
            # Skip invalid format/size combos
            if platform == "prizepicks_flex" and n < 3:
                continue

            best_for_size = []
            for combo in combinations(range(len(top_legs)), n):
                combo_legs = [top_legs[i] for i in combo]
                result = evaluate_parlay(combo_legs, player_posteriors, platform)
                if result is None:
                    continue
                if result.ev_per_dollar < min_ev:
                    continue

                # Apply fractional Kelly and cap
                result.kelly_fraction = min(
                    result.kelly_fraction * kelly_fraction, max_stake,
                )
                best_for_size.append(result)

            # Keep top N per size
            best_for_size.sort(key=lambda r: r.ev_per_dollar, reverse=True)
            results.extend(best_for_size[:max_combos_per_size])

    results.sort(key=lambda r: r.ev_per_dollar, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def parlays_to_dataframe(parlays: list[ParlayResult]) -> pd.DataFrame:
    """Convert parlay results to a dashboard-ready DataFrame.

    Returns one row per parlay with summary fields + individual leg details.
    """
    if not parlays:
        return pd.DataFrame()

    records = []
    for i, p in enumerate(parlays):
        leg_descs = []
        for leg in p.legs:
            name = leg.player_name or str(leg.player_id)
            leg_descs.append(
                f"{name} {leg.stat.upper()} {leg.side} {leg.line}"
            )

        records.append({
            "parlay_id": i,
            "platform": p.platform,
            "n_legs": p.n_legs,
            "multiplier": p.multiplier,
            "joint_prob": round(p.joint_prob, 4),
            "ev_per_dollar": round(p.ev_per_dollar, 3),
            "edge_pct": round(p.edge_pct, 1),
            "kelly_fraction": round(p.kelly_fraction, 4),
            "breakeven_per_leg": round(p.breakeven_per_leg, 4),
            "avg_leg_prob": round(p.avg_leg_prob, 4),
            "leg_surplus": round(p.leg_surplus, 4),
            "legs": " | ".join(leg_descs),
            # Individual leg details for drill-down
            "leg_players": [l.player_name for l in p.legs],
            "leg_stats": [l.stat for l in p.legs],
            "leg_lines": [l.line for l in p.legs],
            "leg_sides": [l.side for l in p.legs],
            "leg_probs": [round(l.model_prob, 4) for l in p.legs],
            "leg_sources": [l.source for l in p.legs],
        })

    return pd.DataFrame(records)


def legs_to_dataframe(legs: list[PropLeg]) -> pd.DataFrame:
    """Convert individual legs to a summary DataFrame (pre-parlay scoring)."""
    if not legs:
        return pd.DataFrame()

    records = []
    for leg in legs:
        records.append({
            "player_id": leg.player_id,
            "player_name": leg.player_name,
            "game_pk": leg.game_pk,
            "stat": leg.stat,
            "line": leg.line,
            "side": leg.side,
            "model_prob": round(leg.model_prob, 4),
            "edge_vs_even": round(leg.model_prob - 0.50, 4),
            "source": leg.source,
            "player_type": leg.player_type,
        })

    return pd.DataFrame(records)
