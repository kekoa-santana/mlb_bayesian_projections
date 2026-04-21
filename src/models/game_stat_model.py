"""BF-draw game-level stat prediction engine.

Binomial/Poisson Monte Carlo simulations for pitcher and batter game
stat totals. Used by game_k_validation and game_prop_validation for
walk-forward backtesting.

Note: The PA-by-PA game simulator (game_sim/simulator.py) is the
production engine for daily predictions. This module provides a simpler
BF-draw approach used for validation and comparison purposes.

Relocated from game_k_model.py (deleted) during legacy model cleanup.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.models.bf_model import draw_bf_samples, get_bf_distribution
from src.models.matchup import score_matchup, score_matchup_for_stat
from src.models.rest_adjustment import (
    apply_rest_to_bf,
    get_rest_adjustment,
)
from src.utils.constants import CLIP_LO, CLIP_HI  # noqa: F401 (used by _safe_logit alias)
from src.utils.math_helpers import safe_logit as _safe_logit
from src.models.game_sim.tto_model import (
    LEAGUE_TTO_LOGIT_LIFTS as _LEAGUE_TTO_LOGIT_LIFTS,
    BF_PER_TTO as _BF_PER_TTO,
    build_tto_logit_lifts,
)

logger = logging.getLogger(__name__)

# Mapping from stat_name → matchup lift key in score_matchup_for_stat results
_STAT_LIFT_KEYS: dict[str, str] = {
    "k": "matchup_k_logit_lift",
    "bb": "matchup_bb_logit_lift",
    "hr": "matchup_hr_logit_lift",
}



# ---------------------------------------------------------------------------
# K-only simulation (legacy, used by game_k_validation)
# ---------------------------------------------------------------------------


def simulate_game_ks(
    pitcher_k_rate_samples: np.ndarray,
    bf_mu: float,
    bf_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    umpire_k_logit_lift: float = 0.0,
    weather_k_logit_lift: float = 0.0,
    tto_logit_lifts: np.ndarray | None = None,
    rest_k_logit_lift: float = 0.0,
    n_draws: int = 4000,
    bf_min: int = 3,
    bf_max: int = 35,
    random_seed: int = 42,
) -> np.ndarray:
    """Monte Carlo simulation of game strikeout totals."""
    rng = np.random.default_rng(random_seed)

    if len(pitcher_k_rate_samples) != n_draws:
        idx = rng.choice(len(pitcher_k_rate_samples), size=n_draws, replace=True)
        k_rate_draws = pitcher_k_rate_samples[idx]
    else:
        k_rate_draws = pitcher_k_rate_samples.copy()

    bf_draws = draw_bf_samples(
        mu_bf=bf_mu, sigma_bf=bf_sigma,
        n_draws=n_draws, bf_min=bf_min, bf_max=bf_max, rng=rng,
    )

    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(9)

    k_logit = (
        _safe_logit(k_rate_draws)
        + umpire_k_logit_lift
        + weather_k_logit_lift
        + rest_k_logit_lift
    )

    k_totals = np.zeros(n_draws, dtype=int)

    unique_bf = np.unique(bf_draws)
    for bf_val in unique_bf:
        mask = bf_draws == bf_val
        n_bf_draws = mask.sum()
        bf_int = int(bf_val)

        game_ks = np.zeros(n_bf_draws, dtype=int)
        k_logit_subset = k_logit[mask]

        if tto_logit_lifts is not None:
            tto_slot_counts: dict[tuple[int, int], int] = {}
            for bf_idx in range(bf_int):
                tto_block = min(bf_idx // _BF_PER_TTO, 2)
                slot = bf_idx % 9
                key = (tto_block, slot)
                tto_slot_counts[key] = tto_slot_counts.get(key, 0) + 1

            for (tto_block, slot), count in tto_slot_counts.items():
                adjusted_logit = (
                    k_logit_subset
                    + lineup_matchup_lifts[slot]
                    + tto_logit_lifts[tto_block]
                )
                adjusted_p = expit(adjusted_logit)
                game_ks += rng.binomial(n=count, p=adjusted_p)
        else:
            base_pa = bf_int // 9
            extra = bf_int % 9
            pa_per_slot = np.full(9, base_pa, dtype=int)
            pa_per_slot[:extra] += 1

            for slot in range(9):
                if pa_per_slot[slot] == 0:
                    continue
                adjusted_logit = k_logit_subset + lineup_matchup_lifts[slot]
                adjusted_p = expit(adjusted_logit)
                slot_ks = rng.binomial(n=pa_per_slot[slot], p=adjusted_p)
                game_ks += slot_ks

        k_totals[mask] = game_ks

    return k_totals


def _compute_lineup_matchup_lifts(
    pitcher_id: int,
    lineup_batter_ids: list[int],
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Score K matchups for a 9-batter lineup."""
    lifts = np.zeros(9)
    matchup_details = []

    for i, batter_id in enumerate(lineup_batter_ids):
        result = score_matchup(
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
        )
        lift = result.get("matchup_k_logit_lift", 0.0)
        if np.isnan(lift):
            lift = 0.0
        lifts[i] = lift
        matchup_details.append(result)

    return lifts, matchup_details


# ---------------------------------------------------------------------------
# Generalized simulation functions (all prop types: K, BB, HR, H, Outs)
# ---------------------------------------------------------------------------


def simulate_game_stat(
    rate_samples: np.ndarray,
    opp_mu: float,
    opp_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    context_logit_lift: float = 0.0,
    tto_logit_lifts: np.ndarray | None = None,
    n_draws: int = 4000,
    opp_min: int = 3,
    opp_max: int = 35,
    n_slots: int = 9,
    random_seed: int = 42,
) -> np.ndarray:
    """Stat-agnostic Binomial Monte Carlo simulation."""
    rng = np.random.default_rng(random_seed)

    if len(rate_samples) != n_draws:
        idx = rng.choice(len(rate_samples), size=n_draws, replace=True)
        rate_draws = rate_samples[idx]
    else:
        rate_draws = rate_samples.copy()

    opp_draws = draw_bf_samples(
        mu_bf=opp_mu, sigma_bf=opp_sigma,
        n_draws=n_draws, bf_min=opp_min, bf_max=opp_max, rng=rng,
    )

    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(n_slots)

    rate_logit = _safe_logit(rate_draws) + context_logit_lift
    stat_totals = np.zeros(n_draws, dtype=int)

    if n_slots == 1:
        adjusted_logit = rate_logit + lineup_matchup_lifts[0]
        adjusted_p = expit(adjusted_logit)
        stat_totals = rng.binomial(n=opp_draws.astype(int), p=adjusted_p)
    else:
        unique_opp = np.unique(opp_draws)
        for opp_val in unique_opp:
            mask = opp_draws == opp_val
            n_opp_draws = mask.sum()
            opp_int = int(opp_val)

            game_stats = np.zeros(n_opp_draws, dtype=int)
            rate_logit_subset = rate_logit[mask]

            if tto_logit_lifts is not None:
                tto_slot_counts: dict[tuple[int, int], int] = {}
                for bf_idx in range(opp_int):
                    tto_block = min(bf_idx // _BF_PER_TTO, 2)
                    slot = bf_idx % n_slots
                    key = (tto_block, slot)
                    tto_slot_counts[key] = tto_slot_counts.get(key, 0) + 1

                for (tto_block, slot), count in tto_slot_counts.items():
                    adjusted_logit = (
                        rate_logit_subset
                        + lineup_matchup_lifts[slot]
                        + tto_logit_lifts[tto_block]
                    )
                    adjusted_p = expit(adjusted_logit)
                    game_stats += rng.binomial(n=count, p=adjusted_p)
            else:
                base_pa = opp_int // n_slots
                extra = opp_int % n_slots
                pa_per_slot = np.full(n_slots, base_pa, dtype=int)
                pa_per_slot[:extra] += 1

                for slot in range(n_slots):
                    if pa_per_slot[slot] == 0:
                        continue
                    adjusted_logit = rate_logit_subset + lineup_matchup_lifts[slot]
                    adjusted_p = expit(adjusted_logit)
                    slot_stats = rng.binomial(n=pa_per_slot[slot], p=adjusted_p)
                    game_stats += slot_stats

            stat_totals[mask] = game_stats

    return stat_totals


def simulate_game_stat_poisson(
    rate_samples: np.ndarray,
    opp_mu: float,
    opp_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    context_logit_lift: float = 0.0,
    park_factor: float = 1.0,
    tto_logit_lifts: np.ndarray | None = None,
    n_draws: int = 4000,
    opp_min: int = 3,
    opp_max: int = 35,
    n_slots: int = 9,
    random_seed: int = 42,
) -> np.ndarray:
    """Poisson Monte Carlo simulation for rare events (HR)."""
    rng = np.random.default_rng(random_seed)

    if len(rate_samples) != n_draws:
        idx = rng.choice(len(rate_samples), size=n_draws, replace=True)
        rate_draws = rate_samples[idx]
    else:
        rate_draws = rate_samples.copy()

    opp_draws = draw_bf_samples(
        mu_bf=opp_mu, sigma_bf=opp_sigma,
        n_draws=n_draws, bf_min=opp_min, bf_max=opp_max, rng=rng,
    )

    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(n_slots)

    rate_logit = _safe_logit(rate_draws) + context_logit_lift
    stat_totals = np.zeros(n_draws, dtype=int)

    if n_slots == 1:
        adjusted_logit = rate_logit + lineup_matchup_lifts[0]
        adjusted_rate = expit(adjusted_logit)
        lam = adjusted_rate * opp_draws * park_factor
        stat_totals = rng.poisson(lam=lam)
    else:
        unique_opp = np.unique(opp_draws)
        for opp_val in unique_opp:
            mask = opp_draws == opp_val
            n_opp_draws = mask.sum()
            opp_int = int(opp_val)

            game_stats = np.zeros(n_opp_draws, dtype=int)
            rate_logit_subset = rate_logit[mask]

            if tto_logit_lifts is not None:
                tto_slot_counts: dict[tuple[int, int], int] = {}
                for bf_idx in range(opp_int):
                    tto_block = min(bf_idx // _BF_PER_TTO, 2)
                    slot = bf_idx % n_slots
                    key = (tto_block, slot)
                    tto_slot_counts[key] = tto_slot_counts.get(key, 0) + 1

                for (tto_block, slot), count in tto_slot_counts.items():
                    adjusted_logit = (
                        rate_logit_subset
                        + lineup_matchup_lifts[slot]
                        + tto_logit_lifts[tto_block]
                    )
                    adjusted_rate = expit(adjusted_logit)
                    lam = adjusted_rate * count * park_factor
                    game_stats += rng.poisson(lam=lam)
            else:
                base_pa = opp_int // n_slots
                extra = opp_int % n_slots
                pa_per_slot = np.full(n_slots, base_pa, dtype=int)
                pa_per_slot[:extra] += 1

                for slot in range(n_slots):
                    if pa_per_slot[slot] == 0:
                        continue
                    adjusted_logit = rate_logit_subset + lineup_matchup_lifts[slot]
                    adjusted_rate = expit(adjusted_logit)
                    lam = adjusted_rate * pa_per_slot[slot] * park_factor
                    slot_stats = rng.poisson(lam=lam)
                    game_stats += slot_stats

            stat_totals[mask] = game_stats

    return stat_totals


def _compute_lineup_matchup_lifts_for_stat(
    stat_name: str,
    pitcher_id: int,
    lineup_batter_ids: list[int],
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Score matchups for a 9-batter lineup for any stat type."""
    lift_key = _STAT_LIFT_KEYS.get(stat_name.lower(), "matchup_logit_lift")

    lifts = np.zeros(len(lineup_batter_ids))
    matchup_details: list[dict[str, Any]] = []

    for i, batter_id in enumerate(lineup_batter_ids):
        result = score_matchup_for_stat(
            stat_name=stat_name,
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
        )
        lift = result.get(lift_key, 0.0)
        if np.isnan(lift):
            lift = 0.0
        lifts[i] = lift
        matchup_details.append(result)

    return lifts, matchup_details


def simulate_batter_game_stat(
    rate_samples: np.ndarray,
    pa_mu: float,
    pa_sigma: float,
    matchup_logit_lift: float = 0.0,
    context_logit_lift: float = 0.0,
    park_factor: float = 1.0,
    model_type: str = "binomial",
    n_draws: int = 4000,
    pa_min: int = 1,
    pa_max: int = 7,
    random_seed: int = 42,
) -> np.ndarray:
    """Monte Carlo simulation for a single batter's game stat total."""
    matchup_array = np.array([matchup_logit_lift])

    if model_type == "poisson":
        return simulate_game_stat_poisson(
            rate_samples=rate_samples,
            opp_mu=pa_mu,
            opp_sigma=pa_sigma,
            lineup_matchup_lifts=matchup_array,
            context_logit_lift=context_logit_lift,
            park_factor=park_factor,
            n_draws=n_draws,
            opp_min=pa_min,
            opp_max=pa_max,
            n_slots=1,
            random_seed=random_seed,
        )
    else:
        return simulate_game_stat(
            rate_samples=rate_samples,
            opp_mu=pa_mu,
            opp_sigma=pa_sigma,
            lineup_matchup_lifts=matchup_array,
            context_logit_lift=context_logit_lift,
            n_draws=n_draws,
            opp_min=pa_min,
            opp_max=pa_max,
            n_slots=1,
            random_seed=random_seed,
        )


def predict_game_stat(
    stat_name: str,
    pitcher_id: int,
    season: int,
    lineup_batter_ids: list[int] | None,
    rate_samples: np.ndarray,
    bf_priors: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame | None = None,
    hitter_vuln: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
    context_logit_lift: float = 0.0,
    lineup_proneness_lift: float = 0.0,
    park_factor: float = 1.0,
    model_type: str = "binomial",
    days_rest: int | None = None,
    tto_logit_lifts: np.ndarray | None = None,
    n_draws: int = 4000,
    random_seed: int = 42,
    bf_min: int = 3,
    bf_max: int = 35,
) -> dict[str, Any]:
    """Full game stat prediction for a pitcher, combining all layers."""
    from src.models.posterior_utils import compute_over_probs

    bf_info = get_bf_distribution(pitcher_id, season, bf_priors)
    bf_mu = bf_info["mu_bf"]
    bf_sigma = bf_info["sigma_bf"]

    _REST_STATS = {"k", "bb", "hr"}
    rest_adj = get_rest_adjustment(days_rest)
    sn_lower = stat_name.lower()
    rest_lift = rest_adj.get(f"{sn_lower}_lift", 0.0) if sn_lower in _REST_STATS else 0.0
    if sn_lower in _REST_STATS:
        bf_mu, bf_sigma = apply_rest_to_bf(bf_mu, bf_sigma, days_rest)

    lineup_lifts = None
    per_batter_details: list[dict[str, Any]] = []
    if lineup_batter_ids is not None and len(lineup_batter_ids) == 9:
        if pitcher_arsenal is not None and hitter_vuln is not None and baselines_pt is not None:
            lineup_lifts, per_batter_details = _compute_lineup_matchup_lifts_for_stat(
                stat_name, pitcher_id, lineup_batter_ids,
                pitcher_arsenal, hitter_vuln, baselines_pt,
            )

    total_context = context_logit_lift + lineup_proneness_lift + rest_lift

    if model_type == "poisson":
        stat_samples = simulate_game_stat_poisson(
            rate_samples=rate_samples,
            opp_mu=bf_mu,
            opp_sigma=bf_sigma,
            lineup_matchup_lifts=lineup_lifts,
            context_logit_lift=total_context,
            park_factor=park_factor,
            tto_logit_lifts=tto_logit_lifts,
            n_draws=n_draws,
            opp_min=bf_min,
            opp_max=bf_max,
            n_slots=9,
            random_seed=random_seed,
        )
    else:
        stat_samples = simulate_game_stat(
            rate_samples=rate_samples,
            opp_mu=bf_mu,
            opp_sigma=bf_sigma,
            lineup_matchup_lifts=lineup_lifts,
            context_logit_lift=total_context,
            tto_logit_lifts=tto_logit_lifts,
            n_draws=n_draws,
            opp_min=bf_min,
            opp_max=bf_max,
            n_slots=9,
            random_seed=random_seed,
        )

    over_probs = compute_over_probs(stat_samples, stat_name=stat_name)

    sn = stat_name.lower()
    return {
        "stat_samples": stat_samples,
        "over_probs": over_probs,
        f"expected_{sn}": float(np.mean(stat_samples)),
        f"std_{sn}": float(np.std(stat_samples)),
        "bf_mu": bf_mu,
        "bf_sigma": bf_sigma,
        "lineup_matchup_lifts": lineup_lifts,
        "per_batter_details": per_batter_details,
        "context_logit_lift": context_logit_lift,
        "lineup_proneness_lift": lineup_proneness_lift,
        "park_factor": park_factor,
        "days_rest": days_rest,
        "rest_bucket": rest_adj["rest_bucket"],
    }


def predict_game(
    pitcher_id: int,
    season: int,
    lineup_batter_ids: list[int] | None,
    pitcher_k_rate_samples: np.ndarray,
    bf_priors: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame | None = None,
    hitter_vuln: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
    umpire_k_logit_lift: float = 0.0,
    weather_k_logit_lift: float = 0.0,
    days_rest: int | None = None,
    n_draws: int = 4000,
    random_seed: int = 42,
    bf_min: int = 3,
    bf_max: int = 35,
) -> dict[str, Any]:
    """Full game K prediction combining all layers."""
    from src.models.posterior_utils import compute_k_over_probs

    bf_info = get_bf_distribution(pitcher_id, season, bf_priors)
    bf_mu = bf_info["mu_bf"]
    bf_sigma = bf_info["sigma_bf"]

    rest_adj = get_rest_adjustment(days_rest)
    rest_k_lift = rest_adj["k_lift"]
    bf_mu, bf_sigma = apply_rest_to_bf(bf_mu, bf_sigma, days_rest)

    lineup_lifts = None
    per_batter_details = []
    if lineup_batter_ids is not None and len(lineup_batter_ids) == 9:
        if pitcher_arsenal is not None and hitter_vuln is not None and baselines_pt is not None:
            lineup_lifts, per_batter_details = _compute_lineup_matchup_lifts(
                pitcher_id, lineup_batter_ids,
                pitcher_arsenal, hitter_vuln, baselines_pt,
            )

    k_samples = simulate_game_ks(
        pitcher_k_rate_samples=pitcher_k_rate_samples,
        bf_mu=bf_mu,
        bf_sigma=bf_sigma,
        lineup_matchup_lifts=lineup_lifts,
        umpire_k_logit_lift=umpire_k_logit_lift,
        weather_k_logit_lift=weather_k_logit_lift,
        rest_k_logit_lift=rest_k_lift,
        n_draws=n_draws,
        bf_min=bf_min,
        bf_max=bf_max,
        random_seed=random_seed,
    )

    k_over_probs = compute_k_over_probs(k_samples)

    return {
        "k_samples": k_samples,
        "k_over_probs": k_over_probs,
        "expected_k": float(np.mean(k_samples)),
        "std_k": float(np.std(k_samples)),
        "bf_mu": bf_mu,
        "bf_sigma": bf_sigma,
        "lineup_matchup_lifts": lineup_lifts,
        "per_batter_details": per_batter_details,
        "umpire_k_logit_lift": umpire_k_logit_lift,
        "weather_k_logit_lift": weather_k_logit_lift,
        "rest_k_logit_lift": rest_k_lift,
        "days_rest": days_rest,
        "rest_bucket": rest_adj["rest_bucket"],
    }


def predict_game_batch(
    game_records: pd.DataFrame,
    pitcher_posteriors: dict[int, np.ndarray],
    bf_priors: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame | None = None,
    hitter_vuln: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
    game_batter_ks: pd.DataFrame | None = None,
    game_lineups: pd.DataFrame | None = None,
    umpire_lifts: dict[int, float] | None = None,
    weather_lifts: dict[int, float] | None = None,
    rest_df: pd.DataFrame | None = None,
    n_draws: int = 4000,
    bf_min: int = 3,
    bf_max: int = 35,
) -> pd.DataFrame:
    """Batch K predictions for backtesting across many games."""
    records = []
    n_games = len(game_records)

    _lineup_index: dict[int, pd.DataFrame] = {}
    if game_lineups is not None and not game_lineups.empty:
        for gpk, grp in game_lineups.groupby("game_pk"):
            _lineup_index[int(gpk)] = grp

    _rest_index: dict[tuple[int, int], int] = {}
    if rest_df is not None and not rest_df.empty:
        for _, rrow in rest_df.iterrows():
            _rest_index[(int(rrow["pitcher_id"]), int(rrow["game_pk"]))] = int(
                rrow["days_rest"]
            )

    for i, (_, game) in enumerate(game_records.iterrows()):
        pitcher_id = int(game["pitcher_id"])
        game_pk = int(game["game_pk"])
        season = int(game["season"])
        actual_k = int(game["strike_outs"])
        actual_bf = int(game["batters_faced"])

        if pitcher_id not in pitcher_posteriors:
            continue

        k_rate_samples = pitcher_posteriors[pitcher_id]

        lineup_ids = None
        n_matched = 0

        if game_pk in _lineup_index:
            lu = _lineup_index[game_pk]
            if game_batter_ks is not None:
                faced = set(
                    game_batter_ks[
                        (game_batter_ks["game_pk"] == game_pk)
                        & (game_batter_ks["pitcher_id"] == pitcher_id)
                    ]["batter_id"].tolist()
                )
                for tid in lu["team_id"].unique():
                    team_lu = lu[lu["team_id"] == tid].sort_values("batting_order")
                    if set(team_lu["player_id"].tolist()) & faced:
                        batter_ids = team_lu["player_id"].tolist()
                        n_matched = len(batter_ids)
                        if n_matched >= 9:
                            lineup_ids = batter_ids[:9]
                        break

        if lineup_ids is None and game_batter_ks is not None:
            game_batters = game_batter_ks[
                (game_batter_ks["game_pk"] == game_pk)
                & (game_batter_ks["pitcher_id"] == pitcher_id)
            ]
            batter_ids = game_batters["batter_id"].tolist()
            n_matched = len(batter_ids)
            if n_matched >= 9:
                lineup_ids = batter_ids[:9]
            elif n_matched > 0:
                lineup_ids = (batter_ids * ((9 // len(batter_ids)) + 1))[:9]

        ump_lift = 0.0
        if umpire_lifts is not None:
            ump_lift = umpire_lifts.get(game_pk, 0.0)

        wx_lift = 0.0
        if weather_lifts is not None:
            wx_lift = weather_lifts.get(game_pk, 0.0)

        dr = _rest_index.get((pitcher_id, game_pk))

        result = predict_game(
            pitcher_id=pitcher_id,
            season=season,
            lineup_batter_ids=lineup_ids,
            pitcher_k_rate_samples=k_rate_samples,
            bf_priors=bf_priors,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
            umpire_k_logit_lift=ump_lift,
            weather_k_logit_lift=wx_lift,
            days_rest=dr,
            n_draws=n_draws,
            random_seed=42 + i,
            bf_min=bf_min,
            bf_max=bf_max,
        )

        rec = {
            "game_pk": game_pk,
            "pitcher_id": pitcher_id,
            "season": season,
            "actual_k": actual_k,
            "actual_bf": actual_bf,
            "expected_k": result["expected_k"],
            "std_k": result["std_k"],
            "pitcher_k_rate_mean": float(np.mean(k_rate_samples)),
            "bf_mu": result["bf_mu"],
            "n_matched_batters": n_matched,
            "umpire_k_logit_lift": ump_lift,
            "weather_k_logit_lift": wx_lift,
            "days_rest": dr,
            "rest_bucket": result["rest_bucket"],
        }

        k_over = result["k_over_probs"]
        for _, row in k_over.iterrows():
            col_name = f"p_over_{row['line']:.1f}".replace(".", "_")
            rec[col_name] = row["p_over"]

        records.append(rec)

        if (i + 1) % 200 == 0:
            logger.info("Predicted %d / %d games", i + 1, n_games)

    logger.info("Batch prediction complete: %d games", len(records))
    return pd.DataFrame(records)


def predict_batter_game(
    stat_name: str,
    batter_id: int,
    pitcher_id: int,
    rate_samples: np.ndarray,
    pa_mu: float,
    pa_sigma: float,
    pitcher_arsenal: pd.DataFrame | None = None,
    hitter_vuln: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
    context_logit_lift: float = 0.0,
    opposing_pitcher_lift: float = 0.0,
    park_factor: float = 1.0,
    model_type: str = "binomial",
    default_lines: list[float] | None = None,
    n_draws: int = 4000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Full game stat prediction for a single batter."""
    from src.models.posterior_utils import compute_over_probs

    sn = stat_name.lower()
    lift_key = _STAT_LIFT_KEYS.get(sn, "matchup_logit_lift")

    matchup_lift = 0.0
    matchup_detail: dict[str, Any] = {}
    if pitcher_arsenal is not None and hitter_vuln is not None and baselines_pt is not None:
        matchup_detail = score_matchup_for_stat(
            stat_name=stat_name,
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
        )
        matchup_lift = matchup_detail.get(lift_key, 0.0)
        if np.isnan(matchup_lift):
            matchup_lift = 0.0

    total_context = context_logit_lift + opposing_pitcher_lift

    stat_samples = simulate_batter_game_stat(
        rate_samples=rate_samples,
        pa_mu=pa_mu,
        pa_sigma=pa_sigma,
        matchup_logit_lift=matchup_lift,
        context_logit_lift=total_context,
        park_factor=park_factor,
        model_type=model_type,
        n_draws=n_draws,
        random_seed=random_seed,
    )

    over_probs = compute_over_probs(stat_samples, lines=default_lines, stat_name=sn)

    return {
        "stat_samples": stat_samples,
        "over_probs": over_probs,
        f"expected_{sn}": float(np.mean(stat_samples)),
        f"std_{sn}": float(np.std(stat_samples)),
        "pa_mu": pa_mu,
        "pa_sigma": pa_sigma,
        "matchup_logit_lift": matchup_lift,
        "matchup_detail": matchup_detail,
        "opposing_pitcher_lift": opposing_pitcher_lift,
    }


def predict_game_batch_stat(
    stat_name: str,
    game_records: pd.DataFrame,
    pitcher_posteriors: dict[int, np.ndarray],
    bf_priors: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame | None = None,
    hitter_vuln: pd.DataFrame | None = None,
    baselines_pt: dict[str, dict[str, float]] | None = None,
    game_batter_ks: pd.DataFrame | None = None,
    game_lineups: pd.DataFrame | None = None,
    context_lifts: dict[int, float] | None = None,
    lineup_proneness_lifts: dict[int, float] | None = None,
    park_factors: dict[int, float] | None = None,
    catcher_framing_lifts: dict[tuple[int, int], float] | None = None,
    rest_df: pd.DataFrame | None = None,
    tto_profiles: pd.DataFrame | None = None,
    model_type: str = "binomial",
    default_lines: list[float] | None = None,
    actual_col: str = "strike_outs",
    n_draws: int = 4000,
    bf_min: int = 3,
    bf_max: int = 35,
) -> pd.DataFrame:
    """Batch predictions for any pitcher stat across many games."""
    sn = stat_name.lower()
    records: list[dict[str, Any]] = []
    n_games = len(game_records)

    _lineup_index: dict[int, pd.DataFrame] = {}
    if game_lineups is not None and not game_lineups.empty:
        for gpk, grp in game_lineups.groupby("game_pk"):
            _lineup_index[int(gpk)] = grp

    _rest_index: dict[tuple[int, int], int] = {}
    if rest_df is not None and not rest_df.empty:
        for _, rrow in rest_df.iterrows():
            _rest_index[(int(rrow["pitcher_id"]), int(rrow["game_pk"]))] = int(
                rrow["days_rest"]
            )

    for i, (_, game) in enumerate(game_records.iterrows()):
        pitcher_id = int(game["pitcher_id"])
        game_pk = int(game["game_pk"])
        season = int(game["season"])
        actual_stat = int(game[actual_col]) if actual_col in game.index else np.nan
        actual_bf = int(game["batters_faced"])

        if pitcher_id not in pitcher_posteriors:
            continue

        rate_samples = pitcher_posteriors[pitcher_id]

        lineup_ids = None
        n_matched = 0

        if game_pk in _lineup_index:
            lu = _lineup_index[game_pk]
            if game_batter_ks is not None:
                faced = set(
                    game_batter_ks[
                        (game_batter_ks["game_pk"] == game_pk)
                        & (game_batter_ks["pitcher_id"] == pitcher_id)
                    ]["batter_id"].tolist()
                )
                for tid in lu["team_id"].unique():
                    team_lu = lu[lu["team_id"] == tid].sort_values("batting_order")
                    if set(team_lu["player_id"].tolist()) & faced:
                        batter_ids = team_lu["player_id"].tolist()
                        n_matched = len(batter_ids)
                        if n_matched >= 9:
                            lineup_ids = batter_ids[:9]
                        break

        if lineup_ids is None and game_batter_ks is not None:
            game_batters = game_batter_ks[
                (game_batter_ks["game_pk"] == game_pk)
                & (game_batter_ks["pitcher_id"] == pitcher_id)
            ]
            batter_ids = game_batters["batter_id"].tolist()
            n_matched = len(batter_ids)
            if n_matched >= 9:
                lineup_ids = batter_ids[:9]
            elif n_matched > 0:
                lineup_ids = (batter_ids * ((9 // len(batter_ids)) + 1))[:9]

        ctx_lift = 0.0
        if context_lifts is not None:
            ctx_lift = context_lifts.get(game_pk, 0.0)

        pf = 1.0
        if park_factors is not None:
            pf = park_factors.get(game_pk, 1.0)

        lp_lift = 0.0
        if lineup_proneness_lifts is not None:
            lp_lift = lineup_proneness_lifts.get(game_pk, 0.0)

        framing_lift = 0.0
        if catcher_framing_lifts is not None:
            framing_lift = catcher_framing_lifts.get((game_pk, pitcher_id), 0.0)

        total_ctx = ctx_lift + framing_lift
        dr = _rest_index.get((pitcher_id, game_pk))

        tto_lifts = None
        if tto_profiles is not None and sn in ("k", "bb", "hr"):
            tto_lifts = build_tto_logit_lifts(
                tto_profiles=tto_profiles,
                pitcher_id=pitcher_id,
                season=season,
                stat_name=sn,
            )

        result = predict_game_stat(
            stat_name=stat_name,
            pitcher_id=pitcher_id,
            season=season,
            lineup_batter_ids=lineup_ids,
            rate_samples=rate_samples,
            bf_priors=bf_priors,
            pitcher_arsenal=pitcher_arsenal,
            hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
            context_logit_lift=total_ctx,
            lineup_proneness_lift=lp_lift,
            park_factor=pf,
            model_type=model_type,
            days_rest=dr,
            tto_logit_lifts=tto_lifts,
            n_draws=n_draws,
            random_seed=42 + i,
            bf_min=bf_min,
            bf_max=bf_max,
        )

        rec: dict[str, Any] = {
            "game_pk": game_pk,
            "pitcher_id": pitcher_id,
            "season": season,
            f"actual_{sn}": actual_stat,
            "actual_bf": actual_bf,
            f"expected_{sn}": result[f"expected_{sn}"],
            f"std_{sn}": result[f"std_{sn}"],
            "pitcher_rate_mean": float(np.mean(rate_samples)),
            "bf_mu": result["bf_mu"],
            "n_matched_batters": n_matched,
            "context_logit_lift": total_ctx,
            "lineup_proneness_lift": lp_lift,
            "catcher_framing_lift": framing_lift,
            "park_factor": pf,
            "days_rest": dr,
            "rest_bucket": result.get("rest_bucket", "normal"),
        }

        over_probs = result["over_probs"]
        for _, row in over_probs.iterrows():
            col_name = f"p_over_{row['line']:.1f}".replace(".", "_")
            rec[col_name] = row["p_over"]

        records.append(rec)

        if (i + 1) % 200 == 0:
            logger.info(
                "Predicted %d / %d games for %s", i + 1, n_games, stat_name
            )

    logger.info(
        "Batch prediction complete for %s: %d games", stat_name, len(records)
    )
    return pd.DataFrame(records)
