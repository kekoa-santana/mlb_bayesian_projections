"""
Step 14: Game-level K posterior Monte Carlo engine.

Combines:
- Pitcher K% posterior samples (Layer 1)
- BF distribution (Step 13)
- Per-batter matchup logit lifts (Layer 2)

to produce a full posterior over game strikeout totals.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.models.bf_model import draw_bf_samples, get_bf_distribution
from src.models.matchup import score_matchup, score_matchup_for_stat

logger = logging.getLogger(__name__)

# Clip bounds for logit transform (avoid infinities)
_CLIP_LO = 1e-6
_CLIP_HI = 1 - 1e-6


def _safe_logit(p: np.ndarray) -> np.ndarray:
    """Logit with clipping."""
    return logit(np.clip(p, _CLIP_LO, _CLIP_HI))


def simulate_game_ks(
    pitcher_k_rate_samples: np.ndarray,
    bf_mu: float,
    bf_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    umpire_k_logit_lift: float = 0.0,
    weather_k_logit_lift: float = 0.0,
    n_draws: int = 4000,
    bf_min: int = 3,
    bf_max: int = 35,
    random_seed: int = 42,
) -> np.ndarray:
    """Monte Carlo simulation of game strikeout totals.

    Parameters
    ----------
    pitcher_k_rate_samples : np.ndarray
        K% posterior samples from Layer 1 (values in [0, 1]).
    bf_mu : float
        Mean batters faced for this pitcher.
    bf_sigma : float
        Std of batters faced.
    lineup_matchup_lifts : np.ndarray or None
        Shape (9,) logit-scale lifts per batting order slot.
        Positive = batter more vulnerable → more Ks.
        None = no matchup adjustment (baseline mode).
    umpire_k_logit_lift : float
        Logit-scale shift for HP umpire K-rate tendency.
        Positive = umpire calls more Ks than average.
    weather_k_logit_lift : float
        Logit-scale shift for weather effect on K-rate.
        Positive = weather conditions increase Ks (e.g. cold).
    n_draws : int
        Number of Monte Carlo draws.
    bf_min : int
        Minimum BF per game.
    bf_max : int
        Maximum BF per game.
    random_seed : int
        For reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_draws,) of integer K totals per simulated game.
    """
    rng = np.random.default_rng(random_seed)

    # Resample pitcher K% to n_draws if needed
    if len(pitcher_k_rate_samples) != n_draws:
        idx = rng.choice(len(pitcher_k_rate_samples), size=n_draws, replace=True)
        k_rate_draws = pitcher_k_rate_samples[idx]
    else:
        k_rate_draws = pitcher_k_rate_samples.copy()

    # Draw BF samples
    bf_draws = draw_bf_samples(
        mu_bf=bf_mu, sigma_bf=bf_sigma,
        n_draws=n_draws, bf_min=bf_min, bf_max=bf_max, rng=rng,
    )

    # Default: no matchup adjustment
    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(9)

    # Convert pitcher K% to logit scale and apply umpire + weather adjustments
    k_logit = _safe_logit(k_rate_draws) + umpire_k_logit_lift + weather_k_logit_lift

    # Vectorize by grouping draws with same BF value
    k_totals = np.zeros(n_draws, dtype=int)

    unique_bf = np.unique(bf_draws)
    for bf_val in unique_bf:
        mask = bf_draws == bf_val
        n_bf_draws = mask.sum()
        bf_int = int(bf_val)

        # Allocate PA across 9 batting order slots
        base_pa = bf_int // 9
        extra = bf_int % 9
        pa_per_slot = np.full(9, base_pa, dtype=int)
        pa_per_slot[:extra] += 1

        # For each slot with PA > 0, simulate Ks
        game_ks = np.zeros(n_bf_draws, dtype=int)
        k_logit_subset = k_logit[mask]

        for slot in range(9):
            if pa_per_slot[slot] == 0:
                continue
            # Adjust K rate by matchup lift for this slot
            adjusted_logit = k_logit_subset + lineup_matchup_lifts[slot]
            adjusted_p = expit(adjusted_logit)
            # Binomial draw: K per slot
            slot_ks = rng.binomial(n=pa_per_slot[slot], p=adjusted_p)
            game_ks += slot_ks

        k_totals[mask] = game_ks

    return k_totals


def compute_k_over_probs(
    k_samples: np.ndarray,
    lines: list[float] | None = None,
) -> pd.DataFrame:
    """Compute P(over X.5) for standard K prop lines.

    Parameters
    ----------
    k_samples : np.ndarray
        Monte Carlo K total samples.
    lines : list[float] or None
        Lines to evaluate. Default: [0.5, 1.5, ..., 12.5].

    Returns
    -------
    pd.DataFrame
        Columns: line, p_over, p_under, expected_k, std_k.
    """
    if lines is None:
        lines = [x + 0.5 for x in range(13)]

    expected_k = float(np.mean(k_samples))
    std_k = float(np.std(k_samples))

    records = []
    for line in lines:
        p_over = float(np.mean(k_samples > line))
        records.append({
            "line": line,
            "p_over": p_over,
            "p_under": 1.0 - p_over,
            "expected_k": expected_k,
            "std_k": std_k,
        })

    return pd.DataFrame(records)


def extract_pitcher_k_rate_samples(
    trace: Any,
    data: dict[str, Any],
    pitcher_id: int,
    season: int,
    project_forward: bool = True,
    random_seed: int = 42,
) -> np.ndarray:
    """Extract raw K% posterior samples for one pitcher.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted pitcher K% model trace.
    data : dict
        Model data dict from ``prepare_pitcher_model_data``.
    pitcher_id : int
        Target pitcher.
    season : int
        Season whose posterior to extract.
    project_forward : bool
        If True, add AR(1) forward projection noise (for out-of-sample
        prediction). Uses rho dampening on the season effect.
    random_seed : int
        For reproducibility of forward projection noise.

    Returns
    -------
    np.ndarray
        K% posterior samples (1D array, values in [0, 1]).

    Raises
    ------
    ValueError
        If pitcher not found in the data for the given season.
    """
    df = data["df"]
    mask = (df["pitcher_id"] == pitcher_id) & (df["season"] == season)
    positions = df.index[mask].tolist()

    if not positions:
        raise ValueError(
            f"Pitcher {pitcher_id} not found in season {season}"
        )

    pos = positions[0]
    # Get the integer position in the DataFrame
    iloc_pos = df.index.get_loc(pos)

    # Extract posterior samples: (chains, draws, n_obs)
    k_rate_post = trace.posterior["k_rate"].values
    k_rate_flat = k_rate_post.reshape(-1, k_rate_post.shape[-1])
    samples = k_rate_flat[:, iloc_pos].copy()

    if project_forward and "sigma_season" in trace.posterior:
        rng = np.random.default_rng(random_seed)
        sigma_samples = trace.posterior["sigma_season"].values.flatten()
        # Resample sigma to match samples length
        if len(sigma_samples) != len(samples):
            sigma_draws = rng.choice(sigma_samples, size=len(samples), replace=True)
        else:
            sigma_draws = sigma_samples

        # Get rho for AR(1) dampening
        if "rho" in trace.posterior:
            rho_samples = trace.posterior["rho"].values.flatten()
            if len(rho_samples) != len(samples):
                rho_draws = rng.choice(rho_samples, size=len(samples), replace=True)
            else:
                rho_draws = rho_samples
        else:
            rho_draws = np.ones(len(samples))  # fallback: pure random walk

        # AR(1) forward projection on logit scale
        logit_samples = _safe_logit(samples)
        innovation = rng.normal(0, sigma_draws)

        # Extract alpha to compute season effect
        alpha_post = trace.posterior["alpha"].values
        alpha_flat = alpha_post.reshape(-1, alpha_post.shape[-1])
        pidx = data["player_map"][pitcher_id]
        alpha_draws = alpha_flat[:, pidx]
        if len(alpha_draws) != len(samples):
            alpha_draws = rng.choice(alpha_draws, size=len(samples), replace=True)

        # season_effect_last = logit(rate) - alpha
        season_effect_last = logit_samples - alpha_draws
        # AR(1) forward: new_effect = rho * last_effect + innovation
        new_effect = rho_draws * season_effect_last + innovation
        samples = expit(alpha_draws + new_effect)

    return samples


def _compute_lineup_matchup_lifts(
    pitcher_id: int,
    lineup_batter_ids: list[int],
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Score matchups for a 9-batter lineup.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    lineup_batter_ids : list[int]
        Exactly 9 batter IDs in batting order.
    pitcher_arsenal : pd.DataFrame
        Pitcher's arsenal profile for the season.
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles for the season.
    baselines_pt : dict
        League baselines per pitch type.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        (lifts array shape (9,), list of per-batter matchup dicts)
    """
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
    n_draws: int = 4000,
    random_seed: int = 42,
    bf_min: int = 3,
    bf_max: int = 35,
) -> dict[str, Any]:
    """Full game K prediction combining all layers.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    season : int
        Season for BF lookup.
    lineup_batter_ids : list[int] or None
        9 batter IDs in order. None = no matchup adjustment.
    pitcher_k_rate_samples : np.ndarray
        Posterior K% samples from Layer 1.
    bf_priors : pd.DataFrame
        BF priors from ``compute_pitcher_bf_priors``.
    pitcher_arsenal : pd.DataFrame or None
        Pitcher arsenal for matchup scoring. Required if lineup given.
    hitter_vuln : pd.DataFrame or None
        Hitter vulnerability profiles. Required if lineup given.
    baselines_pt : dict or None
        League baselines per pitch type. Required if lineup given.
    umpire_k_logit_lift : float
        Logit-scale shift for HP umpire K-rate tendency (0.0 = neutral).
    weather_k_logit_lift : float
        Logit-scale shift for weather effect on K-rate (0.0 = neutral).
    n_draws : int
        Monte Carlo draws.
    random_seed : int
        For reproducibility.
    bf_min, bf_max : int
        BF bounds.

    Returns
    -------
    dict
        Keys: k_samples, k_over_probs, expected_k, std_k,
        bf_mu, bf_sigma, lineup_matchup_lifts, per_batter_details,
        umpire_k_logit_lift, weather_k_logit_lift.
    """
    # BF distribution
    bf_info = get_bf_distribution(pitcher_id, season, bf_priors)
    bf_mu = bf_info["mu_bf"]
    bf_sigma = bf_info["sigma_bf"]

    # Matchup lifts
    lineup_lifts = None
    per_batter_details = []
    if lineup_batter_ids is not None and len(lineup_batter_ids) == 9:
        if pitcher_arsenal is not None and hitter_vuln is not None and baselines_pt is not None:
            lineup_lifts, per_batter_details = _compute_lineup_matchup_lifts(
                pitcher_id, lineup_batter_ids,
                pitcher_arsenal, hitter_vuln, baselines_pt,
            )

    # Monte Carlo simulation
    k_samples = simulate_game_ks(
        pitcher_k_rate_samples=pitcher_k_rate_samples,
        bf_mu=bf_mu,
        bf_sigma=bf_sigma,
        lineup_matchup_lifts=lineup_lifts,
        umpire_k_logit_lift=umpire_k_logit_lift,
        weather_k_logit_lift=weather_k_logit_lift,
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
    n_draws: int = 4000,
    bf_min: int = 3,
    bf_max: int = 35,
) -> pd.DataFrame:
    """Batch predictions for backtesting across many games.

    Parameters
    ----------
    game_records : pd.DataFrame
        One row per starter game. Must have: game_pk, pitcher_id, season,
        strike_outs, batters_faced.
    pitcher_posteriors : dict[int, np.ndarray]
        Mapping of pitcher_id → K% posterior samples.
    bf_priors : pd.DataFrame
        BF priors from ``compute_pitcher_bf_priors``.
    pitcher_arsenal : pd.DataFrame or None
        Full pitcher arsenal data for matchup scoring.
    hitter_vuln : pd.DataFrame or None
        Full hitter vulnerability data.
    baselines_pt : dict or None
        League baselines per pitch type.
    game_batter_ks : pd.DataFrame or None
        Per (game_pk, pitcher_id, batter_id) PA/K. Fallback for lineup
        reconstruction when game_lineups not available.
    game_lineups : pd.DataFrame or None
        Real starting lineups from fact_lineup. Columns: game_pk,
        player_id, batting_order, team_id. Preferred over game_batter_ks.
    umpire_lifts : dict[int, float] or None
        Mapping of game_pk → umpire K logit lift.
    weather_lifts : dict[int, float] or None
        Mapping of game_pk → weather K logit lift.
    n_draws : int
        Monte Carlo draws per game.
    bf_min, bf_max : int
        BF bounds.

    Returns
    -------
    pd.DataFrame
        One row per game with: game_pk, pitcher_id, season, actual_k,
        actual_bf, expected_k, std_k, p_over_X_5 columns,
        pitcher_k_rate_mean, bf_mu, n_matched_batters,
        umpire_k_logit_lift, weather_k_logit_lift.
    """
    records = []
    n_games = len(game_records)

    # Pre-index game_lineups by game_pk for fast lookup
    _lineup_index: dict[int, pd.DataFrame] = {}
    if game_lineups is not None and not game_lineups.empty:
        for gpk, grp in game_lineups.groupby("game_pk"):
            _lineup_index[int(gpk)] = grp

    for i, (_, game) in enumerate(game_records.iterrows()):
        pitcher_id = int(game["pitcher_id"])
        game_pk = int(game["game_pk"])
        season = int(game["season"])
        actual_k = int(game["strike_outs"])
        actual_bf = int(game["batters_faced"])

        if pitcher_id not in pitcher_posteriors:
            continue

        k_rate_samples = pitcher_posteriors[pitcher_id]

        # --- Lineup: prefer real lineups, fall back to game_batter_ks ---
        lineup_ids = None
        n_matched = 0

        if game_pk in _lineup_index:
            # Real lineups: find the opposing team (the team whose batters
            # faced this pitcher, identified via game_batter_ks overlap)
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
            # Fallback: reconstruct from game_batter_ks
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

        # --- Game context lifts ---
        ump_lift = 0.0
        if umpire_lifts is not None:
            ump_lift = umpire_lifts.get(game_pk, 0.0)

        wx_lift = 0.0
        if weather_lifts is not None:
            wx_lift = weather_lifts.get(game_pk, 0.0)

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
        }

        # Add P(over) columns
        k_over = result["k_over_probs"]
        for _, row in k_over.iterrows():
            col_name = f"p_over_{row['line']:.1f}".replace(".", "_")
            rec[col_name] = row["p_over"]

        records.append(rec)

        if (i + 1) % 200 == 0:
            logger.info("Predicted %d / %d games", i + 1, n_games)

    logger.info("Batch prediction complete: %d games", len(records))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Generalized simulation functions (all prop types: K, BB, HR, H, Outs)
# ---------------------------------------------------------------------------

# Mapping from stat_name → matchup lift key in score_matchup_for_stat results
_STAT_LIFT_KEYS: dict[str, str] = {
    "k": "matchup_k_logit_lift",
    "bb": "matchup_bb_logit_lift",
    "hr": "matchup_hr_logit_lift",
}


def simulate_game_stat(
    rate_samples: np.ndarray,
    opp_mu: float,
    opp_sigma: float,
    lineup_matchup_lifts: np.ndarray | None = None,
    context_logit_lift: float = 0.0,
    n_draws: int = 4000,
    opp_min: int = 3,
    opp_max: int = 35,
    n_slots: int = 9,
    random_seed: int = 42,
) -> np.ndarray:
    """Stat-agnostic Binomial Monte Carlo simulation.

    Works for any rate-based stat (K, BB, H) for either pitchers or batters.
    For pitchers: opp = BF, n_slots = 9 (lineup).
    For batters: opp = PA, n_slots = 1 (no lineup splitting needed).

    Parameters
    ----------
    rate_samples : np.ndarray
        Rate posterior samples (values in [0, 1]).
    opp_mu : float
        Mean opportunities (BF for pitchers, PA for batters).
    opp_sigma : float
        Std of opportunities.
    lineup_matchup_lifts : np.ndarray or None
        Shape (n_slots,) logit-scale lifts per slot.
        For batters, shape (1,) with the single matchup lift.
    context_logit_lift : float
        Additional logit-scale context shift (umpire, weather, park).
    n_draws : int
        Number of Monte Carlo draws.
    opp_min, opp_max : int
        Opportunity bounds.
    n_slots : int
        Number of lineup slots to distribute opportunities across.
        Use 9 for pitcher props, 1 for batter props.
    random_seed : int

    Returns
    -------
    np.ndarray
        Shape (n_draws,) of integer stat totals.
    """
    rng = np.random.default_rng(random_seed)

    # Resample rate to n_draws if needed
    if len(rate_samples) != n_draws:
        idx = rng.choice(len(rate_samples), size=n_draws, replace=True)
        rate_draws = rate_samples[idx]
    else:
        rate_draws = rate_samples.copy()

    # Draw opportunity samples (BF or PA)
    opp_draws = draw_bf_samples(
        mu_bf=opp_mu, sigma_bf=opp_sigma,
        n_draws=n_draws, bf_min=opp_min, bf_max=opp_max, rng=rng,
    )

    # Default: no matchup adjustment
    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(n_slots)

    # Convert rate to logit scale and apply context adjustment
    rate_logit = _safe_logit(rate_draws) + context_logit_lift

    stat_totals = np.zeros(n_draws, dtype=int)

    if n_slots == 1:
        # Batter mode: single Binomial draw per MC iteration
        adjusted_logit = rate_logit + lineup_matchup_lifts[0]
        adjusted_p = expit(adjusted_logit)
        stat_totals = rng.binomial(n=opp_draws.astype(int), p=adjusted_p)
    else:
        # Pitcher mode: distribute opportunities across lineup slots
        unique_opp = np.unique(opp_draws)
        for opp_val in unique_opp:
            mask = opp_draws == opp_val
            n_opp_draws = mask.sum()
            opp_int = int(opp_val)

            # Allocate PA across slots
            base_pa = opp_int // n_slots
            extra = opp_int % n_slots
            pa_per_slot = np.full(n_slots, base_pa, dtype=int)
            pa_per_slot[:extra] += 1

            game_stats = np.zeros(n_opp_draws, dtype=int)
            rate_logit_subset = rate_logit[mask]

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
    n_draws: int = 4000,
    opp_min: int = 3,
    opp_max: int = 35,
    n_slots: int = 9,
    random_seed: int = 42,
) -> np.ndarray:
    """Poisson Monte Carlo simulation for rare events (HR).

    For rare events (~3% rate), Poisson is more appropriate than Binomial.
    lambda = rate * opportunities * park_factor

    For pitchers: rate = HR/BF, opp = BF
    For batters: rate = HR/PA, opp = PA

    Parameters
    ----------
    rate_samples : np.ndarray
        Rate posterior samples (values in [0, 1]).
    opp_mu : float
        Mean opportunities (BF for pitchers, PA for batters).
    opp_sigma : float
        Std of opportunities.
    lineup_matchup_lifts : np.ndarray or None
        Shape (n_slots,) logit-scale lifts per slot.
    context_logit_lift : float
        Additional logit-scale context shift.
    park_factor : float
        Multiplicative park factor for the stat (e.g., HR park factor).
    n_draws : int
        Number of Monte Carlo draws.
    opp_min, opp_max : int
        Opportunity bounds.
    n_slots : int
        Number of lineup slots (9 for pitchers, 1 for batters).
    random_seed : int

    Returns
    -------
    np.ndarray
        Shape (n_draws,) of integer stat totals.
    """
    rng = np.random.default_rng(random_seed)

    # Resample rate to n_draws if needed
    if len(rate_samples) != n_draws:
        idx = rng.choice(len(rate_samples), size=n_draws, replace=True)
        rate_draws = rate_samples[idx]
    else:
        rate_draws = rate_samples.copy()

    # Draw opportunity samples
    opp_draws = draw_bf_samples(
        mu_bf=opp_mu, sigma_bf=opp_sigma,
        n_draws=n_draws, bf_min=opp_min, bf_max=opp_max, rng=rng,
    )

    # Default: no matchup adjustment
    if lineup_matchup_lifts is None:
        lineup_matchup_lifts = np.zeros(n_slots)

    # Convert rate to logit scale, apply context shift
    rate_logit = _safe_logit(rate_draws) + context_logit_lift

    stat_totals = np.zeros(n_draws, dtype=int)

    if n_slots == 1:
        # Batter mode: single Poisson draw per MC iteration
        adjusted_logit = rate_logit + lineup_matchup_lifts[0]
        adjusted_rate = expit(adjusted_logit)
        lam = adjusted_rate * opp_draws * park_factor
        stat_totals = rng.poisson(lam=lam)
    else:
        # Pitcher mode: sum Poisson draws across lineup slots
        unique_opp = np.unique(opp_draws)
        for opp_val in unique_opp:
            mask = opp_draws == opp_val
            n_opp_draws = mask.sum()
            opp_int = int(opp_val)

            # Allocate PA across slots
            base_pa = opp_int // n_slots
            extra = opp_int % n_slots
            pa_per_slot = np.full(n_slots, base_pa, dtype=int)
            pa_per_slot[:extra] += 1

            game_stats = np.zeros(n_opp_draws, dtype=int)
            rate_logit_subset = rate_logit[mask]

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


def compute_over_probs(
    stat_samples: np.ndarray,
    lines: list[float] | None = None,
    stat_name: str = "stat",
) -> pd.DataFrame:
    """Compute P(over X.5) for prop lines.

    Parameters
    ----------
    stat_samples : np.ndarray
        MC samples of stat totals.
    lines : list[float] or None
        Lines to evaluate. If None, auto-generate based on stat range.
    stat_name : str
        Name for column labeling.

    Returns
    -------
    pd.DataFrame
        Columns: line, p_over, p_under, expected_{stat_name}, std_{stat_name}.
    """
    if lines is None:
        max_val = int(np.max(stat_samples)) if len(stat_samples) > 0 else 5
        # Generate lines from 0.5 up to max observed + 0.5
        upper = min(max_val + 1, 20)
        lines = [x + 0.5 for x in range(upper)]

    expected = float(np.mean(stat_samples))
    std = float(np.std(stat_samples))

    records = []
    for line in lines:
        p_over = float(np.mean(stat_samples > line))
        records.append({
            "line": line,
            "p_over": p_over,
            "p_under": 1.0 - p_over,
            f"expected_{stat_name}": expected,
            f"std_{stat_name}": std,
        })

    return pd.DataFrame(records)


def _compute_lineup_matchup_lifts_for_stat(
    stat_name: str,
    pitcher_id: int,
    lineup_batter_ids: list[int],
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Score matchups for a 9-batter lineup for any stat type.

    Parameters
    ----------
    stat_name : str
        Stat to score ('k', 'bb', 'hr', etc.).
    pitcher_id : int
        Pitcher MLB ID.
    lineup_batter_ids : list[int]
        Exactly 9 batter IDs in batting order.
    pitcher_arsenal : pd.DataFrame
        Pitcher's arsenal profile for the season.
    hitter_vuln : pd.DataFrame
        Hitter vulnerability profiles for the season.
    baselines_pt : dict
        League baselines per pitch type.

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        (lifts array shape (9,), list of per-batter matchup dicts)
    """
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
    park_factor: float = 1.0,
    model_type: str = "binomial",
    n_draws: int = 4000,
    random_seed: int = 42,
    bf_min: int = 3,
    bf_max: int = 35,
) -> dict[str, Any]:
    """Full game stat prediction for a pitcher, combining all layers.

    Parameters
    ----------
    stat_name : str
        Stat being predicted ('k', 'bb', 'hr', 'h', 'outs').
    pitcher_id : int
        Pitcher MLB ID.
    season : int
        Season for BF lookup.
    lineup_batter_ids : list[int] or None
        9 batter IDs in order. None = no matchup adjustment.
    rate_samples : np.ndarray
        Posterior rate samples from Layer 1 (values in [0, 1]).
    bf_priors : pd.DataFrame
        BF priors from ``compute_pitcher_bf_priors``.
    pitcher_arsenal : pd.DataFrame or None
        Pitcher arsenal for matchup scoring. Required if lineup given.
    hitter_vuln : pd.DataFrame or None
        Hitter vulnerability profiles. Required if lineup given.
    baselines_pt : dict or None
        League baselines per pitch type. Required if lineup given.
    context_logit_lift : float
        Logit-scale context shift (umpire + weather + park combined).
    park_factor : float
        Multiplicative park factor. Used only for Poisson model (HR).
    model_type : str
        'binomial' or 'poisson'. Use 'poisson' for rare events (HR).
    n_draws : int
        Monte Carlo draws.
    random_seed : int
        For reproducibility.
    bf_min, bf_max : int
        BF bounds.

    Returns
    -------
    dict
        Keys: stat_samples, over_probs, expected_{stat_name},
        std_{stat_name}, bf_mu, bf_sigma, lineup_matchup_lifts,
        per_batter_details, context_logit_lift, park_factor.
    """
    # BF distribution
    bf_info = get_bf_distribution(pitcher_id, season, bf_priors)
    bf_mu = bf_info["mu_bf"]
    bf_sigma = bf_info["sigma_bf"]

    # Matchup lifts
    lineup_lifts = None
    per_batter_details: list[dict[str, Any]] = []
    if lineup_batter_ids is not None and len(lineup_batter_ids) == 9:
        if pitcher_arsenal is not None and hitter_vuln is not None and baselines_pt is not None:
            lineup_lifts, per_batter_details = _compute_lineup_matchup_lifts_for_stat(
                stat_name, pitcher_id, lineup_batter_ids,
                pitcher_arsenal, hitter_vuln, baselines_pt,
            )

    # Monte Carlo simulation
    if model_type == "poisson":
        stat_samples = simulate_game_stat_poisson(
            rate_samples=rate_samples,
            opp_mu=bf_mu,
            opp_sigma=bf_sigma,
            lineup_matchup_lifts=lineup_lifts,
            context_logit_lift=context_logit_lift,
            park_factor=park_factor,
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
            context_logit_lift=context_logit_lift,
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
        "park_factor": park_factor,
    }


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
    """Monte Carlo simulation for a single batter's game stat total.

    For batters, there's no lineup splitting — just rate * PA draws.

    Parameters
    ----------
    rate_samples : np.ndarray
        Rate posterior samples (values in [0, 1]).
    pa_mu : float
        Mean PA for this batter in a game.
    pa_sigma : float
        Std of PA.
    matchup_logit_lift : float
        Logit-scale matchup adjustment vs the opposing pitcher.
    context_logit_lift : float
        Additional logit-scale context shift (umpire, weather).
    park_factor : float
        Multiplicative park factor. Used only for Poisson model (HR).
    model_type : str
        'binomial' or 'poisson'. Use 'poisson' for rare events (HR).
    n_draws : int
        Number of Monte Carlo draws.
    pa_min, pa_max : int
        PA bounds.
    random_seed : int

    Returns
    -------
    np.ndarray
        Shape (n_draws,) of integer stat totals.
    """
    combined_lift = matchup_logit_lift + context_logit_lift
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
    park_factor: float = 1.0,
    model_type: str = "binomial",
    default_lines: list[float] | None = None,
    n_draws: int = 4000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Full game stat prediction for a single batter.

    Parameters
    ----------
    stat_name : str
        Stat being predicted ('k', 'bb', 'hr', 'h').
    batter_id : int
        Batter MLB ID.
    pitcher_id : int
        Opposing pitcher MLB ID.
    rate_samples : np.ndarray
        Posterior rate samples from Layer 1 (values in [0, 1]).
    pa_mu : float
        Mean PA for this batter in a game.
    pa_sigma : float
        Std of PA.
    pitcher_arsenal : pd.DataFrame or None
        Pitcher arsenal for matchup scoring.
    hitter_vuln : pd.DataFrame or None
        Hitter vulnerability profiles.
    baselines_pt : dict or None
        League baselines per pitch type.
    context_logit_lift : float
        Logit-scale context shift (umpire + weather).
    park_factor : float
        Multiplicative park factor. Used only for Poisson model (HR).
    model_type : str
        'binomial' or 'poisson'.
    default_lines : list[float] or None
        Lines for P(over) computation. Auto-generated if None.
    n_draws : int
        Monte Carlo draws.
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict
        Keys: stat_samples, over_probs, expected_{stat_name},
        std_{stat_name}, pa_mu, pa_sigma, matchup_logit_lift.
    """
    sn = stat_name.lower()
    lift_key = _STAT_LIFT_KEYS.get(sn, "matchup_logit_lift")

    # Compute matchup lift
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

    # Simulate
    stat_samples = simulate_batter_game_stat(
        rate_samples=rate_samples,
        pa_mu=pa_mu,
        pa_sigma=pa_sigma,
        matchup_logit_lift=matchup_lift,
        context_logit_lift=context_logit_lift,
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
    park_factors: dict[int, float] | None = None,
    model_type: str = "binomial",
    default_lines: list[float] | None = None,
    actual_col: str = "strike_outs",
    n_draws: int = 4000,
    bf_min: int = 3,
    bf_max: int = 35,
) -> pd.DataFrame:
    """Batch predictions for any pitcher stat across many games.

    Generalizes ``predict_game_batch()`` for any stat type.

    Parameters
    ----------
    stat_name : str
        Stat being predicted ('k', 'bb', 'hr', 'h', 'outs').
    game_records : pd.DataFrame
        One row per starter game. Must have: game_pk, pitcher_id, season,
        ``actual_col``, batters_faced.
    pitcher_posteriors : dict[int, np.ndarray]
        Mapping of pitcher_id → rate posterior samples.
    bf_priors : pd.DataFrame
        BF priors from ``compute_pitcher_bf_priors``.
    pitcher_arsenal : pd.DataFrame or None
        Full pitcher arsenal data for matchup scoring.
    hitter_vuln : pd.DataFrame or None
        Full hitter vulnerability data.
    baselines_pt : dict or None
        League baselines per pitch type.
    game_batter_ks : pd.DataFrame or None
        Per (game_pk, pitcher_id, batter_id) PA data. Fallback for lineup
        reconstruction when game_lineups not available.
    game_lineups : pd.DataFrame or None
        Real starting lineups. Columns: game_pk, player_id, batting_order,
        team_id. Preferred over game_batter_ks.
    context_lifts : dict[int, float] or None
        Mapping of game_pk → combined context logit lift.
    park_factors : dict[int, float] or None
        Mapping of game_pk → park factor (multiplicative, Poisson only).
    model_type : str
        'binomial' or 'poisson'.
    default_lines : list[float] or None
        Lines for P(over) computation. Auto-generated if None.
    actual_col : str
        Column name in game_records for actual stat totals.
    n_draws : int
        Monte Carlo draws per game.
    bf_min, bf_max : int
        BF bounds.

    Returns
    -------
    pd.DataFrame
        One row per game with: game_pk, pitcher_id, season,
        actual_{stat_name}, actual_bf, expected_{stat_name},
        std_{stat_name}, p_over_X_5 columns, pitcher_rate_mean,
        bf_mu, n_matched_batters, context_logit_lift, park_factor.
    """
    sn = stat_name.lower()
    records: list[dict[str, Any]] = []
    n_games = len(game_records)

    # Pre-index game_lineups by game_pk for fast lookup
    _lineup_index: dict[int, pd.DataFrame] = {}
    if game_lineups is not None and not game_lineups.empty:
        for gpk, grp in game_lineups.groupby("game_pk"):
            _lineup_index[int(gpk)] = grp

    for i, (_, game) in enumerate(game_records.iterrows()):
        pitcher_id = int(game["pitcher_id"])
        game_pk = int(game["game_pk"])
        season = int(game["season"])
        actual_stat = int(game[actual_col]) if actual_col in game.index else np.nan
        actual_bf = int(game["batters_faced"])

        if pitcher_id not in pitcher_posteriors:
            continue

        rate_samples = pitcher_posteriors[pitcher_id]

        # --- Lineup: prefer real lineups, fall back to game_batter_ks ---
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

        # --- Game context ---
        ctx_lift = 0.0
        if context_lifts is not None:
            ctx_lift = context_lifts.get(game_pk, 0.0)

        pf = 1.0
        if park_factors is not None:
            pf = park_factors.get(game_pk, 1.0)

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
            context_logit_lift=ctx_lift,
            park_factor=pf,
            model_type=model_type,
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
            "context_logit_lift": ctx_lift,
            "park_factor": pf,
        }

        # Add P(over) columns
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
