"""
Season-level Monte Carlo projections via the game simulator.

Starter seasons: batch N starts into ``simulate_game()`` calls, sum per season.
Reliever seasons: lightweight vectorized PA resolution (no ExitModel/TTO).
Produces correlated joint distributions over all counting stats + fantasy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.reliever_roles import RelieverRolePriors, get_role_priors

logger = logging.getLogger(__name__)

# League-average unearned run fraction (runs * this = earned runs)
UNEARNED_RUN_FRACTION = 0.08

# Starter population priors
POP_STARTER_N_STARTS = 30
POP_STARTER_N_STARTS_SD = 6

# Maximum games / BF caps
MAX_STARTER_GAMES = 36
MAX_RELIEVER_GAMES = 85
MAX_SEASON_BF = 900


@dataclass
class SeasonSimResult:
    """Results from a season-level simulation.

    All arrays have shape (n_seasons,) — one value per simulated season.
    """

    pitcher_id: int
    role: str                    # "SP", "CL", "SU", "MR"
    k_season: np.ndarray
    bb_season: np.ndarray
    h_season: np.ndarray
    hr_season: np.ndarray
    hbp_season: np.ndarray
    bf_season: np.ndarray
    outs_season: np.ndarray
    runs_season: np.ndarray
    pitches_season: np.ndarray
    games_season: np.ndarray
    sv_season: np.ndarray
    hld_season: np.ndarray
    dk_season: np.ndarray
    espn_season: np.ndarray
    n_seasons: int = 0

    def ip_season(self) -> np.ndarray:
        """Compute innings pitched from outs."""
        return self.outs_season.astype(float) / 3.0

    def era_season(self) -> np.ndarray:
        """Compute ERA from runs (using unearned run fraction)."""
        er = self.runs_season.astype(float) * (1.0 - UNEARNED_RUN_FRACTION)
        ip = self.ip_season()
        safe_ip = np.where(ip > 0, ip, np.nan)
        return er / safe_ip * 9.0

    def fip_era_season(self, fip_constant: float = 3.15) -> np.ndarray:
        """FIP-derived ERA from the sim's own K/BB/HR distributions.

        Uses the standard FIP formula applied to season-level sim outputs.
        More predictive of future ERA than run-based derivation because it
        strips out BABIP/sequencing noise.

        Parameters
        ----------
        fip_constant : float
            League FIP constant (typically 3.10-3.20).
        """
        ip = self.ip_season()
        safe_ip = np.where(ip > 0, ip, np.nan)
        return (
            13.0 * self.hr_season.astype(float)
            + 3.0 * (self.bb_season + self.hbp_season).astype(float)
            - 2.0 * self.k_season.astype(float)
        ) / safe_ip + fip_constant

    def runs_saved_season(self, lg_fip: float = 4.15) -> np.ndarray:
        """FIP-based runs saved above average.

        Pitcher equivalent of wRAA — measures total run prevention value
        in the same unit (runs above average). ~10 runs saved per 1 WAR.

        Parameters
        ----------
        lg_fip : float
            League-average FIP (typically 4.10-4.20).
        """
        fip = self.fip_era_season()
        ip = self.ip_season()
        safe_ip = np.where(ip > 0, ip, np.nan)
        return (lg_fip - fip) / 9.0 * safe_ip

    def whip_season(self) -> np.ndarray:
        """Compute WHIP from BB, H, IP."""
        ip = self.ip_season()
        safe_ip = np.where(ip > 0, ip, np.nan)
        return (self.bb_season + self.h_season).astype(float) / safe_ip

    def summary(self) -> dict[str, dict[str, float]]:
        """Summary statistics for all stats."""
        stats = {}
        for name, arr in [
            ("k", self.k_season), ("bb", self.bb_season),
            ("h", self.h_season), ("hr", self.hr_season),
            ("hbp", self.hbp_season), ("bf", self.bf_season),
            ("outs", self.outs_season), ("runs", self.runs_season),
            ("pitches", self.pitches_season), ("games", self.games_season),
            ("sv", self.sv_season), ("hld", self.hld_season),
            ("dk", self.dk_season), ("espn", self.espn_season),
        ]:
            stats[name] = _stat_summary(arr)

        # Derived
        ip = self.ip_season()
        stats["ip"] = _stat_summary(ip)
        era = self.era_season()
        era_valid = era[~np.isnan(era)]
        if len(era_valid) > 0:
            stats["era"] = _stat_summary(np.clip(era_valid, 0, 15))
        fip_era = self.fip_era_season()
        fip_valid = fip_era[~np.isnan(fip_era)]
        if len(fip_valid) > 0:
            stats["fip_era"] = _stat_summary(np.clip(fip_valid, 0, 15))
        rs = self.runs_saved_season()
        rs_valid = rs[~np.isnan(rs)]
        if len(rs_valid) > 0:
            stats["runs_saved"] = _stat_summary(rs_valid)
        whip = self.whip_season()
        whip_valid = whip[~np.isnan(whip)]
        if len(whip_valid) > 0:
            stats["whip"] = _stat_summary(np.clip(whip_valid, 0, 5))

        return stats


def _stat_summary(arr: np.ndarray) -> dict[str, float]:
    """Compute summary statistics."""
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "sd": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
    }


# ===================================================================
# Starter Season Simulation
# ===================================================================

def simulate_starter_season(
    pitcher_id: int,
    k_rate_samples: np.ndarray,
    bb_rate_samples: np.ndarray,
    hr_rate_samples: np.ndarray,
    exit_model: Any,
    n_starts_mu: float = 30.0,
    n_starts_sigma: float = 6.0,
    pitcher_avg_pitches: float = 88.0,
    babip_adj: float = 0.0,
    n_seasons: int = 200,
    random_seed: int = 42,
) -> SeasonSimResult:
    """Simulate full seasons for a starter using the game simulator.

    For each simulated season, draws a number of starts and batches them
    into a single ``simulate_game()`` call, then sums game outcomes.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    k_rate_samples, bb_rate_samples, hr_rate_samples : np.ndarray
        Layer 1 posterior rate samples.
    exit_model : ExitModel
        Trained pitcher exit model.
    n_starts_mu, n_starts_sigma : float
        Pitcher-specific starts prior (mean, std).
    pitcher_avg_pitches : float
        Average exit pitch count for exit model.
    babip_adj : float
        Pitcher BABIP adjustment.
    n_seasons : int
        Number of season simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    SeasonSimResult
    """
    from src.models.game_sim.simulator import simulate_game
    from src.models.game_sim.fantasy_scoring import (
        compute_season_pitcher_fantasy,
    )

    rng = np.random.default_rng(random_seed)

    # Draw number of starts per season
    starts_per_season = rng.normal(n_starts_mu, n_starts_sigma, size=n_seasons)
    starts_per_season = np.clip(np.round(starts_per_season), 5, MAX_STARTER_GAMES).astype(int)

    total_sims = int(starts_per_season.sum())

    # League-average lineup: no matchup lifts, no umpire/weather
    no_lifts = {"k": np.zeros(9), "bb": np.zeros(9), "hr": np.zeros(9)}
    no_tto_lifts = {"k": np.zeros(3), "bb": np.zeros(3), "hr": np.zeros(3)}

    # Run all games in one batch
    result = simulate_game(
        pitcher_k_rate_samples=k_rate_samples,
        pitcher_bb_rate_samples=bb_rate_samples,
        pitcher_hr_rate_samples=hr_rate_samples,
        lineup_matchup_lifts=no_lifts,
        tto_lifts=no_tto_lifts,
        pitcher_ppa_adj=0.0,
        batter_ppa_adjs=np.zeros(9),
        exit_model=exit_model,
        pitcher_avg_pitches=pitcher_avg_pitches,
        babip_adj=babip_adj,
        n_sims=total_sims,
        random_seed=random_seed,
    )

    # Split games back into seasons and sum
    k_out = np.zeros(n_seasons, dtype=np.int32)
    bb_out = np.zeros(n_seasons, dtype=np.int32)
    h_out = np.zeros(n_seasons, dtype=np.int32)
    hr_out = np.zeros(n_seasons, dtype=np.int32)
    hbp_out = np.zeros(n_seasons, dtype=np.int32)
    bf_out = np.zeros(n_seasons, dtype=np.int32)
    outs_out = np.zeros(n_seasons, dtype=np.int32)
    runs_out = np.zeros(n_seasons, dtype=np.int32)
    pitches_out = np.zeros(n_seasons, dtype=np.int32)
    games_out = starts_per_season.copy()

    idx = 0
    for s in range(n_seasons):
        n = starts_per_season[s]
        end = idx + n
        k_out[s] = result.k_samples[idx:end].sum()
        bb_out[s] = result.bb_samples[idx:end].sum()
        h_out[s] = result.h_samples[idx:end].sum()
        hr_out[s] = result.hr_samples[idx:end].sum()
        hbp_out[s] = result.hbp_samples[idx:end].sum()
        bf_out[s] = result.bf_samples[idx:end].sum()
        outs_out[s] = result.outs_samples[idx:end].sum()
        runs_out[s] = result.runs_samples[idx:end].sum()
        pitches_out[s] = result.pitch_count_samples[idx:end].sum()
        idx = end

    # Starters: no saves or holds
    sv_out = np.zeros(n_seasons, dtype=np.int32)
    hld_out = np.zeros(n_seasons, dtype=np.int32)

    # Fantasy scoring
    fantasy = compute_season_pitcher_fantasy(
        k=k_out, bb=bb_out, h=h_out, hr=hr_out, hbp=hbp_out,
        outs=outs_out, runs=runs_out, sv=sv_out, hld=hld_out,
    )

    return SeasonSimResult(
        pitcher_id=pitcher_id,
        role="SP",
        k_season=k_out,
        bb_season=bb_out,
        h_season=h_out,
        hr_season=hr_out,
        hbp_season=hbp_out,
        bf_season=bf_out,
        outs_season=outs_out,
        runs_season=runs_out,
        pitches_season=pitches_out,
        games_season=games_out,
        sv_season=sv_out,
        hld_season=hld_out,
        dk_season=fantasy.dk_points,
        espn_season=fantasy.espn_points,
        n_seasons=n_seasons,
    )


# ===================================================================
# Reliever Season Simulation
# ===================================================================

def simulate_reliever_season(
    pitcher_id: int,
    k_rate_samples: np.ndarray,
    bb_rate_samples: np.ndarray,
    hr_rate_samples: np.ndarray,
    role: str,
    role_priors: RelieverRolePriors | None = None,
    babip_adj: float = 0.0,
    n_seasons: int = 200,
    random_seed: int = 42,
) -> SeasonSimResult:
    """Simulate full seasons for a reliever.

    Lightweight: draws BF from role prior, resolves PAs via multinomial
    (no ExitModel, no TTO, no pitch count model). Save/hold opportunities
    assigned as Bernoulli draws per appearance.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    k_rate_samples, bb_rate_samples, hr_rate_samples : np.ndarray
        Layer 1 posterior rate samples.
    role : str
        One of 'CL', 'SU', 'MR'.
    role_priors : RelieverRolePriors, optional
        Override default priors.
    n_seasons : int
        Number of season simulations.
    random_seed : int
        For reproducibility.

    Returns
    -------
    SeasonSimResult
    """
    from src.models.game_sim.fantasy_scoring import (
        compute_season_pitcher_fantasy,
    )
    from src.models.game_sim.pa_outcome_model import LEAGUE_HBP_RATE
    from src.models.game_sim.bip_model import BIPOutcomeModel

    rng = np.random.default_rng(random_seed)
    priors = role_priors or get_role_priors(role)
    bip_model = BIPOutcomeModel()

    # Resample rate posteriors to n_seasons
    def _resample(arr: np.ndarray) -> np.ndarray:
        idx = rng.choice(len(arr), size=n_seasons, replace=True)
        return arr[idx]

    k_rates = np.clip(_resample(k_rate_samples), 0.01, 0.60)
    bb_rates = np.clip(_resample(bb_rate_samples), 0.01, 0.30)
    hr_rates = np.clip(_resample(hr_rate_samples), 0.001, 0.10)
    hbp_rate = LEAGUE_HBP_RATE

    # BIP outcome probabilities (pitcher-specific BABIP adjustment)
    bip_probs = bip_model.get_adjusted_probs(babip_adj)  # [out, single, double, triple]

    # Initialize season accumulators
    k_out = np.zeros(n_seasons, dtype=np.int32)
    bb_out = np.zeros(n_seasons, dtype=np.int32)
    h_out = np.zeros(n_seasons, dtype=np.int32)
    hr_out = np.zeros(n_seasons, dtype=np.int32)
    hbp_out = np.zeros(n_seasons, dtype=np.int32)
    bf_out = np.zeros(n_seasons, dtype=np.int32)
    outs_out = np.zeros(n_seasons, dtype=np.int32)
    runs_out = np.zeros(n_seasons, dtype=np.int32)
    pitches_out = np.zeros(n_seasons, dtype=np.int32)
    sv_out = np.zeros(n_seasons, dtype=np.int32)
    hld_out = np.zeros(n_seasons, dtype=np.int32)

    # Draw number of games per season
    games_per_season = rng.normal(priors.games_mu, priors.games_sigma, size=n_seasons)
    games_per_season = np.clip(np.round(games_per_season), 1, MAX_RELIEVER_GAMES).astype(int)

    for s in range(n_seasons):
        n_games = games_per_season[s]
        k_rate = k_rates[s]
        bb_rate = bb_rates[s]
        hr_rate = hr_rates[s]

        # Draw BF per appearance
        bf_per_app = rng.normal(priors.bf_per_app_mu, priors.bf_per_app_sigma, size=n_games)
        bf_per_app = np.clip(np.round(bf_per_app), 1, 15).astype(int)
        total_bf = bf_per_app.sum()

        # Resolve all PAs at once via multinomial
        # Outcome probabilities: K, BB, HBP, HR, BIP
        bip_rate = max(1.0 - k_rate - bb_rate - hbp_rate - hr_rate, 0.05)
        pa_probs = np.array([k_rate, bb_rate, hbp_rate, hr_rate, bip_rate])
        pa_probs = pa_probs / pa_probs.sum()  # normalize

        outcomes = rng.multinomial(1, pa_probs, size=total_bf)
        # outcomes shape: (total_bf, 5) — columns: K, BB, HBP, HR, BIP
        n_k = outcomes[:, 0].sum()
        n_bb = outcomes[:, 1].sum()
        n_hbp = outcomes[:, 2].sum()
        n_hr = outcomes[:, 3].sum()
        n_bip = outcomes[:, 4].sum()

        # Resolve BIP outcomes
        if n_bip > 0:
            bip_outcomes = rng.multinomial(1, bip_probs, size=n_bip)
            # columns: out, single, double, triple
            n_bip_out = bip_outcomes[:, 0].sum()
            n_single = bip_outcomes[:, 1].sum()
            n_double = bip_outcomes[:, 2].sum()
            n_triple = bip_outcomes[:, 3].sum()
        else:
            n_bip_out = n_single = n_double = n_triple = 0

        n_h = n_single + n_double + n_triple + n_hr
        n_outs = n_k + n_bip_out

        # Simplified run scoring: HR scores 1 + 0.3 * runners_expected
        # Doubles/triples score ~0.7 runs on average, singles ~0.2
        n_runs = int(
            n_hr * 1.4
            + n_triple * 0.8
            + n_double * 0.5
            + n_single * 0.15
            + n_bb * 0.08
            + n_hbp * 0.08
        )

        # Rough pitch estimate: ~4 pitches/PA for relievers
        n_pitches = int(total_bf * 3.9)

        # Save / hold opportunities (Bernoulli per appearance)
        n_save_opp = rng.binomial(n_games, priors.save_opp_pct)
        n_sv = rng.binomial(n_save_opp, priors.save_conversion)
        # Blown saves = save_opp - saves
        n_bs = n_save_opp - n_sv

        n_hold_opp = rng.binomial(n_games - n_save_opp, priors.hold_opp_pct)
        n_hld = rng.binomial(n_hold_opp, priors.hold_conversion)

        k_out[s] = n_k
        bb_out[s] = n_bb
        h_out[s] = n_h
        hr_out[s] = n_hr
        hbp_out[s] = n_hbp
        bf_out[s] = total_bf
        outs_out[s] = n_outs
        runs_out[s] = n_runs
        pitches_out[s] = n_pitches
        sv_out[s] = n_sv
        hld_out[s] = n_hld

    # Fantasy scoring
    from src.models.game_sim.fantasy_scoring import compute_season_pitcher_fantasy
    fantasy = compute_season_pitcher_fantasy(
        k=k_out, bb=bb_out, h=h_out, hr=hr_out, hbp=hbp_out,
        outs=outs_out, runs=runs_out, sv=sv_out, hld=hld_out,
    )

    return SeasonSimResult(
        pitcher_id=pitcher_id,
        role=role,
        k_season=k_out,
        bb_season=bb_out,
        h_season=h_out,
        hr_season=hr_out,
        hbp_season=hbp_out,
        bf_season=bf_out,
        outs_season=outs_out,
        runs_season=runs_out,
        pitches_season=pitches_out,
        games_season=games_per_season,
        sv_season=sv_out,
        hld_season=hld_out,
        dk_season=fantasy.dk_points,
        espn_season=fantasy.espn_points,
        n_seasons=n_seasons,
    )


# ===================================================================
# Orchestrator
# ===================================================================

def simulate_all_pitchers(
    posteriors: dict[int, dict[str, np.ndarray]],
    roles: pd.DataFrame,
    exit_model: Any,
    starter_priors: pd.DataFrame | None = None,
    health_scores: pd.DataFrame | None = None,
    babip_adjs: dict[int, float] | None = None,
    n_seasons: int = 200,
    random_seed: int = 42,
) -> dict[int, SeasonSimResult]:
    """Run season simulations for all pitchers.

    Parameters
    ----------
    posteriors : dict[int, dict[str, np.ndarray]]
        Keyed by pitcher_id. Values have 'k_rate', 'bb_rate', 'hr_rate' arrays.
    roles : pd.DataFrame
        Output of ``classify_reliever_roles`` or similar. Must have
        pitcher_id and role columns. Pitchers not present default to SP
        if they have 'is_starter' flag, else MR.
    exit_model
        Trained ExitModel for starter simulations.
    starter_priors : pd.DataFrame, optional
        Pitcher-specific n_starts priors. Columns: pitcher_id, n_starts_mu,
        n_starts_sigma, avg_pitches.
    health_scores : pd.DataFrame, optional
        Health scores for games/sigma adjustment.
    babip_adjs : dict[int, float], optional
        Pitcher-specific BABIP adjustments (from BIPOutcomeModel.compute_pitcher_babip_adj).
        Positive = more hits on BIP than average.
    n_seasons : int
        Seasons per pitcher.
    random_seed : int

    Returns
    -------
    dict[int, SeasonSimResult]
    """
    rng_base = np.random.default_rng(random_seed)

    # Build role lookup
    role_lookup: dict[int, str] = {}
    if not roles.empty:
        for _, row in roles.iterrows():
            role_lookup[int(row["pitcher_id"])] = row["role"]

    # Build starter priors lookup
    sp_priors: dict[int, dict[str, float]] = {}
    if starter_priors is not None and not starter_priors.empty:
        for _, row in starter_priors.iterrows():
            sp_priors[int(row["pitcher_id"])] = {
                "n_starts_mu": float(row.get("n_starts_mu", POP_STARTER_N_STARTS)),
                "n_starts_sigma": float(row.get("n_starts_sigma", POP_STARTER_N_STARTS_SD)),
                "avg_pitches": float(row.get("avg_pitches", 88.0)),
            }

    # Health adjustment lookup
    health_lookup: dict[int, float] = {}
    if health_scores is not None and not health_scores.empty:
        for _, row in health_scores.iterrows():
            health_lookup[int(row["player_id"])] = float(row["health_score"])

    results: dict[int, SeasonSimResult] = {}
    n_sp = n_rp = 0

    for pid, post in posteriors.items():
        role = role_lookup.get(pid, "SP")  # default to SP if not in reliever roles
        seed = int(rng_base.integers(0, 2**31))

        k_arr = post["k_rate"]
        bb_arr = post["bb_rate"]
        hr_arr = post["hr_rate"]

        if role == "SP":
            # Get pitcher-specific starts prior
            priors = sp_priors.get(pid, {})
            mu = priors.get("n_starts_mu", POP_STARTER_N_STARTS)
            sigma = priors.get("n_starts_sigma", POP_STARTER_N_STARTS_SD)
            avg_pitches = priors.get("avg_pitches", 88.0)

            # Health adjustment
            h = health_lookup.get(pid, 0.85)
            games_mult = 0.75 + 0.27 * h
            sigma_mult = 1.50 - 0.65 * h
            mu *= games_mult
            sigma *= sigma_mult

            pitcher_babip = (babip_adjs or {}).get(pid, 0.0)
            results[pid] = simulate_starter_season(
                pitcher_id=pid,
                k_rate_samples=k_arr,
                bb_rate_samples=bb_arr,
                hr_rate_samples=hr_arr,
                exit_model=exit_model,
                n_starts_mu=mu,
                n_starts_sigma=sigma,
                pitcher_avg_pitches=avg_pitches,
                babip_adj=pitcher_babip,
                n_seasons=n_seasons,
                random_seed=seed,
            )
            n_sp += 1
        else:
            # Reliever: CL, SU, or MR
            role_prior = get_role_priors(role)

            # Health-adjusted games
            h = health_lookup.get(pid, 0.85)
            adjusted_priors = RelieverRolePriors(
                role=role_prior.role,
                games_mu=role_prior.games_mu * (0.75 + 0.27 * h),
                games_sigma=role_prior.games_sigma * (1.50 - 0.65 * h),
                bf_per_app_mu=role_prior.bf_per_app_mu,
                bf_per_app_sigma=role_prior.bf_per_app_sigma,
                save_opp_pct=role_prior.save_opp_pct,
                save_conversion=role_prior.save_conversion,
                hold_opp_pct=role_prior.hold_opp_pct,
                hold_conversion=role_prior.hold_conversion,
            )

            pitcher_babip = (babip_adjs or {}).get(pid, 0.0)
            results[pid] = simulate_reliever_season(
                pitcher_id=pid,
                k_rate_samples=k_arr,
                bb_rate_samples=bb_arr,
                hr_rate_samples=hr_arr,
                role=role,
                role_priors=adjusted_priors,
                babip_adj=pitcher_babip,
                n_seasons=n_seasons,
                random_seed=seed,
            )
            n_rp += 1

    logger.info(
        "Simulated %d pitchers (%d SP, %d RP) x %d seasons",
        len(results), n_sp, n_rp, n_seasons,
    )
    return results


# ===================================================================
# Output Conversion
# ===================================================================

def season_results_to_dataframe(
    results: dict[int, SeasonSimResult],
    pitcher_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Convert simulation results to a summary DataFrame.

    Parameters
    ----------
    results : dict[int, SeasonSimResult]
        Output of ``simulate_all_pitchers``.
    pitcher_names : dict[int, str], optional
        pitcher_id -> name mapping.

    Returns
    -------
    pd.DataFrame
        One row per pitcher with mean/median/sd/p10/p90/p2_5/p97_5
        for every stat, plus derived ERA and WHIP.
    """
    rows = []
    for pid, sim in results.items():
        row: dict[str, Any] = {
            "pitcher_id": pid,
            "pitcher_name": (pitcher_names or {}).get(pid, ""),
            "role": sim.role,
            "n_seasons": sim.n_seasons,
        }

        # Summary for each stat
        for stat_name, arr in [
            ("total_k", sim.k_season),
            ("total_bb", sim.bb_season),
            ("total_h", sim.h_season),
            ("total_hr", sim.hr_season),
            ("total_hbp", sim.hbp_season),
            ("total_bf", sim.bf_season),
            ("total_outs", sim.outs_season),
            ("total_runs", sim.runs_season),
            ("total_pitches", sim.pitches_season),
            ("total_games", sim.games_season),
            ("total_sv", sim.sv_season),
            ("total_hld", sim.hld_season),
            ("dk_season", sim.dk_season),
            ("espn_season", sim.espn_season),
        ]:
            summary = _stat_summary(arr)
            for k, v in summary.items():
                row[f"{stat_name}_{k}"] = v

        # Derived stats
        ip = sim.ip_season()
        ip_summary = _stat_summary(ip)
        for k, v in ip_summary.items():
            row[f"projected_ip_{k}"] = v

        era = sim.era_season()
        era_valid = era[~np.isnan(era)]
        if len(era_valid) > 0:
            era_summary = _stat_summary(np.clip(era_valid, 0, 15))
            for k, v in era_summary.items():
                row[f"projected_era_{k}"] = v

        whip = sim.whip_season()
        whip_valid = whip[~np.isnan(whip)]
        if len(whip_valid) > 0:
            whip_summary = _stat_summary(np.clip(whip_valid, 0, 5))
            for k, v in whip_summary.items():
                row[f"projected_whip_{k}"] = v

        fip_era = sim.fip_era_season()
        fip_valid = fip_era[~np.isnan(fip_era)]
        if len(fip_valid) > 0:
            fip_summary = _stat_summary(np.clip(fip_valid, 0, 15))
            for k, v in fip_summary.items():
                row[f"projected_fip_era_{k}"] = v

        rs = sim.runs_saved_season()
        rs_valid = rs[~np.isnan(rs)]
        if len(rs_valid) > 0:
            rs_summary = _stat_summary(rs_valid)
            for k, v in rs_summary.items():
                row[f"projected_runs_saved_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Converted %d pitcher season results to DataFrame", len(df))
    return df


# ===================================================================
# Hitter Season Simulation
# ===================================================================

@dataclass
class HitterSeasonSimResult:
    """Results from a hitter season-level simulation.

    All arrays have shape (n_seasons,).
    """

    batter_id: int
    k_season: np.ndarray
    bb_season: np.ndarray
    h_season: np.ndarray
    hr_season: np.ndarray
    single_season: np.ndarray
    double_season: np.ndarray
    triple_season: np.ndarray
    tb_season: np.ndarray
    r_season: np.ndarray
    rbi_season: np.ndarray
    hbp_season: np.ndarray
    sb_season: np.ndarray
    pa_season: np.ndarray
    games_season: np.ndarray
    dk_season: np.ndarray
    espn_season: np.ndarray
    n_seasons: int = 0

    # 2024 wOBA linear weights (FanGraphs)
    _WOBA_BB = 0.69
    _WOBA_HBP = 0.72
    _WOBA_1B = 0.88
    _WOBA_2B = 1.25
    _WOBA_3B = 1.58
    _WOBA_HR = 2.03
    _WOBA_SCALE = 1.21    # wOBA scale factor for wRC+ conversion
    _LG_WOBA = 0.310      # 2024 league wOBA
    _LG_R_PER_PA = 0.117  # 2024 league R/PA

    def avg_season(self) -> np.ndarray:
        """Batting average = H / AB (AB ~ PA - BB - HBP)."""
        ab = (self.pa_season - self.bb_season - self.hbp_season).clip(1)
        return self.h_season.astype(float) / ab

    def obp_season(self) -> np.ndarray:
        """On-base percentage."""
        return (self.h_season + self.bb_season + self.hbp_season).astype(float) / self.pa_season.clip(1)

    def slg_season(self) -> np.ndarray:
        """Slugging percentage."""
        ab = (self.pa_season - self.bb_season - self.hbp_season).clip(1)
        return self.tb_season.astype(float) / ab

    def woba_season(self) -> np.ndarray:
        """Weighted on-base average from sim's PA outcomes."""
        numerator = (
            self._WOBA_BB * self.bb_season.astype(float)
            + self._WOBA_HBP * self.hbp_season.astype(float)
            + self._WOBA_1B * self.single_season.astype(float)
            + self._WOBA_2B * self.double_season.astype(float)
            + self._WOBA_3B * self.triple_season.astype(float)
            + self._WOBA_HR * self.hr_season.astype(float)
        )
        return numerator / self.pa_season.clip(1).astype(float)

    def wrc_plus_season(self) -> np.ndarray:
        """wRC+ from sim's wOBA (100 = league average)."""
        woba = self.woba_season()
        wrc = (
            (woba - self._LG_WOBA) / self._WOBA_SCALE + self._LG_R_PER_PA
        ) / self._LG_R_PER_PA * 100
        return wrc

    def wraa_season(self) -> np.ndarray:
        """Weighted runs above average (counting stat version of wOBA)."""
        woba = self.woba_season()
        return ((woba - self._LG_WOBA) / self._WOBA_SCALE) * self.pa_season.astype(float)

    def summary(self) -> dict[str, dict[str, float]]:
        """Summary statistics for all stats."""
        stats = {}
        for name, arr in [
            ("k", self.k_season), ("bb", self.bb_season),
            ("h", self.h_season), ("hr", self.hr_season),
            ("single", self.single_season), ("double", self.double_season),
            ("triple", self.triple_season), ("tb", self.tb_season),
            ("r", self.r_season), ("rbi", self.rbi_season),
            ("hbp", self.hbp_season), ("sb", self.sb_season),
            ("pa", self.pa_season), ("games", self.games_season),
            ("dk", self.dk_season), ("espn", self.espn_season),
        ]:
            stats[name] = _stat_summary(arr)

        # Derived rates
        avg = self.avg_season()
        stats["avg"] = _stat_summary(np.clip(avg, 0, 1))
        obp = self.obp_season()
        stats["obp"] = _stat_summary(np.clip(obp, 0, 1))
        slg = self.slg_season()
        stats["slg"] = _stat_summary(np.clip(slg, 0, 4))
        stats["ops"] = _stat_summary(np.clip(obp + slg, 0, 5))
        stats["woba"] = _stat_summary(np.clip(self.woba_season(), 0, 1))
        stats["wrc_plus"] = _stat_summary(np.clip(self.wrc_plus_season(), 0, 250))
        stats["wraa"] = _stat_summary(self.wraa_season())

        return stats


def simulate_hitter_season(
    batter_id: int,
    k_rate_samples: np.ndarray,
    bb_rate_samples: np.ndarray,
    hr_rate_samples: np.ndarray,
    n_games_mu: float = 145.0,
    n_games_sigma: float = 26.0,
    batting_order: int = 5,
    babip_adj: float = 0.0,
    sb_rate: float = 0.0,
    sb_rate_sd: float = 0.0,
    n_seasons: int = 200,
    random_seed: int = 42,
) -> HitterSeasonSimResult:
    """Simulate full seasons for a hitter using the batter game simulator.

    Batches all games across all seasons into a single
    ``simulate_batter_game()`` call, then slices back into seasons.

    Parameters
    ----------
    batter_id : int
        Batter MLB ID.
    k_rate_samples, bb_rate_samples, hr_rate_samples : np.ndarray
        Layer 1 posterior rate samples.
    n_games_mu, n_games_sigma : float
        Projected games (mean, std) from PA model.
    batting_order : int
        Typical lineup slot (1-9).
    babip_adj : float
        Batter BABIP adjustment.
    sb_rate, sb_rate_sd : float
        Stolen bases per game (mean, std).
    n_seasons : int
        Number of season simulations.
    random_seed : int

    Returns
    -------
    HitterSeasonSimResult
    """
    from src.models.game_sim.batter_simulator import simulate_batter_game
    from src.models.game_sim.fantasy_scoring import compute_season_batter_fantasy

    rng = np.random.default_rng(random_seed)

    # Draw games per season
    games_per_season = rng.normal(n_games_mu, n_games_sigma, size=n_seasons)
    games_per_season = np.clip(np.round(games_per_season), 20, 162).astype(int)
    total_sims = int(games_per_season.sum())

    # Run all games in one batch (league-average opponent, no matchup lifts)
    result = simulate_batter_game(
        batter_k_rate_samples=k_rate_samples,
        batter_bb_rate_samples=bb_rate_samples,
        batter_hr_rate_samples=hr_rate_samples,
        batting_order=batting_order,
        starter_k_rate=0.226,   # league average
        starter_bb_rate=0.082,
        starter_hr_rate=0.031,
        starter_bf_mu=22.0,
        starter_bf_sigma=4.5,
        bullpen_k_rate=0.253,
        bullpen_bb_rate=0.084,
        bullpen_hr_rate=0.024,
        batter_babip_adj=babip_adj,
        n_sims=total_sims,
        random_seed=random_seed,
    )

    # Slice back into seasons and sum
    k_out = np.zeros(n_seasons, dtype=np.int32)
    bb_out = np.zeros(n_seasons, dtype=np.int32)
    h_out = np.zeros(n_seasons, dtype=np.int32)
    hr_out = np.zeros(n_seasons, dtype=np.int32)
    single_out = np.zeros(n_seasons, dtype=np.int32)
    double_out = np.zeros(n_seasons, dtype=np.int32)
    triple_out = np.zeros(n_seasons, dtype=np.int32)
    tb_out = np.zeros(n_seasons, dtype=np.int32)
    r_out = np.zeros(n_seasons, dtype=np.int32)
    rbi_out = np.zeros(n_seasons, dtype=np.int32)
    hbp_out = np.zeros(n_seasons, dtype=np.int32)
    pa_out = np.zeros(n_seasons, dtype=np.int32)

    idx = 0
    for s in range(n_seasons):
        n = games_per_season[s]
        end = idx + n
        k_out[s] = result.k_samples[idx:end].sum()
        bb_out[s] = result.bb_samples[idx:end].sum()
        h_out[s] = result.h_samples[idx:end].sum()
        hr_out[s] = result.hr_samples[idx:end].sum()
        single_out[s] = result.single_samples[idx:end].sum()
        double_out[s] = result.double_samples[idx:end].sum()
        triple_out[s] = result.triple_samples[idx:end].sum()
        tb_out[s] = result.tb_samples[idx:end].sum()
        r_out[s] = result.r_samples[idx:end].sum()
        rbi_out[s] = result.rbi_samples[idx:end].sum()
        hbp_out[s] = result.hbp_samples[idx:end].sum()
        pa_out[s] = result.pa_samples[idx:end].sum()
        idx = end

    # SB: rate x games (not in game sim)
    sb_draws = rng.normal(sb_rate, max(sb_rate_sd, sb_rate * 0.15 + 0.01), size=n_seasons)
    sb_out = np.clip(np.round(sb_draws * games_per_season), 0, 100).astype(np.int32)

    # Fantasy scoring
    fantasy = compute_season_batter_fantasy(
        k=k_out, bb=bb_out, single=single_out, double=double_out,
        triple=triple_out, hr=hr_out, rbi=rbi_out, r=r_out,
        hbp=hbp_out, sb=sb_out,
    )

    return HitterSeasonSimResult(
        batter_id=batter_id,
        k_season=k_out, bb_season=bb_out,
        h_season=h_out, hr_season=hr_out,
        single_season=single_out, double_season=double_out,
        triple_season=triple_out, tb_season=tb_out,
        r_season=r_out, rbi_season=rbi_out,
        hbp_season=hbp_out, sb_season=sb_out,
        pa_season=pa_out, games_season=games_per_season,
        dk_season=fantasy.dk_points, espn_season=fantasy.espn_points,
        n_seasons=n_seasons,
    )


def simulate_all_hitters(
    posteriors: dict[int, dict[str, np.ndarray]],
    pa_priors: pd.DataFrame,
    batting_orders: dict[int, int] | None = None,
    babip_adjs: dict[int, float] | None = None,
    sb_rates: dict[int, tuple[float, float]] | None = None,
    health_scores: pd.DataFrame | None = None,
    n_seasons: int = 200,
    random_seed: int = 42,
) -> dict[int, HitterSeasonSimResult]:
    """Run season simulations for all hitters.

    Parameters
    ----------
    posteriors : dict[int, dict[str, np.ndarray]]
        Keyed by batter_id. Values have 'k_rate', 'bb_rate', 'hr_rate'.
    pa_priors : pd.DataFrame
        From ``pa_model.compute_hitter_pa_priors()``.
        Columns: batter_id, projected_games, sigma_games.
    batting_orders : dict[int, int], optional
        batter_id -> typical lineup slot (1-9).
    babip_adjs : dict[int, float], optional
        batter_id -> BABIP adjustment.
    sb_rates : dict[int, tuple[float, float]], optional
        batter_id -> (sb_per_game_mean, sb_per_game_sd).
    health_scores : pd.DataFrame, optional
        Health scores for games/sigma adjustment.
    n_seasons : int
    random_seed : int

    Returns
    -------
    dict[int, HitterSeasonSimResult]
    """
    rng_base = np.random.default_rng(random_seed)

    # Build PA priors lookup
    pa_lookup: dict[int, tuple[float, float]] = {}
    if pa_priors is not None and not pa_priors.empty:
        for _, row in pa_priors.iterrows():
            pa_lookup[int(row["batter_id"])] = (
                float(row.get("projected_games", 145)),
                float(row.get("sigma_games", 26)),
            )

    # Health lookup
    health_lookup: dict[int, float] = {}
    if health_scores is not None and not health_scores.empty:
        for _, row in health_scores.iterrows():
            health_lookup[int(row["player_id"])] = float(row["health_score"])

    results: dict[int, HitterSeasonSimResult] = {}

    for bid, post in posteriors.items():
        seed = int(rng_base.integers(0, 2**31))

        # Games prior
        games_mu, games_sigma = pa_lookup.get(bid, (145.0, 26.0))

        # Health adjustment
        h = health_lookup.get(bid, 0.85)
        games_mu *= 0.75 + 0.27 * h
        games_sigma *= 1.50 - 0.65 * h

        # Batting order
        order = (batting_orders or {}).get(bid, 5)

        # BABIP
        babip = (babip_adjs or {}).get(bid, 0.0)

        # SB
        sb_mean, sb_sd = (sb_rates or {}).get(bid, (0.0, 0.0))

        results[bid] = simulate_hitter_season(
            batter_id=bid,
            k_rate_samples=post["k_rate"],
            bb_rate_samples=post["bb_rate"],
            hr_rate_samples=post["hr_rate"],
            n_games_mu=games_mu,
            n_games_sigma=games_sigma,
            batting_order=order,
            babip_adj=babip,
            sb_rate=sb_mean,
            sb_rate_sd=sb_sd,
            n_seasons=n_seasons,
            random_seed=seed,
        )

    logger.info("Simulated %d hitters x %d seasons", len(results), n_seasons)
    return results


def hitter_season_results_to_dataframe(
    results: dict[int, HitterSeasonSimResult],
    batter_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Convert hitter simulation results to a summary DataFrame."""
    rows = []
    for bid, sim in results.items():
        row: dict[str, Any] = {
            "batter_id": bid,
            "batter_name": (batter_names or {}).get(bid, ""),
            "n_seasons": sim.n_seasons,
        }

        for stat_name, arr in [
            ("total_k", sim.k_season), ("total_bb", sim.bb_season),
            ("total_h", sim.h_season), ("total_hr", sim.hr_season),
            ("total_1b", sim.single_season), ("total_2b", sim.double_season),
            ("total_3b", sim.triple_season), ("total_tb", sim.tb_season),
            ("total_r", sim.r_season), ("total_rbi", sim.rbi_season),
            ("total_hbp", sim.hbp_season), ("total_sb", sim.sb_season),
            ("total_pa", sim.pa_season), ("total_games", sim.games_season),
            ("dk_season", sim.dk_season), ("espn_season", sim.espn_season),
        ]:
            summary = _stat_summary(arr)
            for k, v in summary.items():
                row[f"{stat_name}_{k}"] = v

        # Derived rate distributions
        for stat_name, arr, lo, hi in [
            ("projected_avg", sim.avg_season(), 0, 1),
            ("projected_obp", sim.obp_season(), 0, 1),
            ("projected_slg", sim.slg_season(), 0, 4),
            ("projected_ops", sim.obp_season() + sim.slg_season(), 0, 5),
            ("projected_woba", sim.woba_season(), 0, 1),
            ("projected_wrc_plus", sim.wrc_plus_season(), 0, 250),
            ("projected_wraa", sim.wraa_season(), -50, 80),
        ]:
            summary = _stat_summary(np.clip(arr, lo, hi))
            for k, v in summary.items():
                row[f"{stat_name}_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Converted %d hitter season results to DataFrame", len(df))
    return df
