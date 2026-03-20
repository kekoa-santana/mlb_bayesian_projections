"""
Counting stat projections: rate posteriors x playing time distributions.

Hitter stats: Total Ks, Total BBs, Total HRs, Total SBs
Pitcher stats: Total Ks, Total BBs, Total Outs

Each produces a posterior distribution over the counting stat via
Monte Carlo: draw rate samples from Bayesian posteriors, draw playing
time samples from shrinkage models, multiply.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# SB era adjustment: 2023 rule change increased SB rates ~80%
# Ratio of league SB/player post-2023 vs pre-2023
SB_ERA_FACTOR_PRE_2023 = 1.8  # inflate pre-2023 SB rates to modern era


@dataclass
class CountingStat:
    """Configuration for a counting stat projection."""
    name: str           # e.g. "total_k"
    display: str        # e.g. "Strikeouts"
    rate_col: str       # rate column name (e.g. "k_rate")
    opportunity: str    # "pa", "games", or "bf"
    bayesian: bool      # True = use Bayesian posterior samples for rate


HITTER_COUNTING_STATS = {
    "total_k":  CountingStat("total_k",  "Strikeouts",    "k_rate",       "pa",    True),
    "total_bb": CountingStat("total_bb", "Walks",         "bb_rate",      "pa",    True),
    "total_hr": CountingStat("total_hr", "Home Runs",     "hr_rate",      "pa",    False),
    "total_sb": CountingStat("total_sb", "Stolen Bases",  "sb_per_game",  "games", False),
}

PITCHER_COUNTING_STATS = {
    "total_k":    CountingStat("total_k",    "Strikeouts",  "k_rate",      "bf", True),
    "total_bb":   CountingStat("total_bb",   "Walks",       "bb_rate",     "bf", True),
    "total_outs": CountingStat("total_outs", "Outs",        "outs_per_bf", "bf", False),
    "total_er":   CountingStat("total_er",   "Earned Runs", "er_per_bf",   "bf", False),
}


def _compute_stat_summary(samples: np.ndarray) -> dict[str, float]:
    """Compute summary statistics from Monte Carlo samples."""
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "sd": float(np.std(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p90": float(np.percentile(samples, 90)),
        "p2_5": float(np.percentile(samples, 2.5)),
        "p97_5": float(np.percentile(samples, 97.5)),
    }


def _shrinkage_rate(
    player_hist: pd.DataFrame,
    rate_col: str,
    weight_col: str = "pa",
    pop_mean: float | None = None,
    shrinkage_k: float = 3.0,
) -> tuple[float, float]:
    """Compute shrinkage rate estimate with Marcel-style weighting.

    Returns (mean, std) for the rate.
    """
    if player_hist.empty:
        return (pop_mean or 0.0, 0.05)

    weights = [5, 4, 3][:len(player_hist)]
    total_w = sum(weights)

    vals = player_hist[rate_col].values
    vol = player_hist[weight_col].values

    # Weighted mean
    weighted = sum(v * w for v, w in zip(vals, weights)) / total_w

    # Population regression
    if pop_mean is not None:
        n = len(player_hist)
        rel = n / (n + shrinkage_k)
        mean = rel * weighted + (1 - rel) * pop_mean
    else:
        mean = weighted

    # Std from observed variation (floor at 15% CV for wider intervals)
    if len(vals) >= 2:
        std = float(np.std(vals, ddof=1))
    else:
        std = abs(mean) * 0.3  # 30% CV default

    floor = abs(mean) * 0.15  # minimum 15% CV
    return (mean, max(std, floor, 0.001))


def project_hitter_counting(
    rate_model_results: dict[str, dict[str, Any]],
    pa_priors: pd.DataFrame,
    hitter_extended: pd.DataFrame,
    from_season: int,
    n_draws: int = 4000,
    min_pa: int = 200,
    random_seed: int = 42,
    park_factors: pd.DataFrame | None = None,
    hitter_venues: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project counting stats for all hitters.

    Parameters
    ----------
    rate_model_results : dict
        Output of hitter_projections.fit_all_models (K%, BB% traces).
    pa_priors : pd.DataFrame
        Output of pa_model.compute_hitter_pa_priors.
    hitter_extended : pd.DataFrame
        Multi-season extended data with sb, games, sprint_speed.
    from_season : int
        Project from this season forward.
    n_draws : int
        Monte Carlo samples.
    min_pa : int
        Minimum PA in from_season to include.
    random_seed : int
    park_factors : pd.DataFrame, optional
        HR park factors from queries.get_park_factors.
        Columns: venue_id, batter_stand, hr_pf.
    hitter_venues : pd.DataFrame, optional
        Hitter-to-venue mapping from queries.get_hitter_team_venue.
        Columns: batter_id, team_id, team_name, venue_id.

    Returns
    -------
    pd.DataFrame
        One row per hitter with counting stat summaries.
    """
    from src.models.hitter_model import STAT_CONFIGS, extract_rate_samples
    from src.models.pa_model import draw_pa_samples

    rng = np.random.default_rng(random_seed)

    # Get eligible players
    season_df = hitter_extended[
        (hitter_extended["season"] == from_season)
        & (hitter_extended["pa"] >= min_pa)
    ]

    if season_df.empty:
        logger.warning("No hitters found in season %d with >= %d PA", from_season, min_pa)
        return pd.DataFrame()

    # Population means for shrinkage rates
    pop_hr_rate = float(season_df["hr_rate"].mean())
    pop_sb_rate = float(season_df["sb_per_game"].mean()) if "sb_per_game" in season_df.columns else 0.3

    # Build park factor lookup: (batter_id) -> hr_pf
    # HR park factor adjusts for ~half the games played at home
    pf_lookup: dict[int, float] = {}
    if park_factors is not None and hitter_venues is not None and not park_factors.empty:
        # Build venue→{stand: pf} lookup
        venue_pf: dict[tuple[int, str], float] = {}
        for _, pf_row in park_factors.iterrows():
            venue_pf[(int(pf_row["venue_id"]), pf_row["batter_stand"])] = float(pf_row["hr_pf"])

        for _, hv_row in hitter_venues.iterrows():
            bid = int(hv_row["batter_id"])
            vid = int(hv_row["venue_id"])
            stand = str(hv_row.get("bat_side", "R"))

            if stand == "S":
                # Switch hitter: average L and R park factors
                pf_l = venue_pf.get((vid, "L"), 1.0)
                pf_r = venue_pf.get((vid, "R"), 1.0)
                raw_pf = (pf_l + pf_r) / 2
            else:
                raw_pf = venue_pf.get((vid, stand), 1.0)

            # Half-weight: ~half games at home, half on the road (road = neutral ~1.0)
            pf_lookup[bid] = 0.5 * raw_pf + 0.5 * 1.0

        logger.info(
            "Park factors applied: %d hitters, mean PF=%.3f, range=[%.3f, %.3f]",
            len(pf_lookup),
            np.mean(list(pf_lookup.values())) if pf_lookup else 1.0,
            min(pf_lookup.values()) if pf_lookup else 1.0,
            max(pf_lookup.values()) if pf_lookup else 1.0,
        )

    results = []
    for _, player in season_df.iterrows():
        bid = int(player["batter_id"])
        row = {
            "batter_id": bid,
            "batter_name": player["batter_name"],
            "age": player["age"],
            "season": from_season,
            "actual_pa": int(player["pa"]),
            "actual_games": int(player["games"]),
        }

        # Draw PA samples
        pa_row = pa_priors[pa_priors["batter_id"] == bid]
        if pa_row.empty:
            pa_samples = np.full(n_draws, int(player["pa"]))
            games_samples = np.full(n_draws, int(player["games"]))
        else:
            pr = pa_row.iloc[0]
            pa_samples = draw_pa_samples(
                pr["projected_games"], pr["sigma_games"],
                pr["projected_pa_per_game"], pr["sigma_pa_rate"],
                n_draws, rng=rng,
            )
            games_samples = pa_samples / max(pr["projected_pa_per_game"], 3.0)
            # Carry health columns from PA priors
            if "health_score" in pa_priors.columns:
                row["health_score"] = pr.get("health_score", None)
                row["health_label"] = pr.get("health_label", "")

        row["projected_pa_mean"] = float(np.mean(pa_samples))
        row["projected_games_mean"] = float(np.mean(games_samples))

        # For each counting stat
        for stat_name, cfg in HITTER_COUNTING_STATS.items():
            if cfg.bayesian and cfg.rate_col.replace("_rate", "") in rate_model_results:
                # Draw rate from Bayesian posterior
                stat_key = cfg.rate_col.replace("_rate", "")
                if stat_key == "k":
                    stat_key = "k_rate"
                elif stat_key == "bb":
                    stat_key = "bb_rate"
                else:
                    stat_key = cfg.rate_col

                if stat_key in rate_model_results:
                    res = rate_model_results[stat_key]
                    try:
                        rate_samples = extract_rate_samples(
                            res["trace"], res["data"], bid, from_season,
                            project_forward=True, random_seed=random_seed,
                        )
                        # Ensure we have enough samples
                        if len(rate_samples) < n_draws:
                            rate_samples = rng.choice(rate_samples, size=n_draws, replace=True)
                        else:
                            rate_samples = rate_samples[:n_draws]
                    except (ValueError, KeyError):
                        rate_samples = np.full(n_draws, float(player.get(cfg.rate_col, 0)))
                else:
                    rate_samples = np.full(n_draws, float(player.get(cfg.rate_col, 0)))
            else:
                # Use shrinkage rate estimate
                player_hist = hitter_extended[
                    (hitter_extended["batter_id"] == bid)
                    & (hitter_extended["season"] <= from_season)
                ].sort_values("season", ascending=False)

                if cfg.name == "total_hr":
                    mean_r, std_r = _shrinkage_rate(
                        player_hist, "hr_rate", "pa", pop_hr_rate, shrinkage_k=4.0
                    )
                elif cfg.name == "total_sb":
                    # Era-adjust pre-2023 SB rates
                    hist = player_hist.copy()
                    if "sb_per_game" in hist.columns:
                        hist.loc[hist["season"] < 2023, "sb_per_game"] *= SB_ERA_FACTOR_PRE_2023
                    mean_r, std_r = _shrinkage_rate(
                        hist, "sb_per_game", "games", pop_sb_rate, shrinkage_k=3.0
                    )
                else:
                    mean_r, std_r = _shrinkage_rate(
                        player_hist, cfg.rate_col, "pa"
                    )

                rate_samples = rng.normal(mean_r, std_r, size=n_draws)
                rate_samples = np.clip(rate_samples, 0, 1)

            # Apply park factor to HR rate
            if cfg.name == "total_hr" and bid in pf_lookup:
                rate_samples = rate_samples * pf_lookup[bid]

            # Multiply rate x opportunity
            if cfg.opportunity == "pa":
                count_samples = np.round(rate_samples * pa_samples).astype(int)
            elif cfg.opportunity == "games":
                count_samples = np.round(rate_samples * games_samples).astype(int)
            else:
                count_samples = np.round(rate_samples * pa_samples).astype(int)

            count_samples = np.clip(count_samples, 0, 999)
            summary = _compute_stat_summary(count_samples)
            for k, v in summary.items():
                row[f"{stat_name}_{k}"] = v

            # Store park factor for HR
            if cfg.name == "total_hr":
                row["hr_park_factor"] = pf_lookup.get(bid, 1.0)

            # Store actual for backtest comparison
            if cfg.name == "total_k":
                row["actual_k"] = int(player.get("k", 0))
            elif cfg.name == "total_bb":
                row["actual_bb"] = int(player.get("bb", 0))
            elif cfg.name == "total_hr":
                row["actual_hr"] = int(player.get("hr", 0))
            elif cfg.name == "total_sb":
                row["actual_sb"] = int(player.get("sb", 0))

        results.append(row)

    result_df = pd.DataFrame(results)
    logger.info("Projected counting stats for %d hitters from %d", len(result_df), from_season)
    return result_df


def project_pitcher_counting(
    rate_model_results: dict[str, dict[str, Any]],
    pitcher_extended: pd.DataFrame,
    from_season: int,
    n_draws: int = 4000,
    min_bf: int = 200,
    random_seed: int = 42,
    health_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project counting stats for all pitchers.

    Parameters
    ----------
    rate_model_results : dict
        Output of pitcher_projections.fit_all_models (K%, BB% traces).
    pitcher_extended : pd.DataFrame
        Multi-season extended pitcher data with outs, games, bf.
    from_season : int
        Project from this season forward.
    n_draws : int
        Monte Carlo samples.
    min_bf : int
        Minimum BF in from_season to include.
    random_seed : int
    health_scores : pd.DataFrame, optional
        Output of health_score.compute_health_scores.
        When provided, adjusts projected games and sigma by health.

    Returns
    -------
    pd.DataFrame
        One row per pitcher with counting stat summaries.
    """
    from src.models.pitcher_model import PITCHER_STAT_CONFIGS, extract_rate_samples
    from src.models.bf_model import draw_bf_samples

    rng = np.random.default_rng(random_seed)

    # Ensure er_per_bf exists (needed for total_er stat)
    if "earned_runs" in pitcher_extended.columns and "er_per_bf" not in pitcher_extended.columns:
        pitcher_extended = pitcher_extended.copy()
        pitcher_extended["er_per_bf"] = (
            pitcher_extended["earned_runs"]
            / pitcher_extended["batters_faced"].replace(0, float("nan"))
        ).round(4)

    season_df = pitcher_extended[
        (pitcher_extended["season"] == from_season)
        & (pitcher_extended["batters_faced"] >= min_bf)
    ]

    if season_df.empty:
        logger.warning("No pitchers found in season %d with >= %d BF", from_season, min_bf)
        return pd.DataFrame()

    pop_outs_per_bf = float(season_df["outs_per_bf"].mean()) if "outs_per_bf" in season_df.columns else 0.70
    pop_er_per_bf = float(season_df["er_per_bf"].mean()) if "er_per_bf" in season_df.columns else 0.11

    results = []
    for _, player in season_df.iterrows():
        pid = int(player["pitcher_id"])
        is_starter = bool(player.get("is_starter", 0))

        row = {
            "pitcher_id": pid,
            "pitcher_name": player.get("pitcher_name", ""),
            "age": player.get("age", 28),
            "season": from_season,
            "is_starter": int(is_starter),
            "actual_bf": int(player["batters_faced"]),
            "actual_games": int(player["games"]),
            "actual_outs": int(player.get("outs", 0)),
        }

        # Project BF: games * BF/game
        player_hist = pitcher_extended[
            (pitcher_extended["pitcher_id"] == pid)
            & (pitcher_extended["season"] <= from_season)
        ].sort_values("season", ascending=False)

        # Games projection (shrinkage)
        if is_starter:
            pop_games = 30.0
            pop_games_sd = 6.0
        else:
            pop_games = 55.0
            pop_games_sd = 17.0

        if len(player_hist) > 0:
            recent_games = player_hist.head(3)
            weights = [5, 4, 3][:len(recent_games)]
            # 2020 adjustment
            adj_games = recent_games["games"].values.copy().astype(float)
            seasons = recent_games["season"].values
            for i, s in enumerate(seasons):
                if s == 2020:
                    adj_games[i] = min(adj_games[i] * (162 / 60), 162 if is_starter else 80)

            weighted_games = sum(g * w for g, w in zip(adj_games, weights)) / sum(weights)
            rel = len(recent_games) / (len(recent_games) + 1.5)
            proj_games = rel * weighted_games + (1 - rel) * pop_games
        else:
            proj_games = pop_games

        # Health/durability adjustment
        h_score = None
        h_label = ""
        if health_scores is not None and not health_scores.empty:
            h_row = health_scores[health_scores["player_id"] == pid]
            if not h_row.empty:
                h_score = float(h_row.iloc[0]["health_score"])
                h_label = str(h_row.iloc[0]["health_label"])
                games_mult = 0.75 + 0.27 * h_score
                sigma_mult = 1.50 - 0.65 * h_score
                proj_games *= games_mult
                pop_games_sd *= sigma_mult

        # BF/game projection
        if len(player_hist) > 0:
            bf_per_game = (player_hist["batters_faced"] / player_hist["games"].replace(0, np.nan)).mean()
            bf_per_game = max(bf_per_game, 3.0)
        else:
            bf_per_game = 22.0 if is_starter else 4.5

        bf_per_game_sd = 4.5 if is_starter else 1.8

        # Draw game samples
        games_samples = rng.normal(proj_games, pop_games_sd, size=n_draws)
        games_samples = np.clip(np.round(games_samples), 1, 162 if is_starter else 85).astype(int)

        # Draw BF/game samples
        bf_game_samples = rng.normal(bf_per_game, bf_per_game_sd, size=n_draws)
        bf_game_samples = np.clip(bf_game_samples, 3, 35)

        # Total BF
        bf_samples = np.round(games_samples * bf_game_samples).astype(int)
        bf_samples = np.clip(bf_samples, 10, 900)

        if h_score is not None:
            row["health_score"] = round(h_score, 4)
            row["health_label"] = h_label

        row["projected_bf_mean"] = float(np.mean(bf_samples))
        row["projected_games_mean"] = float(np.mean(games_samples))

        # For each counting stat
        _saved_counts: dict[str, np.ndarray] = {}
        for stat_name, cfg in PITCHER_COUNTING_STATS.items():
            if cfg.bayesian and cfg.rate_col in rate_model_results:
                res = rate_model_results[cfg.rate_col]
                try:
                    rate_samples = extract_rate_samples(
                        res["trace"], res["data"], pid, from_season,
                        project_forward=True, random_seed=random_seed,
                    )
                    if len(rate_samples) < n_draws:
                        rate_samples = rng.choice(rate_samples, size=n_draws, replace=True)
                    else:
                        rate_samples = rate_samples[:n_draws]
                except (ValueError, KeyError):
                    rate_samples = np.full(n_draws, float(player.get(cfg.rate_col, 0)))
            else:
                # Shrinkage for outs_per_bf / er_per_bf
                pop_mean = pop_outs_per_bf if cfg.name == "total_outs" else pop_er_per_bf
                if cfg.rate_col in player_hist.columns:
                    mean_r, std_r = _shrinkage_rate(
                        player_hist, cfg.rate_col, "batters_faced",
                        pop_mean, shrinkage_k=3.0
                    )
                else:
                    mean_r, std_r = pop_mean, pop_mean * 0.30
                rate_samples = rng.normal(mean_r, std_r, size=n_draws)
                rate_samples = np.clip(rate_samples, 0, 1)

            # Multiply rate x BF
            count_samples = np.round(rate_samples * bf_samples).astype(int)
            count_samples = np.clip(count_samples, 0, 999)
            _saved_counts[stat_name] = count_samples

            summary = _compute_stat_summary(count_samples)
            for k, v in summary.items():
                row[f"{stat_name}_{k}"] = v

            # Actuals for backtest
            if cfg.name == "total_k":
                row["actual_k"] = int(player.get("k", 0))
            elif cfg.name == "total_bb":
                row["actual_bb"] = int(player.get("bb", 0))
            elif cfg.name == "total_outs":
                row["actual_outs"] = int(player.get("outs", 0))
            elif cfg.name == "total_er":
                row["actual_er"] = int(player.get("earned_runs", 0))

        # Derive IP and ERA from outs and ER projections
        if "total_outs" in _saved_counts and "total_er" in _saved_counts:
            ip_samples = _saved_counts["total_outs"].astype(float) / 3.0
            er_samples = _saved_counts["total_er"].astype(float)

            ip_summary = _compute_stat_summary(ip_samples)
            for k, v in ip_summary.items():
                row[f"projected_ip_{k}"] = v

            safe_ip = np.where(ip_samples > 0, ip_samples, np.nan)
            era_samples = er_samples / safe_ip * 9.0
            era_valid = era_samples[~np.isnan(era_samples)]
            if len(era_valid) > 0:
                era_summary = _compute_stat_summary(np.clip(era_valid, 0, 15))
                for k, v in era_summary.items():
                    row[f"projected_era_{k}"] = v

        row["actual_ip"] = float(player.get("ip", 0))

        results.append(row)

    result_df = pd.DataFrame(results)
    logger.info("Projected counting stats for %d pitchers from %d", len(result_df), from_season)
    return result_df


def project_pitcher_counting_sim(
    posteriors: dict[int, dict[str, np.ndarray]],
    roles: pd.DataFrame,
    exit_model: "Any",
    starter_priors: pd.DataFrame | None = None,
    health_scores: pd.DataFrame | None = None,
    babip_adjs: dict[int, float] | None = None,
    pitcher_names: dict[int, str] | None = None,
    n_seasons: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sim-based pitcher counting stat projections.

    Uses the PA-by-PA game simulator for starters and lightweight
    vectorized resolution for relievers. Produces correlated joint
    distributions over all counting stats + fantasy scoring.

    Parameters
    ----------
    posteriors : dict[int, dict[str, np.ndarray]]
        Keyed by pitcher_id. Values have 'k_rate', 'bb_rate', 'hr_rate'.
    roles : pd.DataFrame
        Reliever role classification (pitcher_id, role).
    exit_model
        Trained ExitModel.
    starter_priors : pd.DataFrame, optional
        Columns: pitcher_id, n_starts_mu, n_starts_sigma, avg_pitches.
    health_scores : pd.DataFrame, optional
        Health scores for games/sigma adjustment.
    babip_adjs : dict[int, float], optional
        Pitcher-specific BABIP adjustments (shrinkage-regressed).
    pitcher_names : dict[int, str], optional
        pitcher_id -> name mapping.
    n_seasons : int
        Monte Carlo seasons per pitcher.
    random_seed : int

    Returns
    -------
    pd.DataFrame
        One row per pitcher with counting stat summaries, plus
        H, pitches, SV, HLD, ERA, WHIP, FIP-ERA, DK, ESPN columns.
    """
    from src.models.season_simulator import (
        simulate_all_pitchers,
        season_results_to_dataframe,
    )

    results = simulate_all_pitchers(
        posteriors=posteriors,
        roles=roles,
        exit_model=exit_model,
        starter_priors=starter_priors,
        health_scores=health_scores,
        babip_adjs=babip_adjs,
        n_seasons=n_seasons,
        random_seed=random_seed,
    )

    df = season_results_to_dataframe(results, pitcher_names=pitcher_names)
    logger.info("Sim-based pitcher projections: %d pitchers", len(df))
    return df


def project_hitter_counting_sim(
    posteriors: dict[int, dict[str, np.ndarray]],
    pa_priors: pd.DataFrame,
    batting_orders: dict[int, int] | None = None,
    babip_adjs: dict[int, float] | None = None,
    sb_rates: dict[int, tuple[float, float]] | None = None,
    health_scores: pd.DataFrame | None = None,
    batter_names: dict[int, str] | None = None,
    n_seasons: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sim-based hitter counting stat projections.

    Uses the batter game simulator to produce correlated joint
    distributions over K, BB, H, HR, 1B, 2B, 3B, TB, R, RBI, SB
    plus DK/ESPN fantasy scoring.

    Parameters
    ----------
    posteriors : dict[int, dict[str, np.ndarray]]
        Keyed by batter_id. Values have 'k_rate', 'bb_rate', 'hr_rate'.
    pa_priors : pd.DataFrame
        From ``pa_model.compute_hitter_pa_priors()``.
    batting_orders : dict[int, int], optional
        batter_id -> typical lineup slot (1-9).
    babip_adjs : dict[int, float], optional
    sb_rates : dict[int, tuple[float, float]], optional
        batter_id -> (sb_per_game_mean, sb_per_game_sd).
    health_scores : pd.DataFrame, optional
    batter_names : dict[int, str], optional
    n_seasons : int
    random_seed : int

    Returns
    -------
    pd.DataFrame
        One row per hitter with counting stat summaries, hit breakdown,
        R, RBI, SB, AVG, OBP, SLG, OPS, DK, ESPN.
    """
    from src.models.season_simulator import (
        simulate_all_hitters,
        hitter_season_results_to_dataframe,
    )

    results = simulate_all_hitters(
        posteriors=posteriors,
        pa_priors=pa_priors,
        batting_orders=batting_orders,
        babip_adjs=babip_adjs,
        sb_rates=sb_rates,
        health_scores=health_scores,
        n_seasons=n_seasons,
        random_seed=random_seed,
    )

    df = hitter_season_results_to_dataframe(results, batter_names=batter_names)
    logger.info("Sim-based hitter projections: %d hitters", len(df))
    return df


def marcel_counting_hitter(
    hitter_extended: pd.DataFrame,
    from_season: int,
    min_pa: int = 200,
) -> pd.DataFrame:
    """Marcel baseline for hitter counting stats.

    Marcel counting = Marcel rate x Marcel PA.
    Marcel PA = weighted PA (5/4/3) * 0.90 regression * age factor.

    Parameters
    ----------
    hitter_extended : pd.DataFrame
        Multi-season stacked data.
    from_season : int
        Season being projected.
    min_pa : int
        Minimum PA in from_season.

    Returns
    -------
    pd.DataFrame
        Marcel projections with columns: batter_id, marcel_k, marcel_bb,
        marcel_hr, marcel_sb, marcel_pa.
    """
    season_df = hitter_extended[
        (hitter_extended["season"] == from_season)
        & (hitter_extended["pa"] >= min_pa)
    ]

    results = []
    for _, player in season_df.iterrows():
        bid = int(player["batter_id"])
        age = player.get("age", 28)

        player_hist = hitter_extended[
            (hitter_extended["batter_id"] == bid)
            & (hitter_extended["season"] <= from_season)
        ].sort_values("season", ascending=False).head(3)

        if player_hist.empty:
            continue

        weights = [5, 4, 3][:len(player_hist)]
        total_w = sum(weights)

        # Marcel PA: weighted PA * 0.90 regression
        adj_pa = player_hist["pa"].values.copy().astype(float)
        for i, s in enumerate(player_hist["season"].values):
            if s == 2020:
                adj_pa[i] *= (162 / 60)

        marcel_pa = sum(p * w for p, w in zip(adj_pa, weights)) / total_w * 0.90

        # Age factor for PA
        if age >= 35:
            marcel_pa *= 0.85
        elif age >= 33:
            marcel_pa *= 0.92

        # Marcel rates (weighted average + regression toward league mean)
        league_k = float(hitter_extended[hitter_extended["season"] == from_season]["k_rate"].mean())
        league_bb = float(hitter_extended[hitter_extended["season"] == from_season]["bb_rate"].mean())
        league_hr = float(hitter_extended[hitter_extended["season"] == from_season]["hr_rate"].mean())

        def _marcel_rate(hist, col, league_mean):
            vals = hist[col].values
            w = weights[:len(vals)]
            weighted = sum(v * wt for v, wt in zip(vals, w)) / sum(w)
            # Regress 20% toward league mean
            return 0.80 * weighted + 0.20 * league_mean

        m_k_rate = _marcel_rate(player_hist, "k_rate", league_k)
        m_bb_rate = _marcel_rate(player_hist, "bb_rate", league_bb)
        m_hr_rate = _marcel_rate(player_hist, "hr_rate", league_hr)

        # SB: weighted average with era adjustment
        if "sb_per_game" in player_hist.columns:
            sb_hist = player_hist.copy()
            sb_hist.loc[sb_hist["season"] < 2023, "sb_per_game"] *= SB_ERA_FACTOR_PRE_2023
            league_sb = float(hitter_extended[hitter_extended["season"] == from_season]["sb_per_game"].mean())
            m_sb_rate = _marcel_rate(sb_hist, "sb_per_game", league_sb)
            marcel_games = marcel_pa / 3.85
            m_sb = m_sb_rate * marcel_games
        else:
            m_sb = 0

        results.append({
            "batter_id": bid,
            "marcel_pa": int(marcel_pa),
            "marcel_k": int(m_k_rate * marcel_pa),
            "marcel_bb": int(m_bb_rate * marcel_pa),
            "marcel_hr": int(m_hr_rate * marcel_pa),
            "marcel_sb": int(m_sb),
        })

    return pd.DataFrame(results)


def marcel_counting_pitcher(
    pitcher_extended: pd.DataFrame,
    from_season: int,
    min_bf: int = 200,
) -> pd.DataFrame:
    """Marcel baseline for pitcher counting stats.

    Returns
    -------
    pd.DataFrame
        Marcel projections: pitcher_id, marcel_k, marcel_bb,
        marcel_outs, marcel_bf, marcel_earned_runs.
    """
    # Ensure er_per_bf exists
    if "earned_runs" in pitcher_extended.columns and "er_per_bf" not in pitcher_extended.columns:
        pitcher_extended = pitcher_extended.copy()
        pitcher_extended["er_per_bf"] = (
            pitcher_extended["earned_runs"]
            / pitcher_extended["batters_faced"].replace(0, float("nan"))
        ).round(4)

    season_df = pitcher_extended[
        (pitcher_extended["season"] == from_season)
        & (pitcher_extended["batters_faced"] >= min_bf)
    ]

    results = []
    for _, player in season_df.iterrows():
        pid = int(player["pitcher_id"])

        player_hist = pitcher_extended[
            (pitcher_extended["pitcher_id"] == pid)
            & (pitcher_extended["season"] <= from_season)
        ].sort_values("season", ascending=False).head(3)

        if player_hist.empty:
            continue

        weights = [5, 4, 3][:len(player_hist)]
        total_w = sum(weights)

        # Marcel BF
        adj_bf = player_hist["batters_faced"].values.copy().astype(float)
        for i, s in enumerate(player_hist["season"].values):
            if s == 2020:
                adj_bf[i] *= (162 / 60)
        marcel_bf = sum(b * w for b, w in zip(adj_bf, weights)) / total_w * 0.90

        # Marcel rates
        league_k = float(pitcher_extended[pitcher_extended["season"] == from_season]["k_rate"].mean())
        league_bb = float(pitcher_extended[pitcher_extended["season"] == from_season]["bb_rate"].mean())
        league_outs = float(pitcher_extended[pitcher_extended["season"] == from_season]["outs_per_bf"].mean())

        def _marcel_rate(hist, col, league_mean):
            vals = hist[col].dropna().values
            if len(vals) == 0:
                return league_mean
            w = weights[:len(vals)]
            weighted = sum(v * wt for v, wt in zip(vals, w)) / sum(w)
            return 0.80 * weighted + 0.20 * league_mean

        m_k_rate = _marcel_rate(player_hist, "k_rate", league_k)
        m_bb_rate = _marcel_rate(player_hist, "bb_rate", league_bb)
        m_outs_rate = _marcel_rate(player_hist, "outs_per_bf", league_outs)

        result_row = {
            "pitcher_id": pid,
            "marcel_bf": int(marcel_bf),
            "marcel_k": int(m_k_rate * marcel_bf),
            "marcel_bb": int(m_bb_rate * marcel_bf),
            "marcel_outs": int(m_outs_rate * marcel_bf),
        }

        # Marcel ER
        if "er_per_bf" in player_hist.columns:
            league_er = float(
                pitcher_extended[pitcher_extended["season"] == from_season]["er_per_bf"].mean()
            ) if "er_per_bf" in pitcher_extended.columns else 0.11
            m_er_rate = _marcel_rate(player_hist, "er_per_bf", league_er)
            result_row["marcel_earned_runs"] = int(m_er_rate * marcel_bf)

        results.append(result_row)

    return pd.DataFrame(results)
