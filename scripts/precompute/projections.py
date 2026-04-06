"""Precompute: Projections, counting stats, sim-based projections, health/parks."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.projections")


def run_health_parks(
    *,
    from_season: int = FROM_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute health/durability scores and park factors.

    Returns
    -------
    (health_df, park_factors, hitter_venues)
    """
    from src.models.health_score import compute_health_scores
    from src.data.queries import get_park_factors, get_hitter_team_venue

    logger.info("=" * 60)
    logger.info("Computing health/durability scores...")
    health_df = compute_health_scores(from_season)
    health_df.to_parquet(DASHBOARD_DIR / "health_scores.parquet", index=False)
    logger.info("Saved health scores: %d players", len(health_df))

    logger.info("Loading park factors for HR adjustment...")
    park_factors = get_park_factors(from_season)
    hitter_venues = get_hitter_team_venue(from_season)
    logger.info("Park factors: %d venue-hand combos, %d hitter-venue mappings",
                len(park_factors), len(hitter_venues))

    # Save so _load_park_factors() can find them when health_parks is skipped
    park_factors.to_parquet(DASHBOARD_DIR / "hr_park_factors.parquet", index=False)
    hitter_venues.to_parquet(DASHBOARD_DIR / "hitter_venues.parquet", index=False)

    return health_df, park_factors, hitter_venues


def _load_health_df(from_season: int) -> pd.DataFrame:
    """Load health scores from parquet or compute fresh."""
    try:
        return pd.read_parquet(DASHBOARD_DIR / "health_scores.parquet")
    except FileNotFoundError:
        from src.models.health_score import compute_health_scores
        return compute_health_scores(from_season)


def _load_park_factors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load HR park factors and hitter venues from parquet or compute fresh."""
    try:
        park_factors = pd.read_parquet(DASHBOARD_DIR / "hr_park_factors.parquet")
        hitter_venues = pd.read_parquet(DASHBOARD_DIR / "hitter_venues.parquet")
        return park_factors, hitter_venues
    except FileNotFoundError:
        from src.data.queries import get_park_factors, get_hitter_team_venue
        park_factors = get_park_factors(FROM_SEASON)
        hitter_venues = get_hitter_team_venue(FROM_SEASON)
        return park_factors, hitter_venues


def run_counting(
    *,
    hitter_results: dict,
    pitcher_results: dict,
    health_df: pd.DataFrame | None = None,
    park_factors: pd.DataFrame | None = None,
    hitter_venues: pd.DataFrame | None = None,
    seasons: list[int] = SEASONS,
    from_season: int = FROM_SEASON,
) -> None:
    """Compute counting stat projections (K, BB, HR, SB for hitters; K, BB, Outs for pitchers)."""
    from src.data.feature_eng import (
        build_multi_season_hitter_extended,
        build_multi_season_pitcher_extended,
    )
    from src.models.counting_projections import (
        project_hitter_counting,
        project_pitcher_counting,
    )
    from src.models.pa_model import compute_hitter_pa_priors

    logger.info("=" * 60)
    logger.info("Computing counting stat projections...")

    if health_df is None or health_df.empty:
        health_df = _load_health_df(from_season)

    if park_factors is None or hitter_venues is None:
        park_factors, hitter_venues = _load_park_factors()

    # Hitter counting stats (with park factor adjustment for HR)
    hitter_ext = build_multi_season_hitter_extended(seasons, min_pa=1)
    pa_priors = compute_hitter_pa_priors(
        hitter_ext, from_season=from_season, min_pa=100,
        health_scores=health_df,
    )

    hitter_counting = project_hitter_counting(
        rate_model_results=hitter_results,
        pa_priors=pa_priors,
        hitter_extended=hitter_ext,
        from_season=from_season,
        n_draws=4000,
        min_pa=150,
        random_seed=42,
        park_factors=park_factors,
        hitter_venues=hitter_venues,
    )
    # Pitcher counting stats
    pitcher_ext = build_multi_season_pitcher_extended(seasons, min_bf=9)
    pitcher_counting = project_pitcher_counting(
        rate_model_results=pitcher_results,
        pitcher_extended=pitcher_ext,
        from_season=from_season,
        n_draws=4000,
        min_bf=150,
        random_seed=42,
        health_scores=health_df,
    )

    hitter_counting.to_parquet(DASHBOARD_DIR / "hitter_counting.parquet", index=False)
    logger.info("Saved hitter counting projections: %d players", len(hitter_counting))
    pitcher_counting.to_parquet(DASHBOARD_DIR / "pitcher_counting.parquet", index=False)
    logger.info("Saved pitcher counting projections: %d players", len(pitcher_counting))


def run_sim_pitcher(
    *,
    pitcher_results: dict,
    health_df: pd.DataFrame | None = None,
    seasons: list[int] = SEASONS,
    from_season: int = FROM_SEASON,
    n_seasons: int = 10_000,
) -> pd.DataFrame | None:
    """Run sim-based pitcher season projections + reliever roles & rankings.

    Returns
    -------
    pd.DataFrame or None
        The reliever roles DataFrame (pitcher_id, role, ...) so the
        pipeline can pass it downstream without re-reading from disk.
        Returns None if pitcher_results are unavailable or on failure.
    """
    from src.data.feature_eng import build_multi_season_pitcher_extended
    from src.data.queries import get_exit_model_training_data, get_pitcher_exit_tendencies
    from src.models.counting_projections import (
        add_confidence_tiers,
        project_pitcher_counting_sim,
    )
    from src.models.game_sim.exit_model import ExitModel

    logger.info("=" * 60)
    logger.info("Running sim-based pitcher season projections...")

    if not pitcher_results:
        logger.warning("pitcher_results not available (pitcher_models skipped) -- skipping sim_pitcher")
        return None

    if health_df is None or health_df.empty:
        health_df = _load_health_df(from_season)

    pitcher_ext = build_multi_season_pitcher_extended(seasons, min_bf=9)

    try:
        from src.models.reliever_roles import classify_reliever_roles
        from src.models.reliever_rankings import rank_relievers

        # Classify reliever roles
        logger.info("Classifying reliever roles...")
        reliever_roles = classify_reliever_roles(
            seasons=list(range(max(2020, seasons[0]), seasons[-1] + 1)),
            current_season=from_season,
            min_games=10,
        )
        reliever_roles.to_parquet(DASHBOARD_DIR / "pitcher_roles.parquet", index=False)
        logger.info("Saved reliever roles: %d relievers", len(reliever_roles))

        # Build posteriors dict from existing rate samples
        logger.info("Building posterior rate samples for season sim...")
        pitcher_posteriors: dict[int, dict[str, np.ndarray]] = {}

        active_pitchers = pitcher_ext[
            (pitcher_ext["season"] == from_season)
            & (pitcher_ext["batters_faced"] >= 50)
        ][["pitcher_id", "pitcher_name", "batters_faced", "is_starter"]].drop_duplicates("pitcher_id")

        pitcher_name_lookup = dict(
            zip(active_pitchers["pitcher_id"], active_pitchers["pitcher_name"])
        )

        rng_post = np.random.default_rng(42)
        for _, prow in active_pitchers.iterrows():
            pid = int(prow["pitcher_id"])
            rates: dict[str, np.ndarray] = {}

            # K rate from composite pitcher model
            _k_pre = pitcher_results.get("k_rate", {}).get("rate_samples", {})
            if pid in _k_pre:
                rates["k_rate"] = _k_pre[pid].copy()
            else:
                rates["k_rate"] = rng_post.beta(5, 20, size=8000)

            # BB rate
            _bb_pre = pitcher_results.get("bb_rate", {}).get("rate_samples", {})
            if pid in _bb_pre:
                rates["bb_rate"] = _bb_pre[pid].copy()
            else:
                rates["bb_rate"] = rng_post.beta(3, 30, size=8000)

            # HR rate
            _hr_pre = pitcher_results.get("hr_per_bf", {}).get("rate_samples", {})
            if pid in _hr_pre:
                rates["hr_rate"] = _hr_pre[pid].copy()
            else:
                # Synthetic Beta from observed HR/BF
                bf = int(prow["batters_faced"])
                hr_obs = pitcher_ext[
                    (pitcher_ext["pitcher_id"] == pid)
                    & (pitcher_ext["season"] == from_season)
                ]["hr"].values
                hr_count = int(hr_obs[0]) if len(hr_obs) > 0 else int(bf * 0.03)
                rates["hr_rate"] = rng_post.beta(
                    hr_count + 1, bf - hr_count + 1, size=8000,
                ).astype(np.float32)

            pitcher_posteriors[pid] = rates

        logger.info("Built posteriors for %d pitchers", len(pitcher_posteriors))

        # Build starter-specific priors (n_starts, avg_pitches)
        logger.info("Building starter-specific priors...")
        starter_ids = set(
            active_pitchers[active_pitchers["is_starter"] == 1]["pitcher_id"]
        )

        # Query actual avg pitches per starter (weighted recent seasons)
        from src.data.db import read_sql as _read_pitches
        _pitch_counts = _read_pitches("""
            SELECT player_id as pitcher_id, season,
                   AVG(pit_pitches) as avg_pitches,
                   COUNT(*) as starts
            FROM production.fact_player_game_mlb
            WHERE pit_is_starter = true AND pit_pitches > 0
              AND season >= :min_season
            GROUP BY player_id, season
        """, {"min_season": from_season - 2})
        # Weighted avg (5/4/3) across recent seasons
        _pitch_lookup: dict[int, float] = {}
        _POP_AVG_PITCHES = 88.0
        if not _pitch_counts.empty:
            for pid in starter_ids:
                ph = _pitch_counts[
                    _pitch_counts["pitcher_id"] == pid
                ].sort_values("season", ascending=False).head(3)
                if ph.empty:
                    continue
                wts = [5, 4, 3][:len(ph)]
                wtd = sum(
                    a * s * w
                    for a, s, w in zip(ph["avg_pitches"], ph["starts"], wts)
                ) / sum(s * w for s, w in zip(ph["starts"], wts))
                # Regress toward population mean (more starts = more trust)
                total_starts = ph["starts"].sum()
                rel = min(total_starts / (total_starts + 15), 1.0)
                _pitch_lookup[pid] = rel * wtd + (1 - rel) * _POP_AVG_PITCHES
            logger.info(
                "Pitcher avg pitches: %d starters, range %.0f-%.0f (pop=%.0f)",
                len(_pitch_lookup),
                min(_pitch_lookup.values()) if _pitch_lookup else 0,
                max(_pitch_lookup.values()) if _pitch_lookup else 0,
                _POP_AVG_PITCHES,
            )

        starter_prior_rows = []
        for _, prow in active_pitchers.iterrows():
            pid = int(prow["pitcher_id"])
            if pid not in starter_ids:
                continue

            player_hist = pitcher_ext[
                (pitcher_ext["pitcher_id"] == pid)
                & (pitcher_ext["season"] <= from_season)
                & (pitcher_ext["is_starter"] == 1)
            ].sort_values("season", ascending=False).head(3)

            if player_hist.empty:
                starter_prior_rows.append({
                    "pitcher_id": pid,
                    "n_starts_mu": 30.0,
                    "n_starts_sigma": 6.0,
                    "avg_pitches": _pitch_lookup.get(pid, _POP_AVG_PITCHES),
                })
                continue

            # Weighted starts (5/4/3)
            weights = [5, 4, 3][:len(player_hist)]
            adj_games = player_hist["games"].values.copy().astype(float)
            for i, s in enumerate(player_hist["season"].values):
                if s == 2020:
                    adj_games[i] = min(adj_games[i] * (162 / 60), 36)
            wtd_starts = sum(g * w for g, w in zip(adj_games, weights)) / sum(weights)
            # Regress toward pop mean
            n = len(player_hist)
            rel = n / (n + 1.5)
            mu = rel * wtd_starts + (1 - rel) * 30.0

            # Sigma from variation (floor at 3)
            if len(adj_games) >= 2:
                sigma = max(float(np.std(adj_games, ddof=1)), 3.0)
            else:
                sigma = 6.0

            starter_prior_rows.append({
                "pitcher_id": pid,
                "n_starts_mu": mu,
                "n_starts_sigma": sigma,
                "avg_pitches": _pitch_lookup.get(pid, _POP_AVG_PITCHES),
            })

        starter_priors_df = pd.DataFrame(starter_prior_rows) if starter_prior_rows else None

        # Train exit model
        logger.info("Training exit model for season sim...")
        sim_exit_model = ExitModel()
        _exit_model_path = Path("data/cached/exit_model.pkl")
        try:
            exit_training_data = get_exit_model_training_data(seasons)
            exit_tend = get_pitcher_exit_tendencies(seasons)
            sim_exit_model.train(exit_training_data, exit_tend)
            sim_exit_model.save(_exit_model_path)
            logger.info("Exit model trained and saved to %s", _exit_model_path)
        except Exception as e:
            logger.warning("Exit model training failed, using fallback: %s", e)
            # Try loading a previously saved model
            if _exit_model_path.exists():
                sim_exit_model.load(_exit_model_path)
                logger.info("Loaded cached exit model from %s", _exit_model_path)

        # Run sim-based projections
        t0 = time.time()

        pitcher_counting_sim = project_pitcher_counting_sim(
            posteriors=pitcher_posteriors,
            roles=reliever_roles,
            exit_model=sim_exit_model,
            starter_priors=starter_priors_df,
            health_scores=health_df,
            pitcher_names=pitcher_name_lookup,
            n_seasons=n_seasons,
            random_seed=42,
        )
        elapsed = time.time() - t0

        # Add confidence tiers before saving
        pitcher_counting_sim = add_confidence_tiers(pitcher_counting_sim, player_type="pitcher")

        pitcher_counting_sim.to_parquet(
            DASHBOARD_DIR / "pitcher_counting_sim.parquet", index=False,
        )
        logger.info(
            "Saved sim-based pitcher projections: %d pitchers in %.1fs",
            len(pitcher_counting_sim), elapsed,
        )

        # Reliever rankings
        logger.info("Building reliever rankings...")
        rp_rankings = rank_relievers(
            sim_df=pitcher_counting_sim,
            roles_df=reliever_roles,
            season=from_season,
        )
        rp_rankings.to_parquet(
            DASHBOARD_DIR / "reliever_rankings.parquet", index=False,
        )
        logger.info("Saved reliever rankings: %d relievers", len(rp_rankings))

        return reliever_roles

    except Exception:
        logger.exception("Failed to run sim-based pitcher projections")
        return None


def run_sim_hitter(
    *,
    hitter_results: dict,
    health_df: pd.DataFrame | None = None,
    seasons: list[int] = SEASONS,
    from_season: int = FROM_SEASON,
    n_seasons: int = 10_000,
) -> None:
    """Run sim-based hitter season projections."""
    from src.data.feature_eng import build_multi_season_hitter_extended
    from src.models.pa_model import compute_hitter_pa_priors

    logger.info("=" * 60)
    logger.info("Running sim-based hitter season projections...")

    if not hitter_results:
        logger.warning("hitter_results not available (hitter_models skipped) -- skipping sim_hitter")
        return

    if health_df is None or health_df.empty:
        health_df = _load_health_df(from_season)

    hitter_ext = build_multi_season_hitter_extended(seasons, min_pa=1)

    pa_priors = compute_hitter_pa_priors(
        hitter_ext, from_season=from_season, min_pa=100,
        health_scores=health_df,
    )

    try:
        from src.models.counting_projections import (
            add_confidence_tiers,
            project_hitter_counting_sim,
        )

        # Build hitter posteriors
        logger.info("Building hitter posterior rate samples...")
        hitter_posteriors: dict[int, dict[str, np.ndarray]] = {}
        hitter_name_lookup: dict[int, str] = {}

        _hitter_cols = ["batter_id", "batter_name", "pa", "hr", "sb", "games",
                        "h", "k", "bb", "sprint_speed"]
        _hitter_cols = [c for c in _hitter_cols if c in hitter_ext.columns]
        active_hitters = hitter_ext[
            (hitter_ext["season"] == from_season)
            & (hitter_ext["pa"] >= 50)
        ][_hitter_cols].drop_duplicates("batter_id")

        hitter_name_lookup = dict(
            zip(active_hitters["batter_id"], active_hitters["batter_name"])
        )

        rng_h = np.random.default_rng(42)
        for _, hrow in active_hitters.iterrows():
            bid = int(hrow["batter_id"])
            rates: dict[str, np.ndarray] = {}

            for stat_key in ["k_rate", "bb_rate"]:
                _h_pre = hitter_results.get(stat_key, {}).get("rate_samples", {})
                if bid in _h_pre:
                    rates[stat_key] = _h_pre[bid].copy()
                elif stat_key == "k_rate":
                    rates[stat_key] = rng_h.beta(5, 18, size=8000)
                else:
                    rates[stat_key] = rng_h.beta(3, 30, size=8000)

            # HR: Marcel-weighted Beta (5/4/3 across 3 seasons + regression)
            bid_hist = hitter_ext[
                (hitter_ext["batter_id"] == bid)
                & (hitter_ext["season"] >= from_season - 2)
                & (hitter_ext["season"] <= from_season)
            ]
            marcel_weights = {from_season: 5, from_season - 1: 4, from_season - 2: 3}
            wtd_hr, wtd_pa = 0.0, 0.0
            for _, hist_row in bid_hist.iterrows():
                w = marcel_weights.get(int(hist_row["season"]), 1)
                wtd_hr += w * int(hist_row.get("hr", 0))
                wtd_pa += w * int(hist_row["pa"])
            # Regress toward league average (~3% HR/PA) with 200 PA of regression
            reg_pa = 200
            league_hr_rate = 0.030
            reg_hr = wtd_hr + reg_pa * league_hr_rate
            reg_total = wtd_pa + reg_pa
            if reg_total > 0:
                rates["hr_rate"] = rng_h.beta(
                    max(reg_hr, 0.5), max(reg_total - reg_hr, 0.5), size=8000,
                ).astype(np.float32)
            else:
                rates["hr_rate"] = rng_h.beta(1, 30, size=8000).astype(np.float32)

            hitter_posteriors[bid] = rates

        logger.info("Built hitter posteriors for %d batters", len(hitter_posteriors))

        # SB rates: blend observed SB/game with sprint speed signal
        sb_rates: dict[int, tuple[float, float]] = {}
        pop_sb_rate = active_hitters["sb"].sum() / max(active_hitters["games"].sum(), 1)
        for _, hrow in active_hitters.iterrows():
            bid = int(hrow["batter_id"])
            games = max(int(hrow.get("games", 100)), 1)
            sb = int(hrow.get("sb", 0))
            observed_rate = sb / games

            speed = hrow.get("sprint_speed", None)
            if speed is not None and not np.isnan(speed) and speed > 0:
                speed_implied = max(0, (speed - 27.2) * 0.04 + pop_sb_rate)
                blended = 0.40 * speed_implied + 0.60 * observed_rate
            else:
                blended = observed_rate

            sb_rates[bid] = (blended, max(blended * 0.30, 0.01))

        # Per-player batting order from lineup data
        logger.info("Loading batting order positions...")
        batting_orders = None
        try:
            from src.data.db import read_sql as _rs_bo
            bo_data = _rs_bo("""
                SELECT player_id as batter_id,
                       ROUND(AVG(batting_order))::int as avg_order
                FROM production.fact_lineup
                WHERE season = :season AND is_starter = true
                      AND batting_order BETWEEN 1 AND 9
                GROUP BY player_id
                HAVING COUNT(*) >= 20
            """, {"season": from_season})
            batting_orders = dict(zip(
                bo_data["batter_id"].astype(int),
                bo_data["avg_order"].astype(int).clip(1, 9),
            ))
            logger.info("Batting orders: %d hitters (mean slot %.1f)",
                        len(batting_orders), np.mean(list(batting_orders.values())))
        except Exception as e:
            logger.warning("Batting order lookup failed: %s", e)

        # Hitter BABIP adjustments
        logger.info("Computing hitter BABIP adjustments...")
        hitter_babip: dict[int, float] = {}
        try:
            h_col = "hits_allowed" if "hits_allowed" in hitter_ext.columns else "hits"
            if h_col not in hitter_ext.columns:
                h_col = None
            if h_col:
                recent_h = hitter_ext[
                    (hitter_ext["season"] >= from_season - 2)
                    & (hitter_ext["season"] <= from_season)
                ].copy()
                recent_h["bip"] = (recent_h["pa"] - recent_h["k"] - recent_h["hr"] - recent_h["bb"]).clip(1)
                recent_h["hits_on_bip"] = (recent_h[h_col] - recent_h["hr"]).clip(0)
                recent_h["babip"] = recent_h["hits_on_bip"] / recent_h["bip"]
                recent_h["weight"] = np.where(recent_h["season"] == from_season, 3.0, 1.0)

                h_agg = recent_h.groupby("batter_id").apply(
                    lambda g: pd.Series({
                        "wtd_babip": np.average(g["babip"], weights=g["weight"] * g["bip"]),
                        "total_bip": g["bip"].sum(),
                    }), include_groups=False,
                ).reset_index()
                h_agg["reliability"] = (h_agg["total_bip"] / 500).clip(0, 1)
                h_agg["babip_adj"] = h_agg["reliability"] * (h_agg["wtd_babip"] - 0.293)
                hitter_babip = dict(zip(h_agg["batter_id"].astype(int), h_agg["babip_adj"]))
                logger.info("Hitter BABIP adjustments: %d, mean=%.4f",
                            len(hitter_babip), np.mean(list(hitter_babip.values())))
        except Exception as e:
            logger.warning("Hitter BABIP failed: %s", e)

        # Player-specific BIP profiles (hit-type distribution)
        logger.info("Computing player BIP profiles...")
        bip_profiles: dict[int, np.ndarray] = {}
        try:
            from src.models.game_sim.bip_model import compute_player_bip_probs

            # Load quality metrics from hitter_full_stats (has EV, LA, GB%)
            hfs_path = DASHBOARD_DIR / "hitter_full_stats.parquet"
            if hfs_path.exists():
                hfs = pd.read_parquet(hfs_path)
                hfs_recent = hfs[
                    (hfs["season"] >= from_season - 2)
                    & (hfs["season"] <= from_season)
                ].copy()

                # Marcel-style weighting: 5/4/3
                weight_map = {
                    from_season: 5.0,
                    from_season - 1: 4.0,
                    from_season - 2: 3.0,
                }
                hfs_recent["weight"] = hfs_recent["season"].map(weight_map).fillna(1.0)
                bip_col = "batted_balls" if "batted_balls" in hfs_recent.columns else "total_bip"
                if bip_col not in hfs_recent.columns:
                    hfs_recent[bip_col] = (
                        hfs_recent["pa"] - hfs_recent["k"] - hfs_recent["bb"]
                    ).clip(1)
                hfs_recent["wtd_bip"] = hfs_recent["weight"] * hfs_recent[bip_col].fillna(0).clip(1)

                quality_agg = hfs_recent.groupby("batter_id").apply(
                    lambda g: pd.Series({
                        "avg_ev": np.average(
                            g["avg_exit_velo"].fillna(88.0), weights=g["wtd_bip"],
                        ),
                        "avg_la": np.average(
                            g["avg_launch_angle"].fillna(12.0), weights=g["wtd_bip"],
                        ),
                        "gb_pct": np.average(
                            g["gb_pct"].fillna(0.44), weights=g["wtd_bip"],
                        ),
                    }),
                    include_groups=False,
                ).reset_index()

                # Sprint speed from hitter_ext
                sprint_lookup = dict(
                    hitter_ext[hitter_ext["season"] == from_season]
                    .dropna(subset=["sprint_speed"])
                    .groupby("batter_id")["sprint_speed"]
                    .first()
                )

                # Observed BIP splits from hitter_ext (multi-season)
                recent_ext = hitter_ext[
                    (hitter_ext["season"] >= from_season - 2)
                    & (hitter_ext["season"] <= from_season)
                ].copy()
                recent_ext["singles"] = (
                    recent_ext["hits"] - recent_ext["hr"]
                    - recent_ext["doubles"] - recent_ext["triples"]
                ).clip(0)
                hbp = recent_ext.get("hit_by_pitch", pd.Series(0, index=recent_ext.index)).fillna(0)
                recent_ext["bip_est"] = (
                    recent_ext["pa"] - recent_ext["k"] - recent_ext["bb"]
                    - recent_ext["hr"] - hbp
                ).clip(1)
                recent_ext["outs_on_bip"] = (
                    recent_ext["bip_est"] - recent_ext["singles"]
                    - recent_ext["doubles"] - recent_ext["triples"]
                ).clip(0)

                bip_split_agg = recent_ext.groupby("batter_id").agg(
                    singles=("singles", "sum"),
                    doubles=("doubles", "sum"),
                    triples=("triples", "sum"),
                    outs_on_bip=("outs_on_bip", "sum"),
                    bip_est=("bip_est", "sum"),
                ).reset_index()

                # Build BIP profiles
                for _, row in quality_agg.iterrows():
                    bid = int(row["batter_id"])
                    if bid not in hitter_posteriors:
                        continue

                    ev = float(row["avg_ev"]) if pd.notna(row["avg_ev"]) else 88.0
                    la = float(row["avg_la"]) if pd.notna(row["avg_la"]) else 12.0
                    gb = float(row["gb_pct"]) if pd.notna(row["gb_pct"]) else 0.44
                    speed = sprint_lookup.get(bid, 27.0)

                    # Observed BIP splits for shrinkage blending
                    obs_splits = None
                    bip_n = 0
                    bip_row = bip_split_agg[bip_split_agg["batter_id"] == bid]
                    if not bip_row.empty:
                        br = bip_row.iloc[0]
                        bip_n = int(br["bip_est"])
                        if bip_n > 0:
                            obs_splits = {
                                "out": float(br["outs_on_bip"]) / bip_n,
                                "single": float(br["singles"]) / bip_n,
                                "double": float(br["doubles"]) / bip_n,
                                "triple": float(br["triples"]) / bip_n,
                            }

                    bip_profiles[bid] = compute_player_bip_probs(
                        avg_ev=ev, avg_la=la, gb_pct=gb, sprint_speed=speed,
                        observed_bip_splits=obs_splits,
                        bip_count=bip_n, shrinkage_k=300,
                    )

                logger.info(
                    "BIP profiles: %d hitters, mean BABIP=%.3f",
                    len(bip_profiles),
                    np.mean([1.0 - p[0] for p in bip_profiles.values()]) if bip_profiles else 0,
                )
            else:
                logger.warning("hitter_full_stats.parquet not found, skipping BIP profiles")
        except Exception as e:
            logger.warning("BIP profile computation failed: %s", e)

        # Run sim
        t0_h = time.time()
        hitter_counting_sim = project_hitter_counting_sim(
            posteriors=hitter_posteriors,
            pa_priors=pa_priors,
            batting_orders=batting_orders,
            babip_adjs=hitter_babip if hitter_babip else None,
            bip_profiles=bip_profiles if bip_profiles else None,
            sb_rates=sb_rates,
            health_scores=health_df,
            batter_names=hitter_name_lookup,
            n_seasons=n_seasons,
            random_seed=42,
        )
        elapsed_h = time.time() - t0_h

        # Add confidence tiers before saving
        hitter_counting_sim = add_confidence_tiers(hitter_counting_sim, player_type="hitter")

        hitter_counting_sim.to_parquet(
            DASHBOARD_DIR / "hitter_counting_sim.parquet", index=False,
        )
        logger.info(
            "Saved sim-based hitter projections: %d batters in %.1fs",
            len(hitter_counting_sim), elapsed_h,
        )

    except Exception:
        logger.exception("Failed to run sim-based hitter projections")
