"""Precompute: Game simulator data (samples, exit model, etc.)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.game_sim")


def run(
    *,
    seasons: list[int] = SEASONS,
    from_season: int = FROM_SEASON,
) -> None:
    """Generate game simulator supporting data."""
    from src.data.feature_eng import build_multi_season_hitter_extended
    from src.data.queries import (
        get_batter_pitch_count_features,
        get_exit_model_training_data,
        get_pitcher_exit_tendencies,
        get_pitcher_pitch_count_features,
        get_reliever_stats_by_team,
        get_team_bullpen_rates,
        get_tto_adjustment_profiles,
    )
    from src.models.game_sim.bullpen_model import build_team_bullpen_profiles
    from src.models.game_sim.exit_model import ExitModel

    logger.info("=" * 60)
    logger.info("Generating game simulator data...")

    # 9b. Hitter HR synthetic Beta samples (no Bayesian model -- use Beta posterior)
    logger.info("Generating synthetic hitter HR samples...")
    hitter_ext = build_multi_season_hitter_extended(seasons, min_pa=1)

    hr_season = hitter_ext[hitter_ext["season"] == from_season].copy()
    hr_samples_dict: dict[str, np.ndarray] = {}
    rng_hr = np.random.default_rng(42)
    for _, row in hr_season.iterrows():
        bid = int(row["batter_id"])
        pa = int(row.get("pa", row.get("plate_appearances", 0)))
        hr = int(row.get("hr", row.get("home_runs", 0)))
        if pa < 50:
            continue
        # Beta(hr + 1, pa - hr + 1) posterior
        hr_rate_samples = rng_hr.beta(hr + 1, pa - hr + 1, size=1_000)
        hr_samples_dict[str(bid)] = hr_rate_samples.astype(np.float32)
    np.savez_compressed(DASHBOARD_DIR / "hitter_hr_samples.npz", **hr_samples_dict)
    logger.info("Saved hitter HR Beta posterior samples for %d batters", len(hr_samples_dict))

    # 9c. Train + save exit model
    logger.info("Training pitcher exit model...")
    try:
        exit_training = get_exit_model_training_data(seasons)
        exit_tendencies = get_pitcher_exit_tendencies(seasons)
        exit_model = ExitModel()
        exit_metrics = exit_model.train(exit_training, exit_tendencies)
        logger.info(
            "Exit model: AUC=%.4f, n=%d",
            exit_metrics["auc"], exit_metrics["n_samples"],
        )
        exit_model.save(DASHBOARD_DIR / "exit_model.pkl")
        exit_tendencies.to_parquet(
            DASHBOARD_DIR / "pitcher_exit_tendencies.parquet", index=False,
        )
        logger.info("Saved exit model + tendencies")
    except Exception:
        logger.exception("Failed to train exit model")

    # 9d. Pitch count features
    logger.info("Computing pitch count features...")
    try:
        pitcher_pc = get_pitcher_pitch_count_features(seasons)
        pitcher_pc.to_parquet(
            DASHBOARD_DIR / "pitcher_pitch_count_features.parquet", index=False,
        )
        logger.info("Saved pitcher pitch count features: %d rows", len(pitcher_pc))

        batter_pc = get_batter_pitch_count_features(seasons)
        batter_pc.to_parquet(
            DASHBOARD_DIR / "batter_pitch_count_features.parquet", index=False,
        )
        logger.info("Saved batter pitch count features: %d rows", len(batter_pc))
    except Exception:
        logger.exception("Failed to compute pitch count features")

    # 9e. TTO profiles
    logger.info("Computing TTO adjustment profiles...")
    try:
        tto_profiles = get_tto_adjustment_profiles(seasons)
        tto_profiles.to_parquet(
            DASHBOARD_DIR / "tto_profiles.parquet", index=False,
        )
        logger.info("Saved TTO profiles: %d rows", len(tto_profiles))
    except Exception:
        logger.exception("Failed to compute TTO profiles")

    # 9f. Train + save game-level BB adjustment model
    logger.info("Training game-level BB adjustment model...")
    try:
        import pickle
        from src.models.game_bb_adj import train_game_bb_model

        bb_adj_bundle = train_game_bb_model(seasons)
        if bb_adj_bundle.get("model") is not None:
            with open(DASHBOARD_DIR / "game_bb_adj_model.pkl", "wb") as f:
                pickle.dump(bb_adj_bundle, f)
            logger.info(
                "Saved game BB adjustment model: n=%d, RMSE=%.4f",
                bb_adj_bundle["n_train"],
                bb_adj_bundle.get("rmse_logit", 0),
            )
        else:
            logger.warning("Game BB adjustment model training returned no model")
    except Exception:
        logger.exception("Failed to train game BB adjustment model")

    # 9g. Team bullpen rates
    logger.info("Computing team bullpen rates...")
    try:
        bullpen_rates = get_team_bullpen_rates(seasons)
        bullpen_rates.to_parquet(
            DASHBOARD_DIR / "team_bullpen_rates.parquet", index=False,
        )
        logger.info("Saved team bullpen rates: %d rows", len(bullpen_rates))
    except Exception:
        logger.exception("Failed to compute team bullpen rates")

    # 9h. Two-tier bullpen profiles (high-leverage CL/SU vs low-leverage MR)
    logger.info("Computing team bullpen tier profiles...")
    try:
        from src.models.reliever_roles import classify_reliever_roles

        reliever_stats = get_reliever_stats_by_team(seasons)
        roles = classify_reliever_roles(
            seasons=[from_season - 1, from_season],
            current_season=from_season,
            min_games=5,
        )

        # Build profiles for current season
        profiles = build_team_bullpen_profiles(
            role_history=reliever_stats,
            roles=roles,
            team_aggregate=bullpen_rates,
            season=from_season,
        )

        # Save as parquet for dashboard consumption
        if profiles:
            profile_rows = [
                {
                    "team_id": p.team_id,
                    "high_lev_k_rate": p.high_lev_k_rate,
                    "high_lev_bb_rate": p.high_lev_bb_rate,
                    "high_lev_hr_rate": p.high_lev_hr_rate,
                    "high_lev_bf": p.high_lev_bf,
                    "low_lev_k_rate": p.low_lev_k_rate,
                    "low_lev_bb_rate": p.low_lev_bb_rate,
                    "low_lev_hr_rate": p.low_lev_hr_rate,
                    "low_lev_bf": p.low_lev_bf,
                }
                for p in profiles.values()
            ]
            pd.DataFrame(profile_rows).to_parquet(
                DASHBOARD_DIR / "team_bullpen_profiles.parquet", index=False,
            )
            logger.info("Saved %d team bullpen tier profiles", len(profile_rows))
        else:
            logger.warning("No bullpen tier profiles built")
    except Exception:
        logger.exception("Failed to compute bullpen tier profiles")

    # 9i. Batter BIP profiles (per-batter out/single/double/triple probabilities)
    logger.info("Computing batter BIP profiles...")
    try:
        from src.data.queries.hitter import get_batter_bip_profile
        from src.data.queries.traditional import get_sprint_speed
        from src.models.game_sim.bip_model import compute_player_bip_probs

        bip_raw = get_batter_bip_profile(seasons)
        if bip_raw.empty:
            logger.warning("No batter BIP data returned")
        else:
            # Pull sprint speed for the current season (fallback 27.0 if missing)
            try:
                sprint = get_sprint_speed(from_season)
                sprint = sprint[["player_id", "sprint_speed"]].rename(
                    columns={"player_id": "batter_id"},
                )
            except Exception:
                logger.exception("Failed to fetch sprint speed; defaulting to 27.0")
                sprint = pd.DataFrame(columns=["batter_id", "sprint_speed"])

            # Keep only current season for profile construction; history rows
            # retained in case we ever want multi-season blending.
            cur = bip_raw[bip_raw["season"] == from_season].copy()
            cur = cur.merge(sprint, on="batter_id", how="left")
            cur["sprint_speed"] = cur["sprint_speed"].fillna(27.0)
            cur["avg_ev"] = cur["avg_ev"].fillna(88.0).astype(float)
            cur["avg_la"] = cur["avg_la"].fillna(12.0).astype(float)
            cur["gb_pct"] = cur["gb_pct"].fillna(0.44).astype(float)

            rows = []
            for _, r in cur.iterrows():
                bip = int(r["bip"])
                outs = int(r["bip_outs"])
                sgl = int(r["bip_singles"])
                dbl = int(r["bip_doubles"])
                tpl = int(r["bip_triples"])
                resolved = outs + sgl + dbl + tpl
                if resolved <= 0:
                    continue
                observed = {
                    "out": outs / resolved,
                    "single": sgl / resolved,
                    "double": dbl / resolved,
                    "triple": tpl / resolved,
                }
                probs = compute_player_bip_probs(
                    avg_ev=float(r["avg_ev"]),
                    avg_la=float(r["avg_la"]),
                    gb_pct=float(r["gb_pct"]),
                    sprint_speed=float(r["sprint_speed"]),
                    observed_bip_splits=observed,
                    bip_count=bip,
                    shrinkage_k=900,  # tuned via OOF sweep (2024→2025): beats
                                      # league on both log loss and Brier
                )
                rows.append({
                    "batter_id": int(r["batter_id"]),
                    "season": int(r["season"]),
                    "bip": bip,
                    "p_out": float(probs[0]),
                    "p_single": float(probs[1]),
                    "p_double": float(probs[2]),
                    "p_triple": float(probs[3]),
                    "avg_ev": float(r["avg_ev"]),
                    "avg_la": float(r["avg_la"]),
                    "gb_pct": float(r["gb_pct"]),
                    "sprint_speed": float(r["sprint_speed"]),
                })

            if rows:
                out_df = pd.DataFrame(rows)
                out_df.to_parquet(
                    DASHBOARD_DIR / "batter_bip_profiles.parquet", index=False,
                )
                logger.info(
                    "Saved batter BIP profiles: %d batters. "
                    "Mean p_out=%.3f p_single=%.3f p_double=%.3f p_triple=%.4f",
                    len(out_df),
                    out_df["p_out"].mean(),
                    out_df["p_single"].mean(),
                    out_df["p_double"].mean(),
                    out_df["p_triple"].mean(),
                )
            else:
                logger.warning("No batter BIP rows built")
    except Exception:
        logger.exception("Failed to compute batter BIP profiles")

    logger.info("Game simulator data complete.")
