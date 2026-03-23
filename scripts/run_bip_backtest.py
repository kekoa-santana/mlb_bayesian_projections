"""Walk-forward backtest: BIP profiles + BABIP fix impact on hitter season sim.

Compares projected wRC+, wOBA, H, HR to actual across OOF folds.
Runs each fold twice: with and without BIP profiles / BABIP adjustments.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_eng import (
    build_multi_season_hitter_data,
    build_multi_season_hitter_extended,
)
from src.models.hitter_model import fit_hitter_model, prepare_hitter_data, extract_rate_samples
from src.models.pa_model import compute_hitter_pa_priors
from src.models.counting_projections import project_hitter_counting_sim
from src.models.game_sim.bip_model import compute_player_bip_probs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bip_backtest")

DASHBOARD_DIR = Path("C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard")


def _build_posteriors(
    hitter_results: dict,
    hitter_ext: pd.DataFrame,
    from_season: int,
    min_pa: int = 150,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, str]]:
    """Build rate posteriors for hitters active in from_season."""
    rng = np.random.default_rng(42)
    active = hitter_ext[
        (hitter_ext["season"] == from_season) & (hitter_ext["pa"] >= min_pa)
    ].copy()

    posteriors: dict[int, dict[str, np.ndarray]] = {}
    names: dict[int, str] = {}

    for _, row in active.iterrows():
        bid = int(row["batter_id"])
        rates: dict[str, np.ndarray] = {}

        for stat_key in ["k_rate", "bb_rate"]:
            pre = hitter_results.get(stat_key, {}).get("rate_samples", {})
            if bid in pre:
                rates[stat_key] = pre[bid].copy()
            elif stat_key == "k_rate":
                rates[stat_key] = rng.beta(5, 18, size=2000).astype(np.float32)
            else:
                rates[stat_key] = rng.beta(3, 30, size=2000).astype(np.float32)

        # HR: Marcel-weighted Beta (5/4/3 + regression)
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
        reg_pa = 200
        reg_hr = wtd_hr + reg_pa * 0.030
        reg_total = wtd_pa + reg_pa
        if reg_total > 0:
            rates["hr_rate"] = rng.beta(
                max(reg_hr, 0.5), max(reg_total - reg_hr, 0.5), size=2000,
            ).astype(np.float32)
        else:
            rates["hr_rate"] = rng.beta(1, 30, size=2000).astype(np.float32)

        posteriors[bid] = rates
        names[bid] = str(row.get("batter_name", bid))

    return posteriors, names


def _build_babip_adjs(
    hitter_ext: pd.DataFrame,
    from_season: int,
) -> dict[int, float]:
    """Build BABIP adjustments from recent seasons."""
    recent = hitter_ext[
        (hitter_ext["season"] >= from_season - 2)
        & (hitter_ext["season"] <= from_season)
    ].copy()
    recent["bip"] = (recent["pa"] - recent["k"] - recent["hr"] - recent["bb"]).clip(1)
    recent["hits_on_bip"] = (recent["hits"] - recent["hr"]).clip(0)
    recent["babip"] = recent["hits_on_bip"] / recent["bip"]
    recent["weight"] = np.where(recent["season"] == from_season, 3.0, 1.0)

    agg = recent.groupby("batter_id").apply(
        lambda g: pd.Series({
            "wtd_babip": np.average(g["babip"], weights=g["weight"] * g["bip"]),
            "total_bip": g["bip"].sum(),
        }), include_groups=False,
    ).reset_index()
    agg["reliability"] = (agg["total_bip"] / 500).clip(0, 1)
    agg["babip_adj"] = agg["reliability"] * (agg["wtd_babip"] - 0.293)
    return dict(zip(agg["batter_id"].astype(int), agg["babip_adj"]))


def _build_bip_profiles(
    hitter_ext: pd.DataFrame,
    from_season: int,
    posteriors: dict[int, dict[str, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Build player-specific BIP profiles from train data only."""
    # Load quality metrics from hitter_full_stats
    hfs_path = DASHBOARD_DIR / "hitter_full_stats.parquet"
    if not hfs_path.exists():
        logger.warning("hitter_full_stats.parquet not found")
        return {}

    hfs = pd.read_parquet(hfs_path)
    hfs_recent = hfs[
        (hfs["season"] >= from_season - 2) & (hfs["season"] <= from_season)
    ].copy()

    if hfs_recent.empty:
        return {}

    weight_map = {from_season: 5.0, from_season - 1: 4.0, from_season - 2: 3.0}
    hfs_recent["weight"] = hfs_recent["season"].map(weight_map).fillna(1.0)
    bip_col = "batted_balls" if "batted_balls" in hfs_recent.columns else "total_bip"
    if bip_col not in hfs_recent.columns:
        hfs_recent[bip_col] = (hfs_recent["pa"] - hfs_recent["k"] - hfs_recent["bb"]).clip(1)
    hfs_recent["wtd_bip"] = hfs_recent["weight"] * hfs_recent[bip_col].fillna(0).clip(1)

    quality_agg = hfs_recent.groupby("batter_id").apply(
        lambda g: pd.Series({
            "avg_ev": np.average(g["avg_exit_velo"].fillna(88.0), weights=g["wtd_bip"]),
            "avg_la": np.average(g["avg_launch_angle"].fillna(12.0), weights=g["wtd_bip"]),
            "gb_pct": np.average(g["gb_pct"].fillna(0.44), weights=g["wtd_bip"]),
        }), include_groups=False,
    ).reset_index()

    sprint_lookup = dict(
        hitter_ext[hitter_ext["season"] == from_season]
        .dropna(subset=["sprint_speed"])
        .groupby("batter_id")["sprint_speed"].first()
    )

    # Observed BIP splits
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
        singles=("singles", "sum"), doubles=("doubles", "sum"),
        triples=("triples", "sum"), outs_on_bip=("outs_on_bip", "sum"),
        bip_est=("bip_est", "sum"),
    ).reset_index()

    profiles: dict[int, np.ndarray] = {}
    for _, row in quality_agg.iterrows():
        bid = int(row["batter_id"])
        if bid not in posteriors:
            continue

        ev = float(row["avg_ev"]) if pd.notna(row["avg_ev"]) else 88.0
        la = float(row["avg_la"]) if pd.notna(row["avg_la"]) else 12.0
        gb = float(row["gb_pct"]) if pd.notna(row["gb_pct"]) else 0.44
        speed = sprint_lookup.get(bid, 27.0)

        obs_splits, bip_n = None, 0
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

        profiles[bid] = compute_player_bip_probs(
            avg_ev=ev, avg_la=la, gb_pct=gb, sprint_speed=speed,
            observed_bip_splits=obs_splits, bip_count=bip_n, shrinkage_k=300,
        )

    return profiles


def _get_actuals(season: int, min_pa: int = 300) -> pd.DataFrame:
    """Load actual season totals for the test season."""
    from src.data.feature_eng import get_cached_hitter_season_totals_extended
    df = get_cached_hitter_season_totals_extended(season)
    df = df[df["pa"] >= min_pa].copy()
    # Compute wOBA and wRC+
    if "woba" not in df.columns:
        from src.data.feature_eng import get_cached_season_totals_with_age
        st = get_cached_season_totals_with_age(season)
        st = st[["batter_id", "woba"]].dropna()
        df = df.merge(st, on="batter_id", how="left")
    return df


def run_fold(
    train_seasons: list[int],
    test_season: int,
    use_bip: bool,
    use_babip: bool,
    draws: int = 500,
    tune: int = 250,
    chains: int = 2,
) -> pd.DataFrame:
    """Run one backtest fold."""
    from_season = train_seasons[-1]
    all_seasons = train_seasons

    logger.info("Building hitter data for seasons %s...", all_seasons)
    hitter_data = build_multi_season_hitter_data(all_seasons, min_pa=100)
    hitter_ext = build_multi_season_hitter_extended(all_seasons, min_pa=1)

    # Fit K% and BB% models
    hitter_results: dict = {}
    for stat in ["k_rate", "bb_rate"]:
        logger.info("Fitting %s model...", stat)
        data = prepare_hitter_data(hitter_data, stat=stat)
        model, trace = fit_hitter_model(
            data, draws=draws, tune=tune, chains=chains, random_seed=42,
        )
        # Extract rate samples for all active hitters in from_season
        stat_df = data["df"]
        active_ids = stat_df[stat_df["season"] == from_season]["batter_id"].unique()
        rate_samples: dict[int, np.ndarray] = {}
        for pid in active_ids:
            try:
                rate_samples[int(pid)] = extract_rate_samples(
                    trace, data, batter_id=int(pid), season=from_season,
                    project_forward=True, random_seed=42,
                )
            except Exception:
                pass
        hitter_results[stat] = {"rate_samples": rate_samples}
        logger.info("Extracted %s samples for %d hitters", stat, len(rate_samples))

    # Build posteriors
    posteriors, names = _build_posteriors(hitter_results, hitter_ext, from_season)
    logger.info("Posteriors: %d hitters", len(posteriors))

    # PA priors
    pa_priors = compute_hitter_pa_priors(hitter_ext, from_season=from_season, min_pa=100)

    # BIP profiles
    bip_profiles = None
    if use_bip:
        bip_profiles = _build_bip_profiles(hitter_ext, from_season, posteriors)
        logger.info("BIP profiles: %d hitters", len(bip_profiles))

    # BABIP adjustments
    babip_adjs = None
    if use_babip:
        babip_adjs = _build_babip_adjs(hitter_ext, from_season)
        logger.info("BABIP adjustments: %d hitters", len(babip_adjs))

    # Run sim
    t0 = time.time()
    sim_df = project_hitter_counting_sim(
        posteriors=posteriors,
        pa_priors=pa_priors,
        babip_adjs=babip_adjs,
        bip_profiles=bip_profiles,
        batter_names=names,
        n_seasons=200,
        random_seed=42,
    )
    elapsed = time.time() - t0
    logger.info("Sim: %d hitters in %.1fs", len(sim_df), elapsed)

    # Load actuals for test season
    actuals = _get_actuals(test_season, min_pa=300)

    # Merge sim projections with actuals
    merged = sim_df.merge(
        actuals[["batter_id", "pa", "hits", "hr", "k", "bb", "doubles", "triples",
                 "total_bases", "woba"]],
        on="batter_id", how="inner",
    )

    # Compute actual wRC+ from wOBA
    lg_woba = 0.318
    woba_scale = 1.25
    lg_r_per_pa = 0.118
    merged["actual_wrc_plus"] = (
        (merged["woba"] - lg_woba) / woba_scale + lg_r_per_pa
    ) / lg_r_per_pa * 100

    merged["test_season"] = test_season

    logger.info(
        "Fold %d: %d hitters matched, wRC+ corr=%.3f",
        test_season, len(merged),
        merged["projected_wrc_plus_mean"].corr(merged["actual_wrc_plus"]),
    )

    return merged


def main() -> None:
    folds = [
        ([2018, 2019, 2020, 2021, 2022, 2023], 2024),
        ([2018, 2019, 2020, 2021, 2022, 2023, 2024], 2025),
    ]

    variants = [
        (False, False, "without_bip", "BASELINE (no BIP, no BABIP)"),
        (True, False, "bip_only", "BIP ONLY (no BABIP)"),
        (True, True, "bip_babip", "BIP + BABIP"),
    ]

    all_results = []
    for train, test in folds:
        for use_bip, use_babip, variant_key, label in variants:
            logger.info("=" * 60)
            logger.info("FOLD: train=%s, test=%d, variant=%s", train, test, label)
            result = run_fold(
                train_seasons=train, test_season=test,
                use_bip=use_bip, use_babip=use_babip,
                draws=500, tune=250, chains=2,
            )
            result["variant"] = variant_key
            all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)

    # Compute metrics by variant
    print("\n" + "=" * 70)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)

    for variant in ["without_bip", "bip_only", "bip_babip"]:
        sub = combined[combined["variant"] == variant]
        label = {"without_bip": "BASELINE", "bip_only": "BIP ONLY", "bip_babip": "BIP + BABIP (0.293)"}.get(variant, variant)
        print(f"\n--- {label} (n={len(sub)}) ---")

        for stat, proj_col, actual_col in [
            ("wRC+", "projected_wrc_plus_mean", "actual_wrc_plus"),
            ("wOBA", "projected_woba_mean", "woba"),
            ("H", "total_h_mean", "hits"),
            ("HR", "total_hr_mean", "hr"),
            ("K", "total_k_mean", "k"),
        ]:
            if proj_col not in sub.columns or actual_col not in sub.columns:
                continue
            valid = sub.dropna(subset=[proj_col, actual_col])
            a = valid[actual_col].values
            p = valid[proj_col].values
            mae = np.mean(np.abs(a - p))
            rmse = np.sqrt(np.mean((a - p) ** 2))
            corr = np.corrcoef(a, p)[0, 1] if len(a) > 1 else 0
            bias = np.mean(p - a)
            print(f"  {stat:6s}: MAE={mae:7.2f}  RMSE={rmse:7.2f}  corr={corr:.3f}  bias={bias:+.2f}")

        # By wRC+ bucket
        print("  --- By observed wRC+ bucket ---")
        for lo, hi, label_b in [(0, 80, "<80"), (80, 100, "80-100"), (100, 120, "100-120"), (120, 300, "120+")]:
            bucket = sub[(sub["actual_wrc_plus"] >= lo) & (sub["actual_wrc_plus"] < hi)]
            if len(bucket) < 5:
                continue
            mae_b = np.mean(np.abs(bucket["projected_wrc_plus_mean"] - bucket["actual_wrc_plus"]))
            bias_b = np.mean(bucket["projected_wrc_plus_mean"] - bucket["actual_wrc_plus"])
            print(f"    {label_b:8s} (n={len(bucket):3d}): MAE={mae_b:6.1f}  bias={bias_b:+6.1f}")

    # Save results
    output_path = Path("outputs/bip_backtest_results.csv")
    output_path.parent.mkdir(exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    main()
