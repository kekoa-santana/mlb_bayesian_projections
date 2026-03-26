"""
Automated overnight hill-climbing optimizer for game simulator constants.

Randomly perturbs CONFIG values from train.py, runs the simulator,
and keeps improvements. Fully autonomous — no AI agent required.

Usage:
    python autoresearch/hill_climb.py                      # default: 100 experiments
    python autoresearch/hill_climb.py --n-experiments 200  # run 200 experiments
    python autoresearch/hill_climb.py --max-games 1000     # subsample for speed
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent
BEST_CONFIG_PATH = OUTPUT_DIR / "best_config.json"
LOG_PATH = OUTPUT_DIR / "experiment_log.csv"

# Parameter bounds and perturbation ranges.
# Keys must match CONFIG in train.py.
# (min, max, perturbation_std_fraction)
PARAM_BOUNDS: dict[str, tuple[float, float, float]] = {
    # Fatigue
    "fatigue_pitch_threshold": (60, 110, 0.10),
    "fatigue_k_slope": (-0.015, 0.0, 0.20),
    "fatigue_bb_slope": (0.0, 0.010, 0.20),
    "fatigue_hr_slope": (0.0, 0.005, 0.20),

    # HBP
    "league_hbp_rate": (0.005, 0.020, 0.15),

    # BIP splits (must sum to 1.0 — handled specially)
    "bip_out": (0.65, 0.75, 0.03),
    "bip_single": (0.18, 0.27, 0.03),
    "bip_double": (0.04, 0.10, 0.03),
    "bip_triple": (0.002, 0.012, 0.03),

    # BABIP
    "pop_babip": (0.270, 0.330, 0.05),
    "babip_shrinkage_k": (200, 1000, 0.15),
    "babip_coef_avg_ev": (-0.001, 0.002, 0.20),
    "babip_coef_avg_la": (-0.010, 0.0, 0.20),
    "babip_coef_gb_pct": (-0.15, 0.0, 0.20),
    "babip_coef_sprint_speed": (0.0, 0.010, 0.20),
    "babip_coef_intercept": (0.10, 0.35, 0.15),

    # Hit distribution
    "hit_share_single": (0.60, 0.85, 0.05),
    "hit_share_double": (0.12, 0.32, 0.05),
    "hit_share_triple": (0.005, 0.05, 0.05),
    "ev_double_sensitivity": (0.0, 0.08, 0.20),
    "gb_double_sensitivity": (-0.06, 0.0, 0.20),
    "speed_triple_sensitivity": (0.0, 0.03, 0.20),

    # Pitch count
    "pc_shrinkage_pitcher": (50, 400, 0.15),
    "pc_shrinkage_batter": (50, 400, 0.15),
    "pc_putaway_sensitivity": (-6.0, -1.0, 0.15),
    "pc_foul_rate_sensitivity": (1.0, 5.0, 0.15),
    "pc_contact_sensitivity": (-3.0, 0.0, 0.15),

    # TTO
    "tto_min_reliability_pa": (30, 300, 0.15),

    # Exit model
    "exit_default_avg_pitches": (75, 100, 0.08),
    "exit_shrinkage_k": (3, 30, 0.20),

    # Exit calibration offset (logit scale, negative = pitcher stays longer)
    "exit_calibration_offset": (-0.60, 0.10, 0.15),

    # Per-PA calibration offsets (logit scale)
    "calibration_k_offset": (-0.15, 0.10, 0.20),
    "calibration_bb_offset": (-0.10, 0.10, 0.20),
}


def perturb_config(
    base: dict,
    rng: np.random.Generator,
    n_params: int = 2,
) -> dict:
    """Randomly perturb 1-n_params values from the base config.

    BIP splits are handled as a group to maintain sum = 1.0.
    """
    cfg = copy.deepcopy(base)

    # Select parameters to perturb
    tunable_keys = [k for k in PARAM_BOUNDS if k in cfg]
    bip_keys = {"bip_out", "bip_single", "bip_double", "bip_triple"}
    non_bip_keys = [k for k in tunable_keys if k not in bip_keys]

    # Decide how many to perturb
    n = min(rng.integers(1, n_params + 1), len(non_bip_keys))

    # 30% chance of also perturbing BIP splits as a group
    do_bip = rng.random() < 0.30

    # Perturb non-BIP parameters
    selected = rng.choice(non_bip_keys, size=n, replace=False)
    changes = {}

    for key in selected:
        lo, hi, frac = PARAM_BOUNDS[key]
        current = cfg[key]
        spread = abs(current) * frac if current != 0 else (hi - lo) * frac
        new_val = current + rng.normal(0, spread)
        new_val = np.clip(new_val, lo, hi)
        # Keep integers for integer params
        if isinstance(base[key], int):
            new_val = int(round(new_val))
        else:
            new_val = float(new_val)
        cfg[key] = new_val
        changes[key] = (current, new_val)

    # Perturb BIP splits (maintain sum = 1.0)
    if do_bip:
        for bk in bip_keys:
            lo, hi, frac = PARAM_BOUNDS[bk]
            current = cfg[bk]
            spread = frac
            cfg[bk] = float(np.clip(current + rng.normal(0, spread), lo, hi))
        # Renormalize to sum to 1.0
        total = sum(cfg[bk] for bk in bip_keys)
        for bk in bip_keys:
            old = cfg[bk]
            cfg[bk] = float(cfg[bk] / total)
            changes[bk] = (base[bk], cfg[bk])

    # Hit shares: renormalize too
    hit_keys = {"hit_share_single", "hit_share_double", "hit_share_triple"}
    if any(k in changes for k in hit_keys):
        total = cfg["hit_share_single"] + cfg["hit_share_double"] + cfg["hit_share_triple"]
        if total > 0:
            for hk in hit_keys:
                cfg[hk] = float(cfg[hk] / total)

    return cfg, changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Hill-climbing optimizer")
    parser.add_argument("--n-experiments", type=int, default=100)
    parser.add_argument("--max-games", type=int, default=0,
                        help="Subsample games (0=all)")
    parser.add_argument("--max-perturb", type=int, default=3,
                        help="Max params to perturb per experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from autoresearch.train import CONFIG, run_experiment
    from autoresearch.prepare import evaluate, format_metrics

    rng = np.random.default_rng(args.seed)

    # Initialize with baseline
    logger.info("Running baseline experiment...")
    t0 = time.time()
    baseline_preds = run_experiment(CONFIG, max_games=args.max_games)
    baseline_metrics = evaluate(baseline_preds)
    baseline_score = baseline_metrics["composite"]
    baseline_elapsed = time.time() - t0

    logger.info("Baseline score: %.6f (%.1fs)", baseline_score, baseline_elapsed)
    print("\n=== BASELINE ===")
    print(format_metrics(baseline_metrics))

    best_score = baseline_score
    best_config = copy.deepcopy(CONFIG)
    best_metrics = baseline_metrics

    # Initialize CSV log
    fieldnames = [
        "experiment", "score", "delta", "accepted", "elapsed_s",
        "changes", "k_rmse", "k_brier", "bb_rmse", "h_rmse",
        "outs_rmse", "outs_brier", "ip_rmse",
        "k_cov_50", "k_cov_80", "k_cov_90",
    ]
    write_header = not LOG_PATH.exists()
    log_file = open(LOG_PATH, "a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # Write baseline
    writer.writerow({
        "experiment": 0, "score": baseline_score, "delta": 0.0,
        "accepted": True, "elapsed_s": baseline_elapsed,
        "changes": "baseline",
        "k_rmse": baseline_metrics.get("k_rmse", 0),
        "k_brier": baseline_metrics.get("k_avg_brier", 0),
        "bb_rmse": baseline_metrics.get("bb_rmse", 0),
        "h_rmse": baseline_metrics.get("h_rmse", 0),
        "outs_rmse": baseline_metrics.get("outs_rmse", 0),
        "outs_brier": baseline_metrics.get("outs_avg_brier", 0),
        "ip_rmse": baseline_metrics.get("ip_rmse", 0),
        "k_cov_50": baseline_metrics.get("k_cov_50", 0),
        "k_cov_80": baseline_metrics.get("k_cov_80", 0),
        "k_cov_90": baseline_metrics.get("k_cov_90", 0),
    })
    log_file.flush()

    # Run experiments
    n_improved = 0
    for i in range(1, args.n_experiments + 1):
        cfg, changes = perturb_config(
            best_config, rng, n_params=args.max_perturb,
        )
        change_str = ", ".join(
            f"{k}: {old:.4g}->{new:.4g}" for k, (old, new) in changes.items()
        )

        logger.info(
            "Experiment %d/%d: %s",
            i, args.n_experiments, change_str,
        )

        t0 = time.time()
        try:
            preds = run_experiment(cfg, max_games=args.max_games, random_seed=42 + i)
            metrics = evaluate(preds)
            score = metrics["composite"]
        except Exception as e:
            logger.warning("  Failed: %s", e)
            score = 0.0
            metrics = {}
        elapsed = time.time() - t0

        delta = score - best_score
        accepted = delta > 0.0001  # minimum improvement threshold

        if accepted:
            n_improved += 1
            best_score = score
            best_config = cfg
            best_metrics = metrics

            # Save best config
            with open(BEST_CONFIG_PATH, "w") as f:
                json.dump(best_config, f, indent=2)

            logger.info(
                "  IMPROVED: %.6f -> %.6f (+%.6f)  [%d improvements so far]",
                score - delta, score, delta, n_improved,
            )
        else:
            logger.info(
                "  rejected: %.6f (delta: %+.6f)", score, delta,
            )

        writer.writerow({
            "experiment": i, "score": score, "delta": delta,
            "accepted": accepted, "elapsed_s": elapsed,
            "changes": change_str,
            "k_rmse": metrics.get("k_rmse", 0),
            "k_brier": metrics.get("k_avg_brier", 0),
            "bb_rmse": metrics.get("bb_rmse", 0),
            "h_rmse": metrics.get("h_rmse", 0),
            "outs_rmse": metrics.get("outs_rmse", 0),
            "outs_brier": metrics.get("outs_avg_brier", 0),
            "ip_rmse": metrics.get("ip_rmse", 0),
            "k_cov_50": metrics.get("k_cov_50", 0),
            "k_cov_80": metrics.get("k_cov_80", 0),
            "k_cov_90": metrics.get("k_cov_90", 0),
        })
        log_file.flush()

    log_file.close()

    # Final summary
    print("\n" + "=" * 60)
    print("HILL CLIMBING COMPLETE")
    print("=" * 60)
    print(f"Experiments: {args.n_experiments}")
    print(f"Improvements: {n_improved}")
    print(f"Baseline score: {baseline_score:.6f}")
    print(f"Best score:     {best_score:.6f} ({best_score - baseline_score:+.6f})")
    print()
    print("=== BEST METRICS ===")
    print(format_metrics(best_metrics))
    print()
    print(f"Best config saved to: {BEST_CONFIG_PATH}")
    print(f"Experiment log: {LOG_PATH}")

    # Print changed parameters
    print("\n=== PARAMETER CHANGES (from baseline) ===")
    from autoresearch.train import CONFIG as original
    for key in sorted(best_config.keys()):
        if key in original and best_config[key] != original[key]:
            print(f"  {key}: {original[key]} -> {best_config[key]}")


if __name__ == "__main__":
    main()
