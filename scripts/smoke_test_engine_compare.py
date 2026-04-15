#!/usr/bin/env python
"""Synthetic smoke test for the engine-comparison machinery.

Builds fake 30-game PA-sim and legacy-schema prediction frames, runs
them through the adapter + metrics + diff code to confirm columns,
scoring, and CSV output all work end-to-end. Not a real backtest —
just a plumbing check before we spend hours on the real run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.game_prop_validation import (
    PITCHER_PROP_CONFIGS,
    compute_game_prop_metrics,
)
from src.evaluation.pa_sim_prop_adapter import (
    compute_engine_diff,
    pa_sim_frame_to_legacy_schema,
)


def _make_sim_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Synthesize a unified game-sim frame with the expected columns."""
    rows = []
    for i in range(n):
        # Latent true talent, noisy actuals
        true_k_rate = rng.uniform(0.18, 0.30)
        bf = int(rng.integers(20, 27))
        actual_k = rng.binomial(bf, true_k_rate)
        expected_k = bf * true_k_rate + rng.normal(0, 0.3)
        std_k = 1.8

        actual_bb = rng.binomial(bf, rng.uniform(0.06, 0.11))
        expected_bb = bf * 0.085 + rng.normal(0, 0.2)
        std_bb = 1.2

        actual_h = rng.binomial(bf, rng.uniform(0.18, 0.28))
        expected_h = bf * 0.23 + rng.normal(0, 0.4)
        std_h = 1.9

        actual_hr = rng.binomial(bf, 0.032)
        expected_hr = bf * 0.031 + rng.normal(0, 0.1)
        std_hr = 0.7

        actual_outs = bf - actual_bb - actual_hr  # rough
        expected_outs = bf * 0.64 + rng.normal(0, 0.5)
        std_outs = 2.0

        row = {
            "game_pk": 700000 + i,
            "pitcher_id": 600000 + (i % 15),
            "test_season": 2025,
            "expected_k": expected_k, "std_k": std_k, "actual_k": actual_k,
            "expected_bb": expected_bb, "std_bb": std_bb, "actual_bb": actual_bb,
            "expected_h": expected_h, "std_h": std_h, "actual_h": actual_h,
            "expected_hr": expected_hr, "std_hr": std_hr, "actual_hr": actual_hr,
            "expected_outs": expected_outs, "std_outs": std_outs,
            "actual_outs": actual_outs,
            "expected_bf": bf, "std_bf": 2.0, "actual_bf": bf,
            "pitcher_rate_mean": true_k_rate,
            "matchup_logit_lift": rng.normal(0, 0.05),
            "umpire_lift": rng.normal(0, 0.03),
            "weather_lift": rng.normal(0, 0.02),
            "park_factor": rng.uniform(0.9, 1.1),
            "rest_bucket": "normal",
        }
        # Per-line p_over for each stat
        for stat, mean_c, std_c in [
            ("k", expected_k, std_k), ("bb", expected_bb, std_bb),
            ("h", expected_h, std_h), ("hr", expected_hr, std_hr),
            ("outs", expected_outs, std_outs),
        ]:
            cfg = PITCHER_PROP_CONFIGS[stat]
            for line in cfg.default_lines:
                # Crude Gaussian P(X > line)
                from scipy.stats import norm
                p = float(1 - norm.cdf(line, loc=mean_c, scale=std_c))
                row[f"p_{stat}_over_{line:.1f}"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def _make_legacy_frame(
    sim_df: pd.DataFrame, stat: str, rng: np.random.Generator,
) -> pd.DataFrame:
    """Synthesize a legacy per-stat frame on the same games, slightly worse.

    Legacy has narrower distributions for H/Outs — simulate that by
    adding a small calibration tilt.
    """
    sn = stat
    cfg = PITCHER_PROP_CONFIGS[stat]
    actual_col = f"actual_{sn}"
    exp_col = f"expected_{sn}"
    std_col = f"std_{sn}"

    rows = []
    for _, r in sim_df.iterrows():
        # Legacy expected: similar mean, narrower std
        legacy_exp = float(r[exp_col]) + rng.normal(0, 0.1)
        legacy_std = float(r[std_col]) * 0.85

        rec = {
            "game_pk": int(r["game_pk"]),
            "pitcher_id": int(r["pitcher_id"]),
            "season": 2025,
            f"expected_{sn}": legacy_exp,
            f"std_{sn}": legacy_std,
            f"actual_{sn}": r[actual_col],
            "bf_mu": r["expected_bf"],
            "pitcher_rate_mean": r["pitcher_rate_mean"],
            "matchup_logit_lift": r["matchup_logit_lift"],
            "umpire_lift": r["umpire_lift"],
            "weather_lift": r["weather_lift"],
            "park_factor": r["park_factor"],
            "rest_bucket": "normal",
        }
        for line in cfg.default_lines:
            from scipy.stats import norm
            p = float(1 - norm.cdf(line, loc=legacy_exp, scale=legacy_std))
            rec[f"p_over_{line:.1f}".replace(".", "_")] = p
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(123)
    n = 30

    sim_df = _make_sim_frame(n, rng)
    print(f"Synthetic sim frame: {len(sim_df)} rows, "
          f"{len(sim_df.columns)} cols")

    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "smoke_engine"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_diff_rows = []
    print("\n" + "=" * 80)
    print(f"{'stat':<8}{'engine':<10}{'n':>5}  "
          f"{'Brier@mid':>10}  {'ECE@mid':>9}")
    print("-" * 80)
    for stat in ("k", "bb", "h", "hr", "outs"):
        cfg = PITCHER_PROP_CONFIGS[stat]
        pa_sim_df = pa_sim_frame_to_legacy_schema(sim_df, stat, cfg.default_lines)
        legacy_df = _make_legacy_frame(sim_df, stat, rng)

        m_sim = compute_game_prop_metrics(cfg, pa_sim_df)
        m_leg = compute_game_prop_metrics(cfg, legacy_df)

        # Midline
        mid_line = cfg.default_lines[len(cfg.default_lines) // 2]
        b_sim = m_sim["brier_scores"].get(mid_line, float("nan"))
        b_leg = m_leg["brier_scores"].get(mid_line, float("nan"))
        e_sim = m_sim["ece_per_line"].get(mid_line, float("nan"))
        e_leg = m_leg["ece_per_line"].get(mid_line, float("nan"))

        print(f"{stat:<8}{'pa_sim':<10}{m_sim['n_games']:>5}  "
              f"{b_sim:>10.4f}  {e_sim:>9.4f}")
        print(f"{stat:<8}{'legacy':<10}{m_leg['n_games']:>5}  "
              f"{b_leg:>10.4f}  {e_leg:>9.4f}")

        diff = compute_engine_diff(
            legacy_df, pa_sim_df, stat, cfg.default_lines,
        )
        if not diff.empty:
            diff["prop"] = f"pitcher_{stat}"
            all_diff_rows.append(diff)

        # Dump
        pa_sim_df.to_parquet(out_dir / f"pitcher_{stat}_pa_sim.parquet",
                             index=False)
        legacy_df.to_parquet(out_dir / f"pitcher_{stat}_legacy.parquet",
                             index=False)

    if all_diff_rows:
        combined = pd.concat(all_diff_rows, ignore_index=True)
        diff_path = out_dir / "engine_diff_smoke.csv"
        combined.to_csv(diff_path, index=False)
        print("\n" + "=" * 80)
        print("ENGINE DIFF (pa_sim - legacy)")
        print("=" * 80)
        print(combined[[
            "prop", "line", "n_games",
            "legacy_brier", "pa_sim_brier", "brier_delta",
            "legacy_ece", "pa_sim_ece", "ece_delta",
        ]].to_string(index=False))
        print(f"\nSaved diff CSV: {diff_path}")

    # Column sanity
    print("\nSchema check — PA-sim K frame columns:")
    pa_k = pa_sim_frame_to_legacy_schema(sim_df, "k", [3.5, 4.5, 5.5, 6.5, 7.5])
    expected = {"game_pk", "pitcher_id", "season", "expected_k", "std_k",
                "actual_k", "p_over_3_5", "p_over_4_5", "p_over_5_5",
                "p_over_6_5", "p_over_7_5", "bf_mu"}
    missing = expected - set(pa_k.columns)
    print(f"  missing: {missing if missing else 'none'}")
    print(f"  extras:  {set(pa_k.columns) - expected}")


if __name__ == "__main__":
    main()
