#!/usr/bin/env python
"""
Prop-line calibration test for rolling form lifts.

This test evaluates whether form lifts improve over/under prop accuracy
using distributional metrics (Brier, log loss, calibration).

For each batter-game, computes P(over X.5) with and without form lifts,
then measures Brier score, log loss, and calibration stratified by
form lift magnitude.

Usage
-----
    python scripts/backtest_form_lift_props.py
    python scripts/backtest_form_lift_props.py --max-games 3000
    python scripts/backtest_form_lift_props.py --mode pitcher
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_form_lifts import (
    load_rolling_form_all,
    load_hard_hit_for_backtest,
    lookup_prior_form,
    lookup_prior_hard_hit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard")
SNAPSHOT_DIR = DASHBOARD_DIR / "snapshots"


def _load_npz(name: str) -> dict[str, np.ndarray]:
    for d in [SNAPSHOT_DIR, DASHBOARD_DIR]:
        path = d / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            return {k: data[k] for k in data.files}
    return {}


# ---------------------------------------------------------------------------
# Batter prop calibration
# ---------------------------------------------------------------------------
def run_batter_prop_calibration(
    season: int = 2025,
    max_games: int = 2000,
    n_sims: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run batter sim with/without form lifts and capture prop probabilities."""
    from src.data.db import read_sql
    from src.models.game_sim.batter_simulator import simulate_batter_game
    from src.models.game_sim.form_model import (
        compute_batter_form_lifts, BatterFormLifts,
    )

    logger.info("=" * 60)
    logger.info("Batter prop calibration: season=%d, max=%d", season, max_games)

    games = read_sql(f"""
    SELECT fpg.player_id AS batter_id, fpg.game_pk, fpg.game_date,
           fpg.bat_k AS actual_k, fpg.bat_bb AS actual_bb,
           fpg.bat_hr AS actual_hr, fpg.bat_h AS actual_h,
           fpg.bat_pa AS actual_pa, fpg.bat_tb AS actual_tb
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
    WHERE fpg.player_role = 'batter'
      AND fpg.bat_pa >= 3
      AND dg.season = {season}
      AND dg.game_type = 'R'
      AND fpg.game_date >= '{season}-06-01'
    ORDER BY RANDOM()
    LIMIT {max_games}
    """)
    logger.info("Loaded %d batter-game rows", len(games))

    k_npz = _load_npz("hitter_k_samples")
    bb_npz = _load_npz("hitter_bb_samples")
    hr_npz = _load_npz("hitter_hr_samples")

    batter_ids = games["batter_id"].unique().tolist()
    rolling_data = load_rolling_form_all(batter_ids, "batter", season)
    hh_data = load_hard_hit_for_backtest(batter_ids, season)
    logger.info("Rolling: %d batters, HH: %d", len(rolling_data), len(hh_data))

    # Prop lines to evaluate
    k_lines = [0.5, 1.5]
    hr_lines = [0.5]
    h_lines = [0.5, 1.5]
    tb_lines = [0.5, 1.5, 2.5]

    results = []
    n_skipped = 0
    for i, (_, row) in enumerate(games.iterrows()):
        if (i + 1) % 500 == 0:
            logger.info("  Progress: %d / %d", i + 1, len(games))

        bid = int(row["batter_id"])
        gpk = int(row["game_pk"])
        gdate = str(row["game_date"])
        bid_str = str(bid)

        k_samp = k_npz.get(bid_str)
        bb_samp = bb_npz.get(bid_str)
        hr_samp = hr_npz.get(bid_str)
        if k_samp is None or bb_samp is None or hr_samp is None:
            n_skipped += 1
            continue

        # Compute form lift
        rolling_row = lookup_prior_form(rolling_data, bid, gdate)
        if rolling_row is not None:
            hh_pct, hh_bip = lookup_prior_hard_hit(hh_data, bid, gdate)
            bat_form = compute_batter_form_lifts(
                rolling_row, hard_hit_pct=hh_pct, hard_hit_bip=hh_bip,
            )
        else:
            bat_form = BatterFormLifts()

        shared = dict(
            batter_k_rate_samples=k_samp,
            batter_bb_rate_samples=bb_samp,
            batter_hr_rate_samples=hr_samp,
            batting_order=4,
            starter_k_rate=0.226,
            starter_bb_rate=0.082,
            starter_hr_rate=0.031,
            starter_bf_mu=24.0,
            starter_bf_sigma=3.0,
            n_sims=n_sims,
            random_seed=random_seed + gpk % 10000 + bid % 1000,
        )

        try:
            sim_base = simulate_batter_game(**shared)
            sim_form = simulate_batter_game(
                **shared,
                form_k_lift=bat_form.k_lift,
                form_bb_lift=bat_form.bb_lift,
                form_hr_lift=bat_form.hr_lift + bat_form.hh_lift,
            )
        except Exception:
            n_skipped += 1
            continue

        rec = {
            "batter_id": bid,
            "game_pk": gpk,
            "form_k_lift": bat_form.k_lift,
            "form_bb_lift": bat_form.bb_lift,
            "form_hr_lift": bat_form.hr_lift + bat_form.hh_lift,
            "abs_total_lift": abs(bat_form.k_lift) + abs(bat_form.hr_lift + bat_form.hh_lift),
            "actual_k": int(row["actual_k"]),
            "actual_hr": int(row["actual_hr"]),
            "actual_h": int(row["actual_h"]),
            "actual_tb": int(row["actual_tb"]),
        }

        # Capture P(over) for each prop line
        for stat, lines, sim_b, sim_f in [
            ("k", k_lines, sim_base, sim_form),
            ("hr", hr_lines, sim_base, sim_form),
            ("h", h_lines, sim_base, sim_form),
            ("tb", tb_lines, sim_base, sim_form),
        ]:
            samp_b = getattr(sim_b, f"{stat}_samples")
            samp_f = getattr(sim_f, f"{stat}_samples")
            for line in lines:
                p_base = float(np.mean(samp_b > line))
                p_form = float(np.mean(samp_f > line))
                rec[f"p_over_{stat}_{line:.1f}_base"] = p_base
                rec[f"p_over_{stat}_{line:.1f}_form"] = p_form

        results.append(rec)

    logger.info("Completed: %d games, %d skipped", len(results), n_skipped)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Pitcher prop calibration
# ---------------------------------------------------------------------------
def run_pitcher_prop_calibration(
    season: int = 2025,
    max_games: int = 1500,
    n_sims: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run pitcher sim with/without form lifts and capture prop probabilities."""
    from src.data.db import read_sql
    from src.models.game_sim.exit_model import ExitModel
    from src.models.game_sim.simulator import simulate_game
    from src.models.game_sim.form_model import (
        compute_pitcher_form_lifts, PitcherFormLifts,
    )

    logger.info("=" * 60)
    logger.info("Pitcher prop calibration: season=%d, max=%d", season, max_games)

    games = read_sql(f"""
    SELECT fpg.player_id AS pitcher_id, fpg.game_pk, fpg.game_date,
           fpg.pit_k AS actual_k, fpg.pit_bb AS actual_bb,
           fpg.pit_hr AS actual_hr, fpg.pit_h AS actual_h,
           fpg.pit_bf AS actual_bf
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
    WHERE fpg.player_role = 'pitcher'
      AND fpg.pit_is_starter = TRUE
      AND fpg.pit_bf >= 10
      AND dg.season = {season}
      AND dg.game_type = 'R'
      AND fpg.game_date >= '{season}-06-01'
    ORDER BY RANDOM()
    LIMIT {max_games}
    """)
    logger.info("Loaded %d pitcher starts", len(games))

    k_npz = _load_npz("pitcher_k_samples")
    bb_npz = _load_npz("pitcher_bb_samples")
    hr_npz = _load_npz("pitcher_hr_samples")

    exit_model = ExitModel()
    exit_model.load(DASHBOARD_DIR / "exit_model.pkl")

    pitcher_ids = games["pitcher_id"].unique().tolist()
    rolling_data = load_rolling_form_all(pitcher_ids, "pitcher", season)
    logger.info("Rolling form: %d pitchers", len(rolling_data))

    k_lines = [3.5, 4.5, 5.5, 6.5, 7.5]
    bb_lines = [1.5, 2.5]

    results = []
    n_skipped = 0
    for i, (_, row) in enumerate(games.iterrows()):
        if (i + 1) % 200 == 0:
            logger.info("  Progress: %d / %d", i + 1, len(games))

        pid = int(row["pitcher_id"])
        gpk = int(row["game_pk"])
        gdate = str(row["game_date"])
        pid_str = str(pid)

        k_samp = k_npz.get(pid_str)
        bb_samp = bb_npz.get(pid_str)
        hr_samp = hr_npz.get(pid_str)
        if k_samp is None:
            n_skipped += 1
            continue

        rng_fb = np.random.default_rng(random_seed + pid)
        if bb_samp is None:
            bb_samp = rng_fb.beta(8, 90, size=len(k_samp))
        if hr_samp is None:
            hr_samp = rng_fb.beta(3, 95, size=len(k_samp))

        rolling_row = lookup_prior_form(rolling_data, pid, gdate)
        if rolling_row is not None:
            pit_form = compute_pitcher_form_lifts(rolling_row)
        else:
            pit_form = PitcherFormLifts()

        shared = dict(
            pitcher_k_rate_samples=k_samp,
            pitcher_bb_rate_samples=bb_samp,
            pitcher_hr_rate_samples=hr_samp,
            lineup_matchup_lifts={"k": np.zeros(9), "bb": np.zeros(9), "hr": np.zeros(9)},
            tto_lifts={"k": np.zeros(3), "bb": np.zeros(3), "hr": np.zeros(3)},
            pitcher_ppa_adj=0.0,
            batter_ppa_adjs=np.zeros(9),
            exit_model=exit_model,
            n_sims=n_sims,
            random_seed=random_seed + gpk % 10000,
        )

        try:
            sim_base = simulate_game(**shared, form_bb_lift=0.0)
            sim_form = simulate_game(**shared, form_bb_lift=pit_form.bb_lift)
        except Exception:
            n_skipped += 1
            continue

        rec = {
            "pitcher_id": pid,
            "game_pk": gpk,
            "form_bb_lift": pit_form.bb_lift,
            "abs_bb_lift": abs(pit_form.bb_lift),
            "actual_k": int(row["actual_k"]),
            "actual_bb": int(row["actual_bb"]),
        }

        sb = sim_base.summary()
        sf = sim_form.summary()
        rec["pred_k_base"] = sb["k"]["mean"]
        rec["pred_k_form"] = sf["k"]["mean"]
        rec["pred_bb_base"] = sb["bb"]["mean"]
        rec["pred_bb_form"] = sf["bb"]["mean"]

        for line in k_lines:
            rec[f"p_over_k_{line:.1f}_base"] = float(
                np.mean(sim_base.k_samples > line)
            )
            rec[f"p_over_k_{line:.1f}_form"] = float(
                np.mean(sim_form.k_samples > line)
            )

        for line in bb_lines:
            rec[f"p_over_bb_{line:.1f}_base"] = float(
                np.mean(sim_base.bb_samples > line)
            )
            rec[f"p_over_bb_{line:.1f}_form"] = float(
                np.mean(sim_form.bb_samples > line)
            )

        results.append(rec)

    logger.info("Completed: %d games, %d skipped", len(results), n_skipped)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_prop_calibration(
    df: pd.DataFrame,
    stat: str,
    lines: list[float],
    lift_col: str,
    label: str,
) -> pd.DataFrame:
    """Compute Brier score and calibration for base vs form, stratified."""
    all_rows = []

    for line in lines:
        base_col = f"p_over_{stat}_{line:.1f}_base"
        form_col = f"p_over_{stat}_{line:.1f}_form"
        actual_col = f"actual_{stat}"

        if base_col not in df.columns:
            continue

        actual_over = (df[actual_col] > line).astype(int).values
        p_base = df[base_col].values
        p_form = df[form_col].values

        # Overall Brier
        brier_base = brier_score_loss(actual_over, p_base)
        brier_form = brier_score_loss(actual_over, p_form)

        all_rows.append({
            "line": f"{stat} > {line}",
            "group": "ALL",
            "n": len(df),
            "actual_rate": float(actual_over.mean()),
            "brier_base": round(brier_base, 6),
            "brier_form": round(brier_form, 6),
            "brier_delta": round(brier_form - brier_base, 6),
            "brier_pct": round((brier_form - brier_base) / brier_base * 100, 3),
        })

        # Stratify by lift magnitude
        abs_lift = df[lift_col].abs()
        for group_name, mask in [
            ("no_lift", abs_lift < 0.001),
            ("small_lift", (abs_lift >= 0.001) & (abs_lift < 0.01)),
            ("large_lift", abs_lift >= 0.01),
        ]:
            subset = df[mask]
            if len(subset) < 30:
                continue

            ao = (subset[actual_col] > line).astype(int).values
            pb = subset[base_col].values
            pf = subset[form_col].values

            bs_b = brier_score_loss(ao, pb)
            bs_f = brier_score_loss(ao, pf)

            all_rows.append({
                "line": f"{stat} > {line}",
                "group": group_name,
                "n": len(subset),
                "actual_rate": float(ao.mean()),
                "brier_base": round(bs_b, 6),
                "brier_form": round(bs_f, 6),
                "brier_delta": round(bs_f - bs_b, 6),
                "brier_pct": round((bs_f - bs_b) / bs_b * 100, 3) if bs_b > 0 else 0,
            })

    result = pd.DataFrame(all_rows)
    logger.info("\n%s Prop Calibration:\n%s", label, result.to_string(index=False))
    return result


def analyze_calibration_bins(
    df: pd.DataFrame,
    stat: str,
    line: float,
    label: str,
    n_bins: int = 10,
) -> None:
    """Show calibration curve: predicted prob bins vs actual frequency."""
    base_col = f"p_over_{stat}_{line:.1f}_base"
    form_col = f"p_over_{stat}_{line:.1f}_form"
    actual_col = f"actual_{stat}"

    if base_col not in df.columns:
        return

    actual_over = (df[actual_col] > line).astype(int)

    logger.info("\n  %s calibration bins for %s > %.1f:", label, stat, line)
    logger.info("  %8s  %6s  %8s  %8s  %8s", "bin", "n", "actual", "base", "form")

    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (df[base_col] >= lo) & (df[base_col] < hi)
        n = mask.sum()
        if n < 10:
            continue
        act = actual_over[mask].mean()
        pb = df.loc[mask, base_col].mean()
        pf = df.loc[mask, form_col].mean()
        logger.info(
            "  %3.0f-%3.0f%%  %6d  %8.3f  %8.3f  %8.3f",
            lo * 100, hi * 100, n, act, pb, pf,
        )


def analyze_edge_cases(
    df: pd.DataFrame,
    stat: str,
    line: float,
    lift_col: str,
    label: str,
) -> None:
    """Check if form lifts improve accuracy on close-call props."""
    base_col = f"p_over_{stat}_{line:.1f}_base"
    form_col = f"p_over_{stat}_{line:.1f}_form"
    actual_col = f"actual_{stat}"

    if base_col not in df.columns:
        return

    actual_over = (df[actual_col] > line).astype(int)

    # Close calls: baseline probability between 40-60% (coin-flip territory)
    close = (df[base_col] >= 0.40) & (df[base_col] <= 0.60)
    if close.sum() < 50:
        return

    subset = df[close]
    ao = actual_over[close].values

    # For close calls, did form lifts move probability in the right direction?
    p_base = subset[base_col].values
    p_form = subset[form_col].values
    shift = p_form - p_base  # positive = form says more likely over

    # When form shifted UP, was actual more likely to be over?
    shifted_up = shift > 0.001
    shifted_down = shift < -0.001
    no_shift = ~shifted_up & ~shifted_down

    logger.info(
        "\n  %s edge cases (%s > %.1f, base P in 40-60%%, n=%d):",
        label, stat, line, len(subset),
    )

    for name, mask in [
        ("form shifted UP", shifted_up),
        ("form shifted DOWN", shifted_down),
        ("no shift", no_shift),
    ]:
        n = mask.sum()
        if n < 10:
            continue
        hit_rate = ao[mask].mean()
        avg_base_p = p_base[mask].mean()
        avg_form_p = p_form[mask].mean()
        logger.info(
            "    %s (n=%d): actual_over=%.3f, avg_base_p=%.3f, avg_form_p=%.3f",
            name, n, hit_rate, avg_base_p, avg_form_p,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--max-games", type=int, default=2000)
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--mode", choices=["both", "pitcher", "batter"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    if args.mode in ("both", "batter"):
        bat_df = run_batter_prop_calibration(
            season=args.season, max_games=args.max_games,
            n_sims=args.n_sims, random_seed=args.seed,
        )
        if not bat_df.empty:
            bat_cal = analyze_prop_calibration(
                bat_df, "k", [0.5, 1.5], "form_k_lift", "Batter",
            )
            analyze_prop_calibration(
                bat_df, "hr", [0.5], "form_hr_lift", "Batter",
            )
            analyze_prop_calibration(
                bat_df, "h", [0.5, 1.5], "form_k_lift", "Batter",
            )
            analyze_prop_calibration(
                bat_df, "tb", [0.5, 1.5, 2.5], "form_hr_lift", "Batter",
            )

            analyze_calibration_bins(bat_df, "k", 0.5, "Batter")
            analyze_calibration_bins(bat_df, "hr", 0.5, "Batter")

            analyze_edge_cases(bat_df, "k", 0.5, "form_k_lift", "Batter")
            analyze_edge_cases(bat_df, "k", 1.5, "form_k_lift", "Batter")
            analyze_edge_cases(bat_df, "hr", 0.5, "form_hr_lift", "Batter")

            bat_df.to_csv(out_dir / "form_lift_batter_props.csv", index=False)

    if args.mode in ("both", "pitcher"):
        pit_df = run_pitcher_prop_calibration(
            season=args.season, max_games=args.max_games,
            n_sims=args.n_sims, random_seed=args.seed,
        )
        if not pit_df.empty:
            analyze_prop_calibration(
                pit_df, "k", [3.5, 4.5, 5.5, 6.5, 7.5],
                "form_bb_lift", "Pitcher",
            )
            analyze_prop_calibration(
                pit_df, "bb", [1.5, 2.5],
                "form_bb_lift", "Pitcher",
            )

            analyze_calibration_bins(pit_df, "k", 5.5, "Pitcher")
            analyze_calibration_bins(pit_df, "bb", 1.5, "Pitcher")

            analyze_edge_cases(pit_df, "bb", 1.5, "form_bb_lift", "Pitcher")
            analyze_edge_cases(pit_df, "bb", 2.5, "form_bb_lift", "Pitcher")

            pit_df.to_csv(out_dir / "form_lift_pitcher_props.csv", index=False)


if __name__ == "__main__":
    main()
