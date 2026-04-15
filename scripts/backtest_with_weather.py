#!/usr/bin/env python
"""Diagnostic backtest: retroactively apply the existing weather_effects
table to historical game_props and rerun the validator to measure the
calibration improvement that would have happened if weather had actually
been wired into the simulator.

Tests the hypothesis:
    If the k_multiplier / hr_multiplier in weather_effects.parquet had
    been applied per game during prediction, would Brier/ECE improve on
    completed 2026 games?

The script is read-only on the inputs and writes a single derived parquet
(game_props_with_weather.parquet) that mirrors the original schema.

Usage
-----
    python scripts/backtest_with_weather.py
    python scripts/validate_daily_props.py \
        --props-path ../tdd-dashboard/data/dashboard/game_props_with_weather.parquet \
        --side pitcher --detail

Only Pitcher K and Pitcher HR rows are shifted — these are the two stats
where the weather_effects multipliers apply directly. Other stats are
copied through unchanged so the validator can compute a fair diff.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db import read_sql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(
    r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard"
)


def _temp_bucket(temp) -> str:
    """Match the SQL bucketing in src/data/queries/environment.py:291-296."""
    if pd.isna(temp):
        return "unknown"
    t = float(temp)
    if t < 55:
        return "cold"
    if t <= 69:
        return "cool"
    if t <= 84:
        return "warm"
    return "hot"


def main() -> None:
    # --- Load props ---
    gp = pd.read_parquet(DASHBOARD_DIR / "game_props.parquet")
    logger.info("Loaded %d rows from game_props.parquet", len(gp))
    gp = gp[(gp["game_status"] == "final") & gp["actual"].notna()].copy()
    logger.info("%d rows with final status and actuals", len(gp))

    # --- Load weather effects lookup ---
    we = pd.read_parquet(DASHBOARD_DIR / "weather_effects.parquet")
    wx_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for _, r in we.iterrows():
        wx_lookup[(str(r["temp_bucket"]), str(r["wind_category"]))] = (
            float(r["k_multiplier"]),
            float(r["hr_multiplier"]),
        )
    logger.info("Loaded %d weather effect combinations", len(wx_lookup))

    # --- Query dim_weather for all game_pks in the frame ---
    game_pks = sorted(int(x) for x in gp["game_pk"].unique())
    if not game_pks:
        logger.error("No game_pks to query")
        return

    dw = read_sql(
        f"""
        SELECT game_pk, temperature, wind_category, is_dome
        FROM production.dim_weather
        WHERE game_pk IN ({','.join(str(g) for g in game_pks)})
        """
    )
    logger.info(
        "Fetched %d dim_weather rows for %d distinct game_pks "
        "(missing: %d)",
        len(dw), dw["game_pk"].nunique(), len(game_pks) - dw["game_pk"].nunique(),
    )

    dw["temp_bucket"] = dw["temperature"].apply(_temp_bucket)
    weather_by_gpk = {
        int(r["game_pk"]): (
            str(r["temp_bucket"]),
            str(r["wind_category"]),
            bool(r["is_dome"]),
        )
        for _, r in dw.iterrows()
    }

    # --- Collect numeric p_over_X.X columns (ignore p_over_low/mid/high) ---
    import re
    _p_over_re = re.compile(r"^p_over_(\d+\.\d+)$")
    p_over_cols = []
    lines_list: list[float] = []
    for c in gp.columns:
        m = _p_over_re.match(c)
        if m:
            p_over_cols.append(c)
            lines_list.append(float(m.group(1)))
    # Sort by line
    order = np.argsort(lines_list)
    p_over_cols = [p_over_cols[i] for i in order]
    lines = np.array([lines_list[i] for i in order], dtype=float)
    logger.info(
        "p_over columns: %d (lines %.1f - %.1f)",
        len(p_over_cols), lines.min(), lines.max(),
    )

    # --- Apply the weather shift row-by-row for K and HR ---
    adjusted = gp.copy()
    n_shifted = 0
    n_skipped_dome = 0
    n_skipped_no_weather = 0
    n_skipped_no_lookup = 0
    n_skipped_neutral = 0
    shift_magnitudes: list[float] = []
    p_over_shifts: list[float] = []

    target_stats = ("K", "HR")

    for idx, row in gp.iterrows():
        stat = row["stat"]
        if stat not in target_stats:
            continue
        # Only pitcher K and HR (batters don't have matching semantics in the
        # table — batter HR isn't stored and batter K uses different rates)
        if row.get("player_type") != "pitcher":
            continue

        gpk = int(row["game_pk"])
        weather = weather_by_gpk.get(gpk)
        if weather is None:
            n_skipped_no_weather += 1
            continue
        temp_bucket, wind_category, is_dome = weather
        if is_dome:
            n_skipped_dome += 1
            continue

        mults = wx_lookup.get((temp_bucket, wind_category))
        if mults is None:
            n_skipped_no_lookup += 1
            continue
        k_mult, hr_mult = mults
        mult = k_mult if stat == "K" else hr_mult
        if abs(mult - 1.0) < 1e-6:
            n_skipped_neutral += 1
            continue

        expected = float(row["expected"]) if pd.notna(row["expected"]) else 0.0
        delta = (mult - 1.0) * expected
        if abs(delta) < 1e-4:
            n_skipped_neutral += 1
            continue

        # Shift the p_over curve by delta in count space:
        # new_p_over(line) ~= original_p_over(line - delta).
        # Clamp shifted lines to [min_line, max_line]. For integer-count
        # stats, this is mathematically correct: p_over(0.39) = p_over(0.5)
        # because count > 0.39 iff count >= 1 iff count > 0.5.
        row_vals = np.array(
            [row[c] for c in p_over_cols], dtype=float,
        )
        # Drop NaN positions (older rows may lack widened lines)
        valid = ~np.isnan(row_vals)
        if valid.sum() < 2:
            continue
        vx = lines[valid]
        vy = np.clip(row_vals[valid], 0.0, 1.0)

        shifted_lines = np.clip(lines - delta, vx[0], vx[-1])
        new_vals = np.interp(shifted_lines, vx, vy)
        new_vals = np.clip(new_vals, 0.0, 1.0)
        # Preserve NaN positions in output so downstream validator still
        # filters those out cleanly.
        for i, col in enumerate(p_over_cols):
            if valid[i]:
                adjusted.at[idx, col] = float(new_vals[i])

        # Also shift the row-level p_over at the row's default line
        if pd.notna(row.get("line")):
            default_line = float(row["line"])
            shifted = float(np.clip(default_line - delta, vx[0], vx[-1]))
            new_p = float(np.interp(shifted, vx, vy))
            new_p = float(np.clip(new_p, 0.0, 1.0))
            orig_p = float(row["p_over"]) if pd.notna(row.get("p_over")) else np.nan
            adjusted.at[idx, "p_over"] = new_p
            if not np.isnan(orig_p):
                p_over_shifts.append(new_p - orig_p)

        n_shifted += 1
        shift_magnitudes.append(abs(delta))

    # --- Save ---
    out_path = DASHBOARD_DIR / "game_props_with_weather.parquet"
    adjusted.to_parquet(out_path, index=False)

    # --- Report ---
    print()
    print("=" * 75)
    print("WEATHER BACKTEST SUMMARY")
    print("=" * 75)
    pitcher_kr_mask = (gp["stat"].isin(target_stats)) & (gp["player_type"] == "pitcher")
    print(f"Pitcher K/HR rows in scope: {pitcher_kr_mask.sum()}")
    print(f"  shifted:             {n_shifted}")
    print(f"  skipped dome:        {n_skipped_dome}")
    print(f"  skipped no weather:  {n_skipped_no_weather}")
    print(f"  skipped no lookup:   {n_skipped_no_lookup}")
    print(f"  skipped neutral/tiny: {n_skipped_neutral}")
    if shift_magnitudes:
        sm = np.array(shift_magnitudes)
        print(
            f"\nCount-space shift |delta|: "
            f"mean={sm.mean():.3f}  median={np.median(sm):.3f}  max={sm.max():.3f}"
        )
    if p_over_shifts:
        ps = np.array(p_over_shifts)
        print(
            f"Row-level p_over delta: "
            f"mean={ps.mean():+.4f}  "
            f"abs mean={np.abs(ps).mean():.4f}  "
            f"max |delta|={np.abs(ps).max():.4f}"
        )
    print(f"\nWrote {out_path}")
    print()
    print("Now run:")
    print(
        f"  python scripts/validate_daily_props.py "
        f"--props-path \"{out_path}\" --side pitcher --detail"
    )


if __name__ == "__main__":
    main()
