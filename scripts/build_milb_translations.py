"""
Build MiLB translation factors and translated season aggregates.

Usage:
    python scripts/build_milb_translations.py                   # full build
    python scripts/build_milb_translations.py --factors-only    # just factors
    python scripts/build_milb_translations.py --validate        # + spot checks
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_eng import build_milb_translated_data, CACHE_DIR
from src.data.milb_translation import (
    derive_batter_translation_factors,
    derive_pitcher_translation_factors,
    MILB_LEVELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_factors(factors_df: pd.DataFrame, label: str) -> None:
    """Pretty-print pooled translation factors."""
    pooled = factors_df[factors_df["pooled"]].copy()
    print(f"\n{'='*60}")
    print(f"  {label} Translation Factors (pooled across 2018-2025)")
    print(f"{'='*60}")

    for level in MILB_LEVELS:
        ldf = pooled[pooled["level"] == level]
        if ldf.empty:
            continue
        print(f"\n  {level}:")
        for _, row in ldf.iterrows():
            factor = row["factor"]
            n = row["n"]
            iqr = f"[{row.get('p25', np.nan):.3f} – {row.get('p75', np.nan):.3f}]"
            print(f"    {row['stat']:>8s}: {factor:.3f}  (n={n:>5d}, IQR {iqr})")


def print_year_stability(factors_df: pd.DataFrame, label: str) -> None:
    """Print per-year factor trends to check stability."""
    per_year = factors_df[~factors_df["pooled"]].copy()
    if per_year.empty:
        return

    print(f"\n{'='*60}")
    print(f"  {label} Year-over-Year Factor Stability")
    print(f"{'='*60}")

    for level in MILB_LEVELS:
        for stat in per_year["stat"].unique():
            subset = per_year[(per_year["level"] == level) & (per_year["stat"] == stat)]
            if subset.empty or len(subset) < 3:
                continue
            vals = subset.sort_values("season")
            factors = vals["factor"].dropna()
            if len(factors) < 3:
                continue
            cv = factors.std() / factors.mean() * 100
            print(
                f"  {level} {stat:>8s}: "
                f"range [{factors.min():.3f}–{factors.max():.3f}], "
                f"CV={cv:.1f}%, "
                f"n/yr={vals['n'].median():.0f}"
            )


def validate_known_players(
    translated_bat: pd.DataFrame,
    translated_pit: pd.DataFrame,
) -> None:
    """Spot-check translated rates for known recent call-ups."""
    from src.data.db import read_sql

    print(f"\n{'='*60}")
    print("  Validation: Known Players — Translated MiLB vs Actual MLB")
    print(f"{'='*60}")

    # Find players who debuted recently and have both MiLB + MLB data
    # Check a few manually: Jackson Holliday (702616), Paul Skenes (808902)
    check_players = {
        702616: "Jackson Holliday",
        808902: "Paul Skenes",
    }

    for pid, name in check_players.items():
        print(f"\n  {name} ({pid}):")

        # MiLB translated
        if pid in translated_bat["player_id"].values:
            bat = translated_bat[translated_bat["player_id"] == pid].sort_values(
                ["season", "level"]
            )
            for _, row in bat.iterrows():
                age_str = f"age {row['age_at_level']:.1f}" if pd.notna(row.get("age_at_level")) else ""
                print(
                    f"    MiLB {row['season']} {row['level']:>3s} ({row['pa']:>3.0f} PA, {age_str}): "
                    f"K%={row['raw_k_pct']:.3f}->{row['translated_k_pct']:.3f}  "
                    f"BB%={row['raw_bb_pct']:.3f}->{row['translated_bb_pct']:.3f}  "
                    f"ISO={row['raw_iso']:.3f}->{row['translated_iso']:.3f}  "
                    f"conf={row['translation_confidence']:.2f}"
                )

        if pid in translated_pit["player_id"].values:
            pit = translated_pit[translated_pit["player_id"] == pid].sort_values(
                ["season", "level"]
            )
            for _, row in pit.iterrows():
                age_str = f"age {row['age_at_level']:.1f}" if pd.notna(row.get("age_at_level")) else ""
                print(
                    f"    MiLB {row['season']} {row['level']:>3s} ({row['bf']:>3.0f} BF, {age_str}): "
                    f"K%={row['raw_k_pct']:.3f}->{row['translated_k_pct']:.3f}  "
                    f"BB%={row['raw_bb_pct']:.3f}->{row['translated_bb_pct']:.3f}  "
                    f"conf={row['translation_confidence']:.2f}"
                )

        # Actual MLB stats for comparison
        mlb_query = """
        SELECT
            dg.season,
            COUNT(*) AS pa,
            SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS k,
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE fp.batter_id = :pid AND dg.game_type = 'R'
        GROUP BY dg.season
        ORDER BY dg.season
        """
        mlb_df = read_sql(mlb_query, {"pid": pid})
        if len(mlb_df) > 0:
            for _, row in mlb_df.iterrows():
                if row["pa"] > 0:
                    print(
                        f"    MLB  {int(row['season'])}     ({row['pa']:>3d} PA): "
                        f"K%={row['k']/row['pa']:.3f}  "
                        f"BB%={row['bb']/row['pa']:.3f}"
                    )

        # Check pitcher stats too
        pit_query = """
        SELECT
            dg.season,
            SUM(sb.batters_faced) AS bf,
            SUM(sb.strike_outs) AS k,
            SUM(sb.walks) AS bb
        FROM staging.pitching_boxscores sb
        JOIN production.dim_game dg ON sb.game_pk = dg.game_pk
        WHERE sb.pitcher_id = :pid AND dg.game_type = 'R'
        GROUP BY dg.season
        ORDER BY dg.season
        """
        pit_df = read_sql(pit_query, {"pid": pid})
        if len(pit_df) > 0:
            for _, row in pit_df.iterrows():
                if row["bf"] and row["bf"] > 0:
                    print(
                        f"    MLB  {int(row['season'])}     ({row['bf']:>3.0f} BF): "
                        f"K%={row['k']/row['bf']:.3f}  "
                        f"BB%={row['bb']/row['bf']:.3f}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MiLB translation factors")
    parser.add_argument("--factors-only", action="store_true",
                        help="Only derive factors, skip translated aggregates")
    parser.add_argument("--validate", action="store_true",
                        help="Run spot-check validation on known players")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild (ignore caches)")
    args = parser.parse_args()

    # Derive factors
    logger.info("Deriving batter translation factors...")
    bat_factors = derive_batter_translation_factors()
    print_factors(bat_factors, "Batter")
    print_year_stability(bat_factors, "Batter")

    logger.info("Deriving pitcher translation factors...")
    pit_factors = derive_pitcher_translation_factors()
    print_factors(pit_factors, "Pitcher")
    print_year_stability(pit_factors, "Pitcher")

    if args.factors_only:
        # Still cache factors
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        bat_factors.to_parquet(CACHE_DIR / "milb_batter_factors.parquet", index=False)
        pit_factors.to_parquet(CACHE_DIR / "milb_pitcher_factors.parquet", index=False)
        logger.info("Factors cached. Skipping full translated aggregates.")
        return

    # Full build: aggregate + translate + age context
    logger.info("Building translated MiLB season aggregates...")
    translated_bat, translated_pit = build_milb_translated_data(
        force_rebuild=args.force,
    )

    print(f"\nTranslated batters: {len(translated_bat):,} player-season-levels")
    print(f"Translated pitchers: {len(translated_pit):,} player-season-levels")

    # Summary stats
    if len(translated_bat) > 0:
        print("\nBatter translated rate distributions:")
        for col in ["translated_k_pct", "translated_bb_pct", "translated_iso"]:
            vals = translated_bat[col].dropna()
            print(f"  {col}: mean={vals.mean():.3f}, median={vals.median():.3f}, "
                  f"std={vals.std():.3f}")

    if len(translated_pit) > 0:
        print("\nPitcher translated rate distributions:")
        for col in ["translated_k_pct", "translated_bb_pct"]:
            vals = translated_pit[col].dropna()
            print(f"  {col}: mean={vals.mean():.3f}, median={vals.median():.3f}, "
                  f"std={vals.std():.3f}")

    if args.validate:
        validate_known_players(translated_bat, translated_pit)

    logger.info("Done!")


if __name__ == "__main__":
    main()
