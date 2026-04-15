#!/usr/bin/env python
"""Validate market-style prop legs over recent completed games.

``market_legs.parquet`` is overwritten on each precompute run (today's slate
only). To score accuracy over many days, this script **reconstructs** the same
leg definitions as ``build_prop_legs`` in ``src/models/market_edge.py`` from
accumulated ``game_props.parquet``:

- Requires a book line on the row (``vegas_line`` from the precompute merge).
- Uses model P(over) at that line from the ``p_over_X.X`` grid (same lookup
  chain as the market-line backtest).
- Chooses over vs under whichever has higher probability (ties → over).
- Keeps legs with ``model_prob - 0.5 >= min_edge`` (default 3%, matching
  ``min_edge_vs_even=0.03``).

Hit rules match ``confident_picks`` backfill: over wins if ``actual > line``,
under wins if ``actual < line``; equality at the line counts as a loss.

Usage
-----
python scripts/validate_market_legs.py
python scripts/validate_market_legs.py --days 13
python scripts/validate_market_legs.py --props-path "D:/tdd-dashboard/data/dashboard/game_props.parquet"
python scripts/validate_market_legs.py --min-edge 0.03 --side pitcher
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from backtest_game_props_market_lines import _attach_p_over_at_book  # noqa: E402
from precompute import DASHBOARD_DIR  # noqa: E402
from src.evaluation.metrics import compute_ece  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Same stat keys as confident_picks._STAT_MAP / build_prop_legs
_STAT_MAP = {
    "K": "k",
    "BB": "bb",
    "HR": "hr",
    "H": "h",
    "Outs": "outs",
    "R": "r",
    "RBI": "rbi",
    "TB": "tb",
    "HRR": "hrr",
}


def _reconstruct_legs(
    df: pd.DataFrame,
    *,
    min_edge: float,
) -> pd.DataFrame:
    """Return one row per reconstructed leg with pick_hit and model_prob."""
    need = {"game_pk", "player_id", "stat", "vegas_line", "actual", "game_date"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"game_props missing columns: {sorted(missing)}")

    sub = df[
        df["vegas_line"].notna()
        & df["actual"].notna()
        & df["stat"].isin(_STAT_MAP.keys())
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["vegas_line"] = pd.to_numeric(sub["vegas_line"], errors="coerce")
    sub = sub[sub["vegas_line"].notna()].copy()
    sub["actual"] = pd.to_numeric(sub["actual"], errors="coerce")
    sub = sub[sub["actual"].notna()].copy()

    sub = _attach_p_over_at_book(sub, line_col="vegas_line")
    sub = sub.rename(columns={"p_over_market": "p_over_at_line"})
    sub = sub.dropna(subset=["p_over_at_line"])
    sub["p_over_at_line"] = np.clip(sub["p_over_at_line"].astype(float), 0.0, 1.0)
    sub["p_under_at_line"] = 1.0 - sub["p_over_at_line"]

    # Match build_prop_legs: stronger side; tie → over
    over_better = sub["p_over_at_line"] >= sub["p_under_at_line"]
    sub["side"] = np.where(over_better, "over", "under")
    sub["model_prob"] = np.where(
        over_better,
        sub["p_over_at_line"],
        sub["p_under_at_line"],
    )
    edge_mask = (sub["model_prob"] - 0.5) >= min_edge
    sub = sub.loc[edge_mask].copy()

    line = sub["vegas_line"].values
    act = sub["actual"].values
    over_hit = act > line
    under_hit = act < line
    sub["pick_hit"] = np.where(sub["side"].values == "over", over_hit, under_hit)

    return sub


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate market-leg style picks from game_props history",
    )
    parser.add_argument(
        "--props-path",
        type=Path,
        default=None,
        help="Path to game_props.parquet (default: dashboard DASHBOARD_DIR)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=13,
        help="Include completed games with game_date in the last N days "
        "(relative to the latest final game_date in the file; default: 13)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum model_prob - 0.5 to count as a leg (default: 0.03)",
    )
    parser.add_argument(
        "--side",
        choices=["all", "batter", "pitcher"],
        default="all",
        help="Restrict to batter or pitcher rows",
    )
    args = parser.parse_args()

    props_path = args.props_path or (DASHBOARD_DIR / "game_props.parquet")
    if not props_path.exists():
        logger.error("game_props.parquet not found at %s", props_path)
        sys.exit(1)

    df = pd.read_parquet(props_path)
    logger.info("Loaded %d rows from %s", len(df), props_path)

    if "game_status" in df.columns:
        df = df[df["game_status"].astype(str).str.lower() == "final"].copy()
    if df.empty:
        logger.warning("No final rows after game_status filter")
        sys.exit(0)

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"].notna()].copy()
    day = df["game_date"].dt.normalize()
    max_day = day.max()
    if pd.isna(max_day):
        logger.error("Could not parse game_date")
        sys.exit(1)
    cutoff = max_day - pd.Timedelta(days=args.days)
    df = df[day >= cutoff].copy()
    logger.info(
        "Date window: %s .. %s (%d rows, final only)",
        cutoff.strftime("%Y-%m-%d"),
        max_day.strftime("%Y-%m-%d"),
        len(df),
    )

    if "player_type" in df.columns:
        df["player_type"] = df["player_type"].astype(str).str.lower().str.strip()
        if args.side == "pitcher":
            df = df[df["player_type"] == "pitcher"].copy()
        elif args.side == "batter":
            df = df[df["player_type"] == "batter"].copy()

    if "vegas_line" not in df.columns:
        logger.error(
            "No vegas_line column in game_props — legs cannot be reconstructed. "
            "Run precompute with DK/PP fetch so lines merge into game_props.",
        )
        sys.exit(1)

    legs = _reconstruct_legs(df, min_edge=args.min_edge)
    if legs.empty:
        logger.warning(
            "No legs after filters (vegas_line + actual + min_edge=%.2f). "
            "Check that game_props rows include vegas_line for completed games.",
            args.min_edge,
        )
        sys.exit(0)

    y = legs["pick_hit"].astype(float).values
    p = legs["model_prob"].astype(float).values
    acc = float(y.mean())
    brier = float(brier_score_loss(y, p))
    ece = float(compute_ece(p, y))

    print("\n" + "=" * 72)
    print("MARKET LEGS (reconstructed from game_props)")
    print("=" * 72)
    print(f"  n_legs:        {len(legs)}")
    print(f"  hit_rate:      {acc:.3f}")
    print(f"  mean(model_p): {float(np.mean(p)):.3f}")
    print(f"  brier:         {brier:.4f}")
    print(f"  ece:           {ece:.4f}")
    print(f"  min_edge:      {args.min_edge:.2f}")

    print("\nBy player_type:")
    for pt, g in legs.groupby("player_type"):
        yy = g["pick_hit"].astype(float).values
        pp = g["model_prob"].astype(float).values
        if len(g) == 0:
            continue
        hit_m = float(yy.mean())
        if yy.std() > 0:
            br = float(brier_score_loss(yy, pp))
            print(f"  {pt:8s} n={len(g):4d}  hit={hit_m:.3f}  brier={br:.4f}")
        else:
            print(f"  {pt:8s} n={len(g):4d}  hit={hit_m:.3f}  brier=n/a (no variance)")

    print("\nBy stat:")
    for st, g in legs.groupby("stat"):
        yy = g["pick_hit"].astype(float).values
        pp = g["model_prob"].astype(float).values
        br = brier_score_loss(yy, pp) if yy.std() > 0 else float("nan")
        line = f"  {st:5s} n={len(g):4d}  hit={yy.mean():.3f}"
        if not np.isnan(br):
            line += f"  brier={br:.4f}"
        print(line)

    print("\nBy game_date (top 20 dates by volume):")
    legs["_d"] = pd.to_datetime(legs["game_date"]).dt.strftime("%Y-%m-%d")
    vc = legs["_d"].value_counts().sort_index()
    for d in vc.tail(20).index:
        g = legs[legs["_d"] == d]
        yy = g["pick_hit"].astype(float).values
        print(f"  {d}  n={len(g):3d}  hit={yy.mean():.3f}")


if __name__ == "__main__":
    main()
