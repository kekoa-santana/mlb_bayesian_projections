#!/usr/bin/env python
"""Backtest game props at book lines: game_props + DK + Prize Picks parquets.

Mirrors ``confident_picks`` for batters (DK then PP **standard**). Pitchers also
use PP **demon/goblin** rows when needed (PP rarely has standard pitcher Outs).
If ``player_id`` still does not match the slate, pitcher rows get a PP line via
``game_date_key`` + ``team`` + normalized ``player_name`` + ``stat``.

Uses model P(over) at ``book_line`` from ``p_over_{line}``, legacy low/mid/high,
or the nearest stored half-line (needed for Outs when the grid tops out at 10.5).

By default, uses the three most recent calendar dates among rows with actuals.
Override with ``--dates``.

Usage
-----
python scripts/backtest_game_props_market_lines.py
python scripts/backtest_game_props_market_lines.py --dates 2026-03-28 2026-03-29 2026-03-30
python scripts/backtest_game_props_market_lines.py --n-dates 3 --props-path "D:/tdd-dashboard/data/dashboard/game_props.parquet"
python scripts/backtest_game_props_market_lines.py --side pitcher --min-n-pitcher 1
python scripts/backtest_game_props_market_lines.py --model-lines --side pitcher  # pitcher props at model lines (no book lines needed)
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

from precompute import DASHBOARD_DIR  # noqa: E402
from src.evaluation.metrics import compute_ece, compute_temperature  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_LINE_KEYS = ["player_id", "player_type", "stat", "game_date_key", "book_line"]


def _empty_book_lines() -> pd.DataFrame:
    """Right-hand frame for a no-op left merge (typed columns)."""
    return pd.DataFrame(columns=list(_LINE_KEYS))


def _norm_player_name(name: object) -> str:
    """Lowercase, strip suffixes, for matching PP/DK names to game_props."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    t = str(name).lower().replace(".", "").strip()
    for suf in (" jr", " sr", " ii", " iii", " iv", " v"):
        if t.endswith(suf):
            t = t[: -len(suf)].strip()
    return t


def _pp_odds_rank(odds_type: object) -> int:
    """Lower is preferred (matches typical main line first)."""
    if pd.isna(odds_type):
        return 99
    order = {"standard": 0, "demon": 1, "goblin": 2}
    return order.get(str(odds_type).strip().lower(), 99)


def _p_over_column(line: float) -> str:
    """Parquet column name for P(over) at a half-point line."""
    x = float(line)
    x = round(x * 2) / 2.0
    return f"p_over_{x:.1f}"


def _float_match(a: float, b: object, tol: float = 1e-6) -> bool:
    if pd.isna(b):
        return False
    return abs(float(a) - float(b)) < tol


def _p_over_from_legacy_triplet(row: pd.Series, book: float) -> float:
    """Use p_over_low/mid/high when book line matches line_low/mid/high."""
    for lv_col, p_col in (
        ("line_low", "p_over_low"),
        ("line_mid", "p_over_mid"),
        ("line_high", "p_over_high"),
    ):
        if lv_col not in row.index or p_col not in row.index:
            continue
        if _float_match(book, row[lv_col]) and pd.notna(row[p_col]):
            return float(row[p_col])
    return float("nan")


def _p_over_nearest_grid(row: pd.Series, book: float) -> float:
    """Use closest stored ``p_over_X.X`` when book line is outside the saved grid (e.g. Outs)."""
    best_d = float("inf")
    best_p = float("nan")
    for c in row.index:
        if not isinstance(c, str) or not c.startswith("p_over_"):
            continue
        if c in ("p_over_low", "p_over_mid", "p_over_high"):
            continue
        try:
            lv = float(c.replace("p_over_", ""))
        except ValueError:
            continue
        if pd.isna(row[c]):
            continue
        d = abs(lv - book)
        if d < best_d:
            best_d = d
            best_p = float(row[c])
    return best_p


def _attach_p_over_at_book(df: pd.DataFrame, line_col: str = "book_line") -> pd.DataFrame:
    """Add ``p_over_market`` from grid, legacy triplet, or nearest grid column."""

    def lookup(row: pd.Series) -> float:
        book = float(row[line_col])
        col = _p_over_column(book)
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
        legacy = _p_over_from_legacy_triplet(row, book)
        if not np.isnan(legacy):
            return legacy
        return _p_over_nearest_grid(row, book)

    out = df.copy()
    out["p_over_market"] = out.apply(lookup, axis=1)
    return out


def _load_dk_lines(
    dashboard_dir: Path,
    dk_path: Path | None,
) -> pd.DataFrame:
    """DK rows with ``book_line`` and ``game_date_key`` (YYYY-MM-DD)."""
    if dk_path is not None:
        paths = [dk_path]
    else:
        paths = [
            dashboard_dir / "dk_props_history.parquet",
            dashboard_dir / "dk_props.parquet",
        ]
    dk = pd.DataFrame()
    for p in paths:
        if p.exists():
            dk = pd.read_parquet(p)
            logger.info("Loaded %d DK rows from %s", len(dk), p)
            break
    if dk.empty:
        return _empty_book_lines()
    need = {"player_id", "player_type", "stat", "line", "game_date"}
    missing = need - set(dk.columns)
    if missing:
        logger.error("DK file missing columns: %s", sorted(missing))
        return _empty_book_lines()
    # game_date_key only — omit raw game_date so merge does not clobber game_props.game_date
    keep_cols = ["player_id", "player_type", "stat", "line"]
    if "over_implied" in dk.columns:
        keep_cols.append("over_implied")
    if "over_odds" in dk.columns:
        keep_cols.append("over_odds")
    out = dk[keep_cols].copy()
    out["player_type"] = out["player_type"].astype(str).str.lower().str.strip()
    out["game_date_key"] = pd.to_datetime(dk["game_date"]).dt.strftime("%Y-%m-%d")
    out = out.rename(columns={
        "line": "book_line",
        "over_implied": "book_implied",
        "over_odds": "book_odds",
    })
    out = out.drop_duplicates(
        subset=["player_id", "player_type", "stat", "game_date_key"],
        keep="first",
    )
    return out


def _read_pp_raw(dashboard_dir: Path, pp_path: Path | None) -> pd.DataFrame:
    """Load full PP parquet (used for player_id merge + pitcher fallbacks)."""
    if pp_path is not None:
        paths = [pp_path]
    else:
        paths = [
            dashboard_dir / "pp_props_history.parquet",
            dashboard_dir / "pp_props.parquet",
        ]
    for p in paths:
        if p.exists():
            pp = pd.read_parquet(p)
            logger.info("Loaded %d PP rows from %s", len(pp), p)
            return pp
    return pd.DataFrame()


def _load_pp_lines(
    pp: pd.DataFrame,
) -> pd.DataFrame:
    """PP rows for ``player_id`` merge: batters standard-only; pitchers standard|demon|goblin.

    Pitcher ``Outs`` (and many K/BB/H rows) only exist as demon/goblin on Prize Picks,
    so we include those with odds-type priority: standard, then demon, then goblin.
    """
    if pp.empty:
        return _empty_book_lines()

    need = {"player_id", "player_type", "stat", "line", "game_date"}
    missing = need - set(pp.columns)
    if missing:
        logger.error("PP file missing columns: %s", sorted(missing))
        return _empty_book_lines()

    pp = pp.copy()
    pp["game_date_key"] = pd.to_datetime(pp["game_date"]).dt.strftime("%Y-%m-%d")

    bat = pp.loc[pp["player_type"] != "pitcher"].copy()
    if "odds_type" in bat.columns:
        bat = bat.loc[bat["odds_type"] == "standard"].copy()

    pit = pp.loc[pp["player_type"] == "pitcher"].copy()
    if "odds_type" in pit.columns:
        pit = pit.loc[
            pit["odds_type"].isin(["standard", "demon", "goblin"])
        ].copy()
        pit["_pp_rk"] = pit["odds_type"].map(_pp_odds_rank)
        pit = pit.sort_values("_pp_rk").drop_duplicates(
            subset=["player_id", "player_type", "stat", "game_date_key"],
            keep="first",
        )
        pit = pit.drop(columns=["_pp_rk"], errors="ignore")
    else:
        pit = pit.drop_duplicates(
            subset=["player_id", "player_type", "stat", "game_date_key"],
            keep="first",
        )

    out = pd.concat(
        [
            bat[["player_id", "player_type", "stat", "line", "game_date_key"]],
            pit[["player_id", "player_type", "stat", "line", "game_date_key"]],
        ],
        ignore_index=True,
    )
    out["player_type"] = out["player_type"].astype(str).str.lower().str.strip()
    out = out.rename(columns={"line": "book_line"})
    out = out.drop_duplicates(
        subset=["player_id", "player_type", "stat", "game_date_key"],
        keep="first",
    )
    return out


def _load_pp_pitcher_name_team_fallback(pp: pd.DataFrame) -> pd.DataFrame:
    """One book line per (game_date_key, team, name_key, stat) for pitcher PP rows."""
    if pp.empty or "player_type" not in pp.columns:
        return pd.DataFrame(
            columns=["game_date_key", "team", "name_key", "stat", "book_line_fb"],
        )
    need = {"stat", "line", "game_date", "player_type", "player_name", "team"}
    if not need.issubset(set(pp.columns)):
        return pd.DataFrame(
            columns=["game_date_key", "team", "name_key", "stat", "book_line_fb"],
        )

    pit = pp.loc[pp["player_type"] == "pitcher"].copy()
    if pit.empty:
        return pd.DataFrame(
            columns=["game_date_key", "team", "name_key", "stat", "book_line_fb"],
        )
    if "odds_type" in pit.columns:
        pit = pit.loc[
            pit["odds_type"].isin(["standard", "demon", "goblin"])
        ].copy()
        pit["_pp_rk"] = pit["odds_type"].map(_pp_odds_rank)
        pit = pit.sort_values("_pp_rk")
    pit["game_date_key"] = pd.to_datetime(pit["game_date"]).dt.strftime("%Y-%m-%d")
    pit["name_key"] = pit["player_name"].map(_norm_player_name)
    pit["team"] = pit["team"].astype(str).str.upper().str.strip()
    pit = pit.drop_duplicates(
        subset=["game_date_key", "team", "name_key", "stat"],
        keep="first",
    )
    return pit[
        ["game_date_key", "team", "name_key", "stat", "line"]
    ].rename(columns={"line": "book_line_fb"})


def _metrics_block(
    sub: pd.DataFrame,
    min_n: int,
    line_col: str = "book_line",
) -> dict[str, float | int]:
    """Brier/ECE/temp on P(over) at book line."""
    sub = sub.dropna(subset=["p_over_market", "actual", line_col])
    n = len(sub)
    if n < min_n:
        return {}

    y_prob = np.clip(sub["p_over_market"].values.astype(float), 0.0, 1.0)
    y_true = (sub["actual"].values > sub[line_col].values).astype(float)

    if y_true.std() == 0:
        return {}

    return {
        "n": n,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(compute_ece(y_prob, y_true)),
        "temperature": float(compute_temperature(y_prob, y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest props at book lines (game_props + DK + PP merge)",
    )
    parser.add_argument(
        "--props-path",
        type=Path,
        default=None,
        help="Path to game_props.parquet (default: dashboard data dir)",
    )
    parser.add_argument(
        "--dk-path",
        type=Path,
        default=None,
        help="Path to dk_props_history.parquet or dk_props.parquet (default: auto)",
    )
    parser.add_argument(
        "--pp-path",
        type=Path,
        default=None,
        help="Path to pp_props_history.parquet or pp_props.parquet (default: auto)",
    )
    parser.add_argument(
        "--n-dates",
        type=int,
        default=3,
        help="When --dates is omitted, use this many most recent dates (default: 3)",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=None,
        help="Explicit game_date values (YYYY-MM-DD); overrides --n-dates",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=5,
        help="Minimum rows per (player_type, stat) to report metrics",
    )
    parser.add_argument(
        "--side",
        choices=["all", "batter", "pitcher"],
        default="all",
        help="Restrict rows to batter or pitcher props (default: all)",
    )
    parser.add_argument(
        "--min-n-pitcher",
        type=int,
        default=3,
        help="Min rows for pitcher (player_type, stat) groups (default: 3)",
    )
    parser.add_argument(
        "--model-lines",
        action="store_true",
        help="Fall back to model's own line_mid when no book line is matched. "
             "Enables pitcher prop evaluation even without historical DK/PP lines.",
    )
    args = parser.parse_args()

    props_path = args.props_path or (DASHBOARD_DIR / "game_props.parquet")
    if not props_path.exists():
        logger.error("game_props.parquet not found at %s", props_path)
        sys.exit(1)

    df = pd.read_parquet(props_path)
    logger.info("Loaded %d rows from %s", len(df), props_path)

    if "actual" not in df.columns:
        logger.error("No actual column — backfill results first")
        sys.exit(1)

    df = df[df["actual"].notna()].copy()
    if df.empty:
        logger.warning("No completed props (actual missing)")
        sys.exit(0)

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_date_key"] = df["game_date"].dt.strftime("%Y-%m-%d")
    day = df["game_date"].dt.normalize()

    if args.dates is not None:
        want = {pd.Timestamp(d).normalize() for d in args.dates}
        df = df[day.isin(want)].copy()
        logger.info("Filtered to explicit dates %s: %d rows", args.dates, len(df))
    else:
        unique_sorted = sorted(day.unique())
        if len(unique_sorted) > args.n_dates:
            picked = unique_sorted[-args.n_dates :]
        else:
            picked = unique_sorted
        df = df[day.isin(picked)].copy()
        logger.info(
            "Using %d date(s): %s (%d rows with actuals)",
            len(picked),
            [d.strftime("%Y-%m-%d") for d in picked],
            len(df),
        )

    if df.empty:
        logger.warning("No rows after date filter")
        sys.exit(0)

    df["player_type"] = df["player_type"].astype(str).str.lower().str.strip()

    dk_lines = _load_dk_lines(DASHBOARD_DIR, args.dk_path)
    pp_raw = _read_pp_raw(DASHBOARD_DIR, args.pp_path)
    pp_lines = _load_pp_lines(pp_raw)

    if dk_lines.empty and pp_lines.empty:
        logger.error(
            "No book lines: need dk_props_history/dk_props and/or "
            "pp_props_history/pp_props under dashboard data dir",
        )
        sys.exit(1)

    n_before = len(df)
    dk_rename = {"book_line": "book_line_dk"}
    if "book_implied" in dk_lines.columns:
        dk_rename["book_implied"] = "book_implied_dk"
    if "book_odds" in dk_lines.columns:
        dk_rename["book_odds"] = "book_odds_dk"
    df = df.merge(
        dk_lines.rename(columns=dk_rename),
        on=["player_id", "player_type", "stat", "game_date_key"],
        how="left",
    )
    df = df.merge(
        pp_lines.rename(columns={"book_line": "book_line_pp"}),
        on=["player_id", "player_type", "stat", "game_date_key"],
        how="left",
    )
    df["book_line"] = df["book_line_dk"].fillna(df["book_line_pp"])
    df["book_source"] = np.where(
        df["book_line_dk"].notna(),
        "dk",
        np.where(df["book_line_pp"].notna(), "pp", None),
    )
    # Carry DK implied odds through to edge analysis
    if "book_implied_dk" in df.columns:
        df["vegas_implied"] = df["book_implied_dk"]
    else:
        df["vegas_implied"] = np.nan
    if "book_odds_dk" in df.columns:
        df["vegas_odds"] = df["book_odds_dk"]

    # Pitchers: PP player_id often mismatches slate; fill from team + name + stat.
    fb_pp = _load_pp_pitcher_name_team_fallback(pp_raw)
    if not fb_pp.empty and "team" in df.columns and "player_name" in df.columns:
        pit_miss = (df["player_type"] == "pitcher") & df["book_line"].isna()
        n_miss = int(pit_miss.sum())
        if n_miss:
            sub = df.loc[pit_miss].copy()
            sub["_orig_idx"] = sub.index
            sub["name_key"] = sub["player_name"].map(_norm_player_name)
            sub["team_u"] = np.where(
                sub["team"].notna(),
                sub["team"].astype(str).str.upper().str.strip(),
                "",
            )
            m = sub.merge(
                fb_pp,
                left_on=["game_date_key", "team_u", "name_key", "stat"],
                right_on=["game_date_key", "team", "name_key", "stat"],
                how="left",
            )
            ok = m["book_line_fb"].notna()
            if ok.any():
                oix = m.loc[ok, "_orig_idx"].to_numpy()
                df.loc[oix, "book_line"] = m.loc[ok, "book_line_fb"].to_numpy()
                df.loc[oix, "book_source"] = "pp_nt"
                logger.info(
                    "Pitcher PP name+team fallback: filled %d / %d missing pitcher rows",
                    int(ok.sum()),
                    n_miss,
                )

    # --model-lines fallback: use the model's own line_mid for rows without a book line
    if args.model_lines and "line_mid" in df.columns:
        no_book = df["book_line"].isna()
        n_fill = int(no_book.sum())
        if n_fill:
            df.loc[no_book, "book_line"] = df.loc[no_book, "line_mid"]
            df.loc[no_book, "book_source"] = "model"
            logger.info(
                "Model-line fallback: filled %d rows (no DK/PP) with line_mid",
                n_fill,
            )

    n_book = df["book_line"].notna().sum()
    n_pit_bl = int(((df["player_type"] == "pitcher") & df["book_line"].notna()).sum())
    n_bat_bl = int(((df["player_type"] == "batter") & df["book_line"].notna()).sum())
    logger.info(
        "Book lines: %d / %d rows (DK=%d, PP-only=%d, model=%d); "
        "with line: %d pitcher / %d batter",
        n_book,
        n_before,
        int(df["book_line_dk"].notna().sum()),
        int((df["book_line_pp"].notna() & df["book_line_dk"].isna()).sum()),
        int((df["book_source"] == "model").sum()) if "book_source" in df.columns else 0,
        n_pit_bl,
        n_bat_bl,
    )

    df = df[df["book_line"].notna()].copy()
    if df.empty:
        logger.warning("No rows with a book or model line for selected dates")
        sys.exit(0)

    if args.side == "pitcher":
        df = df.loc[df["player_type"] == "pitcher"].copy()
    elif args.side == "batter":
        df = df.loc[df["player_type"] == "batter"].copy()
    if df.empty:
        logger.warning("No rows after --side %s filter", args.side)
        sys.exit(0)

    df = _attach_p_over_at_book(df, line_col="book_line")
    missing_p = df["p_over_market"].isna().sum()
    if missing_p:
        logger.warning(
            "%d rows have no P(over) at book_line (missing p_over_X.X / legacy match)",
            int(missing_p),
        )

    print("\n" + "=" * 72)
    print("MARKET-LINE BACKTEST (DK + PP book_line, P(over) at that line)")
    print("=" * 72)

    per_date = (
        df.assign(_d=df["game_date"].dt.strftime("%Y-%m-%d"))
        .groupby("_d")
        .size()
        .reset_index(name="n_props")
    )
    print("\nRows per date (actual + book line):")
    print(per_date.to_string(index=False))

    if "book_source" in df.columns:
        src = (
            df.assign(_d=df["game_date"].dt.strftime("%Y-%m-%d"))
            .groupby(["_d", "book_source"])
            .size()
            .unstack(fill_value=0)
        )
        if not src.empty:
            print("\nRows by date x book_source (dk = DK line used, pp = PP only):")
            print(src.to_string())

    summary_rows: list[dict] = []
    for (ptype, stat), group in df.groupby(["player_type", "stat"]):
        mn = (
            args.min_n_pitcher if str(ptype) == "pitcher" else args.min_n
        )
        m = _metrics_block(group, min_n=mn, line_col="book_line")
        if not m:
            continue
        summary_rows.append(
            {
                "side": ptype,
                "stat": stat,
                **{k: v for k, v in m.items()},
            },
        )

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        cols = [
            "side",
            "stat",
            "n",
            "brier",
            "ece",
            "temperature",
        ]
        print("\nBy player_type x stat:")
        float_cols = ["brier", "ece", "temperature"]
        formatted = summary[cols].copy()
        formatted["n"] = formatted["n"].astype(int)
        for c in float_cols:
            formatted[c] = formatted[c].map(lambda x: f"{float(x):.4f}")
        print(formatted.to_string(index=False))
    else:
        logger.warning(
            "No groups met min_n (batter %d / pitcher %d) with valid p_over at book_line",
            args.min_n,
            args.min_n_pitcher,
        )

    pool = df.dropna(subset=["p_over_market", "actual", "book_line"])
    if len(pool) >= args.min_n:
        y_prob = np.clip(pool["p_over_market"].values.astype(float), 0.0, 1.0)
        y_true = (pool["actual"].values > pool["book_line"].values).astype(float)
        if y_true.std() > 0:
            print("\nPooled (all included rows):")
            print(
                f"  n={len(pool)}  brier={brier_score_loss(y_true, y_prob):.4f}  "
                f"ece={compute_ece(y_prob, y_true):.4f}  "
                f"T={compute_temperature(y_prob, y_true):.3f}",
            )

    # ------------------------------------------------------------------
    # Pitcher prop detail section
    # ------------------------------------------------------------------
    pit = df[df["player_type"] == "pitcher"].copy()
    pit = pit.dropna(subset=["p_over_market", "actual", "book_line"])
    if len(pit) >= args.min_n_pitcher:
        _pitcher_detail(pit, min_n=args.min_n_pitcher)

    # ------------------------------------------------------------------
    # Edge analysis: when to recommend a pick
    # ------------------------------------------------------------------
    _edge_analysis(df, min_n=args.min_n)


# ======================================================================
# Edge analysis: when to recommend a pick
# ======================================================================

def _compute_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Add edge columns: model_p_over (at book line), book_implied, edge.

    Works from the p_over grid or the pre-computed model_p_over/model_edge
    columns if they exist.
    """
    out = df.copy()

    # If model_edge already in the data (from confident_picks), use it
    if "model_edge" in out.columns and "model_p_over" in out.columns:
        # Fill any remaining gaps from grid
        missing = out["model_p_over"].isna() & out["book_line"].notna()
        for idx in out.index[missing]:
            bl = float(out.at[idx, "book_line"])
            col = _p_over_column(bl)
            if col in out.columns and pd.notna(out.at[idx, col]):
                out.at[idx, "model_p_over"] = float(out.at[idx, col])
    else:
        # Compute from grid
        out["model_p_over"] = np.nan
        for idx in out.index[out["book_line"].notna()]:
            bl = float(out.at[idx, "book_line"])
            col = _p_over_column(bl)
            if col in out.columns and pd.notna(out.at[idx, col]):
                out.at[idx, "model_p_over"] = float(out.at[idx, col])

    # Book implied probability (from DK odds or use 0.5 for model-lines)
    if "vegas_implied" not in out.columns:
        out["vegas_implied"] = np.nan
    if "book_source" in out.columns:
        # For model-line fallback rows, no book implied — use 0.5
        model_rows = out["book_source"] == "model"
        out.loc[model_rows & out["vegas_implied"].isna(), "vegas_implied"] = 0.5

    if "model_edge" not in out.columns:
        out["model_edge"] = np.nan
    has_both = out["model_p_over"].notna() & out["vegas_implied"].notna()
    out.loc[has_both, "model_edge"] = (
        out.loc[has_both, "model_p_over"].astype(float)
        - out.loc[has_both, "vegas_implied"].astype(float)
    )

    # Pick direction: does model lean over or under?
    out["pick_side"] = np.where(out["model_p_over"] > 0.5, "over", "under")
    # Confidence: distance from 0.5 (regardless of direction)
    out["confidence"] = (out["model_p_over"] - 0.5).abs()
    # Did the pick hit?
    out["pick_hit"] = np.where(
        out["pick_side"] == "over",
        out["actual"] > out["book_line"],
        out["actual"] < out["book_line"],
    )
    # Handle pushes (actual == book_line) as losses
    out.loc[out["actual"] == out["book_line"], "pick_hit"] = False

    return out


def _edge_analysis(df: pd.DataFrame, min_n: int = 5) -> None:
    """Comprehensive edge analysis: at what confidence/edge should we pick?"""

    edf = _compute_edge(df)
    edf = edf.dropna(subset=["model_p_over", "actual", "book_line"])

    if len(edf) < min_n:
        return

    print("\n" + "=" * 72)
    print("EDGE ANALYSIS: WHEN TO RECOMMEND A PICK")
    print("=" * 72)

    # --- 1. Overall pick accuracy by confidence tier ---
    print("\n1. PICK ACCURACY BY MODEL CONFIDENCE")
    print("   (confidence = |P(over) - 0.5|, pick = side model leans)")

    conf_bins = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 1.0]
    conf_labels = ["0-3%", "3-6%", "6-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    edf["conf_bin"] = pd.cut(edf["confidence"], bins=conf_bins, labels=conf_labels, right=False)

    conf_rows: list[dict] = []
    for label in conf_labels:
        grp = edf[edf["conf_bin"] == label]
        n = len(grp)
        if n < 3:
            continue
        hits = grp["pick_hit"].sum()
        win_rate = hits / n if n > 0 else 0
        avg_conf = grp["confidence"].mean()
        avg_edge = grp["model_edge"].mean() if grp["model_edge"].notna().any() else np.nan
        # Expected value at -110 juice: win_rate * 0.909 - (1-win_rate) * 1.0
        ev_110 = win_rate * (100 / 110) - (1 - win_rate) * 1.0
        conf_rows.append({
            "confidence": label,
            "n": n,
            "wins": int(hits),
            "win%": f"{win_rate:.1%}",
            "avg_conf": f"{avg_conf:.3f}",
            "avg_edge": f"{avg_edge:.3f}" if not np.isnan(avg_edge) else "---",
            "EV@-110": f"{ev_110:+.3f}",
            "verdict": "PICK" if ev_110 > 0 else "pass",
        })
    if conf_rows:
        print(pd.DataFrame(conf_rows).to_string(index=False))
    else:
        print("  Insufficient data for confidence analysis")

    # --- 2. Cumulative: "picks with confidence >= X" ---
    print("\n2. CUMULATIVE WIN RATE (all picks at or above threshold)")
    cum_rows: list[dict] = []
    thresholds = [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    for t in thresholds:
        grp = edf[edf["confidence"] >= t]
        n = len(grp)
        if n < 3:
            continue
        hits = grp["pick_hit"].sum()
        wr = hits / n
        ev_110 = wr * (100 / 110) - (1 - wr)
        cum_rows.append({
            "min_conf": f"{t:.0%}",
            "n": n,
            "win%": f"{wr:.1%}",
            "EV@-110": f"{ev_110:+.3f}",
            "keep": "YES" if ev_110 > 0 else "no",
        })
    if cum_rows:
        print(pd.DataFrame(cum_rows).to_string(index=False))

    # --- 3. Edge analysis (model vs book, requires DK/PP odds) ---
    has_edge = edf["model_edge"].notna()
    n_edge = has_edge.sum()
    if n_edge >= min_n:
        print(f"\n3. EDGE ANALYSIS (model P(over) - book implied, n={n_edge})")

        edge_bins = [-1.0, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 1.0]
        edge_labels = ["<-10%", "-10 to -5%", "-5 to 0%", "0 to 5%",
                        "5 to 10%", "10 to 15%", "15 to 20%", "20%+"]
        edf_e = edf[has_edge].copy()
        edf_e["edge_bin"] = pd.cut(edf_e["model_edge"], bins=edge_bins, labels=edge_labels, right=False)

        # For edge picks: pick over when edge>0, under when edge<0
        edf_e["edge_pick_hit"] = np.where(
            edf_e["model_edge"] > 0,
            edf_e["actual"] > edf_e["book_line"],
            edf_e["actual"] < edf_e["book_line"],
        )
        edf_e.loc[edf_e["actual"] == edf_e["book_line"], "edge_pick_hit"] = False

        edge_rows: list[dict] = []
        for label in edge_labels:
            grp = edf_e[edf_e["edge_bin"] == label]
            n = len(grp)
            if n < 3:
                continue
            hits = grp["edge_pick_hit"].sum()
            wr = hits / n
            avg_e = grp["model_edge"].mean()
            ev_110 = wr * (100 / 110) - (1 - wr)
            edge_rows.append({
                "edge_bin": label,
                "n": n,
                "wins": int(hits),
                "win%": f"{wr:.1%}",
                "avg_edge": f"{avg_e:+.3f}",
                "EV@-110": f"{ev_110:+.3f}",
                "verdict": "BET" if ev_110 > 0 else "pass",
            })
        if edge_rows:
            print(pd.DataFrame(edge_rows).to_string(index=False))

        # Cumulative edge
        print("\n   Cumulative (all picks with |edge| >= threshold):")
        abs_edge = edf_e["model_edge"].abs()
        for t in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            grp = edf_e[abs_edge >= t]
            n = len(grp)
            if n < 3:
                continue
            hits = grp["edge_pick_hit"].sum()
            wr = hits / n
            ev = wr * (100 / 110) - (1 - wr)
            print(f"   |edge| >= {t:5.0%}: n={n:4d}  win%={wr:.1%}  EV@-110={ev:+.3f}  {'BET' if ev > 0 else 'pass'}")
    else:
        print(f"\n3. EDGE ANALYSIS: skipped ({n_edge} rows with book implied — need DK/PP history)")

    # --- 4. By stat type ---
    print("\n4. PICK ACCURACY BY STAT TYPE (confidence >= 5%)")
    strong = edf[edf["confidence"] >= 0.05].copy()
    if len(strong) >= min_n:
        stat_rows: list[dict] = []
        for (ptype, stat), grp in strong.groupby(["player_type", "stat"]):
            n = len(grp)
            if n < 3:
                continue
            hits = grp["pick_hit"].sum()
            wr = hits / n
            ev = wr * (100 / 110) - (1 - wr)
            avg_c = grp["confidence"].mean()
            stat_rows.append({
                "side": ptype,
                "stat": stat,
                "n": n,
                "win%": f"{wr:.1%}",
                "avg_conf": f"{avg_c:.3f}",
                "EV@-110": f"{ev:+.3f}",
                "verdict": "PICK" if ev > 0 else "pass",
            })
        if stat_rows:
            sdf = pd.DataFrame(stat_rows).sort_values("EV@-110", ascending=False)
            print(sdf.to_string(index=False))

    # --- 5. Kelly sizing preview ---
    print("\n5. KELLY SIZING PREVIEW (assumes -110 juice, fractional Kelly = 0.25)")
    kelly_strong = edf[edf["confidence"] >= 0.10].copy()
    if len(kelly_strong) >= min_n:
        wr = kelly_strong["pick_hit"].mean()
        # At -110: b = 100/110 = 0.909
        b = 100.0 / 110.0
        # Kelly fraction = (b*p - q) / b where p=win_rate, q=1-p
        p = wr
        q = 1 - p
        full_kelly = (b * p - q) / b if b * p > q else 0.0
        frac_kelly = full_kelly * 0.25
        print(f"  Picks with confidence >= 10%: n={len(kelly_strong)}")
        print(f"  Win rate: {wr:.1%}")
        print(f"  Full Kelly: {full_kelly:.1%} of bankroll per bet")
        print(f"  Quarter Kelly: {frac_kelly:.1%} of bankroll per bet")
        if full_kelly <= 0:
            print("  (Negative Kelly = no edge at this threshold)")
    else:
        print("  Not enough data yet for Kelly sizing")


# ======================================================================
# Pitcher-specific diagnostics
# ======================================================================

def _pitcher_detail(pit: pd.DataFrame, min_n: int = 3) -> None:
    """Detailed pitcher prop diagnostics: per-line, per-pitcher, context."""

    print("\n" + "=" * 72)
    print("PITCHER PROP DETAIL")
    print("=" * 72)

    pit_y_prob = np.clip(pit["p_over_market"].values.astype(float), 0.0, 1.0)
    pit_y_true = (pit["actual"].values > pit["book_line"].values).astype(float)
    if pit_y_true.std() > 0:
        print(
            f"\nPitcher pooled: n={len(pit)}  "
            f"brier={brier_score_loss(pit_y_true, pit_y_prob):.4f}  "
            f"ece={compute_ece(pit_y_prob, pit_y_true):.4f}  "
            f"T={compute_temperature(pit_y_prob, pit_y_true):.3f}"
        )

    # --- Per-line accuracy for each stat ---
    print("\nPer-line accuracy (pitcher):")
    line_rows: list[dict] = []
    for stat, grp in pit.groupby("stat"):
        for bl, bl_grp in grp.groupby("book_line"):
            n = len(bl_grp)
            if n < min_n:
                continue
            yp = np.clip(bl_grp["p_over_market"].values.astype(float), 0.0, 1.0)
            yt = (bl_grp["actual"].values > bl).astype(float)
            if yt.std() == 0:
                continue
            over_rate = float(yt.mean())
            avg_p = float(yp.mean())
            line_rows.append({
                "stat": stat,
                "line": bl,
                "n": n,
                "over%": f"{over_rate:.0%}",
                "avg_P(o)": f"{avg_p:.3f}",
                "brier": f"{brier_score_loss(yt, yp):.4f}",
            })
    if line_rows:
        line_df = pd.DataFrame(line_rows).sort_values(["stat", "line"])
        print(line_df.to_string(index=False))

    # --- Per-pitcher performance ---
    if "player_name" in pit.columns:
        print("\nPer-pitcher (min", min_n, "props):")
        pitcher_rows: list[dict] = []
        for (pid, pname), grp in pit.groupby(["player_id", "player_name"]):
            n = len(grp)
            if n < min_n:
                continue
            yp = np.clip(grp["p_over_market"].values.astype(float), 0.0, 1.0)
            yt = (grp["actual"].values > grp["book_line"].values).astype(float)
            if yt.std() == 0:
                continue
            pitcher_rows.append({
                "pitcher": str(pname)[:20],
                "n": n,
                "stats": "+".join(sorted(grp["stat"].unique())),
                "brier": float(brier_score_loss(yt, yp)),
                "over%": float(yt.mean()),
                "avg_P(o)": float(yp.mean()),
            })
        if pitcher_rows:
            pf = pd.DataFrame(pitcher_rows).sort_values("brier")
            fmt = pf.copy()
            fmt["brier"] = fmt["brier"].map(lambda x: f"{x:.4f}")
            fmt["over%"] = fmt["over%"].map(lambda x: f"{x:.0%}")
            fmt["avg_P(o)"] = fmt["avg_P(o)"].map(lambda x: f"{x:.3f}")
            print(fmt.to_string(index=False))

    # --- Context: umpire lift impact ---
    if "umpire_k_lift" in pit.columns:
        _context_split(pit, "umpire_k_lift", "Umpire K lift", min_n)

    if "umpire_bb_lift" in pit.columns:
        _context_split(pit, "umpire_bb_lift", "Umpire BB lift", min_n)


def _context_split(
    pit: pd.DataFrame,
    col: str,
    label: str,
    min_n: int,
) -> None:
    """Split pitcher props by a context column into terciles and report."""
    vals = pit[col].dropna()
    if len(vals) < min_n * 3:
        return

    has_signal = vals.abs() > 1e-6
    n_signal = int(has_signal.sum())
    n_zero = int((~has_signal).sum())

    if n_signal < min_n:
        print(f"\n{label}: {n_zero} rows with no signal, {n_signal} with signal (too few)")
        return

    # Split into zero-lift vs nonzero, then split nonzero into high/low
    rows: list[dict] = []
    if n_zero >= min_n:
        sub = pit[pit[col].abs() <= 1e-6]
        m = _metrics_block(sub, min_n=min_n, line_col="book_line")
        if m:
            rows.append({"group": "no_signal", **m})

    nz = pit[pit[col].abs() > 1e-6].copy()
    if len(nz) >= min_n * 2:
        median_val = nz[col].median()
        for label_g, mask in [
            ("below_med", nz[col] <= median_val),
            ("above_med", nz[col] > median_val),
        ]:
            sub = nz[mask]
            if len(sub) < min_n:
                continue
            m = _metrics_block(sub, min_n=min_n, line_col="book_line")
            if m:
                rows.append({"group": label_g, **m})
    elif len(nz) >= min_n:
        m = _metrics_block(nz, min_n=min_n, line_col="book_line")
        if m:
            rows.append({"group": "has_signal", **m})

    if rows:
        print(f"\n{label} stratification:")
        cdf = pd.DataFrame(rows)
        for c in ["brier", "ece", "temperature"]:
            if c in cdf.columns:
                cdf[c] = cdf[c].map(lambda x: f"{float(x):.4f}")
        print(cdf.to_string(index=False))


if __name__ == "__main__":
    main()
