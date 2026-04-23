"""Evaluate game-level predictions vs actual scores.

Reads the sim_predictions_archive and game_odds_history from the
dashboard repo, joins with actual game scores from the database,
and reports accuracy metrics across multiple dimensions:

  - Total runs: correlation, bias, MAE
  - Win probability: accuracy, Brier score
  - Margin: correlation
  - O/U calibration per line
  - Vegas blend weight sweep (when odds available)
  - Per-date bias timeline

Usage
-----
    python scripts/eval_game_preds.py                # full report
    python scripts/eval_game_preds.py --days 7       # last 7 days only
    python scripts/eval_game_preds.py --no-vegas     # skip Vegas blend analysis
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.db import read_sql
from src.data.paths import dashboard_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = dashboard_dir()

# Team name mapping: odds descriptions -> standard abbreviations
TEAM_MAP = {
    "ARI Diamondbacks": "AZ", "Arizona Diamondbacks": "AZ",
    "ATL Braves": "ATL", "Atlanta Braves": "ATL",
    "BAL Orioles": "BAL", "Baltimore Orioles": "BAL",
    "BOS Red Sox": "BOS", "Boston Red Sox": "BOS",
    "CHI Cubs": "CHC", "Chicago Cubs": "CHC",
    "CHI White Sox": "CWS", "Chicago White Sox": "CWS",
    "CIN Reds": "CIN", "Cincinnati Reds": "CIN",
    "CLE Guardians": "CLE", "Cleveland Guardians": "CLE",
    "COL Rockies": "COL", "Colorado Rockies": "COL",
    "DET Tigers": "DET", "Detroit Tigers": "DET",
    "HOU Astros": "HOU", "Houston Astros": "HOU",
    "KC Royals": "KC", "Kansas City Royals": "KC",
    "LA Angels": "LAA", "Los Angeles Angels": "LAA",
    "LA Dodgers": "LAD", "Los Angeles Dodgers": "LAD",
    "MIA Marlins": "MIA", "Miami Marlins": "MIA",
    "MIL Brewers": "MIL", "Milwaukee Brewers": "MIL",
    "MIN Twins": "MIN", "Minnesota Twins": "MIN",
    "NY Mets": "NYM", "New York Mets": "NYM",
    "NY Yankees": "NYY", "New York Yankees": "NYY",
    "Athletics": "OAK", "Oakland Athletics": "OAK",
    "PHI Phillies": "PHI", "Philadelphia Phillies": "PHI",
    "PIT Pirates": "PIT", "Pittsburgh Pirates": "PIT",
    "SD Padres": "SD", "San Diego Padres": "SD",
    "SF Giants": "SF", "San Francisco Giants": "SF",
    "SEA Mariners": "SEA", "Seattle Mariners": "SEA",
    "STL Cardinals": "STL", "St. Louis Cardinals": "STL",
    "TB Rays": "TB", "Tampa Bay Rays": "TB",
    "TEX Rangers": "TEX", "Texas Rangers": "TEX",
    "TOR Blue Jays": "TOR", "Toronto Blue Jays": "TOR",
    "WSH Nationals": "WSH", "Washington Nationals": "WSH",
    "WAS Nationals": "WSH",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_actual_scores(season: int = 2026) -> pd.DataFrame:
    """Load actual game scores from the database."""
    return read_sql(f"""
        WITH team_scores AS (
            SELECT fpg.game_pk, fpg.team_id,
                   SUM(COALESCE(fpg.bat_r, 0)) AS team_runs
            FROM production.fact_player_game_mlb fpg
            JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
            WHERE fpg.season = {int(season)} AND dg.game_type = 'R'
            GROUP BY fpg.game_pk, fpg.team_id
        )
        SELECT dg.game_pk, dg.game_date,
               COALESCE(h.team_runs, 0) AS home_runs,
               COALESCE(a.team_runs, 0) AS away_runs,
               COALESCE(h.team_runs, 0) + COALESCE(a.team_runs, 0) AS total_runs
        FROM production.dim_game dg
        LEFT JOIN team_scores h
            ON dg.game_pk = h.game_pk AND dg.home_team_id = h.team_id
        LEFT JOIN team_scores a
            ON dg.game_pk = a.game_pk AND dg.away_team_id = a.team_id
        WHERE dg.season = {int(season)} AND dg.game_type = 'R'
          AND dg.game_date < CURRENT_DATE
    """)


def _parse_teams(desc: str) -> tuple[str | None, str | None]:
    sep = " @ " if " @ " in desc else " vs " if " vs " in desc else None
    if sep is None:
        return None, None
    parts = desc.split(sep)
    return TEAM_MAP.get(parts[0].strip()), TEAM_MAP.get(parts[1].strip())


def load_vegas_odds() -> pd.DataFrame:
    """Load and pivot historical Vegas odds into one row per game."""
    odds_path = DASHBOARD_DIR / "game_odds_history.parquet"
    if not odds_path.exists():
        logger.warning("No game_odds_history.parquet found at %s", odds_path)
        return pd.DataFrame()

    odds = pd.read_parquet(odds_path)
    odds["snapshot_ts"] = pd.to_datetime(odds["snapshot_ts"])

    # Total lines (latest snapshot per game per source, then average)
    totals = odds[
        (odds["market_type"] == "total") & (odds["outcome_type"] == "Over")
    ].copy()
    totals = totals.sort_values("snapshot_ts").drop_duplicates(
        ["game_date", "game_description", "source"], keep="last",
    )
    total_agg = (
        totals.groupby(["game_date", "game_description"])
        .agg(total_line=("line", "mean"))
        .reset_index()
    )

    # Moneyline: home + away for de-vigging
    ml_home = odds[
        (odds["market_type"] == "moneyline") & (odds["outcome_type"] == "Home")
    ].copy()
    ml_home = ml_home.sort_values("snapshot_ts").drop_duplicates(
        ["game_date", "game_description", "source"], keep="last",
    )
    ml_away = odds[
        (odds["market_type"] == "moneyline") & (odds["outcome_type"] == "Away")
    ].copy()
    ml_away = ml_away.sort_values("snapshot_ts").drop_duplicates(
        ["game_date", "game_description", "source"], keep="last",
    )
    h_agg = (
        ml_home.groupby(["game_date", "game_description"])
        .agg(home_raw=("implied_prob", "mean"))
        .reset_index()
    )
    a_agg = (
        ml_away.groupby(["game_date", "game_description"])
        .agg(away_raw=("implied_prob", "mean"))
        .reset_index()
    )
    ml_agg = h_agg.merge(a_agg, on=["game_date", "game_description"], how="inner")
    ml_agg["vig"] = ml_agg["home_raw"] + ml_agg["away_raw"]
    ml_agg["home_ml_fair"] = ml_agg["home_raw"] / ml_agg["vig"]

    vegas = total_agg.merge(
        ml_agg[["game_date", "game_description", "home_ml_fair"]],
        on=["game_date", "game_description"],
        how="outer",
    )
    teams = vegas["game_description"].apply(
        lambda x: pd.Series(_parse_teams(x))
    )
    vegas["away_abbr_v"] = teams[0]
    vegas["home_abbr_v"] = teams[1]

    unmatched = vegas["away_abbr_v"].isna() | vegas["home_abbr_v"].isna()
    if unmatched.any():
        for d in vegas.loc[unmatched, "game_description"].unique()[:5]:
            logger.warning("Unmatched odds game description: %s", d)
    vegas = vegas.dropna(subset=["away_abbr_v", "home_abbr_v"])
    return vegas


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_total_row(
    label: str, pred: pd.Series, actual: pd.Series,
) -> None:
    n = len(pred)
    r = np.corrcoef(pred, actual)[0, 1] if n >= 5 else float("nan")
    bias = float((pred - actual).mean())
    mae = float(np.abs(pred - actual).mean())
    print(
        f"{label:<25} {n:>4} {r:>7.3f} {bias:>+7.2f} {mae:>7.2f} "
        f"{pred.mean():>7.2f} {actual.mean():>7.2f}"
    )


def _print_wp_row(
    label: str, pred: pd.Series, actual: pd.Series,
) -> None:
    n = len(pred)
    acc = float(((pred > 0.5).astype(float) == actual).mean())
    br = brier_score_loss(actual, pred)
    print(f"{label:<25} {n:>4} {acc:>7.3f} {br:>7.4f}")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_core_metrics(merged: pd.DataFrame) -> None:
    """Total runs, win prob, and margin -- full archive."""
    pred_total = merged["total_mean"].fillna(merged["total_runs_mean"])
    pred_hwp = merged["p_home_win"].fillna(merged["home_win_prob"])
    actual_total = merged["total_runs"]
    actual_margin = merged["home_runs"] - merged["away_runs"]
    actual_hwp = (merged["home_runs"] > merged["away_runs"]).astype(float)
    no_tie = merged["home_runs"] != merged["away_runs"]

    date_min = merged["game_date"].min()
    date_max = merged["game_date"].max()
    if hasattr(date_min, "date"):
        date_min, date_max = date_min.date(), date_max.date()

    print()
    print("=" * 70)
    print(f"GAME-LEVEL PREDICTIONS ({len(merged)} games, {date_min} to {date_max})")
    print("=" * 70)

    mask_t = pred_total.notna() & actual_total.notna()
    if mask_t.sum() >= 5:
        _print_total_row("Total runs", pred_total[mask_t], actual_total[mask_t])

    mask_w = pred_hwp.notna() & no_tie
    if mask_w.sum() >= 10:
        _print_wp_row("Win prob", pred_hwp[mask_w], actual_hwp[mask_w])

    pred_margin = merged["margin_mean"].fillna(merged["home_margin_mean"])
    mask_m = pred_margin.notna()
    if mask_m.sum() >= 10:
        r_m = np.corrcoef(pred_margin[mask_m], actual_margin[mask_m])[0, 1]
        print(f"{'Margin':<25} {mask_m.sum():>4} {r_m:>7.3f}")


def report_pre_post_split(merged: pd.DataFrame) -> None:
    """Compare pre-Phase1+2 vs post-Phase1+2 predictions."""
    has_new = merged["sim_total_raw"].notna()
    old = merged[~has_new]
    new = merged[has_new]

    for label, df in [("PRE-CHANGES", old), ("POST-CHANGES", new)]:
        if len(df) < 5:
            continue
        print()
        print("=" * 70)
        print(f"{label} ({len(df)} games)")
        print("=" * 70)

        pt = df["total_mean"].fillna(df["total_runs_mean"])
        at = df["total_runs"]
        m = pt.notna() & at.notna()
        if m.sum() >= 5:
            _print_total_row("Total runs", pt[m], at[m])

        pw = df["p_home_win"].fillna(df["home_win_prob"])
        nt = df["home_runs"] != df["away_runs"]
        mw = pw.notna() & nt
        if mw.sum() >= 5:
            aw = (df.loc[mw, "home_runs"] > df.loc[mw, "away_runs"]).astype(float)
            _print_wp_row("Win prob", pw[mw], aw)


def report_ou_calibration(merged: pd.DataFrame) -> None:
    """Over/under Brier scores per line."""
    actual_total = merged["total_runs"]

    print()
    print("=" * 70)
    print("O/U CALIBRATION")
    print("=" * 70)
    for line in [5.5, 6.5, 7.5, 8.5, 9.5, 10.5]:
        col_new = f"p_over_{line}"
        col_old = f"p_over_{str(line).replace('.', '_')}"
        p = merged.get(col_new, pd.Series(dtype=float))
        if col_old in merged.columns:
            p = p.fillna(merged[col_old])
        mask = p.notna() & actual_total.notna()
        if mask.sum() < 10:
            continue
        actual_over = (actual_total[mask] > line).astype(float)
        pred_rate = float(p[mask].mean())
        actual_rate = float(actual_over.mean())
        br = brier_score_loss(actual_over, p[mask])
        print(
            f"  O/U {line:4.1f}: n={mask.sum():3d}  "
            f"pred={pred_rate:.3f}  actual={actual_rate:.3f}  Brier={br:.4f}"
        )


def report_vegas_blend(merged: pd.DataFrame) -> None:
    """Sweep blend weights: sim vs Vegas."""
    sim_total = merged["total_mean"].fillna(merged["total_runs_mean"])
    sim_hwp = merged["p_home_win"].fillna(merged["home_win_prob"])
    actual_total = merged["total_runs"]
    actual_hwp = (merged["home_runs"] > merged["away_runs"]).astype(float)
    no_tie = merged["home_runs"] != merged["away_runs"]
    has_vegas = merged["total_line_v"].notna()

    v = has_vegas & sim_total.notna()
    if v.sum() < 10:
        logger.info("Not enough games with Vegas odds for blend analysis (%d)", v.sum())
        return

    vl = merged.loc[v, "total_line_v"]

    # --- Total runs ---
    print()
    print("=" * 80)
    print(f"TOTAL RUNS: SIM vs VEGAS vs BLENDS ({v.sum()} games with odds)")
    print("=" * 80)
    header = (
        f"{'Method':<25} {'n':>4} {'r':>7} {'bias':>7} "
        f"{'MAE':>7} {'pred':>7} {'actual':>7}"
    )
    print(header)
    print("-" * 80)
    _print_total_row("Sim", sim_total[v], actual_total[v])
    _print_total_row("Vegas line", vl, actual_total[v])
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        blended = w * sim_total[v] + (1 - w) * vl
        label = f"Blend w={w:.1f}"
        if w == 0.3:
            label += " *"
        _print_total_row(label, blended, actual_total[v])

    # --- Win probability ---
    vw = has_vegas & sim_hwp.notna() & no_tie & merged["home_ml_fair"].notna()
    if vw.sum() >= 10:
        vml = merged.loc[vw, "home_ml_fair"]

        print()
        print("=" * 80)
        print(f"WIN PROBABILITY: SIM vs VEGAS vs BLENDS ({vw.sum()} games)")
        print("=" * 80)
        print(f"{'Method':<25} {'n':>4} {'acc':>7} {'Brier':>7}")
        print("-" * 80)
        _print_wp_row("Sim", sim_hwp[vw], actual_hwp[vw])
        _print_wp_row("Vegas ML (de-vigged)", vml, actual_hwp[vw])
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
            blended = w * sim_hwp[vw] + (1 - w) * vml
            label = f"Blend w={w:.1f}"
            if w == 0.3:
                label += " *"
            _print_wp_row(label, blended, actual_hwp[vw])

    # --- O/U Brier by line ---
    print()
    print("=" * 80)
    print("O/U BRIER BY LINE (blend sweep)")
    print("=" * 80)
    r_nb = 3.6
    for line in [7.5, 8.5, 9.5]:
        act_over = (actual_total[v] > line).astype(float)
        print(f"\n  Line {line} (actual over rate = {act_over.mean():.3f}):")
        for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
            blended_t = w * sim_total[v] + (1 - w) * vl
            p_over = 1.0 - nbinom.cdf(
                int(line), 2 * r_nb, 2 * r_nb / (2 * r_nb + blended_t),
            )
            br = brier_score_loss(act_over, p_over)
            label = f"w={w:.1f}"
            if w == 0.3:
                label += " *"
            print(
                f"    {label:<12} Brier={br:.4f}  pred_rate={p_over.mean():.3f}"
            )


def report_per_date(merged: pd.DataFrame) -> None:
    """Per-date bias timeline."""
    print()
    print("=" * 70)
    print("PER-DATE SUMMARY")
    print("=" * 70)
    for dt, grp in merged.groupby("game_date"):
        st = grp["total_mean"].fillna(grp["total_runs_mean"])
        at = grp["total_runs"]
        vt = grp.get("total_line_v")
        m_s = st.notna() & at.notna()
        if m_s.sum() < 3:
            continue
        sim_bias = float((st[m_s] - at[m_s]).mean())
        dt_str = dt.date() if hasattr(dt, "date") else dt

        m_v = vt.notna() & st.notna() if vt is not None else pd.Series(False, index=grp.index)
        if m_v.sum() >= 3:
            vegas_bias = float((vt[m_v] - at[m_v]).mean())
            blend = 0.3 * st[m_v] + 0.7 * vt[m_v]
            blend_bias = float((blend - at[m_v]).mean())
            print(
                f"  {dt_str}  n={m_s.sum():2d}  "
                f"sim={sim_bias:+5.1f}  vegas={vegas_bias:+5.1f}  "
                f"blend={blend_bias:+5.1f}  actual={at[m_s].mean():.1f}"
            )
        else:
            fmt = "NEW" if grp.get("sim_total_raw") is not None and grp["sim_total_raw"].notna().any() else "old"
            print(
                f"  {dt_str}  n={m_s.sum():2d}  "
                f"sim={sim_bias:+5.1f}  (no vegas)  "
                f"actual={at[m_s].mean():.1f}  [{fmt}]"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Game prediction evaluation")
    parser.add_argument(
        "--days", type=int, default=None,
        help="Only include last N days of completed games",
    )
    parser.add_argument(
        "--no-vegas", action="store_true",
        help="Skip Vegas blend analysis",
    )
    parser.add_argument(
        "--season", type=int, default=2026,
        help="Season to evaluate (default: 2026)",
    )
    args = parser.parse_args()

    # Load data
    archive_path = DASHBOARD_DIR / "sim_predictions_archive.parquet"
    if not archive_path.exists():
        logger.error("No sim_predictions_archive.parquet at %s", archive_path)
        sys.exit(1)

    spa = pd.read_parquet(archive_path)
    spa["game_date"] = pd.to_datetime(spa["game_date"])
    logger.info("Loaded %d archived predictions", len(spa))

    actuals = load_actual_scores(args.season)
    logger.info("Loaded %d actual game scores (R/G = %.2f)", len(actuals), actuals["total_runs"].mean())

    # Merge predictions with actuals
    merged = spa.merge(
        actuals[["game_pk", "home_runs", "away_runs", "total_runs"]],
        on="game_pk", how="inner",
    )
    logger.info("Matched %d games with predictions + actuals", len(merged))

    if merged.empty:
        logger.warning("No games to evaluate")
        sys.exit(0)

    # Date filter
    if args.days is not None:
        merged["game_date"] = pd.to_datetime(merged["game_date"])
        cutoff = merged["game_date"].max() - pd.Timedelta(days=args.days)
        merged = merged[merged["game_date"] >= cutoff]
        logger.info("Filtered to last %d days: %d games", args.days, len(merged))

    # Merge with Vegas odds (for blend analysis)
    if not args.no_vegas:
        vegas = load_vegas_odds()
        if not vegas.empty:
            # Normalize date types for join
            vegas["game_date"] = pd.to_datetime(vegas["game_date"])
            merged["game_date"] = pd.to_datetime(merged["game_date"])
            merged = merged.merge(
                vegas[["game_date", "away_abbr_v", "home_abbr_v",
                        "total_line", "home_ml_fair"]],
                left_on=["game_date", "away_abbr", "home_abbr"],
                right_on=["game_date", "away_abbr_v", "home_abbr_v"],
                how="left",
                suffixes=("", "_v"),
            )
            # Rename if no suffix conflict
            if "total_line_v" not in merged.columns and "total_line" in merged.columns:
                merged.rename(columns={"total_line": "total_line_v"}, inplace=True)
            n_vegas = merged["total_line_v"].notna().sum() if "total_line_v" in merged.columns else 0
            logger.info("Matched %d games with Vegas odds", n_vegas)

    # Reports
    report_core_metrics(merged)
    report_pre_post_split(merged)
    report_ou_calibration(merged)

    if not args.no_vegas and "total_line_v" in merged.columns:
        report_vegas_blend(merged)

    report_per_date(merged)

    # Baselines
    print()
    print("--- Baselines ---")
    print("  Coin flip Brier: 0.2500")
    print("  Home-field-only accuracy: ~0.540")
    print("  Vegas total r (literature): ~0.15-0.25")
    print()


if __name__ == "__main__":
    main()
