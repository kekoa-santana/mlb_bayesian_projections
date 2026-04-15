"""
Prospective market comparison for the game simulator.

Evaluates the game simulator's predictions against Bovada/DraftKings lines
across three surfaces — moneyline, run line (spread), and total. Designed
to run daily on today's predictions and also to back-fill across an archive
of past predictions once we accumulate history.

Historical 2025 odds are not available (collection started in 2026), so the
walk-forward backtest cannot use these functions. This module is the
forward-looking evaluator that builds up a real market-vs-model track
record over the course of the season.

Workflow
--------
1. Daily pipeline appends ``todays_game_predictions.parquet`` to a growing
   ``sim_predictions_archive.parquet`` keyed on ``(game_date, game_pk)``.
2. After games complete, ``fact_game_totals`` provides the ground truth
   (home/away runs scored) for back-filling hit/miss flags.
3. ``scripts/run_market_comparison.py`` joins archive predictions to the
   closing Bovada snapshot for each game, computes edge + pick, and
   aggregates hit rate / ROI by edge bucket and surface.

The bias correction step (``compute_bias_correction``) is optional: when
enough completed games are in the archive, we learn a per-surface bias
and subtract it from predictions before computing edges. This prevents
bias from showing up as spurious edge signal.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team name → abbreviation (Bovada game_description parsing)
# ---------------------------------------------------------------------------

_TEAM_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _parse_game_description(desc: str) -> tuple[str | None, str | None]:
    """Parse ``'Away Full Name @ Home Full Name'`` into ``(away_abbr, home_abbr)``."""
    m = re.match(r"^(.+?) @ (.+?)$", str(desc))
    if not m:
        return None, None
    return _TEAM_ABBR.get(m.group(1).strip()), _TEAM_ABBR.get(m.group(2).strip())


def american_to_prob(odds: Any) -> float:
    """American odds → raw implied probability (includes vig)."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return float("nan")
    try:
        o = int(str(odds).replace("EVEN", "100"))
    except (TypeError, ValueError):
        return float("nan")
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)


def remove_vig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    """Vig-free the two sides of a binary market via proportional scaling."""
    s = p_a + p_b
    if s <= 0 or np.isnan(s):
        return float("nan"), float("nan")
    return p_a / s, p_b / s


def prob_to_american(p: float) -> int:
    """Fair probability → American odds (for reporting no-vig equivalent)."""
    if p is None or np.isnan(p) or p <= 0 or p >= 1:
        return 0
    if p >= 0.5:
        return int(round(-p / (1 - p) * 100))
    return int(round((1 - p) / p * 100))


# ---------------------------------------------------------------------------
# Odds snapshot extraction
# ---------------------------------------------------------------------------

@dataclass
class GameOdds:
    game_date: str
    away_abbr: str
    home_abbr: str
    snapshot_ts: pd.Timestamp
    ml_away: int | None = None
    ml_home: int | None = None
    spread_home_line: float | None = None
    spread_away_line: float | None = None
    total_line: float | None = None
    total_over_odds: int | None = None
    total_under_odds: int | None = None


_CLEAN_ROW_COUNT = 6  # 2 ML + 2 spread + 2 total = main lines only


def extract_closing_odds(
    odds_history: pd.DataFrame,
    game_date: str,
    source: str = "bovada",
    before_time_utc: str | None = None,
) -> pd.DataFrame:
    """Extract one row per game from a long-format odds history parquet.

    For each game, picks the latest snapshot (at or before
    ``before_time_utc``) where that game has exactly 6 rows — 2 moneyline
    + 2 spread + 2 total. This filters out snapshots contaminated by
    Bovada's alternate-line dumps (0.5 run lines, first-inning totals,
    etc.) that otherwise corrupt the row-0/row-1 parser.

    Uses Bovada's row ordering convention: for each ``(game, market_type)``,
    row 0 is away/over and row 1 is home/under.

    Parameters
    ----------
    odds_history : pd.DataFrame
        Long-format odds from ``game_odds_history.parquet``.
    game_date : str
        YYYY-MM-DD date to filter to.
    source : str
        ``'bovada'`` or ``'dk'``.
    before_time_utc : str, optional
        ISO UTC timestamp cutoff — only consider snapshots strictly before
        this. Use the earliest game's first pitch minus a safety margin
        (e.g. 30 min) to guarantee pre-game lines. ``None`` = no cutoff.

    Returns
    -------
    pd.DataFrame
        Columns: ``game_date, away_abbr, home_abbr, snapshot_ts,
        ml_away, ml_home, spread_home_line, spread_away_line, total_line,
        total_over_odds, total_under_odds``.
    """
    df = odds_history[
        (odds_history["game_date"].astype(str) == game_date)
        & (odds_history["source"] == source)
    ].copy()

    if before_time_utc is not None:
        df = df[df["snapshot_ts"].astype(str) < before_time_utc]

    if df.empty:
        return pd.DataFrame()

    # Per-game snapshot selection: within each game, find snapshots that
    # have exactly 6 rows (clean main-line snapshot) and take the latest.
    # This lets different games use different snapshots if their Bovada
    # feeds diverged on cadence.
    snap_counts = (
        df.groupby(["game_description", "snapshot_ts"]).size().reset_index(name="n")
    )
    clean = snap_counts[snap_counts["n"] == _CLEAN_ROW_COUNT]
    if clean.empty:
        logger.warning(
            "No clean %d-row snapshots found for %s/%s — lines may be "
            "contaminated by alternate markets",
            _CLEAN_ROW_COUNT, game_date, source,
        )
        return pd.DataFrame()

    latest_per_game = (
        clean.sort_values("snapshot_ts")
        .groupby("game_description", as_index=False)
        .last()[["game_description", "snapshot_ts"]]
        .rename(columns={"snapshot_ts": "chosen_ts"})
    )

    df = df.merge(latest_per_game, on="game_description")
    df = df[df["snapshot_ts"] == df["chosen_ts"]]

    logger.info(
        "extract_closing_odds: %d clean-snapshot games for %s/%s "
        "(latest ts: %s)",
        len(latest_per_game), game_date, source,
        latest_per_game["chosen_ts"].max(),
    )

    # Sort by (game, market) so row 0 = away/over, row 1 = home/under
    df = df.sort_values(["game_description", "market_type"])

    rows: list[dict[str, Any]] = []
    for game_desc, g in df.groupby("game_description"):
        away, home = _parse_game_description(game_desc)
        if away is None or home is None:
            continue

        rec: dict[str, Any] = {
            "game_date": game_date,
            "away_abbr": away,
            "home_abbr": home,
            "snapshot_ts": g["snapshot_ts"].iloc[0],
        }

        ml = g[g["market_type"] == "moneyline"].reset_index(drop=True)
        if len(ml) == 2:
            rec["ml_away"] = _safe_int(ml.iloc[0]["odds"])
            rec["ml_home"] = _safe_int(ml.iloc[1]["odds"])

        sp = g[g["market_type"] == "spread"].reset_index(drop=True)
        if len(sp) == 2:
            rec["spread_away_line"] = _safe_float(sp.iloc[0]["line"])
            rec["spread_home_line"] = _safe_float(sp.iloc[1]["line"])

        tot = g[g["market_type"] == "total"].reset_index(drop=True)
        if len(tot) == 2:
            rec["total_line"] = _safe_float(tot.iloc[0]["line"])
            rec["total_over_odds"] = _safe_int(tot.iloc[0]["odds"])
            rec["total_under_odds"] = _safe_int(tot.iloc[1]["odds"])

        rows.append(rec)

    return pd.DataFrame(rows)


def _safe_int(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(str(val).replace("EVEN", "100"))
    except (TypeError, ValueError):
        return None


def _safe_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------

def compute_edges(
    preds: pd.DataFrame,
    odds: pd.DataFrame,
    total_bias_correction: float = 0.0,
    margin_bias_correction: float = 0.0,
) -> pd.DataFrame:
    """Join predictions to odds and compute per-game edges across all three surfaces.

    Parameters
    ----------
    preds : pd.DataFrame
        Must include the columns from ``todays_game_predictions.parquet``:
        ``game_pk, away_abbr, home_abbr, p_away_win, p_home_win,
        away_runs_mean, away_runs_std, home_runs_mean, home_runs_std,
        total_mean, total_std, margin_mean``.
    odds : pd.DataFrame
        Output of :func:`extract_closing_odds`.
    total_bias_correction : float
        Subtracted from ``total_mean`` before computing P(over). Use a
        positive value when the model systematically under-predicts totals.
    margin_bias_correction : float
        Subtracted from ``margin_mean`` before computing P(home cover).

    Returns
    -------
    pd.DataFrame
        One row per game with model probabilities, book implied
        probabilities, and edges for ML/RL/Total.
    """
    if preds.empty or odds.empty:
        return pd.DataFrame()

    df = preds.merge(odds, on=["away_abbr", "home_abbr"], how="inner")

    # Margin std via independence approximation (sqrt of summed variances).
    df["margin_std"] = np.sqrt(
        df["away_runs_std"] ** 2 + df["home_runs_std"] ** 2
    )

    # Apply bias corrections
    total_adj = df["total_mean"] - total_bias_correction
    margin_adj = df["margin_mean"] - margin_bias_correction

    # --- Moneyline ---
    # ``p_away_win`` / ``p_home_win`` already come from the sim; book
    # implied probs need vig removal.
    df["book_ml_p_away_raw"] = df["ml_away"].apply(american_to_prob)
    df["book_ml_p_home_raw"] = df["ml_home"].apply(american_to_prob)
    fair_pairs = df.apply(
        lambda r: pd.Series(
            remove_vig_two_way(r["book_ml_p_away_raw"], r["book_ml_p_home_raw"]),
            index=["book_ml_p_away", "book_ml_p_home"],
        ),
        axis=1,
    )
    df = pd.concat([df, fair_pairs], axis=1)

    df["ml_edge_away"] = df["p_away_win"] - df["book_ml_p_away"]
    df["ml_edge_home"] = df["p_home_win"] - df["book_ml_p_home"]

    # --- Run line ---
    # spread_home_line semantics: negative = home favored by that many runs.
    # P(home covers -1.5) = P(margin < -1.5) where margin = away - home.
    df["p_home_cover_rl"] = norm.cdf(
        df["spread_home_line"].astype(float),
        loc=margin_adj,
        scale=df["margin_std"],
    )
    df["p_away_cover_rl"] = 1.0 - df["p_home_cover_rl"]

    # --- Totals ---
    df["p_over"] = 1.0 - norm.cdf(
        df["total_line"].astype(float),
        loc=total_adj,
        scale=df["total_std"],
    )
    df["p_under"] = 1.0 - df["p_over"]

    # Book total implied (vig-removed)
    df["book_tot_p_over_raw"] = df["total_over_odds"].apply(american_to_prob)
    df["book_tot_p_under_raw"] = df["total_under_odds"].apply(american_to_prob)
    fair_tot = df.apply(
        lambda r: pd.Series(
            remove_vig_two_way(r["book_tot_p_over_raw"], r["book_tot_p_under_raw"]),
            index=["book_tot_p_over", "book_tot_p_under"],
        ),
        axis=1,
    )
    df = pd.concat([df, fair_tot], axis=1)

    df["total_edge_over"] = df["p_over"] - df["book_tot_p_over"]
    df["total_edge_under"] = df["p_under"] - df["book_tot_p_under"]

    return df


def pick_side(
    row: pd.Series,
    surface: str,
    min_edge: float = 0.03,
) -> tuple[str | None, float]:
    """Return ``(pick_label, edge)`` for the best side on a given surface.

    Returns ``(None, 0.0)`` when the best edge is below ``min_edge``.
    """
    if surface == "ml":
        a_edge = row.get("ml_edge_away", np.nan)
        h_edge = row.get("ml_edge_home", np.nan)
        if pd.isna(a_edge) or pd.isna(h_edge):
            return None, 0.0
        if a_edge > h_edge and a_edge >= min_edge:
            return row["away_abbr"], float(a_edge)
        if h_edge > a_edge and h_edge >= min_edge:
            return row["home_abbr"], float(h_edge)
        return None, 0.0
    if surface == "total":
        o_edge = row.get("total_edge_over", np.nan)
        u_edge = row.get("total_edge_under", np.nan)
        if pd.isna(o_edge) or pd.isna(u_edge):
            return None, 0.0
        if o_edge > u_edge and o_edge >= min_edge:
            return "OVER", float(o_edge)
        if u_edge > o_edge and u_edge >= min_edge:
            return "UNDER", float(u_edge)
        return None, 0.0
    raise ValueError(f"Unknown surface: {surface!r}")


# ---------------------------------------------------------------------------
# Actuals join + result computation
# ---------------------------------------------------------------------------

def compute_results(
    edges_df: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Join edges to game results and tag each pick as win/loss/push.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Output of :func:`compute_edges`.
    actuals : pd.DataFrame
        Must have columns ``game_pk, away_score, home_score`` (from
        ``fact_game_totals`` or the schedule).

    Returns
    -------
    pd.DataFrame
        ``edges_df`` with added columns:
        ``actual_away, actual_home, actual_total, actual_margin,
        ml_result, rl_result, total_result`` where each result is one of
        ``{'win', 'loss', 'push', None}`` depending on the pick.
    """
    if edges_df.empty:
        return edges_df

    df = edges_df.merge(
        actuals.rename(columns={
            "away_score": "actual_away",
            "home_score": "actual_home",
        })[["game_pk", "actual_away", "actual_home"]],
        on="game_pk",
        how="left",
    )

    df["actual_total"] = df["actual_away"] + df["actual_home"]
    df["actual_margin"] = df["actual_away"] - df["actual_home"]

    # --- ML result: pick side with positive edge (use away by default) ---
    def _ml_outcome(row: pd.Series) -> str | None:
        if pd.isna(row.get("actual_away")):
            return None
        away_edge = row.get("ml_edge_away", np.nan)
        home_edge = row.get("ml_edge_home", np.nan)
        if pd.isna(away_edge) or pd.isna(home_edge):
            return None
        if away_edge <= 0 and home_edge <= 0:
            return None  # no positive-edge pick
        picked_away = away_edge >= home_edge
        away_won = row["actual_margin"] > 0
        if row["actual_margin"] == 0:
            return "push"
        return "win" if picked_away == away_won else "loss"

    df["ml_result"] = df.apply(_ml_outcome, axis=1)

    # --- RL result ---
    def _rl_outcome(row: pd.Series) -> str | None:
        if pd.isna(row.get("actual_away")) or pd.isna(row.get("spread_home_line")):
            return None
        # Pick the side the model prefers
        p_home = row.get("p_home_cover_rl", np.nan)
        if pd.isna(p_home):
            return None
        # Actual home cover: actual_margin < spread_home_line
        home_covered = row["actual_margin"] < row["spread_home_line"]
        away_covered = row["actual_margin"] > -float(row["spread_home_line"]) * (-1)
        # Simpler: away covers when actual_margin > (-spread_home_line)
        away_covered = row["actual_margin"] > -row["spread_home_line"]
        model_picks_home = p_home >= 0.5
        picked_won = home_covered if model_picks_home else away_covered
        # Pushes (rare with -1.5)
        if row["actual_margin"] == row["spread_home_line"]:
            return "push"
        return "win" if picked_won else "loss"

    df["rl_result"] = df.apply(_rl_outcome, axis=1)

    # --- Totals result ---
    def _tot_outcome(row: pd.Series) -> str | None:
        if pd.isna(row.get("actual_total")) or pd.isna(row.get("total_line")):
            return None
        p_over = row.get("p_over", np.nan)
        if pd.isna(p_over):
            return None
        if row["actual_total"] == row["total_line"]:
            return "push"
        over_hit = row["actual_total"] > row["total_line"]
        model_picks_over = p_over >= 0.5
        return "win" if over_hit == model_picks_over else "loss"

    df["total_result"] = df.apply(_tot_outcome, axis=1)

    return df


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------

def compute_bias_correction(
    archive: pd.DataFrame,
) -> dict[str, float]:
    """Learn simple additive bias corrections from completed archive games.

    Uses only rows where ``actual_total`` and ``actual_margin`` are present.
    Returns ``{'total': float, 'margin': float}`` — subtract these from
    predictions to get a bias-corrected estimate.

    When the archive is empty or has no completed games, returns zeros.
    """
    if archive.empty or "actual_total" not in archive.columns:
        return {"total": 0.0, "margin": 0.0}

    done = archive.dropna(subset=["actual_total", "actual_margin"])
    if done.empty:
        return {"total": 0.0, "margin": 0.0}

    total_bias = float((done["total_mean"] - done["actual_total"]).mean())
    margin_bias = float((done["margin_mean"] - done["actual_margin"]).mean())
    return {"total": total_bias, "margin": margin_bias}


# ---------------------------------------------------------------------------
# Archive append
# ---------------------------------------------------------------------------

ARCHIVE_KEYS: tuple[str, ...] = ("game_date", "game_pk")


def append_to_archive(
    preds: pd.DataFrame,
    archive_path: Path,
    game_date: str,
) -> pd.DataFrame:
    """Append today's predictions to the running archive parquet.

    Dedups on ``(game_date, game_pk)`` so re-running the daily pipeline
    during the same slate overwrites existing rows rather than duplicating.

    Parameters
    ----------
    preds : pd.DataFrame
        Output of the game sim (``todays_game_predictions.parquet``).
    archive_path : Path
        Destination parquet (created if absent).
    game_date : str
        YYYY-MM-DD; stamped onto each row.

    Returns
    -------
    pd.DataFrame
        The full archive post-append (useful for diagnostics).
    """
    tagged = preds.copy()
    tagged["game_date"] = game_date

    if archive_path.exists():
        existing = pd.read_parquet(archive_path)
        # Drop any existing rows for the same (game_date, game_pk) pairs so
        # the new ones overwrite.
        keys = list(ARCHIVE_KEYS)
        mask = ~existing.set_index(keys).index.isin(
            tagged.set_index(keys).index
        )
        combined = pd.concat([existing[mask], tagged], ignore_index=True)
    else:
        combined = tagged

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(archive_path, index=False)
    logger.info(
        "Archive updated: +%d rows (total %d) → %s",
        len(tagged), len(combined), archive_path,
    )
    return combined


# ---------------------------------------------------------------------------
# Reporting aggregations
# ---------------------------------------------------------------------------

def summarize_performance(
    results_df: pd.DataFrame,
    edge_buckets: tuple[float, ...] = (0.0, 0.03, 0.05, 0.10),
) -> dict[str, pd.DataFrame]:
    """Aggregate hit rate and unit ROI by surface + edge bucket.

    Assumes bets at -110 (implied 52.38%). Pushes don't count toward W/L.
    """
    out: dict[str, pd.DataFrame] = {}
    if results_df.empty:
        return out

    surfaces = [
        ("ml", "ml_edge_away", "ml_edge_home", "ml_result"),
        ("total", "total_edge_over", "total_edge_under", "total_result"),
    ]

    for surface, a_col, b_col, r_col in surfaces:
        if r_col not in results_df.columns:
            continue
        df = results_df.copy()
        df["best_edge"] = np.maximum(
            df[a_col].fillna(-np.inf), df[b_col].fillna(-np.inf),
        )
        df = df[df[r_col].notna()]
        if df.empty:
            continue

        rows: list[dict[str, Any]] = []
        for lo in edge_buckets:
            sub = df[df["best_edge"] >= lo]
            n = len(sub)
            if n == 0:
                rows.append({
                    "edge_min": lo, "n_bets": 0,
                    "wins": 0, "losses": 0, "pushes": 0,
                    "hit_rate": float("nan"), "roi": float("nan"),
                })
                continue
            wins = int((sub[r_col] == "win").sum())
            losses = int((sub[r_col] == "loss").sum())
            pushes = int((sub[r_col] == "push").sum())
            decided = wins + losses
            hit = wins / decided if decided > 0 else float("nan")
            # -110 economics: wins +100, losses -110, pushes 0
            roi = (
                (wins * 100 - losses * 110) / (decided * 110)
                if decided > 0 else float("nan")
            )
            rows.append({
                "edge_min": lo,
                "n_bets": n,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "hit_rate": hit,
                "roi": roi,
            })
        out[surface] = pd.DataFrame(rows)

    return out
