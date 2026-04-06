"""
Shared backtest harness for lineup simulator backtests.

Extracts duplicated constants, SQL query logic, simulation scaffolding,
and calibration reporting into reusable functions.  Each backtest script
imports what it needs and supplies only the parts that vary (extra SQL
columns, per-game context kwargs, etc.).

This is a pure structural refactor -- zero behavioral changes.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from scripts.precompute import DASHBOARD_DIR
from src.data.db import read_sql
from src.models.game_sim.lineup_simulator import simulate_lineup_game

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------

PROP_LINES: dict[str, list[float]] = {
    "h": [0.5, 1.5, 2.5],
    "r": [0.5, 1.5],
    "rbi": [0.5, 1.5],
    "hrr": [0.5, 1.5, 2.5, 3.5],
    "k": [0.5, 1.5],
    "bb": [0.5],
}

PRIMARY_LINE: dict[str, float] = {
    "h": 0.5,
    "r": 0.5,
    "rbi": 0.5,
    "hrr": 1.5,
    "k": 0.5,
    "bb": 0.5,
}

ALL_STATS: list[str] = ["h", "r", "rbi", "hrr", "k", "bb"]

DEFAULT_BP_K_RATE: float = 0.253
DEFAULT_BP_BB_RATE: float = 0.084
DEFAULT_BP_HR_RATE: float = 0.024

N_SIMS: int = 10_000
BATCH_SIZE: int = 25

# League-average batter fallback parameters (rate, std, min, max)
_FALLBACK_K = (0.226, 0.03, 0.05, 0.50)
_FALLBACK_BB = (0.082, 0.02, 0.02, 0.25)
_FALLBACK_HR = (0.031, 0.01, 0.005, 0.10)
_FALLBACK_N_SAMPLES = 2000


# -----------------------------------------------------------------------
# Shared SQL query
# -----------------------------------------------------------------------

def fetch_backtest_games(
    seasons: list[int],
    max_games: int | None = None,
    extra_select: list[str] | None = None,
    extra_joins: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fetch backtest game data with optional extra columns.

    The core query pulls full 9-man lineups joined to batting boxscores
    and opposing starters.  Callers can inject additional SELECT columns
    (e.g. ``gs.venue_id``, ``du.hp_umpire_name``) and JOIN clauses
    (e.g. a LEFT JOIN to ``dim_umpire``).

    Parameters
    ----------
    seasons : list[int]
        Seasons to pull games from.
    max_games : int | None
        If set, sample down to this many team-game sides.
    extra_select : list[str] | None
        Additional SQL SELECT expressions to append.
    extra_joins : list[str] | None
        Additional JOIN / LEFT JOIN clauses to inject after the
        main FROM/JOIN block.
    random_state : int
        Random seed for reproducible sampling when max_games is set.

    Returns
    -------
    pd.DataFrame
        One row per batter per game, with batting_order 1-9 and actuals.
        Always includes the derived ``actual_hrr`` column.
    """
    season_list = ", ".join(str(s) for s in seasons)

    select_extras = ""
    if extra_select:
        select_extras = ",\n        " + ",\n        ".join(extra_select)

    join_extras = ""
    if extra_joins:
        join_extras = "\n    " + "\n    ".join(extra_joins)

    query = f"""
    WITH starters AS (
        SELECT
            pb.pitcher_id,
            pb.game_pk,
            pb.team_id AS pitcher_team_id
        FROM staging.pitching_boxscores pb
        WHERE pb.is_starter = TRUE
          AND pb.batters_faced >= 9
    ),
    game_season AS (
        SELECT game_pk, season, home_team_id, away_team_id, venue_id
        FROM production.dim_game
        WHERE game_type = 'R' AND season IN ({season_list})
    )
    SELECT
        fl.game_pk,
        gs.season,
        fl.player_id AS batter_id,
        fl.batting_order,
        fl.team_id,
        bb.hits AS actual_h,
        bb.runs AS actual_r,
        bb.rbi AS actual_rbi,
        bb.strikeouts AS actual_k,
        bb.walks AS actual_bb,
        bb.home_runs AS actual_hr,
        bb.doubles AS actual_2b,
        bb.triples AS actual_3b,
        bb.total_bases AS actual_tb,
        bb.plate_appearances AS actual_pa,
        st.pitcher_id AS opp_starter_id,
        st.pitcher_team_id AS opp_team_id{select_extras}
    FROM production.fact_lineup fl
    JOIN game_season gs ON fl.game_pk = gs.game_pk
    JOIN staging.batting_boxscores bb
      ON fl.player_id = bb.batter_id AND fl.game_pk = bb.game_pk
    LEFT JOIN starters st
      ON fl.game_pk = st.game_pk AND st.pitcher_team_id != fl.team_id{join_extras}
    WHERE fl.is_starter = TRUE
      AND fl.batting_order BETWEEN 1 AND 9
      AND bb.plate_appearances >= 1
    ORDER BY fl.game_pk, fl.team_id, fl.batting_order
    """
    logger.info("Fetching backtest game data for seasons %s...", seasons)
    df = read_sql(query)
    logger.info("Raw rows: %d", len(df))

    if df.empty:
        return df

    # Filter to games that have exactly 9 batters per team-side
    counts = df.groupby(["game_pk", "team_id"]).size().reset_index(name="n_batters")
    full_sides = counts[counts["n_batters"] == 9][["game_pk", "team_id"]]
    df = df.merge(full_sides, on=["game_pk", "team_id"])

    # Filter to sides that have an opposing starter identified
    df = df[df["opp_starter_id"].notna()].copy()
    df["opp_starter_id"] = df["opp_starter_id"].astype(int)

    # Compute HRR actual
    df["actual_hrr"] = df["actual_h"] + df["actual_r"] + df["actual_rbi"]

    n_sides = df.groupby(["game_pk", "team_id"]).ngroups
    logger.info(
        "After filtering to full 9-man lineups with starters: %d rows (%d team-game sides)",
        len(df), n_sides,
    )

    if max_games is not None and n_sides > max_games:
        # Sample at the team-game level
        side_keys = df[["game_pk", "team_id"]].drop_duplicates()
        sampled = side_keys.sample(n=max_games, random_state=random_state)
        df = df.merge(sampled, on=["game_pk", "team_id"])
        logger.info("Sampled down to %d team-game sides (%d rows)", max_games, len(df))

    return df


# -----------------------------------------------------------------------
# Shared simulation function
# -----------------------------------------------------------------------

def simulate_one_side(
    side_df: pd.DataFrame,
    posteriors: dict,
    prop_lines: dict[str, list[float]],
    n_sims: int = N_SIMS,
    rng: np.random.Generator | None = None,
    context_kwargs: dict[str, Any] | None = None,
    extra_record_fields: dict[str, Any] | None = None,
) -> list[dict] | None:
    """Simulate one side (home/away) of a game and compute prop predictions.

    Handles: checking pitcher posteriors, building batter sample lists
    with league-average fallback, retrieving BF priors and bullpen rates,
    calling ``simulate_lineup_game()``, extracting per-batter predictions,
    and computing P(over) at each prop line.

    Parameters
    ----------
    side_df : pd.DataFrame
        Rows for one team-side (9 batters), must include batter_id,
        batting_order, opp_starter_id, opp_team_id, season, game_pk,
        team_id, and actual_* columns.
    posteriors : dict
        Output of ``load_posteriors()`` -- NPZ files, bf_lookup, bp_lookup.
    prop_lines : dict[str, list[float]]
        Stat -> list of lines to compute P(over) at.
    n_sims : int
        Monte Carlo simulations per game.
    rng : np.random.Generator | None
        Optional RNG; a default is created per batter if not provided.
    context_kwargs : dict[str, Any] | None
        Extra keyword arguments passed directly to
        ``simulate_lineup_game()`` (e.g. park lifts, umpire lifts,
        batter_babip_adjs).
    extra_record_fields : dict[str, Any] | None
        Extra key-value pairs to include in every batter record.

    Returns
    -------
    list[dict] | None
        One dict per batter with predictions and actuals, or None if
        posteriors are missing for too many batters.
    """
    side_df = side_df.sort_values("batting_order")
    batter_ids = side_df["batter_id"].astype(int).tolist()
    opp_starter_id = int(side_df.iloc[0]["opp_starter_id"])
    opp_team_id = int(side_df.iloc[0]["opp_team_id"])
    pid_str = str(opp_starter_id)

    # Check pitcher posteriors
    for npz_key in ["pitcher_k", "pitcher_bb", "pitcher_hr"]:
        if pid_str not in posteriors[npz_key].files:
            return None

    # Build batter sample lists (need all 9)
    batter_k_samples: list[np.ndarray] = []
    batter_bb_samples: list[np.ndarray] = []
    batter_hr_samples: list[np.ndarray] = []
    valid_count = 0

    for bid in batter_ids:
        bid_str = str(bid)
        has_k = bid_str in posteriors["hitter_k"].files
        has_bb = bid_str in posteriors["hitter_bb"].files
        has_hr = bid_str in posteriors["hitter_hr"].files

        if has_k and has_bb and has_hr:
            batter_k_samples.append(posteriors["hitter_k"][bid_str])
            batter_bb_samples.append(posteriors["hitter_bb"][bid_str])
            batter_hr_samples.append(posteriors["hitter_hr"][bid_str])
            valid_count += 1
        else:
            # Use league-average fallback for missing batters
            fb_rng = np.random.default_rng(bid % 100000)
            mu_k, std_k, lo_k, hi_k = _FALLBACK_K
            mu_bb, std_bb, lo_bb, hi_bb = _FALLBACK_BB
            mu_hr, std_hr, lo_hr, hi_hr = _FALLBACK_HR
            batter_k_samples.append(
                np.clip(fb_rng.normal(mu_k, std_k, _FALLBACK_N_SAMPLES), lo_k, hi_k)
            )
            batter_bb_samples.append(
                np.clip(fb_rng.normal(mu_bb, std_bb, _FALLBACK_N_SAMPLES), lo_bb, hi_bb)
            )
            batter_hr_samples.append(
                np.clip(fb_rng.normal(mu_hr, std_hr, _FALLBACK_N_SAMPLES), lo_hr, hi_hr)
            )

    # Require at least 5 of 9 batters with real posteriors
    if valid_count < 5:
        return None

    # Pitcher rates (posterior means)
    starter_k = float(np.mean(posteriors["pitcher_k"][pid_str]))
    starter_bb = float(np.mean(posteriors["pitcher_bb"][pid_str]))
    starter_hr = float(np.mean(posteriors["pitcher_hr"][pid_str]))

    # BF priors
    bf_info = posteriors["bf_lookup"].get(opp_starter_id)
    if bf_info:
        mu_bf = bf_info["mu_bf"]
        sigma_bf = bf_info["sigma_bf"]
    else:
        mu_bf = 22.0
        sigma_bf = 3.4

    # Bullpen rates
    bp_info = posteriors["bp_lookup"].get(opp_team_id)
    if bp_info:
        bp_k = bp_info["k_rate"]
        bp_bb = bp_info["bb_rate"]
        bp_hr = bp_info["hr_rate"]
    else:
        bp_k = DEFAULT_BP_K_RATE
        bp_bb = DEFAULT_BP_BB_RATE
        bp_hr = DEFAULT_BP_HR_RATE

    # Run simulation
    game_pk = int(side_df.iloc[0]["game_pk"])
    sim_kwargs: dict[str, Any] = dict(
        batter_k_rate_samples=batter_k_samples,
        batter_bb_rate_samples=batter_bb_samples,
        batter_hr_rate_samples=batter_hr_samples,
        starter_k_rate=starter_k,
        starter_bb_rate=starter_bb,
        starter_hr_rate=starter_hr,
        starter_bf_mu=mu_bf,
        starter_bf_sigma=sigma_bf,
        bullpen_k_rate=bp_k,
        bullpen_bb_rate=bp_bb,
        bullpen_hr_rate=bp_hr,
        n_sims=n_sims,
        random_seed=game_pk % (2**31),
    )
    if context_kwargs:
        sim_kwargs.update(context_kwargs)

    result = simulate_lineup_game(**sim_kwargs)

    # Extract per-batter predictions
    records: list[dict] = []
    for i, (_, row) in enumerate(side_df.iterrows()):
        bid = int(row["batter_id"])
        bid_str = str(bid)
        has_posterior = (
            bid_str in posteriors["hitter_k"].files
            and bid_str in posteriors["hitter_bb"].files
            and bid_str in posteriors["hitter_hr"].files
        )

        br = result.batter_result(i)
        h_samples = br["h_samples"]
        r_samples = br["r_samples"]
        rbi_samples = br["rbi_samples"]
        k_samples = br["k_samples"]
        bb_samples = br["bb_samples"]
        hrr_samples = h_samples + r_samples + rbi_samples

        rec: dict[str, Any] = {
            "game_pk": int(row["game_pk"]),
            "season": int(row["season"]),
            "batter_id": bid,
            "batting_order": int(row["batting_order"]),
            "team_id": int(row["team_id"]),
            "opp_starter_id": int(row["opp_starter_id"]),
            "has_posterior": has_posterior,
            # Actuals
            "actual_h": int(row["actual_h"]),
            "actual_r": int(row["actual_r"]),
            "actual_rbi": int(row["actual_rbi"]),
            "actual_hrr": int(row["actual_hrr"]),
            "actual_k": int(row["actual_k"]),
            "actual_bb": int(row["actual_bb"]),
            "actual_pa": int(row["actual_pa"]),
            # Predicted means
            "pred_h": float(np.mean(h_samples)),
            "pred_r": float(np.mean(r_samples)),
            "pred_rbi": float(np.mean(rbi_samples)),
            "pred_hrr": float(np.mean(hrr_samples)),
            "pred_k": float(np.mean(k_samples)),
            "pred_bb": float(np.mean(bb_samples)),
        }

        # Extra fields supplied by the caller
        if extra_record_fields:
            rec.update(extra_record_fields)

        # Per-row extras from the side_df (e.g. venue_id, hp_umpire_name)
        # are NOT automatically copied here -- callers that need them
        # should include them in extra_record_fields or post-process.

        # P(over) at each prop line
        for stat, lines in prop_lines.items():
            if stat == "hrr":
                samples = hrr_samples
            else:
                samples = br[f"{stat}_samples"]
            for line in lines:
                p_over = float(np.mean(samples > line))
                rec[f"p_over_{stat}_{line}"] = p_over

        records.append(rec)

    return records


# -----------------------------------------------------------------------
# Simulation loop helper
# -----------------------------------------------------------------------

def run_sides_loop(
    label: str,
    sides: list[tuple],
    simulate_fn,
    batch_size: int = BATCH_SIZE,
) -> tuple[pd.DataFrame, int]:
    """Run *simulate_fn* on every side with progress logging.

    Parameters
    ----------
    label : str
        Human-readable label for progress messages.
    sides : list[tuple]
        Output of ``list(df.groupby(["game_pk", "team_id"]))``.
    simulate_fn : callable
        ``(side_df) -> list[dict] | None``  -- called once per side.
    batch_size : int
        Print progress every *batch_size* sides.

    Returns
    -------
    (pd.DataFrame, int)
        DataFrame of all batter records and count of skipped sides.
    """
    all_records: list[dict] = []
    n_skipped = 0
    n_sides = len(sides)
    t0 = time.perf_counter()

    for idx, ((game_pk, team_id), side_df) in enumerate(sides):
        recs = simulate_fn(side_df)
        if recs is None:
            n_skipped += 1
        else:
            all_records.extend(recs)

        if (idx + 1) % batch_size == 0 or idx == n_sides - 1:
            elapsed = time.perf_counter() - t0
            pct = 100 * (idx + 1) / n_sides
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "[%s] %d/%d (%.0f%%) | %.1f sides/sec | %d skipped | %d batters",
                label, idx + 1, n_sides, pct, rate, n_skipped, len(all_records),
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s] Done: %d batter-games in %.1fs. Skipped %d.",
        label, len(all_records), elapsed, n_skipped,
    )
    return pd.DataFrame(all_records), n_skipped


# -----------------------------------------------------------------------
# Calibration reporting
# -----------------------------------------------------------------------

_DEFAULT_BUCKET_EDGES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
_DEFAULT_BUCKET_LABELS = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]


def report_calibration_buckets(
    all_predicted: np.ndarray,
    all_actual: np.ndarray,
    bucket_edges: list[float] | None = None,
    label: str = "",
) -> None:
    """Print calibration table by confidence bucket.

    Parameters
    ----------
    all_predicted : np.ndarray
        Predicted P(over) values (pooled across stats/lines).
    all_actual : np.ndarray
        Binary actuals (1 if over, 0 if under).
    bucket_edges : list[float] | None
        Bucket boundaries; defaults to [0.50 .. 0.75, 1.01].
    label : str
        Optional label for the section header.
    """
    if bucket_edges is None:
        bucket_edges = _DEFAULT_BUCKET_EDGES
    bucket_labels = []
    for i in range(len(bucket_edges) - 1):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]
        if hi > 1.0:
            bucket_labels.append(f"{int(lo * 100)}%+")
        else:
            bucket_labels.append(f"{int(lo * 100)}-{int(hi * 100)}%")

    header = "CALIBRATION BY CONFIDENCE BUCKET"
    if label:
        header += f" ({label})"
    print(f"  {'Bucket':>8s}  {'Predicted':>10s}  {'Actual':>10s}  "
          f"{'N':>8s}  {'Gap':>8s}")
    print(f"  {'--------':>8s}  {'----------':>10s}  {'----------':>10s}  "
          f"{'--------':>8s}  {'--------':>8s}")

    for i, blabel in enumerate(bucket_labels):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]
        mask = (all_predicted >= lo) & (all_predicted < hi)
        n_in = mask.sum()
        if n_in == 0:
            print(f"  {blabel:>8s}  {'--':>10s}  {'--':>10s}  {0:>8d}  {'--':>8s}")
            continue
        pred_mean = float(all_predicted[mask].mean())
        actual_mean = float(all_actual[mask].mean())
        gap = actual_mean - pred_mean
        print(f"  {blabel:>8s}  {pred_mean:>10.4f}  {actual_mean:>10.4f}  "
              f"{n_in:>8,}  {gap:>+8.4f}")


def report_per_stat_calibration(
    records_df: pd.DataFrame,
    stats: list[str],
    prop_lines: dict[str, list[float]],
    primary_lines: dict[str, float],
) -> None:
    """Print per-stat calibration at each prop line.

    Parameters
    ----------
    records_df : pd.DataFrame
        Backtest results filtered to batters with real posteriors.
    stats : list[str]
        List of stat names to report on.
    prop_lines : dict[str, list[float]]
        Stat -> list of prop lines.
    primary_lines : dict[str, float]
        Stat -> primary line (used for headline).
    """
    for stat in stats:
        lines = prop_lines.get(stat, [])
        if not lines:
            continue
        print(f"\n  {stat.upper()}:")
        print(f"    {'Line':>5s}  {'P(over) mean':>13s}  {'Actual %':>10s}  "
              f"{'Gap':>8s}  {'N':>8s}  {'>=63% hit':>10s}  {'N_63':>6s}")

        for line in lines:
            col = f"p_over_{stat}_{line}"
            if col not in records_df.columns:
                continue
            p = records_df[col].values
            actual = (records_df[f"actual_{stat}"].values > line).astype(float)

            avg_p = float(p.mean())
            actual_rate = float(actual.mean())
            gap = actual_rate - avg_p
            n = len(p)

            # Hit rate at >= 63%
            m63 = p >= 0.63
            if m63.sum() > 0:
                hr63 = float(actual[m63].mean())
                n63 = m63.sum()
                hr63_str = f"{hr63:.4f}"
                n63_str = f"{n63:,}"
            else:
                hr63_str = "--"
                n63_str = "0"

            print(f"    {line:>5.1f}  {avg_p:>13.4f}  {actual_rate:>10.4f}  "
                  f"{gap:>+8.4f}  {n:>8,}  {hr63_str:>10s}  {n63_str:>6s}")


def pool_predictions(
    df: pd.DataFrame,
    stats: list[str],
    prop_lines: dict[str, list[float]],
    p_col_template: str = "p_over_{stat}_{line}",
) -> tuple[np.ndarray, np.ndarray]:
    """Pool all stat-line P(over) and actuals into flat arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered backtest results.
    stats : list[str]
        Stats to pool.
    prop_lines : dict[str, list[float]]
        Stat -> list of prop lines.
    p_col_template : str
        Column name template; ``{stat}`` and ``{line}`` are replaced.

    Returns
    -------
    (all_predicted, all_actual) : tuple[np.ndarray, np.ndarray]
    """
    all_p: list[np.ndarray] = []
    all_a: list[np.ndarray] = []

    for stat in stats:
        for line in prop_lines.get(stat, []):
            col = p_col_template.format(stat=stat, line=line)
            if col not in df.columns:
                continue
            p = df[col].values
            actual = (df[f"actual_{stat}"].values > line).astype(float)
            all_p.append(p)
            all_a.append(actual)

    if not all_p:
        return np.array([]), np.array([])
    return np.concatenate(all_p), np.concatenate(all_a)
