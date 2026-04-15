"""BvP history research: is career pitcher-batter history real signal or noise?

Three diagnostics, answered against production.fact_pa directly so we can
season-stamp and chronologically split (the cumulative fact_matchup_history
table does not support split-half or walk-forward).

1. Coverage — per (batter, pitcher) career PA distribution. For real 2024
   starter-lineup matchups, what share of pairs reach meaningful PA counts?

2. Split-half reliability — for pairs with >= 20 career PAs, split PAs
   chronologically into two halves by plate appearance order, compute K%
   and BB% per half, and report correlation. This is the standard
   sabermetric signal-vs-noise test.

3. Incremental predictive value — hold out 2024 PAs. For each pair that
   has >= N prior PAs (N in {10, 20, 50}), compute pre-2024 career K% and
   BB%. Score 2024 PAs under:
       (a) flat population rate
       (b) league_rate + pitcher_delta + hitter_delta  (current raw matchup)
       (c) (b) + reliability-weighted BvP residual
   Compare log-loss. If (c) does not beat (b), BvP history is noise at the
   population level.

Output: memory/bvp_history_findings_2026_04_11.md-ready JSON and numbers
printed to stdout.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db import read_sql

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_SEASON = 2024
CAREER_PA_BUCKETS = [0, 5, 10, 20, 30, 50, 100, 1_000_000]
RELIABILITY_PA = 50

OUT_JSON = Path("outputs/bvp_history_findings.parquet")


def _logit(p: float | np.ndarray) -> float | np.ndarray:
    p_arr = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p_arr / (1.0 - p_arr))


def _inv_logit(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def diagnostic_1_coverage() -> pd.DataFrame:
    """Career PA distribution per pair, plus coverage against 2024 starters."""
    logger.info("=" * 60)
    logger.info("Diagnostic 1: coverage")
    logger.info("=" * 60)

    # Career PAs per pair across all historical seasons (pre-2024)
    q = f"""
        SELECT
            pa.batter_id,
            pa.pitcher_id,
            COUNT(*) AS career_pa
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.season < {HOLDOUT_SEASON}
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL
          AND pa.events <> ''
        GROUP BY pa.batter_id, pa.pitcher_id
    """
    logger.info("Loading pre-2024 career BvP PA counts...")
    df_career = read_sql(q)
    logger.info("  %d distinct pre-2024 pairs", len(df_career))

    # Bucket distribution
    buckets = pd.cut(
        df_career["career_pa"],
        bins=CAREER_PA_BUCKETS,
        labels=[f"{CAREER_PA_BUCKETS[i]}-{CAREER_PA_BUCKETS[i+1]-1}"
                for i in range(len(CAREER_PA_BUCKETS) - 1)],
        right=False,
    )
    bucket_counts = buckets.value_counts().sort_index()
    logger.info("\nAll-pair career PA distribution (pre-2024):")
    for bucket, count in bucket_counts.items():
        logger.info("  %-10s  %8d  (%.1f%%)",
                    str(bucket), count, count / len(df_career) * 100)

    # Coverage against actual 2024 starter-lineup matchups
    q_2024 = f"""
        WITH pitcher_game AS (
            SELECT
                pa.game_pk,
                pa.pitcher_id,
                MAX(pa.pitcher_pa_number) AS bf,
                BOOL_OR(pa.pitcher_pa_number = 1) AS faced_first
            FROM production.fact_pa pa
            JOIN production.dim_game g USING (game_pk)
            WHERE g.season = {HOLDOUT_SEASON}
              AND g.game_type = 'R'
            GROUP BY pa.game_pk, pa.pitcher_id
        ),
        starters AS (
            SELECT game_pk, pitcher_id FROM pitcher_game
            WHERE faced_first AND bf >= 15
        )
        SELECT DISTINCT
            s.pitcher_id,
            pa.batter_id
        FROM starters s
        JOIN production.fact_pa pa
          ON pa.game_pk = s.game_pk AND pa.pitcher_id = s.pitcher_id
    """
    logger.info("\nLoading 2024 starter-batter pairs...")
    df_2024_pairs = read_sql(q_2024)
    logger.info("  %d distinct 2024 starter-batter pairs", len(df_2024_pairs))

    merged = df_2024_pairs.merge(
        df_career, on=["batter_id", "pitcher_id"], how="left"
    )
    merged["career_pa"] = merged["career_pa"].fillna(0).astype(int)

    n_total = len(merged)
    logger.info("\n2024 starter-batter pair coverage by pre-2024 career PA:")
    for lo, hi in zip(CAREER_PA_BUCKETS[:-1], CAREER_PA_BUCKETS[1:]):
        n = ((merged["career_pa"] >= lo) & (merged["career_pa"] < hi)).sum()
        logger.info("  %6d - %6d : %8d (%.1f%%)",
                    lo, hi - 1 if hi < 1_000_000 else 99999, n, n / n_total * 100)

    return merged


def diagnostic_2_split_half() -> dict:
    """Split-half reliability for pairs with >= 20 career PAs."""
    logger.info("\n" + "=" * 60)
    logger.info("Diagnostic 2: split-half reliability")
    logger.info("=" * 60)

    q = """
        SELECT
            pa.batter_id,
            pa.pitcher_id,
            pa.pa_id,
            g.game_date,
            (pa.events IN ('strikeout', 'strikeout_double_play'))::int AS is_k,
            (pa.events IN ('walk'))::int AS is_bb
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.game_type = 'R'
          AND pa.events IS NOT NULL
          AND pa.events <> ''
          AND pa.events <> 'intent_walk'
          AND (pa.batter_id, pa.pitcher_id) IN (
              SELECT batter_id, pitcher_id
              FROM production.fact_pa pa2
              JOIN production.dim_game g2 USING (game_pk)
              WHERE g2.game_type = 'R'
                AND pa2.events IS NOT NULL
                AND pa2.events <> ''
                AND pa2.events <> 'intent_walk'
              GROUP BY batter_id, pitcher_id
              HAVING COUNT(*) >= 20
          )
    """
    logger.info("Loading PAs for pairs with career >= 20 PAs...")
    df = read_sql(q)
    logger.info("  %d PAs across %d pairs",
                len(df), df.groupby(["batter_id", "pitcher_id"]).ngroups)

    df = df.sort_values(["batter_id", "pitcher_id", "game_date", "pa_id"]).reset_index(drop=True)
    df["pa_order"] = df.groupby(["batter_id", "pitcher_id"]).cumcount()
    df["total_pa"] = df.groupby(["batter_id", "pitcher_id"])["pa_id"].transform("count")
    df["half"] = np.where(df["pa_order"] < df["total_pa"] / 2, "first", "second")

    halves = df.groupby(["batter_id", "pitcher_id", "half"]).agg(
        pa=("pa_id", "count"),
        k_rate=("is_k", "mean"),
        bb_rate=("is_bb", "mean"),
    ).reset_index()

    pivot_k = halves.pivot(
        index=["batter_id", "pitcher_id"], columns="half", values="k_rate"
    ).dropna()
    pivot_bb = halves.pivot(
        index=["batter_id", "pitcher_id"], columns="half", values="bb_rate"
    ).dropna()
    pivot_pa_first = halves[halves["half"] == "first"].set_index(
        ["batter_id", "pitcher_id"]
    )["pa"]
    pivot_pa_second = halves[halves["half"] == "second"].set_index(
        ["batter_id", "pitcher_id"]
    )["pa"]

    logger.info("\nSplit-half correlation (all pairs >= 20 total PA):")
    r_k = pivot_k["first"].corr(pivot_k["second"])
    r_bb = pivot_bb["first"].corr(pivot_bb["second"])
    logger.info("  K%%  r = %.4f  (n=%d)", r_k, len(pivot_k))
    logger.info("  BB%% r = %.4f  (n=%d)", r_bb, len(pivot_bb))

    results: dict[str, dict] = {"overall_k_r": r_k, "overall_bb_r": r_bb, "overall_n": len(pivot_k)}

    for min_half_pa in (10, 15, 20):
        pair_keys = pivot_pa_first[pivot_pa_first >= min_half_pa].index.intersection(
            pivot_pa_second[pivot_pa_second >= min_half_pa].index
        )
        sub_k = pivot_k.loc[pair_keys]
        sub_bb = pivot_bb.loc[pair_keys]
        if len(sub_k) < 5:
            logger.info("\n  Tier min_half_pa >= %d: too few pairs (n=%d)",
                        min_half_pa, len(sub_k))
            continue
        r_k_sub = sub_k["first"].corr(sub_k["second"])
        r_bb_sub = sub_bb["first"].corr(sub_bb["second"])
        logger.info("\n  Tier min_half_pa >= %d  (n=%d):", min_half_pa, len(sub_k))
        logger.info("    K%%  r = %.4f", r_k_sub)
        logger.info("    BB%% r = %.4f", r_bb_sub)
        results[f"tier_{min_half_pa}_k_r"] = r_k_sub
        results[f"tier_{min_half_pa}_bb_r"] = r_bb_sub
        results[f"tier_{min_half_pa}_n"] = len(sub_k)

    return results


def diagnostic_3_incremental(coverage_df: pd.DataFrame) -> dict:
    """Does BvP history beat league + pitcher_rate + hitter_rate?"""
    logger.info("\n" + "=" * 60)
    logger.info("Diagnostic 3: incremental predictive value")
    logger.info("=" * 60)

    # Build pre-2024 career K%/BB% per pair (with career PA count)
    q_career_rates = f"""
        SELECT
            pa.batter_id,
            pa.pitcher_id,
            COUNT(*) AS prior_pa,
            SUM(CASE WHEN pa.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END)::float / COUNT(*) AS bvp_k_rate,
            SUM(CASE WHEN pa.events = 'walk' THEN 1 ELSE 0 END)::float / COUNT(*) AS bvp_bb_rate
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.season < {HOLDOUT_SEASON}
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL AND pa.events <> '' AND pa.events <> 'intent_walk'
        GROUP BY pa.batter_id, pa.pitcher_id
    """
    logger.info("Loading pre-2024 career BvP rates...")
    career_rates = read_sql(q_career_rates)
    logger.info("  %d pairs with prior history", len(career_rates))

    # Build pre-2024 pitcher season rate and hitter season rate (most recent
    # pre-2024 season they appeared) — pooled 2022-2023 for reliability
    q_pitcher = f"""
        SELECT
            pa.pitcher_id,
            COUNT(*) AS p_bf,
            SUM(CASE WHEN pa.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END)::float / COUNT(*) AS p_k_rate,
            SUM(CASE WHEN pa.events = 'walk' THEN 1 ELSE 0 END)::float / COUNT(*) AS p_bb_rate
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.season IN (2022, 2023)
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL AND pa.events <> '' AND pa.events <> 'intent_walk'
        GROUP BY pa.pitcher_id
        HAVING COUNT(*) >= 50
    """
    q_batter = f"""
        SELECT
            pa.batter_id,
            COUNT(*) AS b_pa,
            SUM(CASE WHEN pa.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END)::float / COUNT(*) AS b_k_rate,
            SUM(CASE WHEN pa.events = 'walk' THEN 1 ELSE 0 END)::float / COUNT(*) AS b_bb_rate
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.season IN (2022, 2023)
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL AND pa.events <> '' AND pa.events <> 'intent_walk'
        GROUP BY pa.batter_id
        HAVING COUNT(*) >= 50
    """
    logger.info("Loading pre-2024 pitcher and batter rates...")
    pitcher_rates = read_sql(q_pitcher)
    batter_rates = read_sql(q_batter)
    logger.info("  %d pitchers, %d batters", len(pitcher_rates), len(batter_rates))

    q_2024_outcomes = f"""
        SELECT
            pa.pitcher_id,
            pa.batter_id,
            (pa.events IN ('strikeout','strikeout_double_play'))::int AS is_k,
            (pa.events = 'walk')::int AS is_bb
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.season = {HOLDOUT_SEASON}
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL AND pa.events <> '' AND pa.events <> 'intent_walk'
    """
    logger.info("Loading 2024 PA outcomes...")
    pa_2024 = read_sql(q_2024_outcomes)
    logger.info("  %d PAs", len(pa_2024))

    league_k = pa_2024["is_k"].mean()
    league_bb = pa_2024["is_bb"].mean()
    logger.info("  2024 league K %% = %.4f, BB %% = %.4f", league_k, league_bb)

    merged = pa_2024.merge(pitcher_rates, on="pitcher_id", how="left")
    merged = merged.merge(batter_rates, on="batter_id", how="left")
    merged = merged.merge(career_rates, on=["batter_id", "pitcher_id"], how="left")
    merged["prior_pa"] = merged["prior_pa"].fillna(0)
    merged = merged.dropna(subset=["p_k_rate", "b_k_rate"]).reset_index(drop=True)
    logger.info("  %d 2024 PAs with matchup rates available", len(merged))

    def model_pred(row_p_rate: np.ndarray, row_b_rate: np.ndarray, league: float) -> np.ndarray:
        logit_league = _logit(league)
        p_delta = _logit(row_p_rate) - logit_league
        h_delta = _logit(row_b_rate) - logit_league
        return _inv_logit(logit_league + p_delta + h_delta)

    results: dict = {}
    for stat, league, col_p, col_b, col_bvp, col_y in [
        ("k", league_k, "p_k_rate", "b_k_rate", "bvp_k_rate", "is_k"),
        ("bb", league_bb, "p_bb_rate", "b_bb_rate", "bvp_bb_rate", "is_bb"),
    ]:
        y = merged[col_y].values.astype(int)
        n = len(y)

        pred_flat = np.full(n, league)
        pred_pxh = model_pred(merged[col_p].values, merged[col_b].values, league)

        reliability = np.clip(merged["prior_pa"].fillna(0).values / RELIABILITY_PA, 0.0, 1.0)
        bvp_logit = _logit(merged[col_bvp].fillna(league).values)
        base_pxh_logit = _logit(pred_pxh)
        shrunk_bvp_logit = reliability * bvp_logit + (1.0 - reliability) * base_pxh_logit
        pred_with_bvp = _inv_logit(shrunk_bvp_logit)

        ll_flat = log_loss(y, pred_flat)
        ll_pxh = log_loss(y, pred_pxh)
        ll_bvp = log_loss(y, pred_with_bvp)

        logger.info("\n%s log-loss (N=%d):", stat.upper(), n)
        logger.info("  flat:        %.5f", ll_flat)
        logger.info("  pitcher*hitter: %.5f  (delta vs flat %+.5f)",
                    ll_pxh, ll_pxh - ll_flat)
        logger.info("  + BvP shrunk:   %.5f  (delta vs pxh %+.5f)",
                    ll_bvp, ll_bvp - ll_pxh)

        # Now restrict to pairs with meaningful BvP history
        for min_pa in (10, 20, 50):
            mask = merged["prior_pa"] >= min_pa
            if mask.sum() < 100:
                continue
            y_sub = y[mask]
            ll_pxh_sub = log_loss(y_sub, pred_pxh[mask])
            ll_bvp_sub = log_loss(y_sub, pred_with_bvp[mask])
            logger.info("  [pairs with >= %d prior PAs, n=%d]", min_pa, mask.sum())
            logger.info("    pitcher*hitter:  %.5f", ll_pxh_sub)
            logger.info("    + BvP shrunk:    %.5f  (delta %+.5f)",
                        ll_bvp_sub, ll_bvp_sub - ll_pxh_sub)
            results[f"{stat}_pxh_ge{min_pa}"] = ll_pxh_sub
            results[f"{stat}_bvp_ge{min_pa}"] = ll_bvp_sub
            results[f"{stat}_n_ge{min_pa}"] = int(mask.sum())

        results[f"{stat}_pxh_all"] = ll_pxh
        results[f"{stat}_bvp_all"] = ll_bvp
        results[f"{stat}_flat_all"] = ll_flat
        results[f"{stat}_n_all"] = n

    return results


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    logger.info("BvP history research\n")

    coverage_df = diagnostic_1_coverage()
    stability_results = diagnostic_2_split_half()
    incremental_results = diagnostic_3_incremental(coverage_df)

    logger.info("\n\nAll diagnostics complete.")
    all_results = {**stability_results, **incremental_results}
    pd.DataFrame([all_results]).to_parquet(OUT_JSON)
    logger.info("Saved to %s", OUT_JSON)


if __name__ == "__main__":
    main()
