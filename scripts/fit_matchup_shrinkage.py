"""Fit calibrated matchup shrinkage slopes for K and BB lifts.

Path A from the Layer 2 BvP diagnostic (2026-04-10). Replaces the blunt
scalar dampeners (K * 0.55, BB * 0.40) with a fitted logistic regression.

Approach:
  1. Walk-forward: for season t in [2020..2024], profiles from t-1 are used
     to score PAs in season t. Prevents look-ahead.
  2. For each unique (pitcher, batter) pair appearing in season t, compute
     the raw matchup_k_logit_lift and matchup_bb_logit_lift from
     score_matchup_for_stat.
  3. Join back to PA-level outcomes (is_strikeout, is_walk).
  4. Fit a per-stat logistic regression:
         logit(P(outcome)) = intercept + slope * raw_matchup_lift
     Training on pooled 2020-2023 PAs, holdout on 2024.
  5. Save coefficients to data/cached/matchup_shrinkage_coefs.parquet.

Starting minimum per user instruction: one intercept and one slope per stat.
If diagnostics show residual structure (e.g., slope varies with
reliability, or pitcher_delta and hitter_delta carry different weights), we
add complexity in a follow-up.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db import read_sql
from src.data.feature_eng import get_hitter_vulnerability, get_pitcher_arsenal
from src.data.league_baselines import get_baselines_dict
from src.models.matchup import score_matchup_for_stat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SEASONS = [2020, 2021, 2022, 2023]
HOLDOUT_SEASON = 2024
OUT_PATH = Path("data/cached/matchup_shrinkage_coefs.parquet")
MIN_PA_PAIR = 1  # include all pairs; regression handles reliability implicitly


def load_pa_outcomes(season: int) -> pd.DataFrame:
    """Load PA-level outcomes for a season.

    Returns columns: pitcher_id, batter_id, is_strikeout, is_walk.
    """
    logger.info("Loading PA outcomes for season %d...", season)
    query = f"""
        SELECT
            pa.pitcher_id,
            pa.batter_id,
            (pa.events IN ('strikeout', 'strikeout_double_play'))::int AS is_strikeout,
            (pa.events IN ('walk'))::int AS is_walk
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.season = {season}
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL
          AND pa.events <> ''
          AND pa.events <> 'intent_walk'
    """
    df = read_sql(query)
    df["pitcher_id"] = df["pitcher_id"].astype(int)
    df["batter_id"] = df["batter_id"].astype(int)
    logger.info("  %d PAs", len(df))
    return df


def score_unique_pairs(
    pair_df: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict,
) -> pd.DataFrame:
    """Score each unique (pitcher, batter) pair for K and BB lifts.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Unique pairs with columns pitcher_id, batter_id.
    pitcher_arsenal, hitter_vuln, baselines_pt
        Profiles from the training season (t-1).

    Returns
    -------
    pd.DataFrame
        pitcher_id, batter_id, k_lift, k_reliability, bb_lift, bb_reliability.
    """
    rows = []
    for i, row in enumerate(pair_df.itertuples(index=False)):
        if i % 10000 == 0 and i > 0:
            logger.info("  Scored %d / %d pairs", i, len(pair_df))
        pid = int(row.pitcher_id)
        bid = int(row.batter_id)
        k_res = score_matchup_for_stat(
            stat_name="k", pitcher_id=pid, batter_id=bid,
            pitcher_arsenal=pitcher_arsenal, hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
        )
        bb_res = score_matchup_for_stat(
            stat_name="bb", pitcher_id=pid, batter_id=bid,
            pitcher_arsenal=pitcher_arsenal, hitter_vuln=hitter_vuln,
            baselines_pt=baselines_pt,
        )
        k_lift = k_res.get("matchup_k_logit_lift", 0.0)
        bb_lift = bb_res.get("matchup_bb_logit_lift", 0.0)
        if isinstance(k_lift, float) and np.isnan(k_lift):
            k_lift = 0.0
        if isinstance(bb_lift, float) and np.isnan(bb_lift):
            bb_lift = 0.0
        rows.append({
            "pitcher_id": pid,
            "batter_id": bid,
            "k_lift": float(k_lift),
            "k_reliability": float(k_res.get("avg_reliability", 0.0)),
            "bb_lift": float(bb_lift),
            "bb_reliability": float(bb_res.get("avg_reliability", 0.0)),
        })
    return pd.DataFrame(rows)


def build_season_dataset(season: int) -> pd.DataFrame:
    """Build a season PA dataset with raw matchup lifts (profiles from t-1).

    Returns columns: is_strikeout, is_walk, k_lift, bb_lift, k_reliability,
    bb_reliability, season.
    """
    profile_season = season - 1
    logger.info("Season %d -> profiles from %d", season, profile_season)

    pitcher_arsenal = get_pitcher_arsenal(profile_season)
    hitter_vuln = get_hitter_vulnerability(profile_season)
    baselines_pt = get_baselines_dict(
        seasons=[profile_season], recency_weights="equal"
    )

    pa_df = load_pa_outcomes(season)

    pairs = pa_df[["pitcher_id", "batter_id"]].drop_duplicates().reset_index(drop=True)
    logger.info("  Scoring %d unique pairs", len(pairs))
    pair_lifts = score_unique_pairs(pairs, pitcher_arsenal, hitter_vuln, baselines_pt)

    merged = pa_df.merge(pair_lifts, on=["pitcher_id", "batter_id"], how="left")
    merged["k_lift"] = merged["k_lift"].fillna(0.0)
    merged["bb_lift"] = merged["bb_lift"].fillna(0.0)
    merged["k_reliability"] = merged["k_reliability"].fillna(0.0)
    merged["bb_reliability"] = merged["bb_reliability"].fillna(0.0)
    merged["season"] = season

    logger.info("  %d merged PA rows", len(merged))
    return merged


def fit_stat(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    stat: str,
) -> dict:
    """Fit logistic regression and report train + holdout diagnostics."""
    label_col = "is_strikeout" if stat == "k" else "is_walk"
    lift_col = f"{stat}_lift"

    X_train = df_train[[lift_col]].values
    y_train = df_train[label_col].values.astype(int)
    X_holdout = df_holdout[[lift_col]].values
    y_holdout = df_holdout[label_col].values.astype(int)

    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    baseline_rate = float(y_train.mean())

    p_train = model.predict_proba(X_train)[:, 1]
    p_holdout = model.predict_proba(X_holdout)[:, 1]

    p_train_raw = 1.0 / (1.0 + np.exp(-(np.log(baseline_rate / (1 - baseline_rate)) + X_train[:, 0])))
    p_holdout_raw = 1.0 / (1.0 + np.exp(-(np.log(baseline_rate / (1 - baseline_rate)) + X_holdout[:, 0])))

    ll_train_fit = log_loss(y_train, p_train)
    ll_holdout_fit = log_loss(y_holdout, p_holdout)
    ll_holdout_raw = log_loss(y_holdout, p_holdout_raw)
    ll_holdout_flat = log_loss(y_holdout, np.full(len(y_holdout), baseline_rate))

    brier_holdout_fit = brier_score_loss(y_holdout, p_holdout)
    brier_holdout_raw = brier_score_loss(y_holdout, p_holdout_raw)
    brier_holdout_flat = brier_score_loss(y_holdout, np.full(len(y_holdout), baseline_rate))

    return {
        "stat": stat,
        "slope": slope,
        "intercept": intercept,
        "baseline_rate": baseline_rate,
        "n_train": int(len(y_train)),
        "n_holdout": int(len(y_holdout)),
        "train_positive_rate": float(y_train.mean()),
        "holdout_positive_rate": float(y_holdout.mean()),
        "log_loss_train_fit": float(ll_train_fit),
        "log_loss_holdout_fit": float(ll_holdout_fit),
        "log_loss_holdout_raw": float(ll_holdout_raw),
        "log_loss_holdout_flat": float(ll_holdout_flat),
        "brier_holdout_fit": float(brier_holdout_fit),
        "brier_holdout_raw": float(brier_holdout_raw),
        "brier_holdout_flat": float(brier_holdout_flat),
        "implied_dampener_vs_raw": slope,
    }


def main() -> None:
    all_seasons = TRAIN_SEASONS + [HOLDOUT_SEASON]
    datasets = {s: build_season_dataset(s) for s in all_seasons}

    df_train = pd.concat([datasets[s] for s in TRAIN_SEASONS], ignore_index=True)
    df_holdout = datasets[HOLDOUT_SEASON]

    logger.info("Train pool: %d PAs across %s", len(df_train), TRAIN_SEASONS)
    logger.info("Holdout: %d PAs in %d", len(df_holdout), HOLDOUT_SEASON)

    k_result = fit_stat(df_train, df_holdout, "k")
    bb_result = fit_stat(df_train, df_holdout, "bb")

    for r in (k_result, bb_result):
        print("\n" + "=" * 72)
        print(f"Stat: {r['stat'].upper()}")
        print("=" * 72)
        print(f"  Fitted slope:            {r['slope']:.4f}")
        print(f"  Fitted intercept:        {r['intercept']:.4f}")
        print(f"  Baseline rate (train):   {r['baseline_rate']:.4f}")
        print(f"  Baseline rate (holdout): {r['holdout_positive_rate']:.4f}")
        print(f"  N train / N holdout:     {r['n_train']:,} / {r['n_holdout']:,}")
        print()
        print(f"  Holdout log-loss  raw-lift : {r['log_loss_holdout_raw']:.5f}")
        print(f"  Holdout log-loss  fit     : {r['log_loss_holdout_fit']:.5f}")
        print(f"  Holdout log-loss  flat-pop: {r['log_loss_holdout_flat']:.5f}")
        improvement_vs_raw = r['log_loss_holdout_raw'] - r['log_loss_holdout_fit']
        improvement_vs_flat = r['log_loss_holdout_flat'] - r['log_loss_holdout_fit']
        print(f"  Fit improvement vs raw : {improvement_vs_raw:+.5f}")
        print(f"  Fit improvement vs flat: {improvement_vs_flat:+.5f}")
        print()
        print(f"  Holdout Brier  raw-lift : {r['brier_holdout_raw']:.5f}")
        print(f"  Holdout Brier  fit     : {r['brier_holdout_fit']:.5f}")
        print(f"  Holdout Brier  flat-pop: {r['brier_holdout_flat']:.5f}")

    coefs_df = pd.DataFrame([k_result, bb_result])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coefs_df.to_parquet(OUT_PATH)
    print(f"\nSaved coefficients to {OUT_PATH}")


if __name__ == "__main__":
    main()
