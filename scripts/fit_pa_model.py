"""
Fit the LightGBM multiclass PA outcome model.

Queries PA-level data from 2022-2024 (train) and 2025 (validation),
builds anti-leakage features, fits a LightGBM multiclass classifier,
and saves the model to data/dashboard/pa_lgbm_model.joblib.

Usage:
    PYTHONPATH=. python scripts/fit_pa_model.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.db import read_sql
from src.data.paths import dashboard_dir
from src.models.ml.pa_lgbm_model import (
    CLASS_NAMES,
    NUM_CLASSES,
    PAOutcomeLGBM,
    map_events_to_classes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_OUT = dashboard_dir() / "pa_lgbm_model.joblib"

TRAIN_SEASONS = [2022, 2023, 2024]
VAL_SEASONS = [2025]
ALL_SEASONS = TRAIN_SEASONS + VAL_SEASONS
SEASON_LIST = ", ".join(str(s) for s in ALL_SEASONS)

# League average fallbacks (approximate 2022-2024)
LG_K_PCT = 0.224
LG_BB_PCT = 0.084
LG_HR_PCT = 0.031

# Minimum sample thresholds for using actual rates
MIN_PITCHER_BF = 50
MIN_BATTER_PA = 50


# ===================================================================== #
# Data loading
# ===================================================================== #


def load_pa_data() -> pd.DataFrame:
    """Load all PA-level data with game metadata."""
    df = read_sql(f"""
        SELECT
            fp.pa_id,
            fp.game_pk,
            fp.pitcher_id,
            fp.batter_id,
            fp.times_through_order,
            fp.outs_when_up,
            fp.inning,
            fp.events,
            fp.bat_score,
            fp.fld_score,
            fp.inning_topbot,
            dg.game_date,
            dg.season,
            dg.venue_id,
            dg.home_team_id,
            dg.away_team_id
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND dg.season IN ({SEASON_LIST})
          AND fp.events IS NOT NULL
          AND fp.events != 'truncated_pa'
        ORDER BY dg.game_date, fp.game_pk, fp.pa_id
    """)
    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info("Loaded %d plate appearances", len(df))
    return df


def load_pitcher_game_stats() -> pd.DataFrame:
    """Load pitcher game-level stats for computing entering-game rates."""
    df = read_sql(f"""
        SELECT
            fg.player_id AS pitcher_id,
            fg.game_pk,
            fg.season,
            dg.game_date,
            fg.pit_bf,
            fg.pit_k,
            fg.pit_bb,
            fg.pit_hr
        FROM production.fact_player_game_mlb fg
        JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
        WHERE fg.player_role = 'pitcher'
          AND dg.game_type = 'R'
          AND dg.season IN ({SEASON_LIST})
          AND fg.pit_bf > 0
        ORDER BY dg.game_date, fg.player_id
    """)
    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info("Loaded %d pitcher game logs", len(df))
    return df


def load_batter_game_stats() -> pd.DataFrame:
    """Load batter game-level stats for computing entering-game rates."""
    df = read_sql(f"""
        SELECT
            fg.player_id AS batter_id,
            fg.game_pk,
            fg.season,
            dg.game_date,
            fg.bat_pa,
            fg.bat_k,
            fg.bat_bb,
            fg.bat_hr
        FROM production.fact_player_game_mlb fg
        JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
        WHERE fg.player_role = 'batter'
          AND dg.game_type = 'R'
          AND dg.season IN ({SEASON_LIST})
          AND fg.bat_pa > 0
        ORDER BY dg.game_date, fg.player_id
    """)
    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info("Loaded %d batter game logs", len(df))
    return df


def load_handedness() -> pd.DataFrame:
    """Load pitcher/batter handedness from dim_player."""
    df = read_sql("""
        SELECT player_id, bat_side, pitch_hand
        FROM production.dim_player
    """)
    logger.info("Loaded handedness for %d players", len(df))
    return df


def load_weather() -> pd.DataFrame:
    """Load weather data."""
    df = read_sql("""
        SELECT game_pk, temperature, is_dome
        FROM production.dim_weather
    """)
    logger.info("Loaded weather for %d games", len(df))
    return df


def load_park_run_factors() -> pd.DataFrame:
    """Compute venue run park factors from 2022-2024."""
    df = read_sql("""
        WITH team_game AS (
            SELECT
                dg.venue_id,
                dg.game_pk,
                fg.team_id,
                CASE WHEN fg.team_id = dg.home_team_id THEN 'home' ELSE 'away' END AS loc,
                SUM(fg.bat_r) AS r,
                SUM(fg.bat_pa) AS pa
            FROM production.fact_player_game_mlb fg
            JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
            WHERE fg.player_role = 'batter'
              AND dg.game_type = 'R'
              AND dg.season BETWEEN 2022 AND 2024
            GROUP BY dg.venue_id, dg.game_pk, fg.team_id, dg.home_team_id
        )
        SELECT
            venue_id,
            COUNT(DISTINCT game_pk) AS games,
            SUM(CASE WHEN loc='home' THEN r END)::float
                / NULLIF(SUM(CASE WHEN loc='home' THEN pa END), 0) AS home_r_rate,
            SUM(CASE WHEN loc='away' THEN r END)::float
                / NULLIF(SUM(CASE WHEN loc='away' THEN pa END), 0) AS away_r_rate
        FROM team_game
        GROUP BY venue_id
        HAVING COUNT(DISTINCT game_pk) >= 50
    """)
    shrinkage_games = 200
    reliability = (df["games"] / shrinkage_games).clip(0, 1)
    raw_pf = df["home_r_rate"] / df["away_r_rate"].replace(0, np.nan)
    df["park_run_factor"] = reliability * raw_pf + (1 - reliability) * 1.0
    logger.info("Park run factors for %d venues", len(df))
    return df[["venue_id", "park_run_factor"]].copy()


# ===================================================================== #
# Feature engineering
# ===================================================================== #


def compute_entering_game_pitcher_rates(pitcher_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute each pitcher's season-to-date K%/BB%/HR% ENTERING each game.

    Anti-leakage: cumulative stats strictly before each game_date.
    Returns one row per (pitcher_id, game_pk) with entering-game rates.
    """
    # Aggregate to one row per pitcher per game (in case of multiple entries)
    agg = (
        pitcher_logs
        .groupby(["pitcher_id", "game_pk", "season", "game_date"], observed=True)
        .agg({"pit_bf": "sum", "pit_k": "sum", "pit_bb": "sum", "pit_hr": "sum"})
        .reset_index()
        .sort_values(["pitcher_id", "season", "game_date"])
    )

    # Cumulative sums BEFORE current game (subtract current game)
    for col in ["pit_bf", "pit_k", "pit_bb", "pit_hr"]:
        agg[f"cum_{col}"] = (
            agg.groupby(["pitcher_id", "season"])[col].cumsum() - agg[col]
        )

    # Apply rates with minimum threshold
    has_enough = agg["cum_pit_bf"] >= MIN_PITCHER_BF
    agg["pitcher_k_pct"] = np.where(
        has_enough, agg["cum_pit_k"] / agg["cum_pit_bf"].clip(lower=1), LG_K_PCT
    )
    agg["pitcher_bb_pct"] = np.where(
        has_enough, agg["cum_pit_bb"] / agg["cum_pit_bf"].clip(lower=1), LG_BB_PCT
    )
    agg["pitcher_hr_pct"] = np.where(
        has_enough, agg["cum_pit_hr"] / agg["cum_pit_bf"].clip(lower=1), LG_HR_PCT
    )

    logger.info("Computed entering-game pitcher rates: %d rows", len(agg))
    return agg[["pitcher_id", "game_pk", "pitcher_k_pct", "pitcher_bb_pct", "pitcher_hr_pct"]]


def compute_entering_game_batter_rates(batter_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute each batter's season-to-date K%/BB%/HR% ENTERING each game.

    Same anti-leakage approach as pitcher rates.
    """
    agg = (
        batter_logs
        .groupby(["batter_id", "game_pk", "season", "game_date"], observed=True)
        .agg({"bat_pa": "sum", "bat_k": "sum", "bat_bb": "sum", "bat_hr": "sum"})
        .reset_index()
        .sort_values(["batter_id", "season", "game_date"])
    )

    for col in ["bat_pa", "bat_k", "bat_bb", "bat_hr"]:
        agg[f"cum_{col}"] = (
            agg.groupby(["batter_id", "season"])[col].cumsum() - agg[col]
        )

    has_enough = agg["cum_bat_pa"] >= MIN_BATTER_PA
    agg["batter_k_pct"] = np.where(
        has_enough, agg["cum_bat_k"] / agg["cum_bat_pa"].clip(lower=1), LG_K_PCT
    )
    agg["batter_bb_pct"] = np.where(
        has_enough, agg["cum_bat_bb"] / agg["cum_bat_pa"].clip(lower=1), LG_BB_PCT
    )
    agg["batter_hr_pct"] = np.where(
        has_enough, agg["cum_bat_hr"] / agg["cum_bat_pa"].clip(lower=1), LG_HR_PCT
    )

    logger.info("Computed entering-game batter rates: %d rows", len(agg))
    return agg[["batter_id", "game_pk", "batter_k_pct", "batter_bb_pct", "batter_hr_pct"]]


def build_features(
    pa_df: pd.DataFrame,
    pitcher_rates: pd.DataFrame,
    batter_rates: pd.DataFrame,
    handedness: pd.DataFrame,
    weather: pd.DataFrame,
    park_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the full feature DataFrame (one row per PA)."""
    df = pa_df.copy()

    # -- Target variable --
    df["target"] = map_events_to_classes(df["events"])

    # -- Score differential from batter perspective --
    df["score_diff"] = df["bat_score"] - df["fld_score"]

    # -- Times through order: cap at 3 --
    df["times_through_order"] = df["times_through_order"].clip(upper=3)

    # -- Merge pitcher entering-game rates --
    df = df.merge(pitcher_rates, on=["pitcher_id", "game_pk"], how="left")
    df["pitcher_k_pct"] = df["pitcher_k_pct"].fillna(LG_K_PCT)
    df["pitcher_bb_pct"] = df["pitcher_bb_pct"].fillna(LG_BB_PCT)
    df["pitcher_hr_pct"] = df["pitcher_hr_pct"].fillna(LG_HR_PCT)

    # -- Merge batter entering-game rates --
    df = df.merge(batter_rates, on=["batter_id", "game_pk"], how="left")
    df["batter_k_pct"] = df["batter_k_pct"].fillna(LG_K_PCT)
    df["batter_bb_pct"] = df["batter_bb_pct"].fillna(LG_BB_PCT)
    df["batter_hr_pct"] = df["batter_hr_pct"].fillna(LG_HR_PCT)

    # -- Platoon advantage --
    pitcher_hand = handedness[["player_id", "pitch_hand"]].rename(
        columns={"player_id": "pitcher_id", "pitch_hand": "p_throws"}
    )
    batter_side = handedness[["player_id", "bat_side"]].rename(
        columns={"player_id": "batter_id"}
    )
    df = df.merge(pitcher_hand, on="pitcher_id", how="left")
    df = df.merge(batter_side, on="batter_id", how="left")

    # Same-hand matchup: RHP vs RHH or LHP vs LHH
    df["platoon"] = (
        ((df["p_throws"] == "R") & (df["bat_side"] == "R"))
        | ((df["p_throws"] == "L") & (df["bat_side"] == "L"))
    ).astype(float)
    # If handedness unknown, leave as NaN (LightGBM handles it)
    df.loc[df["p_throws"].isna() | df["bat_side"].isna(), "platoon"] = np.nan

    # -- Park run factor --
    df = df.merge(park_factors, on="venue_id", how="left")
    df["park_run_factor"] = df["park_run_factor"].fillna(1.0)

    # -- Weather --
    df = df.merge(weather, on="game_pk", how="left")
    df["temperature"] = df["temperature"].fillna(72).clip(30, 120)
    df["is_dome"] = df["is_dome"].fillna(False).infer_objects(copy=False).astype(float)

    logger.info(
        "Feature matrix built: %d PAs, %d features",
        len(df),
        len(PAOutcomeLGBM.FEATURES),
    )
    return df


# ===================================================================== #
# Evaluation
# ===================================================================== #


def compute_logloss(probs: np.ndarray, y: np.ndarray) -> float:
    """Compute multi-class log-loss."""
    eps = 1e-15
    n = len(y)
    clipped = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.log(clipped[np.arange(n), y]))


def evaluate(
    model: PAOutcomeLGBM,
    X: pd.DataFrame,
    y: np.ndarray,
    label: str = "Validation",
) -> float:
    """Evaluate model and print diagnostics. Returns log-loss."""
    probs = model.predict_pa_probs(X)
    ll = compute_logloss(probs, y)

    # Naive baseline: predict league-average class rates
    class_rates = np.bincount(y, minlength=NUM_CLASSES) / len(y)
    naive_probs = np.tile(class_rates, (len(y), 1))
    naive_ll = compute_logloss(naive_probs, y)

    print(f"\n{'='*60}")
    print(f"  {label} Results ({len(y):,} PAs)")
    print(f"{'='*60}")
    print(f"  Log-loss:       {ll:.5f}")
    print(f"  Naive log-loss: {naive_ll:.5f}")
    print(f"  Improvement:    {naive_ll - ll:.5f} ({(naive_ll - ll) / naive_ll * 100:.2f}%)")

    # Per-class calibration
    print(f"\n  Per-class calibration:")
    print(f"    {'Class':<6} {'Predicted':>10} {'Actual':>10} {'Count':>8} {'Ratio':>8}")
    print(f"    {'-'*42}")
    for c in range(NUM_CLASSES):
        pred_rate = probs[:, c].mean()
        actual_rate = (y == c).mean()
        count = (y == c).sum()
        ratio = pred_rate / actual_rate if actual_rate > 0 else 0
        print(f"    {CLASS_NAMES[c]:<6} {pred_rate:>10.4f} {actual_rate:>10.4f} {count:>8,} {ratio:>8.3f}")

    # Feature importance
    if model.model_ is not None:
        importances = model.model_.feature_importances_
        feat_imp = sorted(
            zip(PAOutcomeLGBM.FEATURES, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"\n  Feature importance (gain):")
        for feat, imp in feat_imp:
            bar = "#" * int(imp / max(importances) * 30)
            print(f"    {feat:<22} {imp:>6}  {bar}")

    print(f"{'='*60}\n")
    return ll


def sanity_check(model: PAOutcomeLGBM) -> None:
    """Print predicted probs for high-K vs low-K pitcher scenarios."""
    base = {
        "pitcher_k_pct": LG_K_PCT,
        "pitcher_bb_pct": LG_BB_PCT,
        "pitcher_hr_pct": LG_HR_PCT,
        "batter_k_pct": LG_K_PCT,
        "batter_bb_pct": LG_BB_PCT,
        "batter_hr_pct": LG_HR_PCT,
        "platoon": 0.0,
        "times_through_order": 1,
        "outs_when_up": 0,
        "inning": 1,
        "score_diff": 0,
        "park_run_factor": 1.0,
        "temperature": 72.0,
        "is_dome": 0.0,
    }

    scenarios = {
        "High-K pitcher (32%)": {"pitcher_k_pct": 0.32},
        "Low-K pitcher (16%)": {"pitcher_k_pct": 0.16},
        "High-K batter (30%)": {"batter_k_pct": 0.30},
        "Low-K batter (15%)": {"batter_k_pct": 0.15},
        "Platoon advantage (same-hand)": {"platoon": 1.0},
        "No platoon (opposite-hand)": {"platoon": 0.0},
        "3rd time through order": {"times_through_order": 3},
    }

    print("\nSanity check scenarios:")
    print(f"  {'Scenario':<35} {'K':>6} {'BB':>6} {'HBP':>6} {'HR':>6} {'BIP':>6}")
    print(f"  {'-'*71}")

    for name, overrides in scenarios.items():
        row = {**base, **overrides}
        X = pd.DataFrame([row])
        probs = model.predict_pa_probs(X)[0]
        vals = "  ".join(f"{p:>5.3f}" for p in probs)
        print(f"  {name:<35} {vals}")


# ===================================================================== #
# Main
# ===================================================================== #


def main() -> None:
    t0 = time.time()
    logger.info("Loading data from database...")

    pa_df = load_pa_data()
    pitcher_logs = load_pitcher_game_stats()
    batter_logs = load_batter_game_stats()
    handedness = load_handedness()
    weather = load_weather()
    park_factors = load_park_run_factors()

    logger.info("Computing entering-game rates...")
    pitcher_rates = compute_entering_game_pitcher_rates(pitcher_logs)
    batter_rates = compute_entering_game_batter_rates(batter_logs)

    logger.info("Building features...")
    feature_df = build_features(
        pa_df, pitcher_rates, batter_rates, handedness, weather, park_factors
    )

    # Split train / validation
    train_mask = feature_df["season"].isin(TRAIN_SEASONS)
    val_mask = feature_df["season"].isin(VAL_SEASONS)

    X_train = feature_df.loc[train_mask, PAOutcomeLGBM.FEATURES].copy()
    y_train = feature_df.loc[train_mask, "target"].values
    X_val = feature_df.loc[val_mask, PAOutcomeLGBM.FEATURES].copy()
    y_val = feature_df.loc[val_mask, "target"].values

    logger.info(
        "Train: %d PAs (%d-%d), Val: %d PAs (%s)",
        len(X_train),
        TRAIN_SEASONS[0],
        TRAIN_SEASONS[-1],
        len(X_val),
        VAL_SEASONS,
    )

    # Class distribution
    print("\nClass distribution:")
    for split_name, y in [("Train", y_train), ("Val", y_val)]:
        counts = np.bincount(y, minlength=NUM_CLASSES)
        pcts = counts / len(y) * 100
        print(f"  {split_name}:")
        for c in range(NUM_CLASSES):
            print(f"    {CLASS_NAMES[c]:<4}: {counts[c]:>8,} ({pcts[c]:.1f}%)")

    # Fit
    logger.info("Fitting LightGBM model...")
    model = PAOutcomeLGBM()
    model.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    evaluate(model, X_train, y_train, label="Training (2022-2024)")
    val_ll = evaluate(model, X_val, y_val, label="Validation (2025)")

    # Sanity checks
    sanity_check(model)

    # Save
    model.save(MODEL_OUT)
    logger.info("Model saved to %s", MODEL_OUT)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Validation log-loss: {val_ll:.5f}")
    print(f"  Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
