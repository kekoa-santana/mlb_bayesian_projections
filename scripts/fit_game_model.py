"""
Fit the paired Negative Binomial game-level run model.

Queries the database to build anti-leakage features (entering-game pitcher
rates, rolling team R/G, park run factors, weather), fits on 2022-2024,
validates on 2025, and saves to data/dashboard/game_level_model.pkl.

Usage:
    PYTHONPATH=. python scripts/fit_game_model.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.db import read_sql
from src.models.game_sim.game_level_model import (
    ALL_FEATURES,
    GameLevelModel,
    LEAGUE_AVG_BB_PCT,
    LEAGUE_AVG_HR_PCT,
    LEAGUE_AVG_K_PCT,
    LEAGUE_AVG_RPG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_OUT = PROJECT_ROOT / "data" / "dashboard" / "game_level_model.pkl"

TRAIN_SEASONS = [2022, 2023, 2024]
VAL_SEASONS = [2025]
ALL_SEASONS = TRAIN_SEASONS + VAL_SEASONS
MIN_STARTS_FOR_RATE = 3
ROLLING_WINDOW = 14  # games for team R/G


# ===================================================================== #
# Data loading
# ===================================================================== #

def load_game_runs() -> pd.DataFrame:
    """Load per-game run totals (home_runs, away_runs)."""
    season_list = ", ".join(str(s) for s in ALL_SEASONS)
    df = read_sql(f"""
        SELECT
            dg.game_pk,
            dg.game_date,
            dg.season,
            dg.home_team_id,
            dg.away_team_id,
            dg.venue_id,
            COALESCE(SUM(CASE WHEN fg.team_id = dg.home_team_id THEN fg.bat_r END), 0) AS home_runs,
            COALESCE(SUM(CASE WHEN fg.team_id = dg.away_team_id THEN fg.bat_r END), 0) AS away_runs
        FROM production.dim_game dg
        JOIN production.fact_player_game_mlb fg
            ON fg.game_pk = dg.game_pk AND fg.player_role = 'batter'
        WHERE dg.game_type = 'R'
          AND dg.season IN ({season_list})
        GROUP BY dg.game_pk, dg.game_date, dg.season,
                 dg.home_team_id, dg.away_team_id, dg.venue_id
        HAVING SUM(fg.bat_pa) > 30
        ORDER BY dg.game_date, dg.game_pk
    """)
    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info("Loaded %d games with run totals", len(df))
    return df


def load_starter_game_stats() -> pd.DataFrame:
    """Load per-game starter stats for computing entering-game rates."""
    season_list = ", ".join(str(s) for s in ALL_SEASONS)
    df = read_sql(f"""
        SELECT
            fg.player_id AS pitcher_id,
            fg.game_pk,
            fg.team_id,
            fg.season,
            dg.game_date,
            dg.home_team_id,
            dg.away_team_id,
            fg.pit_bf,
            fg.pit_k,
            fg.pit_bb,
            fg.pit_hr
        FROM production.fact_player_game_mlb fg
        JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
        WHERE fg.pit_is_starter = true
          AND fg.player_role = 'pitcher'
          AND dg.game_type = 'R'
          AND dg.season IN ({season_list})
          AND fg.pit_bf > 0
        ORDER BY dg.game_date, fg.player_id
    """)
    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info("Loaded %d starter game logs", len(df))
    return df


def load_team_game_runs() -> pd.DataFrame:
    """Load per-team per-game runs for rolling R/G."""
    season_list = ", ".join(str(s) for s in ALL_SEASONS)
    df = read_sql(f"""
        SELECT
            fg.team_id,
            dg.game_pk,
            dg.game_date,
            dg.season,
            SUM(fg.bat_r) AS team_runs
        FROM production.fact_player_game_mlb fg
        JOIN production.dim_game dg ON fg.game_pk = dg.game_pk
        WHERE fg.player_role = 'batter'
          AND dg.game_type = 'R'
          AND dg.season IN ({season_list})
        GROUP BY fg.team_id, dg.game_pk, dg.game_date, dg.season
        ORDER BY fg.team_id, dg.game_date
    """)
    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info("Loaded %d team-game run rows", len(df))
    return df


def load_weather() -> pd.DataFrame:
    """Load weather data."""
    df = read_sql("""
        SELECT game_pk, temperature, wind_category, is_dome
        FROM production.dim_weather
    """)
    logger.info("Loaded weather for %d games", len(df))
    return df


def load_park_run_factors() -> pd.DataFrame:
    """Compute venue run park factors from 2022-2024 data."""
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
    # Park factor = home_rate / away_rate, regressed toward 1.0
    shrinkage_games = 200
    reliability = (df["games"] / shrinkage_games).clip(0, 1)
    raw_pf = df["home_r_rate"] / df["away_r_rate"].replace(0, np.nan)
    df["park_run_factor"] = reliability * raw_pf + (1 - reliability) * 1.0
    logger.info("Park run factors for %d venues", len(df))
    return df[["venue_id", "park_run_factor"]].copy()


# ===================================================================== #
# Feature engineering
# ===================================================================== #

def compute_entering_game_pitcher_rates(
    starter_logs: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Compute each starter's season-to-date K%/BB%/HR% ENTERING each game.

    Anti-leakage: for each game, we only use starts strictly before that
    game_date within the same season.

    Returns one row per (game_pk) with columns:
        starter_k_pct_home, starter_bb_pct_home, starter_hr_pct_home,
        starter_k_pct_away, starter_bb_pct_away, starter_hr_pct_away
    """
    # Identify the starter for each game/side
    home_starters = starter_logs[
        starter_logs["team_id"] == starter_logs["home_team_id"]
    ][["game_pk", "pitcher_id", "game_date", "season"]].rename(
        columns={"pitcher_id": "home_starter_id"}
    )
    away_starters = starter_logs[
        starter_logs["team_id"] == starter_logs["away_team_id"]
    ][["game_pk", "pitcher_id", "game_date", "season"]].rename(
        columns={"pitcher_id": "away_starter_id"}
    )

    # For each pitcher, compute cumulative season stats BEFORE each game
    logs = starter_logs.sort_values(["pitcher_id", "season", "game_date"]).copy()
    logs["cum_bf"] = logs.groupby(["pitcher_id", "season"])["pit_bf"].cumsum() - logs["pit_bf"]
    logs["cum_k"] = logs.groupby(["pitcher_id", "season"])["pit_k"].cumsum() - logs["pit_k"]
    logs["cum_bb"] = logs.groupby(["pitcher_id", "season"])["pit_bb"].cumsum() - logs["pit_bb"]
    logs["cum_hr"] = logs.groupby(["pitcher_id", "season"])["pit_hr"].cumsum() - logs["pit_hr"]
    logs["cum_starts"] = logs.groupby(["pitcher_id", "season"]).cumcount()

    # Rates entering this game (use league avg if < MIN_STARTS)
    logs["k_pct"] = np.where(
        logs["cum_starts"] >= MIN_STARTS_FOR_RATE,
        logs["cum_k"] / logs["cum_bf"].clip(lower=1),
        LEAGUE_AVG_K_PCT,
    )
    logs["bb_pct"] = np.where(
        logs["cum_starts"] >= MIN_STARTS_FOR_RATE,
        logs["cum_bb"] / logs["cum_bf"].clip(lower=1),
        LEAGUE_AVG_BB_PCT,
    )
    logs["hr_pct"] = np.where(
        logs["cum_starts"] >= MIN_STARTS_FOR_RATE,
        logs["cum_hr"] / logs["cum_bf"].clip(lower=1),
        LEAGUE_AVG_HR_PCT,
    )

    pitcher_entering = logs[["game_pk", "pitcher_id", "k_pct", "bb_pct", "hr_pct", "team_id",
                             "home_team_id", "away_team_id"]].copy()

    # Split into home/away starters
    home_p = pitcher_entering[pitcher_entering["team_id"] == pitcher_entering["home_team_id"]].copy()
    home_p = home_p.rename(columns={
        "k_pct": "starter_k_pct_home",
        "bb_pct": "starter_bb_pct_home",
        "hr_pct": "starter_hr_pct_home",
    })[["game_pk", "starter_k_pct_home", "starter_bb_pct_home", "starter_hr_pct_home"]]

    away_p = pitcher_entering[pitcher_entering["team_id"] == pitcher_entering["away_team_id"]].copy()
    away_p = away_p.rename(columns={
        "k_pct": "starter_k_pct_away",
        "bb_pct": "starter_bb_pct_away",
        "hr_pct": "starter_hr_pct_away",
    })[["game_pk", "starter_k_pct_away", "starter_bb_pct_away", "starter_hr_pct_away"]]

    result = home_p.merge(away_p, on="game_pk", how="outer")
    logger.info("Pitcher entering-game rates: %d games", len(result))
    return result


def compute_team_rolling_rpg(team_runs: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling 14-game R/G entering each game, per team.

    Returns DataFrame with (game_pk, team_id, team_rpg).
    """
    team_runs = team_runs.sort_values(["team_id", "season", "game_date"]).copy()

    # Rolling mean of previous ROLLING_WINDOW games (excluding current)
    # within the same season
    results = []
    for (team_id, season), grp in team_runs.groupby(["team_id", "season"]):
        grp = grp.sort_values("game_date").copy()
        # shift(1) to exclude current game, then rolling
        shifted = grp["team_runs"].shift(1)
        rolling = shifted.rolling(ROLLING_WINDOW, min_periods=3).mean()
        grp["team_rpg"] = rolling.fillna(LEAGUE_AVG_RPG)
        results.append(grp[["game_pk", "team_id", "team_rpg"]])

    return pd.concat(results, ignore_index=True)


def build_feature_df(
    games: pd.DataFrame,
    pitcher_rates: pd.DataFrame,
    team_rpg: pd.DataFrame,
    weather: pd.DataFrame,
    park_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the full feature DataFrame (one row per game)."""
    df = games.copy()

    # Merge pitcher rates
    df = df.merge(pitcher_rates, on="game_pk", how="left")

    # Fill missing starters with league average
    for side in ["home", "away"]:
        df[f"starter_k_pct_{side}"] = df[f"starter_k_pct_{side}"].fillna(LEAGUE_AVG_K_PCT)
        df[f"starter_bb_pct_{side}"] = df[f"starter_bb_pct_{side}"].fillna(LEAGUE_AVG_BB_PCT)
        df[f"starter_hr_pct_{side}"] = df[f"starter_hr_pct_{side}"].fillna(LEAGUE_AVG_HR_PCT)

    # Merge team rolling R/G
    home_rpg = team_rpg.rename(columns={"team_rpg": "team_rpg_home", "team_id": "home_team_id"})
    away_rpg = team_rpg.rename(columns={"team_rpg": "team_rpg_away", "team_id": "away_team_id"})
    df = df.merge(
        home_rpg[["game_pk", "home_team_id", "team_rpg_home"]],
        on=["game_pk", "home_team_id"], how="left",
    )
    df = df.merge(
        away_rpg[["game_pk", "away_team_id", "team_rpg_away"]],
        on=["game_pk", "away_team_id"], how="left",
    )
    df["team_rpg_home"] = df["team_rpg_home"].fillna(LEAGUE_AVG_RPG)
    df["team_rpg_away"] = df["team_rpg_away"].fillna(LEAGUE_AVG_RPG)

    # Merge park factors
    df = df.merge(park_factors, on="venue_id", how="left")
    df["park_run_factor"] = df["park_run_factor"].fillna(1.0)

    # Merge weather
    df = df.merge(weather, on="game_pk", how="left")
    df["temperature"] = df["temperature"].fillna(72).clip(30, 120)
    df["wind_out"] = (df["wind_category"] == "out").astype(float)
    df["wind_in"] = (df["wind_category"] == "in").astype(float)
    df["is_dome"] = df["is_dome"].fillna(False).infer_objects(copy=False).astype(float)

    logger.info(
        "Feature matrix: %d games, %d missing home RPG, %d missing away RPG",
        len(df),
        df["team_rpg_home"].isna().sum(),
        df["team_rpg_away"].isna().sum(),
    )
    return df


def stack_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Stack game-level DataFrame into batting-team perspective rows.

    Each game becomes 2 rows:
    - home batting (is_home=1, opposing pitcher = away starter)
    - away batting (is_home=0, opposing pitcher = home starter)
    """
    model = GameLevelModel()

    home_rows = model._orient_batting(df, side="home")
    home_rows["runs"] = df["home_runs"].values
    home_rows["game_pk"] = df["game_pk"].values
    home_rows["game_date"] = df["game_date"].values
    home_rows["season"] = df["season"].values

    away_rows = model._orient_batting(df, side="away")
    away_rows["runs"] = df["away_runs"].values
    away_rows["game_pk"] = df["game_pk"].values
    away_rows["game_date"] = df["game_date"].values
    away_rows["season"] = df["season"].values

    stacked = pd.concat([home_rows, away_rows], ignore_index=True)
    logger.info("Stacked training data: %d rows from %d games", len(stacked), len(df))
    return stacked


# ===================================================================== #
# Evaluation
# ===================================================================== #

def evaluate(
    model: GameLevelModel,
    val_df: pd.DataFrame,
    label: str = "Validation",
) -> dict[str, float]:
    """Evaluate model on game-level (unstacked) data.

    Returns dict with RMSE, MAE, correlation for home/away/total.
    """
    preds = model.predict_game(val_df)
    metrics: dict[str, float] = {}

    for side, mu_col, actual_col in [
        ("home", "mu_home", "home_runs"),
        ("away", "mu_away", "away_runs"),
    ]:
        mu = preds[mu_col].values
        actual = val_df[actual_col].values
        resid = actual - mu
        rmse = np.sqrt(np.mean(resid ** 2))
        mae = np.mean(np.abs(resid))
        corr = np.corrcoef(mu, actual)[0, 1] if len(mu) > 1 else 0.0
        metrics[f"{side}_rmse"] = rmse
        metrics[f"{side}_mae"] = mae
        metrics[f"{side}_corr"] = corr

    # Total game runs
    total_pred = preds["mu_home"].values + preds["mu_away"].values
    total_actual = val_df["home_runs"].values + val_df["away_runs"].values
    total_resid = total_actual - total_pred
    metrics["total_rmse"] = np.sqrt(np.mean(total_resid ** 2))
    metrics["total_mae"] = np.mean(np.abs(total_resid))
    metrics["total_corr"] = np.corrcoef(total_pred, total_actual)[0, 1]
    metrics["total_pred_mean"] = float(total_pred.mean())
    metrics["total_actual_mean"] = float(total_actual.mean())

    print(f"\n{'='*60}")
    print(f"  {label} Results ({len(val_df)} games)")
    print(f"{'='*60}")
    for side in ["home", "away", "total"]:
        print(f"\n  {side.upper()}:")
        print(f"    RMSE:  {metrics[f'{side}_rmse']:.3f}")
        print(f"    MAE:   {metrics[f'{side}_mae']:.3f}")
        print(f"    Corr:  {metrics[f'{side}_corr']:.4f}")
    print(f"\n  Predicted total R/G: {metrics['total_pred_mean']:.2f}")
    print(f"  Actual total R/G:   {metrics['total_actual_mean']:.2f}")
    print(f"  Bias:               {metrics['total_pred_mean'] - metrics['total_actual_mean']:+.2f}")

    # Calibration by decile
    print(f"\n  Calibration (predicted total by decile):")
    dec = pd.DataFrame({"pred": total_pred, "actual": total_actual})
    dec["decile"] = pd.qcut(dec["pred"], 10, labels=False, duplicates="drop")
    cal = dec.groupby("decile").agg(
        pred_mean=("pred", "mean"),
        actual_mean=("actual", "mean"),
        n=("actual", "count"),
    )
    print(f"    {'Decile':>6}  {'Pred':>6}  {'Actual':>6}  {'N':>5}")
    for idx, row in cal.iterrows():
        print(f"    {idx:>6}  {row['pred_mean']:>6.2f}  {row['actual_mean']:>6.2f}  {int(row['n']):>5}")

    print(f"{'='*60}\n")
    return metrics


def print_coefficients(model: GameLevelModel) -> None:
    """Print model coefficients."""
    res = model.result_
    print("\nModel Coefficients:")
    print(f"{'Feature':<25} {'Coef':>8} {'Std Err':>8} {'z':>8} {'P>|z|':>8}")
    print("-" * 60)
    for name, coef, se, z, p in zip(
        res.params.index, res.params, res.bse, res.tvalues, res.pvalues
    ):
        print(f"{name:<25} {coef:>8.4f} {se:>8.4f} {z:>8.2f} {p:>8.4f}")


# ===================================================================== #
# Main
# ===================================================================== #

def main() -> None:
    logger.info("Loading data from database...")

    games = load_game_runs()
    starter_logs = load_starter_game_stats()
    team_runs = load_team_game_runs()
    weather = load_weather()
    park_factors = load_park_run_factors()

    logger.info("Computing features...")
    pitcher_rates = compute_entering_game_pitcher_rates(starter_logs, games)
    team_rpg = compute_team_rolling_rpg(team_runs)

    feature_df = build_feature_df(games, pitcher_rates, team_rpg, weather, park_factors)

    # Split train / validation
    train_df = feature_df[feature_df["season"].isin(TRAIN_SEASONS)].copy()
    val_df = feature_df[feature_df["season"].isin(VAL_SEASONS)].copy()
    logger.info("Train: %d games, Val: %d games", len(train_df), len(val_df))

    # Stack for training
    train_stacked = stack_for_training(train_df)

    # Fit
    logger.info("Fitting NegBin model...")
    model = GameLevelModel()
    model.fit(train_stacked)

    print_coefficients(model)

    # Evaluate
    evaluate(model, train_df, label="Training (2022-2024)")
    val_metrics = evaluate(model, val_df, label="Validation (2025)")

    # Save
    model.save(MODEL_OUT)
    logger.info("Model saved to %s", MODEL_OUT)

    # Summary
    print("\nDone. Key validation metrics:")
    print(f"  Total RMSE: {val_metrics['total_rmse']:.3f}")
    print(f"  Total Corr: {val_metrics['total_corr']:.4f}")
    print(f"  Home Corr:  {val_metrics['home_corr']:.4f}")
    print(f"  Away Corr:  {val_metrics['away_corr']:.4f}")


if __name__ == "__main__":
    main()
