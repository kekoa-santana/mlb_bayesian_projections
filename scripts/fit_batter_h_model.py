"""
Fit the BatterHModel: NegBin regression for per-batter hit counts.

Queries per-game batter stats from 2022-2025, builds anti-leakage
entering-game features, fits a NegBin regression, and validates on
2025 walk-forward holdout.

Usage:
    PYTHONPATH=. python scripts/fit_batter_h_model.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db import read_sql
from src.data.paths import dashboard_dir
from src.models.game_sim.batter_h_model import BatterHModel, ALL_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = dashboard_dir()
MODEL_OUT = DASHBOARD_DIR / "batter_h_model.pkl"

TRAIN_SEASONS = [2022, 2023, 2024]
VAL_SEASONS = [2025]
MIN_PA_ENTERING = 50  # minimum PA before a batter's entering-game stats are used


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_batter_game_data(seasons: list[int]) -> pd.DataFrame:
    """Load per-game batter stats with opposing starter info."""
    season_list = ",".join(str(s) for s in seasons)
    return read_sql(f"""
        WITH starters AS (
            SELECT fpg.player_id AS pitcher_id, fpg.game_pk,
                   fpg.team_id AS pitcher_team_id
            FROM production.fact_player_game_mlb fpg
            JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
            WHERE fpg.pit_is_starter = TRUE
              AND dg.game_type = 'R'
              AND fpg.season IN ({season_list})
        )
        SELECT
            fg.player_id AS batter_id,
            fg.game_pk,
            fg.game_date,
            fg.season,
            fg.team_id,
            fl.batting_order,
            fg.bat_pa,
            fg.bat_k,
            fg.bat_bb,
            fg.bat_h,
            fg.bat_hr,
            fg.bat_hbp,
            dg2.home_team_id,
            dg2.away_team_id,
            dg2.venue_id,
            st.pitcher_id AS opp_starter_id
        FROM production.fact_player_game_mlb fg
        JOIN production.dim_game dg2 ON fg.game_pk = dg2.game_pk
        JOIN production.fact_lineup fl
          ON fg.player_id = fl.player_id AND fg.game_pk = fl.game_pk
        LEFT JOIN starters st
          ON fg.game_pk = st.game_pk AND st.pitcher_team_id != fg.team_id
        WHERE fg.player_role = 'batter'
          AND dg2.game_type = 'R'
          AND fg.season IN ({season_list})
          AND fg.bat_pa >= 1
        ORDER BY fg.game_date, fg.game_pk, fl.batting_order
    """)


def load_pitcher_season_rates(seasons: list[int]) -> pd.DataFrame:
    """Load per-game pitcher stats for entering-game rates."""
    season_list = ",".join(str(s) for s in seasons)
    return read_sql(f"""
        SELECT
            fpg.player_id AS pitcher_id,
            fpg.game_pk,
            fpg.game_date,
            fpg.season,
            fpg.pit_bf,
            fpg.pit_k,
            fpg.pit_bb
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.player_role = 'pitcher'
          AND dg.game_type = 'R'
          AND fpg.season IN ({season_list})
          AND fpg.pit_bf >= 1
        ORDER BY fpg.game_date, fpg.game_pk
    """)


def load_sprint_speed() -> pd.DataFrame:
    """Load sprint speed data."""
    try:
        ss = pd.read_parquet(DASHBOARD_DIR / "batter_bip_profiles.parquet")
        return ss[["batter_id", "sprint_speed"]].drop_duplicates("batter_id")
    except FileNotFoundError:
        logger.warning("No sprint speed data found")
        return pd.DataFrame(columns=["batter_id", "sprint_speed"])


def load_park_factors() -> dict[int, float]:
    """Load park BABIP factors."""
    try:
        pf = pd.read_parquet(DASHBOARD_DIR / "park_factor_lifts.parquet")
        return {int(r["venue_id"]): float(r.get("h_babip_adj", 0.0))
                for _, r in pf.iterrows()}
    except FileNotFoundError:
        return {}


def load_pitcher_hands() -> dict[int, str]:
    """Load pitcher throwing hand for platoon computation."""
    df = read_sql("""
        SELECT DISTINCT player_id, pitch_hand
        FROM production.dim_player
        WHERE pitch_hand IS NOT NULL
    """)
    return {int(r["player_id"]): r["pitch_hand"] for _, r in df.iterrows()}


def load_batter_hands() -> dict[int, str]:
    """Load batter batting hand for platoon computation."""
    df = read_sql("""
        SELECT DISTINCT player_id, bat_side
        FROM production.dim_player
        WHERE bat_side IS NOT NULL
    """)
    return {int(r["player_id"]): r["bat_side"] for _, r in df.iterrows()}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_entering_game_batter_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative entering-game stats per batter-season.

    Anti-leakage: uses cumsum - current_game to exclude the game being predicted.
    """
    df = df.sort_values(["batter_id", "season", "game_date", "game_pk"]).copy()

    # Aggregate per (batter, season, game) in case of multi-pitcher splits
    agg = df.groupby(["batter_id", "season", "game_date", "game_pk"]).agg(
        bat_pa=("bat_pa", "sum"),
        bat_h=("bat_h", "sum"),
        bat_k=("bat_k", "sum"),
        bat_bb=("bat_bb", "sum"),
        bat_hr=("bat_hr", "sum"),
        bat_hbp=("bat_hbp", "sum"),
    ).reset_index()

    # Cumulative entering-game stats
    for col in ["bat_pa", "bat_h", "bat_k", "bat_bb", "bat_hr", "bat_hbp"]:
        agg[f"cum_{col}"] = (
            agg.groupby(["batter_id", "season"])[col].cumsum() - agg[col]
        )

    # Rates (entering-game)
    safe_pa = agg["cum_bat_pa"].clip(lower=1)
    agg["batter_h_per_pa"] = agg["cum_bat_h"] / safe_pa
    agg["batter_k_pct"] = agg["cum_bat_k"] / safe_pa
    agg["batter_bb_pct"] = agg["cum_bat_bb"] / safe_pa

    # BABIP = (H - HR) / (PA - K - BB - HR - HBP)  [approximation using PA]
    bip = (agg["cum_bat_pa"] - agg["cum_bat_k"] - agg["cum_bat_bb"]
           - agg["cum_bat_hr"] - agg["cum_bat_hbp"]).clip(lower=1)
    agg["batter_babip"] = (agg["cum_bat_h"] - agg["cum_bat_hr"]).clip(lower=0) / bip

    return agg


def compute_entering_game_pitcher_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute entering-game pitcher K% and BB%."""
    df = df.sort_values(["pitcher_id", "season", "game_date", "game_pk"]).copy()

    agg = df.groupby(["pitcher_id", "season", "game_date", "game_pk"]).agg(
        pit_bf=("pit_bf", "sum"),
        pit_k=("pit_k", "sum"),
        pit_bb=("pit_bb", "sum"),
    ).reset_index()

    for col in ["pit_bf", "pit_k", "pit_bb"]:
        agg[f"cum_{col}"] = (
            agg.groupby(["pitcher_id", "season"])[col].cumsum() - agg[col]
        )

    safe_bf = agg["cum_pit_bf"].clip(lower=1)
    agg["opp_starter_k_pct"] = agg["cum_pit_k"] / safe_bf
    agg["opp_starter_bb_pct"] = agg["cum_pit_bb"] / safe_bf

    return agg[["pitcher_id", "season", "game_pk",
                "opp_starter_k_pct", "opp_starter_bb_pct", "cum_pit_bf"]]


def build_training_data(
    batter_games: pd.DataFrame,
    pitcher_games: pd.DataFrame,
    sprint_speeds: pd.DataFrame,
    park_factors: dict[int, float],
    pitcher_hands: dict[int, str],
    batter_hands: dict[int, str],
) -> pd.DataFrame:
    """Build the full feature matrix for training."""
    # Entering-game batter stats
    batter_stats = compute_entering_game_batter_stats(batter_games)

    # Entering-game pitcher stats
    pitcher_stats = compute_entering_game_pitcher_stats(pitcher_games)

    # Take unique batter-game rows (collapse pitcher splits)
    game_info = (
        batter_games
        .sort_values("batting_order")
        .groupby(["batter_id", "game_pk"])
        .agg(
            game_date=("game_date", "first"),
            season=("season", "first"),
            team_id=("team_id", "first"),
            batting_order=("batting_order", "first"),
            bat_pa=("bat_pa", "sum"),
            bat_h=("bat_h", "sum"),
            home_team_id=("home_team_id", "first"),
            venue_id=("venue_id", "first"),
            opp_starter_id=("opp_starter_id", "first"),
        )
        .reset_index()
    )

    # Merge entering-game batter stats
    game_info = game_info.merge(
        batter_stats[["batter_id", "game_pk", "cum_bat_pa",
                       "batter_h_per_pa", "batter_k_pct", "batter_babip"]],
        on=["batter_id", "game_pk"],
        how="left",
    )

    # Filter: need enough PA for stable entering-game rates
    game_info = game_info[game_info["cum_bat_pa"] >= MIN_PA_ENTERING].copy()

    # Merge pitcher entering-game stats
    game_info = game_info.merge(
        pitcher_stats.rename(columns={"pitcher_id": "opp_starter_id"}),
        on=["opp_starter_id", "game_pk"],
        how="left",
        suffixes=("", "_pit"),
    )

    # Filter: need pitcher with enough BF
    game_info = game_info[game_info["cum_pit_bf"].fillna(0) >= 30].copy()

    # Sprint speed
    game_info = game_info.merge(
        sprint_speeds.rename(columns={"batter_id": "batter_id"}),
        on="batter_id",
        how="left",
    )
    game_info["sprint_speed"] = game_info["sprint_speed"].fillna(27.0)

    # Park BABIP factor
    game_info["park_h_babip"] = game_info["venue_id"].map(
        lambda v: park_factors.get(int(v), 0.0) if pd.notna(v) else 0.0
    )

    # Home indicator
    game_info["is_home"] = (
        game_info["team_id"] == game_info["home_team_id"]
    ).astype(float)

    # Platoon (same hand = 1)
    game_info["pitcher_hand"] = game_info["opp_starter_id"].map(pitcher_hands)
    game_info["batter_hand"] = game_info["batter_id"].map(batter_hands)
    game_info["platoon"] = (
        game_info["pitcher_hand"] == game_info["batter_hand"]
    ).astype(float)
    # Switch hitters: platoon = 0 (neutral)
    game_info.loc[game_info["batter_hand"] == "S", "platoon"] = 0.0
    # Missing handedness: default to neutral
    game_info["platoon"] = game_info["platoon"].fillna(0.0)

    # Target
    game_info["h"] = game_info["bat_h"].astype(float)

    # Drop rows with missing features
    game_info = game_info.dropna(subset=ALL_FEATURES + ["h"])

    logger.info(
        "Built training data: %d batter-game rows, %d unique batters",
        len(game_info), game_info["batter_id"].nunique(),
    )
    return game_info


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model: BatterHModel, df: pd.DataFrame, label: str) -> dict:
    """Evaluate model on a dataset."""
    mu = model.predict_mu(df)
    actual = df["h"].values

    bias = float(np.mean(mu - actual))
    mae = float(np.mean(np.abs(mu - actual)))
    rmse = float(np.sqrt(np.mean((mu - actual) ** 2)))
    corr = float(np.corrcoef(mu, actual)[0, 1])
    mu_std = float(np.std(mu))

    # P(2+ H) calibration by decile
    df_eval = df.copy()
    df_eval["mu"] = mu
    df_eval["actual_multi"] = (actual >= 2).astype(float)

    # NegBin P(H >= 2) = 1 - P(H=0) - P(H=1)
    alpha = model.alpha_
    n_param = 1.0 / alpha
    from scipy.stats import nbinom
    p_param = n_param / (n_param + mu)
    p_ge2 = 1.0 - nbinom.cdf(1, n_param, p_param)
    df_eval["p_multi"] = p_ge2

    # Decile calibration
    df_eval["decile"] = pd.qcut(df_eval["mu"], 10, labels=False, duplicates="drop")
    cal = df_eval.groupby("decile").agg(
        n=("h", "size"),
        pred_mu=("mu", "mean"),
        actual_h=("h", "mean"),
        pred_multi=("p_multi", "mean"),
        actual_multi=("actual_multi", "mean"),
    ).reset_index()

    print(f"\n{'='*70}")
    print(f"{label} ({len(df)} batter-games)")
    print(f"{'='*70}")
    print(f"  Bias: {bias:+.4f}  MAE: {mae:.4f}  RMSE: {rmse:.4f}  r: {corr:.4f}")
    print(f"  Predicted mu std: {mu_std:.4f} (target >= 0.25)")
    print(f"  Predicted mu range: {mu.min():.3f} - {mu.max():.3f}")
    print(f"  Alpha (dispersion): {alpha:.4f}")
    print(f"\n  Calibration by predicted-mu decile:")
    print(f"  {'Decile':>6} {'n':>6} {'pred_H':>7} {'act_H':>6} {'pred_2+':>7} {'act_2+':>7}")
    for _, r in cal.iterrows():
        print(f"  {int(r['decile']):>6} {int(r['n']):>6} {r['pred_mu']:>7.3f} "
              f"{r['actual_h']:>6.3f} {r['pred_multi']:>7.3f} {r['actual_multi']:>7.3f}")

    # Top vs bottom decile multi-hit ratio
    top = cal.iloc[-1]["actual_multi"]
    bot = cal.iloc[0]["actual_multi"]
    ratio = top / bot if bot > 0 else float("inf")
    print(f"\n  Multi-hit ratio (top/bottom decile): {ratio:.2f}x")
    print(f"  Top decile: pred_H={cal.iloc[-1]['pred_mu']:.3f}, actual 2+H rate={top:.3f}")
    print(f"  Bottom decile: pred_H={cal.iloc[0]['pred_mu']:.3f}, actual 2+H rate={bot:.3f}")

    return {
        "label": label,
        "n": len(df),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "mu_std": mu_std,
        "multi_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_seasons = TRAIN_SEASONS + VAL_SEASONS
    logger.info("Loading batter game data for seasons %s...", all_seasons)

    batter_games = load_batter_game_data(all_seasons)
    logger.info("Loaded %d batter-game rows", len(batter_games))

    pitcher_games = load_pitcher_season_rates(all_seasons)
    logger.info("Loaded %d pitcher-game rows", len(pitcher_games))

    sprint_speeds = load_sprint_speed()
    park_factors = load_park_factors()
    pitcher_hands = load_pitcher_hands()
    batter_hands = load_batter_hands()

    logger.info("Building features...")
    full_df = build_training_data(
        batter_games, pitcher_games, sprint_speeds,
        park_factors, pitcher_hands, batter_hands,
    )

    # Split train/val
    train = full_df[full_df["season"].isin(TRAIN_SEASONS)].copy()
    val = full_df[full_df["season"].isin(VAL_SEASONS)].copy()

    logger.info("Train: %d rows (%s), Val: %d rows (%s)",
                len(train), TRAIN_SEASONS, len(val), VAL_SEASONS)

    # Fit model
    model = BatterHModel()
    model.fit(train)

    # Print coefficients
    print("\nModel coefficients:")
    for name, coef in zip(["const"] + ALL_FEATURES, model.result_.params[:-1]):
        print(f"  {name:<25} {coef:+.6f}")
    print(f"  {'alpha':<25} {model.alpha_:.6f}")

    # Evaluate
    evaluate(model, train, "TRAIN")
    evaluate(model, val, "VALIDATION (2025)")

    # Save
    model.save(MODEL_OUT)
    logger.info("Model saved to %s", MODEL_OUT)

    # Compare to current sim differentiation
    print(f"\n{'='*70}")
    print("COMPARISON TO CURRENT SIM")
    print(f"{'='*70}")
    print(f"  Current sim expected H std: 0.150")
    print(f"  BatterHModel predicted mu std: {np.std(model.predict_mu(val)):.4f}")
    print(f"  Improvement: {np.std(model.predict_mu(val)) / 0.150:.1f}x")


if __name__ == "__main__":
    main()
