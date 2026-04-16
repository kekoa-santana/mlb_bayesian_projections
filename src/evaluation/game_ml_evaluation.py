"""
Hybrid ML Evaluation: Direct game-level LightGBM from database features.

Trains on 2008-2022 (~36K games), tests OOF on 2023-2025 + 2026.
Uses prior-season pitcher stats, lineup quality, park factors,
umpire tendencies, weather, rest, defense -- features the sim already
bakes in but that the ML can weight differently.

Usage:
    python -m src.evaluation.game_ml_evaluation
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.constants import LEAGUE_AVG_OVERALL
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DASH_DIR = Path(__file__).resolve().parents[2] / ".." / "tdd-dashboard" / "data" / "dashboard"

LGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    min_child_samples=50, reg_lambda=3.0, reg_alpha=1.0,
    subsample=0.8, colsample_bytree=0.7,
    random_state=42, verbose=-1, n_jobs=-1,
)


# ---------------------------------------------------------------------------
# 1. Build game-level dataset from database
# ---------------------------------------------------------------------------

def build_game_dataset(seasons: list[int] | None = None) -> pd.DataFrame:
    """Build comprehensive game-level dataset from the database.

    For each regular-season game, extracts:
    - Team scores (home/away runs)
    - Starting pitcher prior-season stats (K%, BB%, HR/BF, IP/start)
    - Starting pitcher YTD stats (rolling within season)
    - Team batting quality (prior season OPS proxy)
    - Park HR factor
    - Umpire K/BB logit lift
    - Days rest for starters
    - Home/away indicator
    """
    from src.data.db import read_sql

    if seasons is None:
        seasons = list(range(2008, 2027))

    season_list = ",".join(str(int(s)) for s in seasons)

    log.info("Building game dataset for seasons %s...", f"{min(seasons)}-{max(seasons)}")

    # -----------------------------------------------------------------------
    # A) Game scores + starters + venue
    # -----------------------------------------------------------------------
    log.info("  Querying game scores and starters...")
    games_df = read_sql(f"""
    WITH team_scores AS (
        SELECT bb.game_pk, bb.team_id,
               SUM(bb.runs) AS runs,
               SUM(bb.hits) AS hits,
               SUM(bb.home_runs) AS hr,
               SUM(bb.walks) AS bb,
               SUM(bb.strikeouts) AS k,
               SUM(bb.plate_appearances) AS pa
        FROM staging.batting_boxscores bb
        JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
        WHERE dg.game_type = 'R' AND dg.season IN ({season_list})
        GROUP BY bb.game_pk, bb.team_id
    ),
    starters AS (
        SELECT pb.game_pk, pb.pitcher_id, pb.team_id AS pitcher_team_id,
               pb.strike_outs AS pit_k, pb.walks AS pit_bb,
               pb.hits AS pit_h, pb.home_runs AS pit_hr,
               pb.batters_faced AS pit_bf, pb.innings_pitched AS pit_ip,
               pb.number_of_pitches AS pit_pitches, pb.earned_runs AS pit_er
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE pb.is_starter = true
          AND dg.game_type = 'R' AND dg.season IN ({season_list})
    )
    SELECT
        dg.game_pk, dg.season, dg.game_date,
        dg.home_team_id, dg.away_team_id,
        dg.venue_id, dg.day_night,
        -- Home team scores
        h_ts.runs AS home_runs, h_ts.hits AS home_hits,
        h_ts.hr AS home_hr, h_ts.bb AS home_bb,
        h_ts.k AS home_k, h_ts.pa AS home_pa,
        -- Away team scores
        a_ts.runs AS away_runs, a_ts.hits AS away_hits,
        a_ts.hr AS away_hr, a_ts.bb AS away_bb,
        a_ts.k AS away_k, a_ts.pa AS away_pa,
        -- Home starter
        hs.pitcher_id AS home_starter_id,
        hs.pit_k AS home_s_k, hs.pit_bb AS home_s_bb,
        hs.pit_h AS home_s_h, hs.pit_hr AS home_s_hr,
        hs.pit_bf AS home_s_bf, hs.pit_ip AS home_s_ip,
        hs.pit_pitches AS home_s_pitches, hs.pit_er AS home_s_er,
        -- Away starter
        aws.pitcher_id AS away_starter_id,
        aws.pit_k AS away_s_k, aws.pit_bb AS away_s_bb,
        aws.pit_h AS away_s_h, aws.pit_hr AS away_s_hr,
        aws.pit_bf AS away_s_bf, aws.pit_ip AS away_s_ip,
        aws.pit_pitches AS away_s_pitches, aws.pit_er AS away_s_er
    FROM production.dim_game dg
    LEFT JOIN team_scores h_ts
        ON dg.game_pk = h_ts.game_pk AND dg.home_team_id = h_ts.team_id
    LEFT JOIN team_scores a_ts
        ON dg.game_pk = a_ts.game_pk AND dg.away_team_id = a_ts.team_id
    LEFT JOIN starters hs
        ON dg.game_pk = hs.game_pk AND dg.home_team_id = hs.pitcher_team_id
    LEFT JOIN starters aws
        ON dg.game_pk = aws.game_pk AND dg.away_team_id = aws.pitcher_team_id
    WHERE dg.game_type = 'R' AND dg.season IN ({season_list})
    ORDER BY dg.game_date, dg.game_pk
    """)
    log.info(f"    {len(games_df)} games loaded")

    # Drop games missing scores or starters
    games_df = games_df.dropna(subset=["home_runs", "away_runs",
                                        "home_starter_id", "away_starter_id"])
    games_df["actual_total"] = games_df["home_runs"] + games_df["away_runs"]
    games_df["actual_margin"] = games_df["home_runs"] - games_df["away_runs"]
    games_df["home_won"] = (games_df["home_runs"] > games_df["away_runs"]).astype(int)
    log.info(f"    {len(games_df)} games after dropping missing")

    # -----------------------------------------------------------------------
    # B) Pitcher prior-season stats (features for prediction)
    # -----------------------------------------------------------------------
    log.info("  Computing pitcher prior-season stats...")
    pit_seasons = read_sql(f"""
    SELECT
        pb.pitcher_id,
        dg.season,
        SUM(pb.strike_outs) AS total_k,
        SUM(pb.walks) AS total_bb,
        SUM(pb.home_runs) AS total_hr,
        SUM(pb.batters_faced) AS total_bf,
        SUM(pb.innings_pitched) AS total_ip,
        SUM(pb.earned_runs) AS total_er,
        COUNT(*) AS games_started,
        AVG(pb.innings_pitched) AS avg_ip,
        AVG(pb.number_of_pitches) AS avg_pitches
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    WHERE pb.is_starter = true AND pb.batters_faced >= 9
      AND dg.game_type = 'R' AND dg.season IN ({season_list})
    GROUP BY pb.pitcher_id, dg.season
    """)

    # Compute rates
    pit_seasons["k_rate"] = pit_seasons["total_k"] / pit_seasons["total_bf"].clip(lower=1)
    pit_seasons["bb_rate"] = pit_seasons["total_bb"] / pit_seasons["total_bf"].clip(lower=1)
    pit_seasons["hr_rate"] = pit_seasons["total_hr"] / pit_seasons["total_bf"].clip(lower=1)
    pit_seasons["era"] = pit_seasons["total_er"] / (pit_seasons["total_ip"].clip(lower=1) / 9)
    pit_seasons["whip"] = (pit_seasons["total_bb"] + pit_seasons.get("total_h", 0)) / pit_seasons["total_ip"].clip(lower=1)

    # Shift to prior season: season N stats become features for season N+1
    pit_prior = pit_seasons.copy()
    pit_prior["feature_season"] = pit_prior["season"] + 1
    pit_prior = pit_prior.rename(columns={
        "k_rate": "prior_k_rate", "bb_rate": "prior_bb_rate",
        "hr_rate": "prior_hr_rate", "era": "prior_era",
        "avg_ip": "prior_avg_ip", "avg_pitches": "prior_avg_pitches",
        "games_started": "prior_gs",
    })

    # Join to games
    for side, starter_col in [("home", "home_starter_id"), ("away", "away_starter_id")]:
        merge_cols = ["pitcher_id", "feature_season"]
        feat_cols = ["prior_k_rate", "prior_bb_rate", "prior_hr_rate",
                     "prior_era", "prior_avg_ip", "prior_avg_pitches", "prior_gs"]
        pit_sub = pit_prior[["pitcher_id", "feature_season"] + feat_cols].copy()
        pit_sub = pit_sub.rename(columns={"pitcher_id": starter_col, "feature_season": "season"})
        pit_sub = pit_sub.rename(columns={c: f"{side}_{c}" for c in feat_cols})
        games_df = games_df.merge(pit_sub, on=[starter_col, "season"], how="left")

    # -----------------------------------------------------------------------
    # C) Team batting quality (prior season)
    # -----------------------------------------------------------------------
    log.info("  Computing team batting quality...")
    team_bat = read_sql(f"""
    SELECT
        bb.team_id,
        dg.season,
        SUM(bb.runs) AS team_runs,
        SUM(bb.hits) AS team_hits,
        SUM(bb.home_runs) AS team_hr,
        SUM(bb.walks) AS team_bb,
        SUM(bb.strikeouts) AS team_k,
        SUM(bb.plate_appearances) AS team_pa
    FROM staging.batting_boxscores bb
    JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
    WHERE dg.game_type = 'R' AND dg.season IN ({season_list})
    GROUP BY bb.team_id, dg.season
    """)
    team_bat["team_k_rate"] = team_bat["team_k"] / team_bat["team_pa"].clip(lower=1)
    team_bat["team_bb_rate"] = team_bat["team_bb"] / team_bat["team_pa"].clip(lower=1)
    team_bat["team_hr_rate"] = team_bat["team_hr"] / team_bat["team_pa"].clip(lower=1)
    team_bat["team_rpg"] = team_bat["team_runs"] / (team_bat["team_pa"].clip(lower=1) / 38.0)  # ~38 PA/game

    team_prior = team_bat.copy()
    team_prior["feature_season"] = team_prior["season"] + 1

    for side, team_col in [("home", "home_team_id"), ("away", "away_team_id")]:
        feat_cols = ["team_k_rate", "team_bb_rate", "team_hr_rate", "team_rpg"]
        t_sub = team_prior[["team_id", "feature_season"] + feat_cols].copy()
        t_sub = t_sub.rename(columns={"team_id": team_col, "feature_season": "season"})
        t_sub = t_sub.rename(columns={c: f"{side}_{c}" for c in feat_cols})
        games_df = games_df.merge(t_sub, on=[team_col, "season"], how="left")

    # -----------------------------------------------------------------------
    # D) Park factors
    # -----------------------------------------------------------------------
    log.info("  Loading park factors...")
    pf = read_sql(f"""
    SELECT venue_id, season,
           AVG(COALESCE(hr_pf_3yr, hr_pf_season, 1.0)) AS park_hr_pf
    FROM production.dim_park_factor
    WHERE season IN ({season_list})
    GROUP BY venue_id, season
    """)
    games_df = games_df.merge(pf, on=["venue_id", "season"], how="left")
    games_df["park_hr_pf"] = games_df["park_hr_pf"].fillna(1.0)

    # -----------------------------------------------------------------------
    # E) Umpire tendencies
    # -----------------------------------------------------------------------
    log.info("  Loading umpire assignments...")
    ump = read_sql(f"""
    SELECT du.game_pk, du.hp_umpire_name
    FROM production.dim_umpire du
    JOIN production.dim_game dg ON du.game_pk = dg.game_pk
    WHERE dg.game_type = 'R' AND dg.season IN ({season_list})
    """)

    # Build umpire career stats up to each season (leave-one-out by season)
    ump_career = read_sql(f"""
    SELECT du.hp_umpire_name, dg.season,
           COUNT(DISTINCT du.game_pk) AS ump_games,
           COUNT(fpa.pa_id) AS ump_pa,
           SUM(CASE WHEN fpa.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS ump_k,
           SUM(CASE WHEN fpa.events = 'walk' THEN 1 ELSE 0 END) AS ump_bb
    FROM production.dim_umpire du
    JOIN production.dim_game dg ON du.game_pk = dg.game_pk
    JOIN production.fact_pa fpa ON du.game_pk = fpa.game_pk
    WHERE dg.game_type = 'R' AND dg.season IN ({season_list})
      AND fpa.events IS NOT NULL
    GROUP BY du.hp_umpire_name, dg.season
    """)

    # Build cumulative stats (prior seasons only) for each umpire-season
    ump_career = ump_career.sort_values(["hp_umpire_name", "season"])
    ump_cum = []
    for ump_name, grp in ump_career.groupby("hp_umpire_name"):
        grp = grp.sort_values("season")
        cum_games = grp["ump_games"].cumsum().shift(1).fillna(0)
        cum_pa = grp["ump_pa"].cumsum().shift(1).fillna(0)
        cum_k = grp["ump_k"].cumsum().shift(1).fillna(0)
        cum_bb = grp["ump_bb"].cumsum().shift(1).fillna(0)
        for i, (_, row) in enumerate(grp.iterrows()):
            if cum_pa.iloc[i] > 0:
                ump_cum.append({
                    "hp_umpire_name": ump_name,
                    "season": row["season"],
                    "ump_prior_k_rate": cum_k.iloc[i] / cum_pa.iloc[i],
                    "ump_prior_bb_rate": cum_bb.iloc[i] / cum_pa.iloc[i],
                    "ump_prior_games": cum_games.iloc[i],
                })
    ump_feat = pd.DataFrame(ump_cum)

    # Merge umpire to games
    games_df = games_df.merge(ump, on="game_pk", how="left")
    if len(ump_feat) > 0:
        games_df = games_df.merge(ump_feat, on=["hp_umpire_name", "season"], how="left")
    else:
        games_df["ump_prior_k_rate"] = np.nan
        games_df["ump_prior_bb_rate"] = np.nan
        games_df["ump_prior_games"] = np.nan

    # -----------------------------------------------------------------------
    # F) Days rest
    # -----------------------------------------------------------------------
    log.info("  Loading days rest...")
    rest = read_sql(f"""
    WITH starter_dates AS (
        SELECT fpg.player_id AS pitcher_id, fpg.game_pk, fpg.game_date,
               LAG(fpg.game_date) OVER (PARTITION BY fpg.player_id ORDER BY fpg.game_date) AS prev_date
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true AND fpg.pit_bf >= 9
          AND dg.game_type = 'R' AND dg.season IN ({season_list})
    )
    SELECT pitcher_id, game_pk, (game_date - prev_date) AS days_rest
    FROM starter_dates
    WHERE prev_date IS NOT NULL
    """)

    for side, starter_col in [("home", "home_starter_id"), ("away", "away_starter_id")]:
        r_sub = rest.rename(columns={"pitcher_id": starter_col, "days_rest": f"{side}_rest"})
        games_df = games_df.merge(r_sub[[starter_col, "game_pk", f"{side}_rest"]],
                                   on=[starter_col, "game_pk"], how="left")

    games_df["home_rest"] = games_df["home_rest"].fillna(5.0).clip(3, 14)
    games_df["away_rest"] = games_df["away_rest"].fillna(5.0).clip(3, 14)

    # -----------------------------------------------------------------------
    # G) Derived features
    # -----------------------------------------------------------------------
    log.info("  Engineering features...")

    # Pitcher quality differential
    games_df["starter_k_diff"] = (
        games_df["home_prior_k_rate"].fillna(0.20) - games_df["away_prior_k_rate"].fillna(0.20)
    )
    games_df["starter_bb_diff"] = (
        games_df["home_prior_bb_rate"].fillna(0.08) - games_df["away_prior_bb_rate"].fillna(0.08)
    )
    games_df["starter_era_diff"] = (
        games_df["home_prior_era"].fillna(4.5) - games_df["away_prior_era"].fillna(4.5)
    )
    # Lineup quality differential
    games_df["lineup_rpg_diff"] = (
        games_df["home_team_rpg"].fillna(4.5) - games_df["away_team_rpg"].fillna(4.5)
    )
    _k_fill = LEAGUE_AVG_OVERALL["k_rate"]
    games_df["lineup_k_diff"] = (
        games_df["home_team_k_rate"].fillna(_k_fill) - games_df["away_team_k_rate"].fillna(_k_fill)
    )

    # Night game flag
    games_df["is_night"] = (games_df["day_night"] == "N").astype(float)

    log.info(f"  Dataset: {len(games_df)} games, {games_df['season'].nunique()} seasons")
    return games_df


# ---------------------------------------------------------------------------
# 2. Define features
# ---------------------------------------------------------------------------

FEATURES = [
    # Home starter prior-season stats
    "home_prior_k_rate", "home_prior_bb_rate", "home_prior_hr_rate",
    "home_prior_era", "home_prior_avg_ip", "home_prior_gs",
    # Away starter prior-season stats
    "away_prior_k_rate", "away_prior_bb_rate", "away_prior_hr_rate",
    "away_prior_era", "away_prior_avg_ip", "away_prior_gs",
    # Home team batting
    "home_team_k_rate", "home_team_bb_rate", "home_team_hr_rate", "home_team_rpg",
    # Away team batting
    "away_team_k_rate", "away_team_bb_rate", "away_team_hr_rate", "away_team_rpg",
    # Environment
    "park_hr_pf",
    "ump_prior_k_rate", "ump_prior_bb_rate",
    "home_rest", "away_rest",
    "is_night",
    # Differentials
    "starter_k_diff", "starter_bb_diff", "starter_era_diff",
    "lineup_rpg_diff", "lineup_k_diff",
]


# ---------------------------------------------------------------------------
# 3. Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward_evaluate(
    df: pd.DataFrame,
    train_end: int,
    test_seasons: list[int],
) -> list[dict]:
    """Train on seasons <= train_end, test on each test season."""
    results = []

    for test_season in test_seasons:
        train = df[(df["season"] >= 2008) & (df["season"] <= train_end)].dropna(subset=FEATURES)
        test = df[df["season"] == test_season].dropna(subset=FEATURES)

        if len(train) < 500 or len(test) < 100:
            log.info(f"  Skipping {test_season}: train={len(train)} test={len(test)}")
            continue

        X_tr, X_te = train[FEATURES].values, test[FEATURES].values

        # --- Total runs ---
        y_total_tr = train["actual_total"].values
        y_total_te = test["actual_total"].values

        reg_total = lgb.LGBMRegressor(**LGB_PARAMS)
        reg_total.fit(X_tr, y_total_tr)
        ml_total = reg_total.predict(X_te)

        # Baseline: league-average total
        baseline_total = np.full_like(y_total_te, y_total_tr.mean(), dtype=float)

        # Sim comparison (if backtest exists)
        sim_bt = _load_sim_backtest_totals(test_season)

        # --- Winner ---
        non_tie = test["actual_margin"] != 0
        y_win = test.loc[non_tie, "home_won"].values

        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X_tr, train["home_won"].values)
        ml_win_p = clf.predict_proba(X_te[non_tie])[:, 1]

        # Baselines
        baseline_win_p = np.full_like(y_win, train["home_won"].mean(), dtype=float)

        row = {
            "train_seasons": f"2008-{train_end}",
            "train_n": len(train),
            "test_season": test_season,
            "test_n": len(test),
        }

        # Total metrics
        row["ml_total_rmse"] = float(np.sqrt(mean_squared_error(y_total_te, ml_total)))
        row["ml_total_mae"] = float(mean_absolute_error(y_total_te, ml_total))
        row["ml_total_bias"] = float(np.mean(ml_total - y_total_te))
        c = np.corrcoef(y_total_te, ml_total)[0, 1]
        row["ml_total_corr"] = float(c) if not np.isnan(c) else 0.0
        row["baseline_total_rmse"] = float(np.sqrt(mean_squared_error(y_total_te, baseline_total)))
        row["baseline_total_mae"] = float(mean_absolute_error(y_total_te, baseline_total))

        if sim_bt is not None:
            row["sim_total_rmse"] = sim_bt["rmse"]
            row["sim_total_mae"] = sim_bt["mae"]
            row["sim_total_corr"] = sim_bt["corr"]
            row["sim_total_bias"] = sim_bt["bias"]

        # Winner metrics
        row["ml_win_brier"] = float(brier_score_loss(y_win, ml_win_p))
        row["ml_win_logloss"] = float(log_loss(y_win, np.clip(ml_win_p, 1e-7, 1-1e-7)))
        row["ml_win_acc"] = float(np.mean((ml_win_p > 0.5) == y_win))
        row["baseline_win_brier"] = float(brier_score_loss(y_win, baseline_win_p))
        row["baseline_win_acc"] = float(np.mean((baseline_win_p > 0.5) == y_win))

        # O/U at common lines
        total_std = max(float(np.std(y_total_tr)), 1.0)
        for line in [6.5, 7.5, 8.5, 9.5]:
            actual_over = (y_total_te > line).astype(float)
            ml_p = 1.0 - norm.cdf(line, loc=ml_total, scale=total_std)
            row[f"ml_ou_{line}_brier"] = float(brier_score_loss(actual_over, ml_p))
            row[f"ml_ou_{line}_acc"] = float(np.mean((ml_p > 0.5) == actual_over))

        # Feature importance
        fi = pd.Series(reg_total.feature_importances_, index=FEATURES)
        row["top_features"] = ", ".join(fi.nlargest(7).index.tolist())

        results.append(row)

    return results


def _load_sim_backtest_totals(test_season: int) -> dict | None:
    """Load sim backtest game totals for comparison."""
    path = DASH_DIR / "backtest_game_sim_backtest_predictions.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = df[df["fold_test_season"] == test_season]
    if len(df) == 0:
        return None

    # Pair pitchers to get game totals
    paired = df.groupby("game_pk").filter(lambda g: len(g) == 2)
    game_totals = paired.groupby("game_pk").agg(
        sim_total=("expected_runs", "sum"),
        actual_total=("actual_total_runs", "sum"),
        sim_std=("std_runs", lambda x: np.sqrt((x**2).sum())),
    ).reset_index()

    sim = game_totals["sim_total"].values
    act = game_totals["actual_total"].values
    c = np.corrcoef(act, sim)[0, 1]

    return {
        "rmse": float(np.sqrt(mean_squared_error(act, sim))),
        "mae": float(mean_absolute_error(act, sim)),
        "corr": float(c) if not np.isnan(c) else 0.0,
        "bias": float(np.mean(sim - act)),
        "n": len(game_totals),
    }


# ---------------------------------------------------------------------------
# K prop stacking with full history
# ---------------------------------------------------------------------------

def build_pitcher_k_dataset(seasons: list[int] | None = None) -> pd.DataFrame:
    """Build per-start K prediction dataset from database."""
    from src.data.db import read_sql

    if seasons is None:
        seasons = list(range(2008, 2027))

    season_list = ",".join(str(int(s)) for s in seasons)

    log.info("Building pitcher K dataset...")
    starts = read_sql(f"""
    SELECT
        pb.game_pk, pb.pitcher_id, dg.season, dg.game_date,
        pb.strike_outs AS actual_k, pb.batters_faced AS actual_bf,
        pb.innings_pitched AS actual_ip, pb.number_of_pitches AS actual_pitches,
        pb.walks AS actual_bb, pb.home_runs AS actual_hr,
        pb.team_id AS pitcher_team_id,
        dg.venue_id
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    WHERE pb.is_starter = true AND pb.batters_faced >= 9
      AND dg.game_type = 'R' AND dg.season IN ({season_list})
    ORDER BY dg.game_date, pb.game_pk
    """)
    log.info(f"  {len(starts)} starter-games loaded")

    # Prior-season pitcher stats
    pit_seasons = read_sql(f"""
    SELECT pitcher_id, dg.season,
           SUM(strike_outs) AS total_k, SUM(batters_faced) AS total_bf,
           SUM(walks) AS total_bb, SUM(home_runs) AS total_hr,
           AVG(innings_pitched) AS avg_ip, AVG(batters_faced) AS avg_bf,
           COUNT(*) AS games_started
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    WHERE pb.is_starter = true AND pb.batters_faced >= 9
      AND dg.game_type = 'R' AND dg.season IN ({season_list})
    GROUP BY pitcher_id, dg.season
    """)
    pit_seasons["k_rate"] = pit_seasons["total_k"] / pit_seasons["total_bf"].clip(lower=1)
    pit_seasons["bb_rate"] = pit_seasons["total_bb"] / pit_seasons["total_bf"].clip(lower=1)
    pit_seasons["hr_rate"] = pit_seasons["total_hr"] / pit_seasons["total_bf"].clip(lower=1)
    pit_seasons["k_per_start"] = pit_seasons["total_k"] / pit_seasons["games_started"].clip(lower=1)

    pit_prior = pit_seasons.copy()
    pit_prior["feature_season"] = pit_prior["season"] + 1
    feat_cols = ["k_rate", "bb_rate", "hr_rate", "avg_ip", "avg_bf",
                 "k_per_start", "games_started"]
    pit_prior = pit_prior.rename(columns={c: f"prior_{c}" for c in feat_cols})

    starts = starts.merge(
        pit_prior[["pitcher_id", "feature_season"] + [f"prior_{c}" for c in feat_cols]].rename(
            columns={"feature_season": "season"}
        ),
        on=["pitcher_id", "season"], how="left",
    )

    # Opposing team batting quality
    team_bat = read_sql(f"""
    SELECT bb.team_id, dg.season,
           SUM(strikeouts)::float / NULLIF(SUM(plate_appearances), 0) AS opp_k_rate,
           SUM(walks)::float / NULLIF(SUM(plate_appearances), 0) AS opp_bb_rate,
           SUM(home_runs)::float / NULLIF(SUM(plate_appearances), 0) AS opp_hr_rate
    FROM staging.batting_boxscores bb
    JOIN production.dim_game dg ON bb.game_pk = dg.game_pk
    WHERE dg.game_type = 'R' AND dg.season IN ({season_list})
    GROUP BY bb.team_id, dg.season
    """)
    team_prior = team_bat.copy()
    team_prior["feature_season"] = team_prior["season"] + 1

    # Need opposing team for each start
    opp_team = read_sql(f"""
    SELECT dg.game_pk,
           CASE WHEN pb.team_id = dg.home_team_id THEN dg.away_team_id
                ELSE dg.home_team_id END AS opp_team_id
    FROM staging.pitching_boxscores pb
    JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
    WHERE pb.is_starter = true AND pb.batters_faced >= 9
      AND dg.game_type = 'R' AND dg.season IN ({season_list})
    """)
    starts = starts.merge(opp_team, on="game_pk", how="left")

    starts = starts.merge(
        team_prior[["team_id", "feature_season", "opp_k_rate", "opp_bb_rate", "opp_hr_rate"]].rename(
            columns={"team_id": "opp_team_id", "feature_season": "season"}
        ),
        on=["opp_team_id", "season"], how="left",
    )

    # Park factor
    pf = read_sql(f"""
    SELECT venue_id, season,
           AVG(COALESCE(hr_pf_3yr, hr_pf_season, 1.0)) AS park_hr_pf
    FROM production.dim_park_factor
    WHERE season IN ({season_list})
    GROUP BY venue_id, season
    """)
    starts = starts.merge(pf, on=["venue_id", "season"], how="left")
    starts["park_hr_pf"] = starts["park_hr_pf"].fillna(1.0)

    log.info(f"  {len(starts)} starts with features")
    return starts


K_FEATURES = [
    "prior_k_rate", "prior_bb_rate", "prior_hr_rate",
    "prior_avg_ip", "prior_avg_bf", "prior_k_per_start", "prior_games_started",
    "opp_k_rate", "opp_bb_rate", "opp_hr_rate",
    "park_hr_pf",
]


def evaluate_k_walk_forward(df: pd.DataFrame) -> list[dict]:
    """Walk-forward K prop evaluation."""
    results = []
    sim_bt_k = _load_sim_k_backtest()

    for test_season in [2023, 2024, 2025]:
        train = df[(df["season"] >= 2008) & (df["season"] < test_season)].dropna(subset=K_FEATURES)
        test = df[df["season"] == test_season].dropna(subset=K_FEATURES)

        if len(train) < 500 or len(test) < 100:
            continue

        X_tr, X_te = train[K_FEATURES].values, test[K_FEATURES].values
        y_tr, y_te = train["actual_k"].values, test["actual_k"].values

        reg = lgb.LGBMRegressor(**LGB_PARAMS)
        reg.fit(X_tr, y_tr)
        ml_pred = reg.predict(X_te)

        # Baseline: prior_k_per_start
        baseline = test["prior_k_per_start"].fillna(y_tr.mean()).values

        row = {
            "train_n": len(train),
            "test_season": test_season,
            "test_n": len(test),
            "ml_k_rmse": float(np.sqrt(mean_squared_error(y_te, ml_pred))),
            "ml_k_mae": float(mean_absolute_error(y_te, ml_pred)),
            "ml_k_bias": float(np.mean(ml_pred - y_te)),
            "ml_k_corr": float(np.corrcoef(y_te, ml_pred)[0, 1]),
            "baseline_k_rmse": float(np.sqrt(mean_squared_error(y_te, baseline))),
            "baseline_k_mae": float(mean_absolute_error(y_te, baseline)),
            "baseline_k_corr": float(np.corrcoef(y_te, baseline)[0, 1]),
        }

        # Sim comparison
        if sim_bt_k is not None and test_season in sim_bt_k:
            row["sim_k_rmse"] = sim_bt_k[test_season]["rmse"]
            row["sim_k_mae"] = sim_bt_k[test_season]["mae"]
            row["sim_k_corr"] = sim_bt_k[test_season]["corr"]
            row["sim_k_bias"] = sim_bt_k[test_season]["bias"]

        # P(over) for common lines
        ml_std = float(np.std(ml_pred - y_te))  # residual std for probability
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            actual_over = (y_te > line).astype(float)
            ml_p = 1.0 - norm.cdf(line, loc=ml_pred, scale=ml_std)
            row[f"ml_k_{line}_brier"] = float(brier_score_loss(actual_over, ml_p))
            row[f"ml_k_{line}_acc"] = float(np.mean((ml_p > 0.5) == actual_over))

        fi = pd.Series(reg.feature_importances_, index=K_FEATURES)
        row["top_features"] = ", ".join(fi.nlargest(5).index.tolist())
        results.append(row)

    return results


def _load_sim_k_backtest() -> dict | None:
    """Load sim K backtest for comparison."""
    path = DASH_DIR / "game_prop_predictions_pitcher_k.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    result = {}
    for season in df["fold_test_season"].unique():
        sub = df[df["fold_test_season"] == season]
        sim = sub["expected_k"].values
        act = sub["actual_k"].values
        c = np.corrcoef(act, sim)[0, 1]
        result[int(season)] = {
            "rmse": float(np.sqrt(mean_squared_error(act, sim))),
            "mae": float(mean_absolute_error(act, sim)),
            "corr": float(c) if not np.isnan(c) else 0.0,
            "bias": float(np.mean(sim - act)),
            "n": len(sub),
        }
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(game_results: list[dict], k_results: list[dict]) -> None:
    log.info("\n" + "=" * 80)
    log.info("GAME-LEVEL ML: Direct from DB features (trained on 15+ years)")
    log.info("=" * 80)

    for r in game_results:
        log.info(f"\n  Test {r['test_season']} | Train: {r['train_seasons']} ({r['train_n']:,} games) | Test: {r['test_n']:,} games")
        log.info(f"  {'':20s} {'ML':>8s} {'Sim':>8s} {'League':>8s}")

        ml_rmse = r["ml_total_rmse"]
        sim_rmse = r.get("sim_total_rmse", np.nan)
        bl_rmse = r["baseline_total_rmse"]
        log.info(f"  {'Total RMSE':20s} {ml_rmse:8.3f} {sim_rmse:8.3f} {bl_rmse:8.3f}")

        ml_mae = r["ml_total_mae"]
        sim_mae = r.get("sim_total_mae", np.nan)
        bl_mae = r["baseline_total_mae"]
        log.info(f"  {'Total MAE':20s} {ml_mae:8.3f} {sim_mae:8.3f} {bl_mae:8.3f}")

        ml_corr = r["ml_total_corr"]
        sim_corr = r.get("sim_total_corr", np.nan)
        log.info(f"  {'Total Corr':20s} {ml_corr:8.3f} {sim_corr:8.3f} {'--':>8s}")

        ml_bias = r["ml_total_bias"]
        sim_bias = r.get("sim_total_bias", np.nan)
        log.info(f"  {'Total Bias':20s} {ml_bias:8.3f} {sim_bias:8.3f} {'--':>8s}")

        log.info(f"\n  {'Winner':20s} {'ML':>8s} {'Sim':>8s} {'Base':>8s}")
        log.info(f"  {'Brier':20s} {r['ml_win_brier']:8.4f} {'--':>8s} {r['baseline_win_brier']:8.4f}")
        log.info(f"  {'Accuracy':20s} {r['ml_win_acc']:7.1%} {'--':>8s} {r['baseline_win_acc']:7.1%}")
        log.info(f"  {'Log Loss':20s} {r['ml_win_logloss']:8.4f} {'--':>8s} {'--':>8s}")

        log.info(f"\n  O/U Lines:")
        log.info(f"  {'Line':6s} {'Brier':>8s} {'Acc':>8s}")
        for line in [6.5, 7.5, 8.5, 9.5]:
            b = r.get(f"ml_ou_{line}_brier", np.nan)
            a = r.get(f"ml_ou_{line}_acc", np.nan)
            log.info(f"  {line:<6.1f} {b:8.4f} {a:7.1%}")

        log.info(f"  Top features: {r['top_features']}")

    log.info("\n" + "=" * 80)
    log.info("K PROP ML: Direct from DB features")
    log.info("=" * 80)

    for r in k_results:
        log.info(f"\n  Test {r['test_season']} | Train: {r['train_n']:,} starts | Test: {r['test_n']:,} starts")
        log.info(f"  {'':20s} {'ML':>8s} {'Sim':>8s} {'Baseline':>8s}")

        for m in ["rmse", "mae", "corr", "bias"]:
            ml = r[f"ml_k_{m}"]
            sim = r.get(f"sim_k_{m}", np.nan)
            bl = r.get(f"baseline_k_{m}", np.nan)
            log.info(f"  {'K ' + m.upper():20s} {ml:8.3f} {sim:8.3f} {bl:8.3f}")

        log.info(f"\n  K P(Over):")
        log.info(f"  {'Line':6s} {'Brier':>8s} {'Acc':>8s}")
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            b = r.get(f"ml_k_{line}_brier", np.nan)
            a = r.get(f"ml_k_{line}_acc", np.nan)
            if not np.isnan(b):
                log.info(f"  {line:<6.1f} {b:8.4f} {a:7.1%}")

        log.info(f"  Top features: {r['top_features']}")

    # Summary
    log.info("\n" + "=" * 80)
    log.info("SUMMARY COMPARISON")
    log.info("=" * 80)
    log.info(f"\n  {'Season':8s} {'ML Total':>10s} {'Sim Total':>10s} {'ML Corr':>8s} {'Sim Corr':>8s} {'ML Win%':>8s}")
    for r in game_results:
        yr = r["test_season"]
        ml_r = r["ml_total_rmse"]
        sim_r = r.get("sim_total_rmse", np.nan)
        ml_c = r["ml_total_corr"]
        sim_c = r.get("sim_total_corr", np.nan)
        ml_w = r["ml_win_acc"]
        log.info(f"  {yr:<8d} {ml_r:10.3f} {sim_r:10.3f} {ml_c:8.3f} {sim_c:8.3f} {ml_w:7.1%}")

    log.info(f"\n  {'Season':8s} {'ML K RMSE':>10s} {'Sim K RMSE':>10s} {'ML Corr':>8s} {'Sim Corr':>8s}")
    for r in k_results:
        yr = r["test_season"]
        ml_r = r["ml_k_rmse"]
        sim_r = r.get("sim_k_rmse", np.nan)
        ml_c = r["ml_k_corr"]
        sim_c = r.get("sim_k_corr", np.nan)
        log.info(f"  {yr:<8d} {ml_r:10.3f} {sim_r:10.3f} {ml_c:8.3f} {sim_c:8.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Build datasets from DB
    games = build_game_dataset(list(range(2008, 2027)))
    k_starts = build_pitcher_k_dataset(list(range(2008, 2027)))

    # Game-level: expanding window walk-forward
    log.info("\nRunning game-level walk-forward evaluation...")
    game_results = []
    for test_yr in [2023, 2024, 2025]:
        game_results.extend(walk_forward_evaluate(games, test_yr - 1, [test_yr]))

    # K prop: expanding window
    log.info("\nRunning K prop walk-forward evaluation...")
    k_results = evaluate_k_walk_forward(k_starts)

    print_results(game_results, k_results)


if __name__ == "__main__":
    main()
