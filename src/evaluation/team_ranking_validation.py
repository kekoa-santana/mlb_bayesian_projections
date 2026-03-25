"""
Walk-forward validation for team rankings.

Learns optimal composite weights for team rankings by comparing
predicted team strength against actual season wins.

Metrics: RMSE, MAE, Spearman rank correlation, playoff prediction
accuracy, calibration.

Default folds:
  - Predict 2022 from prior data
  - Predict 2023 from prior data
  - Predict 2024 from prior data
  - Predict 2025 from prior data
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from src.data.db import read_sql

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------
DEFAULT_FOLDS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]

PYTH_EXP = 1.83  # Baseball Pythagorean exponent

# Sub-component names for team composite
_TEAM_SCORE_COLS = [
    "offense_score", "rotation_score", "bullpen_score",
    "defense_score", "depth_score",
]

# Playoff qualification thresholds (approximate, varies by format year)
# 2022+: 6 teams per league (3 division winners + 3 wild cards)
# 12 total playoff teams out of 30
_PLAYOFF_THRESHOLD_WINS = {
    2022: 87,
    2023: 84,
    2024: 86,
    2025: 85,
}


# ===================================================================
# Data loading
# ===================================================================

def _load_team_season_records(
    seasons: list[int],
) -> pd.DataFrame:
    """Aggregate game results into team season records.

    Parameters
    ----------
    seasons : list[int]
        Seasons to load.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, season, wins, losses, rs, ra, games, rpg,
        ra_per_game, run_diff, win_pct, pythagorean_wins.
    """
    query = """
    SELECT
        fgt.team_id,
        dg.season,
        fgt.home_away,
        fgt.runs,
        CASE
            WHEN fgt.home_away = 'home'
                THEN CASE WHEN fgt.runs > opp.runs THEN 1 ELSE 0 END
            ELSE CASE WHEN fgt.runs > opp.runs THEN 1 ELSE 0 END
        END AS win,
        opp.runs AS opp_runs
    FROM production.fact_game_totals fgt
    JOIN production.dim_game dg ON fgt.game_pk = dg.game_pk
    JOIN production.fact_game_totals opp
        ON fgt.game_pk = opp.game_pk
        AND fgt.home_away != opp.home_away
    WHERE dg.season = ANY(:seasons)
      AND dg.game_type = 'R'
    """
    games = read_sql(query, {"seasons": seasons})

    if games.empty:
        return pd.DataFrame()

    agg = games.groupby(["team_id", "season"]).agg(
        wins=("win", "sum"),
        games=("win", "count"),
        rs=("runs", "sum"),
        ra=("opp_runs", "sum"),
    ).reset_index()

    agg["losses"] = agg["games"] - agg["wins"]
    agg["rpg"] = agg["rs"] / agg["games"]
    agg["ra_per_game"] = agg["ra"] / agg["games"]
    agg["run_diff"] = agg["rs"] - agg["ra"]
    agg["win_pct"] = agg["wins"] / agg["games"]

    # Pythagorean expected wins
    agg["pythagorean_wins"] = agg.apply(
        lambda r: _pythagorean_wins(r["rpg"], r["ra_per_game"], int(r["games"])),
        axis=1,
    )

    logger.info(
        "Loaded team records: %d team-seasons across %s",
        len(agg), seasons,
    )
    return agg


def _pythagorean_wins(
    rpg: float,
    ra_per_game: float,
    games: int = 162,
    exp: float = PYTH_EXP,
) -> float:
    """Projected wins from Pythagorean formula."""
    if rpg <= 0 or ra_per_game <= 0:
        return games / 2
    wpct = rpg ** exp / (rpg ** exp + ra_per_game ** exp)
    return round(wpct * games, 1)


def _build_team_sub_scores(
    seasons: list[int],
) -> pd.DataFrame:
    """Build team-level sub-scores from observable data.

    Constructs percentile-rank components for each team-season that
    would have been available at prediction time.

    Parameters
    ----------
    seasons : list[int]
        Seasons to build scores for.

    Returns
    -------
    pd.DataFrame
        Columns: team_id, season, offense_score, rotation_score,
        bullpen_score, defense_score, depth_score, elo_score,
        projected_wins.
    """
    records_list = []

    for season in seasons:
        # --- Build player-to-team mapping for this season ---
        # Use fact_player_game_mlb (has team_id per game) to assign
        # each player to their primary team for the season (most PA).
        try:
            player_teams = read_sql("""
                SELECT player_id,
                       team_id,
                       SUM(COALESCE(bat_pa, 0)) + COUNT(*) AS total_appearances
                FROM production.fact_player_game_mlb
                WHERE season = :season
                GROUP BY player_id, team_id
            """, {"season": season})
            if not player_teams.empty:
                # Keep the team where each player had the most appearances
                idx = player_teams.groupby("player_id")["total_appearances"].idxmax()
                player_team_map = player_teams.loc[idx, ["player_id", "team_id"]]
            else:
                player_team_map = pd.DataFrame(columns=["player_id", "team_id"])
        except Exception:
            player_team_map = pd.DataFrame(columns=["player_id", "team_id"])

        # --- Offense: team-level batting advanced stats ---
        try:
            bat_adv = read_sql("""
                SELECT batter_id, wrc_plus, woba, pa
                FROM production.fact_batting_advanced
                WHERE season = :season AND pa >= 50
            """, {"season": season})
            if not bat_adv.empty and not player_team_map.empty:
                bat_adv = bat_adv.merge(
                    player_team_map.rename(columns={"player_id": "batter_id"}),
                    on="batter_id", how="inner",
                )
                team_offense = bat_adv.groupby("team_id").agg(
                    avg_wrc_plus=("wrc_plus", "mean"),
                    avg_woba=("woba", "mean"),
                    total_pa=("pa", "sum"),
                    n_batters=("batter_id", "count"),
                ).reset_index()
                team_offense = team_offense[team_offense["n_batters"] >= 5]
            else:
                team_offense = pd.DataFrame()
        except Exception:
            team_offense = pd.DataFrame()

        # --- Pitching: team-level pitching advanced stats ---
        try:
            pit_adv = read_sql("""
                SELECT pitcher_id, k_pct, bb_pct, xwoba_against, swstr_pct,
                       batters_faced
                FROM production.fact_pitching_advanced
                WHERE season = :season AND batters_faced >= 50
            """, {"season": season})
            if not pit_adv.empty and not player_team_map.empty:
                pit_adv = pit_adv.merge(
                    player_team_map.rename(columns={"player_id": "pitcher_id"}),
                    on="pitcher_id", how="inner",
                )
                team_pitching = pit_adv.groupby("team_id").agg(
                    avg_k_pct=("k_pct", "mean"),
                    avg_bb_pct=("bb_pct", "mean"),
                    avg_xwoba_against=("xwoba_against", "mean"),
                    avg_swstr_pct=("swstr_pct", "mean"),
                    n_pitchers=("pitcher_id", "count"),
                ).reset_index()
                team_pitching = team_pitching[team_pitching["n_pitchers"] >= 5]
            else:
                team_pitching = pd.DataFrame()
        except Exception:
            team_pitching = pd.DataFrame()

        # --- Defense: team OAA if available ---
        try:
            oaa = read_sql("""
                SELECT player_id,
                       SUM(outs_above_average) AS player_oaa
                FROM production.fact_fielding_oaa
                WHERE season = :season
                GROUP BY player_id
            """, {"season": season})
            if not oaa.empty and not player_team_map.empty:
                oaa = oaa.merge(player_team_map, on="player_id", how="inner")
                team_defense = oaa.groupby("team_id").agg(
                    team_oaa=("player_oaa", "sum"),
                ).reset_index()
            else:
                team_defense = pd.DataFrame()
        except Exception:
            team_defense = pd.DataFrame()

        # --- Season records for depth proxy (games played variability) ---
        try:
            team_records = _load_team_season_records([season])
        except Exception:
            team_records = pd.DataFrame()

        # --- Build per-team scores ---
        # Use game results as the authoritative source of teams for the season
        if not team_records.empty:
            team_ids = set(team_records["team_id"])
        else:
            team_ids = set()
            if not team_offense.empty:
                team_ids |= set(team_offense["team_id"])
            if not team_pitching.empty:
                team_ids |= set(team_pitching["team_id"])

        if not team_ids:
            continue

        season_df = pd.DataFrame({"team_id": list(team_ids), "season": season})

        # Merge offense
        if not team_offense.empty:
            season_df = season_df.merge(team_offense, on="team_id", how="left")
            season_df["offense_score"] = season_df["avg_wrc_plus"].rank(
                pct=True, method="average",
            )
        else:
            season_df["offense_score"] = 0.5

        # Merge pitching
        if not team_pitching.empty:
            season_df = season_df.merge(team_pitching, on="team_id", how="left")
            # For rotation vs bullpen: use K% and xwOBA-against
            # Higher K% = better, lower xwOBA = better
            season_df["rotation_score"] = season_df["avg_k_pct"].rank(
                pct=True, method="average",
            )
            season_df["bullpen_score"] = (
                1.0 - season_df["avg_xwoba_against"].rank(
                    pct=True, method="average",
                )
            )
        else:
            season_df["rotation_score"] = 0.5
            season_df["bullpen_score"] = 0.5

        # Merge defense
        if not team_defense.empty:
            season_df = season_df.merge(team_defense, on="team_id", how="left")
            season_df["team_oaa"] = season_df["team_oaa"].fillna(0)
            season_df["defense_score"] = season_df["team_oaa"].rank(
                pct=True, method="average",
            )
        else:
            season_df["defense_score"] = 0.5

        # --- Depth: concentration + replacement gap + bench quality ---
        # Measures roster resilience, not just average quality.
        try:
            # Hitter depth: need per-player value for this team
            if not bat_adv.empty and "team_id" in bat_adv.columns:
                depth_rows = []
                for tid in team_ids:
                    team_hitters = bat_adv[bat_adv["team_id"] == tid].copy()
                    team_pitchers = (
                        pit_adv[pit_adv["team_id"] == tid].copy()
                        if (not pit_adv.empty and "team_id" in pit_adv.columns)
                        else pd.DataFrame()
                    )

                    # -- Hitter concentration & gap --
                    h_vals = team_hitters.sort_values("wrc_plus", ascending=False)["wrc_plus"].values
                    n_h = len(h_vals)
                    if n_h >= 5:
                        top3_share = h_vals[:3].sum() / (h_vals.sum() + 1e-9)
                        starter_avg = h_vals[:min(9, n_h)].mean()
                        bench_avg = h_vals[9:min(15, n_h)].mean() if n_h > 9 else 80.0
                        h_gap = starter_avg - bench_avg  # smaller = deeper
                        h_bench_q = bench_avg
                    else:
                        top3_share = 0.33
                        h_gap = 30.0
                        h_bench_q = 80.0

                    # -- Pitcher concentration & gap --
                    if not team_pitchers.empty and len(team_pitchers) >= 3:
                        p_vals = team_pitchers.sort_values(
                            "k_pct", ascending=False
                        )["k_pct"].values
                        p_bench_q = p_vals[5:min(10, len(p_vals))].mean() if len(p_vals) > 5 else 0.15
                        p_breadth = len(p_vals)
                    else:
                        p_bench_q = 0.15
                        p_breadth = 5

                    # -- Roster breadth --
                    roster_size = n_h + (len(team_pitchers) if not team_pitchers.empty else 0)

                    depth_rows.append({
                        "team_id": tid,
                        "concentration": top3_share,
                        "replacement_gap": h_gap,
                        "bench_quality": h_bench_q,
                        "pitcher_bench": p_bench_q,
                        "roster_breadth": roster_size,
                    })

                if depth_rows:
                    depth_df = pd.DataFrame(depth_rows)
                    # Percentile-rank each component (lower concentration = better, etc.)
                    conc_pctl = 1.0 - depth_df["concentration"].rank(pct=True, method="average")
                    gap_pctl = 1.0 - depth_df["replacement_gap"].rank(pct=True, method="average")
                    bench_pctl = depth_df["bench_quality"].rank(pct=True, method="average")
                    pbench_pctl = depth_df["pitcher_bench"].rank(pct=True, method="average")
                    breadth_pctl = depth_df["roster_breadth"].rank(pct=True, method="average")

                    depth_df["depth_score"] = (
                        0.30 * conc_pctl      # less top-heavy = better
                        + 0.25 * gap_pctl     # smaller starter-to-bench gap = better
                        + 0.20 * bench_pctl   # better bench quality = better
                        + 0.15 * pbench_pctl  # better pitching depth = better
                        + 0.10 * breadth_pctl # more usable players = better
                    )
                    season_df = season_df.merge(
                        depth_df[["team_id", "depth_score"]],
                        on="team_id", how="left",
                    )
                    season_df["depth_score"] = season_df["depth_score"].fillna(0.5)
                else:
                    season_df["depth_score"] = 0.5
            else:
                season_df["depth_score"] = 0.5
        except Exception:
            season_df["depth_score"] = 0.5

        # Projected wins from Pythagorean (if we have records for this season)
        if not team_records.empty:
            season_df = season_df.merge(
                team_records[["team_id", "pythagorean_wins", "wins", "rpg", "ra_per_game"]],
                on="team_id", how="left",
            )
            if season_df["pythagorean_wins"].notna().any():
                season_df["projected_wins"] = season_df["pythagorean_wins"]
            else:
                season_df["projected_wins"] = 81.0
        else:
            season_df["projected_wins"] = 81.0
            season_df["wins"] = np.nan

        # ELO score placeholder (would come from elo history if available)
        season_df["elo_score"] = 0.5

        # Fill NaN scores with 0.5
        for col in _TEAM_SCORE_COLS + ["elo_score"]:
            season_df[col] = season_df[col].fillna(0.5)

        records_list.append(season_df)

    if not records_list:
        return pd.DataFrame()

    return pd.concat(records_list, ignore_index=True)


# ===================================================================
# Evaluation metrics
# ===================================================================

def evaluate_team_predictions(
    predicted_wins: pd.Series,
    actual_wins: pd.Series,
) -> dict[str, Any]:
    """Evaluate team win predictions against actuals.

    Parameters
    ----------
    predicted_wins : pd.Series
        Predicted win totals.
    actual_wins : pd.Series
        Actual win totals.

    Returns
    -------
    dict
        rmse, mae, spearman_rho, mean_error, max_error.
    """
    # Align indexes before masking (concat across folds may produce
    # different-length Series sharing the same team_id keys).
    common = predicted_wins.index.intersection(actual_wins.index)
    predicted_wins = predicted_wins.loc[common]
    actual_wins = actual_wins.loc[common]
    mask = predicted_wins.notna() & actual_wins.notna()
    pred = predicted_wins[mask].values.astype(float)
    actual = actual_wins[mask].values.astype(float)

    if len(pred) < 5:
        return {
            "rmse": np.nan, "mae": np.nan,
            "spearman_rho": np.nan, "n_teams": 0,
        }

    errors = pred - actual
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    mean_error = float(np.mean(errors))
    max_error = float(np.max(np.abs(errors)))
    rho, pval = spearmanr(pred, actual)

    return {
        "rmse": rmse,
        "mae": mae,
        "mean_error": mean_error,
        "max_error": max_error,
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "n_teams": len(pred),
    }


def _evaluate_playoff_prediction(
    predicted_wins: pd.Series,
    actual_wins: pd.Series,
    season: int,
    n_playoff_teams: int = 12,
) -> dict[str, Any]:
    """Evaluate playoff qualification prediction accuracy.

    Parameters
    ----------
    predicted_wins : pd.Series
        Predicted win totals, indexed by team_id.
    actual_wins : pd.Series
        Actual win totals, indexed by team_id.
    season : int
        Season for threshold lookup.
    n_playoff_teams : int
        Number of playoff teams. Defaults to 12 (2022+ format).

    Returns
    -------
    dict
        accuracy, precision, recall for playoff prediction.
    """
    mask = predicted_wins.notna() & actual_wins.notna()
    pred = predicted_wins[mask]
    actual = actual_wins[mask]

    if len(pred) < 10:
        return {"playoff_accuracy": np.nan, "playoff_precision": np.nan,
                "playoff_recall": np.nan}

    # Top N by predicted wins = predicted playoff teams
    pred_playoff = set(pred.nlargest(n_playoff_teams).index)
    actual_playoff = set(actual.nlargest(n_playoff_teams).index)

    correct = pred_playoff & actual_playoff
    accuracy = len(correct) / n_playoff_teams
    precision = len(correct) / len(pred_playoff) if pred_playoff else 0
    recall = len(correct) / len(actual_playoff) if actual_playoff else 0

    return {
        "playoff_accuracy": accuracy,
        "playoff_precision": precision,
        "playoff_recall": recall,
        "correct_playoff_teams": len(correct),
        "n_playoff_teams": n_playoff_teams,
    }


def compare_ranking_methods(
    methods: dict[str, pd.Series],
    actual: pd.Series,
) -> pd.DataFrame:
    """Head-to-head comparison of multiple ranking methods.

    Parameters
    ----------
    methods : dict[str, pd.Series]
        Method name -> predicted win totals.
    actual : pd.Series
        Actual win totals.

    Returns
    -------
    pd.DataFrame
        One row per method with RMSE, MAE, Spearman rho.
    """
    records = []
    for name, pred in methods.items():
        metrics = evaluate_team_predictions(pred, actual)
        metrics["method"] = name
        records.append(metrics)

    return pd.DataFrame(records).set_index("method")


# ===================================================================
# Weight optimization
# ===================================================================

def learn_team_weights(
    components: pd.DataFrame,
    actual_wins: pd.Series,
    score_cols: list[str] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Optimize composite weights for team rankings.

    Parameters
    ----------
    components : pd.DataFrame
        Team sub-score columns, indexed by team_id.
    actual_wins : pd.Series
        Actual season wins, indexed by team_id.
    score_cols : list[str], optional
        Columns to optimize. Defaults to _TEAM_SCORE_COLS.
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict
        optimal_weights, rmse, mae, spearman_rho.
    """
    if score_cols is None:
        score_cols = _TEAM_SCORE_COLS

    common_idx = components.index.intersection(actual_wins.index)
    X = components.loc[common_idx, score_cols].values
    y = actual_wins.loc[common_idx].values.astype(float)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    if len(y) < 15:
        logger.warning(
            "Too few team-seasons (%d) for weight optimization", len(y),
        )
        n = len(score_cols)
        return {
            "optimal_weights": {col: 1.0 / n for col in score_cols},
            "rmse": np.nan,
            "mae": np.nan,
            "spearman_rho": np.nan,
        }

    n_weights = len(score_cols)

    def _softmax(raw: np.ndarray) -> np.ndarray:
        exp = np.exp(raw - raw.max())
        return exp / exp.sum()

    def _objective(raw: np.ndarray) -> float:
        """Predict wins as linear combo of scores, minimize RMSE."""
        w = _softmax(raw)
        composite = X @ w
        # Linear regression: wins = a * composite + b
        # Solve for a, b via least squares
        A = np.column_stack([composite, np.ones(len(composite))])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coeffs
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    rng = np.random.default_rng(random_seed)
    best_result = None
    best_loss = np.inf

    for _ in range(15):
        x0 = rng.normal(0, 0.5, size=n_weights)
        res = minimize(
            _objective, x0, method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-8},
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_result = res

    optimal_weights = _softmax(best_result.x)

    # Compute final predicted wins for metrics
    composite = X @ optimal_weights
    A = np.column_stack([composite, np.ones(len(composite))])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred_wins = A @ coeffs

    rmse = float(np.sqrt(np.mean((pred_wins - y) ** 2)))
    mae = float(np.mean(np.abs(pred_wins - y)))
    rho, _ = spearmanr(pred_wins, y)

    weight_dict = {col: float(w) for col, w in zip(score_cols, optimal_weights)}

    logger.info(
        "Optimal team weights: %s | RMSE=%.2f, MAE=%.2f, rho=%.3f",
        {k: f"{v:.3f}" for k, v in weight_dict.items()}, rmse, mae, rho,
    )

    return {
        "optimal_weights": weight_dict,
        "rmse": rmse,
        "mae": mae,
        "spearman_rho": float(rho),
        "n_teams": len(y),
        "regression_coeffs": {"slope": float(coeffs[0]), "intercept": float(coeffs[1])},
    }


# ===================================================================
# Walk-forward validation
# ===================================================================

def validate_team_rankings(
    folds: list[tuple[int, int]] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Walk-forward validation for team ranking weights.

    For each fold, builds team sub-scores from the training season
    and compares predicted team strength against actual next-season wins.

    Parameters
    ----------
    folds : list[tuple[int, int]], optional
        (last_train_season, test_season) pairs.
    random_seed : int
        For optimizer reproducibility.

    Returns
    -------
    dict
        per_fold: list of fold results,
        pooled_weights: optimal weights across all data,
        summary: aggregated metrics DataFrame,
        method_comparison: head-to-head of baselines.
    """
    if folds is None:
        folds = DEFAULT_FOLDS

    all_seasons = sorted(set(
        [f[0] for f in folds] + [f[1] for f in folds]
    ))
    all_records = _load_team_season_records(all_seasons)
    if all_records.empty:
        logger.error("No team records loaded")
        return {"per_fold": [], "pooled_weights": {}, "summary": pd.DataFrame()}

    # Build sub-scores for all training seasons
    train_seasons = sorted(set(f[0] for f in folds))
    all_scores = _build_team_sub_scores(train_seasons)

    per_fold: list[dict[str, Any]] = []
    pooled_scores_list: list[pd.DataFrame] = []
    pooled_wins_list: list[pd.Series] = []

    # Accumulate method predictions for overall comparison
    method_preds: dict[str, list[pd.Series]] = {
        "optimized": [],
        "equal_weight": [],
        "pythagorean": [],
        "prior_wins": [],
    }
    method_actuals: list[pd.Series] = []

    for last_train, test_season in folds:
        logger.info(
            "=== Team fold: train %d, test %d ===",
            last_train, test_season,
        )

        # Training sub-scores from last_train season
        train_scores = all_scores[all_scores["season"] == last_train].copy()
        if train_scores.empty:
            logger.warning("No team sub-scores for season %d", last_train)
            continue

        # Test actuals
        test_records = all_records[all_records["season"] == test_season].copy()
        if test_records.empty:
            logger.warning("No team records for season %d", test_season)
            continue

        # Common teams
        common = set(train_scores["team_id"]) & set(test_records["team_id"])
        if len(common) < 15:
            logger.warning(
                "Only %d common teams for fold %d->%d",
                len(common), last_train, test_season,
            )
            continue

        train_fold = train_scores[train_scores["team_id"].isin(common)].set_index("team_id")
        test_fold = test_records[test_records["team_id"].isin(common)].set_index("team_id")
        actual_wins = test_fold["wins"]

        # --- Baseline 1: Equal-weight composite ---
        baseline_composite = train_fold[_TEAM_SCORE_COLS].mean(axis=1)
        baseline_pred = _scores_to_wins(baseline_composite, actual_wins)
        baseline_metrics = evaluate_team_predictions(baseline_pred, actual_wins)

        # --- Baseline 2: Pythagorean wins from prior season ---
        if "projected_wins" in train_fold.columns:
            pyth_pred = train_fold["projected_wins"]
        else:
            pyth_pred = pd.Series(81.0, index=train_fold.index)
        pyth_metrics = evaluate_team_predictions(pyth_pred, actual_wins)

        # --- Baseline 3: Prior season actual wins ---
        prior_records = all_records[all_records["season"] == last_train].copy()
        prior_wins_map = prior_records.set_index("team_id")["wins"]
        prior_wins_pred = prior_wins_map.reindex(actual_wins.index)
        prior_metrics = evaluate_team_predictions(prior_wins_pred, actual_wins)

        # --- Optimized weights ---
        opt = learn_team_weights(
            train_fold, actual_wins, random_seed=random_seed,
        )

        # Generate optimized predictions for this fold
        w_arr = np.array([opt["optimal_weights"][c] for c in _TEAM_SCORE_COLS])
        optimized_composite = train_fold[_TEAM_SCORE_COLS].values @ w_arr
        optimized_series = pd.Series(optimized_composite, index=train_fold.index)
        optimized_pred = _scores_to_wins(optimized_series, actual_wins)

        # --- Playoff prediction ---
        playoff = _evaluate_playoff_prediction(
            optimized_pred, actual_wins, test_season,
        )

        # --- Individual component correlations ---
        component_corrs = {}
        for col in _TEAM_SCORE_COLS:
            sub = train_fold[col]
            valid = sub.notna() & actual_wins.notna()
            if valid.sum() > 5 and sub[valid].std() > 1e-9:
                rho_val, _ = spearmanr(sub[valid], actual_wins[valid])
                component_corrs[col] = float(rho_val) if np.isfinite(rho_val) else 0.0
            else:
                component_corrs[col] = 0.0

        fold_result = {
            "fold": f"{last_train}->{test_season}",
            "last_train": last_train,
            "test_season": test_season,
            "n_teams": len(common),
            "optimal_weights": opt["optimal_weights"],
            "optimized_rmse": opt["rmse"],
            "optimized_mae": opt["mae"],
            "optimized_rho": opt["spearman_rho"],
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_rho": baseline_metrics["spearman_rho"],
            "pythagorean_rmse": pyth_metrics["rmse"],
            "pythagorean_rho": pyth_metrics["spearman_rho"],
            "prior_wins_rmse": prior_metrics["rmse"],
            "prior_wins_rho": prior_metrics["spearman_rho"],
            "playoff_accuracy": playoff["playoff_accuracy"],
            "component_correlations": component_corrs,
        }
        per_fold.append(fold_result)

        logger.info(
            "Fold %s: optimized RMSE=%.2f (baseline=%.2f, pyth=%.2f, prior=%.2f)",
            fold_result["fold"], opt["rmse"],
            baseline_metrics["rmse"], pyth_metrics["rmse"], prior_metrics["rmse"],
        )
        logger.info(
            "  rho: optimized=%.3f, baseline=%.3f, pyth=%.3f, prior=%.3f",
            opt["spearman_rho"], baseline_metrics["spearman_rho"],
            pyth_metrics["spearman_rho"], prior_metrics["spearman_rho"],
        )
        logger.info(
            "  Playoff accuracy: %.0f%% (%d/%d)",
            playoff["playoff_accuracy"] * 100,
            playoff.get("correct_playoff_teams", 0),
            playoff.get("n_playoff_teams", 12),
        )

        # Accumulate for pooled optimization and method comparison
        pooled_scores_list.append(train_fold)
        pooled_wins_list.append(actual_wins)

        method_preds["optimized"].append(optimized_pred)
        method_preds["equal_weight"].append(baseline_pred)
        method_preds["pythagorean"].append(pyth_pred)
        method_preds["prior_wins"].append(prior_wins_pred)
        method_actuals.append(actual_wins)

    # Pooled optimization
    pooled_weights: dict[str, float] = {}
    if pooled_scores_list:
        pooled_scores = pd.concat(pooled_scores_list)
        pooled_wins = pd.concat(pooled_wins_list)
        pooled_opt = learn_team_weights(
            pooled_scores, pooled_wins, random_seed=random_seed,
        )
        pooled_weights = pooled_opt["optimal_weights"]
        logger.info("Pooled team weights: %s", pooled_weights)

    # Overall method comparison
    method_comparison = pd.DataFrame()
    if method_actuals:
        all_actual = pd.concat(method_actuals)
        method_series = {}
        for name, pred_list in method_preds.items():
            method_series[name] = pd.concat(pred_list)
        method_comparison = compare_ranking_methods(method_series, all_actual)
        logger.info("\nMethod comparison:\n%s", method_comparison.to_string())

    # Summary DataFrame
    summary_records = []
    for fold in per_fold:
        summary_records.append({
            "fold": fold["fold"],
            "n_teams": fold["n_teams"],
            "optimized_rmse": fold["optimized_rmse"],
            "baseline_rmse": fold["baseline_rmse"],
            "pythagorean_rmse": fold["pythagorean_rmse"],
            "prior_wins_rmse": fold["prior_wins_rmse"],
            "optimized_rho": fold["optimized_rho"],
            "baseline_rho": fold["baseline_rho"],
            "pythagorean_rho": fold["pythagorean_rho"],
            "prior_wins_rho": fold["prior_wins_rho"],
            "playoff_accuracy": fold["playoff_accuracy"],
        })
    summary = pd.DataFrame(summary_records) if summary_records else pd.DataFrame()

    return {
        "per_fold": per_fold,
        "pooled_weights": pooled_weights,
        "summary": summary,
        "method_comparison": method_comparison,
    }


def _scores_to_wins(
    composite_scores: pd.Series,
    actual_wins: pd.Series,
) -> pd.Series:
    """Convert composite scores to win predictions via linear regression.

    Fits wins = a * score + b on the available data, then returns
    predicted wins for all teams.

    Parameters
    ----------
    composite_scores : pd.Series
        Composite team scores (0-1 scale).
    actual_wins : pd.Series
        Actual season wins for fitting.

    Returns
    -------
    pd.Series
        Predicted win totals.
    """
    common = composite_scores.index.intersection(actual_wins.index)
    x = composite_scores.loc[common].values.astype(float)
    y = actual_wins.loc[common].values.astype(float)

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return pd.Series(81.0, index=composite_scores.index)

    A = np.column_stack([x[valid], np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, y[valid], rcond=None)

    pred = composite_scores.values * coeffs[0] + coeffs[1]
    return pd.Series(pred, index=composite_scores.index)


# ===================================================================
# Compare current vs learned weights
# ===================================================================

def compare_to_current_weights(
    results: dict[str, Any],
) -> pd.DataFrame:
    """Compare learned team weights against current team_rankings.py weights.

    Parameters
    ----------
    results : dict
        Output of ``validate_team_rankings()``.

    Returns
    -------
    pd.DataFrame
        Side-by-side comparison.
    """
    from src.models.team_rankings import _TALENT_WEIGHTS, _ELO_WEIGHT, _PROJ_WEIGHT

    pooled = results.get("pooled_weights", {})
    records = []

    # Map our score_cols to the talent weight keys
    col_to_key = {
        "offense_score": "offense",
        "rotation_score": "rotation",
        "bullpen_score": "bullpen",
        "defense_score": "defense",
        "depth_score": "depth",
    }
    for col in _TEAM_SCORE_COLS:
        key = col_to_key.get(col, col)
        records.append({
            "component": col,
            "current_weight": _TALENT_WEIGHTS.get(key, 0.0),
            "learned_weight": pooled.get(col, 0.0),
            "delta": pooled.get(col, 0.0) - _TALENT_WEIGHTS.get(key, 0.0),
        })

    # Add ELO and projection weights for context
    records.append({
        "component": "elo (not optimized)",
        "current_weight": _ELO_WEIGHT,
        "learned_weight": np.nan,
        "delta": np.nan,
    })
    records.append({
        "component": "projections (not optimized)",
        "current_weight": _PROJ_WEIGHT,
        "learned_weight": np.nan,
        "delta": np.nan,
    })

    return pd.DataFrame(records)


# ===================================================================
# Main entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

    logger.info("=" * 70)
    logger.info("TEAM RANKING WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    results = validate_team_rankings()

    if results["summary"] is not None and not results["summary"].empty:
        logger.info("\nSummary:\n%s", results["summary"].to_string())

    if results["pooled_weights"]:
        logger.info("\nPooled optimal weights:")
        for k, v in sorted(
            results["pooled_weights"].items(), key=lambda x: -x[1],
        ):
            logger.info("  %-20s  %.3f", k, v)

    # Per-fold component correlations
    for fold in results["per_fold"]:
        logger.info(
            "\nFold %s component correlations with actual wins:",
            fold["fold"],
        )
        for col, rho in sorted(
            fold["component_correlations"].items(), key=lambda x: -x[1],
        ):
            logger.info("  %-20s  rho=%.3f", col, rho)

    # Method comparison
    if not results["method_comparison"].empty:
        logger.info("\nOverall method comparison:")
        logger.info("\n%s", results["method_comparison"].to_string())

    # Compare to current weights
    try:
        comparison = compare_to_current_weights(results)
        logger.info("\nCurrent vs learned weights:")
        logger.info("\n%s", comparison.to_string())
    except ImportError:
        logger.warning("Could not import current weights for comparison")

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)
