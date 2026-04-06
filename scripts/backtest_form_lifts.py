#!/usr/bin/env python
"""
A/B backtest: rolling form lifts vs no form lifts.

For each pitcher-game and batter-game in the test set, runs the simulator
twice (with and without form lifts) and compares to actual outcomes.

Uses the rolling data from fact_player_form_rolling as of the game date
(the row from the game immediately before) to compute form lifts.

Usage
-----
    python scripts/backtest_form_lifts.py                     # 2025, 500 games
    python scripts/backtest_form_lifts.py --season 2024       # different year
    python scripts/backtest_form_lifts.py --max-games 2000    # more games
    python scripts/backtest_form_lifts.py --mode batter       # batter only
    python scripts/backtest_form_lifts.py --mode pitcher      # pitcher only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling form lookup: get the form data from the game BEFORE each target game
# ---------------------------------------------------------------------------
def load_rolling_form_all(
    player_ids: list[int],
    player_role: str,
    season: int,
) -> dict[int, pd.DataFrame]:
    """Load all rolling form rows for a set of players in a season.

    Returns a dict keyed by player_id, each value a DataFrame sorted
    by game_date. Use `lookup_prior_form()` to get the row before a
    specific game date.

    Parameters
    ----------
    player_ids : list[int]
        Player IDs.
    player_role : str
        'batter' or 'pitcher'.
    season : int
        Season to restrict to.

    Returns
    -------
    dict[int, pd.DataFrame]
        Keyed by player_id.
    """
    from src.data.db import read_sql

    if not player_ids:
        return {}

    id_list = ", ".join(str(int(p)) for p in set(player_ids))
    query = f"""
    SELECT *
    FROM production.fact_player_form_rolling
    WHERE player_id IN ({id_list})
      AND player_role = :role
      AND season = :season
    ORDER BY player_id, game_date
    """
    df = read_sql(query, {"role": player_role, "season": season})
    if df.empty:
        return {}

    df["game_date"] = pd.to_datetime(df["game_date"])
    result: dict[int, pd.DataFrame] = {}
    for pid, grp in df.groupby("player_id"):
        result[int(pid)] = grp.reset_index(drop=True)
    return result


def lookup_prior_form(
    rolling_data: dict[int, pd.DataFrame],
    player_id: int,
    game_date: str,
) -> pd.Series | None:
    """Get the most recent rolling row BEFORE a specific game date."""
    grp = rolling_data.get(player_id)
    if grp is None or grp.empty:
        return None
    target = pd.Timestamp(game_date)
    prior = grp[grp["game_date"] < target]
    if prior.empty:
        return None
    return prior.iloc[-1]


def load_hard_hit_for_backtest(
    batter_ids: list[int],
    season: int,
) -> dict[int, pd.DataFrame]:
    """Load per-game hard-hit data for rolling computation per game date.

    Returns per-batter DataFrames sorted by game_date so we can compute
    the trailing 15-game hard-hit% as of any target date.

    Returns
    -------
    dict[int, pd.DataFrame]
        batter_id -> DataFrame with columns [game_date, bip, hh].
    """
    from src.data.db import read_sql

    if not batter_ids:
        return {}

    id_list = ", ".join(str(int(b)) for b in set(batter_ids))
    query = f"""
    SELECT
        fp.batter_id, dg.game_date,
        COUNT(*)                                   AS bip,
        COUNT(*) FILTER (WHERE bb.hard_hit = TRUE) AS hh
    FROM production.sat_batted_balls bb
    JOIN production.fact_pa fp ON bb.pa_id = fp.pa_id
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    WHERE fp.batter_id IN ({id_list})
      AND dg.game_type = 'R'
      AND dg.season = :season
    GROUP BY fp.batter_id, dg.game_date
    ORDER BY fp.batter_id, dg.game_date
    """
    df = read_sql(query, {"season": season})
    if df.empty:
        return {}

    df["game_date"] = pd.to_datetime(df["game_date"])
    result: dict[int, pd.DataFrame] = {}
    for bid, grp in df.groupby("batter_id"):
        result[int(bid)] = grp.reset_index(drop=True)
    return result


def lookup_prior_hard_hit(
    hh_data: dict[int, pd.DataFrame],
    batter_id: int,
    game_date: str,
    window: int = 15,
) -> tuple[float | None, int]:
    """Get trailing hard-hit% from the 15 games BEFORE a target date.

    Returns
    -------
    tuple[float | None, int]
        (hard_hit_pct, bip_count). None if no data.
    """
    grp = hh_data.get(batter_id)
    if grp is None or grp.empty:
        return None, 0
    target = pd.Timestamp(game_date)
    prior = grp[grp["game_date"] < target].tail(window)
    if prior.empty:
        return None, 0
    total_bip = int(prior["bip"].sum())
    total_hh = int(prior["hh"].sum())
    if total_bip == 0:
        return None, 0
    return total_hh / total_bip, total_bip


# ---------------------------------------------------------------------------
# Pitcher A/B backtest
# ---------------------------------------------------------------------------
def run_pitcher_ab_backtest(
    season: int = 2025,
    max_games: int = 500,
    n_sims: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run pitcher game sim with and without form lifts.

    Returns DataFrame with columns for both predictions and actuals.
    """
    from src.data.db import read_sql
    from src.models.game_sim.exit_model import ExitModel
    from src.models.game_sim.pa_outcome_model import GameContext
    from src.models.game_sim.simulator import simulate_game, compute_stamina_offset
    from src.models.game_sim.form_model import compute_pitcher_form_lifts, PitcherFormLifts

    logger.info("=" * 60)
    logger.info("Pitcher A/B form lift backtest: season=%d, max_games=%d", season, max_games)

    # --- Load test games (starters, June+ so 15g windows are full) ---
    games_query = f"""
    SELECT fpg.player_id AS pitcher_id, fpg.game_pk, fpg.game_date,
           fpg.pit_k AS actual_k, fpg.pit_bb AS actual_bb,
           fpg.pit_hr AS actual_hr, fpg.pit_h AS actual_h,
           fpg.pit_bf AS actual_bf, fpg.pit_ip AS actual_ip
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
    WHERE fpg.player_role = 'pitcher'
      AND fpg.pit_is_starter = TRUE
      AND fpg.pit_bf >= 10
      AND dg.season = {season}
      AND dg.game_type = 'R'
      AND fpg.game_date >= '{season}-06-01'
    ORDER BY RANDOM()
    LIMIT {max_games}
    """
    games = read_sql(games_query)
    if games.empty:
        logger.warning("No games found for season %d", season)
        return pd.DataFrame()
    logger.info("Loaded %d pitcher-game starts", len(games))

    # --- Load precomputed samples (from dashboard snapshots) ---
    DASHBOARD_DIR = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard")
    SNAPSHOT_DIR = DASHBOARD_DIR / "snapshots"

    def _load_npz(name: str) -> dict[str, np.ndarray]:
        for d in [SNAPSHOT_DIR, DASHBOARD_DIR]:
            path = d / f"{name}.npz"
            if path.exists():
                data = np.load(path)
                return {k: data[k] for k in data.files}
        return {}

    k_npz = _load_npz("pitcher_k_samples")
    bb_npz = _load_npz("pitcher_bb_samples")
    hr_npz = _load_npz("pitcher_hr_samples")
    logger.info("Loaded posterior samples: K=%d, BB=%d, HR=%d",
                len(k_npz), len(bb_npz), len(hr_npz))

    # --- Load exit model ---
    exit_model = ExitModel()
    exit_model_path = DASHBOARD_DIR / "exit_model.pkl"
    if exit_model_path.exists():
        exit_model.load(exit_model_path)

    # --- Load rolling form data ---
    pitcher_ids = games["pitcher_id"].unique().tolist()
    rolling_data = load_rolling_form_all(pitcher_ids, "pitcher", season)
    logger.info("Rolling form loaded for %d pitchers", len(rolling_data))

    # --- Run A/B simulations ---
    results = []
    n_skipped = 0
    for _, row in games.iterrows():
        pid = int(row["pitcher_id"])
        gpk = int(row["game_pk"])
        gdate = str(row["game_date"])
        pid_str = str(pid)

        k_samp = k_npz.get(pid_str)
        bb_samp = bb_npz.get(pid_str)
        hr_samp = hr_npz.get(pid_str)
        if k_samp is None:
            n_skipped += 1
            continue

        rng_fb = np.random.default_rng(random_seed + pid)
        if bb_samp is None:
            bb_samp = rng_fb.beta(8, 90, size=len(k_samp))
        if hr_samp is None:
            hr_samp = rng_fb.beta(3, 95, size=len(k_samp))

        # Compute form lift from the game BEFORE this one
        rolling_row = lookup_prior_form(rolling_data, pid, gdate)
        if rolling_row is not None:
            pit_form = compute_pitcher_form_lifts(rolling_row)
        else:
            pit_form = PitcherFormLifts()

        # Shared sim params (minimal — no matchup/context for clean A/B)
        shared = dict(
            pitcher_k_rate_samples=k_samp,
            pitcher_bb_rate_samples=bb_samp,
            pitcher_hr_rate_samples=hr_samp,
            lineup_matchup_lifts={"k": np.zeros(9), "bb": np.zeros(9), "hr": np.zeros(9)},
            tto_lifts={"k": np.zeros(3), "bb": np.zeros(3), "hr": np.zeros(3)},
            pitcher_ppa_adj=0.0,
            batter_ppa_adjs=np.zeros(9),
            exit_model=exit_model,
            n_sims=n_sims,
            random_seed=random_seed + gpk % 10000,
        )

        try:
            # A: No form lift (baseline)
            sim_a = simulate_game(**shared, game_context=GameContext())
            # B: With form lift
            sim_b = simulate_game(
                **shared, game_context=GameContext(form_bb_lift=pit_form.bb_lift),
            )
        except Exception as e:
            logger.debug("Sim failed pid=%d gpk=%d: %s", pid, gpk, e)
            n_skipped += 1
            continue

        sa = sim_a.summary()
        sb = sim_b.summary()

        results.append({
            "game_pk": gpk,
            "pitcher_id": pid,
            "form_bb_lift": pit_form.bb_lift,
            # Baseline predictions
            "pred_k_base": sa["k"]["mean"],
            "pred_bb_base": sa["bb"]["mean"],
            "pred_hr_base": sa["hr"]["mean"],
            "pred_h_base": sa["h"]["mean"],
            # Form lift predictions
            "pred_k_form": sb["k"]["mean"],
            "pred_bb_form": sb["bb"]["mean"],
            "pred_hr_form": sb["hr"]["mean"],
            "pred_h_form": sb["h"]["mean"],
            # Actuals
            "actual_k": int(row["actual_k"]),
            "actual_bb": int(row["actual_bb"]),
            "actual_hr": int(row["actual_hr"]),
            "actual_h": int(row["actual_h"]),
            "actual_bf": int(row["actual_bf"]),
        })

    logger.info("Completed: %d games, %d skipped", len(results), n_skipped)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Batter A/B backtest
# ---------------------------------------------------------------------------
def run_batter_ab_backtest(
    season: int = 2025,
    max_games: int = 500,
    n_sims: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run batter game sim with and without form lifts."""
    from src.data.db import read_sql
    from src.models.game_sim.batter_simulator import simulate_batter_game
    from src.models.game_sim.form_model import compute_batter_form_lifts, BatterFormLifts

    logger.info("=" * 60)
    logger.info("Batter A/B form lift backtest: season=%d, max_games=%d", season, max_games)

    # --- Load test games (June+ so 15g windows are full) ---
    games_query = f"""
    SELECT fpg.player_id AS batter_id, fpg.game_pk, fpg.game_date,
           fpg.bat_k AS actual_k, fpg.bat_bb AS actual_bb,
           fpg.bat_hr AS actual_hr, fpg.bat_h AS actual_h,
           fpg.bat_pa AS actual_pa, fpg.bat_tb AS actual_tb
    FROM production.fact_player_game_mlb fpg
    JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
    WHERE fpg.player_role = 'batter'
      AND fpg.bat_pa >= 3
      AND dg.season = {season}
      AND dg.game_type = 'R'
      AND fpg.game_date >= '{season}-06-01'
    ORDER BY RANDOM()
    LIMIT {max_games}
    """
    games = read_sql(games_query)
    if games.empty:
        logger.warning("No games found")
        return pd.DataFrame()
    logger.info("Loaded %d batter-game rows", len(games))

    # --- Load precomputed samples ---
    DASHBOARD_DIR = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard")
    SNAPSHOT_DIR = DASHBOARD_DIR / "snapshots"

    def _load_npz(name: str) -> dict[str, np.ndarray]:
        for d in [SNAPSHOT_DIR, DASHBOARD_DIR]:
            path = d / f"{name}.npz"
            if path.exists():
                data = np.load(path)
                return {k: data[k] for k in data.files}
        return {}

    k_npz = _load_npz("hitter_k_samples")
    bb_npz = _load_npz("hitter_bb_samples")
    hr_npz = _load_npz("hitter_hr_samples")
    logger.info("Hitter posterior samples: K=%d, BB=%d, HR=%d",
                len(k_npz), len(bb_npz), len(hr_npz))

    # --- Load rolling form data ---
    batter_ids = games["batter_id"].unique().tolist()
    rolling_data = load_rolling_form_all(batter_ids, "batter", season)
    hh_data = load_hard_hit_for_backtest(batter_ids, season)
    logger.info("Rolling form: %d batters, hard-hit: %d", len(rolling_data), len(hh_data))

    # --- Run A/B simulations ---
    results = []
    n_skipped = 0
    for _, row in games.iterrows():
        bid = int(row["batter_id"])
        gpk = int(row["game_pk"])
        gdate = str(row["game_date"])
        bid_str = str(bid)

        k_samp = k_npz.get(bid_str)
        bb_samp = bb_npz.get(bid_str)
        hr_samp = hr_npz.get(bid_str)
        if k_samp is None or bb_samp is None or hr_samp is None:
            n_skipped += 1
            continue

        # Compute form lift from the game BEFORE this one
        rolling_row = lookup_prior_form(rolling_data, bid, gdate)
        if rolling_row is not None:
            hh_pct, hh_bip = lookup_prior_hard_hit(hh_data, bid, gdate)
            bat_form = compute_batter_form_lifts(
                rolling_row, hard_hit_pct=hh_pct, hard_hit_bip=hh_bip,
            )
        else:
            bat_form = BatterFormLifts()

        shared = dict(
            batter_k_rate_samples=k_samp,
            batter_bb_rate_samples=bb_samp,
            batter_hr_rate_samples=hr_samp,
            batting_order=4,  # assume mid-order for simplicity
            starter_k_rate=0.226,
            starter_bb_rate=0.082,
            starter_hr_rate=0.031,
            starter_bf_mu=24.0,
            starter_bf_sigma=3.0,
            n_sims=n_sims,
            random_seed=random_seed + gpk % 10000 + bid % 1000,
        )

        try:
            sim_a = simulate_batter_game(**shared)
            sim_b = simulate_batter_game(
                **shared,
                form_k_lift=bat_form.k_lift,
                form_bb_lift=bat_form.bb_lift,
                form_hr_lift=bat_form.hr_lift + bat_form.hh_lift,
            )
        except Exception as e:
            logger.debug("Sim failed bid=%d gpk=%d: %s", bid, gpk, e)
            n_skipped += 1
            continue

        sa = sim_a.summary()
        sb = sim_b.summary()

        results.append({
            "game_pk": gpk,
            "batter_id": bid,
            "form_k_lift": bat_form.k_lift,
            "form_bb_lift": bat_form.bb_lift,
            "form_hr_lift": bat_form.hr_lift + bat_form.hh_lift,
            # Baseline
            "pred_k_base": sa["k"]["mean"],
            "pred_bb_base": sa["bb"]["mean"],
            "pred_hr_base": sa["hr"]["mean"],
            "pred_h_base": sa["h"]["mean"],
            "pred_tb_base": sa["tb"]["mean"],
            # Form
            "pred_k_form": sb["k"]["mean"],
            "pred_bb_form": sb["bb"]["mean"],
            "pred_hr_form": sb["hr"]["mean"],
            "pred_h_form": sb["h"]["mean"],
            "pred_tb_form": sb["tb"]["mean"],
            # Actuals
            "actual_k": int(row["actual_k"]),
            "actual_bb": int(row["actual_bb"]),
            "actual_hr": int(row["actual_hr"]),
            "actual_h": int(row["actual_h"]),
            "actual_pa": int(row["actual_pa"]),
            "actual_tb": int(row["actual_tb"]),
        })

    logger.info("Completed: %d games, %d skipped", len(results), n_skipped)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_ab_metrics(df: pd.DataFrame, stats: list[str], label: str) -> pd.DataFrame:
    """Compute bias and correlation for baseline vs form lift predictions."""
    rows = []
    for stat in stats:
        actual = df[f"actual_{stat}"].values
        pred_base = df[f"pred_{stat}_base"].values
        pred_form = df[f"pred_{stat}_form"].values

        bias_base = float(np.mean(pred_base - actual))
        bias_form = float(np.mean(pred_form - actual))
        corr_base = float(np.corrcoef(actual, pred_base)[0, 1]) if len(actual) > 2 else np.nan
        corr_form = float(np.corrcoef(actual, pred_form)[0, 1]) if len(actual) > 2 else np.nan

        rows.append({
            "stat": stat,
            "bias_baseline": round(bias_base, 4),
            "bias_form": round(bias_form, 4),
            "corr_baseline": round(corr_base, 4),
            "corr_form": round(corr_form, 4),
            "n": len(df),
        })

    result = pd.DataFrame(rows)
    logger.info("\n%s A/B Results:\n%s", label, result.to_string(index=False))
    return result


def analyze_by_lift_magnitude(
    df: pd.DataFrame, stat: str, lift_col: str, label: str,
) -> None:
    """Show metrics stratified by form lift magnitude."""
    df = df.copy()
    df["abs_lift"] = df[lift_col].abs()

    # Split into no-lift vs has-lift
    has_lift = df[df["abs_lift"] >= 0.001]

    if has_lift.empty:
        logger.info("  No games with meaningful %s lift", stat)
        return

    # For games WITH a lift, compare baseline vs form bias
    actual = has_lift[f"actual_{stat}"].values
    pred_b = has_lift[f"pred_{stat}_base"].values
    pred_f = has_lift[f"pred_{stat}_form"].values

    bias_b = float(np.mean(pred_b - actual))
    bias_f = float(np.mean(pred_f - actual))
    logger.info(
        "  %s | %s lift games (n=%d): bias base=%+.4f, form=%+.4f",
        label, stat, len(has_lift), bias_b, bias_f,
    )

    # Stratify by lift direction
    for direction, mask_fn in [
        ("positive", lambda x: x > 0.001),
        ("negative", lambda x: x < -0.001),
    ]:
        subset = has_lift[mask_fn(has_lift[lift_col])]
        if len(subset) < 20:
            continue
        a = subset[f"actual_{stat}"].values
        pb = subset[f"pred_{stat}_base"].values
        pf = subset[f"pred_{stat}_form"].values
        logger.info(
            "    %s lift (n=%d): bias base=%+.4f, form=%+.4f",
            direction, len(subset),
            float(np.mean(pb - a)),
            float(np.mean(pf - a)),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="A/B backtest for form lifts")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--mode", choices=["both", "pitcher", "batter"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    if args.mode in ("both", "pitcher"):
        pit_df = run_pitcher_ab_backtest(
            season=args.season,
            max_games=args.max_games,
            n_sims=args.n_sims,
            random_seed=args.seed,
        )
        if not pit_df.empty:
            pit_metrics = compute_ab_metrics(
                pit_df, ["k", "bb", "hr", "h"], "Pitcher",
            )
            analyze_by_lift_magnitude(pit_df, "bb", "form_bb_lift", "Pitcher")
            pit_df.to_csv(out_dir / "form_lift_pitcher_ab.csv", index=False)
            pit_metrics.to_csv(out_dir / "form_lift_pitcher_metrics.csv", index=False)

    if args.mode in ("both", "batter"):
        bat_df = run_batter_ab_backtest(
            season=args.season,
            max_games=args.max_games,
            n_sims=args.n_sims,
            random_seed=args.seed,
        )
        if not bat_df.empty:
            bat_metrics = compute_ab_metrics(
                bat_df, ["k", "bb", "hr", "h", "tb"], "Batter",
            )
            analyze_by_lift_magnitude(bat_df, "k", "form_k_lift", "Batter")
            analyze_by_lift_magnitude(bat_df, "hr", "form_hr_lift", "Batter")
            bat_df.to_csv(out_dir / "form_lift_batter_ab.csv", index=False)
            bat_metrics.to_csv(out_dir / "form_lift_batter_metrics.csv", index=False)


if __name__ == "__main__":
    main()
