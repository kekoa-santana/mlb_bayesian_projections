"""A/B comparison: old hand-picked dampeners vs fitted Path A slopes.

The full game sim backtest retrains MCMC per fold (hours). For this A/B we
only need to isolate the effect of the matchup slope on game-level K and BB
count accuracy, holding everything else constant. So we:

1. Use 2023 pitcher and hitter season rates as point estimates (fast, no MCMC).
2. Use 2023 profiles to compute raw K and BB matchup lifts per lineup slot
   from score_matchup / score_matchup_bb.
3. For every 2024 starter game, compute the PA-weighted expected K and BB
   count under BOTH slope regimes:
       A: K * 0.55,  BB * 0.40  (current hand-picked)
       B: K * 0.787, BB * 0.766 (fitted Path A)
4. Compare vs actual outcomes via:
       - Mean signed bias (are predictions systematically off?)
       - Poisson log-likelihood of actual count under predicted mean
       - Brier score at standard prop lines (K >= 5.5, BB >= 2.5)
       - Distribution of predicted - actual residuals
5. Report which regime calibrates better on 2024 starter games.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db import read_sql
from src.data.feature_eng import get_hitter_vulnerability, get_pitcher_arsenal
from src.data.league_baselines import get_baselines_dict
from src.models.matchup import score_matchup, score_matchup_bb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_SEASON = 2024
PROFILE_SEASON = 2023
MIN_BF = 15
N_GAMES_SAMPLE = 600  # ~600 games is plenty for a calibration A/B
RNG_SEED = 42

SLOPES_A = {"k": 0.55, "bb": 0.40}
SLOPES_B = {"k": 0.7870, "bb": 0.7661}


def _logit(p: float | np.ndarray) -> float | np.ndarray:
    p_arr = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p_arr / (1.0 - p_arr))


def _inv_logit(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_2024_starter_games() -> pd.DataFrame:
    q = f"""
        WITH pitcher_game AS (
            SELECT
                pa.game_pk,
                pa.pitcher_id,
                g.game_date,
                MAX(pa.pitcher_pa_number) AS bf,
                BOOL_OR(pa.pitcher_pa_number = 1) AS faced_first
            FROM production.fact_pa pa
            JOIN production.dim_game g USING (game_pk)
            WHERE g.season = {HOLDOUT_SEASON}
              AND g.game_type = 'R'
            GROUP BY pa.game_pk, pa.pitcher_id, g.game_date
        )
        SELECT
            game_pk,
            pitcher_id,
            game_date,
            bf AS actual_bf
        FROM pitcher_game
        WHERE faced_first AND bf >= {MIN_BF}
    """
    df = read_sql(q)
    return df


def load_game_pa_details(game_pks: list[int]) -> pd.DataFrame:
    """Load PA-level details for starter-vs-batter matchups in target games."""
    placeholders = ",".join(str(int(g)) for g in game_pks)
    q = f"""
        SELECT
            pa.game_pk,
            pa.pitcher_id,
            pa.batter_id,
            pa.events
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE pa.game_pk IN ({placeholders})
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL
          AND pa.events <> ''
    """
    return read_sql(q)


def load_2023_rates() -> tuple[pd.DataFrame, pd.DataFrame]:
    q_p = f"""
        SELECT
            pa.pitcher_id,
            COUNT(*) AS p_bf,
            SUM(CASE WHEN pa.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END)::float / COUNT(*) AS p_k_rate,
            SUM(CASE WHEN pa.events = 'walk' THEN 1 ELSE 0 END)::float / COUNT(*) AS p_bb_rate
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.season = {PROFILE_SEASON}
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL AND pa.events <> '' AND pa.events <> 'intent_walk'
        GROUP BY pa.pitcher_id
        HAVING COUNT(*) >= 100
    """
    q_b = f"""
        SELECT
            pa.batter_id,
            COUNT(*) AS b_pa,
            SUM(CASE WHEN pa.events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END)::float / COUNT(*) AS b_k_rate,
            SUM(CASE WHEN pa.events = 'walk' THEN 1 ELSE 0 END)::float / COUNT(*) AS b_bb_rate
        FROM production.fact_pa pa
        JOIN production.dim_game g USING (game_pk)
        WHERE g.season = {PROFILE_SEASON}
          AND g.game_type = 'R'
          AND pa.events IS NOT NULL AND pa.events <> '' AND pa.events <> 'intent_walk'
        GROUP BY pa.batter_id
        HAVING COUNT(*) >= 50
    """
    return read_sql(q_p), read_sql(q_b)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    logger.info("Loading 2024 starter games...")
    games = load_2024_starter_games()
    logger.info("  %d starter games with BF >= %d", len(games), MIN_BF)

    if len(games) > N_GAMES_SAMPLE:
        games = games.sample(N_GAMES_SAMPLE, random_state=RNG_SEED).reset_index(drop=True)
    logger.info("  sampled %d games for A/B", len(games))

    logger.info("Loading PA details for sampled games...")
    pas = load_game_pa_details(games["game_pk"].tolist())
    pas["is_k"] = pas["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    pas["is_bb"] = (pas["events"] == "walk").astype(int)

    # Actual game totals (only PAs where starter faced batter)
    pitcher_pas = pas.merge(games[["game_pk", "pitcher_id"]], on=["game_pk", "pitcher_id"])
    actual_totals = pitcher_pas.groupby("game_pk").agg(
        actual_k=("is_k", "sum"),
        actual_bb=("is_bb", "sum"),
        actual_bf=("is_k", "count"),
    ).reset_index()
    games = games.merge(actual_totals, on="game_pk", suffixes=("", "_recalc"))
    games["actual_bf"] = games["actual_bf_recalc"]
    games = games.drop(columns=["actual_bf_recalc"])

    logger.info("Loading 2023 profiles and baselines...")
    pitcher_arsenal = get_pitcher_arsenal(PROFILE_SEASON)
    hitter_vuln = get_hitter_vulnerability(PROFILE_SEASON)
    baselines_pt = get_baselines_dict(seasons=[PROFILE_SEASON], recency_weights="equal")

    logger.info("Loading 2023 pitcher/batter season rates...")
    p_rates, b_rates = load_2023_rates()

    # Per-game loop: compute expected K and BB under both regimes
    rows = []
    skipped = 0
    for i, g in enumerate(games.itertuples(index=False)):
        if i % 100 == 0 and i > 0:
            logger.info("  Processed %d / %d games (skipped %d)",
                        i, len(games), skipped)

        pid = int(g.pitcher_id)
        game_pas = pitcher_pas[pitcher_pas["game_pk"] == g.game_pk]
        if len(game_pas) == 0:
            skipped += 1
            continue

        p_row = p_rates[p_rates["pitcher_id"] == pid]
        if p_row.empty:
            skipped += 1
            continue
        p_k_rate = float(p_row["p_k_rate"].iloc[0])
        p_bb_rate = float(p_row["p_bb_rate"].iloc[0])

        # Per-batter expected rates
        batter_ids = game_pas["batter_id"].values
        k_lifts_raw = []
        bb_lifts_raw = []
        for bid in batter_ids:
            k_res = score_matchup(
                pitcher_id=pid, batter_id=int(bid),
                pitcher_arsenal=pitcher_arsenal, hitter_vuln=hitter_vuln,
                baselines_pt=baselines_pt,
            )
            bb_res = score_matchup_bb(
                pitcher_id=pid, batter_id=int(bid),
                pitcher_arsenal=pitcher_arsenal, hitter_vuln=hitter_vuln,
                baselines_pt=baselines_pt,
            )
            k_lift = k_res.get("matchup_k_logit_lift", 0.0)
            bb_lift = bb_res.get("matchup_bb_logit_lift", 0.0)
            if isinstance(k_lift, float) and np.isnan(k_lift):
                k_lift = 0.0
            if isinstance(bb_lift, float) and np.isnan(bb_lift):
                bb_lift = 0.0
            k_lifts_raw.append(float(k_lift))
            bb_lifts_raw.append(float(bb_lift))

        k_lifts_raw_arr = np.array(k_lifts_raw)
        bb_lifts_raw_arr = np.array(bb_lifts_raw)

        p_k_logit = _logit(p_k_rate)
        p_bb_logit = _logit(p_bb_rate)

        # Regime A
        exp_k_A = float(_inv_logit(p_k_logit + SLOPES_A["k"] * k_lifts_raw_arr).sum())
        exp_bb_A = float(_inv_logit(p_bb_logit + SLOPES_A["bb"] * bb_lifts_raw_arr).sum())

        # Regime B
        exp_k_B = float(_inv_logit(p_k_logit + SLOPES_B["k"] * k_lifts_raw_arr).sum())
        exp_bb_B = float(_inv_logit(p_bb_logit + SLOPES_B["bb"] * bb_lifts_raw_arr).sum())

        rows.append({
            "game_pk": int(g.game_pk),
            "pitcher_id": pid,
            "bf": len(batter_ids),
            "actual_k": int(g.actual_k),
            "actual_bb": int(g.actual_bb),
            "exp_k_A": exp_k_A,
            "exp_bb_A": exp_bb_A,
            "exp_k_B": exp_k_B,
            "exp_bb_B": exp_bb_B,
            "p_k_rate": p_k_rate,
            "p_bb_rate": p_bb_rate,
        })

    df = pd.DataFrame(rows)
    logger.info("\nScored %d games (skipped %d)", len(df), skipped)

    # Summary metrics
    print("\n" + "=" * 72)
    print("A/B MATCHUP SLOPE COMPARISON ON 2024 STARTER GAMES")
    print("=" * 72)
    print(f"N games: {len(df)}")
    print(f"Regime A: K x {SLOPES_A['k']}, BB x {SLOPES_A['bb']} (hand-picked)")
    print(f"Regime B: K x {SLOPES_B['k']}, BB x {SLOPES_B['bb']} (Path A fit)")

    for stat in ("k", "bb"):
        actual_col = f"actual_{stat}"
        col_A = f"exp_{stat}_A"
        col_B = f"exp_{stat}_B"

        actual = df[actual_col].values
        pred_A = df[col_A].values
        pred_B = df[col_B].values

        bias_A = float(np.mean(pred_A - actual))
        bias_B = float(np.mean(pred_B - actual))
        std_res_A = float(np.std(pred_A - actual))
        std_res_B = float(np.std(pred_B - actual))

        # Poisson log-likelihood of actual under predicted mean
        ll_pois_A = float(poisson.logpmf(actual, np.clip(pred_A, 0.01, None)).mean())
        ll_pois_B = float(poisson.logpmf(actual, np.clip(pred_B, 0.01, None)).mean())

        # Calibration: slope of actual on predicted
        cov_A = np.cov(pred_A, actual, ddof=0)
        cov_B = np.cov(pred_B, actual, ddof=0)
        calib_slope_A = cov_A[0, 1] / cov_A[0, 0] if cov_A[0, 0] > 0 else float("nan")
        calib_slope_B = cov_B[0, 1] / cov_B[0, 0] if cov_B[0, 0] > 0 else float("nan")

        # Brier at standard props
        prop_line = 5.5 if stat == "k" else 2.5
        y_over = (actual > prop_line).astype(int)
        # Approximate P(stat > line) with Poisson SF
        p_over_A = 1.0 - poisson.cdf(int(prop_line), np.clip(pred_A, 0.01, None))
        p_over_B = 1.0 - poisson.cdf(int(prop_line), np.clip(pred_B, 0.01, None))
        brier_A = brier_score_loss(y_over, p_over_A)
        brier_B = brier_score_loss(y_over, p_over_B)

        print(f"\n{stat.upper()}:")
        print(f"  {'':<20} {'A (0.55/0.40)':>16} {'B (fit)':>16} {'delta':>12}")
        print(f"  {'mean pred':<20} {pred_A.mean():>16.3f} {pred_B.mean():>16.3f} {pred_B.mean() - pred_A.mean():>+12.4f}")
        print(f"  {'mean actual':<20} {actual.mean():>16.3f} {actual.mean():>16.3f}")
        print(f"  {'bias (pred-actual)':<20} {bias_A:>+16.4f} {bias_B:>+16.4f} {bias_B - bias_A:>+12.4f}")
        print(f"  {'residual std':<20} {std_res_A:>16.4f} {std_res_B:>16.4f} {std_res_B - std_res_A:>+12.4f}")
        print(f"  {'Poisson log-lik':<20} {ll_pois_A:>16.5f} {ll_pois_B:>16.5f} {ll_pois_B - ll_pois_A:>+12.5f}")
        print(f"  {'calibration slope':<20} {calib_slope_A:>16.4f} {calib_slope_B:>16.4f} {calib_slope_B - calib_slope_A:>+12.4f}")
        print(f"  {f'Brier >{prop_line}':<20} {brier_A:>16.5f} {brier_B:>16.5f} {brier_B - brier_A:>+12.5f}")

    out = Path("outputs/ab_matchup_slopes.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nSaved per-game A/B results to {out}")


if __name__ == "__main__":
    main()
