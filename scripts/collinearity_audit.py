"""
Phase 0C: Feature Collinearity Audit

Computes pairwise correlations between proposed feature signals to identify
double-counting risks before adding them to the model.

Pairs tested:
1. Lineup K-proneness vs opposing pitcher K%
2. Platoon splits vs BvP history (K rate)
3. Weather temperature vs park HR factor
4. Umpire K tendency vs catcher framing (called strike rate)

Outputs:
  - outputs/collinearity_audit.csv   — summary table with Pearson r, Spearman rho, VIF, N
  - Console report with recommendations
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _vif(r: float) -> float:
    """Variance Inflation Factor from bivariate Pearson r."""
    r2 = r ** 2
    if r2 >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - r2)


# ===================================================================
# Pair 1: Lineup K-proneness vs Opposing Pitcher K%
# ===================================================================
def pair1_lineup_vs_pitcher_k() -> dict:
    """
    For each game-team, compute:
      - avg season K% of the lineup hitters (from fact_platoon_splits for batters)
      - the opposing starting pitcher's season K% (from fact_player_game_mlb pitcher stats)
    Then correlate across all game-team observations.
    """
    logger.info("Pair 1: Lineup K-proneness vs Opposing Pitcher K%")

    # Step 1: Get hitter season K% (overall, not platoon-specific)
    hitter_k = read_sql("""
        SELECT player_id, season,
               SUM(k)::float / NULLIF(SUM(pa), 0) AS hitter_k_pct
        FROM production.fact_platoon_splits
        WHERE player_role = 'batter'
          AND pa >= 50
        GROUP BY player_id, season
    """)

    # Step 2: Get lineup starters per game-team
    lineup = read_sql("""
        SELECT fl.game_pk, fl.team_id, fl.player_id, dg.season
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        WHERE fl.is_starter = true
          AND fl.position != 'P'
          AND dg.game_type = 'R'
    """)

    # Merge hitter K% onto lineup
    lineup = lineup.merge(
        hitter_k, on=["player_id", "season"], how="inner"
    )

    # Avg K% per game-team
    lineup_avg = (
        lineup.groupby(["game_pk", "team_id", "season"])["hitter_k_pct"]
        .mean()
        .reset_index()
        .rename(columns={"hitter_k_pct": "lineup_k_pct"})
    )

    # Step 3: Get starting pitcher K% per season
    pitcher_season_k = read_sql("""
        SELECT player_id AS pitcher_id, season,
               SUM(pit_k)::float / NULLIF(SUM(pit_bf), 0) AS pitcher_k_pct
        FROM production.fact_player_game_mlb
        WHERE pit_is_starter = true
          AND pit_bf > 0
        GROUP BY player_id, season
    """)

    # Step 4: Get the opposing starter per game-team
    # For a given game_pk + team_id (batting team), the opposing pitcher
    # is the starter on the OTHER team.
    starters = read_sql("""
        SELECT fpg.game_pk, fpg.team_id, fpg.player_id AS pitcher_id, fpg.season
        FROM production.fact_player_game_mlb fpg
        JOIN production.dim_game dg ON fpg.game_pk = dg.game_pk
        WHERE fpg.pit_is_starter = true
          AND dg.game_type = 'R'
    """)

    # For each game, map batting team to opposing pitcher
    # The opposing pitcher is the one whose team_id != the batting team's team_id
    games = read_sql("""
        SELECT game_pk, home_team_id, away_team_id
        FROM production.dim_game
        WHERE game_type = 'R'
    """)

    # Merge starters with games to find which team the pitcher is on
    starters = starters.merge(games, on="game_pk", how="inner")

    # Build opposing pitcher lookup: for batting team X in game G,
    # the opposing pitcher is the starter whose team != X
    rows = []
    for _, row in starters.iterrows():
        # This pitcher's team_id; the opposing batting teams are the other side
        opp_batting_team = (
            row["away_team_id"]
            if row["team_id"] == row["home_team_id"]
            else row["home_team_id"]
        )
        rows.append({
            "game_pk": row["game_pk"],
            "batting_team_id": opp_batting_team,
            "opp_pitcher_id": row["pitcher_id"],
            "season": row["season"],
        })
    opp_pitcher = pd.DataFrame(rows)

    # Merge pitcher season K%
    opp_pitcher = opp_pitcher.merge(
        pitcher_season_k,
        left_on=["opp_pitcher_id", "season"],
        right_on=["pitcher_id", "season"],
        how="inner",
    )

    # Join with lineup avg K%
    merged = lineup_avg.merge(
        opp_pitcher,
        left_on=["game_pk", "team_id"],
        right_on=["game_pk", "batting_team_id"],
        how="inner",
        suffixes=("_lu", "_opp"),
    )

    merged = merged.dropna(subset=["lineup_k_pct", "pitcher_k_pct"])
    n = len(merged)
    logger.info("  N = %d game-team observations", n)

    if n < 30:
        logger.warning("  Too few observations for reliable correlation")
        return {"pair": "lineup_k_pct vs pitcher_k_pct", "n": n,
                "pearson_r": np.nan, "spearman_rho": np.nan, "vif": np.nan}

    pr, _ = stats.pearsonr(merged["lineup_k_pct"], merged["pitcher_k_pct"])
    sr, _ = stats.spearmanr(merged["lineup_k_pct"], merged["pitcher_k_pct"])
    vif = _vif(pr)

    logger.info("  Pearson r  = %.4f", pr)
    logger.info("  Spearman ρ = %.4f", sr)
    logger.info("  VIF        = %.3f", vif)

    return {
        "pair": "lineup_k_pct vs pitcher_k_pct",
        "n": n,
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
        "vif": round(vif, 3),
    }


# ===================================================================
# Pair 2: Platoon Splits vs BvP History (K rate)
# ===================================================================
def pair2_platoon_vs_bvp() -> dict:
    """
    For matchups with >= 15 career PA in fact_matchup_history, compare:
      - BvP K rate (k / pa from fact_matchup_history)
      - Platoon-predicted K rate (from fact_platoon_splits for that batter
        vs the pitcher's throwing hand)
    """
    logger.info("Pair 2: Platoon K% vs BvP K%")

    # BvP matchups with 15+ PA
    bvp = read_sql("""
        SELECT batter_id, pitcher_id, pa, k,
               k::float / NULLIF(pa, 0) AS bvp_k_rate
        FROM production.fact_matchup_history
        WHERE pa >= 15
    """)

    # Get pitcher throwing hand from dim_player
    pitcher_hand = read_sql("""
        SELECT player_id, pitch_hand
        FROM production.dim_player
        WHERE pitch_hand IS NOT NULL
    """)

    bvp = bvp.merge(
        pitcher_hand,
        left_on="pitcher_id",
        right_on="player_id",
        how="inner",
    )

    # Map pitcher pitch_hand to platoon_side for the batter:
    # vs RHP -> platoon_side = 'vRH', vs LHP -> platoon_side = 'vLH'
    bvp["platoon_side"] = bvp["pitch_hand"].map({"R": "vRH", "L": "vLH"})
    bvp = bvp.dropna(subset=["platoon_side"])

    # Get batter platoon K% — use the most recent season with enough data
    # We'll take a weighted avg across seasons
    platoon = read_sql("""
        SELECT player_id, platoon_side,
               SUM(k)::float / NULLIF(SUM(pa), 0) AS platoon_k_rate
        FROM production.fact_platoon_splits
        WHERE player_role = 'batter'
          AND pa >= 30
        GROUP BY player_id, platoon_side
    """)

    merged = bvp.merge(
        platoon,
        left_on=["batter_id", "platoon_side"],
        right_on=["player_id", "platoon_side"],
        how="inner",
    )

    merged = merged.dropna(subset=["bvp_k_rate", "platoon_k_rate"])
    n = len(merged)
    logger.info("  N = %d matchups with 15+ PA", n)

    if n < 30:
        logger.warning("  Too few observations for reliable correlation")
        return {"pair": "platoon_k_rate vs bvp_k_rate", "n": n,
                "pearson_r": np.nan, "spearman_rho": np.nan, "vif": np.nan}

    pr, _ = stats.pearsonr(merged["platoon_k_rate"], merged["bvp_k_rate"])
    sr, _ = stats.spearmanr(merged["platoon_k_rate"], merged["bvp_k_rate"])
    vif = _vif(pr)

    logger.info("  Pearson r  = %.4f", pr)
    logger.info("  Spearman ρ = %.4f", sr)
    logger.info("  VIF        = %.3f", vif)

    return {
        "pair": "platoon_k_rate vs bvp_k_rate",
        "n": n,
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
        "vif": round(vif, 3),
    }


# ===================================================================
# Pair 3: Weather Temperature vs Park HR Factor
# ===================================================================
def pair3_weather_vs_park() -> dict:
    """
    For each game, compare:
      - Temperature-based HR multiplier (normalized to mean=1)
      - Park HR factor (hr_pf_3yr from dim_park_factor)
    Exclude dome games (controlled environment).
    """
    logger.info("Pair 3: Weather temp HR multiplier vs Park HR factor")

    df = read_sql("""
        SELECT
            dw.game_pk,
            dw.temperature,
            dw.is_dome,
            dpf.hr_pf_3yr,
            dg.season
        FROM production.dim_weather dw
        JOIN production.dim_game dg ON dw.game_pk = dg.game_pk
        JOIN production.dim_park_factor dpf
             ON dg.venue_id = dpf.venue_id
             AND dg.season = dpf.season
        WHERE dg.game_type = 'R'
          AND dw.is_dome = false
          AND dw.temperature IS NOT NULL
          AND dpf.hr_pf_3yr IS NOT NULL
    """)

    # Aggregate across batter_stand if needed (take avg park factor per venue-season)
    if "batter_stand" in df.columns:
        df = df.groupby(["game_pk", "temperature"]).agg(
            hr_pf_3yr=("hr_pf_3yr", "mean")
        ).reset_index()

    # Simple temperature HR multiplier: +1.5% per degree F above 72
    # (standard physics-based estimate from Alan Nathan's research)
    df["temp_hr_mult"] = 1.0 + 0.015 * (df["temperature"] - 72)

    df = df.dropna(subset=["temp_hr_mult", "hr_pf_3yr"])

    # Since dim_park_factor has per-batter_stand rows, average across stands
    df_dedup = (
        df.groupby("game_pk")
        .agg(temp_hr_mult=("temp_hr_mult", "first"),
             hr_pf_3yr=("hr_pf_3yr", "mean"))
        .reset_index()
    )

    n = len(df_dedup)
    logger.info("  N = %d outdoor games", n)

    if n < 30:
        logger.warning("  Too few observations")
        return {"pair": "temp_hr_mult vs park_hr_factor", "n": n,
                "pearson_r": np.nan, "spearman_rho": np.nan, "vif": np.nan}

    pr, _ = stats.pearsonr(df_dedup["temp_hr_mult"], df_dedup["hr_pf_3yr"])
    sr, _ = stats.spearmanr(df_dedup["temp_hr_mult"], df_dedup["hr_pf_3yr"])
    vif = _vif(pr)

    logger.info("  Pearson r  = %.4f", pr)
    logger.info("  Spearman ρ = %.4f", sr)
    logger.info("  VIF        = %.3f", vif)

    return {
        "pair": "temp_hr_mult vs park_hr_factor",
        "n": n,
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
        "vif": round(vif, 3),
    }


# ===================================================================
# Pair 4: Umpire K Tendency vs Catcher Framing
# ===================================================================
def pair4_umpire_vs_catcher() -> dict:
    """
    Compute per-umpire called_strike_rate and per-catcher called_strike_rate
    from fact_pitch. For games where both are known, correlate the game-level
    umpire and catcher tendencies.

    Catcher identified from fact_lineup where position = 'C'.
    """
    logger.info("Pair 4: Umpire K tendency vs Catcher framing")

    # Step 1: Compute per-umpire called strike rate (career)
    umpire_csr = read_sql("""
        SELECT
            du.hp_umpire_name,
            COUNT(*) AS total_pitches,
            SUM(fp.is_called_strike::int) AS called_strikes,
            SUM(fp.is_called_strike::int)::float / COUNT(*) AS ump_cs_rate
        FROM production.fact_pitch fp
        JOIN production.dim_umpire du ON fp.game_pk = du.game_pk
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
          AND du.hp_umpire_name IS NOT NULL
        GROUP BY du.hp_umpire_name
        HAVING COUNT(*) >= 1000
    """)

    # Step 2: Get catchers per game from fact_lineup
    catchers = read_sql("""
        SELECT fl.game_pk, fl.player_id AS catcher_id, fl.team_id
        FROM production.fact_lineup fl
        JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
        WHERE fl.position = 'C'
          AND fl.is_starter = true
          AND dg.game_type = 'R'
    """)

    # Step 3: Compute per-catcher called strike rate
    # Need to know which pitches the catcher was behind the plate for.
    # Approximate: for games where catcher started, attribute all pitches
    # thrown by their team's pitchers.
    # Identify team's pitchers via fact_lineup or fact_player_game_mlb
    catcher_csr = read_sql("""
        WITH catcher_games AS (
            SELECT fl.game_pk, fl.player_id AS catcher_id, fl.team_id
            FROM production.fact_lineup fl
            JOIN production.dim_game dg ON fl.game_pk = dg.game_pk
            WHERE fl.position = 'C'
              AND fl.is_starter = true
              AND dg.game_type = 'R'
        ),
        catcher_pitches AS (
            SELECT
                cg.catcher_id,
                fp.game_pk,
                fp.is_called_strike
            FROM production.fact_pitch fp
            JOIN catcher_games cg ON fp.game_pk = cg.game_pk
            JOIN production.fact_lineup fl_pitcher
                 ON fp.game_pk = fl_pitcher.game_pk
                 AND fp.pitcher_id = fl_pitcher.player_id
                 AND fl_pitcher.team_id = cg.team_id
        )
        SELECT
            catcher_id,
            COUNT(*) AS total_pitches,
            SUM(is_called_strike::int) AS called_strikes,
            SUM(is_called_strike::int)::float / COUNT(*) AS catch_cs_rate
        FROM catcher_pitches
        GROUP BY catcher_id
        HAVING COUNT(*) >= 1000
    """)

    # Step 4: For each game, get the umpire rate and catcher rate, then correlate
    # Build game-level join: umpire + catcher
    game_ump = read_sql("""
        SELECT game_pk, hp_umpire_name
        FROM production.dim_umpire
        WHERE hp_umpire_name IS NOT NULL
    """)

    # Merge umpire career rate onto games
    game_ump = game_ump.merge(umpire_csr[["hp_umpire_name", "ump_cs_rate"]],
                              on="hp_umpire_name", how="inner")

    # Merge catcher career rate onto games
    catchers = catchers.merge(catcher_csr[["catcher_id", "catch_cs_rate"]],
                              on="catcher_id", how="inner")

    # Each game can have 2 catchers (one per team). We'll have two rows per game.
    merged = game_ump.merge(catchers, on="game_pk", how="inner")

    merged = merged.dropna(subset=["ump_cs_rate", "catch_cs_rate"])
    n = len(merged)
    logger.info("  N = %d game-catcher observations", n)

    if n < 30:
        logger.warning("  Too few observations")
        return {"pair": "umpire_cs_rate vs catcher_cs_rate", "n": n,
                "pearson_r": np.nan, "spearman_rho": np.nan, "vif": np.nan}

    pr, _ = stats.pearsonr(merged["ump_cs_rate"], merged["catch_cs_rate"])
    sr, _ = stats.spearmanr(merged["ump_cs_rate"], merged["catch_cs_rate"])
    vif = _vif(pr)

    logger.info("  Pearson r  = %.4f", pr)
    logger.info("  Spearman ρ = %.4f", sr)
    logger.info("  VIF        = %.3f", vif)

    return {
        "pair": "umpire_cs_rate vs catcher_cs_rate",
        "n": n,
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
        "vif": round(vif, 3),
    }


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("Feature Collinearity Audit — Phase 0C")
    logger.info("=" * 60)

    results = []
    for func in [pair1_lineup_vs_pitcher_k,
                 pair2_platoon_vs_bvp,
                 pair3_weather_vs_park,
                 pair4_umpire_vs_catcher]:
        try:
            results.append(func())
        except Exception:
            logger.exception("Failed: %s", func.__name__)
            results.append({
                "pair": func.__name__,
                "n": 0,
                "pearson_r": np.nan,
                "spearman_rho": np.nan,
                "vif": np.nan,
            })

    # Build summary table
    summary = pd.DataFrame(results)
    summary["flag"] = summary["pearson_r"].abs().apply(
        lambda r: "HIGH" if r > 0.5 else ("MODERATE" if r > 0.3 else "LOW")
    )
    summary["recommendation"] = summary.apply(_recommendation, axis=1)

    out_path = OUTPUT_DIR / "collinearity_audit.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    # Print summary
    print("\n" + "=" * 80)
    print("COLLINEARITY AUDIT RESULTS")
    print("=" * 80)
    for _, row in summary.iterrows():
        print(f"\n  {row['pair']}")
        print(f"    N            = {row['n']:,}")
        print(f"    Pearson r    = {row['pearson_r']}")
        print(f"    Spearman rho = {row['spearman_rho']}")
        print(f"    VIF          = {row['vif']}")
        print(f"    Risk         = {row['flag']}")
        print(f"    -> {row['recommendation']}")

    print("\n" + "=" * 80)

    # Final guidance
    high_risk = summary[summary["flag"] == "HIGH"]
    if len(high_risk) > 0:
        print("\nWARNING: High collinearity detected in:")
        for _, row in high_risk.iterrows():
            print(f"  - {row['pair']} (r={row['pearson_r']})")
        print("  Test these features individually before combining.\n")
    else:
        print("\nNo high-collinearity pairs detected. Safe to combine features.\n")


def _recommendation(row: pd.Series) -> str:
    """Generate a recommendation based on correlation strength."""
    r = abs(row["pearson_r"])
    if pd.isna(r):
        return "Insufficient data — cannot assess."
    if r > 0.5:
        return (
            "HIGH RISK: Test individually before combining. "
            "Consider dropping one or orthogonalizing (residualize)."
        )
    if r > 0.3:
        return (
            "MODERATE: Some shared signal. Monitor VIF in full model. "
            "Likely safe if both add unique predictive lift."
        )
    return "LOW: Minimal overlap. Safe to include both features."


if __name__ == "__main__":
    main()
