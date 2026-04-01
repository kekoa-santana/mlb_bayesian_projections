"""Precompute: Component park factors (K, H, BB, HR) per venue.

Computes park factors using the standard home/away method across
2022-2025 regular season data. Applies Bayesian shrinkage toward 1.0
based on sample size for stability.

Output: data/dashboard/park_factors.parquet
    Columns: venue_id, stat, park_factor, games, regressed

Also outputs a logit-lift version for direct use in the game simulator:
    data/dashboard/park_factor_lifts.parquet
    Columns: venue_id, k_lift, bb_lift, hr_lift, h_babip_adj
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit as _logit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql

logger = logging.getLogger("precompute.park_factors")

DASHBOARD_DIR = Path(
    r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard"
)

# Seasons to use for park factor computation
PF_SEASONS = [2022, 2023, 2024, 2025]

# League-average rates (2022-2025) for logit conversion
LEAGUE_K_RATE = 0.226
LEAGUE_BB_RATE = 0.082
LEAGUE_HR_RATE = 0.031
LEAGUE_H_RATE = 0.230  # H/PA (approx BA on all PAs)
LEAGUE_BABIP = 0.296

# Shrinkage: number of "prior games" worth of data pulling toward 1.0.
# With ~320 games per venue across 4 years, a prior of 80 gives
# moderate regression (80/(80+320) = 20% shrinkage for full-data venues).
SHRINKAGE_GAMES = 80

# Minimum games at a venue to include (filters out neutral-site games)
MIN_VENUE_GAMES = 40

# Venue name mapping (MLB Stats API venue IDs -> display names)
# These are the "extreme" parks for backtest reporting
EXTREME_VENUES = {
    2395: "Coors Field",        # COL - high altitude, inflated offense
    3289: "Oracle Park",        # SF  - pitcher friendly
    3: "Fenway Park",           # BOS - unique dimensions
    2680: "Petco Park",         # SD  - pitcher friendly
    15: "Chase Field",          # ARI - retractable roof, warm
    680: "Tropicana Field",     # TB  - dome
    4705: "Globe Life Field",   # TEX - retractable roof
    12: "Wrigley Field",        # CHC - wind effects
    31: "Great American Ball Park",  # CIN - small, hitter friendly
    5325: "American Family Field",   # MIL - retractable roof
}


def fetch_team_game_stats() -> pd.DataFrame:
    """Fetch per-team per-game batting stats with venue info.

    Returns one row per (game_pk, team_id) with aggregate batting stats
    and whether the team was home or away.
    """
    season_list = ", ".join(str(s) for s in PF_SEASONS)
    query = f"""
    SELECT
        dg.game_pk,
        dg.venue_id,
        dg.season,
        dg.home_team_id,
        bb.team_id,
        SUM(bb.strikeouts) AS k,
        SUM(bb.hits) AS h,
        SUM(bb.walks) AS bb_total,
        SUM(bb.home_runs) AS hr,
        SUM(bb.plate_appearances) AS pa
    FROM staging.batting_boxscores bb
    JOIN production.dim_game dg
      ON bb.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
      AND dg.season IN ({season_list})
      AND bb.plate_appearances > 0
    GROUP BY dg.game_pk, dg.venue_id, dg.season, dg.home_team_id, bb.team_id
    ORDER BY dg.game_pk, bb.team_id
    """
    logger.info("Fetching team-game batting stats for seasons %s...", PF_SEASONS)
    df = read_sql(query)
    df["is_home"] = df["team_id"] == df["home_team_id"]
    logger.info("Fetched %d team-game rows across %d games", len(df), df["game_pk"].nunique())
    return df


def compute_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute component park factors using the balanced home/away method.

    For each venue and stat, computes a per-team park factor:
        team_pf = (team_rate_at_venue / team_rate_away_from_venue)
    then averages across all teams that played there, weighted by PA.

    This controls for team talent: each team is its own control group.

    Then applies Bayesian shrinkage: pf = (n * pf_raw + prior * 1.0) / (n + prior)
    """
    stats = ["k", "h", "bb_total", "hr"]
    stat_labels = {"k": "K", "h": "H", "bb_total": "BB", "hr": "HR"}

    # Pre-compute per-team per-venue aggregates
    team_venue = (
        df.groupby(["team_id", "venue_id"])
        .agg({"k": "sum", "h": "sum", "bb_total": "sum", "hr": "sum", "pa": "sum"})
        .reset_index()
    )

    # Also compute each team's total away stats (games NOT at their home venue)
    # First identify each team's home venue
    home_venues = (
        df[df["is_home"]]
        .groupby("team_id")["venue_id"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"venue_id": "home_venue_id"})
    )

    # Per-team overall away stats (all road games combined)
    df_with_home = df.merge(home_venues, on="team_id", how="left")
    team_away = (
        df_with_home[df_with_home["venue_id"] != df_with_home["home_venue_id"]]
        .groupby("team_id")
        .agg({"k": "sum", "h": "sum", "bb_total": "sum", "hr": "sum", "pa": "sum"})
        .reset_index()
        .rename(columns={s: f"{s}_away" for s in stats + ["pa"]})
    )

    records = []
    for venue_id in df["venue_id"].unique():
        n_games = df[df["venue_id"] == venue_id]["game_pk"].nunique()
        if n_games < MIN_VENUE_GAMES:
            continue

        # Get all teams that batted at this venue
        venue_data = team_venue[team_venue["venue_id"] == venue_id]

        for stat in stats:
            # Compute weighted average park factor across all teams
            weighted_pf_num = 0.0
            weighted_pf_den = 0.0

            for _, trow in venue_data.iterrows():
                tid = trow["team_id"]
                home_stat = trow[stat]
                home_pa = trow["pa"]
                if home_pa < 50:
                    continue

                # This team's road stats
                away_row = team_away[team_away["team_id"] == tid]
                if len(away_row) == 0:
                    continue
                away_stat = float(away_row.iloc[0][f"{stat}_away"])
                away_pa = float(away_row.iloc[0]["pa_away"])
                if away_pa < 50:
                    continue

                home_rate = home_stat / home_pa
                away_rate = away_stat / away_pa
                if away_rate < 1e-6:
                    continue

                team_pf = home_rate / away_rate
                # Weight by PA at this venue
                weighted_pf_num += team_pf * home_pa
                weighted_pf_den += home_pa

            if weighted_pf_den == 0:
                continue

            pf_raw = weighted_pf_num / weighted_pf_den

            # Overall rates for reference
            total_home_pa = venue_data["pa"].sum()
            total_home_stat = venue_data[stat].sum()
            home_rate = total_home_stat / total_home_pa if total_home_pa > 0 else 0

            # Bayesian shrinkage toward 1.0
            pf_regressed = (
                (n_games * pf_raw + SHRINKAGE_GAMES * 1.0)
                / (n_games + SHRINKAGE_GAMES)
            )

            records.append({
                "venue_id": int(venue_id),
                "stat": stat_labels[stat],
                "park_factor": round(float(pf_raw), 4),
                "park_factor_regressed": round(float(pf_regressed), 4),
                "home_rate": round(float(home_rate), 5),
                "n_teams": int(len(venue_data[venue_data["pa"] >= 50])),
                "home_pa": int(total_home_pa),
                "games": int(n_games),
            })

    result = pd.DataFrame(records)
    return result


def compute_logit_lifts(pf_df: pd.DataFrame) -> pd.DataFrame:
    """Convert park factors to logit-additive lifts for the game simulator.

    For K, BB, HR: logit lift = logit(league_rate * pf) - logit(league_rate)
    For H: BABIP adjustment = (pf - 1.0) * league_BABIP * coefficient

    These lifts stack additively in logit space with existing umpire/weather
    lifts in the lineup simulator.
    """
    league_rates = {
        "K": LEAGUE_K_RATE,
        "BB": LEAGUE_BB_RATE,
        "HR": LEAGUE_HR_RATE,
    }

    # Pivot to one row per venue
    pivot = pf_df.pivot_table(
        index="venue_id",
        columns="stat",
        values="park_factor_regressed",
        aggfunc="first",
    ).reset_index()

    records = []
    for _, row in pivot.iterrows():
        venue_id = int(row["venue_id"])
        lifts = {"venue_id": venue_id}

        # K lift (logit-additive)
        k_pf = row.get("K", 1.0)
        if pd.notna(k_pf):
            park_k = np.clip(LEAGUE_K_RATE * k_pf, 1e-4, 1 - 1e-4)
            lifts["k_lift"] = round(
                float(_logit(park_k) - _logit(LEAGUE_K_RATE)), 5
            )
        else:
            lifts["k_lift"] = 0.0

        # BB lift (logit-additive)
        bb_pf = row.get("BB", 1.0)
        if pd.notna(bb_pf):
            park_bb = np.clip(LEAGUE_BB_RATE * bb_pf, 1e-4, 1 - 1e-4)
            lifts["bb_lift"] = round(
                float(_logit(park_bb) - _logit(LEAGUE_BB_RATE)), 5
            )
        else:
            lifts["bb_lift"] = 0.0

        # HR lift (logit-additive) -- uses existing park_hr_lift parameter
        hr_pf = row.get("HR", 1.0)
        if pd.notna(hr_pf):
            park_hr = np.clip(LEAGUE_HR_RATE * hr_pf, 1e-4, 1 - 1e-4)
            lifts["hr_lift"] = round(
                float(_logit(park_hr) - _logit(LEAGUE_HR_RATE)), 5
            )
        else:
            lifts["hr_lift"] = 0.0

        # H (BABIP adjustment): park factor shifts the probability of a BIP
        # becoming a hit. Convert to a BABIP delta that stacks with LD% and
        # sprint speed adjustments.
        h_pf = row.get("H", 1.0)
        if pd.notna(h_pf):
            # BABIP adjustment = (pf - 1.0) * league_BABIP
            # e.g. Coors H_pf=1.12 -> babip_adj = +0.12 * 0.296 = +0.036
            lifts["h_babip_adj"] = round(float((h_pf - 1.0) * LEAGUE_BABIP), 5)
        else:
            lifts["h_babip_adj"] = 0.0

        records.append(lifts)

    return pd.DataFrame(records)


def run() -> None:
    """Compute and save park factors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Computing component park factors...")

    # Step 1: Fetch data
    team_games = fetch_team_game_stats()

    # Step 2: Compute raw + regressed park factors
    pf_df = compute_park_factors(team_games)
    logger.info("Computed park factors for %d venues", pf_df["venue_id"].nunique())

    # Print summary
    for stat in ["K", "H", "BB", "HR"]:
        sub = pf_df[pf_df["stat"] == stat].sort_values("park_factor_regressed")
        print(f"\n--- {stat} Park Factors (regressed) ---")
        print(f"  Lowest:  venue {sub.iloc[0]['venue_id']:>5d}  pf={sub.iloc[0]['park_factor_regressed']:.3f}")
        print(f"  Highest: venue {sub.iloc[-1]['venue_id']:>5d}  pf={sub.iloc[-1]['park_factor_regressed']:.3f}")
        print(f"  Range:   {sub['park_factor_regressed'].min():.3f} - {sub['park_factor_regressed'].max():.3f}")

    # Step 3: Compute logit lifts
    lifts_df = compute_logit_lifts(pf_df)
    logger.info("Computed logit lifts for %d venues", len(lifts_df))

    # Print extreme venues
    print("\n--- Extreme Park Lifts ---")
    for vid, vname in EXTREME_VENUES.items():
        row = lifts_df[lifts_df["venue_id"] == vid]
        if len(row) > 0:
            r = row.iloc[0]
            print(
                f"  {vname:30s}  K={r['k_lift']:+.4f}  BB={r['bb_lift']:+.4f}  "
                f"HR={r['hr_lift']:+.4f}  H_babip={r['h_babip_adj']:+.4f}"
            )

    # Step 4: Save
    pf_path = DASHBOARD_DIR / "park_factors.parquet"
    pf_df.to_parquet(pf_path, index=False)
    logger.info("Saved park factors to %s (%d rows)", pf_path, len(pf_df))

    lifts_path = DASHBOARD_DIR / "park_factor_lifts.parquet"
    lifts_df.to_parquet(lifts_path, index=False)
    logger.info("Saved park factor lifts to %s (%d rows)", lifts_path, len(lifts_df))


if __name__ == "__main__":
    run()
