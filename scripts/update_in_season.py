#!/usr/bin/env python
"""
In-season daily update script.

Performs Beta-Binomial conjugate updating of preseason projections
with observed 2026 data, regenerates dashboard parquets, and
fetches today's schedule with matchup analysis.

Usage
-----
    python scripts/update_in_season.py                    # today's date
    python scripts/update_in_season.py --date 2026-04-15  # specific date
    python scripts/update_in_season.py --skip-schedule    # skip API calls
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_REPO = Path(r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard")
DASHBOARD_DIR = DASHBOARD_REPO / "data" / "dashboard"
SNAPSHOT_DIR = DASHBOARD_DIR / "snapshots"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEASON = 2026

# Rate stats to update via conjugate updating
HITTER_RATE_STATS = [
    {"name": "k_rate", "trials": "pa", "successes": "strikeouts"},
    {"name": "bb_rate", "trials": "pa", "successes": "walks"},
]
PITCHER_RATE_STATS = [
    {"name": "k_rate", "trials": "batters_faced", "successes": "strike_outs"},
    {"name": "bb_rate", "trials": "batters_faced", "successes": "walks"},
]


def load_preseason_snapshot(player_type: str) -> pd.DataFrame:
    """Load frozen preseason projections."""
    fname = f"{player_type}_projections_{SEASON}_preseason.parquet"
    path = SNAPSHOT_DIR / fname
    if not path.exists():
        logger.warning("No preseason snapshot at %s — using live projections", path)
        path = DASHBOARD_DIR / f"{player_type}_projections.parquet"
    if not path.exists():
        logger.error("No projections found for %s", player_type)
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_preseason_k_samples() -> dict[str, np.ndarray]:
    """Load frozen preseason K% samples."""
    path = SNAPSHOT_DIR / "pitcher_k_samples_preseason.npz"
    if not path.exists():
        path = DASHBOARD_DIR / "pitcher_k_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_preseason_rate_samples(stat_name: str) -> dict[str, np.ndarray]:
    """Load frozen preseason BB% or HR/BF samples."""
    npz_name = f"pitcher_{stat_name}_samples"
    path = SNAPSHOT_DIR / f"{npz_name}_preseason.npz"
    if not path.exists():
        path = DASHBOARD_DIR / f"{npz_name}.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_hitter_preseason_samples(stat_name: str) -> dict[str, np.ndarray]:
    """Load frozen preseason hitter rate samples (k, bb, or hr)."""
    npz_name = f"hitter_{stat_name}_samples"
    path = SNAPSHOT_DIR / f"{npz_name}_preseason.npz"
    if not path.exists():
        path = DASHBOARD_DIR / f"{npz_name}.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_observed_hitter_totals() -> pd.DataFrame:
    """Query 2026 hitter season totals from the database."""
    from src.data.db import read_sql
    df = read_sql("""
        SELECT
            fp.batter_id,
            COUNT(DISTINCT fp.pa_id) AS pa,
            SUM(CASE WHEN fp.events = 'strikeout' THEN 1 ELSE 0 END) AS strikeouts,
            SUM(CASE WHEN fp.events = 'walk' THEN 1 ELSE 0 END) AS walks,
            SUM(CASE WHEN fp.events = 'home_run' THEN 1 ELSE 0 END) AS hr
        FROM production.fact_pa fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY fp.batter_id
        HAVING COUNT(DISTINCT fp.pa_id) >= 1
    """, {"season": SEASON})
    logger.info("Observed 2026 hitter totals: %d batters", len(df))
    return df


def get_observed_pitcher_totals() -> pd.DataFrame:
    """Query 2026 pitcher season totals from the database."""
    from src.data.db import read_sql
    df = read_sql("""
        SELECT
            pb.pitcher_id,
            SUM(pb.batters_faced) AS batters_faced,
            SUM(pb.strike_outs) AS strike_outs,
            SUM(pb.walks) AS walks,
            SUM(pb.home_runs) AS hr,
            COUNT(*) AS games_pitched,
            SUM(CASE WHEN pb.is_starter THEN 1 ELSE 0 END) AS games_started
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.season = :season AND dg.game_type = 'R'
        GROUP BY pb.pitcher_id
        HAVING SUM(pb.batters_faced) >= 1
    """, {"season": SEASON})
    logger.info("Observed 2026 pitcher totals: %d pitchers", len(df))
    return df


def update_projections_step() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 1: Conjugate-update rate projections."""
    from src.models.in_season_updater import update_projections

    # Hitters
    h_preseason = load_preseason_snapshot("hitter")
    h_obs = get_observed_hitter_totals()

    if h_preseason.empty:
        logger.error("No hitter preseason projections — skipping hitter update")
        h_updated = pd.DataFrame()
    elif h_obs.empty:
        logger.info("No 2026 hitter data yet — using preseason projections")
        h_updated = h_preseason.copy()
    else:
        h_updated = update_projections(
            h_preseason, h_obs,
            id_col="batter_id",
            rate_stats=HITTER_RATE_STATS,
            min_trials=10,
        )
        n_updated = h_updated.get("obs_2026_pa", pd.Series(dtype=float)).gt(0).sum()
        logger.info("Updated %d hitters with 2026 data", n_updated)

    # Pitchers
    p_preseason = load_preseason_snapshot("pitcher")
    p_obs = get_observed_pitcher_totals()

    if p_preseason.empty:
        logger.error("No pitcher preseason projections — skipping pitcher update")
        p_updated = pd.DataFrame()
    elif p_obs.empty:
        logger.info("No 2026 pitcher data yet — using preseason projections")
        p_updated = p_preseason.copy()
    else:
        p_updated = update_projections(
            p_preseason, p_obs,
            id_col="pitcher_id",
            rate_stats=PITCHER_RATE_STATS,
            min_trials=10,
        )
        n_updated = p_updated.get("obs_2026_batters_faced", pd.Series(dtype=float)).gt(0).sum()
        logger.info("Updated %d pitchers with 2026 data", n_updated)

    return h_updated, p_updated


def update_k_samples_step() -> dict[str, np.ndarray]:
    """Step 2: Regenerate pitcher K% samples via conjugate updating."""
    from src.models.in_season_updater import update_pitcher_k_samples

    preseason_samples = load_preseason_k_samples()
    p_obs = get_observed_pitcher_totals()

    if not preseason_samples:
        logger.warning("No preseason K%% samples — skipping K sample update")
        return {}

    if p_obs.empty:
        logger.info("No 2026 pitcher data — using preseason K%% samples")
        return preseason_samples

    updated = update_pitcher_k_samples(
        preseason_samples, p_obs,
        min_bf=10, n_samples=8000,
    )
    logger.info("Updated K%% samples: %d pitchers", len(updated))
    return updated


def update_rate_samples_step(
    stat_name: str,
    trials_col: str,
    successes_col: str,
    league_avg: float,
) -> dict[str, np.ndarray]:
    """Conjugate-update pitcher BB% or HR/BF samples."""
    from src.models.in_season_updater import update_rate_samples

    preseason = load_preseason_rate_samples(stat_name)
    if not preseason:
        logger.warning("No preseason %s samples — skipping", stat_name)
        return {}

    p_obs = get_observed_pitcher_totals()
    if p_obs.empty:
        logger.info("No 2026 pitcher data — using preseason %s samples", stat_name)
        return preseason

    updated = update_rate_samples(
        preseason, p_obs,
        player_id_col="pitcher_id",
        trials_col=trials_col,
        successes_col=successes_col,
        league_avg=league_avg,
        min_trials=10, n_samples=8000,
    )
    logger.info("Updated %s samples: %d pitchers", stat_name, len(updated))
    return updated


def update_hitter_rate_samples_step(
    stat_name: str,
    trials_col: str,
    successes_col: str,
    league_avg: float,
) -> dict[str, np.ndarray]:
    """Conjugate-update hitter K%, BB%, or HR rate samples."""
    from src.models.in_season_updater import update_rate_samples

    preseason = load_hitter_preseason_samples(stat_name)
    if not preseason:
        logger.warning("No preseason hitter %s samples — skipping", stat_name)
        return {}

    h_obs = get_observed_hitter_totals()
    if h_obs.empty:
        logger.info("No 2026 hitter data — using preseason %s samples", stat_name)
        return preseason

    updated = update_rate_samples(
        preseason, h_obs,
        player_id_col="batter_id",
        trials_col=trials_col,
        successes_col=successes_col,
        league_avg=league_avg,
        min_trials=10, n_samples=8000,
    )
    logger.info("Updated hitter %s samples: %d batters", stat_name, len(updated))
    return updated


def fetch_schedule_step(
    game_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 3: Fetch today's schedule and lineups."""
    from src.data.schedule import fetch_todays_schedule, fetch_all_lineups

    schedule = fetch_todays_schedule(game_date)
    if schedule.empty:
        logger.info("No games scheduled for %s", game_date)
        return schedule, pd.DataFrame()

    lineups = fetch_all_lineups(schedule)
    logger.info("Fetched lineups: %d batters across %d games",
                len(lineups), lineups["game_pk"].nunique() if not lineups.empty else 0)

    return schedule, lineups


def simulate_todays_games(
    schedule: pd.DataFrame,
    lineups: pd.DataFrame,
    k_samples: dict[str, np.ndarray],
    pitcher_proj: pd.DataFrame,
    bf_priors: pd.DataFrame,
) -> pd.DataFrame:
    """Step 4: Run K simulations for each starting pitcher."""
    from src.models.bf_model import get_bf_distribution
    from src.models.game_k_model import compute_k_over_probs, simulate_game_ks
    from src.models.matchup import score_matchup
    from src.data.league_baselines import get_baselines_dict

    # Load matchup data
    arsenal_path = DASHBOARD_DIR / "pitcher_arsenal.parquet"
    vuln_path = DASHBOARD_DIR / "hitter_vuln_career.parquet"
    arsenal_df = pd.read_parquet(arsenal_path) if arsenal_path.exists() else pd.DataFrame()
    vuln_df = pd.read_parquet(vuln_path) if vuln_path.exists() else pd.DataFrame()

    baselines_pt = get_baselines_dict(grouping="pitch_type", recency_weights="marcel")

    results = []
    for _, game in schedule.iterrows():
        gpk = game["game_pk"]

        for side in ("away", "home"):
            pid = game.get(f"{side}_pitcher_id")
            pname = game.get(f"{side}_pitcher_name", "")

            if pd.isna(pid):
                continue
            pid = int(pid)
            pid_str = str(pid)

            if pid_str not in k_samples:
                continue

            samples = k_samples[pid_str]

            # BF distribution
            bf_info = get_bf_distribution(pid, SEASON - 1, bf_priors)
            bf_mu = bf_info["mu_bf"]
            bf_sigma = bf_info["sigma_bf"]

            # Lineup matchup lifts
            lineup_lifts = None
            opp_side = "home" if side == "away" else "away"
            opp_team_id = game.get(f"{opp_side}_team_id")

            if not lineups.empty and not arsenal_df.empty and not vuln_df.empty:
                game_lu = lineups[
                    (lineups["game_pk"] == gpk) &
                    (lineups["team_id"] == opp_team_id)
                ].sort_values("batting_order")

                if len(game_lu) >= 9:
                    lifts = []
                    for _, brow in game_lu.head(9).iterrows():
                        bid = int(brow["batter_id"])
                        m = score_matchup(
                            pid, bid, arsenal_df, vuln_df, baselines_pt,
                        )
                        lift = m.get("matchup_k_logit_lift", 0.0)
                        lifts.append(0.0 if np.isnan(lift) else lift)
                    lineup_lifts = np.array(lifts)

            # Simulate
            game_ks = simulate_game_ks(
                pitcher_k_rate_samples=samples,
                bf_mu=float(bf_mu),
                bf_sigma=float(bf_sigma),
                lineup_matchup_lifts=lineup_lifts,
                n_draws=10000,
                random_seed=42 + gpk,
            )

            k_over = compute_k_over_probs(game_ks)

            # Get P(over) for common lines
            p_over_dict = {}
            for _, kr in k_over.iterrows():
                line = kr["line"]
                if line in (4.5, 5.5, 6.5, 7.5):
                    p_over_dict[f"p_over_{line:.1f}".replace(".", "_")] = kr["p_over"]

            # Pitcher projection info
            p_row = pitcher_proj[pitcher_proj["pitcher_id"] == pid]
            proj_k_rate = float(p_row.iloc[0]["projected_k_rate"]) if not p_row.empty else np.mean(samples)
            composite = float(p_row.iloc[0].get("composite_score", 0)) if not p_row.empty else 0.0

            results.append({
                "game_pk": gpk,
                "side": side,
                "pitcher_id": pid,
                "pitcher_name": pname,
                "team_abbr": game.get(f"{side}_abbr", ""),
                "opp_abbr": game.get(f"{opp_side}_abbr", ""),
                "projected_k_rate": proj_k_rate,
                "composite_score": composite,
                "expected_k": float(np.mean(game_ks)),
                "k_std": float(np.std(game_ks)),
                "median_k": float(np.median(game_ks)),
                "has_lineup": lineup_lifts is not None,
                "avg_matchup_lift": float(np.mean(lineup_lifts)) if lineup_lifts is not None else 0.0,
                **p_over_dict,
            })

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="In-season daily update")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--skip-schedule", action="store_true",
                        help="Skip fetching schedule/lineups from MLB API.")
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    logger.info("=" * 60)
    logger.info("In-season update for %s (season %d)", game_date, SEASON)
    logger.info("=" * 60)

    # Step 1: Update rate projections
    logger.info("Step 1: Updating rate projections...")
    h_updated, p_updated = update_projections_step()

    if not h_updated.empty:
        # Drop snapshot columns if present
        for col in ["snapshot_date", "target_season"]:
            if col in h_updated.columns:
                h_updated = h_updated.drop(columns=[col])
        h_updated.to_parquet(DASHBOARD_DIR / "hitter_projections.parquet", index=False)
        logger.info("Saved updated hitter projections: %d rows", len(h_updated))

    if not p_updated.empty:
        for col in ["snapshot_date", "target_season"]:
            if col in p_updated.columns:
                p_updated = p_updated.drop(columns=[col])
        p_updated.to_parquet(DASHBOARD_DIR / "pitcher_projections.parquet", index=False)
        logger.info("Saved updated pitcher projections: %d rows", len(p_updated))

    # Step 2: Update K% samples
    logger.info("Step 2: Updating pitcher K%% samples...")
    k_samples = update_k_samples_step()
    if k_samples:
        np.savez_compressed(
            DASHBOARD_DIR / "pitcher_k_samples.npz",
            **k_samples,
        )
        logger.info("Saved updated K%% samples for %d pitchers", len(k_samples))

    # Step 2b: Update BB% and HR/BF samples
    logger.info("Step 2b: Updating pitcher BB%% and HR/BF samples...")
    bb_samples = update_rate_samples_step(
        "bb", trials_col="batters_faced", successes_col="walks",
        league_avg=0.08,
    )
    if bb_samples:
        np.savez_compressed(DASHBOARD_DIR / "pitcher_bb_samples.npz", **bb_samples)
        logger.info("Saved updated BB%% samples for %d pitchers", len(bb_samples))

    hr_samples = update_rate_samples_step(
        "hr", trials_col="batters_faced", successes_col="hr",
        league_avg=0.03,
    )
    if hr_samples:
        np.savez_compressed(DASHBOARD_DIR / "pitcher_hr_samples.npz", **hr_samples)
        logger.info("Saved updated HR/BF samples for %d pitchers", len(hr_samples))

    # Step 2c: Update hitter K%, BB%, HR samples
    logger.info("Step 2c: Updating hitter K%%, BB%%, HR samples...")
    h_k_samples = update_hitter_rate_samples_step(
        "k", trials_col="pa", successes_col="strikeouts",
        league_avg=0.226,
    )
    if h_k_samples:
        np.savez_compressed(DASHBOARD_DIR / "hitter_k_samples.npz", **h_k_samples)
        logger.info("Saved updated hitter K%% samples for %d batters", len(h_k_samples))

    h_bb_samples = update_hitter_rate_samples_step(
        "bb", trials_col="pa", successes_col="walks",
        league_avg=0.082,
    )
    if h_bb_samples:
        np.savez_compressed(DASHBOARD_DIR / "hitter_bb_samples.npz", **h_bb_samples)
        logger.info("Saved updated hitter BB%% samples for %d batters", len(h_bb_samples))

    h_hr_samples = update_hitter_rate_samples_step(
        "hr", trials_col="pa", successes_col="hr",
        league_avg=0.031,
    )
    if h_hr_samples:
        np.savez_compressed(DASHBOARD_DIR / "hitter_hr_samples.npz", **h_hr_samples)
        logger.info("Saved updated hitter HR samples for %d batters", len(h_hr_samples))

    # Step 3: Update team assignments
    logger.info("Step 3: Updating team assignments...")
    try:
        from src.data.queries import get_player_teams
        teams = get_player_teams(SEASON - 1)  # will pick up 2026 data when available
        teams.to_parquet(DASHBOARD_DIR / "player_teams.parquet", index=False)
        logger.info("Updated player teams: %d players", len(teams))
    except Exception as e:
        logger.warning("Team update failed: %s", e)

    # Step 4: Fetch schedule and simulate today's games
    if not args.skip_schedule:
        logger.info("Step 4: Fetching schedule for %s...", game_date)
        schedule, lineups = fetch_schedule_step(game_date)

        if not schedule.empty:
            schedule.to_parquet(DASHBOARD_DIR / "todays_games.parquet", index=False)

            if not lineups.empty:
                lineups.to_parquet(DASHBOARD_DIR / "todays_lineups.parquet", index=False)

            # Simulate K props for each starter
            bf_priors_path = DASHBOARD_DIR / "bf_priors.parquet"
            bf_priors = pd.read_parquet(bf_priors_path) if bf_priors_path.exists() else pd.DataFrame()

            logger.info("Simulating K props for today's starters...")
            sim_results = simulate_todays_games(
                schedule, lineups, k_samples, p_updated, bf_priors,
            )
            if not sim_results.empty:
                sim_results.to_parquet(DASHBOARD_DIR / "todays_sims.parquet", index=False)
                logger.info("Saved K simulations for %d pitcher appearances", len(sim_results))
        else:
            logger.info("No games today — skipping simulation")
    else:
        logger.info("Step 4: Skipped (--skip-schedule)")

    # Save update metadata
    metadata = {
        "last_updated": datetime.now().isoformat(),
        "game_date": game_date,
        "season": SEASON,
        "hitters_updated": len(h_updated) if not h_updated.empty else 0,
        "pitchers_updated": len(p_updated) if not p_updated.empty else 0,
        "pitcher_k_samples_count": len(k_samples),
        "pitcher_bb_samples_count": len(bb_samples),
        "pitcher_hr_samples_count": len(hr_samples),
        "hitter_k_samples_count": len(h_k_samples),
        "hitter_bb_samples_count": len(h_bb_samples),
        "hitter_hr_samples_count": len(h_hr_samples),
    }
    meta_path = DASHBOARD_DIR / "update_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved update metadata to %s", meta_path)

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
