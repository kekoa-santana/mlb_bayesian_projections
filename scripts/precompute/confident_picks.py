"""Precompute: Daily game prop projections from game simulator.

Runs pitcher and batter game sims for today's (and tomorrow's) games,
computing P(over) at player-relative lines for each stat.

Output: game_props.parquet — one row per player x stat with expected
value and P(over) at 3 lines centered on the player's projection.

Requires: pre-computed posterior samples, exit model, matchup data,
bullpen rates, etc.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from precompute import DASHBOARD_DIR, FROM_SEASON, SEASONS

logger = logging.getLogger("precompute.confident_picks")

# Batter stats to project (HR excluded — too noisy at game level)
BATTER_STATS = ("h", "tb", "k")


def run(
    *,
    game_date: str | None = None,
    n_sims: int = 10_000,
) -> None:
    """Generate game prop projections for today's and tomorrow's games.

    Parameters
    ----------
    game_date : str or None
        Date as 'YYYY-MM-DD'. Defaults to today.
    n_sims : int
        Monte Carlo sims per game/batter.
    """
    from src.data.schedule import fetch_todays_schedule
    from src.models.game_sim.exit_model import ExitModel
    from src.models.game_sim.simulator import simulate_game
    from src.models.game_sim.batter_simulator import simulate_batter_game
    from src.models.game_sim.tto_model import build_all_tto_lifts
    from src.models.game_sim.pitch_count_model import build_pitch_count_features
    from src.models.matchup import score_matchup_for_stat
    from src.models.bf_model import compute_pitcher_bf_priors

    if game_date is None:
        game_date = date.today().isoformat()

    tomorrow = (
        date.fromisoformat(game_date) + timedelta(days=1)
    ).isoformat()

    logger.info("=" * 60)
    logger.info("Generating edge picks for %s + %s...", game_date, tomorrow)

    # --- Load schedule (today + tomorrow) ---
    schedule_today = fetch_todays_schedule(game_date)
    schedule_tomorrow = fetch_todays_schedule(tomorrow)
    schedule = pd.concat(
        [schedule_today, schedule_tomorrow], ignore_index=True,
    )
    if schedule.empty:
        logger.warning("No games scheduled for %s-%s", game_date, tomorrow)
        _save_empty(game_date)
        return
    logger.info(
        "Games found: %d today + %d tomorrow = %d",
        len(schedule_today), len(schedule_tomorrow), len(schedule),
    )

    # --- Load pre-computed data ---
    logger.info("Loading pre-computed sim inputs...")

    # Posterior samples
    pitcher_k_npz = np.load(DASHBOARD_DIR / "pitcher_k_samples.npz")
    pitcher_bb_npz = np.load(DASHBOARD_DIR / "pitcher_bb_samples.npz")
    pitcher_hr_npz = np.load(DASHBOARD_DIR / "pitcher_hr_samples.npz")
    hitter_k_npz = np.load(DASHBOARD_DIR / "hitter_k_samples.npz")
    hitter_bb_npz = np.load(DASHBOARD_DIR / "hitter_bb_samples.npz")
    hitter_hr_npz = np.load(DASHBOARD_DIR / "hitter_hr_samples.npz")

    # Exit model
    exit_model = ExitModel()
    exit_model.load(DASHBOARD_DIR / "exit_model.pkl")

    # Supporting data
    exit_tendencies = pd.read_parquet(
        DASHBOARD_DIR / "pitcher_exit_tendencies.parquet"
    )
    pitcher_pc = pd.read_parquet(
        DASHBOARD_DIR / "pitcher_pitch_count_features.parquet"
    )
    batter_pc = pd.read_parquet(
        DASHBOARD_DIR / "batter_pitch_count_features.parquet"
    )
    tto_profiles = pd.read_parquet(DASHBOARD_DIR / "tto_profiles.parquet")
    bullpen_rates = pd.read_parquet(
        DASHBOARD_DIR / "team_bullpen_rates.parquet"
    )

    # Matchup data
    pitcher_arsenal = pd.read_parquet(
        DASHBOARD_DIR / "pitcher_arsenal.parquet"
    )
    hitter_vuln = pd.read_parquet(DASHBOARD_DIR / "hitter_vuln.parquet")

    # BF priors
    bf_priors = pd.read_parquet(DASHBOARD_DIR / "bf_priors.parquet")

    # Player names for display
    try:
        player_teams = pd.read_parquet(
            DASHBOARD_DIR / "player_teams.parquet"
        )
    except FileNotFoundError:
        player_teams = pd.DataFrame()

    # Roster + lineup priors for projected lineups
    try:
        roster = pd.read_parquet(DASHBOARD_DIR / "roster.parquet")
    except FileNotFoundError:
        roster = pd.DataFrame()
    try:
        lineup_priors = pd.read_parquet(DASHBOARD_DIR / "lineup_priors.parquet")
    except FileNotFoundError:
        lineup_priors = pd.DataFrame()

    # Umpire/weather (optional)
    try:
        umpire_tend = pd.read_parquet(
            DASHBOARD_DIR / "umpire_tendencies.parquet"
        )
    except FileNotFoundError:
        umpire_tend = pd.DataFrame()

    # Build pitch type baselines for matchup scoring
    baselines_pt = _build_baselines_pt(pitcher_arsenal)

    # Latest bullpen rates per team
    bullpen_latest = (
        bullpen_rates.sort_values("season")
        .groupby("team_id").last().reset_index()
    )

    # Latest exit tendencies per pitcher
    tend_latest = (
        exit_tendencies.sort_values("season")
        .groupby("pitcher_id").last().reset_index()
    )

    last_train = FROM_SEASON

    # --- Run sims per game ---
    pitcher_picks = []
    batter_picks = []

    for _, game in schedule.iterrows():
        game_pk = int(game["game_pk"])

        # --- PITCHER SIMS (both starters) ---
        for side in ("home", "away"):
            pid_col = f"{side}_pitcher_id"
            pname_col = f"{side}_pitcher_name"
            team_col = f"{side}_team_id"
            opp_team_col = "away_team_id" if side == "home" else "home_team_id"
            opp_abbr_col = "away_abbr" if side == "home" else "home_abbr"

            pitcher_id = game.get(pid_col)
            if pd.isna(pitcher_id):
                continue
            pitcher_id = int(pitcher_id)
            pid_str = str(pitcher_id)

            # Check we have posteriors
            if pid_str not in pitcher_k_npz:
                continue

            k_samples = pitcher_k_npz[pid_str]
            bb_samples = pitcher_bb_npz.get(pid_str)
            hr_samples = pitcher_hr_npz.get(pid_str)
            if bb_samples is None or hr_samples is None:
                continue

            # Get opposing lineup (top 9 by batting order)
            opp_team_id = game.get(opp_team_col)
            lineup_ids = _get_lineup(
                roster, lineup_priors,
                int(opp_team_id) if pd.notna(opp_team_id) else 0,
            )
            if len(lineup_ids) < 9:
                # Pad with zeros for missing lineup slots
                lineup_ids = lineup_ids + [0] * (9 - len(lineup_ids))

            # Matchup lifts
            matchup_lifts = _compute_lineup_matchup_lifts(
                pitcher_id, lineup_ids, pitcher_arsenal,
                hitter_vuln, baselines_pt,
            )

            # TTO lifts
            tto_lifts = build_all_tto_lifts(
                tto_profiles, pitcher_id, last_train,
            )

            # Pitch count features
            pitcher_adj, batter_adjs = build_pitch_count_features(
                pitcher_features=pitcher_pc,
                batter_features=batter_pc,
                pitcher_id=pitcher_id,
                batter_ids=lineup_ids,
                season=last_train,
            )

            # Pitcher avg pitches
            tend_row = tend_latest[tend_latest["pitcher_id"] == pitcher_id]
            avg_pitches = (
                float(tend_row.iloc[0]["avg_pitches"])
                if len(tend_row) > 0 else 88.0
            )

            # Run pitcher sim
            try:
                sim = simulate_game(
                    pitcher_k_rate_samples=k_samples,
                    pitcher_bb_rate_samples=bb_samples,
                    pitcher_hr_rate_samples=hr_samples,
                    lineup_matchup_lifts=matchup_lifts,
                    tto_lifts=tto_lifts,
                    pitcher_ppa_adj=pitcher_adj,
                    batter_ppa_adjs=batter_adjs,
                    exit_model=exit_model,
                    pitcher_avg_pitches=avg_pitches,
                    n_sims=n_sims,
                    random_seed=42 + game_pk % 10000,
                )
            except Exception as e:
                logger.warning(
                    "Pitcher sim failed: %d game %d: %s",
                    pitcher_id, game_pk, e,
                )
                continue

            summary = sim.summary()
            pitcher_name = game.get(pname_col, "")
            opp_abbr = game.get(opp_abbr_col, "")
            team_abbr = game.get(f"{side}_abbr", "")
            game_dt = game.get("game_date", game_date)

            # Build player-relative lines for each stat
            for stat_key, stat_label in [("k", "K"), ("outs", "Outs")]:
                expected = summary[stat_key]["mean"]
                std = summary[stat_key]["std"]
                lines = _player_lines(expected, min_line=0.5)
                over_df = sim.over_probs(stat_key, lines)
                p_overs = dict(zip(over_df["line"], over_df["p_over"]))

                pitcher_picks.append({
                    "game_date": game_dt,
                    "game_pk": game_pk,
                    "player_id": pitcher_id,
                    "player_name": pitcher_name,
                    "player_type": "pitcher",
                    "team": team_abbr,
                    "opponent": opp_abbr,
                    "stat": stat_label,
                    "expected": round(expected, 2),
                    "std": round(std, 2),
                    "line_low": lines[0],
                    "line_mid": lines[1],
                    "line_high": lines[2],
                    "p_over_low": round(p_overs.get(lines[0], 0), 3),
                    "p_over_mid": round(p_overs.get(lines[1], 0), 3),
                    "p_over_high": round(p_overs.get(lines[2], 0), 3),
                })

        # --- BATTER SIMS (both teams' lineups) ---
        for side in ("home", "away"):
            team_col = f"{side}_team_id"
            team_abbr_col = f"{side}_abbr"
            opp_side = "away" if side == "home" else "home"
            opp_pitcher_col = f"{opp_side}_pitcher_id"
            opp_abbr_col = f"{opp_side}_abbr"

            team_id = game.get(team_col)
            opp_pitcher_id = game.get(opp_pitcher_col)

            if pd.isna(team_id) or pd.isna(opp_pitcher_id):
                continue
            team_id = int(team_id)
            opp_pitcher_id = int(opp_pitcher_id)
            opp_pid_str = str(opp_pitcher_id)

            # Starter quality
            starter_k = _posterior_mean(pitcher_k_npz, opp_pid_str, 0.226)
            starter_bb = _posterior_mean(pitcher_bb_npz, opp_pid_str, 0.082)
            starter_hr = _posterior_mean(pitcher_hr_npz, opp_pid_str, 0.031)

            # Starter BF distribution
            bf_row = bf_priors[bf_priors["pitcher_id"] == opp_pitcher_id]
            if len(bf_row) > 0:
                bf_mu = float(bf_row.iloc[0]["mu_bf"])
                bf_sigma = float(bf_row.iloc[0]["sigma_bf"])
            else:
                bf_mu, bf_sigma = 22.0, 4.5

            # Bullpen rates
            bp_row = bullpen_latest[bullpen_latest["team_id"] == int(
                game.get(f"{opp_side}_team_id", 0)
            )]
            if len(bp_row) > 0:
                bp_k = float(bp_row.iloc[0]["k_rate"])
                bp_bb = float(bp_row.iloc[0]["bb_rate"])
                bp_hr = float(bp_row.iloc[0]["hr_rate"])
            else:
                bp_k, bp_bb, bp_hr = 0.253, 0.084, 0.024

            # Get lineup
            lineup_ids = _get_lineup(roster, lineup_priors, team_id)
            team_abbr = game.get(team_abbr_col, "")
            opp_abbr = game.get(opp_abbr_col, "")

            for order_idx, batter_id in enumerate(lineup_ids):
                bid_str = str(batter_id)
                if bid_str not in hitter_k_npz:
                    continue

                batter_k = hitter_k_npz[bid_str]
                batter_bb = hitter_bb_npz.get(bid_str)
                batter_hr = hitter_hr_npz.get(bid_str)
                if batter_bb is None or batter_hr is None:
                    continue

                batting_order = order_idx + 1

                # Matchup lifts vs starter
                matchup_k, matchup_bb, matchup_hr = 0.0, 0.0, 0.0
                try:
                    for stat in ("k", "bb", "hr"):
                        res = score_matchup_for_stat(
                            stat, opp_pitcher_id, batter_id,
                            pitcher_arsenal, hitter_vuln, baselines_pt,
                        )
                        val = res.get(f"matchup_{stat}_logit_lift", 0.0)
                        if isinstance(val, float) and not np.isnan(val):
                            if stat == "k":
                                matchup_k = val
                            elif stat == "bb":
                                matchup_bb = val
                            else:
                                matchup_hr = val
                except Exception:
                    pass

                # Run batter sim
                try:
                    bsim = simulate_batter_game(
                        batter_k_rate_samples=batter_k,
                        batter_bb_rate_samples=batter_bb,
                        batter_hr_rate_samples=batter_hr,
                        batting_order=batting_order,
                        starter_k_rate=starter_k,
                        starter_bb_rate=starter_bb,
                        starter_hr_rate=starter_hr,
                        starter_bf_mu=bf_mu,
                        starter_bf_sigma=bf_sigma,
                        matchup_k_lift=matchup_k,
                        matchup_bb_lift=matchup_bb,
                        matchup_hr_lift=matchup_hr,
                        bullpen_k_rate=bp_k,
                        bullpen_bb_rate=bp_bb,
                        bullpen_hr_rate=bp_hr,
                        n_sims=n_sims,
                        random_seed=42 + game_pk % 10000 + batter_id % 1000,
                    )
                except Exception:
                    continue

                bsummary = bsim.summary()
                batter_name = _lookup_name(player_teams, batter_id)
                game_dt = game.get("game_date", game_date)

                # Player-relative lines for each batter stat
                for stat_key in BATTER_STATS:
                    expected = bsummary[stat_key]["mean"]
                    std = bsummary[stat_key]["std"]
                    lines = _player_lines(expected, min_line=0.5)
                    over_df = bsim.over_probs(stat_key, lines)
                    p_overs = dict(zip(over_df["line"], over_df["p_over"]))

                    batter_picks.append({
                        "game_date": game_dt,
                        "game_pk": game_pk,
                        "player_id": batter_id,
                        "player_name": batter_name,
                        "player_type": "batter",
                        "team": team_abbr,
                        "opponent": opp_abbr,
                        "batting_order": batting_order,
                        "stat": stat_key.upper(),
                        "expected": round(expected, 2),
                        "std": round(std, 2),
                        "line_low": lines[0],
                        "line_mid": lines[1],
                        "line_high": lines[2],
                        "p_over_low": round(p_overs.get(lines[0], 0), 3),
                        "p_over_mid": round(p_overs.get(lines[1], 0), 3),
                        "p_over_high": round(p_overs.get(lines[2], 0), 3),
                    })

    # --- Combine and save ---
    all_props = pd.DataFrame(pitcher_picks + batter_picks)
    if len(all_props) > 0:
        # Sort pitchers first, then by expected (descending)
        all_props = all_props.sort_values(
            ["game_date", "player_type", "expected"],
            ascending=[True, True, False],
        )

    all_props.to_parquet(
        DASHBOARD_DIR / "game_props.parquet", index=False,
    )
    logger.info(
        "Saved %d prop projections (%d pitcher, %d batter) to game_props.parquet",
        len(all_props), len(pitcher_picks), len(batter_picks),
    )

    # Summary: show pitchers with most interesting K projections
    if len(all_props) > 0:
        pit_k = all_props[
            (all_props["player_type"] == "pitcher")
            & (all_props["stat"] == "K")
        ]
        if len(pit_k) > 0:
            logger.info("  Pitcher K projections:")
            for _, row in pit_k.iterrows():
                logger.info(
                    "    %s vs %s: %.1f K expected | "
                    "P(over %.1f)=%.0f%% | P(over %.1f)=%.0f%% | P(over %.1f)=%.0f%%",
                    row["player_name"], row["opponent"], row["expected"],
                    row["line_low"], row["p_over_low"] * 100,
                    row["line_mid"], row["p_over_mid"] * 100,
                    row["line_high"], row["p_over_high"] * 100,
                )


def _player_lines(
    expected: float,
    min_line: float = 0.5,
) -> list[float]:
    """Compute 3 prop lines centered on a player's expected value.

    Returns [low, mid, high] where mid is the nearest X.5 to expected.
    Example: expected=6.2 -> [5.5, 6.5, 7.5]
             expected=1.1 -> [0.5, 1.5, 2.5]
             expected=0.3 -> [0.5, 0.5, 1.5]  (clamped at min_line)
    """
    mid = np.floor(expected) + 0.5
    low = max(mid - 1.0, min_line)
    high = mid + 1.0
    return [low, mid, high]


def _save_empty(game_date: str) -> None:
    """Save empty props DataFrame."""
    pd.DataFrame(columns=[
        "game_date", "game_pk", "player_id", "player_name",
        "player_type", "team", "opponent", "stat",
        "expected", "std",
        "line_low", "line_mid", "line_high",
        "p_over_low", "p_over_mid", "p_over_high",
    ]).to_parquet(DASHBOARD_DIR / "game_props.parquet", index=False)


def _build_baselines_pt(
    pitcher_arsenal: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Build league baselines per pitch type."""
    agg = pitcher_arsenal.groupby("pitch_type").agg(
        total_whiffs=("whiffs", "sum"),
        total_swings=("swings", "sum"),
    ).reset_index()
    agg["whiff_rate"] = agg["total_whiffs"] / agg["total_swings"].replace(0, np.nan)
    return {
        row["pitch_type"]: {"whiff_rate": float(row["whiff_rate"])}
        for _, row in agg.iterrows()
        if pd.notna(row["whiff_rate"])
    }


def _get_lineup(
    roster: pd.DataFrame,
    lineup_priors: pd.DataFrame,
    team_id: int,
    n: int = 9,
) -> list[int]:
    """Get projected starting lineup for a team.

    Joins roster (for team membership) with lineup_priors (for batting
    order) to build the most likely starting lineup.
    """
    if roster.empty or team_id == 0:
        return []

    # roster uses org_id for team
    id_col = "org_id" if "org_id" in roster.columns else "team_id"
    team_roster = roster[roster[id_col] == team_id].copy()
    if team_roster.empty:
        return []

    # Filter to active non-pitchers
    if "roster_status" in team_roster.columns:
        team_roster = team_roster[team_roster["roster_status"] == "active"]
    if "primary_position" in team_roster.columns:
        team_roster = team_roster[
            ~team_roster["primary_position"].isin(["P", "SP", "RP"])
        ]

    if team_roster.empty:
        return []

    pids = team_roster["player_id"].values

    # Join with lineup priors for batting order
    if not lineup_priors.empty:
        lp = lineup_priors[
            lineup_priors["player_id"].isin(pids)
            & (lineup_priors["is_primary"] == True)  # noqa
        ].copy()

        if not lp.empty:
            # Sort by batting_order_mode to get projected lineup order
            lp = lp.sort_values("batting_order_mode")
            # Deduplicate (player may appear at multiple positions)
            lp = lp.drop_duplicates(subset="player_id", keep="first")
            return lp["player_id"].head(n).astype(int).tolist()

    # Fallback: return roster players in arbitrary order
    return team_roster["player_id"].head(n).astype(int).tolist()


def _compute_lineup_matchup_lifts(
    pitcher_id: int,
    batter_ids: list[int],
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    """Score matchups for lineup across K, BB, HR."""
    from src.models.matchup import score_matchup_for_stat

    lifts: dict[str, np.ndarray] = {
        "k": np.zeros(9), "bb": np.zeros(9), "hr": np.zeros(9),
    }
    for i, batter_id in enumerate(batter_ids[:9]):
        for stat in ("k", "bb", "hr"):
            try:
                result = score_matchup_for_stat(
                    stat, pitcher_id, batter_id,
                    pitcher_arsenal, hitter_vuln, baselines_pt,
                )
                val = result.get(f"matchup_{stat}_logit_lift", 0.0)
                if isinstance(val, float) and not np.isnan(val):
                    lifts[stat][i] = val
            except Exception:
                pass
    return lifts


def _posterior_mean(
    npz: np.lib.npyio.NpzFile,
    key: str,
    default: float,
) -> float:
    """Get posterior mean from NPZ, with fallback."""
    if key in npz:
        return float(np.mean(npz[key]))
    return default


def _lookup_name(
    player_teams: pd.DataFrame,
    player_id: int,
) -> str:
    """Look up player name from player_teams parquet."""
    if player_teams.empty:
        return str(player_id)
    name_col = "player_name" if "player_name" in player_teams.columns else "name"
    id_col = "player_id" if "player_id" in player_teams.columns else "batter_id"
    row = player_teams[player_teams[id_col] == player_id]
    if len(row) > 0 and name_col in row.columns:
        return str(row.iloc[0][name_col])
    return str(player_id)
