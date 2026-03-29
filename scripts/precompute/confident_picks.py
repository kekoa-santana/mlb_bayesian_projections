"""Precompute: Daily game prop projections from game simulator.

Runs pitcher and batter game sims for today's (and tomorrow's) games,
computing P(over) at player-relative lines for each stat.

Output: game_props.parquet — one row per player x stat with expected
value and P(over) at 3 lines centered on the player's projection.
Includes umpire/weather context, fantasy scoring, and all pitcher stats.

Replaces both the old game_props AND today_sims — single source of truth.

Requires: pre-computed posterior samples, exit model, matchup data,
bullpen rates, etc.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.special import logit as _logit

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
    from src.models.game_sim.simulator import simulate_game, compute_stamina_offset
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

    # Umpire tendencies → per-umpire K logit lift lookup
    ump_lookup: dict[str, float] = {}
    try:
        umpire_tend = pd.read_parquet(
            DASHBOARD_DIR / "umpire_tendencies.parquet"
        )
        for _, ur in umpire_tend.iterrows():
            ump_lookup[ur["hp_umpire_name"]] = float(ur["k_logit_lift"])
        logger.info("Loaded %d umpire tendencies", len(ump_lookup))
    except (FileNotFoundError, KeyError):
        logger.info("No umpire tendencies — skipping umpire adjustments")

    # Weather effects → (temp_bucket, wind_category) lookup
    wx_lookup: dict[tuple[str, str], dict] = {}
    try:
        wx_df = pd.read_parquet(DASHBOARD_DIR / "weather_effects.parquet")
        for _, wr in wx_df.iterrows():
            wx_lookup[(wr["temp_bucket"], wr["wind_category"])] = {
                "k_multiplier": float(wr["k_multiplier"]),
                "overall_k_rate": float(wr["overall_k_rate"]),
            }
        logger.info("Loaded %d weather effect combos", len(wx_lookup))
    except (FileNotFoundError, KeyError):
        logger.info("No weather effects — skipping weather adjustments")

    # Fantasy scoring
    from src.models.game_sim.fantasy_scoring import compute_pitcher_fantasy

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

        # --- Per-game context lifts ---
        hp_ump_name = game.get("hp_umpire_name", "")
        ump_k_lift = ump_lookup.get(hp_ump_name, 0.0) if hp_ump_name else 0.0

        wx_k_lift = 0.0
        temp_bucket = _parse_temp_bucket(game.get("weather_temp"))
        wind_cat = _parse_wind_category(game.get("weather_wind"))
        wx_info = wx_lookup.get((temp_bucket, wind_cat))
        if wx_info:
            k_mult = wx_info["k_multiplier"]
            overall_k = wx_info["overall_k_rate"]
            adj_k = np.clip(overall_k * k_mult, 1e-6, 1 - 1e-6)
            wx_k_lift = float(
                _logit(adj_k) - _logit(np.clip(overall_k, 1e-6, 1 - 1e-6))
            )

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

            # Pitcher avg pitches + stamina offset
            tend_row = tend_latest[tend_latest["pitcher_id"] == pitcher_id]
            avg_pitches = (
                float(tend_row.iloc[0]["avg_pitches"])
                if len(tend_row) > 0 else 88.0
            )
            avg_ip = (
                float(tend_row.iloc[0]["avg_ip"])
                if len(tend_row) > 0 and "avg_ip" in tend_row.columns
                and pd.notna(tend_row.iloc[0].get("avg_ip"))
                else 5.28
            )
            stamina_offset = compute_stamina_offset(avg_ip)

            # Run pitcher sim (with umpire/weather context)
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
                    exit_calibration_offset=stamina_offset,
                    umpire_k_lift=ump_k_lift,
                    weather_k_lift=wx_k_lift,
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

            # Fantasy scoring
            try:
                fantasy = compute_pitcher_fantasy(sim)
                dk = fantasy.dk_summary()
                espn = fantasy.espn_summary()
            except Exception:
                dk = {"mean": 0, "median": 0, "q10": 0, "q90": 0}
                espn = {"mean": 0, "median": 0}

            # Shared metadata for all stat rows from this pitcher
            _meta = {
                "game_date": game_dt,
                "game_pk": game_pk,
                "player_id": pitcher_id,
                "player_name": pitcher_name,
                "player_type": "pitcher",
                "team": team_abbr,
                "opponent": opp_abbr,
                "side": side,
                "umpire_k_lift": round(ump_k_lift, 4),
                "weather_k_lift": round(wx_k_lift, 4),
                "dk_mean": round(dk["mean"], 1),
                "dk_median": round(dk["median"], 1),
                "dk_q10": round(dk["q10"], 1),
                "dk_q90": round(dk["q90"], 1),
                "espn_mean": round(espn["mean"], 1),
                "espn_median": round(espn["median"], 1),
                "expected_ip": float(np.mean(sim.outs_samples)) / 3.0,
                "expected_pitches": round(summary["pitch_count"]["mean"], 0),
                "expected_bf": round(summary["bf"]["mean"], 1),
            }

            # Build props for each pitcher stat
            for stat_key, stat_label in [
                ("k", "K"), ("bb", "BB"), ("h", "H"),
                ("hr", "HR"), ("outs", "Outs"),
            ]:
                expected = summary[stat_key]["mean"]
                std = summary[stat_key]["std"]
                # Default line: nearest X.5 to expected
                default_line = max(np.floor(expected) + 0.5, 0.5)
                # Precompute p_over for all standard lines (for DK resolution)
                all_lines = [x + 0.5 for x in range(16)]
                over_df = sim.over_probs(stat_key, all_lines)
                p_over_map = dict(zip(over_df["line"], over_df["p_over"]))

                pitcher_picks.append({
                    **_meta,
                    "stat": stat_label,
                    "expected": round(expected, 2),
                    "std": round(std, 2),
                    "line": default_line,
                    "p_over": round(p_over_map.get(default_line, 0), 3),
                    "_p_over_map": p_over_map,
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

                # Build props for each batter stat
                for stat_key in BATTER_STATS:
                    expected = bsummary[stat_key]["mean"]
                    std = bsummary[stat_key]["std"]
                    default_line = max(np.floor(expected) + 0.5, 0.5)
                    all_lines = [x + 0.5 for x in range(16)]
                    over_df = bsim.over_probs(stat_key, all_lines)
                    p_over_map = dict(zip(over_df["line"], over_df["p_over"]))

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
                        "line": default_line,
                        "p_over": round(p_over_map.get(default_line, 0), 3),
                        "_p_over_map": p_over_map,
                    })

    # --- Combine new predictions ---
    all_props = pd.DataFrame(pitcher_picks + batter_picks)
    if len(all_props) > 0:
        all_props = all_props.sort_values(
            ["game_date", "player_type", "expected"],
            ascending=[True, True, False],
        )

    # --- Attach DraftKings odds as Vegas baseline ---
    all_props["vegas_line"] = None
    all_props["vegas_odds"] = None
    all_props["vegas_implied"] = None
    try:
        _dk_path = DASHBOARD_DIR.parent.parent / "lib" / "draftkings.py"
        if _dk_path.exists():
            import importlib.util
            _spec = importlib.util.spec_from_file_location("draftkings", _dk_path)
            _dk_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_dk_mod)
            dk_df = _dk_mod.fetch_dk_player_props()
        else:
            from lib.draftkings import fetch_dk_player_props
            dk_df = fetch_dk_player_props()

        if not dk_df.empty and len(all_props) > 0:
            # Build name lookup from projections
            _name_lookup: dict[int, str] = {}
            try:
                _pp = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")
                if "pitcher_name" in _pp.columns:
                    for _, r in _pp.iterrows():
                        _name_lookup[int(r["pitcher_id"])] = r["pitcher_name"]
            except FileNotFoundError:
                pass
            try:
                _hp = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")
                if "batter_name" in _hp.columns:
                    for _, r in _hp.iterrows():
                        _name_lookup[int(r["batter_id"])] = r["batter_name"]
            except FileNotFoundError:
                pass

            # Index DK props by (last_name, stat) for matching
            dk_by_stat: dict[str, list] = {}
            for _, dr in dk_df.iterrows():
                key = dr["stat"]
                if key not in dk_by_stat:
                    dk_by_stat[key] = []
                dk_by_stat[key].append(dr)

            for idx, row in all_props.iterrows():
                pid = int(row["player_id"])
                name = _name_lookup.get(pid, "")
                if not name:
                    continue
                last = name.split()[-1].lower()
                stat = row["stat"]
                for dr in dk_by_stat.get(stat, []):
                    dk_last = dr["player_name"].split()[-1].lower()
                    if last == dk_last:
                        all_props.at[idx, "vegas_line"] = dr["line"]
                        all_props.at[idx, "vegas_odds"] = dr["over_odds"]
                        all_props.at[idx, "vegas_implied"] = dr["over_implied"]
                        # Resolve line + p_over to the DK line
                        dk_line = float(dr["line"])
                        p_map = row.get("_p_over_map")
                        if isinstance(p_map, dict) and dk_line in p_map:
                            all_props.at[idx, "line"] = dk_line
                            all_props.at[idx, "p_over"] = round(
                                p_map[dk_line], 3)
                        break

            n_matched = all_props["vegas_line"].notna().sum()
            logger.info("Matched %d/%d props to DraftKings odds",
                        n_matched, len(all_props))
    except Exception:
        logger.debug("DraftKings odds fetch failed", exc_info=True)

    # Drop the temporary p_over map column
    if "_p_over_map" in all_props.columns:
        all_props.drop(columns=["_p_over_map"], inplace=True)

    # --- Merge with history: freeze started/finished, update pre-game ---
    # Props for games that have started or finished are never overwritten.
    # Props for pre-game slots are replaced with the latest predictions
    # (lineup changes, model updates).  Results are backfilled from the
    # MLB boxscore API for completed games.
    history_path = DASHBOARD_DIR / "game_props.parquet"
    if history_path.exists():
        existing = pd.read_parquet(history_path)
    else:
        existing = pd.DataFrame()

    if existing.empty:
        merged = all_props.copy()
        if "actual" not in merged.columns:
            merged["actual"] = None
            merged["over_hit"] = None
            merged["game_status"] = "scheduled"
    else:
        # Ensure history columns exist
        for col in ("actual", "over_hit", "game_status"):
            if col not in existing.columns:
                existing[col] = None

        # Frozen rows: games that already started (have results or
        # were marked in-progress / final).  These never change.
        frozen_mask = existing["game_status"].isin(
            ["in_progress", "final"]
        )
        frozen = existing[frozen_mask].copy()

        # Updatable rows: everything else in history (scheduled games
        # that haven't started yet).
        updatable_gpks = set(
            existing.loc[~frozen_mask, "game_pk"].unique()
        )

        # New predictions replace updatable rows and add new game_pks
        if len(all_props) > 0:
            fresh = all_props.copy()
            fresh["actual"] = None
            fresh["over_hit"] = None
            fresh["game_status"] = "scheduled"
        else:
            fresh = pd.DataFrame()

        # Keep frozen history + fresh predictions (replacing stale scheduled)
        merged = pd.concat(
            [frozen, fresh], ignore_index=True,
        ).drop_duplicates(
            subset=["game_pk", "player_id", "stat"],
            keep="first",  # frozen rows come first → preserved
        )

    # --- Backfill results for completed games ---
    _backfill_results(merged)

    merged.to_parquet(history_path, index=False)

    n_frozen = (merged["game_status"] == "final").sum()
    n_scheduled = (merged["game_status"] == "scheduled").sum()
    logger.info(
        "Saved %d prop projections (%d pitcher, %d batter) "
        "to game_props.parquet [%d final, %d scheduled]",
        len(merged), len(pitcher_picks), len(batter_picks),
        n_frozen, n_scheduled,
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
                    "Line %.1f | P(over)=%.0f%%",
                    row["player_name"], row["opponent"], row["expected"],
                    row["line"], row["p_over"] * 100,
                )


def _backfill_results(props_df: pd.DataFrame) -> None:
    """Fill in actual stats and over/under results from MLB boxscores.

    Fetches live boxscore data for any game that has started but doesn't
    have results yet.  Modifies ``props_df`` in place.
    """
    import json
    import urllib.request

    if props_df.empty:
        return

    _STAT_TO_BOX = {
        "K": "strikeOuts", "H": "hits", "HR": "homeRuns",
        "BB": "baseOnBalls", "TB": "totalBases", "Outs": None,
    }

    # Find games that need results: scheduled (no results yet)
    needs_results = props_df[
        (props_df["game_status"] != "final")
    ]["game_pk"].unique()

    if len(needs_results) == 0:
        return

    for gpk in needs_results:
        url = f"https://statsapi.mlb.com/api/v1/game/{int(gpk)}/boxscore"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            continue

        # Determine game status from linescore
        try:
            live_url = f"https://statsapi.mlb.com/api/v1.1/game/{int(gpk)}/feed/live"
            with urllib.request.urlopen(live_url, timeout=15) as resp:
                live_data = json.loads(resp.read().decode())
            status = live_data.get("gameData", {}).get(
                "status", {}
            ).get("detailedState", "")
        except Exception:
            status = ""

        is_final = "final" in status.lower() or "game over" in status.lower()
        is_live = "in progress" in status.lower()

        if not is_final and not is_live:
            continue  # game hasn't started yet

        game_status = "final" if is_final else "in_progress"

        # Build player stats lookup from boxscore
        player_stats: dict[int, dict] = {}
        for side in ("away", "home"):
            team_data = data.get("teams", {}).get(side, {})
            players = team_data.get("players", {})
            for pid_key, pdata in players.items():
                person = pdata.get("person", {})
                pid = person.get("id")
                stats = pdata.get("stats", {})
                batting = stats.get("batting", {})
                pitching = stats.get("pitching", {})
                pstats: dict[str, float] = {}
                if batting:
                    for stat_name, box_key in _STAT_TO_BOX.items():
                        if box_key and box_key in batting:
                            pstats[stat_name] = float(batting[box_key])
                if pitching and pitching.get("inningsPitched"):
                    for stat_name, box_key in _STAT_TO_BOX.items():
                        if box_key and box_key in pitching:
                            pstats[stat_name] = float(pitching[box_key])
                    # Outs from IP
                    try:
                        ip = float(pitching["inningsPitched"])
                        pstats["Outs"] = float(
                            int(ip) * 3 + round((ip % 1) * 10)
                        )
                    except (ValueError, TypeError):
                        pass
                if pstats:
                    player_stats[pid] = pstats

        # Fill in results for this game
        game_mask = props_df["game_pk"] == gpk
        for idx in props_df.index[game_mask]:
            pid = int(props_df.at[idx, "player_id"])
            stat = props_df.at[idx, "stat"]
            line = props_df.at[idx, "line"]

            props_df.at[idx, "game_status"] = game_status

            if pid in player_stats and stat in player_stats[pid]:
                actual = player_stats[pid][stat]
                props_df.at[idx, "actual"] = actual
                props_df.at[idx, "over_hit"] = actual > line

    n_filled = props_df["actual"].notna().sum()
    logger.info("Backfilled results: %d rows with actuals", n_filled)



def _save_empty(game_date: str) -> None:
    """Save empty props DataFrame (preserves existing history)."""
    history_path = DASHBOARD_DIR / "game_props.parquet"
    if history_path.exists():
        # Don't overwrite history with empty — just backfill results
        existing = pd.read_parquet(history_path)
        for col in ("actual", "over_hit", "game_status"):
            if col not in existing.columns:
                existing[col] = None
        _backfill_results(existing)
        existing.to_parquet(history_path, index=False)
        return
    pd.DataFrame(columns=[
        "game_date", "game_pk", "player_id", "player_name",
        "player_type", "team", "opponent", "side", "stat",
        "expected", "std",
        "line", "p_over",
        "umpire_k_lift", "weather_k_lift",
        "dk_mean", "dk_median", "dk_q10", "dk_q90",
        "espn_mean", "espn_median",
        "expected_ip", "expected_pitches", "expected_bf",
        "actual", "over_hit", "game_status",
    ]).to_parquet(history_path, index=False)


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


def _parse_temp_bucket(temp) -> str:
    """Convert temperature to bucket string."""
    if temp is None or (isinstance(temp, float) and np.isnan(temp)):
        return "moderate"
    try:
        t = float(temp)
    except (TypeError, ValueError):
        return "moderate"
    if t < 50:
        return "cold"
    if t < 70:
        return "cool"
    if t < 85:
        return "moderate"
    return "hot"


def _parse_wind_category(wind) -> str:
    """Convert wind speed to category string."""
    if wind is None or (isinstance(wind, float) and np.isnan(wind)):
        return "calm"
    try:
        w = float(wind)
    except (TypeError, ValueError):
        return "calm"
    if w < 5:
        return "calm"
    if w < 12:
        return "moderate"
    return "windy"
