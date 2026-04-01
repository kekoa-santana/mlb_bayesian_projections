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
# Lineup sim: stats that benefit from base-state tracking (R, RBI, HRR)
# Per-batter sim: stats that only depend on batter hit types (TB)
LINEUP_SIM_STATS = ("h", "k", "bb", "r", "rbi", "hrr")
BATTER_SIM_STATS = ("tb",)
BATTER_STATS = LINEUP_SIM_STATS + BATTER_SIM_STATS


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
    from src.models.game_sim.lineup_simulator import simulate_lineup_game
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

    # --- Fetch confirmed lineups from MLB API ---
    confirmed_lineups: dict[tuple[int, int], list[int]] = {}
    try:
        from src.data.schedule import fetch_game_lineups
        for gpk in schedule["game_pk"].unique():
            lu = fetch_game_lineups(int(gpk))
            if not lu.empty and "batting_order" in lu.columns:
                for tid, grp in lu.groupby("team_id"):
                    ordered = grp.sort_values("batting_order")
                    bids = ordered["batter_id"].astype(int).tolist()[:9]
                    if bids:
                        confirmed_lineups[(int(gpk), int(tid))] = bids
        if confirmed_lineups:
            n_teams = len(confirmed_lineups)
            logger.info("Fetched confirmed lineups for %d team-games", n_teams)
        else:
            logger.info("No confirmed lineups available yet")
    except Exception:
        logger.warning("Failed to fetch confirmed lineups", exc_info=True)

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

    # Roster + lineup priors for projected lineups (roster also used for name lookups)
    try:
        roster = pd.read_parquet(DASHBOARD_DIR / "roster.parquet")
    except FileNotFoundError:
        roster = pd.DataFrame()
    try:
        lineup_priors = pd.read_parquet(DASHBOARD_DIR / "lineup_priors.parquet")
    except FileNotFoundError:
        lineup_priors = pd.DataFrame()

    # LD%-based BABIP adjustments for batter sim
    ld_babip_lookup: dict[int, float] = {}
    _LD_BABIP_COEFFICIENT = 0.25
    _LEAGUE_LD_RATE = 0.22
    try:
        ld_df = pd.read_parquet(DASHBOARD_DIR / "batter_ld_rate.parquet")
        if not ld_df.empty:
            ld_latest = (
                ld_df.sort_values("season")
                .groupby("player_id").last().reset_index()
            )
            for _, lr in ld_latest.iterrows():
                ld_dev = float(lr["ld_rate_regressed"]) - _LEAGUE_LD_RATE
                ld_babip_lookup[int(lr["player_id"])] = ld_dev * _LD_BABIP_COEFFICIENT
            logger.info(
                "Loaded LD%% BABIP adjustments for %d batters",
                len(ld_babip_lookup),
            )
    except FileNotFoundError:
        logger.info("No batter_ld_rate.parquet found; LD BABIP adjustments disabled")

    # Sprint speed BABIP adjustments (stacks with LD%)
    speed_babip_lookup: dict[int, float] = {}
    _LEAGUE_SPRINT_SPEED = 27.0
    _SPEED_BABIP_COEFFICIENT = 0.010
    try:
        speed_df = pd.read_parquet(DASHBOARD_DIR / "batter_sprint_speed.parquet")
        if not speed_df.empty:
            speed_latest = (
                speed_df.sort_values("season")
                .groupby("player_id").last().reset_index()
            )
            for _, sr in speed_latest.iterrows():
                speed_dev = float(sr["sprint_speed_regressed"]) - _LEAGUE_SPRINT_SPEED
                speed_babip_lookup[int(sr["player_id"])] = (
                    speed_dev * _SPEED_BABIP_COEFFICIENT
                )
            logger.info(
                "Loaded sprint speed BABIP adjustments for %d batters",
                len(speed_babip_lookup),
            )
    except FileNotFoundError:
        logger.info(
            "No batter_sprint_speed.parquet found; speed BABIP adjustments disabled"
        )

    # Park factor lifts (K and HR only; H/BB skipped due to model H bias)
    park_lift_lookup: dict[int, dict[str, float]] = {}
    try:
        pf_lifts = pd.read_parquet(DASHBOARD_DIR / "park_factor_lifts.parquet")
        for _, pr in pf_lifts.iterrows():
            park_lift_lookup[int(pr["venue_id"])] = {
                "k_lift": float(pr["k_lift"]),
                "hr_lift": float(pr["hr_lift"]),
            }
        logger.info("Loaded park factor lifts for %d venues", len(park_lift_lookup))
    except (FileNotFoundError, KeyError):
        logger.info("No park_factor_lifts.parquet; park adjustments disabled")

    # Umpire tendencies -> per-umpire K and BB logit lift lookups
    ump_k_lookup: dict[str, float] = {}
    ump_bb_lookup: dict[str, float] = {}
    try:
        umpire_tend = pd.read_parquet(
            DASHBOARD_DIR / "umpire_tendencies.parquet"
        )
        for _, ur in umpire_tend.iterrows():
            ump_k_lookup[ur["hp_umpire_name"]] = float(ur["k_logit_lift"])
            ump_bb_lookup[ur["hp_umpire_name"]] = float(
                ur.get("bb_logit_lift", 0.0)
            )
        logger.info(
            "Loaded %d umpire tendencies (K + BB lifts)", len(ump_k_lookup)
        )
    except (FileNotFoundError, KeyError):
        logger.info("No umpire tendencies -- skipping umpire adjustments")

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

    # Catcher framing effects → catcher_id lookup
    catcher_framing_lookup: dict[int, float] = {}
    try:
        framing_df = pd.read_parquet(
            DASHBOARD_DIR / "catcher_framing.parquet"
        )
        # Use most recent season per catcher
        framing_latest = (
            framing_df.sort_values("season")
            .groupby("catcher_id").last().reset_index()
        )
        for _, fr in framing_latest.iterrows():
            catcher_framing_lookup[int(fr["catcher_id"])] = float(
                fr.get("logit_lift", 0.0)
            )
        logger.info(
            "Loaded catcher framing for %d catchers", len(catcher_framing_lookup)
        )
    except (FileNotFoundError, KeyError):
        logger.info("No catcher framing data — skipping framing adjustments")

    # Fantasy scoring
    from src.models.game_sim.fantasy_scoring import compute_pitcher_fantasy

    # Build pitch type baselines for matchup scoring
    from src.data.league_baselines import get_baselines_dict
    baselines_pt = get_baselines_dict(grouping="pitch_type", recency_weights="marcel")

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

    # Platoon splits for platoon-aware lineup proneness
    platoon_splits = pd.DataFrame()
    pitcher_hand_lookup: dict[int, str] = {}
    try:
        from src.data.feature_eng import get_cached_season_totals_by_pitcher_hand
        platoon_splits = get_cached_season_totals_by_pitcher_hand(last_train)
        logger.info("Loaded platoon splits: %d batter-hand rows", len(platoon_splits))
    except Exception as e:
        logger.info("No platoon splits: %s", e)
    if "pitch_hand" in pitcher_arsenal.columns:
        hand_df = pitcher_arsenal[["pitcher_id", "pitch_hand"]].drop_duplicates()
        for _, hr in hand_df.iterrows():
            pitcher_hand_lookup[int(hr["pitcher_id"])] = str(hr["pitch_hand"])
        logger.info("Built pitcher hand lookup: %d pitchers", len(pitcher_hand_lookup))

    # --- Run sims per game ---
    pitcher_picks = []
    batter_picks = []

    for _, game in schedule.iterrows():
        game_pk = int(game["game_pk"])

        # --- Per-game context lifts ---
        hp_ump_name = game.get("hp_umpire_name", "")
        ump_k_lift = ump_k_lookup.get(hp_ump_name, 0.0) if hp_ump_name else 0.0
        ump_bb_lift = ump_bb_lookup.get(hp_ump_name, 0.0) if hp_ump_name else 0.0

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

        # Park factor lifts (K and HR only)
        venue_id = game.get("venue_id")
        park_lifts = park_lift_lookup.get(int(venue_id), {}) if pd.notna(venue_id) else {}
        park_k_lift = park_lifts.get("k_lift", 0.0)
        park_hr_lift = park_lifts.get("hr_lift", 0.0)

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

            # Get opposing lineup: prefer confirmed, fall back to priors
            opp_team_id = game.get(opp_team_col)
            opp_tid = int(opp_team_id) if pd.notna(opp_team_id) else 0
            lineup_ids = confirmed_lineups.get(
                (game_pk, opp_tid),
                _get_lineup(roster, lineup_priors, opp_tid),
            )
            if len(lineup_ids) < 9:
                # Pad with zeros for missing lineup slots
                lineup_ids = lineup_ids + [0] * (9 - len(lineup_ids))

            # Matchup lifts
            matchup_lifts = _compute_lineup_matchup_lifts(
                pitcher_id, lineup_ids, pitcher_arsenal,
                hitter_vuln, baselines_pt,
            )

            # Platoon-adjusted lineup proneness (extra context lift)
            platoon_k_lift = 0.0
            platoon_bb_lift = 0.0
            p_hand = pitcher_hand_lookup.get(pitcher_id)
            if p_hand and not platoon_splits.empty:
                from src.models.lineup_adjustments import compute_platoon_adjusted_proneness
                # Build batter posteriors for this lineup
                lu_k_post = {
                    bid: hitter_k_npz[str(bid)]
                    for bid in lineup_ids if str(bid) in hitter_k_npz
                }
                lu_bb_post = {
                    bid: hitter_bb_npz[str(bid)]
                    for bid in lineup_ids if str(bid) in hitter_bb_npz
                }
                platoon_k_lift = compute_platoon_adjusted_proneness(
                    lineup_ids, lu_k_post, platoon_splits, p_hand,
                    league_avg_rate=0.224, stat_name="k", weight=0.3,
                )
                platoon_bb_lift = compute_platoon_adjusted_proneness(
                    lineup_ids, lu_bb_post, platoon_splits, p_hand,
                    league_avg_rate=0.083, stat_name="bb", weight=0.3,
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

            # Catcher framing lift (pitcher's own team's catcher)
            catcher_k_lift = 0.0
            pitcher_team_id = int(game.get(team_col, 0)) if pd.notna(game.get(team_col)) else 0
            if catcher_framing_lookup and pitcher_team_id and not roster.empty:
                id_col_r = "org_id" if "org_id" in roster.columns else "team_id"
                _has_pos = "primary_position" in roster.columns
                team_catchers = roster[
                    (roster[id_col_r] == pitcher_team_id)
                    & (roster["primary_position"] == "C" if _has_pos else False)
                ]
                if not team_catchers.empty:
                    catcher_id = int(team_catchers.iloc[0].get("player_id", 0))
                    catcher_k_lift = catcher_framing_lookup.get(catcher_id, 0.0)

            # Run pitcher sim (with umpire/weather/catcher context)
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
                    umpire_k_lift=ump_k_lift + catcher_k_lift + platoon_k_lift,
                    umpire_bb_lift=ump_bb_lift + platoon_bb_lift,
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
                "umpire_bb_lift": round(ump_bb_lift, 4),
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

                # Store P(over) at standard lines 0.5-10.5
                p_over_cols = {}
                for lv in [x + 0.5 for x in range(11)]:
                    p_over_cols[f"p_over_{lv:.1f}"] = round(
                        p_over_map.get(lv, 0), 3
                    )

                pitcher_picks.append({
                    **_meta,
                    "stat": stat_label,
                    "expected": round(expected, 2),
                    "std": round(std, 2),
                    "line": default_line,
                    "p_over": round(p_over_map.get(default_line, 0), 3),
                    **p_over_cols,
                })

        # --- BATTER SIMS (both teams' lineups via lineup simulator) ---
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

            # Get lineup: prefer confirmed, fall back to priors
            lineup_ids = confirmed_lineups.get(
                (game_pk, team_id),
                _get_lineup(roster, lineup_priors, team_id),
            )
            team_abbr = game.get(team_abbr_col, "")
            opp_abbr = game.get(opp_abbr_col, "")

            # --- Gather posteriors for all 9 lineup slots ---
            lineup_k_samples: list[np.ndarray] = []
            lineup_bb_samples: list[np.ndarray] = []
            lineup_hr_samples: list[np.ndarray] = []
            valid_slots: dict[int, tuple[int, int]] = {}
            _FALLBACK_N = 500

            for order_idx, batter_id in enumerate(lineup_ids[:9]):
                bid_str = str(batter_id)
                bk = hitter_k_npz.get(bid_str)
                bbb = hitter_bb_npz.get(bid_str)
                bhr = hitter_hr_npz.get(bid_str)
                if bk is not None and bbb is not None and bhr is not None:
                    lineup_k_samples.append(bk)
                    lineup_bb_samples.append(bbb)
                    lineup_hr_samples.append(bhr)
                    valid_slots[order_idx] = (batter_id, order_idx + 1)
                else:
                    # League-average fallback for missing batters
                    lineup_k_samples.append(np.full(_FALLBACK_N, 0.226))
                    lineup_bb_samples.append(np.full(_FALLBACK_N, 0.082))
                    lineup_hr_samples.append(np.full(_FALLBACK_N, 0.031))

            # Pad to 9 if lineup is short
            while len(lineup_k_samples) < 9:
                lineup_k_samples.append(np.full(_FALLBACK_N, 0.226))
                lineup_bb_samples.append(np.full(_FALLBACK_N, 0.082))
                lineup_hr_samples.append(np.full(_FALLBACK_N, 0.031))

            if not valid_slots:
                continue

            # Matchup lifts for full lineup vs opposing starter
            padded_ids = (lineup_ids[:9] + [0] * 9)[:9]
            batter_matchup_lifts = _compute_lineup_matchup_lifts(
                opp_pitcher_id, padded_ids,
                pitcher_arsenal, hitter_vuln, baselines_pt,
            )

            # Build per-batter BABIP adjustments (LD% + sprint speed, additive)
            padded_bids = (lineup_ids[:9] + [0] * 9)[:9]
            lineup_babip_adjs = np.array([
                ld_babip_lookup.get(bid, 0.0) + speed_babip_lookup.get(bid, 0.0)
                for bid in padded_bids
            ])

            # Run lineup simulation (single call for all 9 batters)
            try:
                lineup_sim = simulate_lineup_game(
                    batter_k_rate_samples=lineup_k_samples,
                    batter_bb_rate_samples=lineup_bb_samples,
                    batter_hr_rate_samples=lineup_hr_samples,
                    starter_k_rate=starter_k,
                    starter_bb_rate=starter_bb,
                    starter_hr_rate=starter_hr,
                    starter_bf_mu=bf_mu,
                    starter_bf_sigma=bf_sigma,
                    matchup_k_lifts=batter_matchup_lifts["k"],
                    matchup_bb_lifts=batter_matchup_lifts["bb"],
                    matchup_hr_lifts=batter_matchup_lifts["hr"],
                    bullpen_k_rate=bp_k,
                    bullpen_bb_rate=bp_bb,
                    bullpen_hr_rate=bp_hr,
                    batter_babip_adjs=lineup_babip_adjs,
                    umpire_k_lift=ump_k_lift,
                    umpire_bb_lift=ump_bb_lift,
                    park_k_lift=park_k_lift,
                    park_hr_lift=park_hr_lift,
                    weather_k_lift=wx_k_lift,
                    n_sims=n_sims,
                    random_seed=42 + game_pk % 10000,
                )
            except Exception as e:
                logger.warning(
                    "Lineup sim failed: game %d %s: %s",
                    game_pk, side, e,
                )
                continue

            # Extract per-batter results for valid slots
            game_dt = game.get("game_date", game_date)
            for slot_idx, (batter_id, batting_order) in valid_slots.items():
                lineup_summary = lineup_sim.batter_summary(slot_idx)
                batter_name = _lookup_name(roster, batter_id)

                # Per-batter sim for TB (uses calibrated PA model,
                # avoids the lineup sim's PA inflation that biases TB)
                bid_str = str(batter_id)
                try:
                    batter_sim = simulate_batter_game(
                        batter_k_rate_samples=hitter_k_npz[bid_str],
                        batter_bb_rate_samples=hitter_bb_npz[bid_str],
                        batter_hr_rate_samples=hitter_hr_npz[bid_str],
                        batting_order=batting_order,
                        starter_k_rate=starter_k,
                        starter_bb_rate=starter_bb,
                        starter_hr_rate=starter_hr,
                        starter_bf_mu=bf_mu,
                        starter_bf_sigma=bf_sigma,
                        matchup_k_lift=float(
                            batter_matchup_lifts["k"][slot_idx]
                        ),
                        matchup_bb_lift=float(
                            batter_matchup_lifts["bb"][slot_idx]
                        ),
                        matchup_hr_lift=float(
                            batter_matchup_lifts["hr"][slot_idx]
                        ),
                        bullpen_k_rate=bp_k,
                        bullpen_bb_rate=bp_bb,
                        bullpen_hr_rate=bp_hr,
                        batter_babip_adj=ld_babip_lookup.get(batter_id, 0.0),
                        umpire_k_lift=ump_k_lift,
                        umpire_bb_lift=ump_bb_lift,
                        weather_k_lift=wx_k_lift,
                        n_sims=n_sims,
                        random_seed=(
                            42 + game_pk % 10000 + batter_id % 1000
                        ),
                    )
                    batter_summary = batter_sim.summary()
                except Exception:
                    batter_summary = None

                for stat_key in BATTER_STATS:
                    # TB uses per-batter sim; all others use lineup sim
                    if stat_key in BATTER_SIM_STATS:
                        if batter_summary is None:
                            continue
                        expected = batter_summary[stat_key]["mean"]
                        std = batter_summary[stat_key]["std"]
                        all_lines = [x + 0.5 for x in range(16)]
                        over_df = batter_sim.over_probs(
                            stat_key, all_lines,
                        )
                    else:
                        expected = lineup_summary[stat_key]["mean"]
                        std = lineup_summary[stat_key]["std"]
                        all_lines = [x + 0.5 for x in range(16)]
                        over_df = lineup_sim.batter_over_probs(
                            slot_idx, stat_key, all_lines,
                        )

                    default_line = max(np.floor(expected) + 0.5, 0.5)
                    p_over_map = dict(
                        zip(over_df["line"], over_df["p_over"])
                    )

                    # Store P(over) at standard lines 0.5-10.5
                    p_over_cols = {}
                    for lv in [x + 0.5 for x in range(11)]:
                        p_over_cols[f"p_over_{lv:.1f}"] = round(
                            p_over_map.get(lv, 0), 3
                        )

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
                        "p_over": round(
                            p_over_map.get(default_line, 0), 3
                        ),
                        **p_over_cols,
                    })

    # --- Combine new predictions ---
    all_props = pd.DataFrame(pitcher_picks + batter_picks)
    if len(all_props) > 0:
        all_props = all_props.sort_values(
            ["game_date", "player_type", "expected"],
            ascending=[True, True, False],
        )

    # Fall back to mid line for legacy rows
    if "line_mid" in all_props.columns:
        all_props["line"] = all_props["line"].fillna(all_props["line_mid"])
        all_props["p_over"] = all_props["p_over"].fillna(all_props["p_over_mid"])

    # --- Fetch and save book props (DK + Prize Picks) ---
    dk_resolved, pp_resolved = _fetch_book_props(all_props, game_date=game_date)

    # --- Merge vegas_line + odds into all_props ---
    # DK is primary source; PP standard lines fill gaps (e.g. HRR).
    if len(all_props) > 0:
        # DK: one line per (player_id, stat) — line + odds + implied
        if not dk_resolved.empty and "player_id" in dk_resolved.columns:
            dk_cols = ["player_id", "stat", "line"]
            if "over_odds" in dk_resolved.columns:
                dk_cols.append("over_odds")
            if "over_implied" in dk_resolved.columns:
                dk_cols.append("over_implied")
            dk_lines = (
                dk_resolved[dk_cols]
                .drop_duplicates(subset=["player_id", "stat"], keep="first")
                .rename(columns={
                    "line": "vegas_line",
                    "over_odds": "vegas_odds",
                    "over_implied": "vegas_implied",
                })
            )
            all_props = all_props.merge(
                dk_lines, on=["player_id", "stat"], how="left",
                suffixes=("", "_dk"),
            )
            for col in ("vegas_line", "vegas_odds", "vegas_implied"):
                dk_col = f"{col}_dk"
                if dk_col in all_props.columns:
                    all_props[col] = all_props[col].fillna(all_props[dk_col])
                    all_props.drop(columns=[dk_col], inplace=True)

        for col in ("vegas_line", "vegas_odds", "vegas_implied"):
            if col not in all_props.columns:
                all_props[col] = None

        # PP standard lines fill remaining gaps (HRR, etc.)
        if not pp_resolved.empty and "player_id" in pp_resolved.columns:
            pp_std = pp_resolved[
                pp_resolved["odds_type"] == "standard"
            ] if "odds_type" in pp_resolved.columns else pp_resolved
            if not pp_std.empty:
                pp_lines = (
                    pp_std[["player_id", "stat", "line"]]
                    .drop_duplicates(subset=["player_id", "stat"], keep="first")
                    .rename(columns={"line": "vegas_line_pp"})
                )
                all_props = all_props.merge(
                    pp_lines, on=["player_id", "stat"], how="left",
                )
                all_props["vegas_line"] = all_props["vegas_line"].fillna(
                    all_props["vegas_line_pp"]
                )
                all_props.drop(columns=["vegas_line_pp"], inplace=True)

        # --- Compute model edge at book line ---
        # model_edge = model P(over at vegas_line) - book implied P(over)
        # Positive edge = model thinks over is more likely than the book does
        all_props["model_p_over"] = np.nan
        has_vl = all_props["vegas_line"].notna()
        for idx in all_props.index[has_vl]:
            vl = float(all_props.at[idx, "vegas_line"])
            col = f"p_over_{round(vl * 2) / 2:.1f}"
            if col in all_props.columns and pd.notna(all_props.at[idx, col]):
                all_props.at[idx, "model_p_over"] = float(all_props.at[idx, col])
            else:
                # Fall back to legacy triplet
                for lv_col, p_col in (
                    ("line_low", "p_over_low"),
                    ("line_mid", "p_over_mid"),
                    ("line_high", "p_over_high"),
                ):
                    if (lv_col in all_props.columns and p_col in all_props.columns
                            and pd.notna(all_props.at[idx, lv_col])
                            and abs(float(all_props.at[idx, lv_col]) - vl) < 0.01
                            and pd.notna(all_props.at[idx, p_col])):
                        all_props.at[idx, "model_p_over"] = float(all_props.at[idx, p_col])
                        break

        # model_edge: positive = over is underpriced by book
        all_props["model_edge"] = np.nan
        has_both = all_props["model_p_over"].notna() & all_props["vegas_implied"].notna()
        all_props.loc[has_both, "model_edge"] = (
            all_props.loc[has_both, "model_p_over"].astype(float)
            - all_props.loc[has_both, "vegas_implied"].astype(float)
        )

        n_with_line = all_props["vegas_line"].notna().sum()
        n_with_edge = all_props["model_edge"].notna().sum()
        logger.info(
            "Vegas merge: %d / %d with lines, %d with edge",
            n_with_line, len(all_props), n_with_edge,
        )

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


def _fetch_book_props(
    all_props: pd.DataFrame,
    game_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch DraftKings and Prize Picks lines, save snapshots + history.

    Resolves book player names to MLB player IDs so downstream joins
    are by (player_id, stat) instead of fuzzy name matching.

    Returns resolved (dk_df, pp_df) so the caller can merge vegas_line
    into the main props DataFrame.
    """
    from datetime import date as _date
    if game_date is None:
        game_date = _date.today().isoformat()
    # Build name lookup: MLB player_id -> name
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

    # Build reverse lookup: match_name -> player_id
    _SUFFIXES = {"jr.", "sr.", "ii", "iii", "iv", "v"}

    def _match_name(full_name: str) -> str:
        parts = full_name.strip().split()
        parts = [p for p in parts if p.lower().rstrip(".") not in _SUFFIXES]
        if len(parts) >= 2:
            return f"{parts[0]} {parts[-1]}".lower()
        return full_name.lower()

    name_to_id: dict[str, int] = {}
    for pid, name in _name_lookup.items():
        key = _match_name(name)
        name_to_id[key] = pid

    def _resolve_id(book_name: str) -> int | None:
        return name_to_id.get(_match_name(book_name))

    # --- DraftKings ---
    dk_resolved = pd.DataFrame()
    pp_resolved = pd.DataFrame()
    try:
        _dk_path = DASHBOARD_DIR.parent.parent / "lib" / "draftkings.py"
        if _dk_path.exists():
            import importlib.util
            _spec = importlib.util.spec_from_file_location("draftkings", _dk_path)
            _dk_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_dk_mod)
            dk_df = _dk_mod.fetch_dk_player_props()
        else:
            dk_df = pd.DataFrame()

        if not dk_df.empty:
            dk_df["player_id"] = dk_df["player_name"].apply(_resolve_id)
            dk_df = dk_df[dk_df["player_id"].notna()].copy()
            dk_df["player_id"] = dk_df["player_id"].astype(int)
            dk_df["game_date"] = game_date
            dk_df.to_parquet(DASHBOARD_DIR / "dk_props.parquet", index=False)
            # Append to history
            _dk_hist_path = DASHBOARD_DIR / "dk_props_history.parquet"
            if _dk_hist_path.exists():
                _dk_hist = pd.read_parquet(_dk_hist_path)
                # Drop any existing rows for this date to avoid duplicates
                _dk_hist = _dk_hist[_dk_hist["game_date"] != game_date]
                dk_df = pd.concat([_dk_hist, dk_df], ignore_index=True)
            dk_df.to_parquet(_dk_hist_path, index=False)
            # Re-filter to today only for return value
            dk_resolved = dk_df[dk_df["game_date"] == game_date].copy()
            logger.info(
                "Saved %d DK props (%d resolved to MLB IDs), "
                "history now %d rows",
                len(dk_resolved), dk_resolved["player_id"].notna().sum(),
                len(dk_df),
            )
        else:
            logger.info("No DK props available")
    except Exception:
        logger.warning("DK props fetch failed", exc_info=True)

    # --- Prize Picks ---
    try:
        _pp_path = DASHBOARD_DIR.parent.parent / "lib" / "prizepicks.py"
        if _pp_path.exists():
            import importlib.util
            _spec = importlib.util.spec_from_file_location("prizepicks", _pp_path)
            _pp_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_pp_mod)
            pp_df = _pp_mod.fetch_pp_player_props()
        else:
            pp_df = pd.DataFrame()

        if not pp_df.empty:
            pp_df["player_id"] = pp_df["player_name"].apply(_resolve_id)
            pp_df = pp_df[pp_df["player_id"].notna()].copy()
            pp_df["player_id"] = pp_df["player_id"].astype(int)
            pp_df["game_date"] = game_date
            pp_df.to_parquet(DASHBOARD_DIR / "pp_props.parquet", index=False)
            # Append to history
            _pp_hist_path = DASHBOARD_DIR / "pp_props_history.parquet"
            if _pp_hist_path.exists():
                _pp_hist = pd.read_parquet(_pp_hist_path)
                _pp_hist = _pp_hist[_pp_hist["game_date"] != game_date]
                pp_df = pd.concat([_pp_hist, pp_df], ignore_index=True)
            pp_df.to_parquet(_pp_hist_path, index=False)
            pp_resolved = pp_df[pp_df["game_date"] == game_date].copy()
            logger.info(
                "Saved %d PP props (%d resolved to MLB IDs), "
                "history now %d rows",
                len(pp_resolved), pp_resolved["player_id"].notna().sum(),
                len(pp_df),
            )
        else:
            logger.info("No Prize Picks props available")
    except Exception:
        logger.warning("Prize Picks props fetch failed", exc_info=True)

    return (dk_resolved, pp_resolved)



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
        "R": "runs", "RBI": "rbi",
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

            if pid not in player_stats:
                continue

            ps = player_stats[pid]
            if stat == "HRR":
                # Combined: Hits + Runs + RBIs
                if "H" in ps and "R" in ps and "RBI" in ps:
                    actual = ps["H"] + ps["R"] + ps["RBI"]
                    props_df.at[idx, "actual"] = actual
                    props_df.at[idx, "over_hit"] = actual > line
            elif stat in ps:
                actual = ps[stat]
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
        "umpire_k_lift", "umpire_bb_lift", "weather_k_lift",
        "dk_mean", "dk_median", "dk_q10", "dk_q90",
        "espn_mean", "espn_median",
        "expected_ip", "expected_pitches", "expected_bf",
        "actual", "over_hit", "game_status",
    ] + [f"p_over_{x + 0.5:.1f}" for x in range(11)]
    ).to_parquet(history_path, index=False)



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
    shrinkage: float = 0.5,
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
                    shrinkage=shrinkage,
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
