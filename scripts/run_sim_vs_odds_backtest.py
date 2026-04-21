#!/usr/bin/env python
"""Backtest full-game sims against stored Bovada / DK game odds.

Re-runs simulate_full_game_both_teams() for all 2026 games that have
stored market odds (moneyline, spread, over/under), using ONLY preseason
posterior samples (no 2026 leakage).

Evaluates:
  - Moneyline: model win prob vs market implied, Brier, accuracy
  - Spread: model P(cover) vs market implied at -1.5/+1.5
  - Over/Under: model P(over) vs market at posted total line

Usage
-----
python scripts/run_sim_vs_odds_backtest.py                    # all games
python scripts/run_sim_vs_odds_backtest.py --n-sims 5000      # more MC sims
python scripts/run_sim_vs_odds_backtest.py --quick             # fewer sims
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit as _logit
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql
from src.data.paths import dashboard_dir
from src.evaluation.metrics import compute_ece, compute_temperature
from src.models.game_sim.lineup_simulator import simulate_full_game_both_teams
from src.models.matchup import score_matchup_for_stat
from src.utils.weather import parse_temp_bucket, wind_category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DASHBOARD_DIR = dashboard_dir()
SNAPSHOTS_DIR = DASHBOARD_DIR / "snapshots"


# ---------------------------------------------------------------------------
# 1. Load preseason posteriors (no 2026 leakage)
# ---------------------------------------------------------------------------

def _load_preseason_npz(name: str) -> dict[int, np.ndarray]:
    """Load preseason .npz samples, keyed by integer player_id."""
    preseason = SNAPSHOTS_DIR / f"{name}_preseason.npz"
    fallback = DASHBOARD_DIR / f"{name}.npz"

    if preseason.exists():
        path = preseason
    elif fallback.exists():
        logger.warning(
            "Preseason %s_preseason.npz not found — using %s.npz "
            "(may include 2026 conjugate updates)", name, name,
        )
        path = fallback
    else:
        logger.error("No samples found for %s", name)
        return {}

    npz = np.load(path)
    result = {int(k): npz[k] for k in npz.files}
    logger.info("Loaded %s: %d players from %s", name, len(result), path.name)
    return result


# ---------------------------------------------------------------------------
# 2. Load game schedule, lineups, actuals from database
# ---------------------------------------------------------------------------

def _load_games(start_date: str, end_date: str) -> pd.DataFrame:
    """Load game schedule with starters, umpires, and weather."""
    games = read_sql("""
        SELECT dg.game_pk, dg.game_date::text as game_date, dg.venue_id,
               dg.home_team_id, dg.away_team_id,
               dt_h.abbreviation as home_abbr,
               dt_a.abbreviation as away_abbr,
               du.hp_umpire_name,
               dw.temperature as weather_temp,
               dw.condition as weather_condition,
               dw.is_dome,
               dw.wind_category
        FROM production.dim_game dg
        LEFT JOIN production.dim_team dt_h ON dg.home_team_id = dt_h.team_id
        LEFT JOIN production.dim_team dt_a ON dg.away_team_id = dt_a.team_id
        LEFT JOIN production.dim_umpire du ON dg.game_pk = du.game_pk
        LEFT JOIN production.dim_weather dw ON dg.game_pk = dw.game_pk
        WHERE dg.game_date >= :start AND dg.game_date <= :end
          AND dg.game_type = 'R'
        ORDER BY dg.game_date, dg.game_pk
    """, {"start": start_date, "end": end_date})

    # Add starters
    starters = read_sql("""
        SELECT pb.game_pk, pb.pitcher_id, pb.team_id
        FROM staging.pitching_boxscores pb
        JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
        WHERE dg.game_date >= :start AND dg.game_date <= :end
          AND dg.game_type = 'R'
          AND pb.is_starter = true
          AND pb.batters_faced >= 9
    """, {"start": start_date, "end": end_date})

    for side in ("home", "away"):
        side_sp = starters.merge(
            games[["game_pk", f"{side}_team_id"]],
            left_on=["game_pk", "team_id"],
            right_on=["game_pk", f"{side}_team_id"],
            how="inner",
        )[["game_pk", "pitcher_id"]].rename(
            columns={"pitcher_id": f"{side}_sp"}
        )
        games = games.merge(side_sp, on="game_pk", how="left")

    return games


def _load_lineups(game_pks: list[int]) -> pd.DataFrame:
    """Load confirmed lineups from fact_lineup."""
    if not game_pks:
        return pd.DataFrame(columns=["game_pk", "player_id", "team_id", "batting_order"])
    pks_str = ",".join(str(g) for g in game_pks)
    return read_sql(f"""
        SELECT game_pk, player_id, team_id, batting_order
        FROM production.fact_lineup
        WHERE game_pk IN ({pks_str})
          AND is_starter = true
          AND batting_order IS NOT NULL
          AND batting_order BETWEEN 1 AND 9
        ORDER BY game_pk, team_id, batting_order
    """, {})


def _load_game_scores(start_date: str, end_date: str) -> pd.DataFrame:
    """Load actual game scores."""
    return read_sql("""
        WITH team_scores AS (
            SELECT pb.game_pk, pb.team_id,
                   SUM(pb.runs) AS team_runs
            FROM staging.pitching_boxscores pb
            JOIN production.dim_game dg ON pb.game_pk = dg.game_pk
            WHERE dg.game_date >= :start AND dg.game_date <= :end
              AND dg.game_type = 'R'
            GROUP BY pb.game_pk, pb.team_id
        )
        SELECT dg.game_pk,
               COALESCE(h.team_runs, 0) AS home_runs,
               COALESCE(a.team_runs, 0) AS away_runs
        FROM production.dim_game dg
        LEFT JOIN team_scores h
            ON dg.game_pk = h.game_pk AND dg.home_team_id = h.team_id
        LEFT JOIN team_scores a
            ON dg.game_pk = a.game_pk AND dg.away_team_id = a.team_id
        WHERE dg.game_date >= :start AND dg.game_date <= :end
          AND dg.game_type = 'R'
    """, {"start": start_date, "end": end_date})


# ---------------------------------------------------------------------------
# 3. Load and normalize market odds
# ---------------------------------------------------------------------------

def _load_game_odds() -> pd.DataFrame:
    """Load game_odds_history.parquet and pivot to one row per game."""
    path = DASHBOARD_DIR / "game_odds_history.parquet"
    if not path.exists():
        return pd.DataFrame()

    raw = pd.read_parquet(path)

    # Take EARLIEST snapshot per game per source (pre-game opening line).
    # Later snapshots are often in-game live lines with extreme odds.
    raw = (
        raw.sort_values("snapshot_ts")
        .groupby(["game_date", "game_description", "source",
                   "market_type", "team_or_label"])
        .first()
        .reset_index()
    )

    # Normalize game descriptions: map "ARI Diamondbacks @ NY Mets" and
    # "Arizona Diamondbacks @ New York Mets" to the same game.
    # Strategy: extract away/home team from format "AWAY @ HOME",
    # then match by (game_date, away_team_partial, home_team_partial)

    return raw


def _parse_odds_str(odds_str) -> float | None:
    """Convert American odds to implied probability."""
    if pd.isna(odds_str):
        return None
    s = str(odds_str).replace("\u2212", "-").strip()
    if s in ("", "EVEN"):
        return 0.5
    try:
        val = float(s)
    except ValueError:
        return None
    if val > 0:
        return 100.0 / (val + 100.0)
    elif val < 0:
        return abs(val) / (abs(val) + 100.0)
    return 0.5


def _vig_free(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vig to get fair probabilities."""
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total


def _build_team_name_map() -> dict[str, int]:
    """Build mapping from various team name forms to team_id."""
    teams = read_sql("""
        SELECT team_id, abbreviation, team_name, full_name
        FROM production.dim_team
    """, {})

    mapping: dict[str, int] = {}
    for _, r in teams.iterrows():
        tid = int(r["team_id"])
        for col in ["abbreviation", "team_name", "full_name"]:
            val = str(r.get(col, "")).strip()
            if val:
                mapping[val.lower()] = tid
                # Also partial forms: "NY Mets" -> "mets"
                parts = val.lower().split()
                if len(parts) > 1:
                    mapping[parts[-1]] = tid

    # Common abbreviation overrides
    extras = {
        "ari": 109, "az": 109, "diamondbacks": 109, "d-backs": 109,
        "atl": 144, "braves": 144,
        "bal": 110, "orioles": 110,
        "bos": 111, "red sox": 111,
        "chc": 112, "chi cubs": 112,
        "cws": 145, "chw": 145, "chi white sox": 145, "white sox": 145,
        "cin": 113, "reds": 113,
        "cle": 114, "guardians": 114,
        "col": 115, "rockies": 115,
        "det": 116, "tigers": 116,
        "hou": 117, "astros": 117,
        "kc": 118, "royals": 118,
        "laa": 108, "la angels": 108, "angels": 108,
        "lad": 119, "la dodgers": 119, "dodgers": 119,
        "mia": 146, "marlins": 146,
        "mil": 158, "brewers": 158,
        "min": 142, "twins": 142,
        "nym": 121, "ny mets": 121, "mets": 121,
        "nyy": 147, "ny yankees": 147, "yankees": 147,
        "ath": 133, "athletics": 133, "oakland": 133,
        "phi": 143, "phillies": 143,
        "pit": 134, "pirates": 134,
        "sd": 135, "padres": 135,
        "sf": 137, "giants": 137,
        "sea": 136, "mariners": 136,
        "stl": 138, "cardinals": 138,
        "tb": 139, "rays": 139, "tampa bay": 139,
        "tex": 140, "rangers": 140,
        "tor": 141, "blue jays": 141,
        "was": 120, "wsh": 120, "nationals": 120,
    }
    mapping.update(extras)
    return mapping


def _match_odds_to_games(
    odds_raw: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Match odds rows to game_pk by parsing team names from game_description."""
    team_map = _build_team_name_map()

    # Parse game_description: "AWAY_TEAM @ HOME_TEAM"
    def _parse_teams(desc: str) -> tuple[int | None, int | None]:
        if "@" not in desc:
            return None, None
        parts = desc.split("@")
        away_str = parts[0].strip().lower()
        home_str = parts[1].strip().lower()

        away_id = team_map.get(away_str)
        home_id = team_map.get(home_str)

        # Try last word (franchise name)
        if away_id is None:
            away_words = away_str.split()
            for i in range(len(away_words)):
                key = " ".join(away_words[i:])
                if key in team_map:
                    away_id = team_map[key]
                    break
        if home_id is None:
            home_words = home_str.split()
            for i in range(len(home_words)):
                key = " ".join(home_words[i:])
                if key in team_map:
                    home_id = team_map[key]
                    break

        return away_id, home_id

    # Build game lookup: (game_date, away_team_id, home_team_id) -> game_pk
    game_lookup: dict[tuple[str, int, int], int] = {}
    for _, g in games.iterrows():
        key = (str(g["game_date"]), int(g["away_team_id"]), int(g["home_team_id"]))
        game_lookup[key] = int(g["game_pk"])

    # Attach game_pk to odds
    matched_rows = []
    for desc in odds_raw["game_description"].unique():
        away_id, home_id = _parse_teams(desc)
        if away_id is None or home_id is None:
            continue

        desc_rows = odds_raw[odds_raw["game_description"] == desc]
        for gd in desc_rows["game_date"].unique():
            gpk = game_lookup.get((str(gd), away_id, home_id))
            if gpk is None:
                continue
            subset = desc_rows[desc_rows["game_date"] == gd].copy()
            subset["game_pk"] = gpk
            subset["parsed_away_id"] = away_id
            subset["parsed_home_id"] = home_id
            matched_rows.append(subset)

    if not matched_rows:
        return pd.DataFrame()

    return pd.concat(matched_rows, ignore_index=True)


def _pivot_odds(matched: pd.DataFrame) -> pd.DataFrame:
    """Pivot matched odds into one row per game with ML/spread/total columns.

    Prefers DK; falls back to Bovada.
    """
    records: list[dict] = []
    for gpk, grp in matched.groupby("game_pk"):
        rec: dict = {"game_pk": int(gpk)}

        # Prefer DK if available
        for source in ["dk", "bovada"]:
            src_rows = grp[grp["source"] == source]
            if src_rows.empty:
                continue

            # Moneyline
            ml = src_rows[src_rows["market_type"] == "moneyline"]
            for _, r in ml.iterrows():
                ot = str(r["outcome_type"]).lower()
                impl = _parse_odds_str(r["odds"])
                if "home" in ot:
                    rec.setdefault("ml_home_implied", impl)
                elif "away" in ot:
                    rec.setdefault("ml_away_implied", impl)

            # Spread
            sp = src_rows[src_rows["market_type"] == "spread"]
            for _, r in sp.iterrows():
                ot = str(r["outcome_type"]).lower()
                impl = _parse_odds_str(r["odds"])
                line = r.get("line")
                if "home" in ot:
                    rec.setdefault("spread_home_line", line)
                    rec.setdefault("spread_home_implied", impl)
                elif "away" in ot:
                    rec.setdefault("spread_away_line", line)
                    rec.setdefault("spread_away_implied", impl)

            # Total
            tot = src_rows[src_rows["market_type"] == "total"]
            for _, r in tot.iterrows():
                ot = str(r["outcome_type"]).lower()
                impl = _parse_odds_str(r["odds"])
                line = r.get("line")
                if "over" in ot:
                    rec.setdefault("total_line", line)
                    rec.setdefault("total_over_implied", impl)
                elif "under" in ot:
                    rec.setdefault("total_under_implied", impl)

        # Compute vig-free probabilities
        if rec.get("ml_home_implied") and rec.get("ml_away_implied"):
            hf, af = _vig_free(rec["ml_home_implied"], rec["ml_away_implied"])
            rec["ml_home_fair"] = hf
            rec["ml_away_fair"] = af

        if rec.get("total_over_implied") and rec.get("total_under_implied"):
            of, uf = _vig_free(rec["total_over_implied"], rec["total_under_implied"])
            rec["total_over_fair"] = of

        if rec.get("spread_home_implied") and rec.get("spread_away_implied"):
            hf, af = _vig_free(rec["spread_home_implied"], rec["spread_away_implied"])
            rec["spread_home_fair"] = hf
            rec["spread_away_fair"] = af

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Sim runner using preseason posteriors
# ---------------------------------------------------------------------------

# Weather helpers imported from src.utils.weather (canonical thresholds)
_parse_temp_bucket = parse_temp_bucket
_wind_category = wind_category


def _run_full_game_sims(
    games: pd.DataFrame,
    lineups_df: pd.DataFrame,
    pitcher_k: dict[int, np.ndarray],
    pitcher_bb: dict[int, np.ndarray],
    pitcher_hr: dict[int, np.ndarray],
    hitter_k: dict[int, np.ndarray],
    hitter_bb: dict[int, np.ndarray],
    hitter_hr: dict[int, np.ndarray],
    bf_priors: pd.DataFrame,
    bullpen_rates: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    hitter_vuln: pd.DataFrame,
    baselines_pt: dict,
    *,
    ump_k_lookup: dict[str, float] | None = None,
    ump_bb_lookup: dict[str, float] | None = None,
    wx_lookup: dict[tuple[str, str], dict] | None = None,
    park_lift_lookup: dict[int, dict[str, float]] | None = None,
    ld_babip_lookup: dict[int, float] | None = None,
    speed_babip_lookup: dict[int, float] | None = None,
    bip_lookup: dict[int, np.ndarray] | None = None,
    n_sims: int = 3000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run full-game both-teams sim for each game with full context."""
    valid_pitchers = set(pitcher_k) & set(pitcher_bb) & set(pitcher_hr)
    valid_batters = set(hitter_k) & set(hitter_bb) & set(hitter_hr)
    logger.info(
        "Valid pitchers: %d, valid batters: %d",
        len(valid_pitchers), len(valid_batters),
    )

    bullpen_latest = (
        bullpen_rates.sort_values("season")
        .groupby("team_id").last().reset_index()
    )
    _FALLBACK_K = np.full(500, 0.226)
    _FALLBACK_BB = np.full(500, 0.082)
    _FALLBACK_HR = np.full(500, 0.031)

    ump_k_lookup = ump_k_lookup or {}
    ump_bb_lookup = ump_bb_lookup or {}
    wx_lookup = wx_lookup or {}
    park_lift_lookup = park_lift_lookup or {}
    ld_babip_lookup = ld_babip_lookup or {}
    speed_babip_lookup = speed_babip_lookup or {}

    def _get_bf(pid: int) -> tuple[float, float]:
        row = bf_priors[bf_priors["pitcher_id"] == pid]
        if len(row) > 0:
            r = row.sort_values("season").iloc[-1]
            return float(r["mu_bf"]), float(r["sigma_bf"])
        return 22.0, 4.5

    def _get_bp(tid: int) -> tuple[float, float, float]:
        row = bullpen_latest[bullpen_latest["team_id"] == tid]
        if len(row) > 0:
            return (
                float(row.iloc[0].get("k_rate", 0.253)),
                float(row.iloc[0].get("bb_rate", 0.084)),
                float(row.iloc[0].get("hr_rate", 0.024)),
            )
        return 0.253, 0.084, 0.024

    def _lineup_matchup(
        pitcher_id: int, batter_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_lifts, bb_lifts, hr_lifts = [], [], []
        for bid in batter_ids:
            for stat, lst in [("k", k_lifts), ("bb", bb_lifts), ("hr", hr_lifts)]:
                try:
                    res = score_matchup_for_stat(
                        stat, pitcher_id, bid,
                        pitcher_arsenal, hitter_vuln, baselines_pt,
                    )
                    v = res.get(f"matchup_{stat}_logit_lift", 0.0)
                    if isinstance(v, float) and np.isnan(v):
                        v = 0.0
                except Exception:
                    v = 0.0
                lst.append(v)
        return np.array(k_lifts), np.array(bb_lifts), np.array(hr_lifts)

    def _lineup_babip(bids: list[int]) -> np.ndarray:
        return np.array([
            ld_babip_lookup.get(b, 0.0) + speed_babip_lookup.get(b, 0.0)
            for b in bids
        ])

    _LEAGUE_BIP = np.array([0.700, 0.222, 0.065, 0.005])
    bip_lookup = bip_lookup or {}

    def _lineup_bip(bids: list[int]) -> np.ndarray:
        """Build (9, 4) BIP profile array for a lineup."""
        return np.stack([
            bip_lookup.get(b, _LEAGUE_BIP)
            for b in bids
        ], axis=0).astype(np.float64)

    results: list[dict] = []
    n_skipped = 0

    for idx, (_, game) in enumerate(games.iterrows()):
        gpk = int(game["game_pk"])
        home_tid = int(game["home_team_id"])
        away_tid = int(game["away_team_id"])
        home_sp = game.get("home_sp")
        away_sp = game.get("away_sp")

        if pd.isna(home_sp) or pd.isna(away_sp):
            n_skipped += 1
            continue
        home_sp = int(home_sp)
        away_sp = int(away_sp)

        if home_sp not in valid_pitchers or away_sp not in valid_pitchers:
            n_skipped += 1
            continue

        # Get lineups
        glu = lineups_df[lineups_df["game_pk"] == gpk]
        home_lu = glu[glu["team_id"] == home_tid].sort_values("batting_order")
        away_lu = glu[glu["team_id"] == away_tid].sort_values("batting_order")
        home_bids = [int(x) for x in home_lu["player_id"].tolist()[:9]]
        away_bids = [int(x) for x in away_lu["player_id"].tolist()[:9]]

        if len(home_bids) < 9 or len(away_bids) < 9:
            n_skipped += 1
            continue

        # Batter posterior samples with league-avg fallback
        home_k = [hitter_k.get(b, _FALLBACK_K) for b in home_bids]
        home_bb = [hitter_bb.get(b, _FALLBACK_BB) for b in home_bids]
        home_hr = [hitter_hr.get(b, _FALLBACK_HR) for b in home_bids]
        away_k = [hitter_k.get(b, _FALLBACK_K) for b in away_bids]
        away_bb = [hitter_bb.get(b, _FALLBACK_BB) for b in away_bids]
        away_hr = [hitter_hr.get(b, _FALLBACK_HR) for b in away_bids]

        # Starter rates — pass full posterior arrays for per-sim uncertainty
        home_sp_k = np.asarray(pitcher_k[home_sp], dtype=np.float64)
        home_sp_bb = np.asarray(pitcher_bb[home_sp], dtype=np.float64)
        home_sp_hr = np.asarray(pitcher_hr[home_sp], dtype=np.float64)
        away_sp_k = np.asarray(pitcher_k[away_sp], dtype=np.float64)
        away_sp_bb = np.asarray(pitcher_bb[away_sp], dtype=np.float64)
        away_sp_hr = np.asarray(pitcher_hr[away_sp], dtype=np.float64)

        home_bf_mu, home_bf_sig = _get_bf(home_sp)
        away_bf_mu, away_bf_sig = _get_bf(away_sp)

        # Bullpen + matchups
        home_bp_k, home_bp_bb, home_bp_hr = _get_bp(home_tid)
        away_bp_k, away_bp_bb, away_bp_hr = _get_bp(away_tid)

        away_mk, away_mbb, away_mhr = _lineup_matchup(home_sp, away_bids)
        home_mk, home_mbb, home_mhr = _lineup_matchup(away_sp, home_bids)

        # --- Context enrichment ---
        hp_ump = game.get("hp_umpire_name", "")
        ump_k = ump_k_lookup.get(hp_ump, 0.0) if pd.notna(hp_ump) else 0.0
        ump_bb = ump_bb_lookup.get(hp_ump, 0.0) if pd.notna(hp_ump) else 0.0

        # Weather
        wx_k = 0.0
        is_dome = bool(game.get("is_dome", False))
        if not is_dome:
            temp_bucket = _parse_temp_bucket(game.get("weather_temp"))
            wind_cat = _wind_category(game)
            wx_info = wx_lookup.get((temp_bucket, wind_cat))
            if wx_info:
                k_mult = wx_info["k_multiplier"]
                ok = wx_info["overall_k_rate"]
                wx_k = float(
                    _logit(np.clip(ok * k_mult, 1e-6, 1 - 1e-6))
                    - _logit(np.clip(ok, 1e-6, 1 - 1e-6))
                )

        # Park factors
        venue_id = game.get("venue_id")
        pk_lifts = park_lift_lookup.get(int(venue_id), {}) if pd.notna(venue_id) else {}
        pk_k = pk_lifts.get("k_lift", 0.0)
        pk_hr = pk_lifts.get("hr_lift", 0.0)

        # BABIP + BIP adjustments
        away_babip = _lineup_babip(away_bids)
        home_babip = _lineup_babip(home_bids)
        away_bip = _lineup_bip(away_bids)
        home_bip = _lineup_bip(home_bids)

        try:
            result = simulate_full_game_both_teams(
                away_batter_k_rate_samples=away_k,
                away_batter_bb_rate_samples=away_bb,
                away_batter_hr_rate_samples=away_hr,
                home_starter_k_rate=home_sp_k,
                home_starter_bb_rate=home_sp_bb,
                home_starter_hr_rate=home_sp_hr,
                home_starter_bf_mu=home_bf_mu,
                home_starter_bf_sigma=home_bf_sig,
                home_batter_k_rate_samples=home_k,
                home_batter_bb_rate_samples=home_bb,
                home_batter_hr_rate_samples=home_hr,
                away_starter_k_rate=away_sp_k,
                away_starter_bb_rate=away_sp_bb,
                away_starter_hr_rate=away_sp_hr,
                away_starter_bf_mu=away_bf_mu,
                away_starter_bf_sigma=away_bf_sig,
                away_matchup_k_lifts=away_mk,
                away_matchup_bb_lifts=away_mbb,
                away_matchup_hr_lifts=away_mhr,
                home_matchup_k_lifts=home_mk,
                home_matchup_bb_lifts=home_mbb,
                home_matchup_hr_lifts=home_mhr,
                home_bullpen_k_rate=home_bp_k,
                home_bullpen_bb_rate=home_bp_bb,
                home_bullpen_hr_rate=home_bp_hr,
                away_bullpen_k_rate=away_bp_k,
                away_bullpen_bb_rate=away_bp_bb,
                away_bullpen_hr_rate=away_bp_hr,
                away_batter_babip_adjs=away_babip,
                home_batter_babip_adjs=home_babip,
                away_batter_bip_probs=away_bip,
                home_batter_bip_probs=home_bip,
                umpire_k_lift=ump_k,
                umpire_bb_lift=ump_bb,
                park_k_lift=pk_k,
                park_hr_lift=pk_hr,
                weather_k_lift=wx_k,
                n_sims=n_sims,
                random_seed=random_seed + gpk % 10000,
            )
        except Exception as e:
            logger.warning("Sim failed game %d: %s", gpk, e)
            n_skipped += 1
            continue

        total_runs = result.home_runs.astype(float) + result.away_runs.astype(float)
        margin = result.home_runs.astype(float) - result.away_runs.astype(float)

        rec: dict = {
            "game_pk": gpk,
            "game_date": str(game.get("game_date", "")),
            "home_team_id": home_tid,
            "away_team_id": away_tid,
            "home_abbr": game.get("home_abbr", ""),
            "away_abbr": game.get("away_abbr", ""),
            "home_sp": home_sp,
            "away_sp": away_sp,
            "pred_home_win_prob": result.home_win_prob,
            "pred_away_win_prob": 1.0 - result.home_win_prob,
            "pred_home_runs": float(np.mean(result.home_runs)),
            "pred_away_runs": float(np.mean(result.away_runs)),
            "pred_total_runs": float(np.mean(total_runs)),
            "pred_home_margin": float(np.mean(margin)),
        }

        # O/U probs at common lines
        for line in [5.5, 6.5, 7.5, 8.5, 9.5, 10.5]:
            rec[f"p_over_{line:.1f}"] = float(np.mean(total_runs > line))

        # Spread probs
        for line in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            rec[f"p_home_cover_{line:+.1f}"] = float(np.mean(margin > line))

        results.append(rec)

        if (idx + 1) % 25 == 0:
            logger.info("  %d/%d games simulated...", idx + 1, len(games))

    logger.info("Simulated %d games, skipped %d", len(results), n_skipped)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def _evaluate_moneyline(df: pd.DataFrame) -> dict:
    """Evaluate moneyline predictions vs market."""
    m: dict = {}
    mask = df["ml_home_fair"].notna() & df["actual_home_win"].notna()
    d = df[mask]
    if len(d) < 10:
        return m

    y = d["actual_home_win"].values
    p_model = d["pred_home_win_prob"].values
    p_market = d["ml_home_fair"].values

    m["ml_n"] = len(d)
    m["ml_model_brier"] = float(brier_score_loss(y, p_model))
    m["ml_market_brier"] = float(brier_score_loss(y, p_market))
    m["ml_model_accuracy"] = float(((p_model > 0.5) == (y > 0.5)).mean())
    m["ml_market_accuracy"] = float(((p_market > 0.5) == (y > 0.5)).mean())
    m["ml_model_ece"] = compute_ece(p_model, y)
    m["ml_market_ece"] = compute_ece(p_market, y)

    # Edge analysis: where model disagrees with market
    edge = p_model - p_market
    m["ml_edge_mean"] = float(edge.mean())
    m["ml_edge_std"] = float(edge.std())

    # Simulated P&L: bet model side when |edge| > threshold
    for thr in [0.0, 0.03, 0.05, 0.08]:
        strong = np.abs(edge) > thr
        if strong.sum() < 5:
            continue
        # Bet on model's favored side at -110
        model_picks_home = p_model > 0.5
        bet_home = model_picks_home & strong
        bet_away = (~model_picks_home) & strong
        home_wins = y > 0.5
        wins = (bet_home & home_wins) | (bet_away & ~home_wins)
        n_bets = int(strong.sum())
        profit = float(wins.sum() * 90.91 - (n_bets - wins.sum()) * 100)
        m[f"ml_pnl_{thr:.0%}"] = round(profit, 0)
        m[f"ml_roi_{thr:.0%}"] = round(profit / (n_bets * 100) * 100, 1)
        m[f"ml_n_bets_{thr:.0%}"] = n_bets
        m[f"ml_win_pct_{thr:.0%}"] = round(float(wins.mean()) * 100, 1)

    return m


def _evaluate_totals(df: pd.DataFrame) -> dict:
    """Evaluate over/under predictions vs market."""
    m: dict = {}
    mask = df["total_line"].notna() & df["actual_total"].notna()
    d = df[mask]
    if len(d) < 10:
        return m

    m["ou_n"] = len(d)

    # For each game, compute model P(over) at the market's posted line
    model_p_over = []
    for _, r in d.iterrows():
        line = float(r["total_line"])
        # Find nearest column
        col = f"p_over_{line:.1f}"
        if col in r.index and pd.notna(r[col]):
            model_p_over.append(float(r[col]))
        else:
            # Interpolate from pred_total_runs
            model_p_over.append(np.nan)

    d = d.copy()
    d["model_p_over_at_line"] = model_p_over
    d = d.dropna(subset=["model_p_over_at_line"])

    if len(d) < 10:
        return m

    y = (d["actual_total"] > d["total_line"]).astype(float).values
    p_model = d["model_p_over_at_line"].values

    m["ou_n"] = len(d)
    m["ou_model_brier"] = float(brier_score_loss(y, p_model))
    m["ou_model_accuracy"] = float(((p_model > 0.5) == (y > 0.5)).mean())
    m["ou_model_ece"] = compute_ece(p_model, y)

    if "total_over_fair" in d.columns:
        mask2 = d["total_over_fair"].notna()
        if mask2.sum() >= 10:
            p_market = d.loc[mask2, "total_over_fair"].values
            y2 = (d.loc[mask2, "actual_total"] > d.loc[mask2, "total_line"]).astype(float).values
            m["ou_market_brier"] = float(brier_score_loss(y2, p_market))
            m["ou_market_accuracy"] = float(((p_market > 0.5) == (y2 > 0.5)).mean())

    # Bias
    m["ou_total_bias"] = float(
        d["pred_total_runs"].mean() - d["actual_total"].mean()
    )
    m["ou_total_corr"] = float(d["pred_total_runs"].corr(d["actual_total"]))

    # Edge P&L
    if "total_over_fair" in d.columns:
        edge = d["model_p_over_at_line"].values - d["total_over_fair"].fillna(0.5).values
        for thr in [0.0, 0.03, 0.05, 0.08]:
            strong = np.abs(edge) > thr
            if strong.sum() < 5:
                continue
            picks_over = edge > 0
            actual_over = (d["actual_total"].values > d["total_line"].values)
            wins = (picks_over & actual_over & strong) | (~picks_over & ~actual_over & strong)
            n_bets = int(strong.sum())
            profit = float(wins.sum() * 90.91 - (n_bets - wins.sum()) * 100)
            m[f"ou_pnl_{thr:.0%}"] = round(profit, 0)
            m[f"ou_roi_{thr:.0%}"] = round(profit / (n_bets * 100) * 100, 1)
            m[f"ou_n_bets_{thr:.0%}"] = n_bets

    return m


def _evaluate_spread(df: pd.DataFrame) -> dict:
    """Evaluate spread/run-line predictions vs market."""
    m: dict = {}
    mask = df["spread_home_line"].notna() & df["actual_home_margin"].notna()
    d = df[mask]
    if len(d) < 10:
        return m

    m["sp_n"] = len(d)

    # Model P(home covers) at the posted spread
    model_p_home_cover = []
    for _, r in d.iterrows():
        line = float(r["spread_home_line"])
        # Home covers when margin > -line (e.g., home -1.5 means margin > 1.5)
        col = f"p_home_cover_{-line:+.1f}"
        if col in r.index and pd.notna(r[col]):
            model_p_home_cover.append(float(r[col]))
        else:
            model_p_home_cover.append(np.nan)

    d = d.copy()
    d["model_p_home_cover"] = model_p_home_cover
    d = d.dropna(subset=["model_p_home_cover"])

    if len(d) < 10:
        return m

    y = (d["actual_home_margin"] > -d["spread_home_line"]).astype(float).values
    p_model = d["model_p_home_cover"].values

    m["sp_n"] = len(d)
    m["sp_model_brier"] = float(brier_score_loss(y, p_model))
    m["sp_model_accuracy"] = float(((p_model > 0.5) == (y > 0.5)).mean())

    if "spread_home_fair" in d.columns:
        mask2 = d["spread_home_fair"].notna()
        if mask2.sum() >= 10:
            p_market = d.loc[mask2, "spread_home_fair"].values
            y2 = (d.loc[mask2, "actual_home_margin"] > -d.loc[mask2, "spread_home_line"]).astype(float).values
            m["sp_market_brier"] = float(brier_score_loss(y2, p_market))
            m["sp_market_accuracy"] = float(((p_market > 0.5) == (y2 > 0.5)).mean())

    # Margin accuracy
    m["sp_margin_corr"] = float(
        d["pred_home_margin"].corr(d["actual_home_margin"])
    )

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sim vs Game Odds Backtest")
    parser.add_argument("--n-sims", type=int, default=3000)
    parser.add_argument("--quick", action="store_true",
                        help="Fewer sims (1000) for fast iteration")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    if args.quick:
        args.n_sims = 1000

    t0 = time.time()

    # Load market odds
    logger.info("Loading market odds...")
    odds_raw = _load_game_odds()
    if odds_raw.empty:
        logger.error("No game_odds_history.parquet found")
        sys.exit(1)

    # Determine date range from odds
    start_date = args.start_date or str(odds_raw["game_date"].min())
    end_date = args.end_date or str(odds_raw["game_date"].max())
    logger.info("Odds date range: %s to %s", start_date, end_date)

    # Load game schedule
    logger.info("Loading games...")
    games = _load_games(start_date, end_date)
    logger.info("Games loaded: %d", len(games))

    # Match odds to games
    logger.info("Matching odds to games...")
    matched_odds = _match_odds_to_games(odds_raw, games)
    logger.info("Matched odds rows: %d", len(matched_odds))
    odds_pivoted = _pivot_odds(matched_odds)
    logger.info("Games with odds: %d", len(odds_pivoted))

    # Filter games to those with odds
    games_with_odds = games[games["game_pk"].isin(odds_pivoted["game_pk"])].copy()
    logger.info("Games to simulate: %d", len(games_with_odds))

    if games_with_odds.empty:
        logger.error("No games matched to odds")
        sys.exit(1)

    # Load lineups
    lineups = _load_lineups(games_with_odds["game_pk"].tolist())

    # Load actuals
    scores = _load_game_scores(start_date, end_date)

    # Load preseason posteriors
    logger.info("Loading preseason posteriors (no 2026 leakage)...")
    pitcher_k = _load_preseason_npz("pitcher_k_samples")
    pitcher_bb = _load_preseason_npz("pitcher_bb_samples")
    pitcher_hr = _load_preseason_npz("pitcher_hr_samples")
    hitter_k = _load_preseason_npz("hitter_k_samples")
    hitter_bb = _load_preseason_npz("hitter_bb_samples")
    hitter_hr = _load_preseason_npz("hitter_hr_samples")

    # Load supporting data
    logger.info("Loading supporting data...")
    bf_priors = pd.read_parquet(DASHBOARD_DIR / "bf_priors.parquet")
    bullpen_rates = pd.read_parquet(DASHBOARD_DIR / "team_bullpen_rates.parquet")
    pitcher_arsenal = pd.read_parquet(DASHBOARD_DIR / "pitcher_arsenal.parquet")
    hitter_vuln = pd.read_parquet(DASHBOARD_DIR / "hitter_vuln.parquet")

    from src.data.league_baselines import get_baselines_dict
    baselines_pt = get_baselines_dict(
        grouping="pitch_type", recency_weights="marcel",
    )

    # Load enrichment data (park, umpire, weather, BABIP)
    logger.info("Loading enrichment data...")
    ump_k_lookup: dict[str, float] = {}
    ump_bb_lookup: dict[str, float] = {}
    try:
        ump_df = pd.read_parquet(DASHBOARD_DIR / "umpire_tendencies.parquet")
        for _, ur in ump_df.iterrows():
            ump_k_lookup[ur["hp_umpire_name"]] = float(ur.get("k_logit_lift", 0.0))
            ump_bb_lookup[ur["hp_umpire_name"]] = float(ur.get("bb_logit_lift", 0.0))
        logger.info("Umpire tendencies: %d umpires", len(ump_k_lookup))
    except FileNotFoundError:
        logger.info("No umpire_tendencies.parquet — skipping")

    wx_lookup: dict[tuple[str, str], dict] = {}
    try:
        wx_df = pd.read_parquet(DASHBOARD_DIR / "weather_effects.parquet")
        for _, wr in wx_df.iterrows():
            wx_lookup[(wr["temp_bucket"], wr["wind_category"])] = {
                "k_multiplier": float(wr["k_multiplier"]),
                "overall_k_rate": float(wr["overall_k_rate"]),
            }
        logger.info("Weather effects: %d combos", len(wx_lookup))
    except FileNotFoundError:
        logger.info("No weather_effects.parquet — skipping")

    park_lift_lookup: dict[int, dict[str, float]] = {}
    try:
        pf_df = pd.read_parquet(DASHBOARD_DIR / "park_factor_lifts.parquet")
        for _, pr in pf_df.iterrows():
            park_lift_lookup[int(pr["venue_id"])] = {
                "k_lift": float(pr["k_lift"]),
                "hr_lift": float(pr["hr_lift"]),
            }
        logger.info("Park factor lifts: %d venues", len(park_lift_lookup))
    except FileNotFoundError:
        logger.info("No park_factor_lifts.parquet — skipping")

    ld_babip_lookup: dict[int, float] = {}
    try:
        ld_df = pd.read_parquet(DASHBOARD_DIR / "batter_ld_rate.parquet")
        if not ld_df.empty:
            ld_latest = ld_df.sort_values("season").groupby("player_id").last().reset_index()
            for _, lr in ld_latest.iterrows():
                ld_dev = float(lr["ld_rate_regressed"]) - 0.22
                ld_babip_lookup[int(lr["player_id"])] = ld_dev * 0.25
            logger.info("LD BABIP adjustments: %d batters", len(ld_babip_lookup))
    except FileNotFoundError:
        pass

    speed_babip_lookup: dict[int, float] = {}
    try:
        speed_df = pd.read_parquet(DASHBOARD_DIR / "batter_sprint_speed.parquet")
        if not speed_df.empty:
            speed_latest = speed_df.sort_values("season").groupby("player_id").last().reset_index()
            for _, sr in speed_latest.iterrows():
                speed_dev = float(sr["sprint_speed_regressed"]) - 27.0
                speed_babip_lookup[int(sr["player_id"])] = speed_dev * 0.010
            logger.info("Speed BABIP adjustments: %d batters", len(speed_babip_lookup))
    except FileNotFoundError:
        pass

    bip_lookup: dict[int, np.ndarray] = {}
    _LEAGUE_BIP_PROBS = np.array([0.700, 0.222, 0.065, 0.005])
    try:
        bip_df = pd.read_parquet(DASHBOARD_DIR / "batter_bip_profiles.parquet")
        if not bip_df.empty:
            bip_latest = bip_df.sort_values("season").groupby("batter_id").last().reset_index()
            for _, br in bip_latest.iterrows():
                bip_lookup[int(br["batter_id"])] = np.array([
                    br["p_out"], br["p_single"], br["p_double"], br["p_triple"],
                ], dtype=np.float64)
            logger.info("BIP profiles: %d batters", len(bip_lookup))
    except FileNotFoundError:
        logger.info("No batter_bip_profiles.parquet — using league-average BIP")

    # Run sims
    logger.info("Running full-game sims (n_sims=%d)...", args.n_sims)
    preds = _run_full_game_sims(
        games=games_with_odds,
        lineups_df=lineups,
        pitcher_k=pitcher_k,
        pitcher_bb=pitcher_bb,
        pitcher_hr=pitcher_hr,
        hitter_k=hitter_k,
        hitter_bb=hitter_bb,
        hitter_hr=hitter_hr,
        bf_priors=bf_priors,
        bullpen_rates=bullpen_rates,
        pitcher_arsenal=pitcher_arsenal,
        hitter_vuln=hitter_vuln,
        baselines_pt=baselines_pt,
        ump_k_lookup=ump_k_lookup,
        ump_bb_lookup=ump_bb_lookup,
        wx_lookup=wx_lookup,
        park_lift_lookup=park_lift_lookup,
        ld_babip_lookup=ld_babip_lookup,
        speed_babip_lookup=speed_babip_lookup,
        bip_lookup=bip_lookup,
        n_sims=args.n_sims,
    )

    if preds.empty:
        logger.error("No games simulated")
        sys.exit(1)

    # Merge actuals
    preds = preds.merge(
        scores.rename(columns={
            "home_runs": "actual_home_runs",
            "away_runs": "actual_away_runs",
        }),
        on="game_pk", how="left",
    )
    preds["actual_total"] = preds["actual_home_runs"] + preds["actual_away_runs"]
    preds["actual_home_win"] = (preds["actual_home_runs"] > preds["actual_away_runs"]).astype(int)
    preds["actual_home_margin"] = preds["actual_home_runs"] - preds["actual_away_runs"]

    # Merge odds
    preds = preds.merge(odds_pivoted, on="game_pk", how="left")

    elapsed_sim = (time.time() - t0) / 60
    logger.info(
        "Sims complete: %d games in %.1f min", len(preds), elapsed_sim,
    )

    # --- Print results ---
    print("\n" + "=" * 75)
    print(f"FULL-GAME SIM vs MARKET ODDS  ({start_date} to {end_date})")
    print(f"  n_sims={args.n_sims}, preseason posteriors (no 2026 leakage)")
    print(f"  Games simulated: {len(preds)}")
    print("=" * 75)

    # Moneyline
    ml_metrics = _evaluate_moneyline(preds)
    if ml_metrics:
        n = ml_metrics.get("ml_n", 0)
        print(f"\n--- MONEYLINE ({n} games) ---")
        print(f"{'':>16s} {'Brier':>8s} {'Accuracy':>10s} {'ECE':>8s}")
        print(f"{'Model':>16s} "
              f"{ml_metrics.get('ml_model_brier', float('nan')):>8.4f} "
              f"{ml_metrics.get('ml_model_accuracy', float('nan'))*100:>9.1f}% "
              f"{ml_metrics.get('ml_model_ece', float('nan')):>8.4f}")
        print(f"{'Market':>16s} "
              f"{ml_metrics.get('ml_market_brier', float('nan')):>8.4f} "
              f"{ml_metrics.get('ml_market_accuracy', float('nan'))*100:>9.1f}% "
              f"{ml_metrics.get('ml_market_ece', float('nan')):>8.4f}")
        print(f"  Coin-flip Brier: 0.2500")

        print(f"\n  Edge P&L (flat $100 bets at -110):")
        print(f"  {'Threshold':>10s} {'Bets':>6s} {'Win%':>7s} {'P&L':>8s} {'ROI':>7s}")
        for thr in [0.0, 0.03, 0.05, 0.08]:
            key = f"{thr:.0%}"
            nb = ml_metrics.get(f"ml_n_bets_{key}")
            if nb is None:
                continue
            wp = ml_metrics.get(f"ml_win_pct_{key}", 0)
            pnl = ml_metrics.get(f"ml_pnl_{key}", 0)
            roi = ml_metrics.get(f"ml_roi_{key}", 0)
            print(f"  {'>'+key+' edge':>10s} {nb:>6d} {wp:>6.1f}% ${pnl:>7.0f} {roi:>+6.1f}%")

    # Over/Under
    ou_metrics = _evaluate_totals(preds)
    if ou_metrics:
        n = ou_metrics.get("ou_n", 0)
        print(f"\n--- OVER/UNDER ({n} games) ---")
        print(f"{'':>16s} {'Brier':>8s} {'Accuracy':>10s}")
        print(f"{'Model':>16s} "
              f"{ou_metrics.get('ou_model_brier', float('nan')):>8.4f} "
              f"{ou_metrics.get('ou_model_accuracy', float('nan'))*100:>9.1f}%")
        if "ou_market_brier" in ou_metrics:
            print(f"{'Market':>16s} "
                  f"{ou_metrics.get('ou_market_brier', float('nan')):>8.4f} "
                  f"{ou_metrics.get('ou_market_accuracy', float('nan'))*100:>9.1f}%")
        print(f"  Total runs bias: {ou_metrics.get('ou_total_bias', float('nan')):+.2f}")
        print(f"  Total runs corr: {ou_metrics.get('ou_total_corr', float('nan')):.3f}")

        if any(k.startswith("ou_pnl_") for k in ou_metrics):
            print(f"\n  O/U Edge P&L:")
            print(f"  {'Threshold':>10s} {'Bets':>6s} {'P&L':>8s} {'ROI':>7s}")
            for thr in [0.0, 0.03, 0.05, 0.08]:
                key = f"{thr:.0%}"
                nb = ou_metrics.get(f"ou_n_bets_{key}")
                if nb is None:
                    continue
                pnl = ou_metrics.get(f"ou_pnl_{key}", 0)
                roi = ou_metrics.get(f"ou_roi_{key}", 0)
                print(f"  {'>'+key+' edge':>10s} {nb:>6d} ${pnl:>7.0f} {roi:>+6.1f}%")

    # Spread
    sp_metrics = _evaluate_spread(preds)
    if sp_metrics:
        n = sp_metrics.get("sp_n", 0)
        print(f"\n--- SPREAD / RUN LINE ({n} games) ---")
        print(f"{'':>16s} {'Brier':>8s} {'Accuracy':>10s}")
        print(f"{'Model':>16s} "
              f"{sp_metrics.get('sp_model_brier', float('nan')):>8.4f} "
              f"{sp_metrics.get('sp_model_accuracy', float('nan'))*100:>9.1f}%")
        if "sp_market_brier" in sp_metrics:
            print(f"{'Market':>16s} "
                  f"{sp_metrics.get('sp_market_brier', float('nan')):>8.4f} "
                  f"{sp_metrics.get('sp_market_accuracy', float('nan'))*100:>9.1f}%")
        print(f"  Margin correlation: {sp_metrics.get('sp_margin_corr', float('nan')):.3f}")

    # Per-game detail
    print(f"\n--- PER-GAME DETAIL (sample) ---")
    detail_cols = ["game_date", "away_abbr", "home_abbr",
                   "pred_home_win_prob", "ml_home_fair",
                   "pred_total_runs", "total_line",
                   "actual_home_runs", "actual_away_runs"]
    avail = [c for c in detail_cols if c in preds.columns]
    sample = preds[avail].head(15)
    print(sample.round(3).to_string(index=False))

    # Save
    out_path = Path("outputs/sim_vs_game_odds_backtest.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path, index=False)
    logger.info("Saved predictions to %s", out_path)

    elapsed = (time.time() - t0) / 60
    print(f"\nCompleted in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
