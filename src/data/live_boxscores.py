"""Fetch completed-game boxscores from the MLB Stats API for live standouts."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd

from src.data.queries._common import _WOBA_WEIGHTS
from src.data.schedule import MLB_API_BASE

logger = logging.getLogger(__name__)


def _api_get(url: str, timeout: int = 15) -> dict:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_completed_game_pks(game_date: str | None = None) -> list[int]:
    """Return game_pk list for Final games on *game_date* (default today)."""
    if game_date is None:
        game_date = date.today().isoformat()

    url = f"{MLB_API_BASE}/schedule?date={game_date}&sportId=1"
    try:
        data = _api_get(url)
    except Exception as e:
        logger.warning("Schedule fetch failed for %s: %s", game_date, e)
        return []

    pks: list[int] = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = (g.get("status") or {}).get("detailedState", "")
            if status == "Final":
                pks.append(g["gamePk"])
    logger.info("Found %d completed games on %s", len(pks), game_date)
    return pks


def _fetch_single_boxscore(game_pk: int) -> dict:
    url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
    return _api_get(url)


def _parse_ip(ip_str: str | float | None) -> float:
    """Convert MLB IP notation (6.2 = 6 2/3) to true float."""
    if ip_str is None:
        return 0.0
    val = float(ip_str)
    whole = int(val)
    frac = round(val - whole, 1)
    return whole + frac / 0.3 if frac > 0 else float(whole)


def _fetch_raw_boxscores(
    game_date: str | None = None,
) -> tuple[dict[int, dict], str]:
    """Fetch raw boxscore JSON for all completed games on *game_date*.

    Returns (raw_boxes, resolved_date) so callers share one HTTP round.
    """
    pks = fetch_completed_game_pks(game_date)
    resolved = game_date or date.today().isoformat()
    if not pks:
        return {}, resolved

    raw_boxes: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_single_boxscore, pk): pk for pk in pks}
        for fut in as_completed(futs):
            pk = futs[fut]
            try:
                raw_boxes[pk] = fut.result()
            except Exception as e:
                logger.warning("Boxscore fetch failed for %d: %s", pk, e)
    return raw_boxes, resolved


def _parse_hitter_lines(
    raw_boxes: dict[int, dict],
    game_date: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for gpk, box in raw_boxes.items():
        teams = box.get("teams", {})
        away_abbr = teams.get("away", {}).get("team", {}).get("abbreviation", "")
        home_abbr = teams.get("home", {}).get("team", {}).get("abbreviation", "")

        for side in ("away", "home"):
            team_data = teams.get(side, {})
            players = team_data.get("players", {})

            for pid_key, pdata in players.items():
                stats = (pdata.get("stats") or {}).get("batting", {})
                if not stats or stats.get("plateAppearances", 0) == 0:
                    continue

                person = pdata.get("person", {})
                pa = int(stats.get("plateAppearances", 0))
                ab = int(stats.get("atBats", 0))
                hits = int(stats.get("hits", 0))
                doubles = int(stats.get("doubles", 0))
                triples = int(stats.get("triples", 0))
                hr = int(stats.get("homeRuns", 0))
                tb = int(stats.get("totalBases", 0))
                runs = int(stats.get("runs", 0))
                rbi = int(stats.get("rbi", 0))
                bb = int(stats.get("baseOnBalls", 0))
                ibb = int(stats.get("intentionalWalks", 0))
                hbp = int(stats.get("hitByPitch", 0))
                k = int(stats.get("strikeOuts", 0))
                sb = int(stats.get("stolenBases", 0))
                cs = int(stats.get("caughtStealing", 0))
                sf = int(stats.get("sacFlies", 0))

                rows.append({
                    "batter_id": person.get("id"),
                    "batter_name": person.get("fullName", "Unknown"),
                    "game_pk": gpk,
                    "game_date": game_date,
                    "away_team": away_abbr,
                    "home_team": home_abbr,
                    "pa": pa, "ab": ab, "hits": hits,
                    "doubles": doubles, "triples": triples, "hr": hr,
                    "total_bases": tb, "runs": runs, "rbi": rbi,
                    "bb": bb, "ibb": ibb, "hbp": hbp,
                    "k": k, "sb": sb, "cs": cs, "sf": sf,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    ab_s = df["ab"].replace(0, float("nan"))
    pa_s = df["pa"].replace(0, float("nan"))
    non_ibb_bb = (df["bb"] - df["ibb"]).clip(lower=0)

    df["avg"] = (df["hits"] / ab_s).fillna(0.0)
    df["slg"] = (df["total_bases"] / ab_s).fillna(0.0)
    obp_num = df["hits"] + df["bb"] + df["hbp"]
    obp_den = (df["ab"] + df["bb"] + df["hbp"] + df["sf"]).replace(0, float("nan"))
    df["obp"] = (obp_num / obp_den).fillna(0.0)
    df["ops"] = df["obp"] + df["slg"]
    df["iso"] = (df["slg"] - df["avg"]).fillna(0.0)

    singles = df["hits"] - df["doubles"] - df["triples"] - df["hr"]
    woba_num = (
        non_ibb_bb * _WOBA_WEIGHTS["ubb"]
        + df["hbp"] * _WOBA_WEIGHTS["hbp"]
        + singles * _WOBA_WEIGHTS["single"]
        + df["doubles"] * _WOBA_WEIGHTS["double"]
        + df["triples"] * _WOBA_WEIGHTS["triple"]
        + df["hr"] * _WOBA_WEIGHTS["hr"]
    )
    woba_den = (df["ab"] + non_ibb_bb + df["sf"] + df["hbp"]).replace(0, float("nan"))
    df["woba"] = (woba_num / woba_den).fillna(0.0)

    df["k_rate"] = (df["k"] / pa_s).fillna(0.0)
    df["bb_rate"] = (df["bb"] / pa_s).fillna(0.0)

    df["xwoba"] = np.nan
    df["hard_hit_pct"] = 0.0
    df["barrel_pct"] = 0.0
    df["bip"] = 0
    df["max_hr_distance"] = np.nan

    return df


def _parse_pitcher_lines(
    raw_boxes: dict[int, dict],
    game_date: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for gpk, box in raw_boxes.items():
        teams = box.get("teams", {})
        away_abbr = teams.get("away", {}).get("team", {}).get("abbreviation", "")
        home_abbr = teams.get("home", {}).get("team", {}).get("abbreviation", "")

        for side in ("away", "home"):
            team_data = teams.get(side, {})
            players = team_data.get("players", {})
            pitchers_list = team_data.get("pitchers", [])
            starter_id = pitchers_list[0] if pitchers_list else None

            for pid_key, pdata in players.items():
                stats = (pdata.get("stats") or {}).get("pitching", {})
                if not stats or stats.get("battersFaced", 0) == 0:
                    continue

                person = pdata.get("person", {})
                pid = person.get("id")
                bf = int(stats.get("battersFaced", 0))
                ip = _parse_ip(stats.get("inningsPitched", "0"))
                k = int(stats.get("strikeOuts", 0))
                bb = int(stats.get("baseOnBalls", 0)) + int(stats.get("intentionalWalks", 0))
                hr = int(stats.get("homeRuns", 0))
                hits = int(stats.get("hits", 0))
                er = int(stats.get("earnedRuns", 0))
                runs_allowed = int(stats.get("runs", 0))
                w = int(stats.get("wins", 0))
                l = int(stats.get("losses", 0))
                sv = int(stats.get("saves", 0))
                hld = int(stats.get("holds", 0))
                pitches = int(stats.get("numberOfPitches", 0))

                rows.append({
                    "pitcher_id": pid,
                    "pitcher_name": person.get("fullName", "Unknown"),
                    "game_pk": gpk,
                    "game_date": game_date,
                    "away_team": away_abbr,
                    "home_team": home_abbr,
                    "is_starter": pid == starter_id,
                    "bf": bf, "ip": ip, "k": k, "bb": bb,
                    "hr": hr, "hits": hits, "er": er, "runs": runs_allowed,
                    "w": w, "l": l, "sv": sv, "hld": hld, "pitches": pitches,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    bf_s = df["bf"].replace(0, float("nan"))
    ip_s = df["ip"].replace(0, float("nan"))
    df["k_rate"] = (df["k"] / bf_s).fillna(0.0)
    df["bb_rate"] = (df["bb"] / bf_s).fillna(0.0)
    df["k_minus_bb"] = df["k_rate"] - df["bb_rate"]
    df["era"] = ((df["er"] / ip_s) * 9).fillna(0.0)
    df["whip"] = ((df["hits"] + df["bb"]) / ip_s).fillna(0.0)
    df["hr_per_9"] = ((df["hr"] / ip_s) * 9).fillna(0.0)
    df["role"] = np.where(df["is_starter"], "SP", "RP")
    df["qs"] = ((df["ip"] >= 6.0) & (df["er"] <= 3)).astype(int)

    return df


def fetch_live_boxscores(
    game_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch and parse hitter + pitcher boxscores in one pass.

    Returns ``{"hitters": DataFrame, "pitchers": DataFrame}``.
    """
    raw_boxes, resolved_date = _fetch_raw_boxscores(game_date)
    if not raw_boxes:
        return {"hitters": pd.DataFrame(), "pitchers": pd.DataFrame()}

    hitters = _parse_hitter_lines(raw_boxes, resolved_date)
    pitchers = _parse_pitcher_lines(raw_boxes, resolved_date)
    logger.info(
        "Live boxscores: %d hitter lines, %d pitcher lines from %d games",
        len(hitters), len(pitchers), len(raw_boxes),
    )
    return {"hitters": hitters, "pitchers": pitchers}


def fetch_live_hitter_boxscores(game_date: str | None = None) -> pd.DataFrame:
    """Convenience wrapper — hitter lines only."""
    return fetch_live_boxscores(game_date)["hitters"]


def fetch_live_pitcher_boxscores(game_date: str | None = None) -> pd.DataFrame:
    """Convenience wrapper — pitcher lines only."""
    return fetch_live_boxscores(game_date)["pitchers"]
