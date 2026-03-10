#!/usr/bin/env python
"""
Build preseason injuries parquet from ESPN injury data.

This is a temporary pre-season tool — once the season starts,
real IL transactions flow through dim_transaction.

Usage
-----
    python scripts/build_preseason_injuries.py
"""
from __future__ import annotations

import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db import read_sql


def _strip_accents(s: str) -> str:
    """Remove accent marks for fuzzy matching."""
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


# Map ESPN team abbreviations to database abbreviations
_TEAM_ABBR_MAP = {
    "ARI": "AZ",
    "OAK": "ATH",
}


# ── ESPN injury data (scraped 2026-03-09) ──────────────────────────────
# fmt: off
RAW_INJURIES = [
    # (team, player_name, position, injury, status, est_return)
    # Arizona Diamondbacks
    ("ARI", "Corbin Carroll", "RF", "Hand injury", "Out", "2026-03-16"),
    ("ARI", "Merrill Kelly", "SP", "Back injury", "Out", "2026-04-01"),
    ("ARI", "Adrian Del Castillo", "C", "Calf strain", "Out", "2026-03-26"),
    ("ARI", "Cristian Mena", "RP", "Shoulder injury", "Out", "2026-03-26"),
    ("ARI", "Lourdes Gurriel Jr.", "LF", "Knee injury", "Out", "2026-05-01"),
    ("ARI", "Tyler Locklear", "1B", "Shoulder/elbow injury", "Out", "2026-05-18"),
    ("ARI", "A.J. Puk", "RP", "Elbow injury", "60-Day IL", "2026-06-29"),
    ("ARI", "Corbin Burnes", "SP", "Elbow injury", "60-Day IL", "2026-07-17"),
    ("ARI", "Justin Martinez", "RP", "Elbow injury", "60-Day IL", "2026-08-21"),
    ("ARI", "Andrew Saalfrank", "RP", "Shoulder injury", "60-Day IL", "2026-02-01"),
    ("ARI", "Derek Law", "RP", "Forearm injury", "Out", "2026-07-01"),
    ("ARI", "Kyle Amendt", "RP", "Shoulder injury", "Out", "2026-02-01"),
    ("ARI", "Tommy Henry", "RP", "Elbow injury", "Out", "2026-09-01"),
    ("ARI", "Blake Walston", "SP", "Elbow injury", "Out", "2026-07-01"),
    # Athletics
    ("OAK", "Ben Bowden", "RP", "Lat strain", "Day-to-Day", "2026-03-11"),
    ("OAK", "Colby Thomas", "RF", "Right elbow inflammation", "Day-to-Day", "2026-03-10"),
    # Atlanta Braves
    ("ATL", "Joey Wentz", "RP", "Knee injury", "Out", "2026-02-01"),
    ("ATL", "Daysbel Hernandez", "RP", "Shoulder injury", "Day-to-Day", "2026-03-11"),
    ("ATL", "Hurston Waldrep", "SP", "Right elbow loose bodies", "Out", "2026-06-02"),
    ("ATL", "Spencer Schwellenbach", "SP", "Right elbow bone spurs", "60-Day IL", "2026-06-02"),
    ("ATL", "Ha-Seong Kim", "SS", "Finger injury", "Out", "2026-05-01"),
    ("ATL", "Joe Jimenez", "RP", "Knee injury", "60-Day IL", "2026-07-17"),
    ("ATL", "Sean Murphy", "C", "Hip injury", "Out", "2026-05-12"),
    ("ATL", "Danny Young", "RP", "Elbow injury", "Out", "2026-07-01"),
    ("ATL", "AJ Smith-Shawver", "SP", "Elbow injury", "Out", "2026-08-01"),
    # Baltimore Orioles
    ("BAL", "Jackson Holliday", "2B", "Hand injury", "Out", "2026-04-10"),
    ("BAL", "Andrew Kittredge", "RP", "Right shoulder inflammation", "Out", "2026-04-10"),
    ("BAL", "Felix Bautista", "RP", "Shoulder injury", "60-Day IL", "2026-02-01"),
    ("BAL", "Jordan Westburg", "3B", "Partially torn UCL", "Out", "2026-05-01"),
    ("BAL", "Colin Selby", "RP", "Right shoulder inflammation", "60-Day IL", "2026-05-25"),
    ("BAL", "Hans Crouse", "RP", "Unspecified", "Out", "2026-03-26"),
    # Boston Red Sox
    ("BOS", "Triston Casas", "1B", "Knee injury", "Out", "2026-05-01"),
    ("BOS", "Romy Gonzalez", "1B", "Left shoulder injury", "Out", "2026-04-10"),
    ("BOS", "Brendan Rodgers", "2B", "Shoulder injury", "Out", "2026-05-01"),
    ("BOS", "Kutter Crawford", "SP", "Wrist/illness", "Out", "2026-03-26"),
    ("BOS", "Hobie Harris", "RP", "Forearm tightness", "Out", "2026-04-10"),
    ("BOS", "Tanner Houck", "SP", "Elbow injury", "60-Day IL", "2026-02-01"),
    # Chicago Cubs
    ("CHC", "Jordan Wicks", "RP", "Radial nerve irritation", "Out", "2026-04-15"),
    ("CHC", "Porter Hodge", "RP", "Right flexor tendon strain", "Out", "2026-04-12"),
    ("CHC", "Shelby Miller", "RP", "Elbow injury", "60-Day IL", "2026-02-01"),
    ("CHC", "Justin Steele", "SP", "Elbow injury", "Out", "2026-05-01"),
    ("CHC", "Brandon Birdsell", "RP", "Elbow injury", "Out", "2026-05-01"),
    # Chicago White Sox
    ("CWS", "Drew Thorpe", "SP", "Elbow injury", "Out", "2026-06-01"),
    ("CWS", "Prelander Berroa", "RP", "Elbow injury TJS", "Out", "2026-05-01"),
    ("CWS", "Tim Elko", "1B", "Knee injury", "Out", "2026-07-01"),
    ("CWS", "Ky Bush", "SP", "Elbow injury", "Out", "2026-04-01"),
    ("CWS", "Mason Adams", "RP", "Elbow injury TJS", "Out", "2026-07-01"),
    # Cincinnati Reds
    ("CIN", "Hunter Greene", "SP", "Right elbow soreness", "Out", "2026-04-15"),
    ("CIN", "Carson Spiers", "SP", "Elbow injury", "Out", "2026-09-01"),
    ("CIN", "Alex Young", "RP", "Left elbow surgery", "Out", "2026-04-01"),
    # Cleveland Guardians
    ("CLE", "Carlos Hernandez", "RP", "Arm/leg fractures", "Out", "2026-05-01"),
    # Colorado Rockies
    ("COL", "Kris Bryant", "DH", "Back pain", "60-Day IL", "2026-06-01"),
    ("COL", "Jeff Criswell", "RP", "Elbow injury", "60-Day IL", "2026-06-01"),
    ("COL", "RJ Petit", "RP", "Unspecified", "Out", "2026-06-01"),
    ("COL", "Case Williams", "SP", "Right triceps stress reaction", "Out", "2026-03-20"),
    # Detroit Tigers
    ("DET", "Jackson Jobe", "SP", "Elbow injury", "60-Day IL", "2026-09-01"),
    ("DET", "Reese Olson", "SP", "Right shoulder labral repair", "60-Day IL", "2026-02-21"),
    ("DET", "Troy Melton", "RP", "Elbow injury", "Out", "2026-04-14"),
    ("DET", "Josue Briceno", "C", "Right wrist tendon repair", "Out", "2026-06-01"),
    # Houston Astros
    ("HOU", "Josh Hader", "RP", "Biceps injury", "Out", "2026-04-10"),
    ("HOU", "Jeremy Pena", "SS", "Right ring finger fracture", "Out", "2026-03-26"),
    ("HOU", "Nate Pearson", "RP", "Right elbow soreness", "Out", "2026-04-10"),
    ("HOU", "Hayden Wesneski", "SP", "Elbow injury", "Out", "2026-08-01"),
    ("HOU", "Ronel Blanco", "SP", "Elbow injury", "Out", "2026-08-01"),
    # Kansas City Royals
    ("KC", "Alec Marsh", "SP", "Shoulder injury", "60-Day IL", "2026-02-01"),
    # Los Angeles Angels
    ("LAA", "Anthony Rendon", "3B", "Hip injury", "Out", "2026-02-01"),
    ("LAA", "Ben Joyce", "RP", "Shoulder injury", "Out", "2026-03-26"),
    ("LAA", "Josh Lowe", "RF", "Oblique injury", "Day-to-Day", "2026-03-11"),
    # Los Angeles Dodgers
    ("LAD", "Tommy Edman", "2B", "Ankle injury", "Out", "2026-05-01"),
    ("LAD", "Enrique Hernandez", "1B", "Elbow injury", "60-Day IL", "2026-05-24"),
    ("LAD", "Evan Phillips", "RP", "Elbow injury", "60-Day IL", "2026-07-17"),
    ("LAD", "Gavin Stone", "SP", "Right shoulder inflammation", "Out", "2026-04-24"),
    ("LAD", "Brusdar Graterol", "RP", "Shoulder injury", "Out", "2026-05-01"),
    ("LAD", "Blake Snell", "SP", "Shoulder injury", "Out", "2026-04-10"),
    ("LAD", "Bobby Miller", "SP", "Shoulder injury", "Day-to-Day", "2026-03-12"),
    ("LAD", "Brock Stewart", "RP", "Shoulder injury", "Out", "2026-04-10"),
    # Miami Marlins
    ("MIA", "Ronny Henriquez", "RP", "Elbow injury", "60-Day IL", "2026-02-01"),
    # Milwaukee Brewers
    ("MIL", "Gerson Garabito", "RP", "Broken foot bone", "Out", "2026-06-21"),
    ("MIL", "Quinn Priester", "SP", "Wrist injury", "Out", "2026-03-31"),
    # Minnesota Twins
    ("MIN", "Pablo Lopez", "SP", "Elbow injury", "60-Day IL", "2026-04-01"),
    ("MIN", "David Festa", "SP", "Shoulder injury", "Out", "2026-04-10"),
    ("MIN", "Matt Canterino", "SP", "Shoulder injury", "Out", "2026-05-01"),
    ("MIN", "Joe Ryan", "SP", "Back injury", "Day-to-Day", "2026-03-10"),
    # New York Mets
    ("NYM", "Francisco Lindor", "SS", "Hand injury", "Out", "2026-03-26"),
    ("NYM", "A.J. Minter", "RP", "Lat injury", "Out", "2026-05-01"),
    ("NYM", "Dedniel Nunez", "RP", "Elbow injury", "60-Day IL", "2026-09-01"),
    ("NYM", "Tylor Megill", "SP", "Elbow injury", "60-Day IL", "2026-02-01"),
    ("NYM", "Reed Garrett", "RP", "Elbow injury", "60-Day IL", "2026-02-01"),
    # New York Yankees
    ("NYY", "Anthony Volpe", "SS", "Shoulder injury", "Out", "2026-04-24"),
    ("NYY", "Carlos Rodon", "SP", "Elbow injury", "Out", "2026-04-25"),
    ("NYY", "Clarke Schmidt", "SP", "Elbow injury", "60-Day IL", "2026-08-01"),
    ("NYY", "Gerrit Cole", "SP", "Elbow injury", "Out", "2026-06-02"),
    # Philadelphia Phillies
    ("PHI", "Zack Wheeler", "SP", "Shoulder injury", "Out", "2026-04-10"),
    ("PHI", "Christian McGowan", "RP", "Elbow injury TJS", "Out", "2026-07-01"),
    # Pittsburgh Pirates
    ("PIT", "Jared Jones", "SP", "Elbow injury", "60-Day IL", "2026-05-25"),
    # San Diego Padres
    ("SD", "Griffin Canning", "SP", "Achilles injury", "Out", "2026-04-27"),
    ("SD", "Matt Waldron", "SP", "Lower body injury", "Out", "2026-04-09"),
    ("SD", "Yuki Matsui", "RP", "Left groin tightness", "Out", "2026-03-26"),
    ("SD", "Jason Adam", "RP", "Quadriceps injury", "Out", "2026-03-26"),
    ("SD", "Jhony Brito", "RP", "Forearm injury", "60-Day IL", "2026-05-01"),
    # San Francisco Giants
    ("SF", "Jason Foley", "RP", "Shoulder injury", "60-Day IL", "2026-07-01"),
    ("SF", "Rowan Wick", "RP", "Elbow injury", "60-Day IL", "2026-02-01"),
    ("SF", "Randy Rodriguez", "RP", "Elbow injury", "60-Day IL", "2026-02-01"),
    ("SF", "Reiver Sanmartin", "RP", "Minor quadriceps strain", "Out", "2026-06-09"),
    # Seattle Mariners
    ("SEA", "Logan Evans", "SP", "UCL reconstruction", "60-Day IL", "2026-04-01"),
    # St. Louis Cardinals
    ("STL", "Lars Nootbaar", "LF", "Heel injury", "Out", "2026-04-10"),
    ("STL", "Sem Robberse", "SP", "Elbow injury", "Out", "2026-07-01"),
    # Tampa Bay Rays
    ("TB", "Manuel Rodriguez", "RP", "Flexor tendon/UCL", "60-Day IL", "2026-06-01"),
    ("TB", "Austin Vernon", "RP", "Right elbow soreness", "Out", "2026-04-10"),
    ("TB", "Edwin Uceta", "RP", "Shoulder injury", "Out", "2026-04-10"),
    ("TB", "Steven Wilson", "RP", "Lower back injury", "Out", "2026-04-10"),
    # Texas Rangers
    ("TEX", "Jordan Montgomery", "SP", "Elbow injury", "60-Day IL", "2026-07-01"),
    ("TEX", "Josh Jung", "3B", "Hamstring injury", "Out", "2026-03-26"),
    ("TEX", "Cody Bradford", "SP", "Elbow injury", "Out", "2026-05-01"),
    ("TEX", "Sebastian Walcott", "SS", "Right elbow internal brace", "Out", "2026-08-03"),
    # Toronto Blue Jays
    ("TOR", "Shane Bieber", "SP", "Forearm injury", "Out", "2026-04-14"),
    ("TOR", "Bowden Francis", "SP", "UCL reconstruction", "60-Day IL", "2026-04-01"),
    ("TOR", "Anthony Santander", "RF", "Left shoulder/back injury", "Out", "2026-08-01"),
    ("TOR", "Yimi Garcia", "RP", "Right elbow post-surgery", "Out", "2026-04-10"),
    # Washington Nationals
    ("WSH", "Trevor Williams", "SP", "Elbow injury", "60-Day IL", "2026-06-01"),
    ("WSH", "DJ Herz", "SP", "Elbow injury", "60-Day IL", "2026-07-01"),
    ("WSH", "Jarlin Susana", "RP", "Right lat tear", "Out", "2026-07-01"),
    ("WSH", "Tyler Stuart", "SP", "Elbow injury TJS", "Out", "2026-09-01"),
]
# fmt: on


def _estimate_missed_games(est_return: str, season_start: str = "2026-03-26") -> int:
    """Estimate games missed based on return date vs season start."""
    ret = datetime.strptime(est_return, "%Y-%m-%d")
    start = datetime.strptime(season_start, "%Y-%m-%d")
    if ret <= start:
        return 0  # Expected back by opening day
    # ~1 game per day on average (162 games in ~183 days)
    days_missed = (ret - start).days
    return int(days_missed * 162 / 183)


def _categorize_severity(status: str, missed_games: int) -> str:
    """Categorize injury severity for dashboard display."""
    if "60-Day" in status:
        return "major"
    if missed_games >= 60:
        return "major"
    if missed_games >= 20:
        return "significant"
    if missed_games > 0:
        return "minor"
    return "healthy_by_opener"


def build_preseason_injuries() -> pd.DataFrame:
    """Build preseason injuries DataFrame with player_id matching."""
    # Build DataFrame from raw data
    df = pd.DataFrame(
        RAW_INJURIES,
        columns=["team_abbr", "player_name", "position", "injury",
                 "status", "est_return_date"],
    )

    # Normalize team abbreviations to match database conventions
    df["team_abbr"] = df["team_abbr"].replace(_TEAM_ABBR_MAP)

    # Compute derived fields
    df["est_missed_games"] = df["est_return_date"].apply(_estimate_missed_games)
    df["severity"] = df.apply(
        lambda r: _categorize_severity(r["status"], r["est_missed_games"]),
        axis=1,
    )
    df["is_pitcher"] = df["position"].isin(["SP", "RP"])

    # ── Match to dim_player ─────────────────────────────────────────
    players = read_sql(
        "SELECT player_id, player_name FROM production.dim_player", {}
    )

    # Build normalized name → player_id lookup
    name_to_id: dict[str, int] = {}
    for _, row in players.iterrows():
        norm = _strip_accents(row["player_name"]).lower().strip()
        name_to_id[norm] = row["player_id"]

    # Also build last_name lookup for fallback
    lastname_to_ids: dict[str, list[tuple[str, int]]] = {}
    for _, row in players.iterrows():
        norm = _strip_accents(row["player_name"]).lower().strip()
        parts = norm.split()
        if parts:
            last = parts[-1]
            lastname_to_ids.setdefault(last, []).append((norm, row["player_id"]))

    matched = []
    unmatched = []
    for _, row in df.iterrows():
        espn_name = row["player_name"]
        norm_espn = _strip_accents(espn_name).lower().strip()

        # Direct match
        if norm_espn in name_to_id:
            matched.append(name_to_id[norm_espn])
            continue

        # Try common name variations
        # "A.J. Puk" → "aj puk", "Enrique Hernandez" → "enrique hernandez"
        alt = norm_espn.replace(".", "").replace("'", "")
        if alt in name_to_id:
            matched.append(name_to_id[alt])
            continue

        # Last name fallback — find best match with same last name
        parts = norm_espn.split()
        last = parts[-1] if parts else ""
        candidates = lastname_to_ids.get(last, [])
        if len(candidates) == 1:
            matched.append(candidates[0][1])
            continue

        # Multiple candidates — try first initial match
        found = False
        if parts and candidates:
            first_init = parts[0][0]
            for cand_name, cand_id in candidates:
                cand_parts = cand_name.split()
                if cand_parts and cand_parts[0][0] == first_init:
                    matched.append(cand_id)
                    found = True
                    break
        if not found:
            matched.append(None)
            unmatched.append(espn_name)

    df["player_id"] = matched

    if unmatched:
        print(f"WARNING: {len(unmatched)} players could not be matched to dim_player:")
        for name in unmatched:
            print(f"  - {name}")

    # Show match rate
    n_matched = df["player_id"].notna().sum()
    print(f"\nMatched {n_matched}/{len(df)} players ({n_matched/len(df):.0%})")

    return df


def main() -> None:
    df = build_preseason_injuries()

    out_path = PROJECT_ROOT / "data" / "dashboard" / "preseason_injuries.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} injuries to {out_path}")

    # Summary
    print(f"\nSeverity breakdown:")
    print(df["severity"].value_counts().to_string())
    print(f"\nBy position type:")
    print(f"  Pitchers: {df['is_pitcher'].sum()}")
    print(f"  Hitters:  {(~df['is_pitcher']).sum()}")
    print(f"\nMajor injuries (60+ games or 60-day IL):")
    major = df[df["severity"] == "major"].sort_values("est_missed_games", ascending=False)
    for _, r in major.iterrows():
        pid_str = f" (id={int(r['player_id'])})" if pd.notna(r["player_id"]) else " (NO MATCH)"
        print(f"  {r['player_name']}{pid_str} — {r['injury']} — ~{r['est_missed_games']}G missed")


if __name__ == "__main__":
    main()
