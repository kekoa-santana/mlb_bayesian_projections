"""
Backfill production.dim_player from the MLB Stats API.

Collects every unique batter_id and pitcher_id from production.fact_pitch,
fetches each player's bio from the MLB Stats people endpoint in batches,
merges in per-player average sz_top / sz_bot computed from sat_pitch_shape,
then TRUNCATEs dim_player and re-inserts the full set.

Usage
-----
    cd player_profiles
    source myenv/Scripts/activate      # or myenv/bin/activate on Linux
    python scripts/backfill_dim_player.py

Optional flags
--------------
    --dry-run   Fetch and build the DataFrame but do not write to the DB.
    --batch N   Number of player IDs per API request (default 150).
    --delay F   Seconds to sleep between API batches (default 0.3).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
from sqlalchemy import text

from src.data.db import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
DEFAULT_BATCH  = 150
DEFAULT_DELAY  = 0.3   # seconds between API batches


# ---------------------------------------------------------------------------
# Step 1: collect all player IDs from the database
# ---------------------------------------------------------------------------
def get_all_player_ids(engine) -> list[int]:
    """Return sorted list of every unique batter_id + pitcher_id in fact_pitch."""
    sql = """
    SELECT DISTINCT pid
    FROM (
        SELECT batter_id  AS pid FROM production.fact_pitch
        UNION
        SELECT pitcher_id AS pid FROM production.fact_pitch
    ) t
    ORDER BY pid
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        ids = [row[0] for row in result]
    logger.info("Found %d unique player IDs in fact_pitch", len(ids))
    return ids


# ---------------------------------------------------------------------------
# Step 2: fetch player bios from MLB Stats API in batches
# ---------------------------------------------------------------------------
def fetch_batch(player_ids: list[int], retries: int = 3) -> list[dict[str, Any]]:
    """Fetch a batch of players from the MLB Stats people endpoint."""
    params = {
        "personIds": ",".join(str(p) for p in player_ids),
        "hydrate":   "currentTeam",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(MLB_PEOPLE_URL, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("people", [])
        except requests.RequestException as exc:
            logger.warning("Batch attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    logger.error("All retries exhausted for batch starting with id=%d", player_ids[0])
    return []


def fetch_all_players(
    player_ids: list[int],
    batch_size: int = DEFAULT_BATCH,
    delay: float = DEFAULT_DELAY,
) -> list[dict[str, Any]]:
    """Fetch all players in batches, return list of raw API dicts."""
    all_people: list[dict[str, Any]] = []
    batches = [player_ids[i:i + batch_size] for i in range(0, len(player_ids), batch_size)]
    logger.info("Fetching %d players in %d batches of ~%d", len(player_ids), len(batches), batch_size)

    for i, batch in enumerate(batches, 1):
        people = fetch_batch(batch)
        all_people.extend(people)
        logger.info("Batch %d/%d — fetched %d people (total so far: %d)",
                    i, len(batches), len(people), len(all_people))
        if i < len(batches):
            time.sleep(delay)

    logger.info("API fetch complete: %d players retrieved out of %d requested",
                len(all_people), len(player_ids))
    return all_people


# ---------------------------------------------------------------------------
# Step 3: fetch per-player avg sz_top / sz_bot from sat_pitch_shape
# ---------------------------------------------------------------------------
def get_sz_averages(engine) -> pd.DataFrame:
    """Compute per-batter average strike-zone top/bottom from Statcast data."""
    sql = """
    SELECT
        fp.batter_id                                          AS player_id,
        AVG(CASE WHEN sps.sz_top != 'NaN' THEN sps.sz_top END) AS sz_top,
        AVG(CASE WHEN sps.sz_bot != 'NaN' THEN sps.sz_bot END) AS sz_bot
    FROM production.fact_pitch fp
    JOIN production.sat_pitch_shape sps ON fp.pitch_id = sps.pitch_id
    WHERE sps.sz_top IS NOT NULL
      AND sps.sz_bot IS NOT NULL
    GROUP BY fp.batter_id
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info("sz_top/sz_bot averages computed for %d batters", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 4: parse API response into a DataFrame matching dim_player schema
# ---------------------------------------------------------------------------
def parse_players(people: list[dict[str, Any]]) -> pd.DataFrame:
    """Map MLB Stats API person dicts to dim_player column names."""
    records = []
    for p in people:
        team_id = 0
        ct = p.get("currentTeam")
        if isinstance(ct, dict):
            team_id = ct.get("id", 0) or 0

        pos = p.get("primaryPosition") or {}

        records.append({
            "player_id":            int(p["id"]),
            "player_name":          p.get("fullName") or "",
            "team_id":              int(team_id),
            "first_name":           p.get("firstName"),
            "last_name":            p.get("lastName"),
            "birth_date":           p.get("birthDate"),
            "age":                  p.get("currentAge"),
            "height":               p.get("height"),
            "weight":               p.get("weight"),
            "active":               p.get("active", False),
            "primary_position_code": _safe_int(pos.get("code")),
            "primary_position":     pos.get("abbreviation") or pos.get("name"),
            "draft_year":           p.get("draftYear"),
            "mlb_debut_date":       p.get("mlbDebutDate"),
            "bat_side":             (p.get("batSide")   or {}).get("code"),
            "pitch_hand":           (p.get("pitchHand") or {}).get("code"),
            # sz_top / sz_bot filled in from Statcast averages below
            "sz_top":               None,
            "sz_bot":               None,
        })

    df = pd.DataFrame(records)

    # Cast nullable int columns to Int64 (pandas nullable int) to survive NaN
    for col in ["age", "primary_position_code", "draft_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Cast dates
    for col in ["birth_date", "mlb_debut_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    logger.info("Parsed %d player records from API response", len(df))
    return df


def _safe_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Step 5: merge sz averages and write to DB
# ---------------------------------------------------------------------------
def merge_sz(players: pd.DataFrame, sz: pd.DataFrame) -> pd.DataFrame:
    """Merge Statcast sz_top / sz_bot averages into the players DataFrame."""
    merged = players.merge(sz, on="player_id", how="left", suffixes=("_api", "_sc"))
    # Prefer Statcast computed values (more accurate than static API value)
    merged["sz_top"] = merged["sz_top_sc"].where(merged["sz_top_sc"].notna(), merged["sz_top_api"])
    merged["sz_bot"] = merged["sz_bot_sc"].where(merged["sz_bot_sc"].notna(), merged["sz_bot_api"])
    merged = merged.drop(columns=["sz_top_api", "sz_bot_api", "sz_top_sc", "sz_bot_sc"], errors="ignore")
    return merged


def write_to_db(df: pd.DataFrame, engine) -> None:
    """TRUNCATE dim_player and insert all rows."""
    logger.info("Writing %d rows to production.dim_player …", len(df))
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE production.dim_player"))
        df.to_sql(
            "dim_player",
            conn,
            schema="production",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
    logger.info("Done — production.dim_player now has %d rows", len(df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill production.dim_player")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data but do not write to the DB")
    parser.add_argument("--batch",  type=int,   default=DEFAULT_BATCH,
                        help="Player IDs per API request")
    parser.add_argument("--delay",  type=float, default=DEFAULT_DELAY,
                        help="Seconds between API batches")
    args = parser.parse_args()

    engine = get_engine()

    # 1. Collect IDs
    player_ids = get_all_player_ids(engine)

    # 2. Fetch from API
    people = fetch_all_players(player_ids, batch_size=args.batch, delay=args.delay)
    if not people:
        logger.error("No players returned from API — aborting")
        return

    # 3. sz averages from Statcast
    sz_df = get_sz_averages(engine)

    # 4. Parse
    players_df = parse_players(people)

    # 5. Merge sz
    players_df = merge_sz(players_df, sz_df)

    logger.info("Final player DataFrame: %d rows, %d columns",
                len(players_df), len(players_df.columns))
    logger.info("Columns: %s", list(players_df.columns))
    logger.info("bat_side distribution:\n%s", players_df["bat_side"].value_counts(dropna=False))
    logger.info("pitch_hand distribution:\n%s", players_df["pitch_hand"].value_counts(dropna=False))

    # Check for players requested but missing from API
    fetched_ids  = set(players_df["player_id"])
    requested_ids = set(player_ids)
    missing = requested_ids - fetched_ids
    if missing:
        logger.warning(
            "%d player IDs not returned by API (likely test/invalid IDs): %s",
            len(missing), sorted(missing)[:20],
        )

    if args.dry_run:
        logger.info("DRY RUN — skipping DB write. Sample output:")
        print(players_df.head(10).to_string(index=False))
        return

    # 6. Write
    write_to_db(players_df, engine)

    # 7. Verify
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM production.dim_player")).scalar()
    logger.info("Verification: dim_player now has %d rows", count)


if __name__ == "__main__":
    main()
