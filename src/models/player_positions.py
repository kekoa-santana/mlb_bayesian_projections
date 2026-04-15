"""Position assignment logic for the player rankings system.

Extracted from ``player_rankings.py`` -- zero behavioral changes.

Queries ``fact_lineup``, optionally hits the MLB Stats API, and returns
position DataFrames for hitters and pitchers.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models._ranking_utils import DASHBOARD_DIR

logger = logging.getLogger(__name__)

# When weighted_games tie across positions, pick one row per player (defensive value order).
_POSITION_TIEBREAK_PRIORITY: dict[str, int] = {
    "C": 0, "SS": 1, "2B": 2, "3B": 3, "CF": 4, "LF": 5, "RF": 6, "1B": 7, "DH": 8,
}


def _get_career_weighted_positions(season: int = 2025) -> pd.DataFrame:
    """Query career position starts with recency weighting.

    Weighting scheme (by game recency within each player):
      - Last 50 games:  3x  (catches recent position switches)
      - Games 51-162:   2x  (recent full season)
      - Games 163+:     1x  (career history)

    If 80%+ of the last 50 games are at a single position that differs
    from the career-weighted primary, the recent position overrides
    (e.g., Bichette: career SS but playing 3B for the Mets in 2026).

    Parameters
    ----------
    season : int
        Most recent completed season (lineup data through this season
        plus any spring training / early games from season + 1).

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, weighted_games, is_primary,
        recent_position (position from last 50 games if override applies).
    """
    from src.data.db import read_sql

    # Career starts with row number for recency weighting
    raw = read_sql(f"""
        WITH numbered AS (
            SELECT player_id, position, season,
                   ROW_NUMBER() OVER (
                       PARTITION BY player_id
                       ORDER BY season DESC, game_pk DESC
                   ) AS rn
            FROM production.fact_lineup
            WHERE is_starter = true
              AND position NOT IN ('P', 'PR', 'PH', 'EH')
              AND season <= {season + 1}
        )
        SELECT player_id, position, rn
        FROM numbered
    """, {})

    if raw.empty:
        return pd.DataFrame(columns=[
            "player_id", "position", "weighted_games", "is_primary",
        ])

    # Recency weights
    raw["weight"] = np.where(
        raw["rn"] <= 50, 3.0,
        np.where(raw["rn"] <= 162, 2.0, 1.0),
    )

    # Weighted games per player-position
    weighted = (
        raw.groupby(["player_id", "position"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "weighted_games"})
    )

    # --- Override 1: Current season (season+1) position ---
    # If a player has 10+ games in ST / early season and 80%+ are at a
    # NEW position, override.  Catches mid-career switches that show up
    # in spring training before the career data has enough weight
    # (e.g., Bichette SS->3B when traded to the Mets).
    from src.data.db import read_sql as _read_next
    next_season_counts = _read_next(f"""
        SELECT player_id, position, COUNT(*) as games
        FROM production.fact_lineup
        WHERE season = {season + 1} AND is_starter = true
          AND position NOT IN ('P', 'PR', 'PH', 'EH')
        GROUP BY player_id, position
    """, {})

    current_override = pd.DataFrame(columns=["player_id", "recent_position"])
    if not next_season_counts.empty:
        ns_totals = next_season_counts.groupby("player_id")["games"].sum().reset_index(name="total")
        ns_top = next_season_counts.sort_values("games", ascending=False).groupby("player_id").first().reset_index()
        ns_top = ns_top.merge(ns_totals, on="player_id")
        ns_top["pct"] = ns_top["games"] / ns_top["total"]
        current_override = ns_top[
            (ns_top["games"] >= 10) & (ns_top["pct"] >= 0.80)
        ][["player_id", "position"]].rename(columns={"position": "recent_position"})

    # --- Override 2: Last 50 games ---
    # Broader check across seasons: if 80%+ of last 50 games are at one
    # position, override.  Catches switches that span a season boundary.
    recent = raw[raw["rn"] <= 50].copy()
    recent_counts = (
        recent.groupby(["player_id", "position"])
        .size()
        .reset_index(name="recent_games")
    )
    recent_totals = recent.groupby("player_id").size().reset_index(name="total_recent")
    recent_counts = recent_counts.merge(recent_totals, on="player_id")
    recent_counts["recent_pct"] = recent_counts["recent_games"] / recent_counts["total_recent"]

    recent_primary = (
        recent_counts
        .sort_values("recent_games", ascending=False)
        .groupby("player_id")
        .first()
        .reset_index()
    )
    last50_override = recent_primary[recent_primary["recent_pct"] >= 0.80][
        ["player_id", "position"]
    ].rename(columns={"position": "recent_position"})

    # Merge overrides: current season takes priority over last-50
    recent_override = pd.concat([last50_override, current_override], ignore_index=True)
    recent_override = recent_override.drop_duplicates(subset="player_id", keep="last")

    # Career-weighted primary = highest weighted_games
    weighted["is_primary"] = (
        weighted["weighted_games"]
        == weighted.groupby("player_id")["weighted_games"].transform("max")
    )

    # Apply last-50 override: if recent position differs from career primary
    primary = weighted[weighted["is_primary"]].merge(
        recent_override, on="player_id", how="left",
    )
    overrides = primary[
        primary["recent_position"].notna()
        & (primary["position"] != primary["recent_position"])
    ][["player_id", "recent_position"]]

    if not overrides.empty:
        override_map = dict(zip(overrides["player_id"], overrides["recent_position"]))
        # Flip primary flag: old primary -> False, new recent position -> True
        for pid, new_pos in override_map.items():
            mask = weighted["player_id"] == pid
            weighted.loc[mask, "is_primary"] = (
                weighted.loc[mask, "position"] == new_pos
            )
            # If the new position doesn't exist in career data, add it
            if not (mask & (weighted["position"] == new_pos)).any():
                weighted = pd.concat([weighted, pd.DataFrame([{
                    "player_id": pid,
                    "position": new_pos,
                    "weighted_games": 50 * 3,  # assume 50 recent games
                    "is_primary": True,
                }])], ignore_index=True)
                # Unflag old primary
                old_primary = mask & weighted["is_primary"] & (weighted["position"] != new_pos)
                weighted.loc[old_primary, "is_primary"] = False
        logger.info(
            "Position override (last 50 games): %d players — %s",
            len(override_map),
            ", ".join(
                f"{pid}->{pos}" for pid, pos in list(override_map.items())[:5]
            ) + ("..." if len(override_map) > 5 else ""),
        )

    return weighted.sort_values(
        ["player_id", "weighted_games"], ascending=[True, False],
    ).reset_index(drop=True)


def _verify_positions_mlb_api(
    positions: pd.DataFrame,
    season: int = 2026,
    lineup_override_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Cross-check position assignments against the MLB Stats API.

    Fetches current rosters from the API and uses the API's
    ``primaryPosition`` as a final sanity check.  Only overrides when
    the API position is a fielding position and our assignment differs.

    Players in ``lineup_override_ids`` are SKIPPED -- their position was
    already set from actual game lineup data (more current than the
    API's static ``primaryPosition`` field which MLB updates slowly).

    Assignments of **DH** are never overridden: roster ``primaryPosition``
    often lists OF/1B for bat-only profiles even when lineup-weighted
    starts are mostly DH; replacing DH with a fielding position wrongly
    applies fielding penalties.

    Parameters
    ----------
    positions : pd.DataFrame
        Must have ``player_id`` and ``position`` (primary only).
    season : int
        Season for roster lookup.
    lineup_override_ids : set[int] or None
        Player IDs whose position was set by lineup override.
        API will not override these.

    Returns
    -------
    pd.DataFrame
        Same as input with positions corrected where API disagrees.
    """
    import requests

    MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
    FIELDING_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}

    # Fetch all 30 teams
    try:
        teams_resp = requests.get(
            f"{MLB_API_BASE}/teams",
            params={"season": season, "sportId": 1},
            timeout=10,
        )
        teams_resp.raise_for_status()
        team_ids = [t["id"] for t in teams_resp.json().get("teams", [])]
    except Exception as e:
        logger.warning("MLB API teams fetch failed: %s", e)
        return positions

    api_positions: dict[int, str] = {}
    api_teams: dict[int, str] = {}
    for tid in team_ids:
        try:
            resp = requests.get(
                f"{MLB_API_BASE}/teams/{tid}/roster",
                params={"season": season, "rosterType": "fullSeason"},
                timeout=10,
            )
            resp.raise_for_status()
            roster = resp.json().get("roster", [])
            # Get team abbreviation
            team_resp = requests.get(
                f"{MLB_API_BASE}/teams/{tid}",
                timeout=10,
            )
            team_data = team_resp.json().get("teams", [{}])[0]
            team_abbr = team_data.get("abbreviation", "")

            for player in roster:
                pid = player.get("person", {}).get("id")
                pos_abbr = player.get("position", {}).get("abbreviation", "")
                if pid and pos_abbr in FIELDING_POSITIONS:
                    api_positions[pid] = pos_abbr
                    api_teams[pid] = team_abbr
        except Exception:
            continue  # skip team on error, don't block

    if not api_positions:
        logger.warning("MLB API returned no roster data -- skipping verification")
        return positions

    # Compare and override where API disagrees
    positions = positions.copy()
    skip_ids = lineup_override_ids or set()
    n_fixed = 0
    n_skipped = 0
    n_skipped_dh = 0
    for idx, row in positions.iterrows():
        pid = row["player_id"]
        if pid in skip_ids:
            n_skipped += 1
            continue
        if row["position"] == "DH":
            n_skipped_dh += 1
            continue
        if pid in api_positions:
            api_pos = api_positions[pid]
            if api_pos != row["position"] and api_pos in FIELDING_POSITIONS:
                logger.debug(
                    "API override: %d %s -> %s (%s)",
                    pid, row["position"], api_pos,
                    api_teams.get(pid, "?"),
                )
                positions.at[idx, "position"] = api_pos
                n_fixed += 1

    logger.info(
        "MLB API verification: %d players checked, %d corrected, "
        "%d skipped (lineup override), %d skipped (DH protected)",
        len(api_positions), n_fixed, n_skipped, n_skipped_dh,
    )
    return positions


def _assign_hitter_positions(season: int = 2025, min_starts: int = 20) -> pd.DataFrame:
    """Assign primary position to each hitter using career-weighted data.

    Uses career lineup starts with recency weighting (last 50 games 3x,
    recent season 2x, career 1x).  Overrides with last-50-games position
    if 80%+ of recent games are at a new position (catches mid-career
    switches like Bichette SS->3B).  Verifies against the MLB Stats API
    as a final sanity check.

    Falls back to ``dim_player.primary_position`` for players with no
    lineup history.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_starts : int
        Minimum weighted games to qualify.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position.
    """
    from src.data.db import read_sql

    # Career-weighted positions with recency bias
    career = _get_career_weighted_positions(season)

    if not career.empty:
        # Multiple rows can share max weighted_games (is_primary True for each tie).
        # Enforce exactly one primary per player_id for downstream merges.
        primary_cands = career[career["is_primary"]][
            ["player_id", "position", "weighted_games"]
        ].copy()
        top_weighted = career.groupby("player_id")["weighted_games"].max().reset_index()
        qualified = top_weighted[top_weighted["weighted_games"] >= min_starts]["player_id"]
        primary_cands = primary_cands[primary_cands["player_id"].isin(qualified)]
        primary_cands["_pos_pri"] = primary_cands["position"].map(
            _POSITION_TIEBREAK_PRIORITY,
        ).fillna(99)
        primary = (
            primary_cands.sort_values(
                ["player_id", "weighted_games", "_pos_pri"],
                ascending=[True, False, True],
            )
            .drop_duplicates("player_id", keep="first")[["player_id", "position"]]
        )
        n_ties = len(primary_cands) - len(primary)
        if n_ties > 0:
            logger.info(
                "Resolved %d tied-primary position rows -> one row per player",
                n_ties,
            )
    else:
        primary = pd.DataFrame(columns=["player_id", "position"])

    # Fallback: dim_player for players not in lineup data
    fallback = read_sql("""
        SELECT player_id, primary_position as position
        FROM production.dim_player
        WHERE primary_position NOT IN ('P', 'TWP')
          AND active = true
    """, {})

    if not primary.empty:
        missing = fallback[~fallback["player_id"].isin(primary["player_id"])]
        assigned = pd.concat(
            [primary, missing[["player_id", "position"]]],
            ignore_index=True,
        )
    else:
        assigned = fallback[["player_id", "position"]].copy()

    # Collect IDs that were overridden by lineup data (current season
    # or last-50 games).  These should NOT be re-overridden by the API's
    # static primaryPosition field which MLB updates slowly.
    lineup_override_ids: set[int] = set()
    if not career.empty:
        # Players whose is_primary position differs from their highest
        # career weighted_games position before override
        raw_primary = (
            career.sort_values("weighted_games", ascending=False)
            .drop_duplicates("player_id", keep="first")
        )
        for _, row in primary.iterrows():
            pid = row["player_id"]
            raw_row = raw_primary[raw_primary["player_id"] == pid]
            if not raw_row.empty and raw_row.iloc[0]["position"] != row["position"]:
                lineup_override_ids.add(int(pid))
        if lineup_override_ids:
            logger.info(
                "Lineup-overridden positions (protected from API): %d players",
                len(lineup_override_ids),
            )

    # MLB API verification (final pass -- skips lineup-overridden players)
    try:
        assigned = _verify_positions_mlb_api(
            assigned, season=season + 1,
            lineup_override_ids=lineup_override_ids,
        )
    except Exception:
        logger.warning("MLB API verification failed -- using lineup-based positions",
                       exc_info=True)

    n_before = len(assigned)
    assigned = assigned.drop_duplicates(subset=["player_id"], keep="first")
    if len(assigned) < n_before:
        logger.warning(
            "Dropped %d duplicate position rows (same player_id)",
            n_before - len(assigned),
        )
    return assigned


def get_hitter_position_eligibility(
    season: int = 2025, min_starts: int = 10,
) -> pd.DataFrame:
    """Return all positions each hitter is eligible for using career data.

    Uses career-weighted lineup starts (recency-biased) rather than a
    single season.  A player qualifies at a position with ``min_starts``
    or more weighted games there.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_starts : int
        Minimum weighted games at a position to be eligible.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, position, starts, pct, is_primary.
    """
    from src.data.db import read_sql

    career = _get_career_weighted_positions(season)

    if career.empty:
        fallback = read_sql("""
            SELECT player_id, primary_position as position
            FROM production.dim_player
            WHERE primary_position NOT IN ('P', 'TWP')
              AND active = true
        """, {})
        fallback["starts"] = 0
        fallback["pct"] = 1.0
        fallback["is_primary"] = True
        return fallback

    # Filter by minimum weighted games
    starts = career[career["weighted_games"] >= min_starts].copy()
    starts = starts.rename(columns={"weighted_games": "starts"})

    # Compute percentage of each player's total
    totals = starts.groupby("player_id")["starts"].transform("sum")
    starts["pct"] = starts["starts"] / totals

    # Add fallback for players not in career data
    fallback = read_sql("""
        SELECT player_id, primary_position as position
        FROM production.dim_player
        WHERE primary_position NOT IN ('P', 'TWP')
          AND active = true
    """, {})
    missing = fallback[~fallback["player_id"].isin(starts["player_id"])]
    if not missing.empty:
        missing = missing.copy()
        missing["starts"] = 0
        missing["pct"] = 1.0
        missing["is_primary"] = True
        starts = pd.concat(
            [starts, missing[["player_id", "position", "starts", "pct", "is_primary"]]],
            ignore_index=True,
        )

    return starts.sort_values(
        ["player_id", "starts"], ascending=[True, False],
    ).reset_index(drop=True)


def _assign_pitcher_roles() -> pd.DataFrame:
    """Assign SP/RP role from pitcher projections.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, role ('SP' or 'RP').
    """
    proj = pd.read_parquet(DASHBOARD_DIR / "pitcher_projections.parquet")
    proj["role"] = np.where(proj["is_starter"] == 1, "SP", "RP")
    return proj[["pitcher_id", "role"]].copy()
