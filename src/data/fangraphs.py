"""
FanGraphs Depth Charts projection data loader.

Loads FanGraphs Depth Charts projected fWAR from CSV exports,
matches players to MLBAM IDs via dim_player, and returns
standardized DataFrames for comparison with TDD rankings.

Usage:
    1. Download Depth Charts projections from FanGraphs:
       - Batters:  https://www.fangraphs.com/projections?type=depthcharts&stats=bat&pos=all
       - Pitchers: https://www.fangraphs.com/projections?type=depthcharts&stats=pit&pos=all
    2. Export to CSV (requires FanGraphs membership)
    3. Place CSVs in data/fangraphs/

Expected CSV columns:
    Batters:  Name, Team, G, PA, HR, R, RBI, SB, BB%, K%, ISO, BABIP, AVG, OBP, SLG, wOBA, wRC+, BsR, Off, Def, WAR
    Pitchers: Name, Team, W, L, SV, G, GS, IP, K/9, BB/9, HR/9, BABIP, ERA, FIP, WAR (+ others)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

FANGRAPHS_DIR = Path("data/fangraphs")


def _normalize_name(name: str) -> str:
    """Normalize player name for fuzzy matching."""
    name = name.strip().lower()
    # Remove accents/diacritics common in Latin names
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ñ": "n", "ü": "u", "ö": "o", "ä": "a",
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    # Remove Jr., Sr., III, II, etc.
    name = re.sub(r"\s+(jr\.?|sr\.?|ii+|iv|v)$", "", name)
    # Remove periods and extra whitespace
    name = name.replace(".", "").strip()
    return name


def _load_player_lookup() -> pd.DataFrame:
    """Load dim_player for MLBAM ID matching."""
    df = read_sql("""
        SELECT player_id, player_name, primary_position, bat_side, pitch_hand
        FROM production.dim_player
        WHERE active = true
    """)
    df["name_normalized"] = df["player_name"].apply(_normalize_name)
    return df


def load_hitter_projections(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load FanGraphs Depth Charts hitter projections from CSV.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to CSV file. Defaults to data/fangraphs/depthcharts_batters.csv

    Returns
    -------
    pd.DataFrame
        Columns: player_id (MLBAM), player_name, fg_name, team, pa, war,
                 wrc_plus, woba, off, def_, bsr, plus original FG columns.
    """
    if csv_path is None:
        csv_path = FANGRAPHS_DIR / "depthcharts_batters.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"FanGraphs CSV not found at {csv_path}. "
            "Download from: https://www.fangraphs.com/projections"
            "?type=depthcharts&stats=bat&pos=all"
        )

    fg = pd.read_csv(csv_path)
    logger.info("Loaded %d hitter projections from %s", len(fg), csv_path)

    # Standardize column names
    col_map = {}
    for col in fg.columns:
        cl = col.strip().lower().replace("+", "_plus").replace("/", "_per_")
        cl = cl.replace("%", "_pct").replace(" ", "_")
        col_map[col] = cl
    fg = fg.rename(columns=col_map)

    # Ensure key columns exist
    war_col = _find_column(fg, ["war", "fwar"])
    name_col = _find_column(fg, ["name", "player_name", "playername"])
    team_col = _find_column(fg, ["team", "tm"])

    fg = fg.rename(columns={
        war_col: "war",
        name_col: "fg_name",
        team_col: "team",
    })

    # Parse numeric WAR if needed
    fg["war"] = pd.to_numeric(fg["war"], errors="coerce")

    # Optional columns
    for alias, candidates in [
        ("pa", ["pa"]),
        ("wrc_plus", ["wrc_plus", "wrc+"]),
        ("woba", ["woba"]),
        ("off", ["off"]),
        ("def_", ["def", "def_"]),
        ("bsr", ["bsr"]),
    ]:
        found = _find_column(fg, candidates, required=False)
        if found and found != alias:
            fg = fg.rename(columns={found: alias})

    # Match to MLBAM IDs
    fg = _match_players(fg, "hitter")
    fg = fg.sort_values("war", ascending=False).reset_index(drop=True)
    fg["fg_rank"] = range(1, len(fg) + 1)

    matched = fg["player_id"].notna().sum()
    logger.info("Matched %d / %d hitters to MLBAM IDs", matched, len(fg))
    return fg


def load_pitcher_projections(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load FanGraphs Depth Charts pitcher projections from CSV.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to CSV file. Defaults to data/fangraphs/depthcharts_pitchers.csv

    Returns
    -------
    pd.DataFrame
        Columns: player_id (MLBAM), player_name, fg_name, team, ip, war,
                 era, fip, k_per_9, bb_per_9, plus original FG columns.
    """
    if csv_path is None:
        csv_path = FANGRAPHS_DIR / "depthcharts_pitchers.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"FanGraphs CSV not found at {csv_path}. "
            "Download from: https://www.fangraphs.com/projections"
            "?type=depthcharts&stats=pit&pos=all"
        )

    fg = pd.read_csv(csv_path)
    logger.info("Loaded %d pitcher projections from %s", len(fg), csv_path)

    # Standardize column names
    col_map = {}
    for col in fg.columns:
        cl = col.strip().lower().replace("+", "_plus").replace("/", "_per_")
        cl = cl.replace("%", "_pct").replace(" ", "_")
        col_map[col] = cl
    fg = fg.rename(columns=col_map)

    war_col = _find_column(fg, ["war", "fwar"])
    name_col = _find_column(fg, ["name", "player_name", "playername"])
    team_col = _find_column(fg, ["team", "tm"])

    fg = fg.rename(columns={
        war_col: "war",
        name_col: "fg_name",
        team_col: "team",
    })

    fg["war"] = pd.to_numeric(fg["war"], errors="coerce")

    for alias, candidates in [
        ("ip", ["ip"]),
        ("era", ["era"]),
        ("fip", ["fip"]),
        ("k_per_9", ["k_per_9", "k/9"]),
        ("bb_per_9", ["bb_per_9", "bb/9"]),
        ("gs", ["gs"]),
    ]:
        found = _find_column(fg, candidates, required=False)
        if found and found != alias:
            fg = fg.rename(columns={found: alias})

    fg = _match_players(fg, "pitcher")
    fg = fg.sort_values("war", ascending=False).reset_index(drop=True)
    fg["fg_rank"] = range(1, len(fg) + 1)

    matched = fg["player_id"].notna().sum()
    logger.info("Matched %d / %d pitchers to MLBAM IDs", matched, len(fg))
    return fg


def _find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    """Find first matching column name (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    if required:
        raise KeyError(
            f"None of {candidates} found in columns: {list(df.columns)}"
        )
    return None


def _match_players(fg: pd.DataFrame, player_type: str) -> pd.DataFrame:
    """Match FanGraphs names to MLBAM IDs via dim_player.

    Uses normalized name matching with team hint for disambiguation.
    """
    lookup = _load_player_lookup()

    # Check if FG data already has playerid or mlbamid
    if "mlbamid" in fg.columns:
        fg = fg.rename(columns={"mlbamid": "player_id"})
        fg["player_name"] = fg["fg_name"]
        return fg
    if "playerid" in fg.columns:
        # FanGraphs internal ID — not directly usable, but some CSVs
        # also include mlbamid
        pass

    fg["name_normalized"] = fg["fg_name"].apply(_normalize_name)

    # Build name -> player_id mapping, preferring unique matches
    name_to_id = {}
    name_to_fullname = {}
    for norm_name, group in lookup.groupby("name_normalized"):
        if len(group) == 1:
            name_to_id[norm_name] = group.iloc[0]["player_id"]
            name_to_fullname[norm_name] = group.iloc[0]["player_name"]
        else:
            # Multiple players with same name — store all for team disambiguation
            name_to_id[norm_name] = group[["player_id", "player_name"]].to_dict("records")
            name_to_fullname[norm_name] = group.iloc[0]["player_name"]

    player_ids = []
    player_names = []
    for _, row in fg.iterrows():
        norm = row["name_normalized"]
        match = name_to_id.get(norm)
        if match is None:
            player_ids.append(None)
            player_names.append(row["fg_name"])
        elif isinstance(match, int):
            player_ids.append(match)
            player_names.append(name_to_fullname[norm])
        else:
            # Ambiguous — take first match (could improve with team matching)
            player_ids.append(match[0]["player_id"])
            player_names.append(match[0]["player_name"])
            logger.debug(
                "Ambiguous match for '%s' — took %s",
                row["fg_name"], match[0]["player_name"],
            )

    fg["player_id"] = player_ids
    fg["player_name"] = player_names
    fg = fg.drop(columns=["name_normalized"])
    return fg
