"""Depth chart cascading for PA/IP redistribution.

When starters miss games due to injury, redistributes their PA/IP
to backups based on the team's depth chart. Replacement-level players
fill in when all depth options are exhausted.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Replacement-level production (per 600 PA or 180 IP)
# These represent the quality of a "freely available" AAA/waiver player
REPLACEMENT_HITTER_WRC_PLUS = 75
REPLACEMENT_PITCHER_ERA = 5.20
REPLACEMENT_PITCHER_FIP = 4.80

# Standard PA per game by lineup position
PA_PER_GAME_BY_SLOT = {
    1: 4.8, 2: 4.6, 3: 4.5, 4: 4.4, 5: 4.3,
    6: 4.2, 7: 4.1, 8: 4.0, 9: 3.9,
}
TEAM_PA_PER_GAME = sum(PA_PER_GAME_BY_SLOT.values())  # ~38.8


def cascade_hitter_pa(
    starters: pd.DataFrame,
    bench: pd.DataFrame,
    games_missed: dict[int, int],
    total_games: int = 162,
) -> pd.DataFrame:
    """Redistribute PA from injured starters to bench/replacement.

    Parameters
    ----------
    starters : pd.DataFrame
        Starting 9. Columns: player_id, position, projected_pa, value_score.
    bench : pd.DataFrame
        Bench players. Same columns.
    games_missed : dict[int, int]
        player_id -> games missed this sim season.
    total_games : int
        Season length.

    Returns
    -------
    pd.DataFrame
        Adjusted roster with redistributed PA. Includes replacement-level
        players if bench is exhausted.
    """
    roster = pd.concat([starters, bench], ignore_index=True).copy()
    roster["games_missed"] = roster["player_id"].map(games_missed).fillna(0).astype(int)
    roster["games_available"] = (total_games - roster["games_missed"]).clip(0)

    # Scale projected PA by availability fraction
    avail_frac = roster["games_available"] / total_games
    roster["adjusted_pa"] = (roster["projected_pa"] * avail_frac).round().astype(int)

    # Total PA lost from injuries
    pa_lost = (roster["projected_pa"] - roster["adjusted_pa"]).sum()

    if pa_lost <= 0:
        return roster

    # Distribute lost PA to available bench players (pro-rated by availability)
    bench_mask = roster.index >= len(starters)
    available_bench = roster[bench_mask & (roster["games_available"] > 20)].copy()

    if len(available_bench) > 0:
        # Each bench player can absorb PA up to their available games * avg PA/game
        max_absorb = (available_bench["games_available"] * 3.5).clip(0)
        # Already have some PA — can absorb more
        room = (max_absorb - available_bench["adjusted_pa"]).clip(0)
        total_room = room.sum()

        if total_room > 0:
            absorb_frac = room / total_room
            pa_to_bench = min(pa_lost, total_room)
            roster.loc[available_bench.index, "adjusted_pa"] += (
                absorb_frac * pa_to_bench
            ).round().astype(int)
            pa_lost -= pa_to_bench

    # Remaining PA goes to replacement level
    if pa_lost > 50:
        replacement_row = {
            "player_id": -1,
            "position": "UTIL",
            "projected_pa": int(pa_lost),
            "adjusted_pa": int(pa_lost),
            "value_score": 0.0,
            "games_missed": 0,
            "games_available": total_games,
            "is_replacement": True,
        }
        roster = pd.concat([roster, pd.DataFrame([replacement_row])], ignore_index=True)

    if "is_replacement" not in roster.columns:
        roster["is_replacement"] = False

    return roster


def cascade_pitcher_ip(
    starters: pd.DataFrame,
    relievers: pd.DataFrame,
    games_missed: dict[int, int],
    total_games: int = 162,
) -> pd.DataFrame:
    """Redistribute IP from injured starters to depth/replacement.

    Parameters
    ----------
    starters : pd.DataFrame
        Top 5 SP. Columns: player_id, projected_ip, value_score, role.
    relievers : pd.DataFrame
        Bullpen. Same columns.
    games_missed : dict[int, int]
        player_id -> games missed.
    total_games : int
        Season length.

    Returns
    -------
    pd.DataFrame
        Adjusted pitching staff with redistributed IP.
    """
    roster = pd.concat([starters, relievers], ignore_index=True).copy()
    roster["games_missed"] = roster["player_id"].map(games_missed).fillna(0).astype(int)
    roster["games_available"] = (total_games - roster["games_missed"]).clip(0)

    avail_frac = roster["games_available"] / total_games
    roster["adjusted_ip"] = (roster["projected_ip"] * avail_frac).round(1)

    ip_lost = (roster["projected_ip"] - roster["adjusted_ip"]).sum()

    if ip_lost <= 0:
        return roster

    # Distribute to available relievers (they can absorb more innings)
    rp_mask = roster["role"].isin(["RP", "SU", "MR", "CL"]) if "role" in roster.columns else roster.index >= len(starters)
    available_rp = roster[rp_mask & (roster["games_available"] > 20)].copy()

    if len(available_rp) > 0:
        # Each reliever can absorb ~20-30 extra IP
        room = np.full(len(available_rp), 25.0)
        total_room = room.sum()
        if total_room > 0:
            ip_to_rp = min(ip_lost, total_room)
            absorb_frac = room / total_room
            roster.loc[available_rp.index, "adjusted_ip"] += (
                absorb_frac * ip_to_rp
            ).round(1)
            ip_lost -= ip_to_rp

    # Remaining IP to replacement-level pitching
    if ip_lost > 10:
        replacement_row = {
            "player_id": -2,
            "projected_ip": float(ip_lost),
            "adjusted_ip": float(ip_lost),
            "value_score": 0.0,
            "role": "SP",
            "games_missed": 0,
            "games_available": total_games,
            "is_replacement": True,
        }
        roster = pd.concat([roster, pd.DataFrame([replacement_row])], ignore_index=True)

    if "is_replacement" not in roster.columns:
        roster["is_replacement"] = False

    return roster
