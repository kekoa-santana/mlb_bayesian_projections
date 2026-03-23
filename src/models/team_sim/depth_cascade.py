"""Depth chart cascading for PA/IP redistribution.

When starters miss games due to injury, redistributes their PA/IP
to position-eligible backups, then to any available bench player,
and finally to team-specific replacement level.

Uses actual team bench quality instead of a fixed replacement level.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# League-average replacement level (used only when team has NO bench data)
FALLBACK_REPLACEMENT_WRC_PLUS = 75
FALLBACK_REPLACEMENT_ERA = 5.20

# Standard PA per game by lineup slot
TEAM_PA_PER_GAME = 38.5


def cascade_hitter_pa(
    starters: pd.DataFrame,
    bench: pd.DataFrame,
    games_missed: dict[int, int],
    position_eligibility: pd.DataFrame | None = None,
    total_games: int = 162,
) -> pd.DataFrame:
    """Redistribute PA from injured starters to position-eligible backups.

    Cascade order:
    1. Position-eligible bench player (from position_eligibility)
    2. Any available bench player
    3. Team-specific replacement (average of worst bench players)

    Parameters
    ----------
    starters : pd.DataFrame
        Starting 9. Columns: player_id, position, projected_pa, value_score.
    bench : pd.DataFrame
        Bench players. Same columns.
    games_missed : dict[int, int]
        player_id -> games missed this sim season.
    position_eligibility : pd.DataFrame, optional
        Columns: player_id, position, is_primary. Multi-position eligibility.
    total_games : int
        Season length.

    Returns
    -------
    pd.DataFrame
        Adjusted roster with redistributed PA.
    """
    roster = pd.concat([starters, bench], ignore_index=True).copy()
    n_starters = len(starters)
    roster["games_missed"] = roster["player_id"].map(games_missed).fillna(0).astype(int)
    roster["games_available"] = (total_games - roster["games_missed"]).clip(0)
    roster["is_replacement"] = False

    # Scale projected PA by availability
    avail_frac = roster["games_available"] / total_games
    roster["adjusted_pa"] = (roster["projected_pa"] * avail_frac).round().astype(int)

    # Build position eligibility lookup: player_id -> set of eligible positions
    pos_eligible: dict[int, set] = {}
    if position_eligibility is not None:
        for _, row in position_eligibility.iterrows():
            pid = int(row["player_id"])
            pos_eligible.setdefault(pid, set()).add(row["position"])
    # Also add primary position from roster data
    for _, row in roster.iterrows():
        pid = int(row["player_id"])
        if "position" in row and pd.notna(row.get("position")):
            pos_eligible.setdefault(pid, set()).add(str(row["position"]))

    # Compute team-specific replacement quality (worst 3 bench players' average)
    bench_mask = roster.index >= n_starters
    bench_players = roster[bench_mask].sort_values("value_score")
    if len(bench_players) >= 2:
        team_replacement_value = bench_players["value_score"].iloc[:3].mean()
    else:
        team_replacement_value = 0.0  # will use fallback

    # Track PA lost per position from starter injuries
    pa_lost_by_pos: dict[str, float] = {}
    for idx in range(n_starters):
        row = roster.iloc[idx]
        pa_lost = float(row["projected_pa"] - row["adjusted_pa"])
        if pa_lost > 0:
            pos = str(row.get("position", "UTIL"))
            pa_lost_by_pos[pos] = pa_lost_by_pos.get(pos, 0) + pa_lost

    total_pa_lost = sum(pa_lost_by_pos.values())
    if total_pa_lost <= 0:
        return roster

    # Track which bench players have already absorbed PA
    bench_absorbed: dict[int, float] = {}  # pid -> PA already given
    remaining_pa_lost = total_pa_lost

    # --- Pass 1: Position-eligible bench players ---
    for pos, pa_needed in sorted(pa_lost_by_pos.items(), key=lambda x: -x[1]):
        if pa_needed <= 0:
            continue
        # Find bench players eligible at this position
        for idx in bench_players.index:
            brow = roster.loc[idx]
            pid = int(brow["player_id"])
            if brow["games_available"] <= 20:
                continue
            eligible_positions = pos_eligible.get(pid, set())
            if pos not in eligible_positions and pos != "DH":
                continue
            # How much PA can this bench player absorb?
            already = bench_absorbed.get(pid, 0)
            max_pa = brow["games_available"] * 3.5 - already
            if max_pa <= 10:
                continue
            absorb = min(pa_needed, max_pa)
            roster.loc[idx, "adjusted_pa"] += int(absorb)
            bench_absorbed[pid] = already + absorb
            pa_needed -= absorb
            remaining_pa_lost -= absorb
            if pa_needed <= 0:
                break
        pa_lost_by_pos[pos] = pa_needed

    # --- Pass 2: Any available bench player (for remaining PA) ---
    if remaining_pa_lost > 50:
        for idx in bench_players.sort_values("value_score", ascending=False).index:
            brow = roster.loc[idx]
            pid = int(brow["player_id"])
            if brow["games_available"] <= 20:
                continue
            already = bench_absorbed.get(pid, 0)
            max_pa = brow["games_available"] * 3.5 - already
            if max_pa <= 10:
                continue
            absorb = min(remaining_pa_lost, max_pa)
            roster.loc[idx, "adjusted_pa"] += int(absorb)
            bench_absorbed[pid] = already + absorb
            remaining_pa_lost -= absorb
            if remaining_pa_lost <= 50:
                break

    # --- Pass 3: Team-specific replacement level ---
    if remaining_pa_lost > 50:
        roster = pd.concat([roster, pd.DataFrame([{
            "player_id": -1,
            "position": "UTIL",
            "projected_pa": int(remaining_pa_lost),
            "adjusted_pa": int(remaining_pa_lost),
            "value_score": team_replacement_value,
            "games_missed": 0,
            "games_available": total_games,
            "is_replacement": True,
        }])], ignore_index=True)

    return roster


def cascade_pitcher_ip(
    starters: pd.DataFrame,
    relievers: pd.DataFrame,
    games_missed: dict[int, int],
    total_games: int = 162,
) -> pd.DataFrame:
    """Redistribute IP from injured starters to depth arms.

    Cascade order:
    1. 6th/7th starters (SP-eligible relievers with most IP)
    2. Long relievers / swingmen
    3. Team-specific replacement level (worst bullpen arms)

    Parameters
    ----------
    starters : pd.DataFrame
        Top 5 SP. Columns: player_id, projected_ip, value_score, role.
    relievers : pd.DataFrame
        Bullpen. Same columns.
    games_missed : dict[int, int]
    total_games : int

    Returns
    -------
    pd.DataFrame
        Adjusted pitching staff with redistributed IP.
    """
    roster = pd.concat([starters, relievers], ignore_index=True).copy()
    n_starters = len(starters)
    roster["games_missed"] = roster["player_id"].map(games_missed).fillna(0).astype(int)
    roster["games_available"] = (total_games - roster["games_missed"]).clip(0)
    roster["is_replacement"] = False

    avail_frac = roster["games_available"] / total_games
    roster["adjusted_ip"] = (roster["projected_ip"] * avail_frac).round(1)

    ip_lost = (roster["projected_ip"] - roster["adjusted_ip"]).sum()
    if ip_lost <= 0:
        return roster

    # Team-specific replacement: average of worst 3 relievers
    rp_by_value = roster.iloc[n_starters:].sort_values("value_score")
    if len(rp_by_value) >= 2:
        team_replacement_value = rp_by_value["value_score"].iloc[:3].mean()
    else:
        team_replacement_value = 0.0

    remaining_ip = ip_lost

    # --- Pass 1: Non-top-5 SPs (6th/7th starters, swingmen) ---
    # Sort relievers by projected IP descending (long-relief types first)
    long_relief = roster.iloc[n_starters:].sort_values("projected_ip", ascending=False)
    for idx in long_relief.index:
        row = roster.loc[idx]
        if row["games_available"] <= 10:
            continue
        # Each reliever can absorb ~20-40 extra IP
        base_ip = float(row["projected_ip"])
        max_extra = max(base_ip * 0.3, 20.0)  # up to 30% more, min 20 IP
        absorb = min(remaining_ip, max_extra)
        roster.loc[idx, "adjusted_ip"] += absorb
        remaining_ip -= absorb
        if remaining_ip <= 5:
            break

    # --- Pass 2: Team-specific replacement ---
    if remaining_ip > 5:
        roster = pd.concat([roster, pd.DataFrame([{
            "player_id": -2,
            "projected_ip": float(remaining_ip),
            "adjusted_ip": float(remaining_ip),
            "value_score": team_replacement_value,
            "role": "SP",
            "games_missed": 0,
            "games_available": total_games,
            "is_replacement": True,
        }])], ignore_index=True)

    return roster
