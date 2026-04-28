"""Build optimal DraftKings Classic MLB lineups.

Merges our game simulation projections (DK fantasy points) with live DK
salary data to find the highest-projected lineup under the $50K salary
cap and position constraints.

Usage:
    python scripts/build_dk_lineups.py                   # today, 5 lineups
    python scripts/build_dk_lineups.py --date 2026-04-27
    python scripts/build_dk_lineups.py --n-lineups 20
    python scripts/build_dk_lineups.py --mode tiers      # tiers contest

Requires:
    - game_props.parquet (from precompute/confident_picks.py)
    - DK API reachable (for salary / tier data)
    - PuLP for MILP optimization
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pulp import (
    LpMaximize,
    LpProblem,
    LpVariable,
    lpSum,
    PULP_CBC_CMD,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_DIR.parent / "tdd-dashboard" / "data" / "dashboard"
DK_MODULE_PATH = PROJECT_DIR.parent / "tdd-dashboard" / "lib" / "draftkings.py"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from src.models.game_sim.fantasy_scoring import (
    DK_BAT_SINGLE,
    DK_BAT_DOUBLE,
    DK_BAT_TRIPLE,
    DK_BAT_HR,
    DK_BAT_RBI,
    DK_BAT_R,
    DK_BAT_BB,
    DK_BAT_HBP,
    DK_BAT_SB,
    DK_PIT_IP,
    DK_PIT_K,
    DK_PIT_ER,
    DK_PIT_H,
    DK_PIT_BB,
    DK_PIT_HBP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dk_lineups")

# ---------------------------------------------------------------------------
# DK Classic MLB roster: 10 players
#   2 P, 1 C, 1 1B, 1 2B, 1 3B, 1 SS, 3 OF, 1 UTIL
# Salary cap: $50,000
# ---------------------------------------------------------------------------
SALARY_CAP = 50_000
ROSTER_SLOTS = {
    "P": 2,
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
}
ROSTER_SIZE = sum(ROSTER_SLOTS.values())  # 10


def _load_dk_module():
    """Dynamically load draftkings.py from the dashboard repo."""
    spec = importlib.util.spec_from_file_location("draftkings", DK_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Name matching: DK names → game_props player_id
# ---------------------------------------------------------------------------
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize_name(name: str) -> str:
    """Normalize a name for fuzzy matching (first + last, lowercase)."""
    parts = name.strip().split()
    parts = [p for p in parts if p.lower().rstrip(".") not in _SUFFIXES]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}".lower()
    return name.lower()


def _build_name_index(
    game_props: pd.DataFrame,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """Build name→player_id lookups from game_props.

    Returns
    -------
    name_only : dict[str, int]
        normalized_name → player_id (last wins for collisions).
    name_team : dict[(str, str), int]
        (normalized_name, team_abbr) → player_id (disambiguates
        same-name players on different teams like Max Muncy LAD/ATH).
    """
    name_only: dict[str, int] = {}
    name_team: dict[tuple[str, str], int] = {}
    dedup = game_props[["player_id", "player_name", "team"]].drop_duplicates(
        "player_id"
    )
    for _, row in dedup.iterrows():
        key = _normalize_name(row["player_name"])
        pid = int(row["player_id"])
        name_only[key] = pid
        team = str(row["team"]).upper()
        name_team[(key, team)] = pid
    return name_only, name_team


# ---------------------------------------------------------------------------
# DK fantasy point projection from game_props
# ---------------------------------------------------------------------------
def _compute_batter_dk_points(player_props: pd.DataFrame) -> dict[str, float]:
    """Estimate DK fantasy points from game_props expected values.

    Parameters
    ----------
    player_props : pd.DataFrame
        Rows from game_props for a single batter (one row per stat).

    Returns
    -------
    dict with dk_mean, dk_floor (q10 proxy), dk_ceil (q90 proxy)
    """
    stats = player_props.set_index("stat")["expected"].to_dict()

    h = stats.get("H", 0)
    hr = stats.get("HR", 0) if "HR" in stats else 0
    bb = stats.get("BB", 0)
    r = stats.get("R", 0)
    rbi = stats.get("RBI", 0)
    tb = stats.get("TB", 0)

    # Decompose hits into singles/doubles/triples/HR from TB
    # TB = 1*1B + 2*2B + 3*3B + 4*HR  and  H = 1B + 2B + 3B + HR
    # Extra bases = TB - H = 2B + 2*3B + 3*HR
    # Approximate split: most extra bases are doubles
    extra_bases = max(tb - h, 0)
    hr_tb = hr * 3  # HR accounts for 3 extra bases each
    non_hr_extra = max(extra_bases - hr_tb, 0)
    # Rough split: ~85% of non-HR extra bases are doubles, ~15% triples
    doubles = non_hr_extra * 0.85
    triples = non_hr_extra * 0.15
    singles = max(h - hr - doubles - triples, 0)

    dk_pts = (
        DK_BAT_SINGLE * singles
        + DK_BAT_DOUBLE * doubles
        + DK_BAT_TRIPLE * triples
        + DK_BAT_HR * hr
        + DK_BAT_RBI * rbi
        + DK_BAT_R * r
        + DK_BAT_BB * bb
    )
    return {"dk_mean": round(dk_pts, 2)}


def _compute_pitcher_dk_points(player_props: pd.DataFrame) -> dict[str, float]:
    """Estimate DK fantasy points from game_props expected values.

    Uses dk_mean directly from the game_props row (computed by
    confident_picks from the full simulation).
    """
    # Pitcher rows have dk_mean already computed from the sim
    dk_vals = player_props["dk_mean"].dropna()
    if not dk_vals.empty:
        return {"dk_mean": round(float(dk_vals.iloc[0]), 2)}

    # Fallback: reconstruct from expected stats
    stats = player_props.set_index("stat")["expected"].to_dict()
    k = stats.get("K", 0)
    bb = stats.get("BB", 0)
    h = stats.get("H", 0)
    hr = stats.get("HR", 0)
    outs = stats.get("Outs", 0)
    ip = outs / 3.0

    dk_pts = (
        DK_PIT_IP * ip * 3.0  # 0.75 per out
        + DK_PIT_K * k
        + DK_PIT_ER * (h + bb) * 0.25  # rough ER proxy
        + DK_PIT_H * h
        + DK_PIT_BB * bb
    )
    return {"dk_mean": round(dk_pts, 2)}


def build_player_pool(
    game_date: str,
    mode: str = "classic",
) -> pd.DataFrame:
    """Build the DK player pool with projected fantasy points.

    Merges DK salaries/tiers with our game simulation projections.

    Parameters
    ----------
    game_date : str
        Target date (YYYY-MM-DD).
    mode : str
        'classic' for salary-cap or 'tiers' for tier contests.

    Returns
    -------
    pd.DataFrame
        Player pool with columns: player_name, player_id, position,
        team, salary/tier, dk_proj, value (pts/$1K salary).
    """
    dk = _load_dk_module()

    # Fetch DK player pool
    if mode == "tiers":
        pool = dk.fetch_dk_tiers()
        if pool.empty:
            logger.error("No DK Tiers slate found")
            return pd.DataFrame()
    else:
        pool = dk.fetch_dk_salaries()
        if pool.empty:
            logger.error("No DK Classic slate found")
            return pd.DataFrame()

    raw_count = len(pool)
    logger.info(
        "DK %s pool: %d raw players, salary range $%s–$%s",
        mode,
        raw_count,
        f"{pool.get('salary', pd.Series([0])).min():,}",
        f"{pool.get('salary', pd.Series([0])).max():,}",
    )

    # --- Filter out IL / DTD / OUT players ---
    _EXCLUDE_STATUS = {"IL", "DTD", "OUT"}
    if "status" in pool.columns:
        excluded = pool[pool["status"].isin(_EXCLUDE_STATUS)]
        if len(excluded) > 0:
            logger.info(
                "Filtered %d unavailable players (IL=%d, DTD=%d, OUT=%d)",
                len(excluded),
                (excluded["status"] == "IL").sum(),
                (excluded["status"] == "DTD").sum(),
                (excluded["status"] == "OUT").sum(),
            )
        pool = pool[~pool["status"].isin(_EXCLUDE_STATUS)].copy()

    # --- Deduplicate (DK lists multi-position players multiple times) ---
    before_dedup = len(pool)
    pool = pool.drop_duplicates(
        subset=["player_name", "team", "salary"], keep="first",
    )
    if len(pool) < before_dedup:
        logger.info(
            "Deduplicated %d duplicate rows (%d -> %d)",
            before_dedup - len(pool), before_dedup, len(pool),
        )

    # Load game_props
    gp_path = DASHBOARD_DIR / "game_props.parquet"
    if not gp_path.exists():
        logger.error(
            "game_props.parquet not found at %s. "
            "Run precompute/confident_picks.py first.",
            gp_path,
        )
        return pd.DataFrame()

    gp = pd.read_parquet(gp_path)
    gp = gp[gp["game_date"] == game_date].copy()
    if gp.empty:
        logger.error(
            "No game_props data for %s. "
            "Run precompute/confident_picks.py for today's games.",
            game_date,
        )
        return pd.DataFrame()

    logger.info(
        "Game props: %d rows for %s (%d players)",
        len(gp), game_date, gp["player_id"].nunique(),
    )

    # Build name→id indices (team-aware + name-only fallback)
    name_only_idx, name_team_idx = _build_name_index(gp)

    # Teams with projections today (for validating name-only fallback)
    teams_with_props = set(
        gp["team"].dropna().str.upper().unique()
    ) | set(
        gp["opponent"].dropna().str.upper().unique()
    )

    # Match DK players to our projections (prefer team-aware match)
    def _resolve(row: pd.Series) -> int | None:
        key = _normalize_name(row["player_name"])
        team = str(row.get("team", "")).upper()
        # Try exact (name, team) match first
        pid = name_team_idx.get((key, team))
        if pid is not None:
            return pid
        # Only fall back to name-only if the DK player's team is
        # actually playing today — prevents cross-team mismatches
        # (e.g. ATH Max Muncy resolving to LAD Max Muncy's projections)
        if team not in teams_with_props:
            return None
        return name_only_idx.get(key)

    pool["player_id"] = pool.apply(_resolve, axis=1)
    matched = pool[pool["player_id"].notna()].copy()
    matched["player_id"] = matched["player_id"].astype(int)
    unmatched = pool[pool["player_id"].isna()]
    if len(unmatched) > 0:
        logger.warning(
            "%d DK players not matched to projections: %s",
            len(unmatched),
            ", ".join(unmatched["player_name"].head(10).tolist()),
        )

    # Compute DK projected points per player
    dk_projections: list[dict[str, Any]] = []
    for pid in matched["player_id"].unique():
        player_gp = gp[gp["player_id"] == pid]
        if player_gp.empty:
            continue

        ptype = player_gp["player_type"].iloc[0]
        if ptype == "pitcher":
            pts = _compute_pitcher_dk_points(player_gp)
        else:
            pts = _compute_batter_dk_points(player_gp)

        dk_projections.append({
            "player_id": pid,
            "dk_proj": pts["dk_mean"],
        })

    proj_df = pd.DataFrame(dk_projections)
    if proj_df.empty:
        logger.error("No projections computed")
        return pd.DataFrame()

    # Merge projections into pool
    result = matched.merge(proj_df, on="player_id", how="inner")

    # Value metric for classic mode
    if mode == "classic" and "salary" in result.columns:
        result["value"] = (
            result["dk_proj"] / (result["salary"] / 1000.0)
        ).round(2)

    # Sort by projected points
    result = result.sort_values("dk_proj", ascending=False).reset_index(drop=True)

    logger.info(
        "Player pool ready: %d players with projections "
        "(top: %s %.1f pts @ $%s)",
        len(result),
        result.iloc[0]["player_name"] if len(result) > 0 else "N/A",
        result.iloc[0]["dk_proj"] if len(result) > 0 else 0,
        f"{result.iloc[0].get('salary', 0):,}" if len(result) > 0 else "0",
    )
    return result


# ---------------------------------------------------------------------------
# MILP Lineup Optimizer (Classic)
# ---------------------------------------------------------------------------
def optimize_classic_lineup(
    pool: pd.DataFrame,
    *,
    exclude_players: set[int] | None = None,
    lock_players: set[int] | None = None,
    max_per_game: int = 4,
    objective: str = "projection",
) -> pd.DataFrame | None:
    """Find the optimal DK Classic lineup via integer linear programming.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool from build_player_pool() with columns:
        player_name, player_id, position, team, salary, dk_proj, game.
    exclude_players : set[int], optional
        Player IDs to exclude (for generating diverse lineups).
    lock_players : set[int], optional
        Player IDs that must be in the lineup.
    max_per_game : int
        Maximum players from any single game (correlation control).
    objective : str
        'projection' to maximize expected DK points,
        'value' to maximize points per $1K salary.

    Returns
    -------
    pd.DataFrame or None
        The optimal 10-player lineup, or None if infeasible.
    """
    if exclude_players is None:
        exclude_players = set()
    if lock_players is None:
        lock_players = set()

    # Filter pool
    df = pool[~pool["player_id"].isin(exclude_players)].copy()
    df = df[df["salary"] > 0].copy()
    df = df[df["dk_proj"] > 0].copy()
    df = df.reset_index(drop=True)

    if len(df) < ROSTER_SIZE:
        logger.warning("Not enough players (%d) to fill roster", len(df))
        return None

    n = len(df)
    prob = LpProblem("DK_Classic_MLB", LpMaximize)

    # Decision variables: x[i] = 1 if player i is in lineup
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximize projected DK points (or value)
    obj_col = "dk_proj" if objective == "projection" else "value"
    prob += lpSum(x[i] * df.iloc[i][obj_col] for i in range(n))

    # Salary cap
    prob += lpSum(x[i] * df.iloc[i]["salary"] for i in range(n)) <= SALARY_CAP

    # Roster size = 10
    prob += lpSum(x[i] for i in range(n)) == ROSTER_SIZE

    # Position constraints
    # Each player can fill their listed position OR UTIL (if non-pitcher)
    # DK multi-position eligibility: players can have "1B/OF", "SS/2B", etc.
    # We need auxiliary variables for position assignment.

    # Simpler approach: for each player, define which position slots they
    # can fill. Then ensure each slot is filled exactly once.

    # Create slot assignment variables: y[i, slot]
    all_slots = []
    for pos, count in ROSTER_SLOTS.items():
        for k in range(count):
            all_slots.append(f"{pos}_{k}")

    y = {}
    for i in range(n):
        positions = _parse_positions(df.iloc[i]["position"])
        for slot in all_slots:
            slot_pos = slot.rsplit("_", 1)[0]
            if slot_pos in positions:
                y[i, slot] = LpVariable(f"y_{i}_{slot}", cat="Binary")

    # Each slot filled exactly once
    for slot in all_slots:
        eligible = [y[i, slot] for i in range(n) if (i, slot) in y]
        if not eligible:
            logger.warning("No eligible players for slot %s", slot)
            return None
        prob += lpSum(eligible) == 1

    # Each player fills at most one slot (and only if selected)
    for i in range(n):
        player_slots = [y[i, s] for s in all_slots if (i, s) in y]
        if player_slots:
            prob += lpSum(player_slots) == x[i]
        else:
            # Player can't fill any slot — force out
            prob += x[i] == 0

    # Max players per game (game stacking control)
    if "game" in df.columns:
        for game_name in df["game"].unique():
            game_mask = df["game"] == game_name
            prob += (
                lpSum(x[i] for i in range(n) if game_mask.iloc[i])
                <= max_per_game
            )

    # Lock/force specific players
    for pid in lock_players:
        idx = df.index[df["player_id"] == pid]
        for i in idx:
            prob += x[i] == 1

    # Solve
    solver = PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    if prob.status != 1:
        logger.warning("Optimizer returned status %d (infeasible)", prob.status)
        return None

    # Extract lineup
    selected = [i for i in range(n) if x[i].value() and x[i].value() > 0.5]
    lineup = df.iloc[selected].copy()

    # Determine slot assignments
    slot_assignments = {}
    for i in selected:
        for slot in all_slots:
            if (i, slot) in y and y[i, slot].value() and y[i, slot].value() > 0.5:
                slot_assignments[i] = slot.rsplit("_", 1)[0]
                break
    lineup["roster_slot"] = [slot_assignments.get(i, "?") for i in selected]

    return lineup.sort_values(
        "roster_slot",
        key=lambda s: s.map(
            {"P": 0, "C": 1, "1B": 2, "2B": 3, "3B": 4, "SS": 5, "OF": 6, "UTIL": 7}
        ),
    )


def _parse_positions(pos_str: str) -> set[str]:
    """Parse DK position string like '1B/3B/OF' into a set."""
    positions = set()
    for p in str(pos_str).split("/"):
        p = p.strip()
        if p in ("SP", "RP"):
            positions.add("P")
        elif p:
            positions.add(p)
    return positions


# ---------------------------------------------------------------------------
# Tiers Optimizer
# ---------------------------------------------------------------------------
def optimize_tiers_lineup(
    pool: pd.DataFrame,
    *,
    exclude_combos: list[frozenset[int]] | None = None,
) -> pd.DataFrame | None:
    """Pick the best player from each tier (6 tiers).

    Parameters
    ----------
    pool : pd.DataFrame
        Tiers player pool with 'tier' and 'dk_proj' columns.
    exclude_combos : list of frozenset[int], optional
        Previously selected combos to avoid exact duplicates.

    Returns
    -------
    pd.DataFrame or None
        6-player lineup (one per tier).
    """
    if exclude_combos is None:
        exclude_combos = []

    tiers = sorted(pool["tier"].dropna().unique())
    lineup_rows = []

    for tier in tiers:
        tier_players = pool[pool["tier"] == tier].sort_values(
            "dk_proj", ascending=False,
        )
        if tier_players.empty:
            logger.warning("No players in tier %d", tier)
            return None
        # Pick the best available (could extend to avoid past combos)
        lineup_rows.append(tier_players.iloc[0])

    return pd.DataFrame(lineup_rows)


# ---------------------------------------------------------------------------
# Multi-lineup generation with diversity
# ---------------------------------------------------------------------------
def generate_diverse_lineups(
    pool: pd.DataFrame,
    n_lineups: int = 5,
    mode: str = "classic",
    max_exposure: float = 0.6,
) -> list[pd.DataFrame]:
    """Generate multiple diverse lineups.

    Uses iterative exclusion: after generating each lineup, reduce the
    objective for selected players to encourage diversity.

    Parameters
    ----------
    pool : pd.DataFrame
        Player pool from build_player_pool().
    n_lineups : int
        Number of lineups to generate.
    mode : str
        'classic' or 'tiers'.
    max_exposure : float
        Maximum fraction of lineups any single player can appear in.

    Returns
    -------
    list of pd.DataFrame
        Each element is a lineup DataFrame.
    """
    lineups: list[pd.DataFrame] = []
    exposure: dict[int, int] = {}
    max_count = max(1, int(n_lineups * max_exposure))

    working_pool = pool.copy()

    for i in range(n_lineups):
        # Exclude overexposed players
        exclude = {
            pid for pid, cnt in exposure.items() if cnt >= max_count
        }

        if mode == "tiers":
            lu = optimize_tiers_lineup(
                working_pool[~working_pool["player_id"].isin(exclude)],
            )
        else:
            lu = optimize_classic_lineup(
                working_pool,
                exclude_players=exclude,
            )

        if lu is None:
            logger.warning("Could not generate lineup %d/%d", i + 1, n_lineups)
            break

        lineups.append(lu)

        # Update exposure counts
        for pid in lu["player_id"]:
            exposure[pid] = exposure.get(pid, 0) + 1

        # Slightly discount already-selected players to encourage diversity
        for pid in lu["player_id"]:
            mask = working_pool["player_id"] == pid
            working_pool.loc[mask, "dk_proj"] *= 0.97

        logger.info(
            "Lineup %d/%d: %.1f pts, $%s salary",
            i + 1,
            n_lineups,
            lu["dk_proj"].sum(),
            f"{lu.get('salary', pd.Series([0])).sum():,}",
        )

    return lineups


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def _format_classic_lineup(lu: pd.DataFrame, rank: int) -> str:
    """Format a Classic lineup for terminal display."""
    lines = [
        f"\n{'='*65}",
        f"  LINEUP #{rank}  |  "
        f"Proj: {lu['dk_proj'].sum():.1f} pts  |  "
        f"Salary: ${lu['salary'].sum():,}  |  "
        f"Remaining: ${SALARY_CAP - lu['salary'].sum():,}",
        f"{'='*65}",
        f"  {'Slot':<5} {'Player':<25} {'Team':<5} {'Sal':>7} {'Proj':>6} {'Val':>5}",
        f"  {'-'*5} {'-'*25} {'-'*5} {'-'*7} {'-'*6} {'-'*5}",
    ]
    for _, row in lu.iterrows():
        val = row.get("value", 0)
        lines.append(
            f"  {row.get('roster_slot', row['position']):<5} "
            f"{row['player_name']:<25} "
            f"{row['team']:<5} "
            f"${row['salary']:>6,} "
            f"{row['dk_proj']:>6.1f} "
            f"{val:>5.1f}"
        )
    return "\n".join(lines)


def _format_tiers_lineup(lu: pd.DataFrame, rank: int) -> str:
    """Format a Tiers lineup for terminal display."""
    lines = [
        f"\n{'='*55}",
        f"  TIERS LINEUP #{rank}  |  Proj: {lu['dk_proj'].sum():.1f} pts",
        f"{'='*55}",
        f"  {'Tier':<5} {'Player':<25} {'Team':<5} {'Proj':>6}",
        f"  {'-'*5} {'-'*25} {'-'*5} {'-'*6}",
    ]
    for _, row in lu.iterrows():
        lines.append(
            f"  T{int(row['tier']):<4} "
            f"{row['player_name']:<25} "
            f"{row['team']:<5} "
            f"{row['dk_proj']:>6.1f}"
        )
    return "\n".join(lines)


def _print_value_plays(pool: pd.DataFrame, n: int = 15) -> None:
    """Print top value plays (pts per $1K salary)."""
    if "value" not in pool.columns or "salary" not in pool.columns:
        return
    top = pool.nlargest(n, "value")
    print(f"\n{'='*65}")
    print(f"  TOP {n} VALUE PLAYS (DK pts per $1K salary)")
    print(f"{'='*65}")
    print(f"  {'Player':<25} {'Pos':<6} {'Team':<5} {'Sal':>7} {'Proj':>6} {'Val':>5}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*7} {'-'*6} {'-'*5}")
    for _, row in top.iterrows():
        print(
            f"  {row['player_name']:<25} "
            f"{row['position']:<6} "
            f"{row['team']:<5} "
            f"${row['salary']:>6,} "
            f"{row['dk_proj']:>6.1f} "
            f"{row['value']:>5.1f}"
        )


def _print_top_projections(pool: pd.DataFrame, n: int = 15) -> None:
    """Print top raw projected points."""
    top = pool.nlargest(n, "dk_proj")
    print(f"\n{'='*65}")
    print(f"  TOP {n} PROJECTED PLAYERS")
    print(f"{'='*65}")
    print(f"  {'Player':<25} {'Pos':<6} {'Team':<5} {'Sal':>7} {'Proj':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*7} {'-'*6}")
    for _, row in top.iterrows():
        sal = row.get("salary", 0)
        sal_str = f"${sal:>6,}" if sal else "  N/A  "
        print(
            f"  {row['player_name']:<25} "
            f"{row['position']:<6} "
            f"{row['team']:<5} "
            f"{sal_str} "
            f"{row['dk_proj']:>6.1f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build optimal DraftKings MLB lineups",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Game date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--mode",
        choices=["classic", "tiers"],
        default="classic",
        help="Contest type. Default: classic.",
    )
    parser.add_argument(
        "--n-lineups",
        type=int,
        default=5,
        help="Number of lineups to generate. Default: 5.",
    )
    parser.add_argument(
        "--max-exposure",
        type=float,
        default=0.6,
        help="Max fraction of lineups a player can appear in. Default: 0.6.",
    )
    parser.add_argument(
        "--max-per-game",
        type=int,
        default=4,
        help="Max players from any single game (classic). Default: 4.",
    )
    args = parser.parse_args()

    print(f"\n  DraftKings MLB Lineup Optimizer")
    print(f"  Date: {args.date}  |  Mode: {args.mode}  |  Lineups: {args.n_lineups}")
    print(f"  {'='*50}")

    # Build player pool
    pool = build_player_pool(args.date, mode=args.mode)
    if pool.empty:
        logger.error("Empty player pool — cannot optimize")
        sys.exit(1)

    # Show top projections and value plays
    _print_top_projections(pool)
    if args.mode == "classic":
        _print_value_plays(pool)

    # Generate lineups
    lineups = generate_diverse_lineups(
        pool,
        n_lineups=args.n_lineups,
        mode=args.mode,
        max_exposure=args.max_exposure,
    )

    if not lineups:
        logger.error("No lineups generated")
        sys.exit(1)

    # Display lineups
    for i, lu in enumerate(lineups, 1):
        if args.mode == "classic":
            print(_format_classic_lineup(lu, i))
        else:
            print(_format_tiers_lineup(lu, i))

    # Summary
    print(f"\n{'='*65}")
    print(f"  SUMMARY: {len(lineups)} lineups generated")
    projections = [lu["dk_proj"].sum() for lu in lineups]
    print(f"  Projection range: {min(projections):.1f} – {max(projections):.1f} pts")
    if args.mode == "classic":
        salaries = [lu["salary"].sum() for lu in lineups]
        print(f"  Salary range: ${min(salaries):,} – ${max(salaries):,}")

    # Exposure report
    all_players: dict[str, int] = {}
    for lu in lineups:
        for name in lu["player_name"]:
            all_players[name] = all_players.get(name, 0) + 1

    print(f"\n  EXPOSURE ({len(lineups)} lineups):")
    for name, count in sorted(
        all_players.items(), key=lambda x: -x[1]
    ):
        pct = count / len(lineups) * 100
        print(f"    {name:<25} {count}/{len(lineups)} ({pct:.0f}%)")

    # --- Save to files ---
    _save_lineups(lineups, args.date, args.mode)
    print()


def _save_lineups(
    lineups: list[pd.DataFrame],
    game_date: str,
    mode: str,
) -> None:
    """Save lineups to parquet (dashboard) and CSV (DK upload / sharing).

    Output files:
        tdd-dashboard/data/dashboard/dk_lineups.parquet
        outputs/dk_lineups_YYYY-MM-DD.csv
    """
    if not lineups:
        return

    # Build combined DataFrame with lineup_num column
    rows = []
    for i, lu in enumerate(lineups, 1):
        for _, row in lu.iterrows():
            rows.append({
                "lineup_num": i,
                "game_date": game_date,
                "mode": mode,
                "roster_slot": row.get("roster_slot", row.get("position", "")),
                "player_name": row["player_name"],
                "player_id": int(row["player_id"]),
                "player_dk_id": row.get("player_dk_id"),
                "position": row.get("position", ""),
                "team": row.get("team", ""),
                "opponent": row.get("opponent", ""),
                "game": row.get("game", ""),
                "salary": row.get("salary"),
                "tier": row.get("tier"),
                "dk_proj": round(row.get("dk_proj", 0), 1),
                "value": round(row.get("value", 0), 1),
                "fppg": row.get("fppg"),
            })

    combined = pd.DataFrame(rows)

    # --- Parquet (dashboard) ---
    parquet_path = DASHBOARD_DIR / "dk_lineups.parquet"
    # Append to history if file exists (keep past dates)
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        # Replace any rows for this date+mode
        existing = existing[
            ~((existing["game_date"] == game_date) & (existing["mode"] == mode))
        ]
        combined = pd.concat([existing, combined], ignore_index=True)
    combined.to_parquet(parquet_path, index=False)
    logger.info("Saved %d lineup rows to %s", len(combined), parquet_path)

    # --- CSV (quick share / DK upload format) ---
    csv_dir = PROJECT_DIR / "outputs"
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / f"dk_lineups_{game_date}.csv"
    combined[combined["game_date"] == game_date].to_csv(csv_path, index=False)
    logger.info("Saved CSV to %s", csv_path)

    print(f"\n  OUTPUT FILES:")
    print(f"    Parquet: {parquet_path}")
    print(f"    CSV:     {csv_path}")


if __name__ == "__main__":
    main()
