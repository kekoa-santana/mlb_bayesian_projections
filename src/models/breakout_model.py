"""Hitter breakout archetype model.

Two empirically-observed breakout types:

1. **The Hitter** (bat-first breakout):
   High barrel rate, high zone contact rate, pull tendency.
   Defense is irrelevant — value pathway is purely offensive.

2. **The All-Around** (complete player breakout):
   Exit velocity (bat speed proxy), plate discipline, plus runner,
   plus defender.  Tools across the board, bat hasn't fully arrived.

Each player is scored against both archetypes (0-1).  Breakout potential
combines archetype fit with room-to-grow indicators: age, skills-production
gap, and Bayesian trajectory projections.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(
    "C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard"
)

# ---------------------------------------------------------------------------
# Archetype feature weights
# ---------------------------------------------------------------------------

# "The Hitter": pure offensive ceiling.  Defense excluded by design.
_HITTER_WEIGHTS: dict[str, float] = {
    "barrel_pct":    0.35,   # elite contact quality (strongest Hitter signal)
    "z_contact_pct": 0.25,   # zone-contact consistency
    "pull_pct":      0.10,   # pulling for power (weak standalone signal)
    "hard_hit_pct":  0.15,   # hard-contact rate
    "avg_exit_velo": 0.15,   # raw power indicator
}

# "The All-Around": tools across the board.
_ALL_AROUND_WEIGHTS: dict[str, float] = {
    "power":      0.25,   # exit velo + hard-hit (bat speed proxy)
    "discipline": 0.15,   # chase rate (inv) — BB% has weak signal
    "speed":      0.25,   # sprint speed (strongest All-Around signal)
    "defense":    0.20,   # OAA
    "contact":    0.15,   # zone-contact quality
}

# Room-to-grow sub-weights (age is applied multiplicatively, not additively)
_GAP_WEIGHT = 0.55       # skills >> production gap (main signal)
_TRAJECTORY_WEIGHT = 0.45  # Bayesian projection direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1, higher = better)."""
    return series.rank(pct=True, method="average")


def _inv_pctl(series: pd.Series) -> pd.Series:
    """Inverse percentile rank (lower raw value = higher score)."""
    return 1.0 - series.rank(pct=True, method="average")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_breakout_features(
    season: int,
    min_pa: int = 200,
) -> pd.DataFrame:
    """Assemble all features needed for breakout archetype scoring.

    Reads the hitter projections parquet (which already has observed
    enrichment: z_contact, exit velo, hard_hit, chase_rate, sprint_speed)
    and augments with barrel rate, wOBA, spray data, OAA, and position.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_pa : int
        Minimum PA to qualify.

    Returns
    -------
    pd.DataFrame
        One row per qualified hitter with all breakout features.
    """
    from src.data.queries import get_batted_ball_spray

    # 1. Projections parquet — already enriched with observed stats
    proj = pd.read_parquet(DASHBOARD_DIR / "hitter_projections.parquet")
    base = proj[proj["pa"] >= min_pa].copy()

    if base.empty:
        return pd.DataFrame()

    # 2. Barrel rate, wOBA, xwOBA, K%, BB% from fact_batting_advanced
    obs_adv = read_sql(
        f"""
        SELECT batter_id, barrel_pct, woba, xwoba, k_pct, bb_pct, wrc_plus
        FROM production.fact_batting_advanced
        WHERE season = {season} AND pa >= {min_pa}
        """,
        {},
    )
    base = base.merge(obs_adv, on="batter_id", how="inner")

    # 3. Batted-ball spray (pull_pct)
    spray = get_batted_ball_spray(season)
    if not spray.empty:
        base = base.merge(
            spray[["batter_id", "pull_pct"]],
            on="batter_id",
            how="left",
        )
    if "pull_pct" not in base.columns:
        base["pull_pct"] = np.nan

    # 4. OAA (sum across all positions played)
    oaa = read_sql(
        f"""
        SELECT player_id AS batter_id,
               SUM(outs_above_average) AS oaa
        FROM production.fact_fielding_oaa
        WHERE season = {season}
        GROUP BY player_id
        """,
        {},
    )
    base = base.merge(oaa, on="batter_id", how="left")
    base["oaa"] = base["oaa"].fillna(0).astype(float)

    # 5. Position assignment
    from src.models.player_rankings import _assign_hitter_positions

    positions = _assign_hitter_positions(season=season)
    base = base.merge(
        positions.rename(columns={"player_id": "batter_id"}),
        on="batter_id",
        how="left",
    )

    # 6. Fill NaN with medians so percentile ranking is clean
    fill_cols = [
        "barrel_pct", "z_contact_pct", "pull_pct", "hard_hit_pct",
        "avg_exit_velo", "chase_rate", "sprint_speed", "bb_pct",
        "woba", "xwoba", "k_pct",
    ]
    for col in fill_cols:
        if col in base.columns:
            base[col] = base[col].fillna(base[col].median())

    logger.info("Loaded breakout features for %d hitters", len(base))
    return base


# ---------------------------------------------------------------------------
# Archetype scoring
# ---------------------------------------------------------------------------

def _score_hitter_archetype(df: pd.DataFrame) -> pd.Series:
    """Score each player's fit to 'The Hitter' archetype (0-1).

    High barrel rate + zone contact + pull tendency = offensive ceiling.
    Defense is excluded — this archetype is purely offensive.
    """
    return (
        _HITTER_WEIGHTS["barrel_pct"]    * _pctl(df["barrel_pct"])
        + _HITTER_WEIGHTS["z_contact_pct"] * _pctl(df["z_contact_pct"])
        + _HITTER_WEIGHTS["pull_pct"]      * _pctl(df["pull_pct"])
        + _HITTER_WEIGHTS["hard_hit_pct"]  * _pctl(df["hard_hit_pct"])
        + _HITTER_WEIGHTS["avg_exit_velo"] * _pctl(df["avg_exit_velo"])
    )


def _score_all_around_archetype(df: pd.DataFrame) -> pd.Series:
    """Score each player's fit to 'The All-Around' archetype (0-1).

    Exit velo + discipline + speed + defense = complete player tools.
    """
    # Power sub-score (bat speed proxy)
    power = 0.60 * _pctl(df["avg_exit_velo"]) + 0.40 * _pctl(df["hard_hit_pct"])

    # Discipline sub-score (chase rate only — BB% has near-zero predictive signal)
    discipline = _inv_pctl(df["chase_rate"])

    # Speed, defense, contact
    speed = _pctl(df["sprint_speed"])
    defense = _pctl(df["oaa"])
    contact = _pctl(df["z_contact_pct"])

    return (
        _ALL_AROUND_WEIGHTS["power"]      * power
        + _ALL_AROUND_WEIGHTS["discipline"] * discipline
        + _ALL_AROUND_WEIGHTS["speed"]      * speed
        + _ALL_AROUND_WEIGHTS["defense"]    * defense
        + _ALL_AROUND_WEIGHTS["contact"]    * contact
    )


# ---------------------------------------------------------------------------
# Room to grow
# ---------------------------------------------------------------------------

def _compute_room_to_grow(
    df: pd.DataFrame,
    archetype_fit: pd.Series,
) -> pd.Series:
    """Compute room-to-grow score (0-1).

    Age is applied **multiplicatively** — it gates the upside rather than
    competing additively with other signals.  A 33-year-old with elite
    tools-production gap still gets a low score because breakouts at that
    age are rare.

    Components:
    - **Skills-production gap** (55%): archetype_fit pctl vs wOBA pctl.
      Large positive gap = skills outpace production.
    - **Trajectory** (45%): improving K%/BB% from Bayesian projections.
    - **Age multiplier**: linear decay from 23 (1.0) to 35 (0.10).
      Caps the raw potential — young players get full credit, veterans
      are heavily discounted.
    """
    # Age multiplier: linear decay, floors at 0.10
    age = df["age"].fillna(28)
    age_mult = np.clip(1.0 - (age - 23) / 12.0, 0.10, 1.0)

    # Skills-production gap
    woba_pctl = _pctl(df["woba"])
    gap = (archetype_fit - woba_pctl).clip(0, 1)
    gap_score = (gap / 0.35).clip(0, 1)  # 35pp+ gap = max

    # Bayesian trajectory direction
    delta_k = df.get("delta_k_rate", pd.Series(0.0, index=df.index))
    delta_bb = df.get("delta_bb_rate", pd.Series(0.0, index=df.index))
    k_improving = _inv_pctl(delta_k.fillna(0))   # negative delta = improving
    bb_improving = _pctl(delta_bb.fillna(0))      # positive delta = improving
    trajectory = 0.50 * k_improving + 0.50 * bb_improving

    # Raw potential, then gated by age
    raw = _GAP_WEIGHT * gap_score + _TRAJECTORY_WEIGHT * trajectory
    return raw * age_mult


# ---------------------------------------------------------------------------
# Hole identification
# ---------------------------------------------------------------------------

def _identify_hole(row: pd.Series, archetype: str) -> str:
    """Identify the primary fixable weakness for a breakout candidate.

    Returns a brief label for the "hole in game" that, if improved,
    would most likely unlock a breakout.
    """
    holes: list[tuple[float, str]] = []

    if archetype == "Hitter":
        # What suppresses production despite barrel/contact quality?
        if row.get("k_pct_pctl", 0.5) > 0.55:
            holes.append((row["k_pct_pctl"], "Strikeout rate"))
        if row.get("pull_pctl", 0.5) < 0.45:
            holes.append((1.0 - row["pull_pctl"], "Pull approach"))
        xwoba_gap = row.get("xwoba", 0.0) - row.get("woba", 0.0)
        if xwoba_gap > 0.010:
            holes.append((min(xwoba_gap * 30, 1.0), "Batted ball luck"))
    else:  # All-Around
        # What offensive dimension is missing?
        if row.get("barrel_pctl", 0.5) < 0.45:
            holes.append((1.0 - row["barrel_pctl"], "Power development"))
        if row.get("k_pct_pctl", 0.5) > 0.55:
            holes.append((row["k_pct_pctl"], "Contact quality"))
        if row.get("ev_pctl", 0.5) < 0.45:
            holes.append((1.0 - row["ev_pctl"], "Bat speed"))

    if not holes:
        return "General development"

    holes.sort(key=lambda x: x[0], reverse=True)
    return holes[0][1]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_breakout_candidates(
    season: int = 2025,
    min_pa: int = 200,
) -> pd.DataFrame:
    """Score all qualified hitters for breakout archetype potential.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_pa : int
        Minimum PA to qualify.

    Returns
    -------
    pd.DataFrame
        One row per qualified hitter with breakout scores, archetype
        classification, room-to-grow analysis, and hole identification.
    """
    logger.info("Loading breakout features for season %d...", season)
    df = _load_breakout_features(season, min_pa=min_pa)

    if df.empty:
        logger.warning("No qualified hitters for breakout scoring")
        return pd.DataFrame()

    logger.info("Scoring %d hitters against breakout archetypes...", len(df))

    # --- Score both archetypes ---
    df["hitter_fit"] = _score_hitter_archetype(df)
    df["all_around_fit"] = _score_all_around_archetype(df)

    # Primary archetype = whichever scored higher
    df["breakout_type"] = np.where(
        df["hitter_fit"] >= df["all_around_fit"],
        "Hitter",
        "All-Around",
    )
    df["primary_fit"] = df[["hitter_fit", "all_around_fit"]].max(axis=1)

    # --- Room to grow ---
    df["room_to_grow"] = _compute_room_to_grow(df, df["primary_fit"])

    # --- Breakout score = fit × room_to_grow ---
    df["breakout_score"] = df["primary_fit"] * df["room_to_grow"]

    # --- Percentiles for hole identification ---
    df["k_pct_pctl"] = _pctl(df["k_pct"])      # higher = more Ks (bad)
    df["pull_pctl"] = _pctl(df["pull_pct"])
    df["barrel_pctl"] = _pctl(df["barrel_pct"])
    df["ev_pctl"] = _pctl(df["avg_exit_velo"])

    # --- Identify breakout hole ---
    df["breakout_hole"] = df.apply(
        lambda row: _identify_hole(row, row["breakout_type"]),
        axis=1,
    )

    # --- Tier assignment ---
    score_pctl = _pctl(df["breakout_score"])
    df["breakout_tier"] = np.where(
        score_pctl >= 0.90, "High",
        np.where(score_pctl >= 0.75, "Medium", ""),
    )

    # --- Rank ---
    df["breakout_rank"] = (
        df["breakout_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    df = df.sort_values("breakout_rank")

    # --- Select output columns ---
    output_cols = [
        "batter_id", "batter_name", "age", "position", "batter_stand",
        # Archetype scores
        "hitter_fit", "all_around_fit", "breakout_type", "primary_fit",
        "room_to_grow", "breakout_score", "breakout_rank", "breakout_tier",
        "breakout_hole",
        # Key stats for context
        "pa", "woba", "xwoba", "wrc_plus",
        "barrel_pct", "z_contact_pct", "pull_pct",
        "hard_hit_pct", "avg_exit_velo", "sprint_speed", "oaa",
        "k_pct", "bb_pct", "chase_rate",
        # Bayesian projections
        "projected_k_rate", "projected_bb_rate",
        "delta_k_rate", "delta_bb_rate",
    ]
    available = [c for c in output_cols if c in df.columns]
    result = df[available].reset_index(drop=True)

    n_high = (result["breakout_tier"] == "High").sum()
    n_med = (result["breakout_tier"] == "Medium").sum()
    logger.info(
        "Breakout candidates: %d High, %d Medium (of %d total)",
        n_high, n_med, len(result),
    )

    return result
