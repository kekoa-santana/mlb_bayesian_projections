"""
Archetype matchup analysis — pitcher type vs hitter type outcomes.

Builds a matrix of historical outcomes (K%, BB%, HR%, wOBA) for every
(pitcher_archetype, hitter_archetype) pair. Powers narrative analysis
and fallback scoring when individual matchup data is thin.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import CACHE_DIR, dashboard_dir

logger = logging.getLogger(__name__)

DASHBOARD_DIR = dashboard_dir()


def build_archetype_matchup_matrix(
    seasons: list[int] | None = None,
    min_pa: int = 100,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build outcomes matrix by (pitcher_archetype, hitter_archetype) pair.

    Parameters
    ----------
    seasons : list[int] or None
        Seasons to include. Default: 2020-2025.
    min_pa : int
        Minimum PA per archetype pair to include.
    force_rebuild : bool
        Ignore cache.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_archetype_name, hitter_archetype_name,
        pa, k_pct, bb_pct, hr_pct, woba, n_pitchers, n_hitters.
    """
    cache_path = CACHE_DIR / "archetype_matchup_matrix.parquet"
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached archetype matchup matrix")
        return pd.read_parquet(cache_path)

    if seasons is None:
        seasons = list(range(2020, 2026))

    from src.data.player_clustering import get_hitter_archetypes, get_pitcher_archetypes
    from src.data.db import read_sql

    all_records: list[pd.DataFrame] = []
    for season in seasons:
        logger.info("Building archetype matchups for %d", season)

        # Get archetype assignments
        try:
            hitters = get_hitter_archetypes(season)
            pitchers = get_pitcher_archetypes(season)
        except Exception as e:
            logger.warning("Could not load archetypes for %d: %s", season, e)
            continue

        if hitters.empty or pitchers.empty:
            continue

        # Get PA outcomes for this season
        pa_data = read_sql("""
            SELECT fpa.pitcher_id, fpa.batter_id,
                   COUNT(*) AS pa,
                   SUM(CASE WHEN fpa.events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS k,
                   SUM(CASE WHEN fpa.events = 'walk' THEN 1 ELSE 0 END) AS bb,
                   SUM(CASE WHEN fpa.events = 'home_run' THEN 1 ELSE 0 END) AS hr
            FROM production.fact_pa fpa
            JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
            WHERE dg.season = :season AND dg.game_type = 'R'
            GROUP BY fpa.pitcher_id, fpa.batter_id
        """, {"season": season})

        if pa_data.empty:
            continue

        # Join archetypes
        h_map = hitters.set_index("batter_id")["archetype_name"].to_dict()
        p_map = pitchers.set_index("pitcher_id")["archetype_name"].to_dict()

        pa_data["hitter_archetype_name"] = pa_data["batter_id"].map(h_map)
        pa_data["pitcher_archetype_name"] = pa_data["pitcher_id"].map(p_map)

        # Drop rows without archetype assignment
        pa_data = pa_data.dropna(subset=["hitter_archetype_name", "pitcher_archetype_name"])

        if not pa_data.empty:
            all_records.append(pa_data)

    if not all_records:
        logger.warning("No archetype matchup data available")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)

    # Aggregate by archetype pair
    matrix = (
        combined.groupby(["pitcher_archetype_name", "hitter_archetype_name"])
        .agg(
            pa=("pa", "sum"),
            k=("k", "sum"),
            bb=("bb", "sum"),
            hr=("hr", "sum"),
            n_pitchers=("pitcher_id", "nunique"),
            n_hitters=("batter_id", "nunique"),
        )
        .reset_index()
    )

    # Filter by minimum PA
    matrix = matrix[matrix["pa"] >= min_pa].copy()

    # Compute rates
    matrix["k_pct"] = matrix["k"] / matrix["pa"]
    matrix["bb_pct"] = matrix["bb"] / matrix["pa"]
    matrix["hr_pct"] = matrix["hr"] / matrix["pa"]

    # Drop raw counts, keep rates
    matrix = matrix.drop(columns=["k", "bb", "hr"])

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(cache_path, index=False)
    logger.info(
        "Cached archetype matchup matrix: %d pairs, %d total PA",
        len(matrix), matrix["pa"].sum(),
    )
    return matrix


def get_lineup_archetype_profile(
    lineup_batter_ids: list[int],
    season: int,
) -> dict[str, int]:
    """Return archetype distribution for a lineup.

    Parameters
    ----------
    lineup_batter_ids : list[int]
        Batter IDs in the lineup.
    season : int
        Season for archetype assignments.

    Returns
    -------
    dict[str, int]
        Archetype name -> count in lineup.
    """
    from src.data.player_clustering import get_hitter_archetypes

    hitters = get_hitter_archetypes(season)
    if hitters.empty:
        return {}

    lineup = hitters[hitters["batter_id"].isin(lineup_batter_ids)]
    return dict(lineup["archetype_name"].value_counts())


# NOTE: Not currently wired into any pipeline or dashboard view.
# Kept for future use — enables live "pitcher arsenal vs lineup weaknesses"
# matchup narratives. Wire into game-day precompute when ready.
def score_pitcher_vs_lineup_archetypes(
    pitcher_id: int,
    lineup_batter_ids: list[int],
    season: int,
    matrix: pd.DataFrame | None = None,
) -> dict:
    """Score expected outcomes for a pitcher against a lineup's archetypes.

    Parameters
    ----------
    pitcher_id : int
        Pitcher MLB ID.
    lineup_batter_ids : list[int]
        Batter IDs in the opposing lineup.
    season : int
        Season for archetype assignments.
    matrix : pd.DataFrame or None
        Pre-loaded matchup matrix. If None, loads from cache.

    Returns
    -------
    dict
        Keys: pitcher_archetype, lineup_profile, expected_k_pct,
        expected_bb_pct, expected_hr_pct, matchup_details, narrative.
    """
    from src.data.player_clustering import get_hitter_archetypes, get_pitcher_archetypes

    if matrix is None:
        matrix = build_archetype_matchup_matrix()

    if matrix.empty:
        return {"pitcher_archetype": None, "error": "No matchup matrix available"}

    # Get pitcher archetype
    pitchers = get_pitcher_archetypes(season)
    p_row = pitchers[pitchers["pitcher_id"] == pitcher_id]
    if p_row.empty:
        return {"pitcher_archetype": None, "error": "Pitcher not in archetype data"}
    pitcher_arch = p_row.iloc[0]["archetype_name"]

    # Get lineup archetypes
    lineup_profile = get_lineup_archetype_profile(lineup_batter_ids, season)
    if not lineup_profile:
        return {"pitcher_archetype": pitcher_arch, "error": "No lineup archetypes"}

    # Look up expected outcomes per archetype pair
    p_matrix = matrix[matrix["pitcher_archetype_name"] == pitcher_arch]
    details: list[dict] = []
    total_weight = 0
    wtd_k = wtd_bb = wtd_hr = 0.0

    for hitter_arch, count in lineup_profile.items():
        row = p_matrix[p_matrix["hitter_archetype_name"] == hitter_arch]
        if row.empty:
            continue
        r = row.iloc[0]
        details.append({
            "hitter_archetype": hitter_arch,
            "count": count,
            "k_pct": float(r["k_pct"]),
            "bb_pct": float(r["bb_pct"]),
            "hr_pct": float(r["hr_pct"]),
        })
        wtd_k += r["k_pct"] * count
        wtd_bb += r["bb_pct"] * count
        wtd_hr += r["hr_pct"] * count
        total_weight += count

    if total_weight == 0:
        return {"pitcher_archetype": pitcher_arch, "error": "No matching pairs in matrix"}

    exp_k = wtd_k / total_weight
    exp_bb = wtd_bb / total_weight
    exp_hr = wtd_hr / total_weight

    # Build narrative
    best = max(details, key=lambda d: d["k_pct"]) if details else None
    worst = min(details, key=lambda d: d["k_pct"]) if details else None
    narrative_parts = [f"{pitcher_arch} vs this lineup:"]
    if best:
        narrative_parts.append(
            f"Dominates {best['hitter_archetype']}s ({best['k_pct']:.1%} K)"
        )
    if worst and worst != best:
        narrative_parts.append(
            f"Vulnerable to {worst['hitter_archetype']}s ({worst['k_pct']:.1%} K)"
        )

    return {
        "pitcher_archetype": pitcher_arch,
        "lineup_profile": lineup_profile,
        "expected_k_pct": float(exp_k),
        "expected_bb_pct": float(exp_bb),
        "expected_hr_pct": float(exp_hr),
        "matchup_details": details,
        "narrative": " | ".join(narrative_parts),
    }


def export_archetype_matchups_for_dashboard(
    output_dir: Path | str | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Write archetype matchup matrix for dashboard consumption.

    Parameters
    ----------
    output_dir : Path or str or None
        Output directory. Defaults to tdd-dashboard data dir.
    force_rebuild : bool
        Force rebuild of matrix.

    Returns
    -------
    Path
        Path to written parquet.
    """
    if output_dir is None:
        output_dir = DASHBOARD_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_archetype_matchup_matrix(force_rebuild=force_rebuild)

    path = output_dir / "archetype_matchup_matrix.parquet"
    if not matrix.empty:
        matrix.to_parquet(path, index=False)
        logger.info("Exported archetype matchup matrix: %s (%d pairs)", path, len(matrix))
    else:
        logger.warning("Empty archetype matchup matrix, skipping export")

    return path
