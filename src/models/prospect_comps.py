"""
Prospect-to-MLB player comp system.

Matches MiLB prospect translated rates against MLB players' early-career
profiles using weighted Euclidean distance on standardized features.

Produces top-N MLB comps per prospect with similarity scores and
feature-by-feature comparisons for dashboard display.

Feature sets:
  Batters:  K%, BB%, ISO, HR/PA, SB rate, K-BB diff
  Pitchers: K%, BB%, HR/BF, K-BB diff

Position (batters) and role (pitchers) are used as filters, not features.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"
DASHBOARD_DIR = Path(
    r"C:\Users\kekoa\Documents\data_analytics\tdd-dashboard\data\dashboard"
)

N_COMPS = 5

# ---------------------------------------------------------------------------
# Feature definitions and weights
# ---------------------------------------------------------------------------
BATTER_FEATURES = ["k_pct", "bb_pct", "iso", "hr_pa", "sb_rate", "k_bb_diff"]
BATTER_WEIGHTS = {
    "k_pct": 0.25,
    "bb_pct": 0.20,
    "iso": 0.20,
    "hr_pa": 0.10,
    "sb_rate": 0.15,
    "k_bb_diff": 0.10,
}

PITCHER_FEATURES = ["k_pct", "bb_pct", "hr_bf", "k_bb_diff"]
PITCHER_WEIGHTS = {
    "k_pct": 0.30,
    "bb_pct": 0.25,
    "hr_bf": 0.20,
    "k_bb_diff": 0.25,
}

# Age window for MLB comp pool (typical post-prospect development window)
COMP_AGE_RANGE = (21, 27)

# Minimum sample sizes
MIN_PA_COMP = 200
MIN_BF_COMP = 100

# Position groups for soft filtering (match within group)
_POS_GROUPS = {
    "C": "C",
    "1B": "CI",
    "3B": "CI",
    "2B": "MI",
    "SS": "MI",
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "DH": "DH",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _standardize(
    df: pd.DataFrame,
    features: list[str],
    mean: pd.Series,
    std: pd.Series,
) -> np.ndarray:
    """Z-score standardize features using provided stats."""
    X = df[features].values.astype(float)
    m = mean[features].values.astype(float)
    s = std[features].values.astype(float)
    s = np.where(s < 1e-9, 1.0, s)  # avoid div-by-zero
    return (X - m) / s


def _weighted_distance(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted Euclidean distance from a single point to multiple points."""
    diff = a - b  # (n_comps, n_features)
    return np.sqrt((weights * diff**2).sum(axis=1))


def _similarity(distance: np.ndarray) -> np.ndarray:
    """Convert distance to 0-1 similarity score (1 = identical)."""
    return 1.0 / (1.0 + distance)


# ---------------------------------------------------------------------------
# MLB comp pool builders
# ---------------------------------------------------------------------------

def build_mlb_batter_comp_pool(
    age_range: tuple[int, int] = COMP_AGE_RANGE,
    min_pa: int = MIN_PA_COMP,
) -> pd.DataFrame:
    """Build MLB batter comp pool from early-career seasons.

    Aggregates each player's qualifying seasons within the age window
    into a single PA-weighted profile.

    Parameters
    ----------
    age_range : tuple[int, int]
        Min/max age (inclusive) for comp pool seasons.
    min_pa : int
        Minimum total PA across qualifying seasons.

    Returns
    -------
    pd.DataFrame
        One row per MLB player with comp features.
    """
    # Load data sources
    full_stats = pd.read_parquet(CACHE_DIR / "hitter_full_stats.parquet")
    trad_path = DASHBOARD_DIR / "hitter_traditional_all.parquet"
    if not trad_path.exists():
        trad_path = CACHE_DIR / "hitter_traditional_all.parquet"
    trad = pd.read_parquet(trad_path)

    # Merge ISO and SB from traditional stats
    trad_cols = ["batter_id", "season", "iso", "sb"]
    available = [c for c in trad_cols if c in trad.columns]
    merged = full_stats.merge(trad[available], on=["batter_id", "season"], how="left")

    # Filter to age window
    merged = merged[
        (merged["age"] >= age_range[0]) & (merged["age"] <= age_range[1])
    ].copy()

    if merged.empty:
        logger.warning("No batter data in age range %s", age_range)
        return pd.DataFrame()

    # Compute derived features per season
    merged["sb_rate"] = merged["sb"].fillna(0) / merged["pa"].clip(lower=1)
    if "iso" not in merged.columns or merged["iso"].isna().all():
        merged["iso"] = merged["hr_rate"] * 2.5  # rough fallback

    # PA-weighted aggregate per player
    records: list[dict] = []
    for pid, grp in merged.groupby("batter_id"):
        total_pa = grp["pa"].sum()
        if total_pa < min_pa:
            continue

        w = grp["pa"].values.astype(float)
        w_sum = w.sum()

        def _wavg(col: str) -> float:
            vals = grp[col].fillna(0).values.astype(float)
            return float((vals * w).sum() / w_sum)

        k_pct = _wavg("k_rate")
        bb_pct = _wavg("bb_rate")
        records.append({
            "player_id": int(pid),
            "player_name": grp["batter_name"].iloc[-1],
            "position": None,  # filled below
            "total_pa": int(total_pa),
            "age_min": int(grp["age"].min()),
            "age_max": int(grp["age"].max()),
            "n_seasons": int(grp["season"].nunique()),
            "k_pct": k_pct,
            "bb_pct": bb_pct,
            "iso": _wavg("iso"),
            "hr_pa": _wavg("hr_rate"),
            "sb_rate": _wavg("sb_rate"),
            "k_bb_diff": k_pct - bb_pct,
        })

    pool = pd.DataFrame(records)
    if pool.empty:
        return pool

    # Get primary positions — try parquet first, then DB
    pos_map: dict[int, str] = {}

    # Try hitters_rankings parquet (has position column)
    elig_path = DASHBOARD_DIR / "hitter_position_eligibility.parquet"
    rankings_path = DASHBOARD_DIR / "hitters_rankings.parquet"
    if rankings_path.exists():
        try:
            rk = pd.read_parquet(rankings_path, columns=["player_id", "position"])
            pos_map = dict(zip(rk["player_id"], rk["position"]))
        except Exception:
            pass
    if not pos_map and elig_path.exists():
        try:
            elig = pd.read_parquet(elig_path)
            # Take most-started position per player
            primary = elig.sort_values("starts", ascending=False).drop_duplicates("player_id")
            pos_map = dict(zip(primary["player_id"], primary["position"]))
        except Exception:
            pass

    # DB fallback
    if not pos_map:
        try:
            from src.data.db import read_sql
            positions = read_sql(
                "SELECT player_id, primary_position "
                "FROM production.dim_player",
                {},
            )
            pos_map = dict(zip(positions["player_id"], positions["primary_position"]))
        except Exception:
            logger.warning("Could not load positions — comps will lack position info")

    pool["position"] = pool["player_id"].map(pos_map).fillna("DH")

    pool["pos_group"] = pool["position"].map(_POS_GROUPS).fillna("DH")
    pool["age_range_str"] = pool["age_min"].astype(str) + "-" + pool["age_max"].astype(str)

    logger.info(
        "MLB batter comp pool: %d players (age %d-%d, min %d PA)",
        len(pool), age_range[0], age_range[1], min_pa,
    )
    return pool


def build_mlb_pitcher_comp_pool(
    age_range: tuple[int, int] = COMP_AGE_RANGE,
    min_bf: int = MIN_BF_COMP,
) -> pd.DataFrame:
    """Build MLB pitcher comp pool from early-career seasons.

    Parameters
    ----------
    age_range : tuple[int, int]
        Min/max age for comp pool seasons.
    min_bf : int
        Minimum total BF across qualifying seasons.

    Returns
    -------
    pd.DataFrame
        One row per MLB pitcher with comp features.
    """
    full_stats = pd.read_parquet(CACHE_DIR / "pitcher_full_stats.parquet")
    trad_path = DASHBOARD_DIR / "pitcher_traditional_all.parquet"
    if not trad_path.exists():
        trad_path = CACHE_DIR / "pitcher_traditional_all.parquet"
    trad = pd.read_parquet(trad_path)

    # Merge starts/games for role determination
    trad_cols = ["pitcher_id", "season", "starts", "games", "bf"]
    available = [c for c in trad_cols if c in trad.columns]
    merged = full_stats.merge(
        trad[available].rename(columns={"games": "trad_games", "bf": "trad_bf"}),
        on=["pitcher_id", "season"],
        how="left",
    )

    # Filter to age window
    merged = merged[
        (merged["age"] >= age_range[0]) & (merged["age"] <= age_range[1])
    ].copy()

    if merged.empty:
        logger.warning("No pitcher data in age range %s", age_range)
        return pd.DataFrame()

    # BF-weighted aggregate per pitcher
    records: list[dict] = []
    for pid, grp in merged.groupby("pitcher_id"):
        total_bf = grp["batters_faced"].sum()
        if total_bf < min_bf:
            continue

        w = grp["batters_faced"].values.astype(float)
        w_sum = w.sum()

        def _wavg(col: str) -> float:
            vals = grp[col].fillna(0).values.astype(float)
            return float((vals * w).sum() / w_sum)

        # Role from starts/games
        total_starts = grp["starts"].sum() if "starts" in grp.columns else 0
        total_games = grp["trad_games"].sum() if "trad_games" in grp.columns else grp["games"].sum()
        role = "SP" if total_starts / max(total_games, 1) > 0.5 else "RP"

        k_pct = _wavg("k_rate")
        bb_pct = _wavg("bb_rate")
        records.append({
            "player_id": int(pid),
            "player_name": grp["pitcher_name"].iloc[-1],
            "role": role,
            "total_bf": int(total_bf),
            "age_min": int(grp["age"].min()),
            "age_max": int(grp["age"].max()),
            "n_seasons": int(grp["season"].nunique()),
            "k_pct": k_pct,
            "bb_pct": bb_pct,
            "hr_bf": _wavg("hr_per_bf"),
            "k_bb_diff": k_pct - bb_pct,
        })

    pool = pd.DataFrame(records)
    if pool.empty:
        return pool

    pool["age_range_str"] = pool["age_min"].astype(str) + "-" + pool["age_max"].astype(str)

    logger.info(
        "MLB pitcher comp pool: %d pitchers (age %d-%d, min %d BF)",
        len(pool), age_range[0], age_range[1], min_bf,
    )
    return pool


# ---------------------------------------------------------------------------
# Core comp matching
# ---------------------------------------------------------------------------

def compute_prospect_comps(
    prospect_df: pd.DataFrame,
    comp_pool: pd.DataFrame,
    features: list[str],
    weights: dict[str, float],
    prospect_feature_map: dict[str, str],
    n_comps: int = N_COMPS,
    group_col: str | None = None,
    prospect_group_col: str | None = None,
) -> pd.DataFrame:
    """Find top-N MLB comps for each prospect.

    Parameters
    ----------
    prospect_df : pd.DataFrame
        Prospect data with translated rate features.
    comp_pool : pd.DataFrame
        MLB comp pool with matching features.
    features : list[str]
        Feature column names in the comp pool.
    weights : dict[str, float]
        Feature name → weight for distance calculation.
    prospect_feature_map : dict[str, str]
        Maps comp pool feature names → prospect column names.
        E.g. {"k_pct": "wtd_k_pct", "bb_pct": "wtd_bb_pct"}.
    n_comps : int
        Number of comps to return per prospect.
    group_col : str or None
        Column in comp_pool to filter by (e.g. "pos_group" or "role").
    prospect_group_col : str or None
        Matching column in prospect_df.

    Returns
    -------
    pd.DataFrame
        Long-format: one row per (prospect, comp_rank).
    """
    if comp_pool.empty or prospect_df.empty:
        return pd.DataFrame()

    # Compute standardization stats from comp pool
    pool_mean = comp_pool[features].mean()
    pool_std = comp_pool[features].std()

    # Standardize comp pool
    w_arr = np.array([weights[f] for f in features])
    comp_z = _standardize(comp_pool, features, pool_mean, pool_std)

    # Map prospect features to comp pool feature space
    prospect_aligned = pd.DataFrame()
    for comp_feat, prosp_feat in prospect_feature_map.items():
        if prosp_feat in prospect_df.columns:
            prospect_aligned[comp_feat] = prospect_df[prosp_feat].fillna(0).values
        else:
            prospect_aligned[comp_feat] = 0.0

    prospect_z = _standardize(prospect_aligned, features, pool_mean, pool_std)

    results: list[dict] = []
    for i in range(len(prospect_df)):
        row = prospect_df.iloc[i]

        # Filter comp pool by group if specified
        if group_col and prospect_group_col:
            prospect_group = row.get(prospect_group_col)
            mask = comp_pool[group_col] == prospect_group
            # Relax if too few candidates
            if mask.sum() < n_comps:
                mask = pd.Series(True, index=comp_pool.index)
            comp_subset_z = comp_z[mask.values]
            comp_subset_idx = comp_pool.index[mask.values]
        else:
            comp_subset_z = comp_z
            comp_subset_idx = comp_pool.index

        # Exclude the prospect from their own comps (if they debuted)
        pid = row.get("player_id")
        if pid is not None:
            not_self = comp_pool.loc[comp_subset_idx, "player_id"] != pid
            comp_subset_z = comp_subset_z[not_self.values]
            comp_subset_idx = comp_subset_idx[not_self.values]

        if len(comp_subset_z) == 0:
            continue

        # Compute distances
        distances = _weighted_distance(
            prospect_z[i : i + 1],  # (1, n_features)
            comp_subset_z,          # (n_comps_pool, n_features)
            w_arr,
        )
        similarities = _similarity(distances)

        # Top-N
        top_n = min(n_comps, len(similarities))
        top_idx = np.argsort(distances)[:top_n]

        for rank, tidx in enumerate(top_idx, 1):
            comp_row = comp_pool.loc[comp_subset_idx[tidx]]
            rec = {
                "player_id": int(pid),
                "prospect_name": row.get("name", ""),
                "comp_rank": rank,
                "comp_player_id": int(comp_row["player_id"]),
                "comp_name": comp_row["player_name"],
                "similarity_score": round(float(similarities[tidx]), 4),
                "comp_total_pa": int(comp_row.get("total_pa", comp_row.get("total_bf", 0))),
                "comp_age_range": comp_row.get("age_range_str", ""),
            }

            # Feature comparison columns
            for comp_feat, prosp_feat in prospect_feature_map.items():
                rec[f"prospect_{comp_feat}"] = (
                    round(float(row.get(prosp_feat, 0)), 4)
                )
                rec[f"comp_{comp_feat}"] = round(float(comp_row.get(comp_feat, 0)), 4)

            results.append(rec)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_batter_comps(
    prospect_rankings: pd.DataFrame | None = None,
    projection_season: int = 2026,
    n_comps: int = N_COMPS,
) -> pd.DataFrame:
    """Find MLB comps for all ranked batting prospects.

    Parameters
    ----------
    prospect_rankings : pd.DataFrame or None
        If None, loads from dashboard parquet.
    projection_season : int
        Target season (used for loading rankings).
    n_comps : int
        Comps per prospect.

    Returns
    -------
    pd.DataFrame
        Long-format comp results.
    """
    if prospect_rankings is None:
        path = DASHBOARD_DIR / "prospect_rankings.parquet"
        if not path.exists():
            logger.warning("No prospect_rankings.parquet found")
            return pd.DataFrame()
        prospect_rankings = pd.read_parquet(path)

    if prospect_rankings.empty:
        return pd.DataFrame()

    # Add position group for soft filtering
    prospect_rankings = prospect_rankings.copy()
    prospect_rankings["pos_group"] = (
        prospect_rankings["primary_position"].map(_POS_GROUPS).fillna("DH")
    )

    comp_pool = build_mlb_batter_comp_pool()
    if comp_pool.empty:
        logger.warning("Empty MLB batter comp pool")
        return pd.DataFrame()

    feature_map = {
        "k_pct": "wtd_k_pct",
        "bb_pct": "wtd_bb_pct",
        "iso": "wtd_iso",
        "hr_pa": "wtd_hr_pa",
        "sb_rate": "sb_rate",
        "k_bb_diff": "k_bb_diff",
    }

    comps = compute_prospect_comps(
        prospect_df=prospect_rankings,
        comp_pool=comp_pool,
        features=BATTER_FEATURES,
        weights=BATTER_WEIGHTS,
        prospect_feature_map=feature_map,
        n_comps=n_comps,
        group_col="pos_group",
        prospect_group_col="pos_group",
    )

    # Add position info
    if not comps.empty:
        pos_map = dict(zip(comp_pool["player_id"], comp_pool["position"]))
        comps["comp_position"] = comps["comp_player_id"].map(pos_map)

    logger.info(
        "Batter comps: %d prospects × %d comps = %d rows",
        comps["player_id"].nunique() if not comps.empty else 0,
        n_comps,
        len(comps),
    )
    return comps


def find_pitcher_comps(
    prospect_rankings: pd.DataFrame | None = None,
    projection_season: int = 2026,
    n_comps: int = N_COMPS,
) -> pd.DataFrame:
    """Find MLB comps for all ranked pitching prospects.

    Parameters
    ----------
    prospect_rankings : pd.DataFrame or None
        If None, loads from dashboard parquet.
    projection_season : int
        Target season.
    n_comps : int
        Comps per prospect.

    Returns
    -------
    pd.DataFrame
        Long-format comp results.
    """
    if prospect_rankings is None:
        path = DASHBOARD_DIR / "pitching_prospect_rankings.parquet"
        if not path.exists():
            logger.warning("No pitching_prospect_rankings.parquet found")
            return pd.DataFrame()
        prospect_rankings = pd.read_parquet(path)

    if prospect_rankings.empty:
        return pd.DataFrame()

    comp_pool = build_mlb_pitcher_comp_pool()
    if comp_pool.empty:
        logger.warning("Empty MLB pitcher comp pool")
        return pd.DataFrame()

    feature_map = {
        "k_pct": "wtd_k_pct",
        "bb_pct": "wtd_bb_pct",
        "hr_bf": "wtd_hr_bf",
        "k_bb_diff": "k_bb_diff",
    }

    comps = compute_prospect_comps(
        prospect_df=prospect_rankings,
        comp_pool=comp_pool,
        features=PITCHER_FEATURES,
        weights=PITCHER_WEIGHTS,
        prospect_feature_map=feature_map,
        n_comps=n_comps,
        group_col="role",
        prospect_group_col="pitcher_role",
    )

    # Add role info
    if not comps.empty:
        role_map = dict(zip(comp_pool["player_id"], comp_pool["role"]))
        comps["comp_role"] = comps["comp_player_id"].map(role_map)

    logger.info(
        "Pitcher comps: %d prospects × %d comps = %d rows",
        comps["player_id"].nunique() if not comps.empty else 0,
        n_comps,
        len(comps),
    )
    return comps


def find_all_comps(
    projection_season: int = 2026,
    n_comps: int = N_COMPS,
) -> dict[str, pd.DataFrame]:
    """Run both batter and pitcher prospect comp matching.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"batters"``, ``"pitchers"``.
    """
    batters = find_batter_comps(
        projection_season=projection_season, n_comps=n_comps,
    )
    pitchers = find_pitcher_comps(
        projection_season=projection_season, n_comps=n_comps,
    )
    logger.info(
        "All prospect comps: %d batter rows, %d pitcher rows",
        len(batters), len(pitchers),
    )
    return {"batters": batters, "pitchers": pitchers}
