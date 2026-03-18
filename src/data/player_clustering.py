"""
Player-level archetype clustering for hitters and pitchers.

Clusters whole players (not individual pitches) into archetypes based on
their statistical profiles. Used for:
- Player profile pages (archetype label + description)
- Team composition analysis (archetype distribution vs league)

Design follows the same pattern as ``pitch_archetypes.py``:
- Pooled multi-season fit for stable archetypes across years
- StandardScaler + KMeans with weighted samples (by PA/BF)
- Joblib model cache + parquet assignment cache
- Per-season assignment using the fitted model

Hitter features (7):  K%, BB%, ISO, hard_hit%, sprint_speed,
                       chase_rate, whiff_rate
Pitcher features (8): K%, BB%, GB%, whiff_rate, usage_FF,
                       usage_breaking, usage_offspeed, avg_velo
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.feature_eng import (
    build_multi_season_hitter_extended,
    get_cached_hitter_observed_profile,
    get_cached_pitcher_observed_profile,
    get_cached_sprint_speed,
    build_multi_season_pitcher_extended,
    get_pitcher_arsenal,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _get_clustering_seasons() -> list[int]:
    """Read ``seasons.clustering`` from ``config/model.yaml``."""
    path = CONFIG_DIR / "model.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["seasons"]["clustering"]


# ---------------------------------------------------------------------------
# Hitter archetype labels — assigned after fitting based on cluster centers
# ---------------------------------------------------------------------------
_HITTER_ARCHETYPE_RULES: list[tuple[str, str, callable]] = [
    # (name, description, function(center_dict) -> score)
    # Higher score = better match. Assigned greedily.
]


def _name_hitter_clusters(centers: pd.DataFrame, features: list[str]) -> list[dict]:
    """Assign human-readable names to hitter clusters based on center values.

    Uses the cluster center percentile ranks within the centers themselves
    to determine the dominant trait of each cluster.
    """
    n = len(centers)
    # Percentile rank each feature across clusters (0-1)
    pctls = centers[features].rank(pct=True)

    labels = []
    used_names = set()

    # Score each cluster for each archetype template
    templates = [
        ("Power Slugger",
         "Elite power, trades contact for damage",
         lambda p: 0.40 * p["iso"] + 0.35 * p["hard_hit_pct"] + 0.15 * (1 - p["k_rate"]) * -1),
        ("Disciplined Contact",
         "Low K%, high BB%, puts ball in play",
         lambda p: 0.40 * (1 - p["k_rate"]) + 0.35 * p["bb_rate"] + 0.15 * (1 - p["chase_rate"])),
        ("Speed Threat",
         "Speed-first profile with baserunning value",
         lambda p: 0.55 * p["sprint_speed"] + 0.20 * (1 - p["k_rate"]) + 0.15 * p["bb_rate"]),
        ("Balanced All-Around",
         "No glaring weakness, contributes everywhere",
         lambda p: -abs(p["k_rate"] - 0.5) - abs(p["bb_rate"] - 0.5) - abs(p["iso"] - 0.5) + 1.5),
        ("Free Swinger",
         "Aggressive approach, high swing rate, high strikeouts",
         lambda p: 0.35 * p["k_rate"] + 0.30 * p["chase_rate"] + 0.25 * p["whiff_rate"] + 0.10 * (1 - p["bb_rate"])),
        ("Contact-Over-Power",
         "Gets on base with contact, limited pop",
         lambda p: 0.40 * (1 - p["k_rate"]) + 0.30 * (1 - p["iso"]) + 0.20 * (1 - p["whiff_rate"])),
        ("Power-Speed",
         "Rare blend of power and speed",
         lambda p: 0.30 * p["iso"] + 0.30 * p["sprint_speed"] + 0.20 * p["hard_hit_pct"]),
        ("Patient Power",
         "Walks and power with elevated strikeouts",
         lambda p: 0.30 * p["bb_rate"] + 0.30 * p["iso"] + 0.20 * p["hard_hit_pct"] + 0.10 * p["k_rate"]),
    ]

    # For each cluster, find best matching unused template
    # Sort clusters by distinctiveness (max feature pctl - min) to assign most distinct first
    distinctiveness = pctls.max(axis=1) - pctls.min(axis=1)
    cluster_order = distinctiveness.sort_values(ascending=False).index.tolist()

    for idx in cluster_order:
        row = pctls.iloc[idx]
        best_score = -999
        best_name = None
        best_desc = None
        for name, desc, scorer in templates:
            if name in used_names:
                continue
            score = scorer(row)
            if score > best_score:
                best_score = score
                best_name = name
                best_desc = desc
        if best_name is None:
            best_name = f"Cluster {idx + 1}"
            best_desc = "Unlabeled archetype"
        used_names.add(best_name)
        labels.append({
            "cluster_idx": idx,
            "archetype_name": best_name,
            "archetype_desc": best_desc,
        })

    return sorted(labels, key=lambda x: x["cluster_idx"])


def _name_pitcher_clusters(centers: pd.DataFrame, features: list[str]) -> list[dict]:
    """Assign human-readable names to pitcher clusters based on center values."""
    n = len(centers)
    pctls = centers[features].rank(pct=True)

    templates = [
        ("Power Arm",
         "High-velocity strikeout pitcher",
         lambda p: 0.35 * p["k_rate"] + 0.30 * p["avg_velo"] + 0.20 * p["whiff_rate"]),
        ("Ground-Ball Artist",
         "Induces weak contact on the ground",
         lambda p: 0.45 * p["gb_pct"] + 0.20 * (1 - p["k_rate"]) + 0.15 * (1 - p["whiff_rate"])),
        ("Command Specialist",
         "Pitches to contact with elite control",
         lambda p: 0.40 * (1 - p["bb_rate"]) + 0.25 * (1 - p["whiff_rate"]) + 0.15 * (1 - p["k_rate"])),
        ("Swing-and-Miss Reliever",
         "Elite whiff rates from the bullpen",
         lambda p: 0.35 * p["whiff_rate"] + 0.30 * p["k_rate"] + 0.15 * p["usage_breaking"]),
        ("Breaking-Ball Heavy",
         "Relies on curveball/slider as primary weapons",
         lambda p: 0.50 * p["usage_breaking"] + 0.20 * p["whiff_rate"] + 0.15 * p["k_rate"]),
        ("Fastball Dominant",
         "Lives on the heater with plus velocity",
         lambda p: 0.45 * p["usage_ff"] + 0.30 * p["avg_velo"] + 0.10 * p["k_rate"]),
        ("Finesse Pitcher",
         "Low-velocity, high-command pitch-to-contact",
         lambda p: 0.35 * (1 - p["avg_velo"]) + 0.30 * (1 - p["bb_rate"]) + 0.20 * p["gb_pct"]),
        ("Balanced Mix",
         "Even arsenal distribution, no dominant pitch type",
         lambda p: -abs(p["usage_ff"] - 0.5) - abs(p["usage_breaking"] - 0.5) - abs(p["usage_offspeed"] - 0.5) + 1.5),
    ]

    used_names: set[str] = set()
    labels = []
    distinctiveness = pctls.max(axis=1) - pctls.min(axis=1)
    cluster_order = distinctiveness.sort_values(ascending=False).index.tolist()

    for idx in cluster_order:
        row = pctls.iloc[idx]
        best_score = -999
        best_name = None
        best_desc = None
        for name, desc, scorer in templates:
            if name in used_names:
                continue
            score = scorer(row)
            if score > best_score:
                best_score = score
                best_name = name
                best_desc = desc
        if best_name is None:
            best_name = f"Cluster {idx + 1}"
            best_desc = "Unlabeled archetype"
        used_names.add(best_name)
        labels.append({
            "cluster_idx": idx,
            "archetype_name": best_name,
            "archetype_desc": best_desc,
        })

    return sorted(labels, key=lambda x: x["cluster_idx"])


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------
def _model_cache_path(player_type: str, n_clusters: int) -> Path:
    return CACHE_DIR / f"player_archetype_{player_type}_model_k{n_clusters}.joblib"


def _cluster_cache_path(player_type: str, n_clusters: int) -> Path:
    return CACHE_DIR / f"player_archetype_{player_type}_clusters_k{n_clusters}.parquet"


def _assignment_cache_path(player_type: str, season: int, n_clusters: int) -> Path:
    return CACHE_DIR / f"player_archetype_{player_type}_{season}_k{n_clusters}.parquet"


# ===========================================================================
# Hitter feature assembly
# ===========================================================================

HITTER_FEATURES: tuple[str, ...] = (
    "k_rate",
    "bb_rate",
    "iso",
    "hard_hit_pct",
    "sprint_speed",
    "chase_rate",
    "whiff_rate",
)

# Minimum PA per player-season to include in clustering
_HITTER_MIN_PA = 200


def _build_hitter_feature_matrix(seasons: list[int]) -> pd.DataFrame:
    """Assemble hitter feature matrix from cached data.

    Merges extended totals + observed profile + sprint speed into a single
    row-per-player-season DataFrame with the 8 clustering features.
    """
    # Extended totals: PA, K%, BB%, HR, hits, etc.
    extended = build_multi_season_hitter_extended(seasons, min_pa=_HITTER_MIN_PA)

    # Compute ISO = (total_bases - hits) / (pa - bb - hbp)
    # Total bases: singles + 2*doubles + 3*triples + 4*HR
    if "total_bases" in extended.columns and "hits" in extended.columns:
        ab_proxy = extended["pa"] - extended["bb"] - extended["hit_by_pitch"].fillna(0)
        extended["iso"] = (extended["total_bases"] - extended["hits"]) / ab_proxy.clip(lower=1)
    else:
        # Fallback: approximate from HR rate
        extended["iso"] = extended["hr_rate"] * 2.5

    all_frames = []
    for season in seasons:
        season_df = extended[extended["season"] == season].copy()
        if season_df.empty:
            continue

        # Merge observed profile (whiff_rate, chase_rate, hard_hit_pct)
        try:
            obs = get_cached_hitter_observed_profile(season)
            merge_cols = ["batter_id"]
            for col in ["whiff_rate", "chase_rate", "hard_hit_pct"]:
                if col in obs.columns:
                    merge_cols.append(col)
            season_df = season_df.merge(obs[merge_cols], on="batter_id", how="left")
        except Exception:
            logger.warning("No hitter observed profile for %d", season)

        # Merge sprint speed
        try:
            sprint = get_cached_sprint_speed(season)
            season_df = season_df.merge(
                sprint[["player_id", "sprint_speed"]].rename(
                    columns={"player_id": "batter_id"}
                ),
                on="batter_id",
                how="left",
            )
        except Exception:
            logger.warning("No sprint speed for %d", season)

        all_frames.append(season_df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Ensure all feature columns exist, fill missing with medians
    for feat in HITTER_FEATURES:
        if feat not in combined.columns:
            combined[feat] = np.nan

    # Drop rows missing too many features
    n_missing = combined[list(HITTER_FEATURES)].isna().sum(axis=1)
    combined = combined[n_missing <= 3].copy()

    # Fill remaining NaN with column median
    for feat in HITTER_FEATURES:
        median = combined[feat].median()
        combined[feat] = combined[feat].fillna(median)

    logger.info(
        "Hitter feature matrix: %d player-seasons across %s",
        len(combined), seasons,
    )
    return combined


# ===========================================================================
# Pitcher feature assembly
# ===========================================================================

PITCHER_FEATURES: tuple[str, ...] = (
    "k_rate",
    "bb_rate",
    "gb_pct",
    "whiff_rate",
    "usage_ff",
    "usage_breaking",
    "usage_offspeed",
    "avg_velo",
)

_PITCHER_MIN_BF = 150

# Pitch type → usage bucket mapping
_FF_TYPES = {"FF", "SI", "FC"}
_BREAKING_TYPES = {"SL", "CU", "KC", "ST", "SV", "SC"}
_OFFSPEED_TYPES = {"CH", "FS", "FO", "KN", "EP"}


def _build_pitcher_feature_matrix(seasons: list[int]) -> pd.DataFrame:
    """Assemble pitcher feature matrix from cached data.

    Merges extended totals + arsenal profiles into a single
    row-per-pitcher-season DataFrame with the 8 clustering features.
    """
    extended = build_multi_season_pitcher_extended(seasons, min_bf=_PITCHER_MIN_BF)

    all_frames = []
    for season in seasons:
        season_df = extended[extended["season"] == season].copy()
        if season_df.empty:
            continue

        # Arsenal: aggregate pitch-type level to pitcher-level usage buckets + avg velo
        try:
            arsenal = get_pitcher_arsenal(season, force_rebuild=False)

            # Usage buckets
            arsenal["bucket"] = "other"
            arsenal.loc[arsenal["pitch_type"].isin(_FF_TYPES), "bucket"] = "ff"
            arsenal.loc[arsenal["pitch_type"].isin(_BREAKING_TYPES), "bucket"] = "breaking"
            arsenal.loc[arsenal["pitch_type"].isin(_OFFSPEED_TYPES), "bucket"] = "offspeed"

            usage = (
                arsenal.groupby(["pitcher_id", "bucket"])
                .agg(pitches=("pitches", "sum"))
                .reset_index()
            )
            total = arsenal.groupby("pitcher_id")["pitches"].sum().rename("total_pitches")
            usage = usage.merge(total, on="pitcher_id", how="left")
            usage["usage_pct"] = usage["pitches"] / usage["total_pitches"].clip(lower=1)

            usage_pivot = usage.pivot_table(
                index="pitcher_id", columns="bucket",
                values="usage_pct", fill_value=0,
            ).reset_index()
            usage_pivot.columns.name = None
            rename_map = {"ff": "usage_ff", "breaking": "usage_breaking", "offspeed": "usage_offspeed"}
            usage_pivot = usage_pivot.rename(columns=rename_map)

            # Average fastball velocity (weighted by pitch count)
            ff_arsenal = arsenal[arsenal["pitch_type"].isin(_FF_TYPES)]
            if not ff_arsenal.empty:
                velo = (
                    ff_arsenal.groupby("pitcher_id")
                    .apply(
                        lambda g: np.average(g["avg_velo"].dropna(), weights=g.loc[g["avg_velo"].notna(), "pitches"])
                        if g["avg_velo"].notna().any() else np.nan,
                        include_groups=False,
                    )
                    .rename("avg_velo")
                    .reset_index()
                )
                usage_pivot = usage_pivot.merge(velo, on="pitcher_id", how="left")

            # Pitcher-level whiff rate
            pitcher_whiff = (
                arsenal.groupby("pitcher_id")
                .agg(whiffs=("whiffs", "sum"), swings=("swings", "sum"))
                .reset_index()
            )
            pitcher_whiff["whiff_rate"] = (
                pitcher_whiff["whiffs"] / pitcher_whiff["swings"].replace(0, np.nan)
            )

            season_df = season_df.merge(
                usage_pivot[["pitcher_id"] + [c for c in usage_pivot.columns if c.startswith("usage_") or c == "avg_velo"]],
                on="pitcher_id", how="left",
            )
            season_df = season_df.merge(
                pitcher_whiff[["pitcher_id", "whiff_rate"]],
                on="pitcher_id", how="left",
                suffixes=("", "_arsenal"),
            )
            # Prefer arsenal whiff_rate if available
            if "whiff_rate_arsenal" in season_df.columns:
                season_df["whiff_rate"] = season_df["whiff_rate_arsenal"].fillna(
                    season_df.get("whiff_rate", np.nan)
                )
                season_df = season_df.drop(columns=["whiff_rate_arsenal"])

        except Exception:
            logger.warning("No arsenal data for %d", season)

        # Merge gb_pct from pitcher observed profile
        try:
            obs = get_cached_pitcher_observed_profile(season)
            if "gb_pct" in obs.columns:
                season_df = season_df.merge(
                    obs[["pitcher_id", "gb_pct"]], on="pitcher_id", how="left",
                )
        except Exception:
            logger.warning("No pitcher observed profile for %d", season)

        # GB% fallback: use outs_per_bf as rough proxy
        if "gb_pct" not in season_df.columns:
            season_df["gb_pct"] = season_df["outs_per_bf"].fillna(0.65)

        all_frames.append(season_df)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Ensure all feature columns exist
    for feat in PITCHER_FEATURES:
        if feat not in combined.columns:
            combined[feat] = np.nan

    # Drop rows missing too many features
    n_missing = combined[list(PITCHER_FEATURES)].isna().sum(axis=1)
    combined = combined[n_missing <= 3].copy()

    # Fill remaining NaN with column median
    for feat in PITCHER_FEATURES:
        median = combined[feat].median()
        combined[feat] = combined[feat].fillna(median)

    logger.info(
        "Pitcher feature matrix: %d player-seasons across %s",
        len(combined), seasons,
    )
    return combined


# ===========================================================================
# Fit + assign (shared logic)
# ===========================================================================

def fit_player_archetypes(
    player_type: str,
    seasons: list[int] | None = None,
    n_clusters: int = 6,
    random_state: int = 42,
    force_rebuild: bool = False,
) -> dict:
    """Fit StandardScaler + KMeans on pooled player-seasons.

    Parameters
    ----------
    player_type : str
        ``"hitter"`` or ``"pitcher"``.
    seasons : list[int] | None
        Reference seasons. Defaults to model.yaml clustering seasons.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed.
    force_rebuild : bool
        Ignore cache and refit.

    Returns
    -------
    dict
        Keys: ``scaler``, ``kmeans``, ``clusters`` (metadata DataFrame),
        ``features``, ``seasons``.
    """
    model_path = _model_cache_path(player_type, n_clusters)
    cluster_path = _cluster_cache_path(player_type, n_clusters)

    if model_path.exists() and cluster_path.exists() and not force_rebuild:
        logger.info("Loading cached %s archetype model (k=%d)", player_type, n_clusters)
        artifacts = joblib.load(model_path)
        artifacts["clusters"] = pd.read_parquet(cluster_path)
        return artifacts

    if seasons is None:
        seasons = _get_clustering_seasons()

    logger.info(
        "Fitting %s archetypes on seasons %s (k=%d)", player_type, seasons, n_clusters,
    )

    if player_type == "hitter":
        features = list(HITTER_FEATURES)
        data = _build_hitter_feature_matrix(seasons)
        weight_col = "pa"
        id_col = "batter_id"
    elif player_type == "pitcher":
        features = list(PITCHER_FEATURES)
        data = _build_pitcher_feature_matrix(seasons)
        weight_col = "batters_faced"
        id_col = "pitcher_id"
    else:
        raise ValueError(f"Unknown player_type: {player_type!r}")

    if len(data) < n_clusters:
        raise ValueError(
            f"n_clusters ({n_clusters}) exceeds available player-seasons ({len(data)})."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    weights = data[weight_col].to_numpy(dtype=float)
    kmeans.fit(X, sample_weight=weights)

    # Build cluster metadata
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features,
    )
    centers["player_archetype"] = np.arange(1, n_clusters + 1)

    # Count players per cluster
    labels = kmeans.labels_
    data["player_archetype"] = labels + 1
    cluster_counts = (
        data.groupby("player_archetype")
        .agg(n_player_seasons=(id_col, "size"))
        .reset_index()
    )
    centers = centers.merge(cluster_counts, on="player_archetype", how="left")

    # Assign human-readable names
    if player_type == "hitter":
        name_info = _name_hitter_clusters(centers, features)
    else:
        name_info = _name_pitcher_clusters(centers, features)

    for info in name_info:
        idx = info["cluster_idx"]
        centers.loc[idx, "archetype_name"] = info["archetype_name"]
        centers.loc[idx, "archetype_desc"] = info["archetype_desc"]

    centers = centers.sort_values("player_archetype").reset_index(drop=True)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "scaler": scaler,
        "kmeans": kmeans,
        "features": features,
        "seasons": seasons,
    }
    joblib.dump(artifacts, model_path)
    centers.to_parquet(cluster_path, index=False)
    logger.info(
        "Cached %s archetype model → %s, clusters → %s",
        player_type, model_path, cluster_path,
    )

    artifacts["clusters"] = centers
    return artifacts


def assign_player_archetypes(
    player_type: str,
    season: int,
    artifacts: dict | None = None,
    n_clusters: int = 6,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Assign archetypes to one season's players using a fitted model.

    Parameters
    ----------
    player_type : str
        ``"hitter"`` or ``"pitcher"``.
    season : int
        MLB season year.
    artifacts : dict | None
        Pre-loaded model from ``fit_player_archetypes()``.
    n_clusters : int
        Must match fitted model.
    force_rebuild : bool
        Re-assign even if cached.

    Returns
    -------
    pd.DataFrame
        Player data with ``player_archetype`` and ``archetype_name`` columns.
    """
    cache_path = _assignment_cache_path(player_type, season, n_clusters)
    if cache_path.exists() and not force_rebuild:
        logger.info(
            "Loading cached %s archetype assignments for %d (k=%d)",
            player_type, season, n_clusters,
        )
        return pd.read_parquet(cache_path)

    if artifacts is None:
        artifacts = fit_player_archetypes(
            player_type, n_clusters=n_clusters, force_rebuild=False,
        )

    scaler: StandardScaler = artifacts["scaler"]
    kmeans: KMeans = artifacts["kmeans"]
    features: list[str] = artifacts["features"]

    if player_type == "hitter":
        data = _build_hitter_feature_matrix([season])
        id_col = "batter_id"
        name_col = "batter_name"
    else:
        data = _build_pitcher_feature_matrix([season])
        id_col = "pitcher_id"
        name_col = "pitcher_name"

    if data.empty:
        logger.warning("No %s data for %d", player_type, season)
        return pd.DataFrame()

    X = scaler.transform(data[features])
    data["player_archetype"] = kmeans.predict(X) + 1

    # Merge archetype names from cluster metadata
    clusters = artifacts["clusters"]
    name_map = clusters.set_index("player_archetype")["archetype_name"].to_dict()
    desc_map = clusters.set_index("player_archetype")["archetype_desc"].to_dict()
    data["archetype_name"] = data["player_archetype"].map(name_map)
    data["archetype_desc"] = data["player_archetype"].map(desc_map)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache_path, index=False)
    logger.info(
        "Cached %s archetype assignments for %d → %s (%d rows)",
        player_type, season, cache_path, len(data),
    )
    return data


# ===========================================================================
# Public API — main entry points
# ===========================================================================

def get_hitter_archetypes(
    season: int,
    n_clusters: int = 6,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return hitter archetype assignments for a season.

    Ensures the reference model is fitted, then assigns for the requested
    season. This is the main entry point for downstream consumers.
    """
    artifacts = fit_player_archetypes(
        "hitter", n_clusters=n_clusters, force_rebuild=force_rebuild,
    )
    return assign_player_archetypes(
        "hitter", season, artifacts=artifacts,
        n_clusters=n_clusters, force_rebuild=force_rebuild,
    )


def get_pitcher_archetypes(
    season: int,
    n_clusters: int = 6,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return pitcher archetype assignments for a season.

    Ensures the reference model is fitted, then assigns for the requested
    season. This is the main entry point for downstream consumers.
    """
    artifacts = fit_player_archetypes(
        "pitcher", n_clusters=n_clusters, force_rebuild=force_rebuild,
    )
    return assign_player_archetypes(
        "pitcher", season, artifacts=artifacts,
        n_clusters=n_clusters, force_rebuild=force_rebuild,
    )


def get_hitter_cluster_metadata(
    n_clusters: int = 6,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return hitter cluster center metadata."""
    artifacts = fit_player_archetypes(
        "hitter", n_clusters=n_clusters, force_rebuild=force_rebuild,
    )
    return artifacts["clusters"]


def get_pitcher_cluster_metadata(
    n_clusters: int = 6,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return pitcher cluster center metadata."""
    artifacts = fit_player_archetypes(
        "pitcher", n_clusters=n_clusters, force_rebuild=force_rebuild,
    )
    return artifacts["clusters"]


# ===========================================================================
# Dashboard export — simplified parquets for tdd-dashboard consumption
# ===========================================================================

def export_for_dashboard(
    seasons: list[int] | None = None,
    export_season: int = 2025,
    n_hitter_clusters: int = 6,
    n_pitcher_clusters: int = 6,
    output_dir: Path | str | None = None,
    force_rebuild: bool = False,
) -> dict[str, Path]:
    """Export archetype data for the dashboard.

    Writes:
    - ``hitter_archetypes.parquet``  — per-player assignments (id, name, archetype, features)
    - ``pitcher_archetypes.parquet`` — per-player assignments
    - ``hitter_archetype_metadata.parquet``  — cluster centers + labels
    - ``pitcher_archetype_metadata.parquet`` — cluster centers + labels

    Parameters
    ----------
    seasons : list[int] | None
        Seasons to fit on. Defaults to model.yaml clustering seasons.
    export_season : int
        Season to export assignments for.
    n_hitter_clusters, n_pitcher_clusters : int
        Cluster counts.
    output_dir : Path | str | None
        Where to write. Defaults to tdd-dashboard/data/dashboard/.
    force_rebuild : bool
        Refit and re-assign everything.

    Returns
    -------
    dict[str, Path]
        Paths to written parquets.
    """
    if output_dir is None:
        output_dir = Path("C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = {}

    # --- Hitters ---
    h_artifacts = fit_player_archetypes(
        "hitter", seasons=seasons, n_clusters=n_hitter_clusters,
        force_rebuild=force_rebuild,
    )
    h_assignments = assign_player_archetypes(
        "hitter", export_season, artifacts=h_artifacts,
        n_clusters=n_hitter_clusters, force_rebuild=force_rebuild,
    )

    if not h_assignments.empty:
        # Slim down to essential columns
        h_id = "batter_id"
        h_name = "batter_name"
        h_cols = [h_id, h_name, "season", "pa", "player_archetype",
                  "archetype_name", "archetype_desc"] + list(HITTER_FEATURES)
        h_avail = [c for c in h_cols if c in h_assignments.columns]
        h_out = h_assignments[h_avail].copy()

        path = output_dir / "hitter_archetypes.parquet"
        h_out.to_parquet(path, index=False)
        written["hitter_archetypes"] = path
        logger.info("Wrote %s (%d rows)", path, len(h_out))

    h_meta = h_artifacts["clusters"]
    path = output_dir / "hitter_archetype_metadata.parquet"
    h_meta.to_parquet(path, index=False)
    written["hitter_archetype_metadata"] = path
    logger.info("Wrote %s (%d rows)", path, len(h_meta))

    # --- Pitchers ---
    p_artifacts = fit_player_archetypes(
        "pitcher", seasons=seasons, n_clusters=n_pitcher_clusters,
        force_rebuild=force_rebuild,
    )
    p_assignments = assign_player_archetypes(
        "pitcher", export_season, artifacts=p_artifacts,
        n_clusters=n_pitcher_clusters, force_rebuild=force_rebuild,
    )

    if not p_assignments.empty:
        p_id = "pitcher_id"
        p_name = "pitcher_name"
        p_cols = [p_id, p_name, "season", "batters_faced", "is_starter",
                  "player_archetype", "archetype_name", "archetype_desc"] + list(PITCHER_FEATURES)
        p_avail = [c for c in p_cols if c in p_assignments.columns]
        p_out = p_assignments[p_avail].copy()

        path = output_dir / "pitcher_archetypes.parquet"
        p_out.to_parquet(path, index=False)
        written["pitcher_archetypes"] = path
        logger.info("Wrote %s (%d rows)", path, len(p_out))

    p_meta = p_artifacts["clusters"]
    path = output_dir / "pitcher_archetype_metadata.parquet"
    p_meta.to_parquet(path, index=False)
    written["pitcher_archetype_metadata"] = path
    logger.info("Wrote %s (%d rows)", path, len(p_meta))

    return written


# ===========================================================================
# Diagnostic — not called in pipeline
# ===========================================================================

def evaluate_cluster_counts(
    player_type: str,
    seasons: list[int] | None = None,
    k_range: range = range(3, 10),
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate a range of k values for elbow/silhouette analysis.

    Returns a DataFrame with ``k``, ``inertia``, ``silhouette_score``.
    Does NOT modify cached models.
    """
    if seasons is None:
        seasons = _get_clustering_seasons()

    if player_type == "hitter":
        features = list(HITTER_FEATURES)
        data = _build_hitter_feature_matrix(seasons)
        weight_col = "pa"
    else:
        features = list(PITCHER_FEATURES)
        data = _build_pitcher_feature_matrix(seasons)
        weight_col = "batters_faced"

    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    weights = data[weight_col].to_numpy(dtype=float)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X, sample_weight=weights)
        sil = silhouette_score(X, km.labels_)
        results.append({"k": k, "inertia": km.inertia_, "silhouette_score": sil})
        logger.info("k=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, sil)

    return pd.DataFrame(results)
