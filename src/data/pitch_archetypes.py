"""
Pitch archetype clustering built from production.sat_pitch_shape.

Key design decisions vs. the original module:
- **Pooled multi-season fit**: one StandardScaler + KMeans trained on a
  reference window (model.yaml `seasons.clustering`), so archetypes are comparable
  across all seasons.
- **LHP/RHP normalization**: pfx_x is flipped for left-handers before
  clustering so a LHP fastball and a RHP fastball land in the same archetype.
- **SQL-level aggregation**: `get_pitch_shape_offerings()` returns ~3,800
  rows/season instead of ~600K raw pitches.
- **5-feature clustering**: release_pos_x/z kept in output but not used for
  clustering (they are pitcher anatomy, not pitch shape).
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.queries import get_pitch_shape_offerings
from src.utils.constants import PITCH_TO_FAMILY

from src.data.paths import CACHE_DIR, get_clustering_seasons as _get_clustering_seasons

logger = logging.getLogger(__name__)


# Features used for KMeans clustering (5 shape features).
SHAPE_FEATURES: tuple[str, ...] = (
    "release_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "release_extension",
)

# Kept in output for tunneling analysis but NOT used in clustering.
EXTRA_FEATURES: tuple[str, ...] = (
    "release_pos_x",
    "release_pos_z",
)



# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------
def _model_cache_path(n_clusters: int) -> Path:
    return CACHE_DIR / f"pitch_archetype_model_k{n_clusters}.joblib"


def _cluster_cache_path(n_clusters: int) -> Path:
    return CACHE_DIR / f"pitch_archetype_clusters_k{n_clusters}.parquet"


def _assignment_cache_path(season: int, n_clusters: int) -> Path:
    return CACHE_DIR / f"pitch_archetype_offerings_{season}_k{n_clusters}.parquet"


# ---------------------------------------------------------------------------
# LHP normalization
# ---------------------------------------------------------------------------
def _normalize_handedness(offerings: pd.DataFrame) -> pd.DataFrame:
    """Flip ``pfx_x`` for left-handed pitchers so movement is comparable.

    Adds a ``pfx_x_flipped`` boolean column (True where pfx_x was negated).
    """
    df = offerings.copy()
    is_lhp = df["pitch_hand"] == "L"
    df["pfx_x_flipped"] = is_lhp
    df.loc[is_lhp, "pfx_x"] = -df.loc[is_lhp, "pfx_x"]
    return df


# ---------------------------------------------------------------------------
# Pooled multi-season fit
# ---------------------------------------------------------------------------
def fit_pitch_archetypes(
    seasons: list[int] | None = None,
    n_clusters: int = 8,
    random_state: int = 42,
    force_rebuild: bool = False,
) -> dict:
    """Fit StandardScaler + KMeans on pooled offerings from reference seasons.

    Parameters
    ----------
    seasons : list[int] | None
        Reference seasons to pool.  Defaults to ``model.yaml`` clustering seasons.
    n_clusters : int
        Number of KMeans clusters.
    random_state : int
        Random seed for reproducibility.
    force_rebuild : bool
        If True, ignore cached artifacts and refit.

    Returns
    -------
    dict
        Keys: ``scaler``, ``kmeans``, ``clusters`` (DataFrame of cluster
        metadata), ``seasons`` (list of seasons used).
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2.")

    model_path = _model_cache_path(n_clusters)
    cluster_path = _cluster_cache_path(n_clusters)

    if model_path.exists() and cluster_path.exists() and not force_rebuild:
        logger.info("Loading cached pitch archetype model (k=%d)", n_clusters)
        artifacts = joblib.load(model_path)
        artifacts["clusters"] = pd.read_parquet(cluster_path)
        return artifacts

    if seasons is None:
        seasons = _get_clustering_seasons()

    logger.info(
        "Fitting pitch archetypes on seasons %s (k=%d)", seasons, n_clusters
    )

    # Pool offerings across reference seasons
    frames = []
    for s in seasons:
        df = get_pitch_shape_offerings(s)
        df["pitch_family"] = df["pitch_type"].map(PITCH_TO_FAMILY)
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)

    if len(pooled) < n_clusters:
        raise ValueError(
            f"n_clusters ({n_clusters}) exceeds available offerings "
            f"({len(pooled)})."
        )

    pooled = _normalize_handedness(pooled)

    scaler = StandardScaler()
    X = scaler.fit_transform(pooled[list(SHAPE_FEATURES)])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    kmeans.fit(X, sample_weight=pooled["pitches"].to_numpy(dtype=float))

    # Build cluster metadata
    labels = kmeans.labels_
    pooled["pitch_archetype"] = labels + 1

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=list(SHAPE_FEATURES),
    )
    centers["pitch_archetype"] = np.arange(1, n_clusters + 1)

    cluster_usage = (
        pooled.groupby("pitch_archetype")
        .agg(
            n_offerings=("pitcher_id", "size"),
            total_pitches=("pitches", "sum"),
        )
        .reset_index()
    )

    primary_pitch_type = (
        pooled.groupby(["pitch_archetype", "pitch_type"], as_index=False)
        .agg(total_pitches=("pitches", "sum"))
        .sort_values(
            ["pitch_archetype", "total_pitches"], ascending=[True, False]
        )
        .drop_duplicates(subset=["pitch_archetype"])
        .rename(columns={"pitch_type": "primary_pitch_type"})
    )

    centers = centers.merge(cluster_usage, on="pitch_archetype", how="left")
    centers = centers.merge(
        primary_pitch_type[["pitch_archetype", "primary_pitch_type"]],
        on="pitch_archetype",
        how="left",
    )
    centers["archetype_name"] = (
        "Archetype "
        + centers["pitch_archetype"].astype(str)
        + " ("
        + centers["primary_pitch_type"].fillna("mixed")
        + " primary)"
    )
    centers = centers.sort_values("pitch_archetype").reset_index(drop=True)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "scaler": scaler,
        "kmeans": kmeans,
        "seasons": seasons,
    }
    joblib.dump(artifacts, model_path)
    centers.to_parquet(cluster_path, index=False)
    logger.info(
        "Cached pitch archetype model → %s, clusters → %s (%d rows)",
        model_path,
        cluster_path,
        len(centers),
    )

    artifacts["clusters"] = centers
    return artifacts


# ---------------------------------------------------------------------------
# Per-season assignment
# ---------------------------------------------------------------------------
def assign_pitch_archetypes(
    season: int,
    artifacts: dict | None = None,
    n_clusters: int = 8,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Assign archetypes to one season's offerings using a fitted model.

    Parameters
    ----------
    season : int
        MLB season year.
    artifacts : dict | None
        Pre-loaded model artifacts from ``fit_pitch_archetypes()``.
        If None, loads from cache (fitting first if needed).
    n_clusters : int
        Must match the fitted model's k.
    force_rebuild : bool
        If True, re-assign even if cache exists.

    Returns
    -------
    pd.DataFrame
        Offerings with ``pitch_archetype`` column added.
    """
    cache_path = _assignment_cache_path(season, n_clusters)
    if cache_path.exists() and not force_rebuild:
        logger.info(
            "Loading cached archetype assignments for %d (k=%d)",
            season,
            n_clusters,
        )
        return pd.read_parquet(cache_path)

    if artifacts is None:
        artifacts = fit_pitch_archetypes(
            n_clusters=n_clusters, force_rebuild=False
        )

    scaler: StandardScaler = artifacts["scaler"]
    kmeans: KMeans = artifacts["kmeans"]

    offerings = get_pitch_shape_offerings(season)
    offerings["pitch_family"] = offerings["pitch_type"].map(PITCH_TO_FAMILY)
    offerings = _normalize_handedness(offerings)

    X = scaler.transform(offerings[list(SHAPE_FEATURES)])
    offerings["pitch_archetype"] = kmeans.predict(X) + 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    offerings.to_parquet(cache_path, index=False)
    logger.info(
        "Cached archetype assignments for %d → %s (%d rows)",
        season,
        cache_path,
        len(offerings),
    )
    return offerings


# ---------------------------------------------------------------------------
# Public API  (backward-compatible names)
# ---------------------------------------------------------------------------
def get_pitch_archetype_offerings(
    season: int,
    n_clusters: int = 8,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return pitcher/pitch-type offerings with assigned pitch archetype labels.

    Ensures the reference model is fitted, then assigns for the requested
    season.  This is the main entry point for downstream consumers.

    Parameters
    ----------
    season : int
        MLB season year.
    n_clusters : int
        Number of KMeans clusters.
    force_rebuild : bool
        If True, refit model and re-assign.
    """
    artifacts = fit_pitch_archetypes(
        n_clusters=n_clusters, force_rebuild=force_rebuild
    )
    return assign_pitch_archetypes(
        season=season,
        artifacts=artifacts,
        n_clusters=n_clusters,
        force_rebuild=force_rebuild,
    )


def get_pitch_archetype_clusters(
    n_clusters: int = 8,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return cluster-level pitch archetype summaries and shape centroids.

    Parameters
    ----------
    n_clusters : int
        Number of KMeans clusters.
    force_rebuild : bool
        If True, refit model.
    """
    artifacts = fit_pitch_archetypes(
        n_clusters=n_clusters, force_rebuild=force_rebuild
    )
    return artifacts["clusters"]


# ---------------------------------------------------------------------------
# Diagnostic — not called in pipeline
# ---------------------------------------------------------------------------
def evaluate_cluster_counts(
    seasons: list[int] | None = None,
    k_range: range = range(4, 16),
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate a range of k values on pooled offerings.

    Returns a DataFrame with columns ``k``, ``inertia``, ``silhouette_score``
    for elbow/silhouette analysis.  Does NOT modify cached model.
    """
    if seasons is None:
        seasons = _get_clustering_seasons()

    frames = []
    for s in seasons:
        df = get_pitch_shape_offerings(s)
        df["pitch_family"] = df["pitch_type"].map(PITCH_TO_FAMILY)
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    pooled = _normalize_handedness(pooled)

    scaler = StandardScaler()
    X = scaler.fit_transform(pooled[list(SHAPE_FEATURES)])
    weights = pooled["pitches"].to_numpy(dtype=float)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X, sample_weight=weights)
        sil = silhouette_score(X, km.labels_)
        results.append({"k": k, "inertia": km.inertia_, "silhouette_score": sil})
        logger.info("k=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, sil)

    return pd.DataFrame(results)
