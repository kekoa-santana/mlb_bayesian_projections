"""Hitter breakout archetype model — GMM-derived.

Uses Gaussian Mixture Models fit on historical breakout hitters to
discover natural breakout archetypes from the data.

k=2 (BIC-optimal):

1. **Diamond in the Rough**: Low batted-ball quality, low pre-breakout wOBA.
   Breakout comes from approach changes, BABIP regression, or unlocking
   latent tools.  Largest improvement magnitude.

2. **Power Surge**: High exit velo, hard-hit%, xwOBA.  The tools were
   always there — results catch up to underlying quality.

Archetypes are auto-assigned based on cluster centroids (avg_exit_velo
separates the two groups).  Breakout score = GMM profile similarity
to historical breakouts × room to grow (age + trajectory).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.data.db import read_sql

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(
    "C:/Users/kekoa/Documents/data_analytics/tdd-dashboard/data/dashboard"
)

# ---------------------------------------------------------------------------
# GMM configuration
# ---------------------------------------------------------------------------

_N_CLUSTERS = 2  # BIC-optimal

# Features used for GMM (percentile-ranked within season)
_GMM_FEATURES = [
    "barrel_pct", "z_contact_pct", "pull_pct", "hard_hit_pct",
    "avg_exit_velo", "chase_rate", "bb_pct", "sprint_speed",
    "oaa", "age", "k_pct", "woba", "xwoba",
]

# Training folds (prediction_season, outcome_season)
_TRAIN_FOLDS = [(2022, 2023), (2023, 2024), (2024, 2025)]

# Breakout thresholds
_WOBA_GAIN_THRESHOLD = 0.020
_WOBA_OUTCOME_FLOOR = 0.300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pctl(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1, higher = better)."""
    return series.rank(pct=True, method="average")


def _inv_pctl(series: pd.Series) -> pd.Series:
    """Inverse percentile rank (lower raw value = higher score)."""
    return 1.0 - series.rank(pct=True, method="average")


def _pctl_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank GMM features within the season cohort."""
    out = df.copy()
    for col in _GMM_FEATURES:
        if col in out.columns:
            out[f"{col}_pctl"] = out[col].rank(pct=True, method="average")
        else:
            out[f"{col}_pctl"] = 0.50
    return out


# ---------------------------------------------------------------------------
# Data loading (DB-only, works for any season)
# ---------------------------------------------------------------------------

def _load_hitter_features_db(
    season: int,
    min_pa: int = 200,
) -> pd.DataFrame:
    """Load all hitter breakout features for a season from the database."""
    from src.data.queries import (
        get_batted_ball_spray,
        get_hitter_observed_profile,
        get_sprint_speed,
    )

    # Batting advanced
    adv = read_sql(
        f"SELECT batter_id, pa, barrel_pct, hard_hit_pct, woba, xwoba, "
        f"k_pct, bb_pct, wrc_plus "
        f"FROM production.fact_batting_advanced "
        f"WHERE season = {season} AND pa >= {min_pa}",
        {},
    )

    # Observed profile (z_contact, chase, exit velo)
    obs = get_hitter_observed_profile(season)

    # Spray data
    spray = get_batted_ball_spray(season)

    # OAA
    oaa = read_sql(
        f"SELECT player_id AS batter_id, SUM(outs_above_average) AS oaa "
        f"FROM production.fact_fielding_oaa WHERE season = {season} "
        f"GROUP BY player_id",
        {},
    )

    # Sprint speed
    sprint = get_sprint_speed(season)

    # Demographics
    demo = read_sql(
        f"SELECT player_id AS batter_id, player_name AS batter_name, "
        f"({season} - EXTRACT(YEAR FROM birth_date))::int AS age "
        f"FROM production.dim_player",
        {},
    )

    # YoY deltas (trajectory proxy)
    prior = season - 1 if season != 2021 else 2019  # skip 2020
    adv_prior = read_sql(
        f"SELECT batter_id, k_pct AS k_prior, bb_pct AS bb_prior "
        f"FROM production.fact_batting_advanced "
        f"WHERE season = {prior} AND pa >= 100",
        {},
    )

    # Position
    from src.models.player_rankings import _assign_hitter_positions
    positions = _assign_hitter_positions(season=season)

    # --- Assemble ---
    base = adv.copy()
    if not obs.empty:
        obs_cols = ["batter_id", "z_contact_pct", "chase_rate", "avg_exit_velo"]
        obs_cols = [c for c in obs_cols if c in obs.columns]
        base = base.merge(obs[obs_cols], on="batter_id", how="left")
    for c in ["z_contact_pct", "chase_rate", "avg_exit_velo"]:
        if c not in base.columns:
            base[c] = np.nan

    if not spray.empty:
        base = base.merge(spray[["batter_id", "pull_pct"]], on="batter_id", how="left")
    if "pull_pct" not in base.columns:
        base["pull_pct"] = np.nan

    base = base.merge(oaa, on="batter_id", how="left")
    base["oaa"] = base["oaa"].fillna(0).astype(float)

    if not sprint.empty:
        base = base.merge(
            sprint[["player_id", "sprint_speed"]].rename(
                columns={"player_id": "batter_id"}
            ),
            on="batter_id", how="left",
        )
    if "sprint_speed" not in base.columns:
        base["sprint_speed"] = np.nan

    base = base.merge(demo, on="batter_id", how="left")
    base = base.merge(adv_prior, on="batter_id", how="left")
    base["delta_k_rate"] = base["k_pct"] - base["k_prior"]
    base["delta_bb_rate"] = base["bb_pct"] - base["bb_prior"]
    base = base.merge(
        positions.rename(columns={"player_id": "batter_id"}),
        on="batter_id", how="left",
    )

    # Fill NaN with medians
    for col in _GMM_FEATURES + ["delta_k_rate", "delta_bb_rate"]:
        if col in base.columns:
            base[col] = base[col].fillna(base[col].median())
    base["age"] = base["age"].fillna(28)

    return base


# ---------------------------------------------------------------------------
# GMM training
# ---------------------------------------------------------------------------

def _build_training_data(
    folds: list[tuple[int, int]] = _TRAIN_FOLDS,
    min_pa: int = 200,
    min_pa_outcome: int = 150,
) -> pd.DataFrame:
    """Build pooled breakout player feature vectors for GMM training.

    Returns percentile-ranked features for players who actually broke out.
    """
    all_breakouts: list[pd.DataFrame] = []
    pctl_cols = [f"{c}_pctl" for c in _GMM_FEATURES]

    for pred_season, outcome_season in folds:
        features = _load_hitter_features_db(pred_season, min_pa)
        ranked = _pctl_rank_features(features)

        outcomes = read_sql(
            f"SELECT batter_id, woba AS woba_next "
            f"FROM production.fact_batting_advanced "
            f"WHERE season = {outcome_season} AND pa >= {min_pa_outcome}",
            {},
        )
        merged = ranked.merge(outcomes, on="batter_id", how="inner")
        merged["woba_delta"] = merged["woba_next"] - merged["woba"]

        breakouts = merged[
            (merged["woba_delta"] >= _WOBA_GAIN_THRESHOLD)
            & (merged["woba_next"] >= _WOBA_OUTCOME_FLOOR)
        ]
        all_breakouts.append(breakouts[pctl_cols])
        logger.info(
            "  %d->%d: %d breakout hitters (of %d qualified)",
            pred_season, outcome_season, len(breakouts), len(merged),
        )

    pooled = pd.concat(all_breakouts, ignore_index=True).dropna()
    logger.info("Total breakout hitters for GMM training: %d", len(pooled))
    return pooled


def _fit_gmm(
    training_data: pd.DataFrame,
) -> tuple[GaussianMixture, StandardScaler, dict[int, str]]:
    """Fit GMM and auto-assign cluster names from centroids.

    Returns (gmm, scaler, cluster_name_map).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(training_data)

    gmm = GaussianMixture(
        n_components=_N_CLUSTERS,
        covariance_type="full",
        random_state=42,
        n_init=5,
    )
    gmm.fit(X)
    logger.info("GMM converged: %s (BIC=%.0f)", gmm.converged_, gmm.bic(X))

    # Auto-assign names: cluster with higher avg_exit_velo centroid = Power Surge
    centroids = scaler.inverse_transform(gmm.means_)
    ev_idx = [f"{c}_pctl" for c in _GMM_FEATURES].index("avg_exit_velo_pctl")
    high_power = int(centroids[:, ev_idx].argmax())

    names: dict[int, str] = {}
    for i in range(_N_CLUSTERS):
        names[i] = "Power Surge" if i == high_power else "Diamond in the Rough"

    for i, name in names.items():
        logger.info("  Cluster %d (%s): n_train=%d", i, name,
                     (gmm.predict(X) == i).sum())
    return gmm, scaler, names


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

def _generate_narrative(row: pd.Series) -> str:
    """Generate a stat-driven explanation for why this player was flagged."""
    archetype = row.get("breakout_type", "")
    age = int(row.get("age", 0))
    woba = row.get("woba", 0.0) or 0.0
    xwoba = row.get("xwoba", 0.0) or 0.0
    hole = row.get("breakout_hole", "General development")

    if archetype == "Diamond in the Rough":
        # Highlight existing tools
        strengths: list[str] = []
        sprint = row.get("sprint_speed", 0) or 0
        if sprint > 28.0:
            strengths.append(f"{sprint:.1f} ft/s sprint speed")
        oaa = row.get("oaa", 0) or 0
        if oaa >= 5:
            strengths.append(f"{int(oaa)} OAA")
        zc = row.get("z_contact_pct", 0) or 0
        if zc > 0.82:
            strengths.append(f"{zc:.0%} zone contact")
        chase = row.get("chase_rate", 0) or 0
        if chase and chase < 0.25:
            strengths.append(f"elite {chase:.0%} chase rate")

        if not strengths:
            strengths.append(f"age-{age} upside")

        tools = " + ".join(strengths[:2])
        return (
            f"{tools}, but just {woba:.3f} wOBA. "
            f"{hole} is the development key at age {age}."
        )

    else:  # Power Surge
        strengths = []
        ev = row.get("avg_exit_velo", 0) or 0
        if ev > 89:
            strengths.append(f"{ev:.1f} mph exit velo")
        hh = row.get("hard_hit_pct", 0) or 0
        if hh > 0.40:
            strengths.append(f"{hh:.0%} hard-hit rate")
        barrel = row.get("barrel_pct", 0) or 0
        if barrel > 0.05:
            strengths.append(f"{barrel:.1%} barrel rate")

        tools = " + ".join(strengths[:2]) if strengths else "strong batted ball profile"
        gap = xwoba - woba
        if gap > 0.015:
            return (
                f"{tools} with {xwoba:.3f} xwOBA vs {woba:.3f} wOBA "
                f"({gap * 1000:.0f}-pt gap). "
                f"Production due for correction at age {age}."
            )
        return (
            f"{tools} but {woba:.3f} wOBA. "
            f"{hole} is the unlock at age {age}."
        )


# ---------------------------------------------------------------------------
# Hole identification
# ---------------------------------------------------------------------------

def _identify_hole(row: pd.Series, archetype: str) -> str:
    """Identify the primary fixable weakness for a breakout candidate."""
    holes: list[tuple[float, str]] = []

    if archetype == "Power Surge":
        # Tools are there — what's holding back production?
        k_pctl = row.get("k_pct_pctl", 0.5)
        if k_pctl > 0.55:
            holes.append((k_pctl, "Strikeout rate"))
        xwoba_gap = row.get("xwoba", 0) - row.get("woba", 0)
        if xwoba_gap > 0.010:
            holes.append((min(xwoba_gap * 30, 1.0), "Batted ball luck"))
        pull_pctl = row.get("pull_pct_pctl", 0.5)
        if pull_pctl < 0.45:
            holes.append((1.0 - pull_pctl, "Pull approach"))
    else:  # Diamond in the Rough
        barrel_pctl = row.get("barrel_pct_pctl", 0.5)
        if barrel_pctl < 0.45:
            holes.append((1.0 - barrel_pctl, "Power development"))
        k_pctl = row.get("k_pct_pctl", 0.5)
        if k_pctl > 0.55:
            holes.append((k_pctl, "Contact quality"))
        ev_pctl = row.get("avg_exit_velo_pctl", 0.5)
        if ev_pctl < 0.45:
            holes.append((1.0 - ev_pctl, "Bat speed"))

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
    """Score all qualified hitters for breakout potential using GMM archetypes.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_pa : int
        Minimum PA to qualify.

    Returns
    -------
    pd.DataFrame
        One row per qualified hitter with GMM-derived breakout scores,
        archetype assignment, and room-to-grow analysis.
    """
    # --- 1. Train GMM on historical breakouts ---
    logger.info("Building GMM training data from historical breakouts...")
    training = _build_training_data()
    gmm, scaler, cluster_names = _fit_gmm(training)

    # --- 2. Load and rank current season ---
    logger.info("Loading features for season %d...", season)
    df = _load_hitter_features_db(season, min_pa)
    if df.empty:
        logger.warning("No qualified hitters for breakout scoring")
        return pd.DataFrame()

    df = _pctl_rank_features(df)
    pctl_cols = [f"{c}_pctl" for c in _GMM_FEATURES]
    X = scaler.transform(df[pctl_cols].fillna(0.5))

    # --- 3. GMM scoring ---
    log_liks = gmm.score_samples(X)
    df["gmm_fit"] = _pctl(pd.Series(log_liks, index=df.index))

    probs = gmm.predict_proba(X)
    clusters = gmm.predict(X)
    df["breakout_type"] = pd.Series(clusters, index=df.index).map(cluster_names)

    for i, name in cluster_names.items():
        col = f"prob_{name.lower().replace(' ', '_')}"
        df[col] = probs[:, i]

    # --- 4. Room to grow (age + trajectory) ---
    # GMM fit already captures production context (wOBA/xwOBA are features),
    # so room_to_grow focuses on developmental runway only.
    age = df["age"].fillna(28)
    age_mult = np.clip(1.0 - (age - 23) / 12.0, 0.10, 1.0)

    delta_k = df["delta_k_rate"].fillna(0)
    delta_bb = df["delta_bb_rate"].fillna(0)
    k_improving = _inv_pctl(delta_k)
    bb_improving = _pctl(delta_bb)
    trajectory = 0.50 * k_improving + 0.50 * bb_improving

    df["room_to_grow"] = trajectory * age_mult

    # --- 5. Breakout score ---
    df["breakout_score"] = df["gmm_fit"] * df["room_to_grow"]

    # --- 6. Tiers (within-archetype ranking) ---
    # Top 10 per archetype = "Breakout Candidate" (our actual picks)
    # Next 15 per archetype = "On the Radar" (watchlist)
    # Rest = classified by archetype but not flagged
    df["archetype_rank"] = (
        df.groupby("breakout_type")["breakout_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    df["breakout_tier"] = np.where(
        df["archetype_rank"] <= 10, "Breakout Candidate",
        np.where(df["archetype_rank"] <= 25, "On the Radar", ""),
    )
    df["breakout_rank"] = (
        df["breakout_score"].rank(ascending=False, method="min").astype(int)
    )

    # --- 7. Hole identification + narrative ---
    df["breakout_hole"] = df.apply(
        lambda row: _identify_hole(row, row["breakout_type"]),
        axis=1,
    )
    df["breakout_narrative"] = df.apply(_generate_narrative, axis=1)

    # --- 8. Output ---
    df = df.sort_values("breakout_rank")
    output_cols = [
        "batter_id", "batter_name", "age", "position", "batter_stand",
        # GMM scores
        "gmm_fit", "breakout_type",
        "prob_power_surge", "prob_diamond_in_the_rough",
        "room_to_grow", "breakout_score",
        "breakout_rank", "archetype_rank", "breakout_tier",
        "breakout_hole", "breakout_narrative",
        # Key stats
        "pa", "woba", "xwoba", "wrc_plus",
        "barrel_pct", "z_contact_pct", "pull_pct",
        "hard_hit_pct", "avg_exit_velo", "sprint_speed", "oaa",
        "k_pct", "bb_pct", "chase_rate",
        # Trajectory
        "delta_k_rate", "delta_bb_rate",
    ]
    # Rename for backward compat with rankings merge
    if "batter_stand" not in df.columns:
        df["batter_stand"] = ""
    available = [c for c in output_cols if c in df.columns]
    result = df[available].reset_index(drop=True)

    n_high = (result["breakout_tier"] == "High").sum()
    n_med = (result["breakout_tier"] == "Medium").sum()
    logger.info(
        "Breakout candidates: %d High, %d Medium (of %d total)",
        n_high, n_med, len(result),
    )
    return result
