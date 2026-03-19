"""Pitcher breakout archetype model — GMM-derived.

Uses Gaussian Mixture Models fit on historical pitcher breakouts to
discover natural breakout archetypes from the data.

k=3 (data-confirmed):

1. **Command Leap**: High zone%, low BB%, good command metrics.
   ERA catches up to already-good peripherals.  Skews reliever.

2. **Stuff Dominant**: Elite whiff rate, K%, putaway rate, very low
   contact allowed.  Results follow the stuff.  Skews reliever.

3. **ERA Correction**: Highest pre-breakout ERA, lowest K%.  Biggest
   improvement magnitude but worst outcome quality.  Skews starter.
   Breakout through luck regression (HR/FB, BABIP, sequencing).

Archetypes are auto-assigned based on cluster centroids.
Breakout score = GMM profile similarity × room to grow (age + trajectory).
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

_N_CLUSTERS = 3  # data-confirmed

# Features used for GMM (percentile-ranked within season)
_GMM_FEATURES = [
    "whiff_rate", "swstr_pct", "csw_pct", "avg_velo",
    "xwoba_against", "barrel_pct_against",
    "bb_pct", "zone_pct", "chase_pct", "contact_pct", "k_pct",
    "hr_per_fb", "era", "fip", "era_minus_xfip",
    "age", "first_strike_pct", "putaway_rate",
]

# Training folds
_TRAIN_FOLDS = [(2022, 2023), (2023, 2024), (2024, 2025)]

# Breakout thresholds
_ERA_DROP_THRESHOLD = 0.75
_ERA_OUTCOME_CEILING = 4.00


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pctl(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average")


def _inv_pctl(series: pd.Series) -> pd.Series:
    return 1.0 - series.rank(pct=True, method="average")


def _pctl_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _GMM_FEATURES:
        if col in out.columns:
            out[f"{col}_pctl"] = out[col].rank(pct=True, method="average")
        else:
            out[f"{col}_pctl"] = 0.50
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pitcher_features_db(
    season: int,
    min_bf: int = 200,
) -> pd.DataFrame:
    """Load all pitcher breakout features from DB for a given season."""
    from src.data.queries import get_pitcher_efficiency, get_pitcher_fly_ball_data

    # Pitching advanced
    adv = read_sql(
        f"SELECT pitcher_id, batters_faced, k_pct, bb_pct, "
        f"swstr_pct, csw_pct, zone_pct, chase_pct, contact_pct, "
        f"xwoba_against, barrel_pct_against, hard_hit_pct_against, woba_against "
        f"FROM production.fact_pitching_advanced "
        f"WHERE season = {season} AND batters_faced >= {min_bf}",
        {},
    )

    # Pitch-level aggregates (whiff_rate, avg_velo)
    obs = read_sql(
        f"SELECT fp.pitcher_id, "
        f"SUM(fp.is_whiff::int)::float / NULLIF(SUM(fp.is_swing::int), 0) AS whiff_rate, "
        f"AVG(fp.release_speed) AS avg_velo "
        f"FROM production.fact_pitch fp "
        f"JOIN production.dim_game dg ON fp.game_pk = dg.game_pk "
        f"WHERE dg.season = {season} AND dg.game_type = 'R' "
        f"AND fp.pitch_type IS NOT NULL AND fp.release_speed != 'NaN' "
        f"GROUP BY fp.pitcher_id HAVING SUM(fp.is_swing::int) >= 50",
        {},
    )

    # Traditional stats (ERA, FIP, IP, HR)
    trad = read_sql(
        f"SELECT player_id AS pitcher_id, "
        f"SUM(pit_ip) AS ip, SUM(pit_k) AS k_total, "
        f"SUM(pit_bb) AS bb_total, SUM(pit_hr) AS hr_allowed, "
        f"SUM(pit_er) AS earned_runs "
        f"FROM production.fact_player_game_mlb "
        f"WHERE player_role = 'pitcher' AND season = {season} "
        f"GROUP BY player_id HAVING SUM(pit_bf) >= {min_bf}",
        {},
    )
    trad["era"] = trad["earned_runs"] * 9.0 / trad["ip"].clip(lower=1)
    trad["fip"] = (
        (13 * trad["hr_allowed"] + 3 * trad["bb_total"] - 2 * trad["k_total"])
        / trad["ip"].clip(lower=1)
    ) + 3.20

    # Fly ball data
    fb = get_pitcher_fly_ball_data(season)

    # Efficiency
    eff = get_pitcher_efficiency(season)

    # Demographics
    demo = read_sql(
        f"SELECT player_id AS pitcher_id, player_name AS pitcher_name, "
        f"({season} - EXTRACT(YEAR FROM birth_date))::int AS age "
        f"FROM production.dim_player",
        {},
    )

    # YoY deltas
    prior = season - 1 if season != 2021 else 2019
    adv_prior = read_sql(
        f"SELECT pitcher_id, k_pct AS k_prior, bb_pct AS bb_prior "
        f"FROM production.fact_pitching_advanced "
        f"WHERE season = {prior} AND batters_faced >= 100",
        {},
    )

    # --- Assemble ---
    base = adv.copy()
    base = base.merge(obs[["pitcher_id", "whiff_rate", "avg_velo"]],
                       on="pitcher_id", how="left")
    base = base.merge(trad[["pitcher_id", "ip", "hr_allowed", "era", "fip"]],
                       on="pitcher_id", how="left")
    if not fb.empty:
        base = base.merge(fb[["pitcher_id", "fly_balls", "home_runs", "hr_per_fb"]],
                           on="pitcher_id", how="left")
    for c in ["fly_balls", "home_runs", "hr_per_fb"]:
        if c not in base.columns:
            base[c] = np.nan

    # xFIP
    has_fb = (base["fly_balls"].notna() & (base["fly_balls"] > 0)
              & base["ip"].notna() & (base["ip"] > 0))
    if has_fb.any():
        lg_hr_fb = (base.loc[has_fb, "home_runs"].sum()
                    / base.loc[has_fb, "fly_balls"].sum())
    else:
        lg_hr_fb = 0.10
    base["xfip"] = base["fip"].copy()
    base.loc[has_fb, "xfip"] = (
        base.loc[has_fb, "fip"]
        - 13 * (base.loc[has_fb, "hr_allowed"]
                - base.loc[has_fb, "fly_balls"] * lg_hr_fb)
        / base.loc[has_fb, "ip"]
    )
    base["era_minus_xfip"] = base["era"] - base["xfip"]

    if not eff.empty:
        eff_cols = [c for c in ["pitcher_id", "first_strike_pct", "putaway_rate"]
                    if c in eff.columns]
        base = base.merge(eff[eff_cols], on="pitcher_id", how="left")
    for c in ["first_strike_pct", "putaway_rate"]:
        if c not in base.columns:
            base[c] = np.nan

    base = base.merge(demo, on="pitcher_id", how="left")
    base = base.merge(adv_prior, on="pitcher_id", how="left")
    base["delta_k_rate"] = base["k_pct"] - base["k_prior"]
    base["delta_bb_rate"] = base["bb_pct"] - base["bb_prior"]

    # Fill NaN
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
    min_bf: int = 200,
    min_bf_outcome: int = 150,
) -> pd.DataFrame:
    """Build pooled breakout pitcher feature vectors for GMM training."""
    all_breakouts: list[pd.DataFrame] = []
    pctl_cols = [f"{c}_pctl" for c in _GMM_FEATURES]

    for pred_season, outcome_season in folds:
        features = _load_pitcher_features_db(pred_season, min_bf)
        ranked = _pctl_rank_features(features)

        # Outcome ERA
        out_trad = read_sql(
            f"SELECT player_id AS pitcher_id, "
            f"SUM(pit_er)*9.0/NULLIF(SUM(pit_ip),0) AS era_next "
            f"FROM production.fact_player_game_mlb "
            f"WHERE player_role = 'pitcher' AND season = {outcome_season} "
            f"GROUP BY player_id HAVING SUM(pit_bf) >= {min_bf_outcome}",
            {},
        )
        merged = ranked.merge(out_trad, on="pitcher_id", how="inner")
        merged["era_delta"] = merged["era_next"] - merged["era"]

        breakouts = merged[
            (merged["era_delta"] <= -_ERA_DROP_THRESHOLD)
            & (merged["era_next"] <= _ERA_OUTCOME_CEILING)
        ]
        all_breakouts.append(breakouts[pctl_cols])
        logger.info(
            "  %d->%d: %d breakout pitchers (of %d qualified)",
            pred_season, outcome_season, len(breakouts), len(merged),
        )

    pooled = pd.concat(all_breakouts, ignore_index=True).dropna()
    logger.info("Total breakout pitchers for GMM training: %d", len(pooled))
    return pooled


def _fit_gmm(
    training_data: pd.DataFrame,
) -> tuple[GaussianMixture, StandardScaler, dict[int, str]]:
    """Fit GMM and auto-assign cluster names from centroids.

    Naming logic:
    - Highest whiff_rate centroid → "Stuff Dominant"
    - Highest zone_pct centroid (of remaining) → "Command Leap"
    - Remaining → "ERA Correction"
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

    # Auto-assign names from centroids
    pctl_cols = [f"{c}_pctl" for c in _GMM_FEATURES]
    centroids = scaler.inverse_transform(gmm.means_)
    whiff_idx = pctl_cols.index("whiff_rate_pctl")
    zone_idx = pctl_cols.index("zone_pct_pctl")

    # Step 1: highest whiff → Stuff Dominant
    stuff_cluster = int(centroids[:, whiff_idx].argmax())

    # Step 2: of remaining, highest zone% → Command Leap
    remaining = [i for i in range(_N_CLUSTERS) if i != stuff_cluster]
    cmd_cluster = max(remaining, key=lambda i: centroids[i, zone_idx])

    # Step 3: remaining → ERA Correction
    era_cluster = [i for i in range(_N_CLUSTERS)
                   if i not in (stuff_cluster, cmd_cluster)][0]

    names = {
        stuff_cluster: "Stuff Dominant",
        cmd_cluster: "Command Leap",
        era_cluster: "ERA Correction",
    }

    for i, name in names.items():
        logger.info("  Cluster %d (%s): n_train=%d", i, name,
                     (gmm.predict(X) == i).sum())
    return gmm, scaler, names


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

def _generate_pitcher_narrative(row: pd.Series) -> str:
    """Generate a stat-driven explanation for why this pitcher was flagged."""
    archetype = row.get("breakout_type", "")
    age = int(row.get("age", 0))
    era = row.get("era", 0.0) or 0.0
    hole = row.get("breakout_hole", "")

    if archetype == "Stuff Dominant":
        strengths: list[str] = []
        k_pct = row.get("k_pct", 0) or 0
        if k_pct > 0.22:
            strengths.append(f"{k_pct:.1%} K rate")
        swstr = row.get("swstr_pct", 0) or 0
        if swstr > 0.11:
            strengths.append(f"{swstr:.1%} SwStr%")
        velo = row.get("avg_velo", 0) or 0
        if velo > 94:
            strengths.append(f"{velo:.1f} mph")

        tools = " + ".join(strengths[:2]) if strengths else "elite stuff metrics"
        bb = row.get("bb_pct", 0) or 0
        return (
            f"{tools}, but {bb:.1%} walk rate keeps ERA at {era:.2f}. "
            f"Command development is the breakout trigger at age {age}."
        )

    elif archetype == "Command Leap":
        strengths = []
        bb = row.get("bb_pct", 0) or 0
        if bb < 0.08:
            strengths.append(f"low {bb:.1%} walk rate")
        zone = row.get("zone_pct", 0) or 0
        if zone > 0.45:
            strengths.append(f"{zone:.0%} zone rate")
        fs = row.get("first_strike_pct", 0) or 0
        if fs > 0.58:
            strengths.append(f"{fs:.0%} first-pitch strike rate")

        tools = " + ".join(strengths[:2]) if strengths else "solid command profile"
        fip = row.get("fip", 0) or 0
        if fip and abs(era - fip) > 0.3:
            return (
                f"{tools} with {fip:.2f} FIP vs {era:.2f} ERA. "
                f"Command is there — results should converge."
            )
        return (
            f"{tools} but {era:.2f} ERA. "
            f"{hole} at age {age}."
        )

    else:  # ERA Correction
        xfip = row.get("xfip", 0) or 0
        gap = row.get("era_minus_xfip", 0) or 0
        hr_fb = row.get("hr_per_fb", 0) or 0

        parts: list[str] = []
        if gap > 0.5:
            parts.append(f"{era:.2f} ERA vs {xfip:.2f} xFIP ({gap:+.2f} gap)")
        if hr_fb > 0.13:
            parts.append(f"{hr_fb:.1%} HR/FB (league avg ~13%)")

        if parts:
            target = min(era, xfip + 0.50)
            return (
                f"{'. '.join(parts)}. "
                f"Peripherals say sub-{target:.1f} ERA is the true talent level."
            )
        return (
            f"{era:.2f} ERA with peripherals suggesting improvement. "
            f"{hole} is the primary regression driver."
        )


# ---------------------------------------------------------------------------
# Hole identification
# ---------------------------------------------------------------------------

def _identify_pitcher_hole(row: pd.Series, archetype: str) -> str:
    """Identify the primary fixable weakness."""
    holes: list[tuple[float, str]] = []

    if archetype == "Stuff Dominant":
        bb_pctl = row.get("bb_pct_pctl", 0.5)
        if bb_pctl > 0.55:
            holes.append((bb_pctl, "Walk rate"))
        zone_pctl = row.get("zone_pct_pctl", 0.5)
        if zone_pctl < 0.45:
            holes.append((1.0 - zone_pctl, "Zone rate"))
        fs_pctl = row.get("first_strike_pct_pctl", 0.5)
        if fs_pctl < 0.45:
            holes.append((1.0 - fs_pctl, "First-pitch strikes"))
        if not holes:
            holes.append((0.5, "Command development"))

    elif archetype == "Command Leap":
        hr_fb = row.get("hr_per_fb_pctl", 0.5)
        if hr_fb > 0.55:
            holes.append((hr_fb, "HR suppression"))
        era_gap = row.get("era", 4.0) - row.get("fip", 4.0)
        if era_gap > 0.3:
            holes.append((min(era_gap, 1.0), "Run prevention luck"))
        if not holes:
            holes.append((0.5, "Full-season consistency"))

    else:  # ERA Correction
        hr_fb = row.get("hr_per_fb_pctl", 0.5)
        if hr_fb > 0.55:
            holes.append((hr_fb, "HR/FB regression"))
        era_xfip = row.get("era_minus_xfip", 0)
        if era_xfip > 0.5:
            holes.append((min(era_xfip / 2.0, 1.0), "ERA-xFIP convergence"))
        if not holes:
            holes.append((0.5, "Results regression"))

    holes.sort(key=lambda x: x[0], reverse=True)
    return holes[0][1]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_pitcher_breakout_candidates(
    season: int = 2025,
    min_bf: int = 200,
) -> pd.DataFrame:
    """Score all qualified pitchers using GMM-derived breakout archetypes.

    Parameters
    ----------
    season : int
        Most recent completed season.
    min_bf : int
        Minimum batters faced to qualify.

    Returns
    -------
    pd.DataFrame
        One row per pitcher with GMM-derived breakout scores and archetypes.
    """
    # --- 1. Train GMM ---
    logger.info("Building pitcher GMM training data...")
    training = _build_training_data()
    gmm, scaler, cluster_names = _fit_gmm(training)

    # --- 2. Load and rank current season ---
    logger.info("Loading pitcher features for season %d...", season)
    df = _load_pitcher_features_db(season, min_bf)
    if df.empty:
        logger.warning("No qualified pitchers for breakout scoring")
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
        col = f"prob_{name.lower().replace(' ', '_').replace('-', '_')}"
        df[col] = probs[:, i]

    # --- 4. Room to grow (age + trajectory) ---
    age = df["age"].fillna(28)
    age_mult = np.clip(1.0 - (age - 24) / 14.0, 0.15, 1.0)

    delta_k = df["delta_k_rate"].fillna(0)
    delta_bb = df["delta_bb_rate"].fillna(0)
    k_improving = _pctl(delta_k)        # positive delta = more Ks = better
    bb_improving = _inv_pctl(delta_bb)   # negative delta = fewer walks = better
    trajectory = 0.50 * k_improving + 0.50 * bb_improving

    df["room_to_grow"] = trajectory * age_mult

    # --- 5. Breakout score ---
    df["breakout_score"] = df["gmm_fit"] * df["room_to_grow"]

    # --- 6. Tiers (within-archetype ranking) ---
    # Top 10 per archetype = "Breakout Candidate"
    # Next 15 per archetype = "On the Radar"
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
        lambda row: _identify_pitcher_hole(row, row["breakout_type"]),
        axis=1,
    )
    df["breakout_narrative"] = df.apply(_generate_pitcher_narrative, axis=1)

    # --- 8. Output ---
    df = df.sort_values("breakout_rank")

    # Check for is_starter from projections parquet if available
    if "is_starter" not in df.columns:
        proj_path = DASHBOARD_DIR / "pitcher_projections.parquet"
        if proj_path.exists():
            proj = pd.read_parquet(proj_path)[["pitcher_id", "is_starter"]]
            df = df.merge(proj, on="pitcher_id", how="left")
        if "is_starter" not in df.columns:
            df["is_starter"] = np.nan

    output_cols = [
        "pitcher_id", "pitcher_name", "age", "is_starter",
        # GMM scores
        "gmm_fit", "breakout_type",
        "prob_stuff_dominant", "prob_command_leap", "prob_era_correction",
        "room_to_grow", "breakout_score",
        "breakout_rank", "archetype_rank", "breakout_tier",
        "breakout_hole", "breakout_narrative",
        # Key stats
        "batters_faced", "k_pct", "bb_pct",
        "whiff_rate", "swstr_pct", "csw_pct", "avg_velo",
        "contact_pct", "zone_pct", "chase_pct",
        "xwoba_against", "barrel_pct_against",
        "era", "fip", "xfip", "hr_per_fb", "era_minus_xfip",
        "first_strike_pct", "putaway_rate",
        # Trajectory
        "delta_k_rate", "delta_bb_rate",
    ]
    available = [c for c in output_cols if c in df.columns]
    result = df[available].reset_index(drop=True)

    n_high = (result["breakout_tier"] == "High").sum()
    n_med = (result["breakout_tier"] == "Medium").sum()
    logger.info(
        "Pitcher breakout: %d High, %d Medium (of %d total)",
        n_high, n_med, len(result),
    )
    return result
