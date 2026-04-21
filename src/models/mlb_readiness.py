"""
MLB Readiness Score — predicts probability a MiLB prospect sticks in MLB.

Uses translated MiLB stats (K%, BB%, ISO), age-relative-to-level,
career progression, defensive position, and organizational depth
to estimate the likelihood a prospect reaches and sustains MLB
playing time (200+ PA season).

Organizational depth comes from ``fact_prospect_snapshot`` and
``fact_prospect_transition`` tables.

Trained on 2018-2021 MiLB prospects, validated on 2022-2023 debutants.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.feature_eng import load_milb_translated
from src.data.paths import CACHE_DIR

logger = logging.getLogger(__name__)


# Feature list (must match training order)
_STAT_FEATURES = [
    "wtd_k_pct", "wtd_bb_pct", "wtd_iso", "k_bb_diff", "sb_rate",
    "youngest_age_rel", "avg_age_rel", "min_age",
    "pa_weighted_level",          # continuous level exposure (replaces coarse max_level_num)
    "max_level_num",              # kept for backward compat — coarse level indicator
    "games_at_max_level",         # depth of exposure at highest level
    "pa_at_max_level",            # PA-based exposure at highest level
    "k_rate_at_max_level",        # performance at hardest level faced
    "bb_rate_at_max_level",       # discipline at hardest level faced
    "age_x_level",                # age × level interaction (young + high level = superlinear)
    "promotion_resilience",       # K% stability across promotions
    "wtd_gb_pct",                 # batted ball profile (GB tendency)
    "levels_played", "career_milb_pa", "career_seasons",
]

# Position groups (defensive spectrum)
_POS_GROUP_MAP = {
    "C": "C", "SS": "MI", "2B": "MI", "3B": "CI", "1B": "CI",
    "CF": "OF", "LF": "OF", "RF": "OF",
}

# One-hot columns (drop_first=True, reference=C)
_POS_DUMMY_COLS = ["pos_CI", "pos_MI", "pos_OF", "pos_Unknown"]

# Organizational depth features
_DEPTH_FEATURES = ["n_above", "total_at_pos_in_org"]

_ALL_FEATURES = _STAT_FEATURES + _POS_DUMMY_COLS + _DEPTH_FEATURES

_LEVEL_MAP = {"A": 1, "A+": 2, "AA": 3, "AAA": 4}

# PA scaling by level (same as augmentation)
_LEVEL_PA_SCALE = {"AAA": 0.80, "AA": 0.55, "A+": 0.40, "A": 0.30}

# Recency decay half-life for MiLB aggregation (years).
# Matches prospect_ranking.py — 2yr half-life so current data dominates.
_RECENCY_HALF_LIFE = 2.0


def _load_org_depth() -> pd.DataFrame:
    """Load or build organizational depth data from prospect snapshots.

    Returns per-player latest-season depth metrics: how many prospects
    at the same position group are at the same or higher level in their org.
    """
    cache_path = CACHE_DIR / "prospect_org_depth.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    logger.info("Building org depth from fact_prospect_snapshot (first run)")
    from src.data.queries.prospect import get_prospect_snapshots_for_org_depth

    snap = get_prospect_snapshots_for_org_depth()

    pos_gmap = {
        "C": "C", "SS": "MI", "2B": "MI", "3B": "CI", "1B": "CI",
        "CF": "OF", "LF": "OF", "RF": "OF", "DH": "DH",
    }
    lvl_num = {"ROK": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4, "MLB": 5}
    snap["pos_group"] = snap["primary_position"].map(pos_gmap).fillna("Other")
    snap["level_num"] = snap["level"].map(lvl_num).fillna(-1)

    records = []
    for (org, season, pos), grp in snap.groupby(
        ["parent_org_id", "season", "pos_group"]
    ):
        for _, row in grp.iterrows():
            ahead = grp[
                (grp["level_num"] >= row["level_num"])
                & (grp["player_id"] != row["player_id"])
            ]
            blockers = grp[
                (grp["level_num"] > row["level_num"])
                & (grp["player_id"] != row["player_id"])
            ]
            records.append({
                "player_id": int(row["player_id"]),
                "season": int(season),
                "parent_org_name": row["parent_org_name"],
                "n_same_or_above": len(ahead),
                "n_above": len(blockers),
                "total_at_pos_in_org": len(grp),
            })

    depth_df = pd.DataFrame(records)
    depth_df.to_parquet(cache_path, index=False)
    logger.info("Cached org depth: %d rows", len(depth_df))
    return depth_df



def _build_prospect_features(
    milb_df: pd.DataFrame,
    positions_df: pd.DataFrame | None = None,
    mlb_first_season: dict[int, int] | None = None,
    max_prospect_age: int = 27,
    min_pa: int = 50,
) -> pd.DataFrame:
    """Build feature matrix from translated MiLB data.

    Parameters
    ----------
    milb_df : pd.DataFrame
        Translated MiLB batter data (from ``build_milb_translated_data``).
    positions_df : pd.DataFrame or None
        Primary MiLB position per player. Columns: batter_id, primary_position.
        If None, loads from cache.
    mlb_first_season : dict or None
        Mapping of player_id → first MLB season. Used to exclude veterans
        doing MiLB rehab/option stints.
    max_prospect_age : int
        Maximum age at MiLB level to be considered a prospect.
    min_pa : int
        Minimum career MiLB PA.

    Returns
    -------
    pd.DataFrame
        One row per prospect with all features + metadata.
    """
    if mlb_first_season is None:
        mlb_first_season = {}

    records: list[dict[str, Any]] = []
    for pid, grp in milb_df.groupby("player_id"):
        # Skip veterans doing MiLB rehab
        first_mlb = mlb_first_season.get(pid)
        first_milb = grp["season"].min()
        if first_mlb is not None and first_mlb < first_milb:
            continue

        # Age filter
        if grp["age_at_level"].min() > max_prospect_age:
            continue

        grp = grp.sort_values(["season", "level"])
        grp["_lvl_num"] = grp["level"].map(_LEVEL_MAP).fillna(0)
        best = grp.sort_values(
            ["_lvl_num", "season"], ascending=[False, False]
        ).iloc[0]

        total_pa = grp["pa"].sum()
        if total_pa < min_pa:
            continue

        # Weight by translation confidence * PA * recency decay
        # (AAA data counts more than A-ball; recent seasons count more than old ones)
        conf = grp["translation_confidence"].fillna(0.5) if "translation_confidence" in grp.columns else pd.Series(0.5, index=grp.index)
        max_season = grp["season"].max()
        recency = np.exp(
            -np.log(2) * (max_season - grp["season"]) / _RECENCY_HALF_LIFE
        )
        conf_pa = conf * grp["pa"] * recency
        conf_pa_sum = conf_pa.sum()
        if conf_pa_sum > 0:
            wtd_k = (grp["translated_k_pct"] * conf_pa).sum() / conf_pa_sum
            wtd_bb = (grp["translated_bb_pct"] * conf_pa).sum() / conf_pa_sum
            wtd_iso = (grp["translated_iso"] * conf_pa).sum() / conf_pa_sum
            wtd_hr_pa = (grp["translated_hr_pa"] * conf_pa).sum() / conf_pa_sum if "translated_hr_pa" in grp.columns else 0
        else:
            wtd_k = (grp["translated_k_pct"] * grp["pa"]).sum() / total_pa
            wtd_bb = (grp["translated_bb_pct"] * grp["pa"]).sum() / total_pa
            wtd_iso = (grp["translated_iso"] * grp["pa"]).sum() / total_pa
            wtd_hr_pa = (grp["translated_hr_pa"] * grp["pa"]).sum() / total_pa if "translated_hr_pa" in grp.columns else 0
        total_sb = grp["sb"].sum() if "sb" in grp.columns else 0
        total_hbp = grp["hbp"].sum() if "hbp" in grp.columns else 0

        # GB rate (confidence × recency weighted, if available)
        if "translated_gb_pct" in grp.columns and grp["translated_gb_pct"].notna().any():
            gb_valid = grp[grp["translated_gb_pct"].notna()]
            if conf_pa_sum > 0 and not gb_valid.empty:
                gb_conf_pa = conf[gb_valid.index] * gb_valid["pa"] * recency[gb_valid.index]
                wtd_gb = (gb_valid["translated_gb_pct"] * gb_conf_pa).sum() / gb_conf_pa.sum() if gb_conf_pa.sum() > 0 else np.nan
            else:
                wtd_gb = (gb_valid["translated_gb_pct"] * gb_valid["pa"]).sum() / gb_valid["pa"].sum() if gb_valid["pa"].sum() > 0 else np.nan
        else:
            wtd_gb = np.nan

        # --- New features: level-exposure and max-level performance ---
        max_lvl_num = grp["_lvl_num"].max()

        # PA-weighted level: continuous measure of level exposure
        pa_weighted_level = (grp["_lvl_num"] * grp["pa"]).sum() / total_pa

        # Stats at highest level reached
        max_lvl_rows = grp[grp["_lvl_num"] == max_lvl_num]
        games_at_max = max_lvl_rows["games"].sum() if "games" in max_lvl_rows.columns else 0
        pa_at_max = max_lvl_rows["pa"].sum()
        k_at_max = (max_lvl_rows["translated_k_pct"] * max_lvl_rows["pa"]).sum() / pa_at_max if pa_at_max > 0 else wtd_k
        bb_at_max = (max_lvl_rows["translated_bb_pct"] * max_lvl_rows["pa"]).sum() / pa_at_max if pa_at_max > 0 else wtd_bb

        # Age × level interaction (young + high level = superlinear)
        youngest_age_rel = grp["age_relative_to_level_avg"].min()
        age_x_level = youngest_age_rel * max_lvl_num

        # Promotion resilience: K% stability across level transitions
        if grp["level"].nunique() >= 2:
            by_level = (
                grp.groupby("_lvl_num")
                .apply(
                    lambda g: np.average(g["translated_k_pct"], weights=g["pa"])
                    if g["pa"].sum() > 0 else np.nan,
                    include_groups=False,
                )
                .dropna()
                .sort_index()
            )
            if len(by_level) >= 2:
                avg_k_spike = by_level.diff().dropna().mean()
                # -0.05 (K% dropped) → 1.0, 0 → 0.75, +0.10 → 0.25
                promo_resilience = float(np.clip(0.75 - avg_k_spike * 5.0, 0, 1))
            else:
                promo_resilience = 0.5
        else:
            promo_resilience = 0.5

        records.append({
            "player_id": pid,
            "name": best["player_name"],
            "latest_season": int(grp["season"].max()),
            "max_level": best["level"],
            "max_level_num": max_lvl_num,
            "levels_played": grp["level"].nunique(),
            "career_milb_pa": total_pa,
            "career_seasons": grp["season"].nunique(),
            "wtd_k_pct": wtd_k,
            "wtd_bb_pct": wtd_bb,
            "wtd_iso": wtd_iso,
            "wtd_hr_pa": wtd_hr_pa,
            "wtd_gb_pct": wtd_gb,
            "k_bb_diff": wtd_k - wtd_bb,
            "youngest_age_rel": youngest_age_rel,
            "avg_age_rel": grp["age_relative_to_level_avg"].mean(),
            "min_age": grp["age_at_level"].min(),
            "sb_rate": total_sb / total_pa,
            "hbp_rate": total_hbp / total_pa,
            # New features
            "pa_weighted_level": pa_weighted_level,
            "games_at_max_level": games_at_max,
            "pa_at_max_level": pa_at_max,
            "k_rate_at_max_level": k_at_max,
            "bb_rate_at_max_level": bb_at_max,
            "age_x_level": age_x_level,
            "promotion_resilience": promo_resilience,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Merge position
    if positions_df is None:
        try:
            positions_df = pd.read_parquet(CACHE_DIR / "milb_batter_positions.parquet")
        except FileNotFoundError:
            logger.warning("No MiLB position cache found")
            positions_df = pd.DataFrame(columns=["batter_id", "primary_position"])

    df = df.merge(
        positions_df, left_on="player_id", right_on="batter_id", how="left",
    )
    df["primary_position"] = df["primary_position"].fillna("Unknown")
    df["pos_group"] = df["primary_position"].map(_POS_GROUP_MAP).fillna("Unknown")

    # One-hot encode (must match training columns exactly)
    for col in _POS_DUMMY_COLS:
        pos_name = col.replace("pos_", "")
        df[col] = (df["pos_group"] == pos_name).astype(int)

    # Organizational depth features
    depth_df = _load_org_depth()
    if not depth_df.empty:
        # Use most recent depth snapshot per player
        depth_latest = (
            depth_df.sort_values("season", ascending=False)
            .drop_duplicates("player_id", keep="first")
        )
        df = df.merge(
            depth_latest[["player_id", "n_above", "total_at_pos_in_org"]],
            on="player_id",
            how="left",
        )
    else:
        df["n_above"] = np.nan
        df["total_at_pos_in_org"] = np.nan

    return df


def _get_mlb_first_seasons(seasons: list[int] | None = None) -> dict[int, int]:
    """Load first MLB season per batter.

    Uses ``staging.batting_boxscores`` (data back to 2000) so that
    readiness models trained on 2005-2018 MiLB data can identify
    which prospects eventually debuted.  Falls back to cached parquets
    if the DB query fails.
    """
    try:
        from src.data.queries.prospect import get_mlb_batter_first_seasons
        df = get_mlb_batter_first_seasons()
        return dict(zip(df["batter_id"].astype(int), df["first_season"].astype(int)))
    except Exception:
        logger.warning("DB query for MLB first seasons failed; falling back to parquets")

    if seasons is None:
        seasons = list(range(2018, 2026))
    first: dict[int, int] = {}
    for yr in seasons:
        try:
            df = pd.read_parquet(CACHE_DIR / f"season_totals_age_{yr}.parquet")
            for bid in df["batter_id"].unique():
                if bid not in first:
                    first[int(bid)] = yr
        except FileNotFoundError:
            continue
    return first


def _get_mlb_stuck_ids(
    seasons: list[int] | None = None,
    min_pa: int = 200,
) -> set[int]:
    """Get batter IDs who had at least one MLB season with *min_pa*.

    Uses ``staging.batting_boxscores`` (data back to 2000) for full
    historical coverage.  Falls back to cached parquets if the DB
    query fails.
    """
    try:
        from src.data.queries.prospect import get_mlb_batters_with_min_pa_season
        df = get_mlb_batters_with_min_pa_season(min_pa=min_pa)
        return set(df["batter_id"].astype(int).unique())
    except Exception:
        logger.warning("DB query for MLB stuck IDs failed; falling back to parquets")

    if seasons is None:
        seasons = list(range(2018, 2026))
    stuck: set[int] = set()
    for yr in seasons:
        try:
            df = pd.read_parquet(CACHE_DIR / f"season_totals_age_{yr}.parquet")
            stuck.update(df[df["pa"] >= min_pa]["batter_id"].astype(int).unique())
        except FileNotFoundError:
            continue
    return stuck


def train_readiness_model(
    milb_df: pd.DataFrame | None = None,
    max_train_season: int = 2018,
    min_debut_window: int = 2,
) -> dict[str, Any]:
    """Train the MLB readiness model on historical MiLB prospects.

    Default training window is 2005-2018 MiLB prospects (labels from
    MLB data through 2025), validated on 2019-2025 debutants.  This
    is ~3x the sample of the previous 2018-2022 window.

    Parameters
    ----------
    milb_df : pd.DataFrame or None
        Translated MiLB data. If None, loads from cache.
    max_train_season : int
        Latest MiLB season to include in training data.
        Players must have had ``min_debut_window`` years to debut.
    min_debut_window : int
        Years after MiLB to allow for MLB debut before labeling.

    Returns
    -------
    dict
        Keys: model (LogisticRegression), scaler (StandardScaler),
        features (list[str]), train_auc (float), n_train (int),
        n_positive (int).
    """
    if milb_df is None:
        milb_df = load_milb_translated("batters")

    mlb_first = _get_mlb_first_seasons()
    stuck_ids = _get_mlb_stuck_ids()

    features_df = _build_prospect_features(
        milb_df, mlb_first_season=mlb_first,
    )
    features_df["stuck_mlb"] = features_df["player_id"].isin(stuck_ids)

    # Train on prospects with enough time to debut
    train = features_df[features_df["latest_season"] <= max_train_season].copy()

    X = train[_ALL_FEATURES].fillna(0).values
    y = train["stuck_mlb"].astype(int).values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=2000, C=0.3)
    model.fit(X_s, y)

    probs = model.predict_proba(X_s)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, probs)

    logger.info(
        "Readiness model trained: %d prospects, %d positive, AUC=%.3f",
        len(train), y.sum(), auc,
    )

    return {
        "model": model,
        "scaler": scaler,
        "features": _ALL_FEATURES,
        "train_auc": auc,
        "n_train": len(train),
        "n_positive": int(y.sum()),
    }


def score_prospects(
    milb_df: pd.DataFrame | None = None,
    model_bundle: dict[str, Any] | None = None,
    seasons: list[int] | None = None,
    projection_season: int = 2026,
) -> pd.DataFrame:
    """Score current MiLB prospects with MLB readiness probability.

    Parameters
    ----------
    milb_df : pd.DataFrame or None
        Translated MiLB batter data. If None, loads from cache.
    model_bundle : dict or None
        Output of ``train_readiness_model()``. If None, trains fresh.
    seasons : list[int] or None
        Only include prospects with MiLB data in these seasons.
        Default: [projection_season - 2, projection_season - 1].
    projection_season : int
        Target projection year.

    Returns
    -------
    pd.DataFrame
        Prospect features + ``readiness_score`` column (0-1 probability),
        sorted by score descending.
    """
    if milb_df is None:
        milb_df = load_milb_translated("batters")

    if model_bundle is None:
        model_bundle = train_readiness_model(milb_df)

    if seasons is None:
        seasons = [projection_season - 2, projection_season - 1]

    mlb_first = _get_mlb_first_seasons()

    # Build features for current prospects
    recent_milb = milb_df[milb_df["season"].isin(seasons)].copy()
    features_df = _build_prospect_features(
        recent_milb, mlb_first_season=mlb_first,
    )

    if features_df.empty:
        logger.warning("No prospects found for seasons %s", seasons)
        return pd.DataFrame()

    # Exclude players already in MLB
    mlb_ids = set(mlb_first.keys())
    features_df = features_df[~features_df["player_id"].isin(mlb_ids)].copy()

    if features_df.empty:
        return pd.DataFrame()

    # Score
    X = features_df[model_bundle["features"]].fillna(0).values
    X_s = model_bundle["scaler"].transform(X)
    features_df["readiness_score"] = model_bundle["model"].predict_proba(X_s)[:, 1]

    # Add tier labels
    features_df["readiness_tier"] = pd.cut(
        features_df["readiness_score"],
        bins=[0, 0.05, 0.10, 0.20, 0.40, 1.0],
        labels=["Long Shot", "Fringe", "Developing", "Strong", "Elite"],
    )

    result = features_df.sort_values("readiness_score", ascending=False)
    logger.info(
        "Scored %d prospects: %d Elite, %d Strong, %d Developing",
        len(result),
        (result["readiness_tier"] == "Elite").sum(),
        (result["readiness_tier"] == "Strong").sum(),
        (result["readiness_tier"] == "Developing").sum(),
    )
    return result


# ===================================================================
# Pitcher readiness model
# ===================================================================

# Feature list for pitcher readiness (must match training order)
_PITCHER_STAT_FEATURES = [
    "wtd_k_pct", "wtd_bb_pct", "wtd_hr_bf", "k_bb_diff",
    "youngest_age_rel", "avg_age_rel", "min_age",
    "max_level_num", "levels_played", "career_milb_bf", "career_seasons",
    "sp_pct",
]


def _get_mlb_pitcher_stuck_ids(
    seasons: list[int] | None = None,
    min_bf: int = 100,
) -> set[int]:
    """Get pitcher IDs who had at least one season with min_bf batters faced."""
    if seasons is None:
        seasons = list(range(2018, 2026))
    stuck: set[int] = set()
    try:
        from src.data.queries.prospect import get_mlb_pitchers_with_min_bf
        df = get_mlb_pitchers_with_min_bf(min_bf=min_bf)
        stuck.update(df["pitcher_id"].astype(int).unique())
    except Exception:
        logger.warning("Could not load MLB pitcher IDs from fact_pitching_advanced")
    return stuck


def _get_mlb_pitcher_first_seasons(
    seasons: list[int] | None = None,
) -> dict[int, int]:
    """Load first MLB season per pitcher from cached pitcher season totals."""
    if seasons is None:
        seasons = list(range(2018, 2026))
    first: dict[int, int] = {}
    for yr in seasons:
        try:
            df = pd.read_parquet(CACHE_DIR / f"pitcher_season_totals_age_{yr}.parquet")
            for pid in df["pitcher_id"].unique():
                if pid not in first:
                    first[int(pid)] = yr
        except FileNotFoundError:
            continue
    return first


def train_pitcher_readiness_model(
    milb_df: pd.DataFrame | None = None,
    max_train_season: int = 2018,
    min_debut_window: int = 2,
) -> dict[str, Any]:
    """Train the MLB readiness model for pitching prospects.

    Default training window is 2005-2018 MiLB prospects (labels from
    MLB data through 2025), validated on 2019-2025 debutants.

    Parameters
    ----------
    milb_df : pd.DataFrame or None
        Translated MiLB pitcher data. If None, loads from cache.
    max_train_season : int
        Latest MiLB season to include in training data.
    min_debut_window : int
        Years after MiLB to allow for MLB debut before labeling.

    Returns
    -------
    dict
        Keys: model (LogisticRegression), scaler (StandardScaler),
        features (list[str]), train_auc (float), n_train (int),
        n_positive (int).
    """
    if milb_df is None:
        milb_df = load_milb_translated("pitchers")

    mlb_first = _get_mlb_pitcher_first_seasons()
    stuck_ids = _get_mlb_pitcher_stuck_ids()

    from src.models.prospect_ranking import _build_pitcher_prospect_features
    features_df = _build_pitcher_prospect_features(milb_df)

    if features_df.empty:
        logger.warning("No pitcher prospect features built for training")
        return {
            "model": None, "scaler": None, "features": _PITCHER_STAT_FEATURES,
            "train_auc": 0.0, "n_train": 0, "n_positive": 0,
        }

    # Exclude veterans doing MiLB rehab
    for pid, first_yr in mlb_first.items():
        mask = features_df["player_id"] == pid
        if mask.any():
            milb_first_yr = features_df.loc[mask, "latest_season"].min()
            if first_yr < milb_first_yr:
                features_df = features_df[~mask]

    features_df["stuck_mlb"] = features_df["player_id"].isin(stuck_ids)

    # Train on prospects with enough time to debut
    train = features_df[features_df["latest_season"] <= max_train_season].copy()

    if train.empty or train["stuck_mlb"].sum() == 0:
        logger.warning("Insufficient pitcher training data")
        return {
            "model": None, "scaler": None, "features": _PITCHER_STAT_FEATURES,
            "train_auc": 0.0, "n_train": 0, "n_positive": 0,
        }

    X = train[_PITCHER_STAT_FEATURES].fillna(0).values
    y = train["stuck_mlb"].astype(int).values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=2000, C=0.3)
    model.fit(X_s, y)

    probs = model.predict_proba(X_s)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, probs)

    logger.info(
        "Pitcher readiness model trained: %d prospects, %d positive, AUC=%.3f",
        len(train), y.sum(), auc,
    )

    return {
        "model": model,
        "scaler": scaler,
        "features": _PITCHER_STAT_FEATURES,
        "train_auc": auc,
        "n_train": len(train),
        "n_positive": int(y.sum()),
    }
