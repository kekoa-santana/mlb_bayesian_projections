"""
Health/durability score for playing time projections.

Combines IL stint history with biomechanical injury risk factors
from Statcast pitch-level data.

Score = weighted composite of IL history (70%) + biomechanical risk (30%):

IL history sub-components:
  - Total IL days (40%): recency-weighted average days per season
  - Stint frequency (25%): recency-weighted average stints per season
  - Seasons breadth (20%): fraction of lookback seasons with any IL
  - Recency (15%): IL days in the most recent completed season

Biomechanical risk factors (logistic regression):
  - P95 velocity, breaking ball usage %, FB velocity change YoY,
    workload, arm angle, extension variability, prior arm injury

Scores range from 0.0 (worst) to 1.0 (best/healthy).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.db import read_sql
from src.data.paths import CACHE_DIR

logger = logging.getLogger(__name__)


# Recency weights: most recent season gets highest weight
RECENCY_WEIGHTS = [5, 4, 3, 2, 1]
LOOKBACK_YEARS = 5

# Top-level blend: IL history vs biomechanical risk
_IL_HISTORY_WEIGHT = 0.70
_BIO_RISK_WEIGHT = 0.30

# IL history sub-component weights (within the IL portion)
_DAYS_WEIGHT = 0.40
_FREQ_WEIGHT = 0.25
_BREADTH_WEIGHT = 0.20
_RECENCY_WEIGHT = 0.15

# Normalization caps (per-season weighted-average basis)
_DAYS_CAP = 100.0       # weighted avg IL days per season → floor at 100+
_STINTS_CAP = 2.0       # weighted avg stints per season → floor at 2+
_RECENCY_CAP = 100.0    # IL days in most recent season → floor at 100+

# 2020 shortened season scaling
_2020_SCALE = 162.0 / 60.0

# Defaults for players without sufficient IL data
DEFAULT_NO_IL_HISTORY = 0.85   # player exists but no IL stints found
DEFAULT_TRUE_ROOKIE = 0.80     # no MLB history at all

# Label thresholds
HEALTH_LABELS = [
    (0.85, "Iron Man"),
    (0.70, "Durable"),
    (0.50, "Average"),
    (0.30, "Questionable"),
    (0.00, "Injury Prone"),
]

# IL status types in the database
IL_STATUS_TYPES = ("IL-7", "IL-10", "IL-15", "IL-60")


def _get_health_label(score: float) -> str:
    """Map health score to descriptive label."""
    for threshold, label in HEALTH_LABELS:
        if score >= threshold:
            return label
    return "Injury Prone"


def _query_il_history(
    from_season: int,
    player_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Query IL stint history from fact_player_status_timeline.

    Parameters
    ----------
    from_season : int
        Most recent completed season (lookback window ends here).
    player_ids : list[int], optional
        Restrict to these player IDs. If None, query all.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, season, days_on_status, status_type.
    """
    start_season = from_season - LOOKBACK_YEARS + 1
    status_list = ", ".join(f"'{s}'" for s in IL_STATUS_TYPES)

    pid_filter = ""
    if player_ids is not None and len(player_ids) > 0:
        pid_str = ", ".join(str(int(p)) for p in player_ids)
        pid_filter = f"AND player_id IN ({pid_str})"

    query = f"""
        SELECT player_id,
               season,
               COALESCE(days_on_status, 0) AS days_on_status,
               status_type
        FROM production.fact_player_status_timeline
        WHERE status_type IN ({status_list})
          AND season BETWEEN {start_season} AND {from_season}
          {pid_filter}
        ORDER BY player_id, season, status_start_date
    """
    return read_sql(query, {})


# ---------------------------------------------------------------------------
# Biomechanical injury risk
# ---------------------------------------------------------------------------

# Features used by the risk model (must match training order)
_BIO_FEATURES = [
    "fb_velo", "brk_velo", "p95_velo", "slider_pct", "brk_pct",
    "ch_pct", "total_pitches", "any_prior_arm", "arm_angle_proxy",
    "fb_velo_delta", "avg_velo_delta", "avg_extension", "extension_sd",
]

# Pre-fit model coefficients from walk-forward validated logistic regression
# (trained on 2019-2023 pitcher-seasons, AUC=0.556, Q4/Q1 lift=1.37x)
_BIO_COEFS = np.array([
    -0.066670,  # fb_velo
    +0.053653,  # brk_velo
    +0.193859,  # p95_velo
    -0.028534,  # slider_pct
    +0.173431,  # brk_pct
    +0.046850,  # ch_pct
    +0.107509,  # total_pitches
    +0.094155,  # any_prior_arm
    +0.095067,  # arm_angle_proxy
    -0.123162,  # fb_velo_delta
    -0.000159,  # avg_velo_delta
    +0.034720,  # avg_extension
    -0.175915,  # extension_sd
])
_BIO_INTERCEPT = -0.031066

# StandardScaler params fit on 2019-2023 pitcher-seasons (n=2024)
_BIO_MEANS = np.array([
    93.1430, 82.4536, 94.9647, 0.2073, 0.3017,
    0.1266, 1300.7411, 0.0583, 26.2054,
    -0.0622, -0.0927, 6.2488, 0.2050,
])
_BIO_SCALES = np.array([
    2.6226, 3.5583, 2.5056, 0.1603, 0.1459,
    0.1118, 728.7077, 0.2343, 17.0207,
    0.8180, 0.9294, 0.3915, 0.0603,
])


def _query_bio_features(from_season: int) -> pd.DataFrame:
    """Query pitcher biomechanical features for the given season.

    Returns one row per pitcher with velocity, usage, extension, and
    arm angle features. Delta features use prior season comparison.
    """
    query = f"""
        WITH current AS (
            SELECT fp.pitcher_id,
                   AVG(fp.release_speed) as avg_velo,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fp.release_speed) as p95_velo,
                   AVG(CASE WHEN fp.pitch_type IN ('FF','SI','FC') THEN fp.release_speed END) as fb_velo,
                   AVG(CASE WHEN fp.pitch_type IN ('SL','ST','SV','CU','KC') THEN fp.release_speed END) as brk_velo,
                   COUNT(CASE WHEN fp.pitch_type IN ('SL','ST','SV') THEN 1 END)::float / COUNT(*) as slider_pct,
                   COUNT(CASE WHEN fp.pitch_type IN ('SL','ST','SV','CU','KC') THEN 1 END)::float / COUNT(*) as brk_pct,
                   COUNT(CASE WHEN fp.pitch_type IN ('CH','FS') THEN 1 END)::float / COUNT(*) as ch_pct,
                   AVG(sps.release_extension) as avg_extension,
                   STDDEV(sps.release_extension) as extension_sd,
                   AVG(sps.release_pos_x) as avg_rel_x,
                   AVG(sps.release_pos_z) as avg_rel_z,
                   COUNT(*) as total_pitches
            FROM production.fact_pitch fp
            JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
            LEFT JOIN production.sat_pitch_shape sps ON fp.pitch_id = sps.pitch_id
            WHERE dg.game_type = 'R' AND dg.season = {from_season}
                  AND fp.release_speed IS NOT NULL
                  AND CAST(fp.release_speed AS TEXT) != 'NaN'
            GROUP BY fp.pitcher_id
            HAVING COUNT(*) >= 200
        ),
        prior AS (
            SELECT fp.pitcher_id,
                   AVG(fp.release_speed) as avg_velo_prior,
                   AVG(CASE WHEN fp.pitch_type IN ('FF','SI','FC') THEN fp.release_speed END) as fb_velo_prior
            FROM production.fact_pitch fp
            JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
            WHERE dg.game_type = 'R' AND dg.season = {from_season - 1}
                  AND fp.release_speed IS NOT NULL
                  AND CAST(fp.release_speed AS TEXT) != 'NaN'
            GROUP BY fp.pitcher_id
            HAVING COUNT(*) >= 200
        )
        SELECT c.*,
               c.fb_velo - p.fb_velo_prior as fb_velo_delta,
               c.avg_velo - p.avg_velo_prior as avg_velo_delta
        FROM current c
        LEFT JOIN prior p ON c.pitcher_id = p.pitcher_id
    """
    return read_sql(query, {})


def compute_biomechanical_risk(
    from_season: int,
) -> pd.DataFrame:
    """Compute biomechanical injury risk scores for pitchers.

    Uses a pre-fit logistic regression model trained on 2019-2023
    pitcher-season data predicting next-season arm injury from
    Statcast pitch-tracking features.

    Parameters
    ----------
    from_season : int
        Season to compute features from.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, bio_risk (0-1, higher = more risk),
        bio_health (0-1, higher = healthier, = 1 - bio_risk).
    """
    from scipy.special import expit

    bio = _query_bio_features(from_season)
    if bio.empty:
        logger.warning("No biomechanical data for season %d", from_season)
        return pd.DataFrame(columns=["pitcher_id", "bio_risk", "bio_health"])

    # Compute arm angle proxy
    bio["arm_angle_proxy"] = np.degrees(np.arctan2(
        bio["avg_rel_z"].fillna(5.5) - 5.0,
        np.abs(bio["avg_rel_x"].fillna(2.0)),
    ))

    # Get prior arm injury flag
    il_query = f"""
        SELECT DISTINCT player_id as pitcher_id, 1 as any_prior_arm
        FROM production.fact_player_status_timeline
        WHERE status_type LIKE 'IL%%'
          AND season BETWEEN {from_season - 3} AND {from_season}
          AND (LOWER(injury_description) LIKE '%%elbow%%'
               OR LOWER(injury_description) LIKE '%%shoulder%%'
               OR LOWER(injury_description) LIKE '%%ucl%%'
               OR LOWER(injury_description) LIKE '%%forearm%%'
               OR LOWER(injury_description) LIKE '%%flexor%%'
               OR LOWER(injury_description) LIKE '%%lat%%')
    """
    prior_il = read_sql(il_query, {})
    bio = bio.merge(prior_il, on="pitcher_id", how="left")
    bio["any_prior_arm"] = bio["any_prior_arm"].fillna(0)

    # Fill missing values
    bio["fb_velo_delta"] = bio["fb_velo_delta"].fillna(0)
    bio["avg_velo_delta"] = bio["avg_velo_delta"].fillna(0)
    bio["avg_extension"] = bio["avg_extension"].fillna(_BIO_MEANS[11])
    bio["extension_sd"] = bio["extension_sd"].fillna(_BIO_MEANS[12])

    # Build feature matrix
    X = np.column_stack([bio[f].values for f in _BIO_FEATURES])

    # Standardize using pre-fit scaler
    X_scaled = (X - _BIO_MEANS) / np.where(_BIO_SCALES > 0, _BIO_SCALES, 1.0)

    # Predict risk
    logits = X_scaled @ _BIO_COEFS + _BIO_INTERCEPT
    bio_risk = expit(logits)

    result = pd.DataFrame({
        "pitcher_id": bio["pitcher_id"].values,
        "bio_risk": bio_risk,
        "bio_health": 1.0 - bio_risk,
    })

    logger.info(
        "Biomechanical risk: %d pitchers, mean risk=%.3f, range=[%.3f, %.3f]",
        len(result), result["bio_risk"].mean(),
        result["bio_risk"].min(), result["bio_risk"].max(),
    )
    return result


def compute_health_scores(
    from_season: int,
    all_player_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Compute health/durability scores for all players.

    Parameters
    ----------
    from_season : int
        Most recent completed season. Lookback window is
        [from_season - 4, from_season] (5 seasons).
    all_player_ids : list[int], optional
        Player IDs to score. If None, scores all players with IL data
        plus returns defaults for those without.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, health_score, health_label,
        days_score, freq_score, breadth_score, recency_score,
        total_il_days, total_stints, seasons_with_il.
    """
    cache_path = CACHE_DIR / f"health_scores_{from_season}.parquet"
    if cache_path.exists():
        logger.info("Loading cached health scores from %s", cache_path)
        return pd.read_parquet(cache_path)

    start_season = from_season - LOOKBACK_YEARS + 1
    lookback_seasons = list(range(start_season, from_season + 1))

    # Query IL history
    il_df = _query_il_history(from_season, all_player_ids)
    logger.info(
        "IL history: %d stints for %d players (seasons %d-%d)",
        len(il_df),
        il_df["player_id"].nunique() if not il_df.empty else 0,
        start_season,
        from_season,
    )

    # Build recency weight lookup: most recent season = highest weight
    season_weights = {}
    for i, s in enumerate(sorted(lookback_seasons)):
        season_weights[s] = RECENCY_WEIGHTS[LOOKBACK_YEARS - 1 - i]
    # Result: oldest season = weight 1, most recent = weight 5

    # Aggregate per player-season
    if not il_df.empty:
        # Scale 2020 days
        il_df["adj_days"] = il_df["days_on_status"].copy()
        mask_2020 = il_df["season"] == 2020
        il_df.loc[mask_2020, "adj_days"] = (
            il_df.loc[mask_2020, "days_on_status"] * _2020_SCALE
        ).clip(upper=182)  # cap at full season + some

        per_season = (
            il_df.groupby(["player_id", "season"])
            .agg(
                season_days=("adj_days", "sum"),
                season_stints=("player_id", "count"),
            )
            .reset_index()
        )
    else:
        per_season = pd.DataFrame(
            columns=["player_id", "season", "season_days", "season_stints"]
        )

    # Compute scores per player
    total_weight = sum(RECENCY_WEIGHTS[:LOOKBACK_YEARS])
    players_with_il = set(per_season["player_id"].unique()) if not per_season.empty else set()

    # Determine all player IDs to score
    if all_player_ids is not None:
        score_ids = set(int(p) for p in all_player_ids)
    else:
        score_ids = players_with_il

    results = []
    for pid in score_ids:
        player_seasons = per_season[per_season["player_id"] == pid]

        if player_seasons.empty and pid not in players_with_il:
            # No IL data at all — use default
            results.append({
                "player_id": int(pid),
                "il_health": DEFAULT_NO_IL_HISTORY,
                "days_score": DEFAULT_NO_IL_HISTORY,
                "freq_score": DEFAULT_NO_IL_HISTORY,
                "breadth_score": DEFAULT_NO_IL_HISTORY,
                "recency_score": DEFAULT_NO_IL_HISTORY,
                "total_il_days": 0,
                "total_stints": 0,
                "seasons_with_il": 0,
            })
            continue

        # Build per-season data (fill missing seasons with 0)
        season_data = {}
        for s in lookback_seasons:
            row = player_seasons[player_seasons["season"] == s]
            if not row.empty:
                season_data[s] = {
                    "days": float(row["season_days"].sum()),
                    "stints": int(row["season_stints"].sum()),
                }
            else:
                season_data[s] = {"days": 0.0, "stints": 0}

        # Sub-component 1: Total IL days (recency-weighted average)
        weighted_days = sum(
            season_data[s]["days"] * season_weights[s] for s in lookback_seasons
        )
        weighted_avg_days = weighted_days / total_weight
        days_score = max(1.0 - weighted_avg_days / _DAYS_CAP, 0.0)

        # Sub-component 2: Stint frequency (recency-weighted average)
        weighted_stints = sum(
            season_data[s]["stints"] * season_weights[s] for s in lookback_seasons
        )
        weighted_avg_stints = weighted_stints / total_weight
        freq_score = max(1.0 - weighted_avg_stints / _STINTS_CAP, 0.0)

        # Sub-component 3: Seasons breadth (fraction with any IL)
        n_seasons_with_il = sum(
            1 for s in lookback_seasons if season_data[s]["days"] > 0
        )
        breadth_score = 1.0 - n_seasons_with_il / LOOKBACK_YEARS

        # Sub-component 4: Recency (most recent season severity)
        recent_days = season_data[from_season]["days"]
        recency_score = max(1.0 - recent_days / _RECENCY_CAP, 0.0)

        # IL-only combined score
        il_health = (
            _DAYS_WEIGHT * days_score
            + _FREQ_WEIGHT * freq_score
            + _BREADTH_WEIGHT * breadth_score
            + _RECENCY_WEIGHT * recency_score
        )

        total_il = sum(season_data[s]["days"] for s in lookback_seasons)
        total_st = sum(season_data[s]["stints"] for s in lookback_seasons)

        results.append({
            "player_id": int(pid),
            "il_health": round(il_health, 4),
            "days_score": round(days_score, 4),
            "freq_score": round(freq_score, 4),
            "breadth_score": round(breadth_score, 4),
            "recency_score": round(recency_score, 4),
            "total_il_days": int(total_il),
            "total_stints": int(total_st),
            "seasons_with_il": int(n_seasons_with_il),
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    # --- Blend with biomechanical risk ---
    try:
        bio_risk = compute_biomechanical_risk(from_season)
        if not bio_risk.empty:
            result_df = result_df.merge(
                bio_risk[["pitcher_id", "bio_health"]].rename(
                    columns={"pitcher_id": "player_id"}
                ),
                on="player_id", how="left",
            )
            # For pitchers: blend IL history + biomechanics
            has_bio = result_df["bio_health"].notna()
            result_df.loc[has_bio, "health_score"] = (
                _IL_HISTORY_WEIGHT * result_df.loc[has_bio, "il_health"]
                + _BIO_RISK_WEIGHT * result_df.loc[has_bio, "bio_health"]
            )
            # For non-pitchers (hitters): IL-only
            result_df.loc[~has_bio, "health_score"] = result_df.loc[~has_bio, "il_health"]
            logger.info(
                "Blended bio risk for %d pitchers (mean bio_health=%.3f)",
                has_bio.sum(), result_df.loc[has_bio, "bio_health"].mean(),
            )
        else:
            result_df["health_score"] = result_df["il_health"]
            result_df["bio_health"] = np.nan
    except Exception as e:
        logger.warning("Bio risk computation failed, using IL-only: %s", e)
        result_df["health_score"] = result_df["il_health"]
        result_df["bio_health"] = np.nan

    result_df["health_label"] = result_df["health_score"].apply(_get_health_label)

    logger.info(
        "Health scores: %d players, mean=%.3f, "
        "Iron Man=%d, Durable=%d, Average=%d, Questionable=%d, Injury Prone=%d",
        len(result_df),
        result_df["health_score"].mean(),
        (result_df["health_label"] == "Iron Man").sum(),
        (result_df["health_label"] == "Durable").sum(),
        (result_df["health_label"] == "Average").sum(),
        (result_df["health_label"] == "Questionable").sum(),
        (result_df["health_label"] == "Injury Prone").sum(),
    )

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cache_path, index=False)
    logger.info("Cached health scores to %s", cache_path)

    return result_df
