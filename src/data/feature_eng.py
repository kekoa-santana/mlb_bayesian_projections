"""
Feature engineering for hitter vulnerability / strength profiles and pitcher
arsenal profiles. Results are cached as Parquet files to avoid repeated
expensive pitch-level queries.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.queries import (
    get_batter_game_logs,
    get_game_batter_ks,
    get_game_batter_stats,
    get_game_lineups,
    get_hitter_observed_profile,
    get_hitter_pitch_type_profile,
    get_hitter_season_totals_extended,
    get_pitcher_arsenal_profile,
    get_pitcher_game_logs,
    get_pitcher_observed_profile,
    get_pitcher_season_totals,
    get_pitcher_season_totals_with_age,
    get_pitcher_season_totals_extended,
    get_season_totals,
    get_season_totals_by_pitcher_hand,
    get_season_totals_with_age,
    get_sprint_speed,
)
from src.utils.constants import (
    EXCLUDED_PITCH_TYPES,
    LEAGUE_AVG_BY_PITCH_TYPE,
    LEAGUE_AVG_OVERALL,
    PITCH_TO_FAMILY,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"


# ---------------------------------------------------------------------------
# Parquet cache helpers
# ---------------------------------------------------------------------------
def _cache_path(name: str, season: int) -> Path:
    return CACHE_DIR / f"{name}_{season}.parquet"


def _load_or_build(
    name: str,
    season: int,
    builder: callable,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load from Parquet cache if available, otherwise build and cache.

    Parameters
    ----------
    name : str
        Cache file prefix (e.g. "hitter_vuln").
    season : int
        MLB season year.
    builder : callable
        Function that returns a DataFrame when called with (season,).
    force_rebuild : bool
        If True, ignore cache and rebuild.

    Returns
    -------
    pd.DataFrame
    """
    path = _cache_path(name, season)
    if path.exists() and not force_rebuild:
        logger.info("Loading cached %s for %d", name, season)
        return pd.read_parquet(path)

    logger.info("Building %s for %d (no cache or force_rebuild)", name, season)
    df = builder(season)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Cached %s to %s (%d rows)", name, path, len(df))
    return df


# ---------------------------------------------------------------------------
# Hitter vulnerability profile
# ---------------------------------------------------------------------------
def _build_hitter_vulnerability(season: int) -> pd.DataFrame:
    """Compute per-batter, per-pitch-type vulnerability rates.

    Metrics
    -------
    - whiff_rate = whiffs / swings
    - chase_rate = chase_swings / out_of_zone_pitches
    - csw_pct = csw / pitches
    - pitch_family (for hierarchical pooling)
    """
    df = get_hitter_pitch_type_profile(season)
    df = df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)].copy()

    df["whiff_rate"] = df["whiffs"] / df["swings"].replace(0, np.nan)
    df["chase_rate"] = df["chase_swings"] / df["out_of_zone_pitches"].replace(0, np.nan)
    df["csw_pct"] = df["csw"] / df["pitches"].replace(0, np.nan)
    df["pitch_family"] = df["pitch_type"].map(PITCH_TO_FAMILY)
    df["season"] = season

    return df


def get_hitter_vulnerability(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Hitter vulnerability profiles with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per (batter_id, pitch_type) with whiff/chase/csw rates.
    """
    return _load_or_build("hitter_vuln", season, _build_hitter_vulnerability, force_rebuild)


# ---------------------------------------------------------------------------
# Hitter vulnerability by pitch archetype
# ---------------------------------------------------------------------------
def _build_hitter_vulnerability_by_archetype(season: int) -> pd.DataFrame:
    """Compute per-batter, per-pitch-archetype vulnerability rates.

    Joins hitter pitch-type profiles with archetype assignments and
    aggregates whiff/chase/csw rates by (batter_id, pitch_archetype).
    """
    from src.data.pitch_archetypes import get_pitch_archetype_offerings

    vuln = _build_hitter_vulnerability(season)
    offerings = get_pitch_archetype_offerings(season)

    # Build pitch_type -> archetype mapping from pitcher offerings.
    # Multiple pitcher/pitch_type combos may map to different archetypes;
    # use the most common archetype per pitch_type (volume-weighted).
    pt_arch = (
        offerings.groupby(["pitch_type", "pitch_archetype"], as_index=False)
        .agg(total_pitches=("pitches", "sum"))
        .sort_values(
            ["pitch_type", "total_pitches"], ascending=[True, False]
        )
        .drop_duplicates(subset=["pitch_type"])
        [["pitch_type", "pitch_archetype"]]
    )

    merged = vuln.merge(pt_arch, on="pitch_type", how="left")
    merged = merged.dropna(subset=["pitch_archetype"])
    merged["pitch_archetype"] = merged["pitch_archetype"].astype(int)

    # Aggregate to (batter_id, pitch_archetype)
    agg = merged.groupby(["batter_id", "pitch_archetype"], as_index=False).agg(
        pitches=("pitches", "sum"),
        swings=("swings", "sum"),
        whiffs=("whiffs", "sum"),
        out_of_zone_pitches=("out_of_zone_pitches", "sum"),
        chase_swings=("chase_swings", "sum"),
        csw=("csw", "sum"),
    )

    agg["whiff_rate"] = agg["whiffs"] / agg["swings"].replace(0, np.nan)
    agg["chase_rate"] = agg["chase_swings"] / agg["out_of_zone_pitches"].replace(0, np.nan)
    agg["csw_pct"] = agg["csw"] / agg["pitches"].replace(0, np.nan)
    agg["season"] = season

    return agg


def get_hitter_vulnerability_by_archetype(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Hitter vulnerability by pitch archetype with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per (batter_id, pitch_archetype) with whiff/chase/csw rates.
    """
    return _load_or_build(
        "hitter_vuln_arch", season, _build_hitter_vulnerability_by_archetype, force_rebuild
    )


# ---------------------------------------------------------------------------
# Hitter strength profile
# ---------------------------------------------------------------------------
def _build_hitter_strength(season: int) -> pd.DataFrame:
    """Compute per-batter, per-pitch-type strength metrics.

    Metrics
    -------
    - barrel_rate_contact = barrels_proxy / bip
    - xwoba_contact (from query)
    - hard_hit_rate = hard_hits / bip
    """
    df = get_hitter_pitch_type_profile(season)
    df = df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)].copy()

    df["barrel_rate_contact"] = df["barrels_proxy"] / df["bip"].replace(0, np.nan)
    df["hard_hit_rate"] = df["hard_hits"] / df["bip"].replace(0, np.nan)
    df["pitch_family"] = df["pitch_type"].map(PITCH_TO_FAMILY)
    df["season"] = season

    # Keep only strength-relevant columns
    cols = [
        "batter_id", "batter_stand", "pitch_type", "pitch_family", "season",
        "bip", "barrels_proxy", "barrel_rate_contact",
        "xwoba_contact", "hard_hits", "hard_hit_rate",
    ]
    return df[cols]


def get_hitter_strength(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Hitter strength profiles with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per (batter_id, pitch_type) with barrel/xwoba/hard-hit rates.
    """
    return _load_or_build("hitter_str", season, _build_hitter_strength, force_rebuild)


# ---------------------------------------------------------------------------
# Pitcher arsenal profile (cached)
# ---------------------------------------------------------------------------
def _build_pitcher_arsenal(season: int) -> pd.DataFrame:
    """Add derived rates to the raw pitcher arsenal query."""
    df = get_pitcher_arsenal_profile(season)
    df = df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)].copy()

    df["whiff_rate"] = df["whiffs"] / df["swings"].replace(0, np.nan)
    df["csw_pct"] = df["csw"] / df["pitches"].replace(0, np.nan) if "csw" in df.columns else np.nan
    df["barrel_rate_against"] = df["barrels_proxy"] / df["bip"].replace(0, np.nan)
    df["hard_hit_rate_against"] = df["hard_hits"] / df["bip"].replace(0, np.nan)
    df["pitch_family"] = df["pitch_type"].map(PITCH_TO_FAMILY)
    df["season"] = season

    return df


def get_pitcher_arsenal(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Pitcher arsenal profiles with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per (pitcher_id, pitch_type) with usage, whiff, barrel rates.
    """
    return _load_or_build("pitcher_arsenal", season, _build_pitcher_arsenal, force_rebuild)


# ---------------------------------------------------------------------------
# Season totals (cached)
# ---------------------------------------------------------------------------
def get_cached_season_totals(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Season totals with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per batter with season-level slash stats.
    """
    return _load_or_build("season_totals", season, get_season_totals, force_rebuild)


# ---------------------------------------------------------------------------
# Pitcher season totals (cached)
# ---------------------------------------------------------------------------
def get_cached_pitcher_season_totals(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Pitcher season totals with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per pitcher with season-level K/BB/HR/IP stats.
    """
    return _load_or_build(
        "pitcher_season_totals", season, get_pitcher_season_totals, force_rebuild
    )


# ---------------------------------------------------------------------------
# Multi-season player data for Bayesian model
# ---------------------------------------------------------------------------
def build_multi_season_k_data(
    seasons: list[int], min_pa: int = 1
) -> pd.DataFrame:
    """Stack hitter season totals across multiple seasons for the K% model.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_pa : int
        Minimum PA per player-season (no hard cutoff per CLAUDE.md — the
        Bayesian model handles small samples, but this filters out ghost
        rows with 0 PA).

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, batter_stand, season, pa, k,
        k_rate, plus Statcast quality metrics.
    """
    frames = []
    for s in seasons:
        df = get_cached_season_totals(s, force_rebuild=False)
        if min_pa > 1:
            df = df[df["pa"] >= min_pa]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Multi-season K data: %d player-seasons across %s",
        len(combined), seasons,
    )
    return combined


# ---------------------------------------------------------------------------
# Season totals by pitcher hand (cached)
# ---------------------------------------------------------------------------
def get_cached_season_totals_by_pitcher_hand(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Season totals split by pitcher hand, with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per (batter_id, pitch_hand) with season-level stats
        and ``same_side`` flag.
    """
    return _load_or_build(
        "season_totals_by_hand", season,
        get_season_totals_by_pitcher_hand, force_rebuild,
    )


# ---------------------------------------------------------------------------
# Multi-season platoon data for Bayesian model
# ---------------------------------------------------------------------------
def build_multi_season_k_data_platoon(
    seasons: list[int], min_pa: int = 1
) -> pd.DataFrame:
    """Stack platoon-split season totals across multiple seasons.

    Returns ~2x rows vs ``build_multi_season_k_data`` (two rows per
    player-season: one vs LHP, one vs RHP).

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_pa : int
        Minimum PA per player-season-hand row.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, batter_stand, pitch_hand,
        season, pa, k, k_rate, same_side, plus Statcast quality metrics.
    """
    frames = []
    for s in seasons:
        df = get_cached_season_totals_by_pitcher_hand(s, force_rebuild=False)
        if min_pa > 1:
            df = df[df["pa"] >= min_pa]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Multi-season platoon K data: %d player-season-hand rows across %s",
        len(combined), seasons,
    )
    return combined


# ---------------------------------------------------------------------------
# Multi-season pitcher data for Bayesian model
# ---------------------------------------------------------------------------
def build_multi_season_pitcher_k_data(
    seasons: list[int], min_bf: int = 1
) -> pd.DataFrame:
    """Stack pitcher season totals with arsenal-derived covariates.

    Per season:
    1. Load cached pitcher season totals (K/BF/IP from boxscores).
    2. Load cached pitcher arsenal, aggregate to pitcher-level:
       ``whiff_rate = sum(whiffs)/sum(swings)``,
       ``barrel_rate_against = sum(barrels_proxy)/sum(bip)``.
    3. Left-merge on pitcher_id.
    4. Filter by min_bf and stack across seasons.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_bf : int
        Minimum batters faced per player-season.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season,
        batters_faced, k, k_rate, bb, bb_rate, ip, games,
        whiff_rate, barrel_rate_against, is_starter.
    """
    frames = []
    for s in seasons:
        totals = get_cached_pitcher_season_totals(s, force_rebuild=False)
        arsenal = get_pitcher_arsenal(s, force_rebuild=False)

        # Aggregate arsenal from per-pitch-type to pitcher-level
        pitcher_agg = arsenal.groupby("pitcher_id").agg(
            whiffs=("whiffs", "sum"),
            swings=("swings", "sum"),
            barrels_proxy=("barrels_proxy", "sum"),
            bip=("bip", "sum"),
        ).reset_index()
        pitcher_agg["whiff_rate"] = (
            pitcher_agg["whiffs"] / pitcher_agg["swings"].replace(0, np.nan)
        )
        pitcher_agg["barrel_rate_against"] = (
            pitcher_agg["barrels_proxy"] / pitcher_agg["bip"].replace(0, np.nan)
        )

        merged = totals.merge(
            pitcher_agg[["pitcher_id", "whiff_rate", "barrel_rate_against"]],
            on="pitcher_id",
            how="left",
        )

        # Derive starter/reliever role: IP/game >= 3.0 → starter
        merged["is_starter"] = (
            (merged["ip"] / merged["games"].replace(0, np.nan)) >= 3.0
        ).astype(int).fillna(0).astype(int)

        if min_bf > 1:
            merged = merged[merged["batters_faced"] >= min_bf]

        keep_cols = [
            "pitcher_id", "pitcher_name", "pitch_hand", "season",
            "batters_faced", "k", "k_rate", "bb", "bb_rate", "ip", "games",
            "whiff_rate", "barrel_rate_against", "is_starter",
        ]
        frames.append(merged[keep_cols])

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Multi-season pitcher K data: %d player-seasons across %s",
        len(combined), seasons,
    )
    return combined


# ---------------------------------------------------------------------------
# Pitcher game logs (cached)
# ---------------------------------------------------------------------------
def get_cached_pitcher_game_logs(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Pitcher game logs with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per (game_pk, pitcher_id) with K, BF, IP, is_starter.
    """
    return _load_or_build(
        "pitcher_game_logs", season, get_pitcher_game_logs, force_rebuild
    )


# ---------------------------------------------------------------------------
# Game-level batter Ks (cached)
# ---------------------------------------------------------------------------
def get_cached_game_batter_ks(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Per (game, pitcher, batter) PA and K counts with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, pitcher_id, batter_id, pa, k.
    """
    return _load_or_build(
        "game_batter_ks", season, get_game_batter_ks, force_rebuild
    )


# ---------------------------------------------------------------------------
# Game lineups (cached)
# ---------------------------------------------------------------------------
def get_cached_game_lineups(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Starting lineups (batting order 1-9) with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        Columns: game_pk, player_id, batting_order, team_id, batter_name.
    """
    return _load_or_build(
        "game_lineups", season, get_game_lineups, force_rebuild
    )


# ---------------------------------------------------------------------------
# Season totals with age (cached)
# ---------------------------------------------------------------------------
def get_cached_season_totals_with_age(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Season totals with age and age_bucket, with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per batter with season stats + age + age_bucket.
    """
    return _load_or_build(
        "season_totals_age", season, get_season_totals_with_age, force_rebuild
    )


# ---------------------------------------------------------------------------
# Multi-season hitter data with age for expanded projection models
# ---------------------------------------------------------------------------
def build_multi_season_hitter_data(
    seasons: list[int], min_pa: int = 1
) -> pd.DataFrame:
    """Stack hitter season totals with age across multiple seasons.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_pa : int
        Minimum PA per player-season.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, batter_stand, season, age,
        age_bucket, pa, k, bb, ibb, hbp, sf, hits, hr, xwoba_avg,
        barrel_pct, hard_hit_pct, k_rate, bb_rate, hr_rate, woba.
    """
    frames = []
    for s in seasons:
        df = get_cached_season_totals_with_age(s, force_rebuild=False)
        if min_pa > 1:
            df = df[df["pa"] >= min_pa]

        # Merge approach metrics (whiff_rate, chase_rate, z_contact_pct, fb_pct)
        # and chase raw counts (chase_swings, out_of_zone_pitches) for Bayesian chase_rate model
        try:
            obs = get_cached_hitter_observed_profile(s)
            merge_cols = ["batter_id"]
            for col in ["whiff_rate", "chase_rate", "z_contact_pct", "fb_pct",
                         "chase_swings", "out_of_zone_pitches", "avg_exit_velo"]:
                if col in obs.columns:
                    merge_cols.append(col)
            df = df.merge(obs[merge_cols], on="batter_id", how="left")
        except Exception:
            logger.warning("No hitter observed profile for %d, skipping merge", s)
            for col in ["whiff_rate", "chase_rate", "z_contact_pct", "fb_pct",
                         "chase_swings", "out_of_zone_pitches", "avg_exit_velo"]:
                if col not in df.columns:
                    df[col] = np.nan

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with missing age (shouldn't happen but be safe)
    n_before = len(combined)
    combined = combined.dropna(subset=["age", "age_bucket"])
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with missing age", n_dropped)

    combined["age_bucket"] = combined["age_bucket"].astype(int)

    # Power composite — single covariate for HR/FB model that combines
    # ISO + barrel% + hard_hit% + exit_velo into one stable predictor.
    # Validated: predicts next-year barrel% 54% better than barrel% alone,
    # next-year ISO 9% better (grade_prior_analysis.py, 2019-2024).
    # Z-scored within the combined data so it's season-normalized.
    _pow_cols = ["iso", "barrel_pct", "hard_hit_pct", "avg_exit_velo"]
    _pow_weights = [0.35, 0.25, 0.20, 0.20]
    available_pow = [(c, w) for c, w in zip(_pow_cols, _pow_weights) if c in combined.columns]
    if available_pow:
        power_z = np.zeros(len(combined))
        total_w = 0.0
        for col, w in available_pow:
            vals = combined[col].astype(float)
            mu, sd = vals.mean(), vals.std()
            if sd > 1e-9:
                power_z += w * (vals.fillna(mu) - mu) / sd
                total_w += w
        if total_w > 0:
            combined["power_composite"] = power_z / total_w
        else:
            combined["power_composite"] = 0.0
        logger.info("Power composite computed from %d metrics", len(available_pow))
    else:
        combined["power_composite"] = 0.0

    combined = assign_skill_tier(combined, player_type="hitter")

    logger.info(
        "Multi-season hitter data: %d player-seasons across %s",
        len(combined), seasons,
    )
    return combined


# ---------------------------------------------------------------------------
# Statcast skill tier assignment
# ---------------------------------------------------------------------------
N_SKILL_TIERS = 4
SKILL_TIER_LABELS = {
    0: "below-avg",
    1: "average",
    2: "above-avg",
    3: "elite",
}


def assign_skill_tier(
    df: pd.DataFrame,
    player_type: str = "hitter",
) -> pd.DataFrame:
    """Assign Statcast-based skill tiers to each player-season.

    Computes a composite skill score from available Statcast indicators
    and assigns quartile-based tiers (0=below-avg through 3=elite).

    For hitters: barrel_pct + hard_hit_pct (contact quality).
    For pitchers: whiff_rate + (1 - barrel_rate_against) (stuff quality).

    Tiers are computed per season to avoid leaking cross-season info.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-season player data with Statcast columns.
    player_type : str
        "hitter" or "pitcher".

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``skill_tier`` column added (int 0-3).
    """
    df = df.copy()
    df["skill_tier"] = 1  # default to average

    for season in df["season"].unique():
        mask = df["season"] == season

        if player_type == "hitter":
            cols = ["barrel_pct", "hard_hit_pct"]
        else:
            cols = ["whiff_rate", "barrel_rate_against"]

        available = [c for c in cols if c in df.columns]
        if not available:
            logger.warning(
                "No skill tier columns found for %s (season %d), "
                "defaulting to tier 1",
                player_type, season,
            )
            continue

        season_df = df.loc[mask, available].copy()

        # Z-score each indicator within the season
        z_scores = pd.DataFrame(index=season_df.index)
        for col in available:
            vals = season_df[col].astype(float)
            mu, sd = vals.mean(), vals.std()
            if np.isclose(sd, 0.0) or pd.isna(sd):
                z_scores[col] = 0.0
            else:
                z_scores[col] = (vals - mu) / sd

        # For pitchers, invert barrel_rate_against (lower = better)
        if player_type == "pitcher" and "barrel_rate_against" in z_scores.columns:
            z_scores["barrel_rate_against"] = -z_scores["barrel_rate_against"]

        # Composite = mean of available z-scores (NaN-safe)
        composite = z_scores.mean(axis=1).fillna(0.0)

        # Assign quartile-based tiers
        try:
            tiers = pd.qcut(composite, q=N_SKILL_TIERS, labels=False)
        except ValueError:
            # Too few unique values for quartiles — fall back to rank
            tiers = pd.cut(
                composite.rank(method="first"),
                bins=N_SKILL_TIERS,
                labels=False,
            )

        df.loc[mask, "skill_tier"] = tiers.fillna(1).astype(int)

    logger.info(
        "Skill tiers assigned (%s): %s",
        player_type,
        df["skill_tier"].value_counts().sort_index().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# Pitcher season totals with age (cached)
# ---------------------------------------------------------------------------
def get_cached_pitcher_season_totals_with_age(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Pitcher season totals with age and age_bucket, with Parquet caching.

    Parameters
    ----------
    season : int
        MLB season year.
    force_rebuild : bool
        If True, re-query and rebuild the cache.

    Returns
    -------
    pd.DataFrame
        One row per pitcher with season stats + age + age_bucket.
    """
    return _load_or_build(
        "pitcher_season_totals_age", season,
        get_pitcher_season_totals_with_age, force_rebuild,
    )


# ---------------------------------------------------------------------------
# Multi-season pitcher data with age for expanded projection models
# ---------------------------------------------------------------------------
def build_multi_season_pitcher_data(
    seasons: list[int], min_bf: int = 1
) -> pd.DataFrame:
    """Stack pitcher season totals with age and arsenal covariates.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_bf : int
        Minimum batters faced per player-season.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season, age,
        age_bucket, games, ip, k, bb, hr, batters_faced, k_rate,
        bb_rate, hr_per_bf, hr_per_9, is_starter, whiff_rate,
        barrel_rate_against.
    """
    frames = []
    for s in seasons:
        totals = get_cached_pitcher_season_totals_with_age(s, force_rebuild=False)
        arsenal = get_pitcher_arsenal(s, force_rebuild=False)

        # Aggregate arsenal to pitcher-level covariates
        pitcher_agg = arsenal.groupby("pitcher_id").agg(
            whiffs=("whiffs", "sum"),
            swings=("swings", "sum"),
            barrels_proxy=("barrels_proxy", "sum"),
            bip=("bip", "sum"),
        ).reset_index()
        pitcher_agg["whiff_rate"] = (
            pitcher_agg["whiffs"] / pitcher_agg["swings"].replace(0, np.nan)
        )
        pitcher_agg["barrel_rate_against"] = (
            pitcher_agg["barrels_proxy"] / pitcher_agg["bip"].replace(0, np.nan)
        )

        merged = totals.merge(
            pitcher_agg[["pitcher_id", "whiff_rate", "barrel_rate_against"]],
            on="pitcher_id",
            how="left",
        )

        # Merge observed profile (zone_pct, gb_pct, avg_velo)
        try:
            obs = get_cached_pitcher_observed_profile(s)
            merge_cols = ["pitcher_id"]
            for col in ["zone_pct", "gb_pct", "avg_velo"]:
                if col in obs.columns:
                    merge_cols.append(col)
            merged = merged.merge(obs[merge_cols], on="pitcher_id", how="left")
        except Exception:
            logger.warning("No pitcher observed profile for %d, skipping merge", s)
            for col in ["zone_pct", "gb_pct", "avg_velo"]:
                if col not in merged.columns:
                    merged[col] = np.nan

        # Derive starter/reliever role
        merged["is_starter"] = (
            (merged["ip"] / merged["games"].replace(0, np.nan)) >= 3.0
        ).astype(int).fillna(0).astype(int)

        if min_bf > 1:
            merged = merged[merged["batters_faced"] >= min_bf]

        frames.append(merged)

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with missing age
    n_before = len(combined)
    combined = combined.dropna(subset=["age", "age_bucket"])
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        logger.warning("Dropped %d pitcher rows with missing age", n_dropped)

    combined["age_bucket"] = combined["age_bucket"].astype(int)

    combined = assign_skill_tier(combined, player_type="pitcher")

    logger.info(
        "Multi-season pitcher data: %d player-seasons across %s",
        len(combined), seasons,
    )
    return combined


# ---------------------------------------------------------------------------
# MiLB rookie augmentation — synthetic prior rows
# ---------------------------------------------------------------------------

# Effective PA/BF scaling by level — converts MiLB PA into "MLB-equivalent
# prior strength".  AAA translates most reliably so gets ~80% credit;
# A-ball is noisier, ~30%.
_MILB_LEVEL_PA_SCALE: dict[str, float] = {
    "AAA": 0.80,
    "AA": 0.55,
    "A+": 0.40,
    "A": 0.30,
}

# Level priority for picking best MiLB row per player (higher = prefer)
_LEVEL_PRIORITY: dict[str, int] = {"AAA": 4, "AA": 3, "A+": 2, "A": 1}


def _pick_best_milb_row(player_df: pd.DataFrame) -> pd.Series:
    """Pick the most informative MiLB row for a player.

    Prefers the most recent season, then the highest level within
    that season for maximum translation reliability.
    """
    latest_season = player_df["season"].max()
    latest = player_df[player_df["season"] == latest_season].copy()
    latest["_level_pri"] = latest["level"].map(_LEVEL_PRIORITY).fillna(0)
    return latest.sort_values("_level_pri", ascending=False).iloc[0]


def augment_hitters_with_milb_priors(
    mlb_df: pd.DataFrame,
    milb_df: pd.DataFrame | None = None,
    projection_season: int = 2026,
    min_confidence: float = 0.40,
    min_pa: int = 50,
    max_age: int = 28,
) -> pd.DataFrame:
    """Add MiLB rookies as synthetic training rows for Bayesian projection.

    For each MiLB player NOT in the MLB training data, creates a synthetic
    season row using translated rates and scaled PA.  The hierarchical model
    naturally gives these players wider posteriors due to fewer trials.

    Parameters
    ----------
    mlb_df : pd.DataFrame
        Output of ``build_multi_season_hitter_data()``.
    milb_df : pd.DataFrame or None
        Translated MiLB batter data (from ``build_milb_translated_data``).
        If None, loads from cache.
    projection_season : int
        Target season for projections (used to compute prospect age).
    min_confidence : float
        Minimum ``translation_confidence`` to include a prospect.
    min_pa : int
        Minimum MiLB PA to include a prospect.
    max_age : int
        Maximum age at MiLB level. Filters out journeyman minor leaguers.

    Returns
    -------
    pd.DataFrame
        MLB data + synthetic rookie rows, with ``_from_milb`` flag.
    """
    if milb_df is None:
        try:
            milb_df = pd.read_parquet(CACHE_DIR / "milb_translated_batters.parquet")
        except FileNotFoundError:
            logger.warning("No MiLB translated batters cache found, skipping augmentation")
            mlb_df["_from_milb"] = False
            return mlb_df

    mlb_ids = set(mlb_df["batter_id"].unique())

    # Only consider recent MiLB seasons (last 2 years before projection)
    recent_cutoff = projection_season - 2
    candidates = milb_df[
        (~milb_df["player_id"].isin(mlb_ids))
        & (milb_df["translation_confidence"] >= min_confidence)
        & (milb_df["pa"] >= min_pa)
        & (milb_df["season"] >= recent_cutoff)
        & (milb_df["age_at_level"] <= max_age)
    ].copy()

    if candidates.empty:
        logger.info("No MiLB rookies meet augmentation criteria")
        mlb_df["_from_milb"] = False
        return mlb_df

    # Pick best row per player
    rows: list[dict] = []
    for pid, grp in candidates.groupby("player_id"):
        best = _pick_best_milb_row(grp)

        # Compute projected age in target season
        years_ahead = projection_season - int(best["season"])
        age_proj = float(best["age_at_level"]) + years_ahead
        age_bucket = 0 if age_proj <= 25 else (1 if age_proj <= 30 else 2)

        # Scale PA by level reliability — fewer effective trials for lower levels
        level_scale = _MILB_LEVEL_PA_SCALE.get(best["level"], 0.25)
        effective_pa = max(int(best["pa"] * level_scale), 10)

        # Back-solve counts from translated rates
        k_rate = float(best["translated_k_pct"])
        bb_rate = float(best["translated_bb_pct"])
        k_count = int(round(k_rate * effective_pa))
        bb_count = int(round(bb_rate * effective_pa))

        row = {
            "batter_id": int(pid),
            "batter_name": best.get("player_name", ""),
            "batter_stand": None,
            "season": projection_season - 1,  # As if they played last season
            "birth_date": None,
            "age": int(round(age_proj - 1)),  # Age in that "season"
            "age_bucket": age_bucket,
            "pa": effective_pa,
            "k": k_count,
            "bb": bb_count,
            "ibb": 0,
            "hbp": 0,
            "sf": 0,
            "hits": 0,
            "hr": 0,
            "xwoba_avg": np.nan,
            "barrel_pct": np.nan,
            "hard_hit_pct": np.nan,
            "bip": 0,
            "gb": 0,
            "fb": 0,
            "bip_with_la": 0,
            "hr_fb": np.nan,
            "k_rate": k_rate,
            "bb_rate": bb_rate,
            "hr_rate": np.nan,
            "gb_rate": np.nan,
            "fb_rate": np.nan,
            "hr_per_fb": np.nan,
            "woba": np.nan,
            # Statcast covariates — NaN → skill_tier defaults to 1 (average)
            "whiff_rate": np.nan,
            "chase_rate": np.nan,
            "chase_swings": 0,
            "out_of_zone_pitches": 0,
            "z_contact_pct": np.nan,
            "fb_pct": np.nan,
            "skill_tier": 1,  # Default to average (no Statcast for MiLB)
            "_from_milb": True,
            "_milb_level": best["level"],
            "_milb_confidence": float(best["translation_confidence"]),
        }
        rows.append(row)

    rookie_df = pd.DataFrame(rows)
    mlb_df = mlb_df.copy()
    mlb_df["_from_milb"] = False

    combined = pd.concat([mlb_df, rookie_df], ignore_index=True)
    logger.info(
        "Augmented with %d MiLB hitter rookies (min_conf=%.2f, min_pa=%d)",
        len(rookie_df), min_confidence, min_pa,
    )
    return combined


# ---------------------------------------------------------------------------
# Hitter observed profile (cached)
# ---------------------------------------------------------------------------
def get_cached_hitter_observed_profile(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Hitter observed pitch-level profile with Parquet caching.

    Returns per-batter: whiff_rate, chase_rate, z_contact_pct,
    avg_exit_velo, fb_pct, hard_hit_pct.
    """
    return _load_or_build(
        "hitter_observed_profile", season,
        get_hitter_observed_profile, force_rebuild,
    )


# ---------------------------------------------------------------------------
# Sprint speed (cached)
# ---------------------------------------------------------------------------
def get_cached_sprint_speed(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Sprint speed with Parquet caching."""
    return _load_or_build(
        "sprint_speed", season, get_sprint_speed, force_rebuild,
    )


# ---------------------------------------------------------------------------
# Pitcher observed profile (cached)
# ---------------------------------------------------------------------------
def get_cached_pitcher_observed_profile(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Pitcher observed pitch-level profile with Parquet caching.

    Returns per-pitcher: whiff_rate, avg_velo, release_extension,
    zone_pct, gb_pct.
    """
    return _load_or_build(
        "pitcher_observed_profile", season,
        get_pitcher_observed_profile, force_rebuild,
    )


# ---------------------------------------------------------------------------
# Pitcher ERA-FIP gap data (for ERA derivation)
# ---------------------------------------------------------------------------
def get_pitcher_era_fip_data(
    season: int,
) -> dict[int, tuple[float | None, float, float | None, float | None, float | None]]:
    """Get ERA-FIP gap data for each pitcher in a season.

    Merges traditional stats (ERA, FIP, IP) with observed profile (GB%)
    to produce the inputs needed for ERA derivation.

    Parameters
    ----------
    season : int
        MLB season year.

    Returns
    -------
    dict[int, tuple]
        {pitcher_id: (era_fip_gap, ip, gb_pct, observed_era, observed_fip)}.
        era_fip_gap is None if either ERA or FIP is missing/NaN.
    """
    from src.data.queries import get_pitcher_traditional_stats

    trad = get_pitcher_traditional_stats(season)

    try:
        obs = get_cached_pitcher_observed_profile(season)
        if "gb_pct" in obs.columns:
            trad = trad.merge(
                obs[["pitcher_id", "gb_pct"]], on="pitcher_id", how="left",
            )
        else:
            trad["gb_pct"] = np.nan
    except Exception:
        logger.warning("Could not load pitcher observed profile for GB%% (season %d)", season)
        trad["gb_pct"] = np.nan

    result: dict[int, tuple] = {}
    for _, row in trad.iterrows():
        pid = int(row["pitcher_id"])
        era = float(row["era"]) if pd.notna(row.get("era")) else None
        fip = float(row["fip"]) if pd.notna(row.get("fip")) else None
        ip = float(row.get("ip", 0))
        gb = float(row["gb_pct"]) if pd.notna(row.get("gb_pct")) else None

        if era is not None and fip is not None:
            gap = era - fip
        else:
            gap = None

        result[pid] = (gap, ip, gb, era, fip)

    logger.info("ERA-FIP data for %d pitchers (season %d)", len(result), season)
    return result


# ---------------------------------------------------------------------------
# Extended hitter season totals (cached) — includes games, SB, CS
# ---------------------------------------------------------------------------
def get_cached_hitter_season_totals_extended(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Extended hitter season totals with games, SB, CS from boxscores."""
    return _load_or_build(
        "hitter_season_extended", season,
        get_hitter_season_totals_extended, force_rebuild,
    )


def build_multi_season_hitter_extended(
    seasons: list[int], min_pa: int = 1
) -> pd.DataFrame:
    """Stack extended hitter totals and merge sprint speed.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_pa : int
        Minimum PA per player-season.

    Returns
    -------
    pd.DataFrame
        Extended hitter data with sprint_speed merged.
    """
    frames = []
    for s in seasons:
        df = get_cached_hitter_season_totals_extended(s, force_rebuild=False)
        if min_pa > 1:
            df = df[df["pa"] >= min_pa]

        # Merge sprint speed
        try:
            sprint = get_cached_sprint_speed(s)
            df = df.merge(
                sprint[["player_id", "sprint_speed"]].rename(
                    columns={"player_id": "batter_id"}
                ),
                on="batter_id",
                how="left",
            )
        except Exception:
            if "sprint_speed" not in df.columns:
                df["sprint_speed"] = np.nan

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with missing age
    combined = combined.dropna(subset=["age", "age_bucket"])
    combined["age_bucket"] = combined["age_bucket"].astype(int)

    logger.info(
        "Multi-season extended hitter data: %d player-seasons across %s",
        len(combined), seasons,
    )
    return combined


# ---------------------------------------------------------------------------
# Extended pitcher season totals (cached) — includes outs directly
# ---------------------------------------------------------------------------
def get_cached_pitcher_season_totals_extended(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Extended pitcher season totals with outs from pitching boxscores."""
    return _load_or_build(
        "pitcher_season_extended", season,
        get_pitcher_season_totals_extended, force_rebuild,
    )


def build_multi_season_pitcher_extended(
    seasons: list[int], min_bf: int = 1
) -> pd.DataFrame:
    """Stack extended pitcher totals across seasons.

    Parameters
    ----------
    seasons : list[int]
        Seasons to include.
    min_bf : int
        Minimum BF per player-season.

    Returns
    -------
    pd.DataFrame
        Extended pitcher data.
    """
    frames = []
    for s in seasons:
        df = get_cached_pitcher_season_totals_extended(s, force_rebuild=False)
        if min_bf > 1:
            df = df[df["batters_faced"] >= min_bf]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["age", "age_bucket"])
    combined["age_bucket"] = combined["age_bucket"].astype(int)

    logger.info(
        "Multi-season extended pitcher data: %d player-seasons across %s",
        len(combined), seasons,
    )
    return combined


# ---------------------------------------------------------------------------
# Batter game logs (cached)
# ---------------------------------------------------------------------------
def get_cached_batter_game_logs(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Batter game logs with Parquet caching."""
    return _load_or_build(
        "batter_game_logs", season, get_batter_game_logs, force_rebuild
    )


# ---------------------------------------------------------------------------
# Game-level batter stats (cached) — extends game_batter_ks with BB/H/HR
# ---------------------------------------------------------------------------
def get_cached_game_batter_stats(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """Per (game, pitcher, batter) PA and K/BB/H/HR counts with caching."""
    return _load_or_build(
        "game_batter_stats", season, get_game_batter_stats, force_rebuild
    )


# ---------------------------------------------------------------------------
# MiLB season totals (cached)
# ---------------------------------------------------------------------------
def get_cached_milb_batter_season_totals(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """MiLB batter season totals with Parquet caching."""
    from src.data.queries import get_milb_batter_season_totals
    return _load_or_build(
        "milb_batter_season_totals", season, get_milb_batter_season_totals,
        force_rebuild,
    )


def get_cached_milb_pitcher_season_totals(
    season: int, force_rebuild: bool = False
) -> pd.DataFrame:
    """MiLB pitcher season totals with Parquet caching."""
    from src.data.queries import get_milb_pitcher_season_totals
    return _load_or_build(
        "milb_pitcher_season_totals", season, get_milb_pitcher_season_totals,
        force_rebuild,
    )


def build_milb_translated_data(
    seasons: list[int] | None = None,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build translated MiLB season aggregates for all seasons.

    Computes translation factors from overlap data, aggregates raw MiLB
    stats per player-season-level, applies factors, and adds age context.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to process. Defaults to 2018-2025.
    force_rebuild : bool
        If True, recompute even if cached.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (translated_batters, translated_pitchers)
    """
    from src.data.milb_translation import (
        add_age_context,
        derive_batter_translation_factors,
        derive_pitcher_translation_factors,
        translate_batter_season,
        translate_pitcher_season,
    )
    from src.data.queries import get_prospect_info

    if seasons is None:
        # MiLB game logs available from 2005 (no 2020 due to COVID)
        seasons = [y for y in range(2005, 2026) if y != 2020]

    bat_cache = CACHE_DIR / "milb_translated_batters.parquet"
    pit_cache = CACHE_DIR / "milb_translated_pitchers.parquet"

    if bat_cache.exists() and pit_cache.exists() and not force_rebuild:
        logger.info("Loading cached MiLB translated data")
        return pd.read_parquet(bat_cache), pd.read_parquet(pit_cache)

    # Derive factors
    logger.info("Deriving batter translation factors")
    bat_factors = derive_batter_translation_factors()
    logger.info("Deriving pitcher translation factors")
    pit_factors = derive_pitcher_translation_factors()

    # Cache factors
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bat_factors.to_parquet(CACHE_DIR / "milb_batter_factors.parquet", index=False)
    pit_factors.to_parquet(CACHE_DIR / "milb_pitcher_factors.parquet", index=False)

    # Aggregate and translate each season
    bat_frames = []
    pit_frames = []
    for season in seasons:
        bdf = get_cached_milb_batter_season_totals(season, force_rebuild)
        if len(bdf) > 0:
            bat_frames.append(translate_batter_season(bdf, bat_factors))

        pdf = get_cached_milb_pitcher_season_totals(season, force_rebuild)
        if len(pdf) > 0:
            pit_frames.append(translate_pitcher_season(pdf, pit_factors))

    translated_bat = pd.concat(bat_frames, ignore_index=True) if bat_frames else pd.DataFrame()
    translated_pit = pd.concat(pit_frames, ignore_index=True) if pit_frames else pd.DataFrame()

    # Add age context
    prospect_info = get_prospect_info()
    if len(translated_bat) > 0:
        translated_bat = add_age_context(translated_bat, prospect_info)
    if len(translated_pit) > 0:
        translated_pit = add_age_context(translated_pit, prospect_info)

    # Cache
    translated_bat.to_parquet(bat_cache, index=False)
    translated_pit.to_parquet(pit_cache, index=False)
    logger.info(
        "Cached MiLB translated data: %d batters, %d pitchers",
        len(translated_bat), len(translated_pit),
    )

    return translated_bat, translated_pit
