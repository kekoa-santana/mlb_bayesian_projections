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
    get_game_batter_ks,
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



def get_league_overall_k_rate(seasons: list[int] | None = None) -> dict[int, float]:
    """Compute overall league K% per season from season totals.

    Uses LEAGUE_AVG_OVERALL as fallback for seasons without data.

    Parameters
    ----------
    seasons : list[int] | None
        Seasons to compute. Defaults to 2020-2025.

    Returns
    -------
    dict[int, float]
        Mapping of season → league K rate.
    """
    if seasons is None:
        seasons = [2020, 2021, 2022, 2023, 2024, 2025]

    result = {}
    for s in seasons:
        df = get_cached_season_totals(s)
        total_k = df["k"].sum()
        total_pa = df["pa"].sum()
        if total_pa > 0:
            result[s] = total_k / total_pa
        else:
            result[s] = LEAGUE_AVG_OVERALL["k_rate"]
    return result


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
# BF priors (cached)
# ---------------------------------------------------------------------------
def get_cached_bf_priors(
    season: int,
    lookback_seasons: int = 3,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """BF priors with Parquet caching.

    Builds BF priors from pitcher game logs across ``lookback_seasons``
    seasons ending at ``season``.

    Parameters
    ----------
    season : int
        Target season (priors will be computed from data up to this season).
    lookback_seasons : int
        Number of seasons to include (e.g. 3 = season-2, season-1, season).
    force_rebuild : bool
        If True, re-build even if cache exists.

    Returns
    -------
    pd.DataFrame
        BF priors per pitcher-season.
    """
    from src.models.bf_model import compute_pitcher_bf_priors

    def _builder(s: int) -> pd.DataFrame:
        seasons = list(range(s - lookback_seasons + 1, s + 1))
        frames = []
        for yr in seasons:
            try:
                frames.append(get_cached_pitcher_game_logs(yr))
            except Exception:
                logger.warning("No game logs for season %d", yr)
        if not frames:
            return pd.DataFrame()
        all_logs = pd.concat(frames, ignore_index=True)
        return compute_pitcher_bf_priors(all_logs)

    return _load_or_build("bf_priors", season, _builder, force_rebuild)


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
        age_bucket, pa, k, bb, hits, hr, xwoba_avg, barrel_pct,
        hard_hit_pct, k_rate, bb_rate, hr_rate.
    """
    frames = []
    for s in seasons:
        df = get_cached_season_totals_with_age(s, force_rebuild=False)
        if min_pa > 1:
            df = df[df["pa"] >= min_pa]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with missing age (shouldn't happen but be safe)
    n_before = len(combined)
    combined = combined.dropna(subset=["age", "age_bucket"])
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with missing age", n_dropped)

    combined["age_bucket"] = combined["age_bucket"].astype(int)

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
