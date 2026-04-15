"""
Composite pitcher projections — 4-dimension scoring.

Fits K% and BB% Bayesian models (stable, projectable stats), then enriches
with observed profile stats (whiff rate, avg velo, extension, zone%, GB%)
for a comprehensive composite score.

Dimensions
----------
1. Stuff (35%): whiff_rate + avg_velo + release_extension
2. Command (25%): projected BB% delta + zone_pct
3. Ground ball profile (15%): gb_pct
4. Projected trajectory (25%): K% delta + BB% delta
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_eng import (
    build_multi_season_pitcher_data,
    get_cached_pitcher_observed_profile,
)
from src.utils.constants import LEAGUE_BABIP_PITCHER
from src.models.pitcher_model import (
    PITCHER_STAT_CONFIGS,
    check_convergence,
    extract_posteriors,
    extract_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)
from src.models.projection_utils import (
    compute_composite,
    find_breakouts_and_regressions as _find_breakouts_and_regressions,
    fit_all_models_generic,
    project_rate_samples,
)

logger = logging.getLogger(__name__)

# Only project stable stats with the Bayesian model
ALL_STATS = ["k_rate", "bb_rate", "hr_per_bf"]

# --------------------------------------------------------------------------
# Composite dimension weights and components
# --------------------------------------------------------------------------
COMPOSITE_DIMENSIONS: dict[str, tuple[float, list[tuple[str, int, str]]]] = {
    "stuff": (0.35, [
        ("whiff_rate", +1, "observed"),          # higher whiff = better stuff
        ("avg_velo", +1, "observed"),            # harder = better
        ("release_extension", +1, "observed"),   # more extension = better
    ]),
    "command": (0.25, [
        ("delta_bb_rate", -1, "projected_delta"),  # decreasing BB% = better
        ("zone_pct", +1, "observed"),              # more zone pitches = better
    ]),
    "ground_ball": (0.15, [
        ("gb_pct", +1, "observed"),              # more GBs = fewer HRs
    ]),
    "trajectory": (0.25, [
        ("delta_k_rate", +1, "projected_delta"),   # increasing K% = better
        ("delta_bb_rate", -1, "projected_delta"),   # decreasing BB% = better
    ]),
}


def fit_all_models(
    seasons: list[int],
    min_bf: int = 100,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
    stats: list[str] | None = None,
    extract_season: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit K%, BB%, HR/BF pitcher projection models.

    Parameters
    ----------
    seasons : list[int]
        Training seasons.
    min_bf : int
        Minimum BF per player-season.
    draws, tune, chains, random_seed
        MCMC parameters.
    stats : list[str] | None
        Stats to fit. Defaults to ALL_STATS (k_rate, bb_rate, hr_per_bf).
    extract_season : int | None
        If set, pre-extract forward-projected rate samples for all pitchers
        in this season immediately after fitting, then free the MCMC trace
        to save memory.  Results stored in ``results[stat]["rate_samples"]``.

    Returns
    -------
    dict[str, dict]
        Keyed by stat name.  Always contains "data", "convergence",
        "posteriors".  Contains "rate_samples" (dict[int, ndarray]) when
        *extract_season* is set, otherwise "trace" (InferenceData).
    """
    if stats is None:
        stats = ALL_STATS

    return fit_all_models_generic(
        stats=stats,
        data_builder_fn=build_multi_season_pitcher_data,
        data_builder_kwargs={"seasons": seasons, "min_bf": min_bf},
        prepare_data_fn=prepare_pitcher_data,
        model_fitter_fn=fit_pitcher_model,
        check_convergence_fn=check_convergence,
        extract_posteriors_fn=extract_posteriors,
        extract_samples_fn=extract_rate_samples,
        id_col="pitcher_id",
        player_type="pitcher",
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        extract_season=extract_season,
    )


def _enrich_with_observed(
    base: pd.DataFrame,
    from_season: int,
) -> pd.DataFrame:
    """Merge observed pitch-level stats into base."""
    try:
        obs = get_cached_pitcher_observed_profile(from_season)
        merge_cols = ["whiff_rate", "avg_velo", "release_extension",
                      "zone_pct", "gb_pct"]
        base = base.merge(
            obs[["pitcher_id"] + merge_cols],
            on="pitcher_id",
            how="left",
            suffixes=("", "_obs"),
        )
    except Exception as e:
        logger.warning("Could not load pitcher observed profile: %s", e)
        for col in ["whiff_rate", "avg_velo", "release_extension",
                     "zone_pct", "gb_pct"]:
            if col not in base.columns:
                base[col] = np.nan

    return base


def _compute_composite(base: pd.DataFrame) -> pd.DataFrame:
    """Compute 4-dimension composite score using z-score normalization."""
    return compute_composite(base, COMPOSITE_DIMENSIONS)


def _derive_fip_era(
    base: pd.DataFrame,
    pitcher_rate_samples: dict[str, dict[int, np.ndarray]],
    era_fip_data: dict | None,
    rng: np.random.Generator,
    xgb_gap_priors: dict[int, float] | None = None,
    xgb_babip_priors: dict[int, float] | None = None,
) -> None:
    """Add projected FIP/ERA columns to base DataFrame in-place."""
    from src.models.derived_stats import (
        derive_pitcher_fip,
        derive_pitcher_outs_rate,
        derive_pitcher_era,
    )

    needed = ["k_rate", "bb_rate", "hr_per_bf"]
    if not all(s in pitcher_rate_samples for s in needed):
        return

    fip_means: dict[int, float] = {}
    fip_sds: dict[int, float] = {}
    fip_lo: dict[int, float] = {}
    fip_hi: dict[int, float] = {}
    era_means: dict[int, float] = {}
    era_sds: dict[int, float] = {}
    era_lo: dict[int, float] = {}
    era_hi: dict[int, float] = {}

    for pid in base["pitcher_id"]:
        if not all(
            pid in pitcher_rate_samples.get(s, {}) for s in needed
        ):
            continue

        k_s = pitcher_rate_samples["k_rate"][pid]
        bb_s = pitcher_rate_samples["bb_rate"][pid]
        hr_s = pitcher_rate_samples["hr_per_bf"][pid]
        n = min(len(k_s), len(bb_s), len(hr_s))
        k_s, bb_s, hr_s = k_s[:n], bb_s[:n], hr_s[:n]

        # Derive outs — use XGBoost BABIP prior if available, else league avg
        babip_prior = (xgb_babip_priors or {}).get(pid)
        outs_s = derive_pitcher_outs_rate(
            k_s, bb_s, hr_s,
            observed_babip=None, bip=0,
            league_babip=babip_prior if babip_prior is not None else LEAGUE_BABIP_PITCHER,
            rng=rng,
        )

        fip_s = derive_pitcher_fip(k_s, bb_s, hr_s, outs_s)
        fip_means[pid] = float(np.mean(fip_s))
        fip_sds[pid] = float(np.std(fip_s))
        fip_lo[pid] = float(np.percentile(fip_s, 2.5))
        fip_hi[pid] = float(np.percentile(fip_s, 97.5))

        # ERA = FIP + shrunken gap (with XGBoost prior if available)
        gap_prior = (xgb_gap_priors or {}).get(pid)
        if era_fip_data and pid in era_fip_data:
            gap, ip, gb = era_fip_data[pid][:3]
            era_s = derive_pitcher_era(
                fip_s, gap, ip, gb, rng=rng, xgb_gap_prior=gap_prior,
            )
        else:
            era_s = fip_s.copy()

        era_means[pid] = float(np.mean(era_s))
        era_sds[pid] = float(np.std(era_s))
        era_lo[pid] = float(np.percentile(era_s, 2.5))
        era_hi[pid] = float(np.percentile(era_s, 97.5))

    base["projected_fip"] = base["pitcher_id"].map(fip_means)
    base["projected_fip_sd"] = base["pitcher_id"].map(fip_sds)
    base["projected_fip_2_5"] = base["pitcher_id"].map(fip_lo)
    base["projected_fip_97_5"] = base["pitcher_id"].map(fip_hi)

    base["projected_era"] = base["pitcher_id"].map(era_means)
    base["projected_era_sd"] = base["pitcher_id"].map(era_sds)
    base["projected_era_2_5"] = base["pitcher_id"].map(era_lo)
    base["projected_era_97_5"] = base["pitcher_id"].map(era_hi)

    # Observed ERA and FIP from era_fip_data
    if era_fip_data:
        obs_era = {
            pid: vals[3]
            for pid, vals in era_fip_data.items()
            if len(vals) > 3 and vals[3] is not None
        }
        obs_fip = {
            pid: vals[4]
            for pid, vals in era_fip_data.items()
            if len(vals) > 4 and vals[4] is not None
        }
        base["observed_era"] = base["pitcher_id"].map(obs_era)
        base["observed_fip"] = base["pitcher_id"].map(obs_fip)

    logger.info(
        "Derived FIP/ERA projections for %d pitchers", len(fip_means),
    )


def project_forward(
    model_results: dict[str, dict[str, Any]],
    from_season: int,
    min_bf: int = 150,
    random_seed: int = 42,
    era_fip_data: dict[int, tuple] | None = None,
    calibration_t: dict[str, float] | None = None,
    xgb_priors: dict[str, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """Forward-project K%/BB% and build 4-dimension composite.

    Parameters
    ----------
    model_results : dict
        Output of ``fit_all_models``.
    from_season : int
        Season to project from (most recent training season).
    min_bf : int
        Minimum BF in from_season to include in projections.
    random_seed : int
    era_fip_data : dict, optional
        {pitcher_id: (era_fip_gap, ip, gb_pct, observed_era, observed_fip)}.
        Output of ``feature_eng.get_pitcher_era_fip_data()``.
    calibration_t : dict, optional
        Per-stat calibration factors {stat_name: T}.
        T < 1.0 narrows intervals, T > 1.0 widens.
    xgb_priors : dict, optional
        {"era_fip_gap": {pid: value}, "babip": {pid: value}}.
        XGBoost-predicted priors from ``xgb_priors.predict_pitcher_priors``.

    Returns
    -------
    pd.DataFrame
        One row per pitcher with projected K%/BB%, observed stats,
        per-stat deltas, and composite improvement score.
    """
    first_stat = list(model_results.keys())[0]
    first_data = model_results[first_stat]["data"]
    first_df = first_data["df"]

    keep_cols = ["pitcher_id", "pitcher_name", "pitch_hand", "season", "age",
                  "age_bucket", "batters_faced", "is_starter"]
    if "skill_tier" in first_df.columns:
        keep_cols.append("skill_tier")
    base = first_df[
        (first_df["season"] == from_season) & (first_df["batters_faced"] >= min_bf)
    ][keep_cols].copy()

    if len(base) == 0:
        logger.warning("No pitchers found in season %d with >= %d BF",
                        from_season, min_bf)
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    pitcher_rate_samples: dict[str, dict[int, np.ndarray]] = {}

    # For each Bayesian stat (K%, BB%, HR/BF), extract observed + projected
    for stat, res in model_results.items():
        cfg = PITCHER_STAT_CONFIGS[stat]
        data = res["data"]
        pre_extracted = res.get("rate_samples") or {}
        trace = res.get("trace")

        stat_df = data["df"]
        stat_season = stat_df[stat_df["season"] == from_season]

        obs_col = f"observed_{stat}"
        proj_col = f"projected_{stat}"
        delta_col = f"delta_{stat}"
        sd_col = f"projected_{stat}_sd"
        ci_lo_col = f"projected_{stat}_2_5"
        ci_hi_col = f"projected_{stat}_97_5"

        obs_map = dict(zip(stat_season["pitcher_id"], stat_season[cfg.rate_col]))
        base[obs_col] = base["pitcher_id"].map(obs_map)

        # Career BF-weighted average
        career_col = f"career_{stat}"
        career_group = stat_df[
            stat_df["pitcher_id"].isin(base["pitcher_id"])
        ].groupby("pitcher_id")
        career_sum = career_group.apply(
            lambda g: (g[cfg.rate_col] * g["batters_faced"]).sum()
            / g["batters_faced"].sum()
            if g["batters_faced"].sum() > 0 else np.nan,
            include_groups=False,
        )
        base[career_col] = base["pitcher_id"].map(career_sum)

        _cal_t = (calibration_t or {}).get(stat, 1.0)

        proj_means, proj_sds, proj_lo, proj_hi, stat_samples = project_rate_samples(
            ids=base["pitcher_id"],
            pre_extracted=pre_extracted,
            trace=trace,
            data=data,
            from_season=from_season,
            id_col="pitcher_id",
            extract_samples_fn=extract_rate_samples,
            calibration_t=_cal_t,
            random_seed=random_seed,
            collect_samples=True,
        )
        pitcher_rate_samples[stat] = stat_samples if stat_samples is not None else {}

        base[proj_col] = base["pitcher_id"].map(proj_means)
        base[sd_col] = base["pitcher_id"].map(proj_sds)
        base[ci_lo_col] = base["pitcher_id"].map(proj_lo)
        base[ci_hi_col] = base["pitcher_id"].map(proj_hi)
        base[delta_col] = base[proj_col] - base[obs_col]

    # Derive FIP/ERA from rate posteriors
    _xgb = xgb_priors or {}
    _derive_fip_era(
        base, pitcher_rate_samples, era_fip_data, rng,
        xgb_gap_priors=_xgb.get("era_fip_gap"),
        xgb_babip_priors=_xgb.get("babip"),
    )

    # Enrich with observed stats (whiff, velo, extension, zone%, GB%)
    base = _enrich_with_observed(base, from_season)

    # Compute 4-dimension composite
    base = _compute_composite(base)

    base = base.sort_values("composite_score", ascending=False).reset_index(drop=True)

    logger.info(
        "Projected %d pitchers from %d forward",
        len(base), from_season,
    )
    return base


def find_breakouts_and_regressions(
    projections: pd.DataFrame,
    n_top: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split projections into breakout candidates and regression risks.

    Parameters
    ----------
    projections : pd.DataFrame
        Output of ``project_forward``.
    n_top : int
        Number of pitchers per category.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (breakouts, regressions) sorted by |composite_score|.
    """
    return _find_breakouts_and_regressions(projections, n_top=n_top)
