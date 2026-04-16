"""
Generalized hierarchical Bayesian hitter projection model.

Supports multiple target stats with appropriate likelihoods:
- K%, BB%, HR/PA: Binomial(PA, inv_logit(theta))
- xwOBA: Normal(theta, sigma_obs)

All share the same model structure:
- Age-bucket population priors (young/prime/veteran)
- Player-level random intercepts (non-centered)
- AR(2) season process for talent evolution
- Stat-specific covariates (approach metrics, batted ball metrics)
- Full posterior distributions per player per season

Age buckets: 0=young(<=25), 1=prime(26-30), 2=veteran(31+)
"""
from __future__ import annotations

import logging
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data.feature_eng import N_SKILL_TIERS, SKILL_TIER_LABELS  # noqa: F401 — re-exported
from src.utils.constants import LEAGUE_AVG_OVERALL
from src.utils.math_helpers import flatten_posterior, posterior_point_summary

logger = logging.getLogger(__name__)

N_AGE_BUCKETS = 4
AGE_BUCKET_LABELS = {
    0: "young (<=25)",
    1: "development-peak (26-28)",
    2: "maintenance (29-32)",
    3: "veteran (33+)",
}


from src.models._model_utils import StatConfig  # re-exported for external callers


# Pre-defined stat configs
STAT_CONFIGS: dict[str, StatConfig] = {
    "k_rate": StatConfig(
        name="k_rate",
        count_col="k",
        trials_col="pa",
        rate_col="k_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["k_rate"],
        covariates=[
            # chase_rate removed: r=-0.013 with K% (near zero).
            # Chase predicts BB% (r=-0.620), not K%.
            ("whiff_rate", 0.0, 0.2, "whiff% → K%"),
        ],
        # empirical logit-scale yr-to-yr SD ≈ 0.24
        sigma_season_mu=0.20,
        sigma_season_floor=0.15,    # was 0.18 — tightened for CRPS (target ~87% cov at 95% CI)
        # K% is the most persistent hitter stat (r=0.795 YoY)
        rho_alpha=9.0, rho_beta=1.5,  # mean ~0.86
        # Lag-2 partial autocorrelation near zero after rho accounts for lag-1
        rho2_alpha=2.0, rho2_beta=8.0,  # mean ~0.20
    ),
    "bb_rate": StatConfig(
        name="bb_rate",
        count_col="bb",
        trials_col="pa",
        rate_col="bb_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["bb_rate"],
        covariates=[
            ("chase_rate", 0.0, 0.2, "chase% → BB%"),
            # z_contact_pct removed: partial r = -0.064 with next-year BB%
            # after controlling for current BB% — nearly zero, wrong sign.
        ],
        # empirical logit-scale yr-to-yr SD ≈ 0.33
        sigma_season_mu=0.25,
        sigma_season_floor=0.17,    # was 0.20 — tightened for CRPS (target ~85% cov)
        # BB% moderately persistent (r=0.706 YoY) — keep default
        rho_alpha=8.0, rho_beta=2.0,  # mean ~0.80
        # BB% has genuine multi-year approach trends — keep default rho2
        rho2_alpha=3.0, rho2_beta=7.0,  # mean ~0.30
    ),
    "gb_rate": StatConfig(
        name="gb_rate",
        count_col="gb",
        trials_col="bip_with_la",
        rate_col="gb_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["gb_rate"],
        # No covariates — GB% is stable enough (r=0.73) for hierarchical
        # structure + AR(1) to handle. LA-based covariates would be circular.
        covariates=[],
        # empirical logit-scale yr-to-yr SD ≈ 0.20
        sigma_season_mu=0.18,
        sigma_season_floor=0.12,    # was 0.14 — tightened for CRPS
        # GB% moderately persistent (r=0.73 YoY)
        rho_alpha=7.0, rho_beta=2.5,  # mean ~0.74
        # Lag-2 partial near zero — rho handles persistence
        rho2_alpha=2.0, rho2_beta=8.0,  # mean ~0.20
    ),
    "fb_rate": StatConfig(
        name="fb_rate",
        count_col="fb",
        trials_col="bip_with_la",
        rate_col="fb_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["fb_rate"],
        # No covariates — same rationale as GB%
        covariates=[],
        # empirical logit-scale yr-to-yr SD ≈ 0.22
        sigma_season_mu=0.18,
        sigma_season_floor=0.12,    # was 0.14 — tightened for CRPS
        # FB% roughly mirrors GB% stability
        rho_alpha=7.0, rho_beta=2.5,  # mean ~0.74
        # Mirrors GB% — lag-2 partial near zero
        rho2_alpha=2.0, rho2_beta=8.0,  # mean ~0.20
    ),
    "hr_per_fb": StatConfig(
        name="hr_per_fb",
        count_col="hr_fb",
        trials_col="fb",
        rate_col="hr_per_fb",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["hr_per_fb"],
        # Power composite (ISO + barrel% + hard_hit% + exit_velo) as single
        # covariate. Individual barrel_pct (r=0.39 YoY) caused ESS collapse,
        # but the 4-metric composite is more stable and predicts next-year
        # barrel% 54% better and ISO 9% better than raw stats (validated
        # 2019-2024 walk-forward, grade_prior_analysis.py).
        covariates=[
            ("power_composite", 0.0, 0.3, "power_z → HR/FB"),
        ],
        sigma_player_prior=0.5,
        # empirical logit-scale yr-to-yr SD ≈ 0.45 — HR/FB is volatile
        sigma_season_mu=0.35,
        sigma_season_floor=0.25,    # was 0.28 — tightened for CRPS
        # HR/FB is the least persistent rate stat — needs most regression
        rho_alpha=6.0, rho_beta=3.0,  # mean ~0.67
        # High noise, low lag-2 signal
        rho2_alpha=1.5, rho2_beta=8.5,  # mean ~0.15
    ),
    "woba": StatConfig(
        name="woba",
        count_col="woba",           # value column (logit-transformed for fitting)
        trials_col="pa",            # PA weight (for career averages)
        rate_col="woba",
        likelihood="logit_normal",  # Normal on logit scale → bounded [0,1] projections
        league_avg=LEAGUE_AVG_OVERALL["woba"],  # 0.315 (stored on natural scale, logit-transformed internally)
        covariates=[
            ("hard_hit_pct", 0.0, 0.2, "hard_hit% → wOBA"),
            # barrel_pct replaced by xslg: xSLG YoY r=0.765 vs barrel r=0.425,
            # predicts next-year wOBA better (0.409 vs 0.255).
            ("xslg", 0.0, 0.2, "xSLG → wOBA"),
            # xwOBA strips BABIP luck; +2-3% R² beyond hard_hit alone.
            # Catches lucky/unlucky hitters the model would otherwise miss.
            ("xwoba_avg", 0.0, 0.15, "xwOBA → wOBA"),
        ],
        sigma_season_mu=0.15,
        sigma_season_floor=0.14,
        sigma_player_prior=0.5,
        sigma_obs_prior=0.18,       # natural 0.04 → logit ~0.18
        # wOBA composite metric — high persistence for true talent
        rho_alpha=8.0, rho_beta=2.0,  # mean ~0.80
        rho2_alpha=2.0, rho2_beta=8.0,  # mean ~0.20
        mu_pop_sigma=0.30,  # widened from 0.25 to let age×tier cells separate more
        # StudentT(4) tails for player intercepts: EDA shows young Tier-1
        # hitters (xwOBA>=.400) have population wOBA=.350, but Normal
        # shrinkage crushes outliers toward cell means.  StudentT lets
        # elite young players (Caminero, Soto, Acuna archetypes) sit further
        # from the population mean without over-regression.
        # Prior testing: StudentT(4) + mu_pop_sigma=0.30 avoids the ESS
        # collapse seen at mu_pop_sigma=0.45 with Normal tails.
        alpha_prior_nu=4.0,
    ),
    "chase_rate": StatConfig(
        name="chase_rate",
        count_col="chase_swings",
        trials_col="out_of_zone_pitches",
        rate_col="chase_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["chase_rate"],  # 0.30
        # No covariates — chase_rate is self-predictive (r=0.84 YoY);
        # same rationale as GB% (r=0.73): hierarchical + AR(1) handles it.
        # Adding bb_rate would be circular (bb_rate uses chase_rate as covariate).
        covariates=[],
        # empirical logit-scale yr-to-yr SD ≈ 0.15
        sigma_season_mu=0.13,
        sigma_season_floor=0.07,    # was 0.10 — tightened for CRPS
        # Chase rate is the most stable hitter metric (r=0.84 YoY, 87% between-player)
        rho_alpha=9.0, rho_beta=1.0,  # mean ~0.90
        # Very high rho already; lag-2 partial near zero
        rho2_alpha=2.0, rho2_beta=8.0,  # mean ~0.20
    ),
}


def prepare_hitter_data(
    df: pd.DataFrame,
    stat: str,
) -> dict[str, Any]:
    """Prepare multi-season hitter data for a specific stat model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_multi_season_hitter_data``.
        Must contain: batter_id, season, age_bucket, pa, and the
        stat-specific columns.
    stat : str
        Stat key from STAT_CONFIGS (e.g. "k_rate", "bb_rate").

    Returns
    -------
    dict
        Arrays and metadata ready for the model.
    """
    from src.models._model_utils import prepare_player_data

    cfg = STAT_CONFIGS[stat]
    return prepare_player_data(
        df, cfg=cfg, stat=stat,
        id_col="batter_id",
        n_age_buckets=N_AGE_BUCKETS,
        player_type="hitter",
    )


def build_hitter_model(
    data: dict[str, Any],
    random_seed: int = 42,
) -> pm.Model:
    """Build a hierarchical Bayesian model for the specified stat.

    Model structure
    ---------------
    - Age-bucket population means on logit/natural scale
    - Player-level random intercepts (non-centered)
    - AR(2) season process for talent evolution
    - Statcast covariates shift the linear predictor
    - Binomial or Normal likelihood depending on stat type

    Parameters
    ----------
    data : dict
        Output of ``prepare_hitter_data``.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pm.Model
    """
    from src.models._model_utils import build_hierarchical_model

    cfg = STAT_CONFIGS[data["stat"]]
    return build_hierarchical_model(
        data, cfg=cfg, n_age_buckets=N_AGE_BUCKETS, random_seed=random_seed,
    )


def fit_hitter_model(
    data: dict[str, Any],
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and sample the hitter projection model.

    Parameters
    ----------
    data : dict
        Output of ``prepare_hitter_data``.
    draws, tune, chains, target_accept, random_seed
        MCMC sampling parameters.

    Returns
    -------
    tuple[pm.Model, az.InferenceData]
    """
    from src.models._model_utils import fit_model

    stat = data["stat"]
    model = build_hitter_model(data, random_seed=random_seed)
    trace = fit_model(
        model, stat=stat, label="hitter",
        n_players=data["n_players"], n_seasons=data["n_seasons"],
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=random_seed,
    )
    return model, trace


def extract_posteriors(
    trace: az.InferenceData,
    data: dict[str, Any],
) -> pd.DataFrame:
    """Extract posterior summaries per player per season.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace.
    data : dict
        Model data dict.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, batter_name, season, age, age_bucket, pa,
        observed_{stat}, {stat}_mean, {stat}_sd, {stat}_2_5, {stat}_50,
        {stat}_97_5.
    """
    stat = data["stat"]
    cfg = STAT_CONFIGS[stat]
    df = data["df"]

    rate_flat = flatten_posterior(trace, "rate")

    records = []
    for pos, (i, row) in enumerate(df.iterrows()):
        samples = rate_flat[:, pos]
        rec = {
            "batter_id": row["batter_id"],
            "batter_name": row.get("batter_name", ""),
            "batter_stand": row.get("batter_stand", ""),
            "season": row["season"],
            "age": row.get("age", None),
            "age_bucket": row.get("age_bucket", None),
            "skill_tier": row.get("skill_tier", 1),
            "pa": row["pa"],
            f"observed_{stat}": row[cfg.rate_col],
            **posterior_point_summary(samples, prefix=stat),
        }
        records.append(rec)

    return pd.DataFrame(records)


def extract_rate_samples(
    trace: az.InferenceData,
    data: dict[str, Any],
    batter_id: int,
    season: int,
    project_forward: bool = True,
    random_seed: int = 42,
) -> np.ndarray:
    """Extract raw posterior samples for one player-season.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace.
    data : dict
        Model data dict.
    batter_id : int
        Target batter.
    season : int
        Season whose posterior to extract.
    project_forward : bool
        If True, add one random walk step for out-of-sample projection.
    random_seed : int
        For reproducibility.

    Returns
    -------
    np.ndarray
        Posterior samples (1D array).
    """
    from src.models._model_utils import extract_rate_samples_forward

    cfg = STAT_CONFIGS[data["stat"]]
    return extract_rate_samples_forward(
        trace, data, cfg=cfg, player_id=batter_id, season=season,
        id_col="batter_id", project_forward=project_forward,
        random_seed=random_seed,
    )


def check_convergence(trace: az.InferenceData, stat: str) -> dict[str, Any]:
    """Run convergence diagnostics on the trace.

    Parameters
    ----------
    trace : az.InferenceData
    stat : str
        Stat key for covariate naming.

    Returns
    -------
    dict
        Summary with r_hat, ESS, and divergence counts.
    """
    from src.models._model_utils import check_convergence_generic

    cfg = STAT_CONFIGS[stat]
    return check_convergence_generic(trace, cfg=cfg, stat=stat, label=stat)
