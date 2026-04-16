"""
Generalized hierarchical Bayesian pitcher projection model.

Supports multiple target stats with appropriate likelihoods:
- K%, BB%: Binomial(BF, inv_logit(theta))
- HR/BF: Binomial(BF, inv_logit(theta))

All share the same model structure:
- Age-bucket population priors (young/prime/veteran) — 3 buckets (4 hurts pitcher K%)
- Player-level random intercepts (non-centered)
- AR(2) season process for talent evolution
- Starter/reliever role covariate
- Full posterior distributions per player per season

Age buckets: 0=young(<=25), 1=prime(26-30), 2=veteran(31+)

Covariates tested for K%: whiff_rate (r=0.822), avg_velo (r=0.336),
and both combined. None improve Brier score vs Marcel — partial pooling already
captures the signal. See docs/failed_hypotheses.md for details.
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

N_AGE_BUCKETS = 3
AGE_BUCKET_LABELS = {0: "young (<=25)", 1: "prime (26-30)", 2: "veteran (31+)"}


from src.models._model_utils import StatConfig

# Backward-compat alias so existing imports/type hints keep working.
PitcherStatConfig = StatConfig


# Empirical year-to-year volatility (logit scale) from 2018-2025 pitcher data:
# k_rate: ~0.28, bb_rate: ~0.35, hr_per_bf: ~0.55
PITCHER_STAT_CONFIGS: dict[str, PitcherStatConfig] = {
    "k_rate": PitcherStatConfig(
        name="k_rate",
        count_col="k",
        trials_col="batters_faced",
        rate_col="k_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["k_rate"],
        # called_strike_rate captures command-based Ks independently of
        # swing-and-miss (partial r=+0.187 beyond swstr%). Prior attempts
        # to add whiff_rate as covariate failed (ESS collapse due to
        # collinearity with K% itself). called_strike_rate is less collinear.
        # Covariates tested and rejected:
        # - whiff_rate: ESS collapse (r=0.71 collinear with K%)
        # - called_strike_rate + csw_pct: -8.3% vs Marcel (collinearity)
        # - called_strike_rate alone: -3.0% vs Marcel (still hurts)
        # Pitcher K% works best with pure hierarchical AR(1) — no covariates.
        covariates=[],
        season_decay=1.0,  # no decay (tested 0.6, 0.8 — neither improves Brier)
        sigma_season_mu=0.22,
        sigma_season_floor=0.15,    # was 0.18 — tightened for CRPS (target ~87% cov at 95% CI)
        # K% most persistent pitcher stat
        rho_alpha=6.0, rho_beta=2.5,  # mean ~0.71
        # Lag-2 partial near zero after rho accounts for lag-1
        rho2_alpha=2.0, rho2_beta=8.0,  # mean ~0.20
        # Pitcher K% is overdispersed (factor ~1.25 empirical).
        # BetaBinomial captures extra-Binomial variance via phi.
        use_beta_binomial=True,
    ),
    "bb_rate": PitcherStatConfig(
        name="bb_rate",
        count_col="bb",
        trials_col="batters_faced",
        rate_col="bb_rate",
        likelihood="binomial",
        league_avg=LEAGUE_AVG_OVERALL["bb_rate"],
        covariates=[
            ("zone_pct", 0.0, 0.2, "zone% → BB%"),
        ],
        sigma_season_mu=0.28,
        sigma_season_floor=0.18,    # was 0.22 — tightened for CRPS
        # BB% moderately persistent — keep default
        rho_alpha=8.0, rho_beta=2.0,  # mean ~0.80
        # BB% has genuine multi-year approach trends — keep default rho2
        rho2_alpha=3.0, rho2_beta=7.0,  # mean ~0.30
    ),
    "hr_per_bf": PitcherStatConfig(
        name="hr_per_bf",
        count_col="hr",
        trials_col="batters_faced",
        rate_col="hr_per_bf",
        likelihood="binomial",
        league_avg=0.030,  # ~3.0% HR/BF league avg
        covariates=[
            ("gb_pct", 0.0, 0.2, "gb% → HR/BF"),
        ],
        sigma_player_prior=0.8,
        sigma_season_mu=0.40,
        sigma_season_floor=0.30,    # was 0.35 — tightened for CRPS
        # HR/BF least persistent pitcher rate (r=0.267 YoY)
        rho_alpha=6.0, rho_beta=3.0,  # mean ~0.67
        # High noise, low lag-2 signal
        rho2_alpha=1.5, rho2_beta=8.5,  # mean ~0.15
    ),
}


def prepare_pitcher_data(
    df: pd.DataFrame,
    stat: str,
) -> dict[str, Any]:
    """Prepare multi-season pitcher data for a specific stat model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_multi_season_pitcher_data``.
        Must contain: pitcher_id, season, age_bucket, batters_faced,
        and the stat-specific columns.
    stat : str
        Stat key from PITCHER_STAT_CONFIGS.

    Returns
    -------
    dict
        Arrays and metadata ready for the model.
    """
    from src.models._model_utils import prepare_player_data

    cfg = PITCHER_STAT_CONFIGS[stat]
    return prepare_player_data(
        df, cfg=cfg, stat=stat,
        id_col="pitcher_id",
        n_age_buckets=N_AGE_BUCKETS,
        player_type="pitcher",
    )


def build_pitcher_model(
    data: dict[str, Any],
    random_seed: int = 42,
) -> pm.Model:
    """Build a hierarchical Bayesian model for the specified pitcher stat.

    Model structure
    ---------------
    - Age-bucket population means on logit scale
    - Player-level random intercepts (non-centered)
    - AR(2) season process for talent evolution
    - Starter/reliever role shift (optional)
    - Binomial likelihood: count ~ Binomial(BF, inv_logit(theta))
    - BetaBinomial for K% (overdispersed)

    Parameters
    ----------
    data : dict
        Output of ``prepare_pitcher_data``.
    random_seed : int
        For reproducibility.

    Returns
    -------
    pm.Model
    """
    from src.models._model_utils import build_hierarchical_model

    cfg = PITCHER_STAT_CONFIGS[data["stat"]]
    return build_hierarchical_model(
        data, cfg=cfg, n_age_buckets=N_AGE_BUCKETS, random_seed=random_seed,
    )


def fit_pitcher_model(
    data: dict[str, Any],
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and sample the pitcher projection model.

    Parameters
    ----------
    data : dict
        Output of ``prepare_pitcher_data``.
    draws, tune, chains, target_accept, random_seed
        MCMC sampling parameters.

    Returns
    -------
    tuple[pm.Model, az.InferenceData]
    """
    from src.models._model_utils import fit_model

    stat = data["stat"]
    model = build_pitcher_model(data, random_seed=random_seed)
    trace = fit_model(
        model, stat=stat, label="pitcher",
        n_players=data["n_players"], n_seasons=data["n_seasons"],
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=random_seed,
    )
    return model, trace


def extract_posteriors(
    trace: az.InferenceData,
    data: dict[str, Any],
) -> pd.DataFrame:
    """Extract posterior summaries per pitcher per season.

    Parameters
    ----------
    trace : az.InferenceData
        Fitted trace.
    data : dict
        Model data dict.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, pitcher_name, pitch_hand, season, age,
        age_bucket, batters_faced, is_starter, observed_{stat},
        {stat}_mean, {stat}_sd, {stat}_2_5, {stat}_50, {stat}_97_5.
    """
    stat = data["stat"]
    cfg = PITCHER_STAT_CONFIGS[stat]
    df = data["df"]

    rate_flat = flatten_posterior(trace, "rate")

    records = []
    for pos, (i, row) in enumerate(df.iterrows()):
        samples = rate_flat[:, pos]
        rec = {
            "pitcher_id": row["pitcher_id"],
            "pitcher_name": row.get("pitcher_name", ""),
            "pitch_hand": row.get("pitch_hand", ""),
            "season": row["season"],
            "age": row.get("age", None),
            "age_bucket": row.get("age_bucket", None),
            "skill_tier": row.get("skill_tier", 1),
            "batters_faced": row["batters_faced"],
            f"observed_{stat}": row[cfg.rate_col],
            **posterior_point_summary(samples, prefix=stat),
        }
        if "is_starter" in row.index:
            rec["is_starter"] = int(row["is_starter"])
        records.append(rec)

    return pd.DataFrame(records)


def extract_rate_samples(
    trace: az.InferenceData,
    data: dict[str, Any],
    pitcher_id: int,
    season: int,
    project_forward: bool = True,
    random_seed: int = 42,
) -> np.ndarray:
    """Extract raw posterior samples for one pitcher-season.

    Parameters
    ----------
    trace : az.InferenceData
    data : dict
    pitcher_id : int
    season : int
    project_forward : bool
        If True, add one random walk step for out-of-sample projection.
    random_seed : int

    Returns
    -------
    np.ndarray
        Posterior samples (1D).
    """
    from src.models._model_utils import extract_rate_samples_forward

    cfg = PITCHER_STAT_CONFIGS[data["stat"]]
    return extract_rate_samples_forward(
        trace, data, cfg=cfg, player_id=pitcher_id, season=season,
        id_col="pitcher_id", project_forward=project_forward,
        random_seed=random_seed,
    )


def check_convergence(trace: az.InferenceData, stat: str) -> dict[str, Any]:
    """Run convergence diagnostics on the trace.

    Parameters
    ----------
    trace : az.InferenceData
    stat : str

    Returns
    -------
    dict
    """
    from src.models._model_utils import check_convergence_generic

    cfg = PITCHER_STAT_CONFIGS[stat]
    return check_convergence_generic(
        trace, cfg=cfg, stat=stat, label=f"pitcher {stat}",
    )
