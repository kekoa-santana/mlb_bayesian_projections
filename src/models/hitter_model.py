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
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data.feature_eng import N_SKILL_TIERS, SKILL_TIER_LABELS
from src.utils.constants import LEAGUE_AVG_OVERALL

logger = logging.getLogger(__name__)

N_AGE_BUCKETS = 4
AGE_BUCKET_LABELS = {
    0: "young (<=25)",
    1: "development-peak (26-28)",
    2: "maintenance (29-32)",
    3: "veteran (33+)",
}


@dataclass
class StatConfig:
    """Configuration for a single target stat."""

    name: str
    # Column names in the DataFrame
    count_col: str         # numerator (e.g. "k", "bb", "hr") or value col for continuous
    trials_col: str        # denominator (e.g. "pa") — unused for continuous
    rate_col: str          # pre-computed rate (e.g. "k_rate")
    # Likelihood type
    likelihood: str        # "binomial" or "normal"
    # Prior location (league average on natural scale)
    league_avg: float
    # Covariate config: list of (col_name, prior_mu, prior_sigma, direction_label)
    # direction_label is for logging only
    covariates: list[tuple[str, float, float, str]]
    # sigma_season prior: LogNormal(mu, 0.5) — mode on logit/natural scale
    # derived from empirical year-to-year volatility
    sigma_season_mu: float = 0.15
    # Floor for sigma_season in forward projection — ensures CIs reflect
    # known empirical year-to-year volatility even when the model under-
    # estimates it due to short training windows
    sigma_season_floor: float = 0.0
    # sigma prior for player intercepts
    sigma_player_prior: float = 0.5
    # sigma prior for observation noise (normal likelihood only)
    sigma_obs_prior: float = 0.05
    # AR(1) rho prior: Beta(alpha, beta). Higher mean = more year-to-year
    # persistence. Stat-specific because stability varies (OOPSY research):
    # K% most persistent (~0.86), HR/FB least (~0.67).
    rho_alpha: float = 8.0
    rho_beta: float = 2.0
    # AR(2) rho2 prior: Beta(alpha, beta). Stat-specific because lag-2
    # partial autocorrelation varies. K%/GB% have near-zero partial lag-2
    # (rho already explains it), while BB% has genuine multi-year trends.
    rho2_alpha: float = 3.0
    rho2_beta: float = 7.0
    # mu_pop prior width: how far age-bucket × skill-tier cell means
    # can spread from league average.  Default 0.3 works for logit-scale
    # binomial stats; wOBA (natural scale, SD ~0.04) needs wider to let
    # cells separate.
    mu_pop_sigma: float = 0.3


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
        # Logit-scale priors (Jacobian at wOBA=0.315 is ~4.63):
        # Tuned from v1 backtest: mu_pop_sigma=0.45 too wide → ESS collapse,
        # sigma_season_floor=0.09 too tight → 78% coverage at 95% CI.
        sigma_season_mu=0.15,
        sigma_season_floor=0.14,
        sigma_player_prior=0.5,
        sigma_obs_prior=0.18,       # natural 0.04 → logit ~0.18
        # wOBA composite metric — high persistence for true talent
        rho_alpha=8.0, rho_beta=2.0,  # mean ~0.80 (was 0.74 — under-projected young elites)
        rho2_alpha=2.0, rho2_beta=8.0,  # mean ~0.20
        mu_pop_sigma=0.25,  # tighter than binomial stats — helps identification on logit scale
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
    cfg = STAT_CONFIGS[stat]
    df = df.copy()

    # For continuous likelihoods, drop rows with NaN values
    if cfg.likelihood in ("normal", "logit_normal"):
        df = df.dropna(subset=[cfg.rate_col])

    # Encode player IDs as contiguous ints
    player_ids = df["batter_id"].unique()
    player_map = {pid: idx for idx, pid in enumerate(player_ids)}
    df["player_idx"] = df["batter_id"].map(player_map)

    # Season offsets
    min_season = df["season"].min()
    df["season_idx"] = df["season"] - min_season

    n_players = len(player_ids)
    n_seasons = df["season_idx"].max() + 1

    # Age bucket + skill tier per player (use most recent season)
    player_age_bucket = np.zeros(n_players, dtype=int)
    player_skill_tier = np.ones(n_players, dtype=int)  # default: average (1)
    latest_season = df.groupby("player_idx")["season"].max()
    for pidx in range(n_players):
        if pidx in latest_season.index:
            latest_s = latest_season[pidx]
            rows = df[(df["player_idx"] == pidx) & (df["season"] == latest_s)]
            if len(rows) > 0:
                player_age_bucket[pidx] = int(rows.iloc[0]["age_bucket"])
                if "skill_tier" in rows.columns:
                    player_skill_tier[pidx] = int(rows.iloc[0].get(
                        "skill_tier", 1,
                    ))

    # Recency weighting is handled at the projection step (extract_rate_samples)
    # via age-dependent rho multipliers, NOT in the likelihood.
    # Modifying likelihood weights reduces effective sample size and triggers
    # excess hierarchical shrinkage toward the population mean.
    recency_weight = np.ones(len(df), dtype=np.float64)

    # Z-score covariates
    cov_arrays = {}
    for col_name, _, _, _ in cfg.covariates:
        if col_name not in df.columns:
            df[col_name] = np.nan
        vals = df[col_name].values.astype(float)
        vals = np.nan_to_num(vals, nan=0.0)
        mu, sd = vals.mean(), vals.std()
        if np.isclose(sd, 0.0):
            cov_arrays[col_name] = np.zeros_like(vals)
        else:
            cov_arrays[col_name] = (vals - mu) / sd

    recency_wt = recency_weight

    # --- MiLB-informed prior offsets for early-career players ---
    milb_prior_offset = np.zeros(n_players, dtype=np.float64)
    has_milb_prior = np.zeros(n_players, dtype=np.float64)
    try:
        from src.data.feature_eng import compute_milb_prior_offsets
        max_season = int(df["season"].max())
        offsets = compute_milb_prior_offsets(
            df, player_id_col="batter_id", player_type="hitter",
            projection_season=max_season + 1,
        )
        if offsets:
            for pid, stat_offsets in offsets.items():
                if pid in player_map and stat in stat_offsets:
                    pidx = player_map[pid]
                    milb_prior_offset[pidx] = stat_offsets[stat]
                    has_milb_prior[pidx] = 1.0
            n_with_prior = int(has_milb_prior.sum())
            if n_with_prior > 0:
                logger.info(
                    "MiLB prior offsets: %d/%d hitters for %s",
                    n_with_prior, n_players, stat,
                )
    except Exception as exc:
        logger.debug("MiLB prior offsets unavailable: %s", exc)

    result: dict[str, Any] = {
        "player_idx": df["player_idx"].values.astype(int),
        "season_idx": df["season_idx"].values.astype(int),
        "n_players": n_players,
        "n_seasons": n_seasons,
        "player_map": player_map,
        "player_ids": player_ids,
        "min_season": min_season,
        "player_age_bucket": player_age_bucket,
        "player_skill_tier": player_skill_tier,
        "covariates": cov_arrays,
        "milb_prior_offset": milb_prior_offset,
        "has_milb_prior": has_milb_prior,
        "recency_weight": recency_wt,
        "stat": stat,
        "df": df,
    }

    if cfg.likelihood == "binomial":
        result["trials"] = df[cfg.trials_col].values.astype(int)
        result["counts"] = df[cfg.count_col].values.astype(int)
    elif cfg.likelihood == "logit_normal":
        # Logit-transform observations so the model works on unbounded scale
        # but projections are naturally bounded in [0, 1]
        raw_vals = df[cfg.rate_col].values.astype(float)
        clipped = np.clip(raw_vals, 1e-4, 1 - 1e-4)
        result["y_obs"] = np.log(clipped / (1 - clipped))
        result["pa_weight"] = df[cfg.trials_col].values.astype(float)
    else:
        result["y_obs"] = df[cfg.rate_col].values.astype(float)
        # PA as precision weight for xwOBA
        result["pa_weight"] = df[cfg.trials_col].values.astype(float)

    return result


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
    import pytensor.tensor as pt

    stat = data["stat"]
    cfg = STAT_CONFIGS[stat]

    player_idx = data["player_idx"]
    season_idx = data["season_idx"]
    n_players = data["n_players"]
    n_seasons = data["n_seasons"]
    age_bucket = data["player_age_bucket"]
    skill_tier = data["player_skill_tier"]

    if cfg.likelihood in ("binomial", "logit_normal"):
        league_logit = np.log(cfg.league_avg / (1 - cfg.league_avg))
    else:
        league_logit = cfg.league_avg  # not actually logit for normal

    with pm.Model() as model:
        # --- Age-bucket x skill-tier population means ---
        mu_pop = pm.Normal(
            "mu_pop",
            mu=league_logit,
            sigma=cfg.mu_pop_sigma,
            shape=(N_AGE_BUCKETS, N_SKILL_TIERS),
        )

        sigma_player = pm.HalfNormal("sigma_player", sigma=cfg.sigma_player_prior)

        # --- Covariate effects ---
        betas = {}
        for col_name, prior_mu, prior_sigma, label in cfg.covariates:
            betas[col_name] = pm.Normal(
                f"beta_{col_name}", mu=prior_mu, sigma=prior_sigma
            )

        # --- Player-level intercepts (non-centered) ---
        # For early-career players with MiLB data, shift alpha toward
        # their translated MiLB rate (weighted by level reliability).
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_players)
        milb_offset = data.get("milb_prior_offset")
        has_milb = data.get("has_milb_prior")
        if milb_offset is not None and has_milb is not None and has_milb.sum() > 0:
            milb_offset_t = pt.as_tensor_variable(milb_offset.astype(np.float64))
            has_milb_t = pt.as_tensor_variable(has_milb.astype(np.float64))
            alpha = pm.Deterministic(
                "alpha",
                mu_pop[age_bucket, skill_tier]
                + sigma_player * alpha_raw
                + milb_offset_t * has_milb_t,
            )
        else:
            alpha = pm.Deterministic(
                "alpha",
                mu_pop[age_bucket, skill_tier] + sigma_player * alpha_raw,
            )

        # --- AR(2) season process ---
        # LogNormal prior resists collapsing to zero; centered on empirical
        # year-to-year volatility (logit scale for binomial, natural for normal)
        sigma_season = pm.LogNormal(
            "sigma_season",
            mu=np.log(cfg.sigma_season_mu),
            sigma=0.35,  # was 0.5 — tighter to reduce right-tail mass on innovation variance
        )
        rho = pm.Beta("rho", alpha=cfg.rho_alpha, beta=cfg.rho_beta)
        # AR(2) coefficient — stat-specific lag-2 persistence.
        # Informed by empirical partial autocorrelations (Yule-Walker).
        rho2 = pm.Beta("rho2", alpha=cfg.rho2_alpha, beta=cfg.rho2_beta)

        if n_seasons > 1:
            innovation = pm.Normal(
                "innovation", mu=0, sigma=1,
                shape=(n_players, n_seasons),
            )
            # Season 0: just innovation
            s0 = (sigma_season * innovation[:, 0]).dimshuffle(0, "x")
            ar_components = [s0]

            if n_seasons >= 2:
                # Season 1: AR(1) only (no lag-2 available yet)
                s1 = rho * s0[:, -1] + sigma_season * innovation[:, 1]
                ar_components.append(s1.dimshuffle(0, "x"))

            # Seasons 2+: full AR(2)
            for t in range(2, n_seasons):
                prev1 = ar_components[-1][:, -1]   # t-1
                prev2 = ar_components[-2][:, -1]   # t-2
                cur = (rho * prev1 + rho2 * prev2
                       + sigma_season * innovation[:, t])
                ar_components.append(cur.dimshuffle(0, "x"))

            season_effect = pm.Deterministic(
                "season_effect", pt.concatenate(ar_components, axis=1)
            )
        else:
            season_effect = pt.zeros((n_players, 1))

        # --- Linear predictor ---
        theta = (
            alpha[player_idx]
            + season_effect[player_idx, season_idx]
        )

        # Add covariate effects
        for col_name, _, _, _ in cfg.covariates:
            theta = theta + betas[col_name] * data["covariates"][col_name]

        # --- Likelihood ---
        if cfg.likelihood == "binomial":
            rate = pm.Deterministic("rate", pm.math.invlogit(theta))
            pm.Binomial(
                "obs",
                n=data["trials"],
                p=rate,
                observed=data["counts"],
            )
        elif cfg.likelihood == "logit_normal":
            # theta is on logit scale; rate is bounded [0,1] via invlogit.
            # Observations are logit-transformed, so the Normal likelihood
            # operates on an unbounded scale — no negative rate projections.
            rate = pm.Deterministic("rate", pm.math.invlogit(theta))
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=cfg.sigma_obs_prior)
            pm.Normal(
                "obs",
                mu=theta,
                sigma=sigma_obs,
                observed=data["y_obs"],
            )
        else:
            # Normal likelihood (unbounded, for stats not in [0,1])
            rate = pm.Deterministic("rate", theta)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=cfg.sigma_obs_prior)
            pm.Normal(
                "obs",
                mu=rate,
                sigma=sigma_obs,
                observed=data["y_obs"],
            )

    return model


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
    stat = data["stat"]
    model = build_hitter_model(data, random_seed=random_seed)

    with model:
        logger.info(
            "Sampling %s model: %d draws, %d tune, %d chains, "
            "%d players, %d seasons",
            stat, draws, tune, chains, data["n_players"], data["n_seasons"],
        )

        try:
            import nutpie
            logger.info("Using nutpie sampler (Rust backend)")
            compiled = nutpie.compile_pymc_model(model)
            trace = nutpie.sample(
                compiled,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                seed=random_seed,
            )
        except (ImportError, Exception) as e:
            logger.info("nutpie unavailable (%s), falling back to PyMC NUTS", e)
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
            )

        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

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

    rate_post = trace.posterior["rate"].values  # (chains, draws, obs)
    rate_flat = rate_post.reshape(-1, rate_post.shape[-1])

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
            f"{stat}_mean": float(np.mean(samples)),
            f"{stat}_sd": float(np.std(samples)),
            f"{stat}_2_5": float(np.percentile(samples, 2.5)),
            f"{stat}_25": float(np.percentile(samples, 25)),
            f"{stat}_50": float(np.percentile(samples, 50)),
            f"{stat}_75": float(np.percentile(samples, 75)),
            f"{stat}_97_5": float(np.percentile(samples, 97.5)),
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
    stat = data["stat"]
    cfg = STAT_CONFIGS[stat]
    df = data["df"]

    mask = (df["batter_id"] == batter_id) & (df["season"] == season)
    positions = df.index[mask].tolist()
    if not positions:
        raise ValueError(f"Batter {batter_id} not found in season {season}")

    pos = positions[0]
    iloc_pos = df.index.get_loc(pos)

    rate_post = trace.posterior["rate"].values
    rate_flat = rate_post.reshape(-1, rate_post.shape[-1])
    samples = rate_flat[:, iloc_pos].copy()

    if project_forward and "sigma_season" in trace.posterior:
        rng = np.random.default_rng(random_seed)

        # --- Joint posterior sampling ---
        # All posterior variables must use aligned (chain, draw) indices
        # to preserve correlations (especially alpha ↔ rate negative
        # correlation from partial pooling). Independent resampling
        # artificially inflates deviation_last variance → wider CIs.
        n_target = len(samples)
        sigma_samples = trace.posterior["sigma_season"].values.flatten()
        sigma_samples = np.maximum(sigma_samples, cfg.sigma_season_floor)
        n_posterior = len(sigma_samples)

        if n_posterior != n_target:
            # Sizes differ — resample all variables with a SINGLE shared index
            shared_idx = rng.choice(n_posterior, size=n_target, replace=True)
            sigma_draws = sigma_samples[shared_idx]
        else:
            shared_idx = None
            sigma_draws = sigma_samples

        # Get rho for AR(2) dampening
        if "rho" in trace.posterior:
            rho_samples = trace.posterior["rho"].values.flatten()
            rho_draws = rho_samples[shared_idx] if shared_idx is not None else rho_samples
        else:
            rho_draws = np.ones(n_target)  # fallback: pure random walk

        # Get rho2 (AR(2) lag-2 coefficient) from trace
        if "rho2" in trace.posterior:
            rho2_samples = trace.posterior["rho2"].values.flatten()
            rho2_draws = rho2_samples[shared_idx] if shared_idx is not None else rho2_samples
        else:
            rho2_draws = np.zeros(n_target)  # fallback: AR(1) only

        # Extract alpha (player intercept) — aligned with rate samples
        alpha_post = trace.posterior["alpha"].values
        alpha_flat = alpha_post.reshape(-1, alpha_post.shape[-1])
        pidx = data["player_map"][batter_id]
        alpha_draws = alpha_flat[:, pidx]
        if shared_idx is not None:
            alpha_draws = alpha_draws[shared_idx]

        innovation = rng.normal(0, sigma_draws)

        # Age-dependent persistence: young players' improvements are more
        # likely real development (higher rho = less regression), while
        # aging players' outlier seasons are more likely to revert.
        # Research: peak age 26-29, accelerating decline after 33.
        # Stronger multiplier replaces the old post-hoc alpha blending,
        # keeping everything within the AR(2) framework for consistency
        # between convergence diagnostics and projections.
        row_data = df.loc[pos]
        age = row_data.get("age", None)
        if age is not None:
            if age <= 27:
                # Development phase: breakouts persist more.
                # Scale: age 21 → ×1.39, age 25 → ×1.13, age 27 → ×1.00
                age_rho_mult = 1.0 + (27 - age) * 0.065
            elif age <= 32:
                # Prime: standard persistence
                age_rho_mult = 1.0
            else:
                # Aging: career-bests regress harder.
                # Scale: age 33 → ×0.92, age 35 → ×0.84, age 38 → ×0.72
                age_rho_mult = max(0.70, 1.0 - (age - 32) * 0.05)
            effective_rho = np.clip(rho_draws * age_rho_mult, 0, 0.99)
        else:
            effective_rho = rho_draws

        # Enforce AR(2) stationarity: rho + rho2 < 1 required for
        # mean-reverting projections. Without this, high-rho elite players
        # can get diverging forward projections.
        rho2_draws = np.minimum(rho2_draws, 0.98 - effective_rho)

        # --- Compute deviation_prev (lag-2) for AR(2) forward projection ---
        # deviation_last = samples(t-1) - alpha  (computed below)
        # deviation_prev = samples(t-2) - alpha  (second-to-last season)
        # If player has only 1 observed season, deviation_prev = 0.
        prev_season = season - 1
        prev_mask = (df["batter_id"] == batter_id) & (df["season"] == prev_season)
        prev_positions = df.index[prev_mask].tolist()

        if cfg.likelihood in ("binomial", "logit_normal"):
            # Both binomial and logit_normal operate on logit scale
            eps = np.clip(samples, 1e-6, 1 - 1e-6)
            logit_samples = np.log(eps / (1 - eps))
            deviation_last = logit_samples - alpha_draws

            if prev_positions:
                prev_iloc = df.index.get_loc(prev_positions[0])
                prev_samples = rate_flat[:, prev_iloc].copy()
                if shared_idx is not None:
                    prev_samples = prev_samples[shared_idx]
                eps_prev = np.clip(prev_samples, 1e-6, 1 - 1e-6)
                logit_prev = np.log(eps_prev / (1 - eps_prev))
                deviation_prev = logit_prev - alpha_draws
            else:
                # No lag-2 data: zero out rho2 entirely so it doesn't
                # constrain effective_rho via the stationarity clip
                deviation_prev = np.zeros_like(deviation_last)
                rho2_draws = np.zeros_like(rho2_draws)

            new_deviation = (effective_rho * deviation_last
                             + rho2_draws * deviation_prev + innovation)
            samples = 1.0 / (1.0 + np.exp(-(alpha_draws + new_deviation)))
        else:
            deviation_last = samples - alpha_draws

            if prev_positions:
                prev_iloc = df.index.get_loc(prev_positions[0])
                prev_samples = rate_flat[:, prev_iloc].copy()
                if shared_idx is not None:
                    prev_samples = prev_samples[shared_idx]
                deviation_prev = prev_samples - alpha_draws
            else:
                rho2_draws = np.zeros_like(rho2_draws)
                deviation_prev = np.zeros_like(deviation_last)

            new_deviation = (effective_rho * deviation_last
                             + rho2_draws * deviation_prev + innovation)
            samples = alpha_draws + new_deviation

    return samples


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
    cfg = STAT_CONFIGS[stat]
    var_names = ["mu_pop", "sigma_player", "sigma_season"]
    if "rho" in trace.posterior:
        var_names.append("rho")
    if "rho2" in trace.posterior:
        var_names.append("rho2")
    for col_name, _, _, _ in cfg.covariates:
        var_names.append(f"beta_{col_name}")
    if cfg.likelihood == "normal":
        var_names.append("sigma_obs")

    summary = az.summary(trace, var_names=var_names)
    n_divergences = int(trace.sample_stats["diverging"].sum())
    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())

    result = {
        "n_divergences": n_divergences,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "converged": max_rhat < 1.05 and n_divergences == 0 and min_ess_bulk > 400,
        "summary": summary,
    }

    logger.info(
        "%s convergence: r_hat_max=%.4f, ESS_min=%d, divergences=%d → %s",
        stat, max_rhat, min_ess_bulk, n_divergences,
        "OK" if result["converged"] else "ISSUES DETECTED",
    )
    return result
