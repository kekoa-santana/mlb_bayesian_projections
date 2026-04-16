"""Byte-identity check for the merged StatConfig.

After merging ``hitter_model.StatConfig`` and ``pitcher_model.PitcherStatConfig``
into a single ``StatConfig`` in ``_model_utils.py``, every pre-existing
config in ``STAT_CONFIGS`` and ``PITCHER_STAT_CONFIGS`` must retain all of
its original field values — no default-change drift allowed.
"""
from __future__ import annotations

from src.models._model_utils import StatConfig
from src.models.hitter_model import STAT_CONFIGS
from src.models.hitter_model import StatConfig as HitterReexport
from src.models.pitcher_model import PITCHER_STAT_CONFIGS, PitcherStatConfig


def test_hitter_and_pitcher_point_to_same_dataclass():
    assert HitterReexport is StatConfig
    assert PitcherStatConfig is StatConfig


HITTER_EXPECTED = {
    "k_rate": dict(
        count_col="k", trials_col="pa", rate_col="k_rate",
        likelihood="binomial",
        sigma_season_mu=0.20, sigma_season_floor=0.15,
        sigma_player_prior=0.5, sigma_obs_prior=0.05,
        rho_alpha=9.0, rho_beta=1.5,
        rho2_alpha=2.0, rho2_beta=8.0,
        mu_pop_sigma=0.3, alpha_prior_nu=None,
        season_decay=1.0, use_beta_binomial=False,
    ),
}

PITCHER_EXPECTED = {
    "k_rate": dict(
        count_col="k", trials_col="batters_faced", rate_col="k_rate",
        likelihood="binomial",
        covariates=[], season_decay=1.0,
        sigma_season_mu=0.22, sigma_season_floor=0.15,
        sigma_player_prior=0.5, sigma_obs_prior=0.05,
        rho_alpha=6.0, rho_beta=2.5,
        rho2_alpha=2.0, rho2_beta=8.0,
        mu_pop_sigma=0.3, alpha_prior_nu=None,
        use_beta_binomial=True,
    ),
    "hr_per_bf": dict(
        count_col="hr", trials_col="batters_faced", rate_col="hr_per_bf",
        likelihood="binomial",
        sigma_player_prior=0.8,
        sigma_season_mu=0.40, sigma_season_floor=0.30,
        rho_alpha=6.0, rho_beta=3.0,
        rho2_alpha=1.5, rho2_beta=8.5,
        use_beta_binomial=False,
    ),
}


def test_hitter_k_rate_preserved():
    cfg = STAT_CONFIGS["k_rate"]
    for key, val in HITTER_EXPECTED["k_rate"].items():
        assert getattr(cfg, key) == val, f"hitter k_rate.{key}: {getattr(cfg, key)} != {val}"


def test_pitcher_k_rate_preserved():
    cfg = PITCHER_STAT_CONFIGS["k_rate"]
    for key, val in PITCHER_EXPECTED["k_rate"].items():
        assert getattr(cfg, key) == val, f"pitcher k_rate.{key}: {getattr(cfg, key)} != {val}"


def test_pitcher_hr_per_bf_preserved():
    cfg = PITCHER_STAT_CONFIGS["hr_per_bf"]
    for key, val in PITCHER_EXPECTED["hr_per_bf"].items():
        assert getattr(cfg, key) == val, f"pitcher hr_per_bf.{key}: {getattr(cfg, key)} != {val}"


def test_all_hitter_configs_have_no_op_pitcher_fields():
    """Hitter configs get season_decay/use_beta_binomial via merge; must be no-op defaults."""
    for stat, cfg in STAT_CONFIGS.items():
        assert cfg.season_decay == 1.0, f"hitter {stat} season_decay drifted"
        assert cfg.use_beta_binomial is False, f"hitter {stat} use_beta_binomial drifted"


def test_all_pitcher_configs_have_sigma_obs_prior_default():
    """Pitcher configs gain sigma_obs_prior via merge; must be the 0.05 default."""
    for stat, cfg in PITCHER_STAT_CONFIGS.items():
        assert cfg.sigma_obs_prior == 0.05, f"pitcher {stat} sigma_obs_prior drifted"
