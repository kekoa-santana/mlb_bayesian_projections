"""Tests for the generalized pitcher projection model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.pitcher_model import (
    PITCHER_STAT_CONFIGS,
    build_pitcher_model,
    check_convergence,
    extract_posteriors,
    extract_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)


def _make_synthetic_pitcher_data(
    n_players: int = 30,
    n_seasons: int = 3,
    bf_per_season: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic pitcher data for testing."""
    rng = np.random.default_rng(seed)
    records = []
    for p in range(n_players):
        true_k = rng.beta(6, 20)   # ~0.23
        true_bb = rng.beta(3, 33)  # ~0.08
        true_hr = rng.beta(1, 32)  # ~0.03
        age_base = rng.integers(22, 38)
        is_starter = int(rng.random() > 0.4)
        for s in range(n_seasons):
            bf = bf_per_season + rng.integers(-80, 80)
            k = rng.binomial(bf, true_k)
            bb = rng.binomial(bf, true_bb)
            hr = rng.binomial(bf, true_hr)
            age = age_base + s
            if age <= 25:
                ab = 0
            elif age <= 30:
                ab = 1
            else:
                ab = 2
            records.append({
                "pitcher_id": 200000 + p,
                "pitcher_name": f"Pitcher, Test{p}",
                "pitch_hand": rng.choice(["L", "R"]),
                "season": 2022 + s,
                "age": age,
                "age_bucket": ab,
                "games": 30 + rng.integers(-5, 5),
                "ip": 180.0 if is_starter else 60.0,
                "batters_faced": bf,
                "k": k,
                "bb": bb,
                "hr": hr,
                "k_rate": round(k / bf, 4),
                "bb_rate": round(bb / bf, 4),
                "hr_per_bf": round(hr / bf, 4),
                "hr_per_9": round(hr / (180 if is_starter else 60) * 9, 2),
                "is_starter": is_starter,
                "whiff_rate": float(rng.beta(5, 15)),
                "barrel_rate_against": float(rng.beta(2, 30)),
            })
    return pd.DataFrame(records)


class TestPrepareData:
    """Test data preparation for all stat types."""

    def test_prepare_k_rate(self):
        df = _make_synthetic_pitcher_data(n_players=10, n_seasons=2)
        data = prepare_pitcher_data(df, "k_rate")
        assert data["n_players"] == 10
        assert data["n_seasons"] == 2
        assert "trials" in data
        assert "counts" in data
        assert "is_starter" in data

    def test_prepare_all_stats(self):
        df = _make_synthetic_pitcher_data(n_players=10, n_seasons=2)
        for stat in PITCHER_STAT_CONFIGS:
            data = prepare_pitcher_data(df, stat)
            assert data["stat"] == stat
            assert data["n_players"] == 10

    def test_age_buckets(self):
        df = _make_synthetic_pitcher_data(n_players=10, n_seasons=2)
        data = prepare_pitcher_data(df, "k_rate")
        assert len(data["player_age_bucket"]) == 10
        assert all(b in (0, 1, 2) for b in data["player_age_bucket"])


class TestBuildModel:
    """Test model construction (no sampling)."""

    def test_build_model(self):
        df = _make_synthetic_pitcher_data(n_players=10, n_seasons=2)
        data = prepare_pitcher_data(df, "k_rate")
        model = build_pitcher_model(data)
        assert "mu_pop" in model.named_vars
        assert "sigma_player" in model.named_vars
        assert "sigma_season" in model.named_vars
        assert "rate" in model.named_vars
        assert "beta_starter" in model.named_vars
        # mu_pop has shape (N_AGE_BUCKETS, N_SKILL_TIERS) = (3, 4)
        assert model.named_vars["mu_pop"].eval().shape == (3, 4)


class TestFitAndExtract:
    """Test model fitting with minimal draws."""

    @pytest.fixture(scope="class")
    def fitted_k_rate(self):
        df = _make_synthetic_pitcher_data(n_players=15, n_seasons=2)
        data = prepare_pitcher_data(df, "k_rate")
        model, trace = fit_pitcher_model(
            data, draws=200, tune=100, chains=2, random_seed=42,
        )
        return data, trace

    @pytest.fixture(scope="class")
    def fitted_bb_rate(self):
        df = _make_synthetic_pitcher_data(n_players=15, n_seasons=2)
        data = prepare_pitcher_data(df, "bb_rate")
        model, trace = fit_pitcher_model(
            data, draws=200, tune=100, chains=2, random_seed=42,
        )
        return data, trace

    def test_k_rate_posteriors(self, fitted_k_rate):
        data, trace = fitted_k_rate
        posteriors = extract_posteriors(trace, data)
        assert len(posteriors) == len(data["df"])
        assert "k_rate_mean" in posteriors.columns
        assert posteriors["k_rate_mean"].between(0, 1).all()

    def test_bb_rate_posteriors(self, fitted_bb_rate):
        data, trace = fitted_bb_rate
        posteriors = extract_posteriors(trace, data)
        assert "bb_rate_mean" in posteriors.columns
        assert posteriors["bb_rate_mean"].between(0, 1).all()

    def test_extract_samples(self, fitted_k_rate):
        data, trace = fitted_k_rate
        pid = data["player_ids"][0]
        season = data["df"]["season"].max()
        samples = extract_rate_samples(trace, data, pid, season,
                                       project_forward=False)
        assert len(samples) > 0
        assert all(0 < s < 1 for s in samples)

    def test_extract_samples_forward(self, fitted_k_rate):
        data, trace = fitted_k_rate
        pid = data["player_ids"][0]
        season = data["df"]["season"].max()
        base = extract_rate_samples(trace, data, pid, season,
                                    project_forward=False)
        fwd = extract_rate_samples(trace, data, pid, season,
                                   project_forward=True)
        assert np.std(fwd) >= np.std(base) * 0.9

    def test_convergence(self, fitted_k_rate):
        data, trace = fitted_k_rate
        result = check_convergence(trace, "k_rate")
        assert "converged" in result
        assert "max_rhat" in result
        assert "min_ess_bulk" in result

    def test_age_bucket_mu_pop(self, fitted_k_rate):
        """mu_pop should have shape (N_AGE_BUCKETS, N_SKILL_TIERS) = (3, 4)."""
        _, trace = fitted_k_rate
        mu_pop = trace.posterior["mu_pop"].values
        # shape: (chains, draws, 3, 4)
        assert mu_pop.shape[-2:] == (3, 4)
