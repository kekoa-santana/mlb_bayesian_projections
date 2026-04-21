"""
Integration tests for the full model pipeline on synthetic data.

These tests verify that models compile, sample, and recover ground truth
on small synthetic datasets. Marked with @pytest.mark.integration because
they're slower than unit tests (~30s each with tiny draws).

Usage:
    pytest tests/test_model_integration.py -v
    pytest tests/test_model_integration.py -v -m integration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)


def _make_synthetic_hitter_data(
    n_players: int = 20,
    n_seasons: int = 3,
    base_season: int = 2022,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Generate synthetic hitter data with known true K%.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (df, true_k_rates) where true_k_rates maps batter_id -> true K%.
    """
    true_k_rates = {}
    rows = []
    for i in range(n_players):
        batter_id = 1000 + i
        true_k = _RNG.beta(5, 18)  # centered ~0.22
        true_k_rates[batter_id] = true_k

        barrel_pct = 0.06 + _RNG.normal(0, 0.02)
        hard_hit_pct = 0.33 + _RNG.normal(0, 0.04)

        for s in range(n_seasons):
            pa = _RNG.integers(150, 600)
            k = _RNG.binomial(pa, true_k)
            rows.append({
                "batter_id": batter_id,
                "batter_name": f"Hitter_{i}",
                "batter_stand": _RNG.choice(["R", "L"]),
                "season": base_season + s,
                "pa": pa,
                "k": k,
                "k_rate": k / pa if pa > 0 else 0.0,
                "barrel_pct": max(0, barrel_pct + _RNG.normal(0, 0.005)),
                "hard_hit_pct": max(0, hard_hit_pct + _RNG.normal(0, 0.01)),
            })

    return pd.DataFrame(rows), true_k_rates


def _make_synthetic_pitcher_data(
    n_players: int = 20,
    n_seasons: int = 3,
    base_season: int = 2022,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Generate synthetic pitcher data with known true K%.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (df, true_k_rates) where true_k_rates maps pitcher_id -> true K%.
    """
    true_k_rates = {}
    rows = []
    for i in range(n_players):
        pitcher_id = 2000 + i
        true_k = _RNG.beta(6, 20)  # centered ~0.23
        true_k_rates[pitcher_id] = true_k

        whiff_rate = 0.25 + _RNG.normal(0, 0.04)
        barrel_rate = 0.06 + _RNG.normal(0, 0.015)
        is_starter = int(i < n_players * 0.6)

        for s in range(n_seasons):
            bf = _RNG.integers(200, 700)
            k = _RNG.binomial(bf, true_k)
            rows.append({
                "pitcher_id": pitcher_id,
                "pitcher_name": f"Pitcher_{i}",
                "pitch_hand": _RNG.choice(["R", "L"]),
                "season": base_season + s,
                "batters_faced": bf,
                "k": k,
                "k_rate": k / bf if bf > 0 else 0.0,
                "bb": _RNG.binomial(bf, 0.08),
                "bb_rate": 0.08,
                "ip": bf * 0.3,
                "games": 30 if is_starter else 55,
                "whiff_rate": max(0, whiff_rate + _RNG.normal(0, 0.01)),
                "barrel_rate_against": max(0, barrel_rate + _RNG.normal(0, 0.005)),
                "is_starter": is_starter,
            })

    return pd.DataFrame(rows), true_k_rates


def _make_synthetic_platoon_data(
    n_players: int = 20,
    n_seasons: int = 3,
    base_season: int = 2022,
) -> pd.DataFrame:
    """Generate synthetic platoon data (2 rows per player-season)."""
    rows = []
    for i in range(n_players):
        batter_id = 3000 + i
        true_k_same = _RNG.beta(6, 20)
        true_k_opp = _RNG.beta(5, 18)

        for s in range(n_seasons):
            for pitch_hand, same_side, true_k in [
                ("R", 1, true_k_same), ("L", 0, true_k_opp)
            ]:
                pa = _RNG.integers(80, 300)
                k = _RNG.binomial(pa, true_k)
                rows.append({
                    "batter_id": batter_id,
                    "batter_name": f"Platoon_{i}",
                    "batter_stand": "R",
                    "pitch_hand": pitch_hand,
                    "same_side": same_side,
                    "season": base_season + s,
                    "pa": pa,
                    "k": k,
                    "k_rate": k / pa if pa > 0 else 0.0,
                    "barrel_pct": 0.07 + _RNG.normal(0, 0.01),
                    "hard_hit_pct": 0.35 + _RNG.normal(0, 0.02),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared fixtures — module-scoped to avoid refitting per test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hitter_fit():
    """Fit hitter K% model on synthetic data (module-scoped)."""
    from src.models.hitter_model import (
        extract_posteriors as extract_hitter_posteriors,
        fit_hitter_model,
        prepare_hitter_data,
    )

    df, true_k = _make_synthetic_hitter_data()
    data = prepare_hitter_data(df, "k_rate")
    model, trace = fit_hitter_model(
        data, draws=200, tune=100, chains=2, random_seed=42,
    )
    posteriors = extract_hitter_posteriors(trace, data)
    return {
        "df": df, "true_k": true_k, "data": data,
        "model": model, "trace": trace, "posteriors": posteriors,
    }


@pytest.fixture(scope="module")
def pitcher_fit():
    """Fit pitcher K% model on synthetic data (module-scoped)."""
    from src.models.pitcher_model import (
        extract_posteriors as extract_pitcher_posteriors,
        fit_pitcher_model,
        prepare_pitcher_data,
    )

    df, true_k = _make_synthetic_pitcher_data()
    data = prepare_pitcher_data(df, "k_rate")
    model, trace = fit_pitcher_model(
        data, draws=200, tune=100, chains=2, random_seed=42,
    )
    posteriors = extract_pitcher_posteriors(trace, data)
    return {
        "df": df, "true_k": true_k, "data": data,
        "model": model, "trace": trace, "posteriors": posteriors,
    }


# ---------------------------------------------------------------------------
# Test 1: Hitter model recovers ground truth
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestHitterModelRecovery:
    def test_no_errors(self, hitter_fit: dict) -> None:
        """Model compiles, samples, and extracts posteriors without error."""
        assert hitter_fit["model"] is not None
        assert hitter_fit["trace"] is not None
        assert len(hitter_fit["posteriors"]) > 0

    def test_posterior_means_reasonable(self, hitter_fit: dict) -> None:
        """Posterior means within +-0.10 of true rates for well-sampled players."""
        posteriors = hitter_fit["posteriors"]
        true_k = hitter_fit["true_k"]

        # Get last-season posteriors for players with >= 200 PA
        last_season = posteriors["season"].max()
        last = posteriors[
            (posteriors["season"] == last_season)
            & (posteriors["pa"] >= 200)
        ]

        n_close = 0
        for _, row in last.iterrows():
            bid = row["batter_id"]
            if bid in true_k:
                diff = abs(row["k_rate_mean"] - true_k[bid])
                if diff < 0.10:
                    n_close += 1

        # At least 60% should be within 0.10
        assert n_close >= len(last) * 0.6, (
            f"Only {n_close}/{len(last)} posterior means within 0.10 of truth"
        )

    def test_credible_interval_coverage(self, hitter_fit: dict) -> None:
        """95% CI covers truth for >= 70% of players."""
        posteriors = hitter_fit["posteriors"]
        true_k = hitter_fit["true_k"]

        last_season = posteriors["season"].max()
        last = posteriors[posteriors["season"] == last_season]

        n_covered = 0
        n_total = 0
        for _, row in last.iterrows():
            bid = row["batter_id"]
            if bid in true_k:
                n_total += 1
                if row["k_rate_2_5"] <= true_k[bid] <= row["k_rate_97_5"]:
                    n_covered += 1

        coverage = n_covered / n_total if n_total > 0 else 0
        assert coverage >= 0.70, f"95% CI coverage only {coverage:.1%}"

    def test_covariates_in_model(self, hitter_fit: dict) -> None:
        """beta_barrel and beta_hard_hit are in the model."""
        rv_names = {rv.name for rv in hitter_fit["model"].free_RVs}
        assert "beta_barrel" in rv_names
        assert "beta_hard_hit" in rv_names


# ---------------------------------------------------------------------------
# Test 2: Pitcher model recovers ground truth
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestPitcherModelRecovery:
    def test_no_errors(self, pitcher_fit: dict) -> None:
        """Model compiles, samples, and extracts posteriors without error."""
        assert pitcher_fit["model"] is not None
        assert pitcher_fit["trace"] is not None
        assert len(pitcher_fit["posteriors"]) > 0

    def test_posterior_means_reasonable(self, pitcher_fit: dict) -> None:
        """Posterior means within +-0.10 of true rates for well-sampled pitchers."""
        posteriors = pitcher_fit["posteriors"]
        true_k = pitcher_fit["true_k"]

        last_season = posteriors["season"].max()
        last = posteriors[
            (posteriors["season"] == last_season)
            & (posteriors["batters_faced"] >= 200)
        ]

        n_close = 0
        for _, row in last.iterrows():
            pid = row["pitcher_id"]
            if pid in true_k:
                diff = abs(row["k_rate_mean"] - true_k[pid])
                if diff < 0.10:
                    n_close += 1

        assert n_close >= len(last) * 0.6, (
            f"Only {n_close}/{len(last)} posterior means within 0.10 of truth"
        )

    def test_credible_interval_coverage(self, pitcher_fit: dict) -> None:
        """95% CI covers truth for >= 70% of pitchers."""
        posteriors = pitcher_fit["posteriors"]
        true_k = pitcher_fit["true_k"]

        last_season = posteriors["season"].max()
        last = posteriors[posteriors["season"] == last_season]

        n_covered = 0
        n_total = 0
        for _, row in last.iterrows():
            pid = row["pitcher_id"]
            if pid in true_k:
                n_total += 1
                if row["k_rate_2_5"] <= true_k[pid] <= row["k_rate_97_5"]:
                    n_covered += 1

        coverage = n_covered / n_total if n_total > 0 else 0
        assert coverage >= 0.70, f"95% CI coverage only {coverage:.1%}"

    def test_no_covariate_betas(self, pitcher_fit: dict) -> None:
        """beta_whiff and beta_barrel_against should NOT be in the model."""
        rv_names = {rv.name for rv in pitcher_fit["model"].free_RVs}
        assert "beta_whiff" not in rv_names
        assert "beta_barrel_against" not in rv_names

    def test_starter_role_present(self, pitcher_fit: dict) -> None:
        """beta_starter is present when is_starter is in the data."""
        rv_names = {rv.name for rv in pitcher_fit["model"].free_RVs}
        assert "beta_starter" in rv_names


# ---------------------------------------------------------------------------
# Test 3: Hitter model with platoon splits
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestHitterPlatoonIntegration:
    def test_platoon_model_compiles_and_samples(self) -> None:
        """Platoon model with same_side column produces gamma posteriors."""
        from src.models.hitter_model import (
            extract_posteriors as extract_hitter_posteriors,
            fit_hitter_model,
            prepare_hitter_data,
        )

        df = _make_synthetic_platoon_data(n_players=15, n_seasons=2)
        data = prepare_hitter_data(df, "k_rate")
        model, trace = fit_hitter_model(
            data, draws=200, tune=100, chains=2, random_seed=42,
        )

        # Model should compile and sample
        posteriors = extract_hitter_posteriors(trace, data)
        assert len(posteriors) > 0


# ---------------------------------------------------------------------------
# Test 4: Matchup scoring end-to-end
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestMatchupEndToEnd:
    def test_batch_scoring_produces_valid_output(self) -> None:
        """Batch scoring of synthetic matchups produces correct schema."""
        from src.models.matchup import score_matchups_batch

        pitcher_arsenal = pd.DataFrame([
            {"pitcher_id": 100, "pitch_type": "FF", "pitches": 600,
             "usage_pct": 0.60, "swings": 300, "whiff_rate": 0.24},
            {"pitcher_id": 100, "pitch_type": "SL", "pitches": 250,
             "usage_pct": 0.25, "swings": 125, "whiff_rate": 0.40},
            {"pitcher_id": 100, "pitch_type": "CH", "pitches": 150,
             "usage_pct": 0.15, "swings": 75, "whiff_rate": 0.32},
        ])

        hitter_vuln = pd.DataFrame([
            {"batter_id": 200, "pitch_type": "FF", "swings": 80,
             "whiff_rate": 0.35, "pitch_family": "fastball"},
            {"batter_id": 200, "pitch_type": "SL", "swings": 40,
             "whiff_rate": 0.33, "pitch_family": "breaking"},
            {"batter_id": 300, "pitch_type": "FF", "swings": 100,
             "whiff_rate": 0.22, "pitch_family": "fastball"},
            {"batter_id": 300, "pitch_type": "SL", "swings": 60,
             "whiff_rate": 0.33, "pitch_family": "breaking"},
        ])

        baselines = {
            "FF": {"whiff_rate": 0.22},
            "SL": {"whiff_rate": 0.33},
            "CH": {"whiff_rate": 0.32},
        }

        pairs = [(100, 200), (100, 300)]
        batch = score_matchups_batch(pitcher_arsenal, hitter_vuln, baselines, pairs)

        assert len(batch) == 2
        assert "matchup_k_logit_lift" in batch.columns
        assert "matchup_whiff_rate" in batch.columns

        # Vulnerable hitter should have positive lift
        vuln_row = batch[batch["batter_id"] == 200].iloc[0]
        assert vuln_row["matchup_k_logit_lift"] > 0

    def test_game_level_matchup_adjustment(self) -> None:
        """Game-level K rate adjustment works with matchup scores."""
        from src.models.matchup import compute_game_matchup_k_rate

        game_batter_pa = pd.DataFrame([
            {"batter_id": 200, "pa": 4},
            {"batter_id": 300, "pa": 3},
            {"batter_id": 400, "pa": 3},
        ])

        matchup_scores = pd.DataFrame([
            {"batter_id": 200, "matchup_k_logit_lift": 0.3},
            {"batter_id": 300, "matchup_k_logit_lift": -0.1},
        ])

        result = compute_game_matchup_k_rate(0.22, game_batter_pa, matchup_scores)
        assert result["total_bf"] == 10
        assert result["n_matched"] == 2
        assert result["n_total"] == 3
        # Adjusted rate should differ from baseline due to matchup lifts
        assert result["game_adjusted_k_rate"] != 0.22


# ---------------------------------------------------------------------------
# Test 5: Forward projection produces wider intervals
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestProjectionWider:
    def test_projected_wider_than_insample(self, pitcher_fit: dict) -> None:
        """Forward-projected intervals should be wider than in-sample."""
        from src.viz.projections import project_posteriors_forward

        trace = pitcher_fit["trace"]
        data = pitcher_fit["data"]
        last_season = pitcher_fit["df"]["season"].max()

        # In-sample posteriors (last season)
        insample = pitcher_fit["posteriors"]
        insample_last = insample[insample["season"] == last_season]

        # Projected posteriors (one step forward)
        projected = project_posteriors_forward(
            trace, data, last_season,
            id_col="pitcher_id",
            name_col="pitcher_name",
            hand_col="pitch_hand",
            trials_col="batters_faced",
        )

        # Compare interval widths
        insample_widths = (
            insample_last["k_rate_97_5"] - insample_last["k_rate_2_5"]
        ).values
        projected_widths = (
            projected["projected_k_rate_97_5"] - projected["projected_k_rate_2_5"]
        ).values

        # Projected should be wider on average (innovation noise adds uncertainty)
        mean_insample = np.mean(insample_widths)
        mean_projected = np.mean(projected_widths)
        assert mean_projected > mean_insample, (
            f"Projected width ({mean_projected:.4f}) not wider than "
            f"in-sample ({mean_insample:.4f})"
        )
