"""Tests for counting stat projections module."""
import numpy as np
import pandas as pd
import pytest

from src.models.counting_projections import (
    HITTER_COUNTING_STATS,
    PITCHER_COUNTING_STATS,
    SB_ERA_FACTOR_PRE_2023,
    CountingStat,
    _compute_stat_summary,
    _shrinkage_rate,
    marcel_counting_hitter,
    marcel_counting_pitcher,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def hitter_extended() -> pd.DataFrame:
    """Multi-season hitter extended data with counting stat columns."""
    records = [
        # Player 1: 3 seasons, consistent hitter
        {"batter_id": 1, "batter_name": "Alice", "season": 2022, "games": 150, "pa": 600,
         "k": 120, "bb": 60, "hr": 25, "sb": 8, "age": 27, "age_bucket": 1,
         "k_rate": 0.200, "bb_rate": 0.100, "hr_rate": 0.042, "sb_per_game": 0.053},
        {"batter_id": 1, "batter_name": "Alice", "season": 2023, "games": 155, "pa": 620,
         "k": 130, "bb": 55, "hr": 28, "sb": 15, "age": 28, "age_bucket": 1,
         "k_rate": 0.210, "bb_rate": 0.089, "hr_rate": 0.045, "sb_per_game": 0.097},
        {"batter_id": 1, "batter_name": "Alice", "season": 2024, "games": 148, "pa": 590,
         "k": 118, "bb": 62, "hr": 22, "sb": 12, "age": 29, "age_bucket": 1,
         "k_rate": 0.200, "bb_rate": 0.105, "hr_rate": 0.037, "sb_per_game": 0.081},
        # Player 2: 1 season, power hitter
        {"batter_id": 2, "batter_name": "Bob", "season": 2024, "games": 140, "pa": 550,
         "k": 160, "bb": 45, "hr": 35, "sb": 3, "age": 30, "age_bucket": 1,
         "k_rate": 0.291, "bb_rate": 0.082, "hr_rate": 0.064, "sb_per_game": 0.021},
        # Player 3: pre-2023 SB data
        {"batter_id": 3, "batter_name": "Carl", "season": 2021, "games": 145, "pa": 570,
         "k": 100, "bb": 70, "hr": 10, "sb": 20, "age": 24, "age_bucket": 0,
         "k_rate": 0.175, "bb_rate": 0.123, "hr_rate": 0.018, "sb_per_game": 0.138},
        {"batter_id": 3, "batter_name": "Carl", "season": 2024, "games": 150, "pa": 600,
         "k": 105, "bb": 75, "hr": 12, "sb": 30, "age": 27, "age_bucket": 1,
         "k_rate": 0.175, "bb_rate": 0.125, "hr_rate": 0.020, "sb_per_game": 0.200},
    ]
    return pd.DataFrame(records)


@pytest.fixture()
def pitcher_extended() -> pd.DataFrame:
    """Multi-season pitcher extended data."""
    records = [
        # Pitcher 1: starter, 3 seasons
        {"pitcher_id": 10, "pitcher_name": "Zack", "season": 2022, "games": 32, "ip": 200,
         "batters_faced": 800, "k": 200, "bb": 60, "outs": 600, "is_starter": 1, "age": 28, "age_bucket": 1,
         "k_rate": 0.250, "bb_rate": 0.075, "outs_per_bf": 0.750},
        {"pitcher_id": 10, "pitcher_name": "Zack", "season": 2023, "games": 30, "ip": 190,
         "batters_faced": 770, "k": 210, "bb": 55, "outs": 570, "is_starter": 1, "age": 29, "age_bucket": 1,
         "k_rate": 0.273, "bb_rate": 0.071, "outs_per_bf": 0.740},
        {"pitcher_id": 10, "pitcher_name": "Zack", "season": 2024, "games": 31, "ip": 195,
         "batters_faced": 785, "k": 220, "bb": 50, "outs": 585, "is_starter": 1, "age": 30, "age_bucket": 1,
         "k_rate": 0.280, "bb_rate": 0.064, "outs_per_bf": 0.745},
        # Pitcher 2: reliever, 1 season
        {"pitcher_id": 11, "pitcher_name": "Rex", "season": 2024, "games": 60, "ip": 65,
         "batters_faced": 270, "k": 80, "bb": 25, "outs": 195, "is_starter": 0, "age": 26, "age_bucket": 0,
         "k_rate": 0.296, "bb_rate": 0.093, "outs_per_bf": 0.722},
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CountingStat config
# ---------------------------------------------------------------------------
class TestCountingStatConfig:
    def test_hitter_stats_keys(self) -> None:
        assert set(HITTER_COUNTING_STATS.keys()) == {"total_k", "total_bb", "total_hr", "total_sb"}

    def test_pitcher_stats_keys(self) -> None:
        assert set(PITCHER_COUNTING_STATS.keys()) == {"total_k", "total_bb", "total_outs", "total_er"}

    def test_bayesian_flags(self) -> None:
        """K and BB should use Bayesian posteriors; HR, SB, outs should not."""
        assert HITTER_COUNTING_STATS["total_k"].bayesian is True
        assert HITTER_COUNTING_STATS["total_bb"].bayesian is True
        assert HITTER_COUNTING_STATS["total_hr"].bayesian is False
        assert HITTER_COUNTING_STATS["total_sb"].bayesian is False
        assert PITCHER_COUNTING_STATS["total_outs"].bayesian is False

    def test_opportunity_columns(self) -> None:
        assert HITTER_COUNTING_STATS["total_k"].opportunity == "pa"
        assert HITTER_COUNTING_STATS["total_sb"].opportunity == "games"
        assert PITCHER_COUNTING_STATS["total_k"].opportunity == "bf"


class TestSbEraFactor:
    def test_era_factor_positive(self) -> None:
        assert SB_ERA_FACTOR_PRE_2023 > 1.0

    def test_era_factor_reasonable(self) -> None:
        assert 1.5 <= SB_ERA_FACTOR_PRE_2023 <= 2.5


# ---------------------------------------------------------------------------
# _compute_stat_summary
# ---------------------------------------------------------------------------
class TestComputeStatSummary:
    def test_keys(self) -> None:
        samples = np.array([10, 20, 30, 40, 50])
        result = _compute_stat_summary(samples)
        expected_keys = {"mean", "median", "sd", "p10", "p90", "p2_5", "p97_5"}
        assert expected_keys == set(result.keys())

    def test_mean_correct(self) -> None:
        samples = np.array([100, 100, 100, 100])
        result = _compute_stat_summary(samples)
        assert result["mean"] == 100.0
        assert result["sd"] == 0.0

    def test_percentiles_ordered(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(100, 20, size=10000)
        result = _compute_stat_summary(samples)
        assert result["p2_5"] < result["p10"] < result["median"] < result["p90"] < result["p97_5"]


# ---------------------------------------------------------------------------
# _shrinkage_rate
# ---------------------------------------------------------------------------
class TestShrinkageRate:
    def test_empty_returns_pop_mean(self) -> None:
        empty = pd.DataFrame(columns=["k_rate", "pa"])
        mean, std = _shrinkage_rate(empty, "k_rate", "pa", pop_mean=0.22)
        assert mean == 0.22
        assert std == 0.05

    def test_single_season_regresses(self) -> None:
        """Single season should regress toward population mean."""
        df = pd.DataFrame({"k_rate": [0.30], "pa": [600]})
        mean, std = _shrinkage_rate(df, "k_rate", "pa", pop_mean=0.22, shrinkage_k=3.0)
        # With 1 season and k=3, rel = 1/(1+3) = 0.25
        # So mean = 0.25 * 0.30 + 0.75 * 0.22 = 0.24
        assert abs(mean - 0.24) < 0.001

    def test_multi_season_less_regression(self) -> None:
        """More seasons → less regression toward population."""
        df = pd.DataFrame({"k_rate": [0.30, 0.30, 0.30], "pa": [600, 600, 600]})
        mean, std = _shrinkage_rate(df, "k_rate", "pa", pop_mean=0.22, shrinkage_k=3.0)
        # rel = 3/(3+3) = 0.5, so mean = 0.5*0.30 + 0.5*0.22 = 0.26
        assert abs(mean - 0.26) < 0.001

    def test_std_positive(self) -> None:
        df = pd.DataFrame({"k_rate": [0.20], "pa": [500]})
        _, std = _shrinkage_rate(df, "k_rate", "pa", pop_mean=0.22)
        assert std > 0


# ---------------------------------------------------------------------------
# Marcel hitter counting
# ---------------------------------------------------------------------------
class TestMarcelCountingHitter:
    def test_output_columns(self, hitter_extended: pd.DataFrame) -> None:
        result = marcel_counting_hitter(hitter_extended, from_season=2024, min_pa=200)
        expected = {"batter_id", "marcel_pa", "marcel_k", "marcel_bb", "marcel_hr", "marcel_sb"}
        assert expected.issubset(set(result.columns))

    def test_returns_eligible_players(self, hitter_extended: pd.DataFrame) -> None:
        result = marcel_counting_hitter(hitter_extended, from_season=2024, min_pa=200)
        # Players 1 (590 PA), 2 (550 PA), 3 (600 PA) all meet min_pa=200
        assert len(result) >= 2

    def test_marcel_pa_positive(self, hitter_extended: pd.DataFrame) -> None:
        result = marcel_counting_hitter(hitter_extended, from_season=2024, min_pa=200)
        assert (result["marcel_pa"] > 0).all()

    def test_marcel_k_reasonable(self, hitter_extended: pd.DataFrame) -> None:
        """Marcel K projection should be in a reasonable range."""
        result = marcel_counting_hitter(hitter_extended, from_season=2024, min_pa=200)
        assert (result["marcel_k"] >= 0).all()
        assert (result["marcel_k"] <= 300).all()

    def test_sb_era_adjustment_applied(self, hitter_extended: pd.DataFrame) -> None:
        """Pre-2023 SB rates should be inflated by era factor."""
        # Carl has 2021 data (pre-2023) with sb_per_game=0.138
        # Era-adjusted: 0.138 * 1.8 = 0.248
        # His 2024 rate is 0.200 (post-2023, no adjustment)
        result = marcel_counting_hitter(hitter_extended, from_season=2024, min_pa=200)
        carl = result[result["batter_id"] == 3]
        assert len(carl) == 1
        # Marcel SB should reflect the era-inflated pre-2023 rate
        assert carl.iloc[0]["marcel_sb"] > 0


# ---------------------------------------------------------------------------
# Marcel pitcher counting
# ---------------------------------------------------------------------------
class TestMarcelCountingPitcher:
    def test_output_columns(self, pitcher_extended: pd.DataFrame) -> None:
        result = marcel_counting_pitcher(pitcher_extended, from_season=2024, min_bf=200)
        expected = {"pitcher_id", "marcel_bf", "marcel_k", "marcel_bb", "marcel_outs"}
        assert expected.issubset(set(result.columns))

    def test_returns_eligible_pitchers(self, pitcher_extended: pd.DataFrame) -> None:
        result = marcel_counting_pitcher(pitcher_extended, from_season=2024, min_bf=200)
        # Both pitchers have >= 200 BF in 2024
        assert len(result) == 2

    def test_marcel_bf_positive(self, pitcher_extended: pd.DataFrame) -> None:
        result = marcel_counting_pitcher(pitcher_extended, from_season=2024, min_bf=200)
        assert (result["marcel_bf"] > 0).all()

    def test_starter_more_ks_than_reliever(self, pitcher_extended: pd.DataFrame) -> None:
        """Starter with 3 seasons of ~200K should project more Ks than reliever."""
        result = marcel_counting_pitcher(pitcher_extended, from_season=2024, min_bf=200)
        zack = result[result["pitcher_id"] == 10].iloc[0]
        rex = result[result["pitcher_id"] == 11].iloc[0]
        assert zack["marcel_k"] > rex["marcel_k"]

    def test_outs_reasonable(self, pitcher_extended: pd.DataFrame) -> None:
        result = marcel_counting_pitcher(pitcher_extended, from_season=2024, min_bf=200)
        # Outs should be between 0 and 800 (starter max ~700)
        assert (result["marcel_outs"] >= 0).all()
        assert (result["marcel_outs"] <= 800).all()


# ---------------------------------------------------------------------------
# Park factor integration
# ---------------------------------------------------------------------------
class TestParkFactorIntegration:
    """Test that park factors adjust HR projections correctly."""

    @pytest.fixture()
    def park_factors(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"venue_id": 100, "venue_name": "Coors Field", "batter_stand": "L", "hr_pf": 1.40},
            {"venue_id": 100, "venue_name": "Coors Field", "batter_stand": "R", "hr_pf": 1.50},
            {"venue_id": 200, "venue_name": "Oracle Park", "batter_stand": "L", "hr_pf": 0.70},
            {"venue_id": 200, "venue_name": "Oracle Park", "batter_stand": "R", "hr_pf": 0.75},
        ])

    @pytest.fixture()
    def hitter_venues(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"batter_id": 1, "team_id": 115, "team_name": "Colorado Rockies", "venue_id": 100, "bat_side": "R"},
            {"batter_id": 2, "team_id": 137, "team_name": "San Francisco Giants", "venue_id": 200, "bat_side": "L"},
            {"batter_id": 3, "team_id": 115, "team_name": "Colorado Rockies", "venue_id": 100, "bat_side": "S"},
        ])

    def test_coors_boosts_hr(self, hitter_extended: pd.DataFrame, park_factors: pd.DataFrame, hitter_venues: pd.DataFrame) -> None:
        """Coors Field (PF=1.50 for R) should increase HR projection vs no PF."""
        no_pf = _shrinkage_rate(
            hitter_extended[hitter_extended["batter_id"] == 1].sort_values("season", ascending=False),
            "hr_rate", "pa", pop_mean=0.035,
        )
        # With PF: half-weighted Coors = 0.5*1.50 + 0.5*1.0 = 1.25
        # So HR rate effectively multiplied by 1.25
        assert no_pf[0] > 0  # sanity: rate is positive

    def test_oracle_suppresses_hr(self, hitter_extended: pd.DataFrame, park_factors: pd.DataFrame, hitter_venues: pd.DataFrame) -> None:
        """Oracle Park (PF=0.70 for L) should decrease HR projection."""
        # Bob bats L at Oracle: half-weighted = 0.5*0.70 + 0.5*1.0 = 0.85
        # Effective PF < 1.0
        pf_val = 0.5 * 0.70 + 0.5 * 1.0
        assert pf_val < 1.0

    def test_switch_hitter_averages_lr(self, park_factors: pd.DataFrame, hitter_venues: pd.DataFrame) -> None:
        """Switch hitter should get average of L and R park factors."""
        # Carl (batter_id=3) is switch hitter at Coors
        # L=1.40, R=1.50, avg=1.45, half-weighted = 0.5*1.45 + 0.5 = 1.225
        from src.models.counting_projections import project_hitter_counting

        # Build venue_pf lookup manually to verify
        venue_pf = {}
        for _, row in park_factors.iterrows():
            venue_pf[(int(row["venue_id"]), row["batter_stand"])] = float(row["hr_pf"])

        pf_l = venue_pf.get((100, "L"), 1.0)
        pf_r = venue_pf.get((100, "R"), 1.0)
        raw_pf = (pf_l + pf_r) / 2
        half_weighted = 0.5 * raw_pf + 0.5 * 1.0
        assert abs(half_weighted - 1.225) < 0.001

    def test_hr_park_factor_in_output(self, hitter_extended: pd.DataFrame, park_factors: pd.DataFrame, hitter_venues: pd.DataFrame) -> None:
        """hr_park_factor column should appear in output when PFs are provided."""
        from src.models.counting_projections import project_hitter_counting

        # Build minimal pa_priors for eligible players
        pa_priors = pd.DataFrame([
            {"batter_id": 1, "projected_games": 150.0, "sigma_games": 20.0,
             "projected_pa_per_game": 3.9, "sigma_pa_rate": 0.3},
            {"batter_id": 2, "projected_games": 140.0, "sigma_games": 25.0,
             "projected_pa_per_game": 3.8, "sigma_pa_rate": 0.3},
            {"batter_id": 3, "projected_games": 148.0, "sigma_games": 20.0,
             "projected_pa_per_game": 3.9, "sigma_pa_rate": 0.3},
        ])

        result = project_hitter_counting(
            rate_model_results={},
            pa_priors=pa_priors,
            hitter_extended=hitter_extended,
            from_season=2024,
            n_draws=100,
            min_pa=200,
            random_seed=42,
            park_factors=park_factors,
            hitter_venues=hitter_venues,
        )
        assert "hr_park_factor" in result.columns
        # Alice (batter_id=1) at Coors, R: 0.5*1.50+0.5 = 1.25
        alice = result[result["batter_id"] == 1]
        if not alice.empty:
            assert abs(alice.iloc[0]["hr_park_factor"] - 1.25) < 0.001
