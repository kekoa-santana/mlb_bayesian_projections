"""Tests for the league baselines v2 module.

All tests monkeypatch ``get_pitcher_outcomes_by_stand`` and
``get_pitch_archetype_offerings`` to avoid DB dependency.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import league_baselines


# ---------------------------------------------------------------------------
# Helpers — fake outcome data
# ---------------------------------------------------------------------------
def _make_outcomes(rows: list[dict]) -> pd.DataFrame:
    """Build a pitcher outcomes DataFrame from a list of row dicts."""
    defaults = {
        "pitcher_id": 1,
        "pitch_type": "FF",
        "batter_stand": "R",
        "pitches": 100,
        "swings": 50,
        "whiffs": 10,
        "out_of_zone_pitches": 40,
        "chase_swings": 12,
        "called_strikes": 15,
        "csw": 25,
        "bip": 30,
        "hard_hits": 10,
        "barrels_proxy": 3,
        "xwoba_contact_sum": 9.0,
        "xwoba_contact_n": 25,
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


def _make_archetypes(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal archetype offerings DataFrame."""
    defaults = {
        "season": 2024,
        "pitcher_id": 1,
        "pitch_hand": "R",
        "pitch_type": "FF",
        "pitch_name": "4-Seam Fastball",
        "pitches": 300,
        "release_speed": 95.0,
        "pfx_x": -0.5,
        "pfx_z": 1.6,
        "release_spin_rate": 2400.0,
        "release_extension": 6.4,
        "release_pos_x": -1.5,
        "release_pos_z": 5.7,
        "pitch_family": "fastball",
        "pfx_x_flipped": False,
        "pitch_archetype": 1,
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


def _patch_sources(monkeypatch, tmp_path, outcomes_by_season, archetypes_by_season):
    """Monkeypatch both data sources and redirect cache dir."""
    monkeypatch.setattr(league_baselines, "CACHE_DIR", tmp_path)

    def _fake_outcomes(season):
        return outcomes_by_season.get(season, pd.DataFrame()).copy()

    def _fake_archetypes(season, n_clusters=8, force_rebuild=False):
        return archetypes_by_season.get(season, pd.DataFrame()).copy()

    monkeypatch.setattr(league_baselines, "get_pitcher_outcomes_by_stand", _fake_outcomes)
    monkeypatch.setattr(league_baselines, "get_pitch_archetype_offerings", _fake_archetypes)


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------
OUTCOMES_2024 = _make_outcomes([
    # Pitcher 1: FF vs R and L
    {"pitcher_id": 1, "pitch_type": "FF", "batter_stand": "R",
     "pitches": 200, "swings": 100, "whiffs": 20, "out_of_zone_pitches": 80,
     "chase_swings": 24, "called_strikes": 30, "csw": 50, "bip": 60,
     "hard_hits": 20, "barrels_proxy": 6, "xwoba_contact_sum": 18.0, "xwoba_contact_n": 50},
    {"pitcher_id": 1, "pitch_type": "FF", "batter_stand": "L",
     "pitches": 150, "swings": 80, "whiffs": 18, "out_of_zone_pitches": 60,
     "chase_swings": 20, "called_strikes": 25, "csw": 43, "bip": 45,
     "hard_hits": 16, "barrels_proxy": 5, "xwoba_contact_sum": 14.0, "xwoba_contact_n": 40},
    # Pitcher 1: SL vs R and L
    {"pitcher_id": 1, "pitch_type": "SL", "batter_stand": "R",
     "pitches": 100, "swings": 60, "whiffs": 20, "out_of_zone_pitches": 50,
     "chase_swings": 18, "called_strikes": 12, "csw": 32, "bip": 25,
     "hard_hits": 6, "barrels_proxy": 2, "xwoba_contact_sum": 5.0, "xwoba_contact_n": 20},
    {"pitcher_id": 1, "pitch_type": "SL", "batter_stand": "L",
     "pitches": 80, "swings": 45, "whiffs": 12, "out_of_zone_pitches": 35,
     "chase_swings": 10, "called_strikes": 10, "csw": 22, "bip": 20,
     "hard_hits": 5, "barrels_proxy": 1, "xwoba_contact_sum": 4.0, "xwoba_contact_n": 15},
    # Pitcher 2: FF vs R only
    {"pitcher_id": 2, "pitch_type": "FF", "batter_stand": "R",
     "pitches": 180, "swings": 90, "whiffs": 22, "out_of_zone_pitches": 70,
     "chase_swings": 21, "called_strikes": 28, "csw": 50, "bip": 55,
     "hard_hits": 18, "barrels_proxy": 5, "xwoba_contact_sum": 16.0, "xwoba_contact_n": 45},
])

ARCHETYPES_2024 = _make_archetypes([
    {"pitcher_id": 1, "pitch_type": "FF", "pitch_archetype": 1},
    {"pitcher_id": 1, "pitch_type": "SL", "pitch_archetype": 3},
    {"pitcher_id": 2, "pitch_type": "FF", "pitch_archetype": 1},
    # Note: pitcher 2 has no SL archetype — that's fine, he only throws FF
])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAggregateBaselinesComputesRates:
    def test_rate_math(self):
        """whiff_rate = whiffs/swings, etc."""
        df = _make_outcomes([
            {"pitches": 100, "swings": 50, "whiffs": 15,
             "out_of_zone_pitches": 40, "chase_swings": 12,
             "called_strikes": 20, "csw": 35, "bip": 30,
             "hard_hits": 10, "barrels_proxy": 3,
             "xwoba_contact_sum": 9.0, "xwoba_contact_n": 25},
        ])
        agg = league_baselines._aggregate_baselines(df, ["pitch_type"])

        row = agg.iloc[0]
        assert row["whiff_rate"] == pytest.approx(15 / 50)
        assert row["chase_rate"] == pytest.approx(12 / 40)
        assert row["csw_pct"] == pytest.approx(35 / 100)
        assert row["barrel_rate"] == pytest.approx(3 / 30)
        assert row["hard_hit_rate"] == pytest.approx(10 / 30)
        assert row["xwoba_contact"] == pytest.approx(9.0 / 25)


class TestAggregateBaselinesHandlesZeroDenominator:
    def test_zero_swings_gives_nan(self):
        """0 swings -> NaN whiff_rate, not error."""
        df = _make_outcomes([
            {"pitches": 10, "swings": 0, "whiffs": 0,
             "out_of_zone_pitches": 0, "chase_swings": 0,
             "called_strikes": 5, "csw": 5, "bip": 0,
             "hard_hits": 0, "barrels_proxy": 0,
             "xwoba_contact_sum": 0, "xwoba_contact_n": 0},
        ])
        agg = league_baselines._aggregate_baselines(df, ["pitch_type"])

        row = agg.iloc[0]
        assert np.isnan(row["whiff_rate"])
        assert np.isnan(row["chase_rate"])
        assert np.isnan(row["barrel_rate"])
        assert np.isnan(row["hard_hit_rate"])
        assert np.isnan(row["xwoba_contact"])
        # csw_pct should still work (pitches > 0)
        assert row["csw_pct"] == pytest.approx(5 / 10)


class TestArchetypeJoinLeftJoinPreservesUnmatched:
    def test_unmatched_rows_have_nan_archetype(self, monkeypatch, tmp_path):
        """Rows without archetype still appear with pitch_archetype=NaN."""
        outcomes = _make_outcomes([
            {"pitcher_id": 10, "pitch_type": "CH", "batter_stand": "R",
             "pitches": 50, "swings": 25, "whiffs": 8},
        ])
        # No archetype for pitcher 10 / CH
        archetypes = _make_archetypes([
            {"pitcher_id": 99, "pitch_type": "FF", "pitch_archetype": 1},
        ])

        _patch_sources(monkeypatch, tmp_path, {2024: outcomes}, {2024: archetypes})

        result = league_baselines._build_outcomes_with_archetypes(2024, force_rebuild=True)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["pitch_archetype"])


class TestBaselinesByPitchTypeStandShape:
    def test_correct_row_count_and_columns(self, monkeypatch, tmp_path):
        _patch_sources(monkeypatch, tmp_path, {2024: OUTCOMES_2024}, {2024: ARCHETYPES_2024})

        result = league_baselines.get_baselines_by_pitch_type_stand(2024, force_rebuild=True)

        # 2 pitch types (FF, SL) x 2 stands (R, L) = 4, but pitcher 2 only has FF vs R
        # FF+R, FF+L, SL+R, SL+L = 4 rows
        # Actually: pitcher 2 FF vs R exists, so we have FF+R (2 pitchers), FF+L (1), SL+R (1), SL+L (1)
        # After groupby that's still 4 unique (pitch_type, batter_stand) combos
        expected_combos = {("FF", "R"), ("FF", "L"), ("SL", "R"), ("SL", "L")}
        actual_combos = set(zip(result["pitch_type"], result["batter_stand"]))
        assert actual_combos == expected_combos

        assert "season" in result.columns
        for col in league_baselines.RATE_COLS:
            assert col in result.columns


class TestBaselinesByArchetypeStandShape:
    def test_correct_row_count_and_columns(self, monkeypatch, tmp_path):
        _patch_sources(monkeypatch, tmp_path, {2024: OUTCOMES_2024}, {2024: ARCHETYPES_2024})

        result = league_baselines.get_baselines_by_archetype_stand(2024, force_rebuild=True)

        # Archetypes: 1 (FF pitchers 1+2), 3 (SL pitcher 1)
        # Stands for arch 1: R (pitchers 1+2), L (pitcher 1) = 2 rows
        # Stands for arch 3: R (pitcher 1), L (pitcher 1) = 2 rows
        # Total: 4 rows
        assert len(result) == 4
        assert "pitch_archetype" in result.columns
        assert "batter_stand" in result.columns
        assert "season" in result.columns
        for col in league_baselines.RATE_COLS:
            assert col in result.columns


class TestPooledBaselinesWeightsByVolume:
    def test_short_season_doesnt_get_equal_weight(self, monkeypatch, tmp_path):
        """2020 (low volume) shouldn't get equal weight to full seasons."""
        # Full season: 1000 pitches, 500 swings, 150 whiffs -> whiff_rate = 0.30
        full_season = _make_outcomes([
            {"pitches": 1000, "swings": 500, "whiffs": 150,
             "out_of_zone_pitches": 400, "chase_swings": 120,
             "called_strikes": 200, "csw": 350, "bip": 300,
             "hard_hits": 100, "barrels_proxy": 30,
             "xwoba_contact_sum": 90.0, "xwoba_contact_n": 250},
        ])
        # Short season: 100 pitches, 50 swings, 5 whiffs -> whiff_rate = 0.10
        short_season = _make_outcomes([
            {"pitches": 100, "swings": 50, "whiffs": 5,
             "out_of_zone_pitches": 40, "chase_swings": 4,
             "called_strikes": 20, "csw": 25, "bip": 30,
             "hard_hits": 10, "barrels_proxy": 3,
             "xwoba_contact_sum": 9.0, "xwoba_contact_n": 25},
        ])

        archetypes = _make_archetypes([
            {"pitcher_id": 1, "pitch_type": "FF", "pitch_archetype": 1},
        ])

        _patch_sources(
            monkeypatch, tmp_path,
            {2020: short_season, 2021: full_season},
            {2020: archetypes, 2021: archetypes},
        )
        monkeypatch.setattr(league_baselines, "_get_train_seasons", lambda: [2020, 2021])

        result = league_baselines.get_pooled_baselines(
            seasons=[2020, 2021], grouping="pitch_type", force_rebuild=True,
        )
        row = result.iloc[0]

        # Marcel-weighted (default): 2021 (most recent) gets weight 5, 2020 gets weight 4
        # whiffs: 150*5 + 5*4 = 770, swings: 500*5 + 50*4 = 2700
        # NOT simple average: (0.30 + 0.10) / 2 = 0.20
        expected_whiff = (150 * 5 + 5 * 4) / (500 * 5 + 50 * 4)
        assert row["whiff_rate"] == pytest.approx(expected_whiff)
        # Verify it's NOT the simple average
        assert row["whiff_rate"] != pytest.approx(0.20, abs=0.01)


class TestPriorBaselinesFallbackChain:
    def test_falls_back_from_archetype_stand_to_pitch_type_to_overall(
        self, monkeypatch, tmp_path
    ):
        """Falls back from archetype+stand -> pitch_type -> overall."""
        outcomes = _make_outcomes([
            {"pitcher_id": 1, "pitch_type": "FF", "batter_stand": "R",
             "pitches": 200, "swings": 100, "whiffs": 25,
             "out_of_zone_pitches": 80, "chase_swings": 24,
             "called_strikes": 30, "csw": 55, "bip": 60,
             "hard_hits": 20, "barrels_proxy": 6,
             "xwoba_contact_sum": 18.0, "xwoba_contact_n": 50},
        ])
        archetypes = _make_archetypes([
            {"pitcher_id": 1, "pitch_type": "FF", "pitch_archetype": 1},
        ])

        _patch_sources(monkeypatch, tmp_path, {2024: outcomes}, {2024: archetypes})
        monkeypatch.setattr(league_baselines, "_get_train_seasons", lambda: [2024])

        # 1. Archetype 1 + R should find a match
        result_arch = league_baselines.get_prior_baselines(
            pitch_type="FF", pitch_archetype=1, batter_stand="R", seasons=[2024],
        )
        assert result_arch["whiff_rate"] == pytest.approx(25 / 100)

        # 2. Archetype 99 + R (doesn't exist) should fall back to pitch_type+stand
        result_pt = league_baselines.get_prior_baselines(
            pitch_type="FF", pitch_archetype=99, batter_stand="R", seasons=[2024],
        )
        assert result_pt["whiff_rate"] == pytest.approx(25 / 100)

        # 3. pitch_type "CU" + stand "L" (doesn't exist) -> falls back to overall
        result_overall = league_baselines.get_prior_baselines(
            pitch_type="CU", batter_stand="L", seasons=[2024],
        )
        # Overall is the same data (only one row of data), so same rate
        assert result_overall["whiff_rate"] == pytest.approx(25 / 100)
