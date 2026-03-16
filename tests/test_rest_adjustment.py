"""Tests for the days-rest adjustment module."""
import numpy as np
import pandas as pd
import pytest

from src.models.rest_adjustment import (
    apply_rest_to_bf,
    classify_rest_bucket,
    compute_rest_for_game,
    get_rest_adjustment,
    get_rest_bf_modifier,
)


class TestClassifyRestBucket:
    def test_short_rest(self) -> None:
        assert classify_rest_bucket(3) == "short"
        assert classify_rest_bucket(4) == "short"

    def test_normal_rest(self) -> None:
        assert classify_rest_bucket(5) == "normal"

    def test_extended_rest(self) -> None:
        assert classify_rest_bucket(6) == "extended"
        assert classify_rest_bucket(10) == "extended"
        assert classify_rest_bucket(14) == "extended"

    def test_none_returns_normal(self) -> None:
        assert classify_rest_bucket(None) == "normal"

    def test_negative_returns_normal(self) -> None:
        assert classify_rest_bucket(-1) == "normal"


class TestGetRestAdjustment:
    def test_short_rest_negative_k_lift(self) -> None:
        adj = get_rest_adjustment(4)
        assert adj["rest_bucket"] == "short"
        assert adj["k_lift"] < 0
        assert adj["bb_lift"] > 0

    def test_normal_rest_zero_lifts(self) -> None:
        adj = get_rest_adjustment(5)
        assert adj["k_lift"] == 0.0
        assert adj["bb_lift"] == 0.0

    def test_extended_rest(self) -> None:
        adj = get_rest_adjustment(7)
        assert adj["rest_bucket"] == "extended"
        assert adj["k_lift"] == 0.0
        assert adj["bb_lift"] >= 0.0

    def test_none_returns_neutral(self) -> None:
        adj = get_rest_adjustment(None)
        assert adj["k_lift"] == 0.0
        assert adj["bb_lift"] == 0.0
        assert adj["rest_bucket"] == "normal"


class TestGetRestBfModifier:
    def test_short_rest_lower_bf(self) -> None:
        mod = get_rest_bf_modifier(4)
        assert mod["bf_mean_multiplier"] < 1.0
        assert mod["bf_var_multiplier"] > 1.0

    def test_normal_rest_no_change(self) -> None:
        mod = get_rest_bf_modifier(5)
        assert mod["bf_mean_multiplier"] == 1.0
        assert mod["bf_var_multiplier"] == 1.0

    def test_extended_rest(self) -> None:
        mod = get_rest_bf_modifier(8)
        assert mod["bf_mean_multiplier"] <= 1.0


class TestApplyRestToBf:
    def test_short_rest_reduces_mu(self) -> None:
        mu, sigma = apply_rest_to_bf(22.0, 3.5, 4)
        assert mu < 22.0
        assert sigma > 3.5

    def test_normal_rest_unchanged(self) -> None:
        mu, sigma = apply_rest_to_bf(22.0, 3.5, 5)
        assert mu == 22.0
        assert sigma == 3.5

    def test_none_rest_unchanged(self) -> None:
        mu, sigma = apply_rest_to_bf(22.0, 3.5, None)
        assert mu == 22.0
        assert sigma == 3.5


class TestComputeRestForGame:
    def test_found(self) -> None:
        df = pd.DataFrame({
            "pitcher_id": [12345],
            "game_pk": [99999],
            "days_rest": [5],
        })
        assert compute_rest_for_game(12345, 99999, df) == 5

    def test_not_found(self) -> None:
        df = pd.DataFrame({
            "pitcher_id": [12345],
            "game_pk": [99999],
            "days_rest": [5],
        })
        assert compute_rest_for_game(12345, 11111, df) is None

    def test_none_df(self) -> None:
        assert compute_rest_for_game(12345, 99999, None) is None

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["pitcher_id", "game_pk", "days_rest"])
        assert compute_rest_for_game(12345, 99999, df) is None
