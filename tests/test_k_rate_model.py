import numpy as np
import pandas as pd
import pytest

from src.models.k_rate_model import build_k_rate_model, prepare_model_data


def test_prepare_model_data_handles_constant_covariates() -> None:
    df = pd.DataFrame(
        [
            {
                "batter_id": 10,
                "batter_name": "A",
                "season": 2023,
                "pa": 100,
                "k": 20,
                "k_rate": 0.20,
                "barrel_pct": 0.08,
                "hard_hit_pct": 0.35,
            },
            {
                "batter_id": 10,
                "batter_name": "A",
                "season": 2024,
                "pa": 120,
                "k": 26,
                "k_rate": 0.2167,
                "barrel_pct": 0.08,
                "hard_hit_pct": 0.35,
            },
            {
                "batter_id": 11,
                "batter_name": "B",
                "season": 2024,
                "pa": 90,
                "k": 18,
                "k_rate": 0.20,
                "barrel_pct": 0.08,
                "hard_hit_pct": 0.35,
            },
        ]
    )

    data = prepare_model_data(df)

    assert data["n_players"] == 2
    assert data["n_seasons"] == 2
    assert data["pa"].shape[0] == 3
    assert np.all(np.isfinite(data["barrel_z"]))
    assert np.all(np.isfinite(data["hard_hit_z"]))
    assert np.allclose(data["barrel_z"], 0.0)
    assert np.allclose(data["hard_hit_z"], 0.0)


# ---------------------------------------------------------------------------
# Synthetic platoon DataFrame helper
# ---------------------------------------------------------------------------
def _make_platoon_df() -> pd.DataFrame:
    """4 rows: 2 players × 2 pitch hands, with same_side flag."""
    return pd.DataFrame([
        {"batter_id": 10, "batter_name": "A", "season": 2024,
         "pa": 80, "k": 18, "k_rate": 0.225, "barrel_pct": 0.07,
         "hard_hit_pct": 0.33, "pitch_hand": "R", "same_side": 0},
        {"batter_id": 10, "batter_name": "A", "season": 2024,
         "pa": 120, "k": 30, "k_rate": 0.250, "barrel_pct": 0.07,
         "hard_hit_pct": 0.33, "pitch_hand": "L", "same_side": 0},
        {"batter_id": 11, "batter_name": "B", "season": 2024,
         "pa": 150, "k": 28, "k_rate": 0.187, "barrel_pct": 0.10,
         "hard_hit_pct": 0.40, "pitch_hand": "R", "same_side": 1},
        {"batter_id": 11, "batter_name": "B", "season": 2024,
         "pa": 90, "k": 22, "k_rate": 0.244, "barrel_pct": 0.10,
         "hard_hit_pct": 0.40, "pitch_hand": "L", "same_side": 0},
    ])


# ---------------------------------------------------------------------------
# Platoon tests
# ---------------------------------------------------------------------------
def test_prepare_model_data_platoon_basic() -> None:
    """same_side array has correct shape, dtype, and platoon=True in dict."""
    df = _make_platoon_df()
    data = prepare_model_data(df, platoon=True)

    assert data["platoon"] is True
    assert "same_side" in data
    assert data["same_side"].shape == (4,)
    assert data["same_side"].dtype == int
    assert list(data["same_side"]) == [0, 0, 1, 0]


def test_prepare_model_data_platoon_false_is_default() -> None:
    """Backward compat: no same_side key when platoon=False (default)."""
    df = _make_platoon_df()
    data = prepare_model_data(df)  # default platoon=False

    assert data.get("platoon", False) is False
    assert "same_side" not in data


def test_prepare_model_data_platoon_missing_column_raises() -> None:
    """platoon=True without same_side column → clear ValueError."""
    df = _make_platoon_df().drop(columns=["same_side"])
    with pytest.raises(ValueError, match="same_side"):
        prepare_model_data(df, platoon=True)


def test_build_k_rate_model_platoon_has_gamma_variables() -> None:
    """Model has gamma_pop, sigma_gamma, gamma_raw RVs and gamma deterministic."""
    df = _make_platoon_df()
    data = prepare_model_data(df, platoon=True)
    model = build_k_rate_model(data)

    rv_names = {rv.name for rv in model.free_RVs}
    det_names = {d.name for d in model.deterministics}

    assert "gamma_pop" in rv_names
    assert "sigma_gamma" in rv_names
    assert "gamma_raw" in rv_names
    assert "gamma" in det_names


def test_build_k_rate_model_no_platoon_no_gamma() -> None:
    """Model does NOT have gamma variables when platoon=False."""
    df = _make_platoon_df()
    data = prepare_model_data(df, platoon=False)
    model = build_k_rate_model(data)

    all_names = {rv.name for rv in model.free_RVs} | {d.name for d in model.deterministics}

    assert "gamma_pop" not in all_names
    assert "sigma_gamma" not in all_names
    assert "gamma_raw" not in all_names
    assert "gamma" not in all_names
