import numpy as np
import pandas as pd

from src.evaluation import backtesting


def test_compute_metrics_uses_posterior_probability_columns() -> None:
    comp = pd.DataFrame(
        {
            "actual_k_rate": [0.18, 0.27],
            "k_rate_mean": [0.20, 0.25],
            "k_rate_sd": [0.03, 0.02],
            "k_rate_2_5": [0.14, 0.21],
            "k_rate_97_5": [0.26, 0.29],
            "marcel_k_rate": [0.19, 0.24],
            "weighted_pa": [600.0, 700.0],
            "post_p_above_league": [0.20, 0.85],
            "post_ci_50_lo": [0.17, 0.24],
            "post_ci_50_hi": [0.22, 0.27],
            "post_ci_80_lo": [0.15, 0.22],
            "post_ci_80_hi": [0.24, 0.28],
            "post_ci_90_lo": [0.14, 0.21],
            "post_ci_90_hi": [0.25, 0.29],
            "post_ci_95_lo": [0.13, 0.20],
            "post_ci_95_hi": [0.26, 0.30],
        }
    )

    metrics = backtesting.compute_metrics(comp, league_avg_k=0.22)

    assert 0.0 <= metrics["bayes_brier"] <= 1.0
    assert 0.0 <= metrics["marcel_brier"] <= 1.0
    assert np.isclose(metrics["coverage_95"], 1.0)
    assert list(metrics["calibration"]["nominal_level"]) == [0.5, 0.8, 0.9, 0.95]


def test_walk_forward_backtest_mocked_pipeline(monkeypatch) -> None:
    df_train = pd.DataFrame(
        [
            {"batter_id": 1, "batter_name": "A", "season": 2023, "pa": 120, "k": 28, "k_rate": 28 / 120},
            {"batter_id": 2, "batter_name": "B", "season": 2023, "pa": 140, "k": 30, "k_rate": 30 / 140},
        ]
    )
    df_test = pd.DataFrame(
        [
            {"batter_id": 1, "batter_name": "A", "season": 2024, "pa": 130, "k": 29, "k_rate": 29 / 130},
            {"batter_id": 2, "batter_name": "B", "season": 2024, "pa": 150, "k": 34, "k_rate": 34 / 150},
        ]
    )

    def fake_build_multi_season_k_data(seasons: list[int], min_pa: int = 1) -> pd.DataFrame:
        if seasons == [2024]:
            return df_test.copy()
        return df_train[df_train["season"].isin(seasons)].copy()

    monkeypatch.setattr(backtesting, "build_multi_season_k_data", fake_build_multi_season_k_data)
    monkeypatch.setattr(
        backtesting,
        "prepare_model_data",
        lambda df: {"df": df.reset_index(drop=True)},
    )
    monkeypatch.setattr(backtesting, "fit_k_rate_model", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(backtesting, "check_convergence", lambda trace: {"converged": True})
    monkeypatch.setattr(
        backtesting,
        "extract_player_posteriors",
        lambda trace, data: pd.DataFrame(
            [
                {
                    "batter_id": 1,
                    "season": 2023,
                    "k_rate_mean": 0.22,
                    "k_rate_sd": 0.02,
                    "k_rate_2_5": 0.18,
                    "k_rate_97_5": 0.26,
                },
                {
                    "batter_id": 2,
                    "season": 2023,
                    "k_rate_mean": 0.23,
                    "k_rate_sd": 0.02,
                    "k_rate_2_5": 0.19,
                    "k_rate_97_5": 0.27,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        backtesting,
        "_summarize_posterior_for_season",
        lambda trace, data, season, league_avg_k, **kwargs: pd.DataFrame(
            [
                {
                    "batter_id": 1,
                    "season": season,
                    "post_p_above_league": 0.55,
                    "post_ci_50_lo": 0.20,
                    "post_ci_50_hi": 0.24,
                    "post_ci_80_lo": 0.19,
                    "post_ci_80_hi": 0.25,
                    "post_ci_90_lo": 0.18,
                    "post_ci_90_hi": 0.26,
                    "post_ci_95_lo": 0.17,
                    "post_ci_95_hi": 0.27,
                },
                {
                    "batter_id": 2,
                    "season": season,
                    "post_p_above_league": 0.60,
                    "post_ci_50_lo": 0.21,
                    "post_ci_50_hi": 0.25,
                    "post_ci_80_lo": 0.20,
                    "post_ci_80_hi": 0.26,
                    "post_ci_90_lo": 0.19,
                    "post_ci_90_hi": 0.27,
                    "post_ci_95_lo": 0.18,
                    "post_ci_95_hi": 0.28,
                },
            ]
        ),
    )

    metrics = backtesting.walk_forward_backtest(
        train_seasons=[2023],
        test_season=2024,
        min_pa_train=1,
        min_pa_test=1,
        draws=20,
        tune=20,
        chains=1,
    )

    assert metrics["n_players"] == 2
    assert metrics["convergence"]["converged"] is True
    assert "bayes_brier" in metrics
    assert "calibration" in metrics
