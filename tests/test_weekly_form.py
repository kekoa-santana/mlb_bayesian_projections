from __future__ import annotations

import pandas as pd

from src.models import player_rankings
from src.models import weekly_form


def test_rank_core_preseason_uses_preseason_contract(monkeypatch):
    calls: dict[str, int] = {}

    def _fake_rank_all(season, projection_season, health_df=None, pitcher_roles_df=None, min_pa=None, min_bf=None):
        calls["season"] = season
        calls["projection_season"] = projection_season
        return {"hitters": pd.DataFrame(), "pitchers": pd.DataFrame()}

    monkeypatch.setattr(player_rankings, "rank_all", _fake_rank_all)
    player_rankings.rank_core_preseason(anchor_season=2025)
    assert calls["season"] == 2025
    assert calls["projection_season"] == 2026


def test_compute_hitter_weekly_form_shrinks_low_pa_to_neutral():
    df = pd.DataFrame(
        {
            "batter_id": [1, 2],
            "batter_name": ["LowPA", "HighPA"],
            "games_14d": [2, 10],
            "pa_14d": [5, 80],
            "k_rate_14d": [0.10, 0.10],
            "bb_rate_14d": [0.20, 0.20],
            "woba_14d": [0.600, 0.600],
            "xwoba_14d": [0.600, 0.600],
            "ops_14d": [1.200, 1.200],
            "iso_14d": [0.400, 0.400],
            "hard_hit_pct_14d": [0.70, 0.70],
            "barrel_pct_14d": [0.20, 0.20],
            "bip_14d": [3, 40],
        },
    )
    out = weekly_form.compute_hitter_weekly_form(df, min_pa=15, full_pa=80)
    low = out.loc[out["batter_id"] == 1].iloc[0]
    high = out.loc[out["batter_id"] == 2].iloc[0]
    assert abs(low["weekly_form_score"] - 0.50) < abs(high["weekly_form_score"] - 0.50)
    assert low["weekly_form_reliability"] < high["weekly_form_reliability"]


def test_compute_pitcher_weekly_form_shrinks_low_bf_to_neutral():
    df = pd.DataFrame(
        {
            "pitcher_id": [1, 2],
            "pitcher_name": ["LowBF", "HighBF"],
            "games_14d": [1, 4],
            "starts_14d": [0, 2],
            "bf_14d": [8, 100],
            "ip_14d": [2.0, 30.0],
            "k_14d": [5, 30],
            "bb_14d": [0, 5],
            "hr_14d": [0, 1],
            "hits_14d": [1, 20],
            "er_14d": [0, 5],
            "sv_14d": [1, 0],
            "hld_14d": [0, 0],
            "k_rate_14d": [0.625, 0.300],
            "bb_rate_14d": [0.000, 0.050],
            "k_minus_bb_14d": [0.625, 0.250],
            "era_14d": [0.0, 1.5],
            "whip_14d": [0.5, 0.83],
            "hr_per_9_14d": [0.0, 0.3],
            "role_14d": ["RP", "SP"],
        },
    )
    out = weekly_form.compute_pitcher_weekly_form(df, min_bf=20, full_bf=100)
    low = out.loc[out["pitcher_id"] == 1].iloc[0]
    high = out.loc[out["pitcher_id"] == 2].iloc[0]
    assert abs(low["weekly_form_score"] - 0.50) < abs(high["weekly_form_score"] - 0.50)
    assert low["weekly_form_reliability"] < high["weekly_form_reliability"]


def test_build_weekly_form_boards_adds_core_delta(monkeypatch):
    monkeypatch.setattr(
        weekly_form,
        "get_hitter_recent_form",
        lambda days=14, as_of_date=None: pd.DataFrame(
            {
                "batter_id": [1],
                "batter_name": ["A"],
                "games_14d": [5],
                "pa_14d": [40],
                "k_rate_14d": [0.2],
                "bb_rate_14d": [0.1],
                "woba_14d": [0.35],
                "xwoba_14d": [0.34],
                "ops_14d": [0.88],
                "iso_14d": [0.20],
                "hard_hit_pct_14d": [0.45],
                "barrel_pct_14d": [0.09],
                "bip_14d": [18],
            },
        ),
    )
    monkeypatch.setattr(
        weekly_form,
        "get_pitcher_recent_form",
        lambda days=14, as_of_date=None: pd.DataFrame(
            {
                "pitcher_id": [10],
                "pitcher_name": ["P"],
                "games_14d": [2],
                "starts_14d": [1],
                "bf_14d": [45],
                "ip_14d": [12.0],
                "k_14d": [14],
                "bb_14d": [2],
                "hr_14d": [1],
                "hits_14d": [9],
                "er_14d": [3],
                "sv_14d": [0],
                "hld_14d": [0],
                "k_rate_14d": [14 / 45],
                "bb_rate_14d": [2 / 45],
                "k_minus_bb_14d": [12 / 45],
                "era_14d": [2.25],
                "whip_14d": [0.92],
                "hr_per_9_14d": [0.75],
                "role_14d": ["SP"],
            },
        ),
    )

    boards = weekly_form.build_weekly_form_boards(
        core_hitters=pd.DataFrame({"batter_id": [1], "overall_rank": [12]}),
        core_pitchers=pd.DataFrame({"pitcher_id": [10], "overall_rank": [6]}),
    )
    assert "delta_vs_core" in boards["hitters"].columns
    assert "delta_vs_core" in boards["pitchers"].columns
