from __future__ import annotations

import pandas as pd

from src.models import daily_standouts


def _make_hitter_row(**overrides) -> dict:
    base = {
        "batter_id": 1, "batter_name": "A", "game_pk": 100,
        "game_date": "2026-04-07", "away_team": "SEA", "home_team": "LAD",
        "pa": 5, "ab": 4, "hits": 3, "doubles": 1, "triples": 0, "hr": 1,
        "total_bases": 7, "runs": 2, "rbi": 3, "bb": 1, "ibb": 0,
        "hbp": 0, "k": 0, "sb": 0, "cs": 0, "sf": 0,
        "bip": 3, "hard_hit_pct": 0.67, "barrel_pct": 0.33, "xwoba": 0.600,
        "avg": 0.750, "slg": 1.750, "obp": 0.800, "ops": 2.550,
        "iso": 1.000, "woba": 0.700, "k_rate": 0.0, "bb_rate": 0.20,
    }
    base.update(overrides)
    return base


def _make_pitcher_row(**overrides) -> dict:
    base = {
        "pitcher_id": 10, "pitcher_name": "P", "game_pk": 100,
        "game_date": "2026-04-07", "away_team": "SEA", "home_team": "LAD",
        "is_starter": True, "bf": 27, "ip": 7.0, "k": 10, "bb": 1,
        "hr": 0, "hits": 4, "er": 1, "runs": 1, "w": 1, "l": 0,
        "sv": 0, "hld": 0, "pitches": 98, "qs": 1,
        "k_rate": 0.370, "bb_rate": 0.037, "k_minus_bb": 0.333,
        "era": 1.29, "whip": 0.71, "hr_per_9": 0.0, "role": "SP",
    }
    base.update(overrides)
    return base


def test_score_hitter_daily_filters_low_pa():
    df = pd.DataFrame([
        _make_hitter_row(batter_id=1, pa=2, batter_name="TooFew"),
        _make_hitter_row(batter_id=2, pa=4, batter_name="Enough"),
    ])
    out = daily_standouts.score_hitter_daily(df, min_pa=3)
    assert len(out) == 1
    assert out.iloc[0]["batter_name"] == "Enough"


def test_score_hitter_daily_ranks_big_game_higher():
    df = pd.DataFrame([
        _make_hitter_row(batter_id=1, hits=4, hr=2, rbi=5, runs=3,
                         woba=0.800, ops=2.0, iso=0.800, batter_name="Star"),
        _make_hitter_row(batter_id=2, hits=1, hr=0, rbi=0, runs=0,
                         woba=0.250, ops=0.500, iso=0.050, batter_name="Quiet"),
    ])
    out = daily_standouts.score_hitter_daily(df)
    star = out.loc[out["batter_id"] == 1].iloc[0]
    quiet = out.loc[out["batter_id"] == 2].iloc[0]
    assert star["daily_standout_rank"] < quiet["daily_standout_rank"]


def test_score_pitcher_daily_separates_sp_rp():
    df = pd.DataFrame([
        _make_pitcher_row(pitcher_id=10, role="SP", is_starter=True, ip=7.0, k=10),
        _make_pitcher_row(pitcher_id=20, role="RP", is_starter=False, ip=1.0, k=2,
                          bf=6, pitcher_name="Reliever", qs=0, w=0),
    ])
    out = daily_standouts.score_pitcher_daily(df)
    sp = out.loc[out["role"] == "SP"]
    rp = out.loc[out["role"] == "RP"]
    assert len(sp) == 1
    assert len(rp) == 1
    # Both should be rank 1 within their role
    assert sp.iloc[0]["daily_standout_rank"] == 1
    assert rp.iloc[0]["daily_standout_rank"] == 1


def test_empty_input_returns_empty_df():
    empty = pd.DataFrame()
    h = daily_standouts.score_hitter_daily(empty)
    p = daily_standouts.score_pitcher_daily(empty)
    assert len(h) == 0
    assert len(p) == 0
    assert "daily_standout_score" in h.columns
    assert "daily_standout_score" in p.columns


def test_empty_hitter_daily_has_dashboard_required_columns():
    """Empty hitter daily output must carry the columns the dashboard reads."""
    out = daily_standouts.score_hitter_daily(pd.DataFrame())
    required = {
        "batter_id", "batter_name", "daily_standout_score",
        "hits", "hr", "rbi", "ops", "woba",
    }
    assert required.issubset(set(out.columns))


def test_empty_pitcher_daily_has_dashboard_required_columns():
    """Empty pitcher daily output must carry the columns the dashboard reads."""
    out = daily_standouts.score_pitcher_daily(pd.DataFrame())
    required = {
        "pitcher_id", "pitcher_name", "role", "daily_standout_score",
        "k", "era", "whip",
    }
    assert required.issubset(set(out.columns))


def test_empty_hitter_daily_exposes_farthest_hr_column():
    """Nice-to-have: farthest HR candidate column is advertised in empty schema."""
    out = daily_standouts.score_hitter_daily(pd.DataFrame())
    assert "max_hr_distance" in out.columns


def test_hitter_daily_passes_through_max_hr_distance():
    df = pd.DataFrame([
        _make_hitter_row(batter_id=1, batter_name="Bomber", hr=2,
                         **{"max_hr_distance": 445.0}),
        _make_hitter_row(batter_id=2, batter_name="NoHR", hr=0,
                         **{"max_hr_distance": None}),
    ])
    out = daily_standouts.score_hitter_daily(df)
    bomber = out.loc[out["batter_id"] == 1].iloc[0]
    no_hr = out.loc[out["batter_id"] == 2].iloc[0]
    assert bomber["max_hr_distance"] == 445.0
    assert pd.isna(no_hr["max_hr_distance"])


def test_build_daily_standout_boards_adds_core_delta(monkeypatch):
    monkeypatch.setattr(
        daily_standouts,
        "get_hitter_daily_standouts",
        lambda game_date=None: pd.DataFrame([_make_hitter_row()]),
    )
    monkeypatch.setattr(
        daily_standouts,
        "get_pitcher_daily_standouts",
        lambda game_date=None: pd.DataFrame([_make_pitcher_row()]),
    )

    boards = daily_standouts.build_daily_standout_boards(
        core_hitters=pd.DataFrame({"batter_id": [1], "overall_rank": [15]}),
        core_pitchers=pd.DataFrame({"pitcher_id": [10], "overall_rank": [5]}),
    )
    assert "delta_vs_core" in boards["hitters"].columns
    assert "delta_vs_core" in boards["pitchers"].columns
