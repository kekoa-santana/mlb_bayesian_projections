"""Byte-identity check for the merged conjugate-update dispatcher.

Confirms update_player_rate_samples_step dispatches to the correct
preseason-loader + observed-totals + id-column combination for each
player_type, calling update_rate_samples with exactly the arguments the
previous pair of functions passed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from update_in_season import update_player_rate_samples_step  # noqa: E402


def _fake_preseason() -> dict[str, np.ndarray]:
    return {"111": np.array([0.20, 0.22, 0.24]), "222": np.array([0.18, 0.20, 0.22])}


def _fake_obs(id_col: str) -> pd.DataFrame:
    return pd.DataFrame({id_col: [111, 222], "pa": [50, 60], "strikeouts": [12, 13],
                         "batters_faced": [200, 220], "walks": [15, 18], "hr": [3, 4]})


def test_pitcher_dispatch_calls_pitcher_loaders_and_id_col():
    recorded: dict = {}

    def fake_update(preseason, obs, *, player_id_col, trials_col, successes_col,
                    league_avg, min_trials, n_samples):
        recorded.update(dict(
            player_id_col=player_id_col, trials_col=trials_col,
            successes_col=successes_col, league_avg=league_avg,
            min_trials=min_trials, n_samples=n_samples,
        ))
        return {"111": preseason["111"]}

    with patch("update_in_season.load_preseason_rate_samples",
               return_value=_fake_preseason()) as m_pre_p, \
         patch("update_in_season.load_hitter_preseason_samples",
               return_value=_fake_preseason()) as m_pre_h, \
         patch("update_in_season.get_observed_pitcher_totals",
               return_value=_fake_obs("pitcher_id")) as m_obs_p, \
         patch("update_in_season.get_observed_hitter_totals",
               return_value=_fake_obs("batter_id")) as m_obs_h, \
         patch("src.models.in_season_updater.update_rate_samples",
               side_effect=fake_update):

        update_player_rate_samples_step(
            "pitcher", "bb", trials_col="batters_faced",
            successes_col="walks", league_avg=0.08,
        )

    m_pre_p.assert_called_once_with("bb")
    m_pre_h.assert_not_called()
    m_obs_p.assert_called_once()
    m_obs_h.assert_not_called()
    assert recorded["player_id_col"] == "pitcher_id"
    assert recorded["trials_col"] == "batters_faced"
    assert recorded["successes_col"] == "walks"
    assert recorded["league_avg"] == 0.08
    assert recorded["min_trials"] == 10
    assert recorded["n_samples"] == 1000


def test_hitter_dispatch_calls_hitter_loaders_and_id_col():
    recorded: dict = {}

    def fake_update(preseason, obs, *, player_id_col, trials_col, successes_col,
                    league_avg, min_trials, n_samples):
        recorded.update(dict(
            player_id_col=player_id_col, trials_col=trials_col,
            successes_col=successes_col, league_avg=league_avg,
        ))
        return {"111": preseason["111"]}

    with patch("update_in_season.load_preseason_rate_samples",
               return_value=_fake_preseason()) as m_pre_p, \
         patch("update_in_season.load_hitter_preseason_samples",
               return_value=_fake_preseason()) as m_pre_h, \
         patch("update_in_season.get_observed_pitcher_totals",
               return_value=_fake_obs("pitcher_id")) as m_obs_p, \
         patch("update_in_season.get_observed_hitter_totals",
               return_value=_fake_obs("batter_id")) as m_obs_h, \
         patch("src.models.in_season_updater.update_rate_samples",
               side_effect=fake_update):

        update_player_rate_samples_step(
            "hitter", "k", trials_col="pa",
            successes_col="strikeouts", league_avg=0.226,
        )

    m_pre_h.assert_called_once_with("k")
    m_pre_p.assert_not_called()
    m_obs_h.assert_called_once()
    m_obs_p.assert_not_called()
    assert recorded["player_id_col"] == "batter_id"
    assert recorded["trials_col"] == "pa"
    assert recorded["successes_col"] == "strikeouts"
    assert recorded["league_avg"] == 0.226


def test_missing_preseason_returns_empty_dict():
    with patch("update_in_season.load_preseason_rate_samples", return_value={}):
        result = update_player_rate_samples_step(
            "pitcher", "bb", trials_col="batters_faced",
            successes_col="walks", league_avg=0.08,
        )
    assert result == {}


def test_empty_observed_returns_preseason_unchanged():
    preseason = _fake_preseason()
    with patch("update_in_season.load_hitter_preseason_samples", return_value=preseason), \
         patch("update_in_season.get_observed_hitter_totals", return_value=pd.DataFrame()):
        result = update_player_rate_samples_step(
            "hitter", "bb", trials_col="pa",
            successes_col="walks", league_avg=0.082,
        )
    assert result is preseason


def test_unknown_player_type_raises():
    import pytest
    with pytest.raises(ValueError):
        update_player_rate_samples_step(
            "catcher", "k", trials_col="pa",
            successes_col="strikeouts", league_avg=0.226,
        )
