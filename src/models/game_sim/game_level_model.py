"""
Paired Negative Binomial game-level run model.

Predicts (home_runs, away_runs) per game using two independent NegBin
regressions (shared coefficients, ``is_home`` indicator for HFA) fitted
via statsmodels NegativeBinomialP.  This is NOT a true bivariate model
with an explicit correlation parameter -- home and away scores share
context features (park, weather) which induces implicit correlation,
but the residuals are independent.

Features: pitcher quality (K%/BB%/HR%), team offense (rolling R/G),
park run factor, temperature, wind, dome status, home-field advantage.

The model is trained in "stacked" format: each game contributes 2 rows
(one home perspective, one away perspective).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# ---- Feature constants -------------------------------------------------- #

PITCHER_FEATURES = [
    "opp_starter_k_pct",    # opposing starter's season-to-date K%
    "opp_starter_bb_pct",   # opposing starter's season-to-date BB%
    "opp_starter_hr_pct",   # opposing starter's season-to-date HR/BF%
]

TEAM_FEATURES = [
    "team_rpg",             # batting team's rolling 14-game R/G
]

PARK_WEATHER_FEATURES = [
    "park_run_factor",      # venue run park factor (indexed to 1.0)
    "temperature",          # game temperature (F)
    "wind_out",             # 1 if wind blowing out
    "wind_in",              # 1 if wind blowing in
    "is_dome",              # 1 if dome/retractable roof closed
]

HFA_FEATURES = [
    "is_home",              # 1 for home batting team, 0 for away
]

ALL_FEATURES = PITCHER_FEATURES + TEAM_FEATURES + PARK_WEATHER_FEATURES + HFA_FEATURES

# League-average fallbacks (approx 2022-2024)
LEAGUE_AVG_K_PCT = 0.224
LEAGUE_AVG_BB_PCT = 0.083
LEAGUE_AVG_HR_PCT = 0.034
LEAGUE_AVG_RPG = 4.5


class GameLevelModel:
    """Wraps a fitted NegBin regression for runs scored.

    The model is fit on stacked data (home + away rows per game) with an
    `is_home` indicator. At prediction time we produce two rows per game
    (home batting, away batting) and unstack to get (mu_home, mu_away).
    """

    def __init__(self) -> None:
        self.result_: Any | None = None  # fitted statsmodels result
        self.alpha_: float | None = None  # NegBin dispersion parameter

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "GameLevelModel":
        """Fit the NegBin model on stacked training data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns in ALL_FEATURES plus 'runs' (target).

        Returns
        -------
        self
        """
        X = sm.add_constant(df[ALL_FEATURES].astype(float), has_constant="add")
        y = df["runs"].astype(float)

        model = sm.NegativeBinomialP(y, X, p=2)
        self.result_ = model.fit(disp=False, maxiter=200)
        self.alpha_ = float(self.result_.params.get("alpha", self.result_.params.iloc[-1]))

        logger.info(
            "NegBin fit: %d obs, alpha=%.4f, AIC=%.1f",
            len(df), self.alpha_, self.result_.aic,
        )
        return self

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted mean runs (mu) for each row.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ALL_FEATURES columns.

        Returns
        -------
        np.ndarray of shape (n,)
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted yet.")
        X = sm.add_constant(df[ALL_FEATURES].astype(float), has_constant="add")
        return self.result_.predict(X).values

    def predict_game(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict per-game (mu_home, mu_away, alpha).

        Parameters
        ----------
        features_df : pd.DataFrame
            One row per game with columns prefixed home_/away_ for the
            stacked features, plus park/weather columns. Must have the
            columns produced by `build_game_features()`.

        Returns
        -------
        pd.DataFrame with columns: mu_home, mu_away, alpha
        """
        # Build home-batting row
        home_row = self._orient_batting(features_df, side="home")
        away_row = self._orient_batting(features_df, side="away")

        mu_home = self.predict_mu(home_row)
        mu_away = self.predict_mu(away_row)

        return pd.DataFrame({
            "mu_home": mu_home,
            "mu_away": mu_away,
            "alpha": self.alpha_,
        }, index=features_df.index)

    def sample_game(
        self,
        mu_home: float | np.ndarray,
        mu_away: float | np.ndarray,
        n_samples: int = 10_000,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw NegBin samples for a game.

        Parameters
        ----------
        mu_home, mu_away : float or array
            Predicted means.
        n_samples : int
            Number of Monte Carlo draws.
        rng : numpy Generator, optional

        Returns
        -------
        (home_runs_samples, away_runs_samples) each shape (n_samples,)
        """
        if rng is None:
            rng = np.random.default_rng()

        alpha = self.alpha_

        def _draw(mu: float | np.ndarray) -> np.ndarray:
            mu = np.atleast_1d(np.asarray(mu, dtype=float))
            # NegBin parameterization: n = mu/alpha, p = 1/(1+alpha)
            # statsmodels NegBinP(p=2): Var = mu + alpha*mu^2
            # scipy: n = 1/alpha, p = 1/(1 + alpha*mu)
            n_param = 1.0 / alpha
            p_param = n_param / (n_param + mu)
            return rng.negative_binomial(
                np.maximum(n_param, 0.01),
                np.clip(p_param, 1e-8, 1.0 - 1e-8),
                size=(n_samples,) + mu.shape,
            )

        return _draw(mu_home), _draw(mu_away)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _orient_batting(df: pd.DataFrame, side: str) -> pd.DataFrame:
        """Orient a game-level DataFrame to batting-team perspective.

        side='home' means the home team is batting (opposing pitcher = away starter).
        side='away' means the away team is batting (opposing pitcher = home starter).
        """
        opp = "away" if side == "home" else "home"
        return pd.DataFrame({
            "opp_starter_k_pct": df[f"starter_k_pct_{opp}"],
            "opp_starter_bb_pct": df[f"starter_bb_pct_{opp}"],
            "opp_starter_hr_pct": df[f"starter_hr_pct_{opp}"],
            "team_rpg": df[f"team_rpg_{side}"],
            "park_run_factor": df["park_run_factor"],
            "temperature": df["temperature"],
            "wind_out": df["wind_out"],
            "wind_in": df["wind_in"],
            "is_dome": df["is_dome"],
            "is_home": 1.0 if side == "home" else 0.0,
        }, index=df.index)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Pickle the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"result": self.result_, "alpha": self.alpha_},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GameLevelModel":
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls()
        model.result_ = data["result"]
        model.alpha_ = data["alpha"]
        logger.info("Model loaded from %s", path)
        return model
