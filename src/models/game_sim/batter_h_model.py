"""
Negative Binomial regression model for per-batter hit counts.

Predicts H (hits) per game for individual batters using a NegBinP(p=2)
regression, following the same pattern as GameLevelModel for team runs.
The NegBin captures overdispersion (multi-hit games are more common than
a Poisson would predict) and allows meaningful differentiation between
batters based on skill, matchup, and context features.

Features: batter hit skill (H/PA, K%, BABIP), sprint speed, opposing
pitcher quality, batting order, platoon, park BABIP factor, home-field.
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

BATTER_FEATURES = [
    "batter_h_per_pa",      # entering-game season H/PA
    "batter_k_pct",         # entering-game K%
    "batter_babip",         # entering-game BABIP
    "sprint_speed",         # Statcast sprint speed (ft/s)
]

PITCHER_FEATURES = [
    "opp_starter_k_pct",    # opposing starter's entering-game K%
    "opp_starter_bb_pct",   # opposing starter's entering-game BB%
]

CONTEXT_FEATURES = [
    "batting_order",        # lineup slot (1-9)
    "platoon",              # 1 if same hand (pitcher/batter), 0 otherwise
    "park_h_babip",         # park BABIP factor (centered at 0)
    "is_home",              # 1 for home batter
]

ALL_FEATURES = BATTER_FEATURES + PITCHER_FEATURES + CONTEXT_FEATURES

# Fallback values for missing features
DEFAULTS = {
    "batter_h_per_pa": 0.230,
    "batter_k_pct": 0.224,
    "batter_babip": 0.295,
    "sprint_speed": 27.0,
    "opp_starter_k_pct": 0.224,
    "opp_starter_bb_pct": 0.083,
    "batting_order": 5,
    "platoon": 0,
    "park_h_babip": 0.0,
    "is_home": 0.5,
}


class BatterHModel:
    """NegBin regression for per-batter hit counts per game."""

    def __init__(self) -> None:
        self.result_: Any | None = None
        self.alpha_: float | None = None

    def fit(self, df: pd.DataFrame) -> "BatterHModel":
        """Fit Poisson GLM on batter-game training data.

        Uses Poisson rather than NegBin because per-batter H counts
        show no meaningful overdispersion (alpha ~ 0 in NegBin fits).
        Sampling uses a small fixed alpha to add slight overdispersion
        for realistic multi-hit game probabilities.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ALL_FEATURES columns plus 'h' (target: hit count).
        """
        X = sm.add_constant(df[ALL_FEATURES].astype(float), has_constant="add")
        y = df["h"].astype(float)

        model = sm.Poisson(y, X)
        self.result_ = model.fit(disp=False, maxiter=200)
        # Use a small fixed alpha for sampling overdispersion
        # Var = mu + alpha*mu^2; alpha=0.05 gives ~5% extra variance
        self.alpha_ = 0.05

        logger.info(
            "BatterHModel fit (Poisson): %d obs, AIC=%.1f, fixed alpha=%.3f",
            len(df), self.result_.aic, self.alpha_,
        )
        return self

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted mean H for each row."""
        if self.result_ is None:
            raise RuntimeError("Model not fitted.")
        feats = df[ALL_FEATURES].copy()
        for col, default in DEFAULTS.items():
            if col in feats.columns:
                feats[col] = feats[col].fillna(default)
        X = sm.add_constant(feats.astype(float), has_constant="add")
        return self.result_.predict(X).values

    def sample_h(
        self,
        mu: float | np.ndarray,
        n_samples: int = 10_000,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw NegBin hit count samples.

        Parameters
        ----------
        mu : float or array
            Predicted mean H.
        n_samples : int
            Number of Monte Carlo draws.
        rng : numpy Generator, optional

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if rng is None:
            rng = np.random.default_rng()
        alpha = self.alpha_
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        # NegBinP(p=2): Var = mu + alpha*mu^2
        # scipy parameterization: n = 1/alpha, p = n/(n + mu)
        n_param = 1.0 / alpha
        p_param = n_param / (n_param + mu)
        return rng.negative_binomial(
            np.maximum(n_param, 0.01),
            np.clip(p_param, 1e-8, 1.0 - 1e-8),
            size=(n_samples,) + mu.shape,
        ).squeeze()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"result": self.result_, "alpha": self.alpha_},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("BatterHModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BatterHModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls()
        model.result_ = data["result"]
        model.alpha_ = data["alpha"]
        logger.info("BatterHModel loaded from %s", path)
        return model
