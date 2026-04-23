"""
LightGBM multiclass plate-appearance outcome model.

Predicts PA results as one of 5 classes:
    0 = K  (strikeout)
    1 = BB (walk)
    2 = HBP (hit by pitch)
    3 = HR (home run)
    4 = BIP (ball in play -- includes all batted-ball outcomes and non-K outs)

The model wraps a fitted LightGBM multiclass classifier and provides
a clean predict interface returning (n, 5) probability arrays.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Target class indices
CLS_K = 0
CLS_BB = 1
CLS_HBP = 2
CLS_HR = 3
CLS_BIP = 4
CLASS_NAMES = ["K", "BB", "HBP", "HR", "BIP"]
NUM_CLASSES = 5

# Event-to-class mapping
EVENT_TO_CLASS: dict[str, int] = {
    "strikeout": CLS_K,
    "strikeout_double_play": CLS_K,
    "walk": CLS_BB,
    "intent_walk": CLS_BB,
    "hit_by_pitch": CLS_HBP,
    "home_run": CLS_HR,
    # Everything else maps to BIP (default)
}


def map_events_to_classes(events: pd.Series) -> pd.Series:
    """Map PA event strings to 5-class integer targets."""
    return events.map(EVENT_TO_CLASS).fillna(CLS_BIP).astype(int)


class PAOutcomeLGBM:
    """LightGBM multiclass PA outcome model."""

    FEATURES: ClassVar[list[str]] = [
        "pitcher_k_pct",
        "pitcher_bb_pct",
        "pitcher_hr_pct",
        "batter_k_pct",
        "batter_bb_pct",
        "batter_hr_pct",
        "platoon",
        "times_through_order",
        "outs_when_up",
        "inning",
        "score_diff",
        "park_run_factor",
        "temperature",
        "is_dome",
    ]

    DEFAULT_PARAMS: ClassVar[dict] = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self) -> None:
        self.model_: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        params: dict | None = None,
    ) -> "PAOutcomeLGBM":
        """Fit the LightGBM multiclass model."""
        p = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model_ = lgb.LGBMClassifier(**p)

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(50, verbose=True),
                lgb.log_evaluation(50),
            ]

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train[self.FEATURES]
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val[self.FEATURES]
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model_.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict_pa_probs(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict PA outcome probabilities.

        Returns
        -------
        np.ndarray of shape (n, 5) with columns [K, BB, HBP, HR, BIP].
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
        if isinstance(X, pd.DataFrame):
            X = X[self.FEATURES]
        return self.model_.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Save fitted model to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model_, "features": self.FEATURES}, path)
        logger.info("Saved PAOutcomeLGBM to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PAOutcomeLGBM":
        """Load a fitted model from disk."""
        data = joblib.load(path)
        obj = cls()
        obj.model_ = data["model"]
        return obj
