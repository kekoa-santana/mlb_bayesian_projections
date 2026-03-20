"""
Leverage-aware pitcher exit model.

Predicts the probability that a starting pitcher is removed after each PA,
based on cumulative pitch count, game state (inning, outs, score, runners),
recent trouble, and pitcher/team tendencies.

Uses logistic regression trained on historical starter exit data.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Hard caps
MAX_PITCHES = 130
MAX_OUTS = 27  # Complete game

# Score differential buckets
SCORE_DIFF_BINS = [-np.inf, -3, 0, 1, 4, np.inf]
SCORE_DIFF_LABELS = [
    "trailing_4plus",
    "trailing_1_3",
    "tied",
    "leading_1_3",
    "leading_4plus",
]

# Default pitcher exit pitch count (used when no data available)
DEFAULT_AVG_EXIT_PITCHES = 88.0
DEFAULT_STD_EXIT_PITCHES = 14.0

# Shrinkage for pitcher-specific tendencies
_SHRINKAGE_K = 10  # starts needed for full reliability


def _encode_score_diff(score_diff: int | np.ndarray) -> int | np.ndarray:
    """Encode score differential into bucket index (0-4).

    Parameters
    ----------
    score_diff : int or np.ndarray
        Score differential from pitcher's team perspective.

    Returns
    -------
    int or np.ndarray
        Bucket index.
    """
    return np.digitize(score_diff, [-3, 0, 1, 4])


def _encode_runners(runners: int | np.ndarray) -> int | np.ndarray:
    """Encode runners on base into bucket (0, 1, 2+).

    Parameters
    ----------
    runners : int or np.ndarray
        Count of runners on base.

    Returns
    -------
    int or np.ndarray
        Bucket: 0, 1, or 2.
    """
    return np.minimum(runners, 2)


class ExitModel:
    """Logistic regression model for pitcher exit probability.

    The model predicts P(exit | game_state) after each PA. Features are
    designed to capture pitch count fatigue, leverage, and manager tendencies.

    Attributes
    ----------
    model : LogisticRegression or None
        Fitted model (None before training).
    scaler : StandardScaler or None
        Feature scaler (None before training).
    """

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self._feature_names: list[str] = []

    def _build_features(
        self,
        cumulative_pitches: np.ndarray,
        inning: np.ndarray,
        inning_outs: np.ndarray,
        score_diff: np.ndarray,
        runners: np.ndarray,
        tto: np.ndarray,
        recent_trouble: np.ndarray,
        pitcher_avg_pitches: np.ndarray,
    ) -> np.ndarray:
        """Build feature matrix from game state arrays.

        Parameters
        ----------
        cumulative_pitches : np.ndarray
            Total pitches thrown.
        inning : np.ndarray
            Current inning.
        inning_outs : np.ndarray
            Outs in current inning (0, 1, 2).
        score_diff : np.ndarray
            Score differential (pitcher team perspective).
        runners : np.ndarray
            Runners on base count.
        tto : np.ndarray
            Times-through-order (1, 2, 3).
        recent_trouble : np.ndarray
            Count of BB/H in last 2 PAs.
        pitcher_avg_pitches : np.ndarray
            Pitcher's historical average exit pitch count.

        Returns
        -------
        np.ndarray
            Feature matrix, shape (n_samples, n_features).
        """
        n = len(cumulative_pitches)

        # Pitch count features
        pitches = cumulative_pitches.astype(float)
        pitches_sq = pitches ** 2 / 10000.0  # Quadratic term, scaled

        # Deviation from pitcher's expected exit point
        pitches_vs_avg = pitches - pitcher_avg_pitches

        # Inning end indicator (outs = 0 means just completed an inning)
        at_inning_start = (inning_outs == 0).astype(float)
        mid_inning = (inning_outs > 0).astype(float)

        # Score buckets
        score_bucket = _encode_score_diff(score_diff).astype(float)

        # Runner pressure
        runner_bucket = _encode_runners(runners).astype(float)
        runners_on = (runners > 0).astype(float)

        # TTO
        tto_float = tto.astype(float)
        tto_3plus = (tto >= 3).astype(float)

        # Recent trouble
        trouble = recent_trouble.astype(float)
        in_trouble = (recent_trouble >= 2).astype(float)

        # Interaction: high pitch count × inning start (manager pulls between innings)
        pitches_x_inning_start = pitches * at_inning_start / 100.0

        # Interaction: runners on × high pitch count
        pitches_x_runners = pitches * runners_on / 100.0

        features = np.column_stack([
            pitches,
            pitches_sq,
            pitches_vs_avg,
            at_inning_start,
            mid_inning,
            inning.astype(float),
            score_bucket,
            runner_bucket,
            runners_on,
            tto_float,
            tto_3plus,
            trouble,
            in_trouble,
            pitches_x_inning_start,
            pitches_x_runners,
        ])

        self._feature_names = [
            "pitches", "pitches_sq", "pitches_vs_avg",
            "at_inning_start", "mid_inning", "inning",
            "score_bucket", "runner_bucket", "runners_on",
            "tto", "tto_3plus",
            "trouble", "in_trouble",
            "pitches_x_inning_start", "pitches_x_runners",
        ]

        return features

    def train(
        self,
        training_data: pd.DataFrame,
        pitcher_tendencies: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Train the exit model on historical PA-level data.

        Parameters
        ----------
        training_data : pd.DataFrame
            Output of get_exit_model_training_data(). One row per PA.
        pitcher_tendencies : pd.DataFrame, optional
            Output of get_pitcher_exit_tendencies(). Per-pitcher avg exit
            pitch counts.

        Returns
        -------
        dict[str, Any]
            Training metrics: accuracy, AUC, n_samples, n_positive.
        """
        from sklearn.metrics import roc_auc_score

        df = training_data.copy()

        # Compute derived features from raw data
        # Recent trouble: count of BB/H in last 2 PAs per pitcher-game
        df["is_trouble"] = df["events"].isin([
            "walk", "single", "double", "triple", "home_run",
            "hit_by_pitch",
        ]).astype(int)

        # Rolling 2-PA trouble count within each pitcher-game
        df["recent_trouble"] = (
            df.groupby(["pitcher_id", "game_pk"])["is_trouble"]
            .transform(lambda x: x.rolling(2, min_periods=1).sum().shift(0))
        )
        # The rolling should look at the *current* and previous PA
        # Recompute: include current PA outcome for exit decision
        df["recent_trouble"] = (
            df.groupby(["pitcher_id", "game_pk"])["is_trouble"]
            .transform(lambda x: x.rolling(2, min_periods=1).sum())
        )

        # TTO from pitcher_pa_number
        df["tto"] = np.minimum((df["pitcher_pa_number"] - 1) // 9 + 1, 3)

        # Inning outs
        df["inning_outs"] = df["outs_when_up"].astype(int)

        # Runners: approximate from game state
        # We don't have exact runner data, so use a proxy:
        # recent non-out events suggest runners on base
        df["runners"] = (
            df.groupby(["pitcher_id", "game_pk"])["is_trouble"]
            .transform(lambda x: x.rolling(3, min_periods=1).sum())
        ).clip(0, 3).astype(int)

        # Pitcher average exit pitches
        if pitcher_tendencies is not None and not pitcher_tendencies.empty:
            # For each row, look up pitcher's avg from *prior* seasons
            # to avoid leakage
            tend_shifted = pitcher_tendencies.copy()
            tend_shifted["season"] = tend_shifted["season"] + 1

            df = df.merge(
                tend_shifted[["pitcher_id", "season", "avg_pitches"]],
                on=["pitcher_id", "season"],
                how="left",
            )
            df["avg_pitches"] = df["avg_pitches"].fillna(DEFAULT_AVG_EXIT_PITCHES)
        else:
            df["avg_pitches"] = DEFAULT_AVG_EXIT_PITCHES

        # Build feature matrix
        X = self._build_features(
            cumulative_pitches=df["cumulative_pitches"].values,
            inning=df["inning"].values,
            inning_outs=df["inning_outs"].values,
            score_diff=df["score_diff"].values,
            runners=df["runners"].values,
            tto=df["tto"].values,
            recent_trouble=df["recent_trouble"].values,
            pitcher_avg_pitches=df["avg_pitches"].values,
        )
        y = df["is_last_pa"].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit logistic regression — no class weight balancing so that
        # predicted probabilities reflect the true per-PA exit rate
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        # Compute metrics
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        accuracy = self.model.score(X_scaled, y)

        metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "positive_rate": float(y.mean()),
        }

        logger.info(
            "Exit model trained: AUC=%.4f, accuracy=%.4f, n=%d (%.1f%% positive)",
            auc, accuracy, len(y), y.mean() * 100,
        )

        # Log feature importances
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            for name, coef in sorted(
                zip(self._feature_names, coefs),
                key=lambda x: abs(x[1]),
                reverse=True,
            ):
                logger.debug("  %s: %.4f", name, coef)

        return metrics

    def predict_exit_prob(
        self,
        cumulative_pitches: int | np.ndarray,
        inning: int | np.ndarray,
        inning_outs: int | np.ndarray,
        score_diff: int | np.ndarray,
        runners: int | np.ndarray,
        tto: int | np.ndarray,
        recent_trouble: int | np.ndarray,
        pitcher_avg_pitches: float | np.ndarray,
    ) -> float | np.ndarray:
        """Predict probability of pitcher exit after this PA.

        Parameters
        ----------
        cumulative_pitches : int or np.ndarray
            Total pitches thrown.
        inning : int or np.ndarray
            Current inning.
        inning_outs : int or np.ndarray
            Outs in current inning.
        score_diff : int or np.ndarray
            Score differential (pitcher team perspective).
        runners : int or np.ndarray
            Runners on base count.
        tto : int or np.ndarray
            Times-through-order.
        recent_trouble : int or np.ndarray
            Count of BB/H in last 2 PAs.
        pitcher_avg_pitches : float or np.ndarray
            Pitcher's historical average exit pitch count.

        Returns
        -------
        float or np.ndarray
            Exit probability.
        """
        if self.model is None or self.scaler is None:
            return self._fallback_exit_prob(cumulative_pitches, pitcher_avg_pitches)

        # Ensure arrays
        inputs = [
            cumulative_pitches, inning, inning_outs, score_diff,
            runners, tto, recent_trouble, pitcher_avg_pitches,
        ]
        is_scalar = np.isscalar(inputs[0])
        inputs = [np.atleast_1d(np.asarray(x, dtype=float)) for x in inputs]

        X = self._build_features(*inputs)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]

        # Hard caps
        probs = np.where(
            inputs[0] >= MAX_PITCHES, 1.0, probs
        )

        if is_scalar:
            return float(probs[0])
        return probs

    @staticmethod
    def _fallback_exit_prob(
        cumulative_pitches: int | np.ndarray,
        pitcher_avg_pitches: float | np.ndarray,
    ) -> float | np.ndarray:
        """Simple sigmoid fallback when model is not trained.

        Parameters
        ----------
        cumulative_pitches : int or np.ndarray
            Total pitches thrown.
        pitcher_avg_pitches : float or np.ndarray
            Expected exit pitch count.

        Returns
        -------
        float or np.ndarray
            Exit probability per PA (~5% at mean, scaling up sharply).
        """
        # Sigmoid centered at pitcher's avg exit pitches
        x = (np.asarray(cumulative_pitches, dtype=float)
             - np.asarray(pitcher_avg_pitches, dtype=float))
        # Steepness: ~50% exit prob at avg+5, ~90% at avg+15
        base_prob = 1.0 / (1.0 + np.exp(-0.15 * x))

        # Per-PA probability: convert cumulative exit prob to per-PA hazard
        # Approximate: if ~22 PA per game, and exit prob ramps over last ~5 PA
        return np.clip(base_prob * 0.25, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        """Save trained model to disk.

        Parameters
        ----------
        path : str or Path
            Output file path (.pkl).
        """
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self._feature_names,
            }, f)
        logger.info("Exit model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load trained model from disk.

        Parameters
        ----------
        path : str or Path
            Input file path (.pkl).
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._feature_names = data["feature_names"]
        logger.info("Exit model loaded from %s", path)
