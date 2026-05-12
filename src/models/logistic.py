"""
Logistic Regression baseline classifier.

Baseline 2 — linear model with L2 regularization.
Handles correlated features better than Naive Bayes,
and provides interpretable feature coefficients.
"""

import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    ENGINEERED_FEATURES,
    LR_C,
    LR_MAX_ITER,
    MODELS_DIR,
    RANDOM_SEED,
)


class LogisticModel:
    """
    Logistic Regression with standardized features.

    Key detail: Standardize features (zero mean, unit variance)
    before fitting. This is critical because LR is sensitive to
    feature scale — without standardization, features with larger
    magnitudes dominate the regularization penalty.
    """

    def __init__(self, feature_columns: list = None):
        self.feature_columns = feature_columns or ENGINEERED_FEATURES
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            class_weight="balanced",  # Handles class imbalance
            random_state=RANDOM_SEED,
        )
        self.name = "LogisticRegression"

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_subset = self._select_features(X)
        self._fitted_features = [c for c in self.feature_columns if c in X.columns]
        X_scaled = self.scaler.fit_transform(X_subset)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_subset = self._select_features(X)
        X_scaled = self.scaler.transform(X_subset)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_subset = self._select_features(X)
        X_scaled = self.scaler.transform(X_subset)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> dict:
        """Return coefficient magnitudes as feature importance."""
        coeffs = self.model.coef_[0]
        return dict(zip(self._fitted_features, np.abs(coeffs)))

    def _select_features(self, X: pd.DataFrame) -> np.ndarray:
        available = [c for c in self.feature_columns if c in X.columns]
        return X[available].values

    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "logistic.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = None):
        if path is None:
            path = MODELS_DIR / "logistic.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
