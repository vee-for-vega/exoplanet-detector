"""
Naive Bayes baseline classifier.

Baseline 1 — simplest probabilistic model operating on
hand-engineered features. Establishes a performance floor
before graduating to CNNs.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.naive_bayes import GaussianNB

from src.utils.config import (
    ENGINEERED_FEATURES, 
    MODELS_DIR, 
    NB_ALPHA, 
)


class NaiveBayesModel:
    """
    Wrapper around sklearn's GaussianNB with project conventions.

    All models in this project follow the same interface:
    - fit(X, y) -> trains the model
    - predict(X) -> returns class predictions (0 or 1)
    - predict_proba(X) -> returns probability of class 1
    - save(path) / load(path) -> persistence

    This consistent interface is what lets the eval harness
    treat all models interchangeably.
    """

    def __init__(self, feature_columns: list = None):
        self.feature_columns = feature_columns or ENGINEERED_FEATURES
        self.model = GaussianNB(var_smoothing=NB_ALPHA * 1e-9)
        self.name = "NaiveBayes"

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train on feature matrix and labels."""
        X_subset = self._select_features(X)
        self.model.fit(X_subset, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions."""
        X_subset = self._select_features(X)
        return self.model.predict(X_subset)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of being a planet (class 1)."""
        X_subset = self._select_features(X)
        return self.model.predict_proba(X_subset)[:, 1]

    def _select_features(self, X: pd.DataFrame) -> np.ndarray:
        """Select and validate feature columns."""
        available = [c for c in self.feature_columns if c in X.columns]
        return X[available].values

    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "naive_bayes.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = None):
        if path is None:
            path = MODELS_DIR / "naive_bayes.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
