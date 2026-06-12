"""
Gradient-boosted trees baseline classifier.

Baseline 3 — histogram gradient boosting (sklearn's LightGBM-style
implementation). Learns nonlinear feature interactions that the linear
baselines cannot, handles NaN natively, and is the strongest model in
this project that runs on metadata alone (no light curves).
"""

import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

from src.utils.config import (
    ENGINEERED_FEATURES,
    MODELS_DIR,
    RANDOM_SEED,
)


class GradientBoostModel:
    """
    HistGradientBoostingClassifier with project conventions.

    Same interface as the other models (fit / predict / predict_proba /
    save / load) so the eval harness treats it interchangeably.
    class_weight='balanced' handles the Kepler 1:11 imbalance.
    """

    def __init__(self, feature_columns: list = None):
        self.feature_columns = feature_columns or ENGINEERED_FEATURES
        self.model = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        self.name = "GradientBoost"

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model.fit(self._select_features(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._select_features(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._select_features(X))[:, 1]

    def _select_features(self, X: pd.DataFrame) -> np.ndarray:
        available = [c for c in self.feature_columns if c in X.columns]
        return X[available].values

    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "gradient_boost.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = None):
        if path is None:
            path = MODELS_DIR / "gradient_boost.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
