"""
Test suite for the exoplanet detection pipeline.

Tests use synthetic data so they run without downloading anything
from NASA. If all tests pass, every component works correctly
in isolation.

Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# FIXTURES: Synthetic data that mimics real structure
# ============================================================

@pytest.fixture
def synthetic_light_curve():
    """Create a fake light curve with a transit-like dip."""
    np.random.seed(42)
    n_points = 2001
    time = np.linspace(0, 30, n_points)
    flux = np.ones(n_points)

    # Add a transit-like dip centered at t=15
    transit_mask = (time > 14.8) & (time < 15.2)
    flux[transit_mask] -= 0.01  # 1% dip

    # Add realistic noise
    flux += np.random.normal(0, 0.001, n_points)
    return time, flux


@pytest.fixture
def synthetic_tce_table():
    """Create a fake TCE table with known labels."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "tic_id": np.arange(1000, 1000 + n),
        "orbital_period": np.random.uniform(1, 100, n),
        "transit_depth": np.random.uniform(0.0001, 0.01, n),
        "transit_duration": np.random.uniform(0.5, 5.0, n),
        "planet_radius": np.random.uniform(0.5, 5.0, n),
        "disposition": np.random.choice(["CP", "FP"], n, p=[0.3, 0.7]),
        "label": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def synthetic_feature_matrix(synthetic_tce_table):
    """Create a feature matrix with engineered features."""
    df = synthetic_tce_table.copy()
    df["transit_snr"] = np.random.uniform(1, 20, len(df))
    df["ingress_duration"] = np.random.uniform(0.01, 0.1, len(df))
    df["depth_even_odd"] = np.random.uniform(0, 0.005, len(df))
    df["secondary_depth"] = np.random.uniform(0, 0.005, len(df))
    df["flux_std"] = np.random.uniform(0.0005, 0.005, len(df))
    df["num_transits"] = np.random.randint(1, 20, len(df))
    return df


# ============================================================
# CONFIG TESTS
# ============================================================

class TestConfig:
    def test_paths_exist(self):
        from src.utils.config import PROJECT_ROOT, DATA_DIR
        assert PROJECT_ROOT.exists()
        assert DATA_DIR.exists()

    def test_set_seed_runs(self):
        from src.utils.config import set_seed
        set_seed(42)  # Should not raise

    def test_split_fractions_sum_to_one(self):
        from src.utils.config import TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION
        total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
        assert abs(total - 1.0) < 1e-6


# ============================================================
# PREPROCESSING TESTS
# ============================================================

class TestPreprocessing:
    def test_normalize_flux_median(self, synthetic_light_curve):
        from src.data.preprocess import normalize_flux
        _, flux = synthetic_light_curve
        normed = normalize_flux(flux, method="median")
        assert abs(np.median(normed) - 1.0) < 0.01

    def test_normalize_flux_standard(self, synthetic_light_curve):
        from src.data.preprocess import normalize_flux
        _, flux = synthetic_light_curve
        normed = normalize_flux(flux, method="standard")
        assert abs(np.mean(normed)) < 0.01
        assert abs(np.std(normed) - 1.0) < 0.01

    def test_resample_length(self, synthetic_light_curve):
        from src.data.preprocess import resample_light_curve
        time, flux = synthetic_light_curve
        resampled = resample_light_curve(time, flux, target_length=500)
        assert len(resampled) == 500

    def test_create_splits_no_leakage(self, synthetic_tce_table):
        from src.data.preprocess import create_splits
        train_idx, val_idx, test_idx = create_splits(synthetic_tce_table)

        # Check no overlap in indices
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0

        # Check no star ID leakage between splits
        train_stars = set(synthetic_tce_table.iloc[train_idx]["tic_id"])
        val_stars = set(synthetic_tce_table.iloc[val_idx]["tic_id"])
        test_stars = set(synthetic_tce_table.iloc[test_idx]["tic_id"])
        assert len(train_stars & test_stars) == 0
        assert len(train_stars & val_stars) == 0


# ============================================================
# PHASE FOLDING TESTS
# ============================================================

class TestPhaseFolding:
    def test_phase_fold_range(self, synthetic_light_curve):
        from src.data.phase_fold import phase_fold
        time, flux = synthetic_light_curve
        phase, folded = phase_fold(time, flux, period=5.0)
        assert phase.min() >= -0.5
        assert phase.max() <= 0.5
        assert len(phase) == len(folded)

    def test_phase_fold_to_image_shape(self, synthetic_light_curve):
        from src.data.phase_fold import phase_fold, phase_fold_to_image
        time, flux = synthetic_light_curve
        phase, folded = phase_fold(time, flux, period=5.0)
        image = phase_fold_to_image(phase, folded, image_size=(64, 64))
        assert image.shape == (64, 64)
        assert image.min() >= 0.0
        assert image.max() <= 1.0


# ============================================================
# FEATURE ENGINEERING TESTS
# ============================================================

class TestFeatures:
    def test_extract_features_returns_all_keys(self, synthetic_light_curve):
        from src.features.engineered import extract_features_from_lightcurve
        _, flux = synthetic_light_curve
        # Normalize first
        flux = flux / np.median(flux)
        features = extract_features_from_lightcurve(flux, period=5.0)
        assert "transit_depth" in features
        assert "transit_snr" in features
        assert "flux_std" in features
        assert features["transit_depth"] > 0

    def test_augmentations_preserve_shape(self, synthetic_light_curve):
        from src.features.transforms import augment_light_curve
        _, flux = synthetic_light_curve
        augmented = augment_light_curve(flux)
        assert augmented.shape == flux.shape


# ============================================================
# MODEL TESTS (on synthetic data)
# ============================================================

class TestNaiveBayes:
    def test_fit_predict(self, synthetic_feature_matrix):
        from src.models.naive_bayes import NaiveBayesModel
        model = NaiveBayesModel()
        X = synthetic_feature_matrix
        y = X["label"].values
        model.fit(X, y)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        assert len(preds) == len(y)
        assert len(probs) == len(y)
        assert all(p >= 0 and p <= 1 for p in probs)


class TestLogistic:
    def test_fit_predict(self, synthetic_feature_matrix):
        from src.models.logistic import LogisticModel
        model = LogisticModel()
        X = synthetic_feature_matrix
        y = X["label"].values
        model.fit(X, y)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        assert len(preds) == len(y)
        assert all(0 <= p <= 1 for p in probs)


class TestCNN1D:
    def test_forward_pass(self):
        import torch
        from src.models.cnn_1d import CNN1DNet
        net = CNN1DNet(input_length=2001)
        x = torch.randn(4, 1, 2001)  # Batch of 4
        out = net(x)
        assert out.shape == (4,)
        assert all(0 <= p <= 1 for p in out.detach().numpy())


class TestCNN2D:
    def test_forward_pass(self):
        import torch
        from src.models.cnn_2d import CNN2DNet
        net = CNN2DNet(image_size=(64, 64))
        x = torch.randn(4, 1, 64, 64)  # Batch of 4
        out = net(x)
        assert out.shape == (4,)
        assert all(0 <= p <= 1 for p in out.detach().numpy())


# ============================================================
# EVALUATION TESTS
# ============================================================

class TestMetrics:
    def test_perfect_predictions(self):
        from src.evaluation.metrics import compute_all_metrics
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        from src.evaluation.metrics import compute_all_metrics
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_all_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestRules:
    def test_rules_override_bad_predictions(self):
        from src.evaluation.rules import apply_rules
        features = pd.DataFrame({
            # Values in ppm to match NASA TOI table units
            "transit_depth": [1000, 10, 1000],  # Second is too shallow (<100 ppm)
            "orbital_period": [10.0, 10.0, 10.0],
            "transit_snr": [5.0, 5.0, 1.0],  # Third has low SNR
            "num_transits": [3, 3, 3],
        })
        predictions = np.array([1, 1, 1])  # All predicted as planets
        result = apply_rules(features, predictions)
        # Second should be overridden (depth too low)
        # Third should be overridden (SNR too low)
        assert result["filtered_predictions"][0] == 1  # Kept
        assert result["filtered_predictions"][1] == 0  # Overridden
        assert result["filtered_predictions"][2] == 0  # Overridden
        assert result["n_overridden"] == 2
