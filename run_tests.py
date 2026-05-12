"""
Standalone test runner — no pytest or torch required.

Tests every component that doesn't need PyTorch:
- Config, preprocessing, phase folding, feature engineering
- Naive Bayes, Logistic Regression
- Metrics, FOL rules

Run: python run_tests.py
"""

import sys
import os
import traceback
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
skipped = 0


def test(name):
    """Decorator to register and run a test."""
    def decorator(fn):
        global passed, failed, skipped
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            traceback.print_exc(limit=2)
            failed += 1
    return decorator


def skip(name, reason=""):
    """Mark a test as skipped."""
    global skipped
    print(f"  SKIP  {name} ({reason})")
    skipped += 1


# ============================================================
# SYNTHETIC DATA
# ============================================================

np.random.seed(42)

# Fake light curve with a transit dip
_n = 2001
_time = np.linspace(0, 30, _n)
_flux = np.ones(_n)
_transit_mask = (_time > 14.8) & (_time < 15.2)
_flux[_transit_mask] -= 0.01
_flux += np.random.normal(0, 0.001, _n)

# Fake TCE table
_tce = pd.DataFrame({
    "tic_id": np.arange(1000, 1100),
    "orbital_period": np.random.uniform(1, 100, 100),
    "transit_depth": np.random.uniform(0.0001, 0.01, 100),
    "transit_duration": np.random.uniform(0.5, 5.0, 100),
    "planet_radius": np.random.uniform(0.5, 5.0, 100),
    "disposition": np.random.choice(["CP", "FP"], 100, p=[0.3, 0.7]),
    "label": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
})

# Feature matrix
_features = _tce.copy()
_features["transit_snr"] = np.random.uniform(1, 20, 100)
_features["ingress_duration"] = np.random.uniform(0.01, 0.1, 100)
_features["depth_even_odd"] = np.random.uniform(0, 0.005, 100)
_features["secondary_depth"] = np.random.uniform(0, 0.005, 100)
_features["flux_std"] = np.random.uniform(0.0005, 0.005, 100)
_features["num_transits"] = np.random.randint(1, 20, 100)


# ============================================================
# CONFIG TESTS
# ============================================================

print("\n--- Config ---")


@test("Paths exist")
def _():
    from src.utils.config import PROJECT_ROOT, DATA_DIR
    assert PROJECT_ROOT.exists(), "PROJECT_ROOT doesn't exist"
    assert DATA_DIR.exists(), "DATA_DIR doesn't exist"


@test("Split fractions sum to 1.0")
def _():
    from src.utils.config import TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION
    total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
    assert abs(total - 1.0) < 1e-6, f"Sum is {total}, not 1.0"


@test("set_seed runs without error")
def _():
    from src.utils.config import set_seed
    set_seed(42)


# ============================================================
# PREPROCESSING TESTS
# ============================================================

print("\n--- Preprocessing ---")


@test("normalize_flux median: median ≈ 1.0")
def _():
    from src.data.preprocess import normalize_flux
    normed = normalize_flux(_flux.copy(), method="median")
    assert abs(np.median(normed) - 1.0) < 0.01, f"Median is {np.median(normed)}"


@test("normalize_flux standard: mean ≈ 0, std ≈ 1")
def _():
    from src.data.preprocess import normalize_flux
    normed = normalize_flux(_flux.copy(), method="standard")
    assert abs(np.mean(normed)) < 0.01, f"Mean is {np.mean(normed)}"
    assert abs(np.std(normed) - 1.0) < 0.01, f"Std is {np.std(normed)}"


@test("normalize_flux minmax: range [0, 1]")
def _():
    from src.data.preprocess import normalize_flux
    normed = normalize_flux(_flux.copy(), method="minmax")
    assert normed.min() >= -0.001, f"Min is {normed.min()}"
    assert normed.max() <= 1.001, f"Max is {normed.max()}"


@test("resample_light_curve produces correct length")
def _():
    from src.data.preprocess import resample_light_curve
    resampled = resample_light_curve(_time, _flux, target_length=500)
    assert len(resampled) == 500, f"Length is {len(resampled)}"


@test("create_splits: no index overlap")
def _():
    from src.data.preprocess import create_splits
    train_idx, val_idx, test_idx = create_splits(_tce)
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap"


@test("create_splits: no star ID leakage")
def _():
    from src.data.preprocess import create_splits
    train_idx, val_idx, test_idx = create_splits(_tce)
    train_stars = set(_tce.iloc[train_idx]["tic_id"])
    val_stars = set(_tce.iloc[val_idx]["tic_id"])
    test_stars = set(_tce.iloc[test_idx]["tic_id"])
    assert len(train_stars & test_stars) == 0, "Star ID leak train↔test"
    assert len(train_stars & val_stars) == 0, "Star ID leak train↔val"


# ============================================================
# PHASE FOLDING TESTS
# ============================================================

print("\n--- Phase Folding ---")


@test("phase_fold: phase range [-0.5, 0.5]")
def _():
    from src.data.phase_fold import phase_fold
    phase, folded = phase_fold(_time, _flux, period=5.0)
    assert phase.min() >= -0.5, f"Min phase is {phase.min()}"
    assert phase.max() <= 0.5, f"Max phase is {phase.max()}"
    assert len(phase) == len(folded)


@test("phase_fold_to_image: correct shape and range")
def _():
    from src.data.phase_fold import phase_fold, phase_fold_to_image
    phase, folded = phase_fold(_time, _flux, period=5.0)
    image = phase_fold_to_image(phase, folded, image_size=(64, 64))
    assert image.shape == (64, 64), f"Shape is {image.shape}"
    assert image.min() >= 0.0, f"Min pixel is {image.min()}"
    assert image.max() <= 1.0, f"Max pixel is {image.max()}"


# ============================================================
# FEATURE ENGINEERING TESTS
# ============================================================

print("\n--- Feature Engineering ---")


@test("extract_features returns expected keys")
def _():
    from src.features.engineered import extract_features_from_lightcurve
    flux_normed = _flux / np.median(_flux)
    features = extract_features_from_lightcurve(flux_normed, period=5.0)
    for key in ["transit_depth", "transit_snr", "flux_std", "transit_duration"]:
        assert key in features, f"Missing key: {key}"
    assert features["transit_depth"] > 0, "Transit depth should be > 0"


@test("augment_light_curve preserves shape")
def _():
    from src.features.transforms import augment_light_curve
    augmented = augment_light_curve(_flux.copy())
    assert augmented.shape == _flux.shape, f"Shape changed: {augmented.shape}"


@test("augment_image preserves shape")
def _():
    from src.features.transforms import augment_image
    fake_img = np.random.rand(64, 64).astype(np.float32)
    augmented = augment_image(fake_img)
    assert augmented.shape == (64, 64), f"Shape changed: {augmented.shape}"


# ============================================================
# MODEL TESTS (sklearn only)
# ============================================================

print("\n--- Models (sklearn) ---")


@test("NaiveBayes: fit + predict + predict_proba")
def _():
    from src.models.naive_bayes import NaiveBayesModel
    model = NaiveBayesModel()
    y = _features["label"].values
    model.fit(_features, y)
    preds = model.predict(_features)
    probs = model.predict_proba(_features)
    assert len(preds) == len(y), f"Wrong prediction count: {len(preds)}"
    assert len(probs) == len(y), f"Wrong probability count: {len(probs)}"
    assert all(0 <= p <= 1 for p in probs), "Probabilities out of range"


@test("LogisticRegression: fit + predict + predict_proba")
def _():
    from src.models.logistic import LogisticModel
    model = LogisticModel()
    y = _features["label"].values
    model.fit(_features, y)
    preds = model.predict(_features)
    probs = model.predict_proba(_features)
    assert len(preds) == len(y)
    assert all(0 <= p <= 1 for p in probs), "Probabilities out of range"


@test("LogisticRegression: feature importance returns values")
def _():
    from src.models.logistic import LogisticModel
    model = LogisticModel()
    y = _features["label"].values
    model.fit(_features, y)
    importance = model.get_feature_importance()
    assert len(importance) > 0, "No feature importances returned"


# CNN tests require PyTorch
skip("CNN1D: forward pass", "PyTorch not installed")
skip("CNN2D: forward pass", "PyTorch not installed")


# ============================================================
# EVALUATION TESTS
# ============================================================

print("\n--- Evaluation ---")


@test("metrics: perfect predictions score 1.0")
def _():
    from src.evaluation.metrics import compute_all_metrics
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_all_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0, f"Accuracy: {m['accuracy']}"
    assert m["precision"] == 1.0, f"Precision: {m['precision']}"
    assert m["recall"] == 1.0, f"Recall: {m['recall']}"
    assert m["f1"] == 1.0, f"F1: {m['f1']}"


@test("metrics: all-wrong predictions score 0.0")
def _():
    from src.evaluation.metrics import compute_all_metrics
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_all_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


@test("FOL rules: override physically implausible predictions")
def _():
    from src.evaluation.rules import apply_rules
    features = pd.DataFrame({
        # Values in ppm to match NASA TOI table units
        "transit_depth": [1000, 10, 1000],  # Second is too shallow (<100 ppm)
        "orbital_period": [10.0, 10.0, 10.0],
        "transit_snr": [5.0, 5.0, 1.0],
        "num_transits": [3, 3, 3],
    })
    preds = np.array([1, 1, 1])
    result = apply_rules(features, preds)
    assert result["filtered_predictions"][0] == 1, "Should keep valid planet"
    assert result["filtered_predictions"][1] == 0, "Should reject: depth too low"
    assert result["filtered_predictions"][2] == 0, "Should reject: SNR too low"
    assert result["n_overridden"] == 2


@test("FOL rules: don't touch negative predictions")
def _():
    from src.evaluation.rules import apply_rules
    features = pd.DataFrame({
        "transit_depth": [10],   # Too shallow but doesn't matter — already negative
        "orbital_period": [10.0],
        "transit_snr": [1.0],
        "num_transits": [3],
    })
    preds = np.array([0])  # Already predicted as not-planet
    result = apply_rules(features, preds)
    assert result["filtered_predictions"][0] == 0
    assert result["n_overridden"] == 0


# ============================================================
# VISUALIZATION TESTS
# ============================================================

print("\n--- Visualization ---")


@test("plot_confusion_matrix runs without error")
def _():
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    from src.utils.visualization import plot_confusion_matrix
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    fig = plot_confusion_matrix(y_true, y_pred, "Test")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


@test("plot_roc_curve runs without error")
def _():
    import matplotlib
    matplotlib.use("Agg")
    from src.utils.visualization import plot_roc_curve
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.8, 0.9])
    fig = plot_roc_curve(y_true, y_prob, "Test")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close("all")


# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'=' * 50}")
print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
print(f"{'=' * 50}")

if failed > 0:
    print("\n  Some tests FAILED. Fix these before proceeding.")
    sys.exit(1)
else:
    print("\n  All tests PASSED. Ready to download data and train.")
    sys.exit(0)