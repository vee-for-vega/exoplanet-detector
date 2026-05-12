"""
Train 1D and 2D CNN models on phase-folded light curve data.

Trains two models:
  1D CNN — on phase-folded 1D signals (256 median-binned flux values)
  2D CNN — on phase-folded 2D images (64x64 density maps)

Run: python train_cnns.py

Prerequisites:
  pip install torch
  python -m src.data.download --lightcurves --limit 300
  python -m src.data.preprocess
  python -m src.data.phase_fold
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import (
    RAW_DIR,
    RESULTS_DIR, MODELS_DIR, RANDOM_SEED, SPLIT_SEED, set_seed,
    PHASE_FOLD_IMAGE_SIZE, PHASE_FOLD_1D_LENGTH,
    TEST_FRACTION, VAL_FRACTION,
)
from src.data.phase_fold import get_images_dir, get_folded_1d_dir

ACTIVE_IMAGES_DIR = get_images_dir(PHASE_FOLD_IMAGE_SIZE)
ACTIVE_FOLDED_1D_DIR = get_folded_1d_dir(PHASE_FOLD_1D_LENGTH)

set_seed(RANDOM_SEED)


# ============================================================
# CLEAR PREVIOUS CNN RESULTS
# ============================================================

cnn_result_files = ["cnn_test_results.csv", "all_models_comparison.csv"]
for fname in cnn_result_files:
    fpath = RESULTS_DIR / fname
    if fpath.exists():
        fpath.unlink()


# ============================================================
# CHECK PYTORCH
# ============================================================

try:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | Device: {device}")
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)


# ============================================================
# LOAD DATA
# ============================================================

tce_df = pd.read_csv(RAW_DIR / "tce_table_clean.csv")

# Find stars with phase-folded data
lc_tic_ids = {int(f.stem.split("_")[1]) for f in ACTIVE_FOLDED_1D_DIR.glob("tic_*.npy")}
img_tic_ids = {int(f.stem.split("_")[1]) for f in ACTIVE_IMAGES_DIR.glob("tic_*.npy")}

lc_df = tce_df[tce_df["tic_id"].isin(lc_tic_ids)].copy()
img_df = tce_df[tce_df["tic_id"].isin(img_tic_ids)].copy()

print(f"\nData: {len(lc_df)} TCEs with light curves "
      f"({(lc_df['label']==1).sum()} planets, {(lc_df['label']==0).sum()} FP)")

if len(lc_df) == 0:
    print("ERROR: No phase-folded data found. Run: python -m src.data.phase_fold")
    sys.exit(1)


# ============================================================
# TRAIN / VAL / TEST SPLITS (by star ID)
# ============================================================

from sklearn.model_selection import GroupShuffleSplit

gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
train_val_idx, test_idx = next(gss_test.split(lc_df, groups=lc_df["tic_id"]))

train_val_df = lc_df.iloc[train_val_idx]
# Val fraction relative to remaining data: 0.15 / (1 - 0.15) ≈ 0.176
gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION / (1 - TEST_FRACTION), random_state=SPLIT_SEED)
val_rel_idx_train, val_rel_idx_val = next(
    gss_val.split(train_val_df, groups=train_val_df["tic_id"])
)

train_df = lc_df.iloc[train_val_idx[val_rel_idx_train]].reset_index(drop=True)
val_df = lc_df.iloc[train_val_idx[val_rel_idx_val]].reset_index(drop=True)
test_df = lc_df.iloc[test_idx].reset_index(drop=True)

print(f"Splits: Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

# Verify no star leakage
assert not (set(train_df["tic_id"]) & set(test_df["tic_id"])), "Data leakage detected!"


# ============================================================
# LOAD ARRAYS
# ============================================================

def load_flux_arrays(df):
    """Load phase-folded 1D signals for a DataFrame of TCEs."""
    flux_list, labels, valid_idx = [], [], []
    for i, row in df.iterrows():
        path = ACTIVE_FOLDED_1D_DIR / f"tic_{int(row['tic_id'])}.npy"
        if path.exists():
            flux_list.append(np.load(path))
            labels.append(row["label"])
            valid_idx.append(i)
    return flux_list, np.array(labels, dtype=np.float32), valid_idx


def load_image_arrays(df):
    """Load phase-folded images for a DataFrame of TCEs."""
    img_list, labels, valid_idx = [], [], []
    for i, row in df.iterrows():
        path = ACTIVE_IMAGES_DIR / f"tic_{int(row['tic_id'])}.npy"
        if path.exists():
            img_list.append(np.load(path))
            labels.append(row["label"])
            valid_idx.append(i)
    return img_list, np.array(labels, dtype=np.float32), valid_idx


train_flux, train_labels_1d, _ = load_flux_arrays(train_df)
val_flux, val_labels_1d, _ = load_flux_arrays(val_df)
test_flux, test_labels_1d, _ = load_flux_arrays(test_df)

train_imgs, train_labels_2d, _ = load_image_arrays(train_df)
val_imgs, val_labels_2d, _ = load_image_arrays(val_df)
test_imgs, test_labels_2d, _ = load_image_arrays(test_df)

print(f"1D signals: Train {len(train_flux)} | Val {len(val_flux)} | Test {len(test_flux)}")
print(f"2D images:  Train {len(train_imgs)} | Val {len(val_imgs)} | Test {len(test_imgs)}")


# ============================================================
# TRAIN 1D CNN
# ============================================================

from src.models.cnn_1d import CNN1DModel
from src.evaluation.metrics import compute_all_metrics, print_metrics

cnn1d_model = None
cnn1d_trained = False

if len(train_flux) >= 2 and len(val_flux) >= 1:
    print(f"\n--- Training 1D CNN ---")
    cnn1d_model = CNN1DModel()

    total_params = sum(p.numel() for p in cnn1d_model.net.parameters())
    print(f"Parameters: {total_params:,}")

    cnn1d_model.fit(train_flux, train_labels_1d, val_flux, val_labels_1d)
    cnn1d_trained = True

    val_metrics_1d = compute_all_metrics(
        val_labels_1d,
        cnn1d_model.predict(val_flux),
        cnn1d_model.predict_proba(val_flux),
    )
    print_metrics(val_metrics_1d, "1D CNN (Validation)")
else:
    print(f"SKIPPED 1D CNN: insufficient data ({len(train_flux)} train, {len(val_flux)} val)")


# ============================================================
# TRAIN 2D CNN
# ============================================================

from src.models.cnn_2d import CNN2DModel

cnn2d_model = None
cnn2d_trained = False

if len(train_imgs) >= 2 and len(val_imgs) >= 1:
    print(f"\n--- Training 2D CNN ---")
    cnn2d_model = CNN2DModel()

    total_params = sum(p.numel() for p in cnn2d_model.net.parameters())
    print(f"Parameters: {total_params:,}")

    cnn2d_model.fit(train_imgs, train_labels_2d, val_imgs, val_labels_2d)
    cnn2d_trained = True

    val_metrics_2d = compute_all_metrics(
        val_labels_2d,
        cnn2d_model.predict(val_imgs),
        cnn2d_model.predict_proba(val_imgs),
    )
    print_metrics(val_metrics_2d, "2D CNN (Validation)")
else:
    print(f"SKIPPED 2D CNN: insufficient data ({len(train_imgs)} train, {len(val_imgs)} val)")


# ============================================================
# TEST SET EVALUATION + FOL RULES
# ============================================================

from src.evaluation.rules import apply_rules

all_results = []

if cnn1d_trained:
    test_preds_1d = cnn1d_model.predict(test_flux)
    test_probs_1d = cnn1d_model.predict_proba(test_flux)
    test_metrics_1d = compute_all_metrics(test_labels_1d, test_preds_1d, test_probs_1d)
    print_metrics(test_metrics_1d, "1D CNN (TEST)")

    all_results.append({
        "Model": "1D CNN",
        **{k: v for k, v in test_metrics_1d.items() if isinstance(v, (int, float))}
    })

    # FOL post-processing
    fol_1d = apply_rules(test_df, test_preds_1d, test_probs_1d)
    fol_metrics_1d = compute_all_metrics(
        test_labels_1d, fol_1d["filtered_predictions"], test_probs_1d
    )
    all_results.append({
        "Model": "1D CNN + FOL",
        **{k: v for k, v in fol_metrics_1d.items() if isinstance(v, (int, float))}
    })

    cnn1d_model.save()

if cnn2d_trained:
    test_preds_2d = cnn2d_model.predict(test_imgs)
    test_probs_2d = cnn2d_model.predict_proba(test_imgs)
    test_metrics_2d = compute_all_metrics(test_labels_2d, test_preds_2d, test_probs_2d)
    print_metrics(test_metrics_2d, "2D CNN (TEST)")

    all_results.append({
        "Model": "2D CNN",
        **{k: v for k, v in test_metrics_2d.items() if isinstance(v, (int, float))}
    })

    # FOL post-processing
    fol_2d = apply_rules(test_df, test_preds_2d, test_probs_2d)
    fol_metrics_2d = compute_all_metrics(
        test_labels_2d, fol_2d["filtered_predictions"], test_probs_2d
    )
    all_results.append({
        "Model": "2D CNN + FOL",
        **{k: v for k, v in fol_metrics_2d.items() if isinstance(v, (int, float))}
    })

    cnn2d_model.save()


# ============================================================
# SAVE RESULTS
# ============================================================

if all_results:
    results_df = pd.DataFrame(all_results)
    display_cols = ["Model", "precision", "recall", "f1",
                    "auc_roc", "average_precision", "false_positive_rate"]
    available_cols = [c for c in display_cols if c in results_df.columns]

    print("\nCNN TEST RESULTS:")
    print(results_df[available_cols].to_string(index=False, float_format="%.4f"))

    results_df.to_csv(RESULTS_DIR / "cnn_test_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'cnn_test_results.csv'}")

    # Merge with baseline results if available
    baseline_path = RESULTS_DIR / "final_test_results.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        combined = pd.concat([baseline_df, results_df], ignore_index=True)
        combined.to_csv(RESULTS_DIR / "all_models_comparison.csv", index=False)

        print("\nALL MODELS (test set):")
        print(combined[available_cols].to_string(index=False, float_format="%.4f"))

    print(f"\nModels saved to {MODELS_DIR}/")
else:
    print("No CNN models were trained — not enough light curve data.")
