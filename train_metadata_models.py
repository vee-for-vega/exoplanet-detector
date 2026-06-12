"""
Train metadata models (Naive Bayes, Logistic Regression, Gradient Boosting)
on either TCE corpus, with validation-tuned decision thresholds.

Datasets:
    kepler  32,673 labeled DR25 TCEs (1:11 imbalance) + Robovetter-style
            vetting diagnostics. From src/data/download_kepler.py.
    tess    ~5k labeled TOIs (near-balanced), metadata features only.
            From src/data/download.py.

Upgrades over train_baselines.py:
    - Gradient boosting joins the two linear learning-exercise baselines
    - Decision threshold is tuned for F1 on the validation set instead of
      using the default 0.5 (matters enormously under class imbalance)
    - FOL physics rules applied on top of the tuned predictions

The CNNs still require light curves; this script is the ceiling of what
metadata alone can do.

Run:
    python train_metadata_models.py --dataset kepler [--s3]
    python train_metadata_models.py --dataset tess   [--s3]
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

from src.utils.config import (
    RAW_DIR, RESULTS_DIR, METADATA_FEATURES,
    TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION,
    RANDOM_SEED, SPLIT_SEED, set_seed,
)
from src.models.naive_bayes import NaiveBayesModel
from src.models.logistic import LogisticModel
from src.models.gradient_boost import GradientBoostModel
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.rules import apply_rules

set_seed(RANDOM_SEED)

KEPLER_FEATURES = [
    # transit signal
    "orbital_period", "transit_depth", "transit_duration",
    "transit_snr", "num_transits", "mes", "impact", "max_single_event",
    # vetting diagnostics (Robovetter discriminators)
    "odd_even_stat", "centroid_offset_dic", "centroid_offset_kic",
    "boot_fap", "ghost_core_stat", "ghost_halo_stat",
    # inferred planet properties
    "planet_radius", "insolation_flux", "equilibrium_temp",
    # stellar host
    "stellar_temp", "stellar_logg", "stellar_radius",
    # system
    "planet_num",
]

DATASETS = {
    "kepler": {
        "file": "kepler_tce_table_clean.csv",
        "group_col": "kepid",
        "features": KEPLER_FEATURES,
        "download_hint": "python -m src.data.download_kepler",
    },
    "tess": {
        "file": "tce_table_clean.csv",
        "group_col": "tic_id",
        "features": METADATA_FEATURES,
        "download_hint": "python -m src.data.download",
    },
}


def load_data(ds: dict) -> pd.DataFrame:
    path = RAW_DIR / ds["file"]
    if not path.exists():
        print(f"ERROR: {path} not found. Run: {ds['download_hint']}")
        sys.exit(1)
    df = pd.read_csv(path)
    n_pos, n_neg = (df["label"] == 1).sum(), (df["label"] == 0).sum()
    print(f"\nLoaded {len(df)} TCEs ({n_pos} planets, {n_neg} FP, "
          f"imbalance 1:{n_neg/max(n_pos,1):.1f})")
    return df


def make_splits(df: pd.DataFrame, group_col: str):
    """70/15/15 split grouped by star ID so no star spans two sets."""
    groups = df[group_col].values
    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    rest_idx, test_idx = next(gss_test.split(df, groups=groups))

    rest = df.iloc[rest_idx]
    val_frac_of_rest = VAL_FRACTION / (TRAIN_FRACTION + VAL_FRACTION)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_rest, random_state=SPLIT_SEED)
    train_rel, val_rel = next(gss_val.split(rest, groups=rest[group_col].values))

    train_df = rest.iloc[train_rel].copy()
    val_df = rest.iloc[val_rel].copy()
    test_df = df.iloc[test_idx].copy()

    assert not (set(train_df[group_col]) & set(test_df[group_col])), "Star leakage detected!"
    assert not (set(train_df[group_col]) & set(val_df[group_col])), "Star leakage detected!"
    print(f"Splits: Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """F1-optimal decision threshold from the validation PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def metrics_row(name: str, threshold: float, m: dict) -> dict:
    return {"Model": name, "Threshold": threshold,
            "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
            "AUC-ROC": m.get("auc_roc", 0), "PR-AUC": m.get("average_precision", 0)}


def main():
    parser = argparse.ArgumentParser(description="Train metadata models with tuned thresholds.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="kepler")
    parser.add_argument("--s3", action="store_true", help="Push results CSV to the S3 bucket")
    args = parser.parse_args()
    ds = DATASETS[args.dataset]

    df = load_data(ds)
    features = [c for c in ds["features"] if c in df.columns]
    print(f"Features: {len(features)} of {len(ds['features'])} available")
    for col in features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    train_df, val_df, test_df = make_splits(df, ds["group_col"])
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    models = [
        NaiveBayesModel(feature_columns=features),
        LogisticModel(feature_columns=features),
        GradientBoostModel(feature_columns=features),
    ]

    rows = []
    for model in models:
        model.fit(train_df, y_train)
        threshold = tune_threshold(y_val, model.predict_proba(val_df))

        probs = model.predict_proba(test_df)
        preds = (probs >= threshold).astype(int)
        rows.append(metrics_row(model.name, threshold,
                                compute_all_metrics(y_test, preds, probs)))

        fol = apply_rules(test_df, preds, probs)
        rows.append(metrics_row(f"{model.name} + FOL", threshold,
                                compute_all_metrics(y_test, fol["filtered_predictions"], probs)))
        model.save()

    table = pd.DataFrame(rows)
    print(f"\nTEST RESULTS ({args.dataset}, thresholds tuned on validation):")
    print(table.to_string(index=False, float_format="%.4f"))

    out_path = RESULTS_DIR / f"{args.dataset}_metadata_results.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if args.s3:
        from src.data.s3_sync import push
        push(out_path, "results")


if __name__ == "__main__":
    main()
