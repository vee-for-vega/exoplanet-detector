"""
Train baseline models (Naive Bayes, Logistic Regression) on the Kepler DR25
metadata corpus (32,673 labeled TCEs from src/data/download_kepler.py).

Same models and evaluation as train_baselines.py, adapted for Kepler:
  - kepler_tce_table_clean.csv instead of the TESS TOI table
  - groups splits by kepid instead of tic_id
  - Kepler feature set, including columns TESS lacks (MES, impact,
    num_transits, transit SNR straight from the DV fits)
  - 1:11 class imbalance (vs near-balanced TESS) -- PR-AUC is the metric
    that matters; accuracy is meaningless here

The CNNs are not runnable on Kepler yet: they need the phase-folded light
curves, which is the stream-and-discard preprocessing step.

Run: python train_baselines_kepler.py [--s3]
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import GroupShuffleSplit

from src.utils.config import (
    RAW_DIR, RESULTS_DIR,
    TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION,
    RANDOM_SEED, SPLIT_SEED, set_seed,
)
from src.models.naive_bayes import NaiveBayesModel
from src.models.logistic import LogisticModel
from src.evaluation.metrics import compute_all_metrics, print_metrics
from src.evaluation.rules import apply_rules

set_seed(RANDOM_SEED)

KEPLER_FEATURES = [
    # transit signal
    "orbital_period", "transit_depth", "transit_duration",
    "transit_snr", "num_transits", "mes", "impact",
    # inferred planet properties
    "planet_radius", "insolation_flux", "equilibrium_temp",
    # stellar host
    "stellar_temp", "stellar_logg", "stellar_radius",
    # system
    "planet_num",
]


def load_data() -> pd.DataFrame:
    path = RAW_DIR / "kepler_tce_table_clean.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run: python -m src.data.download_kepler")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"\nLoaded {len(df)} Kepler TCEs "
          f"({(df['label']==1).sum()} planets, {(df['label']==0).sum()} FP, "
          f"imbalance 1:{(df['label']==0).sum()/(df['label']==1).sum():.1f})")
    return df


def make_splits(df: pd.DataFrame):
    """70/15/15 split grouped by kepid so no star spans two sets."""
    groups = df["kepid"].values
    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SPLIT_SEED)
    rest_idx, test_idx = next(gss_test.split(df, groups=groups))

    rest = df.iloc[rest_idx]
    val_frac_of_rest = VAL_FRACTION / (TRAIN_FRACTION + VAL_FRACTION)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_rest, random_state=SPLIT_SEED)
    train_rel, val_rel = next(gss_val.split(rest, groups=rest["kepid"].values))

    train_df = rest.iloc[train_rel].copy()
    val_df = rest.iloc[val_rel].copy()
    test_df = df.iloc[test_idx].copy()

    assert not (set(train_df["kepid"]) & set(test_df["kepid"])), "Star leakage detected!"
    assert not (set(train_df["kepid"]) & set(val_df["kepid"])), "Star leakage detected!"
    print(f"Splits: Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


def metrics_row(name: str, m: dict) -> dict:
    return {"Model": name,
            "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
            "AUC-ROC": m.get("auc_roc", 0), "PR-AUC": m.get("average_precision", 0)}


def main():
    parser = argparse.ArgumentParser(description="Train Kepler metadata baselines.")
    parser.add_argument("--s3", action="store_true", help="Push results CSVs to the S3 bucket")
    args = parser.parse_args()

    df = load_data()

    features = [c for c in KEPLER_FEATURES if c in df.columns]
    print(f"Features: {len(features)} of {len(KEPLER_FEATURES)} available")
    for col in features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    train_df, val_df, test_df = make_splits(df)
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    nb = NaiveBayesModel(feature_columns=features).fit(train_df, y_train)
    lr = LogisticModel(feature_columns=features).fit(train_df, y_train)

    print("\nLogistic regression feature coefficients (|weight|):")
    for feat, imp in sorted(lr.get_feature_importance().items(), key=lambda x: -x[1]):
        print(f"  {feat:>20}: {imp:.4f}")

    rows_val, rows_test = [], []
    for model, name in ((nb, "Naive Bayes"), (lr, "Logistic Regression")):
        preds_v, probs_v = model.predict(val_df), model.predict_proba(val_df)
        m_v = compute_all_metrics(y_val, preds_v, probs_v)
        print_metrics(m_v, f"{name} (Validation)")
        rows_val.append(metrics_row(name, m_v))

        fol_v = apply_rules(val_df, preds_v, probs_v)
        rows_val.append(metrics_row(f"{name} + FOL",
                        compute_all_metrics(y_val, fol_v["filtered_predictions"], probs_v)))

        preds_t, probs_t = model.predict(test_df), model.predict_proba(test_df)
        rows_test.append(metrics_row(name, compute_all_metrics(y_test, preds_t, probs_t)))
        fol_t = apply_rules(test_df, preds_t, probs_t)
        rows_test.append(metrics_row(f"{name} + FOL",
                         compute_all_metrics(y_test, fol_t["filtered_predictions"], probs_t)))

    val_table = pd.DataFrame(rows_val)
    test_table = pd.DataFrame(rows_test)
    print("\nValidation comparison (Kepler DR25 metadata):")
    print(val_table.to_string(index=False, float_format="%.4f"))
    print("\nTEST RESULTS (Kepler DR25 metadata):")
    print(test_table.to_string(index=False, float_format="%.4f"))

    val_path = RESULTS_DIR / "kepler_baseline_comparison.csv"
    test_path = RESULTS_DIR / "kepler_final_test_results.csv"
    val_table.to_csv(val_path, index=False)
    test_table.to_csv(test_path, index=False)
    print(f"\nSaved: {val_path}")
    print(f"Saved: {test_path}")

    if args.s3:
        from src.data.s3_sync import push
        push(val_path, "results")
        push(test_path, "results")


if __name__ == "__main__":
    main()
