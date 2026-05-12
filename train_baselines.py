"""
Train baseline models (Naive Bayes, Logistic Regression) on TOI metadata.

Uses 12 metadata features from the NASA Exoplanet Archive — no light
curves needed. Applies FOL physics rules as post-processing.

Run: python train_baselines.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import (
    RAW_DIR, PROCESSED_DIR, SPLITS_DIR, RESULTS_DIR,
    METADATA_FEATURES, RANDOM_SEED, set_seed
)

set_seed(RANDOM_SEED)


# ============================================================
# CLEAR PREVIOUS RESULTS
# ============================================================

existing_results = list(RESULTS_DIR.glob("*"))
if existing_results:
    for f in existing_results:
        if f.is_file():
            f.unlink()


# ============================================================
# LOAD DATA
# ============================================================

tce_path = RAW_DIR / "tce_table_clean.csv"
if not tce_path.exists():
    print(f"ERROR: {tce_path} not found. Run: python -m src.data.download")
    sys.exit(1)

df = pd.read_csv(tce_path)
print(f"\nLoaded {len(df)} TCEs "
      f"({(df['label']==1).sum()} planets, {(df['label']==0).sum()} FP)")


# ============================================================
# FEATURE SELECTION + MISSING DATA
# ============================================================

feature_cols = [c for c in METADATA_FEATURES if c in df.columns]
missing_features = [c for c in METADATA_FEATURES if c not in df.columns]

print(f"Features: {len(feature_cols)} available" +
      (f", {len(missing_features)} missing" if missing_features else ""))

# Impute missing values with median
for col in feature_cols:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        df[col] = df[col].fillna(df[col].median())


# ============================================================
# TRAIN / VAL / TEST SPLITS (by star ID)
# ============================================================

from src.data.preprocess import create_splits

train_idx, val_idx, test_idx = create_splits(df)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()
test_df = df.iloc[test_idx].copy()

print(f"Splits: Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

# Verify no star leakage
assert not (set(train_df["tic_id"]) & set(test_df["tic_id"])), "Data leakage detected!"


# ============================================================
# TRAIN NAIVE BAYES
# ============================================================

from src.models.naive_bayes import NaiveBayesModel
from src.evaluation.metrics import compute_all_metrics, print_metrics

X_train = train_df[feature_cols + ["label"]]
y_train = train_df["label"].values
X_val = val_df[feature_cols + ["label"]]
y_val = val_df["label"].values

nb_model = NaiveBayesModel(feature_columns=feature_cols)
nb_model.fit(X_train, y_train)

nb_preds = nb_model.predict(X_val)
nb_probs = nb_model.predict_proba(X_val)
nb_metrics = compute_all_metrics(y_val, nb_preds, nb_probs)
print_metrics(nb_metrics, "Naive Bayes (Validation)")


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

from src.models.logistic import LogisticModel

lr_model = LogisticModel(feature_columns=feature_cols)
lr_model.fit(X_train, y_train)

# Feature importance
importance = lr_model.get_feature_importance()
print("\nFeature coefficients (|weight|):")
for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {feat:>20}: {imp:.4f}")

lr_preds = lr_model.predict(X_val)
lr_probs = lr_model.predict_proba(X_val)
lr_metrics = compute_all_metrics(y_val, lr_preds, lr_probs)
print_metrics(lr_metrics, "Logistic Regression (Validation)")


# ============================================================
# FOL POST-PROCESSING RULES
# ============================================================

from src.evaluation.rules import apply_rules

nb_rule_result = apply_rules(val_df, nb_preds, nb_probs)
nb_filtered_metrics = compute_all_metrics(
    y_val, nb_rule_result["filtered_predictions"], nb_probs
)

lr_rule_result = apply_rules(val_df, lr_preds, lr_probs)
lr_filtered_metrics = compute_all_metrics(
    y_val, lr_rule_result["filtered_predictions"], lr_probs
)


# ============================================================
# VALIDATION COMPARISON
# ============================================================

comparison = pd.DataFrame([
    {"Model": "Naive Bayes",
     "Precision": nb_metrics["precision"], "Recall": nb_metrics["recall"],
     "F1": nb_metrics["f1"], "AUC-ROC": nb_metrics.get("auc_roc", 0),
     "PR-AUC": nb_metrics.get("average_precision", 0)},
    {"Model": "NB + FOL",
     "Precision": nb_filtered_metrics["precision"], "Recall": nb_filtered_metrics["recall"],
     "F1": nb_filtered_metrics["f1"], "AUC-ROC": nb_filtered_metrics.get("auc_roc", 0),
     "PR-AUC": nb_filtered_metrics.get("average_precision", 0)},
    {"Model": "Logistic Regression",
     "Precision": lr_metrics["precision"], "Recall": lr_metrics["recall"],
     "F1": lr_metrics["f1"], "AUC-ROC": lr_metrics.get("auc_roc", 0),
     "PR-AUC": lr_metrics.get("average_precision", 0)},
    {"Model": "LR + FOL",
     "Precision": lr_filtered_metrics["precision"], "Recall": lr_filtered_metrics["recall"],
     "F1": lr_filtered_metrics["f1"], "AUC-ROC": lr_filtered_metrics.get("auc_roc", 0),
     "PR-AUC": lr_filtered_metrics.get("average_precision", 0)},
])

print("\nValidation comparison:")
print(comparison.to_string(index=False, float_format="%.4f"))
comparison.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)


# ============================================================
# FINAL TEST SET EVALUATION
# ============================================================

X_test = test_df[feature_cols + ["label"]]
y_test = test_df["label"].values

nb_test_preds = nb_model.predict(X_test)
nb_test_probs = nb_model.predict_proba(X_test)
nb_test_metrics = compute_all_metrics(y_test, nb_test_preds, nb_test_probs)

lr_test_preds = lr_model.predict(X_test)
lr_test_probs = lr_model.predict_proba(X_test)
lr_test_metrics = compute_all_metrics(y_test, lr_test_preds, lr_test_probs)

nb_test_rules = apply_rules(test_df, nb_test_preds, nb_test_probs)
nb_test_fol = compute_all_metrics(y_test, nb_test_rules["filtered_predictions"], nb_test_probs)

lr_test_rules = apply_rules(test_df, lr_test_preds, lr_test_probs)
lr_test_fol = compute_all_metrics(y_test, lr_test_rules["filtered_predictions"], lr_test_probs)

final_results = pd.DataFrame([
    {"Model": "Naive Bayes", **{k: v for k, v in nb_test_metrics.items() if isinstance(v, float)}},
    {"Model": "NB + FOL", **{k: v for k, v in nb_test_fol.items() if isinstance(v, float)}},
    {"Model": "Logistic Regression", **{k: v for k, v in lr_test_metrics.items() if isinstance(v, float)}},
    {"Model": "LR + FOL", **{k: v for k, v in lr_test_fol.items() if isinstance(v, float)}},
])

display_cols = ["Model", "precision", "recall", "f1", "auc_roc", "average_precision"]
available_cols = [c for c in display_cols if c in final_results.columns]

print("\nTEST RESULTS:")
print(final_results[available_cols].to_string(index=False, float_format="%.4f"))

final_results.to_csv(RESULTS_DIR / "final_test_results.csv", index=False)
print(f"\nSaved: {RESULTS_DIR / 'final_test_results.csv'}")

# Save models
nb_model.save()
lr_model.save()
print(f"Models saved to models/")
