"""
Evaluation metrics module.

Computes all metrics needed for model comparison.
Every model gets evaluated on the exact same metrics
via the same functions — no inconsistencies.

For imbalanced datasets, accuracy is misleading.
Focus on precision, recall, and PR-AUC instead.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score, 
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: np.ndarray = None) -> dict:
    """
    Compute all evaluation metrics for a single model.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probability of class 1 (for AUC metrics)

    Returns:
        Dictionary of metric_name -> value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Can fail if only one class present in y_true
            metrics["auc_roc"] = 0.0

        try:
            # PR-AUC: better than ROC-AUC for imbalanced data
            # because it focuses on the minority class performance
            metrics["average_precision"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["average_precision"] = 0.0

    # Confusion matrix components for detailed analysis
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    # False positive rate (important for telescope time allocation)
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return metrics


def print_metrics(metrics: dict, model_name: str = ""):
    """Pretty-print metrics for a model."""
    print(f"\n{'=' * 50}")
    print(f"  {model_name or 'Model'} Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1 Score:           {metrics['f1']:.4f}")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
    if "average_precision" in metrics:
        print(f"  PR-AUC:             {metrics['average_precision']:.4f}")
    print(f"  TP: {metrics['true_positives']}  FP: {metrics['false_positives']}  "
          f"FN: {metrics['false_negatives']}  TN: {metrics['true_negatives']}")
    print(f"{'=' * 50}")
