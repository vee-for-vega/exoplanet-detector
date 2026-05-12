"""
Evaluation harness.

The centerpiece of the project's evaluation infrastructure.
Runs every model against the same test set, computes identical
metrics, applies FOL rules, generates comparison tables and plots.

This is what makes the project portfolio-grade vs. homework-grade.
It ensures fair, reproducible comparisons and catches regressions
if you retrain a model.

Usage:
    python -m src.evaluation.eval_harness
"""

import json
import logging
import pandas as pd

from datetime import datetime

from src.evaluation.metrics import compute_all_metrics, print_metrics
from src.evaluation.rules import apply_rules
from src.utils.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, features_df: pd.DataFrame = None,
                   apply_fol_rules: bool = True) -> dict:
    """
    Evaluate a single model end-to-end.

    Steps:
    1. Get predictions and probabilities
    2. Compute metrics BEFORE rule application
    3. Apply FOL rules (if features available)
    4. Compute metrics AFTER rule application
    5. Return both for comparison

    This lets you see exactly how much the rules help/hurt.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Metrics before rules
    metrics_raw = compute_all_metrics(y_test, y_pred, y_prob)

    result = {
        "model_name": model.name,
        "metrics_raw": metrics_raw,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_true": y_test,
    }

    # Apply FOL rules if feature data is available
    if apply_fol_rules and features_df is not None:
        rule_result = apply_rules(features_df, y_pred, y_prob)
        metrics_filtered = compute_all_metrics(
            y_test, rule_result["filtered_predictions"], y_prob
        )
        result["metrics_filtered"] = metrics_filtered
        result["rule_summary"] = {
            "n_overridden": rule_result["n_overridden"],
            "n_positive_before": rule_result["n_positive_before"],
            "n_positive_after": rule_result["n_positive_after"],
        }
        result["y_pred_filtered"] = rule_result["filtered_predictions"]

    return result


def run_harness(models: list, X_test, y_test,
                features_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Run all models through evaluation and generate comparison.

    Args:
        models: List of model instances (must have predict/predict_proba)
        X_test: Test data (format depends on model type)
        y_test: Test labels
        features_df: Feature DataFrame for FOL rules (optional)

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    all_results = []
    detailed_results = {}

    for model in models:
        logger.info(f"Evaluating {model.name}...")

        result = evaluate_model(model, X_test, y_test, features_df)
        print_metrics(result["metrics_raw"], model.name)

        if "metrics_filtered" in result:
            print_metrics(result["metrics_filtered"],
                         f"{model.name} + FOL Rules")
            logger.info(
                f"  Rules overrode {result['rule_summary']['n_overridden']} "
                f"predictions ({result['rule_summary']['n_positive_before']} → "
                f"{result['rule_summary']['n_positive_after']} positives)"
            )

        # Collect for comparison table
        row = {"model": model.name}
        row.update({f"raw_{k}": v for k, v in result["metrics_raw"].items()
                    if isinstance(v, (int, float))})
        if "metrics_filtered" in result:
            row.update({f"fol_{k}": v for k, v in result["metrics_filtered"].items()
                        if isinstance(v, (int, float))})
        all_results.append(row)

        # Store for visualization
        detailed_results[model.name] = result

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_df.to_csv(RESULTS_DIR / f"comparison_{timestamp}.csv", index=False)

    # Save detailed results for visualization
    save_path = RESULTS_DIR / f"detailed_{timestamp}.json"
    serializable = {}
    for name, res in detailed_results.items():
        serializable[name] = {
            "metrics_raw": res["metrics_raw"],
            "metrics_filtered": res.get("metrics_filtered"),
            "rule_summary": res.get("rule_summary"),
        }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"Results saved to {RESULTS_DIR}")
    return comparison_df, detailed_results


def print_comparison(comparison_df: pd.DataFrame):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)

    # Select key metrics for display
    display_cols = ["model", "raw_precision", "raw_recall", "raw_f1", "raw_auc_roc"]
    available = [c for c in display_cols if c in comparison_df.columns]
    print(comparison_df[available].to_string(index=False, float_format="%.4f"))

    if "fol_precision" in comparison_df.columns:
        print("\n  With FOL Rules Applied:")
        fol_cols = ["model", "fol_precision", "fol_recall", "fol_f1"]
        fol_available = [c for c in fol_cols if c in comparison_df.columns]
        print(comparison_df[fol_available].to_string(index=False, float_format="%.4f"))

    print("=" * 70)
