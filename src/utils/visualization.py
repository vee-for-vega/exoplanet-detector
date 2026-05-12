"""
Visualization utilities.

Generates all the plots you'd want in a portfolio:
ROC curves, confusion matrices, light curve samples, and model comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def plot_light_curve(time: np.ndarray, flux: np.ndarray, title: str = "",
                     save_path: Path = None):
    """Plot a single light curve (flux over time)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(time, flux, s=1, alpha=0.5, color="steelblue")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(title or "Light Curve")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_phase_folded(phase: np.ndarray, flux: np.ndarray, title: str = "",
                      save_path: Path = None):
    """Plot a phase-folded light curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(phase, flux, s=1, alpha=0.4, color="steelblue")
    ax.set_xlabel("Orbital Phase")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(title or "Phase-Folded Light Curve")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "", save_path: Path = None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Planet", "Planet"],
                yticklabels=["Not Planet", "Planet"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix{f' - {model_name}' if model_name else ''}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str = "", save_path: Path = None):
    """Plot ROC curve for a single model."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_comparison(results: dict, save_path: Path = None):
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        results: dict of {model_name: {"y_true": array, "y_prob": array}}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]

    for i, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pr_comparison(results: dict, save_path: Path = None):
    """
    Plot Precision-Recall curves for multiple models.
    Better than ROC for imbalanced datasets like ours.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]

    for i, (name, data) in enumerate(results.items()):
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        ax.plot(recall, precision, color=colors[i % len(colors)], lw=2, label=name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(feature_names: list, importances: np.ndarray,
                            model_name: str = "", save_path: Path = None):
    """Plot horizontal bar chart of feature importances."""
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="steelblue")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance{f' - {model_name}' if model_name else ''}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
