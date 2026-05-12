"""
First-Order Logic post-processing rules.

These rules act as a safety net on top of model predictions.
They encode physical constraints that any valid planet candidate
must satisfy, regardless of what the model says.

The model learns statistical patterns; the rules enforce
domain constraints (hybrid ML + logic approach).
"""

import numpy as np
import pandas as pd

from src.utils.config import RULES


def check_transit_depth(row: pd.Series) -> bool:
    """
    Valid_Depth(x) <- depth > min AND depth < max

    Too shallow = likely noise (below instrument sensitivity).
    Too deep = likely eclipsing binary star, not a planet.
    """
    depth = row.get("transit_depth", 0)
    return RULES["min_transit_depth"] <= depth <= RULES["max_transit_depth"]


def check_orbital_period(row: pd.Series) -> bool:
    """
    Valid_Period(x) <- period > min AND period < max

    Too short = physically implausible (planet inside the star).
    Too long = can't confirm periodicity with available data.
    """
    period = row.get("orbital_period", 0)
    return RULES["min_orbital_period"] <= period <= RULES["max_orbital_period"]


def check_transit_snr(row: pd.Series) -> bool:
    """
    Significant_Signal(x) <- SNR > threshold

    Below minimum SNR, can't distinguish signal from noise.
    """
    snr = row.get("transit_snr", 0)
    return snr >= RULES["min_transit_snr"]


def check_num_transits(row: pd.Series) -> bool:
    """
    Repeatable(x) <- num_transits >= min

    A single dip could be anything; need repeated events
    to confirm periodicity.
    """
    n = row.get("num_transits", 0)
    return n >= RULES["min_num_transits"]


# Master rule: all conditions must hold
# Each entry maps: (rule_name, rule_function, required_column)
# The required_column is checked at the DataFrame level BEFORE
# calling the rule. If the column doesn't exist, the rule is
# skipped entirely — no silent 0-default that auto-fails.
ALL_RULES = [
    ("Valid_Depth", check_transit_depth, "transit_depth"),
    ("Valid_Period", check_orbital_period, "orbital_period"),
    ("Significant_Signal", check_transit_snr, "transit_snr"),
    ("Repeatable", check_num_transits, "num_transits"),
]


def apply_rules(features_df: pd.DataFrame, predictions: np.ndarray,
                probabilities: np.ndarray = None) -> dict:
    """
    Apply all FOL rules as post-processing on model predictions.

    For each prediction the model says is a planet (pred=1),
    check all rules. If any rule fails, override to not-planet.

    Rules are SKIPPED when their required column is missing from
    the DataFrame entirely. This prevents silent failures where
    a missing column defaults to 0 and auto-fails every check
    (which is what previously caused FOL to override ALL positives
    when transit_snr and num_transits weren't in the metadata).

    A rule IS applied when the column exists but the value is NaN
    for a specific row — NaN means the data should have been there
    but wasn't, so conservatively fail that check.

    Returns:
        Dictionary with:
        - filtered_predictions: predictions after rule application
        - rule_violations: per-sample breakdown of which rules failed
        - n_overridden: how many positives were overridden
        - rules_applied: which rules were actually checked
        - rules_skipped: which rules were skipped (missing columns)
    """
    # Determine which rules can actually be applied
    available_cols = set(features_df.columns)
    active_rules = []
    skipped_rules = []

    for rule_name, rule_fn, required_col in ALL_RULES:
        if required_col in available_cols:
            active_rules.append((rule_name, rule_fn))
        else:
            skipped_rules.append((rule_name, required_col))

    if skipped_rules:
        skipped_str = ", ".join(
            f"{name} (needs '{col}')" for name, col in skipped_rules
        )
        print(f"    FOL rules skipped (columns not in data): {skipped_str}")

    filtered = predictions.copy()
    violations = []

    for idx in range(len(predictions)):
        if predictions[idx] == 1:  # Only check positive predictions
            row = features_df.iloc[idx]
            failed_rules = []

            for rule_name, rule_fn in active_rules:
                try:
                    if not rule_fn(row):
                        failed_rules.append(rule_name)
                except (KeyError, TypeError):
                    # Unexpected error in rule evaluation — skip
                    pass

            if failed_rules:
                filtered[idx] = 0  # Override to not-planet
                violations.append({
                    "index": idx,
                    "failed_rules": failed_rules,
                    "original_prob": probabilities[idx] if probabilities is not None else None,
                })

    n_overridden = int((predictions != filtered).sum())
    n_positive_before = int(predictions.sum())

    return {
        "filtered_predictions": filtered,
        "rule_violations": violations,
        "n_overridden": n_overridden,
        "n_positive_before": n_positive_before,
        "n_positive_after": int(filtered.sum()),
        "rules_applied": [name for name, _ in active_rules],
        "rules_skipped": [name for name, _ in skipped_rules],
    }
