"""
Hand-crafted feature engineering for classical ML baselines.

These features encode domain knowledge about what makes a transit
signal look like a real planet vs. noise. Used by Naive Bayes
and Logistic Regression.
"""

import numpy as np
import pandas as pd

from src.utils.config import ENGINEERED_FEATURES


def extract_features_from_tce(tce_row: pd.Series) -> dict:
    """
    Extract features from TCE metadata (no light curve needed).

    These come directly from NASA's TCE table — they're pre-computed
    by NASA's pipeline. Use them as-is for baselines.
    """
    return {
        "transit_depth": tce_row.get("transit_depth", np.nan),
        "transit_duration": tce_row.get("transit_duration", np.nan),
        "orbital_period": tce_row.get("orbital_period", np.nan),
    }


def extract_features_from_lightcurve(flux: np.ndarray,
                                     period: float = None) -> dict:
    """
    Extract features from a processed light curve array.

    These features require the actual light curve data, not just
    metadata. They capture signal quality and shape.
    """
    features = {}

    # Transit depth: how much does the brightness dip?
    # Deeper dips = larger planet (relative to star).
    # Very shallow dips might be noise. Very deep might be
    # an eclipsing binary star, not a planet.
    features["transit_depth"] = 1.0 - np.min(flux)

    # Transit SNR: is the dip real or just noise?
    # Compare the dip depth to the scatter in out-of-transit flux.
    # Higher SNR = more confident the dip is real.
    baseline_std = np.std(flux[flux > np.percentile(flux, 25)])
    depth = 1.0 - np.min(flux)
    features["transit_snr"] = depth / baseline_std if baseline_std > 0 else 0.0

    # Flux standard deviation: how noisy is this star overall?
    # Very noisy stars produce more false positives.
    features["flux_std"] = np.std(flux)

    # Transit duration (approximate): count consecutive points below threshold
    # Real transits have smooth, predictable durations.
    # Noise creates random short dips.
    threshold = 1.0 - 0.5 * depth
    below = flux < threshold
    if below.any():
        # Find longest consecutive run below threshold
        transitions = np.diff(below.astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            if ends[0] < starts[0]:
                ends = ends[1:]
            min_len = min(len(starts), len(ends))
            if min_len > 0:
                durations = ends[:min_len] - starts[:min_len]
                features["transit_duration"] = float(np.max(durations)) / len(flux)
            else:
                features["transit_duration"] = 0.0
        else:
            features["transit_duration"] = 0.0
    else:
        features["transit_duration"] = 0.0

    # Ingress duration: time from start of dip to minimum
    # Real planets have smooth, symmetric ingress/egress.
    min_idx = np.argmin(flux)
    if min_idx > 0:
        above_before = np.where(flux[:min_idx] > threshold)[0]
        if len(above_before) > 0:
            features["ingress_duration"] = float(min_idx - above_before[-1]) / len(flux)
        else:
            features["ingress_duration"] = float(min_idx) / len(flux)
    else:
        features["ingress_duration"] = 0.0

    # Even-odd transit depth difference
    # Real planets produce identical dips every orbit.
    # Eclipsing binaries often produce alternating deep/shallow dips.
    # A large even-odd difference suggests a binary, not a planet.
    half = len(flux) // 2
    if half > 0:
        even_depth = 1.0 - np.min(flux[:half])
        odd_depth = 1.0 - np.min(flux[half:])
        features["depth_even_odd"] = abs(even_depth - odd_depth)
    else:
        features["depth_even_odd"] = 0.0

    # Secondary eclipse depth
    # If there's a dip at phase 0.5 (opposite side of orbit),
    # it might be an eclipsing binary, not a planet transiting a star.
    mid = len(flux) // 2
    quarter = len(flux) // 4
    secondary_region = flux[mid - quarter:mid + quarter] if quarter > 0 else flux
    features["secondary_depth"] = 1.0 - np.min(secondary_region)

    # Number of transits (approximate via period)
    features["num_transits"] = 1  # Placeholder; needs period + time span
    if period and period > 0:
        total_time_span = len(flux)  # In resampled points
        features["num_transits"] = max(1, int(total_time_span / (period * 48)))

    features["orbital_period"] = period if period else 0.0

    return features


def build_feature_matrix(tce_df: pd.DataFrame,
                         processed_dir=None) -> pd.DataFrame:
    """
    Build a feature matrix for all TCEs.

    Combines metadata features (from TCE table) with light curve
    features (from processed .npy files). Falls back to metadata-only
    if a light curve file isn't available.

    Returns:
        DataFrame where each row is a TCE and columns are features.
    """
    from src.utils.config import PROCESSED_DIR
    if processed_dir is None:
        processed_dir = PROCESSED_DIR

    rows = []
    for _, tce in tce_df.iterrows():
        tic_id = tce["tic_id"]
        lc_path = processed_dir / f"tic_{int(tic_id)}.npy"

        if lc_path.exists():
            flux = np.load(lc_path)
            features = extract_features_from_lightcurve(
                flux, period=tce.get("orbital_period")
            )
        else:
            features = extract_features_from_tce(tce)

        features["tic_id"] = tic_id
        features["label"] = tce["label"]
        rows.append(features)

    feature_df = pd.DataFrame(rows)

    # Fill any remaining NaN with column medians (safe default)
    for col in ENGINEERED_FEATURES:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    return feature_df
