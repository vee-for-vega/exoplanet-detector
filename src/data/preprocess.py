"""
Preprocessing module.

Takes raw light curves and TCE metadata, then:
1. Normalizes flux values
2. Resamples light curves to uniform length
3. Creates train/val/test splits by star ID (prevents data leakage)
4. Saves processed data to disk

Key design decision: split by star ID, not by observation.
If Star A has 3 candidate signals, ALL go to the same split.
Otherwise the model could memorize star-specific noise patterns
and leak information between train and test.
"""

import logging
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from src.utils.config import (
    FLUX_NORMALIZATION,
    LIGHT_CURVE_LENGTH, 
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR, 
    SPLITS_DIR,
    TEST_FRACTION,
    TRAIN_FRACTION, 
    VAL_FRACTION, 
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_flux(flux: np.ndarray, method: str = FLUX_NORMALIZATION) -> np.ndarray:
    """
    Normalize flux values for a single light curve.

    Why normalize? Raw flux values differ between stars based on
    brightness, distance, and instrument settings. Normalization
    makes transit depths comparable across different stars.

    Methods:
    - "median": Divide by median flux. Transit dips become fractional.
      Most common in exoplanet literature. Robust to outliers.
    - "minmax": Scale to [0, 1]. Simple but sensitive to outliers.
    - "standard": Zero mean, unit variance. Good for neural nets but
      loses the physical meaning of flux values.
    """
    if method == "median":
        med = np.median(flux)
        if med == 0:
            return flux
        return flux / med
    elif method == "minmax":
        fmin, fmax = flux.min(), flux.max()
        if fmax == fmin:
            return np.zeros_like(flux)
        return (flux - fmin) / (fmax - fmin)
    elif method == "standard":
        std = flux.std()
        if std == 0:
            return flux - flux.mean()
        return (flux - flux.mean()) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resample_light_curve(time: np.ndarray, flux: np.ndarray,
                         target_length: int = LIGHT_CURVE_LENGTH) -> np.ndarray:
    """
    Resample a light curve to a fixed number of points.

    Why? Light curves have different lengths depending on how long
    the star was observed. Neural networks need fixed-size inputs.
    Linearly interpolate to a uniform grid.

    Trade-off: Interpolation smooths out some fine structure, but
    ensures all inputs are the same shape. For transit detection,
    the broad dip shape matters more than individual point noise.
    """
    target_time = np.linspace(time.min(), time.max(), target_length)
    resampled_flux = np.interp(target_time, time, flux)
    return resampled_flux


def process_light_curve(raw_path: Path) -> np.ndarray:
    """
    Full preprocessing pipeline for a single light curve file.

    Steps: load → normalize → resample → return fixed-length array.
    """
    data = np.load(raw_path)
    time, flux = data[:, 0], data[:, 1]

    # Remove any remaining NaN/Inf values
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]

    if len(time) < 50:
        # Too few points to be useful
        return None

    flux = normalize_flux(flux)
    flux = resample_light_curve(time, flux)
    return flux


def process_all_light_curves(tce_df: pd.DataFrame):
    """
    Process all downloaded light curves and save as numpy arrays.

    Creates one .npy file per light curve in data/processed/.
    Also creates a manifest CSV mapping TIC IDs to processed file paths.
    """
    raw_lc_dir = RAW_DIR / "lightcurves"
    manifest = []

    tic_ids = tce_df["tic_id"].unique()
    logger.info(f"Processing {len(tic_ids)} light curves...")

    for tic_id in tic_ids:
        raw_path = raw_lc_dir / f"tic_{int(tic_id)}.npy"
        if not raw_path.exists():
            continue

        flux = process_light_curve(raw_path)
        if flux is None:
            continue

        save_path = PROCESSED_DIR / f"tic_{int(tic_id)}.npy"
        np.save(save_path, flux)
        manifest.append({"tic_id": tic_id, "processed_path": str(save_path)})

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(PROCESSED_DIR / "manifest.csv", index=False)
    logger.info(f"Processed {len(manifest_df)} light curves successfully.")
    return manifest_df


def create_splits(tce_df: pd.DataFrame):
    """
    Create train/val/test splits, grouping by star ID.

    Why group by star? A single star can have multiple TCEs
    (multiple candidate signals). If one goes to train and another
    to test, the model can learn star-specific patterns — that's
    data leakage. Grouping ensures all TCEs from the same star
    stay in the same split.

    Saves split definitions (just the indices) so splits are
    reproducible without re-running this function.
    """
    logger.info("Creating train/val/test splits...")

    # First split: separate test set
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )
    train_val_idx, test_idx = next(gss_test.split(tce_df, groups=tce_df["tic_id"]))

    # Second split: separate validation from training
    train_val_df = tce_df.iloc[train_val_idx]
    val_relative_size = VAL_FRACTION / (TRAIN_FRACTION + VAL_FRACTION)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_relative_size, random_state=RANDOM_SEED
    )
    train_idx_rel, val_idx_rel = next(
        gss_val.split(train_val_df, groups=train_val_df["tic_id"])
    )
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    # Save split indices
    np.save(SPLITS_DIR / "train_idx.npy", train_idx)
    np.save(SPLITS_DIR / "val_idx.npy", val_idx)
    np.save(SPLITS_DIR / "test_idx.npy", test_idx)

    # Log split statistics
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        split_df = tce_df.iloc[idx]
        n_pos = (split_df["label"] == 1).sum()
        n_neg = (split_df["label"] == 0).sum()
        n_stars = split_df["tic_id"].nunique()
        logger.info(
            f"{name}: {len(idx)} samples ({n_pos} planets, {n_neg} FP) "
            f"from {n_stars} unique stars"
        )

    return train_idx, val_idx, test_idx


def load_splits():
    """Load previously saved split indices."""
    return (
        np.load(SPLITS_DIR / "train_idx.npy"),
        np.load(SPLITS_DIR / "val_idx.npy"),
        np.load(SPLITS_DIR / "test_idx.npy"),
    )


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    tce_df = pd.read_csv(RAW_DIR / "tce_table_clean.csv")
    manifest = process_all_light_curves(tce_df)

    # Report how many TCEs have light curve data vs total
    n_with_lc = len(manifest) if manifest is not None else 0
    logger.info(
        f"Light curve coverage: {n_with_lc} of {len(tce_df)} TCEs "
        f"({n_with_lc/len(tce_df)*100:.1f}%) have processed light curves."
    )

    # Create splits over the FULL TCE table (used by train_baselines.py)
    logger.info("Creating splits over full TCE table (for baseline models)...")
    create_splits(tce_df)

    # Also show what the CNN splits will look like (light curve subset only)
    if n_with_lc > 0:
        lc_tic_ids = set(manifest["tic_id"].values)
        lc_df = tce_df[tce_df["tic_id"].isin(lc_tic_ids)]
        logger.info(f"Light curve subset: {len(lc_df)} TCEs from {len(lc_tic_ids)} stars")
        n_pos = (lc_df["label"] == 1).sum()
        n_neg = (lc_df["label"] == 0).sum()
        logger.info(f"  Planets: {n_pos} | False Positives: {n_neg}")
        # Approximate 70/15/15 split
        n_train = int(len(lc_df) * 0.70)
        n_val = int(len(lc_df) * 0.15)
        n_test = len(lc_df) - n_train - n_val
        logger.info(
            f"  CNN splits (approx): Train ~{n_train} | Val ~{n_val} | Test ~{n_test}"
        )
