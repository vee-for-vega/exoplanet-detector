"""
Phase folding module.

Converts 1D light curves into two formats for CNN classification:
1. 2D images (64x64 default) — density maps for the 2D CNN
2. 1D folded signals (256 bins default) — binned median flux for the 1D CNN

How phase folding works:
1. Take the raw time-series (brightness over time)
2. "Fold" it at the known orbital period — this stacks all transits
   on top of each other, so phase 0.0 = mid-transit
3. For 2D: bin into a 2D histogram (phase vs. flux)
4. For 1D: bin into phase bins and take the median flux per bin

This is the bridge between raw time-series data and learnable signals.
"""

import logging
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.utils.config import (
    RAW_DIR,
    DATA_DIR,
    PHASE_FOLD_IMAGE_SIZE,
    PHASE_FOLD_IMAGE_SIZES,
    PHASE_FOLD_1D_LENGTH,
    PHASE_FOLD_1D_LENGTHS,
    GAUSSIAN_SMOOTH_SIGMA,
)


def get_images_dir(size=None):
    """Return directory for phase-folded images of given size."""
    if size is None:
        size = PHASE_FOLD_IMAGE_SIZE
    return DATA_DIR / f"images_{size[0]}x{size[1]}"


def get_folded_1d_dir(length=None):
    """Return directory for 1D folded signals of given length."""
    if length is None:
        length = PHASE_FOLD_1D_LENGTH
    return DATA_DIR / f"folded_1d_{length}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def phase_fold(time: np.ndarray, flux: np.ndarray,
               period: float, epoch: float = None) -> tuple:
    """
    Phase-fold a light curve at a given orbital period.

    Args:
        time: Time array (days)
        flux: Flux array (normalized)
        period: Orbital period (days)
        epoch: Time of first transit (if known). If None, estimated
               as the time of minimum flux.

    Returns:
        (phase, flux) where phase is in range [-0.5, 0.5]
        with 0.0 centered on mid-transit.
    """
    if epoch is None:
        # Estimate epoch as time of deepest dip
        epoch = time[np.argmin(flux)]

    # Compute phase: how far through each orbit each measurement is
    phase = ((time - epoch) / period) % 1.0

    # Shift so transit is centered at phase 0
    phase[phase > 0.5] -= 1.0

    # Sort by phase for cleaner plotting/binning
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def phase_fold_to_image(phase: np.ndarray, flux: np.ndarray,
                        image_size: tuple = PHASE_FOLD_IMAGE_SIZE) -> np.ndarray:
    """
    Convert phase-folded data to a 2D image via binned histogram.

    The image is essentially a density map:
    - x-axis = orbital phase (position in orbit)
    - y-axis = flux level (brightness)
    - pixel intensity = number of data points in that bin

    A real transit creates a dense cluster of points in a U-shape
    near phase 0. Noise creates uniform scatter.

    Trade-off: Higher resolution images capture finer detail but
    are sparser (fewer points per bin). 64x64 is a good balance
    for TESS data with typical ~1000-10000 points per light curve.
    """
    h, w = image_size

    # Define bin edges
    phase_edges = np.linspace(-0.5, 0.5, w + 1)

    # For flux, use percentiles to handle outliers robustly
    flux_lo = np.percentile(flux, 0.5)
    flux_hi = np.percentile(flux, 99.5)
    flux_edges = np.linspace(flux_lo, flux_hi, h + 1)

    # Create 2D histogram
    image, _, _ = np.histogram2d(
        flux, phase,
        bins=[flux_edges, phase_edges]
    )

    # Gaussian smooth to spread signal from sparse pixels
    if GAUSSIAN_SMOOTH_SIGMA > 0:
        image = gaussian_filter(image, sigma=GAUSSIAN_SMOOTH_SIGMA)

    # Normalize to [0, 1]
    if image.max() > 0:
        image = image / image.max()

    # Flip vertically so transit dips appear at bottom (intuitive)
    image = np.flipud(image)

    return image.astype(np.float32)


def phase_fold_to_1d(phase: np.ndarray, flux: np.ndarray,
                     n_bins: int = PHASE_FOLD_1D_LENGTH) -> np.ndarray:
    """
    Convert phase-folded data to a 1D binned signal.

    Instead of a 2D density map, bin by phase and take the
    MEDIAN flux in each bin. This produces a clean 1D transit
    profile that the 1D CNN can learn from.

    Why median instead of mean?
    Outliers (cosmic rays, instrument glitches) pull the mean
    but don't affect the median. The transit dip is preserved
    while noise spikes are suppressed.

    The output is a 1D array of shape (n_bins,) where:
    - x-index = phase bin (0 = phase -0.5, n_bins-1 = phase +0.5)
    - value = median normalized flux in that bin

    A real planet shows a smooth dip near the center (phase 0).
    A false positive shows flat or irregular patterns.
    """
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    binned_flux = np.ones(n_bins, dtype=np.float32)  # default to 1.0 (no dip)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            binned_flux[i] = np.median(flux[mask])

    # Normalize to zero-mean, unit-variance for the CNN
    std = binned_flux.std()
    if std > 0:
        binned_flux = (binned_flux - binned_flux.mean()) / std

    return binned_flux


def create_phase_folded_images(tce_df: pd.DataFrame,
                               image_sizes=None, fold_lengths=None):
    """
    Create phase-folded data for all TCEs with downloaded light curves.

    Args:
        tce_df: DataFrame with tic_id and orbital_period columns.
        image_sizes: List of (h, w) tuples for 2D images. Defaults to active size only.
        fold_lengths: List of ints for 1D signal lengths. Defaults to active length only.
    """
    if image_sizes is None:
        image_sizes = [PHASE_FOLD_IMAGE_SIZE]
    if fold_lengths is None:
        fold_lengths = [PHASE_FOLD_1D_LENGTH]

    raw_lc_dir = RAW_DIR / "lightcurves"
    for img_size in image_sizes:
        get_images_dir(img_size).mkdir(parents=True, exist_ok=True)
    for fold_len in fold_lengths:
        get_folded_1d_dir(fold_len).mkdir(parents=True, exist_ok=True)

    success_2d, success_1d, fail = 0, 0, 0
    for _, row in tqdm(tce_df.iterrows(), total=len(tce_df),
                       desc="Creating phase-folded data"):
        tic_id = row["tic_id"]
        period = row["orbital_period"]

        raw_path = raw_lc_dir / f"tic_{int(tic_id)}.npy"
        if not raw_path.exists():
            fail += 1
            continue

        try:
            data = np.load(raw_path)
            time, flux = data[:, 0], data[:, 1]

            mask = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[mask], flux[mask]

            if len(time) < 50 or period <= 0:
                fail += 1
                continue

            med = np.median(flux)
            if med > 0:
                flux = flux / med

            phase, folded_flux = phase_fold(time, flux, period)

            for img_size in image_sizes:
                image = phase_fold_to_image(phase, folded_flux, image_size=img_size)
                np.save(get_images_dir(img_size) / f"tic_{int(tic_id)}.npy", image)
            success_2d += 1

            for fold_len in fold_lengths:
                signal_1d = phase_fold_to_1d(phase, folded_flux, n_bins=fold_len)
                np.save(get_folded_1d_dir(fold_len) / f"tic_{int(tic_id)}.npy", signal_1d)
            success_1d += 1

        except Exception as e:
            logger.warning(f"Failed for TIC {tic_id}: {e}")
            fail += 1

    sizes_str = ", ".join(f"{s[0]}x{s[1]}" for s in image_sizes)
    lengths_str = ", ".join(str(l) for l in fold_lengths)
    logger.info(
        f"Phase-folded: {success_2d} images ({sizes_str}) + "
        f"{success_1d} signals (1D: {lengths_str}) created, {fail} failed."
    )


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate phase-folded data for CNN training.")
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="2D image size (e.g. 64 for 64x64). Default: active size from config."
    )
    parser.add_argument(
        "--fold-length", type=int, default=None,
        help="1D signal length (e.g. 256). Default: active length from config."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate ALL sizes defined in config (for sweep testing)."
    )
    args = parser.parse_args()

    if args.all:
        image_sizes = PHASE_FOLD_IMAGE_SIZES
        fold_lengths = PHASE_FOLD_1D_LENGTHS
    else:
        image_sizes = [(args.image_size, args.image_size)] if args.image_size else None
        fold_lengths = [args.fold_length] if args.fold_length else None

    tce_df = pd.read_csv(RAW_DIR / "tce_table_clean.csv")
    create_phase_folded_images(tce_df, image_sizes=image_sizes, fold_lengths=fold_lengths)
    