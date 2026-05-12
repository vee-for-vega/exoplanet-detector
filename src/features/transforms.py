"""
Data augmentation and transforms.

All probabilities and magnitudes are controlled from config.py
so you can tune augmentation without touching this code.
"""

import numpy as np

from src.utils.config import AUGMENT_1D, AUGMENT_2D


# ============================================================
# 1D Light Curve Augmentations
# ============================================================

def add_gaussian_noise(flux: np.ndarray, sigma: float) -> np.ndarray:
    """Simulate instrument noise (photon noise, readout noise)."""
    return flux + np.random.normal(0, sigma, size=flux.shape)


def random_time_shift(flux: np.ndarray, max_shift: int) -> np.ndarray:
    """Circular shift — transit position shouldn't matter."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(flux, shift)


def random_scale(flux: np.ndarray, scale_range: tuple) -> np.ndarray:
    """Simulate baseline flux calibration differences."""
    return flux * np.random.uniform(*scale_range)


def random_mask(flux: np.ndarray, mask_fraction: float) -> np.ndarray:
    """Simulate data gaps from downlinks or cosmic rays."""
    result = flux.copy()
    n_mask = int(len(flux) * mask_fraction)
    mask_idx = np.random.choice(len(flux), n_mask, replace=False)
    result[mask_idx] = np.median(flux)
    return result


def augment_light_curve(flux: np.ndarray) -> np.ndarray:
    """Apply random 1D augmentations controlled by AUGMENT_1D config."""
    cfg = AUGMENT_1D
    p = cfg["p"]
    result = flux.copy()
    if np.random.random() < p:
        result = add_gaussian_noise(result, cfg["noise_sigma"])
    if np.random.random() < p:
        result = random_time_shift(result, cfg["max_time_shift"])
    if np.random.random() < p:
        result = random_scale(result, cfg["scale_range"])
    if np.random.random() < p:
        result = random_mask(result, cfg["mask_fraction"])
    return result


# ============================================================
# 2D Phase-Folded Image Augmentations
# ============================================================

def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply random 2D augmentations controlled by AUGMENT_2D config.

    Physically valid transforms for phase-folded images:
    1. Gaussian noise — instrument noise varies between observations
    2. Vertical shift — baseline flux calibration differences
    3. Horizontal flip — transits are symmetric (ingress ≈ egress)
    4. Contrast scaling — different stars have different SNR levels
    5. Random erasing — simulates data gaps (controlled by erase_p_scale)
    """
    cfg = AUGMENT_2D
    p = cfg["p"]
    result = image.copy()

    # Gaussian noise
    if np.random.random() < p:
        noise = np.random.normal(0, cfg["noise_sigma"], size=image.shape).astype(np.float32)
        result = np.clip(result + noise, 0, 1)

    # Vertical shift
    if np.random.random() < p:
        shift = np.random.randint(-cfg["max_vshift"], cfg["max_vshift"] + 1)
        result = np.roll(result, shift, axis=0)

    # Horizontal flip (transit symmetry — physically valid)
    if np.random.random() < p:
        result = np.fliplr(result).copy()

    # Contrast scaling
    if np.random.random() < p:
        lo, hi = cfg["contrast_range"]
        factor = np.random.uniform(lo, hi)
        mean_val = result.mean()
        result = np.clip((result - mean_val) * factor + mean_val, 0, 1)

    # Random erasing (lower probability via erase_p_scale)
    if np.random.random() < p * cfg["erase_p_scale"]:
        h, w = result.shape
        max_dim = max(3, h // cfg["erase_max_frac"])
        eh = np.random.randint(2, max_dim)
        ew = np.random.randint(2, max_dim)
        y0 = np.random.randint(0, h - eh)
        x0 = np.random.randint(0, w - ew)
        result[y0:y0+eh, x0:x0+ew] = 0.0

    return result.astype(np.float32)
