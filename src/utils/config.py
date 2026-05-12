"""
Central configuration for the Exoplanet Detection project.

All hyperparameters, paths, and constants live here so nothing is
hardcoded in individual modules. Change settings in one place,
and every module picks them up.
"""

import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for _d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA SETTINGS
# ============================================================

# NASA Exoplanet Archive TAP API
NASA_API_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Labels: how NASA dispositions map to binary classes
# CP/KP = planet (1), FP/FA = not planet (0), PC = excluded (-1)
DISPOSITION_MAP = {
    "CP": 1,  "KP": 1,   # Confirmed / Known Planet
    "FP": 0,  "FA": 0,   # False Positive / False Alarm
    "PC": -1,             # Planet Candidate (ambiguous, exclude from training)
}

# ============================================================
# PREPROCESSING
# ============================================================

LIGHT_CURVE_LENGTH = 2001         # Points after resampling (raw 1D — kept for backward compat)
FLUX_NORMALIZATION = "median"     # "median" | "minmax" | "standard"

# Phase-folded settings
PHASE_FOLD_IMAGE_SIZE = (64, 64)    # Active size for training
PHASE_FOLD_1D_LENGTH = 256          # Active 1D length for training

# All sizes to generate when running phase_fold (for sweep testing)
PHASE_FOLD_IMAGE_SIZES = [(64, 64), (128, 128)]
PHASE_FOLD_1D_LENGTHS = [256, 512, 1024]

# ============================================================
# SPLITS
# ============================================================

# Split by star ID (not observation) to prevent data leakage
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

RANDOM_SEED = 7
SPLIT_SEED = 42

# ============================================================
# ENGINEERED_FEATURES (for classical baselines)
# ============================================================

# Features available directly from NASA TOI metadata (no light curves needed).
# These are used by Naive Bayes and Logistic Regression.
METADATA_FEATURES = [
    # Planet transit properties
    "transit_depth",        # ppm — size of the brightness dip
    "orbital_period",       # days — time for one orbit
    "transit_duration",     # hours — how long the dip lasts
    "planet_radius",        # Earth radii — estimated planet size
    "insolation_flux",      # Earth flux — energy received from star
    "equilibrium_temp",     # Kelvin — estimated planet temperature
    # Stellar host properties (key for FP detection: eclipsing binaries
    # have different stellar params than genuine planet hosts)
    "tess_magnitude",       # brightness of the star in TESS band
    "stellar_temp",         # Kelvin — star surface temperature
    "stellar_logg",         # log(g) — surface gravity (giants vs dwarfs)
    "stellar_radius",       # Solar radii — star size
    "stellar_distance",     # parsec — distance to the star
    # System-level (multi-planet systems are almost always real)
    "planet_num",           # number of planet candidates for this star
]

# Features that require light curves (computed from raw flux data).
# These become available after downloading + preprocessing light curves.
LIGHTCURVE_FEATURES = [
    "transit_snr",
    "ingress_duration",
    "depth_even_odd",
    "secondary_depth",
    "flux_std",
    "num_transits",
]

# Combined list (for backward compatibility with code that references this)
ENGINEERED_FEATURES = METADATA_FEATURES + LIGHTCURVE_FEATURES

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

# Naive Bayes
NB_N_BINS = 10
NB_ALPHA = 1.0

# Logistic Regression
LR_C = 1.0
LR_MAX_ITER = 1000

# 1D CNN
# Now operates on phase-folded 1D signal (512 points) instead of
# raw light curve (2001 points). The folding concentrates transit
# signal and averages out noise, making the 1D CNN's job much easier.
CNN_1D = {
    "n_filters": [32, 64, 128],
    "kernel_sizes": [15, 7, 5],
    "pool_size": 2,
    "dropout": 0.3,
    "fc_units": 128,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "eta_min": 1e-6,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15,
    "threshold": "auto",
    "val_metric": "pr_auc", # "loss" or "pr_auc" — metric for early stopping
}

# 2D CNN
# Upgraded training: class weighting, LR scheduler, gradient clipping
# to match the 1D CNN's training improvements. Larger images (128x128) v. smaller (64x64).
CNN_2D = {
    "n_filters": [32, 64, 128],
    "kernel_sizes": [3, 3, 3],
    "pool_size": 2,
    "dropout": 0.3,
    "fc_units": 128,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "eta_min": 1e-6,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15,
    "threshold": "auto",
    "val_metric": "pr_auc", # "loss" or "pr_auc" — metric for early stopping
}

# ============================================================
# EVALUATION
# ============================================================

EVAL_METRICS = [
    "accuracy",
    "auc_roc", 
    "average_precision",
    "f1",
    "precision", 
    "recall", 
]

# FOL post-processing rule thresholds
# NOTE: transit_depth from NASA TOI table (pl_trandep) is in PPM,
# not fractional. 1 ppm = 0.0001%. A typical hot Jupiter transit
# depth is ~10,000 ppm (1%); an Earth-sized planet is ~100 ppm.
RULES = {
    "min_transit_depth": 100,       # 100 ppm — below this is likely noise
    "max_transit_depth": 50_000,    # 50,000 ppm (5%) — above this is eclipsing binary
    "min_orbital_period": 0.5,      # days
    "max_orbital_period": 365.0,    # days
    "min_transit_snr": 3.0,
    "min_num_transits": 2,
}

# ============================================================
# DATA AUGMENTATION
# ============================================================

AUGMENT_1D = {
    "noise_sigma": 0.001,
    "max_time_shift": 50,
    "scale_range": (0.98, 1.02),
    "mask_fraction": 0.05,
    "p": 0.5,
}

AUGMENT_2D = {
    "noise_sigma": 0.02,
    "max_vshift": 2,
    "contrast_range": (0.9, 1.1),
    "erase_p_scale": 0.0, # disabled for now — test later
    "erase_max_frac": 8,
    "p": 0.5,
}

# Gaussian smoothing for sparse phase-folded images
GAUSSIAN_SMOOTH_SIGMA = 0.0

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch not installed; skip
