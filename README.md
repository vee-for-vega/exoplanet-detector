<p align="center">
  <img src="https://science.nasa.gov/wp-content/uploads/2023/06/tessinspacerender16by9-jpg.webp" alt="TESS Spacecraft" width="600">
  <br>
  <em>NASA's TESS spacecraft surveying the sky for exoplanet transits. Credit: NASA/GSFC</em>
</p>

# Exoplanet Detection Pipeline

A machine learning pipeline that classifies **TESS (Transiting Exoplanet Survey Satellite)** threshold crossing events as confirmed planets or false positives, progressing from classical baselines to convolutional neural networks with physics-based post-processing.

Originally developed for **Foundations of AI** at **Colorado State University**. Now being actively developed as an independent research project with the goal of classifying unconfirmed planet candidates.

---

## The Problem

When TESS observes a star dimming periodically, NASA flags it as a **Threshold Crossing Event (TCE)** — a potential planet. But most TCEs are false positives: eclipsing binary stars, instrument noise, or stellar variability. Human vetting is slow. This pipeline automates that classification.

**Input:** Raw light curves (brightness over time) from NASA's TESS mission
**Output:** Binary prediction — planet or false positive — with a confidence score

## Architecture

The pipeline builds four models of increasing complexity. Each model is justified by the previous one's limitations — baselines validate that the features contain signal; CNNs learn what hand-picked features can't capture.

| Model | Input | Purpose |
|-------|-------|---------|
| **Naive Bayes** | 12 engineered features | Performance floor — simplest probabilistic model |
| **Logistic Regression** | 12 engineered features | Linear baseline with interpretable coefficients |
| **1D CNN** | Raw light curves (2,001 points) | Learns temporal patterns directly from flux data |
| **2D CNN** | Phase-folded images (64x64) | Learns spatial patterns from density maps |

### Phase Folding

The 2D CNN operates on **phase-folded images** — the light curve is folded at the orbital period so all transits stack on top of each other, boosting signal-to-noise ratio. This creates a 2D density map where the x-axis is orbital phase and the y-axis is normalized flux:

```
Raw light curve:             Phase-folded image:
                             ┌──────────────┐
 ╲  ╱   ╲  ╱   ╲  ╱          │              │
  ╲╱     ╲╱     ╲╱   ──►     │   ╲      ╱   │
                             │    ╲    ╱    │
transit transit transit      │     ╲──╱     │
                             └──────────────┘
                            (transits stacked)
```

### Post-Processing (FOL Rules)

All model predictions pass through **First-Order Logic rules** — physical constraints that any valid planet must satisfy. The model learns statistical patterns; the rules enforce domain knowledge.

| Rule | Constraint | Why |
|------|-----------|-----|
| Transit depth | 100 - 50,000 ppm | Too shallow = noise. Too deep = eclipsing binary. |
| Orbital period | 0.5 - 365 days | Too short = inside the star. Too long = can't confirm. |
| SNR | >= 3.0 | Below this, signal is indistinguishable from noise. |
| Transits | >= 2 | A single dip could be anything. Need repetition. |

---

## Current Results

Trained on **2,576 TCEs** (1,279 confirmed planets, 1,297 false positives). All models evaluated on the same held-out test set (n=386). Primary metric is **PR-AUC** — more informative than accuracy for detection tasks.

### Baselines

Naive Bayes and Logistic Regression serve as learning exercises to understand the fundamentals before moving to deep learning. They operate on hand-engineered metadata features only — no light curve data. They are not intended to be competitive with the CNNs; they exist to validate that the features contain signal and to establish a performance floor.

| Model | Precision | Recall | F1 | PR-AUC |
|-------|-----------|--------|----|--------|
| Naive Bayes | 0.593 | 0.959 | 0.733 | 0.692 |
| Logistic Regression | 0.641 | 0.720 | 0.678 | 0.703 |

### CNNs

Results tested across two random seeds to verify stability. The training seed controls weight initialization, dropout masks, and augmentation order. The split seed (42) is fixed — train/val/test sets are always identical.

| Model | Precision | Recall | F1 | PR-AUC | Seed |
|-------|-----------|--------|----|--------|------|
| 1D CNN | 0.681 | 0.780 | 0.727 | 0.772 | 7 |
| 2D CNN | 0.671 | 0.866 | 0.756 | 0.810 | 7 |
| 1D CNN | 0.673 | 0.774 | 0.720 | 0.723 | 42 |
| 2D CNN | 0.702 | 0.812 | 0.753 | 0.807 | 42 |

The 2D CNN consistently outperforms the 1D CNN on PR-AUC (~0.81 vs ~0.75) and F1 (~0.75 vs ~0.72). Precision hovers around 70% across both models and seeds — roughly 3 in 10 "planet" predictions are false positives. This is the current ceiling with ~1,800 training samples; more data (Kepler pre-training) and ensembling the two models are the planned next steps to push past it.

### Metadata Models v2 (tuned thresholds + gradient boosting)

`train_metadata_models.py` upgrades the metadata-only stack: a gradient-boosted
model joins the linear baselines, and every model's decision threshold is tuned
for F1 on the validation set instead of defaulting to 0.5. Evaluated on
held-out, star-grouped test sets:

| Dataset | Model | Precision | Recall | F1 | PR-AUC |
|---------|-------|-----------|--------|----|--------|
| TESS (n=396) | Gradient Boost | 0.849 | 0.899 | 0.873 | 0.905 |
| TESS (n=396) | Logistic Regression (tuned) | 0.672 | 0.870 | 0.758 | 0.785 |
| Kepler (n=4,763) | Gradient Boost | 0.918 | 0.958 | 0.938 | 0.983 |
| Kepler (n=4,763) | Logistic Regression (tuned) | 0.719 | 0.814 | 0.763 | 0.816 |

Notes:
- On TESS metadata alone, gradient boosting (PR-AUC 0.905) currently beats the
  2D CNN on light curves (0.81) — making it the strongest model in the project
  until the CNNs get the Kepler pre-training data.
- The Kepler run includes Robovetter-style vetting diagnostics (odd/even depth,
  centroid offsets, ghost stats, bootstrap FAP). Its 0.983 PR-AUC overstates
  real-world performance on new candidates: the DR25 labels partly descend from
  a vetting process that consumed these same diagnostics.
- Naive Bayes collapses on the Kepler diagnostic features (PR-AUC 0.11) — a
  textbook failure of the Gaussian/independence assumptions on skewed,
  correlated vetting statistics. Kept as a cautionary baseline.

### Kepler DR25 corpus

`src/data/download_kepler.py` builds the pre-training corpus: 32,673 labeled
TCEs (2,730 planets vs 29,943 false positives, 1:11) by joining the DR25 TCE
table against DR25 KOI dispositions, plus 1,359 unconfirmed candidates exported
separately as the future inference set.

### Key Design Decisions

- **GroupShuffleSplit** — splits by star ID to prevent data leakage between train/val/test
- **PR-AUC** as primary metric and early stopping criterion
- **Class-weighted BCE loss** — prevents the model from predicting all-negative
- **Cosine annealing LR** with gradient clipping for stable training
- **Threshold tuning** on validation set — optimal classification threshold is rarely 0.5
- **Physically-motivated augmentations only** — Gaussian noise, time shift, flux scaling, horizontal flip (transit symmetry)

---

## Roadmap

- [x] Naive Bayes + Logistic Regression baselines (learning exercises)
- [x] 1D CNN on raw light curves
- [x] 2D CNN on phase-folded images (64x64)
- [x] FOL post-processing rules
- [x] Phase fold size sweep (64x64 vs 128x128)
- [x] Configurable phase fold generation via CLI
- [ ] **Ensemble model** — combine 1D + 2D CNN predictions
- [ ] **Pre-train on Kepler data** — 34,000 labeled TCEs vs current 2,576 from TESS
- [ ] **Classify unconfirmed candidates** — run the pipeline on TCEs with "Planet Candidate" disposition that NASA hasn't confirmed yet
- [ ] Visualization dashboard for predictions and learned filters

---

## Project Structure

```
├── train_cnns.py              # Main CNN training script
├── train_baselines.py         # Naive Bayes + Logistic Regression
├── run_tests.py               # Integration tests
├── src/
│   ├── data/
│   │   ├── download.py        # NASA Exoplanet Archive API + MAST light curves
│   │   ├── preprocess.py      # Light curve cleaning and normalization
│   │   └── phase_fold.py      # Phase folding (1D signals + 2D images)
│   ├── features/
│   │   ├── engineered.py      # Hand-crafted features for baselines
│   │   └── transforms.py      # Data augmentation (1D + 2D)
│   ├── models/
│   │   ├── naive_bayes.py     # Baseline 1
│   │   ├── logistic.py        # Baseline 2
│   │   ├── cnn_1d.py          # 1D CNN on raw light curves
│   │   └── cnn_2d.py          # 2D CNN on phase-folded images
│   ├── evaluation/
│   │   ├── metrics.py         # Precision, recall, F1, PR-AUC
│   │   └── rules.py           # FOL post-processing rules
│   └── utils/
│       └── config.py          # Central configuration (all hyperparameters)
├── data/                      # Downloaded and processed data (not tracked)
├── models/                    # Saved model files (not tracked)
└── results/                   # Metrics CSVs and comparisons
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download TESS TCE metadata from NASA
python -m src.data.download

# Download the Kepler DR25 pre-training corpus (32,673 labeled TCEs +
# 1,359 unconfirmed candidates). Add --s3 to push the CSVs to the S3 bucket.
python -m src.data.download_kepler

# Train baselines on metadata features
python train_baselines.py

# Download light curves from MAST
python -m src.data.download --lightcurves

# Preprocess and phase-fold
python -m src.data.preprocess
python -m src.data.phase_fold

# Train CNNs
python train_cnns.py
```

## Data Sources

- **TCE metadata:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) (TAP API) — TESS TOI table and Kepler Q1-Q17 DR25 TCE + KOI tables
- **Light curves:** [MAST Archive](https://mast.stsci.edu/) via [Lightkurve](https://docs.lightkurve.org/); also mirrored in AWS at `s3://stpubdata` (us-east-1, requester-pays)
- **Mission:** [TESS (Transiting Exoplanet Survey Satellite)](https://tess.mit.edu/) and [Kepler](https://science.nasa.gov/mission/kepler/)

## Cloud Storage

Pipeline artifacts (metadata CSVs, processed tensors, models, results) live in
a private S3 bucket defined in `terraform/`. Raw light curves are never stored
there — preprocessing streams them and keeps only the small tensors.

```bash
cd terraform && terraform init && terraform apply
export EXOPLANET_S3_BUCKET=$(terraform output -raw bucket_name)
python -m src.data.s3_sync push data/raw/kepler_tce_table_clean.csv metadata/
```

## Tech Stack

Python, PyTorch, scikit-learn, Lightkurve, Astropy, NumPy, Pandas

## License

MIT

---

*Built with PyTorch, scikit-learn, and NASA open data.*
