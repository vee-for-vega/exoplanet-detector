"""
Stream-and-discard Kepler light-curve preprocessor.

Turns the 32,673-TCE Kepler DR25 corpus into CNN-ready tensors without ever
holding the raw archive on disk (~300 GB if downloaded naively; this machine
has ~38 GB free). Per star:

    1. Download all long-cadence quarters from MAST via lightkurve
    2. Stitch + clean (NaNs out; asymmetric outlier clip that keeps transits)
    3. For every TCE of that star, phase-fold at its period/epoch into
       a 64x64 image and a 256-bin 1D signal (same formats as the TESS
       pipeline, reusing src/data/phase_fold.py)
    4. Write the small tensors, delete the FITS cache, move on

Progress is checkpointed to data/processed/kepler/progress.csv, so the job
is resumable: rerunning skips completed stars. Expect multiple days of wall
clock for the full corpus; run it under caffeinate so the Mac doesn't sleep:

    nohup caffeinate -is python3 -m src.data.preprocess_kepler --s3 \
        >> data/processed/kepler/preprocess.log 2>&1 &

Outputs land in data/processed/kepler/ and, with --s3, sync to the bucket
every SYNC_EVERY stars.

Usage:
    python -m src.data.preprocess_kepler --limit 3      # smoke test
    python -m src.data.preprocess_kepler                # full corpus
    python -m src.data.preprocess_kepler --candidates   # also the 1,359 candidates
    python -m src.data.preprocess_kepler --retry-failed # re-attempt failures
"""

import argparse
import logging
import shutil
import time as time_mod

import numpy as np
import pandas as pd

from pathlib import Path

from src.utils.config import (
    PROCESSED_DIR, RAW_DIR,
    PHASE_FOLD_IMAGE_SIZE, PHASE_FOLD_1D_LENGTH,
)
from src.data.phase_fold import phase_fold, phase_fold_to_image, phase_fold_to_1d
from src.data.preprocess import normalize_flux

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = PROCESSED_DIR / "kepler"
IMAGES_DIR = OUT_DIR / f"images_{PHASE_FOLD_IMAGE_SIZE[0]}x{PHASE_FOLD_IMAGE_SIZE[1]}"
FOLDED_1D_DIR = OUT_DIR / f"folded_1d_{PHASE_FOLD_1D_LENGTH}"
PROGRESS_PATH = OUT_DIR / "progress.csv"
CACHE_DIR = RAW_DIR / "kepler_lc_cache"

SYNC_EVERY = 200          # stars between S3 syncs
MIN_POINTS = 500          # minimum valid cadences to keep a star


def load_tces(include_candidates: bool) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "kepler_tce_table_clean.csv")
    if include_candidates:
        cand = pd.read_csv(RAW_DIR / "kepler_candidates.csv")
        df = pd.concat([df, cand], ignore_index=True)
    needed = ["kepid", "planet_num", "orbital_period", "epoch_bkjd", "label"]
    df = df.dropna(subset=["orbital_period", "epoch_bkjd"])
    return df[needed + [c for c in df.columns if c not in needed]]


def load_progress() -> dict:
    """kepid -> status ('done' | 'failed')."""
    if not PROGRESS_PATH.exists():
        return {}
    p = pd.read_csv(PROGRESS_PATH)
    return dict(zip(p["kepid"].astype(int), p["status"]))


def append_progress(kepid: int, status: str, n_tces: int, note: str = ""):
    header = not PROGRESS_PATH.exists()
    row = pd.DataFrame([{"kepid": kepid, "status": status, "n_tces": n_tces, "note": note}])
    row.to_csv(PROGRESS_PATH, mode="a", header=header, index=False)


def download_star(kepid: int) -> np.ndarray:
    """
    Download and stitch all long-cadence Kepler quarters for one star.

    Returns (N, 2) array of [time_bkjd, normalized_flux], or None.
    The asymmetric outlier clip (5 sigma up, 20 sigma down) removes flares
    and instrumental jumps while keeping deep transits and eclipses, which
    are real downward signal in this corpus.
    """
    import lightkurve as lk

    search = lk.search_lightcurve(
        f"KIC {int(kepid)}", mission="Kepler", author="Kepler", cadence="long"
    )
    if len(search) == 0:
        logger.warning(f"KIC {kepid}: no light curves found")
        return None

    lc_collection = search.download_all(download_dir=str(CACHE_DIR))
    if lc_collection is None or len(lc_collection) == 0:
        return None

    # stitch() normalizes each quarter before joining, removing the
    # per-quarter flux offsets Kepler is known for
    lc = lc_collection.stitch() if len(lc_collection) > 1 else lc_collection[0].normalize()
    lc = lc.remove_nans().remove_outliers(sigma_upper=5.0, sigma_lower=20.0)

    time_arr = np.asarray(lc.time.value, dtype=np.float64)
    flux_arr = np.asarray(lc.flux.value, dtype=np.float64)
    valid = np.isfinite(time_arr) & np.isfinite(flux_arr)
    time_arr, flux_arr = time_arr[valid], flux_arr[valid]

    if len(time_arr) < MIN_POINTS:
        logger.warning(f"KIC {kepid}: only {len(time_arr)} valid points, skipping")
        return None
    return np.column_stack([time_arr, flux_arr])


def fold_tce(time_arr: np.ndarray, flux_arr: np.ndarray, tce: pd.Series) -> tuple:
    """Phase-fold one TCE into (image, signal_1d)."""
    flux_norm = normalize_flux(flux_arr)
    phase, folded_flux = phase_fold(
        time_arr, flux_norm,
        period=float(tce["orbital_period"]),
        epoch=float(tce["epoch_bkjd"]),
    )
    image = phase_fold_to_image(phase, folded_flux)
    signal = phase_fold_to_1d(phase, folded_flux)
    return image, signal


def process_star(kepid: int, tces: pd.DataFrame) -> int:
    """Download one star, fold all its TCEs, write tensors. Returns count."""
    data = download_star(kepid)
    if data is None:
        raise RuntimeError("no usable light curve")

    n = 0
    for _, tce in tces.iterrows():
        stem = f"kic_{int(kepid)}_{int(tce['planet_num'])}"
        image, signal = fold_tce(data[:, 0], data[:, 1], tce)
        np.save(IMAGES_DIR / f"{stem}.npy", image.astype(np.float32))
        np.save(FOLDED_1D_DIR / f"{stem}.npy", signal.astype(np.float32))
        n += 1
    return n


def sync_to_s3():
    from src.data.s3_sync import sync
    sync(OUT_DIR, "processed/kepler")


def main():
    parser = argparse.ArgumentParser(description="Stream-and-discard Kepler light-curve preprocessing.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N stars (smoke test)")
    parser.add_argument("--candidates", action="store_true", help="Include the unconfirmed-candidate set")
    parser.add_argument("--retry-failed", action="store_true", help="Re-attempt stars that previously failed")
    parser.add_argument("--s3", action="store_true", help=f"Sync outputs to S3 every {SYNC_EVERY} stars")
    args = parser.parse_args()

    for d in (IMAGES_DIR, FOLDED_1D_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_tces(args.candidates)
    progress = load_progress()
    skip = {k for k, s in progress.items() if s == "done" or (s == "failed" and not args.retry_failed)}

    star_groups = [(int(k), g) for k, g in df.groupby("kepid") if int(k) not in skip]
    if args.limit:
        star_groups = star_groups[: args.limit]

    logger.info(f"{df['kepid'].nunique()} stars total, {len(skip)} already done/failed, "
                f"{len(star_groups)} to process ({sum(len(g) for _, g in star_groups)} TCEs)")

    t0 = time_mod.time()
    n_ok = n_fail = 0
    for i, (kepid, tces) in enumerate(star_groups, 1):
        try:
            n = process_star(kepid, tces)
            append_progress(kepid, "done", n)
            n_ok += 1
        except Exception as e:
            append_progress(kepid, "failed", 0, note=str(e)[:200])
            logger.warning(f"KIC {kepid} failed: {e}")
            n_fail += 1
        finally:
            # the discard half of stream-and-discard
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR, ignore_errors=True)

        if i % 25 == 0 or i == len(star_groups):
            rate = i / max(time_mod.time() - t0, 1)
            remaining = (len(star_groups) - i) / max(rate, 1e-9)
            logger.info(f"[{i}/{len(star_groups)}] ok={n_ok} failed={n_fail} "
                        f"rate={rate*3600:.0f} stars/hr eta={remaining/3600:.1f} hr")
        if args.s3 and i % SYNC_EVERY == 0:
            sync_to_s3()

    if args.s3:
        sync_to_s3()
    logger.info(f"Finished: {n_ok} stars processed, {n_fail} failed. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
