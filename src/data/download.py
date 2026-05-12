"""
Data download module.

Fetches two things from NASA:
1. TCE (Threshold Crossing Event) table — metadata + labels for each candidate
2. Light curves — the actual brightness-over-time data for each star

The TCE table comes from NASA's Exoplanet Archive API.
Light curves come from MAST via the lightkurve library.

Usage:
    python -m src.data.download               # Downloads TCE table only (fast)
    python -m src.data.download --lightcurves # Also downloads light curves (slow)
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd

from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.utils.config import (
    DISPOSITION_MAP, 
    NASA_API_URL,
    RANDOM_SEED,
    RAW_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def download_tce_table(save_path: Path = None, force: bool = False) -> pd.DataFrame:
    """
    Download the TESS TCE table from NASA Exoplanet Archive.

    This table contains one row per candidate transit signal with:
    - Star identifier (TIC ID)
    - Planet properties: period, depth, duration, radius, insolation, temperature
    - Stellar properties: magnitude, temperature, gravity, radius, distance
    - System info: number of candidates, disposition
    - Disposition (confirmed planet, false positive, candidate)

    Args:
        save_path: Where to save the raw CSV
        force: If True, skip the user prompt and re-download

    Returns:
        DataFrame with TCE metadata and labels.
    """
    import requests
    from io import StringIO

    if save_path is None:
        save_path = RAW_DIR / "tce_table.csv"

    clean_path = RAW_DIR / "tce_table_clean.csv"

    if save_path.exists() and not force:
        print(f"\n  Existing TCE data found at: {save_path}")
        print(f"  To download fresh data, the old files need to be removed.\n")
        response = input("  Re-download from NASA? This will delete old data. [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            save_path.unlink()
            if clean_path.exists():
                clean_path.unlink()
            logger.info("Removed old TCE files. Downloading data.")
        else:
            logger.info("Keeping existing data.")
            return pd.read_csv(save_path)

    required_columns = {
        "tid",
        "pl_pnum",
        "pl_orbper",
        "pl_trandep",
        "pl_trandurh",
        "pl_rade",
        "tfopwg_disp",
    }

    logger.info("Performing pre-flight schema validation.")
    metadata_query = "SELECT column_name FROM tap_schema.columns WHERE table_name = 'toi'"
    
    try:
        meta_response = requests.post(
            NASA_API_URL, 
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "format": "csv",
                "query": metadata_query
            },
            timeout=30
        )
        meta_response.raise_for_status()
        available_columns = set(pd.read_csv(StringIO(meta_response.text))["column_name"].tolist())
        
        missing = required_columns - available_columns
        if missing:
            raise RuntimeError(
                f"Schema mismatch detected. The following required columns are missing "
                f"from the NASA 'toi' table: {list(missing)}"
            )
        logger.info("Schema validation successful.")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to TAP metadata service: {e}")
        raise

    # Astronomical Data Query Language (ADQL) query to NASA's TAP service for TESS Objects.
    # Get everything useful for classification:
    #   - planet transit properties (the signal itself)
    #   - stellar host properties (eclipsing binaries have different stellar params)
    #   - system-level info (multi-planet systems are almost always real)
    query = """
    SELECT
        tid           as tic_id,
        pl_pnum       as planet_num,
        pl_orbper     as orbital_period,
        pl_trandep    as transit_depth,
        pl_trandurh   as transit_duration,
        pl_rade       as planet_radius,
        pl_insol      as insolation_flux,
        pl_eqt        as equilibrium_temp,
        st_tmag       as tess_magnitude,
        st_teff       as stellar_temp,
        st_logg       as stellar_logg,
        st_rad        as stellar_radius,
        st_dist       as stellar_distance,
        tfopwg_disp   as disposition
    FROM TOI
    WHERE tfopwg_disp IS NOT NULL
    """

    logger.info("Downloading TESS TOI table from NASA Exoplanet Archive")
    payload = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "format": "csv",
        "query": query
    }

    response = requests.post(NASA_API_URL, data=payload, timeout=120)
    response.raise_for_status()

    save_path.write_text(response.text)
    logger.info(f"Saved raw TCE table to {save_path}")

    df = pd.read_csv(save_path)
    logger.info(f"Downloaded {len(df)} rows with {len(df.columns)} columns: {list(df.columns)}")
    return df


def clean_tce_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the TCE table and add binary labels.

    Steps:
     (1) Map dispositions to binary labels (1=planet, 0=not planet)
     (2) Drop rows with ambiguous disposition (PC = planet candidate)
     (3) Drop rows with missing critical values
     (4) Log class distribution

    Returns:
        Cleaned DataFrame with 'label' column.
    """

    # map dispositions to labels
    df = df.copy()
    df["label"] = df["disposition"].map(DISPOSITION_MAP)

    # log raw distribution
    logger.info(f"Disposition distribution:\n{df['disposition'].value_counts().to_string()}")

    # remove ambiguous candidates (label = -1)
    n_before = len(df)
    df = df[df["label"] >= 0].reset_index(drop=True)
    logger.info(f"Removed {n_before - len(df)} ambiguous candidates (PC).")

    # drop rows missing critical columns
    critical_cols = ["tic_id", "orbital_period", "transit_depth"]
    df = df.dropna(subset=critical_cols).reset_index(drop=True)

    # Log class balance
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    ratio = n_neg / max(n_pos, 1)
    logger.info(
        f"Final dataset: {len(df)} TCEs | "
        f"{n_pos} planets ({n_pos/len(df)*100:.1f}%) | "
        f"{n_neg} false positives ({n_neg/len(df)*100:.1f}%) | "
        f"Imbalance ratio: {ratio:.1f}:1"
    )
    return df


def download_light_curve(tic_id: int, save_dir: Path = None) -> np.ndarray:
    """
    Download a light curve for a single TIC ID using lightkurve.

    lightkurve handles the MAST API calls and returns a cleaned
    time-series of brightness measurements.

    Handles common failure modes:
    - MaskedNDArray from astropy (convert to plain numpy before save)
    - Quality column type mismatches when stitching sectors
      (filter to SPOC/TESS-SPOC author to avoid HLSP products)
    - Corrupt cached FITS files (auto-delete and retry once)

    Args:
        tic_id: TESS Input Catalog identifier
        save_dir: Where to save the .npy file

    Returns:
        Array of shape (N, 2) with columns [time, flux], or None on failure.
    """
    import lightkurve as lk

    if save_dir is None:
        save_dir = RAW_DIR / "lightcurves"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"tic_{int(tic_id)}.npy"
    if save_path.exists():
        return np.load(save_path)

    try:
        # Search for TESS light curves for this star
        search = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
        if len(search) == 0:
            logger.warning(f"No light curves found for TIC {int(tic_id)}")
            return None

        # Filter to official SPOC pipeline products only.
        # HLSP (community) products like DIAMANTE have different column
        # types (str vs int for 'quality') that cause stitching to fail.
        spoc_mask = search.author.data.astype(str)
        is_spoc = np.array([
            a.upper() in ("SPOC", "TESS-SPOC") for a in spoc_mask
        ])

        if is_spoc.any():
            search = search[is_spoc]
        else:
            # No SPOC products — try downloading whatever is available,
            # but only the first sector to avoid stitching issues
            logger.info(
                f"TIC {int(tic_id)}: No SPOC products, "
                f"trying first available product"
            )
            search = search[0:1]

        # Download and stitch all available sectors together
        lc_collection = search.download_all()

        if len(lc_collection) > 1:
            lc = lc_collection.stitch()
        else:
            lc = lc_collection[0]

        # Remove NaN values and outliers
        lc = lc.remove_nans().remove_outliers(sigma=5.0)

        # Extract time and flux as plain numpy arrays.
        # lc.flux.value can return astropy MaskedNDArray which
        # np.save doesn't support — np.asarray converts it.
        time_arr = np.asarray(lc.time.value, dtype=np.float64)
        flux_arr = np.asarray(lc.flux.value, dtype=np.float64)

        # Final NaN safety net after conversion
        valid = np.isfinite(time_arr) & np.isfinite(flux_arr)
        time_arr = time_arr[valid]
        flux_arr = flux_arr[valid]

        if len(time_arr) < 100:
            logger.warning(
                f"TIC {int(tic_id)}: only {len(time_arr)} valid points, skipping"
            )
            return None

        data = np.column_stack([time_arr, flux_arr])
        np.save(save_path, data)
        return data

    except OSError as e:
        # Corrupt cached FITS file — delete and retry once
        if "corrupt" in str(e).lower() or "not recognized" in str(e).lower():
            logger.warning(
                f"TIC {int(tic_id)}: corrupt cache file detected, "
                f"clearing lightkurve cache and retrying..."
            )
            _clear_tic_cache(tic_id)
            try:
                return _download_light_curve_retry(tic_id, save_dir, lk)
            except Exception as retry_e:
                logger.warning(f"TIC {int(tic_id)} retry failed: {retry_e}")
                return None
        logger.warning(f"Failed to download TIC {int(tic_id)}: {e}")
        return None

    except Exception as e:
        logger.warning(f"Failed to download TIC {int(tic_id)}: {e}")
        return None


def _clear_tic_cache(tic_id: int):
    """Remove cached lightkurve files for a specific TIC ID."""
    import shutil
    cache_dir = Path.home() / ".lightkurve" / "cache" / "mastDownload"
    if cache_dir.exists():
        tic_str = str(int(tic_id)).zfill(16)
        for path in cache_dir.rglob(f"*{tic_str}*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def _download_light_curve_retry(tic_id: int, save_dir: Path, lk) -> np.ndarray:
    """Single retry attempt after cache clear."""
    search = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
    if len(search) == 0:
        return None

    # On retry, just grab first sector to keep it simple
    lc = search[0].download()
    lc = lc.remove_nans().remove_outliers(sigma=5.0)

    time_arr = np.asarray(lc.time.value, dtype=np.float64)
    flux_arr = np.asarray(lc.flux.value, dtype=np.float64)

    valid = np.isfinite(time_arr) & np.isfinite(flux_arr)
    time_arr = time_arr[valid]
    flux_arr = flux_arr[valid]

    if len(time_arr) < 100:
        return None

    save_path = save_dir / f"tic_{int(tic_id)}.npy"
    data = np.column_stack([time_arr, flux_arr])
    np.save(save_path, data)
    return data


def download_all_light_curves(tce_df: pd.DataFrame, limit: int = None):
    """
    Download light curves for stars in the TCE table.

    Prioritizes stars already on disk, then fills the remainder
    with new stratified samples. This means --limit 400 after a
    previous --limit 300 run will keep all 300 existing files and
    only download ~100 new ones.

    Args:
        tce_df: TCE table with 'tic_id' and 'label' columns
        limit: Target total number of stars with light curves.
               Includes stars already downloaded.
    """
    save_dir = RAW_DIR / "lightcurves"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find which stars are already downloaded
    existing_tics = set()
    for f in save_dir.glob("tic_*.npy"):
        tic_id = int(f.stem.split("_")[1])
        existing_tics.add(tic_id)

    if limit is not None:
        # Start with all already-downloaded stars that are in the TCE table
        all_tics_in_data = set(tce_df["tic_id"].unique())
        already_have = existing_tics & all_tics_in_data
        n_already = len(already_have)

        n_needed = max(0, limit - n_already)

        if n_needed == 0:
            logger.info(
                f"Already have {n_already} light curves on disk "
                f"(target: {limit}). Nothing to download."
            )
            return

        logger.info(
            f"Already have {n_already} light curves on disk. "
            f"Need {n_needed} more to reach target of {limit}."
        )

        # Stratified sample from stars NOT obtained yet
        remaining_df = tce_df[~tce_df["tic_id"].isin(existing_tics)]
        np.random.seed(RANDOM_SEED)
        planets = remaining_df[remaining_df["label"] == 1]["tic_id"].unique()
        false_pos = remaining_df[remaining_df["label"] == 0]["tic_id"].unique()

        total_remaining = len(planets) + len(false_pos)
        if total_remaining == 0:
            logger.info("No more stars available to download.")
            return

        planet_frac = len(planets) / total_remaining
        n_planets = max(1, int(n_needed * planet_frac))
        n_fp = n_needed - n_planets

        selected_planets = np.random.choice(
            planets, size=min(n_planets, len(planets)), replace=False
        )
        selected_fp = np.random.choice(
            false_pos, size=min(n_fp, len(false_pos)), replace=False
        )
        tic_ids = np.concatenate([selected_planets, selected_fp])
        np.random.shuffle(tic_ids)

        logger.info(
            f"Downloading {len(tic_ids)} new stars "
            f"({len(selected_planets)} planets + {len(selected_fp)} FP)"
        )
    else:
        tic_ids = tce_df["tic_id"].unique()
        # Still skip existing ones
        tic_ids = np.array([t for t in tic_ids if t not in existing_tics])
        if len(tic_ids) == 0:
            logger.info("All stars already downloaded.")
            return
        logger.info(f"Downloading {len(tic_ids)} stars (skipping {len(existing_tics)} already on disk)...")

    success, fail = 0, 0
    for tic_id in tqdm(tic_ids, desc="Downloading light curves"):
        result = download_light_curve(tic_id)
        if result is not None:
            success += 1
        else:
            fail += 1
        # Rate limiting: 0.5 second between requests
        time.sleep(0.5)

    logger.info(
        f"Downloads complete: {success} succeeded, {fail} failed. "
        f"Total on disk: {len(existing_tics) + success}"
    )


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TESS exoplanet data")
    parser.add_argument("--lightcurves", action="store_true",
                        help="Also download light curve files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of stars to download light curves for. "
                             "Uses stratified sampling to keep class balance. "
                             "Example: --limit 300 (~10 min download)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download of TCE table without prompting")
    args = parser.parse_args()

    # Step 1: TCE table (always)
    df = download_tce_table(force=args.force)
    df = clean_tce_table(df)
    df.to_csv(RAW_DIR / "tce_table_clean.csv", index=False)

    # Step 2: Light curves (optional)
    if args.lightcurves:
        download_all_light_curves(df, limit=args.limit)
