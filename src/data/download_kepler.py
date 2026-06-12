"""
Kepler DR25 TCE download module.

Fetches the Q1-Q17 DR25 Threshold Crossing Event table (~34,000 rows) from
the NASA Exoplanet Archive — the pre-training corpus for this project's CNNs
(vs the ~2,600 labeled TESS TCEs we train on today).

Labels: the DR25 TCE table's Autovetter columns (av_training_set) come back
null through the TAP API, so labels are derived by joining against the
Q1-Q17 DR25 KOI table on (kepid, planet number):

    KOI disposition CONFIRMED       -> 1   (planet)
    KOI disposition FALSE POSITIVE  -> 0   (not planet)
    TCE with no federated KOI       -> 0   (Robovetter-rejected junk)
    KOI disposition CANDIDATE       -> excluded from training; exported to
                                       kepler_candidates.csv as the inference
                                       set for the "classify unconfirmed
                                       candidates" roadmap item

Note the class balance: ~2.7k planets vs ~30k false positives (about 1:11),
unlike the curated near-balanced TESS set. Training on this corpus needs
imbalance handling (class-weighted or focal loss, AUPRC metric).

Light curves are NOT downloaded here; that is a separate (disk-aware) step,
since ~34k Kepler light curves exceed typical free disk space if fetched
naively.

Usage:
    python -m src.data.download_kepler            # download + clean (skips if cached)
    python -m src.data.download_kepler --force    # re-download from NASA
    python -m src.data.download_kepler --s3       # also push CSVs to the S3 bucket
"""

import argparse
import logging

import pandas as pd

from pathlib import Path

from src.utils.config import (
    KEPLER_KOI_LABEL_MAP,
    NASA_API_URL,
    RAW_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Table names are case-sensitive in the archive's TAP schema metadata.
KEPLER_TCE_TABLE = "q1_q17_dr25_tce"
KEPLER_TCE_TABLE_SCHEMA = "Q1_Q17_DR25_TCE"
KEPLER_KOI_TABLE = "q1_q17_dr25_koi"

RAW_PATH = RAW_DIR / "kepler_tce_table.csv"
KOI_PATH = RAW_DIR / "kepler_koi_table.csv"
CLEAN_PATH = RAW_DIR / "kepler_tce_table_clean.csv"
CANDIDATES_PATH = RAW_DIR / "kepler_candidates.csv"

# DR25 TCE column -> project-convention name. Mirrors the TESS download where
# an equivalent exists so downstream feature code can stay shared.
COLUMN_MAP = {
    "kepid": "kepid",                       # star identifier (Kepler ID)
    "tce_plnt_num": "planet_num",           # TCE number for this star
    "tce_period": "orbital_period",         # days
    "tce_time0bk": "epoch_bkjd",            # transit epoch (BKJD)
    "tce_duration": "transit_duration",     # hours
    "tce_depth": "transit_depth",           # ppm
    "tce_model_snr": "transit_snr",
    "tce_num_transits": "num_transits",
    "tce_max_mult_ev": "mes",               # multiple event statistic
    "tce_impact": "impact",
    "tce_prad": "planet_radius",            # Earth radii
    "tce_insol": "insolation_flux",         # Earth flux
    "tce_eqt": "equilibrium_temp",          # Kelvin
    "tce_steff": "stellar_temp",            # Kelvin
    "tce_slogg": "stellar_logg",
    "tce_sradius": "stellar_radius",        # Solar radii
    # Robovetter-style vetting diagnostics. These are the discriminators the
    # DR25 pipeline itself computes to separate planets from false positives.
    "tce_bin_oedp_stat": "odd_even_stat",   # odd vs even transit depth (EB tell)
    "tce_dicco_msky": "centroid_offset_dic", # difference-image centroid offset, arcsec
    "tce_dikco_msky": "centroid_offset_kic", # centroid offset vs KIC position, arcsec
    "boot_fap": "boot_fap",                 # bootstrap false-alarm probability
    "tce_cap_stat": "ghost_core_stat",      # ghost diagnostic, core aperture
    "tce_hap_stat": "ghost_halo_stat",      # ghost diagnostic, halo aperture
    "tce_max_sngle_ev": "max_single_event", # max single-event statistic
}


def _tap_query(query: str, timeout: int = 300) -> str:
    import requests

    response = requests.post(
        NASA_API_URL,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "format": "csv", "query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def download_kepler_tce_table(save_path: Path = None, force: bool = False) -> pd.DataFrame:
    """
    Download the Kepler DR25 TCE table from the NASA Exoplanet Archive.

    One row per threshold crossing event (~34k rows) with transit properties
    and stellar host properties. Labels are added separately via the KOI join.

    Args:
        save_path: Where to save the raw CSV
        force: If True, re-download even if a cached copy exists

    Returns:
        DataFrame with DR25 TCE metadata.
    """
    from io import StringIO

    if save_path is None:
        save_path = RAW_PATH

    if save_path.exists() and not force:
        logger.info(f"Using cached Kepler TCE table at {save_path} (pass --force to re-download).")
        return pd.read_csv(save_path)

    logger.info("Performing pre-flight schema validation.")
    available_csv = _tap_query(
        "SELECT column_name FROM tap_schema.columns "
        f"WHERE table_name = '{KEPLER_TCE_TABLE_SCHEMA}'",
        timeout=30,
    )
    available = set(pd.read_csv(StringIO(available_csv))["column_name"].tolist())
    missing = set(COLUMN_MAP) - available
    if missing:
        raise RuntimeError(
            f"Schema mismatch: columns missing from NASA '{KEPLER_TCE_TABLE}': {sorted(missing)}"
        )
    logger.info("Schema validation successful.")

    select = ",\n        ".join(
        f"{src} as {dst}" if src != dst else src for src, dst in COLUMN_MAP.items()
    )
    logger.info("Downloading Kepler DR25 TCE table from NASA Exoplanet Archive")
    text = _tap_query(f"SELECT\n        {select}\n    FROM {KEPLER_TCE_TABLE}")

    save_path.write_text(text)
    logger.info(f"Saved raw Kepler TCE table to {save_path}")

    df = pd.read_csv(save_path)
    logger.info(f"Downloaded {len(df)} rows with {len(df.columns)} columns: {list(df.columns)}")
    return df


def download_kepler_koi_table(save_path: Path = None, force: bool = False) -> pd.DataFrame:
    """
    Download the DR25 KOI table (dispositions used as ground-truth labels).
    """
    if save_path is None:
        save_path = KOI_PATH

    if save_path.exists() and not force:
        logger.info(f"Using cached KOI table at {save_path}.")
        return pd.read_csv(save_path)

    logger.info("Downloading Kepler DR25 KOI table from NASA Exoplanet Archive")
    text = _tap_query(
        "SELECT kepid, kepoi_name, koi_tce_plnt_num as planet_num, "
        "koi_disposition, koi_pdisposition "
        f"FROM {KEPLER_KOI_TABLE}"
    )
    save_path.write_text(text)

    df = pd.read_csv(save_path)
    logger.info(f"Downloaded {len(df)} KOIs. Dispositions:\n{df['koi_disposition'].value_counts().to_string()}")
    return df


def label_kepler_tces(tce: pd.DataFrame, koi: pd.DataFrame) -> pd.DataFrame:
    """
    Attach binary labels to TCEs via the KOI join.

    Steps:
     (1) Left-join KOI dispositions on (kepid, planet_num)
     (2) Map dispositions through KEPLER_KOI_LABEL_MAP; non-KOI TCEs -> 0
     (3) Drop rows with missing critical values
     (4) Log join and class statistics

    Returns:
        DataFrame with 'label' column: 1, 0, or -1 (candidate, excluded
        from training but kept for the inference set).
    """
    koi = koi.dropna(subset=["planet_num"]).copy()
    koi["planet_num"] = koi["planet_num"].astype(int)

    df = tce.merge(
        koi[["kepid", "kepoi_name", "planet_num", "koi_disposition"]],
        on=["kepid", "planet_num"],
        how="left",
    )
    n_matched = df["koi_disposition"].notna().sum()
    logger.info(f"KOI join: {n_matched} of {len(df)} TCEs federate to a KOI "
                f"({len(koi) - n_matched} KOIs unmatched).")

    df["label"] = df["koi_disposition"].map(KEPLER_KOI_LABEL_MAP)
    df.loc[df["koi_disposition"].isna(), "label"] = 0  # no KOI = Robovetter-rejected
    df["label"] = df["label"].astype(int)

    critical_cols = ["kepid", "orbital_period", "transit_duration"]
    df = df.dropna(subset=critical_cols).reset_index(drop=True)

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    n_cand = int((df["label"] == -1).sum())
    logger.info(
        f"Labeled corpus: {n_pos + n_neg} TCEs for training | "
        f"{n_pos} planets ({n_pos/(n_pos+n_neg)*100:.1f}%) | "
        f"{n_neg} false positives | imbalance ratio 1:{n_neg/max(n_pos,1):.1f} | "
        f"{n_cand} unconfirmed candidates exported separately"
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Download Kepler DR25 TCE metadata + labels.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--s3", action="store_true", help="Push raw + clean CSVs to the S3 bucket")
    args = parser.parse_args()

    tce = download_kepler_tce_table(force=args.force)
    koi = download_kepler_koi_table(force=args.force)
    df = label_kepler_tces(tce, koi)

    train = df[df["label"] >= 0].reset_index(drop=True)
    candidates = df[df["label"] == -1].reset_index(drop=True)
    train.to_csv(CLEAN_PATH, index=False)
    candidates.to_csv(CANDIDATES_PATH, index=False)
    logger.info(f"Saved labeled training corpus to {CLEAN_PATH}")
    logger.info(f"Saved unconfirmed-candidate inference set to {CANDIDATES_PATH}")

    if args.s3:
        from src.data.s3_sync import push
        for path in (RAW_PATH, KOI_PATH, CLEAN_PATH, CANDIDATES_PATH):
            push(path, "metadata")


if __name__ == "__main__":
    main()
