"""
S3 sync helper.

Pushes and pulls pipeline artifacts against the project data bucket
(created by terraform/ -- see terraform/main.tf for the layout). Uses the
AWS CLI via subprocess rather than boto3, so no new Python dependency is
needed; the CLI is already configured on this machine.

The bucket name comes from the EXOPLANET_S3_BUCKET environment variable:

    export EXOPLANET_S3_BUCKET=$(cd terraform && terraform output -raw bucket_name)

Usage:
    python -m src.data.s3_sync push data/raw/kepler_tce_table_clean.csv metadata/
    python -m src.data.s3_sync pull metadata/kepler_tce_table_clean.csv data/raw/
    python -m src.data.s3_sync sync data/processed processed/
"""

import argparse
import logging
import subprocess
import sys

from pathlib import Path

from src.utils.config import S3_BUCKET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def _bucket() -> str:
    if not S3_BUCKET:
        raise RuntimeError(
            "EXOPLANET_S3_BUCKET is not set. Apply terraform/ first, then:\n"
            "  export EXOPLANET_S3_BUCKET=$(cd terraform && terraform output -raw bucket_name)"
        )
    return S3_BUCKET


def _run(args: list) -> None:
    logger.info("aws " + " ".join(args))
    subprocess.run(["aws"] + args, check=True)


def push(local: Path, key_prefix: str) -> None:
    """Upload one local file to s3://<bucket>/<key_prefix>/<filename>."""
    local = Path(local)
    key = f"{key_prefix.rstrip('/')}/{local.name}"
    _run(["s3", "cp", str(local), f"s3://{_bucket()}/{key}"])


def pull(key: str, local_dir: Path) -> None:
    """Download one object into a local directory."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    _run(["s3", "cp", f"s3://{_bucket()}/{key}", str(local_dir) + "/"])


def sync(local_dir: Path, key_prefix: str, down: bool = False) -> None:
    """Sync a local directory with an S3 prefix (up by default)."""
    remote = f"s3://{_bucket()}/{key_prefix.rstrip('/')}/"
    if down:
        _run(["s3", "sync", remote, str(local_dir)])
    else:
        _run(["s3", "sync", str(local_dir), remote])


def main():
    parser = argparse.ArgumentParser(description="Sync pipeline artifacts with S3.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="Upload a file to a key prefix")
    p_push.add_argument("local")
    p_push.add_argument("prefix")

    p_pull = sub.add_parser("pull", help="Download an object into a directory")
    p_pull.add_argument("key")
    p_pull.add_argument("local_dir")

    p_sync = sub.add_parser("sync", help="Sync a directory with a prefix")
    p_sync.add_argument("local_dir")
    p_sync.add_argument("prefix")
    p_sync.add_argument("--down", action="store_true", help="Sync S3 -> local instead")

    args = parser.parse_args()
    try:
        if args.cmd == "push":
            push(Path(args.local), args.prefix)
        elif args.cmd == "pull":
            pull(args.key, Path(args.local_dir))
        elif args.cmd == "sync":
            sync(Path(args.local_dir), args.prefix, down=args.down)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
