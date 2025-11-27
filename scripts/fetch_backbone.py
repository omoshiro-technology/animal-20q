#!/usr/bin/env python3
"""
fetch_backbone.py - Download taxonomy backbone data from CoL or NCBI

Usage:
    python fetch_backbone.py --source col --out data/backbone/col/
    python fetch_backbone.py --source ncbi --out data/backbone/ncbi/
"""

import argparse
import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import ssl

# CoL Annual Checklist download URL (latest version)
COL_DOWNLOAD_URL = "https://download.checklistbank.org/col/annual/col-dwca.zip"

# NCBI Taxonomy dump URL
NCBI_TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"


def create_ssl_context():
    """Create SSL context for downloads."""
    ctx = ssl.create_default_context()
    return ctx


def download_file(url: str, dest_path: Path, desc: str = "file") -> bool:
    """Download a file with progress indication."""
    print(f"Downloading {desc} from {url}...")

    try:
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                progress = min(100, block_num * block_size * 100 // total_size)
                print(f"\rProgress: {progress}%", end="", flush=True)

        urlretrieve(url, dest_path, reporthook=report_progress)
        print()  # New line after progress
        return True

    except URLError as e:
        print(f"\nError downloading {desc}: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False


def fetch_col(out_dir: Path) -> bool:
    """
    Download and extract Catalogue of Life (CoL) DwC-A archive.

    The archive contains:
    - Taxon.tsv: Main taxonomy file
    - VernacularName.tsv: Common names
    - meta.xml: Archive metadata
    """
    print("=== Fetching Catalogue of Life (CoL) ===")

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "col-dwca.zip"

    # Download
    if not download_file(COL_DOWNLOAD_URL, zip_path, "CoL DwC-A archive"):
        return False

    # Extract
    print("Extracting archive...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
        print(f"Extracted to {out_dir}")

        # List extracted files
        files = list(out_dir.iterdir())
        print(f"Files: {[f.name for f in files if f.is_file()]}")

        # Clean up zip file to save space
        zip_path.unlink()
        print("Cleaned up archive file")

        return True

    except zipfile.BadZipFile as e:
        print(f"Error extracting archive: {e}")
        return False


def fetch_ncbi(out_dir: Path) -> bool:
    """
    Download and extract NCBI Taxonomy dump.

    The dump contains:
    - nodes.dmp: Taxonomy nodes (taxid, parent_taxid, rank, etc.)
    - names.dmp: Taxonomy names (scientific names, synonyms, common names)
    - division.dmp: Divisions (Bacteria, Plants, Animals, etc.)
    """
    print("=== Fetching NCBI Taxonomy dump ===")

    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "taxdump.tar.gz"

    # Download
    if not download_file(NCBI_TAXDUMP_URL, tar_path, "NCBI taxdump"):
        return False

    # Extract
    print("Extracting archive...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tf:
            tf.extractall(out_dir)
        print(f"Extracted to {out_dir}")

        # List extracted files
        files = list(out_dir.iterdir())
        print(f"Files: {[f.name for f in files if f.is_file()]}")

        # Clean up tar file to save space
        tar_path.unlink()
        print("Cleaned up archive file")

        return True

    except tarfile.TarError as e:
        print(f"Error extracting archive: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download taxonomy backbone data from CoL or NCBI"
    )
    parser.add_argument(
        "--source",
        choices=["col", "ncbi"],
        required=True,
        help="Data source: 'col' for Catalogue of Life, 'ncbi' for NCBI Taxonomy"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for downloaded/extracted files"
    )

    args = parser.parse_args()

    # Fetch data based on source
    if args.source == "col":
        success = fetch_col(args.out)
    else:
        success = fetch_ncbi(args.out)

    if success:
        print(f"\n{args.source.upper()} data fetched successfully!")
        return 0
    else:
        print(f"\nFailed to fetch {args.source.upper()} data")
        return 1


if __name__ == "__main__":
    sys.exit(main())
