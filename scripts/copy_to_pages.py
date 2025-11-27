#!/usr/bin/env python3
"""
copy_to_pages.py - Sync data files to docs/ for GitHub Pages

Copies taxonomy.jsonl and questions.jsonl to docs/ directory
and generates an index.json manifest file.

Usage:
    python copy_to_pages.py
    python copy_to_pages.py --data data/ --docs docs/
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def copy_to_pages(data_dir: Path, docs_dir: Path) -> bool:
    """
    Copy data files to docs directory and create index.json.

    Args:
        data_dir: Source directory containing data files
        docs_dir: Destination directory for GitHub Pages

    Returns:
        True if successful, False otherwise
    """
    print(f"Syncing data from {data_dir} to {docs_dir}...")

    # Ensure docs directory exists
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Files to copy
    files_to_copy = [
        ('taxonomy.jsonl', 'taxonomy.jsonl'),
        ('questions.jsonl', 'questions.jsonl'),
    ]

    copied_files = []

    for src_name, dest_name in files_to_copy:
        src_path = data_dir / src_name
        dest_path = docs_dir / dest_name

        if src_path.exists():
            print(f"  Copying {src_name}...")
            shutil.copy2(src_path, dest_path)
            copied_files.append(dest_name)

            # Get file stats
            size = dest_path.stat().st_size
            lines = sum(1 for _ in open(dest_path, 'r', encoding='utf-8'))
            print(f"    -> {dest_name} ({lines} lines, {size:,} bytes)")
        else:
            print(f"  Warning: {src_name} not found, skipping")

    if not copied_files:
        print("Warning: No data files found to copy")

    # Create index.json
    index_path = docs_dir / 'index.json'
    index_data = {
        'taxonomy': 'taxonomy.jsonl' if 'taxonomy.jsonl' in copied_files else None,
        'questions': 'questions.jsonl' if 'questions.jsonl' in copied_files else None,
        'updated': datetime.now(timezone.utc).isoformat(),
        'version': '1.0.0',
    }

    print(f"  Creating index.json...")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"\nSync complete!")
    print(f"Files in {docs_dir}:")
    for f in sorted(docs_dir.iterdir()):
        if f.is_file():
            print(f"  - {f.name}")

    return len(copied_files) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Sync data files to docs/ for GitHub Pages"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help="Source data directory (default: data/)"
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("docs"),
        help="Destination docs directory (default: docs/)"
    )

    args = parser.parse_args()

    # Find the project root (look for data/ or go up from scripts/)
    if not args.data.is_absolute():
        # Check current directory
        if args.data.exists():
            data_dir = args.data
        else:
            # Try relative to script location
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            data_dir = project_root / args.data
    else:
        data_dir = args.data

    if not args.docs.is_absolute():
        if args.docs.exists():
            docs_dir = args.docs
        else:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            docs_dir = project_root / args.docs
    else:
        docs_dir = args.docs

    print(f"Data directory: {data_dir.absolute()}")
    print(f"Docs directory: {docs_dir.absolute()}")

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    success = copy_to_pages(data_dir, docs_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
