#!/usr/bin/env python3
"""
build_taxonomy.py - Build taxonomy.jsonl from CoL and/or NCBI data

Extracts Animalia kingdom and outputs a normalized taxonomy tree.

Usage:
    python build_taxonomy.py --col data/backbone/col/ --out data/taxonomy.jsonl
    python build_taxonomy.py --ncbi data/backbone/ncbi/ --out data/taxonomy.jsonl
    python build_taxonomy.py --col data/backbone/col/ --ncbi data/backbone/ncbi/ --out data/taxonomy.jsonl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Standard taxonomic rank order (high to low)
RANK_ORDER = [
    "kingdom", "subkingdom", "phylum", "subphylum", "superclass",
    "class", "subclass", "infraclass", "superorder", "order",
    "suborder", "infraorder", "superfamily", "family", "subfamily",
    "tribe", "subtribe", "genus", "subgenus", "species", "subspecies",
    "variety", "form"
]


class TaxonomyNode:
    """Represents a single taxonomy node."""

    def __init__(self, taxon_id: str, parent_id: Optional[str], rank: str,
                 scientific_name: str, vernacular_name: Optional[str] = None):
        self.taxon_id = taxon_id
        self.parent_id = parent_id
        self.rank = rank.lower() if rank else "unranked"
        self.scientific_name = scientific_name
        self.vernacular_name = vernacular_name
        self.synonyms: List[str] = []
        self.children: List[str] = []

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        result = {
            "taxonID": self.taxon_id,
            "parentID": self.parent_id,
            "rank": self.rank,
            "scientificName": self.scientific_name,
        }
        if self.vernacular_name:
            result["vernacularName"] = self.vernacular_name
        if self.synonyms:
            result["synonyms"] = self.synonyms
        if self.children:
            result["children"] = self.children
        return result


class TaxonomyBuilder:
    """Builds taxonomy from CoL and/or NCBI data."""

    def __init__(self):
        self.nodes: Dict[str, TaxonomyNode] = {}
        self.parent_map: Dict[str, str] = {}
        self.children_map: Dict[str, List[str]] = defaultdict(list)

    def load_col(self, col_dir: Path) -> bool:
        """Load taxonomy from Catalogue of Life DwC-A extract."""
        print("Loading CoL data...")

        # Try to find Taxon.tsv
        taxon_file = None
        for candidate in ["Taxon.tsv", "taxon.tsv", "taxa.tsv"]:
            path = col_dir / candidate
            if path.exists():
                taxon_file = path
                break

        if not taxon_file:
            # Try to find in subdirectory
            for subdir in col_dir.iterdir():
                if subdir.is_dir():
                    for candidate in ["Taxon.tsv", "taxon.tsv", "taxa.tsv"]:
                        path = subdir / candidate
                        if path.exists():
                            taxon_file = path
                            break

        if not taxon_file:
            print(f"Error: Could not find Taxon.tsv in {col_dir}")
            return False

        print(f"Reading {taxon_file}...")

        # Read Taxon.tsv
        try:
            with open(taxon_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                count = 0
                animalia_ids: Set[str] = set()

                # First pass: find all Animalia IDs
                rows = list(reader)
                for row in rows:
                    kingdom = row.get('kingdom', row.get('dwc:kingdom', ''))
                    if kingdom.lower() == 'animalia':
                        taxon_id = row.get('taxonID', row.get('id', ''))
                        if taxon_id:
                            animalia_ids.add(taxon_id)

                # Second pass: build nodes for Animalia
                for row in rows:
                    taxon_id = row.get('taxonID', row.get('id', ''))
                    if not taxon_id or taxon_id not in animalia_ids:
                        continue

                    parent_id = row.get('parentNameUsageID', row.get('parentID', ''))
                    rank = row.get('taxonRank', row.get('rank', 'unranked'))
                    sci_name = row.get('scientificName', row.get('canonicalName', ''))
                    vernacular = row.get('vernacularName', '')

                    # Skip if no scientific name
                    if not sci_name:
                        continue

                    # Create node
                    node = TaxonomyNode(
                        taxon_id=taxon_id,
                        parent_id=parent_id if parent_id in animalia_ids else None,
                        rank=rank,
                        scientific_name=sci_name,
                        vernacular_name=vernacular if vernacular else None
                    )

                    self.nodes[taxon_id] = node
                    count += 1

                print(f"Loaded {count} Animalia taxa from CoL")
                return count > 0

        except Exception as e:
            print(f"Error reading CoL data: {e}")
            return False

    def load_ncbi(self, ncbi_dir: Path) -> bool:
        """Load taxonomy from NCBI taxdump."""
        print("Loading NCBI data...")

        nodes_file = ncbi_dir / "nodes.dmp"
        names_file = ncbi_dir / "names.dmp"

        if not nodes_file.exists() or not names_file.exists():
            print(f"Error: Could not find nodes.dmp and names.dmp in {ncbi_dir}")
            return False

        # Read nodes.dmp
        print(f"Reading {nodes_file}...")
        node_data: Dict[str, Tuple[str, str]] = {}  # taxid -> (parent_taxid, rank)

        try:
            with open(nodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 3:
                        tax_id = parts[0]
                        parent_id = parts[1]
                        rank = parts[2]
                        node_data[tax_id] = (parent_id, rank)
        except Exception as e:
            print(f"Error reading nodes.dmp: {e}")
            return False

        # Find Animalia (Metazoa) taxon ID and all descendants
        print("Finding Animalia descendants...")
        animalia_id = None

        # Read names.dmp to find Animalia
        print(f"Reading {names_file}...")
        names: Dict[str, str] = {}  # taxid -> scientific name
        vernaculars: Dict[str, str] = {}  # taxid -> common name
        synonyms: Dict[str, List[str]] = defaultdict(list)  # taxid -> synonyms

        try:
            with open(names_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        tax_id = parts[0]
                        name = parts[1]
                        name_class = parts[3]

                        if name_class == 'scientific name':
                            names[tax_id] = name
                            if name.lower() in ('animalia', 'metazoa'):
                                animalia_id = tax_id
                        elif name_class == 'genbank common name' or name_class == 'common name':
                            if tax_id not in vernaculars:
                                vernaculars[tax_id] = name
                        elif name_class == 'synonym':
                            synonyms[tax_id].append(name)
        except Exception as e:
            print(f"Error reading names.dmp: {e}")
            return False

        if not animalia_id:
            print("Error: Could not find Animalia/Metazoa in NCBI data")
            return False

        print(f"Found Animalia with taxid: {animalia_id}")

        # Find all descendants of Animalia using BFS
        print("Building Animalia subtree...")
        children_map: Dict[str, List[str]] = defaultdict(list)
        for tax_id, (parent_id, _) in node_data.items():
            if parent_id != tax_id:  # Skip root self-reference
                children_map[parent_id].append(tax_id)

        animalia_ids: Set[str] = {animalia_id}
        queue = [animalia_id]
        while queue:
            current = queue.pop(0)
            for child in children_map.get(current, []):
                if child not in animalia_ids:
                    animalia_ids.add(child)
                    queue.append(child)

        print(f"Found {len(animalia_ids)} Animalia taxa")

        # Build nodes
        count = 0
        for tax_id in animalia_ids:
            if tax_id not in names:
                continue

            parent_id, rank = node_data.get(tax_id, (None, 'unranked'))
            if parent_id and parent_id not in animalia_ids:
                parent_id = None

            node = TaxonomyNode(
                taxon_id=f"NCBI:{tax_id}",
                parent_id=f"NCBI:{parent_id}" if parent_id else None,
                rank=rank,
                scientific_name=names[tax_id],
                vernacular_name=vernaculars.get(tax_id)
            )
            node.synonyms = synonyms.get(tax_id, [])

            self.nodes[node.taxon_id] = node
            count += 1

        print(f"Loaded {count} Animalia taxa from NCBI")
        return count > 0

    def build_children(self):
        """Build children lists for each node."""
        print("Building parent-child relationships...")

        for node_id, node in self.nodes.items():
            if node.parent_id and node.parent_id in self.nodes:
                self.children_map[node.parent_id].append(node_id)

        # Assign children to nodes
        for node_id, children in self.children_map.items():
            if node_id in self.nodes:
                self.nodes[node_id].children = sorted(children)

    def validate(self) -> bool:
        """Validate taxonomy integrity."""
        print("Validating taxonomy...")
        errors = []

        # Check for orphans (non-root nodes without valid parent)
        root_count = 0
        for node_id, node in self.nodes.items():
            if node.parent_id is None:
                root_count += 1
            elif node.parent_id not in self.nodes:
                errors.append(f"Orphan node: {node_id} (parent {node.parent_id} not found)")

        if root_count == 0:
            errors.append("No root node found")
        elif root_count > 1:
            # This is OK - we might have multiple top-level taxa
            print(f"Note: {root_count} root nodes found")

        # Check for cycles
        def has_cycle(node_id: str, visited: Set[str]) -> bool:
            if node_id in visited:
                return True
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node and node.parent_id:
                return has_cycle(node.parent_id, visited)
            return False

        cycle_count = 0
        for node_id in list(self.nodes.keys())[:1000]:  # Check first 1000 to avoid long runtime
            if has_cycle(node_id, set()):
                errors.append(f"Cycle detected involving: {node_id}")
                cycle_count += 1
                if cycle_count >= 10:
                    errors.append("... (more cycles)")
                    break

        # Check rank ordering
        rank_errors = 0
        for node_id, node in list(self.nodes.items())[:1000]:
            if node.parent_id and node.parent_id in self.nodes:
                parent = self.nodes[node.parent_id]
                if node.rank in RANK_ORDER and parent.rank in RANK_ORDER:
                    if RANK_ORDER.index(node.rank) < RANK_ORDER.index(parent.rank):
                        rank_errors += 1
                        if rank_errors <= 5:
                            errors.append(
                                f"Rank order issue: {node_id} ({node.rank}) under {parent.taxon_id} ({parent.rank})"
                            )

        if rank_errors > 5:
            errors.append(f"... ({rank_errors - 5} more rank issues)")

        if errors:
            print("Validation warnings:")
            for e in errors[:20]:
                print(f"  - {e}")
            if len(errors) > 20:
                print(f"  ... ({len(errors) - 20} more)")
        else:
            print("Validation passed")

        return len(errors) == 0

    def write_jsonl(self, out_path: Path):
        """Write taxonomy to JSONL format."""
        print(f"Writing {len(self.nodes)} nodes to {out_path}...")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort nodes by rank order, then alphabetically
        def sort_key(node_id):
            node = self.nodes[node_id]
            rank_idx = RANK_ORDER.index(node.rank) if node.rank in RANK_ORDER else 999
            return (rank_idx, node.scientific_name)

        sorted_ids = sorted(self.nodes.keys(), key=sort_key)

        with open(out_path, 'w', encoding='utf-8') as f:
            for node_id in sorted_ids:
                node = self.nodes[node_id]
                json_line = json.dumps(node.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')

        print(f"Wrote {len(sorted_ids)} nodes")


def main():
    parser = argparse.ArgumentParser(
        description="Build taxonomy.jsonl from CoL and/or NCBI data"
    )
    parser.add_argument(
        "--col",
        type=Path,
        help="Path to CoL DwC-A extract directory"
    )
    parser.add_argument(
        "--ncbi",
        type=Path,
        help="Path to NCBI taxdump directory"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for taxonomy.jsonl"
    )

    args = parser.parse_args()

    if not args.col and not args.ncbi:
        print("Error: At least one of --col or --ncbi must be specified")
        return 1

    builder = TaxonomyBuilder()

    # Load data
    loaded = False
    if args.col and args.col.exists():
        if builder.load_col(args.col):
            loaded = True
    if args.ncbi and args.ncbi.exists():
        if builder.load_ncbi(args.ncbi):
            loaded = True

    if not loaded:
        print("Error: Failed to load any taxonomy data")
        return 1

    # Build and validate
    builder.build_children()
    builder.validate()

    # Write output
    builder.write_jsonl(args.out)

    print("\nTaxonomy build complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
