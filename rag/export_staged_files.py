#!/usr/bin/env python
"""
Stage manifest INCLUDE files into the desktop RAG export root.

Input:
- Manifest JSONL produced by export_manifest.py (default: latest in rag/data/manifests)

Output (under the export root, default /home/jayjag/rechtmaschine-rag-export):
- staged_files/nextcloud/<source_rel_path>  — copied source files
- manifests/nextcloud_<run_id>.jsonl        — export-relative manifest with sha256 per item
- checksums/nextcloud_<run_id>.sha256       — `sha256sum -c`-compatible list over staged files

Safety:
- Never writes to the corpus directory.
- Idempotent: existing staged files with matching size are not re-copied;
  hashes are always computed from the source.

Debian pulls the export root via rsync and verifies the checksum file before
processing (see docs/rag-three-machine-ingestion-plan.md, "Export Boundary").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MANIFESTS_DIR = Path(__file__).resolve().parent / "data" / "manifests"
DEFAULT_EXPORT_ROOT = Path("/home/jayjag/rechtmaschine-rag-export")
SOURCE_SYSTEM = "nextcloud"


def _latest_manifest(manifests_dir: Path) -> Path:
    manifests = sorted(
        manifests_dir.glob("manifest_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"No manifest_*.jsonl found in {manifests_dir}")
    return manifests[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage manifest files into the desktop RAG export root."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Manifest JSONL. Default: latest in {DEFAULT_MANIFESTS_DIR}",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Corpus directory the manifest's source_rel_path entries resolve against "
        "(use the snapshot copy, not the live corpus).",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help=f"Export root directory (default: {DEFAULT_EXPORT_ROOT})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        help="Export run id used in output filenames (default: UTC timestamp).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on staged files (for smoke runs).",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete staged files no longer present in the manifest "
        "(e.g. after a filter change). Off by default.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest or _latest_manifest(DEFAULT_MANIFESTS_DIR)
    corpus_dir: Path = args.corpus_dir.resolve()
    export_root: Path = args.export_root.resolve()

    if not corpus_dir.is_dir():
        print(f"ERROR: corpus dir not found: {corpus_dir}", file=sys.stderr)
        return 1
    if export_root == corpus_dir or export_root in corpus_dir.parents:
        print("ERROR: export root must not contain the corpus directory", file=sys.stderr)
        return 1

    staged_dir = export_root / "staged_files" / SOURCE_SYSTEM
    manifests_dir = export_root / "manifests"
    checksums_dir = export_root / "checksums"
    for directory in (staged_dir, manifests_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    out_manifest = manifests_dir / f"{SOURCE_SYSTEM}_{args.run_id}.jsonl"
    out_checksums = checksums_dir / f"{SOURCE_SYSTEM}_{args.run_id}.sha256"

    staged = 0
    skipped_existing = 0
    missing: list[str] = []
    kept_rel: set[str] = set()

    with out_manifest.open("w", encoding="utf-8") as manifest_out, out_checksums.open(
        "w", encoding="utf-8"
    ) as checksums_out:
        with manifest_path.open("r", encoding="utf-8") as manifest_in:
            for line in manifest_in:
                line = line.strip()
                if not line:
                    continue
                item: dict[str, Any] = json.loads(line)
                rel_path = item["source_rel_path"]
                source = corpus_dir / rel_path
                if not source.is_file():
                    missing.append(rel_path)
                    continue

                dest = staged_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.stat().st_size == source.stat().st_size:
                    skipped_existing += 1
                else:
                    shutil.copy2(source, dest)

                sha256 = _sha256(source)
                staged_rel = f"staged_files/{SOURCE_SYSTEM}/{rel_path}"
                record = {
                    "source_system": SOURCE_SYSTEM,
                    "source_rel_path": rel_path,
                    "staged_rel_path": staged_rel,
                    "sha256": sha256,
                    "size_bytes": source.stat().st_size,
                    "extension": item.get("extension"),
                    "case_year": item.get("case_year"),
                    "case_folder": item.get("case_folder"),
                    "filename": item.get("filename"),
                    "signal_codes": item.get("signal_codes", []),
                    "date_prefix": item.get("date_prefix"),
                    "court_token": item.get("court_token"),
                    "doc_token": item.get("doc_token"),
                    "letterhead_confidence": item.get("letterhead_confidence"),
                    "manifest_run_id": args.run_id,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                }
                manifest_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                checksums_out.write(f"{sha256}  {staged_rel}\n")
                kept_rel.add(rel_path)
                staged += 1
                if args.limit is not None and staged >= args.limit:
                    break

    pruned = 0
    if args.prune:
        for path in staged_dir.rglob("*"):
            if not path.is_file():
                continue
            if str(path.relative_to(staged_dir)) not in kept_rel:
                path.unlink()
                pruned += 1
        # Drop directories left empty by pruning.
        for path in sorted(staged_dir.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    print(f"Manifest:        {manifest_path}")
    print(f"Export manifest: {out_manifest}")
    print(f"Checksums:       {out_checksums}")
    print(f"Staged (in manifest): {staged}  (already present, copy skipped: {skipped_existing})")
    if args.prune:
        print(f"Pruned (no longer in manifest): {pruned}")
    if missing:
        print(f"WARNING: {len(missing)} manifest items missing in corpus, first 5:")
        for rel in missing[:5]:
            print(f"  - {rel}")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
