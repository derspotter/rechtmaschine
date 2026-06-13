#!/usr/bin/env python
"""
Build a content-dedup index from the Nextcloud export manifest.

For each staged document we record two keys:
- byte sha256 (already in the manifest) — catches exact file copies.
- normalized-text sha256 — extract text, de-hyphenate line breaks, collapse
  whitespace, lowercase, hash. Catches the same content across different bytes
  and filenames (an authored ODT vs. its filed PDF render).

The j-lawyer top-up connector loads this index and skips any j-lawyer document
whose content is already represented in the Nextcloud corpus, so the same
filing is never embedded twice.

Run where the staged files live (debian import mirror or desktop export):

    .venv-rag/bin/python rag/build_dedup_index.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from ingest_runner import extract_text, strip_boilerplate  # noqa: E402  (shared)


DEFAULT_IMPORT_ROOT = RAG_DIR / "data" / "imports" / "desktop-export"

_WHITESPACE = re.compile(r"\s+")
_HYPHEN_BREAK = re.compile(r"(\w)[­-]\s*\n?\s*(\w)")


def normalize_for_hash(text: str) -> str:
    """Canonicalize text so an ODT and its PDF render hash identically."""
    text = text.replace("­", "-")          # soft hyphen -> regular, then de-hyphenate
    text = _HYPHEN_BREAK.sub(r"\1\2", text)      # join "Verwal-\ntung"/"vertra- ges" -> one word
    return _WHITESPACE.sub(" ", text).strip().lower()


def content_sha256(text: str) -> str:
    """Hash the substantive body. strip_boilerplate first so a clean ODT and its
    PDF render (which carries interleaved letterhead) hash identically — the
    cross-format duplicate case the j-lawyer top-up must catch."""
    return hashlib.sha256(
        normalize_for_hash(strip_boilerplate(text)).encode("utf-8")
    ).hexdigest()


def _latest_manifest(import_root: Path) -> Path:
    manifests = sorted(
        (import_root / "manifests").glob("nextcloud_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"No nextcloud_*.jsonl under {import_root}/manifests")
    return manifests[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build content-dedup index from Nextcloud manifest")
    parser.add_argument("--import-root", type=Path, default=DEFAULT_IMPORT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    manifest = args.manifest or _latest_manifest(args.import_root)
    items = [json.loads(line) for line in manifest.open() if line.strip()]

    byte_shas: set[str] = set()
    text_hashes: dict[str, str] = {}  # text-hash -> sample source byte-sha16 (for tracing)
    short = errors = 0

    # Per-case normalized basenames of text-bearing files. Scoped to the case so
    # that generic names (Klage.pdf, Gericht.odt) recurring across cases never
    # cross-match, and restricted to text-bearing files so a j-lawyer ODT whose
    # only Nextcloud twin is a textless scan still counts as new (better version).
    case_files: dict[str, list[str]] = {}

    for item in items:
        byte_shas.add(item["sha256"])
        source = args.import_root / item["staged_rel_path"]
        try:
            text = extract_text(source)
        except Exception:
            errors += 1
            continue
        if len(text) < args.min_chars:
            short += 1
            continue
        text_hashes.setdefault(content_sha256(text), item["sha256"][:16])

        year, folder = item.get("case_year"), item.get("case_folder")
        if year and folder:
            key = f"{str(folder).split()[0]}/{year}"  # "003/21", matches j-lawyer ref
            stem = Path(item["filename"]).stem.lower()
            case_files.setdefault(key, [])
            if stem not in case_files[key]:
                case_files[key].append(stem)

    out = args.out or (args.import_root / "dedup_index.json")
    out.write_text(
        json.dumps(
            {
                "source": "nextcloud",
                "manifest": manifest.name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "byte_sha256": sorted(byte_shas),
                "text_sha256": text_hashes,
                "case_files": case_files,
                "counts": {
                    "manifest_items": len(items),
                    "unique_text_hashes": len(text_hashes),
                    "cases": len(case_files),
                    "short_skipped": short,
                    "extract_errors": errors,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest.name} ({len(items)} items)")
    print(f"Index:    {out}")
    print(
        f"byte hashes: {len(byte_shas)} | text hashes: {len(text_hashes)} | "
        f"short-skipped: {short} | extract-errors: {errors}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
