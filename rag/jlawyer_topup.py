#!/usr/bin/env python
"""
j-lawyer RAG top-up connector (dedup against the Nextcloud corpus).

j-lawyer holds the same authored filings as Nextcloud for lawyers who use both,
plus correspondence (beA/xjustiz/EGVP) that RAG excludes. The value of this
connector is the *delta*: own-authored documents that exist only in j-lawyer
(lawyers who do not use Nextcloud). It therefore selects authored candidates,
then keeps only those whose normalized text hash is absent from the Nextcloud
dedup index (build_dedup_index.py).

Metadata-first: classification uses the document list (no download); only
authored candidates are downloaded, and only to hash + dedup their text.

Run where j-lawyer is reachable (the jlawyer-cli is configured), with the
Nextcloud dedup index copied alongside:

    python rag/jlawyer_topup.py --cases 089/26,300/24,026/23 \
        --dedup-index /path/to/dedup_index.json

Outputs a per-case report and a JSONL manifest of the NEW authored documents
(content not yet in the corpus) for later staging/ingestion.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

RAG_DIR = Path(__file__).resolve().parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from build_dedup_index import text_sha256  # noqa: E402
from ingest_runner import extract_text  # noqa: E402

DEFAULT_CLI = "/home/jay/.codex/skills/api/scripts/jlawyer-cli"
AUTHORED_EXTENSIONS = {".odt", ".docx"}

# Non-substantive correspondence/admin among authored .odt: transmittals,
# scheduling, power-of-attorney, billing, file-access requests, notes.
_EXCLUDE_NAME = re.compile(
    r"anschreiben|termin|bitte|einladung|vollmacht|\bpkh\b|\bpka\b|rechnung|"
    r"zahlung|notiz|empfangsbek|^eb[_.\d]|übersend|uebersend|anforderung|"
    r"wiedervorlage|merkblatt|datenschutz|\bkfa\b|\bkfb\b|kostenfest|akteneinsicht",
    re.IGNORECASE,
)


def _run_cli(cli: str, *args: str) -> str:
    result = subprocess.run(
        [cli, *args], capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"jlawyer-cli {args[0]} failed: {result.stderr.strip()[:200]}")
    return result.stdout


def list_documents(cli: str, case_ref: str) -> list[dict[str, Any]]:
    return json.loads(_run_cli(cli, "documents", case_ref, "--json"))


def download_document(cli: str, doc_id: str, dest: Path) -> None:
    _run_cli(cli, "download", doc_id, "-o", str(dest))


def is_authored_candidate(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix not in AUTHORED_EXTENSIONS:
        return False
    return not _EXCLUDE_NAME.search(name)


def main() -> int:
    parser = argparse.ArgumentParser(description="j-lawyer RAG top-up dedup connector")
    parser.add_argument("--cases", help="Comma-separated case references, e.g. 089/26,300/24")
    parser.add_argument("--dedup-index", type=Path, required=True)
    parser.add_argument("--jlawyer-cli", default=DEFAULT_CLI)
    parser.add_argument("--out", type=Path, default=RAG_DIR / "data" / "jlawyer_new.jsonl")
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    if not args.cases:
        print("ERROR: provide --cases (full --all-cases discovery is a later step)", file=sys.stderr)
        return 2

    index = json.loads(args.dedup_index.read_text())
    known_text = set(index["text_sha256"])
    known_bytes = set(index["byte_sha256"])
    print(f"Dedup index: {index['counts']['unique_text_hashes']} text hashes from {index['manifest']}\n")

    case_refs = [c.strip() for c in args.cases.split(",") if c.strip()]
    new_records: list[dict[str, Any]] = []
    totals = {"candidates": 0, "new": 0, "duplicate": 0, "too_short": 0, "errors": 0}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for case_ref in case_refs:
            try:
                docs = list_documents(args.jlawyer_cli, case_ref)
            except Exception as exc:
                print(f"[{case_ref}] list failed: {exc}")
                totals["errors"] += 1
                continue
            candidates = [d for d in docs if is_authored_candidate(d["name"])]
            print(f"=== {case_ref}: {len(docs)} docs, {len(candidates)} authored candidates ===")
            for doc in candidates:
                totals["candidates"] += 1
                name = doc["name"]
                dest = tmp_dir / f"{doc['id']}{Path(name).suffix.lower()}"
                try:
                    download_document(args.jlawyer_cli, doc["id"], dest)
                    text = extract_text(dest)
                except Exception as exc:
                    print(f"  ERR   {name} — {exc}")
                    totals["errors"] += 1
                    continue
                finally:
                    dest.unlink(missing_ok=True)
                if len(text) < args.min_chars:
                    print(f"  SHORT {name} ({len(text)}c)")
                    totals["too_short"] += 1
                    continue
                th = text_sha256(text)
                if th in known_text:
                    print(f"  DUP   {name}")
                    totals["duplicate"] += 1
                    continue
                print(f"  NEW   {name} ({len(text)}c)")
                totals["new"] += 1
                new_records.append(
                    {
                        "source_system": "jlawyer",
                        "jlawyer_case_ref": case_ref,
                        "jlawyer_case_id": doc.get("caseId"),
                        "jlawyer_doc_id": doc["id"],
                        "name": name,
                        "size_bytes": doc.get("size"),
                        "creation_date": doc.get("creationDate"),
                        "text_sha256": th,
                    }
                )
                known_text.add(th)  # dedup within this run too

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in new_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"\nCandidates {totals['candidates']} | NEW {totals['new']} | "
        f"duplicate {totals['duplicate']} | short {totals['too_short']} | errors {totals['errors']}"
    )
    print(f"New-document manifest: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
