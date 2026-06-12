#!/usr/bin/env python
"""
Minimal RAG ingestion runner (v0) for the Debian machine.

Walks a desktop-export manifest, and per document: extracts text (PyMuPDF /
ODF / DOCX), anonymizes it via the desktop Qwen service, chunks it with an
anonymized context header, and upserts the chunks into the local RAG API
(which embeds server-side via TEI).

Anonymized texts are written to an inspection directory; review them for
leaks before trusting a larger run (see the validation plan in
docs/rag-three-machine-ingestion-plan.md).

Usage (on debian):

    ANONYMIZATION_API_KEY=... .venv-rag/bin/python rag/ingest_runner.py --limit 10

Skips (no OCR in v0): documents whose extracted text is shorter than
--min-chars. Deterministic chunk ids make re-runs idempotent upserts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from defusedxml import ElementTree as ET

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


RAG_DIR = Path(__file__).resolve().parent
DEFAULT_IMPORT_ROOT = RAG_DIR / "data" / "imports" / "desktop-export"
DEFAULT_OUT_BASE = RAG_DIR / "data" / "ingest_runs"

ROLE_MAP = {
    "klage": "Klage",
    "uklage": "Untätigkeitsklage",
    "azb": "Antrag auf Zulassung der Berufung",
    "bza": "Begründung des Zulassungsantrags",
    "bgrdg": "Begründung",
    "begruendung": "Begründung",
    "erg": "Ergänzender Schriftsatz",
    "stellungnahme": "Stellungnahme",
    "beschwerde": "Beschwerde",
    "eilantrag": "Eilantrag",
    "80v": "Eilantrag",
    "be": "Erwiderung",
    "schriftsatz": "Schriftsatz",
    "ne": "Nichtzulassungsbeschwerde",
    "eb": "Erklärung",
}

COURT_MAP = {"vg": "VG", "ovg": "OVG", "bverwg": "BVerwG", "ag": "AG", "lg": "LG", "sg": "SG"}


def _load_env_file() -> None:
    env_path = RAG_DIR / ".env.debian"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _latest_manifest(import_root: Path) -> Path:
    manifests = sorted(
        (import_root / "manifests").glob("nextcloud_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"No nextcloud_*.jsonl under {import_root}/manifests")
    return manifests[0]


def _extract_pdf(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed")
    doc = fitz.open(path)
    try:
        return "\n\n".join((page.get_text() or "").strip() for page in doc)
    finally:
        doc.close()


def _extract_zip_xml(path: Path, member: str, para_tag: str) -> str:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read(member))
    paragraphs = []
    for elem in root.iter():
        if elem.tag.endswith("}" + para_tag):
            text = "".join(elem.itertext()).strip()
            if text:
                paragraphs.append(text)
    return "\n\n".join(paragraphs)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".odt":
        return _extract_zip_xml(path, "content.xml", "p")
    if ext == ".docx":
        return _extract_zip_xml(path, "word/document.xml", "p")
    raise ValueError(f"Unsupported extension: {ext}")


def case_ref_from_path(rel_path: str) -> Optional[str]:
    # "21/003 BADSHAH ua vs BRD/..." -> "003/21" (digits only, no names)
    m = re.match(r"^(\d{2})/(\d{3})\b", rel_path)
    return f"{m.group(2)}/{m.group(1)}" if m else None


def document_date(date_prefix: Optional[str]) -> Optional[str]:
    if not date_prefix or len(date_prefix) != 6 or not date_prefix.isdigit():
        return None
    yy, mm, dd = date_prefix[:2], date_prefix[2:4], date_prefix[4:6]
    if not ("01" <= mm <= "12" and "01" <= dd <= "31"):
        return None
    return f"20{yy}-{mm}-{dd}"


def chunk_text(text: str, target_chars: int = 1800, max_chars: int = 2400) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # Hard-split single paragraphs that alone exceed the budget.
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > target_chars and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    # Merge a trailing fragment into its predecessor.
    if len(chunks) >= 2 and len(chunks[-1]) < 200:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    return chunks


def anonymize(client: httpx.Client, url: str, api_key: str, text: str, role: str) -> str:
    response = client.post(
        f"{url}/anonymize",
        json={"text": text, "document_type": role},
        headers={"X-API-Key": api_key} if api_key else {},
        timeout=600.0,
    )
    response.raise_for_status()
    data = response.json()
    anonymized = data.get("anonymized_text")
    if not anonymized:
        raise RuntimeError(f"Anonymization returned no text (keys: {list(data)})")
    return anonymized


def upsert_chunks(
    client: httpx.Client,
    rag_url: str,
    rag_key: str,
    collection: str,
    chunks: list[dict[str, Any]],
) -> int:
    total = 0
    for start in range(0, len(chunks), 16):
        batch = chunks[start : start + 16]
        response = client.post(
            f"{rag_url}/v1/rag/chunks/upsert",
            json={"collection": collection, "chunks": batch},
            headers={"X-API-Key": rag_key} if rag_key else {},
            timeout=300.0,
        )
        response.raise_for_status()
        total += int(response.json().get("upserted", 0))
    return total


def main() -> int:
    _load_env_file()
    parser = argparse.ArgumentParser(description="Minimal RAG ingestion runner (v0)")
    parser.add_argument("--import-root", type=Path, default=DEFAULT_IMPORT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--collection", default="kanzlei")
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument(
        "--rag-url", default=os.getenv("RAG_API_URL", f"http://127.0.0.1:{os.getenv('RAG_API_PORT', '8090')}")
    )
    parser.add_argument("--anon-url", default=os.getenv("DESKTOP_QWEN_URL", "http://desktop:8004"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    anon_key = os.getenv("ANONYMIZATION_API_KEY", "")
    rag_key = os.getenv("RAG_SERVICE_API_KEY", "")
    manifest_path = args.manifest or _latest_manifest(args.import_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT_BASE / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in manifest_path.open() if l.strip()]
    # Mixed sample: alternate extensions so the test covers PDF and ODT paths.
    pdfs = [i for i in items if i["extension"] == "pdf"]
    others = [i for i in items if i["extension"] != "pdf"]
    sample: list[dict[str, Any]] = []
    while (pdfs or others) and len(sample) < args.limit:
        if pdfs:
            sample.append(pdfs.pop(0))
        if others and len(sample) < args.limit:
            sample.append(others.pop(0))

    print(f"Manifest: {manifest_path.name} ({len(items)} items), ingesting {len(sample)}")
    print(f"RAG: {args.rag_url} collection={args.collection} | anon: {args.anon_url}")
    print(f"Inspection dir: {out_dir}\n")

    ingested = skipped = failed = chunk_total = 0
    with httpx.Client() as client:
        for item in sample:
            rel = item["source_rel_path"]
            sha16 = item["sha256"][:16]
            token = (item.get("doc_token") or "").lower()
            role = ROLE_MAP.get(token, "Schriftsatz")
            court = COURT_MAP.get((item.get("court_token") or "").lower())
            case_ref = case_ref_from_path(rel)
            date = document_date(item.get("date_prefix"))
            label = f"{sha16} {item['extension']:4s} {role}"

            try:
                source = args.import_root / item["staged_rel_path"]
                text = extract_text(source)
                if len(text) < args.min_chars:
                    print(f"  SKIP  {label} — only {len(text)} chars (needs OCR)")
                    skipped += 1
                    continue

                anonymized = anonymize(client, args.anon_url, anon_key, text, role)
                (out_dir / f"{sha16}.txt").write_text(anonymized, encoding="utf-8")

                header_bits = ["Kanzlei-Schriftsatz", role]
                if court:
                    header_bits.append(court)
                if date:
                    header_bits.append(date)
                context_header = " | ".join(header_bits)

                metadata = {
                    "source_system": "nextcloud",
                    "document_role": role,
                    "court": court,
                    "case_ref": case_ref,
                    "document_date": date,
                    "extension": item["extension"],
                    "language": "de",
                }
                provenance = [
                    f"sha256:{sha16}",
                    f"case:{case_ref}",
                    f"manifest:{item.get('manifest_run_id')}",
                ]
                payload = [
                    {
                        "chunk_id": f"nc-{sha16}-{idx:03d}",
                        "text": chunk,
                        "context_header": context_header,
                        "metadata": {**metadata, "chunk_index": idx},
                        "provenance": provenance,
                    }
                    for idx, chunk in enumerate(chunk_text(anonymized))
                ]
                upserted = upsert_chunks(client, args.rag_url, rag_key, args.collection, payload)
                chunk_total += upserted
                ingested += 1
                print(f"  OK    {label} — {len(text)} chars -> {upserted} chunks")
            except Exception as exc:
                failed += 1
                print(f"  FAIL  {label} — {exc}")

    print(f"\nIngested {ingested}, skipped {skipped}, failed {failed}; {chunk_total} chunks upserted.")
    print(f"Inspect anonymized texts in {out_dir} before scaling up.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
