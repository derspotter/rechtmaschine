#!/usr/bin/env python
"""
Ingestion v0: read-only, manifest-driven parsing to JSONL.

Input:
- Manifest JSONL produced by export_manifest.py (INCLUDE-only list)

Output (all under rag/data/):
- staging/: transient files (e.g., ODT->PDF conversion outputs)
- ingested/: parsed text + metadata (JSONL per run)

Safety:
- Never writes to the corpus directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

import httpx
import fitz  # PyMuPDF

RAG_DIR = Path(__file__).resolve().parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from filter_corpus import guard_corpus_writes  # noqa: E402


DEFAULT_CORPUS_DIR = Path("/home/jayjag/Kanzlei/kanzlei")
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "data"
    / "manifests"
    / "manifest_20260206_221904_include.jsonl"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "ingested"
DEFAULT_STAGING_DIR = Path(__file__).resolve().parent / "data" / "staging"


SERVICE_MANAGER_URL = os.getenv("SERVICE_MANAGER_URL", "http://127.0.0.1:8004")
OCR_ENDPOINT = f"{SERVICE_MANAGER_URL.rstrip('/')}/ocr"
ANON_ENDPOINT = f"{SERVICE_MANAGER_URL.rstrip('/')}/anonymize"
ANON_API_KEY = os.getenv("ANONYMIZATION_API_KEY")


@dataclass(frozen=True)
class IngestResult:
    source_rel_path: str
    source_abs_path: str
    ok: bool
    used_ocr: bool
    extracted_chars: int
    sha256: str
    error: Optional[str]
    metadata: dict[str, Any]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_rel_path": self.source_rel_path,
            "source_abs_path": self.source_abs_path,
            "ok": self.ok,
            "used_ocr": self.used_ocr,
            "extracted_chars": self.extracted_chars,
            "sha256": self.sha256,
            "error": self.error,
            "metadata": self.metadata,
            "text": self.text,
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _read_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def _extract_pdf_text(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    doc = fitz.open(pdf_path)
    try:
        pages = len(doc)
        take = pages if max_pages is None else min(pages, max_pages)
        parts: list[str] = []
        for idx in range(take):
            parts.append(doc[idx].get_text() or "")
        return "\n".join(parts).strip()
    finally:
        doc.close()


def _convert_odt_to_pdf(odt_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = out_dir / "_lo_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = out_dir / "_xdg_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # LibreOffice can fail in headless environments if it cannot access dconf
    # or the user's runtime/config dirs. Force an isolated, writable profile.
    env = os.environ.copy()
    env["HOME"] = str(profile_dir)
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env["XDG_CACHE_HOME"] = str(profile_dir / "cache")
    env["XDG_CONFIG_HOME"] = str(profile_dir / "config")
    env["XDG_DATA_HOME"] = str(profile_dir / "data")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    # Avoid any GUI dependencies.
    env.setdefault("SAL_USE_VCLPLUGIN", "svp")

    cmd = [
        "soffice",
        "--headless",
        "--nologo",
        "--nodefault",
        "--norestore",
        "--nolockcheck",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        f"-env:UserInstallation=file://{profile_dir.as_posix()}",
        str(odt_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        msg = stderr or stdout or f"exit={result.returncode}"
        raise RuntimeError(f"LibreOffice conversion failed: {msg}")

    pdf_path = out_dir / (odt_path.stem + ".pdf")
    if not pdf_path.exists():
        # LibreOffice sometimes alters names; best-effort fallback
        candidates = list(out_dir.glob("*.pdf"))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError("Converted PDF not found in output directory")
    return pdf_path


def _extract_odt_text(odt_path: Path) -> str:
    """
    Fallback ODT text extraction without LibreOffice.
    Reads content.xml and returns concatenated text.
    """
    with zipfile.ZipFile(odt_path, "r") as zf:
        with zf.open("content.xml") as f:
            xml_bytes = f.read()

    root = ET.fromstring(xml_bytes)
    text = " ".join(chunk.strip() for chunk in root.itertext() if chunk and chunk.strip())
    # Normalize whitespace a bit
    return " ".join(text.split()).strip()


def _ocr_pdf_via_service_manager(pdf_path: Path) -> str:
    with pdf_path.open("rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        response = httpx.post(OCR_ENDPOINT, files=files, timeout=600.0)
    response.raise_for_status()
    data = response.json()
    text = data.get("full_text") or ""
    return str(text).strip()


def _anonymize_text_via_service_manager(text: str, document_type: str) -> str:
    payload = {"text": text, "document_type": document_type}
    headers = {"X-API-Key": ANON_API_KEY} if ANON_API_KEY else {}
    response = httpx.post(ANON_ENDPOINT, json=payload, headers=headers, timeout=600.0)
    response.raise_for_status()
    data = response.json()
    anon_text = data.get("anonymized_text") or ""
    return str(anon_text).strip()


def _ingest_one(
    corpus_dir: Path,
    item: dict[str, Any],
    staging_dir: Path,
    min_text_chars: int,
) -> IngestResult:
    rel_path = str(item.get("source_rel_path", ""))
    abs_path = Path(item.get("source_abs_path") or (corpus_dir / rel_path)).resolve()
    extension = str(item.get("extension") or abs_path.suffix.lower().lstrip("."))

    metadata = {
        "extension": extension,
        "case_year": item.get("case_year"),
        "case_folder": item.get("case_folder"),
        "filename": item.get("filename") or abs_path.name,
        "signal_codes": item.get("signal_codes") or [],
        "date_prefix": item.get("date_prefix"),
        "court_token": item.get("court_token"),
        "doc_token": item.get("doc_token"),
        "letterhead_confidence": item.get("letterhead_confidence"),
    }

    try:
        if not abs_path.exists():
            raise FileNotFoundError("Source file does not exist")

        used_ocr = False
        text = ""
        document_type = "Schriftsatz"

        if extension == "pdf":
            text = _extract_pdf_text(abs_path)
            if len(text) < min_text_chars:
                used_ocr = True
                text = _ocr_pdf_via_service_manager(abs_path)
        elif extension == "odt":
            # Prefer LibreOffice conversion when it works, but fall back to pure ODT extraction.
            try:
                out_dir = staging_dir / "odt_to_pdf" / rel_path.replace("/", "__")
                pdf_path = _convert_odt_to_pdf(abs_path, out_dir)
                metadata["converted_pdf"] = str(pdf_path)
                text = _extract_pdf_text(pdf_path)
                if len(text) < min_text_chars:
                    used_ocr = True
                    text = _ocr_pdf_via_service_manager(pdf_path)
            except Exception as exc:
                metadata["odt_fallback"] = "xml_text"
                metadata["odt_conversion_error"] = str(exc)
                text = _extract_odt_text(abs_path)
        else:
            raise ValueError(f"Unsupported extension for ingestion v0: {extension}")

        text = text.strip()
        anon_text = _anonymize_text_via_service_manager(text, document_type=document_type)
        metadata["anonymized"] = True
        metadata["anonymizer"] = "service_manager"
        return IngestResult(
            source_rel_path=rel_path,
            source_abs_path=str(abs_path),
            ok=True,
            used_ocr=used_ocr,
            extracted_chars=len(anon_text),
            sha256=_sha256_text(anon_text),
            error=None,
            metadata=metadata,
            text=anon_text,
        )
    except Exception as exc:
        return IngestResult(
            source_rel_path=rel_path,
            source_abs_path=str(abs_path),
            ok=False,
            used_ocr=False,
            extracted_chars=0,
            sha256=_sha256_text(""),
            error=str(exc),
            metadata=metadata,
            text="",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingestion v0 from manifest to JSONL.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-text-chars", type=int, default=400)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path: Path = args.manifest
    corpus_dir: Path = args.corpus_dir
    out_dir: Path = args.out_dir
    staging_dir: Path = args.staging_dir

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 2
    if not corpus_dir.exists():
        print(f"[ERROR] Corpus not found: {corpus_dir}")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ingested_{run_id}.jsonl"

    items = _read_manifest(manifest_path, limit=args.limit)
    if not items:
        print("[WARN] Manifest is empty.")
        return 0

    print("=" * 80)
    print("RAG INGEST V0")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Corpus:   {corpus_dir}")
    print(f"Limit:    {args.limit}")
    print(f"Output:   {out_path}")
    print()

    ok = 0
    fail = 0
    used_ocr = 0

    with guard_corpus_writes(corpus_dir):
        with out_path.open("w", encoding="utf-8") as f:
            for idx, item in enumerate(items, 1):
                result = _ingest_one(
                    corpus_dir=corpus_dir,
                    item=item,
                    staging_dir=staging_dir,
                    min_text_chars=args.min_text_chars,
                )
                if result.ok:
                    ok += 1
                else:
                    fail += 1
                if result.used_ocr:
                    used_ocr += 1

                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                print(
                    f"{idx}/{len(items)}: {result.source_rel_path} "
                    f"ok={result.ok} chars={result.extracted_chars} ocr={result.used_ocr}"
                )

    print()
    print(f"OK:       {ok}")
    print(f"FAILED:   {fail}")
    print(f"USED OCR: {used_ocr}")
    print(f"Output:   {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
