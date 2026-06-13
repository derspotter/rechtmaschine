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

# --- Boilerplate stripping (pre-anonymization) -----------------------------
# Kanzlei Schriftsätze carry a recurring letterhead/footer/recipient block that
# PyMuPDF extracts inline (as a repeating page header/footer). It is near
# identical across thousands of documents, so embedding it makes every doc's
# edge chunks look alike and crowds out the actual argumentation. Stripping it
# before anonymization also cuts the firm's own data (it never reaches Qwen) and
# reduces Qwen's token load. The Rubrum is deliberately kept so the anonymizer
# still sees the client name once and can resolve [PERSON] consistently.
#
# These structural anchors never occur in legal body text, so dropping any line
# that matches is safe regardless of client.
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*\d{1,2}\s*/\s*\d{1,2}\s*$"),                  # page numbers "1/3"
    re.compile(r"^\s*RECHTSANWALT\s*$"),
    re.compile(r"^\s*Rechtsanwalt\s*$"),
    re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE),     # email
    re.compile(r"https?://|www\.", re.IGNORECASE),                 # url
    re.compile(r"\+49[\s\d]{5,}"),                                 # phone
    re.compile(r"^\s*(?:Fax|Tel\.?|Telefon|Mobil)\s*:?\s*$"),     # contact labels
    re.compile(r"^\s*Bankverbindung\s*:", re.IGNORECASE),
    re.compile(r"^\s*Steuernummer\b", re.IGNORECASE),
    re.compile(r"^\s*LG[-\s]?Fach\s*:", re.IGNORECASE),
    re.compile(r"^\s*Mein\s+Zeichen\s*$", re.IGNORECASE),
    re.compile(r"^\s*Bitte\s+immer\s+angeben\s*$", re.IGNORECASE),
    re.compile(r"^\s*RA\s+.+,.+,"),                                # sender "RA Name, Addr, ..."
    re.compile(r"^\s*An\s+(?:das|die)\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\.\s*Kammer\s*-\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:vorab\s+)?per\s+(?:beA|EGVP|Telefax|Fax|Post)\b", re.IGNORECASE),
    re.compile(r"^\s*\d{3}[a-z]?\s*/\s*\d{2}(?:\s+[A-Z])?\s*$"),   # standalone Aktenzeichen value
]
# Firm name/address anchors: gated by line length so a long body sentence that
# happens to mention the firm (e.g. a "Kanzlei Keienborg" client case) is kept.
_FIRM_ANCHORS = re.compile(
    r"keienborg|friedrich[-\s]?ebert[-\s]?str|\b40210\s+d[uü]sseldorf\b|christian\s+schotte",
    re.IGNORECASE,
)

# Rubrum / case-caption block: the party designations and standard defendant
# boilerplate are byte-identical across asylum cases (similarity noise), and the
# "X ./. Y" caption is a known anonymization blind spot (the client name can
# survive there). All patterns are anchored to forms that do not occur in legal
# body prose: the "./." caption shorthand, standalone role designations, the
# lowercase Rubrum continuation lines, and the fixed defendant phrasing. The
# substantive "wegen ..." subject line is deliberately not matched.
_RUBRUM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\s\./\.\s"),                                          # caption "X ./. Y"
    re.compile(r"^\s*In (?:dem|der)\b.{0,50}\b(?:Verfahren|Rechtsstreit|Streitsache)\b", re.IGNORECASE),
    # standalone role designations (capitalized German nouns, line on its own)
    re.compile(r"^\s*(?:-\s*)?(?:Kläger(?:s|in)?|Antragsteller(?:s|in)?|Beklagte[rn]?|Antragsgegner(?:in)?)(?:\s+und\s+\w+)?\s*,?\s*(?:-\s*)?$"),
    re.compile(r"^\s*gegen\s*$"),
    re.compile(r"^\s*die Bundesrepublik Deutschland\b"),              # lowercase "die": Rubrum, not a body sentence
    re.compile(r"vertreten durch den (?:Bundesminister|Präsidenten des Bundesamtes)"),
    re.compile(r"Prozessbevollmächtigt"),
    # the party line itself, identifiable post-anonymization by its markers
    re.compile(r"^\s*(?:des|der|den)\s+\[PERSON\]"),
    re.compile(r"wohnhaft.{0,20}\[ADRESSE\]"),
]


def _filter_lines(text: str, patterns: list[re.Pattern[str]], firm_gate: bool) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if any(p.search(line) for p in patterns):
            continue
        if firm_gate and len(line) <= 70 and _FIRM_ANCHORS.search(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def strip_boilerplate(text: str) -> str:
    """Pre-anonymization: drop letterhead/footer + Rubrum/caption from raw text."""
    return _filter_lines(text, _BOILERPLATE_PATTERNS + _RUBRUM_PATTERNS, firm_gate=True)


# Lines that consist solely of anonymization markers (and punctuation) are the
# residue of address/recipient/signature blocks — no body text, pure noise.
_MARKER_ONLY_LINE = re.compile(r"^\s*(?:\[[A-ZÄÖÜ][A-ZÄÖÜ \-]*\]\s*)+[.,;:]?\s*$")


def clean_anonymized(text: str) -> str:
    """Post-anonymization: re-apply Rubrum/caption strip (now marker-aware) and
    drop marker-only lines that survived because the raw form differed."""
    text = _filter_lines(text, _RUBRUM_PATTERNS, firm_gate=False)
    return "\n".join(
        line for line in text.splitlines() if not _MARKER_ONLY_LINE.match(line)
    )


# Deterministic scrub of residue the LLM anonymizer leaves behind: the firm's
# own Aktenzeichen and letterhead boilerplate (bank details, tax number, phone
# numbers). A safety net behind strip_boilerplate. Order matters: strip the
# structured blocks before the generic NNN/YY case-number pattern.
_SCRUB_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^.*Bankverbindung:.*$", re.MULTILINE | re.IGNORECASE), "[BANKVERBINDUNG]"),
    (re.compile(r"\bIBAN:?\s*[A-Z]{2}\d{2}(?:\s?\d){10,30}\b", re.IGNORECASE), "[IBAN]"),
    (re.compile(r"\bBIC:?\s*[A-Z0-9]{8,11}\b", re.IGNORECASE), "[BIC]"),
    (re.compile(r"\bSteuernummer\s+[\d/]+(?:\s+Finanzamt[^\n]*)?", re.IGNORECASE), "[STEUERNUMMER]"),
    (re.compile(r"\bLG-Fach:?\s*\d+", re.IGNORECASE), "[LG-FACH]"),
    (re.compile(r"\+49[ \t\d]{6,}"), "[TEL]"),
    # NNN/YY and NNNa/YY (sub-case letter), optional trailing chamber letter.
    (re.compile(r"\b\d{3}[a-z]?/\d{2}(?:\s+[A-Z])?\b"), "[AKTENZEICHEN]"),
]


def scrub_residual(text: str) -> str:
    for pattern, replacement in _SCRUB_RULES:
        text = pattern.sub(replacement, text)
    return text


# --- Content hashing (shared with build_dedup_index / jlawyer_topup) --------
_HASH_WHITESPACE = re.compile(r"\s+")
_HASH_HYPHEN_BREAK = re.compile(r"(\w)[­-]\s*\n?\s*(\w)")


def normalize_for_hash(text: str) -> str:
    """Canonicalize so an ODT and its PDF render hash identically."""
    text = text.replace("­", "-")  # soft hyphen -> regular, then de-hyphenate
    text = _HASH_HYPHEN_BREAK.sub(r"\1\2", text)
    return _HASH_WHITESPACE.sub(" ", text).strip().lower()


def content_sha256(text: str) -> str:
    """Hash the substantive body (boilerplate-stripped, normalized)."""
    return hashlib.sha256(
        normalize_for_hash(strip_boilerplate(text)).encode("utf-8")
    ).hexdigest()


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
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap documents (mixed PDF/ODT sample for tests). Default: all.",
    )
    parser.add_argument("--collection", default="kanzlei")
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument(
        "--rag-url", default=os.getenv("RAG_API_URL", f"http://127.0.0.1:{os.getenv('RAG_API_PORT', '8090')}")
    )
    parser.add_argument("--anon-url", default=os.getenv("DESKTOP_QWEN_URL", "http://desktop:8004"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Extract + dedup only (no anonymization/upsert, no GPU). Reports the "
        "real ingest count, content-duplicates, short/OCR-needed, and errors.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip documents already recorded in the collection's done-log.",
    )
    args = parser.parse_args()

    anon_key = os.getenv("ANONYMIZATION_API_KEY", "")
    rag_key = os.getenv("RAG_SERVICE_API_KEY", "")
    manifest_path = args.manifest or _latest_manifest(args.import_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (DEFAULT_OUT_BASE / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in manifest_path.open() if l.strip()]
    if args.limit:
        # Mixed sample (PDF + ODT) for test runs; full run processes everything.
        pdfs = [i for i in items if i["extension"] == "pdf"]
        others = [i for i in items if i["extension"] != "pdf"]
        to_process: list[dict[str, Any]] = []
        while (pdfs or others) and len(to_process) < args.limit:
            if pdfs:
                to_process.append(pdfs.pop(0))
            if others and len(to_process) < args.limit:
                to_process.append(others.pop(0))
    else:
        to_process = items

    # Done-log enables resume and persists content hashes across runs so a
    # content-duplicate split by a crash is still caught.
    done_log = DEFAULT_OUT_BASE / f"{args.collection}.donelog.jsonl"
    done: set[str] = set()
    seen_content: set[str] = set()
    if args.resume and done_log.exists():
        for line in done_log.open():
            if line.strip():
                rec = json.loads(line)
                done.add(rec["sha16"])
                seen_content.add(rec["content"])

    print(
        f"Manifest: {manifest_path.name} ({len(items)} items), processing {len(to_process)}"
        + (f", resume skips {len(done)} done" if args.resume else "")
        + (" [DRY RUN]" if args.dry_run else "")
    )
    print(f"RAG: {args.rag_url} collection={args.collection} | anon: {args.anon_url}")
    print(f"Inspection dir: {out_dir}\n")

    ingested = skipped = failed = chunk_total = deduped = resumed = 0
    log_handle = None if args.dry_run else done_log.open("a", encoding="utf-8")
    with httpx.Client() as client:
        for item in to_process:
            rel = item["source_rel_path"]
            sha16 = item["sha256"][:16]
            if sha16 in done:
                resumed += 1
                continue
            token = (item.get("doc_token") or "").lower()
            role = ROLE_MAP.get(token, "Schriftsatz")
            court = COURT_MAP.get((item.get("court_token") or "").lower())
            case_ref = case_ref_from_path(rel)
            # Hash the case ref: chunks from one case can still be grouped/filtered
            # by case_hash, but the file number never lands in the store. Tracing
            # back goes via sha256 against the manifest on desktop.
            case_hash = hashlib.sha256(case_ref.encode()).hexdigest()[:12] if case_ref else None
            date = document_date(item.get("date_prefix"))
            label = f"{sha16} {item['extension']:4s} {role}"

            try:
                source = args.import_root / item["staged_rel_path"]
                text = extract_text(source)
                if len(text) < args.min_chars:
                    print(f"  SKIP  {label} — only {len(text)} chars (needs OCR)")
                    skipped += 1
                    continue

                stripped = strip_boilerplate(text)
                content_h = hashlib.sha256(
                    normalize_for_hash(stripped).encode("utf-8")
                ).hexdigest()
                if content_h in seen_content:
                    print(f"  DUP   {label} (content)")
                    deduped += 1
                    continue

                if args.dry_run:
                    seen_content.add(content_h)
                    ingested += 1
                    print(f"  OK*   {label} — {len(text)}->{len(stripped)} chars (dry-run)")
                    continue

                anonymized = clean_anonymized(
                    scrub_residual(anonymize(client, args.anon_url, anon_key, stripped, role))
                )
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
                    "case_hash": case_hash,
                    "document_date": date,
                    "extension": item["extension"],
                    "language": "de",
                }
                provenance = [
                    f"sha256:{sha16}",
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
                seen_content.add(content_h)
                done.add(sha16)
                log_handle.write(json.dumps({"sha16": sha16, "content": content_h}) + "\n")
                log_handle.flush()
                ingested += 1
                print(
                    f"  OK    {label} — {len(text)}->{len(stripped)} chars "
                    f"-> {upserted} chunks"
                )
            except Exception as exc:
                failed += 1
                print(f"  FAIL  {label} — {exc}")
    if log_handle is not None:
        log_handle.close()

    verb = "would ingest" if args.dry_run else "ingested"
    print(
        f"\n{verb} {ingested}, content-dupes {deduped}, short/OCR {skipped}, "
        f"failed {failed}"
        + (f", resumed-skip {resumed}" if args.resume else "")
        + ("" if args.dry_run else f"; {chunk_total} chunks upserted.")
    )
    if not args.dry_run:
        print(f"Inspect anonymized texts in {out_dir}; done-log: {done_log}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
