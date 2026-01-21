#!/usr/bin/env python3
"""
Ad-hoc evaluation for Flair anonymization on OCR text files or PDFs.

Usage:
    python tests/test_anonymization_flair.py [path/to/file1] [path/to/file2 ...]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_PATH = REPO_ROOT / "test_files" / "ocred" / "ocr_bescheid_clean.md"

LEGAL_TOKENS = [
    "egmr",
    "urteil",
    "entscheidung",
    "beschluss",
    "bverwg",
    "vgh",
    "vg ",
    "vg,",
    "rn.",
    "abs.",
    "asyl",
    "dublin",
    "grch",
    "vo",
]

AUTHORITY_LINE_PATTERN = re.compile(
    r"\b(bundesamt|bamf|verwaltungsgericht|oberverwaltungsgericht|amtsgericht|"
    r"landgericht|gericht|ausl[aä]nderbeh[öo]rde|bundesbank|bundeskasse|"
    r"bundesagentur|jobcenter|polizei|staatsanwaltschaft|ministerium|regierung|"
    r"beh[öo]rde|referat|zentrale|kanzlei|rechtsanwalt)\b",
    re.IGNORECASE,
)
AUTHORITY_LABEL_PATTERN = re.compile(
    r"\b(hausanschrift|postanschrift|briefanschrift|dienstsitz|zentrale)\b",
    re.IGNORECASE,
)
ID_CUE_PATTERN = re.compile(
    r"\b(Aktenzeichen|Gesch\.?\s*-?\s*zeichen|Geschäftszeichen|Geschaeftszeichen|"
    r"Geschaftszeichen|Gesch\.?\s*-?\s*Z\.?|AZR\s*-?\s*Nummer\(n\)?|AZR|"
    r"Ihr(?:e)?\s+Zeichen|Mein\s+Zeichen)\b",
    re.IGNORECASE,
)
AZR_CUE_PATTERN = re.compile(r"\bAZR\b|AZR\s*-?\s*Nummer", re.IGNORECASE)
ID_HYPHEN_PATTERN = re.compile(r"\b\d{6,}\s*-\s*\d{1,}\b")
ID_NUMBER_PATTERN = re.compile(r"\b\d{6,}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Flair anonymization against expected PII leaks.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(DEFAULT_DOC_PATH)],
        help=f"Paths to OCR text files or PDFs (default: {DEFAULT_DOC_PATH})",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU by masking CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--anonymize-phones",
        action="store_true",
        help="Enable phone redaction via ANONYMIZE_PHONES=1.",
    )
    parser.add_argument(
        "--include-phones",
        action="store_true",
        help="Include phone numbers in expected/leak checks.",
    )
    parser.add_argument(
        "--include-addresses",
        action="store_true",
        help="Include addresses in expected/leak checks.",
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Print extracted expected items before anonymization.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Max PDF pages to extract when input is a PDF (default: 50, 0 = all).",
    )
    return parser.parse_args()


def has_legal_token(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in LEGAL_TOKENS)


def is_authority_address_context(line: str, prev_line: str) -> bool:
    if not line and not prev_line:
        return False
    if AUTHORITY_LINE_PATTERN.search(line) or AUTHORITY_LINE_PATTERN.search(prev_line):
        return True
    if AUTHORITY_LABEL_PATTERN.search(line) or AUTHORITY_LABEL_PATTERN.search(prev_line):
        return True
    return False


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_expected(
    text: str, include_phones: bool, include_addresses: bool
) -> Dict[str, List[str]]:
    lines = [line.rstrip() for line in text.splitlines()]
    trimmed = [line.strip() for line in lines if line.strip()]

    names = set()
    name_trigger = re.compile(r"\bName\b", re.IGNORECASE)
    name_block = False

    for line in trimmed:
        if name_trigger.search(line) or "Vorname/NAME" in line:
            name_block = True
            continue
        if name_block and re.search(r"Geburtsdatum|Aktenzeichen|Anlagen", line, re.IGNORECASE):
            name_block = False

        if name_block:
            candidate = line.strip()
            if re.match(r"^\d{2}\.\d{2}\.\d{4}$", candidate):
                continue
            if re.search(r"\d", candidate):
                continue
            if has_legal_token(candidate):
                continue
            if re.match(
                r"^[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2}\s+[A-ZÄÖÜ]{2,}$",
                candidate,
            ):
                names.add(candidate)
            if re.match(
                r"^[A-ZÄÖÜ]{2,},\s*[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2}$",
                candidate,
            ):
                names.add(candidate)

    for line in trimmed:
        if not re.search(r"geb\.?\s*(am)?\s*\d{2}\.\d{2}\.\d{4}", line, re.IGNORECASE):
            continue
        for match in re.finditer(
            r"\b([A-ZÄÖÜ]{2,},\s*[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2})",
            line,
        ):
            candidate = match.group(1)
            if not has_legal_token(candidate):
                names.add(candidate)

    dobs = set()
    for line in trimmed:
        if re.search(r"\bgeb\.?\s*(am)?\b", line, re.IGNORECASE):
            dobs.update(re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", line))

    for index, line in enumerate(trimmed):
        if re.search(r"Geburtsdatum", line, re.IGNORECASE):
            for offset in range(index + 1, min(index + 12, len(trimmed))):
                candidate = trimmed[offset].strip()
                if not candidate:
                    break
                if re.search(r"\bName\b|Aktenzeichen|Anlagen", candidate, re.IGNORECASE):
                    break
                if re.match(r"^\d{2}\.\d{2}\.\d{4}$", candidate):
                    dobs.add(candidate)

    addresses = set()
    if include_addresses:
        street_regex = re.compile(
            r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)*\s*"
            r"(?:str\.|straße|strasse|allee|weg|platz|ring|gasse|damm|ufer)\s*\d+[a-zA-Z]?\b",
            re.IGNORECASE,
        )
        street_compact = re.compile(
            r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+str\.?\s*\d+[a-zA-Z]?\b",
            re.IGNORECASE,
        )
        postcode_regex = re.compile(
            r"\b\d{5}\s*[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*\b"
        )

        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            prev_line = lines[index - 1].strip() if index > 0 else ""
            if has_legal_token(line):
                continue
            if ";" in line:
                continue
            if len(line) > 80:
                continue
            if is_authority_address_context(line, prev_line):
                continue
            for match in street_regex.finditer(line):
                addresses.add(match.group(0))
            for match in street_compact.finditer(line):
                addresses.add(match.group(0))
            for match in postcode_regex.finditer(line):
                addresses.add(match.group(0))

    phones = set()
    if include_phones:
        for line in trimmed:
            if re.search(r"\bTel", line, re.IGNORECASE) or re.search(r"\+\d", line):
                for match in re.finditer(r"\+?\d[\d\s/.-]{6,}\d", line):
                    phones.add(match.group(0).strip())

    ids = set()
    for index, line in enumerate(trimmed):
        if not ID_CUE_PATTERN.search(line):
            continue
        window = " ".join(trimmed[index : index + 4])
        if AZR_CUE_PATTERN.search(window):
            ids.update(re.findall(r"\b\d{9,}\b", window))
        for match in ID_HYPHEN_PATTERN.finditer(window):
            ids.add(re.sub(r"\s+", "", match.group(0)))
        for match in ID_NUMBER_PATTERN.finditer(window):
            ids.add(match.group(0))

    return {
        "names": sorted(names),
        "dobs": sorted(dobs),
        "addresses": sorted(addresses),
        "phones": sorted(phones),
        "ids": sorted(ids),
    }


def leak_details(expected: Dict[str, List[str]], anonymized_text: str) -> Dict[str, List[str]]:
    normalized = normalize(anonymized_text)
    leaks: Dict[str, List[str]] = {}
    for key, values in expected.items():
        leaked = []
        for item in values:
            if normalize(item) in normalized:
                leaked.append(item)
        leaks[key] = leaked
    return leaks


def load_text_from_path(path: Path, max_pages: int) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Failed to import pymupdf (fitz): {exc}") from exc

        with fitz.open(path) as doc:
            total_pages = len(doc)
            pages_to_read = total_pages if max_pages == 0 else min(total_pages, max_pages)

            text_parts = []
            for i in range(pages_to_read):
                page = doc[i]
                text = page.get_text()
                if text:
                    text_parts.append(f"--- Page {i + 1} ---\n{text}")

        return "\n\n".join(text_parts)

    return path.read_text(encoding="utf-8")


def main() -> int:
    args = parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.anonymize_phones:
        os.environ["ANONYMIZE_PHONES"] = "1"

    anon_dir = REPO_ROOT / "anon"
    sys.path.append(str(anon_dir))

    try:
        from anonymization_service_flair import anonymize_with_flair
    except Exception as exc:
        print(f"[ERROR] Failed to import anonymization_service_flair: {exc}")
        return 1

    paths = [Path(path).expanduser() for path in args.paths]
    for doc_path in paths:
        if not doc_path.exists():
            print(f"[ERROR] File not found: {doc_path}")
            return 1

        print(f"\n=== {doc_path} ===")
        try:
            text = load_text_from_path(doc_path, args.max_pages)
        except Exception as exc:
            print(f"[ERROR] Failed to read {doc_path}: {exc}")
            return 1

        if not text or len(text.strip()) < 100:
            print("[WARN] Extracted text is empty or too short for evaluation.")
            continue

        expected = extract_expected(
            text,
            include_phones=args.include_phones,
            include_addresses=args.include_addresses,
        )
        counts = {key: len(value) for key, value in expected.items()}
        if not args.include_phones:
            counts["phones"] = 0
        if not args.include_addresses:
            counts["addresses"] = 0
        print("Expected counts:", counts)
        if not args.include_phones:
            print("Note: phones skipped (use --include-phones to include).")
        if not args.include_addresses:
            print("Note: addresses skipped (use --include-addresses to include).")

        if args.show_expected:
            for key, values in expected.items():
                if key == "phones" and not args.include_phones:
                    continue
                if key == "addresses" and not args.include_addresses:
                    continue
                print(f"\nExpected {key} ({len(values)}):")
                for value in values:
                    print(f"- {value}")

        start = time.time()
        flair_anonymized, _, _, _, _ = anonymize_with_flair(text)
        flair_elapsed = time.time() - start
        flair_leaks = leak_details(expected, flair_anonymized)

        print(f"\nNER+regex (Flair) time: {flair_elapsed:.1f}s")
        print("Missed items (still present):")
        for key in ["names", "dobs", "addresses", "phones", "ids"]:
            if key == "phones" and not args.include_phones:
                print("- phones: skipped")
                continue
            if key == "addresses" and not args.include_addresses:
                print("- addresses: skipped")
                continue
            values = flair_leaks.get(key, [])
            print(f"- {key}: {len(values)}")
            for value in values:
                print(f"  * {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
