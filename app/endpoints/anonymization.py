import json
import os
import tempfile
import hashlib
import re
from datetime import datetime
from typing import Any, Optional
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

try:
    from presidio_analyzer import Pattern, PatternRecognizer
except ImportError:  # pragma: no cover - optional dependency
    Pattern = None
    PatternRecognizer = None

from shared import (
    AnonymizationResult,
    DocumentCategory,
    broadcast_documents_snapshot,
    ensure_service_manager_ready,
    limiter,
    load_document_text,
    store_document_text,
    ANONYMIZED_TEXT_DIR,
)
from auth import get_current_active_user
from database import get_db
from models import Document, User
from .ocr import extract_pdf_text, perform_ocr_on_file, check_pdf_needs_ocr
from anon.anonymization_service import (
    filter_non_person_group_labels,
    filter_non_person_organization_labels,
    augment_names_from_role_markers,
    augment_names_from_person_fields,
    apply_regex_replacements,
)

router = APIRouter()

GEMMA_MODEL = os.getenv("OLLAMA_MODEL_GEMMA", os.getenv("OLLAMA_MODEL", "gemma3:12b"))
QWEN_MODEL = os.getenv(
    "OLLAMA_MODEL_QWEN",
    os.getenv("OLLAMA_MODEL", "qwen3.5:9b-q5_k_m"),
)
ANONYMIZATION_ENGINE_DEFAULT = os.getenv(
    "ANONYMIZATION_ENGINE_DEFAULT", "flair_presidio"
).strip().lower()
SUPPORTED_ANONYMIZATION_ENGINES = {"gemma", "qwen_flair", "flair_presidio"}


def _entity_counts(entities: dict) -> dict[str, int]:
    return {
        str(key): len(values)
        for key, values in entities.items()
        if isinstance(values, list) and values
    }


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid int env {name}={raw!r}, using default={default}")
        return default
    if value <= 0:
        print(f"[WARN] Non-positive int env {name}={raw!r}, using default={default}")
        return default
    return value


def _optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_positive_int_env(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid int env {name}={raw!r}, ignoring")
        return None
    if value <= 0:
        return None
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"[WARN] Invalid float env {name}={raw!r}, using default={default}")
        return default
    return value


def _optional_float_env(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Invalid float env {name}={raw!r}, ignoring")
        return None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    print(f"[WARN] Invalid bool env {name}={raw!r}, using default={default}")
    return default


QWEN_NAME_REVIEW_ENABLED = _bool_env("ANONYMIZATION_USE_QWEN_NAME_REVIEW", True)


NAMES_EXTRACTION_PROMPT_PREFIX = """Extract PERSON names from this German legal document.
Return valid JSON only with exactly:
{"names":[]}

Rules:
- names: natural persons only (applicants, family members, officials, signers)
- return exact surface forms from the document only
- include full names, surname-only forms, and "SURNAME, Given" forms
- include abbreviated and hyphen forms only if they clearly refer to a person (e.g. "S. Quast", "A-Nabi")
- if text contains "Es erscheint Herr/Frau X, Y geb.", include the person names X and Y, not the surrounding words
- for family relation lines (Vater, Mutter, Ehemann, Ehefrau, Sohn, Tochter, Geschwister), include only actual names that appear next to the relation words
- include names near role/signature markers (Anhörender Entscheider, Sachbearbeiter, Unterzeichner, Im Auftrag, gez., Unterschrift), but return only the name, never the role word
- include a single-token name only when it is clearly a signer or person mention; otherwise prefer omitting uncertain single words
- never return role labels by themselves (e.g. "Mutter", "Sachbearbeiter", "Unterzeichner")
- never return organizations, courts, authorities, addresses, cities, countries, ethnicities, religions, document titles, legal citations, IDs, numbers, or page/footer text
- never return OCR/debug/schema garbage or strings containing many digits/non-name tokens
- if unsure whether something is a person name, omit it
- deduplicate exact duplicates

Document:
"""

ADDRESSES_EXTRACTION_PROMPT_PREFIX = """Extract address-like location information from this German legal document.
Return valid JSON only with exactly:
{"streets":[], "postal_codes":[], "cities":[]}

Rules:
- streets: any street + house number mentioned in address context
- postal_codes: any 5-digit postal code mentioned in address context
- cities: any city or location mentioned in address context
- include applicant, family, BAMF, court, authority, and correspondence addresses
- prioritize exact surface forms from the text
- deduplicate exact duplicates

Document:
"""

BIRTH_IDS_EXTRACTION_PROMPT_PREFIX = """Extract birth details and personal document IDs from this German legal document.
Return valid JSON only with exactly:
{"birth_dates":[], "birth_places":[], "azr_numbers":[], "aufenthaltsgestattung_ids":[], "case_numbers":[]}

Rules:
- birth_dates: full date strings for person birth data (e.g. DD.MM.YYYY, "geb. am ...")
- if only a birth year is given, include the full surrounding birth phrase (e.g. "1992 geboren")
- birth_places: city/place directly tied to explicit birth context only (e.g. "geboren in", "Geburtsort", "geb. in")
- do NOT copy ordinary cities or residence locations into birth_places
- azr_numbers: AZR numbers only when explicitly labeled "AZR" or unmistakably in AZR context
- do NOT copy Aktenzeichen, Az., BAMF file numbers, or case numbers into azr_numbers
- aufenthaltsgestattung_ids: IDs explicitly labeled as Aufenthaltsgestattung
- do NOT infer aufenthaltsgestattung_ids from fragments, case numbers, or unlabeled numeric strings
- case_numbers: personal/document IDs (e.g. Dolmetscher-Nr, D4S..., numeric id blocks)
- do NOT include court citations/references (ECLI, BVerwG/BVerfG/VG/OVG Az., §/Art. citations)
- if unsure which ID field a value belongs to, prefer case_numbers
- deduplicate exact duplicates

Document:
"""

EXTRACTION_ENTITY_KEYS = [
    "names",
    "birth_dates",
    "birth_places",
    "streets",
    "postal_codes",
    "cities",
    "azr_numbers",
    "aufenthaltsgestattung_ids",
    "case_numbers",
]

EXTRACTION_FIELD_SCHEMA = {
    "names": {"type": "array", "items": {"type": "string"}},
    "birth_dates": {"type": "array", "items": {"type": "string"}},
    "birth_places": {"type": "array", "items": {"type": "string"}},
    "streets": {"type": "array", "items": {"type": "string"}},
    "postal_codes": {"type": "array", "items": {"type": "string"}},
    "cities": {"type": "array", "items": {"type": "string"}},
    "azr_numbers": {"type": "array", "items": {"type": "string"}},
    "aufenthaltsgestattung_ids": {"type": "array", "items": {"type": "string"}},
    "case_numbers": {"type": "array", "items": {"type": "string"}},
}

EXTRACTION_STAGE_SPECS = [
    {
        "name": "names",
        "keys": ["names"],
        "prompt_prefix": NAMES_EXTRACTION_PROMPT_PREFIX,
    },
    {
        "name": "addresses",
        "keys": ["streets", "postal_codes", "cities"],
        "prompt_prefix": ADDRESSES_EXTRACTION_PROMPT_PREFIX,
    },
    {
        "name": "birth_ids",
        "keys": [
            "birth_dates",
            "birth_places",
            "azr_numbers",
            "aufenthaltsgestattung_ids",
            "case_numbers",
        ],
        "prompt_prefix": BIRTH_IDS_EXTRACTION_PROMPT_PREFIX,
    },
]

BIRTH_CONTEXT_PATTERN = re.compile(
    r"(?i)(geboren\s+am|geb\.?\s*am|geburtsdatum|jahrgang|geboren|geb\.(?=[\s,;:)]))"
)
BIRTH_YEAR_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:jahrgang|geboren)\s*:?\s*(?:19|20)\d{2}\b"
)
PLZ_CITY_CAPTURE_PATTERN = re.compile(
    r"\b(\d{5})[ \t]+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß]+(?:[ \t-]+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]+){0,2})\b"
)
AZR_LABEL_STRIP_PATTERN = re.compile(r"(?i)^\s*AZR(?:-Nummer|-Nr\.?)?\s*[:#-]?\s*")
AUFENTHALTSGESTATTUNG_LABEL_STRIP_PATTERN = re.compile(
    r"(?i)^\s*Aufenthaltsgestattung\s*[:#-]?\s*"
)
CASE_NUMBER_LABEL_STRIP_PATTERN = re.compile(
    r"(?i)^\s*(?:Az\.?|Aktenzeichen|Geschäftszeichen|Dolmetscher(?:-Nr\.?|nummer)?)\s*[:#-]?\s*"
)
AZR_LINE_PATTERN = re.compile(
    r"(?im)^\s*AZR(?:-Nummer\(n\)|-Nummer|-Nr\.?)?\s*[:#-]?\s*(.+?)\s*$"
)
BIRTH_PLACE_LINE_PATTERN = re.compile(
    r"(?im)\b(?:geb\.?\s+am\s+\d{1,2}\.\d{1,2}\.\d{4}\s+in|geboren\s+in|Geburtsort\s*[:#-]?)\s*([A-ZÄÖÜ][^\n,;]{1,80})"
)
LONG_NUMERIC_ID_PATTERN = re.compile(r"\b\d{9,}\b")
_PRESIDIO_RULE_RECOGNIZERS: Optional[dict[str, PatternRecognizer]] = None


def resolve_anonymization_engine(requested_engine: Optional[str]) -> str:
    engine = (requested_engine or ANONYMIZATION_ENGINE_DEFAULT).strip().lower()
    if engine in SUPPORTED_ANONYMIZATION_ENGINES:
        return engine
    print(
        f"[WARN] Unknown anonymization engine '{engine}', "
        f"falling back to default '{ANONYMIZATION_ENGINE_DEFAULT}'"
    )
    if ANONYMIZATION_ENGINE_DEFAULT in SUPPORTED_ANONYMIZATION_ENGINES:
        return ANONYMIZATION_ENGINE_DEFAULT
    return "gemma"


def _dedupe_entity_lists(entities: dict) -> dict:
    deduped = {}
    for key, values in entities.items():
        if not isinstance(values, list):
            deduped[key] = values
            continue
        seen = set()
        out = []
        for value in values:
            if not isinstance(value, str):
                continue
            clean = value.strip()
            if not clean:
                continue
            token = clean.casefold()
            if token in seen:
                continue
            seen.add(token)
            out.append(clean)
        deduped[key] = out
    return deduped


def _normalize_extraction_entities(payload: Any) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {key: [] for key in EXTRACTION_ENTITY_KEYS}
    if not isinstance(payload, dict):
        return normalized

    for key in EXTRACTION_ENTITY_KEYS:
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        out: list[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            clean = value.strip()
            if not clean:
                continue
            out.append(clean)
        normalized[key] = out
    return normalized


def _merge_extraction_entities(base: dict[str, list[str]], incoming: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {key: list(base.get(key, [])) for key in EXTRACTION_ENTITY_KEYS}
    for key in EXTRACTION_ENTITY_KEYS:
        merged[key].extend(incoming.get(key, []))
    return _dedupe_entity_lists(merged)


def _build_extraction_format_schema(keys: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {key: EXTRACTION_FIELD_SCHEMA[key] for key in keys},
        "required": keys,
        "additionalProperties": False,
    }


def _build_presidio_rule_recognizers() -> dict[str, PatternRecognizer]:
    global _PRESIDIO_RULE_RECOGNIZERS
    if _PRESIDIO_RULE_RECOGNIZERS is not None:
        return _PRESIDIO_RULE_RECOGNIZERS

    if PatternRecognizer is None or Pattern is None:
        _PRESIDIO_RULE_RECOGNIZERS = {}
        return _PRESIDIO_RULE_RECOGNIZERS

    _PRESIDIO_RULE_RECOGNIZERS = {
        "birth_dates": PatternRecognizer(
            supported_entity="BIRTH_DATE",
            supported_language="de",
            patterns=[
                Pattern(
                    "birth_date",
                    r"\b\d{1,2}\.\s*\d{1,2}\.\s*(?:19|20)\d{2}\b",
                    0.55,
                )
            ],
        ),
        "plz_city_lines": PatternRecognizer(
            supported_entity="PLZ_CITY",
            supported_language="de",
            patterns=[
                Pattern(
                    "plz_city",
                    r"\b\d{5}[ \t]+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]+(?:[ \t-]+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]+){0,2}\b",
                    0.65,
                )
            ],
        ),
        "streets": PatternRecognizer(
            supported_entity="STREET_ADDRESS",
            supported_language="de",
            patterns=[
                Pattern(
                    "street_address",
                    r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\-]+(?:straße|strabe|str\.|weg|platz|allee|gasse|ring|damm|ufer)\s*\d+\s*[A-Za-z]?\b",
                    0.7,
                )
            ],
        ),
        "azr_numbers": PatternRecognizer(
            supported_entity="AZR_NUMBER",
            supported_language="de",
            patterns=[
                Pattern(
                    "azr_label",
                    r"(?i)\bAZR(?:-Nummer|-Nr\.?)?\s*[:#-]?\s*[A-Z0-9][A-Z0-9./\-]{5,}\b",
                    0.85,
                )
            ],
        ),
        "aufenthaltsgestattung_ids": PatternRecognizer(
            supported_entity="AUFENTHALTSGESTATTUNG_ID",
            supported_language="de",
            patterns=[
                Pattern(
                    "aufenthaltsgestattung_label",
                    r"(?i)\bAufenthaltsgestattung\s*[:#-]?\s*[A-Z0-9][A-Z0-9./\-]{4,}\b",
                    0.85,
                )
            ],
        ),
        "case_numbers": PatternRecognizer(
            supported_entity="CASE_NUMBER",
            supported_language="de",
            patterns=[
                Pattern(
                    "aktenzeichen_label",
                    r"(?i)\b(?:Az\.?|Aktenzeichen|Geschäftszeichen|Dolmetscher(?:-Nr\.?|nummer)?)\s*[:#-]?\s*(?:[A-Z0-9./]*-[A-Z0-9./\-]{2,}|[A-Z]\d[A-Z0-9./\-]{3,})\b",
                    0.8,
                ),
                Pattern(
                    "numeric_hyphen_id",
                    r"\b\d{6,10}\s*-\s*\d{2,5}\b",
                    0.7,
                ),
                Pattern(
                    "alpha_numeric_doc_id",
                    r"\b[A-Z]\d[A-Z0-9][A-Z0-9./\-]{3,}\b",
                    0.65,
                ),
            ],
        ),
    }
    return _PRESIDIO_RULE_RECOGNIZERS


def _extract_presidio_rule_entities(text: str) -> dict[str, list[str]]:
    entities = {key: [] for key in EXTRACTION_ENTITY_KEYS}
    recognizers = _build_presidio_rule_recognizers()
    if not recognizers or not text.strip():
        return entities

    def _span_value(start: int, end: int) -> str:
        return text[start:end].strip(" \t\r\n,;:")

    try:
        for result in recognizers["birth_dates"].analyze(
            text=text, entities=["BIRTH_DATE"], nlp_artifacts=None
        ):
            value = _span_value(result.start, result.end)
            prefix_window = text[max(0, result.start - 40) : result.start]
            suffix_window = text[result.end : min(len(text), result.end + 20)]
            if BIRTH_CONTEXT_PATTERN.search(prefix_window) or re.search(
                r"(?i)^\s*(?:,|\)|;)?\s*(geboren|geburtsdatum|jahrgang)\b",
                suffix_window,
            ):
                entities["birth_dates"].append(value)

        for match in BIRTH_YEAR_CONTEXT_PATTERN.finditer(text):
            entities["birth_dates"].append(match.group(0).strip())

        for result in recognizers["plz_city_lines"].analyze(
            text=text, entities=["PLZ_CITY"], nlp_artifacts=None
        ):
            value = _span_value(result.start, result.end)
            match = PLZ_CITY_CAPTURE_PATTERN.search(value)
            if not match:
                continue
            entities["postal_codes"].append(match.group(1).strip())
            entities["cities"].append(match.group(2).strip())

        for result in recognizers["streets"].analyze(
            text=text, entities=["STREET_ADDRESS"], nlp_artifacts=None
        ):
            entities["streets"].append(_span_value(result.start, result.end))

        for result in recognizers["azr_numbers"].analyze(
            text=text, entities=["AZR_NUMBER"], nlp_artifacts=None
        ):
            value = AZR_LABEL_STRIP_PATTERN.sub("", _span_value(result.start, result.end))
            if value:
                entities["azr_numbers"].append(value)

        for result in recognizers["aufenthaltsgestattung_ids"].analyze(
            text=text, entities=["AUFENTHALTSGESTATTUNG_ID"], nlp_artifacts=None
        ):
            value = AUFENTHALTSGESTATTUNG_LABEL_STRIP_PATTERN.sub(
                "", _span_value(result.start, result.end)
            )
            if value:
                entities["aufenthaltsgestattung_ids"].append(value)

        for result in recognizers["case_numbers"].analyze(
            text=text, entities=["CASE_NUMBER"], nlp_artifacts=None
        ):
            value = CASE_NUMBER_LABEL_STRIP_PATTERN.sub(
                "", _span_value(result.start, result.end)
            )
            if value:
                entities["case_numbers"].append(value)

        for match in AZR_LINE_PATTERN.finditer(text):
            line_tail = match.group(1)
            for candidate in LONG_NUMERIC_ID_PATTERN.findall(line_tail):
                entities["azr_numbers"].append(candidate)

        for match in BIRTH_PLACE_LINE_PATTERN.finditer(text):
            candidate = match.group(1).strip(" \t\r\n,;:.")
            if candidate:
                entities["birth_places"].append(candidate)
    except Exception as exc:
        print(f"[WARN] Presidio rule extraction failed: {exc}")
        return {key: [] for key in EXTRACTION_ENTITY_KEYS}

    return _dedupe_entity_lists(entities)


def _split_text_into_pages(text: str) -> list[str]:
    clean_text = text or ""
    if not clean_text.strip():
        return []

    if "\f" in clean_text:
        pages = [p.strip() for p in clean_text.split("\f") if p and p.strip()]
        if pages:
            return pages

    page_header_pattern = r"(?m)^--- Page \d+ ---\s*$"
    if not re.search(page_header_pattern, clean_text):
        return []

    raw_parts = re.split(page_header_pattern, clean_text)
    pages = [part.strip() for part in raw_parts if part and part.strip()]
    return pages


def _split_text_for_extraction(
    text: str, chunk_pages: int, fallback_chunk_chars: int
) -> list[str]:
    if chunk_pages <= 0:
        return [text]

    clean_text = text or ""
    if not clean_text.strip():
        return [clean_text]

    pages = _split_text_into_pages(clean_text)

    if pages:
        chunks: list[str] = []
        for i in range(0, len(pages), chunk_pages):
            chunk = "\n\n\f\n\n".join(pages[i : i + chunk_pages]).strip()
            if chunk:
                chunks.append(chunk)
        if chunks:
            return chunks

    if fallback_chunk_chars <= 0 or len(clean_text) <= fallback_chunk_chars:
        return [clean_text]

    chunks: list[str] = []
    start = 0
    length = len(clean_text)
    while start < length:
        end = min(length, start + fallback_chunk_chars)
        if end < length:
            split_at = clean_text.rfind("\n\n", start, end)
            if split_at > start + 1024:
                end = split_at
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks or [clean_text]


def _build_known_entities_hint(stage_keys: list[str], entities: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for key in stage_keys:
        values = entities.get(key, [])
        if not isinstance(values, list):
            continue
        clean_values = [value.strip() for value in values if isinstance(value, str) and value.strip()]
        if not clean_values:
            continue
        preview = clean_values[:12]
        suffix = " ..." if len(clean_values) > len(preview) else ""
        lines.append(f"- {key}: {json.dumps(preview, ensure_ascii=False)}{suffix}")

    if not lines:
        return ""

    return (
        "Known entities from previous pages (hints only).\n"
        "Use them to resolve OCR variants and repeated mentions, but only return items "
        "that are supported by the current page.\n"
        + "\n".join(lines)
        + "\n\nCurrent page:\n"
    )


def _apply_page_level_entity_tightening(
    entities: dict[str, list[str]], text: str
) -> dict[str, list[str]]:
    tightened = _dedupe_entity_lists(entities)
    tightened = filter_non_person_group_labels(tightened, text)
    tightened = augment_names_from_role_markers(tightened, text)
    tightened = augment_names_from_person_fields(tightened, text)
    tightened = filter_non_person_organization_labels(tightened)
    tightened = _filter_name_artifacts(tightened)
    tightened = _filter_identifier_artifacts(tightened)
    return _dedupe_entity_lists(tightened)


def _merge_flair_names(entities: dict, flair_names: list[str]) -> dict:
    names = entities.get("names", [])
    if not isinstance(names, list):
        names = []

    seen = {n.strip().casefold() for n in names if isinstance(n, str) and n.strip()}
    added = []
    for raw in flair_names:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if len(candidate) < 2:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(candidate)
        added.append(candidate)

    if added:
        print(f"[INFO] Added Flair name hints: count={len(added)}")
    entities["names"] = names
    return entities


def _filter_name_artifacts(entities: dict) -> dict:
    names = entities.get("names")
    if not isinstance(names, list):
        return entities

    noise_tokens = {
        "passersatz",
        "vornamen",
        "vorname",
        "geburtsdatum",
        "grundstücke",
        "grundstucke",
        "placeholder",
        "surface",
        "digits",
        "aktenzeichen",
        "alias",
        "republik",
        "syrien",
        "arabische",
        "gerichtsbescheid",
        "beschluss",
        "urteil",
        "country",
        "report",
        "italy",
        "aida",
        "internazionale",
        "leben",
        "leib",
        "nierensteinen",
        "bandscheibenproblemen",
    }
    filtered: list[str] = []
    for raw in names:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if not candidate:
            continue

        lowered = candidate.casefold()
        if "\n" in candidate or ":" in candidate or "/" in candidate:
            continue
        if re.search(r"(?i)\bgeb\.?\b", candidate):
            continue
        if any(token in lowered for token in noise_tokens):
            continue
        if lowered.startswith("nr") and any(ch.isdigit() for ch in candidate):
            continue
        if not any(ch.isalpha() for ch in candidate):
            continue
        if sum(ch.isdigit() for ch in candidate) >= 3:
            continue
        if len(candidate.split()) > 4:
            continue

        filtered.append(candidate)

    entities["names"] = filtered
    return entities


def _normalize_identifier_value(value: str) -> str:
    normalized = re.sub(r"\s*-\s*", "-", value.strip())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" ,;:.")


def _digit_count(value: str) -> int:
    return sum(ch.isdigit() for ch in value)


def _filter_identifier_artifacts(entities: dict) -> dict:
    case_numbers = entities.get("case_numbers")
    azr_numbers = entities.get("azr_numbers")
    aufenthaltsgestattung_ids = entities.get("aufenthaltsgestattung_ids")

    if isinstance(case_numbers, list):
        cleaned_case_numbers: list[str] = []
        for raw in case_numbers:
            if not isinstance(raw, str):
                continue
            candidate = _normalize_identifier_value(raw)
            if not candidate:
                continue
            if _digit_count(candidate) < 5 and not re.search(r"[A-Za-z]", candidate):
                continue
            cleaned_case_numbers.append(candidate)
        entities["case_numbers"] = cleaned_case_numbers

    normalized_case_numbers = {
        _normalize_identifier_value(value)
        for value in entities.get("case_numbers", [])
        if isinstance(value, str) and value.strip()
    }

    if isinstance(azr_numbers, list):
        cleaned_azr_numbers: list[str] = []
        for raw in azr_numbers:
            if not isinstance(raw, str):
                continue
            candidate = _normalize_identifier_value(raw)
            if not candidate:
                continue
            if candidate in normalized_case_numbers:
                continue
            if _digit_count(candidate) < 6:
                continue
            cleaned_azr_numbers.append(candidate)
        entities["azr_numbers"] = cleaned_azr_numbers

    if isinstance(aufenthaltsgestattung_ids, list):
        cleaned_aufenthaltsgestattung_ids: list[str] = []
        for raw in aufenthaltsgestattung_ids:
            if not isinstance(raw, str):
                continue
            candidate = _normalize_identifier_value(raw)
            if not candidate:
                continue
            if candidate in normalized_case_numbers:
                continue
            if candidate.endswith("-"):
                continue
            if _digit_count(candidate) < 6 and not re.search(r"[A-Za-z]", candidate):
                continue
            cleaned_aufenthaltsgestattung_ids.append(candidate)
        entities["aufenthaltsgestattung_ids"] = cleaned_aufenthaltsgestattung_ids

    return _dedupe_entity_lists(entities)


def _clean_display_names(names: list[str], text: str) -> list[str]:
    entities = {"names": list(names)}
    entities = filter_non_person_group_labels(entities, text)
    entities = filter_non_person_organization_labels(entities)
    entities = _filter_name_artifacts(entities)
    return _dedupe_entity_lists(entities).get("names", [])


def _clean_display_addresses(addresses: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen = set()
    for raw in addresses:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if not candidate:
            continue
        lowered = candidate.casefold()
        if any(
            token in lowered
            for token in (
                "aktenzeichen",
                "geschaftszeichen",
                "geschäftszeichen",
                "gesch.-z",
                "azr-nummer",
                "azr:",
            )
        ):
            continue
        token = candidate.casefold()
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(candidate)
    return cleaned


def _stage_temperature(stage_name: str, is_gemma3: bool, default_temperature: float) -> float:
    if is_gemma3:
        if stage_name == "names":
            return _float_env("OLLAMA_NAMES_TEMP_GEMMA3", 0.5)
        if stage_name == "addresses":
            return _float_env("OLLAMA_ADDRESSES_TEMP_GEMMA3", 0.2)
        if stage_name == "birth_ids":
            return _float_env("OLLAMA_BIRTH_IDS_TEMP_GEMMA3", 0.2)
    if stage_name == "names":
        return _float_env("OLLAMA_NAMES_TEMP_QWEN", min(default_temperature, 0.35))
    if stage_name == "addresses":
        return _float_env("OLLAMA_ADDRESSES_TEMP_QWEN", default_temperature)
    if stage_name == "birth_ids":
        return _float_env("OLLAMA_BIRTH_IDS_TEMP_QWEN", max(0.25, default_temperature - 0.1))
    return default_temperature


def _stage_passes(stage_name: str, is_gemma3: bool) -> int:
    if is_gemma3 and stage_name == "names":
        return max(1, _int_env("OLLAMA_NAMES_PASSES_GEMMA3", 2))
    return 1


async def _fetch_flair_name_hints(
    service_url: str, text: str, document_type: str
) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            status_resp = await client.get(f"{service_url}/status")
            if status_resp.status_code == 200:
                status_payload = status_resp.json()
                anon_backend = (
                    status_payload.get("services", {}).get("anon_backend") or ""
                ).strip().lower()
                if anon_backend != "flair":
                    print(
                        f"[WARN] qwen_flair engine requested but service manager backend "
                        f"is '{anon_backend or 'unknown'}' (expected 'flair')."
                    )
                    return []

            flair_resp = await client.post(
                f"{service_url}/anonymize",
                json={"text": text, "document_type": document_type},
            )
            flair_resp.raise_for_status()
            flair_data = flair_resp.json()
    except Exception as exc:
        print(f"[WARN] Flair name hint fetch failed: {exc}")
        return []

    plaintiff_names = flair_data.get("plaintiff_names") or []
    family_members = flair_data.get("family_members") or []
    if not isinstance(plaintiff_names, list):
        plaintiff_names = []
    if not isinstance(family_members, list):
        family_members = []
    return [*plaintiff_names, *family_members]


async def _fetch_qwen_name_hints(service_url: str, text: str) -> list[str]:
    if not QWEN_NAME_REVIEW_ENABLED or not text.strip():
        return []

    stage_format = _build_extraction_format_schema(["names"])
    stage_temperature = _float_env("OLLAMA_QWEN_NAME_REVIEW_TEMP", 0.2)
    top_k = _int_env("OLLAMA_QWEN_NAME_REVIEW_TOP_K", 20)
    top_p = _float_env("OLLAMA_QWEN_NAME_REVIEW_TOP_P", 0.8)
    min_p = _optional_float_env("OLLAMA_QWEN_NAME_REVIEW_MIN_P")
    repeat_penalty = _float_env("OLLAMA_REPEAT_PENALTY_QWEN", 1.0)
    num_ctx = _int_env("OLLAMA_NUM_CTX_QWEN_NAME_REVIEW", _int_env("OLLAMA_NUM_CTX_DEFAULT", 16384))
    chunk_pages = _optional_positive_int_env("OLLAMA_QWEN_NAMES_CHUNK_PAGES") or 2
    chunk_chars = _optional_positive_int_env("OLLAMA_QWEN_NAMES_CHUNK_CHARS") or 18000
    chunks = _split_text_for_extraction(text, chunk_pages, chunk_chars)

    merged_entities = {"names": []}
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                prompt = NAMES_EXTRACTION_PROMPT_PREFIX + chunk_text
                payload = {
                    "model": QWEN_MODEL,
                    "prompt": prompt,
                    "format": stage_format,
                    "stream": False,
                    "options": {
                        "temperature": stage_temperature,
                        "num_ctx": num_ctx,
                        "top_k": top_k,
                        "top_p": top_p,
                        "min_p": min_p,
                        "repeat_penalty": repeat_penalty,
                    },
                }
                print(
                    f"[INFO] Qwen name review chunk={chunk_idx}/{len(chunks)} "
                    f"model={QWEN_MODEL} chars={len(chunk_text)} temp={stage_temperature}"
                )
                response = await client.post(f"{service_url}/extract-entities", json=payload)
                response.raise_for_status()
                data = response.json()
                raw_response = data.get("response", "{}")
                parsed_payload = json.loads(raw_response)
                parsed_entities = _normalize_extraction_entities(parsed_payload)
                merged_entities = _merge_extraction_entities(
                    merged_entities, {"names": parsed_entities.get("names", [])}
                )
    except Exception as exc:
        print(f"[WARN] Qwen name review failed: {exc}")
        return []

    merged_entities = filter_non_person_group_labels(merged_entities, text)
    merged_entities = filter_non_person_organization_labels(merged_entities)
    merged_entities = augment_names_from_role_markers(merged_entities, text)
    merged_entities = augment_names_from_person_fields(merged_entities, text)
    merged_entities = _filter_name_artifacts(merged_entities)
    merged_entities = _dedupe_entity_lists(merged_entities)
    return merged_entities.get("names", [])


async def _fetch_flair_anonymization_payload(
    service_url: str, text: str, document_type: str
) -> Optional[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            status_resp = await client.get(f"{service_url}/status")
            if status_resp.status_code == 200:
                status_payload = status_resp.json()
                anon_backend = (
                    status_payload.get("services", {}).get("anon_backend") or ""
                ).strip().lower()
                if anon_backend != "flair":
                    print(
                        f"[WARN] flair_presidio requested but service manager backend "
                        f"is '{anon_backend or 'unknown'}' (expected 'flair')."
                    )
                    return None

            response = await client.post(
                f"{service_url}/anonymize",
                json={"text": text, "document_type": document_type},
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        print(f"[WARN] Flair anonymization fetch failed: {exc}")
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _merge_flair_and_presidio_entities(
    flair_payload: dict[str, Any], rule_entities: dict[str, list[str]]
) -> tuple[dict[str, list[str]], list[str], float]:
    merged = {key: list(rule_entities.get(key, [])) for key in EXTRACTION_ENTITY_KEYS}

    flair_names: list[str] = []
    for key in ("plaintiff_names", "family_members"):
        values = flair_payload.get(key) or []
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                flair_names.append(value.strip())

    flair_addresses: list[str] = []
    raw_addresses = flair_payload.get("addresses") or []
    if isinstance(raw_addresses, list):
        for value in raw_addresses:
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                lowered = candidate.casefold()
                if "aktenzeichen" in lowered or lowered.startswith("az:"):
                    continue
                if _digit_count(candidate) >= 6 and not re.search(r"[A-Za-zÄÖÜäöüß]", candidate):
                    continue
                flair_addresses.append(candidate)

    merged["names"] = flair_names
    merged["streets"].extend(flair_addresses)
    merged = _filter_name_artifacts(merged)
    merged = _dedupe_entity_lists(merged)

    confidence = flair_payload.get("confidence")
    try:
        flair_confidence = float(confidence)
    except (TypeError, ValueError):
        flair_confidence = 0.95

    all_addresses = list(flair_addresses)
    for key in ("streets", "postal_codes", "cities"):
        all_addresses.extend(merged.get(key, []))
    seen = set()
    deduped_addresses: list[str] = []
    for value in all_addresses:
        token = value.strip().casefold()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped_addresses.append(value.strip())

    return merged, deduped_addresses, flair_confidence


async def anonymize_document_text(
    text: str,
    document_type: str,
    engine: str,
    extract_chunk_pages: Optional[int] = None,
    extract_num_ctx: Optional[int] = None,
) -> Optional[AnonymizationResult]:
    """Extract entities via desktop LLM, then apply regex anonymization locally."""
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        print("[WARNING] ANONYMIZATION_SERVICE_URL not configured")
        return None

    await ensure_service_manager_ready()

    if engine == "flair_presidio":
        flair_payload = await _fetch_flair_anonymization_payload(
            service_url, text, document_type
        )
        if flair_payload is not None:
            rule_entities = _extract_presidio_rule_entities(text)
            merged_entities, all_addresses, flair_confidence = _merge_flair_and_presidio_entities(
                flair_payload, rule_entities
            )
            qwen_name_hints = await _fetch_qwen_name_hints(service_url, text)
            if qwen_name_hints:
                merged_entities = _merge_flair_names(merged_entities, qwen_name_hints)
            merged_entities = _filter_identifier_artifacts(merged_entities)
            merged_entities = filter_non_person_group_labels(merged_entities, text)
            merged_entities = filter_non_person_organization_labels(merged_entities)
            merged_entities = _filter_name_artifacts(merged_entities)
            merged_entities = _dedupe_entity_lists(merged_entities)
            flair_anonymized_text = flair_payload.get("anonymized_text")
            if not isinstance(flair_anonymized_text, str) or not flair_anonymized_text:
                flair_anonymized_text = apply_regex_replacements(text, merged_entities)
            final_text = apply_regex_replacements(flair_anonymized_text, merged_entities)
            display_names = _clean_display_names(merged_entities.get("names", []), text)
            display_addresses = _clean_display_addresses(all_addresses)
            extraction_inference_params = {
                "engine": "flair_presidio",
                "primary": "flair_service_manager",
                "rules": "presidio_pattern_recognizers",
                "service_url": f"{service_url}/anonymize",
                "service_confidence": flair_confidence,
                "qwen_name_review": bool(qwen_name_hints),
                "qwen_name_model": QWEN_MODEL if qwen_name_hints else None,
                "rule_entity_counts": {
                    key: len(values)
                    for key, values in merged_entities.items()
                    if isinstance(values, list) and values
                },
            }
            return AnonymizationResult(
                anonymized_text=final_text,
                plaintiff_names=display_names,
                birth_dates=merged_entities.get("birth_dates", []),
                addresses=display_addresses,
                confidence=flair_confidence,
                original_text=text,
                processed_characters=len(text),
                extraction_inference_params=extraction_inference_params,
            )
        raise HTTPException(
            status_code=503,
            detail="Flair anonymization backend unavailable for flair_presidio engine.",
        )

    model = GEMMA_MODEL
    default_temperature = 0.3
    top_k = None
    top_p = None
    min_p = None
    repeat_penalty = 1.0
    use_presidio_rules = _bool_env("ANONYMIZATION_USE_PRESIDIO_RULES", True)
    if engine == "qwen_flair":
        model = QWEN_MODEL
        default_temperature = _float_env("OLLAMA_TEMP_QWEN", 0.45)
        top_k = _int_env("OLLAMA_TOP_K_QWEN", 40)
        top_p = _float_env("OLLAMA_TOP_P_QWEN", 0.92)
        min_p = _optional_float_env("OLLAMA_MIN_P_QWEN")
        repeat_penalty = _float_env("OLLAMA_REPEAT_PENALTY_QWEN", 1.0)

    is_gemma3 = (model or "").strip().lower().startswith("gemma3")
    num_ctx = _int_env("OLLAMA_NUM_CTX_DEFAULT", 32768)
    if is_gemma3:
        num_ctx = _int_env("OLLAMA_NUM_CTX_GEMMA3", 32768)
        print(
            f"[INFO] Gemma3 detected (model={model}); using format='json' "
            "instead of JSON schema for extraction stability"
        )

    if extract_num_ctx is not None and extract_num_ctx > 0:
        num_ctx = extract_num_ctx

    env_chunk_pages = _optional_positive_int_env("OLLAMA_EXTRACT_CHUNK_PAGES")
    env_chunk_chars = _optional_positive_int_env("OLLAMA_EXTRACT_CHUNK_CHARS")
    active_chunk_pages = extract_chunk_pages or env_chunk_pages or 0
    fallback_chunk_chars = env_chunk_chars or 18000
    if active_chunk_pages <= 0 and _split_text_into_pages(text):
        active_chunk_pages = _int_env("OLLAMA_AUTO_PAGE_CHUNK_PAGES", 2)

    stage_plans: list[dict[str, Any]] = []
    for stage_spec in EXTRACTION_STAGE_SPECS:
        stage_name = stage_spec["name"]
        stage_keys = list(stage_spec["keys"])
        stage_format: str | dict[str, Any] = (
            "json" if is_gemma3 else _build_extraction_format_schema(stage_keys)
        )
        stage_temperature = _stage_temperature(stage_name, is_gemma3, default_temperature)
        stage_pass_count = _stage_passes(stage_name, is_gemma3)
        stage_plans.append(
            {
                "name": stage_name,
                "keys": stage_keys,
                "prompt_prefix": stage_spec["prompt_prefix"],
                "format": stage_format,
                "temperature": stage_temperature,
                "passes": stage_pass_count,
            }
        )

    def _build_payload(
        prompt_text: str, stage_format: str | dict[str, Any], stage_temperature: float
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": stage_temperature,
            "num_predict": 4096,
            "num_ctx": num_ctx,
            "repeat_penalty": repeat_penalty,
        }
        if top_k is not None:
            options["top_k"] = top_k
        if top_p is not None:
            options["top_p"] = top_p
        if min_p is not None:
            options["min_p"] = min_p
        return {
            "model": model,
            "prompt": prompt_text,
            "stream": False,
            "format": stage_format,
            "options": options,
        }

    primary_stage = stage_plans[0]
    extraction_inference_params: dict[str, Any] = {
        "model": model,
        "format": (
            primary_stage["format"]
            if isinstance(primary_stage["format"], str)
            else "json_schema"
        ),
        "temperature": primary_stage["temperature"],
        "num_ctx": num_ctx,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "extract_chunk_pages": active_chunk_pages or None,
        "extract_chunk_chars": fallback_chunk_chars if active_chunk_pages else None,
        "staged_extraction": True,
        "presidio_rules": use_presidio_rules and bool(_build_presidio_rule_recognizers()),
        "stages": [
            {
                "name": stage["name"],
                "keys": stage["keys"],
                "format": stage["format"] if isinstance(stage["format"], str) else "json_schema",
                "temperature": stage["temperature"],
                "passes": stage["passes"],
            }
            for stage in stage_plans
        ],
    }

    try:
        chunks = [text]
        if active_chunk_pages > 0:
            chunks = _split_text_for_extraction(text, active_chunk_pages, fallback_chunk_chars)

        chunk_mode = len(chunks) > 1
        if chunk_mode:
            print(
                f"[INFO] Chunked staged extraction enabled: chunks={len(chunks)} "
                f"chunk_pages={active_chunk_pages} chunk_chars={fallback_chunk_chars}"
            )
        print(
            f"[INFO] Staged extraction request "
            f"url={service_url}/extract-entities model={model} "
            f"payload_chars={len(text)} document_type={document_type} "
            f"engine={engine} num_ctx={num_ctx} "
            f"stages={[(s['name'], s['passes']) for s in stage_plans]}"
        )

        extraction_prompt_tokens_sum = 0
        extraction_completion_tokens_sum = 0
        extraction_total_duration_ns_sum = 0
        extraction_prompt_tokens = None
        extraction_completion_tokens = None
        extraction_total_duration_ns = None
        merged_entities = {key: [] for key in EXTRACTION_ENTITY_KEYS}

        sequential_page_accumulator = chunk_mode
        extraction_inference_params["sequential_page_accumulator"] = sequential_page_accumulator

        async with httpx.AsyncClient(timeout=300.0) as client:
            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                page_entities = {key: [] for key in EXTRACTION_ENTITY_KEYS}

                for stage in stage_plans:
                    stage_name = stage["name"]
                    stage_keys = stage["keys"]
                    stage_prompt_prefix = stage["prompt_prefix"]
                    stage_format = stage["format"]
                    stage_temperature = stage["temperature"]
                    stage_passes = stage["passes"]

                    for pass_idx in range(1, stage_passes + 1):
                        prompt = stage_prompt_prefix
                        if sequential_page_accumulator:
                            prompt += _build_known_entities_hint(stage_keys, merged_entities)
                        prompt += chunk_text

                        payload = _build_payload(prompt, stage_format, stage_temperature)
                        if chunk_mode or stage_passes > 1:
                            print(
                                f"[INFO] Extraction page={chunk_idx}/{len(chunks)} "
                                f"stage={stage_name} "
                                f"pass={pass_idx}/{stage_passes} "
                                f"payload_chars={len(chunk_text)} temp={stage_temperature}"
                            )
                        response = await client.post(
                            f"{service_url}/extract-entities",
                            json=payload,
                        )
                        response.raise_for_status()
                        data = response.json()

                        raw_response = data.get("response", "{}")
                        parsed_payload = json.loads(raw_response)
                        parsed_entities = _normalize_extraction_entities(parsed_payload)
                        stage_entities = {
                            key: parsed_entities.get(key, []) for key in stage_keys
                        }
                        page_entities = _merge_extraction_entities(page_entities, stage_entities)

                        prompt_tokens = _optional_int(data.get("prompt_eval_count")) or 0
                        completion_tokens = _optional_int(data.get("eval_count")) or 0
                        total_duration_ns = _optional_int(data.get("total_duration")) or 0
                        extraction_prompt_tokens_sum += prompt_tokens
                        extraction_completion_tokens_sum += completion_tokens
                        extraction_total_duration_ns_sum += total_duration_ns

                page_entities = _apply_page_level_entity_tightening(page_entities, chunk_text)
                merged_entities = _merge_extraction_entities(merged_entities, page_entities)

        extraction_prompt_tokens = extraction_prompt_tokens_sum
        extraction_completion_tokens = extraction_completion_tokens_sum
        extraction_total_duration_ns = extraction_total_duration_ns_sum
        extraction_inference_params["chunk_count"] = len(chunks)

        print(
            "[INFO] Entity extraction usage "
            f"prompt_tokens={extraction_prompt_tokens} "
            f"completion_tokens={extraction_completion_tokens} "
            f"total_duration_ns={extraction_total_duration_ns} "
            f"inference_params={extraction_inference_params}"
        )
        entities = _dedupe_entity_lists(merged_entities)

        if use_presidio_rules:
            presidio_entities = _extract_presidio_rule_entities(text)
            presidio_count = sum(
                len(values) for values in presidio_entities.values() if isinstance(values, list)
            )
            if presidio_count:
                print(f"[INFO] Presidio rule extraction added {presidio_count} candidates")
                entities = _merge_extraction_entities(entities, presidio_entities)

        entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        print(f"[INFO] Raw extraction: {entity_count} entities")

        entities = filter_non_person_group_labels(entities, text)
        entities = augment_names_from_role_markers(entities, text)
        entities = augment_names_from_person_fields(entities, text)
        entities = filter_non_person_organization_labels(entities)
        entities = _filter_name_artifacts(entities)
        entities = _filter_identifier_artifacts(entities)
        if engine == "qwen_flair":
            flair_names = await _fetch_flair_name_hints(service_url, text, document_type)
            entities = _merge_flair_names(entities, flair_names)
            entities = filter_non_person_group_labels(entities, text)
            entities = filter_non_person_organization_labels(entities)
            entities = _filter_name_artifacts(entities)
            entities = _filter_identifier_artifacts(entities)
        entities = _dedupe_entity_lists(entities)

        filtered_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        print(f"[INFO] After local filters: {filtered_count} entities")
        counts = _entity_counts(entities)
        if counts:
            print(f"[INFO] Entity counts by category: {counts}")

        anonymized_text = apply_regex_replacements(text, entities)

        all_addresses = entities.get("streets", []) + entities.get("cities", [])

        print("[SUCCESS] Anonymization completed (local regex)")

        return AnonymizationResult(
            anonymized_text=anonymized_text,
            plaintiff_names=entities.get("names", []),
            birth_dates=entities.get("birth_dates", []),
            addresses=all_addresses,
            confidence=0.95,
            original_text=text,
            processed_characters=len(text),
            extraction_prompt_tokens=extraction_prompt_tokens,
            extraction_completion_tokens=extraction_completion_tokens,
            extraction_total_duration_ns=extraction_total_duration_ns,
            extraction_inference_params=extraction_inference_params,
        )

    except HTTPException:
        raise
    except httpx.TimeoutException:
        print("[ERROR] Entity extraction timeout (>300s)")
        raise HTTPException(
            status_code=504,
            detail="Entity extraction timeout (>300s). Please retry.",
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        message = None
        try:
            err = exc.response.json()
            if isinstance(err, dict):
                message = err.get("detail") or err.get("message")
            else:
                message = str(err)
        except Exception:
            message = exc.response.text or str(exc)

        detail_message = message or f"HTTP {status} from service manager"
        print(f"[ERROR] Entity extraction HTTP error: {status} – {detail_message}")
        raise HTTPException(status_code=status, detail=detail_message)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse LLM entity JSON: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse model response as JSON.",
        )
    except Exception as exc:
        print(f"[ERROR] Anonymization error: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Anonymization error: {exc}",
        )


@router.post("/documents/{document_id}/anonymize")
@limiter.limit("100/hour")
async def anonymize_document_endpoint(
    request: Request,
    document_id: str,
    force: bool = Query(False),
    engine: Optional[str] = Query(None),
    extract_chunk_pages: Optional[int] = Query(None, ge=1, le=50),
    extract_num_ctx: Optional[int] = Query(None, ge=1024, le=131072),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize a classified document (Anhörung, Bescheid, or Sonstige gespeicherte Quellen)."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = (
        db.query(Document)
        .filter(
            Document.id == doc_uuid,
            Document.owner_id == current_user.id,
            Document.case_id == current_user.active_case_id,
        )
        .first()
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # All document types can be anonymized (names and addresses are extracted)
    resolved_engine = resolve_anonymization_engine(engine)

    if force and document.is_anonymized:
        print(f"[INFO] Force re-anonymization requested for document_id={document.id}")

    if document.is_anonymized and document.anonymization_metadata and not force:
        # Only path-based anonymized text is supported
        anonymized_text = ""
        path = document.anonymization_metadata.get("anonymized_text_path")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    anonymized_text = f.read()
            except Exception as e:
                print(f"[ERROR] Failed to read anonymized text from file: {e}")
        else:
            print(f"[WARN] Missing anonymized_text_path for document_id={document.id}; reprocessing.")
            anonymized_text = ""

        if anonymized_text:
            processed_chars = document.anonymization_metadata.get(
                "processed_characters"
            )
            remaining_chars = document.anonymization_metadata.get(
                "remaining_characters"
            )
            if processed_chars is not None and remaining_chars is not None:
                input_characters = processed_chars + remaining_chars
            else:
                input_characters = len(anonymized_text)

            return {
                "status": "success",
                "anonymized_text": anonymized_text,
                "plaintiff_names": document.anonymization_metadata.get(
                    "plaintiff_names", []
                ),
                "addresses": document.anonymization_metadata.get("addresses", []),
                "confidence": document.anonymization_metadata.get("confidence", 0.0),
                "input_characters": input_characters,
                "processed_characters": processed_chars,
                "remaining_characters": remaining_chars,
                "extraction_prompt_tokens": document.anonymization_metadata.get(
                    "extraction_prompt_tokens"
                ),
                "extraction_completion_tokens": document.anonymization_metadata.get(
                    "extraction_completion_tokens"
                ),
                "extraction_total_duration_ns": document.anonymization_metadata.get(
                    "extraction_total_duration_ns"
                ),
                "extraction_inference_params": document.anonymization_metadata.get(
                    "extraction_inference_params"
                ),
                "cached": True,
                "engine": document.anonymization_metadata.get("engine", resolved_engine),
            }

    pdf_path = document.file_path
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    extracted_text = None
    ocr_used = False

    cached_text = load_document_text(document)
    if document.ocr_applied and cached_text:
        extracted_text = cached_text
        ocr_used = True
        print(
            f"[INFO] Using cached OCR text for document_id={document.id}: {len(extracted_text)} characters"
        )

    if not extracted_text:
        # Check if we need OCR using shared logic (consistent with classification)
        # We respect the DB flag if it's True, otherwise we verify with the shared check.
        should_use_ocr = document.needs_ocr or check_pdf_needs_ocr(pdf_path)

        if should_use_ocr:
            print(
                f"[INFO] Document needs OCR (flag={document.needs_ocr}). Skipping direct extraction."
            )
            extracted_text = None
        else:
            try:
                extracted_text = extract_pdf_text(
                    pdf_path, max_pages=50, include_page_headers=False
                )
                # Final sanity check: even if check passed, maybe extraction failed or yielded garbage
                if extracted_text and len(extracted_text.strip()) >= 500:
                    print(
                        f"[INFO] Direct text extraction successful: {len(extracted_text)} characters"
                    )
                else:
                    print(
                        f"[INFO] Direct extraction insufficient ({len(extracted_text) if extracted_text else 0} chars), trying OCR..."
                    )
                    extracted_text = None
            except Exception as exc:
                print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
                extracted_text = None

    if not extracted_text:
        extracted_text = await perform_ocr_on_file(pdf_path)
        if extracted_text:
            ocr_used = True
            print(
                f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running.",
            )

    if not extracted_text or len(extracted_text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail="Insufficient text extracted from PDF. The document may be empty or corrupted.",
        )

    document_type = document.category
    if document_type == "Sonstiges":
        document_type = DocumentCategory.SONSTIGES.value

    text_for_anonymization = extracted_text
    try:
        text_hash = hashlib.sha256(text_for_anonymization.encode("utf-8")).hexdigest()
        text_len = len(text_for_anonymization)
        line_count = text_for_anonymization.count("\n") + 1 if text_for_anonymization else 0
        word_count = len(text_for_anonymization.split())
        null_count = text_for_anonymization.count("\x00")
        non_ascii = sum(1 for ch in text_for_anonymization if ord(ch) > 127)
        print(
            "[INFO] Anonymization payload stats "
            f"(document_id={document.id}, type={document_type}, force={force}): "
            f"chars={text_len}, words={word_count}, lines={line_count}, "
            f"non_ascii={non_ascii}, nulls={null_count}, sha256={text_hash}"
        )
    except Exception as exc:
        print(f"[WARN] Failed to compute anonymization payload stats: {exc}")
    print(
        f"[INFO] Sending {len(text_for_anonymization)} characters to anonymization service"
    )

    result = await anonymize_document_text(
        text_for_anonymization,
        document_type,
        resolved_engine,
        extract_chunk_pages=extract_chunk_pages,
        extract_num_ctx=extract_num_ctx,
    )
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Anonymization service unavailable. Please ensure it is running.",
        )

    anonymized_full_text = result.anonymized_text
    processed_chars = result.processed_characters
    remaining_chars = 0

    # Save anonymized text to file
    anonymized_filename = f"{document.id}.txt"
    anonymized_path = ANONYMIZED_TEXT_DIR / anonymized_filename
    try:
        with open(anonymized_path, "w", encoding="utf-8") as f:
            f.write(anonymized_full_text)
    except Exception as e:
        print(f"[ERROR] Failed to write anonymized text to file: {e}")
        # Fallback? Or raise? For now, we proceed, but metadata might be incomplete if we rely on path.
        # Actually, if write fails, we should probably fail the request or fallback to DB storage.
        # Let's fallback to DB storage implicitly if path is missing, but here we want to enforce file.
        raise HTTPException(
            status_code=500, detail=f"Failed to save anonymized text: {e}"
        )

    store_document_text(document, extracted_text)
    document.is_anonymized = True
    document.ocr_applied = ocr_used
    document.anonymization_metadata = {
        "plaintiff_names": result.plaintiff_names,
        "birth_dates": result.birth_dates,
        "addresses": result.addresses,
        "confidence": result.confidence,
        "anonymized_at": datetime.utcnow().isoformat(),
        "anonymized_text_path": str(anonymized_path),
        "anonymized_excerpt": result.anonymized_text,
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "input_characters": len(extracted_text),
        "ocr_used": ocr_used,
        "engine": resolved_engine,
        "extraction_prompt_tokens": result.extraction_prompt_tokens,
        "extraction_completion_tokens": result.extraction_completion_tokens,
        "extraction_total_duration_ns": result.extraction_total_duration_ns,
        "extraction_inference_params": result.extraction_inference_params,
    }
    document.processing_status = "completed"
    db.commit()

    broadcast_documents_snapshot(db, "anonymize", {"document_id": document_id})

    return {
        "status": "success",
        "anonymized_text": anonymized_full_text,
        "plaintiff_names": result.plaintiff_names,
        "birth_dates": result.birth_dates,
        "addresses": result.addresses,
        "confidence": result.confidence,
        "input_characters": len(extracted_text),
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "extraction_prompt_tokens": result.extraction_prompt_tokens,
        "extraction_completion_tokens": result.extraction_completion_tokens,
        "extraction_total_duration_ns": result.extraction_total_duration_ns,
        "extraction_inference_params": result.extraction_inference_params,
        "ocr_used": ocr_used,
        "cached": False,
        "engine": resolved_engine,
    }


@router.post("/anonymize-file")
@limiter.limit("100/hour")
async def anonymize_uploaded_file(
    request: Request,
    document_type: str = Form(...),
    file: UploadFile = File(...),
    engine: Optional[str] = Query(None),
    extract_chunk_pages: Optional[int] = Query(None, ge=1, le=50),
    extract_num_ctx: Optional[int] = Query(None, ge=1024, le=131072),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize an uploaded PDF without storing it in the database."""
    sanitized_type = document_type.strip() or "Sonstiges"
    resolved_engine = resolve_anonymization_engine(engine)

    filename = (file.filename or "upload.pdf").strip()
    _, ext = os.path.splitext(filename.lower())
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmpf.write(chunk)
        tmp_path = tmpf.name

    try:
        extracted_text = None
        ocr_used = False

        # Check if we need OCR using shared logic
        should_use_ocr = check_pdf_needs_ocr(tmp_path)

        if not should_use_ocr:
            try:
                extracted_text = extract_pdf_text(
                    tmp_path, max_pages=50, include_page_headers=False
                )
                # Final sanity check: even if check passed, maybe extraction failed or yielded garbage
                if extracted_text and len(extracted_text.strip()) >= 500:
                    print(
                        f"[INFO] Direct text extraction successful: {len(extracted_text)} characters"
                    )
                else:
                    print(
                        f"[INFO] Direct extraction insufficient ({len(extracted_text) if extracted_text else 0} chars), trying OCR..."
                    )
                    extracted_text = None
            except Exception as exc:
                print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
                extracted_text = None

        if not extracted_text:
            extracted_text = await perform_ocr_on_file(tmp_path)
            if extracted_text:
                ocr_used = True
                print(
                    f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running.",
                )

        if not extracted_text or len(extracted_text.strip()) < 100:
            raise HTTPException(
                status_code=422,
                detail="Insufficient text extracted from PDF. The document may be empty or corrupted.",
            )

        text_for_anonymization = extracted_text
        print(
            f"[INFO] Sending uploaded PDF ({len(text_for_anonymization)} chars) to anonymization service"
        )

        result = await anonymize_document_text(
            text_for_anonymization,
            sanitized_type,
            resolved_engine,
            extract_chunk_pages=extract_chunk_pages,
            extract_num_ctx=extract_num_ctx,
        )
        if result is None:
            raise HTTPException(
                status_code=503,
                detail="Anonymization service unavailable. Please ensure it is running.",
            )

        anonymized_full_text = result.anonymized_text
        processed_chars = result.processed_characters
        remaining_chars = 0

        return {
            "status": "success",
            "filename": filename,
            "anonymized_text": anonymized_full_text,
            "plaintiff_names": result.plaintiff_names,
            "birth_dates": result.birth_dates,
            "addresses": result.addresses,
            "confidence": result.confidence,
            "input_characters": len(extracted_text),
            "processed_characters": processed_chars,
            "remaining_characters": remaining_chars,
            "extraction_prompt_tokens": result.extraction_prompt_tokens,
            "extraction_completion_tokens": result.extraction_completion_tokens,
            "extraction_total_duration_ns": result.extraction_total_duration_ns,
            "extraction_inference_params": result.extraction_inference_params,
            "ocr_used": ocr_used,
            "engine": resolved_engine,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


__all__ = ["router"]
