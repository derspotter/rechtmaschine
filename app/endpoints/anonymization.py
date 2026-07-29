import json
import os
import tempfile
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
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
    ensure_anonymization_service_ready,
    get_owned_document,
    limiter,
    load_document_text,
    should_auto_anonymize_category,
    store_document_text,
    ANONYMIZED_TEXT_DIR,
)
from auth import get_current_active_user
from database import SessionLocal, get_db
from models import AnonymizeJob, Document, User
from .ocr import extract_pdf_text, perform_ocr_on_file, check_pdf_needs_ocr
from anon.anonymization_service import (
    filter_non_person_group_labels,
    filter_non_person_organization_labels,
    augment_names_from_role_markers,
    augment_names_from_person_fields,
    apply_regex_replacements,
    apply_regex_replacements_parallel,
)

router = APIRouter()
IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

GEMMA_MODEL = os.getenv("OLLAMA_MODEL_GEMMA", os.getenv("OLLAMA_MODEL", "gemma3:12b"))
QWEN_MODEL = os.getenv(
    "OLLAMA_MODEL_QWEN",
    os.getenv("OLLAMA_MODEL", "qwen3.5:9b-q5_k_m"),
)
# Direkter Weg zum Gemma-Worker (llama-server auf dem Mac mini). Der
# Anonymization-Service auf dem Desktop bedient ausschliesslich das dort
# geladene Qwen-Modell und ignoriert das "model"-Feld der Anfrage —
# engine="gemma" lief darueber bis 2026-07-29 stillschweigend auf Qwen.
# Ist diese URL gesetzt, gehen Gemma-Extraktionen als OpenAI-Chat-Calls
# direkt zum Mac (Standard: derselbe llama-server wie der Intake-Tagger).
GEMMA_BACKEND_URL = os.getenv(
    "ANONYMIZATION_GEMMA_URL", os.getenv("GEMMA_TAGGER_URL", "")
).strip()
GEMMA_BACKEND_API_KEY = os.getenv("GEMMA_TAGGER_API_KEY", "").strip()
ANONYMIZATION_ENGINE_DEFAULT = os.getenv(
    "ANONYMIZATION_ENGINE_DEFAULT", "qwen"
).strip().lower()
SUPPORTED_ANONYMIZATION_ENGINES = {"gemma", "qwen"}
# Frühere Engine-Namen: "qwen_flair" lief seit dem Wechsel des Desktop-Backends
# auf llama_server faktisch ohne Flair (Hint-Fetch brach bei backend!="flair" ab),
# "flair_presidio" ist mit dem Flair-Backend entfernt worden (settled: Qwen).
LEGACY_ANONYMIZATION_ENGINE_ALIASES = {"qwen_flair": "qwen", "flair_presidio": "qwen"}
ANONYMIZATION_EXTRACTION_MODE_DEFAULT = os.getenv(
    "ANONYMIZATION_EXTRACTION_MODE", "staged"
).strip().lower()
SUPPORTED_EXTRACTION_MODES = {"staged", "single"}


def _entity_counts(entities: dict) -> dict[str, int]:
    return {
        str(key): len(values)
        for key, values in entities.items()
        if isinstance(values, list) and values
    }


def resolve_extraction_mode(requested_mode: Optional[str]) -> str:
    mode = (requested_mode or ANONYMIZATION_EXTRACTION_MODE_DEFAULT).strip().lower()
    if mode in SUPPORTED_EXTRACTION_MODES:
        return mode
    print(
        f"[WARN] Unsupported extraction mode '{requested_mode}', "
        f"falling back to default '{ANONYMIZATION_EXTRACTION_MODE_DEFAULT}'"
    )
    if ANONYMIZATION_EXTRACTION_MODE_DEFAULT in SUPPORTED_EXTRACTION_MODES:
        return ANONYMIZATION_EXTRACTION_MODE_DEFAULT
    return "staged"


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
- questionnaire answers can list SEVERAL relatives in one line, often with an age
  in brackets - return every name in such a line, not just the first:
  "Nennen Sie bitte Familiennamen ... des Familienangehoerigen"
  "Antwort: Meier Fatima (42), Nour Said (34)" -> both names
- people who appear only as figures of public life in country-conditions
  reasoning are NOT parties and must be omitted: heads of state, ministers,
  militia or party leaders, officials of international organisations, authors
  of cited studies and reports, journalists. They are named to describe the
  situation in the country of origin, not to identify anybody in this case.
- include names near role/signature markers (Anhörender Entscheider, Sachbearbeiter, Unterzeichner, Im Auftrag, gez., Unterschrift), but return only the name, never the role word
- include a single-token name only when it is clearly a signer or person mention; otherwise prefer omitting uncertain single words
- never return role labels by themselves (e.g. "Mutter", "Sachbearbeiter", "Unterzeichner")
- never return organizations, courts, authorities, addresses, cities, countries, ethnicities, religions, document titles, legal citations, IDs, numbers, or page/footer text
- never return OCR/debug/schema garbage or strings containing many digits/non-name tokens
- if unsure whether something is a person name, omit it
- deduplicate exact duplicates

Document:
"""

ADDRESSES_EXTRACTION_PROMPT_PREFIX = """Extract the CURRENT RESIDENTIAL ADDRESS of the persons in this German legal document.
Return valid JSON only with exactly:
{"streets":[], "postal_codes":[], "cities":[]}

Scope is deliberately narrow. Only the postal address where somebody CURRENTLY
LIVES or RECEIVES MAIL counts. It normally appears once, in the header block of
the first pages, after a label such as "wohnhaft", "Anschrift", "Wohnanschrift",
"Zustellanschrift", "vertreten durch" or in a ZUE/accommodation line.

Return:
- streets: street + house number from such an address block
- postal_codes: 5-digit postal code from such an address block
- cities: the city that belongs to such an address block

Do NOT return any other place. Specifically NOT:
- the country, region, province or city of origin ("geb. in Kabul",
  "stammt aus Herat") - the reasoning needs those and they stay
- places of transit, entry or former stay ("eingereist über München",
  "in der Nähe von Sofia", "zuletzt in Kinshasa gelebt")
- any place named in country-conditions reasoning, situation reports
  (Lagebericht), security assessments, statistics or cited case law
- seats of courts and authorities ("VG Augsburg", "Bundesamt ... Nürnberg")
- countries and regions on their own
- page or sheet references ("Seite 2", "--- Seite 3 ---", "Seite 2 von 12", "Bl. 64") or other page/footer noise - "Seite"/"Page"/"Bl." + number is a page reference, not a street

If the chunk contains no such address block, return empty lists.
Prioritize exact surface forms from the text and deduplicate exact duplicates.

Document:
"""

BIRTH_IDS_EXTRACTION_PROMPT_PREFIX = """Extract birth details and personal document IDs from this German legal document.
Return valid JSON only with exactly:
{"birth_dates":[], "birth_places":[], "azr_numbers":[], "aufenthaltsgestattung_ids":[], "bamf_geschaeftszeichen":[], "court_aktenzeichen":[], "personal_document_ids":[], "other_reference_numbers":[], "case_numbers":[]}

Rules:
- birth_dates: full date strings for person birth data (e.g. DD.MM.YYYY, "geb. am ...")
- if only a birth year is given, include the full surrounding birth phrase (e.g. "1992 geboren")
- birth_places: city/place directly tied to explicit birth context only (e.g. "geboren in", "Geburtsort", "geb. in")
- do NOT copy ordinary cities or residence locations into birth_places
- azr_numbers: AZR numbers only when explicitly labeled "AZR" or unmistakably in AZR context
- do NOT copy Aktenzeichen, Az., BAMF Geschäftsz., BAMF file numbers, or reference numbers into azr_numbers
- aufenthaltsgestattung_ids: IDs explicitly labeled as Aufenthaltsgestattung
- do NOT infer aufenthaltsgestattung_ids from fragments, case numbers, or unlabeled numeric strings
- bamf_geschaeftszeichen: BAMF Geschäftsz./Geschäftszeichen/file numbers only when explicitly labeled by BAMF context
- court_aktenzeichen: ONLY the Aktenzeichen of THIS proceeding - the number of the case the document itself belongs to (in the Rubrum/header or Betreff, labeled Az., Aktenzeichen, Geschäftsnummer)
- do NOT extract Aktenzeichen of OTHER court decisions cited as case law (e.g. "VG Berlin, Urteil vom 01.02.2020 - 12 K 34/19", "BVerwG 1 C 2.20", "EuGH C-123/45", juris, asyl.net) - published rulings are citation material the reader needs and must stay
- personal_document_ids: personal document IDs such as Dolmetscher-Nr, Aufenthaltsdokument IDs, pass/identity card numbers, or labeled document/person IDs
- other_reference_numbers: other explicitly labeled reference numbers that are not page numbers, paragraph numbers, dates, phone numbers, or OCR fragments
- case_numbers: legacy alias; leave empty unless no more specific field fits
- do NOT include court citations/references (ECLI, BVerwG/BVerfG/VG/OVG Az., §/Art. citations)
- do NOT include bare D#### page/document markers unless explicitly labeled as a personal/document ID
- if unsure whether a value is an identifier, omit it
- deduplicate exact duplicates

Document:
"""

SINGLE_PASS_EXTRACTION_PROMPT_PREFIX = """/no_think
Extract all personally identifying information from this German legal document.
Return valid JSON only with exactly:
{
  "names": [],
  "birth_dates": [],
  "birth_places": [],
  "streets": [],
  "postal_codes": [],
  "cities": [],
  "azr_numbers": [],
  "aufenthaltsgestattung_ids": [],
  "bamf_geschaeftszeichen": [],
  "court_aktenzeichen": [],
  "personal_document_ids": [],
  "other_reference_numbers": [],
  "case_numbers": []
}

Rules:
- Use exact surface forms from the text.
- Extract natural person names, birth data, address data, AZR numbers, Aufenthaltsgestattung IDs, BAMF Geschäftszeichen, court Aktenzeichen, and personal document IDs.
- Do not include legal citations, court names, authorities as person names, countries, religions, ordinary legal terms, or page/footer noise.
- Put BAMF Geschäftsz./Geschäftszeichen into bamf_geschaeftszeichen, court/legal case numbers into court_aktenzeichen, personal IDs into personal_document_ids.
- Put only explicitly labeled residual reference numbers into other_reference_numbers. Leave case_numbers empty unless no more specific field fits.
- Do not include bare D#### page/document markers unless explicitly labeled as a personal/document ID.
- If unsure, omit the item.
- Deduplicate exact duplicates.

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
    "bamf_geschaeftszeichen",
    "court_aktenzeichen",
    "personal_document_ids",
    "other_reference_numbers",
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
    "bamf_geschaeftszeichen": {"type": "array", "items": {"type": "string"}},
    "court_aktenzeichen": {"type": "array", "items": {"type": "string"}},
    "personal_document_ids": {"type": "array", "items": {"type": "string"}},
    "other_reference_numbers": {"type": "array", "items": {"type": "string"}},
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
            "bamf_geschaeftszeichen",
            "court_aktenzeichen",
            "personal_document_ids",
            "other_reference_numbers",
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
BAMF_GESCHAEFTSZEICHEN_LABEL_STRIP_PATTERN = re.compile(
    r"(?i)^\s*(?:Gesch(?:äft|ae?ft|a?ft)szeichen|Gesch\.?-?Z\.?|Geschäftsz\.?|Geschaeftsz\.?|Geschaftsz\.?)\s*[:#-]?\s*"
)
COURT_AKTENZEICHEN_LABEL_STRIP_PATTERN = re.compile(
    r"(?i)^\s*(?:Az\.?|Aktenzeichen|Geschäftsnummer|Geschäftszeichen)\s*[:#-]?\s*"
)
# Zitier-Vorspann direkt vor einem Az-Label: "…, Urteil vom 12.03.2021, Az.: …"
# bzw. "U. v. / B. v. / Beschl. v." - dann gehoert das Az zu einer zitierten,
# veroeffentlichten Entscheidung, nicht zum eigenen Verfahren.
CITED_DECISION_CONTEXT_PATTERN = re.compile(
    r"(?i)(?:urteil|beschluss|beschl\.|entscheidung|gerichtsbescheid|[UB]\.)\s*"
    r"(?:vom|v\.)\s*\d{1,2}\.\d{1,2}\.\d{2,4}\s*[,;]?\s*$"
)
PERSONAL_DOCUMENT_ID_LABEL_STRIP_PATTERN = re.compile(
    r"(?i)^\s*(?:Dolmetscher(?:-Nr\.?|nummer)?|Pass(?:nummer|-Nr\.?)?|Ausweis(?:nummer|-Nr\.?)?|Dokument(?:nummer|-Nr\.?)?|ID(?:-Nr\.?)?)\s*[:#-]?\s*"
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
    if engine in LEGACY_ANONYMIZATION_ENGINE_ALIASES:
        resolved = LEGACY_ANONYMIZATION_ENGINE_ALIASES[engine]
        print(f"[WARN] Legacy anonymization engine '{engine}' requested, using '{resolved}'")
        return resolved
    if engine in SUPPORTED_ANONYMIZATION_ENGINES:
        return engine
    print(
        f"[WARN] Unknown anonymization engine '{engine}', "
        f"falling back to default '{ANONYMIZATION_ENGINE_DEFAULT}'"
    )
    resolved_default = LEGACY_ANONYMIZATION_ENGINE_ALIASES.get(
        ANONYMIZATION_ENGINE_DEFAULT, ANONYMIZATION_ENGINE_DEFAULT
    )
    if resolved_default in SUPPORTED_ANONYMIZATION_ENGINES:
        return resolved_default
    return "qwen"


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


def _extract_service_normalized_entities(
    data: dict[str, Any],
    context: str,
) -> dict[str, list[str]]:
    normalized_entities = data.get("normalized_entities")
    if isinstance(normalized_entities, dict):
        return _normalize_extraction_entities(normalized_entities)

    parsed_from = data.get("parsed_from")
    parse_ok = data.get("parse_ok")
    available_fields = sorted(
        key
        for key in ("normalized_entities", "parsed_payload", "response", "thinking", "parse_error")
        if key in data
    )
    raise ValueError(
        f"{context} response did not include normalized_entities "
        f"(parse_ok={parse_ok!r}, parsed_from={parsed_from!r}, fields={available_fields})"
    )


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
        "bamf_geschaeftszeichen": PatternRecognizer(
            supported_entity="BAMF_GESCHAEFTSZEICHEN",
            supported_language="de",
            patterns=[
                Pattern(
                    "bamf_geschaeftszeichen_label",
                    r"(?i)\b(?:Gesch(?:äft|ae?ft|a?ft)szeichen|Gesch\.?-?Z\.?|Geschäftsz\.?|Geschaeftsz\.?|Geschaftsz\.?)\s*[:#-]?\s*[A-Z0-9][A-Z0-9./\-\s]{3,30}\b",
                    0.85,
                )
            ],
        ),
        "court_aktenzeichen": PatternRecognizer(
            supported_entity="COURT_AKTENZEICHEN",
            supported_language="de",
            patterns=[
                Pattern(
                    "court_aktenzeichen_label",
                    r"(?i)\b(?:Az\.?|Aktenzeichen|Geschäftsnummer)\s*[:#-]?\s*(?:[A-Z]{0,4}\s*\d{1,4}\s*[A-ZÄÖÜa-zäöüß]{0,8}\s*\d{1,5}/\d{2,4}|[A-Z]{1,4}\s*\d{1,5}/\d{2,4})\b",
                    0.8,
                )
            ],
        ),
        "personal_document_ids": PatternRecognizer(
            supported_entity="PERSONAL_DOCUMENT_ID",
            supported_language="de",
            patterns=[
                Pattern(
                    "personal_document_id_label",
                    r"(?i)\b(?:Dolmetscher(?:-Nr\.?|nummer)?|Pass(?:nummer|-Nr\.?)?|Ausweis(?:nummer|-Nr\.?)?|Dokument(?:nummer|-Nr\.?)?|ID(?:-Nr\.?)?)\s*[:#-]?\s*[A-Z0-9][A-Z0-9./\-]{3,}\b",
                    0.8,
                )
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

        for result in recognizers["bamf_geschaeftszeichen"].analyze(
            text=text, entities=["BAMF_GESCHAEFTSZEICHEN"], nlp_artifacts=None
        ):
            value = BAMF_GESCHAEFTSZEICHEN_LABEL_STRIP_PATTERN.sub(
                "", _span_value(result.start, result.end)
            )
            if value:
                entities["bamf_geschaeftszeichen"].append(value)

        for result in recognizers["court_aktenzeichen"].analyze(
            text=text, entities=["COURT_AKTENZEICHEN"], nlp_artifacts=None
        ):
            # Az. zitierter Urteile ("OVG Bautzen, Urteil vom 12.03.2021,
            # Az.: 5 A 1234/19") sind Arbeitsmaterial und bleiben im
            # anonymisierten Dokument (Jay, 2026-07-29). Nur das Az des
            # EIGENEN Verfahrens (Rubrum/Betreff, ohne Zitier-Vorspann)
            # wird geschwaerzt.
            prefix_window = text[max(0, result.start - 60) : result.start]
            if CITED_DECISION_CONTEXT_PATTERN.search(prefix_window):
                continue
            value = COURT_AKTENZEICHEN_LABEL_STRIP_PATTERN.sub(
                "", _span_value(result.start, result.end)
            )
            if value:
                entities["court_aktenzeichen"].append(value)

        for result in recognizers["personal_document_ids"].analyze(
            text=text, entities=["PERSONAL_DOCUMENT_ID"], nlp_artifacts=None
        ):
            value = PERSONAL_DOCUMENT_ID_LABEL_STRIP_PATTERN.sub(
                "", _span_value(result.start, result.end)
            )
            if value:
                entities["personal_document_ids"].append(value)

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

    page_header_pattern = r"(?m)^--- (?:Page|Seite) \d+ ---\s*$"
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


_PAGE_MARKER_RE = re.compile(r"(?im)^[ \t]*-{2,}[ \t]*(?:Seite|Page)[ \t]+(\d+)[ \t]*-{2,}[ \t]*$")


def _page_marker_anchor(page_body: str) -> str:
    """A stable substring from a page body for relocating it in anonymized text.

    Picks the longest run of letters/spaces with no digits from the page head -
    such runs almost never contain the names/dates/addresses that anonymization
    rewrites, so they survive verbatim and make a reliable insertion anchor.
    """
    best = ""
    for run in re.findall(r"[^\W\d_][^\d\n]{19,}", page_body[:800], re.UNICODE):
        run = run.strip()
        if len(run) > len(best):
            best = run
    return best[:60]


def _restore_missing_page_markers(source_text: str, anonymized_text: str) -> str:
    """Re-insert `--- Seite N ---` markers dropped during anonymization.

    Insert-only and order-preserving: it never removes or rewrites existing text,
    so it cannot weaken redaction. A marker is only inserted when its page anchor
    is found in the anonymized text at/after the running cursor; otherwise skipped.
    """
    if not source_text or not anonymized_text:
        return anonymized_text
    src = list(_PAGE_MARKER_RE.finditer(source_text))
    if not src:
        return anonymized_text
    present = {int(m.group(1)) for m in _PAGE_MARKER_RE.finditer(anonymized_text)}

    anchors: Dict[int, str] = {}
    for idx, match in enumerate(src):
        num = int(match.group(1))
        if num in present:
            continue
        body_start = match.end()
        body_end = src[idx + 1].start() if idx + 1 < len(src) else len(source_text)
        anchors[num] = _page_marker_anchor(source_text[body_start:body_end])

    if not anchors:
        return anonymized_text

    # Preferred path: when a marker was clobbered by an entity replacement, the line
    # survives as residue like "--- [ADRESSE] ---" at the EXACT page boundary. If the
    # residue lines match the missing markers 1:1 in order, rewrite them in place -
    # positionally exact, unlike the anchor heuristic below.
    residue_re = re.compile(r"(?m)^[ \t]*-{2,}[ \t]*\[[A-Z\-]+\][ \t]*-{2,}[ \t]*$")
    residues = list(residue_re.finditer(anonymized_text))
    missing_sorted = sorted(anchors)
    if residues and len(residues) == len(missing_sorted):
        result = anonymized_text
        for match, num in zip(reversed(residues), reversed(missing_sorted)):
            result = result[: match.start()] + f"--- Seite {num} ---" + result[match.end():]
        print(
            f"[INFO] Restored {len(missing_sorted)} clobbered page marker(s) in place: {missing_sorted}"
        )
        return result

    result = anonymized_text
    cursor = 0
    inserted: List[int] = []
    for num in sorted(anchors):
        anchor = anchors[num]
        if not anchor:
            continue
        pos = result.find(anchor, cursor)
        if pos == -1:
            continue
        line_start = result.rfind("\n", 0, pos) + 1
        marker_line = f"--- Seite {num} ---\n"
        result = result[:line_start] + marker_line + result[line_start:]
        cursor = line_start + len(marker_line) + len(anchor)
        inserted.append(num)

    if inserted:
        print(
            f"[INFO] Restored {len(inserted)} dropped page marker(s) in anonymized text (anchor heuristic): {inserted}"
        )
    return result


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


# A bare page/sheet reference is never PII. Extraction models intermittently return
# them as entities anyway (observed: "Seite 2"/"Seite 3" extracted as streets from a
# letterhead address block, which then redacted our "--- Seite N ---" page markers and
# desynced citation page numbering). Prompts ask models not to do this; this filter is
# the deterministic guarantee. Matches ONLY strings that are entirely a page reference.
_PAGE_REFERENCE_ENTITY_RE = re.compile(
    r"^(?:-{2,}\s*)?(?:Seite|Page|Bl\.?)\s*:?\s*\d+(?:\s+von\s+\d+)?(?:\s*-{2,})?$",
    re.IGNORECASE,
)


def _filter_page_reference_artifacts(entities: dict) -> dict:
    for key, values in entities.items():
        if not isinstance(values, list):
            continue
        kept: list = []
        for raw in values:
            if isinstance(raw, str) and _PAGE_REFERENCE_ENTITY_RE.match(raw.strip()):
                print(f"[INFO] Dropped page-reference artifact from '{key}': {raw.strip()!r}")
                continue
            kept.append(raw)
        entities[key] = kept
    return entities


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
    tightened = _filter_page_reference_artifacts(tightened)
    return _dedupe_entity_lists(tightened)


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


SENSITIVE_IDENTIFIER_KEYS = [
    "azr_numbers",
    "aufenthaltsgestattung_ids",
    "bamf_geschaeftszeichen",
    "court_aktenzeichen",
    "personal_document_ids",
]


def _looks_like_legacy_case_number(value: str) -> bool:
    candidate = _normalize_identifier_value(value)
    if not candidate:
        return False
    if re.fullmatch(r"(?i)d\d{3,5}", candidate):
        return False
    if re.search(r"\d{1,6}/\d{2,4}", candidate):
        return True
    if "-" in candidate and _digit_count(candidate) >= 5:
        return True
    if re.search(r"[A-Za-zÄÖÜäöüß]", candidate) and _digit_count(candidate) >= 6:
        return True
    return False


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
            if not _looks_like_legacy_case_number(candidate):
                continue
            cleaned_case_numbers.append(candidate)
        entities["case_numbers"] = cleaned_case_numbers

    normalized_sensitive_ids: set[str] = set()
    for key in SENSITIVE_IDENTIFIER_KEYS:
        cleaned_values: list[str] = []
        raw_values = entities.get(key, [])
        if not isinstance(raw_values, list):
            raw_values = []
        for raw in raw_values:
            if not isinstance(raw, str):
                continue
            candidate = _normalize_identifier_value(raw)
            if not candidate:
                continue
            if key == "azr_numbers" and _digit_count(candidate) < 6:
                continue
            if key == "aufenthaltsgestattung_ids" and (
                candidate.endswith("-")
                or (_digit_count(candidate) < 6 and not re.search(r"[A-Za-z]", candidate))
            ):
                continue
            if key == "bamf_geschaeftszeichen" and _digit_count(candidate) < 4:
                continue
            if key == "court_aktenzeichen" and not re.search(r"\d{1,6}/\d{2,4}", candidate):
                continue
            if key == "personal_document_ids" and (
                re.fullmatch(r"(?i)d\d{3,5}", candidate)
                or (_digit_count(candidate) < 4 and not re.search(r"[A-Za-z]", candidate))
            ):
                continue
            marker = candidate.casefold()
            if marker in normalized_sensitive_ids:
                continue
            normalized_sensitive_ids.add(marker)
            cleaned_values.append(candidate)
        entities[key] = cleaned_values

    if isinstance(entities.get("case_numbers"), list):
        entities["case_numbers"] = [
            value
            for value in entities.get("case_numbers", [])
            if isinstance(value, str)
            and _normalize_identifier_value(value).casefold() not in normalized_sensitive_ids
        ]

    normalized_case_numbers = {
        _normalize_identifier_value(value).casefold()
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
            if candidate.casefold() in normalized_case_numbers:
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
            if candidate.casefold() in normalized_case_numbers:
                continue
            if candidate.endswith("-"):
                continue
            if _digit_count(candidate) < 6 and not re.search(r"[A-Za-z]", candidate):
                continue
            cleaned_aufenthaltsgestattung_ids.append(candidate)
        entities["aufenthaltsgestattung_ids"] = cleaned_aufenthaltsgestattung_ids

    return _dedupe_entity_lists(entities)


def _stage_temperature(stage_name: str, is_gemma3: bool) -> float:
    """Beide Engines extrahieren mit Temperatur 0.0 (die _QWEN-Envs gelten
    trotz Namen auch fuer Gemma 4; nur der Gemma3-Legacy-Pfad hat eigene)."""
    if is_gemma3:
        if stage_name == "names":
            return _float_env("OLLAMA_NAMES_TEMP_GEMMA3", 0.0)
        if stage_name == "addresses":
            return _float_env("OLLAMA_ADDRESSES_TEMP_GEMMA3", 0.0)
        if stage_name == "birth_ids":
            return _float_env("OLLAMA_BIRTH_IDS_TEMP_GEMMA3", 0.0)
    if stage_name == "names":
        return _float_env("OLLAMA_NAMES_TEMP_QWEN", 0.0)
    if stage_name == "addresses":
        return _float_env("OLLAMA_ADDRESSES_TEMP_QWEN", 0.0)
    if stage_name == "birth_ids":
        return _float_env("OLLAMA_BIRTH_IDS_TEMP_QWEN", 0.0)
    return 0.0


PROMPT_DOCUMENT_FIRST = _bool_env("ANONYMIZATION_PROMPT_DOCUMENT_FIRST", False)
# Verworfen (2026-07-29): "kumulative Stufen" - spaetere Stufen pruefen die
# Kategorien der frueheren mit der Fundliste als Kontext nach. Messung auf dem
# 10-Dokumente-Korpus: 0 neue Namen in 34 Gelegenheiten, die Adress-Nachfunde
# waren durchweg Behoerdenadressen (verstoessen gegen die Nur-Wohnanschrift-
# Regel), +43 % Laufzeit. Details: Session-Protokoll 29.07.2026.


def _document_first_prompt(stage_prefix: str, chunk_text: str, hint: str) -> str:
    """Chunk-Text nach VORN, Stage-Anweisung ans Ende.

    Die drei Stage-Calls eines Chunks schicken denselben Text. Steht die
    Anweisung vorn (bisheriges Layout), unterscheiden sich die Prompts ab dem
    ersten Token und llama.cpp kann nichts wiederverwenden - der Chunk wird
    dreimal von Grund auf verarbeitet. Steht der Text vorn, ist er gemeinsamer
    Präfix: gemessen 2026-07-29 fielen 1960 von 1982 Prompt-Token in den Cache
    und die Prompt-Zeit von 14,3 s auf 0,7 s.

    Der Hinweis auf bereits gefundene Entitäten muss HINTER den Text, weil er
    je Stage verschieden ist und den gemeinsamen Präfix sonst zerschneidet.
    """
    instructions = stage_prefix
    for marker in ("\nDocument:\n", "\nDokument:\n"):
        if marker in instructions:
            instructions = instructions.split(marker)[0]
            break
    parts = ["Dokument:", chunk_text.strip(), ""]
    if hint:
        parts.append(hint.strip())
    parts.append(instructions.strip())
    return "\n".join(parts)


def _model_family(name: str) -> tuple[str, str]:
    """('gemma3:12b') -> ('gemma', '3'), ('gemma-4-12B-it-qat-…') -> ('gemma', '4'),
    ('qwen3.6-27b-mtp-vision') -> ('qwen', '3').

    Familie plus Hauptversion, damit der Abgleich zwischen angefordertem und
    tatsächlich antwortendem Modell nicht nur qwen/gemma trennt, sondern auch
    gemma3 von gemma4 — die Stage-Workarounds hängen an genau dieser Ziffer.
    """
    clean = (name or "").strip().lower()
    match = re.match(r"([a-z]+)[^0-9]*([0-9]+)", clean)
    if not match:
        return (clean, "")
    return (match.group(1), match.group(2))


def _stage_passes(stage_name: str, is_gemma3: bool) -> int:
    if is_gemma3 and stage_name == "names":
        return max(1, _int_env("OLLAMA_NAMES_PASSES_GEMMA3", 2))
    return 1


async def _extract_entities_via_gemma_backend(
    client: httpx.AsyncClient, payload: dict[str, Any]
) -> dict[str, Any]:
    """Extraktions-Call direkt an den llama-server des Mac mini.

    Uebersetzt den Ollama-Payload des Service-Pfads in einen OpenAI-Chat-Call
    und die Antwort zurueck in die Form, die der Anonymization-Service liefert
    (normalized_entities, prompt_eval_count, eval_count) — der Rest der
    Pipeline sieht keinen Unterschied. enable_thinking=false ueber
    chat_template_kwargs ist auf llama.cpp der einzige wirksame Schalter
    gegen Gemmas Reasoning; thinking/reasoning_effort/reasoning_budget werden
    dort stillschweigend ignoriert (gemessen 2026-07-28)."""
    options = payload.get("options") or {}
    fmt = payload.get("format")
    body: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": [{"role": "user", "content": payload.get("prompt") or ""}],
        "temperature": options.get("temperature", 0),
        "max_tokens": options.get("num_predict", 4096),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for key in ("top_k", "top_p", "repeat_penalty", "min_p"):
        if options.get(key) is not None:
            body[key] = options[key]
    body["response_format"] = (
        {
            "type": "json_schema",
            "json_schema": {"name": "entities", "strict": True, "schema": fmt},
        }
        if isinstance(fmt, dict)
        else {"type": "json_object"}
    )
    headers = {}
    if GEMMA_BACKEND_API_KEY:
        headers["Authorization"] = f"Bearer {GEMMA_BACKEND_API_KEY}"
    response = await client.post(
        f"{GEMMA_BACKEND_URL.rstrip('/')}/v1/chat/completions",
        json=body,
        headers=headers,
    )
    response.raise_for_status()
    data = response.json()
    content = (
        ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    ).strip()
    try:
        parsed = json.loads(content)
    except Exception:
        fallback = re.search(r"\{.*\}", content, re.S)
        parsed = json.loads(fallback.group(0)) if fallback else {}
    usage = data.get("usage") or {}
    return {
        "model": data.get("model"),
        "normalized_entities": parsed if isinstance(parsed, dict) else {},
        "prompt_eval_count": usage.get("prompt_tokens") or 0,
        "eval_count": usage.get("completion_tokens") or 0,
        "total_duration": 0,
    }


async def anonymize_document_text(
    text: str,
    document_type: str,
    engine: str,
    extract_chunk_pages: Optional[int] = None,
    extract_num_ctx: Optional[int] = None,
    extract_mode: Optional[str] = None,
    known_entities: Optional[Dict[str, List[str]]] = None,
) -> Optional[AnonymizationResult]:
    """Extract entities via desktop LLM, then apply regex anonymization locally.

    known_entities: überspringt die LLM-Extraktion komplett und anonymisiert
    in einem Rutsch mit den vom Aufrufer gelieferten Entitäten (Keys wie
    EXTRACTION_ENTITY_KEYS). Für große Akten, deren Beteiligte bekannt sind
    (Jay, 2026-07-07): erfasst dann nur die gelieferten Personen, dafür ohne
    stundenlange Chunk-Extraktion. Presidio-Musterregeln (AZR, IBAN etc.)
    laufen zusätzlich, lokale Augments ergänzen Rollen-/Feldfunde."""
    if known_entities:
        entities = {
            key: [str(v).strip() for v in (known_entities.get(key) or []) if str(v).strip()]
            for key in EXTRACTION_ENTITY_KEYS
        }
        rule_entities = _extract_presidio_rule_entities(text)
        rule_count = sum(len(v) for v in rule_entities.values() if isinstance(v, list))
        if rule_count:
            print(f"[INFO] Presidio rule extraction added {rule_count} candidates")
            entities = _merge_extraction_entities(entities, rule_entities)
        entities = augment_names_from_person_fields(entities, text)
        entities = _dedupe_entity_lists(entities)
        counts = _entity_counts(entities)
        print(f"[INFO] Known-entities anonymization, counts: {counts}")
        anonymized_text = await run_in_threadpool(
            apply_regex_replacements_parallel, text, entities
        )
        return AnonymizationResult(
            anonymized_text=anonymized_text,
            plaintiff_names=entities.get("names", []),
            birth_dates=entities.get("birth_dates", []),
            addresses=entities.get("streets", []) + entities.get("cities", []),
            confidence=1.0,
            original_text=text,
            processed_characters=len(text),
            extraction_inference_params={
                "engine": "known_entities",
                "rules": "presidio_pattern_recognizers",
                "entity_counts": counts,
            },
        )

    use_gemma_backend = engine == "gemma" and bool(GEMMA_BACKEND_URL)
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url and not use_gemma_backend:
        print("[WARNING] ANONYMIZATION_SERVICE_URL not configured")
        return None

    # Der Desktop-Service wird nur fuer den Qwen-Pfad gebraucht; der Mac-Worker
    # ist always-on und hat keinen Ready-Endpoint dieser Art.
    if not use_gemma_backend:
        await ensure_anonymization_service_ready()

    # Beide Engines teilen Prompts, Stages, Chunking, Hint-Logik, Parsing und
    # Presidio-Regeln komplett; hier unterscheiden sich nur Modellname und
    # Sampling-Parameter. Bei Stage-Temperatur 0.0 (greedy, beide Engines)
    # sind top_k/top_p ohnehin wirkungslos.
    model = GEMMA_MODEL
    top_k = None
    top_p = None
    min_p = None
    repeat_penalty = 1.0
    use_presidio_rules = _bool_env("ANONYMIZATION_USE_PRESIDIO_RULES", True)
    if engine == "qwen":
        model = QWEN_MODEL
        top_k = _int_env("OLLAMA_TOP_K_QWEN", 40)
        top_p = _float_env("OLLAMA_TOP_P_QWEN", 0.92)
        min_p = _optional_float_env("OLLAMA_MIN_P_QWEN")
        repeat_penalty = _float_env("OLLAMA_REPEAT_PENALTY_QWEN", 1.0)

    # "detected" wäre gelogen: das ist der KONFIGURIERTE Name (GEMMA_MODEL bzw.
    # QWEN_MODEL), nicht das Modell, das tatsächlich antwortet. Welches das ist,
    # meldet erst die Antwort des Service-Managers zurück — siehe served_model
    # weiter unten.
    is_gemma3 = (model or "").strip().lower().startswith("gemma3")
    num_ctx = _int_env("OLLAMA_NUM_CTX_DEFAULT", 32768)
    if is_gemma3:
        num_ctx = _int_env("OLLAMA_NUM_CTX_GEMMA3", 32768)
        print(
            f"[INFO] Gemma3-Workarounds aktiv, weil der konfigurierte Modellname "
            f"'{model}' mit 'gemma3' beginnt: format='json' statt JSON-Schema, "
            f"names-Stage mit {_int_env('OLLAMA_NAMES_PASSES_GEMMA3', 2)} Pässen"
        )

    if extract_num_ctx is not None and extract_num_ctx > 0:
        num_ctx = extract_num_ctx

    env_chunk_pages = _optional_positive_int_env("OLLAMA_EXTRACT_CHUNK_PAGES")
    env_chunk_chars = _optional_positive_int_env("OLLAMA_EXTRACT_CHUNK_CHARS")
    active_chunk_pages = extract_chunk_pages or env_chunk_pages or 0
    fallback_chunk_chars = env_chunk_chars or 18000
    if active_chunk_pages <= 0 and _split_text_into_pages(text):
        active_chunk_pages = _int_env("OLLAMA_AUTO_PAGE_CHUNK_PAGES", 2)

    extraction_mode = resolve_extraction_mode(extract_mode)
    stage_plans: list[dict[str, Any]] = []
    if extraction_mode == "single":
        single_keys = list(EXTRACTION_ENTITY_KEYS)
        single_format: str | dict[str, Any] = (
            "json" if is_gemma3 else _build_extraction_format_schema(single_keys)
        )
        single_temperature = (
            _float_env("OLLAMA_SINGLE_PASS_TEMP_GEMMA3", 0.0)
            if is_gemma3
            else _float_env("OLLAMA_SINGLE_PASS_TEMP_QWEN", 0.0)
        )
        stage_plans.append(
            {
                "name": "single",
                "keys": single_keys,
                "prompt_prefix": SINGLE_PASS_EXTRACTION_PROMPT_PREFIX,
                "format": single_format,
                "temperature": single_temperature,
                "passes": 1,
            }
        )
    else:
        for stage_spec in EXTRACTION_STAGE_SPECS:
            stage_name = stage_spec["name"]
            stage_keys = list(stage_spec["keys"])
            stage_format: str | dict[str, Any] = (
                "json" if is_gemma3 else _build_extraction_format_schema(stage_keys)
            )
            stage_temperature = _stage_temperature(stage_name, is_gemma3)
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
        "extraction_mode": extraction_mode,
        "staged_extraction": extraction_mode == "staged",
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
                f"[INFO] Chunked {extraction_mode} extraction enabled: chunks={len(chunks)} "
                f"chunk_pages={active_chunk_pages} chunk_chars={fallback_chunk_chars}"
            )
        target_url = (
            f"{GEMMA_BACKEND_URL.rstrip('/')}/v1/chat/completions"
            if use_gemma_backend
            else f"{service_url}/extract-entities"
        )
        print(
            f"[INFO] {extraction_mode.title()} extraction request "
            f"url={target_url} model={model} "
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
        served_model_seen: dict[str, str] = {}

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
                        hint = (
                            _build_known_entities_hint(stage_keys, merged_entities)
                            if sequential_page_accumulator
                            else ""
                        )
                        if PROMPT_DOCUMENT_FIRST:
                            prompt = _document_first_prompt(stage_prompt_prefix, chunk_text, hint)
                        else:
                            prompt = stage_prompt_prefix + hint + chunk_text

                        payload = _build_payload(prompt, stage_format, stage_temperature)
                        if chunk_mode or stage_passes > 1:
                            print(
                                f"[INFO] Extraction page={chunk_idx}/{len(chunks)} "
                                f"stage={stage_name} "
                                f"pass={pass_idx}/{stage_passes} "
                                f"payload_chars={len(chunk_text)} temp={stage_temperature}"
                            )
                        if use_gemma_backend:
                            data = await _extract_entities_via_gemma_backend(
                                client, payload
                            )
                        else:
                            response = await client.post(
                                f"{service_url}/extract-entities",
                                json=payload,
                            )
                            response.raise_for_status()
                            data = response.json()

                        # Welches Modell hat WIRKLICH geantwortet? llama-server
                        # ignoriert das "model"-Feld der Anfrage und bedient
                        # stets das geladene Modell. Ohne diese Prüfung ist die
                        # Engine-Wahl still wirkungslos: engine="gemma" wurde am
                        # 2026-07-28 über tausende Anfragen hinweg von Qwen
                        # beantwortet, ohne jeden Hinweis im Log.
                        served_model = data.get("model")
                        if served_model and served_model != served_model_seen.get("name"):
                            served_model_seen["name"] = served_model
                            if _model_family(served_model) != _model_family(model):
                                print(
                                    f"[WARNING] Engine '{engine}' fordert Modell "
                                    f"'{model}' an, geantwortet hat aber "
                                    f"'{served_model}' — die Engine-Wahl ist "
                                    f"wirkungslos, die Stage-Parameter "
                                    f"(Format, Pässe, Temperatur) passen nicht "
                                    f"zum tatsächlichen Modell"
                                )
                            else:
                                print(f"[INFO] Antwortendes Modell: {served_model}")

                        parsed_entities = _extract_service_normalized_entities(
                            data,
                            (
                                f"extraction page={chunk_idx}/{len(chunks)} "
                                f"stage={stage_name} pass={pass_idx}/{stage_passes}"
                            ),
                        )
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
        # In der Job-Zeile festhalten, welches Modell tatsächlich geantwortet hat,
        # nicht nur welches angefordert wurde.
        extraction_inference_params["served_model"] = served_model_seen.get("name")

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
        # Last gate before replacement: catches page-reference artifacts from ALL
        # sources (chunk extraction, presidio rules, hint merges), not just the
        # page-level tightening pass.
        entities = _filter_page_reference_artifacts(entities)
        entities = _dedupe_entity_lists(entities)

        filtered_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        print(f"[INFO] After local filters: {filtered_count} entities")
        counts = _entity_counts(entities)
        if counts:
            print(f"[INFO] Entity counts by category: {counts}")

        # CPU-gebundene Ersetzung nie auf dem Event-Loop: ein pathologischer
        # Begriff fror sonst die gesamte App ein (2026-07-10, >2h in einem re.sub).
        anonymized_text = await run_in_threadpool(
            apply_regex_replacements_parallel, text, entities
        )

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


ANONYMIZATION_NO_REPLACEMENT_DETAIL = (
    "Anonymisierung hat keine Ersetzungen vorgenommen — Ergebnis verworfen, "
    "bitte prüfen und erneut anonymisieren"
)


def _anonymization_gate_failed(
    original_text: str,
    anonymized_text: str,
    plaintiff_names: list,
    birth_dates: list,
    addresses: list,
) -> bool:
    """Null-Ersetzungs-Gate: True if a fresh anonymization run must be rejected.

    Triggers when (a) the anonymized text is empty, OR (b) it is byte-identical
    to the input text (nothing was actually redacted), OR (c) every extracted
    entity/replacement list is empty. Pure logic, no I/O -- callers must only
    invoke this for a FRESH anonymization run, not for the already-anonymized
    cache/skip branch.
    """
    if not anonymized_text:
        return True
    if anonymized_text == original_text:
        return True
    if not (plaintiff_names or birth_dates or addresses):
        return True
    return False


async def anonymize_document_record(
    db: Session,
    document: Document,
    *,
    force: bool = False,
    engine: Optional[str] = None,
    extract_chunk_pages: Optional[int] = None,
    extract_num_ctx: Optional[int] = None,
    extract_mode: Optional[str] = None,
    known_entities: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """Anonymize a stored document and persist OCR/anonymized text metadata."""

    resolved_engine = resolve_anonymization_engine(engine)

    if force and document.is_anonymized:
        print(f"[INFO] Force re-anonymization requested for document_id={document.id}")

    if document.is_anonymized and document.anonymization_metadata and not force:
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
                "birth_dates": document.anonymization_metadata.get("birth_dates", []),
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
    file_ext = Path(pdf_path).suffix.lower()
    is_image = file_ext in IMAGE_FILE_EXTENSIONS

    cached_text = load_document_text(document)
    if document.ocr_applied and cached_text:
        extracted_text = cached_text
        ocr_used = True
        print(
            f"[INFO] Using cached OCR text for document_id={document.id}: {len(extracted_text)} characters"
        )

    if not extracted_text:
        should_use_ocr = is_image or document.needs_ocr or check_pdf_needs_ocr(pdf_path)

        if should_use_ocr:
            print(
                f"[INFO] Document needs OCR (flag={document.needs_ocr}, image={is_image}). "
                "Skipping direct extraction."
            )
            extracted_text = None
        else:
            try:
                extracted_text = extract_pdf_text(
                    # include_page_headers: keep "--- Page N ---" markers so directly
                    # extracted (non-OCR) documents carry a page frame for citation
                    # verification, same as OCR'd documents do.
                    pdf_path, max_pages=50, include_page_headers=True
                )
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
        extract_mode=extract_mode,
        known_entities=known_entities,
    )
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Anonymization service unavailable. Please ensure it is running.",
        )

    anonymized_full_text = result.anonymized_text
    # Anonymization can drop "--- Seite N ---" page markers (observed with the
    # former flair service: pages 2-3 lost). The model reads THIS anonymized text
    # and the citation verifier checks page numbers against it, so missing markers
    # desync the page numbering between author and checker. Restore any markers
    # that exist in the source OCR text but were dropped.
    # Insert-only: never alters or removes content, so it cannot affect redaction.
    anonymized_full_text = _restore_missing_page_markers(
        extracted_text, anonymized_full_text
    )
    processed_chars = result.processed_characters
    remaining_chars = 0

    if _anonymization_gate_failed(
        extracted_text,
        anonymized_full_text,
        result.plaintiff_names,
        result.birth_dates,
        result.addresses,
    ):
        print(
            "[ERROR] Anonymization gate rejected empty/unchanged result "
            f"(document_id={document.id}, engine={resolved_engine})"
        )
        raise HTTPException(
            status_code=422,
            detail=ANONYMIZATION_NO_REPLACEMENT_DETAIL,
        )

    anonymized_filename = f"{document.id}.txt"
    anonymized_path = ANONYMIZED_TEXT_DIR / anonymized_filename
    try:
        ANONYMIZED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
        with open(anonymized_path, "w", encoding="utf-8") as f:
            f.write(anonymized_full_text)
    except Exception as e:
        print(f"[ERROR] Failed to write anonymized text to file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save anonymized text: {e}"
        )

    store_document_text(document, extracted_text)
    document.is_anonymized = True
    document.ocr_applied = ocr_used
    document.needs_ocr = False
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

    broadcast_documents_snapshot(db, "anonymize", {"document_id": str(document.id)}, owner_id=document.owner_id)

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


async def auto_anonymize_document_bg(
    document_id: uuid.UUID,
    owner_id: uuid.UUID,
    case_id: Optional[uuid.UUID],
) -> None:
    """Background OCR/anonymization for newly ingested auto-anonymize documents
    (Mandantenunterlagen, and by default Bescheid/Anhörung)."""

    db = SessionLocal()
    try:
        document = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.owner_id == owner_id,
                Document.case_id == case_id,
            )
            .first()
        )
        if not document:
            print(f"[AUTO ANON] Document not found: {document_id}")
            return
        if not should_auto_anonymize_category(document.category):
            return

        document.processing_status = "anonymizing"
        db.commit()
        broadcast_documents_snapshot(db, "auto_anonymize_started", {"document_id": str(document_id)}, owner_id=owner_id)

        await anonymize_document_record(db, document)
    except Exception as exc:
        db.rollback()
        status = getattr(exc, "status_code", None)
        detail = getattr(exc, "detail", str(exc))
        print(f"[AUTO ANON ERROR] document_id={document_id} status={status} detail={detail}")
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = "anon_failed"
            metadata = dict(document.anonymization_metadata or {})
            metadata["auto_anonymization_error"] = str(detail)
            metadata["auto_anonymization_failed_at"] = datetime.utcnow().isoformat()
            document.anonymization_metadata = metadata
            db.commit()
            broadcast_documents_snapshot(db, "auto_anonymize_failed", {"document_id": str(document_id)}, owner_id=owner_id)
    finally:
        db.close()


@router.post("/documents/{document_id}/anonymize")
@limiter.limit("100/hour")
async def anonymize_document_endpoint(
    request: Request,
    document_id: str,
    force: bool = Query(False),
    engine: Optional[str] = Query(None),
    extract_chunk_pages: Optional[int] = Query(None, ge=1, le=50),
    extract_num_ctx: Optional[int] = Query(None, ge=1024, le=131072),
    extract_mode: Optional[str] = Query(None, pattern="^(single|staged)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize a stored document and persist OCR text when OCR is required."""
    document = get_owned_document(db, current_user, document_id)

    return await anonymize_document_record(
        db,
        document,
        force=force,
        engine=engine,
        extract_chunk_pages=extract_chunk_pages,
        extract_num_ctx=extract_num_ctx,
        extract_mode=extract_mode,
    )


class AnonymizeJobRequest(BaseModel):
    document_id: str
    force: bool = False
    engine: Optional[str] = None
    extract_chunk_pages: Optional[int] = None
    extract_num_ctx: Optional[int] = None
    extract_mode: Optional[str] = None
    known_entities: Optional[Dict[str, List[str]]] = None


async def _execute_anonymize_request(body: AnonymizeJobRequest, db: Session, user: User) -> dict:
    """Shared core for the AnonymizeJob (job worker). Gibt ein schlankes
    Ergebnis ohne den anonymisierten Volltext zurück - der liegt ohnehin als
    Datei am Dokument (anonymized_text_path)."""
    document = get_owned_document(db, user, body.document_id)
    result = await anonymize_document_record(
        db,
        document,
        force=body.force,
        engine=body.engine,
        extract_chunk_pages=body.extract_chunk_pages,
        extract_num_ctx=body.extract_num_ctx,
        extract_mode=body.extract_mode,
        known_entities=body.known_entities,
    )
    slim = {k: v for k, v in result.items() if k != "anonymized_text"}
    slim["anonymized_text_length"] = len(result.get("anonymized_text") or "")
    return slim


@router.post("/documents/anonymize-jobs", status_code=202)
@limiter.limit("60/hour")
async def create_anonymize_job(
    request: Request,
    body: AnonymizeJobRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a background anonymization job (runs in the job worker)."""
    document = get_owned_document(db, current_user, body.document_id)
    job = AnonymizeJob(
        owner_id=current_user.id,
        case_id=document.case_id,
        status="queued",
        request_payload=body.model_dump(),
        result_payload={},
        updated_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job.to_dict()


@router.get("/documents/anonymize-jobs/{job_id}")
@limiter.limit("240/hour")
async def get_anonymize_job(
    request: Request,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid anonymize job id format")
    job = (
        db.query(AnonymizeJob)
        .filter(AnonymizeJob.id == job_uuid, AnonymizeJob.owner_id == current_user.id)
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="Anonymize job not found")
    return job.to_dict()


@router.post("/anonymize-file")
@limiter.limit("100/hour")
async def anonymize_uploaded_file(
    request: Request,
    document_type: str = Form(...),
    file: UploadFile = File(...),
    engine: Optional[str] = Query(None),
    extract_chunk_pages: Optional[int] = Query(None, ge=1, le=50),
    extract_num_ctx: Optional[int] = Query(None, ge=1024, le=131072),
    extract_mode: Optional[str] = Query(None, pattern="^(single|staged)$"),
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
                    # Keep page markers for the citation-verification page frame.
                    tmp_path, max_pages=50, include_page_headers=True
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
            extract_mode=extract_mode,
        )
        if result is None:
            raise HTTPException(
                status_code=503,
                detail="Anonymization service unavailable. Please ensure it is running.",
            )

        anonymized_full_text = result.anonymized_text
        processed_chars = result.processed_characters
        remaining_chars = 0

        if _anonymization_gate_failed(
            text_for_anonymization,
            anonymized_full_text,
            result.plaintiff_names,
            result.birth_dates,
            result.addresses,
        ):
            print(
                "[ERROR] Anonymization gate rejected empty/unchanged preview result "
                f"(filename={filename}, engine={resolved_engine})"
            )
            raise HTTPException(
                status_code=422,
                detail=ANONYMIZATION_NO_REPLACEMENT_DETAIL,
            )

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
