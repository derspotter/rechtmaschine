"""
Rechtmaschine Anonymization Service (Flair NER Version)

Fast, robust anonymization using Flair's German Legal NER model.
Processes FULL documents (not just first 2k chars).

Features:
- GPU acceleration (auto-detected)
- Batch processing for speed
- Case-inflected placeholders (des Klägers, dem Kläger)
- Consistent pseudonyms (PERSON_1, PERSON_2)
- Family relation detection (Ehemann, Kind, etc.)

Requirements:
    pip install flair fastapi uvicorn

Usage:
    python anonymization_service_flair.py
"""

from collections import OrderedDict
import os
from pathlib import Path
import re
from typing import Optional, Dict, List, Tuple, Any

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# Lazy load Flair to speed up startup
_tagger = None


def get_tagger():
    global _tagger
    if _tagger is None:
        import torch
        import flair

        # Enable GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            flair.device = device
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            flair.device = device
            print("[INFO] Using CPU (no CUDA available)")

        print("[INFO] Loading Flair NER model (first time only, ~30s)...")
        from flair.models import SequenceTagger

        _tagger = SequenceTagger.load("flair/ner-german-legal")
        _tagger.to(device)
        print("[INFO] Flair NER model loaded successfully")
    return _tagger


app = FastAPI(
    title="Rechtmaschine Anonymization Service (Flair)",
    description="Fast document anonymization using Flair German Legal NER",
    version="2.2.4",
)

ANONYMIZATION_API_KEY = os.getenv("ANONYMIZATION_API_KEY")
ANONYMIZE_PHONES = os.getenv("ANONYMIZE_PHONES", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ALLOWLIST_PATH_ENV = os.getenv("ANONYMIZATION_ALLOWLIST_PATH", "").strip()
try:
    ALLOWLIST_CONTEXT_WINDOW = int(
        os.getenv("ANONYMIZATION_ALLOWLIST_CONTEXT_WINDOW", "80")
    )
except ValueError:
    ALLOWLIST_CONTEXT_WINDOW = 80


class AnonymizationRequest(BaseModel):
    text: str
    document_type: str


class AnonymizationResponse(BaseModel):
    is_valid: bool = True
    invalid_reason: Optional[str] = None
    anonymized_text: str
    plaintiff_names: list[str]
    family_members: list[str] = []
    addresses: list[str] = []
    confidence: float


# Entity types to anonymize (from flair/ner-german-legal)
ANONYMIZE_TAGS = {
    "PER",  # Person (plaintiffs, family)
    "STR",  # Straße (street)
    "LDS",  # Landschaft (region) - optional
}

PERSON_TAGS = {"PER", "AN", "RR"}

# Entity types to PRESERVE (don't anonymize)
PRESERVE_TAGS = {
    "RR",  # Richter (Judge)
    "AN",  # Anwalt (Lawyer)
    "GRT",  # Gericht (Court)
    "ORG",  # Organisation
    "GS",  # Gesetz (Law)
    "RS",  # Rechtsprechung
}

# =============================================================================
# GERMAN CASE INFLECTION
# =============================================================================

# Patterns to detect grammatical case from preceding articles/prepositions
CASE_PATTERNS = {
    "genitiv": re.compile(
        r"\b(des|der|eines|einer|meines|meiner|seines|seiner|ihres|ihrer|"
        r"dieses|dieser|jenes|jener|welches|welcher)\s*$",
        re.IGNORECASE,
    ),
    "dativ": re.compile(
        r"\b(dem|der|einem|einer|meinem|meiner|seinem|seiner|ihrem|ihrer|"
        r"diesem|dieser|jenem|jener|welchem|welcher|"
        r"von|mit|bei|nach|aus|zu|seit|gegenüber)\s*$",
        re.IGNORECASE,
    ),
    "akkusativ": re.compile(
        r"\b(den|einen|meinen|seinen|ihren|diesen|jenen|welchen|"
        r"für|durch|gegen|ohne|um)\s*$",
        re.IGNORECASE,
    ),
}

# Placeholder inflections for each case
PLACEHOLDER_INFLECTIONS = {
    "person": {
        "nominativ": "Kläger",
        "genitiv": "Klägers",
        "dativ": "Kläger",
        "akkusativ": "Kläger",
    },
    "person_female": {
        "nominativ": "Klägerin",
        "genitiv": "Klägerin",
        "dativ": "Klägerin",
        "akkusativ": "Klägerin",
    },
    "family": {
        "nominativ": "Familienangehöriger",
        "genitiv": "Familienangehörigen",
        "dativ": "Familienangehörigen",
        "akkusativ": "Familienangehörigen",
    },
    "child": {
        "nominativ": "Kind",
        "genitiv": "Kindes",
        "dativ": "Kind",
        "akkusativ": "Kind",
    },
}


def detect_case(text: str, position: int) -> str:
    """Detect German grammatical case from context before the entity."""
    # Look at the 30 characters before the entity
    context_start = max(0, position - 30)
    context = text[context_start:position]

    for case_name, pattern in CASE_PATTERNS.items():
        if pattern.search(context):
            return case_name

    return "nominativ"  # Default


def get_inflected_placeholder(entity_type: str, case: str, index: int) -> str:
    """Get the correctly inflected placeholder."""
    inflections = PLACEHOLDER_INFLECTIONS.get(
        entity_type, PLACEHOLDER_INFLECTIONS["person"]
    )
    base = inflections.get(case, inflections["nominativ"])

    if index > 0:
        return f"[{base.upper()}_{index + 1}]"
    return f"[{base.upper()}]"


# =============================================================================
# FAMILY RELATION DETECTION
# =============================================================================

# Family relation keywords (German)
FAMILY_RELATIONS = {
    # Spouse
    "ehemann",
    "ehefrau",
    "ehegatte",
    "ehegattin",
    "ehepartner",
    "ehepartnerin",
    "gatte",
    "gattin",
    "mann",
    "frau",
    "lebensgefährte",
    "lebensgefährtin",
    "lebenspartner",
    "lebenspartnerin",
    "partner",
    "partnerin",
    # Children
    "sohn",
    "tochter",
    "kind",
    "kinder",
    "söhne",
    "töchter",
    # Parents
    "vater",
    "mutter",
    "eltern",
    "elternteil",
    # Siblings
    "bruder",
    "schwester",
    "geschwister",
    "brüder",
    "schwestern",
    "familie",
    # Extended family
    "onkel",
    "tante",
    "neffe",
    "nichte",
    "cousin",
    "cousine",
    "großvater",
    "großmutter",
    "opa",
    "oma",
    "enkel",
    "enkelin",
    "schwager",
    "schwägerin",
    "schwiegervater",
    "schwiegermutter",
    "schwiegersohn",
    "schwiegertochter",
}

# Shared name patterns (handles Title Case + ALL CAPS surnames)
NAME_COMPONENT_REGEX = r"(?:[A-ZÄÖÜ][a-zäöüß]+|[A-ZÄÖÜ]{2,})"
NAME_COMPOUND_REGEX = (
    NAME_COMPONENT_REGEX + r"(?:[\s\-\'’]" + NAME_COMPONENT_REGEX + r")*"
)
NAME_COMPOUND_INLINE_REGEX = (
    NAME_COMPONENT_REGEX + r"(?:[ \t\-\'’]" + NAME_COMPONENT_REGEX + r")*"
)
NAME_TOKEN_RE = re.compile(r"[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\'’\-]{1,}")
ALL_CAPS_NAME_RE = re.compile(r"^[A-ZÄÖÜ][A-ZÄÖÜß\'’\-]{1,}$")
TITLECASE_NAME_RE = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:[\-\'’][A-ZÄÖÜ][a-zäöüß]+)*$")
NAME_STOPWORDS = {
    "BAMF",
    "BUNDESAMT",
    "BVERWG",
    "OVG",
    "VG",
    "EGMR",
    "EU",
    "UN",
    "EUAA",
    "UNHCR",
    "AZ",
    "NR",
    "GMBH",
    "AG",
    "KG",
    "ORT",
    "DR",
    "PRO",
    "SIE",
    "IHR",
    "IHRE",
    "IHREN",
    "IHRER",
    "IHREM",
    "IHNEN",
    "SEIN",
    "SEINE",
    "SEINER",
    "SEINEM",
    "SEINEN",
    "ER",
    "ES",
    "WIR",
    "UNS",
    "UNSER",
    "UNSERE",
    "UNSERER",
    "UNSEREM",
    "UNSERN",
    "SCHUTZ",
    "URTEIL",
    "BESCHLUSS",
    "BESCHL",
    "BESCHLUS",
    "BESCHLUSSV",
    "BESCHLUSSVOM",
    "BESCHL.V",
    "GRIECHENLAND",
    "DEUTSCHLAND",
    "BUNDESREPUBLIK",
    "SYRIEN",
    "IRAK",
    "IRAN",
    "AFGHANISTAN",
    "PALASTINA",
    "PALÄSTINA",
    "TÜRKEI",
    "TUERKEI",
    "NOTIZ",
    "BERICHT",
    "AUSKUNFT",
    "LANDERINFORMATION",
    "LÄNDERINFORMATION",
    "LAENDERINFORMATION",
    "LÄNDERINFORMATIONSBLATT",
    "LAENDERINFORMATIONSBLATT",
    "STAATENDOKUMENTATION",
    "BFA",
    "IOM",
    "PERSONEN",
    "PERSON",
    "FLÜCHTLINGE",
    "FLUECHTLINGE",
    "ZENTRALE",
    "HAUSANSCHRIFT",
    "BRIEFANSCHRIFT",
    "INTERNET",
    "BANKVERBINDUNG",
    "KONTOINHABER",
    "DIENSTSITZ",
    "MIGRATION",
    "MIGRANTEN",
}
ADDRESS_WORD_PATTERN = re.compile(
    r"\b(?:str(?:a(?:ß|ss|be|ble|le))|straße|strasse|str\.|weg|platz|allee|gasse|ring|damm|ufer|hof|bruch)\b",
    re.IGNORECASE,
)


# Pattern to find family relations followed by names
FAMILY_PATTERN = re.compile(
    r"\b(sein[es]?|ihr[es]?|der|die|dessen|deren)\s+"
    r"(" + "|".join(FAMILY_RELATIONS) + r")\b"
    r"(?:\s+(?:des|der|dem|den))?\s*"
    r"(" + NAME_COMPOUND_REGEX + r")?",
    re.IGNORECASE,
)

# Direct family mention pattern
# Note: Use inline (?i:...) for case-insensitive relation matching,
# but keep name matching case-sensitive (names must start uppercase)
DIRECT_FAMILY_PATTERN = re.compile(
    r"\b((?i:" + "|".join(FAMILY_RELATIONS) + r"))\s+"
    r"(" + NAME_COMPOUND_REGEX + r")"
)


# =============================================================================
# REGEX PATTERNS FOR PII
# =============================================================================

ID_SCAN_CHARS = 220
BAMF_ID_SCAN_CHARS = 120

DOB_PATTERN = re.compile(r"\b(\d{1,2})\.\s*(\d{1,2})\.\s*(19|20)\d{2}\b")
DOB_CUE_PATTERN = re.compile(
    r"\b(geboren\s+am|geb\.|geb\s|geb\.?\s*datum|Geburtsdatum|Jahrgang)\s*:?\s*"
    r"(\d{1,2}\.\s*\d{1,2}\.\s*(?:19|20)\d{2})",
    re.IGNORECASE,
)
DOB_CONTEXT_PATTERN = re.compile(
    r"\b(geboren\s+am|geb\.?\s*am|geb\.?\s*datum|Geburtsdatum|Jahrgang|b\.\s*[A-ZÄÖÜ])",
    re.IGNORECASE,
)
DOB_CONTEXT_WINDOW = 80
# Fixed: Allow comma/space before PLZ (same line or single line break)
PLZ_CITY_PATTERN = re.compile(
    r"(?:,[ \t]*)?(\d{5})(?:[ \t]+|[ \t]*\n[ \t]*)"
    r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\u00c0-\u017f\-]+"
    r"(?:[ \t]+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\u00c0-\u017f\-]+)?)\b"
)
ADDRESS_PATTERN = re.compile(
    r"\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\u00c0-\u017f\-]+"
    r"(?:[ \t]+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\u00c0-\u017f\-]+)*[ \t]*"
    r"(?:stra(?:ße|sse)|stra[ \t]*b[lI]?e|stra[ \t]*le|str\.|weg|platz|allee|gasse|ring|damm|ufer|hof|bruch))"
    r"(?:[ \t]*[.,]?[ \t]*)?(\d+[ \t]*[a-zA-Z]?)\b",
    re.IGNORECASE,
)
ADDRESS_PREFIX_PATTERN = re.compile(
    r"(?i)\b(?:straße|str\.|strasse|stra[ \t]*b[lI]?e|stra[ \t]*le)\b\s+"
    r"(?-i:[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+"
    r"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+){0,5})"
    r"(?:[^\n]{0,30}?\b\d{1,4}[a-zA-Z]?\b)?"
)
ADDRESS_CUE_PATTERN = re.compile(
    r"\b(wohnhaft\s+in|Adresse|Anschrift|wohnt\s+in)\s*:?\s*"
    r"([A-ZÄÖÜ][a-zäöüß\s]+\d+[a-zA-Z]?(?:\s*,\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+)?)",
    re.IGNORECASE,
)
ADDRESS_LABEL_LINE_PATTERN = re.compile(
    r"(?im)^\s*(Adresse|Anschrift|Wohnanschrift|Postanschrift|Hausanschrift)\s*:?\s*.*$"
)

AKTENZEICHEN_PATTERN = re.compile(
    r"(?<!\w)("
    r"Aktenz(?:ei|el)chen"
    r"|Gesch\.?\s*-?\s*zeichen"
    r"|Geschäftszeichen"
    r"|Geschaeftszeichen"
    r"|Geschaftszeichen"
    r"|Gesch\.?\s*-?\s*Z\.?"
    r"|Az\.?"
    r"|AZR\s*-?\s*Nummer\(n\)?"
    r")(?!\w)"
    r"(?:\s+(?:des|der|vom)\b(?:\s+[A-Za-zÄÖÜäöüß]+){0,3})?"
    r"\s*[:\-]?\s*"
    r"([A-Za-z0-9./\-–]*\d[A-Za-z0-9./\-–]*)",
    re.IGNORECASE,
)

DOB_BLOCK_START_PATTERN = re.compile(
    r"\b(Geburtsdatum|geb\.?\s*datum|geboren\s+am|geb\.?\s*am|geb\.|Jahrgang|Vorname\s*/\s*Name)\b",
    re.IGNORECASE,
)
DOB_BLOCK_END_PATTERN = re.compile(
    r"\b(Anlagen|Aktenzeichen|Bescheid|Sehr\s+geehrte|Ort,?Datum|Eingangsdatum|Hausanschrift|Seite\s+\d+)\b",
    re.IGNORECASE,
)
AKTENZEICHEN_CUE_PATTERN = re.compile(
    r"\b(Aktenz(?:ei|el)chen|Geschäftszeichen|Geschaeftszeichen|Geschaftszeichen|"
    r"Gesch\.?\s*-?\s*Z\.?|AZR\s*-?\s*Nummer\(n\)?|AZR|AZ|Ihr(?:e)?\s+Zeichen|"
    r"Mein\s+Zeichen)\b",
    re.IGNORECASE,
)
AKTENZEICHEN_VALUE_PATTERN = re.compile(r"\b\d{6,}\s*-\s*\d{1,}\b")
ID_NUMBER_PATTERN = re.compile(r"\b\d{6,}\b")
AZR_NUMBER_PATTERN = re.compile(r"\b\d{9,}\b")
AZR_NUMBER_FUZZY_PATTERN = re.compile(r"\d(?:[\s\-]*\d){8,}")
BAMF_CUE_PATTERN = re.compile(
    r"\b(Bundesamt|BAMF|Geschäftszeichen|Geschaeftszeichen|Geschaftszeichen)\b",
    re.IGNORECASE,
)
PHONE_CUE_PATTERN = re.compile(
    r"\b(Tel|Telefon(?:nummer)?|Fax(?:nummer)?)\b\.?",
    re.IGNORECASE,
)
PHONE_NUMBER_PATTERN = re.compile(r"\+?\d[\d\s()./-]{5,}\d")
PHONE_LINE_PATTERN = re.compile(r"^\s*\+\d[\d\s()./-]{5,}\d", re.MULTILINE)
PHONE_CONTEXT_PATTERN = re.compile(
    r"\b(tel|telefon|fax|durchwahl|mobil|handy)\b", re.IGNORECASE
)
PHONE_GLOBAL_PATTERN = re.compile(
    r"(?:\+\d[\d\s()./-]{6,}\d|\((?:\d[\d\s]{1,6}\d)\)\s*\d[\d\s-]{2,}\d|\b0\d{1,4}(?:[\s./-]+\d){2,})"
)
BANK_CONTEXT_PATTERN = re.compile(
    r"\b(iban|bic|bankverbindung|kontoinhaber|bundeskasse)\b", re.IGNORECASE
)
PHONE_SKIP_CONTEXT_PATTERN = re.compile(
    r"\b(iban|bic|bankverbindung|kontoinhaber|aktenz(?:ei|el)chen|geschäftszeichen|"
    r"geschaeftszeichen|geschaftszeichen|azr|az)\b",
    re.IGNORECASE,
)
PERSON_CONTEXT_PATTERN = re.compile(
    r"\b(herrn?|frau|kläger(?:in)?|antragsteller(?:in)?|beschwerdeführer(?:in)?|"
    r"geb\.?|geboren|vorname|nachname|familienname|name|rechtsanwalt|richter|"
    r"dolmetscher|protokollführer|im auftrag|familie|glaubensschwester|"
    r"glaubensbruder)\b",
    re.IGNORECASE,
)
LAST_FIRST_CONTEXT_PATTERN = re.compile(
    r"\b(asylverfahren|vorname|nachname|familienname|name|antragsteller|kläger|"
    r"beschwerdeführer|personalien)\b",
    re.IGNORECASE,
)
SIGNATURE_TRIGGERS = {
    "im auftrag",
    "im auftrage",
    "mit freundlichen grüßen",
    "mit freundlichen grussen",
    "mit freundlichen gruessen",
    "mit freundlichen grußen",
}
SIGNATURE_FOLLOWUP_PATTERN = re.compile(
    r"\b(hausanschrift|briefanschrift|internet|zentrale|bankverbindung)\b",
    re.IGNORECASE,
)

NAME_FIELD_PATTERN = re.compile(
    r"\b(Vorname/NAME|Vorname|Name|Familienname|Nachname|Alias|Personalien)\b"
    r"\s*[:\-]?\s*"
    r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\'’\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\'’\-]+){0,3})"
)

RELATION_LABEL_PATTERN = re.compile(
    r"\b(Mutter|Vater|Eltern|Sohn|Tochter|Bruder|Schwester|Onkel|Tante)\b"
    r"\s*[:\-]?\s*"
    r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\'’\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\'’\-]+){0,3})",
    re.IGNORECASE,
)

CITATION_AUTHOR_PATTERN = re.compile(
    r"\b([A-ZÄÖÜ][a-zäöüß]{2,}),\s*[A-Z]\.",
)

# Fallback patterns for names NER might miss
WITH_NAME_PATTERN = re.compile(r"\bMit\s+(" + NAME_COMPOUND_REGEX + r")")
WITH_NAME_CUE_PATTERN = re.compile(
    r"\b(Glaubensschwester|Glaubensbruder|Schwester|Bruder|Freundin?|Bekannte?r|"
    r"Nachbar(?:in)?|Kollege(?:in)?|Partner(?:in)?|Ehemann|Ehefrau)\b",
    re.IGNORECASE,
)


TITLE_NAME_PATTERN = re.compile(
    r"\b(Herr[n]?|Frau|Kläger(?:in)?|Antragsteller(?:in)?|Beschwerdeführer(?:in)?)\s+"
    r"(" + NAME_COMPOUND_REGEX + r")",
)

TITLE_GLUE_PATTERN = re.compile(
    r"\b(Herr[n]?|Frau|Kläger(?:in)?|Antragsteller(?:in)?|Beschwerdeführer(?:in)?)"
    r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\'’\-]{2,})"
)
LAWYER_TITLE_PATTERN = re.compile(
    r"\b(Rechtsanwalt|Rechtsanwältin|Rechtsanwaeltin|Rechtsanwälte|RA)\b"
    r"(?:[ \t]*\n[ \t]*|[ \t]+)"
    r"(" + NAME_COMPOUND_INLINE_REGEX + r")"
)
KANZLEI_NAME_PATTERN = re.compile(
    r"\bKanzlei\b(?:[ \t]*\n[ \t]*|[ \t]+)(" + NAME_COMPOUND_INLINE_REGEX + r")"
)
LAST_FIRST_PATTERN = re.compile(
    r"\b(" + NAME_COMPONENT_REGEX + r")\s*,\s*(" + NAME_COMPOUND_REGEX + r")"
)
LIST_INLINE_PATTERN = re.compile(r"^\s*\d+\.\s*")
LIST_LINE_PATTERN = re.compile(r"^\s*\d+\.\s*$")


def normalize_name(name: str) -> str:
    """Normalize a name for comparison (lowercase, strip titles)."""
    name = name.strip().lower()
    # Remove common titles
    for title in ["herr", "herrn", "frau", "dr.", "dr", "prof.", "prof"]:
        if name.startswith(title + " "):
            name = name[len(title) + 1 :].strip()
    name = re.sub(r"\s+", " ", name)
    return name.strip(" ,.;:")


def resolve_allowlist_path(raw_path: str) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path
    return Path(__file__).resolve().parent / "flair_allowlist.txt"


def load_allowlist(path: Path) -> Tuple[List[str], List[re.Pattern], List[re.Pattern]]:
    names: List[str] = []
    context_patterns: List[re.Pattern] = []
    address_patterns: List[re.Pattern] = []

    if not path.exists():
        return names, context_patterns, address_patterns

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[WARNING] Failed to read allowlist {path}: {exc}")
        return names, context_patterns, address_patterns

    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.lower().startswith("context:"):
            context_entry = entry.split(":", 1)[1].strip()
            if not context_entry:
                continue
            try:
                context_patterns.append(re.compile(context_entry, re.IGNORECASE))
            except re.error:
                context_patterns.append(
                    re.compile(re.escape(context_entry), re.IGNORECASE)
                )
            continue
        if entry.lower().startswith("address:"):
            address_entry = entry.split(":", 1)[1].strip()
            if not address_entry:
                continue
            try:
                address_patterns.append(re.compile(address_entry, re.IGNORECASE))
            except re.error:
                address_patterns.append(
                    re.compile(re.escape(address_entry), re.IGNORECASE)
                )
            continue
        normalized = normalize_name(entry)
        if normalized:
            names.append(normalized)

    return names, context_patterns, address_patterns


ALLOWLIST_PATH = resolve_allowlist_path(ALLOWLIST_PATH_ENV)
(
    ALLOWLIST_NAMES,
    ALLOWLIST_CONTEXT_PATTERNS,
    ALLOWLIST_ADDRESS_PATTERNS,
) = load_allowlist(ALLOWLIST_PATH)


def is_allowlisted_name(
    name: str,
    text: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> bool:
    if not name:
        return False
    if not ALLOWLIST_NAMES and not ALLOWLIST_CONTEXT_PATTERNS:
        return False
    normalized = normalize_name(name)
    if not normalized:
        return False
    if normalized in ALLOWLIST_NAMES:
        return True
    if text is not None and start is not None and end is not None:
        context_start = max(0, start - ALLOWLIST_CONTEXT_WINDOW)
        context_end = min(len(text), end + ALLOWLIST_CONTEXT_WINDOW)
        context = text[context_start:context_end].lower()
        context_patterns = ALLOWLIST_CONTEXT_PATTERNS + ALLOWLIST_ADDRESS_PATTERNS
        if context_patterns and any(
            pattern.search(context) for pattern in context_patterns
        ):
            return True
    return False


def is_allowlisted_address(address_text: str) -> bool:
    if not address_text:
        return False
    return False


def get_line_context(text: str, start: int, end: int, include_prev: bool = True) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    if not include_prev:
        return text[line_start:line_end]
    prev_start = text.rfind("\n", 0, max(0, line_start - 1))
    if prev_start == -1:
        prev_start = 0
    else:
        prev_start += 1
    return text[prev_start:line_end]


def is_phone_context(text: str, start: int, end: int, include_prev: bool = True) -> bool:
    context = get_line_context(text, start, end, include_prev=include_prev)
    return bool(PHONE_CONTEXT_PATTERN.search(context))


def is_bank_context(text: str, start: int, end: int, include_prev: bool = True) -> bool:
    context = get_line_context(text, start, end, include_prev=include_prev)
    return bool(BANK_CONTEXT_PATTERN.search(context))


def normalize_signature_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"[.:]+$", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.lower()


def extract_signature_names(text: str) -> List[str]:
    if not text:
        return []
    names: List[str] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if normalize_signature_line(line) not in SIGNATURE_TRIGGERS:
            continue
        for look_ahead in range(idx + 1, min(idx + 4, len(lines))):
            candidate = lines[look_ahead].strip()
            if not candidate:
                continue
            if normalize_signature_line(candidate) in SIGNATURE_TRIGGERS:
                continue
            if candidate.lower().startswith("gez."):
                candidate = candidate[4:].strip()
            if any(char.isdigit() for char in candidate):
                continue
            if len(candidate) > 40:
                continue
            if not is_usable_person_name(candidate):
                continue
            names.append(candidate)
            break

    for idx, line in enumerate(lines):
        candidate = line.strip()
        if not candidate:
            continue
        if SIGNATURE_FOLLOWUP_PATTERN.search(candidate):
            continue
        if ":" in candidate:
            continue
        if any(char.isdigit() for char in candidate):
            continue
        if len(candidate) > 40:
            continue
        if not is_usable_person_name(candidate):
            continue
        for look_ahead in range(idx + 1, min(idx + 3, len(lines))):
            next_line = lines[look_ahead].strip()
            if not next_line:
                continue
            if SIGNATURE_FOLLOWUP_PATTERN.search(next_line):
                names.append(candidate)
                break
    return names


def has_person_context(text: str, start: int, end: int) -> bool:
    context = get_line_context(text, start, end, include_prev=True)
    return bool(PERSON_CONTEXT_PATTERN.search(context))


def has_list_context(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_prefix = text[line_start:start]
    if LIST_INLINE_PATTERN.match(line_prefix):
        return True
    prev_start = text.rfind("\n", 0, max(0, line_start - 1))
    prev_line = text[prev_start + 1 : line_start] if prev_start != -1 else text[:line_start]
    return bool(LIST_LINE_PATTERN.match(prev_line.strip()))


def is_probable_name_token(token: str) -> bool:
    if not token:
        return False
    if len(token) < 4:
        return False
    lowered = token.lower()
    if re.search(r"fl[uo0]ch?tling", lowered):
        return False
    if any(char.isdigit() for char in token):
        return False
    if token.upper() in NAME_STOPWORDS:
        return False
    if ALL_CAPS_NAME_RE.match(token):
        return True
    return bool(TITLECASE_NAME_RE.match(token))


def is_usable_person_name(name: str) -> bool:
    tokens = NAME_TOKEN_RE.findall(name)
    if not tokens:
        return False
    if ADDRESS_WORD_PATTERN.search(name):
        return False
    return any(is_probable_name_token(token) for token in tokens)


def is_usable_title_name(name: str) -> bool:
    if is_usable_person_name(name):
        return True
    tokens = NAME_TOKEN_RE.findall(name)
    if len(tokens) != 1:
        return False
    token = tokens[0]
    if len(token) < 3:
        return False
    if token.upper() in NAME_STOPWORDS:
        return False
    if ADDRESS_WORD_PATTERN.search(token):
        return False
    return bool(TITLECASE_NAME_RE.match(token) or ALL_CAPS_NAME_RE.match(token))


def is_safe_span_boundary(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def normalize_ocr_text(text: str) -> str:
    if not text:
        return text

    text = text.replace("\u03b2", "ß")
    text = text.replace("\uff08", "(").replace("\uff09", ")")
    def normalize_strasse_case(fragment: str) -> str:
        compact = re.sub(r"\s+", "", fragment)
        if compact.isupper():
            return "STRASSE"
        if compact[:1].isupper():
            return "Strasse"
        return "strasse"

    def replace_strasse_suffix(match: re.Match) -> str:
        prefix = match.group(1)
        suffix = match.group(2)
        return f"{prefix}{normalize_strasse_case(suffix)}"

    text = re.sub(
        r"(?i)\b([A-Za-zÄÖÜäöüß\u00c0-\u017f]+)(stra\s*b(?:l?e)?|stra\s*le)\b",
        replace_strasse_suffix,
        text,
    )
    text = re.sub(
        r"(?i)\bstra\s*(?:b(?:l?e)?|le)\b",
        lambda match: normalize_strasse_case(match.group(0)),
        text,
    )
    text = re.sub(r"\b(Personen)mit\b", r"\1 mit", text, flags=re.IGNORECASE)
    text = re.sub(r"([a-zäöüß])([A-ZÄÖÜ])", r"\1 \2", text)
    text = re.sub(r"([A-ZÄÖÜ]{2,})([A-ZÄÖÜ][a-zäöüß])", r"\1 \2", text)
    text = re.sub(r"([A-Za-zÄÖÜäöüß])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Za-zÄÖÜäöüß])", r"\1 \2", text)
    text = re.sub(
        r"(?<=\w)(Aktenz(?:ei|el)chen|Gesch\.?\s*-?\s*Z\.?|Geschäftszeichen|"
        r"Geschaeftszeichen|Geschaftszeichen|Az\.?|AZR\s*-?\s*Nummer)",
        r" \1",
        text,
    )
    text = re.sub(r"\s{2,}", " ", text)
    return text


def fix_placeholder_boundaries(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"([A-Za-zÄÖÜäöüß])((?:\[[A-ZÄÖÜ_0-9]+\]))", r"\1 \2", text)
    text = re.sub(r"((?:\[[A-ZÄÖÜ_0-9]+\]))([A-Za-zÄÖÜäöüß])", r"\1 \2", text)
    text = re.sub(r"([0-9])((?:\[[A-ZÄÖÜ_0-9]+\]))", r"\1 \2", text)
    text = re.sub(r"((?:\[[A-ZÄÖÜ_0-9]+\]))([0-9])", r"\1 \2", text)
    return text


def redact_labeled_names(text: str) -> str:
    if not text:
        return text

    text = NAME_FIELD_PATTERN.sub(r"\1 [PERSON]", text)
    text = RELATION_LABEL_PATTERN.sub(r"\1 [PERSON]", text)
    return text


def redact_citation_authors(text: str) -> str:
    if not text:
        return text

    return CITATION_AUTHOR_PATTERN.sub("[AUTOR]", text)


PERSON_PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:KLÄGERIN|KLÄGER|PERSON|KIND|FAMILIANGEHÖRIG)[A-ZÄÖÜ]*?(?:_\d+)?\]"
)


def redact_dobs_in_person_lines(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if PERSON_PLACEHOLDER_PATTERN.search(line):
            lines[idx] = DOB_PATTERN.sub("[GEBURTSDATUM]", line)
    return "\n".join(lines)


def redact_urls(text: str) -> str:
    if not text:
        return text

    return re.sub(r"https?://\S+|www\.\S+", "[URL]", text)


def replace_glued_known_names(
    text: str,
    person_registry: Dict[str, int],
    family_registry: Dict[str, int],
) -> str:
    if not text:
        return text

    for name, idx in person_registry.items():
        if len(name) < 4:
            continue
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        for match in reversed(list(pattern.finditer(text))):
            if is_safe_span_boundary(text, match.start(), match.end()):
                if text[match.start() : match.end()] == name:
                    continue
            if is_allowlisted_name(match.group(0), text, match.start(), match.end()):
                continue
            case = detect_case(text, match.start())
            replacement = get_inflected_placeholder("person", case, idx)
            text = text[: match.start()] + replacement + text[match.end() :]

    for name, idx in family_registry.items():
        if len(name) < 4:
            continue
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        for match in reversed(list(pattern.finditer(text))):
            if is_safe_span_boundary(text, match.start(), match.end()):
                if text[match.start() : match.end()] == name:
                    continue
            if is_allowlisted_name(match.group(0), text, match.start(), match.end()):
                continue
            case = detect_case(text, match.start())
            replacement = get_inflected_placeholder("family", case, idx)
            text = text[: match.start()] + replacement + text[match.end() :]

    return text


def expand_person_span_end(
    text: str, start: int, end: int, next_start: Optional[int]
) -> int:
    current_end = end
    position = end
    tokens_added = 0

    while tokens_added < 2:
        whitespace_match = re.match(r"[ \t]+", text[position:])
        if not whitespace_match:
            break
        position += whitespace_match.end()
        if position >= len(text):
            break
        if text[position] in ",;:()[]{}\n":
            break

        token_match = NAME_TOKEN_RE.match(text[position:])
        if not token_match:
            break

        token = token_match.group(0)
        token_start = position
        token_end = position + token_match.end()

        if next_start and token_start >= next_start:
            break
        if next_start and token_start < next_start < token_end:
            break
        if not is_probable_name_token(token):
            break

        current_end = token_end
        tokens_added += 1
        position = token_end

    return current_end


def extract_comma_name(text: str, end: int) -> Optional[Tuple[str, int, int]]:
    match = re.match(r",\s*(" + NAME_COMPONENT_REGEX + r")", text[end:])
    if not match:
        return None
    name = match.group(1)
    if not is_usable_person_name(name):
        return None
    name_start = end + match.start(1)
    match_end = end + match.end()
    return name, name_start, match_end


def expand_person_entities(
    text: str, entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not entities:
        return entities

    expanded = sorted(entities, key=lambda entry: entry["start"])

    for idx, ent in enumerate(expanded):
        if ent["tag"] not in PERSON_TAGS:
            continue
        next_start = expanded[idx + 1]["start"] if idx + 1 < len(expanded) else None
        expanded_end = expand_person_span_end(
            text, ent["start"], ent["end"], next_start
        )
        if expanded_end > ent["end"]:
            ent["end"] = expanded_end
            ent["text"] = text[ent["start"] : ent["end"]].strip()

    return expanded


def replace_remaining_uppercase_names(
    text: str,
    person_registry: Dict[str, int],
    family_registry: Dict[str, int],
) -> str:
    token_map: Dict[str, Tuple[str, int]] = {}

    for name, idx in person_registry.items():
        for token in NAME_TOKEN_RE.findall(name):
            if len(token) < 3:
                continue
            if not is_probable_name_token(token):
                continue
            token_upper = token.upper()
            token_map.setdefault(token_upper, ("person", idx))

    for name, idx in family_registry.items():
        for token in NAME_TOKEN_RE.findall(name):
            if len(token) < 3:
                continue
            if not is_probable_name_token(token):
                continue
            token_upper = token.upper()
            token_map.setdefault(token_upper, ("family", idx))

    for token_upper, (kind, idx) in token_map.items():
        pattern = re.compile(r"\b" + re.escape(token_upper) + r"\b")

        def repl(match: re.Match) -> str:
            if is_allowlisted_name(match.group(0), text, match.start(), match.end()):
                return match.group(0)
            case = detect_case(text, match.start())
            if kind == "person":
                return get_inflected_placeholder("person", case, idx)

            context = text[max(0, match.start() - 30) : match.start()].lower()
            if any(word in context for word in ["sohn", "tochter", "kind", "kinder"]):
                return get_inflected_placeholder("child", case, idx)
            return get_inflected_placeholder("family", case, idx)

        text = pattern.sub(repl, text)

    return text


def find_name_variants(text: str, known_names: List[str]) -> List[Tuple[int, int, str]]:
    """
    Find all occurrences of known names throughout the text.
    Handles: "Müller", "Maria Müller", "Frau Müller", etc.
    """
    matches = []

    uppercase_token_map: Dict[str, str] = {}

    for name in known_names:
        if not is_usable_person_name(name):
            continue
        # Extract last name (for compound names like "Maria Müller")
        name_parts = name.split()
        last_name = name_parts[-1] if name_parts else name
        first_name = name_parts[0] if len(name_parts) > 1 else None
        given_names = " ".join(name_parts[:-1]) if len(name_parts) > 1 else None

        # Search for the full name (word boundaries)
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), name))

        # Search for "LAST, Given Names" variant
        if given_names and last_name and len(last_name) >= 2:
            pattern = re.compile(
                r"\b" + re.escape(last_name) + r",\s*" + re.escape(given_names) + r"\b"
            )
            for match in pattern.finditer(text):
                already = any(
                    s <= match.start() and e >= match.end() for s, e, _ in matches
                )
                if not already:
                    matches.append((match.start(), match.end(), name))

        # Search for last name alone (with word boundary)
        if last_name and last_name != name and len(last_name) >= 4:
            pattern = re.compile(r"\b" + re.escape(last_name) + r"\b")
            for match in pattern.finditer(text):
                # Avoid if already covered by full name match
                already = any(
                    s <= match.start() and e >= match.end() for s, e, _ in matches
                )
                if not already:
                    matches.append((match.start(), match.end(), name))

        # Search for first name alone
        if first_name and len(first_name) >= 4:
            pattern = re.compile(r"\b" + re.escape(first_name) + r"\b")
            for match in pattern.finditer(text):
                already = any(
                    s <= match.start() and e >= match.end() for s, e, _ in matches
                )
                if not already:
                    matches.append((match.start(), match.end(), name))

        for token in NAME_TOKEN_RE.findall(name):
            if len(token) < 3:
                continue
            if not is_probable_name_token(token):
                continue
            token_upper = token.upper()
            if token_upper not in uppercase_token_map:
                uppercase_token_map[token_upper] = name

    for token_upper, name in uppercase_token_map.items():
        if len(token_upper) < 4:
            continue
        pattern = re.compile(r"\b" + re.escape(token_upper) + r"\b")
        for match in pattern.finditer(text):
            already = any(
                s <= match.start() and e >= match.end() for s, e, _ in matches
            )
            if not already:
                matches.append((match.start(), match.end(), name))

    return matches


# =============================================================================
# MAIN ANONYMIZATION FUNCTION
# =============================================================================


def anonymize_with_flair(
    text: str,
) -> Tuple[str, List[str], List[str], List[str], float]:
    """
    Anonymize text using Flair NER with:
    - Consistent pseudonyms (PERSON_1, PERSON_2)
    - Case-inflected placeholders
    - Family relation detection

    Returns: (anonymized_text, plaintiff_names, family_members, addresses, confidence)
    """
    from flair.data import Sentence

    tagger = get_tagger()

    text = normalize_ocr_text(text)

    # Process in chunks to handle long documents
    MAX_CHUNK_SIZE = 5000

    all_entities = []
    chunks = []

    # Split text into manageable chunks
    remaining = text
    offset = 0
    while remaining:
        chunk = remaining[:MAX_CHUNK_SIZE]
        if len(remaining) > MAX_CHUNK_SIZE:
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > MAX_CHUNK_SIZE // 2:
                chunk = remaining[: break_point + 1]

        chunks.append((offset, chunk))
        offset += len(chunk)
        remaining = remaining[len(chunk) :]

    print(f"[INFO] Processing {len(chunks)} chunks...")

    # Create all sentences for batch processing
    sentences = []
    chunk_offsets = []
    for chunk_offset, chunk_text in chunks:
        try:
            sentences.append(Sentence(chunk_text))
            chunk_offsets.append(chunk_offset)
        except Exception as e:
            print(f"[WARNING] Failed to create sentence: {e}")

    # Batch predict (faster on GPU)
    if sentences:
        tagger.predict(sentences, mini_batch_size=32)

    # Extract entities
    for sentence, chunk_offset in zip(sentences, chunk_offsets):
        for entity in sentence.get_spans("ner"):
            all_entities.append(
                {
                    "text": entity.text,
                    "tag": entity.tag,
                    "start": chunk_offset + entity.start_position,
                    "end": chunk_offset + entity.end_position,
                    "score": entity.score,
                }
            )

    all_entities = expand_person_entities(text, all_entities)

    # ==========================================================================
    # BUILD ENTITY REGISTRY WITH CONSISTENT PSEUDONYMS
    # ==========================================================================

    # BUILD ENTITY REGISTRY WITH CONSISTENT PSEUDONYMS
    # ==========================================================================

    # Track unique persons with consistent IDs
    person_registry: Dict[str, int] = OrderedDict()  # name -> index
    family_registry: Dict[str, int] = OrderedDict()  # name -> index

    plaintiff_names = []
    family_members = []
    addresses = []
    plaintiff_is_male = (
        None  # Track plaintiff gender for "Frau/Herr [LastName]" detection
    )

    # Seed registry with "LAST, First" name patterns that NER may miss
    for match in LAST_FIRST_PATTERN.finditer(text):
        context = get_line_context(text, match.start(), match.end(), include_prev=True)
        if not (
            LAST_FIRST_CONTEXT_PATTERN.search(context)
            or has_list_context(text, match.start())
        ):
            continue
        last_name = match.group(1).strip()
        first_name = match.group(2).strip()
        if len(last_name) < 3 or len(first_name) < 3:
            continue
        full_name = f"{first_name} {last_name}"
        if not is_usable_person_name(full_name):
            continue
        if full_name not in person_registry and full_name not in family_registry:
            person_registry[full_name] = len(person_registry)
            plaintiff_names.append(full_name)

    # Seed registry with lawyer names and Kanzlei references
    for match in LAWYER_TITLE_PATTERN.finditer(text):
        name = match.group(2).strip()
        if not is_usable_person_name(name):
            continue
        if name not in person_registry and name not in family_registry:
            person_registry[name] = len(person_registry)
            plaintiff_names.append(name)

    for match in KANZLEI_NAME_PATTERN.finditer(text):
        name = match.group(1).strip()
        if not is_usable_person_name(name):
            continue
        name_tokens = name.split()
        if len(name_tokens) == 1:
            token = name_tokens[0].lower()
            existing_last_names = {
                existing.split()[-1].lower()
                for existing in list(person_registry.keys()) + list(family_registry.keys())
                if existing.split()
            }
            if token in existing_last_names:
                continue
        if name not in person_registry and name not in family_registry:
            person_registry[name] = len(person_registry)
            plaintiff_names.append(name)

    for name in extract_signature_names(text):
        if name not in person_registry and name not in family_registry:
            person_registry[name] = len(person_registry)
            plaintiff_names.append(name)

    # FIRST: Detect explicit plaintiff designations via title patterns
    # This takes precedence over family context detection
    for match in TITLE_NAME_PATTERN.finditer(text):
        title = match.group(1).lower()
        name = match.group(2).strip()
        if name and name not in person_registry:
            if not is_usable_title_name(name):
                continue
            # Explicit plaintiff/applicant titles
            if (
                "kläger" in title
                or "antragsteller" in title
                or "beschwerdeführer" in title
            ):
                person_registry[name] = len(person_registry)
                plaintiff_names.append(name)
                # Detect gender: "Klägerin" = female, "Kläger" = male
                if plaintiff_is_male is None:
                    plaintiff_is_male = not ("in" in title and title.endswith("in"))
                print(
                    f"[DEBUG] Plaintiff from title: {name} (male={plaintiff_is_male})"
                )

    for match in TITLE_GLUE_PATTERN.finditer(text):
        title = match.group(1).lower()
        name = match.group(2).strip()
        if name and name not in person_registry:
            if not is_usable_title_name(name):
                continue
            if (
                "kläger" in title
                or "antragsteller" in title
                or "beschwerdeführer" in title
            ):
                person_registry[name] = len(person_registry)
                idx = person_registry[name]
                plaintiff_names.append(name)
                extra = extract_comma_name(text, match.end())
                if extra:
                    extra_name, _, _ = extra
                    if extra_name not in person_registry:
                        person_registry[extra_name] = idx
                        plaintiff_names.append(extra_name)
                if plaintiff_is_male is None:
                    plaintiff_is_male = not ("in" in title and title.endswith("in"))
                print(
                    f"[DEBUG] Plaintiff from glued title: {name} (male={plaintiff_is_male})"
                )

    # Second pass: NER-detected persons (respecting already-registered plaintiffs)
    for ent in all_entities:
        if ent["tag"] in PERSON_TAGS:
            name = ent["text"].strip()
            if not is_usable_person_name(name):
                continue
            if is_allowlisted_name(name, text, ent["start"], ent["end"]):
                continue
            if not is_safe_span_boundary(text, ent["start"], ent["end"]):
                continue
            if len(name.split()) == 1 and not has_person_context(
                text, ent["start"], ent["end"]
            ):
                continue
            # Check if this is a family member based on context
            context_start = max(0, ent["start"] - 50)
            context = text[context_start : ent["start"]].lower()

            is_family = any(rel in context for rel in FAMILY_RELATIONS)

            if is_family:
                # Don't override explicit plaintiff designation
                if name not in family_registry and name not in person_registry:
                    family_registry[name] = len(family_registry)
                    family_members.append(name)
            else:
                if name not in person_registry and name not in family_registry:
                    person_registry[name] = len(person_registry)
                    plaintiff_names.append(name)

    # Also detect family members via regex patterns
    for match in DIRECT_FAMILY_PATTERN.finditer(text):
        relation = match.group(1).lower()
        name = match.group(2).strip()
        if name and name not in family_registry and name not in person_registry:
            if not is_usable_person_name(name):
                continue
            if is_allowlisted_name(name, text, match.start(2), match.end(2)):
                continue
            family_registry[name] = len(family_registry)
            family_members.append(name)

    for match in WITH_NAME_PATTERN.finditer(text):
        context = get_line_context(text, match.start(), match.end(), include_prev=False)
        if not WITH_NAME_CUE_PATTERN.search(context):
            continue
        name = match.group(1).strip()
        if name and name not in family_registry and name not in person_registry:
            if not is_usable_person_name(name):
                continue
            if is_allowlisted_name(name, text, match.start(1), match.end(1)):
                continue
            family_registry[name] = len(family_registry)
            family_members.append(name)

    # Fallback: detect names via title patterns that NER might have missed
    for match in TITLE_NAME_PATTERN.finditer(text):
        title = match.group(1).lower()
        name = match.group(2).strip()
        if name and name not in person_registry and name not in family_registry:
            if not is_usable_title_name(name):
                continue
            is_plaintiff_title = (
                "kläger" in title
                or "antragsteller" in title
                or "beschwerdeführer" in title
            )
            if not is_plaintiff_title and is_allowlisted_name(
                name, text, match.start(), match.end()
            ):
                continue
            # Check if title suggests family or plaintiff
            if is_plaintiff_title:
                person_registry[name] = len(person_registry)
                plaintiff_names.append(name)
            elif title in ["herr", "herrn", "frau"]:
                # Check context for family relation
                context_start = max(0, match.start() - 50)
                context = text[context_start : match.start()].lower()
                if any(rel in context for rel in FAMILY_RELATIONS):
                    family_registry[name] = len(family_registry)
                    family_members.append(name)
                else:
                    person_registry[name] = len(person_registry)
                    plaintiff_names.append(name)

    for match in TITLE_GLUE_PATTERN.finditer(text):
        title = match.group(1).lower()
        name = match.group(2).strip()
        if name and name not in person_registry and name not in family_registry:
            if not is_usable_title_name(name):
                continue
            extra = extract_comma_name(text, match.end())
            is_plaintiff_title = (
                "kläger" in title
                or "antragsteller" in title
                or "beschwerdeführer" in title
            )
            if not is_plaintiff_title and is_allowlisted_name(
                name, text, match.start(), match.end()
            ):
                continue
            if is_plaintiff_title:
                person_registry[name] = len(person_registry)
                idx = person_registry[name]
                plaintiff_names.append(name)
                if extra:
                    extra_name, extra_start, extra_end = extra
                    if not is_allowlisted_name(
                        extra_name, text, extra_start, extra_end
                    ):
                        if extra_name not in person_registry:
                            person_registry[extra_name] = idx
                            plaintiff_names.append(extra_name)
            elif title in ["herr", "herrn", "frau"]:
                context_start = max(0, match.start() - 50)
                context = text[context_start : match.start()].lower()
                if any(rel in context for rel in FAMILY_RELATIONS):
                    family_registry[name] = len(family_registry)
                    idx = family_registry[name]
                    family_members.append(name)
                    if extra:
                        extra_name, extra_start, extra_end = extra
                        if not is_allowlisted_name(
                            extra_name, text, extra_start, extra_end
                        ):
                            if extra_name not in family_registry:
                                family_registry[extra_name] = idx
                                family_members.append(extra_name)
                else:
                    person_registry[name] = len(person_registry)
                    idx = person_registry[name]
                    plaintiff_names.append(name)
                    if extra:
                        extra_name, extra_start, extra_end = extra
                        if not is_allowlisted_name(
                            extra_name, text, extra_start, extra_end
                        ):
                            if extra_name not in person_registry:
                                person_registry[extra_name] = idx
                                plaintiff_names.append(extra_name)

    # ==========================================================================
    # DETECT SPOUSE VIA GENDERED TITLE + PLAINTIFF LAST NAME
    # ==========================================================================
    # If plaintiff is male (Kläger) and we see "Frau Müller", that's the spouse
    # If plaintiff is female (Klägerin) and we see "Herr Müller", that's the spouse

    if plaintiff_is_male is not None:
        spouse_title = "frau" if plaintiff_is_male else "herr[n]?"
        spouse_pattern = re.compile(
            r"\b("
            + spouse_title
            + r")\s+("
            + "|".join(re.escape(n) for n in plaintiff_names)
            + r")\b",
            re.IGNORECASE,
        )
        for match in spouse_pattern.finditer(text):
            spouse_ref = match.group(0)  # e.g., "Frau Müller"
            if spouse_ref not in family_registry:
                family_registry[spouse_ref] = len(family_registry)
                family_members.append(spouse_ref)
                print(f"[DEBUG] Spouse reference detected: {spouse_ref}")

    print(
        f"[INFO] Found {len(person_registry)} plaintiffs, {len(family_registry)} family members"
    )

    # ==========================================================================
    # COMBINE FAMILY FIRST NAMES + PLAINTIFF LAST NAMES INTO FULL NAMES
    # ==========================================================================
    # If we have family members like "Maria" and plaintiff "Müller", find "Maria Müller" in text
    # and register it as a full family name (so it takes precedence over partial matches)

    plaintiff_last_names = list(person_registry.keys())
    family_first_names = [
        name for name in family_registry.keys() if " " not in name
    ]  # Single-word names only

    for first_name in family_first_names:
        for last_name in plaintiff_last_names:
            full_name = f"{first_name} {last_name}"
            # Check if this full name appears in text
            if full_name in text and full_name not in family_registry:
                # Get the index of the first name in family_registry
                idx = family_registry[first_name]
                family_registry[full_name] = (
                    idx  # Same index as first name (same person)
                )
                family_members.append(full_name)
                print(f"[DEBUG] Combined family name: {full_name}")

    # ==========================================================================
    # NAME PROPAGATION - Find all variants of known names throughout text
    # ==========================================================================

    # Find all occurrences of plaintiff names (handles "Müller" when we know "Maria Müller")
    all_plaintiff_names = list(person_registry.keys())
    all_family_names = list(family_registry.keys())

    plaintiff_name_matches = find_name_variants(text, all_plaintiff_names)
    family_name_matches = find_name_variants(text, all_family_names)

    print(
        f"[INFO] Name propagation: {len(plaintiff_name_matches)} plaintiff matches, {len(family_name_matches)} family matches"
    )

    # ==========================================================================
    # BUILD REPLACEMENT LIST
    # ==========================================================================

    entities_to_replace = []

    # Add propagated name matches first
    for start, end, original_name in plaintiff_name_matches:
        case = detect_case(text, start)
        idx = person_registry.get(original_name, 0)
        replacement = get_inflected_placeholder("person", case, idx)
        entities_to_replace.append((start, end, replacement))

    for start, end, original_name in family_name_matches:
        case = detect_case(text, start)
        idx = family_registry.get(original_name, 0)
        # Check context for child vs other family
        context = text[max(0, start - 30) : start].lower()
        if any(w in context for w in ["sohn", "tochter", "kind", "kinder"]):
            replacement = get_inflected_placeholder("child", case, idx)
        else:
            replacement = get_inflected_placeholder("family", case, idx)
        entities_to_replace.append((start, end, replacement))

    for match in TITLE_NAME_PATTERN.finditer(text):
        title_text = match.group(1)
        title = title_text.lower()
        name = match.group(2).strip()
        if not is_usable_title_name(name):
            continue
        is_plaintiff_title = (
            "kläger" in title
            or "antragsteller" in title
            or "beschwerdeführer" in title
        )
        if not is_plaintiff_title and is_allowlisted_name(
            name, text, match.start(), match.end()
        ):
            continue
        if not is_safe_span_boundary(text, match.start(), match.end()):
            continue
        end = match.end()
        extra = extract_comma_name(text, end)
        if extra:
            extra_name, extra_start, extra_end = extra
            if not is_allowlisted_name(extra_name, text, extra_start, extra_end):
                end = extra_end
        case = detect_case(text, match.start())
        if name in family_registry:
            idx = family_registry[name]
            context = text[max(0, match.start() - 30) : match.start()].lower()
            if any(w in context for w in ["sohn", "tochter", "kind", "kinder"]):
                placeholder = get_inflected_placeholder("child", case, idx)
            else:
                placeholder = get_inflected_placeholder("family", case, idx)
        elif name in person_registry:
            idx = person_registry[name]
            placeholder = get_inflected_placeholder("person", case, idx)
        else:
            placeholder = "[PERSON]"
        entities_to_replace.append((match.start(), end, f"{title_text} {placeholder}"))

    for match in TITLE_GLUE_PATTERN.finditer(text):
        title_text = match.group(1)
        title = title_text.lower()
        name = match.group(2).strip()
        if not is_usable_title_name(name):
            continue
        is_plaintiff_title = (
            "kläger" in title
            or "antragsteller" in title
            or "beschwerdeführer" in title
        )
        if not is_plaintiff_title and is_allowlisted_name(
            name, text, match.start(), match.end()
        ):
            continue
        if not is_safe_span_boundary(text, match.start(), match.end()):
            continue
        end = match.end()
        extra = extract_comma_name(text, end)
        if extra:
            extra_name, extra_start, extra_end = extra
            if not is_allowlisted_name(extra_name, text, extra_start, extra_end):
                end = extra_end
        case = detect_case(text, match.start())
        if name in family_registry:
            idx = family_registry[name]
            context = text[max(0, match.start() - 30) : match.start()].lower()
            if any(w in context for w in ["sohn", "tochter", "kind", "kinder"]):
                placeholder = get_inflected_placeholder("child", case, idx)
            else:
                placeholder = get_inflected_placeholder("family", case, idx)
        elif name in person_registry:
            idx = person_registry[name]
            placeholder = get_inflected_placeholder("person", case, idx)
        else:
            placeholder = "[PERSON]"
        entities_to_replace.append((match.start(), end, f"{title_text} {placeholder}"))

    # Standalone addresses
    for match in ADDRESS_PATTERN.finditer(text):
        already_covered = any(
            start <= match.start() and end >= match.end()
            for start, end, _ in entities_to_replace
        )
        if not already_covered:
            full_match = match.group(0)
            if is_allowlisted_address(full_match):
                continue
            if full_match not in addresses:
                addresses.append(full_match)
            entities_to_replace.append((match.start(), match.end(), "[ADRESSE]"))

    for match in ADDRESS_PREFIX_PATTERN.finditer(text):
        already_covered = any(
            start <= match.start() and end >= match.end()
            for start, end, _ in entities_to_replace
        )
        if not already_covered:
            full_match = match.group(0)
            if is_allowlisted_address(full_match):
                continue
            if full_match not in addresses:
                addresses.append(full_match)
            entities_to_replace.append((match.start(), match.end(), "[ADRESSE]"))

    # PLZ + City
    for match in PLZ_CITY_PATTERN.finditer(text):
        already_covered = any(
            start <= match.start() and end >= match.end()
            for start, end, _ in entities_to_replace
        )
        if not already_covered:
            entities_to_replace.append((match.start(), match.end(), "[ORT]"))

    # DOB cues (only redact dates near DOB cues)
    for match in DOB_CUE_PATTERN.finditer(text):
        date_text = match.group(2)
        entities_to_replace.append((match.start(2), match.end(2), "[GEBURTSDATUM]"))

    # DOB blocks (e.g. Name/Geburtsdatum lists)
    for block_match in DOB_BLOCK_START_PATTERN.finditer(text):
        block_start = block_match.end()
        block_text = text[block_start:]
        end_match = DOB_BLOCK_END_PATTERN.search(block_text)
        if end_match:
            block_text = block_text[: end_match.start()]
        for date_match in DOB_PATTERN.finditer(block_text):
            start = block_start + date_match.start()
            end = block_start + date_match.end()
            entities_to_replace.append((start, end, "[GEBURTSDATUM]"))

    # DOBs near cues (handles tables where date precedes cue)
    for date_match in DOB_PATTERN.finditer(text):
        window_start = max(0, date_match.start() - DOB_CONTEXT_WINDOW)
        window_end = min(len(text), date_match.end() + DOB_CONTEXT_WINDOW)
        if DOB_CONTEXT_PATTERN.search(text[window_start:window_end]):
            entities_to_replace.append(
                (date_match.start(), date_match.end(), "[GEBURTSDATUM]")
            )

    # DOBs near known names (fallback for list-style blocks without cues)
    all_known_names = [*person_registry.keys(), *family_registry.keys()]
    if all_known_names:
        lowered_names = [name.lower() for name in all_known_names if name]
        for date_match in DOB_PATTERN.finditer(text):
            line_context = get_line_context(
                text, date_match.start(), date_match.end(), include_prev=False
            ).lower()
            if any(name in line_context for name in lowered_names):
                entities_to_replace.append(
                    (date_match.start(), date_match.end(), "[GEBURTSDATUM]")
                )

    # Aktenzeichen / Geschäftszeichen
    for match in AKTENZEICHEN_PATTERN.finditer(text):
        entities_to_replace.append(
            (match.start(), match.end(), f"{match.group(1)} [AKTENZEICHEN]")
        )
        full_match = match.group(0)
        if full_match not in addresses:
            addresses.append(full_match)

    # Bare Aktenzeichen values near cues
    for match in AKTENZEICHEN_CUE_PATTERN.finditer(text):
        cue_label = match.group(1).lower()
        cue_end = match.end()
        tail = text[cue_end : cue_end + ID_SCAN_CHARS]
        if "azr" in cue_label:
            for azr_match in AZR_NUMBER_PATTERN.finditer(tail):
                start = cue_end + azr_match.start()
                end = cue_end + azr_match.end()
                if is_phone_context(text, start, end):
                    continue
                entities_to_replace.append((start, end, "[AKTENZEICHEN]"))
            for azr_match in AZR_NUMBER_FUZZY_PATTERN.finditer(tail):
                start = cue_end + azr_match.start()
                end = cue_end + azr_match.end()
                if is_phone_context(text, start, end):
                    continue
                entities_to_replace.append((start, end, "[AKTENZEICHEN]"))
            continue

        for value_match in AKTENZEICHEN_VALUE_PATTERN.finditer(tail):
            start = cue_end + value_match.start()
            end = cue_end + value_match.end()
            if is_phone_context(text, start, end):
                continue
            entities_to_replace.append((start, end, "[AKTENZEICHEN]"))
        for value_match in ID_NUMBER_PATTERN.finditer(tail):
            start = cue_end + value_match.start()
            end = cue_end + value_match.end()
            if is_phone_context(text, start, end):
                continue
            entities_to_replace.append((start, end, "[AKTENZEICHEN]"))

    # BAMF / Bundesamt numeric IDs near cues
    for match in BAMF_CUE_PATTERN.finditer(text):
        cue_end = match.end()
        tail = text[cue_end : cue_end + BAMF_ID_SCAN_CHARS]
        for bamf_match in AKTENZEICHEN_VALUE_PATTERN.finditer(tail):
            start = cue_end + bamf_match.start()
            end = cue_end + bamf_match.end()
            if is_phone_context(text, start, end):
                continue
            entities_to_replace.append((start, end, "[AKTENZEICHEN]"))

    # Hyphenated IDs without cues
    for match in AKTENZEICHEN_VALUE_PATTERN.finditer(text):
        entities_to_replace.append((match.start(), match.end(), "[AKTENZEICHEN]"))
        for bamf_match in ID_NUMBER_PATTERN.finditer(tail):
            start = cue_end + bamf_match.start()
            end = cue_end + bamf_match.end()
            if is_phone_context(text, start, end):
                continue
            entities_to_replace.append((start, end, "[AKTENZEICHEN]"))

    if ANONYMIZE_PHONES:
        for match in PHONE_LINE_PATTERN.finditer(text):
            entities_to_replace.append((match.start(), match.end(), "[TELEFON]"))

        for match in PHONE_CUE_PATTERN.finditer(text):
            cue_end = match.end()
            line_end = text.find("\n", cue_end)
            if line_end == -1:
                line_end = len(text)
            line_ranges = [(cue_end, line_end)]

            next_line_start = line_end + 1
            for _ in range(2):
                if next_line_start >= len(text):
                    break
                next_line_end = text.find("\n", next_line_start)
                if next_line_end == -1:
                    next_line_end = len(text)
                line_ranges.append((next_line_start, next_line_end))
                next_line_start = next_line_end + 1

            for base_offset, line_end in line_ranges:
                tail = text[base_offset:line_end]
                for phone_match in PHONE_NUMBER_PATTERN.finditer(tail):
                    candidate = phone_match.group(0)
                    if sum(char.isdigit() for char in candidate) < 5:
                        continue
                    if DOB_PATTERN.search(candidate):
                        continue
                    start = base_offset + phone_match.start()
                    end = base_offset + phone_match.end()
                    entities_to_replace.append((start, end, "[TELEFON]"))

        for match in PHONE_GLOBAL_PATTERN.finditer(text):
            if sum(char.isdigit() for char in match.group(0)) < 7:
                continue
            if is_bank_context(text, match.start(), match.end(), include_prev=False):
                continue
            context = get_line_context(text, match.start(), match.end(), include_prev=False)
            if PHONE_SKIP_CONTEXT_PATTERN.search(context):
                continue
            entities_to_replace.append((match.start(), match.end(), "[TELEFON]"))

    # ==========================================================================
    # APPLY REPLACEMENTS
    # ==========================================================================

    # Sort by: start position ascending, then by length descending (longer matches first)
    # This ensures "Maria Müller" (longer) is preferred over just "Müller" (shorter)
    entities_to_replace.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Remove overlapping entities (prefer longer matches)
    filtered_entities = []
    covered_ranges = []  # List of (start, end) tuples we've already covered

    for start, end, replacement in entities_to_replace:
        # Check if this entity overlaps with any already-added entity
        overlaps = any(
            not (end <= cov_start or start >= cov_end)
            for cov_start, cov_end in covered_ranges
        )
        if not overlaps:
            filtered_entities.append((start, end, replacement))
            covered_ranges.append((start, end))

    # Sort descending for safe replacement (replace from end to start)
    filtered_entities.sort(key=lambda x: x[0], reverse=True)

    # Apply replacements
    anonymized = list(text)
    for start, end, replacement in filtered_entities:
        anonymized[start:end] = list(replacement)

    anonymized_text = "".join(anonymized)
    anonymized_text = replace_remaining_uppercase_names(
        anonymized_text, person_registry, family_registry
    )
    anonymized_text = replace_glued_known_names(
        anonymized_text, person_registry, family_registry
    )
    anonymized_text = redact_dobs_in_person_lines(anonymized_text)
    anonymized_text = ADDRESS_LABEL_LINE_PATTERN.sub(
        lambda match: f"{match.group(1)}: [ADRESSE]", anonymized_text
    )
    anonymized_text = ADDRESS_PATTERN.sub("[ADRESSE]", anonymized_text)
    anonymized_text = ADDRESS_PREFIX_PATTERN.sub("[ADRESSE]", anonymized_text)
    anonymized_text = PLZ_CITY_PATTERN.sub("[ORT]", anonymized_text)

    anonymized_text = AKTENZEICHEN_PATTERN.sub(
        lambda match: f"{match.group(1)} [AKTENZEICHEN]", anonymized_text
    )
    anonymized_text = redact_labeled_names(anonymized_text)
    anonymized_text = redact_citation_authors(anonymized_text)
    anonymized_text = redact_urls(anonymized_text)
    anonymized_text = fix_placeholder_boundaries(anonymized_text)

    # Calculate confidence
    if all_entities:
        avg_score = sum(e["score"] for e in all_entities) / len(all_entities)
    else:
        avg_score = 0.5

    return anonymized_text, plaintiff_names, family_members, addresses, avg_score


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_document(
    request: AnonymizationRequest, x_api_key: str = Header(None)
):
    """
    Anonymize plaintiff names and addresses in German legal documents.

    Features:
    - Flair NER (flair/ner-german-legal) for fast, accurate detection
    - Processes the ENTIRE document
    - Case-inflected placeholders (des Klägers, dem Kläger)
    - Consistent pseudonyms (KLÄGER_1, KLÄGER_2)
    - Family member detection
    """

    if ANONYMIZATION_API_KEY:
        if not x_api_key or x_api_key != ANONYMIZATION_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    print(f"[INFO] Processing {request.document_type}: {len(request.text)} characters")

    try:
        anonymized_text, plaintiff_names, family_members, addresses, confidence = (
            anonymize_with_flair(request.text)
        )

        print(
            f"[SUCCESS] Found {len(plaintiff_names)} plaintiffs, {len(family_members)} family, {len(addresses)} addresses"
        )
        print(f"[INFO] Confidence: {confidence:.2%}")

        return AnonymizationResponse(
            is_valid=True,
            invalid_reason=None,
            anonymized_text=anonymized_text,
            plaintiff_names=plaintiff_names,
            family_members=family_members,
            addresses=addresses,
            confidence=confidence,
        )

    except Exception as e:
        print(f"[ERROR] Anonymization failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check - also warms up the model."""
    try:
        tagger = get_tagger()
        return {
            "status": "healthy",
            "model": "flair/ner-german-legal",
            "version": "2.2.4",
            "features": [
                "case-inflected placeholders",
                "consistent pseudonyms",
                "family detection",
            ],
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/")
async def root():
    return {
        "service": "Rechtmaschine Anonymization Service (Flair)",
        "version": "2.2.4",
        "model": "flair/ner-german-legal",
        "features": [
            "GPU acceleration",
            "Full document processing",
            "Case-inflected placeholders (des Klägers, dem Kläger)",
            "Consistent pseudonyms (KLÄGER_1, KLÄGER_2)",
            "Family member detection (Ehemann, Kind, etc.)",
            "Name propagation (finds all name variants throughout document)",
        ],
        "endpoints": {"health": "/health", "anonymize": "/anonymize (POST)"},
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Anonymization Service (Flair NER) v2.2.4")
    print("=" * 60)
    print("Model: flair/ner-german-legal")
    print("Features:")
    print("  - GPU acceleration (auto-detected)")
    print("  - Full document processing")
    print("  - Case-inflected placeholders")
    print("  - Consistent pseudonyms")
    print("  - Family member detection")
    print("  - Name propagation (finds all variants)")
    print("Listening on: 0.0.0.0:9002")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=9002)
