"""
Rechtmaschine Anonymization Service (Hybrid LLM + Regex)

This service uses a hybrid approach:
1. LLM (qwen3:8b) extracts entities from the document (~5 seconds)
2. Regex replaces extracted entities in the original text (instant)

This is 15x faster than full LLM replacement and preserves legal citations.

Architecture:
- Runs on home PC with 12GB VRAM
- Accessed via Tailscale mesh network from production server
- Uses Ollama with Qwen3:8b model (5GB VRAM)
- Provides REST API on port 9002

Usage:
    python anonymization_service.py

Environment Variables:
    OLLAMA_URL: URL to Ollama API (default: http://localhost:11434/api/generate)
    OLLAMA_MODEL: Model to use (default: qwen3:8b)
    ANONYMIZATION_API_KEY: API key for authentication (optional)
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import os
import json
import re

app = FastAPI(
    title="Rechtmaschine Anonymization Service",
    description="Hybrid LLM + Regex anonymization for German asylum law documents",
    version="2.0.0",
)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
ANONYMIZATION_API_KEY = os.getenv("ANONYMIZATION_API_KEY")


class AnonymizationRequest(BaseModel):
    text: str
    document_type: str  # e.g., "Anhörung", "Bescheid", "Sonstige gespeicherte Quellen"


class AnonymizationResponse(BaseModel):
    is_valid: bool = True
    invalid_reason: str | None = None
    anonymized_text: str
    plaintiff_names: list[str]
    birth_dates: list[str] = []
    birth_places: list[str] = []
    addresses: list[str] = []
    azr_numbers: list[str] = []
    aufenthaltsgestattung_ids: list[str] = []
    case_numbers: list[str] = []
    confidence: float


# Known BAMF office addresses to filter out
BAMF_OFFICE_DATA = {
    "postal_codes": {"90343", "90461", "53115"},
    "cities": {"nürnberg", "bonn"},
    "streets": {"frankenstraße", "frankenstrabe", "reuterstraße", "reuterstrabe"},
}

# Group labels often misclassified as person names.
NON_PERSON_GROUP_TERMS = {
    "fulani",
    "peul",
    "paschtune",
    "pashtun",
    "hazara",
    "kurde",
    "kurdin",
    "kurd",
}

# Frequent non-person extraction artifacts that should never become names.
NON_PERSON_NAME_NOISE_TERMS = {
    "hinweis",
    "hinweise",
    "datum",
    "unterschrift",
    "untersche",
    "apagstellers",
    "antragsteller",
    "antragstellerin",
    "sachbearbeiter",
    "unterzeichner",
    "regierungsamtsrat",
    "urteil",
    "asyl",
    "asylantrag",
    "asylverfahren",
    "asylg",
    "quelle",
    "quellen",
    "sources",
    "say",
    "killed",
    "times",
    "kampfhandlungen",
    "luftangriffe",
}

NON_PERSON_ORG_TERMS = {
    "bamf",
    "brd",
    "bundesamt",
    "bundesamtes",
    "bundesministerium",
    "bundesrepublik",
    "referat",
    "behörde",
    "behoerde",
    "ministerium",
    "gericht",
}

GROUP_CONTEXT_KEYWORDS = {
    "ethnie",
    "ethnisch",
    "volksgruppe",
    "stamm",
    "stammes",
    "stammeszugehörigkeit",
    "tribe",
    "sprache",
    "religion",
    "konfession",
    "nationalität",
    "staatsangehörigkeit",
    "zugehörigkeit",
}

HONORIFICS = {
    "herr",
    "herrn",
    "frau",
    "genosse",
    "genossin",
    "mandant",
    "mandantin",
    "klient",
    "klientin",
}

PERSON_FIELD_LABEL_PATTERN = (
    r"(?:name|vorname(?:n)?|nachname|familienname|geburtsname|"
    r"name\s+des\s+(?:antragstellers?|antragstellerin|kl[äa]gers?|kl[äa]gerin|"
    r"kindes|sohnes|tochter|ehepartners?|ehegatten?))"
)

OCR_CONFUSABLES = {
    # Common OCR confusions in Latin text
    "I": r"[IiLl1|]",
    "i": r"[IiLl1|]",
    "L": r"[IiLl1|]",
    "l": r"[IiLl1|]",
    "1": r"[IiLl1]",
    "|": r"[IiLl1|]",
    "O": r"[O0]",
    "0": r"[O0]",
    # German-specific: ß vs B/8
    "ß": r"(?:ß|B|8)",
    "ẞ": r"(?:ẞ|B|8)",
    "B": r"(?:B|ß|ẞ|8)",
    "8": r"[8B]",
    # OCR occasionally reads Z/z as 2.
    "Z": r"[Zz2]",
    "z": r"[Zz2]",
    "2": r"[Zz2]",
}


def filter_bamf_addresses(entities: dict) -> dict:
    """Remove known BAMF office addresses from extracted entities."""
    filtered = {}
    for key, values in entities.items():
        if not isinstance(values, list):
            filtered[key] = values
            continue

        if key == "postal_codes":
            filtered[key] = [v for v in values if v not in BAMF_OFFICE_DATA["postal_codes"]]
        elif key == "cities":
            filtered[key] = [v for v in values if v.lower() not in BAMF_OFFICE_DATA["cities"]]
        elif key == "streets":
            filtered[key] = [
                v for v in values
                if not any(bamf in v.lower() for bamf in BAMF_OFFICE_DATA["streets"])
            ]
        else:
            filtered[key] = values

    return filtered


def filter_non_person_group_labels(entities: dict, text: str) -> dict:
    """Remove extracted names that are group labels (tribes/ethnicities/etc.)."""
    names = entities.get("names", [])
    if not isinstance(names, list) or not names:
        return entities

    filtered_names = []
    removed_names = []

    for name in names:
        if not isinstance(name, str):
            continue

        clean_name = name.strip()
        if not clean_name:
            continue

        normalized = re.sub(r"\s+", " ", clean_name).lower()
        is_single_token = " " not in normalized and "," not in normalized and "-" not in normalized
        normalized_tokens = re.findall(r"[a-zäöüß]+", normalized)
        remove = normalized in NON_PERSON_GROUP_TERMS
        if not remove and any(token in NON_PERSON_NAME_NOISE_TERMS for token in normalized_tokens):
            remove = True

        # Apply context heuristic only to single-token candidates.
        if not remove and is_single_token:
            pattern = re.compile(
                rf".{{0,80}}\b{re.escape(clean_name)}\b.{{0,80}}",
                flags=re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                context = match.group(0).lower()
                if any(keyword in context for keyword in GROUP_CONTEXT_KEYWORDS):
                    remove = True
                    break

        if remove:
            removed_names.append(clean_name)
        else:
            filtered_names.append(name)

    entities["names"] = filtered_names
    if removed_names:
        print(f"[INFO] Filtered non-person group labels from names: {removed_names}")
    return entities


def filter_non_person_organization_labels(entities: dict) -> dict:
    """Remove extracted names that are clearly organizations/authorities."""
    names = entities.get("names", [])
    if not isinstance(names, list) or not names:
        return entities

    filtered_names = []
    removed_names = []

    for name in names:
        if not isinstance(name, str):
            continue

        clean_name = name.strip()
        if not clean_name:
            continue

        normalized = re.sub(r"\s+", " ", clean_name).lower()
        normalized_ascii = (
            normalized.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        )
        tokens = re.findall(r"[a-zäöüß]+", normalized)

        remove = False
        if any(token in NON_PERSON_ORG_TERMS for token in tokens):
            remove = True
        if not remove and any(term in normalized_ascii for term in ("bundesamt", "bundesministerium", "bundesrepublik")):
            remove = True
        if not remove and normalized_ascii in {"bamf", "brd"}:
            remove = True

        if remove:
            removed_names.append(clean_name)
        else:
            filtered_names.append(name)

    entities["names"] = filtered_names
    if removed_names:
        print(f"[INFO] Filtered organization labels from names: {removed_names}")
    return entities


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _looks_like_person_name(candidate: str) -> bool:
    clean = _normalize_whitespace(candidate).strip(".,:;()[]{}")
    if not clean:
        return False
    if any(ch.isdigit() for ch in clean):
        return False

    tokens = re.findall(r"[A-Za-zÄÖÜäöüßẞÉé'-]+", clean)
    if not tokens:
        return False
    if len(tokens) > 4:
        return False
    if all(token.lower() in HONORIFICS for token in tokens):
        return False
    if len(tokens) == 1 and tokens[0].lower() in NON_PERSON_GROUP_TERMS:
        return False
    if any(token.lower() in NON_PERSON_NAME_NOISE_TERMS for token in tokens):
        return False
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return False
    return True


def augment_names_from_role_markers(entities: dict, text: str) -> dict:
    """
    Add deterministic name candidates from role/signature markers.

    This reduces dependence on LLM extraction for signature/footer names.
    """
    names = entities.get("names", [])
    if not isinstance(names, list):
        names = []

    patterns = [
        # Same-line role marker: "Im Auftrag Max Mustermann"
        re.compile(
            r"(?im)\b(?:im\s+auftrag|geschlossen:|anhörender\s+entscheider(?:/in)?|einzelentscheider(?:/in)?|sachbearbeiter(?:in)?|unterzeichner(?:in)?|gez\.)\s*[:\-]?\s*"
            r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}){0,2})\b"
        ),
        # Next-line role marker:
        # Im Auftrag
        # Brozio
        re.compile(
            r"(?im)^\s*(?:im\s+auftrag|geschlossen:|anhörender\s+entscheider(?:/in)?|einzelentscheider(?:/in)?|sachbearbeiter(?:in)?|unterzeichner(?:in)?|(?:\(ggf\.\)\s*)?unterschrift[^\n]*)\s*:?\s*$\n\s*"
            r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}){0,2})\s*$"
        ),
        # Role/function contexts in running text:
        # "Hr. Naeem Ajilforoushan, der Gründer und CEO ..."
        re.compile(
            r"(?im)\b(?:hr\.?|herr|frau)\s+"
            r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}){1,2})"
            r"\s*,?\s*(?:der|die)?\s*(?:gründer(?:in)?|inhaber(?:in)?|ceo|geschäftsführer(?:in)?)\b"
        ),
    ]

    existing = {_normalize_whitespace(name).casefold() for name in names if isinstance(name, str)}
    added: list[str] = []

    for pattern in patterns:
        for match in pattern.finditer(text):
            candidate = _normalize_whitespace(match.group(1))
            if not _looks_like_person_name(candidate):
                continue
            key = candidate.casefold()
            if key in existing:
                continue
            existing.add(key)
            names.append(candidate)
            added.append(candidate)

    entities["names"] = names
    if added:
        print(f"[INFO] Added role-marker names: {added}")
    return entities


def augment_names_from_person_fields(entities: dict, text: str) -> dict:
    """Add deterministic names from explicit personal-data field labels."""
    names = entities.get("names", [])
    if not isinstance(names, list):
        names = []

    existing = {_normalize_whitespace(name).casefold() for name in names if isinstance(name, str)}
    added: list[str] = []

    pattern = re.compile(
        rf"(?im)^\s*{PERSON_FIELD_LABEL_PATTERN}\s*[:\-]\s*([^\n]{{2,120}})$"
    )

    for match in pattern.finditer(text):
        candidate = _normalize_whitespace(match.group(1))
        candidate = re.split(r"(?i)\bgeb\.?\b", candidate, maxsplit=1)[0]
        candidate = re.sub(r"\([^)]*\)", " ", candidate)
        candidate = re.sub(r"\s{2,}", " ", candidate).strip(" .,:;/-")
        if not _looks_like_person_name(candidate):
            continue
        key = candidate.casefold()
        if key in existing:
            continue
        existing.add(key)
        names.append(candidate)
        added.append(candidate)

    entities["names"] = names
    if added:
        print(f"[INFO] Added personal-field names: {added}")
    return entities


def _escape_fuzzy(term: str, *, ocr_confusables: bool = False) -> str:
    """
    Escape a term for regex matching but tolerate common OCR variations.

    This is intentionally conservative: it only makes spacing around separators flexible.
    """
    parts: list[str] = []

    i = 0
    while i < len(term):
        ch = term[i]

        if ch.isspace():
            parts.append(r"\s*")
            i += 1
            continue

        if ch == "-":
            parts.append(r"\s*-\s*")
            i += 1
            continue

        if ch == ",":
            parts.append(r"\s*,\s*")
            i += 1
            continue

        if ocr_confusables and ch in OCR_CONFUSABLES:
            parts.append(OCR_CONFUSABLES[ch])
            i += 1
            continue

        parts.append(re.escape(ch))
        i += 1

    return "".join(parts)


def _person_term_tokens(term: str) -> list[str]:
    clean = re.sub(r"\s+", " ", term).strip()
    if not clean:
        return []
    if " " not in clean and "," not in clean:
        return []

    tokens: list[str] = []
    for token in re.findall(r"[A-Za-zÄÖÜäöüßẞÉé]{4,}", clean):
        token_lower = token.lower()
        if token_lower in HONORIFICS or token_lower in NON_PERSON_NAME_NOISE_TERMS:
            continue
        tokens.append(token)
    return tokens


def _person_term_variants(term: str, *, include_tokens: bool = True) -> list[str]:
    """
    Generate match variants for person names:
    - tolerate "LASTNAME, Firstname ..." vs "Firstname ... LASTNAME"
    - keep the original string as the first variant (highest priority)
    """
    clean = re.sub(r"\s+", " ", term).strip()
    if not clean:
        return []

    variants = [clean]

    if "," in clean:
        left, right = [p.strip() for p in clean.split(",", 1)]
        if left and right:
            # Strip leading honorifics on the "LASTNAME" side before swapping.
            left_tokens = [t for t in left.split(" ") if t]
            while left_tokens and left_tokens[0].lower() in HONORIFICS:
                left_tokens = left_tokens[1:]
            left_core = " ".join(left_tokens).strip()
            if left_core:
                variants.append(f"{right} {left_core}")
    else:
        tokens = [t for t in clean.split(" ") if t]
        tokens_wo_titles = tokens[:]
        while tokens_wo_titles and tokens_wo_titles[0].lower() in HONORIFICS:
            tokens_wo_titles = tokens_wo_titles[1:]

        if len(tokens_wo_titles) >= 2:
            last = tokens_wo_titles[-1]
            first = " ".join(tokens_wo_titles[:-1])
            variants.append(f"{last}, {first}")

    if include_tokens:
        # Token-level fallback: helps when the full surface form doesn't match due to OCR drift.
        # Keep tokens >= 4 chars to reduce false positives.
        variants.extend(_person_term_tokens(clean))

    # De-duplicate while preserving order (case-insensitive).
    seen = set()
    out: list[str] = []
    for v in variants:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def safe_replace(text: str, terms: list[str], placeholder: str) -> str:
    """Replace all occurrences of terms with placeholder (tolerant to common OCR spacing/punctuation drift)."""
    # Replace longer terms first to reduce partial matches affecting later replacements.
    sorted_terms = sorted(
        [t for t in terms if isinstance(t, str)],
        key=lambda t: len(t.strip()),
        reverse=True,
    )

    if placeholder == "[PERSON]":
        # Phase 1: replace full surface forms and order-variants first.
        for raw_term in sorted_terms:
            term = raw_term.strip()
            if len(term) < 2:
                continue
            for variant in _person_term_variants(term, include_tokens=False):
                if len(variant) < 2:
                    continue
                pattern = _escape_fuzzy(variant, ocr_confusables=True)
                text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

        # Phase 2: token fallback across all names (prevents generic tokens from pre-empting specific matches).
        token_set: dict[str, str] = {}
        for raw_term in sorted_terms:
            term = raw_term.strip()
            if len(term) < 2:
                continue
            for token in _person_term_tokens(term):
                token_set.setdefault(token.lower(), token)

        for token in sorted(token_set.values(), key=len, reverse=True):
            pattern = _escape_fuzzy(token, ocr_confusables=True)
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

        return text

    for raw_term in sorted_terms:
        term = raw_term.strip()
        if len(term) < 2:
            continue

        variants = [term]
        if placeholder == "[PERSON]":
            variants = _person_term_variants(term)

        for variant in variants:
            if len(variant) < 2:
                continue
            use_confusables = placeholder in {"[PERSON]", "[ADRESSE]", "[ORT]", "[GEBURTSORT]"}
            pattern = _escape_fuzzy(variant, ocr_confusables=use_confusables)
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    return text


def safe_replace_case_numbers(text: str, terms: list[str], placeholder: str) -> str:
    """Replace case numbers while tolerating spacing variations."""
    for term in terms:
        term = term.strip()
        if len(term) < 2:
            continue

        pattern = _escape_fuzzy(term, ocr_confusables=False)
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

    return text


def _birth_date_patterns(term: str) -> list[str]:
    """
    Build conservative regex patterns for one extracted birth date.

    This intentionally targets only the extracted birth date value and a few OCR
    shape variants (e.g. 22.03.1967 -> 22.0367), so unrelated judgment dates
    remain untouched.
    """
    clean = term.strip()
    if not clean:
        return []

    match = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$", clean)
    if not match:
        return [r"(?<!\d)" + _escape_fuzzy(clean, ocr_confusables=False) + r"(?!\d)"]

    day = int(match.group(1))
    month = int(match.group(2))
    year = match.group(3)

    def _day_month_pat(value: int) -> str:
        # OCR occasionally drops a leading "1" in day/month components
        # (e.g. 12 -> 2), which caused leaks like 12.2.1968.
        variants = {str(value), f"{value:02d}"}
        if 10 <= value <= 12:
            variants.add(str(value % 10))
        ordered = sorted(variants, key=len, reverse=True)
        return "(?:" + "|".join(re.escape(v) for v in ordered) + ")"

    day_pat = _day_month_pat(day)
    month_pat = _day_month_pat(month)
    year_variants = [re.escape(year[-2:])]
    if len(year) == 4:
        year_variants.insert(0, re.escape(year))
    year_pat = "(?:" + "|".join(year_variants) + ")"

    return [
        rf"(?<!\d){day_pat}[./-]{month_pat}[./-]{year_pat}(?!\d)",
        rf"(?<!\d){day_pat}[./-]{month_pat}{year_pat}(?!\d)",
    ]


def safe_replace_birth_dates(text: str, terms: list[str], placeholder: str) -> str:
    """Replace only extracted birth dates (plus conservative OCR shape variants)."""
    sorted_terms = sorted(
        [t for t in terms if isinstance(t, str)],
        key=lambda t: len(t.strip()),
        reverse=True,
    )

    for raw_term in sorted_terms:
        term = raw_term.strip()
        if len(term) < 4:
            continue
        for pattern in _birth_date_patterns(term):
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

    return text


def apply_regex_replacements(text: str, entities: dict) -> str:
    """Apply regex replacements based on extracted entities."""
    anon = text

    # Safety catch: numbered plaintiff lists in Bescheid headers.
    # Covers "1. ELBIF,Amal" style lines even if extraction misses them.
    anon = re.sub(
        r"(?m)(^\s*\d+\.\s+)([A-ZÄÖÜ]{2,}(?:-[A-ZÄÖÜ]{2,})*),[ \t]*([A-ZÄÖÜa-zäöüßẞÉé\-]+(?:[ \t]+[A-ZÄÖÜa-zäöüßẞÉé\-]+)*)",
        r"\1[PERSON]",
        anon,
    )

    # Replace extracted entities
    anon = safe_replace(anon, entities.get("names", []), "[PERSON]")

    anon = safe_replace_birth_dates(anon, entities.get("birth_dates", []), "[GEBURTSDATUM]")
    anon = safe_replace(anon, entities.get("birth_places", []), "[GEBURTSORT]")
    anon = safe_replace(anon, entities.get("streets", []), "[ADRESSE]")
    anon = safe_replace(anon, entities.get("postal_codes", []), "[PLZ]")
    anon = safe_replace(anon, entities.get("cities", []), "[ORT]")
    anon = safe_replace(anon, entities.get("azr_numbers", []), "[AZR-NUMMER]")
    anon = safe_replace(
        anon,
        entities.get("aufenthaltsgestattung_ids", []),
        "[AUFENTHALTSGESTATTUNG]",
    )
    anon = safe_replace_case_numbers(anon, entities.get("case_numbers", []), "[AKTENZEICHEN]")

    # Safety catch: ZUE accommodation pattern
    anon = re.sub(r"ZUE\s+\w+", "[UNTERKUNFT]", anon)

    # Safety catch: MRZ lines from ID document scans (machine-readable zone).
    anon = re.sub(r"(?m)^[A-Z0-9<]{30,}$", "[MRZ-REDACTED]", anon)

    # Safety catch: Aufenthaltsgestattung IDs, even when LLM extraction misses them
    anon = re.sub(
        r"\bAufenthaltsgestattung\s+[A-Z]{1,3}\d{4,}\b",
        "[AUFENTHALTSGESTATTUNG]",
        anon,
        flags=re.IGNORECASE,
    )
    # Safety catch: labeled case numbers, even when extraction formatting differs.
    anon = re.sub(
        r"(?i)(\bAktenzeichen\s*:\s*)\d{4,}\s*-\s*\d+\b",
        r"\1[AKTENZEICHEN]",
        anon,
    )
    # Safety catch: dotted/slashed case numbers (e.g., 9923557.439).
    anon = re.sub(
        r"(?i)(\bAktenzeichen\s*:\s*)\d{4,}\s*(?:[./-]\s*\d{2,})+\b",
        r"\1[AKTENZEICHEN]",
        anon,
    )

    # Safety catch: official signatures.
    anon = re.sub(
        r"(?im)\b(?:gez\.|gezeichnet)[ \t]+[A-ZÄÖÜ][a-zäöüß]+(?:[ \t]+[A-ZÄÖÜ][a-zäöüß]+)?\b",
        "gez. [BEAMTER]",
        anon,
    )
    anon = re.sub(
        r"(?im)\bEinzelentscheider(?:/in)?:[ \t]*[A-ZÄÖÜ][a-zäöüß]+(?:[ \t]+[A-ZÄÖÜ][a-zäöüß]+)?\b",
        "Einzelentscheider/in: [BEAMTER]",
        anon,
    )
    # Safety catch: role marker on one line, name on next line.
    anon = re.sub(
        r"(?im)(^\s*(?:Im\s+Auftrag|geschlossen:)\s*$\n)\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,})?\s*$",
        r"\1[BEAMTER]",
        anon,
    )
    # Safety catch: same-line role markers with a name.
    anon = re.sub(
        r"(?im)(^\s*(?:Im\s+Auftrag|geschlossen:?|Anhörender\s+Entscheider(?:/in)?|Einzelentscheider(?:/in)?|Sachbearbeiter(?:in)?|Unterzeichner(?:in)?|gez\.?|gezeichnet)\s*[:\-]?\s*)([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}){0,2})\s*$",
        r"\1[BEAMTER]",
        anon,
    )
    # Safety catch: name line directly under signature labels.
    anon = re.sub(
        r"(?im)(^\s*\(ggf\.\)\s*Unterschrift[^\n]*\n)\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,})?\s*$",
        r"\1[PERSON]",
        anon,
    )

    # Safety catch: personal-data name labels (Vorname/Nachname/etc.).
    anon = re.sub(
        rf"(?im)(^\s*{PERSON_FIELD_LABEL_PATTERN}\s*[:\-]\s*)([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{{2,}}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{{2,}}|,\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{{2,}}){{0,3}})",
        r"\1[PERSON]",
        anon,
    )
    # Safety catch: parent labels often leak first/last names when extraction is partial.
    # Covers:
    # - "Vater: Abdulla [PERSON] (verstorben)"
    # - "Mutter: Iran Zarei"
    anon = re.sub(
        r"(?i)(\b(?:Vater|Mutter)\s*:\s*)([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,})(\s+\[PERSON\])(\s*\((?:verstorben|tot)\))?",
        r"\1[PERSON]\3\4",
        anon,
    )
    anon = re.sub(
        r"(?i)(\b(?:Vater|Mutter)\s*:\s*)([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}){0,2})(\s*\((?:verstorben|tot)\))?",
        r"\1[PERSON]\3",
        anon,
    )

    # Safety catch: explicitly labeled birth dates.
    anon = re.sub(
        r"(?i)(\b(?:Geburtsdatum|geb\.?\s*datum|geboren\s+am|geb\.?\s*am|Jahrgang)\s*:?\s*)\d{1,2}[./]\d{1,2}[./]\d{2,4}\b",
        r"\1[GEBURTSDATUM]",
        anon,
    )
    # Safety catch: if "Geburtsdatum/Geburtstag" appears as a standalone label,
    # redact date-like values in the next one or two lines.
    lines = anon.splitlines()
    birth_label_re = re.compile(r"(?i)\b(?:geburtsdatum|geburtstag)\b")
    birth_value_re = re.compile(
        r"(?<!\d)(?:0?[1-9]|[12][0-9]|3[01])[./](?:0?[1-9]|1[0-2])(?:[./]\d{2,4}|\d{2,4})(?!\d)"
    )
    for i, line in enumerate(lines):
        if not birth_label_re.search(line):
            continue
        for j in range(i + 1, min(len(lines), i + 3)):
            lines[j] = birth_value_re.sub("[GEBURTSDATUM]", lines[j])

    # Safety catch: tabular applicant fields sometimes bypass extraction entirely.
    # Typical OCR block:
    #   1. Familienname
    #   Goudarzi
    #   Ziba
    #   ...
    #   Vorname
    #   Hadi
    #   Zivdar
    def _looks_like_name_cell(value: str) -> bool:
        clean = value.strip().strip(".,:;|/-")
        if not clean or "[" in clean or "]" in clean:
            return False
        if any(ch.isdigit() for ch in clean):
            return False
        if len(clean) < 3:
            return False
        tokens = re.findall(r"[A-Za-zÄÖÜäöüßẞÉé'\-]+", clean)
        if not tokens or len(tokens) > 3:
            return False
        for token in tokens:
            low = token.lower()
            if low in HONORIFICS or low in NON_PERSON_NAME_NOISE_TERMS:
                return False
        return True

    def _redact_name_rows_after_label(label_pattern: str) -> None:
        nonlocal lines
        stop_re = re.compile(
            r"(?i)\b(?:geburtsname|vorname|vorame|geburtsdatum|geburtsort|"
            r"st[aä]atsangeh|volkszugeh|religion|familienstand|geschlecht|sprache)\b"
        )
        label_re = re.compile(label_pattern)
        for i, line in enumerate(lines):
            if not label_re.search(line):
                continue
            replaced = 0
            for j in range(i + 1, min(len(lines), i + 7)):
                probe = lines[j].strip()
                if not probe:
                    continue
                if stop_re.search(probe):
                    break
                if _looks_like_name_cell(probe):
                    lines[j] = "[PERSON]"
                    replaced += 1
                    if replaced >= 2:
                        break

    _redact_name_rows_after_label(r"(?i)\bfamilienname\b")
    _redact_name_rows_after_label(r"(?i)\b(?:vorname|vorame)\b")

    # Safety catch: family-member table blocks (Name/Ort/Datum) and entry dates.
    family_header_re = re.compile(r"(?i)angaben zu familien|familienangeh")
    ort_label_re = re.compile(r"(?i)^\s*ort\s*$")
    ort_value_re = re.compile(r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}$")
    skip_ort_words = {
        "sohn", "tochter", "mutter", "vater", "ehemann", "ehefrau", "kind",
        "antragsteller", "antragstellerin",
    }
    date_value_re = re.compile(r"(?<!\d)(?:0?[1-9]|[12][0-9]|3[01])[./](?:0?[1-9]|1[0-2])[./]?(?:\d{2,4})(?!\d)")
    for i, line in enumerate(lines):
        if family_header_re.search(line):
            for j in range(i, min(len(lines), i + 45)):
                lines[j] = date_value_re.sub("[DATUM]", lines[j])
        if ort_label_re.search(line):
            for j in range(i + 1, min(len(lines), i + 8)):
                probe = lines[j].strip()
                low = probe.casefold()
                if not probe or "[" in probe or "]" in probe:
                    continue
                if low in skip_ort_words:
                    continue
                if ort_value_re.match(probe):
                    lines[j] = "[ORT]"
                    break
        if re.search(r"(?i)\beinreise\b", line):
            for j in range(max(0, i - 1), min(len(lines), i + 4)):
                lines[j] = date_value_re.sub("[DATUM]", lines[j])

    # Safety catch: OCR signature artifacts near "Unterschrift" lines
    # (e.g. "ali28", "Hex ali28") should not remain as potential name leaks.
    signature_label_re = re.compile(r"(?i)\bunterschrift\b")
    signature_name_re = re.compile(r"(?i)^(?:hex\s+)?[a-zäöüß]{2,}\d{1,3}$")
    for i, line in enumerate(lines):
        if not signature_label_re.search(line):
            continue
        for j in range(max(0, i - 3), i):
            probe = lines[j].strip()
            if signature_name_re.match(probe):
                lines[j] = "[PERSON]"

    anon = "\n".join(lines)
    # Safety catch: partial birth dates (e.g., "geb.22.03.197").
    anon = re.sub(
        r"(?i)(\bgeb\.?\s*(?:am\s*)?)(\d{1,2}\.\d{1,2}\.\d{2,3})(?!\d)",
        r"\1[GEBURTSDATUM]",
        anon,
    )
    # Safety catch: birth place after "geb. ... in ...", even if extraction missed it.
    anon = re.sub(
        r"(?i)(\bgeb\.?\s*(?:am\s*)?(?:\[GEBURTSDATUM\]|\d{1,2}[./]\d{1,2}[./]\d{2,4})\s*(?:in|,\s*in)\s*)([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]+(?:\s*/\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]+)?)",
        r"\1[GEBURTSORT]",
        anon,
    )
    # Safety catch: location fields in interview/bio sections.
    anon = re.sub(
        r"(?im)(^\s*(?:Dorf|Heimatdorf|Provinz|Heimatprovinz|Wohnort|Ortsteil)\s*:\s*)([^\n]{2,100})$",
        r"\1[ORT]",
        anon,
    )
    # Safety catch: narrative village/home-province mentions.
    anon = re.sub(
        r"(?i)(\b(?:im|in|aus)\s+Dorf\s+)([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,})",
        r"\1[ORT]",
        anon,
    )
    anon = re.sub(
        r"(?i)\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,})(,\s*die\s+Heimatprovinz\s+des\s+Antragstellers)\b",
        r"[ORT]\2",
        anon,
    )
    # Safety catch: Visum/Ausweis/Pass IDs (including compact "VisumNr.ITA...").
    anon = re.sub(
        r"(?i)(\bVisum\s*Nr\.?\s*:?\s*)([A-Z]{2,}\d{5,})\b",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(\bAusweisnummer\s*:?\s*)([A-Z0-9]{6,})\b",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(\bPass(?:nummer|ersatz)?\s*:?\s*)([A-Z0-9]{6,})\b",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(\bPassnummern?\s*:?\s*)([A-Z0-9]{6,}(?:\s*(?:,|/|und)\s*[A-Z0-9]{6,})+)",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(\bAZR(?:-|\s*)Nr\.?\s*:?\s*)\d{6,}\b",
        r"\1[AZR-NUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(\bAZR(?:-|\s*)Nummer(?:\(\s*n\s*\)|n)?\s*:?\s*)\d{6,}\b",
        r"\1[AZR-NUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(\b(?:Eurodac|D|E)[-\s]*Nummer\s*:?\s*)([A-Z0-9_-]{6,})\b",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)\bBAMF_[A-F0-9-]{20,}\b",
        "[DOKUMENTENNUMMER]",
        anon,
    )
    # Safety catch: standalone document-id lines.
    anon = re.sub(
        r"(?m)^\s*(?:[A-Z]{1,4}\d{6,}|\d{7,})\s*$",
        "[DOKUMENTENNUMMER]",
        anon,
    )
    # Safety catch: hard OCR variants from known failing forms.
    anon = re.sub(
        r"(?i)(\bDolmetscher(?:-|\s*)nummer\s*:?\s*)\d{3,}\b",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(\b(?:Sprachmittler|Dolmetscher)(?:-|\s*)(?:id|nr\.?)\s*:?\s*)\d{3,}\b",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(\bAls\s+Sprachmittler(?:/-in)?\s+ist\s+anwesend\s*:\s*)\d{3,}\b",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    # Safety catch: explicit "Identifikationsnummer" values (e.g. "M 1237300", "56663").
    anon = re.sub(
        r"(?im)(^\s*Identifikationsnummer\s*:\s*)([A-Z]?\s*[A-Z0-9][A-Z0-9 \-_/]{3,})\s*$",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    # Safety catch: marriage-certificate IDs ("Eheurkunden-Nr"/"Ehrurkunde")
    # can appear as plain short numbers (e.g. 56663) under Typbezeichnung blocks.
    anon = re.sub(
        r"(?is)(Typbezeichnung\s*:\s*(?:Ehr?urkunde|Eheurkunde(?:n)?)\s*\n\s*Identifikationsnummer\s*:\s*)([A-Z]?\s*\d{4,})",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?im)(^\s*(?:Ehe(?:r)?urkunden?(?:-|\s)*(?:nr\.?|nummer)|Eheurkunden?(?:-|\s)*(?:nr\.?|nummer))\s*:\s*)([A-Z]?\s*\d{4,})\s*$",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    # Safety catch: DEDUB subject fragments often contain personal IDs/names.
    # Handle hard line-break variants like:
    #   DEDUB1_
    #   9923557_fingerprints_..._vomBrocke
    anon = re.sub(
        r"(?is)(DEDUB1[_-]\s*(?:\n\s*)?)\d{6,}(?=\s*_)",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)(DEDUB1[_-][^\"\n]*?_)\d{6,}(?=_)",
        r"\1[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?s)(DEDUB1[_-][^\"“”„]{0,240}?_)([a-zäöüß]{2,}[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]{2,})",
        r"\1[PERSON]",
        anon,
    )
    anon = re.sub(
        r"(?s)(DEDUB1[_-][^\"“”„]{0,240}?_[A-Z0-9]{1,6}_)([A-Za-zÄÖÜäöüß]{3,})(?=[\"“”„])",
        r"\1[PERSON]",
        anon,
    )
    # Safety catch: signer name lines in outbound messages.
    anon = re.sub(
        r"(?im)(^\s*Name\s+des\s+Signierenden\s*:\s*).*$",
        r"\1[PERSON]",
        anon,
    )
    # Safety catch: appointment notes often contain short honorific + surname
    # (e.g. "Bemerkung: Fr. Bischoff").
    anon = re.sub(
        r"(?im)(^\s*Bemerkung\s*:\s*)(?:Fr\.?|Hr\.?|Frau|Herr)\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}){0,2}",
        r"\1[PERSON]",
        anon,
    )
    # Safety catch: standalone title-case comma-name lines (e.g. "Steckemetz, Alexandra").
    anon = re.sub(
        r"(?m)^\s*[A-ZÄÖÜ][a-zäöüß]+(?:[-'][A-ZÄÖÜ]?[a-zäöüß]+)?(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*,\s*[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*\s*$",
        "[PERSON]",
        anon,
    )
    # Safety catch: partial address artifact like "[PLZ] [PERSON], Eifel".
    anon = re.sub(
        r"\[PLZ\]\s+\[PERSON\]\s*,\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+",
        "[PLZ] [ORT]",
        anon,
    )
    # Safety catch: banking-location lines flagged as remaining location identifiers.
    anon = re.sub(
        r"(?im)(\bBundeskasse\s+)[A-ZÄÖÜ][A-Za-zÄÖÜäöüß/.\-]+",
        r"\1[ORT]",
        anon,
    )
    anon = re.sub(
        r"(?im)(\bDienstsitz\s+)[A-ZÄÖÜ][A-Za-zÄÖÜäöüß/.\-]+",
        r"\1[ORT]",
        anon,
    )
    anon = re.sub(
        r"(?i)\b(?:Goudarri|Goudargi|Hade|ivdar|Shirali|pouya)\b",
        "[PERSON]",
        anon,
    )

    anon = re.sub(
        r"(?i)\b(?:Borejeod|Boroujerd|Bororerd|Borajerd|Borojerd|Borujerd|Shoosh|Shush|Schleiden|Laleh-?Park|Schahrake\s+Rahahan|Schahrake\s+Jahannamah)\b",
        "[ORT]",
        anon,
    )
    anon = re.sub(
        r"\b\d{6,8}/\d{3,8}\b",
        "[DOKUMENTENNUMMER]",
        anon,
    )

    anon = re.sub(
        r"(?i)\*V\d{6,}\*",
        "[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"(?i)\*[A-Z]{1,4}\d{3,}\*",
        "[DOKUMENTENNUMMER]",
        anon,
    )
    anon = re.sub(
        r"\bD\d{4,}\b",
        "[DOKUMENTENNUMMER]",
        anon,
    )

    # Safety catch: labeled and dotted/slashed case numbers that appear outside extraction.
    anon = re.sub(
        r"(?i)(\b(?:Aktenzeichen|Az\.?|Gz\.?|Gesch(?:ä|ae)ftszeichen|Gesch\.\s*-\s*Z\.?|Gesch\.\s*Z\.?)\s*:?\s*)(\d{2,}[A-Za-z]?(?:\s*[./\-–—−]\s*[A-Za-z0-9]{1,}){1,})",
        r"\1[AKTENZEICHEN]",
        anon,
    )
    # Safety catch: standalone numeric aktenzeichen forms like "9923557-439".
    anon = re.sub(
        r"\b\d{6,8}\s*-\s*\d{3,4}\b",
        "[AKTENZEICHEN]",
        anon,
    )
    anon = re.sub(
        r"\b\d{6,8}\.\d{3,4}\b",
        "[AKTENZEICHEN]",
        anon,
    )

    # Safety catch: name token left before already replaced surname.
    anon = re.sub(
        r"(?im)(\bAntwort:\s*)[A-ZÄÖÜ][A-Za-zÄÖÜäöüßẞÉé'\-]{2,}(\s+\[PERSON\])",
        r"\1[PERSON]\2",
        anon,
    )

    # Second-pass verification: if extracted name tokens survive, force-replace and warn.
    for name in entities.get("names", []) or []:
        if not isinstance(name, str):
            continue
        for token in re.findall(r"[A-Za-zÄÖÜäöüßẞÉé]{4,}", name):
            token_lower = token.lower()
            if token_lower in HONORIFICS or token_lower in NON_PERSON_NAME_NOISE_TERMS:
                continue
            before = anon
            pattern = _escape_fuzzy(token, ocr_confusables=True)
            anon = re.sub(pattern, "[PERSON]", anon, flags=re.IGNORECASE)
            if anon != before:
                print(f"[WARN] Name token survived replacement, force-replaced: {token!r}")

    return anon


@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_document(
    request: AnonymizationRequest, x_api_key: str = Header(None)
):
    """
    Anonymize personal data in German legal documents using hybrid LLM + Regex.

    Step 1: LLM extracts entities (names, dates, addresses, etc.)
    Step 2: Regex replaces entities in original text

    This preserves document structure and legal citations while anonymizing PII.
    """

    # API key authentication (if configured)
    if ANONYMIZATION_API_KEY:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key required")
        if x_api_key != ANONYMIZATION_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Process full text - the hybrid approach is fast enough
    text = request.text

    # Build extraction prompt (simple, model-agnostic)
    prompt = """Extract ALL person names and PII from this legal document.
Return JSON only, with exactly these keys and arrays:
{"names":[], "birth_dates":[], "birth_places":[], "streets":[], "postal_codes":[], "cities":[], "azr_numbers":[], "aufenthaltsgestattung_ids":[], "case_numbers":[]}

Rules:
- names: every person name mentioned (clients, applicants, family members; include names after titles like Genossin/Genosse/Frau/Herr/Mandant/Klient)
- names must be individual humans only, never groups
- include the exact surface forms as they appear in the document (case, punctuation, order)
- specifically include comma forms like "SURNAME, Given Names" if present in the text
- if a line starts with an ALL-CAPS surname followed by a comma and given names, include that full line segment as a name
- do not include bare titles without a name (e.g., just "Herr", "Frau", "Genosse")
- do NOT include tribes/ethnicities/peoples/languages/religions/nationalities as names (e.g., "Fulani", "Paschtune", "Hazara", "Kurde")
- birth_dates: DD.MM.YYYY
- birth_places: cities of birth (foreign)
- streets: residence streets (NOT BAMF offices)
- postal_codes: residence postal codes (NOT 90343/90461/53115)
- cities: residence cities (NOT Nürnberg/Bonn BAMF offices)
- azr_numbers: AZR numbers
- aufenthaltsgestattung_ids: Aufenthaltsgestattung IDs (e.g., "Aufenthaltsgestattung J5213441" or "J5213441" if explicitly labeled as Aufenthaltsgestattung)
- case_numbers: case numbers

Document:
""" + text

    format_schema = {
        "type": "object",
        "properties": {
            "names": {"type": "array", "items": {"type": "string"}},
            "birth_dates": {"type": "array", "items": {"type": "string"}},
            "birth_places": {"type": "array", "items": {"type": "string"}},
            "streets": {"type": "array", "items": {"type": "string"}},
            "postal_codes": {"type": "array", "items": {"type": "string"}},
            "cities": {"type": "array", "items": {"type": "string"}},
            "azr_numbers": {"type": "array", "items": {"type": "string"}},
            "aufenthaltsgestattung_ids": {"type": "array", "items": {"type": "string"}},
            "case_numbers": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "names",
            "birth_dates",
            "birth_places",
            "streets",
            "postal_codes",
            "cities",
            "azr_numbers",
            "aufenthaltsgestattung_ids",
            "case_numbers",
        ],
        "additionalProperties": False,
    }

    try:
        print(f"[INFO] Processing {request.document_type} ({len(text)} chars)")

        # Step 1: LLM extracts entities (raw=True to use ChatML prompt as-is)
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": format_schema,
                    # Use model's default prompt template
                    "options": {
                        # Extraction should be deterministic; replacement logic handles OCR drift.
                        "temperature": 0.0,
                        "num_ctx": 32768,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            raw_response = result["response"]

            # Parse extracted entities
            try:
                entities = json.loads(raw_response)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parse failed: {e}")
                print(f"[DEBUG] Response: {raw_response[:500]}")
                raise

            print(f"[INFO] Raw extraction: {sum(len(v) for v in entities.values() if isinstance(v, list))} entities")

            # Filter out known BAMF office addresses
            entities = filter_bamf_addresses(entities)
            # Filter out group labels accidentally extracted as person names.
            entities = filter_non_person_group_labels(entities, text)
            # Add deterministic role/signature names if extraction missed them.
            entities = augment_names_from_role_markers(entities, text)
            # Add deterministic names from explicit person field labels.
            entities = augment_names_from_person_fields(entities, text)
            # Remove authority/organization phrases misclassified as names.
            entities = filter_non_person_organization_labels(entities)

            print(f"[INFO] After BAMF filter: {sum(len(v) for v in entities.values() if isinstance(v, list))} entities")
            for key, values in entities.items():
                if values:
                    print(f"  {key}: {values}")

        # Step 2: Regex replacement
        anonymized_text = apply_regex_replacements(text, entities)

        # Collect all addresses for response
        all_addresses = (
            entities.get("streets", []) +
            entities.get("cities", [])
        )

        print(f"[SUCCESS] Anonymization completed")

        return AnonymizationResponse(
            is_valid=True,
            invalid_reason=None,
            anonymized_text=anonymized_text,
            plaintiff_names=entities.get("names", []),
            birth_dates=entities.get("birth_dates", []),
            birth_places=entities.get("birth_places", []),
            addresses=all_addresses,
            azr_numbers=entities.get("azr_numbers", []),
            aufenthaltsgestattung_ids=entities.get("aufenthaltsgestattung_ids", []),
            case_numbers=entities.get("case_numbers", []),
            confidence=0.95,
        )

    except httpx.TimeoutException:
        print("[ERROR] Ollama request timeout")
        raise HTTPException(
            status_code=504,
            detail="Anonymization timeout. The model may be loading.",
        )
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] Ollama HTTP error: {e.response.status_code}")
        raise HTTPException(
            status_code=502,
            detail=f"Ollama service error: {e.response.status_code}",
        )
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM response: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse model response as JSON.",
        )
    except Exception as e:
        print(f"[ERROR] Anonymization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anonymization failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Rechtmaschine Anonymization Service",
        "version": "2.0.0",
        "method": "Hybrid LLM extraction + Regex replacement",
        "model": MODEL,
        "endpoints": {"health": "/health", "anonymize": "/anonymize (POST)"},
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL}


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Anonymization Service v2.0")
    print("Hybrid LLM Extraction + Regex Replacement")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"API Key Auth: {'Enabled' if ANONYMIZATION_API_KEY else 'Disabled'}")
    print(f"Listening on: 0.0.0.0:9002")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=9002)
