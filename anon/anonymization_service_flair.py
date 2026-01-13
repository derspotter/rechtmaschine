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

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import re
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict

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
        _tagger = SequenceTagger.load('flair/ner-german-legal')
        _tagger.to(device)
        print("[INFO] Flair NER model loaded successfully")
    return _tagger

app = FastAPI(
    title="Rechtmaschine Anonymization Service (Flair)",
    description="Fast document anonymization using Flair German Legal NER",
    version="2.1.0"
)

ANONYMIZATION_API_KEY = os.getenv("ANONYMIZATION_API_KEY")

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
    'PER',   # Person (plaintiffs, family)
    'STR',   # Straße (street)
    'LDS',   # Landschaft (region) - optional
}

# Entity types to PRESERVE (don't anonymize)
PRESERVE_TAGS = {
    'RR',    # Richter (Judge)
    'AN',    # Anwalt (Lawyer)
    'GRT',   # Gericht (Court)
    'ORG',   # Organisation
    'GS',    # Gesetz (Law)
    'RS',    # Rechtsprechung
}

# =============================================================================
# GERMAN CASE INFLECTION
# =============================================================================

# Patterns to detect grammatical case from preceding articles/prepositions
CASE_PATTERNS = {
    'genitiv': re.compile(
        r'\b(des|der|eines|einer|meines|meiner|seines|seiner|ihres|ihrer|'
        r'dieses|dieser|jenes|jener|welches|welcher)\s*$',
        re.IGNORECASE
    ),
    'dativ': re.compile(
        r'\b(dem|der|einem|einer|meinem|meiner|seinem|seiner|ihrem|ihrer|'
        r'diesem|dieser|jenem|jener|welchem|welcher|'
        r'von|mit|bei|nach|aus|zu|seit|gegenüber)\s*$',
        re.IGNORECASE
    ),
    'akkusativ': re.compile(
        r'\b(den|einen|meinen|seinen|ihren|diesen|jenen|welchen|'
        r'für|durch|gegen|ohne|um)\s*$',
        re.IGNORECASE
    ),
}

# Placeholder inflections for each case
PLACEHOLDER_INFLECTIONS = {
    'person': {
        'nominativ': 'Kläger',
        'genitiv': 'Klägers',
        'dativ': 'Kläger',
        'akkusativ': 'Kläger',
    },
    'person_female': {
        'nominativ': 'Klägerin',
        'genitiv': 'Klägerin',
        'dativ': 'Klägerin',
        'akkusativ': 'Klägerin',
    },
    'family': {
        'nominativ': 'Familienangehöriger',
        'genitiv': 'Familienangehörigen',
        'dativ': 'Familienangehörigen',
        'akkusativ': 'Familienangehörigen',
    },
    'child': {
        'nominativ': 'Kind',
        'genitiv': 'Kindes',
        'dativ': 'Kind',
        'akkusativ': 'Kind',
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

    return 'nominativ'  # Default

def get_inflected_placeholder(entity_type: str, case: str, index: int) -> str:
    """Get the correctly inflected placeholder."""
    inflections = PLACEHOLDER_INFLECTIONS.get(entity_type, PLACEHOLDER_INFLECTIONS['person'])
    base = inflections.get(case, inflections['nominativ'])

    if index > 0:
        return f"[{base.upper()}_{index + 1}]"
    return f"[{base.upper()}]"


# =============================================================================
# FAMILY RELATION DETECTION
# =============================================================================

# Family relation keywords (German)
FAMILY_RELATIONS = {
    # Spouse
    'ehemann', 'ehefrau', 'ehegatte', 'ehegattin', 'ehepartner', 'ehepartnerin',
    'gatte', 'gattin', 'mann', 'frau', 'lebensgefährte', 'lebensgefährtin',
    'lebenspartner', 'lebenspartnerin', 'partner', 'partnerin',
    # Children
    'sohn', 'tochter', 'kind', 'kinder', 'söhne', 'töchter',
    # Parents
    'vater', 'mutter', 'eltern', 'elternteil',
    # Siblings
    'bruder', 'schwester', 'geschwister', 'brüder', 'schwestern',
    # Extended family
    'onkel', 'tante', 'neffe', 'nichte', 'cousin', 'cousine',
    'großvater', 'großmutter', 'opa', 'oma', 'enkel', 'enkelin',
    'schwager', 'schwägerin', 'schwiegervater', 'schwiegermutter',
    'schwiegersohn', 'schwiegertochter',
}

# Pattern to find family relations followed by names
FAMILY_PATTERN = re.compile(
    r'\b(sein[es]?|ihr[es]?|der|die|dessen|deren)\s+'
    r'(' + '|'.join(FAMILY_RELATIONS) + r')\b'
    r'(?:\s+(?:des|der|dem|den))?\s*'
    r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)?',
    re.IGNORECASE
)

# Direct family mention pattern
DIRECT_FAMILY_PATTERN = re.compile(
    r'\b(' + '|'.join(FAMILY_RELATIONS) + r')\s+'
    r'([A-ZÄÖÜ][a-zäöüß]+(?:[-\s][A-ZÄÖÜ][a-zäöüß]+)?)',
    re.IGNORECASE
)


# =============================================================================
# REGEX PATTERNS FOR PII
# =============================================================================

DOB_PATTERN = re.compile(
    r'\b(\d{1,2})\.\s*(\d{1,2})\.\s*(19|20)\d{2}\b'
)
DOB_CUE_PATTERN = re.compile(
    r'\b(geboren\s+am|geb\.|geb\s|Geburtsdatum|Jahrgang)\s*:?\s*'
    r'(\d{1,2}\.\s*\d{1,2}\.\s*(?:19|20)\d{2})',
    re.IGNORECASE
)
PLZ_CITY_PATTERN = re.compile(
    r'\b(\d{5})\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b'
)
ADDRESS_PATTERN = re.compile(
    r'\b([A-ZÄÖÜ][a-zäöüß]+(?:straße|str\.|weg|platz|allee|gasse|ring|damm|ufer))\s*(\d+\s*[a-zA-Z]?)\b',
    re.IGNORECASE
)
ADDRESS_CUE_PATTERN = re.compile(
    r'\b(wohnhaft\s+in|Adresse|Anschrift|wohnt\s+in)\s*:?\s*'
    r'([A-ZÄÖÜ][a-zäöüß\s]+\d+[a-zA-Z]?(?:\s*,\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+)?)',
    re.IGNORECASE
)


# =============================================================================
# MAIN ANONYMIZATION FUNCTION
# =============================================================================

def anonymize_with_flair(text: str) -> Tuple[str, List[str], List[str], List[str], float]:
    """
    Anonymize text using Flair NER with:
    - Consistent pseudonyms (PERSON_1, PERSON_2)
    - Case-inflected placeholders
    - Family relation detection

    Returns: (anonymized_text, plaintiff_names, family_members, addresses, confidence)
    """
    from flair.data import Sentence

    tagger = get_tagger()

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
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > MAX_CHUNK_SIZE // 2:
                chunk = remaining[:break_point + 1]

        chunks.append((offset, chunk))
        offset += len(chunk)
        remaining = remaining[len(chunk):]

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
        for entity in sentence.get_spans('ner'):
            all_entities.append({
                'text': entity.text,
                'tag': entity.tag,
                'start': chunk_offset + entity.start_position,
                'end': chunk_offset + entity.end_position,
                'score': entity.score
            })

    # ==========================================================================
    # BUILD ENTITY REGISTRY WITH CONSISTENT PSEUDONYMS
    # ==========================================================================

    # Track unique persons with consistent IDs
    person_registry: Dict[str, int] = OrderedDict()  # name -> index
    family_registry: Dict[str, int] = OrderedDict()  # name -> index

    plaintiff_names = []
    family_members = []
    addresses = []

    # First pass: identify all persons and categorize
    for ent in all_entities:
        if ent['tag'] == 'PER':
            name = ent['text'].strip()
            # Check if this is a family member based on context
            context_start = max(0, ent['start'] - 50)
            context = text[context_start:ent['start']].lower()

            is_family = any(rel in context for rel in FAMILY_RELATIONS)

            if is_family:
                if name not in family_registry:
                    family_registry[name] = len(family_registry)
                    family_members.append(name)
            else:
                if name not in person_registry:
                    person_registry[name] = len(person_registry)
                    plaintiff_names.append(name)

    # Also detect family members via regex patterns
    for match in DIRECT_FAMILY_PATTERN.finditer(text):
        relation = match.group(1).lower()
        name = match.group(2).strip()
        if name and name not in family_registry and name not in person_registry:
            family_registry[name] = len(family_registry)
            family_members.append(name)

    print(f"[INFO] Found {len(person_registry)} plaintiffs, {len(family_registry)} family members")

    # ==========================================================================
    # BUILD REPLACEMENT LIST
    # ==========================================================================

    entities_to_replace = []

    for ent in all_entities:
        if ent['tag'] == 'PER':
            name = ent['text'].strip()
            case = detect_case(text, ent['start'])

            if name in family_registry:
                idx = family_registry[name]
                # Determine if child or other family
                context = text[max(0, ent['start']-30):ent['start']].lower()
                if any(w in context for w in ['sohn', 'tochter', 'kind', 'kinder']):
                    replacement = get_inflected_placeholder('child', case, idx)
                else:
                    replacement = get_inflected_placeholder('family', case, idx)
            elif name in person_registry:
                idx = person_registry[name]
                replacement = get_inflected_placeholder('person', case, idx)
            else:
                replacement = '[PERSON]'

            entities_to_replace.append((ent['start'], ent['end'], replacement))

        elif ent['tag'] == 'STR':
            if ent['text'] not in addresses:
                addresses.append(ent['text'])
            entities_to_replace.append((ent['start'], ent['end'], '[ADRESSE]'))

    # Add regex-based detections
    # DOB with cue phrases
    for match in DOB_CUE_PATTERN.finditer(text):
        entities_to_replace.append((match.start(), match.end(), '[GEBURTSDATUM]'))

    # Standalone DOB
    for match in DOB_PATTERN.finditer(text):
        # Check if not already covered
        already_covered = any(
            start <= match.start() and end >= match.end()
            for start, end, _ in entities_to_replace
        )
        if not already_covered:
            entities_to_replace.append((match.start(), match.end(), '[GEBURTSDATUM]'))

    # Address with cue phrases
    for match in ADDRESS_CUE_PATTERN.finditer(text):
        addr = match.group(2)
        if addr not in addresses:
            addresses.append(addr)
        entities_to_replace.append((match.start(), match.end(), match.group(1) + ' [ADRESSE]'))

    # Standalone addresses
    for match in ADDRESS_PATTERN.finditer(text):
        already_covered = any(
            start <= match.start() and end >= match.end()
            for start, end, _ in entities_to_replace
        )
        if not already_covered:
            full_match = match.group(0)
            if full_match not in addresses:
                addresses.append(full_match)
            entities_to_replace.append((match.start(), match.end(), '[ADRESSE]'))

    # PLZ + City
    for match in PLZ_CITY_PATTERN.finditer(text):
        already_covered = any(
            start <= match.start() and end >= match.end()
            for start, end, _ in entities_to_replace
        )
        if not already_covered:
            entities_to_replace.append((match.start(), match.end(), '[ORT]'))

    # ==========================================================================
    # APPLY REPLACEMENTS
    # ==========================================================================

    # Sort by start position descending
    entities_to_replace.sort(key=lambda x: x[0], reverse=True)

    # Remove overlapping entities (keep first = longest due to sort)
    filtered_entities = []
    last_start = len(text) + 1
    for start, end, replacement in entities_to_replace:
        if end <= last_start:
            filtered_entities.append((start, end, replacement))
            last_start = start

    # Apply replacements
    anonymized = list(text)
    for start, end, replacement in filtered_entities:
        anonymized[start:end] = list(replacement)

    anonymized_text = ''.join(anonymized)

    # Calculate confidence
    if all_entities:
        avg_score = sum(e['score'] for e in all_entities) / len(all_entities)
    else:
        avg_score = 0.5

    return anonymized_text, plaintiff_names, family_members, addresses, avg_score


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_document(
    request: AnonymizationRequest,
    x_api_key: str = Header(None)
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
        anonymized_text, plaintiff_names, family_members, addresses, confidence = anonymize_with_flair(
            request.text
        )

        print(f"[SUCCESS] Found {len(plaintiff_names)} plaintiffs, {len(family_members)} family, {len(addresses)} addresses")
        print(f"[INFO] Confidence: {confidence:.2%}")

        return AnonymizationResponse(
            is_valid=True,
            invalid_reason=None,
            anonymized_text=anonymized_text,
            plaintiff_names=plaintiff_names,
            family_members=family_members,
            addresses=addresses,
            confidence=confidence
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
            "version": "2.1.0",
            "features": [
                "case-inflected placeholders",
                "consistent pseudonyms",
                "family detection"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/")
async def root():
    return {
        "service": "Rechtmaschine Anonymization Service (Flair)",
        "version": "2.1.0",
        "model": "flair/ner-german-legal",
        "features": [
            "GPU acceleration",
            "Full document processing",
            "Case-inflected placeholders (des Klägers, dem Kläger)",
            "Consistent pseudonyms (KLÄGER_1, KLÄGER_2)",
            "Family member detection (Ehemann, Kind, etc.)"
        ],
        "endpoints": {
            "health": "/health",
            "anonymize": "/anonymize (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Anonymization Service (Flair NER) v2.1.0")
    print("=" * 60)
    print("Model: flair/ner-german-legal")
    print("Features:")
    print("  - GPU acceleration (auto-detected)")
    print("  - Full document processing")
    print("  - Case-inflected placeholders")
    print("  - Consistent pseudonyms")
    print("  - Family member detection")
    print("Listening on: 0.0.0.0:9002")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=9002)
