"""
Rechtmaschine Anonymization Service (Flair NER - Simplified)

Simple, robust anonymization: replaces ALL detected names with [PERSON].
No categorization - GDPR-safe by design.

Features:
- GPU acceleration (auto-detected)
- Full document processing
- Replaces all person names with [PERSON]
- Anonymizes addresses and dates of birth

Requirements:
    pip install flair fastapi uvicorn

Usage:
    python anonymization_service_flair_simple.py
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import re
from typing import Optional, List, Tuple

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
    title="Rechtmaschine Anonymization Service (Flair - Simple)",
    description="Simple, robust anonymization: replaces ALL names with [PERSON]",
    version="3.0.2"
)

ANONYMIZATION_API_KEY = os.getenv("ANONYMIZATION_API_KEY")

class AnonymizationRequest(BaseModel):
    text: str
    document_type: str

class AnonymizationResponse(BaseModel):
    is_valid: bool = True
    invalid_reason: Optional[str] = None
    anonymized_text: str
    plaintiff_names: list[str] = []  # Kept for API compatibility, always empty
    family_members: list[str] = []   # Kept for API compatibility, always empty
    addresses: list[str] = []
    confidence: float


# =============================================================================
# REGEX PATTERNS FOR PII
# =============================================================================

# Date of birth patterns
DOB_PATTERN = re.compile(
    r'\b(\d{1,2})\.\s*(\d{1,2})\.\s*(19|20)\d{2}\b'
)
DOB_CUE_PATTERN = re.compile(
    r'\b(geboren\s+am|geb\.|geb\s|Geburtsdatum|Jahrgang)\s*:?\s*'
    r'(\d{1,2}\.\s*\d{1,2}\.\s*(?:19|20)\d{2})',
    re.IGNORECASE
)

# Address patterns
PLZ_CITY_PATTERN = re.compile(
    r'(?:,\s*)?(\d{5})\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b'
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

def anonymize_simple(text: str) -> Tuple[str, List[str], float]:
    """
    Simple anonymization: replace ALL detected person names with [PERSON].

    Returns: (anonymized_text, addresses, confidence)
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

    print(f"[INFO] Found {len(all_entities)} entities via NER")

    # ==========================================================================
    # BUILD REPLACEMENT LIST
    # ==========================================================================

    entities_to_replace = []
    addresses = []

    # Anonymize entities detected by Flair NER
    # Flair detects: PER, STR, ST, LD, LDS, RR, AN, GRT, etc. (19 types total)
    for ent in all_entities:
        if ent['tag'] in ('PER', 'RR', 'AN'):  # Person, Richter (Judge), Anwalt (Lawyer)
            # Simple mode: anonymize ALL people regardless of role
            entities_to_replace.append((ent['start'], ent['end'], '[PERSON]'))
        elif ent['tag'] == 'STR':  # Straße (Street)
            if ent['text'] not in addresses:
                addresses.append(ent['text'])
            entities_to_replace.append((ent['start'], ent['end'], '[ADRESSE]'))
        elif ent['tag'] == 'ST':  # Stadt (City)
            # Filter out numeric-only detections (false positives like translator IDs)
            entity_text = ent['text'].strip()
            is_numeric = entity_text.isdigit()
            print(f"[DEBUG] ST tag: '{ent['text']}' (stripped: '{entity_text}', isdigit: {is_numeric})")
            if not is_numeric:
                entities_to_replace.append((ent['start'], ent['end'], '[ORT]'))
        elif ent['tag'] in ('LD', 'LDS'):  # Land/Landschaft (Country/Region)
            pass  # Keep for context - "Iran", "Syrien" are important for asylum cases

    # Add regex-based detections for what Flair doesn't catch
    # (PLZ postal codes, dates of birth)
    # DOB with cue phrases
    for match in DOB_CUE_PATTERN.finditer(text):
        entities_to_replace.append((match.start(), match.end(), '[GEBURTSDATUM]'))

    # Standalone DOB
    for match in DOB_PATTERN.finditer(text):
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

    # Sort by: start position ascending, then by length descending (longer matches first)
    entities_to_replace.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Remove overlapping entities (prefer longer matches)
    filtered_entities = []
    covered_ranges = []

    for start, end, replacement in entities_to_replace:
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

    anonymized_text = ''.join(anonymized)

    # Calculate confidence
    if all_entities:
        avg_score = sum(e['score'] for e in all_entities) / len(all_entities)
    else:
        avg_score = 0.5

    return anonymized_text, addresses, avg_score


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_document(
    request: AnonymizationRequest,
    x_api_key: str = Header(None)
):
    """
    Anonymize ALL person names and addresses in German legal documents.

    Simple approach: replaces all detected names with [PERSON].
    GDPR-safe by design - no risk of missing categorization.
    """

    if ANONYMIZATION_API_KEY:
        if not x_api_key or x_api_key != ANONYMIZATION_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    print(f"[INFO] Processing {request.document_type}: {len(request.text)} characters")

    try:
        anonymized_text, addresses, confidence = anonymize_simple(request.text)

        print(f"[SUCCESS] Anonymized all names, found {len(addresses)} addresses")
        print(f"[INFO] Confidence: {confidence:.2%}")

        return AnonymizationResponse(
            is_valid=True,
            invalid_reason=None,
            anonymized_text=anonymized_text,
            plaintiff_names=[],  # Not extracted in simple mode
            family_members=[],   # Not extracted in simple mode
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
            "version": "3.0.2",
            "mode": "simple",
            "features": [
                "Anonymizes ALL person names",
                "GDPR-safe by design",
                "Full document processing",
                "GPU acceleration"
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
        "service": "Rechtmaschine Anonymization Service (Flair - Simple)",
        "version": "3.0.2",
        "mode": "simple",
        "model": "flair/ner-german-legal",
        "description": "Replaces ALL detected person names with [PERSON]. GDPR-safe by design.",
        "features": [
            "GPU acceleration",
            "Full document processing",
            "Anonymizes all person names (no categorization)",
            "Addresses and dates of birth detection"
        ],
        "endpoints": {
            "health": "/health",
            "anonymize": "/anonymize (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Anonymization Service (Flair - Simple) v3.0.2")
    print("=" * 60)
    print("Model: flair/ner-german-legal")
    print("Mode: Simple (anonymizes ALL names)")
    print("Features:")
    print("  - GPU acceleration (auto-detected)")
    print("  - Full document processing")
    print("  - Replaces all person names with [PERSON]")
    print("  - GDPR-safe by design")
    print("Listening on: 0.0.0.0:9002")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=9002)
