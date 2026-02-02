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
    case_numbers: list[str] = []
    confidence: float


# Known BAMF office addresses to filter out
BAMF_OFFICE_DATA = {
    "postal_codes": {"90343", "90461", "53115"},
    "cities": {"nürnberg", "bonn"},
    "streets": {"frankenstraße", "frankenstrabe", "reuterstraße", "reuterstrabe"},
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


def safe_replace(text: str, terms: list[str], placeholder: str) -> str:
    """Replace all occurrences of terms with placeholder."""
    for term in terms:
        term = term.strip()
        if len(term) < 2:
            continue
        pattern = re.escape(term)
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    return text


def apply_regex_replacements(text: str, entities: dict) -> str:
    """Apply regex replacements based on extracted entities."""
    anon = text

    # Replace extracted entities
    anon = safe_replace(anon, entities.get("names", []), "[PERSON]")
    anon = safe_replace(anon, entities.get("birth_dates", []), "[GEBURTSDATUM]")
    anon = safe_replace(anon, entities.get("birth_places", []), "[GEBURTSORT]")
    anon = safe_replace(anon, entities.get("streets", []), "[ADRESSE]")
    anon = safe_replace(anon, entities.get("postal_codes", []), "[PLZ]")
    anon = safe_replace(anon, entities.get("cities", []), "[ORT]")
    anon = safe_replace(anon, entities.get("azr_numbers", []), "[AZR-NUMMER]")
    anon = safe_replace(anon, entities.get("case_numbers", []), "[AKTENZEICHEN]")

    # Safety catch: ZUE accommodation pattern
    anon = re.sub(r"ZUE\s+\w+", "[UNTERKUNFT]", anon)

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

    # Build extraction prompt (concise + /no_think to disable Qwen3 thinking mode)
    prompt = """<|im_start|>system
Extract applicant PII from this asylum document. Return JSON only. /no_think

Find the applicant (Antragsteller/Antragstellerin/Kläger) data:
- names: Full name
- birth_dates: Birth date (DD.MM.YYYY)
- birth_places: Birthplace city (foreign)
- streets: Residence street (NOT BAMF office)
- postal_codes: Residence postal code (NOT 90343/90461/53115)
- cities: Residence city (NOT Nürnberg/Bonn office)
- azr_numbers: AZR numbers
- case_numbers: Case numbers

JSON:
{"names":[], "birth_dates":[], "birth_places":[], "streets":[], "postal_codes":[], "cities":[], "azr_numbers":[], "case_numbers":[]}
<|im_end|>
<|im_start|>user
""" + text + """
<|im_end|>
<|im_start|>assistant
"""

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
                    "format": "json",
                    "raw": True,  # Required for ChatML format with /no_think
                    "options": {
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
