"""
Rechtmaschine Anonymization Service

This service runs on the home PC with GPU and provides document anonymization
for German asylum law documents (Anhörung and Bescheid) using Qwen3-14B via Ollama.

Architecture:
- Runs on home PC with 12GB VRAM
- Accessed via Tailscale mesh network from production server
- Uses Ollama with Qwen3:14b model
- Provides REST API on port 8002

Usage:
    python anonymization_service.py

Environment Variables:
    OLLAMA_URL: URL to Ollama API (default: http://localhost:11435)
    OLLAMA_MODEL: Model to use (default: qwen3:14b)
    ANONYMIZATION_API_KEY: API key for authentication (optional)
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import os
import json

app = FastAPI(
    title="Rechtmaschine Anonymization Service",
    description="Document anonymization service for German asylum law documents",
    version="1.0.0"
)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11435/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
ANONYMIZATION_API_KEY = os.getenv("ANONYMIZATION_API_KEY")

class AnonymizationRequest(BaseModel):
    text: str
    document_type: str  # e.g., "Anhörung", "Bescheid", "Sonstige gespeicherte Quellen"

class AnonymizationResponse(BaseModel):
    is_valid: bool = True  # Is this a valid asylum document?
    invalid_reason: str | None = None  # Reason if invalid (e.g., "Rundschreiben", "Vollmacht")
    anonymized_text: str
    plaintiff_names: list[str]
    addresses: list[str] = []  # Extracted addresses
    confidence: float

@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_document(
    request: AnonymizationRequest,
    x_api_key: str = Header(None)
):
    """
    Anonymize plaintiff names in German legal documents.

    Only anonymizes plaintiff/applicant names, NOT:
    - Judge names
    - Lawyer names
    - Authority names (BAMF, etc.)
    - Place names
    - Court names
    """

    # API key authentication (if configured)
    if ANONYMIZATION_API_KEY:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key required")
        if x_api_key != ANONYMIZATION_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # Document type is passed to the LLM for context; the LLM validates internally
    # No strict type restriction - allow any document type

    # Limit to first ~1 page (2000 chars) - plaintiff names are typically at the beginning
    # Reduced from 4000 to avoid JSON truncation issues with Ollama
    text_limit = 2000
    limited_text = request.text[:text_limit]
    remaining_tail = request.text[text_limit:]

    # Prepare prompt for LLM with Qwen3-specific formatting
    # Adjust validation based on document type
    if request.document_type in ["Anhörung", "Bescheid"]:
        validation_instruction = """FIRST: Validate if this is actually a BAMF asylum document.
- Valid: BAMF decisions (Bescheid), hearing protocols (Anhörung)
- INVALID: Circulars (Rundschreiben), power of attorney (Vollmacht), forms, cover letters

If INVALID, set is_valid to false and provide the reason."""
    else:
        validation_instruction = "Always set is_valid to true. This is a document from saved sources."

    prompt = f"""<|im_start|>system
You are an anonymization system for German legal and medical documents.

{validation_instruction}

Identify and anonymize:
1. NAMES of the subject (plaintiff/applicant/patient and their family members)
2. ADDRESSES of the subject (street, house number, postal code, city)

Look for:
- Names after: Kläger, Klägerin, Antragsteller, Antragstellerin, Patient, Patientin, Herr, Frau, geb.
- Addresses: street names with numbers, postal codes (5 digits), residential addresses

DO NOT anonymize:
- Judges, lawyers, doctors, officials, authorities
- Court addresses, office addresses, hospital addresses
- Country names, regions (these are not personal addresses)

Replace with:
- Names: [PERSON_1], [PERSON_2], etc.
- Addresses: [ADRESSE]

Output valid JSON:
{{
  "is_valid": true/false,
  "invalid_reason": "only if invalid",
  "plaintiff_names": ["name1", "name2"],
  "addresses": ["Musterstr. 12, 12345 Berlin"],
  "anonymized_text": "text with [PERSON_1] and [ADRESSE] placeholders",
  "confidence": 0.95
}}
<|im_end|>
<|im_start|>user
Document type: {request.document_type}

Text:
{limited_text}
<|im_end|>
<|im_start|>assistant
"""

    try:
        print(f"[INFO] Processing anonymization request for {request.document_type}")
        print(f"[INFO] Text length: {len(request.text)} characters (processing first {min(len(request.text), text_limit)} chars)")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.0,
                        "num_ctx": 8192,
                        "num_predict": 16384
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            raw_response = result["response"]

            # Debug: Print response length and preview
            print(f"[DEBUG] Raw LLM response length: {len(raw_response)} characters")
            print(f"[DEBUG] Response preview: {raw_response[:300]}...")

            # Try to parse JSON with error handling
            try:
                llm_output = json.loads(raw_response)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Initial JSON parse failed at char {e.pos}: {e.msg}")
                print(f"[DEBUG] Full response: {raw_response}")

                # Try to extract JSON from response (in case there's extra text)
                import re
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    print(f"[DEBUG] Attempting to parse extracted JSON...")
                    try:
                        llm_output = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        print(f"[ERROR] Failed to parse extracted JSON")
                        raise
                else:
                    print(f"[ERROR] No JSON structure found in response")
                    raise

            # Check if document is valid
            is_valid = llm_output.get("is_valid", True)
            invalid_reason = llm_output.get("invalid_reason")

            if not is_valid:
                print(f"[INFO] Document rejected: {invalid_reason}")
                return AnonymizationResponse(
                    is_valid=False,
                    invalid_reason=invalid_reason,
                    anonymized_text="",
                    plaintiff_names=[],
                    addresses=[],
                    confidence=0.0
                )

            anonymized_section = llm_output.get("anonymized_text", "")

            if remaining_tail:
                separator = ""
                if anonymized_section and not anonymized_section.endswith("\n"):
                    separator = "\n\n"
                anonymized_section = f"{anonymized_section}{separator}{remaining_tail}"

            print(f"[SUCCESS] Anonymization completed")
            print(f"[INFO] Found {len(llm_output.get('plaintiff_names', []))} plaintiff names")
            print(f"[INFO] Found {len(llm_output.get('addresses', []))} addresses")
            print(f"[INFO] Confidence: {llm_output.get('confidence', 0.0)}")

            return AnonymizationResponse(
                is_valid=True,
                invalid_reason=None,
                anonymized_text=anonymized_section,
                plaintiff_names=llm_output.get("plaintiff_names", []),
                addresses=llm_output.get("addresses", []),
                confidence=llm_output.get("confidence", 0.0)
            )

    except httpx.TimeoutException:
        print("[ERROR] Ollama request timeout (>120s)")
        raise HTTPException(
            status_code=504,
            detail="Anonymization timeout. The model may be loading or the request is too complex."
        )
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] Ollama HTTP error: {e.response.status_code}")
        raise HTTPException(
            status_code=502,
            detail=f"Ollama service error: {e.response.status_code}"
        )
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse model response. The model may not have returned valid JSON."
        )
    except Exception as e:
        print(f"[ERROR] Anonymization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anonymization failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns service status and model information.
    """
    try:
        # Check if Ollama is reachable
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11435/api/version")
            ollama_version = response.json().get("version", "unknown")
    except Exception as e:
        print(f"[WARNING] Ollama health check failed: {e}")
        ollama_version = "unreachable"

    return {
        "status": "healthy",
        "model": MODEL,
        "ollama_version": ollama_version,
        "ollama_url": OLLAMA_URL
    }

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Rechtmaschine Anonymization Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "anonymize": "/anonymize (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Rechtmaschine Anonymization Service")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"API Key Auth: {'Enabled' if ANONYMIZATION_API_KEY else 'Disabled'}")
    print(f"Listening on: 0.0.0.0:9002")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=9002)
