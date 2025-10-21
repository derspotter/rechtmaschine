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
    document_type: str  # "Anhörung" or "Bescheid"

class AnonymizationResponse(BaseModel):
    anonymized_text: str
    plaintiff_names: list[str]
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

    # Validate document type
    if request.document_type not in ["Anhörung", "Bescheid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type '{request.document_type}'. Must be 'Anhörung' or 'Bescheid'."
        )

    # Limit to first ~2 pages (4000 chars) - plaintiff names are typically at the beginning
    text_limit = 4000
    limited_text = request.text[:text_limit]

    # Prepare prompt for LLM with Qwen3-specific formatting
    prompt = f"""<|im_start|>system
You are an anonymization system for German asylum law documents. Your task is to identify and anonymize ONLY the names of plaintiffs/applicants and their family members.

DO NOT anonymize:
- Judge names
- Lawyer names
- Authority names (BAMF, Bundesamt, etc.)
- Place names
- Court names
- Decision maker names

Output must be valid JSON with this exact structure:
{{
  "plaintiff_names": ["name1", "name2"],
  "anonymized_text": "the document with placeholders like [ANTRAGSTELLER] or [FAMILY_MEMBER_1]",
  "confidence": 0.95
}}
<|im_end|>
<|im_start|>user
Document type: {request.document_type}

Document text (first 2 pages):
{limited_text}

Identify plaintiff names and provide anonymized version in JSON format.
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
                        "num_predict": 4096
                    }
                }
            )
            response.raise_for_status()
            result = response.json()

            # Debug: Print raw LLM response
            print(f"[DEBUG] Raw LLM response: {result['response'][:500]}...")

            # Parse the LLM response
            llm_output = json.loads(result["response"])

            print(f"[SUCCESS] Anonymization completed")
            print(f"[INFO] Found {len(llm_output.get('plaintiff_names', []))} plaintiff names")
            print(f"[INFO] Confidence: {llm_output.get('confidence', 0.0)}")

            return AnonymizationResponse(
                anonymized_text=llm_output.get("anonymized_text", ""),
                plaintiff_names=llm_output.get("plaintiff_names", []),
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
