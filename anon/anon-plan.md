# Anonymization Service Architecture Plan

## Overview

This document outlines the architecture for adding document anonymization to Rechtmaschine using a local LLM (Qwen3-14B) running on a home PC with 12GB VRAM, connected to the production server via Tailscale.

## Problem Statement

- **Challenge**: Need to anonymize plaintiff/applicant names in "Anhörung" and "Bescheid" documents
- **Privacy Requirement**: Sensitive asylum documents should not be sent to external APIs
- **Infrastructure Constraint**: Production server lacks GPU; home PC has 12GB VRAM but is behind CG-NAT
- **Solution**: Run local LLM on home PC, expose via Tailscale mesh network

## Architecture

### Two-Machine Setup

```
┌─────────────────────────────────────────┐
│ Server (rechtmaschine.de)               │
│ ┌─────────────────────────────────────┐ │
│ │ Caddy (host network)                │ │
│ │   ├─> rechtmaschine-app:8000       │ │
│ │   └─> n8n:5678                      │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ Tailscale (host)                    │ │
│ │ IP: 100.64.1.3                      │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
              │
              │ Tailscale Mesh (encrypted)
              ↓
┌─────────────────────────────────────────┐
│ Home PC (behind CG-NAT)                 │
│ ┌─────────────────────────────────────┐ │
│ │ Tailscale (host)                    │ │
│ │ IP: 100.64.1.2                      │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ Ollama + Qwen3-14B                  │ │
│ │ localhost:11434                     │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ Anonymization Service               │ │
│ │ 0.0.0.0:8001                        │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Technology Choices

### LLM Model: Qwen3-14B-Instruct-Q5_K_M

**Selected for:**
- 14.8B parameters - excellent German language support
- Released April 2025 (most recent suitable model)
- Trained on 36 trillion tokens across 119 languages
- Q5_K_M quantization fits in 12GB VRAM (~10GB usage)
- 32K context window for long documents
- Apache 2.0 license

**Alternatives Considered:**
- Qwen3-VL-30B-A3B: Too large (30B params), needs aggressive Q4 quantization
- Magistral Small 1.2: 24B params, requires 24GB VRAM even quantized
- Llama 3.1 8B: Smaller but less capable for complex German legal text

### Network Solution: Tailscale

**Why Tailscale:**
- ✅ Bypasses CG-NAT automatically
- ✅ No port forwarding needed
- ✅ Stable IPs (100.x.x.x range)
- ✅ Encrypted mesh network
- ✅ Free for personal use
- ✅ Auto-reconnects on network changes
- ✅ Simple installation

**Alternatives Considered:**
- PIA VPN with port forwarding: IP/port changes frequently, complex scripting, public exposure
- Cloudflare Tunnel: Good but adds dependency
- ngrok: Not suitable for production, session limits

### LLM Runtime: Ollama

**Why Ollama:**
- Simple installation and model management
- Efficient quantization support
- HTTP API (localhost:11434)
- Automatic GPU acceleration
- Active development and community

## Implementation Plan

### Phase 1: Network Setup

#### Step 1.1: Install Tailscale on Server
```bash
# On server (rechtmaschine.de)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# Note down Tailscale IP (e.g., 100.64.1.3)
```

#### Step 1.2: Install Tailscale on Home PC
```bash
# On home PC
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# Note down Tailscale IP (e.g., 100.64.1.2)
```

#### Step 1.3: Update Server Configuration
Add to `/var/opt/docker/rechtmaschine/app/.env`:
```bash
ANONYMIZATION_SERVICE_URL=http://100.64.1.2:8001
```

### Phase 2: Home PC Setup

#### Step 2.1: Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Step 2.2: Pull Qwen3-14B Model
```bash
ollama pull qwen3:14b-instruct-q5_K_M
```

#### Step 2.3: Create Anonymization Service

**Directory Structure:**
```
~/rechtmaschine-anonymization/
├── anonymization_service.py
├── requirements.txt
├── Dockerfile (optional)
└── docker-compose.yml (optional)
```

**File: anonymization_service.py**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="Rechtmaschine Anonymization Service")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:14b-instruct-q5_K_M"

class AnonymizationRequest(BaseModel):
    text: str
    document_type: str  # "Anhörung" or "Bescheid"

class AnonymizationResponse(BaseModel):
    anonymized_text: str
    plaintiff_names: list[str]
    confidence: float

@app.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_document(request: AnonymizationRequest):
    """Anonymize plaintiff names in German legal documents"""

    prompt = f"""Du bist ein Anonymisierungssystem für deutsche Asylrechtsdokumente vom Typ "{request.document_type}".

AUFGABE: Identifiziere und anonymisiere NUR die Namen des Antragstellers/Klägers und seiner Familienangehörigen.

WICHTIG - NICHT ANONYMISIEREN:
- Richternamen
- Anwaltsnamen
- Behördennamen (BAMF, Bundesamt, etc.)
- Ortsnamen
- Gerichtsnamen
- Entscheiderdaten

DOKUMENT:
{request.text[:8000]}

Antworte im JSON-Format:
{{
  "plaintiff_names": ["Name1", "Name2"],
  "anonymized_text": "Das anonymisierte Dokument mit Platzhaltern...",
  "confidence": 0.95
}}
"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
            )
            response.raise_for_status()
            result = response.json()

            # Parse the LLM response
            import json
            llm_output = json.loads(result["response"])

            return AnonymizationResponse(**llm_output)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anonymization failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

**File: requirements.txt**
```txt
fastapi==0.115.0
uvicorn==0.31.0
httpx==0.28.1
pydantic==2.9.0
```

**File: Dockerfile** (optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY anonymization_service.py .
CMD ["python", "anonymization_service.py"]
```

**File: docker-compose.yml** (optional)
```yaml
version: '3.8'

services:
  anonymization:
    build: .
    container_name: rechtmaschine-anonymization
    restart: unless-stopped
    ports:
      - "8001:8001"
    network_mode: host  # Allows access to Ollama on localhost
    environment:
      - OLLAMA_URL=http://localhost:11434
```

#### Step 2.4: Start Service

**Option A: Direct Python (for testing)**
```bash
cd ~/rechtmaschine-anonymization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python anonymization_service.py
```

**Option B: Docker (for production)**
```bash
cd ~/rechtmaschine-anonymization
docker-compose up -d
```

### Phase 3: Server Integration

#### Step 3.1: Test Connectivity
```bash
# From server
curl -X POST http://100.64.1.2:8001/health
# Expected: {"status":"healthy","model":"qwen3:14b-instruct-q5_K_M"}
```

#### Step 3.2: Add Anonymization Endpoint to app.py

```python
# In app.py
import httpx
from typing import Optional

ANONYMIZATION_SERVICE_URL = os.getenv("ANONYMIZATION_SERVICE_URL")

class AnonymizationRequest(BaseModel):
    text: str
    document_type: str

class AnonymizationResult(BaseModel):
    anonymized_text: str
    plaintiff_names: list[str]
    confidence: float
    original_text: str

async def anonymize_document_text(text: str, document_type: str) -> Optional[AnonymizationResult]:
    """Call anonymization service on home PC"""
    if not ANONYMIZATION_SERVICE_URL:
        print("Warning: ANONYMIZATION_SERVICE_URL not configured")
        return None

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ANONYMIZATION_SERVICE_URL}/anonymize",
                json={
                    "text": text,
                    "document_type": document_type
                }
            )
            response.raise_for_status()
            data = response.json()
            return AnonymizationResult(
                anonymized_text=data["anonymized_text"],
                plaintiff_names=data["plaintiff_names"],
                confidence=data["confidence"],
                original_text=text
            )
    except Exception as e:
        print(f"Anonymization service error: {e}")
        return None

@app.post("/documents/{document_id}/anonymize")
@limiter.limit("5/hour")
async def anonymize_document_endpoint(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db)
):
    """Anonymize a classified document"""

    # Get document from database
    doc_uuid = uuid.UUID(document_id)
    document = db.query(Document).filter(Document.id == doc_uuid).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Only anonymize Anhörung and Bescheid
    if document.category not in ["Anhörung", "Bescheid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Document type '{document.category}' does not support anonymization"
        )

    # Extract text from PDF
    pdf_path = document.file_path
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")

    extracted_text = extract_pdf_text(pdf_path, max_pages=50)

    # Call anonymization service
    result = await anonymize_document_text(extracted_text, document.category)

    if not result:
        raise HTTPException(status_code=503, detail="Anonymization service unavailable")

    # Store in processed_documents table
    processed_doc = ProcessedDocument(
        document_id=doc_uuid,
        extracted_text=result.original_text,
        is_anonymized=True,
        anonymization_metadata={
            "plaintiff_names": result.plaintiff_names,
            "confidence": result.confidence,
            "anonymized_at": datetime.now().isoformat()
        },
        processing_status="completed"
    )
    db.add(processed_doc)
    db.commit()

    return {
        "status": "success",
        "anonymized_text": result.anonymized_text,
        "plaintiff_names": result.plaintiff_names,
        "confidence": result.confidence
    }
```

#### Step 3.3: Update Frontend (app.py HTML section)

Add anonymization button to document cards:
```javascript
// In the document card rendering
if (doc.category === 'Anhörung' || doc.category === 'Bescheid') {
    html += `<button onclick="anonymizeDocument('${doc.id}')"
                style="margin-left: 8px; padding: 4px 8px; font-size: 11px;">
                🔒 Anonymisieren
             </button>`;
}

async function anonymizeDocument(docId) {
    if (!confirm('Dokument anonymisieren? Dies kann 30-60 Sekunden dauern.')) return;

    try {
        const response = await fetch(`/documents/${docId}/anonymize`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('Anonymization failed');

        const result = await response.json();

        // Show result in modal
        showAnonymizationResult(result);
    } catch (error) {
        alert('Fehler bei der Anonymisierung: ' + error.message);
    }
}

function showAnonymizationResult(result) {
    // Display anonymized text in a modal
    const modal = document.createElement('div');
    modal.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                    background: rgba(0,0,0,0.5); display: flex; align-items: center;
                    justify-content: center; z-index: 1000;">
            <div style="background: white; padding: 30px; border-radius: 8px;
                        max-width: 800px; max-height: 80vh; overflow-y: auto;">
                <h2>Anonymisiertes Dokument</h2>
                <p><strong>Gefundene Namen:</strong> ${result.plaintiff_names.join(', ')}</p>
                <p><strong>Konfidenz:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                <div style="white-space: pre-wrap; font-family: monospace;
                            background: #f5f5f5; padding: 15px; border-radius: 4px;
                            margin-top: 15px;">
                    ${result.anonymized_text}
                </div>
                <button onclick="this.closest('div').parentElement.remove()"
                        style="margin-top: 20px; padding: 10px 20px;">
                    Schließen
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}
```

### Phase 4: Database Schema (Already Exists)

The `processed_documents` table is already defined in `models.py`:
```python
class ProcessedDocument(Base):
    __tablename__ = "processed_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'))
    extracted_text = Column(Text)
    is_anonymized = Column(Boolean, default=False)
    ocr_applied = Column(Boolean, default=False)
    anonymization_metadata = Column(JSONB)  # Stores plaintiff names, confidence
    processing_status = Column(String(50))  # pending, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
```

## Workflow

### User Flow

1. User uploads PDF → Classified as "Anhörung" or "Bescheid"
2. User clicks "🔒 Anonymisieren" button on document card
3. Frontend shows loading state ("Processing...")
4. Backend:
   - Extracts text from PDF using `extract_pdf_text()`
   - Sends text to home PC anonymization service via Tailscale
   - Receives anonymized text + identified names
   - Stores in `processed_documents` table
5. Frontend displays anonymized text in modal with:
   - List of identified plaintiff names
   - Confidence score
   - Full anonymized text
6. User can view/download anonymized version

### Data Flow

```
User Upload
    ↓
[Server] Classification (GPT-5-mini)
    ↓
[Database] Store document metadata
    ↓
User clicks "Anonymisieren"
    ↓
[Server] Extract PDF text (pikepdf)
    ↓
[Server] HTTP POST to http://100.64.1.2:8001/anonymize
    ↓
[Tailscale Tunnel - Encrypted]
    ↓
[Home PC] Anonymization Service
    ↓
[Home PC] Ollama + Qwen3-14B processes text
    ↓
[Tailscale Tunnel - Encrypted]
    ↓
[Server] Receive anonymized text
    ↓
[Database] Store in processed_documents
    ↓
[Frontend] Display in modal
```

## Security Considerations

### Network Security
- ✅ Tailscale provides end-to-end encryption (WireGuard-based)
- ✅ No public exposure of anonymization service
- ✅ Only server IP whitelisted via Tailscale ACLs (optional)

### Data Security
- ✅ Sensitive documents never leave your infrastructure
- ✅ No external API calls with document content
- ✅ Text extraction happens on server (trusted environment)
- ✅ Anonymized results stored separately from originals

### Authentication
- Consider adding API key authentication between server and home PC
- Example: Add `X-API-Key` header validation in anonymization service

```python
# In anonymization_service.py
from fastapi import Header, HTTPException

API_KEY = os.getenv("ANONYMIZATION_API_KEY", "change-me-in-production")

@app.post("/anonymize")
async def anonymize_document(
    request: AnonymizationRequest,
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... rest of function
```

## Performance Considerations

### Processing Time
- **Expected**: 30-60 seconds per document
- **Factors**:
  - Document length (limited to first 8000 chars in prompt)
  - GPU utilization
  - Network latency (minimal with Tailscale)

### Optimization Strategies
1. **Batch Processing**: Process multiple documents in queue
2. **Caching**: Cache anonymization results by document hash
3. **Partial Anonymization**: Only process first N pages
4. **Progress Updates**: Use WebSocket/SSE for real-time progress

### Resource Usage
- **Home PC GPU**: ~10GB VRAM (Qwen3-14B Q5)
- **Home PC RAM**: ~4-6GB for service + Ollama overhead
- **Network**: ~100KB-1MB per request (text only)

## Error Handling

### Failure Scenarios

1. **Home PC offline**:
   - Server detects connection timeout
   - Returns 503 error to user
   - Suggestion: Add retry queue (Redis) for async processing

2. **Model loading failure**:
   - Health check endpoint fails
   - Service auto-restarts (Docker restart policy)

3. **Low confidence anonymization**:
   - If confidence < 0.7, warn user
   - Show manual review interface

4. **PDF extraction failure**:
   - Fallback to OCR (not implemented yet)
   - Return error to user

## Monitoring & Logging

### Health Checks

**Server side:**
```bash
# Add to cron: */5 * * * *
curl -f http://100.64.1.2:8001/health || echo "Anonymization service down"
```

**Home PC side:**
```bash
# Monitor GPU usage
nvidia-smi
# Monitor Ollama logs
journalctl -u ollama -f
```

### Metrics to Track
- Anonymization requests per day
- Average processing time
- Success/failure rate
- Identified names per document (aggregate stats)
- Service uptime

## Future Enhancements

### Phase 5: Advanced Features (Optional)

1. **Queue-Based Processing**:
   - Add Redis queue
   - Process documents asynchronously
   - Email/notification when complete

2. **Multiple Home PCs**:
   - Load balancing across multiple GPUs
   - Failover support

3. **OCR Integration**:
   - For scanned documents
   - Tesseract or PaddleOCR

4. **Manual Review Interface**:
   - Show original + anonymized side-by-side
   - Allow manual corrections
   - Re-run anonymization with feedback

5. **Pseudonymization**:
   - Maintain consistent pseudonyms across documents
   - "Person A" is always the same person

6. **Export Options**:
   - Export anonymized text as PDF
   - Track anonymization history

## Testing Strategy

### Desktop Iteration Notes (2026-01-15)
- Desktop service: `http://desktop:9002` (Tailscale).
- Service reports `mode=simple` with `flair/ner-german-legal`.
- Regression PDF used:
  - `/var/opt/docker/rechtmaschine/extracted_documents/gpt-5-mini/2024-11-11_5599883_24K2623_9923557-439_1pdf_Beiakte_02_Bescheid_p275-345.pdf`
- Observation: `ZIVDAR` / `ZIBA` surnames remain in output (only given name anonymized), so detected PERSON spans are incomplete.
- Saved local outputs for review:
  - `anon/original_bescheid_p275-345.txt`
  - `anon/anonymized_bescheid_p275-345.txt`
  - `anon/anonymized_bescheid_p275-345_flair.txt`

### Unit Tests
```bash
# Test anonymization service locally
curl -X POST http://localhost:8001/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Der Kläger Mustafa Müller stellte einen Asylantrag...",
    "document_type": "Bescheid"
  }'
```

### Integration Tests
```bash
# Test from server
curl -X POST http://100.64.1.2:8001/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test document...",
    "document_type": "Anhörung"
  }'
```

### Load Testing
```bash
# Simulate multiple concurrent requests
ab -n 10 -c 2 -p test_payload.json \
   -T application/json \
   http://100.64.1.2:8001/anonymize
```

## Rollback Plan

If issues arise:
1. **Disable anonymization feature** in frontend (comment out button)
2. **Stop service** on home PC: `docker-compose down`
3. **Revert environment variable** on server
4. Documents remain classified and accessible (no data loss)

## Cost Analysis

### Infrastructure Costs
- **Home PC electricity**: ~$20-30/month (24/7 operation)
- **Tailscale**: $0 (free tier sufficient)
- **Server costs**: No change (no GPU added)

### Comparison to Cloud Solutions
- **AWS GPU instance (g4dn.xlarge)**: ~$350/month
- **RunPod A4000**: ~$150/month
- **Our solution**: ~$25/month (electricity only)

**Savings**: ~$125-325/month

## Deployment Checklist

### Server Setup
- [ ] Install Tailscale
- [ ] Note Tailscale IP
- [ ] Add `ANONYMIZATION_SERVICE_URL` to `.env`
- [ ] Update `app.py` with anonymization endpoint
- [ ] Update frontend with anonymization button
- [ ] Restart app: `docker-compose restart app`
- [ ] Test health check from server

### Home PC Setup
- [ ] Install Tailscale
- [ ] Note Tailscale IP
- [ ] Install Ollama
- [ ] Pull Qwen3-14B model
- [ ] Create `~/rechtmaschine-anonymization/` directory
- [ ] Create service files (Python script, requirements, etc.)
- [ ] Start service
- [ ] Verify GPU usage with `nvidia-smi`
- [ ] Test health check: `curl localhost:8001/health`

### Connectivity Test
- [ ] From server: `curl http://100.64.1.2:8001/health`
- [ ] Test full anonymization with sample text
- [ ] Upload test document and try anonymization
- [ ] Verify result in database `processed_documents` table

### Production Readiness
- [ ] Add API key authentication
- [ ] Set up monitoring/alerting
- [ ] Configure Docker restart policy
- [ ] Document troubleshooting steps
- [ ] Train user on anonymization feature

## Maintenance

### Regular Tasks
- **Weekly**: Check service logs for errors
- **Monthly**: Update Ollama and models
- **Quarterly**: Review anonymization quality
- **As needed**: Restart service after model updates

### Update Procedures

**Update Qwen3 model:**
```bash
ollama pull qwen3:14b-instruct-q5_K_M  # Get latest version
docker-compose restart anonymization    # Restart service
```

**Update Python dependencies:**
```bash
pip install --upgrade -r requirements.txt
docker-compose build && docker-compose up -d
```

## Support & Troubleshooting

### Common Issues

**Issue**: Service unreachable from server
- Check Tailscale status: `tailscale status`
- Verify service is running: `curl localhost:8001/health`
- Check firewall: Tailscale should bypass, but verify

**Issue**: Slow anonymization (>2 minutes)
- Check GPU usage: `nvidia-smi`
- Verify model is loaded in VRAM
- Consider reducing text length in prompt

**Issue**: Low quality anonymization
- Review prompt engineering
- Test with different temperature settings
- Consider fine-tuning or different model

## References

- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-14B)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Tailscale Quick Start](https://tailscale.com/kb/1017/install/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Changelog

- **2025-10-10**: Initial architecture plan created
