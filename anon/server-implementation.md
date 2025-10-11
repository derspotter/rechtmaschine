# Server-Side Implementation Plan for Anonymization Service

## Overview

This document outlines the step-by-step implementation for integrating the anonymization service into the Rechtmaschine server. This assumes the home PC anonymization service is already running and accessible via Tailscale.

**Prerequisites:**
- Home PC running anonymization service (see anon-plan.md Phase 2)
- Tailscale installed on both machines
- Server can reach home PC at `http://100.64.1.2:8001`

---

## Phase 1: Network Configuration

### 1.1 Install and Configure Tailscale on Server

**Action Items:**
```bash
# Install Tailscale on rechtmaschine.de server
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Get server's Tailscale IP
tailscale ip -4
# Expected output: 100.64.1.3 (example)
```

**Verification:**
```bash
# Test connectivity to home PC
curl -X GET http://100.64.1.2:8001/health
# Expected: {"status":"healthy","model":"qwen3:14b-instruct-q5_K_M"}
```

### 1.2 Update Environment Configuration

**File:** `/var/opt/docker/rechtmaschine/app/.env`

**Add:**
```bash
# Anonymization Service (via Tailscale to home PC)
ANONYMIZATION_SERVICE_URL=http://100.64.1.2:8001
ANONYMIZATION_API_KEY=change-me-in-production-use-strong-random-key
```

**Note:** Coordinate the `ANONYMIZATION_API_KEY` with the home PC service configuration.

---

## Phase 2: Backend Implementation

### 2.1 Add Dependencies

**File:** `/var/opt/docker/rechtmaschine/app/requirements.txt`

**Check existing dependencies:**
- `httpx>=0.28.1` ✓ (already present)
- `pikepdf==9.4.2` ✓ (already present for PDF extraction)

No new dependencies needed!

### 2.2 Add Pydantic Models

**File:** `/var/opt/docker/rechtmaschine/app/app.py`

**Add after existing imports (around line 15):**

```python
# Anonymization models
class AnonymizationRequest(BaseModel):
    text: str
    document_type: str

class AnonymizationResult(BaseModel):
    anonymized_text: str
    plaintiff_names: list[str]
    confidence: float
    original_text: str
```

### 2.3 Add Environment Variable Loading

**File:** `/var/opt/docker/rechtmaschine/app/app.py`

**Add after existing environment variables (around line 25):**

```python
# Anonymization service configuration
ANONYMIZATION_SERVICE_URL = os.getenv("ANONYMIZATION_SERVICE_URL")
ANONYMIZATION_API_KEY = os.getenv("ANONYMIZATION_API_KEY")
```

### 2.4 Implement Anonymization Helper Function

**File:** `/var/opt/docker/rechtmaschine/app/app.py`

**Add before the API endpoints (around line 150, after other helper functions):**

```python
async def anonymize_document_text(text: str, document_type: str) -> Optional[AnonymizationResult]:
    """
    Call anonymization service on home PC via Tailscale.

    Args:
        text: Extracted text from PDF document
        document_type: Either "Anhörung" or "Bescheid"

    Returns:
        AnonymizationResult if successful, None if service unavailable
    """
    if not ANONYMIZATION_SERVICE_URL:
        print("[WARNING] ANONYMIZATION_SERVICE_URL not configured")
        return None

    try:
        headers = {}
        if ANONYMIZATION_API_KEY:
            headers["X-API-Key"] = ANONYMIZATION_API_KEY

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ANONYMIZATION_SERVICE_URL}/anonymize",
                json={
                    "text": text,
                    "document_type": document_type
                },
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            return AnonymizationResult(
                anonymized_text=data["anonymized_text"],
                plaintiff_names=data["plaintiff_names"],
                confidence=data["confidence"],
                original_text=text
            )

    except httpx.TimeoutException:
        print("[ERROR] Anonymization service timeout (>120s)")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] Anonymization service HTTP error: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[ERROR] Anonymization service error: {e}")
        return None
```

### 2.5 Add Anonymization API Endpoint

**File:** `/var/opt/docker/rechtmaschine/app/app.py`

**Add after document management endpoints (around line 400):**

```python
@app.post("/documents/{document_id}/anonymize")
@limiter.limit("5/hour")
async def anonymize_document_endpoint(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Anonymize a classified document (Anhörung or Bescheid only).

    Process:
    1. Verify document exists and is anonymizable
    2. Extract text from PDF
    3. Call anonymization service on home PC
    4. Store result in processed_documents table
    5. Return anonymized text

    Rate limit: 5 requests per hour (processing takes 30-60s each)
    """

    # Validate and fetch document
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = db.query(Document).filter(Document.id == doc_uuid).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Only anonymize Anhörung and Bescheid
    if document.category not in ["Anhörung", "Bescheid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Document type '{document.category}' does not support anonymization. Only 'Anhörung' and 'Bescheid' can be anonymized."
        )

    # Check if already processed
    existing_processed = db.query(ProcessedDocument).filter(
        ProcessedDocument.document_id == doc_uuid,
        ProcessedDocument.is_anonymized == True
    ).first()

    if existing_processed:
        # Return cached result
        return {
            "status": "success",
            "anonymized_text": existing_processed.anonymization_metadata.get("anonymized_text", ""),
            "plaintiff_names": existing_processed.anonymization_metadata.get("plaintiff_names", []),
            "confidence": existing_processed.anonymization_metadata.get("confidence", 0.0),
            "cached": True
        }

    # Extract text from PDF
    pdf_path = document.file_path
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server")

    try:
        extracted_text = extract_pdf_text(pdf_path, max_pages=50)
        if not extracted_text or len(extracted_text.strip()) < 100:
            raise HTTPException(
                status_code=422,
                detail="Could not extract sufficient text from PDF. Document may be scanned (OCR not yet implemented)."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF text extraction failed: {str(e)}")

    # Call anonymization service
    result = await anonymize_document_text(extracted_text, document.category)

    if not result:
        raise HTTPException(
            status_code=503,
            detail="Anonymization service unavailable. Please ensure home PC is online and connected via Tailscale."
        )

    # Store in processed_documents table
    processed_doc = ProcessedDocument(
        document_id=doc_uuid,
        extracted_text=result.original_text,
        is_anonymized=True,
        anonymization_metadata={
            "plaintiff_names": result.plaintiff_names,
            "confidence": result.confidence,
            "anonymized_at": datetime.utcnow().isoformat(),
            "anonymized_text": result.anonymized_text
        },
        processing_status="completed"
    )
    db.add(processed_doc)
    db.commit()

    return {
        "status": "success",
        "anonymized_text": result.anonymized_text,
        "plaintiff_names": result.plaintiff_names,
        "confidence": result.confidence,
        "cached": False
    }
```

---

## Phase 3: Frontend Implementation

### 3.1 Add Anonymization Button to Document Cards

**File:** `/var/opt/docker/rechtmaschine/app/app.py`

**Locate the document card HTML generation** (around line 800-900 in the `@app.get("/")` endpoint's HTML).

**Find the delete button code:**
```javascript
<button onclick="deleteDocument('${doc.filename}')"
        style="...">
    🗑️ Löschen
</button>
```

**Add anonymization button after the delete button:**
```javascript
${doc.category === 'Anhörung' || doc.category === 'Bescheid' ? `
    <button onclick="anonymizeDocument('${doc.id}')"
            style="margin-left: 8px; padding: 4px 8px; font-size: 11px;
                   background-color: #4A90E2; color: white; border: none;
                   border-radius: 3px; cursor: pointer;">
        🔒 Anonymisieren
    </button>
` : ''}
```

### 3.2 Add JavaScript Function for Anonymization

**File:** `/var/opt/docker/rechtmaschine/app/app.py`

**Add in the `<script>` section** (around line 1100, after `deleteDocument()` function):

```javascript
async function anonymizeDocument(docId) {
    if (!confirm('Dokument anonymisieren? Dies kann 30-60 Sekunden dauern.')) {
        return;
    }

    // Show loading state
    const button = event.target;
    const originalText = button.innerHTML;
    button.innerHTML = '⏳ Verarbeite...';
    button.disabled = true;

    try {
        const response = await fetch(`/documents/${docId}/anonymize`, {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Anonymisierung fehlgeschlagen');
        }

        const result = await response.json();

        // Show result in modal
        showAnonymizationResult(result);

    } catch (error) {
        alert('Fehler bei der Anonymisierung: ' + error.message);
    } finally {
        // Restore button
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

function showAnonymizationResult(result) {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.5); display: flex; align-items: center;
        justify-content: center; z-index: 1000;
    `;

    // Create modal content
    modal.innerHTML = `
        <div style="background: white; padding: 30px; border-radius: 8px;
                    max-width: 900px; max-height: 85vh; overflow-y: auto;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <h2 style="margin-top: 0; color: #333;">
                Anonymisiertes Dokument
                ${result.cached ? '<span style="font-size: 14px; color: #888;">(aus Cache)</span>' : ''}
            </h2>

            <div style="background: #f0f7ff; padding: 15px; border-radius: 6px;
                        margin-bottom: 20px; border-left: 4px solid #4A90E2;">
                <p style="margin: 5px 0;">
                    <strong>Gefundene Namen:</strong>
                    ${result.plaintiff_names.length > 0
                        ? result.plaintiff_names.join(', ')
                        : '<em>Keine Namen gefunden</em>'}
                </p>
                <p style="margin: 5px 0;">
                    <strong>Konfidenz:</strong>
                    ${(result.confidence * 100).toFixed(1)}%
                    ${result.confidence < 0.7
                        ? '<span style="color: #E74C3C;">⚠️ Niedrig - Manuelle Überprüfung empfohlen</span>'
                        : '<span style="color: #27AE60;">✓ Gut</span>'}
                </p>
            </div>

            <div style="background: #f9f9f9; padding: 20px; border-radius: 6px;
                        border: 1px solid #ddd; max-height: 500px; overflow-y: auto;">
                <h3 style="margin-top: 0; color: #555;">Anonymisierter Text:</h3>
                <div style="white-space: pre-wrap; font-family: 'Courier New', monospace;
                            font-size: 13px; line-height: 1.6; color: #333;">
                    ${escapeHtml(result.anonymized_text)}
                </div>
            </div>

            <div style="margin-top: 20px; text-align: right;">
                <button onclick="this.closest('div').parentElement.remove()"
                        style="padding: 10px 25px; background-color: #4A90E2;
                               color: white; border: none; border-radius: 5px;
                               cursor: pointer; font-size: 14px;">
                    Schließen
                </button>
            </div>
        </div>
    `;

    // Add to page
    document.body.appendChild(modal);

    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

---

## Phase 4: Deployment

### 4.1 Pre-Deployment Checklist

- [ ] Tailscale installed and running on server
- [ ] Home PC anonymization service running and healthy
- [ ] `.env` file updated with `ANONYMIZATION_SERVICE_URL` and `ANONYMIZATION_API_KEY`
- [ ] Code changes committed to git (optional but recommended)
- [ ] Backup current database (optional but recommended)

### 4.2 Deploy Changes

```bash
cd /var/opt/docker/rechtmaschine/app

# Restart the app container to load new .env variables
docker-compose restart app

# Watch logs for any startup errors
docker-compose logs -f app
```

### 4.3 Verification Tests

**Test 1: Health check from server to home PC**
```bash
curl -X GET http://100.64.1.2:8001/health
# Expected: {"status":"healthy","model":"qwen3:14b-instruct-q5_K_M"}
```

**Test 2: Upload a test Anhörung document**
- Upload via web interface
- Verify it's classified as "Anhörung" or "Bescheid"
- Check that anonymization button appears (🔒 Anonymisieren)

**Test 3: Test anonymization**
- Click anonymization button
- Wait 30-60 seconds
- Verify modal displays with:
  - Identified plaintiff names
  - Confidence score
  - Anonymized text

**Test 4: Check database**
```bash
docker exec rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db -c \
  "SELECT id, document_id, is_anonymized, processing_status,
          anonymization_metadata->>'confidence' as confidence
   FROM processed_documents
   ORDER BY created_at DESC
   LIMIT 5;"
```

---

## Phase 5: Monitoring & Maintenance

### 5.1 Monitor Anonymization Service Health

Add to server cron (optional):
```bash
# Check every 5 minutes
*/5 * * * * curl -f http://100.64.1.2:8001/health || echo "Anonymization service down" | mail -s "Alert: Anonymization Service" admin@rechtmaschine.de
```

### 5.2 Log Monitoring

**Check for anonymization errors:**
```bash
docker-compose logs app | grep -i "anonymization"
docker-compose logs app | grep -i "ERROR.*anonymiz"
```

### 5.3 Performance Metrics to Track

- Average anonymization time (should be 30-60s)
- Success rate (target: >95%)
- Cache hit rate (repeated anonymizations)
- Service availability (target: >99%)

---

## Troubleshooting

### Issue: "Anonymization service unavailable"

**Diagnosis:**
```bash
# From server
curl -v http://100.64.1.2:8001/health

# Check Tailscale status
tailscale status

# Check if home PC is online
ping 100.64.1.2
```

**Solutions:**
1. Verify home PC is powered on
2. Check Tailscale is running on both machines: `sudo systemctl status tailscaled`
3. Restart Tailscale: `sudo systemctl restart tailscaled`
4. Check home PC service logs

### Issue: "Could not extract sufficient text from PDF"

**Cause:** Document is scanned/image-based, not text-based PDF

**Solutions:**
1. Implement OCR (future enhancement)
2. Inform user that scanned documents are not yet supported
3. Manual processing required

### Issue: Low confidence score (<0.7)

**Recommendations:**
- Manual review required
- Check if document structure is unusual
- Consider fine-tuning anonymization prompts
- May indicate complex case with multiple parties

### Issue: API key mismatch

**Symptoms:** 401 Unauthorized from anonymization service

**Solution:**
1. Verify `ANONYMIZATION_API_KEY` matches between server `.env` and home PC service
2. Restart both services after updating keys

---

## Security Considerations

### Data Flow Security
✅ **Encrypted:** All data transmitted via Tailscale (WireGuard encryption)
✅ **Private:** No data leaves your infrastructure
✅ **Authenticated:** API key prevents unauthorized access
✅ **Isolated:** Anonymization service not exposed to internet

### Rate Limiting
- Anonymization endpoint: **5 requests/hour** (processing-intensive)
- Prevents abuse and resource exhaustion
- Adjust if needed based on usage patterns

### Data Retention
- Original text stored in `processed_documents.extracted_text`
- Anonymized text stored in `anonymization_metadata` JSONB field
- Consider retention policy for old processed documents

---

## Performance Optimization

### Current Implementation
- Caching: ✅ Results cached in database (check before re-processing)
- Rate limiting: ✅ 5/hour prevents service overload
- Timeout: ✅ 120s client timeout (sufficient for 30-60s processing)

### Future Enhancements
1. **Async processing with Redis queue**
   - User submits request → immediate response
   - Background job processes → notification when done
   - Allows multiple concurrent requests

2. **Progress indicators**
   - WebSocket/SSE for real-time status
   - Show "Processing... 30s elapsed"

3. **Batch processing**
   - Anonymize multiple documents in one request
   - Amortize network overhead

---

## Rollback Plan

If critical issues arise:

1. **Disable feature in frontend:**
   ```javascript
   // Comment out the anonymization button code
   // Line ~850 in app.py HTML section
   ```

2. **Remove endpoint (optional):**
   ```python
   # Comment out @app.post("/documents/{document_id}/anonymize")
   ```

3. **Restart app:**
   ```bash
   docker-compose restart app
   ```

4. **Remove environment variables:**
   ```bash
   # Edit /var/opt/docker/rechtmaschine/app/.env
   # Comment out ANONYMIZATION_SERVICE_URL
   ```

**Data is preserved:** Existing classifications and documents remain intact.

---

## Success Criteria

✅ **Functional Requirements:**
- [ ] User can click "🔒 Anonymisieren" on Anhörung/Bescheid documents
- [ ] System processes document in 30-60 seconds
- [ ] Modal displays plaintiff names, confidence, and anonymized text
- [ ] Results cached in database (no re-processing)
- [ ] Service gracefully handles home PC being offline

✅ **Non-Functional Requirements:**
- [ ] No sensitive data sent to external APIs
- [ ] Encrypted transmission via Tailscale
- [ ] Rate limiting prevents abuse
- [ ] Proper error messages for users
- [ ] Logging for debugging

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Network Setup | 30 minutes | Home PC service running |
| Backend Implementation | 2-3 hours | None |
| Frontend Implementation | 1-2 hours | Backend complete |
| Testing & Verification | 1 hour | All code complete |
| **Total** | **5-7 hours** | Sequential execution |

---

## Next Steps

After successful deployment:

1. **User Training:** Document how to use anonymization feature
2. **Monitor Usage:** Track success rates and processing times
3. **Gather Feedback:** Adjust confidence thresholds if needed
4. **Phase 5 Enhancements:** Consider async processing, OCR, export features (see anon-plan.md)

---

## References

- Main architecture plan: `anon/anon-plan.md`
- Project documentation: `CLAUDE.md`
- Database models: `app/models.py` (ProcessedDocument table)
- Current codebase: `app/app.py`
