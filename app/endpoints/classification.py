import asyncio
import os
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from shared import (
    UPLOADS_DIR,
    ClassificationResult,
    DocumentCategory,
    GeminiClassification,
    UploadDirectResponse,
    broadcast_documents_snapshot,
    get_gemini_client,
    limiter,
    resolve_case_uuid_for_request,
    should_auto_anonymize_category,
    should_auto_ocr_category,
    store_document_text,
)
from auth import get_current_active_user
from database import SessionLocal, get_db
from models import Document, User
from .segmentation import (
    segment_pdf_with_outline,
)
from .ocr import check_pdf_needs_ocr, perform_ocr_on_file
from .ocr import extract_pdf_text

router = APIRouter()
ocr_check_semaphore = asyncio.Semaphore(5)
# Actual OCR runs hit the GPU worker; keep concurrency low so bulk uploads
# queue instead of flooding the service manager.
ocr_auto_run_semaphore = asyncio.Semaphore(2)

ALLOWED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}
IMAGE_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Upload-Größenlimit (Default 100 MB), per Env überschreibbar. Chunked lesen,
# damit das Limit VOR dem Vollpuffern im RAM greift.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024


def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_", ".")).strip()


async def _read_upload_within_limit(file: UploadFile) -> bytes:
    """Liest den Upload in Chunks und bricht VOR dem Vollpuffern ab, wenn
    MAX_UPLOAD_BYTES überschritten wird."""
    chunks: List[bytes] = []
    total_bytes = 0
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Datei überschreitet das Upload-Limit von "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _get_extension(filename: Optional[str]) -> str:
    return Path(filename or "").suffix.lower()


def _is_image_extension(extension: str) -> bool:
    return extension in IMAGE_UPLOAD_EXTENSIONS


def _default_upload_name(extension: str) -> str:
    return f"upload{extension or '.pdf'}"


def process_akte_segmentation(
    stored_path: Path,
    source_filename: str,
    db: Session,
    owner_id: uuid.UUID,
    case_id: Optional[uuid.UUID],
) -> List[Tuple[uuid.UUID, str]]:
    """
    Process automatic segmentation for Akte files.
    Extracts Anhörung and Bescheid documents and adds them to the database.
    Returns list of (document_id, path, category) tuples for background
    OCR checking or auto-anonymization scheduling.
    """
    segments_to_check: List[Tuple[uuid.UUID, str, str]] = []
    try:
        segment_dir = stored_path.parent / f"{stored_path.stem}_segments"
        _, extracted_pairs = segment_pdf_with_outline(
            str(stored_path),
            segment_dir,
            verbose=False,
        )

        for section, path in extracted_pairs:
            try:
                category_enum = DocumentCategory(section.document_type)
            except ValueError:
                category_enum = DocumentCategory.SONSTIGES

            segment_filename = Path(path).name

            existing_segment = (
                db.query(Document)
                .filter(
                    Document.filename == segment_filename,
                    Document.owner_id == owner_id,
                    Document.case_id == case_id,
                )
                .first()
            )
            segment_explanation = (
                f"Segment ({section.document_type}) aus Akte {source_filename}, "
                f"Seiten {section.start_page}-{section.end_page}"
            )
            if section.outline_title:
                segment_explanation += f", TOC: {section.outline_title}"
            if section.hearing_subtype:
                segment_explanation += f", Subtyp: {section.hearing_subtype}"

            if existing_segment:
                existing_segment.category = category_enum.value
                existing_segment.confidence = section.confidence
                existing_segment.explanation = segment_explanation
                existing_segment.file_path = path
                existing_segment.outline_title = section.outline_title
                existing_segment.hearing_subtype = section.hearing_subtype
                segment_doc = existing_segment
            else:
                segment_doc = Document(
                    filename=segment_filename,
                    category=category_enum.value,
                    confidence=section.confidence,
                    explanation=segment_explanation,
                    file_path=path,
                    outline_title=section.outline_title,
                    hearing_subtype=section.hearing_subtype,
                    owner_id=owner_id,
                    case_id=case_id,
                )
                db.add(segment_doc)

            db.flush()
            segments_to_check.append((segment_doc.id, path, category_enum.value))

    except Exception as segmentation_error:
        print(f"Segmentation failed for {stored_path}: {segmentation_error}")

    return segments_to_check


def run_akte_segmentation_sync(
    stored_path: Path,
    source_filename: str,
    owner_id: uuid.UUID,
    case_id: Optional[uuid.UUID],
) -> List[Tuple[uuid.UUID, str, str]]:
    """Run Akte segmentation synchronously in a worker thread."""

    with SessionLocal() as background_db:
        segments_to_check = process_akte_segmentation(
            stored_path,
            source_filename,
            background_db,
            owner_id,
            case_id,
        )
        background_db.commit()
        broadcast_documents_snapshot(
            background_db,
            "segmentation",
            {"filename": source_filename},
            owner_id=owner_id,
        )
        return segments_to_check


def schedule_akte_segmentation(
    stored_path: Path,
    source_filename: str,
    owner_id: uuid.UUID,
    case_id: Optional[uuid.UUID],
) -> None:
    """Queue Akte segmentation so the HTTP response can return immediately."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    async def runner() -> None:
        segments = await loop.run_in_executor(
            None,
            partial(run_akte_segmentation_sync, stored_path, source_filename, owner_id, case_id),
        )

        for doc_id, path, category in segments:
            if should_auto_anonymize_category(category):
                # OCR is handled inside the anonymization pipeline.
                schedule_auto_anonymization(doc_id, owner_id, case_id)
            else:
                loop.create_task(check_and_update_ocr_status_bg(doc_id, path))

    loop.create_task(runner())


def schedule_auto_anonymization(
    document_id: uuid.UUID,
    owner_id: uuid.UUID,
    case_id: Optional[uuid.UUID],
) -> None:
    """Queue automatic OCR/anonymization for client-side personal files."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    async def runner() -> None:
        from .anonymization import auto_anonymize_document_bg

        await auto_anonymize_document_bg(document_id, owner_id, case_id)

    loop.create_task(runner())





async def auto_ocr_document_bg(document_id: uuid.UUID) -> None:
    """OCR a scanned document in an OCR-only source category and store its text.

    Mirrors the segment-child auto-OCR path: embeds a searchable text layer
    into the PDF (via perform_ocr_on_file) and stores the extracted text so
    generation grounding and the fact/citation checks can see the document.
    """
    async with ocr_auto_run_semaphore:
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return
            if document.is_anonymized or (document.ocr_applied and not document.needs_ocr):
                return
            if not document.file_path or not Path(document.file_path).exists():
                print(f"[AUTO OCR] File missing for {document_id}")
                return

            document.processing_status = "ocr_processing"
            db.commit()
            broadcast_documents_snapshot(
                db, "auto_ocr_started", {"document_id": str(document_id)}, owner_id=document.owner_id
            )

            try:
                text = await perform_ocr_on_file(document.file_path)
            except Exception as exc:
                text = None
                print(f"[AUTO OCR ERROR] {document.filename}: {exc}")

            normalized_text = (text or "").strip()
            metadata = dict(document.anonymization_metadata or {})
            if len(normalized_text) < 50:
                document.processing_status = "ocr_failed"
                metadata["auto_ocr_error"] = "OCR returned insufficient text"
                metadata["auto_ocr_failed_at"] = datetime.utcnow().isoformat()
                metadata["ocr_text_length"] = len(normalized_text)
                document.anonymization_metadata = metadata
                db.commit()
                broadcast_documents_snapshot(
                    db, "auto_ocr_failed", {"document_id": str(document_id)}, owner_id=document.owner_id
                )
                return

            store_document_text(document, normalized_text)
            document.ocr_applied = True
            document.needs_ocr = False
            document.processing_status = "ocr_ready"
            metadata["ocr_text_length"] = len(normalized_text)
            metadata["ocr_processed_at"] = datetime.utcnow().isoformat()
            metadata["ocr_source"] = "auto_ocr_ingest"
            document.anonymization_metadata = metadata
            db.commit()
            broadcast_documents_snapshot(
                db, "auto_ocr_completed", {"document_id": str(document_id)}, owner_id=document.owner_id
            )
        except Exception as exc:
            print(f"[AUTO OCR ERROR] Failed for document {document_id}: {exc}")
        finally:
            db.close()


async def check_and_update_ocr_status_bg(document_id: uuid.UUID, pdf_path: str):
    """
    Background task to check if PDF needs OCR and update database.
    Uses semaphore to limit concurrent checks.
    """
    async with ocr_check_semaphore:
        try:
            loop = asyncio.get_event_loop()
            needs_ocr = await loop.run_in_executor(
                None,
                check_pdf_needs_ocr,
                pdf_path,
            )

            db = SessionLocal()
            try:
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.needs_ocr = needs_ocr
                    if not needs_ocr:
                        try:
                            extracted_text = await loop.run_in_executor(
                                None,
                                partial(
                                    extract_pdf_text,
                                    pdf_path,
                                    None,
                                    False,
                                ),
                            )
                            normalized_text = (extracted_text or "").strip()
                            if normalized_text:
                                store_document_text(document, normalized_text)
                                if document.processing_status == "pending":
                                    document.processing_status = "text_ready"
                            else:
                                print(
                                    f"[WARN] Direct extraction yielded no text for {document.filename}"
                                )
                        except Exception as exc:
                            print(
                                f"[WARN] Failed direct extraction for {document.filename}: {exc}"
                            )
                    elif should_auto_ocr_category(document.category):
                        document.processing_status = "ocr_pending"
                        asyncio.create_task(auto_ocr_document_bg(document_id))
                    elif document.processing_status == "pending":
                        document.processing_status = "pending"
                    db.commit()

                    broadcast_documents_snapshot(
                        db,
                        "ocr_check_complete",
                        {"document_id": str(document_id)},
                        owner_id=document.owner_id,
                    )
            finally:
                db.close()

        except Exception as exc:
            print(f"[OCR BG ERROR] Failed to check document {document_id}: {exc}")


async def _classify_document_gemini(file_content: bytes, filename: str) -> ClassificationResult:
    """Classify a document using Gemini (cloud path, opt-in via CLASSIFICATION_BACKEND)."""
    from google.genai import types  # local import to avoid circular on module import

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name

    client = get_gemini_client()

    prompt = """Analysiere dieses deutsche Rechtsdokument und ordne es einer Kategorie zu:

1. **Anhörung** – BAMF-Anhörungsprotokoll / Niederschrift
   - Titel „Niederschrift …“, BAMF-Briefkopf, „Es erscheint …“, Frage-Antwort-Struktur, Unterschriften

2. **Bescheid** – BAMF-Entscheidungsbescheid
   - Überschrift „BESCHEID“, Gesch.-Z., nummerierte Entscheidungen, Rechtsbehelfsbelehrung

3. **Rechtsprechung** – Gerichtliche Entscheidung (Urteil/Beschluss)
   - Gericht als Absender, Aktenzeichen, Tenor, Tatbestand, Entscheidungsgründe

4. **Akte** – Vollständige BAMF-Beakte / Fallakte
   - Enthält mehrere Dokumentarten (z. B. Anhörungen, Bescheide, Vermerke) in einer PDF, oft mit Register- oder Blattnummern.

5. **Mandantenunterlagen** – Persönliche Unterlagen des Mandanten / private Fallnachweise
   - Identitätsdokumente, Urkunden, Familienunterlagen, Fotos, private Nachrichten, Schul-/Arbeits-/Medizinunterlagen, individuelle Belege

6. **Sonstiges** – Externe sonstige Quellen und Kontextmaterial
   - Behörden-/Länderberichte, offizielle Dokumente, Übersetzungen externer Quellen, Hintergrundmaterial, sonstige nicht-persönliche Quellen

Erzeuge ausschließlich JSON:
{
  "category": "<Anhörung|Bescheid|Rechtsprechung|Akte|Mandantenunterlagen|Sonstiges>",
  "confidence": <float 0.0-1.0>,
  "explanation": "kurze deutschsprachige Begründung"
}
"""

    with open(tmp_path, "rb") as pdf_file:
        uploaded = client.files.upload(
            file=pdf_file,
            config={
                "mime_type": "application/pdf",
                "display_name": filename,
            },
        )

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[prompt, uploaded],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=GeminiClassification,
            ),
        )

        parsed: GeminiClassification = response.parsed
        
        # We do NOT delete the file here anymore, as we want to reuse the URI
        # client.files.delete(name=uploaded.name)
        
        return ClassificationResult(
            category=parsed.category,
            confidence=parsed.confidence,
            explanation=parsed.explanation,
            filename=filename,
            gemini_file_uri=uploaded.uri
        )

    finally:
        # client.files.delete(name=uploaded.name)  <-- Commented out to persist
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _classify_document_text_gemini(extracted_text: str, filename: str) -> ClassificationResult:
    """Classify OCR text using Gemini (cloud path, opt-in via CLASSIFICATION_BACKEND)."""
    from google.genai import types  # local import to avoid circular on module import

    client = get_gemini_client()
    prompt = """Analysiere diesen OCR-Text eines deutschen Rechtsdokuments und ordne ihn einer Kategorie zu:

1. **Anhörung** – BAMF-Anhörungsprotokoll / Niederschrift
   - Titel „Niederschrift …“, BAMF-Briefkopf, „Es erscheint …“, Frage-Antwort-Struktur, Unterschriften

2. **Bescheid** – BAMF-Entscheidungsbescheid
   - Überschrift „BESCHEID“, Gesch.-Z., nummerierte Entscheidungen, Rechtsbehelfsbelehrung

3. **Rechtsprechung** – Gerichtliche Entscheidung (Urteil/Beschluss)
   - Gericht als Absender, Aktenzeichen, Tenor, Tatbestand, Entscheidungsgründe

4. **Akte** – Vollständige BAMF-Beakte / Fallakte
   - Enthält mehrere Dokumentarten (z. B. Anhörungen, Bescheide, Vermerke) in einer PDF, oft mit Register- oder Blattnummern.

5. **Mandantenunterlagen** – Persönliche Unterlagen des Mandanten / private Fallnachweise
   - Identitätsdokumente, Urkunden, Familienunterlagen, Fotos, private Nachrichten, Schul-/Arbeits-/Medizinunterlagen, individuelle Belege

6. **Sonstiges** – Externe sonstige Quellen und Kontextmaterial
   - Behörden-/Länderberichte, offizielle Dokumente, Übersetzungen externer Quellen, Hintergrundmaterial, sonstige nicht-persönliche Quellen

Erzeuge ausschließlich JSON:
{
  "category": "<Anhörung|Bescheid|Rechtsprechung|Akte|Mandantenunterlagen|Sonstiges>",
  "confidence": <float 0.0-1.0>,
  "explanation": "kurze deutschsprachige Begründung"
}
"""

    snippet = extracted_text[:10000]
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[f"{prompt}\n\nOCR-Text:\n{snippet}"],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=GeminiClassification,
        ),
    )

    parsed: GeminiClassification = response.parsed
    return ClassificationResult(
        category=parsed.category,
        confidence=parsed.confidence,
        explanation=parsed.explanation,
        filename=filename,
        gemini_file_uri=None,
    )


# ---------------------------------------------------------------------------
# Local Qwen classification (default backend): case material stays on our
# own hardware. Gemini remains available via CLASSIFICATION_BACKEND=gemini.
# ---------------------------------------------------------------------------

CLASSIFICATION_BACKEND = (
    os.getenv("CLASSIFICATION_BACKEND", "qwen").strip().lower() or "qwen"
)

_QWEN_CLASSIFY_PROMPT = """Analysiere dieses deutsche Rechtsdokument und ordne es genau einer Kategorie zu:

1. "Anhörung" – BAMF-Anhörungsprotokoll / Niederschrift (BAMF-Briefkopf, Frage-Antwort-Struktur)
2. "Bescheid" – BAMF-Entscheidungsbescheid (Überschrift BESCHEID, nummerierte Entscheidungen, Rechtsbehelfsbelehrung)
3. "Rechtsprechung" – Gerichtliche Entscheidung oder gerichtliche Korrespondenz (Urteil, Beschluss, Aktenzeichen, Tenor)
4. "Akte" – Vollständige BAMF-Beiakte/Fallakte mit mehreren Dokumentarten in einer PDF
5. "Mandantenunterlagen" – Persönliche Unterlagen des Mandanten (Identitätsdokumente, Urkunden, Zeugnisse, private Belege)
6. "Sonstiges" – Externe sonstige Quellen, Behördenkorrespondenz, Berichte, Hintergrundmaterial

Antworte ausschließlich mit einem JSON-Objekt:
{"category": "<Anhörung|Bescheid|Rechtsprechung|Akte|Mandantenunterlagen|Sonstiges>", "confidence": <0.0-1.0>, "explanation": "kurze deutsche Begründung"}
"""

_QWEN_CATEGORY_MAP = {
    "anhörung": DocumentCategory.ANHOERUNG,
    "anhoerung": DocumentCategory.ANHOERUNG,
    "bescheid": DocumentCategory.BESCHEID,
    "rechtsprechung": DocumentCategory.RECHTSPRECHUNG,
    "vorinstanz": DocumentCategory.VORINSTANZ,
    "akte": DocumentCategory.AKTE,
    "mandantenunterlagen": DocumentCategory.MANDANTENUNTERLAGEN,
    "sonstiges": DocumentCategory.SONSTIGES,
    "sonstige gespeicherte quellen": DocumentCategory.SONSTIGES,
}


async def _call_qwen_pdf_vision(
    service_url: str,
    file_content: bytes,
    filename: str,
    max_pages: int = 3,
) -> Dict[str, Any]:
    """Send a visually-rich PDF (Pass, Duldung, Fotos) through the worker's
    PDF vision route. Pages are rendered server-side with the worker's
    tested page/zoom/dimension bounds."""
    import httpx

    from citation_qwen import parse_qwen_json_response

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{service_url.rstrip('/')}/qwen-vision-segment-pdf",
            files={"file": (filename, file_content, "application/pdf")},
            data={
                "prompt": _QWEN_CLASSIFY_PROMPT,
                "max_pages": str(max(1, max_pages)),
                "zoom": "1.0",
                "num_predict": "400",
                "temperature": "0.0",
            },
        )
        response.raise_for_status()
        return parse_qwen_json_response(response.json())


def _image_to_png_b64(content: bytes) -> str:
    import base64

    import fitz

    with fitz.open(stream=content) as doc:
        pix = doc[0].get_pixmap()
        return base64.b64encode(pix.tobytes("png")).decode("ascii")


async def _classify_image_qwen(content: bytes, filename: str) -> ClassificationResult:
    """Classify an image file (Pass, Duldung, Foto) via Qwen vision."""
    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        raise HTTPException(
            status_code=503,
            detail="ANONYMIZATION_SERVICE_URL ist nicht konfiguriert (lokale Qwen-Klassifikation).",
        )
    await ensure_anonymization_service_ready()

    try:
        image_b64 = _image_to_png_b64(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bild konnte nicht gelesen werden: {exc}")

    parsed = await call_qwen_json(
        service_url,
        f"{_QWEN_CLASSIFY_PROMPT}\nDies ist ein Bild (z.B. Ausweisdokument, Urkunde oder Foto).",
        images=[image_b64],
        num_predict=400,
        temperature=0.0,
    )
    if not parsed:
        raise HTTPException(status_code=502, detail="Klassifikation (Qwen) lieferte kein gültiges JSON.")
    return _qwen_classification_result(parsed, filename)


def _qwen_classification_result(parsed: Dict[str, Any], filename: str) -> ClassificationResult:
    raw_category = str(parsed.get("category") or "").strip().lower()
    category = _QWEN_CATEGORY_MAP.get(raw_category)
    if category is None:
        raise HTTPException(
            status_code=502,
            detail=f"Klassifikation (Qwen) lieferte unbekannte Kategorie: {parsed.get('category')!r}",
        )
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return ClassificationResult(
        category=category,
        confidence=confidence,
        explanation=str(parsed.get("explanation") or "").strip(),
        filename=filename,
        gemini_file_uri=None,
    )


async def _classify_with_qwen(prompt_suffix: str, filename: str) -> ClassificationResult:
    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        raise HTTPException(
            status_code=503,
            detail="ANONYMIZATION_SERVICE_URL ist nicht konfiguriert (lokale Qwen-Klassifikation).",
        )
    await ensure_anonymization_service_ready()

    parsed = await call_qwen_json(
        service_url,
        f"{_QWEN_CLASSIFY_PROMPT}{prompt_suffix}",
        num_predict=400,
        temperature=0.0,
    )
    if not parsed:
        raise HTTPException(
            status_code=502,
            detail="Klassifikation (Qwen) lieferte kein gültiges JSON.",
        )
    return _qwen_classification_result(parsed, filename)


CLASSIFICATION_MIN_NATIVE_TEXT_CHARS = int(
    (os.getenv("CLASSIFICATION_MIN_NATIVE_TEXT_CHARS", "300") or "300").strip()
)


async def _classify_document_qwen(
    file_content: bytes,
    filename: str,
    stored_path: Optional[str] = None,
) -> ClassificationResult:
    """Classify a PDF with local Qwen — always from text.

    Text-readable PDFs are classified from their extracted text. Scans are
    first OCR'd through the worker (ocrmypdf + PaddleOCR, which also embeds
    the searchable text layer into the stored file), then classified from
    that text. Qwen never sees page images here.
    """
    native_text = ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        try:
            native_text = (extract_pdf_text(tmp_path) or "").strip()
        except Exception as exc:
            print(f"[CLASSIFY] Native text extraction failed for {filename}: {exc}")

        if len(native_text) >= CLASSIFICATION_MIN_NATIVE_TEXT_CHARS:
            return await _classify_document_text_qwen(native_text, filename)

        # Scan: OCR via the worker. Prefer the stored upload so the searchable
        # text layer lands in the file the Document row will point at.
        ocr_target = stored_path if stored_path and os.path.exists(stored_path) else tmp_path
        ocr_text = (await perform_ocr_on_file(ocr_target) or "").strip()
        if len(ocr_text) >= CLASSIFICATION_MIN_NATIVE_TEXT_CHARS:
            return await _classify_document_text_qwen(ocr_text, filename)

        # Little to no text even after OCR: a visual document (Pass, Duldung,
        # Fotos) — the one case for the Qwen vision route.
        from shared import ensure_anonymization_service_ready

        service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
        if not service_url:
            raise HTTPException(
                status_code=503,
                detail="ANONYMIZATION_SERVICE_URL ist nicht konfiguriert (lokale Qwen-Klassifikation).",
            )
        await ensure_anonymization_service_ready()
        parsed = await _call_qwen_pdf_vision(service_url, file_content, filename)
        if not parsed:
            raise HTTPException(
                status_code=502,
                detail="Klassifikation (Qwen) lieferte kein gültiges JSON.",
            )
        return _qwen_classification_result(parsed, filename)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _classify_document_text_qwen(extracted_text: str, filename: str) -> ClassificationResult:
    snippet = extracted_text[:10000]
    return await _classify_with_qwen(f"\n\nOCR-Text:\n{snippet}", filename)


async def classify_document(
    file_content: bytes,
    filename: str,
    stored_path: Optional[str] = None,
) -> ClassificationResult:
    """Classify an uploaded PDF. Default backend is the local Qwen worker."""
    if CLASSIFICATION_BACKEND == "gemini":
        return await _classify_document_gemini(file_content, filename)
    return await _classify_document_qwen(file_content, filename, stored_path=stored_path)


async def classify_document_text(extracted_text: str, filename: str) -> ClassificationResult:
    """Classify extracted/OCR text. Default backend is the local Qwen worker."""
    if CLASSIFICATION_BACKEND == "gemini":
        return await _classify_document_text_gemini(extracted_text, filename)
    return await _classify_document_text_qwen(extracted_text, filename)


@router.post("/classify", response_model=ClassificationResult)
@limiter.limit("100/hour")
async def classify(
    request: Request,
    file: UploadFile = File(...),
    case_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> ClassificationResult:
    """Classify uploaded PDF or image document."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    extension = _get_extension(file.filename)
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF or image files are supported",
        )

    is_image = _is_image_extension(extension)

    try:
        content = await _read_upload_within_limit(file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(file.filename) or _default_upload_name(extension)
        unique_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_name}"
        )
        stored_path = UPLOADS_DIR / unique_name
        with open(stored_path, "wb") as out:
            out.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {exc}")

    try:
        ocr_text = None
        if is_image:
            ocr_text = ((await perform_ocr_on_file(str(stored_path))) or "").strip()
            if CLASSIFICATION_BACKEND == "gemini":
                if not ocr_text:
                    raise HTTPException(
                        status_code=503,
                        detail="OCR service unavailable. Image uploads require OCR.",
                    )
                if len(ocr_text) < 100:
                    raise HTTPException(
                        status_code=422,
                        detail="Insufficient text extracted from image for classification.",
                    )
                result = await classify_document_text(ocr_text, file.filename)
            elif len(ocr_text) >= CLASSIFICATION_MIN_NATIVE_TEXT_CHARS:
                result = await classify_document_text(ocr_text, file.filename)
            else:
                # Visual document (Pass, Duldung, Foto): Qwen vision route.
                result = await _classify_image_qwen(content, file.filename)
            ocr_text = ocr_text or None
        else:
            result = await classify_document(content, file.filename, stored_path=str(stored_path))

        existing_doc = (
            db.query(Document)
            .filter(
                Document.filename == result.filename,
                Document.owner_id == current_user.id,
                Document.case_id == target_case_id,
            )
            .first()
        )
        
        if existing_doc:
            existing_doc.category = result.category.value
            existing_doc.confidence = result.confidence
            existing_doc.explanation = result.explanation
            existing_doc.file_path = str(stored_path)
            existing_doc.gemini_file_uri = result.gemini_file_uri
            existing_doc.case_id = target_case_id
            doc = existing_doc
        else:
            new_doc = Document(
                filename=result.filename,
                category=result.category.value,
                confidence=result.confidence,
                explanation=result.explanation,
                file_path=str(stored_path),
                gemini_file_uri=result.gemini_file_uri,
                owner_id=current_user.id,
                case_id=target_case_id,
            )
            db.add(new_doc)
            doc = new_doc

        if ocr_text:
            if not doc.id:
                db.flush()
            store_document_text(doc, ocr_text)
            doc.ocr_applied = True
            doc.needs_ocr = False
            doc.processing_status = "ocr_ready"
        elif should_auto_anonymize_category(result.category.value):
            doc.processing_status = "anon_pending"

        db.commit()
        with SessionLocal() as snapshot_db:
            broadcast_documents_snapshot(snapshot_db, "classify", {"filename": result.filename}, owner_id=current_user.id)

        if should_auto_anonymize_category(result.category.value):
            schedule_auto_anonymization(
                doc.id,
                current_user.id,
                target_case_id,
            )
        elif result.category == DocumentCategory.AKTE and not is_image:
            schedule_akte_segmentation(
                stored_path,
                result.filename,
                current_user.id,
                target_case_id,
            )
        elif not is_image:
            asyncio.create_task(check_and_update_ocr_status_bg(doc.id, str(stored_path)))
        elif should_auto_ocr_category(result.category.value):
            asyncio.create_task(auto_ocr_document_bg(doc.id))

        return result
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}")


@router.post("/upload-direct", response_model=UploadDirectResponse)
@limiter.limit("100/hour")
async def upload_direct(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form(...),
    case_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Upload PDF or image directly to a specified category without classification."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    extension = _get_extension(file.filename)
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF or image files are supported")

    is_image = _is_image_extension(extension)

    try:
        category_enum = DocumentCategory(category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    try:
        content = await _read_upload_within_limit(file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(file.filename) or _default_upload_name(extension)
        unique_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_name}"
        )
        stored_path = UPLOADS_DIR / unique_name
        with open(stored_path, "wb") as out:
            out.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {exc}")

    try:
        existing_doc = (
            db.query(Document)
            .filter(
                Document.filename == unique_name,
                Document.owner_id == current_user.id,
                Document.case_id == target_case_id,
            )
            .first()
        )
        
        if existing_doc:
            existing_doc.category = category_enum.value
            existing_doc.confidence = 1.0
            existing_doc.explanation = f"Direkt hochgeladen als {category_enum.value}"
            existing_doc.file_path = str(stored_path)
            existing_doc.case_id = target_case_id
            doc = existing_doc
        else:
            new_doc = Document(
                filename=unique_name,
                category=category_enum.value,
                confidence=1.0,
                explanation=f"Direkt hochgeladen als {category_enum.value}",
                file_path=str(stored_path),
                owner_id=current_user.id,
                case_id=target_case_id,
            )
            db.add(new_doc)
            doc = new_doc

        if is_image:
            doc.needs_ocr = True
            doc.ocr_applied = False

        if should_auto_anonymize_category(category_enum.value):
            doc.processing_status = "anon_pending"

        db.commit()
        with SessionLocal() as snapshot_db:
            broadcast_documents_snapshot(snapshot_db, "upload_direct", {"filename": unique_name}, owner_id=current_user.id)

        if should_auto_anonymize_category(category_enum.value):
            schedule_auto_anonymization(
                doc.id,
                current_user.id,
                target_case_id,
            )
        elif category_enum == DocumentCategory.AKTE and not is_image:
            schedule_akte_segmentation(
                stored_path,
                unique_name,
                current_user.id,
                target_case_id,
            )
        elif not is_image:
            asyncio.create_task(check_and_update_ocr_status_bg(doc.id, str(stored_path)))
        elif should_auto_ocr_category(category_enum.value):
            asyncio.create_task(auto_ocr_document_bg(doc.id))

        return UploadDirectResponse(
            success=True,
            document_id=str(doc.id),
            original_filename=file.filename or unique_name,
            filename=unique_name,
            category=category_enum.value,
            case_id=str(target_case_id) if target_case_id else None,
            processing_status=doc.processing_status,
            needs_ocr=bool(doc.needs_ocr),
            ocr_applied=bool(doc.ocr_applied),
            message=f"Dokument erfolgreich hochgeladen als {category_enum.value}",
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save document: {exc}")


__all__ = ["router", "sanitize_filename", "classify_document", "process_akte_segmentation"]
