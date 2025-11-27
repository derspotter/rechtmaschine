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
    broadcast_documents_snapshot,
    get_gemini_client,
    limiter,
)
from database import SessionLocal, get_db
from models import Document
from .segmentation import (
    GeminiConfig as SegmentationGeminiConfig,
    segment_pdf_with_gemini,
)
from .ocr import check_pdf_needs_ocr

router = APIRouter()
ocr_check_semaphore = asyncio.Semaphore(5)


def sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", "-", "_", ".")).strip()


def process_akte_segmentation(
    stored_path: Path, source_filename: str, db: Session
) -> List[Tuple[uuid.UUID, str]]:
    """
    Process automatic segmentation for Akte files.
    Extracts Anhörung and Bescheid documents and adds them to the database.
    Returns list of (document_id, path) tuples for background OCR checking.
    """
    segments_to_check: List[Tuple[uuid.UUID, str]] = []
    try:
        segment_client = get_gemini_client()
        segment_dir = stored_path.parent / f"{stored_path.stem}_segments"
        _, extracted_pairs = segment_pdf_with_gemini(
            str(stored_path),
            segment_dir,
            client=segment_client,
            config=SegmentationGeminiConfig(),
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
                .filter(Document.filename == segment_filename)
                .first()
            )
            segment_explanation = (
                f"Segment ({section.document_type}) aus Akte {source_filename}, "
                f"Seiten {section.start_page}-{section.end_page}"
            )

            if existing_segment:
                existing_segment.category = category_enum.value
                existing_segment.confidence = section.confidence
                existing_segment.explanation = segment_explanation
                existing_segment.file_path = path
                segment_doc = existing_segment
            else:
                segment_doc = Document(
                    filename=segment_filename,
                    category=category_enum.value,
                    confidence=section.confidence,
                    explanation=segment_explanation,
                    file_path=path,
                )
                db.add(segment_doc)

            db.flush()
            segments_to_check.append((segment_doc.id, path))

    except Exception as segmentation_error:
        print(f"Segmentation failed for {stored_path}: {segmentation_error}")

    return segments_to_check


def run_akte_segmentation_sync(
    stored_path: Path,
    source_filename: str,
) -> List[Tuple[uuid.UUID, str]]:
    """Run Akte segmentation synchronously in a worker thread."""

    with SessionLocal() as background_db:
        segments_to_check = process_akte_segmentation(stored_path, source_filename, background_db)
        background_db.commit()
        broadcast_documents_snapshot(
            background_db,
            "segmentation",
            {"filename": source_filename},
        )
        return segments_to_check


def schedule_akte_segmentation(stored_path: Path, source_filename: str) -> None:
    """Queue Akte segmentation so the HTTP response can return immediately."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    async def runner() -> None:
        segments = await loop.run_in_executor(
            None,
            partial(run_akte_segmentation_sync, stored_path, source_filename),
        )

        for doc_id, path in segments:
            loop.create_task(check_and_update_ocr_status_bg(doc_id, path))

    loop.create_task(runner())





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
                    if document.processing_status == "pending":
                        document.processing_status = "pending"
                    db.commit()

                    broadcast_documents_snapshot(
                        db,
                        "ocr_check_complete",
                        {"document_id": str(document_id)},
                    )
            finally:
                db.close()

        except Exception as exc:
            print(f"[OCR BG ERROR] Failed to check document {document_id}: {exc}")


async def classify_document(file_content: bytes, filename: str) -> ClassificationResult:
    """Classify a document using Gemini 2.5 Flash."""
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

5. **Sonstiges** – Alle anderen Dokumente

Erzeuge ausschließlich JSON:
{
  "category": "<Anhörung|Bescheid|Rechtsprechung|Akte|Sonstiges>",
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
            model="gemini-2.5-flash-preview-09-2025",
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


@router.post("/classify", response_model=ClassificationResult)
@limiter.limit("20/hour")
async def classify(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ClassificationResult:
    """Classify uploaded PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(file.filename) or "upload.pdf"
        unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        stored_path = UPLOADS_DIR / unique_name
        with open(stored_path, "wb") as out:
            out.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {exc}")

    try:
        result = await classify_document(content, file.filename)

        existing_doc = db.query(Document).filter(Document.filename == result.filename).first()
        if existing_doc:
            existing_doc.category = result.category.value
            existing_doc.confidence = result.confidence
            existing_doc.explanation = result.explanation
            existing_doc.file_path = str(stored_path)
            existing_doc.gemini_file_uri = result.gemini_file_uri
            doc = existing_doc
        else:
            new_doc = Document(
                filename=result.filename,
                category=result.category.value,
                confidence=result.confidence,
                explanation=result.explanation,
                file_path=str(stored_path),
                gemini_file_uri=result.gemini_file_uri,
            )
            db.add(new_doc)
            doc = new_doc

        db.commit()
        with SessionLocal() as snapshot_db:
            broadcast_documents_snapshot(snapshot_db, "classify", {"filename": result.filename})

        if result.category == DocumentCategory.AKTE:
            schedule_akte_segmentation(stored_path, result.filename)
        else:
            asyncio.create_task(check_and_update_ocr_status_bg(doc.id, str(stored_path)))

        return result
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}")


@router.post("/upload-direct")
@limiter.limit("20/hour")
async def upload_direct(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form(...),
    db: Session = Depends(get_db),
):
    """Upload PDF directly to a specified category without classification."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        category_enum = DocumentCategory(category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(file.filename) or "upload.pdf"
        unique_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        stored_path = UPLOADS_DIR / unique_name
        with open(stored_path, "wb") as out:
            out.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {exc}")

    try:
        existing_doc = db.query(Document).filter(Document.filename == unique_name).first()
        if existing_doc:
            existing_doc.category = category_enum.value
            existing_doc.confidence = 1.0
            existing_doc.explanation = f"Direkt hochgeladen als {category_enum.value}"
            existing_doc.file_path = str(stored_path)
            doc = existing_doc
        else:
            new_doc = Document(
                filename=unique_name,
                category=category_enum.value,
                confidence=1.0,
                explanation=f"Direkt hochgeladen als {category_enum.value}",
                file_path=str(stored_path),
            )
            db.add(new_doc)
            doc = new_doc

        db.commit()
        with SessionLocal() as snapshot_db:
            broadcast_documents_snapshot(snapshot_db, "upload_direct", {"filename": unique_name})

        if category_enum == DocumentCategory.AKTE:
            schedule_akte_segmentation(stored_path, unique_name)
        else:
            asyncio.create_task(check_and_update_ocr_status_bg(doc.id, str(stored_path)))

        return {
            "success": True,
            "filename": unique_name,
            "category": category_enum.value,
            "message": f"Dokument erfolgreich hochgeladen als {category_enum.value}",
        }
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save document: {exc}")


__all__ = ["router", "sanitize_filename", "classify_document", "process_akte_segmentation"]
