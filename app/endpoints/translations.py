"""Document translation endpoints."""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import get_db
from document_translation import (
    compare_translations_with_gemini,
    source_hash,
    translate_text,
    translation_text_path,
)
from models import Document, DocumentTranslation, User
from shared import (
    ANONYMIZED_TEXT_DIR,
    TranslationComparisonRequest,
    TranslationRequest,
    limiter,
    load_document_text,
)


router = APIRouter()


def _get_active_document(
    document_id: str,
    db: Session,
    current_user: User,
) -> Document:
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    document = (
        db.query(Document)
        .filter(
            Document.id == doc_uuid,
            Document.owner_id == current_user.id,
            Document.case_id == current_user.active_case_id,
        )
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


def _get_translation(
    translation_id: str,
    document_id: str,
    db: Session,
    current_user: User,
) -> DocumentTranslation:
    document = _get_active_document(document_id, db, current_user)
    try:
        translation_uuid = uuid.UUID(translation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid translation ID format")

    translation = (
        db.query(DocumentTranslation)
        .filter(
            DocumentTranslation.id == translation_uuid,
            DocumentTranslation.document_id == document.id,
            DocumentTranslation.owner_id == current_user.id,
            DocumentTranslation.case_id == current_user.active_case_id,
        )
        .first()
    )
    if not translation:
        raise HTTPException(status_code=404, detail="Translation not found")
    return translation


def _safe_read(path: Optional[str], base_dir: Path) -> Optional[str]:
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        return None
    try:
        path_obj.resolve().relative_to(base_dir.resolve())
    except Exception:
        return None
    return path_obj.read_text(encoding="utf-8")


def _load_translation_source_text(
    document: Document,
    *,
    allow_unanonymized: bool,
) -> tuple[str, str]:
    metadata = document.anonymization_metadata or {}
    anonymized_text = _safe_read(metadata.get("anonymized_text_path"), ANONYMIZED_TEXT_DIR)
    if not anonymized_text:
        fallback = ANONYMIZED_TEXT_DIR / f"{document.id}.txt"
        anonymized_text = _safe_read(str(fallback), ANONYMIZED_TEXT_DIR)

    if anonymized_text:
        return anonymized_text, "anonymized"

    if allow_unanonymized:
        raw_text = load_document_text(document)
        if raw_text:
            return raw_text, "ocr"
        raise HTTPException(
            status_code=409,
            detail="No OCR text available. Run OCR before translating this non-PII document.",
        )

    raise HTTPException(
        status_code=409,
        detail=(
            "Document must be anonymized before translation, unless it is a non-PII "
            "official/public document and allow_unanonymized=true is set."
        ),
    )


def _read_translation_text(translation: DocumentTranslation) -> str:
    path = Path(translation.text_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Translation text file not found")
    return path.read_text(encoding="utf-8")


def _translation_payload(
    translation: DocumentTranslation,
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    payload = translation.to_dict()
    if include_text:
        payload["translated_text"] = _read_translation_text(translation)
    return payload


async def _get_or_create_translation(
    *,
    document: Document,
    request_body: TranslationRequest,
    db: Session,
    current_user: User,
    source_text: str,
    source_text_kind: str,
) -> DocumentTranslation:
    normalized_model = (request_body.model or "").strip()
    target_language = (request_body.target_language or "Deutsch").strip() or "Deutsch"
    source_language = (request_body.source_language or "").strip() or None
    text_hash = source_hash(source_text)

    if not request_body.force:
        existing = (
            db.query(DocumentTranslation)
            .filter(
                DocumentTranslation.document_id == document.id,
                DocumentTranslation.owner_id == current_user.id,
                DocumentTranslation.case_id == current_user.active_case_id,
                DocumentTranslation.model == normalized_model,
                DocumentTranslation.target_language == target_language,
                DocumentTranslation.source_language == source_language,
                DocumentTranslation.source_text_hash == text_hash,
            )
            .order_by(DocumentTranslation.created_at.desc())
            .first()
        )
        if existing and os.path.exists(existing.text_path):
            return existing

    try:
        translated_text, provider = await translate_text(
            source_text,
            model=normalized_model,
            target_language=target_language,
            source_language=source_language,
        )
    except Exception as exc:
        print(f"[TRANSLATION ERROR] {document.filename}: {exc}")
        raise HTTPException(status_code=503, detail=f"Translation failed: {exc}")

    if not translated_text.strip():
        raise HTTPException(status_code=502, detail="Translation model returned empty text")

    translation = DocumentTranslation(
        document_id=document.id,
        owner_id=current_user.id,
        case_id=current_user.active_case_id,
        model=normalized_model,
        provider=provider,
        source_language=source_language,
        target_language=target_language,
        source_text_hash=text_hash,
        text_path="",
        metadata_={
            "filename": document.filename,
            "source_text_length": len(source_text),
            "source_text_kind": source_text_kind,
            "text_length": len(translated_text),
            "created_via": "document_translation_endpoint",
        },
    )
    db.add(translation)
    db.flush()

    path = translation_text_path(translation.id)
    path.write_text(translated_text, encoding="utf-8")
    translation.text_path = str(path)
    translation.created_at = datetime.utcnow()
    db.commit()
    db.refresh(translation)
    return translation


@router.get("/documents/{document_id}/translations")
@limiter.limit("100/hour")
async def list_document_translations(
    request: Request,
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    document = _get_active_document(document_id, db, current_user)
    translations = (
        db.query(DocumentTranslation)
        .filter(
            DocumentTranslation.document_id == document.id,
            DocumentTranslation.owner_id == current_user.id,
            DocumentTranslation.case_id == current_user.active_case_id,
        )
        .order_by(DocumentTranslation.created_at.desc())
        .all()
    )
    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "translations": [translation.to_dict() for translation in translations],
    }


@router.post("/documents/{document_id}/translate")
@limiter.limit("30/hour")
async def translate_document(
    request: Request,
    document_id: str,
    body: TranslationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    document = _get_active_document(document_id, db, current_user)
    source_text, source_text_kind = _load_translation_source_text(
        document,
        allow_unanonymized=body.allow_unanonymized,
    )
    translation = await _get_or_create_translation(
        document=document,
        request_body=body,
        db=db,
        current_user=current_user,
        source_text=source_text,
        source_text_kind=source_text_kind,
    )
    return {
        "status": "success",
        "document_id": str(document.id),
        "filename": document.filename,
        "source_text_kind": source_text_kind,
        "translation": _translation_payload(translation, include_text=True),
    }


@router.post("/documents/{document_id}/translation-comparison")
@limiter.limit("15/hour")
async def compare_document_translations(
    request: Request,
    document_id: str,
    body: TranslationComparisonRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    document = _get_active_document(document_id, db, current_user)
    source_text, source_text_kind = _load_translation_source_text(
        document,
        allow_unanonymized=body.allow_unanonymized,
    )
    translations: list[DocumentTranslation] = []

    for model in body.models:
        translation = await _get_or_create_translation(
            document=document,
            request_body=TranslationRequest(
                model=model,
                target_language=body.target_language,
                source_language=body.source_language,
                allow_unanonymized=body.allow_unanonymized,
                force=body.force,
            ),
            db=db,
            current_user=current_user,
            source_text=source_text,
            source_text_kind=source_text_kind,
        )
        translations.append(translation)

    comparison = ""
    try:
        comparison = await compare_translations_with_gemini(
            source_text=source_text,
            translations=[
                {
                    "model": translation.model,
                    "text": _read_translation_text(translation),
                }
                for translation in translations
            ],
            target_language=body.target_language,
        )
    except Exception as exc:
        comparison = f"Vergleich konnte nicht automatisch erstellt werden: {exc}"

    return {
        "status": "success",
        "document_id": str(document.id),
        "filename": document.filename,
        "source_text_kind": source_text_kind,
        "translations": [
            _translation_payload(translation, include_text=False)
            for translation in translations
        ],
        "comparison": comparison,
    }


@router.get("/documents/{document_id}/translations/{translation_id}/download")
@limiter.limit("100/hour")
async def download_document_translation(
    request: Request,
    document_id: str,
    translation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    translation = _get_translation(translation_id, document_id, db, current_user)
    path = Path(translation.text_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Translation text file not found")
    filename = f"{translation.model.replace('/', '_')}_{translation.target_language}_{translation.id}.txt"
    return FileResponse(path, media_type="text/plain", filename=filename)
