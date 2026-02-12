import json
import os
import tempfile
import hashlib
from datetime import datetime
from typing import Optional
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from shared import (
    AnonymizationResult,
    DocumentCategory,
    broadcast_documents_snapshot,
    ensure_service_manager_ready,
    limiter,
    load_document_text,
    store_document_text,
    ANONYMIZED_TEXT_DIR,
)
from auth import get_current_active_user
from database import get_db
from models import Document, User
from .ocr import extract_pdf_text, perform_ocr_on_file, check_pdf_needs_ocr
from anon.anonymization_service import (
    filter_bamf_addresses,
    filter_non_person_group_labels,
    augment_names_from_role_markers,
    augment_names_from_person_fields,
    apply_regex_replacements,
)

router = APIRouter()

GEMMA_MODEL = os.getenv("OLLAMA_MODEL_GEMMA", os.getenv("OLLAMA_MODEL", "gemma3:12b"))
QWEN_MODEL = os.getenv("OLLAMA_MODEL_QWEN", "qwen3:8b")
ANONYMIZATION_ENGINE_DEFAULT = os.getenv(
    "ANONYMIZATION_ENGINE_DEFAULT", "qwen_flair"
).strip().lower()
SUPPORTED_ANONYMIZATION_ENGINES = {"gemma", "qwen_flair"}


EXTRACTION_PROMPT_PREFIX = """Extract PERSON names and PII from this German legal document.
Return JSON only with exactly:
{"names":[], "birth_dates":[], "birth_places":[], "streets":[], "postal_codes":[], "cities":[], "azr_numbers":[], "aufenthaltsgestattung_ids":[], "case_numbers":[]}

Rules:
- names = humans only (applicants, family, officials, signers)
- include exact surface forms from text, including surname-only and "SURNAME, Given" forms
- include names after role/signature markers: "geschlossen:", "Anhörender Entscheider", "Sachbearbeiter", "Unterzeichner", "gez.", "Unterschrift", "Im Auftrag"
- include names from person fields: "Name", "Vorname", "Nachname", "Familienname", "Geburtsname"
- do NOT include tribes/ethnicities/peoples/nationalities/languages/religions as names (e.g., Fulani, Paschtune, Hazara, Kurde, Afghanisch)
- deduplicate exact duplicates
- return valid JSON only

Document:
"""

EXTRACTION_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "items": {"type": "string"}},
        "birth_dates": {"type": "array", "items": {"type": "string"}},
        "birth_places": {"type": "array", "items": {"type": "string"}},
        "streets": {"type": "array", "items": {"type": "string"}},
        "postal_codes": {"type": "array", "items": {"type": "string"}},
        "cities": {"type": "array", "items": {"type": "string"}},
        "azr_numbers": {"type": "array", "items": {"type": "string"}},
        "aufenthaltsgestattung_ids": {"type": "array", "items": {"type": "string"}},
        "case_numbers": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "names", "birth_dates", "birth_places", "streets",
        "postal_codes", "cities", "azr_numbers",
        "aufenthaltsgestattung_ids", "case_numbers",
    ],
    "additionalProperties": False,
}


def resolve_anonymization_engine(requested_engine: Optional[str]) -> str:
    engine = (requested_engine or ANONYMIZATION_ENGINE_DEFAULT).strip().lower()
    if engine in SUPPORTED_ANONYMIZATION_ENGINES:
        return engine
    print(
        f"[WARN] Unknown anonymization engine '{engine}', "
        f"falling back to default '{ANONYMIZATION_ENGINE_DEFAULT}'"
    )
    if ANONYMIZATION_ENGINE_DEFAULT in SUPPORTED_ANONYMIZATION_ENGINES:
        return ANONYMIZATION_ENGINE_DEFAULT
    return "gemma"


def _dedupe_entity_lists(entities: dict) -> dict:
    deduped = {}
    for key, values in entities.items():
        if not isinstance(values, list):
            deduped[key] = values
            continue
        seen = set()
        out = []
        for value in values:
            if not isinstance(value, str):
                continue
            clean = value.strip()
            if not clean:
                continue
            token = clean.casefold()
            if token in seen:
                continue
            seen.add(token)
            out.append(clean)
        deduped[key] = out
    return deduped


def _merge_flair_names(entities: dict, flair_names: list[str]) -> dict:
    names = entities.get("names", [])
    if not isinstance(names, list):
        names = []

    seen = {n.strip().casefold() for n in names if isinstance(n, str) and n.strip()}
    added = []
    for raw in flair_names:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if len(candidate) < 2:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(candidate)
        added.append(candidate)

    if added:
        print(f"[INFO] Added Flair name hints: {added}")
    entities["names"] = names
    return entities


async def _fetch_flair_name_hints(
    service_url: str, text: str, document_type: str
) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            status_resp = await client.get(f"{service_url}/status")
            if status_resp.status_code == 200:
                status_payload = status_resp.json()
                anon_backend = (
                    status_payload.get("services", {}).get("anon_backend") or ""
                ).strip().lower()
                if anon_backend != "flair":
                    print(
                        f"[WARN] qwen_flair engine requested but service manager backend "
                        f"is '{anon_backend or 'unknown'}' (expected 'flair')."
                    )

            flair_resp = await client.post(
                f"{service_url}/anonymize",
                json={"text": text, "document_type": document_type},
            )
            flair_resp.raise_for_status()
            flair_data = flair_resp.json()
    except Exception as exc:
        print(f"[WARN] Flair name hint fetch failed: {exc}")
        return []

    plaintiff_names = flair_data.get("plaintiff_names") or []
    family_members = flair_data.get("family_members") or []
    if not isinstance(plaintiff_names, list):
        plaintiff_names = []
    if not isinstance(family_members, list):
        family_members = []
    return [*plaintiff_names, *family_members]


async def anonymize_document_text(
    text: str, document_type: str, engine: str
) -> Optional[AnonymizationResult]:
    """Extract entities via desktop LLM, then apply regex anonymization locally."""
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        print("[WARNING] ANONYMIZATION_SERVICE_URL not configured")
        return None

    await ensure_service_manager_ready()

    model = GEMMA_MODEL
    temperature = 0.6
    if engine == "qwen_flair":
        model = QWEN_MODEL
        temperature = 0.0

    prompt = EXTRACTION_PROMPT_PREFIX + text
    extraction_format = EXTRACTION_FORMAT_SCHEMA
    if (model or "").strip().lower().startswith("gemma3"):
        extraction_format = "json"
        print(
            f"[INFO] Gemma3 detected (model={model}); using format='json' "
            "instead of JSON schema for extraction stability"
        )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": extraction_format,
        "options": {
            "temperature": temperature,
            "num_predict": 4096,
            "num_ctx": 32768,
            "repeat_penalty": 1.0,
        },
    }

    try:
        print(
            f"[INFO] Entity extraction request "
            f"url={service_url}/extract-entities "
            f"model={model} payload_chars={len(text)} "
            f"document_type={document_type} engine={engine}"
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{service_url}/extract-entities",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        raw_response = data["response"]
        entities = json.loads(raw_response)
        entities = _dedupe_entity_lists(entities)

        entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        print(f"[INFO] Raw extraction: {entity_count} entities")

        entities = filter_bamf_addresses(entities)
        entities = filter_non_person_group_labels(entities, text)
        entities = augment_names_from_role_markers(entities, text)
        entities = augment_names_from_person_fields(entities, text)
        if engine == "qwen_flair":
            flair_names = await _fetch_flair_name_hints(service_url, text, document_type)
            entities = _merge_flair_names(entities, flair_names)
            entities = filter_non_person_group_labels(entities, text)
        entities = _dedupe_entity_lists(entities)

        filtered_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        print(f"[INFO] After BAMF filter: {filtered_count} entities")
        for key, values in entities.items():
            if values:
                print(f"  {key}: {values}")

        anonymized_text = apply_regex_replacements(text, entities)

        all_addresses = entities.get("streets", []) + entities.get("cities", [])

        print("[SUCCESS] Anonymization completed (local regex)")

        return AnonymizationResult(
            anonymized_text=anonymized_text,
            plaintiff_names=entities.get("names", []),
            birth_dates=entities.get("birth_dates", []),
            addresses=all_addresses,
            confidence=0.95,
            original_text=text,
            processed_characters=len(text),
        )

    except HTTPException:
        raise
    except httpx.TimeoutException:
        print("[ERROR] Entity extraction timeout (>300s)")
        raise HTTPException(
            status_code=504,
            detail="Entity extraction timeout (>300s). Please retry.",
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        message = None
        try:
            err = exc.response.json()
            if isinstance(err, dict):
                message = err.get("detail") or err.get("message")
            else:
                message = str(err)
        except Exception:
            message = exc.response.text or str(exc)

        detail_message = message or f"HTTP {status} from service manager"
        print(f"[ERROR] Entity extraction HTTP error: {status} – {detail_message}")
        raise HTTPException(status_code=status, detail=detail_message)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse LLM entity JSON: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse model response as JSON.",
        )
    except Exception as exc:
        print(f"[ERROR] Anonymization error: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Anonymization error: {exc}",
        )


@router.post("/documents/{document_id}/anonymize")
@limiter.limit("100/hour")
async def anonymize_document_endpoint(
    request: Request,
    document_id: str,
    force: bool = Query(False),
    engine: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize a classified document (Anhörung, Bescheid, or Sonstige gespeicherte Quellen)."""
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

    # All document types can be anonymized (names and addresses are extracted)
    resolved_engine = resolve_anonymization_engine(engine)

    if force and document.is_anonymized:
        print(f"[INFO] Force re-anonymization requested for {document.filename}")

    if document.is_anonymized and document.anonymization_metadata and not force:
        # Only path-based anonymized text is supported
        anonymized_text = ""
        path = document.anonymization_metadata.get("anonymized_text_path")
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    anonymized_text = f.read()
            except Exception as e:
                print(f"[ERROR] Failed to read anonymized text from file: {e}")
        else:
            print(f"[WARN] Missing anonymized_text_path for {document.filename}; reprocessing.")
            anonymized_text = ""

        if anonymized_text:
            processed_chars = document.anonymization_metadata.get(
                "processed_characters"
            )
            remaining_chars = document.anonymization_metadata.get(
                "remaining_characters"
            )
            if processed_chars is not None and remaining_chars is not None:
                input_characters = processed_chars + remaining_chars
            else:
                input_characters = len(anonymized_text)

            return {
                "status": "success",
                "anonymized_text": anonymized_text,
                "plaintiff_names": document.anonymization_metadata.get(
                    "plaintiff_names", []
                ),
                "addresses": document.anonymization_metadata.get("addresses", []),
                "confidence": document.anonymization_metadata.get("confidence", 0.0),
                "input_characters": input_characters,
                "processed_characters": processed_chars,
                "remaining_characters": remaining_chars,
                "cached": True,
                "engine": document.anonymization_metadata.get("engine", resolved_engine),
            }

    pdf_path = document.file_path
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    extracted_text = None
    ocr_used = False

    cached_text = load_document_text(document)
    if document.ocr_applied and cached_text:
        extracted_text = cached_text
        ocr_used = True
        print(
            f"[INFO] Using cached OCR text for {document.filename}: {len(extracted_text)} characters"
        )

    if not extracted_text:
        # Check if we need OCR using shared logic (consistent with classification)
        # We respect the DB flag if it's True, otherwise we verify with the shared check.
        should_use_ocr = document.needs_ocr or check_pdf_needs_ocr(pdf_path)

        if should_use_ocr:
            print(
                f"[INFO] Document needs OCR (flag={document.needs_ocr}). Skipping direct extraction."
            )
            extracted_text = None
        else:
            try:
                extracted_text = extract_pdf_text(
                    pdf_path, max_pages=50, include_page_headers=False
                )
                # Final sanity check: even if check passed, maybe extraction failed or yielded garbage
                if extracted_text and len(extracted_text.strip()) >= 500:
                    print(
                        f"[INFO] Direct text extraction successful: {len(extracted_text)} characters"
                    )
                else:
                    print(
                        f"[INFO] Direct extraction insufficient ({len(extracted_text) if extracted_text else 0} chars), trying OCR..."
                    )
                    extracted_text = None
            except Exception as exc:
                print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
                extracted_text = None

    if not extracted_text:
        extracted_text = await perform_ocr_on_file(pdf_path)
        if extracted_text:
            ocr_used = True
            print(
                f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running.",
            )

    if not extracted_text or len(extracted_text.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail="Insufficient text extracted from PDF. The document may be empty or corrupted.",
        )

    document_type = document.category
    if document_type == "Sonstiges":
        document_type = DocumentCategory.SONSTIGES.value

    text_for_anonymization = extracted_text
    try:
        text_hash = hashlib.sha256(text_for_anonymization.encode("utf-8")).hexdigest()
        text_len = len(text_for_anonymization)
        line_count = text_for_anonymization.count("\n") + 1 if text_for_anonymization else 0
        word_count = len(text_for_anonymization.split())
        null_count = text_for_anonymization.count("\x00")
        non_ascii = sum(1 for ch in text_for_anonymization if ord(ch) > 127)
        print(
            "[INFO] Anonymization payload stats "
            f"(doc={document.filename}, type={document_type}, force={force}): "
            f"chars={text_len}, words={word_count}, lines={line_count}, "
            f"non_ascii={non_ascii}, nulls={null_count}, sha256={text_hash}"
        )
    except Exception as exc:
        print(f"[WARN] Failed to compute anonymization payload stats: {exc}")
    print(
        f"[INFO] Sending {len(text_for_anonymization)} characters to anonymization service"
    )

    result = await anonymize_document_text(
        text_for_anonymization, document_type, resolved_engine
    )
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Anonymization service unavailable. Please ensure it is running.",
        )

    anonymized_full_text = result.anonymized_text
    processed_chars = result.processed_characters
    remaining_chars = 0

    # Save anonymized text to file
    anonymized_filename = f"{document.id}.txt"
    anonymized_path = ANONYMIZED_TEXT_DIR / anonymized_filename
    try:
        with open(anonymized_path, "w", encoding="utf-8") as f:
            f.write(anonymized_full_text)
    except Exception as e:
        print(f"[ERROR] Failed to write anonymized text to file: {e}")
        # Fallback? Or raise? For now, we proceed, but metadata might be incomplete if we rely on path.
        # Actually, if write fails, we should probably fail the request or fallback to DB storage.
        # Let's fallback to DB storage implicitly if path is missing, but here we want to enforce file.
        raise HTTPException(
            status_code=500, detail=f"Failed to save anonymized text: {e}"
        )

    store_document_text(document, extracted_text)
    document.is_anonymized = True
    document.ocr_applied = ocr_used
    document.anonymization_metadata = {
        "plaintiff_names": result.plaintiff_names,
        "birth_dates": result.birth_dates,
        "addresses": result.addresses,
        "confidence": result.confidence,
        "anonymized_at": datetime.utcnow().isoformat(),
        "anonymized_text_path": str(anonymized_path),
        "anonymized_excerpt": result.anonymized_text,
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "input_characters": len(extracted_text),
        "ocr_used": ocr_used,
        "engine": resolved_engine,
    }
    document.processing_status = "completed"
    db.commit()

    broadcast_documents_snapshot(db, "anonymize", {"document_id": document_id})

    return {
        "status": "success",
        "anonymized_text": anonymized_full_text,
        "plaintiff_names": result.plaintiff_names,
        "birth_dates": result.birth_dates,
        "addresses": result.addresses,
        "confidence": result.confidence,
        "input_characters": len(extracted_text),
        "processed_characters": processed_chars,
        "remaining_characters": remaining_chars,
        "ocr_used": ocr_used,
        "cached": False,
        "engine": resolved_engine,
    }


@router.post("/anonymize-file")
@limiter.limit("100/hour")
async def anonymize_uploaded_file(
    request: Request,
    document_type: str = Form(...),
    file: UploadFile = File(...),
    engine: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    """Anonymize an uploaded PDF without storing it in the database."""
    sanitized_type = document_type.strip() or "Sonstiges"
    resolved_engine = resolve_anonymization_engine(engine)

    filename = (file.filename or "upload.pdf").strip()
    _, ext = os.path.splitext(filename.lower())
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmpf.write(chunk)
        tmp_path = tmpf.name

    try:
        extracted_text = None
        ocr_used = False

        # Check if we need OCR using shared logic
        should_use_ocr = check_pdf_needs_ocr(tmp_path)

        if not should_use_ocr:
            try:
                extracted_text = extract_pdf_text(
                    tmp_path, max_pages=50, include_page_headers=False
                )
                # Final sanity check: even if check passed, maybe extraction failed or yielded garbage
                if extracted_text and len(extracted_text.strip()) >= 500:
                    print(
                        f"[INFO] Direct text extraction successful: {len(extracted_text)} characters"
                    )
                else:
                    print(
                        f"[INFO] Direct extraction insufficient ({len(extracted_text) if extracted_text else 0} chars), trying OCR..."
                    )
                    extracted_text = None
            except Exception as exc:
                print(f"[INFO] Direct extraction failed: {exc}, trying OCR...")
                extracted_text = None

        if not extracted_text:
            extracted_text = await perform_ocr_on_file(tmp_path)
            if extracted_text:
                ocr_used = True
                print(
                    f"[SUCCESS] OCR extraction successful: {len(extracted_text)} characters"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Could not extract text from PDF. OCR service unavailable. Please ensure home PC OCR service is running.",
                )

        if not extracted_text or len(extracted_text.strip()) < 100:
            raise HTTPException(
                status_code=422,
                detail="Insufficient text extracted from PDF. The document may be empty or corrupted.",
            )

        text_for_anonymization = extracted_text
        print(
            f"[INFO] Sending uploaded PDF ({len(text_for_anonymization)} chars) to anonymization service"
        )

        result = await anonymize_document_text(
            text_for_anonymization, sanitized_type, resolved_engine
        )
        if result is None:
            raise HTTPException(
                status_code=503,
                detail="Anonymization service unavailable. Please ensure it is running.",
            )

        anonymized_full_text = result.anonymized_text
        processed_chars = result.processed_characters
        remaining_chars = 0

        return {
            "status": "success",
            "filename": filename,
            "anonymized_text": anonymized_full_text,
            "plaintiff_names": result.plaintiff_names,
            "birth_dates": result.birth_dates,
            "addresses": result.addresses,
            "confidence": result.confidence,
            "input_characters": len(extracted_text),
            "processed_characters": processed_chars,
            "remaining_characters": remaining_chars,
            "ocr_used": ocr_used,
            "engine": resolved_engine,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


__all__ = ["router"]
