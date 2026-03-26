import asyncio
import json
import logging
import mimetypes
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import SessionLocal, get_db
from .generation import (
    OPENAI_GPT5_MAX_OUTPUT_TOKENS,
    _build_openai_input_messages,
    _extract_openai_incomplete_reason,
    _upload_documents_to_openai,
)
from models import Document, User, ResearchSource, QueryJob
from shared import (
    QueryJobResponse,
    SelectedDocuments,
    collect_selected_document_identifiers,
    get_gemini_client,
    get_openai_client,
    load_document_text,
    limiter,
    ensure_document_on_gemini,
    get_document_for_upload,
    resolve_case_uuid_for_request,
    resolve_document_identifier,
)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    case_id: Optional[str] = None
    selected_documents: SelectedDocuments
    model: Literal[
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gpt-5.4",
    ] = "gemini-3-flash-preview"
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

class QueryResponse(BaseModel):
    answer: str
    used_documents: List[str]

from fastapi.responses import JSONResponse, StreamingResponse


def _query_job_to_response(job: QueryJob) -> QueryJobResponse:
    return QueryJobResponse(
        id=str(job.id),
        status=job.status,
        case_id=str(job.case_id) if job.case_id else None,
        error_message=job.error_message,
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        result_payload=job.result_payload or None,
    )


def _prepare_query_context(
    body: QueryRequest,
    db: Session,
    current_user: User,
) -> Dict[str, Any]:
    selection = body.selected_documents
    target_case_id = resolve_case_uuid_for_request(db, current_user, body.case_id)
    selected_identifiers = collect_selected_document_identifiers(selection)

    if not selected_identifiers and not selection.saved_sources:
        raise HTTPException(
            status_code=400,
            detail="Bitte wählen Sie mindestens ein Dokument oder eine Quelle aus.",
        )

    documents = []
    seen_document_ids = set()
    for identifier in selected_identifiers:
        doc = resolve_document_identifier(db, current_user, target_case_id, identifier)
        if doc and doc.id not in seen_document_ids:
            documents.append(doc)
            seen_document_ids.add(doc.id)

    sources = []
    if selection.saved_sources:
        import uuid

        valid_uuids = []
        for sid in selection.saved_sources:
            try:
                valid_uuids.append(uuid.UUID(sid))
            except ValueError:
                pass

        if valid_uuids:
            sources = (
                db.query(ResearchSource)
                .filter(
                    ResearchSource.id.in_(valid_uuids),
                    ResearchSource.owner_id == current_user.id,
                    ResearchSource.case_id == target_case_id,
                )
                .all()
            )

    document_entries = []
    for doc in documents:
        document_entries.append(
            {
                "filename": doc.filename,
                "file_path": doc.file_path,
                "extracted_text_path": doc.extracted_text_path,
                "anonymization_metadata": doc.anonymization_metadata,
                "is_anonymized": doc.is_anonymized,
            }
        )

    source_text_blocks = []
    for source in sources:
        content = source.description
        if content:
            source_text_blocks.append(f"QUELLE '{source.title}':\n{content}\n\n")

    history_messages = []
    for msg in (body.chat_history or [])[-12:]:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history_messages.append({"role": role, "content": content[:4000]})

    history_block = ""
    if history_messages:
        rendered_history = ["BISHERIGER VERLAUF (für Folgefragen):"]
        for msg in history_messages:
            speaker = "Nutzer" if msg["role"] == "user" else "Assistent"
            rendered_history.append(f"{speaker}: {msg['content']}")
        history_block = "\n".join(rendered_history)

    system_instruction = (
        "Du bist ein hilfreicher juristischer Assistent. "
        "Beantworte die Frage des Nutzers basierend auf den bereitgestellten Dokumenten und Quellen. "
        "Zitiere Dokumente oder Quellen, wo es sinnvoll ist. "
        "Wenn die Antwort nicht in den Unterlagen steht, sage das klar. "
        "Berücksichtige bei Folgefragen den bisherigen Gesprächsverlauf."
    )

    final_prompt = (
        f"{system_instruction}\n\n"
        f"KONTEXT (zusätzliche Textquellen):\n{''.join(source_text_blocks)}\n\n"
        f"{history_block}\n\n"
        f"AKTUELLE FRAGE: {body.query}"
    )

    used_documents = [doc.filename for doc in documents] + [source.title for source in sources if source.title]
    return {
        "target_case_id": target_case_id,
        "documents": documents,
        "sources": sources,
        "document_entries": document_entries,
        "source_text_blocks": source_text_blocks,
        "history_messages": history_messages,
        "history_block": history_block,
        "system_instruction": system_instruction,
        "final_prompt": final_prompt,
        "used_documents": used_documents,
    }


async def _execute_query_request(
    body: QueryRequest,
    db: Session,
    current_user: User,
) -> Dict[str, Any]:
    prepared = _prepare_query_context(body, db, current_user)
    documents = prepared["documents"]
    document_entries = prepared["document_entries"]
    source_text_blocks = prepared["source_text_blocks"]
    history_messages = prepared["history_messages"]
    history_block = prepared["history_block"]
    system_instruction = prepared["system_instruction"]
    final_prompt = prepared["final_prompt"]
    used_documents = prepared["used_documents"]

    if body.model.startswith("gpt"):
        client = get_openai_client()
        file_blocks = _upload_documents_to_openai(client, document_entries)
        for source_block in source_text_blocks:
            file_blocks.append({"type": "input_text", "text": source_block})
        if not file_blocks:
            raise HTTPException(status_code=400, detail="Kein Inhalt in den ausgewählten Dokumenten gefunden.")

        input_messages = _build_openai_input_messages(
            system_instruction,
            final_prompt,
            file_blocks,
            history_messages,
        )
        response = client.responses.create(
            model=body.model,
            input=input_messages,
            reasoning={"effort": "high"},
            text={"verbosity": "medium"},
            max_output_tokens=OPENAI_GPT5_MAX_OUTPUT_TOKENS,
        )
        status = getattr(response, "status", None)
        if status and status != "completed":
            reason = _extract_openai_incomplete_reason(response)
            if reason:
                raise RuntimeError(f"OpenAI response incomplete: {reason}")
            raise RuntimeError(f"OpenAI response incomplete: {status}")
        answer = getattr(response, "output_text", "") or ""
        return QueryResponse(answer=answer, used_documents=used_documents).model_dump()

    from google.genai import types

    context_parts = []
    for doc in documents:
        upload_entry = {
            "filename": doc.filename,
            "file_path": doc.file_path,
            "extracted_text_path": doc.extracted_text_path,
            "anonymization_metadata": doc.anonymization_metadata,
            "is_anonymized": doc.is_anonymized,
        }
        try:
            selected_path, mime_type, _ = get_document_for_upload(upload_entry)
        except Exception as exc:
            print(f"[WARN] Failed to resolve document for query: {doc.filename} ({exc})")
            continue

        guessed_mime_type, _ = mimetypes.guess_type(selected_path)
        if guessed_mime_type and guessed_mime_type.startswith("image/"):
            mime_type = guessed_mime_type

        if mime_type == "text/plain":
            try:
                with open(selected_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                if text_content:
                    context_parts.append(f"DOKUMENT '{doc.filename}':\n{text_content}\n\n")
            except Exception as exc:
                print(f"[WARN] Failed to read text for {doc.filename}: {exc}")
            continue

        if mime_type.startswith("image/"):
            try:
                with open(selected_path, "rb") as f:
                    image_bytes = f.read()
                context_parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                )
            except Exception as exc:
                print(f"[WARN] Failed to read image for query: {doc.filename} ({exc})")
            continue

        gemini_file = ensure_document_on_gemini(doc, db)
        if gemini_file:
            context_parts.append(gemini_file)

    for source_block in source_text_blocks:
        context_parts.append(source_block)

    if not context_parts:
        raise HTTPException(status_code=400, detail="Kein Inhalt in den ausgewählten Dokumenten gefunden.")

    text_context = ""
    request_contents = []
    for part in context_parts:
        if isinstance(part, str):
            text_context += part
        else:
            request_contents.append(part)

    gemini_prompt = (
        f"{system_instruction}\n\n"
        f"KONTEXT (Text):\n{text_context}\n\n"
        f"{history_block}\n\n"
        f"AKTUELLE FRAGE: {body.query}"
    )
    request_contents.append(gemini_prompt)

    client = get_gemini_client()
    response = client.models.generate_content(
        model=body.model,
        contents=request_contents,
    )
    answer = getattr(response, "text", "") or ""
    return QueryResponse(answer=answer, used_documents=used_documents).model_dump()


async def _run_query_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        import uuid

        job_uuid = uuid.UUID(job_id)
        job = db.query(QueryJob).filter(QueryJob.id == job_uuid).first()
        if not job:
            return

        job.status = "running"
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()

        user = db.query(User).filter(User.id == job.owner_id).first()
        if not user:
            raise RuntimeError("Owner user not found for query job")

        body = QueryRequest.model_validate(dict(job.request_payload or {}))
        result = await _execute_query_request(body, db, user)

        job.status = "completed"
        job.result_payload = result
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()
    except Exception as exc:
        print(f"[ERROR] Query job failed {job_id}: {exc}")
        db.rollback()
        try:
            import uuid

            failed_job = db.query(QueryJob).filter(QueryJob.id == uuid.UUID(job_id)).first()
            if failed_job:
                failed_job.status = "failed"
                failed_job.error_message = str(exc)
                failed_job.completed_at = datetime.utcnow()
                failed_job.updated_at = datetime.utcnow()
                db.commit()
        except Exception as nested_exc:
            print(f"[ERROR] Failed to persist query job failure state {job_id}: {nested_exc}")
            db.rollback()
    finally:
        db.close()

@router.post("/query-documents")
@limiter.limit("20/hour")
async def query_documents(
    request: Request,
    body: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Query selected documents using Gemini or GPT (Streaming).
    """
    prepared = _prepare_query_context(body, db, current_user)
    documents = prepared["documents"]
    document_entries = prepared["document_entries"]
    source_text_blocks = prepared["source_text_blocks"]
    history_messages = prepared["history_messages"]
    history_block = prepared["history_block"]
    system_instruction = prepared["system_instruction"]
    final_prompt = prepared["final_prompt"]

    async def generate_stream():
        try:
            context_parts = []
            if body.model.startswith("gpt"):
                client = get_openai_client()
                file_blocks = _upload_documents_to_openai(client, document_entries)

                for source_block in source_text_blocks:
                    file_blocks.append({"type": "input_text", "text": source_block})

                if not file_blocks:
                    yield "Fehler: Kein Inhalt in den ausgewählten Dokumenten gefunden."
                    return

                input_messages = _build_openai_input_messages(
                    system_instruction,
                    final_prompt,
                    file_blocks,
                    history_messages,
                )

                response = client.responses.create(
                    model=body.model,
                    input=input_messages,
                    reasoning={"effort": "high"},
                    text={"verbosity": "medium"},
                    max_output_tokens=OPENAI_GPT5_MAX_OUTPUT_TOKENS,
                    stream=True,
                )

                for event in response:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            yield delta
                    elif event_type in {"response.failed", "response.incomplete"}:
                        response_obj = getattr(event, "response", None)
                        status = getattr(response_obj, "status", None) if response_obj else None
                        reason = _extract_openai_incomplete_reason(response_obj) if response_obj else ""
                        if reason:
                            raise RuntimeError(f"OpenAI stream ended without completion ({status or 'unknown'}): {reason}")
                        raise RuntimeError(f"OpenAI stream ended without completion ({status or 'unknown'})")
                    elif event_type == "error":
                        message = getattr(event, "message", None) or str(event)
                        raise RuntimeError(message)
                return

            from google.genai import types

            for doc in documents:
                upload_entry = {
                    "filename": doc.filename,
                    "file_path": doc.file_path,
                    "extracted_text_path": doc.extracted_text_path,
                    "anonymization_metadata": doc.anonymization_metadata,
                    "is_anonymized": doc.is_anonymized,
                }

                try:
                    selected_path, mime_type, _ = get_document_for_upload(upload_entry)
                except Exception as exc:
                    print(f"[WARN] Failed to resolve document for query: {doc.filename} ({exc})")
                    continue

                guessed_mime_type, _ = mimetypes.guess_type(selected_path)
                if guessed_mime_type and guessed_mime_type.startswith("image/"):
                    mime_type = guessed_mime_type

                if mime_type == "text/plain":
                    try:
                        with open(selected_path, "r", encoding="utf-8") as f:
                            text_content = f.read()
                        if text_content:
                            context_parts.append(
                                f"DOKUMENT '{doc.filename}':\n{text_content}\n\n"
                            )
                        else:
                            print(f"[WARN] Empty text file for {doc.filename}")
                    except Exception as exc:
                        print(f"[WARN] Failed to read text for {doc.filename}: {exc}")
                    continue

                if mime_type.startswith("image/"):
                    try:
                        with open(selected_path, "rb") as f:
                            image_bytes = f.read()
                        request_image_part = types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        )
                        context_parts.append(request_image_part)
                    except Exception as exc:
                        print(f"[WARN] Failed to read image for query: {doc.filename} ({exc})")
                    continue

                gemini_file = ensure_document_on_gemini(doc, db)
                if gemini_file:
                    context_parts.append(gemini_file)
                else:
                    print(f"[WARN] Failed to utilize Gemini file for {doc.filename}")

            for source_block in source_text_blocks:
                context_parts.append(source_block)

            if not context_parts:
                yield "Fehler: Kein Inhalt in den ausgewählten Dokumenten gefunden."
                return

            text_context = ""
            request_contents = []
            for part in context_parts:
                if isinstance(part, str):
                    text_context += part
                else:
                    request_contents.append(part)

            gemini_prompt = (
                f"{system_instruction}\n\n"
                f"KONTEXT (Text):\n{text_context}\n\n"
                f"{history_block}\n\n"
                f"AKTUELLE FRAGE: {body.query}"
            )
            request_contents.append(gemini_prompt)

            client = get_gemini_client()
            response = client.models.generate_content_stream(
                model=body.model,
                contents=request_contents,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"Fehler bei der Generierung: {str(e)}"
        
        finally:
            # Cleanup uploads
            # for up_file in uploaded_files_to_clean:
            #     try:
            #         client.files.delete(name=up_file.name)
            #     except:
            #         pass
            pass

    return StreamingResponse(generate_stream(), media_type="text/plain")


@router.post("/query-documents/jobs", response_model=QueryJobResponse)
@limiter.limit("40/hour")
async def create_query_job(
    request: Request,
    body: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a background query job for CLI/API usage."""
    target_case_id = resolve_case_uuid_for_request(db, current_user, body.case_id)
    request_payload = body.model_dump()
    request_payload["case_id"] = str(target_case_id) if target_case_id else None

    job = QueryJob(
        owner_id=current_user.id,
        case_id=target_case_id,
        status="queued",
        request_payload=request_payload,
        result_payload={},
        updated_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    asyncio.create_task(_run_query_job(str(job.id)))
    return _query_job_to_response(job)


@router.get("/query-documents/jobs/{job_id}", response_model=QueryJobResponse)
@limiter.limit("120/hour")
async def get_query_job(
    request: Request,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    import uuid

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid query job id format")

    job = (
        db.query(QueryJob)
        .filter(
            QueryJob.id == job_uuid,
            QueryJob.owner_id == current_user.id,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="Query job not found")
    return _query_job_to_response(job)


@router.get("/query-documents/jobs/{job_id}/result")
@limiter.limit("120/hour")
async def get_query_job_result(
    request: Request,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    import uuid

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid query job id format")

    job = (
        db.query(QueryJob)
        .filter(
            QueryJob.id == job_uuid,
            QueryJob.owner_id == current_user.id,
        )
        .first()
    )
    if not job:
        raise HTTPException(status_code=404, detail="Query job not found")
    if job.status == "failed":
        raise HTTPException(status_code=409, detail=job.error_message or "Query job failed")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Query job not completed ({job.status})")
    return JSONResponse(content=job.result_payload or {})


@router.get("/query-documents/jobs/{job_id}/events")
async def stream_query_job_events(
    request: Request,
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Simple SSE stream for query job status changes."""
    import uuid

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid query job id format")

    async def event_generator():
        last_updated = None
        while True:
            if await request.is_disconnected():
                break
            session = SessionLocal()
            try:
                job = (
                    session.query(QueryJob)
                    .filter(
                        QueryJob.id == job_uuid,
                        QueryJob.owner_id == current_user.id,
                    )
                    .first()
                )
            finally:
                session.close()

            if not job:
                yield "event: error\ndata: {\"detail\":\"Query job not found\"}\n\n"
                break

            payload = job.to_dict()
            updated_at = payload.get("updated_at")
            if updated_at != last_updated:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_updated = updated_at

            if payload.get("status") in {"completed", "failed"}:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
