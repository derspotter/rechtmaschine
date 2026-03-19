import logging
import mimetypes
import os
from typing import Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import get_current_active_user
from database import get_db
from .generation import (
    OPENAI_GPT5_MAX_OUTPUT_TOKENS,
    _build_openai_input_messages,
    _upload_documents_to_openai,
)
from models import Document, User, ResearchSource
from shared import (
    SelectedDocuments,
    get_gemini_client,
    get_openai_client,
    load_document_text,
    limiter,
    ensure_document_on_gemini,
    get_document_for_upload,
)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
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

from fastapi.responses import StreamingResponse

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
    
    # 1. Collect documents
    selection = body.selected_documents
    doc_filenames = set()
    
    doc_filenames.update(selection.anhoerung)
    doc_filenames.update(selection.rechtsprechung)
    doc_filenames.update(selection.sonstiges)
    doc_filenames.update(selection.akte)
    
    if selection.bescheid.primary:
        doc_filenames.add(selection.bescheid.primary)
    doc_filenames.update(selection.bescheid.others)
    
    if selection.vorinstanz.primary:
        doc_filenames.add(selection.vorinstanz.primary)
    doc_filenames.update(selection.vorinstanz.others)

    if not doc_filenames and not selection.saved_sources:
        raise HTTPException(status_code=400, detail="Bitte wählen Sie mindestens ein Dokument oder eine Quelle aus.")

    # Fetch documents from DB
    documents = db.query(Document).filter(
        Document.filename.in_(doc_filenames),
        Document.owner_id == current_user.id,
        Document.case_id == current_user.active_case_id,
    ).all()
    
    # Fetch sources from DB (if any)
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
            sources = db.query(ResearchSource).filter(
                ResearchSource.id.in_(valid_uuids),
                ResearchSource.owner_id == current_user.id,
                ResearchSource.case_id == current_user.active_case_id,
            ).all()

    document_entries = []
    context_parts = []
    source_text_blocks = []

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

    async def generate_stream():
        try:
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

            final_prompt = (
                f"{system_instruction}\n\n"
                f"KONTEXT (Text):\n{text_context}\n\n"
                f"{history_block}\n\n"
                f"AKTUELLE FRAGE: {body.query}"
            )
            request_contents.append(final_prompt)

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
