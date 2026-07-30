import asyncio
import json
import os
import uuid as uuid_module
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import rag_context
from auth import get_current_active_user
from database import SessionLocal, get_db
from models import RagChat, User
from rag_ask import (
    RAG_COLLECTIONS,
    build_ask_prompt,
    build_chat_messages,
    number_chunks,
    trim_history,
)
from shared import (
    RagHealthResponse,
    RagRetrieveRequest,
    RagRetrieveResponse,
    RagUpsertRequest,
    RagUpsertResponse,
    get_anthropic_client,
    get_gemini_client,
    get_openai_client,
    get_xai_client,
    limiter,
    resolve_openai_model,
)


router = APIRouter(prefix="/v1/rag", tags=["rag"])

# Collections owned by the firm-wide corpus sync jobs (doktrin_sync.py,
# jurisprudence_ingest.py) — never writable through the authenticated
# user-facing upsert endpoint below, only via the internal API-key path.
RESERVED_RAG_COLLECTIONS = {"jurisprudence", "doktrin"}


def validate_rag_collection(collection) -> None:
    if collection is not None and collection not in RAG_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unbekannte Collection '{collection}'. Erlaubt: {', '.join(RAG_COLLECTIONS)}",
        )


def _read_float_env(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Invalid {env_name} value '{raw}', using {default}")
        return default


def _get_rag_service_url() -> Optional[str]:
    base_url = os.getenv("RAG_SERVICE_URL", "").strip()
    return base_url.rstrip("/") if base_url else None


def _get_request_headers(request: Request, request_id: Optional[str] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    api_key = (
        os.getenv("RAG_API_KEY")
        or os.getenv("RAG_SERVICE_API_KEY")
    )
    if api_key:
        headers["X-API-Key"] = api_key

    if request_id:
        headers["X-Request-ID"] = request_id
    else:
        maybe_request_id = request.headers.get("X-Request-ID")
        if maybe_request_id:
            headers["X-Request-ID"] = maybe_request_id

    return headers


def _extract_rag_error_message(payload: Any, status_code: int) -> str:
    if isinstance(payload, dict):
        error_block = payload.get("error")
        if isinstance(error_block, dict):
            message = error_block.get("message")
            if message:
                return f"{message} (HTTP {status_code})"
        if isinstance(payload.get("detail"), str):
            return f"{payload['detail']} (HTTP {status_code})"
        if isinstance(payload.get("message"), str):
            return f"{payload['message']} (HTTP {status_code})"
    elif isinstance(payload, str):
        return f"{payload} (HTTP {status_code})"

    return f"RAG service returned HTTP {status_code}"


async def _post_to_rag(
    path: str,
    payload: Dict[str, Any],
    request: Request,
    timeout_seconds: float,
) -> Dict[str, Any]:
    base_url = _get_rag_service_url()
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail="RAG service is not configured. Set RAG_SERVICE_URL first.",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                f"{base_url}{path}",
                json=payload,
                headers=_get_request_headers(request),
            )
            if response.status_code < 200 or response.status_code >= 300:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = response.text
                raise HTTPException(
                    status_code=502,
                    detail=_extract_rag_error_message(
                        error_payload, response.status_code
                    ),
                )
            try:
                return response.json()
            except ValueError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"RAG service returned non-JSON response: {exc}",
                )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"RAG service timeout after {timeout_seconds}s for {path}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach RAG service: {exc}",
        )


def _build_upsert_payload(
    body: RagUpsertRequest,
    current_user: User,
) -> Dict[str, Any]:
    payload = body.model_dump()
    user_id = str(current_user.id)
    case_id = str(current_user.active_case_id) if current_user.active_case_id else None

    chunks = []
    for chunk in payload["chunks"]:
        metadata = dict(chunk.get("metadata", {}))
        metadata["owner_id"] = user_id
        if case_id:
            metadata["case_id"] = case_id
        chunk["metadata"] = metadata
        chunks.append(chunk)

    payload["chunks"] = chunks
    return payload


def _build_retrieve_payload(body: RagRetrieveRequest, current_user: User) -> Dict[str, Any]:
    payload = body.model_dump(exclude_none=True)
    # Preserve existing strict contract defaults while allowing upstream-safe empty filter handling.
    payload.setdefault("filters", {})
    # Scope retrieval to the calling user's own chunks plus the owner-less public corpus.
    payload["owner_id"] = str(current_user.id)
    return payload


@router.get("/health", response_model=RagHealthResponse)
@limiter.limit("30/hour")
async def health(
    request: Request,
    current_user: User = Depends(get_current_active_user),
):
    base_url = _get_rag_service_url()
    if not base_url:
        return JSONResponse(
            status_code=503,
            content=RagHealthResponse(
                status="unhealthy",
                qdrant_ok=False,
                desktop_embedder_ok=False,
                details={"reason": "RAG_SERVICE_URL is not configured"},
            ).model_dump(),
        )

    timeout_seconds = _read_float_env("RAG_HEALTH_TIMEOUT_SECONDS", 3.0)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(
                f"{base_url}/v1/rag/health",
                headers=_get_request_headers(request),
            )
            response.raise_for_status()
            data = response.json()
            return RagHealthResponse(
                status=data.get("status", "healthy"),
                qdrant_ok=bool(data.get("qdrant_ok", False)),
                desktop_embedder_ok=bool(data.get("desktop_embedder_ok", False)),
                details=data.get("details"),
            )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=503,
            content=RagHealthResponse(
                status="unhealthy",
                qdrant_ok=False,
                desktop_embedder_ok=False,
                details={"reason": f"RAG health check timed out after {timeout_seconds}s"},
            ).model_dump(),
        )
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content=RagHealthResponse(
                status="unhealthy",
                qdrant_ok=False,
                desktop_embedder_ok=False,
                details={"reason": f"{exc}"},
            ).model_dump(),
        )


@router.post("/chunks/upsert", response_model=RagUpsertResponse)
@limiter.limit("20/hour")
async def upsert_chunks(
    request: Request,
    body: RagUpsertRequest,
    current_user: User = Depends(get_current_active_user),
):
    if body.collection in RESERVED_RAG_COLLECTIONS:
        raise HTTPException(
            status_code=403,
            detail=(
                "Collection ist reserviert für den Kanzlei-Korpus und kann "
                "nicht per Upload beschrieben werden."
            ),
        )
    payload = _build_upsert_payload(body, current_user)
    response_data = await _post_to_rag(
        "/v1/rag/chunks/upsert",
        payload,
        request=request,
        timeout_seconds=_read_float_env("RAG_UPSERT_TIMEOUT_SECONDS", 120.0),
    )

    return RagUpsertResponse(
        upserted=int(response_data.get("upserted", 0)),
        collection=str(response_data.get("collection", body.collection)),
        warnings=response_data.get("warnings", []),
    )


@router.post("/retrieve", response_model=RagRetrieveResponse)
@limiter.limit("30/hour")
async def retrieve_chunks(
    request: Request,
    body: RagRetrieveRequest,
    current_user: User = Depends(get_current_active_user),
):
    validate_rag_collection(body.collection)
    payload = _build_retrieve_payload(body, current_user)
    response_data = await _post_to_rag(
        "/v1/rag/retrieve",
        payload,
        request=request,
        timeout_seconds=_read_float_env("RAG_RETRIEVE_TIMEOUT_SECONDS", 20.0),
    )

    retrieval = response_data.get("retrieval", {})
    chunks = response_data.get("chunks", [])

    return RagRetrieveResponse(
        query=response_data.get("query", body.query),
        retrieval={
            "fusion": retrieval.get("fusion", "rrf"),
            "dense_top_k": int(retrieval.get("dense_top_k", body.dense_top_k)),
            "sparse_top_k": int(retrieval.get("sparse_top_k", body.sparse_top_k)),
            "limit": int(retrieval.get("limit", body.limit)),
            "reranker_applied": bool(retrieval.get("reranker_applied", False)),
        },
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Wissensbasis-Chat: belegpflichtiger Ask-Stream + persistente Chats
# ---------------------------------------------------------------------------


class RagAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    collections: List[str] = Field(default_factory=list)
    model: Literal[
        "gemini-3.6-flash",
        "gemini-3.1-pro-preview",
        "gpt-5.6-terra",
        "claude-sonnet-5",
        "grok-4.5",
    ] = "gemini-3.6-flash"
    chat_id: Optional[str] = None


def collect_ask_sources(question, collections, owner_id, retrieve_fn):
    chosen = [c for c in (collections or []) if c] or list(RAG_COLLECTIONS)
    chunks_by_collection = {}
    for collection in chosen:
        chunks_by_collection[collection] = retrieve_fn(
            query=question, owner_id=owner_id, limit=8, use_reranker=True, collection=collection
        )
    return number_chunks(chunks_by_collection)


def build_stream_envelope(chat_id: str, sources) -> str:
    return json.dumps({"chat_id": chat_id, "sources": sources}, ensure_ascii=False) + "\n<<<ANSWER>>>\n"


def _stream_llm_answer(model: str, system_instruction: str, prompt: str):
    """Synchroner Text-Generator über das gewählte Modell (nur-Text-Prompt)."""
    if model.startswith("gpt"):
        client = get_openai_client()
        response = client.responses.create(
            model=resolve_openai_model(model),
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_instruction}]},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            reasoning={"effort": "high"},
            stream=True,
        )
        for event in response:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield delta
            elif event_type in {"response.failed", "response.incomplete", "error"}:
                raise RuntimeError(f"OpenAI-Stream abgebrochen: {event_type}")
        return
    if model.startswith("claude"):
        client = get_anthropic_client()
        with client.messages.stream(
            model=model,
            max_tokens=8000,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
        return
    if model.startswith("grok"):
        client = get_xai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
        return
    client = get_gemini_client()
    response = client.models.generate_content_stream(
        model=model,
        contents=[f"{system_instruction}\n\n{prompt}"],
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


@router.post("/ask/stream")
@limiter.limit("40/hour")
async def ask_stream(
    request: Request,
    body: RagAskRequest,
    current_user: User = Depends(get_current_active_user),
):
    for collection in body.collections:
        validate_rag_collection(collection)

    history = []
    chat_uuid = None
    if body.chat_id:
        try:
            chat_uuid = uuid_module.UUID(body.chat_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Ungültige chat_id")
        with SessionLocal() as db:
            chat = db.query(RagChat).filter(
                RagChat.id == chat_uuid, RagChat.owner_id == current_user.id
            ).first()
            if not chat:
                raise HTTPException(status_code=404, detail="Chat nicht gefunden")
            history = trim_history(chat.messages or [])

    sources = await asyncio.to_thread(
        collect_ask_sources,
        body.question,
        body.collections,
        str(current_user.id),
        rag_context.retrieve_chunks,
    )
    if not sources:
        raise HTTPException(
            status_code=404,
            detail=(
                "Keine Belege im Bestand gefunden. Ohne Quellen wird keine Antwort "
                "generiert. Falls die Wissensbasis (debian) schlief, wurde sie geweckt — "
                "bitte in einer Minute erneut versuchen."
            ),
        )

    system_instruction, prompt = build_ask_prompt(body.question, sources, history)
    chat_id_str = str(chat_uuid) if chat_uuid else str(uuid_module.uuid4())
    owner_id = current_user.id
    question = body.question
    model = body.model
    request_collections = [c for c in (body.collections or []) if c]
    is_new_chat = chat_uuid is None

    async def generate():
        answer_parts: List[str] = []
        yield build_stream_envelope(chat_id_str, sources)
        try:
            for delta in await asyncio.to_thread(
                lambda: list(_stream_llm_answer(model, system_instruction, prompt))
            ):
                answer_parts.append(delta)
                yield delta
        except Exception as exc:
            print(f"[RAG ASK] Streaming-Fehler: {exc}")
            yield f"\nFehler bei der Generierung: {exc}"
            return
        answer = "".join(answer_parts)
        try:
            now_iso = datetime.utcnow().isoformat() + "Z"
            new_messages = build_chat_messages(question, answer, model, sources, now_iso)
            with SessionLocal() as db:
                if is_new_chat:
                    chat = RagChat(
                        id=uuid_module.UUID(chat_id_str),
                        owner_id=owner_id,
                        title=question[:120],
                        collections=request_collections or list(RAG_COLLECTIONS),
                        messages=new_messages,
                    )
                    db.add(chat)
                else:
                    chat = db.query(RagChat).filter(
                        RagChat.id == uuid_module.UUID(chat_id_str), RagChat.owner_id == owner_id
                    ).first()
                    if chat is not None:
                        chat.messages = (chat.messages or []) + new_messages
                        chat.updated_at = datetime.utcnow()
                db.commit()
        except Exception as exc:
            print(f"[RAG ASK WARN] Chat-Persistenz fehlgeschlagen: {exc}")

    return StreamingResponse(generate(), media_type="text/plain")
