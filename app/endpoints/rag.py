import os
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from auth import get_current_active_user
from models import User
from shared import (
    RagHealthResponse,
    RagRetrieveRequest,
    RagRetrieveResponse,
    RagUpsertRequest,
    RagUpsertResponse,
    limiter,
)


router = APIRouter(prefix="/v1/rag", tags=["rag"])


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


def _build_retrieve_payload(body: RagRetrieveRequest) -> Dict[str, Any]:
    payload = body.model_dump(exclude_none=True)
    # Preserve existing strict contract defaults while allowing upstream-safe empty filter handling.
    payload.setdefault("filters", {})
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
    payload = _build_retrieve_payload(body)
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
