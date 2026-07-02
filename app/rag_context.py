"""Cross-case RAG retrieval for generation/query prompts.

Pulls anonymized argumentation chunks from the firm's own past filings (the
Debian RAG store) and formats them as an advisory prompt block — a sibling to
the case-memory block. Case memory grounds *this* case; RAG supplies reusable
argument patterns from *other* cases.

Gated behind RAG_RETRIEVAL_ENABLED so it can be deployed before the store is
filled and switched on afterwards. Any failure degrades to an empty block:
retrieval must never break generation/query.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Optional

import httpx

_WS = re.compile(r"\s+")


def _dedup_key(text: str) -> str:
    """Normalized hash for dropping duplicate passages that recur across
    documents (e.g. the same BAMF Bescheid quoted in several Schriftsätze)."""
    return hashlib.sha256(_WS.sub(" ", (text or "").lower()).strip().encode("utf-8")).hexdigest()

_CASE_REF = re.compile(r"^\s*(\d{3})\s*/\s*(\d{2})\b")


def retrieval_enabled() -> bool:
    return os.getenv("RAG_RETRIEVAL_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _service_url() -> str:
    return os.getenv("RAG_SERVICE_URL", "").strip().rstrip("/")


def _api_key() -> Optional[str]:
    return os.getenv("RAG_API_KEY") or os.getenv("RAG_SERVICE_API_KEY")


def _collection() -> str:
    # Must match the ingestion runner's --collection (default "kanzlei");
    # debian's retrieve defaults to "rag_chunks", so we name it explicitly.
    return os.getenv("RAG_COLLECTION", "kanzlei")


def case_hash_from_name(name: Optional[str]) -> Optional[str]:
    """Derive the chunk-metadata case_hash from a Rechtmaschine case name.

    Case names lead with the file number ("089/26 Balulov"); ingestion hashed
    the same "NNN/YY" reference into chunk metadata, so this lets us exclude the
    current case's own filings from its retrieval."""
    if not name:
        return None
    match = _CASE_REF.match(name)
    if not match:
        return None
    case_ref = f"{match.group(1)}/{match.group(2)}"
    return hashlib.sha256(case_ref.encode("utf-8")).hexdigest()[:12]


def retrieve_chunks(
    query: str,
    exclude_case_hash: Optional[str] = None,
    limit: int = 6,
    use_reranker: bool = True,
    timeout: float = 20.0,
    collection: Optional[str] = None,
) -> list[dict[str, Any]]:
    base = _service_url()
    if not base or not query.strip():
        return []
    # Over-fetch so client-side self-exclusion still leaves a full result set
    # (debian caps limit at 12).
    fetch = min(limit + 4, 12)
    payload = {
        "query": query.strip()[:2000],
        "collection": collection or _collection(),
        "limit": fetch,
        "use_reranker": use_reranker,
    }
    headers = {"X-API-Key": _api_key()} if _api_key() else {}
    try:
        response = httpx.post(
            f"{base}/v1/rag/retrieve", json=payload, headers=headers, timeout=timeout
        )
        response.raise_for_status()
        chunks = response.json().get("chunks", [])
    except Exception as exc:
        print(f"[RAG] retrieve failed: {exc}")
        return []

    if exclude_case_hash:
        chunks = [
            c for c in chunks
            if (c.get("metadata") or {}).get("case_hash") != exclude_case_hash
        ]

    # Drop duplicate passages (same normalized text from different docs) so a
    # recurring quote doesn't waste result slots.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in chunks:
        key = _dedup_key(c.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped[:limit]


def build_rag_block(
    query: str,
    case_name: Optional[str] = None,
    limit: Optional[int] = None,
    collect: Optional[dict] = None,
) -> str:
    """Return a labelled German precedent block (with trailing newlines), or ''.

    Empty unless retrieval is enabled and chunks come back — so callers can
    always concatenate it unconditionally. When `collect` is a dict, compact
    chunk provenance is recorded under `collect["rag_chunks"]` (same pattern as
    the case-memory/wiki grounding), so drafts can show what was in the prompt."""
    if not retrieval_enabled():
        return ""
    if limit is None:
        try:
            limit = int(os.getenv("RAG_RETRIEVAL_LIMIT", "6"))
        except ValueError:
            limit = 6
    chunks = retrieve_chunks(query, exclude_case_hash=case_hash_from_name(case_name), limit=limit)
    if not chunks:
        return ""
    if collect is not None:
        collect["rag_chunks"] = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "score": chunk.get("score"),
                "context_header": (chunk.get("context_header") or "").strip() or None,
                "chars": len((chunk.get("text") or "").strip()),
            }
            for chunk in chunks
        ]

    lines = [
        "EINSCHLÄGIGE ANONYMISIERTE KANZLEI-PRÄZEDENZ "
        "(Auszüge aus ANDEREN Mandaten der Kanzlei, nur als Argumentations- und "
        "Formulierungsmuster). Übernimm hieraus KEINE fallspezifischen Fakten; "
        "Platzhalter wie [PERSON]/[ORT] sind anonymisiert und betreffen NICHT die "
        "aktuelle Mandantschaft:",
    ]
    for i, chunk in enumerate(chunks, 1):
        header = (chunk.get("context_header") or "").strip()
        text = (chunk.get("text") or "").strip()
        lines.append(f"[{i}] {header}\n{text}" if header else f"[{i}] {text}")
    return "\n\n".join(lines) + "\n\n"
