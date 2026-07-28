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
import shlex
import subprocess
import time
from typing import Any, Optional

import httpx

_WS = re.compile(r"\s+")

# debian faehrt seit dem 28.07.2026 bei Leerlauf herunter (idle-poweroff.timer)
# statt zu suspendieren. Vorher lief sie praktisch durch, ein Abruf traf also
# immer eine wache Maschine.
_WAKE_COMMAND_DEFAULT = "ssh -o BatchMode=yes osmc@osmc /usr/local/bin/wake-debian"
_WAKE_COOLDOWN_S = 300.0
_last_wake_attempt = 0.0


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


def _host_unreachable(exc: Exception) -> bool:
    """Host down or booting — im Unterschied zu einem Read-Timeout, der eine
    wache, aber beschaeftigte Maschine bedeutet und kein Wecken rechtfertigt."""
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError))


def _wake_and_retry(post, wait_s: float) -> Optional[list[dict[str, Any]]]:
    """Weckt debian per Magic Packet ueber osmc und wiederholt den Abruf, bis er
    gelingt oder die Frist ablaeuft. Rueckgabe: die Chunks, sonst None.

    Wiederholt wird der ECHTE Abruf, nicht bloss ein Erreichbarkeits-Ping: kurz
    nach dem Boot antwortet der Host bereits HTTP, die RAG-API aber noch mit
    502 (so gemessen am 28.07.2026 im Kaltstart-Test). Eine beliebige
    HTTP-Antwort ist also kein Bereitschaftssignal.

    Hoechstens ein Weckversuch pro Cooldown-Fenster: eine Generierung ruft
    mehrfach ab, und bei tatsaechlich toter Maschine darf sich die volle
    Wartezeit nicht pro Abruf stapeln."""
    global _last_wake_attempt
    now = time.monotonic()
    if now - _last_wake_attempt < _WAKE_COOLDOWN_S:
        return None
    _last_wake_attempt = now
    command = os.getenv("RAG_WAKE_COMMAND", _WAKE_COMMAND_DEFAULT).strip()
    if not command:
        return None
    print("[RAG] host nicht erreichbar — Weckversuch per WoL")
    try:
        subprocess.run(shlex.split(command), capture_output=True, timeout=60, check=False)
    except Exception as exc:  # noqa: BLE001 - Wecken ist best effort
        print(f"[RAG] wake failed: {exc}")
        return None
    deadline = time.monotonic() + wait_s
    last: Optional[Exception] = None
    while time.monotonic() < deadline:
        time.sleep(5)
        try:
            chunks = post()
            print(f"[RAG] nach Wecken bereit ({wait_s - (deadline - time.monotonic()):.0f}s)")
            return chunks
        except Exception as exc:  # noqa: BLE001 - bootet noch, Dienste starten
            last = exc
    print(f"[RAG] nach {wait_s:.0f}s immer noch nicht bereit: {last}")
    return None


def retrieve_chunks(
    query: str,
    owner_id: Optional[str],
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
        "owner_id": owner_id,
    }
    headers = {"X-API-Key": _api_key()} if _api_key() else {}

    def _post() -> list[dict[str, Any]]:
        response = httpx.post(
            f"{base}/v1/rag/retrieve", json=payload, headers=headers, timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("chunks", [])

    try:
        chunks = _post()
    except Exception as exc:
        # Ist der Host nur ausgeschaltet, einmal wecken und den Abruf
        # wiederholen — sonst laeuft die Generierung ohne Kanzlei-Praezedenz
        # weiter, und zwar ohne dass man es dem Entwurf ansieht. Der Vertrag aus
        # dem Modulkopf bleibt: im Zweifel leere Liste, Retrieval bricht nichts ab.
        if not _host_unreachable(exc):
            print(f"[RAG] retrieve failed: {exc}")
            return []
        woken = _wake_and_retry(_post, float(os.getenv("RAG_WAKE_WAIT_S", "120")))
        if woken is None:
            print(f"[RAG] retrieve failed: {exc}")
            return []
        chunks = woken

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
    owner_id: str,
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
    chunks = retrieve_chunks(
        query, owner_id, exclude_case_hash=case_hash_from_name(case_name), limit=limit
    )
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
