# RAG-Frontend „Wissensbasis" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Neuer Haupt-Tab „📚 Wissensbasis" mit RAG-Treffersuche (kanzlei/jurisprudence/doktrin) und einem belegpflichtigen, persistenten Chat über den Bestand.

**Architecture:** Bestehender `/v1/rag/retrieve`-Proxy bekommt ein `collection`-Feld. Neu: pures Logik-Modul `app/rag_ask.py` (Prompt-Assembly, `[Qn]`-Belegpflicht, Post-Check), Tabelle `rag_chats` (JSONB-Messages inkl. Quellen), Streaming-Endpoint `POST /v1/rag/ask/stream` (Retrieval fail-closed → LLM-Streaming mit Envelope: JSON-Meta-Zeile + `<<<ANSWER>>>`-Separator + Text) plus Chat-CRUD, Frontend-Tab in `index.html`/`app.js`.

**Tech Stack:** FastAPI, SQLAlchemy + JSONB (Migrationen über `MIGRATIONS` in `main.py`, kein Alembic), httpx, bestehende LLM-Clients aus `shared.py`, Vanilla-JS-Frontend.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-30-rag-wissensbasis-design.md` — bei Widerspruch gewinnt die Spec.
- Tests laufen auf dem Host: `.venv/bin/python -m pytest tests/ -q` (KEIN Docker). Teststil: pure Funktionen, injizierte I/O, kein TestClient, keine Live-Calls (Vorbild `tests/test_draft_context.py`).
- Nach Python-Änderungen: `docker compose restart app job-worker` (Code ist volume-gemountet).
- Modell-Allowlist Chat (identisch Befragen-Dropdown): `gemini-3.6-flash` (Default), `gemini-3.1-pro-preview`, `gpt-5.6-terra`, `claude-sonnet-5`, `grok-4.5`.
- Collection-Allowlist: `kanzlei`, `jurisprudence`, `doktrin`.
- UI-Texte Deutsch. Kanzlei-Stilregel im System-Prompt: keine Semikola.
- Fail-closed: ohne Retrieval-Treffer keine LLM-Antwort.
- Cache-Bust: Versions-Query-Strings in `index.html` bei JS/CSS-Änderungen bumpen (`?v=20260730-N`).
- Commits klein, deutsche Commit-Messages wie im Log üblich, Suffix `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Pures Kern-Modul `app/rag_ask.py` (Prompt, Belegpflicht, Verlauf)

**Files:**
- Create: `app/rag_ask.py`
- Test: `tests/test_rag_ask.py`

**Interfaces:**
- Consumes: nichts (stdlib + typing only).
- Produces (von Task 4 + 5 benutzt):
  - `RAG_COLLECTIONS: tuple[str, ...] = ("kanzlei", "jurisprudence", "doktrin")`
  - `number_chunks(chunks_by_collection: dict[str, list[dict]]) -> list[dict]` — flacht zu `[{"label": "Q1", "collection": str, "header": str, "text": str, "score": float, "chunk_id": str}]`
  - `build_ask_prompt(question: str, sources: list[dict], history: list[dict] | None = None) -> tuple[str, str]` — `(system_instruction, user_prompt)`
  - `has_citation_markers(text: str) -> bool`
  - `trim_history(history: list[dict] | None, max_messages: int = 12) -> list[dict]`
  - `build_chat_messages(question: str, answer: str, model: str, sources: list[dict], now_iso: str) -> list[dict]` — `[user_msg, assistant_msg]`, assistant mit `"unbelegt": bool` und `"sources"`

- [ ] **Step 1: Failing Tests schreiben** (`tests/test_rag_ask.py`)

```python
"""RAG-Ask Kernlogik: Nummerierung, Belegpflicht, Prompt-Assembly.
Pure Funktionen, kein Netzwerk/DB."""
from rag_ask import (
    build_ask_prompt,
    build_chat_messages,
    has_citation_markers,
    number_chunks,
    trim_history,
)


def _chunk(text="Argument X.", header="[Klage | 24/014 | VG Düsseldorf]", score=0.9, cid="c1"):
    return {"text": text, "context_header": header, "score": score, "chunk_id": cid}


def test_number_chunks_labels_across_collections_in_order():
    numbered = number_chunks({
        "kanzlei": [_chunk(cid="a"), _chunk(cid="b")],
        "doktrin": [_chunk(cid="c")],
    })
    assert [s["label"] for s in numbered] == ["Q1", "Q2", "Q3"]
    assert numbered[2]["collection"] == "doktrin"
    assert numbered[0]["header"] == "[Klage | 24/014 | VG Düsseldorf]"


def test_build_ask_prompt_contains_sources_question_and_belegpflicht():
    sources = number_chunks({"kanzlei": [_chunk()]})
    system, prompt = build_ask_prompt("Wie argumentieren wir § 60 Abs. 5?", sources)
    assert "[Q1]" in prompt and "Argument X." in prompt
    assert "kanzlei" in prompt  # Herkunftszeile nennt die Collection
    assert "Wie argumentieren wir § 60 Abs. 5?" in prompt
    assert "[Qn]" in system            # Belegpflicht-Anweisung
    assert "nicht belegt" in system    # Kennzeichnungspflicht für Lücken
    assert "Semikolon" in system or "Semikola" in system


def test_build_ask_prompt_includes_trimmed_history():
    history = [{"role": "user", "content": f"Frage {i}"} for i in range(20)]
    _, prompt = build_ask_prompt("Folgefrage", number_chunks({"kanzlei": [_chunk()]}), history)
    assert "Frage 19" in prompt
    assert "Frage 3" not in prompt  # nur die letzten 12


def test_trim_history_drops_invalid_roles_and_caps_length():
    history = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "böse"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "b"},
    ]
    trimmed = trim_history(history)
    assert trimmed == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_has_citation_markers():
    assert has_citation_markers("Das folgt aus [Q2].")
    assert not has_citation_markers("Behauptung ohne Beleg.")


def test_build_chat_messages_sets_unbelegt_flag_and_sources():
    sources = number_chunks({"kanzlei": [_chunk()]})
    msgs = build_chat_messages("F?", "Antwort ohne Marker", "gemini-3.6-flash", sources, "2026-07-30T12:00:00Z")
    assert msgs[0] == {"role": "user", "content": "F?", "created_at": "2026-07-30T12:00:00Z"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["unbelegt"] is True
    assert msgs[1]["model"] == "gemini-3.6-flash"
    assert msgs[1]["sources"][0]["label"] == "Q1"

    msgs_ok = build_chat_messages("F?", "Belegt [Q1].", "gemini-3.6-flash", sources, "2026-07-30T12:00:00Z")
    assert msgs_ok[1]["unbelegt"] is False
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `.venv/bin/python -m pytest tests/test_rag_ask.py -q`
Expected: FAIL / Collection-Error mit `ModuleNotFoundError: No module named 'rag_ask'`

- [ ] **Step 3: `app/rag_ask.py` implementieren**

```python
"""Kernlogik für den Wissensbasis-Chat: Quellen nummerieren, belegpflichtigen
Prompt bauen, [Qn]-Post-Check. Pure Funktionen — Retrieval und LLM-Aufrufe
bleiben im Endpoint (endpoints/rag.py)."""
import re
from typing import Any, Dict, List, Optional, Tuple

RAG_COLLECTIONS: Tuple[str, ...] = ("kanzlei", "jurisprudence", "doktrin")

_CITATION_RE = re.compile(r"\[Q\d+\]")

SYSTEM_INSTRUCTION = (
    "Du bist der Wissensbasis-Assistent einer Anwaltskanzlei für Migrations- und Sozialrecht. "
    "Beantworte die Frage AUSSCHLIESSLICH auf Grundlage der nummerierten Quellen [Q1]..[Qn]. "
    "Belege jede tatsächliche und rechtliche Aussage mit dem Marker der tragenden Quelle, z. B. [Q2]. "
    "Wörtliche Übernahmen stehen in Anführungszeichen mit Marker. "
    "Was die Quellen nicht tragen, kennzeichne ausdrücklich als 'im Bestand nicht belegt' und ergänze es nicht aus eigenem Wissen. "
    "Verwende keine Semikola. Antworte auf Deutsch."
)


def number_chunks(chunks_by_collection: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    numbered: List[Dict[str, Any]] = []
    index = 1
    for collection, chunks in chunks_by_collection.items():
        for chunk in chunks or []:
            numbered.append({
                "label": f"Q{index}",
                "collection": collection,
                "header": str(chunk.get("context_header") or ""),
                "text": str(chunk.get("text") or ""),
                "score": float(chunk.get("score") or 0.0),
                "chunk_id": str(chunk.get("chunk_id") or ""),
            })
            index += 1
    return numbered


def trim_history(history: Optional[List[Dict[str, Any]]], max_messages: int = 12) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for msg in history or []:
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:4000]})
    return cleaned[-max_messages:]


def build_ask_prompt(
    question: str,
    sources: List[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    source_blocks = []
    for src in sources:
        origin = f"{src['collection']} {src['header']}".strip()
        source_blocks.append(f"[{src['label']}] ({origin})\n{src['text']}")

    history_block = ""
    trimmed = trim_history(history)
    if trimmed:
        lines = ["BISHERIGER VERLAUF (für Folgefragen):"]
        for msg in trimmed:
            speaker = "Nutzer" if msg["role"] == "user" else "Assistent"
            lines.append(f"{speaker}: {msg['content']}")
        history_block = "\n".join(lines) + "\n\n"

    prompt = (
        "QUELLEN:\n\n" + "\n\n".join(source_blocks) + "\n\n"
        + history_block
        + f"AKTUELLE FRAGE: {question}"
    )
    return SYSTEM_INSTRUCTION, prompt


def has_citation_markers(text: str) -> bool:
    return bool(_CITATION_RE.search(text or ""))


def build_chat_messages(
    question: str,
    answer: str,
    model: str,
    sources: List[Dict[str, Any]],
    now_iso: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": question, "created_at": now_iso},
        {
            "role": "assistant",
            "content": answer,
            "model": model,
            "sources": sources,
            "unbelegt": not has_citation_markers(answer),
            "created_at": now_iso,
        },
    ]
```

Achtung: `Tuple` in die typing-Imports aufnehmen (`from typing import ... Tuple`).

- [ ] **Step 4: Tests laufen lassen — müssen grün sein**

Run: `.venv/bin/python -m pytest tests/test_rag_ask.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/rag_ask.py tests/test_rag_ask.py
git commit -m "Wissensbasis: Kernlogik für belegpflichtigen RAG-Chat (rag_ask)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Persistenz `rag_chats` (ORM + Migration)

**Files:**
- Modify: `app/models.py` (neues Modell ans Dateiende)
- Modify: `app/main.py` (neuer Eintrag ans Ende der `MIGRATIONS`-Liste, vor der schließenden `]`)
- Test: `tests/test_rag_chat_model.py`

**Interfaces:**
- Consumes: nichts Neues.
- Produces (von Task 4 + 5 benutzt): ORM-Klasse `RagChat` mit Spalten `id (UUID PK)`, `owner_id (UUID, FK users.id)`, `title (Text)`, `collections (JSONB list)`, `messages (JSONB list)`, `created_at`, `updated_at` und Methode `to_dict(include_messages: bool = False) -> dict`.

- [ ] **Step 1: Failing Test schreiben** (`tests/test_rag_chat_model.py`)

```python
"""RagChat-ORM: to_dict-Formen für Liste (ohne Messages) und Detail."""
import uuid
from datetime import datetime

from models import RagChat


def _chat():
    chat = RagChat(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        title="Wie argumentieren wir § 60 Abs. 5?",
        collections=["kanzlei", "doktrin"],
        messages=[{"role": "user", "content": "F?", "created_at": "2026-07-30T12:00:00Z"}],
        created_at=datetime(2026, 7, 30, 12, 0, 0),
        updated_at=datetime(2026, 7, 30, 12, 5, 0),
    )
    return chat


def test_to_dict_list_form_omits_messages():
    d = _chat().to_dict()
    assert d["title"].startswith("Wie argumentieren")
    assert d["collections"] == ["kanzlei", "doktrin"]
    assert "messages" not in d
    assert d["updated_at"] == "2026-07-30T12:05:00"


def test_to_dict_detail_form_includes_messages():
    d = _chat().to_dict(include_messages=True)
    assert d["messages"][0]["content"] == "F?"
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `.venv/bin/python -m pytest tests/test_rag_chat_model.py -q`
Expected: FAIL mit `ImportError: cannot import name 'RagChat'`

- [ ] **Step 3: ORM-Modell in `app/models.py` ergänzen** (Stil der Nachbar-Modelle: `Column(UUID(as_uuid=True), ...)`, `default=uuid.uuid4` — exakt so, wie es die bestehenden Modelle in der Datei tun, deren Imports schon vorhanden sind)

```python
class RagChat(Base):
    __tablename__ = "rag_chats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(Text, nullable=False, default="")
    collections = Column(JSONB, default=list)
    messages = Column(JSONB, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self, include_messages: bool = False) -> dict:
        data = {
            "id": str(self.id),
            "title": self.title or "",
            "collections": self.collections or [],
            "message_count": len(self.messages or []),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            data["messages"] = self.messages or []
        return data
```

Vorher prüfen, dass `uuid` und `datetime` in models.py importiert sind (sind sie — die Nachbar-Modelle nutzen beides).

- [ ] **Step 4: Migration in `app/main.py` anhängen** (letzter Eintrag der `MIGRATIONS`-Liste, Muster „2026-07-07_result_job_id")

```python
    (
        "2026-07-30_rag_chats",
        [
            """
            CREATE TABLE IF NOT EXISTS rag_chats (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT '',
                collections JSONB DEFAULT '[]'::jsonb,
                messages JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT now(),
                updated_at TIMESTAMP DEFAULT now()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_rag_chats_owner_updated
                ON rag_chats (owner_id, updated_at DESC)
            """,
        ],
    ),
```

- [ ] **Step 5: Tests laufen lassen — grün, dann Migration live anwenden und verifizieren**

Run: `.venv/bin/python -m pytest tests/test_rag_chat_model.py -q` → 2 passed
Run: `docker compose restart app` und danach
`docker exec rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db -c "\\d rag_chats"`
Expected: Tabelle existiert mit den 7 Spalten. (Beide Container wenden Migrationen beim Start an — einmal app reicht.)

- [ ] **Step 6: Commit**

```bash
git add app/models.py app/main.py tests/test_rag_chat_model.py
git commit -m "Wissensbasis: rag_chats-Tabelle (persistente RAG-Chats)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Retrieve-Proxy um `collection` erweitern

**Files:**
- Modify: `app/shared.py` (Klasse `RagRetrieveRequest`, ~Zeile 1404)
- Modify: `app/endpoints/rag.py` (`retrieve_chunks`-Endpoint, ~Zeile 247)
- Test: `tests/test_rag_collection_allowlist.py`

**Interfaces:**
- Consumes: `RAG_COLLECTIONS` aus `rag_ask` (Task 1).
- Produces: `RagRetrieveRequest.collection: Optional[str]` (Pydantic), Modul-Funktion `validate_rag_collection(collection: Optional[str]) -> None` in `endpoints/rag.py` (raised `HTTPException(400)` bei unbekannter Collection; `None` = Default `kanzlei` auf debian-Seite, erlaubt).

- [ ] **Step 1: Failing Test schreiben** (`tests/test_rag_collection_allowlist.py`)

```python
"""Collection-Allowlist des RAG-Proxys: nur kanzlei/jurisprudence/doktrin."""
import pytest
from fastapi import HTTPException

from endpoints.rag import validate_rag_collection


@pytest.mark.parametrize("ok", [None, "kanzlei", "jurisprudence", "doktrin"])
def test_known_collections_pass(ok):
    validate_rag_collection(ok)  # darf nicht raisen


def test_unknown_collection_rejected():
    with pytest.raises(HTTPException) as exc:
        validate_rag_collection("geheim")
    assert exc.value.status_code == 400
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `.venv/bin/python -m pytest tests/test_rag_collection_allowlist.py -q`
Expected: FAIL mit `ImportError: cannot import name 'validate_rag_collection'`

- [ ] **Step 3: Implementieren**

`app/shared.py` — in `RagRetrieveRequest` nach `use_reranker` ergänzen:

```python
    collection: Optional[str] = None
```

`app/endpoints/rag.py` — nach `RESERVED_RAG_COLLECTIONS` ergänzen und im Endpoint aufrufen:

```python
from rag_ask import RAG_COLLECTIONS


def validate_rag_collection(collection) -> None:
    if collection is not None and collection not in RAG_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unbekannte Collection '{collection}'. Erlaubt: {', '.join(RAG_COLLECTIONS)}",
        )
```

Im `retrieve_chunks`-Endpoint als erste Zeile: `validate_rag_collection(body.collection)`.
Kein weiterer Umbau — `_build_retrieve_payload` nutzt `body.model_dump(exclude_none=True)`, das neue Feld fließt automatisch durch.

- [ ] **Step 4: Tests laufen lassen — grün**

Run: `.venv/bin/python -m pytest tests/test_rag_collection_allowlist.py tests/test_rag_ask.py -q`
Expected: alle passed

- [ ] **Step 5: Commit**

```bash
git add app/shared.py app/endpoints/rag.py tests/test_rag_collection_allowlist.py
git commit -m "Wissensbasis: collection-Feld im RAG-Retrieve-Proxy (Allowlist)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Streaming-Endpoint `POST /v1/rag/ask/stream`

**Files:**
- Modify: `app/endpoints/rag.py` (Request-Modell, Retrieval-Sammler, LLM-Streaming, Persistenz)
- Test: `tests/test_rag_ask_endpoint_logic.py`

**Interfaces:**
- Consumes: `build_ask_prompt`, `number_chunks`, `build_chat_messages`, `trim_history`, `RAG_COLLECTIONS` (Task 1), `RagChat` (Task 2), `validate_rag_collection` (Task 3), `rag_context.retrieve_chunks(query, owner_id, limit, use_reranker, collection)`, LLM-Factories aus `shared.py` (`get_gemini_client`, `get_openai_client`, `get_anthropic_client`, `get_xai_client`, `resolve_openai_model`).
- Produces: HTTP-Vertrag für Task 6 (Frontend):
  - Request-JSON: `{"question": str, "collections": ["kanzlei", ...] (leer/fehlend = alle), "model": <Allowlist>, "chat_id": uuid|null}`
  - Response: `text/plain`-Stream. Erste Zeile: JSON `{"chat_id": "<uuid>", "sources": [<numbered source>...]}`. Danach Separatorzeile exakt `<<<ANSWER>>>`. Danach Antworttext-Chunks.
  - Fehler vor Stream-Beginn als HTTP 4xx/5xx mit `detail` (Frontend zeigt `detail`).
- Zusätzlich pure Hilfsfunktionen in `endpoints/rag.py` (testbar ohne Netz):
  - `collect_ask_sources(question, collections, owner_id, retrieve_fn) -> list[dict]` — ruft `retrieve_fn(query=..., owner_id=..., limit=8, use_reranker=True, collection=c)` je Collection, nummeriert via `number_chunks`.
  - `build_stream_envelope(chat_id: str, sources: list[dict]) -> str` — `json.dumps(...) + "\n<<<ANSWER>>>\n"`.

- [ ] **Step 1: Failing Tests schreiben** (`tests/test_rag_ask_endpoint_logic.py`)

```python
"""Ask-Endpoint-Logik: Quellen-Sammlung (fail-closed) und Stream-Envelope.
retrieve_fn wird injiziert — kein Netzwerk."""
import json

from endpoints.rag import build_stream_envelope, collect_ask_sources


def test_collect_ask_sources_queries_each_collection_and_numbers():
    calls = []

    def fake_retrieve(query, owner_id, limit, use_reranker, collection):
        calls.append(collection)
        return [{"text": f"T-{collection}", "context_header": "[H]", "score": 0.5, "chunk_id": collection}]

    sources = collect_ask_sources("frage", ["kanzlei", "doktrin"], "u1", fake_retrieve)
    assert calls == ["kanzlei", "doktrin"]
    assert [s["label"] for s in sources] == ["Q1", "Q2"]
    assert sources[1]["collection"] == "doktrin"


def test_collect_ask_sources_empty_collections_means_all_three():
    seen = []

    def fake_retrieve(query, owner_id, limit, use_reranker, collection):
        seen.append(collection)
        return []

    sources = collect_ask_sources("frage", [], "u1", fake_retrieve)
    assert seen == ["kanzlei", "jurisprudence", "doktrin"]
    assert sources == []  # fail-closed entscheidet der Endpoint anhand der leeren Liste


def test_build_stream_envelope_format():
    envelope = build_stream_envelope("abc-123", [{"label": "Q1"}])
    meta_line, rest = envelope.split("\n", 1)
    assert json.loads(meta_line) == {"chat_id": "abc-123", "sources": [{"label": "Q1"}]}
    assert rest == "<<<ANSWER>>>\n"
```

- [ ] **Step 2: Tests laufen lassen — müssen fehlschlagen**

Run: `.venv/bin/python -m pytest tests/test_rag_ask_endpoint_logic.py -q`
Expected: FAIL mit ImportError

- [ ] **Step 3: Endpoint + Helfer in `app/endpoints/rag.py` implementieren**

Imports oben ergänzen:

```python
import asyncio
import json
import uuid as uuid_module
from datetime import datetime
from typing import List, Literal

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import rag_context
from database import SessionLocal
from models import RagChat
from rag_ask import (
    RAG_COLLECTIONS,
    build_ask_prompt,
    build_chat_messages,
    number_chunks,
    trim_history,
)
from shared import (
    get_anthropic_client,
    get_gemini_client,
    get_openai_client,
    get_xai_client,
    resolve_openai_model,
)
```

Request-Modell + Helfer:

```python
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
            if getattr(event, "type", None) == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield delta
            elif getattr(event, "type", None) in {"response.failed", "response.incomplete", "error"}:
                raise RuntimeError(f"OpenAI-Stream abgebrochen: {getattr(event, 'type', '?')}")
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
```

Endpoint (Persistenz mit frischer `SessionLocal` nach Stream-Ende — Muster `send_to_jlawyer`-Reflection in generation.py):

```python
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
    is_new_chat = chat_uuid is None

    async def generate():
        answer_parts: List[str] = []
        yield build_stream_envelope(chat_id_str, sources)
        try:
            for delta in await asyncio.to_thread(lambda: list(_stream_llm_answer(model, system_instruction, prompt))):
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
                        collections=[c for c in (body.collections or []) if c] or list(RAG_COLLECTIONS),
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
```

Hinweis für den Implementierer: `await asyncio.to_thread(lambda: list(...))` sammelt die LLM-Antwort blockierungsfrei, streamt dann aus der Liste — einfach und korrekt, echtes Durchreichen einzelner Deltas wäre mit sync-Generatoren im Threadpool fehleranfällig (Iteration über einen sync-Generator direkt im async-Generator blockiert den Event-Loop). Die gefühlte Latenz bleibt ok, weil die Envelope (Quellen!) sofort kommt. Wer will, darf stattdessen einen `queue.Queue`-Brückenmechanismus bauen — nicht Pflicht.

- [ ] **Step 4: Tests laufen lassen — grün, App startet**

Run: `.venv/bin/python -m pytest tests/test_rag_ask_endpoint_logic.py tests/test_rag_ask.py tests/test_rag_collection_allowlist.py -q` → alle passed
Run: `docker compose restart app job-worker` und `docker compose logs --since 30s app | grep -i "error\|traceback"` → keine Import-Fehler

- [ ] **Step 5: Commit**

```bash
git add app/endpoints/rag.py tests/test_rag_ask_endpoint_logic.py
git commit -m "Wissensbasis: /v1/rag/ask/stream (fail-closed, belegpflichtig, persistent)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Chat-CRUD-Endpoints

**Files:**
- Modify: `app/endpoints/rag.py`
- Test: `tests/test_rag_chat_scoping.py`

**Interfaces:**
- Consumes: `RagChat` (Task 2).
- Produces (Frontend-Vertrag für Task 6):
  - `GET /v1/rag/chats` → `{"chats": [chat.to_dict()…]}` (owner-gescopet, sortiert `updated_at DESC`, Limit 50)
  - `GET /v1/rag/chats/{chat_id}` → `chat.to_dict(include_messages=True)` oder 404
  - `DELETE /v1/rag/chats/{chat_id}` → `{"deleted": true}` oder 404
  - Pure Helfer `query_owner_chat(db, chat_id_str, owner_id)` → RagChat | raises HTTPException(400/404) — von GET/DELETE geteilt.

- [ ] **Step 1: Failing Test schreiben** (`tests/test_rag_chat_scoping.py`)

```python
"""Owner-Scoping der Chat-CRUD-Helfer: fremde/ungültige IDs -> 400/404.
DB-Query wird über ein Fake-Session-Objekt injiziert."""
import uuid

import pytest
from fastapi import HTTPException

from endpoints.rag import query_owner_chat


class _FakeQuery:
    def __init__(self, result):
        self._result = result
    def filter(self, *args):
        return self
    def first(self):
        return self._result


class _FakeDb:
    def __init__(self, result):
        self._result = result
    def query(self, model):
        return _FakeQuery(self._result)


def test_invalid_uuid_raises_400():
    with pytest.raises(HTTPException) as exc:
        query_owner_chat(_FakeDb(None), "kein-uuid", uuid.uuid4())
    assert exc.value.status_code == 400


def test_missing_chat_raises_404():
    with pytest.raises(HTTPException) as exc:
        query_owner_chat(_FakeDb(None), str(uuid.uuid4()), uuid.uuid4())
    assert exc.value.status_code == 404


def test_found_chat_returned():
    sentinel = object()
    assert query_owner_chat(_FakeDb(sentinel), str(uuid.uuid4()), uuid.uuid4()) is sentinel
```

- [ ] **Step 2: Test laufen lassen — muss fehlschlagen**

Run: `.venv/bin/python -m pytest tests/test_rag_chat_scoping.py -q`
Expected: FAIL mit ImportError

- [ ] **Step 3: Implementieren** (in `app/endpoints/rag.py`; `get_db` aus `database` importieren, `Session` aus `sqlalchemy.orm`)

```python
def query_owner_chat(db, chat_id_str: str, owner_id):
    try:
        chat_uuid = uuid_module.UUID(chat_id_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Ungültige chat_id")
    chat = db.query(RagChat).filter(
        RagChat.id == chat_uuid, RagChat.owner_id == owner_id
    ).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat nicht gefunden")
    return chat


@router.get("/chats")
@limiter.limit("120/hour")
async def list_rag_chats(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    chats = (
        db.query(RagChat)
        .filter(RagChat.owner_id == current_user.id)
        .order_by(RagChat.updated_at.desc())
        .limit(50)
        .all()
    )
    return {"chats": [chat.to_dict() for chat in chats]}


@router.get("/chats/{chat_id}")
@limiter.limit("120/hour")
async def get_rag_chat(
    request: Request,
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return query_owner_chat(db, chat_id, current_user.id).to_dict(include_messages=True)


@router.delete("/chats/{chat_id}")
@limiter.limit("60/hour")
async def delete_rag_chat(
    request: Request,
    chat_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    chat = query_owner_chat(db, chat_id, current_user.id)
    db.delete(chat)
    db.commit()
    return {"deleted": True}
```

- [ ] **Step 4: Tests laufen lassen — grün, danach voller Suitelauf**

Run: `.venv/bin/python -m pytest tests/test_rag_chat_scoping.py -q` → 3 passed
Run: `.venv/bin/python -m pytest tests/ -q -m "not slow"` → alles grün

- [ ] **Step 5: Commit**

```bash
git add app/endpoints/rag.py tests/test_rag_chat_scoping.py
git commit -m "Wissensbasis: Chat-Verlauf-Endpoints (Liste/Detail/Löschen, owner-gescopet)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Frontend-Tab „📚 Wissensbasis"

**Files:**
- Modify: `app/templates/index.html` (Tab-Button, Tab-Inhalt, Cache-Bust-Versionen)
- Modify: `app/static/js/app.js` (`switchMainTab` auf 3 Tabs, neue Funktionen)
- Modify: `app/static/css/styles.css` (Karten/Marker-Styles)

**Interfaces:**
- Consumes: HTTP-Verträge aus Task 4 (Stream-Envelope `JSON\n<<<ANSWER>>>\n…`) und Task 5 (Chat-CRUD), bestehender Proxy `POST /v1/rag/retrieve` mit `collection` (Task 3). Auth kommt automatisch: der globale `window.fetch`-Wrapper (app.js ~Z. 730) hängt den JWT an.
- Produces: nichts für spätere Tasks.

- [ ] **Step 1: Markup in `index.html`** — Tab-Leiste erweitern:

```html
    <div class="main-tabs">
        <button class="main-tab active" id="mainTabWork" onclick="switchMainTab('work')">Arbeitsbereich</button>
        <button class="main-tab" id="mainTabMemory" onclick="switchMainTab('memory')">🧠 Fall-Speicher</button>
        <button class="main-tab" id="mainTabRag" onclick="switchMainTab('rag')">📚 Wissensbasis</button>
    </div>
```

Direkt nach dem `#caseMemoryBox`-Block (vor den `<script>`-Tags) den Tab-Inhalt einfügen:

```html
    <div class="category-box" id="ragTabContent" style="display: none;">
        <div class="category-header">
            <h3 style="margin: 0;">📚 Wissensbasis</h3>
        </div>
        <p style="color: #7f8c8d; font-size: 14px; margin: 0 0 12px 0;">
            Suche und belegte Fragen über den Kanzlei-Bestand (eigene Schriftsätze, Rechtsprechung, Doktrin).
        </p>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;">
            <input type="text" id="ragQueryInput" placeholder="Suchbegriff oder Frage..."
                style="flex: 1 1 320px; padding: 8px; border: 1px solid #bdc3c7; border-radius: 5px;">
            <select id="ragCollectionSelect" style="padding: 6px; border: 1px solid #bdc3c7; border-radius: 5px;">
                <option value="" selected>Alle Collections</option>
                <option value="kanzlei">Kanzlei-Schriftsätze</option>
                <option value="jurisprudence">Rechtsprechung</option>
                <option value="doktrin">Doktrin (Wiki)</option>
            </select>
            <select id="ragModelSelect" style="padding: 6px; border: 1px solid #bdc3c7; border-radius: 5px;">
                <option value="gemini-3.6-flash" selected>Gemini 3.6 Flash</option>
                <option value="gemini-3.1-pro-preview">Gemini 3.1 Pro</option>
                <option value="gpt-5.6-terra">GPT-5.6 Terra</option>
                <option value="claude-sonnet-5">Claude Sonnet 5</option>
                <option value="grok-4.5">Grok 4.5</option>
            </select>
            <button class="btn" onclick="ragSearch()" style="background-color: #16a085;">Suchen</button>
            <button class="btn" onclick="ragAsk()" style="background-color: #8e44ad;">Fragen</button>
        </div>
        <div id="ragStatus" style="font-size: 13px; color: #7f8c8d; margin-bottom: 8px;"></div>
        <div id="ragChatArea"></div>
        <div id="ragSearchResults"></div>
        <div style="margin-top: 14px; border-top: 1px solid #eee; padding-top: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <h4 style="margin: 0; font-size: 14px; color: #2c3e50;">Chat-Verlauf</h4>
                <button class="btn btn-small" onclick="ragNewChat()" style="background-color: #2c3e50;">Neuer Chat</button>
            </div>
            <div id="ragChatList" style="margin-top: 8px; max-height: 220px; overflow-y: auto;"></div>
        </div>
    </div>
```

Cache-Bust: `styles.css`, `app.js` auf `?v=20260730-1`.

- [ ] **Step 2: `switchMainTab` in `app.js` auf drei Tabs umbauen** (bestehende Funktion ERSETZEN)

```javascript
const MAIN_TABS = {
    work: { content: 'workTabContent', button: 'mainTabWork' },
    memory: { content: 'caseMemoryBox', button: 'mainTabMemory' },
    rag: { content: 'ragTabContent', button: 'mainTabRag' },
};

function switchMainTab(tab) {
    if (!MAIN_TABS[tab]) tab = 'work';
    for (const [name, ids] of Object.entries(MAIN_TABS)) {
        const content = document.getElementById(ids.content);
        const button = document.getElementById(ids.button);
        if (content) content.style.display = name === tab ? '' : 'none';
        if (button) button.classList.toggle('active', name === tab);
    }
    if (tab === 'memory') {
        loadCaseMemory({ silent: true, skipIfDirty: true }).catch((err) => debugError('loadCaseMemory failed', err));
        loadMemoryProposals().catch((err) => debugError('loadMemoryProposals failed', err));
        loadWikiEntries().catch((err) => debugError('loadWikiEntries failed', err));
    }
    if (tab === 'rag') {
        loadRagChats().catch((err) => debugError('loadRagChats failed', err));
    }
}
```

- [ ] **Step 3: Wissensbasis-Funktionen in `app.js` ergänzen** (am Dateiende, vor eventuellen Export-Blöcken)

```javascript
let ragChatState = { chatId: null };

function ragSelectedCollections() {
    const value = document.getElementById('ragCollectionSelect')?.value || '';
    return value ? [value] : [];
}

function setRagStatus(text, isError = false) {
    const el = document.getElementById('ragStatus');
    if (el) { el.textContent = text || ''; el.style.color = isError ? '#c0392b' : '#7f8c8d'; }
}

// ACHTUNG: escapeHtml existiert bereits in app.js (~Z. 3623) — NICHT neu
// definieren, die bestehende Funktion wird hier wiederverwendet.

function renderRagSourceCard(src) {
    const badge = escapeHtml(src.collection || '');
    const header = escapeHtml(src.header || '');
    const score = typeof src.score === 'number' ? src.score.toFixed(2) : '';
    return `<div class="rag-source-card" id="rag-source-${escapeHtml(src.label)}">
        <div class="rag-source-head"><b>[${escapeHtml(src.label)}]</b>
            <span class="rag-badge">${badge}</span> ${header}
            <span style="color:#95a5a6;">${score}</span></div>
        <div class="rag-source-text">${escapeHtml(src.text || '')}</div>
    </div>`;
}

async function ragSearch() {
    const query = document.getElementById('ragQueryInput')?.value.trim();
    if (!query) { setRagStatus('Bitte Suchbegriff eingeben.', true); return; }
    const collections = ragSelectedCollections();
    const targets = collections.length ? collections : ['kanzlei', 'jurisprudence', 'doktrin'];
    setRagStatus('Suche läuft...');
    const resultsDiv = document.getElementById('ragSearchResults');
    resultsDiv.innerHTML = '';
    try {
        const responses = await Promise.all(targets.map(collection =>
            fetch('/v1/rag/retrieve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, collection, limit: 8, use_reranker: true }),
            }).then(async r => {
                if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
                return { collection, data: await r.json() };
            })
        ));
        let html = '';
        let total = 0;
        for (const { collection, data } of responses) {
            const chunks = data.chunks || [];
            total += chunks.length;
            html += `<h4 class="rag-group-title">${escapeHtml(collection)} (${chunks.length})</h4>`;
            let i = 0;
            for (const chunk of chunks) {
                i += 1;
                html += renderRagSourceCard({
                    label: `${collection}-${i}`, collection,
                    header: chunk.context_header, text: chunk.text, score: chunk.score,
                });
            }
        }
        resultsDiv.innerHTML = html;
        setRagStatus(total ? `${total} Treffer.` : 'Keine Treffer im Bestand.');
    } catch (error) {
        setRagStatus(`Suche fehlgeschlagen: ${error.message}`, true);
    }
}

function renderRagAnswerHtml(answerText, sources, unbelegt) {
    let html = (typeof marked !== 'undefined' && marked.parse)
        ? marked.parse(answerText) : escapeHtml(answerText);
    html = html.replace(/\[(Q\d+)\]/g,
        '<a class="rag-cite" href="#rag-source-$1" onclick="document.getElementById(\'rag-source-$1\')?.scrollIntoView({behavior:\'smooth\'});return false;">[$1]</a>');
    const banner = unbelegt
        ? '<div class="rag-warn">⚠️ Antwort enthält keine [Qn]-Belege — mit Vorsicht behandeln.</div>' : '';
    const sourcesHtml = (sources || []).map(renderRagSourceCard).join('');
    return `${banner}<div class="markdown-content">${html}</div>
        <details class="rag-sources-details" open><summary>Verwendete Quellen (${(sources || []).length})</summary>${sourcesHtml}</details>`;
}

async function ragAsk() {
    const question = document.getElementById('ragQueryInput')?.value.trim();
    if (!question) { setRagStatus('Bitte Frage eingeben.', true); return; }
    const model = document.getElementById('ragModelSelect')?.value || 'gemini-3.6-flash';
    setRagStatus('Belege werden gesucht, Antwort wird generiert...');
    const chatArea = document.getElementById('ragChatArea');
    const questionHtml = `<div class="rag-question">❓ ${escapeHtml(question)}</div>`;
    const answerDiv = document.createElement('div');
    chatArea.insertAdjacentHTML('beforeend', questionHtml);
    chatArea.appendChild(answerDiv);
    try {
        const response = await fetch('/v1/rag/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question, model,
                collections: ragSelectedCollections(),
                chat_id: ragChatState.chatId,
            }),
        });
        if (!response.ok) {
            const errText = (await response.json().catch(() => ({}))).detail || `HTTP ${response.status}`;
            throw new Error(errText);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let meta = null;
        let answerText = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            if (!meta) {
                const sep = buffer.indexOf('\n<<<ANSWER>>>\n');
                if (sep === -1) continue;
                meta = JSON.parse(buffer.slice(0, sep));
                ragChatState.chatId = meta.chat_id;
                buffer = buffer.slice(sep + '\n<<<ANSWER>>>\n'.length);
            }
            answerText = buffer;
            answerDiv.innerHTML = renderRagAnswerHtml(answerText, meta ? meta.sources : [], false);
        }
        const unbelegt = !/\[Q\d+\]/.test(answerText);
        answerDiv.innerHTML = renderRagAnswerHtml(answerText, meta ? meta.sources : [], unbelegt);
        setRagStatus('');
        document.getElementById('ragQueryInput').value = '';
        loadRagChats().catch(() => {});
    } catch (error) {
        answerDiv.innerHTML = `<div class="rag-warn">${escapeHtml(error.message)}</div>`;
        setRagStatus('Fragen fehlgeschlagen.', true);
    }
}

function ragNewChat() {
    ragChatState.chatId = null;
    document.getElementById('ragChatArea').innerHTML = '';
    document.getElementById('ragSearchResults').innerHTML = '';
    setRagStatus('Neuer Chat.');
}

async function loadRagChats() {
    const listDiv = document.getElementById('ragChatList');
    if (!listDiv) return;
    const response = await fetch('/v1/rag/chats');
    if (!response.ok) { listDiv.innerHTML = '<div style="font-size:12px;color:#95a5a6;">Verlauf nicht ladbar.</div>'; return; }
    const data = await response.json();
    if (!data.chats?.length) { listDiv.innerHTML = '<div style="font-size:12px;color:#95a5a6;">Noch keine Chats.</div>'; return; }
    listDiv.innerHTML = data.chats.map(chat => `
        <div class="rag-chat-row">
            <a href="#" onclick="openRagChat('${chat.id}');return false;">${escapeHtml(chat.title || '(ohne Titel)')}</a>
            <span style="color:#95a5a6;font-size:11px;">${(chat.updated_at || '').slice(0, 16).replace('T', ' ')} · ${chat.message_count} Nachrichten</span>
            <button class="btn btn-small" onclick="deleteRagChat('${chat.id}')" style="background:#e0e6e8;color:#7f8c8d;padding:1px 7px;">✕</button>
        </div>`).join('');
}

async function openRagChat(chatId) {
    const response = await fetch(`/v1/rag/chats/${chatId}`);
    if (!response.ok) { setRagStatus('Chat nicht ladbar.', true); return; }
    const chat = await response.json();
    ragChatState.chatId = chat.id;
    const chatArea = document.getElementById('ragChatArea');
    chatArea.innerHTML = '';
    for (const msg of chat.messages || []) {
        if (msg.role === 'user') {
            chatArea.insertAdjacentHTML('beforeend', `<div class="rag-question">❓ ${escapeHtml(msg.content)}</div>`);
        } else {
            const div = document.createElement('div');
            div.innerHTML = renderRagAnswerHtml(msg.content || '', msg.sources || [], !!msg.unbelegt);
            chatArea.appendChild(div);
        }
    }
    setRagStatus('Chat geladen — Folgefragen gehen in diesen Chat.');
}

async function deleteRagChat(chatId) {
    if (!confirm('Diesen Chat endgültig löschen?')) return;
    await fetch(`/v1/rag/chats/${chatId}`, { method: 'DELETE' });
    if (ragChatState.chatId === chatId) ragNewChat();
    loadRagChats().catch(() => {});
}
```

- [ ] **Step 4: CSS in `styles.css` ergänzen**

```css
.rag-source-card {
    border: 1px solid #e0e6e8;
    border-radius: 6px;
    padding: 8px 10px;
    margin: 6px 0;
    font-size: 13px;
}

.rag-source-head { margin-bottom: 4px; color: #2c3e50; }

.rag-source-text { color: #555; white-space: pre-wrap; max-height: 160px; overflow-y: auto; }

.rag-badge {
    background: #eaf2f8;
    color: #2471a3;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 11px;
    margin: 0 4px;
}

.rag-group-title { margin: 12px 0 4px; color: #2c3e50; font-size: 14px; }

.rag-question { font-weight: 600; margin: 14px 0 6px; color: #2c3e50; }

.rag-warn {
    background: #fdecea;
    color: #c0392b;
    border: 1px solid #f5b7b1;
    border-radius: 6px;
    padding: 6px 10px;
    margin: 6px 0;
    font-size: 13px;
}

.rag-cite { color: #8e44ad; font-weight: 600; text-decoration: none; }

.rag-chat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 3px 0;
    font-size: 13px;
}

.rag-sources-details summary { cursor: pointer; font-size: 13px; color: #2c3e50; margin-top: 8px; }
```

- [ ] **Step 5: Neustart + manueller Smoke-Test**

```bash
docker compose restart app
```

Browser (rechtmaschine.de, hartes Reload): Tab „📚 Wissensbasis" öffnen →
(a) Suche mit einem Fachbegriff („gewöhnlicher Aufenthalt") liefert gruppierte Karten,
(b) „Fragen" liefert gestreamte Antwort mit anklickbaren [Qn]-Markern und Quellen-Details,
(c) Chat erscheint im Verlauf, lässt sich neu laden und löschen,
(d) Folgefrage im geladenen Chat referenziert den Verlauf.
Falls debian schläft: erste Anfrage → Meldung „…wird geweckt", zweite nach ~1 Min funktioniert.

- [ ] **Step 6: Commit**

```bash
git add app/templates/index.html app/static/js/app.js app/static/css/styles.css
git commit -m "Wissensbasis: Frontend-Tab (Suche, belegter Chat, Verlauf)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Integrations-Smoke + Gesamtsuite

**Files:**
- keine neuen — Verifikation.

- [ ] **Step 1: Gesamtsuite**

Run: `.venv/bin/python -m pytest tests/ -q -m "not slow"`
Expected: alles grün (Basis vor diesem Feature: 411 passed, 3 skipped).

- [ ] **Step 2: Live-Smoke gegen die laufende App** (Token-Handling siehe Skill `rechtmaschine-api-cli`)

```bash
# Login + Retrieve-Smoke (collection-Feld) + Chat-Liste
TOKEN=$(curl -s -X POST https://rechtmaschine.de/token -d "username=...&password=..." | jq -r .access_token)
curl -s -X POST https://rechtmaschine.de/v1/rag/retrieve \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query": "gewöhnlicher Aufenthalt", "collection": "kanzlei", "limit": 3, "use_reranker": true}' | jq '.chunks | length'
curl -s https://rechtmaschine.de/v1/rag/chats -H "Authorization: Bearer $TOKEN" | jq .
```

Expected: Trefferzahl > 0 (debian wach), Chat-Liste `{"chats": [...]}`.

- [ ] **Step 3: Abschluss-Commit falls Reste** (z. B. Doku-Nachträge), sonst nichts.
