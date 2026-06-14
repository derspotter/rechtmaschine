# RAG Controlled-Vocabulary Schlagwörter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every chunk in every RAG collection canonical Schlagwörter + Herkunftsland + Normen as real metadata, written into both `context_header` (so the existing dense + sparse hybrid channels index them automatically) and `metadata` (for future faceting).

**Architecture:** A single controlled vocabulary (seeded from the 473 asyl.net `RechtsprechungEntry` rows) normalizes all tags onto shared tokens. A uniform "scroll → tag → re-upsert" pass over the existing store re-tags both collections without re-extracting/re-downloading/re-anonymizing anything. Jurisprudence tags come from the DB entries; kanzlei tags come from one desktop-Qwen call per document on already-anonymized text. The tag-writing is then baked into the live ingest paths so new documents are born tagged.

**Tech Stack:** Python 3, FastAPI (RAG API on debian), psycopg2 + pgvector, httpx, TEI bge-m3 (re-embeds on re-upsert), desktop Qwen via `call_qwen_json` (`/qwen-json`), SQLAlchemy (app DB, `RechtsprechungEntry`).

---

## Background facts the implementer must know

- **No Alembic, no pytest suite.** Tests in `tests/` are standalone scripts run with `python tests/<name>.py`; they use plain `assert`. Follow that style.
- **The app container mounts only `./app:/app` and `./anon:/app/anon:ro`** (`docker-compose.yml:33-34`). It does **not** see `rag/`. So all app-side code (vocab, tagger, retag tool) lives under `app/`, and the vocab JSON lives under `app/` so it is mounted.
- **`rag/ingest_runner.py` runs on the import host (debian)** where the full repo is checked out, so it can reach `app/` by inserting that path.
- **Both retag passes run inside the app container** (`docker exec rechtmaschine-app python ...`): the container has the app DB (`from database import SessionLocal`), the vocab JSON, Tailscale reach to desktop Qwen and to the debian RAG API.
- **Both hybrid channels already read `context_header`:** sparse `search_text = to_tsvector('german', context_header || ' ' || text)` (`rag/db/init.sql:12`); dense embedding input is `context_header + "\n\n" + text` (`rag/api/main.py:180`). So tags in `context_header` are picked up with no schema change.
- **Re-embedding on re-upsert:** `/v1/rag/chunks/upsert` re-embeds a chunk only when `chunk.dense` is empty (`rag/api/main.py:346-350`). The retag tool therefore sends `dense: []` to force a fresh embedding with the new `context_header`.
- **Existing Qwen client:** `call_qwen_json(service_url, prompt, *, num_predict, temperature, num_ctx)` in `app/citation_qwen.py:112` POSTs to `{service_url}/qwen-json` with `format:"json"` and returns a parsed dict. `service_url` is read from env `ANONYMIZATION_SERVICE_URL` (see `app/citation_qwen.py:572`).
- **RAG API auth:** `X-API-Key` header (`rag/api/main.py:92`). The app already has the key in env `RAG_SERVICE_API_KEY` (and `RAG_API_KEY`).
- **Existing chunk-id prefixes:** kanzlei `nc-{sha16}-{idx:03d}` (`rag/ingest_runner.py:472`), jurisprudence `juris-{sha16}-{idx:03d}` (`app/jurisprudence_ingest.py:416`). `metadata.chunk_index` carries the order; jurisprudence metadata carries `rechtsprechung_entry_id` (`app/jurisprudence_ingest.py:402`).

---

## File structure

- Create `app/rag_vocabulary.py` — vocab dataclass + loader + `normalize_themen/country/normen` + `tag_line` + `facet_metadata`. Pure, dependency-injectable (functions take a `Vocabulary`).
- Create `app/rag_vocabulary.json` — generated controlled vocabulary (committed).
- Create `app/build_vocabulary.py` — aggregates `RechtsprechungEntry` rows → `app/rag_vocabulary.json`.
- Create `app/qwen_tagger.py` — `tag_document(text, vocab) -> dict` via desktop Qwen, normalized through the vocab.
- Create `app/retag_rag.py` — `jurisprudence` and `kanzlei` subcommands: scroll → tag → re-upsert.
- Modify `rag/api/main.py` — add `POST /v1/rag/chunks/scroll`.
- Modify `app/jurisprudence_ingest.py` — write normalized tags into `context_header` + `metadata` at ingest (`:386`, `:400`).
- Modify `rag/ingest_runner.py` — add Qwen doc-tag step + write tags into `context_header` + `metadata` at ingest (`:455`, `:457`).
- Create `tests/test_rag_vocabulary.py` — assert-script unit tests for the normalizer.

---

## Task 1: Vocabulary normalizer module

**Files:**
- Create: `app/rag_vocabulary.py`
- Test: `tests/test_rag_vocabulary.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_vocabulary.py`:

```python
"""Unit tests for the controlled-vocabulary normalizer. Pure Python, no DB/GPU.
Run: python tests/test_rag_vocabulary.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag_vocabulary import (
    Vocabulary, normalize_themen, normalize_country, normalize_normen,
    tag_line, facet_metadata,
)

VOCAB = Vocabulary(
    themen=["wehrdienstentziehung", "internationaler schutz in eu-staat", "abschiebungsverbot"],
    themen_aliases={"wehrdienstverweigerung": "wehrdienstentziehung",
                    "militärdienstentziehung": "wehrdienstentziehung"},
    laender=["Syrien", "Griechenland", "Afghanistan"],
    laender_aliases={"arabische republik syrien": "Syrien", "syrische": "Syrien"},
    normen=["§ 3 AsylG", "§ 60 Abs. 7 AufenthG", "Art. 3 EMRK"],
    normen_aliases={"§ 3 asylg": "§ 3 AsylG", "art 3 emrk": "Art. 3 EMRK"},
)

def test_themen_canonical_and_alias():
    out = normalize_themen(VOCAB, ["Wehrdienstverweigerung", "Abschiebungsverbot", "unbekanntes thema"])
    assert out == ["wehrdienstentziehung", "abschiebungsverbot"], out

def test_themen_dedup_preserves_order():
    out = normalize_themen(VOCAB, ["Abschiebungsverbot", "Militärdienstentziehung", "Wehrdienstentziehung"])
    assert out == ["abschiebungsverbot", "wehrdienstentziehung"], out

def test_country_alias_and_unknown():
    assert normalize_country(VOCAB, "Arabische Republik Syrien") == "Syrien"
    assert normalize_country(VOCAB, "Griechenland") == "Griechenland"
    assert normalize_country(VOCAB, "Narnia") is None
    assert normalize_country(VOCAB, None) is None

def test_normen_alias_and_filter():
    out = normalize_normen(VOCAB, ["§ 3 AsylG", "art 3 emrk", "§ 99 NichtExist"])
    assert out == ["§ 3 AsylG", "Art. 3 EMRK"], out

def test_tag_line_format():
    line = tag_line(["griechenland"], "Griechenland", ["§ 29 AsylG"])
    assert "Schlagwörter: griechenland" in line
    assert "Herkunftsland: Griechenland" in line
    assert "Normen: § 29 AsylG" in line

def test_tag_line_empty_when_nothing():
    assert tag_line([], None, []) == ""

def test_facet_metadata_keys():
    md = facet_metadata(["abschiebungsverbot"], "Syrien", ["§ 3 AsylG"])
    assert md == {"schlagworte": ["abschiebungsverbot"],
                  "applicant_origin": "Syrien",
                  "citations": ["§ 3 AsylG"]}, md

def test_facet_metadata_omits_empty():
    md = facet_metadata([], None, [])
    assert md == {}, md

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_rag_vocabulary.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_vocabulary'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/rag_vocabulary.py`:

```python
"""Controlled vocabulary for RAG tagging.

A single canonical taxonomy (themen/Herkunftsländer/Normen) shared by every
collection so the same concept is the same token across jurisprudence and the
firm's own filings. Normalization = lowercase/strip/collapse-whitespace, apply
an alias map, then keep only terms in the canonical set. Pure functions that
take a Vocabulary, so they are testable without the generated JSON.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

_WS = re.compile(r"\s+")

DEFAULT_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "rag_vocabulary.json")


@dataclass
class Vocabulary:
    themen: list[str] = field(default_factory=list)
    themen_aliases: dict[str, str] = field(default_factory=dict)
    laender: list[str] = field(default_factory=list)
    laender_aliases: dict[str, str] = field(default_factory=dict)
    normen: list[str] = field(default_factory=list)
    normen_aliases: dict[str, str] = field(default_factory=dict)


def _norm_key(value: str) -> str:
    return _WS.sub(" ", (value or "").strip().lower())


def load_vocabulary(path: str = DEFAULT_VOCAB_PATH) -> Vocabulary:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return Vocabulary(
        themen=data.get("themen", []),
        themen_aliases=data.get("themen_aliases", {}),
        laender=data.get("laender", []),
        laender_aliases=data.get("laender_aliases", {}),
        normen=data.get("normen", []),
        normen_aliases=data.get("normen_aliases", {}),
    )


def _normalize_list(canonical: list[str], aliases: dict[str, str], raw: list[str]) -> list[str]:
    """Map each raw term through aliases, keep only canonical members, dedup
    preserving first-seen order. Canonical membership is checked case-folded;
    the returned token is the canonical entry's own casing."""
    canon_by_key = {_norm_key(c): c for c in canonical}
    alias_by_key = {_norm_key(k): v for k, v in aliases.items()}
    out: list[str] = []
    seen: set[str] = set()
    for term in raw or []:
        key = _norm_key(term)
        if key in alias_by_key:
            key = _norm_key(alias_by_key[key])
        canon = canon_by_key.get(key)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def normalize_themen(vocab: Vocabulary, raw: list[str]) -> list[str]:
    return _normalize_list(vocab.themen, vocab.themen_aliases, raw)


def normalize_normen(vocab: Vocabulary, raw: list[str]) -> list[str]:
    return _normalize_list(vocab.normen, vocab.normen_aliases, raw)


def normalize_country(vocab: Vocabulary, raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    result = _normalize_list(vocab.laender, vocab.laender_aliases, [raw])
    return result[0] if result else None


def tag_line(themen: list[str], country: Optional[str], normen: list[str]) -> str:
    """Compact suffix appended to context_header so both hybrid channels index
    the tags. Empty string when there is nothing to add."""
    parts: list[str] = []
    if themen:
        parts.append("Schlagwörter: " + ", ".join(themen))
    if country:
        parts.append("Herkunftsland: " + country)
    if normen:
        parts.append("Normen: " + ", ".join(normen))
    return " | ".join(parts)


def facet_metadata(themen: list[str], country: Optional[str], normen: list[str]) -> dict:
    """Metadata facets aligned to the RAG API's existing RagFilters keys
    (applicant_origin, citations) plus schlagworte. Omits empty fields."""
    md: dict = {}
    if themen:
        md["schlagworte"] = themen
    if country:
        md["applicant_origin"] = country
    if normen:
        md["citations"] = normen
    return md
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_rag_vocabulary.py`
Expected: `ALL 8 PASSED`.

- [ ] **Step 5: Commit**

```bash
git add app/rag_vocabulary.py tests/test_rag_vocabulary.py
git commit -m "Add controlled-vocabulary normalizer for RAG tagging"
```

---

## Task 2: Build the controlled vocabulary from the asyl.net entries

**Files:**
- Create: `app/build_vocabulary.py`
- Create (generated): `app/rag_vocabulary.json`

- [ ] **Step 1: Write the aggregation script**

Create `app/build_vocabulary.py`:

```python
"""Aggregate the curated asyl.net tags already on RechtsprechungEntry rows into
the shared controlled vocabulary (app/rag_vocabulary.json).

themen  = schlagworte, frequency-ranked, singletons dropped (min_count).
laender = distinct country values (kept even if rare — countries are a closed set).
normen  = distinct normen, frequency-ranked, singletons dropped.

Alias maps start empty; curate them by hand afterwards (e.g. add
"wehrdienstverweigerung" -> "wehrdienstentziehung"). Re-running preserves any
hand-edited alias maps already present in the existing JSON.

Run: docker exec rechtmaschine-app python build_vocabulary.py [--min-count 2]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

from database import SessionLocal
from models import RechtsprechungEntry

_WS = re.compile(r"\s+")
OUT_PATH = os.path.join(os.path.dirname(__file__), "rag_vocabulary.json")


def _clean(value: str) -> str:
    return _WS.sub(" ", (value or "").strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=2,
                    help="Drop themen/normen seen fewer than this many times.")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        rows = db.query(
            RechtsprechungEntry.schlagworte,
            RechtsprechungEntry.normen,
            RechtsprechungEntry.country,
        ).all()
    finally:
        db.close()

    themen_counter: Counter[str] = Counter()
    normen_counter: Counter[str] = Counter()
    laender: dict[str, str] = {}  # lowercased -> display form (first seen)

    for schlagworte, normen, country in rows:
        for sw in (schlagworte or []):
            c = _clean(sw)
            if c:
                themen_counter[c.lower()] += 1
        for n in (normen or []):
            c = _clean(n)
            if c:
                normen_counter[c] += 1
        c = _clean(country or "")
        if c:
            laender.setdefault(c.lower(), c)

    themen = sorted([t for t, n in themen_counter.items() if n >= args.min_count])
    normen = sorted([t for t, n in normen_counter.items() if n >= args.min_count])
    laender_list = sorted(laender.values())

    # Preserve hand-curated alias maps across re-runs.
    existing = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as fh:
            existing = json.load(fh)

    data = {
        "themen": themen,
        "themen_aliases": existing.get("themen_aliases", {}),
        "laender": laender_list,
        "laender_aliases": existing.get("laender_aliases", {}),
        "normen": normen,
        "normen_aliases": existing.get("normen_aliases", {}),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=False)

    print(f"themen={len(themen)} laender={len(laender_list)} normen={len(normen)} "
          f"(from {len(rows)} entries, min_count={args.min_count})")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Generate the vocabulary**

Run: `docker exec rechtmaschine-app python build_vocabulary.py --min-count 2`
Expected: a line like `themen=120 laender=35 normen=80 (from 473 entries, min_count=2)` and `wrote /app/rag_vocabulary.json`.

- [ ] **Step 3: Sanity-check the JSON and load it through the normalizer**

Run:
```bash
docker exec rechtmaschine-app python -c "
from rag_vocabulary import load_vocabulary, normalize_themen, normalize_country
v = load_vocabulary()
print('themen', len(v.themen), 'laender', len(v.laender), 'normen', len(v.normen))
print('sample themen:', v.themen[:8])
print('sample laender:', v.laender[:8])
assert v.themen and v.laender and v.normen
print('Griechenland ->', normalize_country(v, 'Griechenland'))
"
```
Expected: non-empty counts and a sample of real German terms; `Griechenland -> Griechenland` (or `None` if the corpus spells it differently — note it for alias curation).

- [ ] **Step 4: Hand-curate obvious aliases (optional but recommended)**

Open `app/rag_vocabulary.json` and add a handful of high-value entries to `themen_aliases` / `laender_aliases` / `normen_aliases` for synonyms you can see in the generated lists (e.g. `"wehrdienstverweigerung": "wehrdienstentziehung"`). Keep it small; this is curation, not exhaustiveness. Re-running `build_vocabulary.py` preserves these.

- [ ] **Step 5: Commit**

```bash
git add app/build_vocabulary.py app/rag_vocabulary.json
git commit -m "Generate shared RAG controlled vocabulary from asyl.net entries"
```

---

## Task 3: Add the `/v1/rag/chunks/scroll` export endpoint

**Files:**
- Modify: `rag/api/main.py` (add request model near `RagDeleteRequest:73`; add endpoint near `delete_chunks:389`)

- [ ] **Step 1: Add the request model**

In `rag/api/main.py`, after the `RagDeleteRequest` class (ends at `:75`), add:

```python
class RagScrollRequest(BaseModel):
    collection: str = Field(min_length=1)
    cursor: Optional[str] = None          # last chunk_id from the previous page
    limit: int = Field(default=256, ge=1, le=512)
```

- [ ] **Step 2: Add the endpoint**

In `rag/api/main.py`, after `delete_chunks` (ends at `:412`), add:

```python
@app.post("/v1/rag/chunks/scroll")
def scroll_chunks(body: RagScrollRequest, x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    """Keyset-paginated export of a collection's chunks (chunk_id ascending), so
    the app-side retag tool can read text+metadata, attach tags, and re-upsert.
    next_cursor is null when the page is the last one."""
    _validate_api_key(x_api_key)
    cursor = body.cursor or ""
    with _db_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT chunk_id, text, context_header, metadata, provenance
            FROM rag_chunks
            WHERE collection = %s AND chunk_id > %s
            ORDER BY chunk_id
            LIMIT %s
            """,
            [body.collection, cursor, body.limit],
        )
        rows = list(cur.fetchall())
    next_cursor = rows[-1]["chunk_id"] if len(rows) == body.limit else None
    chunks = [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "context_header": r.get("context_header"),
            "metadata": r.get("metadata") or {},
            "provenance": r.get("provenance") or [],
        }
        for r in rows
    ]
    return {"chunks": chunks, "next_cursor": next_cursor, "count": len(chunks)}
```

- [ ] **Step 3: Deploy to debian and restart the RAG API**

On debian (the RAG stack host):
```bash
cd /var/opt/docker/rechtmaschine && git pull
docker compose -f rag/docker-compose.debian.yml restart rag-api
```
(If you are not on debian, do this over SSH on the debian host. Check `hostname` first per CLAUDE.md.)

- [ ] **Step 4: Verify pagination works and terminates**

From the app container (which has the API key + reach):
```bash
docker exec rechtmaschine-app python -c "
import os, httpx
base = os.environ['RAG_SERVICE_URL'].rstrip('/')
key = os.environ.get('RAG_SERVICE_API_KEY') or os.environ.get('RAG_API_KEY')
h = {'X-API-Key': key} if key else {}
seen, cursor, pages = 0, None, 0
while True:
    r = httpx.post(f'{base}/v1/rag/chunks/scroll',
                   json={'collection': 'jurisprudence', 'cursor': cursor, 'limit': 256},
                   headers=h, timeout=30)
    r.raise_for_status(); d = r.json()
    seen += d['count']; pages += 1; cursor = d['next_cursor']
    if cursor is None: break
print('jurisprudence chunks scrolled:', seen, 'pages:', pages)
assert seen > 0
"
```
Expected: a total close to the known ~10,257 jurisprudence chunks; loop terminates.

- [ ] **Step 5: Commit**

```bash
git add rag/api/main.py
git commit -m "Add /v1/rag/chunks/scroll keyset export endpoint"
```

---

## Task 4: Desktop-Qwen document tagger

**Files:**
- Create: `app/qwen_tagger.py`

- [ ] **Step 1: Write the tagger**

Create `app/qwen_tagger.py`:

```python
"""Tag a single (already-anonymized) document against the controlled vocabulary
using the desktop Qwen service. Returns normalized facets only — the model is
asked to choose from the vocab, and whatever it returns is re-normalized so
out-of-vocab hallucinations are dropped.

Anonymization invariant: callers pass anonymized text only. Controlled-vocab
tags are categories and cannot reintroduce PII.
"""
from __future__ import annotations

import os
from typing import Optional

from citation_qwen import call_qwen_json
from rag_vocabulary import (
    Vocabulary, normalize_themen, normalize_country, normalize_normen,
)

# Keep the prompt vocab bounded so it fits the context window comfortably.
_MAX_THEMEN_IN_PROMPT = 300
_MAX_DOC_CHARS = 12000


def _service_url() -> str:
    url = os.environ.get("ANONYMIZATION_SERVICE_URL", "").strip()
    if not url:
        raise RuntimeError("ANONYMIZATION_SERVICE_URL not set; cannot reach desktop Qwen")
    return url


def _build_prompt(vocab: Vocabulary, text: str) -> str:
    themen = vocab.themen[:_MAX_THEMEN_IN_PROMPT]
    laender = vocab.laender
    return (
        "Du bist ein juristischer Klassifikator für deutsches Asyl- und "
        "Aufenthaltsrecht. Wähle ausschließlich aus den vorgegebenen Listen. "
        "Erfinde keine neuen Begriffe.\n\n"
        f"ERLAUBTE SCHLAGWÖRTER:\n{', '.join(themen)}\n\n"
        f"ERLAUBTE HERKUNFTSLÄNDER:\n{', '.join(laender)}\n\n"
        "Gib NUR JSON zurück mit den Feldern: "
        '{"schlagworte": [..], "herkunftsland": "<eines oder null>", "normen": ["§ .. Gesetz", ..]}. '
        "schlagworte: die 3-8 treffendsten aus der Liste. herkunftsland: das "
        "betroffene Herkunftsland oder null. normen: die zentral einschlägigen "
        "Normen (z.B. \"§ 3 AsylG\", \"§ 60 Abs. 7 AufenthG\", \"Art. 3 EMRK\").\n\n"
        f"DOKUMENT (anonymisiert):\n{text[:_MAX_DOC_CHARS]}"
    )


async def tag_document(text: str, vocab: Vocabulary) -> dict:
    """Return {"schlagworte": [...], "herkunftsland": str|None, "normen": [...]},
    all normalized through the vocabulary. On any failure returns empty facets so
    a single bad document never aborts a retag run."""
    if not (text or "").strip():
        return {"schlagworte": [], "herkunftsland": None, "normen": []}
    try:
        parsed = await call_qwen_json(
            _service_url(), _build_prompt(vocab, text),
            num_predict=400, temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, don't abort the batch
        print(f"[tagger] qwen call failed: {exc}")
        return {"schlagworte": [], "herkunftsland": None, "normen": []}

    raw_themen = parsed.get("schlagworte") or []
    raw_country = parsed.get("herkunftsland")
    raw_normen = parsed.get("normen") or []
    if isinstance(raw_themen, str):
        raw_themen = [raw_themen]
    if isinstance(raw_normen, str):
        raw_normen = [raw_normen]
    return {
        "schlagworte": normalize_themen(vocab, raw_themen),
        "herkunftsland": normalize_country(vocab, raw_country),
        "normen": normalize_normen(vocab, raw_normen),
    }
```

- [ ] **Step 2: Live-verify against one document**

Pick any kanzlei chunk's text and tag it (requires desktop Qwen awake — the app reaches it the same way anonymization does):
```bash
docker exec rechtmaschine-app python -c "
import asyncio, os, httpx
from rag_vocabulary import load_vocabulary
from qwen_tagger import tag_document
base = os.environ['RAG_SERVICE_URL'].rstrip('/')
key = os.environ.get('RAG_SERVICE_API_KEY') or os.environ.get('RAG_API_KEY')
h = {'X-API-Key': key} if key else {}
r = httpx.post(f'{base}/v1/rag/chunks/scroll', json={'collection':'kanzlei','limit':3}, headers=h, timeout=30)
text = '\n\n'.join(c['text'] for c in r.json()['chunks'])
print(asyncio.run(tag_document(text, load_vocabulary())))
"
```
Expected: a dict with a few in-vocab `schlagworte`, a plausible `herkunftsland` or `None`, and some `normen`. If Qwen is asleep, wake it first (`docker exec rechtmaschine-app ssh osmc@osmc /usr/local/bin/wake-desktop`).

- [ ] **Step 3: Commit**

```bash
git add app/qwen_tagger.py
git commit -m "Add desktop-Qwen document tagger constrained to the controlled vocabulary"
```

---

## Task 5: Retag tool — jurisprudence subcommand

**Files:**
- Create: `app/retag_rag.py`

- [ ] **Step 1: Write the tool with the jurisprudence subcommand**

Create `app/retag_rag.py`:

```python
"""Uniform retag pass over an existing RAG collection: scroll chunks from the
store, attach controlled-vocabulary tags, and re-upsert by the same chunk_id
(sending empty dense so the API re-embeds with the new context_header). No
re-extraction, re-download, or re-anonymization.

Subcommands:
  jurisprudence  tags come from the chunk's RechtsprechungEntry (DB), normalized.
  kanzlei        tags come from one desktop-Qwen call per document (Task 6).

Run inside the app container, e.g.:
  docker exec rechtmaschine-app python retag_rag.py jurisprudence --dry-run
  docker exec rechtmaschine-app python retag_rag.py jurisprudence
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Optional

import httpx

from rag_vocabulary import (
    load_vocabulary, normalize_themen, normalize_country, normalize_normen,
    tag_line, facet_metadata,
)


def _rag_base() -> str:
    return os.environ["RAG_SERVICE_URL"].rstrip("/")


def _rag_headers() -> dict[str, str]:
    key = os.environ.get("RAG_SERVICE_API_KEY") or os.environ.get("RAG_API_KEY")
    return {"X-API-Key": key} if key else {}


def scroll_all(client: httpx.Client, collection: str, page: int = 256):
    cursor: Optional[str] = None
    while True:
        r = client.post(f"{_rag_base()}/v1/rag/chunks/scroll",
                        json={"collection": collection, "cursor": cursor, "limit": page},
                        headers=_rag_headers(), timeout=60)
        r.raise_for_status()
        data = r.json()
        for chunk in data["chunks"]:
            yield chunk
        cursor = data["next_cursor"]
        if cursor is None:
            break


def _base_header(context_header: Optional[str]) -> str:
    """Strip any previously-appended tag segments so reruns are idempotent."""
    if not context_header:
        return ""
    keep = []
    for seg in context_header.split(" | "):
        s = seg.strip()
        if s.startswith(("Schlagwörter:", "Herkunftsland:", "Normen:")):
            continue
        keep.append(s)
    return " | ".join([s for s in keep if s])


def build_retagged_chunk(chunk: dict[str, Any], themen, country, normen) -> dict[str, Any]:
    base = _base_header(chunk.get("context_header"))
    suffix = tag_line(themen, country, normen)
    header = " | ".join([s for s in (base, suffix) if s])
    metadata = {**(chunk.get("metadata") or {}), **facet_metadata(themen, country, normen)}
    return {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "context_header": header,
        "metadata": metadata,
        "provenance": chunk.get("provenance") or [],
        "dense": [],  # force re-embed with the new context_header
    }


def upsert_batch(client: httpx.Client, collection: str, batch: list[dict[str, Any]]) -> int:
    if not batch:
        return 0
    r = client.post(f"{_rag_base()}/v1/rag/chunks/upsert",
                    json={"collection": collection, "chunks": batch},
                    headers=_rag_headers(), timeout=180)
    r.raise_for_status()
    return int(r.json().get("upserted", 0))


def run_jurisprudence(args) -> int:
    from database import SessionLocal
    from models import RechtsprechungEntry

    vocab = load_vocabulary()
    db = SessionLocal()
    # Cache the entry's raw fields by id so we hit the DB once per entry, not
    # once per chunk.
    cache: dict[str, dict] = {}

    def raw_for_entry(entry_id: str) -> dict:
        if entry_id in cache:
            return cache[entry_id]
        e = db.query(RechtsprechungEntry).filter(RechtsprechungEntry.id == entry_id).first()
        cache[entry_id] = {
            "schlagworte": (e.schlagworte or []) if e else [],
            "country": (e.country if e else None),
            "normen": (e.normen or []) if e else [],
        }
        return cache[entry_id]

    upserted = scanned = skipped = 0
    batch: list[dict[str, Any]] = []
    try:
        with httpx.Client() as client:
            for chunk in scroll_all(client, "jurisprudence"):
                scanned += 1
                md = chunk.get("metadata") or {}
                entry_id = md.get("rechtsprechung_entry_id")
                if not entry_id:
                    skipped += 1
                    continue
                raw = raw_for_entry(str(entry_id))
                # Fold the chunk's own Gemini issue_tags (free-text) in with the
                # curated asyl.net schlagworte; the normalizer keeps only in-vocab.
                themen = normalize_themen(vocab, raw["schlagworte"] + (md.get("issue_tags") or []))
                country = normalize_country(vocab, raw["country"])
                normen = normalize_normen(vocab, raw["normen"])
                new_chunk = build_retagged_chunk(chunk, themen, country, normen)
                if args.dry_run:
                    if scanned <= 5:
                        print(f"  {chunk['chunk_id']}: {new_chunk['metadata'].get('schlagworte')} "
                              f"/ {new_chunk['metadata'].get('applicant_origin')}")
                    continue
                batch.append(new_chunk)
                if len(batch) >= 16:
                    upserted += upsert_batch(client, "jurisprudence", batch)
                    batch = []
            if not args.dry_run:
                upserted += upsert_batch(client, "jurisprudence", batch)
    finally:
        db.close()
    print(f"jurisprudence: scanned={scanned} skipped(no entry)={skipped} re-upserted={upserted} "
          f"{'(dry-run)' if args.dry_run else ''}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    jp = sub.add_parser("jurisprudence")
    jp.add_argument("--dry-run", action="store_true")
    kp = sub.add_parser("kanzlei")
    kp.add_argument("--dry-run", action="store_true")
    kp.add_argument("--limit-docs", type=int, default=0, help="Tag at most N docs (0 = all).")
    args = ap.parse_args()
    if args.cmd == "jurisprudence":
        return run_jurisprudence(args)
    if args.cmd == "kanzlei":
        return asyncio.run(run_kanzlei(args))  # defined in Task 6
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `run_kanzlei` is added in Task 6. The `kanzlei` subcommand will raise `NameError` until then — that is expected and fine; the jurisprudence path is independent.

- [ ] **Step 2: Dry-run the jurisprudence retag**

Run: `docker exec rechtmaschine-app python retag_rag.py jurisprudence --dry-run`
Expected: prints `schlagworte`/`applicant_origin` for the first 5 chunks and a summary line `scanned=~10257 skipped(no entry)=0 re-upserted=0 (dry-run)`. If many are skipped, the chunks lack `rechtsprechung_entry_id` — stop and investigate before the real run.

- [ ] **Step 3: Run the jurisprudence retag for real**

Run: `docker exec rechtmaschine-app python retag_rag.py jurisprudence`
Expected: `re-upserted=~10257`. This re-embeds each chunk through TEI on debian, so it takes a while (minutes to tens of minutes).

- [ ] **Step 4: Verify tags landed and retrieval still works**

```bash
docker exec rechtmaschine-app python -c "
import os, httpx
base=os.environ['RAG_SERVICE_URL'].rstrip('/'); key=os.environ.get('RAG_SERVICE_API_KEY') or os.environ.get('RAG_API_KEY')
h={'X-API-Key':key} if key else {}
r=httpx.post(f'{base}/v1/rag/retrieve', json={'query':'Dublin Griechenland Aufnahmebedingungen','collection':'jurisprudence','limit':5,'use_reranker':True}, headers=h, timeout=60)
for c in r.json()['chunks']:
    md=c.get('metadata') or {}
    print(round(c['score'],3), md.get('applicant_origin'), md.get('schlagworte'))
"
```
Expected: results come back (retrieval intact) and now carry `applicant_origin` + `schlagworte` in metadata, and the tags appear in `context_header`.

- [ ] **Step 5: Commit**

```bash
git add app/retag_rag.py
git commit -m "Add retag tool with jurisprudence subcommand (entry tags -> chunk header+metadata)"
```

---

## Task 6: Retag tool — kanzlei subcommand (Qwen per document)

**Files:**
- Modify: `app/retag_rag.py` (add `run_kanzlei`; add the import for `tag_document`)

- [ ] **Step 1: Add the kanzlei import**

In `app/retag_rag.py`, add near the top imports:

```python
from qwen_tagger import tag_document
```

- [ ] **Step 2: Add `run_kanzlei`**

In `app/retag_rag.py`, add above `def main()`:

```python
def _doc_id(chunk: dict[str, Any]) -> str:
    """Group key for a kanzlei document: the sha16 in chunk_id 'nc-<sha16>-<idx>'."""
    cid = chunk["chunk_id"]
    parts = cid.split("-")
    return parts[1] if len(parts) >= 3 and parts[0] == "nc" else cid


async def run_kanzlei(args) -> int:
    vocab = load_vocabulary()
    # Group all chunks by document first (scroll is cheap; tagging is the cost).
    docs: dict[str, list[dict[str, Any]]] = {}
    with httpx.Client() as client:
        for chunk in scroll_all(client, "kanzlei"):
            docs.setdefault(_doc_id(chunk), []).append(chunk)

    doc_ids = sorted(docs)
    if args.limit_docs:
        doc_ids = doc_ids[: args.limit_docs]
    print(f"kanzlei: {len(docs)} documents, tagging {len(doc_ids)}")

    tagged_docs = upserted = 0
    with httpx.Client() as client:
        for n, did in enumerate(doc_ids, 1):
            chunks = sorted(docs[did], key=lambda c: (c.get("metadata") or {}).get("chunk_index", 0))
            text = "\n\n".join(c["text"] for c in chunks)
            facets = await tag_document(text, vocab)
            themen, country, normen = facets["schlagworte"], facets["herkunftsland"], facets["normen"]
            if args.dry_run:
                if n <= 5:
                    print(f"  {did}: {themen} / {country} / {normen}")
                continue
            batch = [build_retagged_chunk(c, themen, country, normen) for c in chunks]
            for start in range(0, len(batch), 16):
                upserted += upsert_batch(client, "kanzlei", batch[start:start + 16])
            tagged_docs += 1
            if n % 50 == 0:
                print(f"  ... {n}/{len(doc_ids)} docs, {upserted} chunks re-upserted")
    print(f"kanzlei: tagged_docs={tagged_docs} re-upserted={upserted} "
          f"{'(dry-run)' if args.dry_run else ''}")
    return 0
```

- [ ] **Step 3: Dry-run on a few documents**

Run: `docker exec rechtmaschine-app python retag_rag.py kanzlei --dry-run --limit-docs 5`
Expected: 5 lines of `sha16: [themen] / country / [normen]` with in-vocab terms. (Desktop Qwen must be awake.)

- [ ] **Step 4: Real run on a small slice + leak scan**

Run a 20-doc slice first:
```bash
docker exec rechtmaschine-app python retag_rag.py kanzlei --limit-docs 20
```
Then confirm the new headers contain only category tags, no PII, by scrolling and eyeballing the tag segment:
```bash
docker exec rechtmaschine-app python -c "
import os, httpx, re
base=os.environ['RAG_SERVICE_URL'].rstrip('/'); key=os.environ.get('RAG_SERVICE_API_KEY') or os.environ.get('RAG_API_KEY')
h={'X-API-Key':key} if key else {}
r=httpx.post(f'{base}/v1/rag/chunks/scroll', json={'collection':'kanzlei','limit':50}, headers=h, timeout=60)
for c in r.json()['chunks'][:15]:
    ch=c.get('context_header') or ''
    seg=[s for s in ch.split(' | ') if s.startswith(('Schlagwörter:','Herkunftsland:','Normen:'))]
    if seg: print(seg)
"
```
Expected: only controlled category terms appear (no names/addresses). If anything looks like PII, stop — the vocab is the only allowed output, so this would indicate a normalizer bug.

- [ ] **Step 5: Full kanzlei run**

Run: `docker exec rechtmaschine-app python retag_rag.py kanzlei`
Expected: `tagged_docs=~3000 re-upserted=~10660`. This is the long job (~3,000 Qwen calls + re-embedding); run it when desktop Qwen and debian TEI are free. Re-running is safe (idempotent).

- [ ] **Step 6: Commit**

```bash
git add app/retag_rag.py
git commit -m "Add kanzlei retag subcommand (per-document Qwen tagging)"
```

---

## Task 7: Bake tag-writing into the live ingest paths

**Files:**
- Modify: `app/jurisprudence_ingest.py` (`context_header` at `:386`, `metadata` at `:400`)
- Modify: `rag/ingest_runner.py` (`context_header` at `:455`, `metadata` at `:457`)

- [ ] **Step 1: Jurisprudence ingest — normalize asyl.net tags into header + metadata**

In `app/jurisprudence_ingest.py`, add to the imports near the top:

```python
from rag_vocabulary import (
    load_vocabulary, normalize_themen, normalize_country, normalize_normen,
    tag_line, facet_metadata,
)
```

Then replace the `context_header` construction (currently `:384-386`):

```python
                tags = extract_tags(text)
                header_bits = ["Rechtsprechung", tags.court or "", tags.court_level or "",
                               tags.decision_date or "", tags.country or ""]
                context_header = " | ".join(b for b in header_bits if b)
```

with:

```python
                tags = extract_tags(text)
                _vocab = load_vocabulary()
                _themen = normalize_themen(_vocab, (r.get("schlagworte") or []) + (tags.tags or []))
                _country = normalize_country(_vocab, tags.country)
                _normen = normalize_normen(_vocab, r.get("normen") or [])
                header_bits = ["Rechtsprechung", tags.court or "", tags.court_level or "",
                               tags.decision_date or "", tags.country or "",
                               tag_line(_themen, _country, _normen)]
                context_header = " | ".join(b for b in header_bits if b)
```

Then, in the `metadata` dict (currently `:400-412`), add the facet keys by appending after the existing `issue_tags` line:

```python
                    "issue_tags": tags.tags or [],
                    **facet_metadata(_themen, _country, _normen),
```

- [ ] **Step 2: Kanzlei ingest — Qwen-tag each document at ingest**

In `rag/ingest_runner.py`, add near the top imports (after the existing `import` block):

```python
import asyncio
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "app"))
from rag_vocabulary import load_vocabulary, tag_line, facet_metadata  # noqa: E402
from qwen_tagger import tag_document  # noqa: E402
```

Load the vocab once near the start of `main()` (after args are parsed):

```python
    _vocab = load_vocabulary()
```

Then replace the `context_header`/`metadata` construction (currently `:450-465`) so the anonymized document text is tagged and the tags are written in:

```python
                header_bits = ["Kanzlei-Schriftsatz", role]
                if court:
                    header_bits.append(court)
                if date:
                    header_bits.append(date)

                _facets = asyncio.run(tag_document(anonymized, _vocab))
                _themen = _facets["schlagworte"]
                _country = _facets["herkunftsland"]
                _normen = _facets["normen"]
                _tags = tag_line(_themen, _country, _normen)
                if _tags:
                    header_bits.append(_tags)
                context_header = " | ".join(header_bits)

                metadata = {
                    "source_system": "nextcloud",
                    "document_role": role,
                    "court": court,
                    "case_hash": case_hash,
                    "document_date": date,
                    "extension": item["extension"],
                    "language": "de",
                    **facet_metadata(_themen, _country, _normen),
                }
```

(`tag_document` is async; `ingest_runner.py` is synchronous, so each call is wrapped in `asyncio.run`. The tagger degrades to empty facets on failure, so ingest never breaks if Qwen is briefly unavailable.)

- [ ] **Step 3: Verify the live ingest paths import cleanly**

```bash
docker exec rechtmaschine-app python -c "import jurisprudence_ingest; print('juris ingest import OK')"
python -c "import ast; ast.parse(open('rag/ingest_runner.py').read()); print('ingest_runner parses OK')"
```
Expected: both print OK. (Full `rag/ingest_runner.py` import needs `fitz` etc., so a syntax parse check is sufficient here; it is exercised for real on the import host.)

- [ ] **Step 4: Restart the app + worker so the new jurisprudence ingest code is loaded**

```bash
docker compose restart app job-worker
```

- [ ] **Step 5: Commit**

```bash
git add app/jurisprudence_ingest.py rag/ingest_runner.py
git commit -m "Write controlled-vocab tags into context_header+metadata at ingest time"
```

---

## Task 8: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm both collections now carry facets**

```bash
docker exec rechtmaschine-app python -c "
import os, httpx
base=os.environ['RAG_SERVICE_URL'].rstrip('/'); key=os.environ.get('RAG_SERVICE_API_KEY') or os.environ.get('RAG_API_KEY')
h={'X-API-Key':key} if key else {}
for coll in ('jurisprudence','kanzlei'):
    r=httpx.post(f'{base}/v1/rag/chunks/scroll', json={'collection':coll,'limit':200}, headers=h, timeout=60)
    cs=r.json()['chunks']
    with_sw=sum(1 for c in cs if (c.get('metadata') or {}).get('schlagworte'))
    print(coll, 'sample', len(cs), 'with schlagworte:', with_sw)
"
```
Expected: a healthy fraction of sampled chunks carry `schlagworte` in both collections.

- [ ] **Step 2: Re-run the retrieval benchmark and compare**

Run: `docker exec rechtmaschine-app python /tmp/rag_benchmark2.py` (the existing benchmark; copy it into the container first if needed with `docker cp /tmp/rag_benchmark2.py rechtmaschine-app:/tmp/`).
Expected: mean P@5 and rerank-lift at least as good as the pre-tagging baseline; ideally higher on the topical queries (Dublin/country/§ queries). Record the numbers in the handoff.

- [ ] **Step 3: Confirm generation/query still degrade-safely**

Spot-check that a normal query against a case still returns (the consumer code is unchanged, but verify nothing regressed):
```bash
docker exec rechtmaschine-app python tests/test_citations.py 2>/dev/null || echo "skip if it needs live services"
```
Expected: no import/runtime errors from the RAG path.

- [ ] **Step 4: Update the handoff**

Append a dated entry to `.remember/remember.md` summarizing: vocab built (counts), scroll endpoint added, both collections retagged (chunk counts), live ingest baked, benchmark numbers, and the deferred items (RagFilters.schlagworte + query-side facet extraction).

- [ ] **Step 5: Commit**

```bash
git add .remember/remember.md
git commit -m "Handoff: controlled-vocab tagging live across RAG collections"
```

---

## Self-review notes (for the implementer)

- **Idempotency:** `_base_header` strips previously-appended tag segments before re-appending, so retag reruns and live re-ingests do not stack duplicate `Schlagwörter:` segments. Re-upsert is by `chunk_id` with `ON CONFLICT DO UPDATE`.
- **Re-embedding cost:** every retag sends `dense: []`, so ~21k chunks re-embed through TEI bge-m3 on debian. Run the big kanzlei pass when the GPU is free.
- **Anonymization invariant:** kanzlei tagging only ever sees already-anonymized chunk text (reconstructed from the store), via desktop Qwen; output is restricted to the controlled vocabulary, which cannot carry PII.
- **Graceful degradation:** `tag_document` returns empty facets on any Qwen failure, so neither the retag batch nor live ingest aborts on a single bad document.
- **`issue_tags` source:** `RechtsprechungEntry` has no `issue_tags` column — those live on the *chunk* metadata (from the Gemini extraction at ingest). The jurisprudence retag reads `schlagworte/country/normen` from the entry and folds the chunk's own `metadata.issue_tags` in as extra free-text; the normalizer discards anything out-of-vocab.
