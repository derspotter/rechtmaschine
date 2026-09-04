# Case Assessment Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third case-memory target `case_assessment` that holds one structured legal assessment (Gutachten) per legal question, with citations that are checked against the firm's Rechtsprechung store, injected into generation prompts as its own block, and used as the primary source for `wiki distill`.

**Architecture:** A new domain module `app/assessment_memory.py` owns everything specific to assessments: Pydantic content models, id-based patch-op resolution, the store reconciliation, the prompt renderer and the rebase strategy. The generic memory layer in `app/agent_memory_service.py` keeps persistence, revisions, versions and the proposal lifecycle, and reaches the new behaviour through a named `TargetSpec` registry that replaces today's positional seven-tuple.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2 with Postgres JSONB, Pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-case-assessment-memory-design.md` (Fassung 2, plus the Codex review at `docs/superpowers/specs/2026-09-04-case-assessment-codex-review.md`)

## Global Constraints

- All memory code lives under `/var/opt/docker/rechtmaschine`. Run tests on the host with `.venv/bin/python -m pytest tests/ -q -m "not slow"`, never inside Docker.
- German user-facing strings, no semicolons in any German prose you add (Kanzlei rule).
- `store`, `store_entry_id` and `store_checked_at` are server-owned fields. Client-supplied values are stripped on proposal create and rewritten on accept and recheck.
- `verified` means exactly: an active `RechtsprechungEntry` with the same normalized Aktenzeichen exists and its `decision_date` equals the Fundstelle's `datum`. It does not verify court or holding. Never label it "geprüft" in user-facing text.
- Az normalization everywhere uses `jurisprudence_ingest._az_for_compare` re-exported as a public function. Never introduce a second normalizer.
- Assessment content size limits: a single Gutachten max 24 kB serialized JSON, whole `content_json` max 200 kB, field limits per the spec table.
- Patch ops address Gutachten by `id` (`/gutachten/by-id/<id>`), never by list index.
- Reflect (`documents`, `jlawyer`, `consolidate`, `pattern_wiki` extraction into brief/strategy) must not touch `case_assessment`.
- Existing tests that stub `models` and `shared` must be extended, not bypassed, when a new ORM class is referenced at import time.
- Work in a git worktree off `master`. Commit after every task. Do not commit unrelated working-tree changes from parallel sessions (`app/static/js/app.js`, `docker-compose.yml`, `scripts/rechtmaschine_cli.py` and the two memory modules may carry other people's edits — check `git status` before each `git add` and stage only the files the task names).

## File Structure

| File | Responsibility |
|---|---|
| `app/assessment_memory.py` (new) | Pydantic models, validation, size limits, id-based op resolution, store reconciliation, prompt rendering, rebase strategy, recheck. All assessment-specific logic. |
| `app/models.py` (modify) | `CaseAssessment` and `CaseAssessmentSource` ORM classes. |
| `app/main.py` (modify) | Migration `2026-09-03_case_assessments`. |
| `app/shared.py` (modify) | `MemoryTargetType` literal, `CaseAssessmentResponse`. |
| `app/agent_memory_service.py` (modify) | `TargetSpec` NamedTuple registry, delegation hooks for validate/apply/rebase, assessment block in `get_case_memory_prompt_context`, accept returns warnings. |
| `app/endpoints/agent_memory.py` (modify) | Target allowlist on the create route, `_combined_payload` third block, `_proposal_frontend_payload` assessment section, accept route surfaces warnings, recheck route. |
| `app/citation_verifier.py` (modify) | Blocked-Az warning in `verify_facts`. |
| `app/endpoints/generation.py` (modify) | Pass blocked Az from grounding into the fact check. |
| `app/draft_citation_ingest.py` (modify) | `iter_decision_citations` yielding spans. |
| `app/endpoints/pattern_wiki.py` (modify) | Assessment-first FALL-SPEICHER, Az whitelist, provenance. |
| `scripts/rechtmaschine_cli.py` (modify) | `assessment` section, grep flattening, proposal list rendering, `memory assessment recheck`. |
| `~/kanzlei/skills/rechtmaschine/scripts/memory_triage.py` (modify) | SESSION verdict for assessment proposals. |
| `~/kanzlei/skills/rechtmaschine/scripts/memory_hygiene_hook.py` (modify) | Vermerk-without-Gutachten reminder. |
| `~/kanzlei/skills/api/scripts/jlawyer_cli.py` (modify) | Emit `vermerk-upload` activity event. |
| `~/kanzlei/skills/rechtmaschine-memory/SKILL.md` (modify) | Gutachten section. |
| `tests/test_memory_assessment_*.py` (new) | One file per concern, mirroring the existing stub-based ad-hoc style. |

---

### Task 1: Assessment content models and size limits

**Files:**
- Create: `app/assessment_memory.py`
- Test: `tests/test_memory_assessment_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Fundstelle`, `PruefungsPunkt`, `GutachtenEntry`, `CaseAssessmentContent` (Pydantic v2 models, all `extra="forbid"`); `ASSESSMENT_LIST_FIELDS = {"gutachten"}`; `ASSESSMENT_SCALAR_FIELDS = {"notizen"}`; `SERVER_FIELDS = ("store", "store_entry_id", "store_checked_at")`; `default_case_assessment_json() -> dict`; `validate_assessment_content(content: dict) -> dict`; `strip_server_fields(value: Any) -> Any`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_assessment_model.py`:

```python
"""Pure-logic tests for the case_assessment content models.

No DB, no containers. Run from the repo root:

    .venv/bin/python -m pytest tests/test_memory_assessment_model.py -q
"""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    default_case_assessment_json,
    strip_server_fields,
    validate_assessment_content,
)


def _gutachten(**overrides):
    base = {
        "id": "gueb-statt-duldung",
        "rechtsfrage": "Darf die ABH statt einer Duldung nur eine GUEB ausstellen?",
        "ergebnis": "Nein, wenn der Abschiebungszeitpunkt ungewiss ist.",
        "stand": "2026-09-03",
        "status": "aktiv",
        "pruefung": [
            {
                "these": "Kein Raum fuer ungeregelten Aufenthalt",
                "bewertung": "Traegt, solange die Behoerde keine Prognose belegt.",
                "fundstellen": ["18 E 491/12"],
            }
        ],
        "fundstellen": [
            {
                "gericht": "OVG NRW",
                "datum": "2012-06-18",
                "az": "18 E 491/12",
                "art": "Beschluss",
                "aussage": "Die GUEB ersetzt die Duldung nicht.",
                "richtung": "pro",
            }
        ],
        "risiken": ["Behoerde belegt eine zeitnahe Abschiebung."],
        "quelle": "Vermerk_2026-09-02_Recherche.pdf",
    }
    base.update(overrides)
    return base


def test_default_is_empty_and_valid():
    assert default_case_assessment_json() == {"gutachten": [], "notizen": ""}


def test_valid_content_round_trips():
    content = validate_assessment_content({"gutachten": [_gutachten()], "notizen": ""})
    assert content["gutachten"][0]["id"] == "gueb-statt-duldung"
    # server fields are materialized with their default
    assert content["gutachten"][0]["fundstellen"][0]["store"] == "unchecked"


def test_unknown_field_at_any_level_is_rejected():
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [_gutachten(unbekannt="x")], "notizen": ""})
    bad = _gutachten()
    bad["fundstellen"][0]["unbekannt"] = "x"
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [bad], "notizen": ""})


def test_invalid_slug_is_rejected():
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [_gutachten(id="GUEB Statt")], "notizen": ""})


def test_duplicate_id_is_rejected():
    with pytest.raises(ValueError):
        validate_assessment_content(
            {"gutachten": [_gutachten(), _gutachten()], "notizen": ""}
        )


def test_pruefung_reference_to_unknown_az_is_rejected():
    bad = _gutachten()
    bad["pruefung"][0]["fundstellen"] = ["9 X 1/99"]
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [bad], "notizen": ""})


def test_oversized_gutachten_is_rejected():
    huge = _gutachten(risiken=["x" * 300 for _ in range(10)])
    huge["pruefung"] = [
        {"these": "t" * 300, "bewertung": "b" * 800, "fundstellen": ["18 E 491/12"]}
        for _ in range(12)
    ]
    huge["fundstellen"] = [
        {
            "gericht": "OVG NRW",
            "datum": "2012-06-18",
            "az": "18 E 491/12",
            "aussage": "a" * 400,
            "richtung": "pro",
        }
        for _ in range(25)
    ]
    # still within per-field limits but over the 24 kB per-Gutachten cap
    with pytest.raises(ValueError):
        validate_assessment_content({"gutachten": [huge], "notizen": ""})


def test_strip_server_fields_removes_only_server_owned_keys():
    value = _gutachten()
    value["fundstellen"][0]["store"] = "verified"
    value["fundstellen"][0]["store_entry_id"] = "abc"
    cleaned = strip_server_fields(value)
    assert "store" not in cleaned["fundstellen"][0]
    assert cleaned["fundstellen"][0]["az"] == "18 E 491/12"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'assessment_memory'`

- [ ] **Step 3: Write minimal implementation**

Create `app/assessment_memory.py`:

```python
"""Domain logic for the case_assessment memory target (Gutachten).

Everything specific to legal assessments lives here: content models,
id-addressed patch ops, store reconciliation, prompt rendering and the
rebase strategy. The generic memory layer (agent_memory_service) keeps
persistence, revisions, versions and the proposal lifecycle.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ASSESSMENT_TARGET = "case_assessment"
ASSESSMENT_LIST_FIELDS = {"gutachten"}
ASSESSMENT_SCALAR_FIELDS = {"notizen"}
SERVER_FIELDS = ("store", "store_entry_id", "store_checked_at")

MAX_GUTACHTEN_BYTES = 24_000
MAX_CONTENT_BYTES = 200_000

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

StoreState = Literal["verified", "date_mismatch", "not_in_store", "unchecked"]


class Fundstelle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gericht: str = Field(min_length=1, max_length=120)
    datum: str
    az: str = Field(min_length=1, max_length=80)
    art: Optional[Literal["Urteil", "Beschluss"]] = None
    aussage: str = Field(default="", max_length=400)
    richtung: Literal["pro", "contra", "neutral"] = "neutral"
    store: StoreState = "unchecked"
    store_entry_id: Optional[str] = None
    store_checked_at: Optional[str] = None

    @field_validator("datum")
    @classmethod
    def _iso(cls, value: str) -> str:
        if not _ISO_DATE_RE.match(value or ""):
            raise ValueError("datum muss ISO-Format JJJJ-MM-TT haben")
        return value


class PruefungsPunkt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    these: str = Field(min_length=1, max_length=300)
    bewertung: str = Field(default="", max_length=800)
    fundstellen: List[str] = Field(default_factory=list)


class GutachtenEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    rechtsfrage: str = Field(min_length=1, max_length=300)
    ergebnis: str = Field(min_length=1, max_length=800)
    stand: str
    status: Literal["aktiv", "ueberholt"] = "aktiv"
    pruefung: List[PruefungsPunkt] = Field(default_factory=list, max_length=12)
    fundstellen: List[Fundstelle] = Field(default_factory=list, max_length=25)
    risiken: List[str] = Field(default_factory=list, max_length=10)
    quelle: str = Field(default="", max_length=200)

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG_RE.match(value or ""):
            raise ValueError(
                "id muss ein Slug aus Kleinbuchstaben, Ziffern und Bindestrichen sein"
            )
        return value

    @field_validator("stand")
    @classmethod
    def _iso(cls, value: str) -> str:
        if not _ISO_DATE_RE.match(value or ""):
            raise ValueError("stand muss ISO-Format JJJJ-MM-TT haben")
        return value

    @field_validator("risiken")
    @classmethod
    def _risk_len(cls, value: List[str]) -> List[str]:
        for item in value:
            if len(item) > 300:
                raise ValueError("Risiko-Eintrag ist zu lang (max 300 Zeichen)")
        return value


class CaseAssessmentContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gutachten: List[GutachtenEntry] = Field(default_factory=list)
    notizen: str = ""


def default_case_assessment_json() -> Dict[str, Any]:
    return CaseAssessmentContent().model_dump(mode="json")


def strip_server_fields(value: Any) -> Any:
    """Recursively drop server-owned Fundstelle fields from a client value."""
    if isinstance(value, dict):
        return {
            key: strip_server_fields(item)
            for key, item in value.items()
            if key not in SERVER_FIELDS
        }
    if isinstance(value, list):
        return [strip_server_fields(item) for item in value]
    return value


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def validate_assessment_content(content: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize assessment content. Raises ValueError on any
    schema, uniqueness, cross-reference or size violation."""
    try:
        model = CaseAssessmentContent(**(content or {}))
    except Exception as exc:  # pydantic ValidationError
        raise ValueError(f"Ungueltiger Gutachten-Inhalt: {exc}") from exc

    dumped = model.model_dump(mode="json")

    seen: set = set()
    for entry in dumped["gutachten"]:
        if entry["id"] in seen:
            raise ValueError(f"Doppelte Gutachten-id: {entry['id']}")
        seen.add(entry["id"])

        known_az = {f["az"] for f in entry["fundstellen"]}
        for punkt in entry["pruefung"]:
            for az in punkt["fundstellen"]:
                if az not in known_az:
                    raise ValueError(
                        f"Pruefungspunkt verweist auf unbekanntes Az: {az}"
                    )

        if _json_size(entry) > MAX_GUTACHTEN_BYTES:
            raise ValueError(
                f"Gutachten {entry['id']} ueberschreitet {MAX_GUTACHTEN_BYTES} Bytes"
            )

    if _json_size(dumped) > MAX_CONTENT_BYTES:
        raise ValueError(f"Gutachten-Inhalt ueberschreitet {MAX_CONTENT_BYTES} Bytes")

    return dumped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_model.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add app/assessment_memory.py tests/test_memory_assessment_model.py
git commit -m "feat(memory): Gutachten-Inhaltsmodelle mit Groessenlimits

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 2: Id-addressed patch ops

**Files:**
- Modify: `app/assessment_memory.py`
- Test: `tests/test_memory_assessment_ops.py`

**Interfaces:**
- Consumes: `validate_assessment_content`, `strip_server_fields` from Task 1.
- Produces: `apply_assessment_ops(content: dict, ops: list[dict]) -> dict` and `sanitize_assessment_ops(ops: list[dict]) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_assessment_ops.py`:

```python
"""Id-addressed patch ops for case_assessment.

    .venv/bin/python -m pytest tests/test_memory_assessment_ops.py -q
"""

import copy
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    apply_assessment_ops,
    sanitize_assessment_ops,
    validate_assessment_content,
)


def _entry(gid="a-frage"):
    return {
        "id": gid,
        "rechtsfrage": "Frage?",
        "ergebnis": "Antwort.",
        "stand": "2026-09-03",
        "fundstellen": [
            {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"}
        ],
    }


def _content(*entries):
    return validate_assessment_content({"gutachten": list(entries), "notizen": ""})


def test_append_adds_entry():
    out = apply_assessment_ops(
        _content(), [{"op": "append", "path": "/gutachten/-", "value": _entry()}]
    )
    assert [g["id"] for g in out["gutachten"]] == ["a-frage"]


def test_append_with_existing_id_is_rejected():
    with pytest.raises(ValueError):
        apply_assessment_ops(
            _content(_entry()),
            [{"op": "append", "path": "/gutachten/-", "value": _entry()}],
        )


def test_set_by_id_replaces_the_right_entry():
    content = _content(_entry("a-frage"), _entry("b-frage"))
    updated = _entry("b-frage")
    updated["ergebnis"] = "Neue Antwort."
    out = apply_assessment_ops(
        content,
        [{"op": "set", "path": "/gutachten/by-id/b-frage", "value": updated}],
    )
    assert out["gutachten"][0]["ergebnis"] == "Antwort."
    assert out["gutachten"][1]["ergebnis"] == "Neue Antwort."


def test_set_by_id_with_mismatched_value_id_is_rejected():
    with pytest.raises(ValueError):
        apply_assessment_ops(
            _content(_entry("a-frage")),
            [{"op": "set", "path": "/gutachten/by-id/a-frage", "value": _entry("b-frage")}],
        )


def test_set_by_id_unknown_is_rejected():
    with pytest.raises(ValueError):
        apply_assessment_ops(
            _content(_entry()),
            [{"op": "set", "path": "/gutachten/by-id/zzz", "value": _entry("zzz")}],
        )


def test_remove_by_id():
    content = _content(_entry("a-frage"), _entry("b-frage"))
    out = apply_assessment_ops(
        content, [{"op": "remove", "path": "/gutachten/by-id/a-frage"}]
    )
    assert [g["id"] for g in out["gutachten"]] == ["b-frage"]


def test_index_path_is_rejected():
    with pytest.raises(ValueError):
        apply_assessment_ops(
            _content(_entry()),
            [{"op": "set", "path": "/gutachten/0", "value": _entry()}],
        )


def test_scalar_set_on_notizen():
    out = apply_assessment_ops(_content(), [{"op": "set", "path": "/notizen", "value": "x"}])
    assert out["notizen"] == "x"


def test_set_by_id_resets_server_fields():
    content = _content(_entry())
    content["gutachten"][0]["fundstellen"][0]["store"] = "verified"
    updated = _entry()
    out = apply_assessment_ops(
        content, [{"op": "set", "path": "/gutachten/by-id/a-frage", "value": updated}]
    )
    assert out["gutachten"][0]["fundstellen"][0]["store"] == "unchecked"


def test_sanitize_strips_client_supplied_store_values():
    ops = [
        {
            "op": "append",
            "path": "/gutachten/-",
            "value": {
                **_entry(),
                "fundstellen": [
                    {
                        "gericht": "OVG NRW",
                        "datum": "2012-06-18",
                        "az": "18 E 491/12",
                        "store": "verified",
                    }
                ],
            },
        }
    ]
    original = copy.deepcopy(ops)
    cleaned = sanitize_assessment_ops(ops)
    assert "store" not in cleaned[0]["value"]["fundstellen"][0]
    assert ops == original, "sanitize must not mutate its input"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_ops.py -q`
Expected: FAIL with `ImportError: cannot import name 'apply_assessment_ops'`

- [ ] **Step 3: Write minimal implementation**

Append to `app/assessment_memory.py`:

```python
_BY_ID_PREFIX = "by-id"


def sanitize_assessment_ops(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of ops with server-owned Fundstelle fields removed.

    Proposals are stored verbatim, so the stripping has to happen before the
    ops are persisted -- otherwise a client-supplied store="verified" would
    survive into the accepted content."""
    cleaned: List[Dict[str, Any]] = []
    for op in ops:
        item = dict(op)
        if "value" in item:
            item["value"] = strip_server_fields(item["value"])
        cleaned.append(item)
    return cleaned


def _parse_assessment_path(path: str) -> tuple:
    parts = [p for p in str(path or "").split("/") if p != ""]
    if not parts:
        raise ValueError("Patch-Pfad fehlt")
    field = parts[0]
    if field == "notizen":
        if len(parts) != 1:
            raise ValueError("Pfad /notizen erlaubt keine Unterpfade")
        return ("notizen", None)
    if field != "gutachten":
        raise ValueError(f"Patch-Pfad ist nicht erlaubt: /{field}")
    if len(parts) == 1:
        return ("gutachten", None)
    if parts[1] == "-":
        return ("gutachten", "-")
    if parts[1] == _BY_ID_PREFIX and len(parts) == 3:
        return ("gutachten", parts[2])
    raise ValueError(
        "Gutachten werden ueber /gutachten/by-id/<id> adressiert, nicht ueber den Index"
    )


def _index_of(content: Dict[str, Any], gutachten_id: str) -> int:
    for index, entry in enumerate(content.get("gutachten") or []):
        if entry.get("id") == gutachten_id:
            return index
    raise ValueError(f"Unbekannte Gutachten-id: {gutachten_id}")


def apply_assessment_ops(
    content: Dict[str, Any], ops: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply id-addressed ops and return validated content."""
    patched = copy.deepcopy(content or {})
    patched.setdefault("gutachten", [])
    patched.setdefault("notizen", "")

    if not ops:
        raise ValueError("Patch braucht mindestens eine Operation")

    for op in ops:
        operation = op.get("op")
        field, selector = _parse_assessment_path(op.get("path"))
        value = strip_server_fields(op.get("value"))

        if field == "notizen":
            if operation != "set":
                raise ValueError("Auf /notizen ist nur set erlaubt")
            patched["notizen"] = value
            continue

        if operation == "append":
            if selector != "-":
                raise ValueError("append verlangt den Pfad /gutachten/-")
            if not isinstance(value, dict):
                raise ValueError("append verlangt ein Gutachten-Objekt")
            new_id = value.get("id")
            if any(e.get("id") == new_id for e in patched["gutachten"]):
                raise ValueError(
                    f"Gutachten {new_id} existiert bereits, bitte set /gutachten/by-id/{new_id}"
                )
            patched["gutachten"].append(value)
            continue

        if operation == "set":
            if not selector or selector == "-":
                raise ValueError("set verlangt den Pfad /gutachten/by-id/<id>")
            if not isinstance(value, dict):
                raise ValueError("set verlangt ein Gutachten-Objekt")
            if value.get("id") != selector:
                raise ValueError(
                    f"id im Pfad ({selector}) und im Wert ({value.get('id')}) stimmen nicht ueberein"
                )
            patched["gutachten"][_index_of(patched, selector)] = value
            continue

        if operation == "remove":
            if not selector or selector == "-":
                raise ValueError("remove verlangt den Pfad /gutachten/by-id/<id>")
            del patched["gutachten"][_index_of(patched, selector)]
            continue

        raise ValueError(f"Nicht unterstuetzte Operation: {operation}")

    return validate_assessment_content(patched)
```

Add `import copy` to the module's imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_ops.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add app/assessment_memory.py tests/test_memory_assessment_ops.py
git commit -m "feat(memory): id-adressierte Patch-Ops fuer Gutachten

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 3: ORM models and migration

**Files:**
- Modify: `app/models.py` (after `CaseStrategySource`, around line 675)
- Modify: `app/main.py` (migration list, after the newest entry)
- Test: `tests/test_memory_assessment_migration.py`

**Interfaces:**
- Consumes: nothing.
- Produces: ORM classes `CaseAssessment` and `CaseAssessmentSource` with `to_dict()`; migration key `2026-09-03_case_assessments`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_assessment_migration.py`:

```python
"""Schema smoke test for case_assessments.

Needs the live Postgres of the dev stack. Marked slow so the pre-push hook
skips it.

    .venv/bin/python -m pytest tests/test_memory_assessment_migration.py -q
"""

import os
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

pytestmark = pytest.mark.slow


def _engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    from sqlalchemy import create_engine

    return create_engine(url)


def test_migration_key_is_registered():
    main_py = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text()
    assert '"2026-09-03_case_assessments"' in main_py


def test_orm_classes_exist_with_expected_columns():
    from models import CaseAssessment, CaseAssessmentSource

    cols = {c.name for c in CaseAssessment.__table__.columns}
    assert {"id", "owner_id", "case_id", "content_json", "search_text", "version"} <= cols
    assert CaseAssessment.__table__.name == "case_assessments"
    src_cols = {c.name for c in CaseAssessmentSource.__table__.columns}
    assert "case_assessment_id" in src_cols


def test_tables_exist_in_live_database():
    from sqlalchemy import inspect

    inspector = inspect(_engine())
    assert "case_assessments" in inspector.get_table_names()
    assert "case_assessment_sources" in inspector.get_table_names()
    indexes = {i["name"] for i in inspector.get_indexes("case_assessments")}
    assert any("owner_case" in name for name in indexes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_migration.py -q`
Expected: FAIL with `ImportError: cannot import name 'CaseAssessment' from 'models'`

- [ ] **Step 3: Write minimal implementation**

In `app/models.py`, after the `CaseStrategySource` class, add:

```python
class CaseAssessment(Base):
    """Persisted legal assessments (Gutachten) for one case."""
    __tablename__ = "case_assessments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    case_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    content_json = Column(JSONB, nullable=False, default=dict, server_default="{}")
    search_text = Column(Text)
    version = Column(Integer, default=1, nullable=False)
    last_reflected_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ux_case_assessments_owner_case", "owner_id", "case_id", unique=True),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "owner_id": str(self.owner_id) if self.owner_id else None,
            "case_id": str(self.case_id) if self.case_id else None,
            "content_json": self.content_json or {},
            "search_text": self.search_text or "",
            "version": int(self.version or 0),
            "last_reflected_at": self.last_reflected_at.isoformat() if self.last_reflected_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CaseAssessmentSource(Base):
    """Source reference supporting a case assessment statement."""
    __tablename__ = "case_assessment_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_assessment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("case_assessments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    case_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    source_type = Column(String(32), nullable=False, index=True)
    source_id = Column(String(128), index=True)
    label = Column(Text)
    excerpt = Column(Text)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self):
        return {
            "id": str(self.id),
            "case_assessment_id": str(self.case_assessment_id) if self.case_assessment_id else None,
            "owner_id": str(self.owner_id) if self.owner_id else None,
            "case_id": str(self.case_id) if self.case_id else None,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "label": self.label,
            "excerpt": self.excerpt,
            "metadata": self.metadata_ or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
```

Check the top of `app/models.py` for the `Index` import and add it to the
`from sqlalchemy import ...` line if it is missing.

In `app/main.py`, append to the migration list (same structure as the
neighbouring entries — copy the shape of `"2026-08-12_cases_file_reference"`):

```python
        "2026-09-03_case_assessments",
        [
            """
            CREATE TABLE IF NOT EXISTS case_assessments (
                id UUID PRIMARY KEY,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                content_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                search_text TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                last_reflected_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                updated_at TIMESTAMP NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS case_assessment_sources (
                id UUID PRIMARY KEY,
                case_assessment_id UUID NOT NULL
                    REFERENCES case_assessments(id) ON DELETE CASCADE,
                owner_id UUID NOT NULL,
                case_id UUID NOT NULL,
                source_type VARCHAR(32) NOT NULL,
                source_id VARCHAR(128),
                label TEXT,
                excerpt TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP NOT NULL DEFAULT now()
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_case_assessments_owner_case ON case_assessments(owner_id, case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_assessments_owner_id ON case_assessments(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_assessments_case_id ON case_assessments(case_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_assessments_updated_at ON case_assessments(updated_at)",
            "CREATE INDEX IF NOT EXISTS ix_case_assessment_sources_assessment ON case_assessment_sources(case_assessment_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_assessment_sources_owner_id ON case_assessment_sources(owner_id)",
            "CREATE INDEX IF NOT EXISTS ix_case_assessment_sources_case_id ON case_assessment_sources(case_id)",
        ],
```

Match the exact tuple or dict shape the surrounding migration entries use —
read the three entries above the insertion point before writing.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_migration.py -q`
Expected: the first two tests PASS. The live-DB test passes once the app has
restarted against the dev database, otherwise it skips without `DATABASE_URL`.

Then restart the app container so the migration runs, and re-run:

```bash
docker compose -f /var/opt/docker/rechtmaschine/docker-compose.yml restart rechtmaschine-app
docker exec rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db -c "\d case_assessments"
```

Expected: the table listing shows `content_json` as `jsonb not null default '{}'::jsonb`.

- [ ] **Step 5: Commit**

```bash
git add app/models.py app/main.py tests/test_memory_assessment_migration.py
git commit -m "feat(memory): Tabellen und Migration fuer case_assessment

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 4: TargetSpec registry replaces the positional tuple

**Files:**
- Modify: `app/agent_memory_service.py` (lines 202-224 `_target_spec`, plus every unpacking call site)
- Modify: `app/shared.py` (`MemoryTargetType` around line 1442)
- Test: `tests/test_memory_target_registry.py`

**Interfaces:**
- Consumes: `app/assessment_memory` from Tasks 1-2, ORM classes from Task 3.
- Produces: `TargetSpec` NamedTuple with fields `model, source_model, source_fk, default_content, renderer, list_fields, scalar_fields, validate, apply_ops, rebase`; `_target_spec(target_type) -> TargetSpec`; `ASSESSMENT_TARGET = "case_assessment"` re-exported from `agent_memory_service`; `get_or_create_case_assessment(db, owner_id, case_id, for_update=False)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_target_registry.py`, following the stubbing pattern
of `tests/test_memory_rebase_changed_fields.py` (copy its stub block verbatim,
then add `CaseAssessment` and `CaseAssessmentSource` to the list of stubbed
model names):

```python
"""The target registry is a named struct and knows three targets.

    .venv/bin/python -m pytest tests/test_memory_target_registry.py -q
"""

import sys
import types
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

# --- stub the heavy imports (same pattern as test_memory_rebase_changed_fields)
_sqlalchemy = types.ModuleType("sqlalchemy")
_sqlalchemy.desc = lambda *a, **k: None
sys.modules.setdefault("sqlalchemy", _sqlalchemy)
_orm = types.ModuleType("sqlalchemy.orm")


class _Session:
    pass


_orm.Session = _Session
sys.modules.setdefault("sqlalchemy.orm", _orm)
_models = types.ModuleType("models")
for _name in (
    "CaseBrief",
    "CaseStrategy",
    "CaseBriefSource",
    "CaseStrategySource",
    "CaseAssessment",
    "CaseAssessmentSource",
    "CaseMemoryRevision",
    "MemoryUpdateProposal",
    "Document",
    "Case",
    "User",
):
    setattr(_models, _name, type(_name, (), {}))
sys.modules.setdefault("models", _models)

import agent_memory_service as ams  # noqa: E402


def test_registry_knows_three_targets():
    for target in ("case_brief", "case_strategy", "case_assessment"):
        spec = ams._target_spec(target)
        assert spec.model is not None
        assert callable(spec.validate)
        assert callable(spec.renderer)


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        ams._target_spec("case_nonsense")


def test_spec_is_attribute_addressable_not_positional():
    spec = ams._target_spec("case_brief")
    assert hasattr(spec, "list_fields")
    assert "beteiligte" in spec.list_fields


def test_assessment_spec_delegates_to_domain_module():
    from assessment_memory import apply_assessment_ops, validate_assessment_content

    spec = ams._target_spec("case_assessment")
    assert spec.validate is validate_assessment_content
    assert spec.apply_ops is apply_assessment_ops
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_target_registry.py -q`
Expected: FAIL — `_target_spec` returns a tuple, so `spec.model` raises `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

In `app/agent_memory_service.py`:

Add near the other constants:

```python
from typing import Callable, NamedTuple

from assessment_memory import (
    ASSESSMENT_LIST_FIELDS,
    ASSESSMENT_SCALAR_FIELDS,
    ASSESSMENT_TARGET,
    apply_assessment_ops,
    default_case_assessment_json,
    render_case_assessment_compact,
    validate_assessment_content,
)


class TargetSpec(NamedTuple):
    model: Any
    source_model: Any
    source_fk: str
    default_content: Callable[[], Dict[str, Any]]
    renderer: Callable[[Dict[str, Any]], str]
    list_fields: set
    scalar_fields: set
    validate: Callable[[Dict[str, Any]], Dict[str, Any]]
    apply_ops: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]]
```

Replace the body of `_target_spec` with:

```python
def _target_spec(target_type: MemoryTargetType) -> TargetSpec:
    if target_type == BRIEF_TARGET:
        return TargetSpec(
            _model("CaseBrief"),
            _model("CaseBriefSource"),
            "case_brief_id",
            default_case_brief_json,
            render_case_brief_compact,
            BRIEF_LIST_FIELDS,
            BRIEF_SCALAR_FIELDS,
            _validate_brief_content,
            None,
        )
    if target_type == STRATEGY_TARGET:
        return TargetSpec(
            _model("CaseStrategy"),
            _model("CaseStrategySource"),
            "case_strategy_id",
            default_case_strategy_json,
            render_case_strategy_compact,
            STRATEGY_LIST_FIELDS,
            STRATEGY_SCALAR_FIELDS,
            _validate_strategy_content,
            None,
        )
    if target_type == ASSESSMENT_TARGET:
        return TargetSpec(
            _model("CaseAssessment"),
            _model("CaseAssessmentSource"),
            "case_assessment_id",
            default_case_assessment_json,
            render_case_assessment_compact,
            ASSESSMENT_LIST_FIELDS,
            ASSESSMENT_SCALAR_FIELDS,
            validate_assessment_content,
            apply_assessment_ops,
        )
    raise ValueError(f"Unsupported memory target type: {target_type}")
```

Then convert every unpacking call site to attribute access. Find them with:

```bash
grep -n "_target_spec(" app/agent_memory_service.py
```

The sites are in `_target_content`, `_write_target_content`,
`_update_target_content`, `_apply_patch_ops`, `_get_target`,
`_get_or_create_target`, `accept_memory_update_proposal` and
`_create_source_records`. Replace patterns like
`model, _, _, _, _, _, _ = _target_spec(t)` with `spec = _target_spec(t)` and
`spec.model`.

In `_target_content`, replace the brief-or-strategy branch with:

```python
def _target_content(target_type: MemoryTargetType, target: Any) -> Dict[str, Any]:
    spec = _target_spec(target_type)
    if _has_column(target, "content_json"):
        return spec.validate(getattr(target, "content_json", None) or {})
    if target_type == BRIEF_TARGET:
        ...  # keep the existing legacy-column fallback unchanged
```

For `case_assessment` the legacy fallback is not reachable (the table always
has `content_json`), so add at the top of the fallback section:

```python
    if target_type == ASSESSMENT_TARGET:
        return spec.validate({})
```

In `_write_target_content`, `_update_target_content` and `_apply_patch_ops`,
replace the `if target_type == BRIEF_TARGET: _validate_brief_content(...)` /
`else: _validate_strategy_content(...)` pairs with `spec.validate(...)`.

At the top of `_apply_patch_ops`, delegate when the spec provides its own
engine:

```python
def _apply_patch_ops(
    target_type: MemoryTargetType,
    content: Dict[str, Any],
    ops: Iterable[Any],
) -> Dict[str, Any]:
    spec = _target_spec(target_type)
    parsed_ops = [_model_dump(op) for op in ops]
    if spec.apply_ops is not None:
        return spec.apply_ops(content, parsed_ops)
    ... # existing generic implementation, now reading spec.list_fields / spec.scalar_fields
```

Fix `memory_row_to_dict`, which currently classifies any non-`CaseBrief` row
as strategy:

```python
_ROW_CLASS_TO_TARGET = {
    "CaseBrief": BRIEF_TARGET,
    "CaseStrategy": STRATEGY_TARGET,
    "CaseAssessment": ASSESSMENT_TARGET,
}


def memory_row_to_dict(row: Any, rendered: Optional[str] = None) -> Dict[str, Any]:
    target_type = _ROW_CLASS_TO_TARGET.get(type(row).__name__, STRATEGY_TARGET)
    ...
```

Add the accessor next to the two existing ones:

```python
def get_or_create_case_assessment(db: Session, owner_id: Any, case_id: Any, for_update: bool = False) -> Any:
    return _get_or_create_target(db, ASSESSMENT_TARGET, owner_id, case_id, for_update=for_update)
```

In `app/shared.py` change:

```python
MemoryTargetType = Literal["case_brief", "case_strategy", "case_assessment"]
```

and add next to `CaseStrategyResponse`:

```python
class CaseAssessmentResponse(CaseMemoryBaseResponse):
    target_type: Literal["case_assessment"] = "case_assessment"
```

`render_case_assessment_compact` does not exist yet — add a placeholder to
`app/assessment_memory.py` so the import resolves. Task 6 replaces it:

```python
def render_case_assessment_compact(content: Dict[str, Any]) -> str:
    assessment = validate_assessment_content(content)
    if not assessment["gutachten"]:
        return "Rechtliche Wuerdigung: Keine gepflegten Inhalte."
    lines = ["Rechtliche Wuerdigung:"]
    for entry in assessment["gutachten"]:
        lines.append(f"[{entry['id']}, Stand {entry['stand']}] {entry['rechtsfrage']}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_target_registry.py tests/test_memory_rebase_changed_fields.py tests/test_memory_get_projection.py tests/test_memory_combined_payload.py -q`
Expected: PASS. The registry test proves the new shape, the three existing
tests prove the refactor did not break brief and strategy.

Then the whole suite: `.venv/bin/python -m pytest tests/ -q -m "not slow"`
Expected: same pass count as before the task, plus the new tests.

- [ ] **Step 5: Commit**

```bash
git add app/agent_memory_service.py app/assessment_memory.py app/shared.py tests/test_memory_target_registry.py
git commit -m "refactor(memory): TargetSpec-Register statt Positionstupel, drittes Target registriert

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 5: Store reconciliation and the accept hook

**Files:**
- Modify: `app/assessment_memory.py`
- Modify: `app/verify_source.py` (export the normalizer)
- Modify: `app/agent_memory_service.py` (`accept_memory_update_proposal`, `create_memory_update_proposal`)
- Test: `tests/test_memory_assessment_store.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4.
- Produces: `az_for_compare(az) -> str` (public re-export in `verify_source`); `load_store_map(db) -> dict[str, list[dict]]`; `reconcile_store(content: dict, store_map: dict, changed_ids: set[str] | None, now_iso: str) -> tuple[dict, list[dict]]`; `changed_gutachten_ids(previous: dict, new: dict) -> set[str]`; `citation_lines(warnings: list[dict], content: dict) -> list[str]`. `accept_memory_update_proposal` now returns `(proposal, warnings)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_assessment_store.py`:

```python
"""Store reconciliation for Gutachten-Fundstellen.

    .venv/bin/python -m pytest tests/test_memory_assessment_store.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    changed_gutachten_ids,
    citation_lines,
    reconcile_store,
    validate_assessment_content,
)

NOW = "2026-09-04T10:00:00"


def _content(*fundstellen):
    return validate_assessment_content(
        {
            "gutachten": [
                {
                    "id": "a-frage",
                    "rechtsfrage": "Frage?",
                    "ergebnis": "Antwort.",
                    "stand": "2026-09-03",
                    "fundstellen": list(fundstellen),
                }
            ],
            "notizen": "",
        }
    )


def _store_map():
    return {
        "18e491/12": [{"id": "entry-1", "decision_date": "2012-06-18"}],
        "1k2/24": [{"id": "entry-2", "decision_date": "2024-01-01"}],
        "9x9/99": [
            {"id": "entry-3", "decision_date": "1999-01-01"},
            {"id": "entry-4", "decision_date": "1999-06-06"},
        ],
    }


def test_matching_az_and_date_is_verified():
    content = _content({"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"})
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    fundstelle = out["gutachten"][0]["fundstellen"][0]
    assert fundstelle["store"] == "verified"
    assert fundstelle["store_entry_id"] == "entry-1"
    assert fundstelle["store_checked_at"] == NOW
    assert warnings == []


def test_wrong_date_is_date_mismatch():
    content = _content({"gericht": "VG X", "datum": "2024-02-02", "az": "1 K 2/24"})
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    assert out["gutachten"][0]["fundstellen"][0]["store"] == "date_mismatch"
    assert warnings == [{"gutachten_id": "a-frage", "az": "1 K 2/24", "store": "date_mismatch"}]


def test_unknown_az_is_not_in_store():
    content = _content({"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20"})
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    assert out["gutachten"][0]["fundstellen"][0]["store"] == "not_in_store"
    assert warnings[0]["store"] == "not_in_store"


def test_multiple_store_hits_one_matching_date_is_verified():
    content = _content({"gericht": "VG Z", "datum": "1999-06-06", "az": "9 X 9/99"})
    out, _ = reconcile_store(content, _store_map(), None, NOW)
    fundstelle = out["gutachten"][0]["fundstellen"][0]
    assert fundstelle["store"] == "verified"
    assert fundstelle["store_entry_id"] == "entry-4"


def test_changed_ids_limits_reconciliation():
    content = _content({"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"})
    out, _ = reconcile_store(content, _store_map(), set(), NOW)
    assert out["gutachten"][0]["fundstellen"][0]["store"] == "unchecked"


def test_changed_gutachten_ids_ignores_server_fields():
    before = _content({"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"})
    after, _ = reconcile_store(before, _store_map(), None, NOW)
    assert changed_gutachten_ids(before, after) == set()
    after["gutachten"][0]["ergebnis"] = "Andere Antwort."
    assert changed_gutachten_ids(before, after) == {"a-frage"}


def test_citation_lines_use_the_parser_format():
    content = _content(
        {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12", "art": "Beschluss"},
        {"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20"},
    )
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    lines = citation_lines(warnings, out)
    assert lines == ["VG Y, Beschluss vom 01.01.2020 – 7 L 7/20"]


def test_parser_accepts_the_generated_line():
    from draft_citation_ingest import parse_decision_citations

    content = _content({"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20"})
    out, warnings = reconcile_store(content, _store_map(), None, NOW)
    line = citation_lines(warnings, out)[0]
    parsed = parse_decision_citations(line)
    assert parsed and parsed[0]["az"] == "7 L 7/20"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_store.py -q`
Expected: FAIL with `ImportError: cannot import name 'changed_gutachten_ids'`

- [ ] **Step 3: Write minimal implementation**

In `app/verify_source.py`, add a public alias next to the existing import so
other modules never reach into `jurisprudence_ingest` directly:

```python
def az_for_compare(az: Optional[str]) -> str:
    """Public alias for the canonical Aktenzeichen normalizer."""
    return _az_for_compare(az)
```

Append to `app/assessment_memory.py`:

```python
from datetime import datetime


def load_store_map(db: Any) -> Dict[str, List[Dict[str, Any]]]:
    """One pass over the active Rechtsprechung entries, keyed by normalized Az.

    store_lookup() rescans the whole table per call -- for a Gutachten with 25
    Fundstellen that would be 25 full scans, and the nightly recheck multiplies
    that by every case. Load once, compare in memory."""
    from models import RechtsprechungEntry
    from verify_source import az_for_compare

    store_map: Dict[str, List[Dict[str, Any]]] = {}
    rows = (
        db.query(RechtsprechungEntry)
        .filter(
            RechtsprechungEntry.is_active.is_(True),
            RechtsprechungEntry.aktenzeichen.isnot(None),
        )
        .all()
    )
    for row in rows:
        key = az_for_compare(row.aktenzeichen)
        if not key:
            continue
        store_map.setdefault(key, []).append(
            {
                "id": str(row.id),
                "decision_date": row.decision_date.isoformat() if row.decision_date else None,
            }
        )
    return store_map


def _content_without_server_fields(entry: Dict[str, Any]) -> str:
    return json.dumps(strip_server_fields(entry), ensure_ascii=False, sort_keys=True)


def changed_gutachten_ids(previous: Dict[str, Any], new: Dict[str, Any]) -> set:
    """Ids whose substance changed, ignoring server-owned fields."""
    before = {
        e["id"]: _content_without_server_fields(e)
        for e in (previous or {}).get("gutachten") or []
    }
    after = {
        e["id"]: _content_without_server_fields(e)
        for e in (new or {}).get("gutachten") or []
    }
    changed = {gid for gid, body in after.items() if before.get(gid) != body}
    changed |= set(before) - set(after)
    return changed


def reconcile_store(
    content: Dict[str, Any],
    store_map: Dict[str, List[Dict[str, Any]]],
    changed_ids: Optional[set],
    now_iso: Optional[str] = None,
) -> tuple:
    """Set the server-owned store fields. Returns (content, warnings).

    changed_ids=None means every Gutachten is reconciled (recheck). A set
    limits the work to the Gutachten an accept actually touched."""
    from verify_source import az_for_compare

    stamp = now_iso or datetime.utcnow().isoformat()
    patched = copy.deepcopy(content or {})
    warnings: List[Dict[str, str]] = []

    for entry in patched.get("gutachten") or []:
        if changed_ids is not None and entry.get("id") not in changed_ids:
            continue
        for fundstelle in entry.get("fundstellen") or []:
            key = az_for_compare(fundstelle.get("az"))
            hits = store_map.get(key) or []
            match = next(
                (h for h in hits if h.get("decision_date") == fundstelle.get("datum")),
                None,
            )
            if match:
                state, entry_id = "verified", match["id"]
            elif hits:
                state, entry_id = "date_mismatch", None
            else:
                state, entry_id = "not_in_store", None
            fundstelle["store"] = state
            fundstelle["store_entry_id"] = entry_id
            fundstelle["store_checked_at"] = stamp
            if state != "verified":
                warnings.append(
                    {
                        "gutachten_id": entry.get("id"),
                        "az": fundstelle.get("az"),
                        "store": state,
                    }
                )

    return validate_assessment_content(patched), warnings


def citation_lines(warnings: List[Dict[str, str]], content: Dict[str, Any]) -> List[str]:
    """Parser-compatible citation lines for every not_in_store Fundstelle."""
    wanted = {
        (w["gutachten_id"], w["az"]) for w in warnings if w["store"] == "not_in_store"
    }
    lines: List[str] = []
    for entry in content.get("gutachten") or []:
        for fundstelle in entry.get("fundstellen") or []:
            if (entry.get("id"), fundstelle.get("az")) not in wanted:
                continue
            year, month, day = fundstelle["datum"].split("-")
            art = fundstelle.get("art") or "Beschluss"
            lines.append(
                f"{fundstelle['gericht']}, {art} vom {day}.{month}.{year} – {fundstelle['az']}"
            )
    return lines
```

In `app/agent_memory_service.py`, sanitize on create. In
`create_memory_update_proposal`, right after `ops_list = [...]`:

```python
    if target_type == ASSESSMENT_TARGET:
        from assessment_memory import sanitize_assessment_ops

        ops_list = sanitize_assessment_ops(ops_list)
```

In `accept_memory_update_proposal`, between `new_content = _apply_patch_ops(...)`
and `_create_revision(...)`:

```python
    assessment_warnings: List[Dict[str, str]] = []
    citation_requests: List[str] = []
    if target_type == ASSESSMENT_TARGET:
        from assessment_memory import (
            changed_gutachten_ids,
            citation_lines,
            load_store_map,
            reconcile_store,
        )

        changed_ids = changed_gutachten_ids(previous_content, new_content)
        try:
            # A failing store query must not poison the accept transaction, so
            # the lookup runs on its own short-lived session.
            from database import SessionLocal

            with SessionLocal() as store_db:
                store_map = load_store_map(store_db)
            new_content, assessment_warnings = reconcile_store(
                new_content, store_map, changed_ids
            )
            citation_requests = citation_lines(assessment_warnings, new_content)
        except Exception as exc:  # noqa: BLE001 - accept must survive
            print(f"[WARN] Gutachten-Store-Abgleich fehlgeschlagen: {exc}")
            assessment_warnings = [
                {"gutachten_id": gid, "az": "", "store": "unchecked"}
                for gid in sorted(changed_ids)
            ]
```

Check the actual name of the session factory module first:

```bash
grep -rn "SessionLocal" app/database.py app/shared.py | head -3
```

Use whatever the codebase already exposes.

Change the tail of the function so the spawn happens after the commit and the
warnings travel with the proposal:

```python
    db.add(target)
    db.add(proposal)
    db.commit()
    db.refresh(proposal)

    if citation_requests:
        try:
            from draft_citation_ingest import spawn_for_text

            spawn_for_text("\n".join(citation_requests))
        except Exception as exc:  # noqa: BLE001 - best effort
            print(f"[WARN] Zitat-Beschaffung nicht gestartet: {exc}")

    return proposal, assessment_warnings
```

Update the two internal callers of `accept_memory_update_proposal` (find them
with `grep -rn "accept_memory_update_proposal" app scripts`) to unpack the
tuple.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_store.py -q`
Expected: PASS, 8 tests

Run: `.venv/bin/python -m pytest tests/ -q -m "not slow"`
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add app/assessment_memory.py app/verify_source.py app/agent_memory_service.py tests/test_memory_assessment_store.py
git commit -m "feat(memory): Store-Abgleich der Gutachten-Fundstellen beim Accept

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

## Remaining tasks

Tasks 6-12 cover: the prompt renderer with its own budget and the blocked-Az
split (Task 6), the assessment rebase strategy (Task 7), the API surface with
recheck and the frontend payload (Task 8), the CLI (Task 9), the pattern-wiki
whitelist (Task 10), the triage and hook integration (Task 11), and the skill
documentation plus live acceptance on 157/26 (Task 12). They are written in the
same shape as Tasks 1-5 and are appended to this file before execution starts —
see the section below.

---

### Task 6: Prompt renderer, budget and blocked-Az split

**Files:**
- Modify: `app/assessment_memory.py` (replace the placeholder renderer)
- Modify: `app/agent_memory_service.py` (`get_case_memory_prompt_context`)
- Modify: `app/citation_verifier.py` (`verify_facts`)
- Modify: `app/endpoints/generation.py` (`_attach_fact_checks` call site)
- Test: `tests/test_memory_assessment_render.py`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: `render_assessment_block(content: dict, max_chars: int = 4000) -> tuple[str, list[str], list[str]]` returning `(block_text, used_ids, blocked_az)`; `render_case_assessment_compact(content) -> str` (unchanged name, now delegating). `get_case_memory_prompt_context` fills `collect["assessment_used"]`, `collect["assessment_ids"]`, `collect["assessment_blocked_az"]`. `verify_facts` gains the keyword argument `blocked_az: Optional[set] = None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_assessment_render.py`:

```python
"""Prompt block rendering, budget stages and the blocked-Az split.

    .venv/bin/python -m pytest tests/test_memory_assessment_render.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import render_assessment_block, validate_assessment_content  # noqa: E402


def _entry(gid, stand, store="verified", risiken=None, pruefung=None):
    return {
        "id": gid,
        "rechtsfrage": f"Frage {gid}?",
        "ergebnis": f"Ergebnis {gid}.",
        "stand": stand,
        "status": "aktiv",
        "pruefung": pruefung
        if pruefung is not None
        else [
            {
                "these": f"These {gid}",
                "bewertung": f"Bewertung {gid}",
                "fundstellen": ["18 E 491/12"],
            }
        ],
        "fundstellen": [
            {
                "gericht": "OVG NRW",
                "datum": "2012-06-18",
                "az": "18 E 491/12",
                "art": "Beschluss",
                "aussage": "Die GUEB ersetzt die Duldung nicht.",
                "richtung": "pro",
                "store": store,
            }
        ],
        "risiken": risiken if risiken is not None else [f"Risiko {gid}"],
        "quelle": "Vermerk.pdf",
    }


def _content(*entries):
    return validate_assessment_content({"gutachten": list(entries), "notizen": ""})


def test_verified_fundstelle_is_quotable():
    block, used, blocked = render_assessment_block(_content(_entry("a", "2026-09-03")))
    assert "OVG NRW 18 E 491/12 (18.06.2012, pro)" in block
    assert used == ["a"]
    assert blocked == []


def test_unverified_fundstelle_lands_in_the_blocklist_only():
    block, _, blocked = render_assessment_block(
        _content(_entry("a", "2026-09-03", store="not_in_store"))
    )
    assert "Nicht zitierfaehig" in block
    assert "18 E 491/12" in block
    assert "(18.06.2012, pro)" not in block
    assert blocked == ["18 E 491/12"]


def test_only_active_gutachten_are_rendered():
    overholt = _entry("b", "2026-09-04")
    overholt["status"] = "ueberholt"
    _, used, _ = render_assessment_block(_content(_entry("a", "2026-09-03"), overholt))
    assert used == ["a"]


def test_newest_stand_first():
    _, used, _ = render_assessment_block(
        _content(_entry("alt", "2026-01-01"), _entry("neu", "2026-09-03"))
    )
    assert used == ["neu", "alt"]


def test_budget_drops_risiken_then_pruefung_then_whole_gutachten():
    entries = [_entry(f"g{i}", f"2026-0{i+1}-01") for i in range(4)]
    block, used, _ = render_assessment_block(_content(*entries), max_chars=420)
    assert "Risiko" not in block
    assert len(used) < 4
    assert "weitere Gutachten gekuerzt" in block
    for gid in used:
        assert f"Frage {gid}?" in block
        assert f"Ergebnis {gid}." in block


def test_empty_content_renders_nothing():
    block, used, blocked = render_assessment_block(_content())
    assert block == ""
    assert used == []
    assert blocked == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_render.py -q`
Expected: FAIL with `ImportError: cannot import name 'render_assessment_block'`

- [ ] **Step 3: Write minimal implementation**

Replace the placeholder `render_case_assessment_compact` in
`app/assessment_memory.py` with:

```python
ASSESSMENT_BLOCK_HEADER = (
    "RECHTLICHE WUERDIGUNG DER KANZLEI "
    "(Fundstellen mit Store-Abgleich, Stand je Gutachten):"
)


def _de_date(iso: str) -> str:
    year, month, day = iso.split("-")
    return f"{day}.{month}.{year}"


def _render_entry(entry: Dict[str, Any], with_pruefung: bool, with_risiken: bool) -> str:
    verified = {
        f["az"]: f for f in entry.get("fundstellen") or [] if f.get("store") == "verified"
    }
    blocked = [
        f["az"] for f in entry.get("fundstellen") or [] if f.get("store") != "verified"
    ]
    lines = [
        f"[{entry['id']}, Stand {_de_date(entry['stand'])}] "
        f"Rechtsfrage: {entry['rechtsfrage']} Ergebnis: {entry['ergebnis']}"
    ]
    if with_pruefung:
        for punkt in entry.get("pruefung") or []:
            cites = [
                f"{verified[az]['gericht']} {az} "
                f"({_de_date(verified[az]['datum'])}, {verified[az]['richtung']})"
                for az in punkt.get("fundstellen") or []
                if az in verified
            ]
            suffix = f" – Fundstellen: {', '.join(cites)}" if cites else ""
            lines.append(f"  Pruefung: {punkt['these']} – {punkt['bewertung']}{suffix}")
    if with_risiken and entry.get("risiken"):
        lines.append("  Risiken: " + "; ".join(entry["risiken"]))
    if blocked:
        lines.append("  Nicht zitierfaehig (nicht im Bestand): " + ", ".join(blocked))
    return "\n".join(lines)


def render_assessment_block(
    content: Dict[str, Any], max_chars: int = 4000
) -> tuple:
    """Render the prompt block. Returns (text, used_ids, blocked_az).

    Truncation stages, oldest Gutachten first: drop risiken, then pruefung,
    then whole Gutachten from the end. Rechtsfrage, Ergebnis and the blocklist
    always survive for every Gutachten that is rendered at all."""
    assessment = validate_assessment_content(content)
    active = [e for e in assessment["gutachten"] if e.get("status") == "aktiv"]
    if not active:
        return "", [], []
    active.sort(key=lambda e: e["stand"], reverse=True)

    for with_pruefung, with_risiken in ((True, True), (True, False), (False, False)):
        entries = active
        while entries:
            body = "\n".join(
                _render_entry(e, with_pruefung, with_risiken) for e in entries
            )
            dropped = len(active) - len(entries)
            if dropped:
                body += f"\n[weitere Gutachten gekuerzt: {dropped}]"
            text = f"{ASSESSMENT_BLOCK_HEADER}\n{body}"
            if len(text) <= max_chars or (not with_pruefung and not with_risiken and len(entries) == 1):
                if len(text) <= max_chars or len(entries) == 1:
                    used = [e["id"] for e in entries]
                    blocked = [
                        f["az"]
                        for e in entries
                        for f in e.get("fundstellen") or []
                        if f.get("store") != "verified"
                    ]
                    return text, used, blocked
            if with_pruefung or with_risiken:
                break
            entries = entries[:-1]

    # Only reachable when a single Gutachten alone exceeds the budget; the
    # per-field limits keep that under ~1.5 kB, so this is a safety net.
    entry = active[0]
    text = f"{ASSESSMENT_BLOCK_HEADER}\n{_render_entry(entry, False, False)}"
    blocked = [
        f["az"] for f in entry.get("fundstellen") or [] if f.get("store") != "verified"
    ]
    return text, [entry["id"]], blocked


def render_case_assessment_compact(content: Dict[str, Any]) -> str:
    """search_text renderer for the ORM row (no budget)."""
    text, _, _ = render_assessment_block(content, max_chars=MAX_CONTENT_BYTES)
    return text or "Rechtliche Wuerdigung: Keine gepflegten Inhalte."
```

In `app/agent_memory_service.py`, extend `get_case_memory_prompt_context`.
Add the parameter `max_assessment_chars: int = 4000` to the signature. After
the strategy block and before the `rendered = "\n\n".join(chunks).strip()`
line, insert:

```python
    assessment_block = ""
    assessment_ids: List[str] = []
    blocked_az: List[str] = []
    try:
        from assessment_memory import render_assessment_block

        assessment = get_or_create_case_assessment(db, owner_id, case_id)
        assessment_block, assessment_ids, blocked_az = render_assessment_block(
            _target_content(ASSESSMENT_TARGET, assessment), max_chars=max_assessment_chars
        )
    except Exception as exc:
        print(f"[WARN] Failed to render case assessment memory: {exc}")
```

Keep the existing `rendered = "\n\n".join(chunks).strip()` and the existing
`max_chars` truncation exactly where they are, so brief and strategy keep
their own budget. Immediately after that truncation, append the assessment:

```python
    if assessment_block:
        rendered = f"{rendered}\n\n{assessment_block}" if rendered else assessment_block
```

Move the pseudonymization call so it runs after the assessment was appended
(it currently sits before the truncation — the whole text including the
assessment must go through it). Then extend the `collect` block:

```python
    if collect is not None:
        collect["case_memory_used"] = bool(brief_used or strategy_used or assessment_ids)
        collect["case_memory_text"] = rendered
        collect["assessment_used"] = bool(assessment_ids) and bool(rendered)
        collect["assessment_ids"] = assessment_ids if rendered else []
        collect["assessment_blocked_az"] = blocked_az if rendered else []
```

The `base_memory` variable that feeds doktrin/wiki/pack matching is assigned
from `rendered` after this point, so it picks the assessment up automatically.

For the fact corpus, the blocklist lines must not count as evidence. In
`citation_verifier.verify_facts`, add the parameter and the check:

```python
def verify_facts(
    draft_text: str,
    selected_documents: Dict[str, List[Dict[str, Any]]],
    memory_text: str = "",
    blocked_az: Optional[set] = None,
) -> Dict[str, Any]:
```

Right after `corpus_az = {...}`:

```python
    blocked_norm = {_norm_az(a) for a in (blocked_az or set())}
    corpus_az -= blocked_norm
```

and inside the Aktenzeichen loop, before the `not in corpus_az` branch:

```python
        if norm in blocked_norm:
            checks.append({
                "type": "aktenzeichen", "value": raw, "severity": "high",
                "status": "blocked_citation",
                "reason": "Fundstelle steht im Gutachten als nicht zitierfaehig (nicht im Bestand).",
            })
            continue
```

In `app/endpoints/generation.py`, `_attach_fact_checks` receives `grounding`
already. Pass the blocked set through to `verify_facts`:

```python
        result = verify_facts(
            generated_text,
            collected,
            (grounding or {}).get("case_memory_text", ""),
            blocked_az=set((grounding or {}).get("assessment_blocked_az") or []),
        )
```

Read the existing call inside `_attach_fact_checks` first and keep its
positional arguments unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_render.py -q`
Expected: PASS, 6 tests

Run: `.venv/bin/python -m pytest tests/ -q -m "not slow"`
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add app/assessment_memory.py app/agent_memory_service.py app/citation_verifier.py app/endpoints/generation.py tests/test_memory_assessment_render.py
git commit -m "feat(memory): Gutachten-Block im Prompt mit eigenem Budget und Sperrliste

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 7: Assessment rebase strategy

**Files:**
- Modify: `app/assessment_memory.py`
- Modify: `app/agent_memory_service.py` (`_rebase_pending_proposals`)
- Test: `tests/test_memory_assessment_rebase.py`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: `rebase_assessment_ops(ops: list[dict], new_content: dict, conflict_ids: set[str]) -> list[dict]` in `assessment_memory`. `_rebase_pending_proposals` delegates when `target_type == ASSESSMENT_TARGET` and takes `conflict_ids` instead of `curated_fields` for that target.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_assessment_rebase.py`:

```python
"""Gutachten-level rebase: conflicts are per id, not per field.

    .venv/bin/python -m pytest tests/test_memory_assessment_rebase.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import rebase_assessment_ops, validate_assessment_content  # noqa: E402


def _entry(gid):
    return {
        "id": gid,
        "rechtsfrage": "Frage?",
        "ergebnis": "Antwort.",
        "stand": "2026-09-03",
        "fundstellen": [
            {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"}
        ],
    }


def _content(*ids):
    return validate_assessment_content(
        {"gutachten": [_entry(i) for i in ids], "notizen": ""}
    )


def test_op_on_untouched_id_survives():
    ops = [{"op": "set", "path": "/gutachten/by-id/b", "value": _entry("b")}]
    kept = rebase_assessment_ops(ops, _content("a", "b"), conflict_ids={"a"})
    assert kept == ops


def test_op_on_conflicting_id_is_dropped():
    ops = [{"op": "set", "path": "/gutachten/by-id/a", "value": _entry("a")}]
    kept = rebase_assessment_ops(ops, _content("a"), conflict_ids={"a"})
    assert kept == []


def test_append_of_now_existing_id_is_dropped():
    ops = [{"op": "append", "path": "/gutachten/-", "value": _entry("a")}]
    kept = rebase_assessment_ops(ops, _content("a"), conflict_ids=set())
    assert kept == []


def test_op_on_removed_id_is_dropped():
    ops = [{"op": "set", "path": "/gutachten/by-id/gone", "value": _entry("gone")}]
    kept = rebase_assessment_ops(ops, _content("a"), conflict_ids=set())
    assert kept == []


def test_kept_ops_must_apply_together():
    # Both ops individually valid against the new content, but the second
    # depends on the first having added "c" -- applied together they are fine.
    ops = [
        {"op": "append", "path": "/gutachten/-", "value": _entry("c")},
        {"op": "set", "path": "/gutachten/by-id/c", "value": _entry("c")},
    ]
    kept = rebase_assessment_ops(ops, _content("a"), conflict_ids=set())
    assert len(kept) == 2


def test_conflicting_second_op_drops_only_itself():
    ops = [
        {"op": "append", "path": "/gutachten/-", "value": _entry("c")},
        {"op": "set", "path": "/gutachten/by-id/a", "value": _entry("a")},
    ]
    kept = rebase_assessment_ops(ops, _content("a"), conflict_ids={"a"})
    assert [op["op"] for op in kept] == ["append"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_rebase.py -q`
Expected: FAIL with `ImportError: cannot import name 'rebase_assessment_ops'`

- [ ] **Step 3: Write minimal implementation**

Append to `app/assessment_memory.py`:

```python
def _op_target_id(op: Dict[str, Any]) -> Optional[str]:
    try:
        field, selector = _parse_assessment_path(op.get("path"))
    except ValueError:
        return None
    if field != "gutachten":
        return None
    if selector and selector != "-":
        return selector
    value = op.get("value")
    return value.get("id") if isinstance(value, dict) else None


def rebase_assessment_ops(
    ops: List[Dict[str, Any]],
    new_content: Dict[str, Any],
    conflict_ids: set,
) -> List[Dict[str, Any]]:
    """Keep the ops of a pending proposal that still make sense.

    Identity is the Gutachten id. An op is dropped when its id is in the
    accept's conflict set, when an append collides with a now-existing id, or
    when the surviving ops cannot be applied together to the new content."""
    existing_ids = {e.get("id") for e in (new_content or {}).get("gutachten") or []}
    kept: List[Dict[str, Any]] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        gid = _op_target_id(op)
        if gid is None:
            kept.append(op)  # /notizen and friends
            continue
        if gid in conflict_ids:
            continue
        pending_ids = {_op_target_id(k) for k in kept if k.get("op") == "append"}
        if op.get("op") == "append" and gid in (existing_ids | pending_ids):
            continue
        candidate = kept + [op]
        try:
            apply_assessment_ops(new_content, candidate)
        except ValueError:
            continue
        kept = candidate
    return kept
```

In `app/agent_memory_service.py`, teach `_rebase_pending_proposals` to
delegate. At the top of the per-sibling loop:

```python
    for sibling in siblings:
        if target_type == ASSESSMENT_TARGET:
            from assessment_memory import rebase_assessment_ops

            kept = rebase_assessment_ops(
                [op for op in _proposal_ops(sibling) if isinstance(op, dict)],
                new_content,
                conflict_ids=set(curated_fields or set()),
            )
        else:
            kept = ...  # existing generic loop, unchanged
```

At the accept call site, pass ids instead of field names for this target:

```python
    if target_type == ASSESSMENT_TARGET:
        from assessment_memory import changed_gutachten_ids

        rebase_scope = changed_gutachten_ids(previous_content, new_content)
    else:
        rebase_scope = _changed_fields(previous_content, new_content)
    _rebase_pending_proposals(
        db, owner_id, target_type, target, new_content,
        int(getattr(target, "version", 0) or 0),
        accepted_proposal_id=getattr(proposal, "id"),
        curated_fields=rebase_scope,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_rebase.py tests/test_memory_rebase_changed_fields.py -q`
Expected: PASS, 6 new tests plus the existing ones.

- [ ] **Step 5: Commit**

```bash
git add app/assessment_memory.py app/agent_memory_service.py tests/test_memory_assessment_rebase.py
git commit -m "feat(memory): Rebase pending Gutachten-Proposals je id statt je Feld

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 8: API surface, recheck endpoint and frontend payload

**Files:**
- Modify: `app/endpoints/agent_memory.py` (`MemoryProposalCreateRequest` validation, `_combined_payload`, `_proposal_frontend_payload`, accept route, new recheck route)
- Modify: `app/assessment_memory.py` (recheck service function)
- Test: `tests/test_memory_assessment_api.py`

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces: `recheck_assessment(db, owner_id, case_id) -> dict` in `assessment_memory` returning `{"changed_fundstellen": int, "changed_gutachten": int, "warnings": list}`; route `POST /memory/cases/{case_id}/assessment/recheck`; `_combined_payload(brief, strategy, assessment)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_assessment_api.py`:

```python
"""Route wiring for case_assessment: payload shape, target allowlist, recheck.

Source-level assertions plus pure-function tests -- the live API is covered by
the acceptance run in Task 12.

    .venv/bin/python -m pytest tests/test_memory_assessment_api.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

ENDPOINT = (APP_DIR / "endpoints" / "agent_memory.py").read_text(encoding="utf-8")


def test_create_route_validates_the_target_against_the_registry():
    assert "_KNOWN_TARGETS" in ENDPOINT
    assert "Unbekanntes Memory-Target" in ENDPOINT


def test_combined_payload_carries_the_third_block():
    assert '"case_assessment": assessment_payload' in ENDPOINT


def test_frontend_payload_has_an_assessment_section():
    assert '"assessment"' in ENDPOINT
    assert "Gutachten-Vorschlag" in ENDPOINT


def test_accept_route_surfaces_warnings():
    assert "assessment_warnings" in ENDPOINT


def test_recheck_route_exists():
    assert '@router.post("/cases/{case_id}/assessment/recheck")' in ENDPOINT


def test_recheck_counts_only_changed_fundstellen():
    from assessment_memory import reconcile_store, validate_assessment_content

    content = validate_assessment_content(
        {
            "gutachten": [
                {
                    "id": "a",
                    "rechtsfrage": "F?",
                    "ergebnis": "E.",
                    "stand": "2026-09-03",
                    "fundstellen": [
                        {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12"},
                        {"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20"},
                    ],
                }
            ],
            "notizen": "",
        }
    )
    store_map = {"18e491/12": [{"id": "entry-1", "decision_date": "2012-06-18"}]}
    once, _ = reconcile_store(content, store_map, None, "2026-09-04T10:00:00")
    twice, _ = reconcile_store(once, store_map, None, "2026-09-04T10:00:00")
    states_once = [f["store"] for f in once["gutachten"][0]["fundstellen"]]
    states_twice = [f["store"] for f in twice["gutachten"][0]["fundstellen"]]
    assert states_once == ["verified", "not_in_store"]
    assert states_once == states_twice, "recheck must be idempotent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_api.py -q`
Expected: FAIL — the source assertions do not find the new strings.

- [ ] **Step 3: Write minimal implementation**

Append the recheck service to `app/assessment_memory.py`:

```python
def recheck_assessment(db: Any, owner_id: Any, case_id: Any) -> Dict[str, Any]:
    """Re-run the store reconciliation for every Fundstelle of a case.

    Also re-checks entries that are already `verified`: a deactivated or
    corrected store entry must not stay quotable forever. Writes a revision,
    does NOT bump the version and does NOT rebase pending proposals -- only
    server-owned fields change, and proposal ops never carry those."""
    from agent_memory_service import (
        ASSESSMENT_TARGET,
        _create_revision,
        _target_content,
        _write_target_content,
        get_or_create_case_assessment,
        render_case_assessment_compact,
    )

    target = get_or_create_case_assessment(db, owner_id, case_id, for_update=True)
    previous = _target_content(ASSESSMENT_TARGET, target)
    if not (previous.get("gutachten") or []):
        return {"changed_fundstellen": 0, "changed_gutachten": 0, "warnings": []}

    store_map = load_store_map(db)
    new_content, warnings = reconcile_store(previous, store_map, None)

    changed_fundstellen = 0
    changed_gutachten: set = set()
    old_by_id = {e["id"]: e for e in previous["gutachten"]}
    for entry in new_content["gutachten"]:
        old_entry = old_by_id.get(entry["id"], {})
        old_states = {
            f["az"]: (f.get("store"), f.get("store_entry_id"))
            for f in old_entry.get("fundstellen") or []
        }
        for fundstelle in entry.get("fundstellen") or []:
            now_state = (fundstelle.get("store"), fundstelle.get("store_entry_id"))
            if old_states.get(fundstelle["az"]) != now_state:
                changed_fundstellen += 1
                changed_gutachten.add(entry["id"])

    if changed_fundstellen:
        _create_revision(db, ASSESSMENT_TARGET, target, previous, new_content, [], "recheck")
        _write_target_content(ASSESSMENT_TARGET, target, new_content)
        target.search_text = render_case_assessment_compact(new_content)
        target.updated_at = datetime.utcnow()
        db.add(target)
        db.commit()

    return {
        "changed_fundstellen": changed_fundstellen,
        "changed_gutachten": len(changed_gutachten),
        "warnings": warnings,
    }
```

In `app/endpoints/agent_memory.py`:

Import the new target and helpers next to the existing `BRIEF_TARGET` import:

```python
from agent_memory_service import (
    ASSESSMENT_TARGET,
    BRIEF_TARGET,
    STRATEGY_TARGET,
    get_or_create_case_assessment,
    render_case_assessment_compact,
    ...
)

_KNOWN_TARGETS = {BRIEF_TARGET, STRATEGY_TARGET, ASSESSMENT_TARGET}
```

In the create-proposal route, validate the target before calling the service
(the local request model types it as a plain `str`):

```python
    if body.target_type not in _KNOWN_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unbekanntes Memory-Target: {body.target_type}",
        )
```

Extend `_combined_payload`:

```python
def _combined_payload(brief: Any, strategy: Any, assessment: Any) -> Dict[str, Any]:
    brief_content = brief.content_json or {}
    strategy_content = strategy.content_json or {}
    assessment_content = assessment.content_json or {}
    brief_payload = memory_row_to_dict(brief, render_case_brief_compact(brief_content))
    strategy_payload = memory_row_to_dict(strategy, render_case_strategy_compact(strategy_content))
    assessment_payload = memory_row_to_dict(
        assessment, render_case_assessment_compact(assessment_content)
    )
    for payload in (brief_payload, strategy_payload, assessment_payload):
        if payload.get("search_text") == payload.get("rendered"):
            payload.pop("search_text", None)
    return {
        "overview": brief_content.get("notizen", ""),
        "strategy": strategy_content.get("kernstrategie", ""),
        "memory": {
            "overview": brief_content.get("notizen", ""),
            "strategy": strategy_content.get("kernstrategie", ""),
        },
        "case_brief": brief_payload,
        "case_strategy": strategy_payload,
        "case_assessment": assessment_payload,
    }
```

Update every `_combined_payload(` call site (`grep -n "_combined_payload(" app/endpoints/agent_memory.py`)
to fetch and pass the assessment row via `get_or_create_case_assessment`.

Extend `_proposal_frontend_payload`:

```python
    if target_type == ASSESSMENT_TARGET:
        payload["section"] = "assessment"
        payload["title"] = "Gutachten-Vorschlag"
    elif target_type == STRATEGY_TARGET:
        payload["section"] = "strategy"
        payload["title"] = "Strategie-Vorschlag"
    else:
        payload["section"] = "overview"
        payload["title"] = "Fall-Überblick-Vorschlag"
```

and in its op loop, render Gutachten values as a summary line instead of raw
JSON:

```python
        if isinstance(value, dict) and "rechtsfrage" in value:
            count = len(value.get("fundstellen") or [])
            value = f"{value.get('id')} | {value.get('rechtsfrage')} | {count} Fundstellen"
```

In the accept route, unpack the tuple and attach the warnings:

```python
        proposal, assessment_warnings = accept_memory_update_proposal(
            db, current_user.id, proposal_id, actor="user", force=force
        )
    ...
    payload = _proposal_frontend_payload(proposal)
    if assessment_warnings:
        payload["assessment_warnings"] = assessment_warnings
    return payload
```

Add the recheck route next to the other case routes:

```python
@router.post("/cases/{case_id}/assessment/recheck")
@limiter.limit("60/hour")
async def recheck_case_assessment(
    request: Request,
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Store-Abgleich aller Gutachten-Fundstellen neu ausfuehren."""
    from assessment_memory import recheck_assessment

    target_case_id = resolve_case_uuid_for_request(db, current_user, case_id)
    _assert_owned_case(db, current_user, str(target_case_id))
    try:
        result = recheck_assessment(db, current_user.id, target_case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if result["changed_fundstellen"]:
        _notify_memory_changed(target_case_id, "assessment_recheck")
    return result
```

Also update the JS proposal list in `app/static/js/app.js` so an
`assessment_warnings` array on the accept response is rendered as a short
list under the proposal. Find the accept handler
(`grep -n "proposals/.*accept" app/static/js/app.js`) and after the existing
success handling add:

```javascript
      if (Array.isArray(data.assessment_warnings) && data.assessment_warnings.length) {
        const lines = data.assessment_warnings
          .map((w) => `${w.az || '?'}: ${w.store}`)
          .join('\n');
        showToast(`Gutachten-Fundstellen ohne Store-Treffer:\n${lines}`, 'warning');
      }
```

Match the existing toast helper's real name — read the surrounding code first.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_api.py -q`
Expected: PASS, 6 tests

Restart the app and smoke the route:

```bash
docker compose -f /var/opt/docker/rechtmaschine/docker-compose.yml restart rechtmaschine-app
CID=42431c8e-e3b4-4962-acc7-543dbbf43b26
T=$(cat ~/.config/rechtmaschine-cli/token)
curl -s -X POST "https://rechtmaschine.de/v1/memory/cases/$CID/assessment/recheck" \
  -H "Authorization: Bearer $T"
```

Expected: `{"changed_fundstellen":0,"changed_gutachten":0,"warnings":[]}`
(the case has no Gutachten yet). Confirm the memory GET now carries the third
block:

```bash
curl -s "https://rechtmaschine.de/v1/memory/cases/$CID" -H "Authorization: Bearer $T" \
  | python3 -c "import sys,json; print(list(json.load(sys.stdin).keys()))"
```

Expected: the key list includes `case_assessment`.

- [ ] **Step 5: Commit**

```bash
git add app/endpoints/agent_memory.py app/assessment_memory.py app/static/js/app.js tests/test_memory_assessment_api.py
git commit -m "feat(memory): API fuer case_assessment inkl. Recheck-Route und Accept-Warnungen

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 9: CLI support

**Files:**
- Modify: `scripts/rechtmaschine_cli.py` (`_MEMORY_SECTIONS`, `_memory_entries`, `cmd_memory_proposals_list`, `cmd_memory_proposals_accept`, new `memory assessment recheck` subcommand)
- Test: `tests/test_memory_assessment_cli.py`

**Interfaces:**
- Consumes: Task 8's API.
- Produces: `memory get --section assessment`, `--grep` over Gutachten text, `memory assessment recheck`, warning output on accept.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_assessment_cli.py`:

```python
"""CLI projection and subcommands for case_assessment.

    .venv/bin/python -m pytest tests/test_memory_assessment_cli.py -q
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rm_cli", REPO / "scripts" / "rechtmaschine_cli.py"
)
cli = importlib.util.module_from_spec(SPEC)
sys.modules["rm_cli"] = cli
SPEC.loader.exec_module(cli)


def _payload():
    return {
        "case_brief": {"content_json": {"verfahrensstand": ["Etwas passiert."], "notizen": ""}},
        "case_strategy": {"content_json": {"kernstrategie": "Linie A", "offene_fragen": []}},
        "case_assessment": {
            "content_json": {
                "gutachten": [
                    {
                        "id": "gueb-statt-duldung",
                        "rechtsfrage": "GUEB statt Duldung?",
                        "ergebnis": "Nein.",
                        "stand": "2026-09-03",
                        "status": "aktiv",
                        "pruefung": [
                            {"these": "Kein ungeregelter Aufenthalt", "bewertung": "Traegt.",
                             "fundstellen": ["18 E 491/12"]}
                        ],
                        "fundstellen": [
                            {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12",
                             "aussage": "Ersetzt die Duldung nicht.", "richtung": "pro",
                             "store": "verified"}
                        ],
                        "risiken": [],
                        "quelle": "Vermerk.pdf",
                    }
                ],
                "notizen": "",
            }
        },
    }


def test_assessment_section_is_known():
    assert cli._MEMORY_SECTIONS["assessment"] == "case_assessment"


def test_entries_flatten_gutachten_for_grep():
    entries = cli._memory_entries(_payload(), "assessment")
    texts = " ".join(e["text"] for e in entries)
    assert "GUEB statt Duldung?" in texts
    assert "18 E 491/12" in texts
    assert "Kein ungeregelter Aufenthalt" in texts


def test_grep_finds_an_aktenzeichen():
    entries = cli._memory_entries(_payload(), None)
    hits = [e for e in entries if "18 E 491/12" in e["text"]]
    assert hits and hits[0]["section"] == "assessment"


def test_proposal_summary_line_for_gutachten_ops():
    op = {
        "op": "append",
        "path": "/gutachten/-",
        "value": {"id": "a", "rechtsfrage": "F?", "fundstellen": [{"az": "1 A 1/24"}]},
    }
    assert cli._summarize_op(op) == "append /gutachten/-: a | F? | 1 Fundstellen"


def test_recheck_subcommand_is_registered():
    parser = cli.build_parser()
    args = parser.parse_args(["memory", "assessment", "recheck", "--case-id", "x"])
    assert args.func is cli.cmd_memory_assessment_recheck
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_cli.py -q`
Expected: FAIL — `_MEMORY_SECTIONS` has no `assessment` key.

If `build_parser` does not exist under that name, read the bottom of
`scripts/rechtmaschine_cli.py` and use the actual parser-construction function
in the test.

- [ ] **Step 3: Write minimal implementation**

In `scripts/rechtmaschine_cli.py`:

```python
_MEMORY_SECTIONS = {
    "brief": "case_brief",
    "strategy": "case_strategy",
    "assessment": "case_assessment",
}
```

In `_memory_entries`, add a Gutachten flattener so `--grep` can reach the
nested text:

```python
def _assessment_entry_texts(entry: Dict[str, Any]) -> list[str]:
    texts = [
        f"{entry.get('id')} | {entry.get('rechtsfrage', '')}",
        entry.get("ergebnis", ""),
    ]
    for punkt in entry.get("pruefung") or []:
        texts.append(f"{punkt.get('these', '')} – {punkt.get('bewertung', '')}")
    for fundstelle in entry.get("fundstellen") or []:
        texts.append(
            " ".join(
                str(fundstelle.get(key, ""))
                for key in ("gericht", "datum", "az", "aussage")
            ).strip()
        )
    texts.extend(entry.get("risiken") or [])
    return [t for t in texts if t.strip()]
```

and inside `_memory_entries`, when the field is `gutachten`, expand each list
item through `_assessment_entry_texts` instead of `str(value)`.

Extract the proposal op summary into a helper so the test can call it, and
teach it about Gutachten:

```python
def _summarize_op(op: Dict[str, Any]) -> str:
    path = op.get("path", "")
    value = op.get("value")
    if isinstance(value, dict) and "rechtsfrage" in value:
        count = len(value.get("fundstellen") or [])
        value = f"{value.get('id')} | {value.get('rechtsfrage')} | {count} Fundstellen"
    elif isinstance(value, dict):
        value = value.get("name") or value.get("label") or json.dumps(value, ensure_ascii=False)
    text = str(value if value is not None else "")
    return f"{op.get('op')} {path}: {text}"
```

Use `_summarize_op` in `cmd_memory_proposals_list` where the ops are currently
truncated inline.

In `cmd_memory_proposals_accept`, print the warnings when present:

```python
    data = _request_json("POST", args.base_url, f"/memory/proposals/{args.proposal_id}/accept",
                         token=token)
    _print(data)
    for warning in data.get("assessment_warnings") or []:
        print(
            f"⚠️  Fundstelle ohne Store-Treffer: {warning.get('az')} "
            f"({warning.get('store')}) in Gutachten {warning.get('gutachten_id')}",
            file=sys.stderr,
        )
```

Add the subcommand:

```python
def cmd_memory_assessment_recheck(args: argparse.Namespace) -> int:
    token = _load_token(Path(args.token_path).expanduser())
    case_id = _resolve_case_id(args)
    _print(
        _request_json(
            "POST",
            args.base_url,
            f"/memory/cases/{case_id}/assessment/recheck",
            token=token,
        )
    )
    return 0
```

and register it next to the other memory subcommands:

```python
    memory_assessment = memory_sub.add_parser(
        "assessment", help="Gutachten-Operationen (case_assessment)"
    )
    memory_assessment_sub = memory_assessment.add_subparsers(
        dest="assessment_command", required=True
    )
    memory_assessment_recheck = memory_assessment_sub.add_parser(
        "recheck", help="Store-Abgleich aller Gutachten-Fundstellen neu ausfuehren"
    )
    memory_assessment_recheck.add_argument("--case-id", help="Case UUID; defaults to the active case")
    memory_assessment_recheck.set_defaults(func=cmd_memory_assessment_recheck)
```

Match the real names of `_resolve_case_id` and `_load_token` in the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_memory_assessment_cli.py -q`
Expected: PASS, 5 tests

Live smoke:

```bash
~/kanzlei/skills/rechtmaschine/scripts/rechtmaschine-cli memory assessment recheck --case-id 42431c8e-e3b4-4962-acc7-543dbbf43b26
~/kanzlei/skills/rechtmaschine/scripts/rechtmaschine-cli memory get --case-id 42431c8e-e3b4-4962-acc7-543dbbf43b26 --section assessment
```

Expected: the recheck returns zero counters, the section prints an empty
Gutachten list.

- [ ] **Step 5: Commit**

```bash
git add scripts/rechtmaschine_cli.py tests/test_memory_assessment_cli.py
git commit -m "feat(cli): Gutachten-Sektion, grep-Projektion und assessment recheck

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 10: Pattern-wiki reads Gutachten first and enforces an Az whitelist

**Files:**
- Modify: `app/draft_citation_ingest.py` (span-yielding iterator)
- Modify: `app/endpoints/pattern_wiki.py` (`_execute_pattern_wiki_distillation`, `_DISTILL_RULES`, `_forbidden_tokens`, `PatternWikiSource` provenance)
- Modify: `app/assessment_memory.py` (wiki rendering, whitelist)
- Test: `tests/test_pattern_wiki_distill_assessment.py`

**Interfaces:**
- Consumes: Tasks 1-9.
- Produces: `iter_decision_citations(text) -> Iterator[dict]` in `draft_citation_ingest` (each dict has `court`, `kind`, `date`, `az`, `raw`, `start`, `end`); `render_assessment_for_wiki(content, max_chars=24000) -> str` and `verified_az_whitelist(content) -> set[str]` in `assessment_memory`; `strip_foreign_citations(text, whitelist) -> tuple[str, list[dict]]` in `pattern_wiki`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pattern_wiki_distill_assessment.py`:

```python
"""Assessment-first FALL-SPEICHER and the Az whitelist for wiki entries.

    .venv/bin/python -m pytest tests/test_pattern_wiki_distill_assessment.py -q
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from assessment_memory import (  # noqa: E402
    render_assessment_for_wiki,
    validate_assessment_content,
    verified_az_whitelist,
)
from draft_citation_ingest import iter_decision_citations  # noqa: E402


def _content():
    return validate_assessment_content(
        {
            "gutachten": [
                {
                    "id": "a",
                    "rechtsfrage": "GUEB statt Duldung?",
                    "ergebnis": "Nein.",
                    "stand": "2026-09-03",
                    "status": "aktiv",
                    "pruefung": [
                        {"these": "Kein ungeregelter Aufenthalt", "bewertung": "Traegt.",
                         "fundstellen": ["18 E 491/12"]}
                    ],
                    "fundstellen": [
                        {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12",
                         "art": "Beschluss", "aussage": "Ersetzt die Duldung nicht.",
                         "richtung": "pro", "store": "verified"},
                        {"gericht": "VG Y", "datum": "2020-01-01", "az": "7 L 7/20",
                         "art": "Beschluss", "aussage": "Unklar.", "richtung": "neutral",
                         "store": "not_in_store"},
                    ],
                    "risiken": [],
                    "quelle": "Vermerk.pdf",
                }
            ],
            "notizen": "",
        }
    )


def test_wiki_rendering_uses_the_parser_format_and_only_verified():
    text = render_assessment_for_wiki(_content())
    assert "OVG NRW, Beschluss vom 18.06.2012 – 18 E 491/12" in text
    assert "7 L 7/20" not in text
    parsed = list(iter_decision_citations(text))
    assert [p["az"] for p in parsed] == ["18 E 491/12"]


def test_whitelist_contains_only_verified_az():
    assert verified_az_whitelist(_content()) == {"18e491/12"}


def test_iter_yields_spans_that_slice_the_raw_citation():
    text = "Siehe OVG NRW, Beschluss vom 18.06.2012 – 18 E 491/12 dazu."
    hit = next(iter(iter_decision_citations(text)))
    assert text[hit["start"]:hit["end"]] == hit["raw"]
    assert hit["raw"].startswith("OVG NRW")


def test_strip_removes_foreign_citation_phrase_and_reports_it():
    from endpoints.pattern_wiki import strip_foreign_citations

    text = (
        "Argument traegt (OVG NRW, Beschluss vom 18.06.2012 – 18 E 491/12), "
        "anders VG Z, Urteil vom 01.02.2023 – 5 K 9/23."
    )
    cleaned, stripped = strip_foreign_citations(text, {"18e491/12"})
    assert "18 E 491/12" in cleaned
    assert "5 K 9/23" not in cleaned
    assert [s["az"] for s in stripped] == ["5 K 9/23"]


def test_strip_also_catches_a_bare_foreign_az():
    from endpoints.pattern_wiki import strip_foreign_citations

    cleaned, stripped = strip_foreign_citations("Vergleiche 5 K 9/23 hierzu.", {"18e491/12"})
    assert "5 K 9/23" not in cleaned
    assert stripped and stripped[0]["az"] == "5 K 9/23"


def test_whitelisted_bare_az_survives():
    from endpoints.pattern_wiki import strip_foreign_citations

    cleaned, stripped = strip_foreign_citations("Vergleiche 18 E 491/12 hierzu.", {"18e491/12"})
    assert "18 E 491/12" in cleaned
    assert stripped == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pattern_wiki_distill_assessment.py -q`
Expected: FAIL with `ImportError: cannot import name 'render_assessment_for_wiki'`

- [ ] **Step 3: Write minimal implementation**

In `app/draft_citation_ingest.py`, add the span-yielding iterator and make
`parse_decision_citations` use it, so both share one regex pass:

```python
def iter_decision_citations(text: str):
    """Yield each citation with its raw text and span, so callers can cut the
    whole phrase out instead of only replacing the Aktenzeichen."""
    for match in DECISION_RE.finditer(text or ""):
        data = match.groupdict()
        data["raw"] = match.group(0)
        data["start"] = match.start()
        data["end"] = match.end()
        yield data
```

Keep `parse_decision_citations` behaviourally identical by building its list
from `iter_decision_citations`.

Append to `app/assessment_memory.py`:

```python
WIKI_MAX_CHARS = 24_000


def verified_az_whitelist(content: Dict[str, Any]) -> set:
    """Normalized Az of every verified Fundstelle in active Gutachten."""
    from verify_source import az_for_compare

    assessment = validate_assessment_content(content)
    return {
        az_for_compare(f["az"])
        for entry in assessment["gutachten"]
        if entry.get("status") == "aktiv"
        for f in entry.get("fundstellen") or []
        if f.get("store") == "verified"
    }


def render_assessment_for_wiki(content: Dict[str, Any], max_chars: int = WIKI_MAX_CHARS) -> str:
    """Full-detail rendering for the wiki distillation, verified citations only,
    in the format the citation parser understands."""
    assessment = validate_assessment_content(content)
    active = [e for e in assessment["gutachten"] if e.get("status") == "aktiv"]
    if not active:
        return ""
    active.sort(key=lambda e: e["stand"], reverse=True)

    blocks: List[str] = []
    for entry in active:
        verified = {
            f["az"]: f
            for f in entry.get("fundstellen") or []
            if f.get("store") == "verified"
        }
        lines = [
            f"GUTACHTEN {entry['id']} (Stand {_de_date(entry['stand'])})",
            f"Rechtsfrage: {entry['rechtsfrage']}",
            f"Ergebnis: {entry['ergebnis']}",
        ]
        for punkt in entry.get("pruefung") or []:
            lines.append(f"- These: {punkt['these']}")
            lines.append(f"  Bewertung: {punkt['bewertung']}")
            for az in punkt.get("fundstellen") or []:
                f = verified.get(az)
                if not f:
                    continue
                art = f.get("art") or "Beschluss"
                lines.append(
                    f"  Fundstelle ({f['richtung']}): {f['gericht']}, {art} vom "
                    f"{_de_date(f['datum'])} – {f['az']} — {f['aussage']}"
                )
        if entry.get("risiken"):
            lines.append("Risiken: " + "; ".join(entry["risiken"]))
        blocks.append("\n".join(lines))

    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        kept: List[str] = []
        size = 0
        for block in blocks:
            if size + len(block) > max_chars:
                break
            kept.append(block)
            size += len(block) + 2
        text = "\n\n".join(kept)
        text += f"\n\n[weitere Gutachten gekuerzt: {len(blocks) - len(kept)}]"
    return text
```

In `app/endpoints/pattern_wiki.py`, add the stripper:

```python
_BARE_AZ_RE = re.compile(r"\b\d{1,3}\s+[A-Za-z]{1,3}\s+\d+[./]\d+(?:\.[A-Z]{1,2})?\b")


def strip_foreign_citations(text: str, whitelist: set) -> tuple:
    """Remove citations whose Az is not in the whitelist. Returns (text, stripped).

    Two passes: full citations (phrase removed via span) and bare Aktenzeichen
    that the strict parser does not match, so a hallucinated naked Az cannot
    slip through."""
    from draft_citation_ingest import iter_decision_citations
    from verify_source import az_for_compare

    stripped: List[Dict[str, str]] = []
    spans: List[tuple] = []
    protected: List[tuple] = []
    for hit in iter_decision_citations(text):
        if az_for_compare(hit["az"]) in whitelist:
            protected.append((hit["start"], hit["end"]))
            continue
        spans.append((hit["start"], hit["end"]))
        stripped.append({"az": hit["az"], "citation": hit["raw"]})

    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]

    def _drop_bare(match: re.Match) -> str:
        if az_for_compare(match.group(0)) in whitelist:
            return match.group(0)
        stripped.append({"az": match.group(0), "citation": match.group(0)})
        return ""

    text = _BARE_AZ_RE.sub(_drop_bare, text)
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(), stripped
```

In `_execute_pattern_wiki_distillation`, put the assessment first:

```python
    from agent_memory_service import get_or_create_case_assessment
    from assessment_memory import render_assessment_for_wiki, verified_az_whitelist

    assessment = get_or_create_case_assessment(db, current_user.id, target_case_id)
    assessment_content = assessment.content_json or {}
    assessment_block = render_assessment_for_wiki(assessment_content)
    whitelist = verified_az_whitelist(assessment_content)

    memory_block = "\n\n".join(
        part
        for part in (
            assessment_block,
            render_case_brief_compact(brief_content),
            render_case_strategy_compact(strategy_content),
        )
        if part
    )
```

Extend `_DISTILL_RULES` with two sentences:

```python
_DISTILL_RULES = """...existing text...

Fundstellen: Nenne die tragende Entscheidung im Format "Gericht, Urteil oder
Beschluss vom TT.MM.JJJJ - Az" direkt im Argumentationsmuster. Verwende
ausschliesslich Fundstellen, die im FALL-SPEICHER stehen, niemals eigene."""
```

After the extraction and before `_entry_violations`, run the whitelist over
every text field of each entry:

```python
    stripped_citations: List[Dict[str, str]] = []
    if whitelist or assessment_block:
        for entry in parsed_entries:
            entry.summary, removed = strip_foreign_citations(entry.summary or "", whitelist)
            for item in removed:
                item["entry_title"] = entry.title
            stripped_citations.extend(removed)
            for field in ("argument_patterns", "risk_patterns", "evidence_patterns",
                          "recommended_next_steps"):
                cleaned_list = []
                for value in getattr(entry, field) or []:
                    cleaned, removed = strip_foreign_citations(value, whitelist)
                    for item in removed:
                        item["entry_title"] = entry.title
                    stripped_citations.extend(removed)
                    if cleaned:
                        cleaned_list.append(cleaned)
                setattr(entry, field, cleaned_list)
```

Use the real variable name the function uses for the parsed entries — read the
lines after `parsed = await call_qwen_json(...)` first.

Report them in the job result and the warnings:

```python
    result = {
        "created": len(created),
        "trigger": "pattern_wiki",
        "stripped_citations": stripped_citations,
    }
    if stripped_citations:
        result.setdefault("warnings", []).append(
            f"{len(stripped_citations)} Fundstelle(n) ausserhalb des Gutachtens entfernt"
        )
```

Merge this into the existing return dict rather than replacing it.

Fix the provenance where `PatternWikiSource` is created:

```python
            PatternWikiSource(
                pattern_wiki_entry_id=row.id,
                source_type="case_assessment" if assessment_block else "case_brief",
                source_id=str(target_case_id),
                label=(
                    "Gutachten + Brief + Strategie" if assessment_block
                    else "Brief + Strategie"
                ),
                ...
```

Keep the remaining keyword arguments of the existing call unchanged.

Finally, keep the PII gate consistent: in `_forbidden_tokens`, exclude the
whitelisted Aktenzeichen and the Fundstellen dates from the token set, since
they are legitimate wiki vocabulary:

```python
def _forbidden_tokens(
    case: Case,
    brief_content: Dict[str, Any],
    strategy_content: Dict[str, Any],
    allowed_az: Optional[set] = None,
) -> set:
    ...
    if allowed_az:
        from verify_source import az_for_compare

        tokens = {t for t in tokens if az_for_compare(t) not in allowed_az}
    return tokens
```

and pass `allowed_az=whitelist` at the call site.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pattern_wiki_distill_assessment.py -q`
Expected: PASS, 6 tests

Run: `.venv/bin/python -m pytest tests/ -q -m "not slow"`
Expected: no regressions, especially in any existing pattern-wiki tests.

- [ ] **Step 5: Commit**

```bash
git add app/draft_citation_ingest.py app/endpoints/pattern_wiki.py app/assessment_memory.py tests/test_pattern_wiki_distill_assessment.py
git commit -m "feat(wiki): Gutachten als Primaerquelle der Musterableitung mit Az-Whitelist

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 11: Triage verdict, activity event and stop hook

**Files:**
- Modify: `~/kanzlei/skills/rechtmaschine/scripts/memory_triage.py`
- Modify: `~/kanzlei/skills/api/scripts/jlawyer_cli.py` (upload command)
- Modify: `~/kanzlei/skills/rechtmaschine/scripts/memory_hygiene_hook.py`
- Test: `~/kanzlei/skills/rechtmaschine/tests/test_memory_triage_assessment.py`

**Interfaces:**
- Consumes: Task 8's proposal payloads.
- Produces: verdict `SESSION` for every `case_assessment` proposal; activity event `kind="vermerk-upload"`; hook line `📚 Vermerk ohne Gutachten-Proposal: <Datei>`.

Note: these files live in the shared skills repo (`~/kanzlei/skills`, canonical
per the skills memory), not in the Rechtmaschine repo. Claim
`tooling-rechtmaschine-memory-assessment` before editing, commit in that repo
separately.

- [ ] **Step 1: Write the failing test**

Create `~/kanzlei/skills/rechtmaschine/tests/test_memory_triage_assessment.py`:

```python
"""Assessment proposals are always SESSION, never auto-accepted.

    python3 ~/kanzlei/skills/rechtmaschine/tests/test_memory_triage_assessment.py
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import memory_triage as mt


def _proposal(target_type, model="qwen3.6-27b", ops=None):
    return {
        "id": "p1",
        "status": "pending",
        "target_type": target_type,
        "expected_version": 1,
        "model": model,
        "created_at": "2026-09-04T10:00:00",
        "ops": ops
        or [
            {
                "op": "append",
                "path": "/gutachten/-",
                "value": {"id": "a", "rechtsfrage": "F?", "fundstellen": []},
            }
        ],
    }


def main() -> int:
    versions = {"case_assessment": 1, "case_brief": 1, "case_strategy": 1}

    verdict = mt.classify(_proposal("case_assessment"), [], {}, versions)
    assert verdict.action == "hold", verdict
    assert verdict.kind == "SESSION", verdict

    # even a claude-authored one is SESSION, never FACT
    verdict = mt.classify(_proposal("case_assessment", model="claude"), [], {}, versions)
    assert verdict.kind == "SESSION", verdict

    # brief proposals keep their existing behaviour
    brief = _proposal(
        "case_brief",
        ops=[{"op": "append", "path": "/sachverhalt/-", "value": "Neue Tatsache."}],
    )
    verdict = mt.classify(brief, [], {"sachverhalt": []}, versions)
    assert verdict.kind in {"FACT", "DUPLICATE", "EVENT"}, verdict

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Adjust the call signature of `mt.classify` to whatever the module actually
exposes — read `memory_triage.py` around line 267 first and mirror it.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 ~/kanzlei/skills/rechtmaschine/tests/test_memory_triage_assessment.py`
Expected: FAIL on the first assertion — an assessment proposal currently falls
through to FACT because `_op_values` reads only `name` from a dict.

- [ ] **Step 3: Write minimal implementation**

In `memory_triage.py`, add the target check as the very first rule inside the
classifier, before any op inspection:

```python
ASSESSMENT_TARGET = "case_assessment"

    # Gutachten werden immer von der Session entschieden: die Fundstellen
    # muessen am Store geprueft werden, und _op_values liest aus Dicts nur
    # `name` — ein Gutachten-Objekt saehe sonst wie ein harmloser FACT aus.
    if (proposal.get("target_type") or "") == ASSESSMENT_TARGET:
        return Verdict(
            kind="SESSION",
            action="hold",
            reason=(
                "Gutachten: die Session prueft Fundstellen und Bewertung selbst "
                "(rechtmaschine-memory-Skill, Abschnitt Gutachten)"
            ),
        )
```

Use the module's real `Verdict` constructor shape.

In the same file, extend the daily `--apply` path so it calls the recheck for
every own case that has at least one active Gutachten:

```python
def recheck_assessments(case_ids: list[str]) -> None:
    for case_id in case_ids:
        subprocess.run(
            [RM_CLI, "memory", "assessment", "recheck", "--case-id", case_id],
            capture_output=True,
            text=True,
            timeout=120,
        )
```

and call it after the proposal loop when `--apply` is set, skipping foreign
cases exactly as the existing loop does.

In `~/kanzlei/skills/api/scripts/jlawyer_cli.py`, emit the activity event
after a successful upload:

```python
def _emit_vermerk_event(case_ref: str, filename: str) -> None:
    """Feed the stop hook: a Vermerk upload without a Gutachten proposal is
    a reminder-worthy gap. Best effort, never fails the upload."""
    if not Path(filename).name.startswith("Vermerk_"):
        return
    try:
        sys.path.insert(0, str(Path.home() / "kanzlei" / "skills" / "claims" / "scripts"))
        from activity_feed import log_event

        log_event(kind="vermerk-upload", key=case_ref, note=Path(filename).name)
    except Exception:
        pass
```

Read `~/kanzlei/skills/claims/scripts/activity_feed.py` for the real function
name and signature before writing this, then call `_emit_vermerk_event` at the
end of the upload command.

In `memory_hygiene_hook.py`, add a second query alongside the existing
`claim-release` one:

```python
            "SELECT id, key, note FROM events WHERE id > ? AND pane = ? AND kind = 'vermerk-upload'",
```

For each such event, fetch the case's pending proposals through the same CLI
the hook already uses, and if none has `target_type == "case_assessment"` with
a `created_at` after the session baseline, append:

```python
            lines.append(
                f"📚 Vermerk ohne Gutachten-Proposal: {note} ({key}) — "
                "rechtliche Wuerdigung als case_assessment-Proposal schreiben"
            )
```

Respect the existing `MEMORY_HYGIENE_REVIEW_BUDGET` time budget: skip the
check when the budget is already spent.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 ~/kanzlei/skills/rechtmaschine/tests/test_memory_triage_assessment.py`
Expected: `ok`

Dry-run the triage against the live case:

```bash
python3 ~/kanzlei/skills/rechtmaschine/scripts/memory_triage.py --az 157/26
```

Expected: no crash, and any assessment proposal is listed as SESSION.

- [ ] **Step 5: Commit**

```bash
cd ~/kanzlei/skills && git add rechtmaschine/scripts/memory_triage.py \
  rechtmaschine/scripts/memory_hygiene_hook.py \
  rechtmaschine/tests/test_memory_triage_assessment.py \
  api/scripts/jlawyer_cli.py
git commit -m "feat(memory): Gutachten-Proposals immer SESSION, Vermerk-Erinnerung im Stop-Hook

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

---

### Task 12: Skill documentation and live acceptance on 157/26

**Files:**
- Modify: `~/kanzlei/skills/rechtmaschine-memory/SKILL.md`
- Create: `/tmp/.../gutachten_gueb.json`, `/tmp/.../gutachten_heirat.json` (scratchpad payloads, not committed)

**Interfaces:**
- Consumes: everything.
- Produces: the "Gutachten" section in the skill, and two accepted Gutachten in case 157/26 as the acceptance evidence.

- [ ] **Step 1: Write the skill section**

Add to `~/kanzlei/skills/rechtmaschine-memory/SKILL.md`, after the
"Reviewable Proposal" section:

````markdown
## Gutachten (`case_assessment`, seit 04.09.2026)

Drittes Memory-Target neben Brief und Strategie. Haelt je Rechtsfrage ein
strukturiertes Gutachten mit Fundstellen. **Reflect und Consolidate fassen es
nicht an** — es schreibt ausschliesslich die Session, die die Recherche
gemacht hat.

Ablauf nach einer Rechtsrecherche:

1. Vermerk wie bisher als PDF in die j-lawyer-Akte (bleibt Pflicht, ist das
   lesbare Langdokument).
2. Gutachten als Proposal schreiben, `target_type: "case_assessment"`.
3. `memory proposals accept <id>` — der Server gleicht jede Fundstelle gegen
   den Rechtsprechungsstore ab und gibt `assessment_warnings` zurueck.
4. Warnungen lesen. Fundstellen mit `not_in_store` werden automatisch zur
   Beschaffung angestossen (`draft_citation_ingest`), fehlende Entscheidungen
   notfalls per `cited_ingest.py` selbst aufnehmen.
5. `memory assessment recheck --case-id ...` setzt die Store-Felder neu.

Schema je Gutachten: `id` (Slug), `rechtsfrage`, `ergebnis`, `stand` (ISO),
`status` (`aktiv`/`ueberholt`), `pruefung` (These, Bewertung, Az-Verweise),
`fundstellen` (Gericht, Datum ISO, Az, `art`, Aussage, `richtung`),
`risiken`, `quelle` (Vermerk-Dateiname).

**Konvention: keine Namen von Beteiligten im Gutachten**, nur Rollen ("der
Mandant"). Der Block geht in Cloud-Prompts, die Pseudonymisierung ist nur
Sicherheitsnetz.

Patch-Ops adressieren **per id, nie per Index**:

```
append /gutachten/-
set    /gutachten/by-id/gueb-statt-duldung
remove /gutachten/by-id/gueb-statt-duldung
```

`store`, `store_entry_id` und `store_checked_at` setzt ausschliesslich der
Server. Mitgelieferte Werte werden verworfen.

**`verified` heisst nur: Az und Datum passen zu einem aktiven Store-Eintrag.**
Gericht und tragende Aussage sind damit NICHT geprueft. Nur `verified`-Fundstellen
gehen zitierfaehig in den Prompt, alle anderen stehen dort in der Sperrliste
"Nicht zitierfaehig" und loesen im Faktencheck eine Warnung aus, wenn sie
trotzdem im Entwurf auftauchen.

Beispiel-Payload:

```json
{
  "target_type": "case_assessment",
  "expected_version": 1,
  "ops": [
    {"op": "append", "path": "/gutachten/-", "value": {
      "id": "gueb-statt-duldung",
      "rechtsfrage": "Darf die ABH nach Passvorlage statt einer Duldung nur eine GUEB ausstellen?",
      "ergebnis": "Nein, sobald der Abschiebungszeitpunkt ungewiss ist.",
      "stand": "2026-09-03",
      "status": "aktiv",
      "pruefung": [
        {"these": "Kein Raum fuer ungeregelten Aufenthalt",
         "bewertung": "Traegt, solange die Behoerde keine Prognose belegt.",
         "fundstellen": ["18 E 491/12"]}
      ],
      "fundstellen": [
        {"gericht": "OVG NRW", "datum": "2012-06-18", "az": "18 E 491/12",
         "art": "Beschluss", "aussage": "Die GUEB ersetzt die Duldung nicht.",
         "richtung": "pro"}
      ],
      "risiken": ["Behoerde belegt eine zeitnahe Abschiebung."],
      "quelle": "Vermerk_2026-09-02_Recherche_GUEB_statt_Duldung.pdf"
    }}
  ],
  "source_refs": [{"source_type": "document", "label": "Vermerk 02.09.2026",
                   "excerpt": "OVG NRW 18 E 491/12", "metadata": {"origin": "claude"}}],
  "confidence": 0.9,
  "model": "claude"
}
```
````

- [ ] **Step 2: Commit the skill change**

```bash
cd ~/kanzlei/skills && git add rechtmaschine-memory/SKILL.md
git commit -m "docs(memory): Gutachten-Abschnitt im rechtmaschine-memory-Skill

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph"
```

- [ ] **Step 3: Write the two real Gutachten for 157/26**

Build both payloads from the two Vermerke already in the Akte
(`Vermerk_2026-09-02_Recherche_GÜB_statt_Duldung.pdf` and
`Vermerk_2026-09-03_Externe_Recherche_GÜB_statt_Duldung.pdf`, plus
`Vermerk_2026-09-03_Vorbereitung_Telefonat_09.09._Heirat_Eilantrag.pdf`).

Gutachten 1, `id: "gueb-statt-duldung"`: Fundstellen OVG NRW 18 E 491/12
(2012-06-18, pro), OVG NRW 18 B 103/23 (2023-02-10, pro), VG Düsseldorf
8 L 1143/24 (2025-04-25, pro), VG Gelsenkirchen 8 L 1526/25 (2025-11-14, pro),
BayVGH 10 CE 21.1427 (2021-08-02, pro), Hess. VGH 3 B 478/25 (2025-04-10,
contra), OVG Hamburg 6 Bs 176/25 (2026-02-03, contra), VG Köln 5 L 87/24
(2024-02-29, neutral), VG München M 10 K 21.3767 (2022-03-17, contra).

Gutachten 2, `id: "heirat-und-visumverfahren"`: Fundstellen VG München
M 12 E 21.6201 (2022-04-21, pro), BayVGH 10 ZB 22.1187 (2022-09-07, neutral),
VG Köln 27 L 1491/24.A (2024-09-10, contra), VGH Hessen 3 B 2020/22
(2023-09-15, contra), OVG SH 6 MB 37/25 (2025-12-16, contra), OVG LSA
2 M 64/25 (2025-08-11, pro).

```bash
CLI=~/kanzlei/skills/rechtmaschine/scripts/rechtmaschine-cli
CID=42431c8e-e3b4-4962-acc7-543dbbf43b26
$CLI memory get --case-id $CID --versions   # expected_version fuer beide Payloads
$CLI memory proposals create --case-id $CID --payload-file /tmp/.../gutachten_gueb.json
$CLI memory proposals accept <id>
```

- [ ] **Step 4: Verify the acceptance criteria**

```bash
CLI=~/kanzlei/skills/rechtmaschine/scripts/rechtmaschine-cli
CID=42431c8e-e3b4-4962-acc7-543dbbf43b26

# a) both Gutachten present, citations reconciled
$CLI memory get --case-id $CID --section assessment | head -40
$CLI memory get --case-id $CID --grep '18 E 491/12'

# b) recheck is idempotent
$CLI memory assessment recheck --case-id $CID

# c) the prompt block reaches generation
$CLI draft-context --case-id $CID --prompt "Eilantrag GUEB statt Duldung" \
  | grep -c "RECHTLICHE WUERDIGUNG"

# d) wiki distillation carries citations and the new provenance
$CLI wiki distill --case-id $CID --wait
```

Expected: (a) both Gutachten with `store: "verified"` on the nine decisions
that Task 12's session ingested on 03.09.2026, (b) `changed_fundstellen: 0` on
the second run, (c) the grep count is 1, (d) the job result carries
`stripped_citations` (possibly empty) and the created entries name real
Aktenzeichen.

- [ ] **Step 5: Report and release**

Report to Jay: which Gutachten were accepted, which Fundstellen came back as
`not_in_store` and whether the automatic procurement resolved them. Then:

```bash
claim release tooling-rechtmaschine-memory-assessment
```

---

## Self-Review Notes

Spec coverage checked section by section:

- Spec 1 (Datenmodell) → Tasks 1, 2, 3.
- Spec 1 (Register) → Task 4.
- Spec 2 (Store-Abgleich, Accept, Recheck) → Tasks 5, 8.
- Spec 3 (Injektion, Budget, Fakten-Grounding) → Task 6.
- Spec 1 (Rebase) → Task 7.
- Spec 4 (API, CLI, Triage, Hook, Skill, Reflect, Wiki) → Tasks 8, 9, 10, 11, 12.
- Spec 5 (Fehlerfälle) → covered by the tests in Tasks 1, 2, 5, 6, 7.
- Spec 6 (Tests) → one test file per task.
- Spec 7 (Abnahme) → Task 12.

Known gaps deliberately left open, matching the spec's "Nicht in diesem
Schritt": no Qwen-authored Gutachten, no `verify_claim` at accept, no
server-side enforcement of session authorship, no retroactive Gutachten for old
cases, no numbered revisions.
