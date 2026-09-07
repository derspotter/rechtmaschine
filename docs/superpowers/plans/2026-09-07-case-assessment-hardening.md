# case_assessment Härtung — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Die zehn "Important"-Befunde des Codex-Reviews vom 07.09.2026 an der case_assessment-Branch schließen, mit der jeweils kleinsten Maßnahme.

**Architecture:** Ein neues import-leichtes Modul `app/citation_identity.py` liefert die eine Zitat-Erkennung (`find_citations`) und die eine Az-Normalisierung (`canonical_az`); Wiki-Stripping, Wiki-Sperrprüfung, Fakten-Check und Whitelist benutzen nur noch dieses Modul. Proposal-Rebase wird alles-oder-nichts, Server-Felder werden bei jeder berührten id neu abgeglichen, die Größenmessung ignoriert Server-Felder. Auf der Skills-Seite: unbekannter Eigentümer = überspringen, Zeitbudget in die Subprozesse, Recheck-Bilanz im Nightly.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, Pydantic v2, pytest; Skills-Repo `~/kanzlei/skills` (reines Python + pytest).

**Spec:** `docs/superpowers/specs/2026-09-07-case-assessment-hardening-design.md` (bindend). Review als Quelle der Befunde: `docs/superpowers/specs/2026-09-07-case-assessment-branch-review-codex.md`.

## Global Constraints

- Kleinste Maßnahme je Befund; keine neuen Endpunkte, keine neuen Tabellen, keine Migration.
- `app/citation_identity.py` importiert nichts aus `endpoints`, `database`, `models`, `verify_source`, `jurisprudence_ingest`.
- Neue deutsche Strings mit echten Umlauten. Aufzählungen im Fließtext mit ", " verbinden, nie "; ".
- Tests, die `sys.modules`-Stubs setzen, stellen sie nach dem Import wieder her (`monkeypatch.setitem(sys.modules, ...)`), siehe `tests/conftest.py`.
- Vor jedem Commit: `.venv/bin/python -m pytest tests -q` grün (Stand vor Plan: 568 passed, 4 skipped). Skills-Repo: `cd ~/kanzlei/skills && python3 -m pytest rechtmaschine/tests -q`.
- Commit-Trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` und `Claude-Session: https://claude.ai/code/session_01LN4zQDb3bekWdy2BjZA1ph`. Nur explizite Pfade stagen, nie `git add -A`.
- Arbeit im Worktree `.worktrees/case-assessment-hardening` auf Branch `feat/case-assessment-hardening` (Basis `master`); Prod-Container werden aus dem Worktree NICHT neu gestartet.

---

### Task 1: `citation_identity` — eine Zitat-Erkennung, eine Az-Normalisierung

**Files:**
- Create: `app/citation_identity.py`
- Modify: `app/jurisprudence_ingest.py:236-254` (`_AZ_COURT_PREFIX`, `_az_for_compare` → Delegation)
- Modify: `app/draft_citation_ingest.py:32-39` (`DECISION_RE` aus dem Modul beziehen)
- Test: `tests/test_citation_identity.py`

**Interfaces:**
- Produces: `canonical_az(az: str) -> str`; `find_citations(text: str) -> list[Citation]`; `DECISION_RE`, `AZ_RE`, `NORM_RE`; `@dataclass Citation(kind, start, end, raw, az, canonical, date=None, az_start=0, az_end=0)` mit `kind in {"decision", "az"}`.
- `verify_source.az_for_compare` und `jurisprudence_ingest._az_for_compare` liefern weiterhin dieselben Werte (Delegation).

- [ ] **Step 1: Failing tests**

```python
# tests/test_citation_identity.py
"""citation_identity: eine Zitat-Erkennung fuer Sperren, Whitelist, Fakten-Check.

    .venv/bin/python -m pytest tests/test_citation_identity.py -q
"""
import pytest

from citation_identity import Citation, canonical_az, find_citations


@pytest.mark.parametrize("raw, expected", [
    ("18 E 491/12", "18e491/12"),
    ("OVG NRW 18 E 491/12", "18e491/12"),
    ("M 10 K 21.3767", "m10k21.3767"),
    ("VG 18 B 103/23", "18b103/23"),
    ("C-151/22 [Changu]", "c-151/22"),
    ("27 L 1491/24.A", "27l1491/24.a"),
    ("18 E 491/12 - Mustermann", "18e491/12"),
])
def test_canonical_az_matches_previous_normalization(raw, expected):
    assert canonical_az(raw) == expected


def test_find_citations_full_decision_with_bavarian_az():
    text = "Trägt (VGH Bayern, Beschluss vom 07.09.2022 – 10 ZB 22.1187)."
    hits = find_citations(text)
    assert [h.kind for h in hits] == ["decision"]
    hit = hits[0]
    assert hit.az == "10 ZB 22.1187" and hit.canonical == "10zb22.1187"
    assert hit.date == "07.09.2022"
    assert text[hit.az_start:hit.az_end] == "10 ZB 22.1187"
    assert text[hit.start:hit.end].startswith("VGH Bayern")


def test_find_citations_bare_forms():
    text = "Vgl. M 10 K 21.3767, 18E491/12, C-151/22 und EGMR Nr. 12345/19."
    assert [(h.kind, h.canonical) for h in find_citations(text)] == [
        ("az", "m10k21.3767"), ("az", "18e491/12"), ("az", "c-151/22"), ("az", "nr.12345/19"),
    ]


def test_find_citations_ignores_eu_norms_and_statutes():
    text = ("Anspruch aus Art. 14 Abs. 2 RL 2008/115/EG, Art. 3 RL 2011/95, "
            "VO (EU) Nr. 604/2013 und § 60a Abs. 2 S. 1 AufenthG.")
    assert find_citations(text) == []


def test_find_citations_prefix_belongs_to_az_not_to_prose():
    hits = find_citations("Az 5 K 9/23 wurde zitiert.")
    assert len(hits) == 1 and hits[0].raw == "5 K 9/23"


def test_find_citations_decision_span_excludes_names_from_az_span():
    text = "VG Teststadt, Frau Mustermann, Urteil vom 01.02.2020 – 18 E 491/12"
    hit = find_citations(text)[0]
    assert hit.kind == "decision"
    assert "Mustermann" not in text[hit.az_start:hit.az_end]
```

- [ ] **Step 2: Run, expect ImportError**

Run: `.venv/bin/python -m pytest tests/test_citation_identity.py -q`

- [ ] **Step 3: Implement the module**

```python
# app/citation_identity.py
"""Eine Zitat-Identität für alle Verbraucher (Wiki-Stripping, Wiki-Sperrprüfung,
Fakten-Check, Gutachten-Whitelist). Import-leicht: keine DB, keine endpoints.

`canonical_az` ist die frühere `jurisprudence_ingest._az_for_compare`
(verbatim übernommen, dort und in `verify_source` nur noch delegiert)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# --- Normalisierung (verbatim aus jurisprudence_ingest, Stand 07.09.2026) ----
_AZ_COURT_PREFIX = re.compile(
    r"^(?:VG|OVG|VGH|BayVGH|BVerwG|BVerfG|BSG|LSG|SG|BGH|OLG|LG|AG|BFH|FG|EuGH|EGMR)\b[.\s]*",
    re.IGNORECASE,
)


def canonical_az(az: Optional[str]) -> str:
    """Normalize an Az, dropping known-benign format differences (court
    prefix/suffix, [nickname], appended party names, unicode dashes, journal
    citations, internal whitespace) so only substantive discrepancies remain."""
    az = (az or "").translate(str.maketrans({"‑": "-", "–": "-", "—": "-"}))
    az = re.split(r"\s+-\s+", az)[0]
    az = re.sub(r"\s*\[[^\]]*\]", "", az)
    az = re.sub(r"\s*\([^)]*\)", "", az)
    az = _AZ_COURT_PREFIX.sub("", az.strip())
    az = re.sub(r"[\s.]*\b(OVG|VG|VGH)$", "", az.strip())
    return re.sub(r"\s+", "", az).casefold()


# --- Formen -------------------------------------------------------------------
# Az-Formen: EuGH, EGMR (nur mit "Nr."), deutsch mit optionalem ein- bis
# zweibuchstabigem Gerichtspräfix (bayerische VG), kompakt ohne Leerzeichen.
_AZ_FORMS = (
    r"C-\d{1,4}/\d{2}",
    r"Nr\.\s?\d{3,6}/\d{2}",
    r"(?:\b[A-Z][A-Za-z]?\s)?\d{1,3}\s[A-Za-z]{1,3}\s\d+[./]\d+(?:\.[A-Z]{1,2})?",
    r"\d{1,3}[A-Z]{1,3}\d{1,5}/\d{2}",
)
AZ_RE = re.compile(r"(?<![\w/.-])(?:" + "|".join(_AZ_FORMS) + r")(?![\w/])")

# EU-Normen, in denen "2 RL 2008/115" wie ein Az aussieht. Optionaler Vorspann
# "Art. 14 Abs. 2" wird mitgesperrt.
NORM_RE = re.compile(
    r"(?:Art\.?\s?\d+[a-z]?(?:\s?(?:Abs\.|Satz|S\.|Nr\.|Buchst\.|lit\.)\s?\w+\.?)*\s?)?"
    r"(?:RL|VO|Richtlinie|Verordnung)\s?(?:\((?:EG|EU|EWG)\)\s?)?(?:Nr\.\s?)?"
    r"\d{2,4}/\d{1,4}(?:/(?:EG|EU|EWG))?"
)

DECISION_RE = re.compile(
    r"(?P<court>BVerwG|BVerfG|EuGH|EGMR|BGH|BSG|BAG|"
    r"(?:OVG|VGH|VG|LSG|SG|LG|AG)\s+[A-ZÄÖÜ][\wäöüß.-]*(?:\s+[A-ZÄÖÜ][\wäöüß.-]*)?)"
    r"[^();\n]{0,80}?"
    r"(?P<kind>Urteil|Beschluss|Gerichtsbescheid|Urt\.|Beschl\.)\s+(?:vom|v\.)\s+"
    r"(?P<date>\d{1,2}\.\d{1,2}\.\d{4})\s*[–—-]\s*"
    r"(?P<az>" + "|".join(_AZ_FORMS) + r")"
)


@dataclass
class Citation:
    kind: str  # "decision" | "az"
    start: int
    end: int
    raw: str
    az: str
    canonical: str
    date: Optional[str] = None
    az_start: int = 0
    az_end: int = 0


def _overlaps(start: int, end: int, spans: List[tuple]) -> bool:
    return any(s < end and start < e for s, e in spans)


def find_citations(text: str) -> List[Citation]:
    """Alle Zitate im Text, nach Position. Normspannen (EU-Recht) sind für die
    Az-Suche gesperrt, Vollzitate haben Vorrang vor nackten Az."""
    text = text or ""
    blocked = [m.span() for m in NORM_RE.finditer(text)]
    hits: List[Citation] = []
    for m in DECISION_RE.finditer(text):
        if _overlaps(*m.span(), blocked):
            continue
        hits.append(Citation(
            kind="decision", start=m.start(), end=m.end(), raw=m.group(0),
            az=m.group("az"), canonical=canonical_az(m.group("az")),
            date=m.group("date"), az_start=m.start("az"), az_end=m.end("az"),
        ))
    taken = blocked + [(h.start, h.end) for h in hits]
    for m in AZ_RE.finditer(text):
        if _overlaps(*m.span(), taken):
            continue
        hits.append(Citation(
            kind="az", start=m.start(), end=m.end(), raw=m.group(0),
            az=m.group(0), canonical=canonical_az(m.group(0)),
            az_start=m.start(), az_end=m.end(),
        ))
    hits.sort(key=lambda h: h.start)
    return hits
```

Then in `app/jurisprudence_ingest.py` replace the body of `_az_for_compare` and the `_AZ_COURT_PREFIX` definition:

```python
from citation_identity import _AZ_COURT_PREFIX, canonical_az  # noqa: F401  (Kompatibilität)


def _az_for_compare(az: Optional[str]) -> str:
    return canonical_az(az)
```

In `app/draft_citation_ingest.py` replace the `DECISION_RE = re.compile(...)` block with `from citation_identity import DECISION_RE` (keep the `re` import if still used).

- [ ] **Step 4: Run tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_citation_identity.py tests/test_draft_citation_ingest*.py tests/test_verify_source*.py -q` then `.venv/bin/python -m pytest tests -q`. Adjust nothing in other tests unless a test asserted the old regex object identity.

- [ ] **Step 5: Commit**

```bash
git add app/citation_identity.py app/jurisprudence_ingest.py app/draft_citation_ingest.py tests/test_citation_identity.py
git commit -m "feat(citations): citation_identity als eine Zitat-Erkennung und Az-Normalisierung"
```

---

### Task 2: `assessment_memory` — Rebase alles-oder-nichts, Größe ohne Server-Felder, berührte ids, Recheck-Zeitstempel

**Files:**
- Modify: `app/assessment_memory.py` (`validate_assessment_content` 139-172, `rebase_assessment_ops` 508-538, `recheck_assessment` 565-600, `render_assessment_for_wiki` 623-680, `verified_az_whitelist` 609-620)
- Test: `tests/test_memory_assessment_rebase.py`, `tests/test_memory_assessment_model.py`, `tests/test_memory_assessment_render.py`

**Interfaces:**
- Produces: `touched_gutachten_ids(ops) -> set`; `rebase_assessment_ops(ops, new_content, conflict_ids) -> list` (voll oder leer); `render_assessment_for_wiki(content, max_chars) -> tuple[str, set]`; `verified_az_whitelist(content, only_ids: Optional[set] = None) -> set` (Werte über `canonical_az`); `validate_assessment_content` lehnt doppelte `canonical_az` je Gutachten ab.
- Consumes: `citation_identity.canonical_az`.

- [ ] **Step 1: Failing tests** (append to the named files; use the existing `_entry(...)`/fixture helpers of each file for valid Gutachten)

```python
# tests/test_memory_assessment_rebase.py
def test_rebase_remove_plus_append_survives_or_dies_together(gutachten_factory):
    base = {"gutachten": [gutachten_factory("aa"), gutachten_factory("bb")], "notizen": ""}
    ops = [{"op": "remove", "path": "/gutachten/by-id/aa"},
           {"op": "append", "path": "/gutachten/-", "value": gutachten_factory("aa", ergebnis="neu")}]
    kept = rebase_assessment_ops(ops, base, conflict_ids={"bb"})
    assert [o["op"] for o in kept] == ["remove", "append"]
    assert rebase_assessment_ops(ops, base, conflict_ids={"aa"}) == []


def test_rebase_notizen_conflict_supersedes_whole_proposal(gutachten_factory):
    base = {"gutachten": [gutachten_factory("aa")], "notizen": "neu"}
    ops = [{"op": "set", "path": "/notizen", "value": "alt"},
           {"op": "append", "path": "/gutachten/-", "value": gutachten_factory("bb")}]
    assert rebase_assessment_ops(ops, base, conflict_ids={"notizen"}) == []
    assert len(rebase_assessment_ops(ops, base, conflict_ids=set())) == 2


def test_touched_gutachten_ids():
    ops = [{"op": "set", "path": "/gutachten/by-id/aa", "value": {}},
           {"op": "append", "path": "/gutachten/-", "value": {"id": "cc"}},
           {"op": "set", "path": "/notizen", "value": "x"}]
    assert touched_gutachten_ids(ops) == {"aa", "cc"}
```

```python
# tests/test_memory_assessment_model.py
def test_size_limit_ignores_server_fields(gutachten_factory):
    entry = gutachten_factory("aa")
    entry["fundstellen"][0]["aussage"] = "x" * (MAX_GUTACHTEN_BYTES - _json_size(entry) - 10)
    validate_assessment_content({"gutachten": [entry], "notizen": ""})
    for f in entry["fundstellen"]:
        f.update({"store": "verified", "store_entry_id": "e" * 36, "store_checked_at": "2026-09-07T12:00:00.000000"})
    validate_assessment_content({"gutachten": [entry], "notizen": ""})  # darf nicht werfen


def test_duplicate_canonical_az_rejected(gutachten_factory):
    entry = gutachten_factory("aa")
    dup = dict(entry["fundstellen"][0]); dup["az"] = "OVG NRW " + dup["az"]
    entry["fundstellen"].append(dup)
    with pytest.raises(ValueError, match="doppelt"):
        validate_assessment_content({"gutachten": [entry], "notizen": ""})
```

```python
# tests/test_memory_assessment_render.py
def test_render_for_wiki_returns_rendered_ids_and_whitelist_follows(gutachten_factory):
    big = gutachten_factory("aa"); big["ergebnis"] = "x" * 5000
    content = {"gutachten": [big, gutachten_factory("bb")], "notizen": ""}
    text, ids = render_assessment_for_wiki(content, max_chars=3000)
    assert "GUTACHTEN aa" not in text and ids == {"bb"}
    assert verified_az_whitelist(content, only_ids=ids) == verified_az_whitelist({"gutachten": [content["gutachten"][1]], "notizen": ""})
```

If no `gutachten_factory` fixture exists, add one to `tests/conftest.py` built from the `_entry` helper already used in `tests/test_memory_assessment_model.py` (id, optional `ergebnis`, one verified-able Fundstelle `OVG NRW 18 E 491/12 vom 2012-06-18`).

- [ ] **Step 2: Run, expect failures/ImportError**

- [ ] **Step 3: Implement**

`touched_gutachten_ids`:
```python
def touched_gutachten_ids(ops: List[Dict[str, Any]]) -> set:
    """Ids, die eine Op adressiert (set/remove by-id) oder anlegt (append)."""
    ids: set = set()
    for op in ops or []:
        if isinstance(op, dict):
            gid = _op_target_id(op)
            if gid:
                ids.add(gid)
    return ids
```

`rebase_assessment_ops` (replace body):
```python
    existing_ids = {e.get("id") for e in (new_content or {}).get("gutachten") or []}
    ops = [op for op in ops if isinstance(op, dict)]
    for op in ops:
        gid = _op_target_id(op)
        if gid is None:
            if str(op.get("path") or "").strip("/") == "notizen" and "notizen" in conflict_ids:
                return []
            continue
        if gid in conflict_ids:
            return []
        if op.get("op") == "append" and gid in existing_ids:
            return []
    try:
        apply_assessment_ops(new_content, ops)
    except ValueError:
        return []
    return ops
```
Update the docstring: "Alles oder nichts: entweder alle Ops überleben oder das Proposal wird superseded."

`validate_assessment_content`: measure sizes on a stripped copy and check duplicate canonical Az:
```python
    from citation_identity import canonical_az

    def _without_server_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
        e = copy.deepcopy(entry)
        for f in e.get("fundstellen") or []:
            for k in SERVER_FIELDS:
                f.pop(k, None)
        return e
    ...
        seen_az: set = set()
        for f in entry["fundstellen"]:
            key = canonical_az(f["az"])
            if key in seen_az:
                raise ValueError(f"Gutachten {entry['id']}: Fundstelle {f['az']} ist doppelt")
            seen_az.add(key)
        if _json_size(_without_server_fields(entry)) > MAX_GUTACHTEN_BYTES: ...
    stripped_total = {"gutachten": [_without_server_fields(e) for e in dumped["gutachten"]], "notizen": dumped.get("notizen", "")}
    if _json_size(stripped_total) > MAX_CONTENT_BYTES: ...
```

`recheck_assessment`: always persist the re-stamped content; revision + version only on state change:
```python
    new_content, warnings = reconcile_store(previous, store_map, None)
    changed_fundstellen, changed_gutachten_count = count_store_changes(previous, new_content)
    if changed_fundstellen:
        _create_revision(db, ASSESSMENT_TARGET, target, previous, new_content, [], "recheck")
    _write_target_content(ASSESSMENT_TARGET, target, new_content)   # store_checked_at immer aktuell
    target.search_text = render_case_assessment_compact(new_content)
    target.updated_at = datetime.utcnow()
    db.add(target)
    db.commit()
```
(Keep the existing return dict.)

`render_assessment_for_wiki` returns `(text, rendered_ids)`: collect `rendered_ids` while building `blocks` (list of `(entry_id, block)`); the truncation loop keeps only whole blocks and their ids. `verified_az_whitelist(content, only_ids=None)`: skip entries whose id is not in `only_ids` when given; return `canonical_az(f["az"])`.

- [ ] **Step 4: Run the three files, then the suite** — update the existing wiki-related tests that unpack the old string return (`tests/test_pattern_wiki_distill_assessment.py`, `tests/test_memory_assessment_render.py`).

- [ ] **Step 5: Commit** `fix(assessment): Rebase alles-oder-nichts, Groesse ohne Server-Felder, beruehrte ids, Recheck-Zeitstempel`

---

### Task 3: `agent_memory_service` + Recheck-Route — Versionsprüfung bei Erstellung, Abgleich außerhalb des Guards, notizen im Rebase-Scope, Skip ohne Gutachten

**Files:**
- Modify: `app/agent_memory_service.py` (`create_memory_update_proposal` 630-700, accept-Block 966-992, `rebase_scope` 1006-1024)
- Modify: `app/endpoints/agent_memory.py:1973-1990` (Recheck-Route)
- Modify: `app/endpoints/pattern_wiki.py` (Aufruf von `render_assessment_for_wiki` → Tupel, `verified_az_whitelist(content, only_ids=ids)`)
- Test: `tests/test_memory_assessment_store.py`, `tests/test_memory_assessment_routes.py`

**Interfaces:**
- Consumes: `touched_gutachten_ids`, `changed_gutachten_ids`, `reconcile_store`, `load_store_map` (Task 2).
- Produces: Erstellung wirft `ValueError("Memory version mismatch")` bei Versionsabweichung (Route mappt wie bisher auf 400). Recheck-Route antwortet `{"skipped": "keine Gutachten", "changed_fundstellen": 0, "changed_gutachten": 0, "warnings": []}` ohne Store-Laden, Limit `300/hour`.

- [ ] **Step 1: Failing tests** — in `tests/test_memory_assessment_store.py` (uses the in-memory fakes of that file):

```python
def test_create_rejects_stale_expected_version(fake_db_with_assessment):
    db, owner, case = fake_db_with_assessment(version=3)
    with pytest.raises(ValueError, match="version"):
        create_memory_update_proposal(db, owner, ASSESSMENT_TARGET, expected_version=2, ops=[...], source_refs=[...], case_id=case)


def test_identical_set_triggers_reconcile(fake_db_with_assessment, monkeypatch):
    # verified Fundstelle, set by-id mit identischem Inhalt -> reconcile fuer diese id aufgerufen
    called = {}
    monkeypatch.setattr(ams, "reconcile_store", lambda content, store_map, ids, **kw: called.setdefault("ids", ids) or (content, []))
    ... accept ...
    assert called["ids"] == {"aa"}


def test_enriched_validation_error_is_not_swallowed(fake_db_with_assessment, monkeypatch):
    monkeypatch.setattr(ams, "reconcile_store", lambda *a, **k: (_ for _ in ()).throw(ValueError("zu groß")))
    with pytest.raises(ValueError, match="zu groß"):
        accept_memory_update_proposal(...)


def test_store_load_failure_still_accepts(fake_db_with_assessment, monkeypatch):
    monkeypatch.setattr(ams, "load_store_map", lambda db: (_ for _ in ()).throw(RuntimeError("db down")))
    proposal, warnings = accept_memory_update_proposal(...)
    assert proposal.status == "accepted" and warnings and warnings[0]["store"] == "unchecked"
```

In `tests/test_memory_assessment_routes.py` (TestClient pattern of that file): recheck on a case without Gutachten returns `skipped == "keine Gutachten"` and the patched `load_store_map` was not called.

- [ ] **Step 2: Run, expect failures**

- [ ] **Step 3: Implement**

Creation, right after `content = _target_content(target_type, target)`:
```python
    if _target_version(db, target_type, target) != expected_version:
        raise ValueError("Memory version mismatch")
```

Accept block (replace lines 966-989):
```python
        touched = changed_gutachten_ids(previous_content, new_content) | touched_gutachten_ids(_proposal_ops(proposal))
        store_map = None
        try:
            from database import SessionLocal
            with SessionLocal() as store_db:
                store_map = load_store_map(store_db)
        except Exception as exc:  # noqa: BLE001 - Store-Störung darf den Accept nicht kippen
            print(f"[WARN] Gutachten-Store-Abgleich fehlgeschlagen: {exc}")
        if store_map is None:
            assessment_warnings = [{"gutachten_id": gid, "az": "", "store": "unchecked"} for gid in sorted(touched)]
        else:
            new_content, assessment_warnings = reconcile_store(new_content, store_map, touched)
            citation_requests = citation_lines(assessment_warnings, new_content)
```
Import `touched_gutachten_ids` next to `changed_gutachten_ids`. `reconcile_store` must keep raising `ValueError` from its final validation (check it calls `validate_assessment_content`; if it does not, add the call at its end).

`rebase_scope` for the assessment target:
```python
        rebase_scope = changed_gutachten_ids(previous_content, new_content)
        if (previous_content.get("notizen") or "") != (new_content.get("notizen") or ""):
            rebase_scope = set(rebase_scope) | {"notizen"}
```

Recheck route: `@limiter.limit("300/hour")`; before calling `recheck_assessment`, load the target via `get_or_create_case_assessment(db, current_user.id, target_case_id)` and if no entry has `status == "aktiv"` return the skip dict. (Alternatively put the check at the top of `recheck_assessment` before `load_store_map` and add `"skipped"` to its early-return dict — choose this if simpler; the route test only sees the response.)

`pattern_wiki.py`: `assessment_block, rendered_ids = render_assessment_for_wiki(assessment_content)` and `whitelist = verified_az_whitelist(assessment_content, only_ids=rendered_ids)`.

- [ ] **Step 4: Run both files, then the suite**
- [ ] **Step 5: Commit** `fix(memory): Versionspruefung bei Proposal-Erstellung, Store-Abgleich fuer beruehrte ids ausserhalb des Guards, Recheck-Skip`

---

### Task 4: Verbraucher auf `citation_identity` umstellen — Wiki-Stripping, Sperrprüfung, Fakten-Check, verify-facts-Route, Grounding

**Files:**
- Modify: `app/endpoints/pattern_wiki.py` (`_EU_NORM_TOKEN_RE`, `_forbidden_tokens` 150-190, `_DECISION_CITATION_RE`/`_entry_violations` 192-220, `_BARE_AZ_RE`/`strip_foreign_citations` 225-260, Strip-Schleife 338-360)
- Modify: `app/citation_verifier.py` (`_FACT_AZ_RE` 136, `verify_facts` 194-262)
- Modify: `app/draft_context.py:39-52` (`verify_facts_with_sources(..., blocked_az=None)`)
- Modify: `app/endpoints/workflow.py:296-312` (Blocklist sammeln, "Nicht zitierfähig"-Zeilen aus dem Korpus entfernen)
- Modify: `app/agent_memory_service.py:1262-1267` (Grounding `assessment_blocked_az` → `canonical_az`)
- Test: `tests/test_pattern_wiki_distill_assessment.py`, `tests/test_memory_assessment_facts.py`, `tests/test_memory_assessment_prompt_context.py`

**Interfaces:**
- Consumes: `find_citations`, `canonical_az`, `DECISION_RE` (Task 1).
- Produces: `strip_foreign_citations(text, whitelist) -> (text, stripped)` unverändert in der Signatur, Whitelist enthält `canonical_az`-Werte; `verify_facts_with_sources(text, memory_text, sources, blocked_az=None)`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_pattern_wiki_distill_assessment.py (ersetzt die Tests zu Praefix/Kern und EU-Norm)
def test_strip_keeps_eu_norms_and_whitelisted_bavarian_az():
    text = "Art. 3 RL 2011/95 und (VG München, Beschluss vom 17.03.2022 – M 10 K 21.3767)"
    cleaned, stripped = strip_foreign_citations(text, {"m10k21.3767"})
    assert cleaned == text and stripped == []


def test_strip_no_prefix_core_equivalence():
    cleaned, stripped = strip_foreign_citations("(M 10 K 21.3767)", {"10k21.3767"})
    assert "21.3767" not in cleaned and stripped[0]["az"] == "M 10 K 21.3767"


def test_strip_handles_eugh_egmr_and_compact():
    cleaned, stripped = strip_foreign_citations("EuGH C-151/22, EGMR Nr. 12345/19, 18E491/12", set())
    assert [s["az"] for s in stripped] == ["C-151/22", "Nr. 12345/19", "18E491/12"]


def test_distill_strips_title_tags_fingerprint(monkeypatch, ...):  # nutzt das Distill-Harness der Datei
    # Eintrag mit fremdem Az in title, tags[0] und fingerprint["themen"][0] -> alle drei bereinigt


def test_entry_violations_name_inside_citation_span_is_still_caught():
    entry = PatternWikiExtractionEntry(title="x", argument_patterns=[
        "VG Teststadt, Frau Mustermann, Urteil vom 01.02.2020 – 18 E 491/12"])
    assert _entry_violations(entry, {"Mustermann"}) == ["Mustermann"]
    assert _entry_violations(entry, {"01.02.2020", "18 E 491/12"}) == []


def test_entry_violations_casefold():
    entry = PatternWikiExtractionEntry(title="mustermann klagt")
    assert _entry_violations(entry, {"Mustermann"}) == ["Mustermann"]


def test_forbidden_tokens_include_assessment_but_not_stand_or_decision_dates(gutachten_factory):
    entry = gutachten_factory("aa"); entry["rechtsfrage"] = "Geburt am 18.10.1995, Az 9 K 1/26"
    tokens = _forbidden_tokens(SimpleNamespace(name="157/26 X"), {}, {}, assessment_content={"gutachten": [entry], "notizen": ""})
    assert "18.10.1995" in tokens and "9 K 1/26" in tokens
    assert "03.09.2026" not in tokens and "18.06.2012" not in tokens   # stand, Fundstellen-datum
```

```python
# tests/test_memory_assessment_facts.py
def test_blocked_bavarian_and_eugh_az_are_flagged():
    corpus = "Nicht zitierfähig: M 10 K 21.3767, C-151/22"
    res = verify_facts("Vgl. M 10 K 21.3767 und EuGH C-151/22.", {}, memory_text=corpus, blocked_az={"m10k21.3767", "c-151/22"})
    assert sorted(c["value"] for c in res["fact_checks"] if c["status"] == "blocked_citation") == ["C-151/22", "M 10 K 21.3767"]


def test_verify_facts_with_sources_forwards_blocklist_and_strips_blocklist_lines():
    memory = "Fakt: 01.02.2020\nNicht zitierfähig: 18 E 491/12\n"
    res = verify_facts_with_sources("Vgl. 18 E 491/12.", memory, [], blocked_az={"18e491/12"})
    assert res["fact_checks"][0]["status"] == "blocked_citation"


def test_blocked_check_runs_on_empty_corpus():
    res = verify_facts("Vgl. 18 E 491/12.", {}, memory_text="", blocked_az={"18e491/12"})
    assert res["fact_checks"] and res["fact_checks"][0]["status"] == "blocked_citation"
```

```python
# tests/test_memory_assessment_prompt_context.py
def test_grounding_blocked_az_is_canonical(...):
    # Fundstelle az="9 K 1/26 (Max Mustermann)" mit store=not_in_store -> collect["assessment_blocked_az"] == ["9k1/26"]
```

- [ ] **Step 2: Run, expect failures**

- [ ] **Step 3: Implement**

`pattern_wiki.py`:
- delete `_EU_NORM_TOKEN_RE`, `_BARE_AZ_RE`, `_DECISION_CITATION_RE`; `from citation_identity import canonical_az, find_citations`.
- `_forbidden_tokens(case, brief_content, strategy_content, allowed_az=None, assessment_content=None)`: build `blob` from brief, strategy and a copy of `assessment_content` whose entries have `stand` removed and whose Fundstellen have `datum` removed. Drop tokens whose span in the blob lies inside a `NORM_RE` match (use `find_citations`-style overlap on the blob: collect `NORM_RE` spans once, skip `_critical_tokens` hits that overlap). Keep the `allowed_az` filter, compare via `canonical_az`.
- `_entry_violations`: exempt spans are only `(az_start, az_end)` of each `decision` hit plus the span of its `date` (locate via `text.find(hit.date, hit.start, hit.end)`); compare with `re.finditer(re.escape(token), text, flags=re.IGNORECASE)`.
- `strip_foreign_citations`: iterate `find_citations(text)` in reverse; a `decision` hit not in the whitelist is cut as its whole span, an `az` hit as its raw token; both are appended to `stripped` as `{"az": hit.az, "citation": hit.raw}`. Keep the three cosmetic `re.sub` cleanups.
- distill loop: also clean `entry.title`, each `entry.tags` item, and every string / list-of-string value in `entry.fingerprint`.
- call `_forbidden_tokens(..., assessment_content=assessment_content)`.

`citation_verifier.py`: remove `_FACT_AZ_RE`; corpus Az = `{h.canonical for h in find_citations(corpus)}`; draft Az loop over `find_citations(draft_text)` using `h.raw` as `value` and `h.canonical` as key; `blocked_norm = {canonical_az(a) for a in blocked_az or set()}`; the early return on empty corpus becomes: if corpus empty AND no blocked_az → return empty; otherwise still run the Az loop (dates/amounts skipped when corpus empty).

`draft_context.py`: `verify_facts_with_sources(text, memory_text="", sources=(), blocked_az=None)`; drop lines containing "Nicht zitierfähig" from `memory_text` before joining; pass `blocked_az` through.

`workflow.py` verify-facts: `grounding = {}`; `get_case_memory_prompt_context(db, current_user, target_case_id, pseudonymize_for_cloud=False, collect=grounding)`; `blocked = set(grounding.get("assessment_blocked_az") or [])`; call `verify_facts_with_sources(body.text, memory_text, body.sources, blocked_az=blocked)`.

`agent_memory_service.py:1267`: `collect["assessment_blocked_az"] = sorted({canonical_az(a) for a in blocked_az}) if rendered else []`.

- [ ] **Step 4: Run the three files, then the suite**
- [ ] **Step 5: Commit** `fix(citations): Wiki-Stripping, Sperrpruefung, Fakten-Check und Grounding auf citation_identity`

---

### Task 5: UI-Warnung bleibt stehen, Registry-Tests isoliert

**Files:**
- Modify: `app/static/js/app.js:1782-1793`
- Modify: `tests/test_memory_target_registry.py:1-40`

- [ ] **Step 1:** In `app.js`, capture the warning text in a local `warningText` before `await loadCaseMemory()` and call `setCaseMemoryStatus(warningText, false)` again after it returns (no new element needed; the status helper is reused). Verify by reading the function once more that `loadCaseMemory()` is what overwrote it.
- [ ] **Step 2:** In `tests/test_memory_target_registry.py`, replace the module-level `sys.modules.setdefault(...)` stubs with a module-scoped `pytest.fixture(autouse=True)` that uses `monkeypatch.setitem(sys.modules, name, stub)` for each stub and `monkeypatch.delitem(sys.modules, "agent_memory_service", raising=False)` before and after, then imports the module under test inside the tests. Run `pytest tests/test_memory_target_registry.py tests/test_pattern_wiki_distill_assessment.py -q` — must pass in that order and in reverse.
- [ ] **Step 3:** Suite, commit `test(memory): Registry-Stubs isoliert, Accept-Warnung im UI bleibt sichtbar`

---

### Task 6: Skills — unbekannter Eigentümer, Zeitbudget, Recheck-Bilanz, Erinnerung nur für neue Proposals

**Files (Repo `~/kanzlei/skills`):**
- Modify: `rechtmaschine/scripts/memory_triage.py` (`_rm` 116-121, `case_owner`/`is_foreign_case` 380-408, `recheck_assessments` 543-553, `main` 574-620)
- Modify: `rechtmaschine/scripts/memory_hygiene_hook.py` (`_vermerk_lines` 107-160, State 186-202)
- Test: `rechtmaschine/tests/test_memory_triage.py` (anlegen, falls es fehlt), `rechtmaschine/tests/test_memory_hygiene_hook.py`

**Interfaces:**
- Produces: `is_foreign_case(label) -> str | None` liefert `"unbekannt"` bei Lookup-Fehler; Modulvariable `DEADLINE: float | None = None` in `memory_triage`, von `_rm` beachtet; `recheck_assessments(case_ids) -> dict(ok=, skipped=, failed=)`; Hook-State bekommt `started_at` (ISO, UTC).

- [ ] **Step 1: Failing tests**

```python
# rechtmaschine/tests/test_memory_triage.py
def test_is_foreign_case_unknown_on_lookup_failure(monkeypatch):
    monkeypatch.setattr(triage, "case_owner", lambda az: None)
    assert triage.is_foreign_case("157/26 X") == "unbekannt"


def test_rm_respects_deadline(monkeypatch):
    seen = {}
    monkeypatch.setattr(triage.subprocess, "run", lambda *a, **k: seen.update(k) or SimpleNamespace(returncode=0, stdout="{}", stderr=""))
    triage.DEADLINE = time.monotonic() + 5
    triage._rm("whoami")
    assert 0 < seen["timeout"] <= 5


def test_recheck_assessments_counts(monkeypatch):
    outputs = iter(['{"skipped": "keine Gutachten"}', '{"changed_fundstellen": 0}', RuntimeError("rc=2")])
    def fake(*a):
        v = next(outputs)
        if isinstance(v, Exception): raise v
        return v
    monkeypatch.setattr(triage, "_rm", fake)
    assert triage.recheck_assessments(["a", "b", "c"]) == {"ok": 1, "skipped": 1, "failed": 1}
```

```python
# rechtmaschine/tests/test_memory_hygiene_hook.py
def test_vermerk_reminder_ignores_proposals_before_started_at(...):
    # state started_at = 2026-09-07T10:00:00, Proposal created_at 2026-09-06 -> Erinnerung erscheint
    # Proposal created_at 2026-09-07T11:00 -> keine Erinnerung


def test_vermerk_lines_skip_unknown_owner(...):
    # triage.is_foreign_case -> "unbekannt": kein resolve_case_id-Aufruf, Az NICHT abgehakt
```

- [ ] **Step 2: Run, expect failures**

- [ ] **Step 3: Implement**

`memory_triage.py`:
```python
DEADLINE: float | None = None  # monotonic; von Aufrufern mit Zeitbudget gesetzt

def _rm(*args: str) -> str:
    timeout = CLI_TIMEOUT
    if DEADLINE is not None:
        timeout = max(1, min(CLI_TIMEOUT, int(DEADLINE - time.monotonic())))
    r = subprocess.run([RM_CLI, *args], capture_output=True, text=True, timeout=timeout)
    ...
```
`is_foreign_case`: `if owner is None: return "unbekannt"`. In `main`, the existing `if foreign:` branch already skips (prints `fremde Akte (unbekannt)`).
`recheck_assessments`: returns counts; `"skipped"` when the JSON has `skipped`, `"failed"` on exception (message unchanged). `main` prints `recheck: ok=… skipped=… failed=…` after the summary line and returns 1 when `failed`.

`memory_hygiene_hook.py`: state creation adds `"started_at": datetime.now(timezone.utc).isoformat()`; `main` sets `triage.DEADLINE = deadline` after computing it; `_vermerk_lines(pending_v, latest_v, deadline, triage=None, started_at=None)`: `if foreign == "unbekannt": continue` (before the `if foreign:` ack), and `matched` additionally requires `(p.get("created_at") or "") >= started_at` when `started_at` is given. Callers pass `state.get("started_at")`.

- [ ] **Step 4: Run skills tests** `cd ~/kanzlei/skills && python3 -m pytest rechtmaschine/tests -q`, plus `python3 -m py_compile` on both scripts and one dry run `python3 rechtmaschine/scripts/memory_triage.py --az 157/26` (no `--apply`).
- [ ] **Step 5: Commit in the skills repo** `fix(memory): unbekannter Eigentuemer wird uebersprungen, Zeitbudget in _rm, Recheck-Bilanz, Erinnerung nur fuer neue Proposals` and `git push`.

---

### Task 7: Doku-Abgleich und Live-Abnahme

**Files:**
- Modify: `~/kanzlei/skills/rechtmaschine-memory/SKILL.md` (Gutachten-Abschnitt: Rebase alles-oder-nichts, Erstellung prüft `expected_version`, Recheck-Skip), `~/kanzlei/skills/rechtmaschine/SKILL.md` (verify-facts prüft jetzt gesperrte Fundstellen)
- Docs: `docs/superpowers/specs/2026-09-03-case-assessment-memory-design.md` — Abschnitt "Offene Punkte" um die "Bewusst nicht"-Liste der Härtungs-Spec ergänzen.

- [ ] **Step 1:** Doku-Edits, Commit in beiden Repos.
- [ ] **Step 2 (nach Merge + Deploy, außerhalb des Worktrees):** `deploy-worker.sh` (nicht `docker compose restart`), dann auf 157/26: (a) zwei Proposals anlegen — P1 `remove`+`append` derselben id `gueb-statt-duldung` mit geänderter `stand`, P2 `set` auf `heirat-und-visumverfahren` mit identischem Inhalt; P2 zuerst annehmen (Ordnungsblocker beachten: ggf. P1 zuerst anlegen), dann P1; erwartet: P1 überlebt den Rebase vollständig oder wird superseded, nie nur `remove`; das identische `set` liefert alle Fundstellen wieder `verified`. (b) `verify-facts` über die CLI mit einem Text, der eine als `not_in_store` markierte Fundstelle zitiert (dafür ein Test-Gutachten mit erfundenem Az anlegen und danach wieder entfernen) — erwartet `blocked_citation`. (c) `wiki distill` — erwartet einen Eintrag ohne fremde Az in title/tags/fingerprint. (d) `memory assessment recheck` auf einer Akte ohne Gutachten — erwartet `skipped`. Ergebnisse als Vermerk in der Ledger-Datei, nicht in der Akte.
