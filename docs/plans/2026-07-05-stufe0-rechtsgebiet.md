# Stufe 0: Rechtsgebiet-Trennung — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `cases.rechtsgebiet` als First-Class-Feld mit dünner Registry; Asyl-Schichten (Facetten-Hook, Jurisprudenz-Pack) feuern nur noch für Migrationsrecht; Backfill-CLI klassifiziert Bestandsfälle.

**Architecture:** Spec ist `docs/ombudsstelle-rechtsgebiete-plan.md` §4. Reine Registry-/Gating-Logik in neuen puren Modulen (Testbarkeit ohne bcrypt/DB — etabliertes Pillar-Muster); Spalte via MIGRATIONS-Liste in `app/main.py`; API-Erweiterung in `endpoints/cases.py`; Backfill-CLI nach dem Muster von `backfill_case_facets.py`. NULL-Rechtsgebiet verhält sich wie bisher (Migrationsrecht) — Bestandskompatibilität, kein Verhaltenssprung beim Deploy.

**Tech Stack:** FastAPI + SQLAlchemy + Postgres (JSONB/VARCHAR), lokaler Qwen via `citation_qwen.call_qwen_json`, Tests im Plain-assert-Stil mit `__main__`-Runner.

## Global Constraints

- Arbeitsverzeichnis: Worktree `/var/opt/docker/rechtmaschine-research-upgrade`, Branch `ombudsstelle-stufe0`. NIEMALS im Live-Checkout `/var/opt/docker/rechtmaschine` arbeiten.
- Testrunner: `/var/opt/docker/rechtmaschine/.venv/bin/python tests/<file>` (das `.venv` mit Punkt im Hauptcheckout; das Worktree hat keins).
- Testbare Logik gehört in pure Module — `endpoints/*` ist in der Testumgebung nicht importierbar (bcrypt fehlt).
- Git: nur explizite Pfade stagen, nie `git add -A`. Commits enden mit `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Rechtsgebiets-Schlüssel (verbindlich, Spec §4): `asyl | aufenthalt | sozial | miete | inkasso | arbeit | sonstiges`. `None`/NULL = Legacy = Migrationsrecht.
- Qwen-Calls: klein, flach, advisory (niemals blockierend), Muster `facet_extraction.py`.
- Kein Rollout in diesem Plan: Merge/Deploy/Backfill-Ausführung erst nach Jays Go (Spec-Muster).

---

### Task 1: Registry `app/rechtsgebiete.py`

**Files:**
- Create: `app/rechtsgebiete.py`
- Test: `tests/test_rechtsgebiete.py`

**Interfaces:**
- Produces: `RECHTSGEBIETE: dict[str, dict]` (key → `{"label": str, "migrationsrecht": bool}`);
  `normalize_rechtsgebiet(value: Any) -> Optional[str]` (kanonischer Key oder None);
  `uses_asyl_layers(rechtsgebiet: Optional[str]) -> bool` (True für None/asyl/aufenthalt).

- [ ] **Step 1: Failing Test schreiben**

```python
"""Stufe 0: Rechtsgebiets-Registry — Normalisierung + Asyl-Schichten-Gate.
Run: .venv/bin/python tests/test_rechtsgebiete.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rechtsgebiete import (  # noqa: E402
    RECHTSGEBIETE,
    normalize_rechtsgebiet,
    uses_asyl_layers,
)


def test_registry_has_all_spec_keys():
    assert set(RECHTSGEBIETE) == {
        "asyl", "aufenthalt", "sozial", "miete", "inkasso", "arbeit", "sonstiges"
    }, set(RECHTSGEBIETE)
    for key, entry in RECHTSGEBIETE.items():
        assert entry.get("label"), f"{key}: label fehlt"
        assert isinstance(entry.get("migrationsrecht"), bool), key


def test_normalize_canonical_keys_pass_through():
    for key in RECHTSGEBIETE:
        assert normalize_rechtsgebiet(key) == key, key


def test_normalize_aliases_and_case():
    assert normalize_rechtsgebiet("Asylrecht") == "asyl"
    assert normalize_rechtsgebiet("Ausländerrecht") == "aufenthalt"
    assert normalize_rechtsgebiet("auslaenderrecht") == "aufenthalt"
    assert normalize_rechtsgebiet("Aufenthaltsrecht") == "aufenthalt"
    assert normalize_rechtsgebiet("Sozialrecht") == "sozial"
    assert normalize_rechtsgebiet("SGB II") == "sozial"
    assert normalize_rechtsgebiet("Mietrecht") == "miete"
    assert normalize_rechtsgebiet("Arbeitsrecht") == "arbeit"
    assert normalize_rechtsgebiet("  MIETE  ") == "miete"


def test_normalize_unknown_and_empty():
    assert normalize_rechtsgebiet("steuerrecht") is None
    assert normalize_rechtsgebiet("") is None
    assert normalize_rechtsgebiet(None) is None
    assert normalize_rechtsgebiet(42) is None


def test_uses_asyl_layers_gate():
    # NULL = Legacy = Migrationsrecht: Bestandsfälle verhalten sich wie bisher.
    assert uses_asyl_layers(None) is True
    assert uses_asyl_layers("asyl") is True
    assert uses_asyl_layers("aufenthalt") is True
    assert uses_asyl_layers("sozial") is False
    assert uses_asyl_layers("miete") is False
    assert uses_asyl_layers("inkasso") is False
    assert uses_asyl_layers("arbeit") is False
    assert uses_asyl_layers("sonstiges") is False
    # Unbekannte Strings fallen sicher: kein Asyl-Kontext für Nicht-Migrationsfälle.
    assert uses_asyl_layers("steuerrecht") is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
```

- [ ] **Step 2: RED verifizieren**

Run: `cd /var/opt/docker/rechtmaschine-research-upgrade && /var/opt/docker/rechtmaschine/.venv/bin/python tests/test_rechtsgebiete.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'rechtsgebiete'`

- [ ] **Step 3: Implementierung**

```python
"""Dünne Rechtsgebiets-Registry (Ombudsstelle-Plan, Stufe 0).

Ein Eintrag pro Gebiet — nur die Felder, die die aktuelle Ausbaustufe
braucht (Spec §4: keine vorauseilende Generalisierung). Die Asyl-Schichten
(Facetten-Hook, Jurisprudenz-Pack, COI-Research) hängen an
``uses_asyl_layers``; NULL/None bedeutet Legacy-Fall und verhält sich wie
bisher (Migrationsrecht).
"""
from typing import Any, Dict, Optional

RECHTSGEBIETE: Dict[str, Dict[str, Any]] = {
    "asyl":       {"label": "Asylrecht", "migrationsrecht": True},
    "aufenthalt": {"label": "Aufenthaltsrecht", "migrationsrecht": True},
    "sozial":     {"label": "Sozialrecht", "migrationsrecht": False},
    "miete":      {"label": "Mietrecht", "migrationsrecht": False},
    "inkasso":    {"label": "Inkasso/Verbraucher", "migrationsrecht": False},
    "arbeit":     {"label": "Arbeitsrecht", "migrationsrecht": False},
    "sonstiges":  {"label": "Sonstiges", "migrationsrecht": False},
}

_ALIASES = {
    "asylrecht": "asyl",
    "auslaenderrecht": "aufenthalt",
    "ausländerrecht": "aufenthalt",
    "aufenthaltsrecht": "aufenthalt",
    "migrationsrecht": "aufenthalt",
    "sozialrecht": "sozial",
    "sgb ii": "sozial",
    "sgb 2": "sozial",
    "buergergeld": "sozial",
    "bürgergeld": "sozial",
    "jobcenter": "sozial",
    "mietrecht": "miete",
    "wohnraummietrecht": "miete",
    "inkassorecht": "inkasso",
    "verbraucherrecht": "inkasso",
    "arbeitsrecht": "arbeit",
}


def normalize_rechtsgebiet(value: Any) -> Optional[str]:
    """Kanonischer Registry-Key oder None (unbekannt/leer)."""
    if not isinstance(value, str):
        return None
    key = value.strip().casefold()
    if not key:
        return None
    if key in RECHTSGEBIETE:
        return key
    return _ALIASES.get(key)


def uses_asyl_layers(rechtsgebiet: Optional[str]) -> bool:
    """Gate für die asylgebundenen Schichten (Facetten, Jurisprudenz-Pack).
    None = Legacy-Fall (Bestand vor Stufe 0) = Migrationsrecht."""
    if rechtsgebiet is None:
        return True
    entry = RECHTSGEBIETE.get(rechtsgebiet)
    return bool(entry and entry["migrationsrecht"])
```

- [ ] **Step 4: GREEN verifizieren**

Run: `cd /var/opt/docker/rechtmaschine-research-upgrade && /var/opt/docker/rechtmaschine/.venv/bin/python tests/test_rechtsgebiete.py`
Expected: `ALL 5 PASSED`

- [ ] **Step 5: Commit**

```bash
cd /var/opt/docker/rechtmaschine-research-upgrade
git add app/rechtsgebiete.py tests/test_rechtsgebiete.py
git commit -m "Stufe 0: Rechtsgebiets-Registry (Normalisierung + Asyl-Schichten-Gate)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Spalte, Migration, API

**Files:**
- Modify: `app/models.py` (Case-Modell, ~Zeile 16–37: Spalte + to_dict)
- Modify: `app/main.py` (MIGRATIONS-Liste, nach dem Eintrag `2026-07-02_rechtsprechung_enrichment`)
- Modify: `app/endpoints/cases.py` (CaseCreateRequest, `_case_to_dict`, create_case, neuer PUT-Endpoint)
- Test: `tests/test_rechtsgebiet_storage.py`

**Interfaces:**
- Consumes: `normalize_rechtsgebiet` aus Task 1.
- Produces: `Case.rechtsgebiet` (SQLAlchemy Column, String(20), nullable);
  `PUT /cases/{case_id}/rechtsgebiet` mit Body `{"rechtsgebiet": "sozial"}` (oder `null` zum Löschen), Response `{"ok": true, "rechtsgebiet": ...}`, 422 bei unbekanntem Wert;
  `POST /cases` akzeptiert optionales `rechtsgebiet`.

- [ ] **Step 1: Failing Test schreiben** (Muster: `tests/test_case_facets_storage.py` — Model-Metadaten + Migrations-SQL, keine DB)

```python
"""Stufe 0 Storage: cases.rechtsgebiet — Modellspalte, Migration, to_dict.
Run: .venv/bin/python tests/test_rechtsgebiet_storage.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")


def test_case_model_has_rechtsgebiet():
    from models import Case
    col = Case.__table__.columns.get("rechtsgebiet")
    assert col is not None, "cases.rechtsgebiet fehlt im Modell"
    assert col.nullable, "rechtsgebiet muss nullable sein (NULL = Legacy)"


def test_case_to_dict_includes_rechtsgebiet():
    from models import Case
    assert Case(name="x", rechtsgebiet="sozial").to_dict().get("rechtsgebiet") == "sozial"
    assert Case(name="y").to_dict().get("rechtsgebiet") is None


def test_migration_adds_rechtsgebiet():
    with open(os.path.join(APP_DIR, "main.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert "2026-07-05_case_rechtsgebiet" in src, "Migrations-Eintrag fehlt in main.py"
    assert "ADD COLUMN IF NOT EXISTS rechtsgebiet" in src, "ALTER TABLE fehlt in main.py"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
```

- [ ] **Step 2: RED verifizieren**

Run: `cd /var/opt/docker/rechtmaschine-research-upgrade && /var/opt/docker/rechtmaschine/.venv/bin/python tests/test_rechtsgebiet_storage.py`
Expected: FAIL — `assert col is not None` schlägt fehl ("cases.rechtsgebiet fehlt im Modell")

- [ ] **Step 3: Modell + Migration + API implementieren**

`app/models.py`, Case-Klasse — nach der Zeile `facets_json = Column(JSONB, default=dict)`:

```python
    rechtsgebiet = Column(String(20), nullable=True, index=True)
```

und in `Case.to_dict()` nach dem `"facets"`-Eintrag:

```python
            "rechtsgebiet": self.rechtsgebiet,
```

`app/main.py`, MIGRATIONS-Liste — direkt nach dem Tupel `("2026-07-02_rechtsprechung_enrichment", [...])` einfügen:

```python
    (
        "2026-07-05_case_rechtsgebiet",
        [
            """
            ALTER TABLE cases
                ADD COLUMN IF NOT EXISTS rechtsgebiet VARCHAR(20)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_cases_rechtsgebiet
                ON cases (rechtsgebiet)
            """,
        ],
    ),
```

`app/endpoints/cases.py` — vier Änderungen:

1. `CaseCreateRequest` erweitern:

```python
class CaseCreateRequest(BaseModel):
    name: Optional[str] = Field(default=None)
    state: Optional[Dict[str, Any]] = Field(default=None)
    rechtsgebiet: Optional[str] = Field(default=None)
```

2. Neues Request-Modell unter `CaseFacetsRequest`:

```python
class CaseRechtsgebietRequest(BaseModel):
    rechtsgebiet: Optional[str]
```

3. `_case_to_dict` um den Eintrag ergänzen (nach `"archived"`):

```python
        "rechtsgebiet": case.rechtsgebiet,
```

4. In `create_case` vor dem `Case(...)`-Konstruktor normalisieren und mitgeben:

```python
    from rechtsgebiete import normalize_rechtsgebiet

    rechtsgebiet = None
    if body.rechtsgebiet is not None:
        rechtsgebiet = normalize_rechtsgebiet(body.rechtsgebiet)
        if rechtsgebiet is None:
            raise HTTPException(status_code=422, detail=f"Unbekanntes Rechtsgebiet: {body.rechtsgebiet}")
```

und im Konstruktor `rechtsgebiet=rechtsgebiet,` ergänzen.

5. Neuer Endpoint (nach `put_case_facets`, gleicher Aufbau wie dort — UUID-Parse, owner-gescopter Lookup, 404):

```python
@router.put("/cases/{case_id}/rechtsgebiet")
@limiter.limit("200/hour")
async def put_case_rechtsgebiet(
    request: Request,
    case_id: str,
    body: CaseRechtsgebietRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Rechtsgebiet setzen (Intake/Operator) oder mit null löschen (Fall
    verhält sich dann wieder als Legacy-Migrationsfall)."""
    from rechtsgebiete import normalize_rechtsgebiet

    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")

    case = (
        db.query(Case)
        .filter(Case.id == case_uuid, Case.owner_id == current_user.id)
        .first()
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if body.rechtsgebiet is None:
        case.rechtsgebiet = None
    else:
        normalized = normalize_rechtsgebiet(body.rechtsgebiet)
        if normalized is None:
            raise HTTPException(status_code=422, detail=f"Unbekanntes Rechtsgebiet: {body.rechtsgebiet}")
        case.rechtsgebiet = normalized
    case.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True, "rechtsgebiet": case.rechtsgebiet}
```

- [ ] **Step 4: GREEN verifizieren**

Run: `cd /var/opt/docker/rechtmaschine-research-upgrade && /var/opt/docker/rechtmaschine/.venv/bin/python tests/test_rechtsgebiet_storage.py`
Expected: `ALL 3 PASSED`

- [ ] **Step 5: Commit**

```bash
cd /var/opt/docker/rechtmaschine-research-upgrade
git add app/models.py app/main.py app/endpoints/cases.py tests/test_rechtsgebiet_storage.py
git commit -m "Stufe 0: cases.rechtsgebiet — Spalte, Migration, Create-/PUT-API

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Gating der Asyl-Schichten

**Files:**
- Modify: `app/facet_extraction.py` (`maybe_update_case_facets`, Gate am Anfang)
- Modify: `app/backfill_case_facets.py` (Skip in der Fall-Schleife)
- Modify: `app/endpoints/jurisprudence.py` (`_load_case_facets` → liefert zusätzlich rechtsgebiet; Gate in `maybe_render_jurisprudence_context` und den beiden Pack-Endpoints, Zeilen ~440/537/580)
- Test: `tests/test_facet_extraction.py` (ein neuer Test)

**Interfaces:**
- Consumes: `uses_asyl_layers` aus Task 1; `Case.rechtsgebiet` aus Task 2.
- Produces: `_load_case_pack_inputs(db, owner_id, case_id) -> tuple[Optional[dict], Optional[str]]` (ersetzt `_load_case_facets`; Rückgabe `(facets, rechtsgebiet)`).

- [ ] **Step 1: Failing Test schreiben** — in `tests/test_facet_extraction.py` vor dem `__main__`-Block ergänzen:

```python
def test_hook_skips_non_migration_rechtsgebiet():
    # Stufe 0: ein Sozialrechtsfall darf den Asyl-Facetten-Hook nicht mehr
    # triggern (kein nächtliches Anklopfen à la 008/26, keine Asyl-Packs).
    import asyncio
    import facet_extraction as fx

    calls = []

    async def fake_extract(text):
        calls.append(text)
        return {"herkunftsland": "Syrien"}

    class FakeCase:
        id = "c3"
        facets_json = {}
        rechtsgebiet = "sozial"

    orig = fx.extract_facets_from_text
    fx.extract_facets_from_text = fake_extract
    try:
        result = asyncio.run(fx.maybe_update_case_facets(None, FakeCase(), "Jobcenter Bescheid"))
    finally:
        fx.extract_facets_from_text = orig
    assert not calls and result is None
```

- [ ] **Step 2: RED verifizieren**

Run: `cd /var/opt/docker/rechtmaschine-research-upgrade && /var/opt/docker/rechtmaschine/.venv/bin/python tests/test_facet_extraction.py`
Expected: FAIL — `calls` ist nicht leer (der Hook extrahiert trotz rechtsgebiet="sozial")

- [ ] **Step 3: Gates implementieren**

`app/facet_extraction.py`, in `maybe_update_case_facets` direkt nach dem `FACETS_ENABLED`-Check:

```python
    from rechtsgebiete import uses_asyl_layers

    if not uses_asyl_layers(getattr(case, "rechtsgebiet", None)):
        return None
```

`app/backfill_case_facets.py`, in der Fall-Schleife in `main()` VOR dem `has_matchable_facets`-Check:

```python
            from rechtsgebiete import uses_asyl_layers

            if not uses_asyl_layers(case.rechtsgebiet):
                skipped += 1
                continue
```

(Import einmalig an den Dateikopf zu den übrigen Imports ziehen, nicht in die Schleife.)

`app/endpoints/jurisprudence.py`:

1. `_load_case_facets` (Zeile 74) ersetzen durch:

```python
def _load_case_pack_inputs(db: Session, owner_id: Any, case_id: Any) -> tuple:
    """(Facetten, Rechtsgebiet) des Falls, owner-gescoped (Facetten tragen
    Profil-/Gesundheitsdaten). (None, None) bei fehlendem Fall/Fehler."""
    if not case_id:
        return None, None
    try:
        row = (
            db.query(Case.facets_json, Case.rechtsgebiet)
            .filter(Case.id == case_id, Case.owner_id == owner_id)
            .first()
        )
        if not row:
            return None, None
        facets = dict(row[0]) if (FACETS_ENABLED and row[0]) else None
        return facets, row[1]
    except Exception as exc:
        print(f"[WARN] Facet load failed for case {case_id}: {exc}")
        return None, None
```

2. Aufrufstelle in `maybe_render_jurisprudence_context` (~Zeile 440):

```python
    facets, rechtsgebiet = _load_case_pack_inputs(db, owner_id, case_id)
    if not uses_asyl_layers(rechtsgebiet):
        return ""
    if not facets and not (case_memory_text or "").strip():
        return ""
```

3. Die beiden Endpoint-Aufrufstellen (~Zeilen 537 und 580) analog:

```python
    facets, rechtsgebiet = _load_case_pack_inputs(db, current_user.id, target_case_id)
    if not uses_asyl_layers(rechtsgebiet):
        raise HTTPException(status_code=409, detail="Kein Migrationsrechtsfall — Jurisprudenz-Pack ist für dieses Rechtsgebiet nicht verfügbar")
    fp = derive_fingerprint(memory_text, facets=facets)
```

4. Import oben ergänzen: `from rechtsgebiete import uses_asyl_layers`.

- [ ] **Step 4: GREEN verifizieren + volle Suite**

Run:
```bash
cd /var/opt/docker/rechtmaschine-research-upgrade
for t in tests/test_rechtsgebiete.py tests/test_rechtsgebiet_storage.py tests/test_facets.py tests/test_facet_extraction.py tests/test_juris_facets.py tests/test_juris_scoring.py tests/test_juris_enrichment.py tests/test_case_facets_storage.py tests/test_rag_vocabulary.py; do /var/opt/docker/rechtmaschine/.venv/bin/python "$t" | tail -1; done
```
Expected: überall `ALL n PASSED`

- [ ] **Step 5: Commit**

```bash
cd /var/opt/docker/rechtmaschine-research-upgrade
git add app/facet_extraction.py app/backfill_case_facets.py app/endpoints/jurisprudence.py tests/test_facet_extraction.py
git commit -m "Stufe 0: Asyl-Schichten nur noch für Migrationsrecht (Facetten-Hook, Backfill, Packs)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Backfill-CLI `app/backfill_rechtsgebiet.py`

**Files:**
- Create: `app/backfill_rechtsgebiet.py`
- Test: `tests/test_backfill_rechtsgebiet.py`

**Interfaces:**
- Consumes: `normalize_rechtsgebiet` (Task 1), `Case.rechtsgebiet` (Task 2), `call_qwen_json`/`load_document_text` (Bestand).
- Produces: `rechtsgebiet_from_flat(parsed: Any) -> Optional[str]` (pur, testbar); CLI `docker exec rechtmaschine-app python backfill_rechtsgebiet.py [--dry-run] [--limit N]`.

- [ ] **Step 1: Failing Test schreiben**

```python
"""Stufe 0: Rechtsgebiets-Backfill — flaches Qwen-JSON → kanonischer Key.
Pure Python; der LLM-Call selbst wird nicht getestet.
Run: .venv/bin/python tests/test_backfill_rechtsgebiet.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from backfill_rechtsgebiet import rechtsgebiet_from_flat  # noqa: E402


def test_flat_canonical_key():
    assert rechtsgebiet_from_flat({"rechtsgebiet": "sozial"}) == "sozial"


def test_flat_alias_normalized():
    assert rechtsgebiet_from_flat({"rechtsgebiet": "Sozialrecht"}) == "sozial"
    assert rechtsgebiet_from_flat({"rechtsgebiet": "Asylrecht"}) == "asyl"


def test_flat_unknown_or_broken():
    assert rechtsgebiet_from_flat({"rechtsgebiet": "steuerrecht"}) is None
    assert rechtsgebiet_from_flat({"rechtsgebiet": None}) is None
    assert rechtsgebiet_from_flat({}) is None
    assert rechtsgebiet_from_flat(None) is None
    assert rechtsgebiet_from_flat("kaputt") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
```

- [ ] **Step 2: RED verifizieren**

Run: `cd /var/opt/docker/rechtmaschine-research-upgrade && /var/opt/docker/rechtmaschine/.venv/bin/python tests/test_backfill_rechtsgebiet.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'backfill_rechtsgebiet'`

- [ ] **Step 3: CLI implementieren** (Muster `backfill_case_facets.py`: gleiche Material-Sammlung, gleiche CLI-Flags; fill-only — nur Fälle mit `rechtsgebiet IS NULL` werden angefasst)

```python
"""One-time backfill: cases.rechtsgebiet für Bestandsfälle klassifizieren
(Ombudsstelle-Plan Stufe 0).

Fill-only: nur Fälle mit rechtsgebiet IS NULL werden klassifiziert; ein
manuell (PUT) gesetzter Wert wird nie überschrieben. Advisory: bleibt die
Klassifikation leer/unbekannt, bleibt der Fall NULL (= Legacy-Verhalten,
Migrationsrecht).

Run inside the app container:
    docker exec rechtmaschine-app python backfill_rechtsgebiet.py --dry-run
    docker exec rechtmaschine-app python backfill_rechtsgebiet.py [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Optional

from rechtsgebiete import RECHTSGEBIETE, normalize_rechtsgebiet

BACKFILL_MODEL = (
    os.getenv(
        "FACET_EXTRACTION_MODEL",
        os.getenv(
            "MEMORY_EXTRACTION_MODEL",
            os.getenv("LLAMA_SERVER_MODEL", "qwen3.6-27b-udq5xl-vision"),
        ),
    ).strip()
    or "qwen3.6-27b-udq5xl-vision"
)
MAX_CHARS = 20000

_KEYS = "|".join(RECHTSGEBIETE)
_PROMPT = f"""Du ordnest einen deutschen Rechtsfall genau einem Rechtsgebiet zu.

Wähle NUR aus diesen Schlüsseln: {_KEYS}
- asyl: Asylverfahren (BAMF-Bescheid, Klage/Eilverfahren AsylG, Dublin, Widerruf).
- aufenthalt: Aufenthaltsrecht ohne Asylverfahren (Aufenthaltstitel, Duldung, Einbürgerung, Ausweisung).
- sozial: Sozialleistungen (Jobcenter/Bürgergeld, Krankenkasse, Rente, Sozialamt, Wohngeld, BAföG).
- miete: Wohnraummiete (Nebenkosten, Mieterhöhung, Kündigung, Mängel).
- inkasso: Inkasso-/Verbraucherforderungen.
- arbeit: Arbeitsverhältnis (Kündigung, Lohn, Abmahnung).
- sonstiges: nichts davon.

Antworte NUR mit diesem JSON-Objekt:
{{"rechtsgebiet": "{_KEYS}", "begruendung": "string"}}"""


def rechtsgebiet_from_flat(parsed: Any) -> Optional[str]:
    """Kanonischer Key aus der flachen Qwen-Antwort, sonst None."""
    if not isinstance(parsed, dict):
        return None
    return normalize_rechtsgebiet(parsed.get("rechtsgebiet"))


def _case_material(db, case) -> str:
    from models import Document
    from shared import load_document_text

    docs = (
        db.query(Document)
        .filter(Document.case_id == case.id)
        .order_by(Document.created_at.desc())
        .limit(10)
        .all()
    )
    parts = [f"Fallname: {case.name or '(ohne Namen)'}"]
    for doc in docs:
        try:
            text = load_document_text(doc) or ""
        except Exception:
            continue
        if text.strip():
            parts.append(f"### {doc.filename} ({doc.category})\n{text}")
        if sum(len(p) for p in parts) > MAX_CHARS:
            break
    return "\n\n".join(parts)[:MAX_CHARS]


async def _classify(material: str) -> Optional[str]:
    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url or not material.strip():
        return None
    try:
        await ensure_anonymization_service_ready()
        parsed = await call_qwen_json(
            service_url,
            f"{_PROMPT}\n\nFALL:\n{material}",
            model=BACKFILL_MODEL,
            num_predict=300,
            temperature=0.0,
        )
        return rechtsgebiet_from_flat(parsed)
    except Exception as exc:
        print(f"[WARN] Rechtsgebiet-Klassifikation fehlgeschlagen: {exc}")
        return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Nur die ersten N Kandidaten.")
    ap.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts speichern.")
    args = ap.parse_args()

    from database import SessionLocal
    from models import Case

    db = SessionLocal()
    done = skipped = failed = 0
    try:
        cases = (
            db.query(Case)
            .filter(Case.archived == False)  # noqa: E712
            .order_by(Case.updated_at.desc())
            .all()
        )
        for case in cases:
            if case.rechtsgebiet:
                skipped += 1
                continue
            if args.limit and done + failed >= args.limit:
                break
            gebiet = await _classify(_case_material(db, case))
            if not gebiet:
                failed += 1
                print(f"—  {case.name}: keine Klassifikation")
                continue
            print(f"OK {case.name}: {gebiet}")
            if not args.dry_run:
                case.rechtsgebiet = gebiet
                db.add(case)
                db.commit()
            done += 1
    finally:
        db.close()
    print(f"\nbackfill: {done} klassifiziert, {skipped} übersprungen (bereits gesetzt), "
          f"{failed} ohne Ergebnis" + (" [DRY-RUN]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 4: GREEN verifizieren**

Run: `cd /var/opt/docker/rechtmaschine-research-upgrade && /var/opt/docker/rechtmaschine/.venv/bin/python tests/test_backfill_rechtsgebiet.py`
Expected: `ALL 3 PASSED`

- [ ] **Step 5: Commit**

```bash
cd /var/opt/docker/rechtmaschine-research-upgrade
git add app/backfill_rechtsgebiet.py tests/test_backfill_rechtsgebiet.py
git commit -m "Stufe 0: Backfill-CLI klassifiziert Bestandsfälle nach Rechtsgebiet (fill-only)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Rollout-Log + Push

**Files:**
- Modify: `docs/ombudsstelle-rechtsgebiete-plan.md` (§10 Rollout-Log)

- [ ] **Step 1: Rollout-Log-Eintrag anhängen** — an §10 des Spec-Dokuments:

```markdown
- 2026-07-05 — Stufe 0 implementiert (Branch ombudsstelle-stufe0, TDD):
  - Registry app/rechtsgebiete.py (7 Gebiete, Aliase, uses_asyl_layers;
    NULL = Legacy = Migrationsrecht).
  - cases.rechtsgebiet (VARCHAR(20), nullable, Index) + Migration
    2026-07-05_case_rechtsgebiet; POST /cases nimmt rechtsgebiet an,
    PUT /cases/{id}/rechtsgebiet für Intake/Operator (422 bei unbekannt,
    null löscht).
  - Gating: Facetten-Hook, Facetten-Backfill und Jurisprudenz-Pack
    (Render + beide Endpoints) laufen nur noch für
    NULL/asyl/aufenthalt.
  - backfill_rechtsgebiet.py: Qwen-Klassifikation der Bestandsfälle,
    fill-only, --dry-run.
  - Rollout-Schritte (nach Go): merge → Container-Neustart (Migration)
    → backfill_rechtsgebiet.py --dry-run → Liste prüfen → live →
    008/26 als Akzeptanz (rechtsgebiet=sozial, kein Facetten-Versuch im
    nächsten Nachtlauf) → Intake-Anbindung (PUT) in gemma-intake.
```

- [ ] **Step 2: Committen und pushen**

```bash
cd /var/opt/docker/rechtmaschine-research-upgrade
git add docs/ombudsstelle-rechtsgebiete-plan.md docs/plans/2026-07-05-stufe0-rechtsgebiet.md
git commit -m "Stufe 0: Rollout-Log + Implementierungsplan

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin ombudsstelle-stufe0
```

(Branch-Push — kein 3-Maschinen-Sync nötig; der folgt erst dem Master-Merge beim Rollout.)
