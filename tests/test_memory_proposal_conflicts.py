"""Feld-Konflikt statt Target-Blocker (16.09.2026).

``older_pending_proposals`` liefert nur noch die älteren pending Proposals,
die mit dem zu prüfenden Proposal KOLLIDIEREN: gemeinsames Feld, auf dem
mindestens eine Seite nicht nur appended. Ältere Konsolidierungen blockieren
nie. Gutachten (case_assessment) behalten die strikte Target-Ordnung.

Anlass 152/26: ein FACT-Append vom 11.09. wartete acht Tage hinter einem
EVENT-Append vom 08.09. und drei Konsolidierungen, obwohl er kein Feld mit
einem set teilte.

Pure logic, keine DB: sqlalchemy/models werden gestubbt wie in
test_memory_rebase_changed_fields.py. Läuft auf dem Host:

    .venv/bin/python -m pytest tests/test_memory_proposal_conflicts.py -q
"""

import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

_STUBBED = []
for _name, _mod in (
    ("sqlalchemy", types.ModuleType("sqlalchemy")),
    ("sqlalchemy.orm", types.ModuleType("sqlalchemy.orm")),
    ("models", types.ModuleType("models")),
):
    if _name not in sys.modules:
        sys.modules[_name] = _mod
        _STUBBED.append(_name)
sys.modules["sqlalchemy"].desc = getattr(sys.modules["sqlalchemy"], "desc", lambda *a, **k: None)
sys.modules["sqlalchemy.orm"].Session = getattr(sys.modules["sqlalchemy.orm"], "Session", type("Session", (), {}))
for _cls in (
    "CaseBrief", "CaseBriefSource", "CaseStrategy", "CaseStrategySource",
    "CaseMemoryRevision", "MemoryUpdateProposal", "MemoryReflectionJob",
):
    if not hasattr(sys.modules["models"], _cls):
        setattr(sys.modules["models"], _cls, type(_cls, (), {}))
# Die Filter-Ausdrücke in older_pending_proposals greifen auf Spaltenattribute
# zu; das gestubbte Modell braucht sie als beliebige Objekte.
for _col in ("id", "owner_id", "case_id", "target_type", "status", "created_at"):
    if not hasattr(sys.modules["models"].MemoryUpdateProposal, _col):
        setattr(sys.modules["models"].MemoryUpdateProposal, _col, object())


def _content_class(list_fields, scalar_fields):
    class _Content:
        def __init__(self, **kw):
            self._d = {f: list(kw.get(f, []) or []) for f in list_fields}
            self._d.update({f: str(kw.get(f, "") or "") for f in scalar_fields})

        def model_dump(self):
            return dict(self._d)

    return _Content


if "shared" not in sys.modules:
    _shared = types.ModuleType("shared")
    _shared.CaseBriefContent = _content_class(
        ["beteiligte", "verfahrensstand", "sachverhalt", "antraege_ziele",
         "streitige_punkte", "beweismittel", "risiken", "offene_fragen"], ["notizen"])
    _shared.CaseStrategyContent = _content_class(
        ["argumentationslinien", "rechtliche_ansatzpunkte", "beweisstrategie",
         "prozessuale_schritte", "vergleich_oder_taktik", "risiken_und_gegenargumente",
         "offene_fragen"], ["kernstrategie", "notizen"])
    _shared.MemoryPatchOperation = object
    _shared.MemorySourceRef = object
    _shared.MemoryTargetType = str
    sys.modules["shared"] = _shared
    _STUBBED.append("shared")

import agent_memory_service as svc  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _cleanup_stubs():
    yield
    for name in _STUBBED:
        sys.modules.pop(name, None)


T0 = datetime(2026, 9, 8, 9, 5)


def _prop(pid, created, ops, target="case_brief", refs=None):
    return SimpleNamespace(
        id=pid, owner_id="o", case_id="c", target_type=target, status="pending",
        created_at=created, ops=ops,
        source_refs=refs if refs is not None else [{"source_type": "jlawyer_document"}],
    )


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)


class _DB:
    def __init__(self, rows):
        self.rows = rows

    def query(self, *a, **k):
        return _Query(self.rows)


APPEND_VS = [{"op": "append", "path": "/verfahrensstand/-", "value": "Klage am 01.09.2026 eingereicht."}]
APPEND_BM = [{"op": "append", "path": "/beweismittel/-", "value": "Überweisungsschein 10.09.2026."}]
SET_VS = [{"op": "set", "path": "/verfahrensstand", "value": ["alles in einem"]}]
CONSOLIDATION = [
    {"op": "set", "path": "/verfahrensstand", "value": ["x"]},
    {"op": "set", "path": "/beweismittel", "value": ["y"]},
    {"op": "set", "path": "/notizen", "value": "z"},
]


def test_proposal_fields_takes_first_segment():
    assert svc.proposal_fields(APPEND_VS + APPEND_BM) == {"verfahrensstand", "beweismittel"}
    assert svc.proposal_fields([{"op": "set", "path": "/gutachten/by-id/x", "value": {}}]) == {"gutachten"}
    assert svc.proposal_fields([]) == set()


def test_appends_on_different_fields_do_not_conflict():
    assert svc.proposals_conflict(APPEND_BM, APPEND_VS) is False


def test_appends_on_same_field_do_not_conflict():
    other = [{"op": "append", "path": "/verfahrensstand/-", "value": "Bescheid 05.09.2026 zugestellt."}]
    assert svc.proposals_conflict(other, APPEND_VS) is False


def test_set_conflicts_with_append_on_same_field_either_direction():
    assert svc.proposals_conflict(SET_VS, APPEND_VS) is True
    assert svc.proposals_conflict(APPEND_VS, SET_VS) is True


def test_set_on_other_field_does_not_conflict():
    assert svc.proposals_conflict(SET_VS, APPEND_BM) is False


def test_is_consolidation_proposal_reads_source_type():
    assert svc.is_consolidation_proposal(_prop("c", T0, CONSOLIDATION, refs=[{"source_type": "consolidation"}]))
    assert not svc.is_consolidation_proposal(_prop("j", T0, APPEND_VS))


def test_older_fact_on_other_field_passes_a_held_event():
    old_event = _prop("old", T0, APPEND_VS)
    new_fact = _prop("new", T0 + timedelta(days=3), APPEND_BM)
    assert svc.older_pending_proposals(_DB([old_event]), new_fact) == []


def test_older_append_still_blocks_a_newer_set_on_the_same_field():
    old_event = _prop("old", T0, APPEND_VS)
    new_set = _prop("new", T0 + timedelta(days=1), SET_VS)
    assert svc.older_pending_proposals(_DB([old_event]), new_set) == ["old"]


def test_older_consolidation_never_blocks():
    cons = _prop("cons", T0, CONSOLIDATION, refs=[{"source_type": "consolidation"}])
    new_fact = _prop("new", T0 + timedelta(days=1), APPEND_VS)
    assert svc.older_pending_proposals(_DB([cons]), new_fact) == []


def test_newer_consolidation_waits_for_older_appends():
    old_fact = _prop("old", T0, APPEND_BM)
    cons = _prop("cons", T0 + timedelta(minutes=3), CONSOLIDATION, refs=[{"source_type": "consolidation"}])
    assert svc.older_pending_proposals(_DB([old_fact]), cons) == ["old"]


def test_younger_rows_never_block():
    younger = _prop("young", T0 + timedelta(days=1), SET_VS)
    p = _prop("p", T0, APPEND_VS)
    assert svc.older_pending_proposals(_DB([younger]), p) == []


def test_blocking_ids_are_sorted_oldest_first():
    a = _prop("a", T0 + timedelta(hours=2), SET_VS)
    b = _prop("b", T0, SET_VS)
    p = _prop("p", T0 + timedelta(days=1), APPEND_VS)
    assert svc.older_pending_proposals(_DB([a, b]), p) == ["b", "a"]


def test_assessment_keeps_strict_target_order():
    old = _prop("old", T0, [{"op": "append", "path": "/gutachten/-", "value": {"id": "a"}}], target=svc.ASSESSMENT_TARGET)
    new = _prop("new", T0 + timedelta(days=1), [{"op": "append", "path": "/gutachten/-", "value": {"id": "b"}}],
                target=svc.ASSESSMENT_TARGET)
    assert svc.older_pending_proposals(_DB([old]), new) == ["old"]
