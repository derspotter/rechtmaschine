"""Store reconciliation for Gutachten-Fundstellen, pure and via accept.

    .venv/bin/python -m pytest tests/test_memory_assessment_store.py -q
"""

import operator
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.sql.elements import BindParameter

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import agent_memory_service as ams  # noqa: E402
import models  # noqa: E402
from agent_memory_service import (  # noqa: E402
    ASSESSMENT_TARGET,
    accept_memory_update_proposal,
    create_memory_update_proposal,
)
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


# ---------------------------------------------- accept path (service) ---
# The accept path reconciles the store for every touched Gutachten, outside
# the guard that keeps a broken store from failing the accept. It runs
# against an in-memory stand-in for the Session; no database is needed.

SOURCE_REFS = [{"source_type": "document", "source_id": "vermerk", "label": "Vermerk"}]


def _comparable(value):
    return str(value) if isinstance(value, uuid.UUID) else value


def _matches(row, condition):
    """Whether ``row`` satisfies one ``Column == value`` / ``Column != value``
    filter. Any other condition counts as non-restrictive."""
    left = getattr(condition, "left", None)
    right = getattr(condition, "right", None)
    if left is None or not isinstance(right, BindParameter):
        return True
    actual = _comparable(getattr(row, left.key, None))
    expected = _comparable(right.value)
    if condition.operator is operator.ne:
        return actual != expected
    return actual == expected


class _FakeQuery:
    def __init__(self, rows, columns=None):
        self._rows = rows
        self._columns = columns

    def filter(self, *conditions):
        rows = [row for row in self._rows if all(_matches(row, c) for c in conditions)]
        return _FakeQuery(rows, self._columns)

    def with_for_update(self):
        return self

    def order_by(self, *args):
        return self

    def limit(self, count):
        return self

    def all(self):
        if self._columns is None:
            return list(self._rows)
        return [tuple(getattr(row, name) for name in self._columns) for row in self._rows]

    def first(self):
        rows = self.all()
        return rows[0] if rows else None

    def count(self):
        return len(self._rows)


class _FakeDB:
    """ORM rows in a list: ``query(Model)`` and ``query(Model.col, ...)``.

    Equality/inequality filters are honoured, so the sibling-rebase query
    really sees only the pending proposals."""

    def __init__(self):
        self.rows = []

    def query(self, *entities):
        head = entities[0]
        model = head if isinstance(head, type) else head.class_
        columns = None if isinstance(head, type) else [entity.key for entity in entities]
        return _FakeQuery([row for row in self.rows if isinstance(row, model)], columns)

    def add(self, row):
        if not any(row is existing for existing in self.rows):
            self.rows.append(row)

    def commit(self):
        pass

    def refresh(self, row):
        pass


def _set_op(entry):
    return {"op": "set", "path": f"/gutachten/by-id/{entry['id']}", "value": entry}


@pytest.fixture
def fake_db_with_assessment(gutachten_factory):
    """(db, owner_id, case_id) with one stored, store-verified Gutachten "aa"."""

    def _make(version=1, notizen=""):
        db = _FakeDB()
        owner_id = uuid.uuid4()
        case_id = uuid.uuid4()
        content = validate_assessment_content(
            {"gutachten": [gutachten_factory("aa")], "notizen": notizen}
        )
        content, _ = reconcile_store(content, _store_map(), None, NOW)
        db.add(
            models.CaseAssessment(
                id=uuid.uuid4(),
                owner_id=owner_id,
                case_id=case_id,
                content_json=content,
                search_text="",
                version=version,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        )
        return db, owner_id, case_id

    return _make


def _proposal(db, owner_id, case_id, ops, expected_version=1):
    return create_memory_update_proposal(
        db,
        owner_id,
        ASSESSMENT_TARGET,
        expected_version=expected_version,
        ops=ops,
        source_refs=SOURCE_REFS,
        case_id=case_id,
    )


def test_create_rejects_stale_expected_version(fake_db_with_assessment, gutachten_factory):
    db, owner, case = fake_db_with_assessment(version=3)
    with pytest.raises(ValueError, match="version"):
        create_memory_update_proposal(
            db,
            owner,
            ASSESSMENT_TARGET,
            expected_version=2,
            ops=[_set_op(gutachten_factory("aa", ergebnis="Neu."))],
            source_refs=SOURCE_REFS,
            case_id=case,
        )


def test_identical_set_triggers_reconcile(
    fake_db_with_assessment, gutachten_factory, monkeypatch
):
    # A set with identical content changes nothing substantive but drops the
    # server-owned store fields, so the touched id must be reconciled anyway.
    db, owner, case = fake_db_with_assessment()
    proposal = _proposal(db, owner, case, [_set_op(gutachten_factory("aa"))])
    called = {}

    def fake_reconcile(content, store_map, ids, now_iso=None):
        called["ids"] = ids
        return content, []

    monkeypatch.setattr(ams, "load_store_map", lambda db: _store_map())
    monkeypatch.setattr(ams, "reconcile_store", fake_reconcile)
    accept_memory_update_proposal(db, owner, proposal.id)
    assert called["ids"] == {"aa"}


def test_enriched_validation_error_is_not_swallowed(
    fake_db_with_assessment, gutachten_factory, monkeypatch
):
    db, owner, case = fake_db_with_assessment()
    proposal = _proposal(db, owner, case, [_set_op(gutachten_factory("aa", ergebnis="Neu."))])

    def boom(*args, **kwargs):
        raise ValueError("zu groß")

    monkeypatch.setattr(ams, "load_store_map", lambda db: _store_map())
    monkeypatch.setattr(ams, "reconcile_store", boom)
    with pytest.raises(ValueError, match="zu groß"):
        accept_memory_update_proposal(db, owner, proposal.id)


def test_store_load_failure_still_accepts(
    fake_db_with_assessment, gutachten_factory, monkeypatch
):
    db, owner, case = fake_db_with_assessment()
    proposal = _proposal(db, owner, case, [_set_op(gutachten_factory("aa", ergebnis="Neu."))])

    def down(db):
        raise RuntimeError("db down")

    monkeypatch.setattr(ams, "load_store_map", down)
    accepted, warnings = accept_memory_update_proposal(db, owner, proposal.id)
    assert accepted.status == "accepted"
    assert warnings == [{"gutachten_id": "aa", "az": "", "store": "unchecked"}]


def test_notizen_change_supersedes_a_pending_notizen_proposal(
    fake_db_with_assessment, monkeypatch
):
    db, owner, case = fake_db_with_assessment()
    accepted = _proposal(db, owner, case, [{"op": "set", "path": "/notizen", "value": "Neue Notiz."}])
    pending = _proposal(db, owner, case, [{"op": "set", "path": "/notizen", "value": "Andere Notiz."}])
    accepted.created_at = datetime(2026, 9, 4, 10, 0, 0)
    pending.created_at = accepted.created_at + timedelta(minutes=1)

    monkeypatch.setattr(ams, "load_store_map", lambda db: _store_map())
    accept_memory_update_proposal(db, owner, accepted.id)
    assert pending.status == "superseded"
