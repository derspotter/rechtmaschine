"""Route layer of the case_assessment feature, driven through FastAPI.

Covers the two routes this branch touched (final review, finding 6):
  POST /memory/cases/{case_id}/assessment/recheck
  POST /memory/proposals/{id}/accept  -- assessment_warnings in the payload

The heavy collaborators (DB, auth, domain service) are dependency-overridden
or monkeypatched, so no database is needed.

    .venv/bin/python -m pytest tests/test_memory_assessment_routes.py -q
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

import endpoints.agent_memory as am  # noqa: E402
from auth import get_current_active_user  # noqa: E402
from database import get_db  # noqa: E402
from shared import limiter  # noqa: E402

CASE_ID = "11111111-1111-1111-1111-111111111111"
OWNER_ID = "22222222-2222-2222-2222-222222222222"


class _User:
    id = OWNER_ID
    is_active = True


class _Case:
    id = CASE_ID
    owner_id = OWNER_ID


class _Query:
    """db.query(Case).filter(...).first() -> the owned case."""

    def filter(self, *a, **k):
        return self

    def first(self):
        return _Case()


class _DB:
    def query(self, *a, **k):
        return _Query()


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, lambda request, exc: None)
    app.include_router(am.router)
    app.dependency_overrides[get_db] = lambda: _DB()
    app.dependency_overrides[get_current_active_user] = lambda: _User()
    # _assert_owned_case resolves the human Az/UUID first; there is no DB here.
    monkeypatch.setattr(am, "resolve_case_uuid_for_request", lambda db, user, case_id: CASE_ID)
    monkeypatch.setattr(am, "_notify_memory_changed", lambda *a, **k: None)
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------- recheck ---


def test_recheck_returns_the_three_counters(client, monkeypatch):
    import assessment_memory

    seen = {}

    def fake_recheck(db, owner_id, case_id):
        seen["args"] = (owner_id, case_id)
        return {"changed_fundstellen": 0, "changed_gutachten": 0, "version": 3}

    monkeypatch.setattr(assessment_memory, "recheck_assessment", fake_recheck)
    response = client.post(f"/memory/cases/{CASE_ID}/assessment/recheck")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"changed_fundstellen", "changed_gutachten", "version"}
    assert body["changed_fundstellen"] == 0
    assert seen["args"] == (OWNER_ID, CASE_ID)


def test_recheck_maps_domain_error_to_400(client, monkeypatch):
    import assessment_memory

    def boom(db, owner_id, case_id):
        raise ValueError("Ungültiger Gutachten-Inhalt: kaputt")

    monkeypatch.setattr(assessment_memory, "recheck_assessment", boom)
    response = client.post(f"/memory/cases/{CASE_ID}/assessment/recheck")
    assert response.status_code == 400
    assert "Gutachten-Inhalt" in response.json()["detail"]


def test_recheck_rejects_a_foreign_case(client, monkeypatch):
    class _Empty(_Query):
        def first(self):
            return None

    class _EmptyDB:
        def query(self, *a, **k):
            return _Empty()

    client.app.dependency_overrides[get_db] = lambda: _EmptyDB()
    response = client.post(f"/memory/cases/{CASE_ID}/assessment/recheck")
    assert response.status_code == 404


# ---------------------------------------------------------------- accept ---


class _Proposal:
    id = "33333333-3333-3333-3333-333333333333"
    owner_id = OWNER_ID
    case_id = CASE_ID
    target_type = "case_assessment"
    target_id = "44444444-4444-4444-4444-444444444444"
    status = "accepted"
    ops = []
    source_refs = []
    confidence = 0.9
    model = "claude"
    proposal_metadata = {}
    created_at = None
    updated_at = None
    decided_at = None
    expected_version = 1


WARNINGS = [{"gutachten_id": "gueb-statt-duldung", "az": "9 K 77/25", "store": "not_in_store"}]


def test_accept_attaches_assessment_warnings(client, monkeypatch):
    monkeypatch.setattr(
        am, "accept_memory_update_proposal",
        lambda db, owner_id, proposal_id, actor="user", force=False: (_Proposal(), WARNINGS),
    )
    response = client.post(f"/memory/proposals/{_Proposal.id}/accept")
    assert response.status_code == 200
    body = response.json()
    assert body["assessment_warnings"] == WARNINGS
    assert body["id"] == _Proposal.id


def test_accept_without_warnings_omits_the_key(client, monkeypatch):
    monkeypatch.setattr(
        am, "accept_memory_update_proposal",
        lambda db, owner_id, proposal_id, actor="user", force=False: (_Proposal(), []),
    )
    response = client.post(f"/memory/proposals/{_Proposal.id}/accept")
    assert response.status_code == 200
    assert "assessment_warnings" not in response.json()
