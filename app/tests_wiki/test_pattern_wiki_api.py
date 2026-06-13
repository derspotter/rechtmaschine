import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from fastapi.testclient import TestClient

# PatternWikiEntry uses Postgres JSONB; render it as JSON on SQLite so
# create_all works on an in-memory test database.
@compiles(JSONB, "sqlite")
def _jsonb_sqlite(element, compiler, **kw):  # noqa: ANN001
    return "JSON"

import main
import database
import auth
import models

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestingSession = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
database.Base.metadata.create_all(bind=_engine)

FAKE_USER_ID = uuid.uuid4()


class _FakeUser:
    id = FAKE_USER_ID
    email = "tester@example.com"


def _override_db():
    db = _TestingSession()
    try:
        yield db
    finally:
        db.close()


def _override_user():
    return _FakeUser()


main.app.dependency_overrides[database.get_db] = _override_db
main.app.dependency_overrides[auth.get_current_active_user] = _override_user

client = TestClient(main.app, base_url="http://localhost")


def test_create_entry_lands_pending():
    payload = {
        "title": "Testmuster",
        "summary": "Kurzfassung",
        "tags": ["BeurkG", " beurkg "],
        "scope": "firm",
        "argument_patterns": ["arg1"],
    }
    r = client.post("/wiki/entries", json=payload)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["status"] == "pending"
    assert data["owner_id"] == str(FAKE_USER_ID)
    assert data["title"] == "Testmuster"
    assert data["scope"] == "firm"
    assert data["tags"] == ["beurkg", "beurkg"]  # casefolded, no dedup
    assert data["argument_patterns"] == ["arg1"]


def test_create_entry_empty_title_422():
    r = client.post("/wiki/entries", json={"title": "   "})
    assert r.status_code == 422


def test_create_entry_title_too_long_422():
    r = client.post("/wiki/entries", json={"title": "x" * 301})
    assert r.status_code == 422


def test_get_entry_roundtrip_and_404():
    created = client.post("/wiki/entries", json={"title": "Lookup"}).json()
    entry_id = created["id"]
    ok = client.get(f"/wiki/entries/{entry_id}")
    assert ok.status_code == 200
    assert ok.json()["id"] == entry_id
    missing = client.get(f"/wiki/entries/{uuid.uuid4()}")
    assert missing.status_code == 404
