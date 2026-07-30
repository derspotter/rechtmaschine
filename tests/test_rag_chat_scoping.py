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
