"""/v1/rag/health must fail the compose healthcheck when retrieval cannot work.

12.09.–14.09.2026: the embedder container was down for 42 h, every retrieve
returned embed_failed, and rag-api still reported "healthy" because the
endpoint answered 200 with status=degraded in the body. curl -f never saw it.

Run: .venv/bin/python -m pytest tests/test_rag_api_health.py -q
"""
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


def _load_rag_api():
    path = Path(__file__).resolve().parents[1] / "rag" / "api" / "main.py"
    spec = importlib.util.spec_from_file_location("rag_api_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rag_api_main"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def api(monkeypatch):
    mod = _load_rag_api()

    class _Cursor:
        def execute(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class _Conn:
        def cursor(self):
            return _Cursor()

    @contextmanager
    def ok_db():
        yield _Conn()

    monkeypatch.setattr(mod, "_db_conn", ok_db)
    return mod


def _fake_get(outcome_by_host):
    def fake_get(url, timeout=None):
        for host, outcome in outcome_by_host.items():
            if host in url:
                if isinstance(outcome, Exception):
                    raise outcome
                return httpx.Response(outcome, request=httpx.Request("GET", url))
        raise AssertionError(f"unexpected health probe: {url}")

    return fake_get


def test_health_is_200_when_embedder_and_db_reachable(api, monkeypatch):
    monkeypatch.setattr(api.httpx, "get", _fake_get({"rag-embed": 200, "rag-rerank": 200}))
    resp = TestClient(api.app).get("/v1/rag/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_is_503_when_embedder_unreachable(api, monkeypatch):
    monkeypatch.setattr(
        api.httpx,
        "get",
        _fake_get({"rag-embed": httpx.ConnectError("Name or service not known"), "rag-rerank": 200}),
    )
    resp = TestClient(api.app).get("/v1/rag/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["details"]["embedder"] is False


def test_health_is_503_when_database_unreachable(api, monkeypatch):
    @contextmanager
    def broken_db():
        raise RuntimeError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(api, "_db_conn", broken_db)
    monkeypatch.setattr(api.httpx, "get", _fake_get({"rag-embed": 200, "rag-rerank": 200}))
    resp = TestClient(api.app).get("/v1/rag/health")
    assert resp.status_code == 503
    assert resp.json()["details"]["database"] is False
