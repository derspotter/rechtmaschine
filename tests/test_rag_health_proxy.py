"""The app's /rag/health proxy must pass a 503 from the debian RAG API through
with its details (embedder/database flags) instead of collapsing it into a
generic exception text.

Run: .venv/bin/python -m pytest tests/test_rag_health_proxy.py -q
"""
from endpoints.rag import upstream_health_response


def test_upstream_503_keeps_status_and_details():
    status, body = upstream_health_response(
        503,
        {
            "status": "degraded",
            "qdrant_ok": True,
            "desktop_embedder_ok": False,
            "details": {"embedder": False, "embedder_error": "Name or service not known"},
        },
    )
    assert status == 503
    assert body.status == "degraded"
    assert body.qdrant_ok is True
    assert body.desktop_embedder_ok is False
    assert body.details["embedder_error"] == "Name or service not known"


def test_upstream_200_is_healthy_passthrough():
    status, body = upstream_health_response(
        200, {"status": "healthy", "qdrant_ok": True, "desktop_embedder_ok": True, "details": {}}
    )
    assert status == 200
    assert body.status == "healthy"
