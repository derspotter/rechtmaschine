"""Retag-Wake-Logik: Verbindungsfehler lösen genau einen WoL-Versuch aus,
danach wird mit Pausen wiederholt. Kein Netzwerk — Request und Wake injiziert."""
import httpx
import pytest

import retag_rag


def test_connect_error_wakes_once_then_succeeds(monkeypatch):
    wakes = []
    monkeypatch.setattr(retag_rag, "_wake_debian", lambda: wakes.append(1))
    monkeypatch.setattr(retag_rag.time, "sleep", lambda s: None)

    calls = {"n": 0}

    def do_request():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("down")
        return "ok"

    assert retag_rag._request_with_wake(do_request, "test") == "ok"
    assert calls["n"] == 3
    assert wakes == [1]  # nur EIN Weckversuch trotz zweier Verbindungsfehler


def test_persistent_failure_raises_after_attempts(monkeypatch):
    monkeypatch.setattr(retag_rag, "_wake_debian", lambda: None)
    monkeypatch.setattr(retag_rag.time, "sleep", lambda s: None)

    def do_request():
        raise httpx.ConnectError("dead")

    with pytest.raises(RuntimeError, match="test failed after retries"):
        retag_rag._request_with_wake(do_request, "test", attempts=3)


def test_http_status_error_retries_without_wake(monkeypatch):
    wakes = []
    monkeypatch.setattr(retag_rag, "_wake_debian", lambda: wakes.append(1))
    monkeypatch.setattr(retag_rag.time, "sleep", lambda s: None)

    calls = {"n": 0}

    def do_request():
        calls["n"] += 1
        if calls["n"] == 1:
            request = httpx.Request("POST", "http://debian:8090/x")
            response = httpx.Response(502, request=request)
            raise httpx.HTTPStatusError("502", request=request, response=response)
        return 42

    assert retag_rag._request_with_wake(do_request, "test") == 42
    assert wakes == []  # 502 = wach, aber nicht bereit — kein Wecken nötig
