"""Prueft die stop()-Absicherung von JobHeartbeat (Task 8) am ECHTEN Objekt.

Konsolidierung (Sweep-Follow-up 8, 2026-07-13): vorher spiegelte eine lokale
_StopGuardHeartbeat-Klasse die Logik nach; conftest.py stellt die App-Imports
inzwischen bereit, also importieren wir job_worker direkt. Der Heartbeat-
Thread beruehrt die DB erst nach JOB_HEARTBEAT_INTERVAL_SEC (default 5s) —
stop() kommt hier sofort, es findet also nie ein DB-Zugriff statt.

Run: .venv/bin/python -m pytest tests/test_job_heartbeat_stop_guard.py -q
"""
import uuid

from job_worker import JobHeartbeat


def _heartbeat():
    return JobHeartbeat(model=object, job_id=uuid.uuid4(), worker_id="test-worker")


def test_stop_without_start_does_not_raise():
    hb = _heartbeat()
    hb.stop()  # darf NICHT RuntimeError werfen (nie-gestarteter Thread)
    assert hb._started is False


def test_start_then_stop_joins_cleanly():
    hb = _heartbeat()
    hb.start()
    assert hb._thread.is_alive()
    hb.stop()
    assert not hb._thread.is_alive()


if __name__ == "__main__":
    test_stop_without_start_does_not_raise()
    test_start_then_stop_joins_cleanly()
    print("Alle Heartbeat-stop-Guard-Tests bestanden")
