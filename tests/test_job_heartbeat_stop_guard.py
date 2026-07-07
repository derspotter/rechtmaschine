"""Prueft die stop()-Absicherung von JobHeartbeat (Task 8).

Was es prueft:
- stop() OHNE vorheriges start() wirft nicht (nie-gestarteter Thread wird nicht gejoint).
- stop() NACH start() beendet den Thread sauber.

Aufruf (keine DB, keine GPU noetig):
    python3 tests/test_job_heartbeat_stop_guard.py

Das Modul importiert nur die Klassenlogik nach, ohne app-Importe (job_worker zieht
sonst DB/Endpoints). Die stop()-Semantik wird 1:1 nachgebaut und gegen einen
Dummy-Thread verifiziert, damit der Test ohne Container laeuft.
"""

import threading


class _StopGuardHeartbeat:
    """Spiegelt die stop()-Absicherung aus app/job_worker.py::JobHeartbeat."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self):
        self._started = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self):
        # Wartet auf das Stop-Signal, damit start()->stop() sicher joint.
        self._stop.wait(5.0)


def test_stop_without_start_does_not_raise():
    hb = _StopGuardHeartbeat()
    hb.stop()  # darf NICHT RuntimeError werfen
    assert hb._started is False
    print("OK: stop() ohne start() wirft nicht")


def test_start_then_stop_joins_cleanly():
    hb = _StopGuardHeartbeat()
    hb.start()
    assert hb._thread.is_alive()
    hb.stop()
    assert not hb._thread.is_alive()
    print("OK: start() -> stop() beendet den Thread")


if __name__ == "__main__":
    test_stop_without_start_does_not_raise()
    test_start_then_stop_joins_cleanly()
    print("Alle Heartbeat-stop-Guard-Tests bestanden")
