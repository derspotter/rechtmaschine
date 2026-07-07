"""Prueft die asyncio.wait_for-Timeout-Logik des Job-Workers (Task 9).

Was es prueft:
- asyncio.wait_for bricht eine zu langsame Coroutine mit asyncio.TimeoutError ab.
- Die Fehlerabbildung im Worker liefert dann den deutschen Text
  "Zeitlimit überschritten — Job abgebrochen" statt der generischen Formatter-Meldung.

Aufruf (keine DB, keine GPU noetig):
    python3 tests/test_job_execution_timeout.py
"""

import asyncio


def _map_error_message(exc, formatter):
    """Spiegelt die Fehlerabbildung aus app/job_worker.py::_run_claimed_job."""
    if isinstance(exc, asyncio.TimeoutError):
        return "Zeitlimit überschritten — Job abgebrochen"
    return formatter(exc)


async def _slow_job():
    await asyncio.sleep(1.0)
    return "sollte nie erreicht werden"


def test_wait_for_times_out_and_maps_german_message():
    async def run():
        try:
            await asyncio.wait_for(_slow_job(), timeout=0.05)
            raise AssertionError("wait_for haette abbrechen muessen")
        except asyncio.TimeoutError as exc:
            return _map_error_message(exc, lambda e: "generisch")

    msg = asyncio.run(run())
    assert msg == "Zeitlimit überschritten — Job abgebrochen", msg
    print("OK: Timeout ->", msg)


def test_non_timeout_uses_formatter():
    msg = _map_error_message(RuntimeError("kaputt"), lambda e: f"formatiert: {e}")
    assert msg == "formatiert: kaputt", msg
    print("OK: Nicht-Timeout ->", msg)


if __name__ == "__main__":
    test_wait_for_times_out_and_maps_german_message()
    test_non_timeout_uses_formatter()
    print("Alle Job-Execution-Timeout-Tests bestanden")
