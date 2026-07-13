"""Prueft die asyncio.wait_for-Timeout-Logik des Job-Workers (Task 9) am
ECHTEN Fehler-Mapping.

Konsolidierung (Sweep-Follow-up 8, 2026-07-13): vorher spiegelte
_map_error_message die Logik aus _run_claimed_job nach; jetzt ist sie als
job_worker._error_message_for extrahiert und wird direkt importiert.

Was es prueft:
- asyncio.wait_for bricht eine zu langsame Coroutine mit asyncio.TimeoutError ab.
- Das Mapping liefert dann den deutschen Text
  "Zeitlimit überschritten — Job abgebrochen" statt der Formatter-Meldung.

Run: .venv/bin/python -m pytest tests/test_job_execution_timeout.py -q
"""
import asyncio
from types import SimpleNamespace

from job_worker import _error_message_for


async def _slow_job():
    await asyncio.sleep(1.0)
    return "sollte nie erreicht werden"


def test_wait_for_times_out_and_maps_german_message():
    async def run():
        try:
            await asyncio.wait_for(_slow_job(), timeout=0.05)
        except asyncio.TimeoutError as exc:
            return exc
        raise AssertionError("wait_for haette TimeoutError werfen muessen")

    exc = asyncio.run(run())
    spec = SimpleNamespace(error_formatter=lambda err: f"formatter: {err}")
    assert _error_message_for(exc, spec) == "Zeitlimit überschritten — Job abgebrochen"


def test_non_timeout_errors_use_spec_formatter():
    spec = SimpleNamespace(error_formatter=lambda err: f"formatter: {err}")
    assert _error_message_for(ValueError("kaputt"), spec) == "formatter: kaputt"


def test_without_formatter_falls_back_to_str_then_classname():
    spec = SimpleNamespace(error_formatter=None)
    assert _error_message_for(ValueError("kaputt"), spec) == "kaputt"
    assert _error_message_for(ValueError(), spec) == "ValueError"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
