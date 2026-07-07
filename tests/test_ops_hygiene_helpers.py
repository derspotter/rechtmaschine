# Prueft drei reine Logik-Bausteine aus Task 15 (Ops-Hygiene), isoliert von
# DB/FastAPI/GPU-Workern:
# 1. track_background_task (app/shared.py): haelt eine Task im Modul-Set bis
#    sie fertig ist, discarded sie danach (GC-Schutz-Pattern).
# 2. Die Stuck-Status-Reconciliation-Zuordnung (app/main.py:
#    STUCK_DOCUMENT_STATUS_MAP) + die 1h-Alters-Guard-Logik.
# 3. Die Gemini-Upload-Staleness-Erkennung (mtime/size-Vergleich) aus
#    ensure_document_on_gemini (app/shared.py).
#
# Aufruf: python3 tests/test_ops_hygiene_helpers.py
# Keine laufende DB/App noetig - die Funktionen werden 1:1 nachgebaut (statt
# shared.py/main.py zu importieren, was FastAPI/DB/Anthropic/Gemini-Clients
# zieht), analog zum Muster in test_job_heartbeat_stop_guard.py.

import asyncio
from datetime import datetime, timedelta


# --- 1. track_background_task -----------------------------------------------

_background_tasks: set = set()


def track_background_task(task):
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _tiny_coro():
    await asyncio.sleep(0.01)


def test_track_background_task_discards_on_done():
    async def run():
        task = track_background_task(asyncio.get_running_loop().create_task(_tiny_coro()))
        assert task in _background_tasks, "Task sollte direkt nach dem Start im Set sein"
        await task
        # add_done_callback wird erst im naechsten Loop-Tick ausgefuehrt.
        await asyncio.sleep(0)
        assert task not in _background_tasks, "Task sollte nach done() discarded sein"

    asyncio.run(run())
    print("OK: track_background_task haelt die Task bis done() und discarded danach")


# --- 2. Stuck-Status-Reconciliation ------------------------------------------

STUCK_DOCUMENT_STATUS_MAP = {
    "anonymizing": "anon_failed",
    "anon_pending": "anon_failed",
    "ocr_processing": "ocr_failed",
    "ocr_pending": "ocr_failed",
}
STUCK_DOCUMENT_MAX_AGE = timedelta(hours=1)


class _FakeDoc:
    def __init__(self, processing_status, created_at):
        self.processing_status = processing_status
        self.created_at = created_at


def _reconcile(documents, now):
    cutoff = now - STUCK_DOCUMENT_MAX_AGE
    reconciled = []
    for doc in documents:
        if doc.processing_status not in STUCK_DOCUMENT_STATUS_MAP:
            continue
        if doc.created_at >= cutoff:
            continue
        doc.processing_status = STUCK_DOCUMENT_STATUS_MAP[doc.processing_status]
        reconciled.append(doc)
    return reconciled


def test_reconcile_only_touches_old_transient_statuses():
    now = datetime.utcnow()
    fresh_anonymizing = _FakeDoc("anonymizing", now - timedelta(minutes=5))
    old_anonymizing = _FakeDoc("anonymizing", now - timedelta(hours=2))
    old_ocr_pending = _FakeDoc("ocr_pending", now - timedelta(hours=3))
    old_completed = _FakeDoc("completed", now - timedelta(hours=3))

    reconciled = _reconcile(
        [fresh_anonymizing, old_anonymizing, old_ocr_pending, old_completed], now
    )

    assert fresh_anonymizing.processing_status == "anonymizing", (
        "Ein erst vor 5 Minuten erstelltes Dokument darf NICHT reconciled werden "
        "(schuetzt gegen falsch-positive Treffer bei knapper Altersgrenze)"
    )
    assert old_anonymizing.processing_status == "anon_failed"
    assert old_ocr_pending.processing_status == "ocr_failed"
    assert old_completed.processing_status == "completed", (
        "Nicht-transiente Status duerfen nie angefasst werden"
    )
    assert len(reconciled) == 2
    print("OK: Reconciliation trifft nur alte, transiente Status")


# --- 3. Gemini-Upload-Staleness (mtime/size-Vergleich) -----------------------


def _should_reuse(cached_stat, current_stat):
    """Spiegelt die Vergleichslogik aus ensure_document_on_gemini."""
    if not current_stat or not cached_stat:
        return True  # keine Cache-Info -> altes Verhalten (reuse versuchen)
    if cached_stat.get("path") != current_stat["path"]:
        return True
    if cached_stat.get("mtime") != current_stat["mtime"] or cached_stat.get("size") != current_stat["size"]:
        return False
    return True


def test_gemini_reuse_detects_changed_file():
    cached = {"path": "/app/uploads/a.pdf", "mtime": 100.0, "size": 500}
    same_file = {"path": "/app/uploads/a.pdf", "mtime": 100.0, "size": 500}
    reocred_file = {"path": "/app/uploads/a.pdf", "mtime": 205.5, "size": 812}

    assert _should_reuse(cached, same_file) is True
    assert _should_reuse(cached, reocred_file) is False, (
        "Ein re-OCR'tes File (gleicher Pfad, andere mtime/size) darf NICHT "
        "wiederverwendet werden -- sonst liefert Gemini bis zu 48h alten Inhalt"
    )
    print("OK: Gemini-Reuse erkennt geaenderte Datei ueber mtime/size")


if __name__ == "__main__":
    test_track_background_task_discards_on_done()
    test_reconcile_only_touches_old_transient_statuses()
    test_gemini_reuse_detects_changed_file()
    print("Alle Ops-Hygiene-Tests bestanden")
