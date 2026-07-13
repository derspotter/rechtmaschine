"""Prueft drei Ops-Hygiene-Bausteine aus Task 15 gegen die ECHTEN Module.

Konsolidierung (Sweep-Follow-up 8, 2026-07-13): vorher waren die Funktionen
hier 1:1 nachgebaut "wegen App-Import-Abhaengigkeiten" — und bereits
gedriftet (STUCK_DOCUMENT_MAX_AGE stand hier auf 1h, real sind es seit
c540dc5 5 Minuten). conftest.py stellt die App-Imports inzwischen bereit,
also importieren wir die Logik direkt.

Run: .venv/bin/python -m pytest tests/test_ops_hygiene_helpers.py -q
"""
import asyncio
from datetime import datetime, timedelta

from main import STUCK_DOCUMENT_MAX_AGE, STUCK_DOCUMENT_STATUS_MAP
from shared import _background_tasks, gemini_upload_is_stale, track_background_task


# --- 1. track_background_task (shared.py) ------------------------------------


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


# --- 2. Stuck-Status-Reconciliation (main.py) ---------------------------------
# Die DB-Query selbst laeuft nur im App-Start; hier wird ihre Auswahl-Semantik
# (Status-Map + Alters-Guard `created_at < cutoff`) gegen die echten
# Konstanten geprueft, damit Map- oder Schwellen-Aenderungen den Test treffen.


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
    fresh_anonymizing = _FakeDoc("anonymizing", now - STUCK_DOCUMENT_MAX_AGE / 2)
    old_anonymizing = _FakeDoc("anonymizing", now - STUCK_DOCUMENT_MAX_AGE * 3)
    old_ocr_pending = _FakeDoc("ocr_pending", now - STUCK_DOCUMENT_MAX_AGE * 3)
    old_completed = _FakeDoc("completed", now - STUCK_DOCUMENT_MAX_AGE * 3)

    reconciled = _reconcile(
        [fresh_anonymizing, old_anonymizing, old_ocr_pending, old_completed], now
    )

    assert fresh_anonymizing.processing_status == "anonymizing", (
        "Ein Dokument juenger als STUCK_DOCUMENT_MAX_AGE darf NICHT reconciled "
        "werden (schuetzt das Autoreload-Fenster)"
    )
    assert old_anonymizing.processing_status == "anon_failed"
    assert old_ocr_pending.processing_status == "ocr_failed"
    assert old_completed.processing_status == "completed", (
        "Nicht-transiente Status duerfen nie angefasst werden"
    )
    assert len(reconciled) == 2


def test_stuck_map_targets_are_terminal_failure_statuses():
    assert set(STUCK_DOCUMENT_STATUS_MAP.values()) == {"anon_failed", "ocr_failed"}
    assert STUCK_DOCUMENT_MAX_AGE <= timedelta(hours=1), (
        "Der Guard soll ein kurzes Autoreload-Fenster schuetzen, keine Stunden — "
        "beim Startup ist jedes Dokument in transientem Status verwaist"
    )


# --- 3. Gemini-Upload-Staleness (shared.gemini_upload_is_stale) ---------------


def test_gemini_reuse_detects_changed_file():
    cached = {"path": "/app/uploads/a.pdf", "mtime": 100.0, "size": 500}
    same_file = {"path": "/app/uploads/a.pdf", "mtime": 100.0, "size": 500}
    reocred_file = {"path": "/app/uploads/a.pdf", "mtime": 205.5, "size": 812}

    assert gemini_upload_is_stale(cached, same_file) is False
    assert gemini_upload_is_stale(cached, reocred_file) is True, (
        "Ein re-OCR'tes File (gleicher Pfad, andere mtime/size) darf NICHT "
        "wiederverwendet werden -- sonst liefert Gemini bis zu 48h alten Inhalt"
    )
    # Ohne Vergleichsbasis (Altbestand ohne Cache-Eintrag, fehlende Datei)
    # bleibt das alte Reuse-Verhalten: nicht stale.
    assert gemini_upload_is_stale(None, same_file) is False
    assert gemini_upload_is_stale(cached, None) is False
    # Anderer Pfad = anderes File, kein mtime-Vergleich moeglich.
    assert gemini_upload_is_stale(cached, {"path": "/b.pdf", "mtime": 1, "size": 2}) is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
