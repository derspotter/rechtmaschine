"""Rechtmaschine Transcribe Service (debian).

Kleiner HTTP-Dienst für Sprach-Transkription (sipgate-Voicemails u.ä.) mit
faster-whisper. Läuft als systemd-User-Service neben dem OCR-Manager (:8004)
auf Port 8005.

- Lazy Load: das Modell wird erst beim ersten Request geladen und nach
  TRANSCRIBE_IDLE_UNLOAD_S Sekunden Leerlauf wieder entladen (VRAM frei
  für OCR/Anon auf derselben GPU).
- Device-Fallback: erst CUDA (int8_float16), bei Fehlern CPU (int8). Gilt
  auch für CUDA-Fehler zur Laufzeit (fehlende libcublas o.ä. zeigen sich
  erst beim ersten Rechenschritt, nicht beim Laden).
- Ein Request zur Zeit (Lock) — die GPU-Nutzung bleibt seriell.

Endpunkte:
    GET  /health      Status, geladenes Modell, Device
    POST /transcribe  multipart file=<audio>, optional language (default de)
"""

import asyncio
import io
import logging
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transcribe")

MODEL_NAME = os.getenv("TRANSCRIBE_MODEL", "large-v3")
FALLBACK_MODEL_NAME = os.getenv("TRANSCRIBE_FALLBACK_MODEL", "large-v3-turbo")
PORT = int(os.getenv("TRANSCRIBE_PORT", "8005"))
IDLE_UNLOAD_S = float(os.getenv("TRANSCRIBE_IDLE_UNLOAD_S", "300"))
DEFAULT_LANGUAGE = os.getenv("TRANSCRIBE_DEFAULT_LANGUAGE", "de")
MAX_UPLOAD_BYTES = int(os.getenv("TRANSCRIBE_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
# Sprach-Umschalter: Voicemails sind überwiegend (akzentgefärbtes) Deutsch,
# aber ~5 % sind englisch gesprochen — mit erzwungenem de übersetzt Whisper
# die still ins Deutsche. Reines Auto-Detect kippt umgekehrt akzentgefärbte
# deutsche Anrufe nach Englisch (Benchmark 14.07.2026, 177 Voicemails).
# Kompromiss: bei language=de erst Sprache erkennen und NUR bei
# p(en) >= Schwelle auf Englisch umschalten. 0 deaktiviert den Umschalter.
EN_SWITCH_P = float(os.getenv("TRANSCRIBE_EN_SWITCH_P", "0.9"))

app = FastAPI(title="Rechtmaschine Transcribe Service")

_lock = threading.Lock()
_model: Any = None
_model_device: Optional[str] = None
_model_name_loaded: Optional[str] = None
_last_used = 0.0


class SleepInhibitor:
    """Reference-counted systemd 'block' sleep inhibitor.

    Holds a host-level suspend lock while a transcription is in flight, so the
    box can auto-suspend when idle (WOL wakes it on demand) but never naps
    mid-job. Mirrors the inhibitor in service_manager.py. The lock is a
    ``tail --pid=<us>`` child, so it vanishes automatically if this service
    dies uncleanly. First concurrent job takes the lock; last one releases.
    Disable with SLEEP_INHIBIT_WHILE_BUSY=0.
    """

    def __init__(self, who: str, why: str) -> None:
        self.who = who
        self.why = why
        self.enabled = os.getenv("SLEEP_INHIBIT_WHILE_BUSY", "1").lower() not in ("0", "false", "no")
        self._proc: Optional[subprocess.Popen] = None
        self._count = 0
        self._guard = threading.Lock()
        self._warned = False

    def _start(self) -> None:
        binary = shutil.which("systemd-inhibit")
        if not binary:
            if not self._warned:
                log.warning("systemd-inhibit not found; sleep inhibition disabled")
                self._warned = True
            return
        try:
            proc = subprocess.Popen(
                [binary, "--what=sleep", f"--who={self.who}", f"--why={self.why}",
                 "--mode=block", "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            time.sleep(0.3)
            if proc.poll() is not None:
                err = proc.stderr.read() if proc.stderr else b""
                if not self._warned:
                    log.warning(
                        "Sleep inhibitor DENIED - machine may suspend mid-job. Install the "
                        "polkit rule allowing inhibit-block-sleep for this user. (%s)",
                        err.decode(errors="ignore").strip()[:160],
                    )
                    self._warned = True
                return
            self._proc = proc
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to acquire sleep inhibitor: %s", exc)

    def _stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self._proc = None

    @contextmanager
    def hold(self):
        if not self.enabled:
            yield
            return
        with self._guard:
            self._count += 1
            if self._count == 1:
                self._start()
        try:
            yield
        finally:
            with self._guard:
                self._count -= 1
                if self._count == 0:
                    self._stop()


SLEEP_INHIBITOR = SleepInhibitor(
    who="rechtmaschine-transcribe",
    why="Aktive Transkription (Whisper)",
)


def _load_model(cpu_only: bool = False) -> Any:
    """Modell laden; CUDA zuerst, CPU als Fallback. Hält _lock des Aufrufers."""
    global _model, _model_device, _model_name_loaded
    if _model is not None:
        return _model

    from faster_whisper import WhisperModel

    attempts = [
        (MODEL_NAME, "cuda", "int8_float16"),
        (MODEL_NAME, "cpu", "int8"),
        (FALLBACK_MODEL_NAME, "cpu", "int8"),
    ]
    if cpu_only:
        attempts = [a for a in attempts if a[1] == "cpu"]
    last_error: Optional[Exception] = None
    for name, device, compute_type in attempts:
        try:
            start = time.monotonic()
            model = WhisperModel(name, device=device, compute_type=compute_type)
            log.info(
                "Modell %s geladen (device=%s, compute=%s, %.1fs)",
                name, device, compute_type, time.monotonic() - start,
            )
            _model, _model_device, _model_name_loaded = model, device, name
            return model
        except Exception as exc:  # noqa: BLE001 — jeder Ladefehler soll zum nächsten Versuch führen
            last_error = exc
            log.warning("Laden %s auf %s fehlgeschlagen: %s", name, device, exc)
    raise RuntimeError(f"Kein Whisper-Modell ladbar: {last_error}")


def _unload_model() -> None:
    global _model, _model_device, _model_name_loaded
    if _model is None:
        return
    log.info("Entlade Modell %s (idle > %.0fs)", _model_name_loaded, IDLE_UNLOAD_S)
    _model = None
    _model_device = None
    _model_name_loaded = None
    import gc

    gc.collect()


def _p_english(model: Any, audio: Any) -> float:
    try:
        _lang, _p, all_probs = model.detect_language(audio=audio)
        return next((p for code, p in all_probs if code == "en"), 0.0)
    except Exception as exc:  # noqa: BLE001 — Detection darf nie die Transkription verhindern
        log.warning("Sprach-Detection fehlgeschlagen: %s", exc)
        return 0.0


def _run_transcription(model: Any, audio: Any, language: Optional[str], beam_size: int) -> tuple:
    effective = language
    if language == "de" and EN_SWITCH_P > 0:
        p_en = _p_english(model, audio)
        if p_en >= EN_SWITCH_P:
            effective = "en"
            log.info("EN-Umschalter aktiv (p_en=%.2f)", p_en)
    segments_iter, info = model.transcribe(
        audio,
        language=effective or None,
        beam_size=beam_size,
        vad_filter=True,
    )
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments_iter
    ]
    return segments, info


def _transcribe_bytes(data: bytes, language: Optional[str], beam_size: int) -> dict:
    global _last_used
    with SLEEP_INHIBITOR.hold(), _lock:
        model = _load_model()
        start = time.monotonic()
        from faster_whisper.audio import decode_audio

        audio = decode_audio(io.BytesIO(data))
        try:
            segments, info = _run_transcription(model, audio, language, beam_size)
        except Exception as exc:  # noqa: BLE001
            if _model_device != "cuda":
                raise
            # CUDA-Laufzeitfehler (fehlende libcublas, kaputter Treiberzustand)
            # tauchen erst hier auf, nicht in _load_model — der cuda-Ladeversuch
            # "gelingt", weil ctranslate2 die Libs lazy lädt. Einmal CPU-only
            # neu laden und wiederholen statt 503.
            log.warning("CUDA-Laufzeitfehler, wiederhole auf CPU: %s", exc)
            _unload_model()
            model = _load_model(cpu_only=True)
            segments, info = _run_transcription(model, audio, language, beam_size)
        elapsed = time.monotonic() - start
        _last_used = time.monotonic()
    text = " ".join(s["text"] for s in segments).strip()
    return {
        "text": text,
        "segments": segments,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_s": round(info.duration, 2),
        "elapsed_s": round(elapsed, 2),
        "model": _model_name_loaded,
        "device": _model_device,
    }


async def _idle_unloader() -> None:
    while True:
        await asyncio.sleep(30)
        if _model is not None and _last_used and time.monotonic() - _last_used > IDLE_UNLOAD_S:
            if _lock.acquire(blocking=False):
                try:
                    _unload_model()
                finally:
                    _lock.release()


@app.on_event("startup")
async def _startup() -> None:
    asyncio.get_event_loop().create_task(_idle_unloader())


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": _model is not None,
        "device": _model_device,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    beam_size: int = Form(5),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Leere Audiodatei")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audiodatei zu groß")
    lang = language.strip() or None
    if lang == "auto":
        lang = None
    try:
        result = await asyncio.to_thread(_transcribe_bytes, data, lang, beam_size)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    log.info(
        "Transkribiert: %s (%.1fs Audio, %.1fs Rechenzeit, %s/%s)",
        file.filename, result["duration_s"], result["elapsed_s"],
        result["model"], result["device"],
    )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
