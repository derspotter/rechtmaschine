import asyncio
import os
import signal
import socket
import threading
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Type

from sqlalchemy import or_
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from fastapi import HTTPException

from endpoints.agent_memory import MemoryReflectionRequest, _execute_memory_reflection_request
from endpoints.anonymization import AnonymizeJobRequest, _execute_anonymize_request
from endpoints.generation import _execute_generation_request, _format_stream_exception
from endpoints.ocr import OcrJobRequest, _execute_ocr_request
from endpoints.query import QueryRequest, _execute_query_request
from endpoints.research_sources import _execute_research_request
from main import apply_schema_migrations
from models import (
    AnonymizeJob,
    GeneratedDraft,
    GenerationJob,
    MemoryReflectionJob,
    OcrJob,
    QueryJob,
    ResearchJob,
    ResearchRun,
    User,
)
from shared import GenerationRequest, ResearchRequest

JOB_POLL_INTERVAL_SEC = float((os.getenv("JOB_WORKER_POLL_INTERVAL_SEC", "1.0") or "1.0").strip())
JOB_HEARTBEAT_INTERVAL_SEC = float((os.getenv("JOB_WORKER_HEARTBEAT_INTERVAL_SEC", "5.0") or "5.0").strip())
JOB_STALE_AFTER_SEC = int((os.getenv("JOB_WORKER_STALE_AFTER_SEC", "120") or "120").strip())
JOB_EXECUTION_TIMEOUT_SEC = float((os.getenv("JOB_EXECUTION_TIMEOUT_SECONDS", "3600") or "3600").strip())
JOB_WORKER_ID = (
    os.getenv("JOB_WORKER_ID")
    or f"{socket.gethostname()}-{os.getpid()}"
)


@dataclass(frozen=True)
class JobSpec:
    name: str
    model: Type[Any]
    request_model: Type[Any]
    execute_fn: Callable[[Any, Session, User], Any]
    result_id_field: Optional[str] = None
    error_formatter: Optional[Callable[[Exception], str]] = None
    # Result ORM model whose rows carry a job_id (GeneratedDraft / ResearchRun).
    # Set only for job types that persist a durable artifact — enables dedup on
    # requeue and signals that the executor accepts a job_id keyword.
    result_model: Optional[Type[Any]] = None


JOB_SPECS = [
    JobSpec(
        name="query",
        model=QueryJob,
        request_model=QueryRequest,
        execute_fn=_execute_query_request,
    ),
    JobSpec(
        name="generation",
        model=GenerationJob,
        request_model=GenerationRequest,
        execute_fn=_execute_generation_request,
        result_id_field="draft_id",
        error_formatter=_format_stream_exception,
        result_model=GeneratedDraft,
    ),
    JobSpec(
        name="research",
        model=ResearchJob,
        request_model=ResearchRequest,
        execute_fn=_execute_research_request,
        result_id_field="research_run_id",
        result_model=ResearchRun,
    ),
    JobSpec(
        name="memory_reflection",
        model=MemoryReflectionJob,
        request_model=MemoryReflectionRequest,
        execute_fn=_execute_memory_reflection_request,
    ),
    JobSpec(
        name="ocr",
        model=OcrJob,
        request_model=OcrJobRequest,
        execute_fn=_execute_ocr_request,
        error_formatter=lambda err: (
            str(err.detail) if isinstance(err, HTTPException) else (str(err) or err.__class__.__name__)
        ),
    ),
    JobSpec(
        name="anonymize",
        model=AnonymizeJob,
        request_model=AnonymizeJobRequest,
        execute_fn=_execute_anonymize_request,
        error_formatter=lambda err: (
            str(err.detail) if isinstance(err, HTTPException) else (str(err) or err.__class__.__name__)
        ),
    ),
]


class JobHeartbeat:
    def __init__(self, model: Type[Any], job_id: uuid.UUID, worker_id: str):
        self.model = model
        self.job_id = job_id
        self.worker_id = worker_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        # A job can fail before start() (e.g. owner lookup, model_validate). Never
        # join a thread that was never started — join() would raise RuntimeError and
        # mask the original failure.
        self._stop.set()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(JOB_HEARTBEAT_INTERVAL_SEC):
            db = SessionLocal()
            try:
                job = (
                    db.query(self.model)
                    .filter(
                        self.model.id == self.job_id,
                        self.model.status == "running",
                        self.model.claimed_by == self.worker_id,
                    )
                    .first()
                )
                if not job:
                    return
                job.heartbeat_at = datetime.utcnow()
                job.updated_at = datetime.utcnow()
                db.commit()
            except Exception as exc:
                db.rollback()
                print(f"[JOB WORKER] Heartbeat update failed for {self.model.__tablename__}:{self.job_id}: {exc}")
            finally:
                db.close()


def _claim_next_job(spec: JobSpec) -> Optional[uuid.UUID]:
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        job = (
            db.query(spec.model)
            .filter(
                spec.model.status == "queued",
                or_(spec.model.available_at.is_(None), spec.model.available_at <= now),
            )
            .order_by(spec.model.available_at.asc(), spec.model.created_at.asc())
            .with_for_update(skip_locked=True)
            .first()
        )
        if not job:
            return None

        job.status = "running"
        job.claimed_by = JOB_WORKER_ID
        job.claimed_at = now
        job.heartbeat_at = now
        job.started_at = job.started_at or now
        job.updated_at = now
        job.attempt_count = int(job.attempt_count or 0) + 1
        db.commit()
        return job.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _reconcile_stale_running_jobs() -> None:
    cutoff = datetime.utcnow() - timedelta(seconds=JOB_STALE_AFTER_SEC)
    for spec in JOB_SPECS:
        db = SessionLocal()
        try:
            jobs = (
                db.query(spec.model)
                .filter(
                    spec.model.status == "running",
                    or_(spec.model.heartbeat_at.is_(None), spec.model.heartbeat_at < cutoff),
                )
                .all()
            )
            for job in jobs:
                job.status = "failed"
                job.error_message = job.error_message or "Job interrupted by worker restart"
                job.completed_at = datetime.utcnow()
                job.updated_at = datetime.utcnow()
                job.claimed_by = None
                job.claimed_at = None
                job.heartbeat_at = None
            if jobs:
                db.commit()
                print(f"[JOB WORKER] Marked {len(jobs)} stale jobs as failed in {spec.model.__tablename__}")
        except Exception as exc:
            db.rollback()
            print(f"[JOB WORKER] Failed stale-job reconciliation for {spec.model.__tablename__}: {exc}")
        finally:
            db.close()


# (model, job_id) of the job currently executing, for the SIGTERM requeue handler.
_CURRENT_JOB: Optional[tuple] = None


def _requeue_current_and_exit(signum, frame) -> None:
    """SIGTERM/SIGINT: put the in-flight job back to 'queued' and exit immediately.

    Deploy restarts previously killed the running job permanently ("Job interrupted
    by worker restart"). Re-running from scratch is safe: the jlawyer reflect
    persists its seen-state only after the final proposal, consolidation is
    stateless. The handler uses its own session; the main thread's open
    transaction dies with the process and rolls back in Postgres."""
    current = _CURRENT_JOB
    # flush=True throughout: os._exit() skips buffer flushing, silently eating prints.
    print(f"[JOB WORKER] Caught signal {signum}, shutting down", flush=True)
    if current is not None:
        model, job_id = current
        db = SessionLocal()
        try:
            job = db.query(model).filter(model.id == job_id).first()
            if job is not None and job.status == "running":
                job.status = "queued"
                job.claimed_by = None
                job.claimed_at = None
                job.heartbeat_at = None
                job.available_at = datetime.utcnow()
                job.updated_at = datetime.utcnow()
                db.commit()
                print(f"[JOB WORKER] Requeued in-flight job {job_id} for pickup after restart", flush=True)
        except Exception as exc:
            print(f"[JOB WORKER] Requeue on shutdown failed: {exc}", flush=True)
        finally:
            db.close()
    os._exit(0)


async def _run_claimed_job(spec: JobSpec, job_id: uuid.UUID) -> None:
    global _CURRENT_JOB
    db = SessionLocal()
    heartbeat = JobHeartbeat(spec.model, job_id, JOB_WORKER_ID)
    _CURRENT_JOB = (spec.model, job_id)
    try:
        job = db.query(spec.model).filter(spec.model.id == job_id).first()
        if not job:
            return

        # Dedup BEFORE the attempt cap: a persisted artifact from an interrupted run
        # must win over the cap. The SIGTERM requeue re-runs non-idempotent jobs, so a
        # draft/research_run may already exist for this job_id — adopt it instead of
        # producing a duplicate.
        if spec.result_model is not None:
            existing = (
                db.query(spec.result_model)
                .filter(spec.result_model.job_id == job_id)
                .order_by(spec.result_model.created_at.asc())
                .first()
            )
            if existing is not None:
                job.status = "completed"
                job.result_payload = {
                    spec.result_id_field: str(existing.id),
                    "hinweis": "Ergebnis aus unterbrochenem Lauf übernommen",
                }
                if spec.result_id_field:
                    setattr(job, spec.result_id_field, existing.id)
                job.completed_at = datetime.utcnow()
                job.updated_at = datetime.utcnow()
                job.heartbeat_at = datetime.utcnow()
                db.commit()
                print(f"[JOB WORKER] Adopted existing result for {spec.name} job {job_id}, skipped re-run")
                return

        # attempt_count is incremented on claim. More than 3 claims means the job kept
        # getting interrupted or failing — stop retrying to avoid an endless requeue.
        if int(job.attempt_count or 0) > 3:
            job.status = "failed"
            job.error_message = "Mehrfach unterbrochen oder fehlgeschlagen — nicht erneut versucht"
            job.completed_at = datetime.utcnow()
            job.updated_at = datetime.utcnow()
            job.heartbeat_at = datetime.utcnow()
            db.commit()
            print(f"[JOB WORKER] {spec.name} job {job_id} exceeded attempt cap, marked failed")
            return

        user = db.query(User).filter(User.id == job.owner_id).first()
        if not user:
            raise RuntimeError(f"Owner user not found for {spec.name} job")

        body = spec.request_model.model_validate(dict(job.request_payload or {}))
        print(f"[JOB WORKER] Starting {spec.name} job {job_id}")
        heartbeat.start()
        if spec.result_model is not None:
            coro = spec.execute_fn(body, db, user, job_id=job_id)
        else:
            coro = spec.execute_fn(body, db, user)
        # Harte Obergrenze zusaetzlich zu den Client-Timeouts. Greift nur an await-Punkten,
        # die to_thread-basierten Provider awaiten. Verhindert, dass ein haengender Job
        # den seriellen Worker dauerhaft blockiert.
        result = await asyncio.wait_for(coro, timeout=JOB_EXECUTION_TIMEOUT_SEC)
        heartbeat.stop()

        db.expire_all()
        job = db.query(spec.model).filter(spec.model.id == job_id).first()
        if not job:
            return

        job.status = "completed"
        job.result_payload = result if isinstance(result, dict) else result.model_dump()
        if spec.result_id_field:
            result_id = None
            if isinstance(result, dict):
                result_id = result.get(spec.result_id_field)
            else:
                result_id = getattr(result, spec.result_id_field, None)
            if result_id:
                setattr(job, spec.result_id_field, uuid.UUID(str(result_id)))
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        job.heartbeat_at = datetime.utcnow()
        db.commit()
        print(f"[JOB WORKER] Completed {spec.name} job {job_id}")
    except Exception as exc:
        # stop() in its own guard so a heartbeat problem never masks the original
        # failure and never skips the mark-failed path below.
        try:
            heartbeat.stop()
        except Exception as hb_exc:
            print(f"[JOB WORKER] Heartbeat stop failed for {spec.name} job {job_id}: {hb_exc}")
        print(f"[JOB WORKER] {spec.name} job failed {job_id}: {exc}")
        traceback.print_exc()
        db.rollback()
        try:
            failed_job = db.query(spec.model).filter(spec.model.id == job_id).first()
            if failed_job:
                formatter = spec.error_formatter or (lambda err: str(err) or err.__class__.__name__)
                if isinstance(exc, asyncio.TimeoutError):
                    error_message = "Zeitlimit überschritten — Job abgebrochen"
                else:
                    error_message = formatter(exc)
                failed_job.status = "failed"
                failed_job.error_message = error_message
                failed_job.completed_at = datetime.utcnow()
                failed_job.updated_at = datetime.utcnow()
                failed_job.heartbeat_at = datetime.utcnow()
                db.commit()
        except Exception as nested_exc:
            db.rollback()
            print(f"[JOB WORKER] Failed to persist failure for {spec.name} job {job_id}: {nested_exc}")
    finally:
        _CURRENT_JOB = None
        db.close()


async def run_worker_loop() -> None:
    signal.signal(signal.SIGTERM, _requeue_current_and_exit)
    signal.signal(signal.SIGINT, _requeue_current_and_exit)
    Base.metadata.create_all(bind=engine)
    apply_schema_migrations()
    _reconcile_stale_running_jobs()
    print(f"[JOB WORKER] Started as {JOB_WORKER_ID}")

    rotation_index = 0
    while True:
        try:
            claimed = False
            for offset in range(len(JOB_SPECS)):
                spec = JOB_SPECS[(rotation_index + offset) % len(JOB_SPECS)]
                job_id = _claim_next_job(spec)
                if job_id:
                    rotation_index = (rotation_index + offset + 1) % len(JOB_SPECS)
                    claimed = True
                    await _run_claimed_job(spec, job_id)
                    break

            if not claimed:
                _reconcile_stale_running_jobs()
                await asyncio.sleep(JOB_POLL_INTERVAL_SEC)
        except Exception as loop_exc:
            # Transient DB/network errors must not crash the container. Log and back
            # off so a restart loop does not hammer the DB.
            print(f"[WORKER ERROR] Loop iteration failed: {loop_exc}")
            traceback.print_exc()
            await asyncio.sleep(5.0)


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
