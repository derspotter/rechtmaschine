import asyncio
import os
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
from endpoints.generation import _execute_generation_request, _format_stream_exception
from endpoints.query import QueryRequest, _execute_query_request
from endpoints.research_sources import _execute_research_request
from main import apply_schema_migrations
from models import GenerationJob, QueryJob, ResearchJob, User
from shared import GenerationRequest, ResearchRequest

JOB_POLL_INTERVAL_SEC = float((os.getenv("JOB_WORKER_POLL_INTERVAL_SEC", "1.0") or "1.0").strip())
JOB_HEARTBEAT_INTERVAL_SEC = float((os.getenv("JOB_WORKER_HEARTBEAT_INTERVAL_SEC", "5.0") or "5.0").strip())
JOB_STALE_AFTER_SEC = int((os.getenv("JOB_WORKER_STALE_AFTER_SEC", "120") or "120").strip())
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
    ),
    JobSpec(
        name="research",
        model=ResearchJob,
        request_model=ResearchRequest,
        execute_fn=_execute_research_request,
        result_id_field="research_run_id",
    ),
]


class JobHeartbeat:
    def __init__(self, model: Type[Any], job_id: uuid.UUID, worker_id: str):
        self.model = model
        self.job_id = job_id
        self.worker_id = worker_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
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


async def _run_claimed_job(spec: JobSpec, job_id: uuid.UUID) -> None:
    db = SessionLocal()
    heartbeat = JobHeartbeat(spec.model, job_id, JOB_WORKER_ID)
    try:
        job = db.query(spec.model).filter(spec.model.id == job_id).first()
        if not job:
            return

        user = db.query(User).filter(User.id == job.owner_id).first()
        if not user:
            raise RuntimeError(f"Owner user not found for {spec.name} job")

        body = spec.request_model.model_validate(dict(job.request_payload or {}))
        print(f"[JOB WORKER] Starting {spec.name} job {job_id}")
        heartbeat.start()
        result = await spec.execute_fn(body, db, user)
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
        heartbeat.stop()
        print(f"[JOB WORKER] {spec.name} job failed {job_id}: {exc}")
        traceback.print_exc()
        db.rollback()
        try:
            failed_job = db.query(spec.model).filter(spec.model.id == job_id).first()
            if failed_job:
                formatter = spec.error_formatter or (lambda err: str(err) or err.__class__.__name__)
                failed_job.status = "failed"
                failed_job.error_message = formatter(exc)
                failed_job.completed_at = datetime.utcnow()
                failed_job.updated_at = datetime.utcnow()
                failed_job.heartbeat_at = datetime.utcnow()
                db.commit()
        except Exception as nested_exc:
            db.rollback()
            print(f"[JOB WORKER] Failed to persist failure for {spec.name} job {job_id}: {nested_exc}")
    finally:
        db.close()


async def run_worker_loop() -> None:
    Base.metadata.create_all(bind=engine)
    apply_schema_migrations()
    _reconcile_stale_running_jobs()
    print(f"[JOB WORKER] Started as {JOB_WORKER_ID}")

    rotation_index = 0
    while True:
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


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
