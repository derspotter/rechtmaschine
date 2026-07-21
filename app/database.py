"""
Database configuration and session management for Rechtmaschine
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rechtmaschine:password@postgres:5432/rechtmaschine_db")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=10,        # Connection pool size
    max_overflow=20      # Allow up to 20 extra connections
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def persist_with_db_retry(db, obj, attempts=6, base_delay=2.0, what="object"):
    """Add+commit `obj`, surviving short DB outages.

    A postgres restart (e.g. a `docker compose restart` sweeping the stack)
    kills in-flight commits with OperationalError — which on 2026-07-21 threw
    away a fully paid 19-minute LLM generation because the draft INSERT was
    the last step. Retries only connection-level errors (Operational/
    InterfaceError) with backoff up to ~60s total; integrity errors still
    raise immediately. Blocking sleep by design: callers are the serial job
    worker or an already-broken request path.
    """
    import time

    from sqlalchemy.exc import InterfaceError, OperationalError

    for attempt in range(1, attempts + 1):
        try:
            db.add(obj)
            db.commit()
            db.refresh(obj)
            return obj
        except (OperationalError, InterfaceError) as exc:
            try:
                db.rollback()
            except Exception:
                pass
            if attempt == attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
            print(
                f"[DB-RETRY] commit of {what} failed "
                f"(attempt {attempt}/{attempts}): {exc.__class__.__name__} "
                f"- retrying in {delay:.0f}s"
            )
            time.sleep(delay)


# Base class for models
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that provides a database session.
    Automatically closes the session after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
