"""
Cross-worker event broadcasting using PostgreSQL LISTEN/NOTIFY.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Callable, Dict, Optional

import select

import psycopg2
from psycopg2 import extensions


DOCUMENTS_CHANNEL = "documents_updates"
SOURCES_CHANNEL = "sources_updates"


class BroadcastHub:
    """Manage asyncio queues for connected subscribers, scoped per user.

    Each subscriber queue carries the owning user's id. A published payload is
    delivered only to subscribers whose user id matches the payload's
    ``owner_id``. Payloads without an ``owner_id`` (system events such as
    ``resync``) are delivered to everyone. This prevents one user's SSE stream
    from receiving events that carry another user's data (filenames = client
    names, document_ids, case_ids).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        # queue -> owner user id (as str); None means "receives all events".
        self._queues: Dict[asyncio.Queue[str], Optional[str]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, user_id: Optional[str] = None) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        owner = str(user_id) if user_id is not None else None
        async with self._lock:
            self._queues[queue] = owner
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        async with self._lock:
            self._queues.pop(queue, None)

    def publish(self, message: str) -> None:
        """Schedule message delivery to the subscribers it is scoped to."""

        # Extract the target owner id from the payload. Events without an
        # owner_id (e.g. resync) go to every subscriber.
        target_owner: Optional[str] = None
        try:
            parsed = json.loads(message)
            if isinstance(parsed, dict) and parsed.get("owner_id") is not None:
                target_owner = str(parsed["owner_id"])
        except Exception:
            target_owner = None

        def _dispatch() -> None:
            stale = []
            for queue, owner in list(self._queues.items()):
                # Deliver ownerless events to all; owned events only to the owner.
                if target_owner is not None and owner is not None and owner != target_owner:
                    continue
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                        queue.put_nowait(message)
                    except Exception:
                        stale.append(queue)
            for q in stale:
                self._queues.pop(q, None)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            _dispatch()
        else:
            self._loop.call_soon_threadsafe(_dispatch)


class SSETicketStore:
    """One-time, short-lived tickets for authenticating SSE connections.

    EventSource cannot send an Authorization header, so the frontend fetches a
    ticket over an authenticated POST and then connects with ``?ticket=...``.
    Tickets are single-use (popped on connect) and expire after ``ttl_seconds``.

    NOTE: this store is PROCESS-LOCAL (a plain dict guarded by a lock). There is
    exactly one app container, so that is sufficient. A multi-process / multi-
    container deployment would need a shared store (Redis or a DB table) instead.
    """

    def __init__(self, ttl_seconds: int = 60):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        # ticket -> (user_id: str, expiry: float epoch seconds)
        self._tickets: Dict[str, tuple] = {}

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    def _prune_locked(self, now: float) -> None:
        expired = [t for t, (_uid, exp) in self._tickets.items() if exp <= now]
        for t in expired:
            del self._tickets[t]

    def issue(self, user_id: str) -> str:
        ticket = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            self._tickets[ticket] = (str(user_id), now + self._ttl)
        return ticket

    def consume(self, ticket: str) -> Optional[str]:
        """Return the user_id for a valid, unexpired ticket and invalidate it.

        Returns None for unknown, already-used, or expired tickets.
        """
        if not ticket:
            return None
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            entry = self._tickets.pop(ticket, None)
        if not entry:
            return None
        user_id, expiry = entry
        if expiry <= now:
            return None
        return user_id


class PostgresListener:
    """Listen on a PostgreSQL channel and push payloads into the broadcast hub."""

    def __init__(self, dsn: str, hub: BroadcastHub, channel: str = DOCUMENTS_CHANNEL):
        self._dsn = dsn
        self._hub = hub
        self._channel = channel
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # Outer reconnect loop: any connection error (dropped socket, DB restart,
        # network blip) must not kill the listener thread. We reconnect with an
        # exponential backoff (1s -> 30s, capped), re-LISTEN the channel, and after
        # a *successful reconnect* publish a resync event so clients refetch state
        # they may have missed while the listener was down. The thread only exits on
        # an explicit stop().
        backoff = 1.0
        first_connect = True
        while not self._stop.is_set():
            conn = None
            try:
                conn = psycopg2.connect(self._dsn)
                conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = conn.cursor()
                cursor.execute(f"LISTEN {self._channel};")

                if first_connect:
                    first_connect = False
                    print(f"[LISTENER] Connected and listening on {self._channel}")
                else:
                    print(f"[LISTENER] Reconnected on {self._channel}, publishing resync")
                    self._hub.publish(json.dumps({"type": "resync"}))

                # Successful connection -> reset backoff for the next failure.
                backoff = 1.0

                while not self._stop.is_set():
                    ready = select.select([conn], [], [], 0.5)
                    if not ready[0]:
                        continue
                    conn.poll()
                    while conn.notifies:
                        notify = conn.notifies.pop(0)
                        payload = notify.payload
                        if payload:
                            self._hub.publish(payload)
            except Exception as exc:
                print(f"[LISTENER] Error on {self._channel}: {exc}")
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

            if self._stop.is_set():
                break

            print(f"[LISTENER] Reconnecting to {self._channel} in {backoff:.0f}s")
            # Interruptible sleep: returns immediately if stop() is called.
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)


def notify_postgres(dsn: str, payload: str, channel: str = DOCUMENTS_CHANNEL) -> None:
    """Send payload to all listeners via PostgreSQL NOTIFY."""
    try:
        conn = psycopg2.connect(dsn)
        conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT pg_notify(%s, %s);",
            (channel, payload),
        )
        cursor.close()
        conn.close()
    except Exception as exc:
        print(f"notify_postgres error: {exc}")
