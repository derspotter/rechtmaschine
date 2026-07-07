"""
Cross-worker event broadcasting using PostgreSQL LISTEN/NOTIFY.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Callable, Optional, Set

import select

import psycopg2
from psycopg2 import extensions


DOCUMENTS_CHANNEL = "documents_updates"
SOURCES_CHANNEL = "sources_updates"


class BroadcastHub:
    """Manage asyncio queues for connected subscribers."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queues: Set[asyncio.Queue[str]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        async with self._lock:
            self._queues.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        async with self._lock:
            self._queues.discard(queue)

    def publish(self, message: str) -> None:
        """Schedule message delivery to all subscribers."""

        def _dispatch() -> None:
            stale: Set[asyncio.Queue[str]] = set()
            for queue in list(self._queues):
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                        queue.put_nowait(message)
                    except Exception:
                        stale.add(queue)
            if stale:
                for q in stale:
                    self._queues.discard(q)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            _dispatch()
        else:
            self._loop.call_soon_threadsafe(_dispatch)


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
