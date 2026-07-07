"""Ad-hoc test for the SSE one-time ticket store (Task 11).

Pure in-memory logic, no DB and no running app required.
Prueft: Ausstellen/Einloesen, Einmalverwendung, unbekannte Tickets, Ablauf.

Run:
    python3 tests/test_sse_ticket_store.py
Exit code 0 = alle Checks bestanden.
"""

import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# events.py imports psycopg2 at module level (only used by the listener/NOTIFY
# helpers, not by SSETicketStore). Stub it so this pure-logic test runs on any
# host without the DB driver installed.
if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_extensions = types.ModuleType("psycopg2.extensions")
    fake_extensions.ISOLATION_LEVEL_AUTOCOMMIT = 0
    fake_psycopg2.extensions = fake_extensions
    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extensions"] = fake_extensions

from events import SSETicketStore


def test_issue_and_consume():
    store = SSETicketStore(ttl_seconds=60)
    ticket = store.issue("user-1")
    assert isinstance(ticket, str) and ticket
    assert store.consume(ticket) == "user-1"


def test_single_use():
    store = SSETicketStore(ttl_seconds=60)
    ticket = store.issue("user-2")
    assert store.consume(ticket) == "user-2"
    # Second use must fail (ticket was popped on first consume).
    assert store.consume(ticket) is None


def test_unknown_and_empty():
    store = SSETicketStore(ttl_seconds=60)
    assert store.consume("does-not-exist") is None
    assert store.consume("") is None
    assert store.consume(None) is None


def test_expiry():
    store = SSETicketStore(ttl_seconds=60)
    ticket = store.issue("user-3")
    # Force the stored expiry into the past to simulate a lapsed ticket.
    user_id, _expiry = store._tickets[ticket]
    store._tickets[ticket] = (user_id, time.time() - 1.0)
    assert store.consume(ticket) is None
    # Expired ticket must have been pruned lazily on access.
    assert ticket not in store._tickets


def test_user_id_is_str():
    store = SSETicketStore(ttl_seconds=60)
    ticket = store.issue(12345)
    assert store.consume(ticket) == "12345"


if __name__ == "__main__":
    tests = [
        test_issue_and_consume,
        test_single_use,
        test_unknown_and_empty,
        test_expiry,
        test_user_id_is_str,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("All SSE ticket store checks passed.")
