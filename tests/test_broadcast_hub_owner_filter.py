"""Pure-logic tests for BroadcastHub per-user event routing (Sweep 07-07, Follow-up #1).

Prueft: owner-gescopte Events erreichen nur den passenden Subscriber,
ownerlose Events (resync) erreichen alle, None-Owner-Queues empfangen alles,
QueueFull verdraengt die aelteste Nachricht statt Events zu verlieren.

No DB and no running app required; psycopg2 is stubbed if absent.
"""

import asyncio
import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# events.py imports psycopg2 at module level (only used by the listener/NOTIFY
# helpers, not by BroadcastHub). Stub it so this pure-logic test runs on any
# host without the DB driver installed.
_psycopg2_stubbed = False
if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_extensions = types.ModuleType("psycopg2.extensions")
    fake_extensions.ISOLATION_LEVEL_AUTOCOMMIT = 0
    fake_psycopg2.extensions = fake_extensions
    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.extensions"] = fake_extensions
    _psycopg2_stubbed = True

from events import BroadcastHub

# conftest gotcha: stubbed sys.modules entries must be removed again, or they
# poison later test files in the same pytest run.
if _psycopg2_stubbed:
    sys.modules.pop("psycopg2", None)
    sys.modules.pop("psycopg2.extensions", None)


def _drain(queue: asyncio.Queue) -> list:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _event(payload: dict) -> str:
    return json.dumps(payload)


def test_owned_event_reaches_only_owner():
    async def scenario():
        hub = BroadcastHub(asyncio.get_running_loop())
        q_a = await hub.subscribe("user-a")
        q_b = await hub.subscribe("user-b")

        hub.publish(_event({"type": "documents_snapshot", "owner_id": "user-a"}))

        assert len(_drain(q_a)) == 1
        assert _drain(q_b) == []

    asyncio.run(scenario())


def test_ownerless_event_reaches_everyone():
    async def scenario():
        hub = BroadcastHub(asyncio.get_running_loop())
        q_a = await hub.subscribe("user-a")
        q_b = await hub.subscribe("user-b")

        hub.publish(_event({"type": "resync"}))

        assert len(_drain(q_a)) == 1
        assert len(_drain(q_b)) == 1

    asyncio.run(scenario())


def test_none_owner_queue_receives_owned_events():
    async def scenario():
        hub = BroadcastHub(asyncio.get_running_loop())
        q_all = await hub.subscribe(None)

        hub.publish(_event({"type": "documents_snapshot", "owner_id": "user-a"}))

        assert len(_drain(q_all)) == 1

    asyncio.run(scenario())


def test_non_json_message_goes_to_everyone():
    async def scenario():
        hub = BroadcastHub(asyncio.get_running_loop())
        q_a = await hub.subscribe("user-a")
        q_b = await hub.subscribe("user-b")

        hub.publish("not json")

        assert _drain(q_a) == ["not json"]
        assert _drain(q_b) == ["not json"]

    asyncio.run(scenario())


def test_unsubscribed_queue_gets_nothing():
    async def scenario():
        hub = BroadcastHub(asyncio.get_running_loop())
        q_a = await hub.subscribe("user-a")
        await hub.unsubscribe(q_a)

        hub.publish(_event({"type": "documents_snapshot", "owner_id": "user-a"}))

        assert _drain(q_a) == []

    asyncio.run(scenario())


def test_full_queue_drops_oldest_not_newest():
    async def scenario():
        hub = BroadcastHub(asyncio.get_running_loop())
        q_a = await hub.subscribe("user-a")

        for i in range(11):  # queue maxsize is 10
            hub.publish(_event({"owner_id": "user-a", "seq": i}))

        items = [json.loads(m)["seq"] for m in _drain(q_a)]
        assert len(items) == 10
        assert items[-1] == 10  # newest survived
        assert 0 not in items  # oldest was evicted

    asyncio.run(scenario())
