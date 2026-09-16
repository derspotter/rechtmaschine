"""Konsolidierung nur bei leerer Proposal-Queue (16.09.2026).

_maybe_enqueue_consolidation legt keinen consolidate-Job an, solange die
Akte pending Proposals hat; _consolidate_if_queue_empty holt das nach
accept/reject nach, sobald das letzte Proposal entschieden ist.

Läuft IM Container (tests/ ist nicht gemountet):

    cd /var/opt/docker/rechtmaschine
    docker compose exec -T app python - < tests/test_memory_consolidate_queue_gate.py
"""

import sys
from types import SimpleNamespace

sys.path.insert(0, "/app")

import endpoints.agent_memory as am  # noqa: E402

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


class _DB:
    def __init__(self, pending):
        self.pending = pending

    def refresh(self, obj):
        pass

    def query(self, *a):
        db = self

        class _Q:
            def filter(self, *a, **k):
                return self

            def count(self):
                return db.pending

            def all(self):
                return []

        return _Q()


BIG_BRIEF = SimpleNamespace(content_json={"verfahrensstand": [f"e{i}" for i in range(am.MEMORY_CONSOLIDATE_THRESHOLD + 1)]})
SMALL_BRIEF = SimpleNamespace(content_json={"verfahrensstand": ["a"]})

enqueued = []
import agent_memory_service as svc  # noqa: E402

svc.enqueue_memory_reflection = lambda db, owner, case, trigger=None, **k: enqueued.append(trigger)
am.MEMORY_CONSOLIDATE_COOLDOWN_HOURS = 6

am._maybe_enqueue_consolidation(_DB(pending=3), "o", "c", BIG_BRIEF)
check("pending proposals block the auto-consolidation", enqueued == [])

am._maybe_enqueue_consolidation(_DB(pending=0), "o", "c", BIG_BRIEF)
check("empty queue above threshold enqueues consolidate", enqueued == ["consolidate"])

enqueued.clear()
am._maybe_enqueue_consolidation(_DB(pending=0), "o", "c", SMALL_BRIEF)
check("below threshold nothing is enqueued", enqueued == [])

enqueued.clear()
svc.get_or_create_case_brief = lambda db, owner, case, for_update=False: BIG_BRIEF
am._consolidate_if_queue_empty(_DB(pending=1), "o", "c")
check("after review with proposals left: no consolidation", enqueued == [])
am._consolidate_if_queue_empty(_DB(pending=0), "o", "c")
check("after review with empty queue: consolidation enqueued", enqueued == ["consolidate"])

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
