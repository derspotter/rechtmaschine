"""Konsolidierung nur bei leerer Proposal-Queue (16.09.2026) und ohne
Session-Aktivität auf der Akte (23.09.2026).

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
    def __init__(self, pending, last_accepted=None):
        self.pending = pending
        self.last_accepted = last_accepted

    def refresh(self, obj):
        pass

    def query(self, *a):
        db = self

        class _Q:
            def filter(self, *a, **k):
                return self

            def order_by(self, *a, **k):
                return self

            def count(self):
                return db.pending

            def all(self):
                return []

            def first(self):
                return db.last_accepted

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
CONS_ACCEPTED = SimpleNamespace(source_refs=[{"source_type": "consolidation"}], ops=[], proposed_patch={})
am._maybe_enqueue_consolidation(_DB(pending=0, last_accepted=CONS_ACCEPTED), "o", "c", BIG_BRIEF)
check("last accepted proposal is a consolidation: nothing new to compress", enqueued == [])
FACT_ACCEPTED = SimpleNamespace(source_refs=[{"source_type": "jlawyer_document"}], ops=[], proposed_patch={})
am._maybe_enqueue_consolidation(_DB(pending=0, last_accepted=FACT_ACCEPTED), "o", "c", BIG_BRIEF)
check("last accepted proposal is a fact: consolidation enqueued", enqueued == ["consolidate"])

enqueued.clear()
svc.get_or_create_case_brief = lambda db, owner, case, for_update=False: BIG_BRIEF
am._consolidate_if_queue_empty(_DB(pending=1), "o", "c")
check("after review with proposals left: no consolidation", enqueued == [])
am._consolidate_if_queue_empty(_DB(pending=0), "o", "c")
check("after review with empty queue: consolidation enqueued", enqueued == ["consolidate"])

# --- Session-Aktivität (23.09.2026): Claim oder claude/codex-Proposal <24 h ---
import json, os, tempfile, time  # noqa: E402


class _SessionDB(_DB):
    """count() liefert pending=0 für die Queue, aber `session` für die
    Session-Proposal-Abfrage (zweiter count-Aufruf)."""

    def __init__(self, session, case_name="089/26 Balulov"):
        super().__init__(pending=0)
        self.session = session
        self.case = SimpleNamespace(name=case_name, file_reference=None)
        self.calls = 0

    def query(self, *a):
        db = self
        is_case = bool(a) and a[0] is am.Case

        class _Q:
            def filter(self, *a, **k):
                return self

            def order_by(self, *a, **k):
                return self

            def count(self):
                db.calls += 1
                return 0 if db.calls == 1 else db.session

            def all(self):
                return []

            def first(self):
                return db.case if is_case else None

        return _Q()


snap = os.path.join(tempfile.mkdtemp(), "active-claims.json")
am.CLAIMS_SNAPSHOT_PATH = snap


def write_claims(*keys, age=0):
    with open(snap, "w") as fh:
        json.dump({"claims": [{"key": k, "owner": "claude:x", "renewed_at": time.time() - age} for k in keys]}, fh)


enqueued.clear()
am._maybe_enqueue_consolidation(_SessionDB(session=2), "o", "c", BIG_BRIEF)
check("session proposal within quiet window blocks consolidation", enqueued == [])

write_claims("089/26")
am._maybe_enqueue_consolidation(_SessionDB(session=0), "o", "c", BIG_BRIEF)
check("claimed Akte blocks consolidation", enqueued == [])

write_claims("089/26", age=am.CLAIMS_TTL_SECONDS + 60)
am._maybe_enqueue_consolidation(_SessionDB(session=0), "o", "c", BIG_BRIEF)
check("expired claim does not block", enqueued == ["consolidate"])

enqueued.clear()
write_claims("106/26", "tooling-memory-triage")
am._maybe_enqueue_consolidation(_SessionDB(session=0), "o", "c", BIG_BRIEF)
check("claims on other keys do not block", enqueued == ["consolidate"])

enqueued.clear()
os.remove(snap)
am._maybe_enqueue_consolidation(_SessionDB(session=0), "o", "c", BIG_BRIEF)
check("missing snapshot fails open", enqueued == ["consolidate"])

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
