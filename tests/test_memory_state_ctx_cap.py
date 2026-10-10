"""Regression: der Kontextblock "BEREITS GESPEICHERT" der j-lawyer-Extraktion
war ungedeckelt. ee97d275 (taeglich von Sessions bearbeitet, daher nie auto-
konsolidiert) hatte am 10.10.2026 145k Zeichen Fall-Speicher, der Prompt kam auf
69k Tokens, llama-server antwortete 400 exceed_context_size — und der 400 wurde
dreimal identisch wiederholt.

Prueft _bounded_state_json (Limit eingehalten, aelteste zuerst, beteiligte
unangetastet) und dass ein 4xx nicht wiederholt wird.

Laeuft IM Container (alle Deps vorhanden), tests/ ist nicht gemountet:

    cd /var/opt/docker/rechtmaschine
    docker compose exec -T app python - < tests/test_memory_state_ctx_cap.py
"""

import asyncio
import json
import sys

sys.path.insert(0, "/app")

import httpx  # noqa: E402

import endpoints.agent_memory as am  # noqa: E402
from endpoints.agent_memory import (  # noqa: E402
    CaseMemoryExtractionResult,
    _bounded_state_json,
)

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


# --- Kappung ---------------------------------------------------------------
known = CaseMemoryExtractionResult(
    beteiligte=[f"Person {i}" for i in range(16)],
    verfahrensstand=[f"{i:03d} Verfahrensschritt " + "x" * 200 for i in range(109)],
    sachverhalt=[f"{i:03d} Sachverhalt " + "y" * 800 for i in range(40)],
    risiken=["kurzes Risiko"],
    kernstrategie="Kernstrategie bleibt",
)
full = json.dumps(known.model_dump(), ensure_ascii=False)

text, dropped = _bounded_state_json(known, len(full) + 10)
check(dropped == 0 and text == full, "unter dem Limit darf nichts gekuerzt werden")

limit = 20000
text, dropped = _bounded_state_json(known, limit)
data = json.loads(text)
check(len(text) <= limit, f"Limit verletzt: {len(text)} > {limit}")
check(dropped > 0, "ueber dem Limit muss gekuerzt werden")
check(data["beteiligte"] == known.beteiligte, "beteiligte darf nie gekuerzt werden")
check(data["kernstrategie"] == "Kernstrategie bleibt", "Skalare bleiben erhalten")
check(data["risiken"] == ["kurzes Risiko"], "kleine Listen bleiben erhalten")
check(
    data["sachverhalt"] and data["sachverhalt"][-1].startswith("039"),
    "der neueste Eintrag muss bleiben",
)
check(
    not data["sachverhalt"] or not data["sachverhalt"][0].startswith("000"),
    "die aeltesten Eintraege fallen zuerst",
)
kept = len(data["verfahrensstand"]) + len(data["sachverhalt"])
check(kept + dropped == 149, f"Zaehlung stimmt nicht: {kept} + {dropped} != 149")

text, dropped = _bounded_state_json(known, 0)
check(dropped == 0, "Limit 0 heisst: Kappung aus")


# --- kein Retry bei 4xx ----------------------------------------------------
calls = {"n": 0}


async def fake_call(*args, **kwargs):
    calls["n"] += 1
    req = httpx.Request("POST", "http://desktop:8004/qwen-json")
    resp = httpx.Response(400, request=req, text='{"error":{"type":"exceed_context_size_error"}}')
    raise httpx.HTTPStatusError("400", request=req, response=resp)


async def fake_ready():
    return None


import citation_qwen  # noqa: E402
import shared  # noqa: E402

citation_qwen.call_qwen_json = fake_call
shared.ensure_anonymization_service_ready = fake_ready
am.os.environ.setdefault("ANONYMIZATION_SERVICE_URL", "http://desktop:8004")


async def run():
    try:
        await am._run_memory_model_qwen("prompt")
    except Exception as exc:  # HTTPException 502 erwartet
        return exc
    return None


exc = asyncio.run(run())
check(calls["n"] == 1, f"400 wurde {calls['n']}x gesendet, erwartet 1x")
check(exc is not None and "exceed_context_size" in str(getattr(exc, "detail", "")),
      "Fehlerdetail muss den Grund aus dem 400 enthalten")

if failures:
    print("FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("OK test_memory_state_ctx_cap")
