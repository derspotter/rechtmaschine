"""Regression: Qwen liefert bei feldweiser Konsolidierung leere Strings stat
leerer Listen fuer Listenfelder des Extraktionsschemas (beobachtet 24.08.2026,
152/26: {'risiken_und_gegenargumente': ''} -> pydantic list_type error, und da
temperature=0.0 deterministisch ist, schlug jeder Retry identisch fehl und der
Consolidate-Job war hart blockiert).

CaseMemoryExtractionResult muss solche Modell-Ausgaben am Rand normalisieren:
  * ""/None  -> []
  * nicht-leerer String -> [String]  (verlustfrei)
  * echte Listen bleiben unveraendert

Laeuft IM Container (alle Deps vorhanden), tests/ ist nicht gemountet:

    cd /var/opt/docker/rechtmaschine
    docker compose exec -T app python - < tests/test_memory_extraction_coercion.py
"""

import sys

sys.path.insert(0, "/app")

from endpoints.agent_memory import CaseMemoryExtractionResult  # noqa: E402

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


# Der Original-Fehlerfall vom 24.08.2026 (Feld-Konsolidierung fall_notizen).
r = CaseMemoryExtractionResult(
    beteiligte=[],
    verfahrensstand=[],
    fall_notizen="Das Eilverfahren wurde verwiesen.",
    risiken_und_gegenargumente="",
)
check("leerer String wird zu leerer Liste", r.risiken_und_gegenargumente == [])
check("fall_notizen (str-Feld) bleibt String", r.fall_notizen.startswith("Das Eilverfahren"))

r2 = CaseMemoryExtractionResult(risiken="nur ein Risiko als String")
check("nicht-leerer String wird verlustfrei gewrappt", r2.risiken == ["nur ein Risiko als String"])

r3 = CaseMemoryExtractionResult(offene_fragen_fall=None)
check("None wird zu leerer Liste", r3.offene_fragen_fall == [])

r4 = CaseMemoryExtractionResult(beweismittel=["a", "b"])
check("echte Liste bleibt unveraendert", r4.beweismittel == ["a", "b"])

r5 = CaseMemoryExtractionResult(warnings="   ")
check("Whitespace-String wird zu leerer Liste", r5.warnings == [])

if failures:
    print(f"\n{len(failures)} FAILURES")
    sys.exit(1)
print("\nall passed")
