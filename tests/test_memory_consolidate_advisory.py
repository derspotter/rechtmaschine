"""Advisory-Guard fuer den Consolidate-Trigger (Design-Entscheid Jay 24.08.2026).

Konsolidierungs-Proposals gehen ohnehin durch die Annahme des Anwalts. Deshalb
blockiert die Fakten-Erhalt-Guard dort nicht mehr (kein Verwerfen, keine
Feedback-Runden), sondern meldet Verluste sichtbar: die konsolidierte Fassung
kommt IMMER zurueck, verlorene Pflicht-Tokens stehen als "Entfallende
Angaben"-Warning am Extraktionsergebnis. Annahme = genehmigte Kompression,
Ablehnung kostet nichts. (Der unbeaufsichtigte j-lawyer-Fold behaelt seine
harte Guard — er ist hiervon nicht beruehrt.)

Erwartetes Verhalten von endpoints.agent_memory._consolidate_target
(modul-level, damit das Modell stubbar ist):
  1. Verlust -> Ergebnis kommt zurueck, warnings enthalten "Entfallende
     Angaben" mit den fehlenden Tokens.
  2. Kein Verlust -> keine Advisory-Warning.
  3. Genau EIN Modellaufruf (temp 0: Retries/Feedback-Runden sind sinnlos
     und entfallen ersatzlos).

Laeuft IM Container (tests/ ist nicht gemountet):

    cd /var/opt/docker/rechtmaschine
    docker compose exec -T app python - < tests/test_memory_consolidate_advisory.py
"""

import asyncio
import sys

sys.path.insert(0, "/app")

import endpoints.agent_memory as am  # noqa: E402
from endpoints.agent_memory import CaseMemoryExtractionResult  # noqa: E402

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


DISPLAY = {
    "verfahrensstand": [
        "BAMF-Bescheid 01.04.2026 (Az. 10979509-477): Ablehnung als unzulässig",
        "Eilantrag § 123 VwGO 31.07.2026 beim VG Köln (Az. 19 L 1915/26.A)",
    ],
    "offene_fragen": ["Frist Zusicherung 05.08.2026 abgelaufen?"],
}


def run(fake_result):
    calls = []

    async def fake_model(prompt_core, num_predict=2600, images=None):
        calls.append(prompt_core)
        return fake_result

    orig = am._run_memory_model
    am._run_memory_model = fake_model
    try:
        result = asyncio.run(
            am._consolidate_target("FALLBRIEF", "verfahrensstand, offene_fragen", DISPLAY, prune=True)
        )
    finally:
        am._run_memory_model = orig
    return result, calls


# 1. Verlustfall: die konsolidierte Fassung laesst 01.04.2026 und die offene
#    Frage (05.08.2026) fallen -> kommt trotzdem zurueck, Verluste gemeldet.
lossy = CaseMemoryExtractionResult(
    verfahrensstand=[
        "BAMF-Verfahren (Az. 10979509-477): Ablehnung, Eilantrag 31.07.2026 VG Köln (Az. 19 L 1915/26.A)"
    ],
)
res, calls = run(lossy)
check("Verlust-Fassung wird nicht verworfen", res is not None)
adv = [w for w in (res.warnings or []) if "Entfallende Angaben" in w]
check("Advisory-Warning vorhanden", len(adv) == 1)
check("verlorenes Datum 01.04.2026 benannt", adv and "01.04.2026" in adv[0])
check("verlorenes Datum 05.08.2026 benannt", adv and "05.08.2026" in adv[0])
check("genau ein Modellaufruf (kein Retry)", len(calls) == 1)

# 2. Verlustfreier Fall: keine Advisory-Warning.
lossless = CaseMemoryExtractionResult(
    verfahrensstand=[
        "BAMF-Bescheid 01.04.2026 (Az. 10979509-477): Ablehnung als unzulässig; "
        "Eilantrag § 123 VwGO 31.07.2026 beim VG Köln (Az. 19 L 1915/26.A)"
    ],
    offene_fragen_fall=["Frist Zusicherung 05.08.2026 abgelaufen?"],
)
res2, calls2 = run(lossless)
check("verlustfrei: keine Advisory-Warning", not [w for w in (res2.warnings or []) if "Entfallende" in w])
check("verlustfrei: genau ein Modellaufruf", len(calls2) == 1)

if failures:
    print(f"\n{len(failures)} FAILURES")
    sys.exit(1)
print("\nall passed")
