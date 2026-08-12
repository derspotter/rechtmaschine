"""Test for the Versandstand note handed to the memory distiller.

Background (case 003/26, 30.07.2026): the nightly reflection read a finished
but unsent Klagebegründung, saw "per beA" in the letterhead — which is part of
the empty template — and recorded the Schriftsatz as filed. The case memory
then claimed a filing that never happened. ``dispatch_note`` gives the model
the one fact it cannot read off the document itself: whether the Akte holds a
PDF export or a dispatch artifact for that draft.

Pure logic only — no DB, no containers, no network. Run from the repo root:

    python3 tests/test_jlawyer_dispatch_note.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import jlawyer_reader as jlr  # noqa: E402

failures = []


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


# The real document list of case 003/26 on 04.08.2026, reduced to what matters.
DOCS = [
    {"name": "2026-07-28_Klagebegründung.odt"},
    {"name": "2026-07-28_Anforderung_qualifizierte_Bescheinigung_Herzquartier.odt"},
    {"name": "2026-07-28_Anforderung_qualifizierte_Bescheinigung_Herzquartier_gesendet.pdf"},
    {"name": "2026-02-06_2253_Gericht_m.odt"},
    {"name": "2026-03-05_1744_2026-02-06_2253_Gericht_m.pdf per E-Mail.eml"},
    {"name": "Vollmacht.pdf"},
    {"name": "2026-07-22_Email_Anlage_01_scan.pdf"},
]

print("dispatch_note")

note = jlr.dispatch_note("2026-07-28_Klagebegründung.odt", DOCS)
check("unsent draft is flagged as not proven", "Entwurf" in note)
# Regression 12.08.2026 (165/26): ein per beA versandter Schriftsatz hat seinen
# Versandbeleg als .bea in der Akte, das nach dem BETREFF heißt — der
# Stem-Match kann es nie finden. Die Note darf deshalb nicht behaupten, der
# Versand sei NICHT erfolgt; sie muss auf andere Aktendokumente (beA-Nachricht,
# Versand-Mail) als gültigen Beleg verweisen, sonst übersteuert sie die
# danebenliegende .bea-Evidenz im Distiller.
check("negative note does not assert non-dispatch", "NICHT belegt" not in note)
check("negative note names beA-Nachricht as valid proof", "beA-Nachricht" in note)

note = jlr.dispatch_note("2026-07-28_Anforderung_qualifizierte_Bescheinigung_Herzquartier.odt", DOCS)
check("_gesendet.pdf counts as dispatch proof", note.startswith("Versandbeleg"))

note = jlr.dispatch_note("2026-02-06_2253_Gericht_m.odt", DOCS)
check("archived dispatch mail counts as proof", note.startswith("Versandbeleg"))

check("PDFs get no note", jlr.dispatch_note("Vollmacht.pdf", DOCS) is None)
check("incoming scans get no note", jlr.dispatch_note("2026-07-22_Email_Anlage_01_scan.pdf", DOCS) is None)
check("empty name is tolerated", jlr.dispatch_note("", DOCS) is None)

# A same-stem PDF of an unrelated case must not leak in via containment.
OTHER = [{"name": "2026-07-28_Klagebegründung.odt"}, {"name": "Klagebegruendung_Muster.pdf"}]
note = jlr.dispatch_note("2026-07-28_Klagebegründung.odt", OTHER)
check("unrelated PDF is no proof", "Entwurf" in note and not note.startswith("Versandbeleg"))

print()
if failures:
    print(f"{len(failures)} check(s) failed.")
    raise SystemExit(1)
print("All checks passed.")
