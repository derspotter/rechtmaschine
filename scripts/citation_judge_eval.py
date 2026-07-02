#!/usr/bin/env python3
"""Regression eval for the Qwen citation judge (citation_qwen.judge_citation_page_with_qwen).

Run inside the app container (needs /app modules + ANONYMIZATION_SERVICE_URL):

    docker cp scripts/citation_judge_eval.py rechtmaschine-app:/tmp/ && \
    docker exec -w /app rechtmaschine-app python3 /tmp/citation_judge_eval.py

Ground truth: every case below was manually verified against the source pages in the
Claude session of 2026-06-29..07-02 (case 048/26). Claims are drawn from ANONYMIZED
document text (no names / DOB / addresses); documents are referenced by UUID only.
Do not push this file to any public remote.

The judge's contract under test ("is the citation supported"):
  yes  - the cited page contains the content the sentence refers to, whether quoted,
         paraphrased, negation-framed rebuttal, or a sharpened/tacit-premise reading.
  no   - the cited page contains no corresponding content (incl. invented attributions
         and content that only exists on OTHER pages).

Exit code 0 = all cases pass, 1 = regressions (or service unavailable).
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, "/app")
import citation_verifier as cv  # noqa: E402
from citation_qwen import judge_citation_page_with_qwen  # noqa: E402

BESCHEID_ID = "48f0eb2e-1d26-4928-80b2-9536ccc86f0e"   # 048/26 anlage_k2_bescheid.pdf
ANHOERUNG_ID = "09dc78e7-eefe-49c7-a316-3e5536e68e5a"  # 048/26 Anhörung.pdf


def _entry(uid: str, filename: str, role: str) -> dict:
    return {
        "id": uid,
        "filename": filename,
        "role": role,
        "extracted_text_path": f"/app/ocr_text/{uid}.txt",
        "is_anonymized": True,
        "anonymization_metadata": {"anonymized_text_path": f"/app/anonymized_text/{uid}.txt"},
    }


COLLECTED = {
    "bescheid": [_entry(BESCHEID_ID, "20260331_200535_anlage_k2_bescheid.pdf", "primary")],
    "anhoerung": [_entry(ANHOERUNG_ID, "20260331_200529_Anhörung.pdf", "")],
}

# (label, doc_key, cited_pages, claim, sentence, expected_verdict[, "xfail"])
# doc pages: Bescheid keyed by BAMF printed "Seite: N"; Anhörung by physical index.
# "xfail" = known model limitation, tracked but not counted as a regression.
CASES = [
    # --- verbatim / close paraphrase (expect yes) ---
    ("verbatim: KAG-Verbot Art. 300 (B S.5)", "bescheid", [5],
     "Die Kirche des Allmächtigen Gottes ist in China als häretische Lehre eingestuft und verboten, "
     "die Mitgliedschaft ist nach Art. 300 des chinesischen Strafgesetzbuches strafbar, und es kommt "
     "zu Festnahmen, Verurteilungen zu Freiheitsstrafen und Todesfällen infolge von Misshandlungen.",
     "", "yes"),
    ("close para: Verfolgungsmaßnahmen Provinzen (B S.5f)", "bescheid", [5, 6],
     "Es gibt dokumentierte Festnahmen und Verfolgungsmaßnahmen gegen KAG-Mitglieder in zahlreichen "
     "Provinzen Chinas.",
     "", "yes"),
    ("verbatim-ish: Freilassung gegen Zahlung (A S.10)", "anhoerung", [10],
     "Die Klägerin wurde nach Zahlung von 50.000 Yuan aus dem Gewahrsam entlassen.",
     "", "yes"),
    # --- heavy paraphrase (expect yes; the deterministic layer failed these) ---
    ("heavy para: Misshandlungen im Gewahrsam (A S.8f)", "anhoerung", [8, 9],
     "Sie schilderte, wie sie an den Haaren gepackt, mehrfach ins Gesicht geschlagen, mit grellem "
     "Licht und Lärm gequält, am Schlafen gehindert und getreten wurde.",
     "", "yes"),
    ("heavy para: geplante Frankreich-Reise (A S.18)", "anhoerung", [18],
     "Sie erläuterte, sie habe für sich und ihren Sohn eine touristische Reise nach Frankreich geplant.",
     "", "yes"),
    # --- rebuttal / negation framing (expect yes; old prompt failed this) ---
    ("rebuttal: Annahme fehlender Glaubensausübung (B S.9)", "bescheid", [9],
     "Unzutreffend ist auch die Annahme, die Klägerin habe nach der Festnahme ihren Glauben nicht "
     "mehr ausgeübt und in Deutschland keinen ernsthaften Bezug zur KAG gezeigt.",
     "Unzutreffend ist auch die Annahme, die Klägerin habe nach der Festnahme ihren Glauben nicht "
     "mehr ausgeübt und in Deutschland keinen ernsthaften Bezug zur KAG gezeigt, Anlage K2, S. 9.",
     "yes"),
    # --- tacit-premise explication (expect yes) ---
    # History: instruction-only and few-shot prompts failed this pattern (flip-flopped,
    # settled on "no" at temp 0). Fixed 2026-07-02 by the two-step referent prompt
    # (SCHRITT 1: Bezug bestimmen). If this regresses, the two-step structure broke.
    ("tacit premise: Biometrie/Ausreise (B S.4f)", "bescheid", [4, 5],
     "Die Beklagte setzt dem lediglich die Annahme entgegen, die Erfassung biometrischer Daten "
     "müsse zwangsläufig dazu führen, dass jede spätere Ausreise unmöglich wäre.",
     "Die Beklagte setzt dem lediglich die Annahme entgegen, die Erfassung biometrischer Daten müsse "
     "zwangsläufig dazu führen, dass jede spätere Ausreise unmöglich wäre, Anlage K2, S. 4 f.",
     "yes"),
    # invented psychiatric-report attribution: guards the two-step prompt against
    # over-accepting merely topic-adjacent attributions (added with the referent prompt)
    ("invented: psychiatrisches Gutachten (B S.4)", "bescheid", [4],
     "Die Beklagte stützt die Ablehnung auf ein psychiatrisches Gutachten über die Klägerin.",
     "", "no"),
    # --- invented attribution (expect no) ---
    ("invented attribution: Todesstrafe eingeräumt (B S.4)", "bescheid", [4],
     "Die Beklagte räumt ein, dass der Klägerin bei einer Rückkehr mit Sicherheit die Todesstrafe droht.",
     "", "no"),
    ("invented: KP-Mitgliedschaft (A S.7)", "anhoerung", [7],
     "Die Klägerin gab an, mehrere Jahre Mitglied der Kommunistischen Partei gewesen zu sein.",
     "", "no"),
    # --- content absent from cited page (expect no) ---
    ("absent: deutsche Staatsangehörigkeit (B S.9)", "bescheid", [9],
     "Die Klägerin besitzt die deutsche Staatsangehörigkeit und hat nie einen Asylantrag gestellt.",
     "", "no"),
    ("wrong page: KAG-Verbot auf S.4 zitiert (B S.4)", "bescheid", [4],
     "Die Kirche des Allmächtigen Gottes ist als häretische Lehre eingestuft und ihre Mitgliedschaft "
     "nach Art. 300 des chinesischen Strafgesetzbuches strafbar.",
     "", "no"),
    # --- contradicted attribution (expect no) ---
    ("contradicted: nie festgenommen (A S.7)", "anhoerung", [7],
     "Die Klägerin gab in der Anhörung an, zu keinem Zeitpunkt festgenommen worden zu sein.",
     "", "no"),
]


def regex_battery_check() -> int:
    """Regression guard for the anonymization safety-catch regexes (no LLM calls).

    The built-in catches in anon/anonymization_service.py must not redact ordinary
    legal boilerplate (citations, counter nouns, page markers) but must still catch
    real addresses in address-label context. Audited 2026-07-02 after the street
    catch ate "--- Seite 2 ---", "Urteil des EGMR vom 23...", "Kammer 41" etc.
    Returns the number of failures.
    """
    from anon.anonymization_service import apply_regex_replacements

    boiler = (
        "--- Seite 5 ---\n"
        "Seite: 4\n"
        "Nach § 3 Abs. 1 AsylG i.V.m. Art. 16a GG und der Richtlinie 2011/95/EU gilt:\n"
        "BVerwG, Urteil vom 20.02.2013, 10 C 23.12; EuGH, Urteil vom 05.09.2012, C-71/11.\n"
        "Vgl. auch Urteil des EGMR vom 23.03.2016, Nr. 43611/11.\n"
        "Anschrift: wird nachgereicht\n"
        "Nach der Rechtsprechung des Bundesverwaltungsgerichts vom 10.05.1994, 9 C 434.93,\n"
        "ist die Kammer 41 zuständig. Der Streitwert beträgt 5.000 Euro.\n"
        "Wohnhaft: unbekannt verzogen\n"
        "Der Lagebericht 2023 des Auswärtigen Amtes beschreibt die Lage in Kapitel 4.\n"
        "Randnummer 12 der Entscheidung verweist auf Anlage 2 und Artikel 18 Absatz 1.\n"
    )
    addresses = (
        "Wohnanschrift:\nObertorweg 1\n41460 Neuss\n"
        "Anschrift: Erkrather Straße 349, 40231 Düsseldorf\n"
        "wohnhaft: Unter den Linden 5\n"
        "Zustellanschrift: Friedrich-Ebert-Straße 17, 40210 Düsseldorf\n"
    )
    failures = 0
    out = apply_regex_replacements(boiler, {})
    for a, b in zip(boiler.splitlines(), out.splitlines()):
        if a != b:
            failures += 1
            print(f"regex-battery FALSE POSITIVE: {a[:70]!r} -> {b[:70]!r}")
    out2 = apply_regex_replacements(addresses, {})
    for a, b in zip(addresses.splitlines(), out2.splitlines()):
        value = any(s in a for s in ("Obertorweg", "Erkrather", "Linden", "Friedrich", "41460"))
        if value and a == b:
            failures += 1
            print(f"regex-battery MISSED ADDRESS: {a!r}")
    print(f"regex battery: {'OK' if failures == 0 else f'{failures} FAILURES'}")
    return failures


async def main() -> int:
    service_url = os.environ.get("ANONYMIZATION_SERVICE_URL")
    if not service_url:
        print("FAIL: ANONYMIZATION_SERVICE_URL not set (run inside the app container)")
        return 1

    regex_failures = regex_battery_check()

    docs = {}
    for key, doc in zip(("bescheid", "anhoerung"), cv._load_document_texts(COLLECTED)):
        docs[key] = doc
    for key, doc in docs.items():
        if not doc.pages:
            print(f"FAIL: no page texts loaded for {key}")
            return 1

    failures = 0
    xfails = 0
    xpasses = 0
    print(f"{'case':52} {'expect':7} {'got':8} {'ok':5} time")
    for case in CASES:
        label, doc_key, pages, claim, sentence, expected = case[:6]
        known_fail = len(case) > 6 and case[6] == "xfail"
        page_text = "\n".join(docs[doc_key].pages.get(p, "") for p in pages)
        if not page_text.strip():
            print(f"{label:52} {expected:7} {'-':8} ??    cited page text missing")
            failures += 1
            continue
        sample = {
            "claim": claim,
            "sentence": sentence or claim,
            "citation": f"S. {', '.join(str(p) for p in pages)}",
            "page_text": page_text,
        }
        start = time.time()
        judgment = await judge_citation_page_with_qwen(service_url, sample)
        verdict = judgment.get("verdict", "?")
        ok = verdict == expected
        if ok and known_fail:
            mark = "XPASS"
            xpasses += 1
        elif ok:
            mark = "OK"
        elif known_fail:
            mark = "XFAIL"
            xfails += 1
        else:
            mark = "XX"
            failures += 1
        print(f"{label:52} {expected:7} {verdict:8} {mark:5} {time.time()-start:4.0f}s")
        if not ok and not known_fail:
            print(f"    reason: {judgment.get('reason', '')[:160]}")

    total = len(CASES)
    print(f"\n{total - failures - xfails}/{total} passed"
          f"{f', {xfails} known-limit (xfail)' if xfails else ''}"
          f"{f', {xpasses} XPASS (known limit now passing!)' if xpasses else ''}"
          f"{', regex battery FAILED' if regex_failures else ', regex battery OK'}")
    return 0 if failures == 0 and regex_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
