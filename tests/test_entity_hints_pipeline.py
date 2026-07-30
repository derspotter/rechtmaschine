"""Integrationstest: greifen die Aktendaten-Hinweise, wenn die LLM-Extraktion
NICHTS liefert?

Das ist der Fall vom 30.07.2026 (Akte 034/26): Der Extraktionsdienst antwortete
mit leeren Entitätenlisten, obwohl der Nachname des Mandanten fünfmal im Text
stand. Ohne Hinweise bleibt der Name im Klartext stehen und das Null-Gate
verwirft den Lauf. Mit Hinweisen aus j-lawyer muss der Name trotzdem ersetzt
werden.

Der Extraktionsdienst wird gestubbt (leere Antwort), die Ersetzung selbst läuft
echt durch `apply_regex_replacements_parallel`. Kein Netz, keine GPU, keine DB:

    docker cp tests/test_entity_hints_pipeline.py rechtmaschine-app:/tmp/
    docker exec rechtmaschine-app python3 /tmp/test_entity_hints_pipeline.py
"""

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ["ANONYMIZATION_SERVICE_URL"] = "http://stub-extraction-service"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from endpoints import anonymization as anon  # noqa: E402

TEXT = (
    "Besprechung mit Herrn Li am 31.03.2026.\n"
    "Herr Li schildert den Konflikt mit seinem Arbeitgeber.\n"
    "Die Polizei war nach seiner Darstellung politisch motiviert.\n"
)

HINTS = {"names": ["Tiancheng Li", "Li"], "case_numbers": ["034/26"]}

EMPTY_EXTRACTION = {
    "model": "stub-model",
    "normalized_entities": {},
    "prompt_eval_count": 1,
    "eval_count": 0,
    "total_duration": 1,
}


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _StubClient:
    """Ersetzt httpx.AsyncClient im Extraktionspfad."""

    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        type(self).calls += 1
        return _StubResponse(EMPTY_EXTRACTION)


async def _no_service_check():
    """Der Bereitschafts-Check spricht den echten Desktop-Dienst an (eigener
    HTTP-Client, vom Stub unten nicht erfasst) und blockiert den Test sonst."""
    return None


def _run(**kwargs):
    _StubClient.calls = 0
    original = anon.httpx.AsyncClient
    original_ready = anon.ensure_anonymization_service_ready
    anon.httpx.AsyncClient = _StubClient
    anon.ensure_anonymization_service_ready = _no_service_check
    try:
        return asyncio.run(
            anon.anonymize_document_text(
                TEXT,
                "Sonstige gespeicherte Quellen",
                "qwen",
                **kwargs,
            )
        )
    finally:
        anon.httpx.AsyncClient = original
        anon.ensure_anonymization_service_ready = original_ready


def test_without_hints_the_name_survives():
    # Belegt den Ausgangsschaden: ohne Hinweise bleibt "Li" im Klartext.
    result = _run()
    assert result is not None
    assert "Li" in result.anonymized_text
    assert result.plaintiff_names == []


def test_hints_replace_the_name_despite_empty_extraction():
    result = _run(entity_hints=HINTS)
    assert result is not None
    assert "Herr Li" not in result.anonymized_text
    assert "Li" in result.plaintiff_names


def test_short_name_over_redacts_inside_other_words_known_defect():
    """BEKANNTER DEFEKT (dokumentiert, nicht behoben), gefunden 30.07.2026.

    `safe_replace` ersetzt Personennamen ohne Wortgrenzen: der Nachname "Li"
    trifft auch in "Polizei" und "politisch". Für ORTSnamen ist genau das in
    anon/anonymization_service.py schon gelöst (`_wrap_alpha_boundaries`,
    Kommentar "Essen -> Interessen"), für Personen nicht. Dasselbe Muster
    erzeugt die Artefakte in bereits anonymisierten Akten ("passiert" ->
    "pass[DOKUMENT-ID]").

    Nicht einfach Wortgrenzen nachrüsten: deutsche Flexion ("Müllers",
    "Müller-Straße") würde dann nicht mehr erfasst, und das wäre eine
    Unter-Schwärzung. Die Entscheidung liegt bei Jay. Bis dahin hält dieser
    Test den Ist-Zustand fest, damit die Behebung auffällt.
    """
    result = _run(entity_hints=HINTS)
    assert "Po[PERSON]zei" in result.anonymized_text
    assert "Polizei" not in result.anonymized_text


def test_extraction_still_runs_when_hints_are_given():
    # Hinweise ERSETZEN die Extraktion nicht - anders als known_entities.
    _run(entity_hints=HINTS)
    assert _StubClient.calls > 0


def test_known_entities_skips_the_extraction_service():
    # Gegenprobe zur Abgrenzung: known_entities ruft den Dienst gar nicht auf.
    result = _run(known_entities=HINTS)
    assert result is not None
    assert _StubClient.calls == 0


def test_absent_hints_are_not_injected():
    # Ein Beteiligter der Akte, der in DIESEM Dokument nicht vorkommt, darf
    # nicht in den Entitaeten landen.
    result = _run(entity_hints={"names": ["Ergasheva"]})
    assert "Ergasheva" not in (result.plaintiff_names or [])


def main():
    tests = [
        test_without_hints_the_name_survives,
        test_hints_replace_the_name_despite_empty_extraction,
        test_short_name_over_redacts_inside_other_words_known_defect,
        test_extraction_still_runs_when_hints_are_given,
        test_known_entities_skips_the_extraction_service,
        test_absent_hints_are_not_injected,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"OK   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed")


if __name__ == "__main__":
    main()
