"""Test for the `memory get` projection flags (--section/--field/--grep/--versions).

Background: `GET /memory/cases/{id}` returns ~383 KB for a grown case, of which
only ~66 KB is unique. `_combined_payload` emits the brief twice (`brief` and
`case_brief` are byte-identical) and `memory_row_to_dict` carries `search_text`
alongside a `rendered` string with the same content. Before these flags the CLI
had only `--case-id`, so every lookup — even "what version is this?" — paid the
full payload, and the answer had to be grepped out of a dump.

Verifies the pure projection helpers:

  * ``_memory_slim`` drops ``search_text`` only when it duplicates ``rendered``.
  * ``_memory_dig`` walks dicts and lists and raises ApiError with the available
    keys on a bad segment, instead of returning None and hiding the typo.
  * ``_memory_entries`` flattens the content into (section, field, text) rows,
    unwraps the ``beteiligte`` dicts, drops empties and honours the filter.

Pure logic only -- no DB, no containers, no network. Run from the repo root:

    python3 tests/test_memory_get_projection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rechtmaschine_cli as cli  # noqa: E402


def _envelope():
    """Mirrors the real payload shape, including both redundancies."""
    rendered = "Verfahrensstand: SG Duesseldorf S 28 AY 45/26 ER"
    brief = {
        "id": "brief-1",
        "content_json": {
            "beteiligte": [{"name": "Mandant: Obida Alothman"}],
            "verfahrensstand": [
                "Nachweis des Aufenthaltsortes seit Antragseingang gefordert",
                "Kreis Viersen: seit dem 15.06.2026 als 'unbekannt verzogen' gemeldet",
            ],
            "offene_fragen": [],
            "notizen": "",
        },
        "search_text": rendered,
        "rendered": rendered,
        "version": 70,
        "updated_at": "2026-08-18T01:42:13",
        "last_reflected_at": None,
    }
    strategy = {
        "id": "strategy-1",
        "content_json": {"kernstrategie": "Annahme des Unterbringungsangebots"},
        "search_text": "abweichender Suchtext",
        "rendered": "Kernstrategie: Annahme des Unterbringungsangebots",
        "version": 30,
    }
    return {"brief": brief, "case_brief": brief, "case_strategy": strategy}


def test_slim_drops_only_the_duplicate_search_text():
    env = _envelope()

    slimmed = cli._memory_slim(env["case_brief"])
    assert "search_text" not in slimmed, "search_text duplicates rendered and must be dropped"
    assert slimmed["rendered"] == env["case_brief"]["rendered"], "rendered must survive"
    assert slimmed["version"] == 70, "unrelated fields must survive"

    kept = cli._memory_slim(env["case_strategy"])
    assert kept["search_text"] == "abweichender Suchtext", (
        "search_text that differs from rendered carries information and must be kept"
    )

    assert cli._memory_slim("not-a-dict") == "not-a-dict"


def test_dig_walks_dicts_and_lists():
    env = _envelope()

    assert cli._memory_dig(env, "case_brief.version") == 70
    assert cli._memory_dig(env, "case_brief.content_json.verfahrensstand.1").startswith("Kreis Viersen")
    assert cli._memory_dig(env, "") is env, "empty path returns the whole payload"

    for bad, expected in (
        ("case_brief.nope", "not found"),
        ("case_brief.content_json.verfahrensstand.99", "not a valid index"),
        ("case_brief.version.deeper", "cannot descend"),
    ):
        try:
            cli._memory_dig(env, bad)
        except cli.ApiError as exc:
            assert expected in str(exc), f"{bad!r} -> unhelpful message: {exc}"
        else:
            raise AssertionError(f"{bad!r} should have raised ApiError")

    # The message must name the available keys, so a typo is self-correcting.
    try:
        cli._memory_dig(env, "case_brief.nope")
    except cli.ApiError as exc:
        assert "content_json" in str(exc) and "version" in str(exc)


def test_entries_flatten_filter_and_unwrap():
    env = _envelope()

    rows = cli._memory_entries(env, None)
    texts = [r["text"] for r in rows]

    assert "Mandant: Obida Alothman" in texts, "beteiligte dicts must be unwrapped to their name"
    assert any("unbekannt verzogen" in t for t in texts), "list items must be flattened individually"
    assert "Annahme des Unterbringungsangebots" in texts, "scalar strategy fields must be included"
    assert all(t.strip() for t in texts), "empty entries must be dropped"
    assert not any(r["field"] == "offene_fragen" for r in rows), "empty list contributes no rows"

    by_section = {r["section"] for r in rows}
    assert by_section == {"brief", "strategy"}

    only_brief = cli._memory_entries(env, "brief")
    assert {r["section"] for r in only_brief} == {"brief"}
    assert len(only_brief) < len(rows)

    hit = next(r for r in only_brief if "unbekannt verzogen" in r["text"])
    assert hit["field"] == "verfahrensstand", "entries must carry the field they came from"


def test_entries_survive_a_malformed_content_json():
    """A target whose content_json is not a dict must not crash the projection."""
    broken = {"case_brief": {"content_json": ["unexpected"]}, "case_strategy": {}}
    assert cli._memory_entries(broken, None) == []


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
