"""Draft-context assembly: /generate-parity blocks for direct drafting.
Pure functions only — builders are injected, no DB/network."""
from draft_context import assemble_draft_context, verify_facts_with_sources


def test_assemble_returns_all_four_keys():
    out = assemble_draft_context(
        "gewöhnlicher Aufenthalt",
        rag_block_fn=lambda: "RAGBLOCK\n",
        statute_block_fn=lambda: "STATUTES\n",
        memory_block_fn=lambda: "MEMORY\n",
        style_rules="RULES",
    )
    assert out == {
        "rag_block": "RAGBLOCK\n",
        "statute_block": "STATUTES\n",
        "case_memory_block": "MEMORY\n",
        "style_rules": "RULES",
    }


def test_assemble_degrades_failing_block_to_empty_string():
    def boom():
        raise RuntimeError("rag store down")
    out = assemble_draft_context(
        "q", rag_block_fn=boom, statute_block_fn=lambda: None,
        memory_block_fn=lambda: "", style_rules="RULES",
    )
    assert out["rag_block"] == ""
    assert out["statute_block"] == ""      # None normalizes to ""
    assert out["case_memory_block"] == ""
    assert out["style_rules"] == "RULES"   # style rules are static, never degraded


def test_verify_facts_flags_unsourced_date_and_az():
    result = verify_facts_with_sources(
        "Mit Bescheid vom 03.02.2026 (Az. 12 K 345/26) wurde abgelehnt.",
        memory_text="",
        sources=("Der Antrag stammt aus dem Jahr 2020.",),
    )
    high = {c["type"] for c in result["fact_checks"] if c["severity"] == "high"}
    assert "date" in high
    assert "aktenzeichen" in high


def test_verify_facts_empty_corpus_returns_no_checks():
    # Deliberate contract (same as /generate): no memory + no sources means
    # there is nothing to check against, so no checks — never a false-alarm
    # flood on every date/Aktenzeichen in the draft.
    result = verify_facts_with_sources(
        "Mit Bescheid vom 03.02.2026 (Az. 12 K 345/26) wurde abgelehnt.",
        memory_text="", sources=(),
    )
    assert result["fact_checks"] == []


def test_verify_facts_passes_sourced_facts():
    result = verify_facts_with_sources(
        "Mit Bescheid vom 03.02.2026 wurde abgelehnt.",
        memory_text="",
        sources=("Der Bescheid datiert vom 03.02.2026.",),
    )
    assert [c for c in result["fact_checks"] if c["severity"] == "high"] == []
