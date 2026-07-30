"""RAG-Ask Kernlogik: Nummerierung, Belegpflicht, Prompt-Assembly.
Pure Funktionen, kein Netzwerk/DB."""
from rag_ask import (
    build_ask_prompt,
    build_chat_messages,
    has_citation_markers,
    number_chunks,
    trim_history,
)


def _chunk(text="Argument X.", header="[Klage | 24/014 | VG Düsseldorf]", score=0.9, cid="c1"):
    return {"text": text, "context_header": header, "score": score, "chunk_id": cid}


def test_number_chunks_labels_across_collections_in_order():
    numbered = number_chunks({
        "kanzlei": [_chunk(cid="a"), _chunk(cid="b")],
        "doktrin": [_chunk(cid="c")],
    })
    assert [s["label"] for s in numbered] == ["Q1", "Q2", "Q3"]
    assert numbered[2]["collection"] == "doktrin"
    assert numbered[0]["header"] == "[Klage | 24/014 | VG Düsseldorf]"


def test_build_ask_prompt_contains_sources_question_and_belegpflicht():
    sources = number_chunks({"kanzlei": [_chunk()]})
    system, prompt = build_ask_prompt("Wie argumentieren wir § 60 Abs. 5?", sources)
    assert "[Q1]" in prompt and "Argument X." in prompt
    assert "kanzlei" in prompt  # Herkunftszeile nennt die Collection
    assert "Wie argumentieren wir § 60 Abs. 5?" in prompt
    assert "[Qn]" in system or "[Q1]" in system  # Belegpflicht-Anweisung
    assert "nicht belegt" in system    # Kennzeichnungspflicht für Lücken
    assert "Semikolon" in system or "Semikola" in system


def test_build_ask_prompt_includes_trimmed_history():
    history = [{"role": "user", "content": f"Frage {i}"} for i in range(20)]
    _, prompt = build_ask_prompt("Folgefrage", number_chunks({"kanzlei": [_chunk()]}), history)
    assert "Frage 19" in prompt
    assert "Frage 3" not in prompt  # nur die letzten 12


def test_trim_history_drops_invalid_roles_and_caps_length():
    history = [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "böse"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "b"},
    ]
    trimmed = trim_history(history)
    assert trimmed == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_has_citation_markers():
    assert has_citation_markers("Das folgt aus [Q2].")
    assert not has_citation_markers("Behauptung ohne Beleg.")


def test_build_chat_messages_sets_unbelegt_flag_and_sources():
    sources = number_chunks({"kanzlei": [_chunk()]})
    msgs = build_chat_messages("F?", "Antwort ohne Marker", "gemini-3.6-flash", sources, "2026-07-30T12:00:00Z")
    assert msgs[0] == {"role": "user", "content": "F?", "created_at": "2026-07-30T12:00:00Z"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["unbelegt"] is True
    assert msgs[1]["model"] == "gemini-3.6-flash"
    assert msgs[1]["sources"][0]["label"] == "Q1"

    msgs_ok = build_chat_messages("F?", "Belegt [Q1].", "gemini-3.6-flash", sources, "2026-07-30T12:00:00Z")
    assert msgs_ok[1]["unbelegt"] is False
