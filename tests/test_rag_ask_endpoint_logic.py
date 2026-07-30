"""Ask-Endpoint-Logik: Quellen-Sammlung (fail-closed) und Stream-Envelope.
retrieve_fn wird injiziert — kein Netzwerk."""
import json

from endpoints.rag import build_stream_envelope, collect_ask_sources


def test_collect_ask_sources_queries_each_collection_and_numbers():
    calls = []

    def fake_retrieve(query, owner_id, limit, use_reranker, collection):
        calls.append(collection)
        return [{"text": f"T-{collection}", "context_header": "[H]", "score": 0.5, "chunk_id": collection}]

    sources = collect_ask_sources("frage", ["kanzlei", "doktrin"], "u1", fake_retrieve)
    assert calls == ["kanzlei", "doktrin"]
    assert [s["label"] for s in sources] == ["Q1", "Q2"]
    assert sources[1]["collection"] == "doktrin"


def test_collect_ask_sources_empty_collections_means_all_three():
    seen = []

    def fake_retrieve(query, owner_id, limit, use_reranker, collection):
        seen.append(collection)
        return []

    sources = collect_ask_sources("frage", [], "u1", fake_retrieve)
    assert seen == ["kanzlei", "jurisprudence", "doktrin"]
    assert sources == []  # fail-closed entscheidet der Endpoint anhand der leeren Liste


def test_build_stream_envelope_format():
    envelope = build_stream_envelope("abc-123", [{"label": "Q1"}])
    meta_line, rest = envelope.split("\n", 1)
    assert json.loads(meta_line) == {"chat_id": "abc-123", "sources": [{"label": "Q1"}]}
    assert rest == "<<<ANSWER>>>\n"
