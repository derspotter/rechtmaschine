"""juris_embed: synthetische Embedding-Sicht für RechtsprechungEntry-Zeilen
ohne Volltext (research_verified-Write-back, manuelle Playbook-Einträge).

Pure Tests: Text-Synthese, Chunk-Payload/Metadata, deterministische
chunk_ids, Batch-Upsert mit injiziertem post_fn.
"""

import sys
import uuid
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from juris_embed import (  # noqa: E402
    entry_chunks,
    entry_embed_text,
    upsert_chunks,
)


def _entry(**overrides):
    base = dict(
        id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        country="Syrien",
        tags=["lager:pro", "syrien"],
        court="VG Bremen",
        court_level="VG",
        decision_date=date(2024, 2, 16),
        aktenzeichen="3 K 320/22",
        outcome="granted",
        key_facts=[],
        key_holdings=["Widerruf der Flüchtlingseigenschaft rechtswidrig."],
        argument_patterns=[],
        summary="Widerrufsbescheid aufgehoben.",
        source_type="research_verified",
        instance_weight=1,
        schlagworte=["Widerruf"],
        normen=["§ 73 AsylG"],
        leitsatz="Der Widerruf setzt eine erhebliche Lageänderung voraus.",
        is_active=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_embed_text_contains_core_fields():
    text = entry_embed_text(_entry())
    for needle in (
        "VG Bremen",
        "3 K 320/22",
        "16.02.2024",
        "granted",
        "Syrien",
        "Der Widerruf setzt eine erhebliche Lageänderung voraus.",
        "Widerruf der Flüchtlingseigenschaft rechtswidrig.",
        "§ 73 AsylG",
    ):
        assert needle in text


def test_embed_text_renders_dict_argument_patterns_as_prose():
    entry = _entry(
        argument_patterns=[
            {"use_when": "Widerrufsverfahren", "rebuttal": "Lageänderung nicht erheblich"}
        ]
    )
    text = entry_embed_text(entry)
    assert "Widerrufsverfahren: Lageänderung nicht erheblich" in text
    assert "{" not in text


def test_embed_text_skips_empty_fields():
    entry = _entry(leitsatz=None, key_holdings=[], summary=None, normen=[])
    text = entry_embed_text(entry)
    assert "Leitsatz" not in text
    assert "None" not in text


def test_entry_chunks_metadata_and_deterministic_ids():
    entry = _entry()
    chunks = entry_chunks(entry)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["chunk_id"] == "juris-entry-12345678123456781234567812345678-000"
    md = c["metadata"]
    assert md["rechtsprechung_entry_id"] == str(entry.id)
    assert md["source_system"] == "research_verified"
    assert md["court"] == "VG Bremen"
    assert md["aktenzeichen"] == "3 K 320/22"
    assert md["decision_date"] == "2024-02-16"
    assert md["language"] == "de"
    assert md["chunk_index"] == 0
    assert c["context_header"].startswith("Rechtsprechung")
    assert any(p.startswith("entry:") for p in c["provenance"])
    # zweiter Aufruf identisch (Backfill re-run-sicher)
    assert entry_chunks(entry)[0]["chunk_id"] == c["chunk_id"]


def test_entry_chunks_splits_long_text():
    entry = _entry(key_holdings=[f"Holding {i}: " + "x" * 300 for i in range(20)])
    chunks = entry_chunks(entry)
    assert len(chunks) > 1
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    ids = [c["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)


def test_upsert_chunks_batches_and_sums(monkeypatch):
    monkeypatch.setenv("RAG_SERVICE_URL", "http://rag.test")
    calls = []

    def fake_post(url, json, headers):
        calls.append(json)
        return {"upserted": len(json["chunks"])}

    payload = [{"chunk_id": f"c{i}", "text": "t", "metadata": {}} for i in range(35)]
    total = upsert_chunks(payload, collection="jurisprudence", post_fn=fake_post)
    assert total == 35
    assert len(calls) == 3  # 16 + 16 + 3
    assert all(j["collection"] == "jurisprudence" for j in calls)
