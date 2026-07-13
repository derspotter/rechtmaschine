"""render_pack_block (unscorter Legacy-Pfad): Freiform-Einträge, die dicts
enthalten (Argumentationsmuster aus dem Enrichment), müssen als Prosa
rendern, nie als roher Python-dict im Prompt-Block.

Run: .venv/bin/python -m pytest tests/test_render_pack_block.py -q
"""
from types import SimpleNamespace

from endpoints.jurisprudence import render_pack_block


def _pack(decisions):
    return SimpleNamespace(contents={"decisions": decisions})


def test_pack_block_renders_dict_argument_pattern_as_prose():
    block = render_pack_block(_pack([{
        "court": "VG Test",
        "decision_date": "2026-01-01",
        "aktenzeichen": "1 K 1/26",
        "outcome": "stattgegeben",
        "holdings": ["Erste tragende Erwägung"],
        "argument_patterns": [
            {"use_when": "Wann-Text", "rebuttal": "Rebuttal-Text", "notes": "Notiz"}
        ],
    }]))
    assert "{" not in block, block
    assert "Rebuttal-Text" in block
    assert "Wann-Text" in block


def test_pack_block_renders_dict_holding_as_prose():
    block = render_pack_block(_pack([{
        "court": "VG Test",
        "holdings": [{"notes": "Existenzminimum nicht gesichert."}],
    }]))
    assert "{" not in block, block
    assert "Existenzminimum nicht gesichert." in block
