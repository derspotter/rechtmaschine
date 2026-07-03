"""Unit tests for the controlled-vocabulary normalizer. Pure Python, no DB/GPU.
Run: python tests/test_rag_vocabulary.py"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag_vocabulary import (
    Vocabulary, normalize_themen, normalize_country, normalize_normen,
    tag_line, facet_metadata,
)

VOCAB = Vocabulary(
    themen=["wehrdienstentziehung", "internationaler schutz in eu-staat", "abschiebungsverbot"],
    themen_aliases={"wehrdienstverweigerung": "wehrdienstentziehung",
                    "militärdienstentziehung": "wehrdienstentziehung"},
    laender=["Syrien", "Griechenland", "Afghanistan"],
    laender_aliases={"arabische republik syrien": "Syrien", "syrische": "Syrien"},
    normen=["§ 3 AsylG", "§ 60 Abs. 7 AufenthG", "Art. 3 EMRK"],
    normen_aliases={"§ 3 asylg": "§ 3 AsylG", "art 3 emrk": "Art. 3 EMRK"},
)

def test_themen_canonical_and_alias():
    out = normalize_themen(VOCAB, ["Wehrdienstverweigerung", "Abschiebungsverbot", "unbekanntes thema"])
    assert out == ["wehrdienstentziehung", "abschiebungsverbot"], out

def test_themen_dedup_preserves_order():
    out = normalize_themen(VOCAB, ["Abschiebungsverbot", "Militärdienstentziehung", "Wehrdienstentziehung"])
    assert out == ["abschiebungsverbot", "wehrdienstentziehung"], out

def test_country_alias_and_unknown():
    assert normalize_country(VOCAB, "Arabische Republik Syrien") == "Syrien"
    assert normalize_country(VOCAB, "Griechenland") == "Griechenland"
    assert normalize_country(VOCAB, "Narnia") is None
    assert normalize_country(VOCAB, None) is None
    assert normalize_country(VOCAB, "") is None

def test_normen_alias_and_filter():
    out = normalize_normen(VOCAB, ["§ 3 AsylG", "art 3 emrk", "§ 99 NichtExist"])
    assert out == ["§ 3 AsylG", "Art. 3 EMRK"], out

def test_tag_line_format():
    line = tag_line(["griechenland"], "Griechenland", ["§ 29 AsylG"])
    assert "Schlagwörter: griechenland" in line
    assert "Herkunftsland: Griechenland" in line
    assert "Normen: § 29 AsylG" in line

def test_tag_line_empty_when_nothing():
    assert tag_line([], None, []) == ""

def test_facet_metadata_keys():
    md = facet_metadata(["abschiebungsverbot"], "Syrien", ["§ 3 AsylG"])
    assert md == {"schlagworte": ["abschiebungsverbot"],
                  "applicant_origin": "Syrien",
                  "citations": ["§ 3 AsylG"]}, md

def test_facet_metadata_omits_empty():
    md = facet_metadata([], None, [])
    assert md == {}, md

def test_themen_extra_survive_loader():
    # Hand-curated themen live in themen_extra so build_vocabulary re-runs
    # (which regenerate themen from DB counts) don't wipe them.
    import json, tempfile
    from rag_vocabulary import load_vocabulary
    data = {"themen": ["abschiebungsverbot"], "themen_extra": ["netzwerk"],
            "themen_aliases": {}, "laender": [], "laender_aliases": {},
            "normen": [], "normen_aliases": {}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(data, fh)
        path = fh.name
    vocab = load_vocabulary(path)
    assert "netzwerk" in vocab.themen, vocab.themen
    assert "abschiebungsverbot" in vocab.themen

def test_real_vocab_has_netzwerk_and_rueckkehr():
    from rag_vocabulary import load_vocabulary
    vocab = load_vocabulary()
    assert "netzwerk" in vocab.themen, "themen fehlt: netzwerk"
    assert "rückkehr" in vocab.themen, "themen fehlt: rückkehr"

def test_real_vocab_netzwerk_aliases():
    from rag_vocabulary import load_vocabulary
    vocab = load_vocabulary()
    out = normalize_themen(vocab, ["soziale Bindungen", "familiäre Unterstützung"])
    assert out == ["netzwerk"], out

def test_real_vocab_rueckkehr_aliases():
    from rag_vocabulary import load_vocabulary
    vocab = load_vocabulary()
    out = normalize_themen(vocab, ["Rückkehrer", "Rückkehrsituation"])
    assert out == ["rückkehr"], out

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
