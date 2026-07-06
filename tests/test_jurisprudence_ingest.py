"""asyl.net ingest: footer parsing + source-URL sanity (pure parts, no network).

Fixtures are the REAL footer shapes that broke the store (12 entries repaired
by hand on 2026-07-05): the old parser only accepted "Court, vom DATE - AZ -"
and silently dropped court/Az on every other shape.

Run: .venv/bin/python -m pytest tests/test_jurisprudence_ingest.py -q
"""
from endpoints.rechtsprechung_playbook import RechtsprechungExtraction
from jurisprudence_ingest import (
    _extraction_from_payload,
    _strip_ctrl_deep,
    canonical_detail_url,
    extraction_from_asylnet,
    merge_footer_and_llm,
)


def _rec(footer, **kw):
    rec = {"footer": footer, "footer_date": None, "schlagworte": [], "court_heading": ""}
    rec.update(kw)
    return rec


def _extract(footer, **kw):
    return extraction_from_asylnet(_rec(footer, **kw), vocab=None)


def test_legacy_footer_shape_still_parses():
    t = _extract("VG Berlin, vom 09.10.2025 - 1 K 6/24 A - asyl.net: M33834")
    assert t.court == "VG Berlin"
    assert t.decision_date == "09.10.2025"
    assert t.aktenzeichen == "1 K 6/24 A"


def test_decision_type_between_comma_and_vom():
    # Broke M33884: court/Az were dropped because of "Beschluss" before "vom".
    t = _extract("VG Schleswig-Holstein, Beschluss vom 22.10.2025 - 11 B 165/25 - asyl.net: M33884")
    assert t.court == "VG Schleswig-Holstein", t.court
    assert t.decision_date == "22.10.2025"
    assert t.aktenzeichen == "11 B 165/25", t.aktenzeichen


def test_no_comma_before_type():
    # Detail-page shape (M33602): "VG Hamburg Urteil vom ...".
    t = _extract("VG Hamburg Urteil vom 06.06.2025 - 4 A 139/25 - asyl.net: M33602")
    assert t.court == "VG Hamburg", t.court
    assert t.aktenzeichen == "4 A 139/25", t.aktenzeichen


def test_erlass_without_aktenzeichen():
    # M33381: Erlasse legitimately have no Az — court must still be captured,
    # Az stays None WITHOUT a warning.
    t = _extract(
        "Bundesministerium des Innern, Erlass/Behördliche Mitteilung vom 10.04.2025 - - asyl.net: M33381"
    )
    assert t.court == "Bundesministerium des Innern", t.court
    assert t.decision_date == "10.04.2025"
    assert t.aktenzeichen is None
    assert not t.warnings, t.warnings


def test_eugh_party_names_truncated_from_az():
    t = _extract("EuGH, Urteil vom 04.06.2026 - C-147/24 [Safi] - V. gegen Niederlande - asyl.net: M34281")
    assert t.court == "EuGH", t.court
    assert t.aktenzeichen == "C-147/24 [Safi]", t.aktenzeichen


def test_journal_citation_suffix_stripped():
    t = _extract("VG Berlin, vom 09.10.2025 - 1 K 6/24 A (Asylmagazin 10/2025) - asyl.net: M33834")
    assert t.aktenzeichen == "1 K 6/24 A", t.aktenzeichen


def test_unparseable_footer_falls_back_to_heading_and_warns():
    t = _extract("völlig kaputter footer", court_heading="VG Beispielstadt", footer_date="01.02.2026")
    assert t.court == "VG Beispielstadt", t.court
    assert t.decision_date == "01.02.2026"
    assert t.aktenzeichen is None
    assert any("Aktenzeichen" in w for w in t.warnings), t.warnings


def test_court_decision_without_az_warns():
    t = _extract("VG Berlin, Urteil vom 09.10.2025 - - asyl.net: M99999")
    assert t.court == "VG Berlin"
    assert t.aktenzeichen is None
    assert any("Aktenzeichen" in w for w in t.warnings), t.warnings


def test_canonical_url_rebuilt_from_m_number():
    # M34106 shipped with url .../rsdb/m-1 (404). The M-number is authoritative.
    assert (
        canonical_detail_url("https://www.asyl.net/rsdb/m-1", "M34106")
        == "https://www.asyl.net/rsdb/m34106"
    )


def test_canonical_url_kept_when_consistent():
    url = "https://www.asyl.net/rsdb/m33884"
    assert canonical_detail_url(url, "M33884") == url


def test_canonical_url_without_m_number_unchanged():
    url = "https://www.asyl.net/rsdb/whatever"
    assert canonical_detail_url(url, None) == url


# --- footer + LLM merge (regex citation wins, LLM fills semantics) ---

def _footer_tags(**kw):
    base = dict(country="Syrien", tags=["existenzminimum"], court="VG Berlin",
                court_level="VG", decision_date="09.10.2025", aktenzeichen="1 K 6/24 A")
    base.update(kw)
    return RechtsprechungExtraction(**base)


def _llm_tags(**kw):
    base = dict(country="Syrien", tags=["frau"], court="Verwaltungsgericht Berlin",
                court_level="VG", decision_date="2025-10-09", aktenzeichen="1 K 6/24 A",
                outcome="grant", key_facts=["Klägerin aus Syrien"],
                key_holdings=["Tragende Erwägung."], summary="Zusammenfassung.",
                confidence=0.9)
    base.update(kw)
    return RechtsprechungExtraction(**base)


def test_merge_agreement_no_warnings():
    t = merge_footer_and_llm(_footer_tags(), _llm_tags())
    assert t.aktenzeichen == "1 K 6/24 A"
    assert t.court == "VG Berlin"          # footer wins
    assert t.outcome == "grant"            # semantics from LLM
    assert t.key_holdings == ["Tragende Erwägung."]
    assert t.summary == "Zusammenfassung."
    assert not t.warnings, t.warnings
    assert "frau" in t.tags and "existenzminimum" in t.tags


def test_merge_az_mismatch_keeps_footer_and_warns():
    t = merge_footer_and_llm(_footer_tags(), _llm_tags(aktenzeichen="1 K 9/24 A"))
    assert t.aktenzeichen == "1 K 6/24 A", t.aktenzeichen
    assert any("Az-Abweichung" in w for w in t.warnings), t.warnings


def test_merge_az_case_and_whitespace_not_a_mismatch():
    t = merge_footer_and_llm(_footer_tags(), _llm_tags(aktenzeichen="1  k 6/24  a"))
    assert not t.warnings, t.warnings


def test_merge_az_court_prefix_not_a_mismatch():
    # LLMs read "BVerwG 1 C 3.24" from the rubrum; the footer says "1 C 3.24".
    t = merge_footer_and_llm(_footer_tags(aktenzeichen="1 C 3.24"),
                             _llm_tags(aktenzeichen="BVerwG 1 C 3.24"))
    assert not t.warnings, t.warnings


def test_merge_az_party_names_and_nickname_not_a_mismatch():
    t = merge_footer_and_llm(
        _footer_tags(aktenzeichen="C-185/24, C-189/24 [Tudmur] - RL und QS gg. Deutschland"),
        _llm_tags(aktenzeichen="C-185/24, C-189/24"),
    )
    assert not t.warnings, t.warnings


def test_merge_az_benign_format_classes_not_a_mismatch():
    # Observed in the 2026-07-05 backfill: unicode hyphen, trailing court
    # token, internal whitespace, journal cites / "(Wx)" suffixes.
    cases = [
        ("C‑646/21 - K. und L. gg. Niederlande", "C-646/21"),
        ("4 LB 785/20 OVG", "4 LB 785/20"),
        ("1 K 558/23. F.A", "1 K 558/23.F.A"),
        ("RN 2 K 23.30938 (Asylmagazin 1-2/2024, S. 36 ff.)", "RN 2 K 23.30938"),
        ("19 W 3/23", "19 W 3/23 (Wx)"),
    ]
    for footer_az, llm_az in cases:
        t = merge_footer_and_llm(_footer_tags(aktenzeichen=footer_az),
                                 _llm_tags(aktenzeichen=llm_az))
        assert not t.warnings, (footer_az, llm_az, t.warnings)


def test_merge_az_register_letter_difference_still_warns():
    t = merge_footer_and_llm(_footer_tags(aktenzeichen="RN 4 S 24.33223"),
                             _llm_tags(aktenzeichen="RO 4 S 24.33223"))
    assert any("Az-Abweichung" in w for w in t.warnings), t.warnings


def test_merge_az_chamber_difference_still_warns():
    # Substantive: the footer lost the chamber number — must stay loud.
    t = merge_footer_and_llm(_footer_tags(aktenzeichen="K 644/24 A"),
                             _llm_tags(aktenzeichen="VG 8 K 644/24 A"))
    assert any("Az-Abweichung" in w for w in t.warnings), t.warnings


def test_merge_llm_fills_missing_az_with_marker():
    footer = _footer_tags(aktenzeichen=None,
                          warnings=["asyl.net-Footer ohne Aktenzeichen ('...') — Fundstelle vor Zitierung prüfen"])
    t = merge_footer_and_llm(footer, _llm_tags())
    assert t.aktenzeichen == "1 K 6/24 A"
    assert any("aus Volltext" in w for w in t.warnings), t.warnings
    assert not any("ohne Aktenzeichen" in w for w in t.warnings), t.warnings


def test_merge_date_mismatch_warns_keeps_footer():
    t = merge_footer_and_llm(_footer_tags(), _llm_tags(decision_date="2025-09-08"))
    assert t.decision_date == "09.10.2025", t.decision_date
    assert any("Datums-Abweichung" in w for w in t.warnings), t.warnings


def test_merge_without_llm_returns_footer():
    footer = _footer_tags()
    assert merge_footer_and_llm(footer, None) is footer


# --- lenient validation of local-Qwen JSON ---

def test_payload_validation_happy_path():
    t = _extraction_from_payload({
        "country": "Syrien", "court": "VG Berlin", "decision_date": "2025-10-09",
        "aktenzeichen": "1 K 6/24 A", "outcome": "grant",
        "key_holdings": ["Erwägung 1.", "  ", "Erwägung 2."],
        "argument_patterns": [{"use_when": "X", "rebuttal": "Y", "notes": None}],
        "citations": [{"court": "BVerwG", "date": "2024-01-01", "aktenzeichen": "1 C 3.24"}],
        "confidence": "0.85",
    })
    assert t.court == "VG Berlin" and t.outcome == "grant"
    assert t.key_holdings == ["Erwägung 1.", "Erwägung 2."]
    assert t.argument_patterns[0].use_when == "X"
    assert t.citations[0].az == "1 C 3.24"      # aktenzeichen-alias accepted
    assert t.confidence == 0.85


def test_payload_validation_malformed_nested_items_dropped():
    t = _extraction_from_payload({
        "country": None,
        "argument_patterns": ["kein dict", {"use_when": "ok"}],
        "citations": "quatsch",
        "confidence": "hoch",
    })
    assert t.country == "Unbekannt"
    assert len(t.argument_patterns) == 1
    assert t.citations == []
    assert t.confidence is None


def test_payload_validation_non_dict_is_none():
    assert _extraction_from_payload(None) is None
    assert _extraction_from_payload("text") is None


def test_strip_ctrl_deep_cleans_nested_nul():
    # Postgres rejects NUL/C0 control chars; PDFs carry them, Gemini echoes them
    # (killed the first --backfill-llm run after 56 entries).
    dirty = {"summary": "Text\x00mit\x01NUL", "key_facts": ["ok", "auch\x00hier"],
             "nested": {"a": "fein", "b": "\x0cform feed"}, "n": 3}
    clean = _strip_ctrl_deep(dirty)
    assert clean["summary"] == "TextmitNUL"
    assert clean["key_facts"] == ["ok", "auchhier"]
    assert clean["nested"]["b"] == "form feed"
    assert clean["n"] == 3
    assert "\n" in _strip_ctrl_deep("Zeile1\nZeile2")  # newlines survive


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
