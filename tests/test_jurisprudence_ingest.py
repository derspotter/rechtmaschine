"""asyl.net ingest: footer parsing + source-URL sanity (pure parts, no network).

Fixtures are the REAL footer shapes that broke the store (12 entries repaired
by hand on 2026-07-05): the old parser only accepted "Court, vom DATE - AZ -"
and silently dropped court/Az on every other shape.

Run: .venv/bin/python -m pytest tests/test_jurisprudence_ingest.py -q
"""
from jurisprudence_ingest import canonical_detail_url, extraction_from_asylnet


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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} PASSED")
