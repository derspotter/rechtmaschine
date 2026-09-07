"""citation_identity: eine Zitat-Erkennung fuer Sperren, Whitelist, Fakten-Check.

    .venv/bin/python -m pytest tests/test_citation_identity.py -q
"""
import pytest

from citation_identity import Citation, canonical_az, find_citations


@pytest.mark.parametrize("raw, expected", [
    ("18 E 491/12", "18e491/12"),
    # NB: _AZ_COURT_PREFIX strips only the single court token "OVG", not the
    # following "NRW" locality qualifier — that was already the previous
    # jurisprudence_ingest._az_for_compare behaviour (verified against the
    # unmodified function before this refactor); this row pins it, not an
    # aspirational normalization.
    ("OVG NRW 18 E 491/12", "nrw18e491/12"),
    ("M 10 K 21.3767", "m10k21.3767"),
    ("VG 18 B 103/23", "18b103/23"),
    ("C-151/22 [Changu]", "c-151/22"),
    ("27 L 1491/24.A", "27l1491/24.a"),
    ("18 E 491/12 - Mustermann", "18e491/12"),
])
def test_canonical_az_matches_previous_normalization(raw, expected):
    assert canonical_az(raw) == expected


def test_find_citations_full_decision_with_bavarian_az():
    text = "Trägt (VGH Bayern, Beschluss vom 07.09.2022 – 10 ZB 22.1187)."
    hits = find_citations(text)
    assert [h.kind for h in hits] == ["decision"]
    hit = hits[0]
    assert hit.az == "10 ZB 22.1187" and hit.canonical == "10zb22.1187"
    assert hit.date == "07.09.2022"
    assert text[hit.az_start:hit.az_end] == "10 ZB 22.1187"
    assert text[hit.start:hit.end].startswith("VGH Bayern")


def test_find_citations_bare_forms():
    text = "Vgl. M 10 K 21.3767, 18E491/12, C-151/22 und EGMR Nr. 12345/19."
    assert [(h.kind, h.canonical) for h in find_citations(text)] == [
        ("az", "m10k21.3767"), ("az", "18e491/12"), ("az", "c-151/22"), ("az", "nr.12345/19"),
    ]


def test_find_citations_ignores_eu_norms_and_statutes():
    text = ("Anspruch aus Art. 14 Abs. 2 RL 2008/115/EG, Art. 3 RL 2011/95, "
            "VO (EU) Nr. 604/2013 und § 60a Abs. 2 S. 1 AufenthG.")
    assert find_citations(text) == []


def test_find_citations_prefix_belongs_to_az_not_to_prose():
    hits = find_citations("Az 5 K 9/23 wurde zitiert.")
    assert len(hits) == 1 and hits[0].raw == "5 K 9/23"


def test_find_citations_decision_span_excludes_names_from_az_span():
    text = "VG Teststadt, Frau Mustermann, Urteil vom 01.02.2020 – 18 E 491/12"
    hit = find_citations(text)[0]
    assert hit.kind == "decision"
    assert "Mustermann" not in text[hit.az_start:hit.az_end]


@pytest.mark.parametrize("raw", [
    "18 E 491/12", "OVG NRW 18 E 491/12", "M 10 K 21.3767",
    "VG 18 B 103/23", "C-151/22 [Changu]", "27 L 1491/24.A",
    "18 E 491/12 - Mustermann", "", None,
])
def test_jurisprudence_ingest_and_verify_source_delegate_to_canonical_az(raw):
    from jurisprudence_ingest import _az_for_compare
    from verify_source import az_for_compare

    assert _az_for_compare(raw) == canonical_az(raw)
    assert az_for_compare(raw) == canonical_az(raw)


def test_citation_identity_module_stays_import_light():
    """The module must not pull in endpoints/database/models/verify_source/
    jurisprudence_ingest — importing jurisprudence_ingest drags in auth,
    which needs SECRET_KEY at import time."""
    import ast
    from pathlib import Path

    src = Path(__file__).parent.parent.joinpath("app", "citation_identity.py").read_text()
    tree = ast.parse(src)
    forbidden = {"endpoints", "database", "models", "verify_source", "jurisprudence_ingest"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & forbidden), imported & forbidden
