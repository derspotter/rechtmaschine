"""Purge-Entscheidung für Trivial-Korrespondenz im Kanzlei-RAG-Store.
Pure Funktion — konservativ: gelöscht wird nur kurz UND rein prozedural."""
from purge_rag_trivia import purge_decision


def test_short_pure_boilerplate_is_purged():
    bucket, reason = purge_decision(
        total_chars=600,
        schlagworte=["akteneinsicht", "verwaltungsgericht"],
        normen=[],
        herkunftsland=None,
    )
    assert bucket == "PURGE"
    assert "boilerplate" in reason


def test_short_with_specific_tag_goes_to_review():
    bucket, _ = purge_decision(
        total_chars=900,
        schlagworte=["akteneinsicht", "traumatisierung"],
        normen=[],
        herkunftsland=None,
    )
    assert bucket == "REVIEW"


def test_short_with_norm_goes_to_review():
    bucket, _ = purge_decision(
        total_chars=900,
        schlagworte=["akteneinsicht"],
        normen=["§ 60 Abs. 5 AufenthG"],
        herkunftsland=None,
    )
    assert bucket == "REVIEW"


def test_short_with_country_goes_to_review():
    bucket, _ = purge_decision(
        total_chars=900,
        schlagworte=["asylverfahren"],
        normen=[],
        herkunftsland="Syrien",
    )
    assert bucket == "REVIEW"


def test_midlength_boilerplate_goes_to_review():
    bucket, _ = purge_decision(
        total_chars=2200,
        schlagworte=["akteneinsicht", "prozesskostenhilfe"],
        normen=[],
        herkunftsland=None,
    )
    assert bucket == "REVIEW"


def test_long_document_is_kept():
    bucket, _ = purge_decision(
        total_chars=8000,
        schlagworte=["akteneinsicht"],
        normen=[],
        herkunftsland=None,
    )
    assert bucket == "KEEP"


def test_untagged_short_doc_goes_to_review_not_purge():
    # Ohne (frische) Tags fehlt das Inhalts-Signal — nie blind löschen.
    bucket, _ = purge_decision(
        total_chars=400, schlagworte=[], normen=[], herkunftsland=None
    )
    assert bucket == "REVIEW"
