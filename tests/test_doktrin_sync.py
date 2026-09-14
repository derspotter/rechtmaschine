import os
import sys
import types
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("RAG_SERVICE_URL", "http://rag.invalid")

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

# doktrin_sync -> database would spin up the real postgres engine; give it a
# sqlite-backed stand-in instead (JSONB rendered as JSON for sqlite DDL).
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.dialects.postgresql import JSONB  # noqa: E402
from sqlalchemy.ext.compiler import compiles  # noqa: E402
from sqlalchemy.orm import declarative_base, sessionmaker  # noqa: E402


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


Base = declarative_base()
_engine = create_engine(
    "sqlite://", connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=_engine)

sys.modules["database"] = types.SimpleNamespace(
    Base=Base, SessionLocal=SessionLocal, engine=_engine
)
sys.modules.setdefault(
    "shared",
    types.SimpleNamespace(ensure_anonymization_service_ready=lambda *a, **k: None),
)

import doktrin_sync  # noqa: E402
from doktrin_sync import WikiPage, sync  # noqa: E402

_REAL_FETCH_PAGE = doktrin_sync.fetch_page  # _install() swaps the module attr
from models import DoktrinPage  # noqa: E402

# doktrin_sync has captured the stubs; drop them so later test modules that
# import the REAL shared/database (e.g. via app.endpoints) don't get poisoned.
sys.modules.pop("shared", None)
sys.modules.pop("database", None)

Base.metadata.create_all(_engine, tables=[DoktrinPage.__table__])


LONG_BODY = "\n\n".join(
    f"Absatz {i}: " + "Die Lage hat sich weiter verändert. " * 8 for i in range(6)
)

PAGES = {
    "laender:afghanistan": (
        "====== Afghanistan ======\n\n"
        "===== Abschiebungsverbote =====\n\n" + LONG_BODY
    ),
    "asylgesetz:par_3": (
        "====== § 3 AsylG ======\n\n" + LONG_BODY
    ),
    "laender:stub": "====== Stub ======\nkurz.",
}


class Recorder:
    def __init__(self):
        self.upserts: list[list[dict]] = []
        self.deletes: list[list[str]] = []

    def upsert(self, chunks, collection):
        assert collection == "doktrin"
        self.upserts.append(list(chunks))
        return len(chunks)

    def delete(self, chunk_ids, collection):
        assert collection == "doktrin"
        self.deletes.append(list(chunk_ids))
        return len(chunk_ids)


def _install(monkeypatch_pages: dict[str, str], recorder: Recorder):
    def fake_discover(base_url, index_id, timeout):
        return sorted(monkeypatch_pages)

    def fake_fetch(base_url, page_id, timeout):
        text = monkeypatch_pages[page_id].strip()
        return WikiPage(
            page_id=page_id,
            title=doktrin_sync._extract_title(page_id, text),
            url=f"https://wiki.aufentha.lt/{page_id}",
            export_url=f"https://wiki.aufentha.lt/_export/raw/{page_id}",
            text=text,
            last_modified=None,
        )

    doktrin_sync.discover_page_ids = fake_discover
    doktrin_sync.fetch_page = fake_fetch
    doktrin_sync.upsert = recorder.upsert
    doktrin_sync.delete_chunks = recorder.delete


def _args(**overrides):
    argv = ["--delay", "0"]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return doktrin_sync.build_parser().parse_args(argv)


def _reset_db():
    db = SessionLocal()
    db.query(DoktrinPage).delete()
    db.commit()
    db.close()


def test_first_run_upserts_all_and_second_run_skips():
    _reset_db()
    rec = Recorder()
    _install(PAGES, rec)

    assert sync(_args()) == 0
    upserted_ids = [c["chunk_id"] for batch in rec.upserts for c in batch]
    assert upserted_ids and all(cid.startswith("doku-") for cid in upserted_ids)

    db = SessionLocal()
    afg = db.query(DoktrinPage).filter_by(page_id="laender:afghanistan").one()
    assert afg.status == "active"
    assert afg.chunk_count == len(afg.chunk_ids) > 0
    assert afg.title == "Afghanistan"
    stub = db.query(DoktrinPage).filter_by(page_id="laender:stub").one()
    assert stub.status == "thin" and stub.chunk_ids == []
    db.close()

    # Second run: identical content, nothing upserted or deleted.
    rec2 = Recorder()
    _install(PAGES, rec2)
    assert sync(_args()) == 0
    assert rec2.upserts == [] and rec2.deletes == []


def test_changed_page_deletes_exactly_the_old_chunk_ids():
    _reset_db()
    rec = Recorder()
    _install(PAGES, rec)
    sync(_args())

    db = SessionLocal()
    old_ids = list(
        db.query(DoktrinPage).filter_by(page_id="asylgesetz:par_3").one().chunk_ids
    )
    db.close()
    assert old_ids

    changed = dict(PAGES)
    changed["asylgesetz:par_3"] = PAGES["asylgesetz:par_3"] + "\n\nNeuer Absatz zur Rechtslage."
    rec2 = Recorder()
    _install(changed, rec2)
    assert sync(_args()) == 0
    assert rec2.deletes == [old_ids]

    db = SessionLocal()
    row = db.query(DoktrinPage).filter_by(page_id="asylgesetz:par_3").one()
    assert row.chunk_ids and row.chunk_ids != old_ids
    db.close()


def test_vanished_page_marked_gone_and_chunks_deleted():
    _reset_db()
    rec = Recorder()
    _install(PAGES, rec)
    sync(_args())

    db = SessionLocal()
    old_ids = list(
        db.query(DoktrinPage).filter_by(page_id="laender:afghanistan").one().chunk_ids
    )
    db.close()

    remaining = {k: v for k, v in PAGES.items() if k != "laender:afghanistan"}
    rec2 = Recorder()
    _install(remaining, rec2)
    assert sync(_args()) == 0
    assert old_ids in rec2.deletes

    db = SessionLocal()
    row = db.query(DoktrinPage).filter_by(page_id="laender:afghanistan").one()
    assert row.status == "gone" and row.chunk_ids == [] and row.chunk_count == 0
    db.close()


def test_limit_run_does_not_mark_gone():
    _reset_db()
    rec = Recorder()
    _install(PAGES, rec)
    sync(_args())

    rec2 = Recorder()
    _install({"asylgesetz:par_3": PAGES["asylgesetz:par_3"]}, rec2)
    sync(_args(limit=1))

    db = SessionLocal()
    row = db.query(DoktrinPage).filter_by(page_id="laender:afghanistan").one()
    assert row.status == "active"
    db.close()


def test_dry_run_touches_nothing():
    _reset_db()
    rec = Recorder()
    _install(PAGES, rec)
    assert sync(_args(dry_run=True)) == 0
    assert rec.upserts == [] and rec.deletes == []
    db = SessionLocal()
    assert db.query(DoktrinPage).count() == 0
    db.close()


def test_html_response_counts_as_failure_not_content():
    _reset_db()
    rec = Recorder()
    pages = dict(PAGES)
    pages["kaputt"] = "<!DOCTYPE html>\n<html><body>Seite nicht gefunden</body></html>"
    _install(pages, rec)
    assert sync(_args()) == 1  # failed pages -> exit 1
    db = SessionLocal()
    assert db.query(DoktrinPage).filter_by(page_id="kaputt").count() == 0
    db.close()


def test_chunk_payload_shape():
    _reset_db()
    rec = Recorder()
    _install(PAGES, rec)
    sync(_args(pages="laender:afghanistan"))
    chunk = rec.upserts[0][0]
    assert chunk["chunk_id"].startswith("doku-")
    assert "[Aufenthaltswiki | Afghanistan" in chunk["context_header"]
    md = chunk["metadata"]
    assert md["source_system"] == "dokuwiki"
    assert md["page_id"] == "laender:afghanistan"
    assert md["namespace"] == "laender"
    assert md["heading_path"]
    assert any(p.startswith("dokuwiki:") for p in chunk["provenance"])


# --- upsert-before-delete: a failed upsert must never cost existing chunks
# (14.09.2026: two nights of embedder outage deleted ~1,200 doktrin chunks
# because delete ran first and the rollback only restored the bookkeeping).


class OrderRecorder(Recorder):
    """Records the call order and optionally fails every upsert."""

    def __init__(self, fail_upsert: bool = False):
        super().__init__()
        self.events: list[str] = []
        self.fail_upsert = fail_upsert

    def upsert(self, chunks, collection):
        self.events.append("upsert")
        if self.fail_upsert:
            import httpx

            raise httpx.HTTPError("502 Bad Gateway: embed_failed")
        return super().upsert(chunks, collection)

    def delete(self, chunk_ids, collection):
        self.events.append("delete")
        return super().delete(chunk_ids, collection)


def _synced_row(page_id: str) -> DoktrinPage:
    db = SessionLocal()
    row = db.query(DoktrinPage).filter_by(page_id=page_id).one()
    db.expunge(row)
    db.close()
    return row


def test_failed_upsert_keeps_old_chunks_in_store_and_bookkeeping():
    _reset_db()
    _install(PAGES, Recorder())
    sync(_args())
    before = _synced_row("asylgesetz:par_3")

    changed = dict(PAGES)
    changed["asylgesetz:par_3"] = PAGES["asylgesetz:par_3"] + "\n\nNeuer Absatz."
    rec = OrderRecorder(fail_upsert=True)
    _install(changed, rec)
    assert sync(_args()) == 1

    assert rec.deletes == [], "old chunks must survive a failed upsert"
    after = _synced_row("asylgesetz:par_3")
    assert after.chunk_ids == before.chunk_ids
    assert after.content_sha256 == before.content_sha256


def test_changed_page_upserts_new_chunks_before_deleting_old_ones():
    _reset_db()
    _install(PAGES, Recorder())
    sync(_args())

    changed = dict(PAGES)
    changed["asylgesetz:par_3"] = PAGES["asylgesetz:par_3"] + "\n\nNeuer Absatz."
    rec = OrderRecorder()
    _install(changed, rec)
    assert sync(_args()) == 0
    assert rec.events == ["upsert", "delete"]


def test_full_resync_of_unchanged_page_never_deletes_its_own_new_chunks():
    _reset_db()
    _install(PAGES, Recorder())
    sync(_args())
    ids = list(_synced_row("asylgesetz:par_3").chunk_ids)

    rec = OrderRecorder()
    _install(PAGES, rec)
    # --full nulls the stored hashes, then re-runs sync: same content => same
    # chunk ids. Deleting "old" ids after the upsert would wipe the fresh ones.
    db = SessionLocal()
    db.query(DoktrinPage).update({DoktrinPage.content_sha256: None})
    db.commit()
    db.close()
    assert sync(_args()) == 0
    deleted = {cid for batch in rec.deletes for cid in batch}
    assert not deleted & set(ids)
    assert list(_synced_row("asylgesetz:par_3").chunk_ids) == ids


# --- BOM wobble: wiki.aufentha.lt serves the raw export sometimes with and
# sometimes without a leading U+FEFF. Nightly it flipped 100–500 pages as
# "updated" and, worse, the BOM hid the title heading from clean_markup.


def test_fetch_page_strips_utf8_bom_so_hash_and_heading_are_stable(monkeypatch):
    raw = "====== Art. 47 EUAA-VO ======\n\nText."
    calls = iter(["﻿" + raw, raw])
    monkeypatch.setattr(doktrin_sync, "_fetch_text", lambda url, timeout: (next(calls), None))

    with_bom = _REAL_FETCH_PAGE("https://wiki.aufentha.lt", "art._47", timeout=1)
    without = _REAL_FETCH_PAGE("https://wiki.aufentha.lt", "art._47", timeout=1)

    assert with_bom.text == without.text == raw
    assert doktrin_sync._page_sha("art._47", with_bom.text) == doktrin_sync._page_sha("art._47", without.text)
    assert doktrin_sync.clean_markup(with_bom.text).startswith("# Art. 47 EUAA-VO")
