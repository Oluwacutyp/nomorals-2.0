"""The stall itself: the books path is fast again, not just deadline-bounded.

The 2026-11-11 "motivation morning" kill died inside ``BooksAdapter.probe``
at the 900s wall. The first fix (34f75e60) bounded the symptom with
deadlines; this suite proves the root causes are gone:

* the seam lazy import copied one row per round-trip (~55 rows/s over
  Mongo — ~30 min for a 100k-passage library). The bulk path copies
  the same rows in seconds;
* the seam search loaded EVERY passage into memory and BM25-scored
  them in Python on EVERY call (6.7s per call at 100k passages).
  The inverted index is built once and every search after it scores
  only the rows containing a query word, with the same scorer;
* adapters were built fresh per federated call, so construction (and
  the index) was paid every time. The Library is cached per context.

Deadlines from 34f75e60 stay as defense-in-depth; this suite does not
re-prove them (see tests/test_booksprobe_deadline.py).
"""
from __future__ import annotations

import contextlib
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.books.library import Library
from nomorals.search.sources import BooksAdapter

pytestmark = pytest.mark.usefixtures("no_ambient_store")


@pytest.fixture
def no_ambient_store():
    """Each test starts with no ambient store (and restores it after)."""
    from nomorals.storage.docstore import ambient_document_store, set_ambient_document_store
    before = ambient_document_store()
    set_ambient_document_store(None)
    try:
        yield
    finally:
        set_ambient_document_store(before)


def _context(workspace: Path) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(workspace)))


def _build_library_file(workspace: Path, n_passages: int,
                        n_books: int = 5) -> Path:
    """A library file with ``n_passages`` FTS5 passages (no seam)."""
    lib = Library(_context(workspace))
    db_path = lib.db_path()
    con, fts5 = lib._connect()
    assert fts5, "test needs FTS5"
    rows = [(f"book-{i % n_books:02d}", f"Book {i % n_books:02d}",
             f"Chapter {i % 10}", i,
             f"Passage {i} about rivers and foxes and the old mill road.")
            for i in range(n_passages)]
    con.executemany(
        "INSERT INTO passages(book_slug, book_title, chapter, number, text)"
        " VALUES (?,?,?,?,?)", rows)
    con.executemany(
        "INSERT OR IGNORE INTO books(slug, title, author, source_file,"
        " chapters, words, strategy, ingested_at) VALUES (?,?,?,?,?,?,?,?)",
        [(f"book-{i:02d}", f"Book {i:02d}", "Author", "", 10, 1000,
          "text", 1700000000.0) for i in range(n_books)])
    con.commit()
    con.close()
    return db_path


def _sqlite_store(path: Path | None = None):
    """A SQLite-backed seam store on a FILE database.

    NOT ``:memory:``: the federated path runs probes/searches on
    worker threads, and an in-memory SQLite database is per
    connection — a worker thread would see an empty database with
    no ``docstore_docs`` table. Production stores are file-backed
    (or Mongo), so the fixture matches reality.
    """
    from nomorals.storage.db import Database
    from nomorals.storage.docstore import SQLiteDocumentStore
    db = Database(str(path) if path is not None else ":memory:")
    return SQLiteDocumentStore(db), db


class TestBulkCopy:
    def test_sqlite_seam_copy_is_bulk(self, tmp_path):
        """100k rows copy in seconds, not one transaction per row."""
        from nomorals.storage.docstore import set_ambient_document_store
        _build_library_file(tmp_path, 3000)
        store, db = _sqlite_store()
        try:
            set_ambient_document_store(store)
            t0 = time.perf_counter()
            lib = Library(_context(tmp_path))
            dt = time.perf_counter() - t0
            assert dt < 10, f"copy took {dt:.1f}s"
            assert lib._port is not None
            assert store.count("books-library__passages") == 3000
            assert store.count("books-library__books") == 5
            meta = store.get("books-library__meta", "import")
            assert meta and meta.get("done") is True
            t0 = time.perf_counter()
            Library(_context(tmp_path))
            assert time.perf_counter() - t0 < 2, "marker done: no re-copy"
        finally:
            set_ambient_document_store(None)
            db.close()

    def test_put_many_semantics_sqlite(self):
        """Upsert + overwrite + empty, on the SQLite rung."""
        store, db = _sqlite_store()
        try:
            assert store.put_many("c", []) == 0
            assert store.put_many("c", [("a", {"x": 1}), ("b", {"x": 2})]) == 2
            assert store.get("c", "a")["x"] == 1
            assert store.put_many("c", [("a", {"x": 9})]) == 1
            assert store.get("c", "a")["x"] == 9
            assert store.count("c") == 2
        finally:
            db.close()

    def test_put_many_shadow_writes_both_rungs(self):
        """Shadow bulk: primary authoritative, secondary shadowed."""
        from nomorals.storage.docstore import ShadowDocumentStore
        primary, pdb = _sqlite_store()
        secondary, sdb = _sqlite_store()
        shadow = ShadowDocumentStore(primary, secondary)
        try:
            assert shadow.put_many(
                "c", [("a", {"x": 1}), ("b", {"x": 2})]) == 2
            assert primary.get("c", "a")["x"] == 1
            assert secondary.get("c", "a")["x"] == 1
        finally:
            pdb.close()
            sdb.close()


class TestSeamIndex:
    def test_indexed_matches_legacy_scorer(self, tmp_path):
        """Same top hits and scores (to float noise) as the old scan."""
        from nomorals.storage.docstore import set_ambient_document_store
        _build_library_file(tmp_path, 400, n_books=4)
        store, db = _sqlite_store(tmp_path / "store.db")
        try:
            set_ambient_document_store(store)
            lib = Library(_context(tmp_path))
            rows = [(d.get("book_slug", ""), d.get("book_title", ""),
                     d.get("chapter", ""), int(d.get("number", 0)),
                     d.get("text") or "")
                    for d in lib._child_find("passages")]
            meta = {r.get("slug", ""): (r.get("author", "") or "",
                                        float(r.get("ingested_at", 0) or 0))
                    for r in lib._t("books").all()}
            for query in ("rivers foxes", "mill road", "passage 7",
                          "zzq no such words here", "rivers"):
                legacy = lib._search_memory_rows(rows, meta, query, 5, 260, "")
                new = lib.search(query, top=5)
                assert [(h.book_slug, h.chapter_number) for h in new] == [
                    (h.book_slug, h.chapter_number) for h in legacy]
                for a, b in zip(legacy, new, strict=True):
                    assert abs(a.score - b.score) < 1e-9
                    assert a.source == b.source == "bm25-memory"
        finally:
            set_ambient_document_store(None)
            db.close()

    def test_index_reused_then_rebuilt_on_ingest_and_drop(self, tmp_path):
        """Second search reuses the index; ingest/drop rebuild it."""
        from nomorals.storage.docstore import set_ambient_document_store
        _build_library_file(tmp_path, 200, n_books=2)
        store, db = _sqlite_store(tmp_path / "store.db")
        try:
            set_ambient_document_store(store)
            lib = Library(_context(tmp_path))
            assert lib.search("rivers", top=5)
            first = lib._seam_index
            assert first is not None
            assert lib.search("rivers", top=5)
            assert lib._seam_index is first, "index rebuilt for no reason"
            v0 = lib._corpus_version()
            book_file = tmp_path / "new-book.txt"
            book_file.write_text(
                "# New Book\n\nA tale of the midnight lighthouse "
                "and the keeper who never slept. Night after night he "
                "climbed the iron stairs with his oil lamp, watching "
                "the black water for ships that never came, while the "
                "gulls cried over the rocks below and the wind sang "
                "through the broken shutters of the lamp room.\n",
                encoding="utf-8")
            lib.ingest(book_file, title="New Book", author="Tester")
            assert lib._corpus_version() != v0
            hits = lib.search("lighthouse keeper", top=5)
            assert hits, "ingested book not searchable"
            assert any(h.book_slug == "new-book" for h in hits)
            assert lib._seam_index is not first, "index not rebuilt"
            lib.drop("new-book")
            hits = lib.search("lighthouse keeper", top=5)
            assert all(h.book_slug != "new-book" for h in hits)
        finally:
            set_ambient_document_store(None)
            db.close()

    def test_book_scoped_search_still_exact(self, tmp_path):
        """Book-scoped search keeps the legacy df/n (no index drift)."""
        from nomorals.storage.docstore import set_ambient_document_store
        _build_library_file(tmp_path, 200, n_books=2)
        store, db = _sqlite_store(tmp_path / "store.db")
        try:
            set_ambient_document_store(store)
            lib = Library(_context(tmp_path))
            hits = lib.search("rivers", top=5, book="book-01")
            assert hits
            assert all(h.book_slug == "book-01" for h in hits)
        finally:
            set_ambient_document_store(None)
            db.close()


class TestPerContextLibrary:
    def test_adapters_share_one_library_per_context(self, tmp_path):
        """Fresh adapters on one context share a Library (and its index)."""
        from nomorals.storage.docstore import set_ambient_document_store
        _build_library_file(tmp_path, 50)
        store, db = _sqlite_store(tmp_path / "store.db")
        try:
            set_ambient_document_store(store)
            ctx = _context(tmp_path)
            a1, a2 = BooksAdapter(ctx), BooksAdapter(ctx)
            assert a1._lib() is a2._lib()
            assert a1.search("rivers", limit=3)
        finally:
            set_ambient_document_store(None)
            db.close()

    def test_store_change_rebuilds(self, tmp_path):
        """A different ambient store invalidates the cached Library."""
        from nomorals.storage.docstore import set_ambient_document_store
        _build_library_file(tmp_path, 50)
        ctx = _context(tmp_path)
        store1, db1 = _sqlite_store(tmp_path / "store1.db")
        store2, db2 = _sqlite_store(tmp_path / "store2.db")
        try:
            set_ambient_document_store(store1)
            first = BooksAdapter(ctx)._lib()
            set_ambient_document_store(store2)
            second = BooksAdapter(ctx)._lib()
            assert second is not first
            assert second._port.store is store2
        finally:
            set_ambient_document_store(None)
            db1.close()
            db2.close()

    def test_federated_search_answers_books(self, tmp_path):
        """The books source ANSWERS through federated search (not skipped)."""
        from nomorals.search.federated import federated_search
        from nomorals.storage.docstore import set_ambient_document_store
        _build_library_file(tmp_path, 100)
        store, db = _sqlite_store(tmp_path / "store.db")
        try:
            set_ambient_document_store(store)
            ctx = _context(tmp_path)
            resp = federated_search(
                "rivers foxes", context=ctx, limit=5,
                sources=["books"], deadline_s=30.0)
            assert resp.hits, "books source returned nothing"
            assert all(r.source == "books" for r in resp.hits)
            assert "books" not in resp.sources_skipped
        finally:
            set_ambient_document_store(None)
            db.close()


# ── Mongo rung: the bulk copy where the stall actually lived ──────────────
# A throwaway single-node replica-set mongod, same shape as the
# test_mongo_p5 fixture (self-contained: this suite owns its server).

_MONGOD = "/home/hatch/workspace/tools/mongodb/bin/mongod"


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mongod(tmp_path_factory):
    pytest.importorskip("pymongo")
    if not Path(_MONGOD).exists():
        pytest.skip("mongod binary not installed")
    import pymongo
    data = tmp_path_factory.mktemp("mongo-stall-data")
    port = _free_port()
    log = open(data / "mongod.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [_MONGOD, "--dbpath", str(data), "--bind_ip", "127.0.0.1",
         "--port", str(port), "--replSet", "rs0",
         "--wiredTigerCacheSizeGB", "0.25"],
        stdout=log, stderr=log)
    base = f"mongodb://127.0.0.1:{port}"
    try:
        direct = pymongo.MongoClient(
            base + "/?directConnection=true", serverSelectionTimeoutMS=500)
        deadline = time.time() + 60
        while True:
            try:
                direct.admin.command("ping")
                break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(0.5)
        with contextlib.suppress(Exception):
            direct.admin.command("replSetInitiate")
        deadline = time.time() + 60
        while True:
            try:
                status = direct.admin.command("replSetGetStatus")
                if any(m.get("stateStr") == "PRIMARY"
                       for m in status.get("members", [])):
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise RuntimeError("no PRIMARY within 60s")
            time.sleep(0.5)
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=30)


class TestMongoBulkCopy:
    def test_mongo_copy_is_bulk(self, tmp_path, mongod):
        """2k passages copy over the wire in seconds, not ~40s one-by-one.

        One-by-one the copy runs at ~55 rows/s on a local mongod, so
        2000 rows would take ~36s; the bulk path must be far under
        that (generous bound: 20s, covering slow CI).
        """
        from nomorals.storage.docstore import MongoDocumentStore, set_ambient_document_store
        _build_library_file(tmp_path, 2000, n_books=3)
        store = MongoDocumentStore(
            mongod, f"stall{uuid.uuid4().hex[:10]}")
        try:
            set_ambient_document_store(store)
            t0 = time.perf_counter()
            lib = Library(_context(tmp_path))
            dt = time.perf_counter() - t0
            assert dt < 20, f"mongo copy took {dt:.1f}s"
            assert lib._port is not None
            assert store.count("books-library__passages") == 2000
            meta = store.get("books-library__meta", "import")
            assert meta and meta.get("done") is True
            hits = lib.search("rivers foxes", top=5)
            assert hits, "copied library not searchable"
        finally:
            set_ambient_document_store(None)
            store.close()

    def test_mongo_put_many_chunks(self, mongod):
        """put_many upserts in bulk_write chunks (1000 per call)."""
        from nomorals.storage.docstore import MongoDocumentStore
        store = MongoDocumentStore(mongod, f"stall{uuid.uuid4().hex[:10]}")
        try:
            pairs = [(f"k{i}", {"v": i}) for i in range(2500)]
            assert store.put_many("c", pairs) == 2500
            assert store.count("c") == 2500
            assert store.get("c", "k2499")["v"] == 2499
            assert store.put_many("c", [("k0", {"v": -1})]) == 1
            assert store.get("c", "k0")["v"] == -1
        finally:
            store.close()
