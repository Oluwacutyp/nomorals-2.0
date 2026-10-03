"""Source adapters: reuse/wrap behavior for every federated source."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.library import SearchHit
from nomorals.documents import DocumentIndex
from nomorals.documents.model import Document, Section
from nomorals.memory.base import MemoryRecord
from nomorals.memory.manager import RecallResult
from nomorals.search.sources import (
    BooksAdapter,
    CodeAdapter,
    DocsAdapter,
    MemoryAdapter,
    TimelineAdapter,
    WisdomAdapter,
    build_adapters,
    default_doc_index_path,
    list_sources,
)
from nomorals.wisdom.corpus import Answer, ProvenanceHit


def _ctx(tmp=None):
    settings = SimpleNamespace(workspace_dir=tmp or tempfile.mkdtemp())
    return SimpleNamespace(db=None, settings=settings)


# ── books ──────────────────────────────────────────────────────────────

class _FakeLibrary:
    def __init__(self, db_path):
        self._db_path = Path(db_path)

    def db_path(self):
        return self._db_path

    def search(self, query, top=8, **kw):
        assert query
        return [SearchHit(book="The Book", title="The Book", chapter="Ch 1",
                          score=-1.5, passage="dragons fly here",
                          source="fts5", book_slug="slug1")]


def _library_db(path):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE passages (book_slug TEXT, book_title TEXT, "
                "chapter TEXT, number INTEGER, text TEXT)")
    con.execute("INSERT INTO passages VALUES (?,?,?,?,?)",
                ("slug1", "The Book", "Ch 1", 1, "dragons fly here"))
    con.commit()
    con.close()


class BooksAdapterTests(unittest.TestCase):
    def test_probe_and_search(self):
        tmp = tempfile.mkdtemp()
        db = Path(tmp) / "library.db"
        _library_db(db)
        ad = BooksAdapter(_ctx(tmp), library=_FakeLibrary(db))
        self.assertIsNone(ad.probe())
        hits = ad.search("dragons", limit=5)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h.source, "books")
        self.assertEqual(h.type, "book")
        self.assertEqual(h.title, "The Book")
        self.assertEqual(h.snippet, "dragons fly here")
        self.assertEqual(h.provenance["book_slug"], "slug1")
        self.assertEqual(h.provenance["chapter"], "Ch 1")
        self.assertIsNone(h.timestamp)

    def test_probe_missing_db(self):
        tmp = tempfile.mkdtemp()
        ad = BooksAdapter(_ctx(tmp), library=_FakeLibrary(Path(tmp) / "no.db"))
        self.assertIn("no books ingested", ad.probe())

    def test_probe_empty_db(self):
        tmp = tempfile.mkdtemp()
        db = Path(tmp) / "library.db"
        con = sqlite3.connect(str(db))
        con.execute("CREATE TABLE passages (x TEXT)")
        con.commit()
        con.close()
        ad = BooksAdapter(_ctx(tmp), library=_FakeLibrary(db))
        self.assertIn("empty", ad.probe())


# ── docs ───────────────────────────────────────────────────────────────

def _doc_index():
    idx = DocumentIndex()
    doc = Document(id="d1", title="Dragon Notes")
    doc.sections = [Section(level=1, heading="Intro",
                            text="dragons breathe fire over the valley")]
    idx.add(doc)
    return idx


class DocsAdapterTests(unittest.TestCase):
    def test_search_real_index(self):
        ad = DocsAdapter(_ctx(), index=_doc_index())
        self.assertIsNone(ad.probe())
        hits = ad.search("dragons", limit=5)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h.source, "docs")
        self.assertEqual(h.type, "doc")
        self.assertEqual(h.provenance["doc_id"], "d1")
        self.assertIn("dragon", h.snippet.lower())

    def test_no_index_skipped_with_note(self):
        ad = DocsAdapter(_ctx(), index_path="/nonexistent/idx.json")
        note = ad.probe()
        self.assertIsNotNone(note)
        self.assertIn("--dir", note)

    def test_dir_builds_ad_hoc_index(self):
        tmp = tempfile.mkdtemp()
        Path(tmp, "note.md").write_text("# dragons\n\nfire breathing lizards\n")
        ad = DocsAdapter(_ctx(), doc_dir=tmp)
        self.assertIsNone(ad.probe())
        hits = ad.search("dragons", limit=5)
        self.assertTrue(hits)

    def test_default_index_path(self):
        tmp = tempfile.mkdtemp()
        p = default_doc_index_path(_ctx(tmp))
        self.assertEqual(p, Path(tmp) / "documents" / "index.json")


# ── memory ─────────────────────────────────────────────────────────────

class _FakeManager:
    def stats_snapshot(self):
        return {"records": 2}

    def recall(self, query, limit=10):
        real = MemoryRecord(id="m1", kind="fact", content="dragons are real")
        real.semantic = 0.8
        real.lexical = 0.0
        real.score = 0.8
        recent = MemoryRecord(id="m2", kind="episode", content="ate lunch")
        # recency fallback: zero semantic AND zero lexical
        return RecallResult(records=[real, recent], query=query)


class MemoryAdapterTests(unittest.TestCase):
    def test_search_reuses_recall(self):
        ad = MemoryAdapter(_ctx(), manager=_FakeManager())
        self.assertIsNone(ad.probe())
        hits = ad.search("dragons", limit=5)
        self.assertEqual(len(hits), 1)  # recency fallback excluded
        h = hits[0]
        self.assertEqual(h.source, "memory")
        self.assertEqual(h.type, "memory")
        self.assertEqual(h.source_id, "memory:m1")
        self.assertEqual(h.provenance["kind"], "fact")
        self.assertIsNotNone(h.timestamp)

    def test_probe_empty(self):
        class Empty(_FakeManager):
            def stats_snapshot(self):
                return {"records": 0}
        ad = MemoryAdapter(_ctx(), manager=Empty())
        self.assertIn("no memories", ad.probe())


# ── wisdom ─────────────────────────────────────────────────────────────

class _FakeKeeper:
    def status(self):
        return {"corpus": {"ingested": 1}}

    def ask(self, query, top=5):
        return Answer(query=query, passages=[ProvenanceHit(
            work="Tao Te Ching", translator="Legge", section="Ch 1",
            url="https://example.org/tao", snippet="the tao that can be told",
            score=-2.0, canon_status="canon")])


class WisdomAdapterTests(unittest.TestCase):
    def test_provenance_passes_through(self):
        ad = WisdomAdapter(_ctx(), keeper=_FakeKeeper())
        self.assertIsNone(ad.probe())
        hits = ad.search("tao", limit=5)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h.source, "wisdom")
        self.assertEqual(h.type, "passage")
        self.assertEqual(h.title, "Tao Te Ching")
        # provenance is passed through verbatim, not flattened away
        self.assertEqual(h.provenance, {
            "work": "Tao Te Ching", "translator": "Legge", "section": "Ch 1",
            "url": "https://example.org/tao", "canon_status": "canon",
        })

    def test_probe_empty_corpus(self):
        class Empty(_FakeKeeper):
            def status(self):
                return {"corpus": {"ingested": 0}}
        ad = WisdomAdapter(_ctx(), keeper=Empty())
        self.assertIn("no ingested texts", ad.probe())


# ── code ───────────────────────────────────────────────────────────────

class _FakeIndexer:
    def __init__(self):
        self.db = SimpleNamespace(scalar=lambda sql, default=0: 3)

    async def search(self, query, limit=10):
        unit = SimpleNamespace(
            unit_id="u1", file_path="flight.py", name="fly",
            unit_type="function", language="python", code="def fly(): ...",
            docstring="makes dragons fly", line_start=10, line_end=20,
            parent="")
        return [SimpleNamespace(unit=unit, score=0.9)]


class CodeAdapterTests(unittest.TestCase):
    def test_search(self):
        ad = CodeAdapter(_ctx(), indexer=_FakeIndexer())
        self.assertIsNone(ad.probe())
        hits = ad.search("dragon flight", limit=5)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h.source, "code")
        self.assertEqual(h.type, "code")
        self.assertEqual(h.title, "fly (flight.py:10)")
        self.assertEqual(h.snippet, "makes dragons fly")
        self.assertEqual(h.provenance["file"], "flight.py")
        self.assertEqual(h.provenance["unit_type"], "function")

    def test_probe_empty(self):
        idx = _FakeIndexer()
        idx.db = SimpleNamespace(scalar=lambda sql, default=0: 0)
        ad = CodeAdapter(_ctx(), indexer=idx)
        self.assertIn("no code indexed", ad.probe())


# ── timeline ───────────────────────────────────────────────────────────

def _row(**kw):
    base = {"event_id": "e1", "topic": "mission.done", "ts": 1700000000.0,
            "source": "bus", "session_id": "", "project_id": "",
            "mission_id": "", "artifact_id": "",
            "data": {"note": "dragons deployed"}}
    base.update(kw)
    return base


class _FakeTimeline:
    def __init__(self, rows):
        self._rows = rows
        self.seen_kwargs = {}

    def query(self, **kw):
        self.seen_kwargs.update(kw)
        rows = list(self._rows)
        if kw.get("since") is not None:
            rows = [r for r in rows if (r.get("ts") or 0) >= kw["since"]]
        if kw.get("until") is not None:
            rows = [r for r in rows if (r.get("ts") or 0) <= kw["until"]]
        return rows[:kw.get("limit", 200)]


class TimelineAdapterTests(unittest.TestCase):
    def test_search_matches_topic_and_payload(self):
        tl = _FakeTimeline([_row(), _row(event_id="e2", topic="chat.msg",
                                         data={"note": "hello"})])
        ad = TimelineAdapter(tl)
        self.assertIsNone(ad.probe())
        hits = ad.search("dragons", limit=5)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h.source, "timeline")
        self.assertEqual(h.type, "event")
        self.assertEqual(h.title, "mission.done")
        self.assertEqual(h.timestamp, 1700000000.0)
        self.assertEqual(h.provenance["event_id"], "e1")
        self.assertEqual(h.provenance["topic"], "mission.done")

    def test_since_before_pushed_to_query(self):
        tl = _FakeTimeline([_row(ts=100.0), _row(event_id="e2", ts=300.0)])
        ad = TimelineAdapter(tl)
        hits = ad.search("mission", limit=5, since=200.0, before=400.0)
        self.assertEqual([h.provenance["event_id"] for h in hits], ["e2"])
        self.assertEqual(tl.seen_kwargs["since"], 200.0)
        self.assertEqual(tl.seen_kwargs["until"], 400.0)

    def test_empty_timeline_skipped(self):
        ad = TimelineAdapter(_FakeTimeline([]))
        self.assertIn("no events", ad.probe())

    def test_no_timeline_wired(self):
        ad = TimelineAdapter(None)
        self.assertIn("no timeline store", ad.probe())
        self.assertEqual(ad.search("x", limit=5), [])


# ── registry ───────────────────────────────────────────────────────────

class RegistryTests(unittest.TestCase):
    def test_list_sources(self):
        srcs = list_sources()
        self.assertEqual(
            [s["name"] for s in srcs],
            ["memory", "wisdom", "books", "docs", "code", "timeline"])
        self.assertTrue(all(s["type"] and s["description"] for s in srcs))

    def test_build_adapters_needs_context(self):
        from nomorals.search.errors import SearchError
        with self.assertRaises(SearchError):
            build_adapters(None)

    def test_build_adapters_all_six(self):
        ads = build_adapters(_ctx())
        self.assertEqual(set(ads), {"memory", "wisdom", "books", "docs",
                                   "code", "timeline"})


if __name__ == "__main__":
    unittest.main()
