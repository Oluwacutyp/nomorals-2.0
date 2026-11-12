"""BooksAdapter.probe hang + federated deadlines (2026-11-11 live defect).

The "motivation morning" job died at its 900s wall with the watchdog
naming ``BooksAdapter.probe`` as the stuck call: the federated probe
phase ran probes sequentially with no deadline, and the books probe
itself did unbounded work (a full FTS5 ``COUNT(*)`` scan, plus a
``Library`` construction that can run a full seam import when a
document store is ambient). These tests prove the hang is eliminated
at every layer — probe, federated search, _web_hunt — using fake
clocks (short deadlines) and releasable hangs, never a real wall.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nomorals.search.sources as _sources_mod
from nomorals.agents import teachings
from nomorals.search.errors import SearchError
from nomorals.search.federated import federated_search
from nomorals.search.model import SearchResponse, SearchResult
from nomorals.search.sources import BooksAdapter, valid_source_names


def _tmpdir():
    # Disk-backed tmp (this box's /tmp is a 512MB tmpfs under load).
    return tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR"))


def _ctx(tmp):
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings)


def _library_db(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE passages (book_slug TEXT, book_title TEXT, "
                "chapter TEXT, number INTEGER, text TEXT)")
    con.execute("INSERT INTO passages VALUES (?,?,?,?,?)",
                ("slug1", "The Book", "Ch 1", 1, "dragons fly here"))
    con.commit()
    con.close()


def _hit(source, title, raw=1.0):
    return SearchResult(query="q", title=title, snippet=f"snippet {title}",
                        source=source, type="book" if source == "books"
                        else "memory", raw_score=raw,
                        timestamp=None, source_id=f"{source}:{title}")


class _FakeAdapter:
    def __init__(self, name, hits=None, note=None):
        self.name = name
        self.result_type = "memory"
        self.description = "fake"
        self._hits = hits or []
        self._note = note

    def probe(self):
        return self._note

    def search(self, query, *, limit, since=None, before=None):
        return list(self._hits[:limit])


class _HangingAdapter(_FakeAdapter):
    """Blocks in probe() / search() until the test releases it.

    The release event is set by the test's cleanup, so no worker
    thread outlives the test run (a forever-blocked worker would hang
    interpreter exit: ThreadPoolExecutor joins its threads there).
    """

    def __init__(self, name, *, hang_in="probe", hits=None):
        super().__init__(name, hits=hits)
        self.release = threading.Event()
        self._hang_in = hang_in

    def probe(self):
        if self._hang_in == "probe":
            self.release.wait()
        return None

    def search(self, query, *, limit, since=None, before=None):
        if self._hang_in == "search":
            self.release.wait()
        return list(self._hits[:limit])


def _adapters(**over):
    ads = {name: _FakeAdapter(name) for name in valid_source_names()}
    ads.update(over)
    return ads


# ── BooksAdapter.probe fast path ──────────────────────────────────────

class BooksProbeFastPathTests(unittest.TestCase):
    def test_probe_does_not_construct_library(self):
        """The probe answers from the DB file alone: constructing a
        Library is the slow step (a full seam import when a document
        store is ambient), and the probe must never pay it."""
        tmp = _tmpdir()
        db = Path(tmp) / "library" / "library.db"
        _library_db(db)
        ad = BooksAdapter(_ctx(tmp))  # no injected library
        with mock.patch.object(
                _sources_mod, "Library",
                side_effect=AssertionError("probe constructed Library")):
            self.assertIsNone(ad.probe())

    def test_probe_locked_db_fails_fast(self):
        """A library DB held by another writer fails fast into
        'unavailable' (explicit 2s connect timeout) instead of
        stalling the probe phase."""
        tmp = _tmpdir()
        db = Path(tmp) / "library" / "library.db"
        _library_db(db)
        locker = sqlite3.connect(str(db))
        locker.execute("BEGIN EXCLUSIVE")
        locker.execute("INSERT INTO passages VALUES (?,?,?,?,?)",
                       ("slug2", "Other", "Ch 1", 1, "more text"))
        try:
            ad = BooksAdapter(_ctx(tmp))
            t0 = time.perf_counter()
            note = ad.probe()
            elapsed = time.perf_counter() - t0
        finally:
            locker.rollback()
            locker.close()
        self.assertIn("unavailable", note or "")
        self.assertLess(elapsed, 5.0, "locked probe must fail fast, "
                        f"took {elapsed:.1f}s")

    def test_probe_empty_db_still_reports_empty(self):
        tmp = _tmpdir()
        db = Path(tmp) / "library" / "library.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(db))
        con.execute("CREATE TABLE passages (x TEXT)")
        con.commit()
        con.close()
        ad = BooksAdapter(_ctx(tmp))
        self.assertIn("empty", ad.probe() or "")


# ── federated deadlines ───────────────────────────────────────────────

class FederatedDeadlineTests(unittest.TestCase):
    def test_hanging_probe_skipped_within_deadline(self):
        """The 2026-11-11 kill, replayed small: a probe that never
        returns is skipped with a timeout note and the healthy sources
        still contribute — the call returns in well under its deadline."""
        hanging = _HangingAdapter("books", hang_in="probe")
        self.addCleanup(hanging.release.set)
        good = _FakeAdapter("memory", hits=[_hit("memory", "m1")])
        t0 = time.perf_counter()
        resp = federated_search(
            "q", adapters=_adapters(memory=good, books=hanging),
            deadline_s=5.0, probe_timeout_s=0.2)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 4.0, f"call took {elapsed:.1f}s")
        self.assertIn("books", resp.sources_skipped)
        self.assertIn("probe timed out",
                      resp.sources_skipped["books"])
        self.assertIn("memory", resp.sources_searched)
        self.assertEqual([h.title for h in resp.hits], ["m1"])

    def test_probe_that_raises_is_skipped_not_fatal(self):
        class Boom(_FakeAdapter):
            def probe(self):
                raise RuntimeError("keeper exploded")

        bad = Boom("wisdom")
        good = _FakeAdapter("memory", hits=[_hit("memory", "m1")])
        resp = federated_search(
            "q", adapters=_adapters(memory=good, wisdom=bad))
        self.assertIn("wisdom", resp.sources_skipped)
        self.assertIn("keeper exploded",
                      resp.sources_skipped["wisdom"])
        self.assertEqual([h.title for h in resp.hits], ["m1"])

    def test_hanging_search_returns_partial_within_deadline(self):
        """A search source that never returns is skipped with a timeout
        note; the partial results still come back inside the deadline.
        Both the parallel and the sequential collection paths."""
        for parallel in (True, False):
            with self.subTest(parallel=parallel):
                hanging = _HangingAdapter(
                    "books", hang_in="search",
                    hits=[_hit("books", "b1")])
                self.addCleanup(hanging.release.set)
                good = _FakeAdapter("memory", hits=[_hit("memory", "m1")])
                t0 = time.perf_counter()
                resp = federated_search(
                    "q", adapters=_adapters(memory=good, books=hanging),
                    deadline_s=1.0, parallel=parallel)
                elapsed = time.perf_counter() - t0
                self.assertLess(elapsed, 2.5,
                                f"call took {elapsed:.1f}s")
                self.assertIn("books", resp.sources_skipped)
                self.assertIn("timed out",
                              resp.sources_skipped["books"])
                self.assertEqual([h.title for h in resp.hits], ["m1"])

    def test_bad_deadline_rejected(self):
        with self.assertRaises(SearchError):
            federated_search("q", adapters=_adapters(), deadline_s=0)
        with self.assertRaises(SearchError):
            federated_search("q", adapters=_adapters(), probe_timeout_s=0)


# ── _web_hunt deadline ────────────────────────────────────────────────

class WebHuntDeadlineTests(unittest.TestCase):
    def test_web_hunt_passes_default_deadline(self):
        """_web_hunt bounds every hunt at 120s by default: two hunts
        cap at 240s inside the motivation job's 900s wall."""
        seen = {}

        def fake_search(query, **kw):
            seen.update(kw)
            return SearchResponse(query=query, hits=[])

        with mock.patch(
                "nomorals.search.federated.federated_search",
                side_effect=fake_search):
            out = teachings._web_hunt(SimpleNamespace(), "some theme")
        self.assertEqual(out, [])
        self.assertEqual(seen.get("deadline_s"), 120.0)

    def test_web_hunt_hanging_adapters_returns_within_deadline(self):
        """End to end through the real federated_search: a hanging
        books probe degrades (no web results from it) and the hunt
        still returns the healthy source's finds inside its deadline."""
        hanging = _HangingAdapter("books", hang_in="probe")
        self.addCleanup(hanging.release.set)
        good = _FakeAdapter("memory", hits=[_hit("memory", "m1")])
        built = _adapters(memory=good, books=hanging)
        with mock.patch(
                "nomorals.search.federated.build_adapters",
                return_value=built):
            t0 = time.perf_counter()
            out = teachings._web_hunt(SimpleNamespace(), "some theme",
                                      deadline_s=2.0)
            elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.5, f"hunt took {elapsed:.1f}s")
        self.assertEqual([r["title"] for r in out], ["m1"])


if __name__ == "__main__":
    unittest.main()
