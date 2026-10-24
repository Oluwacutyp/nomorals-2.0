"""Museum + Library next level: URL/feed ingest, digests, hybrid
retrieval, source health/shedding, discovery, generalized museum,
learn-journal bridge, vector backfill.

Zero network: every fetch/probe path takes an injected fetcher.
Tmp dirs everywhere; the Library context mirrors test_books_library.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.books.library import Library, LibraryError
from nomorals.characters.character import Character
from nomorals.characters.learn import LearnStore
from nomorals.characters.store import CharacterStore
from nomorals.memory.embeddings import Embedder
from nomorals.museum import Museum


def _ctx(tmp: str):
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings)


BOOK_TEXT = """# The Dragon Codex

by Ember Wright

# Chapter One — Awakening

Dragons fly over the valley at dawn. The wizard watched them from the
tower, counting seven bright wings against the rising sun. More about
dragons and valleys and wizards fills this chapter with searchable
words about fire and flight and ancient bargains.

# Chapter Two — The Bargain

The wizard made a bargain with the eldest dragon. Fire for wisdom,
scale for spell — an old trade, older than the mountains themselves.
The bargain chapter mentions gold and prices and mountain passes.
"""


def _write(tmp: str, name: str, text: str) -> Path:
    p = Path(tmp) / name
    p.write_text(text, encoding="utf-8")
    return p


def _html_doc() -> bytes:
    body = ("Dragons fly over the valley at dawn. " * 20
            + "The wizard counted seven bright wings. " * 20)
    return (f"<html><head><title>Web Dragons</title></head><body>"
            f"<h1>Chapter One</h1><p>{body}</p>"
            f"<h1>Chapter Two</h1><p>{body}</p></body></html>").encode("utf-8")


def _rss(items) -> bytes:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<rss version="2.0"><channel><title>Test Feed</title>']
    for title, link, desc in items:
        parts.append(f"<item><title>{title}</title><link>{link}</link>"
                     f"<description>{desc}</description></item>")
    parts.append("</channel></rss>")
    return "".join(parts).encode("utf-8")


FEED_V1 = _rss([
    ("Feed Item One", "https://example.com/one",
     "First feed story about dragons and valleys and wizards."),
    ("Feed Item Two", "https://example.com/two",
     "Second feed story about gold prices and mountain passes."),
])
FEED_V2 = _rss([
    ("Feed Item One", "https://example.com/one",
     "First feed story about dragons and valleys and wizards."),
    ("Feed Item Two", "https://example.com/two",
     "Second feed story about gold prices and mountain passes."),
    ("Feed Item Three", "https://example.com/three",
     "Third feed story about rivers and bridges and new roads."),
])
ATOM = (b'<?xml version="1.0" encoding="utf-8"?>'
        b'<feed xmlns="http://www.w3.org/2005/Atom">'
        b'<title>Atom Feed</title>'
        b'<entry><title>Atom Entry</title>'
        b'<link href="https://example.com/atom1"/>'
        b'<summary>Atom summary about dragons and valleys.</summary>'
        b'</entry></feed>')


class _RaisingEmbedder:
    """Embedder that always fails — the degradation probe."""
    provider = "raising"
    dimensions = 512
    stats = {"fallbacks": 0}

    def embed(self, text):
        raise RuntimeError("no embedding backend")

    def embed_many(self, texts):
        raise RuntimeError("no embedding backend")


class TestUrlIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))

    def _fetcher(self, data, ctype):
        def fetch(url, timeout):
            return data, ctype
        return fetch

    def test_url_text_ingest_same_shape_as_file(self):
        res = self.lib.ingest_url(
            "https://example.com/dragons.txt",
            fetcher=self._fetcher(BOOK_TEXT.encode("utf-8"), "text/plain"))
        self.assertTrue(res.slug)
        self.assertTrue(res.source_id)
        self.assertGreaterEqual(res.chapters, 2)
        sources = self.lib.list_sources(kind="url")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["origin"], "https://example.com/dragons.txt")
        self.assertEqual(sources[0]["health"], "ok")
        hits = self.lib.search("dragons valley")
        self.assertTrue(hits)
        self.assertEqual(hits[0].book_slug, res.slug)

    def test_url_html_ingest(self):
        res = self.lib.ingest_url(
            "https://example.com/page",
            fetcher=self._fetcher(_html_doc(), "text/html"))
        self.assertGreaterEqual(res.chapters, 1)
        self.assertTrue(self.lib.search("dragons"))

    def test_url_fetch_failure_leaves_nothing(self):
        def boom(url, timeout):
            raise LibraryError("fetch https://example.com/x failed: boom")
        with self.assertRaises(LibraryError):
            self.lib.ingest_url("https://example.com/x", fetcher=boom)
        self.assertEqual(self.lib.list_books(), [])
        self.assertEqual(self.lib.list_sources(), [])

    def test_url_tiny_document_rejected(self):
        with self.assertRaises(LibraryError):
            self.lib.ingest_url(
                "https://example.com/tiny",
                fetcher=self._fetcher(b"too short", "text/plain"))
        self.assertEqual(self.lib.list_books(), [])

    def test_url_non_http_rejected(self):
        with self.assertRaises(LibraryError):
            self.lib.ingest_url("ftp://example.com/book.txt",
                                fetcher=self._fetcher(b"x" * 300,
                                                      "text/plain"))


class TestFeedIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))

    def test_feed_ingest_and_upsert(self):
        res = self.lib.ingest_feed(
            "https://example.com/feed.xml",
            fetcher=lambda url, timeout: (FEED_V1, "application/rss+xml"))
        self.assertEqual(res.chapters, 2)
        sources = self.lib.list_sources(kind="feed")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["origin"], "https://example.com/feed.xml")
        meta = self.lib.load(res.slug)
        links = [c.get("link", "") for c in meta["chapters"]]
        self.assertIn("https://example.com/one", links)
        res2 = self.lib.ingest_feed(
            "https://example.com/feed.xml",
            fetcher=lambda url, timeout: (FEED_V2, "application/rss+xml"))
        self.assertEqual(res2.slug, res.slug)
        self.assertEqual(res2.chapters, 3)
        self.assertIn("upsert", res2.note)
        self.assertEqual(len(self.lib.load(res.slug)["chapters"]), 3)

    def test_feed_bad_xml_rejected(self):
        with self.assertRaises(LibraryError):
            self.lib.ingest_feed(
                "https://example.com/feed.xml",
                fetcher=lambda url, timeout: (b"not xml at all",
                                              "application/rss+xml"))
        self.assertEqual(self.lib.list_books(), [])

    def test_atom_feed(self):
        res = self.lib.ingest_feed(
            "https://example.com/atom.xml",
            fetcher=lambda url, timeout: (ATOM, "application/atom+xml"))
        self.assertEqual(res.chapters, 1)
        links = [c.get("link", "") for c in self.lib.load(res.slug)["chapters"]]
        self.assertIn("https://example.com/atom1", links)


class TestDigests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))
        self.book = self.lib.ingest(_write(self.tmp, "book.txt", BOOK_TEXT))
        self.source_id = self.book.source_id

    def test_create_revise_revisions(self):
        d = self.lib.create_digest(self.source_id, "The sky is blue today",
                                   confidence=0.8)
        self.assertEqual(d["claim"], "The sky is blue today")
        self.assertEqual(len(self.lib.digest_revisions(d["record_id"])), 1)
        r = self.lib.revise_digest(d["record_id"],
                                   claim="The sky is grey today",
                                   note="weather changed")
        self.assertEqual(r["claim"], "The sky is grey today")
        revs = self.lib.digest_revisions(d["record_id"])
        self.assertEqual(len(revs), 2)
        self.assertEqual(revs[0]["claim"], "The sky is blue today")
        self.assertEqual(revs[1]["claim"], "The sky is grey today")
        self.assertEqual(revs[1]["note"], "weather changed")

    def test_create_validation(self):
        with self.assertRaises(LibraryError):
            self.lib.create_digest(self.source_id, "   ")
        with self.assertRaises(LibraryError):
            self.lib.create_digest(self.source_id, "A claim", confidence=1.5)
        with self.assertRaises(LibraryError):
            self.lib.create_digest("no-such-source", "A claim")
        d = self.lib.create_digest(self.source_id, "A claim")
        with self.assertRaises(LibraryError):
            self.lib.revise_digest(d["record_id"])
        with self.assertRaises(LibraryError):
            self.lib.revise_digest("no-such-digest", claim="x")

    def test_value_change_flag(self):
        self.lib.create_digest(self.source_id,
                               "The price of gold is 100 dollars")
        d2 = self.lib.create_digest(self.source_id,
                                    "The price of gold is 200 dollars")
        flags = self.lib.digest_flags(d2["record_id"])
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["strategy"], "value_change")
        self.assertAlmostEqual(flags[0]["confidence"], 0.75)

    def test_negation_flag_and_no_flag(self):
        self.lib.create_digest(self.source_id, "Gold will rise tomorrow")
        d2 = self.lib.create_digest(self.source_id,
                                    "Gold will not rise tomorrow")
        flags = self.lib.digest_flags(d2["record_id"])
        self.assertTrue(any(f["strategy"] == "negation_flip"
                            for f in flags))
        d3 = self.lib.create_digest(self.source_id,
                                    "The weather in Lagos is sunny")
        self.assertEqual(self.lib.digest_flags(d3["record_id"]), [])

    def test_revise_recomputes_flags(self):
        self.lib.create_digest(self.source_id,
                               "The price of gold is 100 dollars")
        d2 = self.lib.create_digest(self.source_id,
                                    "The price of gold is 200 dollars")
        self.assertEqual(len(self.lib.digest_flags(d2["record_id"])), 1)
        self.lib.revise_digest(d2["record_id"],
                               claim="The price of gold is 100 dollars")
        # Now identical to the older claim: no value change anymore.
        self.assertEqual(self.lib.digest_flags(d2["record_id"]), [])

    def test_scope_filtering(self):
        self.lib.create_digest(self.source_id, "Global claim here",
                               scope="")
        self.lib.create_digest(self.source_id, "Zara claim here",
                               scope="character:zara")
        self.lib.create_digest(self.source_id, "Other claim here",
                               scope="character:other")
        scoped = self.lib.list_digests(scope="character:zara")
        claims = [d["claim"] for d in scoped]
        self.assertIn("Global claim here", claims)
        self.assertIn("Zara claim here", claims)
        self.assertNotIn("Other claim here", claims)


class TestHybrid(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))
        self.book = self.lib.ingest(_write(self.tmp, "book.txt", BOOK_TEXT))
        self.lib.create_digest(self.book.source_id,
                               "The price of gold is 100 dollars",
                               confidence=0.9)

    def test_hybrid_returns_both_kinds(self):
        res = self.lib.search_hybrid("price of gold", top=5)
        kinds = [h["kind"] for h in res["hits"]]
        self.assertIn("digest", kinds)
        self.assertIn("passage", kinds)
        digest = next(h for h in res["hits"] if h["kind"] == "digest")
        self.assertEqual(digest["text"], "The price of gold is 100 dollars")
        self.assertGreater(digest["score"], 0)
        self.assertIn("keyword", digest["lanes"])

    def test_hybrid_vector_lane_present(self):
        res = self.lib.search_hybrid("dragons valley wizard", top=5)
        self.assertGreater(res["lanes"]["vector"], 0)
        self.assertEqual(res["lanes"]["degraded"], [])
        self.assertTrue(any("vector" in h["lanes"] for h in res["hits"]))

    def test_hybrid_empty_query(self):
        res = self.lib.search_hybrid("   ")
        self.assertEqual(res["hits"], [])

    def test_hybrid_degrades_when_embedder_raises(self):
        self.lib.embedder = _RaisingEmbedder()
        res = self.lib.search_hybrid("dragons valley", top=5)
        self.assertTrue(res["hits"])  # keyword lane still serves
        self.assertTrue(res["lanes"]["degraded"])
        self.assertTrue(all(h["kind"] == "passage" for h in res["hits"]))

    def test_hybrid_dims_mismatch_degrades_not_crashes(self):
        self.lib.embedder = Embedder(provider="hashing", dimensions=64)
        res = self.lib.search_hybrid("dragons valley", top=5)
        self.assertTrue(res["hits"])
        self.assertTrue(any("dimensions" in d for d in res["lanes"]["degraded"]))

    def test_dims_mismatch_reported_on_write(self):
        self.lib.embedder = Embedder(provider="hashing", dimensions=64)
        d = self.lib.create_digest(self.book.source_id,
                                   "A mismatched digest claim")
        self.assertIn("vector_error", d)
        self.assertIn("dimensions", d["vector_error"])
        # The digest itself still landed — vectors are an accelerator.
        self.assertEqual(self.lib.get_digest(d["record_id"])["claim"],
                         "A mismatched digest claim")


class TestHealthShedding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))
        self.res = self.lib.ingest_url(
            "https://example.com/doc.txt",
            fetcher=lambda url, timeout: (BOOK_TEXT.encode("utf-8"),
                                          "text/plain"))
        self.lib.create_digest(self.res.source_id,
                               "The price of gold is 100 dollars")

    def _failing_probe(self, url, timeout):
        raise LibraryError("boom")

    def test_check_ok_resets_failures(self):
        out = self.lib.check_sources(
            fetcher=lambda url, timeout: 200)
        self.assertEqual(out["ok"], 1)
        src = self.lib.list_sources()[0]
        self.assertEqual(src["health"], "ok")
        self.assertEqual(src["failures"], 0)

    def test_three_failures_dead_then_shed(self):
        for _ in range(2):
            out = self.lib.check_sources(fetcher=self._failing_probe)
        self.assertEqual(out["failing"], 1)
        out = self.lib.check_sources(fetcher=self._failing_probe)
        self.assertEqual(out["dead"], 1)
        shed = self.lib.shed_dead()
        self.assertEqual(len(shed["archived_sources"]), 1)
        self.assertEqual(shed["archived_digests"], 1)
        src = self.lib.list_sources()[0]
        self.assertEqual(src["status"], "archived")
        self.assertEqual(self.lib.list_digests(status="active"), [])
        self.assertEqual(len(self.lib.list_digests(status="archived")), 1)

    def test_archived_excluded_from_hybrid_by_default(self):
        for _ in range(3):
            self.lib.check_sources(fetcher=self._failing_probe)
        self.lib.shed_dead()
        default = self.lib.search_hybrid("price of gold", top=10)
        self.assertFalse(any(h["kind"] == "digest"
                             for h in default["hits"]))
        self.assertFalse(any(h["slug"] == self.res.slug
                             for h in default["hits"]))
        included = self.lib.search_hybrid("price of gold", top=10,
                                          include_archived=True)
        self.assertTrue(any(h["kind"] == "digest"
                            for h in included["hits"]))


class TestDiscoveryAndIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))

    def test_discover_excludes_own_and_known(self):
        text = BOOK_TEXT + "\n\nSee https://example.com/related and " \
            "https://example.com/other for more details here.\n"
        res = self.lib.ingest(_write(self.tmp, "book.txt", text))
        # A known origin must not be proposed again. (Different title
        # so the URL book lands its own slug instead of overwriting
        # the file book — same-slug re-ingest replaces, by design.)
        self.lib.ingest_url(
            "https://example.com/other", title="Other Document",
            fetcher=lambda url, timeout: (BOOK_TEXT.encode("utf-8"),
                                          "text/plain"))
        cands = self.lib.discover()
        urls = [c["url"] for c in cands]
        self.assertIn("https://example.com/related", urls)
        self.assertNotIn("https://example.com/other", urls)

    def test_discover_feed_item_links(self):
        res = self.lib.ingest_feed(
            "https://example.com/feed.xml",
            fetcher=lambda url, timeout: (FEED_V1, "application/rss+xml"))
        cands = self.lib.discover(res.source_id)
        urls = [c["url"] for c in cands]
        self.assertIn("https://example.com/one", urls)
        self.assertIn("https://example.com/two", urls)

    def test_discover_unknown_source_rejected(self):
        with self.assertRaises(LibraryError):
            self.lib.discover("no-such-source")

    def test_incremental_index_survives_ingest_and_drop(self):
        a = self.lib.ingest(_write(self.tmp, "a.txt", BOOK_TEXT))
        b_text = BOOK_TEXT.replace("Dragon", "Unicorn").replace("dragon", "unicorn")
        b = self.lib.ingest(_write(self.tmp, "b.txt", b_text))
        c_text = BOOK_TEXT.replace("Dragon", "Phoenix").replace("dragon", "phoenix")
        c = self.lib.ingest(_write(self.tmp, "c.txt", c_text))
        self.assertTrue(self.lib.search("dragons valley", book=a.slug))
        self.assertTrue(self.lib.search("unicorn", book=b.slug))
        self.assertTrue(self.lib.search("phoenix", book=c.slug))
        self.lib.drop(b.slug)
        self.assertTrue(self.lib.search("dragons valley", book=a.slug))
        self.assertTrue(self.lib.search("phoenix", book=c.slug))
        self.assertFalse(self.lib.search("unicorn"))


class TestMuseum(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))
        self.book = self.lib.ingest(_write(self.tmp, "book.txt", BOOK_TEXT))
        self.digest = self.lib.create_digest(
            self.book.source_id, "The price of gold is 100 dollars")
        chars_dir = str(Path(self.tmp) / "characters")
        self.cs = CharacterStore(chars_dir)
        self.cs.save(Character(id="zara", name="Zara"))
        self.ls = LearnStore(chars_dir)
        entry = self.ls.journal("zara", {
            "line": "Zara loves the morning light",
            "kind": "utterance", "context": "test",
            "candidates": [{"text": "Zara loves the morning light",
                          "features": {"consistency": 0.8,
                                       "catchphrase": 0.0,
                                       "length_fit": 0.5,
                                       "punct_fit": 0.5,
                                       "novelty": 0.5},
                          "score": 0.6}],
            "chosen": 0})
        self.ls.grade("zara", entry, 0.8, "owner")
        self.apk_root = str(Path(self.tmp) / "apk")
        os.makedirs(self.apk_root, exist_ok=True)
        self.museum = Museum(
            apk_root=self.apk_root, library=self.lib,
            char_store=self.cs, learn_store=self.ls,
            learn_dir=str(Path(chars_dir) / "learn"))

    def test_census_shelves(self):
        res = self.museum.census()
        self.assertEqual(res["by_shelf"].get("library-books"), 1)
        self.assertEqual(res["by_shelf"].get("library-digests"), 1)
        self.assertGreaterEqual(res["by_shelf"].get("library-sources"), 1)
        self.assertEqual(res["by_shelf"].get("characters-learned"), 1)
        self.assertEqual(res["errors"], {})

    def test_digest_cards_never_sendable(self):
        res = self.museum.census()
        digests = [c for c in res["artifacts"]
                   if c["shelf"] == "library-digests"]
        self.assertEqual(len(digests), 1)
        self.assertIs(digests[0]["sendable"], False)
        books = [c for c in res["artifacts"]
                 if c["shelf"] == "library-books"]
        self.assertIs(books[0]["sendable"], True)

    def test_find_and_render(self):
        found = self.museum.find(f"library-book:{self.book.slug}")
        self.assertIsNotNone(found)
        prefixed = self.museum.find("library-book:")
        self.assertIsNotNone(prefixed)
        self.assertIsNone(self.museum.find("no-such-thing"))
        self.assertIsNone(self.museum.find(""))
        text = self.museum.render()
        self.assertIn("Zara", text)
        self.assertIn("Dragon", text)

    def test_raising_exhibitor_named_not_fatal(self):
        self.museum.register_exhibitor("boom", lambda: 1 / 0)
        res = self.museum.census()
        self.assertIn("boom", res["errors"])
        self.assertGreater(res["count"], 0)

    def test_character_without_learn_doc_skipped(self):
        self.cs.save(Character(id="nolearn", name="NoLearn"))
        res = self.museum.census()
        learned = [c for c in res["artifacts"]
                   if c["shelf"] == "characters-learned"]
        self.assertEqual(len(learned), 1)


class TestLearnBridge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))
        self.chars_dir = str(Path(self.tmp) / "characters")
        self.ls = LearnStore(self.chars_dir)

        def entry(line):
            return {
                "line": line, "kind": "utterance", "context": "test",
                "candidates": [{"text": line,
                              "features": {"consistency": 0.8,
                                           "catchphrase": 0.0,
                                           "length_fit": 0.5,
                                           "punct_fit": 0.5,
                                           "novelty": 0.5},
                              "score": 0.6}],
                "chosen": 0}
        e1 = self.ls.journal("zara", entry("Zara loves the morning light"))
        self.ls.grade("zara", e1, 0.8, "owner")
        e2 = self.ls.journal("zara", entry("Zara hates loud mornings"))
        self.ls.grade("zara", e2, -0.5, "owner")
        self.ls.journal("zara", entry("Zara waits for no one"))

    def test_bridge_ingests_positive_and_dedupes(self):
        out = self.lib.ingest_learn_journal("zara",
                                            data_dir=self.chars_dir)
        self.assertEqual(out["ingested"], 1)
        self.assertEqual(out["skipped"], 1)
        self.assertEqual(out["scope"], "character:zara")
        digests = self.lib.list_digests(scope="character:zara")
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0]["claim"], "Zara loves the morning light")
        self.assertAlmostEqual(digests[0]["confidence"], 0.9)
        out2 = self.lib.ingest_learn_journal("zara",
                                             data_dir=self.chars_dir)
        self.assertEqual(out2["ingested"], 0)
        out3 = self.lib.ingest_learn_journal("zara",
                                             data_dir=self.chars_dir,
                                             min_grade=-1.0)
        self.assertEqual(out3["ingested"], 1)
        self.assertEqual(len(self.lib.list_digests(scope="character:zara")), 2)


class TestVectors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = Library(_ctx(self.tmp))
        self.book = self.lib.ingest(_write(self.tmp, "book.txt", BOOK_TEXT))
        self.digest = self.lib.create_digest(
            self.book.source_id, "The price of gold is 100 dollars")

    def _wipe_vectors(self):
        con = sqlite3.connect(self.lib.db_path())
        con.execute("DELETE FROM embeddings")
        con.commit()
        con.close()

    def test_backfill_restores_vectors(self):
        before = self.lib.vector_status()
        self.assertGreater(before["vector_count"], 0)
        self.assertEqual(before["unvectorized_digests"], 0)
        self._wipe_vectors()
        wiped = self.lib.vector_status()
        self.assertEqual(wiped["vector_count"], 0)
        self.assertEqual(wiped["unvectorized_digests"], 1)
        out = self.lib.backfill_vectors()
        self.assertEqual(out["errors"], [])
        self.assertGreater(out["passages"], 0)
        self.assertEqual(out["digests"], 1)
        after = self.lib.vector_status()
        self.assertGreater(after["vector_count"], 0)
        self.assertEqual(after["unvectorized_digests"], 0)
        res = self.lib.search_hybrid("dragons valley", top=5)
        self.assertGreater(res["lanes"]["vector"], 0)

    def test_vector_status_backend(self):
        status = self.lib.vector_status()
        self.assertEqual(status["backend"], "legacy")
        self.assertEqual(status["embedder_provider"], "hashing")
        self.assertEqual(status["pinned_dims"], 512)


class TestToolRegistration(unittest.TestCase):
    def test_new_actions_registered(self):
        from nomorals.books.tools import register as books_register

        seen = {}

        class FakeRegistry:
            def __init__(self, context):
                self.context = context

            def register(self, name, description="", capability=None,
                         parameters=None):
                def deco(fn):
                    seen[name] = fn
                    return fn
                return deco

        books_register(FakeRegistry(_ctx(tempfile.mkdtemp())))
        for name in ("library_ingest_url", "library_ingest_feed",
                     "library_sources", "library_digest",
                     "library_hybrid_search", "library_learn_ingest",
                     "library_vectors", "museum_census"):
            self.assertIn(name, seen)


if __name__ == "__main__":
    unittest.main()
