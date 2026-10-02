"""Phase 2 tests: ArchiveIngestor — fetch, cache, parse, ingest, search.

All HTTP is mocked: the fake replaces ``ingestor._http`` (the
``HttpClient``) and robots checks are bypassed with a permissive stub.
No test touches the network.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.core.errors import ProviderError, TimeoutError_
from nomorals.core.http import HttpResponse
from nomorals.wisdom import ArchiveIngestor, IngestError, ManifestEntry
from nomorals.wisdom import ingestor as ing_mod


TEXT = (
    "# Sayings of the Seeker\n\n"
    + "The kingdom of heaven is within you and all around you. " * 60
    + "\n\n# On the Inner Light\n\n"
    + "Blessed are the seekers of the inner light and the quiet mind. " * 40
)
BYTES = TEXT.encode("utf-8")
URL = "https://example.com/texts/seeker.txt"


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-ingestor-test-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings), tmp


def _entry(slug="seeker-sayings", **kw):
    d = {
        "slug": slug,
        "title": "Sayings of the Seeker",
        "tradition": "esoteric",
        "canon_status": "esoteric",
        "translator": "",
        "source_url": URL,
        "license": "public-domain",
    }
    d.update(kw)
    return ManifestEntry.from_dict(d)


class _FakeHttp:
    """Stand-in for HttpClient: handler(url) -> HttpResponse or raises."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        return self.handler(url)


def _resp(body: bytes, status: int = 200, url: str = URL) -> HttpResponse:
    return HttpResponse(status=status, body=body, headers={}, url=url)


def _make_ingestor(fake_http=None, robots_ok=True):
    ctx, tmp = _ctx()
    ing = ArchiveIngestor(ctx)
    ing._http = fake_http or _FakeHttp(lambda url: _resp(BYTES, url=url))
    ing._robots = SimpleNamespace(
        allowed=lambda *a, **k: robots_ok)
    return ing, tmp


class FetchBytesTests(unittest.TestCase):
    def test_fetch_returns_bytes_and_caches_blob(self):
        ing, tmp = _make_ingestor()
        data = ing.fetch_bytes(URL)
        self.assertEqual(data, BYTES)
        sha = hashlib.sha256(BYTES).hexdigest()
        self.assertTrue((ing._blobs / sha).is_file())
        index = json.loads((ing._blobs / "index.json").read_text())
        self.assertEqual(index[URL]["sha"], sha)

    def test_fetch_second_call_served_from_cache(self):
        ing, tmp = _make_ingestor()
        first = ing.fetch_bytes(URL)
        calls_before = len(ing._http.calls)
        ing._http = _FakeHttp(
            lambda url: (_ for _ in ()).throw(
                AssertionError("network must not be hit")))
        second = ing.fetch_bytes(URL)
        self.assertEqual(first, second)
        self.assertEqual(calls_before, 1)

    def test_http_error_fails_fast_no_retry(self):
        fake = _FakeHttp(
            lambda url: (_ for _ in ()).throw(
                ProviderError("404 not found", retryable=False)))
        ing, tmp = _make_ingestor(fake)
        with self.assertRaises(IngestError):
            ing.fetch_bytes(URL)
        self.assertEqual(len(fake.calls), 1)

    def test_transient_failure_retried_then_succeeds(self):
        calls = {"n": 0}

        def handler(url):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError_("timed out")
            return _resp(BYTES, url=url)

        ing, tmp = _make_ingestor(_FakeHttp(handler))
        with mock.patch("time.sleep") as slp:
            data = ing.fetch_bytes(URL)
        self.assertEqual(data, BYTES)
        self.assertEqual(calls["n"], 2)
        self.assertTrue(slp.called)

    def test_persistent_transient_failure_exhausts_retries(self):
        fake = _FakeHttp(
            lambda url: (_ for _ in ()).throw(
                ProviderError("503 upstream", retryable=True)))
        ing, tmp = _make_ingestor(fake)
        with mock.patch("time.sleep"):
            with self.assertRaises(IngestError) as cm:
                ing.fetch_bytes(URL)
        self.assertEqual(len(fake.calls), ing_mod._FETCH_ATTEMPTS)
        self.assertIn("attempts", str(cm.exception))

    def test_empty_body_raises_never_silent(self):
        ing, tmp = _make_ingestor(
            _FakeHttp(lambda url: _resp(b"", url=url)))
        with self.assertRaises(IngestError):
            ing.fetch_bytes(URL)
        # nothing cached for a failed fetch
        index_path = ing._blobs / "index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            self.assertNotIn(URL, index)

    def test_robots_denied_raises_before_network(self):
        fake = _FakeHttp(lambda url: _resp(BYTES, url=url))
        ing, tmp = _make_ingestor(fake, robots_ok=False)
        with self.assertRaises(IngestError):
            ing.fetch_bytes(URL)
        self.assertEqual(len(fake.calls), 0)

    def test_non_http_url_rejected(self):
        ing, tmp = _make_ingestor()
        with self.assertRaises(IngestError):
            ing.fetch_bytes("ftp://example.com/x.txt")

    def test_fetch_entry_uses_source_url(self):
        ing, tmp = _make_ingestor()
        data = ing.fetch(_entry())
        self.assertEqual(data, BYTES)


class ParseTests(unittest.TestCase):
    def test_parse_txt_returns_full_text(self):
        ing, tmp = _make_ingestor()
        text = ing.parse(BYTES, "seeker.txt")
        self.assertIn("kingdom of heaven", text)
        self.assertIn("inner light", text)

    def test_parse_empty_bytes_raises(self):
        ing, tmp = _make_ingestor()
        with self.assertRaises(IngestError):
            ing.parse(b"", "seeker.txt")

    def test_parse_unparseable_raises_ingest_error(self):
        ing, tmp = _make_ingestor()
        with self.assertRaises(IngestError):
            ing.parse(b"\x00\x01\x02\x03garbage", "weird.zzz9")

    def test_parse_missing_filename_raises(self):
        ing, tmp = _make_ingestor()
        with self.assertRaises(IngestError):
            ing.parse(BYTES, "")


class IngestEntryTests(unittest.TestCase):
    def test_ingest_entry_new_text_registers_and_ingests(self):
        ing, tmp = _make_ingestor()
        entry = ing.ingest_entry(_entry())
        self.assertTrue(entry.ingested_at > 0)
        self.assertTrue(entry.sha256)
        stored = ing.corpus.get("seeker-sayings")
        self.assertEqual(stored.sha256, entry.sha256)
        self.assertEqual(ing.corpus.status()["ingested"], 1)
        answer = ing.corpus.ask("kingdom of heaven")
        self.assertTrue(answer.passages)

    def test_ingest_entry_idempotent(self):
        ing, tmp = _make_ingestor()
        # prime the blob cache with one real fetch
        ing.fetch_bytes(URL)
        digest = hashlib.sha256(BYTES).hexdigest()
        entry = _entry()
        entry.sha256 = digest
        entry.ingested_at = time.time()
        ing.corpus.register(entry)
        # network must not be touched again
        ing._http = _FakeHttp(
            lambda url: (_ for _ in ()).throw(
                AssertionError("re-fetch must be free")))
        with mock.patch.object(
                ing.corpus, "ingest_text",
                side_effect=AssertionError("must not re-ingest")):
            result = ing.ingest_entry(entry)
        self.assertIs(result, entry)
        self.assertEqual(result.sha256, digest)

    def test_ingest_text_passthrough(self):
        ing, tmp = _make_ingestor()
        ing.corpus.register(_entry())
        entry = ing.ingest_text("seeker-sayings", TEXT)
        self.assertTrue(entry.ingested_at > 0)
        self.assertEqual(ing.corpus.status()["ingested"], 1)


class SearchTests(unittest.TestCase):
    def _archive_json(self, docs):
        payload = {"response": {"docs": docs}}
        return _resp(json.dumps(payload).encode("utf-8"))

    def test_search_archive_returns_candidates(self):
        docs = [
            {"identifier": "gospelthomas",
             "title": "Gospel of Thomas",
             "description": "Sayings gospel."},
            {"identifier": "pistis-sophia",
             "title": "Pistis Sophia",
             "description": "Gnostic text."},
        ]
        ing, tmp = _make_ingestor(
            _FakeHttp(lambda url: self._archive_json(docs)))
        results = ing.search("thomas sayings", tradition="gnostic")
        self.assertEqual(len(results), 2)
        first = results[0]
        self.assertEqual(first["identifier"], "gospelthomas")
        self.assertEqual(first["title"], "Gospel of Thomas")
        self.assertEqual(
            first["url"], "https://archive.org/details/gospelthomas")
        self.assertEqual(first["description"], "Sayings gospel.")
        # archive.org must be hit, with the query encoded in the URL
        hit = ing._http.calls[0]
        self.assertIn("archive.org/advancedsearch.php", hit)
        self.assertIn("thomas", hit)

    def test_search_does_not_auto_ingest(self):
        docs = [{"identifier": "gospelthomas", "title": "Gospel of Thomas",
                 "description": ""}]
        ing, tmp = _make_ingestor(
            _FakeHttp(lambda url: self._archive_json(docs)))
        with mock.patch.object(
                ing.corpus, "ingest_text",
                side_effect=AssertionError("search must not ingest")):
            results = ing.search("thomas")
        self.assertEqual(len(results), 1)
        self.assertEqual(ing.corpus.status()["texts"], 0)

    def test_search_archive_down_falls_back_to_web(self):
        ddg = (
            '<html><body>'
            '<a class="result-link" '
            'href="https://example.com/thomas">Gospel of Thomas</a>'
            '<a class="result-link" '
            'href="https://sacred-texts.com/bib/thomas.htm">'
            'Thomas at sacred-texts</a>'
            '</body></html>'
        )

        def handler(url):
            if "archive.org/advancedsearch.php" in url:
                raise ProviderError("archive.org down", retryable=True)
            return _resp(ddg.encode("utf-8"), url=url)

        ing, tmp = _make_ingestor(_FakeHttp(handler))
        results = ing.search("thomas sayings")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["url"], "https://example.com/thomas")
        self.assertEqual(results[0]["title"], "Gospel of Thomas")
        self.assertEqual(results[1]["url"],
                         "https://sacred-texts.com/bib/thomas.htm")

    def test_search_both_backends_down_raises(self):
        ing, tmp = _make_ingestor(_FakeHttp(
            lambda url: (_ for _ in ()).throw(
                ProviderError("everything down", retryable=True))))
        with mock.patch("time.sleep"):
            with self.assertRaises(IngestError):
                ing.search("thomas")

    def test_search_empty_query_raises(self):
        ing, tmp = _make_ingestor()
        with self.assertRaises(IngestError):
            ing.search("  ")


class ExportTests(unittest.TestCase):
    def test_archive_ingestor_exported(self):
        import nomorals.wisdom as w
        self.assertIs(w.ArchiveIngestor, ArchiveIngestor)
        self.assertIn("ArchiveIngestor", w.__all__)
        self.assertIn("IngestError", w.__all__)


if __name__ == "__main__":
    unittest.main()
