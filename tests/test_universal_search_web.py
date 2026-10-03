"""Universal web-search backends: parsing, error paths, probing, config.

All HTTP is mocked at ``nomorals.search.web._http`` — no network in
these tests. Each backend is verified against its documented 2026
response shape.
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.search.errors import SearchError
from nomorals.search.model import SearchResult
from nomorals.search.web import (
    BraveWebSource,
    DdgsWebSource,
    ExaWebSource,
    SearXNGWebSource,
    SerperWebSource,
    TavilyWebSource,
    WebBackendError,
    web_backends_configured,
    web_source_names,
)

_ENV_KEYS = [
    "NM_SEARXNG_URL", "SEARXNG_URL",
    "NM_TAVILY_API_KEY", "TAVILY_API_KEY",
    "NM_SERPER_API_KEY", "SERPER_API_KEY",
    "NM_EXA_API_KEY", "EXA_API_KEY",
    "NM_BRAVE_SEARCH_API_KEY", "BRAVE_SEARCH_API_KEY",
    "NM_WEB_RERANK", "WEB_RERANK",
    "NM_WEB_TIMEOUT", "WEB_TIMEOUT",
    "NM_DDGS_BACKEND", "DDGS_BACKEND",
]


class _EnvCleaner:
    def __enter__(self):
        self._saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        return False


def _resp(payload, status=200):
    body = json.dumps(payload).encode("utf-8") if isinstance(payload, dict) else payload
    return (status, body)


class WebSourceNamesTests(unittest.TestCase):
    def test_priority_order_free_first(self):
        names = web_source_names()
        self.assertEqual(names[0], "web_searxng")   # keyless metasearch: primary
        self.assertEqual(names[1], "web_ddgs")      # keyless fallback
        self.assertEqual(names[-1], "web_brave")    # metered since 2026-02: last

    def test_configured_reports_all_backends(self):
        with _EnvCleaner():
            cfg = web_backends_configured()
        self.assertEqual(set(cfg), set(web_source_names()))
        # nothing configured in a clean env
        self.assertFalse(any(cfg.values()))

    def test_env_prefix_precedence(self):
        with _EnvCleaner():
            os.environ["TAVILY_API_KEY"] = "bare-key"
            os.environ["NM_TAVILY_API_KEY"] = "nm-key"
            self.assertIsNone(TavilyWebSource().probe())
            # prove the NM_ value won: wrong bare key alone would differ —
            # instead check the header the adapter actually sends
            with mock.patch(
                "nomorals.search.web._http", return_value=_resp({"results": []})
            ) as m:
                TavilyWebSource().search("q", limit=3)
            _args, kwargs = m.call_args
            self.assertIn("nm-key", kwargs["headers"]["Authorization"])
            self.assertNotIn("bare-key", kwargs["headers"]["Authorization"])


class SearXNGTests(unittest.TestCase):
    def setUp(self):
        self._cleaner = _EnvCleaner()
        self._cleaner.__enter__()
        os.environ["NM_SEARXNG_URL"] = "https://searx.example.com"

    def tearDown(self):
        self._cleaner.__exit__()

    def test_probe_needs_instance(self):
        self.assertIsNone(SearXNGWebSource().probe())
        del os.environ["NM_SEARXNG_URL"]
        note = SearXNGWebSource().probe()
        self.assertIn("NM_SEARXNG_URL", note)

    def test_parses_results(self):
        payload = {
            "query": "python",
            "number_of_results": 2,
            "results": [
                {"url": "https://a.example/x", "title": "A title",
                 "content": "snippet about python here", "engine": "google",
                 "score": 42.5, "publishedDate": "2026-09-01T00:00:00Z"},
                {"url": "https://b.example/y", "title": "B title",
                 "content": "more python text", "engine": "bing"},
            ],
        }
        with mock.patch("nomorals.search.web._http",
                         return_value=_resp(payload)) as m:
            hits = SearXNGWebSource().search("python", limit=5)
        self.assertEqual(m.call_args[0][1], "https://searx.example.com/search")
        self.assertEqual(m.call_args[1]["params"]["format"], "json")
        self.assertEqual(len(hits), 2)
        # BM25 rerank: "snippet about python here" mentions python once in
        # title+snippet vs "B title"+"more python text" — both mention it;
        # check the mapping contract instead of exact order:
        by_url = {h.provenance["url"]: h for h in hits}
        a = by_url["https://a.example/x"]
        self.assertEqual(a.source, "web_searxng")
        self.assertEqual(a.type, "web")
        self.assertEqual(a.provenance["engine"], "google")
        self.assertEqual(a.provenance["backend_score"], 42.5)
        self.assertIsNotNone(a.timestamp)  # publishedDate parsed
        self.assertTrue(a.source_id.startswith("web_searxng:"))
        # native score stashed, raw_score now BM25
        self.assertIsInstance(a.raw_score, float)

    def test_rerank_can_be_disabled(self):
        payload = {"results": [
            {"url": "https://a.example/1", "title": "zzz unrelated",
             "content": "nothing matching here", "engine": "google", "score": 99.0},
            {"url": "https://b.example/2", "title": "python guide",
             "content": "python python python", "engine": "bing", "score": 1.0},
        ]}
        os.environ["NM_WEB_RERANK"] = "0"
        try:
            with mock.patch("nomorals.search.web._http",
                             return_value=_resp(payload)):
                hits = SearXNGWebSource().search("python", limit=5)
        finally:
            del os.environ["NM_WEB_RERANK"]
        # rerank off: native order and native raw scores preserved
        self.assertEqual(hits[0].provenance["url"], "https://a.example/1")
        self.assertEqual(hits[0].raw_score, 99.0)

    def test_403_json_disabled_fails_over(self):
        good = {"results": [{"url": "https://ok.example/", "title": "ok",
                             "content": "fine", "engine": "google"}]}
        os.environ["NM_SEARXNG_URL"] = (
            "https://bad.example.com, https://good.example.com"
        )
        calls = []

        def fake_http(method, url, **kw):
            calls.append(url)
            if "bad.example" in url:
                return (403, b"forbidden")
            return _resp(good)

        with mock.patch("nomorals.search.web._http", side_effect=fake_http):
            hits = SearXNGWebSource().search("q", limit=5)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].provenance["url"], "https://ok.example/")

    def test_all_instances_fail_raises(self):
        with mock.patch("nomorals.search.web._http",
                         return_value=(403, b"nope")):
            with self.assertRaises(WebBackendError) as cm:
                SearXNGWebSource().search("q", limit=5)
        self.assertIn("format=json", str(cm.exception))

    def test_429_maps_to_rate_limit_error(self):
        with mock.patch("nomorals.search.web._http",
                         return_value=(429, b"slow down")):
            with self.assertRaises(WebBackendError) as cm:
                SearXNGWebSource().search("q", limit=5)
        self.assertIn("429", str(cm.exception))

    def test_non_json_body_raises(self):
        with mock.patch("nomorals.search.web._http",
                         return_value=(200, b"<html>not json</html>")):
            with self.assertRaises(WebBackendError) as cm:
                SearXNGWebSource().search("q", limit=5)
        self.assertIn("non-JSON", str(cm.exception))

    def test_transport_failure_raises(self):
        with mock.patch("nomorals.search.web._http",
                         side_effect=WebBackendError("HTTP GET x failed: boom")):
            with self.assertRaises(WebBackendError):
                SearXNGWebSource().search("q", limit=5)

    def test_hits_without_url_or_title_dropped(self):
        payload = {"results": [{"content": "no identity at all"},
                               {"url": "https://u.example/", "title": "t",
                                "content": "c"}]}
        with mock.patch("nomorals.search.web._http",
                         return_value=_resp(payload)):
            hits = SearXNGWebSource().search("q", limit=5)
        self.assertEqual(len(hits), 1)


class DdgsTests(unittest.TestCase):
    def setUp(self):
        self._cleaner = _EnvCleaner()
        self._cleaner.__enter__()
        DdgsWebSource._ddgs_missing = None
        self._saved_module = sys.modules.pop("ddgs", None)

    def tearDown(self):
        self._cleaner.__exit__()
        DdgsWebSource._ddgs_missing = None
        if self._saved_module is not None:
            sys.modules["ddgs"] = self._saved_module
        else:
            sys.modules.pop("ddgs", None)

    def _install_fake_ddgs(self, rows, legacy=False):
        mod = types.ModuleType("ddgs")

        class DDGS:
            def __init__(self, timeout=10):
                self.timeout = timeout

            def text(self, keywords, **kw):
                if not legacy and "backend" not in kw:
                    raise AssertionError("backend kwarg expected")
                if legacy and "backend" in kw:
                    raise TypeError("unexpected backend kwarg")
                return rows

        mod.DDGS = DDGS
        sys.modules["ddgs"] = mod

    def test_probe_missing_package(self):
        note = DdgsWebSource().probe()
        self.assertIn("pip install ddgs", note)

    def test_probe_with_package(self):
        self._install_fake_ddgs([])
        self.assertIsNone(DdgsWebSource().probe())

    def test_parses_rows(self):
        self._install_fake_ddgs([
            {"title": "T1", "href": "https://t1.example/",
             "body": "python body text", "backend": "duckduckgo"},
        ])
        hits = DdgsWebSource().search("python", limit=5)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h.source, "web_ddgs")
        self.assertEqual(h.type, "web")
        self.assertEqual(h.title, "T1")
        self.assertEqual(h.provenance["url"], "https://t1.example/")
        self.assertEqual(h.provenance["engine"], "duckduckgo")

    def test_legacy_ddgs_without_backend_kwarg(self):
        self._install_fake_ddgs(
            [{"title": "T", "href": "https://t.example/", "body": "b"}],
            legacy=True,
        )
        hits = DdgsWebSource().search("q", limit=5)
        self.assertEqual(len(hits), 1)

    def test_ddgs_error_wrapped(self):
        mod = types.ModuleType("ddgs")

        class DDGS:
            def __init__(self, timeout=10):
                pass

            def text(self, keywords, **kw):
                raise RuntimeError("ratelimited")

        mod.DDGS = DDGS
        sys.modules["ddgs"] = mod
        with self.assertRaises(WebBackendError) as cm:
            DdgsWebSource().search("q", limit=5)
        self.assertIn("ratelimited", str(cm.exception))


class TavilyTests(unittest.TestCase):
    def setUp(self):
        self._cleaner = _EnvCleaner()
        self._cleaner.__enter__()
        os.environ["NM_TAVILY_API_KEY"] = "tv-test-key"

    def tearDown(self):
        self._cleaner.__exit__()

    def test_probe(self):
        self.assertIsNone(TavilyWebSource().probe())
        del os.environ["NM_TAVILY_API_KEY"]
        self.assertIn("NM_TAVILY_API_KEY", TavilyWebSource().probe())

    def test_parses_results(self):
        payload = {"results": [
            {"title": "T", "url": "https://t.example/", "content": "python c",
             "score": 0.93, "published_date": "2026-08-15"},
        ]}
        with mock.patch("nomorals.search.web._http",
                         return_value=_resp(payload)) as m:
            hits = TavilyWebSource().search("python", limit=5)
        _args, kwargs = m.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer tv-test-key")
        self.assertEqual(kwargs["json_body"]["query"], "python")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].provenance["backend_score"], 0.93)
        self.assertIsNotNone(hits[0].timestamp)

    def test_401_invalid_key(self):
        with mock.patch("nomorals.search.web._http", return_value=(401, b"{}")):
            with self.assertRaises(WebBackendError) as cm:
                TavilyWebSource().search("q", limit=3)
        self.assertIn("invalid API key", str(cm.exception))

    def test_432_quota_exhausted(self):
        with mock.patch("nomorals.search.web._http", return_value=(432, b"{}")):
            with self.assertRaises(WebBackendError) as cm:
                TavilyWebSource().search("q", limit=3)
        self.assertIn("quota", str(cm.exception))


class SerperTests(unittest.TestCase):
    def setUp(self):
        self._cleaner = _EnvCleaner()
        self._cleaner.__enter__()
        os.environ["NM_SERPER_API_KEY"] = "serper-key"

    def tearDown(self):
        self._cleaner.__exit__()

    def test_probe(self):
        self.assertIsNone(SerperWebSource().probe())
        del os.environ["NM_SERPER_API_KEY"]
        self.assertIn("NM_SERPER_API_KEY", SerperWebSource().probe())

    def test_parses_organic(self):
        payload = {"organic": [
            {"title": "T", "link": "https://t.example/", "snippet": "python s"},
        ]}
        with mock.patch("nomorals.search.web._http",
                         return_value=_resp(payload)) as m:
            hits = SerperWebSource().search("python", limit=5)
        _args, kwargs = m.call_args
        self.assertEqual(kwargs["headers"]["X-API-KEY"], "serper-key")
        self.assertEqual(kwargs["json_body"]["q"], "python")
        self.assertEqual(hits[0].provenance["url"], "https://t.example/")
        self.assertEqual(hits[0].snippet, "python s")

    def test_403_invalid_key(self):
        with mock.patch("nomorals.search.web._http", return_value=(403, b"{}")):
            with self.assertRaises(WebBackendError) as cm:
                SerperWebSource().search("q", limit=3)
        self.assertIn("invalid API key", str(cm.exception))


class ExaTests(unittest.TestCase):
    def setUp(self):
        self._cleaner = _EnvCleaner()
        self._cleaner.__enter__()
        os.environ["NM_EXA_API_KEY"] = "exa-key"

    def tearDown(self):
        self._cleaner.__exit__()

    def test_probe(self):
        self.assertIsNone(ExaWebSource().probe())
        del os.environ["NM_EXA_API_KEY"]
        self.assertIn("NM_EXA_API_KEY", ExaWebSource().probe())

    def test_parses_highlights(self):
        payload = {"results": [
            {"title": "T", "url": "https://t.example/",
             "highlights": ["python one", "python two"],
             "publishedDate": "2026-07-01T00:00:00.000Z"},
        ]}
        with mock.patch("nomorals.search.web._http",
                         return_value=_resp(payload)) as m:
            hits = ExaWebSource().search("python", limit=5)
        _args, kwargs = m.call_args
        self.assertEqual(kwargs["headers"]["x-api-key"], "exa-key")
        self.assertTrue(kwargs["json_body"]["contents"]["highlights"])
        self.assertIn("python one", hits[0].snippet)
        self.assertIn("python two", hits[0].snippet)
        self.assertIsNotNone(hits[0].timestamp)

    def test_429_rate_limited(self):
        with mock.patch("nomorals.search.web._http", return_value=(429, b"{}")):
            with self.assertRaises(WebBackendError) as cm:
                ExaWebSource().search("q", limit=3)
        self.assertIn("429", str(cm.exception))


class BraveTests(unittest.TestCase):
    def setUp(self):
        self._cleaner = _EnvCleaner()
        self._cleaner.__enter__()
        os.environ["NM_BRAVE_SEARCH_API_KEY"] = "brave-key"

    def tearDown(self):
        self._cleaner.__exit__()

    def test_probe(self):
        self.assertIsNone(BraveWebSource().probe())
        del os.environ["NM_BRAVE_SEARCH_API_KEY"]
        self.assertIn("NM_BRAVE_SEARCH_API_KEY", BraveWebSource().probe())

    def test_parses_web_results(self):
        payload = {"web": {"results": [
            {"title": "T", "url": "https://t.example/",
             "description": "python description"},
        ]}}
        with mock.patch("nomorals.search.web._http",
                         return_value=_resp(payload)) as m:
            hits = BraveWebSource().search("python", limit=5)
        _args, kwargs = m.call_args
        self.assertEqual(kwargs["headers"]["X-Subscription-Token"], "brave-key")
        self.assertEqual(kwargs["params"]["q"], "python")
        self.assertEqual(hits[0].snippet, "python description")
        self.assertEqual(hits[0].source, "web_brave")

    def test_401_invalid_token(self):
        with mock.patch("nomorals.search.web._http", return_value=(401, b"{}")):
            with self.assertRaises(WebBackendError) as cm:
                BraveWebSource().search("q", limit=3)
        self.assertIn("invalid subscription token", str(cm.exception))


class ResultMappingTests(unittest.TestCase):
    def test_timestamp_parsing(self):
        src = TavilyWebSource()
        r = src._to_result("q", 0, {"title": "t", "url": "https://u/",
                                   "snippet": "s", "published": "2026-01-15T12:00:00Z"})
        self.assertAlmostEqual(r.timestamp, 1768478400.0, delta=2)
        r2 = src._to_result("q", 0, {"title": "t", "url": "https://u/",
                                    "snippet": "s", "published": "garbage"})
        self.assertIsNone(r2.timestamp)
        r3 = src._to_result("q", 0, {"title": "t", "url": "https://u/",
                                    "snippet": "s"})
        self.assertIsNone(r3.timestamp)

    def test_source_ids_stable_per_url(self):
        src = TavilyWebSource()
        a = src._to_result("q", 0, {"title": "t", "url": "https://u/", "snippet": "s"})
        b = src._to_result("q", 3, {"title": "t2", "url": "https://u/", "snippet": "s2"})
        self.assertEqual(a.source_id, b.source_id)

    def test_web_backend_error_is_search_error(self):
        self.assertTrue(issubclass(WebBackendError, SearchError))

    def test_result_type_is_web(self):
        for cls in (SearXNGWebSource, DdgsWebSource, TavilyWebSource,
                    SerperWebSource, ExaWebSource, BraveWebSource):
            self.assertEqual(cls.result_type, "web")


if __name__ == "__main__":
    unittest.main()
