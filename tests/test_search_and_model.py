"""The model chain must never silently pretend: no last-resort mock unless
opted in, the doctor pings providers with real calls, and the search engine
works offline in tests (stubbed search, canned pages).
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import unittest

from nomorals.agents.context import build_context
from nomorals.agents.search.curate import curate, dedupe, domain, score_result
from nomorals.agents.search.engine import SearchEngine
from nomorals.agents.search.summarize import extractive_summarize, model_summarize
from nomorals.cli import _cmd_models_doctor
from nomorals.core.config import load_settings
from nomorals.core.errors import ConfigError, ToolError
from nomorals.core.result import Ok
from nomorals.llm.base import LLMResponse, Message, SamplingParams


def _settings(tmp: str, **overrides: str):
    base = {"home": tmp}
    base.update(overrides)
    return load_settings(overrides=base)


class GroqModelDriftTest(unittest.TestCase):
    """The 2026-08-16 Groq deprecation: a default pinned to a model that
    later shut down 404s on every call. The default must never be a
    known-dead model ID."""

    DEPRECATED = {
        "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.1-70b-versatile",
        "llama3-70b-8192", "llama3-8b-8192", "llama-3.2-90b-vision-preview",
    }

    def test_default_groq_model_is_not_deprecated(self) -> None:
        settings = load_settings()
        self.assertNotIn(settings.llm.groq_model, self.DEPRECATED)

    def test_groq_model_env_override(self) -> None:
        import os

        os.environ["NM_GROQ_MODEL"] = "openai/gpt-oss-120b"
        try:
            self.assertEqual(load_settings().llm.groq_model, "openai/gpt-oss-120b")
        finally:
            del os.environ["NM_GROQ_MODEL"]


class Http404BodyTest(unittest.TestCase):
    def test_404_keeps_the_response_body(self) -> None:
        from nomorals.core.http import http_error

        err = http_error(404, '{"error":{"message":"model not found: old-model"}}', "https://x/v1/chat/completions")
        self.assertIn("model not found: old-model", err.message)
        self.assertIn("404", err.message)

    def test_404_without_body_is_still_clean(self) -> None:
        from nomorals.core.http import http_error

        err = http_error(404, "", "https://x/v1/chat/completions")
        self.assertIn("404 not found", err.message)
        self.assertFalse(err.message.endswith("— "))

    def test_binary_body_sanitized_no_control_chars(self) -> None:
        from nomorals.core.http import http_error

        # a gzipped 403 page: raw binary must never land in the message
        body = "\x1f\ufffd\x08\x00blocked-by-waf\x00\x01\x02binary"
        err = http_error(403, body, "https://example.com/page")
        self.assertIn("403", err.message)
        self.assertNotIn("blocked-by-waf", err.message)
        for ch in err.message:
            self.assertTrue(ch.isprintable() or ch in " \t\n\r",
                            f"non-printable char {ch!r} in error message")

    def test_mostly_replacement_chars_reports_non_text(self) -> None:
        from nomorals.core.http import http_error

        # gzip decoded as text: mojibake with many U+FFFD
        body = "\ufffd" * 20 + "X s8++M" + "\ufffd" * 10
        err = http_error(403, body, "https://example.com/page")
        self.assertIn("non-text", err.message)


class NoSilentMockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-nomock-")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _chat_capable(self, context) -> list[str]:
        # The invariant is about CHAT: the OCR floor may be present (vision
        # only, cannot answer text) but no text model may be pretending.
        return [
            name
            for name in context.router.providers()
            if "chat" in (context.router.get(name).capabilities if context.router.get(name) else set())
        ]

    def test_broken_chain_boots_model_less_not_fake(self) -> None:
        # A known provider with no credentials is skipped by the chain
        # builder (the real production path — groq without a key can only
        # fail at call time, so it is never registered). The boot must be
        # model-less, never a silent scripted mock.
        settings = _settings(
            self.tmp.name,
            **{"llm.provider": "groq", "llm.fallback_chain": "",
               "llm.groq_api_key": ""},
        )
        context = build_context(settings, with_executor=False, with_tools=False)
        try:
            self.assertNotIn("mock", context.router.providers())
            self.assertEqual(self._chat_capable(context), [])
        finally:
            context.close()

    def test_opt_in_mock_restores_old_behavior(self) -> None:
        settings = _settings(
            self.tmp.name,
            **{"llm.provider": "groq", "llm.fallback_chain": "",
               "llm.groq_api_key": "", "llm.allow_mock_fallback": "1"},
        )
        context = build_context(settings, with_executor=False, with_tools=False)
        try:
            self.assertEqual(self._chat_capable(context), ["mock"])
        finally:
            context.close()

    def test_mock_env_var_coerces_to_bool(self) -> None:
        import os

        os.environ["NM_LLM_ALLOW_MOCK_FALLBACK"] = "1"
        try:
            self.assertTrue(load_settings().llm.allow_mock_fallback)
        finally:
            del os.environ["NM_LLM_ALLOW_MOCK_FALLBACK"]
        self.assertFalse(load_settings().llm.allow_mock_fallback)


class DoctorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-doctor-")
        self.args = argparse.Namespace(json=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mock_chain_passes_without_network(self) -> None:
        settings = _settings(self.tmp.name)  # default provider is mock
        context = build_context(settings, with_executor=False, with_tools=False)
        try:
            self.assertEqual(_cmd_models_doctor(self.args, context), 0)
        finally:
            context.close()

    def test_unknown_provider_fails_loudly(self) -> None:
        # Fail-fast config validation: an unknown provider name is rejected
        # at settings load — a clear ConfigError naming the offender and
        # listing the known providers — instead of booting model-less. The
        # CLI's top-level handler turns this into exit code 1 with the
        # message on stderr.
        with self.assertRaises(ConfigError) as cm:
            _settings(self.tmp.name,
                      **{"llm.provider": "banana", "llm.fallback_chain": ""})
        message = str(cm.exception)
        self.assertIn("banana", message)
        self.assertIn("known", message)


class CurateTest(unittest.TestCase):
    def test_dedupe_by_url(self) -> None:
        out = dedupe([{"url": "https://a.com/x"}, {"url": "https://a.com/x/"}, {"url": "https://b.com"}])
        self.assertEqual([r["url"] for r in out], ["https://a.com/x", "https://b.com"])

    def test_junk_dropped(self) -> None:
        results = [
            {"url": "https://a.com/file.pdf", "title": "a pdf", "snippet": "x"},
            {"url": "https://a.com", "title": "", "snippet": ""},
            {"url": "https://a.com/guide", "title": "The Python guide", "snippet": "A long real snippet about Python history."},
        ]
        out = curate(results, "python guide", top_n=5)
        self.assertEqual([r["url"] for r in out], ["https://a.com/guide"])

    def test_relevant_beats_position(self) -> None:
        results = [
            {"url": "https://a.com/1", "title": "unrelated cooking show", "snippet": "recipes today"},
            {"url": "https://a.com/2", "title": "Python history", "snippet": "Python was released by Guido in 1991"},
        ]
        out = curate(results, "python history", top_n=2)
        self.assertEqual(out[0]["url"], "https://a.com/2")
        self.assertGreaterEqual(out[0]["score"], 0.0)

    def test_domain_strips_www(self) -> None:
        self.assertEqual(domain("https://www.example.com:8443/x"), "example.com")


class SummarizeTest(unittest.TestCase):
    PAGES = [
        {
            "url": "https://a.com/p",
            "title": "Python",
            "domain": "a.com",
            "text": ("Python is a programming language. "
                     "It was created by Guido van Rossum. "
                     "The first release was in 1991. "
                     "Today it is used for web development and data science."),
        },
        {
            "url": "https://b.com/q",
            "title": "Cooking",
            "domain": "b.com",
            "text": "A good stew needs time. The secret is low heat.",
        },
    ]

    def test_extractive_picks_relevant_cited_sentences(self) -> None:
        out = extractive_summarize("who created python and when", self.PAGES)
        self.assertIn("Guido van Rossum", out)
        self.assertIn("[a.com]", out)
        self.assertNotIn("stew", out)

    def test_extractive_empty_pages_honest(self) -> None:
        out = extractive_summarize("anything", [])
        self.assertIn("no readable text", out)

    def test_model_summarize_uses_router_and_raises_on_failure(self) -> None:
        class _Router:
            def __init__(self, ok: bool) -> None:
                self.ok = ok

            def chat(self, messages, params=None, **kw):
                if not self.ok:
                    return LLMResponse(text="", error="provider down")
                return LLMResponse(text="Python was created by Guido van Rossum; first release 1991.", model="fake")

        out = model_summarize(_Router(True), "who created python", self.PAGES)
        self.assertIn("Guido van Rossum", out)
        with self.assertRaises(RuntimeError):
            model_summarize(_Router(False), "who created python", self.PAGES)


class _StubTools:
    def __init__(self, results: list[dict]) -> None:
        self._results = results

    def call(self, name, **kw):
        assert name == "web_search", name
        return Ok({"query": kw.get("query", ""), "count": len(self._results), "results": self._results})


class _FakeRouter:
    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="no real model here", model="fake")


class SearchEngineTest(unittest.TestCase):
    RESULTS = [
        {"url": "https://py.org/history", "title": "Python history", "snippet": "Python was created by Guido van Rossum."},
        {"url": "https://example.com/docs", "title": "Docs", "snippet": "Documentation for everything, python included."},
    ]

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-search-")
        # power_default_on=False: test_deep_requires_power_mode exercises the
        # power gate itself, so the default-on posture must be off here.
        settings = _settings(self.tmp.name, **{"partner.platforms": "local", "chat.local_enabled": "true",
                                               "partner.power_default_on": False})
        self.context = build_context(settings, with_executor=False, with_tools=False)
        self.context.tools = _StubTools(self.RESULTS)
        self.context.router = _FakeRouter()
        self.engine = SearchEngine(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_quick_run_searches_curates_summarizes(self) -> None:
        self.engine.read = lambda url, max_chars=40000: {  # canned page, no network
            "url": url, "title": "Python", "domain": "py.org",
            "text": "Python is a programming language. It was created by Guido van Rossum. "
                    "The first release was in 1991.",
            "chars": 90,
        }
        report = self.engine.run("who created python", mode="quick", pages=2)
        self.assertEqual(report["mode"], "quick")
        self.assertTrue(report["pages_read"], "at least one page should be read")
        self.assertIn("Guido van Rossum", report["summary"])
        self.assertFalse(report["model_summary"])  # fake router is not a real model
        # journal landed in the DB
        rows = self.context.db.query("SELECT * FROM search_log")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "who created python")

    def test_deep_requires_power_mode(self) -> None:
        with self.assertRaises(ToolError) as ctx:
            self.engine.run("anything", mode="deep")
        self.assertIn("power", str(ctx.exception).lower())

    def test_page_cache_roundtrip(self) -> None:
        page = {"url": "https://x.com/one", "title": "T", "text": "hello world", "domain": "x.com", "chars": 11}
        self.engine._cache_put(page)
        got = self.engine._cache_get("https://x.com/one")
        self.assertIsNotNone(got)
        self.assertEqual(got["title"], "T")
        self.assertEqual(self.engine._cache_get("https://x.com/other"), None)

    def test_leads_dedupe_by_domain_and_persist(self) -> None:
        def _fake_search(q, max_results=6):
            return [
                {"url": "https://platform1.com/sign", "title": "Platform 1", "snippet": "get paid for tasks"},
                {"url": "https://platform1.com/pricing", "title": "Platform 1 pricing", "snippet": "get paid for tasks"},
            ]

        self.engine.search = _fake_search
        leads = self.engine.leads()
        domains = [l["domain"] for l in leads]
        self.assertEqual(domains, [d for d in dict.fromkeys(domains)])  # unique
        rows = self.context.db.query("SELECT * FROM search_leads")
        self.assertGreaterEqual(len(rows), 1)

    def test_history_lists_runs(self) -> None:
        self.assertEqual(self.engine.history(), [])
        self.engine.read = lambda url, max_chars=40000: None  # nothing readable, still journals
        self.engine.run("test query", mode="quick")
        rows = self.engine.history(5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "test query")


if __name__ == "__main__":
    unittest.main()
