"""Universal wave — Groq and OpenRouter free-tier providers.

Offline tests against a stubbed HTTP client: provider defaults, env-key
handling, capability sets (Groq has no embeddings endpoint), OpenRouter's
live :free roster discovery, and quota introspection.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from nomorals.core.errors import ProviderError
from nomorals.llm.providers.groq import GROQ_BASE_URL, GROQ_FREE_MODELS, GroqProvider
from nomorals.llm.providers.openai_compat import OpenAICompatProvider
from nomorals.llm.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OPENROUTER_FREE_ALIAS,
    OpenRouterProvider,
)


class FakeResponse:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self._ok = ok

    @property
    def ok(self):
        return self._ok

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []

    def _handle(self, method, url, payload=None):
        self.calls.append((method, url, payload))
        for (rmethod, suffix), result in self.routes.items():
            if method == rmethod and url.endswith(suffix):
                return FakeResponse(result)
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kw):
        return self._handle("GET", url)

    def post_json(self, url, payload, **kw):
        return self._handle("POST", url, payload)


class GroqProviderTests(unittest.TestCase):
    def test_subclasses_openai_compat(self):
        self.assertIsInstance(GroqProvider(model="x"), OpenAICompatProvider)

    def test_defaults(self):
        provider = GroqProvider()
        self.assertEqual(provider.base_url, GROQ_BASE_URL)
        self.assertEqual(provider.model, "openai/gpt-oss-120b")
        self.assertEqual(provider.name, "groq")

    def test_api_key_from_env(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-test-key"}, clear=False):
            provider = GroqProvider()
        self.assertEqual(provider.http.headers.get("Authorization"),
                         "Bearer gsk-test-key")

    def test_explicit_key_beats_env(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-env"}, clear=False):
            provider = GroqProvider(api_key="gsk-explicit")
        self.assertEqual(provider.http.headers.get("Authorization"),
                         "Bearer gsk-explicit")

    def test_no_embed_capability(self):
        # Groq serves no /v1/embeddings — the router must never send embed
        # work there.
        provider = GroqProvider(model="x")
        self.assertNotIn("embed", provider.capabilities)
        self.assertEqual(provider.capabilities, {"chat", "complete", "vision"})

    def test_free_model_table_is_sane(self):
        ids = {m["id"] for m in GROQ_FREE_MODELS}
        self.assertIn("openai/gpt-oss-120b", ids)
        self.assertIn("openai/gpt-oss-20b", ids)
        for entry in GROQ_FREE_MODELS:
            self.assertGreater(entry["context"], 0)
            self.assertGreater(entry["requests_per_day"], 0)
            self.assertGreater(entry["tokens_per_day"], 0)

    def test_is_free_tier(self):
        self.assertTrue(GroqProvider(model="openai/gpt-oss-120b").is_free_tier)
        self.assertFalse(GroqProvider(model="some-paid-model").is_free_tier)

    def test_free_model_cards(self):
        cards = GroqProvider(model="x").free_model_cards()
        self.assertEqual(len(cards), len(GROQ_FREE_MODELS))
        self.assertTrue(all("id" in c for c in cards))


class OpenRouterProviderTests(unittest.TestCase):
    def test_subclasses_openai_compat(self):
        self.assertIsInstance(OpenRouterProvider(model="x"), OpenAICompatProvider)

    def test_defaults(self):
        provider = OpenRouterProvider()
        self.assertEqual(provider.base_url, OPENROUTER_BASE_URL)
        self.assertEqual(provider.model, OPENROUTER_FREE_ALIAS)
        self.assertEqual(provider.name, "openrouter")
        self.assertTrue(provider.is_free_model)

    def test_api_key_from_env(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"},
                             clear=False):
            provider = OpenRouterProvider()
        self.assertEqual(provider.http.headers.get("Authorization"),
                         "Bearer sk-or-test")

    def test_attribution_headers_defaulted(self):
        provider = OpenRouterProvider(model="x")
        headers = provider.http.headers
        self.assertIn("HTTP-Referer", headers)
        self.assertEqual(headers["X-Title"], "Devon")

    def test_caller_headers_win(self):
        provider = OpenRouterProvider(
            model="x",
            extra_headers={"X-Title": "CustomApp",
                           "HTTP-Referer": "https://example.com"},
        )
        headers = provider.http.headers
        self.assertEqual(headers["X-Title"], "CustomApp")
        self.assertEqual(headers["HTTP-Referer"], "https://example.com")

    def test_free_models_filters_free_suffix(self):
        provider = OpenRouterProvider(model="x")
        provider.http = FakeHttp(
            routes={
                ("GET", "/models"): {
                    "data": [
                        {"id": "tencent/hy3:free", "context_length": 262144,
                         "description": "big moe"},
                        {"id": "openai/gpt-4o", "context_length": 128000},
                        {"id": "qwen/qwen3-coder:free", "context_length": 1000000},
                        {"id": "not-a-model"},
                    ]
                }
            }
        )
        free = provider.free_models()
        ids = [m["id"] for m in free]
        self.assertEqual(ids, ["tencent/hy3:free", "qwen/qwen3-coder:free"])
        self.assertEqual(free[0]["context_length"], 262144)
        _, url, _ = provider.http.calls[0]
        self.assertTrue(url.endswith("/models"))

    def test_free_models_bad_shape_raises(self):
        provider = OpenRouterProvider(model="x")
        provider.http = FakeHttp(routes={("GET", "/models"): {"data": "nope"}})
        with self.assertRaises(ProviderError):
            provider.free_models()

    def test_key_info_returns_quota(self):
        provider = OpenRouterProvider(model="x")
        provider.http = FakeHttp(
            routes={
                ("GET", "/key"): {
                    "data": {
                        "label": "dev", "limit_remaining": 42,
                        "is_free_tier": True, "usage_daily": 8,
                    }
                }
            }
        )
        info = provider.key_info()
        self.assertEqual(info["limit_remaining"], 42)
        self.assertTrue(info["is_free_tier"])

    def test_is_free_model_alias_and_suffix(self):
        self.assertTrue(OpenRouterProvider(model="openrouter/free").is_free_model)
        self.assertTrue(OpenRouterProvider(model="deepseek/r1:free").is_free_model)
        self.assertFalse(OpenRouterProvider(model="openai/gpt-4o").is_free_model)


if __name__ == "__main__":
    unittest.main()
