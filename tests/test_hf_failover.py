"""Integration tests for HF provider failover (mocked HTTP).

Covers the multi-model failover path in HFServerlessProvider.chat():
- primary model 400 "not supported" -> tries candidates, succeeds
- 401/403 -> NO failover (auth error returned directly)
- all candidates fail -> original model restored, error returned
- healed_from recorded in raw on successful failover
- max 3 swaps enforced per call
"""

from __future__ import annotations

import unittest
from unittest import mock

from nomorals.llm.providers.hf_serverless import (
    HFServerlessProvider,
    ROUTER_FALLBACK_MODELS,
)
from nomorals.llm.base import Message, SamplingParams


def _chat_response(text="hello", model="test-model"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "model": model,
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _make_provider(model="bad/model-not-hosted"):
    return HFServerlessProvider(token="test-token", model=model)


class FailoverTests(unittest.TestCase):
    def test_failover_on_model_not_supported(self):
        """400 'not supported' on primary -> tries next candidate, succeeds."""
        provider = _make_provider()
        calls = []

        def fake_post(url, payload, **kw):
            calls.append(provider.model)
            if provider.model == "bad/model-not-hosted":
                raise Exception('400: {"error": "Model not supported by provider hf-inference"}')
            return _chat_response(text="healed!", model=provider.model)

        with mock.patch.object(provider, "_post_checked", side_effect=fake_post):
            # Mock catalog to return a known good candidate first
            with mock.patch.object(
                provider, "fetch_catalog",
                return_value=[{"id": "good/model-a"}],
            ):
                resp = provider.chat([Message.user("hi")])

        self.assertTrue(resp.text == "healed!", f"got: {resp!r}")
        self.assertEqual(provider.model, "good/model-a")
        self.assertEqual(resp.raw.get("healed_from"), "bad/model-not-hosted")
        # primary + 1 candidate = 2 calls
        self.assertEqual(len(calls), 2)

    def test_no_failover_on_401(self):
        """401 -> auth error, no model swapping."""
        provider = _make_provider()
        calls = []

        def fake_post(url, payload, **kw):
            calls.append(provider.model)
            raise Exception("401 Unauthorized: invalid token")

        with mock.patch.object(provider, "_post_checked", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])

        self.assertEqual(resp.text, "")
        self.assertTrue(resp.error, "expected an error")
        self.assertEqual(provider.model, "bad/model-not-hosted", "model must not change on auth error")
        self.assertEqual(len(calls), 1, "no retry on auth error")

    def test_no_failover_on_403(self):
        """403 -> auth error, no model swapping."""
        provider = _make_provider()
        calls = []

        def fake_post(url, payload, **kw):
            calls.append(provider.model)
            raise Exception("403 Forbidden")

        with mock.patch.object(provider, "_post_checked", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])

        self.assertEqual(resp.text, "")
        self.assertTrue(resp.error)
        self.assertEqual(provider.model, "bad/model-not-hosted")
        self.assertEqual(len(calls), 1)

    def test_all_candidates_fail_restores_original(self):
        """Every candidate fails -> original model restored, error returned."""
        provider = _make_provider()

        def fake_post(url, payload, **kw):
            raise Exception("503 Service Unavailable: model loading")

        with mock.patch.object(provider, "_post_checked", side_effect=fake_post):
            with mock.patch.object(provider, "fetch_catalog", return_value=[]):
                resp = provider.chat([Message.user("hi")])

        self.assertEqual(resp.text, "")
        self.assertTrue(resp.error)
        self.assertEqual(
            provider.model, "bad/model-not-hosted",
            "original model must be restored after all candidates fail",
        )

    def test_max_three_swaps(self):
        """Failover caps at 3 candidate swaps even with more available."""
        provider = _make_provider()
        attempted = []

        def fake_post(url, payload, **kw):
            attempted.append(provider.model)
            raise Exception("400: model not supported")

        many_models = [{"id": f"org/model-{i}"} for i in range(10)]
        with mock.patch.object(provider, "_post_checked", side_effect=fake_post):
            with mock.patch.object(provider, "fetch_catalog", return_value=many_models):
                resp = provider.chat([Message.user("hi")])

        self.assertTrue(resp.error)
        # 1 primary + 3 candidates = 4 total attempts
        self.assertEqual(len(attempted), 4, f"attempted: {attempted}")

    def test_candidate_excludes_original(self):
        """_candidate_models never returns the current model."""
        provider = _make_provider(model="org/keep-me")
        with mock.patch.object(
            provider, "fetch_catalog",
            return_value=[
                {"id": "org/keep-me"},
                {"id": "org/other"},
            ],
        ):
            candidates = provider._candidate_models(exclude="org/keep-me")
        self.assertNotIn("org/keep-me", candidates)
        self.assertIn("org/other", candidates)

    def test_curated_fallback_used_when_catalog_fails(self):
        """fetch_catalog raising -> falls back to ROUTER_FALLBACK_MODELS."""
        provider = _make_provider()
        with mock.patch.object(provider, "fetch_catalog", side_effect=Exception("no net")):
            candidates = provider._candidate_models(exclude="bad/model-not-hosted")
        for m in ROUTER_FALLBACK_MODELS:
            if m != "bad/model-not-hosted":
                self.assertIn(m, candidates)

    def test_success_on_primary_no_failover(self):
        """Happy path: primary works, no model swapping, no healed_from."""
        provider = _make_provider(model="good/primary")
        with mock.patch.object(
            provider, "_post_checked",
            return_value=_chat_response(text="direct", model="good/primary"),
        ):
            resp = provider.chat([Message.user("hi")])
        self.assertEqual(resp.text, "direct")
        self.assertEqual(provider.model, "good/primary")
        self.assertNotIn("healed_from", resp.raw or {})


if __name__ == "__main__":
    unittest.main()
