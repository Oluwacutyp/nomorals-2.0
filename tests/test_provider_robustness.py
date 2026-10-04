"""Provider robustness: error classification, model failover, timeouts.

Covers the audit fixes:
- core.http.http_error attaches machine-readable ``details["http_status"]``
- core.errors helpers: http_status_of / is_auth_error / is_not_found_error /
  is_rate_limited_error / is_server_error
- HFServerlessProvider: auth detection via helper (not message sniffing);
  all-models-failed error carries the full attempt chain
- OpenAICompatProvider: generic 404 model-id failover via _fallback_models
- GroqProvider: heals a retired GROQ_MODEL onto the free roster
- OpenRouterProvider: heals a rotted :free id onto the openrouter/free alias
"""

from __future__ import annotations

import unittest
from unittest import mock

from nomorals.core.errors import (
    ProviderError,
    RateLimited,
    http_status_of,
    is_auth_error,
    is_not_found_error,
    is_rate_limited_error,
    is_server_error,
)
from nomorals.core.http import http_error
from nomorals.llm.base import Message
from nomorals.llm.providers.groq import GROQ_FREE_MODELS, GroqProvider
from nomorals.llm.providers.hf_serverless import HFServerlessProvider
from nomorals.llm.providers.openai_compat import OpenAICompatProvider
from nomorals.llm.providers.openrouter import OPENROUTER_FREE_ALIAS, OpenRouterProvider


def _chat_payload(text="ok", model="m"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "model": model,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class FakeRaw:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


# ── HTTP status classification ──────────────────────────────────────────────

class HttpErrorTests(unittest.TestCase):
    def test_status_attached(self):
        for status, retryable in [(401, False), (403, False), (404, False),
                                  (429, True), (500, True), (503, True),
                                  (418, False)]:
            err = http_error(status, "body", "https://x.test/")
            self.assertEqual(err.details.get("http_status"), status,
                             f"status {status}")
            self.assertEqual(err.retryable, retryable, f"status {status}")

    def test_429_carries_retry_after(self):
        err = http_error(429, "slow down", "https://x.test/",
                         {"Retry-After": "7"})
        self.assertIsInstance(err, RateLimited)
        self.assertEqual(err.retry_after, 7.0)
        self.assertEqual(err.details.get("http_status"), 429)

    def test_401_never_retryable(self):
        err = http_error(401, "bad key", "https://x.test/")
        self.assertFalse(err.retryable)

    def test_404_message_names_model(self):
        err = http_error(404, '{"error": "model_not_found: dead/model"}',
                         "https://x.test/chat")
        self.assertIn("dead/model", err.message)


class ClassifierTests(unittest.TestCase):
    def test_from_details(self):
        self.assertEqual(http_status_of(http_error(403, "", "u")), 403)
        self.assertTrue(is_auth_error(http_error(401, "", "u")))
        self.assertTrue(is_auth_error(http_error(403, "", "u")))
        self.assertTrue(is_not_found_error(http_error(404, "", "u")))
        self.assertTrue(is_rate_limited_error(http_error(429, "", "u")))
        self.assertTrue(is_server_error(http_error(500, "", "u")))
        self.assertTrue(is_server_error(http_error(503, "", "u")))
        self.assertFalse(is_auth_error(http_error(404, "", "u")))
        self.assertFalse(is_not_found_error(http_error(401, "", "u")))

    def test_from_message_fallback(self):
        # Errors built from raw strings (third-party clients, tests) still
        # classify via the status code embedded in the message.
        self.assertTrue(is_auth_error(Exception("401 Unauthorized: bad token")))
        self.assertTrue(is_auth_error(Exception("403 Forbidden")))
        self.assertTrue(is_not_found_error(Exception("404 not found: https://x")))
        self.assertTrue(is_rate_limited_error(Exception("429 too many")))
        self.assertTrue(is_server_error(Exception("500 internal")))
        self.assertIsNone(http_status_of(Exception("some weird failure")))

    def test_keyword_fallback_without_status(self):
        self.assertTrue(is_auth_error(Exception("unauthorized: key revoked")))
        self.assertFalse(is_auth_error(Exception("connection reset by peer")))


# ── HF provider ─────────────────────────────────────────────────────────────

class HFFailoverChainTests(unittest.TestCase):
    def _provider(self, model="bad/model"):
        return HFServerlessProvider(token="t", model=model)

    def test_all_fail_error_carries_chain(self):
        provider = self._provider()
        with mock.patch.object(
            provider, "_post_checked",
            side_effect=http_error(404, "not hosted", "https://hf.test"),
        ), mock.patch.object(provider, "fetch_catalog", return_value=[
            {"id": "org/a"}, {"id": "org/b"},
        ]):
            resp = provider.chat([Message.user("hi")])
        self.assertTrue(resp.error)
        # every attempted model appears with its reason — not just the last
        self.assertIn("bad/model", resp.error)
        self.assertIn("org/a", resp.error)
        self.assertIn("org/b", resp.error)
        attempts = (resp.raw or {}).get("heal_attempts") or []
        self.assertEqual(len(attempts), 4)  # primary + 3 candidate swaps
        self.assertEqual(provider.model, "bad/model")  # restored

    def test_auth_error_uses_helper_not_sniffing(self):
        provider = self._provider()
        err = http_error(401, "bad credentials", "https://hf.test")
        with mock.patch.object(provider, "_post_checked", side_effect=err):
            resp = provider.chat([Message.user("hi")])
        self.assertTrue(resp.error)
        self.assertEqual(provider.model, "bad/model")

    def test_sane_default_model(self):
        # The default must be a model the HF router actually serves —
        # starting on a 404 id wastes a heal round trip on every boot.
        self.assertEqual(
            HFServerlessProvider().model, "meta-llama/Llama-3.1-8B-Instruct")

    def test_timeout_bounded(self):
        self.assertLessEqual(HFServerlessProvider().timeout, 90.0)


# ── OpenAI-compatible generic failover ──────────────────────────────────────

class _FlakyCompat(OpenAICompatProvider):
    """Test double with a configurable fallback roster."""

    def __init__(self, roster, **kw):
        super().__init__(base_url="https://x.test/v1", **kw)
        self._roster = list(roster)

    def _fallback_models(self):
        return list(self._roster)


class CompatFailoverTests(unittest.TestCase):
    def test_404_heals_onto_fallback(self):
        provider = _FlakyCompat(["good/model"], model="dead/model")

        def fake_post(url, payload):
            if payload["model"] == "dead/model":
                raise http_error(404, '{"error": {"message": "model_not_found"}}',
                                 url)
            return FakeRaw(_chat_payload(text="healed", model=payload["model"]))

        with mock.patch.object(provider.http, "post_json", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])
        self.assertEqual(resp.text, "healed")
        self.assertEqual(provider.model, "good/model")  # healed id sticks
        self.assertEqual((resp.raw or {}).get("healed_from"), "dead/model")

    def test_401_does_not_heal(self):
        provider = _FlakyCompat(["good/model"], model="dead/model")
        calls = []

        def fake_post(url, payload):
            calls.append(payload["model"])
            raise http_error(401, "bad key", url)

        with mock.patch.object(provider.http, "post_json", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])
        self.assertTrue(resp.error)
        self.assertEqual(calls, ["dead/model"])  # single attempt, no swap
        self.assertEqual(provider.model, "dead/model")

    def test_429_does_not_heal_models(self):
        # Rate limits are per-key: swapping the model cannot fix them.
        # Retry-After: 0 keeps the test fast (no sleeping).
        provider = _FlakyCompat(["good/model"], model="m1")
        calls = []

        def fake_post(url, payload):
            calls.append(payload["model"])
            raise http_error(429, "slow", url, {"Retry-After": "0"})

        with mock.patch.object(provider.http, "post_json", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])
        self.assertTrue(resp.error)
        self.assertEqual(set(calls), {"m1"})  # retried, never swapped
        self.assertGreaterEqual(len(calls), 2)  # policy retried the 429

    def test_all_fail_reports_chain(self):
        provider = _FlakyCompat(["dead/b", "dead/c"], model="dead/a")

        def fake_post(url, payload):
            raise http_error(404, "nope", url)

        with mock.patch.object(provider.http, "post_json", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])
        self.assertTrue(resp.error)
        for mid in ("dead/a", "dead/b", "dead/c"):
            self.assertIn(mid, resp.error)
        self.assertEqual(provider.model, "dead/a")  # restored
        attempts = (resp.raw or {}).get("model_failover_attempts") or []
        self.assertEqual([a["model"] for a in attempts],
                         ["dead/a", "dead/b", "dead/c"])

    def test_no_roster_single_attempt(self):
        provider = OpenAICompatProvider(base_url="https://x.test/v1",
                                        model="dead/model")
        calls = []

        def fake_post(url, payload):
            calls.append(1)
            raise http_error(404, "nope", url)

        with mock.patch.object(provider.http, "post_json", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])
        self.assertTrue(resp.error)
        self.assertEqual(len(calls), 1)


# ── Groq / OpenRouter ───────────────────────────────────────────────────────

class GroqFailoverTests(unittest.TestCase):
    def test_fallback_roster_is_free_models(self):
        provider = GroqProvider(api_key="k")
        self.assertEqual(provider._fallback_models(),
                         [m["id"] for m in GROQ_FREE_MODELS])
        self.assertIn("openai/gpt-oss-120b", provider._fallback_models())

    def test_retired_model_heals(self):
        # The user's exact outage: GROQ_MODEL=llama-3.3-70b-versatile 404s.
        provider = GroqProvider(api_key="k", model="llama-3.3-70b-versatile")

        def fake_post(url, payload):
            if payload["model"] == "llama-3.3-70b-versatile":
                raise http_error(
                    404,
                    '{"error": {"message": "The model `llama-3.3-70b-versatile` '
                    'does not exist or you do not have access to it."}}',
                    url)
            return FakeRaw(_chat_payload(text="via gpt-oss",
                                         model=payload["model"]))

        with mock.patch.object(provider.http, "post_json", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])
        self.assertEqual(resp.text, "via gpt-oss")
        self.assertEqual(provider.model, "openai/gpt-oss-120b")
        self.assertEqual((resp.raw or {}).get("healed_from"),
                         "llama-3.3-70b-versatile")

    def test_timeout_bounded(self):
        self.assertLessEqual(GroqProvider(api_key="k").timeout, 60.0)


class OpenRouterFailoverTests(unittest.TestCase):
    def test_fallback_is_alias(self):
        provider = OpenRouterProvider(api_key="k",
                                      model="some/dead-model:free")
        self.assertEqual(provider._fallback_models(), [OPENROUTER_FREE_ALIAS])

    def test_rotted_free_id_heals_to_alias(self):
        provider = OpenRouterProvider(api_key="k",
                                      model="retired/model:free")

        def fake_post(url, payload):
            if payload["model"] != OPENROUTER_FREE_ALIAS:
                raise http_error(404, '{"error": "No endpoints found"}', url)
            return FakeRaw(_chat_payload(text="via alias",
                                         model=payload["model"]))

        with mock.patch.object(provider.http, "post_json", side_effect=fake_post):
            resp = provider.chat([Message.user("hi")])
        self.assertEqual(resp.text, "via alias")
        self.assertEqual(provider.model, OPENROUTER_FREE_ALIAS)

    def test_alias_is_default(self):
        self.assertEqual(OpenRouterProvider(api_key="k").model,
                         OPENROUTER_FREE_ALIAS)


# ── /providers command ────────────────────────────────────────────────────

class ProvidersCommandTests(unittest.TestCase):
    def _runtime(self, router):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin

        runtime = RuntimeMetaMixin()
        runtime.context = SimpleNamespace(router=router)
        return runtime

    def test_report_marks_live_and_dead(self):
        from nomorals.social.chat.control import parse_control

        self.assertEqual(parse_control("/providers").kind, "providers")

        class FakeProvider:
            name = "groq"

            def health(self):
                return True

            @property
            def model_id(self):
                return "openai/gpt-oss-120b"

        class FakeRouter:
            def providers(self):
                return ["groq", "hf_serverless"]

            def get(self, name):
                return FakeProvider() if name == "groq" else None

            def stats_snapshot(self):
                return {
                    "active": "groq",
                    "chain": ["groq", "hf_serverless"],
                    "health": {
                        "groq": {"calls": 12, "failures": 0,
                                 "cooling_down": False, "last_error": ""},
                        "hf_serverless": {"calls": 5, "failures": 5,
                                          "cooling_down": True,
                                          "last_error": "model.provider: 401 "
                                          "unauthorized for https://x: bad key"},
                    },
                }

        report = self._runtime(FakeRouter())._control_providers()
        self.assertIn("groq", report)
        self.assertIn("live", report)
        self.assertIn("hf_serverless", report)
        self.assertIn("cooling down", report)
        # the actionable hint, not just the raw error
        self.assertIn("auth failed", report)

    def test_no_router(self):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin

        runtime = RuntimeMetaMixin()
        runtime.context = SimpleNamespace(router=None)
        self.assertIn("no router", runtime._control_providers())


if __name__ == "__main__":
    unittest.main()
