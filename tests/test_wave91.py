"""Wave 91: the HF fallback heals its own model name.

Phone incident: the status line (wave 90, honest) revealed
``hf_serverless failed too (HTTP 400 from router.huggingface.co)``.
The cause: the configured model (dolphin-2.9.1-llama-3-8b) is NOT
deployed by any Inference Provider — the router 400s it.  Wave 91:

* the provider retries ONCE with a model the LIVE public catalog
  confirms is hosted (token-free), then stays on it for the session
* the config default model is now a verified-hosted one (same
  uncensored family as the phone's local model)
* a surviving 400 reads in plain words in /status
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals.core.config import get_settings  # noqa: E402
from nomorals.llm.base import LLMResponse, Message  # noqa: E402
from nomorals.llm.providers.hf_serverless import (  # noqa: E402
    ROUTER_FALLBACK_MODELS,
    HFServerlessProvider,
)

CONFIGURED = "cognitivecomputations/dolphin-2.9.1-llama-3-8b"
HEALED = ROUTER_FALLBACK_MODELS[0]  # first curated, catalog-verified model


def _fake_ok_response(model: str) -> object:
    class _Raw:
        ok = True

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": f"hi from {model}"},
                             "finish_reason": "stop"}],
                "model": model,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2},
            }

    return _Raw()


def _fake_bad_response() -> object:
    class _Raw:
        ok = False
        status_code = 503
        text = '{"error": "upstream unavailable"}'

        def raise_for_status(self):
            raise Exception("HTTP 503 upstream")

        def json(self):  # pragma: no cover - never reached when not ok
            return {}

    return _Raw()


def _provider() -> HFServerlessProvider:
    p = HFServerlessProvider(
        token="hf_test", model=CONFIGURED, base_url="https://router.huggingface.co/v1",
    )
    return p


class _Catalog:
    """Stand-in for the live public catalog."""

    def __init__(self, hosted):
        self.hosted = hosted
        self.calls = 0

    def __call__(self, timeout=20.0, catalog_url=""):
        self.calls += 1
        return [{"id": m} for m in self.hosted]


class ModelHealTest(unittest.TestCase):
    def test_rejected_model_heals_with_catalog_confirmed_model(self):
        p = _provider()
        catalog = _Catalog(hosted=[CONFIGURED, HEALED, "other/model"])
        posts = []

        def fake_post_checked(url, payload):
            posts.append(("post_checked", payload["model"]))
            raise Exception("model.provider: HTTP 400 from https://router.huggingface.co/")

        def fake_post_json(url, payload):
            posts.append(("post_json", payload["model"]))
            return _fake_ok_response(payload["model"])

        with mock.patch.object(p, "_post_checked", fake_post_checked), \
             mock.patch.object(p.http, "post_json", fake_post_json), \
             mock.patch.object(HFServerlessProvider, "fetch_catalog", catalog):
            resp = p.chat([Message.user("hello")])
        self.assertEqual(resp.text, f"hi from {HEALED}")
        self.assertEqual(resp.model, HEALED)
        self.assertEqual(resp.raw.get("healed_from"), CONFIGURED)
        # stayed on the healed model for the rest of the session
        self.assertEqual(p.model, HEALED)
        self.assertEqual(posts, [("post_checked", CONFIGURED),
                                 ("post_json", HEALED)])

    def test_discovery_when_no_curated_fallback_hosted(self):
        # no curated id in the catalog → discover a small chat model from
        # the LIVE list itself (the phone-day fix: a stale curated list can
        # never leave the chain silent)
        p = _provider()
        catalog = _Catalog(hosted=[CONFIGURED, "org/Qwen3-8B"])
        posts = []

        def fake_post_checked(url, payload):
            posts.append(("post_checked", payload["model"]))
            raise Exception("model.provider: HTTP 400 from https://router.huggingface.co/")

        def fake_post_json(url, payload):
            posts.append(("post_json", payload["model"]))
            return _fake_ok_response(payload["model"])

        with mock.patch.object(p, "_post_checked", fake_post_checked), \
             mock.patch.object(p.http, "post_json", fake_post_json), \
             mock.patch.object(HFServerlessProvider, "fetch_catalog", catalog):
            resp = p.chat([Message.user("hello")])
        self.assertEqual(resp.text, "hi from org/Qwen3-8B")
        self.assertEqual(p.model, "org/Qwen3-8B")
        self.assertEqual(posts, [("post_checked", CONFIGURED),
                                 ("post_json", "org/Qwen3-8B")])

    def test_discovery_failure_restores_model(self):
        p = _provider()
        catalog = _Catalog(hosted=[CONFIGURED, "org/Qwen3-8B"])
        posts = []

        def fake_post_checked(url, payload):
            posts.append(("post_checked", payload["model"]))
            raise Exception("model.provider: HTTP 400 from https://router.huggingface.co/")

        def fake_post_json(url, payload):
            posts.append(("post_json", payload["model"]))
            return _fake_bad_response()

        with mock.patch.object(p, "_post_checked", fake_post_checked), \
             mock.patch.object(p.http, "post_json", fake_post_json), \
             mock.patch.object(HFServerlessProvider, "fetch_catalog", catalog):
            resp = p.chat([Message.user("hello")])
        self.assertTrue(resp.error)
        self.assertEqual(p.model, CONFIGURED)  # restored
        self.assertEqual(posts, [("post_checked", CONFIGURED),
                                 ("post_json", "org/Qwen3-8B")])

    def test_catalog_down_still_tries_directly(self):
        # the phone-day failure was a SILENT no-heal.  Now an unreachable
        # catalog still attempts a direct retry with the first curated id —
        # a failed direct retry is cheap, silence is not.
        p = _provider()

        def dead_catalog(timeout=20.0, catalog_url=""):
            raise OSError("network down")

        posts = []

        def fake_post_checked(url, payload):
            posts.append(("post_checked", payload["model"]))
            raise Exception("model.provider: HTTP 400 from https://router.huggingface.co/")

        def fake_post_json(url, payload):
            posts.append(("post_json", payload["model"]))
            return _fake_ok_response(payload["model"])

        with mock.patch.object(p, "_post_checked", fake_post_checked), \
             mock.patch.object(p.http, "post_json", fake_post_json), \
             mock.patch.object(HFServerlessProvider, "fetch_catalog", dead_catalog):
            resp = p.chat([Message.user("hello")])
        # direct retry with the first curated id SUCCEEDED
        self.assertEqual(resp.text, f"hi from {HEALED}")
        self.assertEqual(posts, [("post_checked", CONFIGURED),
                                 ("post_json", HEALED)])

    def test_401_never_heals(self):
        # bad credentials are not a model problem — no catalog, no retry
        p = _provider()
        catalog = _Catalog(hosted=[HEALED])
        posts = []

        def fake_post(url, payload):
            posts.append(payload["model"])
            raise Exception("HF token rejected (401/403)")

        with mock.patch.object(p, "_post_checked", fake_post), \
             mock.patch.object(HFServerlessProvider, "fetch_catalog", catalog):
            resp = p.chat([Message.user("hello")])
        self.assertTrue(resp.error)
        self.assertEqual(catalog.calls, 0)  # catalog never consulted
        self.assertEqual(len(posts), 1)
        self.assertEqual(p.model, CONFIGURED)

    def test_heal_retry_failure_restores_model(self):
        p = _provider()
        catalog = _Catalog(hosted=[HEALED])
        posts = []

        def fake_post_checked(url, payload):
            posts.append(("post_checked", payload["model"]))
            raise Exception("HTTP 400: unknown model")

        def fake_post_json(url, payload):
            posts.append(("post_json", payload["model"]))
            return _fake_bad_response()

        with mock.patch.object(p, "_post_checked", fake_post_checked), \
             mock.patch.object(p.http, "post_json", fake_post_json), \
             mock.patch.object(HFServerlessProvider, "fetch_catalog", catalog):
            resp = p.chat([Message.user("hello")])
        self.assertTrue(resp.error)
        self.assertEqual(p.model, CONFIGURED)  # restored after failed heal
        self.assertEqual(posts, [("post_checked", CONFIGURED),
                                 ("post_json", HEALED)])  # original + one heal attempt

    def test_complete_goes_through_the_same_heal(self):
        p = _provider()
        catalog = _Catalog(hosted=[HEALED])
        posts = []

        def fake_post_checked(url, payload):
            posts.append(("post_checked", payload["model"]))
            raise Exception("HTTP 400: unknown model")

        def fake_post_json(url, payload):
            posts.append(("post_json", payload["model"]))
            return _fake_ok_response(payload["model"])

        with mock.patch.object(p, "_post_checked", fake_post_checked), \
             mock.patch.object(p.http, "post_json", fake_post_json), \
             mock.patch.object(HFServerlessProvider, "fetch_catalog", catalog):
            resp = p.complete("hello")
        self.assertEqual(resp.text, f"hi from {HEALED}")
        self.assertEqual(posts, [("post_checked", CONFIGURED),
                                 ("post_json", HEALED)])

    def test_config_default_is_the_phone_family(self):
        # default stays the phone's own family; if the router rejects it,
        # the heal swaps to a catalog-verified model (tested above)
        settings = get_settings()
        self.assertEqual(settings.llm.hf_model,
                         "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2")

    def test_curated_list_is_catalog_verified(self):
        for model_id in ROUTER_FALLBACK_MODELS:
            self.assertIn("/", model_id)  # org/repo shape, not a bare name
        self.assertIn("Sao10K/L3-8B-Stheno-v3.2", ROUTER_FALLBACK_MODELS)
        self.assertIn("Qwen/Qwen3-8B", ROUTER_FALLBACK_MODELS)

    def test_discovery_prefers_small_uncensored(self):
        catalog = [
            {"id": "org/Huge-70B-Instruct",
             "providers": [{"status": "live"}]},
            {"id": "org/Small-8B-Instruct",
             "providers": [{"status": "live"}]},
            {"id": "org/abliterated-7B-Instruct",
             "providers": [{"status": "live"}]},
        ]
        found = HFServerlessProvider._discover_catalog_model(catalog, exclude="x")
        self.assertEqual(found, "org/abliterated-7B-Instruct")

    def test_discovery_never_returns_the_current_model(self):
        catalog = [
            {"id": "same/model-8B", "providers": [{"status": "live"}]},
        ]
        found = HFServerlessProvider._discover_catalog_model(catalog,
                                                             exclude="same/model-8B")
        self.assertIsNone(found)

    def test_400_message_format_variants(self):
        # the wire phrasing varies ("400 bad request" vs "HTTP 400 from")
        p = _provider()
        posts = []

        def fake_post_checked(url, payload):
            posts.append(payload["model"])
            raise Exception("model.provider: HTTP 400 from https://router.huggingface.co/")

        def fake_post_json(url, payload):
            posts.append(payload["model"])
            return _fake_ok_response(payload["model"])

        catalog = _Catalog(hosted=[HEALED])
        with mock.patch.object(p, "_post_checked", fake_post_checked), \
             mock.patch.object(p.http, "post_json", fake_post_json), \
             mock.patch.object(HFServerlessProvider, "fetch_catalog", catalog):
            resp = p.chat([Message.user("hello")])
        self.assertEqual(resp.text, f"hi from {HEALED}")


class PlainReasonTest(unittest.TestCase):
    def test_400_reads_in_plain_words(self):
        from nomorals.agents.partner_runtime import PartnerRuntime

        reason = PartnerRuntime._plain_model_reason(
            "model.provider: HTTP 400 from https://router.huggingface.co/")
        self.assertIn("not hosted", reason)
        self.assertNotIn("HTTP 400", reason)

    def test_401_still_bad_credentials(self):
        from nomorals.agents.partner_runtime import PartnerRuntime

        reason = PartnerRuntime._plain_model_reason("HTTP 401: invalid api key")
        self.assertEqual(reason, "bad credentials")


if __name__ == "__main__":
    unittest.main()
