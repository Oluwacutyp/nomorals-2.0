"""Venice AI provider rung (gapfix128).

Venice (venice.ai) wired into the existing provider layer as one more
OpenAI-compatible rung — no parallel provider system.  HTTP is fully
mocked; no network, no real key.  Covers: registry presence + spec
shape, offline construction + env resolution, fail-closed without a
key (plain statement naming VENICE_API_KEY, no HTTP attempted),
scripted chat/embed/vision round-trips, live catalog/traits discovery,
the defaults-chain rung (in only when the key is set, before groq),
the settings-chain branch, and STATUS.md.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from nomorals.core.errors import ProviderError
from nomorals.llm.base import Message
from nomorals.llm.failures import FailureClass, classify_failure
from nomorals.llm.providers import (
    build_provider,
    expected_kinds,
    get_provider_spec,
)
from nomorals.llm.providers.venice import (
    VENICE_BASE_URL,
    VENICE_DEFAULT_MODEL,
    VENICE_MODELS,
    VeniceProvider,
)

_ALL_KEY_ENV = (
    "GROQ_API_KEY", "NM_GROQ_API_KEY", "OPENROUTER_API_KEY",
    "NM_OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "NM_ANTHROPIC_API_KEY",
    "GEMINI_API_KEY", "NM_GEMINI_API_KEY", "DEEPSEEK_API_KEY",
    "NM_DEEPSEEK_API_KEY", "HF_TOKEN", "NM_HF_TOKEN",
    "VENICE_API_KEY", "NM_VENICE_API_KEY",
    "GROQ_MODELS", "NM_GROQ_MODELS", "OPENROUTER_MODELS",
    "NM_OPENROUTER_MODELS", "VENICE_MODELS", "NM_VENICE_MODELS",
    "VENICE_MODEL", "NM_VENICE_MODEL",
)


@contextmanager
def controlled_env(**set_vars: str):
    saved = {k: os.environ.get(k) for k in _ALL_KEY_ENV}
    for key in _ALL_KEY_ENV:
        os.environ.pop(key, None)
    os.environ.update(set_vars)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeHttp:
    """Scripted transport: post_json/get routed by URL; records calls."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def post_json(self, url, payload, **kw):
        self.calls.append(("POST", url, dict(payload)))
        return self.handler(url, payload)

    def get(self, url, **kw):
        self.calls.append(("GET", url, {}))
        return self.handler(url, None)


def _chat_ok(model, text="venice says hi"):
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3,
                  "total_tokens": 7},
    }


def _venice(api_key="sk-venice-test", **kw):
    provider = VeniceProvider(api_key=api_key, **kw)
    return provider


class RegistryShape(unittest.TestCase):
    def test_venice_is_registered(self):
        self.assertIn("venice", expected_kinds())
        spec = get_provider_spec("venice")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.id, "venice")
        self.assertEqual(spec.provider_class,
                         "nomorals.llm.providers.venice:VeniceProvider")
        self.assertEqual(spec.transport, "http-openai")
        self.assertEqual(spec.default_model, VENICE_DEFAULT_MODEL)
        self.assertEqual(spec.default_base_url, VENICE_BASE_URL)
        self.assertEqual(spec.models_hook, "module:VENICE_MODELS")

    def test_auth_shape_names_env_only(self):
        spec = get_provider_spec("venice")
        self.assertEqual(spec.auth.kind, "bearer")
        self.assertEqual(spec.auth.env_keys,
                         ("VENICE_API_KEY", "NM_VENICE_API_KEY"))
        self.assertEqual(spec.auth.kwarg, "api_key")
        self.assertFalse(spec.auth.required)

    def test_capabilities_match_instance(self):
        spec = get_provider_spec("venice")
        provider = build_provider("venice", api_key="sk-test")
        self.assertEqual(set(provider.capabilities),
                         set(spec.capabilities))
        self.assertEqual(set(provider.capabilities),
                         {"chat", "complete", "embed", "vision"})

    def test_roster_resolves_offline(self):
        spec = get_provider_spec("venice")
        roster = spec.roster()
        self.assertEqual(roster, [m["id"] for m in VENICE_MODELS])
        self.assertGreaterEqual(len(roster), 2)
        self.assertEqual(roster[0], VENICE_DEFAULT_MODEL)

    def test_verification_is_docs_not_live(self):
        spec = get_provider_spec("venice")
        self.assertEqual(spec.verification.status, "docs")
        self.assertEqual(spec.verification.when, "2026-11-24")

    def test_status_md_has_venice_row(self):
        path = (Path(__file__).resolve().parent.parent
                / "nomorals" / "llm" / "STATUS.md")
        text = path.read_text(encoding="utf-8")
        self.assertIn("| venice |", text)
        self.assertIn(VENICE_DEFAULT_MODEL, text)


class Construction(unittest.TestCase):
    def test_defaults_pinned(self):
        with controlled_env():
            provider = build_provider("venice", api_key="sk-test")
        self.assertIsInstance(provider, VeniceProvider)
        self.assertEqual(provider.name, "venice")
        self.assertEqual(provider.model, VENICE_DEFAULT_MODEL)
        self.assertEqual(provider.base_url, VENICE_BASE_URL)

    def test_env_key_resolution(self):
        with controlled_env(VENICE_API_KEY="sk-bare"):
            provider = build_provider("venice")
        self.assertEqual(provider.api_key, "sk-bare")
        with controlled_env(NM_VENICE_API_KEY="sk-nm"):
            provider = build_provider("venice")
        self.assertEqual(provider.api_key, "sk-nm")
        with controlled_env(VENICE_API_KEY="sk-bare",
                            NM_VENICE_API_KEY="sk-nm"):
            provider = build_provider("venice")
        self.assertEqual(provider.api_key, "sk-bare")

    def test_renamed_flagship_heals(self):
        provider = _venice(model="venice-uncensored")
        self.assertEqual(provider.model, VENICE_DEFAULT_MODEL)

    def test_model_chain_starts_with_configured_model(self):
        provider = _venice(model="llama-3.3-70b")
        chain = provider._model_chain()  # noqa: SLF001
        self.assertEqual(chain[0], "llama-3.3-70b")
        self.assertEqual(len(chain), len(set(chain)))


class FailClosedWithoutKey(unittest.TestCase):
    def test_chat_names_the_missing_key_and_sends_nothing(self):
        with controlled_env():
            provider = build_provider("venice")
        provider.http = FakeHttp(lambda url, payload: (_ for _ in ()).throw(
            AssertionError("no HTTP must be attempted without a key")))
        with self.assertRaises(ProviderError) as ctx:
            provider.chat([Message.user("hello")])
        self.assertIn("VENICE_API_KEY", str(ctx.exception))
        self.assertEqual(
            classify_failure(str(ctx.exception)).failure_class,
            FailureClass.AUTH)
        self.assertEqual(provider.http.calls, [])

    def test_embed_and_vision_fail_closed_too(self):
        with controlled_env():
            provider = build_provider("venice")
        with self.assertRaises(ProviderError) as ctx:
            provider.embed(["hello"])
        self.assertIn("VENICE_API_KEY", str(ctx.exception))
        with self.assertRaises(ProviderError) as ctx:
            provider.describe_image(b"bytes")
        self.assertIn("VENICE_API_KEY", str(ctx.exception))

    def test_health_detail_names_the_missing_key(self):
        with controlled_env():
            provider = build_provider("venice")
        ok, reason = provider.health_detail()
        self.assertFalse(ok)
        self.assertIn("VENICE_API_KEY", reason)


class ScriptedRoundTrip(unittest.TestCase):
    def test_chat_round_trip_and_bearer(self):
        provider = _venice()
        self.assertEqual(provider.http.headers["Authorization"],
                         "Bearer sk-venice-test")
        provider.http = FakeHttp(
            lambda url, payload: FakeResponse(_chat_ok(payload["model"])))
        resp = provider.chat([Message.user("hello")])
        self.assertTrue(resp.ok)
        self.assertEqual(resp.text, "venice says hi")
        self.assertEqual(resp.usage.total_tokens, 7)
        self.assertEqual(resp.provider, "venice")
        self.assertEqual(provider.http.calls[0][1],
                         VENICE_BASE_URL + "/chat/completions")
        self.assertEqual(provider.http.calls[0][2]["model"],
                         VENICE_DEFAULT_MODEL)

    def test_dead_model_walks_the_roster(self):
        roster = [m["id"] for m in VENICE_MODELS]

        def handler(url, payload):
            if payload["model"] == roster[0]:
                raise ProviderError("404 model_not_found",
                                    details={"http_status": 404})
            return FakeResponse(_chat_ok(payload["model"]))

        provider = _venice(model=roster[0])
        provider.http = FakeHttp(handler)
        resp = provider.chat([Message.user("hello")])
        self.assertTrue(resp.ok)
        attempted = [c[2]["model"] for c in provider.http.calls]
        self.assertEqual(attempted, [roster[0], roster[1]])
        self.assertEqual(resp.model, roster[1])
        self.assertEqual(provider.model, roster[1])

    def test_auth_error_is_typed_and_names_nothing_secret(self):
        sentinel = "sk-venice-sentinel-s3cr3t"
        provider = _venice(api_key=sentinel)
        provider.http = FakeHttp(
            lambda url, payload: (_ for _ in ()).throw(
                ProviderError("401 unauthorized: invalid api key",
                              details={"http_status": 401})))
        resp = provider.chat([Message.user("hello")])
        self.assertFalse(resp.ok)
        self.assertEqual(
            classify_failure(resp.error).failure_class, FailureClass.AUTH)
        self.assertNotIn(sentinel, resp.error)
        self.assertNotIn(sentinel, resp.text)

    def test_embed_round_trip(self):
        provider = _venice()
        provider.http = FakeHttp(
            lambda url, payload: FakeResponse(
                {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}))
        vectors = provider.embed(["hello"])
        self.assertEqual(vectors, [[0.1, 0.2]])

    def test_vision_round_trip(self):
        provider = _venice()
        provider.http = FakeHttp(
            lambda url, payload: FakeResponse(_chat_ok(payload["model"],
                                                       "a cat")))
        resp = provider.describe_image(b"fake-png-bytes", "what is this?")
        self.assertTrue(resp.ok)
        self.assertEqual(resp.text, "a cat")


class LiveDiscovery(unittest.TestCase):
    def test_catalog_returns_live_roster(self):
        provider = _venice()
        provider.http = FakeHttp(
            lambda url, payload: FakeResponse(
                {"data": [{"id": "venice-uncensored-1-2"},
                          {"id": "llama-3.3-70b"}]}))
        models = provider.catalog()
        self.assertEqual([m["id"] for m in models],
                         ["venice-uncensored-1-2", "llama-3.3-70b"])
        self.assertEqual(provider.http.calls[0][1],
                         VENICE_BASE_URL + "/models")

    def test_traits_map_and_default(self):
        provider = _venice()

        def handler(url, payload):
            self.assertIn("type=text", url)
            return FakeResponse({"data": {
                "default": "venice-uncensored-1-2",
                "most_uncensored": "venice-uncensored-1-2",
                "default_code": "qwen3-235b-a22b-instruct-2507",
            }})

        provider.http = FakeHttp(handler)
        traits = provider.traits()
        self.assertEqual(traits["default"], "venice-uncensored-1-2")
        self.assertEqual(provider.default_model_id(),
                         "venice-uncensored-1-2")

    def test_default_model_id_falls_back_when_traits_fail(self):
        provider = _venice()
        provider.http = FakeHttp(
            lambda url, payload: (_ for _ in ()).throw(
                ProviderError("503 service unavailable",
                              details={"http_status": 503})))
        self.assertEqual(provider.default_model_id(), VENICE_DEFAULT_MODEL)


class DefaultsChain(unittest.TestCase):
    def test_venice_in_chain_only_with_key_before_groq(self):
        from nomorals.llm.defaults import specs_from_env

        env = {"VENICE_API_KEY": "v", "GROQ_API_KEY": "g"}
        with mock.patch.dict(os.environ, env, clear=True):
            names = [s.name for s in specs_from_env()]
        self.assertIn("venice", names)
        self.assertLess(names.index("venice"), names.index("groq"))
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "g"}, clear=True):
            names = [s.name for s in specs_from_env()]
        self.assertNotIn("venice", names)

    def test_venice_model_and_fallback_env_reach_the_provider(self):
        from nomorals.llm.defaults import build_chain

        env = {"VENICE_API_KEY": "v",
               "VENICE_MODEL": "llama-3.3-70b",
               "VENICE_MODELS": "qwen3-235b-a22b-instruct-2507"}
        with mock.patch.dict(os.environ, env, clear=True):
            router = build_chain()
        provider = router.get("venice")
        self.assertIsInstance(provider, VeniceProvider)
        self.assertEqual(provider.model, "llama-3.3-70b")
        self.assertEqual(provider._model_chain(),  # noqa: SLF001
                         ["llama-3.3-70b",
                          "qwen3-235b-a22b-instruct-2507"])


class SettingsChain(unittest.TestCase):
    def _settings(self, **llm_overrides):
        from nomorals.core.config import Settings

        home = Path(tempfile.mkdtemp(
            dir=os.environ.get("NM_TEST_TMPDIR") or None))
        settings = Settings(home=home)
        for key, value in llm_overrides.items():
            setattr(settings.llm, key, value)
        return settings

    def test_settings_branch_builds_venice(self):
        from nomorals.agents.context import build_router
        from nomorals.core.events import EventBus

        settings = self._settings(provider="venice",
                                  venice_api_key="venice-fake",
                                  venice_model="llama-3.3-70b")
        with controlled_env():
            router = build_router(settings, EventBus())
        provider = router.get("venice")
        self.assertIsInstance(provider, VeniceProvider)
        self.assertEqual(provider.model, "llama-3.3-70b")
        self.assertEqual(provider.api_key, "venice-fake")

    def test_settings_skips_venice_without_key(self):
        from nomorals.agents.context import build_router
        from nomorals.core.events import EventBus

        settings = self._settings(provider="mock",
                                  fallback_chain=["venice"])
        with controlled_env():
            router = build_router(settings, EventBus())
        self.assertNotIn("venice", router.providers())
        self.assertIn("mock", router.providers())

    def test_env_map_wires_nm_venice_key(self):
        from nomorals.core.config import load_settings

        settings = load_settings(
            env={"NM_VENICE_API_KEY": "sk-nm-env",
                 "NM_VENICE_MODEL": "llama-3.3-70b"},
            use_env_file=False,
        )
        self.assertEqual(settings.llm.venice_api_key, "sk-nm-env")
        self.assertEqual(settings.llm.venice_model, "llama-3.3-70b")


class NoCredentialLeak(unittest.TestCase):
    def test_sentinel_never_in_spec_or_status(self):
        from nomorals.llm.providers import render_status_markdown

        sentinel = "sk-venice-sentinel-s3cr3t"
        spec = get_provider_spec("venice")
        blob = json.dumps(spec.to_dict())
        self.assertNotIn(sentinel, blob)
        self.assertNotIn(sentinel, render_status_markdown())
        provider = build_provider("venice", api_key=sentinel)
        provider.http = FakeHttp(
            lambda url, payload: FakeResponse(_chat_ok(payload["model"])))
        resp = provider.chat([Message.user("hello")])
        self.assertNotIn(sentinel, resp.text)
        self.assertNotIn(sentinel, json.dumps(resp.to_dict()))


if __name__ == "__main__":
    unittest.main()
