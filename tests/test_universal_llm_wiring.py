"""Universal wave — provider wiring, default chain, and broker routing.

Covers: build_provider kind mapping, the free/local-first default chain,
broker card sync (local + free-tier flags so capability routing sees them),
prefer_local selection, and router failover across the new providers.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from nomorals.llm.base import LLMResponse, Message
from nomorals.llm.broker import ModelBroker
from nomorals.llm.capabilities import Capability
from nomorals.llm.defaults import (
    CARD_HINTS,
    ProviderSpec,
    build_chain,
    specs_from_env,
    sync_broker_cards,
)
from nomorals.llm.providers import build_provider
from nomorals.llm.providers.groq import GroqProvider
from nomorals.llm.providers.ollama import OllamaProvider
from nomorals.llm.providers.openai_compat import OpenAICompatProvider
from nomorals.llm.providers.openrouter import OpenRouterProvider


class BuildProviderTests(unittest.TestCase):
    def test_ollama_is_native(self):
        provider = build_provider("ollama", model="llama3.1")
        self.assertIsInstance(provider, OllamaProvider)
        self.assertIsInstance(build_provider("ollama_native"), OllamaProvider)

    def test_groq_is_dedicated(self):
        provider = build_provider("groq")
        self.assertIsInstance(provider, GroqProvider)

    def test_openrouter_is_dedicated(self):
        provider = build_provider("openrouter")
        self.assertIsInstance(provider, OpenRouterProvider)

    def test_generic_kinds_still_openai_compat(self):
        for kind in ("openai", "openai_compat", "vllm", "lmstudio"):
            provider = build_provider(kind, base_url="http://x:8000/v1")
            self.assertIsInstance(provider, OpenAICompatProvider)

    def test_ollama_compat_alias_keeps_old_behavior(self):
        provider = build_provider("ollama_compat", base_url="http://x:11434/v1")
        self.assertIsInstance(provider, OpenAICompatProvider)
        self.assertNotIsInstance(provider, OllamaProvider)

    def test_unknown_kind_still_rejected(self):
        with self.assertRaises(ValueError):
            build_provider("definitely_not_a_provider")

    def test_shared_wiring_kwargs_still_fit(self):
        # agents/context.py passes base_url/api_key/model/timeout for these kinds.
        provider = build_provider(
            "groq", base_url="https://api.groq.com/openai/v1",
            api_key="gsk-x", model="openai/gpt-oss-120b", timeout=60.0,
        )
        self.assertEqual(provider.model, "openai/gpt-oss-120b")
        self.assertEqual(provider.timeout, 60.0)


class SpecsFromEnvTests(unittest.TestCase):
    def clean_env(self):
        return mock.patch.dict(
            os.environ,
            {"GROQ_API_KEY": "", "OPENROUTER_API_KEY": "", "HF_TOKEN": ""},
            clear=False,
        )

    def test_local_providers_always_available(self):
        with self.clean_env():
            specs = specs_from_env()
        names = [s.name for s in specs]
        self.assertEqual(names[:2], ["ollama", "llama_cpp"])
        self.assertNotIn("groq", names)
        self.assertNotIn("openrouter", names)

    def test_keyed_providers_appear_when_keys_set(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-x",
                                          "OPENROUTER_API_KEY": "sk-or-x",
                                          "HF_TOKEN": "hf-x"}, clear=False):
            specs = specs_from_env()
        names = [s.name for s in specs]
        self.assertEqual(names,
                         ["ollama", "llama_cpp", "groq", "openrouter", "hf_serverless"])

    def test_spec_availability(self):
        self.assertTrue(ProviderSpec("ollama", "ollama", local=True).available())
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            self.assertFalse(ProviderSpec("g", "groq", env_key="GROQ_API_KEY").available())
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-x"}, clear=False):
            self.assertTrue(ProviderSpec("g", "groq", env_key="GROQ_API_KEY").available())


class BuildChainTests(unittest.TestCase):
    def test_chain_builds_with_ollama_primary(self):
        specs = [
            ProviderSpec("ollama", "ollama", local=True,
                         kwargs={"model": "llama3.1"}),
            ProviderSpec("groq", "groq", kwargs={"model": "openai/gpt-oss-20b"}),
        ]
        router = build_chain(specs)
        self.assertEqual(router.providers(), ["ollama", "groq"])
        self.assertEqual(router.active, "ollama")
        self.assertIsInstance(router.get("ollama"), OllamaProvider)
        self.assertIsInstance(router.get("groq"), GroqProvider)

    def test_bad_backend_does_not_kill_chain(self):
        specs = [
            ProviderSpec("bogus", "no_such_kind"),
            ProviderSpec("ollama", "ollama", local=True),
        ]
        router = build_chain(specs)
        self.assertEqual(router.providers(), ["ollama"])


class BrokerSyncTests(unittest.TestCase):
    def make_router(self):
        return build_chain([
            ProviderSpec("ollama", "ollama", local=True,
                         kwargs={"model": "llama3.1"}),
            ProviderSpec("groq", "groq",
                         kwargs={"model": "openai/gpt-oss-120b"}),
        ])

    def test_sync_registers_cards_with_local_and_cost_hints(self):
        router = self.make_router()
        broker = ModelBroker()
        cards = sync_broker_cards(broker, router)
        by_id = {c.id: c for c in cards}
        self.assertTrue(by_id["ollama"].local)
        self.assertFalse(by_id["groq"].local)
        self.assertEqual(by_id["ollama"].cost_per_1k, 0.0)
        self.assertEqual(by_id["groq"].cost_per_1k, 0.0)
        self.assertEqual(by_id["groq"].context_len, 131072)
        self.assertIn(Capability.CHAT, by_id["ollama"].capabilities)

    def test_prefer_local_selects_ollama_for_chat(self):
        router = self.make_router()
        broker = ModelBroker()
        sync_broker_cards(broker, router)
        winner = broker.select("chat", constraints={"prefer_local": True})
        self.assertIsNotNone(winner)
        self.assertEqual(winner.id, "ollama")

    def test_local_only_excludes_cloud(self):
        router = self.make_router()
        broker = ModelBroker()
        sync_broker_cards(broker, router)
        winner = broker.select("chat", constraints={"local_only": True})
        self.assertEqual(winner.id, "ollama")

    def test_sync_is_additive(self):
        router = self.make_router()
        broker = ModelBroker()
        sync_broker_cards(broker, router)
        before = len(broker.cards())
        sync_broker_cards(broker, router)
        self.assertEqual(len(broker.cards()), before)

    def test_card_hints_cover_all_default_specs(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            specs = specs_from_env()
        for spec in specs:
            self.assertIn(spec.name, CARD_HINTS,
                          f"spec {spec.name!r} has no CARD_HINTS entry")


class RouterFailoverTests(unittest.TestCase):
    def test_failover_across_new_providers(self):
        ollama = build_provider("ollama", model="llama3.1", max_retries=1)
        groq = build_provider("groq", model="openai/gpt-oss-20b", max_retries=1)

        class BoomHttp:
            def post_json(self, url, payload, **kw):
                raise ConnectionError("ollama is down")

        class OkHttp:
            def __init__(self):
                self.calls = []

            def post_json(self, url, payload, **kw):
                self.calls.append((url, payload))
                from types import SimpleNamespace

                return SimpleNamespace(
                    ok=True,
                    json=lambda: {
                        "choices": [{"message": {"content": "via groq"},
                                     "finish_reason": "stop"}],
                        "model": "openai/gpt-oss-20b",
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2,
                                  "total_tokens": 3},
                    },
                )

        ollama.http = BoomHttp()
        groq_http = OkHttp()
        groq.http = groq_http

        from nomorals.llm.router import LLMRouter

        router = LLMRouter(cooldown_seconds=0)
        router.add(ollama, primary=True, name="ollama")
        router.add(groq, name="groq")

        response = router.chat([Message.user("hello")])
        self.assertTrue(response.ok)
        self.assertEqual(response.text, "via groq")
        self.assertTrue(response.degraded)
        self.assertEqual(response.failed_providers, ["ollama"])
        self.assertIn("ollama", response.fallback_note)
        # groq got the OpenAI-shaped request at its own base URL
        url, payload = groq_http.calls[0]
        self.assertTrue(url.startswith("https://api.groq.com/openai/v1"))
        self.assertEqual(payload["model"], "openai/gpt-oss-20b")

    def test_embed_skips_groq_for_lack_of_capability(self):
        from nomorals.llm.router import LLMRouter

        groq = build_provider("groq", model="x")
        router = LLMRouter()
        router.add(groq, primary=True, name="groq")
        # groq advertises no "embed" token: router must refuse instead of 404ing.
        with self.assertRaises(Exception) as ctx:
            router.embed(["hello"])
        self.assertIn("embed", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
