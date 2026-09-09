"""L3 — model plane: providers, router failover, registry, promotion gate.

Runs fully offline. The mock provider is deterministic, so these assertions are
about behaviour, not about a model being smart.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.core.errors import NotFound, ValidationError
from nomorals.llm.base import (
    LLMResponse,
    Message,
    SamplingParams,
    estimate_messages,
    estimate_tokens,
    messages_to_text,
)
from nomorals.llm.providers import build_provider
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.registry import MODEL_CATALOG, ModelRegistry, search_catalog
from nomorals.llm.router import LLMRouter
from nomorals.storage.db import Database


def temp_db() -> Database:
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    db = Database(handle.name)
    db.migrate()
    return db


class FailingProvider(MockProvider):
    """Always errors, so the router is forced to fail over."""

    def chat(self, messages, params=None, **kw):
        return LLMResponse(
            text="",
            provider=self.name,
            model=self.model,
            error="synthetic upstream failure",
        )


class MessageTests(unittest.TestCase):
    def test_roles_and_openai_shape(self):
        messages = [Message.system("s"), Message.user("u"), Message.assistant("a")]
        self.assertEqual([m.role for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[1].to_openai(), {"role": "user", "content": "u"})

    def test_chatml_template_marks_roles(self):
        text = messages_to_text([Message.system("s"), Message.user("u")])
        self.assertIn("s", text)
        self.assertIn("u", text)
        self.assertGreater(len(text), 3)

    def test_token_estimates_are_monotonic(self):
        self.assertEqual(estimate_tokens(""), 0)
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world " * 50)
        self.assertGreater(long, short)
        self.assertEqual(estimate_messages([]), 0)


class SamplingTests(unittest.TestCase):
    def test_out_of_range_values_are_clamped_not_rejected(self):
        params = SamplingParams(temperature=99.0, max_tokens=-5, top_p=4.0).clamped()
        self.assertLessEqual(params.temperature, 2.0)
        self.assertGreaterEqual(params.max_tokens, 1)
        self.assertLessEqual(params.top_p, 1.0)


class MockProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = MockProvider()

    def test_deterministic_for_identical_input(self):
        messages = [Message.user("tell me about recursion")]
        first = self.provider.chat(messages)
        second = self.provider.chat(messages)
        self.assertTrue(first.ok)
        self.assertEqual(first.text, second.text)

    def test_different_prompts_give_different_text(self):
        a = self.provider.chat([Message.user("alpha topic")])
        b = self.provider.chat([Message.user("completely different beta subject")])
        self.assertNotEqual(a.text, b.text)

    def test_max_tokens_is_respected(self):
        response = self.provider.chat(
            [Message.user("write a long essay")], SamplingParams(max_tokens=8)
        )
        self.assertLessEqual(estimate_tokens(response.text), 12)

    def test_embed_is_normalized_and_stable(self):
        vectors = self.provider.embed(["same text", "same text", "other words"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(vectors[0], vectors[1])
        self.assertNotEqual(vectors[0], vectors[2])
        magnitude = sum(x * x for x in vectors[0]) ** 0.5
        self.assertAlmostEqual(magnitude, 1.0, places=4)

    def test_describe_image_sniffs_dimensions(self):
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            + (320).to_bytes(4, "big")
            + (240).to_bytes(4, "big")
            + b"\x08\x02\x00\x00\x00"
        )
        response = self.provider.describe_image(png, "what is this")
        self.assertTrue(response.ok)
        self.assertIn("320", response.text)
        self.assertIn("240", response.text)

    def test_health_and_capabilities(self):
        self.assertTrue(self.provider.health())
        self.assertIn("chat", self.provider.capabilities)
        self.assertIn("embed", self.provider.capabilities)


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.router = LLMRouter()
        self.router.add(MockProvider(model="primary"), primary=True, name="primary")
        self.router.add(MockProvider(model="backup"), name="backup")

    def test_add_and_list(self):
        self.assertEqual(self.router.providers(), ["primary", "backup"])
        self.assertEqual(self.router.active, "primary")

    def test_chat_uses_the_active_provider(self):
        response = self.router.chat([Message.user("hi")])
        self.assertTrue(response.ok)
        self.assertEqual(response.provider, "primary")

    def test_hot_swap_changes_which_model_answers(self):
        self.router.set_active("backup")
        self.assertEqual(self.router.active, "backup")
        self.assertEqual(self.router.chat([Message.user("hi")]).provider, "backup")

    def test_hot_swap_to_unknown_provider_is_rejected(self):
        with self.assertRaises((NotFound, ValidationError, KeyError)):
            self.router.set_active("nope")

    def test_failover_when_primary_errors(self):
        router = LLMRouter()
        router.add(FailingProvider(model="broken"), primary=True, name="broken")
        router.add(MockProvider(model="healthy"), name="healthy")
        response = router.chat([Message.user("hi")])
        self.assertTrue(response.ok, response.error)
        self.assertEqual(response.provider, "healthy")

    def test_repeated_failures_cool_a_provider_down(self):
        router = LLMRouter(failure_threshold=1, cooldown_seconds=60.0)
        router.add(FailingProvider(model="broken"), primary=True, name="broken")
        router.add(MockProvider(model="healthy"), name="healthy")
        router.chat([Message.user("hi")])
        health = router.stats_snapshot()["health"]["broken"]
        self.assertGreaterEqual(health["failures"], 1)
        self.assertTrue(health["cooling_down"])

    def test_reset_cooldowns_makes_a_provider_available_again(self):
        router = LLMRouter(failure_threshold=1, cooldown_seconds=3600.0)
        router.add(FailingProvider(model="broken"), primary=True, name="broken")
        router.add(MockProvider(model="healthy"), name="healthy")
        router.chat([Message.user("hi")])
        self.assertTrue(router.stats_snapshot()["health"]["broken"]["cooling_down"])
        router.reset_cooldowns()
        self.assertFalse(router.stats_snapshot()["health"]["broken"]["cooling_down"])

    def test_all_providers_failing_returns_an_error_not_an_exception(self):
        router = LLMRouter()
        router.add(FailingProvider(model="a"), primary=True, name="a")
        router.add(FailingProvider(model="b"), name="b")
        response = router.chat([Message.user("hi")])
        self.assertFalse(response.ok)
        self.assertTrue(response.error)

    def test_embed_routes_to_a_provider(self):
        vectors = self.router.embed(["one", "two"])
        self.assertEqual(len(vectors), 2)

    def test_empty_router_does_not_explode(self):
        router = LLMRouter()
        response = router.chat([Message.user("hi")])
        self.assertFalse(response.ok)


class BuildProviderTests(unittest.TestCase):
    def test_mock_is_always_available(self):
        provider = build_provider("mock", model="x")
        self.assertTrue(provider.chat([Message.user("hi")]).ok)

    def test_unknown_backend_raises(self):
        with self.assertRaises(Exception):
            build_provider("definitely-not-a-provider")


class CatalogTests(unittest.TestCase):
    def test_catalog_is_populated_and_consistent(self):
        self.assertGreaterEqual(len(MODEL_CATALOG), 10)
        for entry in MODEL_CATALOG:
            self.assertTrue(entry.repo_id)
            self.assertIn("/", entry.repo_id)
            self.assertGreater(entry.params, 0)
            self.assertGreater(entry.context_length, 0)

    def test_search_filters_by_kind_and_size(self):
        small = search_catalog("", kind="instruct", max_params=9_000_000_000)
        self.assertTrue(small)
        for entry in small:
            self.assertLessEqual(entry.params, 9_000_000_000)

    def test_text_search_matches_a_family(self):
        hits = search_catalog("dolphin")
        self.assertTrue(any("dolphin" in e.repo_id.lower() for e in hits))

    def test_size_hint_scales_with_parameters(self):
        entry = search_catalog("")[0]
        self.assertGreater(entry.size_hint_gb, 0)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.db = temp_db()
        self.registry = ModelRegistry(self.db)

    def tearDown(self):
        self.db.close()
        Path(self.db.path).unlink(missing_ok=True)

    def test_register_requires_a_name(self):
        with self.assertRaises(ValidationError):
            self.registry.register("")

    def test_register_and_retrieve(self):
        record = self.registry.register("local-7b", kind="foundation", path="models/7b.gguf")
        self.assertTrue(record.id)
        self.assertEqual(self.registry.by_name("local-7b").id, record.id)

    def test_only_one_model_is_active_at_a_time(self):
        self.registry.register("a", activate=True)
        self.registry.register("b", activate=True)
        active = self.registry.list(active_only=True)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].name, "b")

    def test_promotion_gate_blocks_a_worse_candidate(self):
        self.registry.register("incumbent", activate=True)
        self.registry.record_eval("incumbent", {"score": 0.80})
        self.registry.register("candidate")
        self.registry.record_eval("candidate", {"score": 0.55})
        self.assertFalse(self.registry.beats_incumbent("candidate"))

    def test_promotion_gate_allows_a_better_candidate(self):
        self.registry.register("incumbent", activate=True)
        self.registry.record_eval("incumbent", {"score": 0.80})
        self.registry.register("candidate")
        self.registry.record_eval("candidate", {"score": 0.91})
        self.assertTrue(self.registry.beats_incumbent("candidate"))

    def test_promotion_gate_honours_the_tolerance(self):
        self.registry.register("incumbent", activate=True)
        self.registry.record_eval("incumbent", {"score": 0.80})
        self.registry.register("candidate")
        self.registry.record_eval("candidate", {"score": 0.79})
        self.assertFalse(self.registry.beats_incumbent("candidate"))
        self.assertTrue(self.registry.beats_incumbent("candidate", tolerance=0.02))

    def test_a_missing_metric_never_promotes(self):
        self.registry.register("incumbent", activate=True)
        self.registry.record_eval("incumbent", {"score": 0.80})
        self.registry.register("candidate")
        self.registry.record_eval("candidate", {"other": 0.99})
        self.assertFalse(self.registry.beats_incumbent("candidate"))

    def test_first_model_is_promotable_with_no_incumbent(self):
        self.registry.register("solo")
        self.registry.record_eval("solo", {"score": 0.1})
        self.assertTrue(self.registry.beats_incumbent("solo"))

    def test_finetune_records_its_lineage(self):
        self.registry.register("base")
        record = self.registry.register_finetune(
            "ft-1", base_model="base", output_path="models/ft1", eval_scores={"score": 0.7}
        )
        self.assertEqual(record.kind, "finetune")
        self.assertEqual(self.registry.by_name("ft-1").base_model, "base")

    def test_stats_counts(self):
        self.registry.register("base")
        self.registry.register_finetune("ft", base_model="base", output_path="p")
        stats = self.registry.stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["finetunes"], 1)


if __name__ == "__main__":
    unittest.main()
