"""Tests for audit-fix features: embedding auto-mode, delivery scoring,
adaptive threshold, autonomy default, router verification."""

import unittest

from nomorals.memory.delivery import DeliveryScorer, _tone, _tone_compatibility
from nomorals.memory.embeddings import Embedder


class EmbedderAutoModeTests(unittest.TestCase):
    def test_auto_defaults_to_hashing_without_router(self):
        e = Embedder(provider="auto", router=None)
        v = e.embed("hello world")
        self.assertEqual(len(v), 512)
        self.assertFalse(e.is_semantic)

    def test_auto_skips_mock_provider(self):
        """Auto mode should not use mock embeddings (not semantic)."""
        from nomorals.llm.router import LLMRouter
        from nomorals.llm.providers.mock import MockProvider

        router = LLMRouter()
        router.add(MockProvider(), name="mock")
        e = Embedder(provider="auto", router=router)
        # Probe should fail (mock skipped), fall back to hashing
        v = e.embed("test")
        self.assertEqual(len(v), 512)
        self.assertFalse(e._auto_works)

    def test_hashing_provider_unchanged(self):
        e = Embedder(provider="hashing")
        v1 = e.embed("hello")
        v2 = e.embed("hello")
        self.assertEqual(v1, v2)  # deterministic
        self.assertFalse(e.is_semantic)


class DeliveryScorerTests(unittest.TestCase):
    def _make_memory(self, mid, content, importance=0.5):
        m = type("M", (), {})()
        m.id = mid
        m.content = content
        m.importance = importance
        return m

    def test_topic_fit_ranks_relevant_higher(self):
        scorer = DeliveryScorer()
        memories = [
            self._make_memory("1", "Python is great for backend services"),
            self._make_memory("2", "The weather is nice today"),
        ]
        scored = scorer.score(
            memories, current_text="What programming language should I use?"
        )
        self.assertEqual(scored[0].memory_id, "1")

    def test_tone_mismatch_penalized(self):
        scorer = DeliveryScorer()
        memories = [
            self._make_memory("1", "I am so happy and excited about the party!"),
            self._make_memory("2", "I am sad and crying about the loss."),
        ]
        # Happy conversation: happy memory should outrank sad one
        scored = scorer.score(
            memories, current_text="I'm having a wonderful day!"
        )
        self.assertEqual(scored[0].memory_id, "1")

    def test_repetition_penalized(self):
        scorer = DeliveryScorer(repeat_cooldown_seconds=3600)
        m = self._make_memory("1", "Python is great")
        # First scoring: fresh
        s1 = scorer.score([m], current_text="tell me about Python")[0]
        # Mark as surfaced, score again immediately
        scorer.mark_surfaced("1")
        s2 = scorer.score([m], current_text="tell me about Python")[0]
        self.assertGreater(s1.score, s2.score)

    def test_tone_helpers(self):
        self.assertEqual(_tone("I am so happy!"), "positive")
        self.assertEqual(_tone("I am sad and crying"), "negative")
        self.assertEqual(_tone("The sky is blue"), "neutral")
        self.assertEqual(_tone_compatibility("positive", "positive"), 1.0)
        self.assertEqual(_tone_compatibility("neutral", "positive"), 1.0)
        self.assertLess(_tone_compatibility("positive", "negative"), 0.5)


class AdaptiveFilterTests(unittest.TestCase):
    def _make_record(self, content, score):
        r = type("R", (), {})()
        r.content = content
        r.score = score
        return r

    def test_filters_low_scores(self):
        from nomorals.partner.responder import _adaptive_filter
        records = [
            self._make_record("good", 0.8),
            self._make_record("bad", 0.1),
        ]
        result = _adaptive_filter(records, 5)
        self.assertIn("good", result)
        self.assertNotIn("bad", result)

    def test_returns_empty_when_nothing_relevant(self):
        from nomorals.partner.responder import _adaptive_filter
        records = [
            self._make_record("a", 0.1),
            self._make_record("b", 0.05),
        ]
        result = _adaptive_filter(records, 5)
        self.assertEqual(result, [])

    def test_keeps_cluster_near_top(self):
        from nomorals.partner.responder import _adaptive_filter
        records = [
            self._make_record("top", 0.9),
            self._make_record("good", 0.6),  # within 50% of top
            self._make_record("tail", 0.3),  # below 50% of top (0.45)
        ]
        result = _adaptive_filter(records, 5)
        self.assertIn("top", result)
        self.assertIn("good", result)
        self.assertNotIn("tail", result)


class AutonomyDefaultTests(unittest.TestCase):
    def test_default_mode_is_auto(self):
        from nomorals.agents.autonomy import AutonomyAgent
        import inspect
        sig = inspect.signature(AutonomyAgent.__init__)
        self.assertEqual(sig.parameters["mode"].default, "auto")

    def test_config_default_is_auto(self):
        from nomorals.core.config import PartnerSettings
        import dataclasses
        for f in dataclasses.fields(PartnerSettings):
            if f.name == "autonomy_mode":
                # Check default via default_factory or default
                default = f.default
                if isinstance(default, dataclasses._MISSING_TYPE):
                    self.fail("no default for autonomy_mode")
                self.assertEqual(default, "auto")


class RouterVerifyTests(unittest.TestCase):
    def test_verify_reports_mock_ok(self):
        from nomorals.llm.router import LLMRouter
        from nomorals.llm.providers.mock import MockProvider

        router = LLMRouter()
        router.add(MockProvider(), name="mock")
        report = router.verify()
        self.assertIn("mock", report["ok"])
        self.assertEqual(report["failed"], [])

    def test_verify_handles_no_providers(self):
        from nomorals.llm.router import LLMRouter

        router = LLMRouter()
        report = router.verify()
        self.assertEqual(report["ok"], [])
        # Should not raise


if __name__ == "__main__":
    unittest.main()
