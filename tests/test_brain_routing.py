"""Brain/LLM routing: failover, circuit breaker, task-aware selection,
owner-model preference, honest errors, never-raises.

Covers the routing improvements in nomorals/llm/{router,broker,brain,
capabilities,defaults}.py.  All offline with mock providers.
"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

from nomorals.core.retry import CircuitBreaker
from nomorals.llm.base import LLMResponse, Message
from nomorals.llm.brain import Brain, explain_failure, get_brain, reset_brain
from nomorals.llm.broker import BrokerConstraints, ModelBroker
from nomorals.llm.capabilities import Capability, ModelCard
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.router import LLMRouter


class _Boom(MockProvider):
    """Always explodes."""

    def chat(self, messages, params=None, **kw):
        raise RuntimeError("chat boom")


class _Flaky(MockProvider):
    """Fails until told otherwise."""

    failing = True

    def chat(self, messages, params=None, **kw):
        if self.failing:
            raise RuntimeError("flaky boom")
        return super().chat(messages, params, **kw)


class _VisionMock(MockProvider):
    """Mock with a vision capability."""

    @property
    def capabilities(self):
        return {"chat", "complete", "vision"}

    def describe_image(self, image, prompt="", params=None, **kw):
        return LLMResponse(text="[vision] saw it", model=self.model_id,
                           provider=self.name)


class _Clock:
    """Controllable monotonic clock for circuit-breaker tests."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _router(clock=None, **kw):
    kw.setdefault("cooldown_seconds", 60.0)
    kw.setdefault("failure_threshold", 2)
    if clock is not None:
        kw["clock"] = clock
    return LLMRouter(**kw)


# ── failover ───────────────────────────────────────────────────────────────

class FailoverTests(unittest.TestCase):
    def test_failover_serves_and_reports_degradation(self):
        r = _router()
        r.add(_Boom(model="bad"), name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        resp = r.chat([Message.user("hi")])
        self.assertTrue(resp.ok)
        self.assertTrue(resp.degraded)
        self.assertEqual(resp.failed_providers, ["bad"])
        self.assertIn("bad", resp.fallback_note)
        self.assertIn("good", resp.fallback_note)

    def test_all_fail_returns_honest_error_not_raise(self):
        r = _router()
        r.add(_Boom(model="b1"), name="b1", primary=True)
        r.add(_Boom(model="b2"), name="b2")
        resp = r.chat([Message.user("hi")])
        self.assertFalse(resp.ok)
        self.assertTrue(resp.error)
        self.assertEqual(resp.failed_providers, ["b1", "b2"])
        self.assertIn("b1", resp.fallback_note)
        self.assertIn("b2", resp.fallback_note)


# ── circuit breaker ────────────────────────────────────────────────────────

class CircuitBreakerTests(unittest.TestCase):
    def test_cooldown_after_threshold_and_skip(self):
        clock = _Clock()
        r = _router(clock)
        r.add(_Boom(model="bad"), name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        r.chat([Message.user("a")])  # bad fails (1)
        r.chat([Message.user("b")])  # bad fails (2) → threshold → cooling
        self.assertTrue(r.is_cooling_down("bad"))
        failures = r._health["bad"].failures
        r.chat([Message.user("c")])  # bad skipped, no new failure recorded
        self.assertEqual(r._health["bad"].failures, failures)
        self.assertGreater(r.cooldown_remaining("bad"), 0)

    def test_backoff_grows_with_consecutive_failures(self):
        clock = _Clock()
        r = _router(clock)
        r.add(_Boom(model="bad"), name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        r.chat([Message.user("a")])
        r.chat([Message.user("b")])  # cooling now, ~60s (jitter ±10%)
        first = r.cooldown_remaining("bad")
        self.assertGreater(first, 40)
        self.assertLess(first, 80)
        clock.advance(first + 5)  # cooldown expired
        self.assertFalse(r.is_cooling_down("bad"))
        r.chat([Message.user("c")])  # probe admitted, fails → backoff doubles
        second = r.cooldown_remaining("bad")
        self.assertGreater(second, 100)  # ~120s ± jitter, clearly > first
        self.assertLess(second, 150)

    def test_backoff_capped(self):
        clock = _Clock()
        r = _router(clock, cooldown_seconds=60.0, max_cooldown_seconds=100.0)
        r.add(_Boom(model="bad"), name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        for i in range(6):
            r.chat([Message.user(f"m{i}")])
            clock.advance(r.cooldown_remaining("bad") + 1)
        self.assertLessEqual(r.cooldown_remaining("bad"), 115)  # cap + jitter

    def test_rate_limit_cools_down_immediately(self):
        clock = _Clock()
        r = _router(clock, failure_threshold=5)
        r.add(_Boom(model="bad"), name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        r._note_failure("bad", "429 rate limit exceeded")
        self.assertTrue(r.is_cooling_down("bad"))
        # well below the normal 5-failure threshold — 429 skips the queue
        self.assertEqual(r._health["bad"].consecutive_failures, 1)

    def test_half_open_probe_recovers(self):
        clock = _Clock()
        flaky = _Flaky(model="flaky")
        r = _router(clock)
        r.add(flaky, name="flaky", primary=True)
        r.add(MockProvider(model="good"), name="good")
        r.chat([Message.user("a")])
        r.chat([Message.user("b")])  # cooling
        self.assertTrue(r.is_cooling_down("bad") is False)  # unknown name
        self.assertTrue(r.is_cooling_down("flaky"))
        clock.advance(r.cooldown_remaining("flaky") + 1)
        flaky.failing = False
        resp = r.chat([Message.user("c")])  # half-open probe admitted
        self.assertTrue(resp.ok)
        self.assertFalse(resp.degraded)  # served by primary, no failover
        self.assertEqual(r._health["flaky"].consecutive_failures, 0)
        self.assertFalse(r.is_cooling_down("flaky"))

    def test_breaker_state_visible_in_snapshot(self):
        clock = _Clock()
        r = _router(clock)
        r.add(_Boom(model="bad"), name="bad", primary=True)
        r.chat([Message.user("a")])
        r.chat([Message.user("b")])
        snap = r.stats_snapshot()
        self.assertEqual(snap["health"]["bad"]["breaker_state"],
                         CircuitBreaker.OPEN)

    def test_reset_cooldowns_resets_breaker(self):
        clock = _Clock()
        r = _router(clock)
        r.add(_Boom(model="bad"), name="bad", primary=True)
        r.add(MockProvider(model="good"), name="good")
        r.chat([Message.user("a")])
        r.chat([Message.user("b")])
        self.assertTrue(r.is_cooling_down("bad"))
        r.reset_cooldowns()
        self.assertFalse(r.is_cooling_down("bad"))


# ── task-aware selection ───────────────────────────────────────────────────

class _SpyBroker(ModelBroker):
    def __init__(self):
        super().__init__()
        self.seen: list[tuple[str, str]] = []

    def consult(self, router, operation, task_kind="", constraints=None):
        self.seen.append((operation, task_kind))
        return None


class TaskAwareTests(unittest.TestCase):
    def test_task_kind_reaches_broker(self):
        r = _router()
        r.add(MockProvider(model="m"), name="m", primary=True)
        spy = _SpyBroker()
        r.set_broker(spy)
        r.chat([Message.user("hi")], task_kind="code")
        r.complete("x", task_kind="judge")
        self.assertEqual(spy.seen, [("chat", "code"), ("complete", "judge")])

    def test_vision_operation_prefers_vl_model(self):
        r = _router()
        r.add(MockProvider(model="chatonly"), name="chatonly", primary=True)
        r.add(_VisionMock(model="seer"), name="seer")
        broker = ModelBroker()
        broker.build_from_router(r)
        r.set_broker(broker)
        resp = r.describe_image(b"fake-png-bytes", "what is this?")
        self.assertTrue(resp.ok)
        self.assertEqual(resp.provider, "seer")
        self.assertIn("saw it", resp.text)

    def test_code_task_prefers_code_tuned_card(self):
        broker = ModelBroker()
        broker.register(ModelCard(
            id="generic", capabilities={Capability.CHAT},
            provider="generic", model_id="some-chat-model"))
        broker.register(ModelCard(
            id="codebeast", capabilities={Capability.CHAT, Capability.CODE},
            provider="codebeast", model_id="Cutyp/codebeast-7b-vl",
            owner=True))
        card = broker.select(Capability.CHAT, task_kind="code")
        self.assertIsNotNone(card)
        self.assertEqual(card.id, "codebeast")

    def test_broker_skips_cooling_provider(self):
        clock = _Clock()
        r = _router(clock)
        r.add(MockProvider(model="a"), name="a", primary=True)
        r.add(MockProvider(model="b"), name="b")
        broker = ModelBroker()
        broker.build_from_router(r)
        r.set_broker(broker)
        # knock "a" out
        r._note_failure("a", "boom")
        r._note_failure("a", "boom")
        self.assertTrue(r.is_cooling_down("a"))
        card = broker.consult(r, "chat")
        self.assertIsNotNone(card)
        self.assertEqual(card.provider, "b")
        self.assertEqual(r.active, "b")


# ── owner-model preference ─────────────────────────────────────────────────

class OwnerPreferenceTests(unittest.TestCase):
    def _broker(self):
        from nomorals.llm.benchmarks import BenchmarkDB
        broker = ModelBroker(benchmarks=BenchmarkDB())
        # the owner's own model: no benchmark history at all
        broker.register(ModelCard(
            id="codebeast", capabilities={Capability.CHAT, Capability.CODE},
            provider="codebeast", model_id="Cutyp/codebeast-7b-vl",
            owner=True, notes="owner fine-tune"))
        # a cloud model with a strong measured record
        broker.register(ModelCard(
            id="cloud", capabilities={Capability.CHAT},
            provider="cloud", model_id="big-cloud-model"))
        for _ in range(10):
            broker.benchmarks.record("cloud", Capability.CHAT,
                                     latency_s=0.2, success=True)
        return broker

    def test_owner_model_preferred_when_available(self):
        broker = self._broker()
        card = broker.select(Capability.CHAT)
        self.assertEqual(card.id, "codebeast")

    def test_prefer_owner_false_uses_evidence(self):
        broker = self._broker()
        card = broker.select(Capability.CHAT,
                             constraints={"prefer_owner": False})
        self.assertEqual(card.id, "cloud")

    def test_operator_override_beats_owner(self):
        broker = self._broker()
        broker.promote("cloud")
        card = broker.select(Capability.CHAT)
        self.assertEqual(card.id, "cloud")

    def test_owner_without_capability_never_selected(self):
        broker = ModelBroker()
        broker.register(ModelCard(
            id="codebeast", capabilities={Capability.CHAT},
            provider="codebeast", model_id="Cutyp/codebeast-7b-vl",
            owner=True))
        self.assertIsNone(broker.select(Capability.VISION))

    def test_owner_flag_in_card_dict(self):
        card = ModelCard(id="x", capabilities={Capability.CHAT}, owner=True)
        self.assertTrue(card.to_dict()["owner"])


# ── honest errors + never-raises ───────────────────────────────────────────

class HonestErrorTests(unittest.TestCase):
    def test_explain_failure_names_everything_tried(self):
        r = _router()
        r.add(_Boom(model="b1"), name="b1", primary=True)
        r.add(_Boom(model="b2"), name="b2")
        resp = r.chat([Message.user("hi")])
        text = explain_failure(resp)
        self.assertIn("b1", text)
        self.assertIn("b2", text)
        self.assertIn("Brain unavailable", text)

    def test_explain_failure_never_raises(self):
        self.assertIn("Brain unavailable", explain_failure(None))
        self.assertIn("Brain unavailable", explain_failure(object()))

    def test_brain_never_raises_on_router_explosion(self):
        class _ExplodingRouter:
            def chat(self, *a, **k):
                raise RuntimeError("router exploded")

            def providers(self):
                raise RuntimeError("also exploded")

        brain = Brain(router=_ExplodingRouter())
        resp = brain.chat([Message.user("hi")])
        self.assertFalse(resp.ok)
        self.assertTrue(resp.error)
        # introspection is honest too, not an exception
        self.assertFalse(brain.available())
        self.assertIn("brain", brain.diagnose().lower())

    def test_brain_complete_never_raises(self):
        class _Bad:
            def complete(self, *a, **k):
                raise ValueError("bad")

        resp = Brain(router=_Bad()).complete("hello")
        self.assertFalse(resp.ok)
        self.assertIn("bad", resp.error)


# ── the Brain facade ───────────────────────────────────────────────────────

class BrainFacadeTests(unittest.TestCase):
    def _wired(self, clock=None):
        r = _router(clock) if clock else _router()
        r.add(MockProvider(model="m"), name="real", primary=True)
        return r

    def test_chat_complete_describe(self):
        brain = Brain(router=self._wired())
        resp = brain.chat([Message.user("hello")], task_kind="chat")
        self.assertTrue(resp.ok)
        self.assertTrue(resp.text)
        resp = brain.complete("say hi")
        self.assertTrue(resp.ok)

    def test_available_true_with_live_provider(self):
        self.assertTrue(Brain(router=self._wired()).available())

    def test_available_false_with_mock_only(self):
        r = _router()
        r.add(MockProvider(model="mock-7b"), name="mock", primary=True)
        self.assertFalse(Brain(router=r).available())

    def test_available_false_when_all_cooling(self):
        clock = _Clock()
        r = _router(clock)
        r.add(_Boom(model="b"), name="real", primary=True)
        brain = Brain(router=r)
        r.chat([Message.user("a")])
        r.chat([Message.user("b")])
        self.assertTrue(r.is_cooling_down("real"))
        self.assertFalse(brain.available())

    def test_diagnose_is_human_readable(self):
        brain = Brain(router=self._wired())
        text = brain.diagnose()
        self.assertIn("brain:", text)
        self.assertIn("real", text)

    def test_status_machine_readable(self):
        st = Brain(router=self._wired()).status()
        self.assertTrue(st["available"])
        self.assertEqual(st["active"], "real")
        self.assertEqual(st["providers"][0]["name"], "real")

    def test_legacy_router_without_task_kind_still_works(self):
        class _Legacy:
            def chat(self, messages, params=None):
                return LLMResponse(text="legacy ok")

        resp = Brain(router=_Legacy()).chat(
            [Message.user("hi")], task_kind="code")
        self.assertTrue(resp.ok)
        self.assertEqual(resp.text, "legacy ok")

    def test_get_brain_singleton(self):
        reset_brain()
        try:
            self.assertIs(get_brain(), get_brain())
        finally:
            reset_brain()


# ── defaults: chain order, codebeast, owner hints ──────────────────────────

class DefaultsTests(unittest.TestCase):
    def _env(self):
        return {
            "GROQ_API_KEY": "g",
            "HF_TOKEN": "h",
            "OPENROUTER_API_KEY": "o",
        }

    def test_groq_is_last_resort(self):
        from nomorals.llm.defaults import specs_from_env
        with patch.dict(os.environ, self._env()):
            # make sure stray real keys do not leak in either direction
            names = [s.name for s in specs_from_env()]
        self.assertEqual(names[-1], "groq")

    def test_codebeast_spec_present_before_generic(self):
        from nomorals.llm.defaults import specs_from_env
        with patch.dict(os.environ, self._env()):
            names = [s.name for s in specs_from_env()]
        self.assertIn("codebeast", names)
        self.assertLess(names.index("codebeast"), names.index("hf_serverless"))
        self.assertLess(names.index("codebeast"), names.index("groq"))

    def test_codebeast_needs_hf_token(self):
        from nomorals.llm.defaults import specs_from_env
        env = dict(self._env())
        del env["HF_TOKEN"]
        with patch.dict(os.environ, env):
            names = [s.name for s in specs_from_env()]
        self.assertNotIn("codebeast", names)

    def test_owner_hints_applied_to_cards(self):
        from nomorals.llm.broker import ModelBroker
        from nomorals.llm.defaults import sync_broker_cards
        r = _router()
        r.add(MockProvider(model="m"), name="llama_cpp", primary=True)
        r.add(MockProvider(model="m"), name="codebeast")
        r.add(MockProvider(model="m"), name="groq")
        broker = ModelBroker()
        sync_broker_cards(broker, r)
        self.assertTrue(broker.card("llama_cpp").owner)
        self.assertTrue(broker.card("codebeast").owner)
        self.assertFalse(broker.card("groq").owner)

    def test_codebeast_model_id_env_override(self):
        from nomorals.llm.defaults import codebeast_model_id
        with patch.dict(os.environ, {"NM_CODEBEAST_MODEL": "Cutyp/custom"}):
            self.assertEqual(codebeast_model_id(), "Cutyp/custom")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_CODEBEAST_MODEL", None)
            self.assertEqual(codebeast_model_id(), "Cutyp/codebeast-7b-vl")

    def test_is_owner_model(self):
        from nomorals.llm.defaults import is_owner_model
        self.assertTrue(is_owner_model("Cutyp/codebeast-7b-vl"))
        self.assertTrue(is_owner_model("cutyp/CodeBeast-3.8b"))
        self.assertFalse(is_owner_model("meta-llama/Llama-3.1-8B-Instruct"))
        self.assertFalse(is_owner_model(""))
        with patch.dict(os.environ, {"NM_CODEBEAST_MODEL": "Cutyp/custom"}):
            self.assertTrue(is_owner_model("Cutyp/custom"))

    def test_from_provider_marks_owner_by_model_id(self):
        from nomorals.llm.capabilities import ModelCard
        p = MockProvider(model="Cutyp/codebeast-7b-vl")
        p.name = "hf_serverless"
        self.assertTrue(ModelCard.from_provider(p, card_id="x").owner)
        p2 = MockProvider(model="meta-llama/Llama-3.1-8B-Instruct")
        p2.name = "hf2"
        self.assertFalse(ModelCard.from_provider(p2, card_id="y").owner)


# ── composer uses the brain ────────────────────────────────────────────────

_LLM_SONG_JSON = """{
  "title": "Lagos Holdup",
  "style": "afrobeats",
  "key": "A",
  "mode": "minor",
  "tempo": 98,
  "time_signature": [4, 4],
  "mood": "restless but grooving",
  "groove": {"feel": "laid-back bounce", "swing": 0.1,
             "kick_style": "syncopated", "snare_style": "backbeat"},
  "sections": [
    {"name": "verse", "bars": 8, "chords": ["i", "VI"],
     "energy": 0.4, "lyrics": ["traffic long, engine humming"]},
    {"name": "chorus", "bars": 8, "chords": ["i", "VII"],
     "energy": 0.9, "lyrics": ["we move, we move"]},
    {"name": "outro", "bars": 4, "chords": ["i"],
     "energy": 0.3, "lyrics": ["lights fade"]}
  ],
  "bass_approach": "root notes locking with the kick"
}"""


class ComposerBrainTests(unittest.TestCase):
    def _ctx(self, reply_text, seen):
        class _SpyRouter:
            def stats_snapshot(self):
                return {"active": "codebeast"}

            def chat(self, messages, params=None, *,
                     task_kind="", tier=None, constraints=None):
                seen["task_kind"] = task_kind
                return LLMResponse(text=reply_text, error="")

        class _Ctx:
            router = _SpyRouter()

        return _Ctx()

    def test_task_kind_threaded_to_router(self):
        from nomorals.media import composer_llm

        seen: dict[str, Any] = {}
        # empty/garbage reply → the algorithmic fallback still delivers
        spec = composer_llm.compose_song_spec(
            self._ctx("not json at all", seen), "a test song")
        self.assertEqual(seen.get("task_kind"), "music")
        self.assertIsNotNone(spec)
        self.assertTrue(spec.sections)
        self.assertEqual(spec.source, "algorithmic")

    def test_llm_path_composes_from_model_json(self):
        """The LLM path was silently dead: _PROMPT_TEMPLATE.format() raised
        KeyError on the schema's braces, so every call fell back to
        algorithmic before reaching the router.  A valid model reply must
        now produce an llm-sourced spec."""
        from nomorals.media import composer_llm

        seen: dict[str, Any] = {}
        spec = composer_llm.compose_song_spec(
            self._ctx(_LLM_SONG_JSON, seen), "a song about Lagos traffic")
        self.assertEqual(spec.source, "llm")
        self.assertEqual(spec.title, "Lagos Holdup")
        self.assertEqual(len(spec.sections), 3)

    def test_composition_prompt_renders(self):
        from nomorals.media import composer_llm
        text = composer_llm._composition_prompt(
            "a song", "afrobeats", "likes: burna boy", "ABC hints")
        self.assertIn("a song", text)
        self.assertIn('"title"', text)  # schema appended intact


if __name__ == "__main__":
    unittest.main()
