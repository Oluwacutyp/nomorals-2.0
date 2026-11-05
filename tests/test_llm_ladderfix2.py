"""LADDERFIX follow-ups: embed/vision model walk + breaker window.

The original ladderfix (tests/test_llm_ladderfix.py) put CHAT on the
in-provider failover policy — model-scoped failures walk the roster,
provider-scoped failures abort, a short Retry-After earns one in-leg
retry — and left two paths on the old retry-the-same-model policy:

* ``OpenAICompatProvider.embed`` / ``describe_image`` retried the SAME
  model in place (a 429 burned the whole retry budget against the
  limited model; a dead id never walked);
* the router's all-down fast-fail window (``next_available_in``) read
  only the health cooldown and ignored each provider's own circuit-
  breaker reset deadline.

These tests pin the follow-up: embed and vision walk under the same
classification chat uses, the breaker is shared across operations (a
provider marked down by embed is down for chat and vice versa), and
the availability window honors breaker reset times.
"""

from __future__ import annotations

import unittest
from unittest import mock

from nomorals.core.errors import ModelError, ProviderError, ProviderUnavailable, RateLimited
from nomorals.core.retry import CircuitBreaker
from nomorals.llm.base import Message
from nomorals.llm.providers.groq import GroqProvider
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.providers.openai_compat import OpenAICompatProvider
from nomorals.llm.router import LLMRouter

PRIMARY = "openai/gpt-oss-120b"
SECOND = "openai/gpt-oss-20b"
ROSTER = [PRIMARY, SECOND, "qwen/qwen3.6-27b", "qwen/qwen3.8-27b"]


def _embed_payload(vectors: list[list[float]]) -> dict:
    return {"data": [{"index": i, "embedding": v}
                     for i, v in enumerate(vectors)]}


def _vision_payload(model: str, text: str = "a cat on a mat") -> dict:
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _ScriptedHttp:
    """Per-model outcome queues; records the model of every POST."""

    def __init__(self, script: dict[str, list]) -> None:
        self._script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    def post_json(self, url, payload, **kw):
        model = payload["model"]
        self.calls.append(model)
        queue = self._script.get(model) or []
        if not queue:
            raise AssertionError(f"no scripted outcome left for {model}")
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _groq(script: dict[str, list], **kw) -> GroqProvider:
    provider = GroqProvider(api_key="test-key", model=PRIMARY, **kw)
    provider.http = _ScriptedHttp(script)  # type: ignore[assignment]
    return provider


# ── embed model walk ─────────────────────────────────────────────────────

class EmbedWalkTests(unittest.TestCase):
    def test_rate_limited_primary_walks_to_sibling(self):
        provider = _groq({
            PRIMARY: [RateLimited("429 quota", retry_after=60.0)],
            SECOND: [_FakeResponse(_embed_payload([[0.5, 0.25]]))],
        })
        vectors = provider.embed(["hello"])
        self.assertEqual(vectors, [[0.5, 0.25]])
        self.assertEqual(provider.http.calls, [PRIMARY, SECOND])

    def test_dead_id_skipped(self):
        provider = _groq({
            PRIMARY: [ProviderError("model not found",
                                    details={"http_status": 404})],
            SECOND: [_FakeResponse(_embed_payload([[1.0]]))],
        })
        vectors = provider.embed(["hello"])
        self.assertEqual(vectors, [[1.0]])
        self.assertEqual(provider.http.calls, [PRIMARY, SECOND])
        # Per-call walk only: an embeddings 404 does NOT rewrite the
        # configured (chat) model — it proves the id cannot embed,
        # not that it cannot chat.
        self.assertEqual(provider.model, PRIMARY)

    def test_auth_never_walks(self):
        provider = _groq({
            PRIMARY: [ProviderError("invalid api key",
                                    details={"http_status": 401})],
            SECOND: [_FakeResponse(_embed_payload([[1.0]]))],
        })
        with self.assertRaises(ProviderError):
            provider.embed(["hello"])
        self.assertEqual(provider.http.calls, [PRIMARY])  # exactly one POST

    def test_explicit_model_never_walks(self):
        provider = _groq({
            "custom/embed": [RateLimited("429 quota", retry_after=60.0)],
            SECOND: [_FakeResponse(_embed_payload([[1.0]]))],
        })
        with self.assertRaises(RateLimited):
            provider.embed(["hello"], model="custom/embed")
        self.assertEqual(provider.http.calls, ["custom/embed"])

    def test_short_retry_after_retries_in_leg(self):
        provider = _groq({
            PRIMARY: [RateLimited("429 quota", retry_after=0.5),
                      _FakeResponse(_embed_payload([[0.5]]))],
            SECOND: [_FakeResponse(_embed_payload([[9.9]]))],
        })
        with mock.patch("time.sleep") as sleep:
            vectors = provider.embed(["hello"])
        self.assertEqual(vectors, [[0.5]])
        self.assertEqual(provider.http.calls, [PRIMARY, PRIMARY])
        sleep.assert_called_once_with(0.5)

    def test_empty_data_walks_then_serves(self):
        provider = _groq({
            PRIMARY: [_FakeResponse({"data": []})],
            SECOND: [_FakeResponse(_embed_payload([[0.5]]))],
        })
        vectors = provider.embed(["hello"])
        self.assertEqual(vectors, [[0.5]])
        self.assertEqual(provider.http.calls, [PRIMARY, SECOND])

    def test_empty_data_everywhere_is_model_error(self):
        provider = _groq({m: [_FakeResponse({"data": []})] for m in ROSTER})
        with self.assertRaises(ModelError) as ctx:
            provider.embed(["hello"])
        self.assertIn("0 vectors", str(ctx.exception))
        self.assertEqual(provider.http.calls, ROSTER)

    def test_partial_count_is_contract_error_no_walk(self):
        provider = _groq({
            PRIMARY: [_FakeResponse(_embed_payload([[0.5]]))],
            SECOND: [_FakeResponse(_embed_payload([[0.5], [0.6]]))],
        })
        with self.assertRaises(ModelError):
            provider.embed(["one", "two"])  # 1 vector for 2 inputs
        self.assertEqual(provider.http.calls, [PRIMARY])

    def test_single_model_provider_terminal_shape_unchanged(self):
        provider = OpenAICompatProvider(
            base_url="http://localhost:9/v1", api_key="k", model="only")
        provider.http = _ScriptedHttp({  # type: ignore[assignment]
            "only": [RateLimited("429 quota", retry_after=60.0)],
        })
        with self.assertRaises(RateLimited):
            provider.embed(["hello"])
        self.assertEqual(provider.http.calls, ["only"])


# ── vision model walk ────────────────────────────────────────────────────

class VisionWalkTests(unittest.TestCase):
    def test_rate_limited_primary_walks_to_sibling(self):
        provider = _groq({
            PRIMARY: [RateLimited("429 quota", retry_after=60.0)],
            SECOND: [_FakeResponse(_vision_payload(SECOND, "from 20b"))],
        })
        resp = provider.describe_image(b"\x89PNG", "what is this?")
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(resp.text, "from 20b")
        self.assertEqual(resp.model, SECOND)
        self.assertEqual(provider.http.calls, [PRIMARY, SECOND])

    def test_dead_id_walks_without_healing(self):
        provider = _groq({
            PRIMARY: [ProviderError("model not found",
                                    details={"http_status": 404})],
            SECOND: [_FakeResponse(_vision_payload(SECOND))],
        })
        resp = provider.describe_image(b"\x89PNG")
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(provider.http.calls, [PRIMARY, SECOND])
        # Deliberate divergence from chat: a vision-leg 404 is
        # ambiguous (dead id OR a server that cannot take image
        # payloads), so the vision walk never rewrites self.model.
        self.assertEqual(provider.model, PRIMARY)

    def test_auth_aborts_with_provider_error(self):
        provider = _groq({
            PRIMARY: [ProviderError("invalid api key",
                                    details={"http_status": 401})],
            SECOND: [_FakeResponse(_vision_payload(SECOND))],
        })
        with self.assertRaises(ProviderError) as ctx:
            provider.describe_image(b"\x89PNG")
        self.assertIn("vision request failed", str(ctx.exception))
        self.assertEqual(provider.http.calls, [PRIMARY])

    def test_all_rate_limited_raises_provider_error(self):
        provider = _groq({
            m: [RateLimited("429 quota", retry_after=60.0)] for m in ROSTER
        })
        with self.assertRaises(ProviderError) as ctx:
            provider.describe_image(b"\x89PNG")
        self.assertIn("vision request failed", str(ctx.exception))
        self.assertEqual(provider.http.calls, ROSTER)

    def test_vision_model_pinned_never_walks(self):
        provider = _groq(
            {
                "vision-x": [RateLimited("429 quota", retry_after=60.0)],
                PRIMARY: [_FakeResponse(_vision_payload(PRIMARY))],
            },
            vision_model="vision-x",
        )
        with self.assertRaises(ProviderError):
            provider.describe_image(b"\x89PNG")
        self.assertEqual(provider.http.calls, ["vision-x"])


# ── router: one breaker across embed / vision / chat ─────────────────────

class _BoomAll(MockProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.chat_calls = 0
        self.embed_calls = 0

    def chat(self, messages, params=None, **kw):
        self.chat_calls += 1
        raise RuntimeError("chat boom")

    def embed(self, texts, **kw):
        self.embed_calls += 1
        raise RuntimeError("embed boom")


def _strict_router(**kw):
    return LLMRouter(cooldown_seconds=60.0, failure_threshold=2, **kw)


class _ChatOnly(MockProvider):
    @property
    def capabilities(self):
        return {"chat"}


class SharedBreakerTests(unittest.TestCase):
    def test_router_embed_in_provider_walk_notes_no_failure(self):
        # A generic compat provider (advertises embed) whose explicit
        # fallback roster the registry would have built from config.
        provider = OpenAICompatProvider(
            base_url="http://localhost:9/v1", api_key="k", model=PRIMARY,
            fallback_models=[SECOND])
        provider.http = _ScriptedHttp({  # type: ignore[assignment]
            PRIMARY: [RateLimited("429 quota", retry_after=60.0)],
            SECOND: [_FakeResponse(_embed_payload([[0.5]]))],
        })
        router = LLMRouter()
        router.add(provider, name="compat", primary=True)
        vectors = router.embed(["hello"])
        self.assertEqual(vectors, [[0.5]])
        # The provider leg healed in-provider (mirrors the chat pin):
        # no router failure, no cooldown.
        self.assertEqual(router._health["compat"].failures, 0)
        self.assertEqual(router.cooldown_remaining("compat"), 0.0)

    def test_embed_failures_mark_provider_down_for_chat(self):
        provider = _BoomAll(model="bad")
        router = _strict_router()
        router.add(provider, name="bad", primary=True)
        # Chat-only fallback: embed has nowhere to fail over TO, chat does.
        router.add(_ChatOnly(model="good"), name="good")
        with self.assertRaises(ProviderUnavailable):
            router.embed(["x"])
        with self.assertRaises(ProviderUnavailable):
            router.embed(["x"])  # 2nd failure → cooldown + breaker open
        self.assertEqual(provider.embed_calls, 2)
        # Third embed: skipped without touching the provider, and the
        # error names the skip instead of an empty "last error".
        with self.assertRaises(ProviderUnavailable) as ctx:
            router.embed(["x"])
        self.assertIn("cooling down", str(ctx.exception))
        self.assertEqual(provider.embed_calls, 2)
        # Chat to the same provider is ALSO down (shared breaker):
        # dispatch skips it and serves via the healthy provider.
        resp = router.chat([Message.user("hi")])
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(provider.chat_calls, 0)

    def test_chat_failures_mark_provider_down_for_embed(self):
        provider = _BoomAll(model="bad")
        router = _strict_router()
        router.add(provider, name="bad", primary=True)
        router.add(MockProvider(model="good"), name="good")
        router.chat([Message.user("hi")])  # bad fails (1)
        router.chat([Message.user("hi")])  # bad fails (2) → cooling
        self.assertEqual(provider.chat_calls, 2)
        vectors = router.embed(["x"])  # served by good; bad untouched
        self.assertEqual(len(vectors), 1)
        self.assertEqual(provider.embed_calls, 0)


# ── breaker reset window ─────────────────────────────────────────────────

class BreakerWindowTests(unittest.TestCase):
    def test_breaker_reset_remaining(self):
        now = [1000.0]
        breaker = CircuitBreaker(
            "t", failure_threshold=2, reset_timeout=30.0,
            clock=lambda: now[0])
        self.assertEqual(breaker.reset_remaining(), 0.0)  # closed
        breaker.record_failure()
        self.assertEqual(breaker.reset_remaining(), 0.0)  # still closed
        breaker.record_failure()  # opens
        self.assertEqual(breaker.state, CircuitBreaker.OPEN)
        self.assertAlmostEqual(breaker.reset_remaining(), 30.0)
        now[0] += 12.0
        self.assertAlmostEqual(breaker.reset_remaining(), 18.0)
        now[0] += 20.0  # past the deadline: a probe is due
        self.assertEqual(breaker.reset_remaining(), 0.0)

    def test_cooldown_remaining_honors_breaker_reset(self):
        now = [1000.0]
        router = LLMRouter(clock=lambda: now[0])
        router.add(MockProvider(model="a"), name="a", primary=True)
        health = router._health["a"]
        # The constructed desync the old window missed: the health
        # cooldown never got set (or already lapsed) while the
        # provider's breaker is open on its own 300s reset window.
        health.breaker.reset_timeout = 300.0
        for _ in range(router.failure_threshold):
            health.breaker.record_failure()
        self.assertEqual(health.cooldown_until, 0.0)
        self.assertTrue(router.is_cooling_down("a"))
        self.assertAlmostEqual(router.cooldown_remaining("a"), 300.0)
        self.assertAlmostEqual(router.next_available_in(), 300.0)
        now[0] += 120.0
        self.assertAlmostEqual(router.cooldown_remaining("a"), 180.0)
        self.assertAlmostEqual(router.next_available_in(), 180.0)
        now[0] += 200.0  # breaker reset lapsed → available again
        self.assertEqual(router.next_available_in(), 0.0)

    def test_cooldown_remaining_health_cooldown_wins_when_longer(self):
        now = [1000.0]
        router = LLMRouter(clock=lambda: now[0])
        router.add(MockProvider(model="a"), name="a", primary=True)
        health = router._health["a"]
        health.breaker.reset_timeout = 300.0
        for _ in range(router.failure_threshold):
            health.breaker.record_failure()
        health.cooldown_until = now[0] + 500.0
        self.assertAlmostEqual(router.cooldown_remaining("a"), 500.0)

    def test_next_available_uses_earliest_breaker_reset(self):
        now = [1000.0]
        router = LLMRouter(clock=lambda: now[0])
        router.add(MockProvider(model="a"), name="a", primary=True)
        router.add(MockProvider(model="b"), name="b")
        for name, reset in (("a", 300.0), ("b", 120.0)):
            health = router._health[name]
            health.breaker.reset_timeout = reset
            for _ in range(router.failure_threshold):
                health.breaker.record_failure()
        self.assertTrue(router.all_cooling_down())
        self.assertAlmostEqual(router.next_available_in(), 120.0)


if __name__ == "__main__":
    unittest.main()
