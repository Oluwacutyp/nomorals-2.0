"""LADDERFIX: in-provider model failover + ladder all-down breaker.

Covers the 2026-11-04 live failure shape: groq's gpt-oss-120b
rate-limited / parse-rejected while sibling models were healthy, and
~40 identical ladder warnings in 2s while every provider cooled down.

* model-scoped failures (per-model 429, model 400/parse, dead id,
  empty choices) walk the provider's fallback model list in-leg;
* provider-scoped failures (auth) abort the leg immediately;
* a short Retry-After gets ONE brief in-leg retry; a long one walks;
* when every ladder rung fails, the breaker opens: subsequent calls
  fast-fail with the same LadderExhausted (details.circuit_open) and
  NO fresh handler attempts until the window expires;
* identical rung-failure warnings are deduped per episode.
"""

from __future__ import annotations

import logging
import time
import unittest
from unittest import mock

import pytest

from nomorals.core.degradation import DegradationLadder, LadderExhausted, Rung
from nomorals.core.error_system import _llm_probe
from nomorals.core.errors import ProviderError, RateLimited
from nomorals.llm.base import Message
from nomorals.llm.providers.groq import GroqProvider
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.providers.openrouter import OpenRouterProvider
from nomorals.llm.router import LLMRouter

PRIMARY = "openai/gpt-oss-120b"
SECOND = "openai/gpt-oss-20b"


def _ok_payload(model: str, text: str = "answer") -> dict:
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
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


def _chat(provider):
    return provider.chat([Message.user("hello")])


# ── in-provider model failover ───────────────────────────────────────────

class ModelWalkTests(unittest.TestCase):
    def test_rate_limited_primary_walks_to_sibling(self):
        provider = _groq({
            PRIMARY: [RateLimited("429 quota", retry_after=60.0)],
            SECOND: [_FakeResponse(_ok_payload(SECOND, "from 20b"))],
        })
        resp = _chat(provider)
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(resp.text, "from 20b")
        self.assertEqual(resp.model, SECOND)
        self.assertEqual(provider.http.calls, [PRIMARY, SECOND])
        # Transient walk: the configured model keeps its place.
        self.assertEqual(provider.model, PRIMARY)
        self.assertEqual(resp.raw["failover_from"], PRIMARY)
        self.assertEqual(
            [a["model"] for a in resp.raw["model_failover_attempts"]],
            [PRIMARY])

    def test_parse_400_walks_to_next_model(self):
        # Groq's live shape: HTTP 400 "Parsing failed" for gpt-oss while
        # other model families accept the same payload.
        provider = _groq({
            PRIMARY: [ProviderError(
                "Groq API error 400: Failed to parse tool call",
                details={"http_status": 400,
                         "body": "Parsing failed"})],
            SECOND: [_FakeResponse(_ok_payload(SECOND))],
        })
        resp = _chat(provider)
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(provider.http.calls, [PRIMARY, SECOND])

    def test_auth_failure_never_walks(self):
        provider = _groq({
            PRIMARY: [ProviderError("invalid api key",
                                    details={"http_status": 401})],
            SECOND: [_FakeResponse(_ok_payload(SECOND))],
        })
        resp = _chat(provider)
        self.assertFalse(resp.ok)
        self.assertEqual(provider.http.calls, [PRIMARY])  # exactly one POST

    def test_dead_id_walk_heals_and_keeps_fallback(self):
        provider = _groq({
            PRIMARY: [ProviderError("model not found",
                                    details={"http_status": 404})],
            SECOND: [_FakeResponse(_ok_payload(SECOND))],
        })
        resp = _chat(provider)
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(resp.raw["healed_from"], PRIMARY)
        self.assertEqual(provider.model, SECOND)  # dead id stays replaced

    def test_empty_choices_walks_then_reports(self):
        provider = OpenRouterProvider(api_key="k", model="qwen/qwen3-8b:free")
        provider.http = _ScriptedHttp({  # type: ignore[assignment]
            "qwen/qwen3-8b:free": [_FakeResponse({"choices": []})],
            "openrouter/free": [_FakeResponse(_ok_payload("openrouter/free"))],
        })
        resp = _chat(provider)
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(provider.http.calls,
                         ["qwen/qwen3-8b:free", "openrouter/free"])

        provider2 = OpenRouterProvider(api_key="k", model="qwen/qwen3-8b:free")
        provider2.http = _ScriptedHttp({  # type: ignore[assignment]
            "qwen/qwen3-8b:free": [_FakeResponse({"choices": []})],
            "openrouter/free": [_FakeResponse({"choices": []})],
        })
        resp2 = _chat(provider2)
        self.assertFalse(resp2.ok)
        self.assertEqual(resp2.error, "provider returned no choices")

    def test_default_chain_and_config_override(self):
        provider = GroqProvider(api_key="k", model=PRIMARY)
        self.assertEqual(provider._model_chain(), [
            PRIMARY, SECOND, "qwen/qwen3.6-27b", "qwen/qwen3.8-27b"])
        custom = GroqProvider(api_key="k", model=PRIMARY,
                              fallback_models=["custom/one", "custom/two"])
        self.assertEqual(custom._model_chain(),
                         [PRIMARY, "custom/one", "custom/two"])
        none = GroqProvider(api_key="k", model=PRIMARY, fallback_models=[])
        self.assertEqual(none._model_chain(), [PRIMARY])

    def test_router_sees_success_not_provider_failure(self):
        provider = _groq({
            PRIMARY: [RateLimited("429 quota", retry_after=60.0)],
            SECOND: [_FakeResponse(_ok_payload(SECOND, "via router"))],
        })
        router = LLMRouter()
        router.add(provider, name="groq", primary=True)
        resp = router.chat([Message.user("hello")])
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(resp.text, "via router")
        # The provider leg healed in-provider: no cooldown, no failures.
        self.assertEqual(router.cooldown_remaining("groq"), 0.0)
        self.assertFalse(router.all_cooling_down())


# ── Retry-After in-leg ───────────────────────────────────────────────────

class RetryAfterTests(unittest.TestCase):
    def test_short_window_retried_once_in_leg(self):
        provider = _groq({
            PRIMARY: [RateLimited("429 quota", retry_after=0.5),
                      _FakeResponse(_ok_payload(PRIMARY, "after wait"))],
            SECOND: [_FakeResponse(_ok_payload(SECOND))],
        })
        with mock.patch("time.sleep") as sleep:
            resp = _chat(provider)
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(resp.text, "after wait")
        self.assertEqual(provider.http.calls, [PRIMARY, PRIMARY])
        sleep.assert_called_once_with(0.5)

    def test_long_window_walks_without_sleeping(self):
        provider = _groq({
            PRIMARY: [RateLimited("429 quota", retry_after=60.0)],
            SECOND: [_FakeResponse(_ok_payload(SECOND))],
        })
        with mock.patch("time.sleep") as sleep:
            resp = _chat(provider)
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(provider.http.calls, [PRIMARY, SECOND])
        sleep.assert_not_called()


# ── router availability surface ──────────────────────────────────────────

class RouterAvailabilityTests(unittest.TestCase):
    def test_all_cooling_down_and_next_available(self):
        now = [1000.0]
        router = LLMRouter(clock=lambda: now[0])
        router.add(MockProvider(model="a"), name="a", primary=True)
        router.add(MockProvider(model="b"), name="b")
        self.assertFalse(router.all_cooling_down())
        self.assertEqual(router.next_available_in(), 0.0)
        router._health["a"].cooldown_until = now[0] + 30.0
        self.assertFalse(router.all_cooling_down())  # b still attempts
        router._health["b"].cooldown_until = now[0] + 10.0
        self.assertTrue(router.all_cooling_down())
        self.assertAlmostEqual(router.next_available_in(), 10.0)
        now[0] += 11.0
        self.assertFalse(router.all_cooling_down())
        self.assertEqual(router.next_available_in(), 0.0)

    def test_empty_router_is_not_an_outage(self):
        router = LLMRouter()
        self.assertFalse(router.all_cooling_down())
        self.assertEqual(router.next_available_in(), 0.0)


# ── llm probe honesty ────────────────────────────────────────────────────

class _StubRouter:
    def __init__(self, chain: list, cooling: bool) -> None:
        self._c = chain
        self._cooling = cooling

    def _chain(self):
        return list(self._c)

    def all_cooling_down(self):
        return self._cooling


class ProbeTests(unittest.TestCase):
    def test_probe_false_when_all_cooling(self):
        probe = _llm_probe(_StubRouter([object()], cooling=True))
        self.assertFalse(probe())

    def test_probe_true_when_one_available(self):
        probe = _llm_probe(_StubRouter([object()], cooling=False))
        self.assertTrue(probe())

    def test_probe_false_when_chain_empty(self):
        probe = _llm_probe(_StubRouter([], cooling=False))
        self.assertFalse(probe())


# ── ladder all-down circuit breaker + log discipline ─────────────────────

def _counting_ladder(fail_first: list[bool], hint, **kw):
    counts = {"primary": 0, "cached": 0}
    state = {"round": 0}

    def primary():
        counts["primary"] += 1
        if fail_first[state["round"]]:
            raise RuntimeError("all LLM providers unavailable")
        return "live answer"

    def cached():
        counts["cached"] += 1
        raise RuntimeError("no cached answer for this prompt")

    ladder = DegradationLadder(
        "llm",
        rungs=[Rung("primary", primary),
               Rung("cached", cached,
                    honesty="All language models are unreachable right now.")],
        exhausted_hint=hint, **kw)
    return ladder, counts, state


def test_breaker_fast_fails_and_logs_once(caplog):
    ladder, counts, _state = _counting_ladder([True], hint=lambda: 30.0)
    with caplog.at_level(logging.DEBUG, logger="nomorals.core.degradation"):
        with pytest.raises(LadderExhausted):
            ladder.call()
        assert counts == {"primary": 1, "cached": 1}
        for _ in range(20):
            with pytest.raises(LadderExhausted) as caught:
                ladder.call()
            assert caught.value.details["circuit_open"] is True
        # No fresh attempts were paid for the 20 fast fails.
        assert counts == {"primary": 1, "cached": 1}
        status = ladder.status()
        assert status["circuit_open"] is True
        assert status["fast_fails"] == 20
        assert 0 < status["retry_in_s"] <= 30.0
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    circuit_logs = [r for r in warnings if "circuit open" in r.getMessage()]
    assert len(circuit_logs) == 1  # one episode line, not per call
    rung_logs = [r for r in warnings if "rung 0 (primary) failed" in r.getMessage()]
    assert len(rung_logs) == 1


def test_breaker_expires_and_providers_retried():
    ladder, counts, state = _counting_ladder(
        [True, False], hint=lambda: 0.05, breaker_floor_s=0.0)
    with pytest.raises(LadderExhausted):
        ladder.call()
    with pytest.raises(LadderExhausted) as caught:
        ladder.call()
    assert caught.value.details.get("circuit_open") is True
    assert counts == {"primary": 1, "cached": 1}
    state["round"] = 1
    time.sleep(0.09)
    result, meta = ladder.call()
    assert result == "live answer"
    assert counts["primary"] == 2  # a fresh attempt round happened
    assert ladder.status()["circuit_open"] is False


def test_rung_warning_dedupe_and_episode_summary(caplog):
    ladder, counts, state = _counting_ladder(
        [True, True, False], hint=lambda: 0.05, breaker_floor_s=0.0)
    with caplog.at_level(logging.DEBUG, logger="nomorals.core.degradation"):
        with pytest.raises(LadderExhausted):
            ladder.call()
        state["round"] = 1
        time.sleep(0.09)  # let the tiny breaker window lapse
        with pytest.raises(LadderExhausted):
            ladder.call()
        state["round"] = 2
        time.sleep(0.09)
        result, _meta = ladder.call()
        assert result == "live answer"
    rung_warnings = [r for r in caplog.records
                     if r.levelno == logging.WARNING
                     and "rung 0 (primary) failed" in r.getMessage()]
    assert len(rung_warnings) == 1  # second identical failure suppressed
    summaries = [r for r in caplog.records
                 if r.levelno == logging.INFO and "suppressed" in r.getMessage()]
    assert summaries, "expected a suppressed-count summary on recovery"
