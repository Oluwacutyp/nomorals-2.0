"""Wave D: the interactive reply budget.

``PartnerResponder.respond_bounded`` must return within its deadline even
when the provider chain stalls — an honest fallback bundle, counted and
logged, never a minutes-long block of the chat thread.
"""

from __future__ import annotations

import time
import unittest

from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import default_persona
from nomorals.partner.relationship import Relationship
from nomorals.partner.responder import (
    INTERACTIVE_REPLY_TIMEOUT_S,
    PartnerResponder,
)


class _StallRouter:
    """chat() blocks far longer than any sane budget."""

    def chat(self, messages, params=None, **kw):
        time.sleep(60.0)
        return LLMResponse(text="too late", model="stall")


class _FastRouter:
    def chat(self, messages, params: SamplingParams | None = None, **kw):
        return LLMResponse(text="mhm, that tracks", model="fake-7b")


class _BoomRouter:
    def chat(self, messages, params=None, **kw):
        raise RuntimeError("provider exploded")


def _responder(router) -> PartnerResponder:
    persona = default_persona()
    return PartnerResponder(
        router,
        persona,
        MoodEngine(persona.baselines),
        Relationship(id="default", stage="in_love", trust=90),
        None,
        None,
    )


def _call(responder, **kw):
    return responder.respond_bounded(
        timeout_s=kw.pop("timeout_s", 1.0),
        chat_platform="local",
        user_text="hey",
        **kw,
    )


class ReplyBudgetTest(unittest.TestCase):
    def test_budget_constant_sane(self):
        self.assertGreaterEqual(INTERACTIVE_REPLY_TIMEOUT_S, 5.0)
        self.assertLessEqual(INTERACTIVE_REPLY_TIMEOUT_S, 60.0)

    def test_stalled_provider_returns_inside_budget(self):
        r = _responder(_StallRouter())
        started = time.perf_counter()
        bundle = _call(r, timeout_s=1.0)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 10.0, f"blocked {elapsed:.1f}s on a stalled provider")
        self.assertTrue(bundle.fallback, "must be the honest fallback bundle")
        self.assertEqual(bundle.model, "fallback")
        self.assertTrue(bundle.parts, "fallback must still produce a reply")
        self.assertEqual(r.reply_timeouts, 1, "timeout must be counted")

    def test_worker_exception_also_falls_back(self):
        r = _responder(_BoomRouter())
        bundle = _call(r, timeout_s=2.0)
        # the router raises immediately -> respond() catches it per-attempt and
        # exhausts retries into the fallback path; either way no raise here
        self.assertTrue(bundle.parts)

    def test_fast_provider_unaffected(self):
        r = _responder(_FastRouter())
        started = time.perf_counter()
        bundle = _call(r, timeout_s=5.0)
        elapsed = time.perf_counter() - started
        self.assertFalse(bundle.fallback)
        self.assertEqual(r.reply_timeouts, 0, "no timeout counted on the fast path")
        self.assertLess(elapsed, 5.0)

    def test_timeout_counter_accumulates(self):
        r = _responder(_StallRouter())
        _call(r, timeout_s=0.5)
        _call(r, timeout_s=0.5)
        self.assertEqual(r.reply_timeouts, 2)

    def test_counter_defaults_zero_without_init(self):
        r = PartnerResponder.__new__(PartnerResponder)
        self.assertEqual(r.reply_timeouts, 0)


if __name__ == "__main__":
    unittest.main()
