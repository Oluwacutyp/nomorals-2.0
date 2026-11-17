"""Calibrated-abstention gate on the ordinary answer path.

Pins the named remainder in ``cognition/calibration_loop.py:40`` — no
live answer path consulted ``CalibratedAbstention`` — now wired into
the ONE ordinary answer path, ``PartnerBrain._generate_and_persist``
(both ``handle_message`` and ``deliver_reply`` flow through it, so
every ordinary reply passes the gate exactly once):

* a model-produced answer to a *knowledge question* with no grounding
  behind it abstains with ``ABSTAIN_TEXT`` — which names what would
  help ("tell me where to look") — instead of fabricating;
* greetings, statements, her-directed questions ("are you coming?"),
  and social openers ("how are you?", "what's up?") are never gated;
* the template fallback bundle and the deterministic STT-failure reply
  are never gated — both are already honest non-answers;
* confidence is measured from the turn, never invented: the research
  formula (``calibrate_confidence``) over word overlap between the
  question's content words and the evidence texts (recalled memories,
  proactive lines, media notes, tool observations);
* the threshold comes from the persisted calibration
  (``threshold_for`` on ``cog_calibration_thresholds``), base 0.55
  when the loop has recorded none — never a hardcoded number in the
  wiring;
* the gate never breaks a reply: any failure inside it fails open to
  the original parts;
* the abstention replaces the reply BEFORE the persist tail, so the
  persisted outbound turn and training pair record the abstention,
  not the fabricated answer.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner.brain import (
    _abstention_confidence,
    _content_words,
    _is_knowledge_question,
)
from nomorals.agents.partner.tool_loop import ToolLoopResult, ToolLoopTrace
from nomorals.agents.partner_runtime import PartnerBrain
from nomorals.core.config import load_settings
from nomorals.llm.abstention import ABSTAIN_TEXT
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.partner.responder import ReplyBundle
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

#: Context homes honour NM_TEST_TMPDIR so a test runner can keep them
#: off the OS tmpdir (see test_partner_brain_gate.py).
_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None


class FakeRouter:
    """A stand-in LLM router. The gate tests mock the responder and
    the tool loop, so the router is never actually called."""

    def chat(self, messages: list[Message],
             params: SamplingParams | None = None,
             **kw: Any) -> LLMResponse:
        return LLMResponse(text="mhm.", model="fake-7b")


def _make_brain() -> tuple[PartnerBrain, Any, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-abstain-",
                                      dir=_TEST_TMPDIR)
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "local",
        # Gating kill-switch off: every DM reads as owner mode, so
        # memories flow and the tool loop is reachable — the gate's
        # own scope rules (not the surface gate) decide what abstains.
        "partner.gate_restricted_chats": "false",
        # No reasoning digest: it would spend LLM calls on owner turns.
        "partner.reasoning": "off",
    })
    context = build_context(settings, with_executor=False,
                            with_router=False, with_memory=False,
                            with_tools=False)
    context.router = FakeRouter()
    brain = PartnerBrain(context, curator=None)
    return brain, context, tmp


def _msg(text: str) -> ChatMessage:
    chat = ChatRef(platform="local", chat_id="console",
                   kind=ChatKind.DM, peer="you")
    return ChatMessage(chat=chat, incoming=True, text=text, sender="you")


def _flags(owner: bool = False) -> dict[str, Any]:
    return {"is_owner": owner, "in_us": False, "last_active": 0.0}


def _bundle(text: str, *, model: str = "fake-7b",
            fallback: bool = False) -> ReplyBundle:
    return ReplyBundle(parts=[text], model=model, fallback=fallback)


def _tool_result(answer: str, *,
                 tools_used: list[str] | None = None,
                 observations: list[str] | None = None) -> ToolLoopResult:
    return ToolLoopResult(
        answer=answer, tools_used=list(tools_used or []),
        trace=[ToolLoopTrace(iteration=0, model_text="",
                             tool_calls=[],
                             observations=list(observations or []))])


class KnowledgeQuestionTests(unittest.TestCase):
    def test_question_shape(self) -> None:
        self.assertTrue(_is_knowledge_question(
            "what did we decide about the venue?"))
        self.assertTrue(_is_knowledge_question("when is the meeting?"))
        self.assertTrue(_is_knowledge_question(
            "do you know the venue?"))
        self.assertTrue(_is_knowledge_question(
            "tell me about the venue"))
        self.assertTrue(_is_knowledge_question(
            "explain quantum computing"))
        self.assertTrue(_is_knowledge_question(
            "can you explain quantum computing"))
        self.assertTrue(_is_knowledge_question("what time is it?"))

    def test_not_knowledge_questions(self) -> None:
        self.assertFalse(_is_knowledge_question("how are you?"))
        self.assertFalse(_is_knowledge_question("what's up?"))
        self.assertFalse(_is_knowledge_question("what's new?"))
        self.assertFalse(_is_knowledge_question("how's it going?"))
        self.assertFalse(_is_knowledge_question(
            "are you coming to the party?"))
        self.assertFalse(_is_knowledge_question("do you want to go?"))
        self.assertFalse(_is_knowledge_question(
            "can you send me the file?"))
        self.assertFalse(_is_knowledge_question("i love this song"))
        self.assertFalse(_is_knowledge_question(""))
        self.assertFalse(_is_knowledge_question("do you know?"))
        # Opinion questions ask for her stance, not a fact.
        self.assertFalse(_is_knowledge_question(
            "what do you think about the venue?"))
        self.assertFalse(_is_knowledge_question(
            "do you think it will rain?"))
        self.assertFalse(_is_knowledge_question(
            "how do you feel about the plan?"))
        self.assertFalse(_is_knowledge_question(
            "wren, wetin you think — u dey smoke?"))

    def test_content_words(self) -> None:
        self.assertEqual(_content_words(
            "what did we decide about the venue?"), ["decide", "venue"])
        self.assertEqual(_content_words("how are you?"), [])
        self.assertEqual(_content_words("tell me about the venue"),
                         ["venue"])


class ConfidenceTests(unittest.TestCase):
    def test_no_evidence_is_floor(self) -> None:
        conf, count = _abstention_confidence(
            "what did we decide about the venue?", [])
        self.assertEqual(count, 0)
        self.assertEqual(conf, 0.16)

    def test_one_covering_memory_answers(self) -> None:
        conf, count = _abstention_confidence(
            "what did we decide about the venue?",
            ["we decided the venue is the rooftop"])
        self.assertEqual(count, 1)
        self.assertEqual(conf, 0.73)

    def test_one_half_covering_memory_abstains(self) -> None:
        conf, count = _abstention_confidence(
            "what did we decide about the venue?",
            ["the venue is downtown"])
        self.assertEqual(count, 1)
        self.assertEqual(conf, 0.50)

    def test_three_irrelevant_memories_abstain(self) -> None:
        conf, count = _abstention_confidence(
            "what did we decide about the venue?",
            ["the cat slept all day", "it rained in Lagos",
             "buy milk tomorrow"])
        self.assertEqual(count, 3)
        self.assertEqual(conf, 0.51)

    def test_two_memories_best_half_cover_answers(self) -> None:
        conf, count = _abstention_confidence(
            "what did we decide about the venue?",
            ["the venue is the rooftop", "we decided on Friday"])
        self.assertEqual(count, 2)
        self.assertEqual(conf, 0.62)


def _teardown(ctx, tmp) -> None:
    """Close the context (stops the DB writer actor) before removing
    the temp home — a still-running actor can recreate the sqlite
    -wal/-shm mid-rmtree and make cleanup fail spuriously."""
    with contextlib.suppress(Exception):
        ctx.close()
    tmp.cleanup()

class GateTests(unittest.TestCase):
    def _run(self, brain: PartnerBrain, text: str, *,
             owner: bool = False, memories: list[str] | None = None,
             bundle: ReplyBundle | None = None,
             tool_result: ToolLoopResult | None = None) -> tuple[list[str], bool, str]:
        with mock.patch.object(brain.responder, "recall",
                               return_value=list(memories or [])), \
            mock.patch.object(
                brain.responder, "respond_bounded",
                return_value=bundle or _bundle("the venue is the rooftop")), \
            mock.patch.object(brain, "generate_with_tools",
                              return_value=tool_result):
            return brain._generate_and_persist(_msg(text), _flags(owner))

    def test_knowledge_question_without_evidence_abstains(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            parts, _degraded, _note = self._run(
                brain, "what did we decide about the venue?")
            self.assertEqual(parts, [ABSTAIN_TEXT])
            self.assertIn("Tell me where to look", parts[0])
        finally:
            _teardown(ctx, tmp)

    def test_knowledge_question_with_covering_memory_answers(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            parts, _degraded, _note = self._run(
                brain, "what did we decide about the venue?",
                memories=["we decided the venue is the rooftop"])
            self.assertEqual(parts, ["the venue is the rooftop"])
        finally:
            _teardown(ctx, tmp)

    def test_greeting_never_abstains(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            parts, _degraded, _note = self._run(brain, "how are you?")
            self.assertEqual(parts, ["the venue is the rooftop"])
        finally:
            _teardown(ctx, tmp)

    def test_statement_never_abstains(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            parts, _degraded, _note = self._run(brain, "i love this song")
            self.assertEqual(parts, ["the venue is the rooftop"])
        finally:
            _teardown(ctx, tmp)

    def test_her_directed_question_never_abstains(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            parts, _degraded, _note = self._run(
                brain, "are you coming to the party?")
            self.assertEqual(parts, ["the venue is the rooftop"])
        finally:
            _teardown(ctx, tmp)

    def test_fallback_bundle_never_gated(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            parts, degraded, _note = self._run(
                brain, "what did we decide about the venue?",
                bundle=_bundle("lol my head is just full today",
                               fallback=True))
            self.assertEqual(parts, ["lol my head is just full today"])
            self.assertTrue(degraded)
        finally:
            _teardown(ctx, tmp)

    def test_tool_grounded_answer_flows(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            parts, _degraded, _note = self._run(
                brain, "when is the meeting?", owner=True,
                tool_result=_tool_result(
                    "the meeting is at 3pm", tools_used=["calendar"],
                    observations=["meeting at 3pm on Friday"]))
            self.assertEqual(parts, ["the meeting is at 3pm"])
        finally:
            _teardown(ctx, tmp)

    def test_tool_loop_empty_observations_abstains(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            parts, _degraded, _note = self._run(
                brain, "when is the meeting?", owner=True,
                tool_result=_tool_result(
                    "the meeting is at 3pm", tools_used=["calendar"],
                    observations=[]))
            self.assertEqual(parts, [ABSTAIN_TEXT])
        finally:
            _teardown(ctx, tmp)

    def test_abstention_is_persisted_not_the_fabrication(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            # Production always persists the inbound first (which
            # creates the conversation row the outbound insert needs).
            msg = _msg("what did we decide about the venue?")
            brain._persist_inbound(msg)
            with mock.patch.object(brain.responder, "recall",
                                  return_value=[]), \
                mock.patch.object(
                    brain.responder, "respond_bounded",
                    return_value=_bundle("the venue is the rooftop")), \
                mock.patch.object(brain, "generate_with_tools",
                                  return_value=None):
                brain._generate_and_persist(msg, _flags())
            rows = ctx.db.query(
                "SELECT content FROM messages WHERE role = 'assistant'"
                " ORDER BY created_at DESC LIMIT 1")
            self.assertTrue(rows)
            self.assertIn("Tell me where to look", rows[0]["content"])
            self.assertNotIn("rooftop", rows[0]["content"])
        finally:
            _teardown(ctx, tmp)

    def test_gate_failure_is_fail_open(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            with mock.patch(
                    "nomorals.llm.abstention.CalibratedAbstention",
                    side_effect=RuntimeError("gate broken")), \
                mock.patch.object(brain.responder, "recall",
                                  return_value=[]), \
                mock.patch.object(
                    brain.responder, "respond_bounded",
                    return_value=_bundle("the venue is the rooftop")), \
                mock.patch.object(brain, "generate_with_tools",
                                  return_value=None):
                parts, _degraded, _note = brain._generate_and_persist(
                    _msg("what did we decide about the venue?"),
                    _flags())
            self.assertEqual(parts, ["the venue is the rooftop"])
        finally:
            _teardown(ctx, tmp)

    def test_threshold_failure_is_fail_open(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            with mock.patch(
                    "nomorals.cognition.calibration_loop.threshold_for",
                    side_effect=RuntimeError("no table")), \
                mock.patch.object(brain.responder, "recall",
                                  return_value=[]), \
                mock.patch.object(
                    brain.responder, "respond_bounded",
                    return_value=_bundle("the venue is the rooftop")), \
                mock.patch.object(brain, "generate_with_tools",
                                  return_value=None):
                parts, _degraded, _note = brain._generate_and_persist(
                    _msg("what did we decide about the venue?"),
                    _flags())
            self.assertEqual(parts, ["the venue is the rooftop"])
        finally:
            _teardown(ctx, tmp)


class PersistedThresholdTests(unittest.TestCase):
    def _persist(self, ctx: Any, task: str, model: str,
                 threshold: float) -> None:
        from nomorals.cognition.calibration_loop import (
            _ensure_thresholds_schema,
        )

        _ensure_thresholds_schema(ctx.db)
        import time as _time

        ctx.db.execute(
            "INSERT OR REPLACE INTO cog_calibration_thresholds"
            " (task, model, threshold, base_threshold, n,"
            " calibration_gap, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task, model, threshold, 0.55, 10, 0.35, _time.time()))

    def test_persisted_threshold_raises_the_bar(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            # One covering memory scores 0.73 — answers at the base
            # 0.55, abstains once the loop persisted 0.90 for this slice.
            self._persist(ctx, "chat", "fake-7b", 0.90)
            with mock.patch.object(brain.responder, "recall",
                                  return_value=[
                                      "we decided the venue is the rooftop"]), \
                mock.patch.object(
                    brain.responder, "respond_bounded",
                    return_value=_bundle("the venue is the rooftop")), \
                mock.patch.object(brain, "generate_with_tools",
                                  return_value=None):
                parts, _degraded, _note = brain._generate_and_persist(
                    _msg("what did we decide about the venue?"),
                    _flags())
            self.assertEqual(parts, [ABSTAIN_TEXT])
        finally:
            _teardown(ctx, tmp)

    def test_persisted_threshold_lowers_the_bar(self) -> None:
        brain, ctx, tmp = _make_brain()
        try:
            # One half-covering memory scores 0.50 — abstains at the
            # base 0.55, answers once the loop persisted 0.30.
            self._persist(ctx, "chat", "fake-7b", 0.30)
            with mock.patch.object(brain.responder, "recall",
                                  return_value=[
                                      "the venue is downtown"]), \
                mock.patch.object(
                    brain.responder, "respond_bounded",
                    return_value=_bundle("the venue is the rooftop")), \
                mock.patch.object(brain, "generate_with_tools",
                                  return_value=None):
                parts, _degraded, _note = brain._generate_and_persist(
                    _msg("what did we decide about the venue?"),
                    _flags())
            self.assertEqual(parts, ["the venue is the rooftop"])
        finally:
            _teardown(ctx, tmp)


if __name__ == "__main__":
    unittest.main()
