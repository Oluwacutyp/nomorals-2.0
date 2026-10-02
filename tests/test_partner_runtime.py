"""End-to-end: gateway -> brain -> fake LLM -> back out the gateway.

The fake router scripts her replies and answers the curator's distillation
prompt with JSON, so the whole loop — mood, memory, persistence, curator,
training collection, and the autonomy approval flow — runs offline.
"""

from __future__ import annotations

import json
import random
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from nomorals.agents.autonomy import AutonomyAgent
from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerBrain, PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.social.chat.base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway


class FakeRouter:
    """A stand-in LLM router with scripted replies."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        self.calls.append(list(messages))
        text_blob = " ".join(m.content for m in messages)
        if "memory-distillation" in text_blob:
            return LLMResponse(
                text=json.dumps({
                    "summary": "They talked about a work win and she teased him about it; warm segment.",
                    "facts": [{"subject": "user", "predicate": "got", "object": "a promotion", "confidence": 0.9}],
                    "milestone": "first time they celebrated something together",
                    "mood_after": "happy",
                }),
                model="fake-curator",
            )
        if self.replies:
            text = self.replies.pop(0)
        else:
            text = "mhm. i was just thinking about you, actually"
        return LLMResponse(text=text, model="fake-7b")


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str = "local") -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self._handler = None
        self.started_flag = False

    def run(self, handler) -> None:
        self._handler = handler
        self.started_flag = True
        while not self.stopped:
            time.sleep(0.01)

    def push(self, chat: ChatRef, text: str) -> None:
        self._deliver(self._handler, ChatMessage(chat=chat, incoming=True, text=text, sender="you"))

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name, message_id=f"m{len(self.sent)}")

    def wait_started(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if getattr(self, "started_flag", False):
                return True
            time.sleep(0.01)
        return False


def _make_context() -> tuple[Any, "tempfile.TemporaryDirectory"]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-")
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        }
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    return context, tmp


def _wait(predicate, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class RuntimeEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter()
        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway({"local": self.adapter}, db=self.context.db)
        self.chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM, peer="you")
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.brain: PartnerBrain = self.runtime.brain
        # Presence is real (busy gaps, read-and-left): seed it so these
        # tests get deterministic immediate replies. Seed 34 rolls no gaps.
        self.brain.presence_rng = random.Random(34)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: E103 - best-effort teardown; must never fail the suite
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_message_in_reply_out_with_mood_shift(self) -> None:
        affection_before = self.brain.mood.value("affection")
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=self.chat, incoming=True,
                                            text="hey, you're the best, i love you", sender="you"))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1), "no reply was sent")
        reply = self.adapter.sent[0]
        self.assertTrue(reply.strip())
        self.assertGreater(self.brain.mood.value("affection"), affection_before)
        # Both sides of the turn persisted.
        rows = self.context.db.query(
            "SELECT role FROM messages WHERE conversation_id = ?", (self.chat.key,)
        )
        roles = sorted(r["role"] for r in rows)
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_training_pair_logged(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=self.chat, incoming=True, text="how's your day going?", sender="you"))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1))
        path = Path(self.tmp.name) / "data" / "training" / "conversations.jsonl"
        self.assertTrue(path.exists(), "training collection did not write")
        record = json.loads(path.read_text().strip().splitlines()[-1])
        self.assertEqual(record["user"], "how's your day going?")
        self.assertIn("assistant", record)
        self.assertIn("mood", record)

    def test_group_stays_quiet_unless_mentioned(self) -> None:
        group = ChatRef(platform="local", chat_id="grp1", kind=ChatKind.GROUP, title="mountain folks")
        self.context.router = FakeRouter(["hi all!"])
        self.brain = PartnerBrain(self.context, curator=None)
        # Seed presence so the mention replies are deterministic (no busy gap).
        self.brain.presence_rng = random.Random(34)
        self.runtime.brain = self.brain
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=group, incoming=True, text="anyone tried the new trail?", sender="bob"))
        time.sleep(0.4)
        self.assertEqual(self.adapter.sent, [], "should stay quiet in a group without a mention")
        self.runtime.on_message(ChatMessage(chat=group, incoming=True,
                                            text="wren, what do you think of that trail?", sender="bob"))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1), "should speak when mentioned by name")
        # A platform tag/mention (message.mentioned) also gets her speaking —
        # the Telegram @username case, where the tag is NOT the persona name.
        self.runtime.on_message(ChatMessage(chat=group, incoming=True,
                                            text="@peacethefirst got a sec?", sender="bob",
                                            mentioned=True))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 2), "should speak when the account is tagged")

    def test_curator_distills_segment_into_memory(self) -> None:
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        for text in ("hey", "so i got the promotion", "really? that's awesome",
                     "yeah, started monday", "lucky me, i guess", "celebrate with me friday?"):
            self.runtime.on_message(ChatMessage(chat=self.chat, incoming=True, text=text, sender="you"))
        self.assertTrue(
            _wait(lambda: self.context.db.scalar(
                "SELECT COUNT(*) FROM memories WHERE source LIKE 'curate:%'", default=0) >= 1,
                timeout=10.0),
            "curator did not distill the segment",
        )
        fact = self.context.db.query_one(
            "SELECT * FROM facts WHERE object = 'a promotion'"
        )
        self.assertIsNotNone(fact, "curator fact was not stored")
        self.assertTrue(any(
            m["text"] == "first time they celebrated something together"
            for m in self.brain.relationship.milestones
        ))

    def test_fast_path_turn_still_persists(self) -> None:
        # Regression: the Core-Mind fast path must not silently drop turns —
        # "hey" is answered with zero model calls, but both sides of the
        # turn are persisted like any other reply.
        self.runtime.start()
        self.assertTrue(self.adapter.wait_started())
        self.runtime.on_message(ChatMessage(chat=self.chat, incoming=True,
                                            text="hey", sender="you"))
        self.assertTrue(_wait(lambda: len(self.adapter.sent) >= 1), "no fast reply sent")
        self.assertTrue(
            _wait(lambda: self.context.db.scalar(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'",
                (self.chat.key,), default=0) >= 1,
                timeout=10.0),
            "fast-path inbound message was not persisted",
        )
        rows = self.context.db.query(
            "SELECT role FROM messages WHERE conversation_id = ?", (self.chat.key,))
        roles = sorted(r["role"] for r in rows)
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)


class PartnerAskCliTest(unittest.TestCase):
    """`nm partner --ask`: the full brain (persona + mood + memory) with NO
    platforms started — the owner's 'start the AI without the chat bot'."""

    def test_ask_replies_without_any_platform(self) -> None:
        import io
        from argparse import Namespace
        from contextlib import redirect_stdout

        from nomorals.cli import _cmd_partner_ask

        context, tmp = _make_context()
        try:
            context.router = FakeRouter(replies=["hey you — yeah, i'm here"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_partner_ask(Namespace(ask="hey there"), context)
            out = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("her [", out)
            self.assertIn("hey you — yeah, i'm here", out)
            # …and nothing was started: no gateway, no adapters.
            row = context.db.query_one(
                "SELECT role FROM messages WHERE conversation_id = 'local:console' "
                "ORDER BY created_at DESC LIMIT 1"
            )
            self.assertEqual(row["role"], "assistant")
        finally:
            context.close()
            tmp.cleanup()


class AutonomyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context, self.tmp = _make_context()
        self.context.router = FakeRouter()
        self.adapter = FakeAdapter("local")
        self.gateway = ChatGateway(
            {"local": self.adapter}, db=self.context.db,
            owner_chats={"local:console"},
        )
        self.gateway.start(lambda m: None)
        self.assertTrue(self.adapter.wait_started())
        self.chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM, peer="you")
        self.gateway.register_chat(self.chat)
        self.brain = PartnerBrain(self.context)
        # Put her in a warm, a-little-needy headspace with a long quiet gap.
        self.brain.mood.note_event("affectionate", 1.0)
        self.brain.mood.note_event("affectionate", 1.0)
        self.brain.mood.note_event("affectionate", 1.0)
        now = time.time()
        self.context.db.execute(
            "UPDATE chats SET last_active = ? WHERE id = ?", (now - 8 * 3600, self.chat.key)
        )

    def tearDown(self) -> None:
        self.gateway.stop()
        self.context.close()
        self.tmp.cleanup()

    def _agent(self, mode: str) -> AutonomyAgent:
        # quiet_start == quiet_end means "never quiet" for deterministic ticks.
        return AutonomyAgent(
            self.context, self.brain, self.gateway,
            mode=mode, owner_chats={"local:console"},
            quiet_start=0, quiet_end=0,
        )

    def test_suggest_mode_holds_proposal(self) -> None:
        agent = self._agent("suggest")
        result = agent.tick()
        self.assertNotIn("decision", result)  # a proposal was created
        self.assertEqual(self.adapter.sent, [], "suggest mode must not send")
        rows = self.context.db.query("SELECT * FROM proactive_log WHERE status = 'pending'")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["content"].strip())

    def test_approve_sends_pending_proposal(self) -> None:
        agent = self._agent("suggest")
        result = agent.tick()
        proposal_id = result["proposal"]
        self.assertEqual(agent.approve(proposal_id)["status"], "sent")
        self.assertEqual(len(self.adapter.sent), 1)
        row = self.context.db.query_one("SELECT status FROM proactive_log WHERE id = ?", (proposal_id,))
        self.assertEqual(row["status"], "sent")

    def test_deny_blocks_proposal(self) -> None:
        agent = self._agent("suggest")
        agent.tick()
        rows = self.context.db.query("SELECT id FROM proactive_log WHERE status = 'pending'")
        self.assertEqual(agent.deny(rows[0]["id"])["status"], "denied")
        self.assertEqual(self.adapter.sent, [])

    def test_auto_mode_sends_directly(self) -> None:
        agent = self._agent("auto")
        result = agent.tick()
        self.assertTrue(result.get("ok"))
        self.assertEqual(result["status"], "sent")
        self.assertEqual(len(self.adapter.sent), 1)
        row = self.context.db.query_one(
            "SELECT status FROM proactive_log ORDER BY decided_at DESC LIMIT 1"
        )
        self.assertEqual(row["status"], "sent")

    def test_quiet_hours_block(self) -> None:
        agent = self._agent("auto")
        agent.quiet_start, agent.quiet_end = 22, 8
        night = time.mktime((2026, 1, 15, 23, 30, 0, 0, 0, 0))
        day = time.mktime((2026, 1, 15, 14, 0, 0, 0, 0, 0))
        self.assertTrue(agent._in_quiet_hours(night))
        self.assertFalse(agent._in_quiet_hours(day))
        result = agent.tick(now=night)
        self.assertEqual(result, {"decision": "quiet hours"})
        self.assertEqual(self.adapter.sent, [])

    def test_daily_dm_cap(self) -> None:
        agent = self._agent("auto")
        agent.max_dm_per_day = 1
        self.assertTrue(agent.tick().get("ok"))
        # Second tick same day, same chat: capped (per-chat interval also blocks it).
        result2 = agent.tick()
        self.assertEqual(len(self.adapter.sent), 1)
        self.assertIn("decision", result2)

    def test_zero_cap_means_unlimited(self) -> None:
        # 0 must mean "no volume cap" (the owner asked for no daily limits),
        # not "never send" as the old check (sent < cap) would imply.
        self.assertFalse(AutonomyAgent._cap_reached(0, 0))
        self.assertFalse(AutonomyAgent._cap_reached(10_000, 0))
        self.assertFalse(AutonomyAgent._cap_reached(4, 5))  # under the cap
        self.assertTrue(AutonomyAgent._cap_reached(5, 5))  # at the cap
        self.assertTrue(AutonomyAgent._cap_reached(6, 5))  # over the cap


if __name__ == "__main__":
    unittest.main()
