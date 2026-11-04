"""Send-boundary scaffold guard for conversational replies.

The incident (owner's WhatsApp group, 2026-11-04): the model's raw
per-message analysis — header, analysis prose, context/vibe verdict
line — posted verbatim AS THE OWNER on his personal account, because
nothing between the composed draft and ``gateway.send`` ever asked
whether the text was a reply at all. These tests pin the guard at
its boundary (``PartnerRuntime._send_reply``, the seam every
conversational send crosses on every platform) and at the detector:

* scaffold-shaped bodies never reach an adapter, on WhatsApp or
  telegram-personal, in groups or DMs;
* ordinary replies — pidgin, quotes of the triggering phrase —
  pass through byte-identical;
* a fully-scaffolded turn resolves to one in-character reaction,
  and every rejection lands in the partner failure ledger as a
  hash + length, never the text.
"""

from __future__ import annotations

import json
import random
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner.failures import (
    clear_failures,
    recent_failures,
)
from nomorals.agents.partner.reply_guard import (
    fallback_reaction,
    guard_reply_parts,
    is_scaffold,
    scaffold_reason,
)
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway

#: The incident body, shape-faithful (posted lowercase by the
#: texting-voice layer): header, analysis prose, context verdict.
INCIDENT = (
    "analyze the user's input:\n"
    "\"u dey smoke\" - this is a nigerian pidgin/english phrase "
    "meaning \"you are smoking,\" often used playfully, sarcastically, "
    "or colloquially (sometimes meaning \"you're being weird\" or "
    "literally \"you're smoking\"). in a casual group chat context, "
    "it's often just a playful, informal way of saying \"you seem "
    "weird\" or \"you're acting strange.\"\n"
    "context: group chat with sucidial and ekpo. the vibe is "
    "neutral, pace is buzzing"
)


class DetectorTests(unittest.TestCase):
    def test_incident_body_is_scaffold(self) -> None:
        self.assertTrue(is_scaffold(INCIDENT))
        self.assertEqual(scaffold_reason(INCIDENT), "analysis header")

    def test_header_only_variant_blocked(self) -> None:
        body = "analyze the user's input:\nhe is asking about the match"
        self.assertTrue(is_scaffold(body))

    def test_context_line_only_variant_blocked(self) -> None:
        body = "context: group chat, the vibe is calm, pace is slow"
        self.assertTrue(is_scaffold(body))

    def test_reasoning_preamble_in_dm_blocked(self) -> None:
        body = "let me analyze this properly — he is probably tired"
        self.assertTrue(is_scaffold(body))
        body2 = "analysis: the message is a joke about the match"
        self.assertTrue(is_scaffold(body2))

    def test_meta_phrase_blocked(self) -> None:
        self.assertTrue(is_scaffold("as an ai, i think he's joking"))
        self.assertTrue(
            is_scaffold("confidence score: high. he is definitely joking"))

    def test_pidgin_reply_passes(self) -> None:
        self.assertFalse(is_scaffold("u dey smoke ke? 😂 na play i dey play o"))
        self.assertFalse(is_scaffold("no be me talk am o, na ekpo start am"))

    def test_quoting_the_phrase_passes(self) -> None:
        self.assertFalse(is_scaffold('"u dey smoke" 😂 who talk that one?'))
        self.assertFalse(
            is_scaffold("when he said \"u dey smoke\" i just weak 😂"))

    def test_ordinary_context_word_passes(self) -> None:
        self.assertFalse(
            is_scaffold("for context, i don already tell am yesterday"))
        self.assertFalse(is_scaffold("the vibe is off today tbh"))

    def test_blank_is_not_scaffold(self) -> None:
        self.assertFalse(is_scaffold(""))
        self.assertFalse(is_scaffold("   \n  "))


class GuardPartsTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()

    def test_clean_parts_pass_untouched(self) -> None:
        parts = ["lol 😂", "u dey smoke ke? na play o"]
        self.assertEqual(
            guard_reply_parts(parts, kind=ChatKind.GROUP,
                              rng=random.Random(1)),
            parts)

    def test_mixed_parts_keep_clean_in_order(self) -> None:
        parts = ["first clean one", INCIDENT, "second clean one"]
        out = guard_reply_parts(parts, kind=ChatKind.GROUP,
                                rng=random.Random(1))
        self.assertEqual(out, ["first clean one", "second clean one"])

    def test_all_scaffold_group_gets_one_reaction(self) -> None:
        out = guard_reply_parts([INCIDENT], kind=ChatKind.GROUP,
                                rng=random.Random(3))
        self.assertEqual(len(out), 1)
        self.assertIn(out[0], (
            "😂", "lol 😅", "😅", "abeg 😂", "😂😂", "ehn 😅"))
        self.assertNotIn("analyze", out[0])

    def test_all_scaffold_dm_gets_one_reaction(self) -> None:
        out = guard_reply_parts(
            ["let me analyze this properly — he's tired"],
            kind=ChatKind.DM, rng=random.Random(3))
        self.assertEqual(len(out), 1)
        self.assertIn(out[0], ("😅", "lol 😅", "hmm 😅", "😂"))

    def test_rejection_is_journaled_as_hash_not_text(self) -> None:
        guard_reply_parts([INCIDENT], kind=ChatKind.GROUP,
                          rng=random.Random(1))
        entries = [f for f in recent_failures(10)
                   if "scaffold_guard" in f.site]
        self.assertEqual(len(entries), 1)
        self.assertIn("sha256=", entries[0].message)
        self.assertIn("analysis header", entries[0].message)
        # The scaffold's own words never enter the ledger.
        self.assertNotIn("sucidal", entries[0].message)
        self.assertNotIn("u dey smoke", entries[0].message)

    def test_fallback_reaction_pools(self) -> None:
        rng = random.Random(7)
        for _ in range(20):
            self.assertIn(fallback_reaction(ChatKind.GROUP, rng),
                          ("😂", "lol 😅", "😅", "abeg 😂", "😂😂", "ehn 😅"))
            self.assertIn(fallback_reaction(ChatKind.DM, rng),
                          ("😅", "lol 😅", "hmm 😅", "😂"))


# ── the send boundary, on both personal surfaces ─────────────────────


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
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

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: list[list[tuple[str, str]]] | None = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def wait_started(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.started_flag:
                return True
            time.sleep(0.01)
        return False


class FakeRouter:
    """Scripted LLM: returns the scaffold for every real prompt."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message],
             params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        self.calls.append(list(messages))
        text_blob = " ".join(m.content for m in messages)
        if "memory-distillation" in text_blob:
            return LLMResponse(
                text=json.dumps({
                    "summary": "Banter in a group chat.",
                    "facts": [],
                    "milestone": "",
                    "mood_after": "happy",
                }),
                model="fake-curator",
            )
        if self.replies:
            return LLMResponse(text=self.replies.pop(0), model="fake-7b")
        return LLMResponse(text="mhm, tell me more", model="fake-7b")


def _make_context() -> tuple[Any, "tempfile.TemporaryDirectory"]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-")
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.enabled": "false",
        }
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    _ws = Path(context.settings.workspace_dir)
    for _marker in ("pulse_last_run.json", "pulse_last_run_afternoon.json",
                    "pulse_last_run_evening.json",
                    "pulse_last_run_midnight.json"):
        (_ws / _marker).write_text(
            json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    from nomorals.agents.morning_briefing import _today_str
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json, generated_at)"
        " VALUES (?, ?, '[]', ?)",
        ("test-seed", _today_str(context), time.time()),
    )
    return context, tmp


def _wait(predicate, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class SendBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        # Scripted BEFORE the runtime builds its brain around the
        # router: the full-turn test's model IS the scaffold.
        self.context.router = FakeRouter([INCIDENT])
        self.adapters = {
            name: FakeAdapter(name)
            for name in ("whatsapp", "telegram-personal")
        }
        self.gateway = ChatGateway(self.adapters, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        # Deterministic immediate replies (no presence busy-gap).
        self.runtime.brain.presence_rng = random.Random(34)
        # The guard, not the typing simulation, is under test here.
        self.runtime._typing_enabled = lambda chat: False

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: E103 - best-effort teardown
            pass
        self.context.close()
        self.tmp.cleanup()

    def _message(self, platform: str, kind: ChatKind,
                 chat_id: str = "room1") -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                       title="Pinnocent 7.2.0", peer=chat_id)
        return ChatMessage(chat=chat, incoming=True, text="u dey smoke",
                           sender="ekpo")

    def test_whatsapp_group_scaffold_never_sent(self) -> None:
        msg = self._message("whatsapp", ChatKind.GROUP)
        self.runtime._send_reply(msg, [INCIDENT])
        sent = self.adapters["whatsapp"].sent
        self.assertEqual(len(sent), 1)
        self.assertNotIn("analyze", sent[0])
        self.assertNotIn("vibe", sent[0])

    def test_telegram_personal_group_scaffold_never_sent(self) -> None:
        msg = self._message("telegram-personal", ChatKind.GROUP)
        self.runtime._send_reply(msg, [INCIDENT])
        sent = self.adapters["telegram-personal"].sent
        self.assertEqual(len(sent), 1)
        self.assertNotIn("analyze", sent[0])

    def test_whatsapp_dm_scaffold_never_sent(self) -> None:
        msg = self._message("whatsapp", ChatKind.DM, chat_id="peer1")
        self.runtime._send_reply(
            msg, ["let me analyze this properly — he's tired"])
        sent = self.adapters["whatsapp"].sent
        self.assertEqual(len(sent), 1)
        self.assertNotIn("analyze", sent[0])

    def test_clean_reply_passes_byte_identical(self) -> None:
        msg = self._message("whatsapp", ChatKind.GROUP)
        clean = "u dey smoke ke? 😂 na play i dey play o"
        self.runtime._send_reply(msg, [clean])
        self.assertEqual(self.adapters["whatsapp"].sent, [clean])

    def test_composed_reply_wins_over_fallback(self) -> None:
        msg = self._message("telegram-personal", ChatKind.GROUP)
        self.runtime._send_reply(msg, [INCIDENT, "real answer here o"])
        self.assertEqual(
            self.adapters["telegram-personal"].sent, ["real answer here o"])

    def test_full_turn_with_scaffold_brain_sends_no_scaffold(self) -> None:
        # Placeholder→final analogue: the brain's final generation IS
        # the scaffold; what lands in the chat must be the fallback,
        # never the analysis. Group speak needs a mention of her name.
        self.runtime.start()
        adapter = self.adapters["whatsapp"]
        self.assertTrue(adapter.wait_started())
        group = ChatRef(platform="whatsapp", chat_id="grp-wa",
                        kind=ChatKind.GROUP, title="Pinnocent 7.2.0",
                        peer="grp-wa")
        self.runtime.on_message(ChatMessage(
            chat=group, incoming=True,
            text="wren, wetin you think — u dey smoke?", sender="ekpo"))
        self.assertTrue(_wait(lambda: len(adapter.sent) >= 1),
                        "no reply was sent")
        for text in adapter.sent:
            self.assertNotIn("analyze the user's input", text)
            self.assertNotIn("the vibe is", text)
        # The shared ledger ring also collects background noise from
        # the runtime's other organs; scan it whole.
        self.assertTrue(any("scaffold_guard" in f.site
                            for f in recent_failures(200)))


if __name__ == "__main__":
    unittest.main()
