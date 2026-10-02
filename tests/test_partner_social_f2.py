"""Wave F2 partner/social: deeper lexicon wiring, anti-echo, gating uniformity, typing timing.

Covers the four F2 workstreams in nomorals/partner + nomorals/social/chat:

* **Lexicon enrichment** — dynamic terms blended into the persona's own
  speech block (one authoritative line), excludable from the voice note,
  measurable via ``lexicon_hits`` / ``lexicon_terms_surfaced``; the owner
  kill-switch keeps the prompt byte-identical.
* **Anti-echo** — ``repair_echo`` strips the user's phrasing from a draft
  that still echoes after every retry; unrepairable echoes are dropped
  for an in-character fallback line (never shipped). ``echo_action`` on
  the bundle records what happened.
* **Gating uniformity** — one owner test (``is_owner_chat``) shared by
  the gateway and the runtime, one gating decision (``gate_decision``);
  owner DM / owner-originated / stranger / group / channel rules.
* **Typing timing** — ``human_typing_seconds`` scales with length; the
  Telegram bot adapter holds the indicator for the full requested
  window; the web adapter exposes a real typing window; local/webhook
  honestly report no indicator.
"""

from __future__ import annotations

import random
import time
import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.llm.base import LLMResponse
from nomorals.partner import (
    PartnerContextBuilder,
    PartnerResponder,
    default_persona,
    lexicon_hits,
    parrot_check,
    repair_echo,
)
from nomorals.partner.gating import (
    MODE_GROUP,
    MODE_OWNER,
    MODE_PRIVATE,
    classify_chat,
    gate_decision,
    is_owner_chat,
    is_restricted,
)
from nomorals.partner.lexicon_feed import LexiconFeed, seed_partner_lexicon
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import SpeechProfile
from nomorals.partner.presence import human_typing_seconds
from nomorals.partner.relationship import Relationship
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.local import LocalAdapter
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.social.chat.web import WebAdapter
from nomorals.social.chat.webhook import WebhookAdapter
from nomorals.storage.db import Database


class FakeRouter:
    """Scripted stand-in LLM router."""

    def __init__(self, replies: list[str] | None = None, *, fail: bool = False) -> None:
        self.replies = list(replies or [])
        self.fail = fail
        self.calls: list[list] = []

    def chat(self, messages, params=None, **kw):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("provider down")
        text = self.replies.pop(0) if self.replies else "mhm, okay"
        return LLMResponse(text=text, model="fake")


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


def _responder(router: FakeRouter, db: Any, *, lexicon: Any = "feed",
               retries: int = 2, rng_seed: int = 0) -> PartnerResponder:
    persona = default_persona()
    mood = MoodEngine(persona.baselines, now=1_000_000.0)
    feed = LexiconFeed(db) if lexicon == "feed" else lexicon
    return PartnerResponder(
        router, persona, mood, Relationship(), None, None,
        rng=random.Random(rng_seed), lexicon=feed, retries=retries,
    )


def _system_prompt(router: FakeRouter) -> str:
    return str(router.calls[-1][0].content)


# ── 1. deeper dynamic lexicon use ─────────────────────────────────────────────

class LexiconEnrichmentTest(unittest.TestCase):
    def test_speech_prompt_default_byte_identical(self) -> None:
        # Kill-switch off state: no dynamic banks -> the static prompt is
        # exactly what it was before (no pet-name line appears from nowhere).
        prompt = SpeechProfile().to_prompt()
        self.assertIn(
            "- The only catchphrases you ever use, and rarely: "
            "'ok real', 'ugh', 'fine, fine', 'okay but hear me out'.",
            prompt,
        )
        self.assertNotIn("What you call them", prompt)

    def test_speech_prompt_blends_dynamic_banks(self) -> None:
        prompt = SpeechProfile().to_prompt(
            dynamic_catchphrases=("okay wait", "ok real"),  # "ok real" dupes static
            dynamic_pet_names=("love",),
        )
        # Static bank first (owner's base), dynamic appended, deduped.
        self.assertIn("'ok real', 'ugh', 'fine, fine', 'okay but hear me out', 'okay wait'",
                      prompt)
        self.assertEqual(prompt.count("'ok real'"), 1)
        self.assertIn("What you call them, when a pet name fits: 'babe', 'you', 'hey you', 'love'.",
                      prompt)

    def test_builder_threads_dynamic_banks_into_persona_block(self) -> None:
        persona = default_persona()
        mood = MoodEngine(persona.baselines, now=1_000_000.0)
        builder = PartnerContextBuilder()
        msg = builder.build(
            persona=persona, mood=mood, relationship=Relationship(),
            dynamic_catchphrases=("okay wait",), dynamic_pet_names=("love",),
        )
        text = str(msg.content)
        self.assertIn("okay wait", text)
        self.assertIn("What you call them", text)

    def test_voice_note_exclude_skips_categories(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        feed = LexiconFeed(db)
        full_note, full_used, _ = feed.voice_note(default_persona(), "calm")
        self.assertIn("okay wait", full_note)
        note, used, fallback = feed.voice_note(
            default_persona(), "calm", exclude=("catchphrase", "pet_name")
        )
        self.assertNotIn("okay wait", note)
        self.assertNotIn("catchphrases you actually use", note)
        self.assertLess(used, full_used)
        self.assertNotIn("catchphrase", fallback)
        self.assertNotIn("pet_name", fallback)

    def test_responder_prompt_has_single_authoritative_catchphrase_line(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        router = FakeRouter()
        responder = _responder(router, db, lexicon="feed")
        responder.respond(chat_platform="local", user_text="hey")
        prompt = _system_prompt(router)
        # The seeded catchphrase appears exactly once: in the persona's own
        # speech block, not duplicated across a second voice-note line.
        self.assertEqual(prompt.count("okay wait"), 1)
        self.assertNotIn("catchphrases you actually use", prompt)

    def test_lexicon_hits_counts_visible_terms(self) -> None:
        self.assertEqual(
            lexicon_hits("okay wait, tell me everything", ("okay wait", "nope")), 1
        )
        self.assertEqual(lexicon_hits("nothing here", ("okay wait",)), 0)
        self.assertEqual(lexicon_hits("", ("okay wait",)), 0)
        # Word-boundary: "wait" alone must not match "okay wait"'s "wait".
        self.assertEqual(lexicon_hits("wait for me", ("okay wait",)), 0)

    def test_bundle_reports_surfaced_terms(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        router = FakeRouter(replies=["okay wait, that's actually wild"])
        responder = _responder(router, db, lexicon="feed")
        bundle = responder.respond(chat_platform="local", user_text="guess what")
        self.assertGreaterEqual(bundle.lexicon_terms_surfaced, 1)
        self.assertGreaterEqual(
            bundle.to_dict()["lexicon_terms_surfaced"], 1
        )

    def test_kill_switch_prompt_untouched(self) -> None:
        db = _db()
        seed_partner_lexicon(db)
        router = FakeRouter()
        responder = _responder(router, db, lexicon=None)
        bundle = responder.respond(chat_platform="local", user_text="hey")
        prompt = _system_prompt(router)
        self.assertNotIn("okay wait", prompt)
        self.assertIn("ok real", prompt)  # owner's static bank intact
        self.assertFalse(bundle.lexicon_dynamic)
        self.assertEqual(bundle.lexicon_terms_used, 0)
        self.assertEqual(bundle.lexicon_terms_surfaced, 0)


# ── 2. anti-echo ──────────────────────────────────────────────────────────────

class AntiEchoTest(unittest.TestCase):
    def test_repair_echo_strips_user_phrasing(self) -> None:
        user = "i had the strangest dream about the ocean last night"
        draft = "oh wow that's wild, i had the strangest dream about the ocean last night too"
        repaired = repair_echo(user, draft)
        self.assertTrue(repaired)
        self.assertTrue(parrot_check(user, repaired).ok)
        self.assertNotIn("strangest dream", repaired.lower())

    def test_repair_echo_unrepairable_returns_empty(self) -> None:
        user = "the meeting is at noon tomorrow"
        self.assertEqual(repair_echo(user, user), "")
        self.assertEqual(repair_echo(user, "the meeting is at noon tomorrow!!"), "")

    def test_repair_echo_leaves_clean_draft_alone(self) -> None:
        draft = "oh that's rough, want to talk about it?"
        self.assertEqual(
            repair_echo("i failed my driving test today", draft), draft
        )

    def test_responder_repairs_echo_after_retries(self) -> None:
        user = "i had the strangest dream about the ocean last night"
        echo = ("oh wow that's wild, i had the strangest dream about the ocean "
                "last night too, tell me everything")
        self.assertFalse(parrot_check(user, echo).ok)  # it really echoes
        router = FakeRouter(replies=[echo])
        responder = _responder(router, None, lexicon=None, retries=0)
        bundle = responder.respond(chat_platform="local", user_text=user)
        self.assertEqual(bundle.echo_action, "repaired")
        self.assertFalse(bundle.fallback)
        self.assertTrue(parrot_check(user, bundle.text).ok)

    def test_responder_drops_unrepairable_echo(self) -> None:
        user = "the meeting is at noon tomorrow"
        router = FakeRouter(replies=[user])  # pure verbatim echo, twice over
        responder = _responder(router, None, lexicon=None, retries=0)
        bundle = responder.respond(chat_platform="local", user_text=user)
        # An echo must never ship: an in-character fallback line goes out
        # instead, and the bundle says so.
        self.assertEqual(bundle.echo_action, "dropped")
        self.assertTrue(bundle.fallback)
        self.assertNotEqual(bundle.text.strip().lower(), user)
        self.assertTrue(parrot_check(user, bundle.text).ok)

    def test_clean_reply_echo_action_empty(self) -> None:
        router = FakeRouter(replies=["oh no, what happened?"])
        responder = _responder(router, None, lexicon=None, retries=0)
        bundle = responder.respond(
            chat_platform="local", user_text="i failed my driving test today"
        )
        self.assertEqual(bundle.echo_action, "")
        self.assertFalse(bundle.fallback)


# ── 3. consistent DM vs group gating ──────────────────────────────────────────

class GatingUniformityTest(unittest.TestCase):
    def _ref(self, platform: str, chat_id: str, kind: str = ChatKind.DM) -> ChatRef:
        return ChatRef(platform=platform, chat_id=chat_id, kind=kind)

    def test_owner_key_set_marks_owner(self) -> None:
        chat = self._ref("telegram", "123")
        self.assertTrue(is_owner_chat(chat, owner_chats={"telegram:123"}))
        self.assertFalse(is_owner_chat(chat, owner_chats={"telegram:999"}))

    def test_db_flag_marks_owner(self) -> None:
        chat = self._ref("whatsapp", "456")
        self.assertTrue(is_owner_chat(chat, db_is_owner=True))
        self.assertFalse(is_owner_chat(chat, db_is_owner=False))

    def test_console_always_owner(self) -> None:
        self.assertTrue(is_owner_chat(self._ref("web", "console")))
        self.assertTrue(is_owner_chat(self._ref("local", "console")))
        # ...but a non-console chat on the same platform is not.
        self.assertFalse(is_owner_chat(self._ref("web", "other")))

    def test_stranger_dm_not_owner(self) -> None:
        self.assertFalse(is_owner_chat(self._ref("discord", " stranger-1 ".strip())))

    def test_gate_decision_owner_dm(self) -> None:
        chat = self._ref("telegram", "123")
        self.assertEqual(gate_decision(chat, is_owner=True), MODE_OWNER)
        self.assertFalse(is_restricted(MODE_OWNER))

    def test_gate_decision_stranger_dm(self) -> None:
        chat = self._ref("telegram", "999")
        self.assertEqual(gate_decision(chat, is_owner=False), MODE_PRIVATE)
        self.assertTrue(is_restricted(MODE_PRIVATE))

    def test_gate_decision_group_stranger(self) -> None:
        chat = self._ref("telegram", "-1001", kind=ChatKind.GROUP)
        self.assertEqual(gate_decision(chat, is_owner=False), MODE_GROUP)
        self.assertTrue(is_restricted(MODE_GROUP))

    def test_gate_decision_owner_originated_anywhere(self) -> None:
        # The owner's own message in a group still gets the full version.
        chat = self._ref("telegram", "-1001", kind=ChatKind.GROUP)
        self.assertEqual(gate_decision(chat, is_owner=True), MODE_OWNER)
        self.assertEqual(classify_chat(chat, is_owner=True), MODE_OWNER)

    def test_gate_decision_channel_is_group(self) -> None:
        chat = self._ref("telegram", "-1002", kind=ChatKind.CHANNEL)
        self.assertEqual(gate_decision(chat, is_owner=False), MODE_GROUP)

    def test_gate_decision_kill_switch(self) -> None:
        # gate_restricted_chats=False -> everything is owner mode.
        chat = self._ref("telegram", "-1001", kind=ChatKind.GROUP)
        self.assertEqual(
            gate_decision(chat, is_owner=False, restricted_enabled=False), MODE_OWNER
        )

    def test_modes_are_stable_strings(self) -> None:
        self.assertEqual((MODE_OWNER, MODE_PRIVATE, MODE_GROUP),
                         ("owner", "private", "group"))


# ── 4. typing indicator scaled by message length ──────────────────────────────

class TypingTimingTest(unittest.TestCase):
    def test_human_typing_scales_with_length(self) -> None:
        short = human_typing_seconds("hey", rng=random.Random(0))
        long = human_typing_seconds("x" * 600, rng=random.Random(0))
        # Same rng stream: the only difference is length, so the long
        # message must take clearly longer to "type".
        self.assertGreater(long, short)
        self.assertGreater(long, short * 5)
        self.assertGreaterEqual(short, 1.0)

    def test_human_typing_cap_respected(self) -> None:
        capped = human_typing_seconds("x" * 10000, cap=10.0, rng=random.Random(0))
        self.assertLessEqual(capped, 10.0)

    def test_telegram_bot_typing_holds_full_window(self) -> None:
        adapter = TelegramBotAdapter(token="test-token")
        calls: list[dict[str, Any]] = []
        adapter._api = lambda method, **kw: calls.append({"method": method, **kw}) or {"ok": True}  # noqa: SLF001
        chat = ChatRef(platform="telegram-bot", chat_id="123", kind=ChatKind.DM)
        started = time.time()
        self.assertTrue(adapter.typing(chat, seconds=5.0))
        elapsed = time.time() - started
        # 5s window with ~5s server-side expiry -> must re-fire at least twice.
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(c["method"] == "sendChatAction" for c in calls))
        self.assertGreaterEqual(elapsed, 4.5)

    def test_telegram_bot_typing_short_window_single_shot(self) -> None:
        adapter = TelegramBotAdapter(token="test-token")
        calls: list[dict[str, Any]] = []
        adapter._api = lambda method, **kw: calls.append({"method": method, **kw}) or {"ok": True}  # noqa: SLF001
        chat = ChatRef(platform="telegram-bot", chat_id="123", kind=ChatKind.DM)
        self.assertTrue(adapter.typing(chat, seconds=2.0))
        self.assertEqual(len(calls), 1)

    def test_telegram_bot_typing_zero_returns_false(self) -> None:
        adapter = TelegramBotAdapter(token="test-token")
        chat = ChatRef(platform="telegram-bot", chat_id="123", kind=ChatKind.DM)
        self.assertFalse(adapter.typing(chat, seconds=0))

    def test_web_typing_exposes_window(self) -> None:
        adapter = WebAdapter()
        chat = ChatRef(platform="web", chat_id="console", kind=ChatKind.DM)
        self.assertTrue(adapter.typing(chat, seconds=60.0))
        self.assertTrue(adapter.typing_active(chat.key))
        # Expired window reads as inactive.
        adapter._typing_until[chat.key] = time.time() - 1.0  # noqa: SLF001
        self.assertFalse(adapter.typing_active(chat.key))

    def test_web_typing_zero_returns_false(self) -> None:
        adapter = WebAdapter()
        chat = ChatRef(platform="web", chat_id="console", kind=ChatKind.DM)
        self.assertFalse(adapter.typing(chat, seconds=0))

    def test_local_typing_honestly_unsupported(self) -> None:
        # The console has no typing indicator surface; the adapter says so
        # instead of faking one.
        adapter = LocalAdapter()
        chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM)
        self.assertFalse(adapter.typing(chat, seconds=5.0))

    def test_webhook_typing_unsupported(self) -> None:
        adapter = WebhookAdapter()
        chat = ChatRef(platform="webhook", chat_id="1", kind=ChatKind.DM)
        self.assertFalse(adapter.typing(chat, seconds=5.0))


if __name__ == "__main__":
    unittest.main()
