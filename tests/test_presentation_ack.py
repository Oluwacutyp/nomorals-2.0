"""Presentation Phase 1 pins: receipt acknowledgment.

When Devon WILL answer an inbound message, the message first
visibly lands — a light reaction chosen by expression.pick_reaction
(never a fixed emoji), or, where no reaction can land, the
read-receipt rung. The constitution pinned here:

- the gateway carries react / mark_read passthroughs to the origin
  adapter, with the base adapter's honest refusal intact;
- the reaction fires exactly once per answered message — the meta
  flag and the per-boot id dedupe both hold;
- never when she won't answer (a stood-down command, a group turn
  she stays silent on);
- personal surfaces honor the §15 live gate: a standing owner stop
  blocks the act, a degraded brain is observe-only, and a landed
  act is journaled in the personal vault;
- DM replies anchor to the reply chain like group replies do;
- the slow-reply notice no longer claims she runs on a phone.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import PartnerSettings, load_settings
from nomorals.social.chat.base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway

OWNER = "2348012345678"  # the pairing suites' standing fake


class _FakeRouter:
    """Scripted LLM router (the partner-runtime suites' shape):
    answers the curator's distillation prompt with JSON and every
    conversational turn with the next scripted reply."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])

    def chat(self, messages, params=None, **kw):  # type: ignore[no-untyped-def]
        from nomorals.llm.base import LLMResponse
        blob = " ".join(m.content for m in messages)
        if "memory-distillation" in blob:
            return LLMResponse(
                text=json.dumps({
                    "summary": "A warm segment.",
                    "facts": [],
                    "milestone": "",
                    "mood_after": "happy",
                }),
                model="fake-curator")
        text = self.replies.pop(0) if self.replies else "mhm."
        return LLMResponse(text=text, model="fake-7b")


# ── adapters (real base-class shapes) ────────────────────────────

class _AckAdapter(ChatAdapter):
    """Carries both verbs; verdicts are configurable per test."""

    def __init__(self, name: str, *, react_ok: bool = True,
                 read_verdict: Any = True) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.reactions: list[tuple[str, str, str]] = []
        self.reads: list[str] = []
        self.sends: list[tuple[str, str]] = []
        self._react_ok = react_ok
        self._read_verdict = read_verdict

    def run(self, handler) -> None:  # pragma: no cover - unused
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             **kw: Any) -> SendResult:
        self.sends.append((text, reply_to))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"s{len(self.sends)}")

    def react(self, chat: ChatRef, message_id: str,
              emoji: str) -> SendResult:
        self.reactions.append((chat.key, message_id, emoji))
        return SendResult(ok=self._react_ok, platform=self.name,
                          message_id=message_id,
                          error="" if self._react_ok else "refused")

    def mark_read(self, chat: ChatRef) -> Any:
        self.reads.append(chat.key)
        return self._read_verdict


class _ReadOnlyAdapter(ChatAdapter):
    """No react verb (inherits the base refusal); mark_read only."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.reads: list[str] = []

    def run(self, handler) -> None:  # pragma: no cover - unused
        pass

    def send(self, chat: ChatRef, text: str, **kw: Any) -> SendResult:
        return SendResult(ok=True, platform=self.name, message_id="s1")

    def mark_read(self, chat: ChatRef) -> bool:
        self.reads.append(chat.key)
        return True


def _wait(predicate, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _msg(platform: str, chat_id: str, text: str, *,
         kind: str = ChatKind.DM, message_id: str = "m1",
         sender: str = "someone", reply_to: str = "") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=kind),
        incoming=True, text=text, sender=sender, media=[],
        reply_to=reply_to, mentioned=False, ts=time.time(),
        message_id=message_id)


# ── gateway passthroughs ─────────────────────────────────────────

class GatewayReceiptTests(unittest.TestCase):

    def test_react_routes_to_origin_adapter(self) -> None:
        adapter = _AckAdapter("telegram-bot")
        gw = ChatGateway({"telegram-bot": adapter})
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        result = gw.react("telegram-bot", chat, "42", "👀")
        self.assertTrue(result.ok)
        self.assertEqual(adapter.reactions,
                         [("telegram-bot:111", "42", "👀")])

    def test_react_base_refusal_is_honest(self) -> None:
        adapter = _ReadOnlyAdapter("telegram")
        gw = ChatGateway({"telegram": adapter})
        chat = ChatRef(platform="telegram", chat_id="111")
        result = gw.react("telegram", chat, "42", "👍")
        self.assertFalse(result.ok)
        self.assertIn("not supported", result.error)

    def test_react_no_adapter_and_dry_run(self) -> None:
        gw = ChatGateway({"whatsapp": _AckAdapter("whatsapp")})
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        self.assertFalse(gw.react("telegram-bot", chat, "1", "👍").ok)
        adapter = _AckAdapter("telegram-bot")
        dry = ChatGateway({"telegram-bot": adapter}, dry_run=True)
        self.assertFalse(dry.react("telegram-bot", chat, "1", "👍").ok)
        self.assertEqual(adapter.reactions, [])

    def test_mark_read_normalizes_adapter_shapes(self) -> None:
        chat = ChatRef(platform="whatsapp", chat_id="222")
        bool_adapter = _AckAdapter("whatsapp", read_verdict=True)
        gw = ChatGateway({"whatsapp": bool_adapter})
        self.assertTrue(gw.mark_read("whatsapp", chat))
        self.assertEqual(bool_adapter.reads, ["whatsapp:222"])
        dict_adapter = _AckAdapter("telegram", read_verdict={"ok": True})
        gw2 = ChatGateway({"telegram": dict_adapter})
        tg_chat = ChatRef(platform="telegram", chat_id="333")
        self.assertTrue(gw2.mark_read("telegram", tg_chat))
        read_only = _ReadOnlyAdapter("telegram")
        gw3 = ChatGateway({"telegram": read_only})
        self.assertTrue(gw3.mark_read("telegram", tg_chat))

    def test_mark_read_absent_verb_is_false(self) -> None:
        class _Bare(ChatAdapter):
            name = "sms"

            def run(self, handler) -> None:  # pragma: no cover
                pass

            def send(self, chat, text, **kw):  # pragma: no cover
                return SendResult(ok=True, platform=self.name)

        gw = ChatGateway({"sms": _Bare()})
        chat = ChatRef(platform="sms", chat_id="1")
        self.assertFalse(gw.mark_read("sms", chat))
        self.assertFalse(gw.mark_read("whatsapp", chat))


# ── config ───────────────────────────────────────────────────────

class ReceiptConfigTests(unittest.TestCase):

    def test_receipt_knobs_default_on(self) -> None:
        cfg = PartnerSettings()
        self.assertTrue(cfg.receipt_reactions)
        self.assertTrue(cfg.receipt_read_rung)


# ── the runtime hook ─────────────────────────────────────────────

class _RuntimeRig(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-ack-", dir=os.environ.get("NM_TEST_TMPDIR")
            or None)
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.enabled": "false",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)
        ws = Path(settings.workspace_dir)
        for marker in ("pulse_last_run.json",
                       "pulse_last_run_afternoon.json",
                       "pulse_last_run_evening.json",
                       "pulse_last_run_midnight.json"):
            (ws / marker).write_text(
                json.dumps({"ts": time.time(), "delivered_text": True}),
                encoding="utf-8")
        from nomorals.agents.morning_briefing import _today_str
        self.context.db.execute(
            "INSERT OR REPLACE INTO briefings"
            " (id, date, sections_json, generated_at)"
            " VALUES (?, ?, '[]', ?)",
            ("test-seed", _today_str(self.context), time.time()))

    def tearDown(self) -> None:
        import contextlib
        with contextlib.suppress(Exception):  # best-effort teardown
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _runtime(self, adapters: dict) -> PartnerRuntime:
        self.gateway = ChatGateway(adapters, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.runtime.brain.presence_rng = random.Random(34)
        return self.runtime

    def _pair_whatsapp(self) -> tuple[Any, Any, list]:
        """Rig the personal bundle: one linked WhatsApp account and
        a journal recorder standing on note_message_acted."""
        from nomorals.social.pairing import AccountRecord
        svc = self.runtime._pairing_personal()
        ws = Path(self.context.settings.workspace_dir)
        rec = svc["registry"].upsert(AccountRecord(
            id=f"whatsapp:{OWNER}", platform="whatsapp",
            identity=OWNER,
            creds_dir=str(ws / "data" / "whatsapp_sessions"
                          / f"wa-{OWNER}"),
            port=0, status="linked", created_at=1.0))
        journal: list[dict] = []
        personal = svc["personal"]
        real_note = personal.note_message_acted

        def _record(platform, **kw):
            journal.append({"platform": platform, **kw})
            return real_note(platform, **kw)

        personal.note_message_acted = _record
        self.addCleanup(
            setattr, personal, "note_message_acted", real_note)
        return svc, rec, journal


class ReceiptHookTests(_RuntimeRig):

    def test_dm_reaction_fires_once_end_to_end(self) -> None:
        self.context.router = _FakeRouter(["glad it landed"])
        adapter = _AckAdapter("telegram-bot")
        runtime = self._runtime({"telegram-bot": adapter})
        runtime.start()
        chat = ChatRef(platform="telegram-bot", chat_id="111",
                       kind=ChatKind.DM, peer="someone")
        runtime.on_message(ChatMessage(
            chat=chat, incoming=True,
            text="thanks, that was sweet", sender="someone",
            message_id="m1", ts=time.time()))
        self.assertTrue(_wait(lambda: len(adapter.reactions) == 1),
                        "no receipt reaction landed")
        # pick_reaction on the thanks context — her vocabulary,
        # not a hardcoded receipt glyph.
        self.assertEqual(adapter.reactions[0],
                         ("telegram-bot:111", "m1", "❤️"))
        self.assertTrue(_wait(lambda: len(adapter.sends) >= 1),
                        "the reply itself never sent")
        # Exactly once: a second answered message is a NEW ack;
        # the first message's ack never repeats.
        runtime.on_message(ChatMessage(
            chat=chat, incoming=True, text="and another thing",
            sender="someone", message_id="m2", ts=time.time()))
        self.assertTrue(_wait(lambda: len(adapter.reactions) == 2))
        time.sleep(0.3)
        self.assertEqual(
            [r[1] for r in adapter.reactions], ["m1", "m2"])

    def test_reaction_choice_comes_from_pick_reaction(self) -> None:
        adapter = _AckAdapter("whatsapp")
        runtime = self._runtime({"whatsapp": adapter})
        self._pair_whatsapp()
        runtime._acknowledge_inbound(_msg(
            "whatsapp", "222", "we won the jackpot!!",
            message_id="w1"))
        runtime._acknowledge_inbound(_msg(
            "whatsapp", "222", "that's funny lol", message_id="w2"))
        self.assertEqual([r[2] for r in adapter.reactions],
                         ["🏆", "😂"])

    def test_exactly_once_per_message_and_per_id(self) -> None:
        adapter = _AckAdapter("telegram-bot")
        runtime = self._runtime({"telegram-bot": adapter})
        msg = _msg("telegram-bot", "111", "hello there")
        runtime._acknowledge_inbound(msg)
        runtime._acknowledge_inbound(msg)  # same object, flag set
        twin = _msg("telegram-bot", "111", "hello there")  # redelivery
        runtime._acknowledge_inbound(twin)
        self.assertEqual(len(adapter.reactions), 1)

    def test_group_silence_gets_no_reaction(self) -> None:
        adapter = _AckAdapter("telegram-bot")
        runtime = self._runtime({"telegram-bot": adapter})
        runtime._group_gate = lambda message, what: True
        runtime.brain.should_speak_in_group = lambda message: False
        runtime._acknowledge_inbound(_msg(
            "telegram-bot", "555", "anyone around?",
            kind=ChatKind.GROUP))
        self.assertEqual(adapter.reactions, [])
        runtime.brain.should_speak_in_group = lambda message: True
        runtime._acknowledge_inbound(_msg(
            "telegram-bot", "555", "devon, you there?",
            kind=ChatKind.GROUP, message_id="g2"))
        self.assertEqual(len(adapter.reactions), 1)

    def test_stood_down_command_gets_no_reaction(self) -> None:
        adapter = _AckAdapter("telegram-bot")
        runtime = self._runtime({"telegram-bot": adapter})
        runtime._command_stands_down = lambda m, c: True
        runtime._acknowledge_inbound(_msg(
            "telegram-bot", "111", "/status"))
        self.assertEqual(adapter.reactions, [])
        runtime._command_stands_down = lambda m, c: False
        runtime._acknowledge_inbound(_msg(
            "telegram-bot", "111", "/status", message_id="m9"))
        self.assertEqual(len(adapter.reactions), 1)

    def test_owner_stop_blocks_and_skips_journal(self) -> None:
        adapter = _AckAdapter("whatsapp")
        runtime = self._runtime({"whatsapp": adapter})
        svc, _rec, journal = self._pair_whatsapp()
        svc["registry"].set_owner_stopped("whatsapp", True)
        runtime._acknowledge_inbound(_msg(
            "whatsapp", "222", "hello", message_id="s1"))
        self.assertEqual(adapter.reactions, [])
        self.assertEqual(adapter.reads, [])
        self.assertEqual(journal, [])
        # The stop lifts: the next message is acked AND journaled.
        svc["registry"].set_owner_stopped("whatsapp", False)
        runtime._acknowledge_inbound(_msg(
            "whatsapp", "222", "hello again", message_id="s2"))
        self.assertEqual(len(adapter.reactions), 1)
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["kind"], "message_reacted")
        self.assertEqual(journal[0]["account_id"], _rec.id)
        self.assertEqual(journal[0]["source_id"], "s2")

    def test_degraded_brain_is_observe_only_on_personal(self) -> None:
        adapter = _AckAdapter("whatsapp")
        runtime = self._runtime({"whatsapp": adapter})
        self._pair_whatsapp()
        gate = runtime._get_brain_gate()
        gate.observe_outcome(True)
        self.assertTrue(gate.degraded_now)
        runtime._acknowledge_inbound(_msg(
            "whatsapp", "222", "hello", message_id="d1"))
        self.assertEqual(adapter.reactions, [])
        self.assertEqual(adapter.reads, [])
        gate.observe_outcome(False)
        runtime._acknowledge_inbound(_msg(
            "whatsapp", "222", "hello", message_id="d2"))
        self.assertEqual(len(adapter.reactions), 1)

    def test_read_rung_when_reaction_cannot_land(self) -> None:
        adapter = _ReadOnlyAdapter("telegram")
        runtime = self._runtime({"telegram": adapter})
        runtime._acknowledge_inbound(_msg("telegram", "333", "hi"))
        self.assertEqual(adapter.reads, ["telegram:333"])
        # Reaction refused at the adapter → same fallback.
        refusing = _AckAdapter("whatsapp", react_ok=False)
        runtime2 = self._runtime({"whatsapp": refusing})
        self._pair_whatsapp()
        runtime2._acknowledge_inbound(_msg(
            "whatsapp", "222", "hi", message_id="r1"))
        self.assertEqual(len(refusing.reactions), 1)  # attempted
        self.assertEqual(refusing.reads, ["whatsapp:222"])

    def test_toggles_silence_the_ladder(self) -> None:
        adapter = _AckAdapter("whatsapp")
        runtime = self._runtime({"whatsapp": adapter})
        self._pair_whatsapp()
        cfg = self.context.settings.partner
        cfg.receipt_reactions = False
        runtime._acknowledge_inbound(_msg(
            "whatsapp", "222", "hi", message_id="t1"))
        self.assertEqual(adapter.reactions, [])
        self.assertEqual(adapter.reads, ["whatsapp:222"])
        cfg.receipt_read_rung = False
        runtime._acknowledge_inbound(_msg(
            "whatsapp", "222", "hi", message_id="t2"))
        self.assertEqual(adapter.reads, ["whatsapp:222"])

    def test_dm_reply_quotes_the_reply_chain(self) -> None:
        adapter = _AckAdapter("telegram-bot")
        runtime = self._runtime({"telegram-bot": adapter})

        class _GatewayStub:
            def __init__(self) -> None:
                self.sent: list[dict] = []

            def typing(self, platform, chat, seconds=0.0,
                       action="typing") -> bool:
                return True

            def send(self, platform, chat, text, **kw) -> SendResult:
                self.sent.append({"text": text, **kw})
                return SendResult(ok=True, platform=platform,
                                  message_id="s1")

        stub = _GatewayStub()
        runtime.gateway = stub
        dm = _msg("telegram-bot", "111", "what about this one?",
                  reply_to="q42")
        runtime._send_reply(dm, ["the answer"])
        self.assertEqual(stub.sent[0].get("reply_to"), "q42")
        plain = _msg("telegram-bot", "111", "fresh topic")
        runtime._send_reply(plain, ["no anchor needed"])
        self.assertEqual(stub.sent[1].get("reply_to"), "")
        group = _msg("telegram-bot", "555", "and this?",
                     kind=ChatKind.GROUP, reply_to="q7")
        runtime._send_reply(group, ["group answer"])
        self.assertEqual(stub.sent[2].get("reply_to"), "q7")


# ── copy ─────────────────────────────────────────────────────────

class SlowReplyCopyTests(unittest.TestCase):

    def test_slow_reply_notice_names_no_phone(self) -> None:
        src = (Path(__file__).resolve().parents[1]
               / "nomorals" / "agents" / "partner"
               / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("slow on the phone", src)
        self.assertIn("the model is slow right now", src)


if __name__ == "__main__":
    unittest.main()
