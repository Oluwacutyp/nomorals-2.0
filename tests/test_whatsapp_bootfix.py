"""Boot-flood fix pins (2026-11-04).

The incident (owner's live WhatsApp self-chat, ~13:56 MST): on
startup the full command catalog landed in the self-chat split into
~13 chunks, unsolicited; his /help after it got NO reply; and it all
happened while his providers were degraded.

Traced mechanisms, each pinned below:

* WhatsApp delivers messages composed while a linked device was
  offline the moment the bridge (re)connects, and the adapter had
  NO staleness check — a /help typed during a restart was processed
  as a fresh live turn at boot (the flood), and queued outbox
  replies replayed the same way (24h TTL).  Stale inbound is now
  stamped ``stale_replay`` and observed-only in the runtime; the
  outbox ages entries out at 300s, journaled, never delivered stale.
* The reply to his next /help was eaten by the send-once guard: an
  identical chunk whose bridge ack was lost in the boot flap stays
  guarded for 600s, and the fresh identical reply aborted on it.
  A fresh solicited reply (gateway ``fresh=True``, first attempt
  only) now releases the guard; retries stay guarded.
* The 4000-char WhatsApp cap was Devon-invented; the protocol cap
  is 65,536 and neither bridge imposes less.  A ~20k reply is ONE
  message on WhatsApp now; Telegram still chunks at its real cap.
* Unsolicited sends (Notifier proactive pushes) are held while the
  brain is degraded, via the same brain-gate episode journaling.
* /chatmode works from the WhatsApp self-chat, keyed per
  platform:chat — it cannot touch a Telegram chat's settings.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.notifier import Notifier, STATE_HELD_BRAIN
from nomorals.agents.partner.brain_gate import BrainGate
from nomorals.agents.partner.failures import clear_failures, recent_failures
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.partner import chat_modes
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.platform_limits import for_platform
from nomorals.social.chat.send_retry import BackoffPolicy
from nomorals.social.chat.whatsapp import (
    STALE_INBOUND_S,
    UNCONFIRMED_SEND_ERROR,
    Outbox,
    WhatsAppAdapter,
    _split_text,
)

#: Honour NM_TEST_TMPDIR like the other full-context suites: context
#: homes carry a fully migrated database and must not land on a
#: small shared /tmp tmpfs (see test_partner_brain_gate).
_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

SELF_JID = "2348012345678@c.us"
SELF_KEY = f"whatsapp:{SELF_JID}"


def _tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="nm-bootfix-", dir=_TEST_TMPDIR)


# ── adapter: stale inbound stamping ─────────────────────────────────


class StaleInboundStampTests(unittest.TestCase):
    def _adapter(self) -> tuple[WhatsAppAdapter, list[ChatMessage]]:
        adapter = WhatsAppAdapter(outbox_enabled=False)
        adapter.user_jid = SELF_JID
        received: list[ChatMessage] = []
        return adapter, received

    def _dispatch(self, adapter: WhatsAppAdapter,
                  received: list[ChatMessage], obj: dict) -> None:
        adapter._dispatch(obj, received.append)

    def test_old_message_stamped_stale(self) -> None:
        adapter, received = self._adapter()
        old_ms = int((time.time() - STALE_INBOUND_S - 3600) * 1000)
        self._dispatch(adapter, received, {
            "type": "message", "id": "wamid.old",
            "chat": {"id": "555@s.whatsapp.net", "kind": "dm"},
            "from": {"id": "555@s.whatsapp.net", "name": "Ada"},
            "text": "hello from before the restart", "ts": old_ms,
        })
        self.assertEqual(len(received), 1)
        self.assertTrue(received[0].meta.get("stale_replay"))

    def test_fresh_message_not_stamped(self) -> None:
        adapter, received = self._adapter()
        self._dispatch(adapter, received, {
            "type": "message", "id": "wamid.new",
            "chat": {"id": "555@s.whatsapp.net", "kind": "dm"},
            "from": {"id": "555@s.whatsapp.net", "name": "Ada"},
            "text": "hello now", "ts": int(time.time() * 1000),
        })
        self.assertEqual(len(received), 1)
        self.assertNotIn("stale_replay", received[0].meta)

    def test_old_self_chat_turn_stamped_stale(self) -> None:
        adapter, received = self._adapter()
        old_ms = int((time.time() - STALE_INBOUND_S - 3600) * 1000)
        self._dispatch(adapter, received, {
            "type": "own_message", "id": "wamid.selfold",
            "chat": {"id": SELF_JID, "kind": "dm"},
            "text": "/help", "name": "owner", "ts": old_ms,
        })
        self.assertEqual(len(received), 1)
        self.assertTrue(received[0].meta.get("self_chat"))
        self.assertTrue(received[0].meta.get("stale_replay"))


# ── adapter: the send-once guard vs a fresh solicited reply ─────────


class SendOnceGuardFreshTests(unittest.TestCase):
    def _adapter(self) -> tuple[WhatsAppAdapter, list[dict]]:
        adapter = WhatsAppAdapter(outbox_enabled=False)
        adapter.connected.set()
        calls: list[dict] = []
        state = {"send_outcome": {"ok": True, "message_id": "wamid.ok"}}

        def fake_send_cmd(cmd: dict, *, timeout: float = 25.0) -> dict:
            calls.append(cmd)
            if cmd.get("cmd") == "history":
                return {"ok": True, "messages": []}
            return dict(state["send_outcome"])

        adapter._send_cmd = fake_send_cmd  # type: ignore[method-assign]
        return adapter, calls

    def test_guard_holds_repeat_but_fresh_reply_dispatches(self) -> None:
        adapter, calls = self._adapter()
        chat = ChatRef(platform="whatsapp", chat_id=SELF_JID,
                       kind=ChatKind.DM)
        text = "the menu text, chunk one"
        # A dispatch whose ack was lost (the boot flap) guards the key.
        adapter._send_cmd = (  # type: ignore[method-assign]
            lambda cmd, *, timeout=25.0:
            {"ok": False, "ambiguous": True,
             "error": "bridge response timeout"}
            if cmd.get("cmd") == "send" else {"ok": True, "messages": []})
        result, known = adapter._send_text_outcome(chat, text)
        self.assertFalse(result.ok)
        self.assertFalse(known)
        self.assertEqual(result.error, UNCONFIRMED_SEND_ERROR)

        # A retry-layer repeat (not fresh) is still refused — and the
        # bridge is never asked to send it again.
        adapter2_calls: list[dict] = []

        def counting_cmd(cmd: dict, *, timeout: float = 25.0) -> dict:
            adapter2_calls.append(cmd)
            if cmd.get("cmd") == "history":
                return {"ok": True, "messages": []}
            return {"ok": True, "message_id": "wamid.ok"}

        adapter._send_cmd = counting_cmd  # type: ignore[method-assign]
        refused = adapter.send(chat, text)
        self.assertFalse(refused.ok)
        self.assertEqual(refused.error, UNCONFIRMED_SEND_ERROR)
        self.assertEqual(
            [c for c in adapter2_calls if c.get("cmd") == "send"], [])

        # The owner's fresh /help composes the identical reply: it
        # dispatches. Exactly one bridge send, result ok.
        sent = adapter.send(chat, text, fresh=True)
        self.assertTrue(sent.ok)
        sends = [c for c in adapter2_calls if c.get("cmd") == "send"]
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0]["text"], text)


# ── gateway: fresh reaches the adapter on the first attempt only ────


class _FreshRecordingAdapter:
    """Adapter double whose send declares ``fresh`` (WhatsApp-shaped)."""

    name = "whatsapp"

    def __init__(self) -> None:
        self.fresh_seen: list[bool] = []
        self.fail_first = True

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             fresh: bool = False) -> SendResult:
        self.fresh_seen.append(fresh)
        if self.fail_first and len(self.fresh_seen) == 1:
            return SendResult(ok=False, platform="whatsapp",
                              error="bridge response timeout")
        return SendResult(ok=True, platform="whatsapp", message_id="m1")


class GatewayFreshTests(unittest.TestCase):
    def test_fresh_first_attempt_only(self) -> None:
        adapter = _FreshRecordingAdapter()
        gateway = ChatGateway(
            adapters={"whatsapp": adapter},
            retry_policy=BackoffPolicy(base=0.01, factor=1.0, cap=0.05))
        chat = ChatRef(platform="whatsapp", chat_id=SELF_JID,
                       kind=ChatKind.DM)
        result = gateway.send("whatsapp", chat, "hello", fresh=True)
        self.assertTrue(result.ok)
        # attempt 1 (transient failure) carried fresh=True; the
        # gateway's own retry re-entered with fresh=False.
        self.assertEqual(adapter.fresh_seen, [True, False])

    def test_fresh_survives_the_durable_outbox(self) -> None:
        # Production's gateway is durable (runtime builds it with
        # durable_outbox=True): the fresh mark must ride the outbox
        # row and reach the adapter on the row's first delivery —
        # otherwise the guard fix would never fire live.
        from nomorals.storage.db import Database

        db = Database(":memory:")
        db.migrate()
        adapter = _FreshRecordingAdapter()
        adapter.fail_first = False
        gateway = ChatGateway(
            adapters={"whatsapp": adapter}, db=db, durable_outbox=True,
            retry_policy=BackoffPolicy(base=0.01, factor=1.0, cap=0.05))
        chat = ChatRef(platform="whatsapp", chat_id=SELF_JID,
                       kind=ChatKind.DM)
        result = gateway.send("whatsapp", chat, "hello", fresh=True)
        self.assertTrue(result.ok)
        self.assertEqual(adapter.fresh_seen, [True])
        # An unmarked send through the same path stays unmarked.
        result = gateway.send("whatsapp", chat, "again")
        self.assertTrue(result.ok)
        self.assertEqual(adapter.fresh_seen, [True, False])


# ── the real WhatsApp cap ────────────────────────────────────────────


class WhatsAppCapTests(unittest.TestCase):
    def test_platform_cap_is_the_protocol_cap(self) -> None:
        self.assertEqual(for_platform("whatsapp").max_text, 65536)
        self.assertEqual(for_platform("telegram").max_text, 4096)

    def test_20k_reply_is_one_whatsapp_message(self) -> None:
        text = ("a long reply line that keeps going. " * 600).strip()
        self.assertGreater(len(text), 20000)
        self.assertEqual(_split_text(text), [text])

    def test_20k_still_chunks_at_telegram_cap(self) -> None:
        text = ("a long reply line that keeps going. " * 600).strip()
        chunks = _split_text(text, limit=for_platform("telegram").max_text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(len(c) <= 4096 for c in chunks))

    def test_adapter_send_of_20k_is_one_bridge_send(self) -> None:
        adapter = WhatsAppAdapter(outbox_enabled=False)
        adapter.connected.set()
        calls: list[dict] = []

        def fake_send_cmd(cmd: dict, *, timeout: float = 25.0) -> dict:
            calls.append(cmd)
            return {"ok": True, "message_id": "wamid.big"}

        adapter._send_cmd = fake_send_cmd  # type: ignore[method-assign]
        chat = ChatRef(platform="whatsapp", chat_id=SELF_JID,
                       kind=ChatKind.DM)
        text = ("menu section body. " * 1100).strip()  # ~22k chars
        result = adapter.send(chat, text)
        self.assertTrue(result.ok)
        sends = [c for c in calls if c.get("cmd") == "send"]
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0]["text"], text)


# ── outbox: aged entries are dropped, never delivered stale ──────────


class OutboxAgingTests(unittest.TestCase):
    def test_entry_past_the_reply_window_is_dropped_not_delivered(
            self) -> None:
        with _tmpdir() as tmp:
            box = Outbox(os.path.join(tmp, "whatsapp.jsonl"))
            box.enqueue({"kind": "text", "chat_id": SELF_JID,
                         "chat_kind": "dm", "text": "the catalog"})
            # Age the entry past the window, as a restart would find it.
            rows = box.pending()
            rows[0]["enqueued_at"] = time.time() - 3600
            box._write_all(rows)
            delivered: list[dict] = []
            report = box.flush(lambda e: delivered.append(e) or True)
            self.assertEqual(delivered, [])
            self.assertEqual(report["dropped"], 1)
            self.assertEqual(box.pending(), [])

    def test_fresh_entry_still_flushes(self) -> None:
        with _tmpdir() as tmp:
            box = Outbox(os.path.join(tmp, "whatsapp.jsonl"))
            box.enqueue({"kind": "text", "chat_id": SELF_JID,
                         "chat_kind": "dm", "text": "a live reply"})
            delivered: list[dict] = []
            report = box.flush(lambda e: delivered.append(e) or True)
            self.assertEqual(len(delivered), 1)
            self.assertEqual(report["sent"], 1)


# ── runtime harness: /help end-to-end, stale silence, /chatmode ─────


class _RecordingGateway:
    adapters: dict = {}

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def status(self) -> dict:
        return {}

    def set_rate_limit(self, *a: Any, **k: Any) -> None:
        pass

    def typing(self, *a: Any, **k: Any) -> None:
        pass

    def send(self, platform: str, chat: Any, text: str,
             **kwargs: Any) -> SimpleNamespace:
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, error="")


def _self_chat_msg(text: str, *, mid: str,
                   stale: bool = False) -> ChatMessage:
    meta: dict[str, Any] = {"self_chat": True, "from_owner": True}
    if stale:
        meta["stale_replay"] = True
    return ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=SELF_JID,
                     kind=ChatKind.DM, peer=SELF_JID),
        incoming=True, text=text, sender="owner", sender_id=SELF_JID,
        message_id=mid, ts=time.time(), meta=meta)


class RuntimeBootfixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tmpdir()
        self.modes_path = os.path.join(self.tmp.name, "chat_modes.json")
        self._old_modes = os.environ.get("NOMORALS_CHAT_MODES_PATH")
        os.environ["NOMORALS_CHAT_MODES_PATH"] = self.modes_path
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.quiet_start": "0",
            "partner.quiet_end": "0",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.gateway = _RecordingGateway()
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        if self._old_modes is None:
            os.environ.pop("NOMORALS_CHAT_MODES_PATH", None)
        else:
            os.environ["NOMORALS_CHAT_MODES_PATH"] = self._old_modes
        self.tmp.cleanup()

    def test_help_answers_and_answers_again(self) -> None:
        # The incident's second half: /help after the menu was
        # already sent once in the session must STILL answer —
        # exactly one reply per ask.
        self.runtime._process(_self_chat_msg("/help", mid="wa-h1"))
        wa_sends = [t for p, t in self.gateway.sent if p == "whatsapp"]
        self.assertEqual(len(wa_sends), 1)
        self.assertIn("what I can do", wa_sends[0])
        self.runtime._process(_self_chat_msg("/help", mid="wa-h2"))
        wa_sends = [t for p, t in self.gateway.sent if p == "whatsapp"]
        self.assertEqual(len(wa_sends), 2)
        self.assertIn("what I can do", wa_sends[1])

    def test_stale_help_is_observed_never_answered(self) -> None:
        # The boot flood itself: attach delivers the offline /help —
        # zero sends to the self-chat.
        self.runtime._process(
            _self_chat_msg("/help", mid="wa-stale", stale=True))
        self.assertEqual(self.gateway.sent, [])
        self.assertEqual(self.runtime.stats.get("stale_held"), 1)

    def test_stale_plain_text_sends_nothing(self) -> None:
        self.runtime._process(
            _self_chat_msg("are you there", mid="wa-stale2", stale=True))
        self.assertEqual(self.gateway.sent, [])

    # ── /chatmode parity on WhatsApp ──

    def test_chatmode_from_whatsapp_self_chat(self) -> None:
        out = self.runtime.handle_control(
            "/chatmode quiet", SELF_KEY,
            message=_self_chat_msg("/chatmode quiet", mid="wa-cm1"))
        self.assertIn("quiet", out)
        self.assertTrue(chat_modes.is_quiet(SELF_KEY))

    def test_chatmode_is_keyed_per_platform_chat(self) -> None:
        self.runtime.handle_control(
            "/chatmode quiet", SELF_KEY,
            message=_self_chat_msg("/chatmode quiet", mid="wa-cm2"))
        self.assertTrue(chat_modes.is_quiet(SELF_KEY))
        # The same chat id on Telegram is a different settings row.
        self.assertFalse(chat_modes.is_quiet(f"telegram:{SELF_JID}"))
        self.assertEqual(
            chat_modes.get_chat_mode(f"telegram:{SELF_JID}"), "normal")
        # A feature toggle on the WhatsApp key stays there too.
        out = self.runtime.handle_control(
            "/chatmode feature music off", SELF_KEY,
            message=_self_chat_msg("/chatmode feature music off",
                                   mid="wa-cm3"))
        self.assertIn("music", out)
        from nomorals.partner.chat_modes import get_chat_settings
        self.assertIn("music", get_chat_settings(SELF_KEY)["disabled"])
        self.assertNotIn(
            "music", get_chat_settings(f"telegram:{SELF_JID}")["disabled"])


# ── brain gate: unsolicited sends held while degraded ────────────────


class UnsolicitedGateTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_failures()

    def _braingate_entries(self) -> list:
        return [f for f in recent_failures(200) if "braingate" in f.site]

    def test_hold_unsolicited_only_when_degraded_and_personal(self) -> None:
        gate = BrainGate()
        # Nothing observed yet: no evidence of degradation, no hold.
        self.assertFalse(gate.hold_unsolicited(
            platform="whatsapp", chat_key="whatsapp:x"))
        gate.observe_outcome(True)
        self.assertTrue(gate.degraded_now)
        self.assertTrue(gate.hold_unsolicited(
            platform="whatsapp", chat_key="whatsapp:x",
            note="proactive push"))
        # One journal entry for the episode, however many holds.
        self.assertTrue(gate.hold_unsolicited(
            platform="whatsapp", chat_key="whatsapp:x"))
        self.assertEqual(len(self._braingate_entries()), 1)
        # A bot surface is never held by this path.
        self.assertFalse(gate.hold_unsolicited(
            platform="telegram-bot", chat_key="telegram-bot:x"))
        # Recovery: the flag clears and sends flow again.
        gate.observe_outcome(False)
        self.assertFalse(gate.degraded_now)
        self.assertFalse(gate.hold_unsolicited(
            platform="whatsapp", chat_key="whatsapp:x"))

    def test_notifier_proactive_push_held_while_degraded(self) -> None:
        tmp = _tmpdir()
        self.addCleanup(tmp.cleanup)
        settings = load_settings(overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.quiet_start": "0",
            "partner.quiet_end": "0",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)
        self.addCleanup(context.close)
        gate = BrainGate()
        context.extras["brain_gate"] = gate
        gateway = _RecordingGateway()
        notifier = Notifier(context, gateway=gateway)

        gate.observe_outcome(True)
        held = notifier.publish("briefing", "Morning briefing", "body")
        self.assertEqual(held["delivery_state"], STATE_HELD_BRAIN)
        self.assertEqual(held["delivered"], 0)
        self.assertEqual(gateway.sent, [])

        gate.observe_outcome(False)
        flowed = notifier.publish("briefing", "Evening briefing", "body")
        self.assertNotEqual(flowed["delivery_state"], STATE_HELD_BRAIN)


if __name__ == "__main__":
    unittest.main()
