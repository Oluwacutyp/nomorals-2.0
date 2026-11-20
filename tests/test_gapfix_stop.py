"""Gapfix: /stop actually confirms and stops (owner live defect).

Regression, traced first (owner law 2026-11-14):

- The ``"/stop takes at most 1 argument(s)"`` refusal came from
  ``CONTROL_COMMANDS["stop"] == (1, 1)`` — exactly one argument
  demanded. Fixed by commit ``cb3ae57d`` (2026-11-14, ".../stop
  actually stops") to ``(0, None)``; that fix is an ancestor of
  the base, and the parse pins below guard it.
- The defect that REMAINED at the base: a self-stop (bare /stop,
  ``/stop <arriving-platform>``, ``/stop all``) returns its
  confirmation through ``_run_control_command`` AFTER
  ``gateway.stop_one`` has already popped the adapter — the
  gateway then answers ``"no adapter for '<platform>'"`` and
  ``_run_control_command`` swallows the non-exception failure
  silently. From the owner's phone the stop is invisible: no
  confirmation, the platform dead without a word, and no /start
  can reach it from that surface (the stop itself is sticky, so
  it stays dead).
- The fix (``nomorals/agents/partner/runtime.py``, the
  ``kind == "stop"`` branch): when the stop will kill the
  platform the command arrived on, the handler sends the
  confirmation via ``_send_control_reply`` BEFORE stopping,
  marks the delivery (``_note_control_delivery``), and returns
  ``""`` — the no-empty law keeps ``""``. A stop aimed at
  another platform keeps the normal return path.

Context homes honour NM_TEST_TMPDIR (small shared /tmp).
"""

from __future__ import annotations

import os
import random
import tempfile
import unittest
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner.failures import clear_failures
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.control import parse_control
from nomorals.social.chat.gateway import ChatGateway

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None


class RecordingAdapter(ChatAdapter):
    """Faithful fake: records text per send."""

    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []

    def run(self, handler) -> None:  # pragma: no cover - unused
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_context() -> tuple[Any, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-",
                                      dir=_TEST_TMPDIR)
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.enabled": "false",
        }
    )
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return context, tmp


class _StopCase(unittest.TestCase):
    PLATFORMS = ("telegram", "whatsapp")

    def setUp(self) -> None:
        clear_failures()
        self.context, self.tmp = _make_context()
        self.adapters = {name: RecordingAdapter(name)
                         for name in self.PLATFORMS}
        self.gateway = ChatGateway(self.adapters, db=self.context.db)
        self.runtime = PartnerRuntime(self.context, gateway=self.gateway)
        self.runtime.brain.presence_rng = random.Random(34)
        self.runtime._typing_enabled = lambda chat: False

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        self.context.close()
        self.tmp.cleanup()

    def _owner_msg(self, platform: str, text: str) -> ChatMessage:
        chat = ChatRef(platform=platform, chat_id="1",
                       kind=ChatKind.DM, title="dm", peer="1")
        msg = ChatMessage(chat=chat, incoming=True, text=text,
                          sender="owner", message_id="m1")
        msg.meta.update({"is_owner": True})
        return msg

    def _delivered(self, platform: str, needle: str) -> bool:
        return any(needle in text
                   for text in self.adapters[platform].sent)


# ── the original regression: the (1, 1) arg-count refusal ──────────

class StopParseTests(unittest.TestCase):
    def test_bare_stop_parses(self) -> None:
        cmd = parse_control("/stop")
        assert cmd is not None
        self.assertEqual(cmd.kind, "stop")

    def test_stop_with_platform_parses(self) -> None:
        cmd = parse_control("/stop telegram")
        assert cmd is not None
        self.assertEqual(cmd.kind, "stop")
        self.assertEqual(cmd.arg, "telegram")

    def test_stop_with_trailing_words_parses(self) -> None:
        # Extra words after the platform are ignored, never a
        # refusal: the platform is one word.
        cmd = parse_control("/stop telegram please hurry")
        assert cmd is not None
        self.assertEqual(cmd.kind, "stop")
        self.assertEqual(cmd.arg, "telegram")


# ── the remaining defect: the confirmation never lands ─────────────

class StopDeliveryTests(_StopCase):
    def test_bare_stop_confirms_before_stopping(self) -> None:
        msg = self._owner_msg("telegram", "/stop")
        reply = self.runtime.handle_control("/stop", "telegram:1",
                                            message=msg)
        # The confirmation went out on the adapter BEFORE the stop
        # (it is already sent, so the dispatch returns "").
        self.assertEqual(reply, "")
        self.assertTrue(
            self._delivered("telegram", "telegram: stopped"),
            f"sent={self.adapters['telegram'].sent!r}")

    def test_stop_arriving_platform_confirms(self) -> None:
        msg = self._owner_msg("telegram", "/stop telegram")
        reply = self.runtime.handle_control(
            "/stop telegram", "telegram:1", message=msg)
        self.assertEqual(reply, "")
        self.assertTrue(
            self._delivered("telegram", "telegram: stopped"),
            f"sent={self.adapters['telegram'].sent!r}")

    def test_stop_with_trailing_words_confirms(self) -> None:
        msg = self._owner_msg("telegram", "/stop telegram please hurry")
        reply = self.runtime.handle_control(
            "/stop telegram please hurry", "telegram:1", message=msg)
        self.assertEqual(reply, "")
        self.assertTrue(
            self._delivered("telegram", "telegram: stopped"),
            f"sent={self.adapters['telegram'].sent!r}")

    def test_stop_all_confirms_on_arriving_platform(self) -> None:
        msg = self._owner_msg("telegram", "/stop all")
        reply = self.runtime.handle_control(
            "/stop all", "telegram:1", message=msg)
        self.assertEqual(reply, "")
        self.assertTrue(
            self._delivered("telegram", "stopped:"),
            f"sent={self.adapters['telegram'].sent!r}")
        self.assertTrue(
            self._delivered("telegram", "telegram"),
            f"sent={self.adapters['telegram'].sent!r}")
        self.assertTrue(
            self._delivered("telegram", "whatsapp"),
            f"sent={self.adapters['telegram'].sent!r}")

    def test_stop_other_platform_keeps_normal_path(self) -> None:
        msg = self._owner_msg("telegram", "/stop whatsapp")
        # The live path: _run_control_command sends the reply
        # itself (handle_control only returns text).
        self.runtime._run_control_command(
            msg, parse_control("/stop whatsapp"))
        self.assertTrue(
            self._delivered("telegram", "whatsapp: stopped"),
            f"sent={self.adapters['telegram'].sent!r}")
        # The other platform is down, the arriving one still works.
        self.assertNotIn("whatsapp",
                         self.gateway._snapshot_adapters())
        self.assertIn("telegram",
                      self.gateway._snapshot_adapters())

    def test_stop_unknown_platform_answers_honestly(self) -> None:
        msg = self._owner_msg("telegram", "/stop nosuch")
        self.runtime._run_control_command(
            msg, parse_control("/stop nosuch"))
        self.assertTrue(
            self._delivered("telegram", "nosuch"),
            f"sent={self.adapters['telegram'].sent!r}")
        self.assertIn("telegram",
                      self.gateway._snapshot_adapters())

    def test_bare_stop_live_path_sends_once(self) -> None:
        # The true live path, end to end: one confirmation, no
        # second send attempt after the stop (the dispatch sees
        # the empty reply and stands down).
        msg = self._owner_msg("telegram", "/stop")
        self.runtime._run_control_command(msg, parse_control("/stop"))
        got = [t for t in self.adapters["telegram"].sent
               if "telegram: stopped" in t]
        self.assertEqual(len(got), 1,
                         f"sent={self.adapters['telegram'].sent!r}")
        self.assertNotIn("telegram",
                         self.gateway._snapshot_adapters())

    def test_stop_without_message_asks_what(self) -> None:
        reply = self.runtime.handle_control("/stop", "telegram:1",
                                            message=None)
        self.assertIn("stop what?", str(reply))


# ── the stopped work actually halts ────────────────────────────────

class StopActuallyHaltsTests(_StopCase):
    def test_stopped_platform_is_really_down(self) -> None:
        msg = self._owner_msg("telegram", "/stop")
        self.runtime.handle_control("/stop", "telegram:1", message=msg)
        self.assertNotIn("telegram",
                         self.gateway._snapshot_adapters())
        chat = ChatRef(platform="telegram", chat_id="1",
                       kind=ChatKind.DM, title="dm", peer="1")
        result = self.gateway.send("telegram", chat, "probe")
        self.assertFalse(result.ok)
        self.assertIn("no adapter", str(result.error))

    def test_stop_all_halts_everything(self) -> None:
        msg = self._owner_msg("telegram", "/stop all")
        self.runtime.handle_control("/stop all", "telegram:1",
                                    message=msg)
        self.assertEqual(self.gateway._snapshot_adapters(), {})

    def test_stop_one_platform_leaves_the_other_running(self) -> None:
        msg = self._owner_msg("telegram", "/stop")
        self.runtime.handle_control("/stop", "telegram:1", message=msg)
        chat = ChatRef(platform="whatsapp", chat_id="1",
                       kind=ChatKind.DM, title="dm", peer="1")
        result = self.gateway.send("whatsapp", chat, "probe")
        self.assertTrue(result.ok)
