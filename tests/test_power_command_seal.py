"""Parser + runtime must accept the 3-arg seal form of /power.

Regression: ``/power on <identity> <passphrase>`` was rejected by the
parser ("/power takes at most 2 argument(s)") because
``CONTROL_COMMANDS["power"]`` was ``(1, 2)``, even though the runtime
dispatch supports the 3-arg seal form (``power.unlock(passphrase,
identity=identity)``). The owner followed the documented 3-part form
and got blocked by the parser.
"""

from __future__ import annotations

import tempfile
import unittest
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.core.owner import make_seal
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    help_text,
    parse_control,
)
from nomorals.social.chat.gateway import ChatGateway

OWNER_ID = "5478650254"
IDENTITY = "peace"  # one of OWNER_IDENTITIES in nomorals.core.owner
PASSPHRASE = "test-seal-passphrase-12345"  # >= MIN_PASSPHRASE_LEN (20)


class TestParsePowerSealForm(unittest.TestCase):
    def test_max_args_allows_three(self):
        self.assertEqual(CONTROL_COMMANDS["power"], (1, 3))

    def test_parse_power_seal_form(self):
        cmd = parse_control(f"/power on {IDENTITY} {PASSPHRASE}")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "power")
        self.assertEqual(cmd.tail, f"on {IDENTITY} {PASSPHRASE}")

    def test_parse_power_key_form_still_works(self):
        cmd = parse_control("/power on mykey")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "power")
        self.assertEqual(cmd.tail, "on mykey")

    def test_parse_power_off_status(self):
        for text in ("/power off", "/power status"):
            cmd = parse_control(text)
            self.assertIsNotNone(cmd)
            assert cmd is not None
            self.assertEqual(cmd.kind, "power")

    def test_parse_power_on_alone_passes_to_runtime(self):
        # "/power on" has 1 arg (min 1) — the parser passes it; the
        # runtime returns the usage message (len(parts) < 2 branch).
        cmd = parse_control("/power on")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "power")
        self.assertEqual(cmd.tail, "on")

    def test_parse_power_bare_errors(self):
        cmd = parse_control("/power")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "error")
        self.assertIn("needs 1 argument(s)", cmd.arg)

    def test_parse_power_too_many_args(self):
        cmd = parse_control("/power on a b c")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "error")
        self.assertIn("takes at most 3 argument(s)", cmd.arg)

    def test_help_mentions_seal_form(self):
        self.assertIn("<passphrase>", COMMAND_DETAILS["power"]["usage"])
        self.assertIn("passphrase", COMMAND_DETAILS["power"]["what"])
        self.assertIn(
            "/power on <identity> <passphrase>", help_text()
        )


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self._handler = None

    def run(self, handler) -> None:
        self._handler = handler

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, **kwargs: Any) -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_runtime(platform: str):
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-power-seal-")
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": platform,
            "partner.owner_chats": OWNER_ID,
            "partner.owner_sender_ids": OWNER_ID,
        }
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = FakeAdapter(platform)
    gateway = ChatGateway({platform: adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, context, tmp


def _owner_msg(platform: str, text: str) -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=OWNER_ID,
                     kind=ChatKind.DM, peer="owner"),
        incoming=True,
        text=text,
        sender="owner",
        sender_id=OWNER_ID,
        meta={},
    )


class TestRuntimePowerSealForm(unittest.TestCase):
    def test_seal_form_unlocks_power(self):
        seal = make_seal(PASSPHRASE)
        with mock.patch("nomorals.core.owner.PASSPHRASE_SEAL", seal):
            runtime, adapter, _ctx, tmp = _make_runtime("telegram-personal")
            try:
                runtime._process(_owner_msg(
                    "telegram-personal",
                    f"/power on {IDENTITY} {PASSPHRASE}",
                ))
            finally:
                tmp.cleanup()
        self.assertTrue(adapter.sent, "expected a reply")
        reply = adapter.sent[-1]
        # The seal passphrase was verified (a wrong one yields "wrong
        # key", see the test below). power_mode_for auto-activates on
        # first consultation (default-on posture), so the success reply
        # here is "already active" — the key check still ran first.
        self.assertIn("active", reply)
        self.assertNotIn("wrong key", reply)

    def test_seal_form_wrong_passphrase_stays_locked(self):
        seal = make_seal(PASSPHRASE)
        with mock.patch("nomorals.core.owner.PASSPHRASE_SEAL", seal):
            runtime, adapter, _ctx, tmp = _make_runtime("telegram-personal")
            try:
                runtime._process(_owner_msg(
                    "telegram-personal",
                    f"/power on {IDENTITY} wrong-passphrase-00000",
                ))
            finally:
                tmp.cleanup()
        self.assertTrue(adapter.sent, "expected a reply")
        self.assertIn("wrong key", adapter.sent[-1])


if __name__ == "__main__":
    unittest.main()
