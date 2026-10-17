"""Workstream 2 (reachability): new chat commands + help sync.

* The nine new commands are registered (CONTROL_COMMANDS), documented
  (COMMAND_DETAILS), listed (/list via LIST_GROUPS), in help, parseable,
  owner-only (never in PUBLIC_COMMANDS), and dispatch to real handlers.
* /tempmail + /checkmail end-to-end against fake providers (no network):
  generate → show address → list → read → code. Per-user scoping: the
  same owner in another chat cannot read the inbox; a non-owner is
  denied at the grant gate before the handler runs.
* /vnumber + /checksms end-to-end the same way.
* /browser against a fake BrowserSession; /paper against fake trading
  fns; /people, /commitments, /focus against their real stores on a
  throwaway home.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.accounts.temp_mail import MailMessage, TempAddress
from nomorals.accounts.temp_sms import SmsMessage, TempNumber
from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    PUBLIC_COMMANDS,
    list_catalog,
    parse_control,
)

NEW_COMMANDS = [
    "tempmail", "checkmail", "vnumber", "checksms", "browser",
    "paper", "people", "commitments", "focus", "backup",
]
OWNER_ID = "5478650254"


def _make_runtime():
    # Scratch under ~/workspace, not /tmp: /tmp is a small tmpfs here and
    # fills up across suites ("database or disk is full" flakes).
    scratch = Path.home() / "workspace" / ".test-tmp-ws2"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-ws2-", dir=str(scratch))
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": OWNER_ID,
        "partner.owner_sender_ids": OWNER_ID,
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    runtime = PartnerRuntime(context, gateway=None)
    return runtime, tmp


def _owner_msg(chat_key: str, sender_id: str = OWNER_ID) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "1", kind=ChatKind.DM),
        incoming=True, text="", sender="owner", sender_id=sender_id,
        meta={"is_owner": True})


def _stranger_msg(chat_key: str) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "1", kind=ChatKind.DM),
        incoming=True, text="", sender="stranger", sender_id="999",
        meta={})


class FakeMailProvider:
    name = "1secmail"

    def __init__(self) -> None:
        self.read_calls: list[str] = []

    def get_messages(self, addr: TempAddress) -> list[MailMessage]:
        return [MailMessage(id="1", sender="noreply@example.com",
                            subject="Your code is 123456",
                            date="2026-10-17", body_text=""),
                MailMessage(id="2", sender="news@example.com",
                            subject="Weekly digest", date="2026-10-17")]

    def read_message(self, addr: TempAddress, msg: MailMessage) -> MailMessage:
        self.read_calls.append(msg.id)
        if msg.id == "1":
            msg.body_text = "Hello! Your verification code is 123456. Thanks!"
            msg.code = ""
            msg.__post_init__()
        else:
            msg.body_text = "This week's news…"
        return msg

    def delete_message(self, addr: TempAddress, msg: MailMessage) -> bool:
        return False


class FakeSmsProvider:
    name = "simcodes"

    def get_messages(self, number: TempNumber, limit: int = 20) -> list[SmsMessage]:
        return [SmsMessage(sender="12345",
                           body="Your code is 987654. Do not share it.",
                           received="2 minutes ago"),
                SmsMessage(sender="99999", body="Sale starts Friday!",
                           received="1 hour ago")]


class RegistrationTests(unittest.TestCase):
    def test_new_commands_registered_documented_listed(self):
        listed = {c for _, v in LIST_GROUPS for c in v}
        for kind in NEW_COMMANDS:
            self.assertIn(kind, CONTROL_COMMANDS, kind)
            self.assertIn(kind, COMMAND_DETAILS, kind)
            self.assertTrue(COMMAND_DETAILS[kind].get("what"), kind)
            self.assertNotIn(kind, PUBLIC_COMMANDS,
                             f"{kind} must stay owner-only")
            self.assertIn(kind, listed, f"{kind} missing from /list groups")
            self.assertIsNotNone(parse_control(f"/{kind}"),
                                 f"/{kind} does not parse")
            self.assertIn(f"/{kind}", list_catalog())

    def test_every_command_is_listed(self):
        """Help sync: no registered command hides from /list."""
        listed = {c for _, v in LIST_GROUPS for c in v}
        missing = sorted(set(CONTROL_COMMANDS) - listed)
        self.assertEqual(missing, [], f"commands missing from /list: {missing}")

    def test_new_commands_in_help_text(self):
        from nomorals.social.chat.control import help_text

        text = help_text()
        for kind in NEW_COMMANDS:
            self.assertIn(f"/{kind}", text, kind)


class TempMailTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.tmp = _make_runtime()
        self.fake_addr = TempAddress(
            address="abc123@1secmail.com", login="abc123",
            domain="1secmail.com", provider="1secmail", token="")
        self.fake_prov = FakeMailProvider()
        p1 = mock.patch(
            "nomorals.accounts.temp_mail.grab_address_cascade",
            return_value=self.fake_addr)
        p2 = mock.patch(
            "nomorals.accounts.temp_mail.get_provider",
            return_value=self.fake_prov)
        self._p1, self._p2 = p1, p2
        p1.start()
        p2.start()

    def tearDown(self):
        self._p1.stop()
        self._p2.stop()
        self.tmp.cleanup()

    def test_end_to_end(self):
        rt = self.runtime
        msg = _owner_msg("telegram:1")
        reply = rt.handle_control("/tempmail new", "telegram:1", message=msg)
        self.assertIn("abc123@1secmail.com", reply)
        reply = rt.handle_control("/tempmail", "telegram:1", message=msg)
        self.assertIn("abc123@1secmail.com", reply)
        reply = rt.handle_control("/checkmail", "telegram:1", message=msg)
        self.assertIn("Your code is 123456", reply)
        self.assertIn("/checkmail read", reply)
        reply = rt.handle_control("/checkmail read 1", "telegram:1", message=msg)
        self.assertIn("Your verification code is 123456", reply)
        self.assertIn("code: 123456", reply)
        reply = rt.handle_control("/checkmail code", "telegram:1", message=msg)
        self.assertIn("code: 123456", reply)
        reply = rt.handle_control("/tempmail forget", "telegram:1", message=msg)
        self.assertIn("forgotten", reply)
        reply = rt.handle_control("/checkmail", "telegram:1", message=msg)
        self.assertIn("/tempmail new", reply)

    def test_per_user_scoping(self):
        rt = self.runtime
        rt.handle_control("/tempmail new", "telegram:1",
                          message=_owner_msg("telegram:1"))
        # Same owner, different chat: the inbox is not visible there.
        reply = rt.handle_control("/checkmail", "telegram:2",
                                  message=_owner_msg("telegram:2"))
        self.assertIn("/tempmail new", reply)
        self.assertNotIn("abc123@1secmail.com", reply)
        # A non-owner is denied at the grant gate, before any handler runs.
        reply = rt.handle_control("/checkmail", "telegram:1",
                                  message=_stranger_msg("telegram:1"))
        self.assertIn("just for the owner", reply)

    def test_provider_failure_is_honest(self):
        with mock.patch(
                "nomorals.accounts.temp_mail.grab_address_cascade",
                side_effect=RuntimeError("all temp-mail providers failed")):
            reply = self.runtime.handle_control(
                "/tempmail new", "telegram:1", message=_owner_msg("telegram:1"))
        self.assertIn("couldn't get a temp address", reply)


class TempSmsTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.tmp = _make_runtime()
        self.info = {"status": "ok", "number": "+15551234567",
                     "masked": "+1555123****", "country": "us",
                     "country_name": "United States",
                     "provider": "simcodes", "inbox_id": "42"}
        p1 = mock.patch(
            "nomorals.accounts.temp_sms.grab_number_cascade",
            return_value=self.info)
        p2 = mock.patch(
            "nomorals.accounts.temp_sms.get_provider",
            return_value=FakeSmsProvider())
        self._p1, self._p2 = p1, p2
        p1.start()
        p2.start()

    def tearDown(self):
        self._p1.stop()
        self._p2.stop()
        self.tmp.cleanup()

    def test_end_to_end(self):
        rt = self.runtime
        msg = _owner_msg("telegram:1")
        reply = rt.handle_control("/vnumber new us", "telegram:1", message=msg)
        self.assertIn("+15551234567", reply)
        reply = rt.handle_control("/vnumber", "telegram:1", message=msg)
        self.assertIn("+15551234567", reply)
        reply = rt.handle_control("/checksms", "telegram:1", message=msg)
        self.assertIn("Your code is 987654", reply)
        self.assertIn("[code 987654]", reply)
        reply = rt.handle_control("/checksms code", "telegram:1", message=msg)
        self.assertIn("code: 987654", reply)

    def test_per_user_scoping(self):
        rt = self.runtime
        rt.handle_control("/vnumber new us", "telegram:1",
                          message=_owner_msg("telegram:1"))
        reply = rt.handle_control("/checksms", "telegram:9",
                                  message=_owner_msg("telegram:9"))
        self.assertIn("/vnumber new", reply)
        self.assertNotIn("987654", reply)


class FakeBrowserSession:
    def open(self, url: str = "", **_: Any) -> dict[str, Any]:
        return {"ok": True, "url": url, "status": 200, "title": "Example",
                "chars": 42, "links": 2, "forms": 1, "cookies": 0}

    def text(self, max_chars: int = 40000, **_: Any) -> dict[str, Any]:
        return {"url": "https://example.com", "title": "Example",
                "chars": 20, "text": "Hello from the example page.",
                "truncated": False}

    def links(self, max_links: int = 100, **_: Any) -> dict[str, Any]:
        return {"url": "https://example.com", "count": 1, "links": [
            {"text": "More", "url": "https://example.com/more"}]}

    def extract(self, target: str = "", kind: str = "", **_: Any) -> dict[str, Any]:
        if kind == "forms":
            return {"url": "https://example.com", "kind": "forms", "count": 1,
                    "forms": [{"id": "", "action": "/go", "method": "GET",
                               "fields": [{"name": "q", "type": "text"}]}]}
        if kind == "headings":
            return {"url": "https://example.com", "kind": "headings",
                    "count": 1, "items": [{"level": 1, "text": "Example"}]}
        return {}


class BrowserPaperTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.tmp = _make_runtime()

    def tearDown(self):
        self.tmp.cleanup()

    def test_browser_open_links_forms_headings(self):
        with mock.patch("nomorals.tools.browser.BrowserSession",
                        FakeBrowserSession):
            rt = self.runtime
            reply = rt.handle_control(
                "/browser open https://example.com", "telegram:1",
                message=_owner_msg("telegram:1"))
            self.assertIn("Example", reply)
            self.assertIn("Hello from the example page", reply)
            reply = rt.handle_control(
                "/browser links https://example.com", "telegram:1",
                message=_owner_msg("telegram:1"))
            self.assertIn("https://example.com/more", reply)
            reply = rt.handle_control(
                "/browser forms https://example.com", "telegram:1",
                message=_owner_msg("telegram:1"))
            self.assertIn("q(text)", reply)
            reply = rt.handle_control(
                "/browser headings https://example.com", "telegram:1",
                message=_owner_msg("telegram:1"))
            self.assertIn("# Example", reply)
            reply = rt.handle_control("/browser open example.com",
                                      "telegram:1",
                                      message=_owner_msg("telegram:1"))
            self.assertIn("http://", reply)

    def test_paper_start_status_stop(self):
        calls: list[tuple[str, tuple, dict]] = []

        def _start(ctx: Any, symbol: str, market: str = "crypto",
                  capital: float = 10000.0) -> dict[str, Any]:
            calls.append(("start", (symbol, market, capital), {}))
            return {"ok": True, "session_id": "paper-1", "resumed": False,
                    "symbol": symbol, "market": market, "capital": capital}

        def _status(ctx: Any, session_id: str = "", symbol: str = "") -> dict[str, Any]:
            return {"ok": True, "session_id": "paper-1", "symbol": "BTC",
                    "market": "crypto", "status": "open", "cash": 9000.0,
                    "units": 0.5, "equity": 11000.0, "pnl": 1000.0,
                    "pnl_pct": 0.1, "orders": 2,
                    "last_action": {"regime": "trend", "bias": "long"}}

        def _stop(ctx: Any, session_id: str = "") -> dict[str, Any]:
            return {"ok": True, "session_id": "paper-1", "symbol": "BTC"}

        with mock.patch("nomorals.tools.trading.paper_start", _start), \
                mock.patch("nomorals.tools.trading.paper_status", _status), \
                mock.patch("nomorals.tools.trading.paper_stop", _stop):
            rt = self.runtime
            msg = _owner_msg("telegram:1")
            reply = rt.handle_control("/paper start BTC crypto 10000",
                                      "telegram:1", message=msg)
            self.assertIn("paper-1", reply)
            self.assertIn("10,000.00", reply)
            reply = rt.handle_control("/paper status", "telegram:1", message=msg)
            self.assertIn("11,000.00", reply)
            self.assertIn("PnL +1,000.00", reply)
            reply = rt.handle_control("/paper stop", "telegram:1", message=msg)
            self.assertIn("closed", reply)
            reply = rt.handle_control("/paper", "telegram:1", message=msg)
            self.assertIn("usage:", reply)


class PeopleCommitmentsFocusTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.tmp = _make_runtime()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        return self.runtime.handle_control(
            text, "telegram:1", message=_owner_msg("telegram:1"))

    def test_people_round_trip(self):
        self.assertIn("no people recorded", self._run("/people list"))
        self.assertIn("Ada", self._run("/people add Ada Lovelace"))
        self.assertIn("Ada Lovelace", self._run("/people list"))
        self.assertIn("loves numbers", self._run(
            "/people note Ada Lovelace loves numbers"))
        shown = self._run("/people show Ada")
        self.assertIn("Ada Lovelace", shown)
        self.assertIn("loves numbers", shown)
        self.assertIn("no one matching", self._run("/people show Zed"))

    def test_commitments_round_trip(self):
        self.assertIn("nothing open", self._run("/commitments list"))
        added = self._run("/commitments add call Ada back by Friday")
        self.assertIn("recorded", added)
        self.assertIn("call Ada back", self._run("/commitments list"))
        # extract the id from the list output
        listed = self._run("/commitments list")
        cid = listed.split("  ", 1)[1].split(":", 1)[0]
        self.assertIn("marked kept", self._run(f"/commitments kept {cid} called her"))
        self.assertIn("nothing open", self._run("/commitments list"))
        rel = self._run("/commitments reliability")
        self.assertIn("1 kept / 0 broken", rel)
        self.assertIn("snapshot matches replay: True",
                      self._run("/commitments verify"))
        self.assertIn("no commitment matching",
                      self._run("/commitments kept cm-nope"))

    def test_focus_round_trip(self):
        self.assertIn("no focus session running", self._run("/focus status"))
        self.assertIn("focus started", self._run("/focus start deep work"))
        self.assertIn("already running",
                      self._run("/focus start something else"))
        self.assertIn("paused", self._run("/focus pause"))
        self.assertIn("resumed", self._run("/focus resume"))
        stopped = self._run("/focus stop done for now")
        self.assertIn("stopped", stopped)
        self.assertIn("focused", stopped)
        today = self._run("/focus today")
        self.assertIn("deep work", today)
        week = self._run("/focus week")
        self.assertIn("Weekly", week)
        self.assertIn("no focus session is running",
                      self._run("/focus pause"))


class TempMailProbeTests(unittest.TestCase):
    def test_onesecmail_probe(self):
        from nomorals.accounts.temp_mail import OneSecMailProvider

        prov = OneSecMailProvider()
        with mock.patch.object(prov, "_get", return_value=["1secmail.com"]):
            self.assertTrue(prov.probe())
        with mock.patch.object(prov, "_get", side_effect=RuntimeError("down")):
            self.assertFalse(prov.probe())

    def test_guerrilla_probe(self):
        from nomorals.accounts.temp_mail import GuerrillaMailProvider

        prov = GuerrillaMailProvider()
        with mock.patch.object(
                prov, "_call",
                return_value={"email_addr": "x@guerrillamail.com",
                              "sid_token": "t"}):
            self.assertTrue(prov.probe())
        with mock.patch.object(prov, "_call", side_effect=RuntimeError("down")):
            self.assertFalse(prov.probe())

    def test_cascade_skips_down_providers(self):
        from nomorals.accounts import temp_mail as _tm

        fake_addr = TempAddress(address="a@x.io", login="a", domain="x.io",
                                provider="mailtm", token="jwt")
        with mock.patch.object(_tm.MailTmProvider, "probe", return_value=False), \
                mock.patch.object(_tm.OneSecMailProvider, "probe",
                                  return_value=False), \
                mock.patch.object(_tm.GuerrillaMailProvider, "probe",
                                  return_value=True), \
                mock.patch.object(_tm.GuerrillaMailProvider, "grab_address",
                                  return_value=fake_addr):
            got = _tm.grab_address_cascade()
        self.assertEqual(got.address, "a@x.io")


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.tmp = _make_runtime()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        return self.runtime.handle_control(
            text, "telegram:1", message=_owner_msg("telegram:1"))

    def test_status_empty_then_with_generation(self):
        self.assertIn("0 generation(s)", self._run("/backup status"))
        from nomorals.backup.manifest import build_manifest, write_manifest

        backup_dir = self.runtime.context.settings.backup_dir
        gen = backup_dir / "gen-1"
        manifest = build_manifest(backup_id="gen-1", protected=[], plan=None)
        write_manifest(gen, manifest)
        status = self._run("/backup status")
        self.assertIn("1 generation(s)", status)
        self.assertIn("gen-1", status)
        self.assertIn("integrity=ok", status)

    def test_verify(self):
        reply = self._run("/backup verify")
        self.assertIn("integrity=ok", reply)
        self.assertIn("no backup generations", reply)
        self.assertIn("usage:", self._run("/backup frobnicate"))


if __name__ == "__main__":
    unittest.main()
