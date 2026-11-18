"""WhatsApp-native /lookup (lane 77, real build B40).

The menu half of the brief is already built at this base (the
buttons completion + the 2026-11-11 repair: /menu, /start and
/help on WhatsApp open the live menu session with buttons —
pinned in tests/test_whatsapp_group_help_menus.py). What is
NOT built is the dossier's live side on WhatsApp: /lookup's
resolution fallback, live identity refresh and card framing
all run through the TELEGRAM userbot (member_by_username,
user_full, getCommonChats) even when the command arrived in
a WhatsApp group — so a WhatsApp account resolves to a
Telegram id, the card says "Telegram ID / Telegram Premium /
account-age estimate / Shared with the userbot account", and
a WhatsApp member who is only on a live roster (not the
ledger) can be missed entirely or mis-resolved.

Pins (failing-first on base 421ad2c9):
- a WhatsApp roster member (not in the ledger) resolves from
  the live roster and the dossier is WhatsApp-framed: phone
  as the identity, display name, group role — no "Telegram"
  anywhere, no userbot board, no account-age estimate;
- a member in ANOTHER served group is found by the bounded
  cross-group roster scan, and the card names the group;
- an unknown account gets an honest not-found naming the
  roster scan — never the userbot;
- a ledger member's dossier is WhatsApp-framed too, with the
  economy/moderation sections intact;
- the Telegram adapter is NEVER called for a WhatsApp lookup;
- Telegram /lookup is unchanged (the userbot fallback still
  serves it);
- the ledger scan matches a bare phone number to a JID key.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

WA_KEY = "whatsapp:120363@g.us"
WA_KEY_B = "whatsapp:120363@g2.us"
WA_KEY_C = "whatsapp:120363@g3.us"
TG_KEY = "telegram-bot:-100123"
NOW = 1_790_000_000.0

JID_ADA = "2348012345678@s.whatsapp.net"
PHONE_ADA = "2348012345678"
JID_BO = "2348098765432@s.whatsapp.net"
PHONE_BO = "2348098765432"
JID_CID = "2348055555555@s.whatsapp.net"
PHONE_CID = "2348055555555"


class _FakeWhatsApp:
    name = "whatsapp"

    def __init__(self, rosters, groups):
        self.rosters = rosters
        self.groups = groups
        self.participant_calls: list = []
        self.list_calls = 0

    def group_participants(self, chat, timeout=20):
        cid = getattr(chat, "chat_id", None) or str(chat)
        self.participant_calls.append((cid, timeout))
        return list(self.rosters.get(cid, []))

    def group_list(self):
        self.list_calls += 1
        return list(self.groups)

    def admin_member(self, chat, user_id):
        return {"status": "administrator"}


class _FakeTelegram:
    name = "telegram"

    def __init__(self):
        self.calls: list = []

    def member_by_username(self, chat, username):
        self.calls.append(("member_by_username", username))
        return {"id": "7", "username": username, "name": "Ada"}

    def user_full(self, user_id):
        self.calls.append(("user_full", user_id))
        return {"id": user_id, "username": "ada", "name": "Ada",
                "phone": "+2348012345678", "bot": False,
                "premium": True}

    def common_chats(self, user_id):
        self.calls.append(("common_chats", user_id))
        return [{"id": "-1001", "title": "Builders"}]


def _groups(rosters):
    out = []
    for cid in sorted(rosters):
        out.append({"id": cid, "subject": f"Group {cid[-4:]}",
                    "size": len(rosters[cid]),
                    "is_community": False, "linked_parent": ""})
    return out


class _WaLookupTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.storage.db import Database

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-wadeep-")
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = SimpleNamespace(
            db=Database(":memory:"),
            settings=SimpleNamespace(
                community_dir=self._tmp.name),
        )
        rt._owner_chats = {"5478650254"}
        rt._owner_sender_ids = {"5478650254"}
        self.rt = rt
        self.wa = _FakeWhatsApp(
            rosters={
                "120363@g.us": [
                    {"jid": JID_ADA, "name": "Ada",
                     "role": "admin"},
                    {"jid": "2348000000001@s.whatsapp.net",
                     "name": "Owner", "role": "superadmin"},
                ],
                "120363@g2.us": [
                    {"jid": JID_BO, "name": "Bo", "role": ""},
                ],
                "120363@g3.us": [
                    {"jid": JID_CID, "name": "Cid", "role": ""},
                ],
            },
            groups=[],
        )
        self.wa.groups = _groups(self.wa.rosters)
        self.tg = _FakeTelegram()
        rt.gateway = SimpleNamespace(
            adapters={"whatsapp": self.wa, "telegram": self.tg})
        surface = GroupSurface(
            GroupStore(Path(self._tmp.name) / "g.json").load())
        surface.store.save()
        self.surface = surface
        self.rt._group_surface_obj = surface

    def tearDown(self):
        self._tmp.cleanup()

    def _msg(self, text, sender_id="5478650254", chat_id="120363@g.us"):
        chat = SimpleNamespace(key=f"whatsapp:{chat_id}",
                               platform="whatsapp", kind=ChatKind.GROUP,
                               chat_id=chat_id)
        return SimpleNamespace(chat=chat, text=text, meta={},
                               sender="Owner", sender_id=sender_id)

    def _lookup(self, tail, **kw):
        out = self.rt.handle_control(
            f"/lookup {tail}", "whatsapp:120363@g.us",
            self._msg(f"/lookup {tail}", **kw))
        return getattr(out, "text", out)

    def test_roster_member_gets_whatsapp_card(self):
        out = self._lookup(PHONE_ADA)
        assert PHONE_ADA in out
        assert "Ada" in out
        assert "admin" in out.lower()
        assert "Telegram" not in out
        assert "userbot" not in out.lower()
        assert "estimate" not in out.lower()
        assert "Shared with the userbot" not in out

    def test_cross_group_scan_finds_member(self):
        out = self._lookup(PHONE_BO)
        assert PHONE_BO in out
        assert "Bo" in out
        assert "Telegram" not in out
        # the card names the group they were found in
        assert "g2.us" in out or "Group g2.us" in out

    def test_unknown_account_is_honest_no_userbot(self):
        out = self._lookup("2348000000999")
        assert "no record" in out.lower()
        assert "userbot" not in out.lower()
        assert "roster" in out.lower()

    def test_ledger_member_whatsapp_framing(self):
        self.surface.record_join(WA_KEY, JID_ADA, username="",
                                 name="Ada", ts=NOW)
        self.surface.economy.award(WA_KEY, JID_ADA, 250, "seed",
                                   now=NOW)
        self.surface.store.save()
        out = self._lookup(PHONE_ADA)
        assert PHONE_ADA in out
        assert "Ada" in out
        assert "Telegram" not in out
        assert "userbot" not in out.lower()
        assert str(self.surface.economy.balance(
            WA_KEY, JID_ADA)) in out

    def test_whatsapp_lookup_never_calls_telegram_adapter(self):
        self._lookup("2348000000999")
        assert self.tg.calls == []

    def test_scan_is_bounded(self):
        # 8 served groups, member only in the 8th: the scan
        # stops at its cap and says so honestly — exactly 5
        # other-group roster reads, and the not-found line
        # names the 5 groups checked.
        rosters = {f"120363@g{i}.us": [] for i in range(1, 9)}
        rosters["120363@g8.us"] = [
            {"jid": JID_CID, "name": "Cid", "role": ""}]
        self.wa.rosters = rosters
        self.wa.groups = _groups(rosters)
        out = self._lookup(PHONE_CID)
        assert "no record" in out.lower()
        assert "5 WhatsApp group" in out
        others = [c for c in self.wa.participant_calls
                  if c[0] != "120363@g.us"]
        assert len(others) == 5

    def test_ledger_scan_matches_phone_digits_to_jid(self):
        self.surface.record_join(WA_KEY, JID_ADA, username="",
                                 name="Ada", ts=NOW)
        target, _groups = self.rt._lookup_scan_ledger(
            self.surface, PHONE_ADA)
        assert target == JID_ADA


class _TgLookupUnchangedTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.storage.db import Database

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-wadeep-tg-")
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = SimpleNamespace(
            db=Database(":memory:"),
            settings=SimpleNamespace(
                community_dir=self._tmp.name),
        )
        rt._owner_chats = {"5478650254"}
        rt._owner_sender_ids = {"5478650254"}
        self.rt = rt
        self.tg = _FakeTelegram()
        self.wa = _FakeWhatsApp(rosters={}, groups=[])
        rt.gateway = SimpleNamespace(
            adapters={"whatsapp": self.wa, "telegram": self.tg})
        surface = GroupSurface(
            GroupStore(Path(self._tmp.name) / "g.json").load())
        surface.store.save()
        self.surface = surface
        self.rt._group_surface_obj = surface

    def tearDown(self):
        self._tmp.cleanup()

    def test_telegram_lookup_still_uses_userbot(self):
        # The userbot surface ("telegram", not the gated
        # "telegram-bot" — on the gated bot nobody is an
        # operator, so the dossier would fall to the public
        # card and the live read would never be exercised).
        chat = SimpleNamespace(key="telegram:-100123",
                               platform="telegram",
                               kind=ChatKind.GROUP, chat_id="-100123")
        msg = SimpleNamespace(chat=chat, text="/lookup @ada",
                              meta={}, sender="Owner",
                              sender_id="5478650254")
        out = self.rt.handle_control("/lookup @ada",
                                     "telegram:-100123", msg)
        text = getattr(out, "text", out)
        assert "7" in text
        assert "ada" in text
        assert "Telegram" in text
        assert "estimate" in text.lower()
        assert self.tg.calls  # the userbot fallback still serves it
