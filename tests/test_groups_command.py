"""The /groups dashboard (DESIGN §8): every group/channel Devon serves —
status, member count, economy health, pending actions — one dashboard
in DM, deep-links into /console per group.

§8 specifies "/groups in DM: every group/channel she serves — status,
member count, economy health, pending actions (captcha holds, appeals,
treasury approvals) — one dashboard, deep-links into each." No such
command exists on base: "groups" is in none of CONTROL_COMMANDS /
PUBLIC_COMMANDS, and there is no _control_groups dispatch. /console home
covers only status + member count + claim (channels: queue + posted);
it shows no economy health and no pending actions. These tests drive
the real handle_control path (pass-55 bare-runtime shape) with a seeded
GroupSurface attached to the runtime, so nothing touches the real home.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    caller_may_run,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.db import Database

OWNER_ID = "5478650254"
GROUP_A = "telegram-bot:-100111"
GROUP_B = "telegram-bot:-100222"
CHANNEL = "telegram-bot:-100333"
NOW = time.time()


def _bare_runtime(community_dir: str) -> PartnerRuntime:
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(community_dir=community_dir),
    )
    rt._owner_chats = {OWNER_ID}
    rt._owner_sender_ids = {OWNER_ID}
    return rt


def _dm_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=f"telegram:{sender_id}", platform="telegram",
                           kind=ChatKind.DM, chat_id=sender_id)
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id)


def _group_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=GROUP_A, platform="telegram",
                           kind=ChatKind.GROUP, chat_id="-100111")
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id)


def _text(out) -> str:
    return getattr(out, "text", out)


def _buttons(out) -> list:
    raw = getattr(out, "buttons", None) or []
    flat: list = []
    for row in raw:
        if isinstance(row, (list, tuple)):
            flat.extend(row)
        else:
            flat.append(row)
    return [str(item[1]) for item in flat
            if isinstance(item, (list, tuple)) and len(item) > 1]


def _seed_surface(tmp_path: Path) -> GroupSurface:
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    # Group A: four members, a live economy, one of everything waiting.
    for uid, name in (("u1", "Ada"), ("u2", "Bo"), ("u3", "Cy"),
                      ("u11", "Dee")):
        surface.record_join(GROUP_A, uid, username=name.lower(),
                            name=name, ts=NOW)
    surface.economy.award(GROUP_A, "u1", 100, "seed", now=NOW)
    surface.economy.treasury_bonus(GROUP_A, 50, "seed", now=NOW)
    surface.propose_spend(GROUP_A, kind="grant",
                          payload={"recipient": "u2", "amount": 600},
                          by="admin1", now=NOW)
    surface.stage_flag(GROUP_A, user_id="u9",
                       person={"id": "u9", "username": "bad"},
                       category="spam", note="seed", by="admin1", now=NOW)
    entry = surface.store.group(GROUP_A)
    entry.setdefault("join_requests", {})["req-1"] = {
        "id": "req-1", "state": "pending", "ts": NOW, "user_id": "u10"}
    entry["members"]["u11"]["acceptance"] = {
        "state": "restricted", "missing": ["username"],
        "reasons": ["username required"], "via": "join",
        "action": "restricted", "since": NOW, "grace_until": None,
        "welcome_pending": True}
    ban = surface.bans.ban("u12", reason="seed", group_id=GROUP_A,
                           banned_by="admin1")
    surface.bans.file_appeal("u12", ban.id, "sorry")
    surface.store.save()
    # Group B: one member, nothing waiting, economy untouched.
    surface.record_join(GROUP_B, "u1", username="ada", name="Ada", ts=NOW)
    surface.store.save()
    # Channel: claimed-channel marker + title, nothing queued/posted.
    centry = surface.store.group(CHANNEL)
    centry.setdefault("extras", {})["channel"] = {"title": "Signals"}
    surface.store.save()
    return surface


class GroupsRegistrationTests(unittest.TestCase):
    def test_groups_is_a_public_command(self):
        self.assertIn("groups", CONTROL_COMMANDS)
        self.assertIn("groups", PUBLIC_COMMANDS)

    def test_groups_grant_admits_non_owner(self):
        self.assertTrue(caller_may_run("groups", False))


class GroupsHandleControlTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-groups-")
        self.rt = _bare_runtime(self._tmp.name)
        self.surface = _seed_surface(Path(self._tmp.name))
        self.rt._group_surface_obj = self.surface

    def tearDown(self):
        self._tmp.cleanup()

    def _groups(self, sender_id: str):
        out = self.rt.handle_control(
            "/groups", f"telegram:{sender_id}",
            message=_dm_msg(sender_id, "/groups"))
        return _text(out), out

    def test_owner_dm_dashboard_shows_groups_channels_and_pending(self):
        text, _out = self._groups(OWNER_ID)
        health = self.surface.economy_health(GROUP_A)
        # Both groups, by key, with their ledger counts.
        self.assertIn(GROUP_A, text)
        self.assertIn("4 in the ledger", text)
        self.assertIn(GROUP_B, text)
        self.assertIn("1 in the ledger", text)
        # Economy health is the surface's own numbers, not a copy.
        self.assertIn(f"treasury {health['treasury']}", text)
        self.assertIn(f"holders {health['holders']}", text)
        self.assertIn(f"active 7d {health['active_7d']}", text)
        # One of each pending action, named.
        self.assertIn("join 1", text)
        self.assertIn("restricted 1", text)
        self.assertIn("flags 1", text)
        self.assertIn("proposals 1", text)
        self.assertIn("appeals 1", text)
        # The quiet group says so instead of faking work.
        self.assertIn("nothing waiting", text)
        # Channel row: title, queue, posted.
        self.assertIn("Signals", text)
        self.assertIn("0 queued", text)
        self.assertIn("0 posted", text)

    def test_dashboard_buttons_deep_link_into_console(self):
        _text_out, out = self._groups(OWNER_ID)
        cbs = _buttons(out)
        self.assertTrue(any(cb.startswith("console g") for cb in cbs),
                        f"no console deep-link in {cbs!r}")
        self.assertTrue(any(cb.startswith("console c") for cb in cbs),
                        f"no channel console link in {cbs!r}")

    def test_non_admin_dm_sees_no_groups(self):
        text, _out = self._groups("u99")
        self.assertIn("don't admin any group", text)
        self.assertNotIn(GROUP_A, text)

    def test_groups_refuses_group_chat(self):
        out = self.rt.handle_control(
            "/groups", GROUP_A, message=_group_msg(OWNER_ID, "/groups"))
        self.assertIn("my DM", _text(out))

    def test_owner_dm_no_groups_empty_state(self):
        fresh = GroupSurface(GroupStore(
            Path(self._tmp.name) / "empty.json").load())
        self.rt._group_surface_obj = fresh
        text, _out = self._groups(OWNER_ID)
        self.assertIn("don't admin any group", text)


if __name__ == "__main__":
    unittest.main()
