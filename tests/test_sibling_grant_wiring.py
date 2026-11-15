"""Sibling grants (party follow-up): ajo / cgroup / room / challenge /
match carry the same dead-grant defect /party had (shipped 9acfaefa).

Their dispatch comments and help text promise "not owner-gated (shared
surfaces)" / "Anyone in the group drives it", but all five are in NONE
of PUBLIC_COMMANDS / GAME_COMMANDS / GROUP_DECISION_COMMANDS /
GROUP_ADMIN_COMMANDS, so a group member's command never reaches the
branch ("that one's just for the owner."). The kind defect is narrower
than party's: only ajo (whole command gates on group kind) and cgroup's
schedule/posts subcommands refuse the default DM-kind chat ref — the
match / room / challenge handlers never read chat.kind, so the owner
in a group reaches them fine on base.

These tests drive the real handle_control path (pass-55 bare-runtime
shape) with a temp community dir AND a temp HOME, so the stores whose
defaults live under Path.home() (GroupStore, RoomStore, the matching
and challenge SQLite DBs) never touch the real home.
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import (
    PUBLIC_COMMANDS,
    caller_may_run,
)
from nomorals.storage.db import Database

OWNER_ID = "5478650254"
GROUP_KEY = "telegram:-100123"

SIBLING_KINDS = ("ajo", "cgroup", "room", "challenge", "match")


def _bare_runtime(community_dir: str) -> PartnerRuntime:
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(community_dir=community_dir),
    )
    rt._owner_chats = {OWNER_ID}
    rt._owner_sender_ids = {OWNER_ID}
    return rt


def _group_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=GROUP_KEY, platform="telegram",
                           kind=ChatKind.GROUP, chat_id="-100123")
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id)


def _dm_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=f"telegram:{sender_id}", platform="telegram",
                           kind=ChatKind.DM, chat_id=sender_id)
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id)


class SiblingGrantTests(unittest.TestCase):
    def test_all_five_are_public_commands(self):
        for kind in SIBLING_KINDS:
            with self.subTest(kind=kind):
                self.assertIn(kind, PUBLIC_COMMANDS)

    def test_grant_admits_non_owner(self):
        for kind in SIBLING_KINDS:
            with self.subTest(kind=kind):
                self.assertTrue(caller_may_run(kind, False))

    def test_party_stays_public(self):
        # Control case: the shipped /party fix must not regress.
        self.assertIn("party", PUBLIC_COMMANDS)
        self.assertTrue(caller_may_run("party", False))


class SiblingHandleControlTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-sibling-grant-")
        self._home = tempfile.TemporaryDirectory(prefix="nm-test-sibling-home-")
        self._home_patch = mock.patch.dict(
            os.environ, {"HOME": self._home.name})
        self._home_patch.start()
        self.rt = _bare_runtime(self._tmp.name)

    def tearDown(self):
        self._home_patch.stop()
        self._home.cleanup()
        self._tmp.cleanup()

    def _drive(self, sender_id, text, group=True):
        key = GROUP_KEY if group else f"telegram:{sender_id}"
        msg = _group_msg(sender_id, text) if group else _dm_msg(sender_id, text)
        return self.rt.handle_control(text, key, message=msg)

    def _create_community(self, sender_id=OWNER_ID):
        out = self._drive(sender_id, "/cgroup new SiblingGroup | probe topic")
        self.assertIn("Group created", out, out)
        m = re.search(r"`([^`]+)`", out)
        self.assertIsNotNone(m, out)
        assert m is not None
        return m.group(1)

    def test_member_match_daily(self):
        out = self._drive("u2", "/match daily")
        self.assertNotIn("just for the owner", out)
        self.assertIn("no candidates", out)

    def test_owner_match_daily(self):
        out = self._drive(OWNER_ID, "/match daily")
        self.assertNotIn("just for the owner", out)
        self.assertIn("no candidates", out)

    def test_member_cgroup_new(self):
        out = self._drive("u2", "/cgroup new MemberGroup | probe topic")
        self.assertNotIn("just for the owner", out)
        self.assertIn("Group created", out)

    def test_owner_cgroup_schedule_reaches_group(self):
        # Kind defect: on base the owner in a group got "Scheduled posts
        # belong to a group chat" because the branch passes the default
        # DM-kind chat ref.
        out = self._drive(OWNER_ID, "/cgroup schedule 5 probe post")
        self.assertNotIn("belong to a group chat", out)
        self.assertIn("Scheduled for this group in 5 min", out)

    def test_member_cgroup_schedule(self):
        out = self._drive("u2", "/cgroup schedule 5 probe post")
        self.assertNotIn("just for the owner", out)
        self.assertNotIn("belong to a group chat", out)
        self.assertIn("Scheduled for this group in 5 min", out)

    def test_member_room_create(self):
        gid = self._create_community()
        out = self._drive("u2", f"/room create {gid} | Sibling Room")
        self.assertNotIn("just for the owner", out)
        self.assertIn("room is live", out)

    def test_owner_room_create(self):
        gid = self._create_community()
        out = self._drive(OWNER_ID, f"/room create {gid} | Owner Room")
        self.assertNotIn("just for the owner", out)
        self.assertIn("room is live", out)

    def test_member_ajo_new(self):
        out = self._drive("u2", "/ajo new 1000 SiblingCircle")
        self.assertNotIn("just for the owner", out)
        self.assertNotIn("live in group chats", out)
        self.assertIn("Ajo **SiblingCircle** opened", out)

    def test_owner_ajo_new(self):
        # Kind defect: on base the owner in a group got the group refusal
        # because the branch passes the default DM-kind chat ref.
        out = self._drive(OWNER_ID, "/ajo new 1000 OwnerCircle")
        self.assertNotIn("live in group chats", out)
        self.assertIn("Ajo **OwnerCircle** opened", out)

    def test_ajo_still_refuses_dm(self):
        out = self._drive(OWNER_ID, "/ajo new 1000 DmCircle", group=False)
        self.assertIn("live in group chats", out)

    def test_member_challenge_create(self):
        out = self._drive(
            "u2", "/challenge create SiblingChallenge | workout-count | 7")
        self.assertNotIn("just for the owner", out)
        self.assertIn("challenge created", out)

    def test_owner_challenge_create(self):
        out = self._drive(
            OWNER_ID, "/challenge create OwnerChallenge | workout-count | 7")
        self.assertNotIn("just for the owner", out)
        self.assertIn("challenge created", out)

    def test_ajo_persists_under_group(self):
        self._drive(OWNER_ID, "/ajo new 1000 PersistCircle")
        from nomorals.community.ajo import AjoStore

        stored = AjoStore(Path(self._tmp.name) / "ajo").list(GROUP_KEY)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].name, "PersistCircle")


if __name__ == "__main__":
    unittest.main()
