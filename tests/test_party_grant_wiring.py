"""Game features and parties (E-6, last item): the /party grant.

The party organ, command, dispatch, store, trivia provider and economy
hook all exist and are pinned by test_community_organs_wiring.py — but
through the REAL handle_control path /party was dead two ways:

1. The grant: "party" is in none of PUBLIC_COMMANDS / GAME_COMMANDS /
   GROUP_DECISION_COMMANDS / GROUP_ADMIN_COMMANDS, so a group member's
   /party never reaches the branch (group gate: "that one's owner-only").
   The help text and the dispatch comment both promise the opposite:
   "Not owner-gated (shared surfaces)" / "Anyone in the group drives it".
   The GOAL.md purpose is "game features and parties to hook users" —
   a party only the owner can drive hooks nobody.
2. The kind: the dispatch branch passes the default chat ref rebuilt
   from the chat key (kind defaults to DM) instead of the message's own
   group ref, so even the OWNER's /party in a group met control_party's
   "Parties live in group chats" refusal. The miniapp branch next door
   already rebinds `chat = message.chat` for exactly this reason.

These tests drive the real handle_control path (pass-55 bare-runtime
shape) with a temp community dir, so nothing touches the real home.
The games half needs no test here: "game" is already in
PUBLIC_COMMANDS, GAME_COMMANDS (55 kinds) already admits every game
kind for non-owners, and the games half is proven live by the owner's
own play plus the navigation audit.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import (
    PUBLIC_COMMANDS,
    caller_may_run,
)
from nomorals.storage.db import Database

OWNER_ID = "5478650254"
GROUP_KEY = "telegram:-100123"


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


class PartyGrantTests(unittest.TestCase):
    def test_party_is_a_public_command(self):
        self.assertIn("party", PUBLIC_COMMANDS)

    def test_party_grant_admits_non_owner(self):
        self.assertTrue(caller_may_run("party", False))

    def test_games_half_already_public(self):
        # Control case: the games half of the item needs no fix.
        self.assertIn("game", PUBLIC_COMMANDS)
        self.assertTrue(caller_may_run("game", False))


class PartyHandleControlTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-party-grant-")
        self.rt = _bare_runtime(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_owner_group_party_creates(self):
        out = self.rt.handle_control(
            "/party new Friday Turnup", GROUP_KEY,
            message=_group_msg(OWNER_ID, "/party new Friday Turnup"))
        self.assertIn("Friday Turnup", out)
        self.assertNotIn("Parties live in group chats", out)
        self.assertNotIn("just for the owner", out)

    def test_member_group_party_creates(self):
        out = self.rt.handle_control(
            "/party new Friday Turnup", GROUP_KEY,
            message=_group_msg("u2", "/party new Friday Turnup"))
        self.assertIn("Friday Turnup", out)
        self.assertNotIn("just for the owner", out)
        self.assertNotIn("Parties live in group chats", out)

    def test_member_can_rsvp_to_owner_party(self):
        created = self.rt.handle_control(
            "/party new Friday Turnup", GROUP_KEY,
            message=_group_msg(OWNER_ID, "/party new Friday Turnup"))
        self.assertIn("Friday Turnup", created)
        out = self.rt.handle_control(
            "/party rsvp yes", GROUP_KEY,
            message=_group_msg("u2", "/party rsvp yes"))
        # EventStore.rsvp returns the choice by design, so the reply
        # is "yes" — the RSVP landing is what matters, not the word.
        self.assertNotIn("just for the owner", out)
        self.assertNotIn("Parties live in group chats", out)
        from nomorals.community.events import EventStore
        from nomorals.community.parties import PartyStore

        party = PartyStore(Path(self._tmp.name) / "parties").load(
            "telegram:-100123")[0]
        event = EventStore(Path(self._tmp.name) / "events").get(
            party.event_id)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.rsvps.get("u2"), "yes")

    def test_party_persists_under_group(self):
        self.rt.handle_control(
            "/party new Friday Turnup", GROUP_KEY,
            message=_group_msg(OWNER_ID, "/party new Friday Turnup"))
        from nomorals.community.parties import PartyStore

        stored = PartyStore(Path(self._tmp.name) / "parties").load(
            "telegram:-100123")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].title, "Friday Turnup")

    def test_party_still_refuses_dm(self):
        out = self.rt.handle_control(
            "/party new Friday Turnup", f"telegram:{OWNER_ID}",
            message=_dm_msg(OWNER_ID, "/party new Friday Turnup"))
        self.assertIn("Parties live in group chats", out)


if __name__ == "__main__":
    unittest.main()
