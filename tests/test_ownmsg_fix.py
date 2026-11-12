"""Own-account WhatsApp intake + money-parser instruction guard.

Owner screenshots 2026-11-12 ~04:17 AM (WhatsApp group "Devon"):
the owner's typed ``/game quit`` earned the pre-fix refusal
("games aren't running in this group") from his own account AND
"no game is live here." from the bot — the exact trap the groupdup
fix (2c8eebe1) was built to close — and the same balance line
posted twice (4:00, 4:01).

Parent trace (no lane): the groupdup quit bypass fires only when
``_control_game`` sees ``is_owner=True``, and ``is_owner`` is True
on the WhatsApp path ONLY when the turn carries ``owner_command``
meta — which the adapter stamps ONLY in the ``own_message``
branch. The bridge also delivers the owner's group messages as
ordinary ``"message"`` events (Baileys ``fromMe``), and that
branch stamps nothing for the paired account: ``_is_operator``
falls through (sender allowlist is Telegram ids; the group chat
is not an owner chat) → ``is_owner=False`` → the §33 gate fires
first → the refusal. The same platform message can also arrive
through BOTH branches, so one typed command becomes two turns
(the duplicate balance line).

The law pinned here:

* A WhatsApp ``"message"`` event whose sender IS the paired
  account (phone JID or LID, either addressing form) is the
  owner's own message: it carries ``from_owner`` and, when the
  text parses as a registered command, ``owner_command`` — the
  same stamps the ``own_message`` carve-out applies. The quit
  bypass (and every owner gate) then fires on EITHER event shape.
* One platform message id gets ONE turn, across BOTH branches:
  the shared ``_seen_inbound`` redelivery set is consulted by
  ``own_message`` as well as ``"message"``.
* A money "recipient" that starts with an instruction verb
  (memorize / remember / save / note / store) is a misparse, not
  a person — the send flow must never ask for that "person's"
  bank details. And a blank recipient is asked WHO, never
  probed for bank details.
"""
from __future__ import annotations

import unittest

from nomorals.finance.send import RecipientStore, resolve_recipient
from nomorals.social.chat.whatsapp import WhatsAppAdapter
from nomorals.voice.money import parse_voice_money

OWNER = "2348012345678@s.whatsapp.net"
LID = "105382746195847@lid"
STRANGER = "2348099999999@s.whatsapp.net"
GROUP = "120363123456789@g.us"


class OwnAccountMessageBranchTests(unittest.TestCase):
    """The owner's group messages arriving as ordinary "message"
    events (the bridge's fromMe shape)."""

    def _adapter(self):
        adapter = WhatsAppAdapter()
        msgs: list = []
        adapter._dispatch(
            {"type": "status", "state": "open", "user": OWNER,
             "lid": LID},
            msgs.append)
        return adapter, msgs

    @staticmethod
    def _msg(mid, sender, text, alt=""):
        event = {"type": "message", "id": mid,
                 "chat": {"id": GROUP, "kind": "group",
                          "title": "Devon"},
                 "from": {"id": sender, "name": "Vrede peace"},
                 "text": text, "ts": 1_700_000_000_000}
        if alt:
            event["from"]["alt"] = alt
        return event

    def test_group_own_command_via_message_branch_is_owner_turn(self):
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._msg("m1", OWNER, "/game quit"), msgs.append)
        self.assertEqual(len(msgs), 1)
        msg = msgs[0]
        self.assertTrue(msg.incoming)
        self.assertEqual(msg.text, "/game quit")
        self.assertEqual(msg.meta.get("from_owner"), True)
        self.assertEqual(msg.meta.get("owner_command"), True)

    def test_group_own_command_via_message_branch_lid_form(self):
        # LID addressing: the bridge reports the LID as from.id
        # and the phone JID as alt (or the reverse) — either form
        # is the paired account.
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._msg("m2", LID, "/game quit", alt=OWNER),
            msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].meta.get("from_owner"), True)
        self.assertEqual(msgs[0].meta.get("owner_command"), True)

    def test_stranger_group_command_via_message_branch_not_owner(self):
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._msg("m3", STRANGER, "/game quit"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].incoming)
        self.assertNotIn("from_owner", msgs[0].meta)
        self.assertNotIn("owner_command", msgs[0].meta)

    def test_group_own_plain_text_via_message_branch_not_promoted(self):
        # Own conversation stays as-him: stamped from_owner, but
        # never an owner_command — commands only.
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._msg("m4", OWNER, "hello everyone"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("from_owner"), True)
        self.assertNotIn("owner_command", msgs[0].meta)

    def test_group_own_unknown_slash_via_message_branch_not_promoted(self):
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._msg("m5", OWNER, "/notacommand"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].meta.get("from_owner"), True)
        self.assertNotIn("owner_command", msgs[0].meta)


class CrossBranchDedupeTests(unittest.TestCase):
    """One platform message id → one turn, whichever branch
    sees it first."""

    def _adapter(self):
        adapter = WhatsAppAdapter()
        msgs: list = []
        adapter._dispatch(
            {"type": "status", "state": "open", "user": OWNER,
             "lid": LID},
            msgs.append)
        return adapter, msgs

    @staticmethod
    def _own(mid, text="/game quit"):
        return {"type": "own_message", "id": mid,
                "chat": {"id": GROUP, "kind": "group",
                         "title": "Devon"},
                "text": text, "ts": 1_700_000_000_000}

    @staticmethod
    def _msg(mid, text="/game quit"):
        return {"type": "message", "id": mid,
                "chat": {"id": GROUP, "kind": "group",
                         "title": "Devon"},
                "from": {"id": OWNER, "name": "Vrede peace"},
                "text": text, "ts": 1_700_000_000_000}

    def test_own_message_then_message_delivers_once(self):
        adapter, msgs = self._adapter()
        adapter._dispatch(self._own("dup1"), msgs.append)
        adapter._dispatch(self._msg("dup1"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("owner_command"), True)

    def test_message_then_own_message_delivers_once(self):
        adapter, msgs = self._adapter()
        adapter._dispatch(self._msg("dup2"), msgs.append)
        adapter._dispatch(self._own("dup2"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("owner_command"), True)

    def test_distinct_ids_still_deliver_twice(self):
        adapter, msgs = self._adapter()
        adapter._dispatch(self._own("d1", text="/game quit"),
                          msgs.append)
        adapter._dispatch(self._msg("d2", text="/game balance"),
                          msgs.append)
        self.assertEqual(len(msgs), 2)


class MoneyInstructionGuardTests(unittest.TestCase):
    """A "recipient" that is really an instruction is a misparse,
    not a person (owner screenshot 2026-11-12 04:01 AM: the bot
    asked for the bank details of "memorize these")."""

    def test_memorize_recipient_not_captured(self):
        intent = parse_voice_money("send 5k for memorize these")
        self.assertIsNone(intent.recipient)

    def test_remember_recipient_not_captured(self):
        intent = parse_voice_money("send 5k to remember these")
        self.assertIsNone(intent.recipient)

    def test_save_recipient_not_captured(self):
        intent = parse_voice_money("send 5k for save this")
        self.assertIsNone(intent.recipient)

    def test_real_recipient_still_captured(self):
        intent = parse_voice_money("send 5k to Mama")
        self.assertEqual(intent.recipient, "Mama")

    def test_blank_recipient_asks_who_not_bank_details(self):
        result = resolve_recipient("", RecipientStore())
        self.assertFalse(result["ok"])
        self.assertEqual(result["needs"], "recipient")
        self.assertNotIn("bank details", result["ask"])
        self.assertIn("who", result["ask"].lower())


if __name__ == "__main__":
    unittest.main()
