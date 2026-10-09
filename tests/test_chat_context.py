"""Tests for per-chat context profiles, double-reply discipline, and
message-level owner recognition.

The point: she knows every chat's reality (who talks, what's discussed),
never answers a message the owner already handled, and recognizes the
master in groups.
"""

from __future__ import annotations

import sqlite3
import time
import unittest

from nomorals.partner.chat_profile import (
    build_context_lines,
    get_profile,
    refresh_profile,
)


def _db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT, "
        "agent TEXT, channel TEXT, created_at REAL, updated_at REAL)"
    )
    db.execute(
        "CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT, "
        "role TEXT, content TEXT, name TEXT, model TEXT, created_at REAL)"
    )
    return db


class _FakeDB:
    """Minimal wrapper exposing the query/query_one/execute/scalar surface."""

    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def query(self, sql: str, params: tuple = ()):
        return [dict(r) for r in self._c.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: tuple = ()):
        r = self._c.execute(sql, params).fetchone()
        return dict(r) if r else None

    def execute(self, sql: str, params: tuple = ()):
        self._c.execute(sql, params)
        self._c.commit()

    def scalar(self, sql: str, params: tuple = (), default: float = 0):
        r = self._c.execute(sql, params).fetchone()
        return r[0] if r and r[0] is not None else default


def _seed(db: _FakeDB, chat: str, rows: list[tuple[str, str, str, float]]):
    db.execute(
        "INSERT INTO conversations (id, title, agent, channel, created_at, updated_at)"
        " VALUES (?, '', 'partner', '', ?, ?)",
        (chat, time.time(), time.time()),
    )
    for i, (role, name, content, ts) in enumerate(rows):
        db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at)"
            " VALUES (?, ?, ?, ?, ?, '', ?)",
            (f"m{i}", chat, role, content, name, ts),
        )


class ChatProfileTest(unittest.TestCase):
    def test_participants_mined_from_history(self):
        db = _FakeDB(_db())
        t = time.time()
        _seed(db, "wa:123", [
            ("user", "Alice", "hey did you see the lagos traffic today", t - 300),
            ("user", "Bob", "yeah the third mainland bridge is blocked", t - 200),
            ("user", "Alice", "traffic in lagos is always crazy", t - 100),
            ("user", "Bob", "we should leave early for the wedding", t - 50),
        ])
        profile = refresh_profile(db, "wa:123", force=True)
        self.assertEqual(profile["participants"]["Alice"], 2)
        self.assertEqual(profile["participants"]["Bob"], 2)
        # Topics come from the chat's own words, not a dictionary.
        self.assertIn("lagos", profile["topics"])
        self.assertIn("traffic", profile["topics"])

    def test_owner_marker_noted(self):
        db = _FakeDB(_db())
        t = time.time()
        _seed(db, "wa:456", [
            ("user", "__owner__", "on my way", t - 100),
            ("user", "Mama", "ok hurry", t - 50),
        ])
        profile = refresh_profile(db, "wa:456", force=True)
        self.assertIn("owner", profile.get("owner_note", ""))

    def test_context_lines_render(self):
        profile = {
            "participants": {"Alice": 10, "Bob": 4},
            "topics": ["lagos", "wedding"],
            "purpose": "a group with 2 voices",
            "owner_note": "the owner is active in this chat",
        }
        lines = build_context_lines(profile)
        self.assertEqual(len(lines), 1)
        self.assertIn("Alice (10)", lines[0])
        self.assertIn("lagos", lines[0])
        self.assertIn("owner is active", lines[0])

    def test_empty_profile_no_lines(self):
        self.assertEqual(build_context_lines({}), [])

    def test_incremental_update(self):
        db = _FakeDB(_db())
        t = time.time()
        _seed(db, "wa:789", [
            ("user", "Zed", "hello world hello world", t - 100),
        ])
        p1 = refresh_profile(db, "wa:789", force=True)
        self.assertEqual(p1["message_count"], 1)
        # Second refresh with no new messages keeps topics.
        p2 = refresh_profile(db, "wa:789")
        self.assertEqual(p2["topics"], p1["topics"])


class DoubleReplyTest(unittest.TestCase):
    def _brain(self):
        from unittest import mock
        from nomorals.agents.partner.brain import PartnerBrain
        ctx = mock.MagicMock()
        db = _FakeDB(_db())
        ctx.db = db
        brain = PartnerBrain.__new__(PartnerBrain)
        brain.context = ctx
        return brain, db

    def test_owner_reply_after_suppresses(self):
        brain, db = self._brain()
        t = time.time()
        _seed(db, "wa:1", [
            ("user", "X", "are you coming?", t - 100),
            ("user", "__owner__", "yes on my way", t - 50),
        ])
        # A message at t-100: the owner's reply at t-50 is newer → addressed.
        self.assertTrue(brain._already_addressed("wa:1", t - 100))

    def test_bot_reply_after_suppresses(self):
        brain, db = self._brain()
        t = time.time()
        _seed(db, "wa:2", [
            ("user", "X", "are you coming?", t - 100),
            ("assistant", "Devon", "on my way!", t - 50),
        ])
        self.assertTrue(brain._already_addressed("wa:2", t - 100))

    def test_fresh_message_not_suppressed(self):
        brain, db = self._brain()
        t = time.time()
        _seed(db, "wa:3", [
            ("user", "X", "are you coming?", t - 100),
        ])
        # Nothing newer than the message itself → not addressed.
        self.assertFalse(brain._already_addressed("wa:3", t - 100))

    def test_new_message_after_owner_reply_allowed(self):
        brain, db = self._brain()
        t = time.time()
        _seed(db, "wa:4", [
            ("user", "X", "are you coming?", t - 200),
            ("user", "__owner__", "yes", t - 150),
            ("user", "X", "bring drinks too", t - 100),
        ])
        # The newest message (t-100) has nothing after it → allowed.
        self.assertFalse(brain._already_addressed("wa:4", t - 100))


class OwnerSenderTest(unittest.TestCase):
    def test_from_owner_meta_flag(self):
        from unittest import mock
        from nomorals.agents.partner.brain import PartnerBrain
        from nomorals.social.chat.base import ChatMessage, ChatRef
        brain = PartnerBrain.__new__(PartnerBrain)
        brain.settings = mock.MagicMock()
        brain.settings.partner.owner_sender_ids = ""
        brain.persona = mock.MagicMock()
        brain.persona.name = "Devon"
        msg = ChatMessage(
            chat=ChatRef(platform="whatsapp", chat_id="g1", kind="group"),
            incoming=False, text="hello", sender="",
        )
        msg.meta["from_owner"] = True
        self.assertTrue(brain._is_owner_sender(msg))
        self.assertTrue(brain._should_speak_in_group(msg))

    def test_owner_sender_id_allowlist(self):
        from unittest import mock
        from nomorals.agents.partner.brain import PartnerBrain
        from nomorals.social.chat.base import ChatMessage, ChatRef
        brain = PartnerBrain.__new__(PartnerBrain)
        brain.settings = mock.MagicMock()
        brain.settings.partner.owner_sender_ids = "2348012345678@s.whatsapp.net"
        brain.persona = mock.MagicMock()
        brain.persona.name = "Devon"
        msg = ChatMessage(
            chat=ChatRef(platform="whatsapp", chat_id="g1", kind="group"),
            incoming=True, text="hello", sender="Me",
            sender_id="2348012345678@s.whatsapp.net",
        )
        self.assertTrue(brain._is_owner_sender(msg))
        stranger = ChatMessage(
            chat=ChatRef(platform="whatsapp", chat_id="g1", kind="group"),
            incoming=True, text="hello", sender="Zed",
            sender_id="2348099999999@s.whatsapp.net",
        )
        self.assertFalse(brain._is_owner_sender(stranger))


if __name__ == "__main__":
    unittest.main()
