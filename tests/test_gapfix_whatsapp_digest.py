"""WhatsApp group digests (audit gap R2-20).

The organ (nomorals/social/chat/whatsapp_digest.py) is finished pure
code — classify_message / should_digest / build_digest — with zero
production callers. These tests pin the wiring: the /digest command
(in-group + DM board + policy verb) and the scheduled digest tick
(durable job, Notifier delivery, per-group watermark).

Rig copied from tests/test_group_health.py: RuntimeGroupMixin with
monkey-patched seams, bare GroupSurface over a tmp GroupStore, and a
real ConversationLog over Database(":memory:") + migrate().
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.conversations import ConversationLog
from nomorals.storage.db import Database

WA = "whatsapp:120363025000000@g.us"
WA2 = "whatsapp:120363025000001@g.us"
TG = "telegram:-100111"
NOW = 1780000000.0
DAY = 86400.0


def _surface(tmp_path: Path, *, registry=None) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load(),
                        owner_registry=registry)


class _OwnerRegistry:
    """Minimal OwnerRegistry stand-in: one owner id per platform."""

    def __init__(self, owner_ids):
        self._owner_ids = dict(owner_ids)

    def identities(self, platform):
        return list(self._owner_ids.get(platform, []))

    def is_owner(self, platform, user_id):
        return user_id in self._owner_ids.get(platform, [])

    def role_of(self, platform, user_id, *, group_key=""):
        return "owner" if self.is_owner(platform, user_id) else ""


def _seed_group(surface, key=WA, *, members=(), now=NOW):
    for uid, name in members:
        surface.note_join(key, uid, username=name.lower(),
                          name=name, ts=now)
    return surface


def _log(tmp_db=None):
    db = tmp_db if tmp_db is not None else Database(":memory:")
    db.migrate()
    return ConversationLog(db)


_ROW_SEQ = [0]


def _seed_messages(log, key, rows, *, now=NOW):
    """rows: (sender_name, text, ts_offset_s). ConversationLog rows."""
    log.ensure_conversation(key, title=key, agent="partner",
                            channel=key.split(":", 1)[0], now=now)
    for sender, text, off in rows:
        _ROW_SEQ[0] += 1
        log.append_message(row_id=f"m{_ROW_SEQ[0]}",
                           conversation_id=key,
                           role="user", content=text, name=sender,
                           model="", created_at=now + off)
    return log


def _group_msg(key=WA, *, sender_id="admin1", platform="whatsapp"):
    return SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind=ChatKind.GROUP,
                             chat_id=key.split(":", 1)[1]),
        sender_id=sender_id, sender="Admin",
        sender_username="boss", text="", meta={}, reply_to="",
        message_id="900")


def _dm_msg(sender_id="owner1", *, platform="whatsapp"):
    chat = SimpleNamespace(key=f"{platform}:{sender_id}",
                           platform=platform, kind=ChatKind.DM,
                           chat_id=sender_id)
    return SimpleNamespace(chat=chat, text="", meta={},
                           sender="someone", sender_id=sender_id)


def _rig(tmp_path: Path, *, registry=None, operator=False,
         admin_ok=True, console_admin=True, db=None):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    surface = _surface(tmp_path, registry=registry)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._operator_or_false = lambda message: operator  # type: ignore[method-assign]
    mixin._group_admin_ok = lambda message: admin_ok  # type: ignore[method-assign]
    mixin._console_is_admin = lambda k, sender: console_admin  # type: ignore[method-assign]
    mixin.context = SimpleNamespace(db=db)
    return mixin, surface


def _text(out) -> str:
    return getattr(out, "text", out)


class _FakeNotifier:
    def __init__(self):
        self.published = []

    def publish(self, kind, title="", body="", **_kw):
        self.published.append({"kind": kind, "title": title,
                               "body": body})
        return {"ok": True, "delivered": 1}


class OrganReadTests(unittest.TestCase):
    def test_log_rows_map_to_organ_dicts_and_digest_counts(self):
        from nomorals.social.chat.whatsapp_digest import read_group_digest

        log = _log()
        _seed_messages(log, WA, [
            ("Ada", "@Death can you look at this", -3600),
            ("Ben", "should we launch Friday?", -1800),
            ("Ada", "sorted, launching Friday ✅", -900),
            ("Cid", "lunch at noon", -300),
        ])
        digest = read_group_digest(log, WA, me="Death", now=NOW)
        self.assertTrue(digest["included"])
        self.assertEqual(digest["mentions"], 1)
        self.assertEqual(digest["decisions"], 1)
        self.assertEqual(digest["resolved"], 1)
        self.assertIn("mentioned ×1", digest["text"])
        self.assertIn("decisions needing you ×1", digest["text"])

    def test_no_messages_under_default_policy_is_not_included(self):
        from nomorals.social.chat.whatsapp_digest import read_group_digest

        log = _log()
        _seed_messages(log, WA, [("Cid", "lunch at noon", -300)])
        digest = read_group_digest(log, WA, me="Death", now=NOW)
        self.assertFalse(digest["included"])
        self.assertEqual(digest["text"], "")

    def test_window_starts_at_since_not_whole_log(self):
        from nomorals.social.chat.whatsapp_digest import read_group_digest

        log = _log()
        _seed_messages(log, WA, [
            ("Ada", "@Death old mention?", -2 * DAY),
            ("Ben", "should we launch Friday?", -600),
        ])
        digest = read_group_digest(log, WA, me="Death",
                                   since=NOW - DAY, now=NOW)
        self.assertEqual(digest["mentions"], 0)
        self.assertEqual(digest["decisions"], 1)


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._tmp_path = Path(self._tmp.name)
        self._now = time.time()

    def _rig_with_log(self, *, registry=None, operator=True,
                      admin_ok=True, console_admin=True):
        db = Database(":memory:")
        db.migrate()
        log = _log(db)
        mixin, surface = _rig(self._tmp_path, registry=registry,
                              operator=operator, admin_ok=admin_ok,
                              console_admin=console_admin, db=db)
        return mixin, surface, log

    def test_digest_in_group_returns_digest_text(self):
        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        mixin, surface, log = self._rig_with_log(registry=registry)
        _seed_group(surface, WA, members=[("owner1", "Death"),
                                          ("admin1", "Admin")])
        _seed_messages(log, WA, [
            ("Ada", "@Death can you look at this", -3600),
            ("Ben", "should we launch Friday?", -1800),
        ], now=self._now)
        out = mixin._control_digest("", _group_msg())
        text = _text(out)
        self.assertIn("mentioned ×1", text)
        self.assertIn("decisions needing you ×1", text)

    def test_digest_in_group_nothing_needs_you_is_honest(self):
        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        mixin, surface, log = self._rig_with_log(registry=registry)
        _seed_group(surface, WA, members=[("owner1", "Death")])
        _seed_messages(log, WA, [("Cid", "lunch at noon", -300)], now=self._now)
        out = mixin._control_digest("", _group_msg())
        text = _text(out)
        self.assertIn("nothing needs you", text)
        self.assertNotIn("mentioned", text)

    def test_digest_in_group_non_admin_refused(self):
        mixin, surface, log = self._rig_with_log(admin_ok=False)
        _seed_group(surface, WA, members=[("admin1", "Admin")])
        out = mixin._control_digest("", _group_msg())
        self.assertIn("admin", _text(out).lower())

    def test_digest_in_group_telegram_refused(self):
        mixin, surface, log = self._rig_with_log()
        _seed_group(surface, TG, members=[("admin1", "Admin")])
        out = mixin._control_digest("", _group_msg(key=TG,
                                                   platform="telegram"))
        self.assertIn("WhatsApp", _text(out))

    def test_digest_dm_boards_admin_groups(self):
        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        mixin, surface, log = self._rig_with_log(registry=registry)
        _seed_group(surface, WA, members=[("owner1", "Death")])
        _seed_group(surface, WA2, members=[("owner1", "Death")])
        _seed_messages(log, WA, [
            ("Ada", "@Death can you look at this?", -3600)], now=self._now)
        _seed_messages(log, WA2, [("Cid", "lunch at noon", -300)], now=self._now)
        out = mixin._control_digest("", _dm_msg())
        text = _text(out)
        self.assertIn("mentioned ×1", text)

    def test_digest_dm_name_picks_one_group(self):
        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        mixin, surface, log = self._rig_with_log(registry=registry)
        _seed_group(surface, WA, members=[("owner1", "Death")])
        _seed_group(surface, WA2, members=[("owner1", "Death")])
        _seed_messages(log, WA, [
            ("Ada", "@Death can you look at this?", -3600)], now=self._now)
        out = mixin._control_digest(WA, _dm_msg())
        self.assertIn("mentioned ×1", _text(out))

    def test_digest_policy_never_then_nothing(self):
        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        mixin, surface, log = self._rig_with_log(registry=registry)
        _seed_group(surface, WA, members=[("owner1", "Death")])
        _seed_messages(log, WA, [
            ("Ada", "@Death can you look at this?", -3600)], now=self._now)
        set_out = mixin._control_digest("policy never", _group_msg())
        self.assertIn("never", _text(set_out).lower())
        out = mixin._control_digest("", _group_msg())
        self.assertIn("never", _text(out).lower())
        self.assertNotIn("mentioned ×", _text(out))

    def test_digest_policy_always_digests_empty_room(self):
        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        mixin, surface, log = self._rig_with_log(registry=registry)
        _seed_group(surface, WA, members=[("owner1", "Death")])
        _seed_messages(log, WA, [("Cid", "lunch at noon", -300)], now=self._now)
        mixin._control_digest("policy always", _group_msg())
        out = mixin._control_digest("", _group_msg())
        self.assertIn("nothing needs you", _text(out))


class TickTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._tmp_path = Path(self._tmp.name)

    def test_tick_publishes_earning_groups_and_advances_watermark(self):
        from nomorals.social.chat.whatsapp_digest import run_group_digest_tick_once

        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        surface = _surface(self._tmp_path, registry=registry)
        surface.set_designated(WA, True, by="owner")
        surface.set_designated(WA2, True, by="owner")
        _seed_group(surface, WA, members=[("owner1", "Death")])
        _seed_group(surface, WA2, members=[("owner1", "Death")])
        db = Database(":memory:")
        db.migrate()
        log = _log(db)
        _seed_messages(log, WA, [
            ("Ada", "@Death can you look at this?", -3600)])
        _seed_messages(log, WA2, [("Cid", "lunch at noon", -300)])
        notifier = _FakeNotifier()
        ctx = SimpleNamespace(db=db)
        result = run_group_digest_tick_once(
            context=ctx, surface=surface, log=log,
            notifier=notifier, now=NOW)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["posted"]), 1)
        self.assertEqual(result["posted"][0]["group"], WA)
        self.assertIn("mentioned ×1", result["posted"][0]["body"])
        self.assertEqual(len(notifier.published), 1)
        self.assertEqual(notifier.published[0]["kind"], "digest")
        self.assertIn("mentioned ×1", notifier.published[0]["body"])
        # Second run, no new messages: nothing publishes.
        result2 = run_group_digest_tick_once(
            context=ctx, surface=surface, log=log,
            notifier=notifier, now=NOW + 60)
        self.assertTrue(result2["ok"])
        self.assertEqual(len(notifier.published), 1)

    def test_tick_skips_never_policy_group(self):
        from nomorals.social.chat.whatsapp_digest import run_group_digest_tick_once

        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        surface = _surface(self._tmp_path, registry=registry)
        surface.set_designated(WA, True, by="owner")
        _seed_group(surface, WA, members=[("owner1", "Death")])
        surface.update_settings(WA, digest_policy="never")
        db = Database(":memory:")
        db.migrate()
        log = _log(db)
        _seed_messages(log, WA, [
            ("Ada", "@Death can you look at this?", -3600)])
        notifier = _FakeNotifier()
        result = run_group_digest_tick_once(
            context=SimpleNamespace(db=db), surface=surface,
            log=log, notifier=notifier, now=NOW)
        self.assertTrue(result["ok"])
        self.assertEqual(notifier.published, [])

    def test_tick_without_notifier_returns_payloads(self):
        from nomorals.social.chat.whatsapp_digest import run_group_digest_tick_once

        registry = _OwnerRegistry({"whatsapp": ["owner1"]})
        surface = _surface(self._tmp_path, registry=registry)
        surface.set_designated(WA, True, by="owner")
        _seed_group(surface, WA, members=[("owner1", "Death")])
        db = Database(":memory:")
        db.migrate()
        log = _log(db)
        _seed_messages(log, WA, [
            ("Ada", "@Death can you look at this", -3600)])
        result = run_group_digest_tick_once(
            context=SimpleNamespace(db=db), surface=surface,
            log=log, now=NOW)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["posted"]), 1)
        self.assertEqual(result["posted"][0]["group"], WA)
        self.assertIn("mentioned ×1", result["posted"][0]["body"])


class JobTests(unittest.TestCase):
    def test_ensure_digest_job_registers_durable_routine_tool_job(self):
        from nomorals.social.chat.whatsapp_digest import DIGEST_JOB, ensure_group_digest_job

        class _Sched:
            def __init__(self):
                self.jobs = []

            def list_jobs(self):
                return list(self.jobs)

            def add(self, name, spec, kind, payload, routine=False):
                job = {"name": name, "spec": spec, "kind": kind,
                       "payload": payload, "routine": routine}
                self.jobs.append(job)
                return job

            def remove(self, job_id):
                self.jobs = [j for j in self.jobs
                             if j.get("id") != job_id]

        sched = _Sched()
        first = ensure_group_digest_job(sched)
        self.assertEqual(first["name"], DIGEST_JOB)
        self.assertTrue(first["routine"])
        second = ensure_group_digest_job(sched)
        self.assertEqual(second, first)
        self.assertEqual(len(sched.jobs), 1)


class RegistryTests(unittest.TestCase):
    def test_digest_is_public_and_has_menu_line(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            PUBLIC_COMMANDS,
            help_text,
            list_catalog,
        )

        self.assertIn("digest", PUBLIC_COMMANDS)
        self.assertIn("digest", COMMAND_DETAILS)
        self.assertIn("/digest", help_text())
        self.assertIn("/digest", list_catalog())


if __name__ == "__main__":
    unittest.main()
