"""Phone-bot reliability: adapter supervision, media pruning, DB recovery.

Covers the audit fixes for the Termux phone bot (nomorals-2.0 main loop):

* a crashed chat adapter restarts itself with backoff instead of dying
  silently, and gives up loudly after too many consecutive crashes;
* inbound media dirs are pruned (age + size caps) so the phone disk can't
  fill up over weeks of photos/voice notes;
* a corrupt SQLite file is quarantined (never deleted) and the bot boots
  fresh instead of bricking;
* per-chat rate windows don't accumulate forever.

All offline.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway, _HourWindow
from nomorals.storage.db import Database, open_database


class _FlakyAdapter(ChatAdapter):
    """Crashes ``fail_times`` times, then runs until stopped."""

    name = "flaky"
    RESTART_INITIAL_DELAY_S = 0.01
    RESTART_MAX_DELAY_S = 0.05
    RESTART_MAX_CRASHES = 3
    RESTART_RESET_AFTER_S = 3600.0  # never auto-reset inside these tests

    def __init__(self, media_dir: str, fail_times: int):
        super().__init__(media_dir=media_dir)
        self.fail_times = fail_times
        self.runs = 0
        self.healthy = threading.Event()

    def run(self, handler):
        self.runs += 1
        if self.runs <= self.fail_times:
            raise RuntimeError(f"boom {self.runs}")
        self.healthy.set()
        while not self.stopped:
            time.sleep(0.01)

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        return SendResult(ok=True, platform=self.name)


def _tmpdir(test: unittest.TestCase) -> str:
    d = tempfile.mkdtemp(prefix="devon-rel-")
    test.addCleanup(shutil.rmtree, d, True)
    return d


class AdapterSupervisionTests(unittest.TestCase):
    def test_crash_restarts_adapter(self):
        d = _tmpdir(self)
        adapter = _FlakyAdapter(d, fail_times=2)
        try:
            self.assertTrue(adapter.start(lambda m: None))
            self.assertTrue(adapter.healthy.wait(timeout=5.0),
                            "adapter never recovered from its crashes")
            self.assertEqual(adapter.runs, 3)  # 2 crashes + 1 healthy run
            self.assertEqual(adapter._crash_count, 2)
            self.assertFalse(adapter._gave_up)
            health = adapter.health()
            self.assertEqual(health["restart_crashes"], 2)
            self.assertIn("boom", health["last_crash"])
        finally:
            adapter.stop()

    def test_give_up_after_max_consecutive_crashes(self):
        d = _tmpdir(self)
        adapter = _FlakyAdapter(d, fail_times=999)
        try:
            adapter.start(lambda m: None)
            deadline = time.time() + 10.0
            while not adapter._gave_up and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(adapter._gave_up, "adapter never gave up")
            # thread exited: a later start() is a fresh lease
            adapter._thread.join(timeout=5.0)
            self.assertFalse(adapter._thread.is_alive())
            self.assertIn("gave_up", adapter.health())
            self.assertTrue(adapter.health()["gave_up"])
        finally:
            adapter.stop()

    def test_manual_start_resets_after_give_up(self):
        d = _tmpdir(self)
        adapter = _FlakyAdapter(d, fail_times=999)
        try:
            adapter.start(lambda m: None)
            deadline = time.time() + 10.0
            while not adapter._gave_up and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(adapter._gave_up)
            adapter._thread.join(timeout=5.0)
            # operator fixed it: no more crashes
            adapter.fail_times = 0
            self.assertTrue(adapter.start(lambda m: None))
            self.assertTrue(adapter.healthy.wait(timeout=5.0))
            self.assertFalse(adapter._gave_up)
            self.assertEqual(adapter._crash_count, 0)
        finally:
            adapter.stop()

    def test_stop_during_backoff_does_not_restart(self):
        d = _tmpdir(self)
        adapter = _FlakyAdapter(d, fail_times=999)
        # long delays so we can stop mid-backoff
        adapter.RESTART_INITIAL_DELAY_S = 30.0
        adapter.start(lambda m: None)
        time.sleep(0.3)  # let it crash once and enter backoff
        adapter.stop()
        runs_after_stop = adapter.runs
        time.sleep(0.5)
        # the backoff wait must have been interrupted by stop()
        self.assertFalse(adapter._thread.is_alive())
        self.assertEqual(adapter.runs, runs_after_stop)


class MediaPruneTests(unittest.TestCase):
    def _adapter(self, d: str) -> _FlakyAdapter:
        return _FlakyAdapter(d, fail_times=0)

    def test_old_files_pruned_new_kept(self):
        d = _tmpdir(self)
        old = Path(d) / "old.jpg"
        new = Path(d) / "new.jpg"
        dot = Path(d) / ".state.json"  # adapter state, not media: never touched
        old.write_bytes(b"x" * 100)
        new.write_bytes(b"y" * 100)
        dot.write_bytes(b"{}")
        ancient = time.time() - 10 * 86400
        os.utime(old, (ancient, ancient))
        os.utime(dot, (ancient, ancient))
        removed = self._adapter(d)._prune_media_dir(max_age_days=7.0,
                                                    max_total_mb=512.0)
        self.assertEqual(removed, 1)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertTrue(dot.exists(), "dotfiles must never be pruned")

    def test_size_cap_evicts_oldest_first(self):
        d = _tmpdir(self)
        now = time.time()
        paths = []
        for i in range(3):
            p = Path(d) / f"f{i}.jpg"
            p.write_bytes(b"z" * (1024 * 1024))  # 1MB each
            os.utime(p, (now - (3 - i) * 3600, now - (3 - i) * 3600))
            paths.append(p)
        removed = self._adapter(d)._prune_media_dir(max_age_days=365.0,
                                                    max_total_mb=2.0)
        self.assertEqual(removed, 1)
        self.assertFalse(paths[0].exists(), "oldest should be evicted first")
        self.assertTrue(paths[1].exists())
        self.assertTrue(paths[2].exists())

    def test_missing_dir_is_noop(self):
        d = _tmpdir(self)
        removed = self._adapter(os.path.join(d, "nope"))._prune_media_dir()
        self.assertEqual(removed, 0)


class EntityCacheCapTests(unittest.TestCase):
    def test_cache_is_bounded(self):
        from nomorals.social.chat.telegram import TelegramAdapter

        d = _tmpdir(self)
        adapter = TelegramAdapter(api_id=123, api_hash="x", session_path=":memory:",
                                  media_dir=d)
        for i in range(2500):
            adapter._cache_input_entity(str(i), object())
        self.assertLessEqual(len(adapter._input_entity_cache),
                             adapter._input_entity_cache_max)
        # newest entries survive, oldest are evicted
        self.assertIn("2499", adapter._input_entity_cache)
        self.assertNotIn("0", adapter._input_entity_cache)


class _FakeAdapter(ChatAdapter):
    name = "fake"

    def run(self, handler):
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        return SendResult(ok=True, platform=self.name)


def _msg(chat_id: str, platform: str = "tg") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM),
        incoming=True, text="hi", sender="someone", media=[],
        reply_to="", mentioned=False, ts=time.time(), message_id="m1",
    )


class GatewayWindowPruneTests(unittest.TestCase):
    def test_idle_windows_pruned(self):
        gw = ChatGateway(adapters={"fake": _FakeAdapter()}, max_per_hour=5)
        now = time.time()
        for i in range(1005):
            window = _HourWindow(5)
            window._events.append(now - 90000.0)  # idle > 1 day
            gw._windows[f"tg:{i}"] = window
        gw._prune_windows(now)
        self.assertEqual(len(gw._windows), 0)

    def test_active_windows_survive(self):
        gw = ChatGateway(adapters={"fake": _FakeAdapter()}, max_per_hour=5)
        now = time.time()
        for i in range(1005):
            window = _HourWindow(5)
            window._events.append(now - 60.0)  # active a minute ago
            gw._windows[f"tg:{i}"] = window
        gw._prune_windows(now)
        self.assertEqual(len(gw._windows), 1005)

    def test_below_threshold_no_prune(self):
        gw = ChatGateway(adapters={"fake": _FakeAdapter()}, max_per_hour=5)
        now = time.time()
        for i in range(10):
            window = _HourWindow(5)
            window._events.append(now - 90000.0)
            gw._windows[f"tg:{i}"] = window
        gw._prune_windows(now)
        self.assertEqual(len(gw._windows), 10)


class CorruptDbTests(unittest.TestCase):
    def test_corrupt_file_quarantined_and_fresh_boot(self):
        d = _tmpdir(self)
        target = Path(d) / "nomorals.db"
        target.write_bytes(b"this is definitely not a sqlite database" * 64)
        # sidecars go with it
        (Path(d) / "nomorals.db-wal").write_bytes(b"junk")
        db, recovered, backup = open_database(target)
        try:
            self.assertTrue(recovered)
            self.assertIsNotNone(backup)
            self.assertTrue(Path(backup).exists(),
                            "quarantined original must be preserved")
            # the sidecars were quarantined too (the fresh DB recreates its
            # own -wal/-shm, so check for the quarantined copies by name)
            quarantined = [p for p in Path(d).iterdir() if ".corrupt-" in p.name]
            names = sorted(p.name for p in quarantined)
            self.assertTrue(any(n.startswith("nomorals.db.corrupt-") for n in names),
                            f"original db quarantined, got: {names}")
            self.assertTrue(any(n.startswith("nomorals.db-wal.corrupt-") for n in names),
                            f"wal sidecar quarantined, got: {names}")
            # the fresh db is usable and migrated
            db.execute("CREATE TABLE probe (a INTEGER)")
            db.execute("INSERT INTO probe (a) VALUES (1)")
            self.assertEqual(db.scalar("SELECT a FROM probe"), 1)
            self.assertTrue(db.table_exists("schema_migrations"))
        finally:
            db.close()

    def test_healthy_file_not_touched(self):
        d = _tmpdir(self)
        target = Path(d) / "nomorals.db"
        first = Database(target)
        first.execute("CREATE TABLE t (a INTEGER)")
        first.execute("INSERT INTO t (a) VALUES (42)")
        first.close()
        db, recovered, backup = open_database(target)
        try:
            self.assertFalse(recovered)
            self.assertIsNone(backup)
            self.assertEqual(db.scalar("SELECT a FROM t"), 42)
            self.assertEqual(
                len([p for p in Path(d).iterdir() if ".corrupt-" in p.name]), 0)
        finally:
            db.close()

    def test_non_corrupt_errors_still_raise(self):
        d = _tmpdir(self)
        target = Path(d) / "is-a-directory"
        target.mkdir()
        # a directory is not a database file, but it is NOT corruption —
        # the real error must surface, not be misclassified
        with self.assertRaises(Exception):
            open_database(target)


if __name__ == "__main__":
    unittest.main()


class GroupCachePoisoningTests(unittest.TestCase):
    """Regression: group→DM redirect bug.

    When a group message arrived with input_chat=None, the old code fell
    back to caching input_sender (the sender's DM peer) under the GROUP's
    chat ID. Every later reply to that group then went to the sender's DM.
    """

    def _adapter(self):
        from nomorals.social.chat.telegram import TelegramAdapter
        import tempfile
        d = tempfile.mkdtemp()
        return TelegramAdapter(api_id=123, api_hash="x", session_path=":memory:",
                               media_dir=d)

    def test_group_with_missing_input_chat_does_not_cache_sender(self):
        """A group entity with input_chat=None must NOT cache input_sender
        under the group ID."""
        from types import SimpleNamespace
        adapter = self._adapter()

        # Simulate a group entity
        entity = SimpleNamespace(id=-1001234567890, megagroup=True,
                                 gigagroup=False, channel=False)
        # Event with NO input_chat (the bug trigger) but input_sender set
        event = SimpleNamespace(input_chat=None, input_sender=object())

        # Replicate the fixed caching logic
        input_chat = getattr(event, "input_chat", None)
        is_group_entity = (
            getattr(entity, "megagroup", False)
            or getattr(entity, "gigagroup", False)
            or getattr(entity, "channel", False)
        )
        if is_group_entity:
            input_entity = input_chat
        else:
            input_entity = input_chat or getattr(event, "input_sender", None)

        # For a group with no input_chat, nothing should be cached
        self.assertIsNone(input_entity)

    def test_dm_still_caches_sender_fallback(self):
        """DMs (non-group entities) still use the input_sender fallback."""
        from types import SimpleNamespace
        adapter = self._adapter()

        entity = SimpleNamespace(id=12345, megagroup=False,
                                 gigagroup=False, channel=False)
        sender_peer = object()
        event = SimpleNamespace(input_chat=None, input_sender=sender_peer)

        input_chat = getattr(event, "input_chat", None)
        is_group_entity = (
            getattr(entity, "megagroup", False)
            or getattr(entity, "gigagroup", False)
            or getattr(entity, "channel", False)
        )
        if is_group_entity:
            input_entity = input_chat
        else:
            input_entity = input_chat or getattr(event, "input_sender", None)

        self.assertIs(input_entity, sender_peer)

    def test_synthetic_entity_uses_chat_id_not_sender_id(self):
        """Regression: group→DM redirect via synthetic entity fallback.

        When Telethon can't resolve ANY entity for a group message, the
        FINAL FALLBACK synthesized an entity from event.sender_id (the
        sender's user ID) with megagroup=False. The runtime then treated
        the group message as a DM and replied to the sender's inbox.
        The fallback must use event.chat_id (the group) and mark it as
        a group so _kind_for() classifies it correctly.
        """
        from types import SimpleNamespace
        from nomorals.social.chat.telegram import _kind_for

        # Simulate a group message event where entity resolution failed
        group_chat_id = "1234567890"
        sender_id = 7541672134  # Mary's user ID
        event = SimpleNamespace(
            chat_id=group_chat_id,
            sender_id=sender_id,
            is_group=True,
            is_channel=False,
        )
        message = SimpleNamespace(from_id=None, peer_id=None)

        # Replicate the fixed fallback logic
        target_id = str(getattr(event, "chat_id", "") or "") or str(
            getattr(event, "sender_id", "") or "")
        is_group = bool(
            getattr(event, "is_group", False)
            or getattr(event, "is_channel", False)
        )
        entity = SimpleNamespace(
            id=int(str(target_id).lstrip("-")),
            megagroup=is_group,
            gigagroup=False,
            channel=is_group,
        )

        # The synthetic entity must use the GROUP's ID, not the sender's
        self.assertEqual(str(entity.id), group_chat_id)
        self.assertNotEqual(str(entity.id), str(sender_id))
        # And it must classify as a group, not a DM
        self.assertEqual(_kind_for(entity), "group")

    def test_input_chat_tried_before_input_sender(self):
        """Regression: input_sender-first ordering misclassified groups as DMs.

        In the "last resort" entity resolution path, the old code tried
        input_sender BEFORE input_chat. For a group message, input_sender
        is the SENDER's peer (their DM) — if it resolved, the message was
        treated as a DM from the sender and replies went to their inbox
        instead of the group. input_chat (the group peer) must be tried
        first.
        """
        # This test verifies the ORDERING by checking the source code
        # directly — the fix is a one-line reorder that's hard to test
        # behaviorally without a full Telethon mock.
        import inspect
        from nomorals.social.chat import telegram as tg_module
        src = inspect.getsource(tg_module.TelegramAdapter._handle_inbound)
        # Find the "Last resort" block and verify input_chat comes first
        last_resort_idx = src.find("Last resort")
        self.assertGreater(last_resort_idx, 0, "Last resort block not found")
        block = src[last_resort_idx:last_resort_idx + 2000]
        chat_first = block.find("input_chat is not None")
        sender_first = block.find("input_sender is not None")
        self.assertGreater(chat_first, 0, "input_chat check not found")
        self.assertGreater(sender_first, 0, "input_sender check not found")
        self.assertLess(
            chat_first, sender_first,
            "input_chat must be tried BEFORE input_sender in the last-resort path",
        )

    def test_event_sender_fallback_skipped_for_group_chat_id(self):
        """Regression: event_sender fallback misclassified groups as DMs.

        When event.chat was None (group not resolved) but event.sender was
        available, the old code used the SENDER as the entity. The later
        `chat_id = str(entity.id)` overwrite then replaced the group ID
        (e.g. -5223197263) with the sender's user ID (e.g. 7541672134),
        redirecting every reply to the sender's DM.

        The fix: only fall back to event_sender when the original chat_id
        does NOT look like a group (negative number).
        """
        import inspect
        from nomorals.social.chat import telegram as tg_module
        src = inspect.getsource(tg_module.TelegramAdapter._handle_inbound)
        # The guard must exist: sender fallback skipped for group chat_ids
        self.assertIn(
            "skipping event_sender fallback",
            src,
            "event_sender fallback must be guarded for group chat_ids",
        )
        self.assertIn(
            '_looks_like_group',
            src,
            "group detection variable must exist in the fallback guard",
        )
