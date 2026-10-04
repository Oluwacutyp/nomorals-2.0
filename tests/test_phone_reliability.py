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
