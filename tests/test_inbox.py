"""Acceptance tests for Prompt 06 — the drop-in inbox.

Covers the spec's contract: PDF drop → summary + dated processed move;
@remind → scheduled reminder; executables → quarantine with one notice;
ambiguous items → exactly one question; stale processing → recovery;
room-scoped add-link; secret redaction; and the Prompt 15 media directives
(@square / @trim / @gif) routing through the media engines.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

import nomorals.workspace.inbox as inbox_mod
from nomorals.storage.db import Database
from nomorals.workspace.inbox import (
    ActionResult,
    Inbox,
    InboxWatcher,
    parse_directive,
    parse_when,
    redact_secrets,
    sanitize_name,
)

try:
    from PIL import Image as _PILImage
    PILLOW = True
except ImportError:
    PILLOW = False


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeNotifier:
    def __init__(self):
        self.published = []

    def publish(self, kind, title, body="", *, force=False, critical=False):
        self.published.append({"kind": kind, "title": title,
                               "body": body, "critical": critical})
        return {"id": f"n{len(self.published)}", "delivered": True}


class FakeScheduler:
    def __init__(self):
        self.reminders = []

    def create_reminder(self, text, due_at, user_id, **kwargs):
        self.reminders.append({"text": text, "due_at": due_at,
                               "user_id": user_id, **kwargs})
        return {"task_id": f"r{len(self.reminders)}"}


def _fake_fetch(url):
    return ("<html><head><title>Test Page</title></head><body>"
            "<p>Hello world. This is a fetched test article about markets.</p>"
            "</body></html>")


def _make_image(path, size=(1600, 1200), color=(90, 140, 200)):
    img = _PILImage.new("RGB", size, color)
    img.save(path, "JPEG", quality=85)


class InboxTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="inbox_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.notifier = FakeNotifier()
        self.scheduler = FakeScheduler()
        self.inbox = Inbox(
            self.root, db=Database(":memory:"),
            notifier=self.notifier, scheduler=self.scheduler,
            fetcher=_fake_fetch)

    def _drop(self, name, data, room=None):
        staging = self.root / "staging"
        staging.mkdir(exist_ok=True)
        src = staging / name
        if isinstance(data, str):
            src.write_text(data, encoding="utf-8")
        else:
            src.write_bytes(data)
        return self.inbox.intake(src, room=room)

    def _inbox(self, *parts):
        return self.root / "inbox" / Path(*parts)

    def test_pdf_drop_summarized_and_processed(self):
        pdf = (b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
               b"Hello world this is a test document about quarterly revenue.\n"
               b"Revenue grew twenty percent year over year in the quarter.\n")
        item = self._drop("report.pdf", pdf)
        report = self.inbox.sweep()
        done = self.inbox.get_item(item.id)
        self.assertEqual(done.status, "done")
        self.assertEqual(report["intents"].get("summarize"), 1)
        kinds = [n["kind"] for n in self.notifier.published]
        self.assertIn("inbox", kinds)
        bodies = " ".join(n["body"] for n in self.notifier.published)
        self.assertIn("summarized", bodies)
        self.assertIn("quarterly", bodies)
        today = datetime.now().strftime("%Y-%m-%d")
        self.assertTrue((self._inbox("processed") / today /
                         "report.pdf").exists())

    def test_remind_directive_schedules_reminder(self):
        item = self._drop("todo.md", "@remind friday 9am\ncall the accountant")
        report = self.inbox.sweep()
        self.assertEqual(self.inbox.get_item(item.id).status, "done")
        self.assertEqual(report["intents"].get("remind"), 1)
        self.assertEqual(len(self.scheduler.reminders), 1)
        due = self.scheduler.reminders[0]["due_at"]
        dt = datetime.fromtimestamp(due)
        self.assertEqual(dt.weekday(), 4)  # Friday
        self.assertEqual(dt.hour, 9)
        self.assertGreater(due, time.time())

    def test_exe_and_double_extension_quarantined_once(self):
        item1 = self._drop("evil.exe", b"MZ" + b"\x90" * 100)
        item2 = self._drop("report.pdf.exe",
                           b"#!/bin/sh\necho pwned\n")
        self.inbox.sweep()
        self.assertEqual(self.inbox.get_item(item1.id).status, "quarantined")
        self.assertEqual(self.inbox.get_item(item2.id).status, "quarantined")
        qdir = self._inbox("quarantine")
        self.assertTrue((qdir / "evil.exe").exists())
        self.assertTrue((qdir / "report.pdf.exe").exists())
        notices = [n for n in self.notifier.published if n["critical"]]
        self.assertEqual(len(notices), 2)
        for n in notices:
            self.assertIn("quarantine", n["title"].lower())
        # nothing was summarized or executed — no other notices at all
        self.assertEqual(len(self.notifier.published), 2)

    def test_ambiguous_item_gets_exactly_one_question(self):
        item = self._drop("mystery.xyz", "some opaque blob content")
        self.inbox.sweep()
        self.assertEqual(self.inbox.get_item(item.id).status, "needs_input")
        self.assertEqual(len(self.notifier.published), 1)
        self.assertIn("mystery.xyz", self.notifier.published[0]["body"])
        # a second sweep asks nothing more
        self.inbox.sweep()
        self.assertEqual(len(self.notifier.published), 1)
        self.assertEqual(self.inbox.get_item(item.id).status, "needs_input")

    def test_add_link_routes_to_room_and_researches(self):
        item = self.inbox.add_link("https://example.com/article",
                                   room="trading", note="read this")
        self.assertEqual(item.room, "trading")
        self.assertTrue((self.root / "rooms" / "trading" / "inbox"
                         / item.name).exists())
        self.inbox.sweep()
        done = self.inbox.get_item(item.id)
        self.assertEqual(done.status, "done")
        bodies = " ".join(n["body"] for n in self.notifier.published)
        self.assertIn("researched", bodies)
        self.assertIn("markets", bodies)

    def test_secrets_redacted_from_notifications(self):
        body = ("notes\napi_key = sk-abcdefghijklmnopqrstuvwx\n"
                "password: hunter2hunter2\nend\n")
        self._drop("creds.md", body)
        self.inbox.sweep()
        joined = " ".join(n["body"] for n in self.notifier.published)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", joined)
        self.assertNotIn("hunter2hunter2", joined)
        self.assertIn("[redacted]", joined)

    def test_size_cap_parks_for_input(self):
        old = inbox_mod.MAX_FILE_BYTES
        inbox_mod.MAX_FILE_BYTES = 64
        try:
            item = self._drop("big.txt", "x" * 1024)
        finally:
            inbox_mod.MAX_FILE_BYTES = old
        self.inbox.sweep()
        done = self.inbox.get_item(item.id)
        self.assertEqual(done.status, "needs_input")
        self.assertIn("too large", (done.error or "").lower())

    def test_release_returns_quarantined_item(self):
        item = self._drop("tool.exe", b"MZ" + b"\x90" * 64)
        self.inbox.sweep()
        self.assertEqual(self.inbox.get_item(item.id).status, "quarantined")
        released = self.inbox.release(item.id)
        self.assertEqual(released.status, "pending")
        self.assertTrue((self._inbox("tool.exe")).exists())
        hist = self.inbox.history(item.id)
        self.assertTrue(any(h["outcome"] == "released" for h in hist))

    def test_transcribe_without_backend_parks_for_input(self):
        item = self._drop("voice.wav", b"RIFF" + b"\x00" * 100)
        self.inbox.sweep()
        done = self.inbox.get_item(item.id)
        self.assertEqual(done.status, "needs_input")
        self.assertEqual(len(self.notifier.published), 1)
        self.assertIn("transcription",
                      self.notifier.published[0]["body"].lower())

    def test_history_records_processing(self):
        item = self._drop("note.md", "hello inbox")
        self.inbox.sweep()
        hist = self.inbox.history(item.id)
        self.assertTrue(hist)
        self.assertEqual(hist[0]["intent"], "summarize")
        self.assertEqual(hist[0]["outcome"], "processed")

    def test_retry_requeues_needs_input(self):
        item = self._drop("mystery.xyz", "blob")
        self.inbox.sweep()
        self.assertEqual(self.inbox.get_item(item.id).status, "needs_input")
        self.inbox.retry(item.id)
        self.assertEqual(self.inbox.get_item(item.id).status, "pending")

    def test_watcher_sweep_now(self):
        self._drop("ping.md", "hello")
        watcher = InboxWatcher(self.inbox)
        watcher.sweep_now()
        items = self.inbox.list_items(status="done")
        self.assertEqual(len(items), 1)


class CrashRecoveryTestCase(unittest.TestCase):
    def test_stale_processing_returns_to_pending_and_runs_once(self):
        root = Path(tempfile.mkdtemp(prefix="inbox_crash_"))
        self.addCleanup(shutil.rmtree, root, True)
        dbfile = root / "inbox.db"

        calls = []

        def classify(inbox, item):
            return "countme"

        def handle(inbox, item):
            calls.append(item.id)
            return ActionResult(summary="counted", notify="counted",
                                disposition="processed")

        inbox1 = Inbox(root, db=Database(str(dbfile)),
                       notifier=FakeNotifier(), scheduler=FakeScheduler(),
                       fetcher=_fake_fetch,
                       classifier=classify, handlers={"countme": handle})
        src = root / "src.txt"
        src.write_text("data", encoding="utf-8")
        item = inbox1.intake(src)
        # simulate a crash: claimed 2h ago, never finished
        inbox1.db.execute(
            "UPDATE inbox_items SET status='processing',"
            " updated_at=? WHERE id=?",
            (time.time() - 7200, item.id))

        # a fresh process boots and recovers
        inbox2 = Inbox(root, db=Database(str(dbfile)),
                       notifier=FakeNotifier(), scheduler=FakeScheduler(),
                       fetcher=_fake_fetch,
                       classifier=classify, handlers={"countme": handle})
        recovered = inbox2.get_item(item.id)
        self.assertEqual(recovered.status, "pending")
        inbox2.sweep()
        self.assertEqual(inbox2.get_item(item.id).status, "done")
        self.assertEqual(calls, [item.id])  # processed exactly once


class IdempotencyTestCase(unittest.TestCase):
    def test_double_sweep_processes_once(self):
        root = Path(tempfile.mkdtemp(prefix="inbox_idem_"))
        self.addCleanup(shutil.rmtree, root, True)
        calls = []

        def handle(inbox, item):
            calls.append(item.id)
            return ActionResult(summary="ok", notify="ok",
                                disposition="processed")

        inbox = Inbox(root, db=Database(":memory:"),
                      notifier=FakeNotifier(), scheduler=FakeScheduler(),
                      fetcher=_fake_fetch,
                      classifier=lambda i, it: "countme",
                      handlers={"countme": handle})
        src = root / "src.txt"
        src.write_text("data", encoding="utf-8")
        inbox.intake(src)
        inbox.sweep()
        inbox.sweep()
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(PILLOW, "Pillow not installed")
class MediaDirectiveTestCase(unittest.TestCase):
    """Prompt 15 directives (@square / @trim / @gif) through the inbox."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="inbox_media_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.notifier = FakeNotifier()
        self.inbox = Inbox(
            self.root, db=Database(":memory:"),
            notifier=self.notifier, scheduler=FakeScheduler(),
            fetcher=_fake_fetch)

    def _drop_image(self, name):
        staging = self.root / "staging"
        staging.mkdir(exist_ok=True)
        src = staging / name
        _make_image(src)
        return self.inbox.intake(src)

    def _drop_note(self, name, text):
        staging = self.root / "staging"
        staging.mkdir(exist_ok=True)
        note = staging / name
        note.write_text(text, encoding="utf-8")
        return self.inbox.intake(note)

    def _inbox(self, *parts):
        return self.root / "inbox" / Path(*parts)

    def test_square_sidecar_edits_image(self):
        self._drop_image("photo.jpg")
        self._drop_note("photo.md", "@square\nmake it square for instagram")
        report = self.inbox.sweep()
        self.assertEqual(report["intents"].get("media_square"), 1)
        today = datetime.now().strftime("%Y-%m-%d")
        outs = list((self._inbox("processed") / today).glob("photo-*.jpeg"))
        self.assertTrue(outs, "edited artifact missing")
        info = _PILImage.open(outs[0])
        self.assertEqual((info.width, info.height), (1080, 1080))
        # the sidecar was consumed, not processed twice
        rows = self.inbox.db.query(
            "SELECT status, intent FROM inbox_items")
        self.assertTrue(all(r["status"] == "done" for r in rows))
        intents = [r["intent"] for r in rows]
        self.assertEqual(intents.count("media_square"), 1)
        self.assertIn("consumed", intents)

    def test_note_targeting_named_file(self):
        self._drop_image("photo2.jpg")
        self._drop_note("direct.md", "@square photo2.jpg")
        self.inbox.sweep()
        today = datetime.now().strftime("%Y-%m-%d")
        outs = list((self._inbox("processed") / today).glob("photo2-*.jpeg"))
        self.assertTrue(outs, "named target was not edited")
        rows = {r["id"]: r for r in
                self.inbox.db.query("SELECT id, status FROM inbox_items")}
        self.assertTrue(all(r["status"] == "done" for r in rows.values()))

    def test_plain_image_drop_describes_without_directive(self):
        # Prompt 09: images now classify to the dedicated "image" intent
        # (describe by default, probe-only when no vision hook is wired).
        self._drop_image("plain.jpg")
        report = self.inbox.sweep()
        self.assertEqual(report["intents"].get("image"), 1)
        bodies = " ".join(n["body"] for n in self.notifier.published)
        self.assertIn("1600x1200", bodies)


class HelperTestCase(unittest.TestCase):
    def test_parse_when_variants(self):
        now = time.time()
        fri = parse_when("friday 9am", now)
        dt = datetime.fromtimestamp(fri)
        self.assertEqual((dt.weekday(), dt.hour), (4, 9))
        self.assertGreater(fri, now)
        self.assertIsNotNone(parse_when("next monday 17:30", now))
        self.assertIsNotNone(parse_when("tomorrow", now))
        self.assertIsNotNone(parse_when("in 2 hours", now))
        self.assertIsNotNone(parse_when("2026-10-05 09:00", now))
        self.assertIsNone(parse_when("sometime eventually", now))

    def test_parse_directive(self):
        self.assertEqual(parse_directive("@square photo.jpg"),
                         ("square", "photo.jpg"))
        self.assertEqual(parse_directive("@trim 0:30-1:00"),
                         ("trim", "0:30-1:00"))
        self.assertEqual(parse_directive("@gif"), ("gif", ""))
        self.assertIsNone(parse_directive("just a note"))
        self.assertEqual(parse_directive("@bogus x"), ("bogus", "x"))
        # unknown directives parse fine; classify() turns them into needs_input

    def test_redact_secrets(self):
        s = redact_secrets("api_key = sk-abcdefghijklmnopqrst and "
                           "password: hunter2hunter2 ok")
        self.assertNotIn("sk-abcdefghijklmnopqrst", s)
        self.assertNotIn("hunter2hunter2", s)
        self.assertIn("[redacted]", s)

    def test_sanitize_name(self):
        self.assertEqual(sanitize_name("../../etc/passwd"), "passwd")
        self.assertNotIn("..", sanitize_name(".."))
        self.assertEqual(sanitize_name("  spaced name .txt "),
                         "spaced name .txt")


class CLIInboxTestCase(unittest.TestCase):
    """Exercise `nm inbox` through the real CLI entry point."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="icli_"))
        self.addCleanup(shutil.rmtree, self.home, True)
        (self.home / ".nomorals" / "workspace").mkdir(parents=True)
        self.env = dict(os.environ, HOME=str(self.home),
                        PYTHONPATH="/home/hatch/workspace/devon")

    def _nm(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "nomorals", "inbox", *args],
            capture_output=True, text=True, timeout=180,
            cwd="/home/hatch/workspace/devon", env=self.env)

    def test_cli_add_link_and_list(self):
        proc = self._nm("add-link", "https://example.com/x",
                        "--room", "trading")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("trading", proc.stdout)
        proc = self._nm("list", "--room", "trading", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn('"room": "trading"', proc.stdout)
        self.assertIn('"kind": "link"', proc.stdout)

    def test_cli_sweep(self):
        proc = self._nm("sweep", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("counts", proc.stdout)


if __name__ == "__main__":
    unittest.main()
