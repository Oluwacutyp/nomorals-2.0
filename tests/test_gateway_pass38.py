"""Pass 38 — gateway + message delivery layer.

Three defects, each reproduced on pristine HEAD before the fix:

1. **Held-quiet-hours rows were delivered during quiet hours.**
   Producers that defer during quiet hours (missions progress, kind
   ``"mission"``) store rows as ``held-quiet-hours`` for
   :meth:`Notifier.redeliver` to send "when quiet ends".  The
   re-check consulted :func:`proactive_gate`, which only governs the
   ``briefing``/``watcher`` kinds (and exempts both from quiet hours),
   so for any other kind it returned ``None`` and the row went out on
   the first redelivery sweep — mid-quiet-hours.  The re-check is a
   clock question, not a per-kind gate question.

2. **Two Outbox instances over one DB could silently drop a submit.**
   Auto-generated idempotency keys were ``seq:<platform>:<chat>:<n>``
   where ``n`` comes from a per-instance scan of the topic.  Two
   instances that both scanned before either submitted the same ``n``
   produced the same key; :meth:`WorkQueue.enqueue` then returned the
   first job's id for the second submit — the second message was never
   enqueued, and its caller held a receipt for someone else's text.

3. **Chunk budgets counted code points; Telegram counts UTF-16 units.**
   A 4096-code-point chunk of astral emoji is 8192 UTF-16 code units —
   Telegram rejects it ("message too long", a terminal class here) and
   the whole send fails.  Platforms whose wire limit is UTF-16-based
   (the Telegram family, Discord, WhatsApp — all UTF-16 stacks) now
   declare ``utf16_lengths`` in their PlatformLimits row and chunk by
   UTF-16 width; other platforms keep code-point budgeting.

Plus a pin for the outbox crash contract: a job leased by a worker
that dies mid-delivery is redelivered after lease expiry
(at-least-once) — never silently lost; a completed job is never
redelivered.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import string
import tempfile
import time
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from nomorals.agents.notifier import (
    DELIVERY_MAX_ATTEMPTS,
    STATE_DEAD,
    STATE_DISABLED,
    STATE_HELD_QUIET,
    Notifier,
)
from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
from nomorals.social.chat.chunking import (
    chunk_for_platform,
    label_chunks,
    split_caption,
    strip_label,
    utf16_len,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.outbox import Outbox
from nomorals.storage.db import Database


def _tmp_db(test: unittest.TestCase) -> tuple[str, Database]:
    tmp = tempfile.mkdtemp(prefix="pass38-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    return tmp, db


# ── 1. held-quiet-hours redelivery ─────────────────────────────────────

class _SendResult:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.message_id = "m1"
        self.error = ""


class _FakeGateway:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def status(self):
        return {"telegram": {"running_in_session": True}}

    def send(self, platform, chat_ref, text, **kw):
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(True)


def _partner(*, quiet: bool, **kw):
    """Partner settings with a quiet window that is deterministically
    in effect (window straddling the current UTC hour) or never in
    effect (start == end)."""
    hour = datetime.now(timezone.utc).hour
    if quiet:
        start, end = (hour + 23) % 24, (hour + 1) % 24
    else:
        start = end = hour
    base = dict(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=start,
        quiet_end=end,
        timezone="UTC",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _notifier(test, *, quiet: bool, **partner_kw):
    tmp, db = _tmp_db(test)
    gw = _FakeGateway()
    partner = _partner(quiet=quiet, **partner_kw)
    settings = SimpleNamespace(workspace_dir=tmp, partner=partner)
    ctx = SimpleNamespace(db=db, settings=settings, extras={"gateway": gw})
    return Notifier(ctx), gw, db, partner


class HeldQuietRedeliveryTests(unittest.TestCase):
    def test_mission_row_stays_held_during_quiet_hours(self):
        n, gw, db, _ = _notifier(self, quiet=True)
        # exactly what missions/progress.py stores during quiet hours
        n._store("mission", "mission update: X — 50%", "body", 0,
                 delivery_state=STATE_HELD_QUIET)
        self.assertEqual(n.redeliver(), 0)
        self.assertEqual(gw.sent, [])
        row = db.query_one("SELECT delivered, delivery_state FROM notifications")
        self.assertEqual(row["delivered"], 0)
        self.assertEqual(row["delivery_state"], STATE_HELD_QUIET)

    def test_mission_row_delivered_once_quiet_ends(self):
        n, gw, db, partner = _notifier(self, quiet=True)
        n._store("mission", "mission update: X — 50%", "body", 0,
                 delivery_state=STATE_HELD_QUIET)
        self.assertEqual(n.redeliver(), 0)
        # quiet hours end (window collapses: start == end -> never quiet)
        hour = datetime.now(timezone.utc).hour
        partner.quiet_start = partner.quiet_end = hour
        self.assertEqual(n.redeliver(), 1)
        self.assertEqual(len(gw.sent), 1)
        row = db.query_one("SELECT delivered, delivery_state FROM notifications")
        self.assertEqual(row["delivered"], 1)
        self.assertEqual(row["delivery_state"], "sent")

    def test_held_row_of_disabled_kind_is_parked_not_sent(self):
        n, gw, db, partner = _notifier(self, quiet=False)
        n._store("briefing", "morning briefing", "body", 0,
                 delivery_state=STATE_HELD_QUIET)
        # owner turns proactive off while the row is held
        partner.proactive_enabled = False
        self.assertEqual(n.redeliver(), 0)
        self.assertEqual(gw.sent, [])
        row = db.query_one("SELECT delivered, delivery_state FROM notifications")
        self.assertEqual(row["delivered"], 0)
        self.assertEqual(row["delivery_state"], STATE_DISABLED)
        # and redelivery never picks it up again
        self.assertEqual(n.redeliver(), 0)
        self.assertEqual(gw.sent, [])


class _FailingGateway:
    def __init__(self) -> None:
        self.calls = 0

    def status(self):
        return {"telegram": {"running_in_session": True}}

    def send(self, platform, chat_ref, text, **kw):
        self.calls += 1
        return _SendResult(False)


class HeldRowRetryCapTests(unittest.TestCase):
    """A held row whose post-hold delivery keeps failing must back off
    and eventually dead-letter — on pristine code the held branch of
    the due query had no backoff/cap and the attempt was never
    consumed, so a dead channel was retried on every sweep forever."""

    def test_failed_post_hold_delivery_backs_off_and_dead_letters(self):
        tmp, db = _tmp_db(self)
        gw = _FailingGateway()
        partner = _partner(quiet=False)
        settings = SimpleNamespace(workspace_dir=tmp, partner=partner)
        ctx = SimpleNamespace(db=db, settings=settings,
                              extras={"gateway": gw})
        n = Notifier(ctx)
        n._store("mission", "mission update", "body", 0,
                 delivery_state=STATE_HELD_QUIET)
        self.assertEqual(n.redeliver(), 0)
        self.assertEqual(gw.calls, 1)
        # backoff honoured: an immediate second sweep does not retry
        self.assertEqual(n.redeliver(), 0)
        self.assertEqual(gw.calls, 1)
        # simulate each backoff elapsing; attempts burn to the cap
        for _ in range(DELIVERY_MAX_ATTEMPTS + 2):
            db.execute("UPDATE notifications SET next_retry_at = 0")
            n.redeliver()
        row = db.query_one(
            "SELECT retry_count, delivery_state, delivered "
            "FROM notifications")
        self.assertEqual(row["delivered"], 0)
        self.assertEqual(row["delivery_state"], STATE_DEAD)
        self.assertGreaterEqual(row["retry_count"], DELIVERY_MAX_ATTEMPTS)
        # dead-lettered: never attempted again
        calls_at_dead = gw.calls
        db.execute("UPDATE notifications SET next_retry_at = 0")
        n.redeliver()
        self.assertEqual(gw.calls, calls_at_dead)


# ── 2. outbox cross-instance submit collision ──────────────────────────

class OutboxCrossInstanceTests(unittest.TestCase):
    def test_interleaved_instances_never_drop_a_submit(self):
        _tmp, db = _tmp_db(self)
        chat = ChatRef(platform="telegram", chat_id="111")
        a, b = Outbox(db), Outbox(db)
        topic = "chat_outbox:telegram:telegram:111"
        # Both instances scan the topic before either submits that seq —
        # the long-lived two-process shape (bot + a second gateway host).
        a._next_seq(topic)
        b._next_seq(topic)
        ja = a.submit(platform="telegram", chat=chat, text="AAAA")
        jb = b.submit(platform="telegram", chat=chat, text="BBBB")
        self.assertNotEqual(ja, jb)
        rows = db.query("SELECT payload FROM work_queue WHERE topic = ?",
                        (topic,))
        texts = sorted(json.loads(r["payload"])["text"] for r in rows)
        self.assertEqual(texts, ["AAAA", "BBBB"])
        # Both messages deliver, in order, with their own text.
        delivered: list[str] = []

        def send_fn(payload):
            delivered.append(payload["text"])
            return SendResult(ok=True, platform="telegram",
                              message_id="x"), 1

        a.deliver_chat(chat.key, "telegram", send_fn)
        self.assertEqual(delivered, ["AAAA", "BBBB"])

    def test_explicit_idempotency_key_still_dedupes_across_instances(self):
        _tmp, db = _tmp_db(self)
        chat = ChatRef(platform="telegram", chat_id="111")
        a, b = Outbox(db), Outbox(db)
        j1 = a.submit(platform="telegram", chat=chat, text="once",
                      idempotency_key="owner-key-1")
        j2 = b.submit(platform="telegram", chat=chat, text="once",
                      idempotency_key="owner-key-1")
        self.assertEqual(j1, j2)
        rows = db.query("SELECT id FROM work_queue")
        self.assertEqual(len(rows), 1)


# ── 3. UTF-16 chunk budgeting ──────────────────────────────────────────

class Utf16ChunkingTests(unittest.TestCase):
    def test_astral_emoji_chunks_fit_telegram_utf16_cap(self):
        text = "🎉" * 4096  # 4096 code points, 8192 UTF-16 units
        chunks = chunk_for_platform(text, "telegram")
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        for labelled in label_chunks(chunks):
            self.assertLessEqual(utf16_len(labelled), 4096)

    def test_mixed_briefing_text_fits_and_is_lossless(self):
        text = "Status 🎉🎉🎉 update — café ☕ done\n\n" * 300
        self.assertGreater(len(text), 4096)
        chunks = chunk_for_platform(text, "telegram-bot")
        self.assertEqual("".join(chunks), text)
        for labelled in label_chunks(chunks):
            self.assertLessEqual(utf16_len(labelled), 4096)

    def test_bmp_text_chunking_unchanged(self):
        text = "a" * 5000
        chunks = chunk_for_platform(text, "telegram")
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        for c in chunks:
            self.assertLessEqual(len(c), 4096)

    def test_non_utf16_platform_keeps_codepoint_budget(self):
        text = "🎉" * 9000  # local cap is 8000, code-point budgeted
        chunks = chunk_for_platform(text, "local")
        self.assertEqual(len(chunks), 2)
        self.assertEqual("".join(chunks), text)
        for c in chunks:
            self.assertLessEqual(len(c), 8000)

    def test_caption_split_by_utf16_width(self):
        caption = "🎉" * 600  # 1200 UTF-16 units, cap 1024
        pieces = split_caption(caption, max_caption=1024, width=utf16_len)
        self.assertGreater(len(pieces), 1)
        self.assertEqual("".join(pieces), caption)
        for p in pieces:
            self.assertLessEqual(utf16_len(p), 1024)

    def test_fuzz_lossless_and_within_cap(self):
        rng = random.Random(3800)
        alphabet = (
            list(string.ascii_letters + string.digits + " .,!?;:-()[]")
            + ["🎉", "☕", "é", "é", "\n", "\n\n", "—", "…"]
            + ["supercalifragilisticexpialidocious" * 3]
        )
        for _ in range(60):
            text = "".join(rng.choice(alphabet)
                           for _ in range(rng.randint(1, 3000)))
            for platform, cap in (("telegram", 4096), ("discord", 2000),
                                  ("local", 8000)):
                chunks = chunk_for_platform(text, platform)
                self.assertEqual("".join(chunks), text)
                width = utf16_len if platform != "local" else len
                for labelled in label_chunks(chunks):
                    self.assertLessEqual(width(labelled), cap)


class _RecordingAdapter(ChatAdapter):
    name = "telegram-bot"

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.media: list[tuple[str, str]] = []

    def run(self, handler):  # pragma: no cover - unused
        pass

    def send(self, chat, text, **kw):
        self.texts.append(text)
        return SendResult(ok=True, platform=self.name, message_id="t1")

    def send_media(self, chat, media, *, caption=""):
        self.media.append((media.path, caption))
        return SendResult(ok=True, platform=self.name, message_id="m1")


class GatewayCaptionTests(unittest.TestCase):
    def test_send_file_caption_fits_utf16_and_is_lossless(self):
        tmp, _db = _tmp_db(self)
        path = os.path.join(tmp, "pic.bin")
        with open(path, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        adapter = _RecordingAdapter()
        gw = ChatGateway({"telegram-bot": adapter})
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        caption = "🎉" * 600  # 1200 UTF-16 units > 1024-unit caption cap
        result = gw.send_file("telegram-bot", chat, path, caption=caption)
        self.assertTrue(result.ok)
        self.assertEqual(len(adapter.media), 1)
        first_caption = adapter.media[0][1]
        self.assertLessEqual(utf16_len(first_caption), 1024)
        rest = "".join(strip_label(t) for t in adapter.texts)
        self.assertEqual(first_caption + rest, caption)


# ── 4. outbox crash contract (pin) ─────────────────────────────────────

class OutboxCrashContractTests(unittest.TestCase):
    def test_crashed_workers_job_is_redelivered_not_lost(self):
        _tmp, db = _tmp_db(self)
        chat = ChatRef(platform="telegram", chat_id="111")
        outbox = Outbox(db)
        outbox.submit(platform="telegram", chat=chat, text="hello")
        topic = "chat_outbox:telegram:telegram:111"
        # A worker leases the job and dies mid-delivery: the platform
        # may already have the message, the row never completes.
        leased = outbox.queue.lease(topic, "dying-worker", batch=1)
        self.assertEqual(len(leased), 1)
        db.execute("UPDATE work_queue SET lease_until = ? WHERE id = ?",
                   (time.time() - 1.0, leased[0].id))
        delivered: list[str] = []

        def send_fn(payload):
            delivered.append(payload["text"])
            return SendResult(ok=True, platform="telegram",
                              message_id="x"), 1

        outcomes = outbox.deliver_chat(chat.key, "telegram", send_fn)
        self.assertEqual(delivered, ["hello"])  # redelivered (at-least-once)
        self.assertTrue(outcomes and outcomes[0]["ok"])
        # Completed now: a second drain sends nothing more.
        outcomes2 = outbox.deliver_chat(chat.key, "telegram", send_fn)
        self.assertEqual(outcomes2, [])
        self.assertEqual(delivered, ["hello"])


if __name__ == "__main__":
    unittest.main()
