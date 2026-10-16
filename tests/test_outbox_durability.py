"""Outbox durability tests (G2): no silent drop, receipts, ordering, isolation."""

import time
import unittest
from types import SimpleNamespace

from nomorals.social.chat.base import ChatRef, SendResult
from nomorals.social.chat.outbox import Outbox, parse_since
from nomorals.storage.db import Database


def make_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class FakeSend:
    """A scripted send function: per-text failure scripts + a ledger."""

    def __init__(self) -> None:
        self.calls: list[str] = []  # every attempt, in order
        self.sent: list[str] = []  # successful deliveries, in order
        self.failures: dict[str, list[str]] = {}  # text -> errors to return first
        self.always: dict[str, str] = {}  # chat_key -> error to always return

    def __call__(self, payload: dict) -> SendResult:
        text = str(payload.get("text") or "")
        chat_key = str(payload.get("chat_key") or "")
        self.calls.append(text)
        if chat_key in self.always:
            return SendResult(ok=False, platform="test",
                              error=self.always[chat_key])
        script = self.failures.get(text, [])
        if script:
            return SendResult(ok=False, platform="test",
                              error=script.pop(0))
        self.sent.append(text)
        return SendResult(ok=True, platform="test",
                          message_id=f"m-{len(self.sent)}")


def _topic(chat: ChatRef) -> str:
    return f"chat_outbox:test:{chat.key}"


class TestOutboxDurability(unittest.TestCase):
    def setUp(self) -> None:
        self.db = make_db()
        self.outbox = Outbox(self.db)
        self.chat = ChatRef(platform="test", chat_id="1")

    def test_submit_and_drain_delivers_and_receipts(self) -> None:
        fake = FakeSend()
        job_id = self.outbox.submit(platform="test", chat=self.chat, text="hi")
        outcomes = self.outbox.deliver_chat(self.chat.key, "test", fake)
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["ok"])
        self.assertEqual(fake.sent, ["hi"])
        rows = self.outbox.receipts(chat_key=self.chat.key)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], job_id)
        self.assertEqual(rows[0]["state"], "sent")
        self.assertEqual(rows[0]["message_id"], "m-1")
        self.assertGreaterEqual(rows[0]["attempts"], 1)

    def test_delivered_means_receipted(self) -> None:
        fake = FakeSend()
        job_id = self.outbox.submit(platform="test", chat=self.chat, text="hi")
        self.outbox.deliver_chat(self.chat.key, "test", fake)
        ids = [r["id"] for r in self.outbox.receipts(chat_key=self.chat.key)]
        self.assertIn(job_id, ids)

    def test_killed_worker_reclaim_redelivers(self) -> None:
        fake = FakeSend()
        job_id = self.outbox.submit(platform="test", chat=self.chat, text="hi")
        # A worker leases the job and dies before sending (lease expires).
        leased = self.outbox.queue.lease(_topic(self.chat), "dead-worker",
                                         lease_seconds=0, batch=1)
        self.assertEqual(len(leased), 1)
        self.assertEqual(leased[0].id, job_id)
        self.assertEqual(fake.sent, [])
        outcomes = self.outbox.deliver_chat(self.chat.key, "test", fake)
        self.assertTrue(outcomes[0]["ok"])
        self.assertEqual(fake.sent, ["hi"])
        self.assertEqual(
            self.outbox.receipts(chat_key=self.chat.key)[0]["state"], "sent")

    def test_idempotent_resubmit(self) -> None:
        fake = FakeSend()
        first = self.outbox.submit(platform="test", chat=self.chat, text="hi",
                                   idempotency_key="k1")
        second = self.outbox.submit(platform="test", chat=self.chat, text="hi",
                                    idempotency_key="k1")
        self.assertEqual(first, second)
        self.outbox.deliver_chat(self.chat.key, "test", fake)
        self.assertEqual(fake.sent, ["hi"])
        self.assertEqual(len(self.outbox.receipts(chat_key=self.chat.key)), 1)
        # Even after completion, a repeat submit returns the same row.
        third = self.outbox.submit(platform="test", chat=self.chat, text="hi",
                                   idempotency_key="k1")
        self.assertEqual(first, third)
        self.outbox.deliver_chat(self.chat.key, "test", fake)
        self.assertEqual(fake.sent, ["hi"])

    def test_ordering_survives_retry(self) -> None:
        fake = FakeSend()
        fake.failures["1"] = ["503 Service Unavailable"]
        for text in ("1", "2", "3"):
            self.outbox.submit(platform="test", chat=self.chat, text=text)
        # First drain: head fails transiently; later jobs must NOT overtake.
        self.outbox.deliver_chat(self.chat.key, "test", fake)
        self.assertEqual(fake.calls, ["1"])
        self.assertEqual(fake.sent, [])
        # Backoff passes; second drain resumes in order.
        self.db.execute("UPDATE work_queue SET available_at = 0")
        self.outbox.deliver_chat(self.chat.key, "test", fake)
        self.assertEqual(fake.sent, ["1", "2", "3"])
        self.assertEqual(fake.calls, ["1", "1", "2", "3"])

    def test_head_leased_elsewhere_blocks_chat(self) -> None:
        fake = FakeSend()
        for text in ("1", "2", "3"):
            self.outbox.submit(platform="test", chat=self.chat, text=text)
        leased = self.outbox.queue.lease(_topic(self.chat), "other-worker",
                                         batch=1)
        self.assertEqual(len(leased), 1)
        self.outbox.drain_all(fake)
        self.assertEqual(fake.calls, [])  # head owned elsewhere: no jumping ahead
        self.outbox.queue.requeue(leased[0].id)
        self.outbox.drain_all(fake)
        self.assertEqual(fake.sent, ["1", "2", "3"])

    def test_terminal_never_retried_at_outbox_level(self) -> None:
        fake = FakeSend()
        fake.always[self.chat.key] = "400 chat not found"
        job_id = self.outbox.submit(platform="test", chat=self.chat, text="hi")
        outcomes = self.outbox.deliver_chat(self.chat.key, "test", fake)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["state"], "failed_terminal")
        self.assertEqual(fake.calls, ["hi"])  # exactly one attempt
        job = self.outbox.queue.get(job_id)
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.status, "dead")
        rows = self.outbox.receipts(chat_key=self.chat.key)
        self.assertEqual(rows[0]["state"], "failed_terminal")
        self.assertIn("400", rows[0]["error"])

    def test_dead_platform_cannot_jam_queue(self) -> None:
        fake = FakeSend()
        chat_a = ChatRef(platform="test", chat_id="A")
        chat_b = ChatRef(platform="test", chat_id="B")
        fake.always[chat_a.key] = "400 chat not found"
        self.outbox.submit(platform="test", chat=chat_a, text="to-A")
        self.outbox.submit(platform="test", chat=chat_b, text="to-B")
        results = self.outbox.drain_all(fake)
        self.assertEqual(len(results), 2)
        self.assertEqual(fake.sent, ["to-B"])
        rows_a = self.outbox.receipts(chat_key=chat_a.key)
        rows_b = self.outbox.receipts(chat_key=chat_b.key)
        self.assertEqual(rows_a[0]["state"], "failed_terminal")
        self.assertEqual(rows_b[0]["state"], "sent")

    def test_receipts_filters(self) -> None:
        fake = FakeSend()
        self.outbox.submit(platform="test", chat=self.chat, text="hi")
        self.outbox.deliver_chat(self.chat.key, "test", fake)
        self.assertEqual(len(self.outbox.receipts(chat_key="test:1")), 1)
        self.assertEqual(len(self.outbox.receipts(chat_key="1")), 1)  # bare id
        self.assertEqual(len(self.outbox.receipts(chat_key="test:2")), 0)
        self.assertEqual(
            len(self.outbox.receipts(chat_key=self.chat.key,
                                     since=time.time() + 1000)), 0)
        self.assertEqual(
            len(self.outbox.receipts(chat_key=self.chat.key, since=0)), 1)


class TestParseSince(unittest.TestCase):
    def test_relative(self) -> None:
        now = time.time()
        self.assertAlmostEqual(parse_since("1h"), now - 3600, delta=5)
        self.assertAlmostEqual(parse_since("30m"), now - 1800, delta=5)
        self.assertAlmostEqual(parse_since("7d"), now - 604800, delta=5)

    def test_iso_and_epoch(self) -> None:
        self.assertGreater(parse_since("2026-10-16"), 0)
        self.assertGreater(parse_since("2026-10-16T10:00:00"), 0)
        self.assertEqual(parse_since(12345), 12345.0)
        self.assertEqual(parse_since("12345"), 12345.0)

    def test_garbage_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_since("bogus")


if __name__ == "__main__":
    unittest.main()
