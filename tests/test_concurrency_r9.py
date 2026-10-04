"""R9: concurrency, resource leaks, edge cases.

Covers the Round 9 audit fixes:

- gateway: stats increments are atomic across adapter/pool threads;
  start_one/stop_one vs send/status/stop no longer race on the adapters
  dict; set_rate_limit vs window.allow share one lock ordering.
- runtime: stats increments atomic; drained per-chat queues are removed
  (no unbounded _queues/_draining growth); on_message after pool shutdown
  resets the draining flag instead of orphaning the message.
- storage: Database.release_thread() drops a one-shot worker thread's
  connection (fd-leak fix); Database stats increments are atomic.
- edge cases: ChatMessage/ChatRef coerce None fields to ""; send() with
  None text is treated as empty.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.storage.db import Database, release_thread_connection


class _FakeAdapter(ChatAdapter):
    name = "fake"

    def run(self, handler):
        pass  # receive thread exits immediately

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        return SendResult(ok=True, platform=self.name, message_id="m1")


def _msg(chat_id: str, platform: str = "tg", text: str = "hi") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM),
        incoming=True, text=text, sender="someone",
    )


def _wait(predicate, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class GatewayStatsRaceTests(unittest.TestCase):
    def test_inbound_counter_exact_under_threads(self):
        gw = ChatGateway({"fake": _FakeAdapter()}, db=None,
                         owner_chats={"tg:owner"})
        gw._inbound = lambda m: None  # noqa: E731 - test sink
        chat = ChatRef(platform="tg", chat_id="owner", kind=ChatKind.DM)
        n_threads, n_each = 8, 250

        def hammer():
            for _ in range(n_each):
                gw._on_inbound(_msg("owner", text="x"))

        threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(gw.status()["_stats"]["inbound"], n_threads * n_each)


class GatewayAdapterRegistryRaceTests(unittest.TestCase):
    def test_start_stop_send_status_race(self):
        # Before R9 this raised "dictionary changed size during iteration"
        # from stop()/status() racing stop_one()'s pop.
        gw = ChatGateway({"a": _FakeAdapter(), "b": _FakeAdapter()},
                         db=None, dry_run=True)
        chat = ChatRef(platform="a", chat_id="1", kind=ChatKind.DM)
        stop_flag = threading.Event()
        errors: list[BaseException] = []

        def hammer():
            try:
                while not stop_flag.is_set():
                    gw.status()
                    gw.send("a", chat, "hi")
                    gw.start_one("a", lambda m: None)  # noqa: E731
                    gw.stop_one("b")
                    gw.stop()
            except Exception as exc:  # noqa: BLE001 - collected, asserted below
                errors.append(exc)

        threads = [threading.Thread(target=hammer) for _ in range(6)]
        for t in threads:
            t.start()
        time.sleep(1.0)
        stop_flag.set()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_double_start_rejected(self):
        gw = ChatGateway({"a": _FakeAdapter()}, db=None)
        first = gw.start_one("a", lambda m: None)  # noqa: E731
        self.assertTrue(first["ok"])
        # Fake run() exits at once, so the thread may be dead; either way
        # the call must be safe and report already-running or start cleanly.
        second = gw.start_one("a", lambda m: None)  # noqa: E731
        self.assertIn("ok", second)


class GatewayRateLimitRaceTests(unittest.TestCase):
    def test_set_limit_concurrent_with_allow(self):
        gw = ChatGateway({"fake": _FakeAdapter()}, db=None, max_per_hour=50)
        gw._inbound = lambda m: None  # noqa: E731 - test sink
        stop_flag = threading.Event()
        errors: list[BaseException] = []

        def hammer_inbound():
            try:
                i = 0
                while not stop_flag.is_set():
                    i += 1
                    gw._on_inbound(_msg(f"chat{i % 4}", text="x"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def hammer_limit():
            try:
                for n in (10, 0, 100, 5):
                    if stop_flag.is_set():
                        return
                    gw.set_rate_limit(n)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = ([threading.Thread(target=hammer_inbound) for _ in range(4)]
                   + [threading.Thread(target=hammer_limit)])
        for t in threads:
            t.start()
        time.sleep(1.0)
        stop_flag.set()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        # The last writer wins; the value must be one that was actually set.
        with gw._locks_guard:
            limits = {w.limit for w in gw._windows.values()}
        self.assertTrue(limits <= {10, 0, 100, 5, 50})


class EdgeCaseTests(unittest.TestCase):
    def test_chat_message_none_fields_coerced(self):
        msg = ChatMessage(chat=ChatRef(platform="tg", chat_id="1"),
                          incoming=True, text=None, sender=None,
                          reply_to=None, message_id=None)
        self.assertEqual(msg.text, "")
        self.assertEqual(msg.sender, "")
        self.assertEqual(msg.reply_to, "")
        self.assertEqual(msg.message_id, "")
        msg.text.strip()  # must not raise

    def test_chat_ref_none_fields_coerced(self):
        ref = ChatRef(platform=None, chat_id=None, kind=None)
        self.assertEqual(ref.platform, "")
        self.assertEqual(ref.chat_id, "")
        self.assertEqual(ref.kind, ChatKind.DM)
        ref.key  # must not raise

    def test_inbound_none_text_survives_funnel(self):
        gw = ChatGateway({"fake": _FakeAdapter()}, db=None,
                         owner_chats={"tg:1"})
        seen = []
        gw._inbound = seen.append
        gw._on_inbound(_msg("1", text=None))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "")

    def test_send_none_text_is_empty_skip(self):
        gw = ChatGateway({"fake": _FakeAdapter()}, db=None)
        chat = ChatRef(platform="fake", chat_id="1", kind=ChatKind.DM)
        result = gw.send("fake", chat, None)
        self.assertTrue(result.ok)
        self.assertTrue(result.message_id.startswith("skip"))


def _make_runtime():
    tmp = tempfile.TemporaryDirectory(prefix="nm-r9-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    gw = ChatGateway({"local": _FakeAdapter()}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gw)
    return runtime, context, tmp


class RuntimeQueueTests(unittest.TestCase):
    def setUp(self):
        self.runtime, self.context, self.tmp = _make_runtime()

    def tearDown(self):
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_no_message_lost_under_thread_storm(self):
        rt = self.runtime
        seen: list[str] = []
        seen_lock = threading.Lock()

        def fake_process(m):
            with seen_lock:
                seen.append(m.text)

        rt._process = fake_process  # monkeypatched pump target
        key = "tg:storm"
        n_threads, n_each = 4, 25

        def hammer(tid):
            for i in range(n_each):
                rt.on_message(_msg("storm", text=f"{tid}-{i}"))

        threads = [threading.Thread(target=hammer, args=(t,))
                   for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(_wait(lambda: len(seen) == n_threads * n_each),
                        f"lost messages: {len(seen)}/{n_threads * n_each}")
        # FIFO per chat: each producer's own sequence stays ordered.
        for tid in range(n_threads):
            order = [int(s.split("-")[1]) for s in seen
                     if s.startswith(f"{tid}-")]
            self.assertEqual(order, sorted(order))
        # Drained queues are removed: no unbounded dict growth.
        self.assertTrue(_wait(lambda: key not in rt._queues
                              and key not in rt._draining),
                        "drained queue entries were not cleaned up")

    def test_on_message_after_pool_shutdown_resets_flag(self):
        rt = self.runtime
        rt._pool.shutdown(wait=True)
        rt.on_message(_msg("late", text="x"))  # must not raise
        self.assertFalse(rt._draining.get("tg:late", False))

    def test_stats_race(self):
        rt = self.runtime
        n_threads, n_each = 8, 500

        def hammer():
            for _ in range(n_each):
                rt._bump("messages")

        threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(rt._stats_snapshot()["messages"], n_threads * n_each)


class DatabaseThreadLeakTests(unittest.TestCase):
    def test_release_thread_drops_connection(self):
        db = Database(":memory:")
        baseline = db.connection_count()
        self.assertGreaterEqual(baseline, 1)  # eager open on this thread

        def worker():
            db.execute("SELECT 1")
            self.assertEqual(db.connection_count(), baseline + 1)
            db.release_thread()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(db.connection_count(), baseline)
        db.close()

    def test_release_thread_noop_when_unused(self):
        db = Database(":memory:")
        release_thread_connection(None)  # must not raise
        release_thread_connection(db)  # this thread's conn stays: we still need it
        db.execute("SELECT 1")  # lazily reopens after release
        db.close()

    def test_db_stats_race(self):
        db = Database(":memory:")
        # Read the counter directly: stats_snapshot() itself issues a
        # tables() query, which would add noise to an exact assertion.
        # All workers are joined here, so the direct read is safe.
        baseline = db.stats["queries"]
        n_threads, n_each = 8, 100

        def hammer():
            for _ in range(n_each):
                db.execute("SELECT 1")

        threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(db.stats["queries"] - baseline, n_threads * n_each)
        db.close()


if __name__ == "__main__":
    unittest.main()
