"""Tests for the chat gateway: simultaneity, ordering, rate limits, dry-run."""

from __future__ import annotations

import threading
import time
import unittest

from nomorals.social.chat.base import ChatAdapter, ChatKind, ChatMessage, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway, _HourWindow
from nomorals.storage.db import Database


class FakeAdapter(ChatAdapter):
    """In-memory stand-in for a real platform adapter."""

    def __init__(self, name: str = "fake", send_delay: float = 0.0) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, str]] = []  # (chat.key, text)
        self.send_delay = send_delay
        self._handler = None
        self.started = threading.Event()

    def run(self, handler) -> None:
        self._handler = handler
        self.started.set()
        while not self.stopped:
            time.sleep(0.01)

    def push(self, chat: ChatRef, text: str, *, sender: str = "someone") -> None:
        assert self._handler is not None, "adapter not started"
        self._deliver(self._handler, ChatMessage(chat=chat, incoming=True, text=text, sender=sender))

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        if self.send_delay:
            time.sleep(self.send_delay)
        self.sent.append((chat.key, text))
        return SendResult(ok=True, platform=self.name, message_id=f"msg-{len(self.sent)}")

    def wait_started(self, timeout: float = 5.0) -> bool:
        return self.started.wait(timeout)


def _dm(platform: str, chat_id: str = "1") -> ChatRef:
    return ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM, peer="someone")


class GatewayTest(unittest.TestCase):
    def _gateway(self, adapters: dict[str, ChatAdapter], **kw) -> ChatGateway:
        db = Database(":memory:")
        db.migrate()
        return ChatGateway(adapters, db=db, **kw)

    def test_all_adapters_run_simultaneously(self) -> None:
        a, b = FakeAdapter("a"), FakeAdapter("b")
        received: list[str] = []
        gw = self._gateway({"a": a, "b": b})
        started = gw.start(lambda m: received.append(m.chat.platform))
        self.assertEqual(sorted(started), ["a", "b"])
        self.assertTrue(a.wait_started() and b.wait_started())
        a.push(_dm("a"), "hello from a")
        b.push(_dm("b"), "hello from b")
        time.sleep(0.05)
        self.assertEqual(sorted(received), ["a", "b"])
        gw.stop()

    def test_inbound_registers_chat_in_db(self) -> None:
        a = FakeAdapter("a")
        gw = self._gateway({"a": a}, owner_chats={"a:1"}, us_chats={"a:2"})
        gw.start(lambda m: None)
        self.assertTrue(a.wait_started())
        a.push(_dm("a", "1"), "hi")
        a.push(_dm("a", "2"), "hi from the states")
        time.sleep(0.05)
        row1 = gw.db.query_one("SELECT * FROM chats WHERE id = 'a:1'")
        row2 = gw.db.query_one("SELECT * FROM chats WHERE id = 'a:2'")
        self.assertEqual(int(row1["is_owner"]), 1)
        self.assertEqual(int(row1["in_us"]), 0)
        self.assertEqual(int(row2["in_us"]), 1)
        self.assertGreater(row1["last_active"], 0)
        gw.stop()

    def test_send_routes_to_correct_adapter(self) -> None:
        a, b = FakeAdapter("a"), FakeAdapter("b")
        gw = self._gateway({"a": a, "b": b})
        gw.start(lambda m: None)
        result = gw.send("a", _dm("a"), "going out")
        self.assertTrue(result.ok)
        result2 = gw.send("b", _dm("b"), "other side")
        self.assertTrue(result2.ok)
        self.assertEqual(a.sent, [(_dm("a").key, "going out")])
        self.assertEqual(b.sent, [(_dm("b").key, "other side")])
        gw.stop()

    def test_send_unknown_platform_fails_cleanly(self) -> None:
        a = FakeAdapter("a")
        gw = self._gateway({"a": a})
        result = gw.send("nope", _dm("nope"), "hi")
        self.assertFalse(result.ok)
        self.assertIn("no adapter", result.error)

    def test_dry_run_never_reaches_adapter(self) -> None:
        a = FakeAdapter("a")
        gw = self._gateway({"a": a}, dry_run=True)
        result = gw.send("a", _dm("a"), "should not send")
        self.assertTrue(result.ok)
        self.assertEqual(a.sent, [])
        self.assertEqual(gw.stats["dry_run_sends"], 1)

    def test_rate_limit_drops_inbound(self) -> None:
        a = FakeAdapter("a")
        received: list[str] = []
        gw = self._gateway({"a": a}, max_per_hour=3)
        gw.start(lambda m: received.append(m.text))
        self.assertTrue(a.wait_started())
        for i in range(5):
            a.push(_dm("a"), f"msg {i}")
        time.sleep(0.05)
        self.assertEqual(len(received), 3)
        self.assertEqual(gw.stats["dropped_rate_limited"], 2)
        gw.stop()

    def test_per_chat_send_ordering_under_concurrency(self) -> None:
        a = FakeAdapter("a", send_delay=0.002)
        gw = self._gateway({"a": a})
        n_per_chat = 8

        def worker(chat_id: str) -> None:
            for i in range(n_per_chat):
                gw.send("a", _dm("a", chat_id), f"{chat_id}-{i}")

        threads = [threading.Thread(target=worker, args=(cid,)) for cid in ("x", "y")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for chat_id in ("x", "y"):
            texts = [text for key, text in a.sent if key == _dm("a", chat_id).key]
            self.assertEqual(
                texts, [f"{chat_id}-{i}" for i in range(n_per_chat)],
                f"chat {chat_id} ordering broken",
            )

    def test_empty_text_is_skipped_not_sent(self) -> None:
        a = FakeAdapter("a")
        gw = self._gateway({"a": a})
        result = gw.send("a", _dm("a"), "   ")
        self.assertTrue(result.ok)
        self.assertEqual(a.sent, [])

    def test_history_and_typing_fall_back_gracefully(self) -> None:
        a = FakeAdapter("a")
        gw = self._gateway({"a": a})
        self.assertEqual(gw.history("missing", _dm("missing"), 5), [])
        self.assertFalse(gw.typing("missing", _dm("missing"), 1.0))


class HourWindowTest(unittest.TestCase):
    def test_window_counts(self) -> None:
        window = _HourWindow(2)
        now = 1_000_000.0
        self.assertTrue(window.allow(now))
        self.assertTrue(window.allow(now + 1))
        self.assertFalse(window.allow(now + 2))
        # One hour later the slot frees up.
        self.assertTrue(window.allow(now + 3700))

    def test_pending_reflects_window(self) -> None:
        window = _HourWindow(10)
        now = 1_000_000.0
        for i in range(4):
            window.allow(now + i)
        self.assertEqual(window.pending(now + 10), 4)
        self.assertEqual(window.pending(now + 3700), 0)

    def test_zero_limit_is_unlimited(self) -> None:
        window = _HourWindow(0)
        now = 1_000_000.0
        for i in range(1000):
            self.assertTrue(window.allow(now + i))
        self.assertEqual(window.pending(now + 10), 0)


class GatewayPreflightTest(unittest.TestCase):
    """One-time interactive logins (Telegram first-run) must run on the main
    thread BEFORE any adapter thread — the console otherwise eats the
    keyboard and the login prompt is unreachable."""

    def _gateway(self, adapters: dict[str, ChatAdapter]) -> ChatGateway:
        db = Database(":memory:")
        db.migrate()
        return ChatGateway(adapters, db=db)

    def test_preflight_runs_before_any_thread_starts(self) -> None:
        order: list[str] = []

        class PreflightAdapter(FakeAdapter):
            def preflight(self) -> None:
                order.append(f"preflight-{self.name}")

            def run(self, handler) -> None:  # noqa: D401
                order.append(f"run-{self.name}")
                super().run(handler)

        a, b = PreflightAdapter("a"), PreflightAdapter("b")
        gw = self._gateway({"a": a, "b": b})
        try:
            started = gw.start(lambda m: None)
            self.assertEqual(sorted(started), ["a", "b"])
            self.assertLess(order.index("preflight-a"), order.index("run-a"))
            self.assertLess(order.index("preflight-b"), order.index("run-b"))
        finally:
            gw.stop()

    def test_failed_preflight_removes_adapter_keeps_the_rest(self) -> None:
        class BrokenPreflight(FakeAdapter):
            def preflight(self) -> None:
                raise RuntimeError("login failed")

        good = FakeAdapter("good")
        gw = self._gateway({"bad": BrokenPreflight("bad"), "good": good})
        try:
            started = gw.start(lambda m: None)
            self.assertEqual(started, ["good"])
        finally:
            gw.stop()

    def test_no_preflight_needed_is_fine(self) -> None:
        a = FakeAdapter("a")  # base no-op
        gw = self._gateway({"a": a})
        try:
            self.assertEqual(gw.start(lambda m: None), ["a"])
        finally:
            gw.stop()


if __name__ == "__main__":
    unittest.main()
