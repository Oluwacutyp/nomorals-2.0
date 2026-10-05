"""Shutdown ordering: the chat pool must drain before the Database closes.

Regression tests for the phone's ``StorageError("database is closed")``:
``PartnerRuntime.stop()`` used to do ``self._pool.shutdown(wait=False)``
and the gateway's ``with build_context(...)`` then closed the Database
while a chat-pool worker was still inside ``_generate_and_persist`` — its
persist tail (``relationship.save`` -> ``db.transaction``) blew up and the
reply was dropped.

Two layers are verified here:

1. ``stop()`` drains in-flight chat handlers (bounded) before the pool
   shuts down and the context closes.
2. The brain's post-reply persist tail survives a DB that closes mid-tail
   (the reply is still delivered; the bookkeeping loss is logged).
"""

from __future__ import annotations

import random
import tempfile
import threading
import time
import unittest
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerBrain, PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.llm.base import LLMResponse, Message, SamplingParams
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway


class FakeRouter:
    """A stand-in LLM router with a scripted reply."""

    def chat(self, messages: list[Message], params: SamplingParams | None = None,
             **kw: Any) -> LLMResponse:
        return LLMResponse(text="mhm. i was just thinking about you, actually",
                           model="fake-7b")


class FakeAdapter(ChatAdapter):
    def __init__(self, name: str = "local") -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self._handler = None

    def run(self, handler) -> None:
        self._handler = handler
        while not self.stopped:
            time.sleep(0.01)

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "") -> SendResult:
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name, message_id=f"m{len(self.sent)}")


def _make_runtime() -> tuple[PartnerRuntime, Any, tempfile.TemporaryDirectory,
                             ChatRef, FakeAdapter]:
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-drain-")
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    context.router = FakeRouter()
    adapter = FakeAdapter("local")
    gateway = ChatGateway({"local": adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)  # deterministic: no busy gaps
    chat = ChatRef(platform="local", chat_id="console", kind=ChatKind.DM, peer="you")
    return runtime, context, tmp, chat, adapter


class ShutdownDrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime, self.context, self.tmp, self.chat, self.adapter = _make_runtime()
        self.brain: PartnerBrain = self.runtime.brain
        self._blockers: list[threading.Event] = []

    def tearDown(self) -> None:
        for blocker in self._blockers:
            blocker.set()
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        try:
            self.context.close()
        except Exception:  # noqa: BLE001
            pass
        self.tmp.cleanup()

    def _slow_work(self, release: threading.Event, marker: dict) -> None:
        release.wait(30)
        marker["done_at"] = time.time()

    def test_drain_waits_for_inflight_work(self) -> None:
        """_drain_chat_pool blocks until a tracked future finishes."""
        release = threading.Event()
        marker: dict = {}
        fut = self.runtime._pool.submit(self._slow_work, release, marker)
        self.runtime._track_inflight(fut)
        threading.Timer(0.4, release.set).start()

        t0 = time.time()
        self.runtime._drain_chat_pool(timeout=5.0)
        elapsed = time.time() - t0

        self.assertTrue(fut.done())
        self.assertGreater(elapsed, 0.3, "drain returned before the work finished")
        self.assertLess(elapsed, 5.0, "drain ignored its timeout")
        with self.runtime._inflight_guard:
            self.assertNotIn(fut, self.runtime._inflight)

    def test_drain_timeout_does_not_hang(self) -> None:
        """A stuck worker can't stall shutdown past the drain timeout."""
        release = threading.Event()
        self._blockers.append(release)
        fut = self.runtime._pool.submit(release.wait)
        self.runtime._track_inflight(fut)

        t0 = time.time()
        self.runtime._drain_chat_pool(timeout=0.5)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 5.0, "drain hung past its timeout")
        self.assertFalse(fut.done(), "stuck future unexpectedly finished")

    def test_stop_drains_before_returning(self) -> None:
        """stop() returns only after in-flight chat work has landed."""
        release = threading.Event()
        marker: dict = {}
        fut = self.runtime._pool.submit(self._slow_work, release, marker)
        self.runtime._track_inflight(fut)
        threading.Timer(0.4, release.set).start()

        self.runtime.stop()  # must not return while the worker is mid-flight

        self.assertTrue(fut.done(), "stop() returned with work still in flight")
        self.assertIn("done_at", marker)

    def test_brain_survives_db_close_mid_persist(self) -> None:
        """The phone race: DB closes between generation and the persist tail.

        Simulates shutdown landing mid-tail by closing the DB inside
        relationship.save, then delegating. The reply parts must still be
        returned — never a StorageError, never a dropped reply.
        """
        real_save = self.brain.relationship.save

        def evil_save(db: Any) -> None:
            db.close()  # the shutdown race, right in the middle of the tail
            return real_save(db)

        self.brain.relationship.save = evil_save  # type: ignore[method-assign]
        message = ChatMessage(chat=self.chat, incoming=True,
                              text="hey", sender="you")
        flags = {"is_owner": True, "in_us": False, "last_active": 0.0}

        parts = self.brain._generate_and_persist(message, flags)

        self.assertTrue(parts, "reply was dropped when the DB closed mid-persist")
        self.assertTrue(any(p.strip() for p in parts))


if __name__ == "__main__":
    unittest.main()
