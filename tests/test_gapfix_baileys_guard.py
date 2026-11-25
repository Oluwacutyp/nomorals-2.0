"""Baileys safety guard wired into the WhatsApp send path (audit gap R1-19).

Failing-first: on the pristine base the adapter has no guard at all —
a burst of sends goes out unpaced, a warning never latches, and there
is no guard state to inspect. Every test here fails on base and passes
once the guard is wired into ``_send_cmd``.

No network beyond localhost, no real WhatsApp session. The fake bridge
is the same shape as tests/test_whatsapp.py's (copied, not imported —
test modules don't import each other).
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nomorals.social.chat.baileys_guard import BaileysPacer
from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef
from nomorals.social.chat.whatsapp import WhatsAppAdapter


class FakeBridge:
    """A scripted JSON-lines bridge on 127.0.0.1:<ephemeral>."""

    def __init__(self, script=None):
        self.script = script or {}
        self.received: list[dict] = []
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        # Accept connections in a loop: a rebuilt adapter connects
        # afresh (see test_guard_state_survives_rebuild), and the
        # bridge must serve the second connection after the first
        # one closes.
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            buf = b""
            try:
                with conn:
                    conn.settimeout(5)
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, _, buf = buf.partition(b"\n")
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line.decode("utf-8"))
                            except json.JSONDecodeError:
                                continue
                            self.received.append(obj)
                            resp = self._respond(obj)
                            if resp is not None:
                                conn.sendall((json.dumps(resp) + "\n").encode())
            except OSError:
                continue

    def _respond(self, obj):
        handler = self.script.get(obj.get("cmd"))
        if handler is None:
            return {"id": obj.get("id"), "ok": False, "error": "unknown command"}
        if callable(handler):
            return handler(obj)
        return {"id": obj.get("id"), **handler}

    def close(self):
        try:
            self._srv.close()
        except OSError:
            return


def make_adapter(bridge, tmpdir, **kw):
    adapter = WhatsAppAdapter(
        host="127.0.0.1", port=bridge.port,
        outbox_dir=str(Path(tmpdir) / "outbox"), **kw)
    sock = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
    adapter._sock = sock
    handler_msgs: list = []
    reader = threading.Thread(
        target=adapter._reader_loop, args=(handler_msgs.append,), daemon=True)
    reader.start()
    adapter.connected.set()
    return adapter, handler_msgs


def chat():
    return ChatRef(platform="whatsapp", chat_id="2348012345678",
                   kind=ChatKind.DM)


class GuardBurstTests(unittest.TestCase):
    def test_burst_of_sends_is_refused_after_five(self):
        bridge = FakeBridge({"send": {"ok": True},
                             "send_media": {"ok": True}})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                clock = [time.time()]
                adapter._pacer = BaileysPacer(
                    clock=lambda: clock[0])
                for i in range(5):
                    result = adapter.send(chat(), f"hello {i}")
                    self.assertTrue(result.ok, result.error)
                refused = adapter.send(chat(), "hello 5")
                self.assertFalse(refused.ok)
                self.assertIn("burst guard", refused.error)
                self.assertIn("slow down", refused.error)
                # A media send rides the same gate.
                media = MediaRef(path=str(Path(tmp) / "pic.jpg"),
                                 kind="image")
                refused_media = adapter.send_media(chat(), media)
                self.assertFalse(refused_media.ok)
                self.assertIn("burst guard", refused_media.error)
                sends = [r for r in bridge.received
                         if r.get("cmd") in ("send", "send_media")]
                self.assertEqual(len(sends), 5)
        finally:
            bridge.close()

    def test_no_daily_cap_sends_past_300_in_one_day(self):
        # The daily cap is gone (owner order 2026-11-25): 350 sends
        # inside one day all go out. The old test pinned the
        # day-1 warm-up cap refusing the 21st send; that contract
        # is dead — the burst window is cleared between groups so
        # only the (removed) daily cap could have refused.
        bridge = FakeBridge({"send": {"ok": True}})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                clock = [time.time()]
                adapter._pacer = BaileysPacer(
                    clock=lambda: clock[0])
                for i in range(350):
                    result = adapter.send(chat(), f"msg {i}")
                    self.assertTrue(result.ok, result.error)
                    if (i + 1) % 5 == 0:
                        clock[0] += 61.0  # clear the burst window
                sends = [r for r in bridge.received
                         if r.get("cmd") == "send"]
                self.assertEqual(len(sends), 350)
        finally:
            bridge.close()

    def test_non_message_commands_are_not_counted(self):
        bridge = FakeBridge({"send": {"ok": True},
                             "history": {"ok": True, "messages": []},
                             "group_participants": {"ok": True,
                                                    "participants": []}})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                adapter._send_cmd({"cmd": "history",
                                   "chat": "2348012345678@c.us",
                                   "limit": 5}, timeout=5)
                adapter._send_cmd({"cmd": "group_participants",
                                   "chat": "12345-678@g.us"}, timeout=5)
                # Non-message commands leave the burst window empty
                # (a full burst of 5 sends still goes out) and the
                # guard state carries no daily-cap keys at all.
                state = adapter.baileys_guard_state()
                for key in ("daily_cap", "sent_today",
                            "session_age_days", "age_assumed"):
                    self.assertNotIn(key, state)
                for i in range(5):
                    result = adapter.send(chat(), f"hello {i}")
                    self.assertTrue(result.ok, result.error)
        finally:
            bridge.close()


class GuardWarningTests(unittest.TestCase):
    def test_warning_latch_stops_all_sends(self):
        bridge = FakeBridge({
            "send": lambda obj: (
                {"id": obj.get("id"), "ok": False,
                 "error": "account temporarily banned for bulk behaviour"}
                if len([r for r in bridge.received
                        if r.get("cmd") == "send"]) <= 1
                else {"id": obj.get("id"), "ok": True}),
        })
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                first = adapter.send(chat(), "first")
                self.assertFalse(first.ok)
                second = adapter.send(chat(), "second")
                self.assertFalse(second.ok)
                self.assertIn("stopped", second.error)
                self.assertIn("disabled", second.error)
                sends = [r for r in bridge.received
                         if r.get("cmd") == "send"]
                self.assertEqual(len(sends), 1)
                state = adapter.baileys_guard_state()
                self.assertTrue(state["warning_stopped"])
        finally:
            bridge.close()

    def test_clear_latch_restores_sends(self):
        bridge = FakeBridge({
            "send": lambda obj: (
                {"id": obj.get("id"), "ok": False,
                 "error": "account temporarily banned for bulk behaviour"}
                if len([r for r in bridge.received
                        if r.get("cmd") == "send"]) <= 1
                else {"id": obj.get("id"), "ok": True}),
        })
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                adapter.send(chat(), "first")
                refused = adapter.send(chat(), "second")
                self.assertFalse(refused.ok)
                cleared = adapter.clear_baileys_warning()
                self.assertIn("cleared", cleared)
                third = adapter.send(chat(), "third")
                self.assertTrue(third.ok, third.error)
                self.assertFalse(
                    adapter.baileys_guard_state()["warning_stopped"])
        finally:
            bridge.close()

    def test_status_event_banned_state_latches(self):
        bridge = FakeBridge({"send": {"ok": True}})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                adapter._dispatch(
                    {"type": "status", "state": "banned",
                     "user": "2348012345678@c.us"}, lambda m: None)
                # The ban state also drops the bridge link; the
                # latch is what matters — it survives the
                # reconnect, so simulate the bridge coming back
                # and assert the next send is refused at the gate.
                adapter.connected.set()
                refused = adapter.send(chat(), "hello")
                self.assertFalse(refused.ok)
                self.assertIn("stopped", refused.error)
                self.assertEqual(
                    len([r for r in bridge.received
                         if r.get("cmd") == "send"]), 0)
        finally:
            bridge.close()


class GuardStateTests(unittest.TestCase):
    def test_no_state_file_has_no_daily_cap(self):
        # The daily cap is gone (owner order 2026-11-25): a fresh
        # adapter reports no daily-cap keys at all — the old test
        # pinned daily_cap == 300 on a warmed-up seed, which no
        # longer exists as a concept.
        bridge = FakeBridge({"send": {"ok": True}})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                state = adapter.baileys_guard_state()
                for key in ("daily_cap", "sent_today",
                            "session_age_days", "age_assumed"):
                    self.assertNotIn(key, state)
                self.assertEqual(state["burst_max"], 5)
                self.assertEqual(state["burst_window_s"], 60.0)
                self.assertFalse(state["warning_stopped"])
        finally:
            bridge.close()

    def test_guard_state_survives_rebuild(self):
        bridge = FakeBridge({"send": {"ok": True}})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                self.assertTrue(adapter.send(chat(), "one").ok)
                self.assertTrue(adapter.send(chat(), "two").ok)
                store = Path(tmp) / "outbox" / "whatsapp_baileys_guard.json"
                self.assertTrue(store.is_file())
                saved = json.loads(store.read_text(encoding="utf-8"))
                # The durable pacer state carries no daily fields
                # anymore — only the burst window survives.
                for key in ("session_age_days", "sent_today", "day"):
                    self.assertNotIn(key, saved["pacer"])
                self.assertIn("recent", saved["pacer"])
                # The rebuilt adapter inherits the burst window: 3
                # more rapid sends go out, the 6th is refused.
                # Close the first adapter's socket so the bridge
                # serves the rebuild's fresh connection.
                adapter._sock.close()
                adapter2, _ = make_adapter(bridge, tmp)
                for i in range(3):
                    result = adapter2.send(chat(), f"again {i}")
                    self.assertTrue(result.ok, result.error)
                refused = adapter2.send(chat(), "again 3")
                self.assertFalse(refused.ok)
                self.assertIn("burst guard", refused.error)
        finally:
            bridge.close()

    def test_flush_delay_is_zero_when_idle_and_human_when_busy(self):
        bridge = FakeBridge({"send": {"ok": True}})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                adapter, _ = make_adapter(bridge, tmp)
                self.assertEqual(adapter._guard_flush_delay_s(), 0.0)
                adapter._guard_last_sent_at = time.time()
                delay = adapter._guard_flush_delay_s()
                self.assertGreater(delay, 0.0)
                self.assertLessEqual(delay, 12.0)
        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main()
