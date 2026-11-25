"""Baileys daily send cap — REMOVED (owner order 2026-11-25).

The daily cap is gone: no daily limit, no warm-up ramp, no
replacement limit dressed up as something else. These tests pin
the removal contract:

* sends past 300 in one day are allowed (pacer + adapter);
* the burst guard still trips at 6-in-60s;
* the first-warning latch still latches;
* a legacy state file carrying the old daily fields loads without
  a cap and is rewritten without them.

Failing-first: on the pristine base every test here fails — the
day-1 cap refuses the 21st send, ``daily_cap`` exists, the old
names import, and a legacy state file with sent_today=299 blocks
the next send.

No network beyond localhost, no real WhatsApp session. The fake
bridge is the same shape as tests/test_gapfix_baileys_guard.py's
(copied, not imported — test modules don't import each other).
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from nomorals.social.chat.baileys_guard import BaileysPacer, WarningDetector
from nomorals.social.chat.base import ChatKind, ChatRef
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
                        return
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
            return

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


def _clear_burst(clock):
    clock[0] += 61.0


class CapRemovedTests(unittest.TestCase):
    def test_pacer_allows_more_than_300_sends_in_one_day(self):
        clock = [time.time()]
        pacer = BaileysPacer(clock=lambda: clock[0])
        for i in range(400):
            ok, reason = pacer.may_send()
            self.assertTrue(ok, f"send {i} refused: {reason}")
            pacer.record_sent()
            if (i + 1) % 5 == 0:
                _clear_burst(clock)

    def test_cap_names_are_gone(self):
        pacer = BaileysPacer()
        self.assertFalse(hasattr(pacer, "daily_cap"))
        self.assertFalse(hasattr(pacer, "sent_today"))
        self.assertFalse(hasattr(pacer, "session_age_days"))
        with self.assertRaises(ImportError):
            from nomorals.social.chat.baileys_guard import WARMUP_SCHEDULE  # noqa: F401
        with self.assertRaises(ImportError):
            from nomorals.social.chat.baileys_guard import STEADY_DAILY_CAP  # noqa: F401
        with self.assertRaises(TypeError):
            BaileysPacer(session_age_days=0)

    def test_burst_guard_still_trips_at_six_in_60s(self):
        clock = [time.time()]
        pacer = BaileysPacer(clock=lambda: clock[0])
        for _ in range(5):
            ok, _ = pacer.may_send()
            self.assertTrue(ok)
            pacer.record_sent()
        ok, reason = pacer.may_send()
        self.assertFalse(ok)
        self.assertIn("burst guard", reason)
        # The window clears: the next send after 61s goes out.
        _clear_burst(clock)
        ok, _ = pacer.may_send()
        self.assertTrue(ok)

    def test_warning_latch_still_latches(self):
        det = WarningDetector()
        self.assertTrue(det.may_send())
        self.assertTrue(
            det.note("account temporarily banned for bulk behaviour"))
        self.assertFalse(det.may_send())
        det.note("all clear now")
        self.assertFalse(det.may_send())  # latched: never talks past it


class CapRemovedAdapterTests(unittest.TestCase):
    def test_legacy_state_file_loads_without_a_cap(self):
        # A state file from an older build: day-1 session, 299
        # sends already counted today — one short of the old
        # steady cap. The new adapter must send anyway, and the
        # store it writes back must carry no daily fields.
        bridge = FakeBridge({"send": {"ok": True}})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                outbox = Path(tmp) / "outbox"
                outbox.mkdir(parents=True)
                today = int(time.time() // 86400)
                legacy = {
                    "first_seen_day": today,
                    "age_assumed": False,
                    "pacer": {
                        "session_age_days": 0,
                        "sent_today": 299,
                        "day": today,
                        "recent": [],
                    },
                    "warning": {
                        "stopped": False,
                        "first_warning": "",
                        "first_warning_at": 0.0,
                    },
                }
                store = outbox / "whatsapp_baileys_guard.json"
                store.write_text(json.dumps(legacy), encoding="utf-8")
                adapter, _ = make_adapter(bridge, tmp)
                result = adapter.send(chat(), "after the cap died")
                self.assertTrue(result.ok, result.error)
                self.assertEqual(
                    len([r for r in bridge.received
                         if r.get("cmd") == "send"]), 1)
                saved = json.loads(store.read_text(encoding="utf-8"))
                for key in ("first_seen_day", "age_assumed"):
                    self.assertNotIn(key, saved)
                for key in ("session_age_days", "sent_today", "day"):
                    self.assertNotIn(key, saved["pacer"])
                state = adapter.baileys_guard_state()
                for key in ("daily_cap", "sent_today",
                            "session_age_days", "age_assumed"):
                    self.assertNotIn(key, state)
        finally:
            bridge.close()

    def test_adapter_warning_latch_still_stops_sends(self):
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
                self.assertTrue(
                    adapter.baileys_guard_state()["warning_stopped"])
                sends = [r for r in bridge.received
                         if r.get("cmd") == "send"]
                self.assertEqual(len(sends), 1)
        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main()
