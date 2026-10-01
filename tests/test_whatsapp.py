"""WhatsApp bridge adapter — fake-bridge tests.

No network beyond localhost, no real WhatsApp session. A tiny
JSON-lines fake bridge answers the adapter's commands so we can assert on
the protocol: JID normalization, send/retry, media, history parsing, QR
surfacing, and graceful degradation when the bridge lacks a command.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest

from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef
from nomorals.social.chat.whatsapp import WhatsAppAdapter, _split_text


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
            pass

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
            pass


def make_adapter(bridge, **kw):
    adapter = WhatsAppAdapter(host="127.0.0.1", port=bridge.port, **kw)
    sock = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
    adapter._sock = sock
    handler_msgs: list = []
    reader = threading.Thread(
        target=adapter._reader_loop, args=(handler_msgs.append,), daemon=True)
    reader.start()
    adapter.connected.set()
    return adapter, handler_msgs


class JidTests(unittest.TestCase):
    def setUp(self):
        self.a = WhatsAppAdapter()

    def test_plain_number_becomes_c_us(self):
        chat = ChatRef(platform="whatsapp", chat_id="2348012345678", kind=ChatKind.DM)
        self.assertEqual(self.a._jid(chat), "2348012345678@c.us")

    def test_already_qualified_unchanged(self):
        chat = ChatRef(platform="whatsapp", chat_id="2348012345678@s.whatsapp.net",
                       kind=ChatKind.DM)
        self.assertEqual(self.a._jid(chat), "2348012345678@s.whatsapp.net")

    def test_group_jid_unchanged(self):
        chat = ChatRef(platform="whatsapp", chat_id="12345-678@g.us", kind=ChatKind.GROUP)
        self.assertEqual(self.a._jid(chat), "12345-678@g.us")


class SplitTextTests(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(_split_text("hello"), ["hello"])

    def test_long_text_splits_under_limit(self):
        text = "\n\n".join(f"paragraph {i} " + "x" * 500 for i in range(20))
        chunks = _split_text(text, limit=4000)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 4000)
        # no content lost
        self.assertIn("paragraph 19", chunks[-1])


class SendTests(unittest.TestCase):
    def test_send_ok(self):
        bridge = FakeBridge({"send": {"ok": True}})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="2348012345678",
                           kind=ChatKind.DM)
            res = adapter.send(chat, "hello")
            self.assertTrue(res.ok)
            self.assertEqual(bridge.received[0]["cmd"], "send")
            self.assertEqual(bridge.received[0]["chat"], "2348012345678@c.us")
            self.assertEqual(bridge.received[0]["text"], "hello")
        finally:
            bridge.close()

    def test_send_rejected_no_retry(self):
        # permanent rejection → exactly one attempt
        bridge = FakeBridge({"send": {"ok": False, "error": "blocked"}})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
            res = adapter.send(chat, "hi")
            self.assertFalse(res.ok)
            self.assertIn("blocked", res.error)
            self.assertEqual(len(bridge.received), 1)
        finally:
            bridge.close()

    def test_send_timeout_retries_once(self):
        calls = []

        def flaky(obj):
            calls.append(obj)
            if len(calls) == 1:
                return {"id": obj.get("id"), "ok": False,
                        "error": "bridge response timeout"}
            return {"id": obj.get("id"), "ok": True}

        bridge = FakeBridge({"send": flaky})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
            res = adapter.send(chat, "hi")
            self.assertTrue(res.ok)
            self.assertEqual(len(calls), 2)
        finally:
            bridge.close()

    def test_long_send_splits(self):
        bridge = FakeBridge({"send": {"ok": True}})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
            res = adapter.send(chat, "x" * 9000)
            self.assertTrue(res.ok)
            self.assertGreater(len(bridge.received), 1)
            total = sum(len(m["text"]) for m in bridge.received)
            self.assertEqual(total, 9000)
        finally:
            bridge.close()

    def test_send_when_disconnected(self):
        adapter = WhatsAppAdapter()
        chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
        res = adapter.send(chat, "hi")
        self.assertFalse(res.ok)
        self.assertIn("not connected", res.error)


class MediaTests(unittest.TestCase):
    def test_send_media_photo(self):
        bridge = FakeBridge({"send_media": {"ok": True}})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="12345-678@g.us",
                           kind=ChatKind.GROUP)
            res = adapter.send_media(
                chat, MediaRef(path="/tmp/pic.jpg", mime="image/jpeg", kind="photo"),
                caption="look")
            self.assertTrue(res.ok)
            cmd = bridge.received[0]
            self.assertEqual(cmd["cmd"], "send_media")
            self.assertEqual(cmd["path"], "/tmp/pic.jpg")
            self.assertEqual(cmd["caption"], "look")
            self.assertNotIn("ptt", cmd)
        finally:
            bridge.close()

    def test_send_voice_sets_ptt(self):
        bridge = FakeBridge({"send_media": {"ok": True}})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="2348012345678",
                           kind=ChatKind.DM)
            res = adapter.send_voice(chat, "/tmp/note.ogg")
            self.assertTrue(res.ok)
            cmd = bridge.received[0]
            self.assertEqual(cmd["cmd"], "send_media")
            self.assertTrue(cmd["ptt"])
            self.assertEqual(cmd["chat"], "2348012345678@c.us")
        finally:
            bridge.close()

    def test_send_media_no_path(self):
        adapter = WhatsAppAdapter()
        adapter.connected.set()
        chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
        res = adapter.send_media(chat, MediaRef(path="", kind="photo"))
        self.assertFalse(res.ok)


class InboundTests(unittest.TestCase):
    def test_status_open_sets_connected(self):
        adapter = WhatsAppAdapter()
        msgs: list = []
        adapter._dispatch({"type": "status", "state": "open",
                           "user": "2348012345678@s.whatsapp.net"}, msgs.append)
        self.assertTrue(adapter.connected.is_set())
        self.assertEqual(adapter.user_jid, "2348012345678@s.whatsapp.net")

    def test_status_closed_clears_connected(self):
        adapter = WhatsAppAdapter()
        adapter.connected.set()
        msgs: list = []
        adapter._dispatch({"type": "status", "state": "closed"}, msgs.append)
        self.assertFalse(adapter.connected.is_set())

    def test_qr_surfaces_in_health(self):
        adapter = WhatsAppAdapter()
        msgs: list = []
        adapter._dispatch({"type": "qr", "data": "QRDATA"}, msgs.append)
        self.assertEqual(adapter.last_qr, "QRDATA")
        health = adapter.health()
        self.assertTrue(health["qr_pending"])
        self.assertIsNotNone(health["qr_age_s"])

    def test_message_dispatch(self):
        adapter = WhatsAppAdapter()
        msgs: list = []
        adapter._dispatch({
            "type": "message",
            "chat": {"id": "2348012345678@c.us", "kind": "dm", "title": "Ada"},
            "from": {"id": "2348012345678@c.us", "name": "Ada"},
            "text": "hey devon",
            "media": [{"path": "/tmp/a.jpg", "mime": "image/jpeg", "kind": "photo"}],
            "reply_to": "mid1",
            "ts": 1759276800000,
        }, msgs.append)
        self.assertEqual(len(msgs), 1)
        msg = msgs[0]
        self.assertEqual(msg.text, "hey devon")
        self.assertEqual(msg.sender, "Ada")
        self.assertEqual(msg.reply_to, "mid1")
        self.assertTrue(msg.has_media)
        self.assertEqual(msg.first_media.kind, "photo")

    def test_response_routes_to_pending(self):
        bridge = FakeBridge({"status": {"ok": True, "state": "open"}})
        try:
            adapter, _ = make_adapter(bridge)
            resp = adapter._send_cmd({"cmd": "status"}, timeout=5)
            self.assertTrue(resp.get("ok"))
        finally:
            bridge.close()


class HistoryTests(unittest.TestCase):
    def test_history_parses_media_and_direction(self):
        bridge = FakeBridge({"history": {"ok": True, "messages": [
            {"text": "hi", "sender": "Ada", "incoming": True, "ts": 1759276800000},
            {"text": "yo", "sender": "me", "incoming": False, "ts": 1759276860000,
             "media": [{"path": "/tmp/v.mp4", "mime": "video/mp4", "kind": "video"}]},
        ]}})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
            msgs = adapter.history(chat, limit=10)
            self.assertEqual(len(msgs), 2)
            self.assertTrue(msgs[0].incoming)
            self.assertFalse(msgs[1].incoming)
            self.assertTrue(msgs[1].has_media)
            self.assertEqual(bridge.received[0]["cmd"], "history")
            self.assertEqual(bridge.received[0]["limit"], 10)
        finally:
            bridge.close()


class DegradeTests(unittest.TestCase):
    def test_mark_read_unknown_command(self):
        bridge = FakeBridge({})  # no "read" command → ok:false
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
            self.assertFalse(adapter.mark_read(chat))
        finally:
            bridge.close()

    def test_mark_read_ok(self):
        bridge = FakeBridge({"read": {"ok": True}})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
            self.assertTrue(adapter.mark_read(chat))
        finally:
            bridge.close()

    def test_chats_unknown_command(self):
        bridge = FakeBridge({})
        try:
            adapter, _ = make_adapter(bridge)
            self.assertEqual(adapter.chats(), [])
        finally:
            bridge.close()

    def test_chats_ok(self):
        bridge = FakeBridge({"chats": {"ok": True, "chats": [
            {"id": "123@c.us", "title": "Ada", "kind": "dm", "ts": 1759276800000},
            {"id": "", "title": "ghost"},
        ]}})
        try:
            adapter, _ = make_adapter(bridge)
            chats = adapter.chats()
            self.assertEqual(len(chats), 1)
            self.assertEqual(chats[0]["title"], "Ada")
        finally:
            bridge.close()

    def test_preflight_unreachable(self):
        adapter = WhatsAppAdapter(port=1)  # nothing listens on port 1
        with self.assertRaises(RuntimeError):
            adapter.preflight()

    def test_typing(self):
        bridge = FakeBridge({"typing": {"ok": True}})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123", kind=ChatKind.DM)
            self.assertTrue(adapter.typing(chat, seconds=2))
            self.assertEqual(bridge.received[0]["seconds"], 2)
        finally:
            bridge.close()


class ResponseRoutingTests(unittest.TestCase):
    def test_typed_response_wakes_command(self):
        # the bridge may send {"type": "response", ...} — must route
        bridge = FakeBridge({
            "send": lambda obj: {"type": "response", "id": obj.get("id"),
                                 "ok": True},
        })
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123",
                            kind=ChatKind.DM)
            result = adapter.send(chat, "hello")
            self.assertTrue(result.ok)
        finally:
            bridge.close()

    def test_untyped_response_still_wakes_command(self):
        bridge = FakeBridge({
            "send": lambda obj: {"id": obj.get("id"), "ok": True},
        })
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123",
                            kind=ChatKind.DM)
            result = adapter.send(chat, "hello")
            self.assertTrue(result.ok)
        finally:
            bridge.close()

    def test_timeout_does_not_leak_pending_entry(self):
        # bridge never answers → timeout; the pending slot must be gone
        bridge = FakeBridge({"send": lambda obj: None})
        try:
            adapter, _ = make_adapter(bridge)
            chat = ChatRef(platform="whatsapp", chat_id="123",
                            kind=ChatKind.DM)
            # shrink the timeout via a direct _send_cmd call
            resp = adapter._send_cmd({"cmd": "send", "to": "123@c.us",
                                      "text": "hi"}, timeout=0.3)
            self.assertFalse(resp["ok"])
            self.assertEqual(resp["error"], "bridge response timeout")
            self.assertEqual(adapter._responses, {})
        finally:
            bridge.close()


class ReconnectTests(unittest.TestCase):
    def test_run_reconnects_after_drop(self):
        # a bridge that accepts, then drops the connection; the adapter
        # must come back and connect again
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        port = srv.getsockname()[1]
        connections: list = []
        stop = threading.Event()

        def serve():
            srv.settimeout(0.5)
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except OSError:
                    continue
                connections.append(conn)
                # drop the first connection after a beat; keep the rest
                if len(connections) == 1:
                    time.sleep(0.4)
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    conn.close()

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        adapter = WhatsAppAdapter(host="127.0.0.1", port=port,
                                  reconnect_delay=0.2)
        runner = threading.Thread(target=adapter.run,
                                  args=(lambda m: None,), daemon=True)
        try:
            runner.start()
            deadline = time.time() + 8
            while time.time() < deadline and len(connections) < 2:
                time.sleep(0.1)
            self.assertGreaterEqual(
                len(connections), 2,
                "adapter did not reconnect after the drop")
        finally:
            adapter.stop()
            stop.set()
            runner.join(timeout=5)
            srv.close()


if __name__ == "__main__":
    unittest.main()
