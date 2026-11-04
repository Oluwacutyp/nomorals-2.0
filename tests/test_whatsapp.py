"""WhatsApp bridge adapter — fake-bridge tests.

No network beyond localhost, no real WhatsApp session. A tiny
JSON-lines fake bridge answers the adapter's commands so we can assert on
the protocol: JID normalization, send/retry, media, history parsing, QR
surfacing, and graceful degradation when the bridge lacks a command.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest

from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef
from nomorals.social.chat.whatsapp import (
    UNCONFIRMED_SEND_ERROR, WhatsAppAdapter, _split_text)


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


class SelfChatLidTests(unittest.TestCase):
    """The self-chat under LID addressing (lidfix).

    On LID-migrated accounts WhatsApp delivers the "Message
    yourself" chat — and the owner's own_message events in it —
    under the account's LID (``<lid>@lid``), whose user part never
    equals the phone number in the status payload's ``user``. The
    bridge therefore also reports ``lid`` in its status events,
    and the adapter must recognise EITHER identity as the
    self-chat — and nothing else.
    """

    OWNER = "2348012345678@s.whatsapp.net"
    LID = "105382746195847@lid"
    STRANGER_LID = "998877665544332211@lid"

    def _adapter(self, status):
        adapter = WhatsAppAdapter()
        msgs: list = []
        adapter._dispatch(status, msgs.append)
        return adapter, msgs

    @staticmethod
    def _own(chat_id, kind="dm", text="/help", mid="m1"):
        return {"type": "own_message", "id": mid,
                "chat": {"id": chat_id, "kind": kind, "title": ""},
                "text": text, "ts": 1_700_000_000_000}

    def test_status_stores_user_lid(self):
        adapter, _ = self._adapter(
            {"type": "status", "state": "open", "user": self.OWNER,
             "lid": self.LID})
        self.assertEqual(adapter.user_jid, self.OWNER)
        self.assertEqual(adapter.user_lid, self.LID)

    def test_lid_self_chat_message_is_an_incoming_owner_turn(self):
        adapter, msgs = self._adapter(
            {"type": "status", "state": "open", "user": self.OWNER,
             "lid": self.LID})
        adapter._dispatch(self._own(self.LID), msgs.append)
        self.assertEqual(len(msgs), 1)
        msg = msgs[0]
        self.assertTrue(msg.incoming)
        self.assertEqual(msg.text, "/help")
        # The turn keeps the chat id AS DELIVERED — WhatsApp
        # addresses this self-chat by its LID, and the reply path
        # must answer on the same address.
        self.assertEqual(msg.chat.chat_id, self.LID)
        self.assertEqual(msg.meta.get("from_owner"), True)
        self.assertEqual(msg.meta.get("self_chat"), True)
        self.assertNotIn("own_message", msg.meta)

    def test_phone_self_chat_still_matches_when_lid_known(self):
        adapter, msgs = self._adapter(
            {"type": "status", "state": "open", "user": self.OWNER,
             "lid": self.LID})
        adapter._dispatch(self._own(self.OWNER, mid="m2"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("self_chat"), True)

    def test_stranger_lid_chat_stays_persist_only(self):
        """An @lid chat that is neither of the account's own
        identities is somebody else's DM — persist-only."""
        adapter, msgs = self._adapter(
            {"type": "status", "state": "open", "user": self.OWNER,
             "lid": self.LID})
        adapter._dispatch(self._own(self.STRANGER_LID, text="hi"),
                          msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertFalse(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("own_message"), True)
        self.assertNotIn("self_chat", msgs[0].meta)

    def test_group_kind_lid_chat_is_never_self_chat(self):
        adapter, msgs = self._adapter(
            {"type": "status", "state": "open", "user": self.OWNER,
             "lid": self.LID})
        adapter._dispatch(
            self._own(self.LID, kind="group", text="gm all"),
            msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertFalse(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("own_message"), True)
        self.assertNotIn("self_chat", msgs[0].meta)

    def test_lid_self_chat_unknown_when_bridge_reports_no_lid(self):
        """Safe default pinned: a bridge that reports no ``lid``
        (older build, or the account has none) leaves @lid chats
        persist-only — the adapter never guesses an identity."""
        adapter, msgs = self._adapter(
            {"type": "status", "state": "open", "user": self.OWNER})
        self.assertEqual(adapter.user_lid, "")
        adapter._dispatch(self._own(self.LID), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertFalse(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("own_message"), True)
        self.assertNotIn("self_chat", msgs[0].meta)

    def test_own_send_echo_in_lid_self_chat_stays_persist_only(self):
        """Devon's reply into the LID self-chat echoes back as an
        own_message carrying the id her send was assigned — the
        sent-id registry suppresses it exactly as on the phone
        form, while the owner's own typing (an id she never sent)
        is a turn."""
        adapter, msgs = self._adapter(
            {"type": "status", "state": "open", "user": self.OWNER,
             "lid": self.LID})
        adapter.connected.set()
        adapter._send_cmd = lambda cmd, timeout=25.0: {
            "id": cmd.get("id"), "ok": True, "message_id": "ECHO-L"}
        chat = ChatRef(platform="whatsapp", chat_id=self.LID,
                       kind=ChatKind.DM, peer=self.LID)
        result = adapter.send(chat, "the help menu")
        self.assertTrue(result.ok)
        self.assertEqual(result.message_id, "ECHO-L")
        adapter._dispatch(
            self._own(self.LID, text="the help menu", mid="ECHO-L"),
            msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertFalse(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("own_message"), True)
        adapter._dispatch(
            self._own(self.LID, text="/pair", mid="OWN-L9"),
            msgs.append)
        self.assertEqual(len(msgs), 2)
        self.assertTrue(msgs[1].incoming)
        self.assertEqual(msgs[1].meta.get("self_chat"), True)


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


class ScriptedSocket:
    """A recv-scripted stand-in for the bridge socket.

    Each script item is bytes (returned by ``recv``) or an exception
    instance (raised by ``recv``); past the end of the script, ``recv``
    returns b"" (orderly close). When ``gate_before`` is a script
    index, the recv that would serve that item first sets
    ``gate_reached`` and blocks until the test sets ``gate_release`` —
    how a test observes adapter state mid-stream.
    """

    def __init__(self, script, gate_before=None):
        self._script = list(script)
        self._gate_before = gate_before
        self.recv_calls = 0
        self.gate_reached = threading.Event()
        self.gate_release = threading.Event()

    def recv(self, _n):
        idx = self.recv_calls
        self.recv_calls += 1
        if self._gate_before is not None and idx == self._gate_before:
            self.gate_reached.set()
            self.gate_release.wait(timeout=5)
        if idx >= len(self._script):
            return b""
        item = self._script[idx]
        if isinstance(item, BaseException):
            raise item
        return item


class ReaderTimeoutTests(unittest.TestCase):
    """The 1.0s socket timeout in run() is a stop-check tick, not a
    disconnect: an idle bridge must never flap (owner EC2 live log,
    2026-11-04: connect → "connection lost" once a second, forever).
    """

    def test_timeouts_around_a_status_line_are_survived(self):
        status = (json.dumps(
            {"type": "status", "state": "open",
             "user": "2348012345678@s.whatsapp.net"}) + "\n").encode()
        script = [socket.timeout("tick"), TimeoutError("tick"), status,
                  socket.timeout("tick"), socket.timeout("tick"), b""]
        sock = ScriptedSocket(script, gate_before=3)
        adapter = WhatsAppAdapter()
        adapter._sock = sock
        reader = threading.Thread(target=adapter._reader_loop,
                                  args=(lambda m: None,), daemon=True)
        reader.start()
        try:
            # The gate fires on the first recv AFTER the status line:
            # by then the reader must have dispatched it and still be
            # alive despite the two timeouts that preceded it.
            self.assertTrue(sock.gate_reached.wait(timeout=5),
                            "reader never got past the status line")
            self.assertTrue(reader.is_alive())
            self.assertTrue(adapter.connected.is_set())
            self.assertEqual(adapter.user_jid,
                             "2348012345678@s.whatsapp.net")
        finally:
            sock.gate_release.set()
        reader.join(timeout=5)
        # Only the final b"" (orderly close) may end the loop: every
        # scripted item was consumed, and the genuine close still
        # clears `connected` on the way out.
        self.assertFalse(reader.is_alive())
        self.assertEqual(sock.recv_calls, len(script))
        self.assertFalse(adapter.connected.is_set())

    def test_timeouts_alone_never_end_the_loop_or_clear_connected(self):
        class IdleSocket:
            def __init__(self):
                self.recv_calls = 0

            def recv(self, _n):
                self.recv_calls += 1
                time.sleep(0.01)  # pace the tick; a real socket blocks
                raise socket.timeout("tick")

        sock = IdleSocket()
        adapter = WhatsAppAdapter()
        adapter._sock = sock
        adapter.connected.set()
        reader = threading.Thread(target=adapter._reader_loop,
                                  args=(lambda m: None,), daemon=True)
        reader.start()
        try:
            deadline = time.time() + 5
            while sock.recv_calls < 5 and time.time() < deadline:
                time.sleep(0.02)
            self.assertGreaterEqual(sock.recv_calls, 5)
            self.assertTrue(reader.is_alive(),
                            "reader died on recv timeouts")
            self.assertTrue(adapter.connected.is_set(),
                            "an idle tick cleared the connected state")
        finally:
            adapter.stop()
        reader.join(timeout=3)
        self.assertFalse(reader.is_alive(),
                         "reader did not notice stop() promptly")


class FlapBridge:
    """A fake bridge that can eat acks — the 2026-11-04 incident rig.

    Unlike FakeBridge (one connection, always answers), this one
    accepts connection after connection, records every send it
    receives as a delivered own message (its history), greets each
    connection with a status event like the real bridge, and can be
    told to never answer send commands — the message lands, the ack
    never comes, which is exactly what the flapping adapter
    connection did to the real bridge.
    """

    def __init__(self):
        self.received: list[dict] = []
        self.delivered: list[dict] = []
        self.ack_sends = True       # False: sends land, acks never come
        self.serve_history = True   # False: history answers empty
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(5)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop,
                                        daemon=True)
        self._thread.start()

    def _accept_loop(self):
        self._srv.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                continue
            threading.Thread(target=self._serve_conn, args=(conn,),
                             daemon=True).start()

    @staticmethod
    def _send_line(conn, obj):
        try:
            conn.sendall((json.dumps(obj) + "\n").encode())
        except OSError:
            pass

    def _serve_conn(self, conn):
        with conn:
            self._send_line(conn, {"type": "status", "state": "open",
                                   "user": "2348012345678@s.whatsapp.net"})
            buf = b""
            conn.settimeout(5)
            try:
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
                        self._handle(conn, obj)
            except OSError:
                return

    def _handle(self, conn, obj):
        cmd = obj.get("cmd")
        with self._lock:
            self.received.append(obj)
        if cmd == "send":
            with self._lock:
                self.delivered.append({
                    "incoming": False, "text": obj.get("text") or "",
                    "sender": "", "media": [], "reply_to": "",
                    "ts": int(time.time() * 1000)})
                n = len(self.delivered)
            if self.ack_sends:
                self._send_line(conn, {"id": obj.get("id"), "ok": True,
                                       "message_id": f"mid-{n}"})
            return
        if cmd == "history":
            with self._lock:
                msgs = [dict(m) for m in self.delivered] \
                    if self.serve_history else []
            self._send_line(conn, {"id": obj.get("id"), "ok": True,
                                   "messages": msgs})
            return
        self._send_line(conn, {"id": obj.get("id"), "ok": True})

    def send_count(self, needle):
        with self._lock:
            return sum(1 for m in self.delivered if needle in m["text"])

    def close(self):
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass


def _shrink_ack_wait(adapter, timeout=1.0):
    """Shorten the adapter's bridge-response wait for flap tests.

    The behaviour under test (dispatch marking, reconciliation, the
    send-once guard) does not depend on the wait's length — only on
    the ack never arriving inside it.
    """
    orig = adapter._send_cmd

    def wrapped(cmd, *, timeout=timeout):
        return orig(cmd, timeout=timeout)

    adapter._send_cmd = wrapped


def _wire(adapter, bridge):
    """Connect an adapter to a bridge the way run() would, minus the
    reconnect loop: one socket + one reader thread."""
    sock = socket.create_connection(("127.0.0.1", bridge.port), timeout=5)
    adapter._sock = sock
    reader = threading.Thread(target=adapter._reader_loop,
                              args=(lambda m: None,), daemon=True)
    reader.start()
    deadline = time.time() + 5
    while not adapter.connected.is_set() and time.time() < deadline:
        time.sleep(0.05)
    return reader


class SendOnceTests(unittest.TestCase):
    """One reply, one copy — however the connection flaps."""

    def _queued_adapter(self, bridge, tmp):
        """An adapter that queued one reply while the bridge was down."""
        adapter = WhatsAppAdapter(host="127.0.0.1", port=bridge.port,
                                  outbox_dir=tmp)
        _shrink_ack_wait(adapter)
        chat = ChatRef(platform="whatsapp", chat_id="120363012345@g.us",
                       kind=ChatKind.GROUP)
        result = adapter.send(chat, "the reply")
        self.assertFalse(result.ok)  # queued, not sent
        rows = adapter.outbox.pending()
        self.assertEqual(len(rows), 1)
        self.assertNotIn("dispatched_at", rows[0])
        return adapter, chat

    def test_outbox_dispatch_with_lost_ack_is_reconciled_not_resent(self):
        bridge = FlapBridge()
        bridge.ack_sends = False  # sends land; acks never come
        tmp = tempfile.mkdtemp()
        try:
            adapter, _chat = self._queued_adapter(bridge, tmp)
            _wire(adapter, bridge)
            # The reconnect flush (async, on the status event) makes
            # the one dispatch and loses its ack; wait for its stamp.
            deadline = time.time() + 8
            rows = adapter.outbox.pending()
            while not rows[0].get("dispatched_at") \
                    and time.time() < deadline:
                time.sleep(0.05)
                rows = adapter.outbox.pending()
            self.assertTrue(rows[0].get("dispatched_at"),
                            "first flush never dispatched the row")
            self.assertEqual(bridge.send_count("the reply"), 1)
            # The next flush must NOT send it again: it reconciles
            # against the bridge's history, finds the message, and
            # retires the row as sent.
            adapter.outbox.flush(adapter._deliver_entry,
                                 reconcile=adapter._reconcile_entry)
            self.assertEqual(bridge.send_count("the reply"), 1)
            self.assertEqual(adapter.outbox.pending(), [])
        finally:
            adapter.stop()
            bridge.close()

    def test_outbox_dispatch_with_no_trace_is_held_not_resent(self):
        bridge = FlapBridge()
        bridge.ack_sends = False
        bridge.serve_history = False  # the bridge has no trace of it
        tmp = tempfile.mkdtemp()
        try:
            adapter, _chat = self._queued_adapter(bridge, tmp)
            _wire(adapter, bridge)
            deadline = time.time() + 8
            rows = adapter.outbox.pending()
            while not rows[0].get("dispatched_at") \
                    and time.time() < deadline:
                time.sleep(0.05)
                rows = adapter.outbox.pending()
            self.assertTrue(rows[0].get("dispatched_at"))
            self.assertEqual(bridge.send_count("the reply"), 1)
            report = adapter.outbox.flush(
                adapter._deliver_entry, reconcile=adapter._reconcile_entry)
            self.assertEqual(report["held"], 1)
            rows = adapter.outbox.pending()
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].get("held"))
            # A held row is never dispatched again, however often
            # later flushes run.
            adapter.outbox.flush(adapter._deliver_entry,
                                 reconcile=adapter._reconcile_entry)
            self.assertEqual(bridge.send_count("the reply"), 1)
        finally:
            adapter.stop()
            bridge.close()

    def test_placeholder_and_final_across_a_flap_send_once(self):
        bridge = FlapBridge()
        bridge.ack_sends = False
        tmp = tempfile.mkdtemp()
        try:
            adapter, _msgs = make_adapter(bridge, outbox_dir=tmp)
            _shrink_ack_wait(adapter)
            chat = ChatRef(platform="whatsapp",
                           chat_id="120363012345@g.us",
                           kind=ChatKind.GROUP)
            placeholder = ("one sec — still thinking, the model is "
                           "slow on the phone")
            r1 = adapter.send(chat, placeholder)
            self.assertFalse(r1.ok)
            self.assertEqual(r1.error, UNCONFIRMED_SEND_ERROR)
            # The old retry re-dispatched on the lost ack within this
            # single send() call — the bridge must hold ONE copy.
            self.assertEqual(bridge.send_count("still thinking"), 1)
            r2 = adapter.send(chat, "the final answer")
            self.assertFalse(r2.ok)
            self.assertEqual(bridge.send_count("final answer"), 1)
            # A caller retrying after the "reconnect" is answered
            # from the bridge's history — success, and STILL one
            # copy of each.
            r3 = adapter.send(chat, placeholder)
            self.assertTrue(r3.ok)
            r4 = adapter.send(chat, "the final answer")
            self.assertTrue(r4.ok)
            self.assertEqual(bridge.send_count("still thinking"), 1)
            self.assertEqual(bridge.send_count("final answer"), 1)
        finally:
            adapter.stop()
            bridge.close()

    def test_never_dispatched_row_flushes_normally_after_reconnect(self):
        bridge = FlapBridge()  # healthy: acks flow
        tmp = tempfile.mkdtemp()
        adapter = WhatsAppAdapter(host="127.0.0.1", port=bridge.port,
                                  outbox_dir=tmp)
        try:
            chat = ChatRef(platform="whatsapp",
                           chat_id="120363012345@g.us",
                           kind=ChatKind.GROUP)
            result = adapter.send(chat, "queued while down")
            self.assertFalse(result.ok)
            rows = adapter.outbox.pending()
            self.assertEqual(len(rows), 1)
            self.assertNotIn("dispatched_at", rows[0])
            _wire(adapter, bridge)
            # The reconnect flush delivers it, exactly once.
            deadline = time.time() + 8
            while adapter.outbox.pending() and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(adapter.outbox.pending(), [])
            self.assertEqual(bridge.send_count("queued while down"), 1)
        finally:
            adapter.stop()
            bridge.close()


if __name__ == "__main__":
    unittest.main()
