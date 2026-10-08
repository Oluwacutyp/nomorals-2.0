"""HA WebSocket event stream: handshake, auth, subscriptions, reconnect,
trigger-engine wiring, proactive alerts.

All tests are offline: a fake Home Assistant server speaks real RFC 6455
framing (server side) in-process, so the client's handshake, masking,
ping/pong and JSON protocol paths are genuinely exercised.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.integrations.smarthome_integration import (
    HAAuthError,
    HAWebSocket,
    HAWebSocketError,
    SmartHomeIntegration,
)
from nomorals.storage.db import Database
from nomorals.triggers import TriggerEngine, validate_definition
from nomorals.triggers.models import OUTCOME_FIRED
from nomorals.triggers.sources import match_entity_state

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ── fake Home Assistant server (real framing, server side) ────────────────

def _srv_encode(text: str) -> bytes:
    data = text.encode("utf-8")
    hdr = bytearray([0x81])  # FIN + text, server frames are unmasked
    n = len(data)
    if n < 126:
        hdr.append(n)
    elif n < 65536:
        hdr.append(126)
        hdr.extend(n.to_bytes(2, "big"))
    else:
        hdr.append(127)
        hdr.extend(n.to_bytes(8, "big"))
    return bytes(hdr) + data


async def _srv_read_frame(reader):
    hdr = await reader.readexactly(2)
    b1, b2 = hdr[0], hdr[1]
    length = b2 & 0x7F
    if length == 126:
        length = int.from_bytes(await reader.readexactly(2), "big")
    elif length == 127:
        length = int.from_bytes(await reader.readexactly(8), "big")
    mask = await reader.readexactly(4) if b2 & 0x80 else None
    payload = await reader.readexactly(length) if length else b""
    if mask:
        payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
    return b1 & 0x0F, payload


class FakeHAServer:
    """In-process Home Assistant WebSocket server."""

    def __init__(self, *, token="good-token", drop_after_auth=False,
                 states=None):
        self.token = token
        self.drop_after_auth = drop_after_auth
        self.states = states or []
        self.connections = 0
        self.received_auths = []
        self._server = None
        self._push_queues = []
        self._n_events = 0

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def stop(self):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}/api/websocket"

    async def push_event(self, entity_id, old_state, new_state,
                         attributes=None):
        self._n_events += 1
        envelope = {
            "id": 1,
            "type": "event",
            "event": {
                "id": f"fake-{self._n_events}",
                "event_type": "state_changed",
                "data": {
                    "entity_id": entity_id,
                    "old_state": {"entity_id": entity_id, "state": old_state,
                                  "attributes": {}},
                    "new_state": {"entity_id": entity_id, "state": new_state,
                                  "attributes": attributes or {}},
                },
            },
        }
        for q in list(self._push_queues):
            await q.put(envelope)

    async def _handle(self, reader, writer):
        self.connections += 1
        try:
            lines = []
            while True:
                line = await reader.readline()
                lines.append(line)
                if line in (b"\r\n", b"\n", b""):
                    break
            key = None
            for line in lines:
                name, _, value = line.decode("latin1").partition(":")
                if name.strip().lower() == "sec-websocket-key":
                    key = value.strip()
            accept = base64.b64encode(
                hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
            writer.write(
                ("HTTP/1.1 101 Switching Protocols\r\n"
                 "Upgrade: websocket\r\n"
                 "Connection: Upgrade\r\n"
                 f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
            await writer.drain()
            # HA greets with auth_required immediately after the handshake
            writer.write(_srv_encode(json.dumps(
                {"type": "auth_required", "ha_version": "2099.1"})))
            await writer.drain()

            _op, payload = await _srv_read_frame(reader)
            auth = json.loads(payload.decode())
            self.received_auths.append(auth.get("access_token"))
            if auth.get("access_token") != self.token:
                writer.write(_srv_encode(json.dumps(
                    {"type": "auth_invalid", "message": "bad token"})))
                await writer.drain()
                writer.close()
                return
            writer.write(_srv_encode(json.dumps(
                {"type": "auth_ok", "ha_version": "2099.1"})))
            await writer.drain()
            if self.drop_after_auth:
                writer.close()
                return

            q = asyncio.Queue()
            self._push_queues.append(q)
            pump = asyncio.create_task(self._pump(writer, q))
            try:
                while True:
                    op, payload = await _srv_read_frame(reader)
                    if op == 0x8:
                        break
                    msg = json.loads(payload.decode())
                    mid = msg.get("id")
                    if msg.get("type") == "subscribe_events":
                        reply = {"id": mid, "type": "result", "success": True,
                                 "result": None}
                    elif msg.get("type") == "get_states":
                        reply = {"id": mid, "type": "result", "success": True,
                                 "result": self.states}
                    else:
                        reply = {"id": mid, "type": "result", "success": False,
                                 "error": {"message": "unknown command"}}
                    writer.write(_srv_encode(json.dumps(reply)))
                    await writer.drain()
            finally:
                pump.cancel()
                try:
                    await pump
                except asyncio.CancelledError:
                    pass
                self._push_queues.remove(q)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _pump(self, writer, q):
        while True:
            item = await q.get()
            writer.write(_srv_encode(json.dumps(item)))
            await writer.drain()


async def _wait_for(cond, timeout=8.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(interval)
    return False


def _states():
    return [
        {"entity_id": "light.kitchen",
         "state": "on",
         "attributes": {"friendly_name": "Kitchen"},
         "last_changed": "2099-01-01T00:00:00"},
        {"entity_id": "light.hall",
         "state": "on",
         "attributes": {"friendly_name": "Hall"},
         "last_changed": "2099-01-01T00:00:00"},
        {"entity_id": "light.porch",
         "state": "on",
         "attributes": {"friendly_name": "Porch"},
         "last_changed": "2099-01-01T00:00:00"},
        {"entity_id": "light.bedroom",
         "state": "off",
         "attributes": {"friendly_name": "Bedroom"},
         "last_changed": "2099-01-01T00:00:00"},
        {"entity_id": "cover.garage_door",
         "state": "open",
         "attributes": {"friendly_name": "Garage Door",
                        "device_class": "garage"},
         "last_changed": "2099-01-01T00:00:00"},
        {"entity_id": "cover.blinds",
         "state": "open",
         "attributes": {"friendly_name": "Blinds"},
         "last_changed": "2099-01-01T00:00:00"},
        {"entity_id": "lock.front_door",
         "state": "unlocked",
         "attributes": {"friendly_name": "Front Door"},
         "last_changed": "2099-01-01T00:00:00"},
    ]


# ── tests ────────────────────────────────────────────────────────────────

class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_auth_and_snapshot(self):
        server = await FakeHAServer(states=_states()).start()
        ws = HAWebSocket(server.url, "good-token")
        try:
            await ws.start()  # must not raise
            self.assertTrue(await _wait_for(lambda: ws.connected),
                            "never connected")
            self.assertTrue(
                await _wait_for(lambda: len(ws.state_snapshot()) == 7),
                "snapshot never populated")
            self.assertEqual(server.received_auths, ["good-token"])
            snap = ws.state_snapshot()
            self.assertEqual(snap["light.kitchen"]["state"], "on")
            self.assertEqual(
                snap["cover.garage_door"]["attributes"]["device_class"],
                "garage")
        finally:
            await ws.stop()
            await server.stop()

    async def test_auth_rejected_stops_quietly(self):
        server = await FakeHAServer().start()
        ws = HAWebSocket(server.url, "bad-token")
        try:
            await ws.start()  # must not raise even for bad auth
            self.assertTrue(await _wait_for(lambda: not ws.running),
                            "kept retrying a rejected token")
            self.assertEqual(server.connections, 1)
            self.assertFalse(ws.connected)
        finally:
            await ws.stop()
            await server.stop()

    async def test_reconnect_on_drop(self):
        server = await FakeHAServer(drop_after_auth=True).start()
        ws = HAWebSocket(server.url, "good-token")
        try:
            await ws.start()
            # first backoff is 1s: a second connection proves the loop
            # survived the drop without raising
            self.assertTrue(await _wait_for(
                lambda: server.connections >= 2, timeout=10),
                "never reconnected after drop")
            self.assertTrue(ws.running)
        finally:
            await ws.stop()
            await server.stop()


class SubscriptionTests(unittest.IsolatedAsyncioTestCase):
    async def _connected(self, server, **kw):
        ws = HAWebSocket(server.url, "good-token", **kw)
        await ws.start()
        self.assertTrue(await _wait_for(lambda: ws.connected))
        return ws

    async def test_state_changed_to_subscription(self):
        server = await FakeHAServer().start()
        ws = await self._connected(server)
        try:
            seen = []
            ws.subscribe_state_changed(
                callback=lambda eid, old, new, attrs: seen.append(
                    (eid, old, new)))
            await server.push_event("light.kitchen", "off", "on",
                                    {"friendly_name": "Kitchen"})
            self.assertTrue(await _wait_for(lambda: len(seen) == 1))
            eid, old, new = seen[0]
            self.assertEqual(eid, "light.kitchen")
            self.assertEqual(old, "off")
            self.assertEqual(new, "on")
        finally:
            await ws.stop()
            await server.stop()

    async def test_domain_filter(self):
        server = await FakeHAServer().start()
        ws = await self._connected(server)
        try:
            seen = []
            ws.subscribe_state_changed(
                domain="light",
                callback=lambda eid, old, new, attrs: seen.append(eid))
            await server.push_event("light.kitchen", "off", "on")
            await server.push_event("switch.porch", "off", "on")
            self.assertTrue(await _wait_for(lambda: len(seen) == 1))
            self.assertEqual(seen, ["light.kitchen"])
        finally:
            await ws.stop()
            await server.stop()

    async def test_entity_id_filter(self):
        server = await FakeHAServer().start()
        ws = await self._connected(server)
        try:
            seen = []
            ws.subscribe_state_changed(
                entity_id="light.kitchen",
                callback=lambda eid, old, new, attrs: seen.append(eid))
            await server.push_event("light.bedroom", "off", "on")
            await server.push_event("light.kitchen", "off", "on")
            self.assertTrue(await _wait_for(lambda: len(seen) == 1))
            self.assertEqual(seen, ["light.kitchen"])
        finally:
            await ws.stop()
            await server.stop()

    async def test_cooldown_suppresses_flapping(self):
        server = await FakeHAServer().start()
        ws = await self._connected(server, cooldown_s=60.0)
        try:
            seen = []
            sub_id = ws.subscribe_state_changed(
                callback=lambda eid, old, new, attrs: seen.append(eid))
            await server.push_event("sensor.temp", "21", "22")
            await server.push_event("sensor.temp", "22", "23")
            await server.push_event("sensor.temp", "23", "24")
            await asyncio.sleep(0.5)
            self.assertEqual(len(seen), 1, f"flapping leaked: {seen}")
            self.assertTrue(ws.unsubscribe(sub_id))
            self.assertFalse(ws.unsubscribe(sub_id))
        finally:
            await ws.stop()
            await server.stop()


class TriggerWiringTests(unittest.IsolatedAsyncioTestCase):
    def _engine(self):
        notified = []
        engine = TriggerEngine(
            Database(":memory:"),
            notify_fn=lambda trig, title, body, eng: notified.append(
                (trig.name, title, body)) or {"ok": True})
        return engine, notified

    async def test_event_fires_entity_state_trigger(self):
        server = await FakeHAServer().start()
        engine, notified = self._engine()
        trigger = engine.add(
            "kitchen light on", "entity_state",
            {"domain": "light", "to": "on"},
            "notify", {"title": "lights", "body": "kitchen is on"})
        ws = HAWebSocket(server.url, "good-token", trigger_engine=engine)
        try:
            await ws.start()
            self.assertTrue(await _wait_for(lambda: ws.connected))
            await server.push_event("light.kitchen", "off", "on")
            self.assertTrue(await _wait_for(
                lambda: any(h["outcome"] == OUTCOME_FIRED
                            for h in engine.history(trigger.id))),
                "trigger never fired")
            self.assertEqual(len(notified), 1)
            self.assertEqual(notified[0][0], "kitchen light on")
        finally:
            await ws.stop()
            await server.stop()

    async def test_non_matching_event_does_not_fire(self):
        server = await FakeHAServer().start()
        engine, notified = self._engine()
        trigger = engine.add(
            "kitchen light on", "entity_state",
            {"domain": "light", "to": "on"},
            "notify", {"title": "lights", "body": "kitchen is on"})
        ws = HAWebSocket(server.url, "good-token", trigger_engine=engine)
        try:
            await ws.start()
            self.assertTrue(await _wait_for(lambda: ws.connected))
            await server.push_event("switch.porch", "off", "on")
            await asyncio.sleep(0.5)
            self.assertEqual(notified, [])
            hist = engine.history(trigger.id)
            self.assertTrue(any(h["outcome"] == "no_match" for h in hist))
        finally:
            await ws.stop()
            await server.stop()

    def test_validate_entity_state(self):
        cond, params, cooldown = validate_definition(
            "entity_state", {"domain": "light", "to": "on"}, "notify", {})
        self.assertEqual(cond, {"domain": "light", "to": "on"})
        cond, _, _ = validate_definition("entity_state", {}, "notify", {})
        self.assertEqual(cond, {})

    def test_match_entity_state(self):
        trig = SimpleNamespace(condition={"domain": "light", "to": "on"})
        hit, ev = match_entity_state(trig, "light.kitchen", "off", "on")
        self.assertTrue(hit)
        self.assertEqual(ev["entity_id"], "light.kitchen")
        hit, ev = match_entity_state(trig, "switch.porch", "off", "on")
        self.assertFalse(hit)
        self.assertEqual(ev["reason"], "domain_mismatch")
        hit, _ = match_entity_state(trig, "light.kitchen", "off", "off")
        self.assertFalse(hit)
        trig2 = SimpleNamespace(
            condition={"entity_id": "lock.front_door", "from": "locked"})
        hit, _ = match_entity_state(trig2, "lock.front_door", "locked",
                                     "unlocked")
        self.assertTrue(hit)
        hit, ev = match_entity_state(trig2, "lock.front_door", "unlocked",
                                     "unlocked")
        self.assertFalse(hit)
        self.assertEqual(ev["reason"], "from_mismatch")
        trig3 = SimpleNamespace(condition={})
        hit, _ = match_entity_state(trig3, "sensor.anything", "1", "2")
        self.assertTrue(hit)


class ProactiveAlertTests(unittest.TestCase):
    def test_alerts(self):
        ws = HAWebSocket("ws://localhost:8123/api/websocket", "x")
        ws._ingest_snapshot(_states())
        alerts = ws.proactive_alerts()
        by_kind = {a.kind: a for a in alerts}
        self.assertEqual(
            by_kind["lights_on"].message,
            "3 lights left on (Hall, Kitchen, Porch) — turn them off?")
        self.assertEqual(by_kind["lights_on"].suggested_action, "turn_off")
        self.assertEqual(
            by_kind["garage_open"].message,
            "garage door is open (Garage Door) — close it?")
        self.assertEqual(by_kind["cover_open"].message,
                         "Blinds is open — close it?")
        self.assertEqual(by_kind["lock_unlocked"].message,
                         "Front Door is unlocked — lock it?")
        self.assertNotIn("light.bedroom",
                         by_kind["lights_on"].entity_ids)

    def test_no_alerts_when_all_quiet(self):
        ws = HAWebSocket("ws://localhost:8123/api/websocket", "x")
        ws._ingest_snapshot([
            {"entity_id": "light.kitchen", "state": "off",
             "attributes": {"friendly_name": "Kitchen"}},
            {"entity_id": "lock.front_door", "state": "locked",
             "attributes": {"friendly_name": "Front Door"}},
        ])
        self.assertEqual(ws.proactive_alerts(), [])


class EventStreamHelperTests(unittest.TestCase):
    def _integration(self, url="http://ha.local:8123", password="tok",
                     raises=False):
        if raises:
            def boom(svc, name):
                raise KeyError("nope")
            am = SimpleNamespace(get_credential=boom)
        else:
            cred = SimpleNamespace(password=password,
                                   metadata={"url": url}, is_active=True)
            am = SimpleNamespace(
                get_credential=lambda svc, name: cred)
        return SmartHomeIntegration(am, None)

    def test_event_stream_builds_websocket(self):
        sm = self._integration()
        ws = sm.event_stream(trigger_engine="ENG")
        self.assertIsInstance(ws, HAWebSocket)
        self.assertEqual(ws._host, "ha.local")
        self.assertEqual(ws._port, 8123)
        self.assertEqual(ws._path, "/api/websocket")
        self.assertEqual(ws._trigger_engine, "ENG")

    def test_event_stream_https_becomes_wss(self):
        sm = self._integration(url="https://ha.example.com:443")
        ws = sm.event_stream()
        self.assertTrue(ws._secure)
        self.assertEqual(ws._port, 443)

    def test_event_stream_no_credential(self):
        sm = self._integration(raises=True)
        with self.assertRaises(HAWebSocketError):
            sm.event_stream()


if __name__ == "__main__":
    unittest.main()
