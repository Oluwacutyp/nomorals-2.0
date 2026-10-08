"""Offline tests for the MQTT bridge (#63).  No broker, no paho needed."""

import json
import threading
import time

import pytest

from nomorals.integrations import mqtt_client as mc
from nomorals.integrations.mqtt_client import (
    MQTTBridge,
    availability_topic,
    get_topic,
    set_topic,
    state_topic,
)


class FakeMsg:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload


class FakeClient:
    """paho-shaped fake: synchronous, fully controllable."""

    instances: list = []

    def __init__(self, *, connect_rc=0, fail_connect=False):
        self.connect_rc = connect_rc
        self.fail_connect = fail_connect
        self.connect_calls = 0
        self.subscribed: list[str] = []
        self.published: list[tuple] = []
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None
        FakeClient.instances.append(self)

    def connect(self, host, port, keepalive=60):
        self.connect_calls += 1
        if self.fail_connect:
            raise ConnectionRefusedError("no broker here")
        if self.on_connect:
            self.on_connect(self, None, None, self.connect_rc)

    def disconnect(self):
        if self.on_disconnect:
            self.on_disconnect(self, None, None, 0)

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def subscribe(self, topic, qos=0):
        self.subscribed.append(topic)

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))

    # test driver: simulate an inbound broker message
    def incoming(self, topic, payload):
        if self.on_message:
            raw = payload if isinstance(payload, bytes) else (
                json.dumps(payload) if not isinstance(payload, str) else payload
            )
            self.on_message(self, None, FakeMsg(topic, raw))

    def drop(self):
        """Simulate a broker-side disconnect."""
        self.on_disconnect(self, None, None, 7)


@pytest.fixture(autouse=True)
def _reset():
    FakeClient.instances.clear()
    yield
    FakeClient.instances.clear()


def _bridge(**kw):
    kw.setdefault("client_factory", lambda: FakeClient())
    b = MQTTBridge(**kw)
    b._backoff_s = 0.01  # fast reconnects in tests
    return b


def _wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ── topic grammar ────────────────────────────────────────────────────

def test_topic_grammar():
    assert state_topic("kitchen_light") == "zigbee2mqtt/kitchen_light"
    assert set_topic("kitchen_light") == "zigbee2mqtt/kitchen_light/set"
    assert get_topic("kitchen_light") == "zigbee2mqtt/kitchen_light/get"
    assert availability_topic("kitchen_light") == \
        "zigbee2mqtt/kitchen_light/availability"


def test_custom_prefix():
    assert state_topic("lamp", prefix="home") == "home/lamp"


# ── connect / subscribe / publish ────────────────────────────────────

def test_connect_and_subscribe():
    b = _bridge()
    b.start()
    try:
        assert _wait(lambda: b.connected), "bridge should connect"
        client = FakeClient.instances[-1]
        assert "zigbee2mqtt/#" in client.subscribed
    finally:
        b.stop()


def test_start_never_raises_without_broker():
    b = _bridge(client_factory=lambda: FakeClient(fail_connect=True))
    b.start()  # must not raise
    try:
        time.sleep(0.2)
        assert not b.connected
    finally:
        b.stop()


def test_subscribe_callback_receives_messages():
    b = _bridge()
    b.start()
    try:
        assert _wait(lambda: b.connected)
        seen = []
        b.subscribe("zigbee2mqtt/sensor1", lambda t, p: seen.append((t, p)))
        FakeClient.instances[-1].incoming("zigbee2mqtt/sensor1",
                                           {"temperature": 24.5})
        assert _wait(lambda: seen), "subscriber should get the message"
        assert seen[0][1]["temperature"] == 24.5
    finally:
        b.stop()


def test_subscribe_wildcards():
    b = _bridge()
    assert MQTTBridge._topic_matches("zigbee2mqtt/#", "zigbee2mqtt/a/b")
    assert MQTTBridge._topic_matches("zigbee2mqtt/+", "zigbee2mqtt/a")
    assert not MQTTBridge._topic_matches("zigbee2mqtt/+", "zigbee2mqtt/a/b")
    assert not MQTTBridge._topic_matches("other/#", "zigbee2mqtt/a")


# ── retained state cache ─────────────────────────────────────────────

def test_retained_cache():
    b = _bridge(cooldown_s=0)
    b.start()
    try:
        assert _wait(lambda: b.connected)
        FakeClient.instances[-1].incoming(
            "zigbee2mqtt/kitchen_light",
            {"state": "ON", "brightness": 200})
        assert _wait(lambda: b.device_state("kitchen_light") is not None)
        st = b.device_state("kitchen_light")
        assert st["state"] == "ON"
        assert st["attributes"]["brightness"] == 200
        assert b.device_state("never_seen") is None
    finally:
        b.stop()


def test_plain_text_state():
    b = _bridge(cooldown_s=0)
    b.start()
    try:
        assert _wait(lambda: b.connected)
        FakeClient.instances[-1].incoming("zigbee2mqtt/door", "open")
        assert _wait(lambda: b.device_state("door") is not None)
        assert b.device_state("door")["state"] == "open"
    finally:
        b.stop()


def test_availability_tracked():
    b = _bridge(cooldown_s=0)
    b.start()
    try:
        assert _wait(lambda: b.connected)
        FakeClient.instances[-1].incoming(
            "zigbee2mqtt/sensor1/availability", "offline")
        assert _wait(lambda: b.availability("sensor1") == "offline")
    finally:
        b.stop()


# ── trigger engine routing (entity_state source) ─────────────────────

class FakeEngine:
    def __init__(self):
        self.events = []

    def on_entity_state(self, entity_id, old_state, new_state, attributes=None):
        self.events.append((entity_id, old_state, new_state,
                            dict(attributes or {})))
        return []


def test_trigger_engine_receives_updates():
    engine = FakeEngine()
    b = _bridge(cooldown_s=0, trigger_engine=engine)
    b.start()
    try:
        assert _wait(lambda: b.connected)
        c = FakeClient.instances[-1]
        c.incoming("zigbee2mqtt/kitchen_light", {"state": "OFF"})
        c.incoming("zigbee2mqtt/kitchen_light", {"state": "ON"})
        assert _wait(lambda: len(engine.events) >= 2)
        eid, old, new, _attrs = engine.events[-1]
        assert eid == "zigbee2mqtt.kitchen_light"
        assert old == "OFF" and new == "ON"
    finally:
        b.stop()


def test_cooldown_suppresses_flapping():
    engine = FakeEngine()
    now = [1000.0]
    b = _bridge(cooldown_s=60.0, trigger_engine=engine,
                clock=lambda: now[0])
    b.start()
    try:
        assert _wait(lambda: b.connected)
        c = FakeClient.instances[-1]
        c.incoming("zigbee2mqtt/pir", {"state": "ON"})
        now[0] += 1.0
        c.incoming("zigbee2mqtt/pir", {"state": "OFF"})
        now[0] += 1.0
        c.incoming("zigbee2mqtt/pir", {"state": "ON"})
        time.sleep(0.2)
        assert len(engine.events) == 1, "flapping within cooldown suppressed"
        now[0] += 61.0  # cooldown expired
        c.incoming("zigbee2mqtt/pir", {"state": "OFF"})
        assert _wait(lambda: len(engine.events) == 2)
    finally:
        b.stop()


def test_engine_failure_never_breaks_bridge():
    class BadEngine:
        def on_entity_state(self, *a, **k):
            raise RuntimeError("boom")

    b = _bridge(cooldown_s=0, trigger_engine=BadEngine())
    b.start()
    try:
        assert _wait(lambda: b.connected)
        FakeClient.instances[-1].incoming("zigbee2mqtt/x", {"state": "ON"})
        assert _wait(lambda: b.device_state("x") is not None)
    finally:
        b.stop()


# ── device commands ──────────────────────────────────────────────────

def test_set_state_publishes_to_set_topic():
    b = _bridge()
    b.start()
    try:
        assert _wait(lambda: b.connected)
        assert b.set_state("kitchen_light", {"state": "ON", "brightness": 150})
        client = FakeClient.instances[-1]
        topics = [p[0] for p in client.published]
        assert "zigbee2mqtt/kitchen_light/set" in topics
        payload = json.loads(
            [p[1] for p in client.published
             if p[0] == "zigbee2mqtt/kitchen_light/set"][0])
        assert payload["state"] == "ON"
    finally:
        b.stop()


def test_request_state_publishes_to_get_topic():
    b = _bridge()
    b.start()
    try:
        assert _wait(lambda: b.connected)
        assert b.request_state("kitchen_light")
        topics = [p[0] for p in FakeClient.instances[-1].published]
        assert "zigbee2mqtt/kitchen_light/get" in topics
    finally:
        b.stop()


def test_publish_fails_closed_when_down():
    b = _bridge(client_factory=lambda: FakeClient(fail_connect=True))
    b.start()
    try:
        time.sleep(0.2)
        assert b.publish("x/y", "z") is False
        assert b.set_state("lamp", "ON") is False
    finally:
        b.stop()


# ── reconnect ────────────────────────────────────────────────────────

def test_auto_reconnect_with_backoff():
    b = _bridge()
    b.start()
    try:
        assert _wait(lambda: b.connected)
        FakeClient.instances[-1].drop()  # broker-side disconnect
        assert _wait(lambda: len(FakeClient.instances) >= 2,
                     timeout=5.0), "bridge should reconnect"
        assert _wait(lambda: b.connected, timeout=5.0)
    finally:
        b.stop()


# ── fail closed without paho ─────────────────────────────────────────

def test_fail_closed_without_paho(monkeypatch):
    monkeypatch.setattr(mc, "_have_paho", lambda: False)
    b = MQTTBridge()  # no factory — real path
    b.start()  # never raises
    try:
        time.sleep(0.3)
        assert not b.connected
        assert b.publish("x", "y") is False
        assert b.set_state("lamp", "ON") is False
    finally:
        b.stop()


def test_recipe_mentions_paho():
    assert "paho-mqtt" in mc.PAHO_RECIPE
