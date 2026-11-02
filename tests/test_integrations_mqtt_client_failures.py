"""Tail-105 pins: nomorals/integrations/mqtt_client.py census sites.

All four sites are held as exclusions. The class contract is
printed on its own methods: ``start``/``stop``/``_loop`` are
"Never raises" — the bridge is a background daemon thread, so
a raise would kill smarthome state delivery silently instead.

223 — ``stop()``: ``client.loop_stop()`` crash is swallowed so
the ``disconnect()`` that follows still runs. Teardown, best
effort, no consumer for a failure value.

227 — ``stop()``: ``client.disconnect()`` crash swallowed at
the end of teardown; ``_connected`` is still reset in the
``finally`` so the state surface stays honest.

424 — ``_serve_once``: ``client.loop_start()`` crash after a
successful ``connect()`` is swallowed and the serve cycle
parks in its wait loop until ``stop()``. WEAKEST EXCLUSION
of this batch: every sibling failure in the module logs
(connect failure at info, serve-cycle failure at debug,
resubscribe at debug) — this one is silent, and because the
swallow keeps ``_serve_once`` from raising, the supervisor's
backoff/reconnect never engages for this failure shape; the
bridge sits down-but-parked. Held anyway: the failure is NOT
success-shaped — ``_connected`` is set only by the
``on_connect`` callback, which a dead loop never fires, so
``connected``/``stats()`` report the bridge down truthfully
throughout; the never-raises contract is explicit; and a
paho ``loop_start`` raise after a successful connect is a
narrow corner (the loop thread failed to spawn). The fix is
a log line — observability, shelved as a sketch, not a
convert.

432 — ``_serve_once``: ``client.loop_stop()`` crash at cycle
teardown swallowed; ``_client``/``_connected`` are reset
unconditionally right after.

NOT-recording is pinned with the snapshot method: the module
logs but never calls a failures-ledger record function, so
both ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

import threading
import time

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.integrations.mqtt_client import MQTTBridge


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


class _FakeClient:
    """paho-shaped stand-in recording every call."""

    def __init__(self, *, loop_start_exc=None, loop_stop_exc=None,
                 disconnect_exc=None, fire_disconnect_on_start=False):
        self.calls = []
        self._lse = loop_start_exc
        self._lste = loop_stop_exc
        self._de = disconnect_exc
        self._fire = fire_disconnect_on_start
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None

    def will_set(self, *a, **k):
        self.calls.append("will_set")

    def connect(self, *a, **k):
        self.calls.append("connect")

    def loop_start(self):
        self.calls.append("loop_start")
        if self._lse is not None:
            raise self._lse
        if self._fire and self.on_disconnect is not None:
            self.on_disconnect(self, None, None, 0)

    def loop_stop(self):
        self.calls.append("loop_stop")
        if self._lste is not None:
            raise self._lste

    def disconnect(self):
        self.calls.append("disconnect")
        if self._de is not None:
            raise self._de

    def subscribe(self, *a, **k):
        self.calls.append("subscribe")

    def publish(self, *a, **k):
        self.calls.append("publish")


def _bridge(fake):
    return MQTTBridge(client_factory=lambda: fake)


# ── 223: stop() loop_stop crash -> disconnect still runs ────────

def test_223_stop_survives_loop_stop_failure():
    before = _snap()
    fake = _FakeClient(loop_stop_exc=RuntimeError("loop_stop boom"))
    bridge = _bridge(fake)
    bridge._client = fake
    bridge._connected = True
    assert bridge.stop() is None
    assert "loop_stop" in fake.calls
    assert "disconnect" in fake.calls  # the swallow did not skip it
    assert bridge.connected is False
    assert _snap() == before


# ── 227: stop() disconnect crash -> state still reset ───────────

def test_227_stop_survives_disconnect_failure():
    before = _snap()
    fake = _FakeClient(disconnect_exc=RuntimeError("disconnect boom"))
    bridge = _bridge(fake)
    bridge._client = fake
    bridge._connected = True
    assert bridge.stop() is None
    assert "disconnect" in fake.calls
    assert bridge.connected is False
    assert _snap() == before


# ── 424 (+432): loop_start crash -> parked, honest, stoppable ──

def test_424_loop_start_failure_parks_without_claiming_connection():
    before = _snap()
    fake = _FakeClient(loop_start_exc=RuntimeError("no loop thread"),
                       loop_stop_exc=RuntimeError("loop_stop boom"))
    bridge = _bridge(fake)
    bridge._running = True
    errors = []

    def serve():
        try:
            bridge._serve_once()
        except Exception as exc:  # noqa: BLE001 - pin asserts none escape
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    time.sleep(0.3)
    # parked in the serve wait loop, and never claimed a connection
    assert thread.is_alive()
    assert bridge.connected is False
    assert "loop_start" in fake.calls
    # stop() still tears the parked cycle down cleanly
    bridge.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert "loop_stop" in fake.calls  # teardown attempted despite raising
    assert bridge._client is None
    assert bridge.connected is False
    assert _snap() == before


# ── 432: loop_stop crash at cycle teardown is swallowed ────────

def test_432_serve_teardown_survives_loop_stop_failure():
    before = _snap()
    fake = _FakeClient(loop_stop_exc=RuntimeError("loop_stop boom"),
                       fire_disconnect_on_start=True)
    bridge = _bridge(fake)
    bridge._running = True
    assert bridge._serve_once() is None
    assert "loop_stop" in fake.calls
    assert bridge._client is None
    assert bridge.connected is False
    assert _snap() == before
