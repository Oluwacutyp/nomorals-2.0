"""Wave 6: the Z-Wave (RaZberry/Z-Way) backend against its emulator
oracle — a strict fake controller serving the real data-tree shape
and mutating node state on Run/Set commands.

Gate tests assert on the emulator's log: while control is not
owner-enabled, zero Run requests may reach the "controller".
"""
from __future__ import annotations

import pytest

from nomorals.core.errors import (
    AuthError,
    DependencyError,
    NetworkError,
)
from nomorals.integrations.home import ControlGate, RazberryBackend
from nomorals.integrations.home.emulators import (
    ZwaveEmulator,
    ZwaveEmulatorTransport,
)
from nomorals.integrations.home.errors import ControlNotEnabledError
from nomorals.integrations.home.zwave import HttpStatusError

NODES = [
    {"id": 2, "name": "Hall Switch", "switch_binary": False},
    {"id": 3, "name": "Lounge Dimmer", "level": 40},
    {"id": 4, "name": "Porch Sensor", "sensors": [
        {"type": "binary", "value": True},
        {"type": "temperature", "value": 21.5, "scale": "°C"},
    ]},
]


def make_backend(emulator=None, gate=None, **kw):
    emulator = emulator or ZwaveEmulator(NODES)
    kw.setdefault("base_url", "http://razberry.local:8083")
    backend = RazberryBackend(
        gate=gate or ControlGate(),
        transport=ZwaveEmulatorTransport(emulator), **kw)
    return backend, emulator


class TestDiscovery:
    def test_models_devices_and_excludes_controller(self):
        backend, _ = make_backend()
        devices = {d.device_id: d for d in backend.discover()}
        assert set(devices) == {"zwave-2", "zwave-3", "zwave-4"}
        assert devices["zwave-2"].device_type == "switch"
        assert devices["zwave-3"].device_type == "light"
        assert devices["zwave-4"].device_type == "sensor"
        assert devices["zwave-4"].capabilities == ()

    def test_discovery_is_read_only(self):
        backend, emulator = make_backend()
        backend.discover()
        assert emulator.control_calls() == []
        assert all(e["kind"] == "data" for e in emulator.log)

    def test_unconfigured_controller_is_dependency_failed(self):
        backend = RazberryBackend(gate=ControlGate(), base_url="")
        with pytest.raises(DependencyError) as ctx:
            backend.discover()
        assert ctx.value.code == "dependency.failed"
        assert ctx.value.details["reason"] == "not_configured"

    def test_unreachable_controller_is_network_error(self):
        class DownTransport:
            def get(self, url, headers, timeout):
                raise OSError("connection refused")

        backend = RazberryBackend(
            gate=ControlGate(), transport=DownTransport(),
            base_url="http://razberry.local:8083")
        with pytest.raises(NetworkError) as ctx:
            backend.discover()
        assert ctx.value.code == "network.error"

    def test_auth_rejection_is_auth_error(self):
        class AuthTransport:
            def get(self, url, headers, timeout):
                raise HttpStatusError(401, url)

        backend = RazberryBackend(
            gate=ControlGate(), transport=AuthTransport(),
            base_url="http://razberry.local:8083")
        with pytest.raises(AuthError):
            backend.discover()

    def test_garbage_json_is_dependency_failed(self):
        class GarbageTransport:
            def get(self, url, headers, timeout):
                return "<html>not the controller</html>"

        backend = RazberryBackend(
            gate=ControlGate(), transport=GarbageTransport(),
            base_url="http://razberry.local:8083")
        with pytest.raises(DependencyError) as ctx:
            backend.discover()
        assert ctx.value.details["reason"] == "protocol_parse"


class TestReads:
    def test_sensor_values(self):
        backend, _ = make_backend()
        backend.discover()
        state = backend.read_state("zwave-4")
        kinds = {s["kind"] for s in state["sensors"]}
        assert kinds == {"binary", "multilevel"}
        temp = next(s for s in state["sensors"]
                    if s["kind"] == "multilevel")
        assert temp["value"] == 21.5
        assert temp["scale"] == "°C"

    def test_switch_state(self):
        backend, _ = make_backend()
        backend.discover()
        assert backend.read_state("zwave-2")["on"] is False
        assert backend.read_state("zwave-3")["level"] == 40


class TestGateAndControl:
    def test_control_refused_by_default_and_never_reaches_controller(
            self):
        backend, emulator = make_backend()
        backend.discover()
        with pytest.raises(ControlNotEnabledError) as ctx:
            backend.control("zwave-2", "turn_on")
        assert ctx.value.code == "home.control_not_enabled"
        assert emulator.control_calls() == []
        assert emulator.node_level(2)["switch_binary"] is False

    def test_enabled_turn_on_actuates_and_reads_back(self):
        backend, emulator = make_backend()
        backend.discover()
        backend._gate.set_enabled("zwave", "zwave-2", True, by="owner")
        result = backend.control("zwave-2", "turn_on")
        assert result["state"]["on"] is True
        assert emulator.node_level(2)["switch_binary"] is True
        assert len(emulator.control_calls()) == 1

    def test_level_saturates_at_99(self):
        # The emulator rejects levels >99 (except 255=restore) like a
        # real controller; the client must clamp 0–99 and never send
        # an implicit 255.
        backend, emulator = make_backend()
        backend.discover()
        backend._gate.set_enabled("zwave", "zwave-3", True, by="owner")
        result = backend.control("zwave-3", "set_level", value=250)
        assert result["state"]["level"] == 99
        sent = emulator.control_calls()
        assert sent and sent[0]["value"] == 99

    def test_toggle(self):
        backend, emulator = make_backend()
        backend.discover()
        backend._gate.set_enabled("zwave", "zwave-3", True, by="owner")
        result = backend.control("zwave-3", "toggle")
        assert result["state"]["level"] == 0
        assert result["state"]["on"] is False

    def test_sensor_control_is_named_refusal_even_when_enabled(self):
        backend, emulator = make_backend()
        backend.discover()
        backend._gate.set_enabled("zwave", "zwave-4", True, by="owner")
        with pytest.raises(DependencyError) as ctx:
            backend.control("zwave-4", "turn_on")
        assert ctx.value.details["reason"] == "no_switch_class"
        assert emulator.control_calls() == []

    def test_capabilities_are_comparable_data(self):
        backend, _ = make_backend()
        caps = backend.capabilities().to_dict()
        assert caps["backend"] == "zwave"
        assert caps["control_gated"] is True
        assert "set_level" in caps["controllable"]
