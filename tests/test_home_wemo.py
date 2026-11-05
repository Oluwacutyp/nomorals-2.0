"""Wave 6: the WeMo backend against its emulator oracle.

The emulator is a strict, independent implementation of the WeMo wire
protocol (SSDP + setup.xml + SOAP with faulting) — the backend is
proven against a fake that *verifies*, not a mock that agrees.  The
gate tests assert on the emulator's log: while control is not
owner-enabled, zero Set* actions may reach the "device".
"""
from __future__ import annotations

import pytest

from nomorals.core.errors import (
    DependencyError,
    NetworkError,
    ValidationError,
)
from nomorals.integrations.home import ControlGate, WemoBackend
from nomorals.integrations.home.emulators import (
    WemoEmulator,
    WemoEmulatorTransport,
)
from nomorals.integrations.home.errors import (
    ControlNotEnabledError,
    HomeDeviceNotFoundError,
)

PLUG = {"host": "192.168.1.50:49152", "name": "Kitchen Plug",
        "model": "Socket", "serial": "AAA1", "binary_state": 0}
LAMP = {"host": "192.168.1.51:49152", "name": "Desk Lamp",
        "model": "Dimmer", "serial": "BBB2", "binary_state": 1,
        "brightness": 128}
HEATER = {"host": "192.168.1.52:49152", "name": "Heater",
          "model": "Insight", "serial": "CCC3", "binary_state": 1,
          "power_mw": 1500000}


def make_backend(emulator=None, gate=None, **kw):
    emulator = emulator or WemoEmulator([PLUG, LAMP, HEATER])
    backend = WemoBackend(
        gate=gate or ControlGate(),
        transport=WemoEmulatorTransport(emulator), **kw)
    return backend, emulator


class TestDiscovery:
    def test_discovers_all_devices_with_types_and_caps(self):
        backend, _ = make_backend()
        devices = {d.name: d for d in backend.discover()}
        assert set(devices) == {"Kitchen Plug", "Desk Lamp", "Heater"}
        assert devices["Kitchen Plug"].device_type == "switch"
        assert devices["Desk Lamp"].device_type == "light"
        assert "set_brightness" in devices["Desk Lamp"].capabilities
        assert "set_brightness" not in \
            devices["Kitchen Plug"].capabilities

    def test_discovery_is_read_only(self):
        backend, emulator = make_backend()
        backend.discover()
        for device in backend.known_devices():
            backend.read_state(device.device_id)
        assert emulator.control_calls() == []
        assert emulator.device_state(PLUG["host"])["binary_state"] == 0

    def test_blackholed_ssdp_ends_in_named_network_error(self):
        emulator = WemoEmulator([PLUG])
        emulator.ssdp_blackhole = True
        backend, _ = make_backend(emulator)
        with pytest.raises(NetworkError) as ctx:
            backend.discover()
        assert ctx.value.code == "network.error"
        assert ctx.value.details["ladder"]  # the ladder is recorded

    def test_blackholed_ssdp_falls_back_to_known_hosts(self):
        emulator = WemoEmulator([PLUG])
        emulator.ssdp_blackhole = True
        backend, _ = make_backend(
            emulator, known_hosts=("192.168.1.50:49152",))
        devices = backend.discover()
        assert [d.name for d in devices] == ["Kitchen Plug"]
        assert any("known hosts" in note
                   for note in backend.last_discovery_notes)

    def test_ssdp_stack_failure_is_dependency_failed(self):
        class DeadTransport:
            def ssdp_search(self, st, timeout):
                raise OSError("multicast unavailable")

            def http_get(self, url, timeout):
                raise OSError("down")

            def http_post(self, url, body, headers, timeout):
                raise OSError("down")

        backend = WemoBackend(gate=ControlGate(),
                              transport=DeadTransport())
        with pytest.raises(DependencyError) as ctx:
            backend.discover()
        assert ctx.value.code == "dependency.failed"

    def test_malformed_description_is_dependency_failed(self):
        emulator = WemoEmulator([PLUG])
        emulator.description_override = b"<root><oops"
        backend, _ = make_backend(emulator)
        with pytest.raises(DependencyError) as ctx:
            backend.discover()
        assert ctx.value.code == "dependency.failed"


class TestReads:
    def test_binary_state_and_insight_power(self):
        backend, _ = make_backend()
        backend.discover()
        heater = next(d for d in backend.known_devices()
                      if d.name == "Heater")
        state = backend.read_state(heater.device_id)
        assert state["on"] is True
        assert state["current_power_mw"] == 1500000

    def test_dimmer_brightness(self):
        backend, _ = make_backend()
        backend.discover()
        lamp = next(d for d in backend.known_devices()
                    if d.name == "Desk Lamp")
        assert backend.read_state(lamp.device_id)["brightness"] == 128

    def test_unknown_device_is_named_not_found(self):
        backend, _ = make_backend()
        backend.discover()
        with pytest.raises(HomeDeviceNotFoundError) as ctx:
            backend.read_state("uuid:nope")
        assert ctx.value.code == "home.device_not_found"


class TestGateAndControl:
    def test_control_refused_by_default_and_never_reaches_device(self):
        backend, emulator = make_backend()
        plug = backend.discover()[0]
        with pytest.raises(ControlNotEnabledError) as ctx:
            backend.control(plug.device_id, "turn_on")
        assert ctx.value.code == "home.control_not_enabled"
        assert ctx.value.details["device_id"] == plug.device_id
        # The oracle's assertion: no Set* action was even attempted.
        assert emulator.control_calls() == []
        assert emulator.device_state(PLUG["host"])["binary_state"] == 0

    def test_enabled_control_actuates_and_reads_back_truth(self):
        backend, emulator = make_backend()
        plug = backend.discover()[0]
        backend._gate.set_enabled("wemo", plug.device_id, True,
                                   by="owner")
        result = backend.control(plug.device_id, "turn_on")
        assert result["state"]["on"] is True
        assert emulator.device_state(PLUG["host"])["binary_state"] == 1
        assert len(emulator.control_calls()) == 1

    def test_toggle_flips_from_device_truth(self):
        backend, emulator = make_backend()
        backend.discover()
        lamp = next(d for d in backend.known_devices()
                    if d.name == "Desk Lamp")
        backend._gate.set_enabled("wemo", lamp.device_id, True,
                                   by="owner")
        result = backend.control(lamp.device_id, "toggle")
        assert result["state"]["on"] is False  # was on

    def test_brightness_saturates_instead_of_faulting(self):
        # The emulator faults on Brightness > 255 like a real dimmer;
        # the client must clamp (point.rs saturating arithmetic).
        backend, emulator = make_backend()
        backend.discover()
        lamp = next(d for d in backend.known_devices()
                    if d.name == "Desk Lamp")
        backend._gate.set_enabled("wemo", lamp.device_id, True,
                                   by="owner")
        result = backend.control(lamp.device_id, "set_brightness",
                                 value=999)
        assert result["state"]["brightness"] == 255
        assert emulator.device_state(LAMP["host"])["brightness"] == 255

    def test_brightness_on_non_dimmer_is_named_refusal(self):
        backend, _ = make_backend()
        plug = backend.discover()[0]
        backend._gate.set_enabled("wemo", plug.device_id, True,
                                   by="owner")
        with pytest.raises(DependencyError) as ctx:
            backend.control(plug.device_id, "set_brightness", value=10)
        assert ctx.value.details["reason"] == "no_brightness_service"

    def test_undeclared_action_is_validation_error(self):
        backend, _ = make_backend()
        plug = backend.discover()[0]
        with pytest.raises(ValidationError):
            backend.control(plug.device_id, "self_destruct")

    def test_capabilities_are_comparable_data(self):
        backend, _ = make_backend()
        caps = backend.capabilities().to_dict()
        assert caps["backend"] == "wemo"
        assert caps["control_gated"] is True
        assert "turn_on" in caps["controllable"]
        status = backend.status()
        assert status["capabilities"] == caps


class TestEmulatorStrictness:
    """The oracle verifies: a hand-built wrong request must fault."""

    def test_soapaction_mismatch_faults(self):
        emulator = WemoEmulator([PLUG])
        body = (
            '<?xml version="1.0"?>'
            '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
            '<s:Body><u:SetBinaryState '
            'xmlns:u="urn:Belkin:service:basicevent:1">'
            "<BinaryState>1</BinaryState></u:SetBinaryState>"
            "</s:Body></s:Envelope>").encode()
        status, payload = emulator.handle_http(
            "POST",
            "http://192.168.1.50:49152/upnp/control/basicevent1",
            {"SOAPACTION":
             '"urn:Belkin:service:basicevent:1#GetBinaryState"'},
            body)
        assert status == 500
        assert b"does not match" in payload
        assert emulator.device_state(PLUG["host"])["binary_state"] == 0

    def test_bad_binary_state_value_faults(self):
        emulator = WemoEmulator([PLUG])
        body = (
            '<?xml version="1.0"?>'
            '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
            '<s:Body><u:SetBinaryState '
            'xmlns:u="urn:Belkin:service:basicevent:1">'
            "<BinaryState>7</BinaryState></u:SetBinaryState>"
            "</s:Body></s:Envelope>").encode()
        status, payload = emulator.handle_http(
            "POST",
            "http://192.168.1.50:49152/upnp/control/basicevent1",
            {"SOAPACTION":
             '"urn:Belkin:service:basicevent:1#SetBinaryState"'},
            body)
        assert status == 500
        assert b"bad BinaryState" in payload
