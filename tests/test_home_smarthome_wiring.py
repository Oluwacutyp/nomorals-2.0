"""Wave 6: SmartHomeIntegration wiring for the native backends.

Pins the retired anti-pattern (no backend → named ``dependency.failed``,
never a silent ``[]``), the native routing for discovery/status/light
control, and the owner gate surface on the integration itself: control
flows only after ``enable_device_control``, and the refusal that comes
back names the gate — with the emulator proving nothing reached the
device while it was off.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nomorals.core.errors import DependencyError
from nomorals.integrations.home import ControlGate, WemoBackend
from nomorals.integrations.home.emulators import (
    WemoEmulator,
    WemoEmulatorTransport,
)
from nomorals.integrations.home.errors import ControlNotEnabledError
from nomorals.integrations.smarthome_integration import (
    Device,
    SmartHomeIntegration,
)

PLUG = {"host": "192.168.1.50:49152", "name": "Kitchen Plug",
        "model": "Socket", "serial": "AAA1", "binary_state": 0}
LAMP = {"host": "192.168.1.51:49152", "name": "Desk Lamp",
        "model": "Dimmer", "serial": "BBB2", "binary_state": 1,
        "brightness": 128}


def run(coro):
    return asyncio.run(coro)


def no_credential_manager():
    manager = MagicMock()
    manager.get_credential.side_effect = KeyError("not configured")
    return manager


def make_integration(emulator=None, gate=None):
    emulator = emulator or WemoEmulator([PLUG, LAMP])
    gate = gate or ControlGate()
    backend = WemoBackend(gate=gate,
                          transport=WemoEmulatorTransport(emulator))
    integration = SmartHomeIntegration(
        no_credential_manager(), MagicMock(),
        wemo_backend=backend, control_gate=gate)
    return integration, emulator, gate


class TestNoBackend:
    def test_list_devices_raises_named_error_not_empty_list(self):
        integration = SmartHomeIntegration(
            no_credential_manager(), MagicMock())
        with pytest.raises(DependencyError) as ctx:
            run(integration.list_devices())
        assert ctx.value.code == "dependency.failed"
        assert ctx.value.details["reason"] == "no_backend_configured"

    def test_status_with_no_backend_still_reports(self):
        integration = SmartHomeIntegration(
            no_credential_manager(), MagicMock())
        status = integration.home_status()
        names = [b["backend"] for b in status["backends"]]
        assert names == ["homeassistant"]
        assert status["backends"][0]["configured"] is False
        assert status["control_gate"] == []


class TestNativeRouting:
    def test_list_devices_merges_native_devices(self):
        integration, _, _ = make_integration()
        devices = run(integration.list_devices())
        assert {d.name for d in devices} == {"Kitchen Plug", "Desk Lamp"}
        assert all(isinstance(d, Device) for d in devices)
        lamp = next(d for d in devices if d.name == "Desk Lamp")
        assert lamp.device_type == "light"
        assert "set_brightness" in lamp.capabilities

    def test_get_device_status_maps_native_state(self):
        integration, _, _ = make_integration()
        run(integration.list_devices())
        status = run(integration.get_device_status("Desk Lamp"))
        assert status.is_on is True
        assert status.brightness == 50  # 128/255 ≈ 50%

    def test_set_light_refused_by_gate_and_device_untouched(self):
        integration, emulator, _ = make_integration()
        run(integration.list_devices())
        with pytest.raises(ControlNotEnabledError) as ctx:
            run(integration.set_light("Kitchen Plug", on=True))
        assert ctx.value.code == "home.control_not_enabled"
        assert emulator.control_calls() == []
        assert emulator.device_state(PLUG["host"])["binary_state"] == 0

    def test_owner_enable_then_set_light_works(self):
        integration, emulator, _ = make_integration()
        devices = run(integration.list_devices())
        plug = next(d for d in devices if d.name == "Kitchen Plug")
        record = integration.enable_device_control(
            "wemo", plug.device_id, note="owner said yes")
        assert record["enabled"] is True
        assert run(integration.set_light("Kitchen Plug", on=True)) is True
        assert emulator.device_state(PLUG["host"])["binary_state"] == 1

    def test_disable_revokes_again(self):
        integration, emulator, _ = make_integration()
        devices = run(integration.list_devices())
        plug = next(d for d in devices if d.name == "Kitchen Plug")
        integration.enable_device_control("wemo", plug.device_id)
        integration.disable_device_control("wemo", plug.device_id)
        with pytest.raises(ControlNotEnabledError):
            run(integration.set_light("Kitchen Plug", on=True))
        assert emulator.control_calls() == []

    def test_brightness_routes_and_scales(self):
        integration, emulator, _ = make_integration()
        devices = run(integration.list_devices())
        lamp = next(d for d in devices if d.name == "Desk Lamp")
        integration.enable_device_control("wemo", lamp.device_id)
        assert run(integration.set_light(
            "Desk Lamp", brightness=100)) is True
        assert emulator.device_state(LAMP["host"])["brightness"] == 255

    def test_color_on_native_is_named_refusal(self):
        integration, _, _ = make_integration()
        devices = run(integration.list_devices())
        lamp = next(d for d in devices if d.name == "Desk Lamp")
        integration.enable_device_control("wemo", lamp.device_id)
        with pytest.raises(DependencyError) as ctx:
            run(integration.set_light("Desk Lamp", color="#FF0000"))
        assert ctx.value.details["reason"] == "capability_unsupported"

    def test_toggle_native(self):
        integration, emulator, _ = make_integration()
        devices = run(integration.list_devices())
        lamp = next(d for d in devices if d.name == "Desk Lamp")
        integration.enable_device_control("wemo", lamp.device_id)
        assert run(integration.toggle("Desk Lamp")) is False  # was on
        assert emulator.device_state(LAMP["host"])["binary_state"] == 0

    def test_home_status_reports_devices_and_gate(self):
        integration, _, _ = make_integration()
        run(integration.list_devices())
        status = integration.home_status()
        wemo = next(b for b in status["backends"]
                    if b["backend"] == "wemo")
        assert wemo["devices_known"] == 2
        assert all(d["control_enabled"] is False
                   for d in wemo["devices"])


class TestHAPathUntouched:
    def test_ha_detection_still_wins_for_list_devices(self):
        manager = MagicMock()
        cred = SimpleNamespace(is_active=True,
                               metadata={"url": "http://ha:8123"},
                               password="tok")
        manager.get_credential.return_value = cred
        integration = SmartHomeIntegration(manager, MagicMock())
        sentinel = [Device("light.kitchen", "Kitchen", "light")]

        async def fake_ha_list(room, device_type):
            return sentinel

        integration._list_devices_ha = fake_ha_list
        assert run(integration.list_devices()) == sentinel
