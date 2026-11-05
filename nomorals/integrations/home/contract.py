"""The shared contract every native home backend answers.

One shape for Home Assistant's new native siblings (WeMo, Z-Wave) and
for any backend that joins later — the conformance discipline of the
earlier waves applied to hardware:

* ``capabilities()`` — what the backend can do, as comparable data
  (transport, discovery method, readable/controllable verbs), never
  prose.
* ``discover()`` — devices as :class:`HomeDevice` data.  Discovery is
  read-only: it must never actuate anything.  Degradation ends in a
  named error (``network.error`` / ``dependency.failed``), never an
  empty list that reads as "no devices exist".
* ``read_state(device_id)`` — current device state as data.  Reads are
  ungated: knowing is always allowed.
* ``control(device_id, action, **params)`` — the *only* public control
  entry.  It is final: it resolves the device, asks the
  :class:`~nomorals.integrations.home.gate.ControlGate` for the owner's
  per-device enable, and only then calls the backend's ``_control``.
  A backend cannot reach its own transport for control without the
  gate, because ``_control`` is the sole caller of the control
  transport path and ``control`` is the sole caller of ``_control``.

Device values are handled with total, saturating arithmetic
(``point.rs``'s embedded never-crash idiom): out-of-range requests are
clamped to the device's range, malformed device payloads become named
errors — nothing raises ``ValueError``/``KeyError`` at the boundary.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ...core.errors import ValidationError
from .errors import HomeDeviceNotFoundError
from .gate import ControlGate

__all__ = [
    "BackendCapabilities",
    "HomeBackend",
    "HomeDevice",
    "clamp",
    "clamp_int",
]


def clamp(value: float, low: float, high: float) -> float:
    """Saturating clamp (``point.rs`` total arithmetic): never raises,
    ``nan`` collapses to the low bound."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return low
    if v != v:  # nan
        return low
    return max(low, min(high, v))


def clamp_int(value: Any, low: int, high: int) -> int:
    """Saturating integer clamp for device value ranges (0–99 Z-Wave
    levels, 0–255 WeMo brightness…)."""
    return int(clamp(value, float(low), float(high)))


@dataclass(frozen=True)
class HomeDevice:
    """One discovered device, backend-neutral."""

    backend: str
    device_id: str
    name: str
    device_type: str  # switch, light, sensor, controller, …
    room: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    #: Backend-private addressing (URLs, node ids, service maps).  Data,
    #: not behaviour — the backend that produced it interprets it.
    address: dict[str, Any] = field(default_factory=dict)
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device_id": self.device_id,
            "name": self.name,
            "type": self.device_type,
            "room": self.room,
            "state": dict(self.state),
            "capabilities": list(self.capabilities),
            "available": self.available,
        }


@dataclass(frozen=True)
class BackendCapabilities:
    """Comparable capability data for one backend."""

    backend: str
    transport: str
    discovery: str
    readable: tuple[str, ...] = ()
    controllable: tuple[str, ...] = ()
    #: Control on this backend is always behind the owner's per-device
    #: enable flag.  Recorded as data so a surface can show the posture
    #: without knowing the backend.
    control_gated: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "transport": self.transport,
            "discovery": self.discovery,
            "readable": list(self.readable),
            "controllable": list(self.controllable),
            "control_gated": self.control_gated,
            "notes": self.notes,
        }


class HomeBackend(abc.ABC):
    """Base class for native home backends.  See module docstring for
    the contract; subclasses implement ``discover``, ``read_state`` and
    ``_control`` and never expose a second control path."""

    name: str = "home"

    def __init__(self, gate: ControlGate | None = None) -> None:
        self._gate = gate if gate is not None else ControlGate()
        self._devices: dict[str, HomeDevice] = {}
        #: Human-readable notes from the last discovery ladder run
        #: (which rungs answered, which degraded) — evidence, as data.
        self.last_discovery_notes: list[str] = []

    # ── capability data ──────────────────────────────────────────────
    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """What this backend can do, as data."""

    # ── discovery & reads (ungated, read-only) ───────────────────────
    @abc.abstractmethod
    def discover(self) -> list[HomeDevice]:
        """Discover devices.  Read-only; refreshes the registry.
        Degradation ends in a named error, never ``[]`` silence."""

    @abc.abstractmethod
    def read_state(self, device_id: str) -> dict[str, Any]:
        """Current state of one device, as data."""

    def known_devices(self) -> list[HomeDevice]:
        """Devices in the registry (from the last discovery run)."""
        return list(self._devices.values())

    def _register(self, devices: list[HomeDevice]) -> list[HomeDevice]:
        self._devices = {d.device_id: d for d in devices}
        return devices

    def _resolve(self, device_id: str) -> HomeDevice:
        """Find a device in the registry, discovering once if empty."""
        device = self._devices.get(device_id)
        if device is not None:
            return device
        if not self._devices:
            self.discover()
            device = self._devices.get(device_id)
            if device is not None:
                return device
        raise HomeDeviceNotFoundError(backend=self.name,
                                      device_id=device_id)

    # ── control (the gated seam — final) ─────────────────────────────
    def control(self, device_id: str, action: str, **params: Any
                ) -> dict[str, Any]:
        """Run one control action on one device.

        THE seam: the owner's per-device enable is checked here, before
        ``_control`` — and ``_control`` is the only path to the control
        transport.  With the flag off (the default) this raises
        :class:`ControlNotEnabledError` and the transport sees nothing.
        """
        device = self._resolve(device_id)
        caps = self.capabilities()
        if action not in caps.controllable:
            raise ValidationError(
                f"backend {self.name!r} cannot perform action {action!r}; "
                f"controllable actions: {sorted(caps.controllable)}",
                details={"backend": self.name, "action": action,
                         "device_id": device_id})
        self._gate.require_enabled(self.name, device.device_id,
                                   action=action)
        state = self._control(device, action, params)
        return {
            "backend": self.name,
            "device_id": device.device_id,
            "action": action,
            "state": state,
        }

    @abc.abstractmethod
    def _control(self, device: HomeDevice, action: str,
                 params: dict[str, Any]) -> dict[str, Any]:
        """Backend control implementation.  Only ever called by
        :meth:`control`, after the gate.  Returns the device's state
        read back after the action — the device, not the request, is
        the truth."""

    # ── status as data ────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """Backend + registry + gate posture, as data."""
        return {
            "backend": self.name,
            "capabilities": self.capabilities().to_dict(),
            "devices_known": len(self._devices),
            "devices": [
                {
                    **d.to_dict(),
                    "control_enabled": self._gate.is_enabled(
                        self.name, d.device_id),
                }
                for d in self._devices.values()
            ],
            "last_discovery_notes": list(self.last_discovery_notes),
        }
