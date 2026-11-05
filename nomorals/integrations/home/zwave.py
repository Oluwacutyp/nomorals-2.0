"""Native Z-Wave backend via a RaZberry / Z-Way controller — a
pattern-port of ``razberry.rs``.

The controller owns the Z-Wave network; Devon talks to its HTTP JSON
API (``/ZWaveAPI/Data/<ts>`` for the device tree,
``/ZWaveAPI/Run/devices[n].instances[i].commandClasses[cc].Set(v)``
for commands), exactly the boundary ``razberry.rs`` was built against
(and heavily fixture-tested, because the boundary is JSON).

* **Discovery** reads the controller's full data tree and models each
  node from its command classes (37 SwitchBinary, 38 SwitchMultilevel,
  48 SensorBinary, 49 SensorMultilevel).  An unreachable controller is
  ``network.error``; an unconfigured one is ``dependency.failed``;
  a tree that doesn't parse is ``dependency.failed`` — never ``[]``.
* **Reads** re-read the tree (the controller is the truth) and extract
  the node's values.  **Control** issues ``Set`` commands through the
  contract's gated seam only, then reads the state back.
* Values use saturating arithmetic: multilevel levels clamp to the
  Z-Wave range 0–99 (255 would mean "restore previous" — never sent
  implicitly).

The transport seam (:class:`ZwaveTransport`) returns raw response
text, so JSON parsing happens in the backend and is genuinely
exercised when tests wire in the emulator double.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from ...core.errors import AuthError, DependencyError, NetworkError
from .contract import BackendCapabilities, HomeBackend, HomeDevice, clamp_int
from .gate import ControlGate

__all__ = ["RazberryBackend", "UrllibZwaveTransport", "ZwaveTransport"]

CC_SWITCH_BINARY = "37"
CC_SWITCH_MULTILEVEL = "38"
CC_SENSOR_BINARY = "48"
CC_SENSOR_MULTILEVEL = "49"
CONTROLLER_NODE = "1"


class ZwaveTransport(Protocol):
    """The wire seam: one GET, raw text back.

    Raises :class:`HttpStatusError` for non-2xx, ``OSError`` for
    transport failure."""

    def get(self, url: str, headers: dict[str, str], timeout_s: float
            ) -> str:
        ...


class HttpStatusError(OSError):
    """Non-2xx HTTP response (an OSError so transports share one
    failure channel; ``status`` carries the code)."""

    def __init__(self, status: int, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.url = url


class UrllibZwaveTransport:
    """The real transport: urllib GET."""

    def get(self, url: str, headers: dict[str, str], timeout_s: float
            ) -> str:
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise HttpStatusError(exc.code, url) from exc


def _value(node: Any) -> Any:
    """Z-Way wraps every datum as ``{"value": …}`` — unwrap totally."""
    if isinstance(node, dict):
        return node.get("value")
    return node


def _device_type(ccs: dict[str, Any]) -> str:
    if CC_SWITCH_MULTILEVEL in ccs:
        return "light"
    if CC_SWITCH_BINARY in ccs:
        return "switch"
    return "sensor"


def _extract_state(ccs: dict[str, Any]) -> dict[str, Any]:
    """Pull readable values out of one node's command classes.
    Total: unrecognized shapes are ignored, never raised on."""
    state: dict[str, Any] = {}
    if CC_SWITCH_BINARY in ccs:
        level = _value(ccs[CC_SWITCH_BINARY].get("data", {}).get("level"))
        if level is not None:
            state["on"] = bool(level)
            state["switch_binary"] = bool(level)
    if CC_SWITCH_MULTILEVEL in ccs:
        level = _value(
            ccs[CC_SWITCH_MULTILEVEL].get("data", {}).get("level"))
        if level is not None:
            state["level"] = clamp_int(level, 0, 99)
            state["on"] = state.get("on", clamp_int(level, 0, 99) > 0)
    sensors: list[dict[str, Any]] = []
    if CC_SENSOR_BINARY in ccs:
        level = _value(ccs[CC_SENSOR_BINARY].get("data", {}).get("level"))
        if level is not None:
            sensors.append({"kind": "binary", "value": bool(level)})
    if CC_SENSOR_MULTILEVEL in ccs:
        data = ccs[CC_SENSOR_MULTILEVEL].get("data", {})
        for key, entry in data.items():
            if not isinstance(entry, dict) or "val" not in entry:
                continue
            reading: dict[str, Any] = {
                "kind": "multilevel",
                "sensor_type": key,
                "value": _value(entry.get("val")),
            }
            scale = _value(entry.get("scaleString"))
            if scale:
                reading["scale"] = scale
            sensors.append(reading)
    if sensors:
        state["sensors"] = sensors
    return state


def _capabilities_for(ccs: dict[str, Any]) -> tuple[str, ...]:
    caps: list[str] = []
    if CC_SWITCH_BINARY in ccs or CC_SWITCH_MULTILEVEL in ccs:
        caps += ["turn_on", "turn_off", "toggle"]
    if CC_SWITCH_MULTILEVEL in ccs:
        caps.append("set_level")
    return tuple(caps)


class RazberryBackend(HomeBackend):
    """Z-Wave nodes behind one RaZberry / Z-Way controller."""

    name = "zwave"

    def __init__(
        self,
        *,
        gate: ControlGate | None = None,
        transport: ZwaveTransport | None = None,
        base_url: str = "",
        username: str = "",
        password: str = "",
        http_timeout_s: float = 5.0,
    ) -> None:
        super().__init__(gate)
        self._transport = transport or UrllibZwaveTransport()
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._http_timeout_s = http_timeout_s

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend="zwave",
            transport="RaZberry/Z-Way controller HTTP JSON API (sync)",
            discovery="GET /ZWaveAPI/Data/0 — full node tree",
            readable=("switch_binary", "switch_multilevel",
                      "sensor_binary", "sensor_multilevel"),
            controllable=("turn_on", "turn_off", "toggle", "set_level"),
            notes="Node 1 is the controller itself and is never "
                  "registered as a controllable device.",
        )

    # ── wire helpers ──────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._username:
            raw = f"{self._username}:{self._password}".encode("utf-8")
            headers["Authorization"] = (
                "Basic " + base64.b64encode(raw).decode("ascii"))
        return headers

    def _get_json(self, path: str) -> Any:
        if not self._base_url:
            raise DependencyError(
                "Z-Wave controller URL is not configured "
                "(set the zwave credential or NM_ZWAVE_URL)",
                dependency="zwave",
                details={"reason": "not_configured"})
        url = f"{self._base_url}{path}"
        try:
            text = self._transport.get(url, self._headers(),
                                       self._http_timeout_s)
        except HttpStatusError as exc:
            if exc.status in (401, 403):
                raise AuthError(
                    f"Z-Wave controller rejected the credentials "
                    f"(HTTP {exc.status})",
                    details={"backend": "zwave",
                             "http_status": exc.status}) from exc
            raise NetworkError(
                f"Z-Wave controller answered HTTP {exc.status} for "
                f"{path}",
                details={"backend": "zwave",
                         "http_status": exc.status}) from exc
        except OSError as exc:
            raise NetworkError(
                f"Z-Wave controller unreachable at {self._base_url}: "
                f"{exc}",
                details={"backend": "zwave"}) from exc
        try:
            return json.loads(text)
        except ValueError as exc:
            raise DependencyError(
                f"Z-Wave controller returned unparseable JSON for "
                f"{path}: {exc}",
                dependency="zwave",
                details={"reason": "protocol_parse", "path": path},
            ) from exc

    def _data_tree(self) -> dict[str, Any]:
        tree = self._get_json("/ZWaveAPI/Data/0")
        if not isinstance(tree, dict) or not isinstance(
                tree.get("devices"), dict):
            raise DependencyError(
                "Z-Wave controller data tree has no devices map",
                dependency="zwave",
                details={"reason": "protocol_parse", "path": "/ZWaveAPI/Data/0"})
        return tree

    # ── discovery & reads ─────────────────────────────────────────────
    def discover(self) -> list[HomeDevice]:
        tree = self._data_tree()
        devices: list[HomeDevice] = []
        for node_id, node in sorted(tree["devices"].items()):
            if node_id == CONTROLLER_NODE or not isinstance(node, dict):
                continue
            ccs = self._command_classes(node)
            data = node.get("data", {}) if isinstance(
                node.get("data"), dict) else {}
            name = _value(data.get("givenName")) or (
                f"Z-Wave device {node_id}")
            devices.append(HomeDevice(
                backend="zwave",
                device_id=f"zwave-{node_id}",
                name=str(name),
                device_type=_device_type(ccs),
                state=_extract_state(ccs),
                capabilities=_capabilities_for(ccs),
                address={"node_id": node_id,
                         "command_classes": sorted(ccs.keys())},
            ))
        self.last_discovery_notes = [
            f"zwave: data tree held {len(tree['devices'])} node(s), "
            f"{len(devices)} device(s) registered "
            f"(controller node excluded)"]
        if not devices:
            raise NetworkError(
                "Z-Wave controller answered but its tree holds no "
                "devices besides the controller",
                details={"ladder": self.last_discovery_notes})
        return self._register(devices)

    @staticmethod
    def _command_classes(node: dict[str, Any]) -> dict[str, Any]:
        instances = node.get("instances", {})
        if not isinstance(instances, dict):
            return {}
        first = instances.get("0", {})
        if not isinstance(first, dict):
            return {}
        ccs = first.get("commandClasses", {})
        return ccs if isinstance(ccs, dict) else {}

    def read_state(self, device_id: str) -> dict[str, Any]:
        device = self._resolve(device_id)
        tree = self._data_tree()
        node = tree["devices"].get(device.address["node_id"])
        if not isinstance(node, dict):
            raise DependencyError(
                f"Z-Wave node {device.address['node_id']} vanished "
                f"from the controller tree",
                dependency="zwave",
                details={"reason": "node_vanished",
                         "device_id": device_id})
        return _extract_state(self._command_classes(node))

    # ── control (only via HomeBackend.control's gate) ────────────────
    def _run(self, device: HomeDevice, command_class: str,
             value: int) -> None:
        node_id = device.address["node_id"]
        path = (f"/ZWaveAPI/Run/devices[{node_id}].instances[0]"
                f".commandClasses[{command_class}].Set({value})")
        result = self._get_json(path)
        if isinstance(result, dict) and result.get("error"):
            raise DependencyError(
                f"Z-Wave controller refused Set on node {node_id}: "
                f"{result['error']}",
                dependency="zwave",
                details={"reason": "command_refused", "path": path})

    def _control(self, device: HomeDevice, action: str,
                 params: dict[str, Any]) -> dict[str, Any]:
        ccs = set(device.address.get("command_classes", []))
        if action in ("turn_on", "turn_off"):
            on = action == "turn_on"
            if CC_SWITCH_BINARY in ccs:
                self._run(device, CC_SWITCH_BINARY, 255 if on else 0)
            elif CC_SWITCH_MULTILEVEL in ccs:
                self._run(device, CC_SWITCH_MULTILEVEL, 99 if on else 0)
            else:
                raise DependencyError(
                    f"Z-Wave device {device.name!r} has no switch "
                    f"command class",
                    dependency="zwave",
                    details={"reason": "no_switch_class",
                             "device_id": device.device_id})
        elif action == "toggle":
            current = self.read_state(device.device_id)
            return self._control(
                device, "turn_off" if current.get("on") else "turn_on",
                params)
        elif action == "set_level":
            if CC_SWITCH_MULTILEVEL not in ccs:
                raise DependencyError(
                    f"Z-Wave device {device.name!r} has no multilevel "
                    f"switch (not a dimmer)",
                    dependency="zwave",
                    details={"reason": "no_multilevel_class",
                             "device_id": device.device_id})
            # 0–99 saturating: 255 means "restore previous level" in
            # Z-Wave and is never sent implicitly.
            self._run(device, CC_SWITCH_MULTILEVEL,
                      clamp_int(params.get("value", 0), 0, 99))
        else:  # pragma: no cover — contract validates actions first
            raise DependencyError(
                f"unsupported Z-Wave action {action!r}",
                dependency="zwave")
        return self.read_state(device.device_id)
