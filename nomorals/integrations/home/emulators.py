"""Emulator-as-oracle test doubles for the native home backends.

The laser stack's testing answer (``etherdream-emulator``): a *strict
protocol server* with a queryable log — a fake that verifies, not a
mock that agrees.  Each emulator here re-implements its protocol from
the wire shape (its own XML/JSON builders and parsers; **no imports
from the backend modules**, so a bug shared by construction is
impossible — the same intentional duplication as the emulator's
second copy of the EtherDream protocol).

* :class:`WemoEmulator` speaks SSDP + UPnP/SOAP: it answers M-SEARCH,
  serves ``setup.xml``, validates SOAP envelopes (action must match
  the ``SOAPACTION`` header and the service's real action set, values
  must be in range) and faults exactly where a real WeMo faults.
* :class:`ZwaveEmulator` serves the Z-Way data tree and ``Run``
  command paths, mutating node state on ``Set`` like the controller.

Both keep an assertion log: every request is recorded, and
``control_calls()`` filters to the requests that *actuate* — the tests
for the owner's control gate assert that log is empty while the gate
is off.  Transports adapters (:class:`WemoEmulatorTransport`,
:class:`ZwaveEmulatorTransport`) route a backend's wire calls into an
emulator in-process; nothing here opens a real socket.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any

__all__ = [
    "WemoEmulator",
    "WemoEmulatorTransport",
    "ZwaveEmulator",
    "ZwaveEmulatorTransport",
]

# ── WeMo ──────────────────────────────────────────────────────────────

_BASIC = "urn:Belkin:service:basicevent:1"
_BRIGHT = "urn:Belkin:service:brightness:1"
_ACTIONS = {
    _BASIC: {"GetBinaryState", "SetBinaryState", "GetInsightParams"},
    _BRIGHT: {"GetBrightness", "SetBrightness"},
}


def _lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class WemoEmulator:
    """A strict fake WeMo network: SSDP + description + SOAP."""

    def __init__(self, devices: list[dict[str, Any]]) -> None:
        """``devices``: dicts with keys ``host`` (ip:port authority),
        ``name``, ``model`` ("Socket" | "Insight" | "Dimmer" |
        "Lightswitch"), ``serial``, optional ``binary_state`` (0/1),
        optional ``brightness`` (0–255, dimmers)."""
        self._devices: dict[str, dict[str, Any]] = {}
        for spec in devices:
            udn = f"uuid:{spec.get('model', 'Socket')}-1_0-{spec['serial']}"
            self._devices[spec["host"]] = {
                "host": spec["host"],
                "name": spec["name"],
                "model": spec.get("model", "Socket"),
                "serial": spec["serial"],
                "udn": udn,
                "binary_state": int(spec.get("binary_state", 0)),
                "brightness": int(spec.get("brightness", 0)),
                "power_mw": int(spec.get("power_mw", 0)),
            }
        self.log: list[dict[str, Any]] = []
        #: Failure injection: pretend multicast is a black hole.
        self.ssdp_blackhole = False
        #: Failure injection: serve this raw body as setup.xml.
        self.description_override: bytes | None = None

    # ── assertion surface ─────────────────────────────────────────────
    def control_calls(self) -> list[dict[str, Any]]:
        """Log entries that actuate a device (Set* SOAP actions)."""
        return [e for e in self.log
                if e["kind"] == "soap" and str(e.get("action", ""))
                .startswith("Set")]

    def device_state(self, host: str) -> dict[str, Any]:
        dev = self._devices[host]
        return {"binary_state": dev["binary_state"],
                "brightness": dev["brightness"]}

    # ── SSDP ──────────────────────────────────────────────────────────
    def handle_ssdp_search(self, search_target: str
                           ) -> list[dict[str, str]]:
        self.log.append({"kind": "ssdp", "st": search_target})
        if self.ssdp_blackhole:
            return []
        if search_target not in (
                "urn:Belkin:device:**", "ssdp:all",
                "upnp:rootdevice", "urn:Belkin:device:controllee:1"):
            return []
        return [{
            "LOCATION": f"http://{host}/setup.xml",
            "ST": "urn:Belkin:device:controllee:1",
            "USN": f"{dev['udn']}::urn:Belkin:device:controllee:1",
            "SERVER": "emulator/1.0 UPnP/1.1 WeMoEmulator/1.0",
        } for host, dev in self._devices.items()]

    # ── HTTP ──────────────────────────────────────────────────────────
    def handle_http(self, method: str, url: str,
                    headers: dict[str, str] | None = None,
                    body: bytes = b"") -> tuple[int, bytes]:
        headers = headers or {}
        authority = url.split("://", 1)[-1].split("/", 1)[0]
        path = "/" + url.split("://", 1)[-1].split("/", 1)[1] \
            if "/" in url.split("://", 1)[-1] else "/"
        dev = self._devices.get(authority)
        if dev is None:
            self.log.append({"kind": "http", "method": method,
                             "url": url, "status": 404})
            return 404, b"not found"
        if method == "GET" and path == "/setup.xml":
            self.log.append({"kind": "http", "method": "GET",
                             "url": url, "status": 200})
            if self.description_override is not None:
                return 200, self.description_override
            return 200, self._description(dev).encode("utf-8")
        if method == "POST" and path.startswith("/upnp/control/"):
            return self._soap(dev, path, headers, body)
        self.log.append({"kind": "http", "method": method, "url": url,
                         "status": 404})
        return 404, b"not found"

    def _description(self, dev: dict[str, Any]) -> str:
        services = [(_BASIC, "basicevent1")]
        if dev["model"] == "Dimmer":
            services.append((_BRIGHT, "brightness1"))
        service_xml = "".join(
            "<service>"
            f"<serviceType>{stype}</serviceType>"
            f"<serviceId>urn:upnp-org:serviceId:{sid}</serviceId>"
            f"<controlURL>/upnp/control/{sid}</controlURL>"
            f"<eventSubURL>/upnp/event/{sid}</eventSubURL>"
            f"<SCPDURL>/eventservice.xml</SCPDURL>"
            "</service>"
            for stype, sid in services)
        return (
            '<?xml version="1.0"?>'
            '<root xmlns="urn:schemas-upnp-org:device-1-0">'
            "<device>"
            "<deviceType>urn:Belkin:device:controllee:1</deviceType>"
            f"<friendlyName>{dev['name']}</friendlyName>"
            "<manufacturer>Belkin International Inc.</manufacturer>"
            f"<modelName>{dev['model']}</modelName>"
            "<modelNumber>1.0</modelNumber>"
            f"<serialNumber>{dev['serial']}</serialNumber>"
            f"<UDN>{dev['udn']}</UDN>"
            f"<serviceList>{service_xml}</serviceList>"
            "</device></root>")

    # ── SOAP: strict on purpose ───────────────────────────────────────
    def _fault(self, dev: dict[str, Any], action: str, why: str
               ) -> tuple[int, bytes]:
        self.log.append({"kind": "soap", "action": action,
                         "device": dev["name"], "fault": why,
                         "status": 500})
        return 500, (
            '<?xml version="1.0"?>'
            '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
            "<s:Body><s:Fault>"
            "<faultcode>s:Client</faultcode>"
            f"<faultstring>{why}</faultstring>"
            "</s:Fault></s:Body></s:Envelope>").encode("utf-8")

    def _soap(self, dev: dict[str, Any], path: str,
              headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        service = _BASIC if path.endswith("basicevent1") else _BRIGHT
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return self._fault(dev, "?", "malformed SOAP envelope")
        action_el = None
        for el in root.iter():
            if _lname(el.tag) in _ACTIONS[service]:
                action_el = el
                break
        if action_el is None:
            return self._fault(dev, "?", "no known action in body")
        action = _lname(action_el.tag)
        declared = headers.get("SOAPACTION", "").strip('"')
        if declared != f"{service}#{action}":
            return self._fault(
                dev, action,
                f"SOAPACTION {declared!r} does not match body action")
        args = {_lname(child.tag): (child.text or "").strip()
                for child in action_el}
        entry = {"kind": "soap", "action": action,
                 "device": dev["name"], "args": args, "status": 200}

        if action == "GetBinaryState":
            payload = f"<BinaryState>{dev['binary_state']}</BinaryState>"
        elif action == "SetBinaryState":
            value = args.get("BinaryState", "")
            if value not in ("0", "1"):
                return self._fault(dev, action,
                                   f"bad BinaryState {value!r}")
            dev["binary_state"] = int(value)
            payload = f"<BinaryState>{value}</BinaryState>"
        elif action == "GetInsightParams":
            if dev["model"] != "Insight":
                return self._fault(dev, action,
                                   "not an Insight device")
            payload = ("<InsightParams>"
                       f"{dev['binary_state']}|0|0|0|0|0|0|"
                       f"{dev['power_mw']}|0|0"
                       "</InsightParams>")
        elif action == "GetBrightness":
            if dev["model"] != "Dimmer":
                return self._fault(dev, action, "not a dimmer")
            payload = f"<Brightness>{dev['brightness']}</Brightness>"
        elif action == "SetBrightness":
            if dev["model"] != "Dimmer":
                return self._fault(dev, action, "not a dimmer")
            try:
                value = int(args.get("Brightness", ""))
            except ValueError:
                return self._fault(dev, action, "bad Brightness value")
            if not 0 <= value <= 255:
                return self._fault(dev, action,
                                   "Brightness out of range")
            dev["brightness"] = value
            if value > 0:
                dev["binary_state"] = 1
            payload = f"<Brightness>{value}</Brightness>"
        else:  # pragma: no cover — action set is closed above
            return self._fault(dev, action, "unknown action")
        self.log.append(entry)
        return 200, (
            '<?xml version="1.0"?>'
            '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
            f'<s:Body><u:{action}Response xmlns:u="{service}">'
            f"{payload}</u:{action}Response>"
            "</s:Body></s:Envelope>").encode("utf-8")


class WemoEmulatorTransport:
    """Routes a :class:`WemoBackend`'s wire calls into an emulator."""

    def __init__(self, emulator: WemoEmulator) -> None:
        self._emulator = emulator

    def ssdp_search(self, search_target: str, timeout_s: float
                    ) -> list[dict[str, str]]:
        return self._emulator.handle_ssdp_search(search_target)

    def http_get(self, url: str, timeout_s: float) -> tuple[int, bytes]:
        return self._emulator.handle_http("GET", url)

    def http_post(self, url: str, body: bytes, headers: dict[str, str],
                  timeout_s: float) -> tuple[int, bytes]:
        return self._emulator.handle_http("POST", url, headers, body)


# ── Z-Wave (RaZberry / Z-Way) ─────────────────────────────────────────

_SET_RE = re.compile(
    r"^/ZWaveAPI/Run/devices\[(\d+)\]\.instances\[(\d+)\]"
    r"\.commandClasses\[(\d+)\]\.Set\((-?\d+)\)$")


def _wrap(value: Any, ts: int) -> dict[str, Any]:
    return {"value": value, "updateTime": ts, "invalidateTime": 0}


class ZwaveEmulator:
    """A strict fake Z-Way controller: data tree + Run commands."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        """``nodes``: dicts with keys ``id`` (int), ``name``,
        ``switch_binary`` (bool, optional), ``level`` (int 0–99,
        optional — presence means a multilevel switch), ``sensors``
        (list of ``{"type": "binary"|"temperature"|…, "value": …}``,
        optional)."""
        self._ts = 1_700_000_000
        devices: dict[str, Any] = {
            "1": {"id": 1, "data": {"givenName": _wrap("Z-Way", self._ts)},
                  "instances": {}},
        }
        for spec in nodes:
            ccs: dict[str, Any] = {}
            if "switch_binary" in spec:
                ccs["37"] = {"name": "SwitchBinary", "data": {
                    "level": _wrap(bool(spec["switch_binary"]),
                                   self._ts)}}
            if "level" in spec:
                ccs["38"] = {"name": "SwitchMultilevel", "data": {
                    "level": _wrap(int(spec["level"]), self._ts)}}
            sensors = spec.get("sensors", [])
            binary = [s for s in sensors if s["type"] == "binary"]
            multi = [s for s in sensors if s["type"] != "binary"]
            if binary:
                ccs["48"] = {"name": "SensorBinary", "data": {
                    "level": _wrap(bool(binary[0]["value"]), self._ts)}}
            if multi:
                data: dict[str, Any] = {}
                for idx, reading in enumerate(multi):
                    data[str(idx + 1)] = {
                        "val": _wrap(reading["value"], self._ts),
                        "scale": _wrap(0, self._ts),
                        "scaleString": _wrap(
                            reading.get("scale", ""), self._ts),
                    }
                ccs["49"] = {"name": "SensorMultilevel", "data": data}
            devices[str(spec["id"])] = {
                "id": spec["id"],
                "data": {"givenName": _wrap(spec["name"], self._ts)},
                "instances": {"0": {"commandClasses": ccs}},
            }
        self._tree: dict[str, Any] = {
            "updateTime": self._ts,
            "invalidateTime": self._ts - 10,
            "devices": devices,
        }
        self.log: list[dict[str, Any]] = []

    # ── assertion surface ─────────────────────────────────────────────
    def control_calls(self) -> list[dict[str, Any]]:
        """Log entries that actuate a node (Run …Set(…) requests)."""
        return [e for e in self.log if e["kind"] == "run"]

    def node_level(self, node_id: int) -> dict[str, Any]:
        ccs = self._tree["devices"][str(node_id)]["instances"]["0"][
            "commandClasses"]
        out: dict[str, Any] = {}
        if "37" in ccs:
            out["switch_binary"] = ccs["37"]["data"]["level"]["value"]
        if "38" in ccs:
            out["level"] = ccs["38"]["data"]["level"]["value"]
        return out

    # ── HTTP ──────────────────────────────────────────────────────────
    def handle_get(self, url: str) -> tuple[int, str]:
        path = "/" + url.split("://", 1)[-1].split("/", 1)[1] \
            if "://" in url and "/" in url.split("://", 1)[-1] else url
        if path.startswith("/ZWaveAPI/Data/"):
            self.log.append({"kind": "data", "path": path,
                             "status": 200})
            return 200, json.dumps(self._tree)
        match = _SET_RE.match(path)
        if match:
            node_id, _instance, cc, raw = match.groups()
            return self._run(node_id, cc, int(raw), path)
        self.log.append({"kind": "unknown", "path": path,
                         "status": 404})
        return 404, json.dumps({"error": f"no such path: {path}"})

    def _run(self, node_id: str, cc: str, value: int,
             path: str) -> tuple[int, str]:
        entry = {"kind": "run", "path": path, "node": node_id,
                 "command_class": cc, "value": value, "status": 200}
        self.log.append(entry)
        node = self._tree["devices"].get(node_id)
        if node is None:
            entry["status"] = 404
            return 404, json.dumps(
                {"error": f"device {node_id} not found"})
        instances = node.get("instances", {})
        ccs = instances.get("0", {}).get("commandClasses", {})
        if cc not in ccs:
            entry["status"] = 404
            return 404, json.dumps(
                {"error": f"command class {cc} not present"})
        if cc == "38" and not 0 <= value <= 99 and value != 255:
            entry["status"] = 400
            return 400, json.dumps(
                {"error": f"level {value} out of range"})
        if cc not in ("37", "38"):
            entry["status"] = 400
            return 400, json.dumps(
                {"error": f"command class {cc} is not settable"})
        self._ts += 1
        if cc == "37":
            ccs["37"]["data"]["level"] = _wrap(value != 0, self._ts)
        elif cc == "38":
            ccs["38"]["data"]["level"] = _wrap(
                99 if value == 255 else value, self._ts)
        self._tree["updateTime"] = self._ts
        return 200, json.dumps({"data": None, "code": 200,
                                "message": "OK"})


class ZwaveEmulatorTransport:
    """Routes a :class:`RazberryBackend`'s GETs into an emulator.

    Raises the backend module's ``HttpStatusError`` on non-200 — the
    same shape the real transport raises — importing it lazily so this
    module stays importable on its own (the independence is about
    protocol logic, not module loading).
    """

    def __init__(self, emulator: ZwaveEmulator) -> None:
        self._emulator = emulator

    def get(self, url: str, headers: dict[str, str], timeout_s: float
            ) -> str:
        status, text = self._emulator.handle_get(url)
        if status != 200:
            from .zwave import HttpStatusError
            raise HttpStatusError(status, url)
        return text
