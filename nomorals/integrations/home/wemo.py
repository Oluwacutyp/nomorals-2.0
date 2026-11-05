"""Native Belkin WeMo backend — a pattern-port of ``wemo.rs``.

Zero-async synchronous UPnP/SOAP, in Devon's sync style:

* **Discovery** is SSDP (M-SEARCH multicast) with a *degradation
  ladder*: SSDP answers → per-device ``setup.xml`` descriptions →
  configured known-host probing when multicast yields nothing → a
  named error.  Degradation never ends in an empty list: a failed
  multicast stack is ``dependency.failed``, silence after the whole
  ladder is ``network.error``, each with the ladder's steps recorded.
* **State** is SOAP ``GetBinaryState`` (plus ``GetInsightParams`` for
  Insight plugs, ``GetBrightness`` for dimmers).
* **Control** is SOAP ``SetBinaryState`` / ``SetBrightness``, reached
  only through the contract's gated seam — this module's transport is
  never touched for control before the owner's per-device enable.

The transport is an injectable seam (:class:`WemoTransport`); the real
implementation uses only the standard library (``socket`` +
``urllib``).  Tests wire the backend to the emulator double in
:mod:`nomorals.integrations.home.emulators`, which speaks the same
protocol from an independent implementation.
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Protocol
from xml.sax.saxutils import escape

from ...core.errors import DependencyError, NetworkError
from ...core.logging_setup import get_logger
from .contract import BackendCapabilities, HomeBackend, HomeDevice, clamp_int
from .gate import ControlGate

__all__ = ["UrllibWemoTransport", "WemoBackend", "WemoTransport"]

_log = get_logger(__name__)

SSDP_ADDR = ("239.255.255.250", 1900)
BELKIN_ST = "urn:Belkin:device:**"
BASIC_EVENT = "urn:Belkin:service:basicevent:1"
BRIGHTNESS = "urn:Belkin:service:brightness:1"
WEMO_PORTS = (49152, 49153, 49154, 49155)


class WemoTransport(Protocol):
    """The wire seam.  Implementations move bytes; they hold no
    device logic, so the emulator can stand in for the network."""

    def ssdp_search(self, search_target: str, timeout_s: float
                    ) -> list[dict[str, str]]:
        """One M-SEARCH round; returns response header dicts
        (keys upper-cased).  Raises ``OSError`` when the multicast
        stack itself is unavailable."""
        ...

    def http_get(self, url: str, timeout_s: float) -> tuple[int, bytes]:
        ...

    def http_post(self, url: str, body: bytes, headers: dict[str, str],
                  timeout_s: float) -> tuple[int, bytes]:
        ...


class UrllibWemoTransport:
    """The real transport: raw UDP multicast + urllib HTTP."""

    def ssdp_search(self, search_target: str, timeout_s: float
                    ) -> list[dict[str, str]]:
        request = (
            "M-SEARCH * HTTP/1.1\r\n"
            f"HOST: {SSDP_ADDR[0]}:{SSDP_ADDR[1]}\r\n"
            'MAN: "ssdp:discover"\r\n'
            "MX: 2\r\n"
            f"ST: {search_target}\r\n"
            "\r\n"
        ).encode("ascii")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                             socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
            sock.settimeout(timeout_s)
            sock.sendto(request, SSDP_ADDR)
            responses: list[dict[str, str]] = []
            seen: set[bytes] = set()
            while True:
                try:
                    data, _addr = sock.recvfrom(65535)
                except socket.timeout:
                    break
                if data in seen:
                    continue
                seen.add(data)
                headers: dict[str, str] = {}
                for line in data.decode("utf-8", "replace").split("\r\n"):
                    if ":" in line:
                        key, _, value = line.partition(":")
                        headers[key.strip().upper()] = value.strip()
                if headers:
                    responses.append(headers)
            return responses
        finally:
            sock.close()

    def http_get(self, url: str, timeout_s: float) -> tuple[int, bytes]:
        req = urllib.request.Request(url, method="GET")
        return self._open(req, timeout_s)

    def http_post(self, url: str, body: bytes, headers: dict[str, str],
                  timeout_s: float) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST")
        return self._open(req, timeout_s)

    @staticmethod
    def _open(req: urllib.request.Request, timeout_s: float
              ) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            # A SOAP fault arrives as HTTP 500 with a body the caller
            # must see; surface status+body, don't raise it away.
            return exc.code, exc.read()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_text(root: ET.Element, name: str) -> str:
    for el in root.iter():
        if _local(el.tag) == name:
            return (el.text or "").strip()
    return ""


def _parse_description(payload: bytes, location: str) -> HomeDevice:
    """Parse one ``setup.xml`` into a device.  Malformed XML is a
    protocol failure of that peer → ``dependency.failed``."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise DependencyError(
            f"WeMo device at {location} returned a malformed "
            f"description: {exc}",
            dependency="wemo",
            details={"reason": "description_parse", "location": location},
        ) from exc
    device_el = None
    for el in root.iter():
        if _local(el.tag) == "device":
            device_el = el
            break
    if device_el is None:
        raise DependencyError(
            f"WeMo description at {location} has no device element",
            dependency="wemo",
            details={"reason": "description_parse", "location": location})

    def child(name: str) -> str:
        for el in device_el:
            if _local(el.tag) == name:
                return (el.text or "").strip()
        return ""

    services: dict[str, str] = {}
    for el in device_el.iter():
        if _local(el.tag) != "service":
            continue
        stype, curl = "", ""
        for sub in el:
            if _local(sub.tag) == "serviceType":
                stype = (sub.text or "").strip()
            elif _local(sub.tag) == "controlURL":
                curl = (sub.text or "").strip()
        if stype and curl:
            base = location.rsplit("/", 1)[0]
            services[stype] = curl if curl.startswith("http") else base + (
                curl if curl.startswith("/") else "/" + curl)
    if BASIC_EVENT not in services:
        raise DependencyError(
            f"WeMo device at {location} offers no basicevent service",
            dependency="wemo",
            details={"reason": "no_basicevent", "location": location})

    model = child("modelName") or "WeMo"
    udn = child("UDN") or location
    caps = ["turn_on", "turn_off", "toggle"]
    if BRIGHTNESS in services:
        caps.append("set_brightness")
    return HomeDevice(
        backend="wemo",
        device_id=udn,
        name=child("friendlyName") or udn,
        device_type="light" if BRIGHTNESS in services else "switch",
        state={},
        capabilities=tuple(caps),
        address={
            "location": location,
            "services": services,
            "model": model,
            "serial": child("serialNumber"),
            "manufacturer": child("manufacturer"),
        },
    )


def _soap_envelope(service: str, action: str, args: dict[str, Any]) -> bytes:
    inner = "".join(
        f"<{k}>{escape(str(v))}</{k}>" for k, v in args.items())
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        f'<s:Body><u:{action} xmlns:u="{service}">{inner}</u:{action}>'
        "</s:Body></s:Envelope>"
    ).encode("utf-8")


class WemoBackend(HomeBackend):
    """WeMo devices over native UPnP/SOAP."""

    name = "wemo"

    def __init__(
        self,
        *,
        gate: ControlGate | None = None,
        transport: WemoTransport | None = None,
        known_hosts: tuple[str, ...] = (),
        ssdp_timeout_s: float = 3.0,
        http_timeout_s: float = 5.0,
    ) -> None:
        super().__init__(gate)
        self._transport = transport or UrllibWemoTransport()
        self._known_hosts = tuple(known_hosts)
        self._ssdp_timeout_s = ssdp_timeout_s
        self._http_timeout_s = http_timeout_s

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend="wemo",
            transport="UPnP/SOAP over HTTP + SSDP discovery (sync)",
            discovery="SSDP M-SEARCH → setup.xml; known-host probe "
                      "fallback; named error when the ladder exhausts",
            readable=("binary_state", "insight_params", "brightness"),
            controllable=("turn_on", "turn_off", "toggle",
                          "set_brightness"),
            notes="Brightness only on devices offering the Belkin "
                  "brightness service (dimmers).",
        )

    # ── discovery: the degradation ladder ────────────────────────────
    def discover(self) -> list[HomeDevice]:
        notes: list[str] = []
        locations: list[str] = []
        ssdp_failed: OSError | None = None
        try:
            responses = self._transport.ssdp_search(
                BELKIN_ST, self._ssdp_timeout_s)
            for headers in responses:
                loc = headers.get("LOCATION", "")
                if loc and loc not in locations:
                    locations.append(loc)
            notes.append(
                f"ssdp: {len(responses)} response(s), "
                f"{len(locations)} unique location(s)")
        except OSError as exc:
            ssdp_failed = exc
            notes.append(f"ssdp: multicast search failed: {exc}")

        devices: list[HomeDevice] = []
        parse_failures = 0
        for loc in locations:
            try:
                status, payload = self._transport.http_get(
                    loc, self._http_timeout_s)
            except OSError as exc:
                parse_failures += 1
                notes.append(f"description {loc}: fetch failed: {exc}")
                continue
            if status != 200:
                parse_failures += 1
                notes.append(f"description {loc}: HTTP {status}")
                continue
            try:
                devices.append(_parse_description(payload, loc))
            except DependencyError as exc:
                parse_failures += 1
                notes.append(f"description {loc}: {exc.message}")

        if not devices and self._known_hosts:
            notes.append("ladder: probing known hosts")
            for host in self._known_hosts:
                devices.extend(self._probe_host(host, notes))

        self.last_discovery_notes = notes
        if devices:
            return self._register(devices)
        if ssdp_failed is not None and not self._known_hosts:
            raise DependencyError(
                f"SSDP discovery unavailable ({ssdp_failed}) and no "
                f"known WeMo hosts are configured",
                dependency="ssdp",
                details={"ladder": notes}) from ssdp_failed
        if parse_failures:
            raise DependencyError(
                "WeMo discovery: devices answered SSDP but every "
                "description fetch/parse failed (see ladder)",
                dependency="wemo", details={"ladder": notes})
        raise NetworkError(
            "WeMo discovery found no devices (SSDP silent and the "
            "known-host ladder is exhausted)",
            details={"ladder": notes})

    def _probe_host(self, host: str, notes: list[str]) -> list[HomeDevice]:
        host = host.rstrip("/")
        urls = [host] if "://" in host and host.endswith(".xml") else None
        if urls is None:
            base = host if "://" in host else f"http://{host}"
            if ":" in base.rsplit("/", 1)[-1]:
                urls = [f"{base}/setup.xml"]
            else:
                urls = [f"{base}:{port}/setup.xml" for port in WEMO_PORTS]
        found: list[HomeDevice] = []
        for url in urls:
            try:
                status, payload = self._transport.http_get(
                    url, self._http_timeout_s)
            except OSError as exc:
                notes.append(f"probe {url}: {exc}")
                continue
            if status == 200:
                try:
                    found.append(_parse_description(payload, url))
                except DependencyError as exc:
                    notes.append(f"probe {url}: {exc.message}")
                break
        return found

    # ── SOAP machinery ────────────────────────────────────────────────
    def _soap(self, device: HomeDevice, service: str, action: str,
              args: dict[str, Any]) -> ET.Element:
        url = device.address["services"][service]
        body = _soap_envelope(service, action, args)
        headers = {
            "Content-Type": 'text/xml; charset="utf-8"',
            "SOAPACTION": f'"{service}#{action}"',
        }
        try:
            status, payload = self._transport.http_post(
                url, body, headers, self._http_timeout_s)
        except OSError as exc:
            raise NetworkError(
                f"WeMo SOAP {action} to {device.name!r} failed: {exc}",
                details={"backend": "wemo", "action": action,
                         "device_id": device.device_id}) from exc
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise DependencyError(
                f"WeMo device {device.name!r} answered {action} with "
                f"unparseable XML (HTTP {status})",
                dependency="wemo",
                details={"reason": "soap_parse", "action": action,
                         "http_status": status}) from exc
        fault = _find_text(root, "faultstring")
        if fault:
            raise DependencyError(
                f"WeMo device {device.name!r} faulted on {action}: "
                f"{fault}",
                dependency="wemo",
                details={"reason": "soap_fault", "action": action,
                         "fault": fault, "http_status": status})
        if status != 200:
            raise DependencyError(
                f"WeMo device {device.name!r} answered {action} with "
                f"HTTP {status} and no SOAP body",
                dependency="wemo",
                details={"reason": "soap_http", "action": action,
                         "http_status": status})
        return root

    # ── reads ─────────────────────────────────────────────────────────
    def read_state(self, device_id: str) -> dict[str, Any]:
        device = self._resolve(device_id)
        root = self._soap(device, BASIC_EVENT, "GetBinaryState", {})
        raw = _find_text(root, "BinaryState")
        if raw == "":
            raise DependencyError(
                f"WeMo device {device.name!r} returned no BinaryState",
                dependency="wemo",
                details={"reason": "soap_parse",
                         "action": "GetBinaryState"})
        # WeMo reports 0/1, and 8 for "on, state unknown" on some
        # firmware — anything non-zero is on.
        state: dict[str, Any] = {
            "binary_state": clamp_int(raw, 0, 8),
            "on": raw.strip() not in ("", "0"),
        }
        services = device.address["services"]
        if device.address.get("model") == "Insight":
            try:
                root = self._soap(device, BASIC_EVENT,
                                  "GetInsightParams", {})
                params = _find_text(root, "InsightParams")
                fields = params.split("|")
                if len(fields) > 7 and fields[7]:
                    state["current_power_mw"] = clamp_int(fields[7], 0,
                                                          10 ** 9)
            except (DependencyError, NetworkError) as exc:
                _log.debug("wemo insight params unavailable for %s: %s",
                           device_id, exc)
                state["insight_error"] = exc.code
        if BRIGHTNESS in services:
            root = self._soap(device, BRIGHTNESS, "GetBrightness", {})
            bright = _find_text(root, "Brightness")
            if bright:
                state["brightness"] = clamp_int(bright, 0, 255)
        return state

    # ── control (only via HomeBackend.control's gate) ────────────────
    def _control(self, device: HomeDevice, action: str,
                 params: dict[str, Any]) -> dict[str, Any]:
        if action == "turn_on":
            self._soap(device, BASIC_EVENT, "SetBinaryState",
                       {"BinaryState": 1})
        elif action == "turn_off":
            self._soap(device, BASIC_EVENT, "SetBinaryState",
                       {"BinaryState": 0})
        elif action == "toggle":
            current = self.read_state(device.device_id)
            self._soap(device, BASIC_EVENT, "SetBinaryState",
                       {"BinaryState": 0 if current["on"] else 1})
        elif action == "set_brightness":
            if BRIGHTNESS not in device.address["services"]:
                raise DependencyError(
                    f"WeMo device {device.name!r} has no brightness "
                    f"service (not a dimmer)",
                    dependency="wemo",
                    details={"reason": "no_brightness_service",
                             "device_id": device.device_id})
            value = clamp_int(params.get("value", 0), 0, 255)
            self._soap(device, BRIGHTNESS, "SetBrightness",
                       {"Brightness": value})
        else:  # pragma: no cover — contract validates actions first
            raise DependencyError(
                f"unsupported WeMo action {action!r}", dependency="wemo")
        # The device, read back, is the truth — not the request.
        return self.read_state(device.device_id)
