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

import dataclasses
import socket
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Protocol
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from ...core.errors import DependencyError, NetworkError
from ...core.logging_setup import get_logger
from .contract import BackendCapabilities, HomeBackend, HomeDevice, clamp_int
from .gate import ControlGate
from .subscriptions import (
    WemoSubscription,
    WemoSubscriptionManager,
    parse_state,
)

__all__ = ["UrllibWemoTransport", "WemoBackend", "WemoTransport"]

_log = get_logger(__name__)

SSDP_ADDR = ("239.255.255.250", 1900)
#: wemo.rs: ``ST: urn:Belkin:device:*``.
BELKIN_ST = "urn:Belkin:device:*"
BASIC_EVENT = "urn:Belkin:service:basicevent:1"
BRIGHTNESS = "urn:Belkin:service:brightness:1"
WEMO_PORTS = (49152, 49153, 49154, 49155)
#: wemo.rs: the first control attempt gets 300ms; relocate + retry
#: spend the rest of the budget.
FIRST_ATTEMPT_TIMEOUT_S = 0.3
RETRY_BUDGET_S = 5.0
#: wemo.rs: M-SEARCH is resent every 300ms until the timeout lapses.
SSDP_RESEND_INTERVAL_S = 0.3


class WemoTransport(Protocol):
    """The wire seam.  Implementations move bytes; they hold no
    device logic, so the emulator can stand in for the network."""

    def ssdp_search(self, search_target: str, timeout_s: float
                    ) -> list[dict[str, str]]:
        """M-SEARCH rounds until the timeout; returns response header
        dicts (keys upper-cased).  Raises ``OSError`` when the
        multicast stack itself is unavailable."""
        ...

    def http_get(self, url: str, timeout_s: float) -> tuple[int, bytes]:
        ...

    def http_post(self, url: str, body: bytes, headers: dict[str, str],
                  timeout_s: float) -> tuple[int, bytes]:
        ...

    def http_subscribe(self, url: str, headers: dict[str, str]
                       ) -> dict[str, str]:
        """GENA SUBSCRIBE (new or renewal); returns the response
        headers (``SID``, ``TIMEOUT``).  Raises ``NetworkError`` /
        ``DependencyError`` on failure."""
        ...

    def http_unsubscribe(self, url: str, sid: str) -> None:
        """GENA UNSUBSCRIBE.  Raises ``OSError`` when the device is
        unreachable (the caller drops the record anyway)."""
        ...


class UrllibWemoTransport:
    """The real transport: raw UDP multicast + urllib HTTP."""

    def __init__(self, ssdp_addr: tuple[str, int] | None = None) -> None:
        # Tests point the multicast at a loopback fake device; the
        # default is the real SSDP multicast group.
        self._ssdp_addr = ssdp_addr or SSDP_ADDR

    def ssdp_search(self, search_target: str, timeout_s: float
                    ) -> list[dict[str, str]]:
        # wemo.rs: M-SEARCH with ST urn:Belkin:device:*, MX:5, resent
        # every 300ms until the timeout lapses.
        request = (
            "M-SEARCH * HTTP/1.1\r\n"
            f"HOST: {self._ssdp_addr[0]}:{self._ssdp_addr[1]}\r\n"
            'MAN: "ssdp:discover"\r\n'
            "MX: 5\r\n"
            f"ST: {search_target}\r\n"
            "\r\n"
        ).encode("ascii")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                             socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
            deadline = time.monotonic() + timeout_s
            responses: list[dict[str, str]] = []
            seen: set[bytes] = set()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                sock.settimeout(min(SSDP_RESEND_INTERVAL_S, remaining))
                try:
                    sock.sendto(request, self._ssdp_addr)
                except OSError:
                    raise
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

    def http_subscribe(self, url: str, headers: dict[str, str]
                       ) -> dict[str, str]:
        req = urllib.request.Request(url, method="SUBSCRIBE")
        for key, value in headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
                return {k.upper(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as exc:
            raise DependencyError(
                f"WeMo device at {url} refused SUBSCRIBE "
                f"(HTTP {exc.code})",
                dependency="wemo",
                details={"reason": "subscribe_refused",
                         "http_status": exc.code}) from exc
        except OSError as exc:
            raise NetworkError(
                f"WeMo SUBSCRIBE to {url} failed: {exc}",
                details={"url": url}) from exc

    def http_unsubscribe(self, url: str, sid: str) -> None:
        req = urllib.request.Request(url, method="UNSUBSCRIBE")
        req.add_header("SID", sid)
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()

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
    event_sub_urls: dict[str, str] = {}
    for el in device_el.iter():
        if _local(el.tag) != "service":
            continue
        stype, curl, eurl = "", "", ""
        for sub in el:
            if _local(sub.tag) == "serviceType":
                stype = (sub.text or "").strip()
            elif _local(sub.tag) == "controlURL":
                curl = (sub.text or "").strip()
            elif _local(sub.tag) == "eventSubURL":
                eurl = (sub.text or "").strip()
        if stype and curl:
            base = location.rsplit("/", 1)[0]
            services[stype] = curl if curl.startswith("http") else base + (
                curl if curl.startswith("/") else "/" + curl)
        if stype and eurl:
            base = location.rsplit("/", 1)[0]
            event_sub_urls[stype] = (
                eurl if eurl.startswith("http") else base + (
                    eurl if eurl.startswith("/") else "/" + eurl))
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
            "event_sub_urls": event_sub_urls,
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
        ssdp_addr: tuple[str, int] | None = None,
        retry_budget_s: float = RETRY_BUDGET_S,
        first_attempt_timeout_s: float = FIRST_ATTEMPT_TIMEOUT_S,
    ) -> None:
        super().__init__(gate)
        self._transport = transport or UrllibWemoTransport(
            ssdp_addr=ssdp_addr)
        self._known_hosts = tuple(known_hosts)
        self._ssdp_timeout_s = ssdp_timeout_s
        self._http_timeout_s = http_timeout_s
        self._retry_budget_s = retry_budget_s
        self._first_attempt_timeout_s = first_attempt_timeout_s
        # A transport without GENA support (a test double implementing
        # only the original three-method seam, or a future transport)
        # still constructs: subscribing then refuses by name
        # (``no_subscribe_transport``) instead of AttributeError at
        # construction.  The manager already handles a None fn.
        self._subscriptions = WemoSubscriptionManager(
            subscribe_fn=getattr(
                self._transport, "http_subscribe", None),
            unsubscribe_fn=getattr(
                self._transport, "http_unsubscribe", None))

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend="wemo",
            transport="UPnP/SOAP over HTTP + SSDP discovery (sync)",
            discovery="SSDP M-SEARCH (ST urn:Belkin:device:*, MX:5, "
                      "resent every 300ms) → setup.xml; known-host "
                      "probe fallback; named error when the ladder "
                      "exhausts",
            readable=("binary_state", "insight_params", "brightness"),
            controllable=("turn_on", "turn_off", "toggle",
                          "set_brightness"),
            notes="Brightness only on devices offering the Belkin "
                  "brightness service (dimmers). Control retries once "
                  "after an SSDP relocate (devices change ports). "
                  "Event subscriptions (SUBSCRIBE/NOTIFY, renewed "
                  "every 30s) listen without actuating.",
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
              args: dict[str, Any], timeout_s: float | None = None
              ) -> ET.Element:
        url = device.address["services"][service]
        body = _soap_envelope(service, action, args)
        headers = {
            "Content-Type": 'text/xml; charset="utf-8"',
            "SOAPACTION": f'"{service}#{action}"',
        }
        try:
            status, payload = self._transport.http_post(
                url, body, headers,
                self._http_timeout_s if timeout_s is None else timeout_s)
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
        return self._read_state(device, self._http_timeout_s)

    def _read_state(self, device: HomeDevice,
                    timeout_s: float) -> dict[str, Any]:
        # wemo.rs: GetBinaryState carries <BinaryState>1</BinaryState>.
        root = self._soap(device, BASIC_EVENT, "GetBinaryState",
                          {"BinaryState": 1}, timeout_s=timeout_s)
        raw = _find_text(root, "BinaryState")
        if raw == "":
            raise DependencyError(
                f"WeMo device {device.name!r} returned no BinaryState",
                dependency="wemo",
                details={"reason": "soap_parse",
                         "action": "GetBinaryState"})
        # wemo.rs State: Off=0, On=1, OnWithoutLoad=8 (Insight),
        # anything else numeric is Unknown — the name is the truth,
        # the raw value stays for the record.
        state: dict[str, Any] = parse_state(raw)
        services = device.address["services"]
        if device.address.get("model") == "Insight":
            try:
                root = self._soap(device, BASIC_EVENT,
                                  "GetInsightParams", {},
                                  timeout_s=timeout_s)
                params = _find_text(root, "InsightParams")
                fields = params.split("|")
                if len(fields) > 7 and fields[7]:
                    state["current_power_mw"] = clamp_int(fields[7], 0,
                                                          10 ** 9)
            except (DependencyError, NetworkError) as exc:
                _log.debug("wemo insight params unavailable for %s: %s",
                           device.device_id, exc)
                state["insight_error"] = exc.code
        if BRIGHTNESS in services:
            root = self._soap(device, BRIGHTNESS, "GetBrightness", {},
                              timeout_s=timeout_s)
            bright = _find_text(root, "Brightness")
            if bright:
                state["brightness"] = clamp_int(bright, 0, 255)
        return state

    # ── relocate (wemo.rs: devices change ports — keep last known,
    #    relocate via SSDP by serial, else by IP) ─────────────────────
    def _relocate(self, device: HomeDevice) -> HomeDevice:
        """Re-find a device whose port moved.  Serial first (the
        device is the same device); IP second (a factory reset keeps
        the address but changes the serial).  The registry keeps the
        original device_id — the owner enabled control for that key."""
        try:
            responses = self._transport.ssdp_search(
                BELKIN_ST, self._ssdp_timeout_s)
        except OSError as exc:
            raise NetworkError(
                f"WeMo relocate for {device.name!r}: SSDP failed: {exc}",
                details={"backend": "wemo",
                         "device_id": device.device_id}) from exc
        locations: list[str] = []
        for headers in responses:
            loc = headers.get("LOCATION", "")
            if loc and loc not in locations:
                locations.append(loc)
        serial = device.address.get("serial", "")
        last_ip = urlparse(device.address.get("location", "")).hostname
        by_ip: HomeDevice | None = None
        for loc in locations:
            try:
                status, payload = self._transport.http_get(
                    loc, self._http_timeout_s)
                if status != 200:
                    continue
                parsed = _parse_description(payload, loc)
            except (OSError, DependencyError):
                continue
            if serial and parsed.address.get("serial") == serial:
                return self._remember_relocated(device, parsed)
            if (last_ip and urlparse(loc).hostname == last_ip
                    and by_ip is None):
                by_ip = parsed
        if by_ip is not None:
            return self._remember_relocated(device, by_ip)
        raise NetworkError(
            f"WeMo relocate for {device.name!r}: no device answered "
            f"SSDP with serial {serial!r} or IP {last_ip!r}",
            details={"backend": "wemo", "device_id": device.device_id,
                     "serial": serial, "last_ip": last_ip or ""})

    def _remember_relocated(self, old: HomeDevice,
                            parsed: HomeDevice) -> HomeDevice:
        updated = dataclasses.replace(parsed, device_id=old.device_id)
        self._devices[old.device_id] = updated
        _log.info("wemo relocated %s: %s -> %s", old.name,
                  old.address.get("location"),
                  parsed.address.get("location"))
        return updated

    # ── control (only via HomeBackend.control's gate) ────────────────
    def _control(self, device: HomeDevice, action: str,
                 params: dict[str, Any]) -> dict[str, Any]:
        """wemo.rs ``with_retry``: a 300ms first attempt, then an SSDP
        relocate (by serial, else by IP) and a retry with the
        remaining budget.  Only a transport failure relocates — a
        device that answered with a fault has nothing to relocate to.
        """
        deadline = time.monotonic() + self._retry_budget_s
        try:
            return self._control_once(
                device, action, params,
                timeout_s=self._first_attempt_timeout_s)
        except NetworkError as first_exc:
            if time.monotonic() >= deadline:
                raise
            try:
                relocated = self._relocate(device)
            except (NetworkError, DependencyError) as exc:
                raise NetworkError(
                    f"WeMo control of {device.name!r} failed "
                    f"({first_exc}) and relocate found no device: {exc}",
                    details={"backend": "wemo",
                             "device_id": device.device_id,
                             "action": action}) from first_exc
            remaining = max(0.1, deadline - time.monotonic())
            return self._control_once(relocated, action, params,
                                      timeout_s=remaining)

    def _control_once(self, device: HomeDevice, action: str,
                      params: dict[str, Any],
                      timeout_s: float) -> dict[str, Any]:
        if action == "turn_on":
            self._soap(device, BASIC_EVENT, "SetBinaryState",
                       {"BinaryState": 1}, timeout_s=timeout_s)
        elif action == "turn_off":
            self._soap(device, BASIC_EVENT, "SetBinaryState",
                       {"BinaryState": 0}, timeout_s=timeout_s)
        elif action == "toggle":
            current = self._read_state(device, timeout_s)
            self._soap(device, BASIC_EVENT, "SetBinaryState",
                       {"BinaryState": 0 if current["on"] else 1},
                       timeout_s=timeout_s)
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
                       {"Brightness": value}, timeout_s=timeout_s)
        else:  # pragma: no cover — contract validates actions first
            raise DependencyError(
                f"unsupported WeMo action {action!r}", dependency="wemo")
        # The device, read back, is the truth — not the request.
        return self._read_state(device, timeout_s)

    # ── subscriptions (wemo.rs subscriptions.rs) ─────────────────────
    # Listening only — a subscription never actuates, so it is ungated
    # like a read.  The manager (one per backend) owns the callback
    # listener and the 30s renewal daemon.
    def subscribe(self, device_id: str, *,
                  timeout_s: int = 300,
                  resubscribe_interval_s: float | None = None
                  ) -> WemoSubscription:
        device = self._resolve(device_id)
        event_url = device.address.get("event_sub_urls", {}).get(
            BASIC_EVENT, "")
        if not event_url:
            raise DependencyError(
                f"WeMo device {device.name!r} offers no basicevent "
                f"event subscription URL",
                dependency="wemo",
                details={"reason": "no_event_sub_url",
                         "device_id": device.device_id})
        return self._subscriptions.subscribe(
            device.device_id, device.name, event_url,
            timeout_s=timeout_s,
            resubscribe_interval_s=resubscribe_interval_s)

    def unsubscribe(self, sid: str) -> bool:
        return self._subscriptions.unsubscribe(sid)

    def subscriptions(self) -> list[WemoSubscription]:
        return self._subscriptions.subscriptions()

    def subscription_events(self, sid: str | None = None
                            ) -> list[dict[str, Any]]:
        return self._subscriptions.events(sid)

    def status(self) -> dict[str, Any]:
        base = super().status()
        base["subscriptions"] = [
            sub.to_dict() for sub in self._subscriptions.subscriptions()]
        return base
