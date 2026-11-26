"""Census §3C.1 — wemo.rs port: subscriptions, relocate+retry, SSDP
fidelity, state fidelity, and the smarthome surface.

The tests prove the whole path against a REAL-socket fake WeMo device
(a tiny SSDP/SOAP/SUBSCRIBE responder built here as a test fixture,
never shipped) plus the in-process emulator oracle for the gate seam.
Nothing here talks to real hardware; the no-device path refuses with
a named error, never an empty list or fake success.
"""

from __future__ import annotations

import asyncio
import http.server
import socket
import threading
import time
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from contextlib import suppress
from typing import Any

import pytest

from nomorals.core.errors import DependencyError, NetworkError
from nomorals.integrations.home import ControlGate, WemoBackend
from nomorals.integrations.home.emulators import (
    WemoEmulator,
    WemoEmulatorTransport,
)
from nomorals.integrations.home.errors import ControlNotEnabledError
from nomorals.integrations.home.subscriptions import (
    WemoSubscriptionManager,
    parse_state,
)

PLUG = {"host": "192.168.1.50:49152", "name": "Kitchen Plug",
        "model": "Socket", "serial": "AAA1", "binary_state": 0}


# ── the fake device: real sockets, stdlib only ───────────────────────

def _setup_xml(dev: dict[str, Any], http_port: int) -> bytes:
    services = (
        "<service>"
        "<serviceType>urn:Belkin:service:basicevent:1</serviceType>"
        "<serviceId>urn:upnp-org:serviceId:basicevent1</serviceId>"
        "<controlURL>/upnp/control/basicevent1</controlURL>"
        "<eventSubURL>/upnp/event/basicevent1</eventSubURL>"
        "<SCPDURL>/eventservice.xml</SCPDURL>"
        "</service>")
    if dev.get("model") == "Dimmer":
        services += (
            "<service>"
            "<serviceType>urn:Belkin:service:brightness:1</serviceType>"
            "<serviceId>urn:upnp-org:serviceId:brightness1</serviceId>"
            "<controlURL>/upnp/control/brightness1</controlURL>"
            "<eventSubURL>/upnp/event/brightness1</eventSubURL>"
            "<SCPDURL>/eventservice.xml</SCPDURL>"
            "</service>")
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
        f"<UDN>uuid:Socket-1_0-{dev['serial']}</UDN>"
        f"<serviceList>{services}</serviceList>"
        "</device></root>").encode()


def _soap_envelope(body_inner: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        f"<s:Body>{body_inner}</s:Body></s:Envelope>").encode()


class FakeWemoDevice:
    """A WeMo plug on real sockets: UDP M-SEARCH responder + HTTP
    server speaking setup.xml, SOAP, SUBSCRIBE and NOTIFY."""

    def __init__(self, dev: dict[str, Any] | None = None) -> None:
        self.dev = dict(dev or PLUG)
        self._lock = threading.Lock()
        self.binary_state = int(self.dev.get("binary_state", 0))
        self.msearch_requests: list[str] = []
        self.soap_requests: list[tuple[str, str]] = []
        self.subscribe_requests: list[dict[str, str]] = []
        self.subscribers: dict[str, str] = {}  # sid -> callback url
        self.notifies_sent: list[dict[str, Any]] = []
        self._running = True

        # UDP: M-SEARCH responder on a loopback port.
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.bind(("127.0.0.1", 0))
        self.ssdp_port = self._udp.getsockname()[1]
        self._udp.settimeout(0.2)
        self._udp_thread = threading.Thread(
            target=self._ssdp_loop, daemon=True)
        self._udp_thread.start()

        # HTTP: setup.xml + SOAP + SUBSCRIBE.
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._start_http()

    # ── HTTP server lifecycle (move_port restarts it elsewhere) ──
    def _make_handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        device = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # quiet
                pass

            def _send(self, status: int, body: bytes,
                      ctype: str = "text/xml") -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server name
                if self.path == "/setup.xml":
                    self._send(200, _setup_xml(
                        device.dev, device.http_port))
                else:
                    self._send(404, b"not found", "text/plain")

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def do_POST(self) -> None:  # noqa: N802 - http.server name
                body = self._read_body()
                if self.path.startswith("/upnp/control/"):
                    self._soap(body)
                else:
                    self._send(404, b"not found", "text/plain")

            def _soap(self, body: bytes) -> None:
                action = ""
                for line in body.decode("utf-8", "replace").splitlines():
                    if "GetBinaryState" in line and "xmlns" in line:
                        action = "GetBinaryState"
                    elif "SetBinaryState" in line and "xmlns" in line:
                        action = "SetBinaryState"
                    elif "GetInsightParams" in line and "xmlns" in line:
                        action = "GetInsightParams"
                    elif "GetBrightness" in line and "xmlns" in line:
                        action = "GetBrightness"
                device.soap_requests.append(
                    (action, body.decode("utf-8", "replace")))
                if action == "GetBinaryState":
                    with device._lock:
                        state = device.binary_state
                    self._send(200, _soap_envelope(
                        "<u:GetBinaryStateResponse "
                        'xmlns:u="urn:Belkin:service:basicevent:1">'
                        f"<BinaryState>{state}</BinaryState>"
                        "</u:GetBinaryStateResponse>"))
                elif action == "SetBinaryState":
                    try:
                        root = ET.fromstring(body)
                        value = ""
                        for el in root.iter():
                            if el.tag.rsplit("}", 1)[-1] == "BinaryState":
                                value = (el.text or "").strip()
                        new_state = int(value)
                    except (ET.ParseError, ValueError):
                        new_state = device.binary_state
                    with device._lock:
                        device.binary_state = new_state
                    self._send(200, _soap_envelope(
                        "<u:SetBinaryStateResponse "
                        'xmlns:u="urn:Belkin:service:basicevent:1">'
                        "</u:SetBinaryStateResponse>"))
                    device._notify(new_state)
                elif action == "GetInsightParams":
                    self._send(200, _soap_envelope(
                        "<u:GetInsightParamsResponse "
                        'xmlns:u="urn:Belkin:service:basicevent:1">'
                        "<InsightParams>1|0|0|0|0|0|0|1500000|0"
                        "</InsightParams>"
                        "</u:GetInsightParamsResponse>"))
                else:
                    self._send(500, _soap_envelope(
                        "<s:Fault><faultstring>unknown action"
                        "</faultstring></s:Fault>"))

            def do_SUBSCRIBE(self) -> None:  # noqa: N802 - http.server
                sid = self.headers.get("SID", "")
                callback = self.headers.get("CALLBACK", "")
                timeout = self.headers.get("TIMEOUT", "Second-300")
                if sid:
                    # Renewal of an existing subscription.
                    device.subscribe_requests.append(
                        {"kind": "renew", "sid": sid,
                         "timeout": timeout})
                    self.send_response(200)
                    self.send_header("SID", sid)
                    self.send_header("TIMEOUT", timeout)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    new_sid = f"uuid:{uuid.uuid4()}"
                    cb = callback.strip("<>")
                    with device._lock:
                        device.subscribers[new_sid] = cb
                    device.subscribe_requests.append(
                        {"kind": "new", "sid": new_sid,
                         "callback": cb, "timeout": timeout})
                    self.send_response(200)
                    self.send_header("SID", new_sid)
                    self.send_header("TIMEOUT", timeout)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def do_UNSUBSCRIBE(self) -> None:  # noqa: N802 - http.server
                sid = self.headers.get("SID", "")
                with device._lock:
                    device.subscribers.pop(sid, None)
                device.subscribe_requests.append(
                    {"kind": "unsubscribe", "sid": sid})
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        return Handler

    def _start_http(self) -> None:
        self._httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), self._make_handler())
        self.http_port = self._httpd.server_address[1]
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05},
            daemon=True)
        self._http_thread.start()

    def location(self) -> str:
        return f"http://127.0.0.1:{self.http_port}/setup.xml"

    # ── SSDP ──
    def _ssdp_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._udp.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            text = data.decode("utf-8", "replace")
            if "M-SEARCH" not in text:
                continue
            self.msearch_requests.append(text)
            st = ""
            for line in text.split("\r\n"):
                if line.upper().startswith("ST:"):
                    st = line.split(":", 1)[1].strip()
            if st not in ("urn:Belkin:device:*", "urn:Belkin:device:**",
                          "ssdp:all", "upnp:rootdevice"):
                continue
            reply = (
                "HTTP/1.1 200 OK\r\n"
                "CACHE-CONTROL: max-age=1800\r\n"
                f"LOCATION: {self.location()}\r\n"
                "SERVER: Unspecified, UPnP/1.0, Unspecified\r\n"
                "ST: urn:Belkin:device:controllee:1\r\n"
                f"USN: uuid:Socket-1_0-{self.dev['serial']}::"
                "urn:Belkin:device:controllee:1\r\n"
                "\r\n").encode("ascii")
            with suppress(OSError):
                self._udp.sendto(reply, addr)

    # ── NOTIFY ──
    def _notify(self, new_state: int) -> None:
        with self._lock:
            targets = list(self.subscribers.items())
        for sid, callback in targets:
            def _send_one(sid: str = sid, callback: str = callback,
                          state: int = new_state) -> None:
                body = (
                    '<?xml version="1.0"?>'
                    '<e:propertyset xmlns:e="urn:schemas-upnp-org:event-1-0">'
                    "<e:property>"
                    f"<BinaryState>{state}|{int(time.time())}|0|0|0|0|0|0"
                    "</BinaryState>"
                    "</e:property></e:propertyset>").encode()
                req = urllib.request.Request(
                    callback, data=body, method="NOTIFY")
                req.add_header("NT", "upnp:event")
                req.add_header("NTS", "upnp:propchange")
                req.add_header("SID", sid)
                req.add_header("SEQ", "0")
                req.add_header("Content-Type", "text/xml; charset=utf-8")
                try:
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        resp.read()
                    self.notifies_sent.append(
                        {"sid": sid, "state": state, "ok": True})
                except Exception:  # noqa: BLE001 - delivery is best-effort
                    self.notifies_sent.append(
                        {"sid": sid, "state": state, "ok": False})
            threading.Thread(target=_send_one, daemon=True).start()

    # ── port move (the crate's headline case: devices change ports) ──
    def move_port(self) -> int:
        assert self._httpd is not None
        self._httpd.shutdown()
        self._httpd.server_close()
        self._http_thread.join(timeout=5)
        self._start_http()
        return self.http_port

    def stop(self) -> None:
        self._running = False
        with suppress(OSError):
            self._udp.close()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)


@pytest.fixture()
def fake_device():
    dev = FakeWemoDevice()
    yield dev
    dev.stop()


def make_backend(fake_device, gate=None, **kw):
    from nomorals.integrations.home.wemo import UrllibWemoTransport
    return WemoBackend(
        gate=gate or ControlGate(),
        transport=UrllibWemoTransport(
            ssdp_addr=("127.0.0.1", fake_device.ssdp_port)),
        # The sandbox forbids UDP send, so SSDP never answers here:
        # the known-host ladder is the discovery path in these tests.
        known_hosts=(f"127.0.0.1:{fake_device.http_port}",),
        ssdp_timeout_s=1.0, **kw)


# ── SSDP, without real UDP (the sandbox forbids UDP send entirely) ───
# The transport's socket seam is stubbed at the socket module: the
# stub records the M-SEARCH bytes and replays canned responses, so the
# request shape, the resend cadence and the response parsing are all
# proven without a datagram ever leaving the process.  Non-UDP sockets
# (the fake device's HTTP server) still use the real socket class.

class _StubUdpSocket:
    """A UDP socket that records sends and replays canned receives.

    ``recvfrom`` blocks for the socket's timeout (like a real
    socket) when nothing is queued, so the transport's resend
    cadence is genuinely exercised, not spun in a tight loop."""

    def __init__(self, responses=None, *, fail_send=None):
        self.sent: list[tuple[bytes, tuple]] = []
        self._responses = list(responses or [])
        self._fail_send = fail_send
        self._timeout: float | None = None

    def setsockopt(self, *args):
        pass

    def settimeout(self, timeout):
        self._timeout = timeout

    def sendto(self, data, addr):
        if self._fail_send is not None:
            raise self._fail_send
        self.sent.append((data, addr))
        return len(data)

    def recvfrom(self, n):
        if self._responses:
            return self._responses.pop(0)
        time.sleep(self._timeout or 0)
        raise TimeoutError

    def close(self):
        pass


def _patch_udp(monkeypatch, responses=None, *, fail_send=None):
    real_socket = socket.socket
    stub = _StubUdpSocket(responses, fail_send=fail_send)

    def factory(family=-1, type=-1, proto=-1, fileno=None):
        if family == socket.AF_INET and type == socket.SOCK_DGRAM:
            return stub
        return real_socket(family, type, proto, fileno)

    monkeypatch.setattr(socket, "socket", factory)
    return stub


def _ssdp_response(location: str, serial: str = "AAA1") -> bytes:
    return (
        "HTTP/1.1 200 OK\r\n"
        "CACHE-CONTROL: max-age=1800\r\n"
        f"LOCATION: {location}\r\n"
        "SERVER: Unspecified, UPnP/1.0, Unspecified\r\n"
        "ST: urn:Belkin:device:controllee:1\r\n"
        f"USN: uuid:Socket-1_0-{serial}::"
        "urn:Belkin:device:controllee:1\r\n"
        "\r\n").encode("ascii")


# ── SSDP fidelity (wemo.rs: M-SEARCH ST urn:Belkin:device:*, MX:5,
#    resend every 300ms until timeout) ─────────────────────────────────
# Proven against the stubbed socket (see above): the sandbox forbids
# UDP send, so no datagram can leave this process.

class TestSsdpFidelity:
    def test_msearch_uses_crate_search_target_and_mx(self, monkeypatch):
        from nomorals.integrations.home.wemo import UrllibWemoTransport
        stub = _patch_udp(monkeypatch)
        transport = UrllibWemoTransport()
        transport.ssdp_search("urn:Belkin:device:*", 0.35)
        assert stub.sent, "no M-SEARCH was sent"
        first = stub.sent[0][0].decode("ascii")
        assert "ST: urn:Belkin:device:*" in first
        assert "MX: 5" in first
        assert 'MAN: "ssdp:discover"' in first
        assert "HOST: 239.255.255.250:1900" in first

    def test_msearch_resends_until_timeout(self, monkeypatch):
        from nomorals.integrations.home.wemo import UrllibWemoTransport
        stub = _patch_udp(monkeypatch)
        transport = UrllibWemoTransport()
        started = time.monotonic()
        transport.ssdp_search("urn:Belkin:device:*", 1.0)
        elapsed = time.monotonic() - started
        # 1.0s window, 300ms cadence → at least 3 sends.
        assert len(stub.sent) >= 3
        assert 0.9 <= elapsed <= 1.5

    def test_msearch_parses_responses(self, monkeypatch, fake_device):
        from nomorals.integrations.home.wemo import UrllibWemoTransport
        response = _ssdp_response(fake_device.location())
        _patch_udp(monkeypatch, [(response, ("127.0.0.1", 1901))])
        transport = UrllibWemoTransport()
        found = transport.ssdp_search("urn:Belkin:device:*", 0.35)
        assert len(found) == 1
        assert found[0]["LOCATION"] == fake_device.location()
        assert found[0]["ST"] == "urn:Belkin:device:controllee:1"

    def test_ssdp_stack_failure_is_named(self, monkeypatch):
        from nomorals.integrations.home.wemo import UrllibWemoTransport
        _patch_udp(monkeypatch, fail_send=OSError(
            "[Errno 1] Operation not permitted"))
        backend = WemoBackend(
            gate=ControlGate(),
            transport=UrllibWemoTransport(), ssdp_timeout_s=0.5)
        with pytest.raises(DependencyError) as ctx:
            backend.discover()
        assert "ssdp" in str(ctx.value).lower()

    def test_no_device_refuses_by_name(self, monkeypatch):
        from nomorals.integrations.home.wemo import UrllibWemoTransport
        _patch_udp(monkeypatch)  # silence: no responses at all
        backend = WemoBackend(
            gate=ControlGate(),
            transport=UrllibWemoTransport(), ssdp_timeout_s=0.5)
        with pytest.raises(NetworkError) as ctx:
            backend.discover()
        assert "no devices" in str(ctx.value).lower()

    def test_discovery_falls_back_to_known_hosts(self, fake_device):
        # In this sandbox SSDP raises (UDP send forbidden) and the
        # known-host ladder carries discovery — the ladder, not
        # silence, is the answer.
        backend = make_backend(fake_device)
        devices = backend.discover()
        assert [d.name for d in devices] == ["Kitchen Plug"]
        assert devices[0].device_id == "uuid:Socket-1_0-AAA1"
        assert devices[0].device_type == "switch"
        assert devices[0].address["serial"] == "AAA1"
        assert any("ladder: probing known hosts" in note
                   for note in backend.last_discovery_notes)


# ── state fidelity (wemo.rs: Off/On/OnWithoutLoad/Unknown; GetBinaryState
#    carries <BinaryState>1</BinaryState>) ─────────────────────────────

class TestStateFidelity:
    def test_get_binary_state_carries_arg(self, fake_device):
        backend = make_backend(fake_device)
        devices = backend.discover()
        backend.read_state(devices[0].device_id)
        gets = [b for a, b in fake_device.soap_requests
                if a == "GetBinaryState"]
        assert gets, "no GetBinaryState reached the fake"
        assert "<BinaryState>1</BinaryState>" in gets[0]

    def test_state_names(self, fake_device):
        backend = make_backend(fake_device)
        devices = backend.discover()
        device_id = devices[0].device_id
        for raw, name, on in ((0, "off", False), (1, "on", True),
                              (8, "on_without_load", True),
                              (5, "unknown", True)):
            with fake_device._lock:
                fake_device.binary_state = raw
            state = backend.read_state(device_id)
            assert state["binary_state"] == raw
            assert state["state"] == name, (raw, state)
            assert state["on"] is on

    def test_insight_first_field_is_the_state(self, fake_device):
        # The crate reads only the first pipe field of InsightParams.
        assert parse_state("1|0|0|0|0|0|0|1500000")["state"] == "on"
        assert parse_state("8|1234567890|0|0")["state"] == "on_without_load"
        assert parse_state("0|0")["state"] == "off"
        assert parse_state("9|0")["state"] == "unknown"
        with pytest.raises(DependencyError):
            parse_state("")
        with pytest.raises(DependencyError):
            parse_state("xx|0")


# ── relocate + retry (wemo.rs with_retry: 300ms first attempt →
#    SSDP relocate by serial else IP → retry with remaining budget) ───
# The SSDP leg runs through the stubbed socket (canned response =
# the fake's new location); the HTTP legs run over real loopback TCP.

class TestRelocateRetry:
    def test_control_relocates_after_port_move(
            self, fake_device, monkeypatch):
        gate = ControlGate()
        backend = make_backend(fake_device, gate=gate)
        devices = backend.discover()
        device_id = devices[0].device_id
        gate.set_enabled("wemo", device_id, True)
        old_port = fake_device.http_port
        new_port = fake_device.move_port()
        assert new_port != old_port
        _patch_udp(monkeypatch, [(
            _ssdp_response(fake_device.location()),
            ("127.0.0.1", 1901))])
        # The old port is dead; control must relocate and still land.
        result = backend.control(device_id, "turn_on")
        assert result["state"]["on"] is True
        assert result["state"]["binary_state"] == 1
        with fake_device._lock:
            assert fake_device.binary_state == 1
        # The registry now tracks the new port.
        device = backend.known_devices()[0]
        assert f":{new_port}/" in device.address["services"][
            "urn:Belkin:service:basicevent:1"]

    def test_relocate_by_ip_when_serial_unmatched(
            self, fake_device, monkeypatch):
        # A device that answers SSDP under a different serial (factory
        # reset) but the same IP is still found by IP.
        gate = ControlGate()
        backend = make_backend(fake_device, gate=gate)
        devices = backend.discover()
        device_id = devices[0].device_id
        gate.set_enabled("wemo", device_id, True)
        fake_device.move_port()
        fake_device.dev["serial"] = "RESET9"
        _patch_udp(monkeypatch, [(
            _ssdp_response(fake_device.location(), serial="RESET9"),
            ("127.0.0.1", 1901))])
        result = backend.control(device_id, "turn_off")
        assert result["state"]["on"] is False

    def test_relocate_exhausted_is_a_named_error(
            self, fake_device, monkeypatch):
        gate = ControlGate()
        backend = make_backend(fake_device, gate=gate)
        devices = backend.discover()
        device_id = devices[0].device_id
        gate.set_enabled("wemo", device_id, True)
        fake_device.stop()  # device gone entirely
        _patch_udp(monkeypatch)  # SSDP silent too
        with pytest.raises((NetworkError, DependencyError)):
            backend.control(device_id, "turn_on")

    def test_gate_still_blocks_relocated_control(self, fake_device):
        backend = make_backend(fake_device)
        devices = backend.discover()
        device_id = devices[0].device_id
        fake_device.move_port()
        with pytest.raises(ControlNotEnabledError):
            backend.control(device_id, "turn_on")
        with fake_device._lock:
            assert fake_device.binary_state == 0


# ── subscriptions (wemo.rs subscriptions.rs) ──────────────────────────

class TestSubscriptions:
    def test_subscribe_receives_notify(self, fake_device):
        backend = make_backend(fake_device)
        devices = backend.discover()
        device_id = devices[0].device_id
        backend._gate.set_enabled("wemo", device_id, True)
        sub = backend.subscribe(device_id)
        assert sub.sid.startswith("uuid:")
        assert sub.device_id == device_id
        backend.control(device_id, "turn_on")
        deadline = time.time() + 5
        while time.time() < deadline:
            if sub.last_state and sub.last_state.get("on") is True:
                break
            time.sleep(0.05)
        assert sub.last_state is not None
        assert sub.last_state["state"] == "on"
        assert sub.last_state["binary_state"] == 1
        assert len(sub.events) >= 1
        assert fake_device.subscribe_requests[0]["kind"] == "new"
        backend.unsubscribe(sub.sid)
        backend._subscriptions.stop()
        assert backend.subscriptions() == []

    def test_resubscribe_renews_sid(self, fake_device):
        backend = make_backend(fake_device)
        devices = backend.discover()
        device_id = devices[0].device_id
        sub = backend.subscribe(device_id, resubscribe_interval_s=0.2)
        deadline = time.time() + 3
        while time.time() < deadline:
            renewals = [r for r in fake_device.subscribe_requests
                        if r["kind"] == "renew"]
            if len(renewals) >= 2:
                break
            time.sleep(0.05)
        renewals = [r for r in fake_device.subscribe_requests
                    if r["kind"] == "renew"]
        assert len(renewals) >= 2
        assert all(r["sid"] == sub.sid for r in renewals)
        backend.unsubscribe(sub.sid)
        backend._subscriptions.stop()

    def test_subscribe_unknown_device_is_named(self, fake_device):
        backend = make_backend(fake_device)
        with pytest.raises(Exception) as ctx:
            backend.subscribe("uuid:nope")
        assert "not found" in str(ctx.value).lower()

    def test_manager_parse_state_unit(self):
        mgr = WemoSubscriptionManager()
        try:
            parsed = mgr.parse_notify_body(
                '<?xml version="1.0"?>'
                '<e:propertyset xmlns:e="urn:schemas-upnp-org:event-1-0">'
                "<e:property><BinaryState>8|1|2</BinaryState>"
                "</e:property></e:propertyset>")
            assert parsed["state"] == "on_without_load"
            assert parsed["binary_state"] == 8
        finally:
            mgr.stop()


# ── emulator oracle keeps the crate's ST form + SUBSCRIBE ─────────────

class TestEmulatorOracle:
    def test_emulator_accepts_crate_search_target(self):
        emulator = WemoEmulator([PLUG])
        backend = WemoBackend(
            gate=ControlGate(),
            transport=WemoEmulatorTransport(emulator))
        devices = backend.discover()
        assert [d.name for d in devices] == ["Kitchen Plug"]

    def test_emulator_subscribe_records_callback(self):
        emulator = WemoEmulator([PLUG])
        backend = WemoBackend(
            gate=ControlGate(),
            transport=WemoEmulatorTransport(emulator))
        devices = backend.discover()
        sub = backend.subscribe(devices[0].device_id)
        assert sub.sid in emulator.subscriptions
        assert emulator.subscriptions[sub.sid]["callback"].startswith(
            "http://")
        backend.unsubscribe(sub.sid)
        backend._subscriptions.stop()
        assert sub.sid not in emulator.subscriptions


# ── the smarthome surface: tool + CLI + chat ──────────────────────────

class TestSmarthomeSurface:
    @pytest.fixture()
    def wired(self, fake_device):
        from nomorals.integrations.smarthome_integration import (
            SmartHomeIntegration,
        )
        from nomorals.tools import smarthome as tool

        backend = make_backend(fake_device)
        gate = backend._gate
        # One gate, shared: the backend checks the same gate the
        # integration's enable/disable writes (an injected backend
        # keeps its own gate unless the integration is given it).
        integration = SmartHomeIntegration(
            _NoCredentials(), _NoSessions(), wemo_backend=backend,
            control_gate=gate)
        tool.init_smarthome(integration)
        yield integration, backend, gate, fake_device
        backend._subscriptions.stop()
        tool.init_smarthome(None)

    def test_tool_discover_and_status(self, wired):
        from nomorals.tools import smarthome as tool
        _, backend, _, _ = wired
        found = asyncio.run(tool.smarthome(action="discover"))
        assert found["ok"] is True
        assert [d["name"] for d in found["devices"]] == ["Kitchen Plug"]
        device_id = found["devices"][0]["device_id"]
        status = asyncio.run(
            tool.smarthome(action="status", device_id=device_id))
        assert status["state"]["state"] == "off"
        assert status["control_enabled"] is False

    def test_tool_control_gated_until_enabled(self, wired):
        from nomorals.tools import smarthome as tool
        _, _, gate, fake_device = wired
        found = asyncio.run(tool.smarthome(action="discover"))
        device_id = found["devices"][0]["device_id"]
        refused = asyncio.run(
            tool.smarthome(action="on", device_id=device_id))
        assert refused["ok"] is False
        assert "not enabled" in refused["error"].lower()
        enabled = asyncio.run(
            tool.smarthome(action="enable", device_id=device_id))
        assert enabled["ok"] is True
        assert gate.is_enabled("wemo", device_id) is True
        done = asyncio.run(
            tool.smarthome(action="on", device_id=device_id))
        assert done["ok"] is True
        assert done["state"]["on"] is True
        assert fake_device.binary_state == 1

    def test_tool_subscribe_round_trip(self, wired):
        from nomorals.tools import smarthome as tool
        _, _, _, fake_device = wired
        found = asyncio.run(tool.smarthome(action="discover"))
        device_id = found["devices"][0]["device_id"]
        asyncio.run(tool.smarthome(action="enable", device_id=device_id))
        sub = asyncio.run(
            tool.smarthome(action="subscribe", device_id=device_id))
        assert sub["ok"] is True
        assert sub["sid"].startswith("uuid:")
        asyncio.run(tool.smarthome(action="on", device_id=device_id))
        deadline = time.time() + 5
        events = []
        while time.time() < deadline:
            events = asyncio.run(tool.smarthome(
                action="events", sid=sub["sid"]))["events"]
            if events:
                break
            time.sleep(0.05)
        assert events and events[0]["state"] == "on"
        listing = asyncio.run(tool.smarthome(action="subscriptions"))
        assert [s["sid"] for s in listing["subscriptions"]] == [sub["sid"]]
        gone = asyncio.run(
            tool.smarthome(action="unsubscribe", sid=sub["sid"]))
        assert gone["ok"] is True

    def test_tool_unknown_action_is_named(self, wired):
        from nomorals.tools import smarthome as tool
        result = asyncio.run(tool.smarthome(action="explode"))
        assert result["ok"] is False
        assert "unknown action" in result["error"].lower()

    def test_cli_parser(self):
        from nomorals.cmdline.parser import _parser
        ns = _parser().parse_args(["smarthome", "discover"])
        assert ns.smarthome_action == "discover"
        ns = _parser().parse_args(
            ["smarthome", "on", "--device", "uuid:x"])
        assert ns.smarthome_action == "on"
        assert ns.device == "uuid:x"
        ns = _parser().parse_args(["smh", "status", "--device", "uuid:x"])
        assert ns.smarthome_action == "status"

    def test_cli_command_runs(self, wired, capsys):
        from nomorals.cmdline.commands.smarthome import _cmd_smarthome
        from nomorals.cmdline.parser import _parser
        args = _parser().parse_args(["smarthome", "discover"])
        assert _cmd_smarthome(args, None) == 0
        out = capsys.readouterr().out
        assert "Kitchen Plug" in out

    def test_chat_catalog(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_ONELINERS,
        )
        assert "smarthome" in CONTROL_COMMANDS
        assert CONTROL_COMMANDS["smarthome"][0] == 0  # owner only
        assert "smarthome" in COMMAND_DETAILS
        assert "smarthome" in LIST_ONELINERS

    def test_chat_handler(self, wired):
        from nomorals.agents.partner.runtime_smarthome import (
            RuntimeSmarthomeMixin,
        )
        found = RuntimeSmarthomeMixin()._control_smarthome(
            "discover", "test-key")
        assert "Kitchen Plug" in found

    def test_tool_count_is_786_after_c1_added_smarthome(self):
        # 782 after this lane (C.1) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label; 794 after gapfix163 added 5 trendclip tools (trendclip_discover, trendclip_analyze, trendclip_render, trendclip_status, trendclip_queue). 800 after gapfix146 added 6 visiondoc tools (vision_document, document_word, document_html, document_text, document_pdf, document_sheet). 801 after gapfix124 added browser_watch. 802 after this lane (SongRecon surface) added songrecon. 803 after gapfix122 added file. 804 after gapfix129 added group_cleanup. 805 after gapfix130 added group_heist. 806 after gapfix119 added matching. 807 after this lane (agents192) added fleet. 808 after this lane (F3) added control.
        from types import SimpleNamespace

        from nomorals.tools.registry import ToolRegistry
        count = len(ToolRegistry(SimpleNamespace()).register_builtins()
                    ._tools)
        assert count == 808


class _NoCredentials:
    """Credential manager with nothing stored (env-only config)."""

    def get_credential(self, *args: Any, **kwargs: Any) -> Any:
        raise KeyError("no credential")


class _NoSessions:
    """Session manager stand-in; the home integration never uses it."""
