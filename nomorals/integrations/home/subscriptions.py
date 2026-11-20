"""WeMo event subscriptions — a pattern-port of ``wemo.rs``
``subscriptions.rs`` (BSD-4-Clause, Brandon Thomas).

The crate's shape, in Python:

* **SUBSCRIBE** — ``SUBSCRIBE <eventSubURL>`` with
  ``CALLBACK: <http://local:port/?from=host>``,
  ``NT: upnp:event`` and ``TIMEOUT: Second-<ttl>``; the device
  answers with a ``SID`` and a ``TIMEOUT``.
* **NOTIFY** — the device POSTs a GENA propertyset to the callback
  URL whenever the state changes; :func:`parse_state` reads the
  pipe-delimited ``BinaryState`` value (the crate reads only the
  first field — the state — and caps the value at 65535).
* **Renewal** — a background thread re-subscribes every 30s (the
  crate's cadence), so a subscription outlives its timeout for as
  long as the device answers.
* **Callback listener** — a stdlib ``http.server`` listener on an
  ephemeral port (the crate uses Iron; the shape is the same: one
  listener owned by the subscribing instance, never a global hub).

Subscriptions are *listening*, not control: subscribing never
actuates a device, so it is ungated like a read.  The manager is
owned by one backend instance (per-device/per-session — no central
hub, owner law).
"""

from __future__ import annotations

import http.server
import socket
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from ...core.errors import DependencyError
from ...core.logging_setup import get_logger
from .contract import clamp_int

__all__ = [
    "RESUBSCRIBE_INTERVAL_S",
    "SUBSCRIBE_TIMEOUT_S",
    "WemoSubscription",
    "WemoSubscriptionManager",
    "get_local_ip_for",
    "parse_notify_body",
    "parse_state",
    "state_name",
]

_log = get_logger(__name__)

#: The crate's default subscription TTL (``TIMEOUT: Second-<ttl>``).
SUBSCRIBE_TIMEOUT_S = 300
#: The crate's re-subscribe cadence: a background thread renews every 30s.
RESUBSCRIBE_INTERVAL_S = 30.0

#: wemo.rs ``State``: Off=0, On=1, OnWithoutLoad=8 (Insight),
#: anything else is Unknown.  ``from_u64`` caps the value at 65535.
_STATE_NAMES = {0: "off", 1: "on", 8: "on_without_load"}


def state_name(raw: Any) -> str:
    """The crate's state name for a BinaryState value.

    Non-numeric input is a protocol failure, never a guess.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise DependencyError(
            f"WeMo BinaryState {raw!r} is not a number",
            dependency="wemo",
            details={"reason": "bad_binary_state"}) from exc
    return _STATE_NAMES.get(clamp_int(value, 0, 65535), "unknown")


def parse_state(value: str) -> dict[str, Any]:
    """Parse a pipe-delimited BinaryState value (the crate's
    ``parse_state``): the first field is the state, the rest are the
    Insight counters.  ``"1|ts|..."`` → on; ``"8|..."`` → on, load
    unknown; anything else numeric → unknown."""
    if not isinstance(value, str) or not value.strip():
        raise DependencyError(
            "WeMo BinaryState value is empty",
            dependency="wemo",
            details={"reason": "bad_binary_state"})
    first = value.split("|", 1)[0].strip()
    name = state_name(first)
    return {
        "binary_state": clamp_int(first, 0, 65535),
        "state": name,
        "on": name != "off",
        "raw": value,
    }


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_notify_body(body: bytes) -> dict[str, Any]:
    """Parse a GENA NOTIFY propertyset body into a state dict."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise DependencyError(
            "WeMo NOTIFY body is not parseable XML",
            dependency="wemo",
            details={"reason": "notify_parse"}) from exc
    for el in root.iter():
        if _local(el.tag) == "BinaryState":
            return parse_state((el.text or "").strip())
    raise DependencyError(
        "WeMo NOTIFY body carries no BinaryState property",
        dependency="wemo",
        details={"reason": "notify_parse"})


def get_local_ip_for(peer_host: str) -> str:
    """The address this box would use to reach ``peer_host`` — the
    callback address a device on that network can answer.

    The crate's ``get_local_ip`` is the first non-loopback non-docker
    IPv4; the UDP-connect trick returns the same interface for the
    peer's network, and falls back to the hostname's address, then
    loopback.  A loopback peer (a local fake device) gets loopback.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((peer_host, 80))
            ip = sock.getsockname()[0]
            if ip:
                return ip
        finally:
            sock.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    return "127.0.0.1"


@dataclass
class WemoSubscription:
    """One live GENA subscription on one device."""

    sid: str
    device_id: str
    device_name: str
    event_url: str
    callback_url: str
    timeout_s: int = SUBSCRIBE_TIMEOUT_S
    created_at: float = field(default_factory=time.time)
    last_renewed_at: float = field(default_factory=time.time)
    renewals: int = 0
    last_state: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    last_renewal_error: str | None = None

    def record_event(self, state: dict[str, Any]) -> None:
        """Land one NOTIFY: the state is the newest truth, the event
        log keeps the history."""
        self.last_state = dict(state)
        self.events.append({**state, "at": time.time()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "event_url": self.event_url,
            "callback_url": self.callback_url,
            "timeout_s": self.timeout_s,
            "created_at": self.created_at,
            "last_renewed_at": self.last_renewed_at,
            "renewals": self.renewals,
            "last_state": dict(self.last_state) if self.last_state else None,
            "events": len(self.events),
            "last_renewal_error": self.last_renewal_error,
        }


class WemoSubscriptionManager:
    """Owns the callback listener and the re-subscribe daemon for one
    backend instance.  One manager per backend — never a global hub.

    The SUBSCRIBE/UNSUBSCRIBE wire calls go through the backend's
    transport (injected callables), so the emulator oracle can stand
    in for the network exactly like it does for SOAP; only the NOTIFY
    listener is a real server (something has to receive the device's
    calls)."""

    def __init__(self, *, resubscribe_interval_s: float
                 = RESUBSCRIBE_INTERVAL_S,
                 subscribe_fn: Any | None = None,
                 unsubscribe_fn: Any | None = None) -> None:
        self._resubscribe_interval_s = resubscribe_interval_s
        self._subscribe_fn = subscribe_fn
        self._unsubscribe_fn = unsubscribe_fn
        self._subscriptions: dict[str, WemoSubscription] = {}
        self._lock = threading.Lock()
        self._listener: http.server.ThreadingHTTPServer | None = None
        self._listener_thread: threading.Thread | None = None
        self._resub_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── listener ───────────────────────────────────────────────────
    def _ensure_listener(self) -> int:
        """Start the callback listener (once); return its port."""
        if self._listener is not None:
            return self._listener.server_address[1]
        manager = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # quiet
                pass

            def _reply(self, status: int) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802 - http.server name
                # Some devices probe the callback URL before NOTIFYing.
                self._reply(200)

            def do_NOTIFY(self) -> None:  # noqa: N802 - http.server name
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                sid = (self.headers.get("SID") or "").strip()
                try:
                    manager._handle_notify(sid, body)
                except DependencyError as exc:
                    _log.debug("wemo notify dropped: %s", exc)
                self._reply(200)

        self._listener = http.server.ThreadingHTTPServer(
            ("0.0.0.0", 0), Handler)
        port = self._listener.server_address[1]
        self._listener_thread = threading.Thread(
            target=self._listener.serve_forever,
            kwargs={"poll_interval": 0.05}, daemon=True)
        self._listener_thread.start()
        _log.info("wemo subscription listener on port %s", port)
        return port

    # ── subscribe / unsubscribe ────────────────────────────────────
    def subscribe(self, device_id: str, device_name: str, event_url: str,
                  *, timeout_s: int = SUBSCRIBE_TIMEOUT_S,
                  resubscribe_interval_s: float | None = None
                  ) -> WemoSubscription:
        """SUBSCRIBE one device's event URL.  Returns the live
        subscription (SID, callback URL, timeout)."""
        from urllib.parse import urlparse
        peer_host = urlparse(event_url).hostname or "127.0.0.1"
        port = self._ensure_listener()
        callback = (f"http://{get_local_ip_for(peer_host)}:{port}/"
                    f"?from={peer_host}")
        if self._subscribe_fn is None:
            raise DependencyError(
                "WeMo subscription manager has no SUBSCRIBE transport",
                dependency="wemo",
                details={"reason": "no_subscribe_transport",
                         "device_id": device_id})
        headers = {
            "CALLBACK": f"<{callback}>",
            "NT": "upnp:event",
            "TIMEOUT": f"Second-{timeout_s}",
        }
        resp_headers = self._subscribe_fn(event_url, headers)
        sid = (resp_headers.get("SID") or "").strip()
        timeout_hdr = (resp_headers.get("TIMEOUT") or "").strip()
        if not sid:
            raise DependencyError(
                f"WeMo device {device_name!r} subscribed without a SID",
                dependency="wemo",
                details={"reason": "subscribe_no_sid",
                         "device_id": device_id})
        granted = timeout_s
        if timeout_hdr.lower().startswith("second-"):
            with suppress(ValueError):
                granted = int(timeout_hdr.split("-", 1)[1])
        sub = WemoSubscription(
            sid=sid, device_id=device_id, device_name=device_name,
            event_url=event_url, callback_url=callback,
            timeout_s=granted)
        with self._lock:
            self._subscriptions[sid] = sub
        if resubscribe_interval_s is not None:
            self._resubscribe_interval_s = resubscribe_interval_s
        self._ensure_resub_thread()
        _log.info("wemo subscribed %s (%s) sid=%s", device_name, device_id,
                  sid)
        return sub

    def unsubscribe(self, sid: str) -> bool:
        """UNSUBSCRIBE one subscription.  The local record is dropped
        even when the device is already gone — a dead device must not
        hold a subscription hostage."""
        with self._lock:
            sub = self._subscriptions.pop(sid, None)
        if sub is None:
            return False
        if self._unsubscribe_fn is not None:
            try:
                self._unsubscribe_fn(sub.event_url, sid)
            except OSError as exc:
                _log.debug(
                    "wemo unsubscribe for %s: device unreachable: %s",
                    sid, exc)
        _log.info("wemo unsubscribed sid=%s", sid)
        return True

    def subscriptions(self) -> list[WemoSubscription]:
        with self._lock:
            return list(self._subscriptions.values())

    def events(self, sid: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            subs = ([self._subscriptions[sid]] if sid
                    and sid in self._subscriptions
                    else list(self._subscriptions.values()))
        out: list[dict[str, Any]] = []
        for sub in subs:
            for event in sub.events:
                out.append({**event, "sid": sub.sid,
                            "device_id": sub.device_id})
        return out

    def parse_notify_body(self, body: bytes) -> dict[str, Any]:
        """Parse a NOTIFY propertyset body (module function, bound for
        the surface)."""
        return parse_notify_body(body)

    # ── NOTIFY intake ───────────────────────────────────────────────
    def _handle_notify(self, sid: str, body: bytes) -> None:
        with self._lock:
            sub = self._subscriptions.get(sid)
        if sub is None:
            _log.debug("wemo notify for unknown sid %s dropped", sid)
            return
        sub.record_event(parse_notify_body(body))

    # ── renewal daemon ─────────────────────────────────────────────
    def _ensure_resub_thread(self) -> None:
        if self._resub_thread is not None and self._resub_thread.is_alive():
            return
        self._stop.clear()
        self._resub_thread = threading.Thread(
            target=self._resub_loop, daemon=True)
        self._resub_thread.start()

    def _resub_loop(self) -> None:
        while not self._stop.wait(self._resubscribe_interval_s):
            with self._lock:
                subs = list(self._subscriptions.values())
            for sub in subs:
                self._renew(sub)

    def _renew(self, sub: WemoSubscription) -> None:
        if self._subscribe_fn is None:
            return
        try:
            self._subscribe_fn(sub.event_url, {
                "SID": sub.sid,
                "TIMEOUT": f"Second-{sub.timeout_s}",
            })
        except OSError as exc:
            sub.last_renewal_error = str(exc)
            _log.debug("wemo resubscribe sid=%s failed: %s", sub.sid, exc)
            return
        sub.last_renewed_at = time.time()
        sub.renewals += 1
        sub.last_renewal_error = None

    def stop(self) -> None:
        """Stop the daemon and the listener.  Subscriptions end with
        the manager — nothing outlives its owner."""
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.shutdown()
                self._listener.server_close()
            except OSError:
                pass
            self._listener = None
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5)
            self._listener_thread = None
        if self._resub_thread is not None:
            self._resub_thread.join(timeout=5)
            self._resub_thread = None
        with self._lock:
            self._subscriptions.clear()
