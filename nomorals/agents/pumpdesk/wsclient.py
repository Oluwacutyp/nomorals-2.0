"""A minimal RFC 6455 WebSocket client — stdlib only.

The pump firehose needs live sockets on hosts where neither
``websockets`` nor ``websocket-client`` is guaranteed (the core
package carries zero mandatory dependencies; Termux ARM may have
no wheels at all).  This client is deliberately small and complete
for the client role: HTTP CONNECT proxy tunnelling from the
standard proxy env (the same convention urllib honours), TLS via
``ssl``, masked client frames, fragmentation reassembly,
ping -> pong, close handshake.

It is a transport only — no pump knowledge lives here.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import ssl
import struct
from typing import Optional
from urllib.parse import urlparse

__all__ = ["WebSocketClient", "WSClosed", "WSHandshakeError"]


class WSClosed(Exception):
    pass


class WSHandshakeError(Exception):
    pass


_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _proxy_for(url: str) -> Optional[str]:
    """The proxy the standard env names for this URL, or None.
    no_proxy hosts connect directly (urllib's convention)."""
    u = urlparse(url)
    host = (u.hostname or "").lower()
    no_proxy = (os.environ.get("no_proxy") or os.environ.get("NO_PROXY")
                or "")
    for entry in no_proxy.split(","):
        entry = entry.strip().lower().lstrip(".")
        if entry and (host == entry or host.endswith("." + entry)):
            return None
    if u.scheme == "wss":
        return (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
                or os.environ.get("all_proxy") or os.environ.get("ALL_PROXY"))
    return (os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
            or os.environ.get("all_proxy") or os.environ.get("ALL_PROXY"))


class WebSocketClient:
    """One blocking WebSocket connection.

    ``recv(timeout)`` returns the next complete text/binary message
    as str, answering pings transparently.  Raises WSClosed on a
    clean close, TimeoutError (socket.timeout) when the timeout
    expires with no message, OSError on transport failure."""

    def __init__(self, url: str, timeout: float = 20.0,
                 user_agent: str = "nomorals-pumpdesk"):
        self.url = url
        self.timeout = timeout
        self.user_agent = user_agent
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._frag = bytearray()
        self._frag_op: Optional[int] = None

    # ── connection ────────────────────────────────────────────────

    def _open_tunnel(self, host: str, port: int) -> socket.socket:
        proxy = _proxy_for(self.url)
        if not proxy:
            return socket.create_connection((host, port),
                                            timeout=self.timeout)
        pu = urlparse(proxy)
        sock = socket.create_connection((pu.hostname, pu.port or 3128),
                                        timeout=self.timeout)
        auth = ""
        if pu.username:
            cred = base64.b64encode(
                f"{pu.username}:{pu.password or ''}".encode()).decode()
            auth = f"Proxy-Authorization: Basic {cred}\r\n"
        sock.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\n"
                     f"Host: {host}:{port}\r\n{auth}\r\n".encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = sock.recv(4096)
            if not chunk:
                raise WSHandshakeError("proxy closed during CONNECT")
            resp += chunk
        status = resp.split(b"\r\n", 1)[0].decode(errors="replace")
        if " 200" not in status:
            raise WSHandshakeError(f"proxy CONNECT failed: {status}")
        return sock

    def connect(self) -> "WebSocketClient":
        u = urlparse(self.url)
        if u.scheme not in ("ws", "wss"):
            raise WSHandshakeError(f"unsupported scheme {u.scheme!r}")
        host = u.hostname or ""
        port = u.port or (443 if u.scheme == "wss" else 80)
        raw = self._open_tunnel(host, port)
        if u.scheme == "wss":
            raw = ssl.create_default_context().wrap_socket(
                raw, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode()
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
               f"User-Agent: {self.user_agent}\r\n\r\n")
        raw.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = raw.recv(4096)
            if not chunk:
                raise WSHandshakeError("closed during handshake")
            resp += chunk
            if len(resp) > 65536:
                raise WSHandshakeError("oversized handshake response")
        head, _, rest = resp.partition(b"\r\n\r\n")
        status_line = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if "101" not in status_line:
            raise WSHandshakeError(f"handshake refused: {status_line}")
        expected = base64.b64encode(
            hashlib.sha1((key + _GUID).encode()).digest()).decode()
        if expected.lower() not in head.decode(errors="replace").lower():
            raise WSHandshakeError("bad Sec-WebSocket-Accept in handshake")
        self._sock = raw
        self._buf = rest
        self._frag = bytearray()
        self._frag_op = None
        return self

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._send_frame(0x8, b"")
            except Exception:  # noqa: BLE001 - closing is best-effort
                pass
            try:
                self._sock.close()
            except Exception:  # noqa: BLE001
                pass
            self._sock = None

    # ── framing ───────────────────────────────────────────────────

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self._sock is None:
            raise WSClosed("not connected")
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            hdr = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < 65536:
            hdr = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            hdr = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(hdr + mask + masked)

    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    def ping(self) -> None:
        self._send_frame(0x9, b"")

    def _read_frame(self) -> Optional[tuple[int, bytes]]:
        if len(self._buf) < 2:
            return None
        b0, b1 = self._buf[0], self._buf[1]
        fin = bool(b0 & 0x80)
        ln = b1 & 0x7F
        off = 2
        if ln == 126:
            if len(self._buf) < 4:
                return None
            ln = struct.unpack("!H", self._buf[2:4])[0]
            off = 4
        elif ln == 127:
            if len(self._buf) < 10:
                return None
            ln = struct.unpack("!Q", self._buf[2:10])[0]
            off = 10
        if b1 & 0x80:  # masked server frame (non-standard; tolerated)
            if len(self._buf) < off + 4:
                return None
            off += 4
        if len(self._buf) < off + ln:
            return None
        payload = self._buf[off:off + ln]
        if b1 & 0x80:
            mask = self._buf[off - 4:off]
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._buf = self._buf[off + ln:]
        return (b0 & 0x0F) | (0x80 if fin else 0), payload

    def recv(self, timeout: Optional[float] = None) -> str:
        if self._sock is None:
            raise WSClosed("not connected")
        self._sock.settimeout(self.timeout if timeout is None else timeout)
        while True:
            frame = self._read_frame()
            if frame is None:
                chunk = self._sock.recv(65536)
                if not chunk:
                    raise WSClosed("peer closed the connection")
                self._buf += chunk
                continue
            op, payload = frame
            fin = bool(op & 0x80)
            op &= 0x0F
            if op == 0x9:      # ping -> pong, keep waiting
                self._send_frame(0xA, payload)
                continue
            if op == 0xA:      # pong — heartbeat answered
                continue
            if op == 0x8:
                raise WSClosed("peer sent close")
            if op in (0x0, 0x1, 0x2):
                if self._frag_op is None:
                    self._frag_op = op
                self._frag += payload
                if fin:
                    msg = bytes(self._frag)
                    self._frag = bytearray()
                    self._frag_op = None
                    return msg.decode("utf-8", errors="replace")
                continue
            # unknown opcode: ignore the frame
