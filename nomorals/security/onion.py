"""Tor / .onion organ.

Detection (SOCKS 9050/9150, control 9051/9151, NM_TOR_SOCKS override),
real v3 address validation (version byte + SHA3-256 checksum), circuit
rotation over the control port (cookie or password auth), and fetch
through SOCKS5h so the hostname resolves at the exit.

No stubs: a missing Tor is named, never faked.
"""
from __future__ import annotations

import base64
import hashlib
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)
__all__ = ["validate_onion", "TorControl", "onion_fetch", "tor_status", "register"]

_ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"
_SOCKS_CANDIDATES = (("127.0.0.1", 9050), ("127.0.0.1", 9150))
_CONTROL_CANDIDATES = (("127.0.0.1", 9051), ("127.0.0.1", 9151))


def validate_onion(addr: str) -> bool:
    """True for a checksum-valid v3 .onion (56 chars, SHA3-256 verified)."""
    host = (addr or "").strip().lower().rstrip(".")
    if host.endswith(".onion"):
        host = host[:-6]
    if len(host) != 56 or any(c not in _ALPHABET for c in host):
        return False
    raw = base64.b32decode(host.upper())
    if len(raw) != 35 or raw[34] != 3:      # version byte must be 3
        return False
    expect = hashlib.sha3_256(b".onion checksum" + raw[:32]
                              + raw[34:35]).digest()[:2]
    return raw[32:34] == expect


def _socks_addr() -> Optional[tuple[str, int]]:
    env = os.environ.get("NM_TOR_SOCKS", "")
    if env:
        h, _, p = env.partition(":")
        return (h, int(p or 9050))
    for host, port in _SOCKS_CANDIDATES:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return (host, port)
        except OSError:
            continue
    return None


@dataclass
class TorControl:
    host: str
    port: int
    password: str = ""
    cookie: str = ""

    @classmethod
    def detect(cls) -> Optional["TorControl"]:
        for host, port in _CONTROL_CANDIDATES:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return cls(host, port,
                               password=os.environ.get("NM_TOR_CONTROL_PASSWORD", ""),
                               cookie=os.environ.get("NM_TOR_COOKIE", ""))
            except OSError:
                continue
        return None

    def _cmd(self, line: str) -> str:
        with socket.create_connection((self.host, self.port), timeout=5.0) as s:
            f = s.makefile("rwb", buffering=0)

            def send(text: str) -> str:
                f.write((text + "\r\n").encode())
                out = []
                while True:
                    row = f.readline().decode("utf-8", "replace").rstrip("\r\n")
                    out.append(row)
                    if len(row) > 3 and row[3] == " ":
                        return "\n".join(out)
            if self.cookie:
                send(f'AUTHENTICATE {self.cookie}')
            elif self.password:
                send(f'AUTHENTICATE "{self.password}"')
            else:
                try:
                    send("AUTHENTICATE")
                except OSError:
                    raise ToolError("Tor control needs a cookie or password "
                                    "(NM_TOR_COOKIE / NM_TOR_CONTROL_PASSWORD)")
            return send(line)

    def newnym(self) -> None:
        self._cmd("SIGNAL NEWNYM")

    def egress(self) -> str:
        return self._cmd("GETINFO circuit-status")


def tor_status() -> dict[str, Any]:
    socks = _socks_addr()
    control = TorControl.detect()
    out: dict[str, Any] = {
        "socks": f"{socks[0]}:{socks[1]}" if socks else None,
        "control": f"{control.host}:{control.port}" if control else None,
        "rotation": bool(control),
        "installed": bool(socks),
    }
    if not socks:
        out["note"] = ("no Tor SOCKS port answering — start the daemon "
                       "(tor / Tor Browser) or set NM_TOR_SOCKS")
    return out


def onion_fetch(url: str, *, timeout: float = 45.0, max_bytes: int = 2_000_000,
                rotate: bool = False) -> dict[str, Any]:
    """Fetch an http(s) URL through Tor. `.onion` is validated first."""
    if not url.lower().startswith(("http://", "https://")):
        raise ToolError("onion_fetch needs an http(s) URL")
    host = url.split("://", 1)[1].split("/", 1)[0].split(":")[0]
    if host.endswith(".onion") and not validate_onion(host):
        raise ToolError(f"invalid v3 onion address: {host}")
    socks = _socks_addr()
    if not socks:
        raise ToolError("Tor is not running (no SOCKS port)")
    if rotate:
        control = TorControl.detect()
        if control:
            control.newnym()
            time.sleep(5.0)
    import socks as pysocks
    import urllib.request
    sock = pysocks.socksocket()
    sock.set_proxy(pysocks.SOCKS5, socks[0], socks[1], rdns=True)
    sock.settimeout(timeout)
    opener = urllib.request.build_opener(
        urllib.request.SOCKSHandler if hasattr(urllib.request, "SOCKSHandler") else object)
    import sockshandler  # from PySocks
    opener = urllib.request.build_opener(sockshandler.SocksiPyHandler(
        pysocks.SOCKS5, socks[0], socks[1], True))
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Tor; security-test)"})
    started = time.monotonic()
    with opener.open(req, timeout=timeout) as resp:
        body = resp.read(max_bytes)
        return {"url": url, "status": resp.status,
                "content_type": resp.headers.get("Content-Type", ""),
                "bytes": len(body),
                "elapsed_s": round(time.monotonic() - started, 2),
                "text": body.decode("utf-8", "replace")}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register("tor_status",
                       description="Tor presence: SOCKS/control ports, rotation available.",
                       capability=Capability.NET_OUT)
    def _status() -> dict[str, Any]:
        return tor_status()

    @registry.register("onion_validate",
                       description="Validate a v3 .onion address (version byte + SHA3-256 checksum).",
                       capability=Capability.DB_READ,
                       parameters={"address": "str — v3 onion address"})
    def _validate(address: str) -> dict[str, Any]:
        return {"address": address, "valid": validate_onion(address)}

    @registry.register("onion_fetch",
                       description="Fetch a URL (or .onion) through Tor, optional circuit rotation.",
                       capability=Capability.NET_OUT,
                       parameters={"url": "str — http(s) URL",
                                   "rotate": "bool (optional) — NEWNYM first",
                                   "timeout": "float (optional, 45)"})
    def _fetch(url: str, *, rotate: str = "", timeout: str = "") -> dict[str, Any]:
        return onion_fetch(url,
                           rotate=(rotate or "").lower() in {"1", "true", "yes"},
                           timeout=float(timeout) if str(timeout).strip() else 45.0)
