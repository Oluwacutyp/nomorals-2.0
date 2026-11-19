"""Endpoints (Craft-Libs port of ai-core ``endpoint.rs``).

The question this answers is "does this connection send data off
this machine?". It is a security decision, so the rules are
strict: only ``http`` and ``https``; no user-info in the authority
(the ``http://127.0.0.1@evil.example/`` trick); plain ``http``
only for loopback hosts; loopback means ``localhost``,
``127.0.0.0/8`` or ``::1``. No DNS, no I/O: a name that is not
literally ``localhost`` is treated as remote.
"""

from __future__ import annotations

import ipaddress

from .ids import IdError

__all__ = ["Endpoint", "MAX_ENDPOINT_LEN"]

MAX_ENDPOINT_LEN = 2048


class Endpoint:
    """A parsed endpoint: scheme, host, optional port."""

    def __init__(self, https: bool, host: str,
                 port: int | None = None) -> None:
        self.https = https
        self.host = host
        self.port = port

    @classmethod
    def parse(cls, text: str) -> Endpoint:
        """Parse ``http(s)://host[:port][/path]``. Rejects anything
        surprising with :class:`IdError`."""
        if not text:
            raise IdError("value is empty")
        if len(text.encode("utf-8")) > MAX_ENDPOINT_LEN:
            raise IdError(
                f"value is {len(text.encode('utf-8'))} bytes, "
                f"the limit is {MAX_ENDPOINT_LEN}")
        for char in text:
            if char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F \
                    or char == "\\":
                raise IdError(f"character {char!r} is not allowed")
        if text.startswith("https://"):
            https, rest = True, text[len("https://"):]
        elif text.startswith("http://"):
            https, rest = False, text[len("http://"):]
        else:
            raise IdError("malformed value: endpoint must start with "
                          "http:// or https://")
        authority = rest
        for sep in ("/", "?", "#"):
            cut = authority.find(sep)
            if cut >= 0:
                authority = authority[:cut]
        if "@" in authority:
            raise IdError("malformed value: user-info in an endpoint "
                          "is not allowed")
        host, port = _split_host_port(authority)
        if not host:
            raise IdError("malformed value: endpoint has no host")
        endpoint = cls(https, host.lower(), port)
        if not endpoint.https and not endpoint.is_loopback():
            raise IdError("malformed value: plain http is only allowed "
                          "for loopback hosts")
        return endpoint

    def is_loopback(self) -> bool:
        """True for ``localhost``, ``127.0.0.0/8`` and ``::1``.
        Nothing else, and no name resolution."""
        if self.host == "localhost":
            return True
        try:
            addr = ipaddress.ip_address(self.host)
        except ValueError:
            return False
        return addr.is_loopback

    def fingerprint(self) -> str:
        """``scheme://host[:port]``. Consent is bound to this, so
        changing the host invalidates it."""
        scheme = "https" if self.https else "http"
        host = f"[{self.host}]" if ":" in self.host else self.host
        if self.port is not None:
            return f"{scheme}://{host}:{self.port}"
        return f"{scheme}://{host}"

    def __eq__(self, other) -> bool:
        return (isinstance(other, Endpoint)
                and self.https == other.https
                and self.host == other.host
                and self.port == other.port)

    def __repr__(self) -> str:
        return f"Endpoint({self.fingerprint()!r})"


def _split_host_port(authority: str):
    if authority.startswith("["):
        close = authority.find("]")
        if close < 0:
            raise IdError("malformed value: unclosed [ in host")
        host = authority[1:close]
        after = authority[close + 1:]
        if after == "":
            return host, None
        if after.startswith(":"):
            return host, _parse_port(after[1:])
        raise IdError("malformed value: junk after ] in host")
    if ":" in authority:
        host, _, port_text = authority.rpartition(":")
        return host, _parse_port(port_text)
    return authority, None


def _parse_port(text: str) -> int:
    if not text.isascii() or not text.isdigit():
        raise IdError("malformed value: port is not a number "
                      "from 0 to 65535")
    value = int(text)
    if value > 65535:
        raise IdError("malformed value: port is not a number "
                      "from 0 to 65535")
    return value
