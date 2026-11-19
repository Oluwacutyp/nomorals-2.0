"""Connections (Craft-Libs port of ai-core ``connection.rs``).

A connection is where a provider lives: its endpoint, its
locality (local or remote), how the key is sent, how it bills,
and which models it offers. :meth:`Connection.validate` refuses
the shapes that would lie about locality — a remote host labelled
local, a loopback host labelled remote, a local route with no
endpoint that still carries a key.
"""

from __future__ import annotations

from enum import Enum

from .endpoint import Endpoint
from .ids import ConnectionId, SecretRef
from .model import ModelInfo

__all__ = [
    "Auth",
    "Billing",
    "Connection",
    "ConnectionError",
    "Locality",
    "check_key_header",
]

_RESERVED_HEADERS = {
    "authorization", "host", "cookie", "content-length",
    "transfer-encoding", "connection", "proxy-authorization",
    "proxy-authenticate", "te", "trailer", "upgrade",
}


def check_key_header(name: str) -> None:
    """The header that carries a key: 1-64 plain characters,
    never a header the network owns."""
    if not name or len(name) > 64:
        raise ValueError("a key header is 1 to 64 characters")
    if not all(c.isascii() and (c.isalnum() or c in "-_") for c in name):
        raise ValueError(
            "a key header holds letters, digits, '-' and '_' only")
    lowered = name.lower()
    if lowered in _RESERVED_HEADERS or lowered.startswith("proxy-") \
            or lowered.startswith("sec-"):
        raise ValueError(f"a key cannot travel in {name!r}")


class ConnectionError(ValueError):
    """Why a connection was refused. ``kind`` is one of
    ``local_but_not_loopback``, ``remote_but_loopback``,
    ``local_without_endpoint_needs_no_auth``, ``duplicate_model``,
    ``empty_label``, ``endpoint``, ``key_header``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class Locality(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class Billing(Enum):
    FREE = "free"
    METERED = "metered"


class Auth:
    """How the key is sent. The key itself is never here — only
    its :class:`SecretRef`."""

    def __init__(self, kind: str, secret: str | None = None,
                 route: str | None = None) -> None:
        self.kind = kind
        self.secret = secret
        self.route = route

    @classmethod
    def none(cls) -> Auth:
        return cls("none")

    @classmethod
    def api_key(cls, secret: str) -> Auth:
        return cls("api_key", secret=SecretRef.new(secret).as_str())

    @classmethod
    def subscription(cls, route: str, secret: str) -> Auth:
        return cls("subscription",
                   secret=SecretRef.new(secret).as_str(), route=route)

    def to_dict(self) -> dict:
        data: dict = {"kind": self.kind}
        if self.secret is not None:
            data["secret"] = self.secret
        if self.route is not None:
            data["route"] = self.route
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Auth:
        kind = data.get("kind", "none")
        if kind == "api_key":
            return cls.api_key(str(data.get("secret", "")))
        if kind == "subscription":
            return cls.subscription(str(data.get("route", "")),
                                    str(data.get("secret", "")))
        return cls.none()

    def __eq__(self, other) -> bool:
        return (isinstance(other, Auth)
                and self.to_dict() == other.to_dict())


class Connection:
    """One provider route."""

    def __init__(self, id, label: str, provider: str,
                 endpoint: str | None = None,
                 locality: Locality = Locality.REMOTE,
                 auth: Auth | None = None,
                 billing: Billing = Billing.METERED,
                 models: list | None = None, enabled: bool = True,
                 key_header: str | None = None) -> None:
        self.id = id if isinstance(id, ConnectionId) \
            else ConnectionId.new(id)
        self.label = label
        self.provider = provider
        self.endpoint = endpoint
        self.locality = locality
        self.auth = auth or Auth.none()
        self.billing = billing
        self.models: list[ModelInfo] = list(models or [])
        self.enabled = enabled
        self.key_header = key_header

    def with_endpoint(self, endpoint: str) -> Connection:
        """A copy of this connection pointed at another endpoint
        (consent is bound to the endpoint, so this is how a move
        is expressed)."""
        copy = Connection(self.id, self.label, self.provider,
                          endpoint, self.locality, self.auth,
                          self.billing, list(self.models), self.enabled,
                          self.key_header)
        return copy

    def validate(self) -> None:
        """Refuse the shapes that would lie about locality."""
        if not self.label.strip():
            raise ConnectionError("empty_label", "a connection needs "
                                  "a label")
        seen = set()
        for model in self.models:
            if model.id in seen:
                raise ConnectionError(
                    "duplicate_model",
                    f"model {model.id.as_str()} is offered twice")
            seen.add(model.id)
        if self.key_header is not None:
            try:
                check_key_header(self.key_header)
            except ValueError as exc:
                raise ConnectionError("key_header", str(exc)) from exc
        if self.endpoint is not None:
            try:
                parsed = Endpoint.parse(self.endpoint)
            except ValueError as exc:
                raise ConnectionError("endpoint", str(exc)) from exc
            if self.locality is Locality.LOCAL and not parsed.is_loopback():
                raise ConnectionError(
                    "local_but_not_loopback",
                    "a local connection must point at a loopback host")
            if self.locality is Locality.REMOTE and parsed.is_loopback():
                raise ConnectionError(
                    "remote_but_loopback",
                    "a loopback host cannot be a remote connection")
        elif self.locality is Locality.LOCAL and self.auth.kind != "none":
            raise ConnectionError(
                "local_without_endpoint_needs_no_auth",
                "a local connection without an endpoint carries no key")

    def to_dict(self) -> dict:
        return {"id": self.id.as_str(), "label": self.label,
                "provider": self.provider,
                "endpoint": self.endpoint,
                "locality": self.locality.value,
                "auth": self.auth.to_dict(),
                "billing": self.billing.value,
                "models": [m.to_dict() for m in self.models],
                "enabled": self.enabled,
                "key_header": self.key_header}

    @classmethod
    def from_dict(cls, data: dict) -> Connection:
        return cls(ConnectionId.new(data["id"]), str(data["label"]),
                   str(data.get("provider", "http")),
                   data.get("endpoint"),
                   Locality(data.get("locality", "remote")),
                   Auth.from_dict(data.get("auth", {"kind": "none"})),
                   Billing(data.get("billing", "metered")),
                   [ModelInfo.from_dict(m)
                    for m in data.get("models", [])],
                   bool(data.get("enabled", True)),
                   data.get("key_header"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Connection)
                and self.to_dict() == other.to_dict())
