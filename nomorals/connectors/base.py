"""Connector framework base: the contract every service adapter implements.

A Connector is the bridge between Devon and one external service. It owns:

* the auth flow for that service (PAT, API key, OAuth, ...),
* the credential's lifecycle in the encrypted vault (never in chat, never in
  logs, never on the command line),
* a live ``status()`` / ``test_connection()`` pair,
* optional *provisioning*: creating things the service's API legitimately
  allows (repos, webhooks, deploy keys, releases, buckets, ...) on the
  owner's request or standing permission. Provisioned secrets are handed to
  the owner and saved in the vault.

What a Connector is NOT: a way to bypass human signup or verification on a
third-party service (automated account creation, CAPTCHA/phone-verification
bypass). That violates those services' terms and gets the *owner's* accounts
banned — it actively harms the owner. For services that need a human signup,
the connector prepares everything possible and reduces the human step to
the minimum.
"""

from __future__ import annotations

import contextlib
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from ..accounts.vault import Credential, CredentialVault
from ..core.errors import NoMoralsError, NotFound
from ..core.events import Event, global_bus
from ..core.http import HttpClient
from ..core.logging_setup import get_logger

__all__ = [
    "AuthMethod",
    "Connector",
    "ConnectorError",
    "ConnectorStatus",
    "ConnectResult",
]

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break a connector flow (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


def _emitting_lifecycle(
    fn: Callable[..., Any], topic: str
) -> Callable[..., Any]:
    """Wrap a subclass ``connect``/``disconnect`` so it emits a bus event.

    Emission is fail-open: it runs after the wrapped method returns
    successfully and can never raise into the caller. A failed
    ``connect()`` (raised exception, or ``ConnectResult.ok`` False)
    emits ``connector.connect_failed`` instead of ``connector.connected``.
    """

    @functools.wraps(fn)
    def wrapper(self: "Connector", *args: Any, **kwargs: Any) -> Any:
        if topic == "connector.connected":
            try:
                result = fn(self, *args, **kwargs)
            except Exception:
                _emit("connector.connect_failed", {
                    "connector_id": self.id, "name": self.name})
                raise
            ok = bool(getattr(result, "ok", False))
            _emit(topic if ok else "connector.connect_failed", {
                "connector_id": self.id,
                "name": self.name,
                "ok": ok,
                "account": getattr(result, "account", "") or "",
            })
            return result
        fn(self, *args, **kwargs)
        _emit(topic, {"connector_id": self.id, "name": self.name})

    return wrapper


class ConnectorError(NoMoralsError):
    """Anything wrong in the connector layer: auth, API, provisioning."""


class AuthMethod(StrEnum):
    """How a connector authenticates to its service."""

    PAT = "pat"  # personal access token (GitHub, ...)
    API_KEY = "api_key"  # static key / key+secret pair
    OAUTH2 = "oauth2"  # OAuth 2.0 (authorization-code or device flow)
    BASIC = "basic"  # username + password (rare; prefer PAT)
    NONE = "none"  # no auth (local/keyless services)


@dataclass
class ConnectorStatus:
    """Live connection state, safe to show the owner (no secrets)."""

    connected: bool
    account: str | None = None
    scopes: list[str] = field(default_factory=list)
    last_checked: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "account": self.account,
            "scopes": list(self.scopes),
            "last_checked": self.last_checked,
            "detail": self.detail,
        }


@dataclass
class ConnectResult:
    """Outcome of a connect() flow (no secrets inside)."""

    ok: bool
    account: str = ""
    scopes: list[str] = field(default_factory=list)
    message: str = ""
    credential_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "account": self.account,
            "scopes": list(self.scopes),
            "message": self.message,
            "credential_id": self.credential_id,
        }


class Connector(ABC):
    """Base class for every service connector.

    Subclasses declare :attr:`id`, :attr:`name`, :attr:`description` and
    :attr:`auth_methods`, then implement the four lifecycle methods.
    Credentials live in the vault under the service
    ``"connector:<id>"`` — never in memory longer than needed, never logged.
    """

    id: str = ""
    name: str = ""
    description: str = ""
    auth_methods: tuple[AuthMethod, ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Wrap every concrete connect/disconnect with fail-open bus
        # telemetry. The wrap happens at class-creation time so all
        # subclasses emit without touching their implementations; only
        # methods defined on the subclass itself are wrapped (never the
        # base's abstract stubs, never an inherited wrapper twice).
        super().__init_subclass__(**kwargs)
        for method_name, topic in (
            ("connect", "connector.connected"),
            ("disconnect", "connector.disconnected"),
        ):
            fn = cls.__dict__.get(method_name)
            if fn is not None and not getattr(fn, "_bus_wrapped", False):
                wrapped = _emitting_lifecycle(fn, topic)
                wrapped._bus_wrapped = True  # type: ignore[attr-defined]
                setattr(cls, method_name, wrapped)

    #: kinds this connector can provision via the service API, e.g.
    #: ("repo", "webhook", "deploy_key"). Empty means no provisioning.
    PROVISIONABLE: tuple[str, ...] = ()

    def __init__(
        self,
        vault: CredentialVault,
        http: HttpClient | None = None,
    ) -> None:
        if not self.id:
            raise ConnectorError(
                f"{type(self).__name__} must declare a connector id"
            )
        self.vault = vault
        self.http = http or HttpClient()

    # ── lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    def connect(self, **kwargs: Any) -> ConnectResult:
        """Guide the owner through the auth flow and store the credential.

        Interactive when needed (prompts), non-interactive when the secret
        arrives via argument or environment. Fails fast with a clear message
        instead of hanging or silently succeeding.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Remove the stored credential. Idempotent."""

    @abstractmethod
    def status(self) -> ConnectorStatus:
        """Live status: stored credential + a real API check."""

    @abstractmethod
    def test_connection(self) -> bool:
        """True when the stored credential authenticates right now."""

    def connect_url(self) -> str | None:
        """OAuth authorize URL when the flow starts in a browser, else None."""
        return None

    # ── provisioning ───────────────────────────────────────────────

    def can_provision(self, kind: str) -> bool:
        """Whether this connector can provision ``kind`` via the service API."""
        return kind in self.PROVISIONABLE

    def provision(self, kind: str, **kwargs: Any) -> dict[str, Any]:
        """Provision ``kind`` on the owner's request or standing permission.

        Anything the service API legitimately allows (repos, webhooks,
        deploy keys, releases, buckets, ...). Returned secrets are handed to
        the owner by the caller and saved in the vault here.
        """
        raise ConnectorError(
            f"{self.name} cannot provision {kind!r} "
            f"(provisionable: {', '.join(self.PROVISIONABLE) or 'nothing'})"
        )

    # ── human-in-the-loop ────────────────────────────────────────

    def request_human(
        self,
        kind: Any,
        title: str,
        instructions: str,
        *,
        db: Any,
        context: Any = None,
        resume_state: dict[str, Any] | None = None,
        ttl_seconds: float = 86400.0,
    ) -> Any:
        """Pause for a human-only step (CAPTCHA, email verify, 2FA, ...).

        Persists a checkpoint, pings the owner through the owner-only
        delivery channel, then either waits (interactive TTY) or raises
        :class:`HumanCheckpointPending` so the flow pauses cleanly.
        """
        from .checkpoints import request_human_action

        return request_human_action(
            self.id,
            kind,
            title,
            instructions,
            db=db,
            context=context,
            resume_state=resume_state,
            ttl_seconds=ttl_seconds,
        )

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Continue a flow after its human checkpoint resolved."""
        raise ConnectorError(
            f"{self.name} does not support resuming checkpoints"
        )

    # ── credential helpers ─────────────────────────────────────────

    @property
    def _service(self) -> str:
        return f"connector:{self.id}"

    def _store_credential(
        self,
        username: str,
        secret: str,
        *,
        credential_type: str,
        scopes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Credential:
        meta = dict(metadata or {})
        if scopes:
            meta["scopes"] = list(scopes)
        cred = self.vault.store(
            service=self._service,
            username=username,
            password=secret,
            credential_type=credential_type,
            tags=["connector", self.id],
            metadata=meta,
        )
        _log.info("connector %s: credential stored for %s", self.id, username)
        return cred

    def _load_credential(self) -> Credential | None:
        """The stored credential (decrypted), or None when not connected."""
        existing = self.vault.list_all(service=self._service)
        if not existing:
            return None
        try:
            return self.vault.get(self._service, existing[0].username)
        except NotFound:
            return None

    def _clear_credential(self) -> None:
        for cred in self.vault.list_all(service=self._service):
            with contextlib.suppress(NotFound):
                self.vault.delete(self._service, cred.username)
        _log.info("connector %s: credential removed", self.id)
