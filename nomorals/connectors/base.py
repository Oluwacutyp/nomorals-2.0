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
    "ConnectorAuthError",
    "ConnectorError",
    "ConnectorNetworkError",
    "ConnectorNotFoundError",
    "ConnectorRateLimitError",
    "ConnectorStatus",
    "ConnectorValidationError",
    "ConnectResult",
    "paginate",
    "request_with_retry",
]

_log = get_logger(__name__)

#: HTTP statuses worth one more try (plus 429 handled separately).
_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


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


class ConnectorAuthError(ConnectorError):
    """The credential is missing, invalid, expired, or revoked.

    Callers treat this as "re-authenticate", never "retry the request":
    it drives the reauth flow the way Home Assistant's
    ``ConfigEntryAuthFailed`` does. Never carries secret material.
    """


class ConnectorRateLimitError(ConnectorError):
    """The service throttled us. Back off for :attr:`retry_after` seconds.

    Mirrors ccxt's ``RateLimitExceeded`` / HA's
    ``UpdateFailed(retry_after=...)``: the error itself carries the wait,
    so callers don't have to parse headers.
    """

    def __init__(self, message: str, *, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = max(0.0, float(retry_after))


class ConnectorNetworkError(ConnectorError):
    """Transport-level failure: DNS, TLS, connect timeout, reset.

    Safe to retry with backoff; distinct from auth (don't re-auth) and
    rate limits (don't wait on Retry-After).
    """


class ConnectorNotFoundError(ConnectorError, NotFound):
    """The remote object (repo, chat, order, file) does not exist.

    Subclasses both so ``except NotFound`` and ``except ConnectorError``
    handlers keep working.
    """


class ConnectorValidationError(ConnectorError):
    """The request was rejected: bad arguments, bad state, 4xx semantics.

    Retrying the identical request will fail identically — fix the call.
    """


def request_with_retry(
    do_request: Callable[[], Any],
    *,
    op: str = "request",
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_after: Callable[[Any], float] | None = None,
) -> Any:
    """Run ``do_request`` with exponential backoff + jitter.

    Retries :class:`ConnectorRateLimitError` (honoring its ``retry_after``),
    :class:`ConnectorNetworkError`, and provider ``RequestError``-shaped
    failures carrying a retryable ``status`` attribute (429/5xx). Anything
    else — auth errors, validation errors, 4xx — fails fast and loud, the
    way ccxt refuses silent fallbacks.

    ``retry_after`` optionally extracts a server-advised wait from the raw
    response/exception when the raised error doesn't carry one.
    """
    import random
    import time as _time

    attempt = 0
    while True:
        attempt += 1
        try:
            return do_request()
        except ConnectorAuthError:
            raise
        except ConnectorValidationError:
            raise
        except ConnectorNotFoundError:
            raise
        except (ConnectorRateLimitError, ConnectorNetworkError) as exc:
            wait = getattr(exc, "retry_after", 0.0) or 0.0
            if retry_after is not None:
                try:
                    advised = float(retry_after(exc) or 0.0)
                    wait = max(wait, advised)
                except Exception:  # noqa: BLE001 - advisory only
                    pass
            if attempt >= max_attempts:
                raise ConnectorError(
                    f"{op} failed after {attempt} attempts (rate "
                    f"limited/unreachable): {exc}"
                ) from exc
            sleep_for = min(max(wait, base_delay * (2 ** (attempt - 1))), max_delay)
            sleep_for *= 0.8 + random.random() * 0.4  # ±20% jitter
            _log.debug("%s: attempt %d failed (%s); retry in %.1fs",
                       op, attempt, exc, sleep_for)
            _time.sleep(sleep_for)
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque
            status = getattr(exc, "status", getattr(exc, "status_code", 0))
            try:
                status = int(status or 0)
            except (TypeError, ValueError):
                status = 0
            if status in _RETRYABLE_STATUSES and attempt < max_attempts:
                sleep_for = min(base_delay * (2 ** (attempt - 1)), max_delay)
                sleep_for *= 0.8 + random.random() * 0.4
                _log.debug("%s: HTTP %s, retry %d in %.1fs",
                           op, status, attempt, sleep_for)
                _time.sleep(sleep_for)
                continue
            raise


def _parse_link_next(headers: Any) -> str:
    """Extract the ``rel="next"`` URL from a Link header, or ""."""
    get = getattr(headers, "get", None)
    link = get("link", "") if callable(get) else ""
    if not link and isinstance(headers, dict):
        link = headers.get("Link", "") or headers.get("link", "")
    for part in str(link).split(","):
        segments = [s.strip() for s in part.split(";")]
        if len(segments) >= 2 and 'rel="next"' in segments[1]:
            url = segments[0].strip()
            if url.startswith("<") and url.endswith(">"):
                return url[1:-1]
    return ""


def paginate(
    fetch_page: Callable[..., Any],
    *,
    style: str = "link",
    page_param: str = "page",
    per_page: int = 100,
    max_pages: int = 50,
    items_key: Callable[[Any], list[Any]] | str = "",
) -> Any:
    """Yield items across a paginated REST list endpoint.

    Two styles (Octokit/PyGithub teach both):

    * ``style="link"`` — ``fetch_page(url=...)`` returns ``(items, headers)``
      and pagination follows the ``Link: <...>; rel="next"`` header.
    * ``style="page"`` — ``fetch_page(page=n, per_page=m)`` returns a list
      (or a dict; then ``items_key`` names the list field); stops on a
      short/empty page.

    ``items_key`` may also be a callable mapping the raw page to a list.
    Bounded by ``max_pages`` so a misbehaving API can't loop forever.
    """
    if style == "link":
        pages = 0
        next_url: str | None = None
        first = True
        while pages < max_pages:
            if first:
                raw = fetch_page()
                first = False
            else:
                if not next_url:
                    return
                raw = fetch_page(url=next_url)
            pages += 1
            if isinstance(raw, tuple):
                items, headers = raw[0], (raw[1] if len(raw) > 1 else {})
            else:
                items, headers = raw, {}
            if callable(items_key):
                items = items_key(items)
            elif isinstance(items_key, str) and items_key and isinstance(items, dict):
                items = items.get(items_key, [])
            yield from items if isinstance(items, list) else []
            next_url = _parse_link_next(headers)
            if not next_url:
                return
        return
    # style == "page"
    page = 1
    pages = 0
    while pages < max_pages:
        raw = fetch_page(page=page, per_page=per_page)
        pages += 1
        if callable(items_key):
            items = items_key(raw)
        elif isinstance(items_key, str) and items_key and isinstance(raw, dict):
            items = raw.get(items_key, [])
        else:
            items = raw
        batch = items if isinstance(items, list) else []
        if not batch:
            return
        yield from batch
        if len(batch) < per_page:
            return
        page += 1


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

    #: n8n/HA-style grouping for fleet views: "payments", "trading",
    #: "social", "messaging", "cloud", "media", "travel", "data",
    #: "network", "productivity", ... Empty = uncategorized.
    CATEGORY: str = ""

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

    def capabilities(self) -> dict[str, Any]:
        """What this connector can do, in one standard shape.

        Several connectors historically defined their own ad-hoc
        ``capabilities()`` dicts; this is the unified contract they
        converge on::

            {
                "id": ..., "name": ..., "category": ...,
                "auth_methods": [...],
                "provisionable": [...],
                "features": ["send_message", "get_updates", ...],
                "webhooks": {"inbound": bool, "outbound": bool},
            }

        ``features`` defaults to the connector's public action methods
        (lifecycle methods excluded); subclasses override to curate.
        """
        lifecycle = {
            "connect", "disconnect", "status", "test_connection",
            "connect_url", "can_provision", "provision",
            "request_human", "resume_checkpoint", "capabilities",
        }
        features = sorted(
            name for name in dir(self)
            if not name.startswith("_")
            and name not in lifecycle
            and callable(getattr(self, name, None))
        )
        return {
            "id": self.id,
            "name": self.name,
            "category": self.CATEGORY,
            "auth_methods": [m.value for m in self.auth_methods],
            "provisionable": list(self.PROVISIONABLE),
            "features": features,
            "webhooks": {"inbound": False, "outbound": False},
        }

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
