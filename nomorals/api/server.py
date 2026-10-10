"""HTTP API — stdlib only, so the service runs anywhere Python does.

Deliberately not FastAPI. A dependency-free server means the whole system starts
on a phone with nothing installed, and the API surface here is small enough that
a hand-rolled router is clearer than a framework would be.

Threaded, because agent runs are long and a single-threaded server would block
health checks behind them.

Authentication and principals
-----------------------------
Every request resolves to a :class:`Principal` (a name plus a capability grant)
from its ``Authorization: Bearer <token>`` header:

* The configured owner token maps **explicitly** to full power
  (``CapabilitySet.all()``). This is operator control, not capability
  reduction: whoever holds the owner token runs the machine.
* Additional named principals can be registered with explicit, limited grants
  via the ``principals`` constructor argument.
* When a token *is* configured, a missing or unknown token is rejected
  with 401.
* When *no* token is configured (local dev / loopback use), requests resolve
  to the ``local`` default principal, which carries a deliberately bounded
  grant (:data:`DEFAULT_GRANT`) — never a silent ``all()``. Binding an
  unauthenticated port to anything but loopback is still the operator's
  mistake; the server logs a warning at startup.

The resolved principal is threaded through ``_handle`` → ``dispatch`` →
route handlers via a ``threading.local`` slot, so route signatures stay
backward compatible. ``/tools/call`` invokes tools with exactly the
principal's grant, and the tool-call ``actor`` is ``api:<principal-name>``
(the request body can no longer self-declare an actor).

Request limits
--------------
``Content-Length`` is validated *before* the body is read: missing or garbage
→ 400, larger than ``max_body_bytes`` → 413 without reading a byte. Bodies
without a usable length (chunked transfer encoding) are read through a hard
cap and minimally decoded. Parsed JSON is walked with guards on nesting depth,
per-string length, total string bytes, and container element counts.

Cross-cutting HTTP behavior
---------------------------
* **Request IDs** — every request gets an ``X-Request-ID`` (a client-sent one
  is honored, otherwise ``req_<ulid12>``). It is echoed on every response,
  carried in error bodies (RFC 9457 ``instance``), and included in log lines,
  so any error can be correlated with server logs.
* **Error envelope** — non-2xx responses are RFC 9457 problem-details
  (``application/problem+json``): ``type``/``title``/``status``/``detail``/
  ``instance`` plus ``code`` (stable machine-readable slug) and the legacy
  ``error`` message as extension members, so old clients keep working.
* **Rate limiting** — opt-in token-bucket limiter (the industry default:
  AWS API Gateway, Stripe) keyed by principal name. Exhaustion → 429 with
  ``Retry-After``; every response carries GitHub-style
  ``X-RateLimit-Limit/Remaining/Reset`` headers. The owner principal is
  exempt; the tokenless ``/live`` probe is exempt.
* **Readiness** — ``GET /live`` stays the tokenless static liveness probe;
  ``GET /ready`` (authenticated) verifies the database answers and returns
  503 when it cannot.
* **CORS** — off by default; ``cors_origins=[...]`` enables origin echoing
  plus an ``OPTIONS`` preflight handler.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..llm.brain import brain_for
from ..core.errors import CapabilityDenied, NoMoralsError, classify
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.policy import Capability, CapabilitySet
from ..version import __version__

__all__ = [
    "APIServer",
    "DEFAULT_GRANT",
    "DEFAULT_PRINCIPAL",
    "OWNER_PRINCIPAL",
    "Principal",
    "RateLimiter",
    "ServiceUnavailable",
    "serve",
]

_log = get_logger(__name__)

_ROUTE = re.compile(r"^/[a-z0-9_/-]+$")
#: Client-supplied request ids are echoed verbatim on the response, so
#: they must be header-safe: anything outside this token charset is
#: ignored and a fresh id is minted (blocks response-splitting).
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.~+/-]{1,128}$")


def _new_request_id(headers: Any) -> str:
    presented = headers.get("X-Request-ID", "").strip()
    if presented and _REQUEST_ID_RE.match(presented):
        return presented
    return f"req_{new_id()[:12]}"

#: Default cap on a single request body (1 MiB). Constructor-configurable;
#: when not given, ``settings.api.max_body_mb`` is honored if positive.
DEFAULT_MAX_BODY_BYTES = 1_048_576
#: Default cap on JSON nesting depth.
DEFAULT_MAX_JSON_DEPTH = 64
#: Default cap on the length of any single JSON string (characters).
DEFAULT_MAX_JSON_STRING_LEN = 1_048_576
#: Default cap on total JSON string payload (UTF-8 bytes, keys included).
DEFAULT_MAX_JSON_TOTAL_STRING_BYTES = 4_194_304
#: Default cap on the element count of any single JSON array or object.
DEFAULT_MAX_JSON_ELEMENTS = 10_000


@dataclass
class Principal:
    """Who a request is acting as: a name plus an explicit capability grant."""

    name: str
    grant: CapabilitySet = field(default_factory=CapabilitySet.none)


#: The configured owner token resolves to this — full power, by explicit
#: operator choice. The token holder *is* the operator.
OWNER_PRINCIPAL = Principal(name="owner", grant=CapabilitySet.all())

#: Safe default grant used when no token is configured. Read-mostly plus
#: chat and memory: enough for local tooling, never destructive.
DEFAULT_GRANT = CapabilitySet.of(
    Capability.FS_READ,
    Capability.NET_OUT,
    Capability.NET_BROWSER,
    Capability.NET_DOWNLOAD,
    Capability.MODEL_CALL,
    Capability.MEM_READ,
    Capability.MEM_WRITE,
    Capability.DB_READ,
    Capability.SOCIAL_READ,
)

#: The principal every request resolves to when the server runs without an
#: auth token. Explicit and bounded — never a silent ``CapabilitySet.all()``.
DEFAULT_PRINCIPAL = Principal(name="local", grant=DEFAULT_GRANT)


class ServiceUnavailable(NoMoralsError):
    """The server cannot serve traffic right now (→ 503). Raised by
    ``GET /ready`` when the database does not answer."""

    code = "service_unavailable"
    retryable = True


#: RFC 9457 ``title`` for each status we emit.
_PROBLEM_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Content Too Large",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
}

#: Machine-readable ``code`` slugs for each status we emit.
_PROBLEM_CODES = {
    400: "bad_request",
    401: "auth_required",
    403: "capability_denied",
    404: "not_found",
    405: "method_not_allowed",
    413: "body_too_large",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


def _problem_body(
    status: int,
    code: str,
    detail: str,
    request_id: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build an RFC 9457 problem-details envelope.

    ``type`` stays ``about:blank`` (no public docs URL exists for these
    codes — explicitly permitted by the RFC); the stable machine-readable
    slug rides in ``code``, and the legacy ``error`` message is kept as an
    extension member so old clients keep working.
    """
    body: dict[str, Any] = {
        "type": "about:blank",
        "title": _PROBLEM_TITLES.get(status, "Error"),
        "status": status,
        "detail": detail,
        "instance": request_id,
        "code": code,
        "error": detail,
    }
    body.update(extra)
    return body


class RateLimiter:
    """Token-bucket rate limiter, O(1) state per key (tokens + timestamp).

    Mined from the industry default for public APIs (AWS API Gateway,
    Stripe): a bucket holds up to ``capacity`` tokens and refills at a
    steady rate; each request spends one token. Short bursts up to the
    bucket capacity are allowed *by design* — the long-term rate stays
    exact. Refill is lazy, computed from elapsed monotonic time on each
    check, so idle keys cost nothing and there is no sweeper thread.

    ``clock`` is injectable so tests can drive time deterministically.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(
        self, key: str, *, per_minute: float
    ) -> tuple[bool, int, float, float]:
        """Try to spend one token for ``key``.

        Returns ``(allowed, remaining, retry_after_s, reset_epoch_s)``:
        ``retry_after_s`` is 0 when allowed, ``reset_epoch_s`` is the wall
        time when the bucket refills completely (GitHub-style
        ``X-RateLimit-Reset``).
        """
        rate = per_minute / 60.0  # tokens per second
        capacity = max(1, int(per_minute))
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = [float(capacity), now]
                self._buckets[key] = bucket
            tokens, updated = bucket
            tokens = min(float(capacity), tokens + (now - updated) * rate)
            if tokens >= 1.0:
                tokens -= 1.0
                allowed, retry_after = True, 0.0
            else:
                allowed = False
                retry_after = (1.0 - tokens) / rate if rate > 0 else 0.0
            bucket[0], bucket[1] = tokens, now
            remaining = int(tokens)
        reset_in = (capacity - tokens) / rate if rate > 0 else 0.0
        return allowed, remaining, retry_after, time.time() + reset_in

    def reset(self, key: str | None = None) -> None:
        """Drop bucket state (``None`` → all keys). Used by tests."""
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)


def _require_capability(principal: Principal, capability: str) -> None:
    """Raise :class:`CapabilityDenied` (→ 403) when ``principal`` lacks
    ``capability``. Route handlers call this so every new endpoint honors
    the resolved principal's grant."""
    if not principal.grant.grants(capability):
        raise CapabilityDenied(
            f"principal {principal.name!r} lacks capability {capability!r}",
            capability=capability,
        )


def _open_timeline(context: Any) -> Any:
    """Open an ``os.Timeline`` on the context's database (same helper the
    ``nm`` CLI uses). ``db_path`` may be ``None`` → in-memory timeline."""
    from ..os.timeline import Timeline

    db_path = getattr(getattr(context, "db", None), "path", None)
    return Timeline(db_path)


def _json_limit_error(
    body: Any,
    *,
    max_depth: int,
    max_string_len: int,
    max_total_string_bytes: int,
    max_elements: int,
) -> tuple[str, int] | None:
    """Walk parsed JSON and return ``(message, status)`` on the first limit
    breach, else ``None``. Iterative, so hostile nesting cannot blow the
    Python stack even if ``json.loads`` survived it."""
    total_string_bytes = 0
    stack: list[tuple[Any, int]] = [(body, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            return f"JSON exceeds max nesting depth of {max_depth}", 400
        if isinstance(node, str):
            if len(node) > max_string_len:
                return (
                    f"JSON string exceeds max length of {max_string_len} characters",
                    413,
                )
            total_string_bytes += len(node.encode("utf-8"))
            if total_string_bytes > max_total_string_bytes:
                return (
                    "JSON string payload exceeds "
                    f"{max_total_string_bytes} bytes total",
                    413,
                )
        elif isinstance(node, dict):
            if len(node) > max_elements:
                return (
                    f"JSON object exceeds max element count of {max_elements}",
                    413,
                )
            for key, value in node.items():
                if isinstance(key, str):
                    if len(key) > max_string_len:
                        return (
                            "JSON object key exceeds max length of "
                            f"{max_string_len} characters",
                            413,
                        )
                    total_string_bytes += len(key.encode("utf-8"))
                    if total_string_bytes > max_total_string_bytes:
                        return (
                            "JSON string payload exceeds "
                            f"{max_total_string_bytes} bytes total",
                            413,
                        )
                stack.append((value, depth + 1))
        elif isinstance(node, (list, tuple)):
            if len(node) > max_elements:
                return (
                    f"JSON array exceeds max element count of {max_elements}",
                    413,
                )
            for value in node:
                stack.append((value, depth + 1))
    return None


class APIServer:
    """Tiny router plus a threaded HTTP server."""

    def __init__(
        self,
        context: Any,
        *,
        token: str = "",
        principals: dict[str, Principal | tuple[str, CapabilitySet]] | None = None,
        max_body_bytes: int | None = None,
        default_grant: CapabilitySet | None = None,
        max_json_depth: int = DEFAULT_MAX_JSON_DEPTH,
        max_json_string_len: int = DEFAULT_MAX_JSON_STRING_LEN,
        max_json_total_string_bytes: int = DEFAULT_MAX_JSON_TOTAL_STRING_BYTES,
        max_json_elements: int = DEFAULT_MAX_JSON_ELEMENTS,
        rate_limit_per_minute: float = 0,
        rate_limits: dict[str, float] | None = None,
        cors_origins: list[str] | None = None,
    ) -> None:
        self.context = context
        self.token = token
        self.principals: dict[str, Principal] = {}
        for tok, entry in (principals or {}).items():
            if isinstance(entry, Principal):
                self.principals[tok] = entry
            else:
                name, grant = entry
                self.principals[tok] = Principal(name=name, grant=grant)
        if max_body_bytes is None:
            configured_mb = getattr(
                getattr(getattr(context, "settings", None), "api", None),
                "max_body_mb",
                0,
            )
            max_body_bytes = (
                int(configured_mb) * 1_048_576
                if configured_mb and int(configured_mb) > 0
                else DEFAULT_MAX_BODY_BYTES
            )
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self.max_body_bytes = max_body_bytes
        self.default_principal = Principal(
            name="local",
            grant=default_grant if default_grant is not None else DEFAULT_GRANT,
        )
        self.max_json_depth = max_json_depth
        self.max_json_string_len = max_json_string_len
        self.max_json_total_string_bytes = max_json_total_string_bytes
        self.max_json_elements = max_json_elements
        # Token-bucket rate limiting, keyed by principal name. 0 disables.
        # The owner principal is always exempt (operator control).
        if rate_limit_per_minute < 0:
            raise ValueError("rate_limit_per_minute must not be negative")
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limits = dict(rate_limits or {})
        self.rate_limiter = RateLimiter()
        # CORS: None (default) → no CORS headers at all; a list of origins
        # (or ["*"]) enables origin echoing + the OPTIONS preflight.
        self.cors_origins = list(cors_origins) if cors_origins else None
        self._tls = threading.local()
        self._routes: dict[tuple[str, str], Callable[..., Any]] = {}
        # (method, path) -> human description, surfaced by GET /docs.
        self._route_docs: dict[tuple[str, str], str] = {}
        # (method, path) keys that skip bearer auth entirely — tokenless
        # liveness only. Anything registered here must return a static
        # payload with no DB, version, or provider facts (see /live).
        self._public_routes: set[tuple[str, str]] = set()
        self._register_defaults()

    # ── principals ─────────────────────────────────────────────────────────

    def resolve_principal(self, authorization: str) -> Principal | None:
        """Map an ``Authorization`` header to a principal.

        Returns ``None`` when the request must be rejected (token configured
        but the presented token is missing/unknown) — the caller turns that
        into a 401.
        """
        presented = ""
        if authorization.startswith("Bearer "):
            presented = authorization[len("Bearer "):].strip()
        if self.token:
            if presented and hmac.compare_digest(presented, self.token):
                return OWNER_PRINCIPAL
            if presented:
                for tok, principal in self.principals.items():
                    if hmac.compare_digest(presented, tok):
                        return principal
            return None
        # No owner token configured: open local mode. A known extra principal
        # token still resolves to its grant; anything else gets the bounded
        # default — never full power by accident.
        if presented:
            for tok, principal in self.principals.items():
                if hmac.compare_digest(presented, tok):
                    return principal
        return self.default_principal

    def _current_principal(self) -> Principal:
        return getattr(self._tls, "principal", None) or self.default_principal

    def _limit_for(self, principal_name: str) -> float:
        """Requests/minute for ``principal_name``; 0 means unlimited.

        The owner principal is always exempt — whoever holds the owner
        token runs the machine and must not be throttled by it.
        """
        if principal_name == OWNER_PRINCIPAL.name:
            return 0
        return self.rate_limits.get(principal_name, self.rate_limit_per_minute)

    def dispatch(
        self,
        method: str,
        path: str,
        body: dict[str, Any],
        query: dict[str, str],
        principal: Principal | None = None,
    ) -> tuple[int, Any]:
        """Route a request. ``principal`` defaults to the bounded default
        principal, so direct callers get the safe grant, never ``all()``."""
        handler = self._routes.get((method.upper(), path))
        if handler is None:
            # A known path under a different method is 405, not 404 — the
            # resource exists, the verb does not.
            allowed = sorted({m for (m, p) in self._routes if p == path})
            if allowed:
                return 405, {
                    "error": f"method {method.upper()} not allowed for {path}",
                    "kind": "MethodNotAllowed",
                    "allow": allowed,
                }
            return 404, {"error": f"no route {method} {path}"}
        previous = getattr(self._tls, "principal", None)
        self._tls.principal = principal if principal is not None else self.default_principal
        try:
            try:
                return 200, handler(body, query)
            except CapabilityDenied as exc:
                return 403, {"error": str(exc) or "capability denied",
                             "kind": "CapabilityDenied"}
            except ServiceUnavailable as exc:
                return 503, {"error": str(exc) or "service unavailable",
                             "kind": "ServiceUnavailable"}
            except NoMoralsError as exc:
                outcome = classify(exc)
                return (429 if outcome.retryable else 400), {
                    "error": outcome.message,
                    "kind": type(exc).__name__,
                }
            except Exception as exc:  # noqa: BLE001 - never leak a traceback to a client
                _log.exception("api error on %s %s", method, path)
                return 500, {"error": type(exc).__name__}
        finally:
            self._tls.principal = previous

    def route(self, method: str, path: str, *, description: str = "",
              public: bool = False):
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            key = (method.upper(), path)
            self._routes[key] = fn
            # An explicit description wins; otherwise fall back to the
            # handler's name so /docs never shows a blank line.
            self._route_docs[key] = description or fn.__name__
            if public:
                self._public_routes.add(key)
            return fn

        return decorator

    # ── routes ───────────────────────────────────────────────────────────────

    def _register_defaults(self) -> None:
        context = self.context
        server = self

        @self.route("GET", "/live", public=True,
                   description="Tokenless liveness probe for dumb supervisors "
                               "(systemd, Docker HEALTHCHECK): static "
                               "{\"ok\": true}, no auth, no DB, no version, "
                               "no provider facts")
        def live(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            # Deliberately static: a supervisor only needs alive/dead.
            # Anything informative (providers, beacon state, last_error)
            # stays behind the bearer token on /health.
            return {"ok": True}

        @self.route("GET", "/ready",
                   description="Readiness probe (authenticated): 200 when "
                               "the database answers, 503 problem when it "
                               "cannot. Liveness stays on tokenless /live.")
        def ready(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            db = getattr(context, "db", None)
            if db is None:
                raise ServiceUnavailable("no database configured")
            try:
                db.query("SELECT 1")
            except Exception as exc:  # noqa: BLE001 - readiness must degrade
                raise ServiceUnavailable(
                    f"database unreachable: {exc}") from exc
            return {"ok": True}

        @self.route("GET", "/health", description="Health check: version, schema, process and bot runtime state")
        def health(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            db = getattr(context, "db", None)
            schema = 0
            if db is not None:
                # The migration table is the source of truth; PRAGMA user_version
                # is never written, so reading it reported 0 forever.
                from ..storage.schema import MigrationRunner

                schema = MigrationRunner(db).current_version()
            payload: dict[str, Any] = {
                "version": __version__,
                "ok": True,
                "schema_version": schema,
                # Local-only facts — no network, no model calls, safe for a
                # supervisor to poll every few seconds.
                "pid": os.getpid(),
                "uptime_s": round(
                    time.time() - getattr(context, "started_at", time.time()), 1
                ),
            }
            router = getattr(context, "router", None)
            try:
                payload["providers"] = list(router.providers()) if router else []
            except Exception:  # noqa: BLE001 - health must degrade, not 500
                payload["providers"] = []
            # Is the chat bot (partner runtime) alive in this home? The
            # runtime writes a status beacon every 10s and marks it stopped
            # on a clean shutdown; SIGTERM now takes the same path.
            try:
                from ..agents.beacon import ALIVE_WINDOW_S, read_status

                home = getattr(getattr(context, "settings", None), "home", "")
                state, age = read_status(home) if home else (None, None)
                runtime: dict[str, Any] = {"state": "not_running"}
                if state is not None:
                    if state.get("stopped"):
                        runtime = {"state": "stopped"}
                    elif age is not None and age <= ALIVE_WINDOW_S:
                        runtime = {
                            "state": "alive",
                            "uptime_s": state.get("uptime_s"),
                        }
                    else:
                        runtime = {"state": "stale"}
                    if age is not None:
                        runtime["beacon_age_s"] = round(age, 1)
                    if state.get("last_error"):
                        runtime["last_error"] = str(state["last_error"])[:200]
                payload["runtime"] = runtime
            except Exception:  # noqa: BLE001 - health must degrade, not 500
                payload["runtime"] = {"state": "unknown"}
            return payload

        @self.route("GET", "/models", description="List registered LLM models plus registry stats")
        def models(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..llm.registry import ModelRegistry

            registry = ModelRegistry(context.db)
            return {"stats": registry.stats(),
                    "models": [r.__dict__ for r in registry.list(limit=100)]}

        @self.route("GET", "/tools", description="List tool schemas")
        def tools(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            return {"tools": context.tools.register_builtins().schemas()}

        # Trigger webhooks (nomorals/triggers): POST /triggers/webhook.
        # The server (L7) imports triggers (L5) — downward, layering-safe.
        # Guarded so a triggers problem never breaks the API server.
        try:
            from ..triggers.webhook import register_trigger_routes

            register_trigger_routes(self, context)
        except Exception:  # noqa: BLE001 - webhook routes are additive
            _log.warning("trigger webhook routes not registered", exc_info=True)

        @self.route(
            "POST", "/tools/call",
            description="Call a tool as the principal, with exactly its capability grant",
        )
        def call_tool(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            principal = server._current_principal()
            name = str(body.get("name") or "")
            arguments = body.get("arguments") or {}
            if not isinstance(arguments, dict):
                return {"ok": False, "error": "arguments must be an object"}
            outcome = context.tools.call(
                name,
                # Actor comes from the resolved principal, never from the
                # request body — the body is untrusted.
                actor=f"api:{principal.name}",
                capabilities=principal.grant,
                confirmation=str(body.get("confirmation") or "") or None,
                **arguments,
            )
            if outcome.ok:
                return {"ok": True, "result": outcome.value}
            # Raise so dispatch maps the failure to a non-200 status:
            # CapabilityDenied -> 403, other tool errors -> 400/500.
            raise outcome.error if outcome.error is not None else NoMoralsError("tool call failed")

        @self.route("POST", "/chat", description="Chat with the model router")
        def chat(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..llm.base import Message, SamplingParams

            raw = body.get("messages") or [{"role": "user", "content": str(body.get("prompt") or "")}]
            messages = [
                Message(role=str(m.get("role") or "user"), content=str(m.get("content") or ""))
                for m in raw
                if isinstance(m, dict)
            ]
            params = SamplingParams(
                temperature=float(body.get("temperature", 0.7)),
                max_tokens=int(body.get("max_tokens", 1024)),
            )
            response = brain_for(context).chat(messages, params, task_kind="chat")
            return response.to_dict()

        @self.route("POST", "/memory/remember", description="Store a memory record")
        def remember(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            record_id = context.memory.remember(
                str(body.get("content") or ""),
                kind=str(body.get("kind") or "episode"),
                source=str(body.get("source") or "api"),
                importance=float(body.get("importance", 0.5)),
            )
            return {"id": record_id}

        @self.route("POST", "/memory/recall", description="Recall memory records")
        def recall(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            result = context.memory.recall(
                str(body.get("query") or ""),
                limit=int(body.get("limit", 8)),
                kind=str(body.get("kind") or "") or None,
            )
            return result.to_dict()

        @self.route("GET", "/memory/stats", description="Memory statistics snapshot")
        def memory_stats(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            return context.memory.stats_snapshot()

        @self.route("POST", "/agents/run", description="Run the master orchestrator toward a goal")
        def run_agent(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..agents.orchestrator import MasterOrchestrator

            goal = str(body.get("goal") or "")
            if not goal:
                return {"ok": False, "error": "goal is required"}
            orchestrator = MasterOrchestrator(context, max_steps=int(body.get("max_steps", 8)))
            result = orchestrator.run(goal, reflect=bool(body.get("reflect", False)))
            return result.to_dict()

        @self.route("POST", "/backup", description="Create a database backup, then rotate")
        def backup(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..storage.backup import BackupManager

            manager = BackupManager(context.db, context.settings.backup_dir)
            info = manager.create(label=str(body.get("label") or "api"))
            manager.rotate()
            return info.to_dict()

        @self.route("GET", "/events", description="Event bus snapshot (static; use /stream for live events)")
        def events(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            bus = getattr(context, "bus", None)
            return {"events": bus.snapshot() if bus is not None else []}

        # ── surface track: module endpoints ──────────────────────────────
        # Every handler resolves the principal's grant first (→ 403 when it
        # lacks the needed capability) and validates its input (→ 400 via
        # NoMoralsError). Organ errors that are plain Exceptions (SearchError,
        # WisdomError, TriggerError) are wrapped in NoMoralsError so the
        # dispatch mapping stays consistent.

        @self.route(
            "POST", "/wisdom/ask",
            description="Ask the WisdomKeeper corpus; every passage carries provenance",
        )
        def wisdom_ask(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            principal = server._current_principal()
            _require_capability(principal, Capability.MEM_READ)
            from ..wisdom.errors import WisdomError
            from ..wisdom.keeper import WisdomKeeper

            question = str(body.get("query") or "").strip()
            if not question:
                raise NoMoralsError("query is required")
            try:
                top = int(body.get("top", 5))
            except (TypeError, ValueError):
                raise NoMoralsError(
                    f"top must be an integer, got {body.get('top')!r}") from None
            top = max(1, min(top, 50))
            keeper = WisdomKeeper(context)
            try:
                answer = keeper.ask(
                    question, top=top, tradition=str(body.get("tradition") or "")
                )
            except WisdomError as exc:
                raise NoMoralsError(str(exc)) from exc
            return {"ok": True, **answer.to_dict()}

        @self.route(
            "POST", "/search",
            description="Federated search across memory, wisdom, books, docs, code, timeline",
        )
        def search(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            principal = server._current_principal()
            _require_capability(principal, Capability.NET_OUT)
            from ..search.errors import SearchError
            from ..search.federated import federated_search

            text = str(body.get("query") or "").strip()
            if not text:
                raise NoMoralsError("query is required")
            try:
                limit = int(body.get("limit", 10))
            except (TypeError, ValueError):
                raise NoMoralsError(
                    f"limit must be an integer, got {body.get('limit')!r}") from None
            sources = body.get("sources")
            if sources is not None and not isinstance(sources, list):
                raise NoMoralsError("sources must be a list of source names")
            types = body.get("types")
            if types is not None and not isinstance(types, list):
                raise NoMoralsError("types must be a list of result types")
            timeline = _open_timeline(context)
            try:
                response = federated_search(
                    text,
                    context=context,
                    sources=[str(s) for s in sources] if sources is not None else None,
                    limit=limit,
                    types=[str(t) for t in types] if types is not None else None,
                    since=body.get("since"),
                    before=body.get("before"),
                    timeline=timeline,
                )
            except SearchError as exc:
                raise NoMoralsError(str(exc)) from exc
            finally:
                timeline.close()
            return {"ok": True, **response.to_dict()}

        @self.route(
            "GET", "/connectors",
            description="List registered service connectors (metadata only, no secrets)",
        )
        def connectors(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            principal = server._current_principal()
            _require_capability(principal, Capability.DB_READ)
            from ..connectors.registry import list_connectors

            return {"connectors": list_connectors()}

        @self.route(
            "GET", "/sessions",
            description="List active OS sessions via the SessionBridge",
        )
        def sessions(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            principal = server._current_principal()
            _require_capability(principal, Capability.DB_READ)
            from ..os.session_bridge import SessionBridge

            bridge = SessionBridge(context.db)
            return {
                "sessions": [s.to_dict() for s in bridge.store.list_active()]
            }

        @self.route(
            "GET", "/triggers",
            description="List automation triggers (?enabled_only=1, ?source=NAME)",
        )
        def triggers_list(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            principal = server._current_principal()
            _require_capability(principal, Capability.DB_READ)
            from ..triggers.store import TriggerStore

            store = TriggerStore(context.db)
            enabled_only = (
                str(query.get("enabled_only", "")).strip().lower()
                in ("1", "true", "yes")
            )
            source = str(query.get("source") or "") or None
            return {
                "triggers": [
                    t.to_dict()
                    for t in store.list(enabled_only=enabled_only, source=source)
                ]
            }

        @self.route(
            "POST", "/triggers",
            description="Create an automation trigger (name, source, condition, action, action_params)",
        )
        def triggers_create(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            principal = server._current_principal()
            _require_capability(principal, Capability.DB_WRITE)
            from ..core.ids import new_short_id
            from ..triggers.models import Trigger, TriggerError, validate_definition
            from ..triggers.store import TriggerStore

            name = str(body.get("name") or "").strip()
            if not name:
                raise NoMoralsError("name is required")
            source = str(body.get("source") or "")
            action = str(body.get("action") or "")
            try:
                cooldown_s = float(body.get("cooldown_s", 0.0))
            except (TypeError, ValueError):
                raise NoMoralsError(
                    f"cooldown_s must be a number, "
                    f"got {body.get('cooldown_s')!r}") from None
            try:
                condition, params, cooldown_s = validate_definition(
                    source,
                    body.get("condition"),
                    action,
                    body.get("action_params"),
                    cooldown_s=cooldown_s,
                )
            except TriggerError as exc:
                raise NoMoralsError(str(exc)) from exc
            trigger = Trigger(
                id=new_short_id("trg_"),
                name=name,
                enabled=bool(body.get("enabled", True)),
                source=source,
                condition=condition,
                action=action,
                action_params=params,
                cooldown_s=cooldown_s,
            )
            TriggerStore(context.db).save(trigger)
            return {"ok": True, "trigger": trigger.to_dict()}

        @self.route(
            "GET", "/timeline",
            description="Query the OS event timeline (?topic=, ?since=, ?until=, ?limit=, ...)",
        )
        def timeline(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            principal = server._current_principal()
            _require_capability(principal, Capability.DB_READ)
            try:
                limit = int(query.get("limit", 100))
            except (TypeError, ValueError):
                raise NoMoralsError(
                    f"limit must be an integer, got {query.get('limit')!r}") from None
            limit = max(1, min(limit, 1000))
            tl = _open_timeline(context)
            try:
                events = tl.query(
                    session_id=str(query.get("session_id") or "") or None,
                    project_id=str(query.get("project_id") or "") or None,
                    mission_id=str(query.get("mission_id") or "") or None,
                    artifact_id=str(query.get("artifact_id") or "") or None,
                    topic=str(query.get("topic") or "") or None,
                    since=str(query.get("since") or "") or None,
                    until=str(query.get("until") or "") or None,
                    limit=limit,
                )
            except ValueError as exc:
                # _coerce_ts rejects unparseable since/until — a 400, not a 500.
                raise NoMoralsError(str(exc)) from exc
            finally:
                tl.close()
            return {"events": events}

        @self.route(
            "GET", "/docs",
            description="Self-documentation: every route on this API",
        )
        def docs(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            keys = sorted(set(server._routes) | set(server._route_docs))
            return {
                "version": __version__,
                "routes": [
                    {
                        "method": method,
                        "path": path,
                        "description": server._route_docs.get((method, path), ""),
                    }
                    for (method, path) in keys
                ],
            }

        # GET /stream is served directly by the request handler (SSE cannot
        # go through the JSON dispatch); its docs entry is registered here.
        self._route_docs[("GET", "/stream")] = (
            "Live Timeline events as Server-Sent Events (?since=EPOCH, ?topic=GLOB)"
        )


def _make_handler(server: APIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"NoMoralsCore/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # route through our logger
            request_id = getattr(self, "_request_id", "-")
            _log.debug("api %s [%s] - %s", self.address_string(), request_id,
                       fmt % args)

        def _cors_origin(self) -> str | None:
            """The origin to echo, or ``None`` when CORS is off / no match."""
            origins = server.cors_origins
            if not origins:
                return None
            if "*" in origins:
                return "*"
            origin = self.headers.get("Origin", "").strip()
            return origin if origin and origin in origins else None

        def _respond(self, status: int, payload: Any, *,
                     extra_headers: dict[str, str] | None = None,
                     problem: bool = False) -> None:
            raw = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            if problem:
                self.send_header("Content-Type",
                                 "application/problem+json; charset=utf-8")
            else:
                self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            request_id = getattr(self, "_request_id", "")
            if request_id:
                self.send_header("X-Request-ID", request_id)
            cors_origin = self._cors_origin()
            if cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Vary", "Origin")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(raw)

        def _send_error(self, status: int, detail: str, *,
                        code: str = "",
                        extra: dict[str, Any] | None = None,
                        extra_headers: dict[str, str] | None = None) -> None:
            """Send an RFC 9457 problem-details error response."""
            request_id = getattr(self, "_request_id", "")
            body = _problem_body(
                status,
                code or _PROBLEM_CODES.get(status, "error"),
                detail,
                request_id,
                **(extra or {}),
            )
            self._respond(status, body, problem=True,
                          extra_headers=extra_headers)

        def _wrap_dispatch_error(self, status: int,
                                 payload: Any) -> tuple[Any, bool]:
            """Turn a dispatch-level error payload into a problem-details
            envelope. Returns ``(payload, is_problem)``."""
            if not isinstance(payload, dict) or "type" in payload:
                return payload, False
            request_id = getattr(self, "_request_id", "")
            detail = str(payload.get("error")
                         or _PROBLEM_TITLES.get(status, "Error"))
            body = _problem_body(
                status,
                _PROBLEM_CODES.get(status, "error"),
                detail,
                request_id,
                **{k: v for k, v in payload.items() if k != "error"},
            )
            return body, True

        def _read_exactly(self, n: int) -> bytes:
            chunks: list[bytes] = []
            remaining = n
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def _read_capped(self, cap: int) -> bytes | None:
            """Read an unknown-length body, returning ``None`` if it exceeds
            ``cap`` bytes (callers translate that to 413)."""
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = self.rfile.read(min(65536, cap + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > cap:
                    return None
            return b"".join(chunks)

        def _read_body(self) -> tuple[bytes | None, str | None, int]:
            """Return ``(data, error, status)``. ``data`` is ``None`` on
            error; a valid-but-empty body yields ``b""``."""
            transfer_encoding = self.headers.get("Transfer-Encoding", "")
            if "chunked" in transfer_encoding.lower():
                raw = self._read_capped(server.max_body_bytes)
                if raw is None:
                    return None, (
                        f"body exceeds max of {server.max_body_bytes} bytes"
                    ), 413
                return self._decode_chunked(raw)
            if transfer_encoding and transfer_encoding.lower() != "identity":
                return None, (
                    f"unsupported Transfer-Encoding {transfer_encoding!r}; "
                    "send Content-Length instead"
                ), 400
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return None, "Content-Length header is required", 400
            try:
                length = int(raw_length.strip())
            except (ValueError, AttributeError):
                return None, f"invalid Content-Length {raw_length!r}", 400
            if length < 0:
                return None, "Content-Length must not be negative", 400
            if length > server.max_body_bytes:
                # Rejected before reading a single byte of the body.
                return None, (
                    f"Content-Length {length} exceeds max body size of "
                    f"{server.max_body_bytes} bytes"
                ), 413
            if length == 0:
                return b"", None, 200
            return self._read_exactly(length), None, 200

        def _decode_chunked(self, raw: bytes) -> tuple[bytes | None, str | None, int]:
            """Minimal chunked-framing decode over already-capped bytes."""
            out = bytearray()
            pos = 0
            while True:
                eol = raw.find(b"\r\n", pos)
                if eol < 0:
                    return None, "malformed chunked body", 400
                try:
                    size = int(raw[pos:eol].split(b";")[0].strip(), 16)
                except ValueError:
                    return None, "malformed chunked body", 400
                pos = eol + 2
                if size == 0:
                    break
                if pos + size > len(raw):
                    return None, "truncated chunked body", 400
                out += raw[pos:pos + size]
                pos += size
                if raw[pos:pos + 2] != b"\r\n":
                    return None, "malformed chunked body", 400
                pos += 2
                if len(out) > server.max_body_bytes:
                    return None, (
                        f"body exceeds max of {server.max_body_bytes} bytes"
                    ), 413
            return bytes(out), None, 200

        def _handle(self, method: str) -> None:
            # One request id per request: honor the client's, else mint one.
            # It is echoed on the response, stamped into error bodies
            # (RFC 9457 ``instance``), and carried in log lines.
            self._request_id = _new_request_id(self.headers)
            parsed = urlparse(self.path)
            if not _ROUTE.match(parsed.path):
                self._send_error(400, "malformed path")
                return
            if (method.upper(), parsed.path) in server._public_routes:
                # Tokenless liveness: skip bearer auth and serve the static
                # route under the bounded default principal (dispatch fills
                # it in when principal is None).
                principal = None
            else:
                principal = server.resolve_principal(
                    self.headers.get("Authorization", "")
                )
                if principal is None:
                    self._send_error(401, "missing or invalid bearer token")
                    return
            # Token-bucket rate limiting (opt-in), keyed by principal name.
            # The owner principal and the tokenless liveness probe are
            # exempt. 401s above never reach this point, so failed logins
            # do not burn tokens.
            rate_headers: dict[str, str] = {}
            if principal is not None and (
                    method.upper(), parsed.path) not in server._public_routes:
                limit = server._limit_for(principal.name)
                if limit > 0:
                    allowed, remaining, retry_after, reset_at = \
                        server.rate_limiter.check(
                            f"api:{principal.name}", per_minute=limit)
                    rate_headers = {
                        "X-RateLimit-Limit": str(int(limit)),
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(int(reset_at)),
                    }
                    if not allowed:
                        self._send_error(
                            429,
                            f"rate limit of {int(limit)} requests/minute "
                            f"exceeded for principal {principal.name!r}",
                            extra_headers={
                                **rate_headers,
                                "Retry-After": str(
                                    max(1, math.ceil(retry_after))),
                            },
                        )
                        return
            if method == "GET" and parsed.path == "/stream":
                # SSE cannot go through the JSON dispatch: the connection is
                # held open and framed as text/event-stream. Reuse the stream
                # server's emitter directly — the streaming API from the mesh
                # track, served on the main API port instead of a separate one.
                from ..stream.server import emit_sse

                try:
                    _require_capability(principal, Capability.DB_READ)
                except CapabilityDenied as exc:
                    self._send_error(403, str(exc) or "capability denied")
                    return
                emit_sse(
                    self,
                    lambda: _open_timeline(server.context),
                    parse_qs(parsed.query),
                )
                return
            if method == "POST" and parsed.path == "/acp" and \
                    "text/event-stream" in \
                    self.headers.get("Accept", "").lower():
                # ACP streaming: same precedent as /stream above — SSE cannot
                # go through the JSON dispatch. The ACP server streams
                # session/update notifications, then the final response.
                from .acp import serve_acp_sse
                acp_server = getattr(server, "_acp_server", None)
                if acp_server is None:
                    self._send_error(503, "ACP is not mounted on this API "
                                          "server (register_acp)")
                    return
                data, error, status = self._read_body()
                if error is not None:
                    self._send_error(status, error)
                    return
                serve_acp_sse(self, server, acp_server, data or b"")
                return
            body: dict[str, Any] = {}
            if method != "GET":
                data, error, status = self._read_body()
                if error is not None:
                    self._send_error(status, error)
                    return
                if data:
                    try:
                        parsed_body = json.loads(data.decode("utf-8"))
                    except UnicodeDecodeError:
                        self._send_error(400, "body is not valid UTF-8")
                        return
                    except json.JSONDecodeError:
                        self._send_error(400, "body is not valid JSON")
                        return
                    except RecursionError:
                        self._send_error(400, "body JSON is nested too deeply")
                        return
                    if not isinstance(parsed_body, dict):
                        self._send_error(400, "body must be a JSON object")
                        return
                    limit_error = _json_limit_error(
                        parsed_body,
                        max_depth=server.max_json_depth,
                        max_string_len=server.max_json_string_len,
                        max_total_string_bytes=server.max_json_total_string_bytes,
                        max_elements=server.max_json_elements,
                    )
                    if limit_error is not None:
                        message, limit_status = limit_error
                        self._send_error(limit_status, message)
                        return
                    body = parsed_body
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            status, payload = server.dispatch(
                method, parsed.path, body, query, principal=principal
            )
            if status >= 400:
                payload, is_problem = self._wrap_dispatch_error(status, payload)
                extra = dict(rate_headers)
                if status == 405 and isinstance(payload, dict):
                    allow = payload.get("allow")
                    if allow:
                        extra["Allow"] = ", ".join(str(m) for m in allow)
                self._respond(status, payload, problem=is_problem,
                              extra_headers=extra or None)
            else:
                self._respond(status, payload,
                              extra_headers=rate_headers or None)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def do_OPTIONS(self) -> None:  # noqa: N802
            # CORS preflight. Only meaningful when cors_origins is
            # configured; otherwise OPTIONS is just another unknown method.
            self._request_id = _new_request_id(self.headers)
            parsed = urlparse(self.path)
            if not _ROUTE.match(parsed.path):
                self._send_error(400, "malformed path")
                return
            origin = self._cors_origin()
            if origin is None:
                self._send_error(
                    404, f"no route OPTIONS {parsed.path}")
                return
            allowed = sorted({m for (m, p) in server._routes
                              if p == parsed.path} | {"OPTIONS"})
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods",
                             ", ".join(allowed))
            self.send_header("Access-Control-Allow-Headers",
                             "Authorization, Content-Type, X-Request-ID")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("X-Request-ID", self._request_id)
            self.end_headers()

    return Handler


def serve(
    context: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    token: str = "",
    background: bool = False,
    principals: dict[str, Principal | tuple[str, CapabilitySet]] | None = None,
    max_body_bytes: int | None = None,
    protocols: bool = True,
) -> int:
    """Start the API. Returns 0 on clean shutdown.

    ``protocols`` (default True) mounts the agent-protocol endpoints on the
    same port: ``POST /acp`` (Agent Client Protocol, incl. SSE streaming)
    and ``POST /mcp`` (Model Context Protocol). Pass ``protocols=False``
    for a bare HTTP API.
    """
    api = APIServer(
        context,
        token=token or context.settings.api.token,
        principals=principals,
        max_body_bytes=max_body_bytes,
    )
    if protocols:
        # The protocol servers share this API's Principal/capability model —
        # nothing weaker. A mount failure must never break the HTTP API.
        try:
            from .acp import ACPServer, register_acp

            register_acp(api, ACPServer(context))
        except Exception:  # noqa: BLE001 - protocol endpoints are additive
            _log.warning("acp routes not mounted", exc_info=True)
        try:
            from .mcp_server import MCPServer, register_mcp

            register_mcp(api, MCPServer(context))
        except Exception:  # noqa: BLE001 - protocol endpoints are additive
            _log.warning("mcp route not mounted", exc_info=True)
    httpd = ThreadingHTTPServer((host, port), _make_handler(api))
    httpd.daemon_threads = True
    _log.info("api listening on http://%s:%s", host, port)
    if not token and not context.settings.api.token:
        _log.warning(
            "api has no auth token configured; requests resolve to the "
            "bounded 'local' principal and this port must stay on loopback"
        )
    if background:
        thread = threading.Thread(target=httpd.serve_forever, name="api", daemon=True)
        thread.start()
        return 0
    # A supervisor's SIGTERM must drain the server the same way Ctrl-C
    # does. httpd.shutdown() may only be called from a thread OTHER than
    # the serve_forever thread, so the handler maps SIGTERM onto
    # KeyboardInterrupt and the finally block below does the teardown.
    from ..core.shutdown import install_sigterm_as_interrupt

    install_sigterm_as_interrupt()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover  # noqa: E103, E106 - deliberate shutdown hook
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0
