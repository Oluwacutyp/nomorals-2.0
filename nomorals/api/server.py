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
"""

from __future__ import annotations

import hmac
import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..core.errors import CapabilityDenied, NoMoralsError, classify
from ..core.logging_setup import get_logger
from ..core.policy import Capability, CapabilitySet
from ..version import __version__

__all__ = [
    "APIServer",
    "DEFAULT_GRANT",
    "DEFAULT_PRINCIPAL",
    "OWNER_PRINCIPAL",
    "Principal",
    "serve",
]

_log = get_logger(__name__)

_ROUTE = re.compile(r"^/[a-z0-9_/-]+$")

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
        self._tls = threading.local()
        self._routes: dict[tuple[str, str], Callable[..., Any]] = {}
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
            return 404, {"error": f"no route {method} {path}"}
        previous = getattr(self._tls, "principal", None)
        self._tls.principal = principal if principal is not None else self.default_principal
        try:
            try:
                return 200, handler(body, query)
            except CapabilityDenied as exc:
                return 403, {"error": str(exc) or "capability denied",
                             "kind": "CapabilityDenied"}
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

    def route(self, method: str, path: str):
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._routes[(method.upper(), path)] = fn
            return fn

        return decorator

    # ── routes ───────────────────────────────────────────────────────────────

    def _register_defaults(self) -> None:
        context = self.context
        server = self

        @self.route("GET", "/health")
        def health(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            db = getattr(context, "db", None)
            schema = 0
            if db is not None:
                # The migration table is the source of truth; PRAGMA user_version
                # is never written, so reading it reported 0 forever.
                from ..storage.schema import MigrationRunner

                schema = MigrationRunner(db).current_version()
            return {"version": __version__, "ok": True, "schema_version": schema}

        @self.route("GET", "/models")
        def models(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..llm.registry import ModelRegistry

            registry = ModelRegistry(context.db)
            return {"stats": registry.stats(),
                    "models": [r.__dict__ for r in registry.list(limit=100)]}

        @self.route("GET", "/tools")
        def tools(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            return {"tools": context.tools.register_builtins().schemas()}

        @self.route("POST", "/tools/call")
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

        @self.route("POST", "/chat")
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
            response = context.router.chat(messages, params)
            return response.to_dict()

        @self.route("POST", "/memory/remember")
        def remember(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            record_id = context.memory.remember(
                str(body.get("content") or ""),
                kind=str(body.get("kind") or "episode"),
                source=str(body.get("source") or "api"),
                importance=float(body.get("importance", 0.5)),
            )
            return {"id": record_id}

        @self.route("POST", "/memory/recall")
        def recall(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            result = context.memory.recall(
                str(body.get("query") or ""),
                limit=int(body.get("limit", 8)),
                kind=str(body.get("kind") or "") or None,
            )
            return result.to_dict()

        @self.route("GET", "/memory/stats")
        def memory_stats(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            return context.memory.stats_snapshot()

        @self.route("POST", "/agents/run")
        def run_agent(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..agents.orchestrator import MasterOrchestrator

            goal = str(body.get("goal") or "")
            if not goal:
                return {"ok": False, "error": "goal is required"}
            orchestrator = MasterOrchestrator(context, max_steps=int(body.get("max_steps", 8)))
            result = orchestrator.run(goal, reflect=bool(body.get("reflect", False)))
            return result.to_dict()

        @self.route("POST", "/backup")
        def backup(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            from ..storage.backup import BackupManager

            manager = BackupManager(context.db, context.settings.backup_dir)
            info = manager.create(label=str(body.get("label") or "api"))
            manager.rotate()
            return info.to_dict()

        @self.route("GET", "/events")
        def events(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
            bus = getattr(context, "bus", None)
            return {"events": bus.snapshot() if bus is not None else []}


def _make_handler(server: APIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"NoMoralsCore/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # route through our logger
            _log.debug("api %s - %s", self.address_string(), fmt % args)

        def _respond(self, status: int, payload: Any) -> None:
            raw = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

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
            parsed = urlparse(self.path)
            if not _ROUTE.match(parsed.path):
                self._respond(400, {"error": "malformed path"})
                return
            principal = server.resolve_principal(
                self.headers.get("Authorization", "")
            )
            if principal is None:
                self._respond(401, {"error": "missing or invalid bearer token"})
                return
            body: dict[str, Any] = {}
            if method != "GET":
                data, error, status = self._read_body()
                if error is not None:
                    self._respond(status, {"error": error})
                    return
                if data:
                    try:
                        parsed_body = json.loads(data.decode("utf-8"))
                    except UnicodeDecodeError:
                        self._respond(400, {"error": "body is not valid UTF-8"})
                        return
                    except json.JSONDecodeError:
                        self._respond(400, {"error": "body is not valid JSON"})
                        return
                    except RecursionError:
                        self._respond(400, {"error": "body JSON is nested too deeply"})
                        return
                    if not isinstance(parsed_body, dict):
                        self._respond(400, {"error": "body must be a JSON object"})
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
                        self._respond(limit_status, {"error": message})
                        return
                    body = parsed_body
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            status, payload = server.dispatch(
                method, parsed.path, body, query, principal=principal
            )
            self._respond(status, payload)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

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
) -> int:
    """Start the API. Returns 0 on clean shutdown."""
    api = APIServer(
        context,
        token=token or context.settings.api.token,
        principals=principals,
        max_body_bytes=max_body_bytes,
    )
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
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover  # noqa: E103, E106 - deliberate shutdown hook
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0
