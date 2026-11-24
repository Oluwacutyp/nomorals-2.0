"""MCP CLIENT — Devon calling external MCP servers.

The mirror image of ``nomorals/api/mcp_server.py``: that module SERVES
MCP (Devon as infrastructure); this one CONSUMES it (Devon as a client
of other servers). Same protocol version (2025-11-25), same JSON-RPC
2.0 shape, same honesty rules — a method the other side does not
implement is a -32601, never fake success; a server that dies, hangs,
or speaks malformed JSON is reported, never papered over.

Transports (stdlib only, no new dependency):
  * stdio — spawn the server as a subprocess, newline-delimited
    JSON-RPC over stdin/stdout (the server may also use
    Content-Length framing; the reader accepts both).
  * Streamable HTTP — POST JSON-RPC to the server's endpoint, Accept
    ``application/json, text/event-stream``; a JSON body or an SSE
    stream comes back. This is the current-spec HTTP transport
    (2025-03-26+); the legacy SSE GET /sse + POST endpoint dance is
    NOT implemented — no named server needs it, and the spec
    deprecated it.

What the client does NOT offer: sampling/*, elicitation/*, roots/* —
server→client requests. A server that sends one gets it logged and
ignored; none of the named servers requires them.

Known servers (``DEFAULT_MCP_SERVERS``) are shipped DEFAULTS only.
The effective registry is built from config (``mcp.servers`` in the
TOML — see ``nomorals/core/config.py`` MCPSettings): a config entry
with a known name overrides that server's default, a new name adds a
server. Nothing connects by default — the master switch
(``mcp.enabled``) AND each server's ``enabled`` flag must be on.
Tokens live in the environment/vault at spawn time (the ``env`` list
names variables to pass through); no token is ever in config or code.

Bounds: per-call timeout (default 30s, per-server overridable),
per-line payload cap (default 1MB — a line past it is a transport
error), per-manager connection budget (``mcp.max_servers``, default
8). A slow server fails its own call; it never stalls the brain.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "MCPClientError",
    "MCPTimeoutError",
    "MCPServerConfig",
    "DEFAULT_MCP_SERVERS",
    "MCP_PROTOCOL_VERSION",
    "MCPClient",
    "MCPClientManager",
    "effective_mcp_servers",
    "control_mcp",
    "register",
]

MCP_PROTOCOL_VERSION = "2025-11-25"

#: Protocol versions we can speak, newest first. initialize negotiates
#: the highest version both sides support (never above our own).
_MCP_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

CLIENT_NAME = "devon"
CLIENT_VERSION = "2.0"

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_PAYLOAD_BYTES = 1_000_000
DEFAULT_MAX_SERVERS = 8

#: JSON-RPC error codes (mirrors the server module).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class MCPClientError(Exception):
    """The server, the transport, or the protocol failed. Never silent."""

    def __init__(self, message: str, code: int = INTERNAL_ERROR) -> None:
        super().__init__(message)
        self.code = code


class MCPTimeoutError(MCPClientError):
    """A single MCP call ran past its timeout. The server is not dead —
    only that call is; the connection stays usable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=INTERNAL_ERROR)


# ── server configuration ────────────────────────────────────────────────

@dataclass
class MCPServerConfig:
    """One MCP server as the client sees it.

    ``env`` names environment variables passed through to the spawned
    server (values come from the process environment / vault at spawn
    time — the config only names them). ``notes`` is human text shown
    by /mcp status (e.g. "needs the Blender add-on running").
    ``transport`` is "stdio" or "http" on a resolved config; the empty
    string means unset and only ever appears on a config override
    (from_dict / MCPServerEntry), where merge() inherits the default.
    """

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    transport: str = "stdio"  # stdio | http
    url: str = ""
    enabled: bool = False
    timeout_s: float = 0.0  # 0 = the manager default
    notes: str = ""
    #: Literal environment values for the spawned server (ACP-declared
    #: servers: the client hands over the values, not just the names).
    #: Constructor-only — ``from_dict`` still rejects it, so no TOML
    #: config can ever carry a literal secret (see the module docstring:
    #: tokens live in the environment/vault, never in config or code).
    env_values: dict[str, str] = field(default_factory=dict)
    #: Literal HTTP headers for Streamable-HTTP servers (ACP-declared
    #: servers with auth). Constructor-only, same reason as env_values.
    headers: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """The config is usable as a server (called on the EFFECTIVE
        registry — a config override is a partial and is never
        validated on its own)."""
        if self.transport not in ("stdio", "http"):
            raise MCPClientError(
                f"mcp server {self.name!r}: transport must be 'stdio' "
                f"or 'http', got {self.transport!r}")
        if self.transport == "stdio" and not self.command:
            raise MCPClientError(
                f"mcp server {self.name!r}: stdio needs a command")
        if self.transport == "http" and not self.url:
            raise MCPClientError(
                f"mcp server {self.name!r}: http needs a url")

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> MCPServerConfig:
        unknown = set(raw) - {
            "command", "args", "env", "transport", "url", "enabled",
            "timeout_s", "notes"}
        if unknown:
            raise MCPClientError(
                f"mcp server {name!r}: unknown key(s) {sorted(unknown)}")
        return cls(
            name=name,
            command=str(raw.get("command") or ""),
            args=[str(a) for a in (raw.get("args") or [])],
            env=[str(e) for e in (raw.get("env") or [])],
            # Empty transport = not set (an override inherits the
            # default's transport in merge(); a brand-new server gets
            # stdio in effective_mcp_servers). Never default to
            # "stdio" here: that would overwrite a shipped HTTP
            # server's transport on merge.
            transport=str(raw.get("transport") or ""),
            url=str(raw.get("url") or ""),
            enabled=bool(raw.get("enabled", False)),
            timeout_s=float(raw.get("timeout_s") or 0.0),
            notes=str(raw.get("notes") or ""),
        )

    def merge(self, override: MCPServerConfig) -> MCPServerConfig:
        """A config entry overriding this default, field by field.

        Only set fields win: a non-empty command/args/env/url/notes,
        an explicit transport, an explicit enabled, a non-zero
        timeout. An override that sets nothing changes nothing — in
        particular an override that leaves transport empty inherits
        this default's transport (an ``{enabled: true}`` override must
        never flip a shipped HTTP server to stdio).
        """
        return MCPServerConfig(
            name=self.name,
            command=override.command or self.command,
            args=list(override.args) if override.args else list(self.args),
            env=list(override.env) if override.env else list(self.env),
            transport=override.transport or self.transport,
            url=override.url or self.url,
            enabled=override.enabled,
            timeout_s=override.timeout_s or self.timeout_s,
            notes=override.notes or self.notes,
        )


#: Shipped defaults for the servers Devon knows. DEFAULTS ONLY — the
#: effective registry comes from config (see effective_mcp_servers).
#: All disabled: nothing connects until the operator enables it.
#: Commands verified against the projects' own docs (2026-11-14):
#:   context7  — npx @upstash/context7-mcp (stdio)
#:   github    — npx @modelcontextprotocol/server-github (stdio;
#:               needs GITHUB_PERSONAL_ACCESS_TOKEN in the environment)
#:   playwright — npx @playwright/mcp@latest (stdio)
#:   serena    — uvx --from git+https://github.com/oraios/serena
#:               serena start-mcp-server (stdio)
#:   graphify  — hosted Streamable HTTP https://api.graphify.com/mcp
#:               (OAuth browser sign-in on first connect; no API key.
#:               Local alternative: `uv tool install "graphifyy[mcp]"`
#:               then `python -m graphify.serve <graph.json>` over stdio.)
#:   blender   — uvx mcp-for-blender (stdio; the MCP for Blender
#:               add-on must be installed and its server started in
#:               Blender — `uvx mcp-for-blender install-addon`).
#:   since-cutoff — uvx since-cutoff@latest mcp (stdio; read-only,
#:               no key: what changed in pinned Python APIs since the
#:               model's training cutoff).
#:   bestax    — npx -y bestax-mcp (stdio; Bulma React component
#:               props, examples and skills).
#:   upload-post — npx -y @upload-post/mcp (stdio; needs
#:               UPLOAD_POST_API_KEY in the environment; social
#:               publishing across TikTok, YouTube, Instagram,
#:               LinkedIn, Facebook, Threads, Pinterest, Reddit,
#:               Bluesky, X, WordPress and more).
#:   humanpages — npx -y humanpages (stdio; hire real humans for
#:               tasks agents cannot do — registers as an agent on
#:               first use; a hire costs money only when the owner
#:               approves one).
#:   sato-hub  — hosted Streamable HTTP https://satohub.ai/api/mcp
#:               (keyless; scored index of onchain-agent tooling +
#:               preflight checks, 41 of 47 tools read-only).
#:   ceki      — hosted Streamable HTTP https://api.ceki.me/mcp/agent
#:               (rents real residential Chrome browsers, per-minute
#:               crypto billing; public tools need no key, authed
#:               tools need an X-Agent-Key header — config cannot yet
#:               pass HTTP headers, so only the public tools are
#:               reachable from a config-enabled server).
DEFAULT_MCP_SERVERS: dict[str, MCPServerConfig] = {
    "context7": MCPServerConfig(
        name="context7",
        command="npx",
        args=["-y", "@upstash/context7-mcp"],
        notes="Version-specific library docs.",
    ),
    "github": MCPServerConfig(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env=["GITHUB_PERSONAL_ACCESS_TOKEN"],
        notes="Needs GITHUB_PERSONAL_ACCESS_TOKEN in the environment.",
    ),
    "playwright": MCPServerConfig(
        name="playwright",
        command="npx",
        args=["-y", "@playwright/mcp@latest"],
        notes="Accessibility-tree browser control.",
    ),
    "serena": MCPServerConfig(
        name="serena",
        command="uvx",
        args=["--from", "git+https://github.com/oraios/serena",
              "serena", "start-mcp-server"],
        notes="Semantic code retrieval + editing.",
    ),
    "graphify": MCPServerConfig(
        name="graphify",
        transport="http",
        url="https://api.graphify.com/mcp",
        notes="Hosted code graph (Streamable HTTP). OAuth browser "
              "sign-in on first connect; no API key.",
    ),
    "blender": MCPServerConfig(
        name="blender",
        command="uvx",
        args=["mcp-for-blender"],
        notes="Needs the MCP for Blender add-on installed and its "
              "server started in Blender.",
    ),
    "since-cutoff": MCPServerConfig(
        name="since-cutoff",
        command="uvx",
        args=["since-cutoff@latest", "mcp"],
        notes="What changed in pinned Python APIs since the model's "
              "training cutoff. Read-only, no key, no model calls.",
    ),
    "bestax": MCPServerConfig(
        name="bestax",
        command="npx",
        args=["-y", "bestax-mcp"],
        notes="Bulma React component props, examples and skills.",
    ),
    "upload-post": MCPServerConfig(
        name="upload-post",
        command="npx",
        args=["-y", "@upload-post/mcp"],
        env=["UPLOAD_POST_API_KEY"],
        notes="Needs UPLOAD_POST_API_KEY in the environment. Social "
              "publishing across TikTok, YouTube, Instagram, "
              "LinkedIn, Facebook, Threads, Pinterest, Reddit, "
              "Bluesky, X, WordPress and more.",
    ),
    "humanpages": MCPServerConfig(
        name="humanpages",
        command="npx",
        args=["-y", "humanpages"],
        notes="Hire real humans for tasks agents cannot do. Registers "
              "as an agent on first use; a hire costs money only "
              "when the owner approves one.",
    ),
    "sato-hub": MCPServerConfig(
        name="sato-hub",
        transport="http",
        url="https://satohub.ai/api/mcp",
        notes="Keyless scored index of onchain-agent tooling + "
              "preflight checks. 41 of 47 tools read-only.",
    ),
    "ceki": MCPServerConfig(
        name="ceki",
        transport="http",
        url="https://api.ceki.me/mcp/agent",
        notes="Rents real residential Chrome browsers (per-minute "
              "crypto billing). Public tools need no key; authed "
              "tools need an X-Agent-Key header, which config cannot "
              "pass yet — authed use needs an ACP-declared server.",
    ),
}


def _field(obj: Any, name: str, default: Any) -> Any:
    """Read a possibly-absent attribute from a settings/config object.

    Config sections may be old, duck-typed, or missing entirely (the
    manager is built from None in tests); vars() reads the instance
    dict without getattr-with-constant (B009).
    """
    if obj is None:
        return default
    try:
        return vars(obj).get(name, default)
    except TypeError:
        return default


def effective_mcp_servers(settings: Any) -> dict[str, MCPServerConfig]:
    """The effective registry: shipped defaults overlaid with config.

    ``settings`` is the app Settings (or None → defaults only, all
    disabled). A config entry with a known name overrides that
    server's default field by field; a new name adds a server.
    """
    out = {name: cfg for name, cfg in DEFAULT_MCP_SERVERS.items()}
    entries = _field(_field(settings, "mcp", None), "servers", None) or {}
    for name, entry in entries.items():
        # Config entries arrive as MCPServerEntry (config.py coerces
        # them) or as plain dicts (tests / direct construction).
        if isinstance(entry, dict):
            override = MCPServerConfig.from_dict(name, entry)
        else:
            try:
                fields = vars(entry)
            except TypeError as exc:
                raise MCPClientError(
                    f"mcp server {name!r}: entry must be a table, got "
                    f"{type(entry).__name__}") from exc
            if not isinstance(fields, dict):
                raise MCPClientError(
                    f"mcp server {name!r}: entry must be a table, got "
                    f"{type(entry).__name__}")
            override = MCPServerConfig(
                name=name,
                command=str(fields.get("command") or ""),
                args=[str(a) for a in (fields.get("args") or [])],
                env=[str(e) for e in (fields.get("env") or [])],
                # Empty = not set (MCPServerEntry.transport defaults to
                # ""); merge() inherits the default's transport.
                transport=str(fields.get("transport") or ""),
                url=str(fields.get("url") or ""),
                enabled=bool(fields.get("enabled", False)),
                timeout_s=float(fields.get("timeout_s") or 0.0),
            )
        if name in out:
            out[name] = out[name].merge(override)
        else:
            # A brand-new server with no transport defaults to stdio
            # (the old from_dict default, preserved for new names only).
            if not override.transport:
                override.transport = "stdio"
            out[name] = override
    for cfg in out.values():
        cfg.validate()
    return out


def _mcp_enabled(settings: Any) -> bool:
    return bool(_field(_field(settings, "mcp", None), "enabled", False))


def _mcp_default_timeout(settings: Any) -> float:
    try:
        return float(
            _field(_field(settings, "mcp", None), "default_timeout_s", 0.0)
            or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mcp_max_payload(settings: Any) -> int:
    try:
        return int(
            _field(_field(settings, "mcp", None), "max_payload_bytes", 0)
            or 0)
    except (TypeError, ValueError):
        return 0


def _mcp_max_servers(settings: Any) -> int:
    try:
        return int(
            _field(_field(settings, "mcp", None), "max_servers", 0) or 0)
    except (TypeError, ValueError):
        return 0


# ── transports ────────────────────────────────────────────────────────────

class _BaseTransport:
    """One JSON-RPC channel to one server."""

    def request(self, payload: dict[str, Any],
                timeout: float) -> dict[str, Any]:
        raise NotImplementedError

    def notify(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _StdioTransport(_BaseTransport):
    """Subprocess over stdin/stdout, newline-delimited JSON-RPC.

    A reader thread owns stdout: it parses each line (Content-Length
    framing is accepted too) and wakes the pending request with the
    matching id. Malformed JSON or EOF is a transport error that wakes
    every pending request — a server that cannot speak JSON-RPC is
    reported, never silently skipped.
    """

    def __init__(self, config: MCPServerConfig, *,
                 max_payload_bytes: int) -> None:
        self._config = config
        self._max_payload = max_payload_bytes
        env = dict(os.environ)
        for var in config.env:
            if var not in env:
                _log.warning("mcp[%s]: env %s is not set; the server "
                             "may refuse to start", config.name, var)
        # Literal values from the declaration win over passthrough
        # names (an ACP client hands over the values themselves).
        for key, value in (config.env_values or {}).items():
            env[str(key)] = str(value)
        try:
            self._proc = subprocess.Popen(
                [config.command, *config.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env)
        except (OSError, ValueError) as exc:
            raise MCPClientError(
                f"mcp[{config.name}]: cannot spawn "
                f"{config.command!r}: {exc}") from exc
        self._pending: dict[Any, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._broken: MCPClientError | None = None
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_loop, name=f"mcp-{config.name}-reader",
            daemon=True)
        self._reader.start()
        # stderr is the server's log channel; drain it so a chatty
        # server cannot block on a full pipe.
        self._stderr_drain = threading.Thread(
            target=self._drain_stderr, name=f"mcp-{config.name}-stderr",
            daemon=True)
        self._stderr_drain.start()

    # ── reader ──
    def _drain_stderr(self) -> None:
        try:
            assert self._proc.stderr is not None
            for line in self._proc.stderr:
                text = line.decode("utf-8", "replace").strip()
                if text:
                    _log.debug("mcp[%s] stderr: %s", self._config.name,
                               text[:500])
        except Exception:  # noqa: BLE001 - drain must never break us
            pass

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        stream = self._proc.stdout
        while True:
            try:
                msg = self._read_message(stream)
            except MCPClientError as exc:
                self._break(exc)
                return
            except Exception as exc:  # noqa: BLE001 - reader never dies silent
                self._break(MCPClientError(
                    f"mcp[{self._config.name}]: stdio read failed: {exc}"))
                return
            if msg is None:
                self._break(MCPClientError(
                    f"mcp[{self._config.name}]: server exited "
                    f"(rc={self._proc.poll()})"))
                return
            self._dispatch(msg)

    def _read_message(self, stream: Any) -> dict[str, Any] | None:
        first = stream.readline()
        if not first:
            return None
        if len(first) > self._max_payload:
            raise MCPClientError(
                f"mcp[{self._config.name}]: frame past the "
                f"{self._max_payload}-byte payload cap")
        if first.strip().lower().startswith(b"content-length:"):
            try:
                length = int(first.split(b":", 1)[1].strip())
            except ValueError as exc:
                raise MCPClientError(
                    f"mcp[{self._config.name}]: bad Content-Length "
                    f"header {first!r}") from exc
            if length > self._max_payload:
                raise MCPClientError(
                    f"mcp[{self._config.name}]: frame past the "
                    f"{self._max_payload}-byte payload cap")
            while True:
                line = stream.readline()
                if not line or line.strip() in (b"", b"\r"):
                    break
            raw = stream.read(length)
            if not raw:
                return None
            return self._parse(raw)
        return self._parse(first)

    def _parse(self, raw: bytes) -> dict[str, Any]:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MCPClientError(
                f"mcp[{self._config.name}]: malformed JSON from server: "
                f"{exc} (first 200 bytes: {raw[:200]!r})",
                code=PARSE_ERROR) from exc
        if not isinstance(msg, dict):
            raise MCPClientError(
                f"mcp[{self._config.name}]: server sent a non-object "
                f"JSON-RPC frame: {str(msg)[:200]!r}",
                code=INVALID_REQUEST)
        return msg

    def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_id = msg.get("id")
        if msg_id is not None and ("result" in msg or "error" in msg):
            with self._pending_lock:
                slot = self._pending.pop(msg_id, None)
            if slot is not None:
                slot.put(msg)
            return
        # A server→client request (sampling/*, elicitation/*, roots/*)
        # or a notification: the client offers none of those, so log
        # and move on. Notifications carry no id and need no reply.
        method = msg.get("method", "")
        if msg_id is not None:
            # A request we cannot serve: answer method_not_found so the
            # server does not hang on us (best effort; stdout may be
            # gone if the server is already dying).
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort only
                self._send_raw({
                    "jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": METHOD_NOT_FOUND,
                              "message": f"method not implemented: {method}"},
                })
        else:
            _log.debug("mcp[%s]: notification %s", self._config.name,
                       method)

    def _break(self, exc: MCPClientError) -> None:
        with self._pending_lock:
            if self._broken is None:
                self._broken = exc
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot.put(exc)

    def _send_raw(self, payload: dict[str, Any]) -> None:
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            if self._closed or self._proc.stdin is None:
                raise MCPClientError(
                    f"mcp[{self._config.name}]: transport is closed")
            try:
                self._proc.stdin.write(raw)
                self._proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise MCPClientError(
                    f"mcp[{self._config.name}]: write failed: {exc}"
                    ) from exc

    # ── interface ──
    def request(self, payload: dict[str, Any],
                timeout: float) -> dict[str, Any]:
        if self._broken is not None:
            raise self._broken
        msg_id = payload.get("id")
        slot: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            if self._broken is not None:
                raise self._broken
            self._pending[msg_id] = slot
        try:
            self._send_raw(payload)
        except MCPClientError:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise
        try:
            reply = slot.get(timeout=timeout)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise MCPTimeoutError(
                f"mcp[{self._config.name}]: no reply to "
                f"{payload.get('method')} after {timeout:g}s") from exc
        if isinstance(reply, MCPClientError):
            raise reply
        return reply

    def notify(self, payload: dict[str, Any]) -> None:
        try:
            self._send_raw(payload)
        except MCPClientError as exc:
            _log.debug("mcp[%s]: notification send failed: %s",
                       self._config.name, exc)

    def close(self) -> None:
        with self._write_lock:
            self._closed = True
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - close must never raise
            pass


class _StreamableHttpTransport(_BaseTransport):
    """Streamable HTTP (spec 2025-03-26+): POST JSON-RPC, get JSON or SSE.

    One request per POST; the server may answer with a plain JSON body
    or a ``text/event-stream`` whose ``data:`` frames carry the
    JSON-RPC messages. A 401/403 names the auth gap (e.g. Graphify's
    OAuth sign-in) instead of pasting a token anywhere.
    """

    def __init__(self, config: MCPServerConfig, *,
                 max_payload_bytes: int) -> None:
        self._config = config
        self._max_payload = max_payload_bytes
        self._session_id: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        for key, value in (self._config.headers or {}).items():
            headers[str(key)] = str(value)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def request(self, payload: dict[str, Any],
                timeout: float) -> dict[str, Any]:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._config.url, data=raw, headers=self._headers(),
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                session = resp.headers.get("Mcp-Session-Id")
                if session:
                    self._session_id = session
                ctype = (resp.headers.get("Content-Type") or "").lower()
                body = resp.read(self._max_payload + 1)
                if len(body) > self._max_payload:
                    raise MCPClientError(
                        f"mcp[{self._config.name}]: response past the "
                        f"{self._max_payload}-byte payload cap")
                if "text/event-stream" in ctype:
                    return self._from_sse(body, payload.get("id"))
                return self._parse_json(body)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise MCPClientError(
                    f"mcp[{self._config.name}]: HTTP {exc.code} — the "
                    f"server needs sign-in (e.g. Graphify's OAuth "
                    f"browser sign-in) before it answers; no token is "
                    f"sent by this client") from exc
            raise MCPClientError(
                f"mcp[{self._config.name}]: HTTP {exc.code} from "
                f"{self._config.url}") from exc
        except urllib.error.URLError as exc:
            reason = str(getattr(exc, "reason", exc))
            if "timed out" in reason.lower():
                raise MCPTimeoutError(
                    f"mcp[{self._config.name}]: no reply to "
                    f"{payload.get('method')} after {timeout:g}s") from exc
            raise MCPClientError(
                f"mcp[{self._config.name}]: cannot reach "
                f"{self._config.url}: {reason}") from exc
        except TimeoutError as exc:
            raise MCPTimeoutError(
                f"mcp[{self._config.name}]: no reply to "
                f"{payload.get('method')} after {timeout:g}s") from exc

    def _parse_json(self, body: bytes) -> dict[str, Any]:
        try:
            msg = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MCPClientError(
                f"mcp[{self._config.name}]: malformed JSON from server: "
                f"{exc} (first 200 bytes: {body[:200]!r})",
                code=PARSE_ERROR) from exc
        if not isinstance(msg, dict):
            raise MCPClientError(
                f"mcp[{self._config.name}]: server sent a non-object "
                f"JSON-RPC frame",
                code=INVALID_REQUEST)
        return msg

    def _from_sse(self, body: bytes, want_id: Any) -> dict[str, Any]:
        text = body.decode("utf-8", "replace")
        last: dict[str, Any] | None = None
        for frame in text.split("\n\n"):
            for line in frame.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        msg = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict) and (
                            msg.get("id") == want_id or "result" in msg
                            or "error" in msg):
                        last = msg
        if last is None:
            raise MCPClientError(
                f"mcp[{self._config.name}]: SSE stream carried no "
                f"JSON-RPC response")
        return last

    def notify(self, payload: dict[str, Any]) -> None:
        # A notification over Streamable HTTP is a POST the server
        # answers 202 to; a failure here is not worth breaking the
        # caller over.
        try:
            self.request(payload, timeout=10.0)
        except MCPClientError as exc:
            _log.debug("mcp[%s]: notification send failed: %s",
                       self._config.name, exc)

    def close(self) -> None:
        self._session_id = None


# ── client ────────────────────────────────────────────────────────────────

class MCPClient:
    """One connection to one MCP server. Transport-agnostic core.

    Connect lazily: constructing the client spawns nothing; ``connect()``
    (or the first operation) starts the transport and runs ``initialize``.
    One request id stream per client; requests are safe to call from
    several threads (stdio matches by id, HTTP is per-request).
    """

    def __init__(self, config: MCPServerConfig, *,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> None:
        config.validate()
        self.config = config
        self.timeout_s = (config.timeout_s or timeout_s or DEFAULT_TIMEOUT_S)
        self._max_payload = max_payload_bytes or DEFAULT_MAX_PAYLOAD_BYTES
        self._transport: _BaseTransport | None = None
        self._lock = threading.RLock()
        self._next_id = 0
        self.connected = False
        self.server_info: dict[str, Any] = {}
        self.protocol_version = ""

    # ── lifecycle ──
    def connect(self) -> MCPClient:
        with self._lock:
            if self.connected:
                return self
            if self.config.transport == "stdio":
                self._transport = _StdioTransport(
                    self.config, max_payload_bytes=self._max_payload)
            elif self.config.transport == "http":
                self._transport = _StreamableHttpTransport(
                    self.config, max_payload_bytes=self._max_payload)
            else:  # pragma: no cover - config validation rejects this
                raise MCPClientError(
                    f"mcp[{self.config.name}]: unknown transport "
                    f"{self.config.transport!r}")
            try:
                result = self._request("initialize", {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": CLIENT_NAME,
                                   "version": CLIENT_VERSION},
                }, timeout=self.timeout_s)
            except MCPClientError:
                self._transport.close()
                self._transport = None
                raise
            server_version = str(result.get("protocolVersion") or "")
            if server_version in _MCP_VERSIONS:
                self.protocol_version = server_version
            else:
                # The server is the authority on the negotiated version;
                # an unknown one still speaks JSON-RPC, so accept it and
                # say so rather than refusing a working server.
                self.protocol_version = server_version or MCP_PROTOCOL_VERSION
                _log.warning("mcp[%s]: server negotiated unknown protocol "
                             "version %r", self.config.name, server_version)
            info = result.get("serverInfo") or {}
            if isinstance(info, dict):
                self.server_info = info
            self.connected = True
            # Notification never breaks connect.
            with contextlib.suppress(Exception):  # noqa: BLE001
                self._transport.notify({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"})
            _log.info("mcp[%s]: connected (protocol %s, server %s)",
                      self.config.name, self.protocol_version,
                      self.server_info.get("name", "?"))
            return self

    def ensure_connected(self) -> MCPClient:
        if not self.connected:
            return self.connect()
        return self

    def close(self) -> None:
        with self._lock:
            transport, self._transport = self._transport, None
            self.connected = False
            self.server_info = {}
            self.protocol_version = ""
        if transport is not None:
            transport.close()

    def __enter__(self) -> MCPClient:
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── JSON-RPC ──
    def _request(self, method: str, params: dict[str, Any] | None = None,
                 *, timeout: float | None = None) -> dict[str, Any]:
        if self._transport is None:
            raise MCPClientError(
                f"mcp[{self.config.name}]: not connected")
        with self._lock:
            self._next_id += 1
            msg_id = self._next_id
        payload: dict[str, Any] = {
            "jsonrpc": "2.0", "id": msg_id, "method": method,
        }
        if params is not None:
            payload["params"] = params
        reply = self._transport.request(
            payload, timeout or self.timeout_s)
        if "error" in reply:
            err = reply["error"] or {}
            code = err.get("code", INTERNAL_ERROR) if isinstance(
                err, dict) else INTERNAL_ERROR
            message = err.get("message", "unknown error") if isinstance(
                err, dict) else str(err)
            raise MCPClientError(
                f"mcp[{self.config.name}]: {method} failed: {message}",
                code=int(code) if isinstance(code, int) else INTERNAL_ERROR)
        result = reply.get("result")
        if not isinstance(result, dict):
            raise MCPClientError(
                f"mcp[{self.config.name}]: {method} returned a "
                f"non-object result: {str(result)[:200]!r}")
        return result

    # ── operations ──
    def ping(self) -> dict[str, Any]:
        self.ensure_connected()
        return self._request("ping", {})

    def list_tools(self) -> list[dict[str, Any]]:
        self.ensure_connected()
        result = self._request("tools/list", {})
        tools = result.get("tools") or []
        if not isinstance(tools, list):
            raise MCPClientError(
                f"mcp[{self.config.name}]: tools/list returned a "
                f"non-list tools field")
        return [t for t in tools if isinstance(t, dict)]

    def call_tool(self, name: str,
                  arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a tool. Returns the raw result (content, isError,
        structuredContent when offered) — the caller decides what an
        ``isError: true`` result means; it is a RESULT, not a
        transport failure, so it is returned, never raised."""
        if not isinstance(name, str) or not name:
            raise MCPClientError("call_tool requires a tool name")
        if arguments is not None and not isinstance(arguments, dict):
            raise MCPClientError("'arguments' must be an object")
        self.ensure_connected()
        return self._request("tools/call", {
            "name": name, "arguments": arguments or {}})

    def call_tool_text(self, name: str,
                       arguments: dict[str, Any] | None = None) -> str:
        """The tool result as plain text (concatenated text content)."""
        result = self.call_tool(name, arguments)
        parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        if parts:
            return "\n".join(parts)
        structured = result.get("structuredContent")
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False)
        return ""

    def list_resources(self) -> list[dict[str, Any]]:
        self.ensure_connected()
        result = self._request("resources/list", {})
        resources = result.get("resources") or []
        if not isinstance(resources, list):
            raise MCPClientError(
                f"mcp[{self.config.name}]: resources/list returned a "
                f"non-list resources field")
        return [r for r in resources if isinstance(r, dict)]

    def read_resource(self, uri: str) -> dict[str, Any]:
        if not isinstance(uri, str) or not uri:
            raise MCPClientError("read_resource requires a uri")
        self.ensure_connected()
        return self._request("resources/read", {"uri": uri})

    def list_prompts(self) -> list[dict[str, Any]]:
        self.ensure_connected()
        result = self._request("prompts/list", {})
        prompts = result.get("prompts") or []
        if not isinstance(prompts, list):
            raise MCPClientError(
                f"mcp[{self.config.name}]: prompts/list returned a "
                f"non-list prompts field")
        return [p for p in prompts if isinstance(p, dict)]

    def get_prompt(self, name: str,
                   arguments: dict[str, Any] | None = None
                   ) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise MCPClientError("get_prompt requires a prompt name")
        self.ensure_connected()
        return self._request("prompts/get", {
            "name": name, "arguments": arguments or {}})


# ── manager ───────────────────────────────────────────────────────────────

class MCPClientManager:
    """The named-server pool: registry, budgets, and connection cache.

    Built from settings (``from_settings``); a manager built from None
    carries the shipped defaults with the master switch off. One
    client object per server, connected lazily on first use; ``close()``
    shuts them all down.
    """

    def __init__(self, settings: Any = None, *,
                 servers: dict[str, MCPServerConfig] | None = None,
                 ) -> None:
        self._settings = settings
        if servers is not None:
            # An explicit server set — an ACP session's declared
            # servers. The declaration itself is the enablement: no
            # config master switch, no per-server config flag. The
            # session principal's capability grant still gates every
            # call made through this manager.
            self._registry = {
                name: replace(cfg, enabled=True)
                for name, cfg in servers.items()}
            self._enabled = True
        else:
            self._registry = effective_mcp_servers(settings)
            self._enabled = _mcp_enabled(settings)
        self._default_timeout = (
            _mcp_default_timeout(settings) or DEFAULT_TIMEOUT_S)
        self._max_payload = (
            _mcp_max_payload(settings) or DEFAULT_MAX_PAYLOAD_BYTES)
        self._max_servers = (
            _mcp_max_servers(settings) or DEFAULT_MAX_SERVERS)
        self._clients: dict[str, MCPClient] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_settings(cls, settings: Any) -> MCPClientManager:
        return cls(settings)

    # ── registry ──
    def servers(self) -> list[MCPServerConfig]:
        return [self._registry[name] for name in sorted(self._registry)]

    def server(self, name: str) -> MCPServerConfig:
        cfg = self._registry.get(name)
        if cfg is None:
            known = ", ".join(sorted(self._registry))
            raise MCPClientError(
                f"unknown mcp server {name!r}; known: {known}")
        return cfg

    def status(self) -> dict[str, Any]:
        """Master state + one line per server (no connections made)."""
        out: dict[str, Any] = {
            "enabled": self._enabled,
            "max_servers": self._max_servers,
            "connected": len([c for c in self._clients.values()
                              if c.connected]),
            "servers": {},
        }
        for cfg in self.servers():
            client = self._clients.get(cfg.name)
            tools: int | None = None
            error = ""
            if client is not None and client.connected:
                try:
                    tools = len(client.list_tools())
                except MCPClientError as exc:
                    error = str(exc)
            out["servers"][cfg.name] = {
                "enabled": cfg.enabled,
                "transport": cfg.transport,
                "connected": bool(client is not None and client.connected),
                "tools": tools,
                "error": error,
                "notes": cfg.notes,
            }
        return out

    # ── connections ──
    def connect(self, name: str) -> MCPClient:
        """Connect (or return the cached connection) to one server.

        Refuses honestly: master off, server unknown, server disabled,
        or the connection budget spent.
        """
        with self._lock:
            if not self._enabled:
                raise MCPClientError(
                    "mcp client is off (mcp.enabled is false in config); "
                    "enable it and the server first")
            cfg = self.server(name)
            if not cfg.enabled:
                raise MCPClientError(
                    f"mcp server {name!r} is disabled in config; "
                    f"enable it under mcp.servers first")
            existing = self._clients.get(name)
            if existing is not None and existing.connected:
                return existing
            live = len([c for c in self._clients.values() if c.connected])
            if live >= self._max_servers:
                raise MCPClientError(
                    f"mcp connection budget spent ({self._max_servers} "
                    f"servers); close one first")
            client = MCPClient(
                cfg, timeout_s=self._default_timeout,
                max_payload_bytes=self._max_payload)
            self._clients[name] = client
        # Connect outside the manager lock (initialize does I/O).
        client.connect()
        return client

    def close(self, name: str | None = None) -> None:
        with self._lock:
            if name is not None:
                client = self._clients.pop(name, None)
                targets = [client] if client is not None else []
            else:
                targets = list(self._clients.values())
                self._clients.clear()
        for client in targets:
            client.close()

    # ── operations (connect on first use) ──
    def list_tools(self, name: str) -> list[dict[str, Any]]:
        return self.connect(name).list_tools()

    def call_tool(self, server: str, tool: str,
                  arguments: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
        return self.connect(server).call_tool(tool, arguments)

    def call_tool_text(self, server: str, tool: str,
                       arguments: dict[str, Any] | None = None) -> str:
        return self.connect(server).call_tool_text(tool, arguments)

    def list_resources(self, name: str) -> list[dict[str, Any]]:
        return self.connect(name).list_resources()

    def read_resource(self, server: str, uri: str) -> dict[str, Any]:
        return self.connect(server).read_resource(uri)

    def list_prompts(self, name: str) -> list[dict[str, Any]]:
        return self.connect(name).list_prompts()

    def get_prompt(self, server: str, name: str,
                   arguments: dict[str, Any] | None = None
                   ) -> dict[str, Any]:
        return self.connect(server).get_prompt(name, arguments)


# ── tool registry wiring ──────────────────────────────────────────────────

def _manager(registry: Any) -> MCPClientManager:
    """The manager for this registry, built once from its settings."""
    manager = getattr(registry, "_mcp_client_manager", None)
    if manager is None:
        context = getattr(registry, "context", None)
        settings = getattr(context, "settings", None)
        manager = MCPClientManager.from_settings(settings)
        # Caching is best-effort.
        with contextlib.suppress(Exception):  # noqa: BLE001
            registry._mcp_client_manager = manager  # noqa: SLF001
    return manager


def register(registry: Any) -> None:
    """Register the MCP client tools on the spine.

    The brain calls external MCP servers through these, the same way
    it calls her own tools. Capability is net.out: an MCP call leaves
    the box (a subprocess, or an HTTP request to a remote server).
    Nothing connects unless the operator enabled it in config.
    """
    from ..core.policy import Capability

    @registry.register(
        "mcp_call",
        description=(
            "Call a tool on an enabled external MCP server (Context7, "
            "GitHub, Playwright, Serena, Graphify, Blender, "
            "since-cutoff, bestax, Upload-Post, Human Pages, Sato Hub, "
            "Ceki, or any server added under mcp.servers in config). "
            "Use when the task needs a capability Devon does not have "
            "herself — library docs, repo operations, browser control, "
            "semantic code search, a code graph, a live Blender "
            "session, stale-API checks for pinned Python deps, "
            "Bulma component docs, social publishing, hiring a human, "
            "preflighting an MCP server, or a real residential "
            "browser. Servers are opt-in: a disabled server refuses "
            "honestly."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "server": "str — the MCP server name (see /mcp list)",
            "tool": "str — the tool name on that server",
            "arguments": "object (optional) — the tool's arguments",
        },
    )
    def _mcp_call(server: str, tool: str,
                  arguments: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
        manager = _manager(registry)
        try:
            result = manager.call_tool(server, tool, arguments or {})
        except MCPClientError as exc:
            return {"ok": False, "error": str(exc)}
        text_parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text") or ""))
        if result.get("isError"):
            return {"ok": False,
                    "error": "\n".join(text_parts) or "tool reported an error",
                    "structured": result.get("structuredContent")}
        return {"ok": True, "text": "\n".join(text_parts),
                "structured": result.get("structuredContent")}

    @registry.register(
        "mcp_list_tools",
        description=(
            "List the tools an enabled external MCP server offers. "
            "No server name lists every enabled server's tools. Use "
            "it to discover what a server can do before calling it."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "server": "str (optional) — the MCP server name; empty "
                      "lists every enabled server",
        },
    )
    def _mcp_list_tools(server: str = "") -> dict[str, Any]:
        manager = _manager(registry)
        names = [server] if server else [
            cfg.name for cfg in manager.servers() if cfg.enabled]
        out: dict[str, Any] = {"ok": True, "servers": {}}
        errors: dict[str, str] = {}
        for name in names:
            try:
                out["servers"][name] = [
                    t.get("name") for t in manager.list_tools(name)]
            except MCPClientError as exc:
                errors[name] = str(exc)
        if errors:
            out["errors"] = errors
        return out

    @registry.register(
        "mcp_status",
        description=(
            "The MCP client state: whether the client is enabled, and "
            "per server whether it is enabled, connected, and how many "
            "tools it offers. No connections are made by this call."
        ),
        capability=Capability.NET_OUT,
        parameters={},
    )
    def _mcp_status() -> dict[str, Any]:
        return {"ok": True, "status": _manager(registry).status()}


# ── /mcp chat command ─────────────────────────────────────────────────────

_MCP_USAGE = (
    "/mcp — MCP client status (which external servers are enabled, "
    "connected, and what they offer)\n"
    "/mcp list — the known servers and their state\n"
    "/mcp tools [server] — the tools a server offers\n"
    "/mcp call <server> <tool> [json-args] — call a tool\n"
    "/mcp close [server] — drop a connection (bare closes all)\n"
    "Servers are opt-in in config (mcp.enabled + mcp.servers.<name>.enabled); "
    "nothing connects until enabled."
)

#: Chat replies cap tool text at this many chars, with an explicit
#: truncation notice (never truncate silently).
_MCP_CHAT_TEXT_CAP = 4000


def control_mcp(tail: str, settings: Any = None) -> str:
    """The /mcp command. Owner-only (CONTROL_COMMANDS, never public).

    /mcp — status · /mcp list — servers · /mcp tools [server] —
    tool lists · /mcp call <server> <tool> [json-args] — one call ·
    /mcp close [server] — drop connections.
    """
    manager = MCPClientManager.from_settings(settings)
    parts = (tail or "").split()
    verb = parts[0].lower() if parts else "status"

    if verb in ("status", ""):
        st = manager.status()
        lines = [f"MCP client: {'on' if st['enabled'] else 'OFF'} "
                 f"(mcp.enabled in config) — "
                 f"{st['connected']}/{st['max_servers']} connected"]
        for name, info in st["servers"].items():
            state = ("connected" if info["connected"]
                     else "enabled" if info["enabled"] else "disabled")
            tools = (f", {info['tools']} tools"
                     if info["tools"] is not None else "")
            err = f" — {info['error']}" if info["error"] else ""
            notes = f" ({info['notes']})" if info["notes"] else ""
            lines.append(f"· {name} [{info['transport']}]: {state}"
                         f"{tools}{err}{notes}")
        if not st["enabled"]:
            lines.append("Enable it in config (mcp.enabled = true) and "
                         "enable a server under mcp.servers to connect.")
        return "\n".join(lines)

    if verb == "list":
        lines = ["MCP servers:"]
        for cfg in manager.servers():
            state = "enabled" if cfg.enabled else "disabled"
            where = cfg.url if cfg.transport == "http" else (
                f"{cfg.command} {' '.join(cfg.args)}".strip())
            lines.append(f"· {cfg.name} [{cfg.transport}]: {state} — "
                         f"{where}")
            if cfg.notes:
                lines.append(f"  {cfg.notes}")
        return "\n".join(lines)

    if verb == "tools":
        names = [parts[1]] if len(parts) > 1 else [
            cfg.name for cfg in manager.servers() if cfg.enabled]
        if not names:
            return ("No MCP server is enabled. Enable one under "
                    "mcp.servers in config first.\n" + _MCP_USAGE)
        lines: list[str] = []
        for name in names:
            try:
                tools = manager.list_tools(name)
            except MCPClientError as exc:
                lines.append(f"· {name}: {exc}")
                continue
            if not tools:
                lines.append(f"· {name}: no tools")
                continue
            lines.append(f"· {name} ({len(tools)} tools):")
            for tool in tools:
                desc = str(tool.get("description") or "").strip()
                lines.append(f"  - {tool.get('name')}"
                             + (f": {desc[:120]}" if desc else ""))
        return "\n".join(lines)

    if verb == "call":
        if len(parts) < 3:
            return "Usage: /mcp call <server> <tool> [json-args]\n" + _MCP_USAGE
        server, tool = parts[1], parts[2]
        args: dict[str, Any] = {}
        if len(parts) > 3:
            try:
                args = json.loads(" ".join(parts[3:]))
            except json.JSONDecodeError as exc:
                return f"Bad JSON arguments: {exc}\n" + _MCP_USAGE
            if not isinstance(args, dict):
                return ("Tool arguments must be a JSON object.\n"
                        + _MCP_USAGE)
        try:
            result = manager.call_tool(server, tool, args)
        except MCPClientError as exc:
            return f"mcp call failed: {exc}"
        if result.get("isError"):
            err_parts = []
            for item in result.get("content") or []:
                if isinstance(item, dict) and item.get("type") == "text":
                    err_parts.append(str(item.get("text") or ""))
            return ("Tool reported an error:\n" +
                    ("\n".join(err_parts) or "(no message)"))
        text_parts = []
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text") or ""))
        text = "\n".join(text_parts)
        if not text and result.get("structuredContent") is not None:
            text = json.dumps(result["structuredContent"],
                              ensure_ascii=False)
        if len(text) > _MCP_CHAT_TEXT_CAP:
            text = (text[:_MCP_CHAT_TEXT_CAP] +
                    f"\n…(truncated, {len(text)} chars total)")
        return text or "(empty result)"

    if verb == "close":
        name = parts[1] if len(parts) > 1 else None
        manager.close(name)
        return (f"Closed the {name} connection."
                if name else "Closed all MCP connections.")

    return f"Unknown /mcp verb {verb!r}.\n" + _MCP_USAGE
