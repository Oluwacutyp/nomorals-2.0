"""A2A CLIENT — Devon talking to other agents over Agent2Agent.

The horizontal counterpart to ``nomorals/tools/mcp_client.py``:
MCP connects Devon DOWNWARD to tools and data; A2A connects her
ACROSS to peer agents. A peer publishes an Agent Card
(``/.well-known/agent-card.json``); Devon fetches it, sends it a
message, and follows the task it opens until the artifacts come back.

She builds every skill herself — that is universal. A2A is an
ADDITION, never a substitute: she does not reach for another agent
because a skill is missing — she builds the skill. A2A is for
working with other agents as PEERS: delegation to a specialist
that owns its own domain, collaboration across organizations,
exchange. Both live, side by side: every capability herself AND
the reach to talk to other agents too.

Protocol (verified against the A2A 1.0.1 specification, 2026-11-17 —
not from memory): JSON-RPC 2.0 over HTTPS, ``A2A-Version: 1.0``
header, PascalCase methods (``SendMessage``, ``SendStreamingMessage``,
``GetTask``, ``ListTasks``, ``CancelTask``); Agent Card fields
``name`` / ``description`` / ``version`` / ``supportedInterfaces``
(``url`` + ``protocolBinding`` + ``protocolVersion``) /
``capabilities`` (``streaming``, ``pushNotifications``) /
``securitySchemes`` / ``securityRequirements`` / ``defaultInputModes``
/ ``defaultOutputModes`` / ``skills``; parts carry exactly one of
``text`` / ``data`` / ``url`` / ``raw`` (no ``kind`` discriminator);
task states are ``TASK_STATE_*`` (submitted, working,
input-required, auth-required, completed, failed, canceled,
rejected).

The client ALSO speaks the pre-1.0 dialect (0.3 / 0.2): legacy
method names (``message/send``, ``message/stream``, ``tasks/get``,
``tasks/list``, ``tasks/cancel``), lowercase hyphenated states,
``kind``-based parts, top-level ``url`` + ``protocolVersion`` cards,
and the legacy ``/.well-known/agent.json`` card path as a fallback.
The card's protocol version picks the dialect — most clients pick
one version; this one negotiates.

What the client does NOT offer: the gRPC and HTTP+JSON (REST)
bindings (JSON-RPC only — a card with no JSONRPC interface is
refused in words naming it), push-notification config (Devon has no
public webhook to receive them on), and OAuth2 / OpenID Connect /
mutual-TLS auth (a card demanding those is refused in words naming
the scheme — Devon cannot complete those flows unattended). API-key
and HTTP-bearer cards work: the token comes from ONE environment
variable named in config (``api_key_env``), never from config or
code.

Session scoping: A2A tools are ordinary registry tools, so ACP's
``_default_turn`` (``nomorals/api/acp.py``) already scopes them —
it wraps ``registry.call`` per turn with ``traced_call`` under the
turn lock, attributing every A2A call to that session. The manager's
connection cache is keyed by agent name and holds no per-session
state: tasks are addressed by task id, and a multi-turn exchange
stays together via the caller-supplied ``context_id``. Nothing else
had to be built for scoping.

Known agents (``DEFAULT_A2A_AGENTS``) are empty ON PURPOSE: no
public A2A endpoint is stable enough to ship as a default. The
effective registry is built from config (``a2a.agents`` in the
TOML — see ``nomorals/core/config.py`` A2ASettings): a config entry
adds an agent. Nothing connects by default — the master switch
(``a2a.enabled``) AND each agent's ``enabled`` flag must be on.

Bounds: per-call timeout (default 30s, per-agent overridable),
per-payload cap (default 1MB), per-manager connection budget
(``a2a.max_agents``, default 8). A slow agent fails its own call;
it never stalls the brain.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "A2AClientError",
    "A2ATimeoutError",
    "A2AAgentConfig",
    "DEFAULT_A2A_AGENTS",
    "A2A_PROTOCOL_VERSION",
    "AgentCard",
    "fetch_agent_card",
    "task_state",
    "task_text",
    "A2AClient",
    "A2AClientManager",
    "effective_a2a_agents",
    "control_a2a",
    "register",
]

#: The protocol version this client prefers. A card may negotiate it
#: down to the 0.3 family; the client speaks both.
A2A_PROTOCOL_VERSION = "1.0"

#: Protocol versions the client can speak, newest first.
_A2A_VERSIONS = ("1.0", "0.3", "0.2")

#: The canonical card path (RFC 8615 well-known). The legacy path is
#: probed only when the canonical one 404s.
_A2A_CARD_PATH = "/.well-known/agent-card.json"
_A2A_LEGACY_CARD_PATH = "/.well-known/agent.json"

CLIENT_NAME = "devon"
CLIENT_VERSION = "2.0"

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_PAYLOAD_BYTES = 1_000_000
DEFAULT_MAX_AGENTS = 8
DEFAULT_WAIT_S = 120.0
_MIN_POLL_S = 0.25

#: JSON-RPC error codes (mirrors the MCP client module).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

#: Task states, short form (after normalization — see task_state).
TERMINAL_STATES = frozenset({"completed", "failed", "canceled", "rejected"})
#: Waiting states end a wait too: the remote agent is waiting on US.
WAITING_STATES = frozenset({"input-required", "auth-required"})

#: JSON-RPC method names per protocol family. 1.0 uses PascalCase
#: operation names; 0.3/0.2 use the slash names.
_METHODS: dict[str, dict[str, str]] = {
    "1.0": {
        "send": "SendMessage",
        "stream": "SendStreamingMessage",
        "get": "GetTask",
        "list": "ListTasks",
        "cancel": "CancelTask",
    },
    "0.3": {
        "send": "message/send",
        "stream": "message/stream",
        "get": "tasks/get",
        "list": "tasks/list",
        "cancel": "tasks/cancel",
    },
}


def _family(version: str) -> str:
    """The method/state family for a card's protocol version."""
    return "1.0" if str(version or "").startswith("1.") else "0.3"


class A2AClientError(Exception):
    """The agent, the transport, or the protocol failed. Never silent."""

    def __init__(self, message: str, code: int = INTERNAL_ERROR) -> None:
        super().__init__(message)
        self.code = code


class A2ATimeoutError(A2AClientError):
    """A single A2A call ran past its timeout. The agent is not dead —
    only that call is; the connection stays usable."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=INTERNAL_ERROR)


# ── agent configuration ─────────────────────────────────────────────────

@dataclass
class A2AAgentConfig:
    """One A2A agent as the client sees it.

    ``api_key_env`` names ONE environment variable holding a
    bearer/API-key token for a card that demands one (the value comes
    from the process environment / vault at call time — the config
    only names it). ``notes`` is human text shown by /a2a status.
    """

    name: str
    url: str = ""
    enabled: bool = False
    timeout_s: float = 0.0  # 0 = the manager default
    api_key_env: str = ""
    notes: str = ""

    def validate(self) -> None:
        """The config is usable as an agent (called on the EFFECTIVE
        registry — a config override is a partial and is never
        validated on its own)."""
        if not self.url:
            raise A2AClientError(
                f"a2a agent {self.name!r}: a base url is required")
        if not (self.url.startswith("http://")
                or self.url.startswith("https://")):
            raise A2AClientError(
                f"a2a agent {self.name!r}: url must be http(s), got "
                f"{self.url!r}")

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> A2AAgentConfig:
        unknown = set(raw) - {
            "url", "enabled", "timeout_s", "api_key_env", "notes"}
        if unknown:
            raise A2AClientError(
                f"a2a agent {name!r}: unknown key(s) {sorted(unknown)}")
        return cls(
            name=name,
            url=str(raw.get("url") or ""),
            enabled=bool(raw.get("enabled", False)),
            timeout_s=float(raw.get("timeout_s") or 0.0),
            api_key_env=str(raw.get("api_key_env") or ""),
            notes=str(raw.get("notes") or ""),
        )

    def merge(self, override: A2AAgentConfig) -> A2AAgentConfig:
        """A config entry overriding this default, field by field.

        Only set fields win: a non-empty url/api_key_env/notes, an
        explicit enabled, a non-zero timeout. An override that sets
        nothing changes nothing.
        """
        return A2AAgentConfig(
            name=self.name,
            url=override.url or self.url,
            enabled=override.enabled,
            timeout_s=override.timeout_s or self.timeout_s,
            api_key_env=override.api_key_env or self.api_key_env,
            notes=override.notes or self.notes,
        )


#: Shipped defaults for the agents Devon knows. EMPTY ON PURPOSE —
#: no public A2A endpoint is stable enough to ship as a default.
#: The effective registry comes from config (see
#: effective_a2a_agents). An operator adds an agent under
#: ``a2a.agents`` in the TOML.
DEFAULT_A2A_AGENTS: dict[str, A2AAgentConfig] = {}


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


def effective_a2a_agents(settings: Any) -> dict[str, A2AAgentConfig]:
    """The effective registry: shipped defaults overlaid with config.

    ``settings`` is the app Settings (or None → defaults only, all
    disabled). A config entry with a known name overrides that
    agent's default field by field; a new name adds an agent.
    """
    out = {name: cfg for name, cfg in DEFAULT_A2A_AGENTS.items()}
    entries = _field(_field(settings, "a2a", None), "agents", None) or {}
    for name, entry in entries.items():
        # Config entries arrive as A2AAgentEntry (config.py coerces
        # them) or as plain dicts (tests / direct construction).
        if isinstance(entry, dict):
            override = A2AAgentConfig.from_dict(name, entry)
        else:
            try:
                fields = vars(entry)
            except TypeError as exc:
                raise A2AClientError(
                    f"a2a agent {name!r}: entry must be a table, got "
                    f"{type(entry).__name__}") from exc
            if not isinstance(fields, dict):
                raise A2AClientError(
                    f"a2a agent {name!r}: entry must be a table, got "
                    f"{type(entry).__name__}")
            override = A2AAgentConfig(
                name=name,
                url=str(fields.get("url") or ""),
                enabled=bool(fields.get("enabled", False)),
                timeout_s=float(fields.get("timeout_s") or 0.0),
                api_key_env=str(fields.get("api_key_env") or ""),
                notes=str(fields.get("notes") or ""),
            )
        if name in out:
            out[name] = out[name].merge(override)
        else:
            out[name] = override
    for cfg in out.values():
        cfg.validate()
    return out


def _a2a_enabled(settings: Any) -> bool:
    return bool(_field(_field(settings, "a2a", None), "enabled", False))


def _a2a_default_timeout(settings: Any) -> float:
    try:
        return float(
            _field(_field(settings, "a2a", None), "default_timeout_s", 0.0)
            or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _a2a_max_payload(settings: Any) -> int:
    try:
        return int(
            _field(_field(settings, "a2a", None), "max_payload_bytes", 0)
            or 0)
    except (TypeError, ValueError):
        return 0


def _a2a_max_agents(settings: Any) -> int:
    try:
        return int(
            _field(_field(settings, "a2a", None), "max_agents", 0) or 0)
    except (TypeError, ValueError):
        return 0


# ── agent card ──────────────────────────────────────────────────────────

@dataclass
class AgentCard:
    """A parsed Agent Card — the peer's public contract.

    ``endpoint_url`` is the JSON-RPC endpoint (from the card's
    JSONRPC interface, else its top-level ``url``, else the base
    URL). ``protocol_version`` is the card's declared version;
    ``family`` is the method/state family it selects ("1.0" or
    "0.3"). ``raw`` keeps the card verbatim for inspection.
    """

    name: str
    description: str
    version: str
    protocol_version: str
    endpoint_url: str
    streaming: bool
    push_notifications: bool
    skills: list[dict[str, Any]]
    security_schemes: dict[str, Any]
    security_requirements: list[dict[str, Any]]
    raw: dict[str, Any]
    card_path: str = _A2A_CARD_PATH

    @property
    def family(self) -> str:
        return _family(self.protocol_version)

    def skill_ids(self) -> list[str]:
        return [str(s.get("id") or "") for s in self.skills
                if isinstance(s, dict) and s.get("id")]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_url: str,
                  card_path: str = _A2A_CARD_PATH) -> AgentCard:
        if not isinstance(raw, dict):
            raise A2AClientError(
                f"a2a: agent card from {base_url} is not an object")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise A2AClientError(
                f"a2a: agent card from {base_url} has no name")
        interfaces = raw.get("supportedInterfaces") or []
        if not isinstance(interfaces, list):
            interfaces = []
        interfaces = [i for i in interfaces if isinstance(i, dict)]
        endpoint = ""
        declared_version = ""
        jsonrpc_url = ""
        for iface in interfaces:
            url = str(iface.get("url") or "").strip()
            if not url:
                continue
            binding = str(iface.get("protocolBinding") or "").upper()
            version = str(iface.get("protocolVersion") or "")
            if binding == "JSONRPC" and not jsonrpc_url:
                jsonrpc_url = url
                declared_version = version
            if not endpoint:
                endpoint = url
                if not declared_version:
                    declared_version = version
        if jsonrpc_url:
            endpoint = jsonrpc_url
        elif interfaces:
            # The card lists interfaces but none of them is JSON-RPC
            # — this client speaks JSON-RPC only; say so plainly.
            raise A2AClientError(
                f"a2a: agent card from {base_url} lists no JSONRPC "
                f"interface (bindings: "
                f"{[str(i.get('protocolBinding')) for i in interfaces]})")
        elif raw.get("url"):
            endpoint = str(raw.get("url"))
        else:
            endpoint = base_url
        version = (declared_version
                   or str(raw.get("protocolVersion") or "")
                   or ("1.0" if interfaces else "0.3"))
        capabilities = raw.get("capabilities") or {}
        if not isinstance(capabilities, dict):
            capabilities = {}
        skills = [s for s in (raw.get("skills") or [])
                  if isinstance(s, dict)]
        schemes = raw.get("securitySchemes") or {}
        if not isinstance(schemes, dict):
            schemes = {}
        requirements = raw.get("securityRequirements") or []
        if not isinstance(requirements, list):
            requirements = []
        requirements = [r for r in requirements if isinstance(r, dict)]
        return cls(
            name=name,
            description=str(raw.get("description") or ""),
            version=str(raw.get("version") or ""),
            protocol_version=version,
            endpoint_url=endpoint,
            streaming=bool(capabilities.get("streaming")),
            push_notifications=bool(capabilities.get("pushNotifications")),
            skills=skills,
            security_schemes=schemes,
            security_requirements=requirements,
            raw=raw,
            card_path=card_path,
        )


def _http_get_json(url: str, *, timeout: float,
                   max_payload_bytes: int) -> Any:
    """GET a URL and parse its JSON body. Transport errors are
    A2AClientErrors naming the URL, never silent."""
    req = urllib.request.Request(
        url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(max_payload_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise A2AClientError(
            f"a2a: HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        if "timed out" in reason.lower():
            raise A2ATimeoutError(
                f"a2a: no reply from {url} after {timeout:g}s") from exc
        raise A2AClientError(
            f"a2a: cannot reach {url}: {reason}") from exc
    except TimeoutError as exc:
        raise A2ATimeoutError(
            f"a2a: no reply from {url} after {timeout:g}s") from exc
    if len(body) > max_payload_bytes:
        raise A2AClientError(
            f"a2a: response from {url} past the "
            f"{max_payload_bytes}-byte payload cap")
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise A2AClientError(
            f"a2a: malformed JSON from {url}: {exc} "
            f"(first 200 bytes: {body[:200]!r})",
            code=PARSE_ERROR) from exc


def fetch_agent_card(base_url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S,
                     max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
                     ) -> AgentCard:
    """Fetch and parse an agent's card.

    The canonical ``/.well-known/agent-card.json`` path first; the
    legacy ``/.well-known/agent.json`` path only when the canonical
    one 404s (pre-1.0 agents). A card that is not JSON, not an
    object, or names no usable endpoint is an error, never a guess.
    """
    base = base_url.rstrip("/")
    for path in (_A2A_CARD_PATH, _A2A_LEGACY_CARD_PATH):
        url = base + path
        try:
            raw = _http_get_json(
                url, timeout=timeout_s, max_payload_bytes=max_payload_bytes)
        except A2AClientError as exc:
            if "HTTP 404" in str(exc) and path == _A2A_CARD_PATH:
                continue
            raise
        return AgentCard.from_dict(raw, base_url, card_path=path)
    raise A2AClientError(
        f"a2a: no agent card at {base}{_A2A_CARD_PATH} or "
        f"{base}{_A2A_LEGACY_CARD_PATH}")


# ── auth ────────────────────────────────────────────────────────────────

def _auth_headers(card: AgentCard, config: A2AAgentConfig
                  ) -> dict[str, str]:
    """Headers satisfying the card's security requirements.

    API-key and HTTP-bearer cards take a token from the ONE
    environment variable named in config (values from the process
    environment / vault at call time — never from config or code).
    A card whose requirements are all OAuth2 / OpenID Connect /
    mutual-TLS is refused in words: Devon cannot complete those
    flows unattended. Requirements are OR-ed: the first satisfiable
    scheme wins.
    """
    requirements = card.security_requirements
    if not requirements:
        return {}
    unsupported: list[str] = []
    for req in requirements:
        schemes = req.get("schemes")
        if not isinstance(schemes, dict):
            # 0.3 shape: the requirement itself maps scheme -> scopes.
            schemes = {k: v for k, v in req.items()
                       if isinstance(v, (list, dict))}
        for scheme_name in schemes:
            definition = card.security_schemes.get(scheme_name)
            if not isinstance(definition, dict):
                raise A2AClientError(
                    f"a2a[{card.name}]: card requires unknown auth "
                    f"scheme {scheme_name!r}")
            if "apiKeySecurityScheme" in definition:
                spec = definition["apiKeySecurityScheme"] or {}
                where = str(spec.get("in") or "header").lower()
                header = str(spec.get("name") or "X-Api-Key")
                if where != "header":
                    unsupported.append(
                        f"{scheme_name} (api key in {where!r})")
                    continue
                token = _token_from_env(config, scheme_name)
                return {header: token}
            if "httpAuthSecurityScheme" in definition:
                spec = definition["httpAuthSecurityScheme"] or {}
                scheme = str(spec.get("scheme") or "bearer").lower()
                if scheme != "bearer":
                    unsupported.append(
                        f"{scheme_name} (http {scheme!r})")
                    continue
                token = _token_from_env(config, scheme_name)
                return {"Authorization": f"Bearer {token}"}
            if "oauth2SecurityScheme" in definition:
                unsupported.append(f"{scheme_name} (OAuth2)")
            elif "openIdConnectSecurityScheme" in definition:
                unsupported.append(f"{scheme_name} (OpenID Connect)")
            elif "mutualTLSSecurityScheme" in definition:
                unsupported.append(f"{scheme_name} (mutual TLS)")
            else:
                unsupported.append(f"{scheme_name} (unknown scheme shape)")
    if unsupported:
        raise A2AClientError(
            f"a2a[{card.name}]: agent requires "
            f"{', '.join(unsupported)} — Devon cannot complete those "
            f"auth flows unattended; use an agent whose card accepts "
            f"an API key or a bearer token")
    raise A2AClientError(
        f"a2a[{card.name}]: card's security requirements name no "
        f"scheme this client can satisfy")


def _token_from_env(config: A2AAgentConfig, scheme_name: str) -> str:
    """The bearer/API-key token for a card, from the environment.

    The config names the variable; the value comes from the process
    environment / vault at call time. A card that demands a token
    with no variable configured — or an empty one — is refused in
    words naming the variable, never answered with a fake token.
    """
    env_name = (config.api_key_env or "").strip()
    if not env_name:
        raise A2AClientError(
            f"a2a[{config.name}]: agent requires auth (scheme "
            f"{scheme_name!r}) but no api_key_env is configured for "
            f"it; set a2a.agents.{config.name}.api_key_env to the "
            f"environment variable holding the token")
    token = os.environ.get(env_name) or ""
    if not token:
        raise A2AClientError(
            f"a2a[{config.name}]: agent requires auth (scheme "
            f"{scheme_name!r}) but ${env_name} is empty; put the "
            f"token in the environment first")
    return token


# ── task shapes ─────────────────────────────────────────────────────────

def task_state(task: dict[str, Any],
               protocol_version: str | None = None) -> str:
    """A task's state in short form (``completed``, ``working``,
    ``input-required``, ...). A bare Message result — a direct
    reply, no task — counts as ``completed``."""
    status = task.get("status")
    if not isinstance(status, dict):
        return "completed"
    raw = str(status.get("state") or "unknown")
    state = raw.removeprefix("TASK_STATE_").lower().replace("_", "-")
    if state in ("unspecified", ""):
        return "unknown"
    return state


def _part_texts(part: dict[str, Any]) -> list[str]:
    """The readable text of one part, in either dialect."""
    if not isinstance(part, dict):
        return []
    text = part.get("text")
    if isinstance(text, str) and text:
        return [text]
    data = part.get("data")
    if data is not None:
        return [json.dumps(data, ensure_ascii=False)]
    file_part = part.get("file")
    if isinstance(file_part, dict):
        name = str(file_part.get("name") or file_part.get("uri") or "")
        return [f"[file: {name}]"] if name else ["[file]"]
    url = part.get("url")
    if isinstance(url, str) and url:
        return [f"[url: {url}]"]
    return []


def task_text(task: dict[str, Any],
              protocol_version: str | None = None) -> str:
    """A task's (or message's) answer as plain text: artifact parts
    in order, then the status message's parts when the agent is
    asking something (input-required). Data parts render as JSON;
    file parts render as a ``[file: name]`` line."""
    parts: list[str] = []
    artifacts = task.get("artifacts") or []
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            for part in artifact.get("parts") or []:
                parts.extend(_part_texts(part))
    status = task.get("status")
    if isinstance(status, dict):
        message = status.get("message")
        if isinstance(message, dict):
            for part in message.get("parts") or []:
                parts.extend(_part_texts(part))
    if not parts and isinstance(task.get("parts"), list):
        # A bare Message result carries its parts at the top level.
        for part in task["parts"]:
            parts.extend(_part_texts(part))
    return "\n".join(parts)


# ── client ────────────────────────────────────────────────────────────────

class A2AClient:
    """One connection to one A2A agent. Transport-agnostic core.

    Connect lazily: constructing the client fetches nothing;
    ``connect()`` (or the first operation) fetches the Agent Card
    and negotiates the protocol version. One request id stream per
    client; requests are safe to call from several threads.
    """

    def __init__(self, config: A2AAgentConfig, *,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> None:
        config.validate()
        self.config = config
        self.timeout_s = (config.timeout_s or timeout_s or DEFAULT_TIMEOUT_S)
        self._max_payload = max_payload_bytes or DEFAULT_MAX_PAYLOAD_BYTES
        self._lock = threading.RLock()
        self._next_id = 0
        self.connected = False
        self.card: AgentCard | None = None
        self.protocol_version = ""
        self.endpoint_url = ""

    # ── lifecycle ──
    def connect(self) -> A2AClient:
        with self._lock:
            if self.connected:
                return self
            card = fetch_agent_card(
                self.config.url, timeout_s=self.timeout_s,
                max_payload_bytes=self._max_payload)
            self.card = card
            self.protocol_version = card.protocol_version
            self.endpoint_url = card.endpoint_url
            self.connected = True
            _log.info("a2a[%s]: connected (agent %s, protocol %s, "
                      "endpoint %s)", self.config.name, card.name,
                      card.protocol_version, card.endpoint_url)
            return self

    def ensure_connected(self) -> A2AClient:
        if not self.connected:
            return self.connect()
        return self

    def close(self) -> None:
        with self._lock:
            self.connected = False
            self.card = None
            self.protocol_version = ""
            self.endpoint_url = ""

    def __enter__(self) -> A2AClient:
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── JSON-RPC ──
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.card is not None and self.card.family == "1.0":
            headers["A2A-Version"] = self.protocol_version or "1.0"
        if self.card is not None:
            headers.update(_auth_headers(self.card, self.config))
        return headers

    def _methods(self) -> dict[str, str]:
        family = self.card.family if self.card is not None else "1.0"
        return _METHODS[family]

    def _post(self, payload: dict[str, Any], timeout: float
              ) -> tuple[bytes, str]:
        """One POST to the endpoint. Returns (body, content-type)."""
        if not self.endpoint_url:
            raise A2AClientError(
                f"a2a[{self.config.name}]: not connected")
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url, data=raw, headers=self._headers(),
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                body = resp.read(self._max_payload + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise A2AClientError(
                    f"a2a[{self.config.name}]: HTTP {exc.code} — the "
                    f"agent refused the call (bad or missing "
                    f"credentials); check api_key_env for this agent",
                    code=exc.code) from exc
            raise A2AClientError(
                f"a2a[{self.config.name}]: HTTP {exc.code} from "
                f"{self.endpoint_url}") from exc
        except urllib.error.URLError as exc:
            reason = str(getattr(exc, "reason", exc))
            if "timed out" in reason.lower():
                raise A2ATimeoutError(
                    f"a2a[{self.config.name}]: no reply after "
                    f"{timeout:g}s") from exc
            raise A2AClientError(
                f"a2a[{self.config.name}]: cannot reach "
                f"{self.endpoint_url}: {reason}") from exc
        except TimeoutError as exc:
            raise A2ATimeoutError(
                f"a2a[{self.config.name}]: no reply after "
                f"{timeout:g}s") from exc
        if len(body) > self._max_payload:
            raise A2AClientError(
                f"a2a[{self.config.name}]: response past the "
                f"{self._max_payload}-byte payload cap")
        return body, ctype

    def _parse_frame(self, body: bytes) -> dict[str, Any]:
        try:
            msg = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise A2AClientError(
                f"a2a[{self.config.name}]: malformed JSON from agent: "
                f"{exc} (first 200 bytes: {body[:200]!r})",
                code=PARSE_ERROR) from exc
        if not isinstance(msg, dict):
            raise A2AClientError(
                f"a2a[{self.config.name}]: agent sent a non-object "
                f"JSON-RPC frame",
                code=INVALID_REQUEST)
        return msg

    def _rpc(self, method: str, params: dict[str, Any] | None = None,
             *, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            msg_id = self._next_id
        payload: dict[str, Any] = {
            "jsonrpc": "2.0", "id": msg_id, "method": method,
        }
        if params is not None:
            payload["params"] = params
        body, _ctype = self._post(payload, timeout or self.timeout_s)
        reply = self._parse_frame(body)
        if "error" in reply:
            err = reply["error"] or {}
            code = err.get("code", INTERNAL_ERROR) if isinstance(
                err, dict) else INTERNAL_ERROR
            message = err.get("message", "unknown error") if isinstance(
                err, dict) else str(err)
            raise A2AClientError(
                f"a2a[{self.config.name}]: {method} failed: {message}",
                code=int(code) if isinstance(code, int) else INTERNAL_ERROR)
        result = reply.get("result")
        if not isinstance(result, dict):
            raise A2AClientError(
                f"a2a[{self.config.name}]: {method} returned a "
                f"non-object result: {str(result)[:200]!r}")
        return result

    def _rpc_stream(self, method: str, params: dict[str, Any] | None = None,
                    *, timeout: float | None = None
                    ) -> Iterator[dict[str, Any]]:
        """One streaming call: yields each StreamResponse result.

        A JSON body is yielded once (a non-streaming agent answering
        a stream call plainly); an SSE body yields one result per
        ``data:`` frame. The caller stops at a terminal state.
        """
        with self._lock:
            self._next_id += 1
            msg_id = self._next_id
        payload: dict[str, Any] = {
            "jsonrpc": "2.0", "id": msg_id, "method": method,
        }
        if params is not None:
            payload["params"] = params
        body, ctype = self._post(payload, timeout or self.timeout_s)
        if "text/event-stream" not in ctype:
            reply = self._parse_frame(body)
            if "error" in reply:
                err = reply["error"] or {}
                message = err.get("message", "unknown error") \
                    if isinstance(err, dict) else str(err)
                raise A2AClientError(
                    f"a2a[{self.config.name}]: {method} failed: "
                    f"{message}")
            result = reply.get("result")
            if isinstance(result, dict):
                yield result
            return
        text = body.decode("utf-8", "replace")
        for frame in text.split("\n\n"):
            for line in frame.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if "error" in msg:
                    err = msg["error"] or {}
                    message = err.get("message", "unknown error") \
                        if isinstance(err, dict) else str(err)
                    raise A2AClientError(
                        f"a2a[{self.config.name}]: {method} failed: "
                        f"{message}")
                result = msg.get("result")
                if isinstance(result, dict):
                    yield result

    # ── messages ──
    def _message(self, text: str, *, context_id: str | None,
                 task_id: str | None) -> dict[str, Any]:
        family = self.card.family if self.card is not None else "1.0"
        role = "ROLE_USER" if family == "1.0" else "user"
        part = ({"text": text} if family == "1.0"
                else {"kind": "text", "text": text})
        message: dict[str, Any] = {
            "messageId": uuid.uuid4().hex,
            "role": role,
            "parts": [part],
        }
        if context_id:
            message["contextId"] = context_id
        if task_id:
            message["taskId"] = task_id
        return message

    def _check_skill(self, skill: str | None) -> None:
        """A named skill must exist on the card — refused before the
        send, naming the skills the agent actually offers."""
        if not skill:
            return
        if self.card is None:
            return
        if skill not in self.card.skill_ids():
            known = ", ".join(self.card.skill_ids()) or "(none)"
            raise A2AClientError(
                f"a2a[{self.config.name}]: agent {self.card.name!r} "
                f"has no skill {skill!r}; known skills: {known}")

    def send_message(self, text: str, *, context_id: str | None = None,
                     task_id: str | None = None,
                     skill: str | None = None,
                     accepted_output_modes: list[str] | None = None,
                     metadata: dict[str, Any] | None = None
                     ) -> dict[str, Any]:
        """Send a message. Returns the agent's reply — a Task (most
        calls) or a bare Message (a direct reply). A task in a
        non-terminal state is a RESULT, not a failure: poll it with
        get_task / wait_for_task, or answer it with a follow-up
        send_message naming its task_id."""
        if not isinstance(text, str) or not text.strip():
            raise A2AClientError("send_message requires non-empty text")
        self.ensure_connected()
        self._check_skill(skill)
        message = self._message(text, context_id=context_id,
                                task_id=task_id)
        params: dict[str, Any] = {"message": message}
        meta = dict(metadata or {})
        if skill:
            meta.setdefault("skill", skill)
        if meta:
            params["metadata"] = meta
        if accepted_output_modes:
            params["configuration"] = {
                "acceptedOutputModes": list(accepted_output_modes)}
        return self._rpc(self._methods()["send"], params)

    def send_message_stream(
            self, text: str, *, context_id: str | None = None,
            task_id: str | None = None, skill: str | None = None,
            accepted_output_modes: list[str] | None = None,
            metadata: dict[str, Any] | None = None
            ) -> Iterator[dict[str, Any]]:
        """Send a message and stream the updates. Yields each
        StreamResponse result (``task`` / ``statusUpdate`` /
        ``artifactUpdate`` / ``message``). The stream ends when the
        task reaches a terminal state. Refuses honestly when the
        card does not declare streaming."""
        if not isinstance(text, str) or not text.strip():
            raise A2AClientError(
                "send_message_stream requires non-empty text")
        self.ensure_connected()
        if self.card is not None and not self.card.streaming:
            raise A2AClientError(
                f"a2a[{self.config.name}]: agent {self.card.name!r} "
                f"does not declare streaming; use send_message and "
                f"poll with get_task instead")
        self._check_skill(skill)
        message = self._message(text, context_id=context_id,
                                task_id=task_id)
        params: dict[str, Any] = {"message": message}
        meta = dict(metadata or {})
        if skill:
            meta.setdefault("skill", skill)
        if meta:
            params["metadata"] = meta
        if accepted_output_modes:
            params["configuration"] = {
                "acceptedOutputModes": list(accepted_output_modes)}
        yield from self._rpc_stream(self._methods()["stream"], params)

    # ── tasks ──
    def get_task(self, task_id: str,
                 *, history_length: int | None = None
                 ) -> dict[str, Any]:
        if not isinstance(task_id, str) or not task_id:
            raise A2AClientError("get_task requires a task id")
        self.ensure_connected()
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = int(history_length)
        return self._rpc(self._methods()["get"], params)

    def list_tasks(self, context_id: str | None = None,
                   *, history_length: int | None = None
                   ) -> list[dict[str, Any]]:
        self.ensure_connected()
        params: dict[str, Any] = {}
        if context_id:
            params["contextId"] = context_id
        if history_length is not None:
            params["historyLength"] = int(history_length)
        result = self._rpc(self._methods()["list"], params)
        tasks = result.get("tasks") or []
        if not isinstance(tasks, list):
            raise A2AClientError(
                f"a2a[{self.config.name}]: tasks/list returned a "
                f"non-list tasks field")
        return [t for t in tasks if isinstance(t, dict)]

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        if not isinstance(task_id, str) or not task_id:
            raise A2AClientError("cancel_task requires a task id")
        self.ensure_connected()
        return self._rpc(self._methods()["cancel"], {"id": task_id})

    def wait_for_task(self, task_id: str, *, timeout_s: float = DEFAULT_WAIT_S,
                      poll_s: float = 2.0) -> dict[str, Any]:
        """Poll a task until it ends. Ends at a terminal state — or
        at input-required / auth-required, where the remote agent is
        waiting on US and further polling would be wrong. A timeout
        names the last state seen; the task is still alive on the
        remote side."""
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        last: dict[str, Any] = {}
        step = max(float(poll_s), _MIN_POLL_S)
        while True:
            last = self.get_task(task_id)
            state = task_state(last, self.protocol_version)
            if state in TERMINAL_STATES or state in WAITING_STATES:
                return last
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise A2ATimeoutError(
                    f"a2a[{self.config.name}]: task {task_id} still "
                    f"{state} after {timeout_s:g}s — the task is "
                    f"still alive on the remote agent")
            time.sleep(min(step, remaining))

    def send_and_wait(self, text: str, *, wait_s: float = DEFAULT_WAIT_S,
                      poll_s: float = 2.0, **kwargs: Any) -> dict[str, Any]:
        """One call: send, then follow the task to its end. Returns
        ``{"task", "text", "state"}`` — the plain-text answer is
        right there, no second call needed."""
        result = self.send_message(text, **kwargs)
        state = task_state(result, self.protocol_version)
        if "status" not in result or state in TERMINAL_STATES \
                or state in WAITING_STATES or wait_s <= 0:
            return {"task": result,
                    "text": task_text(result, self.protocol_version),
                    "state": state}
        task_id = str(result.get("id") or "")
        final = self.wait_for_task(task_id, timeout_s=wait_s,
                                   poll_s=poll_s)
        state = task_state(final, self.protocol_version)
        return {"task": final,
                "text": task_text(final, self.protocol_version),
                "state": state}


# ── manager ───────────────────────────────────────────────────────────────

class A2AClientManager:
    """The named-agent pool: registry, budgets, and connection cache.

    Built from settings (``from_settings``); a manager built from
    None carries the shipped defaults with the master switch off
    (and no agents at all — there are no shipped defaults). One
    client object per agent, connected lazily on first use; ``close()``
    shuts them all down.
    """

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self._registry = effective_a2a_agents(settings)
        self._enabled = _a2a_enabled(settings)
        self._default_timeout = (
            _a2a_default_timeout(settings) or DEFAULT_TIMEOUT_S)
        self._max_payload = (
            _a2a_max_payload(settings) or DEFAULT_MAX_PAYLOAD_BYTES)
        self._max_agents = (
            _a2a_max_agents(settings) or DEFAULT_MAX_AGENTS)
        self._clients: dict[str, A2AClient] = {}
        self._lock = threading.RLock()

    @classmethod
    def from_settings(cls, settings: Any) -> A2AClientManager:
        return cls(settings)

    # ── registry ──
    def agents(self) -> list[A2AAgentConfig]:
        return [self._registry[name] for name in sorted(self._registry)]

    def agent(self, name: str) -> A2AAgentConfig:
        cfg = self._registry.get(name)
        if cfg is None:
            known = ", ".join(sorted(self._registry))
            raise A2AClientError(
                f"unknown a2a agent {name!r}; known: {known}")
        return cfg

    def status(self) -> dict[str, Any]:
        """Master state + one line per agent (no connections made)."""
        out: dict[str, Any] = {
            "enabled": self._enabled,
            "max_agents": self._max_agents,
            "connected": len([c for c in self._clients.values()
                              if c.connected]),
            "agents": {},
        }
        for cfg in self.agents():
            client = self._clients.get(cfg.name)
            card: dict[str, Any] | None = None
            error = ""
            if client is not None and client.connected and client.card:
                card = {
                    "name": client.card.name,
                    "protocol_version": client.card.protocol_version,
                    "skills": client.card.skill_ids(),
                }
            out["agents"][cfg.name] = {
                "enabled": cfg.enabled,
                "url": cfg.url,
                "connected": bool(client is not None and client.connected),
                "card": card,
                "error": error,
                "notes": cfg.notes,
            }
        return out

    # ── connections ──
    def connect(self, name: str) -> A2AClient:
        """Connect (or return the cached connection) to one agent.

        Refuses honestly: master off, agent unknown, agent disabled,
        or the connection budget spent.
        """
        with self._lock:
            if not self._enabled:
                raise A2AClientError(
                    "a2a client is off (a2a.enabled is false in config); "
                    "enable it and the agent first")
            cfg = self.agent(name)
            if not cfg.enabled:
                raise A2AClientError(
                    f"a2a agent {name!r} is disabled in config; "
                    f"enable it under a2a.agents first")
            existing = self._clients.get(name)
            if existing is not None and existing.connected:
                return existing
            live = len([c for c in self._clients.values() if c.connected])
            if live >= self._max_agents:
                raise A2AClientError(
                    f"a2a connection budget spent ({self._max_agents} "
                    f"agents); close one first")
            client = A2AClient(
                cfg, timeout_s=self._default_timeout,
                max_payload_bytes=self._max_payload)
            self._clients[name] = client
        # Connect outside the manager lock (fetching the card is I/O).
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
    def discover(self, name: str) -> dict[str, Any]:
        """The agent's card as a plain summary: name, description,
        protocol version, endpoint, streaming, and its skills."""
        client = self.connect(name)
        card = client.card
        if card is None:  # pragma: no cover - connect sets the card
            raise A2AClientError(f"a2a[{name}]: no card after connect")
        return {
            "name": card.name,
            "description": card.description,
            "version": card.version,
            "protocol_version": card.protocol_version,
            "endpoint": card.endpoint_url,
            "streaming": card.streaming,
            "push_notifications": card.push_notifications,
            "skills": [{
                "id": str(s.get("id") or ""),
                "name": str(s.get("name") or ""),
                "description": str(s.get("description") or ""),
                "tags": [str(t) for t in (s.get("tags") or [])
                         if isinstance(t, str)],
                "examples": [str(e) for e in (s.get("examples") or [])
                             if isinstance(e, str)],
            } for s in card.skills],
        }

    def send(self, agent: str, text: str, **kwargs: Any) -> dict[str, Any]:
        return self.connect(agent).send_message(text, **kwargs)

    def send_and_wait(self, agent: str, text: str, *,
                      wait_s: float = DEFAULT_WAIT_S,
                      poll_s: float = 2.0,
                      **kwargs: Any) -> dict[str, Any]:
        return self.connect(agent).send_and_wait(
            text, wait_s=wait_s, poll_s=poll_s, **kwargs)

    def send_stream(self, agent: str, text: str,
                    **kwargs: Any) -> Iterator[dict[str, Any]]:
        yield from self.connect(agent).send_message_stream(text, **kwargs)

    def get_task(self, agent: str, task_id: str,
                 **kwargs: Any) -> dict[str, Any]:
        return self.connect(agent).get_task(task_id, **kwargs)

    def list_tasks(self, agent: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.connect(agent).list_tasks(**kwargs)

    def cancel_task(self, agent: str, task_id: str) -> dict[str, Any]:
        return self.connect(agent).cancel_task(task_id)

    def wait_for_task(self, agent: str, task_id: str, *,
                      timeout_s: float = DEFAULT_WAIT_S,
                      poll_s: float = 2.0) -> dict[str, Any]:
        return self.connect(agent).wait_for_task(
            task_id, timeout_s=timeout_s, poll_s=poll_s)


# ── tool registry wiring ──────────────────────────────────────────────────

def _manager(registry: Any) -> A2AClientManager:
    """The manager for this registry, built once from its settings."""
    manager = getattr(registry, "_a2a_client_manager", None)
    if manager is None:
        context = getattr(registry, "context", None)
        settings = getattr(context, "settings", None)
        manager = A2AClientManager.from_settings(settings)
        # Caching is best-effort.
        with contextlib.suppress(Exception):  # noqa: BLE001
            registry._a2a_client_manager = manager  # noqa: SLF001
    return manager


def register(registry: Any) -> None:
    """Register the A2A client tools on the spine.

    The brain talks to peer agents through these, the same way it
    calls her own tools. Capability is net.out: an A2A call leaves
    the box (an HTTP request to another agent). Nothing connects
    unless the operator enabled it in config.

    Framing (owner doctrine): A2A is peer work — delegation to a
    specialist that owns its own domain, collaboration across
    organizations, exchange. It is NEVER how Devon gets a skill she
    lacks: she builds every skill herself. A2A is the reach-out
    rung beside the skills she already has, not a replacement rung.
    """
    from ..core.policy import Capability

    @registry.register(
        "a2a_discover",
        description=(
            "Fetch the Agent Cards of Devon's peer agents (A2A): what "
            "each one is, what skills it offers, and how to reach it. "
            "No agent name lists every enabled agent. Use it to find "
            "the right peer before delegating — a specialist that owns "
            "its own domain, another organization's agent, a "
            "collaborator. A2A is peer work, never a substitute for a "
            "skill Devon builds herself."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "agent": "str (optional) — the agent name; empty lists "
                     "every enabled agent",
        },
    )
    def _a2a_discover(agent: str = "") -> dict[str, Any]:
        manager = _manager(registry)
        names = [agent] if agent else [
            cfg.name for cfg in manager.agents() if cfg.enabled]
        out: dict[str, Any] = {"ok": True, "agents": {}}
        errors: dict[str, str] = {}
        for name in names:
            try:
                out["agents"][name] = manager.discover(name)
            except A2AClientError as exc:
                errors[name] = str(exc)
        if errors:
            out["errors"] = errors
        return out

    @registry.register(
        "a2a_send",
        description=(
            "Delegate a task to a peer agent over A2A: send it a "
            "message and follow the task it opens. Use it when the "
            "work itself belongs to another agent — a specialist that "
            "owns its domain, a peer in another organization, a "
            "collaboration — not to borrow a skill: Devon builds "
            "every skill herself, and A2A is the reach-out rung "
            "beside them. wait_s > 0 follows the task to its end and "
            "returns the plain-text answer; wait_s 0 returns at once "
            "with the task id and state. A task that comes back "
            "input-required names what the peer is asking for — "
            "answer it with another a2a_send naming the task_id."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "agent": "str — the peer agent name (see a2a_discover)",
            "text": "str — the task, in plain words",
            "skill": "str (optional) — the skill id on that agent; "
                     "refused before the send when it does not exist",
            "context_id": "str (optional) — keeps a multi-turn "
                          "exchange together; omit to start fresh",
            "wait_s": "number (optional, default 0) — seconds to "
                      "follow the task to its end",
        },
    )
    def _a2a_send(agent: str, text: str, skill: str | None = None,
                  context_id: str | None = None,
                  wait_s: float = 0.0) -> dict[str, Any]:
        manager = _manager(registry)
        try:
            if wait_s and float(wait_s) > 0:
                out = manager.send_and_wait(
                    agent, text, skill=skill, context_id=context_id,
                    wait_s=float(wait_s))
                task = out.get("task") or {}
                return {"ok": True,
                        "task_id": str(task.get("id") or ""),
                        "context_id": str(task.get("contextId") or ""),
                        "state": out.get("state", ""),
                        "text": out.get("text", "")}
            result = manager.send(agent, text, skill=skill,
                                  context_id=context_id)
            return {"ok": True,
                    "task_id": str(result.get("id") or ""),
                    "context_id": str(result.get("contextId") or ""),
                    "state": task_state(result),
                    "text": task_text(result)}
        except A2AClientError as exc:
            return {"ok": False, "error": str(exc)}

    @registry.register(
        "a2a_task",
        description=(
            "The state of a task on a peer agent, with its answer as "
            "plain text when it has one. Use it to check on a task "
            "an earlier a2a_send left running."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "agent": "str — the peer agent name",
            "task_id": "str — the task id from an earlier a2a_send",
        },
    )
    def _a2a_task(agent: str, task_id: str) -> dict[str, Any]:
        manager = _manager(registry)
        try:
            task = manager.get_task(agent, task_id)
        except A2AClientError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True,
                "task_id": str(task.get("id") or ""),
                "context_id": str(task.get("contextId") or ""),
                "state": task_state(task),
                "text": task_text(task),
                "task": task}

    @registry.register(
        "a2a_cancel",
        description=(
            "Cancel a running task on a peer agent. The task ends "
            "canceled; the peer stops its work."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "agent": "str — the peer agent name",
            "task_id": "str — the task id to cancel",
        },
    )
    def _a2a_cancel(agent: str, task_id: str) -> dict[str, Any]:
        manager = _manager(registry)
        try:
            task = manager.cancel_task(agent, task_id)
        except A2AClientError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True,
                "task_id": str(task.get("id") or ""),
                "state": task_state(task)}

    @registry.register(
        "a2a_status",
        description=(
            "The A2A client state: whether the client is enabled, and "
            "per agent whether it is enabled and connected. No "
            "connections are made by this call."
        ),
        capability=Capability.NET_OUT,
        parameters={},
    )
    def _a2a_status() -> dict[str, Any]:
        return {"ok": True, "status": _manager(registry).status()}


# ── /a2a chat command ─────────────────────────────────────────────────────

_A2A_USAGE = (
    "/a2a — A2A client status (which peer agents are enabled and "
    "connected)\n"
    "/a2a list — the known agents and their state\n"
    "/a2a card <agent> — the agent's card: skills and endpoint\n"
    "/a2a send <agent> <text> [--wait N] [--skill id] — delegate a "
    "task to a peer agent\n"
    "/a2a task <agent> <task-id> — a task's state and answer\n"
    "/a2a cancel <agent> <task-id> — cancel a running task\n"
    "/a2a close [agent] — drop a connection (bare closes all)\n"
    "Agents are opt-in in config (a2a.enabled + a2a.agents.<name>.enabled); "
    "nothing connects until enabled. A2A is peer work — delegation and "
    "collaboration with other agents — never a substitute for a skill "
    "Devon builds herself."
)

#: Chat replies cap tool text at this many chars, with an explicit
#: truncation notice (never truncate silently).
_A2A_CHAT_TEXT_CAP = 4000


def control_a2a(tail: str, settings: Any = None) -> str:
    """The /a2a command. Owner-only (CONTROL_COMMANDS, never public).

    /a2a — status · /a2a list — agents · /a2a card <agent> — the
    card · /a2a send <agent> <text> [--wait N] [--skill id] — one
    delegation · /a2a task <agent> <task-id> — state + answer ·
    /a2a cancel <agent> <task-id> — cancel · /a2a close [agent] —
    drop connections.
    """
    manager = A2AClientManager.from_settings(settings)
    parts = (tail or "").split()
    verb = parts[0].lower() if parts else "status"

    if verb in ("status", ""):
        st = manager.status()
        lines = [f"A2A client: {'on' if st['enabled'] else 'OFF'} "
                 f"(a2a.enabled in config) — "
                 f"{st['connected']}/{st['max_agents']} connected"]
        for name, info in st["agents"].items():
            state = ("connected" if info["connected"]
                     else "enabled" if info["enabled"] else "disabled")
            card = ""
            if info["card"]:
                card = (f", {info['card']['name']} "
                        f"(protocol {info['card']['protocol_version']}, "
                        f"{len(info['card']['skills'])} skills)")
            notes = f" ({info['notes']})" if info["notes"] else ""
            lines.append(f"· {name}: {state} — {info['url']}{card}{notes}")
        if not st["enabled"]:
            lines.append("Enable it in config (a2a.enabled = true) and "
                         "enable an agent under a2a.agents to connect.")
        return "\n".join(lines)

    if verb == "list":
        lines = ["A2A agents:"]
        for cfg in manager.agents():
            state = "enabled" if cfg.enabled else "disabled"
            lines.append(f"· {cfg.name}: {state} — {cfg.url}")
            if cfg.notes:
                lines.append(f"  {cfg.notes}")
        return "\n".join(lines)

    if verb == "card":
        if len(parts) < 2:
            return "Usage: /a2a card <agent>\n" + _A2A_USAGE
        name = parts[1]
        try:
            card = manager.discover(name)
        except A2AClientError as exc:
            return f"a2a card failed: {exc}"
        lines = [f"{card['name']} — {card['description']}",
                 f"protocol {card['protocol_version']} · "
                 f"{card['endpoint']} · "
                 f"streaming: {'yes' if card['streaming'] else 'no'}"]
        for skill in card["skills"]:
            lines.append(f"· {skill['id']}: {skill['name']} — "
                         f"{skill['description']}")
            if skill["examples"]:
                lines.append(f"  e.g. {skill['examples'][0]}")
        return "\n".join(lines)

    if verb == "send":
        if len(parts) < 3:
            return "Usage: /a2a send <agent> <text>\n" + _A2A_USAGE
        agent = parts[1]
        rest = parts[2:]
        wait_s = 0.0
        skill: str | None = None
        text_parts: list[str] = []
        i = 0
        while i < len(rest):
            if rest[i] == "--wait" and i + 1 < len(rest):
                try:
                    wait_s = float(rest[i + 1])
                except ValueError:
                    return (f"Bad --wait value {rest[i + 1]!r}.\n"
                            + _A2A_USAGE)
                i += 2
            elif rest[i] == "--skill" and i + 1 < len(rest):
                skill = rest[i + 1]
                i += 2
            else:
                text_parts.append(rest[i])
                i += 1
        text = " ".join(text_parts).strip()
        if not text:
            return "Usage: /a2a send <agent> <text>\n" + _A2A_USAGE
        try:
            if wait_s > 0:
                out = manager.send_and_wait(agent, text, skill=skill,
                                            wait_s=wait_s)
                state = out["state"]
                answer = out["text"]
                task_id = str((out.get("task") or {}).get("id") or "")
            else:
                result = manager.send(agent, text, skill=skill)
                state = task_state(result)
                answer = task_text(result)
                task_id = str(result.get("id") or "")
        except A2AClientError as exc:
            return f"a2a send failed: {exc}"
        head = f"task {task_id} — {state}"
        if not answer:
            return head
        if len(answer) > _A2A_CHAT_TEXT_CAP:
            answer = (answer[:_A2A_CHAT_TEXT_CAP] +
                      f"\n…(truncated, {len(answer)} chars total)")
        return head + "\n" + answer

    if verb == "task":
        if len(parts) < 3:
            return "Usage: /a2a task <agent> <task-id>\n" + _A2A_USAGE
        agent, task_id = parts[1], parts[2]
        try:
            task = manager.get_task(agent, task_id)
        except A2AClientError as exc:
            return f"a2a task failed: {exc}"
        state = task_state(task)
        answer = task_text(task)
        head = f"task {task_id} — {state}"
        if not answer:
            return head
        if len(answer) > _A2A_CHAT_TEXT_CAP:
            answer = (answer[:_A2A_CHAT_TEXT_CAP] +
                      f"\n…(truncated, {len(answer)} chars total)")
        return head + "\n" + answer

    if verb == "cancel":
        if len(parts) < 3:
            return "Usage: /a2a cancel <agent> <task-id>\n" + _A2A_USAGE
        agent, task_id = parts[1], parts[2]
        try:
            task = manager.cancel_task(agent, task_id)
        except A2AClientError as exc:
            return f"a2a cancel failed: {exc}"
        return f"task {task_id} — {task_state(task)}"

    if verb == "close":
        name = parts[1] if len(parts) > 1 else None
        manager.close(name)
        return (f"Closed the {name} connection."
                if name else "Closed all A2A connections.")

    return f"Unknown /a2a verb {verb!r}.\n" + _A2A_USAGE
