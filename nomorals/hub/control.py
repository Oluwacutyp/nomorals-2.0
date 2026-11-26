"""Devon's control plane: one discovery/routing surface over the organs
that already exist.

F3 (305-agent survey): Agent Platform, Agent Trust Stack MCP Server,
AgentField, CerebrumKit, covey, OpenAgents, SandBase Harness, Synapse
Messenger, Cortex, CoWorker Protocol, HOL Standards SDK, Unified AI
System. The gap in every one of those rows was the same: the pieces exist
(hub, api, mesh, relay, MCP client, A2A client, skills, delegation,
jobs, audit) but there is no surface that lists what Devon actually has,
routes a request to the right agent/tool, governs the MCP servers, turns
an OpenAPI description into MCP tools, or serves skills — and no
cryptographic record of what the plane itself did.

This module is that surface. It composes live state only — tools from the
live registry (never a hardcoded list), MCP servers from the effective
config, skills from the live SkillRegistry, roles from the delegation
roster, nodes from the mesh NodeRegistry — and it routes per request.
There is no centre and no shared queue here: every call is one request
in, one ranked decision out, executed through the spine that already
owns the work (registry.call, the MCP manager, SkillRunner,
DelegationManager). The relay is deliberately NOT a routing target: it
is a Discord<->GitHub ticket relay, not a generic agent relay.

Provenance (the Agent Trust Stack piece) is a hash-chained
``control_receipts`` table: one receipt per routing decision and per
governed call, with the arguments stored as a SHA-256 digest only —
never raw, they may carry secrets — and each receipt chained to the
previous hash. ``verify_chain`` recomputes the chain; a tampered row
fails verification naming the receipt. Identity is the authenticated
principal recorded on the receipt; no signing keys are invented here —
key governance is F4's work, not F3's.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..core.ids import new_id
from ..core.policy import Capability, CapabilitySet

# Capabilities the plane itself enforces. Reads need DB_READ (like the
# /tools route); routing to a delegation run needs AGENT_SPAWN (like
# /agents/run); anything that calls out over the network needs NET_OUT
# (like the mcp tool).
NET_OUT = Capability.NET_OUT
AGENT_SPAWN = Capability.AGENT_SPAWN
DB_READ = Capability.DB_READ

CONTROL_RECEIPTS_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS control_receipts (
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL DEFAULT 0,
  actor TEXT NOT NULL DEFAULT '',
  action TEXT NOT NULL DEFAULT '',
  target TEXT NOT NULL DEFAULT '',
  args_digest TEXT NOT NULL DEFAULT '',
  result TEXT NOT NULL DEFAULT '',
  ok INTEGER NOT NULL DEFAULT 1,
  prev_hash TEXT NOT NULL DEFAULT '',
  hash TEXT NOT NULL DEFAULT ''
);
"""

CONTROL_RECEIPTS_DDL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_control_receipts_created
  ON control_receipts(created_at);
"""

# HTTP methods an OpenAPI document may carry per path.
_OPENAPI_METHODS = ("get", "put", "post", "delete", "patch", "head", "options")

# Token-overlap weights, identical to ToolRegistry.ranked_listing: a
# name hit weighs most, a description hit least, tokens shorter than
# three letters never count. The router reuses the registry's rule
# over structured candidates instead of inventing a second one.
_NAME_EXACT = 3.0
_NAME_PARTIAL = 1.5
_DESC_HIT = 1.0
_MIN_TOKEN = 3


class ControlError(Exception):
    """The control plane refused, or could not do, the requested thing.

    Refusals name the reason (unknown/disabled/unlisted/no-grant) so a
    caller can act on them instead of guessing.
    """


def _tokens(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in str(text or "").replace("_", " ").split()
        if len(tok) >= _MIN_TOKEN
    ]


def _score(query_tokens: list[str], name: str, description: str) -> float:
    """Token-overlap score: the registry's ranked_listing rule."""
    name_toks = set(str(name or "").replace("_", " ").lower().split())
    desc_toks = set(str(description or "").lower().split())
    score = 0.0
    for tok in query_tokens:
        if tok in name_toks:
            score += _NAME_EXACT
        elif any(tok in nt or nt in tok for nt in name_toks if len(nt) >= _MIN_TOKEN):
            score += _NAME_PARTIAL
        if tok in desc_toks:
            score += _DESC_HIT
    return score


def _granted(grant: Any, capability: str | None) -> bool:
    """Whether ``grant`` allows ``capability``.

    ``grant`` is None (unrestricted, e.g. the owner on the CLI), a
    CapabilitySet, or an iterable of capability strings. A tool with
    no declared capability needs no grant.
    """
    if not capability:
        return True
    if grant is None:
        return True
    grants = getattr(grant, "grants", None)
    if callable(grants):
        return bool(grants(capability))
    patterns = set(grant)
    if "*" in patterns or capability in patterns:
        return True
    return any(fnmatch.fnmatchcase(capability, pat) for pat in patterns)


def _grant_set(grant: Any) -> CapabilitySet | None:
    """Normalise a grant for registry.call / SkillRunner."""
    if grant is None:
        return None
    if isinstance(grant, CapabilitySet):
        return grant
    return CapabilitySet.of(*[str(cap) for cap in grant])


def _digest_args(args: Any) -> str:
    """SHA-256 digest of canonical JSON — never the raw arguments."""
    try:
        canonical = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = repr(args)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _receipt_hash(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tool_name_from_operation(method: str, path: str, operation: dict[str, Any]) -> str:
    operation_id = str(operation.get("operationId") or "").strip()
    if operation_id and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", operation_id):
        return operation_id
    raw = method.lower() + "_" + path
    name = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "operation"


def _openapi_param_schema(param: dict[str, Any]) -> dict[str, Any]:
    schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
    out: dict[str, Any] = {"type": str(schema.get("type") or "string")}
    if schema.get("description"):
        out["description"] = str(schema["description"])
    if "enum" in schema:
        out["enum"] = schema["enum"]
    if schema.get("default") is not None:
        out["default"] = schema["default"]
    return out


def openapi_to_mcp(doc: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Convert an OpenAPI 3.x document into MCP tool definitions.

    ``doc`` is a dict, a JSON string, or a YAML string (YAML only when
    PyYAML imports — it is a config extra, so a YAML document without it
    fails closed naming the extra). Returns ``(defs, operations)``:
    ``defs`` are MCP-shaped tool definitions (name / description /
    inputSchema) and ``operations`` maps each tool name to the operation
    needed to call it (method, path, server URL, parameters, request
    body, security). Swagger 2.0 documents are refused by name.
    """
    if isinstance(doc, str):
        text = doc.strip()
        if text.startswith("{"):
            try:
                doc = json.loads(text)
            except ValueError as exc:
                raise ControlError(f"OpenAPI document is not valid JSON: {exc}") from exc
        else:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ControlError(
                    "OpenAPI document looks like YAML but PyYAML is not "
                    "installed (config extra 'yaml'); pass a JSON document "
                    "or a parsed dict instead"
                ) from exc
            doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ControlError("OpenAPI document must be an object")
    version = str(doc.get("openapi") or "")
    if not version.startswith("3."):
        if doc.get("swagger"):
            raise ControlError(
                "Swagger 2.0 documents are not supported; convert to "
                "OpenAPI 3.x first"
            )
        raise ControlError("not an OpenAPI 3.x document (no 'openapi: 3.x' field)")
    paths = doc.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ControlError("OpenAPI document has no paths")
    servers = doc.get("servers") or []
    server_url = ""
    if isinstance(servers, list) and servers and isinstance(servers[0], dict):
        server_url = str(servers[0].get("url") or "")
    components = doc.get("components") if isinstance(doc.get("components"), dict) else {}
    schemes = components.get("securitySchemes") if isinstance(components.get("securitySchemes"), dict) else {}
    doc_security = doc.get("security") if isinstance(doc.get("security"), list) else []
    defs: list[dict[str, Any]] = []
    operations: dict[str, dict[str, Any]] = {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method in _OPENAPI_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            name = _tool_name_from_operation(method, path, operation)
            if name in operations:
                name = name + "_" + method.lower()
            description = (
                str(operation.get("summary") or "")
                or str(operation.get("description") or "")
                or (method.upper() + " " + path)
            )
            properties: dict[str, Any] = {}
            required: list[str] = []
            params: list[dict[str, Any]] = []
            for param in operation.get("parameters") or []:
                if not isinstance(param, dict) or not param.get("name"):
                    continue
                pname = str(param["name"])
                params.append(param)
                properties[pname] = _openapi_param_schema(param)
                if param.get("required"):
                    required.append(pname)
            body_schema: dict[str, Any] | None = None
            request_body = operation.get("requestBody")
            if isinstance(request_body, dict):
                content = request_body.get("content") or {}
                if isinstance(content, dict):
                    json_content = content.get("application/json") or {}
                    if isinstance(json_content, dict):
                        body_schema = json_content.get("schema") if isinstance(
                            json_content.get("schema"), dict) else None
            if body_schema:
                body_props = body_schema.get("properties") or {}
                if isinstance(body_props, dict):
                    for pname, pschema in body_props.items():
                        if pname not in properties and isinstance(pschema, dict):
                            entry: dict[str, Any] = {
                                "type": str(pschema.get("type") or "string")}
                            if pschema.get("description"):
                                entry["description"] = str(pschema["description"])
                            properties[pname] = entry
                for pname in body_schema.get("required") or []:
                    if pname not in required:
                        required.append(str(pname))
            security = operation.get("security", doc_security)
            defs.append({
                "name": name,
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(set(required)),
                    "additionalProperties": False,
                },
            })
            operations[name] = {
                "method": method.upper(),
                "path": path,
                "server_url": server_url,
                "parameters": params,
                "request_body": bool(body_schema),
                "security": security if isinstance(security, list) else [],
                "schemes": schemes if isinstance(schemes, dict) else {},
            }
    if not defs:
        raise ControlError("OpenAPI document defines no operations")
    return defs, operations


class ControlPlane:
    """One discovery/routing surface over Devon's existing organs.

    Constructed from a runtime context, or directly with the pieces a
    caller has (registry / settings / db). Anything absent is reported
    as unavailable by the section that needs it — the plane never
    invents an agent, a server, or a skill it cannot see.
    """

    def __init__(self, *, registry: Any = None, settings: Any = None,
                 db: Any = None, context: Any = None,
                 mcp_manager: Any = None, skill_registry: Any = None,
                 delegation_manager: Any = None, mesh_registry: Any = None,
                 job_registry: Any = None, actor: str = "control") -> None:
        if context is not None:
            if registry is None:
                # The runtime/API context carries the registry as
                # ``.tools`` (ToolRegistry); ``.registry`` is the
                # alternate name some callers use.
                registry = getattr(context, "registry", None) or getattr(
                    context, "tools", None)
            if settings is None:
                settings = getattr(context, "settings", None)
            if db is None:
                db = getattr(context, "db", None)
        self.registry = registry
        self.settings = settings
        self.db = db
        self.actor = actor
        self._mcp_manager = mcp_manager
        self._skill_registry = skill_registry
        self._delegation_manager = delegation_manager
        self._mesh_registry = mesh_registry
        self._job_registry = job_registry
        self._receipts_checked = False

    @classmethod
    def from_context(cls, context: Any, *, actor: str = "control") -> ControlPlane:
        return cls(context=context, actor=actor)

    # ── lazy organ access ──────────────────────────────────────────

    def _mcp(self) -> Any:
        if self._mcp_manager is not None:
            return self._mcp_manager
        # Reuse the registry's cached manager when it has one (the
        # mcp tool caches exactly the same way) so the gateway never
        # spawns a second copy of a stdio server.
        if self.registry is not None:
            manager = getattr(self.registry, "_mcp_client_manager", None)
            if manager is not None:
                self._mcp_manager = manager
                return manager
        from ..tools.mcp_client import MCPClientManager

        manager = MCPClientManager.from_settings(self.settings)
        self._mcp_manager = manager
        if self.registry is not None:
            with contextlib.suppress(AttributeError):
                self.registry._mcp_client_manager = manager  # noqa: SLF001
        return manager

    def _skills(self) -> Any:
        if self._skill_registry is not None:
            return self._skill_registry
        if self.db is None:
            return None
        from ..skills.registry import SkillRegistry

        self._skill_registry = SkillRegistry(self.db)
        return self._skill_registry

    def _delegation(self) -> Any:
        if self._delegation_manager is not None:
            return self._delegation_manager
        if self.db is None:
            return None
        from types import SimpleNamespace

        from ..agents.delegation_manager import DelegationManager

        self._delegation_manager = DelegationManager.from_context(
            SimpleNamespace(db=self.db))
        return self._delegation_manager

    def _mesh(self) -> Any:
        if self._mesh_registry is not None:
            return self._mesh_registry
        if self.db is None:
            return None
        from ..mesh.node import NodeRegistry

        self._mesh_registry = NodeRegistry(self.db)
        return self._mesh_registry

    def _jobs(self) -> Any:
        if self._job_registry is not None:
            return self._job_registry
        from ..core.jobs import shared_registry

        self._job_registry = shared_registry()
        return self._job_registry

    def _roster(self) -> dict[str, Any]:
        from ..agents.subagents import ROSTER

        return dict(ROSTER)

    # ── provenance ─────────────────────────────────────────────────

    def _ensure_receipts(self) -> None:
        if self._receipts_checked:
            return
        if self.db is None:
            raise ControlError(
                "provenance receipts need the database, which this "
                "context doesn't carry")
        try:
            # Database.execute takes one statement at a time — table
            # and index are separate executes, like every other organ.
            self.db.execute(CONTROL_RECEIPTS_DDL_TABLE)
            self.db.execute(CONTROL_RECEIPTS_DDL_INDEX)
        except Exception as exc:  # noqa: BLE001
            raise ControlError(
                f"could not create the control_receipts table: {exc}") from exc
        self._receipts_checked = True

    def _record(self, *, actor: str, action: str, target: str,
                args: Any, result: str, ok: bool = True) -> dict[str, Any] | None:
        """Append one hash-chained receipt. None when there is no db."""
        if self.db is None:
            return None
        self._ensure_receipts()
        prev_row = self.db.query_one(
            "SELECT hash FROM control_receipts ORDER BY rowid DESC LIMIT 1")
        prev_hash = str(prev_row["hash"]) if prev_row else ""
        record = {
            "id": new_id(),
            "created_at": time.time(),
            "actor": actor,
            "action": action,
            "target": target,
            "args_digest": _digest_args(args),
            "result": result,
            "ok": 1 if ok else 0,
            "prev_hash": prev_hash,
        }
        record["hash"] = _receipt_hash(record)
        self.db.execute(
            "INSERT INTO control_receipts "
            "(id, created_at, actor, action, target, args_digest, result, "
            " ok, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record["id"], record["created_at"], record["actor"],
             record["action"], record["target"], record["args_digest"],
             record["result"], record["ok"], record["prev_hash"],
             record["hash"]))
        return dict(record)

    def list_receipts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            self._ensure_receipts()
            rows = self.db.query(
                "SELECT * FROM control_receipts ORDER BY rowid DESC LIMIT ?",
                (int(limit),))
        except Exception:  # noqa: BLE001 - a read surface never crashes
            return []
        out = []
        for row in rows:
            out.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "actor": row["actor"],
                "action": row["action"],
                "target": row["target"],
                "args_digest": row["args_digest"],
                "result": row["result"],
                "ok": bool(row["ok"]),
                "prev_hash": row["prev_hash"],
                "hash": row["hash"],
            })
        return out

    def verify_chain(self, *, limit: int = 1000) -> dict[str, Any]:
        """Recompute the receipt chain. A tampered row fails by id."""
        if self.db is None:
            return {"ok": False, "checked": 0,
                    "error": "no database, nothing to verify"}
        self._ensure_receipts()
        rows = self.db.query(
            "SELECT * FROM control_receipts ORDER BY rowid ASC LIMIT ?",
            (int(limit),))
        prev_hash = ""
        broken: list[str] = []
        for row in rows:
            record = {
                "id": row["id"],
                "created_at": row["created_at"],
                "actor": row["actor"],
                "action": row["action"],
                "target": row["target"],
                "args_digest": row["args_digest"],
                "result": row["result"],
                "ok": row["ok"],
                "prev_hash": row["prev_hash"],
            }
            if row["prev_hash"] != prev_hash or _receipt_hash(record) != str(row["hash"]):
                broken.append(str(row["id"]))
            prev_hash = str(row["hash"])
        return {"ok": not broken, "checked": len(rows), "broken": broken}

    # ── discovery ──────────────────────────────────────────────────

    def discover(self, *, grant: Any = None) -> dict[str, Any]:
        """Everything the plane can see, from live state only.

        Each section carries ``available``; an unavailable section
        names why instead of inventing entries.
        """
        out: dict[str, Any] = {"available": True, "sections": {}}
        sections = out["sections"]

        # Tools: the live registry, filtered to the caller's grant.
        try:
            if self.registry is None:
                raise ControlError("no tool registry in this context")
            schemas = self.registry.schemas(capabilities=_grant_set(grant))
            sections["tools"] = {
                "available": True,
                "count": len(schemas),
                "tools": [
                    {"name": s.get("name"), "description": s.get("description"),
                     "capability": s.get("capability"),
                     "parameters": s.get("parameters")}
                    for s in schemas
                ],
            }
        except Exception as exc:  # noqa: BLE001
            sections["tools"] = {"available": False, "reason": str(exc),
                                 "count": 0, "tools": []}

        # MCP servers: effective config + live connection state (the
        # status call makes no connections — discovery never spawns a
        # server process).
        try:
            from ..tools.mcp_client import effective_mcp_servers

            servers = list(effective_mcp_servers(self.settings).values())
            try:
                status = self._mcp().status()
                live = status.get("servers") or {}
            except Exception:  # noqa: BLE001
                live = {}
            entries = []
            for cfg in servers:
                entry = {
                    "name": cfg.name,
                    "enabled": bool(cfg.enabled),
                    "transport": cfg.transport,
                    "url": cfg.url or None,
                    "env": list(cfg.env or []),
                    "notes": cfg.notes or "",
                    "connected": False,
                    "tools": None,
                }
                state = live.get(cfg.name) or {}
                if state:
                    entry["connected"] = bool(state.get("connected"))
                    entry["tools"] = state.get("tools")
                    if state.get("error"):
                        entry["error"] = state["error"]
                entries.append(entry)
            sections["mcp_servers"] = {
                "available": True, "count": len(entries), "servers": entries}
        except Exception as exc:  # noqa: BLE001
            sections["mcp_servers"] = {"available": False, "reason": str(exc),
                                       "count": 0, "servers": []}

        # A2A agents: effective config (empty by design when nothing
        # is configured — nothing connects by default).
        try:
            from ..tools.a2a_client import effective_a2a_agents

            agents = effective_a2a_agents(self.settings)
            sections["a2a_agents"] = {
                "available": True,
                "count": len(agents),
                "agents": [
                    {"name": cfg.name, "url": cfg.url,
                     "enabled": bool(cfg.enabled),
                     "api_key_env": cfg.api_key_env or None,
                     "api_key_present": bool(cfg.api_key_env)
                     and cfg.api_key_env in os.environ,
                     "notes": cfg.notes or ""}
                    for cfg in agents.values()
                ],
            }
        except Exception as exc:  # noqa: BLE001
            sections["a2a_agents"] = {"available": False, "reason": str(exc),
                                      "count": 0, "agents": []}

        # Skills: the live registry.
        try:
            skills = self._skills()
            if skills is None:
                raise ControlError("no database, no skill registry")
            entries = skills.list()
            sections["skills"] = {
                "available": True, "count": len(entries), "skills": entries}
        except Exception as exc:  # noqa: BLE001
            sections["skills"] = {"available": False, "reason": str(exc),
                                  "count": 0, "skills": []}

        # Delegation roles + active runs.
        try:
            roster = self._roster()
            roles = []
            for role, cls in sorted(roster.items()):
                doc = (getattr(cls, "__doc__", "") or "").strip().splitlines()
                roles.append({
                    "role": role,
                    "description": doc[0].strip() if doc else "",
                })
            runs: list[dict[str, Any]] = []
            manager = self._delegation()
            if manager is not None:
                runs = [run.to_dict() for run in manager.list_runs(limit=20)]
            sections["delegation"] = {
                "available": True, "roles": roles, "runs": runs,
                "active_runs": len([r for r in runs
                                    if r.get("status") in ("running", "queued", "pending")]),
            }
        except Exception as exc:  # noqa: BLE001
            sections["delegation"] = {"available": False, "reason": str(exc),
                                      "roles": [], "runs": []}

        # Mesh nodes: the live NodeRegistry.
        try:
            mesh = self._mesh()
            if mesh is None:
                raise ControlError("no database, no mesh node registry")
            nodes = [node.to_dict() for node in mesh.list_all()]
            sections["mesh_nodes"] = {
                "available": True, "count": len(nodes), "nodes": nodes}
        except Exception as exc:  # noqa: BLE001
            sections["mesh_nodes"] = {"available": False, "reason": str(exc),
                                      "count": 0, "nodes": []}

        # Jobs: the shared job registry (list() already returns
        # snapshots — dicts, newest work visible, nothing hidden).
        try:
            jobs = self._jobs().list()[:20]
            sections["jobs"] = {
                "available": True, "count": len(jobs),
                "jobs": jobs,
            }
        except Exception as exc:  # noqa: BLE001
            sections["jobs"] = {"available": False, "reason": str(exc),
                                "count": 0, "jobs": []}

        # Approvals pending: held proactive proposals.
        try:
            if self.db is None:
                raise ControlError("no database, no approvals ledger")
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE status = 'pending'")
            sections["approvals_pending"] = {
                "available": True, "count": int(row["n"]) if row else 0}
        except Exception as exc:  # noqa: BLE001
            sections["approvals_pending"] = {"available": False,
                                             "reason": str(exc), "count": 0}

        sections["receipts"] = {
            "available": self.db is not None,
            "count": len(self.list_receipts(limit=1)) if self.db is not None else 0,
            "reason": "" if self.db is not None else "no database",
        }
        return out

    # ── describe: the per-agent platform view ──────────────────────

    def describe(self, kind: str, name: str, *,
                 grant: Any = None) -> dict[str, Any]:
        """One agent's whole platform view in one place.

        ``kind`` is tool / mcp / a2a / skill / role / mesh_node / job.
        Credentials are env-var PRESENCE only — a value is never
        returned, never logged, never stored.
        """
        kind = str(kind or "").lower()
        name = str(name or "")
        if kind == "tool":
            if self.registry is None:
                raise ControlError("no tool registry in this context")
            schemas = self.registry.schemas(capabilities=_grant_set(grant))
            found = next((s for s in schemas if s.get("name") == name), None)
            if found is None:
                raise ControlError(f"no tool {name!r}")
            return {
                "kind": "tool", "name": name,
                "description": found.get("description"),
                "capability": found.get("capability"),
                "parameters": found.get("parameters"),
                "credentials": [],
                "backlog": [],
                "approvals_pending": self._approvals_count(),
                "run_logs": self._receipts_for(name),
            }
        if kind == "mcp":
            servers = self.gateway_servers()
            found = next((s for s in servers if s["name"] == name), None)
            if found is None:
                raise ControlError(f"no MCP server {name!r}")
            credentials = [
                {"env": var, "present": var in os.environ}
                for var in found.get("env") or []
            ]
            return {
                "kind": "mcp", "name": name,
                "description": found.get("notes") or "",
                "enabled": found.get("enabled"),
                "transport": found.get("transport"),
                "url": found.get("url"),
                "connected": found.get("connected"),
                "tools": found.get("tools"),
                "credentials": credentials,
                "backlog": [],
                "approvals_pending": self._approvals_count(),
                "run_logs": self._receipts_for("mcp:" + name),
            }
        if kind == "a2a":
            from ..tools.a2a_client import effective_a2a_agents

            agents = effective_a2a_agents(self.settings)
            found = agents.get(name)
            if found is None:
                raise ControlError(f"no A2A agent {name!r}")
            credentials = []
            if found.api_key_env:
                credentials.append({
                    "env": found.api_key_env,
                    "present": found.api_key_env in os.environ})
            return {
                "kind": "a2a", "name": name,
                "description": found.notes or "",
                "url": found.url, "enabled": bool(found.enabled),
                "credentials": credentials,
                "backlog": [],
                "approvals_pending": self._approvals_count(),
                "run_logs": self._receipts_for("a2a:" + name),
            }
        if kind == "skill":
            skills = self._skills()
            if skills is None:
                raise ControlError("no database, no skill registry")
            entry = skills.get(name)
            if entry is None:
                raise ControlError(f"no skill {name!r}")
            manifest = entry.manifest
            return {
                "kind": "skill", "name": name,
                "description": manifest.description,
                "version": entry.version,
                "active": bool(entry.active),
                "enabled": bool(entry.enabled),
                "owner": manifest.owner,
                "tags": list(manifest.tags),
                "tools": list(manifest.tools),
                "credentials": [],
                "backlog": [],
                "approvals_pending": self._approvals_count(),
                "run_logs": self._receipts_for("skill:" + name),
            }
        if kind == "role":
            roster = self._roster()
            cls = roster.get(name)
            if cls is None:
                raise ControlError(f"no delegation role {name!r}")
            doc = (getattr(cls, "__doc__", "") or "").strip().splitlines()
            backlog: list[dict[str, Any]] = []
            manager = self._delegation()
            if manager is not None:
                backlog = [run.to_dict() for run in manager.list_runs(limit=20)
                           if run.to_dict().get("role") == name]
            return {
                "kind": "role", "name": name,
                "description": doc[0].strip() if doc else "",
                "credentials": [],
                "backlog": backlog,
                "approvals_pending": self._approvals_count(),
                "run_logs": self._receipts_for("role:" + name),
            }
        if kind == "mesh_node":
            mesh = self._mesh()
            if mesh is None:
                raise ControlError("no database, no mesh node registry")
            node = mesh.get(name)
            if node is None:
                raise ControlError(f"no mesh node {name!r}")
            return {
                "kind": "mesh_node", "name": name,
                "description": node.describe(),
                "node": node.to_dict(),
                "credentials": [],
                "backlog": [],
                "approvals_pending": self._approvals_count(),
                "run_logs": self._receipts_for("mesh:" + name),
            }
        if kind == "job":
            jobs = self._jobs().list()
            found = next((j for j in jobs if j.get("id") == name), None)
            if found is None:
                raise ControlError(f"no job {name!r}")
            return {
                "kind": "job", "name": name,
                "description": found.get("label", ""),
                "job": found,
                "credentials": [],
                "backlog": [],
                "approvals_pending": self._approvals_count(),
                "run_logs": self._receipts_for("job:" + name),
            }
        raise ControlError(
            f"unknown describe kind {kind!r}; expected tool / mcp / a2a / "
            "skill / role / mesh_node / job")

    def _approvals_count(self) -> int:
        if self.db is None:
            return 0
        try:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM proactive_log WHERE status = 'pending'")
            return int(row["n"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    def _receipts_for(self, target: str, *, limit: int = 20) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            self._ensure_receipts()
            rows = self.db.query(
                "SELECT * FROM control_receipts WHERE target = ? "
                "ORDER BY rowid DESC LIMIT ?",
                (target, int(limit)))
        except Exception:  # noqa: BLE001
            return []
        return [{
            "id": row["id"], "created_at": row["created_at"],
            "actor": row["actor"], "action": row["action"],
            "result": row["result"], "ok": bool(row["ok"]),
            "hash": row["hash"],
        } for row in rows]

    # ── routing ────────────────────────────────────────────────────

    def _candidates(self, goal: str, *, grant: Any) -> list[dict[str, Any]]:
        """Ranked candidates for a goal, with a capability verdict each."""
        qtokens = _tokens(goal)
        candidates: list[dict[str, Any]] = []
        if self.registry is not None:
            for schema in self.registry.schemas():
                capability = schema.get("capability")
                candidates.append({
                    "kind": "tool",
                    "name": str(schema.get("name")),
                    "description": str(schema.get("description") or ""),
                    "capability": capability,
                    "granted": _granted(grant, capability),
                    "score": _score(qtokens, str(schema.get("name")),
                                    str(schema.get("description") or "")),
                })
        try:
            from ..tools.mcp_client import effective_mcp_servers

            for cfg in effective_mcp_servers(self.settings).values():
                candidates.append({
                    "kind": "mcp",
                    "name": cfg.name,
                    "description": cfg.notes or "",
                    "capability": NET_OUT,
                    "granted": _granted(grant, NET_OUT),
                    "score": _score(qtokens, cfg.name, cfg.notes or ""),
                    "server": cfg.name,
                })
        except Exception:  # noqa: BLE001
            pass
        skills = self._skills()
        if skills is not None:
            try:
                for entry in skills.list():
                    text = " ".join([entry.get("name", ""),
                                     entry.get("description", ""),
                                     " ".join(entry.get("tags") or [])])
                    candidates.append({
                        "kind": "skill",
                        "name": entry.get("name"),
                        "description": entry.get("description", ""),
                        "capability": None,
                        "granted": True,
                        "score": _score(qtokens, entry.get("name", ""), text),
                    })
            except Exception:  # noqa: BLE001
                pass
        try:
            for role, cls in sorted(self._roster().items()):
                doc = (getattr(cls, "__doc__", "") or "").strip().splitlines()
                desc = doc[0].strip() if doc else ""
                candidates.append({
                    "kind": "delegation",
                    "name": role,
                    "description": desc,
                    "capability": AGENT_SPAWN,
                    "granted": _granted(grant, AGENT_SPAWN),
                    "score": _score(qtokens, role, desc),
                })
        except Exception:  # noqa: BLE001
            pass
        candidates.sort(key=lambda c: (-c["score"], c["kind"], c["name"]))
        return candidates

    def route(self, goal: str, *, grant: Any = None, execute: bool = False,
              actor: str | None = None, args: dict[str, Any] | None = None,
              limit: int = 5) -> dict[str, Any]:
        """Rank candidates for a goal; optionally execute the top one.

        A denied top candidate is reported with the missing capability
        named — never silently skipped to the next. Execution goes
        through the spine that already owns the work, and both the
        decision and the execution get a provenance receipt.
        """
        goal = str(goal or "").strip()
        if not goal:
            raise ControlError("route needs a goal")
        who = actor or self.actor
        candidates = self._candidates(goal, grant=grant)[: int(limit)]
        self._record(actor=who, action="route.decide", target=goal,
                     args={"candidates": [c["name"] for c in candidates]},
                     result="ranked", ok=True)
        if not execute:
            return {"ok": True, "goal": goal, "executed": False,
                    "candidates": candidates}
        if not candidates:
            return {"ok": False, "goal": goal, "executed": False,
                    "error": "nothing to route to", "candidates": []}
        top = candidates[0]
        if not top["granted"]:
            self._record(actor=who, action="route.refused", target=top["name"],
                         args={"goal": goal, "kind": top["kind"]},
                         result="missing capability " + str(top["capability"]),
                         ok=False)
            return {"ok": False, "goal": goal, "executed": False,
                    "error": "top candidate " + top["kind"] + ":" + top["name"]
                    + " needs capability " + str(top["capability"]),
                    "candidates": candidates}
        try:
            result = self._execute(top, goal, grant=grant, actor=who,
                                   args=args or {})
        except ControlError as exc:
            self._record(actor=who, action="route.execute", target=top["name"],
                         args={"goal": goal, "kind": top["kind"]},
                         result=str(exc), ok=False)
            return {"ok": False, "goal": goal, "executed": False,
                    "error": str(exc), "candidates": candidates}
        except Exception as exc:  # noqa: BLE001
            self._record(actor=who, action="route.execute", target=top["name"],
                         args={"goal": goal, "kind": top["kind"]},
                         result=str(exc), ok=False)
            return {"ok": False, "goal": goal, "executed": False,
                    "error": str(exc), "candidates": candidates}
        self._record(actor=who, action="route.execute", target=top["name"],
                     args={"goal": goal, "kind": top["kind"]},
                     result="ok", ok=True)
        return {"ok": True, "goal": goal, "executed": True,
                "candidate": top, "result": result, "candidates": candidates}

    def _execute(self, candidate: dict[str, Any], goal: str, *,
                 grant: Any, actor: str, args: dict[str, Any]) -> Any:
        kind = candidate["kind"]
        name = candidate["name"]
        if kind == "tool":
            if self.registry is None:
                raise ControlError("no tool registry in this context")
            outcome = self.registry.call(
                name, actor=actor, capabilities=_grant_set(grant), **args)
            if outcome.ok:
                return outcome.value
            raise ControlError(str(outcome.error))
        if kind == "mcp":
            # The candidate is a server; pick its best tool by the same
            # scoring rule, then call it through the governed gateway.
            tools = self._mcp().list_tools(name)
            qtokens = _tokens(goal)
            best = None
            best_score = -1.0
            for tool in tools:
                tool_name = str(tool.get("name") or "")
                score = _score(qtokens, tool_name,
                               str(tool.get("description") or ""))
                if score > best_score:
                    best_score = score
                    best = tool_name
            if best is None:
                raise ControlError(f"MCP server {name!r} lists no tools")
            return self.gateway_call(name, best, args, grant=grant, actor=actor)
        if kind == "skill":
            from ..skills.runner import SkillRunner

            if self.registry is None:
                raise ControlError("no tool registry in this context")
            skills = self._skills()
            if skills is None:
                raise ControlError("no database, no skill registry")
            runner = SkillRunner(skills, self.registry, actor=actor,
                                 capabilities=_grant_set(grant))
            outcome = runner.run(name, dict(args) or {"goal": goal})
            if getattr(outcome, "ok", False):
                return getattr(outcome, "summary", None) or str(outcome)
            raise ControlError(str(getattr(outcome, "error", None)
                                   or "skill run failed"))
        if kind == "delegation":
            manager = self._delegation()
            if manager is None:
                raise ControlError("no database, no delegation manager")
            return manager.plan_and_spawn(goal)
        raise ControlError(f"cannot execute candidate kind {kind!r}")

    # ── governed MCP gateway ───────────────────────────────────────

    def gateway_servers(self) -> list[dict[str, Any]]:
        """MCP servers with live state. No connections are made."""
        from ..tools.mcp_client import effective_mcp_servers

        servers = list(effective_mcp_servers(self.settings).values())
        try:
            live = (self._mcp().status().get("servers") or {})
        except Exception:  # noqa: BLE001
            live = {}
        out = []
        for cfg in servers:
            state = live.get(cfg.name) or {}
            out.append({
                "name": cfg.name,
                "enabled": bool(cfg.enabled),
                "transport": cfg.transport,
                "url": cfg.url or None,
                "env": list(cfg.env or []),
                "notes": cfg.notes or "",
                "connected": bool(state.get("connected")),
                "tools": state.get("tools"),
                "error": state.get("error") or "",
            })
        return out

    def gateway_call(self, server: str, tool: str,
                     args: dict[str, Any] | None = None, *,
                     grant: Any = None, actor: str | None = None) -> dict[str, Any]:
        """One governed MCP call: server must exist AND be enabled, the
        tool must be listed by that server, the caller must hold
        net.out. Every call gets a provenance receipt."""
        server = str(server or "")
        tool = str(tool or "")
        who = actor or self.actor
        manager = self._mcp()
        try:
            cfg = manager.server(server)
        except Exception as exc:  # noqa: BLE001
            raise ControlError(f"unknown MCP server {server!r}: {exc}") from exc
        if not cfg.enabled:
            raise ControlError(
                f"MCP server {server!r} is disabled; enable it in config "
                "first (mcp servers are all disabled by default)")
        if not _granted(grant, NET_OUT):
            raise ControlError(
                "MCP calls need capability net.out, which this caller "
                "does not hold")
        try:
            tools = manager.list_tools(server)
        except Exception as exc:  # noqa: BLE001
            raise ControlError(
                f"could not list tools on MCP server {server!r}: {exc}") from exc
        names = [str(t.get("name") or "") for t in tools
                 if isinstance(t, dict)]
        if tool not in names:
            raise ControlError(
                f"MCP server {server!r} does not list tool {tool!r}")
        try:
            result = manager.call_tool(server, tool, dict(args or {}))
        except Exception as exc:  # noqa: BLE001
            self._record(actor=who, action="mcp.call",
                         target="mcp:" + server + ":" + tool,
                         args=args or {}, result=str(exc), ok=False)
            raise ControlError(
                f"MCP call {server!r}:{tool!r} failed: {exc}") from exc
        self._record(actor=who, action="mcp.call",
                     target="mcp:" + server + ":" + tool,
                     args=args or {},
                     result="isError" if result.get("isError") else "ok",
                     ok=not bool(result.get("isError")))
        return dict(result)

    # ── OpenAPI → MCP ──────────────────────────────────────────────

    def openapi_import(self, doc: Any) -> dict[str, Any]:
        """Convert an OpenAPI document into MCP tool definitions."""
        defs, operations = openapi_to_mcp(doc)
        return {"ok": True, "count": len(defs), "tools": defs,
                "servers": sorted({op["server_url"] for op in operations.values()
                                   if op.get("server_url")})}

    def call_openapi_tool(self, doc: Any, tool_name: str,
                          args: dict[str, Any] | None = None, *,
                          auth_env: str | None = None, grant: Any = None,
                          actor: str | None = None,
                          timeout_s: float = 30.0) -> dict[str, Any]:
        """Call one converted OpenAPI operation over HTTP.

        Auth comes from the env var named by ``auth_env`` — the value
        is read at call time, never from config, code, or the request
        body. When the operation declares security and no ``auth_env``
        is given, the call is refused naming the missing auth.
        """
        who = actor or self.actor
        if not _granted(grant, NET_OUT):
            raise ControlError(
                "OpenAPI calls need capability net.out, which this "
                "caller does not hold")
        _, operations = openapi_to_mcp(doc)
        operation = operations.get(str(tool_name))
        if operation is None:
            raise ControlError(
                f"no OpenAPI operation converts to tool {tool_name!r}")
        args = dict(args or {})
        security = operation.get("security") or []
        schemes = operation.get("schemes") or {}
        headers: dict[str, str] = {"Accept": "application/json"}
        if security:
            if not auth_env:
                names = []
                for requirement in security:
                    if isinstance(requirement, dict):
                        names.extend(sorted(requirement.keys()))
                raise ControlError(
                    "operation " + str(tool_name) + " requires auth ("
                    + ", ".join(names) + "); pass auth_env naming the "
                    "environment variable holding the credential")
            value = os.environ.get(str(auth_env))
            if not value:
                raise ControlError(
                    "auth env var " + str(auth_env) + " is not set")
            applied = False
            for requirement in security:
                if not isinstance(requirement, dict):
                    continue
                for scheme_name in requirement:
                    scheme = schemes.get(scheme_name) or {}
                    if scheme.get("type") == "apiKey":
                        header_name = str(scheme.get("name") or "Authorization")
                        where = str(scheme.get("in") or "header")
                        if where == "header":
                            headers[header_name] = value
                            applied = True
                    elif scheme.get("type") == "http":
                        headers["Authorization"] = "Bearer " + value
                        applied = True
            if not applied:
                headers["Authorization"] = "Bearer " + value
        server_url = str(operation.get("server_url") or "").rstrip("/")
        if not server_url:
            raise ControlError(
                "OpenAPI document names no server URL, so " + str(tool_name)
                + " has nowhere to call")
        path = str(operation["path"])
        query: dict[str, str] = {}
        body: dict[str, Any] = {}
        params = operation.get("parameters") or []
        by_name = {str(p.get("name")): p for p in params
                   if isinstance(p, dict) and p.get("name")}
        for key, value in args.items():
            param = by_name.get(str(key))
            if param is None:
                body[str(key)] = value
                continue
            where = str(param.get("in") or "query")
            if where == "path":
                path = path.replace(
                    "{" + str(key) + "}",
                    urllib.parse.quote(str(value), safe=""))
            elif where == "query":
                query[str(key)] = str(value)
            elif where == "header":
                headers[str(key)] = str(value)
            else:
                body[str(key)] = value
        for key, param in by_name.items():
            if param.get("required") and key not in args:
                raise ControlError(
                    "OpenAPI operation " + str(tool_name)
                    + " requires parameter " + key)
        url = server_url + path
        if query:
            url = url + "?" + urllib.parse.urlencode(query)
        data: bytes | None = None
        if operation.get("request_body") and body:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=data, headers=headers,
            method=str(operation["method"]))
        try:
            with urllib.request.urlopen(request, timeout=float(timeout_s)) as resp:
                raw = resp.read()
                content_type = resp.headers.get("Content-Type") or ""
                status = int(resp.status)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001
                detail = ""
            self._record(actor=who, action="openapi.call",
                         target="openapi:" + str(tool_name),
                         args=args, result="HTTP " + str(exc.code), ok=False)
            raise ControlError(
                "OpenAPI call " + str(tool_name) + " returned HTTP "
                + str(exc.code) + (": " + detail if detail else "")) from exc
        except Exception as exc:  # noqa: BLE001
            self._record(actor=who, action="openapi.call",
                         target="openapi:" + str(tool_name),
                         args=args, result=str(exc), ok=False)
            raise ControlError(
                "OpenAPI call " + str(tool_name) + " failed: " + str(exc)) from exc
        if "application/json" in content_type:
            try:
                payload: Any = json.loads(raw.decode("utf-8"))
            except ValueError:
                payload = {"text": raw.decode("utf-8", "replace")}
        else:
            payload = {"text": raw.decode("utf-8", "replace")}
        self._record(actor=who, action="openapi.call",
                     target="openapi:" + str(tool_name),
                     args=args, result="HTTP " + str(status), ok=True)
        return {"ok": True, "status": status, "result": payload}

    # ── skill serving ──────────────────────────────────────────────

    def serve_skill(self, name: str, *,
                    version: str | None = None) -> dict[str, Any]:
        """Serve one installed skill over the control surface: its
        manifest, tool chain, and portable package export."""
        skills = self._skills()
        if skills is None:
            raise ControlError("no database, no skill registry")
        installed = skills.get(str(name), version)
        if installed is None:
            raise ControlError(f"no skill {name!r} installed")
        manifest = installed.manifest
        package = skills.export_skill(str(name), version)
        return {
            "ok": True,
            "name": installed.name,
            "version": installed.version,
            "active": bool(installed.active),
            "enabled": bool(installed.enabled),
            "description": manifest.description,
            "owner": manifest.owner,
            "tags": list(manifest.tags),
            "tools": list(manifest.tools),
            "package": package,
        }
