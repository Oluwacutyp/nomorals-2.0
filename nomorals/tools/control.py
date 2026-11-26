"""The control-plane tool: discovery, routing, the governed MCP gateway,
OpenAPI→MCP, and skill serving over the hub/control.py engine.

One tool, ``control``, with an ``action`` parameter — the brain reaches
every control-plane capability through it the same way it reaches any
other tool. Capability is agent.spawn: discovering what Devon has is
read-only, but routing can execute work and delegate to subagents, so
the tool as a whole needs the spawn grant. The chat command /control
(owner-only) is :func:`control_chat`.
"""

from __future__ import annotations

import json
from typing import Any

from ..hub.control import ControlError, ControlPlane


def _plane(registry: Any) -> ControlPlane:
    context = getattr(registry, "context", None)
    settings = getattr(context, "settings", None)
    db = getattr(context, "db", None)
    return ControlPlane(registry=registry, settings=settings, db=db)


def register(registry: Any) -> None:
    """Register the control-plane tool on the spine."""
    from ..core.policy import Capability

    @registry.register(
        "control",
        description=(
            "Devon's control plane: one discovery/routing surface over "
            "the organs that already exist. action=discover lists the "
            "agents, tools, MCP servers, A2A agents, skills, delegation "
            "roles, mesh nodes, jobs and approvals Devon actually has "
            "(live state, never a hardcoded list); action=describe "
            "shows one agent's whole platform view (credentials "
            "presence, backlog, approvals, run logs); action=route "
            "ranks candidates for a goal and, with execute=true, runs "
            "the top allowed one; action=mcp lists the governed MCP "
            "gateway servers and action=mcp_call calls one tool through "
            "it (server must exist and be enabled, tool must be listed, "
            "caller must hold net.out); action=openapi_import converts "
            "an OpenAPI 3.x document into MCP tool definitions and "
            "action=openapi_call calls one converted operation; "
            "action=skill serves an installed skill; action=receipts "
            "lists the hash-chained provenance receipts and verifies "
            "the chain."
        ),
        capability=Capability.AGENT_SPAWN,
        parameters={
            "action": "str — discover | describe | route | mcp | mcp_call | openapi_import | openapi_call | skill | receipts",
            "goal": "str (route) — what needs doing, in plain words",
            "execute": "bool (route) — run the top allowed candidate instead of just ranking",
            "kind": "str (describe) — tool | mcp | a2a | skill | role | mesh_node | job",
            "name": "str (describe/skill) — the agent, server, skill, role, node or job name",
            "server": "str (mcp_call) — the MCP server name",
            "tool": "str (mcp_call/openapi_call) — the tool or converted operation name",
            "args": "object (optional) — arguments for the routed call",
            "document": "object or str (openapi_*) — the OpenAPI 3.x document (dict, or a JSON string)",
            "auth_env": "str (openapi_call) — environment variable holding the credential, when the operation needs auth",
            "limit": "int (optional) — how many candidates/receipts to return",
        },
    )
    def _control(action: str = "discover", goal: str = "",
                 execute: bool = False, kind: str = "", name: str = "",
                 server: str = "", tool: str = "",
                 args: dict[str, Any] | None = None,
                 document: Any = None, auth_env: str | None = None,
                 limit: int = 5) -> dict[str, Any]:
        plane = _plane(registry)
        action = str(action or "discover").lower()
        try:
            if action == "discover":
                found = plane.discover()
                found["ok"] = True
                return found
            if action == "describe":
                if not kind or not name:
                    return {"ok": False,
                            "error": "describe needs kind and name"}
                described = plane.describe(kind, name)
                described["ok"] = True
                return described
            if action == "route":
                if not goal:
                    return {"ok": False, "error": "route needs a goal"}
                return plane.route(goal, execute=bool(execute),
                                   args=dict(args or {}),
                                   limit=int(limit))
            if action == "mcp":
                return {"ok": True, "servers": plane.gateway_servers()}
            if action == "mcp_call":
                if not server or not tool:
                    return {"ok": False,
                            "error": "mcp_call needs server and tool"}
                result = plane.gateway_call(server, tool, dict(args or {}))
                return {"ok": True, "result": result}
            if action == "openapi_import":
                if document is None:
                    return {"ok": False,
                            "error": "openapi_import needs a document"}
                return plane.openapi_import(document)
            if action == "openapi_call":
                if document is None or not tool:
                    return {"ok": False,
                            "error": "openapi_call needs a document and a tool"}
                return plane.call_openapi_tool(
                    document, tool, dict(args or {}), auth_env=auth_env)
            if action == "skill":
                if not name:
                    found = plane.discover()
                    section = found["sections"].get("skills") or {}
                    return {"ok": True,
                            "available": bool(section.get("available")),
                            "count": section.get("count", 0),
                            "skills": section.get("skills", []),
                            "reason": section.get("reason", "")}
                return plane.serve_skill(name)
            if action == "receipts":
                return {
                    "ok": True,
                    "receipts": plane.list_receipts(limit=int(limit)),
                    "chain": plane.verify_chain(limit=1000),
                }
            return {"ok": False,
                    "error": f"unknown control action {action!r}; expected "
                    "discover | describe | route | mcp | mcp_call | "
                    "openapi_import | openapi_call | skill | receipts"}
        except ControlError as exc:
            return {"ok": False, "error": str(exc)}


def control_chat(tail: str, context: Any) -> str:
    """The /control command. Owner-only (CONTROL_COMMANDS, never public).

    /control — surface summary · /control agents — everything Devon
    has · /control describe <kind> <name> — one agent's platform view ·
    /control route <goal> [execute] — rank (and optionally run) the best
    candidate · /control mcp — gateway servers · /control mcp call
    <server> <tool> [json-args] — one governed call · /control skills
    [name] — skills · /control receipts — provenance chain check.
    """
    plane = ControlPlane.from_context(context, actor="owner")
    parts = (tail or "").split()
    verb = parts[0].lower() if parts else ""

    if verb in ("", "status"):
        try:
            found = plane.discover()
        except ControlError as exc:
            return "Control plane: " + str(exc)
        lines = ["Control plane — one surface over what Devon has:"]
        for name, section in found["sections"].items():
            if section.get("available"):
                count = section.get("count")
                suffix = f" — {count}" if count is not None else ""
                lines.append(f"· {name}{suffix}")
            else:
                lines.append(f"· {name} — unavailable: {section.get('reason', '')}")
        return "\n".join(lines)

    if verb == "agents":
        try:
            found = plane.discover()
        except ControlError as exc:
            return "Control plane: " + str(exc)
        lines = []
        tools = found["sections"].get("tools") or {}
        lines.append(f"Tools: {tools.get('count', 0)} in the live registry")
        servers = found["sections"].get("mcp_servers") or {}
        for entry in servers.get("servers") or []:
            state = ("connected" if entry.get("connected")
                     else "enabled" if entry.get("enabled") else "disabled")
            lines.append(f"· mcp {entry['name']} [{entry.get('transport')}]: {state}")
        agents = found["sections"].get("a2a_agents") or {}
        for entry in agents.get("agents") or []:
            lines.append(f"· a2a {entry['name']}: "
                         + ("enabled" if entry.get("enabled") else "disabled"))
        skills = found["sections"].get("skills") or {}
        lines.append(f"Skills: {skills.get('count', 0)} installed")
        delegation = found["sections"].get("delegation") or {}
        roles = [r["role"] for r in delegation.get("roles") or []]
        lines.append("Delegation roles: " + (", ".join(roles) or "none"))
        lines.append(f"Mesh nodes: {(found['sections'].get('mesh_nodes') or {}).get('count', 0)}")
        lines.append(f"Jobs: {(found['sections'].get('jobs') or {}).get('count', 0)}")
        lines.append(f"Approvals pending: {(found['sections'].get('approvals_pending') or {}).get('count', 0)}")
        return "\n".join(lines)

    if verb == "describe":
        if len(parts) < 3:
            return "Usage: /control describe <tool|mcp|a2a|skill|role|mesh_node|job> <name>"
        try:
            described = plane.describe(parts[1], parts[2])
        except ControlError as exc:
            return "Control plane: " + str(exc)
        lines = [f"{described['kind']}:{described['name']} — {described.get('description') or ''}"]
        if described.get("capability"):
            lines.append(f"Capability: {described['capability']}")
        if described.get("enabled") is not None:
            lines.append(f"Enabled: {described['enabled']}; connected: {described.get('connected')}")
        for cred in described.get("credentials") or []:
            lines.append(f"Credential {cred['env']}: "
                         + ("present" if cred.get("present") else "MISSING"))
        backlog = described.get("backlog") or []
        lines.append(f"Backlog: {len(backlog)} item(s)")
        lines.append(f"Approvals pending: {described.get('approvals_pending', 0)}")
        logs = described.get("run_logs") or []
        lines.append(f"Run logs: {len(logs)} receipt(s)")
        return "\n".join(lines)

    if verb == "route":
        if len(parts) < 2:
            return "Usage: /control route <goal> [execute]"
        execute = parts[-1].lower() == "execute"
        goal = " ".join(parts[1:-1] if execute else parts[1:])
        try:
            result = plane.route(goal, execute=execute, actor="owner")
        except ControlError as exc:
            return "Control plane: " + str(exc)
        if not result.get("ok"):
            return "Control plane: " + str(result.get("error"))
        lines = [f"Route for {goal!r}:"]
        for cand in result.get("candidates") or []:
            verdict = "allowed" if cand.get("granted") else (
                "needs " + str(cand.get("capability")))
            lines.append(f"· {cand['kind']}:{cand['name']} "
                         f"(score {cand['score']:.1f}) — {verdict}")
        if result.get("executed"):
            lines.append(f"Executed {result['candidate']['kind']}:"
                         f"{result['candidate']['name']}")
        return "\n".join(lines)

    if verb == "mcp":
        if len(parts) >= 2 and parts[1].lower() == "call":
            if len(parts) < 4:
                return "Usage: /control mcp call <server> <tool> [json-args]"
            raw_args: dict[str, Any] = {}
            if len(parts) > 4:
                try:
                    raw_args = json.loads(" ".join(parts[4:]))
                except ValueError as exc:
                    return "Control plane: bad JSON args: " + str(exc)
                if not isinstance(raw_args, dict):
                    return "Control plane: args must be a JSON object"
            try:
                result = plane.gateway_call(parts[2], parts[3], raw_args)
            except ControlError as exc:
                return "Control plane: " + str(exc)
            return "MCP " + parts[2] + ":" + parts[3] + " → " + json.dumps(result)[:2000]
        try:
            servers = plane.gateway_servers()
        except ControlError as exc:
            return "Control plane: " + str(exc)
        lines = ["MCP gateway:"]
        for entry in servers:
            state = ("connected" if entry.get("connected")
                     else "enabled" if entry.get("enabled") else "disabled")
            tools = entry.get("tools")
            suffix = f", {tools} tools" if tools is not None else ""
            lines.append(f"· {entry['name']} [{entry.get('transport')}]: {state}{suffix}")
        return "\n".join(lines)

    if verb == "skills":
        if len(parts) >= 2:
            try:
                served = plane.serve_skill(parts[1])
            except ControlError as exc:
                return "Control plane: " + str(exc)
            return (f"Skill {served['name']} v{served['version']} "
                    f"({'enabled' if served['enabled'] else 'disabled'}) — "
                    f"{served['description']} "
                    f"(tools: {', '.join(served['tools']) or 'none'})")
        found = plane.discover()
        section = found["sections"].get("skills") or {}
        if not section.get("available"):
            return "Control plane: skills unavailable: " + str(section.get("reason", ""))
        lines = [f"Skills: {section.get('count', 0)} installed"]
        for entry in section.get("skills") or []:
            lines.append(f"· {entry['name']} v{entry.get('active_version')} "
                         f"({'on' if entry.get('enabled') else 'off'}) — "
                         f"{entry.get('description', '')}")
        return "\n".join(lines)

    if verb == "receipts":
        chain = plane.verify_chain()
        receipts = plane.list_receipts(limit=5)
        lines = [f"Provenance chain: {'OK' if chain.get('ok') else 'BROKEN'} "
                 f"({chain.get('checked', 0)} receipts checked)"]
        if chain.get("broken"):
            lines.append("Broken receipts: " + ", ".join(chain["broken"]))
        for receipt in receipts:
            lines.append(f"· {receipt['action']} → {receipt['target']} "
                         f"({receipt['result']})")
        return "\n".join(lines)

    return ("Unknown /control verb. Try: /control · /control agents · "
            "/control describe <kind> <name> · /control route <goal> "
            "[execute] · /control mcp · /control skills [name] · "
            "/control receipts")
