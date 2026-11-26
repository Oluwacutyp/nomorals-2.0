"""Control-plane routes for the API server (F3).

One discovery/routing surface over the organs that already exist —
registered the same guarded way as the trigger webhooks, so a control
problem never breaks the API server. Every route honors the resolved
principal's grant: reads need db.read, routing that executes needs
agent.spawn, anything that calls out needs net.out. Per-request routing
only — no centre, no shared queue.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..hub.control import ControlError, ControlPlane

_CONTROL_ROUTES = [
    ("GET", "/control", "Control-plane surface index"),
    ("GET", "/control/agents", "Discover agents, tools, servers, skills, nodes, jobs"),
    ("GET", "/control/describe", "One agent's platform view (?kind=&name=)"),
    ("POST", "/control/route", "Rank candidates for a goal; execute the top one"),
    ("GET", "/control/mcp", "Governed MCP gateway: servers + live state"),
    ("POST", "/control/mcp/call", "One governed MCP call"),
    ("POST", "/control/openapi/import", "Convert an OpenAPI document to MCP tools"),
    ("POST", "/control/openapi/call", "Call one converted OpenAPI operation"),
    ("GET", "/control/skills", "List skills, or serve one (?name=)"),
    ("GET", "/control/receipts", "Provenance receipts + chain verification"),
]


def register_control_routes(server: Any, context: Any) -> Any:
    """Register the /control/* routes on an API server instance."""

    from .server import _require_capability

    def _plane(principal_name: str) -> ControlPlane:
        return ControlPlane.from_context(context, actor=principal_name)

    def _principal() -> Any:
        return server._current_principal()  # noqa: SLF001

    @server.route("GET", "/control", description=_CONTROL_ROUTES[0][2])
    def control_index(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.DB_READ)
        plane = _plane(_principal().name)
        found = plane.discover(grant=_principal().grant)
        sections = {}
        for name, section in found["sections"].items():
            sections[name] = {
                "available": bool(section.get("available")),
                "count": section.get("count", 0),
            }
            if not section.get("available"):
                sections[name]["reason"] = section.get("reason", "")
        return {
            "ok": True,
            "routes": [{"method": m, "path": p, "description": d}
                       for m, p, d in _CONTROL_ROUTES],
            "sections": sections,
        }

    @server.route("GET", "/control/agents", description=_CONTROL_ROUTES[1][2])
    def control_agents(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.DB_READ)
        plane = _plane(_principal().name)
        found = plane.discover(grant=_principal().grant)
        found["ok"] = True
        return found

    @server.route("GET", "/control/describe", description=_CONTROL_ROUTES[2][2])
    def control_describe(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.DB_READ)
        kind = str(query.get("kind") or body.get("kind") or "")
        name = str(query.get("name") or body.get("name") or "")
        if not kind or not name:
            return {"ok": False,
                    "error": "describe needs ?kind=<tool|mcp|a2a|skill|role|mesh_node|job>&name=<name>"}
        plane = _plane(_principal().name)
        try:
            described = plane.describe(kind, name, grant=_principal().grant)
        except ControlError as exc:
            return {"ok": False, "error": str(exc)}
        described["ok"] = True
        return described

    @server.route("POST", "/control/route", description=_CONTROL_ROUTES[3][2])
    def control_route(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.DB_READ)
        data = body or {}
        goal = str(data.get("goal") or "")
        execute = bool(data.get("execute", False))
        if execute:
            _require_capability(_principal(), Capability.AGENT_SPAWN)
        if not goal:
            return {"ok": False, "error": "route needs a goal"}
        args = data.get("args") or {}
        if not isinstance(args, dict):
            return {"ok": False, "error": "args must be an object"}
        plane = _plane(_principal().name)
        try:
            return plane.route(
                goal, grant=_principal().grant, execute=execute,
                actor=_principal().name, args=args,
                limit=int(data.get("limit", 5)))
        except ControlError as exc:
            return {"ok": False, "error": str(exc)}

    @server.route("GET", "/control/mcp", description=_CONTROL_ROUTES[4][2])
    def control_mcp(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.DB_READ)
        plane = _plane(_principal().name)
        return {"ok": True, "servers": plane.gateway_servers()}

    @server.route("POST", "/control/mcp/call", description=_CONTROL_ROUTES[5][2])
    def control_mcp_call(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.NET_OUT)
        data = body or {}
        server_name = str(data.get("server") or "")
        tool = str(data.get("tool") or "")
        if not server_name or not tool:
            return {"ok": False,
                    "error": "mcp/call needs server and tool"}
        args = data.get("args") or {}
        if not isinstance(args, dict):
            return {"ok": False, "error": "args must be an object"}
        plane = _plane(_principal().name)
        try:
            result = plane.gateway_call(
                server_name, tool, args, grant=_principal().grant,
                actor=_principal().name)
        except ControlError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "result": result}

    @server.route("POST", "/control/openapi/import", description=_CONTROL_ROUTES[6][2])
    def control_openapi_import(body: dict[str, Any],
                               query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.DB_READ)
        data = body or {}
        doc = data.get("document", data)
        plane = _plane(_principal().name)
        try:
            return plane.openapi_import(doc)
        except ControlError as exc:
            return {"ok": False, "error": str(exc)}

    @server.route("POST", "/control/openapi/call", description=_CONTROL_ROUTES[7][2])
    def control_openapi_call(body: dict[str, Any],
                             query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.NET_OUT)
        data = body or {}
        doc = data.get("document")
        tool = str(data.get("tool") or "")
        if doc is None or not tool:
            return {"ok": False,
                    "error": "openapi/call needs document and tool"}
        args = data.get("args") or {}
        if not isinstance(args, dict):
            return {"ok": False, "error": "args must be an object"}
        plane = _plane(_principal().name)
        try:
            return plane.call_openapi_tool(
                doc, tool, args,
                auth_env=data.get("auth_env"),
                grant=_principal().grant,
                actor=_principal().name,
                timeout_s=float(data.get("timeout_s", 30.0)))
        except ControlError as exc:
            return {"ok": False, "error": str(exc)}

    @server.route("GET", "/control/skills", description=_CONTROL_ROUTES[8][2])
    def control_skills(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.DB_READ)
        name = str(query.get("name") or body.get("name") or "")
        plane = _plane(_principal().name)
        if not name:
            found = plane.discover(grant=_principal().grant)
            section = found["sections"].get("skills") or {}
            return {"ok": True, "available": bool(section.get("available")),
                    "count": section.get("count", 0),
                    "skills": section.get("skills", []),
                    "reason": section.get("reason", "")}
        try:
            return plane.serve_skill(name)
        except ControlError as exc:
            return {"ok": False, "error": str(exc)}

    @server.route("GET", "/control/receipts", description=_CONTROL_ROUTES[9][2])
    def control_receipts(body: dict[str, Any], query: dict[str, str]) -> dict[str, Any]:
        _require_capability(_principal(), Capability.DB_READ)
        plane = _plane(_principal().name)
        try:
            limit = int(query.get("limit") or body.get("limit") or 50)
        except (TypeError, ValueError):
            limit = 50
        return {
            "ok": True,
            "receipts": plane.list_receipts(limit=limit),
            "chain": plane.verify_chain(limit=1000),
        }

    return server
