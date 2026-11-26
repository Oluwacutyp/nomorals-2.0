"""``nm control`` — the control plane on the command line.

``nm control agents`` lists everything Devon has (tools, MCP servers,
A2A agents, skills, delegation roles, mesh nodes, jobs, approvals).
``nm control describe --kind tool --name <name>`` shows one agent's
platform view. ``nm control route --goal "..." [--execute]`` ranks
candidates for a goal and optionally runs the top allowed one.
``nm control mcp`` / ``nm control mcp-call --server S --tool T``
reach the governed MCP gateway. ``nm control openapi-import`` /
``nm control openapi-call`` convert and call OpenAPI operations.
``nm control skills [--name N]`` serves skills. ``nm control receipts``
verifies the hash-chained provenance chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...hub.control import ControlError, ControlPlane
from ..emit import _emit


def _read_document(raw: str) -> Any:
    """--document accepts a file path, @path, or inline JSON."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("openapi needs --document <file.json|@file.json|json>")
    if text.startswith("@"):
        text = Path(text[1:]).read_text(encoding="utf-8")
    elif Path(text).is_file():
        text = Path(text).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(f"document is not valid JSON: {exc}") from exc


def _read_args(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--args is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--args must be a JSON object")
    return data


def _cmd_control(args: argparse.Namespace, context: Any) -> int:
    """``nm control <agents|describe|route|mcp|mcp-call|openapi-import|openapi-call|skills|receipts>``."""
    action = getattr(args, "action", "agents") or "agents"
    name = (getattr(args, "name", "") or "").strip()
    plane = ControlPlane.from_context(context, actor="cli")
    try:
        if action == "agents":
            payload = plane.discover()
            payload["ok"] = True
            sections = payload["sections"]
            lines = ["Control plane — everything Devon has:"]
            for section_name, section in sections.items():
                if section.get("available"):
                    count = section.get("count")
                    suffix = f" — {count}" if count is not None else ""
                    lines.append(f"· {section_name}{suffix}")
                else:
                    lines.append(f"· {section_name} — unavailable: "
                                 f"{section.get('reason', '')}")
            _emit(args, payload, "\n".join(lines))
            return 0
        if action == "describe":
            kind = (getattr(args, "kind", "") or "").strip()
            if not kind or not name:
                print("control describe needs --kind <kind> and a name",
                      file=sys.stderr)
                return 1
            described = plane.describe(kind, name)
            described["ok"] = True
            lines = [f"{described['kind']}:{described['name']} — "
                     f"{described.get('description') or ''}"]
            if described.get("capability"):
                lines.append(f"Capability: {described['capability']}")
            for cred in described.get("credentials") or []:
                lines.append(f"Credential {cred['env']}: "
                             + ("present" if cred.get("present") else "MISSING"))
            lines.append(f"Backlog: {len(described.get('backlog') or [])} item(s)")
            lines.append(f"Approvals pending: {described.get('approvals_pending', 0)}")
            lines.append(f"Run logs: {len(described.get('run_logs') or [])} receipt(s)")
            _emit(args, described, "\n".join(lines))
            return 0
        if action == "route":
            goal = (getattr(args, "goal", "") or "").strip() or name
            if not goal:
                print("control route needs --goal <goal>", file=sys.stderr)
                return 1
            result = plane.route(goal, execute=bool(getattr(args, "execute", False)),
                                 args=_read_args(getattr(args, "args", "")),
                                 actor="cli")
            lines = [f"Route for {goal!r}:"]
            for cand in result.get("candidates") or []:
                verdict = "allowed" if cand.get("granted") else (
                    "needs " + str(cand.get("capability")))
                lines.append(f"· {cand['kind']}:{cand['name']} "
                             f"(score {cand['score']:.1f}) — {verdict}")
            if result.get("executed"):
                lines.append(f"Executed {result['candidate']['kind']}:"
                             f"{result['candidate']['name']}")
            if not result.get("ok"):
                lines.append("Error: " + str(result.get("error")))
            _emit(args, result, "\n".join(lines))
            return 0 if result.get("ok") else 1
        if action == "mcp":
            servers = plane.gateway_servers()
            payload = {"ok": True, "servers": servers}
            lines = ["MCP gateway:"]
            for entry in servers:
                state = ("connected" if entry.get("connected")
                         else "enabled" if entry.get("enabled") else "disabled")
                tools = entry.get("tools")
                suffix = f", {tools} tools" if tools is not None else ""
                lines.append(f"· {entry['name']} [{entry.get('transport')}]: "
                             f"{state}{suffix}")
            _emit(args, payload, "\n".join(lines))
            return 0
        if action == "mcp-call":
            server = (getattr(args, "server", "") or "").strip()
            tool = (getattr(args, "tool", "") or "").strip()
            if not server or not tool:
                print("control mcp-call needs --server <server> --tool <tool>",
                      file=sys.stderr)
                return 1
            result = plane.gateway_call(server, tool,
                                        _read_args(getattr(args, "args", "")))
            _emit(args, {"ok": True, "result": result},
                  json.dumps(result, indent=2)[:4000])
            return 0
        if action == "openapi-import":
            imported = plane.openapi_import(
                _read_document(getattr(args, "document", "")))
            lines = [f"OpenAPI → MCP: {imported['count']} tool(s)"]
            for tool_def in imported["tools"]:
                lines.append(f"· {tool_def['name']} — {tool_def['description']}")
            _emit(args, imported, "\n".join(lines))
            return 0
        if action == "openapi-call":
            tool = (getattr(args, "tool", "") or "").strip()
            if not tool:
                print("control openapi-call needs --tool <name>",
                      file=sys.stderr)
                return 1
            result = plane.call_openapi_tool(
                _read_document(getattr(args, "document", "")), tool,
                _read_args(getattr(args, "args", "")),
                auth_env=(getattr(args, "auth_env", "") or None))
            _emit(args, result, json.dumps(result, indent=2)[:4000])
            return 0
        if action == "skills":
            if not name:
                found = plane.discover()
                section = found["sections"].get("skills") or {}
                if not section.get("available"):
                    print("control: skills unavailable: "
                          + str(section.get("reason", "")), file=sys.stderr)
                    return 1
                payload = {"ok": True, "count": section.get("count", 0),
                           "skills": section.get("skills", [])}
                lines = [f"Skills: {section.get('count', 0)} installed"]
                for entry in section.get("skills") or []:
                    lines.append(f"· {entry['name']} v{entry.get('active_version')} "
                                 f"({'on' if entry.get('enabled') else 'off'}) — "
                                 f"{entry.get('description', '')}")
                _emit(args, payload, "\n".join(lines))
                return 0
            served = plane.serve_skill(name)
            lines = [f"Skill {served['name']} v{served['version']} "
                     f"({'enabled' if served['enabled'] else 'disabled'}) — "
                     f"{served['description']} "
                     f"(tools: {', '.join(served['tools']) or 'none'})"]
            _emit(args, served, "\n".join(lines))
            return 0
        if action == "receipts":
            limit = int(getattr(args, "limit", 5) or 5)
            chain = plane.verify_chain()
            receipts = plane.list_receipts(limit=limit)
            payload = {"ok": True, "receipts": receipts, "chain": chain}
            lines = [f"Provenance chain: {'OK' if chain.get('ok') else 'BROKEN'} "
                     f"({chain.get('checked', 0)} receipts checked)"]
            if chain.get("broken"):
                lines.append("Broken receipts: " + ", ".join(chain["broken"]))
            for receipt in receipts:
                lines.append(f"· {receipt['action']} → {receipt['target']} "
                             f"({receipt['result']})")
            _emit(args, payload, "\n".join(lines))
            return 0
        print(f"unknown control action {action!r}", file=sys.stderr)
        return 1
    except ControlError as exc:
        print("control: " + str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print("control: " + str(exc), file=sys.stderr)
        return 1
