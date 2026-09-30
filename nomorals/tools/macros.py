"""Automation recorder: record action sequences into reusable tools.

Two ways a macro is built:

1. **Explicit** — ``record_start`` / ``record_step`` / ``record_stop`` from
   chat (/record) or from the model itself;
2. **Captured** — while a recording is active, every tool call the runtime
   or a Devon agent executes on the user's behalf is appended
   automatically (the devon loop and the chat commands call
   :func:`record_step`).

A saved macro is a durable row (migration 16) AND a live tool: it is
registered as ``macro_<name>`` in the tool registry, so the main AI and
every sub-agent can call it by name with the same schema as any tool.
``macro_run`` replays the steps sequentially through the registry, stops at
the first failure, and keeps run history (runs / last_result).
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from ..core.errors import ToolError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["record_start", "record_step", "record_stop", "record_status",
           "list_macros", "show_macro", "run_macro", "delete_macro", "register"]

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")

# ── live recorder state (one partner process) ────────────────────────────────

_recorder_lock = threading.Lock()
_active: dict[str, Any] = {}  # name -> {"steps": [...], "started_at": float}


def record_start(name: str, description: str = "") -> dict[str, Any]:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ToolError(f"bad macro name {name!r} — letters/digits/._-, <= 48 chars")
    with _recorder_lock:
        _active["current"] = {"name": name, "description": description,
                              "steps": [], "started_at": time.time()}
    _log.info("macro recording started: %s", name)
    return {"recording": name, "steps": 0}


def record_step(tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append one step. Called explicitly, or automatically by the devon
    loop / runtime while a recording is active."""
    with _recorder_lock:
        rec = _active.get("current")
        if rec is None:
            return {"recording": False, "note": "no recording active"}
        step = {"tool": str(tool or ""), "args": dict(args or {})}
        rec["steps"].append(step)
        count = len(rec["steps"])
        name = rec["name"]
    _log.debug("macro %s: step %d = %s", name, count, tool)
    return {"recording": True, "steps": count}


def record_stop(registry: Any = None, context: Any = None) -> dict[str, Any]:
    with _recorder_lock:
        rec = _active.get("current")
        if rec is None:
            raise ToolError("no recording active — /record start <name> first")
        _active.pop("current", None)
    steps = rec["steps"]
    if not steps:
        raise ToolError("recording captured no steps — nothing to save")
    result = save_macro(context, registry, rec["name"], rec["description"], steps)
    return result


def record_status() -> dict[str, Any]:
    with _recorder_lock:
        rec = _active.get("current")
        if rec is None:
            return {"recording": False}
        return {"recording": True, "name": rec["name"], "steps": len(rec["steps"]),
                "seconds": round(time.time() - rec["started_at"], 1),
                "last": rec["steps"][-1]["tool"] if rec["steps"] else None}


# ── durable store ────────────────────────────────────────────────────────────


def _db(context: Any):
    db = getattr(context, "db", None)
    if db is None:
        raise ToolError("macro store needs a database context")
    return db


def list_macros(context: Any) -> list[dict[str, Any]]:
    db = _db(context)
    rows = db.query("SELECT name, description, steps, runs, last_run, last_result "
                    "FROM macros ORDER BY created_at DESC LIMIT 50")
    out = []
    for row in rows:
        steps = json.loads(row["steps"] or "[]")
        out.append({
            "name": row["name"], "description": row["description"],
            "steps": len(steps), "tools": sorted({s["tool"] for s in steps}),
            "runs": row["runs"],
            "last_result": (row["last_result"] or "")[:120],
        })
    return out


def show_macro(context: Any, name: str) -> dict[str, Any]:
    db = _db(context)
    row = db.query_one("SELECT * FROM macros WHERE name = ?", ((name or "").strip(),))
    if row is None:
        raise ToolError(f"no macro named {name!r}")
    return {"name": row["name"], "description": row["description"],
            "steps": json.loads(row["steps"] or "[]"),
            "runs": row["runs"], "last_result": row["last_result"]}


def delete_macro(context: Any, name: str) -> dict[str, Any]:
    db = _db(context)
    name = (name or "").strip()
    row = db.query_one("SELECT id FROM macros WHERE name = ?", (name,))
    if row is None:
        raise ToolError(f"no macro named {name!r}")
    with db.transaction():
        db.execute("DELETE FROM macros WHERE id = ?", (row["id"],))
    return {"deleted": name}


def save_macro(context: Any, registry: Any, name: str, description: str,
               steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate, persist, and register the macro as a callable tool."""
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ToolError(f"bad macro name {name!r}")
    clean: list[dict[str, Any]] = []
    registry_names = registry.names() if registry is not None else None
    for step in steps:
        tool = str(step.get("tool") or "")
        args = step.get("args") or {}
        if not tool:
            raise ToolError("macro steps need a tool name")
        if registry_names is not None and tool not in registry_names \
                and not tool.startswith("macro_"):
            raise ToolError(f"unknown tool {tool!r} in macro — record real steps only")
        if not isinstance(args, dict):
            raise ToolError("macro step args must be an object")
        clean.append({"tool": tool, "args": args})
    if not clean:
        raise ToolError("macro has no steps")

    db = _db(context)
    existing = db.query_one("SELECT id FROM macros WHERE name = ?", (name,))
    with db.transaction():
        if existing:
            db.execute(
                "UPDATE macros SET description = ?, steps = ? WHERE id = ?",
                (description, json.dumps(clean), existing["id"]),
            )
        else:
            db.execute(
                "INSERT INTO macros (id, name, description, steps, created_at, runs) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (new_id(), name, description, json.dumps(clean), time.time()),
            )
    if registry is not None:
        _register_tool(registry, context, name, clean)
    return {"saved": name, "steps": len(clean),
            "tool": f"macro_{name}", "callable": True}


def _register_tool(registry: Any, context: Any, name: str,
                   steps: list[dict[str, Any]]) -> None:
    """(Re)register ``macro_<name>`` as a live tool."""
    tool_name = f"macro_{name}"
    try:
        registry.unregister(tool_name)
    except Exception:  # noqa: BLE001
        pass

    @registry.register(
        tool_name,
        description=f"recorded automation macro: {len(steps)} steps ({' -> '.join(s['tool'] for s in steps[:5])}{' …' if len(steps) > 5 else ''})",
        capability=Capability.NET_OUT,
        parameters={
            "overrides": "str (optional) — JSON object mapping step index to replacement args"
        },
    )
    def macro_tool(overrides: str = "") -> dict[str, Any]:
        return run_macro(context, registry, name, overrides)

    # keep a stable reference so unregister works later
    setattr(macro_tool, "_nm_macro", name)


def run_macro(context: Any, registry: Any, name: str, overrides: str = "") -> dict[str, Any]:
    name = (name or "").strip()
    db = _db(context)
    row = db.query_one("SELECT * FROM macros WHERE name = ?", (name,))
    if row is None:
        raise ToolError(f"no macro named {name!r}")
    steps = json.loads(row["steps"] or "[]")
    override_map: dict[int, dict[str, Any]] = {}
    if (overrides or "").strip():
        try:
            parsed = json.loads(overrides)
        except (ValueError, TypeError) as exc:
            raise ToolError(f"overrides must be a JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ToolError("overrides must be a JSON object of index -> args")
        for key, value in parsed.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                raise ToolError(f"bad override key {key!r}") from None
            if not isinstance(value, dict):
                raise ToolError(f"override for step {idx} must be an args object")
            override_map[idx] = value

    started = time.time()
    results: list[dict[str, Any]] = []
    failed_at: int | None = None
    for idx, step in enumerate(steps):
        args = override_map.get(idx, step["args"])
        outcome = registry.call(step["tool"], actor="macro", **args)
        ok = bool(outcome.ok)
        results.append({
            "step": idx, "tool": step["tool"], "ok": ok,
            "result": (str(outcome.value)[:300] if ok
                       else str(getattr(outcome.error, "message", outcome.error))[:300]),
        })
        if not ok:
            failed_at = idx
            break
    summary = (f"{len(steps)} steps ok" if failed_at is None
               else f"stopped at step {failed_at}: {results[-1]['result']}")
    with db.transaction():
        db.execute(
            "UPDATE macros SET runs = runs + 1, last_run = ?, last_result = ? WHERE id = ?",
            (time.time(), summary[:500], row["id"]),
        )
    return {"macro": name, "ok": failed_at is None, "steps": results,
            "summary": summary, "seconds": round(time.time() - started, 2)}


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "record_start",
        description="Begin recording tool actions into a named macro.",
        capability=Capability.DB_WRITE,
        parameters={"name": "str — macro name", "description": "str (optional)"},
    )
    def record_start_tool(name: str, *, description: str = "") -> dict[str, Any]:
        return record_start(name, description)

    @registry.register(
        "record_step",
        description="Append an explicit step to the active recording.",
        capability=Capability.DB_WRITE,
        parameters={"tool": "str", "args": "str (optional) — JSON object"},
    )
    def record_step_tool(tool: str, *, args: str = "") -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        if (args or "").strip():
            try:
                loaded = json.loads(args)
            except (ValueError, TypeError) as exc:
                raise ToolError(f"args must be a JSON object: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ToolError("args must be a JSON object")
            parsed = loaded
        return record_step(tool, parsed)

    @registry.register(
        "record_stop",
        description="Finish the recording: save it and register it as a callable tool.",
        capability=Capability.DB_WRITE,
    )
    def record_stop_tool() -> dict[str, Any]:
        return record_stop(registry, context)

    @registry.register(
        "record_status",
        description="Is a recording active? How many steps so far?",
        capability=Capability.DB_READ,
    )
    def record_status_tool() -> dict[str, Any]:
        return record_status()

    @registry.register(
        "macro_list",
        description="List recorded automation macros with their step counts.",
        capability=Capability.DB_READ,
    )
    def macro_list_tool() -> dict[str, Any]:
        return {"macros": list_macros(context)}

    @registry.register(
        "macro_show",
        description="Show a macro's steps.",
        capability=Capability.DB_READ,
        parameters={"name": "str"},
    )
    def macro_show_tool(name: str) -> dict[str, Any]:
        return show_macro(context, name)

    @registry.register(
        "macro_run",
        description=(
            "Replay a recorded macro through the tool registry; stops at the first "
            "failing step. Optional overrides: JSON {step index: replacement args}."
        ),
        capability=Capability.DB_READ,
        parameters={"name": "str", "overrides": "str (optional) — JSON object"},
    )
    def macro_run_tool(name: str, *, overrides: str = "") -> dict[str, Any]:
        return run_macro(context, registry, name, overrides)

    @registry.register(
        "macro_delete",
        description="Delete a recorded macro.",
        capability=Capability.DB_WRITE,
        parameters={"name": "str"},
    )
    def macro_delete_tool(name: str) -> dict[str, Any]:
        out = delete_macro(context, name)
        try:
            registry.unregister(f"macro_{name}")
        except Exception:  # noqa: BLE001 - row is gone; a stale tool is harmless
            _log.debug("could not unregister macro_%s", name, exc_info=True)
        return out

    # Register every macro already on disk as a live tool at boot.
    try:
        rows = context.db.query("SELECT name, steps FROM macros") if getattr(context, "db", None) else []
        for row in rows or []:
            try:
                _register_tool(registry, context, row["name"], json.loads(row["steps"] or "[]"))
            except Exception:  # noqa: BLE001 - one broken macro must not block boot
                _log.warning("could not register macro %s", row.get("name"), exc_info=True)
    except Exception:  # noqa: BLE001
        _log.debug("macro boot registration skipped", exc_info=True)
