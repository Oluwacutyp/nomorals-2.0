"""``nm status`` — status sections."""

from __future__ import annotations

import argparse
from typing import Any
from ...version import __version__
from ..emit import _emit



def _status_section(name: str, fn: Any) -> tuple[dict[str, Any], list[str]]:
    """Run one status probe. A failing probe is reported as unavailable —
    never as zeros that look measured."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - status must degrade, not crash
        err = f"{type(exc).__name__}: {exc}"
        return ({"available": False, "error": err},
                [f"{name}: unavailable ({type(exc).__name__})"])


def _cmd_status(args: argparse.Namespace, context: Any) -> int:
    """`nm status` — system health snapshot with real numbers.

    Every section is a bounded local query (no LLM calls, no network).
    A subsystem that cannot be read is reported as unavailable.
    """
    import platform as _platform

    sections: dict[str, dict[str, Any]] = {}
    blocks: list[list[str]] = []

    def _system() -> tuple[dict[str, Any], list[str]]:
        data = {
            "available": True,
            "version": __version__,
            "profile": context.settings.profile,
            "platform": _platform.platform(),
            "python": _platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
        }
        text = [f"system:   NoMorals Core {data['version']}  "
                f"profile={data['profile']}"]
        text.append(f"          {data['platform']}  "
                    f"py={data['python']}  cpus={data['cpu_count']}")
        return data, text

    def _database() -> tuple[dict[str, Any], list[str]]:
        version = context.db.scalar(
            "SELECT COALESCE(MAX(version),0) FROM schema_migrations")
        tables = len(context.db.tables())
        integrity = context.db.integrity_check()
        data = {"available": True, "path": str(context.settings.db_path),
                "schema_version": version, "tables": tables,
                "integrity": integrity}
        text = [f"database: {data['path']}",
                f"          schema v{version}, {tables} tables, "
                f"integrity {integrity}"]
        return data, text

    def _queue() -> tuple[dict[str, Any], list[str]]:
        from ...storage.queue import WorkQueue

        queue = WorkQueue(context.db)
        pending = int(queue.pending(None))
        topics = queue.topics()
        data = {"available": True, "pending": pending, "topics": topics}
        text = [f"queue:    {pending} pending across {len(topics)} topic(s)"]
        return data, text

    def _missions() -> tuple[dict[str, Any], list[str]]:
        from ...missions import MissionStore

        store = MissionStore(context.db)
        stats = store.stats()
        resumable = store.resumable()
        data = {"available": True, "total": stats["total"],
                "active": stats["active"],
                "interrupted": len(resumable)}
        text = [f"missions: {stats['total']} total, {stats['active']} active, "
                f"{len(resumable)} interrupted (resumable)"]
        return data, text

    def _memory() -> tuple[dict[str, Any], list[str]]:
        stats = context.memory.stats_snapshot()
        data = {"available": True, "records": stats.get("records", 0),
                "by_kind": stats.get("by_kind", {})}
        text = [f"memory:   {data['records']} records"]
        return data, text

    def _proactive() -> tuple[dict[str, Any], list[str]]:
        from ...agents import morning_briefing as mb

        payload = mb.proactive_status(context)
        s = payload["settings"]
        recent = payload.get("recent") or []
        last = recent[0] if recent else None
        health = payload.get("health") or {}
        degraded = health.get("degraded") or []
        data = {"available": True,
                "master": bool(s.get("proactive_enabled")),
                "briefing": bool(s.get("proactive_briefing")),
                "watchers": bool(s.get("proactive_watchers")),
                "counts_24h": payload.get("counts") or {},
                "health_ok": not degraded,
                "health_degraded": degraded,
                "last_delivery_state": last.get("delivery_state") if last else None,
                "last_title": (last.get("title") or "")[:80] if last else None}
        text = [f"proactive: {'ON' if data['master'] else 'OFF'} "
                f"(briefing={'ON' if data['briefing'] else 'OFF'}, "
                f"watchers={'ON' if data['watchers'] else 'OFF'})"]
        if degraded:
            text.append(f"          health: DEGRADED — {degraded[0]}")
        if last:
            text.append(f"          last send: {last.get('delivery_state')} — "
                        f"{(last.get('title') or '')[:60]}")
        else:
            text.append("          no proactive sends recorded yet")
        return data, text

    def _power() -> tuple[dict[str, Any], list[str]]:
        from ...agents.power import power_mode_for

        s = power_mode_for(context).status()
        data = {"available": True, "active": bool(s.get("active")),
                "unlocked_by": s.get("unlocked_by")}
        text = [f"power:    {'ACTIVE' if data['active'] else 'locked'}"
                + (f" (unlocked by {data['unlocked_by']})" if data["active"] else "")]
        return data, text

    def _research_loop() -> tuple[dict[str, Any], list[str]]:
        from ...agents.research_loop import status as _rl_status

        s = _rl_status(context)
        job = s.get("job") or {}
        gates = s.get("gates") or {}
        data = {"available": True,
                "scheduled": bool(job.get("scheduled")),
                "enabled": bool(job.get("enabled")),
                "feature_research": bool(gates.get("feature_research")),
                "pending_proposals": s.get("pending_proposals", 0)}
        text = [f"research: {'scheduled' if data['scheduled'] else 'not scheduled'}"
                + (f" (every {job.get('interval_hours')}h"
                   + ("" if data["enabled"] else ", disabled") + ")"
                   if data["scheduled"] else "")
                + f", feature={'on' if data['feature_research'] else 'off'}"
                + f", {data['pending_proposals']} pending proposals"]
        return data, text

    for name, fn in (("system", _system), ("database", _database),
                     ("queue", _queue), ("missions", _missions),
                     ("memory", _memory), ("proactive", _proactive),
                     ("power", _power), ("research_loop", _research_loop)):
        data, text = _status_section(name, fn)
        sections[name] = data
        blocks.append(text)

    payload = {"command": "status", "sections": sections}
    rendered = "\n".join(line for block in blocks for line in block)
    _emit(args, payload, rendered)
    return 0
