"""``nm render`` — the mission render queue, from the command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..emit import _emit


def _parse_spec(raw: str) -> Any:
    """A render spec: JSON text, or ``@path/to/spec.json``."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("nm render needs --spec <json> or @spec.json")
    if raw.startswith("@"):
        path = Path(raw[1:]).expanduser()
        raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def _parse_pack(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--pack must be a JSON object")
    return data


def _status_text(status: dict[str, Any]) -> str:
    lines = [f"render {status.get('mission_id')} — {status.get('status')}"]
    lines.append(
        "  stages: "
        + ", ".join(status.get("completed_stages") or [])
        + (f" (pending: {', '.join(status.get('pending_stages') or [])})"
           if status.get("pending_stages") else " (all done)"))
    for stage, output in (status.get("outputs") or {}).items():
        lines.append(f"  {stage}: {output}")
    lines.append(
        f"  budget: spent {status.get('spent_wall', 0.0):.1f}s wall / "
        f"{status.get('spent_tokens', 0)} tokens of "
        f"{status.get('budget_wall', 0.0) or '∞'}s / "
        f"{status.get('budget_tokens', 0) or '∞'}")
    if status.get("budget_exhausted"):
        lines.append("  budget exhausted")
    return "\n".join(lines)


def _cmd_render(args, context):
    """Mission render queue: enqueue, advance, status, list."""
    from ...missions import MissionStore
    from ...missions.render_driver import (
        default_render_cache_dir,
        run_render_tick_once,
        stage_render_fn,
    )
    from ...missions.render_queue import (
        advance_render,
        enqueue_render,
        render_mission_status,
    )

    store = MissionStore(context.db)

    if getattr(args, "list", False):
        missions = store.list(tag="render", active_only=True, limit=50)
        payload = [render_mission_status(store, m.id) for m in missions]
        lines = [f"{len(payload)} queued render(s)"]
        for status in payload:
            lines.append(
                f"  {status['mission_id']} — {status['status']} "
                f"({', '.join(status['completed_stages']) or 'no stages yet'})")
        _emit(args, {"renders": payload}, "\n".join(lines))
        return 0

    if getattr(args, "status", ""):
        status = render_mission_status(store, args.status)
        _emit(args, status, _status_text(status))
        return 0

    if getattr(args, "advance", ""):
        mission_id = args.advance
        cache_dir = default_render_cache_dir(
            getattr(getattr(context, "settings", None), "home", None))
        result = advance_render(
            store, mission_id,
            render_fn=lambda *, stage, state: stage_render_fn(
                stage, state, cache_dir=cache_dir))
        _emit(args, result,
              f"render {mission_id} — {result.get('status')} "
              f"(stage {result.get('stage') or '—'}"
              f"{', done' if result.get('done') else ''})")
        return 0 if result.get("status") != "failed" else 1

    if getattr(args, "source", None):
        try:
            spec = _parse_spec(getattr(args, "spec", "") or "")
            pack = _parse_pack(getattr(args, "pack", None))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            _emit(args, {"ok": False, "error": str(exc)},
                  f"nm render: bad spec/pack — {exc}")
            return 1
        mission = enqueue_render(
            store,
            sources=list(args.source),
            spec=spec,
            pack=pack,
            budget_wall=float(getattr(args, "budget_wall", 0.0) or 0.0),
            budget_tokens=int(getattr(args, "budget_tokens", 0) or 0),
            name=getattr(args, "name", "") or "")
        status = render_mission_status(store, mission.id)
        _emit(args, status,
              f"render queued — {mission.id}\n"
              f"  cache key: {status.get('cache_key')}\n"
              "  the render tick advances it (proxy, then full); "
              "use --advance to drive one stage now")
        return 0

    if getattr(args, "tick", False):
        result = run_render_tick_once(context, store=store)
        _emit(args, result,
              f"render tick — advanced {len(result.get('advanced') or [])}, "
              f"done {len(result.get('done') or [])}, "
              f"failed {len(result.get('failed') or [])}")
        return 0 if result.get("ok") else 1

    _emit(args, {"started": False},
          "nothing to do — pass --source … --spec … to queue a render, "
          "--list to see queued renders, --status <id>, --advance <id>, "
          "or --tick to drive one tick now")
    return 0
