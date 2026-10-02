"""``nm mission`` — single-mission control."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from ..emit import _emit



def _cmd_mission(args: argparse.Namespace, context: Any) -> int:
    """Mission control CLI: the goal portfolio at a glance."""
    from ...agents.mission import MissionControl

    mc = MissionControl(context)
    action = getattr(args, "action", "plan") or "plan"
    if action == "plan":
        plan = mc.plan()
        counts = plan.get("counts", {})
        cad = plan.get("cadence", {})
        eff = cad.get("effective_hours", cad.get("base_hours", "?"))
        lines = [
            f"mission plan — ready {counts.get('ready', 0)}, "
            f"blocked {counts.get('blocked', 0)}, "
            f"paused {counts.get('paused', 0)}, done {counts.get('done', 0)}",
            f"heartbeat: every {eff}h (base {cad.get('base_hours', '?')}h, "
            f"{'adaptive' if cad.get('adaptive') else 'fixed'})",
        ]
        nxt = plan.get("next")
        if nxt:
            lines.append(f"next: {nxt.get('id', '')} — {nxt.get('title', '')}")
        for g in plan.get("ready", [])[:10]:
            lines.append(f"  ready:   {g.get('id', '')}  {g.get('title', '')}")
        for b in plan.get("blocked", [])[:10]:
            goal = b.get("goal") or b
            lines.append(f"  blocked: {goal.get('id', '')}  "
                         f"{goal.get('title', '')}")
        ranking = plan.get("ranking") or []
        if ranking:
            try:
                from ...agents.cognition import ModelBudget

                budget = ModelBudget(context).report()
                need = sum(int(r.get("est_calls") or 0) for r in ranking)
                if budget.get("unlimited"):
                    lines.append(f"budget: portfolio (~{need} calls) fits "
                                 "today — unlimited")
                elif need <= int(budget.get("remaining", 0)):
                    lines.append(f"budget: portfolio (~{need} calls) fits "
                                 "today — "
                                 f"{budget.get('remaining', 0)} calls left")
                else:
                    lines.append(f"budget: portfolio (~{need} calls) does "
                                 "NOT fit today — "
                                 f"{budget.get('remaining', 0)} left; cap it "
                                 "up with `nm autonomy budget --cap N`")
            except Exception:  # noqa: BLE001 — fit is a courtesy line
                pass
            lines.append("ranking (expected value):")
            for i, r in enumerate(ranking, 1):
                est = int(r.get("est_calls") or 0)
                lines.append(f"  {i}. {r.get('id', '')} {r.get('title', '')} "
                             f"— EV {r.get('expected_value', 0)} "
                             f"(risk {r.get('risk', '?')}, p{r.get('priority', 0)}, "
                             f"~{est} calls)")
        _emit(args, plan, "\n".join(lines))
        return 0
    if action == "next":
        out = mc.next()
        nxt = out.get("next")
        _emit(args, out,
              f"next: {nxt.get('id', '')} — {nxt.get('title', '')}" if nxt
              else "no goal ready")
        return 0
    if action == "status":
        plan = mc.plan()
        payload = {"counts": plan.get("counts", {}),
                   "cadence": plan.get("cadence", {})}
        _emit(args, payload, json.dumps(payload, indent=2, default=str))
        return 0
    if action == "health":
        # live-mission watchdog: a RUNNING mission whose newest checkpoint
        # went stale lost its worker — flag it for restart or cancel
        from ...missions.runner import MissionRunner

        h = MissionRunner(context).health()
        lines = [f"mission health — {len(h['active'])} active, "
                 f"{len(h['stuck'])} stuck "
                 f"(stale after {int(h['stuck_after_seconds'])}s)"]
        for e in h["active"]:
            age = e.get("checkpoint_age_seconds")
            age_s = "never" if age is None else f"{age:g}s ago"
            lines.append(f"  {'STUCK' if e['stuck'] else 'live '}: "
                         f"{e['mission_id'][:8]} {e['status']} — {e['name']} "
                         f"(last checkpoint {age_s})")
        for mid in h["stuck"]:
            lines.append(f"  restart or cancel stuck mission: {mid}")
        _emit(args, h, "\n".join(lines))
        return 0
    if action == "replay":
        return _cmd_mission_replay(args, context)
    if action == "redrive":
        return _cmd_mission_redrive(args, context)
    print(f"mission: unknown action {action}", file=sys.stderr)
    return 2


def _resolve_mission_ref(ref: str, context: Any) -> str:
    """Resolve a mission id or unambiguous prefix to the full mission id.

    Falls back to the raw reference when it matches no known mission — the
    timeline may hold events for missions since deleted from the store.
    """
    from ...core.ids import resolve_id_prefix
    from ...missions import MissionStore

    ref = (ref or "").strip()
    if not ref:
        return ""
    try:
        ids = [m.id for m in MissionStore(context.db).list(limit=500)]
    except Exception:  # noqa: BLE001 - resolution is a courtesy, not a gate
        return ref
    res = resolve_id_prefix(ref, ids)
    if res.outcome in ("exact", "unique"):
        return res.matches[0]
    if res.outcome == "ambiguous":
        raise ValueError(
            "mission reference {!r} is ambiguous ({} matches); "
            "use a longer prefix".format(ref, len(res.matches)))
    return ref


def _open_timeline(context: Any):
    from ...os.timeline import Timeline

    db_path = getattr(getattr(context, "db", None), "path", None)
    return Timeline(db_path)


def _cmd_mission_replay(args: argparse.Namespace, context: Any) -> int:
    """``nm mission replay --mission <id>`` — retell a mission from the
    persisted event timeline. Strictly read-only."""
    from ...os.replay import replay_mission

    try:
        mission_id = _resolve_mission_ref(getattr(args, "mission", ""), context)
    except ValueError as exc:
        print(f"mission replay: {exc}", file=sys.stderr)
        return 2
    if not mission_id:
        print("mission replay needs --mission <id>", file=sys.stderr)
        return 2
    tl = _open_timeline(context)
    try:
        report = replay_mission(tl, mission_id)
    finally:
        tl.close()
    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2, default=str,
                         ensure_ascii=False))
        return 0
    if not report.event_count:
        print(f"no timeline events recorded for mission {mission_id}")
        return 0
    print(report.narrative())
    return 0


def _cmd_mission_redrive(args: argparse.Namespace, context: Any) -> int:
    """``nm mission redrive --mission <id> --confirm`` — re-drive a terminal
    mission's recorded plan as a new mission, linked derived_from the
    original in the artifact graph. Refuses without --confirm and refuses
    live missions."""
    from pathlib import Path

    from ...missions import MissionStore
    from ...os.replay import RedriveRefused, redrive_mission
    from ...storage.artifacts import ArtifactStore
    from ...storage.blob import BlobStore

    try:
        mission_id = _resolve_mission_ref(getattr(args, "mission", ""), context)
    except ValueError as exc:
        print(f"mission redrive: {exc}", file=sys.stderr)
        return 2
    if not mission_id:
        print("mission redrive needs --mission <id>", file=sys.stderr)
        return 2
    store = MissionStore(context.db)
    db = getattr(context, "db", None)
    db_path = getattr(db, "path", None)
    blob_dir = Path(db_path).parent / "blobs" if db_path else Path("data/blobs")
    artifacts = ArtifactStore(db, BlobStore(db, blob_dir))
    try:
        report = redrive_mission(
            store, mission_id,
            confirm=bool(getattr(args, "confirm", False)),
            artifact_store=artifacts,
            note="redrive from the CLI",
        )
    except RedriveRefused as exc:
        print(f"mission redrive refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - report the failure, don't trace
        print(f"mission redrive failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        _emit(args, report.to_dict(),
              json.dumps(report.to_dict(), indent=2, default=str))
        return 0
    lines = [
        f"redrove {report.original_mission_id} as {report.new_mission_id}",
        f"  plan steps carried over: {report.plan_steps}",
        f"  os state: {report.os_state}",
    ]
    if report.artifact_id:
        lines.append(f"  redrive record: artifact://{report.artifact_id} "
                     f"(derived_from {len(report.derived_from_artifact_ids)} "
                     "original artifact(s))")
    lines.append(f"  start it with: nm missions --resume {report.new_mission_id}")
    print("\n".join(lines))
    return 0
