"""``nm missions`` — mission list and control."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from ...core.errors import NoMoralsError
from ..emit import _emit



def _cmd_missions(args: argparse.Namespace, context: Any) -> int:
    from ...missions import MissionRunner, MissionStore

    store = MissionStore(context.db)
    runner = MissionRunner(context, store=store)
    reflect = not args.no_reflect

    if args.start:
        result = runner.start(
            args.start,
            max_iterations=args.max_iterations,
            budget_wall=args.budget_wall,
            budget_tokens=args.budget_tokens,
            reflect=reflect,
        )
        _emit(args, result.to_dict(), _render_result(result))
        return 0 if result.ok else 1

    if args.resume:
        result = runner.resume(
            args.resume, max_iterations=args.max_iterations, reflect=reflect
        )
        _emit(args, result.to_dict(), _render_result(result))
        return 0 if result.ok else 1

    if args.resume_all:
        results = runner.resume_all(max_iterations=args.max_iterations)
        payload = [r.to_dict() for r in results]
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
        elif not results:
            print("no interrupted missions")
        for result in results:
            print(_render_result(result))
        return 0

    if args.pause:
        try:
            mission = store.set_status(args.pause, "paused", "paused from the CLI")
        except (KeyError, ValueError, NoMoralsError) as exc:
            print(f"missions: {exc}", file=sys.stderr)
            return 2
        _emit(args, mission.to_dict(), f"paused {mission.id} — {mission.name}")
        return 0

    if args.cancel:
        try:
            mission = store.set_status(args.cancel, "cancelled", "cancelled from the CLI")
        except (KeyError, ValueError, NoMoralsError) as exc:
            print(f"missions: {exc}", file=sys.stderr)
            return 2
        _emit(args, mission.to_dict(), f"cancelled {mission.id} — {mission.name}")
        return 0

    if args.resume_status:
        try:
            mission = store.set_status(args.resume_status, "running", "")
        except (KeyError, ValueError, NoMoralsError) as exc:
            print(f"missions: {exc}", file=sys.stderr)
            return 2
        p = store.progress(mission.id)
        _emit(args, {"mission": mission.to_dict(), "progress": p},
              f"running {mission.id} — {mission.name}: "
              f"{p.get('steps_done', 0)}/{p.get('total_steps', 0)} steps "
              f"({p.get('percent', 0):.0f}%)")
        return 0

    if args.show:
        # reconcile on read: a dead runner must not report "running"
        store.reconcile(args.show)
        mission = store.get(args.show)
        history = store.checkpoint_history(mission.id, limit=10)
        payload = {
            "mission": mission.to_dict(),
            "checkpoints": [c.to_row() for c in history],
            "reflections": store.reflections(mission.id),
        }
        if args.json:
            print(json.dumps(payload, indent=2, default=str))
            return 0
        prog = store.progress(mission.id)
        print(f"{mission.id}  [{mission.status}]")
        print(f"  goal:       {mission.goal}")
        print(f"  steps:      {prog['steps_done']}/{prog['total_steps']} "
              f"({prog['percent']:.0f}%) "
              + (f"— current: {prog['current_step']}" if prog["current_step"] else ""))
        print(f"  iterations: {mission.iterations}   success: {mission.success}")
        print(f"  spent:      {mission.spent_wall:.1f}s / {mission.spent_tokens} tokens")
        print(f"  completed:  {mission.state.get('completed_steps') or []}")
        print(f"  checkpoints: {[c.label for c in history]}")
        if mission.status in ("running", "paused"):
            print(f"  resumable: yes — `nm missions --resume {mission.id}`")
        return 0

    rows = store.list(status=args.status, limit=50)
    resumable = store.resumable()
    resumable_ids = {m.id for m in resumable}
    payload = {"stats": store.stats(), "missions": [m.to_dict() for m in rows],
               "resumable": sorted(resumable_ids)}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    stats = store.stats()
    print(f"missions: {stats['total']} total, {stats['active']} active, "
          f"{stats['checkpoints']} checkpoints, "
          f"{len(resumable_ids)} interrupted (resumable)")
    for mission in rows:
        resume_mark = " ↺" if mission.id in resumable_ids else "  "
        print(f"{resume_mark}{mission.id}  [{mission.status:<9}] "
              f"it={mission.iterations} "
              f"success={mission.success}  {mission.goal[:50]}")
    if resumable_ids and not args.status:
        print("interrupted — resume with `nm missions --resume <id>` "
              "or all at once with `nm missions --resume-all`:")
        for mission in resumable:
            point = store.latest_checkpoint(mission.id)
            ckpt = f" (checkpoint: {point.label or point.id})" if point else ""
            print(f"  ↺ {mission.id}{ckpt}")
    return 0


def _render_result(result: Any) -> str:
    lines = [
        f"mission {result.mission_id} -> {result.status} "
        f"(success={result.success}, {result.iterations} iterations, {result.seconds:.1f}s)"
    ]
    if result.resumed_from:
        lines.append(f"  resumed from checkpoint: {result.resumed_from}")
    for step in result.steps:
        mark = "ok " if step.ok else "ERR"
        lines.append(f"  [{mark}] {step.step} ({step.seconds:.2f}s) {step.detail[:80]}")
    if result.error:
        lines.append(f"  error: {result.error}")
    for lesson in result.lessons:
        lines.append(f"  lesson: {lesson}")
    return "\n".join(lines)
