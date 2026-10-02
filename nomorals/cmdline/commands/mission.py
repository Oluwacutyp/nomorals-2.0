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
    print(f"mission: unknown action {action}", file=sys.stderr)
    return 2
