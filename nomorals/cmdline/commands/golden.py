"""``nm golden`` — run the golden end-to-end mission drills."""

from __future__ import annotations

import argparse
import json
from typing import Any

from ..emit import _emit
from ...llm.benchmarks import BenchmarkDB
from ...missions.golden import (
    GOLDEN_MISSIONS,
    GoldenRunner,
    list_golden_missions,
)


def _cmd_golden(args: argparse.Namespace, context: Any) -> int:
    action = getattr(args, "golden_action", "list") or "list"

    if action == "list":
        missions = list_golden_missions()
        if getattr(args, "json", False):
            print(json.dumps(missions, indent=2))
            return 0
        for m in missions:
            print(f"{m['key']}: {m['name']}")
            print(f"  {m['goal']}")
            print(f"  steps: {', '.join(m['steps'])}")
        return 0

    db = getattr(context, "db", None)
    runner = GoldenRunner(db, benchmark_db=BenchmarkDB(db))

    if action == "run":
        key = getattr(args, "target", "")
        if key not in GOLDEN_MISSIONS:
            print(f"unknown golden mission {key!r}; "
                  f"choose from {', '.join(sorted(GOLDEN_MISSIONS))}")
            return 2
        result = runner.run(key, long=bool(getattr(args, "long", False)))
        _emit(args, result.to_dict(), _render_result(result))
        return 0 if result.ok else 1

    if action == "resume":
        mission_id = getattr(args, "target", "")
        if not mission_id:
            print("golden resume needs a mission id")
            return 2
        result = runner.resume(mission_id)
        _emit(args, result.to_dict(), _render_result(result))
        return 0 if result.ok else 1

    print(f"unknown golden action {action!r}")
    return 2


def _render_result(result: Any) -> str:
    lines = [f"golden {result.key}: {result.status} "
             f"({result.seconds:.1f}s, benchmark row {result.benchmark_row})"]
    for step in result.steps:
        mark = "ok  " if step["ok"] else "FAIL"
        extra = " (repaired)" if step.get("repaired") else ""
        lines.append(f"  {mark} {step['step']}{extra}: {step['detail']}")
    if result.error:
        lines.append(f"error: {result.error}")
    return "\n".join(lines)
