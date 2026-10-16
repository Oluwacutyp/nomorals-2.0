"""Subprocess probe for T1.3's wall-time parallelism test.

Run as ``python3 -m tests._walltime_probe`` from the repo root. Prints one
JSON line: ``{"serial": ..., "wall": ..., "pids": [...], "done": N}`` — or
``{"skipped": reason}`` when no process pool is available.

Why a subprocess: the wall-time ratio is a measurement of a global
resource (CPU). In the shared unittest process, background threads leaked
by earlier tests compete for the 2 cores and push the ratio from ~0.60
(clean process) to ~0.76 (shared process), flaking the 0.75 bar. A fresh
process isolates the measurement. The probe duplicates the 3-line burn
helper instead of importing tests.test_agents so it stays standalone.
"""

from __future__ import annotations

import json
import os
import sys
import time


def _burn_with_pid(iterations: int) -> tuple[int, int]:
    total = 0
    for i in range(iterations):
        total += (i * i) % 7
    return total, os.getpid()


def main() -> int:
    from nomorals.agents.runtime import HybridExecutor
    from nomorals.core.tasks import TaskGraph, TaskKind

    n_tasks, n = 8, 4_000_000
    serial_start = time.perf_counter()
    for _ in range(n_tasks):
        _burn_with_pid(n)
    serial = time.perf_counter() - serial_start

    executor = HybridExecutor()
    try:
        if executor._process_pool is None:
            print(json.dumps({"skipped": "process pool unavailable"}))
            return 0
        graph = TaskGraph()
        for i in range(n_tasks):
            graph.add_task(f"cpu{i}", _burn_with_pid, n, kind=TaskKind.CPU)
        report = executor.run(graph)
    finally:
        executor.shutdown(wait=False, timeout=5.0)
    pids = sorted({result[1] for result in report.results.values()})
    print(json.dumps({
        "serial": serial,
        "wall": report.wall_seconds,
        "pids": pids,
        "done": report.done,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
