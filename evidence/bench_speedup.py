"""Measure HybridExecutor's TRUE parallel speedup (not the reported metric).

Runs 8 CPU-bound tasks three ways and prints the wall-clock speedup of each
against a serial baseline:
  1. serial baseline (same work, one task at a time, in-process)
  2. HybridExecutor at DEFAULTS (the T1.1 fix target: >= 1.5x on 2 cores)
  3. HybridExecutor(processes=2) (explicit)

Usage: python3 evidence/bench_speedup.py
Exit 0 when defaults achieve >= 1.5x true speedup, 1 otherwise.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.agents.runtime import HybridExecutor, TaskGraph, Task, TaskKind


def _burn(n: int) -> int:
    # CPU-bound, picklable, deterministic. ~0.25s on a typical core.
    acc = 0
    for i in range(n):
        acc += (i * i) % 97
    return acc


def _serial_wall(n_tasks: int, n: int) -> float:
    start = time.perf_counter()
    for _ in range(n_tasks):
        _burn(n)
    return time.perf_counter() - start


def _executor_wall(n_tasks: int, n: int, **kwargs) -> tuple[float, object, int]:
    executor = HybridExecutor(**kwargs)
    try:
        graph = TaskGraph()
        for i in range(n_tasks):
            graph.add_task(f"cpu{i}", _burn, n, kind=TaskKind.CPU)
        start = time.perf_counter()
        report = executor.run(graph)
        wall = time.perf_counter() - start
        return wall, report, executor.processes
    finally:
        executor.shutdown(wait=False, timeout=5.0)


def main() -> int:
    n_tasks, n = 8, 2_000_000
    serial = _serial_wall(n_tasks, n)
    print(f"serial baseline: {serial:.2f}s")
    default_wall, default_report, default_procs = _executor_wall(n_tasks, n)
    default_speedup = serial / default_wall if default_wall else 0.0
    print(f"defaults: wall={default_wall:.2f}s TRUE speedup={default_speedup:.2f}x "
          f"reported={default_report.speedup_vs_serial} "
          f"processes={default_procs}")
    explicit_wall, explicit_report, _ = _executor_wall(n_tasks, n, processes=2)
    explicit_speedup = serial / explicit_wall if explicit_wall else 0.0
    print(f"processes=2: wall={explicit_wall:.2f}s TRUE speedup={explicit_speedup:.2f}x "
          f"reported={explicit_report.speedup_vs_serial}")
    ok = default_speedup >= 1.5
    print("PASS" if ok else "FAIL: defaults did not reach 1.5x true speedup")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
