"""Control: naive multiprocessing.Pool(2) on the same 8 CPU tasks.

Proves the machine itself can parallelize, so a poor HybridExecutor
number is the executor's fault, not the hardware's.
Usage: python3 evidence/bench_control_multiprocessing.py
"""
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_speedup import _burn, _serial_wall


def main() -> int:
    n_tasks, n = 8, 2_000_000
    serial = _serial_wall(n_tasks, n)
    start = time.perf_counter()
    with multiprocessing.Pool(2) as pool:
        pool.map(_burn, [n] * n_tasks)
    wall = time.perf_counter() - start
    print(f"serial baseline: {serial:.2f}s")
    print(f"Pool(2): wall={wall:.2f}s TRUE speedup={serial / wall:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
