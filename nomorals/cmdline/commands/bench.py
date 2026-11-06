"""``nm bench`` — the system benchmark, on a real command (AA-39).

``nomorals.bench`` is the system benchmark (native kernels vs pure
Python: vector top-k, BPE train/encode, training step, memory
extract) with a saved-baseline workflow and an honesty audit that
re-derives every reported speedup from the raw timings.  It was
orphaned: the only benchmark-flavoured CLI names were ``nm
benchmark`` (the K3 scoreboard in ``agents.benchmark``) and ``nm
skill benchmark`` (per-skill reliability in ``skills.bench``) —
different modules measuring different things — so the system
benchmark could be reached only from Python.  This command is its
surface; the other two keep their jobs.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _cmd_bench(args: argparse.Namespace, settings: Any) -> int:
    from ... import bench as bench_mod

    quick = bool(getattr(args, "quick", False))
    results = bench_mod.run_benchmarks(quick=quick)
    audit = bench_mod.honest_speedup_report(results)

    saved_to = ""
    if getattr(args, "save", False):
        saved_to = str(bench_mod.save_baseline(results, quick=quick))
    baseline = None if saved_to else bench_mod.load_baseline(quick=quick)

    if getattr(args, "json", False):
        print(json.dumps({
            "quick": quick,
            "results": results,
            "honest": audit["honest"],
            "mismatches": audit["mismatches"],
            "saved_to": saved_to,
            "baseline_compared": baseline is not None,
        }, indent=2, default=str))
        return 0 if audit["honest"] else 1

    print(f"system benchmark ({'quick' if quick else 'full'}):")
    for name, payload in results.items():
        if not isinstance(payload, dict):
            print(f"  {name}: {payload}")
            continue
        bits = [f"backend={payload.get('backend', '?')}"]
        if "python_ms" in payload and "native_ms" in payload:
            bits.append(f"python {float(payload['python_ms']):.2f} ms")
            bits.append(f"native {float(payload['native_ms']):.2f} ms")
            bits.append(f"speedup {float(payload.get('speedup', 0)):.2f}x")
        else:
            bits.append(", ".join(f"{k}={v}" for k, v in payload.items()
                                  if k != "backend"))
        if baseline and isinstance(baseline.get(name), dict):
            base = baseline[name]
            if "python_ms" in base and "python_ms" in payload:
                delta = (float(payload["python_ms"])
                         - float(base["python_ms"]))
                pct = (delta / float(base["python_ms"]) * 100.0
                       if base["python_ms"] else 0.0)
                bits.append(f"vs baseline python {pct:+.1f}%")
        print(f"  {name}: " + " · ".join(bits))
    if saved_to:
        print(f"baseline saved → {saved_to}")
    elif baseline is None:
        print("no saved baseline for this workload size "
              "(run `nm bench --save` to record one)")
    if audit["honest"]:
        print("honesty audit: every reported speedup re-derives from "
              "the raw timings ✓")
    else:
        print("honesty audit: MISMATCH — reported speedups that do "
              "not re-derive:")
        for m in audit["mismatches"]:
            print(f"  • {m.get('name')}: claimed {m.get('claimed')}x, "
                  f"achieved {m.get('achieved')}x")
    return 0 if audit["honest"] else 1
