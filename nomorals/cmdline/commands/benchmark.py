"""``nm benchmark`` — benchmark suites."""

from __future__ import annotations

import argparse
import time
from typing import Any
from pathlib import Path
from ..emit import _emit



class _BuildersBuildBackend:
    """Real builders-backed build backend for the K3 scoreboard.

    L7-only by construction (this module is the CLI layer): scaffold the
    app, serve it when it is an HTTP app, smoke-test it, and report one
    boolean per stage. Timeouts surface as ``timed_out`` and fail every
    stage they touch — they are never reported as skips.
    """

    def run(self, kind: str, workdir: str) -> dict[str, Any]:
        import shutil
        import time as _time
        from ...core.ids import new_short_id

        from ...builders.run import serve
        from ...builders.scaffold import scaffold
        from ...builders.smoke import smoke_test

        started = _time.monotonic()
        name = f"k3bench-{kind}-{new_short_id(length=8)}"
        stages: dict[str, bool] = {}
        detail = ""
        timed_out = False
        project_dir = None
        try:
            res = scaffold(kind, name, workdir)
            project_dir = res.project_dir
            stages["scaffold"] = True
            if kind == "webapp":
                handle = serve(project_dir, startup_timeout=15.0)
                stages["serve"] = True
                try:
                    smoke = smoke_test(handle, timeout=10.0)
                finally:
                    handle.stop()
            else:
                smoke = smoke_test(project_dir, timeout=15.0)
            stages["smoke"] = bool(smoke.ok)
            detail = "; ".join(
                f"{c.name}: {'ok' if c.ok else 'FAIL'}"
                for c in smoke.checks)[:300]
        except Exception as exc:  # noqa: BLE001 — a failed stage is data
            msg = str(exc).lower()
            timed_out = ("timed out" in msg or "timeout" in msg
                         or "deadline" in msg)
            detail = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            if project_dir is not None:
                shutil.rmtree(project_dir, ignore_errors=True)
        return {
            "scaffold": stages.get("scaffold", False),
            "serve": stages.get("serve", False),
            "smoke": stages.get("smoke", False),
            "seconds": _time.monotonic() - started,
            "detail": detail,
            "timed_out": timed_out,
        }


def _cmd_benchmark(args: argparse.Namespace, context: Any) -> int:
    """`nm benchmark run|list|compare` — the K3 scoreboard CLI."""
    from ...agents import benchmark as bench_mod

    action = args.benchmark_action or "run"  # bare `nm benchmark` = run all
    db = getattr(context, "db", None)

    if action == "list":
        rows = bench_mod.list_runs(
            db, suite=getattr(args, "suite", "") or "",
            limit=getattr(args, "limit", 20) or 20)
        payload = {"runs": rows}
        if not rows:
            _emit(args, payload, "no benchmark runs recorded yet")
            return 0
        lines = ["id            ts                    suite  mode              "
                 "overall  passed  secs",
                 "-" * 88]
        for r in rows:
            ts = time.strftime("%Y-%m-%d %H:%M",
                               time.localtime(r["ts"] or 0))
            overall = (f"{r['overall']:.3f}" if r["overall"] is not None
                       else "n/a")
            lines.append(
                f"{r['id'][:12]:<13} {ts}  {str(r['suite'])[:6]:<6} "
                f"{str(r['mode'])[:17]:<17} {overall:<7} "
                f"{r['passed']}/{r['total']:<6} {r['seconds']:.0f}")
        _emit(args, payload, "\n".join(lines))
        return 0

    if action == "compare":
        cmp = bench_mod.compare_runs(db, args.run_a, args.run_b)
        if not cmp.get("ok"):
            _emit(args, cmp, f"compare failed — {cmp.get('error')}")
            return 2
        lines = [f"compare {cmp['a']['id'][:12]} (A) vs "
                 f"{cmp['b']['id'][:12]} (B) — mode {cmp['a']['mode']}"]
        oa, ob = cmp["overall_a"], cmp["overall_b"]
        oad = cmp["overall_delta"]
        lines.append(
            f"overall: {oa if oa is not None else 'n/a'} -> "
            f"{ob if ob is not None else 'n/a'} "
            f"({'+' if oad and oad > 0 else ''}{oad if oad is not None else 'n/a'})")
        for name, d in cmp["dimensions"].items():
            delta = d["delta"]
            mark = ("+" if delta and delta > 0 else "") + str(delta)
            lines.append(f"  {name:<14} {d['a_passed']:<9} -> "
                         f"{d['b_passed']:<9}  delta {mark}")
        _emit(args, cmp, "\n".join(lines))
        return 0

    # action == "run"
    suite_arg = (getattr(args, "suite", None) or "all").strip().lower()
    suites: list[str] | None = None
    if suite_arg != "all":
        suites = [s.strip() for s in suite_arg.split(",") if s.strip()]
        unknown = [s for s in suites if s not in bench_mod.list_suites()]
        if unknown:
            _emit(args, {"error": "unknown suite", "unknown": unknown},
                  f"unknown suite(s): {', '.join(unknown)} — available: "
                  f"{', '.join(bench_mod.list_suites())}")
            return 2
    report = bench_mod.run_scoreboard(
        context, suites=suites, limit=getattr(args, "limit", 0) or 0,
        build_backend=_BuildersBuildBackend())
    run_id = bench_mod.save_run(db, report)
    payload = report.as_dict()
    payload["saved_as"] = run_id
    export = (getattr(args, "export", "") or "").strip()
    if export:
        try:
            run = bench_mod.get_run(db, run_id) if run_id else None
            Path(export).expanduser().write_text(
                bench_mod.export_run_json(run or payload),
                encoding="utf-8")
        except OSError as exc:
            _emit(args, payload, f"run {run_id or '(unsaved)'} done, but "
                  f"export failed: {exc}")
            return 2
    lines = [
        f"K3 scoreboard — run {run_id or '(not persisted)'}",
        f"mode: {report.mode}   provider: {report.provider or '-'}   "
        f"{report.seconds:.1f}s",
    ]
    if report.mode == bench_mod.MODE_SELF_TEST:
        lines.append("note: no live LLM — harness self-test, NOT a model score")
    for name, dim in report.scores.items():
        score = "n/a" if dim.score is None else f"{dim.score:.3f}"
        lines.append(f"  {name:<14} {score:<6} ({dim.passed}/{dim.total})")
        for d in dim.details[:4]:
            mark = "?" if d.get("pass") is None else (
                "ok" if d.get("pass") else "FAIL")
            lines.append(f"    [{mark}] {d.get('detail', '')[:100]}")
    overall = (f"{report.overall:.3f}" if report.overall is not None
               else "n/a")
    lines.append(f"overall: {overall}")
    _emit(args, payload, "\n".join(lines))
    return 0
