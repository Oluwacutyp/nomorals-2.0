"""End-to-end verification: the "after each coding mission verify and report" primitive.

:func:`build_and_verify` runs the full lifecycle for one scaffolded
project -- scaffold, install deps, run its test suite, serve + smoke
test (HTTP kinds) or ``--help`` smoke (CLI kinds), export + verify --
and returns a :class:`BuildReport` describing exactly what changed,
what broke, and what remains.  Steps never raise: failures are
captured into the report.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Policy
from .export import export_project, verify_export
from .install import install_deps
from .run import ServeError, run_config, serve
from .scaffold import ScaffoldResult, scaffold
from .smoke import SmokeResult, smoke_test

_log = get_logger(__name__)

__all__ = ["BuildStep", "BuildReport", "build_and_verify", "run_project_tests"]


@dataclass
class BuildStep:
    """One lifecycle step and its outcome."""

    name: str
    ok: bool
    detail: str = ""
    elapsed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok,
                "detail": self.detail, "elapsed": round(self.elapsed, 3)}


@dataclass
class BuildReport:
    """Full lifecycle report for one project."""

    kind: str
    name: str
    project_dir: Path | None = None
    steps: list[BuildStep] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    @property
    def broken(self) -> list[str]:
        return [s.name for s in self.steps if not s.ok]

    def summary(self) -> str:
        lines = [f"build_and_verify {self.kind}/{self.name}: "
                 f"{'OK' if self.ok else 'BROKEN'} ({self.elapsed:.1f}s)"]
        for step in self.steps:
            mark = "ok" if step.ok else "FAIL"
            first = step.detail.splitlines()[0] if step.detail else ""
            lines.append(f"  [{mark}] {step.name} ({step.elapsed:.1f}s) {first}")
        if self.broken:
            lines.append("broken: " + ", ".join(self.broken))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "name": self.name,
            "project_dir": str(self.project_dir) if self.project_dir else None,
            "ok": self.ok, "broken": self.broken,
            "elapsed": round(self.elapsed, 3),
            "steps": [s.to_dict() for s in self.steps],
        }


def run_project_tests(result: ScaffoldResult, timeout: float = 120.0) -> BuildStep:
    """Run the rendered project's own test suite; capture the outcome."""
    started = time.monotonic()
    try:
        proc = subprocess.run(
            result.test_cmd, cwd=str(result.project_dir),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return BuildStep("tests", False,
                         f"test suite timed out after {timeout:.0f}s",
                         time.monotonic() - started)
    except OSError as exc:
        return BuildStep("tests", False, f"could not spawn tests: {exc}",
                         time.monotonic() - started)
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    detail = "\n".join(tail[-12:])
    ok = proc.returncode == 0
    if not ok:
        detail = f"tests exited {proc.returncode}:\n{detail}"
    return BuildStep("tests", ok, detail, time.monotonic() - started)


def build_and_verify(kind: str, name: str, dest: str | Path, *,
                     policy: Policy | None = None,
                     confirmation: str | None = None,
                     export_dir: str | Path | None = None,
                     startup_timeout: float = 10.0) -> BuildReport:
    """Scaffold ``kind``/``name`` into ``dest`` and verify the full lifecycle.

    Steps: scaffold -> install_deps -> run project tests -> serve +
    smoke_test (HTTP kinds) or --help smoke (CLI/console kinds) ->
    export + verify_export.  Every step's outcome lands in the report;
    nothing raises except a catastrophic scaffold failure, which is
    itself captured as a failed step.

    ``confirmation`` is forwarded to :func:`install_deps`: mint it with
    ``policy.issue_confirmation(Capability.EXEC_INSTALL)`` when the
    policy's ``exec.install`` rule requires confirmation.
    """
    started = time.monotonic()
    report = BuildReport(kind=kind, name=name)
    dest = Path(dest).expanduser()

    # 1. scaffold
    step_started = time.monotonic()
    try:
        result = scaffold(kind, name, dest)
    except Exception as exc:  # noqa: BLE001 -- captured into the report
        report.steps.append(BuildStep("scaffold", False, f"{type(exc).__name__}: {exc}",
                                      time.monotonic() - step_started))
        report.elapsed = time.monotonic() - started
        return report
    report.project_dir = result.project_dir
    report.steps.append(BuildStep(
        "scaffold", True,
        f"{len(result.files)} files -> {result.project_dir}",
        time.monotonic() - step_started))

    # 2. install deps (policy-gated; denial is a report entry, not a crash)
    step_started = time.monotonic()
    try:
        install = install_deps(result.project_dir, policy=policy,
                               confirmation=confirmation)
        report.steps.append(BuildStep("install_deps", install.ok, install.detail,
                                      time.monotonic() - step_started))
    except Exception as exc:  # noqa: BLE001
        report.steps.append(BuildStep("install_deps", False,
                                      f"{type(exc).__name__}: {exc}",
                                      time.monotonic() - step_started))

    # 3. project tests
    report.steps.append(run_project_tests(result))

    # 4. serve + smoke (HTTP) or --help smoke (CLI/console)
    step_started = time.monotonic()
    try:
        config = run_config(result.project_dir)
    except Exception as exc:  # noqa: BLE001
        report.steps.append(BuildStep("smoke", False, f"run_config: {exc}",
                                      time.monotonic() - step_started))
        config = None
    if config is not None:
        if config.kind == "http":
            try:
                with serve(result.project_dir, port=0,
                           startup_timeout=startup_timeout) as handle:
                    smoke: SmokeResult = smoke_test(handle, timeout=startup_timeout)
                detail = "; ".join(
                    f"{c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}".strip()
                    for c in smoke.checks)
                report.steps.append(BuildStep("serve+smoke", smoke.ok, detail,
                                              time.monotonic() - step_started))
            except ServeError as exc:
                report.steps.append(BuildStep("serve+smoke", False,
                                              f"{exc}\nstderr:\n{exc.stderr}",
                                              time.monotonic() - step_started))
            except Exception as exc:  # noqa: BLE001
                report.steps.append(BuildStep("serve+smoke", False,
                                              f"{type(exc).__name__}: {exc}",
                                              time.monotonic() - step_started))
        else:
            smoke = smoke_test(result.project_dir, timeout=30.0)
            detail = "; ".join(
                f"{c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}".strip()
                for c in smoke.checks)
            report.steps.append(BuildStep("smoke", smoke.ok, detail,
                                          time.monotonic() - step_started))

    # 5. export + verify round trip
    step_started = time.monotonic()
    try:
        exported = export_project(result.project_dir,
                                  dest=export_dir or result.project_dir.parent)
        verified = verify_export(exported.archive)
        ok = verified.ok
        detail = (f"{exported.archive.name} ({exported.bytes} bytes, "
                  f"{len(exported.files)} files)")
        if not ok:
            detail += " -- verify problems: " + "; ".join(verified.problems)
        report.steps.append(BuildStep("export", ok, detail,
                                      time.monotonic() - step_started))
    except Exception as exc:  # noqa: BLE001
        report.steps.append(BuildStep("export", False, f"{type(exc).__name__}: {exc}",
                                      time.monotonic() - step_started))

    report.elapsed = time.monotonic() - started
    _log.info("\n%s", report.summary())
    return report
