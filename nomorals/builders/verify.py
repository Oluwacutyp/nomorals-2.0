"""End-to-end verification: the "after each coding mission verify and report" primitive.

:func:`build_and_verify` runs the full lifecycle for one scaffolded
project -- scaffold, install deps, validate sources, run its test
suite, serve + smoke test (HTTP kinds) or ``--help`` smoke (CLI kinds),
export + verify -- and returns a :class:`BuildReport` describing
exactly what changed, what broke, and what remains.  Steps never
raise: failures are captured into the report.  Steps are selectable
(``steps=[...]`` / ``skip=[...]``, GitHub-Actions style) and the report
can be saved, reloaded, rendered as Markdown, or printed with the
god-tier styled ``fancy_summary()``.
"""

from __future__ import annotations

import json
import py_compile
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

__all__ = ["BuildStep", "BuildReport", "build_and_verify", "run_project_tests",
           "STEP_NAMES"]

#: The lifecycle phases in execution order (selectable via steps=/skip=).
#: The smoke phase is reported as "serve+smoke" for HTTP kinds and
#: "smoke" for CLI/console kinds — both names select it.
STEP_NAMES = ("scaffold", "install_deps", "validate", "tests",
              "smoke", "serve+smoke", "export")

#: Default phases: exactly the historical pipeline (validate is opt-in
#: via steps=[...] so existing callers see no change).
_DEFAULT_PHASES = ("scaffold", "install_deps", "tests", "smoke", "export")


@dataclass
class BuildStep:
    """One lifecycle step and its outcome."""

    name: str
    ok: bool
    detail: str = ""
    elapsed: float = 0.0
    #: How many attempts the step took (1 normally; >1 when retried).
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok,
                "detail": self.detail, "elapsed": round(self.elapsed, 3),
                "attempts": self.attempts}


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

    def save(self, path: str | Path) -> Path:
        """Persist the report as JSON (reload with :meth:`load`)."""
        path = Path(path).expanduser()
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BuildReport":
        """Reload a report saved with :meth:`save`."""
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        report = cls(kind=data.get("kind", ""), name=data.get("name", ""),
                     elapsed=data.get("elapsed", 0.0))
        if data.get("project_dir"):
            report.project_dir = Path(data["project_dir"])
        for s in data.get("steps", []):
            report.steps.append(BuildStep(
                name=s.get("name", ""), ok=bool(s.get("ok")),
                detail=s.get("detail", ""),
                elapsed=float(s.get("elapsed", 0.0)),
                attempts=int(s.get("attempts", 1))))
        return report

    def as_markdown(self) -> str:
        """Markdown rendering — for chat, issues, PR bodies."""
        lines = [f"# build_and_verify `{self.kind}/{self.name}` — "
                 f"{'✅ OK' if self.ok else '❌ BROKEN'}",
                 "", f"*{self.elapsed:.1f}s total*",
                 "", "| step | result | time | detail |",
                 "|---|---|---|---|"]
        for s in self.steps:
            mark = "✅" if s.ok else "❌"
            first = s.detail.splitlines()[0] if s.detail else ""
            lines.append(f"| {s.name} | {mark} | {s.elapsed:.1f}s | {first} |")
        if self.project_dir:
            lines += ["", f"project: `{self.project_dir}`"]
        if self.broken:
            lines += ["", "**broken:** " + ", ".join(f"`{b}`" for b in self.broken)]
        return "\n".join(lines)

    def fancy_summary(self, theme: str | None = None) -> str:
        """God-tier styled rendering (Backstage-step-list style)."""
        from .style import banner, render_kv, render_steps, resolve_theme

        th = resolve_theme(theme)
        head = banner(f"build {self.kind}/{self.name}",
                      "OK — all steps green" if self.ok
                      else f"BROKEN: {', '.join(self.broken)}",
                      theme=th)
        body = render_steps([s.to_dict() for s in self.steps], theme=th)
        kv = render_kv([
            ("elapsed", f"{self.elapsed:.1f}s"),
            ("project", str(self.project_dir) if self.project_dir else "—"),
        ], theme=th)
        return "\n".join([head, body, kv])


def validate_sources(project_dir: str | Path) -> BuildStep:
    """Byte-compile every ``.py`` file in the project (static validation).

    The AppBuilder validator applied to scaffolded projects: catches
    syntax errors before the test suite even runs.  Projects with no
    Python files pass trivially (``skipped`` in the detail).
    """
    started = time.monotonic()
    project_dir = Path(project_dir).expanduser().resolve()
    py_files = sorted(p for p in project_dir.rglob("*.py")
                      if p.is_file() and "__pycache__" not in p.parts
                      and ".git" not in p.parts)
    if not py_files:
        return BuildStep("validate", True, "skipped: no Python files",
                         time.monotonic() - started)
    bad: list[str] = []
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except (py_compile.PyCompileError, OSError) as exc:
            bad.append(f"{path.relative_to(project_dir)}: {exc}")
    ok = not bad
    detail = (f"{len(py_files)} files compile ok" if ok
              else f"{len(bad)}/{len(py_files)} failed: " + "; ".join(bad[:5]))
    return BuildStep("validate", ok, detail, time.monotonic() - started)


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
                     startup_timeout: float = 10.0,
                     steps: list[str] | None = None,
                     skip: list[str] | None = None,
                     fail_fast: bool = False,
                     scaffold_kwargs: dict[str, Any] | None = None,
                     install_kwargs: dict[str, Any] | None = None) -> BuildReport:
    """Scaffold ``kind``/``name`` into ``dest`` and verify the full lifecycle.

    Steps (in order): scaffold -> install_deps -> validate (opt-in) ->
    tests -> serve+smoke (HTTP kinds) or smoke (CLI/console kinds) ->
    export.  ``steps=[...]`` runs only those phases (in canonical order;
    ``"smoke"`` and ``"serve+smoke"`` both select the smoke phase);
    ``skip=[...]`` drops phases; ``fail_fast=True`` stops at the first
    failure.  With no ``steps``/``skip`` the pipeline is exactly the
    historical five-step run.  ``scaffold_kwargs`` / ``install_kwargs``
    are forwarded to :func:`scaffold` / :func:`install_deps` (e.g.
    ``git_init=True``, ``installer="uv"``, ``venv=".venv"``).

    Every step's outcome lands in the report; nothing raises except a
    catastrophic scaffold failure, which is itself captured as a failed
    step.

    ``confirmation`` is forwarded to :func:`install_deps`: mint it with
    ``policy.issue_confirmation(Capability.EXEC_INSTALL)`` when the
    policy's ``exec.install`` rule requires confirmation.
    """
    started = time.monotonic()
    report = BuildReport(kind=kind, name=name)
    dest = Path(dest).expanduser()

    if steps is None:
        phases = list(_DEFAULT_PHASES)
    else:
        phases = [p for p in ("scaffold", "install_deps", "validate",
                              "tests", "smoke", "export")
                  if p in steps or (p == "smoke" and "serve+smoke" in steps)]
    phases = [p for p in phases if p not in (skip or ())]
    if not phases:
        raise ValueError("no steps selected: steps/skip removed everything")
    wanted = set(phases)

    def add(step: BuildStep) -> bool:
        """Append a step; return False when fail_fast should stop."""
        report.steps.append(step)
        return not (fail_fast and not step.ok)

    # 1. scaffold
    result: ScaffoldResult | None = None
    if "scaffold" in wanted:
        step_started = time.monotonic()
        try:
            result = scaffold(kind, name, dest, **(scaffold_kwargs or {}))
        except Exception as exc:  # noqa: BLE001 -- captured into the report
            report.steps.append(BuildStep(
                "scaffold", False, f"{type(exc).__name__}: {exc}",
                time.monotonic() - step_started))
            report.elapsed = time.monotonic() - started
            return report
        report.project_dir = result.project_dir
        if not add(BuildStep(
                "scaffold", True,
                f"{len(result.files)} files -> {result.project_dir}",
                time.monotonic() - step_started)):
            report.elapsed = time.monotonic() - started
            return report
    else:
        # steps without scaffold: operate on the existing directory
        candidate = dest / name
        if candidate.is_dir():
            report.project_dir = candidate.resolve()
            result = ScaffoldResult(kind=kind, name=name,
                                    project_dir=report.project_dir)

    if result is None or report.project_dir is None:
        report.elapsed = time.monotonic() - started
        return report

    # 2. install deps (policy-gated; denial is a report entry, not a crash)
    if "install_deps" in wanted:
        step_started = time.monotonic()
        try:
            install = install_deps(result.project_dir, policy=policy,
                                   confirmation=confirmation,
                                   **(install_kwargs or {}))
            if not add(BuildStep("install_deps", install.ok, install.detail,
                                 time.monotonic() - step_started)):
                report.elapsed = time.monotonic() - started
                return report
        except Exception as exc:  # noqa: BLE001
            if not add(BuildStep("install_deps", False,
                                 f"{type(exc).__name__}: {exc}",
                                 time.monotonic() - step_started)):
                report.elapsed = time.monotonic() - started
                return report

    # 3. static validation (py_compile every .py)
    if "validate" in wanted:
        if not add(validate_sources(result.project_dir)):
            report.elapsed = time.monotonic() - started
            return report

    # 4. project tests
    if "tests" in wanted:
        if not add(run_project_tests(result)):
            report.elapsed = time.monotonic() - started
            return report

    # 5. serve + smoke (HTTP) or --help smoke (CLI/console)
    if "smoke" in wanted:
        step_started = time.monotonic()
        smoke_name = "serve+smoke"  # refined below once kind is known
        try:
            config = run_config(result.project_dir)
            smoke_name = "serve+smoke" if config.kind == "http" else "smoke"
        except Exception as exc:  # noqa: BLE001
            add(BuildStep(smoke_name, False, f"run_config: {exc}",
                          time.monotonic() - step_started))
            config = None
        if config is not None:
            if config.kind == "http":
                try:
                    with serve(result.project_dir, port=0,
                               startup_timeout=startup_timeout) as handle:
                        smoke: SmokeResult = smoke_test(handle,
                                                        timeout=startup_timeout)
                    detail = "; ".join(
                        f"{c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}".strip()
                        for c in smoke.checks)
                    if not add(BuildStep(smoke_name, smoke.ok, detail,
                                         time.monotonic() - step_started)):
                        report.elapsed = time.monotonic() - started
                        return report
                except ServeError as exc:
                    if not add(BuildStep(smoke_name, False,
                                         f"{exc}\nstderr:\n{exc.stderr}",
                                         time.monotonic() - step_started)):
                        report.elapsed = time.monotonic() - started
                        return report
                except Exception as exc:  # noqa: BLE001
                    if not add(BuildStep(smoke_name, False,
                                         f"{type(exc).__name__}: {exc}",
                                         time.monotonic() - step_started)):
                        report.elapsed = time.monotonic() - started
                        return report
            else:
                smoke = smoke_test(result.project_dir, timeout=30.0)
                detail = "; ".join(
                    f"{c.name}: {'ok' if c.ok else 'FAIL'} {c.detail}".strip()
                    for c in smoke.checks)
                if not add(BuildStep(smoke_name, smoke.ok, detail,
                                     time.monotonic() - step_started)):
                    report.elapsed = time.monotonic() - started
                    return report

    # 6. export + verify round trip
    if "export" in wanted:
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
            add(BuildStep("export", ok, detail,
                          time.monotonic() - step_started))
        except Exception as exc:  # noqa: BLE001
            add(BuildStep("export", False, f"{type(exc).__name__}: {exc}",
                          time.monotonic() - step_started))

    report.elapsed = time.monotonic() - started
    _log.info("\n%s", report.summary())
    return report
