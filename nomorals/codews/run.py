"""Test and build runners for the code workspace.

Both runners report honestly: a nonzero exit is never reported as success,
and when the output cannot be parsed into pass/fail counts the counts come
back as ``-1`` while ``ok`` reflects the exit code alone.

Runner selection is marker-file driven (:func:`detect_stack`): ``package.json``
scripts, ``go.mod``, ``Cargo.toml``, pytest config markers, and Makefile
targets are honored before falling back to ``unittest`` discovery.  Pytest
runs emit JUnit XML (``pytest --junitxml``) parsed by :func:`parse_junit_xml`
into structured per-test results — the same lingua franca CI systems use.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .workspace import WorkspaceError

__all__ = ["run_tests", "run_build", "detect_stack", "parse_junit_xml"]

_log = get_logger(__name__)

_TEST_TIMEOUT = 300
_BUILD_TIMEOUT = 600
_OUTPUT_CAP = 20_000


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break the runners (fail-open telemetry, fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

_PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed")
_PYTEST_FAILED_RE = re.compile(r"(\d+)\s+failed")
_PYTEST_FAILED_LINE_RE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)
_UNITTEST_RAN_RE = re.compile(r"Ran (\d+) tests?")
_UNITTEST_FAILED_RE = re.compile(r"FAILED\s*\(failures=(\d+),\s*errors=(\d+)\)")
_MAKE_TARGET_RE = re.compile(r"^([\w][\w.\-/]*)\s*:(?![:=])")


def _cap(text: str) -> str:
    if len(text) > _OUTPUT_CAP:
        return text[:_OUTPUT_CAP] + "\n...[output truncated]"
    return text


def _run_cmd(args: list[str], cwd: Path, timeout: int,
             env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    try:
        return subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, env=merged)
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError(
            f"{' '.join(args)} timed out after {timeout}s") from exc
    except OSError as exc:
        raise WorkspaceError(f"could not run {' '.join(args)}: {exc}") from exc


def _exec(args: list[str], cwd: Path, timeout: int,
          env: dict[str, str] | None) -> subprocess.CompletedProcess[str]:
    """Call :func:`_run_cmd` without the ``env`` argument when it is None,
    preserving the historical 3-argument call shape."""
    if env is None:
        return _run_cmd(args, cwd, timeout)
    return _run_cmd(args, cwd, timeout, env)


def _resolve_root(root: str | Path) -> Path:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise WorkspaceError(f"not a directory: {base}")
    return base


def _pytest_available() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _parse_pytest(output: str) -> tuple[int, int]:
    """Parse pytest's summary line; (-1, -1) when it cannot be found."""
    tail = "\n".join(output.splitlines()[-15:])
    passed_m = _PYTEST_PASSED_RE.search(tail)
    failed_m = _PYTEST_FAILED_RE.search(tail)
    if not passed_m and not failed_m:
        return -1, -1
    return (int(passed_m.group(1)) if passed_m else 0,
            int(failed_m.group(1)) if failed_m else 0)


def _parse_pytest_failed(output: str) -> list[str]:
    """Node IDs from pytest's ``short test summary info`` (``FAILED x``)."""
    return _PYTEST_FAILED_LINE_RE.findall(output)


def _parse_unittest(output: str) -> tuple[int, int]:
    ran_m = _UNITTEST_RAN_RE.search(output)
    if not ran_m:
        return -1, -1
    ran = int(ran_m.group(1))
    failed_m = _UNITTEST_FAILED_RE.search(output)
    if failed_m:
        failed = int(failed_m.group(1)) + int(failed_m.group(2))
        return ran - failed, failed
    if re.search(r"^OK", output, re.MULTILINE):
        return ran, 0
    return -1, -1


def _make_has_target(makefile: Path, target: str) -> bool:
    for line in makefile.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith((".", "#")):
            continue
        m = _MAKE_TARGET_RE.match(line)
        if m and m.group(1) == target:
            return True
    return False


# ── JUnit XML ─────────────────────────────────────────────────────────────
def parse_junit_xml(path: str | Path) -> dict[str, Any]:
    """Parse a JUnit/xUnit result XML file (``pytest --junitxml``,
    surefire, jest-junit, …) into structured results.

    Returns ``{"tests", "passed", "failed", "errors", "skipped",
    "duration_s", "cases"}``; each case is ``{"nodeid", "classname",
    "name", "duration_s", "outcome", "message"}``.  ``message`` is the
    first line of the failure/error text, capped for sanity.
    """
    tree = ET.parse(str(path))
    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    cases: list[dict[str, Any]] = []
    tests = failed = errors = skipped = 0
    duration = 0.0
    for suite in suites:
        try:
            duration += float(suite.get("time", 0) or 0)
        except ValueError:
            pass
        for case in suite.iter("testcase"):
            tests += 1
            classname = case.get("classname", "")
            name = case.get("name", "")
            try:
                dur = float(case.get("time", 0) or 0)
            except ValueError:
                dur = 0.0
            outcome = "passed"
            message = ""
            for tag in ("failure", "error", "skipped"):
                el = case.find(tag)
                if el is not None:
                    outcome = {"failure": "failed", "error": "error",
                               "skipped": "skipped"}[tag]
                    raw = (el.get("message", "") or "") + "\n" + (el.text or "")
                    message = raw.strip().split("\n")[0][:500]
                    break
            if outcome == "failed":
                failed += 1
            elif outcome == "error":
                errors += 1
            elif outcome == "skipped":
                skipped += 1
            cases.append({
                "nodeid": f"{classname}::{name}" if classname else name,
                "classname": classname,
                "name": name,
                "duration_s": dur,
                "outcome": outcome,
                "message": message,
            })
    return {
        "tests": tests,
        "passed": tests - failed - errors - skipped,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "duration_s": duration,
        "cases": cases,
    }


# ── stack detection ───────────────────────────────────────────────────────
def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def detect_stack(root: str | Path) -> dict[str, Any]:
    """Detect the project's test and build commands from marker files.

    Marker-file first, then manifest contents, then directory conventions —
    the same order the best stack-detection tooling uses.  Returns::

        {"language": str,
         "test": {"runner": str, "args": [str]} | None,
         "build": {"runner": str, "args": [str]} | None}

    ``args`` excludes the selector/target: callers append those.
    """
    base = _resolve_root(root)
    test: dict[str, Any] | None = None
    build: dict[str, Any] | None = None
    language = "unknown"

    pkg = base / "package.json"
    if pkg.is_file():
        language = "node"
        manifest = _read_json(pkg)
        scripts = manifest.get("scripts", {})
        dev_deps = {**manifest.get("devDependencies", {}),
                    **manifest.get("dependencies", {})}
        if "test" in scripts:
            if "vitest" in dev_deps:
                test = {"runner": "vitest", "args": ["npx", "vitest", "run"]}
            elif "jest" in dev_deps:
                test = {"runner": "jest", "args": ["npx", "jest"]}
            elif "mocha" in dev_deps:
                test = {"runner": "mocha", "args": ["npx", "mocha"]}
            else:
                test = {"runner": "npm test", "args": ["npm", "test", "--"]}
        if "build" in scripts:
            build = {"runner": "npm run build",
                     "args": ["npm", "run", "build"]}

    if (base / "go.mod").is_file():
        language = "go"
        test = test or {"runner": "go test", "args": ["go", "test", "./..."]}
        build = build or {"runner": "go build",
                          "args": ["go", "build", "./..."]}

    if (base / "Cargo.toml").is_file():
        language = "rust"
        test = test or {"runner": "cargo test", "args": ["cargo", "test"]}
        build = build or {"runner": "cargo build", "args": ["cargo", "build"]}

    if (base / "build.gradle.kts").is_file() or (base / "build.gradle").is_file():
        language = "java"
        gradle = [str(base / "gradlew")] if (base / "gradlew").is_file() \
            else ["gradle"]
        test = test or {"runner": "gradle test", "args": [*gradle, "test"]}
        build = build or {"runner": "gradle build",
                          "args": [*gradle, "build"]}
    elif (base / "pom.xml").is_file():
        language = "java"
        mvn = [str(base / "mvnw")] if (base / "mvnw").is_file() else ["mvn"]
        test = test or {"runner": "mvn test", "args": [*mvn, "-q", "test"]}
        build = build or {"runner": "mvn package",
                          "args": [*mvn, "-q", "package", "-DskipTests"]}

    py_markers = (base / "pyproject.toml", base / "pytest.ini",
                  base / "setup.cfg", base / "tox.ini")
    has_pytest_cfg = any(p.is_file() for p in py_markers)
    has_test_files = any(base.rglob("test_*.py")) or any(base.rglob("*_test.py"))
    if has_pytest_cfg or has_test_files or (base / "setup.py").is_file():
        language = "python" if language == "unknown" else language
        if test is None:
            if _pytest_available():
                test = {"runner": "pytest",
                        "args": [sys.executable, "-m", "pytest", "-q"]}
            else:
                test = {"runner": "unittest",
                        "args": [sys.executable, "-m", "unittest", "discover"]}

    makefile = base / "Makefile"
    if makefile.is_file():
        if test is None and _make_has_target(makefile, "test"):
            test = {"runner": "make test", "args": ["make", "test"]}
        if build is None:
            first = _first_make_target(makefile)
            if first:
                build = {"runner": f"make {first}", "args": ["make", first]}

    if test is None and has_test_files and _pytest_available():
        test = {"runner": "pytest",
                "args": [sys.executable, "-m", "pytest", "-q"]}

    if build is None and (base / "pyproject.toml").is_file():
        # PEP 517: pyproject-only projects build via `python -m build`,
        # gated on the module being importable.
        if importlib.util.find_spec("build") is not None:
            build = {"runner": "python -m build",
                     "args": [sys.executable, "-m", "build"]}
    if build is None and (base / "setup.py").is_file():
        build = {"runner": "setup.py build",
                 "args": [sys.executable, "setup.py", "build"]}
    return {"language": language, "test": test, "build": build}


def run_tests(root: str | Path, selector: str = "",
              *, timeout: int = _TEST_TIMEOUT,
              env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the test suite in ``root`` and report honest counts.

    The runner comes from :func:`detect_stack` (pytest → ``--junitxml``
    into a temp file for structured per-test results, else ``make test``,
    npm/jest/vitest, ``go test``, ``cargo test``, gradle/maven, and finally
    ``unittest`` discovery).  ``selector`` is appended runner-appropriately
    (pytest node ID, ``go test -run``, cargo filter, …).
    """
    base = _resolve_root(root)
    stack = detect_stack(base)
    spec = stack["test"]
    if spec is None:
        # Legacy fallback: the historical contract runs pytest whenever it
        # is importable, else unittest discovery — even in marker-less dirs.
        if _pytest_available():
            spec = {"runner": "pytest",
                    "args": [sys.executable, "-m", "pytest", "-q"]}
        else:
            spec = {"runner": "unittest",
                    "args": [sys.executable, "-m", "unittest", "discover"]}
    runner, args = spec["runner"], list(spec["args"])
    junit_path: str | None = None
    tmpdir = None
    if runner == "pytest":
        tmpdir = tempfile.TemporaryDirectory(prefix="codews-junit-")
        junit_path = str(Path(tmpdir.name) / "junit.xml")
        args.append(f"--junitxml={junit_path}")
        if selector:
            args.append(selector)
    elif runner == "go test":
        if selector:
            args = ["go", "test", "-run", selector, "./..."]
    elif runner in ("cargo test", "jest", "vitest", "mocha"):
        if selector:
            args.append(selector)
    elif runner == "npm test":
        if selector:
            args.append(selector)
    elif runner == "unittest":
        if selector:
            args = [sys.executable, "-m", "unittest", selector]
    elif runner in ("make test", "gradle test", "mvn test"):
        pass  # selectors are not portable here; run the suite target
    started = time.monotonic()
    try:
        proc = _exec(args, base, timeout, env)
    finally:
        if tmpdir is not None:
            pass  # parsed below, then cleaned
    duration_s = time.monotonic() - started
    output = _cap(proc.stdout + proc.stderr)

    passed = failed = -1
    failed_tests: list[str] = []
    failures: list[dict[str, Any]] = []
    skipped = 0
    junit: dict[str, Any] | None = None
    if junit_path and Path(junit_path).is_file():
        try:
            junit = parse_junit_xml(junit_path)
            passed, failed = junit["passed"], junit["failed"] + junit["errors"]
            skipped = junit["skipped"]
            failed_tests = [c["nodeid"] for c in junit["cases"]
                            if c["outcome"] in ("failed", "error")]
            failures = [
                {"nodeid": c["nodeid"], "classname": c["classname"],
                 "duration_s": c["duration_s"], "message": c["message"]}
                for c in junit["cases"]
                if c["outcome"] in ("failed", "error")][:50]
        except ET.ParseError:
            junit = None
    if tmpdir is not None:
        tmpdir.cleanup()
    if junit is None:
        if runner == "pytest":
            passed, failed = _parse_pytest(output)
            failed_tests = _parse_pytest_failed(output)
        elif runner == "unittest":
            passed, failed = _parse_unittest(output)

    result = {
        "runner": runner,
        "command": args,
        "ok": proc.returncode == 0,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "failed_tests": failed_tests,
        "failures": failures,
        "duration_s": round(duration_s, 2),
        "output": output,
    }
    _emit("codews.tests.run", {
        "root": str(base),
        "runner": runner,
        "ok": result["ok"],
        "passed": passed,
        "failed": failed,
        "duration_s": result["duration_s"],
    })
    return result


def _first_make_target(makefile: Path) -> str:
    for line in makefile.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith((".", "#")) or not line.strip():
            continue
        if "=" in line.split(":", 1)[0]:
            continue  # variable assignment, not a rule
        m = _MAKE_TARGET_RE.match(line)
        if m:
            return m.group(1)
    return ""


def run_build(root: str | Path, target: str = "",
              *, timeout: int = _BUILD_TIMEOUT,
              env: dict[str, str] | None = None) -> dict[str, Any]:
    """Build the project in ``root``.

    The build command comes from :func:`detect_stack`: ``make``,
    ``npm run build``, ``cargo build``, ``go build``, gradle, maven, or
    PEP 517 ``python -m build``.  ``target`` overrides the detected target
    (used for ``make``).  Fails fast with :class:`WorkspaceError` when no
    build system is detected.
    """
    base = _resolve_root(root)
    stack = detect_stack(base)
    spec = stack["build"]
    if spec is not None:
        runner, args = spec["runner"], list(spec["args"])
        if target and runner.startswith("make"):
            args = ["make", target]
    elif (base / "pyproject.toml").is_file():
        # Reached only when the `build` module is not importable
        # (detect_stack gates on find_spec).
        raise WorkspaceError(
            "no build system found: pyproject.toml present but the "
            "'build' module is not importable")
    else:
        raise WorkspaceError("no build system found")
    started = time.monotonic()
    proc = _exec(args, base, timeout, env)
    duration_s = time.monotonic() - started
    result = {"runner": runner,
              "command": args,
              "ok": proc.returncode == 0,
              "duration_s": round(duration_s, 2),
              "output": _cap(proc.stdout + proc.stderr)}
    target_label = (args[1] if runner.startswith("make") and len(args) > 1
                    else target)
    _emit("codews.build.run", {
        "root": str(base),
        "runner": runner,
        "target": target_label,
        "ok": result["ok"],
        "duration_s": result["duration_s"],
    })
    return result
