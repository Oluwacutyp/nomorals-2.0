"""Test and build runners for the code workspace.

Both runners report honestly: a nonzero exit is never reported as success,
and when the output cannot be parsed into pass/fail counts the counts come
back as ``-1`` while ``ok`` reflects the exit code alone.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .workspace import WorkspaceError

__all__ = ["run_tests", "run_build"]

_TEST_TIMEOUT = 300
_BUILD_TIMEOUT = 600
_OUTPUT_CAP = 20_000

_PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed")
_PYTEST_FAILED_RE = re.compile(r"(\d+)\s+failed")
_UNITTEST_RAN_RE = re.compile(r"Ran (\d+) tests?")
_UNITTEST_FAILED_RE = re.compile(r"FAILED\s*\(failures=(\d+),\s*errors=(\d+)\)")
_MAKE_TARGET_RE = re.compile(r"^([\w][\w.\-/]*)\s*:(?![:=])")


def _cap(text: str) -> str:
    if len(text) > _OUTPUT_CAP:
        return text[:_OUTPUT_CAP] + "\n...[output truncated]"
    return text


def _run_cmd(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError(
            f"{' '.join(args)} timed out after {timeout}s") from exc
    except OSError as exc:
        raise WorkspaceError(f"could not run {' '.join(args)}: {exc}") from exc


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


def run_tests(root: str | Path, selector: str = "") -> dict[str, Any]:
    """Run the test suite in ``root`` and report honest counts.

    Runner selection: pytest when ``python -m pytest --version`` works,
    else ``make test`` when the Makefile defines a ``test`` target, else
    ``python -m unittest discover``.
    """
    base = _resolve_root(root)
    runner = ""
    parse = None
    if _pytest_available():
        args = [sys.executable, "-m", "pytest", "-q"]
        if selector:
            args.append(selector)
        runner, parse = "pytest", _parse_pytest
    elif (base / "Makefile").is_file() and _make_has_target(base / "Makefile", "test"):
        args = ["make", "test"]
        runner = "make test"
    else:
        args = [sys.executable, "-m", "unittest"]
        args += [selector] if selector else ["discover"]
        runner, parse = "unittest", _parse_unittest
    proc = _run_cmd(args, base, _TEST_TIMEOUT)
    output = _cap(proc.stdout + proc.stderr)
    passed, failed = parse(output) if parse else (-1, -1)
    return {"runner": runner, "ok": proc.returncode == 0,
            "passed": passed, "failed": failed, "output": output}


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


def run_build(root: str | Path, target: str = "") -> dict[str, Any]:
    """Build the project in ``root``.

    ``make <target or first target>`` when a Makefile exists; otherwise
    ``python -m build`` when ``pyproject.toml`` exists and the ``build``
    module is importable; otherwise fail fast with WorkspaceError.
    """
    base = _resolve_root(root)
    makefile = base / "Makefile"
    if makefile.is_file():
        tgt = target or _first_make_target(makefile)
        if not tgt:
            raise WorkspaceError("Makefile has no usable targets")
        proc = _run_cmd(["make", tgt], base, _BUILD_TIMEOUT)
    elif (base / "pyproject.toml").is_file():
        if importlib.util.find_spec("build") is None:
            raise WorkspaceError(
                "no build system found: pyproject.toml present but the "
                "'build' module is not importable")
        proc = _run_cmd([sys.executable, "-m", "build"], base, _BUILD_TIMEOUT)
    else:
        raise WorkspaceError("no build system found")
    return {"ok": proc.returncode == 0,
            "output": _cap(proc.stdout + proc.stderr)}
