"""Pytest-aware test runner (audit Phase B).

``run_tests`` runs the suite with pytest when it is available (``-x -q``,
per-test failure ids parsed from the short summary) and falls back to the
repo's own ``unittest discover`` command otherwise — same result schema
either way.

``changed_only=True`` selects the tests that touch files changed in the
working tree (``git diff --name-only`` + untracked files), mapped to test
modules by filename stem.  The coding loop calls this instead of a raw
accept command.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["run_tests", "select_changed_tests", "format_test_result",
           "register"]

_FAILED_LINE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)
_SUMMARY_COUNT = re.compile(r"(\d+)\s+(passed|failed|error)")


def _parse_pytest_counts(out: str) -> dict[str, int]:
    """Pass/fail/error counts from pytest's short summary line.

    pytest prints e.g. ``2 passed``, ``1 failed, 2 passed``, ``1 error``.
    Note: ``findall`` returns (number, word) pairs, so the dict must be
    keyed by the word — ``dict(pairs)`` would key by the number and every
    lookup would silently miss.
    """
    counts: dict[str, int] = {}
    for num, word in _SUMMARY_COUNT.findall(out):
        counts[word] = counts.get(word, 0) + int(num)
    if not counts:
        # pytest 9.x in quiet mode (`-q`) omits the summary line when
        # stdout is not a TTY — the run still succeeded, so fall back to
        # the progress characters (one per test: . pass, s skip, x/X
        # xfail/xpass, F fail, E error) on progress lines ("[100%]").
        for line in out.splitlines():
            if "%]" not in line:
                continue
            for ch in line:
                if ch == ".":
                    counts["passed"] = counts.get("passed", 0) + 1
                elif ch == "F":
                    counts["failed"] = counts.get("failed", 0) + 1
                elif ch == "E":
                    counts["error"] = counts.get("error", 0) + 1
    return counts
_UNITTEST_FAIL = re.compile(r"^(FAIL|ERROR):\s+(\S+)", re.MULTILINE)
_UNITTEST_COUNTS = re.compile(r"FAILED\s*\(([^)]*)\)")


def _repo_root(repo: str | None) -> Path:
    return Path(repo).expanduser().resolve() if repo else Path.cwd().resolve()


def _git_changed_files(root: Path) -> list[str]:
    """Changed + untracked files, repo-relative. [] when not a git repo."""
    git = shutil.which("git")
    if not git:
        return []
    try:
        diff = subprocess.run(
            [git, "-C", str(root), "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=30)
        status = subprocess.run(
            [git, "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001 — git is best-effort here
        _log.debug("git changed-files probe failed: %s", exc)
        return []
    if diff.returncode != 0:
        return []
    files = [ln.strip() for ln in diff.stdout.splitlines() if ln.strip()]
    if status.returncode == 0:
        for ln in status.stdout.splitlines():
            m = re.match(r"^\?\?\s+(.+)$", ln)
            if m:
                files.append(m.group(1).strip())
    # de-dup, keep order
    seen: set[str] = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def select_changed_tests(repo: str | None = None) -> list[str]:
    """Test files (repo-relative) covering the working tree's changed files.

    Heuristic: a changed module ``pkg/mod.py`` maps to test files whose
    filename contains the module stem (``tests/test_mod*.py``,
    ``tests/**/test_*mod*.py``); a changed test file maps to itself.
    """
    root = _repo_root(repo)
    changed = _git_changed_files(root)
    if not changed:
        return []
    test_dirs = [d for d in (root / "tests", root / "test") if d.is_dir()]
    if not test_dirs:
        return []
    selected: list[str] = []
    for rel in changed:
        p = Path(rel)
        if "test" in p.name.lower() and p.suffix == ".py":
            candidate = root / rel
            if candidate.is_file():
                selected.append(rel)
            continue
        stem = p.stem
        if not stem or stem.startswith("_"):
            continue
        for tdir in test_dirs:
            for tf in tdir.rglob("test_*.py"):
                if stem in tf.stem and tf.name != "__init__.py":
                    selected.append(str(tf.relative_to(root)))
    seen: set[str] = set()
    return [s for s in selected if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]


def _has_pytest() -> bool:
    # The runner invokes `sys.executable -m pytest`, so availability is about
    # the module being importable — not about a `pytest` script on PATH
    # (pip does not always install one, e.g. user-site installs).
    return importlib.util.find_spec("pytest") is not None


def format_test_result(tres: dict[str, Any]) -> str:
    """One screen of test-runner output for the fix loop and the journal."""
    lines = [f"{tres.get('runner', '?')}: {tres.get('passed', 0)} passed, "
             f"{len(tres.get('failed', []))} failed "
             f"({tres.get('seconds', 0)}s)"]
    for fail in tres.get("failed", [])[:3]:
        lines.append(f"FAILED {fail.get('test_id')}")
        if fail.get("error_snippet"):
            lines.append(str(fail["error_snippet"])[:800])
    note = tres.get("note")
    if note:
        lines.append(str(note))
    return "\n".join(lines)


def _failure_snippet(output: str, limit: int = 1200) -> str:
    """The most useful chunk of a failing run's output for the fix loop."""
    idx = output.find("=== FAILURES ===")
    if idx != -1:
        return output[idx:idx + limit].strip()
    idx = output.find("=== ERRORS ===")
    if idx != -1:
        return output[idx:idx + limit].strip()
    lines = output.strip().splitlines()
    return "\n".join(lines[-25:])[-limit:]


def _run_pytest(root: Path, paths: list[str] | None,
                timeout: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    targets = paths or (["tests"] if (root / "tests").is_dir() else ["."])
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "--tb=short",
           "-rf", *targets]
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "passed": 0, "failed": [], "errors": 1,
                "seconds": round(time.perf_counter() - t0, 2),
                "selected": targets, "runner": "pytest",
                "note": f"timed out after {timeout}s"}
    out = (proc.stdout or "") + (proc.stderr or "")
    counts = _parse_pytest_counts(out)
    passed = counts.get("passed", 0)
    n_failed = counts.get("failed", 0)
    n_errors = counts.get("error", 0)
    snippet = _failure_snippet(out)
    failed = []
    for m in _FAILED_LINE.finditer(out):
        test_id = m.group(1)
        failed.append({
            "test_id": test_id,
            "file": test_id.split("::")[0],
            "error_snippet": snippet[:800],
        })
    # -x stops at the first failure, which may not get a FAILED line if the
    # run died in collection — still report it.
    if (n_failed or n_errors) and not failed:
        failed.append({"test_id": "(see output)", "file": "",
                       "error_snippet": snippet[:800]})
    seconds = round(time.perf_counter() - t0, 2)
    # exit 5 = "no tests collected": not a failure for repos/fixtures
    # without tests — report it honestly instead of failing the loop.
    ok = proc.returncode in (0, 5)
    note = None
    if proc.returncode == 5:
        note = "no tests collected"
    return {"ok": ok, "passed": passed, "failed": failed,
            "errors": n_errors, "seconds": seconds, "selected": targets,
            "runner": "pytest", "note": note}


def _path_to_module(path: str) -> str:
    """``tests/test_mod.py`` -> ``tests.test_mod``.

    pytest accepts file paths on its command line; ``python -m unittest``
    needs dotted module names.  The changed-only selector returns paths,
    so the unittest fallback converts them.
    """
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if p.endswith(".py"):
        p = p[:-3]
    return p.replace("/", ".")


def _unittest_cmd(root: Path,
                  paths: list[str] | None) -> list[str] | None:
    """Argv for the unittest fallback, or None when there is nothing to
    run (mirrors the pytest exit-5 honest skip: no tests is not a
    failure)."""
    if paths:
        return [sys.executable, "-u", "-m", "unittest",
                *[_path_to_module(p) for p in paths]]
    if (root / "tests").is_dir():
        return [sys.executable, "-u", "-m", "unittest", "discover",
                "-s", "tests", "-t", "."]
    return None


def _run_unittest(root: Path, paths: list[str] | None,
                  timeout: float) -> dict[str, Any]:
    """Fallback when pytest is not installed: the repo's own discover cmd."""
    t0 = time.perf_counter()
    cmd = _unittest_cmd(root, paths)
    if cmd is None:
        return {"ok": True, "passed": 0, "failed": [], "errors": 0,
                "seconds": round(time.perf_counter() - t0, 2),
                "selected": [], "runner": "unittest",
                "note": "no tests directory — nothing ran"}
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "passed": 0, "failed": [], "errors": 1,
                "seconds": round(time.perf_counter() - t0, 2),
                "selected": paths or ["tests"], "runner": "unittest",
                "note": f"timed out after {timeout}s"}
    out = (proc.stdout or "") + (proc.stderr or "")
    failed = []
    for m in _UNITTEST_FAIL.finditer(out):
        failed.append({"test_id": m.group(2), "file": "",
                       "error_snippet": _failure_snippet(out)[:800]})
    n_failed = len([f for f in failed if True])
    m = _UNITTEST_COUNTS.search(out)
    n_errors = 0
    if m:
        em = re.search(r"errors=(\d+)", m.group(1))
        if em:
            n_errors = int(em.group(1))
    ok = proc.returncode == 0 and "OK" in out
    # unittest prints dots, not a pass count — report what we know.
    passed = 0
    pm = re.search(r"Ran (\d+) tests?", out)
    ran = int(pm.group(1)) if pm else 0
    if ran == 0 and proc.returncode == 0:
        # nothing collected (e.g. pytest-style test functions with no
        # pytest installed): honest skip, not a failure — mirrors the
        # pytest exit-5 handling above.
        return {"ok": True, "passed": 0, "failed": [], "errors": 0,
                "seconds": round(time.perf_counter() - t0, 2),
                "selected": paths or ["tests"], "runner": "unittest",
                "ran": 0, "note": "no tests collected — nothing ran"}
    if ok:
        passed = ran
    return {"ok": ok, "passed": passed, "failed": failed,
            "errors": n_errors, "seconds": round(time.perf_counter() - t0, 2),
            "selected": paths or ["tests"], "runner": "unittest",
            "ran": ran, "_n_failed_hint": n_failed}


def run_tests(paths: list[str] | None = None,
              changed_only: bool = False,
              repo: str | None = None,
              timeout: float = 300.0) -> dict[str, Any]:
    """Run the test suite and return a structured result.

    ``changed_only`` selects tests via ``select_changed_tests``; when
    nothing matches, the run is skipped honestly (no silent full-suite
    run, no fake pass).
    """
    root = _repo_root(repo)
    selected: list[str] | None = list(paths) if paths else None
    if changed_only:
        selected = select_changed_tests(str(root))
        if not selected:
            return {"ok": True, "passed": 0, "failed": [], "errors": 0,
                    "seconds": 0.0, "selected": [],
                    "runner": "pytest" if _has_pytest() else "unittest",
                    "note": "no tests matched the changed files — nothing ran"}
    if _has_pytest():
        return _run_pytest(root, selected, timeout)
    _log.info("pytest not found — falling back to unittest discover")
    return _run_unittest(root, selected, timeout)


def register(registry: Any) -> None:
    """Attach the test runner to a registry."""
    from ..core.policy import Capability

    @registry.register(
        "run_tests",
        description=("Run the test suite (pytest if installed, else unittest "
                     "discover). changed_only selects tests touching changed "
                     "files. Returns {ok, passed, failed:[{test_id, file, "
                     "error_snippet}], errors, seconds}."),
        capability=Capability.EXEC_SHELL,
    )
    def _run_tests(paths: list[str] | None = None,
                   changed_only: bool = False,
                   repo: str | None = None) -> dict[str, Any]:
        return run_tests(paths=paths, changed_only=changed_only, repo=repo)
