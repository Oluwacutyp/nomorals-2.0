"""Pluggable verifiers for the OS control plane (Wave H2, L6).

A verifier checks a finished piece of work and returns a :class:`Verdict`.
Verifiers are duck-typed — anything with a ``name`` attribute and a
``verify(target: dict) -> Verdict`` method qualifies; the
:class:`Verifier` protocol below is the documented contract and
:class:`VerifierRegistry` is the lookup.

Two production verifiers ship here:

* :class:`CodeTestsVerifier` — runs ``python -m unittest <test_ids>`` in a
  subprocess (with a timeout) and parses the ``Ran`` / ``OK`` / ``FAILED``
  summary. Offline by construction.
* :class:`DocsRenderVerifier` — every ``.md`` doc under the repo root must
  parse as text, and the README's claimed test/line counts must be within
  15% of the values measured from the tree. Reuses
  ``tests/test_docs_consistency.py`` when that module is importable;
  otherwise reimplements its (~20-line) measurement logic locally.

Plus pre-flight gates (bundled in :func:`gate_registry` as the
``pre_update_gates`` :class:`CompositeVerifier`):

* :class:`LintVerifier` — every ``.py`` file byte-compiles.
* :class:`GitCleanVerifier` — the working tree has no uncommitted changes.
* :class:`DiskSpaceVerifier` — enough free disk for the operation.
* :class:`CompositeVerifier` — all/any suites of other verifiers.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from ..core.logging_setup import get_logger

__all__ = [
    "Verdict",
    "Verifier",
    "VerifierRegistry",
    "default_registry",
    "gate_registry",
    "CodeTestsVerifier",
    "DocsRenderVerifier",
    "CompositeVerifier",
    "LintVerifier",
    "GitCleanVerifier",
    "DiskSpaceVerifier",
]

_log = get_logger(__name__)


# ── contract ─────────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    """The outcome of one verification."""

    passed: bool
    details: str = ""
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "details": self.details,
                "artifacts": list(self.artifacts)}

    def render(self) -> str:
        """One-block human rendering of the verdict."""
        mark = "✓" if self.passed else "✗"
        lines = [f"{mark} {'PASS' if self.passed else 'FAIL'}"]
        for line in str(self.details or "").splitlines():
            lines.append(f"  {line}")
        for artifact in self.artifacts:
            lines.append(f"  artifact: {artifact}")
        return "\n".join(lines)


@runtime_checkable
class Verifier(Protocol):
    """Anything with a ``name`` and ``verify(target) -> Verdict``."""

    name: str

    def verify(self, target: dict[str, Any]) -> Verdict:
        ...


class VerifierRegistry:
    """Name -> verifier lookup. Register custom verifiers here; the repair
    loop and the mission state machine resolve them by name."""

    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, verifier: Verifier) -> Verifier:
        name = str(getattr(verifier, "name", "") or type(verifier).__name__)
        self._verifiers[name] = verifier
        _log.debug("verifier registered: %s", name)
        return verifier

    def get(self, name: str) -> Verifier | None:
        return self._verifiers.get(name)

    def list(self) -> list[str]:
        return sorted(self._verifiers)

    def verify(self, name: str, target: dict[str, Any]) -> Verdict:
        verifier = self.get(name)
        if verifier is None:
            raise KeyError(f"unknown verifier: {name!r}")
        return verifier.verify(target)

    def __contains__(self, name: object) -> bool:
        return name in self._verifiers


def default_registry() -> VerifierRegistry:
    """Registry with the two core verifiers pre-registered.

    (Unchanged by the sweep: the pre-flight gates live in
    :func:`gate_registry` so this contract stays stable.)
    """
    registry = VerifierRegistry()
    registry.register(CodeTestsVerifier())
    registry.register(DocsRenderVerifier())
    return registry


def gate_registry() -> VerifierRegistry:
    """Registry with the pre-flight gate verifiers: lint, git_clean,
    disk_space, plus the ``pre_update_gates`` all-of composite."""
    registry = VerifierRegistry()
    lint = LintVerifier()
    git_clean = GitCleanVerifier()
    disk_space = DiskSpaceVerifier()
    for verifier in (lint, git_clean, disk_space):
        registry.register(verifier)
    registry.register(CompositeVerifier(
        "pre_update_gates", [lint, git_clean, disk_space], mode="all"))
    return registry


# ── code tests ───────────────────────────────────────────────────────────────

class CodeTestsVerifier:
    """Runs ``python -m unittest <test_ids>`` as a subprocess.

    The target dict carries ``test_ids`` (required, e.g.
    ``["tests.test_missions"]`` or ``["tests.test_missions.Suite.test_x"]``),
    plus optional ``cwd`` and ``timeout`` overrides. Parses the unittest
    ``Ran N tests`` / ``OK`` / ``FAILED`` summary — a zero exit code alone
    is not trusted.
    """

    name = "code_tests"

    _RAN_RE = re.compile(r"^Ran (\d+) test", re.MULTILINE)
    _OK_RE = re.compile(r"^OK(?:\s|$)", re.MULTILINE)
    _FAILED_RE = re.compile(r"^FAILED", re.MULTILINE)

    def __init__(self, *, timeout: float = 300.0,
                 python: str | None = None) -> None:
        self.timeout = timeout
        self.python = python or sys.executable

    def verify(self, target: dict[str, Any]) -> Verdict:
        test_ids = [str(t) for t in (target.get("test_ids") or [])]
        if not test_ids:
            return Verdict(passed=False, details="no test_ids in target")
        try:
            timeout = float(target.get("timeout") or self.timeout)
        except (TypeError, ValueError):
            timeout = self.timeout
        cwd = str(target.get("cwd") or os.getcwd())
        cmd = [self.python, "-m", "unittest", *test_ids]
        try:
            proc = subprocess.run(  # noqa: S603 - test ids come from the operator
                cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            return Verdict(
                passed=False,
                details=f"unittest timed out after {timeout:g}s: "
                        f"{' '.join(test_ids)}",
            )
        except OSError as exc:
            return Verdict(passed=False,
                           details=f"could not launch unittest: {exc}")
        output = (proc.stderr or "") + "\n" + (proc.stdout or "")
        ran_match = self._RAN_RE.search(output)
        ran = int(ran_match.group(1)) if ran_match else 0
        ok = bool(self._OK_RE.search(output))
        failed = bool(self._FAILED_RE.search(output))
        passed = ok and not failed and proc.returncode == 0
        detail = (f"{' '.join(test_ids)}: Ran {ran} test(s), "
                  f"{'OK' if passed else 'FAILED'} (exit {proc.returncode})")
        if not passed:
            tail = "\n".join(output.strip().splitlines()[-10:])
            if tail:
                detail += "\n" + tail
        return Verdict(passed=passed, details=detail)


# ── docs render ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@contextmanager
def _sys_path(path: Path) -> Iterator[None]:
    """Temporarily prepend ``path`` to ``sys.path``."""
    inserted = False
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted = True
    try:
        yield
    finally:
        if inserted and text in sys.path:
            sys.path.remove(text)


def _import_docs_check(root: Path) -> Any | None:
    """Import ``tests/test_docs_consistency.py`` and return the module.

    Prefers the normal package import when ``tests/`` is a package;
    otherwise loads the module file directly (no ``sys.path`` surgery
    beyond the repo root). Returns ``None`` when the module is missing
    or broken — the caller then falls back to the local reimplementation.
    """
    mod_file = root / "tests" / "test_docs_consistency.py"
    if not mod_file.is_file():
        return None
    try:
        if (root / "tests" / "__init__.py").is_file():
            with _sys_path(root):
                return importlib.import_module("tests.test_docs_consistency")
        spec = importlib.util.spec_from_file_location(
            "_nm_docs_consistency", str(mod_file))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - fall back to the reimplementation
        _log.debug("could not import test_docs_consistency", exc_info=True)
        return None


_TEST_DEF = re.compile(r"^\s*def\s+test_", re.MULTILINE)


def _local_measure_modules(root: Path) -> list[Path]:
    return [p for p in (root / "nomorals").rglob("*.py") if p.is_file()]


def _local_measure_lines(modules: list[Path]) -> int:
    total = 0
    for p in modules:
        total += len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
    return total


def _local_measure_tests(root: Path) -> int:
    total = 0
    for p in sorted((root / "tests").glob("test_*.py")):
        total += len(_TEST_DEF.findall(
            p.read_text(encoding="utf-8", errors="ignore")))
    return total


def _local_parse_int(text: str, pattern: str, label: str) -> int:
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"README.md has no parseable {label} claim")
    return int(m.group(1).replace(",", ""))


class DocsRenderVerifier:
    """The README cannot go stale silently.

    Checks: (1) every ``.md`` file under the repo root parses as text
    (UTF-8); (2) the README's claimed test / line / module counts are each
    within ``tolerance`` (default 15%) of the values measured from the
    tree. The measurement reuses ``tests/test_docs_consistency.py`` when
    that module is importable, else the local reimplementation above.
    """

    name = "docs_render"

    def __init__(self, *, root: str | Path | None = None,
                 tolerance: float = 0.15) -> None:
        self.root = Path(root) if root is not None else _repo_root()
        self.tolerance = tolerance

    def verify(self, target: dict[str, Any]) -> Verdict:
        root = Path(target.get("root") or self.root)
        tolerance = float(target.get("tolerance") or self.tolerance)
        problems: list[str] = []
        notes: list[str] = []

        # 1. every .md doc parses as text
        md_files = sorted(
            p for p in root.rglob("*.md")
            if p.is_file() and ".git" not in p.parts
            and "node_modules" not in p.parts)
        unreadable = 0
        for md in md_files:
            try:
                md.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - record, don't raise
                unreadable += 1
                problems.append(
                    f"{md.relative_to(root)}: not parseable as text ({exc})")
        notes.append(f"{len(md_files)} markdown file(s) checked, "
                     f"{unreadable} unreadable")

        # 2. README claims vs measured tree
        readme = root / "README.md"
        if not readme.is_file():
            problems.append("README.md missing")
        else:
            try:
                text = readme.read_text(encoding="utf-8")
                modules, lines, tests = self._measure(root)
                self._check_within("test count", r"~([\d,]+)\s+tests",
                                   tests, text, tolerance, problems)
                self._check_within("line count", r"([\d,]+)\s+lines of Python",
                                   lines, text, tolerance, problems)
                self._check_within("module count", r"(\d+)\+\s+modules",
                                   len(modules), text, tolerance, problems)
                for stale in ("/home/hatch",):
                    if stale in text:
                        problems.append(
                            f"README.md contains stale absolute path {stale!r}")
                notes.append(f"measured {tests} tests, {lines} lines, "
                             f"{len(modules)} modules")
            except ValueError as exc:
                problems.append(str(exc))

        passed = not problems
        details = "; ".join(notes)
        if problems:
            details += " | PROBLEMS: " + "; ".join(problems)
        return Verdict(passed=passed, details=details)

    def _measure(self, root: Path) -> tuple[list[Path], int, int]:
        """Measure (modules, lines, tests), reusing the canonical test
        module when it is importable."""
        mod = _import_docs_check(root)
        if mod is not None:
            modules = mod._measure_modules()
            return modules, mod._measure_lines(modules), mod._measure_tests()
        modules = _local_measure_modules(root)
        return modules, _local_measure_lines(modules), _local_measure_tests(root)

    @staticmethod
    def _check_within(label: str, pattern: str, measured: int, text: str,
                      tolerance: float, problems: list[str]) -> None:
        try:
            claimed = _local_parse_int(text, pattern, label)
        except ValueError as exc:
            problems.append(str(exc))
            return
        ratio = measured / claimed if claimed else float("inf")
        drift = abs(1.0 - ratio)
        if drift > tolerance:
            problems.append(
                f"{label}: README claims ~{claimed:,} but measured "
                f"{measured:,} (drift {drift * 100:.1f}% > "
                f"{tolerance * 100:.0f}%)")


# ── composite ──────────────────────────────────────────────────────────────

class CompositeVerifier:
    """Run several verifiers as one suite.

    ``mode="all"`` passes only when every member passes; ``mode="any"``
    passes when at least one passes.  The verdict detail carries each
    member's outcome so failures stay attributable.
    """

    def __init__(self, name: str, verifiers: list[Any],
                 *, mode: str = "all") -> None:
        if mode not in ("all", "any"):
            raise ValueError(f"unknown composite mode {mode!r}")
        self.name = name
        self.verifiers = list(verifiers)
        self.mode = mode

    def verify(self, target: dict[str, Any]) -> Verdict:
        results: list[tuple[str, Verdict]] = []
        for verifier in self.verifiers:
            vname = str(getattr(verifier, "name", None)
                        or type(verifier).__name__)
            try:
                verdict = verifier.verify(target)
            except Exception as exc:  # noqa: BLE001 — a raising member fails
                verdict = Verdict(passed=False,
                                  details=f"verifier raised: {exc!r}")
            results.append((vname, verdict))
        passed = (all(v.passed for _, v in results) if self.mode == "all"
                  else any(v.passed for _, v in results))
        lines = [f"{'PASS' if v.passed else 'FAIL'} {name}"
                 for name, v in results]
        details = (f"composite({self.mode}) "
                   f"{sum(v.passed for _, v in results)}/{len(results)} passed"
                   + ("\n" + "\n".join(lines) if lines else ""))
        artifacts = [a for _, v in results for a in v.artifacts]
        return Verdict(passed=passed, details=details, artifacts=artifacts)


# ── pre-flight gates ─────────────────────────────────────────────────────────

class LintVerifier:
    """Every ``.py`` file under the tree must byte-compile.

    Target keys: ``root`` (default: repo root), ``paths`` (explicit file or
    dir list instead of the tree walk).
    """

    name = "lint"

    def verify(self, target: dict[str, Any]) -> Verdict:
        import py_compile

        root = Path(target.get("root") or _repo_root())
        paths = target.get("paths")
        if paths:
            candidates = [Path(p) for p in paths]
        else:
            candidates = [root / "nomorals"]
        files: list[Path] = []
        for candidate in candidates:
            candidate = candidate if candidate.is_absolute() else root / candidate
            if candidate.is_file() and candidate.suffix == ".py":
                files.append(candidate)
            elif candidate.is_dir():
                files.extend(sorted(p for p in candidate.rglob("*.py")
                                    if p.is_file() and ".git" not in p.parts))
        failures: list[str] = []
        for path in files:
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path}: {exc.msg}")
            except OSError as exc:
                failures.append(f"{path}: unreadable ({exc})")
        if failures:
            detail = (f"{len(failures)}/{len(files)} file(s) failed to compile"
                      + "\n" + "\n".join(failures[:10]))
            if len(failures) > 10:
                detail += f"\n… and {len(failures) - 10} more"
            return Verdict(passed=False, details=detail)
        return Verdict(passed=True,
                       details=f"{len(files)} python file(s) compile clean")


class GitCleanVerifier:
    """The working tree must be clean (no uncommitted changes).

    Target keys: ``repo_dir`` (default: repo root), ``allow_untracked``
    (default False).
    """

    name = "git_clean"

    def __init__(self, *, git_bin: str = "git") -> None:
        self.git_bin = git_bin

    def verify(self, target: dict[str, Any]) -> Verdict:
        repo = str(target.get("repo_dir") or _repo_root())
        allow_untracked = bool(target.get("allow_untracked", False))
        try:
            proc = subprocess.run(
                [self.git_bin, "-C", repo, "status", "--porcelain"],
                capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError:
            return Verdict(passed=False, details="git not found")
        except subprocess.TimeoutExpired:
            return Verdict(passed=False, details="git status timed out")
        if proc.returncode != 0:
            return Verdict(passed=False,
                           details=f"git status failed: {proc.stderr.strip()}")
        dirty = [line for line in proc.stdout.splitlines() if line.strip()]
        if not allow_untracked:
            dirty = [line for line in dirty
                     if not line.startswith("??")]
        if dirty:
            detail = (f"working tree dirty ({len(dirty)} change(s)):\n"
                      + "\n".join(dirty[:10]))
            if len(dirty) > 10:
                detail += f"\n… and {len(dirty) - 10} more"
            return Verdict(passed=False, details=detail)
        return Verdict(passed=True, details="working tree clean")


class DiskSpaceVerifier:
    """Enough free disk for the operation ahead.

    Target keys: ``path`` (default: repo root), ``min_free_mb`` (default
    500), ``min_free_pct`` (default 0.05 — at least 5% free).
    """

    name = "disk_space"

    def verify(self, target: dict[str, Any]) -> Verdict:
        import shutil

        path = str(target.get("path") or _repo_root())
        try:
            min_free_mb = float(target.get("min_free_mb", 500))
        except (TypeError, ValueError):
            min_free_mb = 500.0
        try:
            min_free_pct = float(target.get("min_free_pct", 0.05))
        except (TypeError, ValueError):
            min_free_pct = 0.05
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            return Verdict(passed=False, details=f"disk_usage failed: {exc}")
        free_mb = usage.free / (1024 * 1024)
        free_pct = usage.free / usage.total if usage.total else 0.0
        ok = free_mb >= min_free_mb and free_pct >= min_free_pct
        detail = (f"{free_mb:.0f} MB free ({free_pct * 100:.1f}%) at {path};"
                  f" need ≥{min_free_mb:.0f} MB and ≥{min_free_pct * 100:.0f}%")
        return Verdict(passed=ok, details=detail)


def register(registry: Any) -> None:
    """Expose the verifier registry as agent tools."""

    @registry.register(
        "verify",
        description=(
            "Run a named verifier against a target. Verifiers: code_tests "
            "(run python -m unittest on test_ids), docs_render (check README "
            "metrics match measured repo state), lint (py_compile the tree), "
            "git_clean (working tree clean), disk_space (enough free disk), "
            "pre_update_gates (all-of lint+git_clean+disk_space). "
            "Target is a JSON dict."
        ),
        capability="verify.run",
        parameters={
            "verifier": "str — verifier name (code_tests|docs_render)",
            "target_json": "str — JSON-encoded target dict",
        },
    )
    def _verify(verifier: str, target_json: str = "{}") -> dict[str, Any]:
        import json

        try:
            target = json.loads(target_json or "{}")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"bad target_json: {exc}"}
        reg = default_registry()
        for gate in gate_registry().list():
            verifier_obj = gate_registry().get(gate)
            if verifier_obj is not None and gate not in reg:
                reg.register(verifier_obj)
        try:
            verdict = reg.verify((verifier or "").strip(), target)
            return {"ok": True, "verdict": verdict.to_dict()}
        except KeyError:
            return {
                "ok": False,
                "error": f"unknown verifier {verifier!r}; available: {reg.list()}",
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @registry.register(
        "verify_list",
        description="List available verifiers in the registry.",
        capability="verify.run",
        parameters={},
    )
    def _verify_list() -> dict[str, Any]:
        reg = default_registry()
        names = set(reg.list()) | set(gate_registry().list())
        return {"ok": True, "verifiers": sorted(names)}
