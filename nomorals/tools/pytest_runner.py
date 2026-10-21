"""Pytest-aware test runner (audit Phase B).

``run_tests`` runs the suite with pytest when it is available (``-x -q``,
per-test failure ids parsed from the short summary) and falls back to the
repo's own ``unittest discover`` command otherwise — same result schema
either way.

``changed_only=True`` selects the tests that touch files changed in the
working tree (``git diff --name-only`` + untracked files).  Selection is
the UNION of two pickers (pass #62): the filename-stem heuristic, and
:mod:`nomorals.tools.import_graph` — tests that actually import a changed
module, directly or transitively.  The graph only ever ADDS tests; a
graph build failure degrades to stem selection with the degradation
noted in the result.  Every selection-driven result records
``selection_reasons`` (test -> why: self / stem / graph-direct /
graph-transitive) so a verify verdict can be audited.  Callers may pass
``changed_files`` to scope selection explicitly instead of deriving the
changed set from git.  The coding loop calls this instead of a raw
accept command.

``full_suite_gate=True`` (default off) opts a selection run into a
final gate: after the selection goes green, the FULL suite runs once
and is reported as a distinct verdict (``final_gate``) — it never
rewrites the selection run's own ``ok``.
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

__all__ = ["run_tests", "select_changed_tests",
           "select_changed_tests_detailed", "format_test_result",
           "parse_chunk", "slice_chunk",
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


def _annotate_ran(result: dict[str, Any]) -> dict[str, Any]:
    """Add the explicit ran-count signal to a runner result (pass #58).

    ``ran`` = how many tests actually executed; ``nothing_ran`` is True
    when zero did.  ``ok`` deliberately keeps its old meaning (a
    nothing-ran skip is not a runner failure) — but consumers that
    certify work (the coding loop) must never read a zero-ran result as
    verification, so the signal is explicit instead of leaving them to
    infer it from ``passed == 0`` plus a prose note.  A pre-existing
    ``ran`` (the unittest fallback reports one) is respected.
    """
    ran = result.get("ran")
    if ran is None:
        ran = (int(result.get("passed") or 0)
               + len(result.get("failed") or [])
               + int(result.get("errors") or 0))
    result["ran"] = int(ran)
    result["nothing_ran"] = int(ran) == 0
    return result


def _git_changed_files(root: Path) -> list[str]:
    """Changed + untracked files, repo-relative. [] when not a git repo."""
    git = shutil.which("git")
    if not git:
        return []
    try:
        diff = subprocess.run(
            [git, "-C", str(root), "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        status = subprocess.run(
            [git, "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
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


def _stem_selection(root: Path, changed: list[str]) -> dict[str, set[str]]:
    """Filename-stem selection as {test relpath: {"self"|"stem"}}.

    Heuristic: a changed module ``pkg/mod.py`` maps to test files whose
    filename contains the module stem (``tests/test_mod*.py``,
    ``tests/**/test_*mod*.py``); a changed test file maps to itself.
    """
    labels: dict[str, set[str]] = {}
    test_dirs = [d for d in (root / "tests", root / "test") if d.is_dir()]
    if not test_dirs:
        return labels
    for rel in changed:
        p = Path(rel)
        if "test" in p.name.lower() and p.suffix == ".py":
            candidate = root / rel
            if candidate.is_file():
                labels.setdefault(rel, set()).add("self")
            continue
        stem = p.stem
        if not stem or stem.startswith("_"):
            continue
        for tdir in test_dirs:
            for tf in tdir.rglob("test_*.py"):
                if stem in tf.stem and tf.name != "__init__.py":
                    labels.setdefault(
                        str(tf.relative_to(root)), set()).add("stem")
    return labels


#: Selection-reason labels, in the order they are joined for display.
_REASON_ORDER = ("self", "graph-direct", "graph-transitive", "stem")


def select_changed_tests_detailed(
        repo: str | None = None,
        changed_files: list[str] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Selection with provenance: ``(selected, reasons, meta)``.

    ``selected`` is the UNION of stem selection and import-graph
    selection (pass #62) — the graph adds tests whose imports cover a
    changed module; it never removes a stem pick.  ``reasons`` maps each
    selected test to a ``+``-joined label set drawn from ``self`` /
    ``stem`` / ``graph-direct`` / ``graph-transitive``.  ``meta``
    carries the changed-file list and the graph status: ``"ok"``,
    ``"degraded"`` (graph build failed; stem selection used alone, with
    ``graph_error`` saying why), or ``"not-run"`` (no changes).

    ``changed_files`` scopes the selection to an explicit changed set
    (repo-relative paths); None derives it from git, as before.
    """
    root = _repo_root(repo)
    changed = ([str(c) for c in changed_files]
               if changed_files is not None else _git_changed_files(root))
    meta: dict[str, Any] = {"changed_files": list(changed),
                            "graph_status": "not-run", "graph_error": "",
                            "counts": {}}
    labels: dict[str, set[str]] = {}
    if changed:
        labels = _stem_selection(root, changed)
        try:
            from . import import_graph as _import_graph

            graph_selection = _import_graph.graph_test_selection(
                root, changed)
            meta["graph_status"] = "ok"
            for rel, reason in graph_selection.items():
                labels.setdefault(rel, set()).add(reason)
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            meta["graph_status"] = "degraded"
            meta["graph_error"] = f"{type(exc).__name__}: {exc}"
            _log.warning("import-graph selection degraded to stem-only: %s",
                         exc)
    selected = list(labels)
    reasons = {
        rel: "+".join(sorted(ls, key=_REASON_ORDER.index))
        for rel, ls in labels.items()
    }
    meta["counts"] = {
        label: sum(1 for ls in labels.values() if label in ls)
        for label in _REASON_ORDER
    }
    return selected, reasons, meta


def select_changed_tests(repo: str | None = None,
                         changed_files: list[str] | None = None) -> list[str]:
    """Test files (repo-relative) covering the changed files.

    Union of the filename-stem heuristic and import-graph selection —
    see :func:`select_changed_tests_detailed` for the provenance-
    carrying variant.
    """
    selected, _reasons, _meta = select_changed_tests_detailed(
        repo, changed_files)
    return selected


def _has_pytest() -> bool:
    # The runner invokes `sys.executable -m pytest`, so availability is about
    # the module being importable — not about a `pytest` script on PATH
    # (pip does not always install one, e.g. user-site installs).
    return importlib.util.find_spec("pytest") is not None


def format_test_result(tres: dict[str, Any]) -> str:
    """One screen of test-runner output for the fix loop and the journal."""
    from . import _style as _style

    passed = tres.get("passed", 0)
    failed = tres.get("failed", [])
    total = passed + len(failed)
    frac = (passed / total) if total else 1.0
    lines = [_style.banner(
        f"{tres.get('runner', '?')} — {passed} passed, {len(failed)} failed",
        sub=f"{tres.get('seconds', 0)}s")]
    lines.append(_style.progress_bar(frac, label=f"{passed}/{total}"))
    for fail in failed[:3]:
        lines.append(_style.status_line(_style.FAIL,
                                        str(fail.get("test_id"))))
        if fail.get("error_snippet"):
            lines.append(_style.colorize(str(fail["error_snippet"])[:800],
                                         "dim"))
    if len(failed) > 3:
        lines.append(_style.colorize(
            f"… and {len(failed) - 3} more failures", "dim"))
    gate = tres.get("final_gate")
    if isinstance(gate, dict) and gate.get("ran"):
        verdict = str(gate.get("verdict", "?")).upper()
        mark = _style.OK if verdict == "PASS" else _style.FAIL
        gate_line = (f"final gate (full suite): {verdict} — "
                     f"{gate.get('passed', 0)} passed")
        gate_failed = gate.get("failed") or []
        if gate_failed:
            gate_line += f", {len(gate_failed)} failed: " + ", ".join(
                str(t) for t in gate_failed[:3])
            if len(gate_failed) > 3:
                gate_line += f", … +{len(gate_failed) - 3}"
        lines.append(_style.status_line(mark, gate_line))
    note = tres.get("note")
    if note:
        lines.append(str(note))
    if not failed:
        lines.append(_style.status_line(_style.OK, "all green"))
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
                timeout: float, *,
                stop_on_fail: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    targets = paths or (["tests"] if (root / "tests").is_dir() else ["."])
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-rf"]
    if stop_on_fail:
        cmd.append("-x")
    cmd.extend(targets)
    try:
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        # NOT a hang: the suite was still running when the budget ran
        # out. A bare "timed out" reads as "the tests hang" — the exact
        # misdiagnosis a full-suite run with a 120s cap produced. Say
        # what happened and how to continue (bigger timeout / chunk).
        return {"ok": False, "passed": 0, "failed": [], "errors": 1,
                "seconds": round(time.perf_counter() - t0, 2),
                "selected": targets, "runner": "pytest",
                "note": (f"stopped after {timeout}s — suite was still running "
                         "(not necessarily hung); rerun with a larger "
                         "timeout or chunk='i/n' to continue")}
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


def parse_chunk(chunk: str | None) -> tuple[int, int] | None:
    """``"i/n"`` → ``(i, n)``, or None when no chunk was asked for.

    Raises ValueError on a malformed chunk spec — a silent mis-slice
    would run the wrong tests and report them as the whole suite.
    """
    if chunk is None:
        return None
    text = str(chunk).strip()
    if not text:
        return None
    try:
        i_str, n_str = text.split("/", 1)
        i, n = int(i_str), int(n_str)
    except ValueError:
        raise ValueError(
            f"chunk must look like 'i/n' (e.g. '1/4'), got {chunk!r}") from None
    if n < 1 or i < 1 or i > n:
        raise ValueError(
            f"chunk must satisfy 1 <= i <= n, got {chunk!r}")
    return (i, n)


def slice_chunk(files: list[str], chunk: str | None) -> list[str]:
    """One slice of ``files`` for chunk ``"i/n"`` (even split, ceil).

    The full suite is ~815 files / ~85 min in this repo, so callers with
    a short timeout run one slice per call (1/4 … 4/4) instead of timing
    out mid-suite and concluding the tests hang.
    """
    parsed = parse_chunk(chunk)
    if parsed is None:
        return list(files)
    i, n = parsed
    if not files:
        return []
    per = -(-len(files) // n)  # ceil division
    return list(files[(i - 1) * per:i * per])


def _expand_test_files(root: Path, targets: list[str] | None) -> list[str]:
    """Directory targets → sorted test-file list; file/node targets as-is."""
    out: list[str] = []
    for target in targets or []:
        candidate = root / target
        if candidate.is_dir():
            for tf in sorted(candidate.rglob("test_*.py")):
                out.append(str(tf.relative_to(root)))
        else:
            out.append(target)
    return out


def _is_dir_target(root: Path, targets: list[str] | None) -> bool:
    return bool(targets) and any((root / t).is_dir() for t in targets)
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
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "passed": 0, "failed": [], "errors": 1,
                "seconds": round(time.perf_counter() - t0, 2),
                "selected": paths or ["tests"], "runner": "unittest",
                "note": (f"stopped after {timeout}s — suite was still running "
                         "(not necessarily hung); rerun with a larger "
                         "timeout or chunk='i/n' to continue")}
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


def _selection_meta_public(meta: dict[str, Any]) -> dict[str, Any]:
    """The auditable slice of selection meta recorded on results."""
    out: dict[str, Any] = {
        "graph_status": meta.get("graph_status", "not-run"),
        "counts": meta.get("counts", {}),
        "changed_files": meta.get("changed_files", []),
    }
    if meta.get("graph_error"):
        out["graph_error"] = meta["graph_error"]
    return out


def _skipped_gate(reason: str) -> dict[str, Any]:
    return {"ran": False, "verdict": "skipped", "reason": reason}


def _final_gate(root: Path, primary: dict[str, Any], *,
                chunked: bool, primary_full_suite: bool,
                timeout: float, gate_timeout: float | None,
                ) -> dict[str, Any]:
    """The optional full-suite final gate (pass #62): a DISTINCT verdict.

    Runs the whole suite once, only after a selection run went green.
    The gate's verdict (``pass`` / ``fail`` / ``unavailable`` /
    ``skipped``) is reported alongside the selection verdict — it never
    rewrites it: a full-suite failure lists the failing tests so
    pre-existing breakage is attributable instead of poisoning the
    verdict on the change under review.
    """
    if chunked:
        return _skipped_gate(
            "a chunked run does not run the final gate "
            "(a slice of the suite is not the suite)")
    if primary.get("nothing_ran"):
        return _skipped_gate("the selection run executed no tests")
    if not primary.get("ok") or primary.get("failed") \
            or primary.get("errors"):
        return _skipped_gate("the selection run was not green")
    if primary_full_suite:
        return _skipped_gate(
            "the primary run already covered the full suite")
    budget = gate_timeout if gate_timeout is not None else timeout
    if _has_pytest():
        # No -x here: attribution needs EVERY failing test, not the
        # first one.
        gate = _run_pytest(root, None, budget, stop_on_fail=False)
    else:
        gate = _run_unittest(root, None, budget)
    failed_ids = [str(f.get("test_id")) for f in gate.get("failed", [])]
    note = str(gate.get("note") or "")
    if note.startswith("stopped after"):
        verdict = "unavailable"
    elif gate.get("ok") and not failed_ids and not gate.get("errors"):
        verdict = "pass"
    else:
        verdict = "fail"
    return {"ran": True, "verdict": verdict,
            "runner": gate.get("runner"),
            "passed": gate.get("passed", 0),
            "failed": failed_ids,
            "errors": gate.get("errors", 0),
            "seconds": gate.get("seconds", 0.0),
            "note": note or None}


def run_tests(paths: list[str] | None = None,
              changed_only: bool = False,
              repo: str | None = None,
              timeout: float = 300.0,
              chunk: str | None = None,
              changed_files: list[str] | None = None,
              full_suite_gate: bool = False,
              gate_timeout: float | None = None) -> dict[str, Any]:
    """Run the test suite and return a structured result.

    ``changed_only`` selects tests via ``select_changed_tests_detailed``
    (stem ∪ import graph, pass #62); when nothing matches, the run is
    skipped honestly (no silent full-suite run, no fake pass).
    ``changed_files`` scopes that selection to an explicit changed set
    instead of deriving it from git.

    No ``paths`` and no ``changed_only`` → targeted by default: the
    tests touching changed files. A bare call used to run the FULL suite
    (~815 files / ~85 min in this repo) behind whatever timeout the
    caller had — a 120s cap guaranteed a timeout that looked like a
    hang. To run the whole suite, pass ``paths=["tests"]`` with
    ``chunk="i/n"`` (e.g. 1/4 … 4/4), one slice per call.

    Every result carries ``ran`` (tests actually executed) and
    ``nothing_ran`` (``ran == 0``).  ``ok`` stays True for an honest
    nothing-ran skip — but a caller certifying work must check
    ``nothing_ran``: zero executed tests is not a pass (pass #58).

    Selection-driven results also carry ``selection_reasons`` (test ->
    why it was picked) and ``selection`` (graph status + per-reason
    counts).  ``full_suite_gate=True`` adds ``final_gate`` +
    ``final_gate_verdict`` after a green selection run — a separate
    verdict that never rewrites this result's ``ok``.
    """
    root = _repo_root(repo)
    if isinstance(paths, str):
        # cmdline/commands/repo.py calls run_tests(root, ...) with a bare
        # string; list(str) would split it into characters.
        paths = [paths]
    selected: list[str] | None = list(paths) if paths else None
    targeted = False
    sel_reasons: dict[str, str] = {}
    sel_meta: dict[str, Any] | None = None
    gate_skip: dict[str, Any] = {}
    if full_suite_gate:
        gate_skip = {
            "final_gate": _skipped_gate(
                "the selection run executed no tests"),
            "final_gate_verdict": "skipped"}
    if changed_only:
        selected, sel_reasons, sel_meta = select_changed_tests_detailed(
            str(root), changed_files)
        if not selected:
            return _annotate_ran(
                {"ok": True, "passed": 0, "failed": [], "errors": 0,
                 "seconds": 0.0, "selected": [],
                 "runner": "pytest" if _has_pytest() else "unittest",
                 "selection_reasons": sel_reasons,
                 "selection": _selection_meta_public(sel_meta),
                 **gate_skip,
                 "note": "no tests matched the changed files — nothing ran"})
    elif selected is None:
        selected, sel_reasons, sel_meta = select_changed_tests_detailed(
            str(root), changed_files)
        if selected:
            targeted = True
        else:
            return _annotate_ran(
                {"ok": True, "passed": 0, "failed": [], "errors": 0,
                 "seconds": 0.0, "selected": [],
                 "runner": "pytest" if _has_pytest() else "unittest",
                 "selection_reasons": sel_reasons,
                 "selection": _selection_meta_public(sel_meta),
                 **gate_skip,
                 "note": ("no tests matched the changed files — nothing ran. "
                          "Pass paths=['tests'] for the full suite "
                          "(chunk='i/n' slices, e.g. 1/4 … 4/4) or "
                          "paths=[...] for specific files.")})
    chunk_parsed = parse_chunk(chunk)
    full_suite = _is_dir_target(root, selected)
    if chunk_parsed is not None:
        expanded = _expand_test_files(root, selected)
        selected = slice_chunk(expanded, chunk)
        if not selected:
            # An empty slice must NOT fall through to _run_pytest's
            # ["tests"] default — that would run the whole suite.
            chunk_gate: dict[str, Any] = {}
            if full_suite_gate:
                chunk_gate = {
                    "final_gate": _skipped_gate(
                        "a chunked run does not run the final gate "
                        "(a slice of the suite is not the suite)"),
                    "final_gate_verdict": "skipped"}
            return _annotate_ran(
                {"ok": True, "passed": 0, "failed": [], "errors": 0,
                 "seconds": 0.0, "selected": [],
                 "runner": "pytest" if _has_pytest() else "unittest",
                 "chunk": chunk, "chunk_of": f"{chunk_parsed[0]}/{chunk_parsed[1]}",
                 **chunk_gate,
                 "note": (f"chunk {chunk} is empty — only "
                          f"{len(expanded)} test file(s) exist")})
    if _has_pytest():
        result = _run_pytest(root, selected, timeout)
    else:
        _log.info("pytest not found — falling back to unittest discover")
        result = _run_unittest(root, selected, timeout)
    if chunk_parsed is not None:
        result["chunk"] = chunk
        result["chunk_of"] = f"{chunk_parsed[0]}/{chunk_parsed[1]}"
        base = result.get("note") or ""
        result["note"] = (f"chunk {chunk} of the suite"
                          + (f" — {base}" if base else ""))
    elif targeted:
        base = result.get("note") or ""
        result["note"] = ("targeted run: tests touching changed files"
                          + (f" — {base}" if base else ""))
    elif full_suite and not result.get("note"):
        n_files = len(_expand_test_files(root, selected))
        result["note"] = (
            f"full-suite run: {n_files} test files — pass chunk='i/n' "
            "(e.g. 1/4 … 4/4) to run one slice per call")
    if sel_meta is not None:
        result["selection_reasons"] = sel_reasons
        result["selection"] = _selection_meta_public(sel_meta)
        if sel_meta.get("graph_status") == "degraded":
            base = result.get("note") or ""
            result["note"] = (
                f"import-graph selection degraded "
                f"({sel_meta.get('graph_error')}) — stem selection used"
                + (f" — {base}" if base else ""))
    _annotate_ran(result)
    if full_suite_gate:
        gate = _final_gate(root, result, chunked=chunk_parsed is not None,
                           primary_full_suite=full_suite,
                           timeout=timeout, gate_timeout=gate_timeout)
        result["final_gate"] = gate
        result["final_gate_verdict"] = gate["verdict"]
        if gate.get("ran"):
            base = result.get("note") or ""
            gate_line = (f"final gate: {str(gate['verdict']).upper()} — "
                         f"full suite {gate.get('passed', 0)} passed, "
                         f"{len(gate.get('failed') or [])} failed")
            if gate["verdict"] == "fail":
                gate_line += " (selection verdict above is unchanged)"
            result["note"] = gate_line + (f" — {base}" if base else "")
    return result


def register(registry: Any) -> None:
    """Attach the test runner to a registry."""
    from ..core.policy import Capability

    @registry.register(
        "run_tests",
        description=("Run the test suite (pytest if installed, else unittest "
                     "discover). changed_only selects tests touching changed "
                     "files. No paths → targeted run (changed files only); "
                     "pass paths=['tests'] + chunk='i/n' for one slice of the "
                     "full suite (~85 min). Returns {ok, passed, failed:"
                     "[{test_id, file, error_snippet}], errors, seconds, "
                     "ran, nothing_ran}."),
        capability=Capability.EXEC_SHELL,
        parameters={
            "paths": "list[str] (optional) — test files/dirs to run",
            "changed_only": "bool (optional) — only tests touching changed files",
            "repo": "str (optional) — repo root",
            "timeout": "float (optional, 300) — seconds per run; full-suite "
                       "runs should use chunk= instead of a huge timeout",
            "chunk": "str (optional, 'i/n') — run one slice of the suite",
            "changed_files": "list[str] (optional) — explicit changed set "
                             "for selection instead of deriving it from git",
            "full_suite_gate": "bool (optional, False) — after a green "
                               "selection run, run the full suite once and "
                               "report it as final_gate (a distinct "
                               "verdict; never rewrites ok)",
        },
    )
    def _run_tests(paths: list[str] | None = None,
                   changed_only: bool = False,
                   repo: str | None = None,
                   timeout: float = 300.0,
                   chunk: str | None = None,
                   changed_files: list[str] | None = None,
                   full_suite_gate: bool = False) -> dict[str, Any]:
        return run_tests(paths=paths, changed_only=changed_only, repo=repo,
                         timeout=timeout, chunk=chunk,
                         changed_files=changed_files,
                         full_suite_gate=full_suite_gate)
