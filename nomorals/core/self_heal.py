"""Self-healing loop for broken chat commands.

Pipeline: tour (find broken) → error doctor (diagnose) → auto-fix
(conservative patterns only) → re-probe (verify) → report → learn.

Fix strategies live in a registry (:data:`FIX_STRATEGIES`), not hardcoded
branches — new strategies can be added with :func:`register_fix_strategy`.
Two built-in strategies are auto-applied, both provably safe because they
cannot change behavior, only repair structure:

1. ``UnboundLocalError`` — a variable assigned in some branches but
   referenced on a path where no assignment ran. Fix: initialize it to
   ``None`` at the top of the function. This cannot change any path
   where the variable WAS assigned (the init is overwritten), and on
   the broken path it turns a crash into a ``None`` (which downstream
   code already handles — it was written expecting the variable).

2. Dead-code method rescue — a ``_control_*`` method nested inside a
   plain function instead of living on the mixin class, so dispatch
   raises ``AttributeError``. Fix: move it to class level with correct
   indentation. Only when it uses no closure variables from the
   enclosing scope (verified by AST), so the move is semantics-preserving.

Everything else (missing pip packages, logic TypeErrors, etc.) gets a
diagnosis + suggestion but is NEVER auto-fixed.

The learning loop: pass an ``ErrorIntelligence`` as ``intelligence`` and
every outcome (fix verified / fix failed) is reported back, so the error
catcher learns which fixes actually work per error fingerprint.

Each applied fix is committed to git separately so it's reversible.
``diagnose()``/tour probing never raise; neither does anything here.
"""

from __future__ import annotations

import ast
import concurrent.futures as _cf
import os
import re
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .error_doctor import diagnose

__all__ = [
    "heal_probe",
    "fix_unbound_local",
    "fix_dead_method",
    "heal_one",
    "heal_all",
    "HealResult",
    "FixStrategy",
    "register_fix_strategy",
    "FIX_STRATEGIES",
    "diff_preview",
]

# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

class HealResult:
    """Outcome of attempting to heal one broken command."""

    def __init__(self, kind: str):
        self.kind = kind
        self.broken: bool = False
        self.timed_out: bool = False     # probe hung — a finding, not "healthy"
        self.error: str = ""
        self.location: str = ""
        self.fingerprint: str = ""       # error fingerprint for the learning loop
        self.diagnosis: dict[str, Any] | None = None
        self.fixable: bool = False
        self.fix_kind: str = ""          # "unbound_local" | "dead_method" | ""
        self.fix_detail: str = ""       # human description of the fix
        self.fixed: bool = False        # fix applied AND verified by re-probe
        self.dry_run: bool = False
        self.diff_preview: str = ""      # first lines of the change (dry-run)
        self.skip_reason: str = ""      # why not auto-fixed
        self.suggestion: str = ""       # error-doctor suggestion for unfixable
        self.committed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "broken": self.broken,
            "timed_out": self.timed_out, "error": self.error,
            "location": self.location, "fingerprint": self.fingerprint,
            "fixable": self.fixable,
            "fix_kind": self.fix_kind, "fix_detail": self.fix_detail,
            "fixed": self.fixed, "dry_run": self.dry_run,
            "skip_reason": self.skip_reason, "suggestion": self.suggestion,
            "committed": self.committed,
        }


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - the healer never gets sick
        return default


# ---------------------------------------------------------------------------
# probing — like the tour, but keeps the exception OBJECT for diagnosis
# ---------------------------------------------------------------------------

_PROBE_TIMEOUT = 6.0

_VAR_RE = re.compile(r"local variable '([^']+)'")
_ATTR_RE = re.compile(r"'([^']+)' object has no attribute '([^']+)'")
_MOD_RE = re.compile(r"No module named '([^']+)'")


@dataclass
class ProbeReport:
    """Detailed probe outcome. ``heal_probe`` returns the 4-tuple
    ``(broken, exc, err, loc)`` for backward compatibility; the healer uses
    the full report so a hung command is a finding, not a clean bill."""

    broken: bool
    exc: BaseException | None
    err: str
    loc: str
    timed_out: bool = False
    elapsed: float = 0.0


def _probe_inner(handle_control, kind: str, chat_key: str = "") -> ProbeReport:
    """Probe one command, capturing the exception object. Never raises."""
    import time as _time
    start = _time.monotonic()

    def _run():
        return handle_control(f"/{kind}", chat_key, message=None)

    try:
        with _cf.ThreadPoolExecutor(max_workers=1,
                                     thread_name_prefix="heal") as _ex:
            fut = _ex.submit(_run)
            try:
                fut.result(timeout=_PROBE_TIMEOUT)
            except _cf.TimeoutError:
                fut.cancel()
                elapsed = _time.monotonic() - start
                # A hung command IS a finding — report it distinctly instead
                # of the old "slow → not our problem" shrug.
                return ProbeReport(broken=True, exc=None, err="",
                                   loc="",
                                   timed_out=True, elapsed=elapsed)
            except Exception as exc:  # noqa: BLE001 - captured, not raised
                err = f"{type(exc).__name__}: {exc}"
                loc = ""
                frames = _safe(lambda: traceback.extract_tb(exc.__traceback__))
                if frames:
                    loc = (f"{frames[-1].filename.split('/')[-1]}:"
                           f"{frames[-1].lineno}")
                return ProbeReport(broken=True, exc=exc, err=err, loc=loc,
                                   elapsed=_time.monotonic() - start)
    except Exception:  # noqa: BLE001 - the probe itself must never die
        return ProbeReport(broken=False, exc=None, err="", loc="",
                           elapsed=_time.monotonic() - start)
    return ProbeReport(broken=False, exc=None, err="", loc="",
                       elapsed=_time.monotonic() - start)


def heal_probe(handle_control, kind: str, chat_key: str = "") -> tuple[bool, BaseException | None, str, str]:
    """Probe one command, capturing the exception object.

    Returns (is_broken, exc_or_None, error_str, location_str). Never raises.
    """
    r = _probe_inner(handle_control, kind, chat_key)
    return (r.broken, r.exc, r.err, r.loc)


# ---------------------------------------------------------------------------
# shared AST/file surgery helpers
# ---------------------------------------------------------------------------

def _read_source(filename: str) -> list[str] | None:
    def _get():
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines(keepends=True)
    return _safe(_get)


def diff_preview(filename: str, old_lines: list[str], new_lines: list[str],
                 *, context: int = 3, max_lines: int = 40) -> str:
    """A real unified diff of a proposed source change.

    Used for dry-run previews and heal reports — the owner sees exactly
    what would change before (or after) the write. Capped at
    ``max_lines`` diff lines so a big move doesn't flood the chat.
    """
    import difflib

    short = filename.rsplit("/", 1)[-1]
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{short}", tofile=f"b/{short}", n=context)
    lines = list(diff)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (+{len(lines) - max_lines} more lines)\n"]
    return "".join(lines)


def _write_source(filename: str, lines: list[str]) -> bool:
    def _put():
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True
    return bool(_safe(_put, default=False))


def _compiles(filename: str) -> bool:
    def _check():
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        compile(src, filename, "exec")
        return True
    return bool(_safe(_check, default=False))


def _innermost_function(tree: ast.AST, lineno: int):
    """Innermost FunctionDef/AsyncFunctionDef containing lineno."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or lineno
            if node.lineno <= lineno <= end:
                if best is None or node.lineno > best.lineno:
                    best = node
    return best


def _is_user_code(filename: str) -> bool:
    name = filename.replace("\\", "/")
    if "site-packages" in name or "dist-packages" in name:
        return False
    if name.startswith("<") and name.endswith(">"):
        return False
    return True


def _deepest_user_frame(exc: BaseException):
    """Deepest traceback frame in user code, or None."""
    tb = getattr(exc, "__traceback__", None)
    best = None
    while tb is not None:
        fname = tb.tb_frame.f_code.co_filename
        if _is_user_code(fname):
            best = tb
        tb = tb.tb_next
    return best


# ---------------------------------------------------------------------------
# fix 1: UnboundLocalError → initialize the variable at function top
# ---------------------------------------------------------------------------

def fix_unbound_local(exc: BaseException, dry_run: bool = False) -> dict[str, Any]:
    """Attempt the unbound-variable fix. Returns a result dict; never raises.

    Result keys: ok, detail, diff_preview, skip_reason, filename, varname.
    """
    out = {"ok": False, "detail": "", "diff_preview": "",
           "skip_reason": "", "filename": "", "varname": ""}
    try:
        return _fix_unbound_local_inner(exc, dry_run, out)
    except Exception as e:  # noqa: BLE001
        out["skip_reason"] = f"fix machinery failed: {e}"
        return out


def _fix_unbound_local_inner(exc, dry_run, out):
    m = _VAR_RE.search(str(exc))
    if not m:
        out["skip_reason"] = "could not extract variable name from error"
        return out
    var = m.group(1)
    out["varname"] = var

    frame = _deepest_user_frame(exc)
    if frame is None:
        out["skip_reason"] = "no user-code frame in traceback"
        return out
    filename = frame.tb_frame.f_code.co_filename
    fail_line = frame.tb_lineno
    out["filename"] = filename

    lines = _read_source(filename)
    if lines is None:
        out["skip_reason"] = f"could not read {filename}"
        return out
    src = "".join(lines)
    tree = _safe(lambda: ast.parse(src))
    if tree is None:
        out["skip_reason"] = "could not parse source"
        return out

    func = _innermost_function(tree, fail_line)
    if func is None:
        out["skip_reason"] = "could not locate enclosing function"
        return out

    # Safety: the variable must not be a parameter (initializing a
    # parameter to None would CHANGE behavior on working paths).
    param_names = {a.arg for a in list(func.args.args)
                   + list(func.args.kwonlyargs)}
    if func.args.vararg:
        param_names.add(func.args.vararg.arg)
    if func.args.kwarg:
        param_names.add(func.args.kwarg.arg)
    if var in param_names:
        out["skip_reason"] = (f"'{var}' is a parameter of {func.name}() — "
                              "initializing it would change behavior")
        return out

    # Safety: don't double-initialize. If the first real statement
    # already assigns it (or it's assigned unconditionally at top),
    # there's nothing safe to add.
    body = func.body
    insert_at = 0
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        insert_at = 1  # skip docstring

    # Check whether an unconditional top-level assignment already exists
    # in the first few statements (before any branch).
    for stmt in body[insert_at:insert_at + 3]:
        if isinstance(stmt, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            break
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Name) and node.id == var
                    and isinstance(node.ctx, ast.Store)):
                out["skip_reason"] = (f"'{var}' already initialized near "
                                      f"the top of {func.name}()")
                return out

    # Build the init line with the function's own indentation + 1 level.
    # func.body[0].col_offset gives the body indent.
    body_indent = " " * (body[0].col_offset if body else func.col_offset + 4)
    init_line = f"{body_indent}{var} = None  # auto-heal: init unbound local\n"

    target_lineno = body[insert_at].lineno if insert_at < len(body) else func.end_lineno
    # insert_at is an index into body; convert to 0-based line index
    insert_idx = body[insert_at].lineno - 1 if insert_at < len(body) else len(lines)

    new_lines = lines[:insert_idx] + [init_line] + lines[insert_idx:]
    out["diff_preview"] = diff_preview(filename, lines, new_lines)
    out["detail"] = (f"initialized '{var}' to None at the top of "
                     f"{func.name}() [{filename.rsplit('/', 1)[-1]}:{target_lineno}]")

    if dry_run:
        out["ok"] = True  # would-be fix is well-formed
        return out

    if not _write_source(filename, new_lines):
        out["skip_reason"] = "could not write fixed source"
        return out
    if not _compiles(filename):
        # revert — a fix that doesn't compile is worse than the bug
        _write_source(filename, lines)
        out["skip_reason"] = "fixed source did not compile; reverted"
        return out
    out["ok"] = True
    return out


# ---------------------------------------------------------------------------
# fix 2: dead-code method rescue → move nested _control_* to class level
# ---------------------------------------------------------------------------

def fix_dead_method(exc: BaseException, dry_run: bool = False) -> dict[str, Any]:
    """Attempt the dead-method rescue. Returns a result dict; never raises.

    Only fires when:
    - the error is AttributeError "'Type' object has no attribute '_control_x'"
    - attr starts with "_control_"
    - a nested `def _control_x` exists inside another function in a user-code file
    - the nested function uses NO closure variables from its enclosing scope
    - the target class `Type` exists in the same file and lacks the method
    """
    out = {"ok": False, "detail": "", "diff_preview": "",
           "skip_reason": "", "filename": "", "method": "", "classname": ""}
    try:
        return _fix_dead_method_inner(exc, dry_run, out)
    except Exception as e:  # noqa: BLE001
        out["skip_reason"] = f"fix machinery failed: {e}"
        return out


def _fix_dead_method_inner(exc, dry_run, out):
    m = _ATTR_RE.search(str(exc))
    if not m:
        out["skip_reason"] = "could not parse AttributeError"
        return out
    type_name, attr = m.group(1), m.group(2)
    out["classname"], out["method"] = type_name, attr

    if not attr.startswith("_control_"):
        out["skip_reason"] = (f"'{attr}' is not a _control_* method — "
                              "only dead control methods are rescued")
        return out

    frame = _deepest_user_frame(exc)
    if frame is None:
        out["skip_reason"] = "no user-code frame in traceback"
        return out
    # For a missing-method AttributeError, the traceback points at the
    # CALLER. The nested def lives in the file where the class is defined —
    # resolve it from the object's module.
    filename = None
    f_locals = _safe(lambda: dict(frame.tb_frame.f_locals), default={}) or {}
    for v in f_locals.values():
        if _safe(lambda: type(v).__name__) == type_name:
            modname = _safe(lambda: type(v).__module__)
            mod = sys.modules.get(modname) if modname else None
            filename = _safe(lambda: getattr(mod, "__file__", None))
            if filename:
                break
    if not filename:
        # fall back to the traceback file (may still work if the class
        # and the nested def share a file with the call site)
        filename = frame.tb_frame.f_code.co_filename
    out["filename"] = filename

    lines = _read_source(filename)
    if lines is None:
        out["skip_reason"] = f"could not read {filename}"
        return out
    src = "".join(lines)
    tree = _safe(lambda: ast.parse(src))
    if tree is None:
        out["skip_reason"] = "could not parse source"
        return out

    # Find the nested def: a FunctionDef named `attr` whose parent chain
    # contains another FunctionDef before any ClassDef.
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    nested = None
    enclosing_func = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != attr:
            continue
        # walk up: must hit a FunctionDef before a ClassDef/Module
        cur = parents.get(node)
        seen_func = None
        is_method = False
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if seen_func is None:
                    seen_func = cur
            elif isinstance(cur, ast.ClassDef):
                is_method = True
                break
            elif isinstance(cur, ast.Module):
                break
            cur = parents.get(cur)
        if seen_func is not None and not is_method:
            nested, enclosing_func = node, seen_func
            break

    if nested is None:
        out["skip_reason"] = (f"no nested 'def {attr}' found — not a "
                              "dead-code nesting issue")
        return out

    # Safety: closure check. The nested function must not load names that
    # are locals of the enclosing function (params or assigned names),
    # excluding `self`-style first arg usage and globals.
    enclosing_locals: set[str] = set()
    for a in list(enclosing_func.args.args) + list(enclosing_func.args.kwonlyargs):
        enclosing_locals.add(a.arg)
    for node in ast.walk(enclosing_func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            enclosing_locals.add(node.id)
    # names the nested function loads
    nested_loads = {n.id for n in ast.walk(nested)
                    if isinstance(n, ast.Name)
                    and isinstance(n.ctx, ast.Load)}
    # names defined inside the nested function itself are fine
    nested_defs = {n.id for n in ast.walk(nested)
                   if isinstance(n, ast.Name)
                   and isinstance(n.ctx, ast.Store)}
    nested_defs.add(nested.name)
    dangerous = (nested_loads - nested_defs - {"self"} - {"cls"}
                 ) & enclosing_locals
    # allow the enclosing function's own name (recursion ref) and
    # anything that's clearly a module global/import
    module_names = {n.id for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    and parents.get(n) is not None
                    and isinstance(parents.get(n), ast.Module)}
    dangerous -= module_names
    if dangerous:
        out["skip_reason"] = (f"'{attr}' uses closure variables "
                              f"{sorted(dangerous)} from "
                              f"{enclosing_func.name}() — moving it would "
                              f"break it")
        return out

    # Find the target class in the same file.
    target_cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == type_name:
            target_cls = node
            break
    if target_cls is None:
        out["skip_reason"] = (f"class '{type_name}' not found in "
                              f"{filename.rsplit('/', 1)[-1]}")
        return out

    # The class must not already have the method (avoid duplicates).
    for node in target_cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == attr:
            out["skip_reason"] = (f"'{type_name}' already defines '{attr}' "
                                  "— nothing to rescue")
            return out

    # --- perform the move ---
    # Extract source segment of the nested def, dedent to method level.
    seg = _safe(lambda: ast.get_source_segment(src, nested))
    if not seg:
        out["skip_reason"] = "could not extract nested method source"
        return out
    nested_indent = nested.col_offset
    # method indent = class body indent; take it from the class's first method
    method_indent = None
    for node in target_cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_indent = node.col_offset
            break
    if method_indent is None:
        method_indent = target_cls.col_offset + 4
    dedent_by = nested_indent - method_indent
    if dedent_by < 0:
        out["skip_reason"] = "unexpected indentation — refusing to move"
        return out
    # Note: ast.get_source_segment strips the leading indent from the
    # FIRST line only; subsequent lines keep absolute indentation.
    moved_lines = []
    seg_lines = seg.splitlines(keepends=True)
    for i, ln in enumerate(seg_lines):
        if not ln.strip():
            moved_lines.append(ln)
            continue
        if i == 0:
            moved_lines.append(" " * method_indent + ln.lstrip())
        elif ln.startswith(" " * dedent_by):
            moved_lines.append(ln[dedent_by:])
        else:
            # unexpected — be conservative, re-indent to method level
            moved_lines.append(" " * method_indent + ln.lstrip())

    # Removal: blank the nested def's line range (lineno..end_lineno).
    end_lineno = getattr(nested, "end_lineno", None) or nested.lineno
    new_lines = (lines[:nested.lineno - 1] + lines[end_lineno:])

    # Insertion: after the class's last body statement. Recompute positions
    # on the ORIGINAL tree is stale after removal — instead, find the
    # class by name in the new text via a fresh parse.
    new_src = "".join(new_lines)
    new_tree = _safe(lambda: ast.parse(new_src))
    insert_idx = None
    if new_tree is not None:
        for node in ast.walk(new_tree):
            if isinstance(node, ast.ClassDef) and node.name == type_name:
                last = node.body[-1]
                last_end = getattr(last, "end_lineno", None) or last.lineno
                insert_idx = last_end  # 0-based index == line number (insert after)
                break
    if insert_idx is None:
        out["skip_reason"] = "could not locate insertion point in class"
        return out

    block = ["\n"] + moved_lines
    if not block[-1].endswith("\n"):
        block[-1] += "\n"
    final_lines = new_lines[:insert_idx] + block + new_lines[insert_idx:]

    out["diff_preview"] = diff_preview(filename, lines, final_lines)
    out["detail"] = (f"rescued dead method '{attr}' → {type_name} "
                     f"[{filename.rsplit('/', 1)[-1]}]")

    if dry_run:
        out["ok"] = True
        return out

    if not _write_source(filename, final_lines):
        out["skip_reason"] = "could not write fixed source"
        return out
    if not _compiles(filename):
        _write_source(filename, lines)
        out["skip_reason"] = "fixed source did not compile; reverted"
        return out
    out["ok"] = True
    return out


# ---------------------------------------------------------------------------
# fix strategy registry — dynamic, not hardcoded branches
# ---------------------------------------------------------------------------

@dataclass
class FixStrategy:
    """One auto-fix strategy.

    ``predicate`` decides whether the strategy applies to an exception;
    ``fix`` performs it (same contract as :func:`fix_unbound_local`).
    A ``None`` fix means report-only: the strategy matches, but the fix
    needs a human (``report_reason`` says why).
    """

    kind: str
    predicate: Callable[[BaseException], bool]
    fix: Callable[[BaseException, bool], dict[str, Any]] | None
    report_reason: str = ""


FIX_STRATEGIES: list[FixStrategy] = []


def register_fix_strategy(strategy: FixStrategy) -> FixStrategy:
    """Register a new auto-fix (or report-only) strategy. Returns it."""
    FIX_STRATEGIES.append(strategy)
    return strategy


def _is_unbound_local(exc: BaseException) -> bool:
    return isinstance(exc, UnboundLocalError)


def _is_dead_control_method(exc: BaseException) -> bool:
    m = _ATTR_RE.search(str(exc))
    return bool(m) and m.group(2).startswith("_control_")


def _is_missing_package(exc: BaseException) -> bool:
    return isinstance(exc, (ModuleNotFoundError, ImportError))


register_fix_strategy(FixStrategy("unbound_local", _is_unbound_local,
                                  fix_unbound_local))
register_fix_strategy(FixStrategy("dead_method", _is_dead_control_method,
                                  fix_dead_method))
register_fix_strategy(FixStrategy("missing_package", _is_missing_package, None,
                                  report_reason="needs user action (package install)"))


# ---------------------------------------------------------------------------
# git — each fix committed separately so it's reversible
# ---------------------------------------------------------------------------

def _git_commit(repo_root: str, files: list[str], message: str) -> bool:
    """Commit the fix. Best-effort; never raises. Returns True if committed."""
    def _do():
        if not os.path.isdir(os.path.join(repo_root, ".git")):
            return False
        env = dict(os.environ, GIT_EDITOR="true")
        subprocess.run(["git", "add"] + files, cwd=repo_root, env=env,
                       capture_output=True, timeout=30)
        r = subprocess.run(["git", "commit", "-m", message], cwd=repo_root,
                           env=env, capture_output=True, timeout=30)
        return r.returncode == 0
    return bool(_safe(_do, default=False))


def _repo_root_for(filename: str) -> str:
    d = os.path.dirname(os.path.abspath(filename))
    while d and d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return ""


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def _fingerprint_of(exc: BaseException) -> str:
    """Error fingerprint for the learning loop. Never raises."""
    try:
        from .error_intelligence import ErrorFingerprint
        return ErrorFingerprint.of(exc)
    except Exception:  # noqa: BLE001 - fingerprinting never breaks healing
        return ""


def _report_outcome(intelligence: Any, fingerprint: str, fixed: bool,
                    fix_detail: str) -> None:
    """Feed the heal outcome back into the learning KB. Never raises."""
    if intelligence is None or not fingerprint:
        return
    try:
        intelligence.record_outcome(fingerprint, fixed=fixed,
                                    fix=fix_detail or "")
    except Exception:  # noqa: BLE001 - learning never breaks healing
        pass


def heal_one(handle_control, kind: str, chat_key: str = "",
             dry_run: bool = False, intelligence: Any = None) -> HealResult:
    """Probe one command; diagnose; auto-fix if a safe strategy matches.

    ``intelligence`` (an :class:`~nomorals.core.error_intelligence.ErrorIntelligence`)
    closes the learning loop: verified fixes and failed attempts are
    reported back per error fingerprint. Never raises. In dry-run mode
    nothing is written and nothing is learned.
    """
    res = HealResult(kind)
    res.dry_run = dry_run
    try:
        return _heal_one_inner(handle_control, kind, chat_key, dry_run, res,
                               intelligence)
    except Exception as e:  # noqa: BLE001
        res.skip_reason = f"heal machinery failed: {e}"
        return res


def _heal_one_inner(handle_control, kind, chat_key, dry_run, res, intelligence):
    report = _probe_inner(handle_control, kind, chat_key)
    res.timed_out = report.timed_out
    if report.timed_out:
        # A hung command is a finding with its own report — not "healthy",
        # and never auto-fixed (we don't know WHY it hangs).
        res.broken = True
        res.error = (f"probe timed out after {report.elapsed:.1f}s — "
                     f"/{kind} hangs")
        res.fixable = False
        res.skip_reason = ("probe timed out — the command hangs; no safe "
                           "auto-fix for hangs, investigate the blocking call")
        res.suggestion = ("The command did not return within "
                          f"{_PROBE_TIMEOUT:.0f}s. Look for a blocking call "
                          "without a timeout (network I/O, locks, joins).")
        return res

    broken, exc, err, loc = report.broken, report.exc, report.err, report.loc
    if not broken or exc is None:
        res.broken = False
        return res
    res.broken = True
    res.error = err
    res.location = loc
    res.fingerprint = _fingerprint_of(exc)

    diag = _safe(lambda: diagnose(exc, context={"command": f"/{kind}"}))
    res.diagnosis = diag
    res.suggestion = (diag.get("suggested_fix", "") if diag else "")

    # strategy lookup — dynamic registry, not hardcoded branches
    strategy = next(
        (s for s in FIX_STRATEGIES
         if _safe(lambda s=s: s.predicate(exc), default=False)),
        None)
    if strategy is None:
        res.fixable = False
        res.skip_reason = "no safe auto-fix pattern matches"
        return res
    if strategy.fix is None:
        res.fixable = False
        res.skip_reason = strategy.report_reason or "needs user action"
        return res

    res.fix_kind = strategy.kind
    res.fixable = True
    fix_out = strategy.fix(exc, dry_run=dry_run)
    res.fix_detail = fix_out.get("detail", "")
    res.diff_preview = fix_out.get("diff_preview", "")
    if not fix_out.get("ok"):
        res.fixable = False
        res.skip_reason = fix_out.get("skip_reason", "fix failed")
        return res

    if dry_run:
        return res

    # commit the fix for reversibility
    filename = fix_out.get("filename", "")
    if filename:
        repo = _repo_root_for(filename)
        if repo:
            msg = (f"fix(heal): /{kind} — {res.fix_detail[:80]}"
                   if res.fix_detail else f"fix(heal): /{kind} auto-fix")
            res.committed = _git_commit(repo, [filename], msg)

    # verify: re-probe the command
    verify = _probe_inner(handle_control, kind, chat_key)
    res.fixed = not verify.broken
    if not res.fixed:
        res.skip_reason = "fix applied but command still broken"
    # learning loop: teach the error catcher what happened
    _report_outcome(intelligence, res.fingerprint, res.fixed, res.fix_detail)
    return res


def heal_all(handle_control, kinds: list[str], chat_key: str = "",
             dry_run: bool = False,
             skip: frozenset = frozenset(),
             intelligence: Any = None) -> dict[str, Any]:
    """Run the full loop over candidate commands. Never raises."""
    results: list[HealResult] = []
    try:
        for kind in kinds:
            if kind in skip:
                continue
            try:
                results.append(heal_one(handle_control, kind, chat_key,
                                        dry_run=dry_run,
                                        intelligence=intelligence))
            except Exception:  # noqa: BLE001 - one bad apple never kills heal
                r = HealResult(kind)
                r.skip_reason = "heal probe crashed unexpectedly"
                results.append(r)
    except Exception:  # noqa: BLE001
        pass

    healed = [r for r in results if r.fixed]
    still = [r for r in results if r.broken and not r.fixed]
    hung = [r for r in results if r.timed_out]
    would = [r for r in results if r.dry_run and r.fixable and r.broken]
    return {
        "results": [r.to_dict() for r in results],
        "healed": [r.to_dict() for r in healed],
        "still_broken": [r.to_dict() for r in still],
        "timed_out": [r.to_dict() for r in hung],
        "would_fix": [r.to_dict() for r in would],
        "total": len(results),
        "dry_run": dry_run,
    }
