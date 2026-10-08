"""Dynamic error diagnostician.

When a command crashes, don't just report the exception — diagnose the
ROOT CAUSE by inspecting the failing code itself:

* walks the traceback frames and shows source around each failure point
* for ``UnboundLocalError``: parses the failing function's AST, finds every
  assignment of the variable and every reference, reconstructs which
  branch was (not) taken using the live frame locals
* for ``AttributeError``: suggests the closest real attribute (difflib)
* for ``KeyError``: shows the available keys
* for ``ModuleNotFoundError``: tells you the exact ``pip install`` command
* for ``ConnectionError``/``TimeoutError``: extracts the host/endpoint
* for ``TypeError``: reports the operand types involved

Pure stdlib. ``diagnose()`` never raises — the diagnostician is wrapped
end-to-end so a broken diagnosis can never break an error reply.

Usage:
    from nomorals.core.error_doctor import diagnose

    try:
        ...
    except Exception as exc:
        d = diagnose(exc)          # structured dict, never raises
        print(d["summary"])         # one-paragraph human summary
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import linecache
import re
import sys
import traceback
from typing import Any

__all__ = ["diagnose", "diagnosis_to_text"]

# ---------------------------------------------------------------------------
# bulletproofing helpers
# ---------------------------------------------------------------------------

def _safe(fn, default=None):
    """Run fn(); return default on ANY failure. The doctor never gets sick."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 - bulletproof by design
        return default


def _trunc(value: Any, limit: int = 120) -> str:
    s = _safe(lambda: repr(value), default="<unrepresentable>")
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s


# ---------------------------------------------------------------------------
# traceback walking
# ---------------------------------------------------------------------------

def _walk_tb(exc: BaseException) -> list[dict[str, Any]]:
    """Walk the raw traceback; capture filename, lineno, function, a
    snapshot of locals, and 3 lines of source around each frame."""
    frames: list[dict[str, Any]] = []
    tb = getattr(exc, "__traceback__", None)
    seen = 0
    while tb is not None and seen < 32:
        seen += 1
        frame = tb.tb_frame
        code = frame.f_code
        filename = code.co_filename
        lineno = tb.tb_lineno
        loc = _safe(lambda: dict(frame.f_locals), default={}) or {}
        # snapshot locals cheaply — repr truncated, failures dropped
        snapshot: dict[str, str] = {}
        for k, v in loc.items():
            if k.startswith("__"):
                continue
            r = _safe(lambda v=v: _trunc(v))
            if r is not None:
                snapshot[str(k)] = r
            if len(snapshot) >= 24:
                break
        frames.append({
            "filename": filename,
            "short": filename.rsplit("/", 1)[-1] if "/" in filename else filename,
            "lineno": lineno,
            "func": code.co_name,
            "locals": snapshot,
            "raw_locals": loc,
            "source": _source_lines(filename, lineno, context=1),
            "user_code": _is_user_code(filename),
        })
        tb = tb.tb_next
    return frames


def _source_lines(filename: str, lineno: int, context: int = 1) -> list[str]:
    """N lines of source around lineno (context lines each side)."""
    out: list[str] = []

    def _get() -> list[str]:
        for off in range(-context, context + 1):
            line = linecache.getline(filename, lineno + off)
            if line:
                marker = ">>>" if off == 0 else "   "
                out.append(f"{marker} {lineno + off:4d}: {line.rstrip()}")
        return out

    return _safe(_get, default=[]) or []


def _is_user_code(filename: str) -> bool:
    name = filename.replace("\\", "/")
    if "site-packages" in name or "dist-packages" in name:
        return False
    stdlib = _safe(lambda: sys.base_prefix, default="") or ""
    if stdlib and name.startswith(stdlib.replace("\\", "/")):
        # .../lib/python3.x/... is stdlib; anything else under the prefix
        # (e.g. a venv's own code) counts as user code
        rest = name[len(stdlib.replace("\\", "/")):]
        if "/lib/python" in rest:
            return False
    if name.startswith("<") and name.endswith(">"):  # <frozen ...>, <string>
        return False
    return True


def _target_frame(frames: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The frame where the error was raised — prefer the deepest user-code
    frame, else the deepest frame overall."""
    if not frames:
        return None
    for f in reversed(frames):
        if f["user_code"]:
            return f
    return frames[-1]


# ---------------------------------------------------------------------------
# AST helpers for UnboundLocalError analysis
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r"local variable '([^']+)'")


def _parse_function_ast(filename: str, lineno: int):
    """Parse the file; return (tree, innermost FunctionDef containing lineno,
    source text) or (None, None, None)."""
    def _get():
        src = open(filename, "r", encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
        best = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", None) or lineno
                if node.lineno <= lineno <= end:
                    if best is None or node.lineno > best.lineno:
                        best = node
        return tree, best, src

    res = _safe(_get)
    if not res:
        return None, None, None
    return res


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _src_segment(node: ast.AST, source: str) -> str:
    seg = _safe(lambda: ast.get_source_segment(source, node))
    if seg:
        seg = " ".join(seg.split())
        return seg if len(seg) <= 60 else seg[:57] + "..."
    return ""


def _branch_context(node: ast.AST, parents: dict[ast.AST, ast.AST],
                    source: str) -> list[str]:
    """Describe the enclosing if/try/loop/with blocks of a node,
    innermost first."""
    parts: list[str] = []
    cur: ast.AST | None = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.If):
            cond = _src_segment(cur.test, source) or "condition"
            parts.append(f"inside `if {cond}` (line {cur.lineno})")
        elif isinstance(cur, ast.ExceptHandler):
            parts.append(f"inside `except` handler (line {cur.lineno})")
        elif isinstance(cur, ast.Try):
            parts.append(f"inside `try` block (line {cur.lineno})")
        elif isinstance(cur, (ast.For, ast.AsyncFor)):
            parts.append(f"inside `for` loop (line {cur.lineno})")
        elif isinstance(cur, ast.While):
            parts.append(f"inside `while` loop (line {cur.lineno})")
        elif isinstance(cur, (ast.With, ast.AsyncWith)):
            parts.append(f"inside `with` block (line {cur.lineno})")
        elif isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.Lambda, ast.Module)):
            break
    return parts


def _condition_values(test: ast.AST, f_locals: dict[str, Any]) -> str:
    """Best-effort: show the runtime values of names in a branch condition."""
    names = sorted({n.id for n in ast.walk(test)
                    if isinstance(n, ast.Name)})
    bits = []
    for name in names[:6]:
        if name in f_locals:
            bits.append(f"{name}={_trunc(f_locals[name], 40)}")
    return ", ".join(bits)


def _analyze_unbound_local(exc: BaseException, frames: list[dict[str, Any]],
                           target: dict[str, Any]) -> tuple[str, dict, str]:
    m = _VAR_RE.search(str(exc))
    var = m.group(1) if m else "?"
    evidence: dict[str, Any] = {"variable": var}
    filename = target["filename"]
    fail_line = target["lineno"]
    f_locals: dict[str, Any] = target.get("raw_locals", {})

    tree, func, source = _parse_function_ast(filename, fail_line)
    if tree is None or func is None or source is None:
        root = (f"'{var}' was referenced before assignment at "
                f"{target['short']}:{fail_line}, but the source could not "
                f"be read for deeper analysis.")
        return root, evidence, (f"Initialize '{var}' before the branch that "
                                f"assigns it (e.g. `{var} = None`), or move "
                                f"the reference inside that branch.")

    parents = _parent_map(tree)
    stores: list[tuple[int, list[str], ast.AST | None]] = []  # (line, ctx, if-node)
    loads: list[int] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == var:
            if isinstance(node.ctx, ast.Store):
                ctx = _branch_context(node, parents, source)
                # nearest enclosing If, for condition values
                if_node = None
                cur: ast.AST | None = node
                while cur in parents:
                    cur = parents[cur]
                    if isinstance(cur, ast.If):
                        if_node = cur
                        break
                    if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        break
                stores.append((node.lineno, ctx, if_node))
            elif isinstance(node.ctx, ast.Load):
                loads.append(node.lineno)

    evidence["assigned_at_lines"] = sorted({s[0] for s in stores})
    evidence["referenced_at_lines"] = sorted(set(loads))
    evidence["failing_line"] = fail_line

    if not stores:
        root = (f"'{var}' is never assigned anywhere in "
                f"`{func.name}` — it is referenced at line {fail_line} "
                f"but no assignment exists.")
        fix = f"Assign '{var}' before line {fail_line}."
        return root, evidence, fix

    # describe each assignment's branch + live condition values
    assign_descs = []
    cond_bits = []
    for line, ctx, if_node in stores:
        if ctx:
            assign_descs.append(f"line {line} ({'; '.join(ctx)})")
        else:
            assign_descs.append(f"line {line} (unconditional)")
        if if_node is not None:
            vals = _condition_values(if_node.test, f_locals)
            cond = _src_segment(if_node.test, source) or "condition"
            if vals:
                cond_bits.append(f"`{cond}` evaluated with {vals}")

    fail_ctx = []
    # find the Load node at the failing line for its context
    for node in ast.walk(func):
        if (isinstance(node, ast.Name) and node.id == var
                and isinstance(node.ctx, ast.Load)
                and getattr(node, "lineno", None) == fail_line):
            fail_ctx = _branch_context(node, parents, source)
            break

    if fail_ctx:
        where = "; ".join(fail_ctx)
        root = (f"'{var}' is assigned only at {', '.join(assign_descs)}, "
                f"but the crash at line {fail_line} sits {where} — a path "
                f"where the assignment never ran.")
    else:
        root = (f"'{var}' is assigned only at {', '.join(assign_descs)}, "
                f"but it is referenced at line {fail_line} outside those "
                f"branches — when the branch condition is false, the "
                f"reference hits an unassigned variable.")
    if cond_bits:
        root += f" At crash time: {'; '.join(cond_bits)}."
    evidence["condition_values"] = cond_bits

    fix = (f"Initialize '{var}' before the branch (e.g. `{var} = None` at "
           f"the top of `{func.name}`), or move the line-{fail_line} "
           f"reference inside the branch that assigns it.")
    return root, evidence, fix


# ---------------------------------------------------------------------------
# type-specific analyzers
# ---------------------------------------------------------------------------

_ATTR_RE = re.compile(r"'([^']+)' object has no attribute '([^']+)'")
_MOD_RE = re.compile(r"No module named '([^']+)'")


def _analyze_import_error(exc: BaseException, frames, target):
    mod = getattr(exc, "name", None)
    if not mod:
        m = _MOD_RE.search(str(exc))
        mod = m.group(1) if m else "?"
    top = mod.split(".")[0]
    evidence: dict[str, Any] = {"module": mod, "top_level": top}

    spec = _safe(lambda: importlib.util.find_spec(top))
    stdlib_names: set[str] = _safe(lambda: set(sys.stdlib_module_names),
                                   default=set()) or set()
    if top in stdlib_names:
        root = (f"'{mod}' is a standard-library module, so this is not a "
                f"missing install — the import itself is broken "
                f"({exc}).")
        fix = ("Check for a local file shadowing it (e.g. a "
               f"`{top}.py` in the working directory) or a broken "
               f"partial install.")
    elif spec is None:
        root = f"Python module '{mod}' is not installed in this environment."
        fix = f"Install it: `pip install {top}`"
        # common alias corrections
        aliases = {"PIL": "pillow", "cv2": "opencv-python", "sklearn": "scikit-learn",
                   "yaml": "pyyaml", "bs4": "beautifulsoup4"}
        if top in aliases:
            fix += f"  (package name is `{aliases[top]}`)"
    else:
        root = (f"'{top}' resolves to {spec.origin}, but importing '{mod}' "
                f"still failed ({exc}) — the package is present but broken.")
        fix = (f"Reinstall it: `pip install --force-reinstall {top}`, or "
               f"check for a partially-initialized submodule.")
    evidence["spec_origin"] = getattr(spec, "origin", None)
    return root, evidence, fix


def _analyze_attribute_error(exc: BaseException, frames, target):
    m = _ATTR_RE.search(str(exc))
    type_name = m.group(1) if m else "?"
    attr = m.group(2) if m else "?"
    evidence: dict[str, Any] = {"type": type_name, "attribute": attr}
    f_locals: dict[str, Any] = target.get("raw_locals", {}) if target else {}

    obj = None
    for v in f_locals.values():
        tname = _safe(lambda: type(v).__name__)
        if tname == type_name:
            obj = v
            break
    if obj is None:
        root = (f"Attribute '{attr}' does not exist on {type_name} "
                f"(raised at {target['short']}:{target['lineno']}).")
        return root, evidence, (f"Check the spelling of '{attr}' — the "
                                f"object could not be inspected live.")

    names = _safe(lambda: [n for n in dir(obj) if not n.startswith("_")],
                  default=[]) or []
    close = difflib.get_close_matches(attr, names, n=3, cutoff=0.6)
    evidence["close_matches"] = close
    evidence["available_count"] = len(names)
    if close:
        root = (f"'{attr}' does not exist on {type_name}; the closest real "
                f"attributes are: {', '.join(close)}.")
        fix = f"Did you mean `.{close[0]}`?"
    else:
        sample = ", ".join(names[:8])
        root = (f"'{attr}' does not exist on {type_name}. Available "
                f"attributes include: {sample}{'...' if len(names) > 8 else ''}.")
        fix = f"Use one of the available attributes instead of '.{attr}'."
    return root, evidence, fix


def _analyze_key_error(exc: BaseException, frames, target):
    key = exc.args[0] if exc.args else "?"
    evidence: dict[str, Any] = {"missing_key": _trunc(key, 60)}
    f_locals: dict[str, Any] = target.get("raw_locals", {}) if target else {}
    dicts = {k: v for k, v in f_locals.items() if isinstance(v, dict)}
    shown: dict[str, list[str]] = {}
    for name, d in list(dicts.items())[:3]:
        keys = list(d.keys())[:10]
        shown[name] = [_trunc(k, 40) for k in keys]
        if len(d) > 10:
            shown[name].append(f"... +{len(d) - 10} more")
    evidence["dicts_in_scope"] = shown
    if shown:
        first = next(iter(shown))
        root = (f"Key {_trunc(key, 60)} is not in dict '{first}'. "
                f"Available keys: {', '.join(shown[first])}.")
        fix = (f"Use `dict.get(key)` with a default, or check the key "
               f"spelling against the available keys.")
    else:
        root = f"Key {_trunc(key, 60)} was not found in a dict."
        fix = "Use `dict.get(key)` with a default instead of `dict[key]`."
    return root, evidence, fix


_HOST_RE = re.compile(
    r"(https?://[^\s'\"<>]+)|([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(:\d+)?")


def _analyze_connection_error(exc: BaseException, frames, target):
    text = str(exc) or repr(exc)
    m = _HOST_RE.search(text)
    host = m.group(0) if m else None
    # also check common arg shapes: ConnectionError(host), (errno, msg)
    if not host:
        for a in exc.args:
            if isinstance(a, str):
                m2 = _HOST_RE.search(a)
                if m2:
                    host = m2.group(0)
                    break
    evidence: dict[str, Any] = {"host": host, "message": _trunc(text)}
    if host:
        root = f"Could not reach {host} — the network request failed ({type(exc).__name__})."
        fix = (f"Check connectivity to {host} (is the network up? is the "
               f"host correct?), then retry. Transient failures clear on "
               f"their own.")
    else:
        root = f"A network connection failed ({type(exc).__name__}: {_trunc(text, 80)})."
        fix = "Check the network connection and retry; use backoff on repeats."
    return root, evidence, fix


def _analyze_type_error(exc: BaseException, frames, target):
    text = str(exc)
    f_locals: dict[str, Any] = target.get("raw_locals", {}) if target else {}
    # report operand types visible in scope — best effort
    sample = {k: _safe(lambda v=v: type(v).__name__, default="?")
              for k, v in list(f_locals.items())[:12]}
    evidence: dict[str, Any] = {"message": _trunc(text),
                                "operand_types_in_scope": sample}
    # static pass: the failing line's AST reveals literal operand types
    # (e.g. "a" + 1 has no locals at all)
    static_types: list[str] = []
    if target:
        tree, _func, _src = _parse_function_ast(target["filename"],
                                                target["lineno"])
        if tree is not None:
            for node in ast.walk(tree):
                if getattr(node, "lineno", None) != target["lineno"]:
                    continue
                if isinstance(node, ast.BinOp):
                    for side in (node.left, node.right):
                        if isinstance(side, ast.Constant):
                            static_types.append(type(side.value).__name__)
                        elif (isinstance(side, ast.Name)
                                and side.id in f_locals):
                            static_types.append(
                                _safe(lambda: type(f_locals[side.id]).__name__,
                                      default="?") or "?")
                    break
    if static_types:
        evidence["operand_types_on_line"] = static_types
    root = f"Wrong types for an operation: {text}"
    if len(static_types) >= 2:
        root += f" (operands are {static_types[0]} and {static_types[1]})"
    fix = ("Convert the operands to compatible types before the operation "
           "(e.g. `str(x)` / `int(x)`), or guard with isinstance checks.")
    return root, evidence, fix


def _analyze_name_error(exc: BaseException, frames, target):
    m = re.search(r"name '([^']+)' is not defined", str(exc))
    name = m.group(1) if m else "?"
    f_locals: dict[str, Any] = target.get("raw_locals", {}) if target else {}
    close = difflib.get_close_matches(name, list(f_locals.keys()),
                                      n=3, cutoff=0.6)
    evidence: dict[str, Any] = {"name": name, "close_matches": close}
    if close:
        root = f"'{name}' is not defined — did you mean '{close[0]}'?"
        fix = f"Check the spelling; the similar name '{close[0]}' exists in scope."
    else:
        root = f"'{name}' is not defined in this scope."
        fix = f"Define '{name}' before use, or check for a typo."
    return root, evidence, fix


def _analyze_generic(exc: BaseException, frames, target):
    loc = f"{target['short']}:{target['lineno']}" if target else "?"
    root = f"{type(exc).__name__}: {exc} (raised at {loc})"
    evidence: dict[str, Any] = {"message": _trunc(str(exc))}
    fix = "Check the logs for the full traceback; no specific fix known."
    return root, evidence, fix


# dispatch over the MRO so subclasses hit the right analyzer
_ANALYZERS: list[tuple[type, Any]] = [
    (UnboundLocalError, _analyze_unbound_local),
    (ModuleNotFoundError, _analyze_import_error),
    (ImportError, _analyze_import_error),
    (AttributeError, _analyze_attribute_error),
    (KeyError, _analyze_key_error),
    (ConnectionError, _analyze_connection_error),
    (TimeoutError, _analyze_connection_error),
    (TypeError, _analyze_type_error),
    (NameError, _analyze_name_error),
]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def diagnose(exc: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Diagnose an exception. NEVER raises — on any internal failure returns
    a minimal generic diagnosis."""
    try:
        return _diagnose_inner(exc, context or {})
    except Exception as fatal:  # noqa: BLE001 - the doctor never gets sick
        etype = _safe(lambda: type(exc).__name__, default="?")
        return {
            "error_type": etype,
            "location": "?",
            "root_cause": f"{etype}: {exc}",
            "evidence": {"diagnostician_error": str(fatal)},
            "suggested_fix": "Check the logs for the full traceback.",
            "frames": [],
            "summary": f"{etype}: {exc}",
        }


def _diagnose_inner(exc: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(exc, BaseException):
        return {
            "error_type": type(exc).__name__,
            "location": "?",
            "root_cause": f"diagnose() got a non-exception: {_trunc(exc)}",
            "evidence": {},
            "suggested_fix": "Pass a real exception object.",
            "frames": [],
            "summary": f"Not an exception: {_trunc(exc, 80)}",
        }

    error_type = type(exc).__name__
    frames = _walk_tb(exc)
    target = _target_frame(frames)
    location = (f"{target['short']}:{target['lineno']}"
                if target else "?")

    analyzer = _analyze_generic
    for cls, fn in _ANALYZERS:
        if isinstance(exc, cls):
            analyzer = fn
            break

    root_cause, evidence, suggested_fix = _safe(
        lambda: analyzer(exc, frames, target),
        default=(f"{error_type}: {exc}", {}, "Check the logs."),
    )

    # strip raw locals out of the public frames (too big / noisy)
    pub_frames = []
    for f in frames:
        pub_frames.append({k: v for k, v in f.items() if k != "raw_locals"})

    diag = {
        "error_type": error_type,
        "location": location,
        "root_cause": root_cause,
        "evidence": evidence,
        "suggested_fix": suggested_fix,
        "frames": pub_frames,
        "summary": "",
    }
    if context.get("command"):
        diag["evidence"]["command"] = context["command"]
    diag["summary"] = diagnosis_to_text(diag)
    return diag


def diagnosis_to_text(diag: dict[str, Any]) -> str:
    """One-paragraph human-readable summary."""
    def _get() -> str:
        etype = diag.get("error_type", "?")
        loc = diag.get("location", "?")
        cause = diag.get("root_cause", "")
        fix = diag.get("suggested_fix", "")
        parts = [f"{etype} at {loc}: {cause}"]
        if fix:
            parts.append(f"Fix: {fix}")
        return " ".join(parts)

    return _safe(_get, default="Error diagnosis unavailable.") or \
        "Error diagnosis unavailable."
