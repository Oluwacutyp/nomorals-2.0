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
* for ``IndexError``: shows the index vs the sequence length
* for ``ZeroDivisionError``: shows the divisor's live value
* for ``RecursionError``: identifies the recursive cycle in the traceback
* for ``FileNotFoundError``: suggests the closest real filename (did-you-mean)
* for ``JSONDecodeError``: points at the offending character with a snippet
* for ``UnicodeDecodeError``: shows the undecodable byte range
* for ``AssertionError``: shows the failed expression and its operand values
* for ``OSError``: names the errno
* for ``ValueError``: extracts the offending literal (e.g. int("abc"))
* chained exceptions (``__cause__``/``__context__``) are diagnosed recursively

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
import os
import re
import sys
import traceback
from typing import Any

__all__ = ["diagnose", "diagnosis_to_text", "format_diagnosis"]

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


def _analyze_index_error(exc: BaseException, frames, target):
    f_locals: dict[str, Any] = target.get("raw_locals", {}) if target else {}
    seqs = {k: len(v) for k, v in f_locals.items()
            if isinstance(v, (list, tuple, str, bytes)) and not k.startswith("_")}
    evidence: dict[str, Any] = {"sequences_in_scope": dict(list(seqs.items())[:8])}
    # static pass: find the subscript on the failing line
    index_repr = None
    if target:
        tree, _func, _src = _parse_function_ast(target["filename"], target["lineno"])
        if tree is not None:
            for node in ast.walk(tree):
                if getattr(node, "lineno", None) != target["lineno"]:
                    continue
                if isinstance(node, ast.Subscript):
                    sl = node.slice
                    if isinstance(sl, ast.Constant):
                        index_repr = repr(sl.value)
                    elif isinstance(sl, ast.Name) and sl.id in f_locals:
                        index_repr = f"{sl.id}={_trunc(f_locals[sl.id], 40)}"
                    elif isinstance(sl, ast.Name):
                        index_repr = sl.id
                    break
    evidence["index"] = index_repr
    if seqs and index_repr:
        biggest = max(seqs.items(), key=lambda kv: kv[1])
        root = (f"Index {index_repr} is out of range "
                f"(sequences in scope: {', '.join(f'{k}[len {v}]' for k, v in list(seqs.items())[:5])}).")
        fix = (f"Bounds-check before indexing (0 <= i < len(seq)), or use "
               f"a slice / .get-style access that tolerates the edge.")
    elif seqs:
        root = f"Index out of range. Sequences in scope: {evidence['sequences_in_scope']}."
        fix = "Bounds-check the index against len(seq) before indexing."
    else:
        root = f"Index out of range: {exc}."
        fix = "Bounds-check the index against len(seq) before indexing."
    return root, evidence, fix


def _analyze_zero_division(exc: BaseException, frames, target):
    f_locals: dict[str, Any] = target.get("raw_locals", {}) if target else {}
    divisor_repr = None
    op_name = "/"
    if target:
        tree, _func, _src = _parse_function_ast(target["filename"], target["lineno"])
        if tree is not None:
            for node in ast.walk(tree):
                if getattr(node, "lineno", None) != target["lineno"]:
                    continue
                if isinstance(node, ast.BinOp) and isinstance(
                        node.op, (ast.Div, ast.Mod, ast.FloorDiv)):
                    op_name = {ast.Div: "/", ast.Mod: "%",
                               ast.FloorDiv: "//"}[type(node.op)]
                    right = node.right
                    if isinstance(right, ast.Name):
                        val = f_locals.get(right.id, "<unknown>")
                        divisor_repr = f"{right.id}={_trunc(val, 40)}"
                    elif isinstance(right, ast.Constant):
                        divisor_repr = repr(right.value)
                    else:
                        divisor_repr = _src_segment(right, _src) or "expression"
                    break
    evidence: dict[str, Any] = {"operator": op_name, "divisor": divisor_repr}
    if divisor_repr:
        root = f"Division by zero: the divisor ({divisor_repr}) was 0 in `{op_name}`."
    else:
        root = f"Division by zero at {target['short']}:{target['lineno']}." if target else \
            "Division by zero."
    fix = ("Guard the divisor (`if divisor != 0:`) or decide what the "
           "expression should yield when it is zero.")
    return root, evidence, fix


def _analyze_recursion(exc: BaseException, frames, target):
    func = target["func"] if target else "?"
    # Walk the FULL traceback: `frames` is capped at 32, but a recursion
    # error can have thousands. Count the cycle directly.
    total = 0
    in_cycle = 0
    others: set[str] = set()
    tb = getattr(exc, "__traceback__", None)
    seen_tb = 0
    while tb is not None and seen_tb < 100000:
        seen_tb += 1
        total += 1
        name = _safe(lambda: tb.tb_frame.f_code.co_name, default="?")
        if name == func:
            in_cycle += 1
        else:
            others.add(str(name))
        tb = tb.tb_next
    evidence: dict[str, Any] = {
        "recursive_function": func,
        "frames_in_cycle": in_cycle,
        "total_frames": total,
        "other_frames": sorted(others)[:8],
    }
    root = (f"Maximum recursion depth exceeded: `{func}` called itself "
            f"{in_cycle} times without returning — the base case never hit.")
    fix = (f"Check the base case of `{func}` (does every path return without "
           f"recursing?), or rewrite the loop iteratively to remove the "
           f"depth limit entirely.")
    return root, evidence, fix


def _analyze_file_not_found(exc: BaseException, frames, target):
    filename = getattr(exc, "filename", None) or (exc.args[0] if exc.args else "?")
    evidence: dict[str, Any] = {"filename": _trunc(filename, 120)}
    suggestions: list[str] = []
    directory = _safe(
        lambda: os.path.dirname(os.path.abspath(str(filename)))) \
        if filename else None
    if directory:
        entries = _safe(lambda: os.listdir(directory), default=[]) or []
        base = str(filename).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        suggestions = difflib.get_close_matches(base, entries, n=3, cutoff=0.6)
    evidence["close_matches"] = suggestions
    evidence["directory"] = directory
    if suggestions:
        root = (f"File not found: {filename!r}. Did you mean "
                f"{', '.join(repr(s) for s in suggestions)} in {directory}?")
        fix = f"Check the filename spelling — {suggestions[0]!r} exists in that directory."
    else:
        root = f"File not found: {filename!r}."
        fix = ("Verify the path is correct and the file exists (check the "
               "working directory — relative paths resolve against it).")
    return root, evidence, fix


def _analyze_json_error(exc: BaseException, frames, target):
    doc = getattr(exc, "doc", "") or ""
    pos = getattr(exc, "pos", 0) or 0
    lineno = getattr(exc, "lineno", "?")
    colno = getattr(exc, "colno", "?")
    msg = getattr(exc, "msg", str(exc))
    start = max(0, pos - 30)
    snippet = doc[start:pos + 30]
    caret = " " * min(30, pos - start) + "^"
    evidence: dict[str, Any] = {
        "message": _trunc(msg, 80),
        "line": lineno, "column": colno, "char": pos,
        "snippet": snippet[:80],
    }
    root = (f"Invalid JSON at line {lineno}, column {colno} ({msg}):\n"
            f"    ...{snippet}...\n"
            f"    {' ' * 4}{caret}")
    fix = ("The input is not valid JSON at the marked character — check for "
           "truncated responses, HTML error pages, or trailing commas. "
           "Validate with a JSON linter before parsing.")
    return root, evidence, fix


def _analyze_unicode_error(exc: BaseException, frames, target):
    encoding = getattr(exc, "encoding", "?")
    start = getattr(exc, "start", None)
    end = getattr(exc, "end", None)
    obj = getattr(exc, "object", b"")
    bad = _safe(lambda: bytes(obj[start:end]) if isinstance(obj, (bytes, bytearray))
                and start is not None else b"", default=b"") or b""
    evidence: dict[str, Any] = {
        "encoding": encoding,
        "byte_range": [start, end],
        "bad_bytes": repr(bad)[:60],
    }
    root = (f"Cannot decode bytes {start}:{end} ({bad!r}) as {encoding} — "
            f"the input is not valid {encoding}.")
    fix = ("Decode with the correct encoding, or use "
           "`data.decode(encoding, errors='replace')` to tolerate bad bytes.")
    return root, evidence, fix


def _analyze_assertion(exc: BaseException, frames, target):
    f_locals: dict[str, Any] = target.get("raw_locals", {}) if target else {}
    expr_src = None
    values: dict[str, str] = {}
    if target:
        tree, _func, source = _parse_function_ast(target["filename"], target["lineno"])
        if tree is not None and source:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert) and node.lineno == target["lineno"]:
                    expr_src = _src_segment(node.test, source)
                    for n in ast.walk(node.test):
                        if isinstance(n, ast.Name) and n.id in f_locals \
                                and n.id not in values and len(values) < 8:
                            values[n.id] = _trunc(f_locals[n.id], 40)
                    break
    evidence: dict[str, Any] = {"expression": expr_src, "operand_values": values}
    detail = f": {exc}" if str(exc) else ""
    if expr_src:
        root = f"Assertion failed{detail}: `{expr_src}` was False."
        if values:
            root += f" Values: {', '.join(f'{k}={v}' for k, v in values.items())}."
        fix = ("The asserted condition does not hold — inspect the operand "
               "values above to see which side broke the invariant.")
    else:
        root = f"Assertion failed{detail}."
        fix = "The asserted condition does not hold; add a message to the assert to say what was expected."
    return root, evidence, fix


def _analyze_os_error(exc: BaseException, frames, target):
    import errno as _errno
    err_no = getattr(exc, "errno", None)
    name = _errno.errorcode.get(err_no, "?") if err_no else "?"
    strerror = getattr(exc, "strerror", None) or str(exc)
    filename = getattr(exc, "filename", None)
    evidence: dict[str, Any] = {"errno": err_no, "errno_name": name,
                                "strerror": _trunc(strerror, 100)}
    if filename:
        evidence["filename"] = _trunc(filename, 120)
    hints = {
        "EACCES": "Check file permissions / ownership.",
        "ENOENT": "The path does not exist — verify it.",
        "EEXIST": "The path already exists — handle the collision.",
        "ENOSPC": "Disk is full — free space.",
        "EPIPE": "The reader closed the pipe — handle broken pipes.",
        "EADDRINUSE": "The port is already bound — pick another or kill the holder.",
    }
    root = f"OS error [{name}]: {strerror}" + (f" ({filename})" if filename else "")
    fix = hints.get(name, "Check the OS-level cause above; it is outside Python's control.")
    return root, evidence, fix


_INT_LITERAL_RE = re.compile(
    r"invalid literal for (int|float)\(\) with base \d+: (.*)")


def _analyze_value_error(exc: BaseException, frames, target):
    import json as _json
    if isinstance(exc, _json.JSONDecodeError):
        return _analyze_json_error(exc, frames, target)
    m = _INT_LITERAL_RE.search(str(exc))
    if not m:
        return _analyze_generic(exc, frames, target)
    kind, literal = m.group(1), m.group(2)
    evidence: dict[str, Any] = {"literal": _trunc(literal, 80), "target_type": kind}
    root = (f"Cannot convert {literal} to {kind} — the string is not a valid "
            f"{kind} literal (empty string? commas? units like 'px'?).")
    fix = (f"Clean the string before converting (strip whitespace/units), or "
           f"guard with try/except {kind.title()}Error / a regex check.")
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
    (FileNotFoundError, _analyze_file_not_found),   # before OSError
    (UnicodeDecodeError, _analyze_unicode_error),   # before ValueError
    (ZeroDivisionError, _analyze_zero_division),
    (RecursionError, _analyze_recursion),
    (IndexError, _analyze_index_error),
    (AttributeError, _analyze_attribute_error),
    (KeyError, _analyze_key_error),
    (ConnectionError, _analyze_connection_error),
    (TimeoutError, _analyze_connection_error),
    (OSError, _analyze_os_error),
    (AssertionError, _analyze_assertion),
    (TypeError, _analyze_type_error),
    (NameError, _analyze_name_error),
    (ValueError, _analyze_value_error),             # JSONDecodeError routes via here
]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def diagnose(exc: Any, context: dict[str, Any] | None = None,
             _depth: int = 0) -> dict[str, Any]:
    """Diagnose an exception. NEVER raises — on any internal failure returns
    a minimal generic diagnosis."""
    try:
        return _diagnose_inner(exc, context or {}, _depth)
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


def _diagnose_inner(exc: Any, context: dict[str, Any], _depth: int = 0) -> dict[str, Any]:
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
    # Chained exceptions: the visible error is often just the messenger.
    # Diagnose the cause too (depth-limited; the doctor never gets sick).
    if _depth < 2 and isinstance(exc, BaseException):
        cause = exc.__cause__
        if cause is None and not exc.__suppress_context__:
            cause = exc.__context__
        if cause is not None and cause is not exc:
            diag["cause"] = _safe(
                lambda: diagnose(cause, context, _depth=_depth + 1), default=None)
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


def format_diagnosis(diag: dict[str, Any], *, color: bool | None = None,
                     theme: Any = None, verbose: bool = False) -> str:
    """Render a diagnosis dict as a styled multi-section report.

    Sections: error headline (magenta — never red), location, root cause,
    evidence (key: value lines), suggested fix, and — with
    ``verbose=True`` — the traceback frames plus any chained cause.
    ``color=None`` auto-detects the terminal.
    """
    from .style import paint, supports_color, styled_box, header

    if color is None:
        color = supports_color()
    etype = diag.get("error_type", "?")
    loc = diag.get("location", "?")
    lines: list[str] = []
    lines.append(header(f"{etype} at {loc}", theme, color=color))
    lines.append("")
    lines.append(paint("ROOT CAUSE", "label", theme, color=color))
    lines.append(f"  {diag.get('root_cause', '?')}")
    lines.append("")
    evidence = diag.get("evidence") or {}
    if evidence:
        lines.append(paint("EVIDENCE", "label", theme, color=color))
        for key, val in evidence.items():
            val_s = str(val)
            if len(val_s) > 160:
                val_s = val_s[:157] + "…"
            lines.append(f"  {paint(str(key), 'info', theme, color=color)}: {val_s}")
        lines.append("")
    fix = diag.get("suggested_fix", "")
    if fix:
        lines.append(paint("SUGGESTED FIX", "label", theme, color=color))
        lines.append(f"  {paint(fix, 'ok', theme, color=color)}")
        lines.append("")
    if verbose:
        frames = diag.get("frames") or []
        if frames:
            lines.append(paint("TRACEBACK (innermost last)", "label", theme,
                               color=color))
            for f in frames:
                fn = f.get("short") or f.get("filename", "?")
                lines.append(f"  {paint(fn, 'info', theme, color=color)}"
                             f":{f.get('lineno', '?')} in "
                             f"{paint(str(f.get('func', '?')), 'muted', theme, color=color)}")
                src = f.get("source") or []
                for s in src[:3]:
                    lines.append(f"    {s.strip()}")
            lines.append("")
    cause = diag.get("cause")
    if isinstance(cause, dict):
        lines.append(paint("CAUSED BY", "label", theme, color=color))
        lines.append(format_diagnosis(cause, color=color, theme=theme,
                                     verbose=verbose))
    return "\n".join(lines).rstrip()
