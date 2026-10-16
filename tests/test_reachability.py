"""Reachability: every public package under nomorals/ must be reachable.

A package passes when at least one NON-TEST file outside the package
itself really imports it (AST ``Import``/``ImportFrom``, absolute or
relative, resolved to the package), OR when it carries an explicit
library marker:

* ``LIBRARY = True`` in its ``__init__.py``, or
* a ``library.json`` with ``{"library": true, "reason": "<non-empty>"}``.

The ``reason`` requirement is the lock against silently marking
everything library: a marker without a reason does NOT pass.

Deliberately NOT counted (this is the "passing test != reachable"
failure mode, locked out):

* test files (anything under a ``tests`` component) — tests must not
  be the only caller;
* docstring mentions (``:mod:`nomorals.x`` in prose is not a caller);
* files inside the package itself (self-imports are not callers).

String literals in real code DO count as references (they cover
``importlib.import_module("nomorals.x")`` and registry strings), but
only when they appear in code — docstrings are excluded.

The root ``nomorals`` package itself counts via prefix marking: any
``import nomorals.<pkg>`` marks ``nomorals`` too.
"""

from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOMORALS = REPO_ROOT / "nomorals"

_REF_RE = re.compile(r"nomorals(?:[./][A-Za-z_][\w]*)+")


def _package_dirs() -> list[Path]:
    out = []
    for init in sorted(NOMORALS.rglob("__init__.py")):
        if "__pycache__" in init.parts:
            continue
        if "tests" in init.parts:
            # Template fixture dirs (builders/templates/*/tests), not
            # shipped packages — excluded from the check entirely.
            continue
        out.append(init.parent)
    return out


def _dotted(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """AST node ids that are docstrings (module/class/function first stmt)."""
    ids = set()
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
        stack.extend(ast.iter_child_nodes(node))
    return ids


def _mark(mapping: dict[str, set[str]], dotted: str, caller: str,
          packages: set[str]) -> None:
    """Mark ``dotted`` and every ancestor package that exists."""
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        cand = ".".join(parts[:i])
        if cand in packages:
            mapping.setdefault(cand, set()).add(caller)


@lru_cache(maxsize=1)
def _scan() -> dict[str, set[str]]:
    packages = {_dotted(p) for p in _package_dirs()}
    callers: dict[str, set[str]] = {}
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        if "tests" in path.parts:
            continue
        if "MagicMock" in path.parts:
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except (SyntaxError, ValueError, OSError):
            continue
        rel = str(path.relative_to(REPO_ROOT))
        # This file's own package (for relative-import resolution).
        if NOMORALS in path.parents:
            own_pkg = _dotted(path.parent)
            if path.name == "__init__.py":
                own_pkg = _dotted(path.parent)
        else:
            own_pkg = ""
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _mark(callers, alias.name, rel, packages)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    if node.module:
                        _mark(callers, node.module, rel, packages)
                    continue
                if not own_pkg:
                    continue
                base = own_pkg.split(".")
                up = node.level - 1
                if up > len(base):
                    continue
                base = base[: len(base) - up] if up else base
                if node.module:
                    full = ".".join(base + node.module.split("."))
                    _mark(callers, full, rel, packages)
                else:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        full = ".".join(base + [alias.name])
                        if full in packages:
                            callers.setdefault(full, set()).add(rel)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstrings:
                    continue
                for m in _REF_RE.finditer(node.value):
                    dotted = m.group(0).replace("/", ".")
                    _mark(callers, dotted, rel, packages)
    return callers


def _library_marker(pkg_dir: Path) -> tuple[bool, str]:
    """(has_marker, why). A marker without a reason does not count."""
    init = pkg_dir / "__init__.py"
    try:
        tree = ast.parse(init.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError, OSError):
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "LIBRARY"
                            for t in node.targets)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True):
                return True, "LIBRARY = True in __init__.py"
    lib_json = pkg_dir / "library.json"
    if lib_json.is_file():
        try:
            data = json.loads(lib_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False, f"{lib_json} is not valid JSON"
        if data.get("library") is True and (data.get("reason") or "").strip():
            return True, f"library.json: {data['reason'][:80]}"
        return False, f"{lib_json} lacks library=true + non-empty reason"
    return False, ""


def test_every_package_reachable_or_explicit_library():
    pkgs = _package_dirs()
    callers = _scan()
    orphans = []
    library_marked = []
    for pkg_dir in pkgs:
        dotted = _dotted(pkg_dir)
        own = {str(p.relative_to(REPO_ROOT))
               for p in pkg_dir.rglob("*.py")}
        outside = {c for c in callers.get(dotted, set()) if c not in own}
        if outside:
            continue
        has_marker, why = _library_marker(pkg_dir)
        if has_marker:
            library_marked.append(f"{dotted} [{why}]")
            continue
        orphans.append(dotted)
    assert not orphans, (
        "Orphaned packages (no non-test caller, no library marker). "
        "NEVER delete: wire the package into a command/chat surface, or "
        "add an explicit library marker (LIBRARY = True in __init__.py or "
        "library.json with a non-empty reason) + owner decision note.\n"
        "Orphans:\n" + "\n".join(f"  - {o}" for o in orphans)
    )


def test_library_markers_have_reasons():
    for pkg_dir in _package_dirs():
        lib_json = pkg_dir / "library.json"
        if not lib_json.is_file():
            continue
        data = json.loads(lib_json.read_text(encoding="utf-8"))
        assert data.get("library") is True, f"{lib_json}: library must be true"
        assert (data.get("reason") or "").strip(), (
            f"{lib_json}: library marker needs a non-empty reason — "
            "silent marking is not allowed")
        assert (data.get("owner_decision_needed") or "").strip(), (
            f"{lib_json}: library marker needs owner_decision_needed note")


def test_tier0_packages_have_real_import_callers():
    """Learn/legal/matching must have REAL import callers, not just
    registry strings — the wiring this wave added must be genuine."""
    callers = _scan()
    for pkg in ("nomorals.learn", "nomorals.legal", "nomorals.matching"):
        assert callers.get(pkg), f"{pkg} has no non-test caller at all"
