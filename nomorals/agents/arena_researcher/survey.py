"""Live survey of Devon's own system: organs, wiring, dead paths, weak spots.

This is the survey half of the arena researcher (doc 15 R-2-14 — the
live version of the one-off module-gap audit). It walks the checked-out
tree under ``nomorals/`` and returns findings, each carrying a
``file:line`` anchor that verifies against the surveyed root
(:func:`verify_anchor`), the module's importers, and the callers or
evidence that prove the finding is real — not vibes.

Finding kinds:

* ``dead_module`` — a production module with zero production importers
  under ``nomorals/`` AND zero dotted-name references anywhere under
  ``nomorals/``. Entry points (L7 / ``cmdline`` / ``builders`` / top-level
  layer modules from :data:`nomorals.top_level_layer.TOP_LEVEL_LAYER`)
  are never candidates: they are reached from outside the tree.
* ``stub`` — a function whose whole body is one ``raise
  NotImplementedError`` or only ``pass`` / ``...`` / a docstring.
  ``@abstractmethod`` contracts are excluded: they are the interface,
  not a stub.
* ``unwired_tool`` — an entry of the agent-tool registry
  (:data:`nomorals.tools.agents.AGENT_TOOL_MODULES`) whose module cannot
  be imported, or imports but exposes no ``register`` hook, so the tool
  never reaches the runtime.
* ``silent_swallow`` — a module with three or more bare
  ``except: pass`` / ``except Exception: pass`` sites (SOUL.md §1: a
  swallowed error is a silent failure, not a fix).

Known limitation, stated so the owner can judge it: a module loaded
only by a dynamic path (``importlib.util.spec_from_file_location``,
a variable ``import_module`` name) is invisible to a static scan and
can be flagged dead. The finding's evidence states exactly what was
measured, and proposals are never auto-applied — the owner denies a
false positive, and the denial is recorded.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

#: Modules with three or more silent-swallow sites get a finding.
SILENT_SWALLOW_THRESHOLD = 3

#: Packages whose modules are reached from outside the tree (L7 entry
#: points, CLI, builders) and are never dead-module candidates.
ENTRY_POINT_PACKAGES = {"api", "builders", "cmdline"}

#: A dotted ``nomorals....`` reference anywhere in a file's text keeps
#: a module alive (covers string-form dynamic references).
_DOTTED_REF_RE = re.compile(r"nomorals(?:\.[A-Za-z0-9_]+)+")


def _is_stub_body(body: list[ast.stmt]) -> bool:
    """True when a function body does nothing at all."""
    meaningful = [s for s in body
                  if not (isinstance(s, ast.Expr)
                          and isinstance(s.value, ast.Constant)
                          and isinstance(s.value.value, str))]
    if not meaningful:
        return True
    if len(meaningful) != 1:
        return False
    only = meaningful[0]
    if isinstance(only, ast.Pass):
        return True
    if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant) \
            and only.value.value is Ellipsis:
        return True
    if isinstance(only, ast.Raise) and isinstance(only.exc, ast.Call) \
            and isinstance(only.exc.func, ast.Name) \
            and only.exc.func.id == "NotImplementedError":
        return True
    return (isinstance(only, ast.Raise)
            and isinstance(only.exc, ast.Name)
            and only.exc.id == "NotImplementedError")


def _is_abstract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "abstractmethod":
            return True
        if isinstance(dec, ast.Attribute) \
                and dec.attr == "abstractmethod":
            return True
    return False


def _is_silent_handler(handler: ast.ExceptHandler) -> bool:
    """True when an except handler swallows the error silently."""
    if not handler.body:
        return False
    for stmt in handler.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) \
                and isinstance(stmt.value, ast.Constant) \
                and stmt.value.value is Ellipsis:
            continue
        return False
    return True


def _dotted_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def _resolve_relative(dotted: str, level: int,
                      module: str | None) -> str | None:
    """Resolve a relative import to its absolute dotted name.

    *dotted* is the importing module's dotted name; its package is
    everything but the last segment. Level 1 is the current package,
    level 2 its parent, and so on.
    """
    package = dotted.split(".")[:-1]
    if level > len(package) + 1:
        return None
    base = package[:len(package) - level + 1]
    if module:
        base.append(module)
    return ".".join(base)


def _collect_imports(tree: ast.Module, dotted: str) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                base = _resolve_relative(dotted, node.level, node.module) \
                    or ""
            if base:
                imported.add(base)
            # `from X import Y` also imports X.Y (the name, not just
            # the package) — dropping it made every such module look
            # dead.
            for alias in node.names:
                if alias.name == "*":
                    continue
                if base:
                    imported.add(base + "." + alias.name)
    return imported


def _is_entry_module(tree: ast.Module) -> bool:
    """True when a module is an entry point (top-level main / __main__)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == "main":
            return True
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare) \
                    and isinstance(test.left, ast.Name) \
                    and test.left.id == "__name__" \
                    and any(isinstance(c, ast.Constant)
                            and c.value == "__main__"
                            for c in test.comparators):
                return True
    return False


def _entry_points(root: Path) -> set[str]:
    """Dotted names that are reached from outside the tree."""
    names: set[str] = set()
    try:
        from nomorals.top_level_layer import TOP_LEVEL_LAYER
    except Exception:
        TOP_LEVEL_LAYER = {}
    for name in TOP_LEVEL_LAYER:
        names.add("nomorals." + name)
    for pkg in ENTRY_POINT_PACKAGES:
        names.add("nomorals." + pkg)
    return names


def _tool_module_line(root: Path, name: str) -> int:
    registry = root / "nomorals" / "tools" / "agents.py"
    try:
        for lineno, line in enumerate(
                registry.read_text(encoding="utf-8").splitlines(), 1):
            if (f'"{name}"') in line or (f"'{name}'") in line:
                return lineno
    except OSError:
        pass
    return 1


def survey_repo(root: str | Path, *,
                tool_modules: list[str] | None = None) -> list[dict]:
    """Survey *root* and return ranked-ready findings.

    *root* is the repo root (the directory holding ``nomorals/``).
    Findings are dicts with ``id`` (stable), ``kind``, ``path``,
    ``line``, ``organ``, ``evidence`` and ``count``.
    """
    root = Path(root)
    pkg_root = root / "nomorals"
    files = sorted(p for p in pkg_root.rglob("*.py")
                   if "__pycache__" not in p.parts)
    entries = _entry_points(root)
    importers: dict[str, set[str]] = {}
    dotted_refs: set[str] = set()
    per_file: dict[Path, dict] = {}
    for path in files:
        dotted = _dotted_name(root, path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        dotted_refs.update(_DOTTED_REF_RE.findall(text))
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        imported = _collect_imports(tree, dotted)
        for name in imported:
            importers.setdefault(name, set()).add(dotted)
        stubs: list[dict] = []
        swallows = 0
        entry = _is_entry_module(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not _is_abstract(node) \
                    and _is_stub_body(node.body):
                stubs.append({"name": node.name, "line": node.lineno})
            elif isinstance(node, ast.ExceptHandler) and _is_silent_handler(node):
                swallows += 1
        per_file[path] = {"dotted": dotted, "stubs": stubs,
                          "swallows": swallows, "entry": entry}
    findings: list[dict] = []
    for path in files:
        info = per_file.get(path)
        if info is None:
            continue
        rel = str(path.relative_to(root))
        organ = path.relative_to(pkg_root).parts[0]
        dotted = info["dotted"]
        if path.name != "__init__.py" \
                and not info.get("entry") \
                and dotted not in entries \
                and not any(dotted == e or dotted.startswith(e + ".")
                            for e in entries) \
                and dotted not in importers \
                and dotted not in dotted_refs:
            findings.append({
                "id": "ar-dead-{}".format(dotted.replace(".", "-")),
                "kind": "dead_module",
                "path": rel,
                "line": 1,
                "organ": organ,
                "evidence": (
                    "zero production importers in nomorals/ and zero "
                    "dotted-name references in nomorals/"),
                "count": 1,
            })
        for stub in info["stubs"]:
            findings.append({
                "id": f"ar-stub-{dotted.replace('.', '-')}-{stub['line']}",
                "kind": "stub",
                "path": rel,
                "line": stub["line"],
                "organ": organ,
                "evidence": ("function {} has no implementation "
                             "(raises NotImplementedError or empty body)".format(stub["name"])),
                "count": 1,
            })
        if info["swallows"] >= SILENT_SWALLOW_THRESHOLD:
            findings.append({
                "id": "ar-swallow-{}".format(dotted.replace(".", "-")),
                "kind": "silent_swallow",
                "path": rel,
                "line": 1,
                "organ": organ,
                "evidence": (f"{info['swallows']} silent except-pass sites "
                             "in one module; errors vanish without a trace"),
                "count": info["swallows"],
            })
    if tool_modules is None:
        try:
            from nomorals.tools.agents import AGENT_TOOL_MODULES
            tool_modules = list(AGENT_TOOL_MODULES)
        except Exception:
            tool_modules = []
    for name in tool_modules:
        module = None
        for candidate in (f"nomorals.agents.{name}",
                          f"nomorals.{name}"):
            try:
                module = importlib.import_module(candidate)
                break
            except Exception:
                continue
        line = _tool_module_line(root, name)
        if module is None:
            findings.append({
                "id": f"ar-unwired-{name}",
                "kind": "unwired_tool",
                "path": "nomorals/tools/agents.py",
                "line": line,
                "organ": "tools",
                "evidence": (f"tool module {name} cannot be imported, so the "
                             "registry entry never reaches the runtime"),
                "count": 1,
            })
        elif not hasattr(module, "register"):
            findings.append({
                "id": f"ar-unwired-{name}",
                "kind": "unwired_tool",
                "path": "nomorals/tools/agents.py",
                "line": line,
                "organ": "tools",
                "evidence": (f"tool module {name} imports but exposes no "
                             "register hook, so its tool never reaches "
                             "the runtime"),
                "count": 1,
            })
    return findings


def verify_anchor(root: str | Path, path: str, line: int) -> bool:
    """True when *path*:*line* exists in the surveyed root."""
    target = Path(root) / path
    if not target.is_file():
        return False
    try:
        total = sum(1 for _ in target.open(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return False
    return 1 <= line <= total
