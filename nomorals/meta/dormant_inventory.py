"""Dormant-engine inventory — mechanical, no hardcoded list.

An engine is *dormant* when nothing in production code references it:
no static import and no dotted-path string literal (the ``importlib`` /
string-loading route) from any other module under ``nomorals/``. A
module imported only by tests is dormant in production too — tests
prove it runs, not that anything ships it — so test references are
reported separately (``test_importers`` / ``test_string_refs``) and do
NOT rescue a module from dormant status.

What this scan is and is not
----------------------------
* It is a *reference* scan over source text (AST imports + string
  literals). It never imports the modules it scans, so it cannot be
  fooled by import side effects and cannot execute untrusted code.
* Dynamic loads through a *literal name list in the same function*
  count as references: ``import_module(f".{name}", __package__)``
  iterating over ``("archive", "attacker", ...)`` (the tool-registry
  pattern) resolves each literal to ``<package>.<name>``. A computed
  name with no literal list stays a caveat (see ``dynamic_loaders``).
* Directory-walk loads count too: ``pkgutil.iter_modules(pkg.__path__)``
  where ``pkg`` is import-bound in the same file references every
  direct child module of that package (the custom-tools loader
  pattern). Anything else computed stays a caveat.
* It is not a dead-code verdict. A dormant module may be a library
  waiting for its caller, a template, or reachable through a
  *computed* dynamic import (``importlib.import_module(name)`` where
  ``name`` is built at runtime, not a literal). Modules that perform
  such non-literal dynamic loads are listed in
  :attr:`ScanResult.dynamic_loaders` as a caveat: their targets are
  invisible to this scan, so a module flagged dormant could in
  principle be loaded that way. Check the caveat list before deleting
  anything — this scan is an inventory, not a deletion list.
* ``__init__.py`` files are wiring, not engines: they are importers
  and string sources, but never reported as dormant themselves.
* The scan is locale-agnostic: it walks every module uniformly with no
  nation / language / currency / timezone axis. A pack for any nation
  or language is scanned exactly like any other module — universality
  is a property of the scan, not a special case in it.
* ``nomorals/builders/templates/`` is excluded: scaffold code that
  ships as text, not as live modules. Its imports would otherwise
  fake "referenced" evidence for template-local names.
* Docstrings are NOT counted as string references — a module merely
  mentioned in prose is not wired.

Classification
--------------
* ``dormant`` — zero production importers AND zero production string
  references. ``reason`` says whether tests reference it at all.
* ``referenced`` — at least one production importer or string
  reference (the module has a caller; nothing more is claimed).
* ``reachable_from_entry`` (informational, on every entry) — whether
  the module is reachable from an entry surface (``cmdline/``,
  ``console/``, ``social/chat/``, ``__main__``, ``cli``,
  ``tools/registry``, or any module with an ``if __name__ ==
  "__main__"`` guard) by following import + string edges. A
  ``referenced`` module that is NOT reachable from any entry surface
  is wired to something that itself has no entry path — worth a look,
  but not dormant by this scan's definition.

Usage::

    from nomorals.meta.dormant_inventory import scan
    result = scan()  # scans the repo this file lives in
    for mod in result.dormant:
        print(mod.module, mod.reason)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DormantModule",
    "ScanResult",
    "scan",
    "dormant_modules",
    "module_name_for",
]

#: Directories under ``nomorals/`` treated as entry surfaces.
ENTRY_DIRS = ("cmdline", "console", "social/chat")
#: Entry-surface modules by dotted name (in addition to ENTRY_DIRS).
ENTRY_MODULES = ("nomorals.__main__", "nomorals.cli", "nomorals.tools.registry")
#: Subtree whose files are scaffolds, not live code (see module docstring).
EXCLUDED_SUBTREES = ("nomorals/builders/templates",)


# ── data ────────────────────────────────────────────────────────────────────


@dataclass
class DormantModule:
    """One dormant module with the evidence for its status."""

    module: str
    path: str
    public_api: list[str] = field(default_factory=list)
    test_importers: list[str] = field(default_factory=list)
    test_string_refs: list[str] = field(default_factory=list)
    reachable_from_entry: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "path": self.path,
            "public_api": list(self.public_api),
            "test_importers": list(self.test_importers),
            "test_string_refs": list(self.test_string_refs),
            "reachable_from_entry": self.reachable_from_entry,
            "reason": self.reason,
        }


@dataclass
class ScanResult:
    """Full result of :func:`scan`."""

    root: str
    scanned: int
    dormant: list[DormantModule] = field(default_factory=list)
    referenced_count: int = 0
    unparsable: list[str] = field(default_factory=list)
    dynamic_loaders: list[str] = field(default_factory=list)
    entry_roots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "scanned": self.scanned,
            "dormant": [d.to_dict() for d in self.dormant],
            "referenced_count": self.referenced_count,
            "unparsable": list(self.unparsable),
            "dynamic_loaders": list(self.dynamic_loaders),
            "entry_roots": list(self.entry_roots),
        }


# ── helpers ─────────────────────────────────────────────────────────────────


def module_name_for(path: Path, package_root: Path) -> str:
    """Dotted module name for ``path`` relative to ``package_root``.

    ``package_root`` is the directory containing the top-level
    ``nomorals/`` package (usually the repo root).
    """
    rel = path.relative_to(package_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_excluded(path: Path, package_root: Path) -> bool:
    try:
        rel = path.relative_to(package_root).as_posix()
    except ValueError:
        return True
    return any(rel.startswith(prefix + "/") or rel == prefix
               for prefix in EXCLUDED_SUBTREES)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """AST node ids holding docstrings (excluded from string refs)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
    return ids


def _resolve_relative(level: int, module: str | None,
                      importer: str, is_package: bool) -> str:
    """Resolve a relative import (``level`` dots) to a dotted name."""
    base_parts = importer.split(".")
    if not is_package:
        base_parts = base_parts[:-1]  # containing package
    # level 1 == current package, so drop (level - 1) more parts.
    if level > 1:
        base_parts = base_parts[: -(level - 1)] if len(base_parts) >= level - 1 else []
    if module:
        base_parts = base_parts + module.split(".")
    return ".".join(p for p in base_parts if p)


@dataclass
class _FileFacts:
    module: str
    path: Path
    imports: set[str] = field(default_factory=set)
    string_refs: set[str] = field(default_factory=set)
    public_api: list[str] = field(default_factory=list)
    has_main_guard: bool = False
    has_dynamic_load: bool = False


def _analyse(path: Path, module: str, known: set[str]) -> _FileFacts:
    """Parse one file: imports, dotted-path string literals, public API."""
    facts = _FileFacts(module=module, path=path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                         filename=str(path))
    except (SyntaxError, ValueError):
        return facts  # caller records it as unparsable (empty facts)
    is_package = path.name == "__init__.py"
    docstrings = _docstring_nodes(tree)

    # Public API: __all__ if present, else top-level public defs/classes.
    all_names: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        all_names = [str(v) for v in ast.literal_eval(node.value)]
                    except (ValueError, TypeError, SyntaxError, MemoryError,
                            RecursionError):
                        all_names = None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                facts.public_api.append(node.name)
    if all_names is not None:
        facts.public_api = all_names

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                facts.imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base = _resolve_relative(node.level, node.module, module, is_package)
            else:
                base = node.module or ""
            if base:
                facts.imports.add(base)
            # `from X import submodule` — X.submodule may itself be a module.
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{base}.{alias.name}" if base else alias.name
                if candidate in known:
                    facts.imports.add(candidate)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            value = node.value.strip()
            if value in known and value != module:
                facts.string_refs.add(value)
        elif isinstance(node, ast.Compare) and any(
                isinstance(op, ast.Eq) for op in node.ops):
            # `if __name__ == "__main__":` entry guard.
            left = node.left
            if (isinstance(left, ast.Name) and left.id == "__name__"
                    and any(isinstance(c, ast.Constant)
                            and c.value == "__main__" for c in node.comparators)):
                facts.has_main_guard = True
        elif isinstance(node, ast.Call):
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name in ("import_module", "__import__"):
                # Non-literal target => computed dynamic load (caveat).
                has_literal = any(
                    isinstance(a, ast.Constant) and isinstance(a.value, str)
                    and a.value.startswith("nomorals")
                    for a in node.args)
                if not has_literal:
                    facts.has_dynamic_load = True

    # Scope-aware short-name resolution for dynamic imports: a function
    # that calls import_module(f".{name}", __package__) over a literal
    # tuple/list of names references <package>.<name> for each literal.
    # Without this, every tool module loaded by the registry would read
    # as dormant — the exact false positive this scan must not produce.
    scopes: list[ast.AST] = [tree]
    scopes.extend(n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    for scope in scopes:
        dynamic_calls = [
            n for n in ast.walk(scope)
            if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name)
                  and n.func.id in ("import_module", "__import__"))
                 or (isinstance(n.func, ast.Attribute)
                     and n.func.attr in ("import_module", "__import__")))
        ]
        if not dynamic_calls:
            continue
        pkg_hint: str | None = None
        for call in dynamic_calls:
            if len(call.args) >= 2:
                second = call.args[1]
                if isinstance(second, ast.Name) and second.id == "__package__":
                    pkg_hint = (module if is_package
                                else module.rsplit(".", 1)[0])
                elif isinstance(second, ast.Constant) \
                        and isinstance(second.value, str):
                    pkg_hint = second.value
        if not pkg_hint:
            continue
        for node in ast.walk(scope):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings):
                candidate = f"{pkg_hint}.{node.value}"
                if candidate in known and candidate != module:
                    facts.string_refs.add(candidate)

    # Directory-walk loads: pkgutil.iter_modules(pkg.__path__) where pkg
    # is import-bound in this file references every direct child of pkg.
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base = _resolve_relative(node.level, node.module, module,
                                         is_package)
            else:
                base = node.module or ""
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = (
                        f"{base}.{alias.name}" if base else alias.name)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "iter_modules"
                and node.args
                and isinstance(node.args[0], ast.Attribute)
                and node.args[0].attr == "__path__"
                and isinstance(node.args[0].value, ast.Name)):
            continue
        pkg = bindings.get(node.args[0].value.id)
        if not pkg:
            continue
        prefix = pkg + "."
        for candidate in known:
            if candidate.startswith(prefix):
                rest = candidate[len(prefix):]
                if rest and "." not in rest and candidate != module:
                    facts.string_refs.add(candidate)
    return facts


# ── scan ────────────────────────────────────────────────────────────────────


def scan(root: str | Path | None = None) -> ScanResult:
    """Mechanically scan ``root`` (default: repo containing this file).

    ``root`` is the directory containing the top-level ``nomorals/``
    package. Returns a :class:`ScanResult`; never raises on a single
    bad file (unparsable files are listed, not fatal).
    """
    if root is None:
        root = Path(__file__).resolve().parent.parent.parent
    root = Path(root)
    pkg = root / "nomorals"
    if not pkg.is_dir():
        # Caller may have passed the package dir itself.
        pkg = root if root.name == "nomorals" else pkg
        root = pkg.parent

    prod_files = sorted(
        p for p in pkg.rglob("*.py")
        if "__pycache__" not in p.parts and not _is_excluded(p, root)
    )
    test_dir = root / "tests"
    test_files = sorted(
        p for p in test_dir.rglob("*.py")
        if test_dir.is_dir() and "__pycache__" not in p.parts
    )

    known = {module_name_for(p, root) for p in prod_files}
    known |= {module_name_for(p, root) for p in test_files}

    prod_facts: dict[str, _FileFacts] = {}
    unparsable: list[str] = []
    for path in prod_files:
        module = module_name_for(path, root)
        if module.endswith(".__init__") or path.name == "__init__.py":
            # __init__ files are wiring: parsed as importers, never reported.
            pass
        facts = _analyse(path, module, known)
        if not facts.imports and not facts.string_refs and not facts.public_api:
            # Empty or unparsable — distinguish by re-parsing cheaply:
            # _analyse returns empty facts for both; check syntax.
            try:
                ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, ValueError):
                unparsable.append(module)
        prod_facts[module] = facts

    test_facts: dict[str, _FileFacts] = {}
    for path in test_files:
        module = module_name_for(path, root)
        test_facts[module] = _analyse(path, module, known)

    # Reverse index: module -> prod importers / string refs.
    prod_importers: dict[str, set[str]] = {m: set() for m in known}
    prod_string_refs: dict[str, set[str]] = {m: set() for m in known}
    for facts in prod_facts.values():
        for imported in facts.imports:
            if imported in prod_importers and imported != facts.module:
                prod_importers[imported].add(facts.module)
        for ref in facts.string_refs:
            if ref in prod_string_refs and ref != facts.module:
                prod_string_refs[ref].add(facts.module)

    test_importers: dict[str, set[str]] = {m: set() for m in known}
    test_string_refs: dict[str, set[str]] = {m: set() for m in known}
    for facts in test_facts.values():
        for imported in facts.imports:
            if imported in test_importers and imported != facts.module:
                test_importers[imported].add(facts.module)
        for ref in facts.string_refs:
            if ref in test_string_refs and ref != facts.module:
                test_string_refs[ref].add(facts.module)

    # Entry roots: entry dirs + named entry modules + __main__ guards.
    entry_roots: set[str] = set()
    for module in prod_facts:
        rel = module[len("nomorals."):] if module.startswith("nomorals.") else module
        first = rel.split(".")[0] if rel else ""
        second = ".".join(rel.split(".")[:2]) if rel else ""
        if (first in ("cmdline", "console") or second == "social/chat"
                or module in ENTRY_MODULES
                or prod_facts[module].has_main_guard):
            entry_roots.add(module)

    # Reachability from entry roots over import + string edges.
    edges: dict[str, set[str]] = {}
    for module, facts in prod_facts.items():
        targets = {t for t in facts.imports | facts.string_refs if t in prod_facts}
        edges[module] = targets
    reachable: set[str] = set()
    stack = [r for r in entry_roots if r in edges]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(edges.get(current, ()))

    dynamic_loaders = sorted(
        m for m, facts in prod_facts.items() if facts.has_dynamic_load
    )

    dormant: list[DormantModule] = []
    referenced = 0
    for module, facts in sorted(prod_facts.items()):
        if facts.path.name == "__init__.py":
            continue  # wiring, never an engine candidate
        importers = prod_importers.get(module, set())
        srefs = prod_string_refs.get(module, set())
        t_importers = test_importers.get(module, set())
        t_srefs = test_string_refs.get(module, set())
        if importers or srefs:
            referenced += 1
            continue
        if t_importers or t_srefs:
            reason = ("no production importer or string reference; "
                      "referenced only by tests")
        else:
            reason = "no production or test importer or string reference"
        dormant.append(DormantModule(
            module=module,
            path=str(facts.path),
            public_api=list(facts.public_api),
            test_importers=sorted(t_importers),
            test_string_refs=sorted(t_srefs),
            reachable_from_entry=module in reachable,
            reason=reason,
        ))
    dormant.sort(key=lambda d: d.module)
    return ScanResult(
        root=str(root),
        scanned=len([m for m in prod_facts
                     if prod_facts[m].path.name != "__init__.py"]),
        dormant=dormant,
        referenced_count=referenced,
        unparsable=sorted(unparsable),
        dynamic_loaders=dynamic_loaders,
        entry_roots=sorted(entry_roots),
    )


def dormant_modules(root: str | Path | None = None) -> list[DormantModule]:
    """Shortcut for ``scan(root).dormant``."""
    return scan(root).dormant
