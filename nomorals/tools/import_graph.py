"""Import-graph test selection (pass #62).

The pytest runner's changed-file selection used to be a filename-stem
heuristic: a test was picked because its *name* resembled the changed
file's name.  That misses renamed modules, indirect coverage (a test
exercising the changed code through an intermediary module) and any
coverage whose test file is named after the behavior, not the module —
while burning verify budget on same-stem tests that import nothing
relevant.

This module builds the repo's import graph by parsing imports with
``ast`` — target code is never imported or executed — and answers the
selection question precisely: which test files import a changed module,
directly or transitively through other repo modules?  The runner UNIONS
this with the stem selection (graph refines and extends; it never drops
a test the heuristic would have run).

Graph construction is cached per repo root, keyed by a signature of
every Python file's (mtime_ns, size): a file edit, rename, add or
delete invalidates the cache; repeat selections inside one process are
a directory walk + stat per file, with no re-parse.  A file that fails
to parse (mid-edit syntax errors are normal) simply contributes no
edges and is counted in ``ImportGraph.unparsed`` — it never fails the
build.  Whole-build failures raise, and the runner degrades to stem
selection with the degradation noted in its result.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

__all__ = ["ImportGraph", "build_import_graph", "graph_test_selection",
           "module_name_for", "clear_cache"]

#: Directories never descended into while building the graph: VCS and
#: tool state, virtualenvs, vendored third-party trees (their modules
#: are not the repo's own coverage surface), and build output.
_PRUNE_DIRS = frozenset({
    "__pycache__", ".venv", "venv", "node_modules", "vendor",
    ".tox", ".mypy_cache", ".pytest_cache", ".eggs", "build", "dist",
})

#: Files larger than this are not parsed (generated blobs, vendored
#: data modules) — they contribute no edges.
_MAX_PARSE_BYTES = 2_000_000

_CACHE_LIMIT = 8
_CACHE: dict[str, tuple[tuple, "ImportGraph"]] = {}


def clear_cache() -> None:
    """Drop every cached graph (tests, long-lived processes)."""
    _CACHE.clear()


def _iter_python_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _PRUNE_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.endswith(".py"):
                out.append(Path(dirpath) / name)
    return out


def module_name_for(root: Path, path: Path) -> str | None:
    """Dotted module name for ``path`` under ``root`` (None if not a
    Python file or it maps to the empty name).

    Fast path is a plain ``relative_to`` — callers in this module pass
    an already-resolved root and root-prefixed paths, so no filesystem
    resolution happens per file (a ``resolve()`` per file made cold
    graph builds on a ~2,500-file repo take ~20s; the fast path is
    ~10x cheaper).  The resolving fallback serves outside callers
    handing in symlinked or ``..``-laden paths.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        try:
            rel = path.resolve().relative_to(
                Path(root).expanduser().resolve())
        except (ValueError, OSError):
            return None
    parts = list(rel.parts)
    if not parts or not parts[-1].endswith(".py"):
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    name = ".".join(parts)
    return name or None


def _package_of(module: str | None, is_init: bool) -> str:
    """Dotted package containing a module ("" for a top-level one)."""
    if not module:
        return ""
    if is_init:
        return module  # the file IS the package
    return module.rpartition(".")[0]


def _resolve(out: set[str], name: str, known: set[str]) -> None:
    """Add ``name``'s longest prefix that is a known repo module.

    ``import a.b.c`` where only ``a.b`` exists as a file still creates a
    real dependency on ``a.b``; a name with no known prefix (stdlib,
    third-party) creates none.
    """
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in known:
            out.add(candidate)
            return


def _import_nodes(tree: ast.AST):
    """Yield Import/ImportFrom nodes, descending only through
    statement structure.

    Imports are statements: they can hide in function/class bodies,
    conditionals, try blocks and match cases (this codebase lazy-imports
    inside functions heavily), but never inside an expression — so the
    traversal prunes every expression subtree, which is most of the
    node count in a large file.
    """
    stack = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child
            elif isinstance(child, (ast.stmt, ast.ExceptHandler,
                                    ast.match_case)):
                stack.append(child)


def _file_imports(path: Path, known: set[str], package: str) -> set[str]:
    """Repo modules one file imports, resolved against ``known``."""
    data = path.read_bytes()
    if b"import" not in data:
        return set()  # no import statement can exist without the word
    tree = ast.parse(data)
    out: set[str] = set()
    for node in _import_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _resolve(out, alias.name, known)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".") if package else []
                keep = len(base_parts) - (node.level - 1)
                if keep < 0:
                    continue
                base = ".".join(base_parts[:keep])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if module:
                _resolve(out, module, known)
                for alias in node.names:
                    if alias.name != "*":
                        # `from a.b import c` where a.b.c is a module:
                        # a real edge to the submodule as well.
                        candidate = f"{module}.{alias.name}"
                        if candidate in known:
                            out.add(candidate)
    return out


class ImportGraph:
    """A repo's resolved import graph.

    ``edges``: module -> repo modules it imports.
    ``file_edges``: repo-relative file path -> repo modules it imports
    (authoritative for test files, whose module names can collide
    across directories).
    ``unparsed``: files whose parse was attempted and failed, plus
    oversized files — their import edges are lost, and this list is
    where that loss is visible.  (A file containing no ``import``
    substring is skipped without parsing: it can hold no import
    statement, so its edge set is exactly empty, broken or not.)
    """

    def __init__(self, root: Path, edges: dict[str, set[str]],
                 file_edges: dict[str, set[str]],
                 unparsed: list[str]) -> None:
        self.root = root
        self.edges = edges
        self.file_edges = file_edges
        self.unparsed = unparsed
        # Reverse adjacency, built once: module -> modules importing it.
        self._importers: dict[str, set[str]] = {}
        for src, targets in edges.items():
            for target in targets:
                self._importers.setdefault(target, set()).add(src)

    def importers_closure(self, changed: set[str]) -> set[str]:
        """``changed`` plus every module that transitively imports one
        of them — the set of modules whose behavior a change can reach.
        """
        affected = set(changed)
        stack = list(changed)
        while stack:
            module = stack.pop()
            for importer in self._importers.get(module, ()):
                if importer not in affected:
                    affected.add(importer)
                    stack.append(importer)
        return affected

    def covering_tests(self, test_rels: list[str],
                       changed_modules: set[str]) -> dict[str, str]:
        """{test relpath: "graph-direct" | "graph-transitive"} for the
        test files whose imports cover a changed module."""
        if not changed_modules:
            return {}
        affected = self.importers_closure(changed_modules)
        out: dict[str, str] = {}
        for rel in test_rels:
            direct = self.file_edges.get(rel, set())
            if direct & changed_modules:
                out[rel] = "graph-direct"
            elif direct & affected:
                out[rel] = "graph-transitive"
        # conftest.py is imported by pytest for every test in its
        # directory tree without appearing in the test file's imports:
        # conftest coverage is real coverage (indirect by construction).
        for rel in test_rels:
            if rel in out:
                continue
            directory = str(Path(rel).parent)
            while directory and directory != ".":
                conf = str(Path(directory) / "conftest.py")
                direct = self.file_edges.get(conf)
                if direct and (direct & changed_modules
                               or direct & affected):
                    out[rel] = "graph-transitive"
                    break
                directory = str(Path(directory).parent)
        return out


def build_import_graph(root: str | Path) -> ImportGraph:
    """Build (or return the cached) import graph for ``root``.

    Raises only for whole-build failures (unreadable root); the caller
    decides how to degrade.  Cached against a signature of every Python
    file's (relpath, mtime_ns, size).
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"import graph root is not a directory: {root}")
    files = _iter_python_files(root)
    signature_items: list[tuple[str, int, int]] = []
    for path in files:
        try:
            st = path.stat()
        except OSError:
            continue
        signature_items.append(
            (str(path.relative_to(root)), st.st_mtime_ns, st.st_size))
    signature = tuple(sorted(signature_items))
    key = str(root)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    names: dict[str, str | None] = {}
    known: set[str] = set()
    for path in files:
        name = module_name_for(root, path)
        names[str(path.relative_to(root))] = name
        if name:
            known.add(name)
    edges: dict[str, set[str]] = {}
    file_edges: dict[str, set[str]] = {}
    unparsed: list[str] = []
    for path in files:
        rel = str(path.relative_to(root))
        name = names[rel]
        try:
            if path.stat().st_size > _MAX_PARSE_BYTES:
                unparsed.append(rel)
                continue
            imports = _file_imports(
                path, known,
                _package_of(name, path.name == "__init__.py"))
        except Exception:  # noqa: BLE001 — one bad file must not fail the graph
            unparsed.append(rel)
            continue
        file_edges[rel] = imports
        if name:
            edges.setdefault(name, set()).update(imports)
    graph = ImportGraph(root, edges, file_edges, unparsed)
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (signature, graph)
    return graph


def graph_test_selection(root: str | Path,
                         changed_files: list[str]) -> dict[str, str]:
    """{test relpath: reason} for tests covering the changed modules.

    Test files are looked up in the same directories the stem selector
    uses (``tests/`` / ``test/``).  Raises on graph build failure — the
    caller degrades to stem selection and notes it.
    """
    root = Path(root).expanduser().resolve()
    changed_modules: set[str] = set()
    for rel in changed_files:
        if str(rel).endswith(".py"):
            name = module_name_for(root, root / rel)
            if name:
                changed_modules.add(name)
    if not changed_modules:
        return {}
    graph = build_import_graph(root)
    test_rels: list[str] = []
    for dirname in ("tests", "test"):
        tdir = root / dirname
        if tdir.is_dir():
            for tf in sorted(tdir.rglob("test_*.py")):
                if tf.name != "__init__.py":
                    test_rels.append(str(tf.relative_to(root)))
    return graph.covering_tests(test_rels, changed_modules)
