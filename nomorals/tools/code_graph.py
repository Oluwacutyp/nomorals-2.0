"""Queryable code graph over a codebase (Graphify shape, Devon-built).

Graphify turns a codebase into a knowledge graph the agent queries
instead of grepping — tree-sitter parsing, deterministic, zero LLM
calls for the code graph. Devon HAS a memory knowledge graph
(:mod:`nomorals.agents.kg` — entities/relations from decoded content)
but NO codebase graph: her code search is grep + the seam-search
inverted index + the in-memory structural :mod:`nomorals.tools.repo_index`
(which is per-process, name-based callers, no durable store).

This rung is the DURABLE graph: nodes (files, modules, classes,
functions, methods) and RESOLVED call / import / inherit edges,
stored in SQLite under the Devon home, rebuilt incrementally
(only re-parse changed files), queried by the brain, the tools
registry (``graph_build`` / ``graph_query`` / ``graph_status``),
and ``nm graph``.

Parser: tree-sitter when provisioned (``pip install tree-sitter`` +
``tree-sitter-python``, bounded, never under pytest), else the
deterministic stdlib :mod:`ast` fallback — the graph builds either
way and records which parser built it. No LLM call anywhere in
construction or query. The two parsers produce the same intermediate
shape (:class:`_FileParse`) and share one resolver, so the graph
shape does not depend on which parser ran.

What is recorded, honestly
---------------------------
* Calls in function bodies and module top-level statements only.
  Decorators, default values and class-level statements are NOT
  recorded as calls (they run at definition/import time, not at
  call time, and would pollute "who calls X" with decoration sites).
* A call edge is recorded only when the callee resolves to a
  definition in the repo (same-module symbol, ``self``/``cls``
  method, or an imported in-repo name). Unresolvable attribute
  calls (``obj.method()`` on an unknown ``obj``) are dropped, not
  guessed. Calls to imported-but-not-in-repo names become
  ``external:<name>`` nodes so "what does X call" stays informative
  without inventing definitions.
* Inheritance edges resolve the same way; unresolvable bases
  (stdlib, third-party) are dropped.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.failure_store import home_dir as _devon_home_dir
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .repo_index import SKIP_DIRS as _SKIP_DIRS

__all__ = [
    "CodeGraph", "BuildStats", "default_store_path", "ensure_parser",
    "GRAPH_VERSION",
]

_log = get_logger(__name__)

#: Schema version recorded in graph_meta; bump when the tables change.
GRAPH_VERSION = 1

#: Files larger than this are not parsed (generated blobs, vendored
#: data modules) — they contribute no edges. Same bound as the
#: import graph (tools/import_graph.py).
_MAX_PARSE_BYTES = 2_000_000

#: Chunk size for the batched ``WHERE url IN (...)``-style lookups and
#: for executemany batches (SQLite's variable limit is 999 by default
#: on older builds; 500 stays safely under it).
_BATCH = 500

#: Tree-sitter packages provisioned when the rung first needs them.
#: Pinned at or above the GHSA-pcc4-rvhr-2pr8 fix (0.9.70) — that
#: advisory is a path traversal in the Fortran extractor, which this
#: rung never uses (Python only), but the floor is kept anyway.
_TS_PACKAGES = ("tree-sitter>=0.25", "tree-sitter-python>=0.25")

#: pip install bound per package (seconds). Never under pytest.
_PIP_TIMEOUT_S = 300

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS graph_meta ("
    "  key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS graph_files ("
    "  path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL,"
    "  size INTEGER NOT NULL, sha256 TEXT NOT NULL,"
    "  parsed_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS graph_nodes ("
    "  id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,"
    "  qualified TEXT NOT NULL, file TEXT NOT NULL,"
    "  line INTEGER NOT NULL, end_line INTEGER NOT NULL,"
    "  docstring TEXT NOT NULL DEFAULT '',"
    "  parent TEXT NOT NULL DEFAULT '')",
    "CREATE TABLE IF NOT EXISTS graph_edges ("
    "  src TEXT NOT NULL, dst TEXT NOT NULL, kind TEXT NOT NULL,"
    "  file TEXT NOT NULL, line INTEGER NOT NULL,"
    "  PRIMARY KEY (src, dst, kind))",
    "CREATE INDEX IF NOT EXISTS idx_nodes_qualified"
    "  ON graph_nodes(qualified)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_name ON graph_nodes(name)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_file ON graph_nodes(file)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON graph_edges(dst, kind)",
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON graph_edges(src, kind)",
)


# ── intermediate parse shape (shared by both parsers) ────────────────

@dataclass
class _Symbol:
    kind: str  # "class" | "function" | "method"
    name: str
    qualified: str
    line: int
    end_line: int
    docstring: str = ""
    parent: str = ""  # qualified name of the enclosing class, else ""


@dataclass
class _Call:
    caller: str  # qualified name of the enclosing function, else ""
    text: str  # dotted callee text: "foo", "self.bar", "mod.func"
    line: int


@dataclass
class _Import:
    module: str  # resolved dotted module, "" when unresolvable
    names: list[str]  # imported names; ["*"] for star, [] for plain import
    line: int


@dataclass
class _Base:
    class_qualified: str
    text: str  # base text as written: "Animal", "pkg.mod.Base"
    line: int


@dataclass
class _FileParse:
    rel: str  # repo-relative path with forward slashes
    module: str  # dotted module name
    symbols: list[_Symbol] = field(default_factory=list)
    calls: list[_Call] = field(default_factory=list)
    imports: list[_Import] = field(default_factory=list)
    bases: list[_Base] = field(default_factory=list)


@dataclass
class BuildStats:
    files_total: int = 0
    parsed: int = 0
    skipped: int = 0
    removed: int = 0
    nodes: int = 0
    edges: int = 0
    parser: str = "ast"
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_total": self.files_total,
            "parsed": self.parsed,
            "skipped": self.skipped,
            "removed": self.removed,
            "nodes": self.nodes,
            "edges": self.edges,
            "parser": self.parser,
            "seconds": round(self.seconds, 3),
        }


# ── store path ────────────────────────────────────────────────────────

def default_store_path(root: str | Path) -> Path:
    """Durable store for ``root``'s graph: one SQLite file per root
    under the Devon home (``~/.nomorals/code_graph/``), keyed by the
    root's sha1. Repos stay clean; the graph is per-host state like
    every other sidecar under the home dir."""
    digest = hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:16]
    return _devon_home_dir() / "code_graph" / f"{digest}.sqlite"


# ── parser provisioning ───────────────────────────────────────────────

def _under_pytest() -> bool:
    return "pytest" in sys.modules or bool(
        os.environ.get("PYTEST_CURRENT_TEST"))


def ensure_parser(*, provision: bool = True) -> tuple[str, str]:
    """Which parser builds the graph: ``("tree-sitter", detail)`` or
    ``("ast", reason)``.

    Tree-sitter is preferred when importable. When it is missing and
    ``provision`` is true (and this is not a pytest process), pip
    installs ``tree-sitter`` + ``tree-sitter-python`` once, bounded,
    then re-imports. Any failure — missing pip, no network, bad
    version — falls back to the stdlib ``ast`` parser with an honest
    reason; the graph builds either way. Never raises.
    """
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401
        return ("tree-sitter", "tree-sitter + tree-sitter-python")
    except Exception:  # noqa: BLE001 - any import failure = not available
        pass
    if not provision:
        return ("ast", "tree-sitter not installed; using stdlib ast")
    if _under_pytest():
        return ("ast", "tree-sitter not installed; not provisioning "
                       "under pytest — using stdlib ast")
    for package in _TS_PACKAGES:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                capture_output=True, text=True, timeout=_PIP_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - pip failure = fallback
            return ("ast", f"tree-sitter pip install failed ({exc}); "
                           "using stdlib ast")
        if proc.returncode != 0:
            return ("ast", f"tree-sitter pip install failed "
                           f"({package}: rc={proc.returncode}); "
                           "using stdlib ast")
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401
        return ("tree-sitter", "tree-sitter + tree-sitter-python "
                               "(provisioned)")
    except Exception as exc:  # noqa: BLE001 - provisioned but unimportable
        return ("ast", f"tree-sitter installed but unimportable ({exc}); "
                       "using stdlib ast")


# ── ast parser ────────────────────────────────────────────────────────

def _dotted_module(rel: str) -> str:
    """Dotted module name for a repo-relative ``.py`` path."""
    stem = rel[:-3] if rel.endswith(".py") else rel
    parts = [p for p in stem.split("/") if p not in ("", ".")]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str, level: int, name: str) -> str:
    """Resolve ``from .[.]*name import ...`` to a dotted module."""
    if level == 0:
        return name
    base = module.split(".")
    # level 1 = current package: drop the module's own last component.
    up = level - 1
    if up >= len(base):
        return name
    prefix = base[:len(base) - up] if up else base
    if name:
        return ".".join(prefix + [name])
    return ".".join(prefix)


def _ast_docstring(node: ast.AST) -> str:
    doc = ast.get_docstring(node, clean=True) or ""
    return doc.split("\n")[0].strip() if doc else ""


def _parse_ast(rel: str, source: bytes) -> _FileParse | None:
    """Parse one file with stdlib :mod:`ast`. None when unparseable
    (mid-edit syntax errors are normal — the file contributes no
    edges and is counted, never a build failure)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    module = _dotted_module(rel)
    out = _FileParse(rel=rel, module=module)
    # Stack of enclosing (kind, qualified) while walking.
    stack: list[tuple[str, str]] = []

    def _caller() -> str:
        for kind, qual in reversed(stack):
            if kind in ("function", "method"):
                return qual
        return ""

    def _enclosing_class() -> str:
        for kind, qual in reversed(stack):
            if kind == "class":
                return qual
        return ""

    def _visit(node: ast.AST, in_decorator: bool = False) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent = _enclosing_class()
            qual = (f"{parent}.{node.name}" if parent
                    else f"{module}.{node.name}")
            kind = "method" if parent else "function"
            out.symbols.append(_Symbol(
                kind=kind, name=node.name, qualified=qual,
                line=node.lineno, end_line=node.end_lineno or node.lineno,
                docstring=_ast_docstring(node), parent=parent))
            stack.append((kind, qual))
            # Body only: decorators and default values are not calls.
            for stmt in node.body:
                _visit(stmt)
            stack.pop()
            return
        if isinstance(node, ast.ClassDef):
            qual = f"{module}.{node.name}"
            out.symbols.append(_Symbol(
                kind="class", name=node.name, qualified=qual,
                line=node.lineno, end_line=node.end_lineno or node.lineno,
                docstring=_ast_docstring(node), parent=""))
            for base in node.bases:
                text = _ast_name_text(base)
                if text:
                    out.bases.append(_Base(class_qualified=qual,
                                           text=text, line=node.lineno))
            stack.append(("class", qual))
            for stmt in node.body:
                _visit(stmt)
            stack.pop()
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.imports.append(_Import(module=alias.name, names=[],
                                           line=node.lineno))
            return
        if isinstance(node, ast.ImportFrom):
            target = _resolve_relative(
                module, node.level or 0, node.module or "")
            names = [a.name for a in node.names]
            out.imports.append(_Import(module=target, names=names,
                                       line=node.lineno))
            return
        if isinstance(node, ast.Call) and not in_decorator:
            text = _ast_name_text(node.func)
            if text:
                out.calls.append(_Call(caller=_caller(), text=text,
                                       line=node.lineno))
            # Recurse into args (nested calls) but not the func itself
            # (already recorded).
            for child in ast.iter_child_nodes(node):
                if child is not node.func:
                    _visit(child)
            return
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return out


def _ast_name_text(node: ast.AST) -> str:
    """Dotted text for a Name/Attribute chain, else ""."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _ast_name_text(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


# ── tree-sitter parser ────────────────────────────────────────────────

def _ts_text(node: Any, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode(
        "utf-8", "replace")


def _ts_docstring(body: Any, source: bytes) -> str:
    """First statement of a block when it is a string literal."""
    if body is None or body.type != "block":
        return ""
    if not body.children:
        return ""
    first = body.children[0]
    if first.type != "expression_statement":
        return ""
    for child in first.children:
        if child.type == "string":
            raw = _ts_text(child, source).strip()
            for quote in ('"""', "'''", '"', "'"):
                if raw.startswith(quote) and raw.endswith(quote) \
                        and len(raw) >= 2 * len(quote):
                    raw = raw[len(quote):-len(quote)]
                    break
            return raw.split("\n")[0].strip()
    return ""


def _parse_tree_sitter(rel: str, source: bytes) -> _FileParse | None:
    """Parse one file with tree-sitter's Python grammar. None when
    the grammar cannot run (the caller then falls back to ast)."""
    try:
        import tree_sitter
        import tree_sitter_python
        language = tree_sitter.Language(tree_sitter_python.language())
        parser = tree_sitter.Parser(language)
        tree = parser.parse(source)
    except Exception:  # noqa: BLE001 - any grammar failure = fallback
        return None
    module = _dotted_module(rel)
    out = _FileParse(rel=rel, module=module)
    stack: list[tuple[str, str]] = []

    def _caller() -> str:
        for kind, qual in reversed(stack):
            if kind in ("function", "method"):
                return qual
        return ""

    def _enclosing_class() -> str:
        for kind, qual in reversed(stack):
            if kind == "class":
                return qual
        return ""

    def _field(node: Any, name: str) -> Any:
        try:
            return node.child_by_field_name(name)
        except Exception:  # noqa: BLE001 - old bindings may lack fields
            return None

    def _walk(node: Any, in_decorator: bool = False) -> None:
        ntype = node.type
        if ntype == "function_definition":
            name_node = _field(node, "name")
            if name_node is None:
                return
            name = _ts_text(name_node, source)
            parent = _enclosing_class()
            qual = (f"{parent}.{name}" if parent
                    else f"{module}.{name}")
            kind = "method" if parent else "function"
            body = _field(node, "body")
            out.symbols.append(_Symbol(
                kind=kind, name=name, qualified=qual,
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=_ts_docstring(body, source), parent=parent))
            stack.append((kind, qual))
            if body is not None:
                for child in body.children:
                    _walk(child)
            stack.pop()
            return
        if ntype == "class_definition":
            name_node = _field(node, "name")
            if name_node is None:
                return
            name = _ts_text(name_node, source)
            qual = f"{module}.{name}"
            body = _field(node, "body")
            out.symbols.append(_Symbol(
                kind="class", name=name, qualified=qual,
                line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=_ts_docstring(body, source), parent=""))
            sup = _field(node, "superclass")
            if sup is not None:
                for child in sup.children:
                    if child.type == "identifier":
                        out.bases.append(_Base(
                            class_qualified=qual,
                            text=_ts_text(child, source),
                            line=node.start_point[0] + 1))
                    elif child.type == "attribute":
                        out.bases.append(_Base(
                            class_qualified=qual,
                            text=_ts_text(child, source).replace(" ", ""),
                            line=node.start_point[0] + 1))
            stack.append(("class", qual))
            if body is not None:
                for child in body.children:
                    _walk(child)
            stack.pop()
            return
        if ntype in ("import_statement", "import_from_statement"):
            _ts_import(node, source, module, out)
            return
        if ntype == "decorator":
            return  # decorators are not calls
        if ntype == "call" and not in_decorator:
            func = _field(node, "function")
            if func is not None and func.type in ("identifier", "attribute"):
                text = _ts_text(func, source).replace(" ", "")
                if text:
                    out.calls.append(_Call(
                        caller=_caller(), text=text,
                        line=node.start_point[0] + 1))
            for child in node.children:
                if child is not func:
                    _walk(child)
            return
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return out


def _ts_import(node: Any, source: bytes, module: str,
               out: _FileParse) -> None:
    """Record one tree-sitter import statement."""
    text = _ts_text(node, source)
    if node.type == "import_statement":
        # `import a, b as c` — each dotted name is a module import.
        rest = text[len("import"):].strip()
        for part in rest.split(","):
            name = part.strip().split(" as ")[0].strip()
            if name:
                out.imports.append(_Import(module=name, names=[],
                                           line=node.start_point[0] + 1))
        return
    # import_from_statement: `from ...mod import a, b`
    match = re.match(r"from\s+(\.*)([\w\.]*)\s+import\s+(.+)",
                     text.replace("\n", " "), re.DOTALL)
    if not match:
        return
    dots, mod, names_part = match.groups()
    target = _resolve_relative(module, len(dots), mod)
    names = [p.strip().split(" as ")[0].strip()
             for p in names_part.strip().strip("()").split(",")]
    names = [n for n in names if n and n != "*"] or (
        ["*"] if "*" in names_part else [])
    out.imports.append(_Import(module=target, names=names,
                               line=node.start_point[0] + 1))


# ── resolution ────────────────────────────────────────────────────────

def _resolve_file(parse: _FileParse,
                  global_symbols: dict[str, dict[str, str]],
                  in_repo_modules: set[str]) -> tuple[list[dict[str, Any]],
                                                      list[dict[str, Any]]]:
    """Turn one file's parse into node + edge rows.

    ``global_symbols`` maps module → {top-level name → qualified id};
    ``in_repo_modules`` is every module the graph knows. Returns
    (nodes, edges) as plain dicts ready for executemany.
    """
    module = parse.module
    nodes: list[dict[str, Any]] = [{
        "id": f"module:{module}", "kind": "module", "name": module,
        "qualified": module, "file": parse.rel, "line": 1,
        "end_line": 1, "docstring": "", "parent": "",
    }]
    edges: list[dict[str, Any]] = []
    local: dict[str, str] = {}  # top-level name → qualified
    methods: dict[str, dict[str, str]] = {}  # class qual → method → qual
    for sym in parse.symbols:
        nodes.append({
            "id": f"def:{sym.qualified}", "kind": sym.kind,
            "name": sym.name, "qualified": sym.qualified,
            "file": parse.rel, "line": sym.line,
            "end_line": sym.end_line, "docstring": sym.docstring,
            "parent": sym.parent,
        })
        if sym.kind in ("function", "class"):
            local[sym.name] = sym.qualified
        if sym.kind == "method":
            methods.setdefault(sym.parent, {})[sym.name] = sym.qualified
        edges.append({
            "src": f"module:{module}" if not sym.parent
                   else f"def:{sym.parent}",
            "dst": f"def:{sym.qualified}",
            "kind": "contains" if sym.parent else "defines",
            "file": parse.rel, "line": sym.line,
        })
    # Imported names: local name → (target module, original name or None).
    imported: dict[str, tuple[str, str | None]] = {}
    for imp in parse.imports:
        if not imp.module:
            continue
        if imp.module in in_repo_modules:
            edges.append({
                "src": f"module:{module}", "dst": f"module:{imp.module}",
                "kind": "imports", "file": parse.rel, "line": imp.line,
            })
        if imp.names:
            for name in imp.names:
                if name != "*":
                    imported[name] = (imp.module, name)
        else:
            # `import pkg.mod` binds the TOP component locally.
            top = imp.module.split(".")[0]
            imported[top] = (imp.module, None)
    for base in parse.bases:
        target = _resolve_name(base.text, module, local, imported,
                               global_symbols, in_repo_modules)
        if target is not None:
            edges.append({
                "src": f"def:{base.class_qualified}", "dst": target,
                "kind": "inherits", "file": parse.rel, "line": base.line,
            })
    for call in parse.calls:
        target = _resolve_call(call, module, local, methods, imported,
                               global_symbols, in_repo_modules)
        if target is None:
            continue
        src = f"def:{call.caller}" if call.caller else f"module:{module}"
        edges.append({
            "src": src, "dst": target, "kind": "calls",
            "file": parse.rel, "line": call.line,
        })
    return nodes, edges


def _resolve_name(text: str, module: str, local: dict[str, str],
                  imported: dict[str, tuple[str, str | None]],
                  global_symbols: dict[str, dict[str, str]],
                  in_repo_modules: set[str]) -> str | None:
    """Resolve a bare name to a `def:` id, else None."""
    if "." in text:
        mod_part, _, name = text.rpartition(".")
        if mod_part in in_repo_modules:
            qual = global_symbols.get(mod_part, {}).get(name)
            if qual:
                return f"def:{qual}"
        top = mod_part.split(".")[0]
        if top in imported:
            imp_mod, orig = imported[top]
            if orig is None:
                # `import pkg.mod` binds `pkg`; `pkg.sub.Base` names
                # module `pkg.mod.sub` (bound top swapped for the full
                # imported module).
                full_mod = imp_mod + mod_part[len(top):]
                if full_mod in in_repo_modules:
                    qual = global_symbols.get(full_mod, {}).get(name)
                    if qual:
                        return f"def:{qual}"
        return None
    if text in local:
        return f"def:{local[text]}"
    if text in imported:
        mod, orig = imported[text]
        if mod in in_repo_modules and orig:
            qual = global_symbols.get(mod, {}).get(orig)
            if qual:
                return f"def:{qual}"
        return f"external:{text}"
    return None


def _resolve_call(call: _Call, module: str, local: dict[str, str],
                  methods: dict[str, dict[str, str]],
                  imported: dict[str, tuple[str, str | None]],
                  global_symbols: dict[str, dict[str, str]],
                  in_repo_modules: set[str]) -> str | None:
    """Resolve a call's dotted text to a node id (``def:…``,
    ``external:…``), or None when it cannot be resolved honestly."""
    text = call.text
    parts = text.split(".")
    if len(parts) == 1:
        name = parts[0]
        if name in local:
            return f"def:{local[name]}"
        if name in imported:
            mod, orig = imported[name]
            if mod in in_repo_modules and orig:
                qual = global_symbols.get(mod, {}).get(orig)
                if qual:
                    return f"def:{qual}"
            return f"external:{name}"
        return None
    if len(parts) == 2:
        base, attr = parts
        if base in ("self", "cls"):
            # Enclosing class is the caller's parent.
            caller = call.caller
            cls_qual = caller.rsplit(".", 1)[0] if "." in caller else ""
            if cls_qual in methods and attr in methods[cls_qual]:
                return f"def:{methods[cls_qual][attr]}"
            return None
        if base in imported:
            mod, orig = imported[base]
            if orig is None:
                # `import mod` binds `mod`; `import pkg.mod` binds the
                # top component `pkg`. `base.attr()` targets the bound
                # module `base` itself.
                if base in in_repo_modules:
                    qual = global_symbols.get(base, {}).get(attr)
                    if qual:
                        return f"def:{qual}"
                return f"external:{text}"
            # `from x import y` then `y.attr()` — y is a name, not a
            # module; unresolvable without type info.
            return None
        if base in local:
            # A same-module class called as `Cls.method()` — static-ish.
            qual = local[base]
            if qual in methods and attr in methods[qual]:
                return f"def:{methods[qual][attr]}"
            return None
        return None
    return None


# ── the graph ─────────────────────────────────────────────────────────

class CodeGraph:
    """Durable queryable code graph over one repo root.

    ``build()`` parses (incrementally) and stores the graph in
    SQLite; the query methods read the store. The store lives at
    :func:`default_store_path` unless ``db_path`` is given (tests pass
    a temp path). The graph never imports or executes repo code.
    """

    def __init__(self, root: str | Path,
                 db_path: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.db_path = Path(db_path) if db_path is not None \
            else default_store_path(self.root)
        self._db: Database | None = None
        self.parser: str = "ast"
        self.parser_detail: str = ""

    # ── store ──

    def _open(self) -> Database:
        if self._db is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = Database(self.db_path)
            for stmt in _SCHEMA:
                self._db.execute(stmt)
        return self._db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> CodeGraph:
        self._open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── build ──

    def _iter_python_files(self) -> list[Path]:
        out: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                if name.endswith(".py"):
                    out.append(Path(dirpath) / name)
        return out

    def build(self, *, force: bool = False,
              provision: bool = True) -> BuildStats:
        """(Re)build the graph. Incremental by default: only files
        whose (mtime_ns, size, sha256) changed since the last build
        are re-parsed. Returns the build stats."""
        started = time.monotonic()
        self.parser, self.parser_detail = ensure_parser(
            provision=provision)
        db = self._open()
        stats = BuildStats(parser=self.parser)
        files = self._iter_python_files()
        stats.files_total = len(files)
        known: dict[str, tuple[int, int, str]] = {}
        for row in db.query(
                "SELECT path, mtime_ns, size, sha256 FROM graph_files"):
            known[row["path"]] = (row["mtime_ns"], row["size"],
                                  row["sha256"])
        parse_fn = _parse_tree_sitter if self.parser == "tree-sitter" \
            else _parse_ast
        parses: dict[str, _FileParse] = {}
        changed: list[str] = []
        for path in files:
            rel = path.relative_to(self.root).as_posix()
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if len(data) > _MAX_PARSE_BYTES:
                stats.skipped += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            st = path.stat()
            old = None if force else known.get(rel)
            if old is not None and old == (st.st_mtime_ns, st.st_size,
                                           digest):
                stats.skipped += 1
                continue
            parsed = parse_fn(rel, data)
            if parsed is None:
                # Unparseable file: keep the fingerprint so the next
                # build skips it again until it changes, but record
                # no nodes for it.
                parses[rel] = _FileParse(rel=rel,
                                         module=_dotted_module(rel))
            else:
                parses[rel] = parsed
            changed.append(rel)
            with db.transaction():
                db.execute(
                    "INSERT OR REPLACE INTO graph_files"
                    " (path, mtime_ns, size, sha256, parsed_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (rel, st.st_mtime_ns, st.st_size, digest,
                     time.time()))
                # Drop the file's old rows NOW, before the global
                # symbol index is loaded below: a renamed symbol must
                # not survive in the index and attract edges that
                # dangle once the new rows land.
                db.execute("DELETE FROM graph_nodes WHERE file = ?",
                           (rel,))
                db.execute("DELETE FROM graph_edges WHERE file = ?",
                           (rel,))
            stats.parsed += 1
        # Files deleted since the last build drop out of the graph.
        on_disk = {p.relative_to(self.root).as_posix() for p in files}
        for rel in sorted(set(known) - on_disk):
            with db.transaction():
                db.execute("DELETE FROM graph_files WHERE path = ?",
                           (rel,))
                db.execute("DELETE FROM graph_nodes WHERE file = ?",
                           (rel,))
                db.execute("DELETE FROM graph_edges WHERE file = ?",
                           (rel,))
            stats.removed += 1
        # Global symbol index: every module's top-level names, from the
        # store (unchanged files) merged with this build's parses.
        global_symbols: dict[str, dict[str, str]] = {}
        for row in db.query(
                "SELECT qualified, name FROM graph_nodes"
                " WHERE kind IN ('function', 'class') AND parent = ''"):
            mod = row["qualified"].rsplit(".", 1)[0] \
                if "." in row["qualified"] else ""
            global_symbols.setdefault(mod, {})[row["name"]] = \
                row["qualified"]
        in_repo_modules = set(global_symbols)
        for parse in parses.values():
            for sym in parse.symbols:
                if sym.kind in ("function", "class") and not sym.parent:
                    global_symbols.setdefault(parse.module, {})[
                        sym.name] = sym.qualified
            in_repo_modules.add(parse.module)
        # Resolve + write the changed files (old rows for the file go
        # first so a re-parse never leaves stale edges behind).
        for rel, parse in parses.items():
            nodes, edges = _resolve_file(parse, global_symbols,
                                         in_repo_modules)
            with db.transaction():
                db.execute("DELETE FROM graph_nodes WHERE file = ?",
                           (rel,))
                db.execute("DELETE FROM graph_edges WHERE file = ?",
                           (rel,))
                db.executemany(
                    "INSERT OR REPLACE INTO graph_nodes"
                    " (id, kind, name, qualified, file, line, end_line,"
                    "  docstring, parent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(n["id"], n["kind"], n["name"], n["qualified"],
                      n["file"], n["line"], n["end_line"],
                      n["docstring"], n["parent"]) for n in nodes])
                for i in range(0, len(edges), _BATCH):
                    db.executemany(
                        "INSERT OR REPLACE INTO graph_edges"
                        " (src, dst, kind, file, line)"
                        " VALUES (?, ?, ?, ?, ?)",
                        [(e["src"], e["dst"], e["kind"], e["file"],
                          e["line"]) for e in edges[i:i + _BATCH]])
        with db.transaction():
            db.execute(
                "INSERT OR REPLACE INTO graph_meta (key, value)"
                " VALUES ('parser', ?), ('version', ?),"
                " ('built_at', ?), ('root', ?)",
                (self.parser, str(GRAPH_VERSION), str(time.time()),
                 str(self.root)))
        stats.nodes = db.scalar("SELECT COUNT(*) FROM graph_nodes",
                                default=0) or 0
        stats.edges = db.scalar("SELECT COUNT(*) FROM graph_edges",
                                default=0) or 0
        stats.seconds = time.monotonic() - started
        return stats

    # ── queries ──

    def _node_rows(self, sql: str, params: tuple = ()) -> list[dict]:
        return self._open().query(sql, params)

    def who_calls(self, name: str, limit: int = 20) -> list[dict]:
        """Functions that call ``name`` (resolved edges only)."""
        db = self._open()
        targets = self._definition_ids(name)
        if not targets:
            return []
        out: list[dict] = []
        for target in targets:
            for row in db.query(
                    "SELECT e.src, e.file, e.line, n.qualified,"
                    " n.kind FROM graph_edges e"
                    " LEFT JOIN graph_nodes n ON n.id = e.src"
                    " WHERE e.dst = ? AND e.kind = 'calls'"
                    " ORDER BY e.file, e.line",
                    (target,)):
                out.append({
                    "caller": row["qualified"] or row["src"],
                    "kind": row["kind"] or "",
                    "file": row["file"], "line": row["line"],
                    "target": target,
                })
        return out[:limit]

    def calls(self, name: str, limit: int = 20) -> list[dict]:
        """What ``name`` calls (resolved edges + external names)."""
        db = self._open()
        sources = self._definition_ids(name)
        if not sources:
            return []
        out: list[dict] = []
        for src in sources:
            for row in db.query(
                    "SELECT e.dst, e.file, e.line, n.qualified, n.kind"
                    " FROM graph_edges e"
                    " LEFT JOIN graph_nodes n ON n.id = e.dst"
                    " WHERE e.src = ? AND e.kind = 'calls'"
                    " ORDER BY e.file, e.line",
                    (src,)):
                out.append({
                    "callee": row["qualified"] or row["dst"],
                    "kind": row["kind"] or "external",
                    "file": row["file"], "line": row["line"],
                    "source": src,
                })
        return out[:limit]

    def where_defined(self, name: str, limit: int = 20) -> list[dict]:
        """Where ``name`` is defined (file + line + kind)."""
        rows = self._node_rows(
            "SELECT qualified, kind, file, line, end_line, docstring"
            " FROM graph_nodes WHERE kind IN ('function', 'method',"
            " 'class') AND (name = ? OR qualified = ? OR"
            " qualified LIKE ?)"
            " ORDER BY qualified LIMIT ?",
            (name, name, f"%.{name}", limit))
        return [{
            "qualified": r["qualified"], "kind": r["kind"],
            "file": r["file"], "line": r["line"],
            "end_line": r["end_line"],
            "docstring": r["docstring"],
        } for r in rows]

    def owner(self, path: str) -> dict[str, Any]:
        """What owns ``path`` (a repo-relative file or a dotted
        module): the top-level package under the root, the module,
        and the symbols it defines. Root-level files are owned by
        the repo itself."""
        rel = path.replace("\\", "/").strip("/")
        module = (_dotted_module(rel) if rel.endswith(".py")
                  else rel.replace("/", "."))
        top = module.split(".")[0] if module else ""
        if "/" not in rel and "." not in module:
            top = self.root.name
        db = self._open()
        symbols = [{
            "qualified": r["qualified"], "kind": r["kind"],
            "line": r["line"],
        } for r in db.query(
            "SELECT qualified, kind, line FROM graph_nodes"
            " WHERE file = ? AND kind IN ('function', 'method', 'class')"
            " ORDER BY line", (rel,))]
        if not symbols and module:
            symbols = [{
                "qualified": r["qualified"], "kind": r["kind"],
                "line": r["line"],
            } for r in db.query(
                "SELECT qualified, kind, line FROM graph_nodes"
                " WHERE qualified LIKE ? AND kind IN ('function',"
                " 'method', 'class') ORDER BY line",
                (module + ".%",))]
        return {"path": rel, "module": module, "owner": top,
                "symbols": symbols}

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Free-text search over names + docstrings: exact name >
        prefix > substring > docstring hit. No embeddings, no LLM —
        deterministic ranking."""
        q = (query or "").strip().lower()
        if not q:
            return []
        db = self._open()
        scored: list[tuple[float, dict]] = []
        for row in db.query(
                "SELECT qualified, kind, name, file, line, docstring"
                " FROM graph_nodes"
                " WHERE kind IN ('function', 'method', 'class', 'module')"):
            name = (row["name"] or "").lower()
            doc = (row["docstring"] or "").lower()
            score = 0.0
            if name == q:
                score = 100.0
            elif name.startswith(q):
                score = 80.0
            elif q in name:
                score = 60.0
            elif q in doc:
                score = 40.0
            else:
                import difflib
                ratio = difflib.SequenceMatcher(None, q, name).ratio()
                if ratio > 0.6:
                    score = 30.0 * ratio
            if score > 0:
                scored.append((score, {
                    "qualified": row["qualified"], "kind": row["kind"],
                    "name": row["name"], "file": row["file"],
                    "line": row["line"], "docstring": row["docstring"],
                    "score": round(score, 1),
                }))
        scored.sort(key=lambda item: (-item[0], item[1]["qualified"]))
        return [hit for _, hit in scored[:limit]]

    def status(self) -> dict[str, Any]:
        """Graph status: what is stored, which parser built it, when."""
        db = self._open()
        meta = {r["key"]: r["value"] for r in db.query(
            "SELECT key, value FROM graph_meta")}
        return {
            "root": str(self.root),
            "store": str(self.db_path),
            "parser": meta.get("parser", ""),
            "version": meta.get("version", ""),
            "built_at": meta.get("built_at", ""),
            "files": db.scalar("SELECT COUNT(*) FROM graph_files",
                               default=0) or 0,
            "nodes": db.scalar("SELECT COUNT(*) FROM graph_nodes",
                               default=0) or 0,
            "edges": db.scalar("SELECT COUNT(*) FROM graph_edges",
                               default=0) or 0,
        }

    def _definition_ids(self, name: str) -> list[str]:
        """Node ids for definitions named ``name`` (bare or qualified)."""
        rows = self._node_rows(
            "SELECT id FROM graph_nodes"
            " WHERE kind IN ('function', 'method', 'class')"
            " AND (name = ? OR qualified = ?)",
            (name, name))
        return [r["id"] for r in rows]


# ── tools registry ────────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Attach the code-graph tools to a registry."""
    from ..core.errors import ToolError
    from ..core.policy import Capability
    from .filesystem import safe_path

    context = registry.context

    def _graph(path: str) -> CodeGraph:
        target = safe_path(context, path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        return CodeGraph(str(target))

    @registry.register(
        "graph_build",
        description="Build (or incrementally rebuild) the queryable "
                    "code graph over a repo: files, modules, classes, "
                    "functions, resolved call/import/inherit edges.",
        capability=Capability.FS_READ,
    )
    def graph_build(path: str = ".", force: bool = False) -> dict[str, Any]:
        graph = _graph(path)
        try:
            return graph.build(force=force).to_dict()
        finally:
            graph.close()

    @registry.register(
        "graph_query",
        description="Query the code graph: who_calls / calls / "
                    "where_defined / owner / search over the repo's "
                    "own code. op=who_calls|calls|where_defined|"
                    "owner|search.",
        capability=Capability.MEM_READ,
    )
    def graph_query(path: str = ".", op: str = "where_defined",
                    name: str = "", query: str = "",
                    limit: int = 10) -> dict[str, Any]:
        graph = _graph(path)
        try:
            if op == "who_calls":
                return {"op": op, "target": name,
                        "callers": graph.who_calls(name, limit=limit)}
            if op == "calls":
                return {"op": op, "target": name,
                        "callees": graph.calls(name, limit=limit)}
            if op == "where_defined":
                return {"op": op, "target": name,
                        "definitions": graph.where_defined(
                            name, limit=limit)}
            if op == "owner":
                return {"op": op, "target": name,
                        "owner": graph.owner(name)}
            if op == "search":
                return {"op": op, "query": query,
                        "results": graph.search(query, limit=limit)}
            raise ToolError(
                f"unknown op: {op!r} (want who_calls|calls|"
                "where_defined|owner|search)")
        finally:
            graph.close()

    @registry.register(
        "graph_status",
        description="Code-graph status: files/nodes/edges stored, "
                    "which parser built it, when.",
        capability=Capability.MEM_READ,
    )
    def graph_status(path: str = ".") -> dict[str, Any]:
        graph = _graph(path)
        try:
            return graph.status()
        finally:
            graph.close()
