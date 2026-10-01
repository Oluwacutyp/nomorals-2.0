"""Structural repo index: repo map, ranked symbol search, context packing.

Cached per-root AST index for coding missions — the fast structural layer
under the coding agent.  Complements (but does not depend on)
:mod:`nomorals.tools.repo_context`'s Phase-D ``rank_context`` mission
assembler and the embedding-backed :mod:`nomorals.tools.code_indexer`.

Lives in its own module (rather than inside ``repo_context.py``) because
that file is owned by the parallel Phase-D stream; a separate module keeps
both streams' work merge-clean.  Stdlib-only (``ast``, ``difflib``) plus
``core.logging_setup`` — no DB, no embeddings, no network.

Pieces:

- :func:`build_repo_map` — bounded directory tree, per-file purpose lines
  (first docstring line), per-module top-level symbol index
  (``RepoMap`` dataclass).  Not to be confused with
  ``nomorals.agents.repo_map.build_repo_map``, which returns a compact
  planner *string* for ``CodingAgent._plan_files``.
- :class:`RepoIndex` — ``find_symbol`` (ranked exact > prefix > substring
  > fuzzy), ``who_imports`` (AST import scanning), ``callers``
  (name-based call-site lookup), ``refresh`` / ``stale`` (incremental
  re-index via mtime+size).
- :func:`pack_context` — deterministic token-budgeted file packing for a
  coding task (query files → import neighbors → symbol overlap).

Token estimate heuristic
------------------------
``estimated_tokens = chars / 4``.  Rough but standard for English/code
mixed text; real BPE token counts vary by tokenizer (dense identifier
code tokenizes heavier, prose lighter).  ``pack_context`` treats the
budget conservatively — it stops *before* exceeding
``token_budget * 4`` chars — so the heuristic's error is absorbed rather
than propagated.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "CHARS_PER_TOKEN",
    "SKIP_DIRS", "PYTHON_SUFFIX", "MAX_TREE_DEPTH", "MAX_TREE_WIDTH",
    "RepoIndex", "RepoMap", "FileEntry", "SymbolInfo", "SymbolHit",
    "CallerHit", "ImportInfo", "PackedFile", "PackedContext",
    "build_repo_map", "find_symbol", "who_imports", "callers",
    "pack_context", "refresh_index", "stale_files", "get_repo_index",
    "estimate_tokens",
]

_log = get_logger(__name__)

#: Token estimate heuristic: chars per token (documented above).
CHARS_PER_TOKEN = 4

#: Directories never descended into by the structural index.
SKIP_DIRS = frozenset({
    "node_modules", ".git", "__pycache__", ".venv", "venv", ".tox",
    "dist", "build", ".next", ".nuxt", "coverage", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "eggs", ".hg", ".svn",
    "__snapshots__",
})

#: Only Python files get AST treatment (symbols, imports, docstrings).
PYTHON_SUFFIX = ".py"

#: Repo-map tree shape bounds.
MAX_TREE_DEPTH = 6
MAX_TREE_WIDTH = 80

#: Words ignored when scoring symbol-name overlap with task text.
_TASK_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "your",
    "you", "are", "was", "were", "has", "have", "had", "will", "would",
    "should", "could", "there", "their", "which", "when", "what", "how",
    "add", "new", "use", "used", "using", "make", "made", "fix", "fixed",
    "bug", "code", "file", "files", "repo", "please", "need", "needs",
    "want", "over", "under", "also", "just", "like", "than",
    "then", "them", "they", "its", "all", "any", "can", "not", "but",
})


def estimate_tokens(text: str) -> int:
    """Rough token estimate for ``text``: ``len(text) // CHARS_PER_TOKEN``.

    Heuristic (chars/4), not a tokenizer.  Good enough for budget
    enforcement; the packer stops before exceeding the budget so the
    error margin stays on the safe side.
    """
    return len(text) // CHARS_PER_TOKEN


# ── dataclasses ────────────────────────────────────────────────────────────


@dataclass
class SymbolInfo:
    """One symbol extracted from a Python module (AST, no regex)."""

    name: str
    kind: str  # "function" | "class" | "method"
    file: str  # repo-relative path
    line: int
    end_line: int = 0
    parent: str = ""  # enclosing class for methods
    doc: str = ""  # first docstring line

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "parent": self.parent,
            "doc": self.doc,
        }


@dataclass
class ImportInfo:
    """One import statement found in a module."""

    module: str  # dotted module, "" for bare `from . import x`
    names: list[str] = field(default_factory=list)  # `from X import a, b`
    level: int = 0  # leading-dot count for relative imports
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "names": self.names,
            "level": self.level,
            "line": self.line,
        }


@dataclass
class FileEntry:
    """One file in the repo map: purpose line + top-level symbols."""

    path: str  # repo-relative
    purpose: str  # first module-docstring line, or ""
    symbols: list[SymbolInfo] = field(default_factory=list)  # top-level only
    is_python: bool = True
    size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "purpose": self.purpose,
            "symbols": [s.to_dict() for s in self.symbols],
            "is_python": self.is_python,
            "size": self.size,
        }


@dataclass
class RepoMap:
    """Bounded structural map of a repository."""

    root: str
    tree: dict[str, Any]
    files: list[FileEntry]
    total_files: int
    total_symbols: int  # top-level functions + classes across the map
    truncated: bool
    max_files: int
    build_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "tree": self.tree,
            "files": [f.to_dict() for f in self.files],
            "total_files": self.total_files,
            "total_symbols": self.total_symbols,
            "truncated": self.truncated,
            "max_files": self.max_files,
            "build_seconds": self.build_seconds,
        }


@dataclass
class SymbolHit:
    """One ranked symbol-search hit."""

    symbol: SymbolInfo
    match: str  # "exact" | "prefix" | "substring" | "fuzzy"
    score: float

    def to_dict(self) -> dict[str, Any]:
        d = self.symbol.to_dict()
        d.update({"match": self.match, "score": round(self.score, 4)})
        return d


@dataclass
class CallerHit:
    """One call site referencing a function name.

    Name-based, not type-resolved: ``obj.method()`` is attributed to
    ``method`` wherever the attribute name matches.  Documented
    approximation — good for impact analysis, not for refactoring safety.
    """

    file: str  # repo-relative
    line: int
    caller: str  # enclosing function name, or "<module>"
    call_kind: str  # "name" (bare call) | "attribute" (obj.name call)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "caller": self.caller,
            "call_kind": self.call_kind,
        }


@dataclass
class PackedFile:
    """One file's contribution to a packed context."""

    path: str
    score: float
    rank_reason: str  # "query" | "import-neighbor" | "symbol-overlap"
    purpose: str
    symbols: list[str] = field(default_factory=list)  # "name (kind, Lline)"
    content: str = ""  # full or truncated source ("" when header-only)
    content_truncated: bool = False
    chars: int = 0  # chars this file contributes to the packed total

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "score": round(self.score, 3),
            "rank_reason": self.rank_reason,
            "purpose": self.purpose,
            "symbols": self.symbols,
            "content": self.content,
            "content_truncated": self.content_truncated,
            "chars": self.chars,
        }


@dataclass
class PackedContext:
    """Token-budgeted file packing for a coding mission."""

    root: str
    task: str
    files: list[PackedFile]
    total_chars: int
    estimated_tokens: int
    token_budget: int
    truncated: list[str] = field(default_factory=list)  # cut or dropped

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "task": self.task,
            "files": [f.to_dict() for f in self.files],
            "total_chars": self.total_chars,
            "estimated_tokens": self.estimated_tokens,
            "token_budget": self.token_budget,
            "truncated": self.truncated,
        }


class RepoIndex:
    """Cached per-root AST index of a repository.

    Parses every ``*.py`` file once (stdlib ``ast`` — no regex guessing),
    then answers structural queries without re-walking the tree.  Import
    lists are extracted lazily on first use (the full-tree ``ast.walk``
    costs ~2x the parse), so ``build_repo_map``/``find_symbol`` never pay
    for them.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"not a directory: {root}")
        self._files: dict[str, FileEntry] = {}
        self._all_symbols: dict[str, list[SymbolInfo]] = {}  # incl. methods
        # Import lists are extracted LAZILY (the full-tree ast.walk costs
        # ~2x the parse itself).  None = not extracted yet; build_repo_map
        # and find_symbol never pay for it.
        self._imports: dict[str, list[ImportInfo] | None] = {}
        self._modules: dict[str, str] = {}  # rel -> dotted module name
        self._module_to_file: dict[str, str] = {}  # dotted module -> rel
        self._mtimes: dict[str, tuple[float, int]] = {}  # rel -> (mtime, size)
        self.index()

    # -- building ------------------------------------------------------

    def index(self) -> dict[str, int]:
        """(Re)build the whole index.  Returns ``{"files": n, "symbols": m}``.

        File discovery + reads are serial (IO); the AST parse + symbol
        extraction fans out over a small process pool when there are
        enough Python files to amortize the pool startup, with a serial
        fallback if the pool is unavailable.
        """
        started = time.perf_counter()
        self._files.clear()
        self._all_symbols.clear()
        self._imports.clear()
        self._modules.clear()
        self._module_to_file.clear()
        self._mtimes.clear()
        jobs: list[tuple[str, str]] = []  # (rel, source) for the pool
        for rel, abs_path in self._iter_files():
            try:
                stat = abs_path.stat()
            except OSError:
                continue
            self._mtimes[rel] = (stat.st_mtime, stat.st_size)
            is_python = rel.endswith(PYTHON_SUFFIX)
            self._files[rel] = FileEntry(path=rel, purpose="",
                                         is_python=is_python, size=stat.st_size)
            self._all_symbols[rel] = []
            self._imports[rel] = None  # lazy: extracted on first use
            if is_python:
                try:
                    src = abs_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if src:
                    jobs.append((rel, src))
        for rel, purpose, top, methods, dotted in _pmap(_parse_job, jobs):
            entry = self._files[rel]
            entry.purpose = purpose
            entry.symbols = top
            self._all_symbols[rel] = top + methods
            self._modules[rel] = dotted
            self._module_to_file[dotted] = rel
        stats = {
            "files": len(self._files),
            "symbols": sum(len(v) for v in self._all_symbols.values()),
        }
        _log.info("repo index built for %s: %s files in %.2fs",
                  self.root, stats["files"], time.perf_counter() - started)
        return stats

    def _iter_files(self):
        """Yield ``(rel, abs_path)`` for every file, skipping ``SKIP_DIRS``.

        Deterministic: directories and files are visited in sorted order.
        """
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            )
            for fname in sorted(filenames):
                abs_path = Path(dirpath) / fname
                yield abs_path.relative_to(self.root).as_posix(), abs_path

    def _parse_and_store(self, rel: str, abs_path: Path) -> None:
        """Parse one file and (re)store its index entries."""
        try:
            stat = abs_path.stat()
        except OSError:
            return
        self._mtimes[rel] = (stat.st_mtime, stat.st_size)
        is_python = rel.endswith(PYTHON_SUFFIX)
        entry = FileEntry(path=rel, purpose="", is_python=is_python,
                          size=stat.st_size)
        all_symbols: list[SymbolInfo] = []
        if is_python:
            try:
                src = abs_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                src = ""
            if src:
                try:
                    tree = ast.parse(src)
                except (SyntaxError, ValueError):
                    tree = None
                if tree is not None:
                    entry.purpose = _module_purpose(tree)
                    top, methods = _extract_symbols(tree, rel)
                    entry.symbols = top
                    all_symbols = top + methods
                    dotted = _dotted_module(rel)
                    self._modules[rel] = dotted
                    self._module_to_file[dotted] = rel
        self._files[rel] = entry
        self._all_symbols[rel] = all_symbols
        self._imports[rel] = None  # lazy: extracted on first use

    def _ensure_imports(self, rel: str) -> list[ImportInfo]:
        """Import list for ``rel``, extracting on first use (cached)."""
        imports = self._imports.get(rel)
        if imports is None and rel in self._files:
            imports = []
            if self._files[rel].is_python:
                try:
                    src = (self.root / rel).read_text(
                        encoding="utf-8", errors="ignore")
                    tree = ast.parse(src)
                except (OSError, SyntaxError, ValueError):
                    tree = None
                if tree is not None:
                    imports = _extract_imports(tree)
            self._imports[rel] = imports
        return imports or []

    def _drop(self, rel: str) -> None:
        """Remove every index entry for ``rel`` (file deleted)."""
        self._files.pop(rel, None)
        self._all_symbols.pop(rel, None)
        self._imports.pop(rel, None)
        dotted = self._modules.pop(rel, None)
        if dotted:
            self._module_to_file.pop(dotted, None)
        self._mtimes.pop(rel, None)

    # -- incremental ---------------------------------------------------

    def stale(self, paths: list[str] | None = None) -> list[str]:
        """Sorted rel paths whose content changed since indexing.

        Compares ``(mtime, size)`` — cheap, no re-read.  Missing files and
        never-indexed paths also count as stale.  ``paths=None`` scans the
        whole index.
        """
        targets = paths if paths is not None else list(self._files)
        out: list[str] = []
        for rel in targets:
            rec = self._mtimes.get(rel)
            abs_path = self.root / rel
            try:
                stat = abs_path.stat()
            except OSError:
                out.append(rel)
                continue
            if rec is None or (stat.st_mtime, stat.st_size) != rec:
                out.append(rel)
        return sorted(out)

    def refresh(self, paths: list[str]) -> list[str]:
        """Re-parse only ``paths``; everything else is untouched.

        Returns the sorted rel paths that were actually re-parsed (or
        dropped, for deleted files).  Unchanged files are skipped.
        """
        refreshed: list[str] = []
        for rel in paths:
            abs_path = self.root / rel
            if not abs_path.is_file():
                if rel in self._files:
                    self._drop(rel)
                    refreshed.append(rel)
                continue
            try:
                stat = abs_path.stat()
            except OSError:
                continue
            if self._mtimes.get(rel) == (stat.st_mtime, stat.st_size):
                continue  # unchanged — keep the cached parse
            self._parse_and_store(rel, abs_path)
            refreshed.append(rel)
        return sorted(refreshed)

    # -- symbol search ---------------------------------------------------

    def find_symbol(self, name: str, *, kind: str | None = None,
                    fuzzy: bool = False, limit: int = 50) -> list[SymbolHit]:
        """Ranked symbol search.

        Tiers, in order: ``exact`` (score 1.0) > ``prefix`` (0.9) >
        ``substring`` (0.8) > ``fuzzy`` (0.7 * difflib ratio, only when
        ``fuzzy=True`` and ratio >= 0.6).  Ties break deterministically by
        file then line.  ``kind`` filters to ``"function"`` / ``"class"`` /
        ``"method"``.
        """
        name = (name or "").strip()
        if not name:
            return []
        scored: list[tuple[int, float, str, int, str, float, SymbolInfo]] = []
        for _rel, symbols in self._all_symbols.items():
            for sym in symbols:
                if kind and sym.kind != kind:
                    continue
                if sym.name == name:
                    tier, match, score = 0, "exact", 1.0
                elif sym.name.startswith(name):
                    tier, match, score = 1, "prefix", 0.9
                elif name in sym.name:
                    tier, match, score = 2, "substring", 0.8
                elif fuzzy:
                    ratio = difflib.SequenceMatcher(
                        None, name, sym.name).ratio()
                    if ratio < 0.6:
                        continue
                    tier, match, score = 3, "fuzzy", 0.7 * ratio
                else:
                    continue
                scored.append(
                    (tier, -score, sym.file, sym.line, match, score, sym))
        scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [SymbolHit(symbol=t[6], match=t[4], score=t[5])
                for t in scored[:limit]]

    def who_imports(self, symbol_or_module: str) -> list[str]:
        """Sorted rel paths of files importing ``symbol_or_module``.

        Matches dotted module paths (``a.b.c``), bare module names
        (``c`` matches ``a.b.c``), and individual imported names
        (``from x import Thing`` matches ``Thing``).  Pure AST import
        scanning — no regex.
        """
        target = (symbol_or_module or "").strip()
        if not target:
            return []
        importers: set[str] = set()
        for rel in self._files:
            for imp in self._ensure_imports(rel):
                mod = imp.module
                base = mod.rsplit(".", 1)[-1] if mod else ""
                if (target == mod
                        or (mod and mod.endswith("." + target))
                        or base == target
                        or target in imp.names):
                    importers.add(rel)
                    break
        return sorted(importers)

    def callers(self, func_name: str) -> list[CallerHit]:
        """Approximate call sites of ``func_name`` across the repo.

        Finds ``ast.Call`` nodes whose function is the bare name or an
        attribute with that name, recording the enclosing function (or
        ``"<module>"``).  This is **name-based, not type-resolved**: two
        unrelated objects sharing a method name both match.  Useful for
        impact analysis; not safe enough to drive automated refactors.
        """
        name = (func_name or "").strip()
        if not name:
            return []
        hits: list[CallerHit] = []
        for rel in sorted(self._files):
            if not rel.endswith(PYTHON_SUFFIX):
                continue
            try:
                src = (self.root / rel).read_text(
                    encoding="utf-8", errors="ignore")
                tree = ast.parse(src)
            except (OSError, SyntaxError, ValueError):
                continue
            for lineno, caller, call_kind in _find_callers(tree, name):
                hits.append(CallerHit(file=rel, line=lineno, caller=caller,
                                      call_kind=call_kind))
        hits.sort(key=lambda h: (h.file, h.line))
        return hits

    # -- import graph ------------------------------------------------------

    def _package_of(self, rel: str) -> str:
        mod = self._modules.get(rel, "")
        if rel.endswith("/__init__.py") or rel == "__init__.py":
            return mod
        return mod.rpartition(".")[0]

    def _resolve_import(self, rel: str, imp: ImportInfo) -> list[str]:
        """Resolve an ImportInfo to indexed rel paths (best-effort)."""
        candidates: list[str] = []
        if imp.level:
            pkg = self._package_of(rel)
            parts = pkg.split(".") if pkg else []
            up = imp.level - 1
            base = (".".join(parts[:len(parts) - up])
                    if up <= len(parts) else "")
            if imp.module:
                candidates.append(f"{base}.{imp.module}" if base
                                  else imp.module)
            else:
                for nm in imp.names:
                    candidates.append(f"{base}.{nm}" if base else nm)
        elif imp.module:
            candidates.append(imp.module)
        resolved: list[str] = []
        for cand in candidates:
            hit = self._module_to_file.get(cand)
            if hit:
                resolved.append(hit)
                continue
            # suffix fallback: `import x` may mean `a.b.x`
            for modname in sorted(self._module_to_file):
                if modname == cand or modname.endswith("." + cand):
                    resolved.append(self._module_to_file[modname])
                    break
        return sorted(set(resolved))

    def imports_of(self, rel: str) -> set[str]:
        """Rel paths of indexed files that ``rel`` imports."""
        out: set[str] = set()
        for imp in self._ensure_imports(rel):
            out.update(self._resolve_import(rel, imp))
        out.discard(rel)
        return out

    def imported_by(self, rel: str) -> set[str]:
        """Rel paths of indexed files that import ``rel``."""
        out: set[str] = set()
        for other in self._files:
            if other != rel and rel in self.imports_of(other):
                out.add(other)
        return out


# ── parallel parse ─────────────────────────────────────────────────────────


#: Files above this count go through the process pool; below it the pool
#: startup would dominate.
_PARALLEL_THRESHOLD = 32


def _parse_job(job: tuple[str, str]
               ) -> tuple[str, str, list[SymbolInfo], list[SymbolInfo], str]:
    """Parse one ``(rel, source)`` job (runs in a pool worker or serially).

    Returns ``(rel, purpose, top_symbols, methods, dotted_module)``.
    Module-level (picklable) by design.
    """
    rel, src = job
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return (rel, "", [], [], "")
    purpose = _module_purpose(tree)
    top, methods = _extract_symbols(tree, rel)
    return (rel, purpose, top, methods, _dotted_module(rel))


def _pmap(func: Any, jobs: list[Any]) -> list[Any]:
    """Map ``func`` over ``jobs``: process pool when worthwhile, else serial.

    The pool is bounded (never more than 4 workers) and any failure —
    sandbox restrictions, spawn errors, anything — falls back to the
    serial path rather than breaking the index build.
    """
    if len(jobs) <= _PARALLEL_THRESHOLD:
        return [func(job) for job in jobs]
    try:
        import concurrent.futures
        workers = min(4, os.cpu_count() or 2)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers) as pool:
            return list(pool.map(func, jobs, chunksize=8))
    except Exception as exc:  # noqa: BLE001 — pool unavailable: go serial
        _log.debug("parallel parse unavailable (%s); using serial", exc)
        return [func(job) for job in jobs]


# ── AST helpers ──────────────────────────────────────────────────────────


def _module_purpose(tree: ast.Module) -> str:
    """First line of the module docstring, or ``""``."""
    doc = ast.get_docstring(tree) or ""
    doc = doc.strip()
    return doc.splitlines()[0].strip() if doc else ""


def _dotted_module(rel: str) -> str:
    stem = rel[: -len(PYTHON_SUFFIX)]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    elif stem == "__init__":
        stem = ""
    return stem.replace("/", ".")


def _extract_symbols(tree: ast.Module,
                     rel: str) -> tuple[list[SymbolInfo], list[SymbolInfo]]:
    """Top-level functions/classes and their methods (AST, no regex)."""
    top: list[SymbolInfo] = []
    methods: list[SymbolInfo] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top.append(_symbol_info(node, rel, "function"))
        elif isinstance(node, ast.ClassDef):
            top.append(_symbol_info(node, rel, "class"))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        _symbol_info(child, rel, "method", parent=node.name))
    return top, methods


def _symbol_info(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
                 rel: str, kind: str, parent: str = "") -> SymbolInfo:
    doc = ast.get_docstring(node) or ""
    doc = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    return SymbolInfo(
        name=node.name,
        kind=kind,
        file=rel,
        line=node.lineno,
        end_line=node.end_lineno or node.lineno,
        parent=parent,
        doc=doc,
    )


def _extract_imports(tree: ast.Module) -> list[ImportInfo]:
    """Every import statement in the module, via AST."""
    imports: list[ImportInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportInfo(module=alias.name, line=node.lineno))
        elif isinstance(node, ast.ImportFrom):
            imports.append(ImportInfo(
                module=node.module or "",
                names=[a.name for a in node.names],
                level=node.level or 0,
                line=node.lineno,
            ))
    return imports


def _find_callers(tree: ast.Module,
                  name: str) -> list[tuple[int, str, str]]:
    """``(lineno, enclosing, kind)`` for Call nodes referencing ``name``."""
    out: list[tuple[int, str, str]] = []
    stack: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()
            return
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == name:
                out.append((node.lineno,
                            stack[-1] if stack else "<module>", "name"))
            elif isinstance(func, ast.Attribute) and func.attr == name:
                out.append((node.lineno,
                            stack[-1] if stack else "<module>", "attribute"))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return out


# ── repo map ─────────────────────────────────────────────────────────────


def _build_tree(files: list[str], root_name: str) -> tuple[dict[str, Any], bool]:
    """Nested ``{"name","type","children"|"purpose"}`` tree.

    Bounded: at most ``MAX_TREE_DEPTH`` levels and ``MAX_TREE_WIDTH``
    children per directory; excess is collapsed into ``"… +N more"``
    nodes.  Returns ``(tree, capped)``.
    """
    capped = False
    dirs: dict[str, Any] = {}

    def get_dir(parts: list[str]) -> dict[str, Any]:
        node = dirs
        for part in parts:
            node = node.setdefault(part, {})
        return node

    for rel in files:
        parts = rel.split("/")
        if len(parts) > MAX_TREE_DEPTH:
            # collapse the deep tail into one node
            parts = (parts[:MAX_TREE_DEPTH - 1]
                     + ["/".join(parts[MAX_TREE_DEPTH - 1:])])
        *dir_parts, fname = parts
        get_dir(dir_parts)[fname] = None  # file marker

    def convert(name: str, node: Any) -> dict[str, Any]:
        nonlocal capped
        if node is None:  # file
            return {"name": name, "type": "file"}
        children = [convert(child_name, node[child_name])
                    for child_name in sorted(node)]
        if len(children) > MAX_TREE_WIDTH:
            capped = True
            children = (children[:MAX_TREE_WIDTH]
                        + [{"name": f"… +{len(children) - MAX_TREE_WIDTH} more",
                            "type": "truncated"}])
        return {"name": name, "type": "dir", "children": children}

    tree = {"name": root_name, "type": "dir",
            "children": [convert(n, dirs[n]) for n in sorted(dirs)]}
    return tree, capped


def build_repo_map(root: str | Path, *, max_files: int = 400) -> RepoMap:
    """Build a bounded structural map of the repo at ``root``.

    - directory tree (depth ≤ ``MAX_TREE_DEPTH``, width ≤
      ``MAX_TREE_WIDTH`` per dir; skips ``SKIP_DIRS``)
    - per-file purpose = first module-docstring line (``""`` when absent)
    - per-module symbol index: top-level functions/classes with line
      numbers, extracted with ``ast`` (no regex guessing)
    - total file / symbol counts; ``truncated=True`` when the file list
      was cut at ``max_files`` or the tree was capped

    Runs on repos of a few hundred files in well under 5 seconds.
    """
    started = time.perf_counter()
    idx = get_repo_index(root)
    all_rels = sorted(idx._files)
    total_files = len(all_rels)
    shown = all_rels[:max_files]
    truncated = total_files > max_files

    tree, tree_capped = _build_tree(shown, idx.root.name)
    truncated = truncated or tree_capped
    purposes = {rel: idx._files[rel].purpose for rel in shown}

    def annotate(node: dict[str, Any], prefix: str) -> None:
        if node["type"] == "file":
            rel = f"{prefix}{node['name']}" if prefix else node["name"]
            node["purpose"] = purposes.get(rel, "")
        elif node["type"] == "dir":
            child_prefix = (f"{prefix}{node['name']}/"
                            if node["name"] != idx.root.name else "")
            for child in node.get("children", []):
                annotate(child, child_prefix)

    annotate(tree, "")

    files = [idx._files[rel] for rel in shown]
    total_symbols = sum(len(f.symbols) for f in files)
    elapsed = time.perf_counter() - started
    _log.info("repo map for %s: %d files, %d symbols in %.2fs",
              idx.root, total_files, total_symbols, elapsed)
    return RepoMap(
        root=str(idx.root),
        tree=tree,
        files=files,
        total_files=total_files,
        total_symbols=total_symbols,
        truncated=truncated,
        max_files=max_files,
        build_seconds=elapsed,
    )


# ── module-level index cache + convenience functions ─────────────────────

_INDEX_CACHE: dict[str, RepoIndex] = {}


def get_repo_index(root: str | Path) -> RepoIndex:
    """Return the cached :class:`RepoIndex` for ``root``, building on demand."""
    key = str(Path(root).expanduser().resolve())
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = RepoIndex(key)
        _INDEX_CACHE[key] = idx
    return idx


def find_symbol(root: str | Path, name: str, *, kind: str | None = None,
                fuzzy: bool = False, limit: int = 50) -> list[SymbolHit]:
    """Ranked symbol search over ``root``.

    Tiers: exact > prefix > substring > fuzzy (difflib, only when
    ``fuzzy=True``); each hit carries its ``match`` kind and score.
    """
    return get_repo_index(root).find_symbol(name, kind=kind, fuzzy=fuzzy,
                                            limit=limit)


def who_imports(root: str | Path, symbol_or_module: str) -> list[str]:
    """Sorted rel paths of files importing ``symbol_or_module`` (AST scan)."""
    return get_repo_index(root).who_imports(symbol_or_module)


def callers(root: str | Path, func_name: str) -> list[CallerHit]:
    """Approximate call sites of ``func_name`` (name-based, not type-resolved)."""
    return get_repo_index(root).callers(func_name)


def refresh_index(root: str | Path, paths: list[str]) -> list[str]:
    """Re-parse only ``paths`` in the cached index for ``root``."""
    return get_repo_index(root).refresh(paths)


def stale_files(root: str | Path,
                paths: list[str] | None = None) -> list[str]:
    """Sorted rel paths whose content changed since the index was built."""
    return get_repo_index(root).stale(paths)


# ── context packing ──────────────────────────────────────────────────────


def _resolve_query_file(idx: RepoIndex, query: str) -> str | None:
    """Resolve a query path to an indexed rel path (exact → suffix → base)."""
    q = query.strip().lstrip("./")
    if not q:
        return None
    rels = idx._files
    if q in rels:
        return q
    suffix = [r for r in rels if r.endswith("/" + q)]
    if suffix:
        return sorted(suffix)[0]
    base = [r for r in rels if r.rsplit("/", 1)[-1] == q.rsplit("/", 1)[-1]]
    if base:
        return sorted(base)[0]
    return None


def _task_words(task: str) -> list[str]:
    """Identifier-ish words from task text, stopwords removed, sorted."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", (task or "").lower())
    return sorted({w for w in words if w not in _TASK_STOPWORDS})


def pack_context(root: str | Path, query_files: list[str], task: str = "",
                 *, token_budget: int = 6000) -> PackedContext:
    """Pack repo context for a coding task into a token budget.

    Ranking (deterministic — ties break by path):

    1. ``query_files`` first (score 100, reason ``"query"``)
    2. files importing, or imported by, the query files via the import
       graph (score 50, reason ``"import-neighbor"``)
    3. files whose symbol names overlap the task text (score up to 40,
       reason ``"symbol-overlap"``)

    Every file contributes its purpose line + key symbols.  Top-ranked
    files (query + import-neighbors) additionally contribute full source,
    truncated to fit.  Packing **stops before exceeding**
    ``token_budget * CHARS_PER_TOKEN`` chars: the first file whose header
    no longer fits ends the pack, and the rest land in ``truncated``.
    Token counts are the chars/4 heuristic (see the module docstring).
    """
    idx = get_repo_index(root)
    char_budget = max(0, token_budget * CHARS_PER_TOKEN)

    resolved: list[str] = []
    for q in query_files or []:
        r = _resolve_query_file(idx, q)
        if r and r not in resolved:
            resolved.append(r)
    resolved.sort()

    neighbors: set[str] = set()
    for q in resolved:
        neighbors |= idx.imported_by(q)
        neighbors |= idx.imports_of(q)
    neighbors -= set(resolved)
    neighbors = {n for n in neighbors if n in idx._files}

    words = _task_words(task)
    overlap: dict[str, int] = {}
    if words:
        for rel, symbols in idx._all_symbols.items():
            if rel in resolved or rel in neighbors:
                continue
            blob = " ".join(s.name.lower() for s in symbols)
            hits = sum(1 for w in words if w in blob)
            if hits:
                overlap[rel] = hits

    scored: list[tuple[float, str, str]] = []
    for q in resolved:
        scored.append((100.0, "query", q))
    for n in sorted(neighbors):
        scored.append((50.0, "import-neighbor", n))
    for rel, hits in overlap.items():
        scored.append((min(40.0, 10.0 * hits), "symbol-overlap", rel))
    scored.sort(key=lambda t: (-t[0], t[2]))

    packed: list[PackedFile] = []
    truncated: list[str] = []
    total = 0
    for score, reason, rel in scored:
        entry = idx._files[rel]
        sym_strs = [f"{s.name} ({s.kind}, L{s.line})"
                    for s in entry.symbols[:40]]
        header = (f"### {rel}\n"
                  f"Purpose: {entry.purpose or '(none)'}\n"
                  f"Symbols: {'; '.join(sym_strs) if sym_strs else '(none)'}\n")
        if total + len(header) > char_budget:
            truncated.append(rel)
            break  # stop before exceeding the budget
        total += len(header)

        content = ""
        content_truncated = False
        block = ""
        if reason in ("query", "import-neighbor"):
            try:
                content = (idx.root / rel).read_text(
                    encoding="utf-8", errors="ignore")
            except OSError:
                content = ""
            if content:
                fence_open, fence_close = "```\n", "\n```\n"
                room = (char_budget - total
                        - len(fence_open) - len(fence_close))
                if room <= 0:
                    content = ""
                elif len(content) > room:
                    marker = (f"\n…[content truncated: "
                              f"{len(content) - room} more chars]")
                    keep = max(0, room - len(marker))
                    content = content[:keep] + marker
                    content_truncated = True
                    truncated.append(rel)
                if content:
                    block = fence_open + content + fence_close
                    total += len(block)

        packed.append(PackedFile(
            path=rel,
            score=score,
            rank_reason=reason,
            purpose=entry.purpose,
            symbols=sym_strs,
            content=content,
            content_truncated=content_truncated,
            chars=len(header) + len(block),
        ))

    return PackedContext(
        root=str(idx.root),
        task=task or "",
        files=packed,
        total_chars=total,
        estimated_tokens=estimate_tokens("x" * total),
        token_budget=token_budget,
        truncated=truncated,
    )
