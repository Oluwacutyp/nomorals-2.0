"""Phase D: relevance-ranked repository context.

Replaces the old truncation (first 100 files, 2000 chars each, no ranking)
with a ranked assembly under a real token budget:

  rank 0 — direct targets (files the task names explicitly)
  rank 1 — test files covering the targets (same-package test_* matching)
  rank 2 — 1-hop import neighbors (ast-parsed, stdlib only)
  rank 3 — symbol-search hits for task keywords (injected callable)

Files fill top-down until the token budget is spent; everything cut is
recorded in the metadata (included_files, dropped_files, budget_tokens,
used_tokens) so the accounting is visible, never silent.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Conservative token estimate: 4 chars per token.
CHARS_PER_TOKEN = 4

#: Default context budget for ranked assembly.
DEFAULT_BUDGET_TOKENS = 12000

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".hg",
              ".svn", "__snapshots__"}

_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been by "
    "this that these those it its as at from into over under after "
    "before fix fixes fixed please".split()
)


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (4 chars/token, minimum 1)."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def extract_keywords(task: str, limit: int = 8) -> list[str]:
    """Task keywords for symbol search: lowercase words ≥3 chars, no stops."""
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", task.lower())
    seen: list[str] = []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def _iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*.py"):
        if path.is_file() and not any(part in _SKIP_DIRS
                                       for part in path.parts):
            out.append(path)
    return out


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def _resolve_relative(own_pkg: str, level: int, module: str) -> str:
    """Resolve a relative ``from`` import to a dotted module path."""
    base = own_pkg
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    full = f"{base}.{module}" if module else base
    return full.strip(".")


def import_neighbors(root: Path, rel: str) -> list[str]:
    """1-hop import neighbors of ``rel``: files it imports + files that
    import it (stdlib ``ast`` only).  Handles absolute and relative
    imports.  Best-effort — never raises."""
    try:
        target = (root / rel).resolve()
        if not target.is_file():
            return []
        tree = ast.parse(target.read_text(encoding="utf-8",
                                           errors="ignore"))
    except Exception:  # noqa: BLE001 — unparseable file: no neighbors
        return []
    own_mod = _module_name(root, target)
    own_pkg = own_mod.rpartition(".")[0]

    imported_mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                imported_mods.add(
                    _resolve_relative(own_pkg, node.level,
                                      node.module or ""))
            elif node.module:
                imported_mods.add(node.module)
    neighbors: set[str] = set()
    for mod in imported_mods:
        if not mod:
            continue
        # full dotted path -> file, then package __init__
        rel_p = mod.replace(".", "/") + ".py"
        if (root / rel_p).is_file() and (root / rel_p).resolve() != target:
            neighbors.add(rel_p)
            continue
        init_p = mod.replace(".", "/") + "/__init__.py"
        if (root / init_p).is_file() and \
                (root / init_p).resolve() != target:
            neighbors.add(init_p)
            continue
        # fallback: top-level stem anywhere (``import foo.bar`` -> foo.py)
        stem = mod.split(".")[0]
        for path in _iter_py_files(root):
            if path.resolve() == target or path.stem != stem:
                continue
            neighbors.add(str(path.relative_to(root)))
    # reverse direction: files importing this module's stem
    me = target.stem
    for path in _iter_py_files(root):
        if path.resolve() == target:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(rf"(^|\\W)(import\\s+{me}|from\\s+{me}\\s+import)",
                     text):
            neighbors.add(str(path.relative_to(root)))
    return sorted(neighbors)


def test_files_for(root: Path, rel: str) -> list[str]:
    """Test modules covering ``rel``: same-package ``test_<stem>`` /
    ``<stem>_test`` matches."""
    target = root / rel
    stem = target.stem
    parent = target.parent
    found: list[str] = []
    candidates = (f"test_{stem}.py", f"{stem}_test.py")
    for name in candidates:
        p = parent / name
        if p.is_file():
            found.append(str(p.relative_to(root)))
    # also: tests/ mirror (tests/test_<stem>.py)
    tests_dir = root / "tests"
    if tests_dir.is_dir():
        for name in candidates:
            p = tests_dir / name
            if p.is_file():
                found.append(str(p.relative_to(root)))
    return sorted(found)


@dataclass
class RankedContext:
    """Assembled context + its accounting."""
    files: dict[str, str] = field(default_factory=dict)  # rel -> content
    included_files: list[str] = field(default_factory=list)
    dropped_files: list[str] = field(default_factory=list)
    truncated_files: list[str] = field(default_factory=list)
    budget_tokens: int = DEFAULT_BUDGET_TOKENS
    used_tokens: int = 0
    structure: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "included_files": self.included_files,
            "dropped_files": self.dropped_files,
            "truncated_files": self.truncated_files,
            "budget_tokens": self.budget_tokens,
            "used_tokens": self.used_tokens,
            # invariant the caller can assert: never silently exceeded.
            "within_budget": self.used_tokens <= self.budget_tokens,
        }

    def metadata_json(self) -> str:
        return json.dumps(self.metadata(), indent=2)


_TRUNCATION_MARKER = "\n... [truncated to fit the context token budget]"


def _truncate_to_tokens(text: str, room_tokens: int) -> tuple[str, bool]:
    """Cut ``text`` to about ``room_tokens`` (4 chars/token estimate)."""
    max_chars = max(0, room_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def rank_context(
    root: str | Path,
    task: str,
    targets: list[str] | None = None,
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    symbol_search: Callable[[str, int], list[str]] | None = None,
    symbol_limit: int = 10,
) -> RankedContext:
    """Rank repo files by relevance to ``task`` and fill under budget.

    ``symbol_search(query, limit)`` maps a keyword query to candidate rel
    paths (rank 3).  When it also offers ``search_many(queries, limit)``,
    the index warm-up and all initial queries run in one parallel block
    instead of serially.  Pass None to skip symbol hits — ranking still
    works from targets, test files, and import neighbors.
    """
    root_p = Path(root).expanduser().resolve()
    targets = [t for t in (targets or []) if (root_p / t).is_file()]

    # candidate rel -> best (lowest) rank
    ranked: dict[str, int] = {}
    ordered: list[str] = []

    def add(rel: str, rank: int) -> None:
        if rel not in ranked:
            ranked[rel] = rank
            ordered.append(rel)
        elif rank < ranked[rel]:
            ranked[rel] = rank

    for t in targets:
        add(t, 0)
    for t in targets:
        for f in test_files_for(root_p, t):
            add(f, 1)
    for t in targets:
        for f in import_neighbors(root_p, t):
            add(f, 2)
    if symbol_search is not None:
        keywords = extract_keywords(task)
        hits_by_kw: dict[str, list[str]] = {}
        search_many = getattr(symbol_search, "search_many", None)
        if search_many is not None and len(keywords) > 1:
            # Parallel fast path: index warm-up + every initial query in
            # one bounded block instead of warm-then-serial-searches.
            try:
                hits_by_kw = search_many(keywords, symbol_limit) or {}
            except Exception as exc:  # noqa: BLE001 — search is advisory
                _log.debug("parallel symbol search failed: %s", exc)
        else:
            for kw in keywords:
                try:
                    hits_by_kw[kw] = symbol_search(kw, symbol_limit) or []
                except Exception as exc:  # noqa: BLE001 — advisory
                    _log.debug("symbol search %r failed: %s", kw, exc)
        for hits in hits_by_kw.values():
            for h in hits:
                if (root_p / h).is_file():
                    add(h, 3)

    # stable order: rank first, then discovery order
    discovery = {r: i for i, r in enumerate(ordered)}
    ordered.sort(key=lambda r: (ranked[r], discovery[r]))

    ctx = RankedContext(budget_tokens=budget_tokens)
    ctx.structure = "\n".join(sorted(
        str(p.relative_to(root_p)) for p in _iter_py_files(root_p)))
    for rel in ordered:
        try:
            content = (root_p / rel).read_text(encoding="utf-8",
                                               errors="ignore")
        except OSError:
            continue
        header = f"--- {rel} ---\n"
        cost = estimate_tokens(header + content)
        room = budget_tokens - ctx.used_tokens
        if cost > room:
            if not ctx.files:
                # The top-ranked file alone exceeds the budget: truncate
                # it to fit instead of silently exceeding the budget or
                # returning an empty context.  The cut is explicit in the
                # content (marker) and in the metadata (truncated_files).
                fit, was_cut = _truncate_to_tokens(
                    header + content,
                    room - estimate_tokens(_TRUNCATION_MARKER))
                ctx.files[rel] = (fit + _TRUNCATION_MARKER) if was_cut \
                    else fit
                ctx.included_files.append(rel)
                ctx.truncated_files.append(rel)
                ctx.used_tokens = budget_tokens
            else:
                ctx.dropped_files.append(rel)
            continue
        ctx.files[rel] = content
        ctx.included_files.append(rel)
        ctx.used_tokens += cost
    # anything never even considered a candidate is neither included nor
    # dropped — dropped means "ranked but cut by the budget".
    return ctx


def legacy_context(root: str | Path,
                   targets: list[str] | None = None) -> dict[str, str]:
    """Pre-Phase-D behavior (debug fallback): first 100 files, 2000 chars
    each, no ranking."""
    root_p = Path(root).expanduser().resolve()
    context: dict[str, str] = {}
    try:
        files: list[str] = []
        for path in root_p.rglob("*"):
            if path.is_file() and not any(
                skip in str(path)
                for skip in [".git", "__pycache__", "node_modules", ".venv"]
            ):
                files.append(str(path.relative_to(root_p)))
                if len(files) > 100:
                    break
        context["structure"] = "\n".join(sorted(files))
    except Exception:  # noqa: BLE001 — keep the old swallow
        context["structure"] = "Unable to read project structure"
    if targets:
        parts = []
        for f in targets:
            full = root_p / f
            if full.exists():
                parts.append(
                    f"--- {f} ---\n"
                    f"{full.read_text(encoding='utf-8', errors='ignore')[:2000]}")
        context["files"] = "\n\n".join(parts)
    context["metadata"] = json.dumps({"legacy": True, "ranked": False},
                                     indent=2)
    return context


__all__ = [
    "CHARS_PER_TOKEN", "DEFAULT_BUDGET_TOKENS",
    "RankedContext", "estimate_tokens", "extract_keywords",
    "import_neighbors", "test_files_for", "rank_context", "legacy_context",
]
