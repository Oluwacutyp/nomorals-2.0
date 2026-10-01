"""Compact repo map for the coding agent's mission planner (L5).

Builds a token-bounded sketch of a repo — top-level directories, per-file
purposes from module docstrings, and symbol counts from
``nomorals/tools/code_indexer.py``'s :class:`CodeIndexer` AST parser —
so the planner in ``CodingAgent._plan_files`` sees the real layout instead
of guessing paths.

The indexer is used parse-only: ``CodeIndexer._parse_python`` touches no
instance state, so it is invoked without a DB, embedding model, or vector
store.  Any failure (import error, unreadable tree, parse crash) yields
"" — the mission proceeds without the map, never breaks.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["build_repo_map"]

#: Hard caps: the map is planner context, not a listing.
_DEFAULT_MAX_CHARS = 2000
_MAX_FILES = 120
_MAX_FILE_BYTES = 256 * 1024
_MAX_PURPOSE_CHARS = 120

_indexer: object | None = None  # parse-only CodeIndexer (no DB/embeddings)


def _parse_units(code: str, rel: str) -> int:
    """Symbol count for one Python file via CodeIndexer's AST parser."""
    global _indexer
    try:
        if _indexer is None:
            from ..tools.code_indexer import CodeIndexer

            # Parse-only: _parse_python / _extract_* never touch instance
            # state, so no Database / embedding model / VectorStore needed.
            _indexer = CodeIndexer.__new__(CodeIndexer)
        return len(_indexer._parse_python(code, rel, ""))  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 — count is best-effort
        _log.debug("repo map: symbol parse failed for %s: %s", rel, exc)
        return 0


def _module_purpose(code: str) -> str:
    """First line of the module docstring, capped; "" when absent."""
    try:
        doc = ast.get_docstring(ast.parse(code))
    except (SyntaxError, ValueError):
        return ""
    if not doc:
        return ""
    first = doc.strip().splitlines()[0].strip()
    if len(first) > _MAX_PURPOSE_CHARS:
        first = first[:_MAX_PURPOSE_CHARS - 1] + "…"
    return first


def _build(root: Path, max_chars: int) -> str:
    try:
        from ..tools.code_indexer import SKIP_DIRS
    except Exception as exc:  # noqa: BLE001
        _log.debug("repo map: code_indexer unavailable: %s", exc)
        return ""
    skip = set(SKIP_DIRS) | {".git"}

    def _skipped(path: Path) -> bool:
        return any(part in skip or part.startswith(".")
                   for part in path.relative_to(root).parts[:-1])

    top_dirs = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and d.name not in skip and not d.name.startswith("."))
    lines = [f"repo map ({root.name}):",
             "top-level dirs: " + (", ".join(top_dirs) or "(none)")]
    shown = 0
    others = 0
    candidates = sorted(
        (p for p in root.rglob("*")
         if p.is_file() and not _skipped(p)
         and p.stat().st_size <= _MAX_FILE_BYTES),
        key=lambda p: p.relative_to(root).as_posix())
    for path in candidates:
        rel = path.relative_to(root).as_posix()
        if path.suffix != ".py":
            others += 1
            continue
        if shown >= _MAX_FILES:
            others += 1
            continue
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        purpose = _module_purpose(code)
        symbols = _parse_units(code, rel)
        bit = f"{rel} — \"{purpose}\"" if purpose else rel
        lines.append(f"{bit} ({symbols} symbols)")
        shown += 1
    if others:
        lines.append(f"… +{others} more files not shown")
    text = "\n".join(lines)
    if len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        text = (text[:cut] if cut > 0 else text[:max_chars]) \
            + "\n…(truncated)"
    return text


def build_repo_map(root: str | Path, *,
                   max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Compact repo map, bounded to ``max_chars``.  Returns "" when the
    indexer or the tree cannot be used — the caller proceeds without it."""
    try:
        return _build(Path(root).expanduser().resolve(), max_chars)
    except Exception as exc:  # noqa: BLE001 — never break the mission
        _log.debug("repo map failed: %s", exc)
        return ""
