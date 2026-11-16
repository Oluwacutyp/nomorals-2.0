"""Survey the real code before planning (L5).

The survey is a thin function over the existing code graph
(:mod:`nomorals.tools.code_graph`): keywords from the task are looked
up with ``search`` / ``where_defined``, and the top symbols' callers
with ``who_calls``.  The result feeds
:meth:`DelegationManager.plan_and_spawn` as its ``context_hint``.

There is no survey fallback: when the graph cannot be built or
queried, the survey reports ``unavailable`` and the caller must fail
loudly — a plan built without looking at the code is the failure mode
/plan exists to prevent.  Zero matches is different from failure: a
task about a brand-new area legitimately matches nothing, and the
survey says so in the files list (empty) instead of erroring.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = ["survey_task", "survey_text", "default_repo_root"]

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "into", "from", "that", "this",
    "what", "when", "where", "which", "your", "you", "are", "our",
    "her", "his", "she", "how", "why", "not", "but", "all", "any",
    "can", "could", "should", "would", "will", "make", "take",
})

_MAX_KEYWORDS = 8
_MAX_FILES = 12
_MAX_SYMBOLS = 8


def default_repo_root() -> Path:
    """The repo root holding this file (``nomorals/agents/plan/``)."""
    return Path(__file__).resolve().parents[3]


def _keywords(task: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(task or "").lower())
    out: list[str] = []
    for word in words:
        if word not in _STOPWORDS and word not in out:
            out.append(word)
        if len(out) >= _MAX_KEYWORDS:
            break
    return out


def _file_of(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("file") or row.get("path") or "")
    return ""


def _name_of(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("name") or "")
    return ""


def survey_task(task: str, *, graph: Any = None,
               repo_root: str | Path | None = None) -> dict[str, Any]:
    """Survey the code graph for the task's owning modules.

    ``graph`` is an injected graph (tests, or a prebuilt one); when
    it is None a live :class:`CodeGraph` is built over ``repo_root``
    (default: this repo).  Returns ``{"keywords", "files",
    "symbols", "unavailable"}`` — ``files`` is a ranked list of
    ``{"path", "hits"}``; ``unavailable`` is "" on success.
    """
    keywords = _keywords(task)
    live = graph
    if live is None:
        try:
            from ...tools.code_graph import CodeGraph

            live = CodeGraph(repo_root or default_repo_root())
            live.build()
        except Exception as exc:  # noqa: BLE001 — survey failure is data
            return {"keywords": keywords, "files": [], "symbols": [],
                    "unavailable": f"code graph unavailable: {exc}"}
    files: dict[str, int] = {}
    symbols: list[dict[str, Any]] = []
    try:
        for keyword in keywords:
            for row in live.search(keyword, limit=5) or []:
                path = _file_of(row)
                if path:
                    files[path] = files.get(path, 0) + 1
            for row in live.where_defined(keyword, limit=5) or []:
                path = _file_of(row)
                if path:
                    files[path] = files.get(path, 0) + 1
                name = _name_of(row)
                if name and len(symbols) < _MAX_SYMBOLS:
                    symbols.append({
                        "name": name, "file": path,
                        "kind": str(row.get("kind") or "") if isinstance(row, dict) else "",
                    })
        for symbol in symbols[:3]:
            try:
                callers = live.who_calls(symbol["name"], limit=5) or []
            except Exception:  # noqa: BLE001 — callers are a bonus
                callers = []
            symbol["callers"] = [
                {"name": _name_of(row), "file": _file_of(row)}
                for row in callers if _file_of(row)]
    except Exception as exc:  # noqa: BLE001 — survey failure is data
        return {"keywords": keywords, "files": [], "symbols": [],
                "unavailable": f"code graph query failed: {exc}"}
    ranked = sorted(files.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "keywords": keywords,
        "files": [{"path": path, "hits": hits}
                  for path, hits in ranked[:_MAX_FILES]],
        "symbols": symbols,
        "unavailable": "",
    }


def survey_text(survey: dict[str, Any]) -> str:
    """The survey as a planner ``context_hint`` (one compact block)."""
    keywords = survey.get("keywords") or []
    files = survey.get("files") or []
    symbols = survey.get("symbols") or []
    lines = ["Code survey before planning:"]
    if keywords:
        lines.append("keywords: " + ", ".join(str(k) for k in keywords))
    if files:
        lines.append("owning files: " + ", ".join(
            f"{f.get('path')} ({f.get('hits')} hits)" for f in files))
    else:
        lines.append("owning files: none found — treat as greenfield "
                     "unless the task names a module.")
    for symbol in symbols[:5]:
        callers = symbol.get("callers") or []
        caller_text = ("; called by " + ", ".join(
            c.get("file") for c in callers if c.get("file"))
            if callers else "")
        lines.append(f"symbol: {symbol.get('name')} "
                     f"({symbol.get('file')}){caller_text}")
    return "\n".join(lines)
