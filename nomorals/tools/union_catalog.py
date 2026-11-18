"""Union-catalogue tool: the Anna's Archive rung for the brain.

"Find me <title>, any format" over the union catalogue — not a
standalone search. The organ (:mod:`nomorals.books.union_catalog`)
owns the merge/reconciliation pass, the provenance, the store and the
live legs; this module is only the surface between the brain and the
organ.

The answer names which sources hold the work and which are alive
right now. A dead source's holding is still listed — degraded in
ranking, labelled unavailable — never silently dropped. The catalogue
indexes metadata and lawful holdings only: the owner's own library
first, then Open Library, Internet Archive and Project Gutenberg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..books.union_catalog import (
    LAWFUL_SOURCES,
    UnionCatalog,
    format_catalog_reply,
    format_work,
    parse_catalog_request,
    register_source,
)
from ..core.policy import Capability

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _catalog_of(context: Any) -> UnionCatalog:
    home = _home_of(context)
    library = None
    if context is not None:
        try:
            from ..books.library import Library

            library = Library(context)
        except Exception:  # noqa: BLE001 — a broken library leg degrades
            library = None
    return UnionCatalog(home, library=library)


def _work_payload(work: Any) -> dict[str, Any]:
    return work.to_dict()


def register(registry: Any) -> None:
    """Attach the ``union_catalog`` tool."""
    context = registry.context

    @registry.register(
        "union_catalog",
        description=(
            "Union catalogue over many libraries: one record per work, "
            "merged from many sources with provenance. action=find takes "
            "query=<title> and searches your library first, then the "
            "catalogue, then the live open collections (Open Library, "
            "Internet Archive, Project Gutenberg), answering which "
            "sources hold it and which are alive right now. action=show "
            "takes id=<canonical id> and returns one work with its "
            "holdings. action=list lists catalogued works. "
            "action=refresh re-checks every lawful live source. "
            "action=export takes path=<file.jsonl> and writes the bulk "
            "export (one WorkRecord per line, line-delimited JSON). "
            "action=unmerge takes id=<canonical id> and "
            "merged=<merged id> and reverses one merge, restoring the "
            "folded-away record whole. action=status reports catalogue "
            "size and per-source health. action=add_source takes "
            "id=<source id>, label=<label>, licence=<licence> and "
            "legal_basis=<basis> and registers a new lawful source — "
            "a source without a legal basis is refused."
        ),
        capability=Capability.NET_OUT,
        result_protocol="dict_ok",
    )
    def _union_catalog_tool(**kwargs: Any) -> dict[str, Any]:
        action = str(kwargs.get("action", "find") or "find").lower()
        catalog = _catalog_of(context)
        if action == "find":
            query = str(kwargs.get("query", "") or "").strip()
            if not query:
                return {"ok": False, "error": "find needs query=<title>"}
            live = str(kwargs.get("live", "true") or "true").lower() not in (
                "false", "0", "no", "off")
            limit = int(kwargs.get("limit", 8) or 8)
            works = catalog.find(query, live=live, limit=limit)
            return {"ok": True,
                    "query": query,
                    "works": [_work_payload(w) for w in works],
                    "reply": format_catalog_reply(query, works)}
        if action == "show":
            work = catalog.get(str(kwargs.get("id", "") or ""))
            if work is None:
                return {"ok": False,
                        "error": "no catalogued work matches that id"}
            return {"ok": True, "work": _work_payload(work),
                    "reply": format_work(work)}
        if action == "list":
            limit = int(kwargs.get("limit", 20) or 20)
            works = catalog.list_works(limit=limit)
            return {"ok": True,
                    "works": [_work_payload(w) for w in works]}
        if action == "refresh":
            results = catalog.refresh_health()
            return {"ok": True, "sources": results}
        if action == "export":
            path = str(kwargs.get("path", "") or "").strip()
            if not path:
                home = _home_of(context)
                base = (Path(home) if home else Path.home() / ".nomorals")
                path = str(base / "data" / "union_catalog.jsonl")
            receipt = catalog.export_aac(path)
            return {"ok": True, **receipt}
        if action == "unmerge":
            canonical = str(kwargs.get("id", "") or "")
            merged = str(kwargs.get("merged", "") or "")
            if not canonical or not merged:
                return {"ok": False,
                        "error": "unmerge needs id=<canonical id> and "
                                 "merged=<merged id>"}
            if not catalog.unmerge(canonical, merged):
                return {"ok": False,
                        "error": "no such merge — check both ids"}
            return {"ok": True, "unmerged": merged,
                    "canonical": canonical}
        if action == "status":
            return {"ok": True, **catalog.status()}
        if action == "add_source":
            source_id = str(kwargs.get("id", "") or "").strip()
            try:
                row = register_source(
                    source_id,
                    label=str(kwargs.get("label", "") or ""),
                    licence=str(kwargs.get("licence", "") or ""),
                    legal_basis=str(kwargs.get("legal_basis", "") or ""),
                    health_url=str(kwargs.get("health_url", "") or ""))
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "source": source_id, "row": row,
                    "sources": sorted(LAWFUL_SOURCES)}
        return {"ok": False,
                "error": ("unknown action — find | show | list | refresh | "
                          "export | unmerge | status | add_source")}


def parse_request(text: str) -> dict[str, str] | None:
    """Re-export for the runtime hook (same parser, one home)."""
    return parse_catalog_request(text)
