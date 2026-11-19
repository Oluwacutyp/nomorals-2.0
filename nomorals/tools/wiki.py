"""Wiki tool: ``wiki`` — the brain's handle on the wiki organ, plus
the natural-language question parser.

One tool, action-dispatched, over :mod:`nomorals.wiki`; the organ
owns the pages, the links, the search and the history — this module
is only the surface.

``action=create`` writes a page (``title=``, ``body=``, ``type=``,
``category=``, ``parent=``, ``order=``, ``summary=``, ``tags=``);
``action=read`` shows one (``slug=``); ``action=edit`` replaces a
body (``slug=``, ``body=``, ``note=``); ``action=retitle`` renames
one (``slug=``, ``title=`` — the slug never moves, so links stay
valid); ``action=list`` lists pages (``category=``);
``action=search`` ranks matches (``query=``); ``action=links`` lists
a page's outgoing links; ``action=backlinks`` lists the pages linking
to one; ``action=orphans`` lists unlinked pages; ``action=broken``
names every broken link; ``action=history`` shows revisions
(``slug=``); ``action=revert`` restores one (``slug=``, ``rev=``);
``action=template`` scaffolds a page from a type skeleton
(``type=``, ``title=``); ``action=import`` takes markdown with
frontmatter (``text=``); ``action=from-memory`` builds a page from
memory recall (``query=``, ``title=``); ``action=board`` puts a page
on the board as chained section nodes (``slug=``);
``action=remove`` deletes one (``slug=``).

The store lives under the process home (``<home>/wiki/wiki.json``),
the same file the ``/wiki`` verb reads — one wiki, two surfaces.
"""

from __future__ import annotations

from typing import Any

from ..wiki import (
    backlinks_for,
    broken_links,
    export_to_board,
    orphan_pages,
    parse_wiki_question,
    render_broken,
    render_chat,
    render_history,
    render_orphans,
    render_page_list,
    render_search_results,
    resolve_link,
    search_pages,
    template_body,
)
from ..wiki.pages import PAGE_TYPES
from ..wiki.store import WikiStore, wiki_path

__all__ = ["parse_wiki_question", "register"]


def _store(context: Any) -> WikiStore:
    settings = getattr(context, "settings", None)
    return WikiStore(wiki_path(settings))


def _memory(context: Any) -> Any:
    return getattr(context, "memory", None)


def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _page_dict(page: Any) -> dict[str, Any]:
    return {
        "slug": page.slug,
        "title": page.title,
        "type": page.type,
        "category": page.category,
        "parent": page.parent,
        "order": page.order,
        "summary": page.summary,
        "tags": list(page.tags),
        "body": page.body,
        "updated_at": page.updated_at,
        "revisions": len(page.revisions),
    }


def _split_tags(raw: str) -> list[str]:
    return [t.strip() for t in str(raw or "").split(",") if t.strip()]


def register(registry: Any) -> None:
    """Attach the ``wiki`` tool."""
    context = registry.context

    @registry.register(
        "wiki",
        description=(
            "Devon's wiki — durable knowledge pages (craft-wiki "
            "patterns, original Devon code): create, read, edit, "
            "search, link, revise and export pages. action=create "
            "writes one (title=, body=, type= overview|guide|reference|"
            "troubleshooting|howto|decision|runbook, category=, "
            "parent=, order=, summary=, tags=); action=read shows one "
            "(slug=); action=edit replaces a body (slug=, body=, "
            "note=); action=retitle renames one (slug=, title=); "
            "action=list lists pages (category=); action=search ranks "
            "matches (query=); action=links / action=backlinks show a "
            "page's links (slug=); action=orphans lists unlinked pages; "
            "action=broken names every broken link; action=history "
            "shows revisions (slug=); action=revert restores one "
            "(slug=, rev=); action=template scaffolds a page (type=, "
            "title=); action=import takes markdown with frontmatter "
            "(text=); action=from-memory builds a page from memory "
            "recall (query=, title=); action=board puts a page on the "
            "board (slug=); action=remove deletes one (slug=). Pages "
            "link with [[Title]]; a link naming no page is reported, "
            "never silently dropped."
        ),
        capability="owner",
    )
    def wiki_tool(
        action: str = "list",
        slug: str = "",
        title: str = "",
        body: str = "",
        query: str = "",
        text: str = "",
        type: str = "",
        category: str = "",
        parent: str = "",
        order: int | None = None,
        summary: str = "",
        tags: str = "",
        note: str = "",
        rev: int = 0,
    ) -> dict[str, Any]:
        store = _store(context)
        action = (action or "list").strip().lower()

        if action == "create":
            if not (title or "").strip():
                return _err("give a title: action=create title=<title> body=<...>")
            if not (body or "").strip():
                return _err("give a body: action=create title=<title> body=<...>")
            try:
                page = store.create(
                    title.strip(), body,
                    type=(type or "guide").strip().lower(),
                    category=category, parent=parent, order=order,
                    summary=summary, tags=_split_tags(tags))
            except ValueError as exc:
                return _err(str(exc))
            return {"ok": True, "slug": page.slug, "page": _page_dict(page)}

        if action == "read":
            slug = (slug or "").strip().lower()
            page = store.get(slug) if slug else None
            if page is None:
                return _err(f"no page {slug!r} — action=list shows what exists")
            return {"ok": True, "page": _page_dict(page),
                    "render": render_chat(store, page)}

        if action == "edit":
            slug = (slug or "").strip().lower()
            if not store.update(slug, body, note=note):
                return _err(f"no page {slug!r}, or the body was empty")
            page = store.get(slug)
            assert page is not None
            return {"ok": True, "slug": slug, "page": _page_dict(page)}

        if action == "retitle":
            slug = (slug or "").strip().lower()
            if not store.retitle(slug, title):
                return _err(f"no page {slug!r}, or the title was empty")
            page = store.get(slug)
            assert page is not None
            return {"ok": True, "slug": slug, "page": _page_dict(page)}

        if action == "list":
            pages = store.list_pages(category)
            return {"ok": True, "render": render_page_list(store, category),
                    "pages": [_page_dict(p) for p in pages]}

        if action == "search":
            if not (query or "").strip():
                return _err("give a query: action=search query=<words>")
            results = search_pages(store, query)
            return {"ok": True,
                    "render": render_search_results(results, query),
                    "results": [_page_dict(p) for p in results]}

        if action == "links":
            slug = (slug or "").strip().lower()
            page = store.get(slug) if slug else None
            if page is None:
                return _err(f"no page {slug!r}")
            from ..wiki.links import extract_links

            out = []
            for target in extract_links(page.body):
                resolved = resolve_link(store, target)
                out.append({"target": target, "slug": resolved,
                            "broken": resolved is None})
            return {"ok": True, "slug": slug, "links": out}

        if action == "backlinks":
            slug = (slug or "").strip().lower()
            page = store.get(slug) if slug else None
            if page is None:
                return _err(f"no page {slug!r}")
            backs = backlinks_for(store, slug)
            return {"ok": True, "slug": slug, "backlinks": backs}

        if action == "orphans":
            pages = orphan_pages(store)
            return {"ok": True, "render": render_orphans(store),
                    "pages": [_page_dict(p) for p in pages]}

        if action == "broken":
            broken = broken_links(store)
            return {"ok": True, "render": render_broken(store),
                    "broken": broken}

        if action == "history":
            slug = (slug or "").strip().lower()
            page = store.get(slug) if slug else None
            if page is None:
                return _err(f"no page {slug!r}")
            return {"ok": True, "slug": slug,
                    "render": render_history(page),
                    "revisions": [dict(r) for r in page.revisions]}

        if action == "revert":
            slug = (slug or "").strip().lower()
            if not store.revert(slug, int(rev or 0)):
                return _err(f"no page {slug!r}, or no revision {rev}")
            page = store.get(slug)
            assert page is not None
            return {"ok": True, "slug": slug, "page": _page_dict(page)}

        if action == "template":
            type_ = (type or "").strip().lower()
            if type_ not in PAGE_TYPES:
                return _err("give a type: action=template type=<"
                            + "|".join(sorted(PAGE_TYPES)) + "> title=<...>")
            if not (title or "").strip():
                return _err("give a title: action=template type=<type> title=<...>")
            try:
                page = store.create(title.strip(), template_body(type_, title),
                                    type=type_, category=category)
            except ValueError as exc:
                return _err(str(exc))
            return {"ok": True, "slug": page.slug, "page": _page_dict(page)}

        if action == "import":
            if not (text or "").strip():
                return _err("give markdown: action=import text=<markdown with frontmatter>")
            try:
                page = store.import_markdown(text)
            except ValueError as exc:
                return _err(str(exc))
            return {"ok": True, "slug": page.slug, "page": _page_dict(page)}

        if action == "from-memory":
            memory = _memory(context)
            if memory is None:
                return _err("memory is off in this session")
            if not (query or "").strip():
                return _err("give a query: action=from-memory query=<words> [title=]")
            from ..wiki.surface import build_from_memory

            try:
                page, sources = build_from_memory(store, memory, query, title)
            except ValueError as exc:
                return _err(str(exc))
            except Exception as exc:  # noqa: BLE001 — recall must never kill the tool
                return _err(f"memory recall failed: {exc}")
            return {"ok": True, "slug": page.slug, "page": _page_dict(page),
                    "sources": sources}

        if action == "board":
            slug = (slug or "").strip().lower()
            page = store.get(slug) if slug else None
            if page is None:
                return _err(f"no page {slug!r}")
            from ..media.edit_engine.board import BoardStore

            doc = export_to_board(store, slug)
            home = getattr(getattr(context, "settings", None),
                           "home_path", None)
            boards = BoardStore(home=home) if home else BoardStore()
            path = boards.save(doc)
            return {"ok": True, "slug": slug, "board_id": doc.id,
                    "board": path}

        if action == "remove":
            slug = (slug or "").strip().lower()
            if not store.delete(slug):
                return _err(f"no page {slug!r}")
            return {"ok": True, "slug": slug, "removed": True}

        return _err(f"unknown action {action!r} — list, read, create, "
                    "edit, search, links, backlinks, orphans, broken, "
                    "history, revert, template, import, from-memory, "
                    "board, remove")
