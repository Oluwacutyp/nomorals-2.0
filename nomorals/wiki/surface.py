"""Chat renders for the wiki board, lists and reports, plus the
natural-language question parser.

The board is the wiki at a glance (every category, every page, broken
links called out); the reports name names — a broken link is listed
with its source page and its target, an orphan with its slug. The
parser matches only what the wiki actually holds, the way the group
health hook matches only groups Devon serves: a question about a
page that does not exist is not a wiki question.
"""

from __future__ import annotations

import re
from typing import Any

from .links import broken_links, orphan_pages


def build_from_memory(store: Any, memory: Any, query: str,
                      title: str = "") -> tuple[Any, int]:
    """Build a page from memory recall for ``query``.

    Shared by the ``wiki`` brain tool and the ``/wiki`` verb so both
    surfaces cite the same way: one ``## What memory says`` section
    with the top-5 recalled records as bullets, then a Sources line.
    Returns ``(page, sources)``. Raises ``ValueError`` when memory
    holds nothing for the query — a page is never created from an
    empty recall.
    """
    result = memory.recall(query, limit=5)
    texts = list(getattr(result, "texts", []) or [])
    if not texts:
        raise ValueError(f"memory holds nothing for {query!r}")
    page_title = (title or "").strip() or f"Memory: {query}"
    lines = [f"Built from memory recall for {query!r}.", "",
             "## What memory says", ""]
    for text in texts:
        lines.append(f"- {text}")
    lines.append("")
    lines.append("Sources: Devon's memory (recall, top 5).")
    page = store.create(page_title, "\n".join(lines) + "\n",
                        type="reference", category="Memory",
                        summary=f"Memory recall for {query!r}.")
    return page, len(texts)


__all__ = [
    "parse_wiki_question",
    "render_board",
    "render_broken",
    "render_history",
    "render_orphans",
    "render_page_list",
    "render_search_results",
]


def render_board(store: Any) -> str:
    """The wiki board: pages by category, counts, broken links named."""
    pages = store.list_pages()
    if not pages:
        return ("the wiki is empty — /wiki new <type> <title> to start "
                "one, /wiki template <type> <title> for a skeleton.")
    lines = [f"Wiki — {len(pages)} page(s):"]
    for category in store.categories():
        lines.append(f"[{category}]")
        for page in store.list_pages(category):
            lines.append(f"• {page.title} [{page.type}]"
                         f" — /wiki read {page.slug}")
    uncategorised = [p for p in pages if not p.category]
    if uncategorised:
        lines.append("[uncategorised]")
        for page in uncategorised:
            lines.append(f"• {page.title} [{page.type}]"
                         f" — /wiki read {page.slug}")
    broken = broken_links(store)
    if broken:
        lines.append(f"{len(broken)} broken link(s) — /wiki broken "
                      "names them.")
    return "\n".join(lines)


def render_page_list(store: Any, category: str = "") -> str:
    """The page list, optionally inside one category."""
    pages = store.list_pages(category)
    if not pages:
        if category:
            return f"no pages in category {category!r}."
        return "the wiki is empty."
    lines = [f"Wiki pages{(' — ' + category) if category else ''}:"]
    for page in pages:
        lines.append(f"• {page.title} [{page.type}]"
                     f" — /wiki read {page.slug}")
    return "\n".join(lines)


def render_search_results(results: list[Any], query: str) -> str:
    """Search hits, best first, with their type and slug."""
    if not results:
        return f"nothing in the wiki matches {query!r}."
    lines = [f"Wiki — {len(results)} match(es) for {query!r}:"]
    for page in results:
        lines.append(f"• {page.title} [{page.type}]"
                     f" — /wiki read {page.slug}")
    return "\n".join(lines)


def render_broken(store: Any) -> str:
    """The broken-link report: source page + target, named."""
    broken = broken_links(store)
    if not broken:
        return "no broken links in the wiki."
    lines = [f"{len(broken)} broken link(s):"]
    for entry in broken:
        lines.append(f"• {entry['from']} → {entry['target']!r} "
                     "(no such page)")
    return "\n".join(lines)


def render_orphans(store: Any) -> str:
    """Pages nothing links to (overviews excluded — they are roots)."""
    orphans = orphan_pages(store)
    if not orphans:
        return "no orphan pages — every page is linked."
    lines = [f"{len(orphans)} orphan page(s):"]
    for page in orphans:
        lines.append(f"• {page.title} [{page.type}]"
                     f" — /wiki read {page.slug}")
    return "\n".join(lines)


def render_history(page: Any) -> str:
    """A page's revision history, newest last."""
    if not page.revisions:
        return f"{page.title}: no revisions yet — one version only."
    lines = [f"{page.title} — {len(page.revisions)} revision(s):"]
    for entry in page.revisions:
        note = f" — {entry.get('note')}" if entry.get("note") else ""
        lines.append(f"• rev {entry.get('rev')}"
                     f" at {entry.get('at', '')}{note}")
    lines.append(f"current body updated {page.updated_at}")
    return "\n".join(lines)


_READ_RE = re.compile(
    r"(?:what does the wiki say about|what is in the wiki about|"
    r"what'?s in the wiki about|wiki:?)\s+(.+?)\??\s*$",
    re.IGNORECASE)
_SEARCH_RE = re.compile(
    r"(?:search the wiki for|wiki search)\s+(.+?)\s*$", re.IGNORECASE)


def parse_wiki_question(
        text: str,
        known: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Natural phrasing → ``("read", slug)`` or ``("search", query)``.

    ``known`` is ``[(slug, title)]`` for the pages that exist. A read
    matches only a page the wiki holds — the longest title/slug match
    wins — so the question never steals an unrelated one. A search
    needs no known page; it is just a query.
    """
    text = str(text or "").strip()
    match = _SEARCH_RE.search(text)
    if match:
        query = match.group(1).strip()
        return ("search", query) if query else None
    match = _READ_RE.search(text)
    if match is None:
        return None
    asked = match.group(1).strip()
    if not asked:
        return None
    best: tuple[str, str] | None = None
    best_len = 0
    for slug, title in known:
        for candidate in (title, slug):
            if (candidate.lower() in asked.lower()
                    and len(candidate) > best_len):
                best = ("read", slug)
                best_len = len(candidate)
    return best
