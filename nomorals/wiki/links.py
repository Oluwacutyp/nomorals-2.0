"""Wiki links: ``[[Page]]`` extraction, resolution, backlinks,
orphans, and the broken-link report.

craft-wiki links pages with absolute ``/apps/<app>/<page>/`` paths;
Devon's wiki links with ``[[Title]]`` (and ``[[Title|label]]``) plus
``/wiki/<slug>`` markdown links. A link that names no page is
BROKEN and is reported by name — never silently dropped, never
auto-fixed into a different page.
"""

from __future__ import annotations

import re
from typing import Any

from .pages import slugify

__all__ = [
    "backlinks_for",
    "broken_links",
    "extract_wikilinks",
    "orphan_pages",
    "resolve_link",
]

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_MDLINK_RE = re.compile(r"\[[^\]]*\]\(/wiki/([a-z0-9-]+)[^)]*\)")


def extract_wikilinks(body: str) -> list[str]:
    """Every ``[[target]]`` in a body, in order, deduplicated."""
    found: list[str] = []
    for target in _WIKILINK_RE.findall(body or ""):
        target = target.strip()
        if target and target not in found:
            found.append(target)
    return found


def extract_links(body: str) -> list[str]:
    """Every link target in a body: ``[[...]]`` plus ``/wiki/<slug>``."""
    found = extract_wikilinks(body)
    for slug in _MDLINK_RE.findall(body or ""):
        if slug not in found:
            found.append(slug)
    return found


def resolve_link(store: Any, target: str) -> str | None:
    """A link target → a page slug, by slug first, then by title.

    ``[[goldenswing]]`` and ``[[GoldenSwing]]`` resolve to the same
    page; a target naming no page resolves to ``None`` (broken).
    """
    target = str(target or "").strip()
    if not target:
        return None
    try:
        slug = slugify(target)
    except ValueError:
        return None
    if store.get(slug) is not None:
        return slug
    page = store.get_by_title(target)
    return page.slug if page is not None else None


def backlinks_for(store: Any, slug: str) -> list[str]:
    """Slugs of the pages that link to ``slug``, sorted by title."""
    page = store.get(slug)
    if page is None:
        return []
    found: list[str] = []
    for other in store.list_pages():
        if other.slug == page.slug:
            continue
        if page.slug in {
                resolve_link(store, t) for t in extract_links(other.body)}:
            found.append(other.slug)
    return sorted(found,
                  key=lambda s: (store.get(s) or page).title.lower())


def orphan_pages(store: Any) -> list[Any]:
    """Pages nothing links to — except overviews, which are roots.

    An overview page is an entry point by design (craft-wiki's
    per-app overview is the landing page); calling it an orphan
    would be wrong. Everything else with no backlinks is listed.
    """
    orphans = []
    for page in store.list_pages():
        if page.type == "overview":
            continue
        if not backlinks_for(store, page.slug):
            orphans.append(page)
    return orphans


def broken_links(store: Any) -> list[dict[str, str]]:
    """Every link whose target names no page, with its source.

    ``{"from": <source slug>, "target": <the text that was linked>}``
    — the report names the broken link, never a count alone.
    """
    broken: list[dict[str, str]] = []
    for page in store.list_pages():
        for target in extract_links(page.body):
            if resolve_link(store, target) is None:
                broken.append({"from": page.slug, "target": target})
    return broken
