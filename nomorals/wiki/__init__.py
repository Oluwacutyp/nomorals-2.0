"""Devon's wiki — a durable knowledge surface.

Mined from storytold/craft-wiki (pattern only): craft-wiki carries
NO licence, so nothing here copies its text or files. What was mined
is the shape — a content tree of markdown pages with frontmatter
(title, description, order, parent, type, category, summary), a
per-topic template set (overview / getting-started / workflows /
troubleshooting), a sidebar that sorts by order then label with
troubleshooting last, and absolute links between pages. This organ is
Devon's own original implementation of those patterns, superseded
into a LIVE surface: pages are created, edited, searched, linked and
revised from chat and from the brain, with full revision history, a
broken-link report that names the link, and exports onto the board.

Lane 126 (census §3B.3) mined the repo again and added the patterns
the first port left out: ``subcategory`` and ``cover`` frontmatter,
parent validation (a parent must name an existing top-level page),
the nested sidebar tree with category/subcategory groups and landing
pages, prev/next pagination with frontmatter overrides, the canonical
five pages per topic, and a sitemap.

Attribution: patterns mined from https://github.com/storytold/craft-wiki
(★21, unlicensed — docs/pattern only, never ported as code).
"""

from __future__ import annotations

from .board import export_to_board
from .links import backlinks_for, broken_links, orphan_pages, resolve_link
from .nav import (
    CANONICAL_PAGES,
    canonical_missing,
    dangling_parents,
    flatten_tree,
    neighbors,
    sidebar_tree,
    sitemap,
)
from .pages import PAGE_TYPES, Page, slugify
from .render import render_chat, render_html
from .search import search_pages
from .store import WikiStore, wiki_path
from .surface import (
    parse_wiki_question,
    render_board,
    render_broken,
    render_canonical,
    render_history,
    render_orphans,
    render_page_list,
    render_search_results,
    render_sitemap,
)
from .templates import (
    TEMPLATE_DESCRIPTIONS,
    canonical_body,
    template_body,
)

__all__ = [
    "CANONICAL_PAGES",
    "PAGE_TYPES",
    "TEMPLATE_DESCRIPTIONS",
    "Page",
    "WikiStore",
    "backlinks_for",
    "broken_links",
    "canonical_body",
    "canonical_missing",
    "dangling_parents",
    "export_to_board",
    "flatten_tree",
    "neighbors",
    "orphan_pages",
    "parse_wiki_question",
    "render_board",
    "render_broken",
    "render_canonical",
    "render_chat",
    "render_history",
    "render_html",
    "render_orphans",
    "render_page_list",
    "render_search_results",
    "render_sitemap",
    "resolve_link",
    "search_pages",
    "sidebar_tree",
    "sitemap",
    "slugify",
    "template_body",
    "wiki_path",
]
