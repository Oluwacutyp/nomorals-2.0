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

Attribution: patterns mined from https://github.com/storytold/craft-wiki
(★21, unlicensed — docs/pattern only, never ported as code).
"""

from __future__ import annotations

from .board import export_to_board
from .links import backlinks_for, broken_links, orphan_pages, resolve_link
from .pages import PAGE_TYPES, Page, slugify
from .render import render_chat, render_html
from .search import search_pages
from .store import WikiStore, wiki_path
from .surface import (
    parse_wiki_question,
    render_board,
    render_broken,
    render_history,
    render_orphans,
    render_page_list,
    render_search_results,
)
from .templates import TEMPLATE_DESCRIPTIONS, template_body

__all__ = [
    "PAGE_TYPES",
    "TEMPLATE_DESCRIPTIONS",
    "Page",
    "WikiStore",
    "backlinks_for",
    "broken_links",
    "export_to_board",
    "orphan_pages",
    "parse_wiki_question",
    "render_board",
    "render_broken",
    "render_chat",
    "render_history",
    "render_html",
    "render_orphans",
    "render_page_list",
    "render_search_results",
    "resolve_link",
    "search_pages",
    "slugify",
    "template_body",
    "wiki_path",
]
