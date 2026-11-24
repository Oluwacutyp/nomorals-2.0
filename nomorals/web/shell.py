"""Devon's shared HTML page shell — the standard every one of her
HTML pages follows.

The API server's Content-Security-Policy allows no scripts and no
styles at all (``style-src 'none'; script-src 'none'``), so the
design lives in the document itself: one heading hierarchy, tables
with real headers for data, honest empty states, a freshness stamp
on every page, and a footer that says where to manage the thing
(the page never manages anything by itself).  This module is the
standard embodied in code — the signal page uses it first, and the
next page lane adopts it instead of hand-rolling another chrome.

Stdlib only, on purpose: this is a leaf (SHARED_LAYER 0), so any
organ — betdesk, social, media, studio — can import it without a
layering violation, and it can never drag a dependency upward.
"""
from __future__ import annotations

import html
import time
from typing import Any

__all__ = ["esc", "page_shell"]


def esc(value: Any) -> str:
    """Escape one value for HTML text.  Nothing from a store, a
    provider name, or a user ever reaches a page unescaped."""
    return html.escape(str(value if value is not None else ""))


def page_shell(*, title: str, heading: str, body: str,
               updated_at: float | None = None,
               manage_line: str = "") -> str:
    """Wrap ``body`` (already-escaped HTML from the page's own
    renderer) in Devon's page chrome.

    ``title`` is the ``<title>``; ``heading`` is the page's ``<h1>``.
    ``updated_at`` stamps the footer with the moment the data was
    read — every page says how fresh it is, and none pretends live
    data is live when it is a snapshot.  ``manage_line`` names the
    chat surface that manages the thing (the page itself never
    does).  No styles, no scripts: the server's CSP forbids both,
    so the design is structure, order, and honest words.
    """
    stamped = time.strftime(
        "%Y-%m-%d %H:%M:%S UTC",
        time.gmtime(updated_at if updated_at is not None
                    else time.time()))
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,"
        "initial-scale=1'>"
        "<title>" + esc(title) + "</title></head><body>",
        "<header><p><b>Devon</b></p>"
        "<h1>" + esc(heading) + "</h1></header>",
        "<main>" + body + "</main>",
        "<footer><hr><p>Generated " + esc(stamped) + " — this page "
        "is a snapshot of live state; reload it for the latest.</p>",
    ]
    if manage_line:
        parts.append("<p>" + esc(manage_line) + "</p>")
    parts.append("</footer></body></html>")
    return "".join(parts)
