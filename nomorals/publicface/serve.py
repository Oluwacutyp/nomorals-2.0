"""Serving for the public face — the read side of the gate.

:mod:`nomorals.publicface.publish` is the gate: it decides what may
go public and hands back a ``/p/<slug>`` descriptor. This module is
what answers that path: it resolves a slug against the durable
:class:`PublicPageStore` and renders the approved page as plain HTML.

Rules (the gate's, enforced again here — the route never trusts its
own caller):

* ONLY public pages serve. A draft, an unpublished page, an unknown
  slug and a removed page all answer "missing", identically — no
  oracle for guessing slugs.
* ``/p/show-<id>`` needs no special case: the shows composer
  publishes episodes with slug ``show-<episode.id>``, so it resolves
  like any other slug.
* The HTML carries no scripts and no styles (the API's CSP allows
  nothing at all) and every title/content byte is escaped.
"""

from __future__ import annotations

import html
from typing import Any

from .publish import PublicPage, PublicPageStore

__all__ = [
    "render_public_page",
    "resolve_public_page",
    "serve_public_page",
]


def resolve_public_page(
    store: PublicPageStore, ref: str,
) -> PublicPage | None:
    """The public page for a slug, else None.

    ``ref`` is the path segment after ``/p/`` — a slug, ``show-<id>``
    included. Only a page whose status is ``public`` resolves; the
    first public page with that slug wins when slugs collide.
    """
    ref = (ref or "").strip()
    if not ref or "/" in ref:
        return None
    for page in store.pages.values():
        if page.slug == ref and page.status == "public":
            return page
    return None


def render_public_page(page: PublicPage) -> str:
    """The page as a plain HTML document.

    Title + pack + content, every byte escaped, paragraphs split on
    blank lines. No scripts, no styles, no attributes of any kind —
    the API's Content-Security-Policy allows nothing at all.
    """
    title = html.escape(page.title)
    pack = html.escape(page.pack)
    created = html.escape(page.created.isoformat())
    blocks = []
    for chunk in page.content.split("\n\n"):
        text = chunk.strip()
        if not text:
            continue
        blocks.append(f"<p>{html.escape(text).replace(chr(10), '<br>')}</p>")
    body = "\n".join(blocks)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n"
        "</head>\n"
        "<body>\n"
        "<main>\n"
        f"<h1>{title}</h1>\n"
        f"<p>Pack: {pack} · Published {created}</p>\n"
        f"{body}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )


def serve_public_page(
    store: PublicPageStore, ref: str,
) -> tuple[str, dict[str, Any] | None]:
    """Serve a slug: ("ok", payload) or ("missing", None).

    The ("ok", ...) / ("missing", ...) shape mirrors
    :func:`nomorals.storage.share.serve_share` — the handler answers
    200 for "ok" and 404 for anything else, identically.
    """
    page = resolve_public_page(store, ref)
    if page is None:
        return "missing", None
    return "ok", {
        "html": render_public_page(page),
        "title": page.title,
        "slug": page.slug,
        "pack": page.pack,
        "page_id": page.id,
    }
