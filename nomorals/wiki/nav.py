"""Wiki navigation: the sidebar tree, prev/next, sitemap, canonical set.

craft-wiki's navigation, re-implemented in Devon:

* Sidebar tree (``appSidebar.mjs`` ``groupChildren``): a page's children
  group under it — children without a category first, then one group per
  category (ordered by the group's lowest sidebar order, then label —
  craft-wiki's fallback rule for categories outside its fixed list),
  each with the category's landing page (its overview) before the group's
  pages; inside a category, pages without a subcategory come first, then
  one group per subcategory, and when every page of a category has the
  same subcategory the category lists the pages without a subcategory
  group at all.
* Prev/next (``routeData.mjs``): a page's neighbors in the sidebar order
  — each page followed by the pages listed under it, a category's landing
  page before the category's pages. A page that sets ``prev``/``next`` in
  frontmatter keeps its own link when it names a real page.
* Canonical five (``apps.mjs`` ``guides``): every topic has ``overview``,
  ``getting-started``, ``install-update``, ``workflows``,
  ``troubleshooting``, in that order.
* Sitemap: the build emits one; Devon's lists every page as
  ``/wiki/<slug>``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "canonical_missing",
    "dangling_parents",
    "flatten_tree",
    "neighbors",
    "sidebar_tree",
    "sitemap",
]

#: The five pages every topic has, in sidebar order.
CANONICAL_PAGES: tuple[str, ...] = (
    "overview", "getting-started", "install-update", "workflows",
    "troubleshooting",
)


def _order(page: Any) -> float:
    return float(page.order) if page.order is not None else float("inf")


def _sort(pages: list[Any]) -> list[Any]:
    """Sidebar order inside a group: troubleshooting last, then order,
    then title — the same rule the store's flat sort uses."""
    return sorted(
        pages,
        key=lambda p: (p.type == "troubleshooting", _order(p),
                       p.title.lower()))


def _group_children(children: list[Any]) -> list[dict[str, Any]]:
    """Group one page's children the way the sidebar shows them.

    Each item is ``{"page": page}`` or ``{"label": category,
    "landing": page | None, "items": [...]}``; inside a category the
    items are plain pages first, then one group per subcategory.
    """
    items: list[dict[str, Any]] = [
        {"page": page} for page in _sort([p for p in children
                                          if not p.category])]
    by_category: dict[str, list[Any]] = {}
    for page in children:
        if page.category:
            by_category.setdefault(page.category, []).append(page)
    groups = sorted(
        by_category.items(),
        key=lambda kv: (min(_order(p) for p in kv[1]), kv[0].lower()))
    for label, members in groups:
        landing = next(
            (p for p in members if p.type == "overview"), None)
        rest = [p for p in members if p is not landing]
        plain = [p for p in rest if not p.subcategory]
        by_sub: dict[str, list[Any]] = {}
        for page in rest:
            if page.subcategory:
                by_sub.setdefault(page.subcategory, []).append(page)
        subs = sorted(
            by_sub.items(),
            key=lambda kv: (min(_order(p) for p in kv[1]), kv[0].lower()))
        if not plain and len(subs) == 1:
            inner: list[dict[str, Any]] = [
                {"page": p} for p in _sort(subs[0][1])]
        else:
            inner = [{"page": p} for p in _sort(plain)]
            for name, pages in subs:
                inner.append({"label": name,
                              "items": [{"page": p}
                                        for p in _sort(pages)]})
        items.append({"label": label, "landing": landing, "items": inner})
    return items


def sidebar_tree(store: Any) -> list[dict[str, Any]]:
    """The wiki as a nested sidebar: top-level pages in sidebar order.

    Each entry is ``{"page": page, "items": [...]}`` where ``items``
    is the grouped children from :func:`_group_children`. A child
    whose parent names no page is listed at the top level instead —
    listed by :func:`dangling_parents`, never silently dropped.
    """
    pages = store.list_pages()
    by_slug = {p.slug: p for p in pages}
    dangling = {p.slug for p in pages
                if p.parent and p.parent not in by_slug}
    # Top level keeps the store's sidebar order (category, landing
    # first, order, title); children are grouped below by _group_children.
    top = [p for p in pages if not p.parent or p.slug in dangling]
    children: dict[str, list[Any]] = {}
    for page in pages:
        if page.parent and page.parent in by_slug:
            children.setdefault(page.parent, []).append(page)
    tree = []
    for page in top:
        tree.append({"page": page,
                     "items": _group_children(children.get(page.slug, []))})
    return tree


def flatten_tree(tree: list[dict[str, Any]]) -> list[Any]:
    """The sidebar's pages in reading order: a page, then the pages
    listed under it — a group's landing page before the group's pages."""
    out: list[Any] = []

    def _walk(items: list[dict[str, Any]]) -> None:
        for item in items:
            if "page" in item:
                out.append(item["page"])
            else:
                if item.get("landing") is not None:
                    out.append(item["landing"])
                _walk(item["items"])

    for entry in tree:
        out.append(entry["page"])
        _walk(entry["items"])
    return out


def neighbors(store: Any, slug: str) -> tuple[Any | None, Any | None]:
    """(previous, next) page for ``slug`` in sidebar order.

    An explicit ``prev``/``next`` frontmatter value wins when it names
    a real page; a dangling override is ignored and the sidebar
    neighbor stands. The first page has no previous, the last no next.
    """
    page = store.get(slug)
    if page is None:
        return None, None
    ordered = flatten_tree(sidebar_tree(store))
    index = next((i for i, p in enumerate(ordered)
                  if p.slug == page.slug), None)
    if index is None:
        return None, None
    prev = store.get(page.prev) if page.prev else None
    if prev is None and index > 0:
        prev = ordered[index - 1]
    nxt = store.get(page.next) if page.next else None
    if nxt is None and index < len(ordered) - 1:
        nxt = ordered[index + 1]
    return prev, nxt


def sitemap(store: Any) -> list[str]:
    """Every page as a ``/wiki/<slug>`` URL, in sidebar order."""
    return [f"/wiki/{page.slug}"
            for page in flatten_tree(sidebar_tree(store))]


def canonical_missing(store: Any, category: str = "") -> list[str]:
    """Which of the five canonical pages a category lacks, in order.

    The canonical slugs are matched inside the category (or across the
    whole wiki when no category is given); a page counts by its slug,
    so a renamed title still counts.
    """
    pages = store.list_pages(category) if category else store.list_pages()
    have = {p.slug for p in pages}
    return [slug for slug in CANONICAL_PAGES if slug not in have]


def dangling_parents(store: Any) -> list[dict[str, str]]:
    """Pages whose ``parent`` names no page — source slug + target."""
    pages = store.list_pages()
    by_slug = {p.slug for p in pages}
    return [{"from": p.slug, "parent": p.parent}
            for p in pages if p.parent and p.parent not in by_slug]
