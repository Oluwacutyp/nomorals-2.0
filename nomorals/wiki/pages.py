"""Wiki pages: the page model, slugs, and frontmatter.

craft-wiki's page shape, re-implemented in Devon: a page is a slug
plus frontmatter (title, description/summary, type, category,
parent, order, tags) plus a markdown body. Slugs are stable
identifiers — a page keeps its slug when its title changes, so
``[[links]]`` never rot on a rename.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PAGE_TYPES", "Page", "emit_frontmatter", "parse_frontmatter",
           "slugify"]

#: Page types, mined from craft-wiki's ``pageTypes`` (overview / guide /
#: reference / troubleshooting) and extended with the kinds Devon's own
#: knowledge needs (howto / decision / runbook).
PAGE_TYPES = frozenset({
    "overview", "guide", "reference", "troubleshooting",
    "howto", "decision", "runbook",
})

_SLUG_BAD = re.compile(r"[^a-z0-9]+")
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def slugify(title: str) -> str:
    """``GoldenSwing Trading Desk`` → ``goldenswing-trading-desk``."""
    slug = _SLUG_BAD.sub("-", str(title or "").strip().lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if not slug:
        raise ValueError("wiki: a title must contain a letter or digit")
    return slug


@dataclass
class Page:
    """One wiki page, exactly as the store holds it."""

    slug: str
    title: str
    body: str
    type: str = "guide"
    category: str = ""
    parent: str = ""
    order: int | None = None
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    revisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "body": self.body,
            "type": self.type,
            "category": self.category,
            "parent": self.parent,
            "order": self.order,
            "summary": self.summary,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revisions": [dict(r) for r in self.revisions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Page:
        return cls(
            slug=str(data.get("slug", "")),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            type=str(data.get("type", "guide")),
            category=str(data.get("category", "")),
            parent=str(data.get("parent", "")),
            order=data.get("order"),
            summary=str(data.get("summary", "")),
            tags=[str(t) for t in (data.get("tags") or [])],
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            revisions=[dict(r) for r in (data.get("revisions") or [])],
        )


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``--- key: value ---`` frontmatter off a markdown body.

    A tiny scalar parser on purpose — no YAML dependency, and the
    values a wiki page needs are plain scalars and comma lists. An
    unknown ``type`` is an error, never silently stored.
    """
    match = _FRONTMATTER.match(text or "")
    if not match:
        return {}, text
    fields: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip("\"'")
        if key == "tags":
            fields["tags"] = [t.strip() for t in value.split(",") if t.strip()]
        elif key == "order":
            try:
                fields["order"] = int(value)
            except ValueError:
                fields["order"] = None
        else:
            fields[key] = value
    type_ = str(fields.get("type", "") or "")
    if type_ and type_ not in PAGE_TYPES:
        raise ValueError(
            f"wiki: unknown page type {type_!r} — one of "
            + ", ".join(sorted(PAGE_TYPES)))
    # The blank line separating the frontmatter from the body is a
    # separator, not body content.
    return fields, text[match.end():].lstrip("\n")


def emit_frontmatter(page: Page) -> str:
    """Render a page's metadata as a ``---`` frontmatter block."""
    lines = ["---", f"title: {page.title}"]
    if page.type:
        lines.append(f"type: {page.type}")
    if page.category:
        lines.append(f"category: {page.category}")
    if page.parent:
        lines.append(f"parent: {page.parent}")
    if page.order is not None:
        lines.append(f"order: {page.order}")
    if page.summary:
        lines.append(f"summary: {page.summary}")
    if page.tags:
        lines.append("tags: " + ", ".join(page.tags))
    lines.append("---")
    return "\n".join(lines) + "\n\n"
