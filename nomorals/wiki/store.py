"""The wiki store: one durable JSON file under the process home.

House pattern (triage.json, vacancies.json): a single atomic file,
tmp + ``os.replace`` on write, a corrupt file quarantined aside with
a timestamp instead of silently dropped. Pages carry full revision
history (capped at 50 revisions — history is a record, not a leak).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .pages import PAGE_TYPES, Page, parse_frontmatter, slugify

__all__ = ["WikiStore", "wiki_path"]

_MAX_REVISIONS = 50


def wiki_path(settings: Any = None) -> Path:
    """The wiki store path, honoring the runtime settings home."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home is not None:
            return Path(home) / "wiki" / "wiki.json"
    return Path.home() / ".nomorals" / "wiki" / "wiki.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sort_key(page: Page) -> tuple[Any, ...]:
    """craft-wiki's sidebar rule: order, then label, troubleshooting last.

    Pages group by category (uncategorised last); inside a category
    the troubleshooting pages sit at the end, the rest by order then
    title. A page without an order comes after the ordered ones.
    """
    return (
        page.category == "",
        page.category.lower(),
        page.type == "troubleshooting",
        page.order if page.order is not None else 1 << 30,
        page.title.lower(),
    )


class WikiStore:
    """Durable wiki pages: create / read / update / revert / delete."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else wiki_path()
        self._pages: dict[str, Page] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            # Corrupt store: quarantine it aside, never silently drop
            # the record of what was there.
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            with contextlib.suppress(OSError):
                self.path.rename(
                    self.path.with_suffix(f".corrupt-{stamp}.json"))
            return
        if not isinstance(raw, dict):
            return
        for data in (raw.get("pages") or {}).values():
            if not isinstance(data, dict):
                continue
            try:
                page = Page.from_dict(data)
            except (TypeError, ValueError):
                continue
            if page.slug:
                self._pages[page.slug] = page

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "pages": {s: p.to_dict() for s, p in self._pages.items()},
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                   prefix=".wiki-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # ── reads ────────────────────────────────────────────────────

    def get(self, slug: str) -> Page | None:
        return self._pages.get(str(slug or "").strip().lower())

    def get_by_title(self, title: str) -> Page | None:
        wanted = str(title or "").strip().lower()
        for page in self._pages.values():
            if page.title.lower() == wanted:
                return page
        return None

    def list_pages(self, category: str = "") -> list[Page]:
        pages = list(self._pages.values())
        if category:
            pages = [p for p in pages
                     if p.category.lower() == category.lower()]
        return sorted(pages, key=_sort_key)

    def categories(self) -> list[str]:
        seen: list[str] = []
        for page in self._pages.values():
            if page.category and page.category not in seen:
                seen.append(page.category)
        return sorted(seen)

    # ── writes ───────────────────────────────────────────────────

    def _unique_slug(self, title: str) -> str:
        base = slugify(title)
        slug = base
        n = 2
        while slug in self._pages:
            slug = f"{base}-{n}"
            n += 1
        return slug

    def create(self, title: str, body: str, *,
               type: str = "guide", category: str = "",
               parent: str = "", order: int | None = None,
               summary: str = "", tags: list[str] | None = None) -> Page:
        """Create a page. Empty title/body and unknown types are errors."""
        title = str(title or "").strip()
        body = str(body or "")
        if not title:
            raise ValueError("wiki: a page needs a title")
        if not body.strip():
            raise ValueError("wiki: a page needs a body")
        if type not in PAGE_TYPES:
            raise ValueError(
                f"wiki: unknown page type {type!r} — one of "
                + ", ".join(sorted(PAGE_TYPES)))
        now = _now()
        page = Page(
            slug=self._unique_slug(title), title=title, body=body,
            type=type, category=str(category or "").strip(),
            parent=str(parent or "").strip(), order=order,
            summary=str(summary or "").strip(),
            tags=[str(t).strip() for t in (tags or []) if str(t).strip()],
            created_at=now, updated_at=now)
        self._pages[page.slug] = page
        self._save()
        return page

    def update(self, slug: str, body: str, *, note: str = "") -> bool:
        """Replace a page's body, keeping the old body as a revision."""
        page = self.get(slug)
        if page is None or not str(body or "").strip():
            return False
        page.revisions.append({
            "rev": len(page.revisions) + 1,
            "at": page.updated_at,
            "body": page.body,
            "note": str(note or ""),
        })
        page.revisions = page.revisions[-_MAX_REVISIONS:]
        page.body = str(body)
        page.updated_at = _now()
        self._save()
        return True

    def retitle(self, slug: str, new_title: str) -> bool:
        """Change a page's title. The slug never moves — links stay valid."""
        page = self.get(slug)
        new_title = str(new_title or "").strip()
        if page is None or not new_title:
            return False
        page.title = new_title
        page.updated_at = _now()
        self._save()
        return True

    def revert(self, slug: str, rev: int) -> bool:
        """Restore a page's body from revision ``rev`` (1-based)."""
        page = self.get(slug)
        if page is None:
            return False
        wanted = None
        for entry in page.revisions:
            if int(entry.get("rev", 0)) == int(rev):
                wanted = entry
                break
        if wanted is None:
            return False
        return self.update(slug, str(wanted.get("body", "")),
                           note=f"reverted to revision {rev}")

    def delete(self, slug: str) -> bool:
        page = self.get(slug)
        if page is None:
            return False
        del self._pages[page.slug]
        self._save()
        return True

    # ── markdown import / export ─────────────────────────────────

    def import_markdown(self, text: str) -> Page:
        """A markdown document (frontmatter + body) becomes a page.

        The frontmatter title is the page title; the type/category/
        parent/order/summary/tags ride along. A document with no
        title raises — a page without a title is not a page.
        """
        fields, body = parse_frontmatter(text or "")
        title = str(fields.get("title", "") or "").strip()
        if not title:
            raise ValueError("wiki: imported markdown needs a frontmatter title")
        if not body.strip():
            raise ValueError("wiki: imported markdown needs a body")
        return self.create(
            title, body,
            type=str(fields.get("type", "guide") or "guide"),
            category=str(fields.get("category", "") or ""),
            parent=str(fields.get("parent", "") or ""),
            order=fields.get("order"),
            summary=str(fields.get("summary", "") or ""),
            tags=[str(t) for t in (fields.get("tags") or [])])

    def export_markdown(self, slug: str) -> str | None:
        """A page as a markdown document (frontmatter + body)."""
        from .pages import emit_frontmatter

        page = self.get(slug)
        if page is None:
            return None
        return emit_frontmatter(page) + page.body
