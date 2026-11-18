"""Public face — publishing path for what already exists.

MORE_TO_ADD gap 10. NEW module. It publishes nothing new by itself:
it is the gate + descriptor layer over content that already exists in
the repo (JAMB tutor material, Yoruba singing, oral corpus, …).

Rules:

* OWNER-APPROVED ONLY. ``publish()`` with ``owner_approved=False``
  leaves the page a DRAFT — never public. There is no other path to
  public: approval is a parameter on the call, not a default.
* NO PRIVATE DATA LEAK. Every publish attempt runs
  :func:`scan_private_data` over title + content. Any violation keeps
  the page a draft and is returned to the caller. The scanner looks
  for vault/secret/token/password/private-key patterns and credential
  shapes — it is a safety net, not a guarantee; the owner still reviews.
* SHAREABLE LINK, served by the API server. :class:`LinkDescriptor`
  names where the page lives (``/p/<slug>``) — the API server's /p/
  route serves it from this store; this module still serves nothing
  itself and makes no hosting claim beyond that route.

Universal: packs are plain strings. ``KNOWN_PACKS`` names the three
launch packs (JAMB tutor, Yoruba singing, oral corpus); any other pack
is accepted and recorded — Nigeria default, never a boundary.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "KNOWN_PACKS",
    "PublicPage",
    "PublicPageStore",
    "LinkDescriptor",
    "PublishResult",
    "scan_private_data",
    "publish",
    "unpublish",
    "list_public",
]

#: The three launch packs named in the brief. Other packs are allowed.
KNOWN_PACKS = ("jamb-tutor", "yoruba-singing", "oral-corpus")

#: (label, regex) private-marker patterns. A match = violation.
_PRIVATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vault", re.compile(r"\bvault\b", re.IGNORECASE)),
    ("secret", re.compile(r"\bsecret\b", re.IGNORECASE)),
    ("token", re.compile(r"\b(token|api[_-]?key|apikey)\b", re.IGNORECASE)),
    ("password", re.compile(r"\bpassword\b", re.IGNORECASE)),
    ("private key", re.compile(r"private[-_ ]?key", re.IGNORECASE)),
    ("bearer", re.compile(r"\bbearer\s+[A-Za-z0-9._-]{8,}", re.IGNORECASE)),
    ("pem block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("credential assignment", re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*\S{4,}")),
    ("long hex secret", re.compile(r"\b[0-9a-f]{32,}\b")),
    ("seed phrase", re.compile(r"\b(seed phrase|mnemonic phrase)\b", re.IGNORECASE)),
)


@dataclass
class PublicPage:
    """A page candidate: title + content + pack. Starts as DRAFT."""

    title: str
    content: str
    pack: str
    slug: str = ""
    status: str = "draft"  # "draft" | "public"
    owner_approved: bool = False
    created: date = field(default_factory=date.today)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title is required")
        if not self.content.strip():
            raise ValueError("content is required")
        if not self.pack.strip():
            raise ValueError("pack is required")
        if not self.slug:
            base = re.sub(r"[^a-z0-9]+", "-", self.title.strip().lower()).strip("-")
            self.slug = base or f"page-{self.id}"
        if self.status not in ("draft", "public"):
            raise ValueError(f"unknown status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON shape (date as an ISO string)."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "pack": self.pack,
            "slug": self.slug,
            "status": self.status,
            "owner_approved": self.owner_approved,
            "created": self.created.isoformat(),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> PublicPage:
        """Rebuild from :meth:`to_dict`; a bad/missing field fails honestly."""
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"bad page row: no title in {row!r}")
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"bad page row: no content in {row!r}")
        pack = row.get("pack")
        if not isinstance(pack, str) or not pack.strip():
            raise ValueError(f"bad page row: no pack in {row!r}")
        status = row.get("status", "draft")
        if status not in ("draft", "public"):
            raise ValueError(f"bad page row: unknown status {status!r}")
        created = row.get("created")
        try:
            created_date = (
                date.fromisoformat(created) if created else date.today())
        except ValueError as exc:
            raise ValueError(
                f"bad page row: bad created date {created!r}") from exc
        return cls(
            id=str(row["id"]) if row.get("id") else uuid.uuid4().hex[:12],
            title=title,
            content=content,
            pack=pack,
            slug=str(row.get("slug", "")),
            status=status,
            owner_approved=bool(row.get("owner_approved", False)),
            created=created_date,
        )


@dataclass(frozen=True)
class LinkDescriptor:
    """Shareable-link descriptor for a public page (no hosting claim)."""

    slug: str
    url_path: str
    title: str
    pack: str
    page_id: str


@dataclass(frozen=True)
class PublishResult:
    """Outcome of a publish attempt: what happened, and why."""

    page: PublicPage
    published: bool
    link: LinkDescriptor | None
    violations: list[str]
    reason: str


def scan_private_data(*texts: str) -> list[str]:
    """Scan texts for private-data markers; return violation labels.

    Empty list = no markers found. A non-empty list means the content
    must NOT go public. The scan is pattern-based (a safety net); the
    owner review step is still required.
    """
    joined = "\n".join(texts)
    found: list[str] = []
    for label, pattern in _PRIVATE_PATTERNS:
        if pattern.search(joined):
            found.append(label)
    return found


def publish(page: PublicPage, *, owner_approved: bool) -> PublishResult:
    """Attempt to publish a page.

    Publishes ONLY when BOTH hold: ``owner_approved=True`` AND
    :func:`scan_private_data` finds no violations. Otherwise the page
    stays (or returns to) ``draft`` and the result explains why.
    Never raises for unapproved/leaking content — it refuses, in-band.
    """
    violations = scan_private_data(page.title, page.content)
    if not owner_approved:
        page.status = "draft"
        page.owner_approved = False
        return PublishResult(page, False, None, violations, "owner approval required — left as draft")
    if violations:
        page.status = "draft"
        page.owner_approved = False
        return PublishResult(
            page, False, None, violations,
            f"private-data markers found ({', '.join(violations)}) — left as draft",
        )
    page.status = "public"
    page.owner_approved = True
    link = LinkDescriptor(
        slug=page.slug, url_path=f"/p/{page.slug}",
        title=page.title, pack=page.pack, page_id=page.id,
    )
    return PublishResult(page, True, link, [], "published")


def unpublish(page: PublicPage) -> PublicPage:
    """Return a page to draft. Public pages can always be pulled back."""
    page.status = "draft"
    page.owner_approved = False
    return page


def list_public(pages: list[PublicPage]) -> list[LinkDescriptor]:
    """Link descriptors for every page currently public. Drafts excluded."""
    return [
        LinkDescriptor(
            slug=p.slug, url_path=f"/p/{p.slug}",
            title=p.title, pack=p.pack, page_id=p.id,
        )
        for p in pages
        if p.status == "public"
    ]


class PublicPageStore:
    """Durable registry of public-face pages (<home>/data/publicface.json).

    JSON-backed when a ``path`` is given (the /publicface surface stores
    it at ``<home>/data/publicface.json``); in-memory when it is not.
    The persistence shape mirrors
    :class:`nomorals.assets.maintenance.AssetLedger`: tolerant load (a
    missing/corrupt file starts empty, a bad row is skipped), atomic
    tmp+replace save after every mutation. The gate itself stays pure —
    this store is what makes an approval survive a restart.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.pages: dict[str, PublicPage] = {}
        if path:
            self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        for row in raw.get("pages") or []:
            try:
                page = PublicPage.from_dict(row)
            except (TypeError, ValueError, KeyError):
                continue
            self.pages[page.id] = page

    def _save(self) -> None:
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {"pages": [p.to_dict() for p in self.pages.values()]},
                    fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            _log.debug("publicface._save failed", exc_info=True)

    # -- pages --------------------------------------------------------
    def add(self, page: PublicPage) -> PublicPage:
        """Record a new page (draft by construction) and persist it."""
        self.pages[page.id] = page
        self._save()
        return page

    def put(self, page: PublicPage) -> PublicPage:
        """Persist a page after an in-place change (approve/unpublish)."""
        self.pages[page.id] = page
        self._save()
        return page

    def get(self, page_id: str) -> PublicPage | None:
        """Exact id lookup, else None."""
        return self.pages.get(page_id)

    def get_by_slug(self, slug: str) -> PublicPage | None:
        """First page with this slug, else None.

        Slugs are not unique in the store (two drafts may share a
        title); the /p/ route resolves only public pages, so callers
        that need the served page use
        :func:`nomorals.publicface.serve.resolve_public_page` instead.
        """
        for page in self.pages.values():
            if page.slug == slug:
                return page
        return None

    def remove(self, page_id: str) -> bool:
        """Delete a page; False when the id is not in the store."""
        if page_id not in self.pages:
            return False
        del self.pages[page_id]
        self._save()
        return True
