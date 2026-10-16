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
* SHAREABLE LINK DESCRIPTOR, not hosting. :class:`LinkDescriptor`
  describes where the page WOULD live (``/p/<slug>``) — this module
  makes no hosting claim and serves nothing.

Universal: packs are plain strings. ``KNOWN_PACKS`` names the three
launch packs (JAMB tutor, Yoruba singing, oral corpus); any other pack
is accepted and recorded — Nigeria default, never a boundary.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

__all__ = [
    "KNOWN_PACKS",
    "PublicPage",
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
