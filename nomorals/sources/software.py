"""Software-alternatives source (AlternativeTo — 52-websites site #4).

AlternativeTo (alternativeto.net, run since 2009, 150k+ apps) answers
one question: what should I use instead of this app? Every app has a
page listing its alternatives, ordered by the crowd's likes, with
platform / license / feature filters. The data model is the product:
a software catalogue (name, platforms, license, categories) plus a
many-to-many "alternative to" relation.

Devon's native organ is the same shape, built as her own catalogue:

* :class:`SoftwareRecord` — one app, with the real fields the site
  carries (likes, license tier + open-source flag + license id,
  categories, platforms, since year, comments, alternatives count).
* :class:`AlternativeEdge` — the "alternative to" relation.
* :class:`SoftwareCatalog` — the durable catalogue, seeded from the
  live site's real shape and grown by live fetch; stored as JSON
  under ``<home>/software_catalog.json``.
* :class:`AlternativeToSource` — live fetch (search + alternatives
  pages, parsed with the stdlib HTML parser) that merges into the
  catalogue. Never raises: a dead network still answers from seeds.

Honesty rule (from the mining report): seed records are catalogue
facts ("listed as an alternative, license: FOSS"), never crowd
consensus Devon doesn't have — only live-fetched records carry a
like count, and the reply names AlternativeTo as the source of the
rank. Devon herself goes first where she genuinely does the same
main task (image studio for Photoshop, documents for Word): she has
every skill herself, and the alternatives list says so.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ..core.http import HttpClient
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "AlternativeEdge",
    "AlternativeToSource",
    "SoftwareCatalog",
    "SoftwareRecord",
    "format_alternatives_reply",
    "parse_alternatives_page",
    "parse_search_page",
    "resolve_slug",
]

_BASE = "https://alternativeto.net"
_CATALOG_FILENAME = "software_catalog.json"

#: Download blogs and SPAs routinely 403 curl/python-urllib defaults.
_BROWSER_UA = (
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
)

#: Hard ceiling for any single source HTTP round-trip.
_HTTP_TIMEOUT = 20.0

#: The site's license tiers, in its own words.
LICENSE_TIERS = (
    "Free", "Freemium", "Free for personal use", "Paid", "Commercial",
)

#: Section headings the live pages carry, in DOM order.
_SECTION_LICENSE = "cost / license"
_SECTION_TYPES = "application type"
_SECTION_SINCE = "since"
_SECTION_PLATFORMS = "platforms"
_SECTION_ALERTS = "alerts"

_LIKES_RE = re.compile(r"([\d,]+)\s+likes?\b", re.IGNORECASE)
_COMMENTS_RE = re.compile(r"([\d,]+)\s+comments?\b", re.IGNORECASE)
_TOTAL_RE = re.compile(r"of\s+([\d,]+)\b.*\balternatives\b", re.IGNORECASE)
_SINCE_RE = re.compile(r"since\s+(\d{4})", re.IGNORECASE)
_LICENSE_ID_RE = re.compile(r"\(\s*([A-Za-z0-9][A-Za-z0-9.+\-]*)\s*\)")
_ALTS_LINK_RE = re.compile(r"^([\d,]+)\s+(.+?)\s+alternatives$", re.IGNORECASE)
_APP_HREF_RE = re.compile(r"^/software/([a-z0-9][a-z0-9\-]*)/?$")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return re.sub(r"-+", "-", slug)


def _int_or_none(text: str) -> int | None:
    try:
        return int(text.replace(",", ""))
    except (TypeError, ValueError):
        return None


# ── records ────────────────────────────────────────────────────────────────

@dataclass
class SoftwareRecord:
    """One app in the catalogue, with the fields AlternativeTo carries."""

    name: str = ""
    slug: str = ""
    description: str = ""
    url: str = ""
    homepage: str = ""
    license_tier: str = ""          # Free | Freemium | Paid | ...
    open_source: bool = False
    license_id: str = ""            # e.g. GPL-3.0 — "" when not stated
    categories: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    since_year: int | None = None
    likes: int | None = None        # None = catalogue fact, no crowd count
    comments: int | None = None
    alternatives_count: int | None = None
    discontinued: bool = False
    source: str = "catalogue"       # catalogue | alternativeto | devon
    is_devon: bool = False
    reason: str = ""                # set by the ranker, per query

    def __post_init__(self) -> None:
        if not self.slug and self.name:
            self.slug = _slugify(self.name)
        if not self.url and self.slug and not self.is_devon:
            self.url = f"{_BASE}/software/{self.slug}/"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "slug": self.slug,
            "description": self.description, "url": self.url,
            "homepage": self.homepage, "license_tier": self.license_tier,
            "open_source": self.open_source, "license_id": self.license_id,
            "categories": list(self.categories),
            "platforms": list(self.platforms),
            "since_year": self.since_year, "likes": self.likes,
            "comments": self.comments,
            "alternatives_count": self.alternatives_count,
            "discontinued": self.discontinued, "source": self.source,
            "is_devon": self.is_devon,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SoftwareRecord:
        return cls(
            name=data.get("name", ""), slug=data.get("slug", ""),
            description=data.get("description", ""),
            url=data.get("url", ""), homepage=data.get("homepage", ""),
            license_tier=data.get("license_tier", ""),
            open_source=bool(data.get("open_source", False)),
            license_id=data.get("license_id", ""),
            categories=list(data.get("categories", [])),
            platforms=list(data.get("platforms", [])),
            since_year=data.get("since_year"),
            likes=data.get("likes"), comments=data.get("comments"),
            alternatives_count=data.get("alternatives_count"),
            discontinued=bool(data.get("discontinued", False)),
            source=data.get("source", "catalogue"),
            is_devon=bool(data.get("is_devon", False)),
        )


@dataclass
class AlternativeEdge:
    """The "alternative to" relation: ``to`` replaces ``from``."""

    from_slug: str
    to_slug: str
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"from": self.from_slug, "to": self.to_slug,
                "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlternativeEdge:
        return cls(from_slug=data.get("from", ""), to_slug=data.get("to", ""),
                   note=data.get("note", ""))


# ── live-page parsing ──────────────────────────────────────────────────────

class _AlternativesParser(HTMLParser):
    """Tolerant parser for AlternativeTo search + alternatives pages.

    Keyed on the shapes the live site actually carries, not on tag or
    class names: an app entry starts at an ``/software/<slug>/`` anchor
    whose text slugifies to that slug; likes/comments/total come from
    their plain-text patterns; the Cost/License, Application type,
    Since, Platforms and Alerts sections are read off their headings.
    A sparse entry (no sections at all) still parses — the parser
    records what the page carries and leaves the rest blank.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[SoftwareRecord] = []
        self.total: int = 0
        self._cur: dict[str, Any] | None = None
        self._section: str = ""
        self._in_heading = False
        self._in_anchor = False
        self._anchor_href = ""
        self._anchor_text = ""
        self._in_li = False
        self._li_text = ""
        self._in_p = False
        self._p_text = ""
        self._saw_sections = False

    # -- entry lifecycle -------------------------------------------------

    def _start_entry(self, slug: str, name: str) -> None:
        self._finish_entry()
        self._cur = {
            "name": name, "slug": slug, "description": "",
            "license_tier": "", "open_source": False, "license_id": "",
            "categories": [], "platforms": [], "since_year": None,
            "likes": None, "comments": None, "alternatives_count": None,
            "discontinued": False,
        }
        self._section = ""
        self._saw_sections = False

    def _finish_entry(self) -> None:
        if not self._cur:
            return
        cur = self._cur
        self._cur = None
        if not cur["name"]:
            return
        self.records.append(SoftwareRecord(
            name=cur["name"], slug=cur["slug"],
            description=cur["description"],
            license_tier=cur["license_tier"],
            open_source=cur["open_source"], license_id=cur["license_id"],
            categories=cur["categories"], platforms=cur["platforms"],
            since_year=cur["since_year"], likes=cur["likes"],
            comments=cur["comments"],
            alternatives_count=cur["alternatives_count"],
            discontinued=cur["discontinued"], source="alternativeto"))

    # -- HTMLParser hooks -------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_heading = True
        elif tag == "a":
            self._in_anchor = True
            self._anchor_href = attrs_d.get("href", "") or ""
            self._anchor_text = ""
        elif tag == "li":
            self._in_li = True
            self._li_text = ""
        elif tag == "p":
            self._in_p = True
            self._p_text = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_heading = False
        elif tag == "a":
            self._in_anchor = False
            self._handle_anchor()
        elif tag == "li":
            self._in_li = False
            self._handle_li()
        elif tag == "p":
            self._in_p = False
            self._handle_p()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        # Anchor text first: an app name lives inside its <h3> anchor,
        # and the heading branch below would swallow it as heading text.
        if self._in_anchor:
            self._anchor_text += (" " if self._anchor_text else "") + text
            # Anchor text also belongs to its enclosing container (the
            # license id lives inside an anchor in a Cost/License item).
        if self._in_heading and not self._in_anchor:
            heading = text.lower()
            if _SECTION_LICENSE in heading:
                self._section = _SECTION_LICENSE
                self._saw_sections = True
            elif _SECTION_TYPES in heading:
                self._section = _SECTION_TYPES
                self._saw_sections = True
            elif heading == _SECTION_SINCE or heading.startswith("since"):
                self._section = _SECTION_SINCE
                self._saw_sections = True
            elif _SECTION_PLATFORMS in heading:
                self._section = _SECTION_PLATFORMS
                self._saw_sections = True
            elif _SECTION_ALERTS in heading:
                self._section = _SECTION_ALERTS
                self._saw_sections = True
            return
        # Likes ride in their own div, nested inside the entry's outer
        # <li> — catch them before container routing, whatever wraps them.
        if self._cur is not None and self._cur["likes"] is None:
            m = _LIKES_RE.search(text)
            if m:
                self._cur["likes"] = _int_or_none(m.group(1))
        if self._in_p:
            self._p_text += (" " if self._p_text else "") + text
            return
        if self._in_li:
            self._li_text += (" " if self._li_text else "") + text
            return
        self._handle_loose_text(text)

    # -- text handling ----------------------------------------------------

    def _handle_anchor(self) -> None:
        href = self._anchor_href
        text = self._anchor_text.strip()
        self._anchor_href = ""
        self._anchor_text = ""
        m = _APP_HREF_RE.match(href)
        if not m or not text:
            return
        slug = m.group(1)
        alts = _ALTS_LINK_RE.match(text)
        if alts:
            # "238 GIMP alternatives" — the count link for the current entry.
            if self._cur and self._cur["slug"] == slug:
                self._cur["alternatives_count"] = _int_or_none(alts.group(1))
            return
        if _slugify(text) != slug:
            # "More about Adobe Photoshop" and the like: not an entry.
            return
        self._start_entry(slug, text)

    def _handle_li(self) -> None:
        text = self._li_text.strip()
        self._li_text = ""
        # Only a section list's items count: the entry's outer <li>
        # wraps the whole entry, so anything outside a section heading
        # (likes, the alternatives link) is not a section item.
        if not self._cur or not text or not self._section:
            return
        if self._section == _SECTION_LICENSE:
            self._apply_license_item(text)
        elif self._section == _SECTION_TYPES:
            if text not in self._cur["categories"]:
                self._cur["categories"].append(text)
        elif self._section == _SECTION_SINCE:
            m = _SINCE_RE.search(text)
            if m:
                self._cur["since_year"] = int(m.group(1))
        elif self._section == _SECTION_PLATFORMS:
            if text not in self._cur["platforms"]:
                self._cur["platforms"].append(text)
        elif self._section == _SECTION_ALERTS and \
                "discontinued" in text.lower():
            self._cur["discontinued"] = True

    def _apply_license_item(self, text: str) -> None:
        assert self._cur is not None
        low = text.lower()
        # Word-boundary match, longest tier first: "Free" is a
        # substring of "Freemium" and must not win over it.
        for tier in sorted(LICENSE_TIERS, key=len, reverse=True):
            if re.search(r"\b" + re.escape(tier.lower()) + r"\b", low):
                self._cur["license_tier"] = tier
                break
        if "open source" in low:
            self._cur["open_source"] = True
            m = _LICENSE_ID_RE.search(text)
            if m:
                self._cur["license_id"] = m.group(1)
        elif "proprietary" in low:
            self._cur["open_source"] = False

    def _handle_p(self) -> None:
        text = self._p_text.strip()
        self._p_text = ""
        if not self._cur or not text:
            return
        if text.lower().startswith("good alternative?"):
            return  # entry tail marker — the name is already recorded
        m = _TOTAL_RE.search(text)
        if m:
            self.total = _int_or_none(m.group(1)) or 0
            return
        m = _COMMENTS_RE.search(text)
        if m:
            self._cur["comments"] = _int_or_none(m.group(1))
            return
        if (not self._saw_sections and not self._cur["description"]
                and len(text) > 40):
            self._cur["description"] = text

    def _handle_loose_text(self, text: str) -> None:
        if self._cur:
            m = _LIKES_RE.search(text)
            if m:
                self._cur["likes"] = _int_or_none(m.group(1))
                return
        m = _TOTAL_RE.search(text)
        if m:
            self.total = _int_or_none(m.group(1)) or 0

    def close(self) -> None:
        super().close()
        self._finish_entry()


def parse_alternatives_page(html: str) -> tuple[list[SoftwareRecord], int]:
    """Parse an alternatives page; returns (records, page total)."""
    if not html:
        return [], 0
    parser = _AlternativesParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — a broken page parses to nothing
        _log.debug("alternativeto alternatives parse failed", exc_info=True)
        return [], 0
    return parser.records, parser.total


def parse_search_page(html: str) -> list[SoftwareRecord]:
    """Parse a search page; returns the ranked app results."""
    if not html:
        return []
    parser = _AlternativesParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — a broken page parses to nothing
        _log.debug("alternativeto search parse failed", exc_info=True)
        return []
    return parser.records


# ── seeds: the catalogue's day-one facts ───────────────────────────────────

def _rec(name: str, slug: str, **kw: Any) -> SoftwareRecord:
    return SoftwareRecord(name=name, slug=slug, **kw)


#: Seed records. Likes are carried ONLY where they were read off the
#: live site (Photoshop family, 2026-11-17); every other record is a
#: catalogue fact with likes=None — never an invented crowd count.
SEED_RECORDS: list[SoftwareRecord] = [
    _rec("Adobe Photoshop", "adobe-photoshop",
         description="Proprietary raster graphics editor by Adobe.",
         license_tier="Paid", open_source=False,
         categories=["Image Editor", "Digital Painting Tool"],
         platforms=["Windows", "Mac", "Online", "Android", "iPhone",
                    "iPad"],
         since_year=1990, likes=2469, alternatives_count=277),
    _rec("GIMP", "gimp",
         description="Free open-source suite offering advanced photo "
                     "retouching, image composition, original artwork "
                     "creation, batch processing, plugin support and "
                     "custom scripting.",
         license_tier="Free", open_source=True, license_id="GPL-3.0",
         categories=["Image Editor", "Digital Painting"],
         platforms=["Windows", "Mac", "Linux", "BSD", "Flathub",
                    "Snapcraft", "Flatpak", "AppImage"],
         since_year=2009, likes=4888, comments=116,
         alternatives_count=238),
    _rec("Krita", "krita",
         description="Free, open-source painting software designed by "
                     "artists for artists.",
         license_tier="Free", open_source=True,
         categories=["Digital Painting"],
         platforms=["Windows", "Mac", "Linux", "Android Tablet",
                    "F-Droid", "AppImage"],
         since_year=2009, likes=1520, comments=49,
         alternatives_count=175),
    _rec("Paint.NET", "paint-net",
         description="Free Windows image editor with support for layers, "
                     "plugins, and special effects.",
         license_tier="Free", open_source=False,
         categories=["Image Editor"], platforms=["Windows"],
         likes=1674, comments=21, alternatives_count=193),
    _rec("Affinity", "affinity",
         description="Unified design solution merging advanced photo, "
                     "vector, and layout tools.",
         license_tier="Freemium", open_source=False,
         categories=["Image Editor", "Vector Graphic"],
         platforms=["Windows", "Mac", "iPad"],
         since_year=2025, likes=99, comments=6, alternatives_count=122),
    _rec("Photopea", "photopea",
         description="Browser-based image editing supporting PSD, XCF and "
                     "Sketch formats, layers, vector tools and RAW.",
         license_tier="Freemium", open_source=False,
         categories=["Image Editor", "Digital Painting"],
         platforms=["Online", "Software as a Service (SaaS)"],
         since_year=2016, likes=371, comments=31,
         alternatives_count=152),
    _rec("Pixlr", "pixlr",
         description="Web-based editor with AI image generation, quick "
                     "fixes, overlays and advanced filters.",
         license_tier="Paid", open_source=False,
         categories=["Image Editor", "Photo Editor"],
         platforms=["Windows", "Chrome OS", "Online", "Android",
                    "iPhone", "iPad", "Android Tablet"],
         since_year=2012, likes=90, comments=10,
         alternatives_count=265),
    _rec("MyPaint", "mypaint",
         description="Fast, distraction-free painting tool for digital "
                     "artists, with graphics-tablet support.",
         license_tier="Free", open_source=True, license_id="GPL-2.0",
         categories=["Digital Painting"],
         platforms=["Windows", "Mac", "Linux", "PortableApps.com"],
         since_year=2009, likes=398, comments=5,
         alternatives_count=147),
    _rec("Pixelmator", "pixelmator",
         description="Image editing software for macOS and iOS, offering "
                     "nondestructive, GPU-powered editing.",
         license_tier="Paid", open_source=False,
         categories=["Image Editor"], platforms=["Mac", "iPhone", "iPad"],
         likes=424, comments=5, alternatives_count=167),
    _rec("Luminar Neo", "luminar-neo",
         description="Advanced RAW photo editing software with AI-driven "
                     "features.",
         license_tier="Paid", open_source=False,
         categories=["Image Editor", "Photo Editor"],
         platforms=["Windows", "Mac", "Chrome OS", "Android", "iPhone",
                    "iPad"],
         since_year=2017, likes=41, alternatives_count=122),
    _rec("PhotoCraft", "photocraft",
         description="Layers, masks, adjustment layers and brushes in a "
                     "native app written entirely in Rust. Open source, "
                     "offline, and yours.",
         license_tier="Free", open_source=True,
         categories=["Image Editor"], likes=66, comments=1,
         alternatives_count=16),
    _rec("PixiEditor", "pixieditor",
         description="Pixel-perfect sprites, raster paintings and scalable "
                     "vector graphics with frame-by-frame animation.",
         license_tier="Free", open_source=True, license_id="LGPL-3.0",
         categories=["Image Editor"],
         platforms=["Windows", "Linux", "Steam", "Flathub", "Flatpak"],
         since_year=2024, likes=32, alternatives_count=16),
    _rec("Microsoft Office", "microsoft-office",
         description="Microsoft's office suite: Word, Excel, PowerPoint.",
         license_tier="Paid", open_source=False,
         categories=["Office Suite"],
         platforms=["Windows", "Mac", "Online", "Android", "iPhone",
                    "iPad"],
         since_year=1990),
    _rec("LibreOffice", "libreoffice",
         description="Free and open-source office suite: Writer, Calc, "
                     "Impress, Draw, Base, Math.",
         license_tier="Free", open_source=True, license_id="MPL-2.0",
         categories=["Office Suite"],
         platforms=["Windows", "Mac", "Linux", "Android", "iPhone",
                    "iPad"],
         since_year=2010),
    _rec("ONLYOFFICE", "onlyoffice",
         description="Open-source office suite with document, spreadsheet "
                     "and presentation editors.",
         license_tier="Freemium", open_source=True,
         license_id="AGPL-3.0",
         categories=["Office Suite"],
         platforms=["Windows", "Mac", "Linux", "Online"],
         since_year=2009),
    _rec("Google Docs", "google-docs",
         description="Google's web-based word processor, part of Google "
                     "Workspace.",
         license_tier="Freemium", open_source=False,
         categories=["Office Suite", "Word Processor"],
         platforms=["Online", "Android", "iPhone", "iPad"]),
    _rec("WPS Office", "wps-office",
         description="Office suite with writer, spreadsheets and "
                     "presentation, compatible with Microsoft formats.",
         license_tier="Freemium", open_source=False,
         categories=["Office Suite"],
         platforms=["Windows", "Mac", "Linux", "Android", "iPhone",
                    "iPad"]),
    _rec("Notion", "notion",
         description="All-in-one workspace for notes, docs, wikis and "
                     "project tracking.",
         license_tier="Freemium", open_source=False,
         categories=["Note-taking", "Knowledge Management"],
         platforms=["Windows", "Mac", "Online", "Android", "iPhone",
                    "iPad"]),
    _rec("Obsidian", "obsidian",
         description="Local-first note-taking app built on plain Markdown "
                     "files, with a graph view.",
         license_tier="Free for personal use", open_source=False,
         categories=["Note-taking", "Knowledge Management"],
         platforms=["Windows", "Mac", "Linux", "Android", "iPhone",
                    "iPad"]),
    _rec("Joplin", "joplin",
         description="Free, open-source note-taking app with sync and "
                     "Markdown support.",
         license_tier="Free", open_source=True,
         categories=["Note-taking"],
         platforms=["Windows", "Mac", "Linux", "Android", "iPhone",
                    "iPad"]),
    _rec("Anytype", "anytype",
         description="Local-first, open-source workspace for notes, "
                     "tasks and wikis.",
         license_tier="Free", open_source=True,
         categories=["Note-taking", "Knowledge Management"],
         platforms=["Windows", "Mac", "Linux", "Android", "iPhone",
                    "iPad"]),
    _rec("Logseq", "logseq",
         description="Privacy-first, open-source knowledge base with "
                     "outlining and backlinks.",
         license_tier="Free", open_source=True, license_id="AGPL-3.0",
         categories=["Note-taking", "Knowledge Management"],
         platforms=["Windows", "Mac", "Linux", "Android", "iPhone",
                    "iPad"]),
    _rec("Slack", "slack",
         description="Team chat and collaboration hub with channels, "
                     "threads and integrations.",
         license_tier="Freemium", open_source=False,
         categories=["Team Chat"],
         platforms=["Windows", "Mac", "Linux", "Online", "Android",
                    "iPhone", "iPad"]),
    _rec("Mattermost", "mattermost",
         description="Open-source, self-hostable team messaging platform.",
         license_tier="Free", open_source=True, license_id="MIT",
         categories=["Team Chat"],
         platforms=["Windows", "Mac", "Linux", "Online", "Android",
                    "iPhone", "iPad"]),
    _rec("Zulip", "zulip",
         description="Open-source team chat with topic-based threading.",
         license_tier="Free", open_source=True, license_id="Apache-2.0",
         categories=["Team Chat"],
         platforms=["Windows", "Mac", "Linux", "Online", "Android",
                    "iPhone", "iPad"]),
    _rec("Rocket.Chat", "rocket-chat",
         description="Open-source team communication platform, "
                     "self-hostable.",
         license_tier="Free", open_source=True, license_id="MIT",
         categories=["Team Chat"],
         platforms=["Windows", "Mac", "Linux", "Online", "Android",
                    "iPhone", "iPad"]),
    _rec("Discord", "discord",
         description="Voice, video and text chat for communities and "
                     "teams.",
         license_tier="Freemium", open_source=False,
         categories=["Team Chat"],
         platforms=["Windows", "Mac", "Linux", "Online", "Android",
                    "iPhone", "iPad"]),
    _rec("Zoom", "zoom",
         description="Video conferencing with meetings, webinars and "
                     "screen sharing.",
         license_tier="Freemium", open_source=False,
         categories=["Video Conferencing"],
         platforms=["Windows", "Mac", "Linux", "Online", "Android",
                    "iPhone", "iPad"]),
    _rec("Jitsi Meet", "jitsi-meet",
         description="Free, open-source video conferencing in the "
                     "browser — no account needed.",
         license_tier="Free", open_source=True, license_id="Apache-2.0",
         categories=["Video Conferencing"],
         platforms=["Online", "Android", "iPhone", "iPad"]),
    _rec("BigBlueButton", "bigbluebutton",
         description="Open-source web conferencing built for online "
                     "learning.",
         license_tier="Free", open_source=True, license_id="LGPL-3.0",
         categories=["Video Conferencing"],
         platforms=["Online", "Windows", "Mac", "Linux"]),
    _rec("Google Meet", "google-meet",
         description="Google's video conferencing, part of Google "
                     "Workspace.",
         license_tier="Freemium", open_source=False,
         categories=["Video Conferencing"],
         platforms=["Online", "Android", "iPhone", "iPad"]),
]

#: Seed edges: (incumbent slug, alternative slug, note).
SEED_EDGES: list[tuple[str, str, str]] = [
    ("adobe-photoshop", "gimp", "same task: raster image editing"),
    ("adobe-photoshop", "krita", "same task: digital painting"),
    ("adobe-photoshop", "paint-net", "same task: image editing"),
    ("adobe-photoshop", "affinity", "same task: photo + design"),
    ("adobe-photoshop", "photopea", "same task: image editing, in browser"),
    ("adobe-photoshop", "pixlr", "same task: photo editing"),
    ("adobe-photoshop", "mypaint", "same task: digital painting"),
    ("adobe-photoshop", "pixelmator", "same task: image editing"),
    ("adobe-photoshop", "luminar-neo", "same task: photo editing"),
    ("adobe-photoshop", "photocraft", "same task: image editing"),
    ("adobe-photoshop", "pixieditor", "same task: image editing"),
    ("microsoft-office", "libreoffice", "same task: office suite"),
    ("microsoft-office", "onlyoffice", "same task: office suite"),
    ("microsoft-office", "google-docs", "same task: documents"),
    ("microsoft-office", "wps-office", "same task: office suite"),
    ("notion", "obsidian", "same task: notes + knowledge"),
    ("notion", "joplin", "same task: note-taking"),
    ("notion", "anytype", "same task: notes + knowledge"),
    ("notion", "logseq", "same task: knowledge base"),
    ("slack", "mattermost", "same task: team chat"),
    ("slack", "zulip", "same task: team chat"),
    ("slack", "rocket-chat", "same task: team chat"),
    ("slack", "discord", "same task: team chat"),
    ("zoom", "jitsi-meet", "same task: video conferencing"),
    ("zoom", "bigbluebutton", "same task: video conferencing"),
    ("zoom", "google-meet", "same task: video conferencing"),
]

#: Incumbents Devon herself replaces — the same main task, done by her
#: own organs. (slug, Devon name, reason).
DEVON_SELF: dict[str, tuple[str, str]] = {
    "adobe-photoshop": (
        "Devon image studio",
        "Devon herself — she generates and edits images (/imggen, "
        "/video gen, EditStudio), no install, no license",
    ),
    "microsoft-office": (
        "Devon documents",
        "Devon herself — she composes and edits documents "
        "(the documents organ), no install, no license",
    ),
}

#: Name → slug for the incumbents the catalogue knows by heart.
_SEED_SLUGS: dict[str, str] = {
    _slugify(rec.name): rec.slug for rec in SEED_RECORDS
}
_SEED_SLUGS.update({
    "photoshop": "adobe-photoshop",
    "ms-office": "microsoft-office",
    "office": "microsoft-office",
    "word": "microsoft-office",
})


# ── the durable catalogue ──────────────────────────────────────────────────

class SoftwareCatalog:
    """The software catalogue: records + "alternative to" edges.

    ``home`` is a directory; the store lives at
    ``<home>/software_catalog.json``. With ``home=None`` the catalogue
    is in-memory only (seeds, no disk writes) — the shape the tests
    and the offline path use.
    """

    def __init__(self, home: str | Path | None = None) -> None:
        self.path: Path | None = (
            Path(home) / _CATALOG_FILENAME if home is not None else None)
        self.records: dict[str, SoftwareRecord] = {}
        self.edges: list[AlternativeEdge] = []
        if self.path is not None and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for rec in data.get("records", {}).values():
                    record = SoftwareRecord.from_dict(rec)
                    self.records[record.slug] = record
                self.edges = [AlternativeEdge.from_dict(e)
                              for e in data.get("edges", [])]
            except Exception:  # noqa: BLE001 — corrupt store reseeds
                _log.warning("software catalogue unreadable; reseeding",
                             exc_info=True)
                self.records = {}
                self.edges = []
        if not self.records:
            for rec in SEED_RECORDS:
                self.records[rec.slug] = rec
            self.edges = [AlternativeEdge(f, t, n)
                          for f, t, n in SEED_EDGES]
            self._save()

    # -- persistence ------------------------------------------------------

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "records": {s: r.to_dict()
                            for s, r in self.records.items()},
                "edges": [e.to_dict() for e in self.edges],
            }, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:  # noqa: BLE001 — the catalogue stays in memory
            _log.warning("software catalogue save failed", exc_info=True)

    # -- mutation ---------------------------------------------------------

    def upsert(self, record: SoftwareRecord) -> None:
        """Add or refresh a record (live fetch merges here)."""
        old = self.records.get(record.slug)
        if old is not None and record.source == "alternativeto":
            # A live record refreshes the seed but never erases a
            # field the live page didn't carry.
            merged = SoftwareRecord.from_dict(old.to_dict())
            for key in ("description", "license_tier", "license_id",
                        "since_year", "likes", "comments",
                        "alternatives_count"):
                value = getattr(record, key)
                if value not in ("", None):
                    setattr(merged, key, value)
            if record.categories:
                merged.categories = record.categories
            if record.platforms:
                merged.platforms = record.platforms
            merged.open_source = record.open_source
            merged.discontinued = record.discontinued
            merged.source = "alternativeto"
            self.records[record.slug] = merged
        else:
            self.records[record.slug] = record
        self._save()

    def add_edge(self, from_slug: str, to_slug: str, note: str = "") -> None:
        if from_slug == to_slug:
            return
        if any(e.from_slug == from_slug and e.to_slug == to_slug
               for e in self.edges):
            return
        self.edges.append(AlternativeEdge(from_slug, to_slug, note))
        self._save()

    # -- query ------------------------------------------------------------

    def record(self, slug: str) -> SoftwareRecord | None:
        return self.records.get(slug)

    def alternatives_for(
        self, slug: str, *, platform: str | None = None,
        license_tier: str | None = None, open_source: bool | None = None,
        limit: int = 10,
    ) -> list[SoftwareRecord]:
        """Ranked alternatives to ``slug`` (Devon first, then crowd rank).

        Filters drop non-matches; the site's crowd order is the rank —
        Devon only filters and explains, she never invents a rank.
        Each result carries a ``reason`` naming why it matches.
        """
        incumbent = self.records.get(slug)
        out: list[SoftwareRecord] = []
        if slug in DEVON_SELF:
            name, reason = DEVON_SELF[slug]
            devon = SoftwareRecord(
                name=name, slug=f"devon-{slug}", source="devon",
                is_devon=True, license_tier="Free",
                categories=list(incumbent.categories) if incumbent else [],
                platforms=["Online"], reason=reason)
            if self._matches(devon, platform, license_tier, open_source):
                out.append(devon)
        for edge in self.edges:
            if edge.from_slug != slug:
                continue
            rec = self.records.get(edge.to_slug)
            if rec is None or rec.discontinued:
                continue
            if not self._matches(rec, platform, license_tier, open_source):
                continue
            ranked = SoftwareRecord.from_dict(rec.to_dict())
            ranked.reason = self._reason(ranked, incumbent, platform)
            out.append(ranked)
        # Crowd rank: likes desc, unranked (seed-only) after the ranked.
        devon_first = [r for r in out if r.is_devon]
        rest = [r for r in out if not r.is_devon]
        rest.sort(key=lambda r: (
            r.likes is None, -(r.likes or 0), r.name.lower()))
        return (devon_first + rest)[: max(limit, 0)]

    @staticmethod
    def _matches(rec: SoftwareRecord, platform: str | None,
                 license_tier: str | None,
                 open_source: bool | None) -> bool:
        if platform:
            want = platform.strip().lower()
            # "Online" counts as a platform of its own; a Devon
            # record runs in chat, not on a desktop OS.
            if (not any(want == p.lower() or want in p.lower()
                        for p in rec.platforms)
                    and not (rec.is_devon
                             and want in ("online", "web"))):
                return False
        if (license_tier and rec.license_tier.lower()
                != license_tier.strip().lower()):
            return False
        return open_source is None or rec.open_source == open_source

    @staticmethod
    def _reason(rec: SoftwareRecord,
                incumbent: SoftwareRecord | None,
                platform: str | None) -> str:
        parts: list[str] = []
        if rec.license_tier:
            lic = rec.license_tier
            if rec.open_source and rec.license_id:
                lic += f" · Open Source ({rec.license_id})"
            elif rec.open_source:
                lic += " · Open Source"
            parts.append(lic)
        if platform:
            # Name the platform the way the record itself carries it.
            spelling = next(
                (p for p in rec.platforms
                 if platform.lower() == p.lower()
                 or platform.lower() in p.lower()),
                platform)
            parts.append(f"runs on {spelling}")
        elif rec.platforms:
            parts.append(", ".join(rec.platforms[:3]))
        if incumbent:
            shared = [c for c in incumbent.categories
                      if c in rec.categories]
            if shared:
                parts.append("shares: " + ", ".join(shared))
        if rec.likes is not None:
            parts.append(f"{rec.likes:,} likes on AlternativeTo")
        else:
            parts.append("listed in Devon's catalogue")
        return " · ".join(parts)


# ── the live source ────────────────────────────────────────────────────────

def _default_fetch(url: str) -> str:
    try:
        client = HttpClient(timeout=_HTTP_TIMEOUT, user_agent=_BROWSER_UA)
        resp = client.get(url)
        if not resp.ok:
            return ""
        return resp.text
    except Exception:  # noqa: BLE001 — fetch failure is an empty page
        _log.debug("alternativeto fetch failed: %s", url, exc_info=True)
        return ""


Fetcher = Callable[[str], str]


def resolve_slug(name: str, fetcher: Fetcher | None = None) -> str | None:
    """Slug for a software name: seed map first, live search second."""
    key = _slugify(name or "")
    if not key:
        return None
    if key in _SEED_SLUGS:
        return _SEED_SLUGS[key]
    if key in _SEED_SLUGS.values():
        return key
    fetch = fetcher or _default_fetch
    try:
        html = fetch(
            f"{_BASE}/browse/search/?q={urllib.parse.quote_plus(name)}")
    except Exception:  # noqa: BLE001 — no network, no slug
        return None
    results = parse_search_page(html or "")
    if not results:
        return None
    for rec in results:
        if _slugify(rec.name) == key:
            return rec.slug
    return results[0].slug


class AlternativeToSource:
    """AlternativeTo as a Devon source: search + alternatives, merged
    into the durable catalogue. Every public method never raises."""

    name = "alternativeto"
    _BASE = _BASE

    def __init__(self, catalog: SoftwareCatalog | None = None,
                 fetcher: Fetcher | None = None,
                 home: str | Path | None = None) -> None:
        self.catalog = catalog or SoftwareCatalog(home)
        self._fetcher = fetcher or _default_fetch

    # -- URLs -------------------------------------------------------------

    def search_url(self, query: str) -> str:
        return (f"{_BASE}/browse/search/"
                f"?q={urllib.parse.quote_plus(query)}")

    def alternatives_url(self, slug: str) -> str:
        return f"{_BASE}/software/{slug}/"

    # -- queries ----------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> list[SoftwareRecord]:
        """Live search: ranked apps matching ``query``. Never raises."""
        try:
            html = self._fetcher(self.search_url(query))
        except Exception:  # noqa: BLE001
            return []
        if not html:
            return []
        records = parse_search_page(html)
        for rec in records:
            self.catalog.upsert(rec)
        return records[: max(limit, 0)]

    def alternatives(
        self, name: str, *, platform: str | None = None,
        license_tier: str | None = None, open_source: bool | None = None,
        limit: int = 10,
    ) -> list[SoftwareRecord]:
        """Ranked alternatives to ``name`` (Devon first, then crowd rank).

        Resolves the slug, fetches the live alternatives page to grow
        the catalogue, and answers from the catalogue — so a dead
        network still answers from seeds.
        """
        slug = resolve_slug(name, fetcher=self._fetcher)
        if slug is None:
            return []
        try:
            html = self._fetcher(self.alternatives_url(slug))
        except Exception:  # noqa: BLE001
            html = ""
        if html:
            records, _total = parse_alternatives_page(html)
            for rec in records:
                self.catalog.upsert(rec)
                self.catalog.add_edge(
                    slug, rec.slug, "same task: listed as an alternative")
            if self.catalog.record(slug) is None and records:
                self.catalog.upsert(SoftwareRecord(
                    name=name, slug=slug, source="alternativeto"))
        return self.catalog.alternatives_for(
            slug, platform=platform, license_tier=license_tier,
            open_source=open_source, limit=limit)


# ── chat rendering ─────────────────────────────────────────────────────────

def format_alternatives_reply(
    software: str, results: list[SoftwareRecord],
    total: int | None = None,
) -> str:
    """Plain-words reply for /alternatives and the brain tool."""
    if not results:
        return (f"No alternatives to {software} found — try the full "
                "name, or a different app.")
    lines = [f"Alternatives to {software}:"]
    if total:
        lines[0] += f" ({total:,} listed on AlternativeTo)"
    for i, rec in enumerate(results, 1):
        head = f"{i}. {rec.name}"
        if rec.is_devon:
            lines.append(f"{head} — {rec.reason}")
            continue
        bits = []
        if rec.reason:
            bits.append(rec.reason)
        lines.append(f"{head} — {' · '.join(bits) if bits else 'listed'}")
        if rec.description:
            lines.append(f"   {rec.description}")
        if rec.url:
            lines.append(f"   {rec.url}")
    lines.append("Source: AlternativeTo (alternativeto.net) — crowd-ranked; "
                 "Devon herself first where she can do it herself.")
    return "\n".join(lines)
