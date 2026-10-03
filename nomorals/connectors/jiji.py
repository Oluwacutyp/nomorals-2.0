"""Jiji connector — Nigerian classifieds (jiji.ng) via rendered browser tabs.

What exists (verified 2026-10-02): jiji.ng — including /robots.txt — sits
behind a Cloudflare "managed challenge". Plain HTTP fetch (urllib/curl)
gets an HTTP 403 challenge page, so Devon's plain ``BrowserSession`` cannot
pass it. Only a real JavaScript browser passes; this connector drives
``BrowserService.open_rendered_tab()`` (real headless Chromium via
playwright) for every page load.

What does NOT exist (so this connector honestly cannot do):

* **No public API.** Jiji publishes no search/listings/seller API —
  everything here is rendered-DOM scraping of the public site.
* **No automated seller contact or login.** Reaching out to a seller or
  signing in happens through a human-in-the-loop checkpoint: Devon
  prepares the listing link and message, the owner acts personally.
  This connector never auto-messages sellers and never auto-logs-in.

Parsing is heuristic (no API contract to pin to): listing cards are found
by their URL shape (``https://jiji.ng/<location>/<category>/<slug>-<id>.html``)
and their price/title/location are split out of the card text. If Jiji
changes its markup so no cards are found, every search fails fast with
``JijiError("jiji page structure changed ...")`` rather than returning
empty results that look like "no listings".
"""

from __future__ import annotations

import re
import time
import urllib.parse
from html.parser import HTMLParser
from typing import Any

from ..browser.service import BrowserService, RenderedTab
from ..core.logging_setup import get_logger
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = [
    "JijiConnector",
    "JijiError",
    "JIJI_BASE",
    "JIJI_SEARCH_URL",
]

_log = get_logger(__name__)

#: Public site base.
JIJI_BASE = "https://jiji.ng"

#: Canonical search pattern (verified against the live site 2026-10-02:
#: the homepage search form submits here). Kept as the fallback; the
#: connector re-discovers it from the homepage on every fresh session so
#: a site-side change fails fast instead of silently 404ing.
JIJI_SEARCH_URL = "https://jiji.ng/search?query={query}"

#: Jiji rate-limits aggressively; stay polite — minimum gap between page
#: loads, enforced in _throttle().
_MIN_REQUEST_GAP_S = 3.0

#: In-memory result cache TTL (searches, listing details, categories).
_CACHE_TTL_S = 15 * 60.0

#: Hosts that count as jiji.ng for listing-URL detection.
_JIJI_HOSTS = {"jiji.ng", "www.jiji.ng"}

#: Card text that means "no numeric price" (Jiji shows these instead of ₦).
_NO_PRICE_MARKERS = ("contact", "swap", "exchange")


class JijiError(ConnectorError):
    """A Jiji page load or parse failed."""


#: Nigerian states + FCT, for the geography fallback in detail-page
#: location extraction (Jiji ad pages always carry the ad's location,
#: but the exact label/markup varies).
_NG_STATES = frozenset({
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa",
    "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti",
    "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina",
    "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun",
    "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
    "Yobe", "Zamfara", "FCT", "Abuja", "Federal Capital Territory",
})

#: labeled location lines on ad pages, e.g. "Location: Lekki, Lagos"
_LOCATION_LABEL_RE = re.compile(
    r"(?:^|[\s>•·|])(?:ad\s+)?location\s*:\s*(.+)$"
    r"|(?:^|[\s>•·|])address\s*:\s*(.+)$",
    re.IGNORECASE,
)

#: alternation of Nigerian state names, longest-first so "Cross River"
#: wins over a prefix collision
_STATE_ALT = "|".join(
    sorted((re.escape(s) for s in _NG_STATES), key=len, reverse=True)
)

#: "Area, State" geography, e.g. "Lekki Phase 1, Lagos" or "Ikeja, Lagos"
_LOCATION_GEO_RE = re.compile(
    rf"\b([A-Z][\w\-\s']{{1,40}}?),\s*({_STATE_ALT})\b"
)


class _LocationClassCollector(HTMLParser):
    """Text of elements whose class or id mentions "location".

    Jiji ad pages render the ad's area in a location-labeled element; its
    exact tag varies, so this matches on the attribute instead.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.texts: list[str] = []
        self._depth = 0
        self._chunks: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if self._depth:
            self._depth += 1
            return
        attrs_d = dict(attrs)
        hay = f"{attrs_d.get('class', '')} {attrs_d.get('id', '')}".lower()
        if "location" in hay:
            self._depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self._depth:
            self._depth -= 1
            if self._depth == 0:
                text = " ".join("".join(self._chunks).split())
                if text:
                    self.texts.append(text)
                self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._chunks.append(data)


def _clean_area(raw: str) -> str:
    """Reduce a noisy "Area" match to the trailing place name.

    Anchors on the LAST capitalized word ("Lekki" in "for sale in
    Lekki"), keeps capitalized/digit words before it ("Lekki Phase 1")
    and short lowercase/digit words after it ("Lekki phase 1"). Returns
    "" when nothing place-like survives.
    """
    words = [w.strip(" -'") for w in (raw or "").split()]
    words = [w for w in words if w]
    if not words:
        return ""
    anchor = -1
    for i in range(len(words) - 1, -1, -1):
        if words[i][0].isupper():
            anchor = i
            break
    if anchor < 0:
        return ""
    start = anchor
    while start > 0 and (
        words[start - 1][0].isupper() or words[start - 1][0].isdigit()
    ):
        start -= 1
    end = anchor
    while end + 1 < len(words) and (
        words[end + 1][0].isdigit()
        or (words[end + 1][0].islower() and len(words[end + 1]) <= 5)
    ):
        end += 1
    return " ".join(words[start:end + 1])


def _geo_location(text: str) -> str:
    """Pull "Area, State" out of free text; "" when none matches.

    When several candidates match, the shortest cleaned area wins.
    """
    best = ""
    for match in _LOCATION_GEO_RE.finditer(text or ""):
        area = _clean_area(match.group(1))
        if not area:
            continue
        candidate = f"{area}, {match.group(2)}"
        if not best or len(area) < len(best.split(",", 1)[0]):
            best = candidate
    return best


def _extract_detail_location(
    html: str, blocks: list[str], meta_description: str
) -> str:
    """The ad's location from a detail page, best effort.

    Layers, first hit wins: (1) labeled "Location:"/"Address:" lines,
    (2) elements whose class/id mentions location, (3) "Area, State"
    geography scanned over block texts and the meta description.
    Returns "" when the page genuinely carries no location.
    """
    for block in blocks:
        match = _LOCATION_LABEL_RE.search(block)
        if match:
            value = (match.group(1) or match.group(2) or "").strip()
            if value:
                return value
    collector = _LocationClassCollector()
    try:
        collector.feed(html or "")
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not kill the parse
        _log.warning("jiji: location HTML parse hit malformed markup: %s", exc)
    for text in collector.texts:
        cleaned = text.strip()
        if cleaned:
            return cleaned
    candidates = list(blocks)
    if meta_description:
        candidates.append(meta_description)
    for text in candidates:
        geo = _geo_location(text)
        if geo:
            return geo
    return ""


# ── price parsing ────────────────────────────────────────────────────


def parse_price_ngn(text: str) -> int | None:
    """Parse a Jiji price string into naira (int), or None when the seller
    hides the price ("Contact for price", "Swap", ...).

    >>> parse_price_ngn("₦1,200,000")
    1200000
    >>> parse_price_ngn("₦ 250,000")
    250000
    >>> parse_price_ngn("Contact for price") is None
    True
    """
    text = (text or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _NO_PRICE_MARKERS):
        return None
    match = re.search(r"₦\s*([\d][\d,\.\s]*)", text)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


# ── rendered-HTML parsing ────────────────────────────────────────────

#: block-level tags that start a new text line inside a listing card.
_BLOCK_TAGS = {"div", "p", "li", "h1", "h2", "h3", "h4", "br", "section",
               "article", "header", "footer"}


class _AnchorCollector(HTMLParser):
    """Collects every <a> with its absolute href, inner text (block
    boundaries kept as newlines) and first nested <img> src."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base = base_url
        self.anchors: list[dict[str, str]] = []
        self._current: dict[str, Any] | None = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        if tag == "a":
            href = (attrs_d.get("href") or "").strip()
            if self._current is None and href:
                self._current = {
                    "href": urllib.parse.urljoin(self._base, href),
                    "chunks": [],
                    "img": "",
                }
                self._depth = 1
            elif self._current is not None:
                self._depth += 1
            return
        if self._current is not None:
            self._depth += 1
            if tag == "img" and not self._current["img"]:
                src = (attrs_d.get("src") or attrs_d.get("data-src") or "").strip()
                if src:
                    self._current["img"] = urllib.parse.urljoin(self._base, src)
            if tag in _BLOCK_TAGS:
                self._current["chunks"].append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing tags (<br/>, <img/>): capture without disturbing
        # the anchor depth accounting.
        if tag == "img" and self._current is not None:
            attrs_d = dict(attrs)
            src = (attrs_d.get("src") or attrs_d.get("data-src") or "").strip()
            if src and not self._current["img"]:
                self._current["img"] = urllib.parse.urljoin(self._base, src)
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag in _BLOCK_TAGS:
            self._current["chunks"].append("\n")
        self._depth -= 1
        if self._depth <= 0:
            raw = "".join(self._current["chunks"])
            lines = [ln.strip() for ln in raw.split("\n")]
            text = "\n".join(ln for ln in lines if ln)
            self.anchors.append({
                "href": self._current["href"],
                "text": text,
                "img": self._current["img"],
            })
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current["chunks"].append(data)


def is_listing_url(url: str) -> bool:
    """True for Jiji ad URLs: jiji.ng/<location>/<category>/<slug>-<id>.html.

    Category/section links (``/cars``, ``/mobile-phones``) and the search
    page itself never match: they lack the trailing numeric ``.html`` slug.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.netloc not in _JIJI_HOSTS:
        return False
    if not parsed.path.endswith(".html"):
        return False
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) < 2:
        return False
    return bool(re.search(r"\d", segments[-1]))


#: lines that are metadata rather than places — never chosen as a card
#: location (dates, phone numbers, "posted x ago" markers)
_NON_PLACE_RE = re.compile(
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}"  # 12 March 2026
    r"|\d{4}-\d{2}-\d{2}"  # 2026-03-12
    r"|\d{2,}/\d{2,}(/\d{2,})?"  # 12/03/2026
    r"|\+?\d[\d\s\-()]{7,}"  # phone-ish digit runs
    r"|\b(ago|yesterday|today)\b)",
    re.IGNORECASE,
)


def _split_card_text(text: str) -> tuple[str, str, str]:
    """Split a card's text into (price_text, title, location), best effort.

    The price line is the line carrying ₦ or a no-price marker; the title is
    the longest remaining line; the location is the longest short remaining
    line (Jiji cards print location under the title, e.g. "Lekki, Lagos"),
    skipping date/phone/metadata lines that are never places.
    """
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    price_text = ""
    rest: list[str] = []
    for line in lines:
        if not price_text and ("₦" in line or any(
                m in line.lower() for m in _NO_PRICE_MARKERS)):
            price_text = line
        else:
            rest.append(line)
    title = max(rest, key=len, default="")
    rest_wo_title = [ln for ln in rest if ln != title]
    location = max(
        (ln for ln in rest_wo_title
         if len(ln) <= 60 and not _NON_PLACE_RE.search(ln)),
        key=len,
        default="",
    )
    return price_text, title, location


def extract_listing_cards(html: str, base_url: str = JIJI_BASE) -> list[dict[str, Any]]:
    """Pull listing cards out of rendered search/homepage HTML.

    Returns one dict per card: {title, price_ngn, price_text, url,
    location, image}. Anchors that are not listing URLs are skipped.
    """
    collector = _AnchorCollector(base_url)
    try:
        collector.feed(html or "")
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not kill the parse
        _log.warning("jiji: HTML parse hit malformed markup: %s", exc)
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in collector.anchors:
        url = anchor["href"].split("#", 1)[0]
        if not is_listing_url(url) or url in seen:
            continue
        seen.add(url)
        price_text, title, location = _split_card_text(anchor["text"])
        if not title:
            continue
        cards.append({
            "title": title,
            "price_ngn": parse_price_ngn(price_text),
            "price_text": price_text,
            "url": url,
            "location": location,
            "image": anchor["img"],
            "seller_name": "",
        })
    return cards


def extract_categories(html: str, base_url: str = JIJI_BASE) -> list[dict[str, str]]:
    """Category links from the homepage nav: single-segment jiji.ng paths
    (``/cars``, ``/mobile-phones-tablets``, ...) with link text."""
    collector = _AnchorCollector(base_url)
    try:
        collector.feed(html or "")
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not kill the parse
        _log.warning("jiji: HTML parse hit malformed markup: %s", exc)
    cats: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in collector.anchors:
        try:
            parsed = urllib.parse.urlparse(anchor["href"])
        except ValueError:
            continue
        if parsed.netloc not in _JIJI_HOSTS:
            continue
        segments = [s for s in parsed.path.split("/") if s]
        if len(segments) != 1 or parsed.path.endswith(".html"):
            continue
        name = " ".join(anchor["text"].split())
        if not name or len(name) > 60:
            continue
        url = anchor["href"].split("#", 1)[0]
        if url in seen:
            continue
        seen.add(url)
        cats.append({"name": name, "url": url})
    return cats


class _BlockTextCollector(HTMLParser):
    """Visible text of each block-level element as a separate string
    (entities decoded) — used to find labeled fields like "Seller: X"
    without one block's text bleeding into the next."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._chunks: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS and self._depth:
            self._depth -= 1
            if self._depth == 0:
                text = " ".join("".join(self._chunks).split())
                if text:
                    self.blocks.append(text)
                self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._chunks.append(data)


def _collect_blocks(html: str) -> list[str]:
    collector = _BlockTextCollector()
    try:
        collector.feed(html or "")
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not kill the parse
        _log.warning("jiji: detail HTML parse hit malformed markup: %s", exc)
    return collector.blocks


class _H1Collector(HTMLParser):
    """First <h1> text on the page (the listing title on ad pages)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_h1 = False
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h1" and not self.title:
            self._in_h1 = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self._in_h1:
            self._in_h1 = False
            self.title = " ".join("".join(self._chunks).split())
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_h1:
            self._chunks.append(data)


class _ImgCollector(HTMLParser):
    """All <img> srcs in document order (ad gallery images)."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base = base_url
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "img":
            return
        attrs_d = dict(attrs)
        src = (attrs_d.get("src") or attrs_d.get("data-src") or "").strip()
        if not src or src.startswith("data:"):
            return
        absolute = urllib.parse.urljoin(self._base, src)
        if absolute not in self.images:
            self.images.append(absolute)


class _MetaCollector(HTMLParser):
    """<title> and <meta name=description> for detail pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page_title = ""
        self.meta_description = ""
        self._in_title = False
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            attrs_d = dict(attrs)
            if attrs_d.get("name", "").lower() == "description":
                self.meta_description = (attrs_d.get("content") or "").strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._in_title:
            self._in_title = False
            self.page_title = " ".join("".join(self._chunks).split())

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._chunks.append(data)


def parse_listing_detail(html: str, url: str) -> dict[str, Any]:
    """Parse an ad detail page. Fails fast when the page has no listing
    title — callers must not mistake a challenge/block page for an ad."""
    h1 = _H1Collector()
    try:
        h1.feed(html or "")
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not kill the parse
        _log.warning("jiji: detail HTML parse hit malformed markup: %s", exc)
    if not h1.title:
        raise JijiError(
            f"jiji page structure changed: no listing title found at {url} "
            "(the page may be a Cloudflare challenge or a removed ad)"
        )
    meta = _MetaCollector()
    try:
        meta.feed(html or "")
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not kill the parse
        _log.warning("jiji: detail HTML parse hit malformed markup: %s", exc)
    imgs = _ImgCollector(url)
    try:
        imgs.feed(html or "")
    except Exception as exc:  # noqa: BLE001 - malformed HTML must not kill the parse
        _log.warning("jiji: detail HTML parse hit malformed markup: %s", exc)

    blocks = _collect_blocks(html)

    price_text = ""
    for block in blocks:
        price_match = re.search(r"₦\s*[\d][\d,\.\s]*", block)
        if price_match:
            price_text = price_match.group(0).strip()
            break
    posted = ""
    for block in blocks:
        if "posted" not in block.lower():
            continue
        posted_match = re.search(
            r"(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2})", block
        )
        if posted_match:
            posted = posted_match.group(1)
            break
    seller = ""
    for block in blocks:
        seller_match = re.match(r"[Ss]eller\s*:\s*(.+)", block)
        if seller_match:
            seller = seller_match.group(1).strip()
            break

    location = _extract_detail_location(
        html, blocks, meta.meta_description
    )

    return {
        "title": h1.title,
        "price_ngn": parse_price_ngn(price_text),
        "price_text": price_text,
        "description": meta.meta_description,
        "seller": seller,
        "location": location,
        "images": imgs.images[:20],
        "posted_date": posted,
        "url": url,
    }


@register_connector
class JijiConnector(Connector):
    """Devon's Jiji adapter: public Nigerian classifieds via rendered tabs.

    No API, no auth — every page load goes through a real headless
    Chromium tab (``BrowserService.open_rendered_tab``) because Cloudflare
    blocks plain HTTP fetching. Seller contact and login are
    human-in-the-loop only.
    """

    id = "jiji"
    name = "Jiji"
    description = (
        "Nigerian classifieds (jiji.ng): search public listings, read ad "
        "details, and list categories through rendered browser tabs "
        "(Cloudflare blocks plain HTTP fetch). No API and no auth — "
        "seller contact and login are human-in-the-loop only."
    )
    auth_methods = (AuthMethod.NONE,)
    PROVISIONABLE: tuple[str, ...] = ()

    _SESSION_NAME = "jiji"

    def __init__(self, vault: Any, http: Any = None,
                 browser: BrowserService | None = None) -> None:
        super().__init__(vault, http=http)
        self._browser_override = browser
        self._browser_svc: BrowserService | None = None
        self._tab_ids: list[str] = []
        self._connected = False
        self._search_url_pattern: str = ""
        self._last_request_at = 0.0
        self._cache: dict[str, tuple[float, Any]] = {}

    # ── browser plumbing ─────────────────────────────────────────

    def _browser(self) -> BrowserService:
        if self._browser_override is not None:
            return self._browser_override
        if self._browser_svc is None:
            from pathlib import Path

            self._browser_svc = BrowserService(
                data_dir=Path.home() / ".nomorals" / "connectors" / "jiji"
            )
        return self._browser_svc

    def _throttle(self) -> None:
        """Enforce the minimum gap between page loads (Jiji is aggressive)."""
        now = time.monotonic()
        wait = self._last_request_at + _MIN_REQUEST_GAP_S - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        self._last_request_at = now

    def _render(self, url: str) -> tuple[RenderedTab, str]:
        """Throttled: open a rendered tab, navigate, return (tab, html).

        The caller owns the tab and must hand it to _close_tab(). A failed
        navigation raises JijiError — never a half-loaded page.
        """
        self._throttle()
        service = self._browser()
        try:
            tab = service.open_rendered_tab(self._SESSION_NAME, url)
        except Exception as exc:  # noqa: BLE001 - tab errors are opaque here
            raise JijiError(f"jiji could not render {url}: {exc}") from exc
        self._tab_ids.append(tab.tab_id)
        try:
            html = tab.html()["html"]
        except Exception as exc:  # noqa: BLE001 - extraction errors are opaque
            self._close_tab(tab)
            raise JijiError(f"jiji could not read rendered HTML of {url}: {exc}") from exc
        return tab, html

    def _close_tab(self, tab: RenderedTab) -> None:
        try:
            self._browser().close_rendered_tab(tab.tab_id)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask results
            _log.warning("jiji: closing rendered tab failed: %s", exc)
        finally:
            if tab.tab_id in self._tab_ids:
                self._tab_ids.remove(tab.tab_id)

    # ── cache ────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Any:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > _CACHE_TTL_S:
            del self._cache[key]
            return None
        return value

    def _cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(self, **kwargs: Any) -> ConnectResult:
        """Load the homepage in a rendered tab and verify listings render.

        No credentials — this just proves the rendered pipeline reaches
        jiji.ng and the card parser still matches the site's markup.
        """
        tab, html = self._render(JIJI_BASE)
        try:
            cards = extract_listing_cards(html)
            if not cards:
                raise JijiError(
                    "jiji page structure changed: the homepage rendered "
                    "but no listing cards were found — the connector's "
                    "card parser no longer matches jiji.ng markup"
                )
        finally:
            self._close_tab(tab)
        self._connected = True
        _log.info("jiji connected: homepage rendered with %d listing cards",
                  len(cards))
        return ConnectResult(
            ok=True,
            account="jiji.ng public",
            scopes=[],
            message=(
                f"connected to jiji.ng (public listings, no login). The "
                f"homepage rendered with {len(cards)} listing cards. "
                "Seller contact and login stay human-in-the-loop."
            ),
        )

    def disconnect(self) -> None:
        for tab_id in list(self._tab_ids):
            try:
                self._browser().close_rendered_tab(tab_id)
            except Exception as exc:  # noqa: BLE001 - teardown best effort
                _log.warning("jiji: closing rendered tab %s failed: %s",
                             tab_id, exc)
        self._tab_ids.clear()
        self._connected = False

    def status(self) -> ConnectorStatus:
        if not self._connected:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect --name jiji`",
            )
        return ConnectorStatus(
            connected=True,
            account="jiji.ng public",
            last_checked=time.time(),
            detail="public listings via rendered browser tab (no login)",
        )

    def test_connection(self) -> bool:
        if not self._connected:
            return False
        try:
            tab, html = self._render(JIJI_BASE)
            try:
                return bool(extract_listing_cards(html))
            finally:
                self._close_tab(tab)
        except JijiError:
            return False

    # ── search ───────────────────────────────────────────────────

    def _search_pattern(self) -> str:
        """The search URL pattern, re-discovered from the homepage when not
        cached. Fail fast when the homepage no longer exposes a search."""
        if self._search_url_pattern:
            return self._search_url_pattern
        tab, html = self._render(JIJI_BASE)
        try:
            pattern = self._discover_search_pattern(html)
        finally:
            self._close_tab(tab)
        self._search_url_pattern = pattern
        return pattern

    @staticmethod
    def _discover_search_pattern(html: str) -> str:
        """Find the site's search submission target in homepage HTML.

        Prefers a search <form>'s action; falls back to a /search link;
        falls back to the canonical pattern. Raises JijiError when none of
        them is present — the site structure changed.
        """
        forms = re.findall(
            r'<form[^>]*action="([^"]*)"[^>]*>.*?</form>',
            html or "", flags=re.IGNORECASE | re.DOTALL,
        )
        for action in forms:
            absolute = urllib.parse.urljoin(JIJI_BASE, action.strip())
            if "search" in urllib.parse.urlparse(absolute).path.lower():
                return absolute + (
                    "&" if "?" in absolute else "?") + "query={query}"
        links = re.findall(r'href="([^"]*search[^"]*)"', html or "",
                           flags=re.IGNORECASE)
        for href in links:
            absolute = urllib.parse.urljoin(JIJI_BASE, href.strip())
            parsed = urllib.parse.urlparse(absolute)
            if parsed.netloc in _JIJI_HOSTS:
                base = absolute.split("?", 1)[0]
                return base + "?query={query}"
        if "/search" in (html or "").lower():
            return JIJI_SEARCH_URL
        raise JijiError(
            "jiji page structure changed: no search form or /search link "
            "found on the homepage"
        )

    def _build_search_url(self, query: str, category: str = "",
                          location: str = "") -> str:
        pattern = self._search_pattern()
        url = pattern.replace("{query}", urllib.parse.quote_plus(query))
        extra: dict[str, str] = {}
        if category.strip():
            extra["category"] = category.strip()
        if location.strip():
            extra["location"] = location.strip()
        if extra:
            sep = "&" if "?" in url else "?"
            url += sep + urllib.parse.urlencode(extra)
        return url

    def search_listings(
        self,
        query: str,
        *,
        limit: int = 20,
        category: str = "",
        location: str = "",
    ) -> list[dict[str, Any]]:
        """Search public listings. Returns [{title, price_ngn, url,
        location, image, seller_name}] — seller_name is empty on search
        cards (only ad pages show it); use get_listing() for the seller.

        ``category``/``location`` are appended to the search URL as query
        params (best effort — the site ignores params it doesn't support);
        ``location`` additionally filters the parsed cards client-side.
        Fails fast with JijiError when no listing cards are found
        (structure changed) — never a silent empty list.
        """
        query = (query or "").strip()
        if not query:
            raise JijiError("search_listings needs a query")
        if limit <= 0:
            raise JijiError("search_listings needs limit > 0")
        cache_key = (
            f"search:{query}:{limit}:{category.strip().lower()}:"
            f"{location.strip().lower()}"
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        url = self._build_search_url(query, category, location)
        tab, html = self._render(url)
        try:
            cards = extract_listing_cards(html, base_url=url)
            if not cards:
                raise JijiError(
                    "jiji page structure changed: the search page rendered "
                    f"but no listing cards were found for {query!r} — the "
                    "connector's card parser no longer matches jiji.ng "
                    "markup"
                )
            if location.strip():
                needle = location.strip().lower()
                cards = [c for c in cards
                         if needle in (c.get("location") or "").lower()]
            results = [
                {
                    "title": c["title"],
                    "price_ngn": c["price_ngn"],
                    "url": c["url"],
                    "location": c["location"],
                    "image": c["image"],
                    "seller_name": c["seller_name"],
                }
                for c in cards[:limit]
            ]
        finally:
            self._close_tab(tab)
        self._cache_set(cache_key, results)
        return results

    # ── listing detail ───────────────────────────────────────────

    def get_listing(self, url: str) -> dict[str, Any]:
        """One ad's detail: {title, price_ngn, price_text, description,
        seller, location, images, posted_date, url}."""
        url = (url or "").strip()
        if not url:
            raise JijiError("get_listing needs a url")
        cached = self._cache_get(f"listing:{url}")
        if cached is not None:
            return cached
        tab, html = self._render(url)
        try:
            detail = parse_listing_detail(html, url)
        finally:
            self._close_tab(tab)
        self._cache_set(f"listing:{url}", detail)
        return detail

    # ── categories ───────────────────────────────────────────────

    def list_categories(self) -> list[dict[str, str]]:
        """Homepage category nav: [{name, url}]."""
        cached = self._cache_get("categories")
        if cached is not None:
            return cached
        tab, html = self._render(JIJI_BASE)
        try:
            cats = extract_categories(html)
            if not cats:
                raise JijiError(
                    "jiji page structure changed: no category links found "
                    "in the homepage nav"
                )
        finally:
            self._close_tab(tab)
        self._cache_set("categories", cats)
        return cats

    # ── human-in-the-loop: contact seller / login ────────────────

    def contact_seller(
        self,
        listing_url: str,
        message: str,
        *,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Ask the owner to contact a seller personally.

        Devon never messages sellers itself: this raises a MANUAL_STEP
        checkpoint with the listing link and the owner's message; the flow
        resumes only after the owner confirms they sent it.
        """
        from .checkpoints import CheckpointKind

        listing_url = (listing_url or "").strip()
        if not listing_url:
            raise JijiError("contact_seller needs a listing url")
        if not (message or "").strip():
            raise JijiError("contact_seller needs the message to send")
        if db is None:
            raise JijiError(
                "contact_seller needs a database for checkpoints (pass db=)"
            )
        cp = self.request_human(
            CheckpointKind.MANUAL_STEP,
            "Contact the Jiji seller",
            "\n".join([
                "Devon does not message sellers on your behalf — contacting "
                "a stranger is a human step. Your part:",
                f"1. Open the listing: {listing_url}",
                "2. Tap 'Show contact' / 'Chat' on the ad and sign in to "
                "your own Jiji account if asked.",
                "3. Send this message (yours to edit):",
                "",
                message.strip(),
                "",
                "Resolve this checkpoint once you have sent the message.",
            ]),
            db=db,
            context=context,
            resume_state={
                "stage": "contact_seller",
                "listing_url": listing_url,
                "message": message.strip(),
            },
        )
        return self.resume_checkpoint(cp, db=db, context=context)

    def login(self, *, db: Any = None, context: Any = None) -> dict[str, Any]:
        """Guide the owner through signing in to their own Jiji account.

        The owner types their own credentials into the rendered tab
        themselves — Devon never sees or stores them. Cookies persist in
        the session's playwright storage_state afterwards.
        """
        from .checkpoints import CheckpointKind

        if db is None:
            raise JijiError(
                "login needs a database for checkpoints (pass db=)"
            )
        cp = self.request_human(
            CheckpointKind.MANUAL_STEP,
            "Sign in to your Jiji account",
            "\n".join([
                "Devon never handles your Jiji credentials — this step is "
                "yours alone:",
                "1. Devon has opened jiji.ng in a rendered tab for you.",
                "2. Sign in with your own Jiji account (phone/email) and "
                "solve any verification yourself.",
                "3. Your session cookies stay in this machine's browser "
                "session storage — nothing is sent to Devon.",
                "Resolve this checkpoint once you are signed in.",
            ]),
            db=db,
            context=context,
            resume_state={"stage": "login"},
        )
        return self.resume_checkpoint(cp, db=db, context=context)

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Continue a human-in-the-loop flow after the owner resolves it."""
        from .checkpoints import CheckpointKind, CheckpointState

        if checkpoint.state != CheckpointState.RESOLVED:
            raise JijiError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must complete the human step first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        if stage == "contact_seller":
            return {
                "done": True,
                "identity": "owner",
                "listing_url": (checkpoint.resume_state or {}).get("listing_url", ""),
                "message": (
                    "Owner confirmed the seller was contacted personally. "
                    "Devon sent nothing itself."
                ),
            }
        if stage == "login":
            return {
                "done": True,
                "identity": "owner",
                "account": "owner's Jiji account",
                "message": (
                    "Owner signed in personally; session cookies persist in "
                    "the rendered-tab storage for this machine only."
                ),
            }
        raise JijiError(
            f"jiji cannot resume checkpoint stage {stage!r} "
            f"(kind: {CheckpointKind.MANUAL_STEP.value})"
        )
