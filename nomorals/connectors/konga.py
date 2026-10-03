"""Konga connector — buyer-side browsing of konga.com via structured data.

What exists (verified 2026-10-02, plain HTTPS probe):

* **No public Konga API.** Konga publishes no product-search, cart, or
  order API. All buyer flows are web-only.
* **Server-rendered pages carrying JSON-LD.** ``/search?search=<query>``
  returns HTTP 200 with server-rendered markup (~260KB) embedding
  ``<script type="application/ld+json">`` blocks with schema.org
  **Product** entries. Each Product has: ``@id`` / ``url`` of the form
  ``https://www.konga.com/product/<slug>-<numeric-id>``, ``name`` (full
  product title), ``image`` (Cloudinary URL), and
  ``offers: {@type: "Offer", price: <number>, priceCurrency: "NGN",
  availability: "https://schema.org/InStock"}``.
* Product detail pages at ``/product/<slug>-<id>`` carry the same JSON-LD.

What this connector does:

* Drives those pages through Devon's browser service
  (``nomorals.browser.service.BrowserService``) and parses ONLY the
  JSON-LD — never the HTML product cards, which are far more fragile.
  If a page yields zero Product entries, it raises immediately
  ("structure changed") instead of silently returning [].
* No auth at all: public browsing needs no login, so
  ``auth_methods = (AuthMethod.NONE,)``.
* Rate limiting: minimum 2 seconds between requests to konga.com,
  enforced in code. Search results are cached in memory with a
  15-minute TTL.

What does NOT exist (so this connector honestly cannot do):

* **No order automation.** Konga checkout is web-only and checkout
  flows are intentionally human. :meth:`place_order` opens a
  human-in-the-loop checkpoint telling the owner to complete the
  purchase in their own browser — Devon never submits orders itself.
* No seller-side API, no price-history endpoint, no cart API.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["KongaConnector", "KongaError", "KONGA_BASE", "KONGA_SEARCH_URL"]

_log = get_logger(__name__)

#: Konga storefront.
KONGA_BASE = "https://www.konga.com"
KONGA_SEARCH_URL = f"{KONGA_BASE}/search"

#: Browser-like UA for the direct-HTTP fallback path (public pages only,
#: no cookies needed). The browser-service path uses its own UA.
KONGA_HTTP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

#: Minimum seconds between consecutive requests to konga.com.
_MIN_REQUEST_INTERVAL = 2.0

#: In-memory search cache TTL, seconds.
_CACHE_TTL = 15 * 60.0

#: ld+json script extraction.
_LD_JSON_RE = re.compile(
    r'<script[^>]*type\s*=\s*["\']application/ld\+json["\'][^>]*>'
    r"(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)

_AVAILABILITY_MAP = {
    "instock": "in_stock",
    "outofstock": "out_of_stock",
    "preorder": "preorder",
    "limitedavailability": "limited_availability",
    "discontinued": "discontinued",
}


class KongaError(ConnectorError):
    """A Konga browse operation failed (network, parse, or structure)."""

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


@register_connector
class KongaConnector(Connector):
    """Devon's Konga adapter: public product search/browse on konga.com.

    Auth-free. Drives pages through Devon's browser service and parses
    the embedded schema.org JSON-LD Product blocks. Orders are never
    automated — they go through a human checkpoint.
    """

    id = "konga"
    name = "Konga"
    description = (
        "Public buyer-side browsing of Konga (Nigerian marketplace): "
        "search products, read product details, list categories, and "
        "find today's deals — parsed from the site's JSON-LD, no login. "
        "Konga has no public API and no checkout automation."
    )
    auth_methods = (AuthMethod.NONE,)
    PROVISIONABLE = ()

    # ── lifecycle ────────────────────────────────────────────────

    def __init__(
        self,
        vault: Any,
        http: Any = None,
        *,
        browser_service: Any = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(vault, http)
        #: BrowserService instance (real one built lazily); tests inject a
        #: fake here so no browser machinery is touched.
        self._browser_service = browser_service
        self._session: Any = None
        #: injectable clock/sleep for rate-limit tests.
        self._clock = clock or time.monotonic
        self._sleep = sleeper or time.sleep
        self._last_request_at = 0.0
        #: query.lower() -> (expires_at, products)
        self._search_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        #: product url -> (expires_at, detail)
        self._detail_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._connected = False
        self._last_ok = 0.0

    def connect(self) -> ConnectResult:
        """Verify Konga is reachable and its product JSON-LD parses.

        No credentials exist for Konga — "connecting" means the homepage
        loads and carries Product structured data, i.e. the site's shape
        still matches what this connector parses.
        """
        products = self._homepage_products()
        self._connected = True
        self._last_ok = time.time()
        _log.info("konga connected: homepage carries %d product blocks",
                  len(products))
        return ConnectResult(
            ok=True,
            account="konga.com public",
            scopes=[],
            message=(
                f"connected to Konga public browsing (konga.com). "
                f"Homepage check: {len(products)} product JSON-LD blocks "
                f"parsed — search and browse are available. No login, no "
                f"order automation: checkout is human-only."
            ),
        )

    def disconnect(self) -> None:
        service, self._browser_service = self._browser_service, None
        if service is not None and self._session is not None:
            try:
                service.close_session("konga")
            except Exception as exc:  # noqa: BLE001 - teardown is best-effort
                _log.debug("konga session close failed: %s", exc)
        self._session = None
        self._search_cache.clear()
        self._detail_cache.clear()
        self._connected = False
        self._last_ok = 0.0

    def status(self) -> ConnectorStatus:
        if not self._connected:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect --name konga`",
            )
        if self.test_connection():
            return ConnectorStatus(
                connected=True,
                account="konga.com public",
                last_checked=time.time(),
                detail="homepage product JSON-LD present",
            )
        return ConnectorStatus(
            connected=False,
            account="konga.com public",
            last_checked=time.time(),
            detail=(
                "konga homepage check failed — the site may have changed "
                "shape; reconnect to re-verify"
            ),
        )

    def test_connection(self) -> bool:
        try:
            self._homepage_products()
        except ConnectorError:
            return False
        self._last_ok = time.time()
        return True

    # ── browse: search ───────────────────────────────────────────

    def search_products(
        self, query: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search Konga products (``/search?search=<query>``).

        Parses the page's JSON-LD Product blocks only. Raises
        KongaError("konga search page structure changed") when the page
        yields zero Product entries — fail fast, never silent [].
        """
        query = (query or "").strip()
        if not query:
            raise KongaError("search_products needs a query")
        if limit <= 0:
            raise KongaError("search_products limit must be positive")
        cached = self._cache_get(self._search_cache, query.lower())
        if cached is not None:
            return cached[:limit]
        url = f"{KONGA_SEARCH_URL}?{urllib.parse.urlencode({'search': query})}"
        html = self._page_html(url)
        entries = self._product_entries(html)
        if not entries:
            raise KongaError(
                "konga search page structure changed: no JSON-LD Product "
                f"entries found for query {query!r}"
            )
        products = [self._normalize_product(e) for e in entries]
        products = [p for p in products if p]
        if not products:
            raise KongaError(
                "konga search page structure changed: Product entries "
                "present but none had a name and price"
            )
        self._cache_set(self._search_cache, query.lower(), products)
        return products[:limit]

    def get_product(self, url_or_id: str) -> dict[str, Any]:
        """Product detail for a ``/product/<slug>-<id>`` URL or bare slug.

        Returns name, price_ngn, currency, availability, seller, brand,
        image, url. Raises KongaError("structure changed") when the page
        carries no Product JSON-LD.
        """
        url = self._normalize_product_url(url_or_id)
        cached = self._cache_get(self._detail_cache, url)
        if cached is not None:
            return dict(cached)
        html = self._page_html(url)
        entries = self._product_entries(html)
        if not entries:
            raise KongaError(
                "konga product page structure changed: no JSON-LD Product "
                f"entry found at {url}"
            )
        detail = self._normalize_detail(entries[0], url)
        self._cache_set(self._detail_cache, url, detail)
        return dict(detail)

    def list_categories(self) -> list[dict[str, str]]:
        """Category names/urls from the homepage navigation links."""
        tab = self._open_tab(KONGA_BASE)
        try:
            result = tab.links()
        except Exception as exc:  # noqa: BLE001 - delegate wraps ToolError
            raise KongaError(f"konga homepage links failed: {exc}") from exc
        finally:
            self._close_tab(tab)
        links = result.get("links", []) if isinstance(result, dict) else []
        seen: set[str] = set()
        categories: list[dict[str, str]] = []
        for link in links:
            if not isinstance(link, dict):
                continue
            href = str(link.get("url", ""))
            text = str(link.get("text", "")).strip()
            if not text or "/category/" not in urllib.parse.urlparse(href).path:
                continue
            key = href.split("#", 1)[0].rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            categories.append({"name": text, "url": href})
        if not categories:
            raise KongaError(
                "konga homepage structure changed: no /category/ links "
                "in the homepage navigation"
            )
        return categories

    def deals(self) -> list[dict[str, Any]]:
        """Today's deals: discovers the deals page from the homepage nav.

        Raises KongaError when no deals page is discoverable — never
        guesses a URL.
        """
        tab = self._open_tab(KONGA_BASE)
        try:
            result = tab.links()
        except Exception as exc:  # noqa: BLE001 - delegate wraps ToolError
            raise KongaError(f"konga homepage links failed: {exc}") from exc
        finally:
            self._close_tab(tab)
        links = result.get("links", []) if isinstance(result, dict) else []
        deals_url = ""
        for link in links:
            if not isinstance(link, dict):
                continue
            href = str(link.get("url", ""))
            text = str(link.get("text", "")).strip()
            if "deal" not in text.lower():
                continue
            parsed = urllib.parse.urlparse(href)
            if parsed.netloc.endswith("konga.com") and parsed.scheme in {
                "http", "https",
            }:
                deals_url = href
                break
        if not deals_url:
            raise KongaError(
                "konga has no deals page discoverable from the homepage "
                "navigation — refusing to guess a URL"
            )
        html = self._page_html(deals_url)
        entries = self._product_entries(html)
        if not entries:
            raise KongaError(
                "konga deals page structure changed: no JSON-LD Product "
                "entries found"
            )
        products = [p for p in (self._normalize_product(e) for e in entries) if p]
        if not products:
            raise KongaError(
                "konga deals page structure changed: Product entries "
                "present but none had a name and price"
            )
        return products

    # ── orders: human-only ───────────────────────────────────────

    def place_order(
        self,
        items: list[dict[str, Any]],
        *,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Ask the owner to complete a Konga purchase in their own browser.

        Konga has no order API and checkout automation is off the table —
        this raises a human checkpoint (``MANUAL_STEP``) with the cart and
        pauses until the owner completes the purchase personally.
        """
        from .checkpoints import CheckpointKind

        if not items:
            raise KongaError("place_order needs at least one item")
        if db is None:
            raise KongaError(
                "place_order needs a database for the human checkpoint "
                "(pass db=)"
            )
        cart_lines = []
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                raise KongaError(f"place_order item #{i} must be a dict")
            name = str(item.get("name", "")).strip()
            url = str(item.get("url", "")).strip()
            if not name:
                raise KongaError(f"place_order item #{i} needs a name")
            line = f"{i}. {name}"
            if url:
                line += f" — {url}"
            cart_lines.append(line)
        cp = self.request_human(
            CheckpointKind.MANUAL_STEP,
            "Complete your Konga purchase",
            "\n".join([
                "Devon cannot place Konga orders — Konga has no order API "
                "and checkout is web-only. The human part:",
                "1. Open konga.com in your own browser and sign in.",
                f"2. Add these {len(items)} item(s) to your cart:",
                *("   " + line for line in cart_lines),
                "3. Complete checkout and payment yourself.",
                "Resolve this checkpoint once the order is placed.",
            ]),
            db=db,
            context=context,
            resume_state={"stage": "owner_checkout", "items": list(items)},
        )
        return self.resume_checkpoint(cp, db=db, context=context)

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
    ) -> dict[str, Any]:
        """Continue after the owner completes their Konga checkout."""
        from .checkpoints import CheckpointState

        if checkpoint.state != CheckpointState.RESOLVED:
            raise KongaError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — the owner must complete the purchase first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        if stage != "owner_checkout":
            raise KongaError(
                f"konga cannot resume checkpoint stage {stage!r}"
            )
        items = (checkpoint.resume_state or {}).get("items", [])
        return {
            "done": True,
            "items": items,
            "message": (
                "Owner completed the Konga purchase in their own browser — "
                "Devon did not automate any checkout step."
            ),
        }

    # ── page plumbing ────────────────────────────────────────────

    def _browser_or_none(self) -> Any:
        """The injected or lazily-built BrowserService; None if unavailable."""
        if self._browser_service is not None:
            return self._browser_service
        try:
            from ..browser.service import BrowserService
        except Exception as exc:  # noqa: BLE001 - browser is optional here
            _log.debug("konga: no BrowserService available: %s", exc)
            return None
        try:
            self._browser_service = BrowserService()
        except Exception as exc:  # noqa: BLE001
            _log.debug("konga: BrowserService() failed: %s", exc)
            return None
        return self._browser_service

    def _ensure_session(self, service: Any) -> Any:
        if self._session is not None:
            return self._session
        try:
            self._session = service.open_session("konga")
        except Exception as exc:  # noqa: BLE001 - session may already exist
            _log.debug("konga open_session failed (%s); trying get_session", exc)
            self._session = service.get_session("konga")
        return self._session

    def _open_tab(self, url: str) -> Any:
        """One tab on the konga session, navigated to ``url``."""
        service = self._browser_or_none()
        if service is None:
            raise KongaError(
                "konga browser service is unavailable and the direct-HTTP "
                "fallback does not expose tab links"
            )
        session = self._ensure_session(service)
        try:
            tab = session.open_tab(url)
        except Exception as exc:  # noqa: BLE001 - open wraps ToolError
            raise KongaError(f"konga could not open {url}: {exc}") from exc
        if getattr(tab, "error", ""):
            raise KongaError(f"konga could not open {url}: {tab.error}")
        return tab

    def _close_tab(self, tab: Any) -> None:
        try:
            self._session.close_tab(tab.tab_id)
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass

    def _page_html(self, url: str) -> str:
        """Raw HTML for ``url``: browser service first, direct HTTP fallback.

        Public pages need no cookies, so a browser-UA GET is a legitimate
        fallback when the browser machinery is unavailable.
        """
        self._pace()
        service = self._browser_or_none()
        if service is not None:
            try:
                tab = self._open_tab(url)
                try:
                    result = tab.html()
                finally:
                    self._close_tab(tab)
                html = str(result.get("html", "")) if isinstance(result, dict) else ""
                if not html:
                    raise KongaError(f"konga returned empty HTML for {url}")
                return html
            except KongaError:
                raise
            except Exception as exc:  # noqa: BLE001 - try HTTP fallback
                _log.debug("konga browser path failed (%s); HTTP fallback", exc)
        return self._page_html_via_http(url)

    def _page_html_via_http(self, url: str) -> str:
        try:
            resp = self.http.get(url, headers={"User-Agent": KONGA_HTTP_UA})
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise KongaError(f"konga request failed: {exc}") from exc
        if resp.status == 404:
            raise KongaError(f"konga page not found: {url}", status_code=404)
        if not resp.ok:
            raise KongaError(
                f"konga request failed ({resp.status}): {resp.text[:200]}",
                status_code=resp.status,
            )
        html = resp.text or ""
        if not html:
            raise KongaError(f"konga returned empty HTML for {url}")
        return html

    def _homepage_products(self) -> list[dict[str, Any]]:
        html = self._page_html(KONGA_BASE)
        entries = self._product_entries(html)
        if not entries:
            raise KongaError(
                "konga homepage structure changed: no JSON-LD Product "
                "entries found"
            )
        return [p for p in (self._normalize_product(e) for e in entries) if p]

    # ── rate limit + cache ───────────────────────────────────────

    def _pace(self) -> None:
        """Enforce the minimum 2s between requests to konga.com."""
        now = self._clock()
        wait = _MIN_REQUEST_INTERVAL - (now - self._last_request_at)
        if wait > 0:
            self._sleep(wait)
            now = self._clock()
        self._last_request_at = now

    def _cache_get(
        self, cache: dict[str, tuple[float, Any]], key: str
    ) -> Any | None:
        entry = cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del cache[key]
            return None
        return value

    def _cache_set(
        self, cache: dict[str, tuple[float, Any]], key: str, value: Any
    ) -> None:
        cache[key] = (self._clock() + _CACHE_TTL, value)

    # ── JSON-LD parsing ──────────────────────────────────────────

    @staticmethod
    def _product_entries(html: str) -> list[dict[str, Any]]:
        """All schema.org Product nodes from the page's ld+json blocks."""
        entries: list[dict[str, Any]] = []
        for match in _LD_JSON_RE.finditer(html or ""):
            try:
                block = json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                continue  # malformed block: skip, don't poison the page
            nodes = block if isinstance(block, list) else [block]
            for node in nodes:
                entries.extend(KongaConnector._find_products(node))
        return entries

    @staticmethod
    def _find_products(node: Any) -> list[dict[str, Any]]:
        """Recursively collect nodes whose @type includes Product."""
        found: list[dict[str, Any]] = []
        if isinstance(node, dict):
            types = node.get("@type", "")
            if isinstance(types, str):
                types = [types]
            if any(str(t).lower() == "product" for t in types):
                found.append(node)
            else:
                for value in node.values():
                    found.extend(KongaConnector._find_products(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(KongaConnector._find_products(item))
        return found

    @staticmethod
    def _first_offer(entry: dict[str, Any]) -> dict[str, Any]:
        offers = entry.get("offers")
        if isinstance(offers, list):
            offers = next((o for o in offers if isinstance(o, dict)), {})
        return offers if isinstance(offers, dict) else {}

    @staticmethod
    def _coerce_price(value: Any) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.]", "", value.strip())
            if cleaned and cleaned.count(".") <= 1:
                try:
                    return float(cleaned)
                except ValueError:
                    return None
        return None

    @staticmethod
    def _normalize_availability(value: Any) -> str:
        raw = str(value or "").strip().lower()
        token = raw.rsplit("/", 1)[-1].replace("_", "").replace("-", "")
        return _AVAILABILITY_MAP.get(token, "unknown")

    @classmethod
    def _normalize_product(
        cls, entry: dict[str, Any]
    ) -> dict[str, Any] | None:
        """One search/deals result. None when the entry lacks name/price."""
        name = str(entry.get("name", "")).strip()
        offer = cls._first_offer(entry)
        price = cls._coerce_price(offer.get("price"))
        if not name or price is None:
            return None
        url = str(entry.get("url") or entry.get("@id") or "").strip()
        image = entry.get("image", "")
        if isinstance(image, list):
            image = next((i for i in image if isinstance(i, str)), "")
        return {
            "name": name,
            "price_ngn": price,
            "url": url,
            "image": str(image or ""),
            "availability": cls._normalize_availability(
                offer.get("availability")
            ),
        }

    @classmethod
    def _normalize_detail(
        cls, entry: dict[str, Any], url: str
    ) -> dict[str, Any]:
        """Full product detail from the detail page's Product block."""
        offer = cls._first_offer(entry)
        seller = ""
        raw_seller = offer.get("seller")
        if isinstance(raw_seller, dict):
            seller = str(raw_seller.get("name", "")).strip()
        elif isinstance(raw_seller, str):
            seller = raw_seller.strip()
        brand = entry.get("brand", "")
        if isinstance(brand, dict):
            brand = str(brand.get("name", "")).strip()
        image = entry.get("image", "")
        if isinstance(image, list):
            image = next((i for i in image if isinstance(i, str)), "")
        return {
            "name": str(entry.get("name", "")).strip(),
            "price_ngn": cls._coerce_price(offer.get("price")),
            "currency": str(offer.get("priceCurrency", "") or "NGN").strip(),
            "availability": cls._normalize_availability(
                offer.get("availability")
            ),
            "seller": seller,
            "brand": str(brand or "").strip(),
            "image": str(image or ""),
            "url": str(entry.get("url") or entry.get("@id") or url).strip(),
        }

    @staticmethod
    def _normalize_product_url(url_or_id: str) -> str:
        """Full product URL from a URL, a slug, or a /product path."""
        value = (url_or_id or "").strip()
        if not value:
            raise KongaError(
                "get_product needs a product URL or slug "
                "(e.g. https://www.konga.com/product/<slug>-<id>)"
            )
        if re.fullmatch(r"\d+", value):
            raise KongaError(
                "get_product needs the product URL or slug — a bare "
                "numeric id is not enough to build the Konga URL"
            )
        if value.startswith(("http://", "https://")):
            parsed = urllib.parse.urlparse(value)
            if "konga.com" not in parsed.netloc:
                raise KongaError(
                    f"get_product only handles konga.com URLs, got {value!r}"
                )
            return value
        path = value if value.startswith("/") else f"/product/{value}"
        return f"{KONGA_BASE}{path}"


# ── JSON-LD fixture notes for tests ──────────────────────────────────────
#
# The canned HTML in tests/test_konga_connector.py mirrors the real
# konga.com search page shape probed 2026-10-02: multiple
# <script type="application/ld+json"> blocks, Product entries with @id +
# url = https://www.konga.com/product/<slug>-<numeric-id>, name, image
# (Cloudinary URL), and offers {@type Offer, price (number),
# priceCurrency "NGN", availability "https://schema.org/InStock"}.
