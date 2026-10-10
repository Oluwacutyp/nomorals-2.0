"""Naija Shopping Engine - price tracking and deal detection across Nigerian marketplaces.

Uses BrowserSession for stateful scraping with cookie persistence.
Routes through proxy manager for bot protection.
Computes steal scores based on price vs 30-day median and cross-site comparison.

Sites:
- Jumia Nigeria (jumia.com.ng) - JS-heavy, needs human pacing
- Jiji Nigeria (jiji.ng) - marketplace, variable quality
- Konga (konga.com) - easiest target, start here
- Kara Nigeria (kara.com.ng) - electronics/gadgets, Lagos-based
- SLOT Systems (slot.ng) - phones & electronics, nationwide
- Temu (temu.com) - JS-heavy, Playwright path
- AliExpress (aliexpress.com) - ships to Nigeria
- eBay (ebay.com) - price anchor for used/refurbished
- Banggood (banggood.com) - ships to Nigeria
- Amazon (amazon.com) - price anchor for comparison

FX: USD/EUR/GBP -> NGN conversion uses a live rate (cached 1h) with a
documented fallback when the network is unavailable.

Usage:
    engine = NaijaShoppingEngine(db, browser_session, proxy_manager)

    # Scan for deals
    deals = await engine.scan_category("phones", max_price=200000)

    # Cross-site price comparison
    report = await engine.compare_prices("iphone 13")

    # Track a specific product
    await engine.track_url("https://www.jumia.com.ng/...", user_id="user123")

    # Get steals (high steal score)
    steals = await engine.get_steals(threshold=80)
"""

from __future__ import annotations

import asyncio
import html
import json
import random
import re
import statistics
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..tools.browser import BrowserSession

__all__ = [
    "NaijaShoppingEngine",
    "PriceSnapshot",
    "Steal",
    "Watchlist",
    "SITES",
    "get_fx_rate",
    "extract_titles",
]

_log = get_logger(__name__)

# Site configurations
SITES = {
    "jumia": {
        "name": "Jumia Nigeria",
        "base_url": "https://www.jumia.com.ng",
        "search_path": "/catalog/?q={query}",
        "trust_score": 0.85,
        "bot_protection": "high",
        "min_delay": 3.0,
        "max_delay": 8.0,
        "currency": "NGN",
    },
    "konga": {
        "name": "Konga",
        "base_url": "https://www.konga.com",
        "search_path": "/search?search={query}",
        "trust_score": 0.80,
        "bot_protection": "low",
        "min_delay": 1.5,
        "max_delay": 3.0,
        "currency": "NGN",
    },
    "jiji": {
        "name": "Jiji Nigeria",
        "base_url": "https://jiji.ng",
        "search_path": "/search?query={query}",
        "trust_score": 0.65,
        "bot_protection": "medium",
        "min_delay": 2.0,
        "max_delay": 5.0,
        "currency": "NGN",
    },
    "kara": {
        "name": "Kara Nigeria",
        "base_url": "https://kara.com.ng",
        "search_path": "/catalogsearch/result/?q={query}",
        "trust_score": 0.78,
        "bot_protection": "medium",
        "min_delay": 2.0,
        "max_delay": 5.0,
        "currency": "NGN",
    },
    "slot": {
        "name": "SLOT Systems",
        "base_url": "https://www.slot.ng",
        "search_path": "/catalogsearch/result/?q={query}",
        "trust_score": 0.80,
        "bot_protection": "medium",
        "min_delay": 2.0,
        "max_delay": 5.0,
        "currency": "NGN",
    },
    "temu": {
        "name": "Temu",
        "base_url": "https://www.temu.com",
        "search_path": "/search_result.html?search_key={query}",
        "trust_score": 0.70,
        "bot_protection": "high",
        "min_delay": 4.0,
        "max_delay": 10.0,
        "requires_playwright": True,
        "currency": "NGN",
    },
    "aliexpress": {
        "name": "AliExpress",
        "base_url": "https://www.aliexpress.com",
        "search_path": "/w/wholesale-{query}.html",
        "trust_score": 0.75,
        "bot_protection": "medium",
        "min_delay": 2.5,
        "max_delay": 6.0,
        "currency": "USD",
    },
    "ebay": {
        "name": "eBay",
        "base_url": "https://www.ebay.com",
        "search_path": "/sch/i.html?_nkw={query}",
        "trust_score": 0.88,
        "bot_protection": "high",
        "min_delay": 3.0,
        "max_delay": 7.0,
        "currency": "USD",
    },
    "banggood": {
        "name": "Banggood",
        "base_url": "https://www.banggood.com",
        "search_path": "/search/{query}.html",
        "trust_score": 0.72,
        "bot_protection": "medium",
        "min_delay": 2.5,
        "max_delay": 6.0,
        "currency": "USD",
    },
    "amazon": {
        "name": "Amazon (price anchor)",
        "base_url": "https://www.amazon.com",
        "search_path": "/s?k={query}",
        "trust_score": 0.95,
        "bot_protection": "high",
        "min_delay": 3.0,
        "max_delay": 7.0,
        "currency": "USD",
    },
}


# ── Foreign exchange (USD/EUR/GBP -> NGN) ────────────────────────────────────
#
# Live rates are fetched from https://open.er-api.com/v6/latest/{BASE}
# (no API key required) and cached in-memory for FX_CACHE_TTL (1 hour), so a
# full multi-site scan does not hammer the endpoint.
#
# FALLBACK: when the network is unreachable or the API response changes shape,
# get_fx_rate() falls back to FALLBACK_RATES below. These are *approximate*
# hard-coded values (deliberately conservative) and must be treated as
# estimates, not market rates. Converted prices are still stored as plain NGN,
# so downstream code needs no special-casing of fallback-converted values.

FX_API_URL = "https://open.er-api.com/v6/latest/{base}"
FX_CACHE_TTL = 3600.0  # 1 hour

FALLBACK_RATES: dict[str, float] = {
    "USD": 1550.0,
    "EUR": 1680.0,
    "GBP": 1980.0,
}

_FX_CACHE: dict[str, tuple[float, float]] = {}  # "SRC:DST" -> (rate, fetched_at)


def clear_fx_cache() -> None:
    """Clear the in-memory FX cache (useful for tests and forced refreshes)."""
    _FX_CACHE.clear()


def _fetch_fx_rate(base: str, quote: str, timeout: float = 5.0) -> float:
    """Fetch a live rate from the open FX API; raises on any failure."""
    url = FX_API_URL.format(base=base)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rate = float(payload["rates"][quote])
    if rate <= 0:
        raise ValueError(f"bad rate from FX API: {rate!r}")
    return rate


def get_fx_rate(base: str = "USD", quote: str = "NGN", timeout: float = 5.0) -> float:
    """Get the base -> quote exchange rate.

    Tries the live FX API first (5s timeout), caches the result for 1 hour,
    and falls back to FALLBACK_RATES (approximate) when the network fails.

    Args:
        base: Source currency, one of USD/EUR/GBP.
        quote: Target currency (NGN for the shopping engine).
        timeout: HTTP timeout in seconds for the live fetch.

    Returns:
        The exchange rate (multiply a base-currency price by this to get quote).
    """
    base = base.upper()
    quote = quote.upper()
    if base == quote:
        return 1.0

    key = f"{base}:{quote}"
    now = time.time()
    cached = _FX_CACHE.get(key)
    if cached and (now - cached[1]) < FX_CACHE_TTL:
        return cached[0]

    try:
        rate = _fetch_fx_rate(base, quote, timeout)
        _log.debug(f"FX live rate {key} = {rate}")
    except Exception as e:
        fallback = FALLBACK_RATES.get(base)
        if fallback is None:
            raise ValueError(f"No FX rate available for {base}->{quote}") from e
        _log.warning(
            f"FX fetch failed for {base}->{quote} ({e}); "
            f"using fallback rate {fallback} (approximate, not a market rate)"
        )
        rate = fallback

    _FX_CACHE[key] = (rate, now)
    return rate


# ── Title extraction & dedupe ────────────────────────────────────────────────

_META_TITLE_RE = re.compile(
    r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_TITLE_TAG_RE = re.compile(r"<title[^>]*>\s*(.+?)\s*</title>", re.IGNORECASE | re.DOTALL)
_HEADING_RE = re.compile(r"<h[123][^>]*>(.*?)</h[123]>", re.IGNORECASE | re.DOTALL)


def _norm_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, alnum+spaces only."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _clean_title(raw: str, *, from_page_title: bool = False) -> str:
    """Strip tags/entities/noise from a raw title candidate."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if from_page_title:
        # "<title>" tags carry site suffixes: "Search results for x | Jumia Nigeria"
        text = re.split(r"\s*[|｜]\s*", text)[0].strip()
    if len(text) < 4 or len(text) > 160:
        return ""
    low = text.lower()
    if low.startswith(
        ("search results", "search for", "products for", "category:", "shop ", "home")
    ):
        return ""
    return text


def extract_titles(page_html: str, limit: int = 20) -> list[str]:
    """Best-effort product-title extraction from raw HTML.

    Tries og:title, then <title>, then h1/h2/h3 headings (product cards are
    usually headings on listing pages). Returns de-duplicated candidates.
    """
    titles: list[str] = []

    def _push(raw: str, *, page: bool = False) -> None:
        title = _clean_title(raw, from_page_title=page)
        if title and all(_norm_title(title) != _norm_title(t) for t in titles):
            titles.append(title)

    for m in _META_TITLE_RE.finditer(page_html):
        _push(m.group(1))
        if len(titles) >= limit:
            return titles

    m = _TITLE_TAG_RE.search(page_html)
    if m:
        _push(m.group(1), page=True)

    for m in _HEADING_RE.finditer(page_html):
        _push(m.group(1))
        if len(titles) >= limit:
            break

    return titles[:limit]


def _dedupe_snapshots(snapshots: list["PriceSnapshot"]) -> list["PriceSnapshot"]:
    """Drop near-identical snapshots.

    Two snapshots are duplicates when they are from the same site, their
    prices are within 1% of each other, and their normalized titles match
    (or either title is generic/empty).
    """
    unique: list[PriceSnapshot] = []
    for snap in snapshots:
        is_dup = False
        for kept in unique:
            if kept.site != snap.site:
                continue
            base = max(kept.price_ngn, snap.price_ngn)
            if base <= 0 or abs(kept.price_ngn - snap.price_ngn) / base > 0.01:
                continue
            t1, t2 = _norm_title(kept.title), _norm_title(snap.title)
            if (t1 and t2 and (t1 == t2 or t1 in t2 or t2 in t1)) or not t1 or not t2:
                is_dup = True
                break
        if not is_dup:
            unique.append(snap)
    return unique


@dataclass
class PriceSnapshot:
    """A price snapshot for a product at a point in time."""
    
    product_url: str
    site: str
    price_ngn: float
    title: str
    timestamp: float = field(default_factory=time.time)
    in_stock: bool = True
    original_price: float = 0.0  # "Was" price if shown
    discount_badge: float = 0.0  # Discount % if shown
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "product_url": self.product_url,
            "site": self.site,
            "price_ngn": self.price_ngn,
            "title": self.title,
            "timestamp": self.timestamp,
            "in_stock": self.in_stock,
        }


@dataclass
class Steal:
    """A detected steal/deal based on price analysis."""
    
    product_url: str
    site: str
    title: str
    current_price: float
    median_30d: float
    cross_site_best: float
    steal_score: float  # 0-100
    savings_vs_median: float  # % below 30-day median
    savings_vs_cross_site: float  # % below best cross-site price
    discovered_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "product_url": self.product_url,
            "site": self.site,
            "title": self.title,
            "current_price": self.current_price,
            "median_30d": self.median_30d,
            "steal_score": self.steal_score,
            "savings_vs_median": self.savings_vs_median,
        }
    
    def to_message(self, *, show_usd: bool = False,
                   fx_rate: float = 0.0) -> str:
        badge = ("🟢 GOD-TIER" if self.steal_score >= 85 else
                 "🟡 HOT" if self.steal_score >= 70 else "🟠 DECENT")
        fx = ""
        if show_usd and fx_rate:
            fx = f" (≈ ${self.current_price / fx_rate:,.2f})"
        cross = ""
        if self.cross_site_best and self.cross_site_best < self.current_price:
            cross = (f"\n🏷️ cross-site best: ₦{self.cross_site_best:,.0f} "
                     f"({self.savings_vs_cross_site:.0f}% cheaper elsewhere!)")
        elif self.cross_site_best:
            cross = "\n🏆 cheapest across all sites"
        return (
            f"🔥 **STEAL** {badge} `{self.steal_score:.0f}/100`\n\n"
            f"**{self.title[:60]}**\n"
            f"💰 ₦{self.current_price:,.0f}{fx}\n"
            f"📊 30-day median ₦{self.median_30d:,.0f} "
            f"({self.savings_vs_median:.0f}% below){cross}\n"
            f"🏪 {self.site}\n"
            f"🔗 {self.product_url}"
        )


@dataclass
class Watchlist:
    """A product watchlist entry."""
    
    watchlist_id: str
    user_id: str
    product_url: str
    site: str
    target_price: float
    title: str = ""
    created_at: float = field(default_factory=time.time)
    last_checked: float = 0.0
    last_price: float = 0.0
    is_triggered: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "watchlist_id": self.watchlist_id,
            "product_url": self.product_url,
            "site": self.site,
            "target_price": self.target_price,
            "last_price": self.last_price,
            "is_triggered": self.is_triggered,
        }


class NaijaShoppingEngine:
    """Price tracking and deal detection across Nigerian marketplaces."""
    
    def __init__(
        self,
        db: Database,
        browser: BrowserSession,
        proxy_manager: Any = None,
    ) -> None:
        self.db = db
        self.browser = browser
        self.proxy_manager = proxy_manager
        self._ensure_schema()
        _log.info("Naija Shopping Engine initialized")
    
    def _ensure_schema(self) -> None:
        """Create shopping engine tables."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    product_url TEXT NOT NULL,
                    site TEXT NOT NULL,
                    price_ngn REAL NOT NULL,
                    title TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    in_stock INTEGER NOT NULL DEFAULT 1,
                    original_price REAL NOT NULL DEFAULT 0,
                    discount_badge REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (product_url, timestamp)
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    watchlist_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    product_url TEXT NOT NULL,
                    site TEXT NOT NULL,
                    target_price REAL NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    last_checked REAL NOT NULL DEFAULT 0,
                    last_price REAL NOT NULL DEFAULT 0,
                    is_triggered INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS steals (
                    product_url TEXT NOT NULL,
                    site TEXT NOT NULL,
                    title TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    median_30d REAL NOT NULL,
                    cross_site_best REAL NOT NULL,
                    steal_score REAL NOT NULL,
                    savings_vs_median REAL NOT NULL,
                    savings_vs_cross_site REAL NOT NULL,
                    discovered_at REAL NOT NULL,
                    PRIMARY KEY (product_url, discovered_at)
                )
            """)
            
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_url ON price_snapshots(product_url, timestamp)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_steals_score ON steals(steal_score DESC)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_watchlists_user ON watchlists(user_id, is_triggered)")
    
    # ── Scraping ─────────────────────────────────────────────────────────────
    
    async def scan_category(
        self,
        query: str,
        *,
        sites: list[str] | None = None,
        max_price: float = float("inf"),
        max_pages: int = 3,
    ) -> list[PriceSnapshot]:
        """Scan for products across sites.
        
        Args:
            query: Search query or category
            sites: Sites to scan (default: all)
            max_price: Maximum price filter
            max_pages: Max pages per site
            
        Returns:
            List of PriceSnapshot objects
        """
        sites = sites or ["konga", "jumia", "jiji"]
        all_snapshots: list[PriceSnapshot] = []
        
        for site_id in sites:
            if site_id not in SITES:
                continue
            
            try:
                snapshots = await self._scan_site(site_id, query, max_price, max_pages)
                all_snapshots.extend(snapshots)
                
                # Store snapshots
                for snap in snapshots:
                    self._store_snapshot(snap)
                
                # Human pacing delay between sites
                await self._human_delay(site_id)
            except Exception as e:
                _log.warning(f"Failed to scan {site_id}: {e}")
        
        return _dedupe_snapshots(all_snapshots)
    
    async def compare_prices(
        self,
        query: str,
        *,
        sites: list[str] | None = None,
        max_price: float = float("inf"),
        max_pages: int = 1,
    ) -> dict[str, Any]:
        """Cross-site price comparison for a query.

        Scans each site (fail-soft: one dead site never kills the scan),
        stores + dedupes snapshots, and returns the cheapest snapshot per
        site, the overall best deal, and savings vs the median price.

        Args:
            query: Search query or category
            sites: Sites to compare (default: all)
            max_price: Maximum price filter in NGN
            max_pages: Pages per site (1 keeps comparison fast)

        Returns:
            Dict with query, per_site cheapest snapshots, best_deal,
            median_price, savings_vs_median_pct, sites_scanned/failed.
        """
        sites = sites or list(SITES.keys())
        per_site: dict[str, Optional[dict[str, Any]]] = {}
        sites_failed: list[str] = []
        all_snaps: list[PriceSnapshot] = []

        for site_id in sites:
            if site_id not in SITES:
                continue

            try:
                snaps = await self._scan_site(site_id, query, max_price, max_pages)
                snaps = _dedupe_snapshots(snaps)

                for snap in snaps:
                    self._store_snapshot(snap)

                per_site[site_id] = (
                    min(snaps, key=lambda s: s.price_ngn).to_dict() if snaps else None
                )
                all_snaps.extend(snaps)
            except Exception as e:
                sites_failed.append(site_id)
                _log.warning(f"compare_prices: {site_id} failed: {e}")

            # Human pacing delay between sites (kept for comparisons too)
            await self._human_delay(site_id)

        all_snaps = _dedupe_snapshots(all_snaps)
        prices = sorted(s.price_ngn for s in all_snaps if s.price_ngn > 0)
        best = min(all_snaps, key=lambda s: s.price_ngn) if all_snaps else None
        median_price = statistics.median(prices) if prices else 0.0
        savings_pct = 0.0
        if best and median_price > 0:
            savings_pct = max(0.0, (median_price - best.price_ngn) / median_price * 100)

        return {
            "query": query,
            "sites_scanned": [s for s in sites if s in SITES and s not in sites_failed],
            "sites_failed": sites_failed,
            "per_site": per_site,
            "best_deal": best.to_dict() if best else None,
            "best_site": best.site if best else None,
            "median_price": median_price,
            "savings_vs_median_pct": round(savings_pct, 1),
            "result_count": len(all_snaps),
        }
    
    async def _scan_site(
        self,
        site_id: str,
        query: str,
        max_price: float,
        max_pages: int,
    ) -> list[PriceSnapshot]:
        """Scan a specific site."""
        site = SITES[site_id]
        search_url = site["base_url"] + site["search_path"].format(query=query.replace(" ", "+"))
        
        snapshots = []
        
        for page in range(max_pages):
            # Navigate to search page
            await self.browser.navigate(search_url)
            
            # Human pacing
            await self._human_delay(site_id)
            
            # Parse results
            page_snapshots = self._parse_search_results(
                self.browser.current_page,
                site_id,
                max_price,
            )
            
            snapshots.extend(page_snapshots)
            
            # Try next page
            next_url = self._extract_next_page(self.browser.current_page)
            if not next_url:
                break
            search_url = next_url
        
        return snapshots
    
    def _parse_search_results(
        self,
        html: str,
        site_id: str,
        max_price: float,
    ) -> list[PriceSnapshot]:
        """Parse search results from HTML.

        Extracts prices with per-site regex patterns, converts foreign
        currencies to NGN via the FX helper, pairs prices with extracted
        product titles by position, and dedupes near-identical hits.
        """
        snapshots = []
        site_cfg = SITES[site_id]

        # Site-specific price patterns
        price_patterns = {
            "jumia": [r'data-price="(\d[\d,]*)"', r'₦\s*([\d,]+)'],
            "konga": [r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)', r'"price":\s*"?([\d,]+)"?'],
            "jiji": [r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)'],
            "kara": [r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)', r'"price":\s*"?([\d,]+)"?'],
            "slot": [r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)', r'"price":\s*"?([\d,]+)"?'],
            "temu": [r'"price":\s*(\d+)', r'₦\s*([\d,]+)'],
            "aliexpress": [r'"minPrice":\s*([\d.]+)', r'US \$\s*([\d.]+)'],
            "ebay": [r's-item__price[^>]*>\s*(?:US\s*)?\$([\d,.]+)', r'"price":"([\d.]+)"'],
            "banggood": [r'"salePrice":"([\d.]+)"', r'\$\s*([\d.]+)'],
            "amazon": [r'"price":"([\d.]+)"', r'\$([\d.]+)'],
        }

        patterns = price_patterns.get(site_id, [r'₦\s*([\d,]+)'])
        titles = extract_titles(html)

        # Collect raw prices across all patterns first, then pair with titles.
        raw_prices: list[float] = []
        for pattern in patterns:
            matches = re.findall(pattern, html)
            for price_str in matches:
                try:
                    price = float(price_str.replace(",", ""))
                except ValueError:
                    continue
                if price <= 0:
                    continue

                # Convert foreign currencies to NGN via the FX helper
                # (live rate, cached 1h, documented fallback when offline).
                currency = site_cfg.get("currency", "NGN")
                if currency != "NGN":
                    price *= get_fx_rate(currency, "NGN")

                raw_prices.append(price)

        for i, price in enumerate(raw_prices[:20]):
            if price > max_price or price < 100:  # Filter noise
                continue

            title = titles[i] if i < len(titles) else f"Product from {site_id}"

            snapshots.append(PriceSnapshot(
                product_url=site_cfg["base_url"],
                site=site_id,
                price_ngn=price,
                title=title,
            ))

        return _dedupe_snapshots(snapshots)
    
    def _extract_next_page(self, html: str) -> Optional[str]:
        """Extract next page URL from pagination."""
        # Simplified - real implementation would parse pagination
        return None
    
    async def _human_delay(self, site_id: str) -> None:
        """Apply human-like delay to avoid bot detection."""
        site = SITES[site_id]
        delay = random.uniform(site["min_delay"], site["max_delay"])
        
        # Add jitter (±20%)
        jitter = delay * random.uniform(-0.2, 0.2)
        delay += jitter
        
        await asyncio.sleep(max(0.5, delay))
    
    # ── Price Analysis ───────────────────────────────────────────────────────
    
    async def compute_steal_score(self, product_url: str) -> Optional[Steal]:
        """Compute steal score for a product based on price history.
        
        Steal score = (savings vs 30-day median) × (savings vs cross-site best) × trust
        
        Args:
            product_url: Product URL
            
        Returns:
            Steal object if score > 0, None otherwise
        """
        # Get 30-day price history
        thirty_days_ago = time.time() - (30 * 24 * 3600)
        
        snapshots = self.db.query("""
            SELECT * FROM price_snapshots
            WHERE product_url = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (product_url, thirty_days_ago))
        
        if not snapshots:
            return None
        
        # Current price (most recent)
        current = snapshots[-1]
        current_price = current["price_ngn"]
        
        # 30-day median
        prices = [s["price_ngn"] for s in snapshots if s["price_ngn"] > 0]
        if not prices:
            return None
        
        prices.sort()
        median_30d = prices[len(prices) // 2]
        
        # Cross-site best (lowest price for same product across all sites)
        cross_site = self.db.query("""
            SELECT MIN(price_ngn) as best FROM price_snapshots
            WHERE title = ? AND timestamp >= ?
        """, (current["title"], thirty_days_ago))
        
        cross_site_best = cross_site[0]["best"] if cross_site else current_price
        
        # Compute savings
        savings_vs_median = ((median_30d - current_price) / median_30d) * 100 if median_30d > 0 else 0
        savings_vs_cross_site = ((cross_site_best - current_price) / cross_site_best) * 100 if cross_site_best > 0 else 0
        
        # Steal score (0-100)
        # Weighted: 50% median savings, 30% cross-site savings, 20% trust
        trust = SITES.get(current["site"], {}).get("trust_score", 0.5)
        steal_score = (
            (savings_vs_median * 0.5) +
            (savings_vs_cross_site * 0.3) +
            (trust * 20)
        )
        
        if steal_score <= 0:
            return None
        
        steal = Steal(
            product_url=product_url,
            site=current["site"],
            title=current["title"],
            current_price=current_price,
            median_30d=median_30d,
            cross_site_best=cross_site_best,
            steal_score=min(100, steal_score),
            savings_vs_median=savings_vs_median,
            savings_vs_cross_site=savings_vs_cross_site,
        )
        
        # Store if score is high enough
        if steal_score >= 50:
            self._store_steal(steal)
        
        return steal
    
    async def get_steals(self, *, threshold: float = 70, limit: int = 20) -> list[Steal]:
        """Get recent steals above threshold."""
        rows = self.db.query("""
            SELECT * FROM steals
            WHERE steal_score >= ? AND discovered_at >= ?
            ORDER BY steal_score DESC
            LIMIT ?
        """, (threshold, time.time() - (7 * 24 * 3600), limit))

        return [
            Steal(
                product_url=r["product_url"],
                site=r["site"],
                title=r["title"],
                current_price=r["current_price"],
                median_30d=r["median_30d"],
                cross_site_best=r["cross_site_best"],
                steal_score=r["steal_score"],
                savings_vs_median=r["savings_vs_median"],
                savings_vs_cross_site=r["savings_vs_cross_site"],
                discovered_at=r["discovered_at"],
            )
            for r in rows
        ]

    async def format_steal_digest(self, *, threshold: float = 70,
                                  limit: int = 10,
                                  show_usd: bool = False) -> str:
        """God-tier steals briefing: ranked cards with cross-site flags."""
        steals = await self.get_steals(threshold=threshold, limit=limit)
        if not steals:
            return ("💸 **steals**\n_nothing above the bar right now — "
                    "I'll keep watching._")
        fx = get_fx_rate("USD", "NGN") if show_usd else 0.0
        lines = [f"💸 **steals** ({len(steals)} @ ≥{threshold:.0f})"]
        for i, s in enumerate(steals, 1):
            lines.append(f"\n**{i}.** " + s.to_message(
                show_usd=show_usd, fx_rate=fx).replace("\n", "\n    "))
        return "\n".join(lines)

    def match_cross_site(self, snapshots: list["PriceSnapshot"]
                         ) -> list[dict[str, Any]]:
        """Group snapshots that are the SAME product on different sites.

        Reuses the deal-hunter's word-order-invariant title matching.
        """
        from .naija_deals import NaijaDealHunter, Deal
        deals = [Deal(deal_id=s.product_url, title=s.title, vendor=s.site,
                      current_price=s.price, original_price=s.price,
                      discount_percent=0.0, url=s.product_url,
                      deal_score=0.0)
                 for s in snapshots]
        groups = NaijaDealHunter.match_products(deals)
        return [{"title": g["title"],
                 "sites": [(d.vendor, d.current_price) for d in g["deals"]],
                 "best_site": g["best"].vendor if g["best"] else "",
                 "best_price": g["best"].current_price if g["best"] else 0}
                for g in groups]
    
    # ── Watchlists ───────────────────────────────────────────────────────────
    
    async def track_url(
        self,
        product_url: str,
        user_id: str,
        *,
        target_price: float = 0,
    ) -> Watchlist:
        """Add a product to watchlist."""
        watchlist_id = new_id("watch")
        site = self._detect_site_from_url(product_url)
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO watchlists
                (watchlist_id, user_id, product_url, site, target_price, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (watchlist_id, user_id, product_url, site, target_price, time.time()))
        
        return Watchlist(
            watchlist_id=watchlist_id,
            user_id=user_id,
            product_url=product_url,
            site=site,
            target_price=target_price,
        )
    
    async def check_watchlists(self) -> list[Watchlist]:
        """Check all active watchlists for price drops."""
        watchlists = self.db.query("""
            SELECT * FROM watchlists WHERE is_triggered = 0
        """)
        
        triggered = []
        
        for row in watchlists:
            try:
                # Get latest price
                latest = self.db.query_one("""
                    SELECT price_ngn FROM price_snapshots
                    WHERE product_url = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (row["product_url"],))
                
                if not latest:
                    continue
                
                current_price = latest["price_ngn"]
                
                # Update last checked
                with self.db.transaction():
                    self.db.execute("""
                        UPDATE watchlists SET last_checked = ?, last_price = ?
                        WHERE watchlist_id = ?
                    """, (time.time(), current_price, row["watchlist_id"]))
                
                # Check if target reached
                if row["target_price"] > 0 and current_price <= row["target_price"]:
                    with self.db.transaction():
                        self.db.execute("""
                            UPDATE watchlists SET is_triggered = 1 WHERE watchlist_id = ?
                        """, (row["watchlist_id"]))
                    
                    watchlist = Watchlist(
                        watchlist_id=row["watchlist_id"],
                        user_id=row["user_id"],
                        product_url=row["product_url"],
                        site=row["site"],
                        target_price=row["target_price"],
                        last_price=current_price,
                        is_triggered=True,
                    )
                    triggered.append(watchlist)
            except Exception as e:
                _log.warning(f"Watchlist check failed: {e}")
        
        return triggered
    
    # ── Helpers ──────────────────────────────────────────────────────────────
    
    def _store_snapshot(self, snapshot: PriceSnapshot) -> None:
        """Store a price snapshot."""
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO price_snapshots
                (product_url, site, price_ngn, title, timestamp, in_stock, original_price, discount_badge)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.product_url, snapshot.site, snapshot.price_ngn,
                snapshot.title, snapshot.timestamp, int(snapshot.in_stock),
                snapshot.original_price, snapshot.discount_badge,
            ))
    
    def _store_steal(self, steal: Steal) -> None:
        """Store a detected steal."""
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO steals
                (product_url, site, title, current_price, median_30d, cross_site_best,
                 steal_score, savings_vs_median, savings_vs_cross_site, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                steal.product_url, steal.site, steal.title, steal.current_price,
                steal.median_30d, steal.cross_site_best, steal.steal_score,
                steal.savings_vs_median, steal.savings_vs_cross_site, steal.discovered_at,
            ))
    
    def _detect_site_from_url(self, url: str) -> str:
        """Detect site from product URL."""
        url_lower = url.lower()
        for site_id, site in SITES.items():
            if site["base_url"].replace("https://", "") in url_lower:
                return site_id
        return "unknown"
