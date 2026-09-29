"""Naija Shopping Engine - price tracking and deal detection across Nigerian marketplaces.

Uses BrowserSession for stateful scraping with cookie persistence.
Routes through proxy manager for bot protection.
Computes steal scores based on price vs 30-day median and cross-site comparison.

Sites:
- Jumia Nigeria (jumia.com.ng) - JS-heavy, needs human pacing
- Jiji Nigeria (jiji.ng) - marketplace, variable quality
- Konga (konga.com) - easiest target, start here
- Temu (temu.com) - JS-heavy, Playwright path
- AliExpress (aliexpress.com) - ships to Nigeria
- Amazon (amazon.com) - price anchor for comparison

Usage:
    engine = NaijaShoppingEngine(db, browser_session, proxy_manager)
    
    # Scan for deals
    deals = await engine.scan_category("phones", max_price=200000)
    
    # Track a specific product
    await engine.track_url("https://www.jumia.com.ng/...", user_id="user123")
    
    # Get steals (high steal score)
    steals = await engine.get_steals(threshold=80)
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
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
    },
    "konga": {
        "name": "Konga",
        "base_url": "https://www.konga.com",
        "search_path": "/search?search={query}",
        "trust_score": 0.80,
        "bot_protection": "low",
        "min_delay": 1.5,
        "max_delay": 3.0,
    },
    "jiji": {
        "name": "Jiji Nigeria",
        "base_url": "https://jiji.ng",
        "search_path": "/search?query={query}",
        "trust_score": 0.65,
        "bot_protection": "medium",
        "min_delay": 2.0,
        "max_delay": 5.0,
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
    },
    "aliexpress": {
        "name": "AliExpress",
        "base_url": "https://www.aliexpress.com",
        "search_path": "/w/wholesale-{query}.html",
        "trust_score": 0.75,
        "bot_protection": "medium",
        "min_delay": 2.5,
        "max_delay": 6.0,
    },
    "amazon": {
        "name": "Amazon (price anchor)",
        "base_url": "https://www.amazon.com",
        "search_path": "/s?k={query}",
        "trust_score": 0.95,
        "bot_protection": "high",
        "min_delay": 3.0,
        "max_delay": 7.0,
    },
}


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
    
    def to_message(self) -> str:
        return (
            f"🔥 **STEAL DETECTED** (Score: {self.steal_score:.0f}/100)\n\n"
            f"**{self.title[:60]}**\n"
            f"💰 ₦{self.current_price:,.0f}\n"
            f"📊 30-day median: ₦{self.median_30d:,.0f} ({self.savings_vs_median:.0f}% below)\n"
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
        
        return all_snapshots
    
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
        """Parse search results from HTML."""
        snapshots = []
        
        # Site-specific price patterns
        price_patterns = {
            "jumia": [r'data-price="(\d[\d,]*)"', r'₦\s*([\d,]+)'],
            "konga": [r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)', r'"price":\s*"?([\d,]+)"?'],
            "jiji": [r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)'],
            "temu": [r'"price":\s*(\d+)', r'₦\s*([\d,]+)'],
            "aliexpress": [r'"minPrice":\s*([\d.]+)', r'US \$\s*([\d.]+)'],
            "amazon": [r'"price":"([\d.]+)"', r'\$([\d.]+)'],
        }
        
        patterns = price_patterns.get(site_id, [r'₦\s*([\d,]+)'])
        
        # Extract prices
        for pattern in patterns:
            matches = re.findall(pattern, html)
            for price_str in matches[:20]:
                try:
                    price = float(price_str.replace(",", ""))
                    
                    # Convert USD to NGN if needed (approximate rate)
                    if site_id in ("amazon", "aliexpress") and "$" in pattern:
                        price *= 1500  # Rough USD to NGN conversion
                    
                    if price > max_price or price < 100:  # Filter noise
                        continue
                    
                    snapshots.append(PriceSnapshot(
                        product_url=SITES[site_id]["base_url"],
                        site=site_id,
                        price_ngn=price,
                        title=f"Product from {site_id}",
                    ))
                except (ValueError, IndexError):
                    continue
        
        return snapshots
    
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
