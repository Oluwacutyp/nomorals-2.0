"""Generalized price tracker - one engine, many watchlists.

Tracks prices on ANYTHING with a URL: flights, GPUs, cars, sneakers, domains,
crypto, stocks, concert tickets, etc.

The Naija Shopping Engine is a specialization of this for Nigerian marketplaces.
This is the general-purpose version.

Usage:
    tracker = PriceTracker(db, browser)
    
    # Track a flight
    await tracker.watch(
        url="https://www.google.com/travel/flights/...",
        user_id="user123",
        target_price=500,  # USD
        category="flights",
    )
    
    # Track a GPU
    await tracker.watch(
        url="https://www.newegg.com/rtx-4090/...",
        user_id="user123",
        target_price=1500,
        category="gpu",
    )
    
    # Check all watchlists
    drops = await tracker.check_all()
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database
from ..tools.browser import BrowserSession

__all__ = ["PriceTracker", "WatchItem", "PriceDrop"]

_log = get_logger(__name__)

# Category-specific price extraction patterns
PRICE_PATTERNS = {
    "default": [
        r'\$\s*([\d,]+\.?\d*)',
        r'USD\s*([\d,]+\.?\d*)',
        r'"price":\s*"?([\d.]+)"?',
        r'"amount":\s*"?([\d.]+)"?',
        r'€\s*([\d,]+\.?\d*)',
        r'£\s*([\d,]+\.?\d*)',
        r'₦\s*([\d,]+\.?\d*)',
    ],
    "flights": [
        r'\$\s*([\d,]+)',
        r'"price":\s*(\d+)',
        r'"totalPrice":\s*"?([\d.]+)"?',
    ],
    "crypto": [
        r'\$\s*([\d,]+\.?\d*)',
        r'"price_usd":\s*"?([\d.]+)"?',
        r'"current_price":\s*([\d.]+)',
    ],
    "gpu": [
        r'\$\s*([\d,]+\.?\d*)',
        r'"price":\s*"?([\d.]+)"?',
    ],
}

# Human pacing per domain
DOMAIN_DELAYS = {
    "google.com": (2.0, 5.0),
    "newegg.com": (2.0, 4.0),
    "amazon.com": (3.0, 7.0),
    "bestbuy.com": (2.0, 5.0),
    "jumia.com.ng": (3.0, 8.0),
    "konga.com": (1.5, 3.0),
    "temu.com": (4.0, 10.0),
    "aliexpress.com": (2.5, 6.0),
    "coindesk.com": (1.0, 3.0),
    "coinbase.com": (2.0, 4.0),
}


@dataclass
class WatchItem:
    """An item being price-tracked."""
    
    watch_id: str
    user_id: str
    url: str
    category: str
    target_price: float
    currency: str = "USD"
    title: str = ""
    current_price: float = 0.0
    lowest_seen: float = float("inf")
    highest_seen: float = 0.0
    check_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_checked: float = 0.0
    is_triggered: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "watch_id": self.watch_id,
            "url": self.url,
            "category": self.category,
            "target_price": self.target_price,
            "current_price": self.current_price,
            "lowest_seen": self.lowest_seen,
            "title": self.title,
            "is_triggered": self.is_triggered,
        }


@dataclass
class PriceDrop:
    """A detected price drop."""
    
    watch_id: str
    url: str
    title: str
    category: str
    old_price: float
    new_price: float
    drop_percent: float
    target_price: float
    detected_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "watch_id": self.watch_id,
            "url": self.url,
            "title": self.title,
            "category": self.category,
            "old_price": self.old_price,
            "new_price": self.new_price,
            "drop_percent": self.drop_percent,
        }
    
    def to_message(self) -> str:
        emoji = {
            "flights": "✈️", "gpu": "🖥️", "crypto": "🪙",
            "cars": "🚗", "sneakers": "👟", "stocks": "📈",
        }.get(self.category, "💰")
        
        return (
            f"{emoji} **Price Drop Alert!**\n\n"
            f"**{self.title[:60]}**\n"
            f"💰 Was: ${self.old_price:,.2f} → Now: ${self.new_price:,.2f}\n"
            f"📉 Drop: {self.drop_percent:.1f}%\n"
            f"🎯 Target: ${self.target_price:,.2f}\n"
            f"🔗 {self.url}"
        )


class PriceTracker:
    """General-purpose price tracker for any URL."""
    
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
        _log.info("Price Tracker initialized")
    
    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS watch_items (
                    watch_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    target_price REAL NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'USD',
                    title TEXT NOT NULL DEFAULT '',
                    current_price REAL NOT NULL DEFAULT 0,
                    lowest_seen REAL NOT NULL DEFAULT 999999999,
                    highest_seen REAL NOT NULL DEFAULT 0,
                    check_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_checked REAL NOT NULL DEFAULT 0,
                    is_triggered INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS price_drops (
                    watch_id TEXT NOT NULL,
                    old_price REAL NOT NULL,
                    new_price REAL NOT NULL,
                    drop_percent REAL NOT NULL,
                    detected_at REAL NOT NULL,
                    FOREIGN KEY (watch_id) REFERENCES watch_items(watch_id)
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    watch_id TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (watch_id, timestamp)
                )
            """)
            
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_watch_user ON watch_items(user_id, is_triggered)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_watch_url ON watch_items(url)")
    
    async def watch(
        self,
        url: str,
        user_id: str,
        *,
        target_price: float,
        category: str = "general",
        currency: str = "USD",
        title: str = "",
    ) -> WatchItem:
        """Start watching a URL for price drops.
        
        Args:
            url: URL to watch
            user_id: User requesting the watch
            target_price: Alert when price drops below this
            category: Category (flights, gpu, crypto, etc.)
            currency: Price currency
            title: Optional title override
            
        Returns:
            WatchItem object
        """
        watch_id = new_id("watch")
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO watch_items
                (watch_id, user_id, url, category, target_price, currency, title, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (watch_id, user_id, url, category, target_price, currency, title, time.time()))
        
        item = WatchItem(
            watch_id=watch_id,
            user_id=user_id,
            url=url,
            category=category,
            target_price=target_price,
            currency=currency,
            title=title,
        )
        
        # Do initial price check
        await self._check_single(item)
        
        _log.info(f"Watching: {url} (target: {currency} {target_price})")
        return item
    
    async def check_all(self) -> list[PriceDrop]:
        """Check all active watchlists for price drops.
        
        Returns:
            List of newly triggered price drops
        """
        items = self.db.query("""
            SELECT * FROM watch_items WHERE is_triggered = 0
        """)
        
        drops = []
        
        for row in items:
            item = WatchItem(
                watch_id=row["watch_id"],
                user_id=row["user_id"],
                url=row["url"],
                category=row["category"],
                target_price=row["target_price"],
                currency=row["currency"],
                title=row["title"],
                current_price=row["current_price"],
                lowest_seen=row["lowest_seen"],
                highest_seen=row["highest_seen"],
                check_count=row["check_count"],
                created_at=row["created_at"],
                last_checked=row["last_checked"],
            )
            
            try:
                drop = await self._check_single(item)
                if drop:
                    drops.append(drop)
                
                # Human pacing delay between checks
                await self._domain_delay(item.url)
            except Exception as e:
                _log.warning(f"Failed to check {item.url}: {e}")
        
        return drops
    
    async def _check_single(self, item: WatchItem) -> Optional[PriceDrop]:
        """Check a single item for price changes."""
        old_price = item.current_price
        
        # Fetch current price
        new_price = await self._fetch_price(item.url, item.category)
        
        if new_price <= 0:
            return None
        
        # Update item
        item.current_price = new_price
        item.check_count += 1
        item.last_checked = time.time()
        
        if new_price < item.lowest_seen:
            item.lowest_seen = new_price
        if new_price > item.highest_seen:
            item.highest_seen = new_price
        
        # Store price history
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO price_history (watch_id, price, timestamp)
                VALUES (?, ?, ?)
            """, (item.watch_id, new_price, item.last_checked))
            
            self.db.execute("""
                UPDATE watch_items SET
                    current_price = ?, check_count = ?, last_checked = ?,
                    lowest_seen = ?, highest_seen = ?
                WHERE watch_id = ?
            """, (new_price, item.check_count, item.last_checked,
                  item.lowest_seen, item.highest_seen, item.watch_id))
        
        # Check for drop
        drop = None
        if old_price > 0 and new_price < old_price:
            drop_percent = ((old_price - new_price) / old_price) * 100
            
            drop = PriceDrop(
                watch_id=item.watch_id,
                url=item.url,
                title=item.title or item.url,
                category=item.category,
                old_price=old_price,
                new_price=new_price,
                drop_percent=drop_percent,
                target_price=item.target_price,
            )
            
            # Store drop
            with self.db.transaction():
                self.db.execute("""
                    INSERT INTO price_drops (watch_id, old_price, new_price, drop_percent, detected_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (item.watch_id, old_price, new_price, drop_percent, time.time()))
        
        # Check if target reached
        if new_price <= item.target_price and not item.is_triggered:
            with self.db.transaction():
                self.db.execute("""
                    UPDATE watch_items SET is_triggered = 1 WHERE watch_id = ?
                """, (item.watch_id,))
            item.is_triggered = True
            _log.info(f"🎯 Target reached: {item.url} at {item.currency} {new_price}")
        
        return drop
    
    async def _fetch_price(self, url: str, category: str) -> float:
        """Fetch current price from URL."""
        try:
            await self.browser.navigate(url)
            await self._domain_delay(url)
            
            html = self.browser.current_page or ""
            
            # Extract price using category-specific patterns
            patterns = PRICE_PATTERNS.get(category, PRICE_PATTERNS["default"])
            
            prices = []
            for pattern in patterns:
                matches = re.findall(pattern, html)
                for match in matches:
                    try:
                        price = float(match.replace(",", ""))
                        if 0.01 < price < 10_000_000:  # Sanity check
                            prices.append(price)
                    except (ValueError, TypeError):
                        continue
            
            if not prices:
                return 0.0
            
            # Return the most likely price (median of found prices)
            prices.sort()
            return prices[len(prices) // 2]
            
        except Exception as e:
            _log.warning(f"Failed to fetch price from {url}: {e}")
            return 0.0
    
    async def _domain_delay(self, url: str) -> None:
        """Apply human-like delay based on domain."""
        domain = urlparse(url).netloc.lower()
        
        # Find matching domain delay
        min_delay, max_delay = 2.0, 5.0
        for d, delays in DOMAIN_DELAYS.items():
            if d in domain:
                min_delay, max_delay = delays
                break
        
        delay = random.uniform(min_delay, max_delay)
        await asyncio.sleep(delay)
    
    async def get_user_watchlist(self, user_id: str) -> list[WatchItem]:
        """Get all watch items for a user."""
        rows = self.db.query(
            "SELECT * FROM watch_items WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        
        return [
            WatchItem(
                watch_id=r["watch_id"],
                user_id=r["user_id"],
                url=r["url"],
                category=r["category"],
                target_price=r["target_price"],
                currency=r["currency"],
                title=r["title"],
                current_price=r["current_price"],
                lowest_seen=r["lowest_seen"],
                highest_seen=r["highest_seen"],
                check_count=r["check_count"],
                created_at=r["created_at"],
                last_checked=r["last_checked"],
                is_triggered=bool(r["is_triggered"]),
            )
            for r in rows
        ]
    
    async def get_price_history(self, watch_id: str) -> list[dict[str, Any]]:
        """Get price history for a watch item."""
        rows = self.db.query(
            "SELECT price, timestamp FROM price_history WHERE watch_id = ? ORDER BY timestamp ASC",
            (watch_id,)
        )
        return [{"price": r["price"], "timestamp": r["timestamp"]} for r in rows]
    
    async def remove_watch(self, watch_id: str) -> bool:
        """Remove a watch item."""
        with self.db.transaction():
            self.db.execute("DELETE FROM watch_items WHERE watch_id = ?", (watch_id,))
            self.db.execute("DELETE FROM price_history WHERE watch_id = ?", (watch_id,))
            self.db.execute("DELETE FROM price_drops WHERE watch_id = ?", (watch_id,))
        return True
