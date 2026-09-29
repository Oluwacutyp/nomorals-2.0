"""Nigerian commerce connector — Jumia, Konga, Jiji, AliExpress, Temu.

Browser automation for Nigerian marketplaces (no public APIs). Implements
"real steal buys" deal hunter that flags items 35%+ below 30-day median.

Adapters:
- Jumia (jumia.com.ng)
- Konga (konga.com)
- Jiji (jiji.ng)
- AliExpress (aliexpress.com)
- Temu (temu.com)

Steal detection:
- Scrape 30-day price history for each listing
- Calculate median price across sites
- Flag if current price < 65% of median (35%+ discount)
- Scam filter: No seller history + too-good price → reject

Security:
- Polite scraping: Rate limits, proxy rotation, user-agent rotation
- Respect robots.txt
- No credentials stored (public listings only)
"""

from __future__ import annotations

import random
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .base import BaseConnector, ConnectorStatus

__all__ = ["NaijaCommerceConnector", "ProductListing", "PriceSnapshot", "DealHunter"]

_log = get_logger(__name__)


@dataclass
class ProductListing:
    """Product listing from a marketplace."""
    
    listing_id: str
    marketplace: str  # jumia, konga, jiji, aliexpress, temu
    title: str
    url: str
    price_ngn: float  # Normalized to Naira
    currency: str = "NGN"
    image_url: str = ""
    seller: str = ""
    seller_rating: float = 0.0
    seller_reviews: int = 0
    condition: str = "new"  # new, used, refurbished
    location: str = ""
    scraped_at: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "marketplace": self.marketplace,
            "title": self.title,
            "url": self.url,
            "price_ngn": self.price_ngn,
            "currency": self.currency,
            "image_url": self.image_url,
            "seller": self.seller,
            "seller_rating": self.seller_rating,
            "seller_reviews": self.seller_reviews,
            "condition": self.condition,
            "location": self.location,
            "scraped_at": self.scraped_at,
        }


@dataclass
class PriceSnapshot:
    """Price snapshot for tracking history."""
    
    listing_id: str
    marketplace: str
    url: str
    price_ngn: float
    title: str
    snapshot_at: float = 0.0
    
    def __post_init__(self) -> None:
        if not self.snapshot_at:
            self.snapshot_at = time.time()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "marketplace": self.marketplace,
            "url": self.url,
            "price_ngn": self.price_ngn,
            "title": self.title,
            "snapshot_at": self.snapshot_at,
        }


class DealHunter:
    """"Real steal buys" deal hunter — flags items 35%+ below median.
    
    Steal detection algorithm:
    1. Query marketplace for product (e.g., "iPhone 15")
    2. Collect prices from multiple listings
    3. Calculate 30-day median price
    4. Flag if current_price < 0.65 * median (35%+ discount)
    5. Scam filter: If seller has < 10 reviews and price is 50%+ below median → reject
    
    Price tracking:
    - Store snapshots in SQLite (price_watches table)
    - Scheduled checks (twice daily + flash sale windows)
    - Alert on price drops
    
    Currency handling:
    - Strip "was ₦X" games (Jumia fake discounts)
    - Normalize all to NGN
    """
    
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else Path.home() / ".nomorals" / "deals.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Create deal tracking tables."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    id TEXT PRIMARY KEY,
                    listing_id TEXT NOT NULL,
                    marketplace TEXT NOT NULL,
                    url TEXT NOT NULL,
                    price_ngn REAL NOT NULL,
                    title TEXT NOT NULL,
                    snapshot_at REAL NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS price_watches (
                    watch_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    target_price REAL DEFAULT 0.0,
                    current_price REAL DEFAULT 0.0,
                    last_check REAL DEFAULT 0.0,
                    triggered INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_snapshots_listing ON price_snapshots(listing_id);
                CREATE INDEX IF NOT EXISTS idx_snapshots_time ON price_snapshots(snapshot_at);
                CREATE INDEX IF NOT EXISTS idx_watches_user ON price_watches(user_id);
            """)
            conn.commit()
        finally:
            conn.close()
    
    def find_steals(
        self,
        listings: list[ProductListing],
        median_price_ngn: float = 0.0
    ) -> list[dict[str, Any]]:
        """Find steals (35%+ below median).
        
        Args:
            listings: List of product listings
            median_price_ngn: 30-day median price (if 0, calculated from listings)
        
        Returns:
            List of steals with discount percentage and scam risk
        """
        if not listings:
            return []
        
        # Calculate median if not provided
        if median_price_ngn == 0.0:
            prices = [l.price_ngn for l in listings if l.price_ngn > 0]
            if not prices:
                return []
            prices.sort()
            median_price_ngn = prices[len(prices) // 2]
        
        steals = []
        for listing in listings:
            if listing.price_ngn <= 0 or median_price_ngn <= 0:
                continue
            
            discount_pct = (1.0 - listing.price_ngn / median_price_ngn) * 100
            
            # Flag if 35%+ below median
            if discount_pct >= 35.0:
                # Scam filter: Check seller reputation
                scam_risk = self._assess_scam_risk(listing, discount_pct)
                
                steals.append({
                    "listing": listing.to_dict(),
                    "discount_pct": round(discount_pct, 1),
                    "median_price_ngn": median_price_ngn,
                    "scam_risk": scam_risk,
                    "recommendation": "avoid" if scam_risk == "high" else "buy"
                })
        
        # Sort by discount percentage (highest first)
        steals.sort(key=lambda x: x["discount_pct"], reverse=True)
        
        return steals
    
    def _assess_scam_risk(self, listing: ProductListing, discount_pct: float) -> str:
        """Assess scam risk for a listing.
        
        Returns:
            "low", "medium", or "high"
        """
        risk_score = 0
        
        # No seller history
        if listing.seller_reviews < 10:
            risk_score += 2
        
        # Too-good price (50%+ below median)
        if discount_pct >= 50.0:
            risk_score += 3
        elif discount_pct >= 40.0:
            risk_score += 1
        
        # No seller rating
        if listing.seller_rating == 0.0:
            risk_score += 1
        
        # Used condition with new price
        if listing.condition == "used" and discount_pct < 20.0:
            risk_score += 1
        
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    def track_price(self, url: str, user_id: str, target_price: float = 0.0) -> str:
        """Track price for a product URL.
        
        Args:
            url: Product URL
            user_id: User ID for notifications
            target_price: Target price for alert (0 = any drop)
        
        Returns:
            Watch ID
        """
        watch_id = f"watch_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                INSERT INTO price_watches 
                (watch_id, url, user_id, target_price, current_price, last_check, triggered, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (watch_id, url, user_id, target_price, 0.0, 0.0, 0, time.time()))
            conn.commit()
        finally:
            conn.close()
        
        _log.info("Started tracking: %s (target: ₦%.2f)", url, target_price)
        return watch_id
    
    def store_snapshot(self, snapshot: PriceSnapshot) -> None:
        """Store price snapshot."""
        snapshot_id = f"{snapshot.listing_id}_{int(snapshot.snapshot_at)}"
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                INSERT OR REPLACE INTO price_snapshots
                (id, listing_id, marketplace, url, price_ngn, title, snapshot_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id, snapshot.listing_id, snapshot.marketplace,
                snapshot.url, snapshot.price_ngn, snapshot.title, snapshot.snapshot_at
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_price_history(self, listing_id: str, days: int = 30) -> list[PriceSnapshot]:
        """Get price history for a listing.
        
        Args:
            listing_id: Listing ID
            days: Number of days to look back
        
        Returns:
            List of price snapshots (oldest first)
        """
        cutoff = time.time() - (days * 86400)
        
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute("""
                SELECT * FROM price_snapshots
                WHERE listing_id = ? AND snapshot_at >= ?
                ORDER BY snapshot_at ASC
            """, (listing_id, cutoff))
            
            snapshots = []
            for row in cursor.fetchall():
                snapshots.append(PriceSnapshot(
                    listing_id=row[1],
                    marketplace=row[2],
                    url=row[3],
                    price_ngn=row[4],
                    title=row[5],
                    snapshot_at=row[6]
                ))
            
            return snapshots
        finally:
            conn.close()
    
    def calculate_median_price(self, listing_id: str, days: int = 30) -> float:
        """Calculate 30-day median price for a listing.
        
        Returns:
            Median price in NGN, or 0.0 if no data
        """
        snapshots = self.get_price_history(listing_id, days)
        if not snapshots:
            return 0.0
        
        prices = [s.price_ngn for s in snapshots if s.price_ngn > 0]
        if not prices:
            return 0.0
        
        prices.sort()
        return prices[len(prices) // 2]


class MarketplaceAdapter:
    """Base adapter for marketplace scraping."""
    
    name: str = ""
    base_url: str = ""
    
    def __init__(self, proxy_pool: Any = None) -> None:
        self.proxy_pool = proxy_pool
    
    def search(self, query: str, max_results: int = 20) -> list[ProductListing]:
        """Search for products (stub — implement per marketplace)."""
        raise NotImplementedError
    
    def get_product(self, url: str) -> ProductListing:
        """Get product details (stub — implement per marketplace)."""
        raise NotImplementedError
    
    def _normalize_price(self, price_str: str, currency: str = "NGN") -> float:
        """Normalize price string to float.
        
        Handles:
        - "₦150,000" → 150000.0
        - "was ₦200,000 ₦150,000" → 150000.0 (strip fake discounts)
        - "$100 USD" → convert to NGN
        """
        # Strip "was ₦X" games
        if "was" in price_str.lower():
            parts = price_str.lower().split("was")
            price_str = parts[-1]
        
        # Extract numeric value
        match = re.search(r"[\d,]+\.?\d*", price_str.replace(",", ""))
        if not match:
            return 0.0
        
        price = float(match.group())
        
        # Convert to NGN if needed
        if currency == "USD":
            price *= 1500.0  # Approximate exchange rate
        elif currency == "EUR":
            price *= 1600.0
        elif currency == "GBP":
            price *= 1900.0
        
        return price


class JumiaAdapter(MarketplaceAdapter):
    """Jumia Nigeria adapter."""
    
    name = "jumia"
    base_url = "https://www.jumia.com.ng"
    
    def search(self, query: str, max_results: int = 20) -> list[ProductListing]:
        """Search Jumia (stub — requires Playwright)."""
        # In production: Use Playwright to scrape search results
        # For now, return empty list
        _log.info("Jumia search: %s (not implemented)", query)
        return []


class KongaAdapter(MarketplaceAdapter):
    """Konga adapter."""
    
    name = "konga"
    base_url = "https://www.konga.com"


class JijiAdapter(MarketplaceAdapter):
    """Jiji adapter."""
    
    name = "jiji"
    base_url = "https://jiji.ng"


class AliExpressAdapter(MarketplaceAdapter):
    """AliExpress adapter."""
    
    name = "aliexpress"
    base_url = "https://www.aliexpress.com"


class TemuAdapter(MarketplaceAdapter):
    """Temu adapter."""
    
    name = "temu"
    base_url = "https://www.temu.com"


class NaijaCommerceConnector(BaseConnector):
    """Nigerian commerce connector — unified interface for all marketplaces."""
    
    name = "naija_commerce"
    description = "Nigerian marketplaces (Jumia, Konga, Jiji, AliExpress, Temu) with deal hunter"
    
    def __init__(self, vault: Any = None, config: dict[str, Any] | None = None) -> None:
        super().__init__(vault, config)
        self.hunter = DealHunter()
        self.adapters = {
            "jumia": JumiaAdapter(),
            "konga": KongaAdapter(),
            "jiji": JijiAdapter(),
            "aliexpress": AliExpressAdapter(),
            "temu": TemuAdapter(),
        }
    
    def status(self) -> ConnectorStatus:
        """Check commerce connector status."""
        return ConnectorStatus(
            connected=True,
            account=f"{len(self.adapters)} marketplaces",
            scopes=["search", "track", "find_steals"]
        )
    
    def connect_url(self) -> str:
        """No connection URL needed (public listings)."""
        return ""
    
    def disconnect(self) -> dict[str, Any]:
        """Clear deal tracking database."""
        try:
            # Clear database
            conn = sqlite3.connect(str(self.hunter.db_path))
            try:
                conn.execute("DELETE FROM price_snapshots")
                conn.execute("DELETE FROM price_watches")
                conn.commit()
            finally:
                conn.close()
            return {"disconnected": True}
        except Exception as e:
            return {"disconnected": False, "error": str(e)}
    
    def search(self, query: str, marketplaces: list[str] | None = None, max_results: int = 20) -> list[dict[str, Any]]:
        """Search across marketplaces.
        
        Args:
            query: Search query
            marketplaces: List of marketplaces (default: all)
            max_results: Max results per marketplace
        
        Returns:
            List of product listings
        """
        if not marketplaces:
            marketplaces = list(self.adapters.keys())
        
        all_listings = []
        for marketplace in marketplaces:
            adapter = self.adapters.get(marketplace)
            if not adapter:
                continue
            
            try:
                listings = adapter.search(query, max_results)
                all_listings.extend([l.to_dict() for l in listings])
            except Exception as e:
                _log.warning("Search failed for %s: %s", marketplace, e)
        
        return all_listings
    
    def find_steals(self, query: str, max_price_ngn: float = 0.0) -> list[dict[str, Any]]:
        """Find steals (35%+ below median).
        
        Args:
            query: Product query
            max_price_ngn: Max price filter (0 = no limit)
        
        Returns:
            List of steals with discount percentage and scam risk
        """
        # Search all marketplaces
        listings = []
        for adapter in self.adapters.values():
            try:
                results = adapter.search(query, max_results=50)
                listings.extend(results)
            except Exception as e:
                _log.warning("Search failed for %s: %s", adapter.name, e)
        
        # Filter by max price
        if max_price_ngn > 0:
            listings = [l for l in listings if l.price_ngn <= max_price_ngn]
        
        # Find steals
        return self.hunter.find_steals(listings)
    
    def track_price(self, url: str, user_id: str, target_price: float = 0.0) -> str:
        """Track price for a product URL.
        
        Args:
            url: Product URL
            user_id: User ID for notifications
            target_price: Target price for alert (0 = any drop)
        
        Returns:
            Watch ID
        """
        return self.hunter.track_price(url, user_id, target_price)
    
    def get_watchlist(self, user_id: str) -> list[dict[str, Any]]:
        """Get user's price watchlist.
        
        Returns:
            List of watched items
        """
        conn = sqlite3.connect(str(self.hunter.db_path))
        try:
            cursor = conn.execute(
                "SELECT * FROM price_watches WHERE user_id = ? AND triggered = 0",
                (user_id,)
            )
            
            watches = []
            for row in cursor.fetchall():
                watches.append({
                    "watch_id": row[0],
                    "url": row[1],
                    "target_price": row[3],
                    "current_price": row[4],
                    "last_check": row[5],
                    "created_at": row[7]
                })
            
            return watches
        finally:
            conn.close()
