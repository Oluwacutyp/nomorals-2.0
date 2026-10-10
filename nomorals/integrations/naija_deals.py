"""Naija Deal Hunter - finds deals, discounts, and price drops across Nigerian marketplaces.

Monitors:
- Jumia Nigeria (jumia.com.ng)
- Konga (konga.com)
- Jiji Nigeria (jiji.ng)
- Slot.ng (electronics)
- PayPorte (fashion/lifestyle)
- Instagram sellers (via hashtag monitoring)

Features:
- Price tracking across multiple vendors
- Flash sale alerts
- Price drop notifications
- Deal scoring (discount %, vendor trust, historical price)
- Category-specific hunting (electronics, fashion, food, etc.)
- Naira conversion for international deals

Usage:
    hunter = NaijaDealHunter(account_manager, shopping)
    
    # Track a product
    await hunter.track_product(
        url="https://www.jumia.com.ng/...",
        target_price=50000,  # Alert when price drops below ₦50,000
        user_id="user123",
    )
    
    # Find deals in a category
    deals = await hunter.find_deals(
        category="electronics",
        max_price=100000,
        min_discount=20,  # At least 20% off
    )
    
    # Get price history
    history = await hunter.price_history("product_id")
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..integrations.shopping_integration import Product
from ..storage.db import Database

__all__ = [
    "NaijaDealHunter",
    "Deal",
    "PriceAlert",
    "PriceHistory",
    "NaijaVendor",
]

_log = get_logger(__name__)

# Nigerian marketplace URLs
VENDORS = {
    "jumia": {
        "name": "Jumia Nigeria",
        "base_url": "https://www.jumia.com.ng",
        "search_url": "https://www.jumia.com.ng/catalog/?q={query}",
        "trust_score": 0.85,
        "currency": "NGN",
    },
    "konga": {
        "name": "Konga",
        "base_url": "https://www.konga.com",
        "search_url": "https://www.konga.com/search?search={query}",
        "trust_score": 0.80,
        "currency": "NGN",
    },
    "jiji": {
        "name": "Jiji Nigeria",
        "base_url": "https://jiji.ng",
        "search_url": "https://jiji.ng/search?query={query}",
        "trust_score": 0.65,  # Marketplace, variable quality
        "currency": "NGN",
    },
    "slot": {
        "name": "Slot.ng",
        "base_url": "https://slot.ng",
        "search_url": "https://slot.ng/catalogsearch/result/?q={query}",
        "trust_score": 0.90,
        "currency": "NGN",
    },
    "payporte": {
        "name": "PayPorte",
        "base_url": "https://www.payporte.com",
        "search_url": "https://www.payporte.com/catalogsearch/result/?q={query}",
        "trust_score": 0.75,
        "currency": "NGN",
    },
}

# Category mappings per vendor
CATEGORY_MAP = {
    "electronics": ["phones", "laptops", "tv", "gaming", "audio"],
    "fashion": ["men-fashion", "women-fashion", "shoes", "watches"],
    "home": ["home-office", "kitchen", "appliances"],
    "groceries": ["supermarket", "health-beauty"],
    "gaming": ["gaming", "video-games"],
}


@dataclass
class Deal:
    """A discovered deal."""
    
    deal_id: str
    title: str
    vendor: str
    current_price: float
    original_price: float
    discount_percent: float
    url: str
    image_url: str = ""
    category: str = ""
    deal_score: float = 0.0  # 0-100, combines discount + trust + urgency
    is_flash_sale: bool = False
    expires_at: Optional[float] = None
    in_stock: bool = True
    discovered_at: float = field(default_factory=time.time)
    # — sweep upgrades —
    is_lowest_90d: bool = False   # cheapest in 90-day history curve
    coupons: list[str] = field(default_factory=list)
    usd_price: float = 0.0        # FX secondary display

    def to_dict(self) -> dict[str, Any]:
        return {
            "deal_id": self.deal_id,
            "title": self.title,
            "vendor": self.vendor,
            "current_price": self.current_price,
            "original_price": self.original_price,
            "discount_percent": self.discount_percent,
            "url": self.url,
            "deal_score": self.deal_score,
            "is_flash_sale": self.is_flash_sale,
            "in_stock": self.in_stock,
            "is_lowest_90d": self.is_lowest_90d,
            "coupons": self.coupons,
        }

    def _score_badge(self) -> str:
        if self.deal_score >= 85:
            return "🟢 GOD-TIER"
        if self.deal_score >= 70:
            return "🟡 HOT"
        if self.deal_score >= 50:
            return "🟠 DECENT"
        return "⚪ MEH"

    def to_message(self, *, show_usd: bool = False) -> str:
        """God-tier deal card: score badge, was→now, lowest-ever flag,
        coupon line, vendor trust, stock."""
        flash = "⚡ **FLASH SALE** " if self.is_flash_sale else ""
        vendor_info = VENDORS.get(self.vendor, {})
        trust = vendor_info.get("trust_score", 0)
        trust_s = f" · trust {trust:.0%}" if trust else ""
        stock = "" if self.in_stock else "\n⚠️ _out of stock — price only_"
        lowest = "\n📉 **lowest in 90 days**" if self.is_lowest_90d else ""
        coupon = ""
        if self.coupons:
            coupon = "\n🎟️ coupons: " + ", ".join(self.coupons[:3])
        fx = ""
        if show_usd and self.usd_price:
            fx = f" (≈ ${self.usd_price:,.2f})"
        return (
            f"{flash}🔥 **{self.title}**\n"
            f"{self._score_badge()}  `{self.deal_score:.0f}/100`\n"
            f"💰 ₦{self.current_price:,.0f}{fx} "
            f"~~₦{self.original_price:,.0f}~~ "
            f"**−{self.discount_percent:.0f}%**{lowest}{coupon}\n"
            f"🏪 {self.vendor}{trust_s}{stock}\n"
            f"🔗 {self.url}"
        )


@dataclass
class PriceAlert:
    """A price tracking alert."""
    
    alert_id: str
    user_id: str
    product_url: str
    product_title: str
    vendor: str
    target_price: float
    current_price: float = 0.0
    is_triggered: bool = False
    triggered_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    last_checked: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "product_title": self.product_title,
            "vendor": self.vendor,
            "target_price": self.target_price,
            "current_price": self.current_price,
            "is_triggered": self.is_triggered,
        }


@dataclass
class PriceHistory:
    """Price history for a product."""
    
    product_url: str
    vendor: str
    prices: list[dict[str, Any]] = field(default_factory=list)  # [{timestamp, price}]
    lowest_price: float = 0.0
    highest_price: float = 0.0
    average_price: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "product_url": self.product_url,
            "vendor": self.vendor,
            "lowest_price": self.lowest_price,
            "highest_price": self.highest_price,
            "average_price": self.average_price,
            "price_points": len(self.prices),
        }


@dataclass
class NaijaVendor:
    """A Nigerian marketplace vendor."""
    
    vendor_id: str
    name: str
    base_url: str
    trust_score: float
    currency: str = "NGN"


class NaijaDealHunter:
    """Finds and tracks deals across Nigerian marketplaces."""
    
    def __init__(
        self,
        account_manager: AccountManager,
        db: Database,
    ) -> None:
        self.account_manager = account_manager
        self.db = db
        self._ensure_schema()
        _log.info("Naija Deal Hunter initialized")
    
    def _ensure_schema(self) -> None:
        """Create deal hunter tables."""
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS deals (
                    deal_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    vendor TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    original_price REAL NOT NULL,
                    discount_percent REAL NOT NULL,
                    url TEXT NOT NULL,
                    image_url TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT '',
                    deal_score REAL NOT NULL DEFAULT 0,
                    is_flash_sale INTEGER NOT NULL DEFAULT 0,
                    expires_at REAL,
                    in_stock INTEGER NOT NULL DEFAULT 1,
                    discovered_at REAL NOT NULL
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS price_alerts (
                    alert_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    product_url TEXT NOT NULL,
                    product_title TEXT NOT NULL,
                    vendor TEXT NOT NULL,
                    target_price REAL NOT NULL,
                    current_price REAL NOT NULL DEFAULT 0,
                    is_triggered INTEGER NOT NULL DEFAULT 0,
                    triggered_at REAL,
                    created_at REAL NOT NULL,
                    last_checked REAL NOT NULL DEFAULT 0
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    product_url TEXT NOT NULL,
                    vendor TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (product_url, timestamp)
                )
            """)

            # sweep: stock + coupon tracking on the history curve
            for _ddl in (
                "ALTER TABLE price_history ADD COLUMN in_stock INTEGER"
                " NOT NULL DEFAULT 1",
                "ALTER TABLE price_history ADD COLUMN coupon TEXT"
                " NOT NULL DEFAULT ''",
            ):
                try:
                    self.db.execute(_ddl)
                except Exception:  # noqa: BLE001 - column already there
                    pass
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_deals_score
                ON deals(deal_score DESC)
            """)
            
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_user
                ON price_alerts(user_id, is_triggered)
            """)
    
    # ── Deal Discovery ───────────────────────────────────────────────────────
    
    async def find_deals(
        self,
        *,
        category: str = "",
        query: str = "",
        max_price: float = float("inf"),
        min_discount: float = 10.0,
        vendors: list[str] | None = None,
        limit: int = 20,
    ) -> list[Deal]:
        """Find deals across Nigerian marketplaces.
        
        Args:
            category: Product category (electronics, fashion, etc.)
            query: Search query
            max_price: Maximum price in Naira
            min_discount: Minimum discount percentage
            vendors: Specific vendors to search (default: all)
            limit: Maximum deals to return
            
        Returns:
            List of Deal objects sorted by deal score
        """
        vendors = vendors or list(VENDORS.keys())
        all_deals: list[Deal] = []
        
        for vendor_id in vendors:
            if vendor_id not in VENDORS:
                continue
            
            try:
                deals = await self._search_vendor(
                    vendor_id,
                    query=query or category,
                    max_price=max_price,
                    min_discount=min_discount,
                )
                all_deals.extend(deals)
            except Exception as e:
                _log.warning(f"Failed to search {vendor_id}: {e}")
        
        # Sort by deal score
        all_deals.sort(key=lambda d: d.deal_score, reverse=True)
        
        # Store top deals
        for deal in all_deals[:limit]:
            self._store_deal(deal)
        
        return all_deals[:limit]
    
    async def find_flash_sales(self, *, limit: int = 10) -> list[Deal]:
        """Find current flash sales across all vendors."""
        deals = await self.find_deals(min_discount=30.0, limit=limit * 2)
        
        flash = [d for d in deals if d.is_flash_sale or d.discount_percent >= 40]
        return flash[:limit]
    
    # ── Price Tracking ───────────────────────────────────────────────────────
    
    async def track_product(
        self,
        url: str,
        target_price: float,
        user_id: str,
    ) -> PriceAlert:
        """Start tracking a product's price.
        
        Args:
            url: Product URL
            target_price: Alert when price drops below this (in Naira)
            user_id: User to notify
            
        Returns:
            PriceAlert object
        """
        alert_id = new_id("alert")
        
        # Detect vendor from URL
        vendor = self._detect_vendor_from_url(url)
        if not vendor:
            raise ValueError(f"Unsupported marketplace URL: {url}")
        
        # Get current price
        current_price = await self._get_price(url, vendor)
        
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO price_alerts
                (alert_id, user_id, product_url, product_title, vendor,
                 target_price, current_price, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, user_id, url, "", vendor, target_price, current_price, time.time()))
        
        alert = PriceAlert(
            alert_id=alert_id,
            user_id=user_id,
            product_url=url,
            product_title="",
            vendor=vendor,
            target_price=target_price,
            current_price=current_price,
        )
        
        _log.info(f"Tracking product: {url} (target: ₦{target_price:,.0f})")
        return alert
    
    async def check_alerts(self) -> list[PriceAlert]:
        """Check all active alerts for price drops.
        
        Returns:
            List of newly triggered alerts
        """
        alerts = self.db.query("""
            SELECT * FROM price_alerts WHERE is_triggered = 0
        """)
        
        triggered = []
        
        for row in alerts:
            try:
                current_price = await self._get_price(row["product_url"], row["vendor"])
                
                # Update last checked
                with self.db.transaction():
                    self.db.execute("""
                        UPDATE price_alerts SET current_price = ?, last_checked = ?
                        WHERE alert_id = ?
                    """, (current_price, time.time(), row["alert_id"]))
                
                # Check if target reached
                if current_price <= row["target_price"]:
                    now = time.time()
                    with self.db.transaction():
                        self.db.execute("""
                            UPDATE price_alerts SET is_triggered = 1, triggered_at = ?
                            WHERE alert_id = ?
                        """, (now, row["alert_id"]))
                    
                    alert = PriceAlert(
                        alert_id=row["alert_id"],
                        user_id=row["user_id"],
                        product_url=row["product_url"],
                        product_title=row["product_title"],
                        vendor=row["vendor"],
                        target_price=row["target_price"],
                        current_price=current_price,
                        is_triggered=True,
                        triggered_at=now,
                    )
                    triggered.append(alert)
                    _log.info(f"Price alert triggered: {row['product_url']} at ₦{current_price:,.0f}")
            except Exception as e:
                _log.warning(f"Failed to check alert {row['alert_id']}: {e}")
        
        return triggered
    
    async def price_history(self, product_url: str) -> PriceHistory:
        """Get price history for a product."""
        vendor = self._detect_vendor_from_url(product_url)
        
        rows = self.db.query("""
            SELECT price, timestamp FROM price_history
            WHERE product_url = ? ORDER BY timestamp ASC
        """, (product_url,))
        
        prices = [{"price": r["price"], "timestamp": r["timestamp"]} for r in rows]
        
        if not prices:
            return PriceHistory(product_url=product_url, vendor=vendor or "unknown")
        
        price_values = [p["price"] for p in prices]
        
        return PriceHistory(
            product_url=product_url,
            vendor=vendor or "unknown",
            prices=prices,
            lowest_price=min(price_values),
            highest_price=max(price_values),
            average_price=sum(price_values) / len(price_values),
        )
    
    async def get_user_alerts(self, user_id: str, *, active_only: bool = True) -> list[PriceAlert]:
        """Get all alerts for a user."""
        query = "SELECT * FROM price_alerts WHERE user_id = ?"
        if active_only:
            query += " AND is_triggered = 0"
        query += " ORDER BY created_at DESC"
        
        rows = self.db.query(query, (user_id,))
        
        return [
            PriceAlert(
                alert_id=r["alert_id"],
                user_id=r["user_id"],
                product_url=r["product_url"],
                product_title=r["product_title"],
                vendor=r["vendor"],
                target_price=r["target_price"],
                current_price=r["current_price"],
                is_triggered=bool(r["is_triggered"]),
                triggered_at=r["triggered_at"],
            )
            for r in rows
        ]
    
    # ── Vendor Search ────────────────────────────────────────────────────────
    
    async def _search_vendor(
        self,
        vendor_id: str,
        *,
        query: str,
        max_price: float,
        min_discount: float,
    ) -> list[Deal]:
        """Search a specific vendor for deals."""
        vendor = VENDORS[vendor_id]
        search_url = vendor["search_url"].format(query=urllib.parse.quote_plus(query))
        
        try:
            req = urllib.request.Request(
                search_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36",
                    "Accept": "text/html",
                    "Accept-Language": "en-NG,en;q=0.9",
                },
            )
            
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="ignore")
            
            return self._parse_deals(html, vendor_id, max_price, min_discount)
        except Exception as e:
            _log.warning(f"Vendor search failed ({vendor_id}): {e}")
            return []
    
    def _parse_deals(
        self,
        html: str,
        vendor_id: str,
        max_price: float,
        min_discount: float,
    ) -> list[Deal]:
        """Parse deals from vendor HTML."""
        deals = []
        vendor = VENDORS[vendor_id]
        
        # Extract prices using vendor-specific patterns
        price_patterns = {
            "jumia": [
                r'data-price="(\d[\d,]*)"',
                r'₦\s*([\d,]+)',
                r'"price":"([\d.]+)"',
            ],
            "konga": [
                r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)',
                r'"price":\s*"?([\d,]+)"?',
            ],
            "jiji": [
                r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)',
            ],
            "slot": [
                r'class="price"[^>]*>₦\s*([\d,]+)',
            ],
            "payporte": [
                r'class="[^"]*price[^"]*"[^>]*>₦\s*([\d,]+)',
            ],
        }
        
        patterns = price_patterns.get(vendor_id, [r'₦\s*([\d,]+)'])
        
        # Find all prices
        all_prices = []
        for pattern in patterns:
            matches = re.findall(pattern, html)
            all_prices.extend(matches)
        
        # Extract titles (simplified)
        title_pattern = re.compile(r'class="[^"]*(?:name|title)[^"]*"[^>]*>([^<]+)')
        titles = title_pattern.findall(html)
        
        # Extract discount percentages
        discount_pattern = re.compile(r'(-?\d+)%')
        discounts = discount_pattern.findall(html)
        
        # Build deals
        for i, price_str in enumerate(all_prices[:20]):
            try:
                price = float(price_str.replace(",", ""))
                
                if price > max_price:
                    continue
                
                # Estimate original price and discount
                discount = 0.0
                if i < len(discounts):
                    try:
                        discount = abs(float(discounts[i]))
                    except ValueError:  # noqa: E103 - unparseable scraped discount treated as 0
                        pass
                
                if discount < min_discount:
                    continue
                
                original = price / (1 - discount / 100) if discount > 0 else price * 1.2
                
                title = titles[i].strip() if i < len(titles) else f"Product {i+1}"
                
                deal = Deal(
                    deal_id=new_id("deal"),
                    title=title,
                    vendor=vendor["name"],
                    current_price=price,
                    original_price=original,
                    discount_percent=discount,
                    url=vendor["base_url"],
                    deal_score=self._calculate_deal_score(
                        discount, vendor["trust_score"], price
                    ),
                    is_flash_sale=discount >= 50,
                )
                
                deals.append(deal)
            except (ValueError, IndexError):
                continue
        
        return deals
    
    def _calculate_deal_score(
        self,
        discount: float,
        trust: float,
        price: float,
    ) -> float:
        """Calculate deal score (0-100)."""
        # Discount contributes 50%
        discount_score = min(discount * 1.5, 50)
        
        # Trust contributes 30%
        trust_score = trust * 30
        
        # Price value contributes 20% (cheaper = better)
        price_score = max(0, 20 - (price / 50000))
        
        return min(100, discount_score + trust_score + price_score)
    
    def _detect_vendor_from_url(self, url: str) -> Optional[str]:
        """Detect vendor from product URL."""
        url_lower = url.lower()
        for vendor_id, vendor in VENDORS.items():
            if vendor["base_url"].replace("https://", "") in url_lower:
                return vendor_id
        return None
    
    async def _get_price(self, url: str, vendor: str) -> float:
        """Get current price from a product URL.

        Also sniffs stock status + promo codes and records them on the
        price-history curve (see :meth:`get_price_detail`).
        """
        detail = await self.get_price_detail(url, vendor)
        return detail["price"]

    async def get_price_detail(self, url: str, vendor: str) -> dict[str, Any]:
        """Price + stock + coupons from a product page."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36",
                    "Accept": "text/html",
                },
            )

            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="ignore")

            # Extract price
            price = 0.0
            price_patterns = [
                r'data-price="(\d[\d,]*)"',
                r'₦\s*([\d,]+)',
                r'"price":"([\d.]+)"',
            ]

            for pattern in price_patterns:
                match = re.search(pattern, html)
                if match:
                    price = float(match.group(1).replace(",", ""))
                    break

            # Stock sniffing
            html_l = html.lower()
            out_phrases = ("out of stock", "sold out", "currently unavailable",
                           "no longer available")
            in_stock = not any(p in html_l for p in out_phrases)

            # Coupon / promo-code sniffing
            coupons = sorted(set(
                m.group(1).upper()
                for m in re.finditer(
                    r"(?:coupon|promo(?:tion)?(?:al)?|voucher|discount)\s*"
                    r"(?:code)?\s*[:\-]?\s*([A-Z0-9]{4,16})",
                    html, re.IGNORECASE)))

            if price:
                # Record in history (with stock + coupon context)
                with self.db.transaction():
                    self.db.execute("""
                        INSERT OR REPLACE INTO price_history
                        (product_url, vendor, price, timestamp, in_stock,
                         coupon)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (url, vendor, price, time.time(),
                          int(in_stock), ",".join(coupons[:3])))

            return {"price": price, "in_stock": in_stock,
                    "coupons": coupons,
                    "is_lowest_90d": self._is_lowest_90d(url, price)
                    if price else False}
        except Exception as e:
            _log.warning(f"Failed to get price: {e}")

        return {"price": 0.0, "in_stock": True, "coupons": [],
                "is_lowest_90d": False}

    def _is_lowest_90d(self, product_url: str, price: float) -> bool:
        """True when ``price`` beats every point on the 90-day curve."""
        rows = self.db.query("""
            SELECT MIN(price) AS m FROM price_history
            WHERE product_url = ? AND timestamp > ?
        """, (product_url, time.time() - 90 * 86400))
        if not rows or rows[0]["m"] is None:
            return False
        return price <= float(rows[0]["m"])
    
    def _store_deal(self, deal: Deal) -> None:
        """Store a deal in the database."""
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO deals
                (deal_id, title, vendor, current_price, original_price, discount_percent,
                 url, image_url, category, deal_score, is_flash_sale, expires_at, in_stock, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                deal.deal_id, deal.title, deal.vendor, deal.current_price,
                deal.original_price, deal.discount_percent, deal.url, deal.image_url,
                deal.category, deal.deal_score, int(deal.is_flash_sale),
                deal.expires_at, int(deal.in_stock), deal.discovered_at,
            ))
    
    # ── Top Deals ────────────────────────────────────────────────────────────
    
    async def get_top_deals(self, *, limit: int = 10, category: str = "") -> list[Deal]:
        """Get top scored deals from database."""
        query = "SELECT * FROM deals WHERE in_stock = 1"
        params: list[Any] = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " ORDER BY deal_score DESC LIMIT ?"
        params.append(limit)
        
        rows = self.db.query(query, params)
        
        return [
            Deal(
                deal_id=r["deal_id"],
                title=r["title"],
                vendor=r["vendor"],
                current_price=r["current_price"],
                original_price=r["original_price"],
                discount_percent=r["discount_percent"],
                url=r["url"],
                image_url=r["image_url"],
                category=r["category"],
                deal_score=r["deal_score"],
                is_flash_sale=bool(r["is_flash_sale"]),
                in_stock=bool(r["in_stock"]),
                discovered_at=r["discovered_at"],
            )
            for r in rows
        ]
    
    async def get_deal_summary(self) -> str:
        """Generate a summary of current deals."""
        return await self.format_digest(limit=5)

    async def format_digest(self, *, limit: int = 10, category: str = "",
                            show_usd: bool = False) -> str:
        """God-tier daily deals briefing: ranked cards, not a flat list."""
        top = await self.get_top_deals(limit=limit, category=category)

        if not top:
            return ("🇳🇬 **naija deals**\n_no deals on the board right now — "
                    "try a search and I'll hunt._")

        # enrich with 90-day-low flags + FX
        fx = self._fx_rate()
        for deal in top:
            deal.is_lowest_90d = self._is_lowest_90d(deal.url,
                                                     deal.current_price)
            if fx:
                deal.usd_price = deal.current_price / fx

        lines = [f"🇳🇬 **top naija deals** ({len(top)})"]
        for i, deal in enumerate(top, 1):
            lines.append(f"\n**{i}.** " + deal.to_message(
                show_usd=show_usd).replace("\n", "\n    "))
        return "\n".join(lines)

    def _fx_rate(self) -> float:
        """USD→NGN for the FX toggle (cached 1h, best-effort)."""
        now = time.time()
        if now - getattr(self, "_fx_at", 0) < 3600 and getattr(
                self, "_fx", 0):
            return self._fx
        try:
            req = urllib.request.Request(
                "https://api.frankfurter.dev/v2/rate/USD/NGN",
                headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                self._fx = float(json.loads(resp.read().decode())["rate"])
                self._fx_at = now
                return self._fx
        except Exception:  # noqa: BLE001
            return getattr(self, "_fx", 0.0)

    @staticmethod
    def match_products(deals: list[Deal]) -> list[dict[str, Any]]:
        """Group deals that are the SAME product on different vendors.

        Word-order-invariant title matching → one product, all vendor
        prices side-by-side. Shared with the shopping engine.
        """
        groups: dict[str, dict[str, Any]] = {}
        for deal in deals:
            key = " ".join(sorted(
                re.sub(r"[^a-z0-9]+", " ",
                       (deal.title or "").lower()).split()))
            if not key:
                continue
            g = groups.setdefault(key, {"title": deal.title, "deals": []})
            g["deals"].append(deal)
        out = []
        for g in groups.values():
            ranked = sorted(g["deals"], key=lambda d: d.current_price)
            out.append({"title": g["title"], "deals": ranked,
                        "best": ranked[0] if ranked else None,
                        "vendors": len(ranked)})
        out.sort(key=lambda g: (g["best"].deal_score
                                if g["best"] else 0), reverse=True)
        return out

    @staticmethod
    def format_matches(groups: list[dict[str, Any]],
                       *, limit: int = 5) -> str:
        """Side-by-side vendor prices for matched products."""
        if not groups:
            return "_no cross-vendor matches._"
        lines = ["🔁 **same product, every vendor**"]
        for g in groups[:limit]:
            lines.append(f"\n**{g['title'][:60]}**")
            for d in g["deals"]:
                lines.append(f"  • ₦{d.current_price:,.0f} — {d.vendor} "
                             f"(`{d.deal_score:.0f}`)")
            if g["best"]:
                lines.append(f"  🏆 best: {g['best'].vendor} @ "
                             f"₦{g['best'].current_price:,.0f}")
        return "\n".join(lines)


def register(registry: Any) -> None:
    """Expose the Naija deal hunter as agent tools."""
    import asyncio

    def _hunter() -> "NaijaDealHunter":
        context = registry.context
        account_manager = getattr(context, "account_manager", None)
        db = getattr(context, "db", None)
        return NaijaDealHunter(account_manager, db)

    @registry.register(
        "naija_deals",
        description=(
            "Find deals across Nigerian marketplaces (Jumia, Konga, Jiji, Slot). "
            "Track prices, find flash sales, get top deals by category. "
            "action=find|flash|top|summary."
        ),
        capability="shopping.deals",
        parameters={
            "action": "str — find|flash|top|summary",
            "query": "str — product search query (for find)",
            "category": "str — category filter (for top)",
            "limit": "int — max results (default 10)",
        },
    )
    def _naija_deals(
        action: str = "top",
        query: str = "",
        category: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        hunter = _hunter()
        action = (action or "top").strip().lower()
        try:
            if action == "find" and query:
                deals = asyncio.run(hunter.find_deals(query, limit=int(limit)))
            elif action == "flash":
                deals = asyncio.run(hunter.find_flash_sales(limit=int(limit)))
            elif action == "summary":
                return {"ok": True, "summary": asyncio.run(hunter.get_deal_summary())}
            else:
                deals = asyncio.run(
                    hunter.get_top_deals(limit=int(limit), category=category or "")
                )
            return {
                "ok": True,
                "deals": [d.to_dict() if hasattr(d, "to_dict") else str(d) for d in deals],
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @registry.register(
        "naija_price_alert",
        description=(
            "Track a product URL for price drops across Nigerian vendors. "
            "Alerts fire when the price drops below the target."
        ),
        capability="shopping.deals",
        parameters={
            "product_url": "str — product page URL",
            "target_price": "float — alert when price drops to this (NGN)",
            "user_id": "str — owner user id for the alert",
        },
    )
    def _naija_price_alert(
        product_url: str, target_price: float, user_id: str = "owner"
    ) -> dict[str, Any]:
        hunter = _hunter()
        try:
            alert = asyncio.run(
                hunter.track_product(product_url, float(target_price), user_id)
            )
            return {
                "ok": True,
                "alert": alert.to_dict() if hasattr(alert, "to_dict") else str(alert),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
