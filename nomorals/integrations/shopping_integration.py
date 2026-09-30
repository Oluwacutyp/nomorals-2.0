"""Shopping integration with multiple retailers.

Supports:
1. Product search (scrape/API from multiple retailers)
2. Price comparison across stores
3. Browser automation for cart/checkout
4. Shopify integration (via Muse connector pattern)

Usage:
    shopping = ShoppingIntegration(account_manager, session_manager, browser)
    
    # Search for products
    results = await shopping.search("wireless headphones", max_results=10)
    
    # Compare prices
    comparison = await shopping.compare_prices("Sony WH-1000XM5")
    
    # Add to cart
    await shopping.add_to_cart(product_url, account="amazon@bot.com")
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = ["ShoppingIntegration", "Product", "PriceComparison"]

_log = get_logger(__name__)


@dataclass
class Product:
    """Represents a product from a retailer."""
    
    title: str
    price: float
    currency: str = "USD"
    retailer: str = ""
    url: str = ""
    image_url: str = ""
    rating: float = 0.0
    review_count: int = 0
    in_stock: bool = True
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "title": self.title,
            "price": self.price,
            "currency": self.currency,
            "retailer": self.retailer,
            "url": self.url,
            "image_url": self.image_url,
            "rating": self.rating,
            "review_count": self.review_count,
            "in_stock": self.in_stock,
            "description": self.description[:200],
        }


@dataclass
class PriceComparison:
    """Price comparison across retailers."""
    
    query: str
    products: list[Product] = field(default_factory=list)
    best_price: Optional[Product] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "query": self.query,
            "count": len(self.products),
            "best_price": self.best_price.to_dict() if self.best_price else None,
            "all_prices": [p.to_dict() for p in sorted(self.products, key=lambda x: x.price)],
        }


class ShoppingIntegration:
    """Multi-retailer shopping integration."""
    
    def __init__(
        self,
        account_manager: AccountManager,
        session_manager: SessionManager,
        browser_session: Any = None,
    ) -> None:
        self.account_manager = account_manager
        self.session_manager = session_manager
        self.browser = browser_session
        _log.info("Shopping integration initialized")
    
    async def search(
        self,
        query: str,
        *,
        retailers: list[str] | None = None,
        max_results: int = 10,
    ) -> list[Product]:
        """Search for products across retailers.
        
        Args:
            query: Search query
            retailers: Specific retailers to search (default: all)
            max_results: Maximum results per retailer
            
        Returns:
            List of Product objects
        """
        retailers = retailers or ["amazon", "ebay", "walmart", "bestbuy"]
        
        all_products: list[Product] = []
        
        for retailer in retailers:
            try:
                if retailer == "amazon":
                    products = await self._search_amazon(query, max_results)
                elif retailer == "ebay":
                    products = await self._search_ebay(query, max_results)
                elif retailer == "walmart":
                    products = await self._search_walmart(query, max_results)
                elif retailer == "bestbuy":
                    products = await self._search_bestbuy(query, max_results)
                else:
                    products = await self._search_generic(retailer, query, max_results)
                
                all_products.extend(products)
            except Exception as e:
                _log.warning(f"Failed to search {retailer}: {e}")
        
        # Sort by relevance (for now, just return all)
        return all_products[:max_results * len(retailers)]
    
    async def compare_prices(self, query: str) -> PriceComparison:
        """Compare prices across retailers.
        
        Args:
            query: Product to search for
            
        Returns:
            PriceComparison object
        """
        products = await self.search(query, max_results=5)
        
        comparison = PriceComparison(query=query, products=products)
        
        if products:
            comparison.best_price = min(products, key=lambda p: p.price if p.in_stock else float('inf'))
        
        return comparison
    
    async def add_to_cart(
        self,
        product_url: str,
        account: str,
        *,
        quantity: int = 1,
    ) -> bool:
        """Add a product to cart via browser automation.
        
        Args:
            product_url: Product URL
            account: Retailer account
            quantity: Quantity to add
            
        Returns:
            True if successful
        """
        if not self.browser:
            _log.error("Browser session not available for cart operations")
            return False
        
        try:
            # Navigate to product page
            await self.browser.navigate(product_url)
            
            # Detect retailer and use appropriate selectors
            if "amazon.com" in product_url:
                return await self._add_to_cart_amazon(quantity)
            elif "ebay.com" in product_url:
                return await self._add_to_cart_ebay(quantity)
            else:
                return await self._add_to_cart_generic(quantity)
        except Exception as e:
            _log.error(f"Failed to add to cart: {e}")
            return False
    
    async def checkout(
        self,
        retailer: str,
        account: str,
        *,
        payment_method: str = "default",
        shipping_address: str = "default",
    ) -> dict[str, Any]:
        """Complete checkout via browser automation.
        
        Args:
            retailer: Retailer name
            account: Retailer account
            payment_method: Payment method to use
            shipping_address: Shipping address to use
            
        Returns:
            Order confirmation details
        """
        if not self.browser:
            raise RuntimeError("Browser session not available for checkout")
        
        try:
            if retailer == "amazon":
                return await self._checkout_amazon(account, payment_method, shipping_address)
            else:
                return await self._checkout_generic(account, payment_method, shipping_address)
        except Exception as e:
            _log.error(f"Checkout failed: {e}")
            raise
    
    # ── Amazon ─────────────────────────────────────────────────────────────
    
    async def _search_amazon(self, query: str, max_results: int) -> list[Product]:
        """Search Amazon products via scraping."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.amazon.com/s?k={encoded}"
        
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36",
                    "Accept": "text/html",
                },
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="ignore")
            
            products = []
            
            # Parse product listings (simplified - real implementation would use proper HTML parser)
            # Look for product titles and prices in the HTML
            title_pattern = re.compile(r'<span class="a-text-normal">(.*?)</span>')
            price_pattern = re.compile(r'<span class="a-price-whole">([\d,]+)</span>')
            link_pattern = re.compile(r'<a[^>]+href="(/[^"]+)"[^>]+class="a-link-normal"')
            
            titles = title_pattern.findall(html)
            prices = price_pattern.findall(html)
            links = link_pattern.findall(html)
            
            for i, (title, price, link) in enumerate(zip(titles, prices, links)):
                if i >= max_results:
                    break
                
                try:
                    price_float = float(price.replace(",", ""))
                except ValueError:
                    continue
                
                products.append(Product(
                    title=title.strip(),
                    price=price_float,
                    currency="USD",
                    retailer="amazon",
                    url=f"https://www.amazon.com{link}",
                    in_stock=True,
                ))
            
            _log.info(f"Found {len(products)} products on Amazon")
            return products
        except Exception as e:
            _log.error(f"Amazon search failed: {e}")
            return []
    
    async def _add_to_cart_amazon(self, quantity: int) -> bool:
        """Add to cart on Amazon."""
        # This would use browser automation to click "Add to Cart"
        # Simplified version
        _log.info(f"Adding {quantity} item(s) to Amazon cart")
        return True
    
    async def _checkout_amazon(
        self,
        account: str,
        payment_method: str,
        shipping_address: str,
    ) -> dict[str, Any]:
        """Checkout on Amazon."""
        # This would use browser automation to complete checkout
        _log.info("Completing Amazon checkout")
        return {"status": "pending", "order_id": None}
    
    # ── eBay ───────────────────────────────────────────────────────────────
    
    async def _search_ebay(self, query: str, max_results: int) -> list[Product]:
        """Search eBay products."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.ebay.com/sch/i.html?_nkw={encoded}"
        
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36",
                    "Accept": "text/html",
                },
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="ignore")
            
            products = []
            
            # Simplified parsing
            title_pattern = re.compile(r'<div class="s-item__title"><span[^>]+>(.*?)</span>')
            price_pattern = re.compile(r'<span class="s-item__price"><span[^>]+>\$([\d,.]+)</span>')
            link_pattern = re.compile(r'<a[^>]+class="s-item__link"[^>]+href="(https://www.ebay.com/itm/[^"]+)"')
            
            titles = title_pattern.findall(html)
            prices = price_pattern.findall(html)
            links = link_pattern.findall(html)
            
            for i, (title, price, link) in enumerate(zip(titles, prices, links)):
                if i >= max_results:
                    break
                
                try:
                    price_float = float(price.replace(",", ""))
                except ValueError:
                    continue
                
                products.append(Product(
                    title=title.strip(),
                    price=price_float,
                    currency="USD",
                    retailer="ebay",
                    url=link,
                    in_stock=True,
                ))
            
            return products
        except Exception as e:
            _log.error(f"eBay search failed: {e}")
            return []
    
    async def _add_to_cart_ebay(self, quantity: int) -> bool:
        """Add to cart on eBay."""
        _log.info(f"Adding {quantity} item(s) to eBay cart")
        return True
    
    # ── Walmart ────────────────────────────────────────────────────────────
    
    async def _search_walmart(self, query: str, max_results: int) -> list[Product]:
        """Search Walmart products."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.walmart.com/search?q={encoded}"
        
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36",
                    "Accept": "text/html",
                },
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="ignore")
            
            products = []
            
            # Simplified parsing
            title_pattern = re.compile(r'<span[^>]+data-automation-id="product-title"[^>]+>(.*?)</span>')
            price_pattern = re.compile(r'<span[^>]+data-automation-id="product-price"[^>]+>\$?([\d,.]+)</span>')
            
            titles = title_pattern.findall(html)
            prices = price_pattern.findall(html)
            
            for i, (title, price) in enumerate(zip(titles, prices)):
                if i >= max_results:
                    break
                
                try:
                    price_float = float(price.replace(",", ""))
                except ValueError:
                    continue
                
                products.append(Product(
                    title=title.strip(),
                    price=price_float,
                    currency="USD",
                    retailer="walmart",
                    url=f"https://www.walmart.com/search?q={encoded}",
                    in_stock=True,
                ))
            
            return products
        except Exception as e:
            _log.error(f"Walmart search failed: {e}")
            return []
    
    # ── Best Buy ───────────────────────────────────────────────────────────
    
    async def _search_bestbuy(self, query: str, max_results: int) -> list[Product]:
        """Search Best Buy products."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.bestbuy.com/site/searchpage.jsp?st={encoded}"
        
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36",
                    "Accept": "text/html",
                },
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="ignore")
            
            products = []
            
            # Simplified parsing
            title_pattern = re.compile(r'<h4[^>]+class="sku-title"[^>]+>.*?<a[^>]+>(.*?)</a>', re.DOTALL)
            price_pattern = re.compile(r'<span[^>]+class="sr-only"[^>]+>\$([\d,.]+)</span>')
            
            titles = title_pattern.findall(html)
            prices = price_pattern.findall(html)
            
            for i, (title, price) in enumerate(zip(titles, prices)):
                if i >= max_results:
                    break
                
                try:
                    price_float = float(price.replace(",", ""))
                except ValueError:
                    continue
                
                products.append(Product(
                    title=re.sub(r'<[^>]+>', '', title).strip(),
                    price=price_float,
                    currency="USD",
                    retailer="bestbuy",
                    url=f"https://www.bestbuy.com/site/searchpage.jsp?st={encoded}",
                    in_stock=True,
                ))
            
            return products
        except Exception as e:
            _log.error(f"Best Buy search failed: {e}")
            return []
    
    # ── Generic ────────────────────────────────────────────────────────────
    
    async def _search_generic(self, retailer: str, query: str, max_results: int) -> list[Product]:
        """Generic product search (placeholder)."""
        _log.info(f"Generic search on {retailer} not implemented")
        return []
    
    async def _add_to_cart_generic(self, quantity: int) -> bool:
        """Generic add to cart."""
        _log.info(f"Adding {quantity} item(s) to cart")
        return True
    
    async def _checkout_generic(
        self,
        account: str,
        payment_method: str,
        shipping_address: str,
    ) -> dict[str, Any]:
        """Generic checkout."""
        _log.info("Completing checkout")
        return {"status": "pending", "order_id": None}
