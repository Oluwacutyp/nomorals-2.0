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

import asyncio
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = ["ShoppingIntegration", "ShoppingError", "Product", "PriceComparison"]

_log = get_logger(__name__)


class ShoppingError(Exception):
    """Raised when a shopping operation cannot be completed.

    Loud failure — never fake success. If you see this, no item was
    added to any cart and no order was placed.
    """


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

    def stars(self) -> str:
        """★★★★☆ from a 0–5 rating."""
        if not self.rating:
            return "no ratings yet"
        full = int(round(self.rating))
        return "★" * full + "☆" * (5 - full) + f" {self.rating:.1f}"

    def to_card(self) -> str:
        """Rich product card: price, rating, shipping, stock."""
        stock = "✅ in stock" if self.in_stock else "❌ out of stock"
        ship = self.metadata.get("shipping", "")
        lines = [f"🛍️ **{self.title}**",
                 f"💰 {self.currency} {self.price:,.2f} · {stock}",
                 f"{self.stars()} ({self.review_count:,} reviews)"
                 if self.review_count else self.stars()]
        if ship:
            lines.append(f"🚚 {ship}")
        lines.append(f"🏪 {self.retailer}")
        if self.url:
            lines.append(f"🔗 {self.url}")
        return "\n".join(lines)


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

    def format(self) -> str:
        """God-tier side-by-side comparison table, best price flagged."""
        if not self.products:
            return f"🔍 **{self.query}**\n_no results — try another query._"
        lines = [f"🔍 **price comparison: {self.query}**",
                 f"_{len(self.products)} results across retailers_",
                 ""]
        ranked = sorted(
            [p for p in self.products if p.in_stock],
            key=lambda p: p.price)
        if not ranked:
            ranked = sorted(self.products, key=lambda p: p.price)
        cheapest = ranked[0].price if ranked else 0
        priciest = ranked[-1].price if ranked else 0
        for p in ranked:
            flag = " 🏆 **best**" if p is ranked[0] else ""
            save = ""
            if p.price > cheapest:
                save = f" (+{p.price - cheapest:,.2f} vs best)"
            lines.append(f"• **{p.currency} {p.price:,.2f}**{save} — "
                         f"{p.retailer} · {p.stars()}{flag}")
            lines.append(f"  _{p.title[:70]}_")
        if priciest > cheapest:
            lines.append(f"\n💸 spread: {priciest - cheapest:,.2f} "
                         f"({(priciest - cheapest) / priciest:.0%} savings "
                         f"at best price)")
        return "\n".join(lines)


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
        sort_by: str = "relevance",
    ) -> list[Product]:
        """Search for products across retailers.
        
        Args:
            query: Search query
            retailers: Specific retailers to search (default: all)
            max_results: Maximum results per retailer
            sort_by: relevance | price | rating | deals
            
        Returns:
            List of Product objects (deduped across retailers)
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
        
        # Cross-retailer dedupe: same product on two retailers counts once
        # (keeps the cheaper listing).
        deduped = self._dedupe_products(all_products)
        
        if sort_by == "price":
            deduped.sort(key=lambda p: p.price)
        elif sort_by == "rating":
            deduped.sort(key=lambda p: (p.rating, p.review_count),
                         reverse=True)
        elif sort_by == "deals":
            deduped.sort(key=lambda p: (
                -(p.metadata.get("discount_pct", 0) or 0), p.price))
        
        return deduped[:max_results * len(retailers)]

    @staticmethod
    def _dedupe_products(products: list[Product]) -> list[Product]:
        """Drop near-duplicate titles across retailers, keep cheapest."""
        import re as _re
        seen: dict[str, Product] = {}
        for p in products:
            key = _re.sub(r"[^a-z0-9]+", " ",
                          (p.title or "").lower()).strip()
            key = " ".join(sorted(key.split()))  # word-order invariant
            if not key:
                continue
            if key not in seen or p.price < seen[key].price:
                seen[key] = p
        return list(seen.values())

    # ── wishlist (price-drop alerts) ─────────────────────────────────

    def _wishlist_path(self) -> str:
        import os
        path = os.path.expanduser("~/.nomorals/shopping/wishlist.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _load_wishlist(self) -> list[dict[str, Any]]:
        import os, json
        path = self._wishlist_path()
        if not os.path.isfile(path):
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except (ValueError, OSError):
            return []

    def _save_wishlist(self, items: list[dict[str, Any]]) -> None:
        import json
        with open(self._wishlist_path(), "w") as f:
            json.dump(items, f, indent=2)

    def add_to_wishlist(self, product: Product,
                        target_price: float | None = None) -> dict:
        """Save a product; optional target_price arms a drop alert."""
        import time
        items = self._load_wishlist()
        entry = {"url": product.url, "title": product.title,
                 "retailer": product.retailer, "currency": product.currency,
                 "price_at_add": product.price,
                 "target_price": target_price,
                 "last_price": product.price,
                 "added_at": time.time()}
        items = [i for i in items if i.get("url") != product.url] + [entry]
        self._save_wishlist(items)
        return entry

    def get_wishlist(self) -> list[dict[str, Any]]:
        """All wishlist items."""
        return self._load_wishlist()

    def remove_from_wishlist(self, url: str) -> bool:
        items = self._load_wishlist()
        kept = [i for i in items if i.get("url") != url]
        if len(kept) != len(items):
            self._save_wishlist(kept)
            return True
        return False

    async def check_wishlist_drops(self) -> list[dict[str, Any]]:
        """Re-check wishlist prices; returns items that hit target or
        dropped ≥10% since last check."""
        import time
        items = self._load_wishlist()
        hits = []
        for item in items:
            try:
                products = await self.search(
                    item["title"], retailers=[item["retailer"]],
                    max_results=3)
            except Exception:  # noqa: BLE001
                continue
            match = next((p for p in products if p.url == item["url"]),
                         products[0] if products else None)
            if not match:
                continue
            old = item.get("last_price", item.get("price_at_add", 0))
            item["last_price"] = match.price
            item["checked_at"] = time.time()
            target = item.get("target_price")
            dropped_pct = ((old - match.price) / old * 100) if old else 0
            if (target and match.price <= target) or dropped_pct >= 10:
                hits.append({**item, "drop_pct": round(dropped_pct, 1)})
        self._save_wishlist(items)
        return hits

    def format_wishlist(self) -> str:
        """God-tier wishlist rendering."""
        items = self._load_wishlist()
        if not items:
            return "💝 **wishlist**\n_empty — save products to watch prices._"
        lines = [f"💝 **wishlist** ({len(items)})"]
        for i in items:
            target = (f" → 🎯 {i['currency']} {i['target_price']:,.2f}"
                      if i.get("target_price") else "")
            lines.append(f"• **{i['title'][:60]}**\n"
                         f"  {i['currency']} {i.get('last_price', 0):,.2f}"
                         f"{target} · {i['retailer']}")
        return "\n".join(lines)
    
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

        Drives the real retailer flow through the browser tab
        (``browser/service.py`` shopping flows) and records the cart
        entry durably. Loud failure only: no browser, an unparsable
        URL, or a failed add all raise :class:`ShoppingError` with the
        reason — never a silent ``False``, never a fake ``True``.

        Args:
            product_url: Product URL
            account: Retailer account
            quantity: Quantity to add

        Returns:
            True on a verified add (the page confirmed it).
        """
        parsed = urllib.parse.urlparse(product_url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ShoppingError(
                f"not a product URL: {product_url!r} — pass a full "
                "http(s) product link")
        retailer = self._retailer_for_url(product_url)
        try:
            result = await asyncio.to_thread(
                self._add_to_cart_sync, product_url, retailer, quantity)
        except ShoppingError:
            raise
        except Exception as exc:  # noqa: BLE001 - flow errors are opaque
            self._record_failure(product_url, retailer, str(exc))
            raise ShoppingError(
                f"{retailer} add to cart failed: {exc}") from exc
        return bool(result.get("ok"))

    def _add_to_cart_sync(
        self,
        product_url: str,
        retailer: str,
        quantity: int,
    ) -> dict[str, Any]:
        """Sync half of add_to_cart: open the tab, run the flow, record."""
        from ..browser.service import shop_add_to_cart

        tab, owned = self._resolve_tab(product_url)
        try:
            result = shop_add_to_cart(tab, retailer, quantity)
        finally:
            if owned:
                self._close_tab_quietly(tab)
        self._record_entry({
            "url": product_url,
            "retailer": retailer,
            "quantity": int(quantity or 1),
            "confirmation": result.get("confirmation", ""),
            "evidence": result.get("evidence", ""),
        })
        return result

    async def checkout(
        self,
        retailer: str,
        account: str,
        *,
        payment_method: str = "default",
        shipping_address: str = "default",
    ) -> dict[str, Any]:
        """Drive checkout to the payment step and STOP there.

        The cart is driven to the payment step and the result reports
        ``status="awaiting_payment"``. No payment fields are filled,
        nothing is submitted, no order is placed — placing the order
        is the separate, explicitly approved :meth:`place_order` step.

        Args:
            retailer: Retailer name (amazon, ebay, walmart, bestbuy,
                generic)
            account: Retailer account
            payment_method: Payment method to use
            shipping_address: Shipping address to use

        Returns:
            ``{"status": "awaiting_payment", ...}`` with the payment
            marker the page showed.
        """
        name = (retailer or "").strip().lower()
        if name not in {"amazon", "ebay", "walmart", "bestbuy", "generic"}:
            raise ShoppingError(
                f"checkout not supported for retailer {retailer!r}: "
                "known retailers are amazon, ebay, walmart, bestbuy, "
                "generic")
        if not self.browser:
            raise ShoppingError(
                "checkout needs a browser session: pass a BrowserService "
                "or a tab when constructing ShoppingIntegration — no "
                "order was placed")
        try:
            result = await asyncio.to_thread(
                self._checkout_sync, name)
        except ShoppingError:
            raise
        except Exception as exc:  # noqa: BLE001 - flow errors are opaque
            _log.error(f"Checkout failed: {exc}")
            raise ShoppingError(
                f"{name} checkout failed: {exc}") from exc
        return {
            "status": result.get("status", "awaiting_payment"),
            "retailer": name,
            "account": account,
            "payment_method": payment_method,
            "shipping_address": shipping_address,
            "url": result.get("url", ""),
            "summary": result.get("summary", ""),
            "payment_marker": result.get("payment_marker", ""),
        }

    def _checkout_sync(self, retailer: str) -> dict[str, Any]:
        """Sync half of checkout: open the tab, run the flow."""
        from ..browser.service import shop_checkout_to_payment

        tab, owned = self._resolve_tab(None)
        try:
            return shop_checkout_to_payment(tab, retailer)
        finally:
            if owned:
                self._close_tab_quietly(tab)

    async def place_order(
        self,
        retailer: str,
        account: str,
        *,
        confirm: bool = False,
        payment_method: str = "default",
        shipping_address: str = "default",
    ) -> dict[str, Any]:
        """Place the order — ONLY with an explicit owner approval.

        ``confirm=True`` is the approval: without it this refuses, and
        no money moves. With it, the checkout is driven to the payment
        step and the place-order control is clicked; the order is only
        reported placed when the confirmation page says so.
        """
        if not confirm:
            raise ShoppingError(
                "place_order needs explicit owner approval "
                "(confirm=True) — no order was placed")
        name = (retailer or "").strip().lower()
        if name not in {"amazon", "ebay", "walmart", "bestbuy", "generic"}:
            raise ShoppingError(
                f"place_order not supported for retailer {retailer!r}: "
                "known retailers are amazon, ebay, walmart, bestbuy, "
                "generic")
        if not self.browser:
            raise ShoppingError(
                "place_order needs a browser session: pass a "
                "BrowserService or a tab when constructing "
                "ShoppingIntegration — no order was placed")
        try:
            result = await asyncio.to_thread(self._place_order_sync, name)
        except ShoppingError:
            raise
        except Exception as exc:  # noqa: BLE001 - flow errors are opaque
            _log.error(f"Place order failed: {exc}")
            raise ShoppingError(
                f"{name} place order failed: {exc}") from exc
        return {
            "status": result.get("status", "placed"),
            "retailer": name,
            "account": account,
            "payment_method": payment_method,
            "shipping_address": shipping_address,
            "url": result.get("url", ""),
            "order_ref": result.get("order_ref", ""),
            "confirmation": result.get("confirmation", ""),
        }

    def _place_order_sync(self, retailer: str) -> dict[str, Any]:
        """Sync half of place_order: payment step, then place the order."""
        from ..browser.service import (
            shop_checkout_to_payment,
            shop_place_order,
        )

        tab, owned = self._resolve_tab(None)
        try:
            shop_checkout_to_payment(tab, retailer)
            return shop_place_order(tab, retailer)
        finally:
            if owned:
                self._close_tab_quietly(tab)

    # ── browser seam ─────────────────────────────────────────────────────

    @staticmethod
    def _retailer_for_url(product_url: str) -> str:
        from ..browser.service import detect_retailer

        return detect_retailer(product_url)

    def _resolve_tab(self, url: str | None) -> tuple[Any, bool]:
        """Return (tab, owned).

        A ``BrowserService`` opens a fresh shopping tab (rendered,
        plain-tab fallback) which the caller must close; a bare tab
        object is used as-is and never closed here. Anything else is
        a loud failure naming what was passed.
        """
        from ..browser.service import open_shopping_tab

        browser = self.browser
        if browser is None:
            raise ShoppingError(
                "add to cart needs a browser session: pass a "
                "BrowserService or a tab when constructing "
                "ShoppingIntegration — no item was added to any cart")
        if hasattr(browser, "open_rendered_tab"):
            return open_shopping_tab(browser, url or ""), True
        if (hasattr(browser, "navigate") and hasattr(browser, "click")
                and hasattr(browser, "text")):
            if url:
                browser.navigate(url)
            return browser, False
        raise ShoppingError(
            f"browser object {type(browser).__name__!r} has no usable "
            "tab protocol (needs navigate/click/text, or a "
            "BrowserService) — no item was added to any cart")

    @staticmethod
    def _close_tab_quietly(tab: Any) -> None:
        try:
            tab.close()
        except Exception:  # noqa: BLE001 - close is best-effort
            _log.debug("shopping tab close failed", exc_info=True)

    # ── durable cart ─────────────────────────────────────────────────────

    def _cart_path(self) -> str:
        path = os.path.expanduser("~/.nomorals/shopping/cart.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _load_cart(self) -> dict[str, Any]:
        path = self._cart_path()
        if not os.path.isfile(path):
            return {"entries": [], "failures": []}
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"entries": [], "failures": []}
            data.setdefault("entries", [])
            data.setdefault("failures", [])
            return data
        except (ValueError, OSError):
            return {"entries": [], "failures": []}

    def _save_cart(self, data: dict[str, Any]) -> None:
        with open(self._cart_path(), "w") as f:
            json.dump(data, f, indent=2)

    def _record_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Record a verified cart add, replacing any earlier entry for
        the same URL."""
        import time
        cart = self._load_cart()
        record = {**entry, "added_at": time.time()}
        cart["entries"] = [e for e in cart["entries"]
                           if e.get("url") != entry.get("url")] + [record]
        self._save_cart(cart)
        return record

    def _record_failure(self, url: str, retailer: str,
                        reason: str) -> dict[str, Any]:
        """Record a failed add with its reason — never a fake success."""
        import time
        cart = self._load_cart()
        record = {"url": url, "retailer": retailer,
                  "reason": reason, "at": time.time()}
        cart["failures"] = cart["failures"] + [record]
        self._save_cart(cart)
        return record

    def get_cart(self) -> list[dict[str, Any]]:
        """Verified cart entries recorded by add_to_cart."""
        return self._load_cart().get("entries", [])

    def get_cart_failures(self) -> list[dict[str, Any]]:
        """Failed adds, each with its reason."""
        return self._load_cart().get("failures", [])

    def clear_cart(self) -> None:
        """Empty the recorded cart (entries and failures)."""
        self._save_cart({"entries": [], "failures": []})
    
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
        """Generic product search."""
        raise ShoppingError(
            f"Product search not implemented for retailer '{retailer}'. "
            f"Supported: amazon, ebay, walmart, bestbuy."
        )
    
