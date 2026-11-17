"""Gapfix R2-36: browser-driven shopping cart + checkout.

Failing-first: the cart/checkout paths raise ShoppingError ("not yet
wired") or return a silent False. After the fix, add_to_cart drives
the real retailer flow through the browser tab, records the cart
entry, checkout stops honestly at the payment step, and every failure
mode is a loud ShoppingError naming the reason.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from nomorals.browser.errors import BrowserError
from nomorals.integrations.shopping_integration import (
    ShoppingError,
    ShoppingIntegration,
)


class FakeTab:
    """Sync stand-in for RenderedTab/Tab: the same verbs, canned pages.

    pages: url -> {"text": str, "buttons": {target: effect}, "links": [...]}
    A button effect is either a new page-text string (same URL) or a
    dict {"text": ..., "url": ...} (navigation). A click on a target
    that is not a button of the current page raises BrowserError, the
    way a real tab does for a missing selector.
    """

    def __init__(self, pages: dict) -> None:
        self.pages = pages
        self.calls: list[tuple] = []
        self.url = ""
        self._text = ""
        self._buttons: dict = {}
        self.closed = False

    def navigate(self, url: str) -> dict:
        self.calls.append(("navigate", url))
        page = self.pages.get(url)
        if page is None:
            raise BrowserError(f"no such page {url!r} (fake tab)")
        self.url = url
        self._text = page.get("text", "")
        self._buttons = page.get("buttons", {})
        return {"ok": True, "url": url, "title": page.get("title", "")}

    def click(self, target: str) -> dict:
        self.calls.append(("click", target))
        effect = self._buttons.get(target)
        if effect is None:
            # text-match fallback: a button whose label contains the target
            for label, eff in self._buttons.items():
                if target.lower() in label.lower():
                    effect = eff
                    target = label
                    break
        if effect is None:
            raise BrowserError(
                f"click {target!r} not found on {self.url}")
        if isinstance(effect, dict):
            self.url = effect.get("url", self.url)
            dest = self.pages.get(self.url, {})
            self._text = effect.get("text", dest.get("text", ""))
            self._buttons = effect.get("buttons", dest.get("buttons", {}))
        else:
            self._text = effect
        return {"ok": True, "target": target, "url": self.url}

    def select(self, name: str, value: str) -> dict:
        self.calls.append(("select", name, value))
        return {"ok": True, "name": name, "value": value}

    def text(self, max_chars: int = 40000) -> dict:
        return {"url": self.url, "text": self._text[:max_chars],
                "chars": len(self._text)}

    def links(self, max_links: int = 100) -> dict:
        page = self.pages.get(self.url, {})
        return {"url": self.url, "links": page.get("links", [])[:max_links]}

    def close(self) -> None:
        self.closed = True


class FakeService:
    """Stand-in for BrowserService: hands out one FakeTab per session."""

    def __init__(self, tab: FakeTab, *, rendered_ok: bool = True) -> None:
        self.tab = tab
        self.rendered_ok = rendered_ok
        self.opened: list[tuple] = []

    def open_rendered_tab(self, session_name: str, url: str = "") -> FakeTab:
        self.opened.append(("rendered", session_name, url))
        if not self.rendered_ok:
            raise BrowserError("playwright unavailable (fake)")
        if url:
            self.tab.navigate(url)
        return self.tab

    def open_session(self, name: str) -> FakeHandle:
        self.opened.append(("session", name))
        return FakeHandle(self.tab)


class FakeHandle:
    def __init__(self, tab: FakeTab) -> None:
        self._tab = tab

    def open_tab(self, url: str = "") -> FakeTab:
        if url:
            self._tab.navigate(url)
        return self._tab


AMAZON_PRODUCT = "https://www.amazon.com/dp/B0EXAMPLE"
AMAZON_CART = "https://www.amazon.com/gp/cart/view.html"

AMAZON_PAGES = {
    AMAZON_PRODUCT: {
        "text": "Sony WH-1000XM5 Wireless Headphones $349.99 In Stock",
        "buttons": {
            "#add-to-cart-button": (
                "Added to Cart\nSony WH-1000XM5 Wireless Headphones\n"
                "Cart subtotal (1 item): $349.99"),
        },
    },
    AMAZON_CART: {
        "text": "Shopping Cart\nSony WH-1000XM5 $349.99\nSubtotal (1 item): $349.99",
        "buttons": {
            "Proceed to checkout": {
                "url": "https://www.amazon.com/gp/buy/spc/handlers/display.html",
                "text": ("Checkout\nSelect a payment method\n"
                         "Add a new payment method\nShipping address\n"
                         "Review your order"),
            },
        },
    },
}


def _cart_path(tmp_home) -> str:
    return os.path.join(str(tmp_home), ".nomorals", "shopping", "cart.json")


def _read_cart(tmp_home) -> dict:
    path = _cart_path(tmp_home)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _integration(browser, tmp_home, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_home))
    return ShoppingIntegration(None, None, browser)


# ── add_to_cart ────────────────────────────────────────────────────────


def test_add_to_cart_drives_real_flow_and_records_cart(tmp_path, monkeypatch):
    tab = FakeTab(AMAZON_PAGES)
    shop = _integration(tab, tmp_path, monkeypatch)
    ok = asyncio.run(shop.add_to_cart(
        AMAZON_PRODUCT, account="buyer@example.com", quantity=1))
    assert ok is True
    assert ("navigate", AMAZON_PRODUCT) in tab.calls
    assert any(c[0] == "click" and "add-to-cart" in str(c[1])
               for c in tab.calls)
    cart = _read_cart(tmp_path)
    entries = cart.get("entries", [])
    assert len(entries) == 1
    assert entries[0]["url"] == AMAZON_PRODUCT
    assert entries[0]["retailer"] == "amazon"
    assert entries[0]["quantity"] == 1
    assert "Added to Cart" in entries[0].get("confirmation", "")


def test_add_to_cart_via_browser_service(tmp_path, monkeypatch):
    tab = FakeTab(AMAZON_PAGES)
    service = FakeService(tab)
    shop = _integration(service, tmp_path, monkeypatch)
    ok = asyncio.run(shop.add_to_cart(AMAZON_PRODUCT, account="buyer@example.com"))
    assert ok is True
    assert ("rendered", "shopping", AMAZON_PRODUCT) in service.opened
    assert _read_cart(tmp_path).get("entries")


def test_add_to_cart_service_falls_back_to_plain_tab(tmp_path, monkeypatch):
    tab = FakeTab(AMAZON_PAGES)
    service = FakeService(tab, rendered_ok=False)
    shop = _integration(service, tmp_path, monkeypatch)
    ok = asyncio.run(shop.add_to_cart(AMAZON_PRODUCT, account="buyer@example.com"))
    assert ok is True
    assert ("session", "shopping") in service.opened


def test_add_to_cart_no_browser_refuses_loudly(tmp_path, monkeypatch):
    shop = _integration(None, tmp_path, monkeypatch)
    with pytest.raises(ShoppingError, match="browser"):
        asyncio.run(shop.add_to_cart(AMAZON_PRODUCT, account="buyer@example.com"))


def test_add_to_cart_missing_button_is_a_recorded_failure(tmp_path, monkeypatch):
    pages = {
        AMAZON_PRODUCT: {
            "text": "Sony WH-1000XM5 — this page changed, no add button here",
            "buttons": {},
        },
    }
    tab = FakeTab(pages)
    shop = _integration(tab, tmp_path, monkeypatch)
    with pytest.raises(ShoppingError, match="add to cart|add-to-cart|button"):
        asyncio.run(shop.add_to_cart(AMAZON_PRODUCT, account="buyer@example.com"))
    cart = _read_cart(tmp_path)
    assert cart.get("entries") in ([], None) or not cart.get("entries")
    failures = cart.get("failures", [])
    assert len(failures) == 1
    assert failures[0]["url"] == AMAZON_PRODUCT
    assert failures[0]["reason"]
    assert failures[0]["retailer"] == "amazon"


def test_add_to_cart_unparsable_url_refuses(tmp_path, monkeypatch):
    tab = FakeTab(AMAZON_PAGES)
    shop = _integration(tab, tmp_path, monkeypatch)
    with pytest.raises(ShoppingError):
        asyncio.run(shop.add_to_cart("not a url at all", account="buyer@example.com"))


# ── checkout ─────────────────────────────────────────────────────────


def test_checkout_stops_at_payment_step(tmp_path, monkeypatch):
    tab = FakeTab(AMAZON_PAGES)
    shop = _integration(tab, tmp_path, monkeypatch)
    result = asyncio.run(shop.checkout("amazon", account="buyer@example.com"))
    assert result["status"] == "awaiting_payment"
    assert result["retailer"] == "amazon"
    assert "payment" in result.get("summary", "").lower()
    # the flow never clicked a place-order control
    assert not any("Place your order" in str(c) for c in tab.calls)


def test_checkout_unknown_retailer_names_it(tmp_path, monkeypatch):
    tab = FakeTab(AMAZON_PAGES)
    shop = _integration(tab, tmp_path, monkeypatch)
    with pytest.raises(ShoppingError, match="target"):
        asyncio.run(shop.checkout("target", account="buyer@example.com"))


def test_checkout_no_browser_refuses_loudly(tmp_path, monkeypatch):
    shop = _integration(None, tmp_path, monkeypatch)
    with pytest.raises(ShoppingError, match="browser"):
        asyncio.run(shop.checkout("amazon", account="buyer@example.com"))


def test_place_order_needs_explicit_approval(tmp_path, monkeypatch):
    tab = FakeTab(AMAZON_PAGES)
    shop = _integration(tab, tmp_path, monkeypatch)
    with pytest.raises(ShoppingError, match="approv"):
        asyncio.run(shop.place_order("amazon", account="buyer@example.com"))


def test_place_order_with_approval_verifies_confirmation(tmp_path, monkeypatch):
    pages = dict(AMAZON_PAGES)
    pages["https://www.amazon.com/gp/buy/spc/handlers/display.html"] = {
        "text": ("Checkout\nSelect a payment method\n"
                 "Review your order"),
        "buttons": {
            "Place your order": {
                "url": "https://www.amazon.com/gp/buy/thankyou/handlers/display.html",
                "text": ("Thank you for your order!\n"
                         "Order # 112-3948574-2222222\n"
                         "Order confirmation sent to buyer@example.com"),
            },
        },
    }
    tab = FakeTab(pages)
    shop = _integration(tab, tmp_path, monkeypatch)
    result = asyncio.run(shop.place_order(
        "amazon", account="buyer@example.com", confirm=True))
    assert result["status"] == "placed"
    assert result.get("order_ref")


# ── service flows directly ───────────────────────────────────────────


def test_service_flow_detects_retailer_from_url():
    from nomorals.browser.service import detect_retailer

    assert detect_retailer("https://www.amazon.com/dp/X") == "amazon"
    assert detect_retailer("https://www.ebay.com/itm/123") == "ebay"
    assert detect_retailer("https://www.walmart.com/ip/123") == "walmart"
    assert detect_retailer("https://www.bestbuy.com/site/x/1") == "bestbuy"
    assert detect_retailer("https://www.example-shop.com/p/1") == "generic"


def test_service_flow_ebay_add_to_cart():
    from nomorals.browser.service import shop_add_to_cart

    pages = {
        "https://www.ebay.com/itm/123": {
            "text": "Vintage Camera $89.00",
            "buttons": {"Add to cart": "Added to your cart\nVintage Camera"},
        },
    }
    tab = FakeTab(pages)
    tab.navigate("https://www.ebay.com/itm/123")
    result = shop_add_to_cart(tab, "ebay", quantity=1)
    assert result["ok"] is True
    assert result["retailer"] == "ebay"
    assert "cart" in result["confirmation"].lower()


def test_service_flow_checkout_generic():
    from nomorals.browser.service import shop_checkout_to_payment

    pages = {
        "https://www.example-shop.com/cart": {
            "text": "Your cart\nWidget $12.00",
            "buttons": {
                "Proceed to checkout": {
                    "url": "https://www.example-shop.com/checkout",
                    "text": "Checkout\nPayment method\nBilling address",
                },
            },
        },
    }
    tab = FakeTab(pages)
    tab.navigate("https://www.example-shop.com/cart")
    result = shop_checkout_to_payment(tab, "generic")
    assert result["ok"] is True
    assert result["status"] == "awaiting_payment"


# ── planner wiring ───────────────────────────────────────────────────


def test_planner_registers_cart_and_checkout_handlers():
    from nomorals.agents.planner import AgentPlanner

    planner = AgentPlanner()
    assert "add_to_cart" in planner._action_handlers
    assert "checkout" in planner._action_handlers


def test_planner_cart_handler_needs_shopping_integration():
    from nomorals.agents.planner import AgentPlanner

    planner = AgentPlanner()
    with pytest.raises(RuntimeError, match="Shopping integration"):
        asyncio.run(planner._action_add_to_cart(
            product_url=AMAZON_PRODUCT, account="buyer@example.com"))


def test_expose_make_shopping(tmp_path, monkeypatch):
    from nomorals.integrations.expose import make_shopping

    monkeypatch.setenv("HOME", str(tmp_path))
    shop = make_shopping(None, None, FakeTab(AMAZON_PAGES))
    assert isinstance(shop, ShoppingIntegration)
