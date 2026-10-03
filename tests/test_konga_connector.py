"""Konga buyer connector: offline tests against faked page plumbing.

Nothing here hits konga.com. The connector's two page sources are:

* the browser-service path (``_page_html`` via an injected fake
  BrowserService/Tab returning canned HTML), and
* the direct-HTTP fallback (a fake HttpClient)

The canned fixtures mirror the REAL konga.com search page shape probed
2026-10-02: ``https://www.konga.com/search?search=<query>`` returned
HTTP 200 (~260KB) with 5 ``<script type="application/ld+json">`` blocks
carrying schema.org Product entries: ``@id`` / ``url`` =
``https://www.konga.com/product/<slug>-<numeric-id>``, ``name``, image
(a Cloudinary URL), ``offers: {@type: "Offer", price: <number>,
priceCurrency: "NGN", availability: "https://schema.org/InStock"}``.

Pinned contracts:

* JSON-LD is the only parse source (HTML cards are never scraped)
* zero Product entries -> "structure changed" KongaError (never [])
* min 2s between requests (fake clock + recorded sleeps)
* 15-minute in-memory search cache
* auth_methods = (NONE,); PROVISIONABLE = ()
* orders -> human checkpoint (HumanCheckpointPending), never auto-submit
"""

from __future__ import annotations

import json
import time
import unittest
from typing import Any

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors import (
    AuthMethod,
    KongaConnector,
    KongaError,
    ConnectorError,
    get_connector,
)
from nomorals.connectors.checkpoints import (
    CheckpointKind,
    CheckpointState,
    CheckpointStore,
    HumanCheckpointPending,
)
from nomorals.connectors.konga import KONGA_BASE
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


# ── fixtures ─────────────────────────────────────────────────────────────

def _product_block(*products: dict[str, Any]) -> str:
    return json.dumps(list(products))


def _product(
    pid: int,
    name: str,
    price: float,
    *,
    availability: str = "https://schema.org/InStock",
    seller: str = "",
) -> dict[str, Any]:
    slug = name.lower().replace(" ", "-").replace("(", "").replace(")", "")
    offer: dict[str, Any] = {
        "@type": "Offer",
        "price": price,
        "priceCurrency": "NGN",
        "availability": availability,
    }
    if seller:
        offer["seller"] = {"@type": "Organization", "name": seller}
    return {
        "@context": "https://schema.org",
        "@type": "Product",
        "@id": f"https://www.konga.com/product/{slug}-{pid}",
        "url": f"https://www.konga.com/product/{slug}-{pid}",
        "name": name,
        "image": f"https://res.cloudinary.com/konga/image/{pid}.jpg",
        "offers": offer,
    }


def _page(*ld_blocks: str) -> str:
    scripts = "\n".join(
        f'<script type="application/ld+json">{b}</script>' for b in ld_blocks
    )
    return (
        "<!DOCTYPE html><html><head><title>Konga Search</title></head>"
        f"<body>{scripts}<div class='cards'>fragile html cards here</div>"
        "</body></html>"
    )


SEARCH_HTML = _page(
    _product_block(
        _product(101, "Samsung Galaxy A15 6GB RAM 128GB", 145000.0),
        _product(
            202,
            "Oraimo FreePods 3C Wireless Earbuds",
            18900.0,
            availability="https://schema.org/OutOfStock",
        ),
        _product(303, "Binatone 32 Inch HD TV", 98000.0, seller="Konga Retail"),
    ),
    # a non-Product block (WebSite) must be ignored, not crash parsing
    json.dumps({
        "@context": "https://schema.org",
        "@type": "WebSite",
        "url": "https://www.konga.com/",
    }),
    # a malformed block must be skipped, not poison the page
    "{not valid json",
)

DETAIL_HTML = _page(
    _product_block(
        _product(
            101,
            "Samsung Galaxy A15 6GB RAM 128GB",
            145000.0,
            seller="Konga Retail",
        )
    )
)

HOMEPAGE_HTML = _page(
    _product_block(_product(404, "HP Pavilion 15 Laptop", 785000.0))
)

NO_PRODUCTS_HTML = _page(
    json.dumps({"@context": "https://schema.org", "@type": "WebSite"})
)

HOMEPAGE_LINKS = {
    "links": [
        {"text": "Phones & Tablets", "url": "https://www.konga.com/category/phones-tablets-10"},
        {"text": "Electronics", "url": "https://www.konga.com/category/electronics-11"},
        {"text": "Fashion", "url": "https://www.konga.com/category/fashion-12#top"},
        {"text": "Phones & Tablets", "url": "https://www.konga.com/category/phones-tablets-10"},  # dup
        {"text": "Sell on Konga", "url": "https://www.konga.com/sell"},
        {"text": "Daily Deals", "url": "https://www.konga.com/deals"},
        {"text": "Track Order", "url": "https://www.konga.com/order/tracking"},
    ],
    "count": 7,
}


# ── fakes ────────────────────────────────────────────────────────────────

class FakeTab:
    def __init__(self, html: str, links: dict[str, Any] | None = None) -> None:
        self.tab_id = "tab-1"
        self.html_text = html
        self.links_result = links or {"links": []}
        self.error = ""
        self.html_calls = 0

    def html(self) -> dict[str, Any]:
        self.html_calls += 1
        return {"url": "https://www.konga.com/", "html": self.html_text}

    def links(self) -> dict[str, Any]:
        return self.links_result


class FakeSession:
    def __init__(self, tab: FakeTab) -> None:
        self.tab = tab
        self.tabs_open = 0

    def open_tab(self, url: str = "") -> FakeTab:
        self.tabs_open += 1
        return self.tab

    def close_tab(self, tab_id: str) -> None:
        pass


class FakeBrowserService:
    def __init__(self, tab: FakeTab) -> None:
        self.session = FakeSession(tab)
        self.closed_sessions: list[str] = []

    def open_session(self, name: str) -> FakeSession:
        return self.session

    def get_session(self, name: str) -> FakeSession:
        return self.session

    def close_session(self, name: str) -> None:
        self.closed_sessions.append(name)


class FakeHttp:
    """Scripted HttpClient stand-in; never touches the network."""

    def __init__(self, pages: dict[str, tuple[int, str]]) -> None:
        self.pages = pages  # url -> (status, body)
        self.requests: list[str] = []

    def get(self, url: str, headers: dict[str, str] | None = None) -> Any:
        self.requests.append(url)
        status, body = self.pages.get(url, (404, "not found"))
        return _Resp(status, body)


class _Resp:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# ── tests ────────────────────────────────────────────────────────────────

class KongaRegistrationTest(unittest.TestCase):
    def test_registered_and_declared(self) -> None:
        self.assertIs(get_connector("konga"), KongaConnector)
        self.assertEqual(KongaConnector.id, "konga")
        self.assertEqual(KongaConnector.auth_methods, (AuthMethod.NONE,))
        self.assertEqual(KongaConnector.PROVISIONABLE, ())


class KongaSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tab = FakeTab(SEARCH_HTML, HOMEPAGE_LINKS)
        self.browser = FakeBrowserService(self.tab)
        self.conn = KongaConnector(
            _vault(),
            browser_service=self.browser,
            clock=self.clock,
            sleeper=self.clock.sleep,
        )

    def test_search_parses_products(self) -> None:
        products = self.conn.search_products("samsung")
        self.assertEqual(len(products), 3)
        first = products[0]
        self.assertEqual(first["name"], "Samsung Galaxy A15 6GB RAM 128GB")
        self.assertEqual(first["price_ngn"], 145000.0)
        self.assertEqual(
            first["url"],
            "https://www.konga.com/product/"
            "samsung-galaxy-a15-6gb-ram-128gb-101",
        )
        self.assertEqual(first["image"], "https://res.cloudinary.com/konga/image/101.jpg")
        self.assertEqual(first["availability"], "in_stock")

    def test_search_normalizes_availability(self) -> None:
        products = self.conn.search_products("earbuds")
        by_name = {p["name"]: p for p in products}
        self.assertEqual(
            by_name["Oraimo FreePods 3C Wireless Earbuds"]["availability"],
            "out_of_stock",
        )

    def test_search_limit(self) -> None:
        products = self.conn.search_products("tv", limit=2)
        self.assertEqual(len(products), 2)

    def test_search_rejects_empty_query(self) -> None:
        with self.assertRaises(KongaError):
            self.conn.search_products("   ")

    def test_search_rejects_nonpositive_limit(self) -> None:
        with self.assertRaises(KongaError):
            self.conn.search_products("tv", limit=0)

    def test_search_structure_change_fails_fast(self) -> None:
        conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(NO_PRODUCTS_HTML)),
            clock=self.clock,
            sleeper=self.clock.sleep,
        )
        with self.assertRaises(KongaError) as ctx:
            conn.search_products("samsung")
        self.assertIn("structure changed", str(ctx.exception))

    def test_search_falls_back_to_http_when_browser_missing(self) -> None:
        http = FakeHttp({f"{KONGA_BASE}/search?search=samsung": (200, SEARCH_HTML)})
        conn = KongaConnector(_vault(), http=http,
                              clock=self.clock, sleeper=self.clock.sleep)
        conn._browser_service = None
        conn._browser_or_none = lambda: None  # type: ignore[method-assign]
        products = conn.search_products("samsung")
        self.assertEqual(len(products), 3)
        self.assertTrue(http.requests[0].startswith(f"{KONGA_BASE}/search"))

    def test_search_http_404_fails_fast(self) -> None:
        http = FakeHttp({})
        conn = KongaConnector(_vault(), http=http,
                              clock=self.clock, sleeper=self.clock.sleep)
        conn._browser_or_none = lambda: None  # type: ignore[method-assign]
        with self.assertRaises(KongaError) as ctx:
            conn.search_products("samsung")
        self.assertIn("not found", str(ctx.exception).lower())


class KongaRateLimitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tab = FakeTab(SEARCH_HTML)
        self.conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(self.tab),
            clock=self.clock,
            sleeper=self.clock.sleep,
        )

    def test_minimum_two_seconds_between_requests(self) -> None:
        self.conn.search_products("one")
        self.conn.search_products("two")  # different query: no cache hit
        self.conn.search_products("three")
        self.assertEqual(self.clock.sleeps, [2.0, 2.0])

    def test_first_request_needs_no_wait(self) -> None:
        self.conn.search_products("one")
        self.assertEqual(self.clock.sleeps, [])


class KongaCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.tab = FakeTab(SEARCH_HTML)
        self.conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(self.tab),
            clock=self.clock,
            sleeper=self.clock.sleep,
        )

    def test_cache_serves_second_query(self) -> None:
        first = self.conn.search_products("samsung")
        second = self.conn.search_products("samsung")
        self.assertEqual(first, second)
        self.assertEqual(self.tab.html_calls, 1)

    def test_cache_expires_after_ttl(self) -> None:
        self.conn.search_products("samsung")
        self.clock.now += 15 * 60.0 + 1  # past the 15-min TTL
        self.conn.search_products("samsung")
        self.assertEqual(self.tab.html_calls, 2)

    def test_detail_cached(self) -> None:
        detail_conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(DETAIL_HTML)),
            clock=self.clock,
            sleeper=self.clock.sleep,
        )
        url = "https://www.konga.com/product/samsung-galaxy-a15-6gb-ram-128gb-101"
        first = detail_conn.get_product(url)
        second = detail_conn.get_product(url)
        self.assertEqual(first, second)
        # one fetch for the detail page
        self.assertEqual(detail_conn._browser_service.session.tabs_open, 1)


class KongaDetailTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(DETAIL_HTML)),
            clock=self.clock,
            sleeper=self.clock.sleep,
        )

    def test_get_product_detail(self) -> None:
        detail = self.conn.get_product(
            "https://www.konga.com/product/samsung-galaxy-a15-6gb-ram-128gb-101"
        )
        self.assertEqual(detail["name"], "Samsung Galaxy A15 6GB RAM 128GB")
        self.assertEqual(detail["price_ngn"], 145000.0)
        self.assertEqual(detail["currency"], "NGN")
        self.assertEqual(detail["availability"], "in_stock")
        self.assertEqual(detail["seller"], "Konga Retail")
        self.assertEqual(
            detail["url"],
            "https://www.konga.com/product/samsung-galaxy-a15-6gb-ram-128gb-101",
        )

    def test_get_product_accepts_bare_slug(self) -> None:
        detail = self.conn.get_product("samsung-galaxy-a15-6gb-ram-128gb-101")
        self.assertEqual(detail["name"], "Samsung Galaxy A15 6GB RAM 128GB")

    def test_get_product_rejects_bare_numeric_id(self) -> None:
        with self.assertRaises(KongaError):
            self.conn.get_product("101")

    def test_get_product_rejects_foreign_url(self) -> None:
        with self.assertRaises(KongaError):
            self.conn.get_product("https://example.com/product/x-1")

    def test_get_product_structure_change_fails_fast(self) -> None:
        conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(NO_PRODUCTS_HTML)),
            clock=self.clock,
            sleeper=self.clock.sleep,
        )
        with self.assertRaises(KongaError) as ctx:
            conn.get_product("https://www.konga.com/product/x-1")
        self.assertIn("structure changed", str(ctx.exception))


class KongaLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()

    def _conn(self, html: str = HOMEPAGE_HTML) -> KongaConnector:
        return KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(html, HOMEPAGE_LINKS)),
            clock=self.clock,
            sleeper=self.clock.sleep,
        )

    def test_connect_verifies_homepage_jsonld(self) -> None:
        conn = self._conn()
        result = conn.connect()
        self.assertTrue(result.ok)
        self.assertEqual(result.account, "konga.com public")

    def test_connect_fails_fast_when_structure_changed(self) -> None:
        conn = self._conn(NO_PRODUCTS_HTML)
        with self.assertRaises(KongaError) as ctx:
            conn.connect()
        self.assertIn("structure changed", str(ctx.exception))

    def test_test_connection_true(self) -> None:
        self.assertTrue(self._conn().test_connection())

    def test_test_connection_false_on_structure_change(self) -> None:
        self.assertFalse(self._conn(NO_PRODUCTS_HTML).test_connection())

    def test_status_not_connected(self) -> None:
        status = self._conn().status()
        self.assertFalse(status.connected)

    def test_status_connected_after_connect(self) -> None:
        conn = self._conn()
        conn.connect()
        status = conn.status()
        self.assertTrue(status.connected)
        self.assertEqual(status.account, "konga.com public")

    def test_disconnect_clears_state(self) -> None:
        conn = self._conn()
        conn.connect()
        conn.search_products("samsung")
        service = conn._browser_service
        conn.disconnect()
        self.assertIn("konga", service.closed_sessions)
        self.assertEqual(conn._search_cache, {})
        self.assertFalse(conn.status().connected)
        conn.disconnect()  # idempotent

    def test_kongaerror_is_connector_error(self) -> None:
        self.assertTrue(issubclass(KongaError, ConnectorError))


class KongaCategoriesTest(unittest.TestCase):
    def test_list_categories_from_nav_links(self) -> None:
        clock = FakeClock()
        conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(HOMEPAGE_HTML, HOMEPAGE_LINKS)),
            clock=clock,
            sleeper=clock.sleep,
        )
        cats = conn.list_categories()
        names = [c["name"] for c in cats]
        self.assertEqual(names, ["Phones & Tablets", "Electronics", "Fashion"])
        self.assertTrue(
            all(c["url"].startswith(f"{KONGA_BASE}/category/") for c in cats)
        )

    def test_list_categories_fails_fast_without_nav(self) -> None:
        clock = FakeClock()
        conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(HOMEPAGE_HTML)),
            clock=clock,
            sleeper=clock.sleep,
        )
        with self.assertRaises(KongaError) as ctx:
            conn.list_categories()
        self.assertIn("structure changed", str(ctx.exception))


class KongaDealsTest(unittest.TestCase):
    def test_deals_discovers_and_parses(self) -> None:
        clock = FakeClock()
        conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(SEARCH_HTML, HOMEPAGE_LINKS)),
            clock=clock,
            sleeper=clock.sleep,
        )
        deals = conn.deals()
        self.assertEqual(len(deals), 3)
        self.assertEqual(deals[0]["name"], "Samsung Galaxy A15 6GB RAM 128GB")

    def test_deals_refuses_when_undiscoverable(self) -> None:
        clock = FakeClock()
        conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(HOMEPAGE_HTML)),
            clock=clock,
            sleeper=clock.sleep,
        )
        with self.assertRaises(KongaError) as ctx:
            conn.deals()
        self.assertIn("discoverable", str(ctx.exception))


class KongaOrderCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(SEARCH_HTML)),
            clock=self.clock,
            sleeper=self.clock.sleep,
        )

    def test_place_order_needs_items(self) -> None:
        db = Database(":memory:")
        with self.assertRaises(KongaError):
            self.conn.place_order([], db=db)

    def test_place_order_needs_db(self) -> None:
        with self.assertRaises(KongaError):
            self.conn.place_order([{"name": "TV"}])

    def test_place_order_raises_human_checkpoint(self) -> None:
        db = Database(":memory:")
        with self.assertRaises(HumanCheckpointPending) as ctx:
            self.conn.place_order(
                [{"name": "Samsung Galaxy A15",
                  "url": "https://www.konga.com/product/x-101"}],
                db=db,
            )
        cp = ctx.exception.checkpoint
        self.assertEqual(cp.state, CheckpointState.PENDING)
        self.assertEqual(cp.kind, CheckpointKind.MANUAL_STEP)
        self.assertIn("Samsung Galaxy A15", cp.instructions)

    def test_resume_unresolved_checkpoint_raises(self) -> None:
        db = Database(":memory:")
        with self.assertRaises(HumanCheckpointPending) as ctx:
            self.conn.place_order([{"name": "TV"}], db=db)
        cp = ctx.exception.checkpoint
        with self.assertRaises(KongaError):
            self.conn.resume_checkpoint(cp, db=db)

    def test_resume_resolved_checkpoint_returns_done(self) -> None:
        db = Database(":memory:")
        with self.assertRaises(HumanCheckpointPending) as ctx:
            self.conn.place_order([{"name": "TV"}], db=db)
        cp = ctx.exception.checkpoint
        resolved = CheckpointStore(db).resolve(cp.id)
        out = self.conn.resume_checkpoint(resolved, db=db)
        self.assertTrue(out["done"])
        self.assertIn("not automate", out["message"])

    def test_resume_unknown_stage_raises(self) -> None:
        db = Database(":memory:")
        store = CheckpointStore(db)
        cp = store.create(
            "konga", CheckpointKind.MANUAL_STEP, "t", "i",
            resume_state={"stage": "bogus"},
        )
        resolved = store.resolve(cp.id)
        with self.assertRaises(KongaError):
            self.conn.resume_checkpoint(resolved, db=db)

    def test_provision_nothing_provisionable(self) -> None:
        with self.assertRaises(ConnectorError):
            self.conn.provision("anything")
        self.assertFalse(self.conn.can_provision("anything"))


class KongaJsonLdParsingTest(unittest.TestCase):
    def test_product_in_nested_structure_found(self) -> None:
        html = _page(json.dumps({
            "@context": "https://schema.org",
            "@type": "ItemList",
            "itemListElement": [
                {"@type": "ListItem", "item": _product(9, "Nested Widget", 500.0)}
            ],
        }))
        entries = KongaConnector._product_entries(html)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "Nested Widget")

    def test_at_type_list_matches(self) -> None:
        html = _page(json.dumps({
            "@type": ["Product", "SomethingElse"],
            "name": "Multi Typed",
            "url": "https://www.konga.com/product/multi-typed-7",
            "offers": {"price": 100.0},
        }))
        entries = KongaConnector._product_entries(html)
        self.assertEqual(len(entries), 1)

    def test_entries_missing_price_are_skipped_not_fatal(self) -> None:
        html = _page(_product_block(
            _product(1, "Priced", 100.0),
            {"@type": "Product", "name": "No Price"},
        ))
        conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab(html)),
            clock=FakeClock(),
            sleeper=FakeClock().sleep,
        )
        products = conn.search_products("x")
        self.assertEqual([p["name"] for p in products], ["Priced"])

    def test_empty_html_is_structure_change(self) -> None:
        conn = KongaConnector(
            _vault(),
            browser_service=FakeBrowserService(FakeTab("<html></html>")),
            clock=FakeClock(),
            sleeper=FakeClock().sleep,
        )
        with self.assertRaises(KongaError):
            conn.search_products("x")


if __name__ == "__main__":
    unittest.main()
