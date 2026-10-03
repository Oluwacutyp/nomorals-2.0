"""Jiji connector: offline tests against a fake rendered-tab browser service.

Everything runs against FakeBrowserService — canned rendered HTML, no real
network, no real jiji.ng, no real Chromium. The fake pins down the
contract this connector parses against:

* listing cards are <a> elements whose href matches
  jiji.ng/<location>/<category>/<slug>-<id>.html, carrying title, ₦ price
  (or "Contact for price"), and location as block-level text lines
* price strings: "₦1,200,000" -> 1200000, "Contact for price" -> None
* the homepage search form submits to /search (discovery fallback)
* rate limit: >= 3s between page loads; cache TTL 15 minutes
* seller contact / login are human-in-the-loop checkpoints only
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import nomorals.connectors.jiji as jiji_module
from nomorals.accounts.vault import CredentialVault
from nomorals.connectors import (
    AuthMethod,
    ConnectorError,
    get_connector,
)
from nomorals.connectors.checkpoints import (
    CheckpointKind,
    CheckpointState,
    CheckpointStore,
    HumanCheckpointPending,
)
from nomorals.connectors.jiji import (
    JIJI_BASE,
    JIJI_SEARCH_URL,
    JijiConnector,
    JijiError,
    extract_categories,
    extract_listing_cards,
    is_listing_url,
    parse_listing_detail,
    parse_price_ngn,
)
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


SEARCH_HTML = """\
<html><head><title>Jiji search: camry</title></head><body>
<form action="/search" method="get"><input name="query" type="text"/></form>
<nav>
  <a href="https://jiji.ng/cars">Cars</a>
  <a href="https://jiji.ng/mobile-phones-tablets">Mobile Phones &amp; Tablets</a>
</nav>
<div class="ads">
  <a href="https://jiji.ng/lagos/cars/toyota-camry-2015-se-AB12cd34.html">
    <img src="https://jiji.ng/img/camry.jpg"/>
    <div class="title">Toyota Camry 2015 SE</div>
    <div class="price">&#8358;4,500,000</div>
    <div class="location">Lekki, Lagos</div>
  </a>
  <a href="https://jiji.ng/abuja/cars/honda-accord-2018-XY98zz77.html">
    <img data-src="https://jiji.ng/img/accord.jpg"/>
    <div class="title">Honda Accord 2018</div>
    <div class="price">Contact for price</div>
    <div class="location">Maitama, Abuja</div>
  </a>
  <a href="https://jiji.ng/cars">Cars category</a>
</div>
</body></html>
"""

HOME_HTML = """\
<html><head><title>Jiji.ng: Buy and Sell Online</title></head><body>
<form action="/search" method="get"><input name="query" type="text"/></form>
<nav>
  <a href="https://jiji.ng/cars">Cars</a>
  <a href="https://jiji.ng/mobile-phones-tablets">Mobile Phones &amp; Tablets</a>
</nav>
<div class="fresh">
  <a href="https://jiji.ng/lagos/phones/iphone-13-pro-9z8y7x6w.html">
    <img src="https://jiji.ng/img/iphone.jpg"/>
    <div>iPhone 13 Pro 256GB</div>
    <div>&#8358;650,000</div>
    <div>Ikeja, Lagos</div>
  </a>
</div>
</body></html>
"""

NO_CARDS_HTML = """\
<html><head><title>Jiji search</title></head><body>
<form action="/search" method="get"><input name="query" type="text"/></form>
<p class="empty">No ads found for your search.</p>
</body></html>
"""

NO_SEARCH_HTML = """\
<html><head><title>Just a moment...</title></head><body>
<p>Verifying you are human. This may take a few seconds.</p>
</body></html>
"""

DETAIL_HTML = """\
<html><head><title>Toyota Camry 2015 SE - Cars - Jiji.ng</title>
<meta name="description" content="Neatly used Toyota Camry 2015, first body, buy and drive."/>
</head><body>
<h1>Toyota Camry 2015 SE</h1>
<div class="price">&#8358;4,500,000</div>
<div class="gallery">
  <img src="https://jiji.ng/img/camry1.jpg"/>
  <img src="https://jiji.ng/img/camry2.jpg"/>
</div>
<p>Seller: Adaeze Motors</p>
<p>Posted on 12 September 2026</p>
</body></html>
"""

NO_TITLE_HTML = """\
<html><head><title>Just a moment...</title></head><body>
<p>challenge</p>
</body></html>
"""


class FakeRenderedTab:
    """Stand-in for RenderedTab: canned rendered HTML per URL."""

    def __init__(self, tab_id: str, html: str, url: str) -> None:
        self.tab_id = tab_id
        self.session_name = "jiji"
        self.url = url
        self.title = "Jiji"
        self.history = [{"url": url, "title": "Jiji", "ts": 0.0}] if url else []
        self.error = ""
        self._html = html
        self.closed = False
        self.navigated: list[str] = []

    def navigate(self, url: str) -> dict[str, Any]:
        self.navigated.append(url)
        self.url = url
        self.history.append({"url": url, "title": self.title, "ts": 0.0})
        return {"url": url, "title": self.title, "ts": 0.0}

    def text(self, max_chars: int = 40000) -> dict[str, Any]:
        return {"url": self.url, "title": self.title, "chars": len(self._html),
                "text": self._html[:max_chars], "truncated": False}

    def links(self, max_links: int = 100) -> dict[str, Any]:
        return {"url": self.url, "count": 0, "links": []}

    def html(self, max_chars: int = 2_000_000) -> dict[str, Any]:
        return {"url": self.url, "title": self.title,
                "chars": len(self._html), "html": self._html[:max_chars],
                "truncated": False}

    def close(self) -> None:
        self.closed = True


class FakeBrowserService:
    """Stand-in for BrowserService.open_rendered_tab/close_rendered_tab."""

    def __init__(self, router: Any) -> None:
        self.router = router
        self.tabs: dict[str, FakeRenderedTab] = {}
        self.opens: list[str] = []
        self._n = 0

    def open_rendered_tab(self, session_name: str, url: str = "") -> FakeRenderedTab:
        self._n += 1
        tab = FakeRenderedTab(f"tab-{self._n}", self.router(url), url)
        self.tabs[tab.tab_id] = tab
        self.opens.append(url)
        return tab

    def close_rendered_tab(self, tab_id: str) -> None:
        tab = self.tabs.pop(tab_id)
        tab.close()

    def find_rendered_tab(self, tab_id: str) -> FakeRenderedTab | None:
        return self.tabs.get(tab_id)


def _router(url: str) -> str:
    if url in (JIJI_BASE, JIJI_BASE + "/"):
        return HOME_HTML
    if url.startswith("https://jiji.ng/search"):
        return SEARCH_HTML
    if "toyota-camry" in url:
        return DETAIL_HTML
    return "<html><body>nothing here</body></html>"


class FakeClock:
    """Deterministic stand-in for the time module (monotonic/sleep/time)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.now += seconds

    def time(self) -> float:
        return self.now


class _JijiTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.browser = FakeBrowserService(_router)
        self.vault = _vault()
        self._time_patch = mock.patch.object(jiji_module, "time", self.clock)
        self._time_patch.start()
        self.addCleanup(self._time_patch.stop)

    def _conn(self, router: Any = None) -> JijiConnector:
        return JijiConnector(
            self.vault,
            browser=FakeBrowserService(router or _router),
        )


class PriceParsingTests(unittest.TestCase):
    def test_naira_with_commas(self) -> None:
        self.assertEqual(parse_price_ngn("₦1,200,000"), 1200000)

    def test_naira_with_space(self) -> None:
        self.assertEqual(parse_price_ngn("₦ 250,000"), 250000)

    def test_contact_for_price_is_none(self) -> None:
        self.assertIsNone(parse_price_ngn("Contact for price"))

    def test_swap_is_none(self) -> None:
        self.assertIsNone(parse_price_ngn("Swap deal"))

    def test_empty_is_none(self) -> None:
        self.assertIsNone(parse_price_ngn(""))
        self.assertIsNone(parse_price_ngn("   "))

    def test_no_digits_is_none(self) -> None:
        self.assertIsNone(parse_price_ngn("₦"))

    def test_html_entity_decoded(self) -> None:
        # &#8358; is ₦ — the HTML parser decodes it before we see it.
        self.assertEqual(parse_price_ngn("\u20a64,500,000"), 4500000)


class ListingUrlTests(unittest.TestCase):
    def test_ad_url_matches(self) -> None:
        self.assertTrue(is_listing_url(
            "https://jiji.ng/lagos/cars/toyota-camry-2015-AB12cd34.html"))

    def test_category_url_rejected(self) -> None:
        self.assertFalse(is_listing_url("https://jiji.ng/cars"))
        self.assertFalse(is_listing_url("https://jiji.ng/search?query=camry"))

    def test_foreign_host_rejected(self) -> None:
        self.assertFalse(is_listing_url(
            "https://example.com/lagos/cars/ad-123.html"))

    def test_relative_url_rejected_without_base(self) -> None:
        self.assertFalse(is_listing_url("/lagos/cars/ad-123.html"))


class CardExtractionTests(unittest.TestCase):
    def test_search_cards(self) -> None:
        cards = extract_listing_cards(SEARCH_HTML)
        self.assertEqual(len(cards), 2)
        first, second = cards
        self.assertEqual(first["title"], "Toyota Camry 2015 SE")
        self.assertEqual(first["price_ngn"], 4500000)
        self.assertEqual(first["location"], "Lekki, Lagos")
        self.assertEqual(
            first["url"],
            "https://jiji.ng/lagos/cars/toyota-camry-2015-se-AB12cd34.html")
        self.assertEqual(first["image"], "https://jiji.ng/img/camry.jpg")
        self.assertIsNone(second["price_ngn"])
        self.assertEqual(second["title"], "Honda Accord 2018")
        self.assertEqual(second["location"], "Maitama, Abuja")
        # data-src fallback for lazy images
        self.assertEqual(second["image"], "https://jiji.ng/img/accord.jpg")

    def test_no_cards_found(self) -> None:
        self.assertEqual(extract_listing_cards(NO_CARDS_HTML), [])

    def test_category_link_not_a_card(self) -> None:
        urls = [c["url"] for c in extract_listing_cards(SEARCH_HTML)]
        self.assertNotIn("https://jiji.ng/cars", urls)

    def test_dedupe(self) -> None:
        doubled = SEARCH_HTML.replace(
            "AB12cd34.html\">", "AB12cd34.html#frag\">", 0)
        cards = extract_listing_cards(doubled + SEARCH_HTML)
        urls = [c["url"] for c in cards]
        self.assertEqual(len(urls), len(set(urls)))

    def test_categories(self) -> None:
        cats = extract_categories(SEARCH_HTML)
        by_name = {c["name"]: c["url"] for c in cats}
        self.assertEqual(by_name["Cars"], "https://jiji.ng/cars")
        self.assertEqual(by_name["Mobile Phones & Tablets"],
                         "https://jiji.ng/mobile-phones-tablets")

    def test_detail_parse(self) -> None:
        detail = parse_listing_detail(
            DETAIL_HTML,
            "https://jiji.ng/lagos/cars/toyota-camry-2015-se-AB12cd34.html")
        self.assertEqual(detail["title"], "Toyota Camry 2015 SE")
        self.assertEqual(detail["price_ngn"], 4500000)
        self.assertEqual(
            detail["description"],
            "Neatly used Toyota Camry 2015, first body, buy and drive.")
        self.assertEqual(detail["seller"], "Adaeze Motors")
        self.assertEqual(detail["images"],
                         ["https://jiji.ng/img/camry1.jpg",
                          "https://jiji.ng/img/camry2.jpg"])
        self.assertEqual(detail["posted_date"], "12 September 2026")

    def test_detail_without_title_fails_fast(self) -> None:
        with self.assertRaises(JijiError) as ctx:
            parse_listing_detail(NO_TITLE_HTML, "https://jiji.ng/x.html")
        self.assertIn("no listing title", str(ctx.exception))


class JijiSearchTests(_JijiTestBase):
    def test_search_returns_listing_shape(self) -> None:
        conn = self._conn()
        conn._search_url_pattern = JIJI_SEARCH_URL
        results = conn.search_listings("camry")
        self.assertEqual(len(results), 2)
        first = results[0]
        self.assertEqual(
            set(first),
            {"title", "price_ngn", "url", "location", "image",
             "seller_name"})
        self.assertEqual(first["title"], "Toyota Camry 2015 SE")
        self.assertEqual(first["price_ngn"], 4500000)
        self.assertEqual(first["location"], "Lekki, Lagos")
        self.assertEqual(first["image"], "https://jiji.ng/img/camry.jpg")
        self.assertEqual(first["seller_name"], "")
        self.assertIsNone(results[1]["price_ngn"])

    def test_search_limit(self) -> None:
        conn = self._conn()
        conn._search_url_pattern = JIJI_SEARCH_URL
        results = conn.search_listings("camry", limit=1)
        self.assertEqual(len(results), 1)

    def test_search_location_filter(self) -> None:
        conn = self._conn()
        conn._search_url_pattern = JIJI_SEARCH_URL
        results = conn.search_listings("cars", location="abuja")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Honda Accord 2018")

    def test_search_needs_query(self) -> None:
        conn = self._conn()
        with self.assertRaises(JijiError):
            conn.search_listings("  ")
        with self.assertRaises(JijiError):
            conn.search_listings("camry", limit=0)

    def test_search_discovers_pattern_from_homepage(self) -> None:
        conn = self._conn()
        browser = conn._browser_override
        assert browser is not None
        results = conn.search_listings("camry")
        self.assertEqual(len(results), 2)
        # one homepage render for discovery + one search render
        self.assertEqual(browser.opens,
                         [JIJI_BASE, "https://jiji.ng/search?query=camry"])
        self.assertEqual(conn._search_url_pattern, JIJI_SEARCH_URL)

    def test_search_structure_changed_fails_fast(self) -> None:
        def router(url: str) -> str:
            if url.startswith("https://jiji.ng/search"):
                return NO_CARDS_HTML
            return _router(url)

        conn = self._conn(router)
        conn._search_url_pattern = JIJI_SEARCH_URL
        with self.assertRaises(JijiError) as ctx:
            conn.search_listings("camry")
        self.assertIn("page structure changed", str(ctx.exception))

    def test_search_discovery_failure_fails_fast(self) -> None:
        conn = self._conn(lambda url: NO_SEARCH_HTML)
        with self.assertRaises(JijiError) as ctx:
            conn.search_listings("camry")
        self.assertIn("page structure changed", str(ctx.exception))

    def test_rate_limit_between_requests(self) -> None:
        conn = self._conn()
        conn._search_url_pattern = JIJI_SEARCH_URL
        conn.search_listings("camry")
        self.assertEqual(self.clock.slept, 0.0)  # first request is immediate
        conn.search_listings("accord")
        self.assertGreaterEqual(self.clock.slept, 3.0)

    def test_search_cache_hit_skips_network(self) -> None:
        conn = self._conn()
        browser = conn._browser_override
        assert browser is not None
        conn._search_url_pattern = JIJI_SEARCH_URL
        first = conn.search_listings("camry")
        self.assertEqual(len(browser.opens), 1)
        second = conn.search_listings("camry")
        self.assertEqual(len(browser.opens), 1)  # no new render
        self.assertEqual(first, second)

    def test_search_cache_ttl_expiry(self) -> None:
        conn = self._conn()
        browser = conn._browser_override
        assert browser is not None
        conn._search_url_pattern = JIJI_SEARCH_URL
        conn.search_listings("camry")
        self.assertEqual(len(browser.opens), 1)
        self.clock.now += 901.0  # past the 15-minute TTL
        conn.search_listings("camry")
        self.assertEqual(len(browser.opens), 2)


class JijiListingTests(_JijiTestBase):
    def test_get_listing(self) -> None:
        conn = self._conn()
        detail = conn.get_listing(
            "https://jiji.ng/lagos/cars/toyota-camry-2015-se-AB12cd34.html")
        self.assertEqual(detail["title"], "Toyota Camry 2015 SE")
        self.assertEqual(detail["price_ngn"], 4500000)
        self.assertEqual(detail["seller"], "Adaeze Motors")
        self.assertEqual(len(detail["images"]), 2)
        self.assertEqual(detail["posted_date"], "12 September 2026")

    def test_get_listing_cached(self) -> None:
        conn = self._conn()
        browser = conn._browser_override
        assert browser is not None
        url = "https://jiji.ng/lagos/cars/toyota-camry-2015-se-AB12cd34.html"
        conn.get_listing(url)
        conn.get_listing(url)
        self.assertEqual(len(browser.opens), 1)

    def test_get_listing_needs_url(self) -> None:
        conn = self._conn()
        with self.assertRaises(JijiError):
            conn.get_listing("")

    def test_get_listing_challenge_page_fails_fast(self) -> None:
        conn = self._conn(lambda url: NO_TITLE_HTML)
        with self.assertRaises(JijiError):
            conn.get_listing("https://jiji.ng/lagos/cars/x-123.html")

    def test_list_categories(self) -> None:
        conn = self._conn()
        cats = conn.list_categories()
        names = [c["name"] for c in cats]
        self.assertIn("Cars", names)
        self.assertIn("Mobile Phones & Tablets", names)

    def test_list_categories_cached(self) -> None:
        conn = self._conn()
        browser = conn._browser_override
        assert browser is not None
        conn.list_categories()
        conn.list_categories()
        self.assertEqual(len(browser.opens), 1)


class JijiLifecycleTests(_JijiTestBase):
    def test_connect_verifies_rendered_listings(self) -> None:
        conn = self._conn()
        browser = conn._browser_override
        assert browser is not None
        result = conn.connect()
        self.assertTrue(result.ok)
        self.assertEqual(result.account, "jiji.ng public")
        self.assertEqual(browser.opens, [JIJI_BASE])
        self.assertEqual(len(browser.tabs), 0)  # tab closed after verify
        status = conn.status()
        self.assertTrue(status.connected)
        self.assertEqual(status.account, "jiji.ng public")

    def test_connect_fails_fast_without_cards(self) -> None:
        conn = self._conn(lambda url: NO_CARDS_HTML)
        with self.assertRaises(JijiError) as ctx:
            conn.connect()
        self.assertIn("page structure changed", str(ctx.exception))
        self.assertFalse(conn.status().connected)

    def test_disconnect_closes_tabs(self) -> None:
        conn = self._conn()
        browser = conn._browser_override
        assert browser is not None
        conn.connect()
        # simulate a leaked tab the connector still tracks
        leaked = browser.open_rendered_tab("jiji", JIJI_BASE)
        conn._tab_ids.append(leaked.tab_id)
        conn.disconnect()
        self.assertTrue(leaked.closed)
        self.assertFalse(conn.status().connected)

    def test_test_connection(self) -> None:
        conn = self._conn()
        self.assertFalse(conn.test_connection())  # not connected yet
        conn.connect()
        self.assertTrue(conn.test_connection())

    def test_test_connection_false_on_broken_render(self) -> None:
        conn = self._conn(lambda url: NO_CARDS_HTML)
        conn._connected = True  # pretend an earlier connect succeeded
        self.assertFalse(conn.test_connection())

    def test_render_failure_wraps_in_jiji_error(self) -> None:
        class BrokenBrowser(FakeBrowserService):
            def open_rendered_tab(self, session_name: str, url: str = ""):
                raise RuntimeError("chromium exploded")

        conn = JijiConnector(self.vault, browser=BrokenBrowser(_router))
        with self.assertRaises(JijiError) as ctx:
            conn.connect()
        self.assertIn("could not render", str(ctx.exception))


class JijiHumanLoopTests(_JijiTestBase):
    def test_contact_seller_raises_checkpoint(self) -> None:
        db = Database(":memory:")
        conn = self._conn()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.contact_seller(
                "https://jiji.ng/lagos/cars/toyota-camry-2015-se-AB12cd34.html",
                "Hello, is this still available?",
                db=db,
            )
        cp = ctx.exception.checkpoint
        self.assertEqual(cp.kind, CheckpointKind.MANUAL_STEP)
        self.assertEqual(cp.state, CheckpointState.PENDING)
        self.assertIn("toyota-camry", cp.instructions)

    def test_contact_seller_resume(self) -> None:
        db = Database(":memory:")
        conn = self._conn()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.contact_seller("https://jiji.ng/x.html", "hi", db=db)
        cp = ctx.exception.checkpoint
        store = CheckpointStore(db)
        store.resolve(cp.id)
        out = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertTrue(out["done"])
        self.assertEqual(out["identity"], "owner")
        self.assertEqual(out["listing_url"], "https://jiji.ng/x.html")

    def test_contact_seller_needs_db_and_message(self) -> None:
        conn = self._conn()
        with self.assertRaises(JijiError):
            conn.contact_seller("https://jiji.ng/x.html", "hi")
        with self.assertRaises(JijiError):
            conn.contact_seller("https://jiji.ng/x.html", "  ",
                                db=Database(":memory:"))
        with self.assertRaises(JijiError):
            conn.contact_seller("", "hi", db=Database(":memory:"))

    def test_login_checkpoint_flow(self) -> None:
        db = Database(":memory:")
        conn = self._conn()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.login(db=db)
        cp = ctx.exception.checkpoint
        self.assertEqual(cp.kind, CheckpointKind.MANUAL_STEP)
        store = CheckpointStore(db)
        store.resolve(cp.id)
        out = conn.resume_checkpoint(store.get(cp.id), db=db)
        self.assertTrue(out["done"])
        self.assertEqual(out["identity"], "owner")

    def test_login_needs_db(self) -> None:
        conn = self._conn()
        with self.assertRaises(JijiError):
            conn.login()

    def test_resume_unresolved_raises(self) -> None:
        db = Database(":memory:")
        conn = self._conn()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            conn.login(db=db)
        cp = ctx.exception.checkpoint
        with self.assertRaises(JijiError):
            conn.resume_checkpoint(cp, db=db)

    def test_resume_unknown_stage_raises(self) -> None:
        from nomorals.connectors.checkpoints import request_human_action

        db = Database(":memory:")
        conn = self._conn()
        with self.assertRaises(HumanCheckpointPending) as ctx:
            request_human_action(
                "jiji", CheckpointKind.MANUAL_STEP, "title", "instructions",
                db=db, resume_state={"stage": "bogus"},
            )
        cp = ctx.exception.checkpoint
        store = CheckpointStore(db)
        store.resolve(cp.id)
        with self.assertRaises(JijiError):
            conn.resume_checkpoint(store.get(cp.id), db=db)

    def test_provisionable_empty(self) -> None:
        conn = self._conn()
        self.assertEqual(conn.PROVISIONABLE, ())
        with self.assertRaises(ConnectorError):
            conn.provision("anything")


class JijiRegistryTests(unittest.TestCase):
    def test_registered(self) -> None:
        cls = get_connector("jiji")
        self.assertIs(cls, JijiConnector)

    def test_no_auth(self) -> None:
        self.assertEqual(JijiConnector.auth_methods, (AuthMethod.NONE,))

    def test_imports_from_package(self) -> None:
        from nomorals.connectors import JijiConnector as C2
        from nomorals.connectors import JijiError as E2
        self.assertIs(C2, JijiConnector)
        self.assertIs(E2, JijiError)


if __name__ == "__main__":
    unittest.main()
