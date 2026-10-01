"""Shopping engine god-tier tests: new sites, FX helper, compare, parsing, dedupe.

Covers:
- new sites present in SITES with complete configs
- FX helper: cached value reuse, clean fallback when the network fails (mocked)
- compare_prices merges mocked per-site results, picks best deal, computes
  median + savings, and stays fail-soft when one site dies
- price-pattern parsing for new sites (kara, ebay, banggood) on canned HTML
- title extraction + near-duplicate snapshot dedupe
- `compare` action in the deals tool layer

No live network anywhere in this file.
"""

import asyncio
import unittest
from unittest import mock
from urllib.error import URLError

from nomorals.integrations import naija_shopping
from nomorals.integrations.naija_shopping import (
    FALLBACK_RATES,
    SITES,
    NaijaShoppingEngine,
    PriceSnapshot,
    _dedupe_snapshots,
    clear_fx_cache,
    extract_titles,
    get_fx_rate,
)
from nomorals.storage.db import Database
from nomorals.tools.browser import BrowserSession


def _make_engine():
    return NaijaShoppingEngine(Database(":memory:"), BrowserSession())


class NewSitesConfigTests(unittest.TestCase):
    def test_new_sites_present(self):
        for site_id in ("kara", "slot", "ebay", "banggood"):
            self.assertIn(site_id, SITES, f"{site_id} missing from SITES")

    def test_all_sites_have_valid_configs(self):
        required = {
            "name", "base_url", "search_path", "trust_score",
            "bot_protection", "min_delay", "max_delay", "currency",
        }
        for site_id, cfg in SITES.items():
            with self.subTest(site=site_id):
                self.assertTrue(required <= set(cfg), f"{site_id} missing keys")
                self.assertTrue(cfg["base_url"].startswith("https://"))
                self.assertIn("{query}", cfg["search_path"])
                self.assertIn(cfg["currency"], ("NGN", "USD", "EUR", "GBP"))
                self.assertIn(cfg["bot_protection"], ("low", "medium", "high"))
                self.assertGreaterEqual(cfg["trust_score"], 0)
                self.assertLessEqual(cfg["trust_score"], 1)
                self.assertLess(cfg["min_delay"], cfg["max_delay"])

    def test_site_count_grew(self):
        # 6 original + 4 new
        self.assertEqual(len(SITES), 10)

    def test_old_sites_kept_currency(self):
        self.assertEqual(SITES["jumia"]["currency"], "NGN")
        self.assertEqual(SITES["amazon"]["currency"], "USD")
        self.assertEqual(SITES["aliexpress"]["currency"], "USD")


class FxHelperTests(unittest.TestCase):
    def setUp(self):
        clear_fx_cache()
        self.addCleanup(clear_fx_cache)

    def test_cached_value_reused(self):
        with mock.patch.object(
            naija_shopping, "_fetch_fx_rate", return_value=1400.0
        ) as fetch:
            first = get_fx_rate("USD")
            second = get_fx_rate("usd")  # case-insensitive, same cache key
            self.assertEqual(first, 1400.0)
            self.assertEqual(second, 1400.0)
            self.assertEqual(fetch.call_count, 1)

    def test_fallback_when_network_fails(self):
        with mock.patch.object(
            naija_shopping, "_fetch_fx_rate", side_effect=URLError("nope")
        ):
            for cur, expected in (
                ("USD", FALLBACK_RATES["USD"]),
                ("EUR", FALLBACK_RATES["EUR"]),
                ("GBP", FALLBACK_RATES["GBP"]),
            ):
                with self.subTest(currency=cur):
                    self.assertEqual(get_fx_rate(cur), expected)

    def test_fallback_is_cached(self):
        with mock.patch.object(
            naija_shopping, "_fetch_fx_rate", side_effect=URLError("nope")
        ) as fetch:
            get_fx_rate("USD")
            get_fx_rate("USD")
            self.assertEqual(fetch.call_count, 1)

    def test_same_currency_returns_one(self):
        with mock.patch.object(naija_shopping, "_fetch_fx_rate") as fetch:
            self.assertEqual(get_fx_rate("NGN", "NGN"), 1.0)
            fetch.assert_not_called()

    def test_unknown_currency_raises(self):
        with mock.patch.object(
            naija_shopping, "_fetch_fx_rate", side_effect=URLError("nope")
        ):
            with self.assertRaises(ValueError):
                get_fx_rate("XYZ")


KARA_HTML = """
<title>Search results for iphone | Kara Nigeria</title>
<div class="product"><h3>Apple iPhone 13 128GB Midnight</h3>
<span class="price">₦ 745,000</span></div>
<div class="product"><h3>Samsung Galaxy S23 256GB Cream</h3>
<span class="price">₦ 689,000</span></div>
"""

EBAY_HTML = """
<h3 class="s-item__title">iPhone 13 128GB - Unlocked</h3>
<span class="s-item__price">$299.99</span>
"""

BANGGOOD_HTML = """
<meta property="og:title" content="Original Xiaomi Redmi Note 13 Pro Smartphone" />
<script>var data = {"salePrice":"189.90","currency":"USD"};</script>
"""


class ParseNewSitesTests(unittest.TestCase):
    def setUp(self):
        self.engine = _make_engine()
        # Keep parsing tests fully offline: pin the FX rate.
        patcher = mock.patch.object(naija_shopping, "get_fx_rate", return_value=1500.0)
        self.mock_fx = patcher.start()
        self.addCleanup(patcher.stop)

    def test_kara_ngn_parsing_with_real_titles(self):
        snaps = self.engine._parse_search_results(KARA_HTML, "kara", float("inf"))
        self.assertGreaterEqual(len(snaps), 2)
        by_title = {s.title: s.price_ngn for s in snaps}
        self.assertIn("Apple iPhone 13 128GB Midnight", by_title)
        self.assertAlmostEqual(by_title["Apple iPhone 13 128GB Midnight"], 745000.0)
        self.assertIn("Samsung Galaxy S23 256GB Cream", by_title)
        # Kara prices are already NGN: no conversion applied
        self.mock_fx.assert_not_called()

    def test_ebay_usd_parsing_converts_via_fx(self):
        snaps = self.engine._parse_search_results(EBAY_HTML, "ebay", float("inf"))
        self.assertEqual(len(snaps), 1)
        snap = snaps[0]
        self.assertEqual(snap.site, "ebay")
        self.assertAlmostEqual(snap.price_ngn, 299.99 * 1500.0)
        self.assertEqual(snap.title, "iPhone 13 128GB - Unlocked")
        self.mock_fx.assert_called_with("USD", "NGN")

    def test_banggood_saleprice_parsing(self):
        snaps = self.engine._parse_search_results(BANGGOOD_HTML, "banggood", float("inf"))
        self.assertEqual(len(snaps), 1)
        snap = snaps[0]
        self.assertAlmostEqual(snap.price_ngn, 189.90 * 1500.0)
        self.assertEqual(snap.title, "Original Xiaomi Redmi Note 13 Pro Smartphone")

    def test_noise_prices_filtered(self):
        html = '<span class="price">₦ 50</span><span class="price">₦ 0</span>'
        snaps = self.engine._parse_search_results(html, "kara", float("inf"))
        self.assertEqual(snaps, [])

    def test_max_price_filter(self):
        snaps = self.engine._parse_search_results(KARA_HTML, "kara", 700000)
        self.assertEqual(len(snaps), 1)
        self.assertAlmostEqual(snaps[0].price_ngn, 689000.0)


class TitleExtractionTests(unittest.TestCase):
    def test_prefers_product_headings_over_page_title(self):
        titles = extract_titles(KARA_HTML)
        self.assertIn("Apple iPhone 13 128GB Midnight", titles)
        self.assertNotIn("Search results for iphone", titles)

    def test_og_title_used(self):
        titles = extract_titles(BANGGOOD_HTML)
        self.assertIn("Original Xiaomi Redmi Note 13 Pro Smartphone", titles)

    def test_dedupes_repeated_titles(self):
        html = "<h3>Same Phone</h3><h3>Same Phone</h3>"
        self.assertEqual(extract_titles(html), ["Same Phone"])


class DedupeTests(unittest.TestCase):
    def _snap(self, site, price, title="iPhone 13 128GB"):
        return PriceSnapshot(
            product_url=f"https://{site}/p1", site=site,
            price_ngn=price, title=title,
        )

    def test_same_site_price_within_1pct_deduped(self):
        snaps = [self._snap("kara", 100000.0), self._snap("kara", 100500.0)]
        self.assertEqual(len(_dedupe_snapshots(snaps)), 1)

    def test_same_site_price_5pct_apart_kept(self):
        snaps = [self._snap("kara", 100000.0), self._snap("kara", 105000.0)]
        self.assertEqual(len(_dedupe_snapshots(snaps)), 2)

    def test_different_sites_never_deduped(self):
        snaps = [self._snap("kara", 100000.0), self._snap("slot", 100000.0)]
        self.assertEqual(len(_dedupe_snapshots(snaps)), 2)

    def test_different_titles_kept(self):
        snaps = [
            self._snap("kara", 100000.0, "iPhone 13 128GB"),
            self._snap("kara", 100500.0, "Samsung Galaxy S23"),
        ]
        self.assertEqual(len(_dedupe_snapshots(snaps)), 2)

    def test_generic_titles_deduped_by_price(self):
        snaps = [
            self._snap("kara", 100000.0, "Product from kara"),
            self._snap("kara", 100100.0, "Product from kara"),
        ]
        self.assertEqual(len(_dedupe_snapshots(snaps)), 1)


class ComparePricesTests(unittest.TestCase):
    def setUp(self):
        clear_fx_cache()
        self.addCleanup(clear_fx_cache)
        self.engine = _make_engine()

    def _run_compare(self, scan_impl):
        async def _go():
            with mock.patch.object(
                self.engine, "_scan_site", side_effect=scan_impl
            ), mock.patch.object(
                self.engine, "_human_delay", new=mock.AsyncMock()
            ):
                return await self.engine.compare_prices(
                    "iphone 13", sites=["kara", "slot", "ebay"]
                )
        return asyncio.run(_go())

    def test_merges_and_picks_best_deal(self):
        async def fake_scan(site_id, query, max_price, max_pages):
            return {
                "kara": [PriceSnapshot("https://kara.com.ng/x", "kara", 150000.0,
                                      "iPhone 13 128GB")],
                "slot": [
                    PriceSnapshot("https://www.slot.ng/x", "slot", 120000.0,
                                  "iPhone 13 128GB"),
                    # 0.5% apart -> near-duplicate, must be folded
                    PriceSnapshot("https://www.slot.ng/x", "slot", 120600.0,
                                  "iPhone 13 128GB"),
                ],
                "ebay": [PriceSnapshot("https://www.ebay.com/x", "ebay", 180000.0,
                                       "iPhone 13 128GB")],
            }.get(site_id, [])

        report = self._run_compare(fake_scan)

        self.assertEqual(report["query"], "iphone 13")
        self.assertEqual(set(report["sites_scanned"]), {"kara", "slot", "ebay"})
        self.assertEqual(report["sites_failed"], [])

        per_site = report["per_site"]
        self.assertAlmostEqual(per_site["kara"]["price_ngn"], 150000.0)
        # slot deduped to a single cheapest entry
        self.assertAlmostEqual(per_site["slot"]["price_ngn"], 120000.0)
        self.assertAlmostEqual(per_site["ebay"]["price_ngn"], 180000.0)

        # best deal = cheapest overall
        self.assertEqual(report["best_site"], "slot")
        self.assertAlmostEqual(report["best_deal"]["price_ngn"], 120000.0)

        # median of [120000, 150000, 180000] = 150000; savings = 20%
        self.assertAlmostEqual(report["median_price"], 150000.0)
        self.assertAlmostEqual(report["savings_vs_median_pct"], 20.0)
        self.assertEqual(report["result_count"], 3)

    def test_fail_soft_when_one_site_dies(self):
        async def fake_scan(site_id, query, max_price, max_pages):
            if site_id == "ebay":
                raise RuntimeError("bot wall")
            return [PriceSnapshot(f"https://{site_id}/x", site_id, 100000.0,
                                  "Widget")]

        report = self._run_compare(fake_scan)

        self.assertEqual(report["sites_failed"], ["ebay"])
        self.assertIn("kara", report["sites_scanned"])
        self.assertIn("slot", report["sites_scanned"])
        self.assertIsNotNone(report["best_deal"])
        self.assertEqual(report["best_deal"]["site"], "kara")

    def test_no_results_gives_empty_report(self):
        async def fake_scan(site_id, query, max_price, max_pages):
            return []

        report = self._run_compare(fake_scan)
        self.assertIsNone(report["best_deal"])
        self.assertIsNone(report["best_site"])
        self.assertEqual(report["median_price"], 0.0)
        self.assertEqual(report["result_count"], 0)


class DealsToolCompareTests(unittest.TestCase):
    def test_compare_action_wired(self):
        from nomorals.tools import deals as deals_tool

        engine = _make_engine()
        deals_tool.init_deals_tool(engine)
        self.addCleanup(lambda: setattr(deals_tool, "_engine", None))

        async def fake_scan(site_id, query, max_price, max_pages):
            return [PriceSnapshot(f"https://{site_id}/x", site_id, 90000.0,
                                  "Widget")]

        async def _go():
            with mock.patch.object(
                engine, "_scan_site", side_effect=fake_scan
            ), mock.patch.object(engine, "_human_delay", new=mock.AsyncMock()):
                return await deals_tool.deals(
                    action="compare", query="widget", sites=["kara", "slot"]
                )

        result = asyncio.run(_go())
        self.assertEqual(result["action"], "compare")
        self.assertIsNotNone(result["best_deal"])
        self.assertEqual(result["best_deal"]["site"], "kara")
        self.assertIn("savings_vs_median_pct", result)

    def test_unknown_action_still_errors(self):
        from nomorals.tools import deals as deals_tool

        deals_tool.init_deals_tool(_make_engine())
        self.addCleanup(lambda: setattr(deals_tool, "_engine", None))
        result = asyncio.run(deals_tool.deals(action="bogus"))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
