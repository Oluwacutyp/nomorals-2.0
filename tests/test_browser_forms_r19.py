"""Round 19: browser form controls (select/check/type-aware fill), SPA waits,
one-call captcha detection — raw tabs and rendered tabs.

Rendered-tab tests use a fake playwright page; no real browser is launched.
"""

import contextlib
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from nomorals.browser.service import BrowserError, BrowserService
from nomorals.tools.browser import BrowserSession, parse_html


# ── raw tab: select/check on the parsed DOM ──────────────────────────────────

FORM_HTML = """
<html><body><form id="f" action="/go" method="post">
<input type="text" name="q" value="">
<select name="country">
  <option value="us">United States</option>
  <option value="ng">Nigeria</option>
  <option>Other</option>
</select>
<input type="checkbox" name="agree" value="yes">
<input type="checkbox" name="news" value="weekly">
<input type="radio" name="plan" value="free">
<input type="radio" name="plan" value="pro">
<input type="file" name="avatar">
</form></body></html>
"""


def _raw_session(html=FORM_HTML):
    sess = BrowserSession.__new__(BrowserSession)
    sess.dom = parse_html(html)
    sess.url = "https://example.com/form"
    sess._raw = html
    sess._form_values = {}
    return sess


class RawSelectCheckTests(unittest.TestCase):
    def test_select_by_value(self):
        s = _raw_session()
        out = s.select("country", "ng")
        self.assertTrue(out["ok"])
        self.assertEqual("ng", out["picked"])
        self.assertEqual("ng", s._form_values["country"])

    def test_select_by_label(self):
        s = _raw_session()
        out = s.select("country", "Nigeria")
        self.assertEqual("ng", out["picked"])

    def test_select_option_without_value_submits_label(self):
        s = _raw_session()
        out = s.select("country", "Other")
        self.assertEqual("Other", out["picked"])

    def test_select_unknown_option_fails_fast(self):
        s = _raw_session()
        with self.assertRaises(Exception) as ctx:
            s.select("country", "xx")
        self.assertIn("no option", str(ctx.exception))
        self.assertNotIn("country", s._form_values)

    def test_select_on_text_input_fails_fast(self):
        s = _raw_session()
        with self.assertRaises(Exception) as ctx:
            s.select("q", "x")
        self.assertIn("not a <select>", str(ctx.exception))

    def test_select_unknown_field_fails_fast(self):
        s = _raw_session()
        with self.assertRaises(Exception):
            s.select("nope", "x")

    def test_check_checkbox_stores_its_value(self):
        s = _raw_session()
        out = s.check("agree", True)
        self.assertTrue(out["checked"])
        self.assertEqual("yes", s._form_values["agree"])

    def test_uncheck_checkbox_removes_field(self):
        s = _raw_session()
        s.check("agree", True)
        out = s.check("agree", False)
        self.assertFalse(out["checked"])
        self.assertNotIn("agree", s._form_values)

    def test_check_radio(self):
        s = _raw_session()
        out = s.check("plan", True)
        self.assertTrue(out["checked"])
        self.assertEqual("free", s._form_values["plan"])

    def test_uncheck_radio_fails_fast(self):
        s = _raw_session()
        with self.assertRaises(Exception) as ctx:
            s.check("plan", False)
        self.assertIn("cannot be unchecked", str(ctx.exception))

    def test_check_on_text_input_fails_fast(self):
        s = _raw_session()
        with self.assertRaises(Exception) as ctx:
            s.check("q", True)
        self.assertIn("not a checkbox/radio", str(ctx.exception))


# ── rendered tab: fake playwright page ───────────────────────────────────────

class _FakeLocator:
    def __init__(self, page, text):
        self._page = page
        self._text = text

    @property
    def first(self):
        return self

    def wait_for(self, timeout=10000):
        self._page.text_waits.append((self._text, timeout))
        if self._text == "never appears":
            raise TimeoutError("text timeout")


class _FakeRPage:
    """Fake playwright page with a configurable field map."""

    def __init__(self):
        self.url = "https://example.com/form"
        # name -> (tag, type)
        self.fields = {
            "q": ("input", "text"),
            "country": ("select", ""),
            "agree": ("input", "checkbox"),
            "plan": ("input", "radio"),
            "avatar": ("input", "file"),
        }
        self.fill_calls = []
        self.select_calls = []
        self.check_calls = []
        self.uncheck_calls = []
        self.url_waits = []
        self.text_waits = []
        self.page_html = (
            "<html><body><form>"
            '<div class="g-recaptcha" data-sitekey="6Le-AAAAv2sitekey123"></div>'
            "</form></body></html>"
        )
        self.options = {"country": [("us", "United States"),
                                    ("ng", "Nigeria")]}

    def title(self):
        return "Fake"

    def content(self):
        return self.page_html

    def evaluate(self, js, selector):
        m = re.search(r'name="([^"]+)"', selector)
        name = m.group(1) if m else ""
        hit = self.fields.get(name)
        if hit is None:
            return None
        return {"tag": hit[0], "type": hit[1]}

    def fill(self, selector, value):
        self.fill_calls.append((selector, value))

    def select_option(self, selector, **kw):
        self.select_calls.append((selector, kw))
        m = re.search(r'name="([^"]+)"', selector)
        name = m.group(1) if m else ""
        want = kw.get("value") or kw.get("label")
        for val, label in self.options.get(name, []):
            if want == val or want == label:
                return [val]
        return []

    def check(self, selector):
        self.check_calls.append(selector)

    def uncheck(self, selector):
        self.uncheck_calls.append(selector)

    def wait_for_url(self, pattern, timeout=10000):
        self.url_waits.append((pattern, timeout))
        if "never" in str(pattern):
            raise TimeoutError("url timeout")

    def get_by_text(self, text):
        return _FakeLocator(self, text)


class _FakeRContext:
    def __init__(self, page):
        self._page = page

    def new_page(self):
        return self._page

    def storage_state(self, path=None):
        pass

    def cookies(self):
        return []

    def close(self):
        pass


class _FakeRBrowser:
    def __init__(self, context):
        self._context = context

    def new_context(self, **kwargs):
        return self._context

    def close(self):
        pass


class _FakeRChromium:
    def __init__(self, browser):
        self._browser = browser

    def launch(self, **kwargs):
        return self._browser


class _FakeRSyncPlaywright:
    def __init__(self, chromium):
        self.chromium = chromium

    def start(self):
        return self


@contextlib.contextmanager
def _mocked(page):
    context = _FakeRContext(page)
    browser = _FakeRBrowser(context)
    chromium = _FakeRChromium(browser)

    def factory():
        return _FakeRSyncPlaywright(chromium)

    sync_api = mock.MagicMock()
    sync_api.sync_playwright = factory
    pw_pkg = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"playwright": pw_pkg,
                                       "playwright.sync_api": sync_api}), \
            mock.patch("importlib.util.find_spec", return_value=object()):
        yield


class RenderedFormTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))

    def _tab(self, page):
        with _mocked(page):
            tab = self.svc.open_rendered_tab("s")
        # fake navigation: mark a page loaded without real playwright
        tab._page = page
        tab.url = page.url
        return tab

    def test_fill_text_input_uses_page_fill(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            out = tab.fill("q", "hello")
        self.assertTrue(out["ok"])
        self.assertEqual(1, len(page.fill_calls))
        self.assertIn('name="q"', page.fill_calls[0][0])

    def test_fill_routes_select_to_select_option(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            out = tab.fill("country", "ng")
        self.assertTrue(out["ok"])
        self.assertEqual(0, len(page.fill_calls))
        self.assertEqual(1, len(page.select_calls))

    def test_fill_routes_checkbox_to_check(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            tab.fill("agree", "yes")
        self.assertEqual(1, len(page.check_calls))
        self.assertEqual(0, len(page.uncheck_calls))

    def test_fill_falsey_routes_checkbox_to_uncheck(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            tab.fill("agree", "off")
        self.assertEqual(1, len(page.uncheck_calls))

    def test_fill_on_file_input_fails_fast_with_hint(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page), self.assertRaises(BrowserError) as ctx:
            tab.fill("avatar", "/tmp/x.png")
        self.assertIn("upload", str(ctx.exception))

    def test_fill_unknown_field_fails_fast(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page), self.assertRaises(BrowserError) as ctx:
            tab.fill("nope", "x")
        self.assertIn("no form field", str(ctx.exception))

    def test_select_picks_value(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            out = tab.select("country", "ng")
        self.assertTrue(out["ok"])
        self.assertEqual(["ng"], out["picked"])

    def test_select_by_label(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            out = tab.select("country", "Nigeria", by="label")
        self.assertEqual(["ng"], out["picked"])

    def test_select_missing_option_fails_fast(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page), self.assertRaises(BrowserError) as ctx:
            tab.select("country", "xx")
        self.assertIn("no option", str(ctx.exception))

    def test_select_on_text_input_fails_fast(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page), self.assertRaises(BrowserError) as ctx:
            tab.select("q", "x")
        self.assertIn("not a <select>", str(ctx.exception))

    def test_check_and_uncheck(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            self.assertTrue(tab.check("agree", True)["checked"])
            self.assertFalse(tab.check("agree", False)["checked"])
        self.assertEqual(1, len(page.check_calls))
        self.assertEqual(1, len(page.uncheck_calls))

    def test_uncheck_radio_fails_fast(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page), self.assertRaises(BrowserError) as ctx:
            tab.check("plan", False)
        self.assertIn("cannot be", str(ctx.exception))

    def test_wait_for_url_success(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            out = tab.wait_for_url("dashboard")
        self.assertTrue(out["ok"])
        self.assertIn("dashboard", page.url_waits[0][0])

    def test_wait_for_url_timeout_fails_fast(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page), self.assertRaises(BrowserError) as ctx:
            tab.wait_for_url("never")
        self.assertIn("timed out", str(ctx.exception))

    def test_wait_for_text_success(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            out = tab.wait_for_text("Welcome back")
        self.assertTrue(out["ok"])
        self.assertEqual("Welcome back", page.text_waits[0][0])

    def test_wait_for_text_timeout_fails_fast(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page), self.assertRaises(BrowserError) as ctx:
            tab.wait_for_text("never appears")
        self.assertIn("timed out", str(ctx.exception))

    def test_check_captcha_finds_recaptcha(self):
        page = _FakeRPage()
        tab = self._tab(page)
        with _mocked(page):
            out = tab.check_captcha()
        self.assertEqual(1, out["count"])
        self.assertEqual("recaptcha_v2", out["challenges"][0]["kind"])
        self.assertEqual("6Le-AAAAv2sitekey123",
                         out["challenges"][0]["sitekey"])

    def test_check_captcha_clean_page(self):
        page = _FakeRPage()
        page.page_html = "<html><body>plain</body></html>"
        tab = self._tab(page)
        with _mocked(page):
            out = tab.check_captcha()
        self.assertEqual(0, out["count"])


if __name__ == "__main__":
    unittest.main()
