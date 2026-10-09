"""Browser automation improvements: smart field resolution, fill_form,
set_date, navigate retries, load-state waits, evaluate/localStorage,
stealth profiles, and screenshot-on-failure.

Rendered-tab tests use fake playwright pages; no real browser is launched.
"""

import contextlib
import http.server
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nomorals.browser import daemon as daemon_mod
from nomorals.browser.daemon import DaemonError
from nomorals.browser.forms import (
    FIELD_MARKER,
    normalize_date,
)
from nomorals.browser.service import (
    BrowserError,
    BrowserService,
    RenderedTab,
)

MARKER = f'[{FIELD_MARKER}="1"]'


class _FakePage:
    """Fake playwright page that speaks the resolver protocol."""

    def __init__(self):
        self.url = "https://example.com/signup"
        self.goto_calls = []
        self.goto_failures = 0
        self.fill_calls = []
        self.select_calls = []
        self.check_calls = []
        self.uncheck_calls = []
        self.press_calls = []
        self.click_calls = []
        self.evaluate_calls = []
        self.wait_state_calls = []
        self.screenshot_calls = []
        self.page_html = "<html><body>signup form</body></html>"
        #: field name -> resolver info dict
        self.fields_by_name = {}
        self.describe_result = []
        self.readback_value = None
        self.set_value_result = "ok"
        self.local = {}
        self.fail_load_state = False

    # -- navigation ------------------------------------------------------
    def goto(self, url, **kwargs):
        self.goto_calls.append((url, kwargs))
        if self.goto_failures > 0:
            self.goto_failures -= 1
            raise RuntimeError("net::ERR_CONNECTION_RESET")
        self.url = url

    def title(self):
        return "Signup"

    # -- evaluation ------------------------------------------------------
    def evaluate(self, js, arg=None):
        self.evaluate_calls.append((js, arg))
        js = js or ""
        # the field resolver also clears stale markers (removeAttribute) —
        # check the resolver first, the clear script has no offsetParent.
        if "offsetParent" in js:
            # the field resolver: arg is the bare field name
            return self.fields_by_name.get(arg)
        if "removeAttribute" in js:
            return True
        if "localStorage" in js:
            return self._local_storage(js, arg)
        if "getOwnPropertyDescriptor" in js:
            return self.set_value_result
        if '[role="button"]' in js:
            return [dict(f) for f in self.describe_result]
        if "el.value" in js:
            return self.readback_value
        return "eval-result"

    def _local_storage(self, js, arg):
        if "getItem" in js:
            return self.local.get(arg)
        if "setItem" in js:
            k, v = arg
            self.local[k] = str(v)
            return True
        if "removeItem" in js:
            self.local.pop(arg, None)
            return True
        if "clear" in js:
            self.local.clear()
            return True
        return None

    # -- interaction -----------------------------------------------------
    def fill(self, selector, value):
        self.fill_calls.append((selector, value))

    def press(self, selector, key):
        self.press_calls.append((selector, key))

    def click(self, selector):
        self.click_calls.append(selector)

    def check(self, selector):
        self.check_calls.append(selector)

    def uncheck(self, selector):
        self.uncheck_calls.append(selector)

    def select_option(self, selector, **kw):
        self.select_calls.append((selector, kw))
        return [kw.get("value") or kw.get("label")]

    def wait_for_load_state(self, state, timeout=30000):
        self.wait_state_calls.append((state, timeout))
        if self.fail_load_state:
            raise TimeoutError("load state timeout")

    def wait_for_selector(self, selector, state="visible", timeout=10000):
        pass

    def screenshot(self, path=None, full_page=False):
        self.screenshot_calls.append(path)
        Path(path).write_bytes(b"\x89PNG-fake")

    def content(self):
        return self.page_html

    def close(self):
        pass


class _FakeContext:
    def __init__(self, page):
        self._page = page
        self.kwargs = {}
        self.init_scripts = []
        self.saved_path = None

    def new_page(self):
        return self._page

    def add_init_script(self, script):
        self.init_scripts.append(script)

    def storage_state(self, path=None):
        self.saved_path = path

    def close(self):
        pass


class _FakeBrowser:
    def __init__(self, context):
        self._context = context
        self.context_kwargs = None

    def new_context(self, **kwargs):
        self.context_kwargs = kwargs
        return self._context

    def close(self):
        pass


class _FakeChromium:
    def __init__(self, browser):
        self._browser = browser
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return self._browser


class _FakePlaywright:
    def __init__(self, chromium):
        self.chromium = chromium

    def start(self):
        return self


@contextlib.contextmanager
def _mocked_playwright(page):
    """Yield (tab_factory_kwargs, context, browser, chromium) with sys.modules
    patched so RenderedTab uses the fakes."""
    context = _FakeContext(page)
    browser = _FakeBrowser(context)
    chromium = _FakeChromium(browser)
    fake_pw = _FakePlaywright(chromium)

    sync_api = mock.MagicMock()
    sync_api.sync_playwright = lambda: fake_pw
    pw_pkg = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"playwright": pw_pkg,
                                       "playwright.sync_api": sync_api}), \
            mock.patch("importlib.util.find_spec", return_value=object()):
        yield fake_pw, context, browser, chromium


def _direct_tab(tmpdir, page, **kwargs):
    """A RenderedTab wired straight to a fake page (no service)."""
    context = _FakeContext(page)
    browser = _FakeBrowser(context)
    chromium = _FakeChromium(browser)
    tab = RenderedTab(
        tab_id="tab1",
        session_name="sess",
        storage_state_path=os.path.join(tmpdir, "cookies", "sess",
                                        "playwright-storage.json"),
        playwright=_FakePlaywright(chromium),
        **kwargs,
    )
    tab._browser = browser
    tab._context = context
    tab._page = page
    tab.url = page.url
    return tab, context, browser, chromium


def _field(tag="input", ftype="text", by="name", name=""):
    return {"tag": tag, "type": ftype, "by": by, "score": 70,
            "name": name, "id": ""}


class FieldResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_fill_matches_by_visible_label(self):
        page = _FakePage()
        page.fields_by_name = {
            "Email address": _field(by="label", name="email"),
        }
        tab, *_ = _direct_tab(self.tmp, page)
        out = tab.fill("Email address", "a@example.com")
        self.assertTrue(out["ok"])
        self.assertEqual("label", out["matched_via"])
        self.assertEqual([(MARKER, "a@example.com")], page.fill_calls)

    def test_fill_matches_by_placeholder(self):
        page = _FakePage()
        page.fields_by_name = {
            "Search products": _field(by="placeholder", name="q"),
        }
        tab, *_ = _direct_tab(self.tmp, page)
        out = tab.fill("Search products", "phone")
        self.assertEqual("placeholder", out["matched_via"])

    def test_fill_not_found_lists_page_fields(self):
        page = _FakePage()
        page.fields_by_name = {}
        page.describe_result = [{
            "tag": "input", "type": "email", "name": "user_email",
            "id": "", "label": "Email address", "placeholder": "",
            "aria": "",
        }]
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError) as ctx:
            tab.fill("nope", "x")
        msg = str(ctx.exception)
        self.assertIn("no form field 'nope'", msg)
        self.assertIn("fields on this page", msg)
        self.assertIn("Email address", msg)

    def test_describe_fields(self):
        page = _FakePage()
        page.describe_result = [{
            "tag": "select", "type": "", "name": "country",
            "id": "", "label": "Country", "placeholder": "", "aria": "",
        }]
        tab, *_ = _direct_tab(self.tmp, page)
        out = tab.describe_fields()
        self.assertEqual(1, out["count"])
        self.assertEqual("Country", out["fields"][0]["label"])

    def test_fill_form_fills_many(self):
        page = _FakePage()
        page.fields_by_name = {
            "First name": _field(by="label", name="fname"),
            "country": _field(tag="select", by="name", name="country"),
            "agree": _field(ftype="checkbox", by="name", name="agree"),
        }
        tab, *_ = _direct_tab(self.tmp, page)
        out = tab.fill_form({"First name": "Ada", "country": "ng",
                             "agree": "yes"})
        self.assertTrue(out["ok"])
        self.assertEqual(["First name", "country", "agree"], out["filled"])
        self.assertEqual({}, out["failed"])
        self.assertEqual([(MARKER, "Ada")], page.fill_calls)
        self.assertEqual(1, len(page.select_calls))
        self.assertEqual(1, len(page.check_calls))

    def test_fill_form_collects_failures_when_not_stopping(self):
        page = _FakePage()
        page.fields_by_name = {"ok-field": _field(name="ok-field")}
        tab, *_ = _direct_tab(self.tmp, page)
        out = tab.fill_form({"ok-field": "v", "missing": "v"},
                            stop_on_error=False)
        self.assertFalse(out["ok"])
        self.assertEqual(["ok-field"], out["filled"])
        self.assertIn("missing", out["failed"])

    def test_fill_form_fail_fast_by_default(self):
        page = _FakePage()
        page.fields_by_name = {}
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError):
            tab.fill_form({"a": "1", "b": "2"})

    def test_fill_form_rejects_non_mapping(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError):
            tab.fill_form([])


class SetDateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_native_date_input(self):
        page = _FakePage()
        page.fields_by_name = {"dob": _field(ftype="date", name="dob")}
        tab, *_ = _direct_tab(self.tmp, page)
        out = tab.set_date("dob", "04/10/2026")
        self.assertTrue(out["ok"])
        self.assertEqual("2026-10-04", out["date"])
        self.assertEqual([(MARKER, "2026-10-04")], page.fill_calls)

    def test_text_picker_types_and_confirms(self):
        page = _FakePage()
        page.fields_by_name = {"dob": _field(ftype="text", name="dob")}
        page.readback_value = "2026-10-04"  # typed value stuck
        tab, *_ = _direct_tab(self.tmp, page)
        out = tab.set_date("dob", "4 Oct 2026")
        self.assertEqual("2026-10-04", out["date"])
        self.assertEqual([(MARKER, "2026-10-04")], page.fill_calls)
        self.assertEqual([(MARKER, "Enter")], page.press_calls)

    def test_text_picker_js_fallback_when_typing_does_not_stick(self):
        page = _FakePage()
        page.fields_by_name = {"dob": _field(ftype="text", name="dob")}
        page.readback_value = ""  # React swallowed the typed value
        tab, *_ = _direct_tab(self.tmp, page)
        tab.set_date("dob", "2026-10-04")
        set_calls = [c for c in page.evaluate_calls
                     if "getOwnPropertyDescriptor" in (c[0] or "")]
        self.assertEqual(1, len(set_calls))

    def test_bad_date_fails_fast(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError) as ctx:
            tab.set_date("dob", "not a date")
        self.assertIn("cannot parse date", str(ctx.exception))
        self.assertEqual([], page.fill_calls)

    def test_normalize_date_formats(self):
        self.assertEqual("2026-10-04", normalize_date("2026-10-04"))
        self.assertEqual("2026-10-04", normalize_date("04/10/2026"))
        # ambiguous numeric dates resolve DD/MM first (documented), so
        # month-first input must be unambiguous to parse as MM/DD.
        self.assertEqual("2026-12-25", normalize_date("12/25/2026"))
        self.assertEqual("2026-10-04", normalize_date("4 Oct 2026"))
        self.assertEqual("2026-10-04", normalize_date("October 4, 2026"))
        with self.assertRaises(ValueError):
            normalize_date("yesterday")


class NavigateRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_navigate_retries_transient_failure(self):
        page = _FakePage()
        page.goto_failures = 1
        tab, *_ = _direct_tab(self.tmp, page, retries=2)
        out = tab.navigate("https://example.com/")
        self.assertEqual("https://example.com/", out["url"])
        self.assertEqual(2, len(page.goto_calls))
        self.assertEqual("", tab.error)

    def test_navigate_gives_up_and_names_evidence(self):
        page = _FakePage()
        page.goto_failures = 9
        tab, *_ = _direct_tab(self.tmp, page, retries=1)
        with self.assertRaises(BrowserError) as ctx:
            tab.navigate("https://example.com/")
        msg = str(ctx.exception)
        self.assertIn("(2 attempts)", msg)
        self.assertIn("evidence:", msg)
        self.assertIn("screenshot=", msg)
        self.assertIn("dom=", msg)
        self.assertEqual(1, len(page.screenshot_calls))
        self.assertTrue(tab.error)

    def test_navigate_no_retry_when_disabled(self):
        page = _FakePage()
        page.goto_failures = 9
        tab, *_ = _direct_tab(self.tmp, page, retries=0)
        with self.assertRaises(BrowserError):
            tab.navigate("https://example.com/")
        self.assertEqual(1, len(page.goto_calls))

    def test_navigate_no_snapshot_when_shot_on_error_off(self):
        page = _FakePage()
        page.goto_failures = 9
        tab, *_ = _direct_tab(self.tmp, page, retries=0,
                              shot_on_error=False)
        with self.assertRaises(BrowserError) as ctx:
            tab.navigate("https://example.com/")
        self.assertNotIn("evidence:", str(ctx.exception))
        self.assertEqual([], page.screenshot_calls)

    def test_navigate_rejects_bad_wait_until(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError) as ctx:
            tab.navigate("https://example.com/", wait_until="whenever")
        self.assertIn("wait_until", str(ctx.exception))

    def test_navigate_networkidle_passthrough(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        tab.navigate("https://example.com/", wait_until="networkidle")
        _, kwargs = page.goto_calls[0]
        self.assertEqual("networkidle", kwargs["wait_until"])


class WaitStrategyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_wait_for_network_idle(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        out = tab.wait_for_network_idle(timeout=5000)
        self.assertTrue(out["ok"])
        self.assertEqual([("networkidle", 5000)], page.wait_state_calls)

    def test_wait_for_load_state_variants(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        for state in ("load", "domcontentloaded", "networkidle"):
            out = tab.wait_for_load_state(state)
            self.assertEqual(state, out["state"])
        with self.assertRaises(BrowserError):
            tab.wait_for_load_state("whenever")

    def test_wait_failure_names_evidence(self):
        page = _FakePage()
        page.fail_load_state = True
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError) as ctx:
            tab.wait_for_load_state("networkidle")
        self.assertIn("evidence:", str(ctx.exception))


class EvaluateAndStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_evaluate_passthrough(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        self.assertEqual("eval-result", tab.evaluate("1 + 1"))
        self.assertEqual("eval-result",
                         tab.evaluate("(a) => a * 2", 21))

    def test_evaluate_requires_js(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError):
            tab.evaluate("   ")

    def test_evaluate_fails_fast_without_js_driver(self):
        page = _FakePage()
        page.evaluate = None  # duck-typed driver
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError) as ctx:
            tab.evaluate("1+1")
        self.assertIn("cannot evaluate JavaScript", str(ctx.exception))

    def test_local_storage_round_trip(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        self.assertIsNone(tab.local_storage("get", "token")["value"])
        tab.local_storage("set", "token", "abc123")
        self.assertEqual("abc123",
                         tab.local_storage("get", "token")["value"])
        tab.local_storage("remove", "token")
        self.assertIsNone(tab.local_storage("get", "token")["value"])
        tab.local_storage("set", "a", "1")
        tab.local_storage("clear")
        self.assertEqual({}, page.local)

    def test_local_storage_bad_action(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError):
            tab.local_storage("explode", "k")

    def test_local_storage_needs_key(self):
        page = _FakePage()
        tab, *_ = _direct_tab(self.tmp, page)
        with self.assertRaises(BrowserError):
            tab.local_storage("get", "")

    def test_persist_writes_storage_file(self):
        page = _FakePage()
        tab, context, *_ = _direct_tab(self.tmp, page)
        out = tab.persist()
        expected = os.path.join(self.tmp, "cookies", "sess",
                                "playwright-storage.json")
        self.assertEqual(expected, out["path"])
        self.assertEqual(expected, context.saved_path)


class StealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_stealth_profile_applied_by_default(self):
        page = _FakePage()
        with _mocked_playwright(page) as (pw, context, browser, chromium):
            svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))
            tab = svc.open_rendered_tab("s", "https://example.com/")
            self.assertIn("--disable-blink-features=AutomationControlled",
                          chromium.launch_kwargs["args"])
            ctx = browser.context_kwargs
            self.assertIn("Chrome/", ctx["user_agent"])
            self.assertNotIn("Headless", ctx["user_agent"])
            self.assertEqual({"width": 1366, "height": 768}, ctx["viewport"])
            self.assertEqual("en-US", ctx["locale"])
            self.assertEqual("Africa/Lagos", ctx["timezone_id"])
            # navigator.webdriver hidden via init script
            self.assertEqual(1, len(context.init_scripts))
            self.assertIn("webdriver", context.init_scripts[0])
            svc.close_rendered_tab(tab.tab_id)

    def test_stealth_disabled_launch_is_plain(self):
        page = _FakePage()
        with _mocked_playwright(page) as (pw, context, browser, chromium):
            svc = BrowserService(
                data_dir=os.path.join(self.tmp, "data"),
                stealth={"enabled": False})
            tab = svc.open_rendered_tab("s", "https://example.com/")
            self.assertEqual({"headless": True}, chromium.launch_kwargs)
            self.assertNotIn("user_agent", browser.context_kwargs)
            self.assertEqual([], context.init_scripts)
            svc.close_rendered_tab(tab.tab_id)

    def test_stealth_per_tab_override(self):
        page = _FakePage()
        with _mocked_playwright(page) as (pw, context, browser, chromium):
            svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))
            tab = svc.open_rendered_tab(
                "s", "https://example.com/",
                stealth={"viewport": {"width": 1920, "height": 1080}})
            self.assertEqual({"width": 1920, "height": 1080},
                             browser.context_kwargs["viewport"])
            svc.close_rendered_tab(tab.tab_id)

    def test_stealth_unknown_key_fails_fast(self):
        with self.assertRaises(BrowserError) as ctx:
            BrowserService(data_dir=os.path.join(self.tmp, "data"),
                           stealth={"fingerprint": "spoof"})
        self.assertIn("unknown stealth profile keys", str(ctx.exception))

    def test_storage_state_survives_restart(self):
        """Sessions survive restarts: cookies AND localStorage ride in the
        storage_state file, which the next service loads into its context."""
        data = os.path.join(self.tmp, "data")
        state_path = os.path.join(data, "cookies", "sess",
                                  "playwright-storage.json")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        Path(state_path).write_text(
            '{"cookies": [{"name": "sess", "value": "abc"}], '
            '"origins": [{"origin": "https://example.com", '
            '"localStorage": [{"name": "token", "value": "xyz"}]}]}')
        page = _FakePage()
        with _mocked_playwright(page) as (pw, context, browser, chromium):
            svc = BrowserService(data_dir=data)
            tab = svc.open_rendered_tab("sess")
            tab.navigate("https://example.com/")
            self.assertEqual(state_path,
                             browser.context_kwargs.get("storage_state"))
            svc.close_rendered_tab(tab.tab_id)


class PlainTabRetryTests(unittest.TestCase):
    """Plain-HTTP tabs retry failed loads with backoff."""

    @classmethod
    def setUpClass(cls):
        class Handler(http.server.BaseHTTPRequestHandler):
            failures_left = 0

            def log_message(self, *args):
                pass

            def do_GET(self):
                if type(self).failures_left > 0:
                    type(self).failures_left -= 1
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"boom")
                    return
                body = b"<html><body>ok</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        cls.Handler = Handler
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                    Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))

    def test_plain_tab_navigate_retries(self):
        type(self).Handler.failures_left = 2
        handle = self.svc.open_session("r")
        tab = handle.open_tab()
        out = tab.navigate(f"http://127.0.0.1:{self.port}/", retries=2)
        self.assertTrue(out.get("ok", True))
        self.assertIn("ok", tab.text()["text"])

    def test_plain_tab_navigate_gives_up(self):
        type(self).Handler.failures_left = 99
        handle = self.svc.open_session("r2")
        tab = handle.open_tab()
        with self.assertRaises(BrowserError):
            tab.navigate(f"http://127.0.0.1:{self.port}/", retries=1)


class _StubRenderedTab:
    """Minimal stand-in for daemon op tests."""

    def __init__(self):
        self.calls = []

    def evaluate(self, js, arg=None):
        self.calls.append(("evaluate", js, arg))
        return "stubbed"

    def fill_form(self, fields, stop_on_error=True, verify=True):
        self.calls.append(("fill_form", fields, stop_on_error, verify))
        return {"ok": True}

    def set_date(self, name, value):
        self.calls.append(("set_date", name, value))
        return {"ok": True}

    def wait_for_load_state(self, state, timeout=30000):
        self.calls.append(("wait_for_load_state", state, timeout))
        return {"ok": True}

    def wait_for_network_idle(self, timeout=15000):
        self.calls.append(("wait_for_network_idle", timeout))
        return {"ok": True}

    def describe_fields(self):
        self.calls.append(("describe_fields",))
        return {"count": 0}

    def local_storage(self, action="get", key="", value=None):
        self.calls.append(("local_storage", action, key, value))
        return {"action": action}

    def persist(self):
        self.calls.append(("persist",))
        return {"path": "/tmp/x"}


class DaemonNewOpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))
        self.stub = _StubRenderedTab()
        self.svc._rendered_tabs["t1"] = self.stub

    def test_r_evaluate(self):
        out = daemon_mod._handle_op(
            self.svc, "r_evaluate",
            {"tab_id": "t1", "js": "1+1", "arg": 2})
        self.assertEqual("stubbed", out["result"])
        self.assertEqual(("evaluate", "1+1", 2), self.stub.calls[0])

    def test_r_evaluate_without_arg(self):
        out = daemon_mod._handle_op(
            self.svc, "r_evaluate", {"tab_id": "t1", "js": "1+1"})
        self.assertEqual("stubbed", out["result"])
        self.assertEqual(("evaluate", "1+1", None), self.stub.calls[0])

    def test_r_fill_form(self):
        out = daemon_mod._handle_op(
            self.svc, "r_fill_form",
            {"tab_id": "t1", "fields": {"a": "1"}})
        self.assertTrue(out["result"]["ok"])
        self.assertEqual(("fill_form", {"a": "1"}, True, True),
                         self.stub.calls[0])

    def test_r_fill_form_rejects_non_mapping(self):
        with self.assertRaises(DaemonError):
            daemon_mod._handle_op(
                self.svc, "r_fill_form",
                {"tab_id": "t1", "fields": ["a"]})

    def test_r_set_date(self):
        out = daemon_mod._handle_op(
            self.svc, "r_set_date",
            {"tab_id": "t1", "name": "dob", "value": "2026-10-04"})
        self.assertTrue(out["result"]["ok"])

    def test_r_wait_state(self):
        out = daemon_mod._handle_op(
            self.svc, "r_wait_state",
            {"tab_id": "t1", "state": "networkidle", "timeout": 5000})
        self.assertTrue(out["result"]["ok"])
        self.assertEqual(("wait_for_load_state", "networkidle", 5000),
                         self.stub.calls[0])

    def test_r_wait_idle(self):
        out = daemon_mod._handle_op(
            self.svc, "r_wait_idle", {"tab_id": "t1", "timeout": 7000})
        self.assertTrue(out["result"]["ok"])

    def test_r_fields(self):
        out = daemon_mod._handle_op(self.svc, "r_fields", {"tab_id": "t1"})
        self.assertEqual(0, out["result"]["count"])

    def test_r_storage(self):
        out = daemon_mod._handle_op(
            self.svc, "r_storage",
            {"tab_id": "t1", "action": "set", "key": "k", "value": "v"})
        self.assertEqual("set", out["result"]["action"])

    def test_r_persist(self):
        out = daemon_mod._handle_op(self.svc, "r_persist", {"tab_id": "t1"})
        self.assertEqual("/tmp/x", out["result"]["path"])

    def test_unknown_rendered_tab(self):
        with self.assertRaises(BrowserError):
            daemon_mod._handle_op(self.svc, "r_evaluate",
                                  {"tab_id": "nope", "js": "1"})


if __name__ == "__main__":
    unittest.main()
