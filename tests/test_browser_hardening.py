"""Browser automation hardening: error taxonomy, pacing, verified fills,
wait_for_field, upload verification, download auto-capture wiring,
tab switching, and session/profile persistence.

Rendered-tab tests use fake pages; no real browser or network is used.
"""

import http.cookiejar
import os
import shutil
import socket
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

from nomorals.browser import (
    BrowserBotDetectedError,
    BrowserError,
    BrowserNetworkError,
    BrowserService,
    BrowserSiteError,
    Pacing,
    RenderedTab,
    classify_exception,
    classify_http_status,
    describe,
    detect_challenge,
)
from nomorals.browser.daemon import _handle_op, _handle_pacing_op
from nomorals.browser.pacing import pacing_from_env


# ── fake playwright page ───────────────────────────────────────────────────
# Dispatches on JS content the same way the real page would execute it:
# resolver ("offsetParent"), marker clear ("removeAttribute"), JS setter
# ("dispatchEvent"), forms readback ("fileName:"), select readback
# ("el.value : null"), upload readback ("el.files.length").


class _FakePage:
    def __init__(self):
        self.url = "https://example.com/form"
        self.fill_calls = []
        self.select_calls = []
        self.check_calls = []
        self.uncheck_calls = []
        self.click_calls = []
        self.evaluate_calls = []
        self.set_input_files_calls = []
        #: field name -> resolver info dict (returned for the resolver JS)
        self.fields_by_name = {}
        #: live field state used by readback JS
        self.field_value = ""
        self.field_checked = False
        self.field_files = 0
        self.field_file_name = ""
        #: when True, fills never stick (readback always returns "")
        self.stubborn = False
        self.fail_evaluate = False

    def title(self):
        return "Form"

    def fill(self, selector, value):
        self.fill_calls.append((selector, value))
        if not self.stubborn:
            self.field_value = value

    def select_option(self, selector, **kw):
        self.select_calls.append((selector, kw))
        picked = kw.get("value") or kw.get("label")
        if not self.stubborn:
            self.field_value = picked
        return [picked]

    def check(self, selector):
        self.check_calls.append(selector)
        if not self.stubborn:
            self.field_checked = True

    def uncheck(self, selector):
        self.uncheck_calls.append(selector)
        if not self.stubborn:
            self.field_checked = False

    def set_input_files(self, selector, path):
        self.set_input_files_calls.append((selector, path))
        if not self.stubborn:
            self.field_files = 1
            self.field_file_name = os.path.basename(path)

    def click(self, selector):
        self.click_calls.append(selector)

    def evaluate(self, js, arg=None):
        self.evaluate_calls.append((js, arg))
        if self.fail_evaluate:
            raise RuntimeError("boom")
        js = js or ""
        if "offsetParent" in js:
            return self.fields_by_name.get(arg)
        if "removeAttribute" in js:
            return True
        if "dispatchEvent" in js:
            # the React-compatible JS setter: value sticks through it
            # (unless the page is stubborn and overwrites everything)
            if not self.stubborn:
                self.field_value = str(arg)
            return "ok"
        if "fileName" in js:
            # forms.read_value_js
            return {"value": self.field_value,
                    "checked": self.field_checked,
                    "files": self.field_files or None,
                    "fileName": self.field_file_name or None}
        if "el.files.length" in js:
            return {"n": self.field_files,
                    "name": self.field_file_name}
        if "el.value : null" in js:
            return self.field_value
        return None


def _field(tag="input", ftype="text", by="name", name=""):
    return {"tag": tag, "type": ftype, "by": by, "score": 70,
            "name": name, "id": ""}


def _rtab(page, **kwargs):
    tmp = kwargs.pop("tmp", None) or tempfile.mkdtemp()
    tab = RenderedTab(
        tab_id="tab1",
        session_name="sess",
        storage_state_path=os.path.join(tmp, "cookies", "sess",
                                        "playwright-storage.json"),
        playwright=None,
        **kwargs,
    )
    tab._page = page
    tab.url = page.url
    tab.shot_on_error = False  # no snapshots in unit tests
    return tab


# ── error taxonomy ─────────────────────────────────────────────────────────


class TaxonomyTests(unittest.TestCase):
    def test_net_dns(self):
        err = classify_exception(
            Exception("net::ERR_NAME_NOT_RESOLVED at https://x.example/"),
            url="https://x.example/")
        self.assertIsInstance(err, BrowserNetworkError)
        self.assertEqual("dns", err.reason)
        self.assertTrue(err.next_steps)

    def test_socket_gaierror(self):
        err = classify_exception(socket.gaierror("Name or service not known"))
        self.assertIsInstance(err, BrowserNetworkError)
        self.assertEqual("dns", err.reason)

    def test_urlerror_timeout(self):
        inner = TimeoutError("timed out")
        err = classify_exception(urllib.error.URLError(inner),
                                 url="https://x.example/")
        self.assertIsInstance(err, BrowserNetworkError)
        self.assertEqual("timeout", err.reason)

    def test_connection_refused(self):
        err = classify_exception(
            Exception("net::ERR_CONNECTION_REFUSED at https://x.example/"))
        self.assertIsInstance(err, BrowserNetworkError)
        self.assertEqual("connection_refused", err.reason)

    def test_tls(self):
        err = classify_exception(
            Exception("net::ERR_CERT_AUTHORITY_INVALID"))
        self.assertIsInstance(err, BrowserNetworkError)
        self.assertEqual("tls", err.reason)

    def test_selector_timeout_is_not_network(self):
        # A playwright wait timeout has no transport signal — it must NOT
        # be misreported as a network failure.
        err = classify_exception(Exception("Timeout 10000ms exceeded."))
        self.assertNotIsInstance(err, BrowserNetworkError)
        self.assertIsInstance(err, BrowserError)
        self.assertEqual("browser_error", err.kind)

    def test_unknown_stays_plain(self):
        err = classify_exception(ValueError("weird widget state"))
        self.assertIsInstance(err, BrowserError)
        self.assertNotIsInstance(err, (BrowserNetworkError,
                                       BrowserSiteError,
                                       BrowserBotDetectedError))

    def test_typed_passthrough(self):
        original = BrowserSiteError("gone", status=404)
        self.assertIs(classify_exception(original), original)

    def test_chained_cause_classified(self):
        outer = RuntimeError("goto failed")
        outer.__cause__ = Exception("net::ERR_CONNECTION_RESET")
        err = classify_exception(outer)
        self.assertIsInstance(err, BrowserNetworkError)
        self.assertEqual("connection_reset", err.reason)

    def test_403_cloudflare_challenge(self):
        html = ("<html><head><title>Just a moment...</title></head>"
                "<body><script src='/cdn-cgi/challenge-platform/x.js'>"
                "</script></body></html>")
        err = classify_http_status(403, url="https://x.example/", html=html)
        self.assertIsInstance(err, BrowserBotDetectedError)
        self.assertEqual("cloudflare-challenge", err.detection)
        self.assertTrue(err.next_steps)
        self.assertIn("check_captcha", " ".join(err.next_steps))

    def test_403_bare_is_forbidden_block(self):
        err = classify_http_status(
            403, url="https://x.example/",
            html="<html><body>Forbidden</body></html>")
        self.assertIsInstance(err, BrowserBotDetectedError)
        self.assertEqual("forbidden", err.detection)

    def test_429_is_rate_limit(self):
        err = classify_http_status(429, url="https://x.example/")
        self.assertIsInstance(err, BrowserBotDetectedError)
        self.assertEqual("rate-limit", err.detection)

    def test_404_is_site_error(self):
        err = classify_http_status(404, url="https://x.example/nope")
        self.assertIsInstance(err, BrowserSiteError)
        self.assertEqual(404, err.status)

    def test_500_is_site_error(self):
        err = classify_http_status(500, url="https://x.example/")
        self.assertIsInstance(err, BrowserSiteError)

    def test_503_cloudflare_edge_hint(self):
        err = classify_http_status(
            503, url="https://x.example/",
            headers={"server": "cloudflare"},
            html="<html><body>error</body></html>")
        self.assertIsInstance(err, BrowserSiteError)
        self.assertIn("Cloudflare", str(err))

    def test_success_status_rejected(self):
        with self.assertRaises(ValueError):
            classify_http_status(200, url="https://x.example/")

    def test_with_message_preserves_type(self):
        err = BrowserBotDetectedError("blocked", detection="rate-limit")
        wrapped = err.with_message("rendered click failed: blocked")
        self.assertIsInstance(wrapped, BrowserBotDetectedError)
        self.assertEqual("rate-limit", wrapped.detection)
        self.assertTrue(wrapped.next_steps)

    def test_describe_shape(self):
        err = classify_http_status(429, url="https://x.example/")
        payload = describe(err)
        self.assertEqual("bot_detection", payload["kind"])
        self.assertEqual("rate-limit", payload["detail"]["detection"])
        self.assertTrue(payload["next_steps"])

    def test_all_kinds_catchable_as_browser_error(self):
        for err in (BrowserNetworkError("n", reason="dns"),
                    BrowserSiteError("s", status=500),
                    BrowserBotDetectedError("b", detection="forbidden")):
            with self.assertRaises(BrowserError):
                raise err


class ChallengeDetectionTests(unittest.TestCase):
    def test_cloudflare_title(self):
        self.assertEqual(
            "cloudflare-challenge",
            detect_challenge("<html></html>", title="Just a moment..."))

    def test_cloudflare_structural_marker(self):
        html = "<html><body><script>window._cf_chl_opt={}</script></body></html>"
        self.assertEqual("cloudflare-challenge", detect_challenge(html))

    def test_cloudflare_mitigated_header(self):
        self.assertEqual(
            "cloudflare-challenge",
            detect_challenge("", headers={"cf-mitigated": "challenge"}))

    def test_perimeterx(self):
        self.assertEqual(
            "perimeterx",
            detect_challenge('<html><body><div id="px-captcha"></div></body></html>'))

    def test_datadome(self):
        self.assertEqual(
            "datadome",
            detect_challenge(
                '<html><script src="https://ct.captcha-delivery.com/x.js">'
                "</script></html>"))

    def test_captcha_wall_needs_both_halves(self):
        wall = ('<html><body>Please verify you are human'
                '<div class="g-recaptcha" data-sitekey="x"></div></body></html>')
        self.assertEqual("captcha-wall", detect_challenge(wall))
        # wall text alone is not a wall (could be a help article)
        self.assertEqual(
            "", detect_challenge("<html><body>verify you are human</body></html>"))
        # a login form with a captcha widget is not a wall either
        self.assertEqual(
            "", detect_challenge(
                '<html><body><form><div class="g-recaptcha"></div>'
                "</form></body></html>", title="Login"))

    def test_ordinary_page_clean(self):
        self.assertEqual(
            "", detect_challenge(
                "<html><body><h1>News</h1><p>today's events</p></body></html>",
                title="News"))


# ── pacing ─────────────────────────────────────────────────────────────────


class PacingTests(unittest.TestCase):
    def test_disabled_is_noop(self):
        p = Pacing.disabled()
        start = time.monotonic()
        self.assertEqual(0.0, p.pause("fill"))
        self.assertLess(time.monotonic() - start, 0.05)

    def test_enabled_sleeps_in_range(self):
        p = Pacing(enabled=True, delay_ms=60, jitter_ms=40)
        start = time.monotonic()
        slept = p.pause("click")
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.055)
        self.assertLess(elapsed, 0.5)
        self.assertGreaterEqual(slept, 0.055)

    def test_delay_without_enabled_rejected(self):
        with self.assertRaises(ValueError):
            Pacing(enabled=False, delay_ms=100)

    def test_human_preset(self):
        p = Pacing.human()
        self.assertTrue(p.enabled)
        self.assertGreater(p.delay_ms, 0)

    def test_from_profile_returns_suggestion(self):
        p = Pacing.from_profile()
        self.assertIsInstance(p, Pacing)
        self.assertTrue(p.enabled)

    def test_env_parsing(self):
        with mock.patch.dict(os.environ,
                             {"NOMORALS_BROWSER_PACING": "400,300"}):
            p = pacing_from_env()
            self.assertIsNotNone(p)
            assert p is not None
            self.assertEqual((400, 300), (p.delay_ms, p.jitter_ms))
            self.assertTrue(p.enabled)

    def test_env_unset_is_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOMORALS_BROWSER_PACING", None)
            self.assertIsNone(pacing_from_env())

    def test_env_garbage_rejected(self):
        with mock.patch.dict(os.environ,
                             {"NOMORALS_BROWSER_PACING": "fast"}):
            with self.assertRaises(ValueError):
                pacing_from_env()

    def test_pause_never_raises(self):
        p = Pacing(enabled=True, delay_ms=10)
        with mock.patch("time.sleep", side_effect=OSError("nope")):
            self.assertEqual(0.0, p.pause("x"))

    def test_service_set_pacing_live(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        svc = BrowserService(data_dir=tmp)
        out = svc.set_pacing(enabled=True, delay_ms=50, jitter_ms=10)
        self.assertTrue(out["enabled"])
        self.assertEqual(50, out["delay_ms"])
        # configuring delays while disabled is a loud error, not silent
        with self.assertRaises(BrowserError):
            svc.set_pacing(enabled=False)
        # presets
        out = svc.set_pacing(Pacing.disabled())
        self.assertFalse(out["enabled"])

    def test_daemon_pacing_op(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        svc = BrowserService(data_dir=tmp)
        got = _handle_pacing_op(svc, {"action": "get"})
        self.assertFalse(got["enabled"])
        got = _handle_pacing_op(svc, {"action": "set", "preset": "human"})
        self.assertTrue(got["enabled"])
        got = _handle_op(svc, "pacing", {"action": "get"})
        self.assertTrue(got["result"]["enabled"])
        with self.assertRaises(Exception):
            _handle_pacing_op(svc, {"action": "set", "preset": "nope"})


# ── verified fills ─────────────────────────────────────────────────────────


class VerifiedFillTests(unittest.TestCase):
    def test_fill_verified_by_readback(self):
        page = _FakePage()
        page.fields_by_name = {"Email address": _field(by="label",
                                                      name="email")}
        tab = _rtab(page)
        out = tab.fill("Email address", "a@example.com")
        self.assertTrue(out["ok"])
        self.assertTrue(out["verified"])
        self.assertEqual("label", out["matched_via"])

    def test_fill_falls_back_to_js_setter(self):
        page = _FakePage()
        page.fields_by_name = {"q": _field(name="q")}
        # native fill "fails" the first readback, JS setter fixes it
        calls = {"n": 0}
        orig_fill = page.fill

        def flaky_fill(selector, value):
            orig_fill(selector, value)
            if calls["n"] == 0:
                page.field_value = ""  # framework swallowed it
            calls["n"] += 1

        page.fill = flaky_fill
        tab = _rtab(page)
        out = tab.fill("q", "hello")
        self.assertTrue(out["ok"])
        self.assertTrue(out["verified"])
        # the JS setter ran (dispatchEvent in the evaluate log)
        self.assertTrue(any("dispatchEvent" in (c[0] or "")
                            for c in page.evaluate_calls))

    def test_fill_that_wont_stick_raises(self):
        page = _FakePage()
        page.stubborn = True
        page.fields_by_name = {"q": _field(name="q")}
        tab = _rtab(page)
        with self.assertRaises(BrowserError) as ctx:
            tab.fill("q", "hello")
        msg = str(ctx.exception)
        self.assertIn("did not stick", msg)
        self.assertIn("hello", msg)

    def test_fill_verify_off_skips_readback(self):
        page = _FakePage()
        page.stubborn = True
        page.fields_by_name = {"q": _field(name="q")}
        tab = _rtab(page)
        out = tab.fill("q", "hello", verify=False)
        self.assertTrue(out["ok"])
        self.assertFalse(out["verified"])

    def test_fill_form_verifies_each_field(self):
        page = _FakePage()
        page.fields_by_name = {"Email": _field(by="label", name="email"),
                               "Name": _field(by="label", name="name")}
        tab = _rtab(page)
        out = tab.fill_form({"Email": "a@b.c", "Name": "Ann"})
        self.assertTrue(out["ok"])
        self.assertEqual(["Email", "Name"], out["filled"])
        self.assertEqual(["Email", "Name"], out["verified"])

    def test_select_verified(self):
        page = _FakePage()
        page.fields_by_name = {"country": _field(tag="select",
                                                by="name", name="country")}
        tab = _rtab(page)
        out = tab.select("country", "ng")
        self.assertTrue(out["ok"])
        self.assertEqual(["ng"], out["picked"])
        self.assertTrue(out["verified"])

    def test_select_mismatch_raises(self):
        page = _FakePage()
        page.stubborn = True
        page.fields_by_name = {"country": _field(tag="select",
                                                by="name", name="country")}
        tab = _rtab(page)
        with self.assertRaises(BrowserError) as ctx:
            tab.select("country", "ng")
        self.assertIn("did not stick", str(ctx.exception))

    def test_check_verified(self):
        page = _FakePage()
        page.fields_by_name = {"agree": _field(ftype="checkbox",
                                              by="name", name="agree")}
        tab = _rtab(page)
        out = tab.check("agree", True)
        self.assertTrue(out["ok"])
        self.assertTrue(out["verified"])

    def test_check_undone_by_page_raises(self):
        page = _FakePage()
        page.stubborn = True
        page.fields_by_name = {"agree": _field(ftype="checkbox",
                                              by="name", name="agree")}
        tab = _rtab(page)
        with self.assertRaises(BrowserError) as ctx:
            tab.check("agree", True)
        self.assertIn("did not stick", str(ctx.exception))

    def test_fill_file_input_with_real_file_uploads(self):
        page = _FakePage()
        page.fields_by_name = {"avatar": _field(ftype="file", by="name",
                                                name="avatar")}
        tab = _rtab(page)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(b"fake-png")
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        out = tab.fill("avatar", tmp.name)
        self.assertTrue(out["ok"])
        self.assertEqual(1, len(page.set_input_files_calls))
        self.assertTrue(out["verified"])
        self.assertTrue(out["attached"].endswith(".png"))

    def test_fill_file_input_with_missing_path_fails_fast(self):
        page = _FakePage()
        page.fields_by_name = {"avatar": _field(ftype="file", by="name",
                                                name="avatar")}
        tab = _rtab(page)
        with self.assertRaises(BrowserError) as ctx:
            tab.fill("avatar", "/no/such/file.png")
        self.assertIn("file input", str(ctx.exception))

    def test_upload_verifies_attachment(self):
        page = _FakePage()
        tab = _rtab(page)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(b"fake-pdf")
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        out = tab.upload("input[type=file]", tmp.name)
        self.assertTrue(out["ok"])
        self.assertTrue(out["attached"].endswith(".pdf"))
        self.assertTrue(out["verified"])

    def test_upload_cleared_by_page_raises(self):
        page = _FakePage()
        page.stubborn = True
        tab = _rtab(page)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(b"x")
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        with self.assertRaises(BrowserError) as ctx:
            tab.upload("input[type=file]", tmp.name)
        self.assertIn("did not attach", str(ctx.exception))


class WaitForFieldTests(unittest.TestCase):
    def test_waits_for_dynamic_field(self):
        page = _FakePage()
        tab = _rtab(page)
        calls = {"n": 0}
        orig_evaluate = page.evaluate

        def late_evaluate(js, arg=None):
            calls["n"] += 1
            if calls["n"] >= 3 and "offsetParent" in (js or ""):
                return _field(by="label", name="email")
            return orig_evaluate(js, arg)

        page.evaluate = late_evaluate
        out = tab.wait_for_field("Email", timeout=5000, poll_ms=50)
        self.assertTrue(out["ok"])
        self.assertEqual("label", out["matched_via"])

    def test_timeout_lists_inventory(self):
        page = _FakePage()
        page.fields_by_name = {}
        # describe_fields uses _DESCRIBE_JS (no offsetParent) — fake it
        orig_evaluate = page.evaluate

        def fake_eval(js, arg=None):
            if "offsetParent" in (js or ""):
                return None  # resolver: nothing matches
            if "querySelectorAll" in (js or ""):
                return [{"tag": "input", "type": "text", "name": "q",
                         "id": "", "label": "Search",
                         "placeholder": "", "aria": ""}]
            return orig_evaluate(js, arg)

        page.evaluate = fake_eval
        tab = _rtab(page)
        with self.assertRaises(BrowserError) as ctx:
            tab.wait_for_field("Email", timeout=200, poll_ms=50)
        msg = str(ctx.exception)
        self.assertIn("did not appear", msg)
        self.assertIn("Search", msg)  # the inventory

    def test_fill_wait_ms_waits_first(self):
        page = _FakePage()
        tab = _rtab(page)
        calls = {"n": 0}
        orig_evaluate = page.evaluate

        def late_evaluate(js, arg=None):
            calls["n"] += 1
            if calls["n"] >= 2 and "offsetParent" in (js or ""):
                return _field(name="q")
            return orig_evaluate(js, arg)

        page.evaluate = late_evaluate
        out = tab.fill("q", "hi", wait_ms=3000)
        self.assertTrue(out["ok"])


# ── download interception wiring ───────────────────────────────────────────


class _FakeDownload:
    def __init__(self, suggested="report.pdf"):
        self.suggested_filename = suggested
        self.saved_to = ""

    def save_as(self, path):
        self.saved_to = path
        with open(path, "wb") as fh:
            fh.write(b"%PDF-1.4 fake")


class DownloadCaptureTests(unittest.TestCase):
    def _svc(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return BrowserService(data_dir=tmp), tmp

    def test_auto_capture_saves_and_registers(self):
        svc, tmp = self._svc()
        page = _FakePage()
        tab = _rtab(page, tmp=tmp)
        tab.session_name = "sess"
        tab._download_dir = svc.data_dir / "downloads" / "sess"
        tab._on_auto_download = (
            lambda record: svc._record_auto_download(tab, record))
        tab._handle_auto_download(_FakeDownload("report.pdf"))
        recs = svc.list_downloads("sess")
        self.assertEqual(1, len(recs))
        rec = recs[0]
        self.assertEqual("completed", rec["status"])
        self.assertEqual("report.pdf", rec["filename"])
        self.assertEqual("auto", rec["trigger"])
        self.assertTrue(rec["path"].endswith("report.pdf"))
        self.assertGreater(rec["size"], 0)
        self.assertTrue(os.path.isfile(rec["path"]))

    def test_suppress_flag_skips_auto_capture(self):
        svc, tmp = self._svc()
        page = _FakePage()
        tab = _rtab(page, tmp=tmp)
        tab._download_dir = svc.data_dir / "downloads" / "sess"
        tab._auto_suppress = True
        tab._handle_auto_download(_FakeDownload("x.pdf"))
        self.assertEqual([], svc.list_downloads("sess"))

    def test_no_download_dir_ignores(self):
        svc, tmp = self._svc()
        page = _FakePage()
        tab = _rtab(page, tmp=tmp)
        tab._download_dir = None
        tab._handle_auto_download(_FakeDownload("x.pdf"))  # must not raise
        self.assertEqual([], svc.list_downloads("sess"))


# ── tab/session management ─────────────────────────────────────────────────


class TabSessionTests(unittest.TestCase):
    def _svc(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return BrowserService(data_dir=tmp), tmp

    def test_plain_tab_switching(self):
        svc, _ = self._svc()
        handle = svc.open_session("s1")
        t1 = handle.open_tab()
        t2 = handle.open_tab()
        self.assertEqual(t2.tab_id, handle.active_tab.tab_id)
        handle.switch_tab(t1.tab_id)
        self.assertEqual(t1.tab_id, handle.active_tab.tab_id)
        handle.close_tab(t2.tab_id)
        self.assertEqual(1, len(handle.list_tabs()))
        with self.assertRaises(BrowserError):
            handle.switch_tab("nope")

    def test_rendered_tab_switching(self):
        svc, tmp = self._svc()
        page = _FakePage()
        t1 = _rtab(page, tmp=tmp)
        t1.tab_id = "r1"
        t1.session_name = "sess"
        t2 = _rtab(page, tmp=tmp)
        t2.tab_id = "r2"
        t2.session_name = "sess"
        svc._rendered_tabs["r1"] = t1
        svc._rendered_tabs["r2"] = t2
        out = svc.switch_rendered_tab("sess", "r1")
        self.assertTrue(out["ok"])
        self.assertEqual("r1", svc.active_rendered_tab("sess")["tab_id"])
        svc.switch_rendered_tab("sess", "r2")
        self.assertEqual("r2", svc.active_rendered_tab("sess")["tab_id"])
        with self.assertRaises(BrowserError):
            svc.switch_rendered_tab("sess", "nope")
        t3 = _rtab(page, tmp=tmp)
        t3.tab_id = "r3"
        t3.session_name = "other"
        svc._rendered_tabs["r3"] = t3
        with self.assertRaises(BrowserError):
            svc.switch_rendered_tab("sess", "r3")  # wrong session

    def test_rendered_close_falls_back_active(self):
        svc, tmp = self._svc()
        page = _FakePage()
        t1 = _rtab(page, tmp=tmp)
        t1.tab_id = "r1"
        t1.session_name = "sess"
        t2 = _rtab(page, tmp=tmp)
        t2.tab_id = "r2"
        t2.session_name = "sess"
        svc._rendered_tabs["r1"] = t1
        svc._rendered_tabs["r2"] = t2
        svc._active_rendered["sess"] = "r2"
        svc.close_rendered_tab("r2")
        self.assertEqual("r1", svc.active_rendered_tab("sess")["tab_id"])
        svc.close_rendered_tab("r1")
        self.assertIsNone(svc.active_rendered_tab("sess"))

    def test_cookie_jar_survives_restart(self):
        from nomorals.tools.browser import BrowserSession

        svc, tmp = self._svc()
        handle = svc.open_session("profile1")
        tab = handle.open_tab()
        cookie = http.cookiejar.Cookie(
            version=0, name="sessionid", value="abc123", port=None,
            port_specified=False, domain="example.com",
            domain_specified=True, domain_initial_dot=False, path="/",
            path_specified=True, secure=False, expires=None, discard=True,
            comment=None, comment_url=None, rest={}, rfc2109=False)
        tab.session.cookie_jar.set_cookie(cookie)
        tab.session._save_cookies()
        cookie_file = os.path.join(
            svc._cookie_dir("profile1"), "profile1_" + tab.tab_id + ".json")
        # note: tab_id contains ":" which the saver sanitizes to "_"
        self.assertTrue(os.path.isfile(cookie_file),
                        f"cookie file not persisted: {cookie_file}")
        # a fresh process reopens the same profile cookie file by name
        fresh = BrowserSession(name=tab.session.name,
                               session_dir=svc._cookie_dir("profile1"))
        names = {c.name: c.value for c in fresh.cookie_jar}
        self.assertEqual("abc123", names.get("sessionid"))

    def test_sessions_save_and_restore(self):
        svc, tmp = self._svc()
        handle = svc.open_session("s1")
        handle.open_tab()  # no navigation: restore must not hit network
        handle.open_tab()
        svc.save()
        svc2 = BrowserService(data_dir=tmp)
        restored = svc2.restore()
        self.assertEqual(1, restored)
        self.assertEqual(["s1"], svc2.list_sessions())
        h2 = svc2.get_session("s1")
        self.assertEqual(2, len(h2.list_tabs()))

    def test_pacing_shared_with_new_tabs(self):
        svc, tmp = self._svc()
        svc.set_pacing(enabled=True, delay_ms=40, jitter_ms=0)
        handle = svc.open_session("s1")
        tab = handle.open_tab()
        self.assertIs(svc.pacing, tab.pacing)
        start = time.monotonic()
        tab._pace("navigate")
        self.assertGreaterEqual(time.monotonic() - start, 0.035)


class DaemonOpTests(unittest.TestCase):
    def _svc(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return BrowserService(data_dir=tmp), tmp

    def test_r_switch_and_r_active_ops(self):
        svc, tmp = self._svc()
        page = _FakePage()
        tab = _rtab(page, tmp=tmp)
        tab.tab_id = "r1"
        tab.session_name = "sess"
        svc._rendered_tabs["r1"] = tab
        out = _handle_op(svc, "r_switch",
                         {"session": "sess", "tab_id": "r1"})
        self.assertTrue(out["result"]["ok"])
        out = _handle_op(svc, "r_active", {"session": "sess"})
        self.assertEqual("r1", out["result"]["tab_id"])

    def test_r_upload_op(self):
        svc, tmp = self._svc()
        page = _FakePage()
        tab = _rtab(page, tmp=tmp)
        tab.tab_id = "r1"
        svc._rendered_tabs["r1"] = tab
        f = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        f.write(b"data")
        f.close()
        self.addCleanup(os.unlink, f.name)
        out = _handle_op(svc, "r_upload",
                         {"tab_id": "r1", "selector": "input[type=file]",
                          "path": f.name})
        self.assertTrue(out["result"]["ok"])
        self.assertTrue(out["result"]["verified"])

    def test_r_wait_field_op(self):
        svc, tmp = self._svc()
        page = _FakePage()
        page.fields_by_name = {"Email": _field(by="label", name="email")}
        tab = _rtab(page, tmp=tmp)
        tab.tab_id = "r1"
        svc._rendered_tabs["r1"] = tab
        out = _handle_op(svc, "r_wait_field",
                         {"tab_id": "r1", "name": "Email"})
        self.assertTrue(out["result"]["ok"])


if __name__ == "__main__":
    unittest.main()
