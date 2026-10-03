"""Browser capabilities: uploads, proxies, cookies, downloads, rendered-tab
interaction, daemon ops. Uses local HTTP servers only — no real network."""

import contextlib
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nomorals.browser import BrowserError, BrowserService, RenderedTab
from nomorals.browser.daemon import _handle_op
from nomorals.browser.service import _mask_proxy, _mime_category
from nomorals.core.errors import ToolError
from nomorals.tools.browser import (
    BrowserSession,
    _encode_multipart,
)

FORM_HTML = (
    b"<html><head><title>Form</title></head><body>"
    b"<form id='up' action='/upload' method='post' enctype='multipart/form-data'>"
    b"<input type='text' name='note' value=''>"
    b"<input type='file' name='doc'>"
    b"<input type='submit' value='Send'>"
    b"</form>"
    b"<form id='getform' action='/search' method='get'>"
    b"<input type='file' name='doc'>"
    b"<input type='submit' value='Go'>"
    b"</form>"
    b"<a href='/page2'>page two</a>"
    b"</body></html>"
)


class _Handler(http.server.BaseHTTPRequestHandler):
    received = []  # class-level log of (method, path, headers, body)

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/form":
            self._send(200, FORM_HTML, "text/html; charset=utf-8")
        elif self.path == "/page2":
            self._send(200, b"<html><body><h1>Two</h1></body></html>",
                       "text/html; charset=utf-8")
        elif self.path == "/file.bin":
            self._send(200, b"BINARY" * 128, "application/octet-stream")
        elif self.path == "/pic.png":
            self._send(200, b"\x89PNG" + b"x" * 100, "image/png")
        elif self.path == "/cookie":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Set-Cookie", "sess=abc123; Path=/")
            self.send_header("Content-Length", "20")
            self.end_headers()
            self.wfile.write(b"<html><body>hi</body></html>")
        else:
            self._send(404, b"nope", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        type(self).received.append(
            ("POST", self.path, dict(self.headers), body))
        if self.path == "/upload":
            self._send(200, b'{"ok": true}', "application/json")
        elif self.path == "/echo":
            self._send(200, b"<html><body>echo</body></html>", "text/html")
        else:
            self._send(404, b"nope", "text/plain")

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    """Fake forward proxy: answers everything with a canned page."""

    def log_message(self, *args):
        pass

    def _answer(self):
        body = ("<html><head><title>P</title></head><body>"
                f"proxied-response for {self.path}</body></html>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Set-Cookie", "pc=1; Path=/")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _answer
    do_POST = _answer


def _serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.base = _serve(_Handler)
        cls.proxy_server, cls.proxy_base = _serve(_ProxyHandler)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.proxy_server.shutdown()
        cls.server.server_close()
        cls.proxy_server.server_close()

    def setUp(self):
        _Handler.received = []
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))

    def _session(self, **kw):
        kw.setdefault("respect_robots", False)
        return BrowserSession(name="t", session_dir=self.tmp, **kw)


# ── multipart uploads ────────────────────────────────────────────────────


class TestMultipartEncoding(unittest.TestCase):
    def test_structure(self):
        body, ctype = _encode_multipart(
            {"note": "hello"},
            {"doc": ("hello.txt", b"FILEBYTES", "text/plain")},
        )
        boundary = ctype.split("boundary=")[1]
        self.assertTrue(ctype.startswith("multipart/form-data"))
        self.assertIn(f"--{boundary}\r\n".encode(), body)
        self.assertIn(b'name="note"', body)
        self.assertIn(b"hello", body)
        self.assertIn(b'filename="hello.txt"', body)
        self.assertIn(b"FILEBYTES", body)
        self.assertIn(b"Content-Type: text/plain", body)
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))

    def test_two_calls_different_boundaries(self):
        _, c1 = _encode_multipart({}, {"a": ("a", b"x", "text/plain")})
        _, c2 = _encode_multipart({}, {"a": ("a", b"x", "text/plain")})
        self.assertNotEqual(c1, c2)


class TestUploads(_Base):
    def _upload_file(self, content=b"UPLOAD-CONTENT-123"):
        path = os.path.join(self.tmp, "up.txt")
        with open(path, "wb") as fh:
            fh.write(content)
        return path

    def test_submit_multipart_end_to_end(self):
        sess = self._session()
        sess.open(self.base + "/form")
        sess.fill("note", "hello")
        result = sess.submit("0", uploads={"doc": self._upload_file()})
        self.assertTrue(result["ok"])
        self.assertEqual(result["uploaded"], ["doc"])
        self.assertEqual(len(_Handler.received), 1)
        method, path, headers, body = _Handler.received[0]
        self.assertEqual(path, "/upload")
        self.assertIn("multipart/form-data", headers.get("Content-Type", ""))
        self.assertIn(b'filename="up.txt"', body)
        self.assertIn(b"UPLOAD-CONTENT-123", body)
        self.assertIn(b'name="note"', body)
        self.assertIn(b"hello", body)

    def test_fill_on_file_field_then_submit(self):
        sess = self._session()
        sess.open(self.base + "/form")
        sess.fill("doc", self._upload_file())
        result = sess.submit("0")
        self.assertEqual(result["uploaded"], ["doc"])
        _, _, _, body = _Handler.received[0]
        self.assertIn(b"UPLOAD-CONTENT-123", body)

    def test_missing_file_fails_fast(self):
        sess = self._session()
        sess.open(self.base + "/form")
        with self.assertRaises(ToolError):
            sess.submit("0", uploads={"doc": "/no/such/file.txt"})

    def test_upload_on_get_form_fails_fast(self):
        sess = self._session()
        sess.open(self.base + "/form")
        with self.assertRaises(ToolError) as ctx:
            sess.submit("1", uploads={"doc": self._upload_file()})
        self.assertIn("POST", str(ctx.exception))

    def test_uploads_through_do(self):
        sess = self._session()
        sess.do("open", url=self.base + "/form")
        result = sess.do("submit", target="0",
                         uploads={"doc": self._upload_file()})
        self.assertTrue(result["ok"])
        self.assertEqual(result["uploaded"], ["doc"])

    def test_tab_upload_stages_then_submit(self):
        handle = self.svc.open_session("s")
        tab = handle.open_tab(self.base + "/form")
        staged = tab.upload("doc", self._upload_file())
        self.assertTrue(staged["ok"])
        result = tab.submit("0")
        self.assertEqual(result["uploaded"], ["doc"])

    def test_tab_upload_missing_file_fails_fast(self):
        handle = self.svc.open_session("s")
        tab = handle.open_tab(self.base + "/form")
        with self.assertRaises(BrowserError):
            tab.upload("doc", "/no/such/file.txt")


# ── proxies ──────────────────────────────────────────────────────────────


class _FakePool:
    def __init__(self, urls):
        self.urls = list(urls)
        self.i = 0

    def rotate(self):
        url = self.urls[self.i % len(self.urls)]
        self.i += 1
        return {"url_with_auth": url, "id": f"proxy-{self.i}"}


class TestProxy(_Base):
    def test_traffic_goes_through_proxy(self):
        with mock.patch.dict(os.environ, {"no_proxy": "", "NO_PROXY": ""}):
            sess = self._session(proxy_url=self.proxy_base)
            result = sess.open("http://example.com/")
            self.assertTrue(result["ok"])
            self.assertIn("proxied-response", sess.text()["text"])
            self.assertEqual(sess.proxy_url, self.proxy_base)

    def test_set_proxy_keeps_cookies(self):
        with mock.patch.dict(os.environ, {"no_proxy": "", "NO_PROXY": ""}):
            sess = self._session(proxy_url=self.proxy_base)
            sess.open("http://example.com/")
            self.assertGreater(len(sess.cookie_jar), 0)  # pc=1 set
            out = sess.set_proxy("")
            self.assertEqual(out["proxy"], "direct")
            self.assertGreater(len(sess.cookie_jar), 0)  # jar survived
            self.assertEqual(sess.proxy_url, "")

    def test_mask_proxy(self):
        masked = _mask_proxy("http://user:s3cret@host:8080")
        self.assertNotIn("s3cret", masked)
        self.assertIn("user:***@host:8080", masked)
        self.assertEqual(_mask_proxy("http://host:8080"), "http://host:8080")

    def test_attach_pool_and_rotate(self):
        handle = self.svc.open_session("s")
        tab = handle.open_tab(self.base + "/form")
        pool = _FakePool([self.proxy_base])
        attached = self.svc.attach_proxy_pool(pool)
        self.assertTrue(attached["attached"])
        self.assertTrue(self.svc.proxy_pool_attached())
        result = self.svc.rotate_proxy("s")
        self.assertEqual(result["sessions"], ["s"])
        self.assertEqual(result["tabs"], 1)
        self.assertEqual(tab.session.proxy_url, self.proxy_base)
        self.assertIn("proxy", result)
        self.assertNotIn("127.0.0.1:", result["proxy"][:0])  # masked form

    def test_rotate_all_sessions(self):
        self.svc.open_session("a").open_tab(self.base + "/form")
        self.svc.open_session("b").open_tab(self.base + "/form")
        self.svc.attach_proxy_pool(_FakePool([self.proxy_base]))
        result = self.svc.rotate_proxy()
        self.assertEqual(sorted(result["sessions"]), ["a", "b"])
        self.assertEqual(result["tabs"], 2)

    def test_new_tabs_inherit_session_proxy(self):
        self.svc.open_session("s")
        self.svc.attach_proxy_pool(_FakePool([self.proxy_base]))
        self.svc.rotate_proxy("s")
        tab = self.svc.get_session("s").open_tab(self.base + "/form")
        self.assertEqual(tab.session.proxy_url, self.proxy_base)

    def test_set_session_proxy_explicit(self):
        self.svc.open_session("s").open_tab(self.base + "/form")
        result = self.svc.set_session_proxy("s", self.proxy_base)
        self.assertIn("proxy", result)
        tab = self.svc.get_session("s").list_tabs()
        self.assertEqual(
            self.svc.find_tab(tab[0]["tab_id"]).session.proxy_url,
            self.proxy_base)

    def test_attach_bad_pool_fails_fast(self):
        with self.assertRaises(BrowserError):
            self.svc.attach_proxy_pool(object())
        with self.assertRaises(BrowserError):
            self.svc.attach_proxy_pool(_FakePool([""]))

    def test_rotate_without_pool_fails_fast(self):
        self.svc.open_session("s")
        with self.assertRaises(BrowserError) as ctx:
            self.svc.rotate_proxy("s")
        self.assertIn("attach_proxy_pool", str(ctx.exception))

    def test_rotate_unknown_session_fails_fast(self):
        self.svc.attach_proxy_pool(_FakePool([self.proxy_base]))
        with self.assertRaises(BrowserError):
            self.svc.rotate_proxy("nope")

    def test_detach_clears_proxies(self):
        handle = self.svc.open_session("s")
        tab = handle.open_tab(self.base + "/form")
        self.svc.attach_proxy_pool(_FakePool([self.proxy_base]))
        self.svc.rotate_proxy("s")
        self.svc.detach_proxy_pool()
        self.assertFalse(self.svc.proxy_pool_attached())
        self.assertEqual(tab.session.proxy_url, "")


# ── cookies ──────────────────────────────────────────────────────────────


class TestCookies(_Base):
    def _open_with_cookie(self, svc=None, session="s"):
        svc = svc or self.svc
        handle = svc.open_session(session)
        tab = handle.open_tab(self.base + "/cookie")
        return handle, tab

    def test_tab_cookies_inspection(self):
        _, tab = self._open_with_cookie()
        cookies = tab.cookies()
        self.assertEqual(len(cookies), 1)
        self.assertEqual(cookies[0]["name"], "sess")
        self.assertEqual(cookies[0]["value"], "abc123")
        self.assertIn("domain", cookies[0])

    def test_export_import_netscape_roundtrip(self):
        self._open_with_cookie()
        out = os.path.join(self.tmp, "cookies.txt")
        exported = self.svc.export_cookies("s", out, format="netscape")
        self.assertEqual(exported["cookies"], 1)
        self.assertTrue(Path(out).is_file())
        self.assertIn("sess", Path(out).read_text())

        svc2 = BrowserService(data_dir=os.path.join(self.tmp, "data2"))
        svc2.open_session("s2").open_tab()  # tab needed as import target
        imported = svc2.import_cookies("s2", out, format="netscape")
        self.assertEqual(imported["imported"], 1)
        tab = svc2.get_session("s2").active_tab
        names = {c["name"] for c in tab.cookies()}
        self.assertIn("sess", names)

    def test_export_import_json_roundtrip(self):
        self._open_with_cookie()
        out = os.path.join(self.tmp, "cookies.json")
        self.assertEqual(
            self.svc.export_cookies("s", out, format="json")["cookies"], 1)
        payload = json.loads(Path(out).read_text())
        self.assertEqual(payload[0]["name"], "sess")

        svc2 = BrowserService(data_dir=os.path.join(self.tmp, "data2"))
        svc2.open_session("s2").open_tab()
        imported = svc2.import_cookies("s2", out, format="json")
        self.assertEqual(imported["imported"], 1)
        names = {c["name"] for c in svc2.get_session("s2").active_tab.cookies()}
        self.assertIn("sess", names)

    def test_export_unknown_session_fails_fast(self):
        with self.assertRaises(BrowserError):
            self.svc.export_cookies("nope", os.path.join(self.tmp, "c.txt"))

    def test_export_bad_format_fails_fast(self):
        self._open_with_cookie()
        with self.assertRaises(BrowserError):
            self.svc.export_cookies("s", os.path.join(self.tmp, "c.txt"),
                                    format="yaml")

    def test_import_missing_file_fails_fast(self):
        self._open_with_cookie()
        with self.assertRaises(BrowserError):
            self.svc.import_cookies("s", os.path.join(self.tmp, "nope.txt"))

    def test_import_no_tabs_fails_fast(self):
        self.svc.open_session("empty")
        out = os.path.join(self.tmp, "c.txt")
        Path(out).write_text(
            "# Netscape HTTP Cookie File\n.example.com\tTRUE\t/\tFALSE\t0\tsess\tabc\n")
        with self.assertRaises(BrowserError):
            self.svc.import_cookies("empty", out)

    def test_import_malformed_json_fails_fast(self):
        self._open_with_cookie()
        out = os.path.join(self.tmp, "bad.json")
        Path(out).write_text('[{"name": "", "domain": ""}]')
        with self.assertRaises(BrowserError):
            self.svc.import_cookies("s", out, format="json")


# ── downloads ────────────────────────────────────────────────────────────


class TestDownloads(_Base):
    def test_download_registers_and_lists(self):
        handle = self.svc.open_session("s")
        tab = handle.open_tab(self.base + "/form")
        result = self.svc.download(tab, self.base + "/file.bin")
        recs = self.svc.list_downloads()
        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["size"], result.size)
        self.assertEqual(rec["mime"], "application/octet-stream")
        self.assertEqual(rec["category"], "other")
        self.assertEqual(rec["session"], "s")
        self.assertEqual(rec["tab_id"], tab.tab_id)
        self.assertEqual(rec["path"], result.path)

    def test_download_organize_by_category(self):
        result = self.svc.download(self.base + "/pic.png", organize=True)
        self.assertIn(os.sep + "images" + os.sep, result.path)
        recs = self.svc.list_downloads(category="images")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["category"], "images")

    def test_download_failure_registers_failed(self):
        with self.assertRaises(BrowserError):
            self.svc.download(self.base + "/missing")
        recs = self.svc.list_downloads()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["status"], "failed")
        self.assertTrue(recs[0]["error"])

    def test_wait_for_download_completed_returns(self):
        self.svc.download(self.base + "/file.bin")
        rec = self.svc.list_downloads()[0]
        done = self.svc.wait_for_download(rec["id"], timeout=5)
        self.assertEqual(done["status"], "completed")

    def test_wait_for_download_unknown_id_fails_fast(self):
        with self.assertRaises(BrowserError):
            self.svc.wait_for_download("nope", timeout=1)

    def test_wait_for_download_finalizes_in_progress(self):
        path = os.path.join(self.tmp, "late.bin")
        Path(path).write_bytes(b"late-bytes")
        rec_id = "dl-test-1"
        self.svc._record_download({
            "id": rec_id, "session": "s", "tab_id": "", "rendered_tab_id": "",
            "url": self.base + "/file.bin", "path": path, "size": 0,
            "mime": "application/octet-stream", "category": "",
            "status": "in_progress", "started_at": 0.0, "finished_at": 0.0,
            "error": "",
        })
        done = self.svc.wait_for_download(rec_id, timeout=5)
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["size"], len(b"late-bytes"))

    def test_registry_persists(self):
        self.svc.download(self.base + "/file.bin")
        self.assertTrue((Path(self.svc.data_dir) / "downloads.json").is_file())
        svc2 = BrowserService(data_dir=self.svc.data_dir)
        self.assertEqual(len(svc2.list_downloads()), 1)

    def test_mime_category(self):
        self.assertEqual(_mime_category("image/png"), "images")
        self.assertEqual(_mime_category("video/mp4"), "videos")
        self.assertEqual(_mime_category("text/html"), "documents")
        self.assertEqual(_mime_category("application/pdf"), "documents")
        self.assertEqual(_mime_category("application/octet-stream"), "other")
        self.assertEqual(_mime_category(""), "other")


# ── rendered tabs: interaction on the live DOM ───────────────────────────


class _FakeDownload:
    def __init__(self, page):
        self.page = page
        self.suggested_filename = "report.csv"

    def save_as(self, path):
        Path(path).write_bytes(self.page.download_bytes)


class _FakeDownloadCtx:
    def __init__(self, page):
        self.page = page

    def __enter__(self):
        info = mock.MagicMock()
        info.value = _FakeDownload(self.page)
        return info

    def __exit__(self, *args):
        return False


class _FakeRPage:
    def __init__(self):
        self.url = "https://example.com/"
        self.goto_calls = []
        self.fill_calls = []
        self.click_calls = []
        self.eval_calls = []
        self.wait_calls = []
        self.screenshot_calls = []
        self.set_input_files_calls = []
        self.closed = False
        self.body_text = "rendered body"
        self.download_bytes = b"csv-bytes"
        self.page_html = (
            "<html><head><title>T</title></head><body>"
            "<h1>Hi</h1>"
            "<form id='f' action='/go' method='post'>"
            "<input type='text' name='q' value=''>"
            "<input type='submit' value='Go'>"
            "</form>"
            "<a href='/x'>link</a></body></html>"
        )
        self.cookies_list = [{"name": "rc", "value": "1",
                              "domain": "example.com", "path": "/",
                              "secure": False, "expires": -1,
                              "httpOnly": False}]

    def goto(self, url, **kw):
        self.goto_calls.append((url, kw))
        self.url = url

    def title(self):
        return "Fake T"

    def inner_text(self, selector):
        return self.body_text

    def content(self):
        return self.page_html

    def eval_on_selector(self, selector, js):
        self.eval_calls.append((selector, js))

    def eval_on_selector_all(self, selector, js):
        return []

    def fill(self, selector, value):
        self.fill_calls.append((selector, value))

    def click(self, selector):
        self.click_calls.append(selector)
        self.url = "https://example.com/clicked"

    def wait_for_selector(self, selector, state="visible", timeout=10000):
        self.wait_calls.append((selector, state, timeout))
        if selector == "#never":
            raise TimeoutError("timeout waiting")

    def screenshot(self, path=None, full_page=False):
        self.screenshot_calls.append((path, full_page))
        Path(path).write_bytes(b"\x89PNG-fake")

    def set_input_files(self, selector, path):
        self.set_input_files_calls.append((selector, path))

    def expect_download(self):
        return _FakeDownloadCtx(self)

    def close(self):
        self.closed = True


class _FakeRContext:
    def __init__(self, page, **kwargs):
        self._page = page
        self.kwargs = kwargs
        self.saved_path = None

    def new_page(self):
        return self._page

    def storage_state(self, path=None):
        self.saved_path = path

    def cookies(self):
        return list(self._page.cookies_list)

    def close(self):
        pass


class _FakeRBrowser:
    def __init__(self, context):
        self._context = context
        self.closed = False

    def new_context(self, **kwargs):
        return self._context

    def close(self):
        self.closed = True


class _FakeRChromium:
    def __init__(self, browser):
        self._browser = browser
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return self._browser


class _FakeRSyncPlaywright:
    def __init__(self, chromium):
        self.chromium = chromium

    def start(self):
        return self


@contextlib.contextmanager
def _mocked_rplaywright(page):
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
        yield page, context, browser, chromium


class TestRenderedInteraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))

    def _open(self, page, url="https://example.com/"):
        with _mocked_rplaywright(page):
            return self.svc.open_rendered_tab("s", url)

    def test_fill_by_name(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page):
            result = tab.fill("q", "hello")
        self.assertTrue(result["ok"])
        selector, value = page.fill_calls[0]
        self.assertIn('name="q"', selector)
        self.assertEqual(value, "hello")

    def test_fill_empty_name_fails_fast(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page), self.assertRaises(BrowserError):
            tab.fill("", "x")

    def test_click_text_navigates_and_updates_history(self):
        page = _FakeRPage()
        tab = self._open(page)
        before = len(tab.history)
        with _mocked_rplaywright(page):
            result = tab.click("link")
        self.assertTrue(result["navigated"])
        self.assertEqual(result["url"], "https://example.com/clicked")
        self.assertEqual(len(tab.history), before + 1)
        self.assertIn("text=link", page.click_calls[0])

    def test_click_selector_passthrough(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page):
            tab.click("#go")
        self.assertEqual(page.click_calls[0], "#go")

    def test_submit_form_direct(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page):
            result = tab.submit()
        self.assertTrue(result["ok"])
        self.assertEqual(page.eval_calls[0][0], "form")

    def test_wait_for_ok_and_timeout(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page):
            result = tab.wait_for("#f")
        self.assertTrue(result["ok"])
        self.assertEqual(page.wait_calls[0][0], "#f")
        with _mocked_rplaywright(page), self.assertRaises(BrowserError):
            tab.wait_for("#never", timeout=50)

    def test_screenshot_captures_live_page(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page):
            shot = tab.screenshot()
        self.assertTrue(Path(shot["path"]).is_file())
        self.assertGreater(Path(shot["path"]).stat().st_size, 0)
        # no fresh load: the fake page's url was NOT re-navigated
        self.assertEqual(len(page.goto_calls), 1)

    def test_extract_kinds(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page):
            headings = tab.extract(kind="headings")
            forms = tab.extract(kind="forms")
        self.assertEqual(headings["items"][0]["text"], "Hi")
        self.assertEqual(forms["forms"][0]["id"], "f")
        field_names = {f["name"] for f in forms["forms"][0]["fields"]}
        self.assertIn("q", field_names)

    def test_extract_bad_kind_fails_fast(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page), self.assertRaises(BrowserError):
            tab.extract(kind="bogus")

    def test_cookies(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page):
            cookies = tab.cookies()
        self.assertEqual(cookies[0]["name"], "rc")
        self.assertFalse(cookies[0]["http_only"])

    def test_upload(self):
        page = _FakeRPage()
        tab = self._open(page)
        path = os.path.join(self.tmp, "f.txt")
        Path(path).write_bytes(b"x")
        with _mocked_rplaywright(page):
            result = tab.upload("input[type=file]", path)
        self.assertTrue(result["ok"])
        self.assertEqual(page.set_input_files_calls[0][1], path)

    def test_upload_missing_file_fails_fast(self):
        page = _FakeRPage()
        tab = self._open(page)
        with _mocked_rplaywright(page), self.assertRaises(BrowserError):
            tab.upload("input[type=file]", "/no/such/file")

    def test_trigger_download(self):
        page = _FakeRPage()
        tab = self._open(page)
        dest = os.path.join(self.tmp, "dl")
        with _mocked_rplaywright(page):
            result = tab.trigger_download("download link", dest)
        self.assertTrue(Path(result["path"]).is_file())
        self.assertEqual(Path(result["path"]).read_bytes(), b"csv-bytes")

    def test_service_rendered_tab_download_registers(self):
        page = _FakeRPage()
        with _mocked_rplaywright(page):
            tab = self.svc.open_rendered_tab("s", "https://example.com/")
            rec = self.svc.rendered_tab_download(tab.tab_id, "dl")
        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["rendered_tab_id"], tab.tab_id)
        self.assertEqual(len(self.svc.list_downloads()), 1)

    def test_service_screenshot_rendered(self):
        page = _FakeRPage()
        with _mocked_rplaywright(page):
            tab = self.svc.open_rendered_tab("s", "https://example.com/")
            shot = self.svc.screenshot_rendered(tab.tab_id)
        self.assertTrue(Path(shot.path).is_file())

    def test_open_rendered_tab_with_proxy(self):
        page = _FakeRPage()
        with _mocked_rplaywright(page) as (_, _, _, chromium):
            self.svc.open_rendered_tab(
                "s", "https://example.com/",
                proxy="http://user:pw@proxyhost:8080")
        proxy_cfg = chromium.launch_kwargs.get("proxy")
        self.assertEqual(proxy_cfg["server"], "http://proxyhost:8080")
        self.assertEqual(proxy_cfg["username"], "user")
        self.assertEqual(proxy_cfg["password"], "pw")

    def test_open_rendered_tab_inherits_session_proxy(self):
        page = _FakeRPage()
        self.svc.open_session("s")
        self.svc.attach_proxy_pool(_FakePool(["http://h:3128"]))
        self.svc.rotate_proxy("s")
        with _mocked_rplaywright(page) as (_, _, _, chromium):
            self.svc.open_rendered_tab("s", "https://example.com/")
        self.assertEqual(chromium.launch_kwargs["proxy"]["server"],
                         "http://h:3128")

    def test_interaction_without_navigate_fails_fast(self):
        page = _FakeRPage()
        with _mocked_rplaywright(page):
            tab = self.svc.open_rendered_tab("s")  # no url
            with self.assertRaises(BrowserError):
                tab.fill("q", "x")
            with self.assertRaises(BrowserError):
                tab.click("x")


# ── daemon ops ─────────────────────────────────────────────────────────


class TestDaemonOps(_Base):
    def _open_form(self):
        return _handle_op(self.svc, "open",
                          {"session": "s", "url": self.base + "/form"})

    def test_fill_click_submit_extract(self):
        self._open_form()
        filled = _handle_op(self.svc, "fill",
                            {"session": "s", "name": "note", "value": "hi"})
        self.assertTrue(filled["result"]["ok"])
        clicked = _handle_op(self.svc, "click",
                             {"session": "s", "target": "page two"})
        self.assertIn("/page2", clicked["result"]["url"])
        self._open_form()
        extracted = _handle_op(self.svc, "extract",
                               {"session": "s", "kind": "forms"})
        self.assertEqual(extracted["result"]["count"], 2)

    def test_submit_with_uploads(self):
        self._open_form()
        path = os.path.join(self.tmp, "d.txt")
        Path(path).write_bytes(b"daemon-upload")
        result = _handle_op(self.svc, "submit", {
            "session": "s", "target": "0", "uploads": {"doc": path}})
        self.assertEqual(result["result"]["uploaded"], ["doc"])

    def test_submit_bad_uploads_type(self):
        self._open_form()
        from nomorals.browser.daemon import DaemonError
        with self.assertRaises(DaemonError):
            _handle_op(self.svc, "submit",
                       {"session": "s", "uploads": ["not-a-dict"]})

    def test_task_op(self):
        self._open_form()  # daemon task op runs on the session's active tab
        result = _handle_op(self.svc, "task", {
            "session": "s",
            "steps": [
                {"act": "open", "url": self.base + "/form"},
                {"act": "extract", "kind": "forms"},
            ],
        })
        self.assertTrue(result["result"]["ok"])
        self.assertEqual(result["result"]["steps_done"], 2)

    def test_cookies_ops(self):
        _handle_op(self.svc, "open",
                   {"session": "s", "url": self.base + "/cookie"})
        cookies = _handle_op(self.svc, "cookies", {"session": "s"})
        self.assertEqual(cookies["cookies"][0]["name"], "sess")
        out = os.path.join(self.tmp, "dc.txt")
        exported = _handle_op(self.svc, "cookies_export",
                              {"session": "s", "path": out})
        self.assertEqual(exported["result"]["cookies"], 1)
        imported = _handle_op(self.svc, "cookies_import",
                              {"session": "s", "path": out})
        self.assertEqual(imported["result"]["imported"], 1)

    def test_proxy_ops(self):
        self._open_form()
        status = _handle_op(self.svc, "proxy",
                            {"session": "s", "action": "status"})
        self.assertFalse(status["attached"])
        # attach a pool directly on the service, then rotate via the op
        self.svc.attach_proxy_pool(_FakePool(["http://h:3128"]))
        rotated = _handle_op(self.svc, "proxy",
                             {"session": "s", "action": "rotate"})
        self.assertIn("proxy", rotated["result"])
        status = _handle_op(self.svc, "proxy",
                            {"session": "s", "action": "status"})
        self.assertTrue(status["attached"])
        self.assertIn("h:3128", status["proxy"])
        cleared = _handle_op(self.svc, "proxy",
                             {"session": "s", "action": "clear"})
        self.assertEqual(cleared["result"]["proxy"], "direct")
        from nomorals.browser.daemon import DaemonError
        with self.assertRaises(DaemonError):
            _handle_op(self.svc, "proxy",
                       {"session": "s", "action": "bogus"})

    def test_downloads_op(self):
        _handle_op(self.svc, "open",
                   {"session": "s", "url": self.base + "/form"})
        _handle_op(self.svc, "download",
                   {"session": "s", "url": self.base + "/file.bin"})
        recs = _handle_op(self.svc, "downloads", {"session": "s"})
        self.assertEqual(len(recs["downloads"]), 1)
        waited = _handle_op(
            self.svc, "wait_download",
            {"download_id": recs["downloads"][0]["id"], "timeout": 5})
        self.assertEqual(waited["result"]["status"], "completed")

    def test_rendered_ops_fail_fast_without_playwright(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(BrowserError):
                _handle_op(self.svc, "r_open",
                           {"session": "s", "url": "https://example.com/"})
        with self.assertRaises(BrowserError):
            _handle_op(self.svc, "r_close", {"tab_id": "nope"})


# ── tool-level regression: extractor refactor kept behavior ────────────


class TestExtractorRefactor(_Base):
    def test_structured_extract_shapes_unchanged(self):
        sess = self._session()
        sess.open(self.base + "/form")
        forms = sess.extract(kind="forms")
        self.assertEqual(forms["kind"], "forms")
        self.assertEqual(forms["count"], 2)
        headings = sess.extract(kind="headings")
        self.assertEqual(headings["kind"], "headings")

    def test_unknown_kind_still_fails_fast(self):
        sess = self._session()
        sess.open(self.base + "/form")
        with self.assertRaises(ToolError):
            sess.extract(kind="bogus")


if __name__ == "__main__":
    unittest.main()
