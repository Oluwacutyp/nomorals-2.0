"""Browser tool: stdlib session — open, read, click, forms, extract.

All tests run against a localhost HTTP server (no external egress needed):
cookies persist per session, scripts never leak into text, markdown renders
headings/links/emphasis, fills override field defaults per-form, and the
registry exposes one ``browser`` tool behind NET_BROWSER.
"""

from __future__ import annotations

import http.server
import socketserver
import tempfile
import threading
import unittest
from typing import Any
from urllib.parse import unquote_plus, urlparse

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.core.errors import ToolError
from nomorals.core.policy import Capability, CapabilitySet
from nomorals.tools.browser import (
    BrowserSession,
    drop_session,
    get_session,
    node_to_markdown,
    parse_html,
)

PAGE = b"""<!DOCTYPE html>
<html><head><title>Test Page</title></head>
<body>
<h1>Welcome</h1>
<p id="intro">Hello <b>world</b>, this is a test page.</p>
<ul><li>one</li><li>two</li></ul>
<a href="/second">second page</a>
<a href="/ext">external</a>
<form id="f1" action="/submit" method="post">
  <input name="q" value="default">
  <input name="x" value="1">
</form>
<form action="/getform" method="get"><input name="q"></form>
<script>var secret = "should never leak";</script>
</body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:
        pass

    def _send(self, body: bytes, code: int = 200, ctype: str = "text/html") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path
        if path == "/":
            self._send(PAGE)
        elif path.startswith("/second"):
            self._send(b"<html><head><title>Second</title></head>"
                       b"<body><p>second page body</p></body></html>")
        elif path.startswith("/ext"):
            self._send(b"<html><head><title>Ext</title></head>"
                       b"<body><p>the external page</p></body></html>")
        elif path.startswith("/getform"):
            qs = urlparse(path).query
            self._send(f"<html><head><title>GotForm</title></head>"
                       f"<body><p>you asked: {unquote_plus(qs)}</p></body></html>".encode())
        elif path == "/setcookie":
            self.send_response(302)
            self.send_header("Set-Cookie", "visited=yes; Path=/")
            self.send_header("Location", "/")
            self.end_headers()
        elif path == "/check":
            cookie = self.headers.get("Cookie", "")
            self._send(f"<html><head><title>Check</title></head>"
                       f"<body><p>cookie: {cookie}</p></body></html>".encode())
        elif path == "/missing":
            self._send(b"nope", 404, "text/plain")
        else:
            self._send(b"nf", 404, "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        self._send(f"<html><head><title>Submitted</title></head>"
                   f"<body><p>received: {body}</p></body></html>".encode())


class _Site:
    def __init__(self) -> None:
        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class BrowserSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.site = _Site()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.site.stop()

    def setUp(self) -> None:
        self.session_name = "test-" + tempfile.mkdtemp().split("/")[-1]
        self.s = get_session(self.session_name, respect_robots=False, timeout=10)

    def tearDown(self) -> None:
        drop_session(self.session_name)

    def open_home(self) -> dict[str, Any]:
        return self.s.do("open", url=self.site.base + "/")

    def test_open_reports_page_shape(self) -> None:
        r = self.open_home()
        self.assertTrue(r["ok"])
        self.assertEqual(r["status"], 200)
        self.assertEqual(r["title"], "Test Page")
        self.assertEqual(r["links"], 2)
        self.assertEqual(r["forms"], 2)

    def test_text_excludes_scripts(self) -> None:
        self.open_home()
        t = self.s.do("text")
        self.assertNotIn("should never leak", t["text"])
        self.assertIn("Hello world, this is a test page.", t["text"])

    def test_markdown_renders_structure(self) -> None:
        self.open_home()
        m = self.s.do("markdown")
        self.assertIn("# Welcome", m["markdown"])
        self.assertIn("**world**", m["markdown"])
        self.assertIn("[second page](/second)", m["markdown"])
        self.assertIn("- one", m["markdown"])

    def test_links_are_absolute_and_deduped(self) -> None:
        self.open_home()
        l = self.s.do("links")
        pairs = [(x["text"], x["url"]) for x in l["links"]]
        self.assertIn(("second page", self.site.base + "/second"), pairs)
        self.assertIn(("external", self.site.base + "/ext"), pairs)
        self.assertEqual(len(pairs), len(set(u for _, u in pairs)))

    def test_click_follows_link_and_keeps_session(self) -> None:
        self.open_home()
        c = self.s.do("click", target="second page")
        self.assertIn("Second", c["title"])
        self.assertTrue(self.s.url.endswith("/second"))

    def test_click_by_index(self) -> None:
        self.open_home()
        c = self.s.do("click", target="1")  # the second link
        self.assertIn("Ext", c["title"])
        self.assertTrue(self.s.url.endswith("/ext"))

    def test_fill_and_post_submit(self) -> None:
        self.open_home()
        self.s.do("fill", name="q", value="nm test")
        self.s.do("submit", target="f1")
        t = self.s.do("text")
        self.assertIn("q=nm+test&x=1", t["text"])  # fill + untouched default

    def test_submit_without_fill_uses_field_defaults(self) -> None:
        self.open_home()
        self.s.do("submit", target="f1")
        t = self.s.do("text")
        self.assertIn("q=default&x=1", t["text"])

    def test_get_form_submission(self) -> None:
        self.open_home()
        self.s.do("fill", name="q", value="hello world")
        self.s.do("submit", target="1")
        t = self.s.do("text")
        # server decodes the url-encoded + back to a space
        self.assertIn("you asked: q=hello world", t["text"])

    def test_fills_do_not_leak_between_forms(self) -> None:
        # form 0 has q=default; form 1 (GET) has its own empty q.
        self.open_home()
        self.s.do("submit", target="1")
        t = self.s.do("text")
        self.assertIn("you asked: q=", t["text"])
        self.assertNotIn("q=default", t["text"])

    def test_extract_by_id_tag_and_class(self) -> None:
        self.open_home()
        by_id = self.s.do("extract", target="#intro")
        self.assertEqual(by_id["count"], 1)
        self.assertIn("Hello world", by_id["matches"][0])
        by_tag = self.s.do("extract", target="li")
        self.assertEqual(by_tag["count"], 2)
        self.assertEqual(by_tag["matches"][0], "one")
        missing = self.s.do("extract", target="#nope")
        self.assertEqual(missing["count"], 0)

    def test_state_tracks_requests(self) -> None:
        self.open_home()
        self.s.do("text")
        st = self.s.do("state")
        self.assertGreaterEqual(st["requests"], 1)
        self.assertEqual(st["session"], self.session_name)

    def test_cookies_persist_within_the_session(self) -> None:
        self.s.do("open", url=self.site.base + "/setcookie")  # 302 -> / with Set-Cookie
        self.s.do("open", url=self.site.base + "/check")
        body = self.s.do("text")["text"]
        self.assertIn("visited=yes", body)

    def test_sessions_are_isolated(self) -> None:
        self.s.do("open", url=self.site.base + "/setcookie")
        other = get_session(self.session_name + "-other", respect_robots=False, timeout=10)
        try:
            other.do("open", url=self.site.base + "/check")
            body = other.do("text")["text"]
            self.assertNotIn("visited=yes", body)
        finally:
            drop_session(self.session_name + "-other")

    def test_errors_are_tool_errors(self) -> None:
        with self.assertRaises(ToolError):
            self.s.do("open")  # no url
        with self.assertRaises(ToolError):
            self.s.do("text")  # no page open
        with self.assertRaises(ToolError):
            self.s.do("teleport")  # unknown action
        self.open_home()
        with self.assertRaises(ToolError):
            self.s.do("click", target="no such link")
        with self.assertRaises(ToolError):
            self.s.do("fill")  # no name

    def test_http_error_page_is_readable(self) -> None:
        r = self.s.do("open", url=self.site.base + "/missing")
        self.assertFalse(r["ok"])
        self.assertEqual(r["status"], 404)

    def test_markdown_helper_on_raw_dom(self) -> None:
        dom = parse_html("<h2>Hi</h2><a href='/x'>go</a><strong>now</strong>")
        md = node_to_markdown(dom)
        self.assertIn("## Hi", md)
        self.assertIn("[go](/x)", md)
        self.assertIn("**now**", md)


class BrowserToolRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.site = _Site()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.site.stop()

    def test_registered_behind_net_browser(self) -> None:
        ctx = build_context(
            Settings(home=tempfile.mkdtemp(prefix="nm-brow-")),
            with_router=False, with_memory=False,
            with_executor=False, with_tools=True,
        )
        ctx.__enter__()
        try:
            reg = ctx.tools
            self.assertIn("browser", reg.names())
            schema = next(s for s in reg.schemas() if s["name"] == "browser")
            self.assertEqual(schema["capability"], Capability.NET_BROWSER)

            # granted: open works
            ok = reg.call("browser", action="open", url=self.site.base + "/")
            self.assertTrue(ok.ok)
            self.assertEqual(ok.value["title"], "Test Page")
            text = reg.call("browser", action="text", session="default")
            self.assertTrue(text.ok)
            self.assertIn("Hello world", text.value["text"])

            # not granted: denied, nothing runs
            denied = reg.call(
                "browser", action="state",
                capabilities=CapabilitySet.of("net.out"),
            )
            self.assertFalse(denied.ok)
        finally:
            ctx.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
