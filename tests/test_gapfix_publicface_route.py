"""Public-face /p/<slug> serving route (lane 68).

The gate (nomorals/publicface/publish.py) and the /publicface chat
surface already exist, and approving a page hands back a /p/<slug>
link — but no route served that path, so every link 404'd. This lane
serves it: GET /p/<slug> returns the approved page as plain HTML
(no scripts, no styles — the API's CSP allows nothing at all), and
drafts, unpublished, unknown and removed pages all answer 404,
identically. /p/show-<id> resolves the same way: it IS a slug
(shows/composer.py publishes episodes with slug show-<episode.id>),
so no special-casing is needed — but a published episode's page must
reach the same store the route reads, which is why /show publish
persists the page (and /show remove/delete drops it again).

Every test here drives the PRODUCTION path: the durable
PublicPageStore on a temp home plus the real APIServer on loopback.
Teeth: on the pristine base there is no /p/ branch in _handle, so
every HTTP test fails (404 "no route") and the serve-unit tests fail
on the missing module.
"""

from __future__ import annotations

import http.client
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.api.server import APIServer, _make_handler
from nomorals.publicface.publish import (
    PublicPage,
    PublicPageStore,
    publish,
    unpublish,
)
from nomorals.publicface.serve import (
    render_public_page,
    resolve_public_page,
    serve_public_page,
)
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


def _store(tmp: str) -> PublicPageStore:
    return PublicPageStore(str(Path(tmp) / "data" / "publicface.json"))


def _public_page(tmp: str, slug: str = "jamb-physics-101",
                 title: str = "JAMB Physics 101") -> PublicPage:
    store = _store(tmp)
    page = PublicPage(
        title=title, content="Forces and motion, plain and clear.",
        pack="jamb-tutor", slug=slug)
    store.add(page)
    result = publish(page, owner_approved=True)
    assert result.published, result.reason
    store.put(page)
    return page


def _context(home: str) -> Any:
    settings = SimpleNamespace(
        home=home,
        api=SimpleNamespace(token="", max_body_mb=0),
        backup_dir=home,
        workspace_dir=home,
    )
    return SimpleNamespace(
        db=Database(":memory:"), settings=settings,
        tools=ToolRegistry(), bus=None, memory=None,
    )


def _server(context: Any) -> APIServer:
    return APIServer(context, token="", max_body_bytes=64 * 1024)


class _LiveServer:
    """Real HTTP server on loopback; http.client for full header control."""

    def __init__(self, api: APIServer) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(api))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def get(self, path: str) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path)
        return conn.getresponse()


class ServeUnitTests(unittest.TestCase):
    def test_public_page_serves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = _public_page(tmp)
            status, payload = serve_public_page(_store(tmp), page.slug)
            self.assertEqual("ok", status)
            self.assertIn("JAMB Physics 101", payload["html"])
            self.assertIn("Forces and motion", payload["html"])
            self.assertEqual(page.slug, payload["slug"])
            resolved = resolve_public_page(_store(tmp), page.slug)
            self.assertIsNotNone(resolved)
            self.assertEqual(page.id, resolved.id)
            self.assertEqual("public", resolved.status)

    def test_draft_never_serves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            page = PublicPage(
                title="Draft only", content="not yet approved",
                pack="jamb-tutor", slug="draft-only")
            store.add(page)
            self.assertEqual("missing", serve_public_page(store, "draft-only")[0])
            self.assertIsNone(resolve_public_page(store, "draft-only"))

    def test_unknown_slug_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                "missing", serve_public_page(_store(tmp), "nope")[0])
            self.assertIsNone(resolve_public_page(_store(tmp), "nope"))

    def test_unpublish_then_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = _public_page(tmp)
            store = _store(tmp)
            unpublish(store.get(page.id))
            store.put(store.get(page.id))
            self.assertEqual("missing", serve_public_page(store, page.slug)[0])

    def test_show_slug_resolves_like_any_slug(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _public_page(tmp, slug="show-abc123", title="Pilot")
            status, payload = serve_public_page(_store(tmp), "show-abc123")
            self.assertEqual("ok", status)
            self.assertIn("Pilot", payload["html"])

    def test_html_escapes_content_no_scripts_no_styles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            page = PublicPage(
                title="<script>alert(1)</script>",
                content="a <b>bold</b> line\nsecond line",
                pack="jamb-tutor", slug="xss-test")
            store.add(page)
            publish(page, owner_approved=True)
            store.put(page)
            html = render_public_page(page)
            self.assertNotIn("<script>", html)
            self.assertNotIn("<b>bold</b>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("&lt;b&gt;bold&lt;/b&gt;", html)
            self.assertNotIn("<style", html)
            self.assertNotIn('style="', html)

    def test_empty_and_subpath_refs_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _public_page(tmp)
            store = _store(tmp)
            self.assertEqual("missing", serve_public_page(store, "")[0])
            self.assertEqual("missing", serve_public_page(store, "a/b")[0])

    def test_slug_collision_first_public_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            first = PublicPage(
                title="First", content="first copy",
                pack="jamb-tutor", slug="dupe")
            second = PublicPage(
                title="Second", content="second copy",
                pack="jamb-tutor", slug="dupe")
            store.add(first)
            store.add(second)
            publish(first, owner_approved=True)
            publish(second, owner_approved=True)
            store.put(first)
            store.put(second)
            self.assertIs(
                resolve_public_page(store, "dupe"), first)


class HttpRouteTests(unittest.TestCase):
    def test_http_public_page_200_no_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = _public_page(tmp)
            live = _LiveServer(_server(_context(tmp)))
            try:
                resp = live.get(f"/p/{page.slug}")
                body = resp.read().decode("utf-8")
                self.assertEqual(200, resp.status)
                self.assertIn("text/html", resp.getheader("Content-Type"))
                self.assertIn("JAMB Physics 101", body)
                self.assertIn("Forces and motion", body)
                self.assertEqual(
                    "nosniff", resp.getheader("X-Content-Type-Options"))
                self.assertIsNotNone(
                    resp.getheader("Content-Security-Policy"))
                self.assertIsNotNone(resp.getheader("X-Request-ID"))
            finally:
                live.close()

    def test_http_draft_and_unknown_404_identically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            draft = PublicPage(
                title="Draft only", content="not yet approved",
                pack="jamb-tutor", slug="draft-only")
            store.add(draft)
            _public_page(tmp)
            live = _LiveServer(_server(_context(tmp)))
            try:
                for path in ("/p/draft-only", "/p/nope", "/p/"):
                    resp = live.get(path)
                    resp.read()
                    self.assertEqual(404, resp.status, path)
            finally:
                live.close()

    def test_http_unpublish_then_404(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = _public_page(tmp)
            store = _store(tmp)
            unpublish(store.get(page.id))
            store.put(store.get(page.id))
            live = _LiveServer(_server(_context(tmp)))
            try:
                resp = live.get(f"/p/{page.slug}")
                resp.read()
                self.assertEqual(404, resp.status)
            finally:
                live.close()

    def test_http_show_slug_200(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _public_page(tmp, slug="show-abc123", title="Pilot")
            live = _LiveServer(_server(_context(tmp)))
            try:
                resp = live.get("/p/show-abc123")
                body = resp.read().decode("utf-8")
                self.assertEqual(200, resp.status)
                self.assertIn("Pilot", body)
            finally:
                live.close()


if __name__ == "__main__":
    unittest.main()
