"""Wave 46 — browser console (nm web): localhost, key-gated.

- social/chat/web.py: WebAdapter — replies captured in-process,
  inbound pushed by the console server
- api/console.py: WebConsole — the same partner brain (mood, memory,
  slash commands) driven over localhost HTTP with a per-boot random
  access key; a single offline HTML page (lock screen + chat)

Everything hermetic: a scripted FakeRouter stands in for the LLM, the
server runs on an ephemeral port of 127.0.0.1, and auth is verified
end to end (no key / wrong key / right key).
"""
from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from tests.test_partner_runtime import FakeRouter, _make_context

from nomorals.api.console import WebConsole, console_page
from nomorals.social.chat.web import WebAdapter


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class WebAdapterTest(unittest.TestCase):
    def test_send_captures_and_take_drains(self) -> None:
        adapter = WebAdapter()
        chat = adapter.chat
        adapter.send(chat, "part one")
        adapter.send(chat, "part two")
        self.assertEqual(adapter.pending_parts(chat.key),
                         ["part one", "part two"])
        self.assertEqual(adapter.take_parts(chat.key),
                         ["part one", "part two"])
        self.assertEqual(adapter.take_parts(chat.key), [])
        self.assertIsNotNone(adapter.last_send_age(chat.key))

    def test_console_chat_key_is_operator(self) -> None:
        # PartnerRuntime._is_operator() treats ":console" chats as owner
        chat = WebAdapter().chat
        self.assertTrue(chat.key.endswith(":console"))
        self.assertEqual(chat.platform, "web")


class ConsoleServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _make_context()
        self.ctx.router = FakeRouter(replies=[
            "hey. i was just thinking about you, actually",
        ])
        self.console = WebConsole(self.ctx, token="test-key-123")
        self.port = _free_port()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port),
                                         self.console.handler_class())
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.console.stop()
        self.ctx.close()
        self.tmp.cleanup()

    def _get(self, path: str, key: str | None = None) -> tuple[int, bytes]:
        req = urllib.request.Request(self.base + path)
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def _post(self, path: str, payload: dict, key: str | None = None,
              timeout: float = 60.0) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")

    def test_health_is_open_and_page_serves(self) -> None:
        status, body = self._get("/health")
        self.assertEqual(status, 200)
        self.assertIn(b"ok", body)
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"NM CONSOLE", body)
        self.assertIn(b"access key", body)
        # the page itself must not leak the key
        self.assertNotIn(b"test-key-123", body)

    def test_api_requires_the_key(self) -> None:
        status, _ = self._get("/api/status")          # no key
        self.assertEqual(status, 401)
        status, _ = self._get("/api/status", key="wrong")
        self.assertEqual(status, 401)
        status, body = self._get("/api/status", key="test-key-123")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        self.assertIn("mood", payload)

    def test_chat_round_trip_end_to_end(self) -> None:
        status, out = self._post(
            "/api/chat", {"text": "are you there?"}, key="test-key-123")
        self.assertEqual(status, 200)
        self.assertTrue(out["ok"])
        self.assertIn("thinking about you", out["reply"])
        self.assertTrue(out["parts"])

    def test_chat_rejects_empty_text(self) -> None:
        status, out = self._post("/api/chat", {"text": "  "},
                                 key="test-key-123")
        self.assertEqual(status, 400)

    def test_random_key_defaults(self) -> None:
        ctx2, tmp2 = _make_context()
        try:
            c1 = WebConsole(ctx2)
            self.assertGreaterEqual(len(c1.token), 16)
            ctx3, tmp3 = _make_context()
            try:
                c2 = WebConsole(ctx3)
                self.assertNotEqual(c1.token, c2.token)  # fresh per boot
            finally:
                c2.runtime.stop()
                ctx3.close()
                tmp3.cleanup()
            c1.runtime.stop()
        finally:
            ctx2.close()
            tmp2.cleanup()

    def test_pinned_token_is_used(self) -> None:
        self.assertEqual(self.console.token, "test-key-123")

    def test_status_reflects_provider_and_version(self) -> None:
        status, body = self._get("/api/status", key="test-key-123")
        payload = json.loads(body)
        self.assertEqual(payload["version"],
                         __import__("nomorals.version", fromlist=["x"]).__version__)
        self.assertIn("provider", payload)
        self.assertIn("uptime_seconds", payload)


class ConsolePageTest(unittest.TestCase):
    def test_page_is_self_contained(self) -> None:
        page = console_page()
        # offline-first: no external assets, no CDN, no fetch to the internet
        self.assertNotIn("https://", page)
        self.assertNotIn("http://", page)
        self.assertIn("sessionStorage", page)
        self.assertIn("Bearer", page)
        self.assertIn("#key=", page)  # auto-unlock from the printed URL

    def test_page_has_lock_and_chat(self) -> None:
        page = console_page()
        self.assertIn("unlock", page)
        self.assertIn("/api/chat", page)
        self.assertIn("/api/status", page)
        self.assertIn("typing", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
