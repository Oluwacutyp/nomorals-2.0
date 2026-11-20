"""Auth-plugin wiring: HttpClient cookie/UA surface + the owner's pushes.

The owner's auth plugins (gmail et al.) call ``client.cookie_dict()``,
``client.user_agent``, ``client.post(url, data=...)`` and
``client.set_cookie(...)`` — none of which existed on HttpClient — and
``session_forge.py`` / ``payloads.py`` imported ``..core`` from inside
``plugins/`` (ModuleNotFoundError), while the plugin registry lived in a
never-imported ``_init_.py``. This file proves the wiring end to end
against a local server: cookies ride on requests, Set-Cookie is stored
(including on an error response), post() covers the plugins' shapes,
all 12 plugins register, the forge stores + verifies a session, and
the registry counts 774 tools. 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh.
"""
from __future__ import annotations

import json
import stat
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

from nomorals.core.errors import ProviderError
from nomorals.core.http import HttpClient
from nomorals.security import payloads as _payloads
from nomorals.security import session_forge as _forge
from nomorals.security.plugins import PLUGINS
from nomorals.security.plugins.interface import (
    AuthResult,
    LoginResult,
    SessionData,
    WebPlugin,
)
from nomorals.tools.registry import ToolRegistry


def _make_server() -> tuple[ThreadingHTTPServer, dict]:
    state: dict = {"last_headers": {}, "last_body": b""}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D102 - silence test server
            pass

        def _cookie_ok(self) -> bool:
            return "SID=test123" in (self.headers.get("Cookie") or "")

        def do_GET(self):
            state["last_headers"] = dict(self.headers.items())
            if self.path == "/login":
                self.send_response(200)
                self.send_header("Set-Cookie", "SID=test123; Path=/; HttpOnly")
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html>login page</html>")
            elif self.path == "/home":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"Welcome back, user" if self._cookie_ok()
                    else b"logged out")
            elif self.path == "/challenge":
                self.send_response(401)
                self.send_header("Set-Cookie", "CHAL=xyz; Path=/")
                self.end_headers()
                self.wfile.write(b"denied")
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            state["last_body"] = self.rfile.read(length) if length else b""
            state["last_headers"] = dict(self.headers.items())
            if self.path == "/login":
                self.send_response(200)
                self.send_header("Set-Cookie", "SID=test123; Path=/")
                self.end_headers()
                self.wfile.write(b"ok")
            elif self.path == "/echo":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(state["last_body"])
            else:
                self.send_response(404)
                self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    return server, state


class _StubPlugin(WebPlugin):
    name, domain = "stub", "127.0.0.1"
    auth_cookies = ("SID",)

    def __init__(self, base_url: str):
        self.base_url = base_url

    def login(self, username, password, client):
        client.get(f"{self.base_url}/login")
        return LoginResult(
            AuthResult.SUCCESS, username, password, "stub login", 0.0,
            session=SessionData(cookies=dict(client.cookie_dict()),
                                user_agent=client.user_agent))


class AuthWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.state = _make_server()
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _client(self) -> HttpClient:
        return HttpClient(allow_private_ips=True)

    # -- HttpClient surface ------------------------------------------
    def test_user_agent_property_round_trips(self):
        client = self._client()
        self.assertTrue(client.user_agent)
        client.user_agent = "TestAgent/1.0"
        self.assertEqual(client.user_agent, "TestAgent/1.0")
        self.assertEqual(client.headers.get("User-Agent"), "TestAgent/1.0")
        # an explicit headers UA wins, as the old request() merge did
        other = HttpClient(allow_private_ips=True,
                           headers={"User-Agent": "FromHeaders/2.0"})
        self.assertEqual(other.user_agent, "FromHeaders/2.0")

    def test_set_cookie_and_cookie_dict(self):
        client = self._client()
        self.assertEqual(client.cookie_dict(), [])
        client.set_cookie("SID", "abc123")
        self.assertEqual(dict(client.cookie_dict()), {"SID": "abc123"})

    def test_cookie_header_sent_and_set_cookie_stored(self):
        client = self._client()
        page = client.get(f"{self.base}/login")
        self.assertTrue(page.ok)
        self.assertEqual(dict(client.cookie_dict()), {"SID": "test123"})
        home = client.get(f"{self.base}/home")
        self.assertIn("Welcome back", home.text or "")
        # a fresh client without the jar is logged out
        bare = self._client()
        self.assertIn("logged out", bare.get(f"{self.base}/home").text or "")

    def test_set_cookie_on_error_response_is_kept(self):
        client = self._client()
        with self.assertRaises(ProviderError):
            client.get(f"{self.base}/challenge")
        self.assertEqual(dict(client.cookie_dict()), {"CHAL": "xyz"})

    def test_post_form_mapping_and_str_body(self):
        client = self._client()
        resp = client.post(f"{self.base}/echo",
                           data={"Email": "u@example.com", "Passwd": "p"})
        self.assertTrue(resp.ok)
        self.assertEqual(parse_qs(resp.text or ""),
                         {"Email": ["u@example.com"], "Passwd": ["p"]})
        self.assertIn("application/x-www-form-urlencoded",
                      self.state["last_headers"].get("Content-Type", ""))
        resp = client.post(f"{self.base}/echo", data='{"a": 1}',
                           headers={"Content-Type": "application/json"})
        self.assertEqual(resp.text, '{"a": 1}')

    # -- plugin package + forge --------------------------------------
    def test_plugins_registry_has_twelve(self):
        self.assertEqual(len(PLUGINS), 12)
        for name, cls in PLUGINS.items():
            self.assertTrue(issubclass(cls, WebPlugin), name)
            self.assertEqual(cls.name, name)
            self.assertTrue(cls.domain)

    def test_session_forge_and_payloads_import(self):
        self.assertTrue(hasattr(_forge, "forge"))
        self.assertTrue(hasattr(_payloads, "png_bomb"))

    def test_client_from_restores_headers_ua_and_cookies(self):
        stored = {"session": {"cookies": {"SID": "s1"},
                              "headers": {"X-Test": "1"},
                              "user_agent": "Custom/9"}}
        client = _forge._client_from(stored, 10.0, None, True)
        self.assertEqual(client.headers.get("X-Test"), "1")
        self.assertEqual(client.user_agent, "Custom/9")
        self.assertEqual(dict(client.cookie_dict()), {"SID": "s1"})

    def test_forge_stores_and_verifies_a_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            result = _forge.forge(home, _StubPlugin(self.base), "u", "p",
                                  retries=0, escalate=False,
                                  allow_private_ips=True)
            self.assertTrue(result.ok)
            stored_path = home / "data" / "sessions" / "127.0.0.1" / "u.json"
            self.assertTrue(stored_path.is_file())
            self.assertEqual(stat.S_IMODE(stored_path.stat().st_mode), 0o600)
            stored = json.loads(stored_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["session"]["cookies"],
                             {"SID": "test123"})
            loaded = _forge.load(_forge._root(home), "127.0.0.1", "u")
            self.assertIsNotNone(loaded)
            verdict = _forge.verify_session(
                _forge._root(home), "127.0.0.1", "u", f"{self.base}/home",
                marker="Welcome back", allow_private_ips=True)
            self.assertEqual(verdict["state"], "valid")
            self.assertTrue(_forge.drop(_forge._root(home), "127.0.0.1", "u"))
            self.assertEqual(
                _forge.verify_session(_forge._root(home), "127.0.0.1", "u",
                                      f"{self.base}/home",
                                      allow_private_ips=True)["state"],
                "missing")

    def test_pdf_bomb_has_shared_content_stream(self):
        data = _payloads.pdf_bomb(pages=3)
        self.assertTrue(data.startswith(b"%PDF-1.4"))
        # 6 objects: catalog, pages, content stream, 3 pages
        self.assertIn(b"3 0 obj\n<< /Length ", data)
        self.assertIn(b"stream\nBT /F1 12 Tf 20 780 Td (.) Tj ET\nendstream",
                      data)
        self.assertIn(b"4 0 obj\n<< /Type /Page", data)
        self.assertNotIn(b"7 0 obj", data)
        self.assertIn(b"/Kids [4 0 R 5 0 R 6 0 R]", data)
        self.assertIn(b"/Contents 3 0 R", data)

    # -- registry ------------------------------------------------------
    def test_tool_count_is_780_after_auth_wiring(self):
        # 748 after the auth-plugin wiring added 5 tools (session_forge,
        # session_verify, payload_png_bomb, payload_svg_bomb,
        # payload_keylogger) on top of the bughunter port (lane 121). 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after B.19 added scene_spawn; 776 after B.20 added cloudcam; 777 after B.22 added show; 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh.
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertEqual(len(registry._tools), 780)
        for name in ("session_forge", "session_verify", "payload_png_bomb",
                     "payload_svg_bomb", "payload_keylogger"):
            self.assertIn(name, registry._tools)


if __name__ == "__main__":
    unittest.main()
