"""MWG legacy-scan audit pins (2026-11-14): security headers on every
Devon web surface, the three.js JSON-escape fix, and the pairing
autofill finding.

The scan itself found exactly one deterministic finding
(``autofill-sign-in-form`` on ``web/pairing.php:358`` — see
``~/workspace/mwguidance-verify/SCAN_FINDINGS.md``); the hand audit
found the rest. These tests pin the fixes live where a server can be
started in-process, and pin the policy text where it cannot.

Servers covered live: the Node motion host, the hub, the API, the
takeover page. The PHP pairing page's live header test lives in
``tests/test_pairing_php_page.py`` (it owns the PHP harness); the
static pins for its forms live here.
"""
from __future__ import annotations

import http.client
import json
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import nomorals.media.motion_rungs as mr
from nomorals.api.server import CSP as API_CSP
from nomorals.api.server import APIServer
from nomorals.api.server import _make_handler as _api_handler
from nomorals.hub.server import CSP as HUB_CSP
from nomorals.hub.server import serve as hub_serve
from nomorals.storage.db import Database

REPO_ROOT = Path(__file__).resolve().parent.parent
MOTION_SERVER = REPO_ROOT / "bridge" / "motion-server.mjs"
PAIRING_PHP = REPO_ROOT / "web" / "pairing.php"
API_SERVER_PY = REPO_ROOT / "nomorals" / "api" / "server.py"

#: Every Devon server must send these on every response.
REQUIRED_HEADERS = (
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "Referrer-Policy",
    "X-Frame-Options",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(port: int, path: str, timeout: float = 10.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    raw = resp.read()
    headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, headers, raw


def _wait_up(port: int, path: str = "/health", timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _, _ = _get(port, path, timeout=2.0)
            if status in (200, 401, 404):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server on {port} never came up")


# ── the motion host (Node, live) ─────────────────────────────────────

@pytest.mark.skipif(not shutil.which("node"), reason="node not on PATH")
class TestMotionServerHeaders:
    def setup_method(self):
        self.port = _free_port()
        env = {"PATH": "/usr/bin:/bin",
               "NM_MOTION_PORT": str(self.port)}
        self.proc = subprocess.Popen(
            ["node", str(MOTION_SERVER)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_up(self.port)

    def teardown_method(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def test_every_route_sets_the_security_headers(self):
        for path in ("/", "/health", "/lottie", "/gsap", "/three"):
            status, headers, _ = _get(self.port, path)
            assert status == 200, path
            for name in REQUIRED_HEADERS:
                assert name.lower() in headers, (path, name)
            csp = headers["content-security-policy"]
            assert "frame-ancestors 'none'" in csp
            assert "cdnjs.cloudflare.com" in csp
            assert "cdn.jsdelivr.net" in csp
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["referrer-policy"] == "no-referrer"
            assert headers["x-frame-options"] == "DENY"

    def test_malicious_scene_spec_cannot_break_out_of_the_script(self):
        # A tweak string carrying </script> must arrive escaped, never
        # as a live closing tag inside the embedded spec JSON.
        evil = {"tweak": {"x": "</script><script>alert(1)</script>"}}
        query = urllib.parse.quote(json.dumps(evil))
        status, _, raw = _get(self.port, f"/three?scene={query}")
        assert status == 200
        body = raw.decode("utf-8")
        assert "</script><script>alert(1)</script>" not in body
        assert "<\\/script><script>alert(1)<\\/script>" in body


# ── the motion pages (Python generator) ──────────────────────────────

class TestMotionRungsPages:
    def test_three_scene_page_escapes_a_malicious_tweak(self):
        spec = mr.default_scene_spec()
        spec["tweak"] = {"x": "</script><script>alert(1)</script>"}
        page = mr.three_scene_page(spec)
        assert "</script><script>alert(1)</script>" not in page
        assert "<\\/script><script>alert(1)<\\/script>" in page

    def test_lottie_page_escapes_a_malicious_animation(self):
        anim = {"v": "5.7.4", "nm": "</script><script>alert(1)</script>",
                "layers": []}
        page = mr.lottie_player_page(anim)
        blob = re.search(
            r'<script type="application/json" id="lottie-data">(.*?)</script>',
            page, re.DOTALL)
        assert blob is not None
        assert "</script>" not in blob.group(1)

    def test_every_page_carries_the_meta_csp(self):
        pages = [
            mr.lottie_player_page({"v": "5.7.4", "layers": []}),
            mr.gsap_demo_page(),
            mr.three_scene_page(mr.default_scene_spec()),
        ]
        for page in pages:
            assert '<meta http-equiv="Content-Security-Policy"' in page
            assert mr.MOTION_CSP in page

    def test_python_csp_matches_the_node_servers_csp(self):
        # One policy, two copies (the served header in
        # bridge/motion-server.mjs, the <meta> tag in the static
        # pages) — drift between them is a test failure.
        text = MOTION_SERVER.read_text(encoding="utf-8")
        m = re.search(r'const CSP = "([^"]+)"', text)
        assert m, "node CSP const not found"
        assert m.group(1) == mr.MOTION_CSP


# ── the hub (live) ───────────────────────────────────────────────────

class TestHubHeaders:
    def test_index_sets_the_security_headers(self, tmp_path):
        db = Database(str(tmp_path / "hub.db"))
        db.migrate()
        server = hub_serve(db, port=0, token="test-token")
        try:
            status, headers, _ = _get(server.port, "/")
            assert status == 200
            for name in REQUIRED_HEADERS:
                assert name.lower() in headers, name
            assert headers["content-security-policy"] == HUB_CSP
            assert HUB_CSP == "default-src 'none'; frame-ancestors 'none'"
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["x-frame-options"] == "DENY"
        finally:
            server.stop()


# ── the API server (live + source pin) ───────────────────────────────

def _api_context():
    return SimpleNamespace(
        db=None, tools=None,
        settings=SimpleNamespace(
            api=SimpleNamespace(token="test-token", max_body_mb=0)))


class TestApiHeaders:
    def test_unknown_path_still_sets_the_security_headers(self):
        api = APIServer(_api_context(), token="test-token")
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), _api_handler(api))
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            status, headers, _ = _get(port, "/definitely-not-a-route")
            assert status in (401, 404), status
            for name in REQUIRED_HEADERS:
                assert name.lower() in headers, name
            assert "script-src 'none'" in headers["content-security-policy"]
            assert "frame-ancestors 'none'" in headers["content-security-policy"]
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["x-frame-options"] == "DENY"
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_csp_is_set_at_all_three_send_sites(self):
        # _respond (JSON + errors), the site page, the pairing page —
        # one send_header per site, or a later edit silently drops a
        # surface.
        text = API_SERVER_PY.read_text(encoding="utf-8")
        count = text.count('send_header("Content-Security-Policy", CSP)')
        assert count == 3, f"expected 3 CSP send sites, found {count}"
        assert "script-src 'none'" in API_CSP


# ── the takeover page (live) ─────────────────────────────────────────

class TestTakeoverHeaders:
    def test_takeover_page_sets_the_security_headers(self):
        from nomorals.browser.takeover import (
            TakeoverConfig,
            TakeoverManager,
        )

        mgr = TakeoverManager(TakeoverConfig())
        grant = mgr.request_takeover(
            session_id="test-session", url="https://example.com",
            task_label="test", signal_kind="test",
            attempts=[], resume_state={}, tab=None)
        server = mgr.ensure_server()
        port = server.server_address[1]
        try:
            status, headers, raw = _get(port, f"/takeover/{grant.token}")
            assert status == 200
            for name in REQUIRED_HEADERS:
                assert name.lower() in headers, name
            assert "frame-ancestors 'none'" in headers["content-security-policy"]
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["x-frame-options"] == "DENY"
            body = raw.decode("utf-8")
            assert '<html lang="en">' in body
            assert 'aria-label="Text to send to the browser"' in body
        finally:
            mgr.shutdown()


# ── the PHP pairing page (static pins; live header test is in
#    tests/test_pairing_php_page.py, which owns the PHP harness) ──────

class TestPairingPhpPage:
    def test_scan_finding_is_fixed_and_forms_are_labelled(self):
        text = PAIRING_PHP.read_text(encoding="utf-8")
        # The scan finding itself: the 2FA password input.
        assert 'autocomplete="current-password"' in text
        # Same-family companions: the one-time login code, and the
        # login page-token input (was autocomplete="off").
        assert 'autocomplete="one-time-code"' in text
        assert 'autocomplete="off"' not in text
        # Labels, not placeholders alone, on the credential inputs.
        assert "<label>login code " in text
        assert "<label>2FA password " in text
        # Both pages declare their language.
        assert text.count('<html lang="en">') == 2
        # The security headers are sent once, up top, for every response.
        assert "Content-Security-Policy:" in text
        assert "X-Content-Type-Options: nosniff" in text
        assert "X-Frame-Options: DENY" in text


# ── the Python-rendered pages (static pins) ──────────────────────────

class TestPythonPageLang:
    def test_site_and_pairing_pages_declare_their_language(self):
        for rel in ("nomorals/social/site.py",
                    "nomorals/social/pairing_page.py"):
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            assert "<html>" not in text, rel
            assert "<html lang='en'>" in text, rel
