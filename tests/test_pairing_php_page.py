"""The PHP pairing page, served live (polyglot rung 1).

The page is exercised over real HTTP against PHP's built-in
server, with a stub Devon API behind it that speaks the real
contract shapes (and demands the owner bearer token, like the
real gate).  The page must: refuse the unauthenticated, log in
with its page token only, render the pool the API reports —
counts, accounts, the live pairing code, real toggle states —
and round-trip an action into the contract, CSRF and all.

Skips when no PHP runtime can be resolved on this machine
(Devon's own ladder, managed copy included).
"""
from __future__ import annotations

import http.client
import json
import re
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from nomorals.social.pairing_web import PAGE_FILE, resolve_php

OWNER_AUTH = "pairing-page-" + "owner-cred"
PAGE_AUTH = "pairing-page-" + "gate-cred"


def _php() -> str | None:
    home_ws = Path.home() / "workspace"
    return resolve_php(workspace=home_ws) or resolve_php()


PHP = _php()
pytestmark = pytest.mark.skipif(
    PHP is None, reason="no PHP runtime resolvable on this machine")


def canned_state() -> dict:
    return {
        "ok": True,
        "counts": {"registered": 1, "connected": 1, "dropped": 0},
        "accounts": [{
            "id": "whatsapp:2348012345678", "platform": "whatsapp",
            "identity": "2348012345678", "label": "main line",
            "status": "pairing", "primary": True, "port": 8787,
            "toggles": {"status_saving": False, "view_once": False,
                        "antidelete": False, "always_online": True,
                        "status_viewing": False, "status_react": False,
                        "anticall": False, "antitag": False,
                        "antiedit": False},
            "last_error": "", "process": "running",
            "pairing_code": {"code": "WXYZ-2345", "age_s": 12.0},
            "created_at": 1.0, "last_seen": 2.0,
        }],
        "owner_stopped": {"whatsapp": False, "telegram": False},
        "toggles": ["status_saving", "view_once"],
        "rungs": [],
        "generated_at": 3.0,
    }


class StubAPI:
    """The Devon contract, canned: owner-token gate, real shapes,
    and a toggle that actually flips so re-renders tell the truth."""

    def __init__(self):
        self.state = canned_state()
        self.actions: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _auth_ok(self):
                return self.headers.get("Authorization") == \
                    f"Bearer {OWNER_AUTH}"

            def _send(self, status, payload):
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path != "/pair/state":
                    return self._send(404, {"error": "nope"})
                if not self._auth_ok():
                    return self._send(403, {"error": "capability denied"})
                self._send(200, outer.state)

            def do_POST(self):
                if self.path != "/pair/action":
                    return self._send(404, {"error": "nope"})
                if not self._auth_ok():
                    return self._send(403, {"error": "capability denied"})
                body = json.loads(
                    self.rfile.read(
                        int(self.headers.get("Content-Length", 0)))
                    or b"{}")
                outer.actions.append(body)
                if body.get("action") == "toggle":
                    acct = outer.state["accounts"][0]
                    acct["toggles"][body["name"]] = bool(body["on"])
                    note = f"{body['name']} on for {body['id']}"
                else:
                    note = "done"
                self._send(200, {"ok": True, "action": body.get("action"),
                                 "note": note, "state": outer.state})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class PageServer:
    def __init__(self, api_port: int, tmp_path: Path, *,
                 page_token: str = PAGE_AUTH,
                 api_token: str = OWNER_AUTH):
        self.port = free_port()
        self.cookie = ""  # the client's session cookie jar
        env = {
            "PATH": "/usr/bin:/bin",
            "DEVON_API_BASE": f"http://127.0.0.1:{api_port}",
            "DEVON_API_TOKEN": api_token,
            "DEVON_PAIR_PAGE_TOKEN": page_token,
        }
        sess = tmp_path / "php_sessions"
        sess.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [PHP, "-d", f"session.save_path={sess}",
             "-S", f"127.0.0.1:{self.port}", str(PAGE_FILE)],
            env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port),
                                              timeout=1.0):
                    return
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        "php server exited early") from None
                time.sleep(0.05)
        raise RuntimeError("php server never listened")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def request(self, method, path, *, cookie=None, form=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port,
                                          timeout=10)
        headers = {}
        body = None
        # A browser carries its session cookie on every request —
        # the CSRF token lives in that session, so the jar is the
        # difference between a working login and a stale form.
        jar = self.cookie if cookie is None else cookie
        if jar:
            headers["Cookie"] = jar
        if form is not None:
            body = "&".join(
                f"{k}={_urlencode(str(v))}" for k, v in form.items())
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
        out = (resp.status, dict(resp.getheaders()), raw)
        conn.close()
        fresh = _session_cookie(out[1])
        if fresh:
            self.cookie = fresh
        return out


def _urlencode(value: str) -> str:
    from urllib.parse import quote_plus
    return quote_plus(value)


def _session_cookie(headers: dict) -> str:
    raw = headers.get("Set-Cookie", "")
    return raw.split(";")[0] if raw else ""


def _csrf(html: str) -> str:
    m = re.search(r'name="csrf" value="([0-9a-f]+)"', html)
    assert m, "no csrf field rendered"
    return m.group(1)


def _login(page: PageServer) -> str:
    status, headers, html = page.request("GET", "/login")
    assert status == 200
    csrf = _csrf(html)
    status, headers, _ = page.request(
        "POST", "/login", form={"csrf": csrf, "token": PAGE_AUTH})
    assert status == 302
    cookie = _session_cookie(headers)
    assert cookie.startswith("PHPSESSID=")
    return cookie


def test_page_refuses_the_unauthenticated(tmp_path):
    api = StubAPI()
    page = PageServer(api.port, tmp_path)
    try:
        status, headers, _ = page.request("GET", "/")
        assert status == 302 and "/login" in headers["Location"]
        status, _, body = page.request("GET", "/state.json")
        assert status == 401
        assert json.loads(body)["ok"] is False
        # A wrong page token does not log in.
        status, headers, html = page.request("GET", "/login")
        csrf = _csrf(html)
        status, _, html = page.request(
            "POST", "/login", form={"csrf": csrf, "token": "wrong"})
        assert status == 200
        assert "not the page token" in html
    finally:
        page.close()
        api.close()


def test_page_renders_the_pool_and_round_trips_a_toggle(tmp_path):
    api = StubAPI()
    page = PageServer(api.port, tmp_path)
    try:
        cookie = _login(page)
        status, _, html = page.request("GET", "/", cookie=cookie)
        assert status == 200
        assert "Registered: <b>1</b>" in html
        assert "whatsapp:2348012345678" in html
        assert "main line" in html
        # The live pairing code is displayed, big and verbatim.
        assert "WXYZ-2345" in html
        # Real toggle states render as chips.
        assert "status_saving: off" in html
        assert "always_online: on" in html

        status, _, body = page.request("GET", "/state.json",
                                       cookie=cookie)
        assert status == 200
        assert json.loads(body)["counts"]["connected"] == 1

        # Flip status_saving on through the page's own form: the
        # stub contract must receive the real action payload…
        csrf = _csrf(html)
        status, headers, _ = page.request(
            "POST", "/action", cookie=cookie,
            form={"csrf": csrf, "action": "toggle",
                  "id": "whatsapp:2348012345678",
                  "name": "status_saving", "on": "1"})
        assert status == 302
        assert api.actions[-1] == {
            "action": "toggle", "id": "whatsapp:2348012345678",
            "name": "status_saving", "on": True}
        # …the redirect carries the contract's own note as the
        # flash, and the re-render tells the new truth.
        status, _, html = page.request("GET", headers["Location"],
                                       cookie=cookie)
        assert status == 200
        assert "status_saving on for whatsapp:2348012345678" in html
        status, _, html = page.request("GET", "/", cookie=cookie)
        assert "status_saving: on" in html

        # A POST without the CSRF token never reaches the API.
        before = len(api.actions)
        status, headers, _ = page.request(
            "POST", "/action", cookie=cookie,
            form={"action": "toggle", "id": "whatsapp:2348012345678",
                  "name": "view_once", "on": "1"})
        assert status == 302
        assert len(api.actions) == before
    finally:
        page.close()
        api.close()


def test_page_wears_the_absorbed_presentation(tmp_path):
    """Design absorption (Uiverse kit + Magic UI recipes, vanilla):
    glass tiles, state-wired loaders, switch toggles on the same
    form, the border beam on the primary card, ticker-hooked
    counts and the flash as a toast — all in the served HTML."""
    api = StubAPI()
    page = PageServer(api.port, tmp_path)
    try:
        cookie = _login(page)
        status, _, html = page.request("GET", "/", cookie=cookie)
        assert status == 200
        # Glass account tiles with a blur-fade stagger index.
        assert 'class="card" data-pair-state="waiting"' in html
        assert 'style="--i:0"' in html
        assert "backdrop-filter:blur(10px)" in html
        # The canned account is mid-pairing with a live code: the
        # waiting loader rides it, worded for the phone step.
        assert 'class="loader"' in html
        assert "waiting for this code to be typed on the phone" in html
        # Toggles are real switches (role=switch, state as
        # aria-checked) submitting the page's existing toggle form.
        assert 'class="switch" role="switch" aria-checked="true"' in html
        assert 'class="switch" role="switch" aria-checked="false"' in html
        # The border beam accents the one primary card only.
        assert '<fieldset class="beam">' in html
        assert "@keyframes beamspin" in html
        # Counts carry the ticker's data hooks; values unchanged.
        assert 'data-registered="1"' in html
        assert 'data-connected="1"' in html
        assert 'data-dropped="0"' in html

        # A landed action's flash renders as a toast.
        csrf = _csrf(html)
        status, headers, _ = page.request(
            "POST", "/action", cookie=cookie,
            form={"csrf": csrf, "action": "toggle",
                  "id": "whatsapp:2348012345678",
                  "name": "view_once", "on": "1"})
        assert status == 302
        status, _, html = page.request("GET", headers["Location"],
                                       cookie=cookie)
        assert status == 200
        assert 'class="toast ok"' in html
        assert "view_once on for whatsapp:2348012345678" in html
    finally:
        page.close()
        api.close()


def test_page_without_a_gate_token_refuses_everything(tmp_path):
    api = StubAPI()
    page = PageServer(api.port, tmp_path, page_token="")
    try:
        status, _, body = page.request("GET", "/")
        assert status == 503
        assert "not armed" in body
    finally:
        page.close()
        api.close()


def test_page_reports_an_unreachable_api_honestly(tmp_path):
    page = PageServer(free_port(), tmp_path)  # nothing listens there
    try:
        cookie = _login(page)
        status, _, html = page.request("GET", "/", cookie=cookie)
        assert status == 200
        assert "not reachable" in html
    finally:
        page.close()
