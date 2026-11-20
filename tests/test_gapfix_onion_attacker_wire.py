"""Lane 55: wire in the owner's onion.py + attacker.py refactor.

- onion.py: v3 validation (real checksum), tor_status honesty, fetch
  through SOCKS5h (never direct), control-port auth refusal detection.
- attacker.py: the v2 refactor dropped run_attack/register (its own
  __all__ still promises them) — restored, and the brain tool registers.
- Chat surface: /tor status, /onion validate|fetch (owner-only by
  absence from PUBLIC_COMMANDS).
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import socket
import sys
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from nomorals.core.errors import ToolError
from nomorals.security import onion as onion_mod
from nomorals.security.onion import (
    TorControl,
    onion_fetch,
    tor_status,
    validate_onion,
)
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
)
from nomorals.tools import attacker as attacker_mod
from nomorals.tools.attacker import run_attack
from nomorals.tools.registry import ToolRegistry


def _make_v3() -> str:
    """A checksum-valid v3 .onion, built from a fixed pubkey."""
    pubkey = bytes(range(32))
    checksum = hashlib.sha3_256(
        b".onion checksum" + pubkey + b"\x03").digest()[:2]
    return base64.b32encode(pubkey + checksum + b"\x03").decode().lower()


class OnionValidationTests(unittest.TestCase):
    def test_real_v3_validates(self):
        self.assertTrue(validate_onion(_make_v3()))

    def test_uppercase_and_trailing_dot_normalised(self):
        self.assertTrue(validate_onion(_make_v3().upper() + "."))

    def test_flipped_character_fails(self):
        addr = _make_v3()
        flipped = addr[:-1] + ("a" if addr[-1] != "a" else "b")
        self.assertFalse(validate_onion(flipped))

    def test_v2_short_and_garbage_fail(self):
        self.assertFalse(validate_onion("abcdefghijklmnop.onion"))
        self.assertFalse(validate_onion("not-an-address"))
        self.assertFalse(validate_onion(""))
        self.assertFalse(validate_onion("x" * 56 + ".onion"))


class TorStatusTests(unittest.TestCase):
    def test_no_daemon_is_honest(self):
        with mock.patch("socket.create_connection",
                        side_effect=OSError("no listener")):
            st = tor_status()
        self.assertFalse(st["installed"])
        self.assertIsNone(st["socks"])
        self.assertFalse(st["rotation"])
        self.assertIn("no Tor", st["note"])

    def test_malformed_env_is_named(self):
        with mock.patch.dict("os.environ", {"NM_TOR_SOCKS": "host:notaport"}):
            st = tor_status()
        self.assertFalse(st["installed"])
        self.assertIn("NM_TOR_SOCKS", st["note"])

    def test_control_detect_none_without_port(self):
        with mock.patch("socket.create_connection",
                        side_effect=OSError("no listener")):
            self.assertIsNone(TorControl.detect())


class _ControlHandler:
    """Tiny fake Tor control port: one canned reply per command."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.seen: list[str] = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        with contextlib.suppress(OSError):
            self.server.close()
        return False

    def _serve(self):
        try:
            conn, _ = self.server.accept()
        except OSError:
            return
        with conn:
            f = conn.makefile("rwb", buffering=0)
            while True:
                line = f.readline().decode("utf-8", "replace").strip()
                if not line:
                    return
                self.seen.append(line)
                reply = self.replies.get(
                    line.split()[0], "250 OK\r\n")
                f.write(reply.encode())


class TorControlTests(unittest.TestCase):
    def test_auth_refusal_is_an_error_not_a_result(self):
        with _ControlHandler({"AUTHENTICATE": "515 Authentication failed\r\n"}) as h:
            ctl = TorControl("127.0.0.1", h.port)
            with self.assertRaises(ToolError) as cm:
                ctl.newnym()
        # The 515 is detected (not returned as a result) and translated
        # into the real cause: this control port wants credentials.
        self.assertIn("cookie or password", str(cm.exception))

    def test_good_control_newnym(self):
        with _ControlHandler({"AUTHENTICATE": "250 OK\r\n",
                              "SIGNAL": "250 OK\r\n"}) as h:
            ctl = TorControl("127.0.0.1", h.port)
            ctl.newnym()  # no raise
        self.assertIn("AUTHENTICATE", h.seen)
        self.assertIn("SIGNAL NEWNYM", h.seen)

    def test_cookie_auth_sent(self):
        with _ControlHandler({"AUTHENTICATE": "250 OK\r\n",
                              "SIGNAL": "250 OK\r\n"}) as h:
            ctl = TorControl("127.0.0.1", h.port, cookie="deadbeef")
            ctl.newnym()
        self.assertIn("AUTHENTICATE deadbeef", h.seen)


class OnionFetchTests(unittest.TestCase):
    def test_no_tor_refuses_and_never_fetches_direct(self):
        with (mock.patch.object(onion_mod, "_socks_addr", return_value=None),
              mock.patch("urllib.request.build_opener",
                         side_effect=AssertionError("direct fetch!")),
              self.assertRaises(ToolError) as cm):
            onion_fetch("https://example.com")
        self.assertIn("Tor is not running", str(cm.exception))

    def test_invalid_onion_refuses_before_any_network(self):
        with (mock.patch("urllib.request.build_opener",
                         side_effect=AssertionError("network!")),
              self.assertRaises(ToolError) as cm):
            onion_fetch("http://abcdefghijklmnop.onion/")
        self.assertIn("invalid v3 onion", str(cm.exception))

    def test_missing_pysocks_is_named_not_faked(self):
        fake_socks = ("127.0.0.1", 9999)
        with (mock.patch.object(onion_mod, "_socks_addr",
                                return_value=fake_socks),
              mock.patch.dict(sys.modules,
                              {"socks": None, "sockshandler": None}),
              self.assertRaises(ToolError) as cm):
            onion_fetch("https://example.com")
        self.assertIn("PySocks", str(cm.exception))

    def test_rotation_degrades_without_control_port(self):
        # No control port: rotation is skipped, the fetch then fails
        # on the missing PySocks — the error is the PySocks one,
        # never a control error.
        fake_socks = ("127.0.0.1", 9999)
        with (mock.patch.object(onion_mod, "_socks_addr",
                                return_value=fake_socks),
              mock.patch.object(TorControl, "detect", return_value=None),
              mock.patch.dict(sys.modules,
                              {"socks": None, "sockshandler": None}),
              self.assertRaises(ToolError) as cm):
            onion_fetch("https://example.com", rotate=True)
        self.assertIn("PySocks", str(cm.exception))


class _LoginHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        ln = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(ln).decode("utf-8", "replace")
        params = urllib.parse.parse_qs(body)
        login = params.get("username", [""])[0]
        password = params.get("password", [""])[0]
        if login == "alice" and password == "s3cret":
            payload = b"welcome to your dashboard"
            self.send_response(302)
            self.send_header("Location", "/dashboard")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            payload = b"Invalid credentials, try again."
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, *args):
        pass


class AttackerBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _LoginHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_baseline_differential_finds_only_the_real_pair(self):
        # The secprobe wire-up made a scope mandatory: an attacker run
        # needs a Governor built from a scope file — no unscoped brute.
        # The loopback scope below allows only this test's own server.
        from nomorals.security.scope import Governor, Scope
        governor = Governor(Scope(include=["127.0.0.1"]),
                            max_total=100, max_per_host=1000.0)
        res = run_attack(
            "127.0.0.1", port=self.port, protocol="http_form",
            usernames="alice", passwords="wrong1,s3cret",
            fail_string="Invalid credentials", path="/login",
            workers=1, rate=100.0, delay=0.0, jitter=0.0,
            quiet=True, baseline_tries=2, confirm_tries=1, timeout=5.0,
            governor=governor)
        self.assertEqual(res["attempts"], 2)
        self.assertEqual([f["login"] for f in res["found"]], ["alice"])
        self.assertEqual([f["password"] for f in res["found"]], ["s3cret"])

    def test_constructor_validation(self):
        with self.assertRaises(ToolError):
            attacker_mod.Attacker("127.0.0.1", usernames=[])
        with self.assertRaises(ToolError):
            attacker_mod.Attacker("127.0.0.1", protocol="smb",
                                  usernames=["a"], passwords=["b"])


class _Ctx:
    """Bare tool context for the registry (mirrors the real shape)."""


class AttackerRegisterTests(unittest.TestCase):
    def test_register_restored(self):
        self.assertTrue(hasattr(attacker_mod, "register"))
        self.assertTrue(hasattr(attacker_mod, "run_attack"))

    def test_brain_tool_registers(self):
        reg = ToolRegistry(_Ctx())
        attacker_mod.register(reg)
        self.assertIn("attacker", reg.names())

    def test_builtins_no_longer_fail_attacker_or_security(self):
        reg = ToolRegistry(_Ctx())
        reg.register_builtins()
        failed = [f["module"] for f in reg.failed_modules()]
        self.assertNotIn("attacker", failed)
        self.assertNotIn("security", failed)
        for name in ("attacker", "tor_status", "onion_validate",
                     "onion_fetch"):
            self.assertIn(name, reg.names())


class ChatSurfaceTests(unittest.TestCase):
    def test_tor_and_onion_registered_owner_only(self):
        for kind in ("tor", "onion"):
            self.assertIn(kind, CONTROL_COMMANDS)
            page = COMMAND_DETAILS[kind]
            for field in ("what", "usage", "example", "related"):
                self.assertTrue(str(page.get(field, "")).strip(),
                                f"/{kind} DETAILS.{field} is empty")
            self.assertTrue(LIST_ONELINERS.get(kind, "").strip())
            self.assertNotIn(kind, PUBLIC_COMMANDS)
        grouped = [k for _, kinds in LIST_GROUPS for k in kinds]
        for kind in ("tor", "onion"):
            self.assertEqual(grouped.count(kind), 1,
                             f"/{kind} must sit in exactly one /list group")


if __name__ == "__main__":
    unittest.main()
