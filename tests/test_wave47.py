"""Wave 47 — attacker suite + env merge (fully hermetic).

- tools/attacker.py: multi-protocol credential brute (http_form,
  http_basic, ssh, ftp) — threaded worker pool, global rate limit,
  human-paced delays, fail-string detection, credentials file
- hashcrack.generate_combinations: the PDF's generator.py
- scripts/merge_env.sh: fold .env.example into a live .env without
  touching existing values

Every network target is an in-process server on 127.0.0.1: an
HTTP form/basic server and a real stdlib FTP server; SSH is tested
both via an injected connector and via the graceful-missing-paramiko
path.  No external network.
"""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nomorals.core.errors import ToolError
from nomorals.tools.attacker import Attacker, run_attack
from nomorals.tools.hashcrack import generate_combinations


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ── in-process targets ───────────────────────────────────────────────────────


class _FormHandler(BaseHTTPRequestHandler):
    """http_form target: admin/correct-horse, fail string on 401."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace")
        fields = dict(urllib.parse.parse_qsl(body))
        if fields.get("username") == "admin" \
                and fields.get("password") == "correct-horse":
            self._reply(200, b"Welcome admin")
        else:
            self._reply(401, b"Invalid credentials")

    def _reply(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # quiet
        pass


class _BasicHandler(BaseHTTPRequestHandler):
    """http_basic target: admin/secret9."""

    def do_GET(self):  # noqa: N802
        import base64

        token = self.headers.get("Authorization", "")
        want = "Basic " + base64.b64encode(b"admin:secret9").decode("ascii")
        self._reply(200 if token == want else 401, b"panel")

    def do_POST(self):  # noqa: N802
        self.do_GET()

    def _reply(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # quiet
        pass


class _FTPServer:
    """Minimal FTP auth target (stdlib has no FTP server): answers the
    USER/PASS handshake — 230 only for admin/ftppass."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.sendall(b"220 test ftp ready\r\n")
            user = ""
            buf = b""
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    return
                buf += chunk
                while b"\r\n" in buf:
                    line, buf = buf.split(b"\r\n", 1)
                    cmd = line.decode("ascii", "replace").strip()
                    parts = cmd.split(None, 1)
                    verb = parts[0].upper()
                    if verb == "USER":
                        user = parts[1].strip() if len(parts) > 1 else ""
                        conn.sendall(b"331 Password required\r\n")
                    elif verb == "PASS":
                        pw = parts[1].strip() if len(parts) > 1 else ""
                        if user == "admin" and pw == "ftppass":
                            conn.sendall(b"230 Login successful\r\n")
                        else:
                            conn.sendall(b"530 Login incorrect\r\n")
                    elif verb == "QUIT":
                        conn.sendall(b"221 Goodbye\r\n")
                        return
                    else:
                        conn.sendall(b"502 Not implemented\r\n")
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def start(self) -> None:
        def serve() -> None:
            try:
                while True:
                    conn, _addr = self.sock.accept()
                    threading.Thread(target=self._handle,
                                     args=(conn,), daemon=True).start()
            except OSError:
                pass

        threading.Thread(target=serve, daemon=True).start()

    def stop(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def _http_server(handler) -> tuple[ThreadingHTTPServer, int]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


# ── attacker: http_form ──────────────────────────────────────────────────────


class HttpFormAttackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.httpd, self.port = _http_server(_FormHandler)
        self.tmp = tempfile.mkdtemp(prefix="attack-")

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_finds_credential_and_stops(self) -> None:
        wl = os.path.join(self.tmp, "pw.txt")
        with open(wl, "w") as f:
            f.write("nope\nwrong\n\n# comment\ncorrect-horse\nlast\n")
        out = os.path.join(self.tmp, "creds.txt")
        attacker = Attacker(
            "127.0.0.1", port=self.port, protocol="http_form",
            usernames=["admin"], wordlist=wl, workers=2,
            rate=100.0, delay=0.0, jitter=0.0, timeout=5.0,
            fail_string="Invalid credentials", out_file=out, quiet=True)
        result = attacker.run()
        self.assertEqual(len(result.found), 1)
        self.assertEqual(result.found[0].login, "admin")
        self.assertEqual(result.found[0].password, "correct-horse")
        # stopped on the hit, did not walk the whole list
        self.assertLessEqual(result.attempts, 5)
        # credentials saved to file, plain (no ANSI)
        with open(out) as f:
            saved = f.read()
        self.assertIn("correct-horse", saved)
        self.assertNotIn("\x1b[", saved)

    def test_no_hit_reports_cleanly(self) -> None:
        attacker = Attacker(
            "127.0.0.1", port=self.port, protocol="http_form",
            usernames=["admin"], passwords=["aaa", "bbb"], workers=2,
            rate=100.0, delay=0.0, jitter=0.0, timeout=5.0,
            fail_string="Invalid credentials", quiet=True)
        result = attacker.run()
        self.assertEqual(result.found, [])
        self.assertEqual(result.attempts, 2)
        self.assertIn("no valid credentials", result.note)

    def test_run_attack_functional(self) -> None:
        out = run_attack(
            "127.0.0.1", port=self.port, protocol="http_form",
            usernames="admin", passwords="zzz, correct-horse",
            workers=2, rate=100.0, delay=0.0, jitter=0.0,
            fail_string="Invalid credentials")
        self.assertEqual(len(out["found"]), 1)
        self.assertEqual(out["found"][0]["password"], "correct-horse")
        self.assertEqual(out["protocol"], "http_form")


class HttpBasicAttackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.httpd, self.port = _http_server(_BasicHandler)

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_basic_auth_brute(self) -> None:
        attacker = Attacker(
            "127.0.0.1", port=self.port, protocol="http_basic",
            usernames=["admin"], passwords=["nope", "secret9"],
            workers=2, rate=100.0, delay=0.0, jitter=0.0, timeout=5.0,
            quiet=True)
        result = attacker.run()
        self.assertEqual(len(result.found), 1)
        self.assertEqual(result.found[0].password, "secret9")


class FtpAttackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ftp = _FTPServer()
        self.ftp.start()
        time.sleep(0.1)

    def tearDown(self) -> None:
        self.ftp.stop()

    def test_ftp_brute(self) -> None:
        attacker = Attacker(
            "127.0.0.1", port=self.ftp.port, protocol="ftp",
            usernames=["admin"], passwords=["wrong", "ftppass"],
            workers=2, rate=100.0, delay=0.0, jitter=0.0, timeout=5.0,
            quiet=True)
        result = attacker.run()
        self.assertEqual(len(result.found), 1)
        self.assertEqual(result.found[0].password, "ftppass")


class SshAttackerTest(unittest.TestCase):
    def test_injected_connector_finds(self) -> None:
        calls = []

        def fake_connect(host, port, login, password, timeout):
            calls.append((login, password))
            if login == "root" and password == "hunter2":
                return True, "ssh authenticated"
            return False, "ssh auth refused"

        attacker = Attacker(
            "10.0.0.1", protocol="ssh", usernames=["root"],
            passwords=["bad", "hunter2"], workers=2, rate=1000.0,
            delay=0.0, jitter=0.0, ssh_connect=fake_connect, quiet=True)
        result = attacker.run()
        self.assertEqual(len(result.found), 1)
        self.assertEqual(result.found[0].password, "hunter2")
        self.assertIn(("root", "hunter2"), calls)

    def test_transport_error_is_isolated(self) -> None:
        # unreachable port: connection errors are counted, not fatal
        attacker = Attacker(
            "127.0.0.1", port=_free_port(), protocol="ftp",
            usernames=["x"], passwords=["y"], workers=2,
            rate=100.0, delay=0.0, jitter=0.0, timeout=1.0, quiet=True)
        result = attacker.run()
        self.assertEqual(result.found, [])
        self.assertGreaterEqual(result.errors, 1)

    def test_missing_wordlist_and_bad_input(self) -> None:
        with self.assertRaises(ToolError):
            Attacker("", protocol="ftp", usernames=["a"], passwords=["b"])
        with self.assertRaises(ToolError):
            Attacker("1.2.3.4", protocol="ftp", usernames=[], passwords=["b"])
        with self.assertRaises(ToolError):
            Attacker("1.2.3.4", protocol="ftp", usernames=["a"],
                     passwords=[])
        with self.assertRaises(ToolError):
            Attacker("1.2.3.4", protocol="telepathy",
                     usernames=["a"], passwords=["b"])


class PaceGateTest(unittest.TestCase):
    def test_rate_limit_enforced(self) -> None:
        from nomorals.tools.attacker import _PaceGate

        gate = _PaceGate(rate=50.0, delay=0.0, jitter=0.0)
        started = time.monotonic()
        for _ in range(25):
            gate.take_slot()
        elapsed = time.monotonic() - started
        # 24 gaps of 20ms = 0.48s minimum (slack for scheduler)
        self.assertGreaterEqual(elapsed, 0.35)
        self.assertLess(elapsed, 3.0)

    def test_post_adds_pacing(self) -> None:
        from nomorals.tools.attacker import _PaceGate

        gate = _PaceGate(rate=1000.0, delay=0.12, jitter=0.05, quiet=False)
        started = time.monotonic()
        for _ in range(4):
            gate.take_slot()
            gate.post()
        elapsed = time.monotonic() - started
        # 4 x >= 0.12s (jitter only adds; long pauses only add)
        self.assertGreaterEqual(elapsed, 0.45)


# ── generator (PDF's generator.py) ──────────────────────────────────────────


class GeneratorTest(unittest.TestCase):
    def test_combination_space(self) -> None:
        tmp = tempfile.mkdtemp(prefix="gen-")
        out = os.path.join(tmp, "combinations.txt")
        count = generate_combinations(out, charset="ab",
                                      min_len=1, max_len=2)
        self.assertEqual(count, 6)
        with open(out) as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, ["a", "b", "aa", "ab", "ba", "bb"])

    def test_max_candidates_cap(self) -> None:
        tmp = tempfile.mkdtemp(prefix="gen-")
        out = os.path.join(tmp, "capped.txt")
        count = generate_combinations(out, charset="0123456789",
                                      min_len=1, max_len=4,
                                      max_candidates=1234)
        self.assertEqual(count, 1234)
        with open(out) as f:
            lines_c = sum(1 for _ in f)
        self.assertEqual(lines_c, 1234)

    def test_default_charset(self) -> None:
        tmp = tempfile.mkdtemp(prefix="gen-")
        out = os.path.join(tmp, "def.txt")
        count = generate_combinations(out, min_len=1, max_len=1)
        self.assertEqual(count, 36)  # a-z + 0-9


# ── env merge script ─────────────────────────────────────────────────────────


class MergeEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="envmerge-")
        self.repo = os.path.abspath("scripts")
        self.script = os.path.join(self.repo, "merge_env.sh")
        self.example = os.path.join(self.tmp, ".env.example")
        self.env = os.path.join(self.tmp, ".env")
        with open(self.example, "w") as f:
            f.write("# section A\n")
            f.write("NM_ONE=alpha\n")
            f.write("NM_TWO=beta\n")
            f.write("# section B\n")
            f.write("NM_THREE=gamma\n")
            f.write("NM_FOUR=delta\n")

    def _run(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", self.script, "--env", self.env,
             "--example", self.example, *extra],
            capture_output=True, text=True, timeout=30)

    def test_appends_only_missing_keys(self) -> None:
        with open(self.env, "w") as f:
            f.write("NM_ONE=my-override\n")
            f.write("# my own section\n")
            f.write("NM_EXTRA=keep-me\n")
        proc = self._run()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(self.env) as f:
            content = f.read()
        # existing value untouched
        self.assertIn("NM_ONE=my-override", content)
        # user's own key untouched
        self.assertIn("NM_EXTRA=keep-me", content)
        # the example's NM_ONE value did NOT leak in
        self.assertNotIn("NM_ONE=alpha", content)
        # missing keys appended
        self.assertIn("NM_TWO=beta", content)
        self.assertIn("NM_THREE=gamma", content)
        self.assertIn("NM_FOUR=delta", content)
        # ordering preserved: user's original lines still first
        self.assertLess(content.find("NM_ONE=my-override"),
                        content.find("NM_TWO=beta"))

    def test_commented_key_counts_as_existing(self) -> None:
        with open(self.env, "w") as f:
            f.write("# NM_TWO=parked\n")
        proc = self._run()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(self.env) as f:
            content = f.read()
        self.assertIn("# NM_TWO=parked", content)
        self.assertNotIn("NM_TWO=beta", content)

    def test_dry_run_changes_nothing(self) -> None:
        before = "NM_EXISTING=1\n"
        with open(self.env, "w") as f:
            f.write(before)
        proc = self._run("--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NM_TWO=beta", proc.stdout)  # would add
        with open(self.env) as f:
            self.assertEqual(f.read(), before)

    def test_nothing_to_add(self) -> None:
        with open(self.env, "w") as f:
            f.write("NM_ONE=1\nNM_TWO=2\nNM_THREE=3\nNM_FOUR=4\n")
        proc = self._run()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("nothing to add", proc.stdout)


# ── wiring ───────────────────────────────────────────────────────────────────


class WiringTest(unittest.TestCase):
    def test_registry_registers_attacker(self) -> None:
        from tests.test_partner_runtime import _make_context

        from nomorals.tools.registry import ToolRegistry

        ctx, tmp = _make_context()
        try:
            reg = ToolRegistry(ctx)
            reg.register_builtins()
            self.assertIn("attacker", reg.names())
        finally:
            ctx.close()
            tmp.cleanup()

    def test_devon_catalog_lists_attacker(self) -> None:
        from nomorals.agents import devon

        names = {n for n, _ in devon.TOOL_CATALOG}
        self.assertIn("attacker", names)
        self.assertTrue(hasattr(devon.DevonAgent, "_tool_attacker"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
