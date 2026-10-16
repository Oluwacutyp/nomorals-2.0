"""Sandbox L2: token-gated network on top of offline L1.

Invariants under test:

* no token → no network (L1, no proxy env);
* same result shape L1 vs L2 (stdout/stderr/exit/timed_out/seconds/backend);
* post-resolution checks enforced (IP literal, private IP, disguised
  IP forms, off-allowlist redirect, redirect to IP literal);
* per-token + global kill switches work, with in-flight logging;
* budgets (requests / bytes / TTL) enforced.
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from nomorals.core.cassette import GenericCassetteStore
from nomorals.core.egress import (
    EgressClient,
    EgressController,
    EgressDenied,
    EgressToken,
)
from nomorals.execbox import CodeRunner
from nomorals.tools.sandbox_l2 import EgressProxy, run_sandboxed_l2
from nomorals.tools.shell import detect_backend


class _TestServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, body, headers=None, location=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if location is not None:
            self.send_header("Location", location)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        port = self.server.server_address[1]
        if self.path == "/hello":
            self._send(200, b"hello-l2", {"X-Test": "yes"})
        elif self.path == "/final":
            self._send(200, b"final")
        elif self.path == "/redir-ok":
            self._send(302, b"", location="/final")
        elif self.path == "/redir-evil":
            self._send(302, b"", location="http://evil.example/")
        elif self.path == "/redir-literal":
            self._send(302, b"", location=f"http://127.0.0.1:{port}/hello")
        elif self.path == "/slow":
            time.sleep(2.0)
            self._send(200, b"slow")
        elif self.path == "/big":
            self._send(200, b"x" * 10240)
        else:
            self._send(404, b"not found")


class SandboxL2Tests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-l2-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._server = _TestServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
            daemon=True)
        self._thread.start()
        self.port = self._server.server_address[1]
        self.addCleanup(self._server.shutdown)
        self.addCleanup(self._server.server_close)

    def _token(self, **kw):
        kw.setdefault("domains", ("localhost",))
        kw.setdefault("allow_private_ips", True)
        return EgressToken.mint("task-l2", **kw)

    def _client(self, controller=None, **kw):
        return EgressClient(controller=controller or EgressController(), **kw)

    def _py_fetch(self, url):
        """Shell snippet: fetch ``url`` through the env proxy, print body."""
        return (
            "python3 -c \"import urllib.request; "
            f"print(urllib.request.urlopen({url!r}).read().decode())\""
        )

    # ── L1: no token = no network ──
    def test_no_token_is_L1_without_proxy_env(self):
        result = run_sandboxed_l2("env", cwd=self.tmp)
        self.assertEqual(result["level"], "L1")
        self.assertFalse(result["egress"]["network"])
        self.assertNotIn("http_proxy", result["stdout"].lower())
        self.assertNotIn("https_proxy", result["stdout"].lower())
        self.assertEqual(result["exit_code"], 0)

    def test_no_token_blocks_live_network_when_namespaces_exist(self):
        backend = detect_backend()
        if backend not in ("bwrap", "unshare"):
            self.skipTest(f"no namespace backend here ({backend})")
        url = f"http://localhost:{self.port}/hello"
        result = run_sandboxed_l2(self._py_fetch(url), cwd=self.tmp)
        self.assertEqual(result["level"], "L1")
        self.assertNotEqual(result["exit_code"], 0)

    # ── same result shape ──
    def test_L1_and_L2_share_result_shape(self):
        l1 = run_sandboxed_l2("echo hi", cwd=self.tmp)
        l2 = run_sandboxed_l2("echo hi", cwd=self.tmp,
                              egress_token=self._token())
        for key in ("stdout", "stderr", "exit_code", "timed_out",
                    "seconds", "backend", "level", "egress"):
            self.assertIn(key, l1)
            self.assertIn(key, l2)
        self.assertEqual(l1["stdout"].strip(), "hi")
        self.assertEqual(l2["stdout"].strip(), "hi")
        self.assertEqual(l2["level"], "L2")
        self.assertTrue(l2["egress"]["network"])

    # ── L2 live: shell HTTP through the proxy ──
    def test_L2_shell_fetch_through_proxy(self):
        url = f"http://localhost:{self.port}/hello"
        result = run_sandboxed_l2(self._py_fetch(url), cwd=self.tmp,
                                  egress_token=self._token())
        self.assertEqual(result["level"], "L2")
        self.assertEqual(result["exit_code"], 0, result["stderr"])
        self.assertIn("hello-l2", result["stdout"])
        self.assertEqual(result["egress"]["requests"], 1)

    def test_L2_shell_fetch_off_allowlist_denied(self):
        cmd = (
            "python3 -c \""
            "import urllib.request, urllib.error\n"
            "try:\n"
            "    urllib.request.urlopen('http://evil.example/')\n"
            "    print('LEAKED')\n"
            "except urllib.error.HTTPError as e:\n"
            "    print('DENIED', e.code)\n"
            "\""
        )
        result = run_sandboxed_l2(cmd, cwd=self.tmp,
                                  egress_token=self._token())
        self.assertIn("DENIED 403", result["stdout"])
        self.assertNotIn("LEAKED", result["stdout"])

    def test_L2_replay_mode_is_L1_offline(self):
        store = GenericCassetteStore(self.tmp / "net",
                                     filename="network_cassettes.jsonl")
        url = f"http://localhost:{self.port}/hello"
        result = run_sandboxed_l2(
            self._py_fetch(url), cwd=self.tmp, egress_token=self._token(),
            cassette=store, cassette_mode="replay")
        self.assertEqual(result["level"], "L1")
        self.assertFalse(result["egress"]["network"])
        self.assertEqual(result["egress"]["mode"], "L1-replay")

    # ── post-resolution checks ──
    def test_ip_literal_refused(self):
        client = self._client()
        for url in (f"http://127.0.0.1:{self.port}/hello",
                    "http://169.254.169.254/latest/meta-data/",
                    "http://192.168.1.1/",
                    "http://10.0.0.5/",
                    "http://172.16.0.9/"):
            with self.assertRaises(EgressDenied, msg=url):
                client.get(self._token(), url)

    def test_disguised_ip_literals_refused(self):
        client = self._client()
        for url in ("http://2130706433/", "http://0x7f000001/",
                    "http://127.1/", "http://0177.0.0.1/"):
            with self.assertRaises(EgressDenied, msg=url):
                client.get(self._token(), url)

    def test_dns_resolving_to_private_ip_refused(self):
        # Post-resolution: an allowlisted name whose CURRENT answer is
        # link-local must be refused — the check runs after resolution.
        real_getaddrinfo = socket.getaddrinfo

        def fake_getaddrinfo(host, port, *args, **kwargs):
            if host == "allowed.example":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                         ("169.254.169.254", 0))]
            return real_getaddrinfo(host, port, *args, **kwargs)

        token = EgressToken.mint("task-dns", domains=("allowed.example",),
                                 allow_private_ips=False)
        with mock.patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
            with self.assertRaises(EgressDenied):
                self._client().get(token, "http://allowed.example/")

    def test_redirect_to_off_allowlist_refused(self):
        client = self._client()
        url = f"http://localhost:{self.port}/redir-evil"
        with self.assertRaises(EgressDenied):
            client.get(self._token(), url)

    def test_redirect_to_ip_literal_refused(self):
        client = self._client()
        url = f"http://localhost:{self.port}/redir-literal"
        with self.assertRaises(EgressDenied):
            client.get(self._token(), url)

    def test_redirect_to_same_host_followed(self):
        client = self._client()
        url = f"http://localhost:{self.port}/redir-ok"
        response = client.get(self._token(), url)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body, b"final")
        self.assertEqual(response.hops, [url, url.replace("/redir-ok", "/final")])

    def test_credentials_in_url_refused(self):
        client = self._client()
        with self.assertRaises(EgressDenied):
            client.get(self._token(), f"http://user:pass@localhost:{self.port}/hello")

    def test_non_http_scheme_refused(self):
        client = self._client()
        with self.assertRaises(EgressDenied):
            client.get(self._token(), "file:///etc/passwd")
        with self.assertRaises(EgressDenied):
            client.get(self._token(), "ftp://localhost/file")

    # ── kill switches ──
    def test_per_token_revoke_denies(self):
        controller = EgressController()
        client = EgressClient(controller=controller)
        token = self._token()
        controller.kill_token(token, "test revoke")
        with self.assertRaises(EgressDenied):
            client.get(token, f"http://localhost:{self.port}/hello")

    def test_global_kill_switch_denies_and_logs_in_flight(self):
        controller = EgressController()
        client = EgressClient(controller=controller)
        token = self._token()
        controller.register(token)
        controller.kill_all("test global kill")
        with self.assertRaises(EgressDenied):
            client.get(token, f"http://localhost:{self.port}/hello")
        # A second token on the same controller is dead too.
        with self.assertRaises(EgressDenied):
            client.get(self._token(), f"http://localhost:{self.port}/hello")

    def test_kill_switch_aborts_in_flight_request(self):
        controller = EgressController()
        client = EgressClient(controller=controller)
        token = self._token()
        errors: list[BaseException] = []

        def _fetch():
            try:
                client.get(token, f"http://localhost:{self.port}/slow")
            except BaseException as exc:  # noqa: BLE001 - captured for assertion
                errors.append(exc)

        thread = threading.Thread(target=_fetch, daemon=True)
        thread.start()
        time.sleep(0.3)
        report = controller.kill_all("test in-flight kill")
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], EgressDenied)
        self.assertIn("kill", str(errors[0]).lower())
        in_flight = report["in_flight"]
        self.assertEqual(len(in_flight), 1)
        self.assertIn("/slow", in_flight[0]["url"])

    # ── budgets ──
    def test_request_budget_enforced(self):
        controller = EgressController()
        client = EgressClient(controller=controller)
        token = self._token(max_requests=1)
        url = f"http://localhost:{self.port}/hello"
        client.get(token, url)
        with self.assertRaises(EgressDenied):
            client.get(token, url)
        self.assertEqual(token.requests_used, 1)

    def test_byte_budget_enforced(self):
        client = self._client()
        token = self._token(max_bytes=100)
        with self.assertRaises(EgressDenied):
            client.get(token, f"http://localhost:{self.port}/big")

    def test_token_expiry_enforced(self):
        client = self._client()
        token = self._token(ttl_s=0.05)
        time.sleep(0.12)
        with self.assertRaises(EgressDenied):
            client.get(token, f"http://localhost:{self.port}/hello")

    def test_path_prefix_enforced(self):
        client = self._client()
        token = EgressToken.mint("task-path", domains=("localhost",),
                                 path_prefixes=("/api/",),
                                 allow_private_ips=True)
        with self.assertRaises(EgressDenied):
            client.get(token, f"http://localhost:{self.port}/hello")

    # ── CONNECT tunnel ──
    def test_connect_tunnel_allowed_and_denied(self):
        controller = EgressController()
        token = self._token()
        with EgressProxy(token=token, controller=controller) as proxy:
            host, port = proxy.proxy_url.split("://", 1)[1].split(":", 1)
            # Allowed: CONNECT to the local server, then a plain GET.
            sock = socket.create_connection((host, int(port)), timeout=5)
            try:
                sock.sendall(
                    f"CONNECT localhost:{self.port} HTTP/1.1\r\n"
                    f"Host: localhost:{self.port}\r\n\r\n".encode())
                head = b""
                while b"\r\n\r\n" not in head:
                    head += sock.recv(4096)
                self.assertIn(b"200", head.split(b"\r\n")[0])
                sock.sendall(b"GET /hello HTTP/1.0\r\nHost: localhost\r\n\r\n")
                data = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                self.assertIn(b"hello-l2", data)
            finally:
                sock.close()
            # Denied: CONNECT to an off-allowlist host.
            sock = socket.create_connection((host, int(port)), timeout=5)
            try:
                sock.sendall(b"CONNECT evil.example:443 HTTP/1.1\r\n"
                              b"Host: evil.example:443\r\n\r\n")
                head = b""
                while b"\r\n\r\n" not in head:
                    head += sock.recv(4096)
                self.assertIn(b"403", head.split(b"\r\n")[0])
            finally:
                sock.close()

    # ── CodeRunner wiring ──
    def test_coderunner_L1_default_and_L2_with_token(self):
        box = CodeRunner(context=None)
        l1 = box.run("print('hi')", lang="python")
        self.assertEqual(l1["level"], "L1")
        self.assertIn("hi", l1["stdout"])
        l2 = box.run("print('hi')", lang="python",
                     egress_token=self._token())
        self.assertEqual(l2["level"], "L2")
        self.assertIn("hi", l2["stdout"])
        replay = box.run("print('hi')", lang="python",
                         egress_token=self._token(),
                         cassette_mode="replay")
        self.assertEqual(replay["level"], "L1")
        self.assertEqual(replay["egress"]["mode"], "L1-replay")


if __name__ == "__main__":
    unittest.main()
