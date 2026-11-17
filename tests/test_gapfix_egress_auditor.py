"""Gapfix R3-63: the egress auditor records live L2 proxy decisions.

Failing-first: on the unmodified tree ``EgressAuditor`` has no
``record()`` (record-only logging of an externally-made decision),
``run_sandboxed_l2`` takes no ``egress_auditor``, and no L2 run
carries an ``"audit"`` section in its egress summary.
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nomorals.core.egress import EgressController, EgressToken
from nomorals.security.egress_audit import EgressAuditor
from nomorals.tools.sandbox_l2 import EgressProxy, run_sandboxed_l2


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
        if self.path == "/hello":
            self._send(200, b"hello-l2", {"X-Test": "yes"})
        elif self.path == "/redir-evil":
            self._send(302, b"", location="http://evil.example/")
        else:
            self._send(404, b"not found")


class RecordTests(unittest.TestCase):
    """``record()`` logs a decision made elsewhere — it never re-decides."""

    def test_record_logs_allowed_without_redeciding(self):
        auditor = EgressAuditor()
        # The auditor's own policy would BLOCK a private IP — but the
        # live token may allow it (allow_private_ips). record() must log
        # the given verdict, not its own.
        record = auditor.record("10.0.0.1", 80, allowed=True,
                                reason="allowed by egress token t-1",
                                caller="sandbox_l2:proxy")
        self.assertTrue(record.allowed)
        self.assertEqual(record.reason, "allowed by egress token t-1")
        self.assertEqual(record.host, "10.0.0.1")
        self.assertEqual(record.port, 80)
        self.assertEqual(record.caller, "sandbox_l2:proxy")
        self.assertEqual(len(auditor.entries()), 1)

    def test_record_logs_blocked_with_given_reason(self):
        auditor = EgressAuditor()
        # The auditor's own policy would ALLOW a public IP — record()
        # must still log the given block with the given reason.
        auditor.record("8.8.8.8", 53, allowed=False,
                       reason="host '8.8.8.8' not on allowlist",
                       caller="sandbox_l2:proxy")
        entries = auditor.entries()
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0].allowed)
        self.assertEqual(entries[0].reason,
                         "host '8.8.8.8' not on allowlist")

    def test_record_mirrors_to_jsonl(self):
        with tempfile.TemporaryDirectory(prefix="nm-egress-audit-") as tmp:
            path = Path(tmp) / "egress.jsonl"
            auditor = EgressAuditor(log_path=path)
            auditor.record("example.com", 443, allowed=True,
                           reason="allowed by egress token t-1")
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn('"host": "example.com"', lines[0])
            self.assertIn('"allowed": true', lines[0])

    def test_record_never_raises(self):
        with tempfile.TemporaryDirectory(prefix="nm-egress-audit-") as tmp:
            auditor = EgressAuditor(log_path=Path(tmp) / "egress.jsonl")
            with mock.patch("builtins.open",
                            side_effect=OSError("disk gone")):
                # The file mirror is dead — the call must still not raise,
                # and the in-memory entry must still land.
                auditor.record("example.com", 443, allowed=True,
                               reason="allowed by egress token t-1")
            self.assertEqual(len(auditor.entries()), 1)
            self.assertTrue(auditor.entries()[0].allowed)


class ProxyAuditTests(unittest.TestCase):
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

    def _py_fetch(self, url):
        return (
            "python3 -c \"import urllib.request; "
            f"print(urllib.request.urlopen({url!r}).read().decode())\""
        )

    def test_L2_run_records_allowed_fetch(self):
        url = f"http://localhost:{self.port}/hello"
        result = run_sandboxed_l2(self._py_fetch(url), cwd=self.tmp,
                                  egress_token=self._token())
        self.assertEqual(result["level"], "L2")
        self.assertEqual(result["exit_code"], 0, result["stderr"])
        self.assertIn("hello-l2", result["stdout"])
        # The run's outcome is unchanged — and now it carries a record.
        self.assertEqual(result["egress"]["requests"], 1)
        audit = result["egress"]["audit"]
        self.assertGreaterEqual(audit["total"], 1)
        self.assertEqual(audit["allowed"], audit["total"])
        # audit_report() lists blocked rows explicitly; allowed rows are
        # summarised in by_host / by_reason.
        self.assertEqual(audit["by_host"].get("localhost"), 1)
        self.assertTrue(any("allowed by egress token" in reason
                            for reason in audit["by_reason"]))

    def test_L2_run_records_blocked_fetch(self):
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
        audit = result["egress"]["audit"]
        self.assertGreaterEqual(audit["blocked"], 1)
        blocked = audit["blocked_entries"]
        self.assertTrue(any("evil.example" in e["reason"]
                            for e in blocked),
                        f"no blocked entry names evil.example: {blocked}")
        self.assertTrue(all(e["host"] == "evil.example" for e in blocked))

    def test_L2_run_records_redirect_denial_with_live_reason(self):
        url = f"http://localhost:{self.port}/redir-evil"
        result = run_sandboxed_l2(self._py_fetch(url), cwd=self.tmp,
                                  egress_token=self._token())
        # The fetch itself fails inside the shell (403 from the proxy);
        # the run still completes and the denial is recorded.
        audit = result["egress"]["audit"]
        self.assertGreaterEqual(audit["blocked"], 1)
        self.assertTrue(any("evil.example" in e["reason"]
                            for e in audit["blocked_entries"]))

    def test_caller_passed_auditor_is_shared(self):
        auditor = EgressAuditor()
        url = f"http://localhost:{self.port}/hello"
        result = run_sandboxed_l2(self._py_fetch(url), cwd=self.tmp,
                                  egress_token=self._token(),
                                  egress_auditor=auditor)
        self.assertEqual(result["egress"]["audit"]["total"], 1)
        self.assertEqual(len(auditor.entries()), 1)
        self.assertEqual(auditor.entries()[0].host, "localhost")

    def test_failing_auditor_never_breaks_run(self):
        auditor = EgressAuditor()
        with mock.patch.object(auditor, "record",
                               side_effect=RuntimeError("audit broken")):
            url = f"http://localhost:{self.port}/hello"
            result = run_sandboxed_l2(self._py_fetch(url), cwd=self.tmp,
                                      egress_token=self._token(),
                                      egress_auditor=auditor)
        self.assertEqual(result["level"], "L2")
        self.assertEqual(result["exit_code"], 0, result["stderr"])
        self.assertIn("hello-l2", result["stdout"])

    def test_L1_run_has_no_audit(self):
        result = run_sandboxed_l2("echo hi", cwd=self.tmp)
        self.assertEqual(result["level"], "L1")
        self.assertNotIn("audit", result["egress"])

    def test_connect_tunnel_records_allowed_and_blocked(self):
        controller = EgressController()
        token = self._token()
        auditor = EgressAuditor()
        with EgressProxy(token=token, controller=controller,
                         auditor=auditor) as proxy:
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
        entries = auditor.entries()
        allowed = [e for e in entries
                   if e.allowed and e.host == "localhost"]
        self.assertEqual(len(allowed), 1)
        self.assertEqual(allowed[0].port, self.port)
        self.assertTrue(allowed[0].resolved_ips)
        blocked = [e for e in entries if not e.allowed]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].host, "evil.example")
        self.assertEqual(blocked[0].port, 443)
        self.assertIn("evil.example", blocked[0].reason)


if __name__ == "__main__":
    unittest.main()
