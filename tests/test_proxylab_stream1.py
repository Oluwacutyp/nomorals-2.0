"""Stream 1 — proxy lab upgrade tests (2026-10-09).

Hermetic where the network is involved: in-process fake HTTP forward
proxy, SOCKS5, and SOCKS4 servers prove the tester + HttpClient routing
relay traffic for real.  Pure-logic tests (decay scoring, backoff,
affinity, protocol-aware picks, health dashboard, parser fixes) run
with mocked time and no network at all.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import socketserver
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

from nomorals.tools import proxylab as PL
from nomorals.core import http as core_http


# ── fake proxy fixtures (in-process, hermetic) ────────────────────────────────

FAKE_EGRESS_IP = "203.0.113.7"     # TEST-NET-3 — the "proxy's" egress
FAKE_REAL_IP = "198.51.100.9"      # TEST-NET-2 — "our" real IP


class _ForwardProxyHandler(BaseHTTPRequestHandler):
    """A real HTTP forward proxy (absolute-form requests), in-process.

    Path /ip      → the egress IP as the body
    Path /headers → JSON {"headers": <received headers>} (+ an injected
                    X-Forwarded-For when the server wants to look
                    "anonymous" instead of "elite")
    """
    inject_leak = False
    seen_requests: list = []

    def log_message(self, *args):  # noqa: D102 - quiet
        pass

    def _serve(self):
        parsed = urlparse(self.path)
        # absolute-form (through a proxy) or origin-form (direct)
        path = parsed.path if parsed.scheme else self.path.split("?")[0]
        hdrs = {k: v for k, v in self.headers.items()}
        type(self).seen_requests.append((self.command, self.path, hdrs))
        if path == "/ip":
            body = FAKE_EGRESS_IP.encode()
        elif path == "/headers":
            if type(self).inject_leak:
                hdrs["X-Forwarded-For"] = FAKE_REAL_IP
            body = json.dumps({"headers": hdrs}).encode()
        elif path == "/country":
            body = json.dumps({"country": "NG"}).encode()
        else:
            body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    do_GET = _serve


class _Socks5Handler(socketserver.BaseRequestHandler):
    """Minimal RFC 1928 server: no-auth greeting, CONNECT → success, then
    the same canned HTTP responses as the forward proxy."""

    def handle(self):  # noqa: D102
        f = self.request.makefile("rwb")
        ver, nmethods = f.read(2)
        f.read(nmethods)  # methods
        f.write(b"\x05\x00")  # no auth
        f.flush()
        head = f.read(4)
        atyp = head[3]
        if atyp == 0x01:
            f.read(6)
        elif atyp == 0x03:
            ln = f.read(1)[0]
            f.read(ln + 2)
        elif atyp == 0x04:
            f.read(18)
        f.write(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
        f.flush()
        self._http(f)

    def _http(self, f):
        # read the header block line-by-line: a single read(n) on the
        # buffered makefile would block until n bytes or EOF, but the
        # client holds the connection open (Connection: close)
        head_lines = []
        while True:
            line = f.readline(65536)
            if not line or line in (b"\r\n", b"\n"):
                break
            head_lines.append(line)
        if not head_lines:
            return
        head = b"".join(head_lines).decode("latin-1", "replace")
        line = head.split("\r\n", 1)[0]
        parts = line.split(" ")
        path = parts[1] if len(parts) > 1 else "/"
        hdrs = {}
        for hline in head.split("\r\n")[1:]:
            if ":" in hline:
                k, v = hline.split(":", 1)
                hdrs[k.strip()] = v.strip()
        if path == "/ip":
            body = FAKE_EGRESS_IP.encode()
        elif path == "/headers":
            if getattr(type(self), "inject_leak", False):
                hdrs["X-Forwarded-For"] = FAKE_REAL_IP
            body = json.dumps({"headers": hdrs}).encode()
        elif path == "/country":
            body = json.dumps({"country": "NG"}).encode()
        else:
            body = b"ok"
        f.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n"
                b"Connection: close\r\n\r\n" % len(body))
        f.write(body)
        f.flush()


class _Socks4Handler(socketserver.BaseRequestHandler):
    """Minimal SOCKS4 server: CONNECT → 0x5A granted, then canned HTTP."""

    def handle(self):  # noqa: D102
        f = self.request.makefile("rwb")
        req = f.read(8)
        # userid (NUL-terminated), then optional domain for 4a
        while f.read(1) != b"\x00":
            pass
        if req[4:8] == b"\x00\x00\x00\x01":
            while f.read(1) != b"\x00":
                pass
        f.write(b"\x00\x5a\x00\x00\x00\x00\x00\x00")
        f.flush()
        _Socks5Handler._http(self, f)


def _start(server_cls, handler_cls):
    server_cls.allow_reuse_address = True
    srv = server_cls(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv


class FakeProxyLab(unittest.TestCase):
    """Spin up the fake proxies once for the hermetic relay tests."""

    @classmethod
    def setUpClass(cls):
        cls.http_proxy = _start(socketserver.ThreadingTCPServer,
                                _ForwardProxyHandler)
        cls.http_port = cls.http_proxy.server_address[1]
        cls.socks5 = _start(socketserver.ThreadingTCPServer,
                            _Socks5Handler)
        cls.socks5_port = cls.socks5.server_address[1]
        cls.socks4 = _start(socketserver.ThreadingTCPServer,
                            _Socks4Handler)
        cls.socks4_port = cls.socks4.server_address[1]
        _ForwardProxyHandler.seen_requests = []
        _ForwardProxyHandler.inject_leak = False
        _Socks5Handler.inject_leak = False

    @classmethod
    def tearDownClass(cls):
        for srv in (cls.http_proxy, cls.socks5, cls.socks4):
            srv.shutdown()
            srv.server_close()

    def _tester(self, leak=False):
        _ForwardProxyHandler.inject_leak = leak
        _Socks5Handler.inject_leak = leak
        base = f"http://127.0.0.1:{self.http_port}"
        return PL.ProxyTester(
            ip_url=f"{base}/ip", echo_url=f"{base}/headers",
            country_url=f"{base}/country",
            local_ip_provider=lambda: FAKE_REAL_IP,
            timeout=5.0, max_workers=4)

    def test_http_proxy_relay_detected(self):
        t = self._tester()
        p = PL.Proxy(host="127.0.0.1", port=self.http_port, scheme="http")
        t.test_one(p)
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.egress_ip, FAKE_EGRESS_IP)
        self.assertGreater(p.latency_ms, 0)
        self.assertEqual(len(p.latency_samples), 1)

    def test_socks5_proxy_relay_detected(self):
        t = self._tester()
        p = PL.Proxy(host="127.0.0.1", port=self.socks5_port,
                     scheme="socks5")
        t.test_one(p)
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.egress_ip, FAKE_EGRESS_IP)

    def test_socks4_proxy_relay_detected(self):
        t = self._tester()
        p = PL.Proxy(host="127.0.0.1", port=self.socks4_port,
                     scheme="socks4")
        t.test_one(p)
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.egress_ip, FAKE_EGRESS_IP)

    def test_anonymity_elite_when_no_leak_headers(self):
        t = self._tester(leak=False)
        p = PL.Proxy(host="127.0.0.1", port=self.http_port, scheme="http")
        t.test_one(p)
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.anonymity, "elite")

    def test_anonymity_anonymous_when_xff_leaks(self):
        t = self._tester(leak=True)
        p = PL.Proxy(host="127.0.0.1", port=self.http_port, scheme="http")
        t.test_one(p)
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.anonymity, "anonymous")

    def test_anonymity_transparent_when_egress_is_local(self):
        t = self._tester()
        t._local_ip_provider = lambda: FAKE_EGRESS_IP  # egress == "local"
        p = PL.Proxy(host="127.0.0.1", port=self.http_port, scheme="http")
        t.test_one(p)
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.anonymity, "transparent")

    def test_country_detected_without_api_key(self):
        t = self._tester()
        p = PL.Proxy(host="127.0.0.1", port=self.http_port, scheme="http")
        t.test_one(p)
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.country, "NG")

    def test_dead_proxy_marked_not_alive(self):
        t = self._tester()
        p = PL.Proxy(host="127.0.0.1", port=1, scheme="http")  # nothing here
        t.test_one(p)
        self.assertFalse(p.alive)
        self.assertTrue(p.error)

    def test_rate_limit_page_is_not_proof_of_relay(self):
        # a probe URL that answers 200 with {"error": ...} must NOT mark
        # the proxy alive with garbage as egress_ip
        t = self._tester()
        t.ip_url = f"http://127.0.0.1:{self.http_port}/headers"
        p = PL.Proxy(host="127.0.0.1", port=self.http_port, scheme="http")
        t.test_one(p)
        self.assertFalse(p.alive)
        self.assertIn("non-IP", p.error)


# ── HttpClient proxy wiring (hermetic) ───────────────────────────────────────


class HttpRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proxy = _start(socketserver.ThreadingTCPServer,
                           _ForwardProxyHandler)
        cls.proxy_port = cls.proxy.server_address[1]
        _ForwardProxyHandler.seen_requests = []

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown()
        cls.proxy.server_close()

    def setUp(self):
        core_http.set_proxy_resolver(None)
        core_http.set_proxy_error_reporter(None)
        core_http.set_proxy_success_reporter(None)
        core_http.set_default_proxy("")
        # urllib honors no_proxy (this env bypasses loopback) and the
        # HTTP(S)_PROXY env vars (this env routes through an egress
        # proxy) — the lab's explicit proxy must win in these tests, so
        # drop all of them
        self._saved_env = {}
        for var in ("no_proxy", "NO_PROXY", "http_proxy", "HTTP_PROXY",
                    "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            if var in os.environ:
                self._saved_env[var] = os.environ.pop(var)

    def tearDown(self):
        core_http.set_proxy_resolver(None)
        core_http.set_proxy_error_reporter(None)
        core_http.set_proxy_success_reporter(None)
        core_http.set_default_proxy("")
        os.environ.update(self._saved_env)

    def test_resolver_routes_http_client_through_proxy(self):
        seen_targets = []

        def resolver(target=""):
            seen_targets.append(target)
            return f"http://127.0.0.1:{self.proxy_port}"

        core_http.set_proxy_resolver(resolver)
        _ForwardProxyHandler.seen_requests = []
        client = core_http.HttpClient(timeout=5.0, allow_private_ips=True)
        resp = client.get(f"http://127.0.0.1:{self.proxy_port}/ip")
        self.assertEqual(resp.status, 200)
        # the proxy saw an absolute-form request = it relayed
        self.assertTrue(_ForwardProxyHandler.seen_requests)
        method, path, _hdrs = _ForwardProxyHandler.seen_requests[-1]
        self.assertTrue(path.startswith("http://"),
                        f"not absolute-form: {path}")
        # the resolver got the target URL (domain affinity needs it)
        self.assertTrue(seen_targets and seen_targets[0].startswith("http"))

    def test_default_proxy_routes_when_no_resolver(self):
        core_http.set_default_proxy(f"http://127.0.0.1:{self.proxy_port}")
        _ForwardProxyHandler.seen_requests = []
        client = core_http.HttpClient(timeout=5.0, allow_private_ips=True)
        resp = client.get(f"http://127.0.0.1:{self.proxy_port}/ip")
        self.assertEqual(resp.status, 200)
        self.assertTrue(_ForwardProxyHandler.seen_requests)

    def test_explicit_client_proxy_wins(self):
        core_http.set_default_proxy("http://127.0.0.1:1")  # dead default
        _ForwardProxyHandler.seen_requests = []
        client = core_http.HttpClient(
            timeout=5.0, allow_private_ips=True,
            proxy_url=f"http://127.0.0.1:{self.proxy_port}")
        resp = client.get(f"http://127.0.0.1:{self.proxy_port}/ip")
        self.assertEqual(resp.status, 200)
        self.assertTrue(_ForwardProxyHandler.seen_requests)

    def test_error_reporter_fires_on_connection_failure(self):
        reports = []
        core_http.set_proxy_resolver(lambda target="": "http://127.0.0.1:1")
        core_http.set_proxy_error_reporter(
            lambda url, reason: reports.append((url, reason)))
        client = core_http.HttpClient(timeout=3.0, allow_private_ips=True)
        with self.assertRaises(Exception):
            client.get("http://127.0.0.1:9/nope")
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0][0], "http://127.0.0.1:1")
        self.assertTrue(reports[0][1])

    def test_success_reporter_fires_with_latency(self):
        reports = []
        core_http.set_proxy_resolver(
            lambda target="": f"http://127.0.0.1:{self.proxy_port}")
        core_http.set_proxy_success_reporter(
            lambda url, ms: reports.append((url, ms)))
        client = core_http.HttpClient(timeout=5.0, allow_private_ips=True)
        client.get(f"http://127.0.0.1:{self.proxy_port}/ip")
        self.assertEqual(len(reports), 1)
        self.assertGreater(reports[0][1], 0)

    def test_no_proxy_no_reports(self):
        reports = []
        core_http.set_proxy_success_reporter(
            lambda url, ms: reports.append((url, ms)))
        client = core_http.HttpClient(timeout=5.0, allow_private_ips=True)
        # direct to the fake "origin" server (it answers origin-form too)
        resp = client.get(f"http://127.0.0.1:{self.proxy_port}/ip")
        self.assertEqual(resp.status, 200)
        self.assertEqual(reports, [])


# ── latency decay + scoring (mocked time, no network) ────────────────────────


class DecayTests(unittest.TestCase):
    def test_recent_sample_weighs_more(self):
        # half-life 15min; samples an hour apart → the old one has
        # decayed to ~1/16 of its weight
        p = PL.Proxy(host="1.1.1.1", port=8080)
        p.record_latency(100.0, at=0.0)       # old: fast
        p.record_latency(2000.0, at=3500.0)   # recent: slow
        decayed = p.decayed_latency(at=3600.0, half_life_hours=0.25)
        self.assertGreater(decayed, 1500.0)
        # and the reverse: old slow, recent fast
        q = PL.Proxy(host="2.2.2.2", port=8080)
        q.record_latency(2000.0, at=0.0)
        q.record_latency(100.0, at=3500.0)
        self.assertLess(q.decayed_latency(at=3600.0, half_life_hours=0.25),
                        600.0)

    def test_very_old_samples_decay_away(self):
        p = PL.Proxy(host="1.1.1.1", port=8080)
        p.record_latency(5000.0, at=0.0)
        p.record_latency(100.0, at=30 * 24 * 3600.0)  # 30 days later
        d = p.decayed_latency(at=30 * 24 * 3600.0, half_life_hours=6.0)
        self.assertAlmostEqual(d, 100.0, delta=1.0)

    def test_samples_capped(self):
        p = PL.Proxy(host="1.1.1.1", port=8080)
        for i in range(30):
            p.record_latency(float(i), at=float(i))
        self.assertLessEqual(len(p.latency_samples), 12)

    def test_score_ordering(self):
        now = 5000.0

        def mk(anon, ms, country="", failures=0, backoff=0.0):
            p = PL.Proxy(host="9.9.9.9", port=8080, scheme="http",
                         anonymity=anon, country=country, alive=True,
                         consecutive_failures=failures,
                         backoff_until=backoff)
            p.record_latency(ms, at=now)
            return p

        elite = mk("elite", 800.0)
        anon = mk("anonymous", 100.0)
        transp = mk("transparent", 50.0)
        # anonymity outranks speed: slow elite beats fast transparent
        self.assertGreater(PL.score_proxy(elite, at=now),
                           PL.score_proxy(transp, at=now))
        self.assertGreater(PL.score_proxy(anon, at=now),
                           PL.score_proxy(transp, at=now))
        # backoff → unroutable
        backed = mk("elite", 50.0, backoff=now + 600)
        self.assertEqual(PL.score_proxy(backed, at=now), float("-inf"))
        # dead → unroutable
        dead = mk("elite", 50.0)
        dead.alive = False
        self.assertEqual(PL.score_proxy(dead, at=now), float("-inf"))
        # NG bonus beats equal EU, EU beats equal unknown
        ng = mk("elite", 300.0, country="NG")
        de = mk("elite", 300.0, country="DE")
        us = mk("elite", 300.0, country="US")
        self.assertGreater(PL.score_proxy(ng, at=now),
                           PL.score_proxy(de, at=now))
        self.assertGreater(PL.score_proxy(de, at=now),
                           PL.score_proxy(us, at=now))
        # failure streak drags the score
        clean = mk("elite", 300.0)
        streaky = mk("elite", 300.0, failures=4)
        self.assertGreater(PL.score_proxy(clean, at=now),
                           PL.score_proxy(streaky, at=now))

    def test_rank_excludes_backoff_and_dead_first(self):
        now = 9000.0
        good = PL.Proxy(host="1.1.1.1", port=1, scheme="http",
                        anonymity="elite", alive=True)
        good.record_latency(200.0, at=now)
        bad = PL.Proxy(host="2.2.2.2", port=2, scheme="http",
                       anonymity="elite", alive=True,
                       backoff_until=now + 300)
        bad.record_latency(50.0, at=now)
        dead = PL.Proxy(host="3.3.3.3", port=3, scheme="http", alive=False)
        ranked = PL.rank_proxies([bad, dead, good], at=now)
        self.assertEqual(ranked[0].host, "1.1.1.1")
        self.assertEqual(ranked[-1].host, "3.3.3.3")


class FailureBackoffTests(unittest.TestCase):
    def test_exponential_backoff_and_cap(self):
        p = PL.Proxy(host="1.1.1.1", port=8080, alive=True)
        t0 = time.time()
        until1 = p.record_failure("rst", backoff_seconds=300.0, at=t0)
        self.assertEqual(p.consecutive_failures, 1)
        self.assertAlmostEqual(until1, t0 + 300.0)
        until2 = p.record_failure("rst", backoff_seconds=300.0, at=t0)
        self.assertAlmostEqual(until2, t0 + 600.0)
        p.consecutive_failures = 10
        until3 = p.record_failure("rst", backoff_seconds=300.0, at=t0)
        self.assertAlmostEqual(until3, t0 + 3600.0)  # capped at 1h
        self.assertEqual(p.error, "rst")
        # routing failure is transient: alive untouched (only the lab's
        # own test marks a proxy dead), but it is out of routing now
        self.assertTrue(p.alive)
        self.assertTrue(p.in_backoff)

    def test_success_clears_streak_and_backoff(self):
        p = PL.Proxy(host="1.1.1.1", port=8080, alive=True)
        now = time.time()
        p.record_failure("boom", at=now)
        self.assertTrue(p.in_backoff)
        p.record_success(120.0, at=now)
        self.assertEqual(p.consecutive_failures, 0)
        self.assertEqual(p.backoff_until, 0.0)
        self.assertTrue(p.alive)
        self.assertEqual(p.last_success, now)

    def test_roundtrip_persistence(self):
        p = PL.Proxy(host="1.2.3.4", port=8080, scheme="socks5",
                     anonymity="elite", country="NG", alive=True)
        p.record_latency(111.0, at=1000.0)
        p.record_latency(222.0, at=2000.0)
        p.record_failure("timeout", backoff_seconds=300.0, at=3000.0)
        d = p.to_dict()
        q = PL.Proxy.from_dict(d)
        self.assertEqual(q.url, p.url)
        self.assertEqual(q.anonymity, "elite")
        self.assertEqual(q.country, "NG")
        self.assertEqual(q.consecutive_failures, 1)
        self.assertAlmostEqual(q.backoff_until, 3300.0)
        self.assertEqual(len(q.latency_samples), 2)


# ── store feedback ───────────────────────────────────────────────────────────


class StoreFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.store = PL.ProxyStore(self.tmp)

    def _seed(self, url="http://1.1.1.1:8080"):
        p = PL.Proxy(host="1.1.1.1", port=8080, scheme="http",
                     anonymity="elite", alive=True, tested_at=time.time())
        p.record_latency(150.0)
        self.store.save([p])
        return p

    def test_record_routing_failure_demotes_not_drops(self):
        self._seed()
        updated = self.store.record_routing_result(
            "http://1.1.1.1:8080", ok=False, reason="rst")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.consecutive_failures, 1)
        self.assertGreater(updated.backoff_until, time.time())
        # still in the store file (failure history kept), but not routable
        self.assertEqual(self.store.pool(), [])
        raw = json.loads((self.store.working_json).read_text())
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["consecutive_failures"], 1)

    def test_record_routing_success_samples_latency(self):
        self._seed()
        updated = self.store.record_routing_result(
            "http://1.1.1.1:8080", ok=True, latency_ms=90.0)
        self.assertEqual(updated.consecutive_failures, 0)
        self.assertEqual(len(self.store.pool()), 1)

    def test_unknown_url_returns_none(self):
        self.assertIsNone(self.store.record_routing_result(
            "http://9.9.9.9:9999", ok=False, reason="x"))


# ── domain affinity ──────────────────────────────────────────────────────────


class AffinityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.aff = PL.DomainAffinityManager(f"{self.tmp}/affinity.json")
        self.store = PL.ProxyStore(f"{self.tmp}/pool")

    def _live_proxy(self, url="http://1.1.1.1:8080"):
        p = PL.Proxy(host="1.1.1.1", port=8080, scheme="http",
                     anonymity="elite", alive=True, tested_at=time.time())
        p.record_latency(100.0)
        self.store.save([p])
        return p

    def test_bind_and_lookup(self):
        self._live_proxy()
        self.aff.bind("example.com", "http://1.1.1.1:8080")
        self.assertEqual(self.aff.lookup("https://example.com/path",
                                         store=self.store),
                         "http://1.1.1.1:8080")

    def test_lookup_drops_dead_proxy(self):
        p = self._live_proxy()
        self.aff.bind("example.com", p.url)
        self.store.record_routing_result(p.url, ok=False, reason="dead")
        self.assertEqual(self.aff.lookup("example.com", store=self.store),
                         "")
        # binding was dropped, not left dangling
        self.assertEqual(self.aff.bindings(), [])

    def test_lookup_drops_backed_off_proxy(self):
        p = self._live_proxy()
        self.aff.bind("example.com", p.url)
        p2 = self.store.by_url(p.url)
        p2.record_failure("x", backoff_seconds=600.0)
        self.store.save([p2])
        self.assertEqual(self.aff.lookup("example.com", store=self.store),
                         "")

    def test_ttl_expiry(self):
        self._live_proxy()
        self.aff.bind("example.com", "http://1.1.1.1:8080", ttl_hours=0.5)
        # age the binding past its TTL
        self.aff._data["example.com"]["bound_at"] -= 3600.0
        self.assertEqual(self.aff.lookup("example.com", store=self.store),
                         "")

    def test_release(self):
        self.aff.bind("example.com", "http://1.1.1.1:8080")
        self.assertTrue(self.aff.release("example.com"))
        self.assertFalse(self.aff.release("example.com"))

    def test_domain_normalization(self):
        self.aff.bind("https://WWW.Example.COM/a?b=c",
                      "http://1.1.1.1:8080")
        bindings = self.aff.bindings()
        self.assertEqual(bindings[0]["domain"], "www.example.com")


# ── protocol-aware picks ─────────────────────────────────────────────────────


def _ctx_no_db():
    return types.SimpleNamespace(settings=types.SimpleNamespace(home="/tmp"))


class PickTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        ctx = types.SimpleNamespace(
            settings=types.SimpleNamespace(home=self.tmp), db=None)
        self.lab = PL.ProxyLab(ctx)
        now = time.time()
        proxies = [
            PL.Proxy(host="10.0.0.1", port=8080, scheme="http",
                     anonymity="elite", country="NG", alive=True,
                     tested_at=now),
            PL.Proxy(host="10.0.0.2", port=1080, scheme="socks5",
                     anonymity="anonymous", country="DE", alive=True,
                     tested_at=now),
            PL.Proxy(host="10.0.0.3", port=1080, scheme="socks5",
                     anonymity="elite", country="US", alive=True,
                     tested_at=now),
        ]
        for p in proxies:
            p.record_latency(200.0, at=now)
        self.lab.store.save(proxies)
        self.rot = PL.ProxyRotationManager(ctx)

    def test_needs_udp_picks_socks5_only(self):
        res = self.rot.pick_for_job(needs_udp=True)
        self.assertTrue(res["proxy"].startswith("socks5://"), res)
        self.assertIn("UDP", res["why"])

    def test_country_filter(self):
        res = self.rot.pick_for_job(country="NG")
        self.assertIn("10.0.0.1", res["proxy"])

    def test_min_anonymity_filters(self):
        res = self.rot.pick_for_job(min_anonymity="elite")
        self.assertNotIn("10.0.0.2", res["proxy"])  # anonymous is out

    def test_needs_auth_is_honest(self):
        res = self.rot.pick_for_job(needs_auth=True)
        self.assertEqual(res["proxy"], "")
        self.assertIn("no authenticated", res["note"].lower())

    def test_empty_pool_guidance_names_refresh(self):
        self.lab.store.clear()
        res = self.rot.pick_for_job()
        self.assertEqual(res["proxy"], "")
        self.assertIn("/proxy refresh", res["note"])

    def test_target_pins_affinity(self):
        res = self.rot.pick_for_job(target="https://example.com/x")
        self.assertTrue(res["proxy"])
        aff = PL.DomainAffinityManager(self.lab.store.dir / "affinity.json")
        self.assertEqual(aff.lookup("example.com", store=self.lab.store),
                         res["proxy"])


# ── health dashboard ─────────────────────────────────────────────────────────


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        ctx = types.SimpleNamespace(
            settings=types.SimpleNamespace(home=self.tmp), db=None)
        self.lab = PL.ProxyLab(ctx)

    def test_empty_pool_guidance(self):
        h = self.lab.health()
        self.assertEqual(h["pool"]["routable"], 0)
        self.assertIn("/proxy refresh", h["guidance"])
        self.assertEqual(h["tap"], "/proxy refresh")

    def test_populated_dashboard_counts(self):
        now = time.time()
        proxies = [
            PL.Proxy(host="10.0.0.1", port=8080, scheme="http",
                     anonymity="elite", country="NG", alive=True,
                     tested_at=now),
            PL.Proxy(host="10.0.0.2", port=1080, scheme="socks5",
                     anonymity="anonymous", country="DE", alive=True,
                     tested_at=now),
            PL.Proxy(host="10.0.0.3", port=3128, scheme="http",
                     anonymity="transparent", country="US", alive=True,
                     tested_at=now),
            PL.Proxy(host="10.0.0.4", port=8080, scheme="http",
                     alive=False, tested_at=now),  # dead record
        ]
        for p in proxies[:3]:
            p.record_latency(150.0, at=now)
        self.lab.store.save(proxies)
        h = self.lab.health()
        pool = h["pool"]
        self.assertEqual(pool["routable"], 3)
        self.assertEqual(pool["dead"], 1)
        self.assertEqual(pool["by_scheme"], {"http": 2, "socks5": 1})
        self.assertEqual(pool["by_anonymity"]["elite"], 1)
        self.assertEqual(pool["top_countries"]["NG"], 1)
        self.assertEqual(pool["ng_or_eu"], 2)  # NG + DE
        self.assertGreater(pool["avg_latency_ms"], 0)
        self.assertNotIn("guidance", h)

    def test_backoff_counted_not_routable(self):
        now = time.time()
        p = PL.Proxy(host="10.0.0.1", port=8080, scheme="http",
                     anonymity="elite", alive=True, tested_at=now)
        p.record_latency(100.0, at=now)
        p.record_failure("rst", backoff_seconds=600.0, at=now)
        self.lab.store.save([p])
        h = self.lab.health()
        self.assertEqual(h["pool"]["routable"], 0)
        self.assertEqual(h["pool"]["in_backoff"], 1)
        self.assertIn("backoff", h["guidance"])


# ── parser fixes ─────────────────────────────────────────────────────────────


class ParserFixTests(unittest.TestCase):
    def test_display_none_port_poison_stripped(self):
        html = (
            "<table>"
            "<tr>"
            '<td><a href="/1.2.3.4/8080#http">1.2.3.4</a></td>'
            '<td><div style="display:none">12</div>'
            '<a href="/1.2.3.4/8080#http">8080</a></td>'
            "<td>HTTP</td>"
            '<td><abbr title="Nigeria">NG</abbr></td>'
            "</tr>"
            "<tr>"
            '<td><a href="/5.6.7.8/1090#socks5">5.6.7.8</a></td>'
            '<td><div style="display:none">99</div>'
            '<a href="/5.6.7.8/1090#socks5">1090</a></td>'
            "<td>SOCKS5</td>"
            '<td><abbr title="Germany">DE</abbr></td>'
            "</tr>"
            "</table>")
        out = PL.ProxyScraper.parse_html(html)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].port, 8080)   # was 1280 (corrupted)
        self.assertEqual(out[1].port, 1090)   # was dropped entirely
        self.assertEqual(out[1].scheme, "socks5")
        self.assertEqual(out[0].country, "NG")

    def test_claimed_anonymity_parsed(self):
        html = (
            "<table><tr>"
            "<td>1.2.3.4</td><td>8080</td><td>HTTP</td>"
            "<td>US</td>"
            '<td><span title="does not reveal IP">High Anonymous</span></td>'
            "</tr></table>")
        out = PL.ProxyScraper.parse_html(html)
        self.assertEqual(out[0].claimed_anonymity, "elite")
        # the claim never masquerades as a lab test
        self.assertEqual(out[0].anonymity, "")

    def test_catalog_new_sources(self):
        from nomorals.tools import proxysources as PS
        by_name = {n: (u, k) for n, u, k in PS.BUILT_IN_SOURCES}
        self.assertEqual(by_name["proxydb-http"][1], "html")
        self.assertEqual(by_name["freeproxyworld-socks5"][1], "html")
        self.assertEqual(by_name["proxydb-socks5"][1], "html")  # kept
        names = [n for n, _, _ in PS.BUILT_IN_SOURCES]
        self.assertEqual(len(names), len(set(names)))


# ── rotation strategies + tool registration ──────────────────────────────────


class RotationStrategyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ctx = types.SimpleNamespace(
            settings=types.SimpleNamespace(home=self.tmp), db=None)
        now = time.time()
        proxies = [PL.Proxy(host=f"10.0.0.{i}", port=8080, scheme="http",
                            anonymity="elite", alive=True, tested_at=now)
                   for i in range(1, 4)]
        for p in proxies:
            p.record_latency(100.0, at=now)
        PL.ProxyLab(self.ctx).store.save(proxies)
        self.rot = PL.ProxyRotationManager(self.ctx)
        self.rot.state["strategy"] = "round_robin"

    def test_round_robin_cycles(self):
        picks = [self.rot.pick() for _ in range(4)]
        self.assertEqual(len(set(picks[:3])), 3)
        self.assertEqual(picks[3], picks[0])

    def test_report_error_demotes_with_backoff(self):
        first = self.rot.pick()
        self.rot.report_error(first, "rst")
        # the offender is cooling; next picks avoid it
        for _ in range(6):
            self.assertNotEqual(self.rot.pick(), first)

    def test_sticky_fails_over_on_error(self):
        self.rot.state["strategy"] = "sticky"
        first = self.rot.pick()
        self.assertEqual(self.rot.pick(), first)
        self.rot.report_error(first, "rst")
        self.assertNotEqual(self.rot.pick(), first)

    def test_pick_schemes_filter(self):
        self.assertEqual(self.rot.pick(schemes=("socks5",)), "")


class ToolRegistrationTests(unittest.TestCase):
    def test_new_tools_register(self):
        from nomorals.tools.registry import ToolRegistry
        r = ToolRegistry()
        PL.register(r)
        names = set(r._tools)
        for expected in ["proxy_health", "proxy_pick", "proxy_affinity",
                         "proxy_lab_status", "proxy_pool", "proxy_rotate",
                         "proxy_schedule", "proxy_refresh", "proxy_scrape"]:
            self.assertIn(expected, names, expected)


class ProfileGatingTests(unittest.TestCase):
    def test_worker_counts_per_profile(self):
        self.assertEqual(PL.profile_workers("termux", "test"), 4)
        self.assertEqual(PL.profile_workers("termux", "scrape"), 3)
        self.assertEqual(PL.profile_workers("laptop", "test"), 8)
        self.assertEqual(PL.profile_workers("workstation", "test"), 12)
        self.assertEqual(PL.profile_workers("workstation", "scrape"), 8)
        # unknown profile → workstation defaults (full capability)
        self.assertEqual(PL.profile_workers("toaster", "test"), 12)

    def test_scraper_tester_honor_profile(self):
        self.assertEqual(PL.ProxyScraper(profile="termux").max_workers, 3)
        self.assertEqual(PL.ProxyTester(profile="termux").max_workers, 4)
        self.assertEqual(PL.ProxyScraper(profile="laptop").max_workers, 6)
        # explicit max_workers still wins over the profile
        self.assertEqual(
            PL.ProxyScraper(max_workers=8, profile="termux").max_workers, 8)

    def test_detect_profile_termux_heuristic(self):
        import os
        old = os.environ.get("PREFIX")
        try:
            os.environ["PREFIX"] = "/data/data/com.termux/files/usr"
            ctx = types.SimpleNamespace(settings=None)
            self.assertEqual(PL.detect_profile(ctx), "termux")
        finally:
            if old is None:
                os.environ.pop("PREFIX", None)
            else:
                os.environ["PREFIX"] = old

    def test_new_catalog_sources_scheme_filter(self):
        # the scraper's scheme filter keys off the name suffix
        self.assertEqual(
            PL.ProxyScraper._source_scheme("proxydb-http", "html"), "http")
        self.assertEqual(
            PL.ProxyScraper._source_scheme("freeproxyworld-socks5", "html"),
            "socks5")


if __name__ == "__main__":
    unittest.main()
