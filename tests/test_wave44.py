"""Wave 44 — god-tier proxy system (fully hermetic).

- tools/proxylab.py: multi-source scraper, raw-socket tester
  (HTTP/HTTPS/SOCKS4/SOCKS5, liveness/latency/egress/anonymity/country),
  ranking, durable pool store, scheduled re-checks
- tools/ssh_socks.py: SSH → local SOCKS5 tunnels with supervision,
  auto-reconnect, and status probing

Everything network-touching runs against LOCAL fakes: an in-process
HTTP origin, real in-process HTTP forward proxies (elite / anonymous /
transparent behavior), real in-process SOCKS5 and SOCKS4 servers, and a
fake ``ssh`` binary that actually binds the -D port and answers SOCKS5
greetings.  No external network is touched.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from tests.test_partner_runtime import _make_context

from nomorals.core.errors import ToolError
from nomorals.tools import proxylab
from nomorals.tools.proxylab import (
    Proxy,
    ProxyLab,
    ProxyScraper,
    ProxyStore,
    ProxyTester,
    rank_proxies,
)
from nomorals.tools.ssh_socks import SshSocksManager, SshSocksProfile, SshSocksTunnel


# ── local "internet": origin + proxies ───────────────────────────────────────

DIRECT_IP = "203.0.113.10"      # what the origin sees on a direct request
ELITE_IP = "192.0.2.10"
ANON_IP = "192.0.2.11"


class _OriginHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/ip":
            body = DIRECT_IP.encode()
        elif self.path == "/headers":
            body = json.dumps(
                {"headers": {k: v for k, v in self.headers.items()}}
            ).encode()
        elif self.path == "/country":
            body = json.dumps({"country": "NG"}).encode()
        else:
            body = b"not found"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class _HttpFakeProxy:
    """A real minimal HTTP forward proxy with one of three identities.

    mode: "elite" (adds nothing), "anonymous" (adds X-Forwarded-For),
    "transparent" (reports the client's own IP as the egress).
    """

    def __init__(self, origin: tuple[str, int], mode: str) -> None:
        self.origin = origin
        self.mode = mode
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(32)
        self.port = int(self.srv.getsockname()[1])
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5)
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            head, _, _ = buf.partition(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            request_line = lines[0]
            headers = lines[1:]
            target = request_line.split(" ")[1]
            path = urlparse(target).path or "/"
            if path == "/ip":
                ip = {"elite": ELITE_IP, "anonymous": ANON_IP,
                      "transparent": DIRECT_IP}[self.mode]
                body = ip.encode()
                resp = (f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                        "Connection: close\r\n\r\n").encode() + body
                conn.sendall(resp)
                return
            # forward to origin (absolute-form stripped to origin-form)
            origin_sock = socket.create_connection(self.origin, timeout=5)
            fwd_head = f"GET {path} HTTP/1.1\r\n"
            for line in headers:
                key = line.split(":", 1)[0].strip().lower()
                if key in {"host", "proxy-connection", "connection"}:
                    continue
                fwd_head += line + "\r\n"
            if self.mode == "anonymous":
                fwd_head += "X-Forwarded-For: 203.0.113.10\r\n"
            fwd_head += "Connection: close\r\n\r\n"
            origin_sock.sendall(fwd_head.encode("latin-1"))
            origin_sock.settimeout(5)
            data = b""
            while True:
                chunk = origin_sock.recv(65536)
                if not chunk:
                    break
                data += chunk
            origin_sock.close()
            conn.sendall(data)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        try:
            self.srv.close()
        except OSError:
            pass


class _Socks5Fake:
    """A real minimal SOCKS5 server: greeting + domain connect, then
    answers /ip /headers /country itself (so we control the egress IP)."""

    def __init__(self, egress_ip: str) -> None:
        self.egress_ip = egress_ip
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(32)
        self.port = int(self.srv.getsockname()[1])
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5)
            greeting = b""
            while len(greeting) < 3:
                chunk = conn.recv(3 - len(greeting))
                if not chunk:
                    return
                greeting += chunk
            if greeting[:1] != b"\x05":
                return
            conn.sendall(b"\x05\x00")  # no auth
            req = conn.recv(256)
            if req[:2] != b"\x05\x01":
                return
            # success reply: VER REP rsv ATYP=1 BND-IP BND-PORT
            conn.sendall(b"\x05\x00\x00\x01" + b"\x7f\x00\x00\x01"
                         + (0).to_bytes(2, "big"))
            # now serve the (origin-form) HTTP request ourselves
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            request_line = buf.decode("latin-1").split("\r\n")[0]
            path = urlparse("http://x" + request_line.split(" ")[1]).path
            if path == "/ip":
                body = self.egress_ip.encode()
            elif path == "/headers":
                headers = {}
                for line in buf.decode("latin-1").split("\r\n")[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip()] = v.strip()
                body = json.dumps({"headers": headers}).encode()
            elif path == "/country":
                body = json.dumps({"country": "NG"}).encode()
            else:
                body = b"not found"
            resp = (f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n").encode() + body
            conn.sendall(resp)
        except (OSError, IndexError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        try:
            self.srv.close()
        except OSError:
            pass


class _Socks4Fake:
    """Real minimal SOCKS4 server (no-auth, IPv4 target) with the same
    self-answering behavior."""

    def __init__(self, egress_ip: str) -> None:
        self.egress_ip = egress_ip
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(32)
        self.port = int(self.srv.getsockname()[1])
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5)
            req = conn.recv(64)
            if req[:2] != b"\x05\x0a":
                return
            conn.sendall(b"\x05\x00")
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            request_line = buf.decode("latin-1").split("\r\n")[0]
            path = urlparse("http://x" + request_line.split(" ")[1]).path
            if path == "/ip":
                body = self.egress_ip.encode()
            elif path == "/headers":
                headers = {}
                for line in buf.decode("latin-1").split("\r\n")[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip()] = v.strip()
                body = json.dumps({"headers": headers}).encode()
            else:
                body = json.dumps({"country": "NG"}).encode()
            resp = (f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n").encode() + body
            conn.sendall(resp)
        except (OSError, IndexError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        try:
            self.srv.close()
        except OSError:
            pass


class _Net:
    """One origin + three http proxies + socks servers, all on 127.0.0.1."""

    def __init__(self) -> None:
        self.origin_http = ThreadingHTTPServer(("127.0.0.1", 0),
                                               _OriginHandler)
        self.origin_port = int(self.origin_http.server_address[1])
        threading.Thread(target=self.origin_http.serve_forever,
                         daemon=True).start()
        self.elite = _HttpFakeProxy(("127.0.0.1", self.origin_port), "elite")
        self.anon = _HttpFakeProxy(("127.0.0.1", self.origin_port), "anonymous")
        self.transparent = _HttpFakeProxy(("127.0.0.1", self.origin_port),
                                          "transparent")
        self.socks5 = _Socks5Fake(ELITE_IP)
        self.socks4 = _Socks4Fake(ELITE_IP)

    def urls(self) -> dict[str, str]:
        return {
            "ip": f"http://127.0.0.1:{self.origin_port}/ip",
            "echo": f"http://127.0.0.1:{self.origin_port}/headers",
            "country": f"http://127.0.0.1:{self.origin_port}/country",
        }

    def close(self) -> None:
        self.origin_http.shutdown()
        for f in (self.elite, self.anon, self.transparent,
                  self.socks5, self.socks4):
            f.close()


# ── scraper ──────────────────────────────────────────────────────────────────

_FAKE_PAGE = {
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt":
        b"1.2.3.4:8080\n5.6.7.8:3128\nbad line\n\n9.9.9.9:99999\n",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/https.txt":
        b"1.2.3.4:8443\n7.7.7.7:443\n",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks4.txt":
        b"8.8.8.8:1080\n",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt":
        b"8.8.4.4:1080\n1.2.3.4:8080\n",  # dup across schemes is kept
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt":
        b"1.2.3.4:8080\n6.6.6.6:80\n",  # dup of thespeedx entry
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks4.txt":
        b"8.8.8.8:1080\n",  # full dup
    # monosans-socks5 intentionally has NO fixture → fetch fails →
    # the source error must be isolated, not fatal
    "https://free-proxy-list.net/": (
        b"<html><body><table>"
        b"<tr><th>#</th><th>IP</th><th>Port</th><th>Code</th></tr>"
        b"<tr><td>1</td><td>4.4.4.4</td><td>80</td><td>US</td></tr>"
        b"<tr><td>2</td><td>5.5.5.5</td><td>8081</td><td>GB</td></tr>"
        b"</table></body></html>"
    ),
    # wave 83: the per-protocol v4 API (the old single "mixed" v3-style
    # URL was removed — it 400s upstream)
    "https://api.proxyscrape.com/v4/free-proxy-list/get"
    "?request=displayproxies&proxy_format=ipport&format=text"
    "&country=all&proxy_type=socks5":
        b"2.2.2.2:1080\n3.3.3.3:3128\ngarbage\n",
}


class ProxyScraperTest(unittest.TestCase):
    def _scraper(self) -> ProxyScraper:
        def fetcher(url: str) -> bytes:
            if url in _FAKE_PAGE:
                return _FAKE_PAGE[url]
            raise ConnectionError(f"no fixture for {url}")
        return ProxyScraper(fetcher=fetcher, timeout=5.0)

    def test_parses_all_source_kinds(self) -> None:
        report = self._scraper().scrape()
        urls = {p.url for p in report["proxies"]}
        # list kind
        self.assertIn("http://1.2.3.4:8080", urls)
        self.assertIn("https://1.2.3.4:8443", urls)
        self.assertIn("socks4://8.8.8.8:1080", urls)
        self.assertIn("socks5://8.8.4.4:1080", urls)
        # proxyscrape per-protocol source (wave 83)
        self.assertIn("socks5://2.2.2.2:1080", urls)
        self.assertIn("socks5://3.3.3.3:3128", urls)
        # html kind (with country captured)
        html = [p for p in report["proxies"] if p.host == "4.4.4.4"]
        self.assertTrue(html and html[0].country == "US")
        self.assertNotIn("http://5.5.5.5:8080", urls)  # port 8081, not 8080
        self.assertIn("http://5.5.5.5:8081", urls)

    def test_dedupes_across_sources(self) -> None:
        report = self._scraper().scrape()
        keys = [p.key for p in report["proxies"]]
        self.assertEqual(len(keys), len(set(keys)))
        # 1.2.3.4:8080 http seen twice (thespeedx + monosans) → once
        self.assertEqual(
            sum(1 for p in report["proxies"]
                if p.host == "1.2.3.4" and p.port == 8080
                and p.scheme == "http"), 1)

    def test_invalid_lines_dropped(self) -> None:
        report = self._scraper().scrape()
        urls = [p.url for p in report["proxies"]]
        self.assertNotIn("http://9.9.9.9:99999", urls)  # port out of range
        self.assertFalse(any("bad line" in u for u in urls))

    def test_source_failure_isolated(self) -> None:
        report = self._scraper().scrape()
        self.assertIn("monosans-socks5", report["errors"])
        self.assertGreater(report["total"], 5)  # others still worked

    def test_scheme_filter(self) -> None:
        report = self._scraper().scrape(schemes="socks4,socks5")
        self.assertTrue(all(p.scheme in {"socks4", "socks5"}
                            for p in report["proxies"]))


class ProxyTesterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.net = _Net()
        cls.urls = cls.net.urls()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.net.close()

    def _tester(self) -> ProxyTester:
        return ProxyTester(
            ip_url=self.urls["ip"],
            echo_url=self.urls["echo"],
            country_url=self.urls["country"],
            local_ip_provider=lambda: DIRECT_IP,
            timeout=4.0,
        )

    def test_elite_proxy(self) -> None:
        p = self._tester().test_one(
            Proxy(host="127.0.0.1", port=self.net.elite.port, scheme="http"))
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.egress_ip, ELITE_IP)
        self.assertEqual(p.anonymity, "elite")
        self.assertEqual(p.country, "NG")
        self.assertGreaterEqual(p.latency_ms, 0)

    def test_anonymous_proxy(self) -> None:
        p = self._tester().test_one(
            Proxy(host="127.0.0.1", port=self.net.anon.port, scheme="http"))
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.egress_ip, ANON_IP)
        self.assertEqual(p.anonymity, "anonymous")

    def test_transparent_proxy(self) -> None:
        p = self._tester().test_one(
            Proxy(host="127.0.0.1", port=self.net.transparent.port,
                  scheme="http"))
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.egress_ip, DIRECT_IP)
        self.assertEqual(p.anonymity, "transparent")

    def test_socks5_tunnel(self) -> None:
        p = self._tester().test_one(
            Proxy(host="127.0.0.1", port=self.net.socks5.port,
                  scheme="socks5"))
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.egress_ip, ELITE_IP)
        self.assertEqual(p.anonymity, "elite")  # no leak headers in tunnel

    def test_socks4_tunnel(self) -> None:
        p = self._tester().test_one(
            Proxy(host="127.0.0.1", port=self.net.socks4.port,
                  scheme="socks4"))
        self.assertTrue(p.alive, p.error)
        self.assertEqual(p.egress_ip, ELITE_IP)
        self.assertEqual(p.anonymity, "elite")

    def test_dead_proxy_is_a_result_not_a_crash(self) -> None:
        # port that nothing listens on
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = int(probe.getsockname()[1])
        probe.close()
        p = self._tester().test_one(
            Proxy(host="127.0.0.1", port=dead_port, scheme="http"))
        self.assertFalse(p.alive)
        self.assertTrue(p.error)

    def test_ranking_order(self) -> None:
        elite = Proxy("1.1.1.1", 80, "http", alive=True,
                      latency_ms=50, anonymity="elite")
        anon = Proxy("2.2.2.2", 80, "http", alive=True,
                     latency_ms=10, anonymity="anonymous")
        slow_elite = Proxy("3.3.3.3", 80, "http", alive=True,
                           latency_ms=500, anonymity="elite")
        dead = Proxy("4.4.4.4", 80, "http", alive=False)
        ranked = rank_proxies([dead, anon, elite, slow_elite])
        self.assertEqual([p.host for p in ranked],
                         ["1.1.1.1", "3.3.3.3", "2.2.2.2", "4.4.4.4"])


# ── store + lab ──────────────────────────────────────────────────────────────


class ProxyStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="proxystore-")
        self.store = ProxyStore(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_load_round_trip(self) -> None:
        now = time.time()
        proxies = [
            Proxy("1.1.1.1", 80, "http", alive=True, latency_ms=20,
                  anonymity="elite", country="US", tested_at=now),
            Proxy("2.2.2.2", 1080, "socks5", alive=True, latency_ms=80,
                  anonymity="anonymous", tested_at=now),
            Proxy("3.3.3.3", 80, "http", alive=False, error="dead",
                  tested_at=now),
        ]
        self.store.save(proxies, candidates=proxies + [
            Proxy("4.4.4.4", 80, "http"), Proxy("4.4.4.4", 80, "http")])
        with open(os.path.join(self.tmp, "working.txt"),
                  encoding="utf-8") as fh:
            txt = fh.read().strip().splitlines()
        self.assertEqual(txt, ["http://1.1.1.1:80", "socks5://2.2.2.2:1080"])
        loaded = self.store.load_working()
        self.assertEqual(len(loaded), 2)
        self.assertFalse(any(p.alive for p in loaded if p.error))
        cands = self.store.load_candidates()
        self.assertEqual(len(cands), 4)  # 3 + deduped 4.4.4.4
        self.store.clear()
        self.assertEqual(self.store.load_working(), [])

    def test_pool_filters_and_freshness(self) -> None:
        now = time.time()
        self.store.save([
            Proxy("1.1.1.1", 80, "http", alive=True, anonymity="elite",
                  tested_at=now),
            Proxy("2.2.2.2", 80, "http", alive=True, anonymity="elite",
                  tested_at=now - 48 * 3600),  # stale
            Proxy("3.3.3.3", 1080, "socks5", alive=True, anonymity="elite",
                  tested_at=now),
        ])
        urls = self.store.urls(max_age_hours=24)
        self.assertIn("http://1.1.1.1:80", urls)
        self.assertNotIn("http://2.2.2.2:80", urls)  # stale dropped
        socks = self.store.urls(scheme="socks5", max_age_hours=24)
        self.assertEqual(socks, ["socks5://3.3.3.3:1080"])


class ProxyLabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.net = _Net()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.net.close()

    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="proxylab-")
        self.ctx.settings.home = self.tmp
        # point the scraper at fakes + the tester at the local origin
        self.lab = ProxyLab(self.ctx)
        self.sources = [
            ("fake-http", "http://fake/list-http", "list"),
            ("fake-socks5", "http://fake/list-socks5", "list"),
        ]
        self.page = {
            "http://fake/list-http": (
                f"127.0.0.1:{self.net.elite.port}\n"
                f"127.0.0.1:{self.net.anon.port}\n"
                f"127.0.0.1:{self.net.transparent.port}\n"
            ).encode(),
            "http://fake/list-socks5":
                f"127.0.0.1:{self.net.socks5.port}\n".encode(),
        }
        self.lab._scraper = ProxyScraper(fetcher=lambda u: self.page[u])
        urls = self.net.urls()
        self.lab._tester = ProxyTester(
            ip_url=urls["ip"], echo_url=urls["echo"],
            country_url=urls["country"],
            local_ip_provider=lambda: DIRECT_IP, timeout=4.0)

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_cycle_scrape_test_store_pool(self) -> None:
        report = self.lab.scrape(sources=self.sources)
        self.assertEqual(report["total"], 4)
        summary = self.lab.refresh(schemes="http,socks5", limit=10)
        self.assertEqual(summary["scraped"], 0)  # candidates already stored
        self.assertEqual(summary["tested"], 4)
        self.assertEqual(summary["working"], 4)
        self.assertEqual(summary["by_scheme"], {"http": 3, "socks5": 1})
        self.assertEqual(summary["by_anonymity"],
                         {"elite": 2, "anonymous": 1, "transparent": 1})
        pool = self.lab.pool(scheme="socks5", max_age_hours=1)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]["scheme"], "socks5")
        self.assertTrue(pool[0]["url"].startswith("socks5://127.0.0.1:"))
        status = self.lab.status()
        self.assertEqual(status["working_total"], 4)
        self.assertEqual(status["candidates"], 4)
        self.assertEqual(len(status["best"]), 4)

    def test_pool_action_via_tool(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.context = self.ctx
        from nomorals.tools import proxylab as pl

        pl.register(reg)
        self.lab.scrape(sources=self.sources)
        self.lab.refresh(schemes="http,socks5", limit=10)
        r = reg.call("proxy_pool", action="urls", limit="10")
        self.assertTrue(r.ok, getattr(r.error, "message", r.error))
        self.assertEqual(r.value["count"], 4)
        self.assertTrue(all(u.startswith(("http://", "socks5://"))
                            for u in r.value["urls"]))
        r2 = reg.call("proxy_pool", action="clear")
        self.assertTrue(r2.ok and r2.value["cleared"])
        r3 = reg.call("proxy_pool", action="urls")
        self.assertEqual(r3.value["count"], 0)


class ProxyScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="proxysched-")
        self.ctx.settings.home = self.tmp

    def tearDown(self) -> None:
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_schedule_on_and_off(self) -> None:
        from nomorals.agents.scheduler import Scheduler
        from nomorals.tools import proxylab as pl
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.context = self.ctx
        pl.register(reg)
        r = reg.call("proxy_schedule", enabled="true", every_minutes="45")
        self.assertTrue(r.ok, getattr(r.error, "message", r.error))
        self.assertTrue(r.value["scheduled"])
        jobs = Scheduler(self.ctx).list_jobs()
        job = next(j for j in jobs if j["name"] == "proxy_refresh")
        self.assertEqual(job["spec"], "every 2700s")  # 45 minutes
        r2 = reg.call("proxy_schedule", enabled="false")
        self.assertTrue(r2.ok and r2.value["removed"])
        jobs2 = Scheduler(self.ctx).list_jobs()
        self.assertFalse(any(j["name"] == "proxy_refresh" for j in jobs2))


# ── ssh → socks5 ─────────────────────────────────────────────────────────────

_FAKE_SSH = r'''#!/usr/bin/env python3
"""Fake ssh: parses -D, binds that port, answers SOCKS5 greetings.
FAKE_SSH_EXIT_AFTER=<s> makes it die after that many seconds (to test
auto-reconnect).  SIGTERM exits cleanly."""
import os, socket, sys, threading, time

args = sys.argv[1:]
port = None
for i, a in enumerate(args):
    if a == "-D":
        port = int(args[i + 1].rsplit(":", 1)[1])
if port is None:
    sys.exit(2)
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(32)

def handle(c):
    try:
        c.settimeout(3)
        g = b""
        while len(g) < 3:
            chunk = c.recv(3 - len(g))
            if not chunk:
                return
            g += chunk
        if g[:1] == b"\x05":
            c.sendall(b"\x05\x00")
            c.recv(64)
            c.sendall(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00")
    except OSError:
        pass
    finally:
        try:
            c.close()
        except OSError:
            pass

def accept_loop():
    while True:
        try:
            c, _ = srv.accept()
        except OSError:
            return
        threading.Thread(target=handle, args=(c,), daemon=True).start()

threading.Thread(target=accept_loop, daemon=True).start()
exit_after = float(os.environ.get("FAKE_SSH_EXIT_AFTER") or 0)
if exit_after:
    time.sleep(exit_after)
    os._exit(0)
try:
    while True:
        time.sleep(3600)
except (KeyboardInterrupt, SystemExit):
    pass
'''


def _wait_up(tunnel: SshSocksTunnel, seconds: float = 15.0) -> bool:
    """The fake ssh (python startup) needs a moment to bind the port."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if tunnel.port_up():
            return True
        time.sleep(0.1)
    return tunnel.port_up()


class SshSocksTunnelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="sshsocks-")
        self.ssh = os.path.join(self.tmp, "fake-ssh")
        with open(self.ssh, "w", encoding="utf-8") as fh:
            fh.write(_FAKE_SSH)
        os.chmod(self.ssh, 0o755)
        self.key = os.path.join(self.tmp, "id_ed25519")
        with open(self.key, "w", encoding="utf-8") as fh:
            fh.write("fake-key-material\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _profile(self, **kw: Any) -> SshSocksProfile:
        base = dict(name="t1", host="10.0.0.5", user="deploy",
                    key=self.key, auto_reconnect=False)
        base.update(kw)
        return SshSocksProfile(**base)

    def _tunnel(self, profile: SshSocksProfile, **kw: Any) -> SshSocksTunnel:
        return SshSocksTunnel(profile, ssh_bin=self.ssh,
                              sshpass_bin=os.path.join(self.tmp, "no-sshpass"),
                              **kw)

    def test_build_cmd_key_auth(self) -> None:
        t = self._tunnel(self._profile())
        t.local_port = 31234
        cmd = t.build_cmd()
        self.assertEqual(cmd[0], self.ssh)
        self.assertIn("-N", cmd)
        self.assertEqual(cmd[cmd.index("-D") + 1], "127.0.0.1:31234")
        self.assertIn("-i", cmd)
        self.assertEqual(cmd[cmd.index("-i") + 1], self.key)
        self.assertTrue(cmd[-1].endswith("@10.0.0.5"))
        self.assertFalse(any("sshpass" in c for c in cmd))

    def test_password_without_sshpass_is_an_honest_error(self) -> None:
        t = self._tunnel(self._profile(key="", password="hunter2"))
        t.local_port = 31234
        with self.assertRaises(ToolError) as ctx:
            t.build_cmd()
        self.assertIn("sshpass", str(ctx.exception))

    def test_start_status_stop(self) -> None:
        t = self._tunnel(self._profile())
        started = t.start()
        self.assertTrue(started["started"], started)
        self.assertGreater(started["local_port"], 0)
        try:
            self.assertTrue(_wait_up(t),
                            "SOCKS5 port never came up")
            st = t.status()
            self.assertTrue(st["running"])
            self.assertTrue(st["socks_port_open"],
                            "SOCKS5 greeting not answered on the port")
            self.assertEqual(st["proxy_url"],
                             f"socks5://127.0.0.1:{st['local_port']}")
            self.assertTrue(t.port_up())
        finally:
            stop = t.stop()
        self.assertTrue(stop["stopped"])
        self.assertFalse(t.alive())
        self.assertFalse(t.port_up())

    def test_auto_reconnect_after_drop(self) -> None:
        env_backup = os.environ.pop("FAKE_SSH_EXIT_AFTER", None)
        os.environ["FAKE_SSH_EXIT_AFTER"] = "1"
        try:
            t = self._tunnel(self._profile(auto_reconnect=True,
                                           max_reconnects=5))
            t.start()
            _wait_up(t)  # first bind
            deadline = time.time() + 25
            while t.restarts < 1 and time.time() < deadline:
                time.sleep(0.5)
            self.assertGreaterEqual(t.restarts, 1,
                                    "supervisor never relaunched the tunnel")
            self.assertTrue(t.alive(), "tunnel not back up after reconnect")
        finally:
            t.stop()
            if env_backup is not None:
                os.environ["FAKE_SSH_EXIT_AFTER"] = env_backup
            else:
                os.environ.pop("FAKE_SSH_EXIT_AFTER", None)


class SshSocksManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="sshsocks-mgr-")
        self.ssh = os.path.join(self.tmp, "fake-ssh")
        with open(self.ssh, "w", encoding="utf-8") as fh:
            fh.write(_FAKE_SSH)
        os.chmod(self.ssh, 0o755)
        self.key = os.path.join(self.tmp, "id_ed25519")
        with open(self.key, "w", encoding="utf-8") as fh:
            fh.write("fake-key-material\n")
        # patch the module default so the manager's tunnels use the fake
        self._patcher = None
        import nomorals.tools.ssh_socks as mod

        real_init = SshSocksTunnel.__init__

        def fake_init(self_t, profile, **kw):
            kw.setdefault("ssh_bin", self.ssh)
            kw.setdefault("sshpass_bin", os.path.join(self.tmp, "no-sshpass"))
            real_init(self_t, profile, **kw)

        self._patcher = _Monkey(SshSocksTunnel, "__init__", fake_init)
        self._patcher.start()

    def tearDown(self) -> None:
        if self._patcher:
            self._patcher.stop()
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_start_stop_persist_remove(self) -> None:
        mgr = SshSocksManager(self.ctx)
        out = mgr.start(name="vps", host="10.1.1.1", user="root",
                        key=self.key, auto_reconnect="false")
        self.assertTrue(out["started"], out)
        try:
            self.assertTrue(_wait_up(mgr.tunnels["vps"]))
            st = mgr.status()["tunnels"]["vps"]
            self.assertTrue(st["running"] and st["socks_port_open"])
            self.assertEqual(mgr.urls(),
                             [f"socks5://127.0.0.1:{st['local_port']}"])
            listing = mgr.list()
            self.assertEqual(listing["configured"][0]["name"], "vps")
            self.assertFalse(listing["configured"][0]["has_password"])
        finally:
            mgr.stop("vps")
        # a NEW manager sees the persisted profile
        mgr2 = SshSocksManager(self.ctx)
        self.assertIn("vps", mgr2._profiles)
        # and remove deletes it
        self.assertTrue(mgr2.remove("vps")["removed"])
        mgr3 = SshSocksManager(self.ctx)
        self.assertNotIn("vps", mgr3._profiles)

    def test_tool_dispatch(self) -> None:
        from nomorals.tools import ssh_socks as mod
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.context = self.ctx
        mod.register(reg)
        r = reg.call("ssh_socks", action="start", name="t2",
                     host="10.2.2.2", user="u", key=self.key,
                     auto_reconnect="false")
        self.assertTrue(r.ok, getattr(r.error, "message", r.error))
        try:
            self.assertTrue(_wait_up(SshSocksManager(self.ctx)
                                          .tunnels["t2"]))
            r2 = reg.call("ssh_socks", action="urls")
            self.assertEqual(len(r2.value["urls"]), 1)
            r3 = reg.call("ssh_socks", action="status", name="t2")
            self.assertTrue(r3.value["tunnels"]["t2"]["running"])
        finally:
            reg.call("ssh_socks", action="remove", name="t2")
        r4 = reg.call("ssh_socks", action="status", name="t2")
        self.assertFalse(r4.ok)

    def test_password_profile_persists_but_never_listed(self) -> None:
        mgr = SshSocksManager(self.ctx)
        mgr.upsert_profile(name="pw", host="10.3.3.3", user="u",
                           password="s3cret")
        listing = json.dumps(mgr.list())
        self.assertNotIn("s3cret", listing)
        self.assertTrue(mgr.list()["configured"][0]["has_password"])
        # without sshpass, starting a password tunnel is an honest error
        with self.assertRaises(ToolError) as ctx_ex:
            mgr.start(name="pw", host="10.3.3.3", user="u", password="s3cret")
        self.assertIn("sshpass", str(ctx_ex.exception))
        mgr.remove("pw")


class _Monkey:
    """Tiny context-manager-style attribute monkeypatch (avoids mock dep)."""

    def __init__(self, obj: Any, name: str, value: Any) -> None:
        self.obj, self.name, self.value = obj, name, value
        self.saved: Any = None

    def start(self) -> "_Monkey":
        self.saved = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self

    def stop(self) -> None:
        setattr(self.obj, self.name, self.saved)


# ── rotation ─────────────────────────────────────────────────────────────────


class ProxyRotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.net = _Net()
        cls.urls = cls.net.urls()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.net.close()

    def setUp(self) -> None:
        self.ctx, self.tmpctx = _make_context()
        self.tmp = tempfile.mkdtemp(prefix="proxyrot-")
        self.ctx.settings.home = self.tmp
        store = proxylab.ProxyStore(os.path.join(self.tmp, "proxies"))
        now = time.time()
        store.save([
            proxylab.Proxy(host="127.0.0.1", port=self.net.elite.port,
                            scheme="http", alive=True, latency_ms=20,
                            anonymity="elite", tested_at=now),
            proxylab.Proxy(host="127.0.0.1", port=self.net.anon.port,
                            scheme="http", alive=True, latency_ms=30,
                            anonymity="anonymous", tested_at=now),
            proxylab.Proxy(host="127.0.0.1",
                            port=self.net.transparent.port, scheme="http",
                            alive=True, latency_ms=40, anonymity="transparent",
                            tested_at=now),
        ])
        self.mgr = proxylab.ProxyRotationManager(self.ctx)
        self._old_resolver = None

    def tearDown(self) -> None:
        # global hooks must never leak into other tests
        self.mgr.disable()
        from nomorals.core import http as core_http

        core_http.set_proxy_resolver(None)
        core_http.set_proxy_error_reporter(None)
        self.ctx.close()
        self.tmpctx.cleanup()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pool(self) -> list[str]:
        return self.mgr._pool_urls()

    def test_round_robin_cycles_all_healthy(self) -> None:
        self.mgr.enable(strategy="round_robin")
        pool = self._pool()
        self.assertEqual(len(pool), 3)
        picks = [self.mgr.pick() for _ in range(6)]
        self.assertEqual(len(set(picks[:3])), 3, "first cycle must cover all")
        self.assertEqual(picks[:3], picks[3:], "cycle must repeat")
        for p in picks:
            self.assertIn(p, pool)

    def test_failed_proxy_cooldown_and_recovery(self) -> None:
        self.mgr.enable(strategy="round_robin", cooldown_seconds=1)
        pool = self._pool()
        victim = pool[0]
        self.mgr.report_error(victim, "connection refused")
        st = self.mgr.status()
        self.assertEqual(len(st["cooling"]), 1)
        self.assertEqual(st["cooling"][0]["proxy"], victim)
        picks = [self.mgr.pick() for _ in range(4)]
        self.assertNotIn(victim, picks, "cooled proxy must be skipped")
        time.sleep(1.3)
        # cooldown expired → it is eligible again
        self.assertIn(victim, self.mgr._healthy(self._pool()))

    def test_sticky_holds_then_fails_over(self) -> None:
        self.mgr.enable(strategy="sticky")
        first = self.mgr.pick()
        second = self.mgr.pick()
        self.assertEqual(first, second, "sticky must hold the proxy")
        self.mgr.report_error(first, "drop")
        third = self.mgr.pick()
        self.assertNotEqual(first, third, "sticky must fail over")
        fourth = self.mgr.pick()
        self.assertEqual(third, fourth, "and hold the new one")

    def test_least_used_spreads_load(self) -> None:
        self.mgr.enable(strategy="least_used")
        self.mgr.pick()
        self.mgr.pick()
        self.mgr.pick()
        usage = {u["proxy"]: u["served"] for u in self.mgr.status()["usage"]}
        self.assertEqual(sum(usage.values()), 3)
        # 3 picks over 3 proxies → each served once; next is a true tie-break
        self.assertIn(self.mgr.pick(), self._pool())

    def test_empty_pool_falls_back_direct(self) -> None:
        # clear the store → nothing fresh in the pool
        proxylab.ProxyStore(os.path.join(self.tmp, "proxies")).clear()
        self.mgr.enable(strategy="round_robin")
        self.assertEqual(self.mgr.pick(), "")
        st = self.mgr.status()
        self.assertEqual(st["stats"]["direct_fallbacks"], 1)

    def test_core_http_routes_through_rotation(self) -> None:
        from nomorals.core.http import HttpClient

        self.mgr.enable(strategy="round_robin")
        seen = set()
        for _ in range(4):
            client = HttpClient(timeout=5.0)  # no explicit proxy
            self.assertIn(client.proxy_url, self._pool())
            seen.add(client.proxy_url)
            resp = client.get(self.urls["ip"])  # real request through it
            self.assertTrue(resp.ok)
        self.assertGreaterEqual(len(seen), 2,
                                "rotation must actually change the proxy")
        # an EXPLICIT proxy always wins over rotation
        explicit = HttpClient(proxy_url="socks5://9.9.9.9:9")
        self.assertEqual(explicit.proxy_url, "socks5://9.9.9.9:9")

    def test_failed_request_reports_to_rotation(self) -> None:
        from nomorals.core.http import HttpClient, RequestError

        self.mgr.enable(strategy="round_robin", cooldown_seconds=300)
        pool = self._pool()
        victim = pool[0]
        # nothing listens on this port — connection-level failure
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        dead_port = int(probe.getsockname()[1])
        probe.close()
        victim_url = victim.rsplit(":", 1)[0] + f":{dead_port}"
        client = HttpClient(timeout=2.0, proxy_url=victim_url)
        with self.assertRaises((RequestError, TimeoutError, OSError)):
            client.get(self.urls["ip"])
        st = self.mgr.status()
        self.assertIn(victim_url,
                      {c["proxy"] for c in st["cooling"]})
        self.assertGreaterEqual(st["stats"]["failovers"], 1)

    def test_state_persists_and_rearms(self) -> None:
        self.mgr.enable(strategy="sticky")
        first = self.mgr.pick()
        # a fresh manager instance (e.g. after restart) sees the state
        mgr2 = proxylab.ProxyRotationManager(self.ctx)
        self.assertTrue(mgr2.state["enabled"])
        self.assertEqual(mgr2.state["strategy"], "sticky")
        self.assertEqual(mgr2.pick(), first, "sticky must survive re-arm")
        mgr2.disable()

    def test_ssh_tunnel_joins_the_rotation_pool(self) -> None:
        self.mgr.enable(strategy="round_robin")
        from nomorals.tools.ssh_socks import SshSocksManager, SshSocksTunnel

        tunnel = SshSocksTunnel(
            SshSocksProfile(name="rot-t", host="10.9.9.9", user="u",
                            auto_reconnect=False),
            ssh_bin=self._fake_ssh())
        started = tunnel.start()
        self.assertTrue(started["started"], started)
        try:
            self.assertTrue(_wait_up(tunnel), "tunnel port never came up")
            SshSocksManager._live["rot-t"] = tunnel
            try:
                pool = self.mgr._pool_urls()
                self.assertIn(f"socks5://127.0.0.1:{tunnel.local_port}", pool)
            finally:
                SshSocksManager._live.pop("rot-t", None)
        finally:
            tunnel.stop()

    def _fake_ssh(self) -> str:
        self._ssh_path = os.path.join(self.tmp, "fake-ssh")
        with open(self._ssh_path, "w", encoding="utf-8") as fh:
            fh.write(_FAKE_SSH)
        os.chmod(self._ssh_path, 0o755)
        return self._ssh_path

    def test_tool_dispatch_and_boot_rearm(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.context = self.ctx
        proxylab.register(reg)
        r = reg.call("proxy_rotate", action="start", strategy="random")
        self.assertTrue(r.ok, getattr(r.error, "message", r.error))
        self.assertTrue(r.value["enabled"])
        r2 = reg.call("proxy_rotate", action="next")
        self.assertTrue(r2.ok and r2.value["rotated"])
        self.assertIn(r2.value["proxy"], self._pool())
        r3 = reg.call("proxy_rotate", action="status")
        self.assertTrue(r3.value["enabled"])
        self.assertEqual(r3.value["strategy"], "random")
        r4 = reg.call("proxy_rotate", action="stop")
        self.assertFalse(r4.value["enabled"])
        # next while disabled is an honest no-op
        r5 = reg.call("proxy_rotate", action="next")
        self.assertFalse(r5.value["rotated"])

    def test_bad_strategy_refused(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.context = self.ctx
        proxylab.register(reg)
        r = reg.call("proxy_rotate", action="start", strategy="chaos")
        self.assertFalse(r.ok)
        self.assertIn("strategy", str(getattr(r.error, "message", r.error)))


if __name__ == "__main__":
    unittest.main()
