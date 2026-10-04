"""Wave L2: proxypool connector.

Offline by design — health checks run against in-process fake HTTP
forward proxies on loopback (one open, one requiring proxy auth) plus a
fake target server they forward to. No real external proxies, no network
beyond 127.0.0.1.
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import socket
import time
import socketserver
import threading
import unittest
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.base import ConnectorError
from nomorals.connectors.proxypool import ProxyPoolConnector, ProxyPoolError
from nomorals.connectors.registry import (
    create_connector,
    get_connector,
    list_connectors,
)
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


def _connector() -> ProxyPoolConnector:
    return ProxyPoolConnector(_vault())


#: Proxy env vars neutralized so urllib really goes through the fake proxy.
_CLEAR_PROXY_ENV = {
    "http_proxy": "",
    "https_proxy": "",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "all_proxy": "",
    "ALL_PROXY": "",
    "no_proxy": "",
    "NO_PROXY": "",
}


# ── fake network: target + forward proxies ────────────────────────────


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args: object) -> None:  # noqa: D102
        pass


class _TargetHandler(_QuietHandler):
    """The origin server the fake proxies forward to."""

    def do_GET(self) -> None:  # noqa: D102
        body = b"target-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ForwardProxyHandler(_QuietHandler):
    """Minimal real HTTP forward proxy (absolute-URI GET).

    Optionally demands Proxy-Authorization (``server.require_auth`` /
    ``server.expected_auth``), else forwards to the target directly.
    """

    def do_GET(self) -> None:  # noqa: D102
        if getattr(self.server, "require_auth", False):
            got = self.headers.get("Proxy-Authorization", "")
            if got != self.server.expected_auth:
                body = b"proxy authentication required"
                self.send_response(407)
                self.send_header(
                    "Proxy-Authenticate", 'Basic realm="proxypool-test"'
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        import time
        import urllib.request

        # Retry upstream fetch: under full-suite load the loopback target
        # can be slow to accept connections. Retry briefly, then fail
        # fast with 502 — the client retries, which beats one very long
        # stall that would trip the caller's own timeout instead.
        body = b"bad gateway"
        code = 502
        for _ in range(3):
            try:
                with urllib.request.urlopen(
                    self.path, timeout=3
                ) as upstream:
                    body = upstream.read()
                    code = int(upstream.status)
                break
            except Exception:  # noqa: BLE001 - test double, retry then 502
                time.sleep(0.1)
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(handler_cls: type, **attrs: object) -> http.server.ThreadingHTTPServer:
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_cls
    )
    for key, value in attrs.items():
        setattr(server, key, value)
    thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="proxypool-test-server"
    )
    thread.start()
    return server


def _closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class _ProxyNetCase(unittest.TestCase):
    """Base: fake target + open proxy + auth proxy on loopback."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.target = _serve(_TargetHandler)
        cls.target_port = cls.target.server_address[1]
        cls.open_proxy = _serve(_ForwardProxyHandler)
        cls.open_port = cls.open_proxy.server_address[1]
        cls.auth_user = "testuser"
        cls.auth_pass = "s3cret-pw"
        expected = "Basic " + base64.b64encode(
            f"{cls.auth_user}:{cls.auth_pass}".encode()
        ).decode()
        cls.auth_proxy = _serve(
            _ForwardProxyHandler, require_auth=True, expected_auth=expected
        )
        cls.auth_port = cls.auth_proxy.server_address[1]
        cls.target_url = f"http://127.0.0.1:{cls.target_port}/"

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (cls.target, cls.open_proxy, cls.auth_proxy):
            server.shutdown()
            server.server_close()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self._env = mock.patch.dict(os.environ, _CLEAR_PROXY_ENV)
        self._env.start()
        self.addCleanup(self._env.stop)


# ── fake network: SOCKS servers ─────────────────────────────────────


def _read_n(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    while n:
        chunk = sock.recv(n)
        if not chunk:
            raise ConnectionError("eof mid-handshake")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def _socks_relay(a: socket.socket, b: socket.socket) -> None:
    """Bidirectional byte relay between the tunneled sockets."""

    def fwd(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    t1 = threading.Thread(target=fwd, args=(a, b), daemon=True)
    t2 = threading.Thread(target=fwd, args=(b, a), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


class _Socks5Handler(socketserver.BaseRequestHandler):
    """Minimal real SOCKS5 server: greeting, optional user/pass auth,
    CONNECT, then a byte relay to the target. Honors ``server.require_auth``,
    ``server.expected_user``, ``server.expected_pass``."""

    def handle(self) -> None:  # noqa: D102
        conn = self.request
        try:
            _ver, nmethods = _read_n(conn, 2)
            methods = _read_n(conn, nmethods)
            if getattr(self.server, "require_auth", False):
                if 0x02 not in methods:
                    conn.sendall(b"\x05\xff")
                    return
                conn.sendall(b"\x05\x02")
                _read_n(conn, 1)  # auth version
                ulen = _read_n(conn, 1)[0]
                user = _read_n(conn, ulen)
                plen = _read_n(conn, 1)[0]
                pwd = _read_n(conn, plen)
                if ((user, pwd) != (self.server.expected_user,
                                    self.server.expected_pass)):
                    conn.sendall(b"\x01\x01")
                    return
                conn.sendall(b"\x01\x00")
            else:
                conn.sendall(b"\x05\x00")
            _ver, cmd, _rsv, atyp = _read_n(conn, 4)
            if cmd != 0x01:
                conn.sendall(b"\x05\x07\x00\x01" + b"\x00" * 6)
                return
            if atyp == 0x01:
                host = socket.inet_ntoa(_read_n(conn, 4))
            elif atyp == 0x03:
                ln = _read_n(conn, 1)[0]
                host = _read_n(conn, ln).decode("idna")
            elif atyp == 0x04:
                host = socket.inet_ntop(socket.AF_INET6, _read_n(conn, 16))
            else:
                conn.sendall(b"\x05\x08\x00\x01" + b"\x00" * 6)
                return
            port = int.from_bytes(_read_n(conn, 2), "big")
            try:
                upstream = socket.create_connection((host, port), timeout=10)
            except OSError:
                conn.sendall(b"\x05\x05\x00\x01" + b"\x00" * 6)
                return
            conn.sendall(b"\x05\x00\x00\x01" + b"\x00" * 6)
            _socks_relay(conn, upstream)
        except (ConnectionError, OSError):
            pass


class _Socks4aHandler(socketserver.BaseRequestHandler):
    """Minimal real SOCKS4/4a server: CONNECT (4a when the address is the
    0.0.0.1 marker, hostname follows the userid), then a byte relay."""

    def handle(self) -> None:  # noqa: D102
        conn = self.request
        try:
            hdr = _read_n(conn, 8)
            _vn, _cd = hdr[0], hdr[1]
            port = int.from_bytes(hdr[2:4], "big")
            ip = socket.inet_ntoa(hdr[4:8])
            while _read_n(conn, 1) != b"\x00":  # userid, ignored
                pass
            if ip == "0.0.0.1":
                domain = b""
                while True:
                    ch = _read_n(conn, 1)
                    if ch == b"\x00":
                        break
                    domain += ch
                host = domain.decode("idna")
            else:
                host = ip
            try:
                upstream = socket.create_connection((host, port), timeout=10)
            except OSError:
                conn.sendall(b"\x00\x5b" + b"\x00" * 6)
                return
            conn.sendall(b"\x00\x5a" + b"\x00" * 6)
            _socks_relay(conn, upstream)
        except (ConnectionError, OSError):
            pass


class _ThreadedSocksServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _serve_socks(handler_cls: type, **attrs: object) -> _ThreadedSocksServer:
    server = _ThreadedSocksServer(("127.0.0.1", 0), handler_cls)
    for key, value in attrs.items():
        setattr(server, key, value)
    thread = threading.Thread(
        target=server.serve_forever, daemon=True,
        name="proxypool-socks-test",
    )
    thread.start()
    return server


class _SocksNetCase(unittest.TestCase):
    """Base: fake target + SOCKS5 (open + auth) + SOCKS4a on loopback."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.target = _serve(_TargetHandler)
        cls.target_port = cls.target.server_address[1]
        cls.target_url = f"http://127.0.0.1:{cls.target_port}/"
        cls.socks5_open = _serve_socks(_Socks5Handler)
        cls.socks5_open_port = cls.socks5_open.server_address[1]
        cls.socks5_user = b"socksbob"
        cls.socks5_pass = b"socks-pw"
        cls.socks5_auth = _serve_socks(
            _Socks5Handler,
            require_auth=True,
            expected_user=cls.socks5_user,
            expected_pass=cls.socks5_pass,
        )
        cls.socks5_auth_port = cls.socks5_auth.server_address[1]
        cls.socks4a = _serve_socks(_Socks4aHandler)
        cls.socks4a_port = cls.socks4a.server_address[1]

    @classmethod
    def tearDownClass(cls) -> None:
        for server in (
            cls.target, cls.socks5_open, cls.socks5_auth, cls.socks4a
        ):
            server.shutdown()
            server.server_close()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.conn = _connector()


# ── SOCKS: add / validation ────────────────────────────────────────────


class SocksAddTests(_SocksNetCase):
    def test_add_socks5(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks5_open_port, protocol="socks5"
        )
        self.assertEqual(proxy["protocol"], "socks5")
        self.assertEqual(
            proxy["id"], f"socks5://127.0.0.1:{self.socks5_open_port}"
        )

    def test_socks_alias_normalizes_to_socks5(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks5_open_port, protocol="socks"
        )
        self.assertEqual(proxy["protocol"], "socks5")
        self.assertTrue(proxy["id"].startswith("socks5://"))

    def test_add_socks4_and_socks4a(self) -> None:
        p4 = self.conn.add_proxy(
            "127.0.0.1", self.socks4a_port, protocol="socks4"
        )
        self.assertEqual(p4["protocol"], "socks4")
        p4a = self.conn.add_proxy(
            "127.0.0.1", self.socks4a_port, protocol="socks4a"
        )
        self.assertEqual(p4a["protocol"], "socks4a")

    def test_socks4_password_rejected(self) -> None:
        with self.assertRaises(ProxyPoolError):
            self.conn.add_proxy(
                "127.0.0.1", self.socks4a_port, protocol="socks4",
                username="u", password="p",
            )

    def test_socks4a_password_rejected(self) -> None:
        with self.assertRaises(ProxyPoolError):
            self.conn.add_proxy(
                "127.0.0.1", self.socks4a_port, protocol="socks4a",
                username="u", password="p",
            )

    def test_socks5_allows_user_pass(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks5_auth_port, protocol="socks5",
            username="socksbob", password="socks-pw",
        )
        self.assertTrue(proxy["has_auth"])
        details = self.conn.get_proxy(proxy["id"])
        self.assertTrue(
            details["url_with_auth"].startswith("socks5://socksbob:")
        )


# ── SOCKS: health checks through real tunnels ───────────────────────────


class SocksHealthTests(_SocksNetCase):
    def test_health_check_socks5_no_auth(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks5_open_port, protocol="socks5"
        )
        summary = self.conn.health_check(url=self.target_url)
        self.assertEqual(summary["healthy"], 1)
        view = summary["results"][proxy["id"]]
        self.assertEqual(view["health"]["status"], "healthy")
        self.assertEqual(view["health"]["status_code"], 200)
        self.assertGreater(view["health"]["latency_ms"], 0)

    def test_health_check_socks5_with_auth(self) -> None:
        self.conn.add_proxy(
            "127.0.0.1", self.socks5_auth_port, protocol="socks5",
            username="socksbob", password="socks-pw",
        )
        summary = self.conn.health_check(url=self.target_url)
        self.assertEqual(summary["healthy"], 1)

    def test_health_check_socks5_bad_auth_fails_fast(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks5_auth_port, protocol="socks5",
            username="socksbob", password="wrong",
        )
        with self.assertRaises(ProxyPoolError) as ctx:
            self.conn.health_check(proxy["id"], url=self.target_url)
        self.assertIn("authentication failed", str(ctx.exception))
        view = self.conn.list_proxies()[0]
        self.assertEqual(view["health"]["status"], "unhealthy")

    def test_health_check_socks4a(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks4a_port, protocol="socks4a"
        )
        summary = self.conn.health_check(url=self.target_url)
        self.assertEqual(summary["healthy"], 1)
        self.assertEqual(
            summary["results"][proxy["id"]]["health"]["status_code"], 200
        )

    def test_health_check_socks4_ipv4_literal(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks4a_port, protocol="socks4"
        )
        summary = self.conn.health_check(url=self.target_url)
        self.assertEqual(summary["healthy"], 1)

    def test_health_check_socks5_hostname_target(self) -> None:
        # SOCKS5 ATYP=0x03 (domain form): the proxy resolves the name.
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks5_open_port, protocol="socks5"
        )
        url = f"http://localhost:{self.target_port}/"
        summary = self.conn.health_check(url=url)
        self.assertEqual(summary["healthy"], 1)
        self.assertEqual(
            summary["results"][proxy["id"]]["health"]["status_code"], 200
        )

    def test_health_check_socks_unreachable(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", _closed_port(), protocol="socks5"
        )
        summary = self.conn.health_check(url=self.target_url)
        self.assertEqual(summary["healthy"], 0)
        self.assertEqual(summary["unhealthy"], 1)
        view = summary["results"][proxy["id"]]
        self.assertIn("unreachable", view["health"]["last_error"])

    def test_rotate_picks_healthy_socks(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", self.socks5_open_port, protocol="socks5"
        )
        self.conn.health_check(url=self.target_url)
        picked = self.conn.rotate()
        self.assertEqual(picked["url"], proxy["url"])
        self.assertTrue(picked["url_with_auth"].startswith("socks5://"))


# ── add / list / remove / get ─────────────────────────────────────────


class AddProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _connector()

    def test_add_and_list_masks_password(self) -> None:
        proxy = self.conn.add_proxy(
            "127.0.0.1", 8080, username="u", password="pw-secret",
            tags=["research"],
        )
        self.assertEqual(proxy["id"], "http://127.0.0.1:8080")
        self.assertEqual(proxy["password"], "***")
        self.assertTrue(proxy["has_auth"])
        self.assertEqual(proxy["username"], "u")
        self.assertEqual(proxy["health"]["status"], "unknown")
        listed = self.conn.list_proxies()
        self.assertEqual(len(listed), 1)
        blob = json.dumps(listed)
        self.assertNotIn("pw-secret", blob)

    def test_add_defaults(self) -> None:
        proxy = self.conn.add_proxy("Example.COM ", "3128")
        self.assertEqual(proxy["id"], "http://example.com:3128")
        self.assertEqual(proxy["protocol"], "http")
        self.assertFalse(proxy["has_auth"])
        self.assertEqual(proxy["password"], "")
        self.assertEqual(proxy["tags"], [])

    def test_add_rejects_bad_input(self) -> None:
        bad = [
            dict(host="", port=8080),
            dict(host="   ", port=8080),
            dict(host="ho st", port=8080),
            dict(host="h", port=0),
            dict(host="h", port=65536),
            dict(host="h", port="abc"),
            dict(host="h", port=-1),
            dict(host="h", port=8080, protocol="ftp"),
            dict(host="h", port=8080, username="u"),  # user w/o password
            dict(host="h", port=8080, password="p"),  # password w/o user
            dict(host="h", port=8080, tags="nope"),  # type: ignore[dict-item]
            dict(host="h", port=8080, tags=["ok", 5]),  # type: ignore[list-item]
        ]
        for kwargs in bad:
            with self.assertRaises(ConnectorError, msg=str(kwargs)):
                self.conn.add_proxy(**kwargs)  # type: ignore[arg-type]

    def test_add_is_idempotent_update(self) -> None:
        first = self.conn.add_proxy("127.0.0.1", 8080, tags=["a"])
        second = self.conn.add_proxy(
            "127.0.0.1", 8080, username="u2", password="p2",
            protocol="HTTP", tags=["b"],
        )
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(self.conn.list_proxies()), 1)
        updated = self.conn.list_proxies()[0]
        self.assertEqual(updated["tags"], ["b"])
        self.assertTrue(updated["has_auth"])

    def test_vault_holds_encrypted_secret(self) -> None:
        self.conn.add_proxy("127.0.0.1", 8080, username="u",
                            password="topsecret-pw")
        row = self.conn.vault.db.query_one(
            "SELECT password_encrypted FROM credentials "
            "WHERE service = 'connector:proxypool'"
        )
        self.assertIsNotNone(row)
        self.assertNotIn("topsecret-pw", row["password_encrypted"])
        # ...while get_proxy can still decrypt it for component use.
        self.assertEqual(
            self.conn.get_proxy("http://127.0.0.1:8080")["password"],
            "topsecret-pw",
        )


class RemoveProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _connector()
        self.conn.add_proxy("127.0.0.1", 8080)
        self.conn.add_proxy("127.0.0.1", 8081)

    def test_remove(self) -> None:
        self.conn.remove_proxy("http://127.0.0.1:8080")
        remaining = self.conn.list_proxies()
        self.assertEqual([p["id"] for p in remaining],
                         ["http://127.0.0.1:8081"])

    def test_remove_is_idempotent(self) -> None:
        self.conn.remove_proxy("http://127.0.0.1:9999")  # unknown: no-op
        self.conn.remove_proxy("http://127.0.0.1:8080")
        self.conn.remove_proxy("http://127.0.0.1:8080")  # twice: no-op
        self.assertEqual(len(self.conn.list_proxies()), 1)

    def test_remove_normalizes_id(self) -> None:
        self.conn.remove_proxy(" HTTP://127.0.0.1:8080 ")
        self.assertEqual(len(self.conn.list_proxies()), 1)

    def test_remove_rejects_malformed_id(self) -> None:
        with self.assertRaises(ConnectorError):
            self.conn.remove_proxy("not-a-proxy-id")


class GetProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _connector()
        self.conn.add_proxy("127.0.0.1", 8080, username="u", password="p@ss:w0rd")

    def test_get_returns_full_details_with_credential(self) -> None:
        details = self.conn.get_proxy("http://127.0.0.1:8080")
        self.assertEqual(details["host"], "127.0.0.1")
        self.assertEqual(details["port"], 8080)
        self.assertEqual(details["username"], "u")
        self.assertEqual(details["password"], "p@ss:w0rd")
        # url_with_auth is the hand-to-HttpClient form; url stays clean.
        self.assertIn("u:", details["url_with_auth"])
        self.assertNotIn("p@ss", details["url"])
        self.assertTrue(details["url_with_auth"].startswith("http://"))

    def test_get_unknown_raises(self) -> None:
        with self.assertRaises(ConnectorError):
            self.conn.get_proxy("http://127.0.0.1:1234")


# ── health checks ───────────────────────────────────────────────────


class HealthCheckTests(_ProxyNetCase):
    def setUp(self) -> None:
        super().setUp()
        self.conn = _connector()

    def test_single_healthy_proxy(self) -> None:
        self.conn.add_proxy("127.0.0.1", self.open_port)
        pid = f"http://127.0.0.1:{self.open_port}"
        # Retry: under full-suite load the loopback target can be slow.
        # A slow probe either times out (health_check RAISES
        # ProxyPoolError — it does not return an unhealthy dict) or comes
        # back 502 while the fake upstream recovers. Retry both outcomes;
        # each attempt records a check, so assert checks >= 1 (not == 1).
        health: dict | None = None
        result: dict | None = None
        for attempt in range(12):
            try:
                result = self.conn.health_check(pid, url=self.target_url,
                                                timeout=5)
            except ProxyPoolError:
                time.sleep(0.5)
                continue
            health = result["health"]
            if health["status_code"] == 200:
                break
            time.sleep(0.5 * (attempt + 1))
        self.assertIsNotNone(health, "proxy never became healthy")
        assert health is not None and result is not None
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["status_code"], 200)
        self.assertGreaterEqual(health["latency_ms"], 0)
        self.assertGreater(health["last_checked"], 0)
        self.assertGreaterEqual(health["checks"], 1)
        self.assertEqual(health["consecutive_failures"], 0)
        # Password still masked in the health result.
        self.assertNotIn("s3cret", json.dumps(result))

    def test_single_failure_raises_and_records(self) -> None:
        port = _closed_port()
        self.conn.add_proxy("127.0.0.1", port)
        pid = f"http://127.0.0.1:{port}"
        with self.assertRaises(ProxyPoolError):
            self.conn.health_check(pid, url=self.target_url, timeout=5)
        stored = self.conn.list_proxies()[0]
        health = stored["health"]
        self.assertEqual(health["status"], "unhealthy")
        self.assertTrue(health["last_error"])
        self.assertEqual(health["consecutive_failures"], 1)
        self.assertEqual(health["checks"], 1)
        self.assertGreater(health["last_checked"], 0)

    def test_failure_counters_accumulate(self) -> None:
        port = _closed_port()
        self.conn.add_proxy("127.0.0.1", port)
        pid = f"http://127.0.0.1:{port}"
        for _ in range(2):
            with self.assertRaises(ProxyPoolError):
                self.conn.health_check(pid, url=self.target_url, timeout=5)
        health = self.conn.list_proxies()[0]["health"]
        self.assertEqual(health["consecutive_failures"], 2)
        self.assertEqual(health["checks"], 2)
        # Recovery resets the failure streak.
        self.conn.remove_proxy(pid)
        self.conn.add_proxy("127.0.0.1", self.open_port)
        pid2 = f"http://127.0.0.1:{self.open_port}"
        self.conn.health_check(pid2, url=self.target_url, timeout=10)
        self.assertEqual(
            self.conn.list_proxies()[0]["health"]["consecutive_failures"], 0
        )

    def test_auth_proxy_healthy_with_right_credentials(self) -> None:
        self.conn.add_proxy(
            "127.0.0.1", self.auth_port,
            username=self.auth_user, password=self.auth_pass,
        )
        pid = f"http://127.0.0.1:{self.auth_port}"
        result = self.conn.health_check(pid, url=self.target_url, timeout=10)
        self.assertEqual(result["health"]["status"], "healthy")

    def test_auth_proxy_fails_with_wrong_password(self) -> None:
        self.conn.add_proxy(
            "127.0.0.1", self.auth_port,
            username=self.auth_user, password="wrong",
        )
        pid = f"http://127.0.0.1:{self.auth_port}"
        with self.assertRaises(ProxyPoolError) as ctx:
            self.conn.health_check(pid, url=self.target_url, timeout=10)
        self.assertIn("407", str(ctx.exception))
        self.assertEqual(
            self.conn.list_proxies()[0]["health"]["status"], "unhealthy"
        )

    def test_check_all_returns_summary(self) -> None:
        good = f"http://127.0.0.1:{self.open_port}"
        bad = f"http://127.0.0.1:{_closed_port()}"
        self.conn.add_proxy("127.0.0.1", self.open_port)
        self.conn.add_proxy("127.0.0.1", int(bad.rsplit(":", 1)[1]))
        summary = self.conn.health_check(url=self.target_url, timeout=10)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["healthy"], 1)
        self.assertEqual(summary["unhealthy"], 1)
        self.assertEqual(summary["results"][good]["health"]["status"], "healthy")
        self.assertEqual(summary["results"][bad]["health"]["status"], "unhealthy")

    def test_check_all_empty_pool_raises(self) -> None:
        with self.assertRaises(ProxyPoolError):
            self.conn.health_check()

    def test_check_unknown_id_raises(self) -> None:
        with self.assertRaises(ConnectorError):
            self.conn.health_check("http://127.0.0.1:1", url=self.target_url)


# ── rotation ────────────────────────────────────────────────────────


class RotateTests(_ProxyNetCase):
    def setUp(self) -> None:
        super().setUp()
        self.conn = _connector()

    def _make_pool(self) -> tuple[str, str]:
        port_a, port_b, port_bad = self.open_port, self.auth_port, _closed_port()
        self.conn.add_proxy("127.0.0.1", port_a, tags=["a"])
        self.conn.add_proxy(
            "127.0.0.1", port_b, username=self.auth_user,
            password=self.auth_pass, tags=["b"],
        )
        self.conn.add_proxy("127.0.0.1", port_bad, tags=["dead"])
        self.conn.health_check(url=self.target_url, timeout=10)
        return (
            f"http://127.0.0.1:{port_a}",
            f"http://127.0.0.1:{port_b}",
        )

    def test_round_robin_over_healthy(self) -> None:
        id_a, id_b = self._make_pool()
        picks = [self.conn.rotate()["id"] for _ in range(4)]
        self.assertEqual(picks, [id_a, id_b, id_a, id_b])

    def test_rotate_skips_unhealthy(self) -> None:
        self.conn.add_proxy("127.0.0.1", self.open_port)
        self.conn.add_proxy("127.0.0.1", _closed_port())
        self.conn.health_check(url=self.target_url, timeout=10)
        good = f"http://127.0.0.1:{self.open_port}"
        for _ in range(3):
            self.assertEqual(self.conn.rotate()["id"], good)

    def test_rotate_returns_credential_for_components(self) -> None:
        self.conn.add_proxy(
            "127.0.0.1", self.auth_port, username=self.auth_user,
            password=self.auth_pass,
        )
        self.conn.health_check(url=self.target_url, timeout=10)
        details = self.conn.rotate()
        self.assertEqual(details["password"], self.auth_pass)
        self.assertIn("url_with_auth", details)

    def test_rotate_no_healthy_raises(self) -> None:
        self.conn.add_proxy("127.0.0.1", _closed_port())
        self.conn.health_check(url=self.target_url, timeout=10)
        with self.assertRaises(ProxyPoolError) as ctx:
            self.conn.rotate()
        self.assertIn("no healthy proxies", str(ctx.exception))

    def test_rotate_empty_pool_raises(self) -> None:
        with self.assertRaises(ProxyPoolError):
            self.conn.rotate()

    def test_disconnect_resets_rotation(self) -> None:
        id_a, _id_b = self._make_pool()
        self.conn.rotate()
        self.conn.disconnect()
        self.conn.disconnect()  # idempotent
        self.assertEqual(self.conn.rotate()["id"], id_a)


# ── lifecycle ───────────────────────────────────────────────────────


class LifecycleTests(_ProxyNetCase):
    def setUp(self) -> None:
        super().setUp()
        self.conn = _connector()

    def test_connect_empty_pool_is_honest_failure(self) -> None:
        result = self.conn.connect(check=False)
        self.assertFalse(result.ok)
        self.assertIn("add_proxy", result.message)

    def test_connect_without_check(self) -> None:
        self.conn.add_proxy("127.0.0.1", 8080)
        result = self.conn.connect(check=False)
        self.assertTrue(result.ok)
        self.assertIn("1 proxies", result.message)

    def test_connect_with_live_check(self) -> None:
        self.conn.add_proxy("127.0.0.1", self.open_port)
        with mock.patch.object(
            ProxyPoolConnector, "DEFAULT_CHECK_URL", self.target_url
        ):
            result = self.conn.connect()
        self.assertTrue(result.ok)
        self.assertIn("1/1", result.message)

    def test_connect_fails_when_nothing_healthy(self) -> None:
        self.conn.add_proxy("127.0.0.1", _closed_port())
        with mock.patch.object(
            ProxyPoolConnector, "DEFAULT_CHECK_URL", self.target_url
        ):
            result = self.conn.connect()
        self.assertFalse(result.ok)
        self.assertIn("none passed", result.message)

    def test_status_empty(self) -> None:
        status = self.conn.status()
        self.assertFalse(status.connected)
        self.assertIn("empty", status.detail)

    def test_status_reports_counts(self) -> None:
        self.conn.add_proxy("127.0.0.1", self.open_port)
        self.conn.add_proxy("127.0.0.1", _closed_port())
        self.conn.health_check(url=self.target_url, timeout=10)
        status = self.conn.status()
        self.assertTrue(status.connected)
        self.assertIn("2 proxies", status.detail)
        self.assertIn("1 healthy", status.detail)
        self.assertIn("1 unhealthy", status.detail)
        self.assertGreater(status.last_checked, 0)

    def test_status_not_connected_when_none_healthy(self) -> None:
        self.conn.add_proxy("127.0.0.1", _closed_port())
        self.conn.health_check(url=self.target_url, timeout=10)
        self.assertFalse(self.conn.status().connected)

    def test_test_connection(self) -> None:
        self.assertFalse(self.conn.test_connection())  # empty pool
        self.conn.add_proxy("127.0.0.1", self.open_port)
        with mock.patch.object(
            ProxyPoolConnector, "DEFAULT_CHECK_URL", self.target_url
        ):
            self.assertTrue(self.conn.test_connection())
        dead = _connector()
        dead.add_proxy("127.0.0.1", _closed_port())
        with mock.patch.object(
            ProxyPoolConnector, "DEFAULT_CHECK_URL", self.target_url
        ):
            self.assertFalse(dead.test_connection())


# ── provisioning / registry ─────────────────────────────────────────


class ProvisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = _connector()

    def test_provision_proxy(self) -> None:
        self.assertTrue(self.conn.can_provision("proxy"))
        result = self.conn.provision(
            "proxy", host="127.0.0.1", port=8080, tags=["paid"]
        )
        self.assertEqual(result["id"], "http://127.0.0.1:8080")
        self.assertEqual(len(self.conn.list_proxies()), 1)

    def test_provision_unknown_kind_fails(self) -> None:
        self.assertFalse(self.conn.can_provision("repo"))
        with self.assertRaises(ConnectorError) as ctx:
            self.conn.provision("repo", name="x")
        self.assertIn("proxy", str(ctx.exception))


class RegistryTests(unittest.TestCase):
    def test_registered_as_proxypool(self) -> None:
        self.assertIs(get_connector("proxypool"), ProxyPoolConnector)
        conn = create_connector("proxypool", _vault())
        self.assertIsInstance(conn, ProxyPoolConnector)
        infos = {info["id"]: info for info in list_connectors()}
        self.assertIn("proxypool", infos)
        self.assertEqual(infos["proxypool"]["auth_methods"], ["none"])
        self.assertEqual(infos["proxypool"]["provisionable"], ["proxy"])


class NoLeakTests(_ProxyNetCase):
    SECRET = "leak-me-not-pw"

    def test_no_secret_in_any_owner_facing_output(self) -> None:
        conn = _connector()
        conn.add_proxy(
            "127.0.0.1", self.open_port, username="u", password=self.SECRET
        )
        blob = json.dumps(conn.list_proxies())
        self.assertNotIn(self.SECRET, blob)
        summary = conn.health_check(url=self.target_url, timeout=10)
        self.assertNotIn(self.SECRET, json.dumps(summary))
        self.assertNotIn(self.SECRET, json.dumps(conn.status().to_dict()))
        self.assertNotIn(self.SECRET, conn.connect(check=False).message)


if __name__ == "__main__":
    unittest.main()
