"""Lane 75 (audit D4) — proxy rotation into HttpClient, fail-closed.

Proves, against REAL local faked HTTP-proxy endpoints (loopback
``ThreadingHTTPServer``s, each answering 200 with its own id — no live
network, no paid services):

- round-robin actually cycles proxies across real ``HttpClient``
  requests (distinct hosts per request, so affinity cannot mask it)
- same-host requests stay sticky on one proxy (domain affinity, named)
- a dead proxy is retired on connection failure and revives after its
  cooldown expires
- fail-closed: an exhausted pool raises ``ProxyExhaustedError`` from
  ``pick()``, from the resolver, and from ``HttpClient(require_proxy)``
  — never a silent direct fallback that pretends rotation happened
- config gating: rotation off means direct, named; ``require_proxy``
  with nothing configured is a named refusal, not a crash
- the owner's own proxies (``NM_PROXY_KNOWN``) join the rotation pool,
  are journalled as tier ``paid``, and no credential ever lands in a
  receipt or a status row
- surface: ``proxy_rotate`` tool ``fail_closed`` flag, ``nm proxy
  rotate`` CLI (start/status/next/stop incl. persistence across calls),
  ``/proxy rotate start [fail-closed]`` chat parsing
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.cmdline.commands.proxy import _cmd_proxy
from nomorals.core import http as core_http
from nomorals.core.errors import ProviderUnavailable
from nomorals.core.http import HttpClient, ProxyExhaustedError, RequestError
from nomorals.tools import proxylab as PL
from nomorals.tools.proxylab import ProxyRotationManager
from nomorals.tools.registry import ToolRegistry

_PROXY_ENV_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                   "ALL_PROXY", "all_proxy", "no_proxy", "NO_PROXY",
                   "NM_PROXY_URL", "NM_PROXY_KNOWN")


class _ProxyHandler(BaseHTTPRequestHandler):
    """A faked HTTP proxy: answers 200 with its own id, or — in dead
    mode — closes the connection without a response (the client sees
    a connection failure, exactly like a dead real proxy)."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:  # silence
        return

    def handle_error(self, *args: object) -> None:  # dead-mode noise
        return

    def _serve(self) -> None:
        if getattr(self.server, "dead", False):
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            self.close_connection = True
            return
        body = f"via:{self.server.proxy_id}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.server.hits.append((self.server.proxy_id, self.path))
        self._serve()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.server.hits.append((self.server.proxy_id, self.path))
        self._serve()


def _start_server(proxy_id: str):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProxyHandler)
    server.daemon_threads = True
    server.proxy_id = proxy_id
    server.hits = []
    server.dead = False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _reset_http() -> None:
    core_http.set_proxy_resolver(None)
    core_http.set_proxy_error_reporter(None)
    core_http.set_proxy_success_reporter(None)


def _ctx(home: str, known: str = "", db=None) -> SimpleNamespace:
    proxy = SimpleNamespace(probe_url="http://127.0.0.1/",
                            probe_timeout=1.0, known=known,
                            active="", url="")
    return SimpleNamespace(settings=SimpleNamespace(home=home, proxy=proxy),
                           db=db)


class _RotationCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {k: os.environ.pop(k) for k in _PROXY_ENV_VARS
                           if k in os.environ}
        self.addCleanup(self._restore_env)
        _reset_http()
        self.addCleanup(_reset_http)
        self.tmp = tempfile.mkdtemp(prefix="d4-proxy-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.servers: list[ThreadingHTTPServer] = []
        self.addCleanup(self._stop_servers)

    def _restore_env(self) -> None:
        os.environ.update(self._saved_env)

    def _stop_servers(self) -> None:
        for s in self.servers:
            s.shutdown()
            s.server_close()

    def _proxy(self, proxy_id: str) -> tuple[ThreadingHTTPServer, str]:
        server = _start_server(proxy_id)
        self.servers.append(server)
        port = server.server_address[1]
        return server, f"http://127.0.0.1:{port}"

    def _seed_lab(self, ctx, urls: list[str]) -> None:
        lab = PL.ProxyLab(ctx)
        now = time.time()
        proxies = []
        for url in urls:
            host, port = url.split("://", 1)[1].rsplit(":", 1)
            p = PL.Proxy(host=host, port=int(port), scheme="http",
                         anonymity="elite", country="NG", alive=True,
                         tested_at=now)
            p.record_latency(200.0, at=now)
            proxies.append(p)
        lab.store.save(proxies)

    def _client(self, **kw) -> HttpClient:
        return HttpClient(allow_private_ips=True, timeout=5.0, **kw)

    def _journal_rows(self) -> list[dict]:
        path = os.path.join(self.tmp, "proxies", "route_receipts.jsonl")
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


class TestRoundRobinThroughHttpClient(_RotationCase):
    def test_rotation_cycles_across_real_requests(self) -> None:
        _, url_a = self._proxy("A")
        _, url_b = self._proxy("B")
        ctx = _ctx(self.tmp)
        self._seed_lab(ctx, [url_a, url_b])
        manager = ProxyRotationManager(ctx)
        manager.enable()
        self.addCleanup(manager.disable)
        client = self._client()
        served: list[str] = []
        for i in range(4):
            # Distinct hosts: affinity cannot pin them, so each
            # request takes a fresh round-robin pick.
            resp = client.get(f"http://host-{i}.test/")
            self.assertEqual(resp.status, 200)
            served.append(resp.text)
        self.assertEqual(sorted(served),
                         ["via:A", "via:A", "via:B", "via:B"])
        # Strict alternation — not just two proxies that both work.
        self.assertNotEqual(served[0], served[1])
        self.assertNotEqual(served[1], served[2])
        self.assertNotEqual(served[2], served[3])
        # Both faked proxies really served traffic.
        hits = [h[0] for s in self.servers for h in s.hits]
        self.assertEqual(sorted(hits), ["A", "A", "B", "B"])

    def test_same_host_stays_sticky(self) -> None:
        _, url_a = self._proxy("A")
        _, url_b = self._proxy("B")
        ctx = _ctx(self.tmp)
        self._seed_lab(ctx, [url_a, url_b])
        manager = ProxyRotationManager(ctx)
        manager.enable()
        self.addCleanup(manager.disable)
        client = self._client()
        served = {client.get("http://sticky.test/").text for _ in range(3)}
        self.assertEqual(len(served), 1, served)


class TestDeadProxyRetirementAndRevival(_RotationCase):
    def test_dead_proxy_retires_then_revives(self) -> None:
        dead_srv, dead_url = self._proxy("DEAD")
        _, live_url = self._proxy("LIVE")
        ctx = _ctx(self.tmp)
        self._seed_lab(ctx, [dead_url, live_url])
        manager = ProxyRotationManager(ctx)
        manager.enable(cooldown_seconds=1)
        self.addCleanup(manager.disable)
        client = self._client()
        failed = False
        for i in range(2):
            # The dead proxy kills the connection mid-request; arm
            # it only once the live proxy has served (order of the
            # first two round-robin picks is not assumed).
            dead_srv.dead = True
            try:
                client.get(f"http://victim-{i}.test/")
            except (RequestError, ProviderUnavailable):
                # Connection failure through the dead proxy: urllib
                # surfaces it as ProviderUnavailable (OSError branch),
                # plain HTTP failures as RequestError.
                failed = True
                break
        self.assertTrue(failed, "no request ever hit the dead proxy")
        # The failure retired it: it is cooling, and the next
        # request succeeds through the live proxy alone.
        self.assertIn(dead_url, manager.state["cooldown"])
        self.assertEqual(
            client.get("http://after-failure.test/").text, "via:LIVE")
        # Revival: after the cooldown expires it is healthy again
        # and round-robin hands it traffic once more.
        dead_srv.dead = False
        time.sleep(1.2)
        self.assertIn(dead_url, manager._healthy(manager._pool_urls()))
        served = {client.get(f"http://revived-{i}.test/").text
                  for i in range(2)}
        self.assertIn("via:DEAD", served)


class TestFailClosed(_RotationCase):
    def test_empty_pool_refuses_from_pick(self) -> None:
        ctx = _ctx(self.tmp)
        manager = ProxyRotationManager(ctx)
        out = manager.enable(fail_closed=True)
        self.addCleanup(manager.disable)
        self.assertIn("REFUSE", out["note"])
        self.assertTrue(out["fail_closed"])
        with self.assertRaises(ProxyExhaustedError) as caught:
            manager.pick()
        self.assertEqual(caught.exception.reason_code, "pool_exhausted")
        rows = self._journal_rows()
        refusal = [r for r in rows
                   if r.get("reason_code") == "pool_exhausted_fail_closed"]
        self.assertEqual(len(refusal), 1)
        self.assertEqual(refusal[0]["route"], "none")

    def test_fail_closed_request_refuses_never_direct(self) -> None:
        ctx = _ctx(self.tmp)
        manager = ProxyRotationManager(ctx)
        manager.enable(fail_closed=True)
        self.addCleanup(manager.disable)
        # A direct target that WOULD answer proves the refusal is
        # real: nothing is sent direct behind the error.
        direct = _start_server("direct")
        self.servers.append(direct)
        port = direct.server_address[1]
        client = self._client(require_proxy=True)
        with self.assertRaises(ProxyExhaustedError):
            client.get(f"http://127.0.0.1:{port}/")
        self.assertEqual(direct.hits, [])

    def test_require_proxy_with_nothing_configured_is_named(self) -> None:
        self.assertIsNone(core_http.get_proxy_resolver())
        client = self._client(require_proxy=True)
        with self.assertRaises(ProxyExhaustedError) as caught:
            client.get("http://host-9.test/")
        self.assertEqual(caught.exception.reason_code,
                         "no_proxy_configured")

    def test_not_fail_closed_still_falls_back_direct(self) -> None:
        ctx = _ctx(self.tmp)
        manager = ProxyRotationManager(ctx)
        manager.enable()  # default: fail_closed=False
        self.addCleanup(manager.disable)
        direct = _start_server("direct")
        self.servers.append(direct)
        port = direct.server_address[1]
        client = self._client()
        resp = client.get(f"http://127.0.0.1:{port}/")
        self.assertEqual(resp.text, "via:direct")
        self.assertEqual(len(direct.hits), 1)


class TestConfigGating(_RotationCase):
    def test_rotation_off_means_direct_and_named(self) -> None:
        ctx = _ctx(self.tmp)
        manager = ProxyRotationManager(ctx)
        self.assertFalse(manager.state.get("enabled"))
        self.assertIsNone(core_http.get_proxy_resolver())
        self.assertIn("rotation is off",
                      manager.status()["direct_routing_when"])
        direct = _start_server("direct")
        self.servers.append(direct)
        port = direct.server_address[1]
        resp = self._client().get(f"http://127.0.0.1:{port}/")
        self.assertEqual(resp.text, "via:direct")

    def test_broken_resolver_still_goes_direct(self) -> None:
        def _boom(target: str = "") -> str:
            raise RuntimeError("resolver exploded")

        core_http.set_proxy_resolver(_boom)
        direct = _start_server("direct")
        self.servers.append(direct)
        port = direct.server_address[1]
        resp = self._client().get(f"http://127.0.0.1:{port}/")
        self.assertEqual(resp.text, "via:direct")


class TestOwnerProxiesJoinRotation(_RotationCase):
    def test_owner_proxies_rotate_with_paid_receipts(self) -> None:
        _, url_a = self._proxy("OWN-A")
        _, url_b = self._proxy("OWN-B")
        ctx = _ctx(self.tmp, known=f"{url_a},{url_b}")
        manager = ProxyRotationManager(ctx)
        self.assertEqual(manager._pool_urls(), [url_a, url_b])
        manager.enable()
        self.addCleanup(manager.disable)
        status = manager.status()
        self.assertEqual(status["owner_proxies"], 2)
        self.assertEqual(status["pool_size"], 2)
        client = self._client()
        served = {client.get(f"http://owner-{i}.test/").text
                  for i in range(2)}
        self.assertEqual(served, {"via:OWN-A", "via:OWN-B"})
        rows = [r for r in self._journal_rows()
                if r.get("consumer") == "core_http"]
        self.assertTrue(rows)
        self.assertTrue(all(r["route"] == "paid" for r in rows), rows)

    def test_owner_proxy_credentials_never_leak(self) -> None:
        srv, url = self._proxy("CRED")
        port = srv.server_address[1]
        cred_url = f"http://alice:s3cretpass@127.0.0.1:{port}"
        ctx = _ctx(self.tmp, known=cred_url)
        manager = ProxyRotationManager(ctx)
        manager.enable()
        self.addCleanup(manager.disable)
        manager.pick()
        manager.report_error(cred_url, "boom")
        blob = json.dumps(self._journal_rows()) + json.dumps(
            manager.status())
        self.assertNotIn("s3cretpass", blob)
        self.assertIn("alice:***@", json.dumps(manager.status()))

    def test_owner_affinity_survives_lookup(self) -> None:
        _, url_a = self._proxy("OWN-A")
        _, url_b = self._proxy("OWN-B")
        ctx = _ctx(self.tmp, known=f"{url_a},{url_b}")
        manager = ProxyRotationManager(ctx)
        manager.enable()
        self.addCleanup(manager.disable)
        client = self._client()
        first = client.get("http://pinned.test/").text
        # The binding is to an owner proxy (not in the lab store) —
        # it must survive lookup, not be dropped as "gone".
        self.assertEqual(
            {client.get("http://pinned.test/").text for _ in range(3)},
            {first})


class TestToolAndCliSurface(_RotationCase):
    def _registry(self, ctx):
        reg = ToolRegistry(context=ctx)
        PL.register(reg)
        return reg

    def test_proxy_rotate_tool_fail_closed_flag(self) -> None:
        # State persists across tool calls through the KV db, so the
        # status call below reads the start call's fail_closed.
        from nomorals.storage.db import Database

        db = Database(":memory:")
        db.migrate()
        ctx = _ctx(self.tmp, db=db)
        reg = self._registry(ctx)
        out = reg.call("proxy_rotate", action="start", fail_closed="true")
        self.assertTrue(out.ok, out)
        self.assertTrue(out.value["fail_closed"])
        try:
            status = reg.call("proxy_rotate", action="status")
            self.assertTrue(status.ok, status)
            self.assertTrue(status.value["fail_closed"])
            self.assertIn("refuses", status.value["direct_routing_when"])
        finally:
            reg.call("proxy_rotate", action="stop")
        self.assertIsNone(core_http.get_proxy_resolver())

    def test_registry_tool_count_still_804(self) -> None:
        # No new brain tool was added by this lane (the proxy_rotate
        # tool gained a parameter, not a new registration).
        # 794, not 787: the trendclip port (gapfix163) added 5 tools
        # (trendclip_discover, trendclip_analyze, trendclip_render,
        # trendclip_status, trendclip_queue) after this lane's base
        # (parent re-bump, 2026-11-25).
        # 787, not 786: the museum ship's museum_library tool landed
        # after this lane's base (parent re-bump, 2026-11-25).
        # (786, not 785: the swarm ship's build_swarm tool landed
        # after this lane's base (parent re-bump, 2026-11-24).)
        reg = ToolRegistry(context=_ctx(self.tmp))
        reg.register_builtins()
        self.assertEqual(len(reg._tools), 804)  # trendclip adds 5 tools

    def _run_cli(self, ctx, **fields):
        base = {"proxy_action": "", "proxy_sources_action": "",
                "rotate_action": "status", "strategy": "",
                "fail_closed": False, "cooldown_seconds": 300.0,
                "max_age_hours": 24.0, "limit": 10, "consumer": "",
                "source": "", "json": False}
        base.update(fields)
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            rc = _cmd_proxy(argparse.Namespace(**base), ctx)
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def test_cli_rotate_start_status_next_stop(self) -> None:
        from nomorals.storage.db import Database

        db = Database(":memory:")
        db.migrate()
        ctx = _ctx(self.tmp, db=db)
        try:
            rc, out, _ = self._run_cli(ctx, proxy_action="rotate",
                                       rotate_action="start",
                                       fail_closed=True)
            self.assertEqual(rc, 0)
            self.assertIn("REFUSE", out)
            rc, out, _ = self._run_cli(ctx, proxy_action="rotate",
                                       rotate_action="status")
            self.assertEqual(rc, 0)
            self.assertIn("fail-closed", out)
            self.assertIn("pool: 0", out)
            # Fail-closed + empty pool: next refuses, named, rc 1.
            rc, _out, err = self._run_cli(ctx, proxy_action="rotate",
                                          rotate_action="next")
            self.assertEqual(rc, 1)
            self.assertIn("proxy rotation refused", err)
            rc, out, _ = self._run_cli(ctx, proxy_action="rotate",
                                       rotate_action="stop")
            self.assertEqual(rc, 0)
            self.assertIn("off", out)
        finally:
            _reset_http()

    def test_cli_rotate_next_empty_pool_goes_direct_when_open(self) -> None:
        from nomorals.storage.db import Database

        db = Database(":memory:")
        db.migrate()
        ctx = _ctx(self.tmp, db=db)
        try:
            rc, _, _ = self._run_cli(ctx, proxy_action="rotate",
                                     rotate_action="start")
            self.assertEqual(rc, 0)
            rc, out, _ = self._run_cli(ctx, proxy_action="rotate",
                                       rotate_action="next")
            self.assertEqual(rc, 0)
            self.assertIn("go direct", out)
        finally:
            self._run_cli(ctx, proxy_action="rotate", rotate_action="stop")
            _reset_http()


class TestChatVerb(_RotationCase):
    def test_rotate_start_fail_closed_flag_parsed(self) -> None:
        rt = RuntimeSystemMixin()
        calls: list[tuple[str, dict]] = []

        def _fake_reply(tool: str, chat_key: str, **kw):
            calls.append((tool, kw))
            return "ok"

        rt._tool_reply = _fake_reply  # type: ignore[method-assign]
        rt._control_proxy("rotate start round_robin fail-closed")
        self.assertEqual(calls, [("proxy_rotate",
                                  {"action": "start",
                                   "strategy": "round_robin",
                                   "fail_closed": "true"})])
        calls.clear()
        rt._control_proxy("rotate start")
        self.assertEqual(calls, [("proxy_rotate",
                                  {"action": "start", "strategy": "",
                                   "fail_closed": "false"})])
        calls.clear()
        rt._control_proxy("rotate status")
        self.assertEqual(calls, [("proxy_rotate", {"action": "status"})])


if __name__ == "__main__":
    unittest.main()
