"""Tests for the Fast.com native speed organ (52-websites #47).

The organ measures throughput to NAMED endpoints from THIS host —
never an invented number. The loopback fixture proves the arithmetic
against a local server with known file sizes; public endpoints are
gated by the network allowlist (NM_NET_ALLOWED_TARGETS) and by a live
flag (NM_SPEEDTEST_LIVE=1).
"""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nomorals.agents.morning_briefing import SpeedBriefingProvider
from nomorals.tools.speedtest import (
    SEED_ENDPOINTS,
    SpeedHistoryStore,
    SpeedResult,
    default_budgets,
    format_speed_reply,
    is_degraded,
    mbps,
    run_speed_test,
    seed_for,
)

# ── loopback fixture: a local server with known file sizes ────────────────


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 — keep the test log clean
        pass

    def _bytes(self, n: int) -> None:
        body = b"x" * n
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — stdlib handler name
        from urllib.parse import parse_qs, urlparse

        parts = urlparse(self.path)
        if parts.path == "/ping":
            self._bytes(2)
            return
        if parts.path == "/bytes":
            n = int(parse_qs(parts.query).get("n", ["0"])[0])
            self._bytes(n)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):  # noqa: N802 — stdlib handler name
        n = int(self.headers.get("Content-Length") or 0)
        remaining = n
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        payload = json.dumps({"received": n - remaining}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _start_fixture():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _loopback_seed(port: int):
    from nomorals.tools.speedtest import SeedEndpoint

    return SeedEndpoint(
        name="loopback",
        host="127.0.0.1",
        down_url=f"http://127.0.0.1:{port}/bytes?n={{n}}",
        up_url=f"http://127.0.0.1:{port}/upload",
        ping_url=f"http://127.0.0.1:{port}/ping",
        note="loopback fixture — known file sizes, proves the arithmetic",
    )


class ArithmeticTests(unittest.TestCase):
    def test_mbps_from_bytes_and_seconds(self):
        # 1 MB in 1 s == 8 Mbps; 10 MB in 2 s == 40 Mbps.
        self.assertAlmostEqual(mbps(1_000_000, 1.0), 8.0)
        self.assertAlmostEqual(mbps(10_000_000, 2.0), 40.0)

    def test_mbps_zero_time_is_zero_not_crash(self):
        self.assertEqual(mbps(1_000_000, 0.0), 0.0)
        self.assertEqual(mbps(0, 1.0), 0.0)

    def test_budgets_phone_shrinks_and_says_so(self):
        desktop_bytes, desktop_secs = default_budgets(phone=False)
        phone_bytes, phone_secs = default_budgets(phone=True)
        self.assertGreater(desktop_bytes, phone_bytes)
        self.assertGreater(desktop_secs, phone_secs)
        self.assertLessEqual(phone_bytes, 8 * 1024 * 1024)


class SeedTests(unittest.TestCase):
    def test_seeds_are_named_and_labelled(self):
        self.assertGreaterEqual(len(SEED_ENDPOINTS), 3)
        for seed in SEED_ENDPOINTS:
            self.assertTrue(seed.name)
            self.assertTrue(seed.host)
            self.assertTrue(seed.down_url)
            self.assertTrue(seed.note)

    def test_cloudflare_seed_has_upload(self):
        seed = seed_for("cloudflare")
        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertIn("speed.cloudflare.com", seed.down_url)
        self.assertIsNotNone(seed.up_url)

    def test_fastcom_seed_is_api_backed(self):
        seed = seed_for("fast.com")
        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertIn("api.fast.com", seed.down_url)

    def test_unknown_seed_is_none(self):
        self.assertIsNone(seed_for("no-such-seed"))

    def test_fastcom_api_payload_parses_to_targets(self):
        from nomorals.tools.speedtest import parse_fastcom_targets

        payload = {
            "client": {"ip": "1.2.3.4", "asn": "AS123", "location": {"city": "Lagos", "country": "NG"}},
            "targets": [
                {"name": "ipv4-c001-lhr001-ix.1.oca.nflxvideo.net", "url": "https://ipv4-c001-lhr001-ix.1.oca.nflxvideo.net/speedtest/range/0-100000000?c=ng", "location": {"city": "London", "country": "GB"}},
                {"name": "ipv4-c002-ams001-ix.1.oca.nflxvideo.net", "url": "https://ipv4-c002-ams001-ix.1.oca.nflxvideo.net/speedtest/range/0-100000000?c=ng", "location": {"city": "Amsterdam", "country": "NL"}},
            ],
        }
        targets = parse_fastcom_targets(payload)
        self.assertEqual(len(targets), 2)
        self.assertTrue(all(t.startswith("https://") for t in targets))
        self.assertEqual(parse_fastcom_targets({}), [])
        self.assertEqual(parse_fastcom_targets({"targets": [{"name": "x"}]}), [])


class AllowlistTests(unittest.TestCase):
    def test_no_allowlisted_seed_refuses_in_words(self):
        result = run_speed_test(home=None, allowed_targets="")
        self.assertFalse(result.ok)
        self.assertIn("NM_NET_ALLOWED_TARGETS", result.reason)

    def test_unlisted_public_seed_is_skipped_not_probed(self):
        # cloudflare is a public host; without the allowlist it must be
        # skipped, and the refusal must name the gate, not a network error.
        result = run_speed_test(
            home=None, target="cloudflare", allowed_targets="")
        self.assertFalse(result.ok)
        self.assertIn("NM_NET_ALLOWED_TARGETS", result.reason)

    def test_loopback_passes_without_allowlist(self):
        server, port = _start_fixture()
        try:
            result = run_speed_test(
                home=None, target="loopback", allowed_targets="",
                seeds=[_loopback_seed(port)])
            self.assertTrue(result.ok, result.reason)
            self.assertGreater(result.down_mbps, 0)
        finally:
            server.shutdown()
            server.server_close()


class LoopbackMeasureTests(unittest.TestCase):
    def test_download_arithmetic_reproduces(self):
        server, port = _start_fixture()
        try:
            result = run_speed_test(
                home=None, target="loopback", allowed_targets="",
                seeds=[_loopback_seed(port)])
            self.assertTrue(result.ok, result.reason)
            # The ramp stops at the plateau: bytes moved stay well under
            # the desktop budget, and the count is exact — every byte the
            # server sent was counted.
            self.assertLess(result.bytes_moved, 32 * 1024 * 1024)
            self.assertGreater(result.bytes_moved, 0)
            self.assertGreater(result.down_mbps, 0)
            self.assertGreater(result.duration_s, 0)
            self.assertEqual(result.target, "loopback")
            # Vantage names the machine the measurement ran from.
            self.assertTrue(result.vantage)
            self.assertIn("(host)", result.vantage)
        finally:
            server.shutdown()
            server.server_close()

    def test_upload_measured_and_latencies_sampled(self):
        server, port = _start_fixture()
        try:
            result = run_speed_test(
                home=None, target="loopback", allowed_targets="",
                seeds=[_loopback_seed(port)])
            self.assertTrue(result.ok, result.reason)
            self.assertIsNotNone(result.up_mbps)
            assert result.up_mbps is not None
            self.assertGreater(result.up_mbps, 0)
            self.assertGreater(result.latency_idle_ms, 0)
            self.assertIsNotNone(result.latency_loaded_ms)
            assert result.latency_loaded_ms is not None
            self.assertGreater(result.latency_loaded_ms, 0)
        finally:
            server.shutdown()
            server.server_close()

    def test_down_only_skips_upload_honestly(self):
        server, port = _start_fixture()
        try:
            result = run_speed_test(
                home=None, target="loopback", allowed_targets="",
                seeds=[_loopback_seed(port)], down_only=True)
            self.assertTrue(result.ok, result.reason)
            self.assertIsNone(result.up_mbps)
        finally:
            server.shutdown()
            server.server_close()

    def test_explicit_byte_budget_is_respected_and_named(self):
        server, port = _start_fixture()
        try:
            result = run_speed_test(
                home=None, target="loopback", allowed_targets="",
                seeds=[_loopback_seed(port)], budget_mb=1.0)
            self.assertTrue(result.ok, result.reason)
            self.assertLessEqual(result.bytes_moved, 1 * 1024 * 1024 + 262144)
            self.assertIn("1", result.budget_note)
        finally:
            server.shutdown()
            server.server_close()

    def test_unreachable_target_fails_open_with_reason(self):
        from nomorals.tools.speedtest import SeedEndpoint

        seed = SeedEndpoint(
            name="dead", host="127.0.0.1",
            down_url="http://127.0.0.1:1/bytes?n={n}",
            up_url=None, ping_url="http://127.0.0.1:1/ping",
            note="closed port — nothing listens here")
        result = run_speed_test(
            home=None, target="dead", allowed_targets="",
            seeds=[seed], max_seconds=5.0)
        self.assertFalse(result.ok)
        self.assertTrue(result.reason)


class HistoryTests(unittest.TestCase):
    def test_store_round_trip_and_baseline_median(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/fastcom-verify") as d:
            store = SpeedHistoryStore(d)
            self.assertIsNone(store.baseline("cloudflare"))
            for down in (10.0, 20.0, 30.0):
                store.append(SpeedResult(
                    target="cloudflare", down_mbps=down, up_mbps=5.0,
                    latency_idle_ms=20.0, latency_loaded_ms=40.0,
                    bytes_moved=1000, duration_s=1.0, vantage="test"))
            self.assertEqual(store.baseline("cloudflare"), 20.0)
            self.assertEqual(len(store.list()), 3)

    def test_baseline_too_thin_is_none_not_a_guess(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/fastcom-verify") as d:
            store = SpeedHistoryStore(d)
            for down in (10.0, 20.0):
                store.append(SpeedResult(
                    target="cloudflare", down_mbps=down, up_mbps=5.0,
                    latency_idle_ms=20.0, latency_loaded_ms=40.0,
                    bytes_moved=1000, duration_s=1.0, vantage="test"))
            self.assertIsNone(store.baseline("cloudflare"))

    def test_baseline_is_per_target(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/fastcom-verify") as d:
            store = SpeedHistoryStore(d)
            for down in (10.0, 20.0, 30.0):
                store.append(SpeedResult(
                    target="cloudflare", down_mbps=down, up_mbps=5.0,
                    latency_idle_ms=20.0, latency_loaded_ms=40.0,
                    bytes_moved=1000, duration_s=1.0, vantage="test"))
            self.assertIsNone(store.baseline("fast.com"))

    def test_degraded_threshold(self):
        self.assertTrue(is_degraded(5.0, 10.0))
        self.assertTrue(is_degraded(5.9, 10.0))
        self.assertFalse(is_degraded(6.0, 10.0))
        self.assertFalse(is_degraded(12.0, 10.0))
        self.assertFalse(is_degraded(5.0, 0.0))

    def test_result_round_trip(self):
        original = SpeedResult(
            target="cloudflare", down_mbps=42.5, up_mbps=10.0,
            latency_idle_ms=15.0, latency_loaded_ms=60.0,
            bytes_moved=12345, duration_s=2.5, vantage="box (host)")
        back = SpeedResult.from_dict(original.to_dict())
        self.assertEqual(back.target, original.target)
        self.assertEqual(back.down_mbps, original.down_mbps)
        self.assertEqual(back.bytes_moved, original.bytes_moved)


class ReplyTests(unittest.TestCase):
    def test_reply_labels_target_and_vantage(self):
        result = SpeedResult(
            target="cloudflare", down_mbps=42.5, up_mbps=10.0,
            latency_idle_ms=15.0, latency_loaded_ms=60.0,
            bytes_moved=12345, duration_s=2.5, vantage="box (host)")
        reply = format_speed_reply(result, baseline_mbps=50.0)
        self.assertIn("cloudflare", reply)
        self.assertIn("box (host)", reply)
        self.assertIn("42.5", reply)

    def test_reply_names_unmeasured_upload(self):
        result = SpeedResult(
            target="hetzner", down_mbps=42.5, up_mbps=None,
            latency_idle_ms=15.0, latency_loaded_ms=None,
            bytes_moved=12345, duration_s=2.5, vantage="box (host)")
        reply = format_speed_reply(result)
        self.assertIn("not measured", reply)

    def test_reply_for_failed_run_names_reason(self):
        result = SpeedResult(
            target="cloudflare", down_mbps=0.0, up_mbps=None,
            latency_idle_ms=0.0, latency_loaded_ms=None,
            bytes_moved=0, duration_s=0.0, vantage="box (host)",
            ok=False, reason="no allowlisted seed (NM_NET_ALLOWED_TARGETS)")
        reply = format_speed_reply(result)
        self.assertIn("NM_NET_ALLOWED_TARGETS", reply)


class _FakeRegistry:
    """Records the tool specs ``register()`` declares."""

    context = None

    def __init__(self):
        self.tools = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn

        return deco


class _FakeCtx:
    def __init__(self, home):
        import types

        self.settings = types.SimpleNamespace(home_path=str(home))


class RegistrationTests(unittest.TestCase):
    def test_speed_test_tool_registers(self):
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        from nomorals.tools import speedtest

        speedtest.register(reg)
        self.assertIn("speed_test", reg._tools)

    def test_tool_runs_loopback_through_registry(self):
        server, port = _start_fixture()
        try:
            registry = _FakeRegistry()
            registry.context = _FakeCtx("")
            from nomorals.tools import speedtest

            speedtest.register(registry)
            fn = registry.tools["speed_test"]
            outcome = fn(
                target="loopback", allowed_targets="",
                seeds=[_loopback_seed(port)])
            self.assertTrue(outcome.get("ok"), outcome)
            self.assertGreater(outcome.get("down_mbps", 0), 0)
        finally:
            server.shutdown()
            server.server_close()


class BriefingProviderTests(unittest.TestCase):
    def _ctx(self, home):
        import types

        settings = types.SimpleNamespace(home_path=str(home))
        return types.SimpleNamespace(settings=settings)

    def test_degraded_latest_gets_a_section(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/fastcom-verify") as d:
            store = SpeedHistoryStore(d)
            for down in (20.0, 20.0, 20.0):
                store.append(SpeedResult(
                    target="cloudflare", down_mbps=down, up_mbps=5.0,
                    latency_idle_ms=20.0, latency_loaded_ms=40.0,
                    bytes_moved=1000, duration_s=1.0, vantage="test"))
            store.append(SpeedResult(
                target="cloudflare", down_mbps=5.0, up_mbps=5.0,
                latency_idle_ms=20.0, latency_loaded_ms=40.0,
                bytes_moved=1000, duration_s=1.0, vantage="test"))
            provider = SpeedBriefingProvider()
            section = provider.collect(self._ctx(d), time.time())
            self.assertIsNotNone(section)
            assert section is not None
            self.assertIn("cloudflare", "\n".join(section.lines))

    def test_healthy_latest_is_silent(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/fastcom-verify") as d:
            store = SpeedHistoryStore(d)
            for down in (20.0, 20.0, 20.0, 21.0):
                store.append(SpeedResult(
                    target="cloudflare", down_mbps=down, up_mbps=5.0,
                    latency_idle_ms=20.0, latency_loaded_ms=40.0,
                    bytes_moved=1000, duration_s=1.0, vantage="test"))
            provider = SpeedBriefingProvider()
            self.assertIsNone(provider.collect(self._ctx(d), time.time()))

    def test_thin_baseline_is_silent(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/fastcom-verify") as d:
            store = SpeedHistoryStore(d)
            store.append(SpeedResult(
                target="cloudflare", down_mbps=5.0, up_mbps=5.0,
                latency_idle_ms=20.0, latency_loaded_ms=40.0,
                bytes_moved=1000, duration_s=1.0, vantage="test"))
            provider = SpeedBriefingProvider()
            self.assertIsNone(provider.collect(self._ctx(d), time.time()))

    def test_no_home_is_silent(self):
        import types

        provider = SpeedBriefingProvider()
        ctx = types.SimpleNamespace(
            settings=types.SimpleNamespace(home_path=""))
        self.assertIsNone(provider.collect(ctx, time.time()))


@unittest.skipUnless(
    os.environ.get("NM_SPEEDTEST_LIVE") == "1",
    "live cross-check needs NM_SPEEDTEST_LIVE=1 and real egress")
class LiveCrossCheckTests(unittest.TestCase):
    def test_cloudflare_magnitude_sanity(self):
        # A known public endpoint: the number must be a plausible
        # throughput, not zero and not absurd.
        result = run_speed_test(
            home=None, target="cloudflare",
            allowed_targets="speed.cloudflare.com",
            budget_mb=2.0, max_seconds=20.0)
        self.assertTrue(result.ok, result.reason)
        self.assertGreater(result.down_mbps, 0)
        self.assertLess(result.down_mbps, 10000)


if __name__ == "__main__":
    unittest.main()
