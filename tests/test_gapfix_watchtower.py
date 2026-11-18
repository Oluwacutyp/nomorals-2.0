"""Tests for the Downdetector-native watchtower (52-websites #45).

The organ watches external services the owner depends on — exchanges,
brokers, Telegram, GitHub — by actively probing them from this host
and applying the baseline-deviation mathematics Downdetector uses:
a three-state verdict (no problems / possible problems / problems),
an incident only when current failures sit significantly above the
baseline, and honest "insufficient baseline" below 30 samples.

The honest limit is stated on every verdict: Devon has no crowd of
millions of reporters. Her sensors are one host's probes plus public
status-page feeds, so a verdict means "unreachable/broken from here
+ corroborating public signal" — a narrower claim, labelled as such.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nomorals.sources.watchtower import (
    MIN_SAMPLES,
    WATCHTOWER_JOB,
    WATCHTOWER_SPEC,
    WATCHTOWER_TOOL,
    Incident,
    ProbeBaseline,
    ProbeSample,
    WatchTarget,
    WatchtowerStore,
    check_target,
    ensure_watchtower_job,
    evaluate_verdict,
    format_watchtower_reply,
    probe_feed,
    probe_http,
    probe_target,
    run_watchtower_tick_once,
)

VERIFY = Path("/home/hatch/workspace/downdetector-verify")


# ── loopback fixture: a local server with known behaviours ──────────────


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 — keep the test log clean
        pass

    def _send(self, status: int, body: bytes,
              ctype: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — stdlib handler name
        if self.path == "/ok":
            self._send(200, b"fine")
        elif self.path == "/down":
            self._send(503, b"unavailable")
        elif self.path == "/feed-ok":
            self._send(200, b'{"status": "operational"}',
                       "application/json")
        elif self.path == "/feed-major":
            self._send(200, b'{"status": "major_outage"}',
                       "application/json")
        elif self.path == "/feed-page":
            self._send(200, b"<html><body>We are investigating a "
                            b"major outage affecting logins.</body></html>",
                       "text/html")
        elif self.path == "/feed-degraded":
            self._send(200, b'{"status": "degraded_performance"}',
                       "application/json")
        else:
            self._send(404, b"nope")


def _start_fixture():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _sample(at: float, ok: bool, latency_ms: float = 42.0) -> ProbeSample:
    return ProbeSample(at=at, ok=ok, latency_ms=latency_ms,
                       status=200 if ok else 503,
                       error="" if ok else "http 503")


def _series(n_ok: int, n_bad: int, *, start: float = 1000.0,
            step: float = 240.0) -> list[ProbeSample]:
    """n_ok healthy probes then n_bad failing ones, oldest first."""
    out: list[ProbeSample] = []
    at = start
    for _ in range(n_ok):
        out.append(_sample(at, True))
        at += step
    for _ in range(n_bad):
        out.append(_sample(at, False))
        at += step
    return out


class _FakeNotifier:
    def __init__(self) -> None:
        self.published: list[dict] = []

    def publish(self, kind: str, title: str = "", body: str = "",
                **kw) -> dict:
        self.published.append({"kind": kind, "title": title,
                               "body": body, **kw})
        return {"ok": True}


class _FakeScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict] = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        job = {"name": name, "spec": spec, "id": f"job-{len(self.jobs)}"}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


class BaselineTests(unittest.TestCase):
    def test_thin_baseline_is_insufficient(self):
        base = ProbeBaseline()
        for sample in _series(5, 0):
            base.record(sample)
        thresholds = base.thresholds()
        self.assertEqual(thresholds.status, "insufficient")
        self.assertEqual(thresholds.samples, 5)
        self.assertIn(str(MIN_SAMPLES), thresholds.reason)

    def test_ready_baseline_reports_p95(self):
        base = ProbeBaseline()
        for sample in _series(29, 1):
            base.record(sample)
        thresholds = base.thresholds()
        self.assertEqual(thresholds.status, "ready")
        self.assertEqual(thresholds.samples, 30)
        self.assertAlmostEqual(thresholds.failure_rate, 1 / 30)
        self.assertIsNotNone(thresholds.latency_p95)
        self.assertGreater(thresholds.latency_p95, 0)

    def test_degenerate_latency_leg_is_unavailable(self):
        # 30 healthy probes that report 0.0 ms latency: the failure
        # legs still work (the service is up), but the latency leg
        # cannot say what "slow" is — it is skipped, never invented.
        base = ProbeBaseline()
        for i in range(30):
            base.record(ProbeSample(at=float(i), ok=True, latency_ms=0.0,
                                    status=200))
        thresholds = base.thresholds()
        self.assertEqual(thresholds.status, "ready")
        self.assertFalse(thresholds.latency_ready)
        self.assertIsNone(thresholds.latency_p95)
        verdict = evaluate_verdict(
            "svc", [ProbeSample(at=float(i), ok=True, latency_ms=0.0,
                                status=200) for i in range(30)])
        self.assertEqual(verdict.state, "ok")

    def test_window_bounds_samples(self):
        base = ProbeBaseline(window=10)
        for sample in _series(15, 0):
            base.record(sample)
        self.assertEqual(len(base), 10)


class VerdictTests(unittest.TestCase):
    def test_thin_baseline_never_confident(self):
        verdict = evaluate_verdict("binance", _series(5, 0))
        self.assertEqual(verdict.state, "unknown")
        self.assertIn("insufficient baseline", verdict.reason)

    def test_healthy_series_is_ok(self):
        verdict = evaluate_verdict("github", _series(40, 0))
        self.assertEqual(verdict.state, "ok")

    def test_one_failure_on_perfect_baseline_is_possible_not_down(self):
        # A single failure is noise, not an incident — possible only.
        verdict = evaluate_verdict("github", _series(39, 1))
        self.assertEqual(verdict.state, "possible")

    def test_three_consecutive_failures_is_down(self):
        verdict = evaluate_verdict("binance", _series(37, 3))
        self.assertEqual(verdict.state, "down")
        self.assertGreater(verdict.evidence["baseline_deviation"], 0.25)

    def test_five_consecutive_failures_is_down_even_on_noisy_baseline(self):
        # Baseline itself flaky (10 of 40 failed), then 5 straight down.
        samples = _series(30, 10) + _series(0, 5, start=10000.0)
        verdict = evaluate_verdict("mt5", samples)
        self.assertEqual(verdict.state, "down")

    def test_latency_spike_is_possible(self):
        samples = _series(35, 0)
        for i in range(5):
            samples.append(ProbeSample(at=9000.0 + i, ok=True,
                                       latency_ms=500.0, status=200))
        verdict = evaluate_verdict("exness", samples)
        self.assertEqual(verdict.state, "possible")

    def test_public_signal_lifts_ok_to_possible(self):
        verdict = evaluate_verdict(
            "telegram", _series(40, 0),
            signals=["feed reports: degraded performance"])
        self.assertEqual(verdict.state, "possible")
        self.assertIn("degraded performance",
                      verdict.evidence["signal_refs"][0])

    def test_major_signal_plus_failures_is_down(self):
        verdict = evaluate_verdict(
            "binance", _series(38, 2),
            signals=["feed reports: major outage"])
        self.assertEqual(verdict.state, "down")

    def test_verdict_carries_narrow_claim_label(self):
        verdict = evaluate_verdict("github", _series(40, 0))
        self.assertIn("not crowd", verdict.vantage)
        self.assertIn("one host", verdict.vantage)


class ProbeTests(unittest.TestCase):
    def test_http_ok_and_down_on_loopback(self):
        server, port = _start_fixture()
        try:
            ok = probe_http(f"http://127.0.0.1:{port}/ok")
            self.assertTrue(ok.ok)
            self.assertEqual(ok.status, 200)
            self.assertIsNotNone(ok.latency_ms)
            bad = probe_http(f"http://127.0.0.1:{port}/down")
            self.assertFalse(bad.ok)
            self.assertEqual(bad.status, 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_http_connection_refused_is_a_sample_not_a_raise(self):
        sample = probe_http("http://127.0.0.1:1/nope", timeout=1.0)
        self.assertFalse(sample.ok)
        self.assertTrue(sample.error)

    def test_feed_json_operational_is_ok(self):
        server, port = _start_fixture()
        try:
            sample, refs = probe_feed(
                f"http://127.0.0.1:{port}/feed-ok")
            self.assertTrue(sample.ok)
            self.assertEqual(refs, [])
        finally:
            server.shutdown()
            server.server_close()

    def test_feed_json_major_outage_fails_with_signal_ref(self):
        server, port = _start_fixture()
        try:
            sample, refs = probe_feed(
                f"http://127.0.0.1:{port}/feed-major")
            self.assertFalse(sample.ok)
            self.assertTrue(any("major_outage" in r for r in refs))
        finally:
            server.shutdown()
            server.server_close()

    def test_feed_page_keyword_is_a_signal_ref(self):
        server, port = _start_fixture()
        try:
            sample, refs = probe_feed(
                f"http://127.0.0.1:{port}/feed-page")
            self.assertFalse(sample.ok)
            self.assertTrue(any("major outage" in r for r in refs))
        finally:
            server.shutdown()
            server.server_close()

    def test_feed_degraded_is_reachable_with_signal(self):
        server, port = _start_fixture()
        try:
            sample, refs = probe_feed(
                f"http://127.0.0.1:{port}/feed-degraded")
            self.assertTrue(sample.ok)
            self.assertTrue(any("degraded" in r for r in refs))
        finally:
            server.shutdown()
            server.server_close()

    def test_tcp_probe_loopback(self):
        server, port = _start_fixture()
        try:
            target = WatchTarget(name="local", url=f"127.0.0.1:{port}",
                                 kind="tcp")
            sample, refs = probe_target(target)
            self.assertTrue(sample.ok)
            self.assertEqual(refs, [])
        finally:
            server.shutdown()
            server.server_close()

    def test_unknown_kind_never_raises(self):
        target = WatchTarget(name="weird", url="http://127.0.0.1:1/",
                             kind="carrier-pigeon")
        sample, _refs = probe_target(target)
        self.assertFalse(sample.ok)
        self.assertIn("unknown probe kind", sample.error)


class StoreTests(unittest.TestCase):
    def test_add_list_remove_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            store = WatchtowerStore(home)
            self.assertEqual(store.list_targets(), [])
            target = WatchTarget(name="github",
                                 url="https://github.com",
                                 kind="http", critical=True)
            self.assertTrue(store.add_target(target))
            self.assertFalse(store.add_target(target))
            self.assertEqual([t.name for t in store.list_targets()],
                             ["github"])
            self.assertTrue(store.remove_target("github"))
            self.assertFalse(store.remove_target("github"))
            self.assertEqual(store.list_targets(), [])

    def test_samples_and_incidents_persist(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            store = WatchtowerStore(home)
            store.add_target(WatchTarget(name="x", url="http://x/"))
            for sample in _series(40, 0):
                store.record_sample("x", sample)
            reopened = WatchtowerStore(home)
            self.assertEqual(len(reopened.samples_for("x")), 40)
            incident = reopened.open_incident(
                "x", "down", {"probe_failures": 5}, now=5000.0)
            self.assertEqual(incident.state, "down")
            self.assertFalse(incident.alerted)
            again = WatchtowerStore(home)
            active = again.active_incident("x")
            self.assertIsNotNone(active)
            self.assertEqual(active.since, 5000.0)
            again.resolve_incident("x", now=6000.0)
            self.assertIsNone(WatchtowerStore(home).active_incident("x"))

    def test_corrupt_store_is_empty_not_a_crash(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            path = Path(home) / "data" / "watchtower.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not json", encoding="utf-8")
            store = WatchtowerStore(home)
            self.assertEqual(store.list_targets(), [])


class ReplayTests(unittest.TestCase):
    """The report's proof: a recorded series with a known outage window."""

    def _replay(self, series):
        import tempfile

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            store = WatchtowerStore(home)
            store.add_target(WatchTarget(name="binance",
                                         url="https://binance.com"))
            states = []
            for sample in series:
                store.record_sample("binance", sample)
                verdict = evaluate_verdict(
                    "binance", store.samples_for("binance"))
                states.append(verdict.state)
            return states

    def test_outage_window_flips_at_the_deviation(self):
        # 40 healthy probes, one stray failure (noise), then 10 down.
        series = _series(40, 0)
        series.append(_sample(20000.0, False))
        series.extend(_series(0, 10, start=20240.0))
        states = self._replay(series)
        self.assertEqual(states[40], "possible")  # one stray failure
        self.assertEqual(states[41], "possible")  # second failure
        self.assertEqual(states[42], "down")      # third consecutive
        self.assertEqual(states[-1], "down")

    def test_noise_never_opens_an_incident(self):
        # Isolated stray failures, each followed by healthy probes:
        # at most "possible", never an incident.
        series = _series(40, 0)
        at = 20000.0
        for _ in range(4):
            series.append(_sample(at, False))
            at += 240.0
            series.extend(_series(9, 0, start=at))
            at += 9 * 240.0
        states = self._replay(series)
        self.assertNotIn("down", states)
        self.assertIn("possible", states)


class CheckTargetTests(unittest.TestCase):
    def test_check_opens_and_resolves_incident(self):
        import tempfile

        def failing(_url, _timeout):
            return 503, b"down", 12.0

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            store = WatchtowerStore(home)
            target = WatchTarget(name="svc", url="http://127.0.0.1:9/",
                                 kind="http", critical=True)
            store.add_target(target)
            for sample in _series(40, 0):
                store.record_sample("svc", sample)
            result = check_target(store, target, fetcher=failing,
                                  now=50000.0)
            self.assertEqual(result["state"], "possible")
            self.assertIsNotNone(store.active_incident("svc"))
            for i in range(2):
                result = check_target(store, target, fetcher=failing,
                                      now=50240.0 + i * 240.0)
            self.assertEqual(result["state"], "down")
            self.assertEqual(store.active_incident("svc").state, "down")

    def test_check_never_raises(self):
        import tempfile

        def exploding(_url, _timeout):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            store = WatchtowerStore(home)
            target = WatchTarget(name="svc", url="http://127.0.0.1:9/")
            store.add_target(target)
            result = check_target(store, target, fetcher=exploding,
                                  now=1.0)
            # An exploding fetcher is a failed probe sample, not a
            # failed check: the check succeeds and the error is
            # recorded on the sample, never raised.
            self.assertTrue(result["ok"])
            self.assertEqual(result["state"], "unknown")
            samples = store.samples_for("svc")
            self.assertEqual(len(samples), 1)
            self.assertFalse(samples[0].ok)
            self.assertIn("boom", samples[0].error)


class TickTests(unittest.TestCase):
    def _seed(self, home, *, critical=True, n_ok=40):
        store = WatchtowerStore(home)
        store.add_target(WatchTarget(
            name="binance", url="http://127.0.0.1:9/", kind="http",
            critical=critical))
        for sample in _series(n_ok, 0):
            store.record_sample("binance", sample)
        return store

    def test_quiet_tick_makes_no_network_call(self):
        import tempfile

        calls: list[str] = []

        def counting(_url, _timeout):
            calls.append(_url)
            return 200, b"ok", 1.0

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            result = run_watchtower_tick_once(
                home=home, fetcher=counting, now=100.0)
            self.assertTrue(result["ok"])
            self.assertEqual(result["checked"], 0)
            self.assertEqual(calls, [])

    def test_critical_down_pages_once(self):
        import tempfile

        def failing(_url, _timeout):
            return 503, b"down", 12.0

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            self._seed(home)
            notifier = _FakeNotifier()
            last = None
            for i in range(4):
                last = run_watchtower_tick_once(
                    home=home, notifier=notifier, fetcher=failing,
                    now=50000.0 + i * 240.0)
            # Tick 3 flips to down and pages; tick 4 finds the incident
            # already alerted and stays silent.
            self.assertEqual(last["alerts"], 0)
            self.assertEqual(len(notifier.published), 1)
            self.assertEqual(notifier.published[0]["kind"], "watchtower")
            self.assertIn("binance", notifier.published[0]["title"])
            self.assertTrue(notifier.published[0].get("critical"))

    def test_non_critical_down_never_pages(self):
        import tempfile

        def failing(_url, _timeout):
            return 503, b"down", 12.0

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            self._seed(home, critical=False)
            notifier = _FakeNotifier()
            for _ in range(3):
                run_watchtower_tick_once(
                    home=home, notifier=notifier, fetcher=failing,
                    now=50000.0)
            self.assertEqual(notifier.published, [])

    def test_recovery_never_pages(self):
        import tempfile

        def failing(_url, _timeout):
            return 503, b"down", 12.0

        def healthy(_url, _timeout):
            return 200, b"ok", 12.0

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            self._seed(home)
            notifier = _FakeNotifier()
            for _ in range(3):
                run_watchtower_tick_once(
                    home=home, notifier=notifier, fetcher=failing,
                    now=50000.0)
            self.assertEqual(len(notifier.published), 1)
            for _ in range(12):
                run_watchtower_tick_once(
                    home=home, notifier=notifier, fetcher=healthy,
                    now=60000.0)
            # Still exactly one page: the recovery digests silently.
            self.assertEqual(len(notifier.published), 1)
            self.assertIsNone(
                WatchtowerStore(home).active_incident("binance"))

    def test_tick_never_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            result = run_watchtower_tick_once(home=home, now=1.0)
            self.assertTrue(result["ok"])


class JobTests(unittest.TestCase):
    def test_ensure_job_idempotent(self):
        scheduler = _FakeScheduler()
        job = ensure_watchtower_job(scheduler)
        self.assertIsNotNone(job)
        self.assertEqual(job["spec"], WATCHTOWER_SPEC)
        self.assertEqual(job["name"], WATCHTOWER_JOB)
        again = ensure_watchtower_job(scheduler)
        self.assertEqual(again["id"], job["id"])
        self.assertEqual(len(scheduler.jobs), 1)

    def test_ensure_job_replaces_stale_spec(self):
        scheduler = _FakeScheduler()
        scheduler.add(WATCHTOWER_JOB, "0 * * * *", "tool", {})
        job = ensure_watchtower_job(scheduler)
        self.assertEqual(job["spec"], WATCHTOWER_SPEC)
        self.assertEqual(len(scheduler.jobs), 1)

    def test_tick_tool_name(self):
        self.assertEqual(WATCHTOWER_TOOL, "__watchtower_tick__")


class ReplyTests(unittest.TestCase):
    def test_reply_names_state_and_narrow_claim(self):
        verdict = evaluate_verdict("github", _series(40, 0))
        reply = format_watchtower_reply(
            {"ok": True, "target": "github", "state": verdict.state,
             "reason": verdict.reason, "evidence": verdict.evidence,
             "vantage": verdict.vantage})
        self.assertIn("github", reply)
        self.assertIn("no problems", reply)
        self.assertIn("not crowd", reply)

    def test_incident_serialises(self):
        incident = Incident(target="x", state="down", since=1.0,
                            evidence={"probe_failures": 5})
        back = Incident.from_dict(incident.to_dict())
        self.assertEqual(back.target, "x")
        self.assertEqual(back.evidence["probe_failures"], 5)


class BriefingProviderTests(unittest.TestCase):
    def _ctx(self, home):
        import types

        settings = types.SimpleNamespace(home_path=str(home))
        return types.SimpleNamespace(settings=settings)

    def test_active_incident_gets_a_section(self):
        import tempfile

        from nomorals.agents.morning_briefing import (
            WatchtowerBriefingProvider,
        )

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            store = WatchtowerStore(home)
            store.add_target(WatchTarget(name="binance",
                                         url="https://binance.com",
                                         critical=True))
            store.open_incident("binance", "down",
                                {"probe_failures": 8,
                                 "baseline_deviation": 0.8}, now=1000.0)
            section = WatchtowerBriefingProvider().collect(
                self._ctx(home), since=0.0)
            self.assertIsNotNone(section)
            self.assertIn("binance", section.lines[0])
            self.assertIn("DOWN", section.lines[0])

    def test_no_incident_no_section(self):
        import tempfile

        from nomorals.agents.morning_briefing import (
            WatchtowerBriefingProvider,
        )

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            section = WatchtowerBriefingProvider().collect(
                self._ctx(home), since=0.0)
            self.assertIsNone(section)


class ToolTests(unittest.TestCase):
    def test_service_watch_tool_registers_and_runs(self):
        import tempfile

        from nomorals.tools import watchtower as wt
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        wt.register(reg)
        self.assertIn("service_watch", reg._tools)

        with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
            import types

            class _Reg:
                def __init__(self):
                    self.context = types.SimpleNamespace(
                        settings=types.SimpleNamespace(
                            home_path=str(home)))
                    self.tools = {}

                def register(self, name, **kw):
                    def deco(fn):
                        self.tools[name] = fn
                        return fn
                    return deco

            registry = _Reg()
            wt.register(registry)
            fn = registry.tools["service_watch"]
            watched = fn(action="watch", name="github",
                         url="https://github.com")
            self.assertTrue(watched.get("ok"), watched)
            listed = fn(action="list")
            self.assertIn("github", json.dumps(listed))
            status = fn(action="status", name="github")
            self.assertIn("insufficient baseline", status["reply"])
            removed = fn(action="unwatch", name="github")
            self.assertTrue(removed.get("ok"), removed)

    def test_tool_check_loopback(self):
        import tempfile
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            from nomorals.tools import watchtower as wt

            with tempfile.TemporaryDirectory(dir=str(VERIFY)) as home:
                import types

                class _Reg:
                    def __init__(self):
                        self.context = types.SimpleNamespace(
                            settings=types.SimpleNamespace(
                                home_path=str(home)))
                        self.tools = {}

                    def register(self, name, **kw):
                        def deco(fn):
                            self.tools[name] = fn
                            return fn
                        return deco

                registry = _Reg()
                wt.register(registry)
                fn = registry.tools["service_watch"]
                port = server.server_address[1]
                fn(action="watch", name="local",
                   url=f"http://127.0.0.1:{port}/")
                out = fn(action="check", name="local")
                self.assertTrue(out.get("ok"), out)
                self.assertEqual(out.get("state"), "unknown")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
