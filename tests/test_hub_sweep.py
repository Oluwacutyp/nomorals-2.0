"""Sweep tests for nomorals.hub: the mined upgrades.

Covers the new routes and hardening added in the hub sweep —
long-polling, job introspection, dead-letter replay, per-record sync
push results, pull cursors (last_seq/pending), health/readiness,
Prometheus metrics, rate limiting, CORS, TLS, and env config — plus a
backward-compat pass proving the existing HttpTransport client still
works against the new server.

All live-HTTP against a real HubServer on 127.0.0.1:0 (stdlib urllib).
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from nomorals.hub import (
    DEFAULT_RATE_LIMIT,
    MAX_WAIT_SECONDS,
    HubServer,
    serve,
)
from nomorals.storage.db import Database


def _tmp_db(name: str) -> tuple[Database, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="nm-hub-sweep-"))
    db = Database(str(tmp / name))
    db.migrate()
    return db, tmp


def _http(base: str, method: str, path: str, body=None,
          token: str | None = "s3cret",
          headers: dict[str, str] | None = None,
          timeout: float = 20.0,
          ctx=None) -> tuple[int, dict, bytes]:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _json(raw: bytes):
    return json.loads(raw.decode("utf-8"))


class HubSweepFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db, cls._tmp = _tmp_db("hub.db")
        cls.hub = serve(cls.db, host="127.0.0.1", port=0, token="s3cret",
                        background=True)
        cls.url = cls.hub.url

    @classmethod
    def tearDownClass(cls):
        cls.hub.stop()
        cls.db.close()

    # shortcuts
    def get(self, path, **kw):
        return _http(self.url, "GET", path, **kw)

    def post(self, path, body=None, **kw):
        return _http(self.url, "POST", path, body=body, **kw)

    def node(self, name="worker-1", **kw):
        status, _, raw = self.post("/mesh/register",
                                   {"name": name, "platform": "linux",
                                    "capabilities": ["exec"], **kw})
        self.assertEqual(status, 200, raw)
        return _json(raw)["node"]


# ── ops: index / health / ready / metrics ─────────────────────────────────


class OpsTest(HubSweepFixture):
    def test_index_no_auth(self):
        status, _, raw = self.get("/", token=None)
        self.assertEqual(status, 200)
        doc = _json(raw)
        self.assertEqual(doc["service"], "nomorals-hub")
        self.assertIn("GET  /health", doc["routes"])
        self.assertIn("POST /mesh/poll", doc["routes"])

    def test_health_rich_no_auth(self):
        status, _, raw = self.get("/health", token=None)
        self.assertEqual(status, 200)
        doc = _json(raw)
        self.assertTrue(doc["ok"])
        self.assertIn("version", doc)
        self.assertIn("uptime_seconds", doc)
        self.assertFalse(doc["tls"])
        self.assertEqual(doc["components"]["mesh"]["status"], "SERVING")
        self.assertEqual(doc["components"]["sync"]["status"], "SERVING")
        self.assertIn("latency_ms", doc["components"]["mesh"])

    def test_ready(self):
        status, _, raw = self.get("/ready", token=None)
        self.assertEqual(status, 200)
        self.assertTrue(_json(raw)["ready"])

    def test_metrics_prometheus_format(self):
        # hit a route first so the counter is non-empty
        self.get("/health", token=None)
        status, headers, raw = self.get("/metrics")
        self.assertEqual(status, 200)
        self.assertIn("version=0.0.4", headers.get("Content-Type", ""))
        text = raw.decode("utf-8")
        self.assertIn("# HELP nomorals_hub_requests_total", text)
        self.assertIn("# TYPE nomorals_hub_requests_total counter", text)
        self.assertIn('nomorals_hub_requests_total{route="/health",'
                      'status="2xx"}', text)
        self.assertIn("nomorals_hub_uptime_seconds", text)
        self.assertIn("nomorals_hub_nodes_active", text)

    def test_metrics_requires_auth(self):
        status, _, raw = self.get("/metrics", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(_json(raw)["code"], "unauthorized")

    def test_unknown_route_404(self):
        status, _, raw = self.get("/nope")
        self.assertEqual(status, 404)
        self.assertEqual(_json(raw)["code"], "not_found")


# ── auth ──────────────────────────────────────────────────────────────────


class AuthTest(HubSweepFixture):
    def test_missing_token_401(self):
        status, _, raw = self.get("/mesh/nodes", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(_json(raw)["code"], "unauthorized")

    def test_wrong_token_401(self):
        status, _, raw = self.get("/mesh/nodes", token="wrong")
        self.assertEqual(status, 401)

    def test_ok_token_200(self):
        status, _, raw = self.get("/mesh/nodes")
        self.assertEqual(status, 200)

    def test_non_loopback_without_token_refused(self):
        db, tmp = _tmp_db("refuse.db")
        try:
            with self.assertRaises(ValueError):
                HubServer(db, host="0.0.0.0")
        finally:
            db.close()

    def test_env_token_fallback(self):
        db, tmp = _tmp_db("env.db")
        try:
            os.environ["NM_HUB_TOKEN"] = "envtok"
            srv = HubServer(db, host="127.0.0.1")
            self.assertEqual(srv.token, "envtok")
            del os.environ["NM_HUB_TOKEN"]
            os.environ["NM_HUB_RATE_LIMIT"] = "0"
            srv2 = HubServer(db, host="127.0.0.1")
            self.assertIsNone(srv2._limiter)
        finally:
            os.environ.pop("NM_HUB_TOKEN", None)
            os.environ.pop("NM_HUB_RATE_LIMIT", None)
            db.close()

    def test_cert_key_must_pair(self):
        db, tmp = _tmp_db("pair.db")
        try:
            with self.assertRaises(ValueError):
                HubServer(db, host="127.0.0.1", certfile="/tmp/x.pem")
        finally:
            db.close()


# ── mesh: nodes ───────────────────────────────────────────────────────────


class NodesTest(HubSweepFixture):
    def test_register_heartbeat_deregister(self):
        node = self.node("sweep-node", labels={"role": "test"})
        nid = node["node_id"]
        status, _, raw = self.post("/mesh/heartbeat",
                                   {"node_id": nid, "info": {"load": 0.1}})
        self.assertEqual(status, 200, raw)

        status, _, raw = self.get("/mesh/nodes")
        nodes = _json(raw)["nodes"]
        self.assertTrue(any(n["node_id"] == nid for n in nodes))

        status, _, raw = self.get("/mesh/nodes?all=1")
        self.assertEqual(status, 200)
        self.assertTrue(any(n["node_id"] == nid
                            for n in _json(raw)["nodes"]))

        status, _, raw = self.post("/mesh/deregister", {"node_id": nid})
        self.assertEqual(status, 200, raw)
        status, _, raw = self.get("/mesh/nodes")
        self.assertFalse(any(n["node_id"] == nid
                             for n in _json(raw)["nodes"]))

    def test_deregister_unknown_404(self):
        status, _, raw = self.post("/mesh/deregister",
                                   {"node_id": "no-such-node"})
        self.assertEqual(status, 404)


# ── mesh: dispatch / poll / complete / fail ───────────────────────────────


class TasksTest(HubSweepFixture):
    def test_dispatch_dedupe_key_idempotent(self):
        body = {"task_type": "sweep.echo", "payload": {"n": 1},
                "origin_node": "t", "dedupe_key": "sweep-dedupe-1"}
        _, _, raw1 = self.post("/mesh/dispatch", body)
        _, _, raw2 = self.post("/mesh/dispatch", body)
        j1 = _json(raw1)["job_id"]
        j2 = _json(raw2)["job_id"]
        self.assertEqual(j1, j2)
        # without the key a second dispatch is a new job
        del body["dedupe_key"]
        _, _, raw3 = self.post("/mesh/dispatch", body)
        self.assertNotEqual(j1, _json(raw3)["job_id"])

    def test_dispatch_many(self):
        _, _, raw = self.post("/mesh/dispatch-many", {
            "task_type": "sweep.fan", "origin_node": "t",
            "payloads": [{"i": 0}, {"i": 1}, {"i": 2}]})
        job_ids = _json(raw)["job_ids"]
        self.assertEqual(len(job_ids), 3)
        self.assertEqual(len(set(job_ids)), 3)

    def test_dispatch_bad_payload_400(self):
        status, _, raw = self.post("/mesh/dispatch", {
            "task_type": "x", "payload": [1, 2], "origin_node": "t"})
        self.assertEqual(status, 400)

    def test_poll_complete_result_flow(self):
        nid = self.node("flow-node")["node_id"]
        _, _, raw = self.post("/mesh/dispatch", {
            "task_type": "sweep.flow", "payload": {"q": "?"},
            "origin_node": "t", "target_node": nid})
        job_id = _json(raw)["job_id"]

        _, _, raw = self.post("/mesh/poll", {"node_id": nid, "batch": 5})
        tasks = _json(raw)["tasks"]
        self.assertTrue(any(t["job_id"] == job_id for t in tasks))

        # progress heartbeat extends the lease and checkpoints detail
        _, _, raw = self.post("/mesh/progress", {
            "job_id": job_id, "node_id": nid,
            "detail": {"pct": 50}})
        self.assertTrue(_json(raw)["lease_alive"])

        _, _, raw = self.post("/mesh/complete", {
            "job_id": job_id, "result": {"answer": 42}})
        self.assertEqual(_json(raw)["ok"], True)

        _, _, raw = self.get(f"/mesh/job?job_id={job_id}")
        job = _json(raw)["job"]
        self.assertEqual(job["result"], {"answer": 42})
        self.assertEqual(job["progress"], {"pct": 50})

    def test_poll_wait_longpoll(self):
        nid = self.node("wait-node")["node_id"]
        got: dict = {}

        def _poll():
            started = time.monotonic()
            status, _, raw = self.post(
                "/mesh/poll", {"node_id": nid, "batch": 5, "wait": 5},
                timeout=15)
            got["status"] = status
            got["elapsed"] = time.monotonic() - started
            got["tasks"] = _json(raw)["tasks"] if status == 200 else []

        th = threading.Thread(target=_poll, daemon=True)
        th.start()
        time.sleep(0.6)
        _, _, raw = self.post("/mesh/dispatch", {
            "task_type": "sweep.wake", "payload": {}, "origin_node": "t",
            "target_node": nid})
        job_id = _json(raw)["job_id"]
        th.join(timeout=15)
        self.assertFalse(th.is_alive(), "long-poll did not return")
        self.assertEqual(got["status"], 200)
        self.assertLess(got["elapsed"], 4.5, "poll slept through the task")
        self.assertTrue(any(t["job_id"] == job_id for t in got["tasks"]))

    def test_poll_wait_capped(self):
        nid = self.node("cap-node")["node_id"]
        started = time.monotonic()
        status, _, raw = self.post(
            "/mesh/poll", {"node_id": nid, "batch": 1,
                           "wait": MAX_WAIT_SECONDS + 1000},
            timeout=MAX_WAIT_SECONDS + 20)
        elapsed = time.monotonic() - started
        self.assertEqual(status, 200)
        self.assertEqual(_json(raw)["tasks"], [])
        self.assertLess(elapsed, MAX_WAIT_SECONDS + 5)

    def test_cancel(self):
        nid = self.node("cancel-node")["node_id"]
        _, _, raw = self.post("/mesh/dispatch", {
            "task_type": "sweep.cancel", "payload": {},
            "origin_node": "t", "target_node": nid})
        job_id = _json(raw)["job_id"]
        _, _, raw = self.post("/mesh/cancel", {"job_id": job_id})
        self.assertTrue(_json(raw)["cancelled"])
        # second cancel is a no-op, not an error
        _, _, raw = self.post("/mesh/cancel", {"job_id": job_id})
        self.assertFalse(_json(raw)["cancelled"])

    def test_fail_dead_retry(self):
        nid = self.node("dead-node")["node_id"]
        _, _, raw = self.post("/mesh/dispatch", {
            "task_type": "sweep.doomed", "payload": {},
            "origin_node": "t", "target_node": nid})
        job_id = _json(raw)["job_id"]
        self.post("/mesh/poll", {"node_id": nid, "batch": 5})
        self.post("/mesh/fail", {"job_id": job_id, "error": "boom",
                                 "retry": False})

        _, _, raw = self.get("/mesh/dead")
        dead = _json(raw)["jobs"]
        self.assertTrue(any(j["job_id"] == job_id for j in dead))

        _, _, raw = self.post("/mesh/retry-dead", {"job_id": job_id})
        self.assertTrue(_json(raw)["replayed"])

        _, _, raw = self.get("/mesh/dead")
        self.assertFalse(any(j["job_id"] == job_id
                             for j in _json(raw)["jobs"]))

    def test_stats_and_jobs(self):
        nid = self.node("stats-node")["node_id"]
        for i in range(2):
            self.post("/mesh/dispatch", {
                "task_type": "sweep.stat", "payload": {"i": i},
                "origin_node": "t", "target_node": nid})
        _, _, raw = self.get("/mesh/stats")
        totals = _json(raw)["stats"]["totals"]
        self.assertGreaterEqual(totals.get("ready", 0), 2)
        _, _, raw = self.get(f"/mesh/jobs?node={nid}&limit=10")
        jobs = _json(raw)["jobs"]
        self.assertGreaterEqual(len(jobs), 2)

    def test_reclaim_reap_shape(self):
        _, _, raw = self.post("/mesh/reclaim", {})
        self.assertIn("reclaimed", _json(raw))
        _, _, raw = self.post("/mesh/reap-expired", {})
        self.assertIn("reaped", _json(raw))

    def test_job_not_found_404(self):
        status, _, raw = self.get("/mesh/job?job_id=missing")
        self.assertEqual(status, 404)
        self.assertEqual(_json(raw)["code"], "job_not_found")

    def test_wait_for_result(self):
        nid = self.node("waitres-node")["node_id"]
        _, _, raw = self.post("/mesh/dispatch", {
            "task_type": "sweep.waitres", "payload": {},
            "origin_node": "t", "target_node": nid})
        job_id = _json(raw)["job_id"]
        self.post("/mesh/poll", {"node_id": nid, "batch": 5})

        def _finish():
            time.sleep(0.6)
            _http(self.url, "POST", "/mesh/complete",
                  {"job_id": job_id, "result": "done!"})

        th = threading.Thread(target=_finish, daemon=True)
        th.start()
        started = time.monotonic()
        status, _, raw = self.get(
            f"/mesh/wait?job_id={job_id}&timeout=5", timeout=15)
        elapsed = time.monotonic() - started
        th.join(timeout=10)
        doc = _json(raw)
        self.assertEqual(status, 200)
        self.assertTrue(doc["ready"])
        self.assertEqual(doc["result"], "done!")
        self.assertLess(elapsed, 4.5)

    def test_wait_timeout_not_ready(self):
        nid = self.node("waitto-node")["node_id"]
        _, _, raw = self.post("/mesh/dispatch", {
            "task_type": "sweep.never", "payload": {},
            "origin_node": "t", "target_node": nid})
        job_id = _json(raw)["job_id"]
        status, _, raw = self.get(f"/mesh/wait?job_id={job_id}&timeout=0.5",
                                  timeout=10)
        doc = _json(raw)
        self.assertEqual(status, 200)
        self.assertFalse(doc["ready"])

    def test_wait_unknown_404(self):
        status, _, raw = self.get("/mesh/wait?job_id=missing&timeout=1")
        self.assertEqual(status, 404)


# ── sync ──────────────────────────────────────────────────────────────────


class SyncTest(HubSweepFixture):
    def _record(self, key, value):
        return {"key": key, "value": value, "updated_at": time.time(),
                "device_id": "test", "deleted": False,
                "hlc_ts": time.time(), "hlc_count": 0, "clocks": {}}

    def test_push_pull_roundtrip(self):
        recs = [self._record("sweep:a", {"v": 1}),
                self._record("sweep:b", {"v": 2})]
        status, _, raw = self.post("/sync/push", {"records": recs})
        doc = _json(raw)
        self.assertEqual(status, 200)
        self.assertEqual(doc["applied"], 2)
        self.assertEqual(len(doc["results"]), 2)
        self.assertTrue(all(r["ok"] for r in doc["results"]))

        status, _, raw = self.get("/sync/pull?since_seq=0&limit=100")
        doc = _json(raw)
        self.assertEqual(status, 200)
        keys = {r["key"] for r in doc["records"]}
        self.assertTrue({"sweep:a", "sweep:b"} <= keys)
        self.assertGreater(doc["last_seq"], 0)
        self.assertEqual(doc["pending"], 0)

    def test_push_per_record_errors(self):
        recs = ["not-a-dict",
                {"key": "sweep:good", "value": {"v": 1},
                 "updated_at": time.time(), "device_id": "test",
                 "deleted": False, "hlc_ts": time.time(), "hlc_count": 0,
                 "clocks": {}},
                {"key": "sweep:bad", "updated_at": "not-a-time",
                 "value": {}}]  # bad updated_at → from_dict raises
        status, _, raw = self.post("/sync/push", {"records": recs})
        doc = _json(raw)
        self.assertEqual(status, 200)
        self.assertEqual(doc["applied"], 1)
        self.assertFalse(doc["results"][0]["ok"])
        self.assertTrue(doc["results"][1]["ok"])
        self.assertFalse(doc["results"][2]["ok"])

    def test_pull_needs_cursor(self):
        status, _, raw = self.get("/sync/pull")
        self.assertEqual(status, 400)

    def test_pull_bad_cursor(self):
        status, _, raw = self.get("/sync/pull?since_seq=abc")
        self.assertEqual(status, 400)

    def test_pull_longpoll_wakes_on_push(self):
        status, _, raw = self.get("/sync/pull?since_seq=0&limit=1")
        head = _json(raw)["last_seq"]
        got: dict = {}

        def _pull():
            started = time.monotonic()
            status, _, raw = self.get(
                f"/sync/pull?since_seq={head}&timeout=5", timeout=15)
            got["status"] = status
            got["elapsed"] = time.monotonic() - started
            got["doc"] = _json(raw) if status == 200 else {}

        th = threading.Thread(target=_pull, daemon=True)
        th.start()
        time.sleep(0.6)
        self.post("/sync/push",
                  {"records": [self._record("sweep:wake", {"v": 9})]})
        th.join(timeout=15)
        self.assertFalse(th.is_alive(), "sync long-poll did not return")
        self.assertEqual(got["status"], 200)
        self.assertLess(got["elapsed"], 4.5, "pull slept through the push")
        keys = {r["key"] for r in got["doc"]["records"]}
        self.assertIn("sweep:wake", keys)
        self.assertEqual(got["doc"]["pending"], 0)


# ── hardening: rate limit / CORS / TLS ────────────────────────────────────


class HardeningTest(unittest.TestCase):
    def _srv(self, **kw):
        db, tmp = _tmp_db("hard.db")
        hub = serve(db, host="127.0.0.1", port=0, token="s3cret",
                    background=True, **kw)
        self.addCleanup(hub.stop)
        self.addCleanup(db.close)
        return hub

    def test_rate_limit_429_with_retry_after(self):
        hub = self._srv(rate_limit=3, rate_window=60.0)
        codes = []
        for _ in range(5):
            status, headers, _ = _http(hub.url, "GET", "/mesh/stats")
            codes.append(status)
        self.assertEqual(codes[:3], [200, 200, 200])
        self.assertIn(429, codes[3:])
        # the 429 carries Retry-After
        for _ in range(6):
            status, headers, _ = _http(hub.url, "GET", "/mesh/stats")
            if status == 429:
                self.assertIn("Retry-After", headers)
                break
        else:
            self.fail("never got a 429 to inspect headers")
        # probes are exempt from rate limiting
        status, _, _ = _http(hub.url, "GET", "/health", token=None)
        self.assertEqual(status, 200)

    def test_rate_limit_disabled_by_zero(self):
        hub = self._srv(rate_limit=0)
        for _ in range(5):
            status, _, _ = _http(hub.url, "GET", "/mesh/stats")
            self.assertEqual(status, 200)

    def test_cors_preflight_and_echo(self):
        hub = self._srv(cors_origins=("https://dash.example",))
        status, headers, _ = _http(hub.url, "OPTIONS", "/mesh/nodes",
                                   token=None)
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"),
                         "https://dash.example")
        status, headers, _ = _http(
            hub.url, "GET", "/mesh/nodes",
            headers={"Origin": "https://dash.example"})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"),
                         "https://dash.example")
        # non-listed origin gets no CORS headers
        status, headers, _ = _http(
            hub.url, "GET", "/mesh/nodes",
            headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 200)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_cors_off_by_default(self):
        hub = self._srv()
        status, _, _ = _http(hub.url, "OPTIONS", "/mesh/nodes", token=None)
        self.assertEqual(status, 404)

    def test_tls(self):
        if shutil.which("openssl") is None:
            self.skipTest("openssl not available")
        tmp = Path(tempfile.mkdtemp(prefix="nm-hub-tls-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        cert, key = str(tmp / "cert.pem"), str(tmp / "key.pem")
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048",
             "-keyout", key, "-out", cert, "-days", "1", "-nodes",
             "-subj", "/CN=127.0.0.1"],
            check=True, capture_output=True)
        hub = self._srv(certfile=cert, keyfile=key)
        self.assertTrue(hub.url.startswith("https://"))
        ctx = ssl._create_unverified_context()
        status, _, raw = _http(hub.url, "GET", "/health", token=None,
                               ctx=ctx)
        self.assertEqual(status, 200)
        self.assertTrue(_json(raw)["tls"])


# ── backward compat: the existing HTTP clients ────────────────────────────


class ClientCompatTest(HubSweepFixture):
    def test_http_transport_end_to_end(self):
        from nomorals.mesh.http_transport import HttpTransport
        t = HttpTransport(self.url, token="s3cret")
        try:
            node = t.register("compat-node", platform="linux",
                              capabilities=["exec"])
            t.heartbeat(node.node_id)
            self.assertTrue(any(n.node_id == node.node_id
                                for n in t.active_nodes()))
            job_id = t.dispatch("compat.task", {"x": 1},
                                origin_node="compat-origin",
                                target_node=node.node_id)
            tasks = t.poll(node.node_id, batch=5)
            self.assertTrue(any(x.job_id == job_id for x in tasks))
            t.complete(job_id, result={"ok": True})
            self.assertGreater(t.ping(), 0)
        finally:
            t.close()

    def test_http_sync_peer_end_to_end(self):
        from nomorals.sync.http_peer import HttpSyncPeer
        from nomorals.sync.store import SyncStore
        peer = HttpSyncPeer(self.url, token="s3cret")
        # a remote device has its OWN database — sharing the hub's db
        # would make the push a no-op by LWW design
        dev_db, dev_tmp = _tmp_db("compat-dev.db")
        self.addCleanup(dev_db.close)
        store = SyncStore(dev_db, device_id="compat-dev")
        store.put("compat:k", {"v": 1})
        recs = [store.get_record("compat:k")]
        self.assertEqual(peer.push_records(recs), 1)
        back = peer.fetch_since_seq(0)
        self.assertTrue(any(r.key == "compat:k" for r in back))


if __name__ == "__main__":
    unittest.main()
