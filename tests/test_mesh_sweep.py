"""Sweep tests for the upgraded mesh: presence, dispatch, resilience.

Covers the new capability surface added in the mesh sweep:
labels/selectors, suspicion window, graceful leave, RetryPolicy,
task heartbeats, dedupe keys, expiry, capability routing, queue
introspection, jittered HTTP retries, and the circuit breaker.
"""
from __future__ import annotations

import threading
import time
import unittest

from nomorals.mesh import (
    AuthError,
    CircuitBreaker,
    CircuitOpen,
    HttpTransport,
    HubUnreachable,
    LocalTransport,
    MeshError,
    MeshNode,
    MeshTask,
    MeshTasks,
    NodeRegistry,
    NodeSuspect,
    NodeUnknown,
    PayloadTooLarge,
    RetryPolicy,
    TaskNotFound,
    TransportError,
    format_nodes_table,
    format_tasks_table,
)
from nomorals.mesh.http_transport import _full_jitter, _retry_after_seconds
from nomorals.storage.db import Database


def _db() -> Database:
    from nomorals.storage.migrations import MIGRATIONS
    from nomorals.storage.schema import MigrationRunner
    db = Database(":memory:")
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


def _file_db() -> Database:
    """File-backed DB: Database connections are thread-local, so threads
    sharing an in-memory DB each see an empty database. Tests that span
    threads must use a file."""
    import tempfile
    from nomorals.storage.migrations import MIGRATIONS
    from nomorals.storage.schema import MigrationRunner
    path = tempfile.mktemp(suffix=".db")
    db = Database(path)
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


def _age_node(reg: NodeRegistry, node_id: str, seconds: float) -> None:
    reg.db.execute(
        "UPDATE mesh_nodes SET last_seen=? WHERE node_id=?",
        (time.time() - seconds, node_id),
    )


class NodeLabelsTests(unittest.TestCase):
    def setUp(self):
        self.reg = NodeRegistry(_db())

    def test_labels_round_trip(self):
        n = self.reg.register("gpu", labels={"gpu": "true", "region": "eu"})
        got = self.reg.get(n.node_id)
        assert got is not None
        self.assertEqual(got.labels, {"gpu": "true", "region": "eu"})
        # Wire format round-trip.
        n2 = MeshNode.from_dict(n.to_dict())
        self.assertEqual(n2.labels, {"gpu": "true", "region": "eu"})

    def test_matches_semantics(self):
        n = MeshNode(node_id="x", name="n", platform="linux",
                     capabilities=["gpu", "camera"],
                     labels={"region": "eu", "gpu": "true"})
        self.assertTrue(n.matches())
        self.assertTrue(n.matches(capabilities=["gpu"]))
        self.assertTrue(n.matches(labels={"region": "eu"}))
        self.assertFalse(n.matches(capabilities=["tpu"]))
        self.assertFalse(n.matches(labels={"region": "us"}))
        self.assertFalse(n.matches(labels={"nope": "1"}))

    def test_select_by_capabilities_and_labels(self):
        self.reg.register("a", capabilities=["gpu"], labels={"region": "eu"})
        self.reg.register("b", capabilities=["gpu"], labels={"region": "us"})
        self.reg.register("c", capabilities=["camera"])
        got = self.reg.select(capabilities=["gpu"], labels={"region": "eu"})
        self.assertEqual([n.name for n in got], ["a"])
        self.assertEqual(self.reg.select(capabilities=["tpu"]), [])
        limited = self.reg.select(capabilities=["gpu"], limit=1)
        self.assertEqual(len(limited), 1)

    def test_select_ignores_stale_nodes(self):
        n = self.reg.register("old", capabilities=["gpu"])
        _age_node(self.reg, n.node_id, 3600)
        self.assertEqual(self.reg.select(capabilities=["gpu"]), [])

    def test_update_labels(self):
        n = self.reg.register("n")
        updated = self.reg.update_labels(n.node_id, {"zone": "a"})
        self.assertEqual(updated.labels, {"zone": "a"})
        with self.assertRaises(NodeUnknown):
            self.reg.update_labels("missing", {"zone": "a"})


class PresenceTests(unittest.TestCase):
    def setUp(self):
        self.reg = NodeRegistry(_db())

    def test_suspicion_window(self):
        ready = self.reg.register("ready")
        suspect = self.reg.register("suspect")
        gone = self.reg.register("gone")
        _age_node(self.reg, suspect.node_id, 300)     # past stale, within prune
        _age_node(self.reg, gone.node_id, 5000)       # past prune horizon
        suspects = self.reg.list_suspect()
        self.assertEqual([n.name for n in suspects], ["suspect"])
        self.assertEqual(self.reg.get(suspect.node_id).presence(), "suspect")
        self.assertEqual(self.reg.get(gone.node_id).presence(), "gone")
        self.assertEqual(self.reg.get(ready.node_id).presence(), "ready")

    def test_suspect_can_refute_by_heartbeating(self):
        n = self.reg.register("flaky")
        _age_node(self.reg, n.node_id, 300)
        self.assertEqual(len(self.reg.list_suspect()), 1)
        self.reg.heartbeat(n.node_id)
        self.assertEqual(self.reg.list_suspect(), [])

    def test_prune_only_removes_gone(self):
        suspect = self.reg.register("suspect")
        gone = self.reg.register("gone")
        _age_node(self.reg, suspect.node_id, 300)
        _age_node(self.reg, gone.node_id, 5000)
        removed = self.reg.prune()
        self.assertEqual(removed, 1)
        self.assertIsNotNone(self.reg.get(suspect.node_id))
        self.assertIsNone(self.reg.get(gone.node_id))

    def test_deregister(self):
        n = self.reg.register("leaving")
        removed = self.reg.deregister(n.node_id)
        self.assertEqual(removed.node_id, n.node_id)
        self.assertIsNone(self.reg.get(n.node_id))
        with self.assertRaises(NodeUnknown):
            self.reg.deregister(n.node_id)

    def test_heartbeat_info(self):
        n = self.reg.register("n")
        self.reg.heartbeat(n.node_id, info={"load": 0.5, "version": "2.0"})
        got = self.reg.get(n.node_id)
        assert got is not None
        self.assertEqual(got.info["load"], 0.5)
        self.assertEqual(got.info["version"], "2.0")


class PresentationTests(unittest.TestCase):
    def test_describe_and_tables(self):
        n = MeshNode(node_id="abcdef123456", name="phone", platform="termux",
                     capabilities=["camera"], labels={"zone": "a"},
                     last_seen=time.time())
        text = n.describe()
        self.assertIn("phone", text)
        self.assertIn("camera", text)
        self.assertIn("zone=a", text)
        table = format_nodes_table([n])
        self.assertIn("phone", table)
        self.assertIn("state", table)
        self.assertEqual(format_nodes_table([]), "no mesh nodes registered")

        t = MeshTask(job_id="job123456", task_type="render", payload={},
                     target_node=None, origin_node="o1", priority=3,
                     attempts=1, status="leased")
        self.assertIn("render", t.describe())
        self.assertIn("broadcast", t.describe())
        table = format_tasks_table([t])
        self.assertIn("render", table)
        self.assertEqual(format_tasks_table([]), "no mesh tasks")


class RetryPolicyTests(unittest.TestCase):
    def test_jitter_bounds(self):
        rp = RetryPolicy(initial_delay=1.0, backoff=2.0, max_delay=10.0)
        for attempt in range(6):
            d = rp.next_delay(attempt)
            cap = min(10.0, 1.0 * 2.0 ** attempt)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, cap)

    def test_non_retryable(self):
        rp = RetryPolicy(non_retryable_errors=["validation", "auth"])
        self.assertTrue(rp.is_retryable("connection reset"))
        self.assertFalse(rp.is_retryable("VALIDATION failed: bad input"))
        self.assertFalse(rp.is_retryable("auth token expired"))

    def test_wire_round_trip(self):
        rp = RetryPolicy(initial_delay=2.0, non_retryable_errors=["nope"])
        rp2 = RetryPolicy.from_dict(rp.to_dict())
        assert rp2 is not None
        self.assertEqual(rp2.initial_delay, 2.0)
        self.assertEqual(rp2.non_retryable_errors, ["nope"])
        self.assertIsNone(RetryPolicy.from_dict(None))
        self.assertIsNone(RetryPolicy.from_dict({}))


class MeshTasksSweepTests(unittest.TestCase):
    def setUp(self):
        self.db = _db()
        self.reg = NodeRegistry(self.db)
        self.tasks = MeshTasks(self.db, registry=self.reg)
        self.origin = self.reg.register("origin").node_id

    def test_retry_policy_non_retryable_goes_dead(self):
        rp = RetryPolicy(max_attempts=5, non_retryable_errors=["validation"])
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin,
                                  retry_policy=rp)
        tasks = self.tasks.poll(self.origin)
        self.tasks.fail(tasks[0].job_id, error="validation: bad payload")
        dead = self.tasks.dead()
        self.assertEqual([t.job_id for t in dead], [jid])

    def test_retry_policy_retryable_requeues(self):
        rp = RetryPolicy(max_attempts=5, non_retryable_errors=["validation"])
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin,
                                  retry_policy=rp)
        tasks = self.tasks.poll(self.origin)
        self.tasks.fail(tasks[0].job_id, error="connection reset")
        # Still live (requeued with backoff), not dead.
        self.assertEqual(self.tasks.dead(), [])
        self.assertGreaterEqual(self.tasks.stats()["totals"].get("ready", 0), 0)
        # Poll again once available_at passes — reclaim via fresh poll after
        # forcing availability.
        self.db.execute(
            "UPDATE work_queue SET available_at=0 WHERE id=?", (jid,))
        again = self.tasks.poll(self.origin)
        self.assertEqual([t.job_id for t in again], [jid])

    def test_dedupe_key(self):
        j1 = self.tasks.dispatch("t", {"n": 1}, origin_node=self.origin,
                                 dedupe_key="once")
        j2 = self.tasks.dispatch("t", {"n": 1}, origin_node=self.origin,
                                 dedupe_key="once")
        self.assertEqual(j1, j2)
        self.assertEqual(self.tasks.pending_count(), 1)
        # After the task completes, the key is free again.
        tasks = self.tasks.poll(self.origin)
        self.tasks.complete(tasks[0].job_id)
        j3 = self.tasks.dispatch("t", {"n": 1}, origin_node=self.origin,
                                 dedupe_key="once")
        self.assertNotEqual(j3, j1)

    def test_expire_after_and_reap(self):
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin,
                                  expire_after=0.05)
        time.sleep(0.08)
        reaped = self.tasks.reap_expired()
        self.assertEqual(reaped, 1)
        self.assertEqual([t.job_id for t in self.tasks.dead()], [jid])
        self.assertEqual(self.tasks.poll(self.origin), [])

    def test_reap_ignores_unexpired(self):
        self.tasks.dispatch("t", {}, origin_node=self.origin,
                            expire_after=600)
        self.assertEqual(self.tasks.reap_expired(), 0)
        self.assertEqual(self.tasks.pending_count(), 1)

    def test_task_heartbeat_progress_and_lease(self):
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin)
        tasks = self.tasks.poll(self.origin, lease_seconds=60)
        alive = self.tasks.heartbeat(tasks[0].job_id, self.origin,
                                     detail={"pct": 42, "stage": "encode"})
        self.assertTrue(alive)
        self.assertEqual(self.tasks.progress(jid),
                         {"pct": 42, "stage": "encode"})
        # A different node cannot extend a lease it doesn't own.
        other = self.reg.register("other").node_id
        self.assertFalse(
            self.tasks.heartbeat(tasks[0].job_id, other, detail={}))

    def test_progress_empty_when_never_heartbeated(self):
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin)
        self.assertEqual(self.tasks.progress(jid), {})

    def test_cancel(self):
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin)
        self.assertTrue(self.tasks.cancel(jid))
        self.assertEqual(self.tasks.poll(self.origin), [])
        self.assertFalse(self.tasks.cancel(jid))
        self.assertFalse(self.tasks.cancel("missing"))

    def test_result_and_wait(self):
        # Threaded: needs a file-backed DB (see _file_db).
        self.db = _file_db()
        self.reg = NodeRegistry(self.db)
        self.tasks = MeshTasks(self.db, registry=self.reg)
        self.origin = self.reg.register("origin").node_id
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin)
        self.assertIsNone(self.tasks.result(jid))  # unfinished
        with self.assertRaises(TaskNotFound):
            self.tasks.result("missing")
        with self.assertRaises(TaskNotFound):
            self.tasks.wait_for_result("missing")

        def _finish():
            time.sleep(0.15)
            tasks = self.tasks.poll(self.origin)
            self.tasks.complete(tasks[0].job_id, result={"ok": True})

        threading.Thread(target=_finish, daemon=True).start()
        self.assertEqual(self.tasks.wait_for_result(jid, timeout=5),
                         {"ok": True})

    def test_wait_for_result_timeout(self):
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin)
        with self.assertRaises(TimeoutError):
            self.tasks.wait_for_result(jid, timeout=0.1)

    def test_fail_unknown_raises(self):
        with self.assertRaises(TaskNotFound):
            self.tasks.fail("missing", error="x")

    def test_retry_dead(self):
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin,
                                  max_attempts=1)
        tasks = self.tasks.poll(self.origin)
        self.tasks.fail(tasks[0].job_id, error="boom")
        self.assertEqual(len(self.tasks.dead()), 1)
        self.assertTrue(self.tasks.retry_dead(jid))
        self.assertEqual(self.tasks.dead(), [])
        self.assertEqual(len(self.tasks.poll(self.origin)), 1)

    def test_stats(self):
        self.tasks.dispatch("a", {}, origin_node=self.origin)
        self.tasks.dispatch("b", {}, origin_node=self.origin,
                            target_node=self.origin)
        stats = self.tasks.stats()
        self.assertEqual(stats["totals"].get("ready"), 2)
        self.assertIn("mesh:broadcast", stats["topics"])
        self.assertIn(f"mesh:node:{self.origin}", stats["topics"])

    def test_dispatch_many(self):
        ids = self.tasks.dispatch_many("t", [{"n": i} for i in range(4)],
                                       origin_node=self.origin)
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(self.tasks.pending_count(), 4)

    def test_capability_routing(self):
        worker = self.reg.register("gpu", capabilities=["gpu"]).node_id
        jid = self.tasks.dispatch("encode", {}, origin_node=self.origin,
                                  target_capabilities=["gpu"])
        tasks = self.tasks.poll(worker)
        self.assertEqual([t.job_id for t in tasks], [jid])
        # Origin node must not see another node's targeted task.
        self.assertEqual(
            [t.job_id for t in self.tasks.poll(self.origin)
             if t.job_id == jid], [])

    def test_capability_routing_no_match(self):
        with self.assertRaises(MeshError):
            self.tasks.dispatch("t", {}, origin_node=self.origin,
                                target_capabilities=["tpu"])

    def test_capability_routing_needs_registry(self):
        tasks = MeshTasks(_db())  # no registry
        with self.assertRaises(ValueError):
            tasks.dispatch("t", {}, origin_node="o",
                           target_capabilities=["gpu"])

    def test_payload_too_large(self):
        with self.assertRaises(PayloadTooLarge):
            self.tasks.dispatch("t", {"blob": "x" * (2 * 1024 * 1024)},
                                origin_node=self.origin)

    def test_reclaim(self):
        jid = self.tasks.dispatch("t", {}, origin_node=self.origin,
                                  )
        self.tasks.poll(self.origin, lease_seconds=0.01)
        time.sleep(0.03)
        self.assertGreaterEqual(self.tasks.reclaim(), 1)
        again = self.tasks.poll(self.origin)
        self.assertEqual([t.job_id for t in again], [jid])

    def test_list_live(self):
        self.tasks.dispatch("a", {}, origin_node=self.origin)
        jid2 = self.tasks.dispatch("b", {}, origin_node=self.origin,
                                   target_node=self.origin)
        live = self.tasks.list_live()
        self.assertEqual({t.job_id for t in live},
                         {t.job_id for t in self.tasks.list_live(
                             node_id=self.origin)})
        self.assertEqual(len(live), 2)
        self.tasks.cancel(jid2)
        self.assertEqual(len(self.tasks.list_live()), 1)


class LocalTransportSweepTests(unittest.TestCase):
    def setUp(self):
        self.t = LocalTransport(_db())

    def test_ping(self):
        latency = self.t.ping()
        self.assertGreaterEqual(latency, 0.0)

    def test_deregister(self):
        n = self.t.register("n")
        self.t.deregister(n.node_id)
        self.assertEqual(self.t.active_nodes(), [])

    def test_long_poll_returns_when_task_arrives(self):
        # Threaded: needs a file-backed DB (see _file_db).
        self.t = LocalTransport(_file_db())
        node = self.t.register("worker").node_id
        origin = self.t.register("origin").node_id

        def _dispatch_later():
            time.sleep(0.2)
            self.t.dispatch("t", {}, origin_node=origin, target_node=node)

        threading.Thread(target=_dispatch_later, daemon=True).start()
        tasks = self.t.poll(node, wait=3.0)
        self.assertEqual(len(tasks), 1)

    def test_long_poll_timeout_returns_empty(self):
        node = self.t.register("worker").node_id
        start = time.monotonic()
        self.assertEqual(self.t.poll(node, wait=0.3), [])
        self.assertGreaterEqual(time.monotonic() - start, 0.25)

    def test_cancel_result_stats(self):
        origin = self.t.register("o").node_id
        jid = self.t.dispatch("t", {}, origin_node=origin)
        self.assertTrue(self.t.cancel(jid))
        jid2 = self.t.dispatch("t2", {}, origin_node=origin)
        tasks = self.t.poll(origin)
        self.t.complete(tasks[0].job_id, result={"done": 1})
        self.assertEqual(self.t.result(jid2), {"done": 1})
        stats = self.t.stats()
        self.assertIn("totals", stats)

    def test_context_manager(self):
        with LocalTransport(_db()) as t:
            n = t.register("n")
            self.assertTrue(n.node_id)


class FakePool:
    """Scripted stand-in for _KeepAlivePool."""

    def __init__(self, script):
        self.script = list(script)

    def request(self, method, url, body, headers, timeout):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


def _hub_transport(script, **kwargs):
    t = HttpTransport("http://hub:8861", token="secret", **kwargs)
    t._pool = FakePool(script)
    return t


class HttpResilienceTests(unittest.TestCase):
    def test_full_jitter_bounds(self):
        for attempt in range(5):
            d = _full_jitter(attempt, base=1.0, cap=8.0)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, min(8.0, 2.0 ** attempt))

    def test_retry_after_parsing(self):
        self.assertEqual(_retry_after_seconds({"Retry-After": "5"}), 5.0)
        self.assertIsNone(_retry_after_seconds({}))
        self.assertIsNone(_retry_after_seconds({"Retry-After": "soon"}))

    def test_retry_on_500_then_success(self):
        t = _hub_transport([
            (500, {}, b'{"detail":"boom"}'),
            (200, {}, b'{"ok": true, "nodes": []}'),
        ], backoff_base=0.001, backoff_cap=0.01)
        nodes = t.active_nodes()
        self.assertEqual(nodes, [])
        self.assertEqual(t.circuit_state, "closed")

    def test_retry_on_429_with_retry_after(self):
        t = _hub_transport([
            (429, {"Retry-After": "0"}, b'{"detail":"slow down"}'),
            (200, {}, b'{"ok": true}'),
        ], backoff_base=0.001, backoff_cap=0.01)
        t.heartbeat("n1")  # should not raise

    def test_no_retry_on_400(self):
        t = _hub_transport([(400, {}, b'{"detail":"bad"}')])
        with self.assertRaises(ValueError):
            t.dispatch("t", {}, origin_node="o")
        # Client errors must not trip the circuit.
        self.assertEqual(t.circuit_state, "closed")

    def test_401_maps_to_auth_error(self):
        t = _hub_transport([(401, {}, b'{"detail":"nope"}')])
        with self.assertRaises(AuthError) as ctx:
            t.active_nodes()
        self.assertFalse(ctx.exception.retryable)

    def test_404_node_unknown(self):
        t = _hub_transport(
            [(404, {}, b'{"code":"node_unknown","detail":"gone"}')])
        with self.assertRaises(NodeUnknown):
            t.heartbeat("ghost")

    def test_exhausted_retries_raise_hub_unreachable(self):
        import socket
        t = _hub_transport(
            [socket.timeout("timed out")] * 3,
            backoff_base=0.001, backoff_cap=0.01)
        with self.assertRaises(HubUnreachable) as ctx:
            t.active_nodes()
        self.assertTrue(ctx.exception.retryable)

    def test_circuit_opens_and_fails_fast(self):
        import socket
        t = _hub_transport(
            [socket.timeout("x")] * 20,
            retries=1, backoff_base=0.001, backoff_cap=0.01,
            circuit_failure_threshold=3, circuit_reset_timeout=60.0)
        for _ in range(3):
            with self.assertRaises(HubUnreachable):
                t.active_nodes()
        self.assertEqual(t.circuit_state, "open")
        # Fail fast: no pool interaction, no waiting.
        with self.assertRaises(CircuitOpen):
            t.active_nodes()
        self.assertTrue(t.circuit.describe().startswith("○"))

    def test_circuit_half_open_recovers(self):
        import socket
        t = _hub_transport(
            [socket.timeout("x")] * 2 + [(200, {}, b'{"ok": true}')],
            retries=1, backoff_base=0.001, backoff_cap=0.01,
            circuit_failure_threshold=2, circuit_reset_timeout=0.05)
        for _ in range(2):
            with self.assertRaises(HubUnreachable):
                t.active_nodes()
        self.assertEqual(t.circuit_state, "open")
        time.sleep(0.08)
        self.assertEqual(t.circuit_state, "half_open")
        t.ping()  # trial probe succeeds
        self.assertEqual(t.circuit_state, "closed")

    def test_circuit_reset_override(self):
        t = _hub_transport([], circuit_failure_threshold=1,
                           circuit_reset_timeout=60.0)
        t.circuit.after_failure()
        self.assertEqual(t.circuit_state, "open")
        t.reset_circuit()
        self.assertEqual(t.circuit_state, "closed")

    def test_poll_wait_long_poll(self):
        calls = {"n": 0}

        class _SeqPool(FakePool):
            def request(self, *a):
                calls["n"] += 1
                if calls["n"] < 3:
                    return (200, {}, b'{"tasks": []}')
                return (200, {}, b'{"tasks": [{"job_id": "j1", '
                                 b'"task_type": "t", "payload": {}, '
                                 b'"origin_node": "o"}]}')

        t = HttpTransport("http://hub:8861")
        t._pool = _SeqPool([])
        tasks = t.poll("n1", wait=5.0)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].job_id, "j1")
        self.assertIsInstance(tasks[0], MeshTask)

    def test_ping(self):
        t = _hub_transport([(200, {}, b'{"ok": true}')])
        self.assertGreaterEqual(t.ping(), 0.0)

    def test_heartbeat_sends_info(self):
        seen = {}

        class _CapPool(FakePool):
            def request(self, method, url, body, headers, timeout):
                seen["body"] = body
                return (200, {}, b'{"ok": true}')

        t = HttpTransport("http://hub:8861")
        t._pool = _CapPool([])
        t.heartbeat("n1", info={"load": 0.2})
        import json
        self.assertEqual(json.loads(seen["body"])["info"], {"load": 0.2})

    def test_unsupported_hub_ops_raise_clearly(self):
        t = _hub_transport([])
        for op in (lambda: t.deregister("n"), lambda: t.cancel("j"),
                   lambda: t.result("j"), lambda: t.stats()):
            with self.assertRaises(MeshError):
                op()

    def test_close(self):
        t = _hub_transport([])
        t.close()  # must not raise


class ErrorModelTests(unittest.TestCase):
    def test_codes_and_retryability(self):
        self.assertEqual(NodeUnknown("x").code, "node_unknown")
        self.assertFalse(NodeUnknown("x").retryable)
        self.assertTrue(HubUnreachable("x").retryable)
        self.assertTrue(CircuitOpen("x").retryable)
        self.assertFalse(AuthError("x").retryable)
        self.assertIsInstance(HubUnreachable("x"), TransportError)
        self.assertIsInstance(NodeSuspect("x"), MeshError)

    def test_wire_round_trip(self):
        err = CircuitOpen("overloaded", detail={"reset_in": 3.0})
        rebuilt = MeshError.from_dict(err.to_dict())
        self.assertIsInstance(rebuilt, CircuitOpen)
        self.assertEqual(rebuilt.code, "circuit_open")
        self.assertEqual(rebuilt.detail["reset_in"], 3.0)

    def test_str_shows_code(self):
        self.assertIn("[node_unknown]", str(NodeUnknown("gone")))

    def test_unknown_code_falls_back(self):
        rebuilt = MeshError.from_dict({"code": "nope", "message": "m"})
        self.assertIsInstance(rebuilt, MeshError)
        self.assertEqual(type(rebuilt), MeshError)


if __name__ == "__main__":
    unittest.main()
