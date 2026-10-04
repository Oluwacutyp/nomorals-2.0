"""Tests for the device mesh: nodes, heartbeats, task dispatch."""
from __future__ import annotations

import time
import unittest

from nomorals.mesh import (
    LocalTransport,
    MeshNode,
    MeshTasks,
    NodeRegistry,
    NodeUnknown,
)
from nomorals.storage.db import Database


def _db() -> Database:
    from nomorals.storage.migrations import MIGRATIONS
    from nomorals.storage.schema import MigrationRunner
    db = Database(":memory:")
    MigrationRunner(db).apply_all(MIGRATIONS)
    return db


class NodeRegistryTests(unittest.TestCase):
    def setUp(self):
        self.reg = NodeRegistry(_db())

    def test_register_returns_node(self):
        node = self.reg.register("phone", platform="termux",
                                 capabilities=["camera"])
        self.assertIsInstance(node, MeshNode)
        self.assertEqual(node.name, "phone")
        self.assertEqual(node.platform, "termux")
        self.assertIn("camera", node.capabilities)
        self.assertTrue(node.node_id)

    def test_register_is_idempotent(self):
        a = self.reg.register("phone", node_id="n1")
        b = self.reg.register("phone-v2", node_id="n1")
        self.assertEqual(a.node_id, b.node_id)
        self.assertEqual(b.name, "phone-v2")

    def test_heartbeat_updates_last_seen(self):
        node = self.reg.register("phone")
        old = node.last_seen
        time.sleep(0.01)
        self.reg.heartbeat(node.node_id)
        self.assertGreater(self.reg.get(node.node_id).last_seen, old)

    def test_heartbeat_unknown_raises(self):
        with self.assertRaises(NodeUnknown):
            self.reg.heartbeat("nope")

    def test_list_active_filters_stale(self):
        fresh = self.reg.register("fresh")
        stale = self.reg.register("stale")
        # Age the stale node manually.
        self.reg.db.execute(
            "UPDATE mesh_nodes SET last_seen=? WHERE node_id=?",
            (time.time() - 1000, stale.node_id),
        )
        active = self.reg.list_active(stale_after=120)
        ids = [n.node_id for n in active]
        self.assertIn(fresh.node_id, ids)
        self.assertNotIn(stale.node_id, ids)

    def test_prune_removes_ancient(self):
        node = self.reg.register("old")
        self.reg.db.execute(
            "UPDATE mesh_nodes SET last_seen=? WHERE node_id=?",
            (time.time() - 100000, node.node_id),
        )
        self.assertEqual(self.reg.prune(stale_after=1000), 1)
        self.assertIsNone(self.reg.get(node.node_id))


class MeshTasksTests(unittest.TestCase):
    def setUp(self):
        self.tasks = MeshTasks(_db())

    def test_dispatch_and_poll_targeted(self):
        jid = self.tasks.dispatch("backup", {"x": 1}, origin_node="cloud",
                                  target_node="phone")
        self.assertTrue(jid)
        # Other node gets nothing.
        self.assertEqual(self.tasks.poll("laptop"), [])
        # Target node gets it.
        got = self.tasks.poll("phone")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].task_type, "backup")
        self.assertEqual(got[0].payload["x"], 1)
        self.assertEqual(got[0].target_node, "phone")
        self.assertEqual(got[0].origin_node, "cloud")

    def test_dispatch_broadcast(self):
        self.tasks.dispatch("ping", {}, origin_node="cloud")
        self.assertEqual(len(self.tasks.poll("phone")), 1)
        # Broadcast is claimed once (leased), second node gets nothing.
        self.assertEqual(self.tasks.poll("laptop"), [])

    def test_complete_and_fail(self):
        jid = self.tasks.dispatch("work", {}, origin_node="a",
                                  target_node="b")
        tasks = self.tasks.poll("b")
        self.tasks.complete(tasks[0].job_id, result={"ok": True})
        self.assertEqual(self.tasks.pending_count("b"), 0)

    def test_dispatch_requires_fields(self):
        with self.assertRaises(ValueError):
            self.tasks.dispatch("", {}, origin_node="a")
        with self.assertRaises(ValueError):
            self.tasks.dispatch("t", {}, origin_node="")
        with self.assertRaises(ValueError):
            self.tasks.poll("")

    def test_dispatch_rejects_oversized_payload(self):
        big = {"blob": "x" * (2 * 1024 * 1024)}
        with self.assertRaises(ValueError) as ctx:
            self.tasks.dispatch("t", big, origin_node="a")
        self.assertIn("artifact", str(ctx.exception))

    def test_pending_count(self):
        self.tasks.dispatch("a", {}, origin_node="x", target_node="phone")
        self.tasks.dispatch("b", {}, origin_node="x")  # broadcast
        self.assertEqual(self.tasks.pending_count("phone"), 2)
        self.assertEqual(self.tasks.pending_count("laptop"), 1)
        self.assertEqual(self.tasks.pending_count(), 2)


class LocalTransportTests(unittest.TestCase):
    def setUp(self):
        self.t = LocalTransport(_db())

    def test_full_cycle(self):
        phone = self.t.register("phone", platform="termux")
        cloud = self.t.register("cloud", platform="linux")
        self.t.heartbeat(phone.node_id)
        active = self.t.active_nodes()
        self.assertEqual(len(active), 2)

        jid = self.t.dispatch("sync", {"k": "v"}, origin_node=cloud.node_id,
                              target_node=phone.node_id)
        tasks = self.t.poll(phone.node_id)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].job_id, jid)
        self.t.complete(jid)
        self.assertEqual(self.t.poll(phone.node_id), [])

    def test_fail_no_retry(self):
        t = self.t
        jid = t.dispatch("work", {}, origin_node="a", target_node="b")
        polled = t.poll("b")
        self.assertEqual(len(polled), 1)
        t.fail(jid, error="boom", retry=False)
        # No retry: the job never becomes pollable again.
        self.assertEqual(t.poll("b"), [])
        self.assertEqual(t.tasks.pending_count("b"), 0)


class MeshFreshDbTests(unittest.TestCase):
    """R22: MeshTasks on a database that never ran migrations.

    WorkQueue doesn't create its own table; dispatch used to crash with
    'no such table: work_queue' on a fresh DB (e.g. the mesh CLI path).
    """

    def test_dispatch_poll_complete_on_fresh_db(self):
        db = Database(":memory:")  # no migrations applied
        tasks = MeshTasks(db)
        jid = tasks.dispatch("backup", {"x": 1}, origin_node="cloud",
                             target_node="phone")
        got = tasks.poll("phone")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].job_id, jid)
        tasks.complete(jid, result={"ok": True})
        self.assertEqual(tasks.pending_count("phone"), 0)


class MeshCLITests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # Fresh DB, no migrations — exercises the R22 schema fix too.
        db = Database(str(Path(self._tmp.name) / "t.db"))
        self.ctx = SimpleNamespace(
            db=db, device_id="test",
            settings=SimpleNamespace(workspace_dir=self._tmp.name))

    def _run(self, *words, **kw):
        import io
        from contextlib import redirect_stdout, redirect_stderr
        from types import SimpleNamespace
        from nomorals.cmdline.commands.mesh import _cmd_mesh
        args = SimpleNamespace(task=list(words), json=kw.get("json", False))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = _cmd_mesh(args, self.ctx)
        return rc, out.getvalue(), err.getvalue()

    def _node_id(self):
        rc, out, _ = self._run("register", "worker", "--platform", "linux")
        self.assertEqual(rc, 0)
        return out.strip()

    def test_dispatch_poll_complete_cycle(self):
        nid = self._node_id()
        rc, out, _ = self._run("dispatch", "summarize",
                               "--json-args", '{"n": 3}')
        self.assertEqual(rc, 0, out)
        jid = out.strip()
        rc, out, _ = self._run("poll", nid)
        self.assertEqual(rc, 0)
        self.assertIn(jid, out)
        self.assertIn("summarize", out)
        rc, out, _ = self._run("pending")
        self.assertIn("0 pending", out)
        rc, _, _ = self._run("complete", jid, "--result", '{"ok": true}')
        self.assertEqual(rc, 0)

    def test_dispatch_rejects_bad_json_cleanly(self):
        rc, _, err = self._run("dispatch", "t", "--json-args", "{bad")
        self.assertEqual(rc, 2)
        self.assertIn("invalid JSON", err)
        self.assertNotIn("Traceback", err)

    def test_fail_no_retry(self):
        nid = self._node_id()
        rc, out, _ = self._run("dispatch", "work", "--target", nid)
        jid = out.strip()
        self._run("poll", nid)
        rc, _, _ = self._run("fail", jid, "--error", "boom", "--no-retry")
        self.assertEqual(rc, 0)
        rc, out, _ = self._run("pending", "--node", nid)
        self.assertIn("0 pending", out)

    def test_prune(self):
        nid = self._node_id()
        self.ctx.db.execute(
            "UPDATE mesh_nodes SET last_seen=? WHERE node_id=?",
            (time.time() - 100000, nid))
        rc, out, _ = self._run("prune", "--stale-after", "60")
        self.assertEqual(rc, 0)
        self.assertIn("pruned 1", out)


if __name__ == "__main__":
    unittest.main()
