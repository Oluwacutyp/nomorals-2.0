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


if __name__ == "__main__":
    unittest.main()
