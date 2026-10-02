"""L6 — os artifact graph: lineage, descendants, rebuild, created events.

Unit tier: fully offline. Uses an in-memory database and a temp blob dir;
the event bus subscription is synchronous and removed in tearDown.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from nomorals.core.events import Event, global_bus
from nomorals.storage.artifacts import ArtifactStore, Provenance
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database


def make_store(test=None):
    tmp = tempfile.mkdtemp()
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(":memory:")
    db.migrate()
    blobs = BlobStore(db, Path(tmp) / "blobs")
    return ArtifactStore(db, blobs), db


class ArtifactGraphTests(unittest.TestCase):
    def setUp(self):
        self.store, self.db = make_store(self)

    def tearDown(self):
        self.db.close()

    def test_lineage_over_three_level_chain(self):
        a = self.store.put_text("raw", type="dataset")
        b = self.store.derive(b"cleaned", from_ids=[a.id], type="dataset")
        c = self.store.derive(b"report", from_ids=[b.id], type="report")
        lineage = self.store.lineage(c.id)
        self.assertEqual([x.id for x in lineage], [b.id, a.id])
        # root has no ancestors; the artifact itself is never included
        self.assertEqual(self.store.lineage(a.id), [])
        self.assertNotIn(c.id, [x.id for x in lineage])

    def test_descendants_reverse_lookup(self):
        a = self.store.put_text("raw", type="dataset")
        b = self.store.derive(b"cleaned", from_ids=[a.id], type="dataset")
        c = self.store.derive(b"report", from_ids=[b.id], type="report")
        unrelated = self.store.put_text("other", type="note")
        # direct reverse lookup: only artifacts that name the id
        self.assertEqual([x.id for x in self.store.descendants(a.id)], [b.id])
        self.assertEqual([x.id for x in self.store.descendants(b.id)], [c.id])
        self.assertEqual([x.id for x in self.store.descendants(c.id)], [])
        self.assertEqual([x.id for x in self.store.descendants(unrelated.id)],
                         [])

    def test_lineage_is_cycle_safe(self):
        a = self.store.put_text("a")
        b = self.store.derive(b"b", from_ids=[a.id])
        # Forge a cycle: a now claims to derive from b, closing a<->b.
        prov = a.provenance.to_dict()
        prov["derived_from"] = [b.id]
        self.db.execute("UPDATE artifacts SET provenance = ? WHERE id = ?",
                        (json.dumps(prov), a.id))
        lineage = self.store.lineage(a.id)  # must terminate
        self.assertEqual([x.id for x in lineage], [b.id])

    def test_descendants_covers_supersedes_link(self):
        v1 = self.store.put_text("v1", type="report")
        v2 = self.store.put(
            b"v2", type="report",
            provenance=Provenance(supersedes=[v1.id]))
        self.assertEqual([x.id for x in self.store.superseded_by(v1.id)],
                         [v2.id])
        # superseded_by only follows the supersedes link, not derived_from
        d = self.store.derive(b"d", from_ids=[v1.id], type="report")
        self.assertEqual([x.id for x in self.store.superseded_by(v1.id)],
                         [v2.id])
        self.assertIn(d.id, {x.id for x in self.store.descendants(v1.id)})

    def test_rebuild_creates_proper_link(self):
        original = self.store.put_text("payload", type="report",
                                       creator="analyst", mission_id="m1",
                                       task_id="t1",
                                       metadata={"k": "v"})
        rebuilt = self.store.rebuild(original.id)
        self.assertNotEqual(rebuilt.id, original.id)
        self.assertEqual(rebuilt.provenance.derived_from, [original.id])
        self.assertEqual(self.store.read_text(rebuilt.id), "payload")
        self.assertEqual(rebuilt.type, original.type)
        self.assertEqual(rebuilt.creator, original.creator)
        self.assertEqual(rebuilt.mission_id, "m1")
        self.assertEqual(rebuilt.task_id, "t1")
        self.assertEqual(rebuilt.metadata, {"k": "v"})
        # lineage of the rebuild reaches the original
        self.assertEqual([x.id for x in self.store.lineage(rebuilt.id)],
                         [original.id])
        # creator override
        rebuilt2 = self.store.rebuild(original.id, creator="rebuilder")
        self.assertEqual(rebuilt2.creator, "rebuilder")

    def test_rebuild_unknown_id_raises(self):
        with self.assertRaises(KeyError):
            self.store.rebuild("nope")

    def test_artifact_created_event_on_put(self):
        seen: list[Event] = []
        sub = global_bus.subscribe("artifact.created", seen.append, sync=True)
        try:
            art = self.store.put_text("hello", type="report", creator="analyst",
                                      mission_id="m1", task_id="t1")
        finally:
            global_bus.unsubscribe(sub)
        self.assertEqual(len(seen), 1)
        data = seen[0].data
        self.assertEqual(data["artifact_id"], art.id)
        self.assertEqual(data["uri"], art.uri)
        self.assertEqual(data["type"], "report")
        self.assertEqual(data["creator"], "analyst")
        self.assertEqual(data["mission_id"], "m1")
        self.assertEqual(data["task_id"], "t1")

    def test_artifact_created_event_on_derive(self):
        parent = self.store.put_text("p")
        seen: list[Event] = []
        sub = global_bus.subscribe("artifact.created", seen.append, sync=True)
        try:
            child = self.store.derive(b"c", from_ids=[parent.id])
        finally:
            global_bus.unsubscribe(sub)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].data["artifact_id"], child.id)


if __name__ == "__main__":
    unittest.main()
