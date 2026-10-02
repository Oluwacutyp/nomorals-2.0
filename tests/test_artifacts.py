"""First-class artifacts: store, resolve, reference, provenance."""

import shutil
import tempfile
import unittest
from pathlib import Path

from nomorals.storage.artifacts import (
    Artifact,
    ArtifactStore,
    Provenance,
    ARTIFACT_URI_SCHEME,
)
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


class TestArtifacts(unittest.TestCase):
    def setUp(self):
        self.store, self.db = make_store(self)

    def tearDown(self):
        self.db.close()

    def test_put_and_read_roundtrip(self):
        art = self.store.put_text("hello world", type="report", creator="analyst",
                                  mission_id="m1", task_id="t1")
        self.assertTrue(art.id)
        self.assertEqual(art.type, "report")
        self.assertEqual(art.creator, "analyst")
        self.assertEqual(self.store.read_text(art.id), "hello world")
        self.assertEqual(len(art.content_hash), 64)

    def test_uri_and_resolve(self):
        art = self.store.put_json({"a": 1}, type="json")
        self.assertTrue(art.uri.startswith(ARTIFACT_URI_SCHEME))
        self.assertEqual(self.store.resolve(art.uri).id, art.id)
        self.assertEqual(self.store.resolve(art.id).id, art.id)
        self.assertIsNone(self.store.resolve("artifact://nope"))
        self.assertIsNone(self.store.resolve(""))

    def test_content_dedup_through_blob_store(self):
        a = self.store.put_text("same bytes")
        b = self.store.put_text("same bytes")
        self.assertEqual(a.content_hash, b.content_hash)
        self.assertNotEqual(a.id, b.id)  # distinct artifact records

    def test_find_references(self):
        art = self.store.put_text("x")
        text = f"see {art.uri} and also {art.uri} plus artifact://bogus"
        refs = ArtifactStore.find_references(text)
        self.assertEqual(refs, [art.id, "bogus"])

    def test_resolve_all_skips_unknown(self):
        art = self.store.put_text("x")
        resolved = self.store.resolve_all(f"a {art.uri} b artifact://nope")
        self.assertEqual(list(resolved), [art.uri])

    def test_derive_links_provenance(self):
        parent = self.store.put_text("source data", type="dataset")
        child = self.store.derive(b"analysis", from_ids=[parent.id],
                                  type="report", source_type="inference")
        self.assertIn(parent.id, child.provenance.derived_from)
        self.assertEqual(child.provenance.source_type, "inference")

    def test_provenance_roundtrip(self):
        prov = Provenance(source_type="tool", source_id="search",
                          confidence=0.8, verification_state="verified",
                          contradicts=["other-id"])
        art = self.store.put_text("x", provenance=prov)
        fetched = self.store.get(art.id)
        self.assertEqual(fetched.provenance.source_id, "search")
        self.assertEqual(fetched.provenance.confidence, 0.8)
        self.assertEqual(fetched.provenance.verification_state, "verified")
        self.assertEqual(fetched.provenance.contradicts, ["other-id"])

    def test_for_mission_and_task(self):
        a1 = self.store.put_text("one", mission_id="m9", task_id="t1")
        a2 = self.store.put_text("two", mission_id="m9", task_id="t2")
        self.store.put_text("three", mission_id="other")
        self.assertEqual({a.id for a in self.store.for_mission("m9")}, {a1.id, a2.id})
        self.assertEqual([a.id for a in self.store.for_task("t1")], [a1.id])

    def test_artifact_to_dict(self):
        art = self.store.put_text("x", type="text")
        d = art.to_dict()
        self.assertEqual(d["uri"], art.uri)
        self.assertIn("provenance", d)
        self.assertIn("content_hash", d)

    def test_migration_60_applied(self):
        self.assertTrue(self.db.table_exists("artifacts"))
        self.assertGreaterEqual(
            self.db.scalar("SELECT MAX(version) FROM schema_migrations"), 60)


if __name__ == "__main__":
    unittest.main()
