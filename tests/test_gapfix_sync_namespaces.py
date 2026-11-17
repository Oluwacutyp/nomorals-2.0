"""Gapfix R2-39: `nm sync ns` verbs on top of NamespaceSet.

Failing-first: every test here requires the namespace persistence
seam (`nomorals/sync/ns_store.py`), the `Namespace.resolve` /
`open_conflicts` API, and the `ns` branch of `_cmd_sync` — none of
which exist until the lane lands.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from nomorals.storage.db import Database
from nomorals.sync import LocalPeer, SyncEngine, SyncStore
from nomorals.sync.namespaces import Namespace, NamespaceSet
from nomorals.sync.ns_store import (
    load_set,
    merge_peer_namespaces,
    ns_key,
    save_namespace,
)


def _tmpdir() -> tempfile.TemporaryDirectory:
    # Never write test scratch to the box /tmp tmpfs when the harness
    # gives us a disk-backed dir.
    return tempfile.TemporaryDirectory(dir=os.environ.get("NM_TEST_TMPDIR"))


def _ctx(db: Database, device_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        db=db, device_id=device_id,
        settings=SimpleNamespace(workspace_dir=""))


def _run(ctx: SimpleNamespace, *words: str, **kw) -> tuple[int, str, str]:
    from nomorals.cmdline.commands.sync import _cmd_sync
    args = SimpleNamespace(task=list(words), json=kw.get("json", False),
                           peer_db=kw.get("peer_db", ""))
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = _cmd_sync(args, ctx)
    return rc, out.getvalue(), err.getvalue()


class NsCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = _tmpdir()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ctx = _ctx(Database(str(self.root / "dev.db")), "phone")

    def _store(self) -> SyncStore:
        return SyncStore(self.ctx.db, device_id="phone")

    def test_ns_create_survives_and_lists(self):
        rc, out, _ = _run(self.ctx, "ns", "create", "prefs", "--type", "map")
        self.assertEqual(rc, 0, out)
        self.assertIn("prefs", out)
        rc, out, _ = _run(self.ctx, "ns", "list")
        self.assertEqual(rc, 0, out)
        self.assertIn("prefs", out)
        self.assertIn("map", out)
        # Survives a fresh load from the same store: it is a real
        # record, not an in-memory object.
        names = load_set(self._store(), "phone").names()
        self.assertIn("prefs", names)
        self.assertIn(ns_key("prefs"), self._store().keys())

    def test_ns_put_stores_typed_record(self):
        _run(self.ctx, "ns", "create", "prefs")
        rc, out, _ = _run(self.ctx, "ns", "put", "prefs", "theme", '"dark"')
        self.assertEqual(rc, 0, out)
        rc, out, _ = _run(self.ctx, "ns", "get", "prefs")
        self.assertEqual(rc, 0, out)
        self.assertEqual(json.loads(out), {"theme": "dark"})
        rec = self._store().get(ns_key("prefs"))
        self.assertIsNotNone(rec)
        self.assertEqual(rec.value["kind"], "map")

    def test_ns_create_wrong_type_refuses(self):
        _run(self.ctx, "ns", "create", "prefs", "--type", "map")
        rc, _, err = _run(self.ctx, "ns", "create", "prefs",
                          "--type", "counter")
        self.assertEqual(rc, 2)
        self.assertIn("already exists as", err)

    def test_ns_put_wrong_kind_refuses(self):
        _run(self.ctx, "ns", "create", "todo", "--type", "list")
        rc, _, err = _run(self.ctx, "ns", "put", "todo", "k", '"v"')
        self.assertNotEqual(rc, 0)
        self.assertIn("map", err)

    def test_ns_unknown_refuses(self):
        rc, _, err = _run(self.ctx, "ns", "get", "nope")
        self.assertEqual(rc, 1)
        self.assertIn("no such namespace", err)

    def test_ns_list_and_counter_and_append(self):
        _run(self.ctx, "ns", "create", "todo", "--type", "list")
        rc, out, _ = _run(self.ctx, "ns", "append", "todo", '"buy milk"',
                          "--id", "i1")
        self.assertEqual(rc, 0, out)
        self.assertIn("i1", out)
        _run(self.ctx, "ns", "create", "clicks", "--type", "counter")
        rc, out, _ = _run(self.ctx, "ns", "incr", "clicks", "3")
        self.assertEqual(rc, 0, out)
        self.assertIn("3", out)
        rc, out, _ = _run(self.ctx, "ns", "get", "clicks")
        self.assertEqual(json.loads(out), 3)


class NsMergeTests(unittest.TestCase):
    """Two-device merges at the helper level (deterministic timestamps)
    and through the CLI (real peer-db transport)."""

    def setUp(self):
        self._tmp = _tmpdir()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _device(self, name: str, device_id: str):
        db = Database(str(self.root / f"{name}.db"))
        store = SyncStore(db, device_id=device_id)
        return db, store

    def test_different_fields_both_survive(self):
        _, store_a = self._device("a", "phone")
        _, store_b = self._device("b", "laptop")
        set_a = NamespaceSet(device_id="phone")
        set_b = NamespaceSet(device_id="laptop")
        set_a.get_or_create("prefs", "map").put(
            "theme", "dark", ts=100.0, device="phone")
        set_b.get_or_create("prefs", "map").put(
            "lang", "en", ts=100.0, device="laptop")
        save_namespace(store_a, set_a.get_or_create("prefs", "map"))
        save_namespace(store_b, set_b.get_or_create("prefs", "map"))
        new = merge_peer_namespaces(
            load_set(store_a, "phone"), load_set(store_b, "laptop"), store_a)
        self.assertEqual(new, [])
        merged = load_set(store_a, "phone").get("prefs")
        self.assertEqual(merged.value(), {"theme": "dark", "lang": "en"})

    def test_same_field_same_ts_fork_is_visible(self):
        _, store_a = self._device("a", "phone")
        _, store_b = self._device("b", "laptop")
        set_a = NamespaceSet(device_id="phone")
        set_b = NamespaceSet(device_id="laptop")
        set_a.get_or_create("prefs", "map").put(
            "theme", "dark", ts=100.0, device="phone")
        set_b.get_or_create("prefs", "map").put(
            "theme", "light", ts=100.0, device="laptop")
        save_namespace(store_a, set_a.get_or_create("prefs", "map"))
        save_namespace(store_b, set_b.get_or_create("prefs", "map"))
        new = merge_peer_namespaces(
            load_set(store_a, "phone"), load_set(store_b, "laptop"), store_a)
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0].values, ["dark", "light"])
        # Tiebreak is (ts, device): "phone" > "laptop", so "dark" wins.
        self.assertEqual(new[0].resolved_to, "dark")
        merged = load_set(store_a, "phone").get("prefs")
        self.assertEqual(merged.value()["theme"], "dark")
        self.assertEqual(len(merged.conflicts()), 1)
        self.assertEqual(len(merged.open_conflicts()), 1)

    def test_list_appends_never_lost_and_counter_sums(self):
        _, store_a = self._device("a", "phone")
        _, store_b = self._device("b", "laptop")
        set_a = NamespaceSet(device_id="phone")
        set_b = NamespaceSet(device_id="laptop")
        set_a.get_or_create("todo", "list").append(
            "buy milk", item_id="i1", ts=1.0, device="phone")
        set_b.get_or_create("todo", "list").append(
            "buy eggs", item_id="i2", ts=1.0, device="laptop")
        set_a.get_or_create("clicks", "counter").increment(3, device="phone")
        set_b.get_or_create("clicks", "counter").increment(4, device="laptop")
        for name in ("todo", "clicks"):
            save_namespace(store_a, set_a.get_or_create(name, "list" if name == "todo" else "counter"))
            save_namespace(store_b, set_b.get_or_create(name, "list" if name == "todo" else "counter"))
        merge_peer_namespaces(
            load_set(store_a, "phone"), load_set(store_b, "laptop"), store_a)
        merged = load_set(store_a, "phone")
        self.assertEqual(sorted(merged.get("todo").value()),
                         ["buy eggs", "buy milk"])
        self.assertEqual(merged.get("clicks").value(), 7)

    def test_cli_merge_over_peer_db(self):
        db_a, _ = self._device("a", "phone")
        db_b, store_b = self._device("b", "laptop")
        ctx_a = _ctx(db_a, "phone")
        set_b = NamespaceSet(device_id="laptop")
        set_b.get_or_create("prefs", "map").put(
            "lang", "en", ts=100.0, device="laptop")
        save_namespace(store_b, set_b.get_or_create("prefs", "map"))
        _run(ctx_a, "ns", "create", "prefs")
        _run(ctx_a, "ns", "put", "prefs", "theme", '"dark"')
        rc, out, _ = _run(ctx_a, "ns", "merge",
                          peer_db=str(self.root / "b.db"))
        self.assertEqual(rc, 0, out)
        self.assertIn("prefs", out)
        rc, out, _ = _run(ctx_a, "ns", "get", "prefs")
        self.assertEqual(json.loads(out), {"theme": "dark", "lang": "en"})

    def test_cli_conflicts_and_resolve(self):
        db_a, store_a = self._device("a", "phone")
        _, store_b = self._device("b", "laptop")
        ctx_a = _ctx(db_a, "phone")
        set_a = NamespaceSet(device_id="phone")
        set_b = NamespaceSet(device_id="laptop")
        set_a.get_or_create("prefs", "map").put(
            "theme", "dark", ts=100.0, device="phone")
        set_b.get_or_create("prefs", "map").put(
            "theme", "light", ts=100.0, device="laptop")
        save_namespace(store_a, set_a.get_or_create("prefs", "map"))
        save_namespace(store_b, set_b.get_or_create("prefs", "map"))
        rc, out, _ = _run(ctx_a, "ns", "merge",
                          peer_db=str(self.root / "b.db"))
        self.assertEqual(rc, 0, out)
        self.assertIn("conflict", out.lower())
        rc, out, _ = _run(ctx_a, "ns", "conflicts")
        self.assertEqual(rc, 0, out)
        self.assertIn("prefs", out)
        self.assertIn("theme", out)
        self.assertIn("dark", out)
        self.assertIn("light", out)
        # The owner picks the loser's value; the resolution is recorded
        # and the conflict clears from the open list.
        rc, out, _ = _run(ctx_a, "ns", "resolve", "prefs", "theme", '"light"')
        self.assertEqual(rc, 0, out)
        self.assertIn("light", out)
        rc, out, _ = _run(ctx_a, "ns", "get", "prefs")
        self.assertEqual(json.loads(out)["theme"], "light")
        rc, out, _ = _run(ctx_a, "ns", "conflicts")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("theme", out)
        # Re-merging the same peer state records no new conflict: the
        # resolution's fresh timestamp wins deterministically.
        rc, out, _ = _run(ctx_a, "ns", "merge",
                          peer_db=str(self.root / "b.db"))
        self.assertEqual(rc, 0, out)
        self.assertNotIn("new conflict", out.lower())
        merged = load_set(store_a, "phone").get("prefs")
        self.assertEqual(len(merged.open_conflicts()), 0)
        self.assertEqual(len(merged.resolutions()), 1)

    def test_merged_state_rides_plain_push_pull(self):
        db_a, store_a = self._device("a", "phone")
        db_b, store_b = self._device("b", "laptop")
        ctx_a = _ctx(db_a, "phone")
        ctx_b = _ctx(db_b, "laptop")
        set_b = NamespaceSet(device_id="laptop")
        set_b.get_or_create("prefs", "map").put(
            "lang", "en", ts=100.0, device="laptop")
        save_namespace(store_b, set_b.get_or_create("prefs", "map"))
        _run(ctx_a, "ns", "create", "prefs")
        _run(ctx_a, "ns", "put", "prefs", "theme", '"dark"')
        _run(ctx_a, "ns", "merge", peer_db=str(self.root / "b.db"))
        # The merged state is a plain sync record: push replicates it.
        rc, out, _ = _run(ctx_a, "push", peer_db=str(self.root / "b.db"))
        self.assertEqual(rc, 0, out)
        self.assertIn("pushed 1", out)
        rc, out, _ = _run(ctx_b, "ns", "get", "prefs")
        self.assertEqual(rc, 0, out)
        self.assertEqual(json.loads(out), {"theme": "dark", "lang": "en"})


class NsAutoSyncTests(unittest.TestCase):
    def test_autosync_replicates_namespace_record(self):
        from nomorals.sync import AutoSync
        tmp = _tmpdir()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        db_a = Database(str(root / "a.db"))
        db_b = Database(str(root / "b.db"))
        store_a = SyncStore(db_a, device_id="phone")
        store_b = SyncStore(db_b, device_id="laptop")
        engine_a = SyncEngine(db_a, store_a)
        auto = AutoSync(engine_a, LocalPeer(store_b), peer_id="laptop",
                        interval_s=60.0, debounce_s=0.0)
        auto.start()
        try:
            ns = Namespace("prefs", "map", device_id="phone")
            ns.put("theme", "dark")
            save_namespace(store_a, ns)
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if store_b.get(ns_key("prefs")) is not None:
                    break
                time.sleep(0.1)
            rec = store_b.get(ns_key("prefs"))
            self.assertIsNotNone(rec, "AutoSync never replicated the ns record")
            self.assertEqual(rec.value["map"]["theme"]["value"], "dark")
            self.assertGreaterEqual(auto.stats()["runs"], 1)
        finally:
            auto.stop()


if __name__ == "__main__":
    unittest.main()
