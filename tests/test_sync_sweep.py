"""Sweep tests for nomorals.sync: HLC, field-level merge, GC, engine
upgrades (chunked checkpoints, retry, directions, preview, history,
AutoSync), HTTP peer, and presentation."""
from __future__ import annotations

import http.server
import json
import threading
import time
import unittest
import urllib.parse
from unittest import mock

from nomorals.storage.db import Database
from nomorals.sync import (
    AutoSync, HttpSyncPeer, LocalPeer, SyncEngine, SyncPeer, SyncRecord,
    SyncStore,
)
from nomorals.sync.errors import (
    SyncAuthError, SyncConnectionError, SyncError, SyncHubError,
)


def _db() -> Database:
    return Database(":memory:")


def _pair():
    db_a, db_b = _db(), _db()
    sa, sb = SyncStore(db_a, "a"), SyncStore(db_b, "b")
    return SyncEngine(db_a, sa), sa, sb


class HlcTests(unittest.TestCase):
    def test_tick_monotonic_under_clock_skew(self):
        s = SyncStore(_db(), "d1")
        vals = iter([1000.0, 900.0, 800.0, 700.0])
        with mock.patch("time.time", side_effect=lambda: next(vals, 500.0)):
            r1 = s.put("a", {"v": 1})
            r2 = s.put("b", {"v": 2})
        # Wall clock went backwards; the HLC never does.
        self.assertGreater((r2.hlc_ts, r2.hlc_count),
                           (r1.hlc_ts, r1.hlc_count))

    def test_receive_pulls_lagging_clock_forward(self):
        s = SyncStore(_db(), "phone")
        future = time.time() + 3600.0
        remote = SyncRecord("k", {"v": "hub"}, updated_at=future,
                            device_id="hub", hlc_ts=future, hlc_count=0)
        s.apply(remote)
        rec = s.put("mine", {"v": 1})
        self.assertGreaterEqual(rec.hlc_ts, future)

    def test_skewed_device_edit_wins_after_receiving(self):
        # The phone-behind scenario: phone receives hub's write, its clock
        # jumps forward, so its next edit is causally later and wins.
        hub, phone = SyncStore(_db(), "hub"), SyncStore(_db(), "phone")
        hub.put("k", {"v": "hub"})
        phone.apply(hub.get_record("k"))
        phone.put("k", {"v": "phone"})
        hub.apply(phone.get_record("k"))
        self.assertEqual(hub.get("k").value["v"], "phone")

    def test_wins_prefers_hlc_over_wall_clock(self):
        a = SyncRecord("k", {"v": "a"}, updated_at=200.0, device_id="a",
                       hlc_ts=100.0, hlc_count=0)
        b = SyncRecord("k", {"v": "b"}, updated_at=100.0, device_id="b",
                       hlc_ts=300.0, hlc_count=0)
        # Later wall clock, older HLC -> loses.
        self.assertIs(SyncRecord.wins(a, b), b)
        self.assertIs(SyncRecord.wins(b, a), b)

    def test_wins_legacy_fallback_when_either_side_lacks_hlc(self):
        # Mixed HLC/legacy pair: old (updated_at, device_id) rule.
        modern = SyncRecord("k", {"v": "m"}, updated_at=100.0,
                            device_id="a", hlc_ts=9999.0, hlc_count=0)
        legacy = SyncRecord("k", {"v": "l"}, updated_at=200.0,
                            device_id="b")
        self.assertIs(SyncRecord.wins(modern, legacy), legacy)
        self.assertIs(SyncRecord.wins(legacy, modern), legacy)

    def test_legacy_record_adopts_via_old_lww_path(self):
        s = SyncStore(_db(), "d1")
        s.put("k", {"v": 1}, updated_at=100.0)
        remote = SyncRecord("k", {"v": 2}, updated_at=200.0,
                            device_id="d2")
        self.assertTrue(s.apply(remote))
        self.assertEqual(s.get("k").value["v"], 2)
        # ...and an older legacy write still loses.
        older = SyncRecord("k", {"v": 3}, updated_at=50.0, device_id="d2")
        self.assertFalse(s.apply(older))
        self.assertEqual(s.get("k").value["v"], 2)

    def test_hlc_backfill_migration(self):
        db = _db()
        db.execute(
            """CREATE TABLE sync_records (
                key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL DEFAULT 0,
                device_id TEXT NOT NULL DEFAULT '',
                deleted INTEGER NOT NULL DEFAULT 0)""")
        db.execute("INSERT INTO sync_records (key, value, updated_at,"
                   " device_id) VALUES ('a', '{\"x\": 1}', 123.0, 'x')")
        s = SyncStore(db, "d1")
        rec = s.get_record("a")
        self.assertEqual(rec.hlc_ts, 123.0)
        self.assertIn("x", rec.clocks)


class FieldMergeTests(unittest.TestCase):
    def _remote(self, key, value, clock_ts, device="r", clocks=None):
        return SyncRecord(
            key, value, updated_at=clock_ts, device_id=device,
            hlc_ts=clock_ts, hlc_count=0,
            clocks=clocks or {
                f: {"ts": clock_ts, "c": 0, "d": device, "tomb": False}
                for f in value
            })

    def test_concurrent_different_fields_both_win(self):
        s = SyncStore(_db(), "d1")
        s.put("k", {"a": 1, "b": 1})
        base = s.get_record("k")
        clk = base.clocks["b"]
        remote = self._remote("k", {"a": 1, "b": 2},
                              clk["ts"] + 10, clocks={
                                  "b": {"ts": clk["ts"] + 10, "c": 0,
                                        "d": "r", "tomb": False}})
        self.assertTrue(s.apply(remote))
        self.assertEqual(s.get("k").value, {"a": 1, "b": 2})

    def test_same_field_newer_hlc_wins(self):
        s = SyncStore(_db(), "d1")
        s.put("k", {"a": 1})
        base = s.get_record("k")
        clk = base.clocks["a"]
        remote = self._remote("k", {"a": 2}, clk["ts"] + 10, clocks={
            "a": {"ts": clk["ts"] + 10, "c": 0, "d": "r", "tomb": False}})
        s.apply(remote)
        self.assertEqual(s.get("k").value["a"], 2)
        # Stale field write loses.
        stale = self._remote("k", {"a": 3}, clk["ts"] - 10, clocks={
            "a": {"ts": clk["ts"] - 10, "c": 0, "d": "r", "tomb": False}})
        self.assertFalse(s.apply(stale))
        self.assertEqual(s.get("k").value["a"], 2)

    def test_field_delete_tombstone_replicates(self):
        s = SyncStore(_db(), "d1")
        s.put("k", {"a": 1, "b": 1})
        s.put("k", {"a": 1})  # deletes field b
        self.assertNotIn("b", s.get("k").value)
        tomb = s.get_record("k").clocks["b"]
        self.assertTrue(tomb["tomb"])
        # A stale remote that still has b cannot resurrect it.
        stale = self._remote("k", {"a": 1, "b": 9}, tomb["ts"] - 10,
                             clocks={
                                 "a": {"ts": tomb["ts"] - 10, "c": 0,
                                       "d": "r", "tomb": False},
                                 "b": {"ts": tomb["ts"] - 10, "c": 0,
                                       "d": "r", "tomb": False}})
        self.assertFalse(s.apply(stale))
        self.assertNotIn("b", s.get("k").value)
        # A newer remote revives b.
        fresh = self._remote("k", {"a": 1, "b": 9}, tomb["ts"] + 10,
                             clocks={
                                 "a": {"ts": tomb["ts"] + 10, "c": 0,
                                       "d": "r", "tomb": False},
                                 "b": {"ts": tomb["ts"] + 10, "c": 0,
                                       "d": "r", "tomb": False}})
        self.assertTrue(s.apply(fresh))
        self.assertEqual(s.get("k").value["b"], 9)

    def test_delete_vs_edit(self):
        s = SyncStore(_db(), "d1")
        s.put("k", {"a": 1})
        s.delete("k")
        self.assertIsNone(s.get("k"))
        gone = s.get_record("k")
        # Stale edit loses to the tombstone.
        stale = self._remote("k", {"a": 2}, gone.hlc_ts - 10)
        self.assertFalse(s.apply(stale))
        self.assertIsNone(s.get("k"))
        # Newer edit resurrects with merged fields.
        fresh = self._remote("k", {"a": 2, "c": 3}, gone.hlc_ts + 10)
        self.assertTrue(s.apply(fresh))
        self.assertEqual(s.get("k").value, {"a": 2, "c": 3})

    def test_put_many(self):
        s = SyncStore(_db(), "d1")
        recs = s.put_many({"a": {"v": 1}, "b": {"v": 2}})
        self.assertEqual(len(recs), 2)
        self.assertEqual(s.count(), 2)


class GcTests(unittest.TestCase):
    def _aged_store(self):
        s = SyncStore(_db(), "d1")
        s.put("live", {"v": 1})
        s.put("old", {"v": 1})
        s.delete("old")
        s.put("young", {"v": 1})
        s.delete("young")
        now = time.time()
        # Age the "old" tombstone 100 days; "young" stays fresh.
        s.db.execute("UPDATE sync_records SET updated_at=? WHERE key='old'",
                     (now - 100 * 86400,))
        return s, now

    def test_gc_dry_run_by_default(self):
        s, now = self._aged_store()
        rep = s.gc_tombstones(older_than_s=90 * 86400, now=now)
        self.assertTrue(rep["dry_run"])
        self.assertEqual(rep["eligible"], 1)
        self.assertEqual(rep["deleted"], 0)
        self.assertEqual(s.tombstone_count(), 2)

    def test_gc_apply_reaps_only_old(self):
        s, now = self._aged_store()
        rep = s.gc_tombstones(older_than_s=90 * 86400, apply=True, now=now)
        self.assertEqual(rep["deleted"], 1)
        self.assertIsNone(s.get_record("old"))
        self.assertIsNotNone(s.get_record("young"))  # young tombstone kept
        self.assertIsNotNone(s.get("live"))
        self.assertEqual(s.tombstone_count(), 1)


class DigestTests(unittest.TestCase):
    def test_digest_converges_after_sync(self):
        eng_a, sa, sb = _pair()
        sa.put("x", {"v": 1})
        sb.put("y", {"v": 2})
        self.assertNotEqual(sa.digest(), sb.digest())
        eng_a.sync(LocalPeer(sb))
        eng_b = SyncEngine(sb.db, sb)
        eng_b.sync(LocalPeer(sa))
        self.assertEqual(sa.digest(), sb.digest())
        self.assertEqual(sa.diff_keys(sb),
                         {"only_in_self": [], "only_in_other": [],
                          "different": []})

    def test_diff_keys_finds_divergence(self):
        _, sa, sb = _pair()
        sa.put("x", {"v": 1})
        sb.put("y", {"v": 2})
        sa.put("z", {"v": 1})
        sb.put("z", {"v": 999})
        d = sa.diff_keys(sb)
        self.assertEqual(d["only_in_self"], ["x"])
        self.assertEqual(d["only_in_other"], ["y"])
        self.assertEqual(d["different"], ["z"])


class SubscribeTests(unittest.TestCase):
    def test_subscribe_fires_on_writes(self):
        s = SyncStore(_db(), "d1")
        seen = []
        s.subscribe(seen.append)
        s.put("a", {"v": 1})
        s.delete("a")
        s.apply(SyncRecord("b", {"v": 2}, updated_at=1.0, device_id="x"))
        self.assertEqual([(r.key, r.deleted) for r in seen],
                         [("a", False), ("a", True), ("b", False)])
        s.unsubscribe(seen.append)
        n = len(seen)
        s.put("c", {"v": 3})
        self.assertEqual(len(seen), n)  # unsubscribed: nothing fires


class FlakyPeer(SyncPeer):
    """Test peer with scripted failures."""

    def __init__(self, store, fail_push_calls=(), fail_pull_calls=(),
                 exc=SyncConnectionError("boom")):
        self.store = store
        self.fail_push_calls = set(fail_push_calls)
        self.fail_pull_calls = set(fail_pull_calls)
        self.exc = exc
        self.push_calls = 0
        self.pull_calls = 0

    def push_records(self, records):
        self.push_calls += 1
        if self.push_calls in self.fail_push_calls:
            raise self.exc
        n = 0
        for r in records:
            if self.store.apply(r):
                n += 1
        return n

    def fetch_since(self, since):
        self.pull_calls += 1
        if self.pull_calls in self.fail_pull_calls:
            raise self.exc
        return self.store.list_changed_since(since)

    def fetch_since_seq(self, seq):
        self.pull_calls += 1
        if self.pull_calls in self.fail_pull_calls:
            raise self.exc
        return self.store.list_since_seq(seq)


class EngineDirectionTests(unittest.TestCase):
    def test_push_only(self):
        eng_a, sa, sb = _pair()
        sa.put("a", {"v": 1})
        sb.put("b", {"v": 2})
        r = eng_a.sync(LocalPeer(sb), direction="push")
        self.assertEqual((r.pushed, r.pulled), (1, 0))
        self.assertIsNotNone(sb.get("a"))
        self.assertIsNone(sa.get("b"))

    def test_pull_only(self):
        eng_a, sa, sb = _pair()
        sa.put("a", {"v": 1})
        sb.put("b", {"v": 2})
        r = eng_a.sync(LocalPeer(sb), direction="pull")
        self.assertEqual((r.pushed, r.pulled), (0, 1))
        self.assertIsNone(sb.get("a"))
        self.assertIsNotNone(sa.get("b"))

    def test_bad_direction_rejected(self):
        eng_a, _, sb = _pair()
        with self.assertRaises(ValueError):
            eng_a.sync(LocalPeer(sb), direction="sideways")


class EngineCheckpointTests(unittest.TestCase):
    def test_chunked_push_checkpoints_and_resumes(self):
        eng_a, sa, sb = _pair()
        for i in range(5):
            sa.put(f"k{i}", {"v": i})
        peer = FlakyPeer(sb, fail_push_calls={2},
                         exc=SyncConnectionError("net down"))
        eng_a.retry_attempts = 0
        with self.assertRaises(SyncConnectionError):
            eng_a.sync(peer, chunk_size=2)
        # First chunk (2 records) was checkpointed; cursor survived.
        st = eng_a.status()
        self.assertEqual(st["push_seq"], 2)
        self.assertEqual(st["pending_push"], 3)
        # Heal the peer: sync resumes after the checkpoint, no loss.
        peer.fail_push_calls.clear()
        r = eng_a.sync(peer, chunk_size=2)
        self.assertEqual(r.pushed, 3)
        self.assertEqual(sorted(sb.keys()),
                         [f"k{i}" for i in range(5)])

    def test_retry_on_transient_failure(self):
        eng_a, sa, sb = _pair()
        sa.put("k", {"v": 1})
        peer = FlakyPeer(sb, fail_push_calls={1},
                         exc=SyncConnectionError("blip"))
        eng_a.retry_attempts = 2
        eng_a.retry_base_s = 0.01
        r = eng_a.sync(peer)
        self.assertEqual(r.pushed, 1)
        self.assertEqual(peer.push_calls, 2)

    def test_no_retry_on_auth_error(self):
        eng_a, sa, sb = _pair()
        sa.put("k", {"v": 1})
        peer = FlakyPeer(sb, fail_push_calls={1, 2, 3, 4},
                         exc=SyncAuthError("bad token"))
        eng_a.retry_attempts = 3
        with self.assertRaises(SyncAuthError):
            eng_a.sync(peer)
        self.assertEqual(peer.push_calls, 1)  # failed fast, no retries

    def test_on_progress_called_per_chunk(self):
        eng_a, sa, sb = _pair()
        for i in range(3):
            sa.put(f"k{i}", {"v": i})
        calls = []
        eng_a.sync(LocalPeer(sb), chunk_size=2,
                   on_progress=lambda ph, d, t, peer_id: calls.append(
                       (ph, d, t)))
        push_calls = [c for c in calls if c[0] == "push"]
        self.assertEqual(push_calls, [("push", 2, 3), ("push", 3, 3)])
        pull_calls = [c for c in calls if c[0] == "pull"]
        self.assertTrue(pull_calls and pull_calls[-1][1] == pull_calls[-1][2])


class PreviewTests(unittest.TestCase):
    def test_dry_run_changes_nothing(self):
        eng_a, sa, sb = _pair()
        sa.put("a", {"v": 1})
        sb.put("b", {"v": 2})
        r = eng_a.sync(LocalPeer(sb), dry_run=True)
        self.assertTrue(r.dry_run)
        self.assertEqual((r.pushed, r.pulled), (1, 1))
        self.assertIsNone(sb.get("a"))
        self.assertIsNone(sa.get("b"))
        self.assertEqual(eng_a.history(), [])  # dry runs aren't recorded

    def test_preview_reports_conflicts(self):
        eng_a, sa, sb = _pair()
        sa.put("a", {"v": 1})
        sb.put("shared", {"v": "old"})
        sa.put("shared", {"v": "new"})
        pv = eng_a.preview(LocalPeer(sb))
        self.assertEqual(pv.push_total, 2)
        self.assertEqual(pv.pull_total, 1)
        self.assertEqual(pv.would_conflict, ["shared"])
        text = pv.format("rich")
        self.assertIn("preview", text)
        self.assertIn("shared", text)


class HistoryTests(unittest.TestCase):
    def test_history_and_last_run_recorded(self):
        eng_a, sa, sb = _pair()
        sa.put("k", {"v": 1})
        eng_a.sync(LocalPeer(sb))
        eng_a.sync(LocalPeer(sb))
        hist = eng_a.history()
        self.assertEqual(len(hist), 2)
        self.assertTrue(all(h["ok"] for h in hist))
        self.assertEqual(hist[0]["pushed"], 0)  # newest first: no-op run
        self.assertEqual(hist[1]["pushed"], 1)
        last = eng_a.last_run()
        self.assertIsNotNone(last)
        st = eng_a.status()
        self.assertIsNotNone(st["last_run"])
        self.assertEqual(st["tombstones"], 0)

    def test_failed_run_recorded(self):
        eng_a, sa, sb = _pair()
        sa.put("k", {"v": 1})
        peer = FlakyPeer(sb, fail_push_calls={1},
                         exc=SyncConnectionError("down"))
        eng_a.retry_attempts = 0
        with self.assertRaises(SyncConnectionError):
            eng_a.sync(peer)
        last = eng_a.last_run()
        self.assertFalse(last["ok"])
        self.assertIn("down", last["error"])

    def test_sync_all_captures_per_peer_errors(self):
        eng_a, sa, sb = _pair()
        sa.put("k", {"v": 1})
        bad = FlakyPeer(sb, fail_push_calls={1},
                        exc=SyncConnectionError("down"))
        eng_a.retry_attempts = 0
        results = eng_a.sync_all({"good": LocalPeer(sb), "bad": bad})
        self.assertTrue(results["good"].ok)
        self.assertFalse(results["bad"].ok)
        self.assertIn("down", results["bad"].error)


class FormatTests(unittest.TestCase):
    def _result(self):
        from nomorals.sync.engine import SyncResult
        return SyncResult(peer_id="hub", pushed=12, pulled=5,
                          conflicts_resolved=2, duration_s=0.84)

    def test_result_format_styles(self):
        r = self._result()
        rich = r.format("rich")
        self.assertIn("✓", rich)
        self.assertIn("12", rich)
        plain = r.format("plain")
        self.assertIn("[OK]", plain)
        self.assertNotIn("✓", plain)
        compact = r.format("compact")
        self.assertNotIn("\n", compact)
        self.assertIn("hub", compact)
        bad = self._result()
        bad.ok = False
        bad.error = "boom"
        self.assertIn("✗", bad.format("rich"))
        self.assertIn("boom", bad.format("rich"))

    def test_format_status(self):
        eng_a, sa, sb = _pair()
        sa.put("k", {"v": 1})
        eng_a.sync(LocalPeer(sb))
        text = eng_a.format_status("hub", style="rich")
        self.assertIn("hub", text)
        self.assertIn("✓", text)
        compact = eng_a.format_status("hub", style="compact")
        self.assertNotIn("\n", compact)
        fresh = eng_a.format_status("never-seen", style="rich")
        self.assertIn("never", fresh)


class AutoSyncTests(unittest.TestCase):
    """Note: Database uses thread-local connections, so :memory: DBs are
    invisible across threads — these tests use a temp file DB."""

    def _file_pair(self):
        import os
        import tempfile
        paths = []
        for _ in range(2):
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            tmp.close()
            paths.append(tmp.name)
            self.addCleanup(os.unlink, tmp.name)
        db_a, db_b = Database(paths[0]), Database(paths[1])
        sa, sb = SyncStore(db_a, "a"), SyncStore(db_b, "b")
        return SyncEngine(db_a, sa), sa, sb

    def test_start_stop_and_change_trigger(self):
        eng_a, sa, sb = self._file_pair()
        results = []
        auto = AutoSync(eng_a, LocalPeer(sb), interval_s=3600,
                        debounce_s=0.05, on_result=results.append)
        self.assertFalse(auto.running)
        auto.start()
        try:
            self.assertTrue(auto.running)
            deadline = time.time() + 5
            while not results and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(results, "initial sync never ran")
            n = len(results)
            sa.put("later", {"v": 1})  # store change -> debounced re-sync
            deadline = time.time() + 5
            while len(results) == n and time.time() < deadline:
                time.sleep(0.05)
            self.assertGreater(len(results), n)
            self.assertIsNotNone(sb.get("later"))
            stats = auto.stats()
            self.assertGreaterEqual(stats["runs"], 2)
            self.assertEqual(stats["consecutive_failures"], 0)
        finally:
            auto.stop()
        self.assertFalse(auto.running)

    def test_backoff_on_error(self):
        eng_a, sa, _ = self._file_pair()
        eng_a.retry_attempts = 0
        errors = []
        bad = FlakyPeer(SyncStore(_db(), "x"), fail_push_calls={1, 2, 3, 4},
                        exc=SyncConnectionError("down"))
        sa.put("k", {"v": 1})
        auto = AutoSync(eng_a, bad, interval_s=3600, debounce_s=0,
                        backoff_base_s=0.05, backoff_max_s=0.1,
                        on_error=errors.append)
        auto.start()
        try:
            deadline = time.time() + 5
            while not errors and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(errors)
            stats = auto.stats()
            self.assertGreaterEqual(stats["consecutive_failures"], 1)
            self.assertTrue(auto.running)  # keeps trying, doesn't die
        finally:
            auto.stop()


class _HubHandler(http.server.BaseHTTPRequestHandler):
    records: list = []
    mode = "ok"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _guard(self):
        if _HubHandler.mode == "auth":
            self._json(401, {"detail": "bad token"})
            return True
        if _HubHandler.mode == "error":
            self._json(500, {"detail": "kaput"})
            return True
        return False

    def do_POST(self):
        if self._guard():
            return
        if self.path == "/sync/push":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            recs = payload.get("records") or []
            _HubHandler.records.extend(recs)
            self._json(200, {"ok": True, "applied": len(recs)})
        else:
            self._json(404, {})

    def do_GET(self):
        if self._guard():
            return
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/sync/pull":
            since = int(qs.get("since_seq", [0])[0])
            limit = int(qs.get("limit", [500])[0])
            recs = [r for r in _HubHandler.records
                    if r.get("seq", 0) > since][:limit]
            self._json(200, {"ok": True, "records": recs})
        else:
            self._json(404, {})

    def log_message(self, *args):
        pass


class HttpPeerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _HubHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)

    def setUp(self):
        _HubHandler.records = []
        _HubHandler.mode = "ok"
        self.peer = HttpSyncPeer(f"http://127.0.0.1:{self.port}",
                                 retries=1, timeout=5)

    def test_roundtrip(self):
        s = SyncStore(_db(), "d1")
        s.put("a", {"v": 1})
        s.put("b", {"v": 2})
        self.assertTrue(self.peer.ping())
        self.assertEqual(self.peer.push_records(s.list_since_seq(0)), 2)
        recs = self.peer.fetch_since_seq(0)
        self.assertEqual({r.key for r in recs}, {"a", "b"})
        st = self.peer.stats
        self.assertGreater(st["requests"], 0)
        self.assertGreater(st["bytes_sent"], 0)
        self.assertGreater(st["bytes_received"], 0)

    def test_engine_sync_over_http(self):
        db = _db()
        store = SyncStore(db, "d1")
        eng = SyncEngine(db, store)
        store.put("a", {"v": 1})
        r = eng.sync(self.peer)
        self.assertEqual(r.pushed, 1)
        # Hub-side record appears on the next pull.
        hub_rec = SyncRecord("from_hub", {"v": 9}, updated_at=time.time(),
                             device_id="hub", hlc_ts=time.time(),
                             hlc_count=0, seq=99).to_dict()
        _HubHandler.records.append(hub_rec)
        r = eng.sync(self.peer)
        self.assertEqual(r.pulled, 1)
        self.assertEqual(store.get("from_hub").value["v"], 9)

    def test_401_raises_auth_error(self):
        _HubHandler.mode = "auth"
        rec = SyncRecord("k", {"v": 1}, updated_at=time.time(),
                         device_id="d1")
        with self.assertRaises(SyncAuthError):
            self.peer.push_records([rec])
        with self.assertRaises(SyncAuthError):
            self.peer.ping()

    def test_500_raises_retryable_hub_error(self):
        _HubHandler.mode = "error"
        try:
            self.peer.ping()
            self.fail("expected SyncHubError")
        except SyncHubError as exc:
            self.assertEqual(exc.status_code, 500)
            self.assertTrue(exc.retryable)

    def test_unreachable_raises_connection_error(self):
        peer = HttpSyncPeer("http://127.0.0.1:1", retries=1, timeout=1)
        with self.assertRaises(SyncConnectionError):
            peer.ping()

    def test_describe_hides_token(self):
        peer = HttpSyncPeer("http://x", token="secret")
        d = peer.describe()
        self.assertNotIn("secret", json.dumps(d))
        self.assertEqual(d["auth"], "bearer")
        self.assertIn("http://x", repr(peer))


class ErrorHierarchyTests(unittest.TestCase):
    def test_hierarchy(self):
        for cls in (SyncAuthError, SyncConnectionError, SyncHubError):
            self.assertTrue(issubclass(cls, SyncError))
        err = SyncHubError("x", status_code=503, detail="d", retryable=True)
        self.assertTrue(err.retryable)
        self.assertEqual(err.status_code, 503)


if __name__ == "__main__":
    unittest.main()
