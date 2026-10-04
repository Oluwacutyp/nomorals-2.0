"""R23 audit tests.

- doctor fix: ``LLMRouter(context.settings)`` raised TypeError (keyword-only
  args); the setup wizard now rebuilds via the public ``build_router`` —
  the same builder the boot path uses. Pins the bug and the fix.
- HTTP transport (the multi-device gap): ``HttpTransport`` (mesh) and
  ``HttpSyncPeer`` (sync) against a live ``nomorals.hub`` server —
  register/heartbeat/dispatch/poll/complete/fail, two-device sync with
  hub fan-out, LWW conflicts, tombstones, backdated writes, and the auth
  posture (token required, loopback-only without one).
- stream: same-timestamp events were re-emitted on every poll (inclusive
  ``since`` + advance-only-on-greater cursor). Now exactly-once per
  connection, at-least-once across reconnects.
- native: ``bpe_train`` unpacked the whole output capacity (the C++
  returns a merge count, not cells) so the length check failed on every
  run — the native path never delivered. Plus a C++ inverted-index bug
  that re-picked the same pair forever. Both fixed; the native trainer
  now matches the pure-Python reference exactly. ``topk`` now fails
  fast on ragged rows / short queries instead of over-reading the heap.
"""

from __future__ import annotations

import array
import json
import random
import socket
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals import native
from nomorals.agents.context import _build_router, build_router
from nomorals.core.config import Settings, load_settings
from nomorals.core.events import EventBus
from nomorals.hub import HubServer, serve
from nomorals.mesh import (
    HttpTransport,
    LocalTransport,
    MeshNode,
    MeshTask,
    NodeUnknown,
    TransportError,
)
from nomorals.mesh.errors import TransportError as MeshTransportError
from nomorals.os.timeline import Timeline
from nomorals.storage.db import Database
from nomorals.stream import StreamServer
from nomorals.sync import HttpSyncPeer, LocalPeer, SyncEngine, SyncRecord, SyncStore
from nomorals.sync.errors import SyncError


def _tmp_db(name: str) -> tuple[Database, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="nm-r23-"))
    db = Database(str(tmp / name))
    db.migrate()
    return db, tmp


# ── 1. doctor fix ─────────────────────────────────────────────────────────

class DoctorRouterFixTest(unittest.TestCase):
    def test_llm_router_rejects_positional_settings(self):
        """Pin the original bug: LLMRouter() takes keyword-only args."""
        from nomorals.llm.router import LLMRouter
        settings = Settings()
        with self.assertRaises(TypeError):
            LLMRouter(settings)  # noqa: B018 - the doctor did exactly this

    def test_build_router_is_public_and_aliased(self):
        self.assertIs(_build_router, build_router)
        self.assertIn("build_router", dir(__import__(
            "nomorals.agents.context", fromlist=["x"])))

    def test_doctor_rebuild_path_produces_working_router(self):
        """What ``nm setup``'s test step now does: rebuild + chat."""
        settings = Settings()
        settings.llm.provider = "mock"
        settings.llm.fallback_chain = []
        bus = EventBus().start()
        try:
            router = build_router(settings, bus)
            self.assertTrue(router.providers())
            from nomorals.llm.base import Message, SamplingParams
            resp = router.chat(
                [Message.user("Say 'Hello, I'm working!' in one sentence.")],
                SamplingParams(max_tokens=50),
            )
            self.assertTrue(resp.ok)
            self.assertTrue(resp.text)
        finally:
            bus.stop()

    def test_doctor_py_uses_build_router(self):
        src = Path(__file__).resolve().parents[1] / "nomorals" / "cmdline" / \
            "commands" / "doctor.py"
        text = src.read_text()
        self.assertIn("build_router", text)
        self.assertNotIn("LLMRouter(context.settings)", text)

    def test_hub_settings_env_map(self):
        settings = load_settings(env={
            "NM_HUB_URL": "http://192.168.1.5:8861",
            "NM_HUB_TOKEN": "s3cret",
            "NM_HUB_PORT": "8877",
        }, use_env_file=False)
        self.assertEqual(settings.hub.url, "http://192.168.1.5:8861")
        self.assertEqual(settings.hub.token, "s3cret")
        self.assertEqual(settings.hub.port, 8877)


# ── 2. HTTP transport ─────────────────────────────────────────────────────

class HubFixture(unittest.TestCase):
    def setUp(self):
        self.hub_db, self._tmp = _tmp_db("hub.db")
        self.hub = serve(self.hub_db, host="127.0.0.1", port=0,
                         background=True)
        self.url = self.hub.url

    def tearDown(self):
        self.hub.stop()
        self.hub_db.close()


class HttpMeshTransportTest(HubFixture):
    def test_full_mesh_cycle_over_http(self):
        t = HttpTransport(self.url, retries=2)
        node = t.register("phone", platform="termux",
                          capabilities=["chat", "camera"])
        self.assertEqual(node.name, "phone")
        self.assertEqual(node.platform, "termux")
        self.assertEqual(node.capabilities, ["chat", "camera"])

        t.heartbeat(node.node_id)
        active = t.active_nodes()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].node_id, node.node_id)
        self.assertLess(active[0].age, 5.0)

        job = t.dispatch("summarize", {"text": "hi"}, origin_node=node.node_id,
                         priority=3)
        self.assertTrue(job)
        tasks = t.poll(node.node_id, batch=5)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_type, "summarize")
        self.assertEqual(tasks[0].payload, {"text": "hi"})
        self.assertEqual(tasks[0].priority, 3)
        self.assertEqual(tasks[0].origin_node, node.node_id)
        t.complete(tasks[0].job_id, result={"ok": True})

        job2 = t.dispatch("ping", {}, origin_node=node.node_id,
                          target_node=node.node_id)
        got = t.poll(node.node_id, batch=5)
        self.assertEqual([g.job_id for g in got], [job2])
        t.fail(job2, error="boom", retry=False)
        # failed with retry=False → gone from the queue
        self.assertEqual(t.poll(node.node_id, batch=5), [])

    def test_heartbeat_unknown_node_raises_node_unknown(self):
        t = HttpTransport(self.url, retries=1)
        with self.assertRaises(NodeUnknown):
            t.heartbeat("no-such-node")

    def test_register_is_idempotent_by_node_id(self):
        t = HttpTransport(self.url, retries=2)
        a = t.register("phone", node_id="fixed-id")
        b = t.register("phone-renamed", node_id="fixed-id")
        self.assertEqual(a.node_id, b.node_id)
        self.assertEqual(b.name, "phone-renamed")

    def test_matches_local_transport_behavior(self):
        """The documented contract: the remote transport must match the
        reference (LocalTransport) call-for-call."""
        local_db, _ = _tmp_db("local.db")
        local = LocalTransport(local_db)
        remote = HttpTransport(self.url, retries=2)
        for transport in (local, remote):
            node = transport.register("n1", platform="linux")
            transport.heartbeat(node.node_id)
            self.assertEqual(len(transport.active_nodes()), 1)
            job = transport.dispatch("t", {"k": "v"}, origin_node=node.node_id)
            tasks = transport.poll(node.node_id)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].job_id, job)
            self.assertEqual(tasks[0].payload, {"k": "v"})
            transport.complete(job, result=1)
            self.assertEqual(transport.poll(node.node_id), [])

    def test_wire_round_trip_dataclasses(self):
        node = MeshNode(node_id="x", name="n", platform="p",
                        capabilities=["c"], last_seen=1.5, created_at=1.0)
        self.assertEqual(MeshNode.from_dict(node.to_dict()).to_dict(),
                         node.to_dict())
        task = MeshTask(job_id="j", task_type="t", payload={"a": 1},
                        target_node="n", origin_node="o", priority=2,
                        attempts=3)
        self.assertEqual(MeshTask.from_dict(task.to_dict()).to_dict(),
                         task.to_dict())
        rec = SyncRecord(key="k", value={"v": 1}, updated_at=1.0,
                         device_id="d", deleted=True, seq=9)
        back = SyncRecord.from_dict(rec.to_dict())
        self.assertEqual(back.key, "k")
        self.assertEqual(back.value, {"v": 1})
        self.assertTrue(back.deleted)


class HttpSyncPeerTest(HubFixture):
    def _device(self, name: str):
        db, _ = _tmp_db(f"{name}.db")
        store = SyncStore(db, device_id=name)
        return SyncEngine(db, store), store

    def test_two_devices_sync_through_hub(self):
        engine_a, store_a = self._device("phone")
        engine_b, store_b = self._device("laptop")
        store_a.put("note:1", {"text": "hello hub"})
        store_a.put("note:2", {"text": "second"})

        ra = engine_a.sync(HttpSyncPeer(self.url, retries=2), peer_id="hub")
        self.assertEqual((ra.pushed, ra.pulled), (2, 0))

        # hub fans out to the second device
        rb = engine_b.sync(HttpSyncPeer(self.url, retries=2), peer_id="hub")
        self.assertEqual((rb.pushed, rb.pulled), (0, 2))
        self.assertEqual(store_b.get("note:1").value, {"text": "hello hub"})

        # idempotent re-sync: nothing new either way
        rc = engine_a.sync(HttpSyncPeer(self.url, retries=2), peer_id="hub")
        self.assertEqual((rc.pushed, rc.pulled), (0, 0))

    def test_lww_conflict_resolves_over_http(self):
        engine_a, store_a = self._device("phone")
        engine_b, store_b = self._device("laptop")
        peer = lambda: HttpSyncPeer(self.url, retries=2)  # noqa: E731
        store_a.put("k", {"v": "phone"})
        engine_a.sync(peer(), peer_id="hub")
        engine_b.sync(peer(), peer_id="hub")
        # laptop writes newer
        store_b.put("k", {"v": "laptop"})
        engine_b.sync(peer(), peer_id="hub")
        engine_a.sync(peer(), peer_id="hub")
        self.assertEqual(store_a.get("k").value, {"v": "laptop"})

    def test_tombstone_replicates_over_http(self):
        engine_a, store_a = self._device("phone")
        engine_b, store_b = self._device("laptop")
        peer = lambda: HttpSyncPeer(self.url, retries=2)  # noqa: E731
        store_a.put("gone", {"v": 1})
        engine_a.sync(peer(), peer_id="hub")
        engine_b.sync(peer(), peer_id="hub")
        self.assertIsNotNone(store_b.get("gone"))
        store_a.delete("gone")
        engine_a.sync(peer(), peer_id="hub")
        engine_b.sync(peer(), peer_id="hub")
        self.assertIsNone(store_b.get("gone"))

    def test_backdated_write_not_missed(self):
        """Seq cursors (not timestamps) drive the pull — an old
        updated_at can never slip past unseen."""
        engine_a, store_a = self._device("phone")
        engine_b, store_b = self._device("laptop")
        peer = lambda: HttpSyncPeer(self.url, retries=2)  # noqa: E731
        engine_b.sync(peer(), peer_id="hub")  # advance pull cursor
        store_a.put("old", {"v": 1}, updated_at=1.0)  # backdated
        engine_a.sync(peer(), peer_id="hub")
        rb = engine_b.sync(peer(), peer_id="hub")
        self.assertEqual(rb.pulled, 1)
        self.assertEqual(store_b.get("old").value, {"v": 1})

    def test_push_batches_large_records(self):
        engine_a, store_a = self._device("phone")
        big = {"blob": "x" * 200_000}
        for i in range(5):
            store_a.put(f"big:{i}", dict(big, i=i))
        res = engine_a.sync(HttpSyncPeer(self.url, retries=2), peer_id="hub")
        self.assertEqual(res.pushed, 5)


class HubAuthTest(unittest.TestCase):
    def test_no_token_refuses_non_loopback_bind(self):
        db, _ = _tmp_db("h.db")
        with self.assertRaises(ValueError):
            HubServer(db, host="0.0.0.0", port=8861, token="")

    def test_token_required_on_protected_routes(self):
        db, _ = _tmp_db("h.db")
        hub = serve(db, host="127.0.0.1", port=0, token="s3cret",
                    background=True)
        try:
            with self.assertRaises(MeshTransportError):
                HttpTransport(hub.url, retries=1).active_nodes()
            with self.assertRaises(MeshTransportError):
                HttpTransport(hub.url, token="wrong",
                              retries=1).active_nodes()
            with self.assertRaises(SyncError):
                HttpSyncPeer(hub.url, retries=1).fetch_since_seq(0)
            # correct token works
            node = HttpTransport(hub.url, token="s3cret",
                                 retries=1).register("authed")
            self.assertEqual(node.name, "authed")
            # /health stays tokenless (load-balancer friendly)
            import urllib.request
            with urllib.request.urlopen(hub.url + "/health",
                                        timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertTrue(json.loads(resp.read())["ok"])
        finally:
            hub.stop()
            db.close()

    def test_unreachable_hub_raises_transport_error(self):
        t = HttpTransport("http://127.0.0.1:9", retries=2, timeout=0.5)
        with self.assertRaises(TransportError):
            t.active_nodes()
        p = HttpSyncPeer("http://127.0.0.1:9", retries=2, timeout=0.5)
        with self.assertRaises(SyncError):
            p.fetch_since_seq(0)

    def test_hub_rejects_oversized_body(self):
        db, _ = _tmp_db("h.db")
        hub = serve(db, host="127.0.0.1", port=0, background=True)
        try:
            import http.client
            big = b"x" * (9 * 1024 * 1024)
            conn = http.client.HTTPConnection("127.0.0.1", hub.port,
                                              timeout=10)
            try:
                conn.request("POST", "/sync/push", body=big,
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                # Either the server answers 413, or it closes the
                # connection mid-upload (RemoteDisconnected) — both are
                # a rejection: the body is never processed.
                self.assertEqual(resp.status, 413)
                resp.read()
            except (http.client.RemoteDisconnected, ConnectionResetError,
                    BrokenPipeError):
                pass  # server hung up on the oversized body — rejected
            finally:
                conn.close()
        finally:
            hub.stop()
            db.close()


# ── 3. stream ─────────────────────────────────────────────────────────────

class StreamDedupTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name
        self.timeline = Timeline(self.db_path)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.server = StreamServer(
            lambda: Timeline(self.db_path), host="127.0.0.1", port=self.port)
        self.server.start(background=True)
        for _ in range(50):
            try:
                import urllib.request
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.05)

    def tearDown(self):
        self.server.stop()
        self.timeline.close()
        try:
            Path(self.db_path).unlink()
        except OSError:
            pass

    def _record_at(self, topic: str, ts: float):
        # Same-timestamp events: the shape that used to duplicate forever.
        ev = SimpleNamespace(topic=topic, session_id="s1",
                             data={"v": topic})
        eid = self.timeline.record(ev)
        # force the timestamp (record() uses time.time())
        tl = Timeline(self.db_path)
        try:
            tl._conn.execute("UPDATE event_log SET ts=? WHERE event_id=?",
                             (ts, eid))
            tl._conn.commit()
        finally:
            tl.close()
        return eid

    def _read_frames(self, since: float, duration: float = 3.5):
        import urllib.request
        frames: list[dict] = []
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/stream?since={since}",
            headers={"Accept": "text/event-stream"})
        # stream until duration elapses, then close from our side
        stop_at = time.time() + duration
        resp = urllib.request.urlopen(req, timeout=duration + 5)
        buf = b""
        try:
            while time.time() < stop_at:
                try:
                    chunk = resp.read(1)
                except Exception:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    if frame.startswith(b"event: timeline"):
                        for line in frame.split(b"\n"):
                            if line.startswith(b"data: "):
                                frames.append(json.loads(line[6:]))
        finally:
            resp.close()
        return frames

    def test_same_timestamp_events_emitted_exactly_once(self):
        ts = time.time()
        ids = [self._record_at(f"dup.{i}", ts) for i in range(5)]
        frames = self._read_frames(since=0, duration=3.5)
        seen = [f["event_id"] for f in frames]
        for eid in ids:
            self.assertEqual(seen.count(eid), 1,
                             f"event {eid} emitted {seen.count(eid)}x")
        self.assertEqual(len(seen), len(set(seen)), "duplicate frames")

    def test_resume_is_at_least_once(self):
        ts = time.time()
        eid = self._record_at("resume.one", ts)
        frames = self._read_frames(since=ts, duration=2.5)
        # inclusive resume: the boundary event arrives once on the new
        # connection, not zero times and not repeatedly.
        self.assertEqual([f["event_id"] for f in frames].count(eid), 1)


# ── 4. native ─────────────────────────────────────────────────────────────

def _bpe_available():
    return native.bpe_available()


class NativeBpeParityTest(unittest.TestCase):
    """The native BPE trainer must match the pure-Python reference.

    Two bugs hid here because no test exercised the present-kernel path:
    (1) the merge count was unpacked as cells, so every run returned
    None; (2) the C++ inverted index went stale, re-picking one pair.
    """

    def _py_merges(self, texts, target, min_freq):
        words: Counter = Counter()
        for text in texts:
            for w in text.split():
                words[w] += 1
        splits = {w: list(w) for w in words}
        merges: list[tuple[str, str]] = []
        while len(merges) < target:
            counts: Counter = Counter()
            for word, f in words.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    counts[(syms[i], syms[i + 1])] += f
            if not counts:
                break
            pair, freq = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            if freq < min_freq:
                break
            merges.append(pair)
            merged = pair[0] + pair[1]
            for word in list(splits):
                syms = splits[word]
                if pair[0] not in syms:
                    continue
                out, i = [], 0
                while i < len(syms):
                    if (i < len(syms) - 1 and syms[i] == pair[0]
                            and syms[i + 1] == pair[1]):
                        out.append(merged)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                splits[word] = out
        return merges

    @unittest.skipUnless(_bpe_available(), "BPE kernel not built")
    def test_train_matches_python_reference(self):
        rng = random.Random(20261004)
        vocab = ["hello", "world", "help", "held", "helmet", "hero", "her",
                 "hell", "low", "lower", "lowest", "new", "newer", "newest",
                 "wide", "wider", "shelf", "shell"]
        for _ in range(10):
            texts = [" ".join(rng.choice(vocab)
                               for _ in range(rng.randint(5, 40)))
                     for _ in range(rng.randint(1, 4))]
            target = rng.randint(1, 12)
            min_freq = rng.randint(1, 3)
            c: Counter = Counter()
            for t in texts:
                for w in t.split():
                    c[w] += 1
            words, freqs = list(c.keys()), list(c.values())
            expected = self._py_merges(texts, target, min_freq)
            got = native.bpe_train(words, freqs, target_merges=target,
                                   min_frequency=min_freq)
            self.assertIsNotNone(got, "native bpe_train must deliver")
            self.assertEqual(list(got), expected)

    @unittest.skipUnless(_bpe_available(), "BPE kernel not built")
    def test_train_never_repeats_a_pair(self):
        # the stale-index shape: one pair winning forever
        words = ["hello", "world", "hello", "help", "held", "helmet"]
        freqs = [10, 8, 10, 5, 3, 2]
        got = native.bpe_train(words, freqs, target_merges=8,
                               min_frequency=2)
        self.assertIsNotNone(got)
        self.assertEqual(len(got), len(set(got)),
                         f"repeated merges: {got}")

    @unittest.skipUnless(_bpe_available(), "BPE kernel not built")
    def test_encode_roundtrip_on_native_merges(self):
        from nomorals.native import bpe_pack_stream
        words = ["hello", "world", "low", "lower"]
        freqs = [10, 8, 6, 4]
        merges = native.bpe_train(words, freqs, target_merges=8,
                                  min_frequency=2)
        self.assertIsNotNone(merges)
        stream = bpe_pack_stream(merges)
        for w in words + ["unseenword"]:
            syms = native.bpe_encode_word(w, stream)
            self.assertIsNotNone(syms)
            self.assertEqual("".join(syms), w)


class NativeTopkValidationTest(unittest.TestCase):
    @unittest.skipUnless(native.available(), "vecsim kernel not built")
    def test_short_query_fails_fast(self):
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with self.assertRaises(ValueError):
            native.topk(matrix, [1.0, 2.0], 1)

    @unittest.skipUnless(native.available(), "vecsim kernel not built")
    def test_ragged_matrix_fails_fast(self):
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0]]
        with self.assertRaises(ValueError):
            native.topk(matrix, [1.0, 2.0, 3.0], 1)

    @unittest.skipUnless(native.available(), "vecsim kernel not built")
    def test_topk_matches_reference(self):
        rng = random.Random(7)
        dim = 16
        matrix = [[rng.uniform(-1, 1) for _ in range(dim)]
                  for _ in range(50)]
        query = [rng.uniform(-1, 1) for _ in range(dim)]
        got = native.topk(matrix, query, 5)
        want = native.python_topk(matrix, query, 5)
        self.assertEqual([i for _, i in got], [i for _, i in want])
        for (gs, _), (ws, _) in zip(got, want):
            self.assertAlmostEqual(gs, ws, places=4)


if __name__ == "__main__":
    unittest.main()
