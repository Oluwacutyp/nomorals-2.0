"""Wave 90: honest model status + real C++ in the vector-search hot path.

1. The /status model line must only call a provider "answering" if it has
   ACTUALLY succeeded — the screenshot incident: "hf_serverless is
   answering for now" while 80 messages went unanswered because that
   fallback was failing too (no token).

2. The main system's memory recall is a top-k cosine search per reply.
   On a phone (no numpy) that was pure Python.  Wave 90 adds
   ``nomorals/native/vecsim.cpp``: a C ABI top-k, built on-device
   (``nm native --build``), with the pure-Python implementation as the
   reference fallback — every native result is checked against it.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals import native  # noqa: E402
from nomorals.core.config import Settings, get_settings  # noqa: E402
from nomorals.storage.db import Database  # noqa: E402
from nomorals.storage.vectors import VectorStore, topk_native  # noqa: E402

HAS_COMPILER = native.find_compiler() is not None


# ── the C++ module itself ────────────────────────────────────────────────────

@unittest.skipUnless(HAS_COMPILER, "no C++ compiler in this environment")
class NativeBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build ONCE for the whole process (shared by every native test).
        ok, message = native.build()
        assert ok, f"native build failed: {message}"
        cls.lib = native.load()
        assert cls.lib is not None

    def test_version_string(self):
        self.assertRegex(self.lib.nm_vecsim_version().decode(), r"^\d+\.\d+\.\d+$")

    def test_available_reports_true(self):
        self.assertTrue(native.available())
        self.assertEqual(native.info()["backend"], "native-cpp")


class NativeNumericsTest(unittest.TestCase):
    """Native vs pure-Python reference — the reference is the spec."""

    def setUp(self):
        if not HAS_COMPILER:
            self.skipTest("no C++ compiler in this environment")
        if not native.available():
            ok, message = native.build()
            self.assertTrue(ok, message)

    def test_dot_matches_reference(self):
        rng = random.Random(7)
        for dim in (1, 2, 3, 16, 127, 128):
            a = [rng.uniform(-1, 1) for _ in range(dim)]
            b = [rng.uniform(-1, 1) for _ in range(dim)]
            self.assertAlmostEqual(native.dot(a, b), native.python_dot(a, b),
                                   places=5, msg=f"dim={dim}")

    def test_topk_matches_reference(self):
        rng = random.Random(42)
        rows, dim, k = 500, 64, 10
        matrix = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(rows)]
        query = [rng.uniform(-1, 1) for _ in range(dim)]
        got = native.topk(matrix, query, k)
        ref = native.python_topk(matrix, query, k)
        self.assertEqual(len(got), k)
        self.assertEqual([i for _, i in got], [i for _, i in ref],
                         "top-k indices differ from the reference")
        for (s1, _), (s2, _) in zip(got, ref):
            self.assertAlmostEqual(s1, s2, places=4)

    def test_topk_packed_buffer_matches(self):
        # the form VectorStore feeds it: one flat float32 array
        import array as _array

        rng = random.Random(99)
        rows, dim, k = 300, 32, 5
        matrix = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(rows)]
        query = [rng.uniform(-1, 1) for _ in range(dim)]
        flat = _array.array("f")
        for row in matrix:
            flat.extend(row)
        got = native.topk(flat, query, k)
        ref = native.python_topk(matrix, query, k)
        self.assertEqual([i for _, i in got], [i for _, i in ref])

    def test_topk_edge_cases(self):
        rng = random.Random(1)
        matrix = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(4)]
        query = [0.5] * 8
        self.assertEqual(native.topk(matrix, query, 0), [])
        self.assertEqual(native.topk([], query, 3), [])
        # k > rows → all rows, best first
        got = native.topk(matrix, query, 10)
        self.assertEqual(len(got), 4)
        scores = [s for s, _ in got]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_benchmark_agrees_with_reference(self):
        bench = native.benchmark(n=800, dim=64)
        self.assertTrue(bench["match"], bench)
        self.assertEqual(bench["native"], bench["python"])


class NativeFallbackTest(unittest.TestCase):
    """Without the library, everything still works via pure Python."""

    def test_topk_falls_back(self):
        rng = random.Random(5)
        matrix = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(50)]
        query = [rng.uniform(-1, 1) for _ in range(16)]
        got = native.topk(matrix, query, 5)
        ref = native.python_topk(matrix, query, 5)
        self.assertEqual([i for _, i in got], [i for _, i in ref])

    def test_find_compiler_is_sane(self):
        self.assertIsInstance(native.find_compiler(), (str, type(None)))


# ── VectorStore integration ──────────────────────────────────────────────────

def _store(tmp: str, *, use_native: bool, use_numpy: bool = False) -> VectorStore:
    db = Database(f"{tmp}/v.db")
    db.migrate()
    return VectorStore(db, use_native=use_native, use_numpy=use_numpy)


def _populate(store: VectorStore, n: int = 200, dim: int = 32, seed: int = 7) -> None:
    rng = random.Random(seed)
    vectors = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(n)]
    store.put_many(
        ((v, "memory", f"mem-{i}") for i, v in enumerate(vectors)),
    )
    return vectors


@unittest.skipUnless(HAS_COMPILER, "no C++ compiler in this environment")
class VectorStoreNativeTest(unittest.TestCase):
    def setUp(self):
        if not native.available():
            ok, message = native.build()
            self.assertTrue(ok, message)
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="nm-w90-")

    def tearDown(self):
        import shutil as _sh

        _sh.rmtree(self.tmp, ignore_errors=True)

    def test_native_and_python_paths_return_the_same_hits(self):
        store_nat = _store(self.tmp + "/nat", use_native=True)
        vectors = _populate(store_nat)
        store_py = _store(self.tmp + "/py", use_native=False)
        _populate(store_py, n=200, dim=32, seed=7)  # same seed → same vectors

        rng = random.Random(123)
        query = [rng.uniform(-1, 1) for _ in range(32)]
        hits_nat = store_nat.search(query, limit=10)
        hits_py = store_py.search(query, limit=10)
        self.assertEqual(len(hits_nat), 10)
        # ULIDs differ between stores; owner_id is the deterministic key
        self.assertEqual([h.owner_id for h in hits_nat],
                         [h.owner_id for h in hits_py])
        for a, b in zip(hits_nat, hits_py):
            self.assertAlmostEqual(a.score, b.score, places=4)

    def test_backend_reported(self):
        store = _store(self.tmp + "/b", use_native=True)
        _populate(store, n=20)
        self.assertEqual(store.stats_snapshot()["backend"], "native-cpp")
        store_py = _store(self.tmp + "/b2", use_native=False)
        _populate(store_py, n=20)
        self.assertEqual(store_py.stats_snapshot()["backend"], "pure-python")

    def test_owner_filter_stays_correct_with_native(self):
        store = _store(self.tmp + "/o", use_native=True)
        vectors = _populate(store, n=100)
        # extra vectors under a different owner
        rng = random.Random(55)
        for i in range(50):
            store.put([rng.uniform(-1, 1) for _ in range(32)],
                      owner_type="note", owner_id=f"note-{i}")
        query = [rng.uniform(-1, 1) for _ in range(32)]
        hits = store.search(query, limit=10, owner_type="memory")
        self.assertTrue(hits)
        self.assertTrue(all(h.owner_type == "memory" for h in hits))


# ── the honest model line (screenshot incident) ──────────────────────────────

class HonestModelLineTest(unittest.TestCase):
    def _line(self, active, chain, health, caps=None):
        from nomorals.agents import partner_runtime as pr

        class _Router:
            @staticmethod
            def stats_snapshot():
                snap = {"active": active, "chain": chain, "health": health}
                if caps is not None:
                    snap["chain_caps"] = {k: sorted(v) for k, v in caps.items()}
                return snap

        ctx = types.SimpleNamespace(router=_Router())
        runtime = pr.PartnerRuntime.__new__(pr.PartnerRuntime)  # no __init__
        runtime.context = ctx
        return runtime._model_status_line()

    def test_dead_chain_says_no_model_answering(self):
        line = self._line(
            "llama_cpp",
            ["llama_cpp", "hf_serverless", "ocr"],
            {
                "llama_cpp": {"failures": 5, "last_error": "local server not responding — skipped"},
                "hf_serverless": {"failures": 4, "last_error": "HTTP 401: invalid api key"},
                "ocr": {"failures": 1, "last_error": "tesseract not installed"},
            },
        )
        self.assertIn("no model is answering", line)
        self.assertNotIn("is answering for now", line)
        self.assertIn("bad credentials", line)  # hf's reason, in plain words
        self.assertNotIn("HTTP 401", line)

    def test_successful_fallback_is_named(self):
        line = self._line(
            "llama_cpp",
            ["llama_cpp", "hf_serverless"],
            {
                "llama_cpp": {"failures": 3, "last_error": "timed out"},
                "hf_serverless": {"failures": 0, "last_success": 1700000000.0},
            },
        )
        self.assertIn("hf_serverless is answering for now", line)
        self.assertIn("took too long", line)

    def test_healthy_primary_unchanged(self):
        line = self._line("llama_cpp", ["llama_cpp", "hf_serverless"],
                          {"llama_cpp": {"failures": 0}})
        self.assertIn("llama_cpp", line)
        self.assertIn("chain", line)
        self.assertNotIn("no model is answering", line)

    def test_no_backup_configured(self):
        line = self._line(
            "llama_cpp", ["llama_cpp"],
            {"llama_cpp": {"failures": 2, "last_error": "connection refused"}},
        )
        self.assertIn("no model is answering", line)
        self.assertIn("no backup configured", line)
        self.assertIn("not running", line)

    def test_non_chat_backup_is_not_called_untried(self):
        # ocr cannot chat — it is never "untried", it is simply not a
        # reply fallback.  The line must not imply it was skipped.
        line = self._line(
            "llama_cpp",
            ["llama_cpp", "hf_serverless", "ocr"],
            {
                "llama_cpp": {"failures": 5, "last_error": "local server not responding — skipped"},
                "hf_serverless": {"failures": 4, "last_error": "HTTP 400: model not served"},
                "ocr": {"failures": 0},
            },
            caps={"llama_cpp": ["chat"], "hf_serverless": ["chat"],
                  "ocr": ["describe_image"]},
        )
        self.assertIn("no model is answering", line)
        self.assertIn("hf_serverless failed too", line)
        self.assertNotIn("ocr has not been tried", line)

    def test_only_non_chat_backups_says_only_local_can_reply(self):
        line = self._line(
            "llama_cpp",
            ["llama_cpp", "ocr"],
            {
                "llama_cpp": {"failures": 5, "last_error": "local server not responding — skipped"},
                "ocr": {"failures": 0},
            },
            caps={"llama_cpp": ["chat"], "ocr": ["describe_image"]},
        )
        self.assertIn("no model is answering", line)
        self.assertIn("cannot chat", line)


# ── CLI: nm native ───────────────────────────────────────────────────────────

class NativeCliTest(unittest.TestCase):
    def test_status_runs(self):
        proc = subprocess.run(
            [sys.executable, "-m", "nomorals.cli", "native"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("native vector search:", proc.stdout)

    @unittest.skipUnless(HAS_COMPILER, "no C++ compiler in this environment")
    def test_build_and_benchmark(self):
        root = str(Path(__file__).resolve().parents[1])
        proc = subprocess.run(
            [sys.executable, "-m", "nomorals.cli", "native", "--build"],
            capture_output=True, text=True, cwd=root, timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK", proc.stdout)
        proc = subprocess.run(
            [sys.executable, "-m", "nomorals.cli", "native", "--benchmark"],
            capture_output=True, text=True, cwd=root, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("benchmark", proc.stdout)
        self.assertIn("agreement: True", proc.stdout)


if __name__ == "__main__":
    unittest.main()
