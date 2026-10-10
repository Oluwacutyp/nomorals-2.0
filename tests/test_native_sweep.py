"""Sweep tests for nomorals.native v2 — SIMD dispatch, BPE v2, new APIs.

Covers the behavior changed by the native sweep:
- vecsim 2.0.0: runtime SIMD dispatch reporting (``simd_level()``), AVX2 dot
  parity with the pure-Python reference, prefetch/blocked top-k parity.
- New public API: ``normalize()``, ``topk_flat()`` (explicit-dim packed
  search), ``simd_level()``; ``info()``/``benchmark()`` report ``simd``.
- ``build()``: parallel compile, (bool, str) contract, no-rebuild fast path.
- Thread-safe lazy ``load()`` under concurrent use.
- bpe 2.0.0: incremental counting + max-heap selection stays list-equal to
  the Python reference (incl. non-ASCII corpora — the old kernel diverged
  there), never repeats a pair, encode round-trips.
- biometric v2: rich ``BiometricResult`` statuses, 5-denial/30s lockout,
  in-flight ``busy`` guard, bool contract of ``request_biometric`` kept.

Tests needing a compiled library skip when no compiler/.so is available;
pure-Python paths are tested through the missing-library harness.
"""

from __future__ import annotations

import array
import random
import subprocess
import threading
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from nomorals import native
from nomorals.native import biometric as bio_mod


def _lib_present() -> bool:
    return native.available()


def _py_bpe_merges(words, freqs, target, min_freq):
    """The pure-Python BPE reference (mirrors training/tokenize.py)."""
    w2f = dict(zip(words, freqs))
    splits = {w: list(w) for w in words}
    merges: list[tuple[str, str]] = []
    while len(merges) < target:
        counts: Counter = Counter()
        for word, f in w2f.items():
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


class SimdDispatchTests(unittest.TestCase):
    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_simd_level_is_known(self):
        level = native.simd_level()
        self.assertIn(level, ("avx2", "sse2", "neon", "scalar"))

    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_info_and_benchmark_report_simd(self):
        self.assertEqual(native.info()["simd"], native.simd_level())
        bench = native.benchmark(n=200, dim=32)
        self.assertEqual(bench["simd"], native.simd_level())
        self.assertTrue(bench["match"])

    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_dot_matches_reference_small_and_avx2_sized(self):
        rng = random.Random(11)
        for dim in (3, 15, 16, 17, 64, 130):
            a = [rng.uniform(-1, 1) for _ in range(dim)]
            b = [rng.uniform(-1, 1) for _ in range(dim)]
            self.assertAlmostEqual(native.dot(a, b),
                                   native.python_dot(a, b), places=5,
                                   msg=f"dim={dim}")

    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_topk_matches_reference_avx2_dims(self):
        rng = random.Random(12)
        for dim in (16, 64, 100):
            matrix = [[rng.uniform(-1, 1) for _ in range(dim)]
                      for _ in range(60)]
            query = [rng.uniform(-1, 1) for _ in range(dim)]
            got = native.topk(matrix, query, 7)
            want = native.python_topk(matrix, query, 7)
            self.assertEqual([i for _, i in got], [i for _, i in want],
                             msg=f"dim={dim}")
            for (gs, _), (ws, _) in zip(got, want):
                self.assertAlmostEqual(gs, ws, places=4, msg=f"dim={dim}")

    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_topk_deterministic_tie_break(self):
        matrix = [[1.0, 0.0]] * 8
        got = native.topk(matrix, [1.0, 0.0], 8)
        self.assertEqual([i for _, i in got], list(range(8)))


class NormalizeTests(unittest.TestCase):
    def _reference(self, values):
        norm = sum(v * v for v in values) ** 0.5
        if norm == 0.0:
            return list(values)
        return [v / norm for v in values]

    def test_normalize_matches_reference(self):
        rng = random.Random(13)
        for _ in range(20):
            vec = [rng.uniform(-2, 2) for _ in range(rng.randint(1, 40))]
            got = native.normalize(vec)
            want = self._reference(vec)
            self.assertEqual(len(got), len(want))
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=5)

    def test_normalize_zero_and_empty(self):
        self.assertEqual(native.normalize([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])
        self.assertEqual(native.normalize([]), [])

    def test_normalize_unit_vector(self):
        self.assertEqual(native.normalize([0.0, 1.0]), [0.0, 1.0])


class TopkFlatTests(unittest.TestCase):
    def _data(self, seed=21, rows=40, dim=16):
        rng = random.Random(seed)
        flat = array.array("f", (rng.uniform(-1, 1)
                                 for _ in range(rows * dim)))
        query = [rng.uniform(-1, 1) for _ in range(dim)]
        return flat, query, rows, dim

    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_topk_flat_matches_list_topk(self):
        flat, query, rows, dim = self._data()
        matrix = [list(flat[i * dim:(i + 1) * dim]) for i in range(rows)]
        self.assertEqual(native.topk_flat(flat, dim, query, 6),
                         native.topk(matrix, query, 6))

    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_topk_flat_accepts_bytes_and_memoryview(self):
        flat, query, rows, dim = self._data()
        raw = flat.tobytes()
        want = native.topk_flat(flat, dim, query, 5)
        self.assertEqual(native.topk_flat(raw, dim, query, 5), want)
        self.assertEqual(native.topk_flat(memoryview(raw), dim, query, 5),
                         want)

    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_topk_flat_long_query_truncated(self):
        flat, query, rows, dim = self._data()
        long_query = query + [0.25, -0.5]
        self.assertEqual(native.topk_flat(flat, dim, long_query, 5),
                         native.topk_flat(flat, dim, query, 5))

    def test_topk_flat_bad_dim_raises(self):
        flat, query, _, dim = self._data()
        with self.assertRaises(ValueError):
            native.topk_flat(flat, 0, query, 5)
        with self.assertRaises(ValueError):
            native.topk_flat(flat, dim, query[: dim - 1], 5)

    def test_topk_flat_empty_and_nonpositive_k(self):
        flat, query, _, dim = self._data()
        self.assertEqual(native.topk_flat(b"", dim, query, 5), [])
        self.assertEqual(native.topk_flat(flat, dim, query, 0), [])
        self.assertEqual(native.topk_flat(flat, dim, query, -2), [])


class BuildTests(unittest.TestCase):
    def test_build_returns_bool_and_message(self):
        ok, msg = native.build()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(msg, str)
        self.assertTrue(msg)

    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_build_noop_when_already_built(self):
        ok, msg = native.build(force=False)
        self.assertTrue(ok)
        # all four targets report the fast path — nothing recompiled
        self.assertEqual(msg.count("already built"), 4)

    def test_find_compiler(self):
        compiler = native.find_compiler()
        # either a path or None (no compiler) — never raises
        self.assertTrue(compiler is None or isinstance(compiler, str))


class ThreadSafetyTests(unittest.TestCase):
    @unittest.skipUnless(_lib_present(), "vecsim kernel not built")
    def test_concurrent_load_and_topk(self):
        rng = random.Random(31)
        dim = 32
        matrix = [[rng.uniform(-1, 1) for _ in range(dim)]
                  for _ in range(50)]
        query = [rng.uniform(-1, 1) for _ in range(dim)]
        want = native.python_topk(matrix, query, 5)
        errors: list[BaseException] = []
        results: list = []

        def worker():
            try:
                for _ in range(10):
                    results.append(native.topk(matrix, query, 5))
                    native.load()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 80)
        for got in results:
            self.assertEqual([i for _, i in got], [i for _, i in want])


class BpeV2Tests(unittest.TestCase):
    @unittest.skipUnless(native.bpe_available(), "BPE kernel not built")
    def test_version_is_2(self):
        lib = native.load_bpe()
        self.assertEqual(lib.nm_bpe_version().decode(), "2.0.0")

    @unittest.skipUnless(native.bpe_available(), "BPE kernel not built")
    def test_train_matches_reference_randomized(self):
        rng = random.Random(20261010)
        vocab = ["hello", "world", "help", "held", "helmet", "hero", "her",
                 "hell", "low", "lower", "lowest", "new", "newer", "newest",
                 "wide", "wider", "shelf", "shell", "aaa", "aab", "tok",
                 "token", "tokens", "detokenize", "un", "undo", "redo"]
        for trial in range(12):
            texts = [" ".join(rng.choice(vocab)
                              for _ in range(rng.randint(5, 40)))
                     for _ in range(rng.randint(1, 4))]
            target = rng.randint(1, 20)
            min_freq = rng.randint(1, 3)
            c: Counter = Counter()
            for t in texts:
                for w in t.split():
                    c[w] += 1
            words, freqs = list(c.keys()), list(c.values())
            expected = _py_bpe_merges(words, freqs, target, min_freq)
            got = native.bpe_train(words, freqs, target_merges=target,
                                   min_frequency=min_freq)
            self.assertIsNotNone(got, f"trial {trial}: native must deliver")
            self.assertEqual(list(got), expected, f"trial {trial}")

    @unittest.skipUnless(native.bpe_available(), "BPE kernel not built")
    def test_train_matches_reference_unicode(self):
        # the old kernel split multi-byte characters into bytes and diverged
        # from the reference here; v2 must stay list-equal.
        rng = random.Random(4242)
        vocab = ["café", "naïve", "über", "hello", "world", "tok", "token",
                 "héllo", "wörld", "😀", "a😀b"]
        for trial in range(8):
            texts = [" ".join(rng.choice(vocab)
                              for _ in range(rng.randint(5, 30)))
                     for _ in range(rng.randint(1, 3))]
            target = rng.randint(1, 15)
            min_freq = rng.randint(1, 2)
            c: Counter = Counter()
            for t in texts:
                for w in t.split():
                    c[w] += 1
            words, freqs = list(c.keys()), list(c.values())
            expected = _py_bpe_merges(words, freqs, target, min_freq)
            got = native.bpe_train(words, freqs, target_merges=target,
                                   min_frequency=min_freq)
            self.assertIsNotNone(got, f"trial {trial}: native must deliver")
            self.assertEqual(list(got), expected, f"trial {trial}")

    @unittest.skipUnless(native.bpe_available(), "BPE kernel not built")
    def test_train_edge_cases(self):
        cases = [
            (["a"], [5], 3, 1),
            (["aaa", "aa", "a"], [4, 3, 2], 5, 1),
            (["hello"], [1], 0, 1),
            (["hello", "world"], [1, 1], 10, 99),
            (["ab", "ba"], [2, 2], 6, 1),
            (["x" * 50], [1], 10, 1),
        ]
        for words, freqs, target, min_freq in cases:
            expected = _py_bpe_merges(words, freqs, target, min_freq)
            got = native.bpe_train(words, freqs, target_merges=target,
                                   min_frequency=min_freq)
            self.assertIsNotNone(got)
            self.assertEqual(list(got), expected,
                             msg=f"words={words} target={target}")

    @unittest.skipUnless(native.bpe_available(), "BPE kernel not built")
    def test_train_never_repeats_a_pair(self):
        words = ["hello", "world", "hello", "help", "held", "helmet"]
        freqs = [10, 8, 10, 5, 3, 2]
        got = native.bpe_train(words, freqs, target_merges=8,
                               min_frequency=2)
        self.assertIsNotNone(got)
        self.assertEqual(len(got), len(set(got)), f"repeated merges: {got}")

    @unittest.skipUnless(native.bpe_available(), "BPE kernel not built")
    def test_encode_roundtrip_on_native_merges(self):
        words = ["hello", "world", "low", "lower", "café"]
        freqs = [10, 8, 6, 4, 5]
        merges = native.bpe_train(words, freqs, target_merges=8,
                                  min_frequency=2)
        self.assertIsNotNone(merges)
        stream = native.bpe_pack_stream(merges)
        for w in words + ["unseenword", "caféx"]:
            syms = native.bpe_encode_word(w, stream)
            self.assertIsNotNone(syms)
            self.assertEqual("".join(syms), w)


class NativeFallbackContractTests(unittest.TestCase):
    """New APIs degrade cleanly with no usable .so (same harness as the
    existing fallback suite)."""

    def setUp(self) -> None:
        self._tmp = Path(__import__("tempfile").mkdtemp(prefix="nm-sweep-"))
        self._old_here = native._HERE
        native._HERE = self._tmp
        self._saved = dict(native._LIBS)
        for slot in native._LIBS:
            native._LIBS[slot] = native._NOT_GIVEN
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        native._HERE = self._old_here
        native._LIBS.update(self._saved)
        for child in self._tmp.iterdir():
            child.unlink()
        self._tmp.rmdir()

    def test_simd_level_unknown_without_library(self):
        self.assertEqual(native.simd_level(), "unknown")
        self.assertEqual(native.info()["simd"], "unknown")

    def test_normalize_falls_back_to_python(self):
        self.assertEqual(native.normalize([3.0, 4.0]), [0.6, 0.8])

    def test_topk_flat_falls_back_to_python(self):
        rng = random.Random(5)
        dim, rows = 8, 20
        flat = array.array("f", (rng.uniform(-1, 1)
                                 for _ in range(rows * dim)))
        query = [rng.uniform(-1, 1) for _ in range(dim)]
        got = native.topk_flat(flat, dim, query, 4)
        rows_list = [list(flat[i * dim:(i + 1) * dim]) for i in range(rows)]
        want = native.python_topk(rows_list, query, 4)
        self.assertEqual([(round(s, 6), i) for s, i in got],
                         [(round(s, 6), i) for s, i in want])


class BiometricV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        bio_mod._reset_attempt_ledger()
        self.addCleanup(bio_mod._reset_attempt_ledger)

    def _run(self, **kw):
        run_kwargs = {"returncode": 0}
        run_kwargs.update(kw)
        fake = subprocess.CompletedProcess(args=["termux-fingerprint"],
                                           **run_kwargs)
        with patch("subprocess.run", return_value=fake):
            return bio_mod.request_biometric_ex("test title")

    def test_approved_result(self):
        res = self._run(returncode=0)
        self.assertEqual(res.status, bio_mod.BiometricStatus.APPROVED)
        self.assertTrue(res.approved)
        self.assertTrue(bool(res))

    def test_denied_result(self):
        res = self._run(returncode=1)
        self.assertEqual(res.status, bio_mod.BiometricStatus.DENIED)
        self.assertFalse(res.approved)
        self.assertIn("exited 1", res.reason)

    def test_timeout_result(self):
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired("x", 1)):
            res = bio_mod.request_biometric_ex("t", timeout_s=1)
        self.assertEqual(res.status, bio_mod.BiometricStatus.TIMEOUT)

    def test_missing_binary_is_unavailable(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("nope")):
            res = bio_mod.request_biometric_ex("t")
        self.assertEqual(res.status, bio_mod.BiometricStatus.UNAVAILABLE)

    def test_bool_contract_preserved(self):
        with patch("subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       args=["termux-fingerprint"], returncode=0)):
            self.assertTrue(bio_mod.request_biometric("t"))
        with patch("subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       args=["termux-fingerprint"], returncode=1)):
            self.assertFalse(bio_mod.request_biometric("t"))

    def test_five_denials_arm_lockout(self):
        with patch("subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       args=["termux-fingerprint"], returncode=1)):
            for _ in range(5):
                res = bio_mod.request_biometric_ex("t")
                self.assertEqual(res.status, bio_mod.BiometricStatus.DENIED)
            # the 6th prompt never reaches the binary
            with patch("subprocess.run") as m:
                res = bio_mod.request_biometric_ex("t")
        self.assertEqual(res.status, bio_mod.BiometricStatus.LOCKED_OUT)
        m.assert_not_called()
        self.assertFalse(bio_mod.request_biometric("t"))

    def test_available_reports_lockout(self):
        with patch("subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       args=["termux-fingerprint"], returncode=1)):
            for _ in range(5):
                bio_mod.request_biometric_ex("t")
        with patch("nomorals.core.profiles.get_profile_kind",
                   return_value="termux"), \
             patch("shutil.which", return_value="/usr/bin/termux-fingerprint"):
            ok, reason = bio_mod.biometric_available()
        self.assertFalse(ok)
        self.assertIn("locked out", reason)

    def test_lockout_expires(self):
        with patch.object(bio_mod, "_LOCKOUT_S", 0.15), \
             patch("subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       args=["termux-fingerprint"], returncode=1)):
            for _ in range(5):
                bio_mod.request_biometric_ex("t")
            res = bio_mod.request_biometric_ex("t")
            self.assertEqual(res.status, bio_mod.BiometricStatus.LOCKED_OUT)
        import time as _time
        _time.sleep(0.2)
        with patch("subprocess.run",
                   return_value=subprocess.CompletedProcess(
                       args=["termux-fingerprint"], returncode=0)):
            res = bio_mod.request_biometric_ex("t")
        self.assertEqual(res.status, bio_mod.BiometricStatus.APPROVED)

    def test_approval_resets_denial_count(self):
        calls = {"n": 0}

        def fake_run(*a, **k):
            calls["n"] += 1
            # 4 denials, 1 approval, then denials again — never 5 consecutive
            rc = 0 if calls["n"] == 5 else 1
            return subprocess.CompletedProcess(args=["termux-fingerprint"],
                                               returncode=rc)

        with patch("subprocess.run", side_effect=fake_run):
            for _ in range(9):
                res = bio_mod.request_biometric_ex("t")
                self.assertNotEqual(res.status,
                                    bio_mod.BiometricStatus.LOCKED_OUT)

    def test_busy_when_prompt_in_flight(self):
        self.assertTrue(bio_mod._prompt_lock.acquire(blocking=False))
        try:
            with patch("subprocess.run") as m:
                res = bio_mod.request_biometric_ex("t")
            self.assertEqual(res.status, bio_mod.BiometricStatus.BUSY)
            m.assert_not_called()
            self.assertFalse(bio_mod.request_biometric("t"))
        finally:
            bio_mod._prompt_lock.release()

    def test_result_bool_semantics(self):
        approved = bio_mod.BiometricResult(bio_mod.BiometricStatus.APPROVED)
        denied = bio_mod.BiometricResult(bio_mod.BiometricStatus.DENIED,
                                         "nope")
        self.assertTrue(approved.approved and bool(approved))
        self.assertFalse(denied.approved or bool(denied))
        self.assertEqual(denied.reason, "nope")


if __name__ == "__main__":
    unittest.main()
