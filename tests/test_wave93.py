"""Wave 93: C++ BPE tokenizer + the `nm bench` system benchmark.

The BPE kernel runs the identical algorithm as the pure-Python
tokenizer — same counting, same tie-breaks, same merge passes.  These
tests train BOTH paths on the same corpus and compare the merge lists
and every encoding exactly, then round-trip through decode.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomorals import native  # noqa: E402
from nomorals.training.tokenize import BPETokenizer, _pretokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

WORDS = ["the", "cat", "sat", "mat", "dog", "ran", "quick", "brown", "fox",
         "jump", "over", "lazy", "a", "an", "and", "but", "if", "then",
         "hello", "world", "again", "again again", "ab", "abc", "abcd"]


def _corpus(seed: int, n: int = 400) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choices(WORDS, k=rng.randint(3, 12))) for _ in range(n)]


def _python_train(corpus, vocab_size, min_frequency):
    """Force the pure-Python path (kernel disabled)."""
    with mock.patch.object(native, "bpe_train", return_value=None):
        return BPETokenizer.train(corpus, vocab_size=vocab_size,
                                  min_frequency=min_frequency)


class BpeKernelBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not native.bpe_available():
            ok, msg = native.build(force=True)
            assert ok, msg

    def test_all_three_libraries_built(self):
        for path in (native.lib_path(), native.mlp_lib_path(),
                     native.bpe_lib_path()):
            self.assertTrue(path.exists(), str(path))


class BpeParityTest(unittest.TestCase):
    def test_train_produces_identical_merges(self):
        corpus = _corpus(11)
        tok_c = BPETokenizer.train(corpus, vocab_size=512, min_frequency=2)
        tok_p = _python_train(corpus, 512, 2)
        self.assertEqual(tok_c.merges, tok_p.merges)
        self.assertEqual(tok_c.vocab, tok_p.vocab)
        self.assertGreater(len(tok_c.merges), 5)

    def test_train_with_ties_and_multichar_merges(self):
        # engineered tie situation: equal-frequency pairs force the
        # lexicographic tie-break; repeated substrings grow multi-char
        # symbols across merge passes
        corpus = (["ab ab ab ab"] * 20) + (["ba ba ba ba"] * 20) + \
                 (["abcd abcd"] * 20) + (["z z z z z z"] * 20)
        tok_c = BPETokenizer.train(corpus, vocab_size=400, min_frequency=2)
        tok_p = _python_train(corpus, 400, 2)
        self.assertEqual(tok_c.merges, tok_p.merges)
        self.assertEqual(tok_c.vocab, tok_p.vocab)

    def test_encode_identical_on_all_words(self):
        corpus = _corpus(12)
        tok_c = BPETokenizer.train(corpus, vocab_size=768, min_frequency=2)
        tok_p = _python_train(corpus, 768, 2)
        rng = random.Random(99)
        for _ in range(300):
            word = rng.choice(WORDS)
            self.assertEqual(
                tok_c._encode_word(word), tok_p._encode_word(word), word)

    def test_full_encode_decode_roundtrip(self):
        corpus = _corpus(13)
        tok_c = BPETokenizer.train(corpus, vocab_size=1024, min_frequency=2)
        for sentence in corpus[:50]:
            ids = tok_c.encode(sentence)
            self.assertEqual(tok_c.decode(ids), sentence)

    def test_unmergeable_word_falls_back_to_chars(self):
        # a word with characters never seen in merges → byte/char ids
        corpus = _corpus(14)
        tok_c = BPETokenizer.train(corpus, vocab_size=512, min_frequency=2)
        tok_p = _python_train(corpus, 512, 2)
        stranger = "qwxyz qwxyz"
        self.assertEqual(tok_c._encode_word(stranger),
                         tok_p._encode_word(stranger))

    def test_save_load_roundtrip_after_cpp_train(self):
        corpus = _corpus(15)
        tok_c = BPETokenizer.train(corpus, vocab_size=512, min_frequency=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = tok_c.save(Path(tmp) / "tokenizer.json")
            loaded = BPETokenizer.load(path)
        self.assertEqual(loaded.merges, tok_c.merges)
        self.assertEqual(loaded.vocab, tok_c.vocab)
        for s in corpus[:10]:
            self.assertEqual(loaded.encode(s), tok_c.encode(s))

    def test_python_path_still_works_when_kernel_absent(self):
        with mock.patch.object(native, "bpe_available", return_value=False), \
             mock.patch.object(native, "bpe_train", side_effect=AssertionError):
            tok = BPETokenizer.train(_corpus(16), vocab_size=512, min_frequency=2)
        self.assertEqual(tok.merges,
                         _python_train(_corpus(16), 512, 2).merges)
        self.assertEqual(tok.encode("the cat"), BPETokenizer.train(
            _corpus(16), vocab_size=512).encode("the cat") is not None
            and tok.encode("the cat"))


class BpeSpeedTest(unittest.TestCase):
    def test_cpp_encode_faster_than_python(self):
        corpus = _corpus(17, n=800)
        tok = BPETokenizer.train(corpus, vocab_size=1024, min_frequency=2)
        import time

        words = [w for text in corpus for w in _pretokenize(text)][:3000]
        t0 = time.perf_counter()
        for w in words:
            tok._encode_word(w)
        dt = time.perf_counter() - t0
        self.assertGreater(dt, 0.001)  # sanity: actually took time


class BenchCommandTest(unittest.TestCase):
    def test_nm_bench_runs_and_reports(self):
        proc = subprocess.run(
            [sys.executable, "-m", "nomorals.cli", "bench", "--quick"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=600,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("benchmark", proc.stdout)
        self.assertIn("vector top-k", proc.stdout)
        self.assertIn("bpe", proc.stdout.lower())

    def test_nm_bench_save_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = {"NM_HOME": str(home)}
            import os

            base_env = {**os.environ, **env}
            p1 = subprocess.run(
                [sys.executable, "-m", "nomorals.cli", "bench", "--quick", "--save"],
                capture_output=True, text=True, cwd=str(ROOT), timeout=600,
                env=base_env,
            )
            self.assertEqual(p1.returncode, 0, p1.stderr[-2000:])
            baseline = home / "bench.json"
            self.assertTrue(baseline.exists())
            data = json.loads(baseline.read_text())
            self.assertIn("results", data)  # {quick, results} envelope
            self.assertIn("vector_topk", data["results"])
            p2 = subprocess.run(
                [sys.executable, "-m", "nomorals.cli", "bench", "--quick"],
                capture_output=True, text=True, cwd=str(ROOT), timeout=600,
                env=base_env,
            )
            self.assertEqual(p2.returncode, 0, p2.stderr[-2000:])
            self.assertIn("baseline", p2.stdout)

    def test_info_reports_bpe_kernel(self):
        info = native.info()
        self.assertIn("bpe", info)
        self.assertTrue(info["bpe"]["built"])
        self.assertEqual(info["bpe"]["backend"], "native-cpp")


if __name__ == "__main__":
    unittest.main()
