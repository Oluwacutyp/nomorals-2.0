"""The hash cracker: offline verification of candidate plaintexts against
hash strings (md5 · sha1 · sha256 · sha512 · ntlm).

The MD4 implementation is pinned against the RFC 1320 test vectors and the
known NTLM hash of "password", so the ntlm path is verifiable without any
external crypto library.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

from nomorals.tools.hashcrack import (
    ALGORITHMS,
    Engine,
    MarkovChain,
    auto_detect,
    benchmark,
    crack_hash,
    get_hash_fn,
    md4,
    mutate,
)


def _wordlist(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class Md4VectorTest(unittest.TestCase):
    """RFC 1320 test vectors, plus multi-block and padding edge cases."""

    def test_rfc_vectors(self) -> None:
        self.assertEqual(md4(b""), "31d6cfe0d16ae931b73c59d7e0c089c0")
        self.assertEqual(md4(b"abc"), "a448017aaf21d8525fc10ae87aa6729d")
        self.assertEqual(md4(b"message digest"), "d9130a8164549fe818874806e1c7014b")
        self.assertEqual(
            md4(b"The quick brown fox jumps over the lazy dog"),
            "1bee69a46ba811185c194762abaeae90",  # cross-checked vs PyCryptodome
        )

    def test_multi_block_and_padding_edges(self) -> None:
        # lengths that exercise every padding branch (block-boundary cases)
        for n in (55, 56, 63, 64, 65, 127, 128, 129):
            digest = md4(b"a" * n)
            self.assertEqual(len(digest), 32)
        # cross-checked against PyCryptodome (multi-block + boundary padding)
        self.assertEqual(md4(b"a" * 64), "52f5076fabd22680234a3fa9f9dc5732")
        self.assertEqual(md4(b"a" * 128), "cb4a20a561558e29460190c91dced59f")

    def test_ntlm_of_password_is_the_well_known_hash(self) -> None:
        # classic NTLM digest of "password" — independent of the md4 impl
        self.assertEqual(ALGORITHMS["ntlm"]("password"), "8846f7eaee8fb117ad06bdd830b7586c")

    def test_ntlm_is_md4_of_utf16le(self) -> None:
        for pw in ("", "password", "P@ssw0rd!"):
            self.assertEqual(ALGORITHMS["ntlm"](pw), md4(pw.encode("utf-16-le")))


class AlgorithmTest(unittest.TestCase):
    def test_auto_detect_by_length(self) -> None:
        self.assertEqual(auto_detect("a" * 32), "md5")
        self.assertEqual(auto_detect("a" * 40), "sha1")
        self.assertEqual(auto_detect("a" * 64), "sha256")
        self.assertEqual(auto_detect("a" * 128), "sha512")
        self.assertIsNone(auto_detect("a" * 16))
        self.assertEqual(auto_detect("  " + "b" * 32), "md5")

    def test_get_hash_fn_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_hash_fn("rot13")


class MutateTest(unittest.TestCase):
    def test_variants_and_dedup(self) -> None:
        out = mutate("password")
        for expected in ("Password", "PASSWORD", "drowssap", "p@$$w0rd", "password1", "password123", "1password", "passwordpassword"):
            self.assertIn(expected, out)
        self.assertEqual(len(out), len(set(out)))  # no duplicates


class MarkovTest(unittest.TestCase):
    def test_generates_unique_in_range(self) -> None:
        words = MarkovChain().train_default().generate(300, min_len=4, max_len=8)
        self.assertGreater(len(words), 100)
        self.assertEqual(len(words), len(set(words)))
        for w in words:
            self.assertTrue(4 <= len(w) <= 8, w)


class CrackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp: list[str] = []

    def tearDown(self) -> None:
        for p in self._tmp:
            os.unlink(p)

    def wl(self, content: str) -> str:
        p = _wordlist(content)
        self._tmp.append(p)
        return p

    def test_wordlist_hit_stops_early(self) -> None:
        wl = self.wl("letmein\nqwerty\npassword\nadmin\n")
        r = crack_hash(hashlib.md5(b"password").hexdigest(), mode="hybrid", wordlist=wl, threads=2)
        self.assertEqual(r.found, {hashlib.md5(b"password").hexdigest(): "password"})
        self.assertEqual(r.tested, 3)  # letmein, qwerty, password — then stop
        self.assertEqual(r.remaining, [])
        self.assertEqual(r.algo, "md5")

    def test_mutation_rule_crack(self) -> None:
        # target is the "Password1" mutation of the wordlist word "password"
        wl = self.wl("password\n")
        r = crack_hash(hashlib.md5(b"Password1").hexdigest(), mode="hybrid", wordlist=wl, threads=2)
        self.assertEqual(r.found, {hashlib.md5(b"Password1").hexdigest(): "Password1"})

    def test_brute_force_crack(self) -> None:
        r = crack_hash(
            hashlib.sha1(b"abc").hexdigest(),
            mode="brute",
            charset="abc",
            min_len=1,
            max_len=3,
            threads=2,
        )
        self.assertEqual(r.found, {hashlib.sha1(b"abc").hexdigest(): "abc"})

    def test_miss_reports_incomplete(self) -> None:
        wl = self.wl("alpha\nbeta\n")
        r = crack_hash(hashlib.sha256(b"gamma").hexdigest(), mode="wordlist", wordlist=wl, threads=2)
        self.assertEqual(r.found, {})
        self.assertEqual(r.remaining, [hashlib.sha256(b"gamma").hexdigest()])

    def test_max_candidates_caps_the_run(self) -> None:
        r = crack_hash(
            "a" * 64,
            mode="brute",
            charset="ab",
            min_len=1,
            max_len=2,
            max_candidates=3,
            threads=2,
        )
        self.assertEqual(r.tested, 3)
        self.assertTrue(r.capped)
        self.assertEqual(r.found, {})

    def test_multiple_targets(self) -> None:
        wl = self.wl("password\nletmein\n")
        h1, h2 = hashlib.md5(b"password").hexdigest(), hashlib.md5(b"letmein").hexdigest()
        engine = Engine([h1, h2], mode="wordlist", wordlist=wl, threads=2, quiet=True)
        result = engine.run()
        self.assertEqual(sorted(result.found.values()), ["letmein", "password"])
        self.assertEqual(result.cracked, 2)


class CheckpointTest(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(path)
        try:
            targets = ["a" * 32, "b" * 32]
            engine = Engine(targets, algo="md5", checkpoint=path, quiet=True)
            engine.found = {targets[0]: "x"}
            engine.remaining.discard(targets[0])
            engine._tested = 42
            engine._save_checkpoint()
            self.assertTrue(os.path.exists(path))

            resumed = Engine(targets, algo="md5", checkpoint=path, quiet=True)
            self.assertEqual(resumed.found, {targets[0]: "x"})
            self.assertEqual(resumed.remaining, {targets[1]})
            self.assertEqual(resumed._tested, 42)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_returns_one_row_per_algorithm(self) -> None:
        rows = benchmark(secs=0.1)
        self.assertEqual({name for name, _, _ in rows}, set(ALGORITHMS))
        for name, hps, ops in rows:
            self.assertGreater(hps, 0, name)
            self.assertGreater(ops, 0, name)


class RegistryTest(unittest.TestCase):
    def test_tool_is_registered_and_callable(self) -> None:
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register_builtins()
        self.assertIn("hash_crack", registry.names())

        wl = _wordlist("password\n")
        try:
            outcome = registry.call(
                "hash_crack",
                target=hashlib.md5(b"password").hexdigest(),
                mode="wordlist",
                wordlist=wl,
            )
            self.assertTrue(outcome.ok)
            payload = outcome.unwrap()
            self.assertEqual(payload["cracked"], 1)
            self.assertEqual(payload["found"], {hashlib.md5(b"password").hexdigest(): "password"})
            self.assertGreaterEqual(len(registry.calls), 1)  # audited
        finally:
            os.unlink(wl)


if __name__ == "__main__":
    unittest.main()
