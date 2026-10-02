"""H1 — the pure-Python fallback contract for nomorals.native.

The committed x86-64 .so blobs were dropped from the repo; this pins the
behavior that matters: with no usable native library (missing, corrupt, or
wrong-arch), every public function degrades cleanly to pure Python and
``*_available()`` returns False.  Corrupt libraries must be swallowed as
OSError, never crash.
"""

from __future__ import annotations

import array
import random
import unittest
from pathlib import Path

from nomorals import native


class NativeFallbackTest(unittest.TestCase):
    """Simulate "no usable .so" by pointing _HERE at an empty temp dir."""

    def setUp(self) -> None:
        self._tmp = Path(__import__("tempfile").mkdtemp(prefix="nm-native-"))
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

    def _write_bad_lib(self, stem: str, payload: bytes) -> None:
        (self._tmp / f"{stem}.so").write_bytes(payload)

    # ── absent libraries ─────────────────────────────────────────────

    def test_all_available_false_when_missing(self) -> None:
        self.assertFalse(native.available())
        self.assertFalse(native.mlp_available())
        self.assertFalse(native.bpe_available())
        self.assertFalse(native.mem_available())

    def test_info_reports_pure_python(self) -> None:
        info = native.info()
        self.assertEqual(info["backend"], "pure-python")
        self.assertFalse(info["built"])
        self.assertFalse(info["loaded"])
        for key in ("mlp", "bpe", "mem"):
            self.assertEqual(info[key]["backend"], "pure-python")
            self.assertFalse(info[key]["built"])

    def test_topk_matches_reference(self) -> None:
        rng = random.Random(42)
        matrix = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(60)]
        query = [rng.uniform(-1, 1) for _ in range(16)]
        got = native.topk(matrix, query, 8)
        # the fallback truncates the query to float32; mirror that exactly
        q32 = list(array.array("f", (float(v) for v in query)))
        want = native.python_topk(matrix, q32, 8)
        self.assertEqual([(round(s, 6), i) for s, i in got],
                         [(round(s, 6), i) for s, i in want])

    def test_topk_packed_matches_reference(self) -> None:
        rng = random.Random(9)
        dim, rows = 8, 40
        flat = array.array("f", (rng.uniform(-1, 1) for _ in range(rows * dim)))
        query = [rng.uniform(-1, 1) for _ in range(dim)]
        got = native.topk(flat, query, 5)
        rows_list = [list(flat[i * dim:(i + 1) * dim]) for i in range(rows)]
        want = native.python_topk(rows_list, query, 5)
        self.assertEqual([(round(s, 6), i) for s, i in got],
                         [(round(s, 6), i) for s, i in want])

    def test_dot_matches_reference(self) -> None:
        a = array.array("f", [0.25, -1.5, 3.0])
        b = array.array("f", [2.0, 0.5, -1.0])
        self.assertAlmostEqual(native.dot(a, b), native.python_dot(a, b))

    def test_kernel_absent_contracts_return_none(self) -> None:
        self.assertIsNone(native.bpe_train(["hi"], [1], target_merges=2,
                                           min_frequency=1))
        self.assertIsNone(native.bpe_encode_word("hi", array.array("i", [7])))
        self.assertIsNone(native.mem_heuristic("I prefer tea"))

    def test_mlp_batch_raises_with_build_hint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "nm native --build"):
            native.mlp_batch(array.array("i", [1]), array.array("i", [1]),
                             array.array("d", [0.1]), array.array("d", [0.1]),
                             array.array("d", [0.0]), array.array("d", [0.1]),
                             array.array("d", [0.0]), context=1)

    # ── corrupt / wrong-arch libraries ─────────────────────────────

    def test_garbage_file_falls_back(self) -> None:
        self._write_bad_lib("libvecsim", b"this is not an ELF file")
        self.assertIsNone(native.load())
        self.assertFalse(native.available())
        m = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(native.topk(m, [1.0, 0.0], 1)[0][1], 0)

    def test_truncated_elf_header_falls_back(self) -> None:
        self._write_bad_lib("libvecsim", b"\x7fELF" + b"\x00" * 8)
        self.assertIsNone(native.load())
        self.assertFalse(native.available())

    def test_corrupt_mlp_and_bpe_do_not_crash(self) -> None:
        self._write_bad_lib("libmlptrain", b"garbage")
        self._write_bad_lib("libbpe", b"garbage")
        self.assertIsNone(native.load_mlp())
        self.assertIsNone(native.load_bpe())
        self.assertFalse(native.mlp_available())
        self.assertFalse(native.bpe_available())
        self.assertIsNone(native.bpe_train(["hi"], [1], target_merges=2,
                                           min_frequency=1))

    def test_corrupt_mem_kernel_falls_back(self) -> None:
        self._write_bad_lib("libmemextract", b"garbage")
        self.assertIsNone(native.load_mem())
        self.assertFalse(native.mem_available())
        self.assertIsNone(native.mem_heuristic("hello"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class NativeDoctorTest(unittest.TestCase):
    """nm doctor's native section: statuses and honest rendering."""

    def _info(self, *, built: bool, loaded: bool) -> dict:
        block = {"built": built, "loaded": loaded, "backend": "x"}
        return {"built": built, "loaded": loaded, "compiler": "/usr/bin/c++",
                "backend": "x", "mlp": block, "bpe": block, "mem": block}

    def test_all_loaded_is_native(self) -> None:
        from nomorals.compat import native_section
        sec = native_section(self._info(built=True, loaded=True))
        self.assertEqual(sec["overall"], "native")
        self.assertTrue(all(s == "native" for s in sec["kernels"].values()))

    def test_built_but_unloadable_is_stale_arch(self) -> None:
        from nomorals.compat import native_section
        sec = native_section(self._info(built=True, loaded=False))
        self.assertTrue(all(s == "stale-arch" for s in sec["kernels"].values()))
        self.assertEqual(sec["overall"], "pure-python")

    def test_nothing_built_is_missing(self) -> None:
        from nomorals.compat import native_section
        sec = native_section(self._info(built=False, loaded=False))
        self.assertTrue(all(s == "missing" for s in sec["kernels"].values()))
        self.assertEqual(sec["fallback"], "pure-python")

    def test_partial_mix(self) -> None:
        from nomorals.compat import native_section
        info = self._info(built=False, loaded=False)
        info["mlp"] = {"built": True, "loaded": True, "backend": "native-cpp"}
        sec = native_section(info)
        self.assertEqual(sec["kernels"]["mlptrain"], "native")
        self.assertEqual(sec["kernels"]["vecsim"], "missing")
        self.assertEqual(sec["overall"], "partial")

    def test_report_renders_section_and_build_hint(self) -> None:
        from nomorals.compat import feature_report, native_section, report_as_text
        report = feature_report()
        report["native"] = native_section(self._info(built=False, loaded=False))
        text = report_as_text(report)
        self.assertIn("native accelerators:", text)
        self.assertIn("missing", text)
        self.assertIn("nm native --build", text)

    def test_report_omits_section_when_native_unknown(self) -> None:
        from nomorals.compat import feature_report, report_as_text
        report = feature_report()
        self.assertIsNone(report["native"])
        self.assertNotIn("native accelerators:", report_as_text(report))
