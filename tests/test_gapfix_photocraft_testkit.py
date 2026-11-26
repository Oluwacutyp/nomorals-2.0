"""PhotoCraft testkit — lane 117: the shared test support lands.

The testkit (``tests/photo_testkit.py``) ports upstream's
``crates/testkit`` habit: one module every photo test imports
(Sweep, oracle helpers, corpus paths, ``require_corpus``),
instead of each file re-defining them. This file proves the
testkit itself, drives the gated full-corpus oracle over the
258-file mining-checkout corpus sliced by ``Sweep``, and proves
the photo NL hook fires from plain text (it used to live inside
``_handle_control_inner``, where plain text never reaches — and
it truncated ``/photo edit <id> <op> ...`` to ``edit <id>``
before the dispatch ever saw the op).
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.photo_testkit import (
    CORPUS,
    FIXTURES,
    Sweep,
    default_full_corpus_dir,
    flatten_vs_composite,
    full_corpus_files,
    max_diff,
    read_bytes,
    require_corpus,
)


class SweepTests(unittest.TestCase):
    def test_parts_cover_every_case_once(self) -> None:
        # Port of upstream testkit's the_parts_cover_every_case_once.
        parts = [Sweep.__new__(Sweep) for _ in range(4)]
        for i, sweep in enumerate(parts):
            sweep.part = i
            sweep.parts = 4
        for i in range(1000):
            hits = [s for s in parts if s.includes(i)]
            self.assertEqual(len(hits), 1, f"case {i}")
        for sweep in parts:
            count = sum(1 for i in range(1000)
                        if sweep.includes(i))
            self.assertGreater(count, 200)

    def test_full_sweep_env_runs_every_case(self) -> None:
        with mock.patch.dict(os.environ,
                             {"NM_PHOTOCRAFT_FULL_SWEEP": "1"},
                             clear=False):
            sweep = Sweep(7)
        self.assertEqual(sweep.parts, 1)
        for i in range(100):
            self.assertTrue(sweep.includes(i))

    def test_explicit_part_replays_that_slice(self) -> None:
        with mock.patch.dict(os.environ,
                             {"NM_PHOTOCRAFT_SWEEP_PART": "3"},
                             clear=False):
            sweep = Sweep(7)
        self.assertEqual((sweep.part, sweep.parts), (3, 7))
        self.assertTrue(any(sweep.includes(i)
                            for i in range(100)))
        self.assertFalse(all(sweep.includes(i)
                             for i in range(100)))

    def test_invalid_part_is_an_error(self) -> None:
        with mock.patch.dict(os.environ,
                             {"NM_PHOTOCRAFT_SWEEP_PART": "7"},
                             clear=False), self.assertRaises(ValueError):
            Sweep(7)

    def test_default_slice_is_one_of_seven(self) -> None:
        env = {k: v for k, v in os.environ.items()
               if k not in ("NM_PHOTOCRAFT_FULL_SWEEP",
                            "NM_PHOTOCRAFT_SWEEP_PART")}
        with mock.patch.dict(os.environ, env, clear=True):
            sweep = Sweep()
        self.assertEqual(sweep.parts, 7)
        self.assertIn(sweep.part, range(7))
        self.assertTrue(sweep.describe().startswith("sweep part "))


class HelperTests(unittest.TestCase):
    def test_max_diff_identical_is_zero(self) -> None:
        from PIL import Image

        image = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
        self.assertEqual(max_diff(image, image), 0)

    def test_max_diff_shape_mismatch_raises(self) -> None:
        from PIL import Image

        small = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
        big = Image.new("RGBA", (8, 8), (0, 0, 0, 255))
        with self.assertRaises(AssertionError):
            max_diff(small, big)

    def test_flatten_vs_composite_on_known_fixture(self) -> None:
        require_corpus([CORPUS / "adjustments" / "rgb8"
                        / "gradient-map.psd"])
        self.assertLessEqual(
            flatten_vs_composite(
                CORPUS / "adjustments" / "rgb8"
                / "gradient-map.psd"), 1)

    def test_require_corpus_fails_naming_the_file(self) -> None:
        missing = FIXTURES / "no-such-corpus-file.psd"
        with self.assertRaises(AssertionError) as caught:
            require_corpus([missing])
        self.assertIn("no-such-corpus-file.psd",
                      str(caught.exception))


class FullCorpusOracleTests(unittest.TestCase):
    """The extended oracle: the full 258-file corpus, sliced by
    Sweep so every run covers its day's slice and the whole
    corpus is covered every seven days (full with
    NM_PHOTOCRAFT_FULL_SWEEP=1). A requested run whose corpus
    is missing FAILS naming the path; an unrequested run with
    no corpus on the box skips with the reason — the vendored
    50 always run, this is the extended run, like upstream's
    opt-in corpus feature."""

    def test_full_corpus_slice(self) -> None:
        corpus_dir = Path(default_full_corpus_dir())
        full_requested = (
            os.environ.get("NM_PHOTOCRAFT_FULL_SWEEP") == "1"
            or "NM_PHOTOCRAFT_SWEEP_PART" in os.environ)
        if full_requested and not corpus_dir.is_dir():
            self.fail(
                "full corpus requested but missing: "
                f"{corpus_dir} — set NM_PHOTOCRAFT_CORPUS to the "
                "photocraft-corpus checkout or fetch it")
        files = full_corpus_files(corpus_dir)
        if not files:
            self.skipTest(
                "full 258-file corpus not on this box "
                f"({corpus_dir}) — the vendored 50 cover the suite; "
                "set NM_PHOTOCRAFT_CORPUS to run the extended oracle")
        sweep = Sweep()
        sliced = [f for i, f in enumerate(files)
                  if sweep.includes(i)]
        self.assertGreater(len(sliced), 0)
        if sweep.parts > 1:
            self.assertLess(len(sliced), len(files))
        from nomorals.media.photo_editor.oracle import run_oracle

        start = time.perf_counter()
        report = run_oracle(
            str(corpus_dir),
            paths=[str(f) for f in sliced])
        elapsed = time.perf_counter() - start
        summary = report["summary"]
        print(f"\nfull-corpus oracle: {sweep.describe()}; "
              f"{len(sliced)}/{len(files)} files in {elapsed:.1f}s "
              f"({elapsed / max(len(sliced), 1):.2f}s/file); "
              f"exact {summary['exact']}, "
              f"byte-stable {summary['byte_stable']}, "
              f"doc byte-stable {summary['doc_byte_stable']}, "
              f"round-trip ok {summary['roundtrip_ok']}")
        self.assertEqual(summary["files"], len(sliced))
        for entry in report["files"]:
            with self.subTest(file=entry["name"]):
                self.assertNotEqual(entry["status"], "failed",
                                    entry)
                self.assertTrue(entry["byte_stable"], entry)
                self.assertTrue(entry["doc_byte_stable"], entry)
                self.assertTrue(
                    entry["roundtrip_ok"]
                    or entry["roundtrip_depth_reduced"], entry)


class PhotoNlHookTests(unittest.TestCase):
    """The photo NL hook fires from plain text.

    The hook used to live inside ``_handle_control_inner``,
    where plain text never reaches — so "open this photo ..."
    did nothing in chat — and it intercepted slash
    ``/photo edit <id> <op> ...`` first, truncating it to
    ``edit <id>`` before the dispatch ever saw the op. It now
    lives in ``_process_inner`` (``_photo_surface_hook``), next
    to the film hook; slash text goes through the slash dispatch
    untouched."""

    def _runtime(self, tmp: str):
        from nomorals.agents.partner.runtime import PartnerRuntime

        runtime = object.__new__(PartnerRuntime)
        runtime.context = SimpleNamespace(
            settings=SimpleNamespace(home=tmp))
        return runtime

    def _message(self, text: str):
        return SimpleNamespace(text=text, chat=SimpleNamespace())

    def test_open_by_path_fires_from_plain_text(self) -> None:
        fixture = CORPUS / "adjustments" / "rgb8" \
            / "gradient-map.psd"
        require_corpus([fixture])
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime._is_operator = lambda message: True
            calls: list[str] = []
            runtime._control_photo = (
                lambda tail, message=None: calls.append(tail)
                or "photo opened")
            reply = runtime._photo_surface_hook(
                self._message(f"open this photo {fixture}"))
            self.assertEqual(reply, "photo opened")
            self.assertEqual(calls, [f"open {fixture}"])

    def test_show_by_title_fires_from_plain_text(self) -> None:
        from nomorals.media.photo_editor.document import PhotoDocument
        from nomorals.media.photo_editor.store import PhotoStore

        fixture = CORPUS / "adjustments" / "rgb8" \
            / "gradient-map.psd"
        require_corpus([fixture])
        with tempfile.TemporaryDirectory() as tmp:
            doc = PhotoDocument.from_psd_bytes(read_bytes(fixture))
            store = PhotoStore(home=tmp)
            doc_id = store.save(doc, title="gradient")
            runtime = self._runtime(tmp)
            runtime._is_operator = lambda message: True
            calls: list[str] = []
            runtime._control_photo = (
                lambda tail, message=None: calls.append(tail)
                or "photo shown")
            reply = runtime._photo_surface_hook(
                self._message("show this photo gradient"))
            self.assertEqual(reply, "photo shown")
            self.assertEqual(calls, [f"show {doc_id}"])

    def test_slash_text_is_not_eaten_by_the_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime._is_operator = lambda message: True
            runtime._control_photo = (
                lambda tail, message=None: "should not fire")
            self.assertIsNone(runtime._photo_surface_hook(
                self._message("/photo open /tmp/x.psd")))
            self.assertIsNone(runtime._photo_surface_hook(
                self._message("/photo edit abc123 crop 0 0 10 10")))

    def test_unrelated_text_and_non_operator_do_not_fire(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            runtime._is_operator = lambda message: True
            runtime._control_photo = (
                lambda tail, message=None: "should not fire")
            self.assertIsNone(runtime._photo_surface_hook(
                self._message("hello, how are you today?")))
            self.assertIsNone(runtime._photo_surface_hook(
                self._message("open this photo")))
            runtime._is_operator = lambda message: False
            fixture = CORPUS / "adjustments" / "rgb8" \
                / "gradient-map.psd"
            self.assertIsNone(runtime._photo_surface_hook(
                self._message(f"open this photo {fixture}")))

    def test_slash_edit_reaches_control_whole(self) -> None:
        """The truncation the old placement caused is gone:
        ``/photo edit <id> crop ...`` reaches ``_control_photo``
        with its full tail, not as a bare ``edit <id>``."""
        from nomorals.core.policy import CapabilitySet
        from nomorals.tools.registry import ToolRegistry

        fixture = CORPUS / "adjustments" / "rgb8" \
            / "gradient-map.psd"
        require_corpus([fixture])
        with tempfile.TemporaryDirectory() as tmp:
            context = SimpleNamespace(
                settings=SimpleNamespace(home=tmp))
            registry = ToolRegistry(context)
            registry.register_builtins()
            opened = registry.call(
                "photo_editor", actor="test",
                capabilities=CapabilitySet.all(),
                action="open", path=str(fixture))
            self.assertTrue(opened.ok, opened)
            doc_id = opened.unwrap()["id"]
            from nomorals.agents.partner.runtime import PartnerRuntime

            runtime = object.__new__(PartnerRuntime)
            runtime.context = SimpleNamespace(
                settings=SimpleNamespace(home=tmp),
                tools=registry)
            reply = runtime._control_photo(
                f"edit {doc_id} crop 0 0 10 10", None)
            self.assertNotIn("usage:", reply)
            self.assertIn("photo ", reply)


if __name__ == "__main__":
    unittest.main()
