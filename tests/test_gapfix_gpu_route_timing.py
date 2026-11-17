"""GPU-route timing — local render durations + CPU-slow trigger.

Every successful local render records how long the pipeline took,
and when the local median exceeds SLOW_LOCAL_P50_S the next
single-image job routes to the attested forwarded GPU instead of
making the owner wait on a slow CPU — the same branch the
not-ready case already takes. The timing module is imported lazily
inside the tests so the suite fails per-test (not at collection)
on a base without it.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from PIL import Image

from nomorals.media.imggen import ImgGenError
from nomorals.media.imggen import remote as remote_mod
from nomorals.media.imggen.studio import Studio

ENDPOINT_URL = "http://127.0.0.1:9999"


def _timing():
    from nomorals.media.imggen import timing

    return timing


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (1, 1), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakePipe:
    last_warnings: list[str] = []

    def generate(self, prompt: str, cfg: Any) -> list[Any]:
        return [Image.new("RGB", (2, 2), (0, 255, 0))]


class _FailingPipe:
    last_warnings: list[str] = []

    def generate(self, prompt: str, cfg: Any) -> list[Any]:
        raise ImgGenError("fake pipeline failure")


class TimingStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.path = os.path.join(self.tmp, "render_timings.json")

    def _store(self, name: str = "render_timings.json"):
        return _timing().RenderTimingStore(
            os.path.join(self.tmp, name))

    def test_empty_store_has_no_median(self) -> None:
        store = self._store()
        self.assertEqual(store.count, 0)
        self.assertEqual(store.durations(), [])
        self.assertIsNone(store.p50())

    def test_p50_over_seeded_history(self) -> None:
        store = self._store()
        for seconds in (10.0, 20.0, 30.0, 40.0):
            store.record(seconds)
        self.assertEqual(store.p50(), 25.0)
        store2 = self._store("other.json")
        for seconds in (5.0, 15.0, 25.0):
            store2.record(seconds)
        self.assertEqual(store2.p50(), 15.0)

    def test_store_survives_restart(self) -> None:
        store = self._store()
        store.record(12.5)
        store.record(130.0)
        reopened = self._store()
        self.assertEqual(reopened.durations(), [12.5, 130.0])
        self.assertEqual(reopened.p50(), 71.25)

    def test_corrupt_file_reads_empty(self) -> None:
        with open(self.path, "w") as handle:
            handle.write("{not json at all")
        store = self._store()
        self.assertEqual(store.durations(), [])
        self.assertIsNone(store.p50())
        # ... and recording over it repairs the file.
        store.record(7.0)
        self.assertEqual(self._store().durations(), [7.0])

    def test_ring_cap_keeps_newest(self) -> None:
        timing = _timing()
        store = self._store()
        for i in range(timing.MAX_SAMPLES + 10):
            store.record(float(i))
        self.assertEqual(store.count, timing.MAX_SAMPLES)
        self.assertEqual(store.durations()[0], 10.0)
        self.assertEqual(store.durations()[-1],
                         float(timing.MAX_SAMPLES + 9))

    def test_slow_reason_names_median_and_threshold(self) -> None:
        timing = _timing()
        store = self._store()
        for _ in range(3):
            store.record(200.0)
        slow, reason = timing.local_is_slow(store)
        self.assertTrue(slow)
        self.assertIn("200.0s", reason)
        self.assertIn("120", reason)
        fast = self._store("fast.json")
        fast.record(10.0)
        self.assertEqual(timing.local_is_slow(fast), (False, ""))
        self.assertEqual(
            timing.local_is_slow(self._store("empty.json")),
            (False, ""))

    def test_timing_line(self) -> None:
        timing = _timing()
        self.assertIsNone(timing.local_timing_line(self._store()))
        store = self._store()
        for seconds in (10.0, 20.0, 30.0):
            store.record(seconds)
        line = timing.local_timing_line(store)
        assert line is not None
        self.assertIn("20.0s", line)
        self.assertIn("3 renders", line)
        self.assertIn("120", line)


class RouteTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.path = os.path.join(self.tmp, "render_timings.json")

    def _store(self, name: str = "render_timings.json"):
        return _timing().RenderTimingStore(
            os.path.join(self.tmp, name))

    def _studio(self, store) -> Studio:
        studio = Studio()
        studio.timing_store = store
        return studio

    def _local(self, studio: Studio, prompt: str = "a green square",
               **kwargs: Any):
        save_to = os.path.join(self.tmp, "out.png")
        with patch.object(Studio, "_local_ready",
                          return_value=(True, "")), \
             patch.object(Studio, "pipeline",
                          return_value=_FakePipe()):
            return studio.generate(prompt, save_to=save_to, **kwargs)

    def test_render_records_its_duration(self) -> None:
        store = self._store()
        studio = self._studio(store)
        self._local(studio)
        self.assertEqual(store.count, 1)
        self.assertGreaterEqual(store.durations()[0], 0.0)
        self.assertIsNotNone(store.p50())

    def test_failed_render_records_nothing(self) -> None:
        store = self._store()
        studio = self._studio(store)
        with patch.object(Studio, "_local_ready",
                          return_value=(True, "")), \
             patch.object(Studio, "pipeline",
                          return_value=_FailingPipe()), \
             self.assertRaises(ImgGenError):
            studio.generate("a green square",
                            save_to=os.path.join(self.tmp, "out.png"))
        self.assertEqual(store.count, 0)

    def test_empty_history_routes_local(self) -> None:
        store = self._store()
        studio = self._studio(store)
        paths = self._local(studio)
        self.assertEqual(studio.last_route, "local")
        self.assertEqual(len(paths), 1)
        # ... and the render itself seeded the history for next time.
        self.assertEqual(store.count, 1)

    def test_below_threshold_stays_local(self) -> None:
        store = self._store()
        for _ in range(5):
            store.record(10.0)
        studio = self._studio(store)
        calls: list[Any] = []
        with patch.object(remote_mod, "remote_generate",
                          side_effect=lambda *a, **k: calls.append(1)):
            paths = self._local(studio)
        self.assertEqual(calls, [])
        self.assertEqual(studio.last_route, "local")
        self.assertEqual(len(paths), 1)

    def test_above_threshold_routes_remote(self) -> None:
        store = self._store()
        for _ in range(5):
            store.record(200.0)
        studio = self._studio(store)
        calls: list[Any] = []

        def fake_remote(*args: Any, **kwargs: Any):
            calls.append((args, kwargs))
            return _png_bytes(), {
                "endpoint": ENDPOINT_URL, "model": "sd15",
                "format": "png", "bytes": len(_png_bytes()),
            }

        with patch.object(remote_mod, "remote_available",
                          return_value=(True, "")), \
             patch.object(remote_mod, "remote_generate",
                          side_effect=fake_remote):
            paths = studio.generate(
                "a red dot",
                save_to=os.path.join(self.tmp, "remote.png"))
        self.assertEqual(len(calls), 1)
        self.assertTrue(studio.last_route.startswith("forwarded"))
        self.assertEqual(len(paths), 1)
        with open(paths[0], "rb") as handle:
            self.assertTrue(handle.read().startswith(b"\x89PNG"))
        # A forwarded render is not a local render — it must not
        # feed the local median.
        self.assertEqual(store.count, 5)

    def test_above_threshold_remote_unavailable_stays_local(self) -> None:
        store = self._store()
        for _ in range(5):
            store.record(200.0)
        studio = self._studio(store)
        with patch.object(remote_mod, "remote_available",
                          return_value=(False, "no forwarded GPU")), \
             patch.object(remote_mod, "remote_generate",
                          side_effect=AssertionError("must not route")):
            paths = self._local(studio)
        self.assertEqual(studio.last_route, "local")
        self.assertEqual(len(paths), 1)

    def test_above_threshold_batch_never_routes(self) -> None:
        store = self._store()
        for _ in range(5):
            store.record(200.0)
        studio = self._studio(store)
        with patch.object(Studio, "_local_ready",
                          return_value=(True, "")), \
             patch.object(Studio, "pipeline",
                          return_value=_FakePipe()), \
             patch.object(remote_mod, "remote_generate",
                          side_effect=AssertionError("must not route")):
            paths = studio.batch(
                ["one", "two"],
                save_to=os.path.join(self.tmp, "batch.png"))
        self.assertEqual(studio.last_route, "local")
        self.assertEqual(len(paths), 2)

    def test_above_threshold_allow_remote_false_stays_local(self) -> None:
        store = self._store()
        for _ in range(5):
            store.record(200.0)
        studio = self._studio(store)
        with patch.object(remote_mod, "remote_generate",
                          side_effect=AssertionError("must not route")):
            paths = self._local(studio, allow_remote=False)
        self.assertEqual(studio.last_route, "local")
        self.assertEqual(len(paths), 1)

    def test_above_threshold_explicit_dtype_refuses(self) -> None:
        store = self._store()
        for _ in range(5):
            store.record(200.0)
        studio = self._studio(store)
        with patch.object(remote_mod, "remote_available",
                          return_value=(True, "")), \
             patch.object(remote_mod, "remote_generate",
                          side_effect=AssertionError("must not route")), \
             self.assertRaisesRegex(ImgGenError,
                                    "cannot be forwarded"):
            self._local(studio, dtype="fp32")

    def test_studio_timing_line(self) -> None:
        store = self._store()
        studio = self._studio(store)
        self.assertIsNone(studio.timing_line())
        for seconds in (10.0, 20.0, 30.0):
            store.record(seconds)
        line = studio.timing_line()
        assert line is not None
        self.assertIn("20.0s", line)
        self.assertIn("3 renders", line)


if __name__ == "__main__":
    unittest.main()
