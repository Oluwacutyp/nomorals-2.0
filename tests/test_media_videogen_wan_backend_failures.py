"""Pins for nomorals/media/videogen/wan_backend.py (tail-116):
behavior + NOT-recording.

Sites pinned (all exclusions — documented fallback rungs and
best-effort cleanup on a heavyweight GPU backend):

* 163 — ``_load``: ``enable_model_cpu_offload()`` raising falls back
  to the plain ``pipe.to("cuda")`` placement (the in-code comment
  prints the ladder: offload preferred, direct placement otherwise).
  If the fallback itself fails, that error propagates — the ladder
  has exactly two honest rungs.
* 167 — ``_load``: ``vae.enable_tiling()`` raising is a pass: tiling
  is a VRAM optimization toggle, not a correctness input; the loaded
  pipeline is returned and generation proceeds untiled.
* 237 — ``unload``: the CUDA cache clear is best-effort cleanup run
  AFTER both pipeline handles are already dropped; a raising torch /
  CUDA probe changes no state and masks nothing.

NOT-recording: the three guards emit no warning/error; the module
``_log`` is patched with a MagicMock around the triggers and no
``warning``/``error``/``exception``/``critical`` call may appear
(the healthy "loading …" info line is the module's normal voice).
This file contains NO except handlers: fake pipelines raise inside
the target's own guards.
"""
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.media.videogen import wan_backend as wan


class _FakeVAE:
    def __init__(self, boom):
        self.boom = boom
        self.tiled = 0

    def enable_tiling(self):
        if self.boom:
            raise RuntimeError("tiling unsupported")
        self.tiled += 1


class _FakePipe:
    def __init__(self, offload_boom=False, tiling_boom=False):
        self.offload_boom = offload_boom
        self.offloaded = 0
        self.to_calls = []
        self.vae = _FakeVAE(tiling_boom)

    def enable_model_cpu_offload(self):
        if self.offload_boom:
            raise RuntimeError("offload failed")
        self.offloaded += 1

    def to(self, dev):
        self.to_calls.append(dev)
        return self


def _backend_with_pipe(pipe):
    class _FakeWanPipe:
        @classmethod
        def from_pretrained(cls, model_id, torch_dtype=None):
            return pipe

    fake_diffusers = SimpleNamespace(WanPipeline=_FakeWanPipe,
                                     WanImageToVideoPipeline=_FakeWanPipe)
    fake_torch = SimpleNamespace(bfloat16="bf16")
    return mock.patch.dict(sys.modules, {"diffusers": fake_diffusers,
                                         "torch": fake_torch})


def _loud_calls(log):
    return [c for c in log.method_calls
            if c[0] in ("warning", "error", "exception", "critical")]


class _WanCase(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(wan, "diffusers_available", lambda: True),
            mock.patch.object(wan, "torch_cuda_available", lambda: True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()


class TestOffloadFallback163(_WanCase):
    def test_offload_crash_falls_back_to_direct_cuda(self):
        pipe = _FakePipe(offload_boom=True)
        backend = wan.WanBackend(model_id="fake-model")
        with _backend_with_pipe(pipe), mock.patch.object(wan, "_log") as log:
            out = backend._load()
        self.assertIs(out, pipe)
        self.assertEqual(pipe.to_calls, ["cuda"])
        self.assertEqual(_loud_calls(log), [])

    def test_healthy_control_offload_used_no_fallback(self):
        pipe = _FakePipe()
        backend = wan.WanBackend(model_id="fake-model")
        with _backend_with_pipe(pipe):
            out = backend._load()
        self.assertIs(out, pipe)
        self.assertEqual(pipe.offloaded, 1)
        self.assertEqual(pipe.to_calls, [])


class TestTilingGuard167(_WanCase):
    def test_tiling_crash_still_returns_pipe(self):
        pipe = _FakePipe(tiling_boom=True)
        backend = wan.WanBackend(model_id="fake-model")
        with _backend_with_pipe(pipe), mock.patch.object(wan, "_log") as log:
            out = backend._load()
        self.assertIs(out, pipe)
        self.assertEqual(pipe.offloaded, 1)
        self.assertEqual(_loud_calls(log), [])


class TestUnloadGuard237(unittest.TestCase):
    def test_cuda_probe_crash_unload_still_drops_pipes(self):
        backend = wan.WanBackend(model_id="m")
        backend._pipe = object()
        backend._i2v_pipe = object()

        def _dead():
            raise RuntimeError("cuda dead")

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=_dead, empty_cache=lambda: None))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(wan, "_log") as log:
            result = backend.unload()
        self.assertIsNone(result)
        self.assertIsNone(backend._pipe)
        self.assertIsNone(backend._i2v_pipe)
        self.assertEqual(_loud_calls(log), [])

    def test_healthy_control_empty_cache_called(self):
        backend = wan.WanBackend(model_id="m")
        backend._pipe = object()
        calls = []
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append(1)))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            backend.unload()
        self.assertEqual(calls, [1])
        self.assertIsNone(backend._pipe)


if __name__ == "__main__":
    unittest.main()
