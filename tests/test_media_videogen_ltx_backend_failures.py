"""Tail-77 failure pins — nomorals/media/videogen/ltx_backend.py.

Every guard in this module is a rung of a designed ladder, an
optional memory nicety, or teardown: import-gated per its own
docstring ("without diffusers/torch/CUDA this module loads fine
and every call fails honestly"), the memory plan falls back from
``enable_model_cpu_offload()`` to a direct ``.to("cuda")``, I2V
falls back from the native pipeline to the conditioning strategy
(logged), and ``unload`` clears the CUDA cache best-effort after
the references are already dropped.  These pins prove each
guarded failure leaves an honest, working state — and that when
both rungs of a placement fail, the failure propagates instead
of being hidden (there is no pin for that here because _load's
fallback raising IS the unguarded ``.to`` call itself).

Probed live first (nm-tail77-verify/probes.py): fake
diffusers/torch modules injected via sys.modules.
"""
import logging
import sys
import types

import pytest

from nomorals.media.videogen import ltx_backend as ltx


class _FakeVAE:
    def __init__(self, raises=False):
        self.raises = raises
        self.calls = []

    def enable_tiling(self):
        self.calls.append("tiling")
        if self.raises:
            raise RuntimeError("tiling unsupported")

    def enable_slicing(self):
        self.calls.append("slicing")
        if self.raises:
            raise RuntimeError("slicing unsupported")


class _FakePipe:
    def __init__(self, offload_raises=False, vae_raises=False):
        self.calls = []
        self.vae = _FakeVAE(vae_raises)
        self._offload_raises = offload_raises

    def enable_model_cpu_offload(self):
        self.calls.append("offload")
        if self._offload_raises:
            raise RuntimeError("offload unsupported")

    def to(self, dev):
        self.calls.append(("to", dev))
        return self


def _fake_torch(available_raises=False):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    cuda = types.SimpleNamespace(cleared=False)

    def is_available():
        if available_raises:
            raise RuntimeError("cuda probe blew up")
        return True

    def empty_cache():
        cuda.cleared = True

    cuda.is_available = is_available
    cuda.empty_cache = empty_cache
    torch.cuda = cuda
    return torch


def _install(monkeypatch, pipe, *, with_i2v=True, torch=None):
    diffusers = types.ModuleType("diffusers")
    maker = staticmethod(lambda *a, **k: pipe)
    diffusers.LTXPipeline = types.SimpleNamespace(from_pretrained=maker)
    if with_i2v:
        diffusers.LTXImageToVideoPipeline = types.SimpleNamespace(
            from_pretrained=maker)
    diffusers.LTXConditionPipeline = types.SimpleNamespace(
        from_pretrained=maker)
    cond = types.ModuleType(
        "diffusers.pipelines.ltx.pipeline_ltx_condition")
    cond.LTXVideoCondition = object
    utils = types.ModuleType("diffusers.utils")
    utils.export_to_video = utils.load_image = utils.load_video = (
        lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch", torch or _fake_torch())
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.pipelines.ltx.pipeline_ltx_condition", cond)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    monkeypatch.setattr(ltx, "diffusers_available", lambda: True)
    monkeypatch.setattr(ltx, "torch_cuda_available", lambda: True)


def test_187_offload_failure_falls_back_to_direct_cuda(monkeypatch):
    pipe = _FakePipe(offload_raises=True)
    _install(monkeypatch, pipe)
    backend = ltx.LTXBackend()
    assert backend._load() is pipe
    assert pipe.calls == ["offload", ("to", "cuda")]
    # control: a working offload never touches the fallback rung
    pipe2 = _FakePipe(offload_raises=False)
    _install(monkeypatch, pipe2)
    assert ltx.LTXBackend()._load() is pipe2
    assert pipe2.calls == ["offload"]


def test_192_vae_tiling_failure_keeps_the_pipeline(monkeypatch):
    pipe = _FakePipe(vae_raises=True)
    _install(monkeypatch, pipe)
    backend = ltx.LTXBackend()
    assert backend._load() is pipe
    assert backend._pipe is pipe
    # tiling was attempted; the guard swallowed its failure and the
    # load completed — the untiled VAE is a first-class state
    assert pipe.vae.calls == ["tiling"]


def test_215_native_i2v_offload_failure_falls_back(monkeypatch):
    pipe = _FakePipe(offload_raises=True)
    _install(monkeypatch, pipe)
    backend = ltx.LTXBackend()
    got, strategy = backend._load_i2v()
    assert got is pipe
    assert strategy == "native-i2v"
    assert pipe.calls == ["offload", ("to", "cuda")]


def test_219_native_i2v_unavailable_falls_back_to_condition(monkeypatch,
                                                            caplog):
    pipe = _FakePipe(offload_raises=False)
    _install(monkeypatch, pipe, with_i2v=False)
    backend = ltx.LTXBackend()
    with caplog.at_level(logging.INFO, logger=ltx.__name__):
        result = backend._load_i2v()
    assert result[0] is pipe
    assert result[1] == "condition"
    # the strategy switch is announced, not silent
    assert any("ConditionPipeline" in r.getMessage()
               for r in caplog.records)


def test_231_condition_offload_failure_falls_back(monkeypatch):
    pipe = _FakePipe(offload_raises=True)
    _install(monkeypatch, pipe, with_i2v=False)
    backend = ltx.LTXBackend()
    result = backend._load_i2v()
    assert result[1] == "condition"
    assert pipe.calls == ["offload", ("to", "cuda")]


def test_330_unload_swallows_cuda_probe_failure(monkeypatch):
    backend = ltx.LTXBackend()
    backend._pipe = object()
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(available_raises=True))
    assert backend.unload() is None
    # references were dropped BEFORE the guarded cache clear
    assert backend._pipe is None
    assert backend._i2v_pipe is None
    # control: a healthy torch gets its cache cleared
    torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    backend.unload()
    assert torch.cuda.cleared is True
