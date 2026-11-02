"""Tail-91 pins: nomorals/media/ace_step.py census sites.

Module posture (its own docstring): honest boundaries, no fake
audio ever — probe() checks install/CUDA/weights and generate()
fails closed with a precise reason when anything is missing.
Every site is held as an exclusion:

236 — ``_importable`` import crash -> False. The probe helper:
a module that raises on import is, for availability purposes,
not importable. The verdict it feeds (ProbeResult.available)
fails CLOSED — a crash can only ever read as "can't run here",
never as "installed". Nothing is logged or recorded.
244 — ``_cuda_available`` probe crash -> False. Same shape:
a torch/CUDA probe that explodes means CUDA is unusable here;
False routes probe_acestep to its honest "no CUDA GPU
detected" refusal, never to a generation attempt.
275 — ``from diffusers import AceStepPipeline`` crash ->
has_diffusers=False. One rung of the adapter detection
ladder: failure just means the diffusers rung is absent; the
official-repo rung is tried on its own merits. Probe-proven:
with acestep importable and the pipeline import broken, the
probe proceeds to the CUDA gate instead of claiming the code
isn't installed.
352 — ``_DiffusersAdapter.unload`` cache-clear crash -> pass.
Teardown: the pipe reference is already dropped and gc has
run; torch.cuda.empty_cache() is best-effort VRAM hygiene with
no consumer and no claim attached.
406 — ``_OfficialAdapter.unload`` cache-clear crash -> pass.
Identical teardown shape (dit/llm refs dropped first).
"""
from __future__ import annotations

import logging
import sys
import types

from nomorals.media import ace_step as A


# ── 236: import crash reads as not-importable ───────────────────

class _BoomFinder:
    def find_module(self, name, path=None):
        if name == "boom_mod_t91":
            return self
        return None

    def load_module(self, name):
        raise RuntimeError("import exploded")


def test_236_importable_controls_and_crash():
    assert A._importable("json") is True            # control: real module
    assert A._importable("no_such_mod_t91") is False  # control: missing
    finder = _BoomFinder()
    sys.meta_path.insert(0, finder)
    try:
        crashed = A._importable("boom_mod_t91")
    finally:
        sys.meta_path.remove(finder)
    # a module that explodes on import is indistinguishable from
    # a missing one — and that is the honest availability answer
    assert crashed is False


# ── 244: CUDA probe crash reads as no-CUDA ──────────────────────

def test_244_cuda_probe_crash_returns_false(monkeypatch):
    fake_torch = types.ModuleType("torch")

    def _boom():
        raise RuntimeError("cuda boom")

    fake_torch.cuda = types.SimpleNamespace(is_available=_boom)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert A._cuda_available() is False
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    # control: whatever the host has, the probe answers a bool
    assert isinstance(A._cuda_available(), bool)


# ── 275: broken diffusers rung falls through the ladder ─────────

def test_275_pipeline_import_failure_uses_official_rung(monkeypatch):
    monkeypatch.setitem(sys.modules, "acestep", types.ModuleType("acestep"))
    # a diffusers module WITHOUT AceStepPipeline: the from-import fails
    monkeypatch.setitem(sys.modules, "diffusers",
                        types.ModuleType("diffusers"))
    monkeypatch.setattr(A, "_cuda_available", lambda: False)
    res = A.probe_acestep("workstation")
    # the probe did NOT conclude "code isn't installed" — it moved
    # on to the CUDA gate, i.e. the official rung carried it
    assert res.available is False
    assert "CUDA" in res.reason
    assert "isn't installed" not in res.reason


def test_275_control_nothing_importable_reports_code_missing(monkeypatch):
    monkeypatch.setattr(A, "_importable", lambda name: False)
    res = A.probe_acestep("workstation")
    assert res.available is False
    assert "isn't installed" in res.reason


# ── 352 / 406: unload teardowns swallow cache-clear failure ─────

def _boom_torch(monkeypatch):
    fake_torch = types.ModuleType("torch")

    def _boom():
        raise RuntimeError("cache boom")

    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True,
                                            empty_cache=_boom)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_352_diffusers_unload_swallows_cache_failure(monkeypatch):
    _boom_torch(monkeypatch)
    adapter = A._DiffusersAdapter()
    adapter._pipe = object()
    assert adapter.unload() is None
    assert adapter._pipe is None  # the model ref was dropped first


def test_406_official_unload_swallows_cache_failure(monkeypatch):
    _boom_torch(monkeypatch)
    adapter = A._OfficialAdapter("/nonexistent")
    adapter._dit = object()
    assert adapter.unload() is None
    assert adapter._dit is None


# ── NOT-recording: the probe/teardown sites log nothing ─────────

def test_sites_record_nothing(caplog):
    with caplog.at_level(logging.DEBUG,
                         logger="nomorals.media.ace_step"):
        A._importable("no_such_mod_t91")
        A._DiffusersAdapter().unload()
        A._OfficialAdapter("/nonexistent").unload()
    assert [r for r in caplog.records
            if r.name == "nomorals.media.ace_step"] == []
