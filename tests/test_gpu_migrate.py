"""GPU probe migration — the per-organ probes delegate to the shared
detector (nomorals.core.gpu).

Each test fakes the two seams to DISAGREE: the old probe's seam says
one thing, the shared detector's seam says another. On the pristine
base the probe trusts its own seam (test fails); after the migration
the probe delegates, so both answers agree (test passes).

The shared detector's seam is ``nomorals.core.gpu._run`` (nvidia-smi /
torch subprocess output). The old seams are the in-process torch
import (videogen, lipsync, video_models), ``host``'s ``subprocess.run``
(edit engine), and ``models_provision``'s ``_run`` (ACE-Step note).
"""
from __future__ import annotations

import importlib.machinery
import shutil
import subprocess
import sys
import types

import pytest

import nomorals.media.directed.lipsync as ls
import nomorals.media.edit_engine.host as host
import nomorals.media.videogen.capabilities as C
import nomorals.media_edit.video_models as vm
import nomorals.tools.models_provision as mp
from nomorals.core import gpu as gpu_mod
from nomorals.core.gpu import detect_gpu

#: Fake nvidia-smi output for an RTX 4090 (24576 MiB = 24.0 GB).
_SMI_4090 = "NVIDIA GeForce RTX 4090, 24576\n"


def _smi_ok(args, timeout=None, cwd=None):
    return types.SimpleNamespace(returncode=0, stdout=_SMI_4090, stderr="")


def _run_dead(args, timeout=None, cwd=None):
    return types.SimpleNamespace(returncode=1, stdout="", stderr="")


def _fake_gpu_present(monkeypatch):
    """The shared detector's seams answer: local RTX 4090, 24 GB.

    Both seams matter: ``_probe_nvidia_smi`` checks ``shutil.which``
    before it ever calls ``_run`` — without the which patch the
    detector skips nvidia-smi and the fake output is parsed as a torch
    answer instead (one line, no VRAM).
    """
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod, "_run", _smi_ok)


def _fake_gpu_absent(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod, "_run", _run_dead)


def _fake_torch(*, cuda: bool):
    """A torch module in sys.modules (never the real one)."""
    mod = types.ModuleType("torch")
    mod.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_properties=lambda i: types.SimpleNamespace(
            total_memory=25769803776))
    mod.__version__ = "2.0.0+cu118"
    return mod


def _restore_real_gpu():
    """Re-probe with the real box (no patches active) so the cache
    holds the true state, never a faked GPU."""
    detect_gpu(refresh=True)


@pytest.fixture(autouse=True)
def _clean_gpu_env(monkeypatch):
    monkeypatch.delenv("NM_GPU", raising=False)
    monkeypatch.delenv("NM_GPU_URL", raising=False)
    gpu_mod._GPU_CACHE = None
    yield
    # Undo the test's probe/env patches FIRST (fixture teardown runs
    # before monkeypatch's own teardown), then re-probe with the real
    # box — otherwise the restore would re-cache the faked GPU.
    monkeypatch.undo()
    _restore_real_gpu()


def _local_vram_gb() -> float:
    return detect_gpu(refresh=True).vram_gb


# ── probe 1: videogen ──────────────────────────────────────────────

def test_videogen_cuda_vram_delegates(monkeypatch):
    # Old seam (in-process torch) says no CUDA; shared detector's seam
    # (nvidia-smi) says 24 GB. They must agree after the migration.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_present(monkeypatch)
    assert C.cuda_vram_gb() == _local_vram_gb()


def test_videogen_torch_cuda_available_delegates(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_present(monkeypatch)
    assert C.torch_cuda_available() == (
        detect_gpu(refresh=True).path == "local")


def test_videogen_neural_capability_uses_shared_detector(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_present(monkeypatch)
    monkeypatch.setattr(C, "get_profile_kind", lambda: "pc")
    monkeypatch.setattr(C, "diffusers_available", lambda: True)
    cap = C.neural_capability(prefer="ltx")
    assert cap.available is True
    assert cap.backend == "ltx"


def test_videogen_no_gpu_agrees(monkeypatch):
    # Guard: both seams say no GPU — the answer stays "none" either way.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_absent(monkeypatch)
    assert C.cuda_vram_gb() is None
    assert C.torch_cuda_available() is False
    assert C.neural_capability().available is False


def test_videogen_forwarded_reports_no_local_vram(monkeypatch):
    # A forwarded GPU is not local CUDA: the local-only probes report
    # no local GPU. (Passes on base too — the guard pins the rule.)
    monkeypatch.setenv("NM_GPU_URL", "https://gpu.example.invalid:443")
    assert C.cuda_vram_gb() is None
    assert C.torch_cuda_available() is False


# ── probe 2: edit engine host ──────────────────────────────────────

def _host_run_nvidia_dead(args, **kwargs):
    if args and args[0] == "nvidia-smi":
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")
    return subprocess.run(args, **kwargs)


def test_host_vram_delegates(monkeypatch):
    # Old seam (host's own nvidia-smi call) fails; shared detector's
    # seam says 24576 MiB. They must agree after the migration.
    monkeypatch.setattr(host, "subprocess",
                        types.SimpleNamespace(run=_host_run_nvidia_dead))
    _fake_gpu_present(monkeypatch)
    assert host._probe_vram_mb() == int(_local_vram_gb() * 1024)


def test_host_probe_host_vram(monkeypatch):
    monkeypatch.setattr(host, "subprocess",
                        types.SimpleNamespace(run=_host_run_nvidia_dead))
    _fake_gpu_present(monkeypatch)
    assert host.probe_host().vram_mb == int(_local_vram_gb() * 1024)


# ── probe 3: lipsync ───────────────────────────────────────────────

def test_lipsync_has_torch_cuda_delegates(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_present(monkeypatch)
    has_torch, cuda = ls._has_torch_cuda()
    assert has_torch is True
    assert cuda == (detect_gpu(refresh=True).path == "local")


def test_lipsync_latentsync_vram_delegates(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_present(monkeypatch)
    assert ls.latentsync_status()["vram_gb"] == round(_local_vram_gb(), 1)


def test_lipsync_sadtaker_cuda_delegates(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_present(monkeypatch)
    assert ls.sadtaker_status()["cuda"] == (
        detect_gpu(refresh=True).path == "local")


# ── probe 4: models provision (ACE-Step note) ───────────────────────

def _ace_run(args, timeout=None, cwd=None):
    """mp._run dispatcher: git/uv/pip succeed; the old torch check
    says no CUDA."""
    if "-c" in args:
        return types.SimpleNamespace(returncode=0, stdout="False\n",
                                     stderr="")
    return types.SimpleNamespace(returncode=0, stdout="ok\n", stderr="")


def test_models_ace_note_delegates(monkeypatch):
    # Old seam (subprocess torch check) says no CUDA; shared
    # detector's seam says a local GPU is present. The installed
    # report must carry no GPU note after the migration.
    monkeypatch.setattr(mp, "_run", _ace_run)
    _fake_gpu_present(monkeypatch)
    report = mp._provision_ace_step({})
    assert report["status"] == "installed"
    assert "none detected here" not in report["detail"]


def test_models_gpu_note_helper(monkeypatch):
    note_fn = getattr(mp, "_gpu_note", None)
    assert callable(note_fn), "models_provision has no _gpu_note helper yet"
    _fake_gpu_present(monkeypatch)
    assert note_fn() == ""
    _fake_gpu_absent(monkeypatch)
    gpu_mod._GPU_CACHE = None
    assert note_fn() == (" — generation needs an NVIDIA GPU, "
                         "none detected here")


# ── found during survey: video_models ──────────────────────────────

def test_video_models_cuda_vram_delegates(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_present(monkeypatch)
    assert vm._cuda_vram_gb() == _local_vram_gb()


def test_video_models_wan_pick_delegates(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    _fake_gpu_present(monkeypatch)
    assert vm.VideoModelRouter()._pick_wan().model == "wan22-14b"
