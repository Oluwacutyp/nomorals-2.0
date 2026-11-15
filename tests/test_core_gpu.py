"""Shared GPU-capability detection (nomorals/core/gpu.py): one detection
point for the local-GPU / network-forwarded / CPU paths, wired into the
profile runtime (register §11(2), owner-flagged IMPORTANT).

Every probe test patches the probe seams (``_run`` / ``shutil.which``)
or pins via env — nothing here needs a real GPU; this box has none.
Tests that patch the probes restore the process-wide GPU cache to the
real box state before returning, so a faked GPU never leaks into a
later test file (the cache is process-global).
"""

from __future__ import annotations

import shutil
import types

import pytest

from nomorals.core import gpu as gpu_mod
from nomorals.core.gpu import GpuInfo, detect_gpu, gpu_up


def _smi_run(stdout):
    """Fake _run answering nvidia-smi-style output for any args."""
    return lambda args, timeout=None, cwd=None: types.SimpleNamespace(
        returncode=0, stdout=stdout, stderr="")


def _restore_real_gpu():
    """Re-probe with the real box (no patches active) so the cache
    holds the true state, never a faked GPU."""
    detect_gpu(refresh=True)


@pytest.fixture(autouse=True)
def _clean_gpu_env(monkeypatch):
    monkeypatch.delenv("NM_GPU", raising=False)
    monkeypatch.delenv("NM_GPU_URL", raising=False)
    yield
    # Undo the test's probe/env patches FIRST (fixture teardown runs
    # before monkeypatch's own teardown), then re-probe with the real
    # box — otherwise the restore would re-cache the faked GPU.
    monkeypatch.undo()
    _restore_real_gpu()


# ── local detection ───────────────────────────────────────────────

def test_cpu_path_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(gpu_mod, "_run",
                        lambda args, timeout=None, cwd=None:
                        types.SimpleNamespace(returncode=0, stdout="ok",
                                              stderr=""))
    got = detect_gpu(refresh=True)
    assert got.path == "cpu"
    assert got.has_gpu is False
    assert got.vram_gb == 0.0
    assert got.source == ""


def test_local_nvidia_smi(monkeypatch):
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod, "_run",
                        _smi_run("NVIDIA H100 80GB HBM3, 81559\n"))
    got = detect_gpu(refresh=True)
    assert got.path == "local"
    assert got.has_gpu is True
    assert got.name == "NVIDIA H100 80GB HBM3"
    assert got.vram_gb == 79.65
    assert got.source == "nvidia-smi"
    assert "H100" in got.detail


def test_local_torch_fallback(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(gpu_mod, "_run",
                        _smi_run("NVIDIA GeForce RTX 4090\n25769803776\n"))
    got = detect_gpu(refresh=True)
    assert got.path == "local"
    assert got.has_gpu is True
    assert got.name == "NVIDIA GeForce RTX 4090"
    assert got.vram_gb == 25.77
    assert got.source == "torch.cuda"


# ── pins: NM_GPU / NM_GPU_URL ─────────────────────────────────────

def test_pin_cpu_forces_cpu_even_with_gpu_present(monkeypatch):
    monkeypatch.setenv("NM_GPU", "cpu")
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod, "_run",
                        _smi_run("NVIDIA H100 80GB HBM3, 81559\n"))
    got = detect_gpu(refresh=True)
    assert got.path == "cpu"
    assert got.has_gpu is False
    assert "NM_GPU=cpu" in got.detail


def test_pin_forwarded_with_url(monkeypatch):
    monkeypatch.setenv("NM_GPU", "forwarded")
    monkeypatch.setenv("NM_GPU_URL", "https://gpu.example:8787")
    # no local probe must run: a present local GPU is irrelevant
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi")
    got = detect_gpu(refresh=True)
    assert got.path == "forwarded"
    assert got.has_gpu is True
    assert got.forward_url == "https://gpu.example:8787"
    assert got.source == "env"
    assert "not probed" in got.detail


def test_pin_forwarded_without_url_fails_closed(monkeypatch):
    monkeypatch.setenv("NM_GPU", "forwarded")
    got = detect_gpu(refresh=True)
    assert got.path == "forwarded"
    assert got.has_gpu is False
    assert "NM_GPU_URL" in got.detail


def test_url_alone_implies_forwarded(monkeypatch):
    monkeypatch.setenv("NM_GPU_URL", "gpu-box.local:9999")
    got = detect_gpu(refresh=True)
    assert got.path == "forwarded"
    assert got.has_gpu is True
    assert got.forward_url == "gpu-box.local:9999"


def test_pin_local_ignores_url(monkeypatch):
    monkeypatch.setenv("NM_GPU", "local")
    monkeypatch.setenv("NM_GPU_URL", "https://gpu.example:8787")
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod, "_run",
                        _smi_run("NVIDIA H100 80GB HBM3, 81559\n"))
    got = detect_gpu(refresh=True)
    assert got.path == "local"
    assert got.has_gpu is True
    assert got.source == "nvidia-smi"


def test_unknown_pin_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("NM_GPU", "quantum")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(gpu_mod, "_run",
                        lambda args, timeout=None, cwd=None:
                        types.SimpleNamespace(returncode=0, stdout="ok",
                                              stderr=""))
    got = detect_gpu(refresh=True)
    assert got.path == "cpu"
    assert got.has_gpu is False


# ── cache ─────────────────────────────────────────────────────────

def test_detect_caches_and_refresh_reprobes(monkeypatch):
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod, "_run",
                        _smi_run("NVIDIA H100 80GB HBM3, 81559\n"))
    first = detect_gpu(refresh=True)
    assert first.vram_gb == 79.65
    monkeypatch.setattr(gpu_mod, "_run",
                        _smi_run("NVIDIA GeForce RTX 4090, 24576\n"))
    assert detect_gpu() is first  # cached: no re-probe
    second = detect_gpu(refresh=True)
    assert second is not first
    assert second.vram_gb == 24.0


# ── gpu_up: the activation seam ───────────────────────────────────

_LOCAL = GpuInfo(path="local", has_gpu=True, name="NVIDIA RTX 4090",
                 vram_gb=24.0, source="nvidia-smi",
                 detail="NVIDIA RTX 4090 with 24.0 GB VRAM (nvidia-smi)")
_CPU = GpuInfo(path="cpu", has_gpu=False, detail="no NVIDIA GPU detected")


@pytest.mark.parametrize("family", ["seeing", "speaking", "generating",
                                    "training"])
def test_gpu_up_for_gpu_families_on_gpu_box(family):
    assert gpu_up(family, _LOCAL) is True


def test_gpu_up_text_is_never_gpu():
    assert gpu_up("text", _LOCAL) is False
    assert gpu_up("text", _CPU) is False


def test_gpu_up_unknown_family_is_false():
    assert gpu_up("mining", _LOCAL) is False
    assert gpu_up("", _LOCAL) is False


def test_gpu_up_false_without_gpu():
    for family in ("seeing", "speaking", "generating", "training"):
        assert gpu_up(family, _CPU) is False


def test_gpu_up_defaults_to_detected_box(monkeypatch):
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod, "_run",
                        _smi_run("NVIDIA H100 80GB HBM3, 81559\n"))
    detect_gpu(refresh=True)
    assert gpu_up("generating") is True
    assert gpu_up("text") is False


# ── profile wiring ────────────────────────────────────────────────

def test_profile_carries_gpu(monkeypatch):
    from nomorals.core import profile as profile_mod

    canned = GpuInfo(path="local", has_gpu=True, name="NVIDIA RTX 4090",
                     vram_gb=24.0, source="nvidia-smi",
                     detail="NVIDIA RTX 4090 with 24.0 GB VRAM (nvidia-smi)")
    monkeypatch.setattr(profile_mod, "detect_gpu", lambda **kw: canned)
    prof = profile_mod.detect_profile()
    assert prof.gpu is canned
    assert prof.to_dict()["gpu"]["path"] == "local"
    assert prof.to_dict()["gpu"]["vram_gb"] == 24.0
    assert prof.to_dict()["gpu"]["has_gpu"] is True


def test_describe_profile_includes_gpu(monkeypatch):
    from nomorals.core import profile as profile_mod
    from nomorals.core.profiles import describe_profile

    canned = GpuInfo(path="forwarded", has_gpu=True, name="",
                     vram_gb=0.0, source="env",
                     detail="forwarded GPU declared at gpu-box:9999 "
                            "(not probed)",
                     forward_url="gpu-box:9999")
    monkeypatch.setattr(profile_mod, "detect_gpu", lambda **kw: canned)
    info = describe_profile()
    assert info["gpu"]["path"] == "forwarded"
    assert info["gpu"]["forward_url"] == "gpu-box:9999"
    assert info["environment"]["gpu"]["path"] == "forwarded"


def test_format_profile_shows_gpu(monkeypatch):
    from nomorals.core import profile as profile_mod
    from nomorals.core.profiles import format_profile

    canned = GpuInfo(path="local", has_gpu=True, name="NVIDIA RTX 4090",
                     vram_gb=24.0, source="nvidia-smi",
                     detail="NVIDIA RTX 4090 with 24.0 GB VRAM (nvidia-smi)")
    monkeypatch.setattr(profile_mod, "detect_gpu", lambda **kw: canned)
    out = format_profile(color=False)
    assert "gpu:" in out
    assert "RTX 4090" in out


def test_re_detect_notices_gpu_change(monkeypatch):
    from nomorals.agents import runtime_profile as rp_mod
    from nomorals.core.profile import EnvironmentProfile

    def _prof(gpu):
        return EnvironmentProfile(kind="pc", min_vcpus=2, target_vcpus=4,
                                  max_vcpus=8, cpu=4, memory_mb=8192,
                                  detail="pc", gpu=gpu)

    mgr = rp_mod.RuntimeProfileManager(profile=_prof(_CPU))
    new = _prof(_LOCAL)
    monkeypatch.setattr(rp_mod, "detect_profile", lambda **kw: new)
    result = mgr.re_detect()
    assert result["changed"] is True
    assert "gpu" in result["changes"]
    assert mgr.profile.gpu is _LOCAL


def test_profile_summary_shows_gpu(monkeypatch):
    from nomorals.core import profile as profile_mod
    from nomorals.core.profile_visible import profile_summary

    canned = GpuInfo(path="local", has_gpu=True, name="NVIDIA RTX 4090",
                     vram_gb=24.0, source="nvidia-smi",
                     detail="NVIDIA RTX 4090 with 24.0 GB VRAM (nvidia-smi)")
    monkeypatch.setattr(profile_mod, "detect_gpu", lambda **kw: canned)
    out = profile_summary(shed="none")
    assert out["gpu"]["path"] == "local"
    assert "gpu" in out["line"]
    assert "RTX 4090" in out["line"]


def test_profile_summary_no_gpu_token_without_gpu(monkeypatch):
    from nomorals.core.profile_visible import profile_summary

    out = profile_summary(shed="none")
    assert out["gpu"]["path"] == "cpu"
    assert "gpu" not in out["line"]
