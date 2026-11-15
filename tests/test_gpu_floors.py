"""Per-family VRAM floors in ``gpu_up`` (nomorals/core/gpu.py).

Each family clears only when the GPU's VRAM meets its floor, derived
from the repo's own organ floors (see the survey for the evidence
trail — every number traces to a real organ requirement):

* seeing 4 GB — Qwen3-VL-4B, ~3.3 GB weights + mmproj, under the
  4 GB phone target (llm/model_catalog.py).
* speaking 4 GB — XTTS / Fish-S2 / Bark / SadTalker all pin 4.0 GB
  (voice/efficiency.py RESOURCE_BUDGETS, lipsync SADTALKER_MIN_VRAM_GB).
* generating 16 GB — Wan 2.2 TI2V-5B, the default video-gen floor
  (media/videogen/capabilities.py).
* training 24 GB — TRELLIS 4B, the repo's heaviest GPU floor
  (media/trellis.py), corroborated by the LoRA rank-64 @ 24 GB
  recommendation (media/afrobeats.py).

Unknown VRAM (an unattested forwarded GPU, vram 0.0) refuses every
family — an unproven number clears no floor. ``text`` and unknown
families never clear. ``gpu_up_reason`` names the refusal; ``gpu_up``
keeps its bool shape.
"""

from __future__ import annotations

import pytest

from nomorals.core import gpu as gpu_mod
from nomorals.core import profiles as profiles_mod
from nomorals.core.gpu import (
    GPU_FAMILY_FLOORS,
    GpuInfo,
    gpu_up,
    gpu_up_reason,
)


def _gpu(vram: float, *, path: str = "local", attested: bool = False,
         url: str = "") -> GpuInfo:
    return GpuInfo(path=path, has_gpu=True, name="NVIDIA RTX 4090",
                   vram_gb=vram, source="attested" if attested else "env",
                   forward_url=url, attested=attested,
                   detail="test GPU")


_CPU = GpuInfo(path="cpu", has_gpu=False, detail="no NVIDIA GPU detected")
_UNATTESTED = GpuInfo(path="forwarded", has_gpu=True, source="env",
                      forward_url="http://gpu-host:8787",
                      detail="forwarded GPU declared at "
                             "http://gpu-host:8787 (not probed — "
                             "attestation failed or timed out)")


# ── the floor table ─────────────────────────────────────────────────

def test_floor_table_is_the_documented_one():
    assert GPU_FAMILY_FLOORS == {
        "seeing": 4.0,
        "speaking": 4.0,
        "generating": 16.0,
        "training": 24.0,
    }


# ── each family clears at its floor, refuses below it ─────────────

@pytest.mark.parametrize(("family", "floor"), [
    ("seeing", 4.0), ("speaking", 4.0),
    ("generating", 16.0), ("training", 24.0),
])
def test_family_clears_at_its_floor_local(family, floor):
    assert gpu_up(family, _gpu(floor)) is True
    assert gpu_up_reason(family, _gpu(floor)) == ""


@pytest.mark.parametrize(("family", "floor"), [
    ("seeing", 4.0), ("speaking", 4.0),
    ("generating", 16.0), ("training", 24.0),
])
def test_family_refuses_below_its_floor_local(family, floor):
    below = _gpu(round(floor - 0.1, 1))
    assert gpu_up(family, below) is False
    reason = gpu_up_reason(family, below)
    assert f"{floor:g}" in reason
    assert "GB VRAM" in reason


@pytest.mark.parametrize(("family", "floor"), [
    ("seeing", 4.0), ("speaking", 4.0),
    ("generating", 16.0), ("training", 24.0),
])
def test_family_clears_at_its_floor_attested_forwarded(family, floor):
    info = _gpu(floor, path="forwarded", attested=True,
                url="http://gpu-host:8787")
    assert gpu_up(family, info) is True
    assert gpu_up_reason(family, info) == ""


@pytest.mark.parametrize(("family", "floor"), [
    ("seeing", 4.0), ("speaking", 4.0),
    ("generating", 16.0), ("training", 24.0),
])
def test_family_refuses_below_its_floor_attested_forwarded(family, floor):
    info = _gpu(round(floor - 0.1, 1), path="forwarded", attested=True,
                url="http://gpu-host:8787")
    assert gpu_up(family, info) is False


def test_attested_judged_exactly_like_local():
    # Same VRAM, same verdicts — attestation proves the number, the
    # floors judge it like a local one.
    for vram in (4.0, 8.0, 16.0, 18.0, 24.0):
        local = _gpu(vram)
        fwd = _gpu(vram, path="forwarded", attested=True,
                   url="http://gpu-host:8787")
        for family in ("seeing", "speaking", "generating", "training"):
            assert gpu_up(family, local) == gpu_up(family, fwd)


# ── unknown VRAM refuses every family ─────────────────────────────

@pytest.mark.parametrize("family", ["seeing", "speaking", "generating",
                                    "training"])
def test_unattested_forwarded_refuses_every_family(family):
    assert gpu_up(family, _UNATTESTED) is False
    reason = gpu_up_reason(family, _UNATTESTED)
    assert "never attested" in reason or "unknown" in reason.lower()


@pytest.mark.parametrize("family", ["seeing", "speaking", "generating",
                                    "training"])
def test_unknown_vram_local_shape_refuses_every_family(family):
    unknown = _gpu(0.0)
    assert gpu_up(family, unknown) is False
    assert gpu_up_reason(family, unknown) != ""


# ── no GPU, text, unknown family ──────────────────────────────────

@pytest.mark.parametrize("family", ["seeing", "speaking", "generating",
                                    "training"])
def test_no_gpu_refuses_every_family(family):
    assert gpu_up(family, _CPU) is False
    assert "no GPU" in gpu_up_reason(family, _CPU)


def test_text_never_gpu():
    assert gpu_up("text", _gpu(24.0)) is False
    assert gpu_up("text", _CPU) is False
    assert "text" in gpu_up_reason("text", _gpu(24.0)).lower()


def test_unknown_family_never_gpu():
    assert gpu_up("mining", _gpu(24.0)) is False
    assert gpu_up("", _gpu(24.0)) is False
    assert gpu_up_reason("mining", _gpu(24.0)) != ""


# ── the reason names the numbers ──────────────────────────────────

def test_reason_names_floor_and_measured_vram():
    reason = gpu_up_reason("training", _gpu(18.0))
    assert "24" in reason and "18" in reason


def test_reason_empty_when_up():
    for family, floor in GPU_FAMILY_FLOORS.items():
        assert gpu_up_reason(family, _gpu(floor)) == ""


# ── profile-layer passthrough ─────────────────────────────────────

def test_profiles_gpu_up_reason_matches(monkeypatch):
    monkeypatch.setattr(gpu_mod, "detect_gpu",
                        lambda *a, **k: _gpu(8.0))
    assert profiles_mod.gpu_up("seeing") is True
    assert profiles_mod.gpu_up("training") is False
    assert profiles_mod.gpu_up_reason("training") != ""
    assert "24" in profiles_mod.gpu_up_reason("training")
    assert profiles_mod.gpu_up_reason("seeing") == ""
