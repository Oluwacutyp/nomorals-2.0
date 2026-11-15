"""GPU capability detection — the single detection point (register
§11(2), owner-flagged IMPORTANT).

Every organ used to probe the GPU its own way (trellis's ``probe_gpu``,
videogen's ``torch.cuda`` import, the edit engine's ``nvidia-smi``,
lipsync's per-backend checks, the models provisioner's subprocess
gate) — same question, five answers, none of them visible to the
profile runtime. This module answers it once, and the profile runtime
carries the answer:

* ``path`` — ``"local"`` (an NVIDIA GPU on this box), ``"forwarded"``
  (a GPU reached over the network, declared by configuration), or
  ``"cpu"`` (no GPU path at all).
* ``detect_gpu()`` — precedence: explicit ``pin`` argument →
  ``NM_GPU`` env var → ``NM_GPU_URL`` (a configured endpoint is itself
  the declaration) → local probing (``nvidia-smi``, then ``torch``) →
  CPU fallback. An explicit pin always wins, like ``NM_PROFILE`` and
  ``NM_CLOUD``. Detection never raises and never pretends: a forwarded
  GPU is reported as declared-but-not-probed (there is no verification
  protocol for a remote endpoint yet), and a forwarded pin with no
  endpoint fails closed (``has_gpu`` False — nothing routes to a GPU
  with no address).
* ``gpu_up(family)`` — the activation seam: ``seeing`` / ``speaking`` /
  ``generating`` / ``training`` run on the GPU when one is present;
  ``text`` never does. Per-organ VRAM floors (LatentSync 18 GB, …)
  stay with the organs — this module answers "is there a GPU path",
  not "does this model fit".

Probing is cached per process (the torch probe spawns a subprocess);
``refresh=True`` re-probes. Tests that fake the probes must restore
the cache to the real box afterwards — see ``tests/test_core_gpu.py``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GPU_PATHS",
    "GPU_FAMILIES",
    "GPU_PIN_ENV",
    "GPU_URL_ENV",
    "GpuInfo",
    "detect_gpu",
    "gpu_up",
]

#: GPU path values: local GPU / network-forwarded GPU / CPU only.
GPU_PATHS: tuple[str, ...] = ("local", "forwarded", "cpu")

#: Capability families that run on the GPU when one is present.
#: Text is deliberately absent — text-only work never needs a GPU.
GPU_FAMILIES: frozenset[str] = frozenset(
    {"seeing", "speaking", "generating", "training"})

#: Env vars: NM_GPU pins the path (auto|local|forwarded|cpu);
#: NM_GPU_URL names the forwarded endpoint (implies forwarded).
GPU_PIN_ENV = "NM_GPU"
GPU_URL_ENV = "NM_GPU_URL"


def _run(args: list[str], timeout: int | None = None,
         cwd: str | None = None) -> Any:
    """Subprocess seam (same shape as media.trellis._run)."""
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout, cwd=cwd)


@dataclass(frozen=True)
class GpuInfo:
    """One GPU-capability answer: which path, and what it offers."""

    path: str = "cpu"          # local | forwarded | cpu
    has_gpu: bool = False
    name: str = ""             # device name, "" when unknown/unprobed
    vram_gb: float = 0.0       # 0.0 when unknown/unprobed
    source: str = ""           # nvidia-smi | torch.cuda | env | ""
    detail: str = ""           # human-readable: what was found, and how
    forward_url: str = ""      # forwarded endpoint, "" when not forwarded

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "has_gpu": self.has_gpu,
            "name": self.name,
            "vram_gb": self.vram_gb,
            "source": self.source,
            "detail": self.detail,
            "forward_url": self.forward_url,
        }


def _probe_nvidia_smi() -> dict[str, Any] | None:
    """VRAM via nvidia-smi. None when it cannot answer."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"], timeout=60)
    except Exception:  # noqa: BLE001 - a failed probe is "no GPU"
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        vram_mib = float(parts[1])
    except ValueError:
        return None
    return {"has_gpu": True, "name": parts[0],
            "vram_gb": round(vram_mib / 1024.0, 2),
            "source": "nvidia-smi"}


def _probe_torch() -> dict[str, Any] | None:
    """VRAM via torch.cuda (subprocess — torch stays out of this
    process). None when torch/CUDA cannot answer."""
    snippet = ("import torch; "
               "p = torch.cuda.get_device_properties(0); "
               "print(p.name); print(p.total_memory)")
    try:
        proc = _run([sys.executable, "-c", snippet], timeout=120)
    except Exception:  # noqa: BLE001 - a failed probe is "no GPU"
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").strip().splitlines()
    if len(lines) < 2:
        return None
    try:
        total_bytes = float(lines[1])
    except ValueError:
        return None
    return {"has_gpu": True, "name": lines[0].strip(),
            "vram_gb": round(total_bytes / 1e9, 2),
            "source": "torch.cuda"}


def _cpu_gpu(detail: str) -> GpuInfo:
    return GpuInfo(path="cpu", has_gpu=False, vram_gb=0.0, source="",
                   detail=detail)


def _local_gpu() -> GpuInfo:
    """Local probing: nvidia-smi first, torch second, CPU fallback."""
    for probe in (_probe_nvidia_smi, _probe_torch):
        try:
            found = probe()
        except Exception:  # noqa: BLE001 - a probe never kills detection
            found = None
        if found is not None:
            return GpuInfo(
                path="local", has_gpu=True, name=found["name"],
                vram_gb=found["vram_gb"], source=found["source"],
                detail=f"{found['name']} with {found['vram_gb']} GB VRAM "
                       f"({found['source']})")
    return _cpu_gpu("no NVIDIA GPU detected")


def _forwarded_gpu(url: str) -> GpuInfo:
    """A declared remote GPU. has_gpu is True — the owner's endpoint
    configuration IS the claim, exactly like an NM_PROFILE pin — but
    the detail says plainly that the endpoint was never probed: there
    is no verification protocol for a remote GPU yet, so no VRAM or
    device name is ever reported here."""
    return GpuInfo(path="forwarded", has_gpu=True, source="env",
                   forward_url=url,
                   detail=f"forwarded GPU declared at {url} "
                          "(not probed — no verification protocol)")


#: Process-wide cache: probing spawns subprocesses, so the first
#: detection pays and the rest reuse it. refresh=True re-probes.
_GPU_CACHE: GpuInfo | None = None


def detect_gpu(*, pin: str | None = None,
               forward_url: str | None = None,
               refresh: bool = False) -> GpuInfo:
    """Detect this box's GPU path. Never raises.

    Precedence: explicit ``pin`` → ``NM_GPU`` → ``NM_GPU_URL`` (a
    configured endpoint implies forwarded) → local probing → CPU.
    ``"auto"``, ``""`` and unknown pins all fall through to local
    probing; ``"cpu"`` disables the GPU outright; ``"forwarded"``
    without an endpoint fails closed (no address to route to).
    """
    global _GPU_CACHE
    if not refresh and _GPU_CACHE is not None:
        return _GPU_CACHE
    pin_value = (pin if pin is not None
                 else os.environ.get(GPU_PIN_ENV, "") or "")
    pin_value = str(pin_value).strip().lower()
    url = (forward_url if forward_url is not None
           else os.environ.get(GPU_URL_ENV, "") or "")
    url = str(url).strip()
    if pin_value == "cpu":
        info = _cpu_gpu("GPU disabled by NM_GPU=cpu")
    elif pin_value == "forwarded":
        if url:
            info = _forwarded_gpu(url)
        else:
            info = GpuInfo(path="forwarded", has_gpu=False, source="env",
                           detail="forwarded GPU declared "
                                  "(NM_GPU=forwarded) but no endpoint "
                                  "configured — set NM_GPU_URL")
    elif pin_value == "local":
        info = _local_gpu()
    elif url:
        info = _forwarded_gpu(url)
    else:
        info = _local_gpu()
    _GPU_CACHE = info
    return info


def gpu_up(family: str, gpu: GpuInfo | None = None) -> bool:
    """True when ``family`` should run on the GPU on this box.

    ``seeing`` / ``speaking`` / ``generating`` / ``training`` are GPU
    work when a GPU path exists (local or forwarded); ``text`` is
    never GPU work; anything else is not a known family. ``gpu``
    overrides the detected box (tests and embedders).
    """
    fam = (family or "").strip().lower()
    if fam == "text" or fam not in GPU_FAMILIES:
        return False
    info = gpu if gpu is not None else detect_gpu()
    return bool(info.has_gpu)
