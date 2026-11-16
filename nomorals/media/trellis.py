"""TRELLIS.2 image-to-3D: one photo -> a textured 3D asset (GLB).

Upstream: microsoft/TRELLIS.2 (4B parameters). One image in, a fully
textured asset out — GLB with PBR materials (base colour, metallic,
roughness, alpha), up to 1536^3 resolution. The GLB feeds straight
into Devon's Blender rung (``media.edit_engine.render3d_blender``):
Blender imports it with ``bpy.ops.import_scene.gltf`` — see
:func:`blender_import_snippet`.

Licence facts (pinned by test, never hidden):

* TRELLIS.2 code AND the TRELLIS.2-4B weights are MIT (Microsoft).
* The render deps carry their own terms: nvdiffrast and nvdiffrec
  (NVIDIA) are not MIT — they ship under NVIDIA's licence.
* Background removal uses BiRefNet (``ZhengPeng7/BiRefNet``), which
  carries its own weights licence — it is a separate model from
  TRELLIS.2, not covered by the MIT above.

Hardware (upstream, verified A100/H100):

* Linux only. An NVIDIA GPU with at least 24 GB VRAM is necessary.
* CUDA Toolkit 12.4, conda, Python 3.8+.

Devon has no image-to-3D at all — she can render a scene she is
given, but cannot conjure the asset from a photo. This module is the
provision + tool half: :mod:`nomorals.tools.models_provision` owns the
``/models`` rung (GPU-gated, honest refusal on EC2/phone, same shape
as XTTS); this module owns the GPU probe, the install layout, and the
thin honest wrapper that turns an image into a GLB. The wrapper never
reimplements the model and never fakes an asset — when TRELLIS.2 is
not installed it says so and names the provision command.

No real downloads happen in tests — the seams are the module-level
``_run`` and ``_trellis_python`` functions, patched per test.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

#: Upstream repo (MIT) — cloned with --recursive (o-voxel is a submodule).
TRELLIS_REPO = "https://github.com/microsoft/TRELLIS.2.git"
#: Pretrained weights on Hugging Face (MIT).
TRELLIS_HF_REPO = "microsoft/TRELLIS.2-4B"
#: Upstream's conda env name (setup.sh --new-env creates it).
TRELLIS_CONDA_ENV = "trellis2"
#: Minimum VRAM in GB (upstream: "at least 24GB of memory").
TRELLIS_MIN_VRAM_GB = 24.0
#: Pipeline resolutions (upstream ``run(pipeline_type=...)``).
TRELLIS_PIPELINE_TYPES: tuple[str, ...] = (
    "512", "1024", "1024_cascade", "1536_cascade")
#: Default resolution (upstream default_pipeline_type).
TRELLIS_DEFAULT_PIPELINE = "1024_cascade"
#: nvdiffrast mesh-simplify cap (upstream example.py).
TRELLIS_SIMPLIFY_LIMIT = 16777216

#: Licence/terms note — carried into the /models report verbatim.
TRELLIS_TERMS_NOTE = (
    "TRELLIS.2 code + TRELLIS.2-4B weights are MIT. The render deps "
    "(nvdiffrast, nvdiffrec) carry NVIDIA's own terms, and background "
    "removal uses BiRefNet (ZhengPeng7/BiRefNet) under its own weights "
    "licence — neither is covered by the MIT above.")

#: Env overrides (same shape as the other provisioners).
TRELLIS_DIR_ENV = "NM_TRELLIS_DIR"
TRELLIS_PYTHON_ENV = "NM_TRELLIS_PYTHON"
TRELLIS_WEIGHTS_ENV = "NM_TRELLIS_WEIGHTS"


# ── seams ────────────────────────────────────────────────────────────

def _run(args: list[str], timeout: int | None = None,
         cwd: str | Path | None = None) -> Any:
    """Subprocess seam (same shape as tools.models_provision._run)."""
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout, cwd=str(cwd) if cwd else None)


# ── layout ───────────────────────────────────────────────────────────

def trellis_dir() -> Path:
    """Where the TRELLIS.2 repo lives (upstream's own layout)."""
    override = (os.environ.get(TRELLIS_DIR_ENV, "") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / "TRELLIS.2"


def _weights_dir() -> Path | None:
    """Where the TRELLIS.2-4B weights live, if anywhere yet."""
    env = (os.environ.get(TRELLIS_WEIGHTS_ENV, "") or "").strip()
    if env:
        cand = Path(os.path.expanduser(env))
        if cand.is_dir():
            return cand
        return None
    hub = (Path.home() / ".cache" / "huggingface" / "hub"
           / "models--microsoft--TRELLIS.2-4B")
    snaps = hub / "snapshots"
    if snaps.is_dir() and any(p.is_dir() for p in snaps.iterdir()):
        return hub
    return None


def _trellis_state() -> str:
    """missing | repo-only | installed."""
    repo = trellis_dir()
    pipeline = (repo / "trellis2" / "pipelines"
                / "trellis2_image_to_3d.py")
    if not pipeline.is_file():
        return "missing"
    if _weights_dir() is None:
        return "repo-only"
    return "installed"


def is_installed() -> bool:
    """True when the repo AND the weights are present."""
    try:
        return _trellis_state() == "installed"
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


# ── GPU probe + gate ─────────────────────────────────────────────────

def probe_gpu() -> dict[str, Any]:
    """Measure this box's GPU. Never raises.

    ``{"has_gpu", "name", "vram_gb", "source", "detail", "forward_url",
    "attested"}`` — vram_gb is 0.0 when no NVIDIA GPU answers.

    Delegates to the shared detector (:mod:`nomorals.core.gpu`) — one
    detection point for the whole codebase. The local probing here
    (nvidia-smi, then torch) is exactly the shared detector's local
    path, run fresh every call (``refresh=True``), so the gate always
    measures the box as it is right now. A pinned ``NM_GPU`` /
    ``NM_GPU_URL`` is honoured: a forwarded GPU that passes attestation
    reports its attested VRAM here, so a VRAM floor gate can pass it;
    a forwarded GPU that was never attested still reports ``has_gpu``
    True with unknown VRAM, so the floor still refuses honestly.
    """
    from ..core.gpu import detect_gpu

    info = detect_gpu(refresh=True)
    return {"has_gpu": info.has_gpu, "name": info.name,
            "vram_gb": info.vram_gb, "source": info.source,
            "detail": info.detail, "forward_url": info.forward_url,
            "attested": info.attested}


def gpu_gate(box: dict[str, Any] | None = None) -> tuple[bool, str]:
    """The install gate: Linux + NVIDIA GPU + 24 GB+ VRAM.

    Returns ``(ok, reason)`` — the reason always names the measured
    numbers, so a refusal on EC2 (2 vCPU / 2 GB RAM, no GPU) or the
    phone says exactly what is missing.
    """
    box = dict(box or {})
    platform = str(box.get("platform") or sys.platform)
    cpus = box.get("cpus", "?")
    ram = box.get("total_ram_gb", "?")
    if not platform.startswith("linux"):
        return (False,
                f"TRELLIS.2 is tested only on Linux — this box runs "
                f"{platform} ({cpus} CPUs, {ram} GB RAM); it cannot "
                f"install here")
    gpu = probe_gpu()
    if not gpu.get("has_gpu"):
        return (False,
                f"TRELLIS.2 needs an NVIDIA GPU with "
                f"{TRELLIS_MIN_VRAM_GB:.0f} GB+ VRAM — no NVIDIA GPU "
                f"detected on this box ({cpus} CPUs, {ram} GB RAM)")
    vram = float(gpu.get("vram_gb") or 0.0)
    # A forwarded GPU is named as one — a passing floor means the
    # *remote* GPU clears it, never a local one.
    holder = (f"forwarded GPU at {gpu['forward_url']}"
              if gpu.get("forward_url") else "this box")
    if vram < TRELLIS_MIN_VRAM_GB:
        return (False,
                f"TRELLIS.2 needs an NVIDIA GPU with "
                f"{TRELLIS_MIN_VRAM_GB:.0f} GB+ VRAM — {holder} has "
                f"{gpu.get('name')} with {vram} GB")
    return (True,
            f"GPU {gpu.get('name')} with {vram} GB VRAM — meets the "
            f"{TRELLIS_MIN_VRAM_GB:.0f} GB floor ({holder})")


def install_detail() -> str:
    """One-line install state for the /models inventory.

    Filesystem only — no GPU probe, so the inventory never runs a
    subprocess.
    """
    state = _trellis_state()
    if state == "installed":
        return "installed"
    if state == "repo-only":
        return ("repo present, weights (~8 GB) download on first "
                "generation")
    return ("needs Linux + NVIDIA GPU "
            f"{TRELLIS_MIN_VRAM_GB:.0f} GB+ VRAM")


def trellis_status() -> dict[str, Any]:
    """Install state + GPU, for the /models inventory detail."""
    state = _trellis_state()
    gpu = probe_gpu()
    if state == "installed":
        detail = "installed"
    elif state == "repo-only":
        detail = ("repo present, weights (~8 GB) download on first "
                  "generation")
    else:
        detail = ("needs Linux + NVIDIA GPU "
                  f"{TRELLIS_MIN_VRAM_GB:.0f} GB+ VRAM; this box: "
                  f"{gpu['detail']}")
    return {"state": state, "dir": str(trellis_dir()), "gpu": gpu,
            "detail": detail}


# ── the tool: image -> GLB ───────────────────────────────────────────

_RUNNER_TEMPLATE = string.Template(
    """import os
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline


def main():
    image_path, out_path = sys.argv[1], sys.argv[2]
    pipeline_type, seed = sys.argv[3], int(sys.argv[4])
    texture_size, decimation_target = int(sys.argv[5]), int(sys.argv[6])
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "$hf_repo")
    pipeline.cuda()
    image = Image.open(image_path)
    mesh = pipeline.run(image, seed=seed,
                        pipeline_type=pipeline_type)[0]
    mesh.simplify($simplify_limit)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(out_path, extension_webp=True)


main()
""")


def _runner_source() -> str:
    """The generation script, mirroring upstream example.py exactly.

    The env-map/video half of example.py is dropped — the GLB export
    needs no HDRI. Everything else (from_pretrained, .cuda(), run,
    simplify, to_glb, export) is upstream's own flow.
    """
    return _RUNNER_TEMPLATE.substitute(
        hf_repo=TRELLIS_HF_REPO, simplify_limit=TRELLIS_SIMPLIFY_LIMIT)


def _trellis_python() -> list[str] | None:
    """The interpreter that can run TRELLIS.2, as a command prefix.

    Env override first; then the provisioned conda env. None when
    there is no usable interpreter — the install cannot run.
    """
    override = (os.environ.get(TRELLIS_PYTHON_ENV, "") or "").strip()
    if override:
        return [override]
    if shutil.which("conda"):
        return ["conda", "run", "-n", TRELLIS_CONDA_ENV, "python"]
    return None


def blender_import_snippet(glb_path: str | Path) -> str:
    """The bpy lines that pull a TRELLIS.2 GLB into Blender.

    This is the feed into Devon's Blender rung
    (``media.edit_engine.render3d_blender``): a GLB is a mesh file,
    and Blender imports it natively — no conversion, no re-export.
    """
    path = str(glb_path).replace("\\", "\\\\").replace("'", "\\'")
    return ("\n".join([
        "import bpy",
        "bpy.ops.wm.read_factory_settings(use_empty=True)",
        f"bpy.ops.import_scene.gltf(filepath='{path}')",
    ]))


def image_to_3d(image: str | Path, out: str | Path, *,
                resolution: str = TRELLIS_DEFAULT_PIPELINE,
                seed: int = 42, texture_size: int = 4096,
                decimation_target: int = 1000000,
                timeout: int = 1800) -> dict[str, Any]:
    """One image -> a textured GLB (PBR materials). Never raises.

    Returns ``{"ok", "status", "detail", "glb", "blender_snippet"}``.
    When TRELLIS.2 is not installed the result is ``not-installed``
    and names the provision command — it never fakes an asset.
    """
    image_path = Path(image)
    out_path = Path(out)
    if resolution not in TRELLIS_PIPELINE_TYPES:
        return {"ok": False, "status": "failed",
                "detail": (f"unknown resolution {resolution!r} — "
                           f"use one of {', '.join(TRELLIS_PIPELINE_TYPES)}"),
                "glb": None, "blender_snippet": ""}
    if not image_path.is_file():
        return {"ok": False, "status": "failed",
                "detail": f"image not found: {image_path}",
                "glb": None, "blender_snippet": ""}
    if not is_installed():
        return {"ok": False, "status": "not-installed",
                "detail": ("TRELLIS.2 is not installed on this box — "
                           "run /models trellis to install it "
                           "(Linux + NVIDIA GPU 24 GB+ VRAM)"),
                "glb": None, "blender_snippet": ""}
    python = _trellis_python()
    if python is None:
        return {"ok": False, "status": "failed",
                "detail": ("no interpreter can run TRELLIS.2 — the "
                           "provisioned conda env is missing; re-run "
                           "/models trellis"),
                "glb": None, "blender_snippet": ""}
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - a bad out path is a failure
        return {"ok": False, "status": "failed",
                "detail": f"cannot create {out_path.parent}: {exc}",
                "glb": None, "blender_snippet": ""}
    try:
        with tempfile.NamedTemporaryFile(
                "w", suffix="_trellis_run.py", delete=False,
                encoding="utf-8") as fh:
            fh.write(_runner_source())
            runner = fh.name
    except Exception as exc:  # noqa: BLE001 - a bad temp dir is a failure
        return {"ok": False, "status": "failed",
                "detail": f"cannot write the runner script: {exc}",
                "glb": None, "blender_snippet": ""}
    try:
        proc = _run([*python, runner, str(image_path), str(out_path),
                     resolution, str(seed), str(texture_size),
                     str(decimation_target)],
                    timeout=timeout, cwd=trellis_dir())
    except Exception as exc:  # noqa: BLE001 - a crashed run is a failure
        return {"ok": False, "status": "failed",
                "detail": f"TRELLIS.2 run crashed: {exc}",
                "glb": None, "blender_snippet": ""}
    finally:
        with contextlib.suppress(OSError):
            os.unlink(runner)
    if proc.returncode != 0:
        return {"ok": False, "status": "failed",
                "detail": ("TRELLIS.2 run failed "
                           f"(rc={proc.returncode}): "
                           f"{(proc.stderr or '')[-300:]}"),
                "glb": None, "blender_snippet": ""}
    if not out_path.is_file():
        return {"ok": False, "status": "failed",
                "detail": ("TRELLIS.2 finished but no GLB appeared at "
                           f"{out_path}"),
                "glb": None, "blender_snippet": ""}
    return {"ok": True, "status": "ok",
            "detail": (f"GLB written to {out_path} "
                       f"({resolution}, seed {seed})"),
            "glb": str(out_path),
            "blender_snippet": blender_import_snippet(out_path)}


# ── tool registration ────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Register the TRELLIS.2 run tool on the spine (audit gap R2-25).

    The tool is a thin gate-then-call: the organ's own ``gpu_gate()``
    (Linux + NVIDIA GPU 24 GB+) fires first and refuses with the measured
    numbers; only then does :func:`image_to_3d` run. The wrapper's dict
    is returned verbatim — never re-shaped, never faked.
    """
    from ..core.policy import Capability

    @registry.register(
        "trellis_image_to_3d",
        description=(
            "Turn one photo into a textured 3D asset (GLB with PBR "
            "materials) with TRELLIS.2 (image, out, resolution, seed, "
            "texture_size, decimation_target, timeout). GPU-gated: Linux "
            "+ NVIDIA GPU with 24 GB+ VRAM — refuses honestly with the "
            "measured numbers when the floor does not clear, and says "
            "when TRELLIS.2 is not installed (/models trellis). Returns "
            "the GLB path plus the Blender import snippet."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "image": "str — local path to the photo",
            "out": "str — local path for the GLB to write",
            "resolution": ("str (optional) — 512 | 1024 | 1024_cascade | "
                           "1536_cascade"),
            "seed": "int (optional) — generation seed",
            "texture_size": "int (optional)",
            "decimation_target": "int (optional)",
            "timeout": "int (optional) — seconds",
        },
    )
    def trellis_image_to_3d(
        image: str, out: str, resolution: str = TRELLIS_DEFAULT_PIPELINE,
        seed: int = 42, texture_size: int = 4096,
        decimation_target: int = 1000000, timeout: int = 1800,
    ) -> dict[str, Any]:
        ok, reason = gpu_gate()
        if not ok:
            return {"ok": False, "status": "gpu-refused", "detail": reason,
                    "glb": None, "blender_snippet": ""}
        return image_to_3d(image, out, resolution=resolution, seed=seed,
                           texture_size=texture_size,
                           decimation_target=decimation_target,
                           timeout=timeout)
