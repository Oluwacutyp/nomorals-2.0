"""UniRig auto-rigging: any mesh -> a rigged, animatable character.

Upstream: VAST-AI-Research/UniRig (SIGGRAPH '25, Tsinghua + VAST/Tripo).
One model rigs them all — an autoregressive transformer predicts a
topologically valid skeleton hierarchy (Skeleton Tree Tokenization)
and a bone-point cross-attention stage predicts per-vertex skinning
weights, for humans, animals and objects alike. A mesh goes in
(.glb/.obj/.fbx/.vrm, plus .gltf/.dae) and a rigged asset comes out
(FBX with skeleton + skinning by default; the merge export also
writes .glb/.gltf/.vrm/.dae/.blend). The rigged FBX feeds straight
into Devon's Blender rung (``media.edit_engine.render3d_blender``
for rendering, ``media.blender_live`` for live posing) via
:func:`blender_import_snippet`.

Licence facts (pinned by test, never hidden):

* UniRig code AND the UniRig checkpoints are MIT
  (VAST-AI-Research, 2025).

Skinning-model state — VERIFIED 2026-11-14 against the repo at
6793c664 (2026-06-04) and the Hugging Face repo: the skinning model
is SHIPPED, not "coming soon". Both checkpoints live on
``VAST-AI/UniRig`` and download automatically through upstream's own
``src/inference/download.py``:

* skeleton ``experiments/skeleton/articulation-xl_quantization_256/
  model.ckpt`` — 1,439,617,174 bytes (measured from the HF API)
* skin ``experiments/skin/articulation-xl/model.ckpt`` —
  4,375,464,854 bytes (measured from the HF API)

Both are trained on Articulation-XL2.0. What is still coming soon is
only bone-attribute prediction (stiffness for physics) and the full
Rig-XL/VRoid-trained checkpoints (the paper's main results) — neither
blocks rigging; a rigged asset needs skeleton + skinning only.

Successor — SkinTokens (VAST-AI-Research/SkinTokens, MIT) EXISTS:
code + pretrained models + a single-command CLI
(``python demo.py --input mesh.glb --output rigged.glb``), 645 stars,
created 2026-02-08. It unifies skeleton + skinning into one
autoregressive sequence (TokenRig, Qwen3-0.6B backbone, GRPO-refined)
and claims 98%–133% better skinning accuracy and 17%–22% better bone
prediction than UniRig, at a 14 GB VRAM floor (heavier than UniRig's
8 GB). This rung builds UniRig — the shipped, stable release with the
three-stage pipeline upstream documents. SkinTokens is WATCHED, not
built here: a future lane evaluates whether it replaces this rung.

Hardware (upstream "System Requirements"):

* Linux. A CUDA-enabled GPU with at least 8 GB VRAM.
* Python 3.11, PyTorch >= 2.3.1, flash_attn, spconv,
  torch_scatter/torch_cluster, numpy==1.26.4.

Devon has no auto-rigging — she can turn a photo into a GLB
(TRELLIS.2) and steer Blender live, but the mesh in between cannot
be posed or animated. This module closes that chain: photo -> GLB
-> rigged, animatable character. It is the provision + tool half:
:mod:`nomorals.tools.models_provision` owns the ``/models`` rung
(GPU-gated, honest refusal on EC2/phone, same shape as TRELLIS.2);
this module owns the GPU probe (shared with TRELLIS.2 — one probe
in the codebase), the install layout, and the thin honest wrapper
that runs upstream's own three launch scripts in order
(skeleton -> skin -> merge). The wrapper never reimplements rigging
and never fakes a rig — when UniRig is not installed it says so and
names the provision command.

No real downloads happen in tests — the seams are the module-level
``_run`` and ``_unirig_python`` functions, patched per test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .trellis import probe_gpu as _trellis_probe_gpu

#: Upstream repo (MIT).
UNIRIG_REPO = "https://github.com/VAST-AI-Research/UniRig.git"
#: Pretrained checkpoints on Hugging Face (MIT).
UNIRIG_HF_REPO = "VAST-AI/UniRig"
#: Upstream's conda env name (README install step 3).
UNIRIG_CONDA_ENV = "UniRig"
#: Minimum VRAM in GB (upstream "System Requirements").
UNIRIG_MIN_VRAM_GB = 8.0
#: The two checkpoints, as upstream's download.py maps them.
UNIRIG_SKELETON_CKPT = (
    "experiments/skeleton/articulation-xl_quantization_256/model.ckpt")
UNIRIG_SKIN_CKPT = "experiments/skin/articulation-xl/model.ckpt"
#: Measured checkpoint sizes (HF API, 2026-11-14) — ~5.8 GB total.
UNIRIG_SKELETON_CKPT_BYTES = 1439617174
UNIRIG_SKIN_CKPT_BYTES = 4375464854
#: Upstream's quick-inference task configs (the launch scripts' defaults).
UNIRIG_SKELETON_TASK = (
    "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml")
UNIRIG_SKIN_TASK = "configs/task/quick_inference_unirig_skin.yaml"
#: Upstream's default seed (both launch scripts).
UNIRIG_DEFAULT_SEED = 12345
#: Accepted mesh inputs (upstream launch scripts' require_suffix).
UNIRIG_INPUT_SUFFIXES: tuple[str, ...] = (
    ".glb", ".obj", ".fbx", ".vrm", ".gltf", ".dae")
#: Merge-export suffixes (upstream src/inference/merge.py).
UNIRIG_EXPORT_SUFFIXES: tuple[str, ...] = (
    ".fbx", ".glb", ".gltf", ".vrm", ".dae", ".blend")
#: Skinning-model state, verified 2026-11-14 (see module docstring).
UNIRIG_SKINNING_STATE = "shipped"
#: The successor, verified to exist 2026-11-14 (see module docstring).
UNIRIG_SUCCESSOR_REPO = "https://github.com/VAST-AI-Research/SkinTokens"
UNIRIG_SUCCESSOR_MIN_VRAM_GB = 14.0

#: Licence/terms note — carried into the /models report verbatim.
UNIRIG_TERMS_NOTE = (
    "UniRig code + checkpoints are MIT. Skinning is shipped "
    "(Articulation-XL2.0 checkpoints on VAST-AI/UniRig); bone "
    "attributes and the full Rig-XL/VRoid checkpoints are still "
    "coming soon. SkinTokens (VAST-AI-Research/SkinTokens, MIT) is "
    "the announced successor — it exists, with code + pretrained "
    "models, and is watched, not built here.")

#: Env overrides (same shape as the other provisioners).
UNIRIG_DIR_ENV = "NM_UNIRIG_DIR"
UNIRIG_PYTHON_ENV = "NM_UNIRIG_PYTHON"


# ── seams ────────────────────────────────────────────────────────────

def _run(args: list[str], timeout: int | None = None,
         cwd: str | Path | None = None) -> Any:
    """Subprocess seam (same shape as tools.models_provision._run)."""
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout, cwd=str(cwd) if cwd else None)


# ── layout ───────────────────────────────────────────────────────────

def unirig_dir() -> Path:
    """Where the UniRig repo lives (upstream's own layout)."""
    override = (os.environ.get(UNIRIG_DIR_ENV, "") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / "UniRig"


def _hf_cache_dir() -> Path | None:
    """The HF hub cache dir for VAST-AI/UniRig, if it holds a snapshot."""
    hub = (Path.home() / ".cache" / "huggingface" / "hub"
           / "models--VAST-AI--UniRig")
    snaps = hub / "snapshots"
    if snaps.is_dir() and any(p.is_dir() for p in snaps.iterdir()):
        return hub
    return None


def _checkpoint_present(rel: str) -> bool:
    """True when one checkpoint file exists (repo tree or HF cache)."""
    if (unirig_dir() / "experiments" / rel).is_file():
        return True
    hub = _hf_cache_dir()
    if hub is None:
        return False
    return any((snap / rel).is_file()
               for snap in (hub / "snapshots").iterdir())


def _unirig_state() -> str:
    """missing | repo-only | installed."""
    repo = unirig_dir()
    scripts = (repo / "launch" / "inference")
    if not (scripts / "generate_skeleton.sh").is_file():
        return "missing"
    if not (scripts / "generate_skin.sh").is_file():
        return "missing"
    if not (scripts / "merge.sh").is_file():
        return "missing"
    if (_checkpoint_present(UNIRIG_SKELETON_CKPT)
            and _checkpoint_present(UNIRIG_SKIN_CKPT)):
        return "installed"
    return "repo-only"


def is_installed() -> bool:
    """True when the repo AND both checkpoints are present."""
    try:
        return _unirig_state() == "installed"
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


# ── GPU probe + gate ─────────────────────────────────────────────────

def gpu_gate(box: dict[str, Any] | None = None) -> tuple[bool, str]:
    """The install gate: Linux + CUDA GPU + 8 GB+ VRAM.

    The probe is TRELLIS.2's (one GPU probe in the codebase) — only
    the floor differs. Returns ``(ok, reason)`` — the reason always
    names the measured numbers, so a refusal on EC2 (2 vCPU / 2 GB
    RAM, no GPU) or the phone says exactly what is missing.
    """
    box = dict(box or {})
    platform = str(box.get("platform") or sys.platform)
    cpus = box.get("cpus", "?")
    ram = box.get("total_ram_gb", "?")
    if not platform.startswith("linux"):
        return (False,
                f"UniRig is tested only on Linux — this box runs "
                f"{platform} ({cpus} CPUs, {ram} GB RAM); it cannot "
                f"install here")
    gpu = _trellis_probe_gpu()
    if not gpu.get("has_gpu"):
        return (False,
                f"UniRig needs a CUDA GPU with "
                f"{UNIRIG_MIN_VRAM_GB:.0f} GB+ VRAM — no NVIDIA GPU "
                f"detected on this box ({cpus} CPUs, {ram} GB RAM)")
    vram = float(gpu.get("vram_gb") or 0.0)
    # A forwarded GPU is named as one — a passing floor means the
    # *remote* GPU clears it, never a local one.
    holder = (f"forwarded GPU at {gpu['forward_url']}"
              if gpu.get("forward_url") else "this box")
    if vram < UNIRIG_MIN_VRAM_GB:
        return (False,
                f"UniRig needs a CUDA GPU with "
                f"{UNIRIG_MIN_VRAM_GB:.0f} GB+ VRAM — {holder} has "
                f"{gpu.get('name')} with {vram} GB")
    return (True,
            f"GPU {gpu.get('name')} with {vram} GB VRAM — meets the "
            f"{UNIRIG_MIN_VRAM_GB:.0f} GB floor ({holder})")


def install_detail() -> str:
    """One-line install state for the /models inventory.

    Filesystem only — no GPU probe, so the inventory never runs a
    subprocess.
    """
    state = _unirig_state()
    if state == "installed":
        return "installed"
    if state == "repo-only":
        return ("repo present, checkpoints (~5.8 GB) download on first "
                "rig")
    return ("needs Linux + CUDA GPU "
            f"{UNIRIG_MIN_VRAM_GB:.0f} GB+ VRAM")


def unirig_status() -> dict[str, Any]:
    """Install state + GPU, for the /models inventory detail."""
    state = _unirig_state()
    gpu = _trellis_probe_gpu()
    if state == "installed":
        detail = "installed"
    elif state == "repo-only":
        detail = ("repo present, checkpoints (~5.8 GB) download on "
                  "first rig")
    else:
        detail = ("needs Linux + CUDA GPU "
                  f"{UNIRIG_MIN_VRAM_GB:.0f} GB+ VRAM; this box: "
                  f"{gpu['detail']}")
    return {"state": state, "dir": str(unirig_dir()), "gpu": gpu,
            "detail": detail}


# ── the tool: mesh -> rigged FBX ─────────────────────────────────────

def _unirig_python() -> list[str] | None:
    """The interpreter that can run UniRig, as a command prefix.

    Env override first; then the provisioned conda env. None when
    there is no usable interpreter — the install cannot run.
    """
    override = (os.environ.get(UNIRIG_PYTHON_ENV, "") or "").strip()
    if override:
        return [override]
    if shutil.which("conda"):
        return ["conda", "run", "-n", UNIRIG_CONDA_ENV, "python"]
    return None


def blender_import_snippet(fbx_path: str | Path) -> str:
    """The bpy lines that pull a rigged UniRig FBX into Blender.

    This is the feed into Devon's Blender rung
    (``media.edit_engine.render3d_blender`` for rendering,
    ``media.blender_live`` for live posing): an FBX is a native
    Blender import — no conversion, no re-export. The op is the same
    one upstream's own merge step uses to load FBX
    (``bpy.ops.import_scene.fbx``).
    """
    path = str(fbx_path).replace("\\", "\\\\").replace("'", "\\'")
    return ("\n".join([
        "import bpy",
        "bpy.ops.wm.read_factory_settings(use_empty=True)",
        f"bpy.ops.import_scene.fbx(filepath='{path}')",
    ]))


def rig_mesh(mesh: str | Path, out: str | Path, *,
             seed: int = UNIRIG_DEFAULT_SEED,
             timeout: int = 1800) -> dict[str, Any]:
    """Any mesh -> a rigged asset (skeleton + skinning). Never raises.

    Runs upstream's own three launch scripts in order, in the repo
    dir: skeleton, then skin (which takes the skeleton FBX), then
    merge (which takes the skin FBX + the original mesh and writes
    the rigged asset). The merge export follows the output suffix —
    FBX is the default rigged-asset format.

    Returns ``{"ok", "status", "detail", "rigged",
    "blender_snippet"}``. When UniRig is not installed the result is
    ``not-installed`` and names the provision command — it never
    fakes a rig.
    """
    mesh_path = Path(mesh)
    out_path = Path(out)
    suffix = mesh_path.suffix.lower()
    if suffix not in UNIRIG_INPUT_SUFFIXES:
        return {"ok": False, "status": "failed",
                "detail": (f"unsupported mesh {mesh_path.suffix!r} — "
                           f"use one of "
                           f"{', '.join(UNIRIG_INPUT_SUFFIXES)}"),
                "rigged": None, "blender_snippet": ""}
    out_suffix = out_path.suffix.lower()
    if out_suffix not in UNIRIG_EXPORT_SUFFIXES:
        return {"ok": False, "status": "failed",
                "detail": (f"unsupported output {out_path.suffix!r} — "
                           f"use one of "
                           f"{', '.join(UNIRIG_EXPORT_SUFFIXES)}"),
                "rigged": None, "blender_snippet": ""}
    if not mesh_path.is_file():
        return {"ok": False, "status": "failed",
                "detail": f"mesh not found: {mesh_path}",
                "rigged": None, "blender_snippet": ""}
    if not is_installed():
        return {"ok": False, "status": "not-installed",
                "detail": ("UniRig is not installed on this box — "
                           "run /models unirig to install it "
                           "(Linux + CUDA GPU 8 GB+ VRAM)"),
                "rigged": None, "blender_snippet": ""}
    python = _unirig_python()
    if python is None:
        return {"ok": False, "status": "failed",
                "detail": ("no interpreter can run UniRig — the "
                           "provisioned conda env is missing; re-run "
                           "/models unirig"),
                "rigged": None, "blender_snippet": ""}
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - a bad out path is a failure
        return {"ok": False, "status": "failed",
                "detail": f"cannot create {out_path.parent}: {exc}",
                "rigged": None, "blender_snippet": ""}
    repo = unirig_dir()
    stem = mesh_path.stem
    try:
        with tempfile.TemporaryDirectory(prefix="unirig-") as tmp:
            skel = Path(tmp) / f"{stem}_skeleton.fbx"
            skin = Path(tmp) / f"{stem}_skin.fbx"
            stages = (
                ("skeleton",
                 ["bash", "launch/inference/generate_skeleton.sh",
                  "--input", str(mesh_path), "--output", str(skel),
                  "--seed", str(seed)]),
                ("skin",
                 ["bash", "launch/inference/generate_skin.sh",
                  "--input", str(skel), "--output", str(skin),
                  "--seed", str(seed)]),
                ("merge",
                 ["bash", "launch/inference/merge.sh",
                  "--source", str(skin), "--target", str(mesh_path),
                  "--output", str(out_path)]),
            )
            for name, argv in stages:
                try:
                    proc = _run(argv, timeout=timeout, cwd=repo)
                except Exception as exc:  # noqa: BLE001 - a crashed
                    # stage is a failure, not a crash of this call
                    return {"ok": False, "status": "failed",
                            "detail": f"UniRig {name} stage crashed: "
                                      f"{exc}",
                            "rigged": None, "blender_snippet": ""}
                if proc.returncode != 0:
                    return {"ok": False, "status": "failed",
                            "detail": (f"UniRig {name} stage failed "
                                       f"(rc={proc.returncode}): "
                                       f"{(proc.stderr or '')[-300:]}"),
                            "rigged": None, "blender_snippet": ""}
    except Exception as exc:  # noqa: BLE001 - a temp-dir failure is
        # a failure of the rig, not of this call
        return {"ok": False, "status": "failed",
                "detail": f"UniRig rig failed: {exc}",
                "rigged": None, "blender_snippet": ""}
    if not out_path.is_file():
        return {"ok": False, "status": "failed",
                "detail": ("UniRig finished but no rigged asset "
                           f"appeared at {out_path}"),
                "rigged": None, "blender_snippet": ""}
    return {"ok": True, "status": "ok",
            "detail": (f"rigged asset written to {out_path} "
                       f"(seed {seed})"),
            "rigged": str(out_path),
            "blender_snippet": blender_import_snippet(out_path)}


# ── tool registration ────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Register the UniRig run tool on the spine (audit gap R2-25).

    The tool is a thin gate-then-call: the organ's own ``gpu_gate()``
    (Linux + CUDA GPU 8 GB+) fires first and refuses with the measured
    numbers; only then does :func:`rig_mesh` run. The wrapper's dict is
    returned verbatim — never re-shaped, never faked.
    """
    from ..core.policy import Capability

    @registry.register(
        "unirig_rig_mesh",
        description=(
            "Rig any mesh into an animatable character (skeleton + "
            "skinning) with UniRig (mesh, out, seed, timeout). GPU-gated: "
            "Linux + CUDA GPU with 8 GB+ VRAM — refuses honestly with the "
            "measured numbers when the floor does not clear, and says "
            "when UniRig is not installed (/models unirig). Returns the "
            "rigged asset path plus the Blender import snippet."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "mesh": "str — local path to the mesh (.glb/.obj/.fbx/.vrm/.gltf/.dae)",
            "out": "str — local path for the rigged asset (.fbx default)",
            "seed": "int (optional) — rig seed",
            "timeout": "int (optional) — seconds",
        },
    )
    def unirig_rig_mesh(mesh: str, out: str,
                        seed: int = UNIRIG_DEFAULT_SEED,
                        timeout: int = 1800) -> dict[str, Any]:
        ok, reason = gpu_gate()
        if not ok:
            return {"ok": False, "status": "gpu-refused", "detail": reason,
                    "rigged": None, "blender_snippet": ""}
        return rig_mesh(mesh, out, seed=seed, timeout=timeout)
