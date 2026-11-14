"""UniRig auto-rigging: provision rung + mesh-to-rigged-FBX tool.

UniRig (VAST-AI-Research/UniRig, MIT, SIGGRAPH '25) predicts a
skeleton and per-vertex skinning weights for ANY mesh — humans,
animals, objects — from one autoregressive model. It needs Linux +
a CUDA GPU with 8 GB+ VRAM, so the provision rung is GPU-gated and
honestly refuses on EC2 (2 vCPU / 2 GB RAM) and the phone, the same
shape as TRELLIS.2. The rigged FBX feeds Devon's Blender rung via
the bpy import snippet.

The skinning model is SHIPPED (not "coming soon"): both checkpoints
live on VAST-AI/UniRig and download through upstream's own
download.py — verified 2026-11-14 against the repo at 6793c664 and
the HF API (skeleton 1,439,617,174 bytes; skin 4,375,464,854 bytes).
SkinTokens, the announced successor, EXISTS (code + pretrained
models + single-command CLI, 14 GB VRAM floor) — it is watched, not
built here; the module pins that fact.

Every install/download is mocked here — no real clone, no real conda
env, no real 5.8 GB checkpoints. The seams are module-level
functions on ``nomorals.media.unirig`` (``_run``,
``_trellis_probe_gpu``, ``_unirig_python``, ``is_installed``) and on
``nomorals.tools.models_provision`` (``_run``, ``_GATES``,
``_provision_unirig``), patched per test.
"""

from __future__ import annotations

import types
from typing import Any

EC2_BOX = {"ok": True, "total_ram_gb": 2.0, "free_ram_gb": 1.0,
           "free_disk_gb": 20.0, "cpus": 2, "platform": "win32",
           "profile": "pc", "detail": "ec2-like box"}

BIG_LINUX_BOX = {"ok": True, "total_ram_gb": 64.0, "free_ram_gb": 40.0,
                 "free_disk_gb": 500.0, "cpus": 16, "platform": "linux",
                 "profile": "workstation", "detail": "big linux box"}


def _ok_run(args, timeout=None, cwd=None):
    return types.SimpleNamespace(returncode=0, stdout="ok\n", stderr="")


def _unirig():
    import nomorals.media.unirig as u

    return u


def _modelsprov():
    import nomorals.tools.models_provision as mp

    return mp


# ── licence + identity pins ────────────────────────────────────────────

def test_licence_facts_pinned():
    u = _unirig()
    assert u.UNIRIG_REPO == "https://github.com/VAST-AI-Research/UniRig.git"
    assert u.UNIRIG_HF_REPO == "VAST-AI/UniRig"
    assert u.UNIRIG_CONDA_ENV == "UniRig"
    assert u.UNIRIG_MIN_VRAM_GB == 8.0
    assert u.UNIRIG_SKELETON_CKPT == (
        "experiments/skeleton/articulation-xl_quantization_256/"
        "model.ckpt")
    assert u.UNIRIG_SKIN_CKPT == (
        "experiments/skin/articulation-xl/model.ckpt")
    # Measured checkpoint sizes (HF API, 2026-11-14) — ~5.8 GB total.
    assert u.UNIRIG_SKELETON_CKPT_BYTES == 1439617174
    assert u.UNIRIG_SKIN_CKPT_BYTES == 4375464854
    assert u.UNIRIG_SKELETON_TASK == (
        "configs/task/quick_inference_skeleton_articulationxl_ar_256"
        ".yaml")
    assert u.UNIRIG_SKIN_TASK == (
        "configs/task/quick_inference_unirig_skin.yaml")
    assert u.UNIRIG_DEFAULT_SEED == 12345
    assert set(u.UNIRIG_INPUT_SUFFIXES) == {
        ".glb", ".obj", ".fbx", ".vrm", ".gltf", ".dae"}
    assert ".fbx" in u.UNIRIG_EXPORT_SUFFIXES
    # The skinning model is SHIPPED — the survey's "coming soon" is
    # outdated (verified against the repo + HF, 2026-11-14).
    assert u.UNIRIG_SKINNING_STATE == "shipped"
    # SkinTokens exists (code + models + CLI) — watched, not built.
    assert u.UNIRIG_SUCCESSOR_REPO == (
        "https://github.com/VAST-AI-Research/SkinTokens")
    assert u.UNIRIG_SUCCESSOR_MIN_VRAM_GB == 14.0
    note = u.UNIRIG_TERMS_NOTE
    assert "MIT" in note
    assert "shipped" in note
    assert "SkinTokens" in note
    # The terms are never hidden: the /models report carries them.
    mp = _modelsprov()
    assert note == mp._unirig.UNIRIG_TERMS_NOTE


# ── GPU gate ───────────────────────────────────────────────────────────

def test_gate_refuses_non_linux_with_numbers(monkeypatch):
    u = _unirig()
    monkeypatch.setattr(u, "_trellis_probe_gpu",
                        lambda: {"has_gpu": False, "name": "",
                                 "vram_gb": 0.0, "source": "",
                                 "detail": "no NVIDIA GPU detected"})
    ok, reason = u.gpu_gate(dict(EC2_BOX))
    assert ok is False
    assert "Linux" in reason
    assert "win32" in reason
    assert "2" in reason  # 2 CPUs, 2 GB RAM — the measured numbers


def test_gate_refuses_no_gpu_with_numbers(monkeypatch):
    u = _unirig()
    monkeypatch.setattr(u, "_trellis_probe_gpu",
                        lambda: {"has_gpu": False, "name": "",
                                 "vram_gb": 0.0, "source": "",
                                 "detail": "no NVIDIA GPU detected"})
    ok, reason = u.gpu_gate(dict(BIG_LINUX_BOX))
    assert ok is False
    assert "8 GB+ VRAM" in reason
    assert "no NVIDIA GPU" in reason
    assert "16" in reason  # 16 CPUs, 64 GB RAM


def test_gate_refuses_small_gpu_with_numbers(monkeypatch):
    u = _unirig()
    monkeypatch.setattr(u, "_trellis_probe_gpu",
                        lambda: {"has_gpu": True,
                                 "name": "NVIDIA GeForce RTX 3060",
                                 "vram_gb": 6.0, "source": "nvidia-smi",
                                 "detail": "RTX 3060 with 6.0 GB VRAM"})
    ok, reason = u.gpu_gate(dict(BIG_LINUX_BOX))
    assert ok is False
    assert "RTX 3060" in reason
    assert "6.0" in reason
    assert "8 GB+ VRAM" in reason


def test_gate_accepts_8gb_gpu(monkeypatch):
    u = _unirig()
    monkeypatch.setattr(u, "_trellis_probe_gpu",
                        lambda: {"has_gpu": True,
                                 "name": "NVIDIA GeForce RTX 4060",
                                 "vram_gb": 8.0, "source": "nvidia-smi",
                                 "detail": "RTX 4060 with 8.0 GB VRAM"})
    ok, reason = u.gpu_gate(dict(BIG_LINUX_BOX))
    assert ok is True
    assert "8.0" in reason


# ── install state ──────────────────────────────────────────────────────

def _make_repo(tmp_path, monkeypatch):
    u = _unirig()
    repo = tmp_path / "UniRig"
    scripts = repo / "launch" / "inference"
    scripts.mkdir(parents=True)
    for name in ("generate_skeleton.sh", "generate_skin.sh",
                 "merge.sh"):
        (scripts / name).write_text("#!/bin/bash\n")
    monkeypatch.setenv("NM_UNIRIG_DIR", str(repo))
    return u, repo


def _make_checkpoints(tmp_path, monkeypatch):
    u, repo = _make_repo(tmp_path, monkeypatch)
    for rel in (u.UNIRIG_SKELETON_CKPT, u.UNIRIG_SKIN_CKPT):
        ckpt = repo / "experiments" / rel
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"ckpt-stub")
    return u, repo


def test_install_state_missing_repo_only_installed(monkeypatch, tmp_path):
    u, repo = _make_checkpoints(tmp_path, monkeypatch)
    assert u._unirig_state() == "installed"
    assert u.is_installed() is True
    assert u.install_detail() == "installed"


def test_install_state_repo_only(monkeypatch, tmp_path):
    u, repo = _make_repo(tmp_path, monkeypatch)
    monkeypatch.delenv("NM_UNIRIG_DIR", raising=False)
    monkeypatch.setenv("NM_UNIRIG_DIR", str(repo))
    assert u._unirig_state() == "repo-only"
    assert u.is_installed() is False
    assert "5.8 GB" in u.install_detail()


def test_install_state_missing(monkeypatch, tmp_path):
    u = _unirig()
    monkeypatch.setenv("NM_UNIRIG_DIR", str(tmp_path / "nope"))
    assert u._unirig_state() == "missing"
    assert u.is_installed() is False
    assert "8 GB+ VRAM" in u.install_detail()


def test_checkpoint_found_in_hf_cache(monkeypatch, tmp_path):
    u = _unirig()
    repo = tmp_path / "UniRig"
    scripts = repo / "launch" / "inference"
    scripts.mkdir(parents=True)
    for name in ("generate_skeleton.sh", "generate_skin.sh",
                 "merge.sh"):
        (scripts / name).write_text("#!/bin/bash\n")
    monkeypatch.setenv("NM_UNIRIG_DIR", str(repo))
    # No checkpoints under the repo — but the HF cache holds them.
    home = tmp_path / "home"
    hub = (home / ".cache" / "huggingface" / "hub"
           / "models--VAST-AI--UniRig" / "snapshots" / "abc123")
    for rel in (u.UNIRIG_SKELETON_CKPT, u.UNIRIG_SKIN_CKPT):
        ckpt = hub / rel
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"ckpt-stub")
    monkeypatch.setenv("HOME", str(home))
    assert u._unirig_state() == "installed"


# ── the tool: mesh -> rigged FBX ───────────────────────────────────────

def test_tool_not_installed_names_the_command(monkeypatch, tmp_path):
    u = _unirig()
    monkeypatch.setattr(u, "is_installed", lambda: False)
    mesh = tmp_path / "giraffe.glb"
    mesh.write_bytes(b"glTF-binary-stub")
    out = tmp_path / "giraffe_rigged.fbx"
    result = u.rig_mesh(mesh, out)
    assert result["ok"] is False
    assert result["status"] == "not-installed"
    assert "/models unirig" in result["detail"]
    assert result["rigged"] is None
    assert not out.exists()  # never fakes a rig


def test_tool_rejects_unsupported_mesh(monkeypatch, tmp_path):
    u = _unirig()
    result = u.rig_mesh(tmp_path / "model.stl",
                        tmp_path / "out.fbx")
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert ".stl" in result["detail"]


def test_tool_rejects_missing_mesh(monkeypatch, tmp_path):
    u = _unirig()
    result = u.rig_mesh(tmp_path / "nope.glb", tmp_path / "out.fbx")
    assert result["ok"] is False
    assert "not found" in result["detail"]


def test_tool_rejects_unsupported_output(monkeypatch, tmp_path):
    u = _unirig()
    mesh = tmp_path / "giraffe.glb"
    mesh.write_bytes(b"glTF-binary-stub")
    result = u.rig_mesh(mesh, tmp_path / "out.png")
    assert result["ok"] is False
    assert ".png" in result["detail"]


def test_tool_runs_the_three_stages_in_order(monkeypatch, tmp_path):
    u = _unirig()
    monkeypatch.setattr(u, "is_installed", lambda: True)
    monkeypatch.setattr(u, "_unirig_python", lambda: ["python3"])
    monkeypatch.setenv("NM_UNIRIG_DIR", str(tmp_path / "repo"))
    mesh = tmp_path / "giraffe.glb"
    mesh.write_bytes(b"glTF-binary-stub")
    out = tmp_path / "giraffe_rigged.fbx"
    seen: dict[str, Any] = {"calls": []}

    def _fake_run(args, timeout=None, cwd=None):
        seen["calls"].append((list(args), cwd))
        # The merge stage (3rd call) writes the rigged asset.
        if args[1].endswith("merge.sh"):
            out.write_bytes(b"fbx-stub")
        return types.SimpleNamespace(returncode=0, stdout="ok\n",
                                     stderr="")

    monkeypatch.setattr(u, "_run", _fake_run)
    result = u.rig_mesh(mesh, out, seed=42)
    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["rigged"] == str(out)
    assert "seed 42" in result["detail"]
    calls = seen["calls"]
    assert len(calls) == 3
    # Stage 1: skeleton — upstream's own launch script, real args.
    skel_argv, skel_cwd = calls[0]
    assert skel_argv[1].endswith("generate_skeleton.sh")
    assert skel_argv[skel_argv.index("--input") + 1] == str(mesh)
    assert skel_argv[skel_argv.index("--seed") + 1] == "42"
    assert str(skel_cwd) == str(tmp_path / "repo")
    skel_out = skel_argv[skel_argv.index("--output") + 1]
    assert skel_out.endswith("_skeleton.fbx")
    # Stage 2: skin — takes the SKELETON fbx, not the mesh.
    skin_argv, _ = calls[1]
    assert skin_argv[1].endswith("generate_skin.sh")
    assert skin_argv[skin_argv.index("--input") + 1] == skel_out
    skin_out = skin_argv[skin_argv.index("--output") + 1]
    assert skin_out.endswith("_skin.fbx")
    # Stage 3: merge — takes the SKIN fbx + the original mesh.
    merge_argv, _ = calls[2]
    assert merge_argv[1].endswith("merge.sh")
    assert merge_argv[merge_argv.index("--source") + 1] == skin_out
    assert merge_argv[merge_argv.index("--target") + 1] == str(mesh)
    assert merge_argv[merge_argv.index("--output") + 1] == str(out)
    # The Blender feed: the snippet imports this FBX.
    snippet = result["blender_snippet"]
    assert "bpy.ops.import_scene.fbx" in snippet
    assert str(out) in snippet


def test_tool_failed_stage_reports_stderr(monkeypatch, tmp_path):
    u = _unirig()
    monkeypatch.setattr(u, "is_installed", lambda: True)
    monkeypatch.setattr(u, "_unirig_python", lambda: ["python3"])
    monkeypatch.setenv("NM_UNIRIG_DIR", str(tmp_path / "repo"))
    mesh = tmp_path / "giraffe.glb"
    mesh.write_bytes(b"glTF-binary-stub")

    def _fail_run(args, timeout=None, cwd=None):
        return types.SimpleNamespace(returncode=1, stdout="",
                                     stderr="CUDA out of memory")

    monkeypatch.setattr(u, "_run", _fail_run)
    result = u.rig_mesh(mesh, tmp_path / "out.fbx")
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert "skeleton" in result["detail"]
    assert "CUDA out of memory" in result["detail"]


def test_tool_no_python_reports_failed(monkeypatch, tmp_path):
    u = _unirig()
    monkeypatch.setattr(u, "is_installed", lambda: True)
    monkeypatch.setattr(u, "_unirig_python", lambda: None)
    mesh = tmp_path / "giraffe.glb"
    mesh.write_bytes(b"glTF-binary-stub")
    result = u.rig_mesh(mesh, tmp_path / "out.fbx")
    assert result["ok"] is False
    assert "conda" in result["detail"]


def test_blender_snippet_imports_the_fbx():
    snippet = _unirig().blender_import_snippet(
        "/tmp/out/giraffe_rigged.fbx")
    assert "import bpy" in snippet
    assert "bpy.ops.import_scene.fbx" in snippet
    assert "/tmp/out/giraffe_rigged.fbx" in snippet


# ── /models registration ───────────────────────────────────────────────

def test_models_registers_unirig(monkeypatch):
    mp = _modelsprov()
    assert "unirig" in mp.BACKENDS
    info = mp._BACKEND_INFO["unirig"]
    assert info["family"] == "image-3d"
    assert info["disk_gb"] == 14.0
    assert info["ram_gb"] == 4.0
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_unirig", lambda: False)
    rows = {row["id"]: row for row in mp.inventory()}
    assert rows["unirig"]["family"] == "image-3d"
    assert rows["unirig"]["installed"] is False
    assert "8 GB+ VRAM" in rows["unirig"]["detail"]
    assert "unirig" in mp.status_text()
    assert "unirig" in mp.backend_status()


def test_models_command_dispatches_unirig(monkeypatch):
    from nomorals.audio.models import control_models

    mp = _modelsprov()
    calls: list[list[str]] = []

    def _fake_provision(wanted):
        calls.append(list(wanted))
        return {"ok": True, "box": dict(BIG_LINUX_BOX),
                "backends": {"unirig": {"ok": True, "status": "installed",
                                        "detail": "unirig installed",
                                        "steps": []}},
                "learners": []}

    monkeypatch.setattr(mp, "provision_models", _fake_provision)
    text = control_models("unirig")
    assert calls == [["unirig"]]
    assert "unirig" in text
    assert "MIT" in text  # the terms note rides the report
    assert "SkinTokens" in text


# ── provision ──────────────────────────────────────────────────────────

def test_provision_refuses_ec2_before_any_install(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(EC2_BOX))
    monkeypatch.setattr(mp, "_detect_unirig", lambda: False)
    calls: list[list[str]] = []
    monkeypatch.setattr(mp, "_run",
                        lambda *a, **k: calls.append(list(a[0]))
                        or _ok_run(a[0]))
    result = mp.provision_models(["unirig"])
    rep = result["backends"]["unirig"]
    assert rep["status"] == "unavailable"
    assert rep["ok"] is False
    assert "Linux" in rep["detail"]
    assert "2" in rep["detail"]  # 2 CPUs, 2 GB RAM
    assert calls == []  # nothing was installed, no clone, no pip


def test_provision_refuses_no_gpu_linux_box(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_unirig", lambda: False)
    monkeypatch.setattr(
        mp._unirig, "_trellis_probe_gpu",
        lambda: {"has_gpu": False, "name": "", "vram_gb": 0.0,
                 "source": "", "detail": "no NVIDIA GPU detected"})
    result = mp.provision_models(["unirig"])
    rep = result["backends"]["unirig"]
    assert rep["status"] == "unavailable"
    assert "8 GB+ VRAM" in rep["detail"]
    assert "no NVIDIA GPU" in rep["detail"]


def test_provision_needs_git_and_conda(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_unirig", lambda: False)
    monkeypatch.setitem(mp._GATES, "unirig",
                        lambda box: (True, "mock GPU"))
    # The provisioner re-checks the gate directly — stub it too.
    monkeypatch.setattr(mp._unirig, "gpu_gate",
                        lambda box: (True, "mock GPU"))
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = mp.provision_models(["unirig"])
    rep = result["backends"]["unirig"]
    assert rep["status"] == "unavailable"
    assert "git" in rep["detail"]


def _cap_run_factory(calls, *, env_has_unirig=True, cuda=""):
    """A fake _run that answers the provisioner's probes."""
    def _cap_run(args, timeout=None, cwd=None):
        calls.append(list(args))
        if args[:3] == ["conda", "env", "list"]:
            stdout = ("# conda environments:\nUniRig\n"
                      if env_has_unirig else
                      "# conda environments:\nbase\n")
            return types.SimpleNamespace(returncode=0, stdout=stdout,
                                         stderr="")
        if "-c" in args and "torch.version.cuda" in " ".join(args):
            return types.SimpleNamespace(returncode=0, stdout=cuda,
                                         stderr="")
        if "-c" in args and "torch.__version__" in " ".join(args):
            return types.SimpleNamespace(returncode=0, stdout="2.3.1\n",
                                         stderr="")
        return types.SimpleNamespace(returncode=0, stdout="ok\n",
                                     stderr="")

    return _cap_run


def test_provision_installs_on_a_capable_box(monkeypatch, tmp_path):
    mp = _modelsprov()
    u, repo = _make_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_unirig", lambda: False)
    monkeypatch.setitem(mp._GATES, "unirig",
                        lambda box: (True, "mock GPU"))
    # The provisioner re-checks the gate directly — stub it too.
    monkeypatch.setattr(mp._unirig, "gpu_gate",
                        lambda box: (True, "mock GPU"))
    monkeypatch.setattr(mp._unirig, "_unirig_python",
                        lambda: ["python3"])
    monkeypatch.setattr(mp._unirig, "is_installed", lambda: True)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []
    monkeypatch.setattr(mp, "_run",
                        _cap_run_factory(calls, env_has_unirig=True,
                                         cuda=""))
    result = mp.provision_models(["unirig"])
    rep = result["backends"]["unirig"]
    assert rep["status"] == "installed", rep
    assert rep["ok"] is True
    # Clone skipped (repo present), env present (no conda create),
    # pip installs ran, warm load (checkpoint download) ran.
    assert not any(c[0] == "git" for c in calls)
    assert not any(c[:2] == ["conda", "create"] for c in calls)
    assert any("requirements.txt" in " ".join(c) for c in calls)
    assert any("download" in " ".join(c) for c in calls)
    assert any("CUDA version unknown" in s for s in rep["steps"])


def test_provision_creates_conda_env_when_missing(monkeypatch, tmp_path):
    mp = _modelsprov()
    u, repo = _make_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_unirig", lambda: False)
    monkeypatch.setitem(mp._GATES, "unirig",
                        lambda box: (True, "mock GPU"))
    monkeypatch.setattr(mp._unirig, "gpu_gate",
                        lambda box: (True, "mock GPU"))
    monkeypatch.setattr(mp._unirig, "_unirig_python",
                        lambda: ["python3"])
    monkeypatch.setattr(mp._unirig, "is_installed", lambda: True)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []
    monkeypatch.setattr(mp, "_run",
                        _cap_run_factory(calls, env_has_unirig=False,
                                         cuda=""))
    result = mp.provision_models(["unirig"])
    rep = result["backends"]["unirig"]
    assert rep["status"] == "installed", rep
    assert any(c[:2] == ["conda", "create"] for c in calls)
    assert any("UniRig" in " ".join(c) and "python=3.11" in " ".join(c)
               for c in calls if c[:2] == ["conda", "create"])


def test_provision_installs_cuda_matched_extras(monkeypatch, tmp_path):
    mp = _modelsprov()
    u, repo = _make_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_unirig", lambda: False)
    monkeypatch.setitem(mp._GATES, "unirig",
                        lambda box: (True, "mock GPU"))
    monkeypatch.setattr(mp._unirig, "gpu_gate",
                        lambda box: (True, "mock GPU"))
    monkeypatch.setattr(mp._unirig, "_unirig_python",
                        lambda: ["python3"])
    monkeypatch.setattr(mp._unirig, "is_installed", lambda: True)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []
    monkeypatch.setattr(mp, "_run",
                        _cap_run_factory(calls, env_has_unirig=True,
                                         cuda="12.1\n"))
    result = mp.provision_models(["unirig"])
    rep = result["backends"]["unirig"]
    assert rep["status"] == "installed", rep
    # The CUDA version came from the env's torch, never guessed.
    assert any("spconv-cu121" in " ".join(c) for c in calls)
    assert any("torch_scatter" in " ".join(c) for c in calls)
    assert any("data.pyg.org/whl/torch-2.3.1+cu121.html" in " ".join(c)
               for c in calls)
