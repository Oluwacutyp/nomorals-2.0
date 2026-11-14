"""TRELLIS.2 image-to-3D: provision rung + image-to-GLB tool.

TRELLIS.2 (microsoft/TRELLIS.2, MIT code + weights) turns one image
into a textured 3D asset (GLB, PBR materials). It needs Linux + an
NVIDIA GPU with 24 GB+ VRAM (verified A100/H100) — so the provision
rung is GPU-gated and honestly refuses on EC2 (2 vCPU / 2 GB RAM) and
the phone, the same shape as XTTS. The GLB feeds Devon's Blender rung
via the bpy import snippet.

Every install/download is mocked here — no real clone, no real conda
env, no real 4B weights. The seams are module-level functions on
``nomorals.media.trellis`` (``_run``, ``probe_gpu``,
``_trellis_python``, ``is_installed``) and on
``nomorals.tools.models_provision`` (``_run``, ``_GATES``,
``_provision_trellis``), patched per test.
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


def _trellis():
    import nomorals.media.trellis as t

    return t


def _modelsprov():
    import nomorals.tools.models_provision as mp

    return mp


# ── licence + identity pins ────────────────────────────────────────────

def test_licence_facts_pinned():
    t = _trellis()
    assert t.TRELLIS_REPO == "https://github.com/microsoft/TRELLIS.2.git"
    assert t.TRELLIS_HF_REPO == "microsoft/TRELLIS.2-4B"
    assert t.TRELLIS_MIN_VRAM_GB == 24.0
    assert t.TRELLIS_CONDA_ENV == "trellis2"
    assert t.TRELLIS_DEFAULT_PIPELINE == "1024_cascade"
    assert set(t.TRELLIS_PIPELINE_TYPES) == {
        "512", "1024", "1024_cascade", "1536_cascade"}
    note = t.TRELLIS_TERMS_NOTE
    assert "MIT" in note
    assert "nvdiffrast" in note and "nvdiffrec" in note
    assert "BiRefNet" in note
    # The terms are never hidden: the /models report carries them.
    mp = _modelsprov()
    assert note == mp._trellis.TRELLIS_TERMS_NOTE


def test_runner_mirrors_upstream_example():
    """The generation script is upstream example.py's flow, verbatim."""
    src = _trellis()._runner_source()
    assert "Trellis2ImageTo3DPipeline.from_pretrained" in src
    assert "microsoft/TRELLIS.2-4B" in src
    assert "pipeline.cuda()" in src
    assert "pipeline.run(image" in src
    assert "mesh.simplify(16777216)" in src
    assert "o_voxel.postprocess.to_glb" in src
    assert "glb.export(" in src
    assert "extension_webp=True" in src


# ── GPU probe ──────────────────────────────────────────────────────────

def test_probe_gpu_nvidia_smi(monkeypatch):
    t = _trellis()
    import shutil

    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        t, "_run",
        lambda args, timeout=None, cwd=None: types.SimpleNamespace(
            returncode=0, stdout="NVIDIA H100 80GB HBM3, 81559\n",
            stderr=""))
    got = t.probe_gpu()
    assert got["has_gpu"] is True
    assert got["name"] == "NVIDIA H100 80GB HBM3"
    assert got["vram_gb"] == 79.65
    assert got["source"] == "nvidia-smi"


def test_probe_gpu_torch_fallback(monkeypatch):
    t = _trellis()
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(
        t, "_run",
        lambda args, timeout=None, cwd=None: types.SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090\n25769803776\n", stderr=""))
    got = t.probe_gpu()
    assert got["has_gpu"] is True
    assert got["vram_gb"] == 25.77
    assert got["source"] == "torch.cuda"


def test_probe_gpu_none_when_nothing_answers(monkeypatch):
    t = _trellis()
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(t, "_run", _ok_run)  # stdout "ok" is unparseable
    got = t.probe_gpu()
    assert got["has_gpu"] is False
    assert got["vram_gb"] == 0.0


# ── GPU gate ───────────────────────────────────────────────────────────

def test_gate_refuses_non_linux_with_numbers(monkeypatch):
    t = _trellis()
    monkeypatch.setattr(t, "probe_gpu",
                        lambda: {"has_gpu": False, "name": "",
                                 "vram_gb": 0.0, "source": "",
                                 "detail": "no NVIDIA GPU detected"})
    ok, reason = t.gpu_gate(dict(EC2_BOX))
    assert ok is False
    assert "Linux" in reason
    assert "win32" in reason
    assert "2" in reason  # 2 CPUs, 2 GB RAM — the measured numbers


def test_gate_refuses_no_gpu_with_numbers(monkeypatch):
    t = _trellis()
    monkeypatch.setattr(t, "probe_gpu",
                        lambda: {"has_gpu": False, "name": "",
                                 "vram_gb": 0.0, "source": "",
                                 "detail": "no NVIDIA GPU detected"})
    ok, reason = t.gpu_gate(dict(BIG_LINUX_BOX))
    assert ok is False
    assert "24 GB+ VRAM" in reason
    assert "no NVIDIA GPU" in reason
    assert "16" in reason  # 16 CPUs, 64 GB RAM


def test_gate_refuses_small_gpu_with_numbers(monkeypatch):
    t = _trellis()
    monkeypatch.setattr(t, "probe_gpu",
                        lambda: {"has_gpu": True,
                                 "name": "NVIDIA GeForce RTX 3060",
                                 "vram_gb": 12.0, "source": "nvidia-smi",
                                 "detail": "RTX 3060 with 12.0 GB VRAM"})
    ok, reason = t.gpu_gate(dict(BIG_LINUX_BOX))
    assert ok is False
    assert "RTX 3060" in reason
    assert "12.0" in reason
    assert "24 GB+ VRAM" in reason


def test_gate_accepts_24gb_gpu(monkeypatch):
    t = _trellis()
    monkeypatch.setattr(t, "probe_gpu",
                        lambda: {"has_gpu": True,
                                 "name": "NVIDIA H100 80GB HBM3",
                                 "vram_gb": 79.64, "source": "nvidia-smi",
                                 "detail": "H100 with 79.64 GB VRAM"})
    ok, reason = t.gpu_gate(dict(BIG_LINUX_BOX))
    assert ok is True
    assert "79.64" in reason


# ── install state ──────────────────────────────────────────────────────

def _make_repo(tmp_path, monkeypatch):
    t = _trellis()
    repo = tmp_path / "TRELLIS.2"
    pipe = repo / "trellis2" / "pipelines"
    pipe.mkdir(parents=True)
    (pipe / "trellis2_image_to_3d.py").write_text("# pipeline\n")
    monkeypatch.setenv("NM_TRELLIS_DIR", str(repo))
    return t, repo


def test_install_state_missing_repo_only_installed(monkeypatch, tmp_path):
    t, repo = _make_repo(tmp_path, monkeypatch)
    weights = tmp_path / "weights"
    (weights / "snapshots" / "abc").mkdir(parents=True)
    monkeypatch.setenv("NM_TRELLIS_WEIGHTS", str(weights))
    assert t._trellis_state() == "installed"
    assert t.is_installed() is True
    assert t.install_detail() == "installed"


def test_install_state_repo_only(monkeypatch, tmp_path):
    t, repo = _make_repo(tmp_path, monkeypatch)
    monkeypatch.delenv("NM_TRELLIS_WEIGHTS", raising=False)
    assert t._trellis_state() == "repo-only"
    assert t.is_installed() is False
    assert "weights" in t.install_detail()


def test_install_state_missing(monkeypatch, tmp_path):
    t = _trellis()
    monkeypatch.setenv("NM_TRELLIS_DIR", str(tmp_path / "nope"))
    assert t._trellis_state() == "missing"
    assert t.is_installed() is False
    assert "24 GB+ VRAM" in t.install_detail()


# ── the tool: image -> GLB ─────────────────────────────────────────────

def test_tool_not_installed_names_the_command(monkeypatch, tmp_path):
    t = _trellis()
    monkeypatch.setattr(t, "is_installed", lambda: False)
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")
    out = tmp_path / "asset.glb"
    result = t.image_to_3d(image, out)
    assert result["ok"] is False
    assert result["status"] == "not-installed"
    assert "/models trellis" in result["detail"]
    assert result["glb"] is None
    assert not out.exists()  # never fakes an asset


def test_tool_rejects_unknown_resolution(monkeypatch, tmp_path):
    t = _trellis()
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")
    result = t.image_to_3d(image, tmp_path / "a.glb",
                           resolution="2048")
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert "2048" in result["detail"]


def test_tool_rejects_missing_image(monkeypatch, tmp_path):
    t = _trellis()
    result = t.image_to_3d(tmp_path / "nope.jpg", tmp_path / "a.glb")
    assert result["ok"] is False
    assert "not found" in result["detail"]


def test_tool_runs_the_runner_and_returns_the_glb(monkeypatch, tmp_path):
    t = _trellis()
    monkeypatch.setattr(t, "is_installed", lambda: True)
    monkeypatch.setattr(t, "_trellis_python", lambda: ["python3"])
    monkeypatch.setenv("NM_TRELLIS_DIR", str(tmp_path / "repo"))
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")
    out = tmp_path / "asset.glb"
    seen: dict[str, Any] = {}

    def _fake_run(args, timeout=None, cwd=None):
        # The runner script is a temp file — read it while it exists.
        runner = args[-7]
        with open(runner, encoding="utf-8") as fh:
            seen["runner"] = fh.read()
        seen["argv"] = args
        seen["cwd"] = cwd
        out.write_bytes(b"glTF-binary-stub")
        return types.SimpleNamespace(returncode=0, stdout="ok\n",
                                     stderr="")

    monkeypatch.setattr(t, "_run", _fake_run)
    result = t.image_to_3d(image, out, resolution="1024", seed=7)
    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["glb"] == str(out)
    assert "1024" in result["detail"] and "seed 7" in result["detail"]
    # The runner is upstream's flow, invoked with the real args.
    assert "Trellis2ImageTo3DPipeline.from_pretrained" in seen["runner"]
    assert seen["argv"][1].endswith("_trellis_run.py")
    assert seen["argv"][2] == str(image)
    assert seen["argv"][3] == str(out)
    assert seen["argv"][4] == "1024"
    assert seen["argv"][5] == "7"
    # The Blender feed: the snippet imports this GLB.
    snippet = result["blender_snippet"]
    assert "bpy.ops.import_scene.gltf" in snippet
    assert str(out) in snippet


def test_tool_failed_run_reports_stderr(monkeypatch, tmp_path):
    t = _trellis()
    monkeypatch.setattr(t, "is_installed", lambda: True)
    monkeypatch.setattr(t, "_trellis_python", lambda: ["python3"])
    monkeypatch.setenv("NM_TRELLIS_DIR", str(tmp_path / "repo"))
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")

    def _fail_run(args, timeout=None, cwd=None):
        return types.SimpleNamespace(returncode=1, stdout="",
                                     stderr="CUDA out of memory")

    monkeypatch.setattr(t, "_run", _fail_run)
    result = t.image_to_3d(image, tmp_path / "a.glb")
    assert result["ok"] is False
    assert result["status"] == "failed"
    assert "CUDA out of memory" in result["detail"]


def test_tool_no_python_reports_failed(monkeypatch, tmp_path):
    t = _trellis()
    monkeypatch.setattr(t, "is_installed", lambda: True)
    monkeypatch.setattr(t, "_trellis_python", lambda: None)
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0")
    result = t.image_to_3d(image, tmp_path / "a.glb")
    assert result["ok"] is False
    assert "conda" in result["detail"]


def test_blender_snippet_imports_the_glb():
    snippet = _trellis().blender_import_snippet("/tmp/out/asset.glb")
    assert "import bpy" in snippet
    assert "bpy.ops.import_scene.gltf" in snippet
    assert "/tmp/out/asset.glb" in snippet


# ── /models registration ───────────────────────────────────────────────

def test_models_registers_trellis(monkeypatch):
    mp = _modelsprov()
    assert "trellis" in mp.BACKENDS
    info = mp._BACKEND_INFO["trellis"]
    assert info["family"] == "image-3d"
    assert info["disk_gb"] == 20.0
    assert info["ram_gb"] == 4.0
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_trellis", lambda: False)
    rows = {row["id"]: row for row in mp.inventory()}
    assert rows["trellis"]["family"] == "image-3d"
    assert rows["trellis"]["installed"] is False
    assert "24 GB+ VRAM" in rows["trellis"]["detail"]
    assert "trellis" in mp.status_text()
    assert "trellis" in mp.backend_status()


def test_models_command_dispatches_trellis(monkeypatch):
    from nomorals.audio.models import control_models

    mp = _modelsprov()
    calls: list[list[str]] = []

    def _fake_provision(wanted):
        calls.append(list(wanted))
        return {"ok": True, "box": dict(BIG_LINUX_BOX),
                "backends": {"trellis": {"ok": True, "status": "installed",
                                         "detail": "trellis installed",
                                         "steps": []}},
                "learners": []}

    monkeypatch.setattr(mp, "provision_models", _fake_provision)
    text = control_models("trellis")
    assert calls == [["trellis"]]
    assert "trellis" in text
    assert "MIT" in text  # the terms note rides the report


# ── provision ──────────────────────────────────────────────────────────

def test_provision_refuses_ec2_before_any_install(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(EC2_BOX))
    monkeypatch.setattr(mp, "_detect_trellis", lambda: False)
    calls: list[list[str]] = []
    monkeypatch.setattr(mp, "_run",
                        lambda *a, **k: calls.append(list(a[0]))
                        or _ok_run(a[0]))
    result = mp.provision_models(["trellis"])
    rep = result["backends"]["trellis"]
    assert rep["status"] == "unavailable"
    assert rep["ok"] is False
    assert "Linux" in rep["detail"]
    assert "2" in rep["detail"]  # 2 CPUs, 2 GB RAM
    assert calls == []  # nothing was installed, no clone, no pip


def test_provision_refuses_no_gpu_linux_box(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_trellis", lambda: False)
    monkeypatch.setattr(
        mp._trellis, "probe_gpu",
        lambda: {"has_gpu": False, "name": "", "vram_gb": 0.0,
                 "source": "", "detail": "no NVIDIA GPU detected"})
    result = mp.provision_models(["trellis"])
    rep = result["backends"]["trellis"]
    assert rep["status"] == "unavailable"
    assert "24 GB+ VRAM" in rep["detail"]
    assert "no NVIDIA GPU" in rep["detail"]


def test_provision_needs_git_and_conda(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_trellis", lambda: False)
    monkeypatch.setitem(mp._GATES, "trellis",
                        lambda box: (True, "mock GPU"))
    # The provisioner re-checks the gate directly — stub it too.
    monkeypatch.setattr(mp._trellis, "gpu_gate",
                        lambda box: (True, "mock GPU"))
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = mp.provision_models(["trellis"])
    rep = result["backends"]["trellis"]
    assert rep["status"] == "unavailable"
    assert "git" in rep["detail"]


def test_provision_installs_on_a_capable_box(monkeypatch, tmp_path):
    mp = _modelsprov()
    t, repo = _make_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_LINUX_BOX))
    monkeypatch.setattr(mp, "_detect_trellis", lambda: False)
    monkeypatch.setitem(mp._GATES, "trellis",
                        lambda box: (True, "mock GPU"))
    # The provisioner re-checks the gate directly — stub it too.
    monkeypatch.setattr(mp._trellis, "gpu_gate",
                        lambda box: (True, "mock GPU"))
    monkeypatch.setattr(mp._trellis, "_trellis_python",
                        lambda: ["python3"])
    monkeypatch.setattr(mp._trellis, "is_installed", lambda: True)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []

    def _cap_run(args, timeout=None, cwd=None):
        calls.append(list(args))
        return types.SimpleNamespace(returncode=0, stdout="ok\n",
                                     stderr="")

    monkeypatch.setattr(mp, "_run", _cap_run)
    result = mp.provision_models(["trellis"])
    rep = result["backends"]["trellis"]
    assert rep["status"] == "installed", rep
    assert rep["ok"] is True
    # Clone skipped (repo present), setup.sh ran, warm load ran.
    assert any("setup.sh" in " ".join(c) for c in calls)
    assert any("from_pretrained" in " ".join(c) for c in calls)
    assert not any(c[0] == "git" for c in calls)
