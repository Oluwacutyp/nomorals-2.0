"""Directed lip-sync self-provision.

Pins: the provision exists and is wired (tool action registered), it
delegates latentsync + wav2lip to the avatar provision (one provisioner,
not two), it installs a missing SadTalker (mocked clone/download), it
skips already-installed backends, it refuses honestly with measured
numbers where the floor is unmet (no fake success), and it never emits
a manual-step list. The LatentSync driver drives scripts.inference
with explicit args — never the arg-less inference.sh.
"""
import sys
import types

import pytest

import nomorals.media.directed.lipsync as ls
import nomorals.media_edit.avatar as avatar

GPU_BOX = {"ok": True, "total_ram_gb": 32.0, "free_ram_gb": 16.0,
           "free_disk_gb": 100.0, "cpus": 8, "platform": "linux",
           "profile": "laptop", "detail": ""}
GPU_24 = {"has_gpu": True, "name": "RTX 4090", "vram_gb": 24.0}
NO_GPU = {"has_gpu": False, "name": "", "vram_gb": 0.0}
CPU_BOX = {"ok": True, "total_ram_gb": 8.32, "free_ram_gb": 3.32,
           "free_disk_gb": 36.84, "cpus": 4, "platform": "linux",
           "profile": "laptop", "detail": ""}
TINY_BOX = {"ok": True, "total_ram_gb": 1.0, "free_ram_gb": 0.5,
            "free_disk_gb": 0.5, "cpus": 2, "platform": "linux",
            "profile": "laptop", "detail": ""}
TERMUX_BOX = dict(GPU_BOX, profile="termux")


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeRun:
    """Simulates git clone / venv / pip / bash for the SadTalker install.

    Every step creates the files the real step would create, so the
    provision's verify pass sees a real install.
    """

    def __init__(self, home):
        self.home = home
        self.calls = []

    def __call__(self, args, timeout=None, cwd=None):
        self.calls.append(list(args))
        cmd = list(args)
        if cmd[:2] == ["git", "clone"]:
            repo = self.home / cmd[-1].split("/")[-1]
            (repo / "scripts").mkdir(parents=True, exist_ok=True)
            (repo / "inference.py").write_text("# sadtaker\n")
            (repo / "scripts" / "download_models.sh").write_text("#!/bin/bash\n")
            return _FakeProc(0)
        if cmd[1:3] == ["-m", "venv"]:
            venv = cmd[3]
            from pathlib import Path
            (Path(venv) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(venv) / "pyvenv.cfg").write_text("")
            (Path(venv) / "bin" / "python").write_text("#!/bin/sh\n")
            return _FakeProc(0)
        if cmd[0] == "bash":
            repo = self.home / "SadTalker"
            (repo / "checkpoints").mkdir(parents=True, exist_ok=True)
            (repo / "checkpoints" / "SadTalker_V0.0.2_256.safetensors").write_text("x")
            return _FakeProc(0)
        return _FakeProc(0)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """HOME + avatar canonical dirs under tmp; nothing is installed."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # MODELS_DIR is frozen at import time — re-point it for the tests.
    monkeypatch.setattr(ls, "MODELS_DIR", tmp_path / ".devon-models")
    ls_dir = tmp_path / "LatentSync"
    w2l_dir = tmp_path / "Wav2Lip"
    monkeypatch.setattr(avatar, "latentsync_dir", lambda: ls_dir)
    monkeypatch.setattr(avatar, "wav2lip_dir", lambda: w2l_dir)
    monkeypatch.setattr(avatar, "latentsync_python",
                        lambda: str(ls_dir / ".venv" / "bin" / "python"))
    monkeypatch.setattr(avatar, "wav2lip_python",
                        lambda: str(w2l_dir / ".venv" / "bin" / "python"))
    monkeypatch.setattr(ls, "_measure_box", lambda: dict(CPU_BOX))
    monkeypatch.setattr(ls, "_probe_gpu", lambda: dict(NO_GPU))
    return tmp_path


def _make_fake_torch(monkeypatch, vram_gb):
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda i: types.SimpleNamespace(
            total_memory=vram_gb * 1e9))
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # The status probes read GPU state from the shared detector now,
    # not from in-process torch (GPU probe migration): torch presence
    # from find_spec, CUDA + VRAM from detect_gpu. Fake both seams
    # with the same numbers, so the gate under test still sees exactly
    # vram_gb. (find_spec would raise on the spec-less fake above, so
    # it is faked directly too.)
    from nomorals.core.gpu import GpuInfo
    monkeypatch.setattr(ls, "find_spec", lambda name: object())
    monkeypatch.setattr(
        ls, "detect_gpu",
        lambda: GpuInfo(path="local", has_gpu=True, name="fake GPU",
                        vram_gb=float(vram_gb), source="torch.cuda",
                        detail="fake"))
    monkeypatch.setattr(ls, "_has_torch_cuda", lambda: (True, True))


def _install_latentsync(tmp_path):
    repo = tmp_path / "LatentSync"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts" / "inference.py").write_text("# inference\n")
    (repo / "checkpoints" / "whisper").mkdir(parents=True, exist_ok=True)
    (repo / "checkpoints" / "latentsync_unet.pt").write_text("x")
    (repo / "checkpoints" / "whisper" / "tiny.pt").write_text("x")
    return repo


def _install_wav2lip(tmp_path):
    repo = tmp_path / "Wav2Lip"
    (repo / "checkpoints").mkdir(parents=True, exist_ok=True)
    (repo / "face_detection" / "detection" / "sfd").mkdir(parents=True,
                                                         exist_ok=True)
    (repo / "inference.py").write_text("# inference\n")
    (repo / "checkpoints" / "wav2lip_gan.pth").write_text("x")
    (repo / "face_detection" / "detection" / "sfd" / "s3fd.pth").write_text("x")
    return repo


# ── delegation ───────────────────────────────────────────────────────

def test_delegates_latentsync_and_wav2lip_to_avatar(fake_home, monkeypatch):
    calls = []

    def fake_avatar(backends, *, box=None):
        calls.append(list(backends))
        return {"ok": True, "box": box or {},
                "backends": {b: {"ok": True, "status": "installed",
                                 "detail": f"{b} installed by avatar",
                                 "steps": []} for b in backends}}

    monkeypatch.setattr(avatar, "provision_backends", fake_avatar)
    res = ls.provision_backends(["latentsync", "wav2lip"], box=dict(GPU_BOX))
    assert calls == [["latentsync", "wav2lip"]]
    assert res["backends"]["latentsync"]["status"] == "installed"
    assert res["backends"]["wav2lip"]["status"] == "installed"
    assert res["ok"] is True


def test_already_installed_skips_avatar_provision(fake_home, monkeypatch):
    _install_latentsync(fake_home)
    _install_wav2lip(fake_home)
    (fake_home / "SadTalker" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fake_home / "SadTalker" / "inference.py").write_text("# sadtaker\n")
    (fake_home / "SadTalker" / "checkpoints"
     / "SadTalker_V0.0.2_256.safetensors").write_text("x")

    def exploding_avatar(backends, *, box=None):
        raise AssertionError("avatar provision must not run for installed backends")

    monkeypatch.setattr(avatar, "provision_backends", exploding_avatar)
    monkeypatch.setattr(ls, "_measure_box", lambda: dict(GPU_BOX))
    monkeypatch.setattr(ls, "_probe_gpu", lambda: dict(GPU_24))
    res = ls.provision_backends(box=dict(GPU_BOX))
    assert res["ok"] is True
    for name in ("latentsync", "wav2lip", "sadtaker"):
        assert res["backends"][name]["status"] == "already-installed", name


def test_legacy_install_counts_as_installed(fake_home, monkeypatch):
    """A manual install under ~/.devon-models is already-installed too."""
    legacy = fake_home / ".devon-models" / "wav2lip"
    (legacy / "repo").mkdir(parents=True, exist_ok=True)
    (legacy / "repo" / "inference.py").write_text("# inference\n")
    (legacy / "wav2lip.pth").write_text("x")
    (legacy / "s3fd.pth").write_text("x")

    def exploding_avatar(backends, *, box=None):
        raise AssertionError("avatar provision must not run for a legacy install")

    monkeypatch.setattr(avatar, "provision_backends", exploding_avatar)
    res = ls.provision_backends(["wav2lip"], box=dict(CPU_BOX))
    assert res["backends"]["wav2lip"]["status"] == "already-installed"


# ── SadTalker install / refusal ──────────────────────────────────────

def test_sadtaker_installs_on_gpu_box(fake_home, monkeypatch):
    fake_run = _FakeRun(fake_home)
    monkeypatch.setattr(ls, "_run", fake_run)
    monkeypatch.setattr(ls, "_measure_box", lambda: dict(GPU_BOX))
    monkeypatch.setattr(ls, "_probe_gpu", lambda: dict(GPU_24))
    res = ls.provision_backends(["sadtaker"], box=dict(GPU_BOX))
    rep = res["backends"]["sadtaker"]
    assert rep["status"] == "installed", rep
    assert res["ok"] is True
    clone = [c for c in fake_run.calls if c[:2] == ["git", "clone"]]
    assert clone and "OpenTalker/SadTalker" in clone[0][-2]
    assert ls._sadtaker_installed() is True


def test_sadtaker_refused_without_gpu(fake_home, monkeypatch):
    fake_run = _FakeRun(fake_home)
    monkeypatch.setattr(ls, "_run", fake_run)
    res = ls.provision_backends(["sadtaker"], box=dict(CPU_BOX))
    rep = res["backends"]["sadtaker"]
    assert rep["status"] == "unavailable"
    assert rep["ok"] is False
    assert "no NVIDIA GPU" in rep["detail"]
    assert "4" in rep["detail"]  # measured floor, not a bare refusal
    assert not [c for c in fake_run.calls if c[:2] == ["git", "clone"]]


def test_sadtaker_refused_on_termux(fake_home, monkeypatch):
    fake_run = _FakeRun(fake_home)
    monkeypatch.setattr(ls, "_run", fake_run)
    monkeypatch.setattr(ls, "_probe_gpu", lambda: dict(GPU_24))
    res = ls.provision_backends(["sadtaker"], box=dict(TERMUX_BOX))
    rep = res["backends"]["sadtaker"]
    assert rep["status"] == "unavailable"
    assert "termux" in rep["detail"]
    assert not [c for c in fake_run.calls if c[:2] == ["git", "clone"]]


def test_sadtaker_floor_refused_on_tiny_box(fake_home, monkeypatch):
    fake_run = _FakeRun(fake_home)
    monkeypatch.setattr(ls, "_run", fake_run)
    monkeypatch.setattr(ls, "_probe_gpu", lambda: dict(GPU_24))
    res = ls.provision_backends(["sadtaker"], box=dict(TINY_BOX))
    rep = res["backends"]["sadtaker"]
    assert rep["status"] == "unavailable"
    assert rep["ok"] is False
    assert "disk" in rep["detail"] and "0.5" in rep["detail"]
    assert not [c for c in fake_run.calls if c[:2] == ["git", "clone"]]


def test_provision_never_raises_on_unknown_backend(fake_home):
    res = ls.provision_backends(["not_a_backend"], box=dict(CPU_BOX))
    assert res["ok"] is False
    assert "unknown backend" in res["error"]


# ── no manual-step list, ever ────────────────────────────────────────

def test_no_manual_steps_anywhere(fake_home, monkeypatch):
    monkeypatch.setattr(avatar, "provision_backends",
                        lambda backends, *, box=None: {
                            "ok": False, "box": box or {},
                            "backends": {b: {"ok": False,
                                             "status": "unavailable",
                                             "detail": "no GPU here",
                                             "steps": []}
                                        for b in backends}})
    res = ls.provision_backends(box=dict(CPU_BOX))
    text = ls.provision_report_text(res)
    assert "git clone" not in text
    assert "manual" not in text.lower()
    assert "provision" in text
    for fn in (ls.wav2lip_status, ls.latentsync_status, ls.sadtaker_status):
        assert "git clone" not in fn()["reason"]
        assert "provision" in fn()["reason"]


# ── status gates ─────────────────────────────────────────────────────

def test_wav2lip_available_on_cpu_without_cuda(fake_home, monkeypatch):
    _install_wav2lip(fake_home)
    monkeypatch.setattr(ls, "_has_torch_cuda", lambda: (True, False))
    st = ls.wav2lip_status()
    assert st["available"] is True
    assert st["cuda"] is False


def test_latentsync_vram_gate_is_18gb(fake_home, monkeypatch):
    _install_latentsync(fake_home)
    _make_fake_torch(monkeypatch, 8.0)
    assert ls.latentsync_status()["available"] is False
    _make_fake_torch(monkeypatch, 24.0)
    assert ls.latentsync_status()["available"] is True


# ── driver: scripts.inference, not inference.sh ──────────────────────

def test_latentsync_driver_uses_scripts_inference(fake_home, monkeypatch):
    repo = _install_latentsync(fake_home)
    _make_fake_torch(monkeypatch, 24.0)
    calls = []
    out = fake_home / "out.mp4"

    def fake_run(args, **kw):
        calls.append(list(args))
        idx = args.index("--video_out_path") + 1
        open(args[idx], "wb").close()
        return types.SimpleNamespace(returncode=0, stderr=b"")

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    res = ls.latentsync_sync("v.mp4", "a.wav", out_path=str(out))
    assert res.path == str(out)
    cmd = calls[0]
    assert "inference.sh" not in " ".join(cmd)
    assert "-m" in cmd and "scripts.inference" in cmd
    assert cmd[cmd.index("--video_path") + 1] == "v.mp4"
    assert cmd[cmd.index("--audio_path") + 1] == "a.wav"
    assert cmd[cmd.index("--inference_ckpt_path") + 1] == str(
        repo / "checkpoints" / "latentsync_unet.pt")


def test_wav2lip_driver_uses_venv_python(fake_home, monkeypatch):
    repo = _install_wav2lip(fake_home)
    monkeypatch.setattr(ls, "_has_torch_cuda", lambda: (True, False))
    calls = []
    out = fake_home / "out.mp4"

    def fake_run(args, **kw):
        calls.append(list(args))
        open(args[args.index("--outfile") + 1], "wb").close()
        return types.SimpleNamespace(returncode=0, stderr=b"")

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    res = ls.wav2lip_sync("v.mp4", "a.wav", out_path=str(out))
    assert res.path == str(out)
    assert calls[0][0] == str(repo / ".venv" / "bin" / "python")
    assert calls[0][1] == "inference.py"


# ── tool action ──────────────────────────────────────────────────────

def test_lip_sync_provision_tool_registered_and_status(fake_home):
    from nomorals.tools import directed as mod
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry()
    mod.register(reg)
    assert "lip_sync_provision" in reg.names()
    out = reg.call("lip_sync_provision", rest="status")
    assert out.ok is True
    report = out.value["report"]
    assert "latentsync" in report
    assert "wav2lip" in report
    assert "sadtaker" in report
    assert "git clone" not in report


def test_lip_sync_provision_tool_installs_sadtaker(fake_home, monkeypatch):
    fake_run = _FakeRun(fake_home)
    monkeypatch.setattr(ls, "_run", fake_run)
    monkeypatch.setattr(ls, "_measure_box", lambda: dict(GPU_BOX))
    monkeypatch.setattr(ls, "_probe_gpu", lambda: dict(GPU_24))
    from nomorals.tools import directed as mod
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry()
    mod.register(reg)
    out = reg.call("lip_sync_provision", rest="sadtaker")
    assert out.ok is True
    report = out.value["report"]
    assert "installed" in report
    # Step records ("git clone: rc=0") are the install log, not a chore
    # list — the ban is on handing the owner a clone instruction.
    assert "git clone https://github.com" not in report
