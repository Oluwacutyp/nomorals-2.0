"""Tests for the talking-head avatar pipeline (#27). All offline.

LatentSync is never actually run: the subprocess call is mocked and the
repo is faked via LATENTSYNC_REPO pointing at a temp dir containing
scripts/inference.py.
"""

import os
import subprocess
import types as _types
import wave
from pathlib import Path

import pytest

from nomorals.media_edit.avatar import (
    AvatarError,
    NotConfigured,
    avatar_available,
    latentsync_repo,
    list_backends,
    talking_head,
    _audio_duration_s,
    _resolve_audio,
    _run_wav2lip,
)
from nomorals.agents.coremind import _avatar_intent, understand


@pytest.fixture()
def fake_repo(tmp_path, monkeypatch):
    repo = tmp_path / "LatentSync"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "inference.py").write_text("# fake\n")
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints" / "latentsync_unet.pt").write_bytes(b"fake-ckpt")
    monkeypatch.setenv("LATENTSYNC_REPO", str(repo))
    monkeypatch.delenv("LATENTSYNC_PYTHON", raising=False)
    return repo


@pytest.fixture()
def wav_file(tmp_path):
    p = tmp_path / "voice.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1 second of silence
    return p


@pytest.fixture()
def photo_file(tmp_path):
    from PIL import Image
    p = tmp_path / "face.png"
    Image.new("RGB", (256, 256), (200, 150, 100)).save(p)
    return p


# ── repo resolution ──────────────────────────────────────────────────

def test_latentsync_repo_ok(fake_repo):
    assert latentsync_repo() == fake_repo


def test_latentsync_repo_missing_env(monkeypatch):
    monkeypatch.delenv("LATENTSYNC_REPO", raising=False)
    # Superseded 2026-11-15 (talking-head self-provision): the old
    # assertion pinned the manual-step text ("git clone") that the
    # owner's no-manual-chore-lists rule forbids. The message now
    # points at the provision, which installs it herself.
    with pytest.raises(NotConfigured, match="provision"):
        latentsync_repo()
    with pytest.raises(NotConfigured) as excinfo:
        latentsync_repo()
    assert "git clone" not in str(excinfo.value)


def test_latentsync_repo_no_script(tmp_path, monkeypatch):
    repo = tmp_path / "empty"
    repo.mkdir()
    monkeypatch.setenv("LATENTSYNC_REPO", str(repo))
    with pytest.raises(NotConfigured, match="no scripts/inference.py"):
        latentsync_repo()


# ── audio resolution ─────────────────────────────────────────────────

def test_resolve_audio_passthrough(wav_file, tmp_path):
    path, synth = _resolve_audio(str(wav_file), voice=None, workdir=tmp_path)
    assert path == wav_file and synth is False


def test_resolve_audio_empty_text(tmp_path):
    with pytest.raises(AvatarError, match="needs text"):
        _resolve_audio("   ", voice=None, workdir=tmp_path)


def test_resolve_audio_tts(monkeypatch, tmp_path):
    calls = {}

    class FakeTTS:
        def __init__(self, backend="auto"):
            pass

        def speak(self, text, voice_name=None, out_path=""):
            calls["text"] = text
            Path(out_path).write_bytes(b"RIFF" + b"\x00" * 100)
            return {"path": out_path, "backend": "fake"}

    monkeypatch.setattr("nomorals.voice.tts.UniversalTTS", FakeTTS)
    path, synth = _resolve_audio("hello there", voice="me", workdir=tmp_path)
    assert synth is True and path.exists()
    assert calls["text"] == "hello there"


def test_audio_duration_wav(wav_file):
    assert abs(_audio_duration_s(wav_file) - 1.0) < 0.05


# ── talking_head pipeline (mocked subprocess) ────────────────────────

def _mock_run_factory(out_path_holder):
    def fake_run(cmd, **kw):
        # emulate LatentSync writing its output video
        out = Path(cmd[cmd.index("--video_out_path") + 1])
        out.write_bytes(b"fake-mp4-bytes")
        out_path_holder.append(out)
        assert "--video_path" in cmd and "--audio_path" in cmd
        assert "--inference_steps" in cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return fake_run


def test_pipeline_audio_file(fake_repo, monkeypatch, tmp_path, wav_file,
                             photo_file):
    outs = []
    monkeypatch.setattr(subprocess, "run", _mock_run_factory(outs))
    monkeypatch.setattr(
        "nomorals.media_edit.avatar._photo_to_video",
        lambda photo, dur, work: tmp_path / "still.mp4")
    (tmp_path / "still.mp4").write_bytes(b"x")
    monkeypatch.setattr(
        "nomorals.media_edit.avatar._audio_duration_s",
        lambda p: 1.0)
    out = talking_head(photo_file, str(wav_file), workdir=tmp_path)
    assert out.exists() and outs, "subprocess never wrote the video"


def test_pipeline_text_goes_through_tts(fake_repo, monkeypatch, tmp_path,
                                         photo_file, wav_file):
    tts_calls = []

    class FakeTTS:
        def __init__(self, backend="auto"):
            pass

        def speak(self, text, voice_name=None, out_path=""):
            tts_calls.append(text)
            import shutil
            shutil.copy(wav_file, out_path)
            return {"path": out_path, "backend": "fake"}

    monkeypatch.setattr("nomorals.voice.tts.UniversalTTS", FakeTTS)
    outs = []
    monkeypatch.setattr(subprocess, "run", _mock_run_factory(outs))
    monkeypatch.setattr(
        "nomorals.media_edit.avatar._photo_to_video",
        lambda photo, dur, work: tmp_path / "still.mp4")
    (tmp_path / "still.mp4").write_bytes(b"x")
    monkeypatch.setattr(
        "nomorals.media_edit.avatar._audio_duration_s",
        lambda p: 1.0)
    out = talking_head(photo_file, "say hello world", workdir=tmp_path)
    assert tts_calls == ["say hello world"]
    assert out.exists()


def test_missing_checkpoint_fails_closed(tmp_path, monkeypatch, wav_file,
                                         photo_file):
    repo = tmp_path / "LatentSync"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "inference.py").write_text("# fake\n")
    monkeypatch.setenv("LATENTSYNC_REPO", str(repo))
    with pytest.raises(NotConfigured, match="checkpoint not found"):
        talking_head(photo_file, str(wav_file), workdir=tmp_path)


def test_latentsync_nonzero_exit(fake_repo, monkeypatch, tmp_path, wav_file,
                                 photo_file):
    def boom(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "CUDA OOM boom")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(
        "nomorals.media_edit.avatar._photo_to_video",
        lambda photo, dur, work: tmp_path / "still.mp4")
    (tmp_path / "still.mp4").write_bytes(b"x")
    with pytest.raises(AvatarError, match="OOM boom"):
        talking_head(photo_file, str(wav_file), workdir=tmp_path)


def test_wav2lip_driver_passes_checkpoint_path(monkeypatch, tmp_path):
    # Upstream Wav2Lip's inference.py takes --checkpoint_path
    # (required); --checkpoint is not an argument and argparse would
    # reject it. The driver must pass the upstream flag name.
    repo = tmp_path / "Wav2Lip"
    (repo / "checkpoints").mkdir(parents=True)
    (repo / "inference.py").write_text("# fake\n")
    (repo / "checkpoints" / "wav2lip_gan.pth").write_bytes(b"fake-ckpt")
    monkeypatch.setenv("WAV2LIP_REPO", str(repo))
    monkeypatch.delenv("WAV2LIP_CKPT", raising=False)
    monkeypatch.delenv("WAV2LIP_PYTHON", raising=False)
    video = tmp_path / "still.mp4"
    video.write_bytes(b"fake-video")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"fake-audio")
    out = tmp_path / "synced.mp4"
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = list(cmd)
        out.write_bytes(b"fake-mp4-bytes")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _run_wav2lip(video, audio, out) == out
    cmd = seen["cmd"]
    assert cmd[1] == "inference.py"
    # Exact-element membership: "--checkpoint" is a substring of
    # "--checkpoint_path", so only list membership distinguishes them.
    assert "--checkpoint_path" in cmd
    assert "--checkpoint" not in cmd
    assert cmd[cmd.index("--checkpoint_path") + 1] == str(
        repo / "checkpoints" / "wav2lip_gan.pth")
    assert "--face" in cmd and "--audio" in cmd and "--outfile" in cmd


def test_echomimic_honest_stub(tmp_path, photo_file, wav_file):
    # Fixed 2026-11-15 (avatar driver fix): the old message carried a
    # manual-step list ("git clone ...") that the owner's
    # no-manual-chore-lists rule forbids. The stub stays honest (raises,
    # never fakes a video) and reports there is no install path yet.
    with pytest.raises(NotConfigured, match="EchoMimicV2"):
        talking_head(photo_file, str(wav_file), backend="echomimic",
                     workdir=tmp_path)
    with pytest.raises(NotConfigured) as excinfo:
        talking_head(photo_file, str(wav_file), backend="echomimic",
                     workdir=tmp_path)
    msg = str(excinfo.value)
    assert "git clone" not in msg
    assert "export" not in msg
    assert "pip install" not in msg
    assert "no install path" in msg


def test_api_honest_stub(tmp_path, photo_file, wav_file):
    # Fixed 2026-11-15 (avatar api stub): the old message carried a
    # manual-step list (endpoint URL / key env / pricing) that the owner's
    # no-manual-chore-lists rule forbids. The stub stays honest (raises,
    # never fakes a video) and reports there is no install path yet.
    with pytest.raises(NotConfigured, match="PrunaAI"):
        talking_head(photo_file, str(wav_file), backend="api",
                     workdir=tmp_path)
    with pytest.raises(NotConfigured) as excinfo:
        talking_head(photo_file, str(wav_file), backend="api",
                     workdir=tmp_path)
    msg = str(excinfo.value)
    assert "git clone" not in msg
    assert "export" not in msg
    assert "pip install" not in msg
    assert "api.pruna.ai" not in msg
    assert "PRUNAAI_API_KEY" not in msg
    assert "$0.025" not in msg
    assert "no install path" in msg


def test_unknown_backend(tmp_path, photo_file, wav_file):
    with pytest.raises(AvatarError, match="unknown avatar backend"):
        talking_head(photo_file, str(wav_file), backend="nope",
                     workdir=tmp_path)


# ── availability / introspection ─────────────────────────────────────

def test_avatar_available_termux(monkeypatch):
    monkeypatch.setenv("NM_PROFILE", "termux")
    ok, reason = avatar_available()
    assert ok is False and "GPU host" in reason
    # Fixed 2026-11-15 (avatar leftovers): the old reason pointed at the
    # API fallback (backend='api', PrunaAI P-Video) as if it worked — it
    # is not implemented and raises NotConfigured, so the pointer was a
    # dead end. The reason now says plainly that it is not available
    # either, and names no backend as a working fallback.
    assert "backend='api'" not in reason
    assert "not available" in reason
    monkeypatch.delenv("NM_PROFILE", raising=False)


def test_avatar_available_ready(fake_repo, monkeypatch):
    monkeypatch.setenv("NM_PROFILE", "workstation")
    ok, reason = avatar_available()
    assert ok is True
    monkeypatch.delenv("NM_PROFILE", raising=False)


def test_avatar_available_no_repo(monkeypatch):
    monkeypatch.setenv("NM_PROFILE", "workstation")
    monkeypatch.delenv("LATENTSYNC_REPO", raising=False)
    ok, _ = avatar_available()
    assert ok is False
    monkeypatch.delenv("NM_PROFILE", raising=False)


def test_list_backends(fake_repo, monkeypatch):
    monkeypatch.setenv("NM_PROFILE", "workstation")
    backends = {b["name"]: b for b in list_backends()}
    assert backends["latentsync"]["available"] is True
    assert backends["echomimic"]["available"] is False
    assert backends["api"]["available"] is False
    # Fixed 2026-11-15 (avatar leftovers): the api backend's cost was
    # "~$0.025/sec" — an unverified price with no test, doc, or upstream
    # backing, and no consumer of list_backends() reads the cost field.
    # The honest value for an unimplemented, unpriced backend is unknown.
    assert backends["api"]["cost"] == "unknown"
    monkeypatch.delenv("NM_PROFILE", raising=False)


def test_docstring_no_setup_steps():
    # Fixed 2026-11-15 (avatar leftovers): the module docstring said the
    # EchoMimic stub "raises with setup steps" — stale since the 70f74e93
    # rewrite removed its manual-step list. The docstring now carries the
    # honest-stub wording (no install path yet), like the api line.
    import nomorals.media_edit.avatar as avatar_mod

    doc = avatar_mod.__doc__ or ""
    assert "setup steps" not in doc
    assert "no install path" in doc


# ── NL intent ────────────────────────────────────────────────────────

def test_nl_talk_colon():
    it = _avatar_intent("make this photo talk: hello world")
    assert it is not None and it.kind == "avatar"
    assert it.meta["text"] == "hello world"


def test_nl_say():
    it = _avatar_intent("make this photo say good morning everyone")
    assert it is not None and it.meta["text"] == "good morning everyone"


def test_nl_misfires():
    assert _avatar_intent("make this photo talk to me") is None
    assert _avatar_intent("make this photo talk") is None
    assert _avatar_intent("make this photo talk:   ") is None
    assert _avatar_intent("can you make this photo talk") is None
    assert _avatar_intent("talk to me") is None


def test_nl_wired_into_understand():
    intents = understand("make this photo say hi there")
    assert any(i.kind == "avatar" for i in intents)


# ── talking-head self-provision ──────────────────────────────────────
# The provision installs the backends Devon herself (clone + deps +
# checkpoints + wiring), following the /models honest-reporting shape:
# per-backend reports (installed / already-installed / unavailable /
# failed), never raises, never a manual-step list. Every install /
# download is mocked here — no real pip installs, no real weight
# downloads. The seams are module-level functions on
# ``nomorals.media_edit.avatar`` (``_run``, ``_measure_box``,
# ``_probe_gpu``, ``_download_url``, ``latentsync_dir``,
# ``wav2lip_dir``) patched per test.

CPU_BOX = {"ok": True, "total_ram_gb": 8.0, "free_ram_gb": 5.0,
           "free_disk_gb": 40.0, "cpus": 4, "platform": "linux",
           "profile": "laptop", "detail": "cpu box"}
GPU_BOX = {"ok": True, "total_ram_gb": 32.0, "free_ram_gb": 20.0,
           "free_disk_gb": 120.0, "cpus": 8, "platform": "linux",
           "profile": "workstation", "detail": "gpu box"}
TINY_BOX = {"ok": True, "total_ram_gb": 0.5, "free_ram_gb": 0.4,
            "free_disk_gb": 0.5, "cpus": 1, "platform": "linux",
            "profile": "laptop", "detail": "tiny box"}
TERMUX_BOX = {"ok": True, "total_ram_gb": 4.0, "free_ram_gb": 2.0,
              "free_disk_gb": 20.0, "cpus": 8, "platform": "linux",
              "profile": "termux", "detail": "termux box"}
NO_GPU = {"has_gpu": False, "name": "", "vram_gb": 0.0, "source": "",
          "detail": "no NVIDIA GPU detected"}
GPU_24 = {"has_gpu": True, "name": "RTX 4090", "vram_gb": 24.0,
          "source": "nvidia-smi", "detail": "RTX 4090 with 24.0 GB VRAM"}


def _avatar_mod():
    import nomorals.media_edit.avatar as av

    return av


def _clear_avatar_env(monkeypatch):
    for key in ("LATENTSYNC_REPO", "LATENTSYNC_PYTHON",
                "LATENTSYNC_UNET_CONFIG", "LATENTSYNC_CKPT",
                "LATENTSYNC_STEPS", "LATENTSYNC_GUIDANCE",
                "WAV2LIP_REPO", "WAV2LIP_PYTHON", "WAV2LIP_CKPT"):
        monkeypatch.delenv(key, raising=False)


class _FakeProvisionRun:
    """Simulates the provision's subprocesses by creating the files
    each step would create. Records every call."""

    def __init__(self, ls_root, w2l_root):
        self.ls_root = Path(ls_root)
        self.w2l_root = Path(w2l_root)
        self.calls = []

    def __call__(self, args, timeout=None, cwd=None):
        self.calls.append(list(args))
        joined = " ".join(str(a) for a in args)
        if args and args[0] == "git":
            dest = Path(str(args[-1]))
            if "LatentSync" in joined:
                (dest / "scripts").mkdir(parents=True, exist_ok=True)
                (dest / "scripts" / "inference.py").write_text("# fake\n")
                (dest / "configs" / "unet").mkdir(parents=True, exist_ok=True)
                (dest / "configs" / "unet" / "stage2_512.yaml").write_text(
                    "model:\n  cross_attention_dim: 384\n")
                (dest / "requirements.txt").write_text("torch==2.5.1\n")
            else:
                (dest / "face_detection" / "detection" / "sfd").mkdir(
                    parents=True, exist_ok=True)
                (dest / "inference.py").write_text("# fake\n")
                (dest / "requirements.txt").write_text("torch==1.1.0\n")
            return _types.SimpleNamespace(returncode=0, stdout="ok\n",
                                         stderr="")
        if len(args) >= 3 and args[1:3] == ["-m", "venv"]:
            venv = Path(str(args[3]))
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "python").write_text("# fake venv python\n")
            return _types.SimpleNamespace(returncode=0, stdout="ok\n",
                                         stderr="")
        if "-c" in args:
            snippet = str(args[args.index("-c") + 1])
            if "import huggingface_hub" in snippet:
                return _types.SimpleNamespace(returncode=1, stdout="",
                                             stderr="No module named 'huggingface_hub'")
            if "import gdown" in snippet:
                return _types.SimpleNamespace(returncode=1, stdout="",
                                             stderr="No module named 'gdown'")
            if "snapshot_download" in snippet and "LatentSync" in snippet:
                ck = self.ls_root / "checkpoints"
                (ck / "whisper").mkdir(parents=True, exist_ok=True)
                (ck / "latentsync_unet.pt").write_bytes(b"fake-unet")
                (ck / "whisper" / "tiny.pt").write_bytes(b"fake-tiny")
                return _types.SimpleNamespace(returncode=0, stdout="ok\n",
                                             stderr="")
            return _types.SimpleNamespace(returncode=0, stdout="ok\n",
                                         stderr="")
        if len(args) >= 3 and args[1:3] == ["-m", "gdown"]:
            dest = Path(str(args[args.index("-O") + 1]))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake-gan")
            return _types.SimpleNamespace(returncode=0, stdout="ok\n",
                                         stderr="")
        return _types.SimpleNamespace(returncode=0, stdout="ok\n",
                                     stderr="")


def _fake_download_url(url, dest):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(b"fake-s3fd")
    return Path(dest)


def _patch_provision(monkeypatch, tmp_path, box, gpu, run=None):
    av = _avatar_mod()
    ls_root = tmp_path / "LatentSync"
    w2l_root = tmp_path / "Wav2Lip"
    monkeypatch.setattr(av, "latentsync_dir", lambda: ls_root)
    monkeypatch.setattr(av, "wav2lip_dir", lambda: w2l_root)
    monkeypatch.setattr(av, "_measure_box", lambda: dict(box))
    monkeypatch.setattr(av, "_probe_gpu", lambda: dict(gpu))
    monkeypatch.setattr(av, "_download_url", _fake_download_url)
    fake = run if run is not None else _FakeProvisionRun(ls_root, w2l_root)
    monkeypatch.setattr(av, "_run", fake)
    _clear_avatar_env(monkeypatch)
    return av, ls_root, w2l_root, fake


def test_provision_installs_wav2lip_refuses_latentsync_on_cpu_box(
        monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, CPU_BOX, NO_GPU)
    result = av.provision_backends()
    assert result["backends"]["latentsync"]["status"] == "unavailable"
    assert "18" in result["backends"]["latentsync"]["detail"]
    assert "no NVIDIA GPU" in result["backends"]["latentsync"]["detail"]
    assert result["backends"]["wav2lip"]["status"] == "installed"
    # The refused backend was never cloned; the fitting one was.
    assert not any("bytedance/LatentSync" in " ".join(c)
                   for c in fake.calls)
    assert any("Rudrabha/Wav2Lip" in " ".join(c) for c in fake.calls)
    # The provisioned paths are wired: the resolvers find them with
    # no env var set.
    assert av.wav2lip_repo() == w2l_root
    assert (w2l_root / "checkpoints" / "wav2lip_gan.pth").exists()
    assert (w2l_root / "face_detection" / "detection" / "sfd"
            / "s3fd.pth").exists()


def test_provision_installs_both_on_gpu_box(monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, GPU_BOX, GPU_24)
    result = av.provision_backends()
    assert result["ok"] is True
    assert result["backends"]["latentsync"]["status"] == "installed"
    assert result["backends"]["wav2lip"]["status"] == "installed"
    assert av.latentsync_repo() == ls_root
    assert av.wav2lip_repo() == w2l_root
    assert (ls_root / "checkpoints" / "latentsync_unet.pt").exists()
    assert (ls_root / "checkpoints" / "whisper" / "tiny.pt").exists()
    assert "LatentSync" in av.provision_report_text(result)


def test_provision_skips_already_installed(monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, GPU_BOX, GPU_24)
    for root, script in ((ls_root, "scripts/inference.py"),
                         (w2l_root, "inference.py")):
        (root / script).parent.mkdir(parents=True, exist_ok=True)
        (root / script).write_text("# fake\n")
    (ls_root / "checkpoints" / "whisper").mkdir(parents=True, exist_ok=True)
    (ls_root / "checkpoints" / "latentsync_unet.pt").write_bytes(b"x")
    (ls_root / "checkpoints" / "whisper" / "tiny.pt").write_bytes(b"x")
    (w2l_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (w2l_root / "checkpoints" / "wav2lip_gan.pth").write_bytes(b"x")
    (w2l_root / "face_detection" / "detection" / "sfd").mkdir(
        parents=True, exist_ok=True)
    (w2l_root / "face_detection" / "detection" / "sfd" / "s3fd.pth"
     ).write_bytes(b"x")
    result = av.provision_backends()
    assert result["backends"]["latentsync"]["status"] == "already-installed"
    assert result["backends"]["wav2lip"]["status"] == "already-installed"
    assert fake.calls == []


def test_provision_latentsync_refuses_without_gpu_with_numbers(
        monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, CPU_BOX, NO_GPU)
    result = av.provision_backends(["latentsync"])
    rep = result["backends"]["latentsync"]
    assert rep["status"] == "unavailable" and rep["ok"] is False
    assert "18" in rep["detail"] and "0" in rep["detail"]
    assert fake.calls == []


def test_provision_refuses_on_tiny_box(monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, TINY_BOX, NO_GPU)
    result = av.provision_backends()
    for name in ("latentsync", "wav2lip"):
        rep = result["backends"][name]
        assert rep["status"] == "unavailable" and rep["ok"] is False
        assert "GB" in rep["detail"]
    assert fake.calls == []


def test_provision_refuses_on_termux(monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, TERMUX_BOX, NO_GPU)
    result = av.provision_backends()
    for name in ("latentsync", "wav2lip"):
        rep = result["backends"][name]
        assert rep["status"] == "unavailable" and rep["ok"] is False
        assert "termux" in rep["detail"].lower()
    assert fake.calls == []


def test_provision_never_emits_manual_steps(monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, CPU_BOX, NO_GPU)
    result = av.provision_backends()
    text = av.provision_report_text(result)
    assert "export LATENTSYNC" not in text
    assert "export WAV2LIP" not in text
    assert "Set it up" not in text
    assert "cd LatentSync" not in text
    assert "cd Wav2Lip" not in text
    # The NotConfigured messages point at the provision, not a chore
    # list for the owner.
    for setup in (av.LATENTSYNC_SETUP, av.WAV2LIP_SETUP):
        assert "git clone" not in setup
        assert "export" not in setup
        assert "pip install" not in setup
        assert "provision" in setup.lower()
    with pytest.raises(NotConfigured, match="provision"):
        av.latentsync_repo()
    with pytest.raises(NotConfigured) as excinfo:
        av.latentsync_repo()
    assert "git clone" not in str(excinfo.value)
    assert "export" not in str(excinfo.value)


def test_resolvers_default_to_canonical_dirs(monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, CPU_BOX, NO_GPU)
    (ls_root / "scripts").mkdir(parents=True)
    (ls_root / "scripts" / "inference.py").write_text("# fake\n")
    (ls_root / "checkpoints").mkdir()
    (ls_root / "checkpoints" / "latentsync_unet.pt").write_bytes(b"x")
    (ls_root / ".venv" / "bin").mkdir(parents=True)
    (ls_root / ".venv" / "bin" / "python").write_text("# fake\n")
    assert av.latentsync_repo() == ls_root
    assert av.latentsync_python() == str(ls_root / ".venv" / "bin"
                                         / "python")
    # Wav2Lip has no layout here: the resolver raises, pointing at
    # the provision rather than a manual-step list.
    with pytest.raises(NotConfigured, match="provision"):
        av.wav2lip_repo()


def test_provision_single_backend_only(monkeypatch, tmp_path):
    av, ls_root, w2l_root, fake = _patch_provision(
        monkeypatch, tmp_path, CPU_BOX, NO_GPU)
    result = av.provision_backends(["wav2lip"])
    assert set(result["backends"]) == {"wav2lip"}
    assert result["backends"]["wav2lip"]["status"] == "installed"
    assert not any("LatentSync" in " ".join(c) for c in fake.calls)


# ── the agent provisions instead of handing over the manual list ───

def _fake_mind(tmp_path, monkeypatch):
    from nomorals.agents.coremind import CoreMind

    mind = CoreMind.__new__(CoreMind)
    mind.runtime = None

    class _Settings:
        def resolve(self, relative):
            return tmp_path / relative

    mind.context = _types.SimpleNamespace(
        settings=_Settings(), db=None)
    mind._send_async = lambda chat_key, job, job_id, note, kind="job": job()
    return mind


def _avatar_message(photo_path):
    return _types.SimpleNamespace(
        media=[_types.SimpleNamespace(kind="image", path=str(photo_path))])


def test_avatar_job_provisions_then_renders_with_wav2lip(
        monkeypatch, tmp_path, photo_file):
    av = _avatar_mod()
    calls = {"n": 0}

    def fake_talking_head(photo, audio_or_text, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise NotConfigured("LatentSync is not installed")
        assert kw.get("backend") == "wav2lip"
        return tmp_path / "talking-head.mp4"

    monkeypatch.setattr(av, "talking_head", fake_talking_head)
    monkeypatch.setattr(
        av, "provision_backends",
        lambda backends=None, **kw: {
            "ok": False,
            "box": dict(CPU_BOX),
            "backends": {
                "latentsync": {"ok": False, "status": "unavailable",
                               "detail": "no NVIDIA GPU detected",
                               "steps": []},
                "wav2lip": {"ok": True, "status": "installed",
                            "detail": "Wav2Lip installed",
                            "steps": []}}})
    monkeypatch.setattr(
        av, "provision_report_text",
        lambda result: "provision report")
    mind = _fake_mind(tmp_path, monkeypatch)
    intent = _avatar_intent("make this photo talk: hello world")
    out = mind._dispatch_avatar(intent, "job-1", "tg:1",
                                _avatar_message(photo_file))
    assert out.startswith("🗣️ saved to")
    assert calls["n"] == 2


def test_avatar_job_reports_provision_refusal_plainly(
        monkeypatch, tmp_path, photo_file):
    av = _avatar_mod()

    def always_missing(photo, audio_or_text, **kw):
        raise NotConfigured("LatentSync is not installed")

    monkeypatch.setattr(av, "talking_head", always_missing)
    monkeypatch.setattr(
        av, "provision_backends",
        lambda backends=None, **kw: {
            "ok": False,
            "box": dict(CPU_BOX),
            "backends": {
                "latentsync": {"ok": False, "status": "unavailable",
                               "detail": "no NVIDIA GPU detected (0 GB VRAM); needs 18 GB",
                               "steps": []},
                "wav2lip": {"ok": False, "status": "failed",
                            "detail": "Wav2Lip clone failed",
                            "steps": []}}})
    monkeypatch.setattr(
        av, "provision_report_text",
        lambda result: ("🗣️ talking-head backends — provision\n"
                        "• latentsync: unavailable — no NVIDIA GPU detected "
                        "(0 GB VRAM); needs 18 GB\n"
                        "• wav2lip: failed — Wav2Lip clone failed"))
    mind = _fake_mind(tmp_path, monkeypatch)
    intent = _avatar_intent("make this photo talk: hello world")
    out = mind._dispatch_avatar(intent, "job-1", "tg:1",
                                _avatar_message(photo_file))
    assert "unavailable" in out and "18 GB" in out
    assert "git clone" not in out
    assert "export" not in out
