"""Tests for the talking-head avatar pipeline (#27). All offline.

LatentSync is never actually run: the subprocess call is mocked and the
repo is faked via LATENTSYNC_REPO pointing at a temp dir containing
scripts/inference.py.
"""

import os
import subprocess
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
    with pytest.raises(NotConfigured, match="git clone"):
        latentsync_repo()


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


def test_echomimic_honest_stub(tmp_path, photo_file, wav_file):
    with pytest.raises(NotConfigured, match="EchoMimicV2"):
        talking_head(photo_file, str(wav_file), backend="echomimic",
                     workdir=tmp_path)


def test_api_honest_stub(tmp_path, photo_file, wav_file):
    with pytest.raises(NotConfigured, match="PRUNAAI_API_KEY"):
        talking_head(photo_file, str(wav_file), backend="api",
                     workdir=tmp_path)


def test_unknown_backend(tmp_path, photo_file, wav_file):
    with pytest.raises(AvatarError, match="unknown avatar backend"):
        talking_head(photo_file, str(wav_file), backend="nope",
                     workdir=tmp_path)


# ── availability / introspection ─────────────────────────────────────

def test_avatar_available_termux(monkeypatch):
    monkeypatch.setenv("NM_PROFILE", "termux")
    ok, reason = avatar_available()
    assert ok is False and "GPU host" in reason
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
    monkeypatch.delenv("NM_PROFILE", raising=False)


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
