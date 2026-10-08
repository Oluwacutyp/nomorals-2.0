"""Offline tests for the unified voice pipeline (#108). All mock stages — no audio, no STT/TTS."""

import os
import tempfile
import wave

import pytest

from nomorals.audio.pipeline import (
    STAGES,
    EngineRegistry,
    KeytermStore,
    PipelineConfig,
    VoicePipeline,
    control_voice,
    detect_profile,
    run_pipeline,
    transcribe_clean,
)


def _make_wav(path, seconds=1.0, rate=16000):
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)
    return path


@pytest.fixture()
def wav_file():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "note.wav")
    return _make_wav(p)


@pytest.fixture()
def keyterms():
    ks = KeytermStore(db_path=os.path.join(tempfile.mkdtemp(), "k.db"))
    ks.add("Adaeze", kind="name")
    ks.add("ẹẹm", kind="dialect", lang="yo-ekiti")
    ks.add("agbada", kind="jargon")
    return ks


# -- engine registry ----------------------------------------------------------

def test_stages_cover_pipeline():
    assert STAGES == ("denoise", "transcribe", "diarize",
                      "edit", "synthesize", "master")


def test_detect_profile_termux(monkeypatch):
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.delenv("NM_PROFILE", raising=False)
    assert detect_profile() == "termux"


def test_detect_profile_env_override(monkeypatch):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setenv("NM_PROFILE", "workstation")
    assert detect_profile() == "workstation"


def test_registry_profile_orders():
    termux = EngineRegistry(profile="termux")
    laptop = EngineRegistry(profile="laptop")
    # termux prefers light STT
    assert termux.stt_candidates()[0] == "parakeet-onnx"
    assert laptop.stt_candidates()[0] == "faster-whisper"
    # termux TTS has no xtts; laptop private does
    assert "xtts" not in termux.tts_candidates("private")
    assert "xtts" in laptop.tts_candidates("private")


def test_audience_routing_structural():
    reg = EngineRegistry(profile="workstation")
    private = reg.tts_candidates("private")
    public = reg.tts_candidates("public")
    assert "xtts" in private
    assert "xtts" not in public  # structural, not advisory
    assert public  # but public still has options


def test_pick_tts_respects_audience():
    reg = EngineRegistry(profile="laptop")
    avail = ["xtts", "chatterbox"]
    assert reg.pick_tts("private", avail) == "xtts"
    assert reg.pick_tts("public", avail) == "chatterbox"


def test_pick_stt_best_available():
    reg = EngineRegistry(profile="termux")
    assert reg.pick_stt(["faster-whisper", "whisper-cpp"]) == "whisper-cpp"
    assert reg.pick_stt([]) is None
    assert reg.pick_stt(None) is None


def test_registry_describe_shape():
    d = EngineRegistry(profile="pc").describe()
    assert d["profile"] == "pc"
    assert "xtts" in d["noncommercial_excluded_public"]


# -- keyterms -----------------------------------------------------------------

def test_keyterm_crud(keyterms):
    assert len(keyterms.list()) == 3
    assert keyterms.remove("Adaeze")
    assert len(keyterms.list()) == 2
    assert not keyterms.remove("nobody")
    assert not keyterms.add("")  # bad input rejected


def test_keyterm_kinds(keyterms):
    names = keyterms.list(kind="name")
    assert len(names) == 1 and names[0]["term"] == "Adaeze"
    assert any(t["kind"] == "dialect" for t in keyterms.list())


def test_initial_prompt_grouped(keyterms):
    prompt = keyterms.initial_prompt()
    assert "Adaeze" in prompt
    assert "agbada" in prompt
    assert "Names" in prompt or "name" in prompt.lower()


def test_initial_prompt_empty():
    ks = KeytermStore(db_path=os.path.join(tempfile.mkdtemp(), "e.db"))
    assert ks.initial_prompt() == ""


def test_keyterm_never_raises():
    # a regular file in the way of the db path → sqlite cannot open it
    blocker = os.path.join(tempfile.mkdtemp(), "blocker")
    with open(blocker, "w") as f:
        f.write("x")
    ks = KeytermStore(db_path=os.path.join(blocker, "k.db"))
    assert ks.add("x") is False
    assert ks.list() == []
    assert ks.initial_prompt() == ""


# -- pipeline -----------------------------------------------------------------

def _mock_transcribe(audio, **kw):
    return {"ok": True, "audio": audio, "text": "hello world",
            "language": "en", "words": [], "keyterm_prompt_used": True}


def test_pipeline_stage_registry(wav_file):
    p = VoicePipeline(profile="pc")
    assert set(p.stage_names()) == set(STAGES)
    assert p.register_stage("denoise", lambda audio, **kw: {"ok": True, "audio": audio})
    assert not p.register_stage("nope", lambda: {})
    assert not p.register_stage("denoise", "not-callable")


def test_pipeline_missing_audio():
    p = VoicePipeline(profile="pc")
    r = p.run("/no/such/file.wav", {"transcribe": True})
    assert r["ok"] is False
    assert r["stage"] == "input"


def test_pipeline_transcribe_only(wav_file):
    p = VoicePipeline(profile="pc")
    p.register_stage("transcribe", _mock_transcribe)
    p.register_stage("denoise", lambda audio, **kw: {"ok": True, "audio": audio})
    r = p.run(wav_file, {"transcribe": True, "denoise": True,
                         "master": False})
    assert r["ok"] is True
    assert r["transcript"] == "hello world"
    assert "transcribe" in r["stages"]
    assert r["profile"] == "pc"


def test_pipeline_transcribe_failure_honest(wav_file):
    p = VoicePipeline(profile="pc")
    p.register_stage("transcribe",
                     lambda audio, **kw: {"ok": False, "stage": "transcribe",
                                          "reason": "no backend"})
    r = p.run(wav_file, {"transcribe": True, "denoise": False,
                         "master": False})
    assert r["ok"] is False
    assert r["stage"] == "transcribe"


def test_pipeline_full_mock(wav_file):
    p = VoicePipeline(profile="laptop")
    p.register_stage("denoise", lambda audio, **kw: {"ok": True, "audio": audio})
    p.register_stage("transcribe", _mock_transcribe)
    p.register_stage("diarize", lambda audio, **kw: {
        "ok": True, "speakers": ["A"], "segments": [], "method": "mock"})
    p.register_stage("edit", lambda audio, **kw: {
        "ok": True, "audio": audio, "notes": ["fillers cut"]})
    p.register_stage("synthesize", lambda audio, **kw: {
        "ok": True, "audio": audio, "audience": kw.get("audience")})
    p.register_stage("master", lambda audio, **kw: {
        "ok": True, "audio": audio, "lufs": -16.0})
    r = p.run(wav_file, PipelineConfig(
        transcribe=True, diarize=True, remove_fillers=True,
        synthesize=True, segments=[{"speaker": "A", "text": "hi"}],
        master=True, audience="private"))
    assert r["ok"] is True
    assert len(r["stages"]) == 6
    assert r["stages"]["synthesize"]["audience"] == "private"


def test_synthesize_public_blocks_xtts():
    from nomorals.audio.pipeline import _stage_synthesize
    r = _stage_synthesize(
        "/tmp/x.wav",
        segments=[{"speaker": "A", "text": "hello"}],
        audience="public", tts_backend="xtts")
    assert r["ok"] is False
    assert "non-commercial" in r["reason"]


def test_synthesize_private_allows_xtts_path():
    from nomorals.audio.pipeline import _stage_synthesize
    # No segments → honest refusal, not a backend complaint
    r = _stage_synthesize("/tmp/x.wav", segments=[], audience="private")
    assert r["ok"] is False
    assert "nothing to speak" in r["reason"]


def test_pipeline_config_from_dict_garbage():
    cfg = PipelineConfig.from_dict(None)
    assert cfg.lang == "en" and cfg.audience == "private"
    cfg2 = PipelineConfig.from_dict({"audience": "public", "target_lufs": "-14"})
    assert cfg2.audience == "public" and cfg2.target_lufs == -14.0


def test_run_pipeline_convenience(wav_file):
    p_called = []

    class FakePipe(VoicePipeline):
        def run(self, audio, config=None):
            p_called.append(audio)
            return {"ok": True, "audio": audio}

    # convenience goes through the real constructor; patch via monkeypatch
    import nomorals.audio.pipeline as mod
    real = mod.VoicePipeline
    mod.VoicePipeline = FakePipe
    try:
        r = run_pipeline(wav_file, {"transcribe": True})
        assert r["ok"] is True and p_called == [wav_file]
    finally:
        mod.VoicePipeline = real


def test_transcribe_clean_convenience(wav_file, monkeypatch):
    import nomorals.audio.pipeline as mod
    real = mod.run_pipeline

    def fake(audio, config=None, **kw):
        return {"ok": True, "audio": audio, "config": config}

    monkeypatch.setattr(mod, "run_pipeline", fake)
    r = transcribe_clean(wav_file, lang="yo")
    assert r["ok"] is True
    assert r["config"]["denoise"] is True
    assert r["config"]["synthesize"] is False


def test_pipeline_never_raises():
    p = VoicePipeline(profile="pc")
    r = p.run(None, None)  # type: ignore
    assert r["ok"] is False


# -- chat ---------------------------------------------------------------------

def test_chat_engines():
    out = control_voice("engines")
    assert "profile" in out
    assert "XTTS excluded" in out


def test_chat_keyterms_add_list_remove():
    out = control_voice("keyterms add TestNameXYZ name")
    assert "added" in out
    out = control_voice("keyterms list")
    assert "TestNameXYZ" in out
    out = control_voice("keyterms remove TestNameXYZ")
    assert "removed" in out


def test_chat_keyterms_empty_list(monkeypatch):
    import nomorals.audio.pipeline as mod
    real = mod.KeytermStore

    class Empty(mod.KeytermStore):
        def list(self, kind="", lang=""):
            return []

    monkeypatch.setattr(mod, "KeytermStore", Empty)
    out = control_voice("keyterms list")
    assert "no keyterms" in out


def test_chat_pipeline_missing_file():
    out = control_voice("pipeline /no/such.wav")
    assert "no such audio" in out or "stopped" in out


def test_chat_usage():
    out = control_voice("")
    assert "pipeline" in out and "engines" in out


def test_chat_garbage_never_raises():
    out = control_voice("frobnicate \x00\xff")
    assert isinstance(out, str)
