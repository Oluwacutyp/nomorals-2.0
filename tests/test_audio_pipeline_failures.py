"""Pass #160 pins: the audio failure ledger (REUSED from #157) + pipeline conversions.

``nomorals/audio/failures.py`` (namespace ``audio``) already exists from
#157; this pass reuses it and converts 23 silent swallows in
``nomorals/audio/pipeline.py`` (1 exclusion: the pyannote-import probe).
The 17 durability tests below mirror #157's shape exactly; the 23 site
pins prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name); the exclusion pin
proves the honest token-missing outcome AND no ledger entry. No pin
hardcodes scan totals or pipeline counts. Every test redirects stores
with the app's own ``NM_HOME`` override (never the real home, never the
repo tree). Restart tests use real subprocesses. Every trigger was
probed live before its pin was written. VoicePipeline/KeytermStore
objects under test are built via ``__new__`` or explicit tmp db paths —
never the real-home default store.
"""
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.audio import failures as af
from nomorals.audio import pipeline
from nomorals.core import failure_store as fs


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in af.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #157) -----------------------------

def test_store_round_trip():
    store = fs.Store("audio")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("audio").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("audio")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("audio").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("audio")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("audio").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("audio").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("audio").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("audio")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("audio").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("audio").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("audio").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("audio").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("audio")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("audio").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        af.record_failure("no-home", ValueError("boom"))
        assert af.failure_count() == 1
        assert [f.site for f in af.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert af.failure_count() == 0
    assert af.recent_failures() == []
    assert af.format_failures() == "no audio failures recorded."
    af.record_failure("audio.site-a", ValueError("boom"))
    af.record_failure("audio.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == ["audio.site-b", "audio.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["audio.site-b"]
    text = af.format_failures()
    assert "2 total" in text and "audio.site-a" in text and "audio.site-b" in text
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("audio.disk", ValueError("boom"))
    entries, total = fs.Store("audio").load()
    assert total == 1 and entries[0]["site"] == "audio.disk"
    af.clear_failures()
    assert fs.Store("audio").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("audio.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["audio.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"audio.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert af.failure_count() == 400
    assert len(af.recent_failures(1000)) == 200
    entries, total = af._store.load()
    assert len(entries) == 200
    assert total == 400


# ---- Restart (subprocess) ------------------------------------------------

def _run_subprocess(home: Path, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.audio import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.audio import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.audio import failures as f\n"
               "f.clear_failures()\nprint('ok')\n")


def test_restart_record_survives(tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


def test_restart_clear_is_gone(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    assert _run_subprocess(home, _CLEAR_CODE).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


def test_restart_total_preserved_and_cap_holds(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(250)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("audio").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Converted-site behavior: fallback unchanged + recorded -------------

def _env_get(script):
    """os.environ.get stand-in with scripted behavior.

    Patches only the ``get`` method (never os.environ itself — pytest
    internals read os.environ['COLUMNS'] during the run).
    """

    def _get(key, default=""):
        for k, v in script:
            if k == key:
                if isinstance(v, Exception):
                    raise v
                return v
        return default

    return _get


class _BadIter:
    def __iter__(self):
        raise RuntimeError("boom")


class _BadGet:
    def get(self, *args, **kwargs):
        raise RuntimeError("boom")


class _BadLower:
    def lower(self):
        raise RuntimeError("boom")


class _BadProfile:
    @property
    def profile(self):
        raise RuntimeError("boom")


class _Ctx:
    def __init__(self, settings):
        self.settings = settings


def _keyterm_store(tmp_path):
    store = pipeline.KeytermStore(db_path=str(tmp_path / "k.db"))
    assert store._db is not None
    return store


class _BoomConn:
    def execute(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_converted_detect_profile_settings_records_and_falls_through(monkeypatch):
    monkeypatch.setattr(pipeline.os.environ, "get", _env_get([]))
    assert pipeline.detect_profile(_Ctx(_BadProfile())) == "pc"
    assert _sites() == ["pipeline.detect_profile.settings"]


def test_converted_detect_profile_termux_env_records_and_falls_through(monkeypatch):
    monkeypatch.setattr(
        pipeline.os.environ, "get",
        _env_get([("PREFIX", RuntimeError("boom"))]))
    assert pipeline.detect_profile(None) == "pc"
    assert _sites() == ["pipeline.detect_profile.termux_env"]


def test_converted_detect_profile_nm_profile_records_and_falls_through(monkeypatch):
    monkeypatch.setattr(
        pipeline.os.environ, "get",
        _env_get([("PREFIX", ""), ("NM_PROFILE", RuntimeError("boom"))]))
    assert pipeline.detect_profile(None) == "pc"
    assert _sites() == ["pipeline.detect_profile.nm_profile"]


def test_converted_pick_stt_records_and_returns_none():
    reg = pipeline.EngineRegistry("pc")
    assert reg.pick_stt(_BadIter()) is None
    assert _sites() == ["pipeline.pick_stt"]


def test_converted_tts_candidates_records_and_returns_empty():
    reg = pipeline.EngineRegistry("pc")
    assert reg.tts_candidates(_BadLower()) == []
    assert _sites() == ["pipeline.tts_candidates"]


def test_converted_pick_tts_records_and_returns_none():
    reg = pipeline.EngineRegistry("pc")
    assert reg.pick_tts("private", _BadIter()) is None
    assert _sites() == ["pipeline.pick_tts"]


def test_converted_default_keyterm_db_records_and_still_returns_path(monkeypatch):
    def _boom_mkdir(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(Path, "mkdir", _boom_mkdir)
    out = pipeline._default_keyterm_db()
    assert out.endswith("keyterms.db")
    assert _sites() == ["pipeline.default_keyterm_db.mkdir"]


def test_converted_keyterm_add_records_and_returns_false(tmp_path):
    store = _keyterm_store(tmp_path)
    store._db = _BoomConn()
    assert store.add("hello") is False
    assert _sites() == ["pipeline.keyterm_add"]


def test_converted_keyterm_remove_records_and_returns_false(tmp_path):
    store = _keyterm_store(tmp_path)
    store._db = _BoomConn()
    assert store.remove("hello") is False
    assert _sites() == ["pipeline.keyterm_remove"]


def test_converted_keyterm_list_records_and_returns_empty(tmp_path):
    store = _keyterm_store(tmp_path)
    store._db = _BoomConn()
    assert store.list() == []
    assert _sites() == ["pipeline.keyterm_list"]


def test_converted_keyterm_prompt_records_and_returns_empty(tmp_path, monkeypatch):
    store = _keyterm_store(tmp_path)
    monkeypatch.setattr(store, "list", _boom)
    assert store.initial_prompt() == ""
    assert _sites() == ["pipeline.keyterm_prompt"]


class _FakeTranscript:
    text = "hi"
    language = "en"
    words = []


def _stub_edit_with_transcriber(audio, language=None, enhance=None,
                                transcriber=None, **kw):
    if transcriber is not None:
        transcriber(audio, language or "en")
    return _FakeTranscript()


def _fake_stt_module():
    # NOTE: the backend must instantiate cleanly — _prompted instantiates
    # it BEFORE the wav-extraction step, so a raising __init__ would
    # short-circuit past the 328 extraction site. Failure belongs at
    # transcribe time, where the outer handler catches it.
    mod = types.ModuleType("nomorals.voice.stt")

    class _BoomTranscribe:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("no transcribe")

    class _FakeBackend:
        def __init__(self, *args, **kwargs):
            self.model = _BoomTranscribe()

    mod.FasterWhisperBackend = _FakeBackend
    return mod


class _Keyterms:
    def initial_prompt(self, lang=""):
        return "Names: Ada."


def test_converted_transcribe_extract_import_records_and_transcribes(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.media_edit.captions", None)
    monkeypatch.setitem(sys.modules, "nomorals.voice.stt", _fake_stt_module())
    import nomorals.audio.edit as edit_mod
    monkeypatch.setattr(edit_mod, "transcript_edit",
                        _stub_edit_with_transcriber)
    res = pipeline._stage_transcribe("x.wav", lang="en", keyterms=_Keyterms())
    assert res["ok"] is True and res["text"] == "hi"
    assert _sites() == ["pipeline.transcribe_extract_import"]


def test_converted_transcribe_extract_wav_records_and_transcribes(monkeypatch):
    fake_cap = types.ModuleType("nomorals.media_edit.captions")

    def _boom_wav(p):
        raise RuntimeError("boom")

    fake_cap._extract_wav = _boom_wav
    monkeypatch.setitem(sys.modules, "nomorals.media_edit.captions", fake_cap)
    monkeypatch.setitem(sys.modules, "nomorals.voice.stt", _fake_stt_module())
    import nomorals.audio.edit as edit_mod
    monkeypatch.setattr(edit_mod, "transcript_edit",
                        _stub_edit_with_transcriber)
    res = pipeline._stage_transcribe("x.wav", lang="en", keyterms=_Keyterms())
    assert res["ok"] is True and res["text"] == "hi"
    assert _sites() == ["pipeline.transcribe_extract_wav"]


def test_converted_diarize_kwargs_fallback_records_and_still_diarizes(monkeypatch):
    monkeypatch.setenv("WHISPERX_HF_TOKEN", "test-token")
    fake_pa = types.ModuleType("pyannote.audio")

    class _Turn:
        start = 0.0
        end = 1.0

    class _FakeDiar:
        def itertracks(self, yield_label=True):
            return [(_Turn(), None, "SPEAKER_00")]

    class _FakePipeline:
        def __call__(self, audio, min_speakers=None, max_speakers=None):
            if min_speakers is not None or max_speakers is not None:
                raise RuntimeError("boom")
            return _FakeDiar()

    class _FakePipelineCls:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakePipeline()

    fake_pa.Pipeline = _FakePipelineCls
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pa)
    res = pipeline.diarize_real("x.wav", [], min_speakers=1, max_speakers=2)
    assert res["ok"] is True
    assert _sites() == ["pipeline.diarize_kwargs_fallback"]


def test_converted_group_turns_records_and_returns_empty():
    assert pipeline.group_turns(_BadIter()) == []
    assert _sites() == ["pipeline.group_turns"]


def test_converted_srt_records_and_returns_empty():
    assert pipeline.transcript_to_srt([_BadGet()]) == ""
    assert _sites() == ["pipeline.srt"]


def test_converted_vtt_records_and_returns_empty():
    assert pipeline.transcript_to_vtt([_BadGet()]) == ""
    assert _sites() == ["pipeline.vtt"]


def test_converted_minutes_records_and_returns_empty():
    assert pipeline.meeting_minutes([_BadGet()]) == ""
    assert _sites() == ["pipeline.minutes"]


def test_converted_config_from_dict_records_and_returns_defaults():
    cfg = pipeline.PipelineConfig.from_dict({"min_speakers": "not-a-number"})
    assert cfg == pipeline.PipelineConfig()
    assert _sites() == ["pipeline.config_from_dict"]


def test_converted_register_stage_records_and_returns_false():
    class _BadName:
        def __eq__(self, other):
            raise RuntimeError("boom")

    pipe = pipeline.VoicePipeline.__new__(pipeline.VoicePipeline)
    pipe._stages = dict(pipeline._DEFAULT_STAGES)
    assert pipe.register_stage(_BadName(), lambda: {}) is False
    assert _sites() == ["pipeline.register_stage"]


def test_converted_run_workdir_mkdir_records_and_still_runs(tmp_path, monkeypatch):
    def _boom_mkdir(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(Path, "mkdir", _boom_mkdir)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    pipe = pipeline.VoicePipeline.__new__(pipeline.VoicePipeline)
    pipe.registry = pipeline.EngineRegistry("pc")
    pipe.keyterms = None
    pipe._stages = dict(pipeline._DEFAULT_STAGES)
    cfg = pipeline.PipelineConfig(
        workdir=str(tmp_path), denoise=False, transcribe=False,
        diarize=False, synthesize=False, master=False, keyterms=False)
    res = pipe.run(audio, cfg)
    assert res["ok"] is True
    assert _sites() == ["pipeline.run_workdir_mkdir"]


def test_converted_pipeline_synthesizer_records_and_returns_none(monkeypatch):
    monkeypatch.setattr(pipeline, "_stage_synthesize", _boom)
    pipe = pipeline.VoicePipeline.__new__(pipeline.VoicePipeline)
    pipe.registry = pipeline.EngineRegistry("pc")
    cfg = pipeline.PipelineConfig()
    assert pipeline._pipeline_synthesizer(pipe, cfg, "hello", "ref.wav") is None
    assert _sites() == ["pipeline.pipeline_synthesizer"]


def test_converted_turns_for_records_and_returns_none(monkeypatch):
    monkeypatch.setattr(pipeline, "run_pipeline", _boom)
    assert pipeline._turns_for("x.wav") is None
    assert _sites() == ["pipeline.turns_for"]


# ---- Exclusion pin: honest outcome, NOT recorded -------------------------

def test_excluded_pyannote_import_not_recorded_and_reason_honest(monkeypatch):
    # Exercise the EXCLUDED handler itself: token present, pyannote absent.
    # (The no-token path returns before the import; it does not trip this site.)
    monkeypatch.setenv("WHISPERX_HF_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    res = pipeline.diarize_real("x.wav")
    assert res["ok"] is False
    assert "pyannote.audio not installed" in res["reason"]
    assert _sites() == []
