"""Windows-compat regression tests: paths, encoding, process management.

Covers the audit-pathproc fixes:

* temp-dir fallbacks use ``tempfile.gettempdir()`` (honours TMPDIR /
  TEMP / TMP), never a hardcoded ``/tmp`` (``C:\\tmp`` on Windows);
* generated filenames are sanitised (Windows-illegal ``<>:"/\\|?*``,
  traversal, reserved stems, trailing dots/spaces);
* LoRA training runs list-form argv with ``shell=False`` (no
  ``cmd.exe`` / POSIX-shell quoting);
* subprocess text-mode calls decode as UTF-8;
* dataset names that are not safe filenames are rejected fail-fast.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import pytest

from nomorals.core.text import safe_filename


# ── safe_filename ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("plain-name", "plain-name"),
    ("My Voice 2.wav", "My Voice 2.wav"),  # safe names pass through verbatim
    ("a:b", "a_b"),
    ("a<b>c|d?e*f", "a_b_c_d_e_f"),
    ('a"b', "a_b"),
    ("a/b", "a_b"),          # separators become "_" — no traversal
    ("../escape", ".._escape"),
    ("..\\escape", ".._escape"),
    ("C:\\temp\\x", "C__temp_x"),
    ("CON", "_CON"),       # reserved stems
    ("con.txt", "_con.txt"),
    ("COM1", "_COM1"),
    ("LPT9.dat", "_LPT9.dat"),
    ("NUL", "_NUL"),
    ("trailing.", "trailing"),
    ("trailing... ", "trailing"),
    ("", "file"),
    ("...", "file"),
    ("a\x01b", "a_b"),     # control chars
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_safe_filename_fallback_and_never_empty():
    assert safe_filename(None) == "file"
    assert safe_filename("", fallback="voice") == "voice"
    assert safe_filename("///") == "___"  # separators → "_", still safe
    assert safe_filename("a" * 500) != ""
    assert len(safe_filename("a" * 500)) <= 200


# ── temp-dir fallbacks ──────────────────────────────────────────────────────

def test_temp_fallbacks_use_gettempdir(monkeypatch, tmp_path):
    """When the primary dir fails, caches fall back under gettempdir()."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")

    from nomorals.community.audio_rooms import RoomStore
    from nomorals.media.download_kit import IntegrityLedger
    from nomorals.media.source_registry import ContentAddressedCache

    cache = ContentAddressedCache(root=str(blocker / "sub"))
    assert Path(cache.root).parent == tmp_path

    ledger = IntegrityLedger(path=str(blocker / "ledger.jsonl"))
    assert Path(ledger.path).parent == tmp_path

    store = RoomStore(data_dir=str(blocker / "rooms"))
    assert Path(store._dir).parent == tmp_path


def test_audio_pipeline_workdir_default_is_platform_temp():
    from nomorals.audio.pipeline import PipelineConfig

    assert PipelineConfig().workdir == tempfile.gettempdir()
    assert PipelineConfig.from_dict({}).workdir == tempfile.gettempdir()
    assert (PipelineConfig.from_dict({"workdir": "/custom"}).workdir
            == "/custom")


# ── generated filenames ─────────────────────────────────────────────────────

def test_invent_save_design_sanitises_filename(monkeypatch, tmp_path):
    from nomorals.games import invent

    monkeypatch.setattr(invent, "invention_dir", lambda: tmp_path)
    design = {"name": "my game: part 1/2", "title": "T",
              "min_players": 2, "max_players": 4, "setup": "s",
              "turn": "t", "win": "w", "examples": []}
    p = invent.save_design(design)
    assert p.parent == tmp_path
    assert ":" not in p.name and "/" not in p.name
    assert p.is_file()
    # The design's own name field is kept verbatim.
    assert "my game" in p.read_text(encoding="utf-8")


def test_voice_library_upload_sanitises_filename(tmp_path):
    from nomorals.voice.tts import VoiceLibrary

    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)

    lib = VoiceLibrary(storage_dir=str(tmp_path / "voices"))
    profile = lib.upload_voice("my:voice/test", str(clip))
    assert profile.name == "my:voice/test"  # display name unchanged
    dest = Path(profile.reference_audio_path)
    assert ":" not in dest.name and "/" not in dest.name
    assert dest.is_file()


def test_datasci_rejects_unsafe_dataset_names(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    from nomorals.datasci import DataWorkspace, LoadError

    ws = DataWorkspace(tmp_path / "ws")
    frame = pd.DataFrame({"a": [1, 2]})
    for bad in ("a/b", "a:b", "a*b", "a?b", "..", "CON", "trailing.",
                "a\\b", "a|b", 'a"b', "a<b"):
        with pytest.raises(LoadError):
            ws.load_frame(bad, frame)
    ds = ws.load_frame("good-name_1", frame)
    assert ds.name == "good-name_1"
    with pytest.raises(LoadError):
        ws.rename("good-name_1", "evil/name")


# ── process management: list-form argv, no shell ────────────────────────────

class _FakeProc:
    returncode = 0
    stdout = ""
    stderr = ""


def test_lora_jobs_run_list_form_without_shell(monkeypatch, tmp_path):
    import subprocess as _sp

    from nomorals.media.afrobeats import LoRATrainingJob
    from nomorals.media.lora_global import GenreLoRAJob

    calls: list[dict] = []

    def fake_run(argv, **kw):
        calls.append({"argv": argv, "kwargs": kw})
        return _FakeProc()

    monkeypatch.setattr(_sp, "run", fake_run)

    # Corpus/output paths with spaces and "&" must stay single argv items.
    corpus = tmp_path / "my corpus & stuff"
    out = tmp_path / "out dir"
    corpus.mkdir()
    job = GenreLoRAJob(genre="afrobeats", corpus_dir=str(corpus),
                       output_dir=str(out))
    monkeypatch.setattr(job, "validate", lambda: [])
    assert job.run()["ok"] is True

    job2 = LoRATrainingJob(corpus_dir=str(corpus), output_dir=str(out),
                           tag="my tag & more")
    monkeypatch.setattr(job2, "validate", lambda: [])
    assert job2.run()["ok"] is True

    assert len(calls) == 6  # 3 steps x 2 jobs
    for call in calls:
        assert isinstance(call["argv"], list)
        assert call["kwargs"].get("shell") is False
        assert call["kwargs"].get("encoding") == "utf-8"
        assert all(isinstance(a, str) for a in call["argv"])
    # The "&" path is one argv element, not a shell operator.
    assert str(corpus) in calls[0]["argv"]
    assert "my tag & more" in calls[3]["argv"]  # job2 prepare step


def test_subprocess_text_mode_decodes_utf8():
    """Explicit UTF-8 decoding survives non-ASCII child output."""
    import sys

    r = subprocess.run(
        [sys.executable, "-c", "print('héllo → wörld')"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=30)
    assert "héllo" in r.stdout
