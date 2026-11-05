"""§1.4 acceptance: caption timing, measured on real speech.

Before this corpus, the only accuracy proof for measured caption
timings was a fake transcriber returning the reference words
themselves — MAE 0 by construction, plumbing dressed as proof.
The corpus (:mod:`nomorals.media.edit_engine.caption_corpus`)
synthesizes its scripts through the real TTS ladder and transcribes
them through the real §2.4 path; ``tests/golden/
caption_corpus.json`` records per-fixture WER / timing-MAE floors
and the values actually measured when the manifest was written.

These tests split honestly:

* scoring math + manifest honesty run everywhere, rung-free;
* the harness plumbing is proven with fakes (labelled as such);
* the end-to-end run enforces the recorded floors on machines
  where the rungs exist, and *skips with the manifest's own named
  gap* where they do not — a missing rung is a recorded fact,
  never a silent pass and never an invented number.
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path

import pytest

from nomorals.media.edit_engine import caption_corpus as cc

MANIFEST = json.loads(
    (Path(__file__).parent / "golden" / "caption_corpus.json")
    .read_text(encoding="utf-8"))
BY_NAME = {f["name"]: f for f in MANIFEST["fixtures"]}
NAMES = cc.fixture_names()


# ---------------------------------------------------------------------------
# scoring math — pure, runs everywhere
# ---------------------------------------------------------------------------

def _w(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_wer_perfect_and_empty():
    assert cc.word_error_rate(["the", "cat"], ["the", "cat"]) == 0.0
    assert cc.word_error_rate(["the", "cat"], []) == 1.0
    assert cc.word_error_rate([], []) == 0.0


def test_wer_counts_substitution_deletion_insertion():
    # one substitution of three
    assert cc.word_error_rate(["a", "b", "c"], ["a", "x", "c"]) \
        == pytest.approx(1 / 3)
    # one deletion + one insertion of four
    assert cc.word_error_rate(["a", "b", "c", "d"], ["a", "c", "d", "e"]) \
        == pytest.approx(2 / 4)


def test_wer_normalizes_case_and_punctuation():
    assert cc.word_error_rate(["The", "cat."], ["the", "cat"]) == 0.0


def test_score_transcription_timing_mae():
    ref = [_w("the", 0.30, 0.55), _w("cat", 1.00, 1.40),
           _w("sat", 1.85, 2.20)]
    got = [_w("the", 0.36, 0.60), _w("cat", 1.10, 1.44),
           _w("sat", 1.80, 2.26)]
    out = cc.score_transcription("the cat sat", ref, got)
    assert out["wer"] == 0.0
    assert out["timing"] == "scored"
    assert out["mae_start_s"] == pytest.approx((0.06 + 0.10 + 0.05) / 3)
    assert out["mae_end_s"] == pytest.approx((0.05 + 0.04 + 0.06) / 3)
    assert out["timed_pairs"] == 3 and out["exact_pairs"] == 3


def test_score_transcription_dropped_word_still_times_the_rest():
    ref = [_w("the", 0.30, 0.55), _w("cat", 1.00, 1.40),
           _w("sat", 1.85, 2.20)]
    got = [_w("the", 0.31, 0.56), _w("sat", 1.86, 2.21)]
    out = cc.score_transcription("the cat sat", ref, got)
    assert out["wer"] == pytest.approx(1 / 3, abs=1e-3)
    assert out["timed_pairs"] == 2
    assert out["mae_start_s"] == pytest.approx(0.01)


def test_score_transcription_fluent_records_timing_not_scored():
    got = [_w("caption", 0.0, 0.5), _w("timing", 0.6, 1.0)]
    out = cc.score_transcription("caption timing matters", None, got)
    assert out["timing"] == "not_scored"
    assert "mae_start_s" not in out
    assert out["wer"] == pytest.approx(1 / 3, abs=1e-3)


# ---------------------------------------------------------------------------
# manifest honesty — the recorded numbers must be self-consistent
# ---------------------------------------------------------------------------

def test_manifest_covers_exactly_the_corpus():
    assert MANIFEST["kind"] == "caption_corpus"
    assert sorted(BY_NAME) == sorted(NAMES)
    for name, entry in BY_NAME.items():
        fixture = cc.get_fixture(name)
        assert entry["text"] == fixture["text"], name
        assert entry["construction"] == fixture["construction"], name


def test_manifest_floors_sit_above_the_measurements():
    for name, entry in BY_NAME.items():
        if entry["status"] != "measured":
            continue
        floors, measured = entry["floors"], entry["measured"]
        assert floors["wer"] >= measured["wer"], name
        if measured["timing"] == "scored":
            assert floors["mae_start_s"] >= measured["mae_start_s"], name
            assert floors["mae_end_s"] >= measured["mae_end_s"], name


def test_manifest_gaps_are_named_never_numbered():
    """A fixture that could not be measured carries a reason, and
    no score fields — inventing a number is the failure this
    corpus exists to prevent."""
    for name, entry in BY_NAME.items():
        if entry["status"] == "gap":
            assert entry["gap"], name
            assert "measured" not in entry, name
            assert "floors" not in entry, name


def test_fluent_fixtures_record_timing_not_scored():
    for name, entry in BY_NAME.items():
        if entry["construction"] != "fluent" or entry["status"] != "measured":
            continue
        assert entry["measured"]["timing"] == "not_scored", name
        assert "mae_start_s" not in entry["measured"], name


# ---------------------------------------------------------------------------
# harness plumbing — fakes, labelled as plumbing (not measurement)
# ---------------------------------------------------------------------------

class _FakeTTS:
    """Writes token-length silence clips through speak()'s shape."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def speak(self, text, voice_name=None, out_path="", **kw):
        self.calls.append(text)
        sr = 16000
        n = int(sr * (0.18 + 0.05 * len(text)))
        cc._write_wav_samples(Path(out_path), array("h", bytes(2 * n)), sr)
        return {"path": out_path, "sample_rate": sr,
                "backend": "fake", "bytes": 2 * n}


class _ReferenceTranscriber:
    """Returns each fixture's reference words verbatim (plumbing).

    The fixture is identified from the audio path run_corpus hands
    over; its reference is rebuilt with the same deterministic
    fake TTS, so the words line up with the harness's own build.
    """

    def __init__(self, scratch: Path) -> None:
        self._scratch = scratch

    def transcribe_words(self, path):
        fixture = cc.get_fixture(Path(path).stem)
        built = cc.build_fixture_audio(fixture, tts=_FakeTTS(),
                                       out_dir=self._scratch
                                       / fixture["name"])
        if built["reference_words"] is not None:
            return [dict(w) for w in built["reference_words"]]
        return [{"text": tok, "start": 0.1 * i, "end": 0.1 * i + 0.08}
                for i, tok in enumerate(fixture["text"].split())]


def test_word_by_word_reference_is_exact_by_construction(tmp_path):
    fixture = cc.get_fixture("small_tools")
    built = cc.build_fixture_audio(fixture, tts=_FakeTTS(),
                                   out_dir=tmp_path)
    ref = built["reference_words"]
    assert [w["text"] for w in ref] == fixture["text"].split()
    cursor = cc.LEAD_S
    for w in ref:
        assert w["start"] == pytest.approx(cursor, abs=1e-3)
        assert w["end"] > w["start"]
        cursor = w["end"] + cc.GAP_S
    samples, sr = cc._read_wav_samples(Path(built["path"]))
    assert len(samples) / sr == pytest.approx(ref[-1]["end"], abs=1e-3)


def test_harness_plumbing_with_fakes(tmp_path):
    results = cc.run_corpus(
        tmp_path / "run", tts=_FakeTTS(),
        transcriber=_ReferenceTranscriber(tmp_path / "refs"))
    for entry in results["fixtures"]:
        assert entry["status"] == "measured", entry
        assert entry["wer"] == 0.0, entry
        if entry["construction"] == "word_by_word":
            assert entry["mae_start_s"] == 0.0, entry


def test_harness_records_gap_when_stt_is_silent(tmp_path):
    class _SilentTranscriber:
        def transcribe_words(self, path):
            return []

    results = cc.run_corpus(
        tmp_path / "run", tts=_FakeTTS(),
        transcriber=_SilentTranscriber())
    for entry in results["fixtures"]:
        assert entry["status"] == "gap", entry
        assert "no STT rung" in entry["gap"], entry


# ---------------------------------------------------------------------------
# end to end — enforced where the rungs exist, named gap where not
# ---------------------------------------------------------------------------

def _rungs_usable() -> tuple[bool, str]:
    """Probe the real ladder the way the corpus runs it."""
    try:
        from nomorals.voice.stt import available_stt_backends
        if not available_stt_backends():
            return False, "no STT backend installed"
    except Exception as exc:
        return False, f"STT probe failed: {exc}"
    try:
        from nomorals.voice.tts import available_backends
        if "piper" not in available_backends():
            return False, "piper TTS rung not installed"
    except Exception as exc:
        return False, f"TTS probe failed: {exc}"
    return True, ""


@pytest.fixture(scope="module")
def corpus_run(tmp_path_factory):
    """One real end-to-end corpus run per test session, shared by
    every per-fixture floor assertion (synthesis + transcription
    are the expensive part; scoring is per-fixture and cheap)."""
    usable, why = _rungs_usable()
    if not usable:
        return {"usable": False, "why": why, "results": {}}
    from nomorals.voice.tts import UniversalTTS
    work = tmp_path_factory.mktemp("caption_corpus")
    tts = UniversalTTS(backend="piper",
                       voices_dir=str(work / "voices"))
    results = cc.run_corpus(work / "run", tts=tts)
    return {"usable": True, "why": "",
            "results": {r["name"]: r for r in results["fixtures"]}}


@pytest.mark.parametrize("name", NAMES)
def test_corpus_floors_hold_end_to_end(name, corpus_run):
    entry = BY_NAME[name]
    if not corpus_run["usable"]:
        if entry["status"] == "measured":
            pytest.skip(
                f"manifest measured this fixture where rungs exist; "
                f"this machine cannot re-run it: {corpus_run['why']}")
        pytest.skip(f"manifest records the gap for this fixture; "
                    f"this machine: {corpus_run['why']}")
    got = corpus_run["results"][name]
    assert got["status"] == "measured", got.get("gap")
    assert got["wer"] <= entry["floors"]["wer"], (name, got)
    if entry["measured"]["timing"] == "scored":
        assert got["mae_start_s"] <= entry["floors"]["mae_start_s"], \
            (name, got)
        assert got["mae_end_s"] <= entry["floors"]["mae_end_s"], \
            (name, got)
