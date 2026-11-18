"""Tests for word-level caption timing on code videos — captions that
follow the spoken word.

The recheck (2026-11-18) proved there is NO WhisperX word-level
caption timing anywhere in videogen/motion_studio: zero
``whisperx|word-level|word_level`` hits. The video style the owner
asked for needs captions timed to the spoken word, and the narration
rung (lane 79) now gives the captions real speech to time against.

The timing source is real, never invented: the narration mix wav is
transcribed by the repo's own faster-whisper word-timestamp seam
(``media_edit.captions.transcribe_words`` — the WhisperX pattern),
and each script word is matched to a MEASURED word in order. When
faster-whisper is not installed the words are split across the
segment's own spoken window — the window itself is measured (the
spoken audio length), only the within-window distribution is an
estimate — and every such word is labelled ``"estimated"``, never
``"measured"``. Provenance is per word, following the motion-studio
§2.4 discipline (``typography.py``): "measured" is a claim only a
measurement earns.

The end-to-end tests speak with REAL piper synthesis and align with
REAL faster-whisper (base model) — they skip honestly (importorskip)
on a host without either, the way the narration battery does.
"""
from __future__ import annotations

import pytest

from nomorals.media.videogen.codegen import render_code_scene
from nomorals.media.videogen.formats import render_format

# The unit tests below run everywhere (fake transcriber / estimate /
# validator / draw). Only the e2e + QC tests need the real backends,
# and they importorskip inside the test so the file stays green on a
# host without piper or faster-whisper (EC2).

HEADLINE = {
    "engine": "equation",
    "kind": "equation",
    "title": "Lagos to Abuja",
    "lines": [{"text": "Lagos to Abuja", "box": True},
              {"text": "The road opens", "box": False},
              {"text": "The journey begins", "box": False}],
}

SIZE = (320, 180)


def _narration() -> dict:
    """A synthesize_narration-shaped record (no TTS needed)."""
    return {"ok": True, "path": "", "backend": "piper",
            "duration": 4.0,
            "segments": [
                {"text": "Lagos to Abuja. The road opens.",
                 "spoken_start": 0.0, "spoken_end": 2.0},
                {"text": "The journey begins.",
                 "spoken_start": 2.15, "spoken_end": 3.6},
            ]}


def _measured_words():
    """The script's words as a transcriber would measure them."""
    from nomorals.media_edit.captions import Word

    return [
        Word(0.10, 0.45, "Lagos"), Word(0.50, 0.75, "to"),
        Word(0.80, 1.20, "Abuja."), Word(1.30, 1.55, "The"),
        Word(1.60, 1.80, "road"), Word(1.85, 1.98, "opens."),
        Word(2.30, 2.60, "The"), Word(2.65, 3.00, "journey"),
        Word(3.05, 3.45, "begins."),
    ]


# -- alignment: measured -----------------------------------------------

def test_caption_narration_measured_fake_transcriber():
    from nomorals.media.videogen.captions import caption_narration

    got = caption_narration(_narration(),
                            transcriber=lambda p, **kw: _measured_words())
    assert got["status"] == "measured"
    assert got["timing_source"] == "measured"
    words = got["words"]
    assert [w["text"] for w in words] == [
        "Lagos", "to", "Abuja.", "The", "road", "opens.",
        "The", "journey", "begins."]
    assert all(w["timing_source"] == "measured" for w in words)
    # in order, no overlaps, inside their segment windows
    for i, w in enumerate(words):
        assert w["start"] < w["end"]
        if i:
            assert w["start"] >= words[i - 1]["end"] - 1e-6
        seg = _narration()["segments"][w["segment"]]
        assert w["start"] >= seg["spoken_start"] - 1e-6
        assert w["end"] <= seg["spoken_end"] + 1e-6
    assert got["words_measured"] == 9
    assert got["words_estimated"] == 0


def test_caption_narration_estimated_without_transcriber():
    from nomorals.media.videogen.captions import caption_narration

    def _missing(path, **kw):
        raise ImportError("faster-whisper not installed")

    got = caption_narration(_narration(), transcriber=_missing)
    assert got["status"] == "estimated"
    assert got["timing_source"] == "estimated"
    words = got["words"]
    assert [w["text"] for w in words] == [
        "Lagos", "to", "Abuja.", "The", "road", "opens.",
        "The", "journey", "begins."]
    # every word labelled estimated — never dressed up as measured
    assert all(w["timing_source"] == "estimated" for w in words)
    # split across each segment's own spoken window, in order
    for i, w in enumerate(words):
        assert w["start"] < w["end"]
        if i:
            assert w["start"] >= words[i - 1]["end"] - 1e-6
        seg = _narration()["segments"][w["segment"]]
        assert w["start"] >= seg["spoken_start"] - 1e-6
        assert w["end"] <= seg["spoken_end"] + 1e-6
    assert got["words_measured"] == 0
    assert got["words_estimated"] == 9
    assert "faster-whisper" in got["reason"]


def test_caption_narration_mixed_when_a_word_is_misheard():
    """One script word the transcriber mis-hears is interpolated and
    labelled estimated; the rest stay measured; the record says so."""
    from nomorals.media.videogen.captions import caption_narration
    from nomorals.media_edit.captions import Word

    heard = [w for w in _measured_words() if w.text != "road"]
    heard.insert(4, Word(1.60, 1.80, "rode"))  # mis-heard "road"
    got = caption_narration(_narration(),
                            transcriber=lambda p, **kw: heard)
    assert got["status"] == "mixed"
    words = got["words"]
    assert [w["text"] for w in words] == [
        "Lagos", "to", "Abuja.", "The", "road", "opens.",
        "The", "journey", "begins."]
    road = words[4]
    assert road["text"] == "road"
    assert road["timing_source"] == "estimated"
    assert got["words_measured"] == 8
    assert got["words_estimated"] == 1
    assert got["unmatched_measured"] == 1


def test_caption_narration_missing_when_nothing_to_narrate():
    from nomorals.media.videogen.captions import caption_narration

    got = caption_narration({"ok": True, "path": "", "backend": "",
                             "duration": 5.0, "segments": []})
    assert got["status"] == "missing"
    assert got["words"] == []
    assert got["reason"]


# -- the structural check ----------------------------------------------

def test_check_caption_timing_valid_and_broken():
    from nomorals.media.videogen.captions import caption_narration, check_caption_timing

    good = caption_narration(_narration(),
                             transcriber=lambda p, **kw: _measured_words())
    ok, reasons = check_caption_timing(good, duration=4.0)
    assert ok, reasons

    import copy
    overlap = copy.deepcopy(good)
    overlap["words"][1]["start"] = overlap["words"][0]["start"] + 0.01
    overlap["words"][1]["end"] = overlap["words"][0]["end"] + 0.5
    ok, reasons = check_caption_timing(overlap, duration=4.0)
    assert not ok
    assert any("overlap" in r for r in reasons)

    outside = copy.deepcopy(good)
    outside["words"][-1]["end"] = 9.9
    ok, reasons = check_caption_timing(outside, duration=4.0)
    assert not ok
    assert any("outside" in r for r in reasons)

    dropped = copy.deepcopy(good)
    dropped["words"] = dropped["words"][:-1]
    ok, reasons = check_caption_timing(dropped, duration=4.0)
    assert not ok
    assert any("coverage" in r for r in reasons)


# -- the overlay --------------------------------------------------------

def test_draw_captions_lights_the_active_word(tmp_path):
    from PIL import Image

    from nomorals.media.videogen.captions import caption_narration, draw_captions

    got = caption_narration(_narration(),
                            transcriber=lambda p, **kw: _measured_words())
    base = Image.new("RGB", SIZE, (10, 10, 14))
    before = base.copy()
    # t=0.0 — no word has started: the frame is untouched
    draw_captions(base, got["words"], 0.0)
    assert list(base.getdata()) == list(before.getdata())
    # t inside "Lagos": the caption zone changes
    draw_captions(base, got["words"], 0.25)
    assert list(base.getdata()) != list(before.getdata())
    # deterministic: same t, same pixels
    again = Image.new("RGB", SIZE, (10, 10, 14))
    draw_captions(again, got["words"], 0.25)
    assert list(base.getdata()) == list(again.getdata())


def test_captions_srt_sidecar():
    from nomorals.media.videogen.captions import caption_narration, captions_srt

    got = caption_narration(_narration(),
                            transcriber=lambda p, **kw: _measured_words())
    srt = captions_srt(got["words"])
    assert "00:00:00,100 --> 00:00:01,980" in srt
    assert "00:00:02,300 --> 00:00:03,450" in srt
    assert "Lagos" in srt
    assert "begins." in srt
    assert srt.count("-->") >= 1


# -- wiring: render -----------------------------------------------------

def test_render_code_scene_silent_has_captions_off(tmp_path):
    rec = render_code_scene(HEADLINE, tmp_path / "silent.mp4",
                            duration=2.0, size=SIZE, fps=8,
                            narrate=False)
    assert rec["captions"]["status"] == "off"
    assert rec["captions"]["words"] == []


def test_render_code_scene_narration_missing_reports_captions(tmp_path):
    def _no_tts(text, out_path):
        return {"ok": False, "reason": "no TTS backend installed"}

    rec = render_code_scene(HEADLINE, tmp_path / "notts.mp4",
                            duration=2.0, size=SIZE, fps=8,
                            narrate=True, tts_fn=_no_tts)
    assert rec["narration"]["status"] == "missing"
    assert rec["captions"]["status"] == "missing"
    assert rec["captions"]["reason"]


def test_render_format_no_captions_flag(tmp_path):
    res = render_format("headline", headline="Lagos to Abuja",
                        lines="The road opens",
                        duration_s=2.0, out=str(tmp_path / "nocap.mp4"),
                        narrate=True, captions=False)
    assert res.meta["captions"]["status"] == "off"


# -- end to end: real voice, real alignment -----------------------------

def test_e2e_real_voice_real_alignment(tmp_path, monkeypatch):
    """A real narrated headline: piper speaks, faster-whisper
    measures, the caption words are the script words in order and
    their windows sit inside the narration's segment windows."""
    pytest.importorskip("faster_whisper",
                        reason="faster-whisper not installed on this host")
    pytest.importorskip("piper", reason="piper-tts not installed")
    # pin piper: faster-whisper drags in huggingface_hub, which makes
    # the cloud hf-endpoint backend "available" — and auto scores it
    # above piper by quality, then its synthesize fails (no provider
    # can serve the model in this env). The cascade only covers
    # construction failures, so auto would go silent. Pinning the
    # local voice here is the test's job; the auto-poisoning is a
    # named remainder in WORDCAPTIONS_SURVEY.md.
    monkeypatch.setattr(
        "nomorals.voice.tts.available_backends", lambda: ["piper"])
    out = tmp_path / "headline.mp4"
    res = render_format("headline", headline="Lagos to Abuja",
                        lines="The road opens|The journey begins",
                        duration_s=5.0, size=SIZE, fps=8,
                        out=str(out))
    rec = res.meta
    assert rec["narration"]["status"] == "spoken", rec["narration"]
    caps = rec["captions"]
    assert caps["status"] in ("measured", "mixed"), caps
    assert caps["timing_source"] in ("measured", "mixed")
    words = caps["words"]
    assert [w["text"] for w in words] == [
        "Lagos", "to", "Abuja", "The", "road", "opens",
        "The", "journey", "begins"]
    # most words measured, not estimated
    assert caps["words_measured"] >= len(words) - 2
    # windows inside the narration's segment windows, in order
    segs = caps["segments"]
    for w in words:
        assert w["start"] < w["end"]
        seg = segs[w["segment"]]
        assert w["start"] >= seg["spoken_start"] - 1e-6
        assert w["end"] <= seg["spoken_end"] + 1e-6
    for i, w in enumerate(words):
        if i:
            assert w["start"] >= words[i - 1]["end"] - 1e-6
    # the SRT sidecar exists beside the MP4
    srt = out.with_name(out.stem + ".captions.srt")
    assert srt.is_file()
    assert "Lagos" in srt.read_text(encoding="utf-8")
    # the MP4 really carries the voice
    from nomorals.media_edit.videos import video_probe

    probe = video_probe(out)
    assert any(s.get("type") == "audio"
               for s in probe.get("streams", [])), probe


def test_qc_existing_caption_drift_fails(tmp_path, monkeypatch):
    """QC measures the caption axis for real: true captions pass, a
    +2s shift of every caption word fails naming the drift."""
    pytest.importorskip("faster_whisper",
                        reason="faster-whisper not installed on this host")
    pytest.importorskip("piper", reason="piper-tts not installed")
    monkeypatch.setattr(
        "nomorals.voice.tts.available_backends", lambda: ["piper"])
    from nomorals.media.videogen.qc import qc_existing

    out = tmp_path / "headline.mp4"
    res = render_format("headline", headline="Lagos to Abuja",
                        lines="The road opens|The journey begins",
                        duration_s=5.0, size=SIZE, fps=8,
                        out=str(out))
    caps = res.meta["captions"]
    assert caps["status"] in ("measured", "mixed")
    good = qc_existing(str(out), out_dir=str(tmp_path / "qc-good"),
                       captions=caps)
    assert good["verdict"] == "pass", good
    drift_axis = good["gate"]["axes"]["caption_sync_drift"]
    assert drift_axis["status"] == "pass", drift_axis

    import copy
    drifted = copy.deepcopy(caps)
    for w in drifted["words"]:
        w["start"] += 2.0
        w["end"] += 2.0
    bad = qc_existing(str(out), out_dir=str(tmp_path / "qc-bad"),
                      captions=drifted)
    assert bad["verdict"] == "fail", bad
    assert any("caption" in r and "drift" in r for r in bad["reasons"])


def test_qc_existing_silent_render_unchanged(tmp_path):
    """No captions handed in → the silent render QCs exactly as
    before (visual axes only, verdict pass)."""
    from nomorals.media.videogen.qc import qc_existing

    rec = render_code_scene(HEADLINE, tmp_path / "silent.mp4",
                            duration=2.0, size=SIZE, fps=8,
                            narrate=False)
    out = qc_existing(rec["path"], out_dir=str(tmp_path / "qc"))
    assert out["verdict"] == "pass"
