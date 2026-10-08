"""Tests for nomorals/media_edit/captions.py + the "add captions" NL intent.

All offline: the STT backend and ffmpeg are mocked; .ass/.srt generation,
grouping, styles, transcript decoupling, and intent wiring are real.
"""

import json
import re
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from nomorals.media_edit.captions import (
    Word,
    _ass_ts,
    _event_text,
    _group_events,
    _srt_ts,
    burn_captions,
    caption_styles,
    load_transcript,
    recaption,
    save_transcript,
    transcript_path,
    words_to_ass,
    words_to_srt,
)


def _words(n=8, start=0.0, gap=0.4):
    texts = ["hello", "world", "this", "is", "a", "test.", "of", "captions",
             "right", "here.", "and", "more"]
    out = []
    t = start
    for i in range(n):
        w = texts[i % len(texts)]
        out.append(Word(start=t, end=t + gap * 0.8, text=w))
        t += gap
    return out


# ── timestamps ────────────────────────────────────────────────────────────

def test_ass_ts_format():
    assert _ass_ts(0) == "0:00:00.00"
    assert _ass_ts(61.5) == "0:01:01.50"
    assert _ass_ts(3661.25) == "1:01:01.25"


def test_srt_ts_format():
    assert _srt_ts(0) == "00:00:00,000"
    assert _srt_ts(61.5) == "00:01:01,500"


# ── grouping ──────────────────────────────────────────────────────────────

def test_grouping_max_words():
    events = _group_events(_words(9))
    assert all(len(e) <= 4 for e in events)
    assert sum(len(e) for e in events) == 9


def test_grouping_sentence_break():
    # "test." ends a sentence → event boundary after it
    events = _group_events(_words(6))
    assert events[1][-1].text == "test."
    assert all(len(e) <= 4 for e in events)


def test_grouping_max_duration():
    ws = _words(8, gap=1.0)  # 0.8s apart → 4 words would be 3.2s > 2s cap
    events = _group_events(ws)
    for e in events:
        assert e[-1].end - e[0].start < 2.5


# ── .ass structure ─────────────────────────────────────────────────────────

def test_ass_structure():
    ass = words_to_ass(_words(6), style="hormozi")
    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass
    assert ass.count("Dialogue:") >= 2
    assert "Style: Caption" in ass


def test_ass_styles_differ():
    a = words_to_ass(_words(4), style="hormozi")
    b = words_to_ass(_words(4), style="mrbeast")
    c = words_to_ass(_words(4), style="minimal")
    assert a != b != c
    assert "&H0000FFFF" in b  # mrbeast yellow
    assert "Fontsize: 44" in c or ",44," in c  # minimal smaller


def test_karaoke_kf_tags():
    ass = words_to_ass(_words(4), style="karaoke")
    assert re.search(r"\{\\kf\d+\}", ass), "karaoke must emit {\\kf} tags"
    plain = words_to_ass(_words(4), style="hormozi")
    assert "\\kf" not in plain


def test_karaoke_durations_match_words():
    ws = [Word(0.0, 1.0, "hi"), Word(1.0, 1.5, "there")]
    text = _event_text(ws, "karaoke")
    assert "{\\kf100}hi" in text
    assert "{\\kf50}there" in text


def test_ass_event_no_overlap():
    ws = [Word(0.0, 0.9, "one"), Word(1.0, 1.9, "two"),
          Word(2.0, 2.9, "three"), Word(3.0, 3.9, "four"),
          Word(4.0, 4.9, "five")]
    ass = words_to_ass(ws, style="hormozi")
    # first event ends before the second event starts
    ends = re.findall(r"Dialogue: 0,([\d:.]+),([\d:.]+),", ass)
    assert len(ends) >= 2
    first_end = ends[0][1]
    second_start = ends[1][0]
    assert first_end <= second_start


def test_ass_unknown_style():
    with pytest.raises(Exception, match="unknown caption style"):
        words_to_ass(_words(2), style="nope")


def test_ass_empty_words():
    with pytest.raises(Exception, match="no words"):
        words_to_ass([])


def test_caption_styles_list():
    styles = caption_styles()
    assert {"hormozi", "mrbeast", "karaoke", "minimal"} <= set(styles)


# ── .srt ───────────────────────────────────────────────────────────────────

def test_srt_structure():
    srt = words_to_srt(_words(6))
    assert re.search(r"^1\n00:00:00,000 --> ", srt, re.M)
    assert re.search(r"^2\n", srt, re.M)
    assert "\\kf" not in srt  # plain text only


# ── transcript decoupling ─────────────────────────────────────────────────

def test_save_load_transcript_roundtrip(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    ws = _words(5)
    tp = save_transcript(video, ws)
    assert tp == transcript_path(video)
    assert tp.name == "clip.words.json"
    loaded = load_transcript(video)
    assert loaded is not None
    assert [(w.start, w.end, w.text) for w in loaded] == \
        [(w.start, w.end, w.text) for w in ws]


def test_load_transcript_missing(tmp_path):
    assert load_transcript(tmp_path / "nope.mp4") is None


def test_load_transcript_corrupt(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    transcript_path(video).write_text("{not json", encoding="utf-8")
    assert load_transcript(video) is None


# ── burn / recaption (mocked ffmpeg) ──────────────────────────────────────

def _fake_burn(src, subtitles, **kw):
    out = Path(str(src).replace(".mp4", "-captioned.mp4"))
    out.write_bytes(b"fake mp4")
    return {"input": str(src), "output": str(out), "bytes": 8,
            "seconds": 1.0}


def test_burn_delegates_to_burn_subtitles(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    ws = _words(6)
    with patch("nomorals.media_edit.captions.burn_subtitles",
               side_effect=_fake_burn) as m, \
         patch("nomorals.media_edit.captions.has_libass",
               return_value=True):
        run = burn_captions(video, ws, style="hormozi")
    assert m.called
    assert run["output"].endswith("-captioned.mp4")
    # sidecars written
    assert transcript_path(video).exists()
    assert video.with_suffix(".srt").exists()
    assert video.with_name("clip.hormozi.ass").exists()


def test_burn_with_ass_path(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    ass = tmp_path / "subs.ass"
    ass.write_text("[Events]\n", encoding="utf-8")
    with patch("nomorals.media_edit.captions.burn_subtitles",
               side_effect=_fake_burn), \
         patch("nomorals.media_edit.captions.has_libass",
               return_value=True):
        run = burn_captions(video, ass)
    assert run["ass"] == str(ass)


def test_burn_refuses_without_libass(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    with patch("nomorals.media_edit.captions.has_libass",
               return_value=False):
        with pytest.raises(Exception, match="no libass"):
            burn_captions(video, _words(4))


def test_recaption_skips_stt(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    ws = _words(5)
    save_transcript(video, ws)
    edited = [Word(w.start, w.end, "FIXED" if i == 0 else w.text)
              for i, w in enumerate(ws)]
    with patch("nomorals.media_edit.captions.burn_subtitles",
               side_effect=_fake_burn) as m, \
         patch("nomorals.media_edit.captions.has_libass",
               return_value=True), \
         patch("nomorals.media_edit.captions.transcribe_words") as stt:
        run = recaption(video, edited)
    stt.assert_not_called()  # the whole point: no STT re-run
    assert m.called
    assert run["output"].endswith("-captioned.mp4")


def test_recaption_from_json_file(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    ws = _words(4)
    tp = save_transcript(video, ws)
    with patch("nomorals.media_edit.captions.burn_subtitles",
               side_effect=_fake_burn), \
         patch("nomorals.media_edit.captions.has_libass",
               return_value=True), \
         patch("nomorals.media_edit.captions.transcribe_words") as stt:
        recaption(video, tp)
    stt.assert_not_called()


def test_recaption_empty_rejected(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    with pytest.raises(Exception, match="empty"):
        recaption(video, [])


# ── transcribe_words (mocked STT) ─────────────────────────────────────────

def _fake_backend_cls():
    class FakeWord:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class FakeSeg:
        def __init__(self, words):
            self.words = words

    class FakeModel:
        def transcribe(self, audio, **kw):
            assert kw.get("word_timestamps") is True
            return ([FakeSeg([FakeWord("hello", 0.0, 0.4),
                              FakeWord("world", 0.5, 0.9)])], None)

    class FakeBackend:
        def __init__(self, model=""):
            self.model = FakeModel()

    return FakeBackend


def test_transcribe_words_flattens(tmp_path):
    from nomorals.media_edit import captions as cap
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    fake_wav = tmp_path / "a.wav"
    fake_wav.write_bytes(b"RIFF")
    with patch.object(cap, "_extract_wav", return_value=fake_wav), \
         patch("nomorals.voice.stt.FasterWhisperBackend",
               _fake_backend_cls()):
        words = cap.transcribe_words(video)
    assert [(w.start, w.end, w.text) for w in words] == [
        (0.0, 0.4, "hello"), (0.5, 0.9, "world")]
    assert not fake_wav.exists()  # temp wav cleaned up


def test_transcribe_words_missing_video(tmp_path):
    from nomorals.media_edit import captions as cap
    with pytest.raises(Exception, match="no such video"):
        cap.transcribe_words(tmp_path / "ghost.mp4")


def test_transcribe_words_missing_faster_whisper(tmp_path, monkeypatch):
    from nomorals.media_edit import captions as cap
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    real_import = __import__

    def fake_import(name, *a, **k):
        if name == "nomorals.voice.stt" or name.startswith("faster_whisper"):
            raise ImportError("no faster_whisper")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(Exception, match="pip install faster-whisper"):
        cap.transcribe_words(video)


# ── NL intent ─────────────────────────────────────────────────────────────

def test_captions_intent_matches():
    from nomorals.agents.coremind import _image_intent
    for text in ["add captions", "Add Captions",
                 "add captions to this video", "caption this video"]:
        intent = _image_intent(text)
        assert intent is not None and intent.kind == "vision_captions", text
        assert intent.action == "captions"


def test_captions_intent_misfires():
    from nomorals.agents.coremind import _image_intent
    for text in ["add captions to the meeting notes",
                 "caption this contest",
                 "add some captions I guess",
                 "what are captions"]:
        assert _image_intent(text) is None, text
