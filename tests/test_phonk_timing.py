"""§1.4 karaoke protocol ride: phonk captions carry their label.

Phonk's karaoke layers previously dropped §2.4 provenance: text
timed by *estimation* reached the render rung notes as "authored",
and a ``captions_from_audio`` result couldn't be fed in at all.
Now the layer (and the direct-burn result) carries the words'
``timing_source`` — estimated stays estimated, measured stays
measured, unlabelled stays unlabelled.
"""

from __future__ import annotations

import subprocess

import pytest

from nomorals.media.contentops.styles import phonk
from nomorals.media_edit.videos import MediaEditError


def _layers(spec):
    return spec.text_layers


def test_plain_text_layer_is_labelled_estimated():
    spec = phonk.build_spec(captions="hello brave new world")
    (layer,) = _layers(spec)
    assert layer.timing_source == "estimated"
    assert layer.to_dict()["timing_source"] == "estimated"


def test_captions_from_audio_result_keeps_its_measured_label():
    got = {"words": [{"text": "hello", "start": 0.0, "end": 0.4},
                     {"text": "world", "start": 0.5, "end": 0.9}],
           "timing_source": "measured", "speakers": [],
           "transcriber": "STTTranscriber"}
    spec = phonk.build_spec(captions=got, caption_sync="voiceover")
    (layer,) = _layers(spec)
    assert layer.timing_source == "measured"
    assert [w["text"] for w in layer.words] == ["hello", "world"]


def test_uniform_word_stamps_carry_to_the_layer():
    words = [{"text": "a", "start": 0.0, "end": 0.3,
              "timing_source": "measured"},
             {"text": "b", "start": 0.4, "end": 0.7,
              "timing_source": "measured"}]
    spec = phonk.build_spec(captions=words, caption_sync="voiceover")
    (layer,) = _layers(spec)
    assert layer.timing_source == "measured"


def test_mixed_or_unlabelled_words_stay_unlabelled():
    mixed = [{"text": "a", "start": 0.0, "end": 0.3,
              "timing_source": "measured"},
             {"text": "b", "start": 0.4, "end": 0.7,
              "timing_source": "estimated"}]
    (layer,) = _layers(phonk.build_spec(
        captions=mixed, caption_sync="voiceover"))
    assert layer.timing_source == ""
    plain = [{"text": "a", "start": 0.0, "end": 0.3},
             {"text": "b", "start": 0.4, "end": 0.7}]
    (layer,) = _layers(phonk.build_spec(
        captions=plain, caption_sync="voiceover"))
    assert layer.timing_source == ""
    assert "timing_source" not in layer.to_dict()


def test_burn_result_reports_timing_source(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc=size=320x240:rate=10:duration=1",
         "-pix_fmt", "yuv420p", str(video)],
        check=True, capture_output=True)
    words = [{"text": "hello", "start": 0.05, "end": 0.4,
              "timing_source": "measured"},
             {"text": "world", "start": 0.5, "end": 0.9,
              "timing_source": "measured"}]
    try:
        run = phonk.burn_beat_captions(video, words, sync=False,
                                       out_dir=tmp_path)
    except MediaEditError as exc:
        pytest.skip(f"caption burn unavailable on this machine: {exc}")
    assert run["timing_source"] == "measured"
    run2_words = [{"text": "hello", "start": 0.05, "end": 0.4},
                  {"text": "world", "start": 0.5, "end": 0.9}]
    run2 = phonk.burn_beat_captions(video, run2_words, sync=False,
                                    out_dir=tmp_path)
    assert run2["timing_source"] == "authored"


def test_words_timing_source_helper():
    assert phonk.words_timing_source(
        [{"timing_source": "measured"}, {"timing_source": "measured"}]) \
        == "measured"
    assert phonk.words_timing_source(
        [{"timing_source": "measured"}, {"timing_source": "estimated"}]) \
        == ""
    assert phonk.words_timing_source([{"text": "a"}]) == ""
    assert phonk.words_timing_source([]) == ""
