"""Tests for transcript-first video editing (#28). All offline."""

import json
import re
import xml.etree.ElementTree as ET

import pytest

from nomorals.media_edit.captions import Word
from nomorals.media_edit import transcript_edit as te
from nomorals.media_edit.videos import MediaEditError


def W(s, e, t):
    return Word(start=s, end=e, text=t)


@pytest.fixture()
def words():
    return [
        W(0.0, 0.4, "hello"), W(0.4, 0.6, "um"), W(0.6, 1.0, "world"),
        W(1.0, 1.4, "this"), W(1.4, 1.6, "is"), W(1.6, 2.0, "an"),
        W(2.0, 2.4, "umbrella"), W(2.4, 2.8, "uh"), W(2.8, 3.2, "test"),
        W(3.2, 3.8, "you"), W(3.8, 4.2, "know"), W(4.2, 4.6, "yeah"),
    ]


# ── filler detection ─────────────────────────────────────────────────

class TestFindFillers:
    def test_whole_word_match(self, words):
        hits = te.find_fillers(words)
        texts = [(s, e) for s, e in hits]
        assert (0.4, 0.6) in texts  # um
        assert (2.4, 2.8) in texts  # uh
        assert (3.2, 4.2) in texts  # you know (merged)

    def test_umbrella_is_not_um(self, words):
        hits = te.find_fillers(words)
        for s, e in hits:
            assert not (2.0 <= s < 2.4 and e <= 2.4), \
                "umbrella must not match 'um'"

    def test_case_insensitive(self):
        ws = [W(0.0, 0.5, "UM"), W(0.5, 1.0, "Uh")]
        # adjacent fillers merge into one cut region — still detected
        assert te.find_fillers(ws) == [(0.0, 1.0)]

    def test_custom_fillers(self, words):
        assert te.find_fillers(words, fillers=("world",)) == [(0.6, 1.0)]

    def test_empty(self):
        assert te.find_fillers([]) == []


# ── range math ───────────────────────────────────────────────────────

class TestRangeMath:
    def test_merge(self):
        assert te._merge_ranges([(0, 1), (0.9, 2), (5, 6)]) == [(0, 2), (5, 6)]

    def test_merge_gap(self):
        assert te._merge_ranges([(0, 1), (1.04, 2)]) == [(0, 2)]
        assert te._merge_ranges([(0, 1), (1.2, 2)]) == [(0, 1), (1.2, 2)]

    def test_complement(self):
        keeps = te._complement([(1.0, 2.0), (3.0, 4.0)], 0.0, 5.0)
        assert keeps == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]

    def test_complement_edges(self):
        assert te._complement([(0.0, 1.0)], 0.0, 5.0) == [(1.0, 5.0)]
        assert te._complement([], 0.0, 5.0) == [(0.0, 5.0)]
        assert te._complement([(0.0, 5.0)], 0.0, 5.0) == []

    def test_pad_clamped(self):
        assert te._pad([(1.0, 2.0)], 0.5, 0.0, 5.0) == [(0.5, 2.5)]
        assert te._pad([(0.1, 4.9)], 0.5, 0.0, 5.0) == [(0.0, 5.0)]


# ── transcript wrapper ───────────────────────────────────────────────

class TestTranscript:
    def test_text(self, words):
        tt = te.TimestampedTranscript(words=words)
        assert "hello um world" in tt.text()

    def test_ranges_for_single_word(self, words):
        tt = te.TimestampedTranscript(words=words)
        assert tt.ranges_for("umbrella") == [(2.0, 2.4)]
        assert tt.ranges_for("nope") == []

    def test_ranges_for_phrase(self, words):
        tt = te.TimestampedTranscript(words=words)
        hits = tt.ranges_for("you know")
        assert hits and abs(hits[0][0] - 3.2) < 0.01
        assert abs(hits[0][1] - 4.2) < 0.01

    def test_from_dicts(self, words):
        ds = [w.to_dict() for w in words]
        tt = te.TimestampedTranscript.from_words(ds)
        assert len(tt.words) == len(words)


# ── remove_fillers (mocked ffmpeg layer) ─────────────────────────────

@pytest.fixture()
def mock_cuts(monkeypatch):
    calls = {"trim": [], "concat": 0}
    monkeypatch.setattr(te, "video_probe", lambda p: {"duration": 10.0})

    def fake_trim(src, s, e, **kw):
        calls["trim"].append((round(s, 3), round(e, 3)))
        return {"output": f"/tmp/trim-{s:.2f}-{e:.2f}.mp4"}

    def fake_concat(parts, **kw):
        calls["concat"] += 1
        return {"output": "/tmp/joined.mp4"}

    monkeypatch.setattr(te, "trim", fake_trim)
    monkeypatch.setattr(te, "concat", fake_concat)
    return calls


class TestRemoveFillers:
    def test_cuts_fillers_with_padding(self, words, mock_cuts):
        res = te.remove_fillers("/tmp/v.mp4", words, pad=0.2)
        # um (0.4-0.6) padded → cut (0.2, 0.8); uh (2.4-2.8) + "you know"
        # (3.2-4.2) padded → (2.2, 3.0) and (3.0, 4.4), which merge → 2 cuts
        assert res["cut"] == 2
        trims = mock_cuts["trim"]
        # padded cuts: um (0.2, 0.8), merged uh+you-know (2.2, 4.4)
        assert trims == [(0.0, 0.2), (0.8, 2.2), (4.4, 10.0)]
        assert res["output"] == "/tmp/joined.mp4"

    def test_no_fillers_returns_original(self, mock_cuts):
        ws = [W(0.0, 1.0, "clean"), W(1.0, 2.0, "speech")]
        res = te.remove_fillers("/tmp/v.mp4", ws)
        assert res["output"] == "/tmp/v.mp4"
        assert res["cut"] == 0
        assert "no filler" in res["note"]
        assert mock_cuts["trim"] == []  # nothing cut, nothing trimmed

    def test_single_keep_skips_concat(self, monkeypatch, words):
        monkeypatch.setattr(te, "video_probe", lambda p: {"duration": 5.0})
        monkeypatch.setattr(te, "trim",
                            lambda src, s, e, **kw: {"output": "/tmp/only.mp4"})
        called = []
        monkeypatch.setattr(te, "concat",
                            lambda parts, **kw: called.append(parts) or
                            {"output": "/tmp/joined.mp4"})
        # fillers everywhere except one clean span → single trim
        ws = [W(0.0, 0.5, "um"), W(4.5, 5.0, "clean")]
        res = te.remove_fillers("/tmp/v.mp4", ws, pad=0.0)
        assert res["output"] == "/tmp/only.mp4"
        assert called == []


# ── silence detection parsing ────────────────────────────────────────

class TestSilenceDetect:
    FAKE_STDERR = """[silencedetect @ 0x1] silence_start: 1.234
[silencedetect @ 0x1] silence_end: 2.500 | silence_duration: 1.266
[silencedetect @ 0x1] silence_start: 8.000
[silencedetect @ 0x1] silence_end: 8.750 | silence_duration: 0.750
"""

    def test_parse(self, monkeypatch):
        class P:
            stderr = self.FAKE_STDERR
            returncode = 0

        def fake_run(*a, **kw):
            assert "silencedetect=noise=-40.0dB:d=0.5" in kw.get("stdin", "") \
                or any("silencedetect" in str(x) for x in a[0])
            return P()

        monkeypatch.setattr(te.subprocess, "run", fake_run)
        spans = te.detect_silences("/tmp/v.mp4")
        assert spans == [(1.234, 2.5), (8.0, 8.75)]

    def test_remove_silences_assembles(self, monkeypatch, mock_cuts):
        monkeypatch.setattr(te, "detect_silences",
                            lambda *a, **kw: [(2.0, 3.0)])
        res = te.remove_silences("/tmp/v.mp4", pad=0.15)
        assert res["cut"] == 1
        # padded cut (1.85, 3.15); keeps (0, 1.85) and (3.15, 10)
        trims = mock_cuts["trim"]
        assert trims[0] == (0.0, 1.85)
        assert trims[1] == (3.15, 10.0)

    def test_no_silence_returns_original(self, monkeypatch, mock_cuts):
        monkeypatch.setattr(te, "detect_silences", lambda *a, **kw: [])
        res = te.remove_silences("/tmp/v.mp4")
        assert res["output"] == "/tmp/v.mp4"
        assert "no silences" in res["note"]


# ── edit_by_transcript validation ────────────────────────────────────

class TestEditByTranscript:
    def test_needs_llm_fn(self, words):
        with pytest.raises(MediaEditError, match="llm_fn"):
            te.edit_by_transcript("/tmp/v.mp4", words, "cut the intro")

    def test_needs_instruction(self, words):
        with pytest.raises(MediaEditError, match="instruction"):
            te.edit_by_transcript("/tmp/v.mp4", words, " ",
                                  llm_fn=lambda p: "{}")

    def test_garbage_ranges_rejected(self, words, monkeypatch):
        monkeypatch.setattr(te, "video_probe", lambda p: {"duration": 10.0})
        with pytest.raises(MediaEditError, match="out-of-bounds"):
            te.edit_by_transcript("/tmp/v.mp4", words, "keep it",
                                  llm_fn=lambda p: '{"keep": [[0, 99]]}')
        with pytest.raises(MediaEditError, match="malformed"):
            te.edit_by_transcript("/tmp/v.mp4", words, "keep it",
                                  llm_fn=lambda p: '{"keep": [[1]]}')
        with pytest.raises(MediaEditError, match="JSON"):
            te.edit_by_transcript("/tmp/v.mp4", words, "keep it",
                                  llm_fn=lambda p: "not json at all")
        with pytest.raises(MediaEditError, match="no usable ranges"):
            te.edit_by_transcript("/tmp/v.mp4", words, "keep it",
                                  llm_fn=lambda p: '{"keep": []}')

    def test_valid_ranges_assemble(self, words, monkeypatch, mock_cuts):
        monkeypatch.setattr(te, "video_probe", lambda p: {"duration": 10.0})
        res = te.edit_by_transcript(
            "/tmp/v.mp4", words, "keep the middle",
            llm_fn=lambda p: '```json\n{"keep": [[2, 4], [6, 8]]}\n```')
        assert res["instruction"] == "keep the middle"
        assert mock_cuts["trim"] == [(2.0, 4.0), (6.0, 8.0)]
        assert res["output"] == "/tmp/joined.mp4"

    def test_empty_words_rejected(self, monkeypatch):
        with pytest.raises(MediaEditError, match="no transcript"):
            te.edit_by_transcript("/tmp/v.mp4", [], "cut it",
                                  llm_fn=lambda p: '{"keep": [[0, 1]]}')


# ── keep_topic ───────────────────────────────────────────────────────

class TestKeepTopic:
    def test_keeps_mentions_with_context(self, words, monkeypatch, mock_cuts):
        monkeypatch.setattr(te, "video_probe", lambda p: {"duration": 10.0})
        ws = [W(0.0, 1.0, "intro"), W(5.0, 6.0, "pricing"),
              W(6.0, 7.0, "details"), W(9.0, 10.0, "outro")]
        res = te.keep_topic("/tmp/v.mp4", ws, "pricing", context_s=1.0)
        assert res["mentions"] == 1
        # padded (4.0, 7.0)
        assert mock_cuts["trim"] == [(4.0, 7.0)]

    def test_unmentioned_topic_errors(self, words, monkeypatch):
        monkeypatch.setattr(te, "video_probe", lambda p: {"duration": 10.0})
        with pytest.raises(MediaEditError, match="never mentioned"):
            te.keep_topic("/tmp/v.mp4", words, "quantum")


# ── export_xml ───────────────────────────────────────────────────────

class TestExportXml:
    def test_valid_xml_with_markers(self, words, monkeypatch, tmp_path):
        monkeypatch.setattr(te, "video_probe", lambda p: {"duration": 10.0})
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"x")
        out = te.export_xml(words, src)
        tree = ET.parse(str(out))
        root = tree.getroot()
        assert root.tag == "fcpxml"
        markers = root.findall(".//marker")
        assert markers, "expected transcript markers"
        assert all(m.get("value") for m in markers)

    def test_empty_words_rejected(self, tmp_path):
        with pytest.raises(MediaEditError, match="needs transcript"):
            te.export_xml([], tmp_path / "clip.mp4")


# ── chat NL patterns ─────────────────────────────────────────────────

class TestNLPatterns:
    def test_matches(self):
        from nomorals.agents.coremind import understand
        cases = {
            "cut the umms": "fillers",
            "cut the ums": "fillers",
            "remove fillers": "fillers",
            "remove the filler words": "fillers",
            "cut the silences": "silences",
            "remove silence": "silences",
            "cut the silences and umms": "both",
            "keep only the part about pricing": "topic",
        }
        for text, edit in cases.items():
            intents = understand(text)
            hits = [i for i in intents
                    if i.kind == "vision_transcript_edit"]
            assert hits, f"no intent for {text!r}"
            assert hits[0].meta["edit"] == edit, text
            if edit == "topic":
                assert hits[0].meta["topic"] == "pricing"

    def test_misfires(self):
        from nomorals.agents.coremind import understand
        for text in ["cut the cake", "remove the background",
                     "add captions", "keep only the best parts",
                     "silence is golden", "umm what was that"]:
            intents = understand(text)
            assert not [i for i in intents
                        if i.kind == "vision_transcript_edit"], text

    def test_registered_in_understand(self):
        from nomorals.agents.coremind import understand
        intents = understand("cut the umms")
        assert intents[0].kind == "vision_transcript_edit"
        assert intents[0].action == "transcript_edit"
        assert intents[0].route == "media"
