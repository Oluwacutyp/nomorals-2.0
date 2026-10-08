"""Build-map #103 — transcript-as-timeline audio editing. All offline."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nomorals.audio.edit import (
    Edit,
    EditableTranscript,
    FILLERS_BY_LANG,
    _complement,
    _merge_ranges,
    _pad,
    apply_edits,
    control_audio,
    enhance_audio,
    fillers_for,
    find_fillers,
    nl_audio_intent,
    remove_fillers,
    transcript_edit,
)
from nomorals.media_edit.captions import Word


def _w(text, s, e):
    return Word(start=s, end=e, text=text)


def _tmpfile(suffix=".wav"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


class TestFillerSets(unittest.TestCase):
    def test_english_base(self):
        f = fillers_for("en")
        self.assertIn("um", f)
        self.assertIn("you know", f)

    def test_yoruba_ekiti(self):
        self.assertIn("ẹẹm", fillers_for("yo"))
        self.assertIn("ẹẹm", fillers_for("yo-ekiti"))

    def test_aliases(self):
        self.assertEqual(fillers_for("ekiti"), fillers_for("yo-ekiti"))
        self.assertEqual(fillers_for("pidgin"), fillers_for("pcm"))
        self.assertIn("ehn", fillers_for("pcm"))
        self.assertIn("toh", fillers_for("ha"))
        self.assertIn("eh", fillers_for("ig"))

    def test_unknown_lang_falls_back_to_english(self):
        self.assertEqual(fillers_for("xx"), FILLERS_BY_LANG["en"])
        self.assertEqual(fillers_for(""), FILLERS_BY_LANG["en"])


class TestFindFillers(unittest.TestCase):
    def test_english_fillers_found(self):
        words = [_w("hello", 0.0, 0.5), _w("um", 0.5, 0.8),
                 _w("world", 0.8, 1.2), _w("uh", 1.2, 1.5)]
        hits = find_fillers(words, "en")
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(hits[0][0], 0.5)
        self.assertAlmostEqual(hits[1][1], 1.5)

    def test_whole_word_only(self):
        # "umbrella" must never match "um"
        words = [_w("umbrella", 0.0, 0.8), _w("likeable", 0.8, 1.4)]
        self.assertEqual(find_fillers(words, "en"), [])

    def test_multiword_filler(self):
        words = [_w("well", 0.0, 0.4), _w("you", 0.4, 0.6),
                 _w("know", 0.6, 0.9), _w("yeah", 0.9, 1.2)]
        hits = find_fillers(words, "en")
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0][0], 0.4)
        self.assertAlmostEqual(hits[0][1], 0.9)

    def test_yoruba_ekiti_unicode(self):
        words = [_w("mo", 0.0, 0.3), _w("ẹẹm", 0.3, 0.7), _w("lọ", 0.7, 1.0)]
        hits = find_fillers(words, "yo-ekiti")
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0][0], 0.3)

    def test_case_insensitive(self):
        words = [_w("UM", 0.0, 0.5)]
        self.assertEqual(len(find_fillers(words, "en")), 1)

    def test_empty(self):
        self.assertEqual(find_fillers([], "en"), [])
        self.assertEqual(find_fillers(None, "en"), [])

    def test_dict_words(self):
        words = [{"text": "uh", "start": 1.0, "end": 1.3}]
        self.assertEqual(len(find_fillers(words, "en")), 1)


class TestRangeMath(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(_merge_ranges([(0, 1), (0.9, 2), (5, 6)]),
                         [(0, 2), (5, 6)])

    def test_complement(self):
        keeps = _complement([(2, 3)], 0.0, 10.0)
        self.assertEqual(keeps, [(0.0, 2.0), (3.0, 10.0)])

    def test_complement_all_cut(self):
        self.assertEqual(_complement([(0, 10)], 0.0, 10.0), [])

    def test_pad(self):
        padded = _pad([(2, 3)], 0.5, 0.0, 10.0)
        self.assertEqual(padded, [(1.5, 3.5)])

    def test_pad_clamped(self):
        padded = _pad([(0.1, 0.5)], 1.0, 0.0, 10.0)
        self.assertEqual(padded[0][0], 0.0)


class TestTranscriptEdit(unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(transcript_edit("/nonexistent/xyz.wav"))

    def test_mock_transcriber(self):
        path = _tmpfile()
        try:
            words = [_w("hello", 0.0, 0.5), _w("um", 0.5, 0.8),
                     _w("world", 0.8, 1.2)]
            t = transcript_edit(path, enhance=False,
                                transcriber=lambda p, l: words)
            self.assertIsNotNone(t)
            self.assertEqual(t.text, "hello um world")
            self.assertEqual(len(t.filler_hits), 1)
            self.assertEqual(t.language, "en")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_transcriber_failure_returns_none(self):
        path = _tmpfile()
        try:
            def boom(p, l):
                raise RuntimeError("nope")
            self.assertIsNone(transcript_edit(path, enhance=False,
                                              transcriber=boom))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_roundtrip(self):
        t = EditableTranscript(audio_path="a.wav", language="yo-ekiti",
                               words=[_w("ẹẹm", 0.0, 0.5)],
                               filler_hits=[(0.0, 0.5)])
        t2 = EditableTranscript.from_dict(t.to_dict())
        self.assertEqual(t2.language, "yo-ekiti")
        self.assertEqual(t2.words[0].text, "ẹẹm")
        self.assertEqual(t2.filler_hits, [(0.0, 0.5)])


class TestRemoveFillers(unittest.TestCase):
    def test_missing_file(self):
        res = remove_fillers("/nonexistent/xyz.wav")
        self.assertFalse(res["ok"])

    def test_no_fillers_returns_original(self):
        path = _tmpfile()
        try:
            words = [_w("hello", 0.0, 0.5), _w("world", 0.5, 1.0)]
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=1.0):
                res = remove_fillers(path, words=words)
            self.assertTrue(res["ok"])
            self.assertEqual(res["output"], path)
            self.assertEqual(res["cut"], 0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_fillers_cut(self):
        path = _tmpfile()
        try:
            words = [_w("hello", 0.0, 0.5), _w("um", 0.5, 0.9),
                     _w("world", 0.9, 1.4)]
            captured = {}
            def fake_splice(audio, keeps, **kw):
                captured["keeps"] = keeps
                return {"ok": True, "output": "/tmp/out.wav"}
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=1.4), \
                 patch("nomorals.audio.edit._splice", fake_splice):
                res = remove_fillers(path, words=words, pad=0.0)
            self.assertTrue(res["ok"])
            self.assertEqual(res["cut"], 1)
            keeps = captured["keeps"]
            # the um range (0.5, 0.9) must be gone from keeps
            for s, e in keeps:
                self.assertFalse(s < 0.9 and e > 0.5,
                                 f"keep {s, e} overlaps filler")
        finally:
            Path(path).unlink(missing_ok=True)


class TestApplyEdits(unittest.TestCase):
    def _transcript(self, path, dur=10.0):
        return EditableTranscript(audio_path=path, enhanced_path=path,
                                  language="en",
                                  words=[_w("a", 0.0, dur)])

    def test_no_edits(self):
        path = _tmpfile()
        try:
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=10.0):
                res = apply_edits(self._transcript(path), [])
            self.assertFalse(res["ok"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unknown_kind(self):
        path = _tmpfile()
        try:
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=10.0):
                res = apply_edits(self._transcript(path),
                                  [Edit("zap", 1.0, 2.0)])
            self.assertFalse(res["ok"])
            self.assertIn("unknown edit kind", res["reason"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_out_of_range(self):
        path = _tmpfile()
        try:
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=10.0):
                res = apply_edits(self._transcript(path),
                                  [Edit.delete(9.0, 99.0)])
            self.assertFalse(res["ok"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_overlapping_refused(self):
        path = _tmpfile()
        try:
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=10.0):
                res = apply_edits(self._transcript(path),
                                  [Edit.delete(1.0, 3.0),
                                   Edit.delete(2.5, 4.0)])
            self.assertFalse(res["ok"])
            self.assertIn("overlapping", res["reason"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_deletion_only(self):
        path = _tmpfile()
        try:
            trims = []
            def fake_trim(src, s, e, **kw):
                trims.append((s, e))
                return {"output": f"/tmp/keep{len(trims)}.wav"}
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=10.0), \
                 patch("nomorals.media_edit.videos.trim", fake_trim), \
                 patch("nomorals.media_edit.videos.concat",
                       return_value={"output": "/tmp/joined.wav"}):
                res = apply_edits(self._transcript(path),
                                  [Edit.delete(2.0, 3.0)], pad=0.0)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["deletes"], 1)
            self.assertEqual(res["rewrites"], 0)
            # keeps must skip [2.0, 3.0]
            self.assertEqual(len(trims), 2)
            self.assertAlmostEqual(trims[0][1], 2.0)
            self.assertAlmostEqual(trims[1][0], 3.0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_rewrite_without_synth_refused(self):
        path = _tmpfile()
        try:
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=10.0), \
                 patch("nomorals.audio.edit._normalize_wav",
                       side_effect=lambda p, sr, d: p), \
                 patch("nomorals.media_edit.videos.trim",
                       return_value={"output": "/tmp/k.wav"}):
                res = apply_edits(
                    self._transcript(path),
                    [Edit.rewrite(2.0, 3.0, "new words")], pad=0.0,
                    synthesizer=lambda *a, **k: None)
            self.assertFalse(res["ok"])
            self.assertIn("TTS", res["reason"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_rewrite_with_synth(self):
        path = _tmpfile()
        try:
            pieces = []
            def fake_trim(src, s, e, **kw):
                out = f"/tmp/keep{len(pieces)}.wav"
                pieces.append(("keep", out))
                return {"output": out}
            def fake_concat(srcs, **kw):
                pieces.append(("concat", list(srcs)))
                return {"output": "/tmp/final.wav"}
            calls = {}
            def fake_synth(text, ref, **kw):
                calls["text"] = text
                return "/tmp/synth.wav"
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=10.0), \
                 patch("nomorals.audio.edit._normalize_wav",
                       side_effect=lambda p, sr, d: p), \
                 patch("nomorals.media_edit.videos.trim", fake_trim), \
                 patch("nomorals.media_edit.videos.concat", fake_concat):
                res = apply_edits(
                    self._transcript(path),
                    [Edit.rewrite(2.0, 3.0, "new words")], pad=0.0,
                    synthesizer=fake_synth)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["rewrites"], 1)
            self.assertEqual(calls["text"], "new words")
            # concat must see: keep, synth, keep — in order
            concat_call = [p for p in pieces if p[0] == "concat"][0][1]
            self.assertEqual(concat_call[1], "/tmp/synth.wav")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_dict_edits_accepted(self):
        path = _tmpfile()
        try:
            with patch("nomorals.audio.edit._audio_duration",
                       return_value=10.0), \
                 patch("nomorals.media_edit.videos.trim",
                       return_value={"output": "/tmp/k.wav"}), \
                 patch("nomorals.media_edit.videos.concat",
                       return_value={"output": "/tmp/joined.wav"}):
                res = apply_edits(
                    self._transcript(path).to_dict(),
                    [{"kind": "delete", "start": 8.0, "end": 9.0}], pad=0.0)
            self.assertTrue(res["ok"], res)
        finally:
            Path(path).unlink(missing_ok=True)


class TestEnhance(unittest.TestCase):
    def test_missing_file(self):
        res = enhance_audio("/nonexistent/xyz.wav")
        self.assertFalse(res["ok"])

    def test_no_ffmpeg_honest(self):
        path = _tmpfile()
        try:
            with patch("nomorals.audio.edit._ffmpeg", return_value=None):
                res = enhance_audio(path)
            self.assertTrue(res["ok"])
            self.assertEqual(res["output"], path)
            self.assertEqual(res["filter"], "none")
        finally:
            Path(path).unlink(missing_ok=True)


class TestNLIntent(unittest.TestCase):
    def test_filler_sentence(self):
        r = nl_audio_intent(
            "remove all the filler words from this voice note")
        self.assertIsNotNone(r)
        self.assertEqual(r["command"], "fillers")

    def test_cut_the_umms(self):
        r = nl_audio_intent("cut the umms from meeting.m4a")
        self.assertIsNotNone(r)
        self.assertEqual(r["file"], "meeting.m4a")

    def test_language_detected(self):
        r = nl_audio_intent("remove filler words in yoruba from note.wav")
        self.assertEqual(r["lang"], "yo")

    def test_unrelated(self):
        self.assertIsNone(nl_audio_intent("what's the weather like?"))
        self.assertIsNone(nl_audio_intent(""))
        self.assertIsNone(nl_audio_intent(None))


class TestChat(unittest.TestCase):
    def test_usage(self):
        out = control_audio("")
        self.assertIn("/audio", out)

    def test_unknown_verb_usage(self):
        out = control_audio("frobnicate x.wav")
        self.assertIn("/audio", out)

    def test_edit_missing_file(self):
        out = control_audio("edit /nonexistent/xyz.wav")
        self.assertIn("couldn't transcribe", out)

    def test_fillers_missing_file(self):
        out = control_audio("fillers /nonexistent/xyz.wav")
        self.assertIn("no such audio file", out)

    def test_enhance_missing_file(self):
        out = control_audio("enhance /nonexistent/xyz.wav")
        self.assertIn("no such audio file", out)

    def test_never_raises(self):
        self.assertIsInstance(control_audio(None), str)
        self.assertIsInstance(control_audio("edit"), str)


if __name__ == "__main__":
    unittest.main()
