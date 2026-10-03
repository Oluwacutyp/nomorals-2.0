"""BookForge + Music quality: dynamic chapters, clean titles, real audio + score.

Offline suite — no network, no model required (heuristic paths tested).
"""

import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCleanTitle(unittest.TestCase):
    def setUp(self):
        from nomorals.books.forge import clean_title
        self.clean_title = clean_title

    def test_strips_write_me_prefix(self):
        t = self.clean_title("Write me a book about trust in relationships", None)
        self.assertNotIn("write me", t.lower())
        self.assertIn("Trust", t)

    def test_strips_medium_words(self):
        t = self.clean_title("make me a novel about a space voyage", None)
        self.assertNotIn("novel about", t.lower())
        self.assertIn("Space", t)

    def test_drops_feel_better_framing(self):
        t = self.clean_title(
            "Write me a story or book to feel better my girlfriend has been "
            "acting strange after her operation", None)
        self.assertNotIn("write me", t.lower())
        self.assertNotIn("feel better", t.lower())
        # a real title, not the raw prompt
        self.assertLess(len(t), 90)

    def test_plain_topic_untouched(self):
        t = self.clean_title("trust and communication in relationships", None)
        self.assertEqual(t, "Trust and Communication in Relationships")

    def test_empty(self):
        self.assertEqual(self.clean_title("", None), "Untitled")


class TestInferChapters(unittest.TestCase):
    def setUp(self):
        from nomorals.books.forge import infer_chapter_count
        self.infer = infer_chapter_count

    def test_novel_gets_more_than_guide(self):
        novel = self.infer("write me an epic novel about a space voyage", None)
        guide = self.infer("a practical guide to sourdough baking", None)
        short = self.infer("a short essay on focus", None)
        self.assertGreater(novel, guide)
        self.assertGreater(guide, short)

    def test_never_hardcoded_eight(self):
        # different topics must not all collapse to 8
        counts = {
            self.infer(t, None)
            for t in ("epic novel about dragons", "short essay on focus",
                      "guide to python programming with examples and projects",
                      "memoir of a chef in paris france italy spain travel food")
        }
        self.assertGreater(len(counts), 1)

    def test_bounds(self):
        for topic in ("a", "x y z", "novel " * 50):
            n = self.infer(topic, None)
            self.assertGreaterEqual(n, 3)
            self.assertLessEqual(n, 24)


class TestTemplateQuality(unittest.TestCase):
    def setUp(self):
        from nomorals.books.model import Book, Chapter
        from nomorals.books.write import template_chapter
        self.Book = Book
        self.Chapter = Chapter
        self.template_chapter = template_chapter

    def _book(self, topic="trust in relationships", title="Trust in Relationships"):
        b = self.Book(topic=topic, slug="test-q", title=title)
        b.notes = ""
        return b

    def test_no_boilerplate(self):
        book = self._book()
        ch = self.Chapter(number=2, title="Foundations",
                          beats=["first principle", "second principle",
                                 "third principle"])
        text = self.template_chapter(book, ch, prev_tail="prev")
        for banned in ("Start here:", "In practice this is where most people",
                       "Two details make this click",
                       "treat it as a system with inputs, outputs"):
            self.assertNotIn(banned, text)

    def test_no_raw_prompt_in_prose(self):
        book = self._book(
            topic="Write me a book about trust in relationships please",
            title="Trust in Relationships")
        ch = self.Chapter(number=1, title="Intro", beats=["opening idea"])
        text = self.template_chapter(book, ch, prev_tail="")
        self.assertNotIn("Write me a book", text)

    def test_sections_vary(self):
        book = self._book()
        beats = [f"principle number {i}" for i in range(4)]
        ch = self.Chapter(number=3, title="Deep", beats=beats)
        text = self.template_chapter(book, ch, prev_tail="prev")
        # the same lead sentence must not repeat for every beat
        paras = [p for p in text.split("\n\n") if p.startswith("##") is False]
        first_sentences = [p.split(".")[0] for p in paras if p.strip()]
        self.assertGreater(len(set(first_sentences)), 1)

    def test_headings_capitalized(self):
        book = self._book()
        ch = self.Chapter(number=1, title="T", beats=["the first idea"])
        text = self.template_chapter(book, ch, prev_tail="")
        self.assertIn("## The first idea", text)


class TestSynth(unittest.TestCase):
    def test_renders_valid_wav(self):
        from nomorals.core.midi import NoteEvent
        from nomorals.media.synth import render_wav
        parts = {
            "melody": [NoteEvent(60, 0, 1), NoteEvent(67, 1, 2)],
            "chords": [NoteEvent(48, 0, 2), NoteEvent(55, 0, 2)],
            "bass": [NoteEvent(36, 0, 2)],
            "drums": [NoteEvent(36, 0, 0.25, 100, 9),
                      NoteEvent(42, 1, 0.25, 80, 9)],
            "counter": [],
        }
        data = render_wav(parts, 120.0, seed=1)
        self.assertEqual(data[:4], b"RIFF")
        self.assertGreater(len(data), 50000)

    def test_wav_readable(self):
        import io
        from nomorals.core.midi import NoteEvent
        from nomorals.media.synth import render_wav
        parts = {"melody": [NoteEvent(69, 0, 2)], "chords": [],
                 "bass": [], "drums": [], "counter": []}
        data = render_wav(parts, 100.0, seed=2)
        w = wave.open(io.BytesIO(data))
        self.assertEqual(w.getnchannels(), 1)
        self.assertEqual(w.getframerate(), 22050)
        self.assertGreater(w.getnframes(), 22050)  # >1s of audio
        w.close()

    def test_drums_render(self):
        from nomorals.core.midi import NoteEvent
        from nomorals.media.synth import render_wav
        parts = {"melody": [], "chords": [], "bass": [], "counter": [],
                 "drums": [NoteEvent(36, 0, 0.25, 110, 9),
                           NoteEvent(38, 1, 0.25, 100, 9),
                           NoteEvent(42, 2, 0.25, 90, 9),
                           NoteEvent(49, 3, 0.5, 90, 9)]}
        data = render_wav(parts, 120.0, seed=3)
        self.assertGreater(len(data), 20000)


class TestSongArtifacts(unittest.TestCase):
    def test_score_markdown_is_lead_sheet(self):
        from nomorals.media.music import Song, Section
        song = Song(title="Test Song", style="pop", topic="test", key="C",
                    mode="major", tempo=120,
                    sections=[Section("verse", 4, chords=("I", "V"),
                                      lyrics=["hello world"])],
                    mood="happy")
        md = song.to_score_markdown()
        self.assertIn("# Test Song", md)
        self.assertIn("Chords:", md)
        self.assertIn("hello world", md)
        self.assertIn("VERSE", md)

    def test_to_dict_has_new_paths(self):
        from nomorals.media.music import Song
        song = Song(title="T", style="pop", topic="t", key="C",
                    mode="major", tempo=120, audio_path="/a.wav",
                    score_pdf_path="/s.pdf")
        d = song.to_dict()
        self.assertEqual(d["audio_path"], "/a.wav")
        self.assertEqual(d["score_pdf_path"], "/s.pdf")


class TestForgeDynamic(unittest.TestCase):
    def test_create_defaults_organic_and_grows(self):
        from nomorals.books.forge import BookForge

        class FakeSettings:
            def __init__(self, d):
                self.workspace_dir = d

        class FakeCtx:
            router = None

            def __init__(self, d):
                self.settings = FakeSettings(d)

        with tempfile.TemporaryDirectory() as d:
            forge = BookForge(FakeCtx(d))
            book = forge.create("Write me a novel about a voyage to mars",
                                research=False)
            # title is cleaned, not the raw prompt
            self.assertNotIn("write me", book.title.lower())
            # organic: no count decided up front — just the opening arc
            self.assertTrue(book.organic)
            self.assertFalse(book.concluded)
            self.assertEqual(len(book.chapters), 3)
            self.assertEqual(book.chapters_written, 0)
            # slug comes from the clean title
            self.assertNotIn("write-me-a-novel-about-a-voyage-to-mars",
                             book.slug)
            # writing grows the book until the topic is covered —
            # no count was ever decided; the size emerged from the content
            result = forge.write_all(book.slug)
            self.assertTrue(result["complete"])
            self.assertTrue(result["concluded"])
            self.assertGreaterEqual(result["total_chapters"], 5)
            self.assertEqual(result["chapters_written"],
                             result["total_chapters"])

    def test_explicit_chapters_stays_count_based(self):
        from nomorals.books.forge import BookForge

        class FakeSettings:
            def __init__(self, d):
                self.workspace_dir = d

        class FakeCtx:
            router = None

            def __init__(self, d):
                self.settings = FakeSettings(d)

        with tempfile.TemporaryDirectory() as d:
            forge = BookForge(FakeCtx(d))
            book = forge.create("a guide to sourdough", chapters=5,
                                research=False)
            self.assertFalse(book.organic)
            self.assertEqual(len(book.chapters), 5)


if __name__ == "__main__":
    unittest.main()
