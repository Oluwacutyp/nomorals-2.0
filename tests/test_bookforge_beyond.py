"""BookForge beyond-spec: collab, branches, publishing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nomorals.books.collab import CriticAgent, CollaborativeSession, Critique
from nomorals.books.branches import StoryBranch, list_branches
from nomorals.books.publish import SerialPublication


class TestCritic:
    def test_heuristic_short(self):
        c = CriticAgent().review("Too short.", {"summary": "test"})
        assert isinstance(c, Critique)
        assert "short" in " ".join(c.issues).lower()
        assert c.verdict in ("ship", "revise", "rewrite")

    def test_heuristic_good_draft(self):
        text = (
            '"I never meant for it to end like this," Mara whispered, her hands '
            "trembling around the empty cup. " * 40
        )
        c = CriticAgent().review(text, {"summary": "emotional confrontation"})
        assert c.score >= 0.5

    def test_heuristic_telling(self):
        text = ("He felt sad. She felt sad. They felt sad. Everyone felt sad. " * 30)
        c = CriticAgent().review(text, {"summary": "sad scene"})
        assert any("telling" in i.lower() or "showing" in s.lower()
                   for i in c.issues for s in [i] + c.suggestions)

    def test_parse(self):
        raw = ("SCORE: 85\nSTRENGTHS: pacing, dialogue\n"
               "ISSUES: thin ending\nSUGGESTIONS: strengthen climax\nVERDICT: revise")
        c = CriticAgent()._parse_critique(raw)
        assert c.score == pytest.approx(0.85)
        assert c.verdict == "revise"
        assert "pacing" in c.strengths


class TestCollab:
    def test_scene_no_character(self):
        sess = CollaborativeSession("test-story")
        r = sess.write_scene("Nobody", "do something")
        assert r["ok"] is False

    def test_critique_path(self):
        sess = CollaborativeSession("test-story")
        r = sess.critique("A short draft.", {"summary": "test"})
        assert r["ok"] is True
        assert "verdict" in r["critique"]


class TestBranches:
    @pytest.fixture
    def story(self, tmp_path, monkeypatch):
        root = tmp_path / "books_data"
        (root / "mystory").mkdir(parents=True)
        monkeypatch.setattr("nomorals.books.branches._books_root", lambda: root)
        return root

    def test_fork_write_merge(self, story):
        b = StoryBranch("mystory", "dark-timeline")
        r = b.fork(5, "what if the hero dies")
        assert r["ok"] is True
        r = b.add_chapter("The hero fell. Darkness rose.", "The Fall")
        assert r["ok"] is True
        assert len(b.chapters()) == 1
        r = b.merge()
        assert r["ok"] is True
        assert r["chapters_merged"] == 1

    def test_abandon(self, story):
        b = StoryBranch("mystory", "doomed")
        b.fork(3, "bad idea")
        assert b.exists
        r = b.abandon()
        assert r["ok"] is True
        assert not b.exists

    def test_list(self, story):
        StoryBranch("mystory", "a").fork(1, "first")
        StoryBranch("mystory", "b").fork(2, "second")
        branches = list_branches("mystory")
        assert len(branches) == 2


class TestPublish:
    @pytest.fixture
    def story(self, tmp_path, monkeypatch):
        root = tmp_path / "books_data"
        sdir = root / "serial1"
        sdir.mkdir(parents=True)
        (sdir / "chapter_001.md").write_text("# One\n\nFirst chapter text here.")
        (sdir / "chapter_002.md").write_text("# Two\n\nSecond chapter text here.")
        monkeypatch.setattr("nomorals.books.publish._books_root", lambda: root)
        return root

    def test_start_release(self, story):
        pub = SerialPublication("serial1")
        r = pub.start("manual", title="My Serial")
        assert r["ok"] is True
        d = pub.due()
        assert d["due"] is True
        r = pub.release()
        assert r["ok"] is True
        assert r["chapter"] == 1
        assert "First chapter" in r["text"]
        r = pub.release()
        assert r["chapter"] == 2
        r = pub.release()
        assert r["ok"] is False  # nothing left

    def test_follow_feedback(self, story):
        pub = SerialPublication("serial1")
        pub.start("manual")
        pub.follow("reader1")
        pub.follow("reader2")
        r = pub.feedback("reader1", 1, "love", "Amazing chapter!")
        assert r["ok"] is True
        s = pub.feedback_summary()
        assert s["ok"] is True
        assert s["by_chapter"][1]["reactions"]["love"] == 1
        st = pub.status()
        assert st["followers"] == 2

    def test_pause_resume(self, story):
        pub = SerialPublication("serial1")
        pub.start("manual")
        pub.pause()
        assert pub.status()["status"] == "paused"
        pub.resume()
        assert pub.status()["status"] == "active"
