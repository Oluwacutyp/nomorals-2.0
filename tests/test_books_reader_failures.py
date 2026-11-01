"""Tail-86 pins: nomorals/books/reader.py census sites.

216 — resolve_slug's scan skips a corrupt story.json ("a corrupt
entry never resolves"): resolving by a scan-derived key (prefix /
title) for that entry raises the honest-looking "no followed
story" ReaderError; the exact-slug fast path never parses and
still resolves. Exclusion: the surface is a loud error, not a
clean-success report (its listing twin at 346 is the convert).
346 — CONVERT (tail-59 forge pattern): following() used to skip
a corrupt story.json silently, so an all-corrupt shelf rendered
"not following any stories" — absence claimed for presence.
Now the skip binds the exception, counts it on
reader.last_following_unreadable, warns, and records it via
books/failures.record_failure; consumers (CLI book following,
story_following tool, /novel list) render the count honestly.
392 — _cached_chapter corrupt meta JSON -> None, the genuine
cache-miss value; every caller refetches from the source
(chapter_text/read treat None as "fetch it"), so it self-heals.
524 — sync's total-refresh SourceError -> pass: an optional
refresh rung; the report carries the last-known total and the
truthful fetched list; sibling fetch errors log and break loudly.
706 — _read_json_list corrupt -> [], the genuine no-file value,
estate idiom for on-disk JSON collections (bookmarks/highlights);
add_bookmark then rewrites the file with the new mark (the
corrupt bytes were already unparseable).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nomorals.books import reader as reader_mod
from nomorals.books.reader import FollowedStory, ReaderError, StoryReader
from nomorals.books.sources import SourceError


def _reader(tmp_path) -> StoryReader:
    ctx = SimpleNamespace(
        db=None, settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    return StoryReader(ctx)


def _mk_story(r: StoryReader, slug: str, title: str, total: int = 7):
    d = r.reader_dir() / slug
    d.mkdir(parents=True, exist_ok=True)
    story = FollowedStory(slug=slug, title=title, url=f"https://fake/{slug}",
                          source="fake", author="A", status="Ongoing",
                          total_chapters=total)
    (d / "story.json").write_text(
        json.dumps(story.to_dict()), encoding="utf-8")
    return story


def _mk_corrupt(r: StoryReader, slug: str):
    d = r.reader_dir() / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "story.json").write_text("{not json", encoding="utf-8")


# ── 216: resolve scan skip ──────────────────────────────────────────

def test_216_resolve_scan_skips_corrupt_entry(tmp_path):
    r = _reader(tmp_path)
    _mk_story(r, "good-one", "Good One")
    _mk_corrupt(r, "corrupt-one")
    # exact slug: fast path, never parses -> still resolves
    assert r.resolve_slug("corrupt-one") == "corrupt-one"
    # a scan-derived key (unique prefix) skips the corrupt entry
    with pytest.raises(ReaderError) as ei:
        r.resolve_slug("corrupt")
    assert "no followed story" in str(ei.value)
    # control: the good story resolves by prefix through the same scan
    assert r.resolve_slug("good") == "good-one"


# ── 346: CONVERT — following() unreadable surfacing ─────────────────

def test_346_following_corrupt_entry_counted_warned_recorded(
        tmp_path, monkeypatch):
    r = _reader(tmp_path)
    _mk_story(r, "good-one", "Good One")
    _mk_corrupt(r, "corrupt-one")
    warnings, recorded = [], []
    fake_log = SimpleNamespace(
        warning=lambda fmt, *a: warnings.append(fmt % a if a else fmt),
        info=lambda *a, **k: None, debug=lambda *a, **k: None)
    monkeypatch.setattr(reader_mod, "_log", fake_log)
    monkeypatch.setattr(
        reader_mod, "record_failure",
        lambda site, exc: recorded.append((site, exc)))
    out = r.following()
    assert [e["slug"] for e in out] == ["good-one"]
    assert r.last_following_unreadable == 1
    assert len(warnings) == 1 and "corrupt-one" in warnings[0]
    assert len(recorded) == 1 and recorded[0][0] == "reader.following"
    assert isinstance(recorded[0][1], Exception)
    # NOT-recording control: a clean shelf counts and records nothing
    r2 = _reader(tmp_path / "clean")
    _mk_story(r2, "good-one", "Good One")
    monkeypatch.setattr(reader_mod, "_log", fake_log)
    out2 = r2.following()
    assert [e["slug"] for e in out2] == ["good-one"]
    assert r2.last_following_unreadable == 0
    assert len(recorded) == 1  # unchanged


# ── 392: cached chapter meta ────────────────────────────────────────

def test_392_cached_chapter_corrupt_meta_is_cache_miss(tmp_path):
    r = _reader(tmp_path)
    _mk_story(r, "good-one", "Good One")
    ch_dir = r.story_dir("good-one") / "chapters"
    (ch_dir / "00001.md").write_text("# T\n\nsrc\n\nbody text",
                                     encoding="utf-8")
    (ch_dir / "00001.json").write_text("{corrupt", encoding="utf-8")
    assert r._cached_chapter("good-one", 1) is None
    # control: a valid meta returns the chapter
    meta = {"number": 1, "title": "One", "url": "https://fake/ch/1",
            "prev_url": "", "next_url": "", "words": 2}
    (ch_dir / "00001.json").write_text(json.dumps(meta), encoding="utf-8")
    ch = r._cached_chapter("good-one", 1)
    assert ch is not None and ch.title == "One"


# ── 524: sync total refresh ─────────────────────────────────────────

def test_524_sync_total_refresh_failure_keeps_known_total(
        tmp_path, monkeypatch):
    r = _reader(tmp_path)
    _mk_story(r, "good-one", "Good One", total=7)

    class BadAdapter:
        name = "fake"

        def fetch_chapter(self, url):
            raise SourceError("fetch down")

        def novel(self, url):
            raise SourceError("meta down")

        def chapter_list(self, url, *, limit=0):
            raise SourceError("list down")

        def chapter_url(self, url, number):
            return None

    monkeypatch.setattr(reader_mod, "adapter_for_url",
                        lambda url: BadAdapter())
    out = r.sync("good-one", chapters=1)
    assert out["synced"][0]["fetched"] == []
    assert out["synced"][0]["new_total"] == 7


# ── 706: json list collections ──────────────────────────────────────

def test_706_read_json_list_corrupt_returns_empty(tmp_path):
    r = _reader(tmp_path)
    _mk_story(r, "good-one", "Good One")
    path = r._bookmarks_path("good-one")
    path.write_text("[{corrupt", encoding="utf-8")
    assert r._read_json_list(path) == []
    # control: a missing file is the same genuine empty value
    assert r._read_json_list(path.parent / "nope.json") == []
    # and a write over the corrupt file starts the list fresh
    mark = r.add_bookmark("good-one", 3, label="here")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 1 and data[0]["id"] == mark["bookmark"]["id"] == 1
